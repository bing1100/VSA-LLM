"""E14 RD-12 — the interleaved passive ↔ active reading cycle (pre-registration §12; proposal §9.6; the author's
"passive learning — active reading — passive learning — active reading …"). Pilot: T4, SmolLM2-360M C5 s1, one seed.

One job per arm runs every round in order:
  frames of the 160 read terms (by the arm) → ontology → passive phase (continued training, a subprocess of
  `training.lm` with `train.init_mode: continue`) → evaluation → [cycle: utility test and retraction, reflection,
  the next active reading round] → next round.

Arms: `cycle` (RD-11's active step each round, with retraction by held-out utility), `single` (round-0 reading, no
discovery), `gold` (gold frames), `passive` (no frames). Measures per round: loss after the read terms in T4's evaluation
windows (per-term arrays saved for the bootstrap), E11 items (implicit and explicit recall), general loss, gold recall,
retractions.

    python -m vsa_embed.experiments.e14_rd12 run --arm cycle --run RUN --items DIR --output OUT --data DATA [--rounds 3]
        [--tokens 2000000] [--limit 160] [--device cuda] [--smoke]
    python -m vsa_embed.experiments.e14_rd12 report --runs OUT ... --output DIR
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from . import e11_read_to_learn as e11
from . import e14_cycle as cyc
from .e5_common import write_json

ALIAS = Path("~/data/vsa-llm/e9/alias-tables/t4.json").expanduser()
DOCS = Path("~/data/vsa-llm/tracks/t4-chemistry/v1/docs/train.jsonl.gz").expanduser()
ARMS = ("cycle", "single", "gold", "passive", "still-none", "still-gold")    # still-*: RD-12b's no-training controls
GENERAL = Path("~/data/vsa-llm/fineweb-edu/sample/10BT/000_00000.parquet").expanduser()   # T4's general training source
GENERAL_SKIP = 2000                                       # T4 train-general read this shard from document 2,000 on
RECIPES = {"pilot": {"def_repeats": 40, "general_docs": 0},
           "gentle": {"def_repeats": 10, "general_docs": 8000, "lr_scale": 0.1, "constant_lr": True}}  # RD-12b (§18)
STRUCTURAL = {"contains_element", "charge", "branch"}


def select_defs(read_set: e11.ReadSet, *, limit: int, seed: int = 0) -> list[dict[str, Any]]:
    """RD-11's selection (same order, same subset)."""
    concepts = {c["concept"]: c for c in read_set.concepts}
    defs = [{"id": d["concept"], "headword": d.get("headword") or concepts[d["concept"]]["surface"], "text": d["text"]}
            for d in read_set.definitions if d["style"] == "chebi" and d["concept"] in concepts]
    random.Random(seed).shuffle(defs)
    return defs[:limit]


def subset(read_set: e11.ReadSet, ids: set[str]) -> e11.ReadSet:
    return e11.ReadSet(read_set.path, read_set.manifest, [c for c in read_set.concepts if c["concept"] in ids],
                       [d for d in read_set.definitions if d["concept"] in ids], [i for i in read_set.items if i["concept"] in ids])


# -- active reading (RD-11's step) ---------------------------------------------------------------------------------------------

def active_round(defs: Sequence[dict[str, Any]], relations: dict[str, str], teacher: cyc.Teacher, ctx: e11.TrackContext, *,
                 rnd: int, seed: int = 0, batch: int = 10) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Read twice (relation order shuffled per pass), keep facts both passes agree on; returns (consolidated facts by
    concept, consolidated NEW proposals). Prompts are RD-11's, so round 0 reuses its cache."""
    passes = []
    for p in range(2):
        order = sorted(relations)
        random.Random(f"{seed}:{rnd}:{p}").shuffle(order)
        facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for start in range(0, len(defs), batch):
            chunk = defs[start:start + batch]
            ans = teacher.ask(cyc.read_prompt(chunk, relations, order, second=p == 1), cyc.read_schema(list(relations)))
            for item in (ans or {}).get("definitions", []):
                facts[item.get("id")].extend(item.get("facts", []))
        passes.append(facts)
    consolidated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposals = []
    for d in defs:
        keyed = []
        for facts in passes:
            ks = {}
            for f in facts.get(d["id"], []):
                atom = cyc.resolve(ctx, f.get("partner", ""))
                rel = f.get("relation")
                ks[(rel if rel != "NEW" else "NEW", atom if atom is not None else cyc.norm(f.get("partner", "")))] = {**f, "atom": atom}
            keyed.append(ks)
        for key, f in keyed[0].items():
            if key in keyed[1]:
                consolidated[d["id"]].append(f)
                if key[0] == "NEW":
                    proposals.append({"concept": d["id"], "headword": d["headword"], "partner": f.get("partner", ""),
                                      "new_name": f.get("new_name", ""), "new_description": f.get("new_description", ""), "atom": f["atom"]})
    return consolidated, proposals


def reflect(proposals: Sequence[dict[str, Any]], relations: dict[str, str], teacher: cyc.Teacher, *, support: int = 8,
            max_proposals: int = 250) -> list[dict[str, Any]]:
    import re
    sample = list(proposals)[:max_proposals]
    if not sample:
        return []
    ans = teacher.ask(cyc.reflect_prompt(sample, relations), cyc.reflect_schema())
    out = []
    for rt in (ans or {}).get("relations", []):
        members = [m for m in rt.get("members", []) if isinstance(m, int) and 0 <= m < len(sample)]
        name = re.sub(r"[^a-z0-9_]", "", rt.get("name", "").lower().replace(" ", "_"))
        if len(members) >= support and name and name not in relations and name != "new":
            out.append({"name": name, "description": rt.get("description", name), "support": len(members)})
    return out


# -- ontology, corpus, passive phase --------------------------------------------------------------------------------------------

def frames_from_facts(facts: dict[str, list[dict[str, Any]]], names: list[str]) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
    """concept → [(relation id, atom)]; relations outside `names` are appended (new relation ids)."""
    names = list(names)
    out: dict[str, list[tuple[int, int]]] = {}
    for c, fs in facts.items():
        frame = []
        for f in fs:
            rel, atom = f.get("relation"), f.get("atom")
            if rel in (None, "NEW") or atom is None:
                continue
            if rel not in names:
                names.append(rel)
            frame.append((names.index(rel), int(atom)))
        out[c] = list(dict.fromkeys(frame))
    return out, names


def write_ontology(base: dict[str, Any], read_entries: dict[str, int], frames: dict[str, list[tuple[int, int]]], names: list[str],
                   path: Path) -> dict[str, Any]:
    from .e13_cycle import frames_of, with_frames
    all_frames = frames_of(base)
    for c, e in read_entries.items():
        all_frames[e] = list(frames.get(c, []))
    onto = with_frames(base, all_frames, relation_names=list(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(onto, path)
    return onto


def build_passive_corpus(defs: Sequence[dict[str, Any]], out_dir: Path, *, tokenizer_name: str, def_repeats: int = 40,
                         domain_docs: int = 8000, general_docs: int = 0, seed: int = 0) -> dict[str, Any]:
    """The passive phases' fixed text: the read definitions, repeated (≈ 10% of tokens), mixed with T4 training-domain
    abstracts (which never name a held-out term); read terms linked by the full T4 alias table."""
    import gzip
    from transformers import AutoTokenizer
    from ..data.corpus import build_corpus, tokenizer_fingerprint
    from ..evaluation import channel_probes as cp
    from .host_corpus import tokenizer_normalization
    if (out_dir / "manifest.json").exists():
        return json.loads((out_dir / "manifest.json").read_text())
    texts = [d["text"] for d in defs] * def_repeats
    with gzip.open(DOCS, "rt") as handle:
        for i, line in enumerate(handle):
            if i >= domain_docs:
                break
            texts.append(json.loads(line)["text"])
    if general_docs:                                      # RD-12b: rehearse general text the base model trained on
        import pandas as pd
        frame = pd.read_parquet(GENERAL, columns=["text"])
        texts += list(frame["text"].iloc[GENERAL_SKIP:GENERAL_SKIP + general_docs])
    random.Random(seed).shuffle(texts)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    normalization = tokenizer_normalization(tokenizer)
    return build_corpus(iter(texts), out_dir, tokenizer_name=tokenizer_name, table=cp.load_alias_table(ALIAS), eos_id=tokenizer.eos_token_id,
                        vocab_size=len(tokenizer), max_tokens=10 ** 12, workers=4, reuse=True,
                        extra_manifest={"tokenizer_sha256": tokenizer_fingerprint(tokenizer), "e14": "rd12-passive"},
                        **({"normalization": normalization} if normalization else {}))


def passive_phase(base_config: dict[str, Any], *, init_from: Path, ontology: Path, corpus: Path, tokens: int, out_dir: Path,
                  device: str | None, smoke: bool, recipe: str = "pilot") -> Path:
    cfg = json.loads(json.dumps(base_config))
    cfg["train"].update({"init_mode": "continue", "init_from": str(init_from), "total_tokens": int(tokens),
                         "warmup_tokens": int(max(1, tokens // 10)), "checkpoint_minutes": 100000})
    settings = RECIPES[recipe]
    if "lr_scale" in settings:                            # RD-12b: lower learning rates, no re-warm, no decay
        cfg["train"]["lr"] = float(cfg["train"]["lr"]) * settings["lr_scale"]
        if cfg["train"].get("host_lr") is not None:
            cfg["train"]["host_lr"] = float(cfg["train"]["host_lr"]) * settings["lr_scale"]
    if settings.get("constant_lr"):
        cfg["train"].update({"warmup_tokens": 1, "min_lr_ratio": 1.0})
    cfg["data"].update({"train": str(corpus), "ontology": str(ontology)})
    cfg["channel"]["skip_empty_frames"] = True      # held-out terms not (yet) read have no frame: no row, as in evaluation
    cfg["eval"].update({"windows": 8 if smoke else 64, "first_tokens": int(tokens)})
    if device:
        cfg["device"] = device
    if smoke:
        cfg["train"].update({"micro_batch": 1, "grad_accum": 1})
        cfg["model"]["seq_len"] = 256
    if out_dir.exists():                    # a failed or earlier attempt's folder: the trainer needs an empty one
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir.parent / f"{out_dir.name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run([sys.executable, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(out_dir)],
                          capture_output=True, text=True)
    (out_dir.parent / f"{out_dir.name}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    if proc.returncode != 0 or not (out_dir / "final.pt").exists():
        raise RuntimeError(f"passive phase failed ({proc.returncode}); see {out_dir.parent / (out_dir.name + '.log')}")
    return out_dir / "final.pt"


# -- evaluation -------------------------------------------------------------------------------------------------------------------

def term_losses(run: Any, wset: Any, corpus: Path, frames: dict[int, list[tuple[int, int]]] | None = None) -> np.ndarray:
    """Per read term: mean token loss after the term in the evaluation windows (NaN when it has no occurrence)."""
    import contextlib
    with (e11.swapped_frames(run.channel, frames) if frames is not None else contextlib.nullcontext()):
        out = e11.window_term_losses(run, corpus, wset)
    s, n = np.asarray(out["sum_other"], dtype=float), np.asarray(out["count_other"], dtype=float)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def evaluate(model_dir: Path, read_set: e11.ReadSet, frames: dict[str, list[tuple[int, int]]], *, device: str | None,
             smoke: bool, swap: bool = False) -> tuple[dict[str, Any], Any, Any, dict[str, int]]:
    """Per round. `swap` (RD-12b's still arms, an unchanged model): the read terms' frames are swapped in for the loss.
    Items: implicit, explicit (whole frame; tier P's null), `none`; and (RD-12b) `explicit_rel` — selective recall of the
    arm's frames — and `none_plain`, both with the context-free null; their per-item values are returned in
    `doc["item_values"]`."""
    from . import e14_reading as rd
    from .e5_common import open_run
    run = open_run(model_dir, device=device, batch_size=8 if smoke else 16, alias_table=ALIAS)
    if run.channel is not None:
        run.channel.skip_empty_frames = True                     # a read term with no frame gets no row (E11's "none")
    ctx = e11.run_context(run)
    entry_of = e11.link_entry_of(read_set, run)
    entries = [entry_of[c["concept"]] for c in read_set.concepts]
    headwords = {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts}
    wset = e11.build_window_set(run, entries, headwords, length=256 if smoke else 1024)
    corpus = Path(run.config["data"]["eval"])
    losses = term_losses(run, wset, corpus, {entry_of[c]: list(frames.get(c) or []) for c in frames} if swap else None)
    scorer = e11.ItemScorer(run, read_set, ctx)
    text = rd.RecallText(run.ontology, ctx.lexicon)
    concepts = scorer.concepts
    implicit = scorer.score({c: frames.get(c) or None for c in concepts})
    explicit = scorer.score({c: None for c in concepts}, {c: text.symbolic(scorer.surface[c], frames.get(c) or []) for c in concepts})
    none = scorer.score({c: None for c in concepts})
    null = rd.plain_null_scores(scorer)
    rel_frames, rel_contexts, _ = rd.item_condition_inputs("explicit-rel:arm", scorer, {"arm": {c: list(frames.get(c) or []) for c in concepts}},
                                                           text, {}, rd.filler_pools(run.ontology), ctx, seed=0)
    explicit_rel = rd.score_rows(scorer, rel_frames, rel_contexts, null_scores=null)
    none_plain = rd.score_rows(scorer, {c: None for c in concepts}, None, null_scores=null)
    vals = {name: e11.test_values(rows, read_set.kind) for name, rows in (("implicit", implicit), ("explicit", explicit), ("none", none),
                                                                          ("explicit_rel", explicit_rel), ("none_plain", none_plain))}
    items = {name: {t: float(np.mean(list(v.values()))) for t, v in by.items() if v} for name, by in vals.items()}
    gen = e11.general_loss(run, e11.general_windows(run, count=8 if smoke else 64, length=256 if smoke else 512))
    doc = {"loss_after_read_terms": float(np.nanmean(losses)), "terms_with_windows": int(np.isfinite(losses).sum()),
           "items": items, "general_loss": float(gen),
           "item_values": {k: {t: dict(v) for t, v in vals[k].items()} for k in ("explicit_rel", "none_plain")}}
    return doc, run, (wset, corpus, losses), entry_of


def utility_test(run: Any, wset_info: tuple, frames: dict[str, list[tuple[int, int]]], entry_of: dict[str, int], names: list[str],
                 discovered: Sequence[str], *, threshold: float = 0.005) -> dict[str, Any]:
    """Per discovered relation: loss after the read terms with its edges removed against with them; retract it when
    removing it lowers the loss by more than `threshold` (relative)."""
    wset, corpus, base = wset_info
    full = float(np.nanmean(base))
    out = {}
    for rel in discovered:
        if rel not in names:
            continue
        rid = names.index(rel)
        ablated = {entry_of[c]: [(r, a) for r, a in fr if r != rid] for c, fr in frames.items() if any(r == rid for r, _ in fr)}
        if not ablated:
            continue
        loss = float(np.nanmean(term_losses(run, wset, corpus, ablated)))
        out[rel] = {"loss_without": loss, "loss_with": full, "relative": (loss - full) / full, "retract": (full - loss) / full > threshold}
    return out


def gold_recall(frames: dict[str, list[tuple[int, int]]], gold: dict[str, set[tuple[int, int]]], names: list[str], base_names: list[str]) -> float:
    G = {(c, base_names[r], a) for c, es in gold.items() for r, a in es if base_names[r] not in STRUCTURAL}
    P = {(c, names[r], a) for c, fr in frames.items() for r, a in fr}
    return len(P & G) / max(1, len(G))


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}")
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    read_full = e11.load_read_set(args.items)
    defs = select_defs(read_full, limit=args.limit)
    ids = {d["id"] for d in defs}
    read_set = subset(read_full, ids)
    ctx = e11.track_context("t4")
    base_config = yaml.safe_load((Path(args.run) / "resolved_config.yaml").read_text())
    base_onto = torch.load(base_config["data"]["ontology"], weights_only=False)
    base_names = list(base_onto["relation_names"])
    concepts = {c["concept"]: c for c in read_set.concepts}
    read_entries = {c: int(concepts[c]["entry"]) for c in ids}
    gold = {c: set(ctx.ids(concepts[c]["frame"]) or []) for c in ids}
    recipe = getattr(args, "recipe", "pilot")
    corpus_dir = data.parent / ("passive-corpus" if recipe == "pilot" else f"passive-corpus-{recipe}") / "train"
    still = args.arm.startswith("still-")
    if not still and not args.corpus:
        build_passive_corpus(defs, corpus_dir, tokenizer_name=base_config["model"]["pretrained"],
                             def_repeats=RECIPES[recipe]["def_repeats"], general_docs=RECIPES[recipe]["general_docs"])
    teacher = cyc.Teacher(Path(args.cache), args.model, args.max_usd)
    hidden = [h for h in cyc.T4_HIDDEN if h in base_names]
    textual = [r for r in base_names if r not in STRUCTURAL]
    relations = {r: cyc.describe(ctx, r) for r in textual if r not in hidden}
    rounds, state, retracted_all = [], Path(args.run) / "final.pt", []
    single_facts = None
    for rnd in range(1 if still else args.rounds):
        t0 = time.monotonic()
        rec: dict[str, Any] = {"round": rnd, "arm": args.arm}
        if args.arm in {"cycle", "single"}:
            if args.arm == "single" and single_facts is not None:
                facts = single_facts
            else:
                facts, proposals = active_round(defs, relations if args.arm == "cycle" else
                                                {r: cyc.describe(ctx, r) for r in textual if r not in hidden}, teacher, ctx, rnd=rnd if args.arm == "cycle" else 0)
                if args.arm == "single":
                    single_facts = facts
            frames, names = frames_from_facts(facts, base_names)
            rec["relations"] = sorted(relations) if args.arm == "cycle" else sorted(r for r in textual if r not in hidden)
        elif args.arm in {"gold", "still-gold"}:
            frames, names = {c: sorted(gold[c]) for c in ids}, list(base_names)
        else:
            frames, names = {c: [] for c in ids}, list(base_names)
        rec["frames_per_term"] = float(np.mean([len(frames.get(c, [])) for c in ids]))
        rec["gold_recall"] = gold_recall(frames, gold, names, base_names)
        if still:                                         # RD-12b: the unchanged model, the arm's frames swapped in
            model_dir, new_state = Path(args.run), state
        else:
            onto_path = data / f"round{rnd}" / "ontology.pt"
            write_ontology(base_onto, read_entries, frames, names, onto_path)
            model_dir = data / f"round{rnd}" / "model"
            new_state = passive_phase(base_config, init_from=state, ontology=onto_path, corpus=Path(args.corpus) if args.corpus else
                                      corpus_dir, tokens=args.tokens, out_dir=model_dir, device=args.device, smoke=args.smoke,
                                      recipe=recipe)
        doc, run, wset_info, entry_of = evaluate(model_dir, read_set, frames, device=args.device, smoke=args.smoke, swap=still)
        write_json(out / f"items_round{rnd}.json", doc.pop("item_values"))
        rec.update(doc)
        np.save(out / f"term_losses_round{rnd}.npy", wset_info[2])
        if args.arm == "cycle":
            discovered = [n for n in names if n not in base_names or n in hidden]
            test = utility_test(run, wset_info, frames, entry_of, names, discovered)
            rec["utility"] = test
            retract = [n for n, v in test.items() if v["retract"]]
            retracted_all += retract
            for n in retract:
                relations.pop(n, None)
            if rnd + 1 < args.rounds:
                accepted = [a for a in reflect(proposals, relations, teacher) if a["name"] not in retracted_all]
                for a in accepted:
                    relations[a["name"]] = a["description"]
                rec["accepted_next"] = accepted
            rec["retracted"] = retract
        del run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if state != Path(args.run) / "final.pt" and state.exists() and not args.keep:
            state.unlink()                                          # keep only the latest round's weights
        state = new_state
        rec["teacher_spent_usd"] = round(teacher.spent, 3)
        rec["seconds"] = round(time.monotonic() - t0, 1)
        rounds.append(rec)
        write_json(out / "rounds.json", {"arm": args.arm, "run": str(args.run), "tokens_per_round": 0 if still else args.tokens,
                                          "terms": len(ids), "recipe": recipe, "smoke": bool(args.smoke), "rounds": rounds})
        print(json.dumps({k: v for k, v in rec.items() if k not in {"utility"}}, default=str), flush=True)
    return {"rounds": rounds}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    arms = {}
    for p in args.runs:
        f = Path(p) / "rounds.json"
        if f.exists():
            d = json.loads(f.read_text())
            arms[d["arm"]] = (Path(p), d)
    recipe = next((d.get("recipe", "pilot") for _, d in arms.values()), "pilot")
    title = ("interleaved passive ↔ active reading cycle (pilot: T4, SmolLM2-360M C5 s1)" if recipe == "pilot" else
             f"RD-12b, the cycle rerun with the `{recipe}` passive phase and no-training controls (pre-registration §18)")
    lines = [f"# E14 RD-12 — {title}", "",
             "| arm | round | frames/term | gold recall | loss after read terms | general loss | explicit property | implicit property | "
             "selective property | none (plain null) | retracted |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, (_, d) in sorted(arms.items()):
        for r in d["rounds"]:
            prop = lambda k: r["items"].get(k, {}).get("property")  # noqa: E731
            f = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
            lines.append(f"| {arm} | {r['round']} | {r['frames_per_term']:.2f} | {r['gold_recall']:.3f} | {r['loss_after_read_terms']:.4f} | "
                         f"{r['general_loss']:.4f} | {f(prop('explicit'))} | {f(prop('implicit'))} | {f(prop('explicit_rel'))} | "
                         f"{f(prop('none_plain'))} | {', '.join(r.get('retracted', [])) or '—'} |")
    doc: dict[str, Any] = {"recipe": recipe, "contrasts": [], "selective": []}
    last = {arm: max(r["round"] for r in d["rounds"]) for arm, (_, d) in arms.items()}
    for a, b in (("cycle", "passive"), ("cycle", "single"), ("gold", "passive"), ("single", "passive"),
                 ("passive", "still-none"), ("gold", "still-gold"), ("still-gold", "still-none")):
        if a in arms and b in arms:
            la = np.load(arms[a][0] / f"term_losses_round{last[a]}.npy"); lb = np.load(arms[b][0] / f"term_losses_round{last[b]}.npy")
            ok = np.isfinite(la) & np.isfinite(lb)
            c = paired_difference(la[ok], lb[ok], resamples=args.resamples, seed=0)
            doc["contrasts"].append({"a": a, "b": b, **c})
    lines += ["", "Loss after the read terms at the last round, a − b (paired over terms; negative = a better):", ""]
    lines += [f"- {c['a']} − {c['b']}: {c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] (p {c['p_value']:.4f}, n {c['n']})" for c in doc["contrasts"]]
    drift = next((c for c in doc["contrasts"] if (c["a"], c["b"]) == ("passive", "still-none")), None)
    if drift is not None:                                 # RD-12b's recipe gate (§18): the passive phase alone must not hurt
        base = float(np.nanmean(np.load(arms["still-none"][0] / "term_losses_round0.npy")))
        doc["recipe_gate"] = {"relative_drift": drift["mean"] / base, "passes": drift["mean"] / base <= 0.005}
        lines += ["", f"**Recipe gate** (passive − still-none ≤ +0.5% of still-none's loss): drift "
                      f"{100 * doc['recipe_gate']['relative_drift']:+.2f}% → {'passes' if doc['recipe_gate']['passes'] else 'fails'}."]
    for a, b in (("cycle", "single"), ("cycle", "passive"), ("gold", "passive"), ("still-gold", "still-none")):
        if a in arms and b in arms and (arms[a][0] / f"items_round{last[a]}.json").exists() and (arms[b][0] / f"items_round{last[b]}.json").exists():
            va = json.loads((arms[a][0] / f"items_round{last[a]}.json").read_text())["explicit_rel"]
            vb = json.loads((arms[b][0] / f"items_round{last[b]}.json").read_text())["explicit_rel"]
            for test in ("property", "paraphrase", "entailment"):
                ids = sorted(set(va.get(test, {})) & set(vb.get(test, {})))
                if ids:
                    c = paired_difference(np.array([va[test][i] for i in ids]), np.array([vb[test][i] for i in ids]),
                                          resamples=args.resamples, seed=0)
                    doc["selective"].append({"a": a, "b": b, "test": test, **c})
    if doc["selective"]:
        lines += ["", "Selective recall (each arm's frames, the asked relation's lines; context-free null) at the last round, a − b "
                      "(paired over items; positive = a better):", ""]
        lines += [f"- {c['a']} − {c['b']}, {c['test']}: {c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] (p {c['p_value']:.4f}, n {c['n']})"
                  for c in doc["selective"]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True); r.add_argument("--run", type=Path, required=True); r.add_argument("--items", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True); r.add_argument("--data", type=Path, required=True)
    r.add_argument("--rounds", type=int, default=3); r.add_argument("--tokens", type=int, default=2_000_000); r.add_argument("--limit", type=int, default=160)
    r.add_argument("--corpus", type=Path, default=None); r.add_argument("--device", default=None); r.add_argument("--smoke", action="store_true")
    r.add_argument("--keep", action="store_true"); r.add_argument("--max-usd", type=float, default=6.0); r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--cache", type=Path, default=Path("experiments/e14-logic/teacher-cache/rd11"))
    r.add_argument("--recipe", choices=tuple(RECIPES), default="pilot", help="passive-phase recipe (RD-12b: gentle)")
    p = sub.add_parser("report"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=4000)
    args = parser.parse_args(argv)
    (run_arm if args.command == "run" else run_report)(args)


if __name__ == "__main__":
    main()
