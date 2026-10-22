"""E15 — HRR as a training target: splits, read accuracy and the pooled report (pre-registration
`experiments/e15-hrr-target/preregistration.md`).

    python -m vsa_embed.experiments.e15_eval split --ontology ONT.pt --new-items DIR --output DIR [--seen 300] [--seed 0]
    python -m vsa_embed.experiments.e15_eval read --run RUN --splits DIR/splits.json --new-items DIR --output OUT
        [--alias-table T.json] [--template "Note: {x}"] [--limit N] [--smoke]
    python -m vsa_embed.experiments.e15_eval report --runs-root R --qa-root Q --read-root D --splits S --output OUT

`split` writes the fixed held-out splits (sha256 in `manifest.json`): **seen** (training entries linked ≥ 10 times, a
seeded sample), **compositional** (the T5 "new words" whose every role and filler atom occurs in a training entry's frame
and whose frame, as a set of edges, is new; **strict**: every (role, filler) edge occurs in some training frame),
**unseen_filler** (new words and held-out ontology entries with an atom no training frame holds) and **heldout_ontology**
(the ontology's held-out entries, every status).

`read` measures filler recovery from the trained map `W` (`hrr_target.HRRTarget`): the term alone after a neutral
prefix, its composed vector injected (new words inserted zero-shot with their gold frame; held-out entries read their
gold frame; seen entries read what the run injects), the hidden state at its last subtoken at layer ℓ, `W h` unbound by
each role of the frame and cleaned up by top-1 cosine over the whole atom table (chance 1/|A|). Also the typed top-1
(atoms observed under the role), the gold atom's reciprocal rank, and a **post-hoc ridge probe** fitted on the seen
split's states (does the structure sit in `h` without the losses? — the baselines' only read accuracy; on the seen
split itself the probe is in-sample).

`report` pools seeds (item differences paired within seed, pooled over seeds, percentile bootstrap, Holm over the
primary family), computes the loss endpoints from `eval_windows.npz` (paired windows against C5) and reads the
decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..statistics import holm_adjust, paired_ratio_bootstrap
from .e5_common import finish_output, json_ready, open_run, start_output, write_json

ARMS = ("C5WR", "C5R", "C2R", "C5shWR")
BASELINES = ("C5", "C2", "C5sh")
SEEDS = (1, 2, 3)
TESTS = ("property", "paraphrase", "entailment")
SPLITS = ("seen", "compositional", "compositional_strict", "unseen_filler", "heldout_ontology")
TEMPLATE = "Note: {x}"


# -- splits ---------------------------------------------------------------------------------------------------------

def frame_sets(ontology: dict[str, Any]) -> list[set[tuple[int, int]]]:
    offsets, relations, fillers = (np.asarray(ontology[k]) for k in ("offsets", "relations", "fillers"))
    return [set(zip(relations[a:b].tolist(), fillers[a:b].tolist())) for a, b in zip(offsets[:-1], offsets[1:])]


def split_check(frames: Sequence[set[tuple[int, int]]], train_frames: Sequence[set[tuple[int, int]]]) -> list[dict[str, Any]]:
    """Per frame: are all its atoms / all its edges in some training frame, and is the frame (as a set) new?"""
    train_edges = set().union(*train_frames) if train_frames else set()
    train_atoms = {a for _, a in train_edges}
    seen_frames = {frozenset(f) for f in train_frames}
    return [{"atoms_seen": all(a in train_atoms for _, a in f), "edges_seen": all(e in train_edges for e in f),
             "frame_new": frozenset(f) not in seen_frames, "edge_seen_share": (sum(e in train_edges for e in f) / len(f)) if f else None,
             "degree": len(f)} for f in frames]


def build_splits(ontology: dict[str, Any], new_concepts: Sequence[dict[str, Any]], *, seen_count: int = 300, seed: int = 0) -> dict[str, Any]:
    frames = frame_sets(ontology)
    heldout = set(int(e) for e in ontology["heldout_entries"])
    frequency = np.asarray(ontology["train_frequency"])
    train = [i for i in range(int(ontology["entry_count"])) if i not in heldout]
    train_frames = [frames[i] for i in train]
    concept_names, relation_names = list(ontology["concept_names"]), list(ontology["relation_names"])
    # An entry's display name: `concept_names` is indexed by concept, `entry_concepts` maps entries to concepts (E11's convention).
    entry_concepts = ontology.get("entry_concepts")
    names = [concept_names[c[0]] for c in entry_concepts] if entry_concepts else concept_names
    atom_id = {a: i for i, a in enumerate(ontology["atomic_names"])}
    relation_id = {r: i for i, r in enumerate(relation_names)}
    rng = random.Random(seed)
    eligible = [i for i in train if frequency[i] >= 10]
    seen = sorted(rng.sample(eligible, min(seen_count, len(eligible))))
    new_frames = [{(relation_id[r], atom_id[a]) for r, a in c["frame"]} for c in new_concepts]
    checks = split_check(new_frames, train_frames)
    held_checks = split_check([frames[e] for e in sorted(heldout)], train_frames)

    def new_record(c: dict[str, Any], f: set, check: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "new", "concept": c["concept"], "surface": c["surface"], "frame": sorted(f), **check}

    compositional = [new_record(c, f, k) for c, f, k in zip(new_concepts, new_frames, checks) if k["atoms_seen"] and k["frame_new"]]
    strict = [r for r in compositional if r["edges_seen"]]
    unseen = [new_record(c, f, k) for c, f, k in zip(new_concepts, new_frames, checks) if not k["atoms_seen"]]
    heldout_rows = [{"kind": "heldout", "entry": e, "surface": names[e], "frame": sorted(frames[e]), **k}
                    for e, k in zip(sorted(heldout), held_checks)]
    unseen += [r for r in heldout_rows if not r["atoms_seen"]]
    seen_rows = [{"kind": "seen", "entry": e, "surface": names[e], "frame": sorted(frames[e]), "train_frequency": int(frequency[e])} for e in seen]
    stats = {"train_entries": len(train), "train_edges": len(set().union(*train_frames)), "train_atoms": len({a for f in train_frames for _, a in f}),
             "atoms": int(ontology["atomic_count"]), "new_concepts": len(new_concepts),
             "new_all_atoms_seen": sum(k["atoms_seen"] for k in checks), "new_all_edges_seen": sum(k["edges_seen"] for k in checks),
             "new_frame_new": sum(k["frame_new"] for k in checks), "new_edge_seen_share": float(np.mean([k["edge_seen_share"] for k in checks])),
             "heldout_entries": len(heldout), "heldout_all_atoms_seen": sum(k["atoms_seen"] for k in held_checks),
             "heldout_all_edges_seen": sum(k["edges_seen"] for k in held_checks)}
    return {"schema": "e15-splits/1", "seed": seed, "seen_count": seen_count, "template": TEMPLATE, "stats": stats,
            "splits": {"seen": seen_rows, "compositional": compositional, "compositional_strict": strict, "unseen_filler": unseen,
                       "heldout_ontology": heldout_rows}}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_split(args: argparse.Namespace) -> dict[str, Any]:
    from . import e11_read_to_learn as e11
    ontology = torch.load(args.ontology, weights_only=False)
    read_set = e11.load_read_set(Path(args.new_items))
    splits = build_splits(ontology, read_set.concepts, seen_count=args.seen, seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "splits.json"
    path.write_text(json.dumps(json_ready(splits), indent=1, sort_keys=True) + "\n")
    manifest = {"schema": "e15-splits/1", "ontology": str(args.ontology), "ontology_alias_sha256": ontology.get("alias_table_sha256"),
                "new_items": str(args.new_items), "seed": args.seed, "seen_count": args.seen, "sha256": sha256_file(path),
                "sizes": {k: len(v) for k, v in splits["splits"].items()}, "stats": splits["stats"]}
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def load_splits(path: Path) -> dict[str, Any]:
    splits = json.loads(Path(path).read_text())
    if splits.get("schema") != "e15-splits/1":
        raise ValueError(f"{path} is not an e15-splits/1 file")
    return splits


# -- read accuracy ----------------------------------------------------------------------------------------------------

def _layer_of(run: Any) -> int:
    target = getattr(run.channel, "target", None) if run.channel is not None else None
    if target is not None:
        return int(target.layer)
    from ..readout import readout_layer
    return readout_layer("third", run.adapter.model.model)


@torch.no_grad()
def term_states(run: Any, adapter: Any, texts: Sequence[str], char_spans: Sequence[tuple[int, int]], expected: Sequence[int],
                layer: int) -> tuple[torch.Tensor, list[bool]]:
    """Hidden state at the input of decoder layer `layer` at the read position of each text's term (`read_index`), and
    whether the expected entry is injected there."""
    from ..evaluation.channel_probes import read_index
    from ..readout import decoder_layers
    model = adapter.model
    grabbed: list[torch.Tensor] = []
    hook = decoder_layers(model.model)[layer].register_forward_pre_hook(
        lambda m, a, kw: grabbed.append((a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)
    try:
        states, linked = [], []
        for start in range(0, len(texts), adapter.batch_size):
            batch = list(texts[start:start + adapter.batch_size])
            grabbed.clear()
            encoded, _, _ = adapter._run_batch(batch)
            hidden = grabbed[0].float().cpu()
            offsets_list = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in
                            zip(encoded["offset_mapping"], encoded["attention_mask"])]
            found = adapter.link_targets(batch, char_spans[start:start + len(batch)]) if adapter.linker is not None else [[] for _ in batch]
            for row, (text, offsets) in enumerate(zip(batch, offsets_list)):
                s, e = char_spans[start + row]
                states.append(hidden[row, read_index(offsets, s, e)])
                linked.append(int(expected[start + row]) in found[row])
    finally:
        hook.remove()
    return torch.stack(states) if states else torch.zeros(0, 1), linked


def ridge_fit(x: torch.Tensor, y: torch.Tensor, *, alpha: float = 1e-2) -> torch.Tensor:
    """`W` (out × in) minimising `‖X Wᵀ − Y‖² + α tr(XᵀX)/d ‖W‖²` (the post-hoc probe)."""
    x, y = x.double(), y.double()
    gram = x.T @ x
    lam = alpha * float(gram.diagonal().mean())
    return torch.linalg.solve(gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype), x.T @ y).T.float()


def summarize_edges(result: dict[str, torch.Tensor], concepts: int, *, resamples: int, seed: int) -> dict[str, Any]:
    """Accuracy over edges, mean per-concept accuracy with a bootstrap interval over concepts, MRR, typed accuracy."""
    if result["filler"].numel() == 0:
        return {"edges": 0, "concepts": concepts, "accuracy": None}
    hit = (result["top1"] == result["filler"]).float()
    occ = result["occurrence"]
    per = torch.zeros(concepts).index_add(0, occ, hit) / torch.zeros(concepts).index_add(0, occ, torch.ones_like(hit)).clamp_min(1)
    rng = np.random.default_rng(seed)
    draws = per.numpy()[rng.integers(0, concepts, size=(resamples, concepts))].mean(1)
    out = {"edges": int(hit.numel()), "concepts": concepts, "accuracy": float(hit.mean()), "concept_accuracy": float(per.mean()),
           "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
           "mrr": float((1.0 / result["rank"].float()).mean()), "per_concept": per.tolist()}
    if "typed_top1" in result and result["typed_top1"].numel():
        out["typed_accuracy"] = float((result["typed_top1"] == result["filler"]).float().mean())
    return out


def read_accuracy(run: Any, splits: dict[str, Any], new_items: Any, *, template: str = TEMPLATE, limit: int | None = None,
                  resamples: int = 2000, seed: int = 0, log=print) -> dict[str, Any]:
    from ..cleanup import relation_candidates
    from ..hrr_target import HRRTarget, ontology_decoder
    from . import e9_ontology_edit as edit
    from .e11_read_to_learn import link_entry_of, swapped_frames
    channel = run.channel
    target: HRRTarget | None = getattr(channel, "target", None) if channel is not None else None
    compose = channel is not None and channel.mode == "compose"
    layer = _layer_of(run)
    if target is not None:
        decoder = target.decoder
    elif compose:
        decoder = channel.composer
    else:                                   # a free-table baseline: read through a fixed random code (probe only)
        decoder = ontology_decoder(run.ontology, 256, seed=int(run.config.get("seed", 0))).to(run.device)
    reader = target if target is not None else HRRTarget(decoder, int(run.adapter.model.model.get_input_embeddings().weight.shape[1]),
                                                        layer=layer, read_weight=1.0).to(run.device)
    atoms = decoder.atomics.shape[0]
    candidates = relation_candidates(decoder.frame_relations, decoder.frame_fillers, decoder.relation_count, atoms)
    candidates[~candidates.any(-1)] = True
    entry_of = link_entry_of(new_items, run)
    schedule = decoder.schedule
    out: dict[str, Any] = {"layer": layer, "atoms": atoms, "chance": 1.0 / atoms, "template": template, "trained_map": target is not None,
                           "compose": compose, "groups": {}}
    states_by_group: dict[str, tuple[torch.Tensor, list[list[tuple[int, int]]], list[bool]]] = {}
    for name in SPLITS:
        rows = splits["splits"][name][:limit] if limit else splits["splits"][name]
        if not rows:
            continue
        t0 = time.monotonic()
        news = [r for r in rows if r["kind"] == "new"]
        olds = [r for r in rows if r["kind"] != "new"]
        states, frames, linked = [], [], []
        if news:                            # new words: inserted zero-shot with their gold frames (compose), fallback rows (free)
            scorer_adapter = edit.extended_adapter(run, {c["surface"]: entry_of[c["concept"]] for c in new_items.concepts})
            all_frames = [next((r["frame"] for r in news if r["concept"] == c["concept"]), None) or [tuple(e) for e in splits_frame(c, run)]
                          for c in new_items.concepts]
            with edit.inserted_entries(channel, len(new_items.concepts), [[tuple(e) for e in f] for f in all_frames]):
                texts = [template.format(x=r["surface"]) for r in news]
                spans = [(template.index("{x}"), template.index("{x}") + len(r["surface"])) for r in news]
                s, l = term_states(run, scorer_adapter, texts, spans, [entry_of[r["concept"]] for r in news], layer)
            states.append(s); linked += l; frames += [[tuple(e) for e in r["frame"]] for r in news]
        if olds:
            entries = [int(r["entry"]) for r in olds]
            surfaces = [alias_surface(run, e, r["surface"]) for e, r in zip(entries, olds)]
            gold = {e: [tuple(x) for x in r["frame"]] for e, r in zip(entries, olds)}
            if name == "seen":              # what the run injects (C5sh: the shuffled frame)
                frames_old = [list(zip(schedule.relations[schedule.offsets[e]:schedule.offsets[e + 1]].tolist(),
                                       schedule.fillers[schedule.offsets[e]:schedule.offsets[e + 1]].tolist())) for e in entries]
                context = swapped_frames(None, {})
            else:                           # held-out entries read their gold frame (as E14's implicit:oracle)
                frames_old = [gold[e] for e in entries]
                context = swapped_frames(channel, gold) if compose else swapped_frames(None, {})
            texts = [template.format(x=x) for x in surfaces]
            spans = [(template.index("{x}"), template.index("{x}") + len(x)) for x in surfaces]
            with context:
                s, l = term_states(run, run.adapter, texts, spans, entries, layer)
            states.append(s); linked += l; frames += frames_old
        all_states = torch.cat(states)
        states_by_group[name] = (all_states, frames, linked)
        group: dict[str, Any] = {"concepts": len(rows), "linked_share": float(np.mean(linked)) if linked else None,
                                 "seconds": round(time.monotonic() - t0, 1)}
        if target is not None:
            result = target.read_frames(all_states.to(run.device), frames, candidates=candidates)
            group["trained"] = summarize_edges({k: v.cpu() for k, v in result.items()}, len(rows), resamples=resamples, seed=seed)
        out["groups"][name] = group
        log(f"  {name}: {len(rows)} concepts, linked {group['linked_share']}, trained acc "
            f"{group.get('trained', {}).get('accuracy')}")
    # Post-hoc ridge probe fitted on the seen split's states → the bundles the run injects (or the fixed code).
    if "seen" in states_by_group:
        x, frames, _ = states_by_group["seen"]
        bundles = reader.bundles(torch.tensor([int(r["entry"]) for r in splits["splits"]["seen"][:len(frames)]], device=run.device)).cpu()
        w = ridge_fit(x, bundles)
        out["probe"] = {"fitted_on": "seen", "alpha": 1e-2, "rows": int(x.shape[0])}
        for name, (s, fr, _) in states_by_group.items():
            codes = (s @ w.T).to(run.device)
            result = reader.read_frames(s.to(run.device), fr, codes=codes, candidates=candidates)
            out["groups"][name]["probe"] = summarize_edges({k: v.cpu() for k, v in result.items()}, len(fr), resamples=resamples, seed=seed)
    return out


def alias_surface(run: Any, entry: int, name: str) -> str:
    """The text an existing entry is read under: its display name when the run's alias table links it to the entry,
    else the entry's longest alias (synthetic T5 entries carry a `synthetic:` prefix in their names)."""
    table = run.table
    if table is None:
        return name
    from ..span_channel import normalize_alias
    if table.alias_to_entry.get(normalize_alias(name)) == entry:
        return name
    aliases = [a for a, e in table.alias_to_entry.items() if e == entry]
    return max(aliases, key=len) if aliases else name


def splits_frame(concept: dict[str, Any], run: Any) -> list[tuple[int, int]]:
    """A new concept's gold frame in ids (for concepts outside the evaluated split, inserted for linking only)."""
    atom_id = {a: i for i, a in enumerate(run.ontology["atomic_names"])}
    relation_id = {r: i for i, r in enumerate(run.ontology["relation_names"])}
    return [(relation_id[r], atom_id[a]) for r, a in concept["frame"]]


def render_read(document: dict[str, Any], *, smoke: bool) -> str:
    lines = (["**SMOKE TEST — not a result.**", ""] if smoke else []) + [
        f"# E15 read accuracy — `{document.get('source', {}).get('run', '?')}`", "",
        f"Layer {document['layer']}; {document['atoms']} atoms (chance {document['chance']:.5f}); trained map: {document['trained_map']}; "
        f"template `{document['template']}`.", "",
        "| split | concepts | linked | trained top-1 (edges) | per-concept [95% CI] | MRR | typed top-1 | probe top-1 |",
        "|---|---:|---:|---:|---|---:|---:|---:|"]
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    for name, g in document["groups"].items():
        t, p = g.get("trained") or {}, g.get("probe") or {}
        ci = f"{f(t.get('concept_accuracy'))} [{f(t.get('ci_low'))}, {f(t.get('ci_high'))}]" if t.get("accuracy") is not None else "—"
        lines.append(f"| {name} | {g['concepts']} | {f(g['linked_share'], 3)} | {f(t.get('accuracy'))} | {ci} | {f(t.get('mrr'))} | "
                     f"{f(t.get('typed_accuracy'))} | {f(p.get('accuracy'))} |")
    return "\n".join(lines) + "\n"


def run_read(args: argparse.Namespace) -> dict[str, Any]:
    from . import e11_read_to_learn as e11
    config = {"experiment": "e15-read", "run": str(args.run), "splits": str(args.splits), "new_items": str(args.new_items),
              "template": args.template, "limit": args.limit, "seed": args.seed, "resamples": args.resamples, "smoke": bool(args.smoke),
              "alias_table": str(args.alias_table) if args.alias_table else None}
    output = Path(args.output)
    git_at_start = start_output(output, config)
    splits = load_splits(Path(args.splits))
    new_items = e11.load_read_set(Path(args.new_items))
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    document = {"source": run.describe(), "smoke": bool(args.smoke), "splits_sha256": sha256_file(Path(args.splits)),
                **read_accuracy(run, splits, new_items, template=args.template, limit=args.limit, resamples=args.resamples, seed=args.seed)}
    write_json(output / "read.json", document)
    (output / "report.md").write_text(render_read(document, smoke=bool(args.smoke)))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


# -- report -----------------------------------------------------------------------------------------------------------

def _values(folder: Path, condition: str, test: str, keep: set[str]) -> dict[str, float]:
    """item id → value of one condition and test from a QA folder's `predictions.jsonl`, items of `keep` concepts."""
    out: dict[str, float] = {}
    path = folder / "predictions.jsonl"
    if not path.exists():
        return out
    for line in path.open():
        row = json.loads(line)
        if row["condition"] != condition or row["concept"] not in keep:
            continue
        if test == "paraphrase":
            if row["test"] == "property" and row.get("consistent") is not None:
                out[row["id"]] = float(row["consistent"])
        elif row["test"] == test and row.get("correct") is not None:
            out[row["id"]] = float(row["correct"])
    return out


def implicit_condition(folder: Path) -> str | None:
    """`implicit:oracle` for compose channels; a free table's implicit route is its `none` (fallback row, no information)."""
    summary = folder / "summary.json"
    if not summary.exists():
        return None
    conditions = json.loads(summary.read_text())["conditions"]
    return "implicit:oracle" if "implicit:oracle" in conditions else "none"


def pooled_contrast(a_folders: Sequence[Path], b_folders: Sequence[Path], a_cond: str, b_cond: str, test: str, keep: set[str], *,
                    resamples: int, seed: int) -> dict[str, Any] | None:
    """`a − b`, items paired within seed (a's seed k with b's seed k), item differences of every seed pooled."""
    from .e5_zeroshot import paired_difference
    diffs = []
    for fa, fb in zip(a_folders, b_folders):
        va, vb = _values(fa, a_cond, test, keep), _values(fb, b_cond, test, keep)
        diffs += [va[i] - vb[i] for i in sorted(set(va) & set(vb))]
    if not diffs:
        return None
    return {**paired_difference(np.asarray(diffs), np.zeros(len(diffs)), resamples=resamples, seed=seed), "seeds": len(a_folders)}


def mean_value(folders: Sequence[Path], cond: str, test: str, keep: set[str]) -> float | None:
    values = [v for f in folders for v in _values(f, cond, test, keep).values()]
    return float(np.mean(values)) if values else None


def read_values(folders: Sequence[Path], split: str, key: str = "trained") -> list[float]:
    """Per-concept read accuracies of a split over the seeds (in order), from `read.json`."""
    out: list[float] = []
    for f in folders:
        path = f / "read.json"
        if path.exists():
            out += (json.loads(path.read_text())["groups"].get(split, {}).get(key) or {}).get("per_concept") or []
    return out


def bootstrap_mean(values: Sequence[float], *, resamples: int, seed: int, null: float = 0.0) -> dict[str, Any] | None:
    from .e5_zeroshot import paired_difference
    if not values:
        return None
    return {**paired_difference(np.asarray(values), np.full(len(values), null), resamples=resamples, seed=seed), "mean_value": float(np.mean(values))}


def loss_contrast(arm_runs: Sequence[Path], base_runs: Sequence[Path], stratum: str, *, resamples: int, seed: int) -> dict[str, Any] | None:
    """Relative change of the final evaluation's loss on `stratum`, arm − base, paired by evaluation window, summed over
    seeds per window (cluster bootstrap over windows), as the E9 reports."""
    from ..training.lm import load_window_losses
    d = n = b = None
    for ra, rb in zip(arm_runs, base_runs):
        pa, pb = ra / "eval_windows.npz", rb / "eval_windows.npz"
        if not (pa.exists() and pb.exists()):
            continue
        wa, wb = load_window_losses(pa), load_window_losses(pb)
        last = max(wa["evals"]); last_b = max(wb["evals"])
        ia, ib = wa["strata"].index(stratum), wb["strata"].index(stratum)
        sa, ca = wa["evals"][last]; sb, cb = wb["evals"][last_b]
        if not np.array_equal(wa["starts"], wb["starts"]):
            raise ValueError(f"{ra} and {rb} evaluate different windows")
        da, nb, bb = sa[ia] - sb[ib], ca[ia], sb[ib]
        d = da if d is None else d + da; n = nb if n is None else n + nb; b = bb if b is None else b + bb
    if d is None:
        return None
    r = paired_ratio_bootstrap(d, n, b, resamples=resamples, seed=seed)
    return {k: r.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative", "relative_ci_low", "relative_ci_high", "clusters")}


def fmt(row: dict[str, Any] | None, key: str = "mean") -> str:
    if not row or row.get(key) is None:
        return "—"
    lo, hi = (row.get("ci_low"), row.get("ci_high")) if key == "mean" else (row.get("relative_ci_low"), row.get("relative_ci_high"))
    scale = 100.0 if key == "relative" else 1.0
    unit = "%" if key == "relative" else ""
    text = f"{row[key] * scale:+.4g}{unit}" if key == "relative" else f"{row[key]:+.4f}"
    if lo is not None:
        text += f" [{lo * scale:+.4g}{unit}, {hi * scale:+.4g}{unit}]" if key == "relative" else f" [{lo:+.4f}, {hi:+.4f}]"
    if row.get("p_value") is not None:
        text += f" (p {row['p_value']:.4f}" + (f", Holm {row['p_holm']:.4f})" if "p_holm" in row else ")")
    return text


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    splits = load_splits(Path(args.splits))
    keep = {name: {r["concept"] for r in rows if r["kind"] == "new"} for name, rows in splits["splits"].items()}
    runs_root, base_root, qa_root, read_root = Path(args.runs_root), Path(args.baseline_root), Path(args.qa_root), Path(args.read_root)
    host = args.host
    folders = lambda root, arm, suffix="": [root / f"{host}-full-{arm}-s{k}{suffix}" for k in SEEDS]  # noqa: E731
    qa = {arm: folders(qa_root, arm) for arm in (*ARMS, *BASELINES)}
    rd = {arm: folders(read_root, arm) for arm in (*ARMS, *BASELINES)}
    runs = {**{arm: folders(runs_root, arm) for arm in ARMS}, **{arm: folders(base_root, arm) for arm in BASELINES}}
    missing = sorted([str(f) for fs in qa.values() for f in fs if not (f / "predictions.jsonl").exists()]
                     + [str(f) for fs in rd.values() for f in fs if not (f / "read.json").exists()])
    implicit = {arm: next((implicit_condition(f) for f in fs if (f / "summary.json").exists()), None) for arm, fs in qa.items()}
    res, seed = args.resamples, args.seed
    comp, strict, unseen = keep["compositional"], keep["compositional_strict"], keep["unseen_filler"]

    def qa_contrast(a: str, b: str, test: str, items: set[str], *, a_cond: str | None = None, b_cond: str | None = None) -> dict[str, Any] | None:
        ca, cb = a_cond or implicit[a], b_cond or implicit[b]
        if ca is None or cb is None:
            return None
        r = pooled_contrast(qa[a], qa[b], ca, cb, test, items, resamples=res, seed=seed)
        return None if r is None else {"a": a, "b": b, "a_condition": ca, "b_condition": cb, "test": test, **r}

    primary: list[dict[str, Any]] = []
    p1 = qa_contrast("C5WR", "C5", "property", comp)
    p2 = qa_contrast("C5WR", "C2R", "property", comp)
    read_a, read_b = read_values(rd["C5WR"], "compositional"), read_values(rd["C2R"], "compositional")
    p3 = None
    if read_a and read_b and len(read_a) == len(read_b):
        from .e5_zeroshot import paired_difference
        p3 = {"a": "C5WR", "b": "C2R", "test": "read_accuracy", **paired_difference(np.asarray(read_a), np.asarray(read_b), resamples=res, seed=seed),
              "a_mean": float(np.mean(read_a)), "b_mean": float(np.mean(read_b))}
    for name, row in (("P1 implicit property, compositional: C5WR − C5", p1), ("P2 implicit property, compositional: C5WR − C2R", p2),
                      ("P3 read accuracy, compositional: C5WR − C2R", p3)):
        if row is not None:
            primary.append({"name": name, **row})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    chance = None
    for f in rd["C5WR"]:
        if (f / "read.json").exists():
            chance = json.loads((f / "read.json").read_text())["chance"]; break
    p3_chance = bootstrap_mean(read_a, resamples=res, seed=seed, null=chance or 0.0) if read_a else None

    secondary: list[dict[str, Any]] = []
    for label, row in (
        ("C5shWR − C5WR, implicit property, compositional", qa_contrast("C5shWR", "C5WR", "property", comp)),
        ("C5R − C5WR, implicit property, compositional", qa_contrast("C5R", "C5WR", "property", comp)),
        ("C5WR − C5, implicit paraphrase, compositional", qa_contrast("C5WR", "C5", "paraphrase", comp)),
        ("C5WR − C5, implicit entailment, compositional", qa_contrast("C5WR", "C5", "entailment", comp)),
        ("C5WR − C5, implicit property, strict compositional", qa_contrast("C5WR", "C5", "property", strict)),
        ("C5WR − C5, implicit property, unseen-filler new words", qa_contrast("C5WR", "C5", "property", unseen)),
        ("C5WR − C5, explicit-rel:oracle property, compositional (explicit recall unchanged?)",
         qa_contrast("C5WR", "C5", "property", comp, a_cond="explicit-rel:oracle", b_cond="explicit-rel:oracle")),
        ("C5WR − C5, none property, compositional", qa_contrast("C5WR", "C5", "property", comp, a_cond="none", b_cond="none")),
        ("C2R − C2, none property, compositional", qa_contrast("C2R", "C2", "property", comp, a_cond="none", b_cond="none")),
        ("C5WR − C5sh, implicit property, compositional", qa_contrast("C5WR", "C5sh", "property", comp)),
    ):
        if row is not None:
            secondary.append({"name": label, **row})
    reads: dict[str, dict[str, Any]] = {}
    for arm in (*ARMS, *BASELINES):
        reads[arm] = {}
        for split in SPLITS:
            for key in ("trained", "probe"):
                values = read_values(rd[arm], split, key)
                if values:
                    reads[arm][f"{split}/{key}"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
    losses: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        losses[arm] = {}
        for stratum in ("after_heldout", "after", "unlinked", "all"):
            r = loss_contrast(runs[arm], runs["C5"], stratum, resamples=res, seed=seed)
            if r is not None:
                losses[arm][stratum] = r
    means = {arm: {test: {cond: mean_value(qa[arm], cond, test, comp) for cond in ("none", "implicit:oracle", "explicit-rel:oracle")}
                   for test in TESTS} for arm in (*ARMS, *BASELINES)}
    # Decision rule (pre-registration §9).
    p1_value = p1["mean"] if p1 else None
    p3_value = p3_chance["mean_value"] if p3_chance else None
    if p1_value is None or p3_value is None:
        decision = "incomplete"
    elif p1_value >= 0.15 and p3_value >= 0.8:
        decision = "algebra learnable with the right signal: stage 2 (T4) is to be queued"
    elif p3_value >= 0.8:
        decision = "decodes but does not use (reading–use gap)"
    elif p3_value < 0.6:
        decision = "the algebra is not learnable by this host at this scale; the implicit route is closed"
    else:
        decision = "intermediate (0.6 ≤ read accuracy < 0.8): no pre-registered reading; report as inconclusive"
    out = {"host": host, "missing": missing, "implicit_condition": implicit, "splits_sha256": sha256_file(Path(args.splits)),
           "split_sizes": {k: len(v) for k, v in splits["splits"].items()}, "primary": primary, "primary3_vs_chance": p3_chance,
           "chance": chance, "secondary": secondary, "read": reads, "losses_vs_C5": losses, "means_compositional": means,
           "decision": decision}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text(render_report(out))
    print(json.dumps({"decision": decision, "primary": [(r["name"], r["mean"]) for r in primary]}, indent=2))
    return out


def render_report(out: dict[str, Any]) -> str:
    lines = ["# E15 — HRR as a training target: T5 report (pre-registration `experiments/e15-hrr-target/preregistration.md`)", "",
             f"Host {out['host']}; splits sha256 `{out['splits_sha256']}`; split sizes {out['split_sizes']}.", ""]
    if out["missing"]:
        lines += ["**Missing evaluation folders:** " + ", ".join(out["missing"]), ""]
    lines += ["## Primary (Holm over the three)", "", "| contrast | difference [95% CI] | n |", "|---|---|---:|"]
    lines += [f"| {r['name']} | {fmt(r)} | {r['n']} |" for r in out["primary"]]
    if out.get("primary3_vs_chance"):
        r = out["primary3_vs_chance"]
        lines += ["", f"P3 against chance ({out['chance']:.5f}): C5WR compositional read accuracy {r['mean_value']:.4f}, "
                  f"difference to chance {fmt(r)}."]
    lines += ["", f"**Decision rule reading:** {out['decision']}", ""]
    lines += ["## Means on the compositional items (3 seeds pooled)", "", "| arm | test | none | implicit:oracle | explicit-rel:oracle |", "|---|---|---:|---:|---:|"]
    for arm, by_test in out["means_compositional"].items():
        for test, by_cond in by_test.items():
            f = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
            lines.append(f"| {arm} | {test} | {f(by_cond['none'])} | {f(by_cond['implicit:oracle'])} | {f(by_cond['explicit-rel:oracle'])} |")
    lines += ["", "## Read accuracy (per-concept mean over seeds; `trained` = the run's W, `probe` = post-hoc ridge fitted on seen terms)", "",
              "| arm | split / map | accuracy | difference to chance [95% CI] | n |", "|---|---|---:|---|---:|"]
    for arm, by in out["read"].items():
        for key, r in by.items():
            lines.append(f"| {arm} | {key} | {r['mean_value']:.4f} | {fmt(r)} | {r['n']} |")
    lines += ["", "## Loss endpoints against C5 (final evaluation, relative change, paired windows summed over seeds)", "",
              "| arm | stratum | relative change [95% CI] |", "|---|---|---|"]
    for arm, by in out["losses_vs_C5"].items():
        for stratum, r in by.items():
            lines.append(f"| {arm} | {stratum} | {fmt(r, 'relative')} |")
    lines += ["", "## Secondary (unadjusted)", "", "| contrast | difference [95% CI] | n |", "|---|---|---:|"]
    lines += [f"| {r['name']} | {fmt(r)} | {r['n']} |" for r in out["secondary"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="write the fixed splits (with sha256)")
    s.add_argument("--ontology", type=Path, required=True); s.add_argument("--new-items", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True); s.add_argument("--seen", type=int, default=300); s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("read", help="read accuracy of a trained run on the splits")
    r.add_argument("--run", type=Path, required=True); r.add_argument("--splits", type=Path, required=True)
    r.add_argument("--new-items", type=Path, required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--alias-table", type=Path, default=None); r.add_argument("--checkpoint", default="final.pt")
    r.add_argument("--device", default=None); r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--template", default=TEMPLATE); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--resamples", type=int, default=2000); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--smoke", action="store_true")
    p = sub.add_parser("report", help="pool the arms' QA, read accuracy and loss endpoints")
    p.add_argument("--runs-root", type=Path, required=True); p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--qa-root", type=Path, required=True); p.add_argument("--read-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="SmolLM2-360M"); p.add_argument("--resamples", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"split": run_split, "read": run_read, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
