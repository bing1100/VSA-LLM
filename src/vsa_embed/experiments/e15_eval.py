"""E15 — HRR as a training target: splits, read accuracy and the pooled report (pre-registration
`experiments/e15-hrr-target/preregistration.md`).

    python -m vsa_embed.experiments.e15_eval split --ontology ONT.pt --new-items DIR --output DIR [--seen 300] [--seed 0]
    python -m vsa_embed.experiments.e15_eval read --run RUN --splits DIR/splits.json --new-items DIR --output OUT
        [--alias-table T.json] [--template "Note: {x}"] [--limit N] [--smoke]
    python -m vsa_embed.experiments.e15_eval report --runs-root R --qa-root Q --read-root D --splits S --output OUT
    python -m vsa_embed.experiments.e15_eval report --stage 1b ... [--checkpoints 50000000,100000000,150000000,200000000]
    python -m vsa_embed.experiments.e15_eval report --stage 1c ...
    python -m vsa_embed.experiments.e15_eval report --stage 1d ... [--seeds 1] [--checkpoints 100000000,...,500000000]
    python -m vsa_embed.experiments.e15_eval report --stage 1e ...
    python -m vsa_embed.experiments.e15_eval read ... [--gate-closed]

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
decision rule. `report --stage 1b` (pre-registration, "Stage 1b") reads the checkpoint evaluations of the long runs
(`{read,qa}/t5/<run>/ckpt-<tokens>/`): the read-accuracy and QA curve over the checkpoints, the three stage-1b
primaries (implicit property at the checkpoint with the highest compositional read accuracy, C5WR-long − C5; read
accuracy at the last checkpoint; the slope over the checkpoints with the last increment), the secondaries and the
stage-1b decision reading; since 2026-10-23 the loss endpoints are also read against the matched-length control
`C5-long` (E9's C5 recipe at 200M tokens, seed 1) when its run exists. `report --stage 1c` (pre-registration, "Stage
1c") pools the aligned-objective arms C5RD / C5FV (3 seeds) and their free-table controls C2RD / C2FV (seed 1) against
E9's C5 / C2 and stage 1's C5WR: the three primaries (implicit property on the compositional split, C5RD − C5,
C5FV − C5, C5RD − C2RD), the secondaries, the probe read accuracy (no trained `W` in these arms), the loss endpoints,
the end-of-training teacher–student KL relative to its start (from the runs' train rows) and the stage-1c decision
reading. `report --stage 1d` (pre-registration, "Stage 1d": 500M tokens, seed 1 unless `--seeds` names more) reads the
checkpoint evaluations of the matched-length pair `C5WR-xl` / `C5-xl` at 100–500M: the read and QA curves of both arms,
the three primaries (implicit property C5WR-xl − C5-xl at 500M, the objective's share at matched length; compositional
read accuracy at 500M against the 0.80 bar; the 400 → 500M increment), the secondaries (continuity C5WR-xl − C5, length
alone C5-xl − C5, loss endpoints, strict / unseen-filler splits, the consistency check against stage 1b's 200M checkpoint)
and the stage-1d decision reading with the extrapolated crossing when the curve is still rising below the bar.
`report --stage 1e` (pre-registration, "Stage 1e": read, then inject) pools the injection arms C5WRI (3 seeds), the
exploratory C5WRI-nogate and the controls C2RI / C5shWRI (seed 1) against stage 1's C5WR / C2R / C5shWR and E9's C5: the
three primaries (implicit property on the compositional split, C5WRI − C5WR, C5WRI − C2RI, C5WRI − C5WRI with the gate
closed at evaluation — the QA folders `<run>/gate-closed/` written by `e14_reading recall --gate-closed`), the
secondaries, the trained-`W` read accuracy against C5WR's, the loss endpoints against C5 and C5WR, the learned gate
values (from `read.json` or the runs' train rows) and the stage-1e decision reading. `read --gate-closed` evaluates with
the injection switched off (the capture layer is before the injection, so read accuracy does not change; the flag is
recorded) and `read.json` carries the run's `injection` record when it has one.
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
STAGE1B_CHECKPOINTS = (50_000_000, 100_000_000, 150_000_000, 200_000_000)
STAGE1B_ARMS = ("C5WR-long", "C2R-long", "C5WR-long-lr0.10")
STAGE1D_CHECKPOINTS = (100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000)
STAGE1D_ARMS = ("C5WR-xl", "C5-xl")
STAGE1D_READ_BAR, STAGE1D_SHARE_BAR = 0.80, 0.10
STAGE1C_ARMS = ("C5RD", "C5FV", "C2RD", "C2FV")
STAGE1C_SEEDS = {"C5RD": (1, 2, 3), "C5FV": (1, 2, 3), "C2RD": (1,), "C2FV": (1,)}
STAGE1E_ARMS = ("C5WRI", "C5WRI-nogate", "C2RI", "C5shWRI")
STAGE1E_SEEDS = {"C5WRI": (1, 2, 3), "C5WRI-nogate": (1,), "C2RI": (1,), "C5shWRI": (1,)}
STAGE1E_GATE_CLOSED = "gate-closed"          # the QA sub-folder of the gate-closed evaluation (`e14_reading recall --gate-closed`)
STAGE1E_LEVEL_BAR, STAGE1E_PARTIAL_BAR, STAGE1E_CONTROL_BAR = 0.50, 0.35, 0.15
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
              "alias_table": str(args.alias_table) if args.alias_table else None, "gate_closed": bool(getattr(args, "gate_closed", False))}
    output = Path(args.output)
    git_at_start = start_output(output, config)
    splits = load_splits(Path(args.splits))
    new_items = e11.load_read_set(Path(args.new_items))
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    from ..hrr_target import injection_record, set_injection
    if getattr(args, "gate_closed", False) and not set_injection(run.adapter.model, False):     # stage 1e: the injection ablation
        raise SystemExit(f"{args.run}: --gate-closed needs a run with channel.target.inject")
    injection = injection_record(run.adapter.model)
    document = {"source": run.describe(), "smoke": bool(args.smoke), "splits_sha256": sha256_file(Path(args.splits)),
                **({"injection": injection, "gate_closed": bool(getattr(args, "gate_closed", False))} if injection is not None else {}),
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


# -- report, stage 1b (the checkpoint curve) ---------------------------------------------------------------------------

def parse_checkpoints(text: str | None, default: Sequence[int]) -> list[int]:
    """`--checkpoints`: a comma-separated list of token counts; absent or empty → the stage's registered checkpoints."""
    if text is None or not str(text).strip():
        return list(default)
    return sorted({int(t) for t in str(text).split(",") if t.strip()}) or list(default)


def read_curve(folders_by_checkpoint: dict[int, Sequence[Path]], split: str, key: str = "trained") -> dict[int, list[float]]:
    """Checkpoint → per-concept read accuracies pooled over seeds (same order at every checkpoint), only checkpoints
    whose every folder has a `read.json`."""
    out: dict[int, list[float]] = {}
    for t, folders in folders_by_checkpoint.items():
        if all((f / "read.json").exists() for f in folders):
            values = read_values(folders, split, key)
            if values:
                out[t] = values
    return out


def per_item_slopes(curve: dict[int, Sequence[float]]) -> list[float]:
    """Least-squares slope of each item's accuracy over the checkpoints, in accuracy per 50M tokens (items paired across
    checkpoints: the same concept and seed at the same index)."""
    points = sorted(curve)
    if len(points) < 2 or len({len(curve[t]) for t in points}) != 1:
        return []
    x = np.asarray(points, dtype=float) / 50e6
    y = np.asarray([curve[t] for t in points], dtype=float)       # checkpoints × items
    xc = x - x.mean()
    return ((xc[:, None] * (y - y.mean(0))).sum(0) / (xc ** 2).sum()).tolist()


def run_report_1b(args: argparse.Namespace) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    splits = load_splits(Path(args.splits))
    keep = {name: {r["concept"] for r in rows if r["kind"] == "new"} for name, rows in splits["splits"].items()}
    comp, strict, unseen = keep["compositional"], keep["compositional_strict"], keep["unseen_filler"]
    runs_root, base_root, qa_root, read_root = Path(args.runs_root), Path(args.baseline_root), Path(args.qa_root), Path(args.read_root)
    host, res, seed = args.host, args.resamples, args.seed
    checkpoints = parse_checkpoints(getattr(args, "checkpoints", None), STAGE1B_CHECKPOINTS)
    last = checkpoints[-1]
    name = lambda arm, k: f"{host}-full-{arm}-s{k}"  # noqa: E731
    long_read = {t: [read_root / name("C5WR-long", k) / f"ckpt-{t}" for k in SEEDS] for t in checkpoints}
    long_qa = {t: [qa_root / name("C5WR-long", k) / f"ckpt-{t}" for k in SEEDS] for t in checkpoints}
    long_runs = [runs_root / name("C5WR-long", k) for k in SEEDS]
    c5_qa, c5_runs = [qa_root / name("C5", k) for k in SEEDS], [base_root / name("C5", k) for k in SEEDS]
    s1_read, s1_runs = [read_root / name("C5WR", k) for k in SEEDS], [runs_root / name("C5WR", k) for k in SEEDS]
    c2r_read, c2r_qa, c2r_run = read_root / name("C2R-long", 1) / f"ckpt-{last}", qa_root / name("C2R-long", 1) / f"ckpt-{last}", runs_root / name("C2R-long", 1)
    lr_read, lr_run = read_root / name("C5WR-long-lr0.10", 1) / f"ckpt-{last}", runs_root / name("C5WR-long-lr0.10", 1)
    expected_qa = [f for fs in long_qa.values() for f in fs] + c5_qa + [c2r_qa]
    expected_read = [f for fs in long_read.values() for f in fs] + s1_read + [c2r_read, lr_read]
    missing = sorted([str(f) for f in expected_qa if not (f / "predictions.jsonl").exists()]
                     + [str(f) for f in expected_read if not (f / "read.json").exists()])
    chance = None
    for f in expected_read:
        if (f / "read.json").exists():
            chance = json.loads((f / "read.json").read_text())["chance"]; break
    implicit_c5 = next((implicit_condition(f) for f in c5_qa if (f / "summary.json").exists()), None)
    implicit_c2r = implicit_condition(c2r_qa) if (c2r_qa / "summary.json").exists() else None

    def qa_contrast(a_folders, b_folders, a_cond, b_cond, test, items) -> dict[str, Any] | None:
        if a_cond is None or b_cond is None:
            return None
        r = pooled_contrast(a_folders, b_folders, a_cond, b_cond, test, items, resamples=res, seed=seed)
        return None if r is None else {"a_condition": a_cond, "b_condition": b_cond, "test": test, **r}

    # The curve: read accuracy per split and the QA contrast against C5 at every checkpoint.
    curve: dict[int, dict[str, Any]] = {}
    for t in checkpoints:
        row: dict[str, Any] = {"read": {}, "qa_vs_C5": {}, "means": {}}
        for split in SPLITS:
            for key in ("trained", "probe"):
                values = read_curve({t: long_read[t]}, split, key).get(t)
                if values:
                    row["read"][f"{split}/{key}"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
        for test in TESTS:
            r = qa_contrast(long_qa[t], c5_qa, "implicit:oracle", implicit_c5, test, comp)
            if r is not None:
                row["qa_vs_C5"][test] = r
        row["means"] = {test: {cond: mean_value(long_qa[t], cond, test, comp) for cond in ("none", "implicit:oracle", "explicit-rel:oracle")}
                        for test in TESTS}
        curve[t] = row
    comp_curve = read_curve(long_read, "compositional")
    # Best checkpoint by compositional read accuracy (never by QA): the checkpoint whose pooled per-concept mean is highest.
    best = max(comp_curve, key=lambda t: float(np.mean(comp_curve[t]))) if comp_curve else None

    primary: list[dict[str, Any]] = []
    p1 = qa_contrast(long_qa[best], c5_qa, "implicit:oracle", implicit_c5, "property", comp) if best is not None else None
    if p1 is not None:
        primary.append({"name": f"P1 implicit property, compositional, best checkpoint by read accuracy ({best:,} tokens): C5WR-long − C5",
                        "checkpoint": best, **p1})
    p2 = bootstrap_mean(comp_curve.get(last, []), resamples=res, seed=seed, null=chance or 0.0)
    if p2 is not None:
        primary.append({"name": f"P2 compositional read accuracy at {last:,} tokens, C5WR-long (difference to chance)", "checkpoint": last, **p2})
    slopes = per_item_slopes(comp_curve) if len(comp_curve) == len(checkpoints) else []
    p3 = bootstrap_mean(slopes, resamples=res, seed=seed) if slopes else None
    if p3 is not None:
        primary.append({"name": f"P3 slope of compositional read accuracy over {len(checkpoints)} checkpoints (per 50M tokens, paired per concept)",
                        "checkpoints": checkpoints, **p3})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    increment = None
    if len(checkpoints) >= 2 and checkpoints[-2] in comp_curve and last in comp_curve and len(comp_curve[last]) == len(comp_curve[checkpoints[-2]]):
        increment = {"from": checkpoints[-2], "to": last,
                     **paired_difference(np.asarray(comp_curve[last]), np.asarray(comp_curve[checkpoints[-2]]), resamples=res, seed=seed)}
        increment["still_rising"] = bool(increment["ci_low"] > 0)

    secondary: list[dict[str, Any]] = []
    for split, label in (("compositional_strict", "strict"), ("unseen_filler", "unseen-filler")):
        for t in checkpoints:
            r = curve[t]["read"].get(f"{split}/trained")
            if r is not None:
                secondary.append({"name": f"read accuracy, {label}, C5WR-long at {t:,} tokens (difference to chance)", **r})
    a, b = read_values([c2r_read], "compositional"), read_values([read_root / name("C5WR-long", 1) / f"ckpt-{last}"], "compositional")
    if a and b and len(a) == len(b):
        secondary.append({"name": f"read accuracy, compositional at {last:,} tokens: C2R-long − C5WR-long (seed 1, paired per concept)",
                          **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    if a:
        secondary.append({"name": f"read accuracy, compositional at {last:,} tokens: C2R-long (difference to chance)",
                          **bootstrap_mean(a, resamples=res, seed=seed, null=chance or 0.0)})
    r = qa_contrast([c2r_qa], [qa_root / name("C5WR-long", 1) / f"ckpt-{last}"], implicit_c2r, "implicit:oracle", "property", comp)
    if r is not None:
        secondary.append({"name": f"implicit property, compositional at {last:,} tokens: C2R-long − C5WR-long (seed 1)", **r})
    a = read_values([lr_read], "compositional")
    if a and b and len(a) == len(b):
        secondary.append({"name": f"read accuracy, compositional at {last:,} tokens: lr0.10 − C5WR-long seed 1 (paired per concept)",
                          **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    # The matched-length control C5-long (added 2026-10-23, §15 of the pre-registration, before any 1b evaluation ran): E9's C5
    # recipe at 200M tokens, seed 1; its QA and read at 200M are evaluated like (b)'s. Every loss contrast against it pairs the
    # arm's windows with the one C5-long run's (the three C5WR-long seeds each against the same control).
    c5long_run = runs_root / name("C5-long", 1)
    c5long_qa, c5long_read = qa_root / name("C5-long", 1) / f"ckpt-{last}", read_root / name("C5-long", 1) / f"ckpt-{last}"
    has_c5long = (c5long_run / "eval_windows.npz").exists()
    contrasts = [("C5WR-long − C5", long_runs, c5_runs), ("C5WR-long − C5WR (stage 1)", long_runs, s1_runs),
                 ("C2R-long − C5", [c2r_run], [c5_runs[0]]), ("lr0.10 − C5WR-long s1", [lr_run], [long_runs[0]]), ("lr0.10 − C5", [lr_run], [c5_runs[0]])]
    if has_c5long:
        contrasts += [("C5WR-long − C5-long (matched length; 3 seeds against the one control run)", long_runs, [c5long_run] * len(long_runs)),
                      ("C2R-long − C5-long", [c2r_run], [c5long_run]), ("lr0.10 − C5-long", [lr_run], [c5long_run]),
                      ("C5-long − C5 (length alone, seed 1)", [c5long_run], [c5_runs[0]])]
    losses: dict[str, dict[str, Any]] = {}
    for label, arm_runs, base_runs in contrasts:
        losses[label] = {}
        for stratum in ("after_heldout", "after", "unlinked", "all"):
            r = loss_contrast(arm_runs, base_runs, stratum, resamples=res, seed=seed)
            if r is not None:
                losses[label][stratum] = r
    c5long: dict[str, Any] | None = None
    if has_c5long or (c5long_qa / "predictions.jsonl").exists() or (c5long_read / "read.json").exists():
        c5long = {"run": str(c5long_run), "losses_present": has_c5long,
                  "means": {test: {cond: mean_value([c5long_qa], cond, test, comp) for cond in ("none", "implicit:oracle", "explicit-rel:oracle")} for test in TESTS},
                  "read_probe": {split: bootstrap_mean(read_values([c5long_read], split, "probe"), resamples=res, seed=seed, null=chance or 0.0) for split in SPLITS}}
        implicit_c5long = implicit_condition(c5long_qa) if (c5long_qa / "summary.json").exists() else None
        r = qa_contrast([qa_root / name("C5WR-long", 1) / f"ckpt-{last}"], [c5long_qa], "implicit:oracle", implicit_c5long, "property", comp)
        if r is not None:
            secondary.append({"name": f"implicit property, compositional at {last:,} tokens: C5WR-long − C5-long (seed 1, matched length)", **r})
        a, b = read_values([read_root / name("C5WR-long", 1) / f"ckpt-{last}"], "compositional"), read_values([c5long_read], "compositional", "probe")
        if a and b and len(a) == len(b):
            secondary.append({"name": f"read accuracy, compositional at {last:,} tokens: C5WR-long (trained W) − C5-long (probe), seed 1, paired per concept",
                              **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    consistency = None
    a, b = comp_curve.get(checkpoints[0], []), read_values(s1_read, "compositional")
    if a and b and len(a) == len(b):
        consistency = {"checkpoint": checkpoints[0], **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed),
                       "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))}
    # Decision rule (pre-registration, stage 1b). The QA gate reads every checkpoint; the read gate reads the last one.
    qa_means = [curve[t]["qa_vs_C5"]["property"]["mean"] for t in checkpoints if "property" in curve[t]["qa_vs_C5"]]
    read_last = p2["mean_value"] if p2 else None
    readings: list[str] = []
    if qa_means and max(qa_means) >= 0.15:
        readings.append("implicit property ≥ +0.15 over C5 at some checkpoint: stage 2 (T4, registered) is to be queued")
    if read_last is None or increment is None:
        readings.append("incomplete")
    elif read_last >= 0.8:
        readings.append("read accuracy ≥ 0.8 at the last checkpoint: the algebra is learnable to the original standard")
    elif read_last >= 0.6 and increment["ci_low"] > 0:
        readings.append("read accuracy in [0.6, 0.8) and still rising (last increment's CI excludes 0): register stage 1c at 500M before any verdict")
    elif read_last < 0.6 and increment["ci_low"] <= 0 <= increment["ci_high"]:
        readings.append("read accuracy flat below 0.6 (last increment's CI includes 0): the implicit route is closed")
    else:
        readings.append("no pre-registered reading for this combination of level and slope; report the numbers")
    out = {"stage": "1b", "host": host, "checkpoints": checkpoints, "best_checkpoint_by_read": best, "missing": missing,
           "implicit_condition": {"C5": implicit_c5, "C2R-long": implicit_c2r}, "splits_sha256": sha256_file(Path(args.splits)),
           "split_sizes": {k: len(v) for k, v in splits["splits"].items()}, "chance": chance, "curve": curve, "primary": primary,
           "increment": increment, "secondary": secondary, "losses": losses, "consistency_50M_vs_stage1": consistency,
           "c5_long_control": c5long, "decision": "; ".join(readings)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text(render_report_1b(out))
    print(json.dumps({"decision": out["decision"], "primary": [(r["name"], r.get("mean")) for r in primary]}, indent=2))
    return out


def render_report_1b(out: dict[str, Any]) -> str:
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["# E15 stage 1b — longer training of the HRR-as-target arms: T5 report (pre-registration "
             "`experiments/e15-hrr-target/preregistration.md`, \"Stage 1b\")", "",
             f"Host {out['host']}; checkpoints {[f'{t:,}' for t in out['checkpoints']]} tokens; splits sha256 `{out['splits_sha256']}`; "
             f"split sizes {out['split_sizes']}; chance {f(out['chance'], 5)}.", ""]
    if out["missing"]:
        lines += ["**Missing evaluation folders:** " + ", ".join(out["missing"]), ""]
    lines += ["## The curve (C5WR-long, 3 seeds pooled; per-concept read accuracy, QA = implicit property vs C5)", "",
              "| tokens | compositional | strict | unseen-filler | seen | heldout | QA property C5WR-long − C5 | implicit:oracle | none | explicit |",
              "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for t, row in out["curve"].items():
        rd = lambda k: f((row["read"].get(f"{k}/trained") or {}).get("mean_value"))  # noqa: E731
        qa = row["qa_vs_C5"].get("property")
        m = row["means"].get("property", {})
        lines.append(f"| {int(t):,} | {rd('compositional')} | {rd('compositional_strict')} | {rd('unseen_filler')} | {rd('seen')} | "
                     f"{rd('heldout_ontology')} | {fmt(qa)} | {f(m.get('implicit:oracle'))} | {f(m.get('none'))} | {f(m.get('explicit-rel:oracle'))} |")
    lines += ["", f"Best checkpoint by compositional read accuracy (not by QA): "
              f"{out['best_checkpoint_by_read']:,} tokens." if out["best_checkpoint_by_read"] else "", ""]
    lines += ["## Primary (Holm over the three)", "", "| endpoint | estimate [95% CI] | n |", "|---|---|---:|"]
    for r in out["primary"]:
        est = fmt(r) if "mean_value" not in r else f"{r['mean_value']:.4f}; to chance {fmt(r)}"
        lines.append(f"| {r['name']} | {est} | {r['n']} |")
    if out.get("increment"):
        i = out["increment"]
        lines += ["", f"Last increment {i['from']:,} → {i['to']:,} tokens (paired per concept): {fmt(i)} — "
                  + ("**still rising** (CI excludes 0)." if i["still_rising"] else "not distinguishable from flat (CI includes 0).")]
    lines += ["", f"**Decision rule reading:** {out['decision']}", ""]
    if out.get("consistency_50M_vs_stage1"):
        c = out["consistency_50M_vs_stage1"]
        lines += [f"Consistency check: the {c['checkpoint']:,}-token checkpoint of C5WR-long ({c['a_mean']:.4f}) against the stage-1 C5WR "
                  f"runs ({c['b_mean']:.4f}; same tokens, different lr schedule): {fmt(c)}.", ""]
    lines += ["## Loss endpoints (final evaluation, relative change, paired windows summed over seeds)", "",
              "Against C5 (50M tokens) and, where its run exists, against the matched-length control C5-long (E9's C5 recipe at 200M "
              "tokens, seed 1; added 2026-10-23 before any 1b evaluation ran, because a 200M arm against a 50M baseline confounds the "
              "objective with the tokens).", "",
              "| contrast | stratum | relative change [95% CI] |", "|---|---|---|"]
    for label, by in out["losses"].items():
        for stratum, r in by.items():
            lines.append(f"| {label} | {stratum} | {fmt(r, 'relative')} |")
    if out.get("c5_long_control"):
        c = out["c5_long_control"]
        m = c["means"].get("property", {})
        lines += ["", f"C5-long at 200M (seed 1): implicit property {f(m.get('implicit:oracle'))}, none {f(m.get('none'))}, explicit "
                  f"{f(m.get('explicit-rel:oracle'))}; compositional probe read accuracy "
                  f"{f((c['read_probe'].get('compositional') or {}).get('mean_value'))}" + ("" if c["losses_present"] else "; loss endpoints not yet available") + "."]
    lines += ["", "## Secondary (unadjusted)", "", "| contrast | estimate [95% CI] | n |", "|---|---|---:|"]
    for r in out["secondary"]:
        est = fmt(r) if "mean_value" not in r else f"{r['mean_value']:.4f}; to chance {fmt(r)}"
        lines.append(f"| {r['name']} | {est} | {r['n']} |")
    return "\n".join(lines) + "\n"


# -- report, stage 1c (aligned objectives) -----------------------------------------------------------------------------

def kl_ratio(run: Path, key: str = "kl_loss", share: float = 0.05) -> dict[str, Any] | None:
    """End-of-training value of a train-row column relative to its start: the mean over the last `share` of the train rows
    over the mean over the first `share` (rows with the column present; None without them)."""
    path = run / "metrics.jsonl"
    if not path.exists():
        return None
    values = [row[key] for line in path.open() if (row := json.loads(line)).get("type") == "train" and key in row]
    if len(values) < 2:
        return None
    n = max(1, int(round(share * len(values))))
    start, end = float(np.mean(values[:n])), float(np.mean(values[-n:]))
    return {"start": start, "end": end, "ratio": end / start if start else None, "rows": len(values), "rows_averaged": n}


def run_report_1c(args: argparse.Namespace) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    splits = load_splits(Path(args.splits))
    keep = {name: {r["concept"] for r in rows if r["kind"] == "new"} for name, rows in splits["splits"].items()}
    comp, strict, unseen = keep["compositional"], keep["compositional_strict"], keep["unseen_filler"]
    runs_root, base_root, qa_root, read_root = Path(args.runs_root), Path(args.baseline_root), Path(args.qa_root), Path(args.read_root)
    host, res, seed = args.host, args.resamples, args.seed
    name = lambda arm, k: f"{host}-full-{arm}-s{k}"  # noqa: E731
    seeds = {**STAGE1C_SEEDS, "C5": SEEDS, "C2": (1,), "C5WR": SEEDS}
    qa = {arm: [qa_root / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    rd = {arm: [read_root / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    runs = {arm: [(base_root if arm in BASELINES else runs_root) / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    missing = sorted([str(f) for arm in (*STAGE1C_ARMS, "C5", "C2") for f in qa[arm] if not (f / "predictions.jsonl").exists()]
                     + [str(f) for arm in (*STAGE1C_ARMS, "C5") for f in rd[arm] if not (f / "read.json").exists()])
    implicit = {arm: next((implicit_condition(f) for f in fs if (f / "summary.json").exists()), None) for arm, fs in qa.items()}

    def qa_contrast(a: str, b: str, test: str, items: set[str], *, a_cond: str | None = None, b_cond: str | None = None) -> dict[str, Any] | None:
        ca, cb = a_cond or implicit[a], b_cond or implicit[b]
        if ca is None or cb is None:
            return None
        fa, fb = qa[a], qa[b]
        if len(fa) != len(fb):                      # a 3-seed arm against a 1-seed control: seed 1 against seed 1
            fa, fb = fa[:1], fb[:1]
        r = pooled_contrast(fa, fb, ca, cb, test, items, resamples=res, seed=seed)
        return None if r is None else {"a": a, "b": b, "a_condition": ca, "b_condition": cb, "test": test, **r}

    primary: list[dict[str, Any]] = []
    for label, row in (("P1 implicit property, compositional: C5RD − C5", qa_contrast("C5RD", "C5", "property", comp)),
                       ("P2 implicit property, compositional: C5FV − C5", qa_contrast("C5FV", "C5", "property", comp)),
                       ("P3 implicit property, compositional: C5RD − C2RD (structure against memorisation; seed 1)", qa_contrast("C5RD", "C2RD", "property", comp))):
        if row is not None:
            primary.append({"name": label, **row})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    secondary: list[dict[str, Any]] = []
    for label, row in (
        ("C5RD − C5, implicit paraphrase, compositional", qa_contrast("C5RD", "C5", "paraphrase", comp)),
        ("C5RD − C5, implicit entailment, compositional", qa_contrast("C5RD", "C5", "entailment", comp)),
        ("C5FV − C5, implicit paraphrase, compositional", qa_contrast("C5FV", "C5", "paraphrase", comp)),
        ("C5FV − C5, implicit entailment, compositional", qa_contrast("C5FV", "C5", "entailment", comp)),
        ("C5RD − C5WR (stage 1), implicit property, compositional", qa_contrast("C5RD", "C5WR", "property", comp)),
        ("C5FV − C5WR (stage 1), implicit property, compositional", qa_contrast("C5FV", "C5WR", "property", comp)),
        ("C5RD − C5FV, implicit property, compositional", qa_contrast("C5RD", "C5FV", "property", comp)),
        ("C5FV − C2FV, implicit property, compositional (seed 1)", qa_contrast("C5FV", "C2FV", "property", comp)),
        ("C5RD − C5, explicit-rel:oracle property, compositional (explicit recall unchanged?)",
         qa_contrast("C5RD", "C5", "property", comp, a_cond="explicit-rel:oracle", b_cond="explicit-rel:oracle")),
        ("C5FV − C5, explicit-rel:oracle property, compositional (explicit recall unchanged?)",
         qa_contrast("C5FV", "C5", "property", comp, a_cond="explicit-rel:oracle", b_cond="explicit-rel:oracle")),
        ("C5RD − C5, none property, compositional", qa_contrast("C5RD", "C5", "property", comp, a_cond="none", b_cond="none")),
        ("C5FV − C5, none property, compositional", qa_contrast("C5FV", "C5", "property", comp, a_cond="none", b_cond="none")),
        ("C2RD − C2, none property, compositional (seed 1)", qa_contrast("C2RD", "C2", "property", comp, a_cond="none", b_cond="none")),
        ("C2FV − C2, none property, compositional (seed 1)", qa_contrast("C2FV", "C2", "property", comp, a_cond="none", b_cond="none")),
        ("C5RD − C5, implicit property, strict compositional", qa_contrast("C5RD", "C5", "property", strict)),
        ("C5FV − C5, implicit property, strict compositional", qa_contrast("C5FV", "C5", "property", strict)),
        ("C5RD − C5, implicit property, unseen-filler new words", qa_contrast("C5RD", "C5", "property", unseen)),
        ("C5FV − C5, implicit property, unseen-filler new words", qa_contrast("C5FV", "C5", "property", unseen)),
    ):
        if row is not None:
            secondary.append({"name": label, **row})
    chance = None
    for f in rd["C5RD"] + rd["C5"]:
        if (f / "read.json").exists():
            chance = json.loads((f / "read.json").read_text())["chance"]; break
    reads: dict[str, dict[str, Any]] = {}
    for arm in (*STAGE1C_ARMS, "C5", "C2"):           # no trained W in these arms: the post-hoc probe is the only read accuracy
        reads[arm] = {}
        for split in SPLITS:
            values = read_values(rd[arm], split, "probe")
            if values:
                reads[arm][f"{split}/probe"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
    a, b = read_values(rd["C5RD"], "compositional", "probe"), read_values(rd["C5"], "compositional", "probe")
    if a and b and len(a) == len(b):
        secondary.append({"name": "probe read accuracy, compositional: C5RD − C5 (paired per concept and seed)",
                          **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    a = read_values(rd["C5FV"], "compositional", "probe")
    if a and b and len(a) == len(b):
        secondary.append({"name": "probe read accuracy, compositional: C5FV − C5 (paired per concept and seed)",
                          **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    losses: dict[str, dict[str, Any]] = {}
    for arm in STAGE1C_ARMS:
        losses[arm] = {}
        base = runs["C5"][:len(runs[arm])]
        for stratum in ("after_heldout", "after", "unlinked", "all"):
            r = loss_contrast(runs[arm], base, stratum, resamples=res, seed=seed)
            if r is not None:
                losses[arm][stratum] = r
    ratios = {arm: {k: v for k, v in ((f"s{k}", kl_ratio(run, key)) for k, run in zip(seeds[arm], runs[arm])) if v is not None}
              for arm, key in (("C5RD", "kl_loss"), ("C2RD", "kl_loss"), ("C5FV", "cloze_loss"), ("C2FV", "cloze_loss"))}
    ratio_summary = {arm: {"start": float(np.mean([v["start"] for v in by.values()])), "end": float(np.mean([v["end"] for v in by.values()])),
                           "ratio": float(np.mean([v["end"] for v in by.values()]) / np.mean([v["start"] for v in by.values()])), "seeds": len(by)}
                     for arm, by in ratios.items() if by and np.mean([v["start"] for v in by.values()]) > 0}
    means = {arm: {test: {cond: mean_value(qa[arm], cond, test, comp) for cond in ("none", "implicit:oracle", "explicit-rel:oracle")}
                   for test in TESTS} for arm in (*STAGE1C_ARMS, "C5", "C2", "C5WR")}
    implicit_level = {arm: mean_value(qa[arm], implicit[arm], "property", comp) if implicit[arm] else None for arm in (*STAGE1C_ARMS, "C5", "C2", "C5WR")}
    # Decision rule (pre-registration, stage 1c): the level of C5RD's implicit property on the compositional split and P3.
    level, p3 = implicit_level.get("C5RD"), next((r for r in primary if r["name"].startswith("P3")), None)
    if level is None or p3 is None:
        decision = "incomplete"
    elif level >= 0.50 and p3["mean"] >= 0.15:
        decision = (f"C5RD implicit property {level:.3f} ≥ 0.50 with C5RD − C2RD {p3['mean']:+.3f} ≥ +0.15: the model learns to use composed "
                    "facts when the objective asks for it; stage 2 (T4) is to be queued with the RD objective")
    elif level >= 0.50:
        decision = f"C5RD implicit property {level:.3f} ≥ 0.50 but C5RD − C2RD {p3['mean']:+.3f} < +0.15: no pre-registered reading (memorisation not excluded); report the numbers"
    elif level >= 0.35:
        teacher = means["C5RD"]["property"].get("explicit-rel:oracle")
        decision = (f"partial: C5RD implicit property {level:.3f} in [0.35, 0.50); gap to the teacher (explicit recall "
                    f"{teacher:.3f}) {teacher - level:.3f}" if teacher is not None else f"partial: C5RD implicit property {level:.3f} in [0.35, 0.50)")
    else:
        decision = (f"C5RD implicit property {level:.3f} < 0.35: aligned objectives do not close the gap at this scale; the implicit route is closed "
                    "and the manuscript keeps the loss-domain result as the structural-use finding")
    out = {"stage": "1c", "host": host, "missing": missing, "implicit_condition": implicit, "splits_sha256": sha256_file(Path(args.splits)),
           "split_sizes": {k: len(v) for k, v in splits["splits"].items()}, "chance": chance, "primary": primary, "secondary": secondary,
           "read": reads, "losses_vs_C5": losses, "means_compositional": means, "implicit_property_compositional": implicit_level,
           "kl_ratio": {**ratio_summary, "per_seed": ratios}, "decision": decision}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text(render_report_1c(out))
    print(json.dumps({"decision": decision, "primary": [(r["name"], r["mean"]) for r in primary]}, indent=2))
    return out


def render_report_1c(out: dict[str, Any]) -> str:
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["# E15 stage 1c — aligned objectives (route distillation, fact verbalisation): T5 report (pre-registration "
             "`experiments/e15-hrr-target/preregistration.md`, \"Stage 1c\")", "",
             f"Host {out['host']}; splits sha256 `{out['splits_sha256']}`; split sizes {out['split_sizes']}.", "",
             "Arms: C5RD (C5 + route distillation from a teacher that reads the gold frame as explicit recall lines), C5FV (C5 + fact "
             "verbalisation, a token-space cloze on the frame's fillers), 3 seeds each; controls C2RD / C2FV (free table + the same "
             "objective, seed 1: can memorise seen terms, cannot generalise to compositional held-out words); baselines E9's C5 / C2 and "
             "stage 1's C5WR. Implicit route = `implicit:oracle` for compose channels, `none` for free tables.", ""]
    if out["missing"]:
        lines += ["**Missing evaluation folders:** " + ", ".join(out["missing"]), ""]
    lines += ["## Primary (Holm over the three)", "", "| contrast | difference [95% CI] | n |", "|---|---|---:|"]
    lines += [f"| {r['name']} | {fmt(r)} | {r['n']} |" for r in out["primary"]]
    lines += ["", "Implicit property on the compositional split (level, 3 seeds pooled where available): "
              + ", ".join(f"{arm} {f(v)}" for arm, v in out["implicit_property_compositional"].items()) + ".", "",
              f"**Decision rule reading:** {out['decision']}", ""]
    lines += ["## Means on the compositional items", "", "| arm | test | none | implicit:oracle | explicit-rel:oracle |", "|---|---|---:|---:|---:|"]
    for arm, by_test in out["means_compositional"].items():
        for test, by_cond in by_test.items():
            lines.append(f"| {arm} | {test} | {f(by_cond['none'])} | {f(by_cond['implicit:oracle'])} | {f(by_cond['explicit-rel:oracle'])} |")
    lines += ["", "## Teacher–student convergence (train rows: mean of the last 5% of rows over the mean of the first 5%)", "",
              "| arm | column | start | end | end / start | seeds |", "|---|---|---:|---:|---:|---:|"]
    for arm, r in out["kl_ratio"].items():
        if arm != "per_seed":
            lines.append(f"| {arm} | {'kl_loss' if 'RD' in arm else 'cloze_loss'} | {f(r['start'])} | {f(r['end'])} | {f(r['ratio'], 3)} | {r['seeds']} |")
    lines += ["", "## Read accuracy (post-hoc ridge probe fitted on seen terms; these arms train no `W`, so trained-map read accuracy does not apply)", "",
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


# -- report, stage 1d (500M tokens: the matched-length pair C5WR-xl / C5-xl) --------------------------------------------

def parse_seeds(text: str | None) -> tuple[int, ...]:
    """`--seeds`: stage 1d registers seed 1; seeds 2–3 are a conditional follow-up pooled by naming them."""
    seeds = tuple(int(t) for t in str(text or "").split(",") if t.strip())
    return seeds or (1,)


def extrapolated_crossing(level: float, increment: float, *, step_tokens: int, last_tokens: int, bar: float = STAGE1D_READ_BAR) -> float | None:
    """Tokens at which the linear continuation of the last increment (per `step_tokens`) reaches `bar` — a lower bound for
    a concave curve; None when the level is at the bar already or the increment is not positive."""
    if level >= bar or increment <= 0:
        return None
    return float(last_tokens + (bar - level) / increment * step_tokens)


def run_report_1d(args: argparse.Namespace) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    splits = load_splits(Path(args.splits))
    keep = {name: {r["concept"] for r in rows if r["kind"] == "new"} for name, rows in splits["splits"].items()}
    comp, strict, unseen = keep["compositional"], keep["compositional_strict"], keep["unseen_filler"]
    runs_root, base_root, qa_root, read_root = Path(args.runs_root), Path(args.baseline_root), Path(args.qa_root), Path(args.read_root)
    host, res, seed = args.host, args.resamples, args.seed
    checkpoints = parse_checkpoints(getattr(args, "checkpoints", None), STAGE1D_CHECKPOINTS)
    seeds = parse_seeds(getattr(args, "seeds", None))
    last, before = checkpoints[-1], (checkpoints[-2] if len(checkpoints) >= 2 else None)
    name = lambda arm, k: f"{host}-full-{arm}-s{k}"  # noqa: E731
    xl_read = {t: [read_root / name("C5WR-xl", k) / f"ckpt-{t}" for k in seeds] for t in checkpoints}
    xl_qa = {t: [qa_root / name("C5WR-xl", k) / f"ckpt-{t}" for k in seeds] for t in checkpoints}
    c5xl_read = {t: [read_root / name("C5-xl", k) / f"ckpt-{t}" for k in seeds] for t in checkpoints}
    c5xl_qa = {t: [qa_root / name("C5-xl", k) / f"ckpt-{t}" for k in seeds] for t in checkpoints}
    xl_runs, c5xl_runs = [runs_root / name("C5WR-xl", k) for k in seeds], [runs_root / name("C5-xl", k) for k in seeds]
    c5_qa, c5_runs = [qa_root / name("C5", k) for k in seeds], [base_root / name("C5", k) for k in seeds]
    long200_read = [read_root / name("C5WR-long", k) / "ckpt-200000000" for k in seeds]       # stage 1b's 200M (consistency)
    expected_qa = [f for fs in xl_qa.values() for f in fs] + [f for fs in c5xl_qa.values() for f in fs] + c5_qa
    expected_read = [f for fs in xl_read.values() for f in fs] + [f for fs in c5xl_read.values() for f in fs]
    missing = sorted([str(f) for f in expected_qa if not (f / "predictions.jsonl").exists()]
                     + [str(f) for f in expected_read if not (f / "read.json").exists()])
    chance = None
    for f in expected_read:
        if (f / "read.json").exists():
            chance = json.loads((f / "read.json").read_text())["chance"]; break
    implicit_c5 = next((implicit_condition(f) for f in c5_qa if (f / "summary.json").exists()), None)
    implicit_c5xl = next((implicit_condition(f) for fs in c5xl_qa.values() for f in fs if (f / "summary.json").exists()), None)

    def qa_contrast(a_folders, b_folders, a_cond, b_cond, test, items) -> dict[str, Any] | None:
        if a_cond is None or b_cond is None:
            return None
        r = pooled_contrast(a_folders, b_folders, a_cond, b_cond, test, items, resamples=res, seed=seed)
        return None if r is None else {"a_condition": a_cond, "b_condition": b_cond, "test": test, **r}

    conditions = ("none", "implicit:oracle", "explicit-rel:oracle")
    # The pair of curves: read accuracy per split (C5WR-xl's trained W, C5-xl's probe), the QA means of both arms, and at
    # every checkpoint the objective's share (C5WR-xl − C5-xl) and the continuity contrast (C5WR-xl − C5).
    curve: dict[int, dict[str, Any]] = {}
    for t in checkpoints:
        row: dict[str, Any] = {"read": {"C5WR-xl": {}, "C5-xl": {}}, "means": {}, "share": {}, "vs_C5": {}}
        for split in SPLITS:
            for key in ("trained", "probe"):
                values = read_curve({t: xl_read[t]}, split, key).get(t)
                if values:
                    row["read"]["C5WR-xl"][f"{split}/{key}"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
            values = read_curve({t: c5xl_read[t]}, split, "probe").get(t)
            if values:
                row["read"]["C5-xl"][f"{split}/probe"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
        for test in TESTS:
            r = qa_contrast(xl_qa[t], c5xl_qa[t], "implicit:oracle", implicit_c5xl, test, comp)
            if r is not None:
                row["share"][test] = r
            r = qa_contrast(xl_qa[t], c5_qa, "implicit:oracle", implicit_c5, test, comp)
            if r is not None:
                row["vs_C5"][test] = r
        row["means"] = {arm: {test: {cond: mean_value(folders[t], cond, test, comp) for cond in conditions} for test in TESTS}
                        for arm, folders in (("C5WR-xl", xl_qa), ("C5-xl", c5xl_qa))}
        curve[t] = row
    comp_curve = read_curve(xl_read, "compositional")

    primary: list[dict[str, Any]] = []
    p1 = qa_contrast(xl_qa[last], c5xl_qa[last], "implicit:oracle", implicit_c5xl, "property", comp)
    if p1 is not None:
        primary.append({"name": f"P1 implicit property, compositional, at {last:,} tokens: C5WR-xl − C5-xl (matched length; the objective's share)",
                        "checkpoint": last, **p1})
    p2 = bootstrap_mean(comp_curve.get(last, []), resamples=res, seed=seed, null=chance or 0.0)
    if p2 is not None:
        primary.append({"name": f"P2 compositional read accuracy at {last:,} tokens, C5WR-xl (difference to chance)", "checkpoint": last, **p2})
    p3 = None
    if before is not None and before in comp_curve and last in comp_curve and len(comp_curve[last]) == len(comp_curve[before]):
        p3 = {"from": before, "to": last, **paired_difference(np.asarray(comp_curve[last]), np.asarray(comp_curve[before]), resamples=res, seed=seed)}
        p3["still_rising"] = bool(p3["ci_low"] > 0)
        primary.append({"name": f"P3 compositional read-accuracy increment {before:,} → {last:,} tokens, C5WR-xl (paired per concept)", **p3})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p

    secondary: list[dict[str, Any]] = []
    r = qa_contrast(xl_qa[last], c5_qa, "implicit:oracle", implicit_c5, "property", comp)
    if r is not None:
        secondary.append({"name": f"implicit property, compositional at {last:,} tokens: C5WR-xl − C5 (50M; continuity with stage 1b)", **r})
    r = qa_contrast(c5xl_qa[last], c5_qa, implicit_c5xl, implicit_c5, "property", comp)
    if r is not None:
        secondary.append({"name": f"implicit property, compositional at {last:,} tokens: C5-xl − C5 (length alone)", **r})
    for items, label in ((strict, "strict compositional"), (unseen, "unseen-filler new words")):
        r = qa_contrast(xl_qa[last], c5xl_qa[last], "implicit:oracle", implicit_c5xl, "property", items)
        if r is not None:
            secondary.append({"name": f"implicit property, {label}, at {last:,} tokens: C5WR-xl − C5-xl", **r})
    for test in ("paraphrase", "entailment"):
        r = qa_contrast(xl_qa[last], c5xl_qa[last], "implicit:oracle", implicit_c5xl, test, comp)
        if r is not None:
            secondary.append({"name": f"implicit {test}, compositional at {last:,} tokens: C5WR-xl − C5-xl", **r})
    for split, label in (("compositional_strict", "strict"), ("unseen_filler", "unseen-filler")):
        for t in checkpoints:
            r = curve[t]["read"]["C5WR-xl"].get(f"{split}/trained")
            if r is not None:
                secondary.append({"name": f"read accuracy, {label}, C5WR-xl at {t:,} tokens (difference to chance)", **r})
    for t in checkpoints:
        r = curve[t]["read"]["C5-xl"].get("compositional/probe")
        if r is not None:
            secondary.append({"name": f"probe read accuracy, compositional, C5-xl at {t:,} tokens (difference to chance)", **r})
    a, b = read_values(xl_read[last], "compositional"), read_values(c5xl_read[last], "compositional", "probe")
    if a and b and len(a) == len(b):
        secondary.append({"name": f"read accuracy, compositional at {last:,} tokens: C5WR-xl (trained W) − C5-xl (probe), paired per concept",
                          **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed), "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    consistency = None
    if 200_000_000 in checkpoints:
        a, b = read_values(xl_read[200_000_000], "compositional"), read_values(long200_read, "compositional")
        if a and b and len(a) == len(b):
            consistency = {"checkpoint": 200_000_000, **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed),
                           "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))}
    losses: dict[str, dict[str, Any]] = {}
    for label, arm_runs, base_runs in (("C5WR-xl − C5-xl (matched length)", xl_runs, c5xl_runs), ("C5WR-xl − C5 (50M)", xl_runs, c5_runs),
                                       ("C5-xl − C5 (length alone)", c5xl_runs, c5_runs)):
        losses[label] = {}
        for stratum in ("after_heldout", "after", "unlinked", "all"):
            r = loss_contrast(arm_runs, base_runs, stratum, resamples=res, seed=seed)
            if r is not None:
                losses[label][stratum] = r

    # Decision rule (pre-registration, stage 1d): P2's level, P1's point estimate, P3's interval; four exhaustive cases.
    read_last = p2["mean_value"] if p2 else None
    share = p1["mean"] if p1 else None
    crossing = None
    if read_last is None or share is None or p3 is None:
        decision = "incomplete"
    elif read_last >= STAGE1D_READ_BAR and share >= STAGE1D_SHARE_BAR:
        decision = (f"read accuracy ≥ {STAGE1D_READ_BAR:.2f} at {last:,} tokens and the objective's share ≥ +{STAGE1D_SHARE_BAR:.2f}: the algebra is "
                    "learnable to the original standard and the matched-length gain is substantial; seeds 2–3 and stage 2 (T4) are to be queued")
    elif read_last >= STAGE1D_READ_BAR:
        decision = (f"read accuracy ≥ {STAGE1D_READ_BAR:.2f} at {last:,} tokens but the objective's share < +{STAGE1D_SHARE_BAR:.2f}: reads but the "
                    "matched-length use gain is small; no further length stages")
    elif p3["still_rising"]:
        crossing = extrapolated_crossing(read_last, p3["mean"], step_tokens=last - before, last_tokens=last)
        decision = (f"read accuracy < {STAGE1D_READ_BAR:.2f} and still rising (increment's CI excludes 0): extrapolated crossing of the bar at "
                    f"≈ {crossing / 1e6:,.0f}M tokens (linear in the last increment, a lower bound); stop, no stage 1e — length alone is not the programme's lever")
    else:
        decision = (f"read accuracy < {STAGE1D_READ_BAR:.2f} and the {before:,} → {last:,} increment's CI includes 0"
                    + (" (lies below it)" if p3["ci_high"] < 0 else "") + ": the curve has saturated below the bar; closed")
    out = {"stage": "1d", "host": host, "seeds": list(seeds), "checkpoints": checkpoints, "missing": missing,
           "implicit_condition": {"C5": implicit_c5, "C5-xl": implicit_c5xl}, "splits_sha256": sha256_file(Path(args.splits)),
           "split_sizes": {k: len(v) for k, v in splits["splits"].items()}, "chance": chance, "curve": curve, "primary": primary,
           "increment": p3, "secondary": secondary, "losses": losses, "consistency_200M_vs_stage1b": consistency,
           "extrapolated_crossing_tokens": crossing, "decision": decision}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text(render_report_1d(out))
    print(json.dumps({"decision": out["decision"], "primary": [(r["name"], r.get("mean")) for r in primary]}, indent=2))
    return out


def render_report_1d(out: dict[str, Any]) -> str:
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["# E15 stage 1d — 500M tokens, the matched-length pair C5WR-xl / C5-xl: T5 report (pre-registration "
             "`experiments/e15-hrr-target/preregistration.md`, \"Stage 1d\")", "",
             f"Host {out['host']}; seeds {out['seeds']}; checkpoints {[f'{t:,}' for t in out['checkpoints']]} tokens; splits sha256 "
             f"`{out['splits_sha256']}`; split sizes {out['split_sizes']}; chance {f(out['chance'], 5)}.", "",
             "C5WR-xl: stage 1b's C5 + WR recipe (λ_w 0.20, λ_r 0.035, layer 11) at 500M tokens; C5-xl: E9's C5 recipe at 500M, the "
             "matched-length control (probe read accuracy only: it trains no `W`). The pair of curves is the result; the objective's share "
             "is C5WR-xl − C5-xl at the same checkpoint.", ""]
    if out["missing"]:
        lines += ["**Missing evaluation folders:** " + ", ".join(out["missing"]), ""]
    lines += ["## Read accuracy over the checkpoints (per-concept; C5WR-xl trained `W`, C5-xl probe)", "",
              "| tokens | compositional | strict | unseen-filler | seen | heldout | C5-xl compositional (probe) | C5-xl unseen-filler (probe) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, row in out["curve"].items():
        rd = lambda arm, k: f((row["read"][arm].get(k) or {}).get("mean_value"))  # noqa: E731
        lines.append(f"| {int(t):,} | {rd('C5WR-xl', 'compositional/trained')} | {rd('C5WR-xl', 'compositional_strict/trained')} | "
                     f"{rd('C5WR-xl', 'unseen_filler/trained')} | {rd('C5WR-xl', 'seen/trained')} | {rd('C5WR-xl', 'heldout_ontology/trained')} | "
                     f"{rd('C5-xl', 'compositional/probe')} | {rd('C5-xl', 'unseen_filler/probe')} |")
    lines += ["", "## Implicit-route QA over the checkpoints (property, compositional split)", "",
              "| tokens | C5WR-xl − C5-xl (the objective's share) | C5WR-xl − C5 (50M) | C5WR-xl implicit | C5-xl implicit | C5WR-xl none | C5-xl none | C5WR-xl explicit | C5-xl explicit |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for t, row in out["curve"].items():
        m = lambda arm, cond: f(((row["means"].get(arm) or {}).get("property") or {}).get(cond))  # noqa: E731
        lines.append(f"| {int(t):,} | {fmt(row['share'].get('property'))} | {fmt(row['vs_C5'].get('property'))} | {m('C5WR-xl', 'implicit:oracle')} | "
                     f"{m('C5-xl', 'implicit:oracle')} | {m('C5WR-xl', 'none')} | {m('C5-xl', 'none')} | {m('C5WR-xl', 'explicit-rel:oracle')} | "
                     f"{m('C5-xl', 'explicit-rel:oracle')} |")
    lines += ["", "## Primary (Holm over the three)", "", "| endpoint | estimate [95% CI] | n |", "|---|---|---:|"]
    for r in out["primary"]:
        est = fmt(r) if "mean_value" not in r else f"{r['mean_value']:.4f}; to chance {fmt(r)}"
        lines.append(f"| {r['name']} | {est} | {r['n']} |")
    if out.get("increment"):
        i = out["increment"]
        lines += ["", f"The {i['from']:,} → {i['to']:,} increment: " + ("**still rising** (CI excludes 0)." if i["still_rising"] else "not distinguishable from flat (CI includes 0).")]
    lines += ["", f"**Decision rule reading:** {out['decision']}", ""]
    if out.get("extrapolated_crossing_tokens"):
        lines += [f"Extrapolated crossing of the 0.80 bar: ≈ {out['extrapolated_crossing_tokens'] / 1e6:,.0f}M tokens (the linear continuation of the "
                  "last increment; a lower bound for a concave curve; reported, not acted on).", ""]
    if out.get("consistency_200M_vs_stage1b"):
        c = out["consistency_200M_vs_stage1b"]
        lines += [f"Consistency check: C5WR-xl's {c['checkpoint']:,}-token checkpoint ({c['a_mean']:.4f}) against stage 1b's C5WR-long seed 1 at its "
                  f"200M checkpoint ({c['b_mean']:.4f}; same tokens, different lr schedule — the xl run is at 0.69 × peak there, the long run at the "
                  f"floor), paired per concept: {fmt(c)}.", ""]
    lines += ["## Loss endpoints (final evaluation, relative change, paired windows)", "",
              "| contrast | stratum | relative change [95% CI] |", "|---|---|---|"]
    for label, by in out["losses"].items():
        for stratum, r in by.items():
            lines.append(f"| {label} | {stratum} | {fmt(r, 'relative')} |")
    lines += ["", "## Secondary (unadjusted)", "", "| contrast | estimate [95% CI] | n |", "|---|---|---:|"]
    for r in out["secondary"]:
        est = fmt(r) if "mean_value" not in r else f"{r['mean_value']:.4f}; to chance {fmt(r)}"
        lines.append(f"| {r['name']} | {est} | {r['n']} |")
    return "\n".join(lines) + "\n"


# -- report, stage 1e (read, then inject) ------------------------------------------------------------------------------

def gate_of_run(run: Path, read_folder: Path | None = None) -> dict[str, Any] | None:
    """The learned gate of a stage-1e run at the end of training: from `read.json`'s `injection` record when the read
    evaluation exists, else from the last train row's `gate` column (`metrics.jsonl`); None without either."""
    if read_folder is not None and (read_folder / "read.json").exists():
        record = json.loads((read_folder / "read.json").read_text()).get("injection")
        if record is not None and record.get("gate") is not None:
            return {"gate": float(record["gate"]), "gate_bias": record.get("gate_bias"), "gated": record.get("gated"), "from": "read.json"}
    path = run / "metrics.jsonl"
    if path.exists():
        last = None
        for line in path.open():
            row = json.loads(line)
            if row.get("type") == "train" and "gate" in row:
                last = row
        if last is not None:
            return {"gate": float(last["gate"]), "step": last.get("step"), "from": "metrics.jsonl"}
    return None


def run_report_1e(args: argparse.Namespace) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    splits = load_splits(Path(args.splits))
    keep = {name: {r["concept"] for r in rows if r["kind"] == "new"} for name, rows in splits["splits"].items()}
    comp, strict, unseen = keep["compositional"], keep["compositional_strict"], keep["unseen_filler"]
    runs_root, base_root, qa_root, read_root = Path(args.runs_root), Path(args.baseline_root), Path(args.qa_root), Path(args.read_root)
    host, res, seed = args.host, args.resamples, args.seed
    name = lambda arm, k: f"{host}-full-{arm}-s{k}"  # noqa: E731
    seeds = {**STAGE1E_SEEDS, "C5": SEEDS, "C5WR": SEEDS, "C2R": (1,), "C5shWR": (1,)}
    qa = {arm: [qa_root / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    qa["C5WRI[gate closed]"] = [f / STAGE1E_GATE_CLOSED for f in qa["C5WRI"]]
    rd = {arm: [read_root / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    runs = {arm: [(base_root if arm in BASELINES else runs_root) / name(arm, k) for k in ks] for arm, ks in seeds.items()}
    missing = sorted([str(f) for arm in (*STAGE1E_ARMS, "C5WRI[gate closed]", "C5", "C5WR", "C2R", "C5shWR") for f in qa[arm]
                      if not (f / "predictions.jsonl").exists()]
                     + [str(f) for arm in (*STAGE1E_ARMS, "C5WR") for f in rd[arm] if not (f / "read.json").exists()])
    implicit = {arm: next((implicit_condition(f) for f in fs if (f / "summary.json").exists()), None) for arm, fs in qa.items()}

    def qa_contrast(a: str, b: str, test: str, items: set[str], *, a_cond: str | None = None, b_cond: str | None = None) -> dict[str, Any] | None:
        ca, cb = a_cond or implicit[a], b_cond or implicit[b]
        if ca is None or cb is None:
            return None
        fa, fb = qa[a], qa[b]
        if len(fa) != len(fb):                      # a 3-seed arm against a 1-seed arm: seed 1 against seed 1
            fa, fb = fa[:1], fb[:1]
        r = pooled_contrast(fa, fb, ca, cb, test, items, resamples=res, seed=seed)
        return None if r is None else {"a": a, "b": b, "a_condition": ca, "b_condition": cb, "test": test, **r}

    primary: list[dict[str, Any]] = []
    for label, row in (("P1 implicit property, compositional: C5WRI − C5WR (what injection adds to decoding)", qa_contrast("C5WRI", "C5WR", "property", comp)),
                       ("P2 implicit property, compositional: C5WRI − C2RI (structure against memorisation; seed 1)", qa_contrast("C5WRI", "C2RI", "property", comp)),
                       ("P3 implicit property, compositional: C5WRI − C5WRI[gate closed at evaluation] (the gain flows through the injected content)",
                        qa_contrast("C5WRI", "C5WRI[gate closed]", "property", comp))):
        if row is not None:
            primary.append({"name": label, **row})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    secondary: list[dict[str, Any]] = []
    for label, row in (
        ("C5WRI − C5, implicit property, compositional", qa_contrast("C5WRI", "C5", "property", comp)),
        ("C5shWRI − C5WRI, implicit property, compositional (content; seed 1)", qa_contrast("C5shWRI", "C5WRI", "property", comp)),
        ("C5shWRI − C5shWR, implicit property, compositional (seed 1)", qa_contrast("C5shWRI", "C5shWR", "property", comp)),
        ("C5WRI-nogate − C5WRI, implicit property, compositional (seed 1)", qa_contrast("C5WRI-nogate", "C5WRI", "property", comp)),
        ("C2RI − C2R, none property, compositional (seed 1)", qa_contrast("C2RI", "C2R", "property", comp, a_cond="none", b_cond="none")),
        ("C5WRI − C5WR, implicit paraphrase, compositional", qa_contrast("C5WRI", "C5WR", "paraphrase", comp)),
        ("C5WRI − C5WR, implicit entailment, compositional", qa_contrast("C5WRI", "C5WR", "entailment", comp)),
        ("C5WRI − C5WRI[gate closed], implicit paraphrase, compositional", qa_contrast("C5WRI", "C5WRI[gate closed]", "paraphrase", comp)),
        ("C5WRI − C5WRI[gate closed], implicit entailment, compositional", qa_contrast("C5WRI", "C5WRI[gate closed]", "entailment", comp)),
        ("C5WRI − C5WR, implicit property, strict compositional", qa_contrast("C5WRI", "C5WR", "property", strict)),
        ("C5WRI − C5WR, implicit property, unseen-filler new words", qa_contrast("C5WRI", "C5WR", "property", unseen)),
        ("C5WRI − C5WR, explicit-rel:oracle property, compositional (explicit recall unchanged?)",
         qa_contrast("C5WRI", "C5WR", "property", comp, a_cond="explicit-rel:oracle", b_cond="explicit-rel:oracle")),
        ("C5WRI − C5WR, none property, compositional", qa_contrast("C5WRI", "C5WR", "property", comp, a_cond="none", b_cond="none")),
        ("C5WRI[gate closed] − C5WR, implicit property, compositional (the trained model without its path)",
         qa_contrast("C5WRI[gate closed]", "C5WR", "property", comp)),
    ):
        if row is not None:
            secondary.append({"name": label, **row})
    chance = None
    for f in rd["C5WRI"] + rd["C5WR"]:
        if (f / "read.json").exists():
            chance = json.loads((f / "read.json").read_text())["chance"]; break
    reads: dict[str, dict[str, Any]] = {}
    for arm in (*STAGE1E_ARMS, "C5WR", "C2R", "C5shWR"):
        reads[arm] = {}
        for split in SPLITS:
            for key in ("trained", "probe"):
                values = read_values(rd[arm], split, key)
                if values:
                    reads[arm][f"{split}/{key}"] = bootstrap_mean(values, resamples=res, seed=seed, null=chance or 0.0)
    read_contrasts: list[dict[str, Any]] = []
    for label, a_arm, b_arm, split in (("read accuracy (trained W), compositional: C5WRI − C5WR (paired per concept and seed)", "C5WRI", "C5WR", "compositional"),
                                       ("read accuracy (trained W), unseen-filler: C5WRI − C5WR (paired per concept and seed)", "C5WRI", "C5WR", "unseen_filler"),
                                       ("read accuracy (trained W), seen: C5WRI − C5WR (paired per concept and seed)", "C5WRI", "C5WR", "seen"),
                                       ("read accuracy (trained W), compositional: C5WRI − C2RI (seed 1)", "C5WRI", "C2RI", "compositional"),
                                       ("read accuracy (trained W), compositional: C5WRI-nogate − C5WRI (seed 1)", "C5WRI-nogate", "C5WRI", "compositional")):
        fa, fb = rd[a_arm], rd[b_arm]
        if len(fa) != len(fb):
            fa, fb = fa[:1], fb[:1]
        a, b = read_values(fa, split, "trained"), read_values(fb, split, "trained")
        if a and b and len(a) == len(b):
            read_contrasts.append({"name": label, "split": split, **paired_difference(np.asarray(a), np.asarray(b), resamples=res, seed=seed),
                                   "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b))})
    secondary += read_contrasts
    losses: dict[str, dict[str, Any]] = {}
    for arm in STAGE1E_ARMS:
        for base in ("C5", "C5WR"):
            key = f"{arm} − {base}"
            losses[key] = {}
            for stratum in ("after_heldout", "after", "unlinked", "all"):
                r = loss_contrast(runs[arm], runs[base][:len(runs[arm])], stratum, resamples=res, seed=seed)
                if r is not None:
                    losses[key][stratum] = r
    for key, a_arm, b_arm in (("C5shWRI − C5WRI (seed 1)", "C5shWRI", "C5WRI"), ("C5WRI-nogate − C5WRI (seed 1)", "C5WRI-nogate", "C5WRI"),
                              ("C2RI − C2R (seed 1)", "C2RI", "C2R"), ("C5shWRI − C5shWR (seed 1)", "C5shWRI", "C5shWR")):
        losses[key] = {}
        for stratum in ("after_heldout", "after", "unlinked", "all"):
            r = loss_contrast(runs[a_arm][:1], runs[b_arm][:1], stratum, resamples=res, seed=seed)
            if r is not None:
                losses[key][stratum] = r
    losses = {k: v for k, v in losses.items() if v}
    gates = {arm: {f"s{k}": g for k, run, read in zip(seeds[arm], runs[arm], rd[arm]) if (g := gate_of_run(run, read)) is not None}
             for arm in STAGE1E_ARMS}
    conditions = ("none", "implicit:oracle", "explicit-rel:oracle")
    arms_shown = (*STAGE1E_ARMS, "C5WRI[gate closed]", "C5WR", "C2R", "C5shWR", "C5")
    means = {arm: {test: {cond: mean_value(qa[arm], cond, test, comp) for cond in conditions} for test in TESTS} for arm in arms_shown}
    implicit_level = {arm: mean_value(qa[arm], implicit[arm], "property", comp) if implicit[arm] else None for arm in arms_shown}
    # Decision rule (pre-registration, stage 1e): read on C5WRI's level, P2's point estimate and P3's interval (P3 > 0 = the
    # interval lies above 0; P3 ≈ 0 = the interval includes 0 or lies below it) and P1's sign.
    level = implicit_level.get("C5WRI")
    p1, p2, p3 = (next((r for r in primary if r["name"].startswith(tag)), None) for tag in ("P1", "P2", "P3"))
    if level is None or p1 is None or p2 is None or p3 is None:
        decision, path_used = "incomplete", None
    else:
        path_used = p3["ci_low"] > 0
        if not path_used or p1["mean"] <= 0:
            why = (f"P3 {p3['mean']:+.3f} [{p3['ci_low']:+.3f}, {p3['ci_high']:+.3f}] does not lie above 0" if not path_used
                   else f"C5WRI − C5WR {p1['mean']:+.3f} ≤ 0")
            decision = (f"{why}: the internal path is not used; stop adding use objectives at this scale (the remaining variable is "
                        "model size)")
        elif level >= STAGE1E_LEVEL_BAR and p2["mean"] >= STAGE1E_CONTROL_BAR:
            decision = (f"C5WRI implicit property {level:.3f} ≥ {STAGE1E_LEVEL_BAR:.2f}, C5WRI − C2RI {p2['mean']:+.3f} ≥ +{STAGE1E_CONTROL_BAR:.2f} and P3 "
                        f"{p3['mean']:+.3f} > 0: the model uses decoded facts through an internal read path; stage 2 (T4) is to be queued with WRI")
        elif level >= STAGE1E_LEVEL_BAR:
            decision = (f"C5WRI implicit property {level:.3f} ≥ {STAGE1E_LEVEL_BAR:.2f} with P3 > 0 but C5WRI − C2RI {p2['mean']:+.3f} < "
                        f"+{STAGE1E_CONTROL_BAR:.2f}: no pre-registered reading (memorisation not excluded); report the numbers")
        elif level >= STAGE1E_PARTIAL_BAR:
            decision = (f"partial: C5WRI implicit property {level:.3f} in [{STAGE1E_PARTIAL_BAR:.2f}, {STAGE1E_LEVEL_BAR:.2f}) with P3 {p3['mean']:+.3f} > 0 — the "
                        "gain flows through the path but is small; report, and a 200M run of WRI only on the author's wish (registered then)")
        else:
            decision = (f"C5WRI implicit property {level:.3f} < {STAGE1E_PARTIAL_BAR:.2f} with P3 {p3['mean']:+.3f} > 0 and C5WRI − C5WR {p1['mean']:+.3f} > 0: "
                        "no pre-registered reading (a real but small gain through the path); report the numbers")
    out = {"stage": "1e", "host": host, "missing": missing, "implicit_condition": implicit, "splits_sha256": sha256_file(Path(args.splits)),
           "split_sizes": {k: len(v) for k, v in splits["splits"].items()}, "chance": chance, "primary": primary, "secondary": secondary,
           "read": reads, "losses": losses, "gates": gates, "means_compositional": means, "implicit_property_compositional": implicit_level,
           "path_used": path_used, "decision": decision}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text(render_report_1e(out))
    print(json.dumps({"decision": decision, "primary": [(r["name"], r["mean"]) for r in primary]}, indent=2))
    return out


def render_report_1e(out: dict[str, Any]) -> str:
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = ["# E15 stage 1e — read, then inject: T5 report (pre-registration `experiments/e15-hrr-target/preregistration.md`, \"Stage 1e\")", "",
             f"Host {out['host']}; splits sha256 `{out['splits_sha256']}`; split sizes {out['split_sizes']}.", "",
             "Arms: C5WRI (C5 + write + read losses + the injection path: the fillers decoded from `W h` at layer 11, soft cleanup, summed over "
             "the frame's relations, projected and gated into the input of layer 12), 3 seeds; C5WRI-nogate (gate fixed open, exploratory), "
             "C2RI (free table + read loss + injection: the memorisation control) and C5shWRI (shuffled facts: decodes and injects wrong "
             "content), seed 1; baselines stage 1's C5WR / C2R / C5shWR and E9's C5. `C5WRI[gate closed]` is the trained C5WRI evaluated "
             "with its gate forced closed (`e14_reading recall --gate-closed`). Implicit route = `implicit:oracle` for compose channels, "
             "`none` for free tables.", ""]
    if out["missing"]:
        lines += ["**Missing evaluation folders:** " + ", ".join(out["missing"]), ""]
    lines += ["## Primary (Holm over the three)", "", "| contrast | difference [95% CI] | n |", "|---|---|---:|"]
    lines += [f"| {r['name']} | {fmt(r)} | {r['n']} |" for r in out["primary"]]
    lines += ["", "Implicit property on the compositional split (level; seeds pooled where available): "
              + ", ".join(f"{arm} {f(v)}" for arm, v in out["implicit_property_compositional"].items()) + ".", "",
              f"**Decision rule reading:** {out['decision']}", ""]
    lines += ["## Learned gate at the end of training (σ(b); initialised at σ(−4) = 0.018; `nogate`: fixed 1)", "",
              "| arm | seed | gate | source |", "|---|---|---:|---|"]
    for arm, by in out["gates"].items():
        for k, g in by.items():
            lines.append(f"| {arm} | {k} | {g['gate']:.4f} | {g['from']} |")
    lines += ["", "## Means on the compositional items", "", "| arm | test | none | implicit:oracle | explicit-rel:oracle |", "|---|---|---:|---:|---:|"]
    for arm, by_test in out["means_compositional"].items():
        for test, by_cond in by_test.items():
            lines.append(f"| {arm} | {test} | {f(by_cond['none'])} | {f(by_cond['implicit:oracle'])} | {f(by_cond['explicit-rel:oracle'])} |")
    lines += ["", "## Read accuracy (per-concept mean over seeds; `trained` = the run's W, `probe` = post-hoc ridge fitted on seen terms)", "",
              "| arm | split / map | accuracy | difference to chance [95% CI] | n |", "|---|---|---:|---|---:|"]
    for arm, by in out["read"].items():
        for key, r in by.items():
            lines.append(f"| {arm} | {key} | {r['mean_value']:.4f} | {fmt(r)} | {r['n']} |")
    lines += ["", "## Loss endpoints (final evaluation, relative change, paired windows summed over seeds)", "",
              "| contrast | stratum | relative change [95% CI] |", "|---|---|---|"]
    for key, by in out["losses"].items():
        for stratum, r in by.items():
            lines.append(f"| {key} | {stratum} | {fmt(r, 'relative')} |")
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
    r.add_argument("--gate-closed", action="store_true", help="stage 1e: switch the run's injection off (recorded; the read layer is pre-injection)")
    p = sub.add_parser("report", help="pool the arms' QA, read accuracy and loss endpoints")
    p.add_argument("--runs-root", type=Path, required=True); p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--qa-root", type=Path, required=True); p.add_argument("--read-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="SmolLM2-360M"); p.add_argument("--resamples", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", choices=("1", "1b", "1c", "1d", "1e"), default="1",
                   help="1b: the checkpoint curve of the long runs; 1c: the aligned objectives; 1d: the 500M matched-length pair; "
                        "1e: read, then inject")
    p.add_argument("--checkpoints", default=None, help="stages 1b/1d: token counts of the named checkpoints (default: the stage's registered ones)")
    p.add_argument("--seeds", default=None, help="stage 1d: seeds to pool (default 1; seeds 2–3 are the conditional follow-up)")
    args = parser.parse_args(argv)
    if args.command == "report" and args.stage == "1b":
        run_report_1b(args)
        return
    if args.command == "report" and args.stage == "1c":
        run_report_1c(args)
        return
    if args.command == "report" and args.stage == "1d":
        run_report_1d(args)
        return
    if args.command == "report" and args.stage == "1e":
        run_report_1e(args)
        return
    {"split": run_split, "read": run_read, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
