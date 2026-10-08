"""E9 binding and unbinding program, step 1: behavioural role items (decision 60; pre-registration
`experiments/e9-retrofit/preregistration-binding.md` §5). Evaluation only, on finished runs.

**Role-swap twins** (`kind=twins`, T5). Pairs of new words A and B (invented names, inserted at evaluation with the
dimension-3 new-word machinery: `e9_ontology_edit.inserted_entries`, `extended_adapter`) whose frames hold the same filler
multiset with two roles swapped: A = {(r1, X), (r2, Y), rest}, B = {(r1, Y), (r2, X), rest}. The relation pairs are the
templated relations of the track that share fillers (X and Y are both observed under r1 and under r2 in the ontology,
so either assignment is plausible) and that co-occur in real frames; `rest` is a real donor's frame (one with exactly
one r1 and one r2 filler, so the twins are of a type that carries both relations) whose category edges are kept and
whose other fillers are resampled frequency-weighted. An untyped bundle or a translation gives A and B the same vector
up to attention weights, so a model can tell them apart only through binding.

**Natural role-ambiguous items** (`kind=natural`, T4 and T5). Existing entries whose frame holds a filler F under r1
and a different filler G under r2 (F not under r2 and G not under r1 in that frame), with F also observed under r2 and G
under r1 elsewhere in the ontology (each is a plausible filler of the other's role); one such pair per anchor (seeded),
anchors from the `seen`, `rare` and `heldout` subsets.

**Items.** Per concept and relation: `choice` (the relation's two property prompts, candidates [X, Y] / [F, G] in a
fixed order, PMI against the null prompt with the term replaced by "this") and `cloze` (the relation's held-out wording,
the third paraphrase or the statement prefix; per-token log-probability, null-corrected the same way).

**Scores.**
- *Twin contrast* (twins; the pre-registered P1 measure on `choice`): for relation r and template t,
  `z = ±[(s_A(X) − s_A(Y)) − (s_B(X) − s_B(Y))]`, signed so that `z > 0` when the twins' preferences differ in the
  direction of their frames; the null prompt is the same for A and B, so it cancels. Accuracy = share of (relation,
  template) with `z > 0` (ties 0.5), per pair. A model that cannot tell the twins apart scores 0.5.
- *Role contrast* (natural): `z = [PMI_r1(F) − PMI_r1(G)] − [PMI_r2(F) − PMI_r2(G)]` per template index; > 0 when
  the model prefers F more under r1 than under r2. A role-blind model scores 0.5.
- *Item accuracy*: PMI argmax = gold, per item and template (chance 0.5).

**Sources.** `own`; `none` (the concepts' rows zeroed: what the host alone does); `swap` (twins, composing channels:
each twin reads its partner's frame — a model that follows its frames flips).

    python -m vsa_embed.experiments.e9_binding_items items --track t5 --kind twins --output experiments/e9-retrofit/items/role-twins-t5-smollm2-v1
    python -m vsa_embed.experiments.e9_binding_items items --track t4 --kind natural --output experiments/e9-retrofit/items/role-natural-t4-smollm2-v1
    python -m vsa_embed.experiments.e9_binding_items evaluate --run RUN --items DIR [--sources own,swap,none]
    python -m vsa_embed.experiments.e9_binding_items queue --stage t5 --items DIR --priority 50 [--models …] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml

from ..evaluation import channel_probes as cp
from . import e9_ontology_edit as edit
from . import e9_understanding as und
from .e5_common import E5Run, finish_output, json_ready, open_run, override_rows, start_output, write_json

SCHEMA = "e9-role-items/1"
KINDS = ("twins", "natural")
ITEM_KINDS = ("choice", "cloze")
SOURCES = ("own", "none", "swap")
ROOT = Path("experiments/e9-retrofit")
ITEMS_ROOT = ROOT / "items"
OUTPUT_PREFIX = "role-"
NATURAL_COUNTS = {"seen": 300, "rare": 300, "heldout": 600}
RESULT_FILES = ("summary.json", "predictions.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json")


# ---------------------------------------------------------------- shared builder pieces


def templated_relations(ctx: und.BuildContext) -> list[str]:
    """Relations with two property prompts and a held-out wording in the track lexicon."""
    lexicon = ctx.lexicon
    return [r for r in ctx.view.relation_names if lexicon.prompts("*", r) and lexicon.statements("*", r)]


def shared_fillers(ctx: und.BuildContext, r1: str, r2: str) -> list[int]:
    """Atomics observed under both relations, with a readable text (sorted ids)."""
    return sorted(a for a in ctx.pools.get(r1, {}) if a in ctx.pools.get(r2, {}) and ctx.text(a))


def relation_pairs(ctx: und.BuildContext, *, min_shared: int) -> list[tuple[str, str, int]]:
    """Templated relation pairs sharing ≥ `min_shared` fillers, most shared first: (r1, r2, shared count)."""
    names = templated_relations(ctx)
    pairs = []
    for i, r1 in enumerate(names):
        for r2 in names[i + 1:]:
            shared = len(shared_fillers(ctx, r1, r2))
            if shared >= min_shared:
                pairs.append((r1, r2, shared))
    return sorted(pairs, key=lambda p: (-p[2], p[0], p[1]))


def _choice_items(ctx: und.BuildContext, *, prefix: str, concept: str, relation: str, options: Sequence[int], gold: int,
                  meta: dict[str, Any]) -> list[dict[str, Any]]:
    """The `choice` and `cloze` items of one (concept, relation): candidates in the order of `options`."""
    lexicon = ctx.lexicon
    texts = [ctx.text(a) for a in options]
    base = {"concept": concept, "relation": relation, "gold": int(gold), "null": lexicon.null_surface, "meta": meta}
    return [{"id": f"{prefix}-{relation}-choice", "kind": "choice", "templates": lexicon.prompts("*", relation),
             "candidates": [lexicon.answer(relation, t) for t in texts], **base},
            {"id": f"{prefix}-{relation}-cloze", "kind": "cloze", "templates": lexicon.statements("*", relation),
             "candidates": [lexicon.statement_answer(relation, t) for t in texts], **base}]


def _write(output: Path, manifest: dict[str, Any], concepts: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "concepts.jsonl").write_text("".join(json.dumps(json_ready(c)) + "\n" for c in concepts))
    und.write_jsonl_gz(output / "items.jsonl.gz", items)
    manifest = {"schema": SCHEMA, **manifest, "counts": {"concepts": len(concepts), "items": len(items),
                                                         **dict(sorted(Counter(i["kind"] for i in items).items()))}}
    manifest["sha256"] = {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in ("concepts.jsonl", "items.jsonl.gz")}
    write_json(output / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- role-swap twins


def invented_names(ctx: und.BuildContext, count: int, *, name_seed: int, contamination_texts: Sequence[str],
                   reserved: set[str], wordnet: Any = None) -> tuple[list[str], dict[str, Any]]:
    """`count` invented names (the dimension-3 generator: not a WordNet lemma, not an alias word, not one token, ≥ ℓ_min
    subtokens, absent from the contamination text), reserved names removed."""
    from . import e5_zeroshot as zs
    if wordnet is None:
        from nltk.corpus import wordnet
    reserved = {n.lower() for n in reserved}
    # the generator is sequential: the first k names of a longer draw are the names of a shorter one
    names, checks = zs.invent_names(list(range(count + 64 + len(reserved))), ctx.tokenizer, ctx.table, wordnet,
                                    min_subtokens=ctx.min_subtokens, seed=name_seed, contamination_texts=contamination_texts)
    surfaces = list(dict.fromkeys(n for _, n in sorted(names.items()) if n.lower() not in reserved))[:count]
    if len(surfaces) < count or len({s.lower() for s in surfaces}) != count:
        raise ValueError("not enough unique invented names after removing reserved ones")
    checks.update(reserved_names=len(reserved), reserved_clashes=sum(n.lower() in reserved for n in names.values()))
    return surfaces, checks


def build_twins(ctx: und.BuildContext, output: Path, *, count: int = 300, seed: int = 0, name_seed: int = 23, min_shared: int = 20,
                contamination_texts: Sequence[str] = (), reserved_names: set[str] | None = None, wordnet: Any = None,
                pairs: Sequence[tuple[str, str]] | None = None, exclude_frames: Sequence[Sequence[tuple[int, int]]] = (),
                exclude_fillers: Sequence[tuple[int, int]] = ()) -> dict[str, Any]:
    """Role-swap twin pairs (module docstring); `count` pairs split evenly over the usable relation pairs. Opt-in (E12 3b's
    training twins): no twin frame equals one of `exclude_frames` ((relation, atomic) id pairs) and no pair swaps an
    unordered filler pair {X, Y} of `exclude_fillers`; with both empty the draw is unchanged."""
    view, lexicon = ctx.view, ctx.lexicon
    rng = random.Random(seed)
    found = relation_pairs(ctx, min_shared=min_shared)
    if pairs is not None:
        wanted = {tuple(p) for p in pairs}
        found = [p for p in found if (p[0], p[1]) in wanted]
    kept_ids = {view.relation_id[r] for r in lexicon.kept_relations if r in view.relation_id}
    single = [e for e in range(view.entry_count) if len(ctx.table.entry_concepts[e]) == 1 and e not in view.heldout
              and e not in ctx.synthetic]
    usable, donors = [], {}
    for r1, r2, shared in found:
        a, b = view.relation_id[r1], view.relation_id[r2]
        members = [e for e in single if sum(r == a for r, _ in view.frame(e)) == 1 and sum(r == b for r, _ in view.frame(e)) == 1]
        if members:
            usable.append((r1, r2, shared)); donors[(r1, r2)] = members
    if not usable:
        raise ValueError("no relation pair shares enough fillers and has a donor frame")
    quota = [count // len(usable) + (1 if k < count % len(usable) else 0) for k in range(len(usable))]
    frames_seen = {frozenset(view.frame(e)) for e in range(view.entry_count)}
    frames_seen |= {frozenset((int(r), int(f)) for r, f in frame) for frame in exclude_frames}
    banned = {frozenset((int(x), int(y))) for x, y in exclude_fillers}
    pools = {r: ctx.pools[view.relation_names[r]] for r in range(len(view.relation_names)) if view.relation_names[r] in ctx.pools}
    built = []
    for (r1, r2, _), n in zip(usable, quota):
        a, b = view.relation_id[r1], view.relation_id[r2]
        shared = shared_fillers(ctx, r1, r2)
        made, tries = 0, 0
        while made < n:
            tries += 1
            if tries > 200 * n:
                raise ValueError(f"could not build {n} twin pairs for ({r1}, {r2})")
            donor = rng.choice(donors[(r1, r2)])
            x, y = rng.sample(shared, 2)
            if ctx.text(x) == ctx.text(y) or frozenset((x, y)) in banned:
                continue
            rest = []
            used = {x, y}
            ok = True
            for r, f in view.frame(donor):
                if r in (a, b):
                    continue
                if r not in kept_ids:
                    choice = edit._weighted_choice(rng, pools[r], used | {f})
                    if choice is None:
                        ok = False; break
                    f = choice
                if f in used:
                    ok = False; break
                rest.append((r, f)); used.add(f)
            if not ok:
                continue
            twin_a, twin_b = rest + [(a, x), (b, y)], rest + [(a, y), (b, x)]
            if frozenset(twin_a) in frames_seen or frozenset(twin_b) in frames_seen:
                continue
            frames_seen.update({frozenset(twin_a), frozenset(twin_b)})
            built.append({"r1": r1, "r2": r2, "x": x, "y": y, "donor": donor, "A": twin_a, "B": twin_b})
            made += 1
    reserved = set(reserved_names or ())
    names, checks = invented_names(ctx, 2 * len(built), name_seed=name_seed, contamination_texts=contamination_texts,
                                   reserved=reserved, wordnet=wordnet)
    concepts, items = [], []
    for p, spec in enumerate(built):
        for k, twin in enumerate(("A", "B")):
            cid, partner = f"tw-{p:04d}-{twin}", f"tw-{p:04d}-{'B' if twin == 'A' else 'A'}"
            concepts.append({"concept": cid, "surface": names[2 * p + k], "entry": None, "subset": "twins", "pair": p, "twin": twin,
                             "partner": partner, "relations": [spec["r1"], spec["r2"]],
                             "fillers": {"X": view.atomic_names[spec["x"]], "Y": view.atomic_names[spec["y"]]},
                             "frame": view.readable(spec[twin]), "donor_entry": spec["donor"], "degree": len(spec[twin])})
            for role, relation in (("r1", spec["r1"]), ("r2", spec["r2"])):
                mine = spec["x"] if (role == "r1") == (twin == "A") else spec["y"]
                meta = {"pair": p, "twin": twin, "role": role, "pair_relations": f"{spec['r1']}|{spec['r2']}"}
                items += _choice_items(ctx, prefix=cid, concept=cid, relation=relation, options=[spec["x"], spec["y"]],
                                       gold=0 if mine == spec["x"] else 1, meta=meta)
    manifest = {"kind": "twins", "track": ctx.track, "family": ctx.family, "tokenizer": ctx.tokenizer_name, "seed": seed,
                "name_seed": name_seed, "pairs": len(built), "ontology": str(ctx.ontology_path),
                "ontology_alias_sha256": ctx.table.digest(), "alias_table": str(ctx.alias_table_path), "name_checks": checks,
                "relation_pairs": [{"r1": r1, "r2": r2, "shared_fillers": s, "donors": len(donors[(r1, r2)]), "pairs": q}
                                   for (r1, r2, s), q in zip(usable, quota)],
                "rules": {"twins": "A = rest + (r1, X) + (r2, Y), B = rest + (r1, Y) + (r2, X); X ≠ Y drawn uniformly from the "
                                   "fillers observed under both r1 and r2 (distinct texts); rest = a donor frame with exactly one "
                                   "r1 and one r2 filler (not held out, not synthetic), category edges kept, other fillers "
                                   "resampled frequency-weighted (never X or Y); every frame new",
                          "candidates": "[X, Y] in this order for both twins and both relations",
                          "scoring": "choice: PMI against the null prompt; cloze: per-token log-probability minus the null "
                                     "prompt's; twin contrast z = ±[(s_A(X) − s_A(Y)) − (s_B(X) − s_B(Y))]"}}
    if exclude_frames or exclude_fillers:              # opt-in record (absent from every earlier build)
        manifest["excluded"] = {"frames": len({frozenset(map(tuple, f)) for f in exclude_frames}), "filler_pairs": len(banned)}
    return _write(output, manifest, concepts, items)


# ---------------------------------------------------------------- natural role-ambiguous items


def natural_pairs(ctx: und.BuildContext, entry: int, relations: Sequence[str]) -> list[tuple[str, int, str, int]]:
    """(r1, F, r2, G) of an entry's frame: r1, r2 templated, F a filler of r1 and G of r2 here, F ≠ G with distinct texts,
    F not a filler of r2 and G not of r1 in this frame (so each forced choice has one correct option), F observed under r2
    and G under r1 elsewhere in the ontology."""
    by_rel = ctx.frame_by_relation(ctx.view.frame(entry))
    present = [r for r in relations if by_rel.get(r)]
    out = []
    for i, r1 in enumerate(present):
        for r2 in present[i + 1:]:
            for f in by_rel[r1]:
                for g in by_rel[r2]:
                    if f == g or f in by_rel[r2] or g in by_rel[r1]:
                        continue
                    if not ctx.text(f) or not ctx.text(g) or ctx.text(f) == ctx.text(g):
                        continue
                    if f in ctx.pools.get(r2, {}) and g in ctx.pools.get(r1, {}):
                        out.append((r1, f, r2, g))
    return out


def build_natural(ctx: und.BuildContext, output: Path, *, counts: dict[str, int] | None = None, seed: int = 0) -> dict[str, Any]:
    """Natural role-ambiguous items (module docstring)."""
    counts = {**NATURAL_COUNTS, **(counts or {})}
    view = ctx.view
    rng = random.Random(seed)
    relations = templated_relations(ctx)
    frequency = view.frequency if view.frequency is not None else np.zeros(view.entry_count)
    pools: dict[str, list[int]] = defaultdict(list)
    candidates: dict[int, list[tuple[str, int, str, int]]] = {}
    for e, info in sorted(ctx.surfaces.items()):
        if not info["linkable"]:
            continue
        found = natural_pairs(ctx, e, relations)
        if not found:
            continue
        candidates[e] = found
        subset = "heldout" if e in view.heldout else "seen" if frequency[e] >= 10 else "rare" if frequency[e] >= 1 else None
        if subset:
            pools[subset].append(e)
    concepts, items = [], []
    for subset in ("seen", "rare", "heldout"):
        chosen = sorted(rng.sample(pools[subset], min(counts.get(subset, len(pools[subset])), len(pools[subset]))))
        for e in chosen:
            r1, f, r2, g = rng.choice(candidates[e])
            cid = f"rn-{ctx.track}-{e}"
            concepts.append({"concept": cid, "surface": ctx.shown(e), "entry": int(e), "subset": subset,
                             "relations": [r1, r2], "fillers": {"F": view.atomic_names[f], "G": view.atomic_names[g]},
                             "frequency": int(frequency[e]), "degree": len(view.frame(e)),
                             "split": "synthetic" if e in ctx.synthetic else subset})
            for role, relation, gold in (("r1", r1, 0), ("r2", r2, 1)):
                meta = {"role": role, "pair_relations": f"{r1}|{r2}", "subset": subset}
                items += _choice_items(ctx, prefix=cid, concept=cid, relation=relation, options=[f, g], gold=gold, meta=meta)
    manifest = {"kind": "natural", "track": ctx.track, "family": ctx.family, "tokenizer": ctx.tokenizer_name, "seed": seed,
                "ontology": str(ctx.ontology_path), "ontology_alias_sha256": ctx.table.digest(), "alias_table": str(ctx.alias_table_path),
                "counts_requested": counts, "anchors": dict(Counter(c["subset"] for c in concepts)),
                "eligible": {s: len(v) for s, v in pools.items()},
                "relation_pairs": dict(Counter(f"{c['relations'][0]}|{c['relations'][1]}" for c in concepts).most_common()),
                "rules": {"anchors": "single-concept entries whose shown surface links (≥ ℓ_min subtokens) with a frame pair (r1, F), "
                                     "(r2, G): templated, F ≠ G (distinct texts), F not a filler of r2 and G not of r1 in the frame, F observed under r2 and G under r1 "
                                     "elsewhere; one pair per anchor (seeded); seeded sample per subset",
                          "candidates": "[F, G] in this order; gold F for r1, G for r2",
                          "scoring": "role contrast z = [PMI_r1(F) − PMI_r1(G)] − [PMI_r2(F) − PMI_r2(G)] per template index"}}
    return _write(output, manifest, concepts, items)


def build_items(track: str, kind: str, output: Path, *, family: str = "smollm2", count: int = 300, seed: int = 0,
                name_seed: int = 23, counts: dict[str, int] | None = None, contamination_tokens: int = 20_000_000) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ctx = und.BuildContext.for_track(track, family)
    if kind == "natural":
        return build_natural(ctx, output, counts=counts, seed=seed)
    from .e9_tracks import track_spec
    spec = track_spec(track, family)
    texts = edit._contamination_texts([spec.data_root / "train"], ctx.tokenizer_name, contamination_tokens)
    reserved = set()                         # the track's dimension-3 new words (every family) and its synthetic terms;
    for path in sorted(ITEMS_ROOT.glob(f"new-words-{track}-*")):          # twins of another family reuse the same names
        reserved |= edit._reserved_names([path])
    if spec.items_dir is not None and (spec.items_dir / "synthetic_concepts.jsonl").exists():
        reserved |= {a.lower() for line in (spec.items_dir / "synthetic_concepts.jsonl").read_text().splitlines() if line.strip()
                     for a in json.loads(line).get("aliases", [])}
    return build_twins(ctx, output, count=count, seed=seed, name_seed=name_seed, contamination_texts=texts, reserved_names=reserved)


def load_items(items_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    items_dir = Path(items_dir)
    manifest = json.loads((items_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{items_dir} is not an {SCHEMA} item directory")
    concepts = und.read_jsonl(items_dir / "concepts.jsonl")
    items = und.read_jsonl(items_dir / "items.jsonl")
    known = {c["concept"] for c in concepts}
    for item in items:
        if item["concept"] not in known or not 0 <= int(item["gold"]) < len(item["candidates"]) or not item["templates"]:
            raise ValueError(f"item {item['id']} is malformed")
    return manifest, concepts, items


# ---------------------------------------------------------------- scoring


@torch.no_grad()
def continuation_scores(adapter: cp.ModelAdapter, prefixes: Sequence[str], continuations: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(Σ log p, token count) of each continuation after its prefix: `e9_understanding.continuation_logprob` (the same
    batches, spans and autocast, output head in float32) with the number of continuation tokens."""
    from torch.nn import functional as F
    texts = [p + c for p, c in zip(prefixes, continuations)]
    sums, counts = np.zeros(len(texts)), np.zeros(len(texts))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), adapter.batch_size):
        chunk = order[start:start + adapter.batch_size]
        encoded, out, head = adapter._run_batch([texts[i] for i in chunk])
        hidden = out.hidden_states[-1]
        mask = encoded["attention_mask"]
        rows, cols, targets, owners = [], [], [], []
        for r, i in enumerate(chunk):
            n = int(mask[r].sum())
            offsets = encoded["offset_mapping"][r, :n].tolist()
            for t in range(1, n):
                if offsets[t][1] > len(prefixes[i]):
                    rows.append(r); cols.append(t - 1); targets.append(int(encoded["input_ids"][r, t])); owners.append(i)
        if not rows:
            continue
        device = hidden.device
        with torch.autocast(device.type, enabled=False):
            selected = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)].float()
            logits = F.linear(selected, head.weight.float(), None if head.bias is None else head.bias.float())
            picked = torch.log_softmax(logits, -1).gather(-1, torch.tensor(targets, device=device)[:, None]).squeeze(-1)
        np.add.at(sums, owners, picked.cpu().double().numpy())
        np.add.at(counts, owners, 1.0)
    return sums, counts


def score_items(adapter: cp.ModelAdapter, items: Sequence[dict[str, Any]], surfaces: dict[str, str],
                null_cache: dict[tuple[str, str], tuple[float, float]] | None = None
                ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[float, float]]]:
    """Per item and template × candidate: the real and null (term → the lexicon's null surface) summed log-probabilities
    and token counts; `s` = the comparison score (choice: Σ log p; cloze: per-token log p), `pmi` = s − s_null."""
    real, null, shapes = [], [], []
    for item in items:
        surface = surfaces[item["concept"]]
        for template in item["templates"]:
            for candidate in item["candidates"]:
                real.append((und.render(template, {"x": surface}), candidate))
                null.append((und.render(template, {"x": item["null"]}), candidate))
        shapes.append((len(item["templates"]), len(item["candidates"])))
    sums, counts = continuation_scores(adapter, [p for p, _ in real], [c for _, c in real]) if real else (np.zeros(0), np.zeros(0))
    cache = dict(null_cache or {})
    missing = sorted(set(null) - set(cache))
    if missing:
        ms, mc = continuation_scores(adapter, [p for p, _ in missing], [c for _, c in missing])
        cache.update(zip(missing, zip(ms.tolist(), mc.tolist())))
    null_sums = np.asarray([cache[pair][0] for pair in null]); null_counts = np.asarray([cache[pair][1] for pair in null])
    out, cursor = [], 0
    for item, (n_t, k) in zip(items, shapes):
        span = slice(cursor, cursor + n_t * k)
        cursor += n_t * k
        per_token = item["kind"] == "cloze"
        s = (sums[span] / np.maximum(counts[span], 1) if per_token else sums[span]).reshape(n_t, k)
        s0 = (null_sums[span] / np.maximum(null_counts[span], 1) if per_token else null_sums[span]).reshape(n_t, k)
        pmi = s - s0
        gold = int(item["gold"])
        correct = [(1.0 if np.argmax(row) == gold else 0.0) if np.ptp(row) > 0 else 0.5 for row in pmi]
        out.append({"id": item["id"], "kind": item["kind"], "concept": item["concept"], "relation": item["relation"], "gold": gold,
                    "s": s.round(5).tolist(), "pmi": pmi.round(5).tolist(), "correct": correct, "meta": item.get("meta", {})})
    return out, cache


def _sign_accuracy(z: Sequence[float]) -> float:
    z = np.asarray(z, dtype=float)
    return float(np.mean(np.where(z > 0, 1.0, np.where(z < 0, 0.0, 0.5))))


def twin_scores(rows: Sequence[dict[str, Any]], kind: str = "choice") -> dict[int, dict[str, float]]:
    """Per twin pair: twin-contrast accuracy (`contrast`), mean contrast (`margin`) and item accuracy (`item`) over its
    two relations × templates of item kind `kind`."""
    by_key = {(r["meta"]["pair"], r["meta"]["twin"], r["relation"]): r for r in rows if r["kind"] == kind}
    out: dict[int, dict[str, float]] = {}
    pairs = sorted({k[0] for k in by_key})
    for p in pairs:
        zs, items = [], []
        for (q, twin, relation), row in by_key.items():
            if q != p or twin != "A":
                continue
            partner = by_key.get((p, "B", relation))
            if partner is None:
                continue
            sign = 1.0 if row["gold"] == 0 else -1.0                    # A's filler under this relation is X (index 0)
            for sa, sb in zip(row["s"], partner["s"]):
                zs.append(sign * ((sa[0] - sa[1]) - (sb[0] - sb[1])))
            items += row["correct"] + partner["correct"]
        if zs:
            out[p] = {"contrast": _sign_accuracy(zs), "margin": float(np.mean(zs)), "item": float(np.mean(items))}
    return out


def natural_scores(rows: Sequence[dict[str, Any]], kind: str = "choice") -> dict[str, dict[str, float]]:
    """Per anchor: role-contrast accuracy (`contrast`), mean contrast (`margin`) and item accuracy (`item`)."""
    by_concept: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["kind"] == kind:
            by_concept[r["concept"]][r["meta"]["role"]] = r
    out = {}
    for concept, roles in by_concept.items():
        if set(roles) != {"r1", "r2"}:
            continue
        first, second = roles["r1"], roles["r2"]
        zs = [(a[0] - a[1]) - (b[0] - b[1]) for a, b in zip(first["pmi"], second["pmi"])]
        out[concept] = {"contrast": _sign_accuracy(zs), "margin": float(np.mean(zs)),
                        "item": float(np.mean(first["correct"] + second["correct"]))}
    return out


# ---------------------------------------------------------------- evaluation


def available_sources(run: E5Run, requested: Sequence[str] | None, kind: str) -> list[str]:
    mode = run.mode
    allowed = ["own"] if mode in {"none", "hashed"} else ["own", "none"] + (["swap"] if mode == "compose" and kind == "twins" else [])
    return [s for s in (requested or SOURCES) if s in allowed]


def evaluate(run: E5Run, items_dir: Path, *, sources: Sequence[str] | None = None, seed: int = 0, limit: int | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score the item directory under each source (module docstring). `limit` (smoke tests only): the first N concepts'
    items (twins: N pairs)."""
    manifest, concepts, items = load_items(items_dir)
    kind = manifest["kind"]
    if limit:
        keep_ids = ({c["concept"] for c in concepts if c["pair"] < limit} if kind == "twins"
                    else {c["concept"] for c in concepts[:limit]})
        concepts = [c for c in concepts if c["concept"] in keep_ids]
        items = [i for i in items if i["concept"] in keep_ids]
    ontology = run.ontology
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    channel = run.channel
    if kind == "twins":
        base = int(ontology["entry_count"])
        ids = {c["concept"]: base + i for i, c in enumerate(concepts)}
        frames = [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in concepts]
        index = {c["concept"]: i for i, c in enumerate(concepts)}
        swapped = [frames[index[c["partner"]]] for c in concepts]
        adapter = edit.extended_adapter(run, {c["surface"]: ids[c["concept"]] for c in concepts})
    else:
        ids = {c["concept"]: int(c["entry"]) for c in concepts}
        frames = swapped = None
        adapter = run.adapter
    resolved = edit.link_check(adapter, concepts, items, ids)
    sources = available_sources(run, sources, kind)
    started = time.monotonic()
    results: dict[str, list[dict[str, Any]]] = {}
    cache = None
    for source in sources:
        log(f"  role items ({kind}): source {source}")
        zero = None
        if source == "none" and channel is not None:
            width = channel.gate.in_features // 2
            zero = {e: torch.zeros(width) for e in ids.values()}
        inserted = (edit.inserted_entries(channel, len(concepts), swapped if source == "swap" else frames, seed=seed)
                    if kind == "twins" else contextlib.nullcontext())
        with inserted, (override_rows(channel, zero) if zero else contextlib.nullcontext()):
            rows, cache = score_items(adapter, items, surfaces, cache)
        results[source] = rows
    return {"manifest": manifest, "kind": kind, "resolved": resolved, "sources": sources, "results": results,
            "seconds": time.monotonic() - started}


def unit_scores(evaluation: dict[str, Any], source: str = "own", kind: str = "choice") -> dict[Any, dict[str, float]]:
    """Per unit (twins: pair; natural: anchor concept) the scores of one source and item kind, linked units only (a twin
    pair needs both twins linked)."""
    rows = evaluation["results"].get(source)
    if rows is None:
        return {}
    linked = {c for c, r in evaluation["resolved"].items() if r["status"] == "linked"}
    rows = [r for r in rows if r["concept"] in linked]
    if evaluation["kind"] == "twins":
        units = twin_scores(rows, kind)
        complete = {int(c.split("-")[1]) for c in linked if f"{c[:-1]}{'B' if c.endswith('A') else 'A'}" in linked}
        return {p: v for p, v in units.items() if p in complete}
    return natural_scores(rows, kind)


def summarize(evaluation: dict[str, Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per source × item kind × subset (twins: relation pair; natural: seen / rare / heldout): mean contrast accuracy,
    margin and item accuracy over units, with a bootstrap interval of the contrast accuracy against 0.5; and `own` −
    other sources (paired over units)."""
    from . import e5_zeroshot as zs
    kind = evaluation["kind"]
    concepts = evaluation["manifest"]
    group_of: dict[Any, str] = {}
    for row in next(iter(evaluation["results"].values()), []):
        key = row["meta"]["pair"] if kind == "twins" else row["concept"]
        group_of[key] = row["meta"]["pair_relations"] if kind == "twins" else row["meta"]["subset"]
    out: dict[str, Any] = {"kind": kind, "units": {}, "sources": {}, "comparisons": [],
                           "linked": sum(r["status"] == "linked" for r in evaluation["resolved"].values()),
                           "concepts": len(evaluation["resolved"])}
    for source in evaluation["sources"]:
        block = {}
        for item_kind in ITEM_KINDS:
            units = unit_scores(evaluation, source, item_kind)
            groups = defaultdict(list)
            for key, v in units.items():
                groups["all"].append(v); groups[group_of.get(key, "?")].append(v)
            block[item_kind] = {}
            for group, values in sorted(groups.items()):
                contrast = np.asarray([v["contrast"] for v in values])
                ci = zs.paired_difference(contrast, np.full_like(contrast, 0.5), resamples=resamples, seed=seed) if contrast.size > 1 else None
                block[item_kind][group] = {"units": len(values), "contrast": float(contrast.mean()),
                                           "margin": float(np.mean([v["margin"] for v in values])),
                                           "item": float(np.mean([v["item"] for v in values])),
                                           "contrast_minus_chance": ci}
        out["sources"][source] = block
    for source in evaluation["sources"]:
        if source == "own":
            continue
        for item_kind in ITEM_KINDS:
            a, b = unit_scores(evaluation, "own", item_kind), unit_scores(evaluation, source, item_kind)
            common = sorted(set(a) & set(b), key=str)
            if len(common) < 2:
                continue
            ci = zs.paired_difference(np.asarray([a[k]["contrast"] for k in common]), np.asarray([b[k]["contrast"] for k in common]),
                                      resamples=resamples, seed=seed)
            out["comparisons"].append({"kind": item_kind, "baseline": source, **ci})
    return out


def render_report(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# Role items ({summary['kind']}) — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Items `{header['items']}`: {summary['linked']} of {summary['concepts']} concepts link. `contrast` = share of "
             "(relation, template) whose twin / role contrast has the frame's sign (chance 0.5); `item` = PMI-argmax accuracy.", ""]
    for name, block in summary["sources"].items():
        lines += [f"## source `{name}`", "", "| kind | group | units | contrast | margin | item |", "|---|---|---:|---:|---:|---:|"]
        for item_kind, groups in block.items():
            for group, v in groups.items():
                lines.append(f"| {item_kind} | {group} | {v['units']} | {v['contrast']:.3f} | {v['margin']:+.3f} | {v['item']:.3f} |")
        lines.append("")
    if summary["comparisons"]:
        lines += ["## own − source (contrast accuracy, paired over units)", "", "| kind | baseline | difference [95% CI] | n |",
                  "|---|---|---|---:|"]
        lines += [f"| {c['kind']} | {c['baseline']} | {c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['n']} |"
                  for c in summary["comparisons"]]
    return "\n".join(lines) + "\n"


def output_folder(run_dir: Path, items: Path | str) -> Path:
    """`RUN/<items directory name>` (the names start with `role-`)."""
    name = Path(items).name
    return Path(run_dir) / (name if name.startswith(OUTPUT_PREFIX) else f"{OUTPUT_PREFIX}{name}")


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    manifest = json.loads((Path(args.items) / "manifest.json").read_text())
    requested = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    output = Path(args.output or output_folder(args.run, args.items))
    alias_table = args.alias_table or ensure_alias_table(track_spec(manifest["track"], manifest["family"]))
    config = {"experiment": "e9-role-items", "run": str(args.run), "items": str(args.items), "sources": requested, "seed": args.seed,
              "limit": args.limit, "alias_table": str(alias_table) if alias_table else None, "batch_size": args.batch_size}
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    run = open_run(args.run, device=args.device, batch_size=args.batch_size, alias_table=alias_table)
    evaluation = evaluate(run, args.items, sources=requested, seed=args.seed, limit=args.limit)
    summary = summarize(evaluation, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "items": str(args.items), "track": manifest["track"], "kind": manifest["kind"]}
    und.write_jsonl_gz(output / "predictions.jsonl.gz", ({"source": source, **row}
                                                         for source, rows in evaluation["results"].items() for row in rows))
    write_json(output / "summary.json", {**header, "resolved": evaluation["resolved"], "summary": summary,
                                         "seconds": evaluation["seconds"], "items_manifest_sha256": manifest.get("sha256")})
    (output / "report.md").write_text(render_report(summary, header))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary, "output": str(output)}


def load_evaluation(folder: Path) -> dict[str, Any] | None:
    """A finished evaluation folder as `evaluate` returns it (results per source, resolution, kind); None if absent."""
    folder = Path(folder)
    if not (folder / "summary.json").exists() or not (folder / "predictions.jsonl.gz").exists():
        return None
    document = json.loads((folder / "summary.json").read_text())
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in und.read_jsonl(folder / "predictions.jsonl"):
        by_source[row.pop("source")].append(row)
    return {"kind": document["kind"], "resolved": document["resolved"], "sources": list(by_source), "results": dict(by_source),
            "manifest": {}, "document": document}


# ---------------------------------------------------------------- queue

COMPOSING = ("C5", "C5rf", "C5ut", "C5tr", "C5sh")


def evaluate_command(run_dir: Path, items_dir: Path, *, python: str = sys.executable, batch_size: int | None = None,
                     sources: Sequence[str] | None = None) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_binding_items", "evaluate", "--run", str(run_dir), "--items", str(items_dir),
            "--overwrite", *(["--batch-size", str(batch_size)] if batch_size else []), *(["--sources", ",".join(sources)] if sources else [])]


def job_sources(model: str, kind: str, channel_mode: str) -> list[str]:
    """Sources of a model's job: composing channels own, none (and swap on twins); other channels own, none; no channel own."""
    if channel_mode == "none":
        return ["own"]
    if channel_mode == "compose":
        return ["own", "none", "swap"] if kind == "twins" else ["own", "none"]
    return ["own", "none"]


def queue_stage(stage: str, items_dir: Path, *, priority: int = 50, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None, dry_run: bool = False) -> list[dict[str, Any]]:
    """One GPU-lane evaluation job per config of `stage` (named `<stage>-<stem>-<output folder>`, idempotent)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import EVAL_JOB_BATCH, _config_host, stage_python, stem_model
    kind = json.loads((Path(items_dir) / "manifest.json").read_text())["kind"]
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        model = stem_model(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        config = yaml.safe_load(path.read_text())
        host = _config_host(config) or ""
        mode = (config.get("channel") or {}).get("mode", "none")
        run_dir = Path(root) / "runs" / stage / path.stem
        jobs.append({"name": f"{stage}-{path.stem}-{output_folder(run_dir, items_dir).name}", "priority": int(priority), "model": model,
                     "command": evaluate_command(run_dir, items_dir, python=python, batch_size=EVAL_JOB_BATCH.get(host, 32),
                                                 sources=job_sources(model, kind, mode))})
    if dry_run:
        return jobs
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"], min_free_gb=5,
                env={"PYTHONPATH": "src"}, resume_args=[])
            queued.append(job)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("items", help="build role-swap twins or natural role-ambiguous items for a track")
    build.add_argument("--track", required=True); build.add_argument("--kind", required=True, choices=KINDS)
    build.add_argument("--family", default="smollm2"); build.add_argument("--output", type=Path, required=True)
    build.add_argument("--count", type=int, default=300, help="twin pairs")
    build.add_argument("--subset-count", action="append", default=[], help="natural: subset=N anchors (defaults seen=300 rare=300 heldout=600)")
    build.add_argument("--seed", type=int, default=0); build.add_argument("--name-seed", type=int, default=23)
    ev = sub.add_parser("evaluate", help="score a finished run on a role item directory")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--output", type=Path, default=None); ev.add_argument("--alias-table", type=Path, default=None)
    ev.add_argument("--sources", default=""); ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--resamples", type=int, default=2000); ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--device", default=None); ev.add_argument("--overwrite", action="store_true")
    ev.add_argument("--limit", type=int, default=None, help="smoke tests only: the first N concepts (twins: pairs)")
    queue = sub.add_parser("queue", help="queue one evaluation per config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--items", type=Path, required=True)
    queue.add_argument("--priority", type=int, default=50); queue.add_argument("--models", nargs="*", default=None)
    queue.add_argument("--seeds", type=int, nargs="*", default=None); queue.add_argument("--root", type=Path, default=ROOT)
    queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "items":
        counts = {k: int(v) for k, v in (c.split("=", 1) for c in args.subset_count)}
        manifest = build_items(args.track, args.kind, args.output, family=args.family, count=args.count, seed=args.seed,
                               name_seed=args.name_seed, counts=counts)
        print(json.dumps({k: manifest.get(k) for k in ("kind", "counts", "relation_pairs", "anchors")}, indent=2, default=str))
    elif args.command == "evaluate":
        result = run_evaluate(args)
        print(json.dumps({"output": result["output"], "contrast": {s: {k: v.get("all", {}).get("contrast") for k, v in block.items()}
                                                                    for s, block in result["summary"]["sources"].items()}}, indent=2))
    else:
        jobs = queue_stage(args.stage, args.items, priority=args.priority, models=args.models, seeds=args.seeds, root=args.root,
                           dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
