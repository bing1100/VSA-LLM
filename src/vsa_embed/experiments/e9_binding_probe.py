"""E9 binding and unbinding program, step 1: the decodability probe (decision 60; pre-registration
`experiments/e9-retrofit/preregistration-binding.md` §3–4). Evaluation only, on finished runs.

**Probe A — algebraic.** For a run whose channel composes (C5 and its arms), every role of every probed concept is
unbound from the concept's bundle with the operator's own unbinding (`relations.RelationTransform.unbind`: learned HRR
→ circular correlation, plus the regularized-division variant `inverse`; fixed unitary C5rf → conjugate; translation
C5tr → subtraction; untyped C5ut → the bundle readout, which ignores the role) and cleaned up against the run's trained
atomic dictionary (`cleanup`): cosine nearest neighbour over every atomic (`all`) and over the atomics observed under
that relation in the frames (`typed`), with filtered ranks (the frame's other fillers of the same relation removed) and
ties at half weight. The bundle is read in four conditions:

- `static` — the static frame bundle `Σ_e T_{r_e}(a_e)` (every edge weight 1): the store;
- `nocontext` — the composer's attentive weights without a context term;
- `neutral` — the attentive weights with the P1 context of a neutral mention ("We discussed X", `NEUTRAL_TEMPLATE`);
- `context` — the attentive weights in the term's real evaluation contexts (the first `contexts` occurrences in the
  run's own evaluation windows; values averaged over occurrences).

Unbinding reads the un-normalized sum (`FrameComposer.raw_bundle`): equal to the normalized vector for every linear
operator, and the only fair input for the affine translation. *Role recovery* asks the reverse question: which
relation holds a given filler? Every relation `r'` is scored by `cos(c, T_{r'}(a_f))` (for HRR this is the filler
unbound from the bundle and matched against the role vectors), with the same filtered ranking over all relations or the
relations observed with that filler. Model-free references on the same edges: the relation-conditional frequency of
each filler in the training-visible frames (`frequency`) and the expected reciprocal rank of a random ranking
(`chance`).

**Probe B — learned (LRE-style).** Per relation, a linear map (ridge on standardized features, strength by GCV) from a
concept representation to the multi-hot set of its fillers, trained on the seen concepts (not held out, training
frequency ≥ 1; out-of-fold over 5 folds for them) and tested on the held-out and unseen ones; ranks over the fillers the
training concepts carry (a filler never seen in training is uncovered and scores 0). Representations: the composed
vector (`composed_static`, `composed`; every composing arm), C2's free row (`free_row`, the mean-row fallback for
held-out terms as the model sees it), C6's frozen source row (`source_row`), the injected row after the projector
(`injected_row`; every channel) and the host's hidden state at the term's last subtoken in the neutral mention
(`hidden_middle` at layer ⌊L/2⌋, `hidden_final`; P0, C0′, C2, C5 by default). Targets are the ontology frames (for the
shuffled-frame arm C5sh, probe A reads its own schedule and probe B the true frames).

**Subsets and splits.** `seen` (training frequency ≥ 10), `rare` (1–9), `unseen` (0, not held out), `heldout` (held out
and generator zero-shot terms, composed zero-shot); at most `cap` entries per subset (a seeded sample on large tracks,
identical for every run of a track; every T5 entry). Edges are split by frame size (superposition load) and role
ambiguity: `ambiguous` (the filler occurs under ≥ 2 relations in the frames), `reuse` (the same filler under two roles
within this frame), `multi` (the frame has ≥ 2 fillers of this relation).

**Outputs** (`RUN/binding-probe/`): `summary.json` (aggregates), `edges.npz` (per edge: ids, flags and the reciprocal
rank of every probe; float16), `report.md`, `resolved_config.yaml`, `manifest.json`. `--licensed` (T1c): aggregates only —
no `edges.npz`, and relations named by index.

    python -m vsa_embed.experiments.e9_binding_probe probe --run RUN [--hidden auto|all|none] [--device cuda] [--licensed]
    python -m vsa_embed.experiments.e9_binding_probe queue --stage t5 --priority 50 [--models …] [--seeds …] [--licensed]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from .. import cleanup as cl
from ..relations import readout_method
from ..span_channel import normalize_alias
from .e5_common import E5Run, canonical_surfaces, clear_output, finish_output, json_ready, open_run, start_output, write_json

SCHEMA = "e9-binding-probe/1"
ROOT = Path("experiments/e9-retrofit")
OUTPUT = "binding-probe"
SUBSETS = ("seen", "rare", "unseen", "heldout")
CONDITIONS = ("static", "nocontext", "neutral", "context")
CLEANUPS = ("all", "typed")
NEUTRAL_TEMPLATE = "We discussed {x}"
HIDDEN_MODELS = ("P0", "C0p", "C2", "C5")
DEGREE_BINS = ((1, 4), (5, 7), (8, 10), (11, None))
FLAGS = ("ambiguous", "unambiguous", "reuse", "multi")
RESULT_FILES = ("summary.json", "edges.npz", "report.md", "resolved_config.yaml", "manifest.json")


@dataclass
class ProbeSettings:
    cap: int = 3000                          # entries per subset (seeded sample when larger; every T5 entry)
    contexts: int = 4                        # real evaluation contexts per entry
    folds: int = 5                           # probe B: out-of-fold predictions for the training concepts
    seed: int = 0
    neutral: str = NEUTRAL_TEMPLATE
    chunk: int = 4096
    role_chunk: int = 1024
    batch_size: int = 32
    min_train: int = 10                      # probe B: concepts per relation needed to fit a map
    alphas: tuple[float, ...] = tuple(10.0 ** p for p in range(-2, 5))


# ---------------------------------------------------------------- entries and edges


def entry_subsets(ontology: dict[str, Any]) -> np.ndarray:
    """Subset label of every entry (`SUBSETS`; the trainer's frequency strata, held-out terms first)."""
    frequency = np.asarray(ontology.get("train_frequency") or np.zeros(int(ontology["entry_count"])), dtype=np.int64)
    labels = np.where(frequency >= 10, "seen", np.where(frequency >= 1, "rare", "unseen")).astype(object)
    labels[[int(e) for e in ontology.get("heldout_entries", ())]] = "heldout"
    return labels.astype(str)


def select_entries(ontology: dict[str, Any], degrees: np.ndarray, *, cap: int, seed: int = 0) -> np.ndarray:
    """Sorted entry ids: every entry with ≥ 1 edge, at most `cap` per subset (a seeded uniform sample; depends only on the
    ontology, so every run of a track probes the same entries)."""
    labels = entry_subsets(ontology)
    rng = np.random.default_rng(seed)
    chosen = []
    for subset in SUBSETS:
        members = np.flatnonzero((labels == subset) & (np.asarray(degrees) > 0))
        if members.size > cap:
            members = np.sort(rng.choice(members, cap, replace=False))
        chosen.append(members)
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=np.int64)


def frame_edges(offsets: np.ndarray, relations: np.ndarray, fillers: np.ndarray, entries: np.ndarray) -> dict[str, np.ndarray]:
    """The edges of `entries` (in entry order): schedule edge id, entry, relation, filler, frame degree and the role-ambiguity
    flags (`ambiguous` = the filler occurs under ≥ 2 relations anywhere in the schedule; `reuse` = under ≥ 2 relations in
    this frame; `multi` = this frame has ≥ 2 fillers of this relation)."""
    offsets, relations, fillers = (np.asarray(x, dtype=np.int64) for x in (offsets, relations, fillers))
    relations_of: dict[int, set[int]] = defaultdict(set)
    for r, f in zip(relations.tolist(), fillers.tolist()):
        relations_of[f].add(r)
    ambiguous_filler = {f for f, rs in relations_of.items() if len(rs) >= 2}
    ids, owners, reuse, multi = [], [], [], []
    for e in np.asarray(entries, dtype=np.int64).tolist():
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        frame_r, frame_f = relations[lo:hi].tolist(), fillers[lo:hi].tolist()
        roles_of: dict[int, set[int]] = defaultdict(set)
        fills_of: dict[int, set[int]] = defaultdict(set)
        for r, f in zip(frame_r, frame_f):
            roles_of[f].add(r); fills_of[r].add(f)
        for k, (r, f) in enumerate(zip(frame_r, frame_f)):
            ids.append(lo + k); owners.append(e)
            reuse.append(len(roles_of[f]) >= 2); multi.append(len(fills_of[r]) >= 2)
    ids = np.asarray(ids, dtype=np.int64)
    owners = np.asarray(owners, dtype=np.int64)
    return {"edge": ids, "entry": owners, "relation": relations[ids] if ids.size else ids, "filler": fillers[ids] if ids.size else ids,
            "degree": (offsets[owners + 1] - offsets[owners]) if ids.size else ids,
            "ambiguous": np.asarray([f in ambiguous_filler for f in fillers[ids].tolist()], dtype=bool) if ids.size else ids.astype(bool),
            "reuse": np.asarray(reuse, dtype=bool), "multi": np.asarray(multi, dtype=bool)}


def within_frame_exclusions(segments: Tensor, relations: Tensor, fillers: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """For edges grouped by occurrence (`segments`, contiguous): the pairs (edge, atomic) of the same occurrence's other
    fillers of the edge's relation (filtered filler ranking) and (edge, relation) of the other relations holding the edge's
    filler (filtered role ranking)."""
    device = segments.device
    n = segments.numel()
    if n == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return (empty, empty), (empty, empty)
    counts = torch.bincount(segments)
    sizes = counts[segments]                                       # degree of each edge's occurrence
    starts = torch.cumsum(counts, 0) - counts
    first = starts[segments]
    rows = torch.repeat_interleave(torch.arange(n, device=device), sizes)
    offsets_within = torch.arange(rows.numel(), device=device) - torch.repeat_interleave(torch.cumsum(sizes, 0) - sizes, sizes)
    cols = first[rows] + offsets_within
    other = rows != cols
    rows, cols = rows[other], cols[other]
    same_relation = (relations[rows] == relations[cols]) & (fillers[rows] != fillers[cols])
    same_filler = (fillers[rows] == fillers[cols]) & (relations[rows] != relations[cols])
    return ((rows[same_relation], fillers[cols[same_relation]]), (rows[same_filler], relations[cols[same_filler]]))


def _pairs_mask(pairs: tuple[Tensor, Tensor], part: slice, width: int, device: torch.device) -> Tensor | None:
    rows, cols = pairs
    hit = (rows >= part.start) & (rows < part.stop)
    if not bool(hit.any()):
        return None
    mask = torch.zeros(part.stop - part.start, width, dtype=torch.bool, device=device)
    mask[rows[hit] - part.start, cols[hit]] = True
    return mask


def _chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def chance_reciprocal(candidates: Tensor) -> Tensor:
    """Expected reciprocal rank of a uniformly random ranking among `n` candidates: `H_n / n`."""
    n = candidates.clamp_min(1).double()
    harmonic = torch.special.digamma(n + 1) + 0.5772156649015329
    return (harmonic / n).float()


# ---------------------------------------------------------------- contexts


def entry_surfaces(run: E5Run, entries: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Per entry the surface shown in prompts: the concept's own name when it is an alias of the entry with ≥ ℓ_min
    subtokens (its casing), else the canonical alias (`e5_common.canonical_surfaces`); `linkable` as there."""
    table = run.table
    surfaces = canonical_surfaces(table, run.tokenizer, run.min_subtokens, entries)
    names = list(run.ontology.get("concept_names") or [])
    for e in entries:
        info = surfaces.get(int(e))
        if info is None or not names or len(table.entry_concepts[int(e)]) != 1:
            continue
        name = str(names[table.entry_concepts[int(e)][0]])
        bare = name.split(":", 1)[1] if name.startswith("synthetic:") else name
        if table.alias_to_entry.get(normalize_alias(bare)) == int(e) and \
                len(run.tokenizer.encode(" " + bare, add_special_tokens=False)) >= run.min_subtokens:
            surfaces[int(e)] = {**info, "surface": bare, "linkable": True}
    return surfaces


@torch.no_grad()
def neutral_contexts(run: E5Run, entries: Sequence[int], texts: Sequence[str], *, batch_size: int = 64) -> Tensor:
    """The P1 context at the last token of each text (the term's last subtoken in the neutral mention)."""
    model, tokenizer = run.model, run.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    embedding = model.model.get_input_embeddings()
    out = []
    for part in _chunks(len(texts), batch_size):
        encoded = tokenizer(list(texts[part]), add_special_tokens=False, padding=True, return_tensors="pt")
        ids = encoded["input_ids"].to(run.device)
        last = encoded["attention_mask"].sum(1).to(run.device) - 1
        pooled = model.context(embedding(ids).float(), encoded["attention_mask"].to(run.device))
        out.append(pooled[torch.arange(ids.shape[0], device=run.device), last].float())
    return torch.cat(out) if out else torch.zeros(0, model.context.projection.out_features, device=run.device)


@torch.no_grad()
def evaluation_contexts(run: E5Run, wanted: Sequence[int], *, per_entry: int, batch: int = 8) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """(occurrence entries, P1 contexts) of up to `per_entry` occurrences of each wanted entry in the run's own evaluation
    windows (the trainer's `eval_windows`, in window order)."""
    from ..data.corpus import TokenCorpus, collate_windows, eval_windows
    config = run.config
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    length = int(config["model"]["seq_len"])
    starts = eval_windows(corpus, count=int(config["eval"]["windows"]), length=length)
    wanted_set = {int(e) for e in wanted}
    counts: Counter = Counter()
    model = run.model
    embedding = model.model.get_input_embeddings()
    entries, contexts = [], []
    scanned = 0
    for part in _chunks(len(starts), batch):
        if len(wanted_set) and all(counts[e] >= per_entry for e in wanted_set):
            break
        windows = [corpus.window(s, length, min_subtokens=int(config["data"]["min_subtokens"])) for s in starts[part]]
        scanned += len(windows)
        ids, spans = collate_windows(windows)
        keep = []
        for i, e in enumerate(spans["entry"].tolist()):
            if e in wanted_set and counts[e] < per_entry:
                counts[e] += 1
                keep.append(i)
        if not keep:
            continue
        index = torch.tensor(keep)
        pooled = model.context(embedding(ids.to(run.device)).float())
        contexts.append(pooled[spans["batch"][index].to(run.device), spans["inject"][index].to(run.device)].float())
        entries.append(spans["entry"][index])
    info = {"windows_scanned": scanned, "windows": len(starts), "entries_with_contexts": len(counts),
            "occurrences": int(sum(counts.values()))}
    if not entries:
        width = model.context.projection.out_features
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, width, device=run.device), info
    return torch.cat(entries).long(), torch.cat(contexts), info


# ---------------------------------------------------------------- probe A


@torch.no_grad()
def decode_fillers(composer: Any, summed: Tensor, segments: Tensor, relations: Tensor, fillers: Tensor, candidates: Tensor,
                   exclusions: tuple[Tensor, Tensor], *, method: str, chunk: int, unbinder: Any = None) -> dict[str, Tensor]:
    """Per edge the filtered (rank, hit) of its filler after unbinding its relation from its occurrence's bundle and
    cleaning up over every atomic (`all`) or the relation's candidates (`typed`). `unbinder(relations, vectors)` replaces
    the operator's own unbinding (step 3: exact division of learned HRR)."""
    atomics = F.normalize(composer.atomic_vectors().detach().float(), dim=-1)
    n = relations.numel()
    out = {f"{c}_{k}": torch.empty(n, device=summed.device) for c in CLEANUPS for k in ("rank", "hit")}
    masks = composer.slot_masks() if hasattr(composer, "slot_masks") else None
    if masks is not None:                     # slotted layout: clean up within the relation's slot (decision 61c)
        slot_keys = F.normalize(composer.atomic_vectors().detach().float()[None] * masks[:, None, :].to(summed.device), dim=-1)
        slot_of = composer.slot_of().to(summed.device)
    for part in _chunks(n, chunk):
        unbound = (unbinder(relations[part], summed[segments[part]]) if unbinder is not None
                   else composer.unbind(relations[part], summed[segments[part]], method=method))
        queries = F.normalize(unbound.float(), dim=-1)
        if masks is None:
            scores = queries @ atomics.T
        else:
            scores = torch.empty(queries.shape[0], atomics.shape[0], device=summed.device)
            slots = slot_of[relations[part]]
            for g in range(masks.shape[0]):
                rows = (slots == g).nonzero(as_tuple=True)[0]
                if rows.numel():
                    scores[rows] = queries[rows] @ slot_keys[g].T
        exclude = _pairs_mask(exclusions, part, atomics.shape[0], summed.device)
        for cleanup, allowed in (("all", None), ("typed", candidates[relations[part]])):
            rank, hit = cl.filtered_ranks(scores, fillers[part], allowed=allowed, exclude=exclude, return_hits=True)
            out[f"{cleanup}_rank"][part], out[f"{cleanup}_hit"][part] = rank, hit
    return out


@torch.no_grad()
def decode_roles(composer: Any, summed: Tensor, segments: Tensor, relations: Tensor, fillers: Tensor, candidates: Tensor,
                 exclusions: tuple[Tensor, Tensor], *, chunk: int) -> dict[str, Tensor]:
    """Per edge the filtered (rank, hit) of its relation among every relation (`all`) or the relations observed with its
    filler (`typed`), scoring `r'` by `cos(c, T_{r'}(a_f))`."""
    atomics = composer.atomic_vectors().detach().float()
    count = composer.relation_count
    n = relations.numel()
    out = {f"{c}_{k}": torch.empty(n, device=summed.device) for c in CLEANUPS for k in ("rank", "hit")}
    every = torch.arange(count, device=summed.device)
    masks = composer.slot_masks() if hasattr(composer, "slot_masks") else None
    for part in _chunks(n, chunk):
        size = part.stop - part.start
        filler_vectors = atomics[fillers[part]]
        bound = composer.transform(every.repeat(size), filler_vectors.repeat_interleave(count, 0)).float().view(size, count, -1)
        bundles = summed[segments[part]].float()
        if masks is None:
            scores = torch.einsum("nd,nrd->nr", F.normalize(bundles, dim=-1), F.normalize(bound, dim=-1))
        else:                                 # slotted: compare within each relation's slot (decision 61c)
            slot_norms = torch.sqrt((bundles[:, None, :] * masks[None].to(bundles.device)).square().sum(-1)).clamp_min(1e-12)
            norms = slot_norms[:, composer.slot_of().to(bundles.device)]
            scores = torch.einsum("nd,nrd->nr", bundles, F.normalize(bound, dim=-1)) / norms
        exclude = _pairs_mask(exclusions, part, count, summed.device)
        for cleanup, allowed in (("all", None), ("typed", candidates[:, fillers[part]].T)):
            rank, hit = cl.filtered_ranks(scores, relations[part], allowed=allowed, exclude=exclude, return_hits=True)
            out[f"{cleanup}_rank"][part], out[f"{cleanup}_hit"][part] = rank, hit
    return out


@torch.no_grad()
def frequency_baseline(counts: Tensor, segments: Tensor, relations: Tensor, fillers: Tensor, candidates: Tensor,
                       filler_exclusions: tuple[Tensor, Tensor], role_exclusions: tuple[Tensor, Tensor], *, chunk: int) -> dict[str, Tensor]:
    """Model-free references on the same edges: rank by the relation-conditional filler frequency (`counts` = relations ×
    atomics over the training-visible frames) and by the filler-conditional relation frequency; and the chance reciprocal
    rank of the filtered candidate sets."""
    n = relations.numel()
    device = counts.device
    out: dict[str, Tensor] = {}
    keys = [f"filler_{c}_{k}" for c in CLEANUPS for k in ("rank", "hit")] + [f"role_{c}_{k}" for c in CLEANUPS for k in ("rank", "hit")]
    out.update({k: torch.empty(n, device=device) for k in keys})
    out.update({f"chance_filler_{c}": torch.empty(n, device=device) for c in CLEANUPS})
    out.update({f"chance_role_{c}": torch.empty(n, device=device) for c in CLEANUPS})
    atomics, count = counts.shape[1], counts.shape[0]
    for part in _chunks(n, chunk):
        scores = counts[relations[part]].float()
        exclude = _pairs_mask(filler_exclusions, part, atomics, device)
        for cleanup, allowed in (("all", None), ("typed", candidates[relations[part]])):
            rank, hit = cl.filtered_ranks(scores, fillers[part], allowed=allowed, exclude=exclude, return_hits=True)
            out[f"filler_{cleanup}_rank"][part], out[f"filler_{cleanup}_hit"][part] = rank, hit
            pool = torch.full_like(scores, True, dtype=torch.bool) if allowed is None else allowed.clone()
            if exclude is not None:
                pool &= ~exclude
            pool[torch.arange(scores.shape[0], device=device), fillers[part]] = True
            out[f"chance_filler_{cleanup}"][part] = chance_reciprocal(pool.sum(-1))
        role_scores = counts[:, fillers[part]].T.float()
        exclude = _pairs_mask(role_exclusions, part, count, device)
        for cleanup, allowed in (("all", None), ("typed", candidates[:, fillers[part]].T)):
            rank, hit = cl.filtered_ranks(role_scores, relations[part], allowed=allowed, exclude=exclude, return_hits=True)
            out[f"role_{cleanup}_rank"][part], out[f"role_{cleanup}_hit"][part] = rank, hit
            pool = torch.full_like(role_scores, True, dtype=torch.bool) if allowed is None else allowed.clone()
            if exclude is not None:
                pool &= ~exclude
            pool[torch.arange(role_scores.shape[0], device=device), relations[part]] = True
            out[f"chance_role_{cleanup}"][part] = chance_reciprocal(pool.sum(-1))
    return out


def unbinding_methods(composer: Any) -> list[str]:
    """The methods probe A reads: the operator's primary method, plus every other method it has (learned HRR: `inverse`)."""
    primary = readout_method(composer.transform)
    others = [m for m in composer.transform.unbind_methods if m != primary and primary != "bundle"]
    return [primary, *others]


@torch.no_grad()
def algebraic_probe(run: E5Run, entries: np.ndarray, edges: dict[str, np.ndarray], surfaces: dict[int, dict[str, Any]],
                    settings: ProbeSettings, log: Any = print) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Probe A on the composer's own schedule: per (condition, method, cleanup, target) the per-edge reciprocal rank and
    hit, aligned with `edges` (the static edge table), plus the model-free references."""
    composer = run.channel.composer
    device = run.device
    schedule = composer.schedule
    count, atomics = composer.relation_count, composer.atomics.shape[0]
    candidates = cl.relation_candidates(schedule.relations, schedule.fillers, count, atomics).to(device)
    ontology = run.ontology
    heldout = torch.tensor(sorted(int(e) for e in ontology.get("heldout_entries", ())), dtype=torch.long)
    offsets = schedule.offsets.cpu()
    owner = torch.repeat_interleave(torch.arange(offsets.numel() - 1), offsets[1:] - offsets[:-1])
    visible = ~torch.isin(owner, heldout)
    counts = torch.zeros(count, atomics, device=device)
    counts.index_put_((schedule.relations.cpu()[visible].to(device), schedule.fillers.cpu()[visible].to(device)),
                      torch.ones(int(visible.sum()), device=device), accumulate=True)
    position = {int(edge): i for i, edge in enumerate(edges["edge"].tolist())}
    n_edges = edges["edge"].size
    methods = unbinding_methods(composer)
    values: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"methods": methods, "conditions": {}, "operator": composer.operator,
                            "family": composer.transform.family, "mode": composer.mode}
    ids = torch.as_tensor(entries, dtype=torch.long, device=device)

    def record(prefix: str, result: dict[str, Tensor], edge_index: Tensor) -> None:
        """Average per schedule edge (over occurrences) and store as reciprocal rank and hit."""
        slots = torch.tensor([position[int(e)] for e in edge_index.tolist()], dtype=torch.long)
        weight = torch.bincount(slots, minlength=n_edges).double()
        for cleanup in CLEANUPS:
            rr = (1.0 / result[f"{cleanup}_rank"]).double().cpu()
            hit = result[f"{cleanup}_hit"].double().cpu()
            mean_rr = torch.zeros(n_edges, dtype=torch.float64).index_add_(0, slots, rr) / weight
            mean_hit = torch.zeros(n_edges, dtype=torch.float64).index_add_(0, slots, hit) / weight
            values[f"{prefix}.{cleanup}.rr"] = mean_rr.numpy()
            values[f"{prefix}.{cleanup}.hit"] = mean_hit.numpy()

    def read(condition: str, summed: Tensor, edge_index: Tensor, segments: Tensor) -> None:
        rel = schedule.relations[edge_index].to(device)
        fil = schedule.fillers[edge_index].to(device)
        filler_ex, role_ex = within_frame_exclusions(segments, rel, fil)
        for method in methods:
            result = decode_fillers(composer, summed, segments, rel, fil, candidates, filler_ex, method=method, chunk=settings.chunk)
            record(f"a.{condition}.{method}.filler", result, edge_index)
        roles = decode_roles(composer, summed, segments, rel, fil, candidates, role_ex, chunk=settings.role_chunk)
        record(f"a.{condition}.match.role", roles, edge_index)
        if condition == "static":
            base = frequency_baseline(counts, segments, rel, fil, candidates, filler_ex, role_ex, chunk=settings.chunk)
            for target in ("filler", "role"):
                record(f"b.frequency.{target}", {k[len(target) + 1:]: v for k, v in base.items() if k.startswith(target + "_")}, edge_index)
                for cleanup in CLEANUPS:
                    values[f"b.chance.{target}.{cleanup}.rr"] = base[f"chance_{target}_{cleanup}"].double().cpu().numpy()
        info["conditions"][condition] = {"occurrences": int(summed.shape[0]), "edges": int(edge_index.numel())}

    started = time.monotonic()
    summed, _, edge_index, segments = composer.raw_bundle(ids, uniform=True)
    if not torch.equal(edge_index.cpu(), torch.as_tensor(edges["edge"])):
        raise RuntimeError("the static edge table does not match the composer's schedule")
    read("static", summed.float(), edge_index, segments)
    attentive = composer.mode != "bundle"
    if attentive:
        summed, _, edge_index, segments = composer.raw_bundle(ids, None)
        read("nocontext", summed.float(), edge_index, segments)
    if attentive and getattr(run.model, "context", None) is not None and composer.context is not None:
        texts = [settings.neutral.format(x=surfaces[int(e)]["surface"]) for e in entries]
        contexts = neutral_contexts(run, entries, texts, batch_size=settings.batch_size)
        summed, _, edge_index, segments = composer.raw_bundle(ids, contexts.to(composer.atomics.dtype))
        read("neutral", summed.float(), edge_index, segments)
        occurrences, contexts, context_info = evaluation_contexts(run, entries, per_entry=settings.contexts)
        info["contexts"] = context_info
        if occurrences.numel():
            summed, _, edge_index, segments = composer.raw_bundle(occurrences.to(device), contexts.to(composer.atomics.dtype))
            read("context", summed.float(), edge_index, segments)
    info["seconds"] = round(time.monotonic() - started, 1)
    log(f"  probe A: {sorted(info['conditions'])} × {methods} in {info['seconds']} s")
    return values, info


# ---------------------------------------------------------------- probe B


def ridge_fit(x: Tensor, y: Tensor, alphas: Sequence[float]) -> tuple[Any, float]:
    """Multi-output ridge on standardized inputs, strength by generalized cross-validation (`e5_zeroshot.ridge_map` in
    torch); returns (predict, alpha)."""
    x, y = x.double(), y.double()
    mean, std = x.mean(0), x.std(0, unbiased=False) + 1e-8
    y_mean = y.mean(0)
    u, s, vt = torch.linalg.svd((x - mean) / std, full_matrices=False)
    uy = u.T @ (y - y_mean)
    centered = y - y_mean
    best = None
    for alpha in alphas:
        shrink = s ** 2 / (s ** 2 + alpha)
        residual = centered - u @ (shrink[:, None] * uy)
        denominator = max(1e-12, 1 - float(shrink.sum()) / x.shape[0]) ** 2
        score = float(residual.square().mean()) / denominator
        if best is None or score < best[0]:
            best = (score, float(alpha))
    w = vt.T @ ((s / (s ** 2 + best[1]))[:, None] * uy)
    return (lambda z: ((z.double() - mean) / std) @ w + y_mean), best[1]


def learned_probe(features: dict[str, np.ndarray], entries: np.ndarray, subsets: np.ndarray, edges: dict[str, np.ndarray],
                  settings: ProbeSettings, device: torch.device, log: Any = print) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Probe B: per representation and relation a ridge map to the multi-hot filler set, trained on the seen concepts
    (out-of-fold for them) and applied to the others; per edge of `edges` (ontology frames) the filtered reciprocal rank,
    hit and coverage (NaN where the representation is missing or the relation could not be fitted)."""
    position = {int(e): i for i, e in enumerate(entries.tolist())}
    edge_row = np.asarray([position[int(e)] for e in edges["entry"].tolist()], dtype=np.int64)
    subset_of = subsets[entries]
    train_entries = np.isin(subset_of, ("seen", "rare"))
    values: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    n_edges = edges["edge"].size
    for name, matrix in features.items():
        started = time.monotonic()
        rr = np.full(n_edges, np.nan); hit = np.full(n_edges, np.nan); covered = np.zeros(n_edges, dtype=bool)
        available = np.isfinite(matrix).all(1)
        fitted: dict[str, Any] = {}
        x_all = torch.as_tensor(np.nan_to_num(matrix), dtype=torch.float64, device=device)
        for relation in np.unique(edges["relation"]).tolist():
            on = np.flatnonzero(edges["relation"] == relation)
            rows = np.unique(edge_row[on])
            rows = rows[available[rows]]
            train_rows = rows[train_entries[rows]]
            if train_rows.size < settings.min_train:
                continue
            train_edges = on[np.isin(edge_row[on], train_rows)]
            classes = np.unique(edges["filler"][train_edges])
            if classes.size < 2:
                continue
            class_of = {int(c): k for k, c in enumerate(classes.tolist())}
            target = torch.zeros(train_rows.size, classes.size, dtype=torch.float64, device=device)
            local = {int(r): k for k, r in enumerate(train_rows.tolist())}
            for i in train_edges.tolist():
                target[local[int(edge_row[i])], class_of[int(edges["filler"][i])]] = 1.0
            scores = torch.full((matrix.shape[0], classes.size), float("nan"), dtype=torch.float32, device=device)
            folds = np.random.default_rng(settings.seed + int(relation)).permutation(train_rows.size) % settings.folds
            index = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
            for k in range(settings.folds):
                fit, test = folds != k, folds == k
                if fit.sum() < 2 or not test.any():
                    continue
                predict, _ = ridge_fit(x_all[index(train_rows[fit])], target[index(np.flatnonzero(fit))], settings.alphas)
                scores[index(train_rows[test])] = predict(x_all[index(train_rows[test])]).float()
            predict, alpha = ridge_fit(x_all[index(train_rows)], target, settings.alphas)
            others = rows[~train_entries[rows]]
            if others.size:
                scores[index(others)] = predict(x_all[index(others)]).float()
            # rank each edge's filler among the classes; other fillers of the relation in the same frame removed
            mine = on[available[edge_row[on]]]
            gold = np.asarray([class_of.get(int(f), -1) for f in edges["filler"][mine].tolist()])
            ok = gold >= 0
            covered[mine[ok]] = True
            rr[mine[~ok]] = 0.0; hit[mine[~ok]] = 0.0
            if ok.any():
                chosen = mine[ok]
                row_scores = scores[index(edge_row[chosen])]
                exclude = torch.zeros_like(row_scores, dtype=torch.bool)
                by_row: dict[int, list[int]] = defaultdict(list)
                for i in mine.tolist():
                    by_row[int(edge_row[i])].append(class_of.get(int(edges["filler"][i]), -1))
                for j, i in enumerate(chosen.tolist()):
                    for c in by_row[int(edge_row[i])]:
                        if c >= 0 and c != class_of[int(edges["filler"][i])]:
                            exclude[j, c] = True
                rank, h = cl.filtered_ranks(row_scores, torch.as_tensor(gold[ok], device=device), exclude=exclude, return_hits=True)
                rr[chosen] = (1.0 / rank).cpu().numpy(); hit[chosen] = h.cpu().numpy()
            fitted[str(relation)] = {"train": int(train_rows.size), "classes": int(classes.size), "alpha": alpha}
        values[f"l.{name}.rr"] = rr; values[f"l.{name}.hit"] = hit; values[f"l.{name}.covered"] = covered
        info[name] = {"dimension": int(matrix.shape[1]), "available_entries": int(available.sum()), "relations": fitted,
                      "seconds": round(time.monotonic() - started, 1)}
        log(f"  probe B: {name} ({matrix.shape[1]} dims, {len(fitted)} relations) in {info[name]['seconds']} s")
    return values, info


@torch.no_grad()
def channel_representations(run: E5Run, entries: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The channel's per-concept vectors of `entries`: the composed vector (static bundle and as composed without
    context), C2's free row as the model reads it (held-out terms: the mean-row fallback), C6's source row, and the row
    injected after the projector."""
    from .e5_common import entry_rows
    channel = run.channel
    reps: dict[str, np.ndarray] = {}
    described: dict[str, str] = {}
    if channel is None:
        return reps, described
    ids = torch.as_tensor(entries, dtype=torch.long, device=run.device)
    if channel.mode == "compose":
        composer = channel.composer
        static, *_ = composer.raw_bundle(ids, uniform=True)
        reps["composed_static"] = F.normalize(static.float(), dim=-1).cpu().numpy()
        reps["composed"] = torch.cat([composer.compose(part, None).float() for part in ids.split(4096)]).cpu().numpy()
        described.update(composed_static="static frame bundle (normalized)", composed="composed vector without context")
    elif channel.mode == "free":
        rows = channel.table(ids).float()
        if bool(channel.unseen.any()):
            fallback = channel.table.weight[~channel.unseen].mean(0).float()
            rows = torch.where(channel.unseen[ids][:, None], fallback.expand_as(rows), rows)
        reps["free_row"] = rows.cpu().numpy()
        described["free_row"] = "free row before the projector (held-out: the mean-row fallback)"
    elif channel.mode == "source":
        reps["source_row"] = channel.source_rows[ids].float().cpu().numpy()
        described["source_row"] = "frozen source row before the projector"
    if channel.mode in {"compose", "free", "source"}:
        reps["injected_row"] = entry_rows(channel, ids.cpu()).numpy()
        described["injected_row"] = "row injected after the projector (host scale included)"
    return reps, described


@torch.no_grad()
def hidden_representations(run: E5Run, entries: np.ndarray, surfaces: dict[int, dict[str, Any]], settings: ProbeSettings
                           ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The host's hidden state at the term's last subtoken in the neutral mention (layers ⌊L/2⌋ and L), for entries whose
    surface links to them (the same entries for every model of a track); other rows are NaN."""
    from ..evaluation import channel_probes as cp
    adapter = run.adapter
    prefix = settings.neutral.split("{x}")[0]
    texts = [settings.neutral.format(x=surfaces[int(e)]["surface"]) for e in entries]
    spans = [(len(prefix), len(prefix) + len(surfaces[int(e)]["surface"])) for e in entries]
    linkable = np.asarray([bool(surfaces[int(e)]["linkable"]) for e in entries])
    found = adapter.link_targets(texts, spans) if adapter.linker is not None else [[] for _ in texts]
    linked = np.asarray([f == [int(e)] for f, e in zip(found, entries.tolist())]) & linkable
    keep = np.flatnonzero(linked)
    middle = final = None
    layers = None
    for part in _chunks(keep.size, settings.batch_size):
        batch = [texts[i] for i in keep[part]]
        encoded, out, _ = adapter._run_batch(batch)
        states = out.hidden_states
        layers = len(states) - 1
        rows, cols = [], []
        for r, i in enumerate(keep[part].tolist()):
            n = int(encoded["attention_mask"][r].sum())
            offsets = [tuple(o) for o in encoded["offset_mapping"][r, :n].tolist()]
            rows.append(r); cols.append(cp.read_index(offsets, *spans[i]))
        mid = states[layers // 2][rows, cols].float().cpu().numpy()
        last = states[-1][rows, cols].float().cpu().numpy()
        if middle is None:
            middle = np.full((entries.size, mid.shape[1]), np.nan, dtype=np.float32)
            final = np.full((entries.size, last.shape[1]), np.nan, dtype=np.float32)
        middle[keep[part]] = mid; final[keep[part]] = last
    info = {"template": settings.neutral, "linked": int(keep.size), "entries": int(entries.size), "layers": layers,
            "middle_layer": None if layers is None else layers // 2}
    if middle is None:
        return {}, info
    return {"hidden_middle": middle, "hidden_final": final}, info


# ---------------------------------------------------------------- aggregation


def degree_bin(degree: int) -> str:
    for lo, hi in DEGREE_BINS:
        if degree >= lo and (hi is None or degree <= hi):
            return f"{lo}+" if hi is None else f"{lo}-{hi}"
    return "0"


def flag_masks(edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"ambiguous": edges["ambiguous"], "unambiguous": ~edges["ambiguous"], "reuse": edges["reuse"], "multi": edges["multi"]}


def entry_means(values: np.ndarray, owners: np.ndarray, mask: np.ndarray | None = None) -> dict[int, float]:
    """Per entry the mean of its finite values (within `mask`)."""
    keep = np.isfinite(values) if mask is None else (np.isfinite(values) & mask)
    sums: dict[int, float] = defaultdict(float); counts: Counter = Counter()
    for e, v in zip(owners[keep].tolist(), values[keep].tolist()):
        sums[e] += v; counts[e] += 1
    return {e: sums[e] / counts[e] for e in counts}


def _block(rr: np.ndarray, hit: np.ndarray | None, owners: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    keep = mask & np.isfinite(rr)
    if not keep.any():
        return None
    per_entry = entry_means(rr, owners, keep)
    out = {"mrr": float(np.mean(list(per_entry.values()))), "mrr_edges": float(rr[keep].mean()), "entries": len(per_entry),
           "edges": int(keep.sum())}
    if hit is not None:
        out["top1"] = float(np.mean(list(entry_means(hit, owners, keep).values())))
        out["top1_edges"] = float(hit[keep].mean())
    return out


def aggregate(values: dict[str, np.ndarray], edges: dict[str, np.ndarray], subsets: np.ndarray, *, relation_names: Sequence[str],
              licensed: bool = False, by_relation: Sequence[str] = ()) -> dict[str, Any]:
    """Per stored probe (`<key>.rr` / `<key>.hit`): entry-weighted MRR and top-1 per subset, and on held-out and all entries by
    frame-size bin and role-ambiguity flag; per relation for the keys in `by_relation`."""
    owners = edges["entry"]
    edge_subset = subsets[owners]
    flags = flag_masks(edges)
    bins = np.asarray([degree_bin(int(d)) for d in edges["degree"].tolist()])
    keys = sorted({k.rsplit(".", 1)[0] for k in values if k.endswith(".rr")})
    out: dict[str, Any] = {}
    for key in keys:
        rr, hit = values[f"{key}.rr"], values.get(f"{key}.hit")
        block: dict[str, Any] = {"subsets": {}, "degree": {}, "flags": {}}
        for subset in (*SUBSETS, "all"):
            mask = np.ones_like(rr, dtype=bool) if subset == "all" else edge_subset == subset
            found = _block(rr, hit, owners, mask)
            if found:
                block["subsets"][subset] = found
        for subset in ("heldout", "all"):
            base = np.ones_like(rr, dtype=bool) if subset == "all" else edge_subset == subset
            block["degree"][subset] = {b: found for b in sorted(set(bins.tolist())) if (found := _block(rr, hit, owners, base & (bins == b)))}
            block["flags"][subset] = {f: found for f, m in flags.items() if (found := _block(rr, hit, owners, base & m))}
        if f"{key}.covered" in values:
            covered = values[f"{key}.covered"]
            block["coverage"] = {s: float(covered[(edge_subset == s) & np.isfinite(rr)].mean()) if ((edge_subset == s) & np.isfinite(rr)).any()
                                 else None for s in SUBSETS}
        if key in by_relation:
            block["relations"] = {}
            for r in np.unique(edges["relation"]).tolist():
                name = f"r{r}" if licensed else str(relation_names[r])
                block["relations"][name] = {s: found for s in ("heldout", "all")
                                            if (found := _block(rr, hit, owners, (edges["relation"] == r)
                                                                & (np.ones_like(rr, dtype=bool) if s == "all" else edge_subset == s)))}
        out[key] = block
    return out


# ---------------------------------------------------------------- run


def run_identity(run: E5Run) -> dict[str, Any]:
    from .e9_plan import stem_model
    return {"model": stem_model(run.path.name) if re.search(r"-s\d+$", run.path.name) else run.condition, "seed": run.seed,
            "stem": run.path.name, "stage": run.path.parent.name}


def primary_key(methods: Sequence[str]) -> str:
    return f"a.static.{methods[0]}.filler"


def probe_run(run_dir: Path, output: Path | None = None, *, hidden: str = "auto", device: str | None = None,
              licensed: bool = False, overwrite: bool = False, settings: ProbeSettings | None = None,
              alias_table: Path | None = None, log: Any = print) -> dict[str, Any]:
    """Probes A and B on one finished run (module docstring); `hidden` = auto (P0, C0′, C2, C5), all or none."""
    from .e9_tracks import ensure_alias_table, track_spec
    settings = settings or ProbeSettings()
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / OUTPUT
    config_record = {"experiment": "e9-binding-probe", "run": str(run_dir), "hidden": hidden, "licensed": licensed,
                     "settings": asdict(settings)}
    if overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config_record)
    started = time.monotonic()
    from .e5_common import run_config
    config = run_config(run_dir)
    if alias_table is None and config.get("e9_track"):
        alias_table = ensure_alias_table(track_spec(config["e9_track"], config.get("e9_family", "smollm2")))
    run = open_run(run_dir, device=device, batch_size=settings.batch_size, alias_table=alias_table)
    identity = run_identity(run)
    ontology = run.ontology
    offsets = np.asarray(ontology["offsets"], dtype=np.int64)
    degrees = np.diff(offsets)
    subsets = entry_subsets(ontology)
    entries = select_entries(ontology, degrees, cap=settings.cap, seed=settings.seed)
    surfaces = entry_surfaces(run, entries.tolist())
    learned_edges = frame_edges(offsets, np.asarray(ontology["relations"]), np.asarray(ontology["fillers"]), entries)
    log(f"{run_dir.name}: {entries.size} entries ({dict(Counter(subsets[entries].tolist()))}), {learned_edges['edge'].size} edges")
    values: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {"schema": SCHEMA, **identity, "run": str(run_dir), "track": config.get("e9_track"),
                              "family": config.get("e9_family", "smollm2"), "channel": run.mode, "licensed": licensed,
                              "entries": {s: int((subsets[entries] == s).sum()) for s in SUBSETS},
                              "settings": asdict(settings)}
    algebraic_edges = None
    if run.mode == "compose":
        schedule = run.channel.composer.schedule
        algebraic_edges = frame_edges(schedule.offsets.cpu().numpy(), schedule.relations.cpu().numpy(), schedule.fillers.cpu().numpy(),
                                      entries)
        a_values, record["algebraic"] = algebraic_probe(run, entries, algebraic_edges, surfaces, settings, log=log)
        values.update(a_values)
        record["same_edges"] = bool(np.array_equal(algebraic_edges["edge"], learned_edges["edge"])
                                    and np.array_equal(algebraic_edges["filler"], learned_edges["filler"]))
    features, described = channel_representations(run, entries)
    want_hidden = hidden == "all" or (hidden == "auto" and identity["model"] in HIDDEN_MODELS)
    if want_hidden:
        states, record["hidden"] = hidden_representations(run, entries, surfaces, settings)
        features.update(states)
        described.update(hidden_middle=f"host hidden state, layer {record['hidden'].get('middle_layer')}",
                         hidden_final="host hidden state, final layer")
    # The ridge fits run on the CPU in float64 (LAPACK): FP64 SVDs on a consumer GPU are ~10× slower (§11 smoke).
    l_values, record["learned"] = learned_probe(features, entries, subsets, learned_edges, settings, torch.device("cpu"), log=log)
    for name, text in described.items():
        if name in record["learned"]:
            record["learned"][name]["description"] = text
    relation_names = list(ontology["relation_names"])
    summary = {}
    if algebraic_edges is not None:
        a_keys = {k: v for k, v in values.items() if k.startswith(("a.", "b."))}
        methods = record["algebraic"]["methods"]
        focus = [f"{prefix}.{cleanup}" for prefix in (primary_key(methods), f"a.neutral.{methods[0]}.filler", "a.static.match.role",
                                                     "b.frequency.filler") for cleanup in CLEANUPS]
        summary["algebraic"] = aggregate(a_keys, algebraic_edges, subsets, relation_names=relation_names, licensed=licensed,
                                         by_relation=focus)
    summary["learned"] = aggregate(l_values, learned_edges, subsets, relation_names=relation_names, licensed=licensed,
                                   by_relation=[k.rsplit(".", 1)[0] for k in l_values if k.endswith(".rr")])
    record["summary"] = summary
    record["seconds"] = round(time.monotonic() - started, 1)
    if not licensed:
        arrays = {f"le_{k}": v for k, v in learned_edges.items()}
        if algebraic_edges is not None:
            arrays.update({f"ae_{k}": v for k, v in algebraic_edges.items()})
        arrays["entries"] = entries
        arrays["entry_subset"] = subsets[entries]
        for key, value in {**values, **l_values}.items():
            arrays[key] = value.astype(np.float16) if value.dtype.kind == "f" else value
        np.savez_compressed(output / "edges.npz", **arrays)
    write_json(output / "summary.json", json_ready(record))
    (output / "report.md").write_text(render_report(record))
    finish_output(output, config_record, git_at_start=git_at_start, device=run.device, source=run.describe() if not licensed else
                  {"run": str(run_dir), "condition": run.condition, "seed": run.seed})
    return record


def _fmt(block: dict[str, Any] | None, key: str = "mrr") -> str:
    return "–" if not block or block.get(key) is None else f"{block[key]:.3f}"


def render_report(record: dict[str, Any]) -> str:
    lines = [f"# Binding probe — {record['stem']} ({record.get('track')})", "",
             f"Channel `{record['channel']}`; entries {record['entries']}. MRR = mean over entries of the mean reciprocal rank over "
             "their edges (filtered, ties at half weight).", ""]
    algebraic = record.get("summary", {}).get("algebraic")
    if algebraic:
        info = record["algebraic"]
        lines += [f"## Probe A (algebraic; operator `{info['operator']}`, unbinding {info['methods']})", "",
                  "| probe | cleanup | " + " | ".join(SUBSETS) + " |", "|---|---|" + "---:|" * len(SUBSETS)]
        for key in sorted(algebraic):
            base, _, cleanup = key.rpartition(".")
            if cleanup not in CLEANUPS:
                base, cleanup = key, ""
            lines.append(f"| `{base}` | {cleanup} | " + " | ".join(_fmt(algebraic[key]["subsets"].get(s)) for s in SUBSETS) + " |")
        lines.append("")
    learned = record.get("summary", {}).get("learned")
    if learned:
        lines += ["## Probe B (learned, LRE-style; out-of-fold for seen and rare)", "",
                  "| representation | " + " | ".join(SUBSETS) + " | coverage (held-out) |", "|---|" + "---:|" * (len(SUBSETS) + 1)]
        for key in sorted(learned):
            cov = (learned[key].get("coverage") or {}).get("heldout")
            lines.append(f"| `{key}` | " + " | ".join(_fmt(learned[key]["subsets"].get(s)) for s in SUBSETS)
                         + f" | {'–' if cov is None else f'{cov:.2f}'} |")
    return "\n".join(lines) + "\n"


def load_probe(folder: Path) -> dict[str, Any] | None:
    """`{"summary": summary.json, "edges": edges.npz arrays or None}`; None when the probe has not run."""
    folder = Path(folder)
    if not (folder / "summary.json").exists():
        return None
    record = json.loads((folder / "summary.json").read_text())
    arrays = None
    if (folder / "edges.npz").exists():
        with np.load(folder / "edges.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
    return {"summary": record, "edges": arrays}


# ---------------------------------------------------------------- queue


def probe_command(run_dir: Path, *, python: str = sys.executable, hidden: str = "auto", licensed: bool = False) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_binding_probe", "probe", "--run", str(run_dir), "--hidden", hidden,
            "--overwrite", *(["--licensed"] if licensed else [])]


def queue_stage(stage: str, *, priority: int = 50, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None, licensed: bool = False,
                dry_run: bool = False) -> list[dict[str, Any]]:
    """One GPU-lane probe job per config of `stage` (run folders need not exist yet; named `<stage>-<stem>-binding-probe`,
    idempotent)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import _config_host, stage_python, stem_model
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        model = stem_model(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        run_dir = Path(root) / "runs" / stage / path.stem
        jobs.append({"name": f"{stage}-{path.stem}-binding-probe", "priority": int(priority), "model": model,
                     "command": probe_command(run_dir, python=python, licensed=licensed)})
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
    probe = sub.add_parser("probe", help="probes A and B on one finished run")
    probe.add_argument("--run", type=Path, required=True); probe.add_argument("--output", type=Path, default=None)
    probe.add_argument("--hidden", default="auto", choices=["auto", "all", "none"])
    probe.add_argument("--device", default=None); probe.add_argument("--licensed", action="store_true")
    probe.add_argument("--overwrite", action="store_true"); probe.add_argument("--alias-table", type=Path, default=None)
    probe.add_argument("--cap", type=int, default=ProbeSettings.cap, help="entries per subset (smoke tests: small)")
    probe.add_argument("--contexts", type=int, default=ProbeSettings.contexts)
    probe.add_argument("--batch-size", type=int, default=ProbeSettings.batch_size)
    queue = sub.add_parser("queue", help="queue one probe job per config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--priority", type=int, default=50)
    queue.add_argument("--models", nargs="*", default=None); queue.add_argument("--seeds", type=int, nargs="*", default=None)
    queue.add_argument("--root", type=Path, default=ROOT); queue.add_argument("--licensed", action="store_true")
    queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "probe":
        settings = ProbeSettings(cap=args.cap, contexts=args.contexts, batch_size=args.batch_size)
        record = probe_run(args.run, args.output, hidden=args.hidden, device=args.device, licensed=args.licensed,
                           overwrite=args.overwrite, settings=settings, alias_table=args.alias_table)
        print(json.dumps({"run": record["stem"], "seconds": record["seconds"]}))
    else:
        jobs = queue_stage(args.stage, priority=args.priority, models=args.models, seeds=args.seeds, root=args.root,
                           licensed=args.licensed, dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
