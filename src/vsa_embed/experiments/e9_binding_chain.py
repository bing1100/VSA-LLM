"""E9 binding and unbinding program, step 3: chained two-hop, reverse lookup, capacity and path order (decisions 60–61;
pre-registration `experiments/e9-retrofit/preregistration-binding.md` §14). Algebra only, on the trained composer of a
finished run (the channel's composer is read from `final.pt`; the host is not loaded, so jobs run on the CPU).

Every store is a concept's static frame bundle `Σ_e T_{r_e}(a_e)` (as step 1's `static` condition); unbinding is the
operator's own (`relations`); cleanup is typed (the atomics observed under the relation; slotted: within its slot).

- **Fidelity** — filler recovery by frame size k for every unbinding method of the operator, plus exact (unregularized)
  spectral division for learned HRR (decision 61's prediction: unitary ≈ bounded ≥ learned HRR with the adjoint ≫ learned HRR
  with exact division).
- **Two-hop** — unbind r1 from the anchor's store, clean up, take the recovered filler's own store, unbind r2, clean up.
  On ontology paths (anchor → a filler that names a concept with a frame → that frame's filler; every track but T7) and on
  the WP-UB two-hop items (T5, T4: the items' path, bridge and options), with the first hop, the second hop from the true
  bridge (`oracle`) and a soft chain (the bridges' stores mixed by the first cleanup's weights).
- **Reverse lookup** — score every concept c by `cos(c, T_r(a_F))` (`⟨c, r ⊛ F⟩`): the anchor's filtered rank among all
  concepts (MRR) and, on the WP-UB reverse items, the anchor against the item's partner.
- **Capacity** — one global holographic memory of the same facts, `G_N = Σ_{j ≤ N} k_j ⊛ ĉ_j` over N concept stores (keys:
  the unitary projection of the trained atomic that names the concept, else of a seeded random vector; Kumar-style), from
  which a store is recovered by its key before unbinding: filler recovery as N grows, and two-hop through the global memory
  against the local stores.
- **Path order** (decision 61b) — two paths over the same two relations in opposite orders stored in one vector,
  `T_{r2}T_{r1} a_f + T_{r1}T_{r2} a_g` (plus m distractor paths): does unbinding in the order r1, r2 recover f rather
  than g? Commutative operators (circulants, translation, untyped) tie; the block-diagonal unitary need not.

`sweep` runs the same algebra on random vectors (no checkpoint): local recovery over dimension × load per family, the
global memory against N, and path order.

Outputs (`RUN/binding-chain/`): `summary.json` (aggregates), `paths.npz` (per path and per query: ids and outcomes) and
`items.jsonl.gz` (the WP-UB items' outcomes; neither for licensed
runs), `report.md`, `resolved_config.yaml`, `manifest.json`.

    python -m vsa_embed.experiments.e9_binding_chain chain --run RUN [--items DIR] [--licensed]
    python -m vsa_embed.experiments.e9_binding_chain sweep --output experiments/e9-retrofit/report/binding-sweep
    python -m vsa_embed.experiments.e9_binding_chain queue --stage t5 --priority 51 [--models …] [--items DIR] [--licensed] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from .. import cleanup as cl
from ..algebra import HRRAlgebra, UnitaryHRRAlgebra
from ..compose import FrameComposer, FrameSchedule
from ..relations import readout_method, spectral_inverse
from ..training import lm
from . import e9_binding_probe as bp
from .e5_common import finish_output, json_ready, start_output, write_json

SCHEMA = "e9-binding-chain/1"
OUTPUT = "binding-chain"
ROOT = Path("experiments/e9-retrofit")
RESULT_FILES = ("summary.json", "items.jsonl.gz", "paths.npz", "report.md", "resolved_config.yaml", "manifest.json")
MEMORY_SIZES = (1, 4, 16, 64, 256, 1024, 4096, 16384)
EXACT_RIDGE = 1e-12
SOFT_BETA = 16.0
DISTRACTORS = (0, 2, 6)


@dataclass
class ChainSettings:
    cap: int = 3000               # anchors per subset (as step 1's probe)
    paths: int = 2000             # ontology two-hop paths
    reverse: int = 1000           # ontology reverse queries
    memory_targets: int = 300     # concepts read back from the global memory per size
    global_paths: int = 300       # two-hop paths through the global memory
    path_order: int = 1000        # path-order tuples per distractor count
    seed: int = 0
    chunk: int = 2048


# ---------------------------------------------------------------- loading


def load_composer(run_dir: Path) -> tuple[FrameComposer, dict[str, Any], dict[str, Any]]:
    """The run's trained composer (rebuilt from the config and the ontology, its state read from `final.pt`), the resolved
    config and the ontology. The host is not loaded."""
    from ..evaluation.channel_probes import restore_composer_schedule
    final = torch.load(Path(run_dir) / "final.pt", weights_only=False, map_location="cpu")
    config = lm.resolve_config(final["config"])
    settings = config["channel"]
    if settings["mode"] != "compose":
        raise ValueError(f"{run_dir} has no composer (channel mode {settings['mode']!r})")
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    relations = int(ontology["relation_count"])
    schedule = lm.frame_variant(FrameSchedule(ontology["offsets"], ontology["relations"], ontology["fillers"]), settings["frames"],
                                relations, seed=int(config["seed"]))
    atomic_count = relations if settings["frames"] == "relation_only" else int(ontology["atomic_count"])
    window = int(settings["context_window"])
    composer = FrameComposer(schedule, atomic_count, relations, int(settings["dimension"]), operator=settings["operator"],
                             mode=settings["composition"], concept_factor=settings["concept_factor"],
                             key_dimension=int(settings["key_dimension"]), context_dimension=int(settings["key_dimension"]) if window else 0,
                             **({"slots": int(settings["slots"])} if settings.get("slots") else {}))
    if final.get("composer_schedule"):
        restore_composer_schedule(composer, final["composer_schedule"])
    composer.load_state_dict({k[len("channel.composer."):]: v for k, v in final["model"].items() if k.startswith("channel.composer.")})
    del final
    return composer.eval(), config, ontology


def atom_entries(ontology: dict[str, Any]) -> np.ndarray:
    """Per atomic the single entry of the concept it names (`kind:value` with value a concept name), else −1."""
    names = {str(n): i for i, n in enumerate(ontology.get("concept_names") or [])}
    concepts = ontology.get("entry_concepts") or [(e,) for e in range(int(ontology["entry_count"]))]
    entries_of: dict[int, list[int]] = defaultdict(list)
    for e, members in enumerate(concepts):
        for c in members:
            entries_of[int(c)].append(e)
    out = np.full(len(ontology["atomic_names"]), -1, dtype=np.int64)
    for a, atom in enumerate(ontology["atomic_names"]):
        value = atom.split(":", 1)[1] if ":" in atom else atom
        concept = names.get(value)
        found = entries_of.get(concept, []) if concept is not None else []
        if len(found) == 1:
            out[a] = found[0]
    return out


class Store:
    """Frames, static stores and typed cleanup of one composer."""

    def __init__(self, composer: FrameComposer, *, chunk: int = 2048) -> None:
        self.composer = composer
        schedule = composer.schedule
        self.offsets = schedule.offsets.cpu().numpy()
        self.relations = schedule.relations.cpu().numpy()
        self.fillers = schedule.fillers.cpu().numpy()
        self.count = composer.relation_count
        self.atomics = composer.atomic_vectors().detach().float()
        self.candidates = cl.relation_candidates(schedule.relations, schedule.fillers, self.count, self.atomics.shape[0])
        self.candidates[~self.candidates.any(-1)] = True
        self.method = readout_method(composer.transform)
        self.chunk = chunk
        self._stores: Tensor | None = None
        masks = composer.slot_masks()
        self.slot_masks = masks
        self.slot_of = composer.slot_of()

    def frame(self, entry: int) -> list[tuple[int, int]]:
        lo, hi = int(self.offsets[entry]), int(self.offsets[entry + 1])
        return list(zip(self.relations[lo:hi].tolist(), self.fillers[lo:hi].tolist()))

    @torch.no_grad()
    def stores(self) -> Tensor:
        """`(entries, d)` static stores of every entry (zero for an entry without edges)."""
        if self._stores is None:
            entries = int(self.offsets.size - 1)
            out = torch.zeros(entries, self.atomics.shape[1])
            have = np.flatnonzero(np.diff(self.offsets) > 0)
            for start in range(0, have.size, 8192):
                ids = torch.as_tensor(have[start:start + 8192])
                out[ids] = self.composer.raw_bundle(ids, uniform=True)[0].float()
            self._stores = out
        return self._stores

    @torch.no_grad()
    def frame_store(self, frame: Sequence[tuple[int, int]]) -> Tensor:
        relations = torch.tensor([r for r, _ in frame], dtype=torch.long)
        return self.composer.transform(relations, self.atomics[torch.tensor([f for _, f in frame])]).float().sum(0)

    def unbinder(self, method: str | None = None) -> Callable[[Tensor, Tensor], Tensor]:
        method = method or self.method
        if method == "exact":
            roles = getattr(self.composer.transform, "roles", None)
            if roles is None:
                raise ValueError("exact division is defined here for learned HRR roles only")
            return lambda r, v: spectral_inverse(v.float(), roles.detach()[r].float(), EXACT_RIDGE)
        return lambda r, v: self.composer.unbind(r, v.float(), method=method).float()

    @torch.no_grad()
    def scores(self, relations: Tensor, vectors: Tensor, *, method: str | None = None) -> Tensor:
        """`(n, atomics)` cosine of each unbound vector to every atomic (slotted: within the relation's slot), with the
        relation's non-candidates at −inf (typed cleanup)."""
        unbind = self.unbinder(method)
        queries = F.normalize(unbind(relations, vectors), dim=-1)
        if self.slot_masks is None:
            scores = queries @ F.normalize(self.atomics, dim=-1).T
        else:
            scores = torch.empty(queries.shape[0], self.atomics.shape[0])
            slots = self.slot_of[relations]
            for g in range(self.slot_masks.shape[0]):
                rows = (slots == g).nonzero(as_tuple=True)[0]
                if rows.numel():
                    scores[rows] = queries[rows] @ F.normalize(self.atomics * self.slot_masks[g], dim=-1).T
        return scores.masked_fill(~self.candidates[relations], float("-inf"))


def _rank(scores: Tensor, gold: Tensor, exclude: list[list[int]] | None = None) -> tuple[Tensor, Tensor]:
    mask = None
    if exclude is not None:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, others in enumerate(exclude):
            if others:
                mask[i, torch.tensor(others)] = True
    allowed = torch.isfinite(scores)
    return cl.filtered_ranks(scores.masked_fill(~allowed, -1e9), gold, allowed=allowed, exclude=mask, return_hits=True)


def _others(frame: Sequence[tuple[int, int]], relation: int, filler: int) -> list[int]:
    return sorted({f for r, f in frame if r == relation and f != filler})


# ---------------------------------------------------------------- fidelity by frame size


@torch.no_grad()
def fidelity(store: Store, entries: np.ndarray, subsets: np.ndarray, *, chunk: int) -> dict[str, Any]:
    """Typed filler recovery from the static stores by frame size, per unbinding method (+ exact division for HRR)."""
    composer = store.composer
    methods = [store.method] + [m for m in composer.transform.unbind_methods if m != store.method and store.method != "bundle"]
    if getattr(composer.transform, "family", "") in {"hrr", "hrr_identity"}:
        methods.append("exact")
    edges = bp.frame_edges(store.offsets, store.relations, store.fillers, entries)
    ids = torch.as_tensor(entries, dtype=torch.long)
    summed, _, edge_index, segments = composer.raw_bundle(ids, uniform=True)
    relations = torch.as_tensor(store.relations[edge_index.numpy()]); fillers = torch.as_tensor(store.fillers[edge_index.numpy()])
    exclusions, _ = bp.within_frame_exclusions(segments, relations, fillers)
    out: dict[str, Any] = {}
    owners = edges["entry"]
    for method in methods:
        unbinder = store.unbinder(method) if method == "exact" else None
        result = bp.decode_fillers(composer, summed.float(), segments, relations, fillers, store.candidates, exclusions, method=method,
                                   chunk=chunk, unbinder=unbinder)
        rr = (1.0 / result["typed_rank"]).numpy(); hit = result["typed_hit"].numpy()
        by_degree = {}
        for k in sorted(set(edges["degree"].tolist())):
            m = edges["degree"] == k
            by_degree[str(int(k))] = {"mrr": float(rr[m].mean()), "top1": float(hit[m].mean()), "edges": int(m.sum())}
        by_subset = {}
        edge_subset = subsets[owners]
        for s in (*bp.SUBSETS, "all"):
            m = np.ones_like(rr, dtype=bool) if s == "all" else edge_subset == s
            if m.any():
                by_subset[s] = {"mrr": float(np.mean(list(bp.entry_means(rr, owners, m).values()))), "top1": float(hit[m].mean()),
                                "edges": int(m.sum())}
        out[method] = {"by_degree": by_degree, "subsets": by_subset}
    return {"methods": methods, "results": out}


# ---------------------------------------------------------------- two-hop


def ontology_paths(store: Store, anchors: np.ndarray, atom_entry: np.ndarray, *, limit: int, seed: int) -> list[tuple[int, int, int, int, int, int]]:
    """(anchor, r1, f1, bridge, r2, f2): f1 names the bridge concept, whose frame holds (r2, f2); a seeded sample."""
    paths = []
    for e in anchors.tolist():
        for r1, f1 in store.frame(e):
            bridge = int(atom_entry[f1]) if f1 < atom_entry.size else -1
            if bridge < 0 or bridge == e:
                continue
            for r2, f2 in store.frame(bridge):
                paths.append((e, r1, f1, bridge, r2, f2))
    rng = np.random.default_rng(seed)
    if len(paths) > limit:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), limit, replace=False).tolist())]
    return paths


@torch.no_grad()
def chain_paths(store: Store, paths: Sequence[tuple[int, ...]], atom_entry: np.ndarray, *, anchor_stores: Tensor | None = None,
                bridge_store: Callable[[Tensor], Tensor] | None = None, chunk: int = 2048,
                options: Sequence[Sequence[int]] | None = None) -> dict[str, np.ndarray]:
    """Per path: first hop (rank, hit of f1), the chained second hop (rank, hit of f2; a bridge that names no concept
    fails), the second hop from the true bridge (`oracle`) and the soft chain. `anchor_stores` overrides the anchors'
    stores (new words; a global memory), `bridge_store(entries)` the bridges' (a global memory); `options` restricts the
    second hop's cleanup to the given atomics (the WP-UB items' options)."""
    if not paths:
        return {}
    stores = store.stores()
    get_bridge = bridge_store or (lambda entries: stores[entries])
    chainable = torch.as_tensor(atom_entry >= 0)
    atom_stores = torch.zeros(atom_entry.size, stores.shape[1])
    valid = np.flatnonzero(atom_entry >= 0)
    atom_stores[torch.as_tensor(valid)] = get_bridge(torch.as_tensor(atom_entry[valid]))
    out: dict[str, list[float]] = defaultdict(list)
    for start in range(0, len(paths), chunk):
        part = paths[start:start + chunk]
        anchors = torch.tensor([p[0] for p in part]); r1 = torch.tensor([p[1] for p in part]); f1 = torch.tensor([p[2] for p in part])
        bridges = torch.tensor([p[3] for p in part]); r2 = torch.tensor([p[4] for p in part]); f2 = torch.tensor([p[5] for p in part])
        first_vectors = anchor_stores[start:start + len(part)] if anchor_stores is not None else stores[anchors]
        first = store.scores(r1, first_vectors)
        exclude1 = [_others(store.frame(int(a)) if anchor_stores is None else [], int(r), int(f)) for a, r, f in zip(anchors, r1, f1)]
        rank1, hit1 = _rank(first, f1, exclude1)
        predicted = first.argmax(-1)
        bridge_hat = torch.as_tensor(atom_entry[predicted.numpy()])
        second_vectors = torch.zeros_like(first_vectors)
        ok = bridge_hat >= 0
        if bool(ok.any()):
            second_vectors[ok] = get_bridge(bridge_hat[ok])
        exclude2 = [_others(store.frame(int(b)), int(r), int(f)) for b, r, f in zip(bridges, r2, f2)]
        restrict = None
        if options is not None:
            restrict = torch.zeros(len(part), store.atomics.shape[0], dtype=torch.bool)
            for i, opts in enumerate(options[start:start + len(part)]):
                restrict[i, torch.tensor(list(opts))] = True
        def second_hop(vectors: Tensor) -> tuple[Tensor, Tensor]:
            scores = store.scores(r2, vectors)
            if restrict is not None:
                scores = scores.masked_fill(~restrict, float("-inf"))
            return _rank(scores, f2, None if restrict is not None else exclude2)
        rank2, hit2 = second_hop(second_vectors)
        rank2 = torch.where(ok, rank2, torch.full_like(rank2, float(store.atomics.shape[0])))
        hit2 = torch.where(ok, hit2, torch.zeros_like(hit2))
        oracle_rank, oracle_hit = second_hop(get_bridge(bridges))
        weights = torch.softmax(SOFT_BETA * first.masked_fill(~chainable[None], float("-inf")), -1).nan_to_num(0.0)
        soft_rank, soft_hit = second_hop(weights @ atom_stores)
        for key, value in (("hop1_rr", 1 / rank1), ("hop1_hit", hit1), ("bridge_found", ok.float()), ("chain_rr", 1 / rank2),
                           ("chain_hit", hit2), ("oracle_rr", 1 / oracle_rank), ("oracle_hit", oracle_hit), ("soft_rr", 1 / soft_rank),
                           ("soft_hit", soft_hit), ("hop1_correct", (predicted == f1).float())):
            out[key] += value.tolist()
    return {k: np.asarray(v) for k, v in out.items()}


def _mean(values: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
    return {k: float(v[mask].mean()) if mask is not None else float(v.mean()) for k, v in values.items() if v.size and (mask is None or mask.any())}


# ---------------------------------------------------------------- reverse lookup


@torch.no_grad()
def reverse_scores(store: Store, relations: Tensor, fillers: Tensor) -> Tensor:
    """`(queries, entries)`: `cos(c, T_r(a_F))` for every entry's static store (slotted: within the relation's slot)."""
    probes = store.composer.transform(relations, store.atomics[fillers]).float()
    stores = store.stores()
    if store.slot_masks is None:
        return F.normalize(probes, dim=-1) @ F.normalize(stores, dim=-1).T
    masks = store.slot_masks[store.slot_of[relations]].float()                  # (queries, d)
    numerator = F.normalize(probes, dim=-1) @ stores.T
    norms = torch.sqrt((stores.square()) @ masks.T).T.clamp_min(1e-12)          # (queries, entries): each store's slot norm
    return numerator / norms


@torch.no_grad()
def reverse_lookup(store: Store, queries: Sequence[tuple[int, int, int]], *, chunk: int = 256) -> dict[str, np.ndarray]:
    """Per (entry, relation, filler) query: the entry's filtered rank among every entry by `cos(c, T_r(a_F))` (entries whose
    frame also holds (r, F) removed)."""
    holders: dict[tuple[int, int], list[int]] = defaultdict(list)
    owner = np.repeat(np.arange(store.offsets.size - 1), np.diff(store.offsets))
    for e, r, f in zip(owner.tolist(), store.relations.tolist(), store.fillers.tolist()):
        holders[(r, f)].append(e)
    rr, hit = [], []
    for start in range(0, len(queries), chunk):
        part = queries[start:start + chunk]
        scores = reverse_scores(store, torch.tensor([q[1] for q in part]), torch.tensor([q[2] for q in part]))
        exclude = [sorted(set(holders[(q[1], q[2])]) - {q[0]}) for q in part]
        rank, h = _rank(scores, torch.tensor([q[0] for q in part]), exclude)
        rr += (1 / rank).tolist(); hit += h.tolist()
    return {"rr": np.asarray(rr), "hit": np.asarray(hit)}


# ---------------------------------------------------------------- capacity: one global holographic memory


def memory_keys(store: Store, atom_entry: np.ndarray, *, seed: int) -> Tensor:
    """Unitary keys per entry: the trained atomic that names the entry, else a seeded random vector, projected to unit
    spectrum magnitude (so key unbinding is exact and the loss comes from superposition alone)."""
    entries = int(store.offsets.size - 1)
    generator = torch.Generator().manual_seed(seed + 101)
    raw = torch.randn(entries, store.atomics.shape[1], generator=generator)
    named = np.flatnonzero(atom_entry >= 0)
    raw[torch.as_tensor(atom_entry[named])] = store.atomics[torch.as_tensor(named)]
    return UnitaryHRRAlgebra.make_role(raw)


@torch.no_grad()
def global_memory(store: Store, members: Tensor, keys: Tensor) -> Tensor:
    """`G = Σ_j k_j ⊛ ĉ_j` over the members' normalized stores."""
    stores = F.normalize(store.stores()[members], dim=-1)
    algebra = HRRAlgebra()
    total = torch.zeros(stores.shape[1])
    for start in range(0, members.numel(), 8192):
        total += algebra.bind(keys[members[start:start + 8192]], stores[start:start + 8192]).sum(0)
    return total


@torch.no_grad()
def capacity(store: Store, entries: np.ndarray, atom_entry: np.ndarray, settings: ChainSettings) -> dict[str, Any]:
    """Filler recovery from one global memory of N concepts (a nested random order), against N."""
    total = int(store.offsets.size - 1)
    order = torch.as_tensor(np.random.default_rng(settings.seed + 7).permutation(np.flatnonzero(np.diff(store.offsets) > 0)))
    keys = memory_keys(store, atom_entry, seed=settings.seed)
    algebra = HRRAlgebra()
    norms = store.stores().norm(dim=-1).clamp_min(1e-12)        # stores enter normalized; their scale is restored on recall
    sizes = sorted({n for n in MEMORY_SIZES if n < order.numel()} | {int(order.numel())})
    out = {}
    for n in sizes:
        members = order[:n]
        memory = global_memory(store, members, keys)
        targets = members[:settings.memory_targets]
        recovered = algebra.unbind(memory.expand(targets.numel(), -1), keys[targets]) * norms[targets, None]
        rows, rel, fil, excl = [], [], [], []
        for i, e in enumerate(targets.tolist()):
            frame = store.frame(e)
            for r, f in frame:
                rows.append(i); rel.append(r); fil.append(f); excl.append(_others(frame, r, f))
        if not rows:
            continue
        scores = store.scores(torch.tensor(rel), recovered[torch.tensor(rows)])
        rank, hit = _rank(scores, torch.tensor(fil), excl)
        out[str(n)] = {"mrr": float((1 / rank).mean()), "top1": float(hit.mean()), "targets": int(targets.numel()),
                       "edges": len(rows)}
    return {"entries": total, "sizes": out, "keys": "unitary projection of the naming atomic (else seeded random)"}


@torch.no_grad()
def global_two_hop(store: Store, paths: Sequence[tuple[int, ...]], atom_entry: np.ndarray, settings: ChainSettings) -> dict[str, float]:
    """Two-hop through one global memory of every concept: the anchor's and the bridges' stores recovered by their keys."""
    if not paths:
        return {}
    order = torch.as_tensor(np.flatnonzero(np.diff(store.offsets) > 0))
    keys = memory_keys(store, atom_entry, seed=settings.seed)
    memory = global_memory(store, order, keys)
    algebra = HRRAlgebra()
    norms = store.stores().norm(dim=-1).clamp_min(1e-12)
    recall = lambda entries: algebra.unbind(memory.expand(entries.numel(), -1), keys[entries]) * norms[entries, None]
    part = list(paths[:settings.global_paths])
    anchors = recall(torch.tensor([p[0] for p in part]))
    result = chain_paths(store, part, atom_entry, anchor_stores=anchors, bridge_store=recall, chunk=settings.chunk)
    return {**_mean(result), "paths": len(part), "memory_concepts": int(order.numel())}


# ---------------------------------------------------------------- path order (decision 61b)


@torch.no_grad()
def path_order(transform: Any, atomics: Tensor, *, samples: int, seed: int, method: str | None = None,
               slot_of: Tensor | None = None) -> dict[str, Any]:
    """Two paths over the same relations in opposite orders in one vector (plus m distractor paths): the share of samples
    in which unbinding in the order r1, r2 scores f above g (ties count 0.5)."""
    method = method or readout_method(transform)
    generator = torch.Generator().manual_seed(seed + 31)
    count, atoms = transform.relation_count, atomics.shape[0]
    out: dict[str, Any] = {"method": method, "by_distractors": {}}
    unit = F.normalize(atomics.float(), dim=-1)
    for m in DISTRACTORS:
        r1 = torch.randint(0, count, (samples,), generator=generator)
        r2 = (r1 + torch.randint(1, count, (samples,), generator=generator)) % count if count > 1 else r1
        if slot_of is not None:                                            # slotted: chained binding stays in one slot
            same = slot_of[r1] == slot_of[r2]
            r1, r2 = r1[same], r2[same]
            if r1.numel() == 0:
                out["by_distractors"][str(m)] = None
                continue
        n = r1.numel()
        f = torch.randint(0, atoms, (n,), generator=generator)
        g = (f + torch.randint(1, atoms, (n,), generator=generator)) % atoms
        bundle = transform(r2, transform(r1, unit[f])) + transform(r1, transform(r2, unit[g]))
        for _ in range(m):
            a = torch.randint(0, count, (n,), generator=generator); b = torch.randint(0, count, (n,), generator=generator)
            h = torch.randint(0, atoms, (n,), generator=generator)
            bundle = bundle + transform(b, transform(a, unit[h]))
        decoded = transform.unbind(r1, transform.unbind(r2, bundle, method=method), method=method)
        decoded = F.normalize(decoded.float(), dim=-1)
        sf, sg = (decoded * unit[f]).sum(-1), (decoded * unit[g]).sum(-1)
        close = (sf - sg).abs() <= 1e-5
        score = torch.where(close, torch.full_like(sf, 0.5), (sf > sg).float())
        out["by_distractors"][str(m)] = {"accuracy": float(score.mean()), "samples": int(n), "ties": float(close.float().mean())}
    return out


# ---------------------------------------------------------------- WP-UB items


def _item_atoms(ontology: dict[str, Any], track: str | None, family: str) -> tuple[Callable[[str, str], int | None], dict[str, int]]:
    """(answer text → atomic of a relation's pool, concept name → entry)."""
    from ..evaluation.channel_probes import load_alias_table
    from . import e9_ontology_edit as edit
    from .e9_tracks import ensure_alias_table, lexicon_for, track_spec
    spec = track_spec(track, family)
    table = load_alias_table(ensure_alias_table(spec))
    lexicon = lexicon_for(spec, ontology)
    view = edit.OntologyView(ontology, table, lexicon)
    pools: dict[str, set[int]] = defaultdict(set)
    for r, f in zip(view.relations.tolist(), view.fillers.tolist()):
        pools[view.relation_names[r]].add(f)
    cache: dict[str, dict[str, int]] = {}

    templated = getattr(lexicon, "templates", {})

    def answer_atom(relation: str, text: str) -> int | None:
        """The atomic whose answer wording is `text` (as `e9_understanding._answer` words options: the relation's answer
        template, else a space and the filler's text)."""
        if relation not in cache:
            cache[relation] = {}
            for a in sorted(pools.get(relation, ())):
                if view.text(a):
                    wording = lexicon.answer(relation, view.text(a)) if relation in templated else " " + view.text(a)
                    cache[relation].setdefault(wording, a)
        return cache[relation].get(text)

    names = [str(n) for n in ontology.get("concept_names") or []]
    concept_entry = {}
    for e, members in enumerate(table.entry_concepts):
        if len(members) == 1:
            concept_entry.setdefault(names[members[0]], e)
    return answer_atom, concept_entry


@torch.no_grad()
def item_outcomes(store: Store, ontology: dict[str, Any], items_dir: Path, atom_entry: np.ndarray, *, track: str | None,
                  family: str, settings: ChainSettings) -> list[dict[str, Any]]:
    """Per WP-UB two-hop and reverse item: the algebraic outcome (local stores; two-hop also through the global memory)."""
    from . import e9_ontology_edit as edit
    from . import e9_understanding as und
    _, concepts, items = und.load_items(items_dir)
    answer_atom, concept_entry = _item_atoms(ontology, track, family)
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    by_concept = {c["concept"]: c for c in concepts}
    stores = store.stores()

    def concept_store(cid: str) -> Tensor:
        c = by_concept[cid]
        if c.get("entry") is not None:
            return stores[int(c["entry"])]
        return store.frame_store(edit.resolve_frame(c["frame"], relation_id, atomic_id))

    out = []
    two_hop = [i for i in items if i["family"] == "two_hop" and i["test"] == "two_hop"]
    paths, options, kept = [], [], []
    for item in two_hop:
        r1, r2 = (relation_id[r] for r in item["meta"]["path"])
        bridge = concept_entry.get(item["meta"]["bridge"])
        answer = atomic_id.get(item["meta"]["answer"])
        bridge_atom = next((a for a in np.flatnonzero(atom_entry == bridge).tolist()), None) if bridge is not None else None
        opts = [answer_atom(item["meta"]["path"][1], c) for c in item["candidates"]]
        if bridge is None or answer is None or bridge_atom is None or any(o is None for o in opts):
            continue
        paths.append((-1, r1, bridge_atom, bridge, r2, answer)); options.append(opts); kept.append(item)
    if paths:
        anchors = torch.stack([concept_store(i["anchor"]) for i in kept])
        local = chain_paths(store, paths, atom_entry, anchor_stores=anchors, options=options, chunk=settings.chunk)
        for j, item in enumerate(kept):
            out.append({"id": item["id"], "family": "two_hop", "subset": item["subset"], "anchor": item["anchor"],
                        **{k: float(v[j]) for k, v in local.items()}})
    for item in (i for i in items if i["family"] == "reverse"):
        relation = relation_id.get(item["relation"]); filler = atomic_id.get(item["meta"]["filler"])
        if relation is None or filler is None:
            continue
        probe = F.normalize(store.composer.transform(torch.tensor([relation]), store.atomics[[filler]]).float(), dim=-1)[0]
        if store.slot_masks is not None:
            mask = store.slot_masks[store.slot_of[relation]].float()
            score = lambda v: float((F.normalize(v * mask, dim=-1) * probe).sum())
        else:
            score = lambda v: float((F.normalize(v, dim=-1) * probe).sum())
        sx, sy = score(concept_store(item["slots"]["x"])), score(concept_store(item["slots"]["y"]))
        out.append({"id": item["id"], "family": "reverse", "subset": item["subset"], "anchor": item["anchor"],
                    "pair_correct": 0.5 if abs(sx - sy) <= 1e-6 else float(sx > sy)})
    return out


# ---------------------------------------------------------------- run


def chain_run(run_dir: Path, output: Path | None = None, *, items: Path | None = None, licensed: bool = False,
              overwrite: bool = False, settings: ChainSettings | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    settings = settings or ChainSettings()
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / OUTPUT
    config_record = {"experiment": "e9-binding-chain", "run": str(run_dir), "items": str(items) if items else None, "licensed": licensed,
                     "settings": asdict(settings)}
    if overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config_record)
    started = time.monotonic()
    torch.manual_seed(settings.seed)
    composer, config, ontology = load_composer(run_dir)
    store = Store(composer, chunk=settings.chunk)
    atom_entry = atom_entries(ontology)
    degrees = np.diff(store.offsets)
    subsets = bp.entry_subsets(ontology)
    entries = bp.select_entries(ontology, degrees, cap=settings.cap, seed=settings.seed)
    from .e9_plan import stem_model
    record: dict[str, Any] = {"schema": SCHEMA, "run": str(run_dir), "stem": run_dir.name,
                              "model": stem_model(run_dir.name) if re.search(r"-s\d+$", run_dir.name) else run_dir.name,
                              "track": config.get("e9_track"), "operator": composer.operator, "family": composer.transform.family,
                              "method": store.method, "licensed": licensed, "chainable_atoms": int((atom_entry >= 0).sum())}
    log(f"{run_dir.name}: fidelity")
    record["fidelity"] = fidelity(store, entries, subsets, chunk=settings.chunk)
    log(f"{run_dir.name}: two-hop")
    paths = ontology_paths(store, entries, atom_entry, limit=settings.paths, seed=settings.seed)
    local = chain_paths(store, paths, atom_entry, chunk=settings.chunk)
    path_subsets = np.asarray([subsets[p[0]] for p in paths]) if paths else np.zeros(0, dtype=str)
    record["two_hop"] = {"paths": len(paths), "all": _mean(local) if paths else {},
                         "subsets": {s: _mean(local, path_subsets == s) for s in bp.SUBSETS if (path_subsets == s).any()}}
    record["two_hop_global"] = global_two_hop(store, paths, atom_entry, settings)
    record["two_hop_local_same_paths"] = _mean({k: v[:settings.global_paths] for k, v in local.items()}) if paths else {}
    log(f"{run_dir.name}: reverse")
    rng = np.random.default_rng(settings.seed + 3)
    queries = [(e, r, f) for e in entries.tolist() for r, f in store.frame(e)]
    if len(queries) > settings.reverse:
        queries = [queries[i] for i in sorted(rng.choice(len(queries), settings.reverse, replace=False).tolist())]
    reverse = reverse_lookup(store, queries)
    relations_of: dict[int, set[int]] = defaultdict(set)
    for r, f in zip(store.relations.tolist(), store.fillers.tolist()):
        relations_of[f].add(r)
    ambiguous = np.asarray([len(relations_of[q[2]]) >= 2 for q in queries])
    query_subsets = np.asarray([subsets[q[0]] for q in queries])
    record["reverse"] = {"queries": len(queries), "all": _mean(reverse),
                         "ambiguous_filler": _mean(reverse, ambiguous) if ambiguous.any() else None,
                         "unambiguous_filler": _mean(reverse, ~ambiguous) if (~ambiguous).any() else None,
                         "subsets": {s: _mean(reverse, query_subsets == s) for s in bp.SUBSETS if (query_subsets == s).any()}}
    log(f"{run_dir.name}: capacity")
    record["capacity"] = capacity(store, entries, atom_entry, settings)
    record["path_order"] = path_order(composer.transform, store.atomics, samples=settings.path_order, seed=settings.seed,
                                      slot_of=store.slot_of)
    outcomes = []
    if items is not None and not licensed:
        log(f"{run_dir.name}: WP-UB items")
        outcomes = item_outcomes(store, ontology, Path(items), atom_entry, track=config.get("e9_track"),
                                 family=config.get("e9_family", "smollm2"), settings=settings)
        two = [o for o in outcomes if o["family"] == "two_hop"]
        rev = [o for o in outcomes if o["family"] == "reverse"]
        record["items"] = {"two_hop": {"items": len(two), **({k: float(np.mean([o[k] for o in two])) for k in ("hop1_correct", "chain_hit", "oracle_hit", "soft_hit")} if two else {})},
                           "reverse": {"items": len(rev), "pair_correct": float(np.mean([o["pair_correct"] for o in rev])) if rev else None}}
    record["seconds"] = round(time.monotonic() - started, 1)
    if outcomes:
        from .e9_understanding import write_jsonl_gz
        write_jsonl_gz(output / "items.jsonl.gz", outcomes)
    if not licensed:                          # per path and per query, for the paired analyses (§14); ids only
        arrays: dict[str, np.ndarray] = {}
        if paths:
            arrays.update({f"path_{k}": np.asarray([p[i] for p in paths], dtype=np.int64)
                           for i, k in enumerate(("anchor", "r1", "f1", "bridge", "r2", "f2"))})
            arrays.update({f"path_{k}": v.astype(np.float32) for k, v in local.items()})
            global_part = list(paths[:settings.global_paths])
            norms_order = torch.as_tensor(np.flatnonzero(np.diff(store.offsets) > 0))
            keys = memory_keys(store, atom_entry, seed=settings.seed)
            memory = global_memory(store, norms_order, keys)
            norms = store.stores().norm(dim=-1).clamp_min(1e-12)
            recall = lambda e: HRRAlgebra().unbind(memory.expand(e.numel(), -1), keys[e]) * norms[e, None]
            through = chain_paths(store, global_part, atom_entry, anchor_stores=recall(torch.tensor([p[0] for p in global_part])),
                                  bridge_store=recall, chunk=settings.chunk)
            arrays.update({f"global_{k}": v.astype(np.float32) for k, v in through.items()})
        if queries:
            arrays.update({"query_entry": np.asarray([q[0] for q in queries]), "query_relation": np.asarray([q[1] for q in queries]),
                           "query_filler": np.asarray([q[2] for q in queries]), "query_rr": reverse["rr"].astype(np.float32),
                           "query_hit": reverse["hit"].astype(np.float32), "query_ambiguous": ambiguous})
        if arrays:
            np.savez_compressed(output / "paths.npz", **arrays)
    write_json(output / "summary.json", json_ready(record))
    (output / "report.md").write_text(render(record))
    finish_output(output, config_record, git_at_start=git_at_start, device="cpu")
    return record


def render(record: dict[str, Any]) -> str:
    lines = [f"# Binding chain — {record['stem']} (`{record['family']}`, unbinding `{record['method']}`)", ""]
    fid = record.get("fidelity", {})
    if fid:
        lines += ["## Fidelity (typed MRR by frame size)", "", "| method | " + " | ".join(sorted(next(iter(fid["results"].values()))["by_degree"], key=int)) + " |"]
        lines.append("|---|" + "---:|" * len(next(iter(fid["results"].values()))["by_degree"]))
        for method, block in fid["results"].items():
            lines.append(f"| {method} | " + " | ".join(f"{v['mrr']:.3f}" for _, v in sorted(block["by_degree"].items(), key=lambda kv: int(kv[0]))) + " |")
        lines.append("")
    two = record.get("two_hop", {}).get("all") or {}
    if two:
        lines += ["## Two-hop (ontology paths)", "", f"{record['two_hop']['paths']} paths: first hop top-1 {two.get('hop1_hit', 0):.3f}, chained "
                  f"top-1 {two.get('chain_hit', 0):.3f}, from the true bridge {two.get('oracle_hit', 0):.3f}, soft chain {two.get('soft_hit', 0):.3f}; "
                  f"through one global memory: {record.get('two_hop_global', {}).get('chain_hit', float('nan')):.3f}.", ""]
    rev = record.get("reverse", {}).get("all") or {}
    if rev:
        lines += [f"Reverse lookup ({record['reverse']['queries']} queries over every concept): MRR {rev['rr']:.3f}, top-1 {rev['hit']:.3f}.", ""]
    cap = record.get("capacity", {}).get("sizes", {})
    if cap:
        lines += ["## Global memory (typed MRR against N)", "", "| N | " + " | ".join(cap) + " |", "|---|" + "---:|" * len(cap),
                  "| MRR | " + " | ".join(f"{v['mrr']:.3f}" for v in cap.values()) + " |", ""]
    order = record.get("path_order", {}).get("by_distractors", {})
    if order:
        lines += ["Path order (r1, r2 vs r2, r1 in one vector): " + ", ".join(f"m = {m}: {v['accuracy']:.3f}" for m, v in order.items() if v), ""]
    return "\n".join(lines) + "\n"


def load_chain(folder: Path) -> dict[str, Any] | None:
    """`{"summary", "paths" (npz arrays or None), "items" (list)}` of a finished chain job; None if absent."""
    folder = Path(folder)
    if not (folder / "summary.json").exists():
        return None
    out: dict[str, Any] = {"summary": json.loads((folder / "summary.json").read_text()), "paths": None, "items": []}
    if (folder / "paths.npz").exists():
        with np.load(folder / "paths.npz") as data:
            out["paths"] = {k: data[k] for k in data.files}
    if (folder / "items.jsonl.gz").exists():
        from .e9_understanding import read_jsonl
        out["items"] = read_jsonl(folder / "items.jsonl")
    return out


# ---------------------------------------------------------------- synthetic sweep (random vectors; no checkpoint)


SWEEP_FAMILIES = ("unitary_hrr", "hrr", "hrr_exact", "spectral_bounded", "spectral_bounded_exact", "block_unitary", "translation", "additive")
SWEEP_DIMENSIONS = (64, 128, 256, 512, 1024, 2048)
SWEEP_LOADS = (1, 2, 4, 8, 16, 32, 64)


def _sweep_transform(family: str, relations: int, dimension: int, generator: torch.Generator) -> tuple[Any, str | None]:
    from ..relations import create_composition_operator
    base = family.replace("_exact", "")
    transform = create_composition_operator(base, relations, dimension)
    with torch.no_grad():
        if base == "spectral_bounded":
            transform.magnitude_logits.copy_((torch.rand(transform.magnitude_logits.shape, generator=generator) * 2 - 1) * 4)
    method = "exact" if family.endswith("_exact") else None
    return transform, method


@torch.no_grad()
def sweep(*, atoms: int = 4096, relations: int = 32, trials: int = 64, seed: int = 0, dimensions: Sequence[int] = SWEEP_DIMENSIONS,
          loads: Sequence[int] = SWEEP_LOADS, families: Sequence[str] = SWEEP_FAMILIES) -> dict[str, Any]:
    """Local filler recovery (top-1 over every atomic) by dimension × load per family; one global memory of N k = 8 stores
    at d = 256; path order at d = 256. Random vectors only."""
    out: dict[str, Any] = {"atoms": atoms, "relations": relations, "trials": trials, "seed": seed, "local": {}, "global": {}, "path_order": {}}
    for family in families:
        out["local"][family] = {}
        for d in dimensions:
            generator = torch.Generator().manual_seed(seed + d)
            torch.manual_seed(seed + d)
            transform, method = _sweep_transform(family, relations, d, generator)
            if method == "exact" and family.startswith("hrr"):
                roles = transform.roles
                unbind = lambda r, v, roles=roles: spectral_inverse(v, roles[r], EXACT_RIDGE)
            else:
                unbind = lambda r, v, t=transform, m=method: t.unbind(r, v, method=m or readout_method(t))
            dictionary = F.normalize(torch.randn(atoms, d, generator=generator), dim=-1)
            row = {}
            for k in loads:
                if k > relations:
                    continue
                rel = torch.stack([torch.randperm(relations, generator=generator)[:k] for _ in range(trials)])
                fil = torch.stack([torch.randperm(atoms, generator=generator)[:k] for _ in range(trials)])
                bound = transform(rel.reshape(-1), dictionary[fil.reshape(-1)]).reshape(trials, k, d).sum(1)
                decoded = unbind(rel.reshape(-1), bound.repeat_interleave(k, 0))
                index, _ = cl.nearest(decoded, dictionary)
                row[str(k)] = float((index == fil.reshape(-1)).float().mean())
            out["local"][family][str(d)] = row
    d = 256
    for family in ("unitary_hrr", "hrr", "block_unitary", "additive"):
        generator = torch.Generator().manual_seed(seed + 999)
        torch.manual_seed(seed + 999)
        transform, _ = _sweep_transform(family, relations, d, generator)
        dictionary = F.normalize(torch.randn(atoms, d, generator=generator), dim=-1)
        keys = UnitaryHRRAlgebra.make_role(torch.randn(2048, d, generator=generator))
        algebra = HRRAlgebra()
        rel = torch.stack([torch.randperm(relations, generator=generator)[:8] for _ in range(2048)])
        fil = torch.stack([torch.randperm(atoms, generator=generator)[:8] for _ in range(2048)])
        stores = F.normalize(transform(rel.reshape(-1), dictionary[fil.reshape(-1)]).reshape(2048, 8, d).sum(1), dim=-1)
        row = {}
        for n in (1, 4, 16, 64, 256, 1024, 2048):
            memory = algebra.bind(keys[:n], stores[:n]).sum(0)
            targets = min(n, 64)
            recovered = algebra.unbind(memory.expand(targets, -1), keys[:targets])
            decoded = transform.unbind(rel[:targets].reshape(-1), recovered.repeat_interleave(8, 0), method=readout_method(transform))
            index, _ = cl.nearest(decoded, dictionary)
            row[str(n)] = float((index == fil[:targets].reshape(-1)).float().mean())
        out["global"][family] = row
    for family in ("unitary_hrr", "hrr", "spectral_bounded", "block_unitary", "translation", "additive"):
        generator = torch.Generator().manual_seed(seed + 555)
        torch.manual_seed(seed + 555)
        transform, _ = _sweep_transform(family, relations, d, generator)
        dictionary = F.normalize(torch.randn(atoms, d, generator=generator), dim=-1)
        out["path_order"][family] = path_order(transform, dictionary, samples=1000, seed=seed)
    return out


def render_sweep(result: dict[str, Any]) -> str:
    lines = ["# Binding algebra on random vectors (synthetic sweep; not a result about trained models)", "",
             f"{result['atoms']} random atomics, {result['relations']} relations, {result['trials']} frames per cell; top-1 filler recovery "
             "after unbinding one role from a bundle of k bound pairs (cleanup over every atomic).", ""]
    for family, by_d in result["local"].items():
        loads = sorted({k for row in by_d.values() for k in row}, key=int)
        lines += [f"## {family}", "", "| d | " + " | ".join(f"k={k}" for k in loads) + " |", "|---|" + "---:|" * len(loads)]
        for d, row in by_d.items():
            lines.append(f"| {d} | " + " | ".join(f"{row[k]:.2f}" if k in row else "–" for k in loads) + " |")
        lines.append("")
    if result["global"]:
        sizes = list(next(iter(result["global"].values())))
        lines += ["## One global memory of N concepts (k = 8 each, d = 256)", "", "| family | " + " | ".join(f"N={n}" for n in sizes) + " |",
                  "|---|" + "---:|" * len(sizes)]
        lines += [f"| {f} | " + " | ".join(f"{row[n]:.2f}" for n in sizes) + " |" for f, row in result["global"].items()]
        lines.append("")
    if result["path_order"]:
        lines += ["## Path order in one vector (d = 256; accuracy of r1, r2 against r2, r1)", "", "| family | " +
                  " | ".join(f"m={m}" for m in DISTRACTORS) + " |", "|---|" + "---:|" * len(DISTRACTORS)]
        for f, block in result["path_order"].items():
            lines.append(f"| {f} | " + " | ".join(f"{block['by_distractors'][str(m)]['accuracy']:.2f}" for m in DISTRACTORS) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- queue


def chain_command(run_dir: Path, *, python: str = sys.executable, items: Path | None = None, licensed: bool = False) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_binding_chain", "chain", "--run", str(run_dir), "--overwrite",
            *(["--items", str(items)] if items else []), *(["--licensed"] if licensed else [])]


def queue_stage(stage: str, *, priority: int = 51, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                items: Path | None = None, root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None,
                licensed: bool = False, dry_run: bool = False) -> list[dict[str, Any]]:
    """One CPU-lane chain job per config of `stage` whose channel composes (`<stage>-<stem>-binding-chain`, idempotent)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import _config_host, stage_python, stem_model
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        config = yaml.safe_load(path.read_text())
        model = stem_model(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if (config.get("channel") or {}).get("mode") != "compose":
            continue
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        jobs.append({"name": f"{stage}-{path.stem}-{OUTPUT}", "priority": int(priority), "model": model,
                     "command": chain_command(Path(root) / "runs" / stage / path.stem, python=python, items=items, licensed=licensed)})
    if dry_run:
        return jobs
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"], min_free_gb=1,
                env={"PYTHONPATH": "src"}, resume_args=[], lane="cpu")
            queued.append(job)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("chain", help="two-hop, reverse lookup, capacity and path order on one run's composer (CPU)")
    run.add_argument("--run", type=Path, required=True); run.add_argument("--output", type=Path, default=None)
    run.add_argument("--items", type=Path, default=None, help="WP-UB understanding items (two-hop and reverse)")
    run.add_argument("--licensed", action="store_true"); run.add_argument("--overwrite", action="store_true")
    run.add_argument("--cap", type=int, default=ChainSettings.cap); run.add_argument("--paths", type=int, default=ChainSettings.paths)
    run.add_argument("--reverse", type=int, default=ChainSettings.reverse)
    sw = sub.add_parser("sweep", help="the same algebra on random vectors (no checkpoint)")
    sw.add_argument("--output", type=Path, required=True); sw.add_argument("--trials", type=int, default=64)
    sw.add_argument("--seed", type=int, default=0); sw.add_argument("--overwrite", action="store_true")
    queue = sub.add_parser("queue", help="queue one CPU-lane chain job per composing config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--priority", type=int, default=51)
    queue.add_argument("--models", nargs="*", default=None); queue.add_argument("--seeds", type=int, nargs="*", default=None)
    queue.add_argument("--items", type=Path, default=None); queue.add_argument("--root", type=Path, default=ROOT)
    queue.add_argument("--licensed", action="store_true"); queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "chain":
        settings = ChainSettings(cap=args.cap, paths=args.paths, reverse=args.reverse)
        record = chain_run(args.run, args.output, items=args.items, licensed=args.licensed, overwrite=args.overwrite, settings=settings)
        print(json.dumps({"run": record["stem"], "seconds": record["seconds"]}))
    elif args.command == "sweep":
        config = {"experiment": "e9-binding-sweep", "trials": args.trials, "seed": args.seed}
        if args.overwrite:
            for name in ("sweep.json", "report.md", "resolved_config.yaml", "manifest.json"):
                if (args.output / name).is_file():
                    (args.output / name).unlink()
        git_at_start = start_output(args.output, config)
        result = sweep(trials=args.trials, seed=args.seed)
        write_json(args.output / "sweep.json", json_ready(result))
        (args.output / "report.md").write_text(render_sweep(result))
        finish_output(args.output, config, git_at_start=git_at_start, device="cpu")
        print(json.dumps({"output": str(args.output)}))
    else:
        jobs = queue_stage(args.stage, priority=args.priority, models=args.models, seeds=args.seeds, items=args.items, root=args.root,
                           licensed=args.licensed, dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
