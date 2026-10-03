"""Experiment E10 — self-learned semantics (hypothesis H-H): E10.0 synthetic, E10.1 WordNet.

A learnable ontology (`LearnableOntologyComposer`) regularized toward a curated prior, run on a world
with known gold so every self-supervised decision can be audited afterwards:

- **a** learnability ablation: {atomics, relations, mapping, frames} × {fixed, l2, free}, full grid
  (incl. leave-one-component-out), with 30% erased edges, 5% spurious asserted edges, noisy priors;
- **b** erasure & recovery: erase 10–70% of edges (fixed principal portion 90–30%), learn candidate
  masses → recovery P/R/F1/AUC vs random and filler-frequency baselines;
- **c** blank-relation discovery: hidden relations absent (candidates offered) or collapsed
  sub-types; additive curriculum (± rule enforcement) vs all slots at once vs M3 splitting vs the
  stem-cell pool vs oracle labels; riddle-style structural interpretation (E10.6); human-likeness
  trajectories (overextension → refinement);
- **d** self-tested acceptance: edge and slot hypotheses accepted by held-out tests, scored against
  gold and a hidden audit split; random acceptance at the matched rate; in-sample acceptance as the
  confirmation-bias control;
- **e** new-word frame inference from k ∈ {1, 2, 4, 8} observations, zero-shot composition;
- **seed** (E10.4) start from empty / core / noisy / 30% / full ontologies → recovered ontology and
  reasoning probes (multi-hop is-a, transitivity, inverse and symmetry consistency);
- **dream** (E10.5) offline revision passes after injected corruptions (wrong edges, merged
  relations, a wrongly crystallized relation) at several frequencies;
- **variety** (E10.7) source variety per concept vs wrong self-acceptance on the hidden audit;
- **continual** (E10.8) relations arriving in stages: additive ± dreaming vs plastic vs all at once.

Learning is self-supervised: gold relation labels, gold pairs and the audit observations are read
only by the evaluation code in this module (functions prefixed `eval_` / `_gold`), never by the
learning loop (`vsa_embed.ontology_discovery`, `vsa_embed.self_test`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from vsa_embed.developmental import DevelopmentalConfig, DevelopmentalDictionary
from vsa_embed.frame_inference import candidate_dictionary, compose_frame, frame_scores, infer_frames
from vsa_embed.learnable_ontology import ORIGIN, EdgeTable, LearnabilityConfig, LearnableOntologyComposer, train_composer
from vsa_embed.ontology_discovery import (
    DiscoverySettings, DreamSettings, LearningContext, additive_curriculum, all_at_once, dream_pass, edge_pairs,
    own_relations, revisit_crystallized,
)
from vsa_embed.ontology_hypotheses import (
    HypothesisSettings, StructuralHypothesis, best_match, compose_pairs, generate_hypotheses, inverse, jaccard,
    partition_scores, property_holds, symmetry, transitivity,
)
from vsa_embed.provenance import apply_thread_setting, prepare_output_dir, write_run_metadata
from vsa_embed.self_test import (
    HeldOutData, NodeMap, accept_edges, base_rows, bootstrap_lower, edge_contributions, edit_effects, score_hypotheses,
)
from vsa_embed.statistics import mean_confidence_interval, wilson_interval
from vsa_embed.synthetic_ontology import OntologyWorld, make_ontology_world

SETTINGS = ("fixed", "l2", "free")
COMPONENTS = ("atomics", "relations", "mapping", "frames")


# =====================================================================================================
# worlds and scenarios
# =====================================================================================================

def build_world(config: dict[str, Any], seed: int, **overrides: Any) -> OntologyWorld:
    spec = {**config["world"], **overrides}
    kind = spec.pop("kind", "synthetic")
    if kind == "synthetic":
        return make_ontology_world(seed=seed, **spec)
    if kind == "wordnet":
        from vsa_embed.experiments.e10_wordnet import build_wordnet_world
        return build_wordnet_world(spec, seed)
    raise ValueError(f"unknown world kind {kind!r}")


@dataclass
class Scenario:
    name: str
    table: EdgeTable
    relation_names: list[str]                 # the model's names for its seed relations (columns)
    column_world: list[int | None]            # gold world relation of each column (evaluation only)
    relation_prior: torch.Tensor | None
    hidden: list[int] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def _framed(world: OntologyWorld) -> set[int]:
    return set(torch.cat([world.splits["train"], world.splits["validation"], world.splits["test"]]).tolist())


def _pools(world: OntologyWorld) -> dict[int, list[int]]:
    return {r: sorted(set(world.fillers[world.relations == r].tolist())) for r in range(len(world.relation_names))}


def _relation_prior(world: OntologyWorld, columns: list[int | list[int] | None], seed: int) -> torch.Tensor | None:
    if world.relation_prior is None:
        return None
    g = torch.Generator().manual_seed(seed + 3)
    rows = []
    for col in columns:
        if col is None:
            rows.append(F.normalize(torch.randn(world.dimension, generator=g), dim=0))
        elif isinstance(col, list):
            rows.append(F.normalize(world.relation_prior[col].mean(0), dim=0))
        else:
            rows.append(world.relation_prior[col])
    return torch.stack(rows)


def erasure_scenario(world: OntologyWorld, *, rate: float, seed: int, distractors: int = 2, spurious: float = 0.0,
                     relation_confusion: float = 0.5, max_slots: int = 0, extra_candidates: list | None = None,
                     erase_relations: list[str] | None = None) -> Scenario:
    """All relations known; a fraction `rate` of edges erased. Training heads get typed candidates:
    the erased edge plus `distractors` per erased edge (random filler of the same relation, or the
    same filler under another relation), never a gold edge. `spurious` adds wrong asserted edges."""
    rng = random.Random(seed)
    framed, train = _framed(world), set(world.splits["train"].tolist())
    gold = set(world.edge_list())
    pools = _pools(world)
    relations = list(range(len(world.relation_names)))
    erasable = set(relations) if not erase_relations else {world.relation_id(n) for n in erase_relations}
    asserted, candidates, erased = [], [], []
    for h, r, a in world.edge_list():
        if h not in framed:
            continue
        if r in erasable and rng.random() < rate:
            if h in train:
                erased.append((h, r, a)); candidates.append((h, r, a))
        else:
            asserted.append((h, r, a))
    seen = set(candidates)
    for h, r, a in erased:
        for _ in range(distractors):
            for _attempt in range(20):
                if rng.random() < relation_confusion:
                    others = [q for q in erasable if q != r and a in set(pools[q])]
                    if not others:
                        continue
                    edge = (h, rng.choice(others), a)
                else:
                    edge = (h, r, rng.choice(pools[r]))
                if edge not in gold and edge not in seen:
                    seen.add(edge); candidates.append(edge); break
    for edge in extra_candidates or []:
        if edge not in gold and edge not in seen and edge[0] in train:
            seen.add(edge); candidates.append(edge)
    wrong = []
    framed_list = sorted(framed)
    erasable_list = sorted(erasable)
    for _ in range(int(round(spurious * len(asserted)))):
        for _attempt in range(50):
            h = rng.choice(framed_list); r = rng.choice(erasable_list); a = rng.choice(pools[r])
            if (h, r, a) not in gold and (h, r, a) not in set(wrong):
                wrong.append((h, r, a)); break
    table = EdgeTable.build(world.concept_count, len(relations), max_slots=max_slots, asserted=asserted + wrong,
                            candidates=candidates)
    return Scenario(f"erasure-{rate}", table, list(world.relation_names), relations, _relation_prior(world, relations, seed),
                    info={"erased": erased, "spurious": wrong, "candidates": len(candidates)})


def hidden_scenario(world: OntologyWorld, *, hidden: list[str], seed: int, coverage: float = 0.7,
                    distractor_ratio: float = 1.0, max_slots: int = 8, candidate_known_options: bool = False) -> Scenario:
    """Hidden relations absent from the ontology; their edges at training heads are offered as open
    candidates (with probability `coverage`) among same-type distractors."""
    rng = random.Random(seed)
    hidden_ids = [world.relation_id(n) for n in hidden]
    known = [r for r in range(len(world.relation_names)) if r not in hidden_ids]
    column = {r: i for i, r in enumerate(known)}
    framed, train = _framed(world), set(world.splits["train"].tolist())
    gold_pairs = set(world.relations_of_pair())
    pools = _pools(world)
    asserted, candidates = [], []
    for h, r, a in world.edge_list():
        if h not in framed:
            continue
        if r in column:
            asserted.append((h, column[r], a))
        elif h in train and rng.random() < coverage:
            candidates.append((h, r, a))
    open_edges, seen = [], set()
    for h, r, a in candidates:
        open_edges.append((h, -1, a)); seen.add((h, a))
        count = int(distractor_ratio) + (rng.random() < distractor_ratio - int(distractor_ratio))
        for _ in range(count):
            for _attempt in range(20):
                b = rng.choice(pools[r])
                if (h, b) not in gold_pairs and (h, b) not in seen:
                    seen.add((h, b)); open_edges.append((h, -1, b)); break
    table = EdgeTable.build(world.concept_count, len(known), max_slots=max_slots, asserted=asserted,
                            candidates=open_edges, candidate_known_options=candidate_known_options)
    return Scenario("hidden-absent", table, [world.relation_names[r] for r in known], known,
                    _relation_prior(world, known, seed), hidden_ids, {"hidden_candidates": len(candidates),
                                                                      "open_candidates": len(open_edges)})


def collapsed_scenario(world: OntologyWorld, *, collapse: dict[str, list[str]], seed: int, max_slots: int = 8) -> Scenario:
    """Sub-types merged under one label (E3-style): the label's edges are asserted but open, so a
    blank slot can take a subset of them."""
    merged = {world.relation_id(n): label for label, names in collapse.items() for n in names}
    known = [r for r in range(len(world.relation_names)) if r not in merged]
    labels = list(collapse)
    names = [world.relation_names[r] for r in known] + labels
    column = {r: i for i, r in enumerate(known)}
    label_column = {label: len(known) + i for i, label in enumerate(labels)}
    framed = _framed(world)
    asserted, opened = [], []
    for h, r, a in world.edge_list():
        if h not in framed:
            continue
        if r in merged:
            opened.append((h, label_column[merged[r]], a))
        else:
            asserted.append((h, column[r], a))
    table = EdgeTable.build(world.concept_count, len(names), max_slots=max_slots, asserted=asserted, open_asserted=opened)
    prior_cols = list(known) + [[world.relation_id(n) for n in collapse[label]] for label in labels]
    return Scenario("hidden-collapsed", table, names, list(known) + [None] * len(labels),
                    _relation_prior(world, prior_cols, seed), [r for r in merged],
                    {"collapse": collapse, "open_asserted": len(opened)})


def seed_scenario(world: OntologyWorld, kind: str, *, seed: int, coverage: float = 0.7, distractor_ratio: float = 1.0,
                  max_slots: int = 10, core_relations: tuple[str, ...] = ("is_a", "part_of", "has_attribute"),
                  core_levels: int = 2, curated_fraction: float = 0.3, wrong_rate: float = 0.2) -> Scenario:
    """E10.4 starting ontologies: empty · core · noisy (stand-in for a host-authored seed) · curated30 · full."""
    rng = random.Random(seed)
    framed, train = _framed(world), set(world.splits["train"].tolist())
    gold_pairs = set(world.relations_of_pair())
    gold = set(world.edge_list())
    pools = _pools(world)
    all_rel = list(range(len(world.relation_names)))
    level = world.concept_level.tolist()
    if kind == "empty":
        known: list[int] = []
    elif kind == "core":
        known = [world.relation_id(n) for n in core_relations]
    else:
        known = all_rel
    column = {r: i for i, r in enumerate(known)}
    names = [world.relation_names[r] for r in known] or ["unused"]
    asserted, rest = [], []
    for h, r, a in world.edge_list():
        if h not in framed:
            continue
        if kind == "full":
            asserted.append((h, column[r], a))
        elif kind == "core" and r in column and level[h] <= core_levels:
            asserted.append((h, column[r], a))
        elif kind in {"noisy", "curated30"} and rng.random() < curated_fraction:
            asserted.append((h, column[r], a))
        else:
            rest.append((h, r, a))
    wrong = []
    if kind == "noisy":
        framed_list = sorted(framed)
        for _ in range(int(round(wrong_rate * len(asserted)))):
            for _attempt in range(50):
                h = rng.choice(framed_list); r = rng.choice(all_rel); a = rng.choice(pools[r])
                if (h, r, a) not in gold:
                    wrong.append((h, column[r], a)); break
    open_edges, seen = [], set()
    for h, r, a in rest:
        if h not in train or rng.random() >= coverage:
            continue
        open_edges.append((h, -1, a)); seen.add((h, a))
        count = int(distractor_ratio) + (rng.random() < distractor_ratio - int(distractor_ratio))
        for _ in range(count):
            for _attempt in range(20):
                b = rng.choice(pools[r])
                if (h, b) not in gold_pairs and (h, b) not in seen:
                    seen.add((h, b)); open_edges.append((h, -1, b)); break
    if not asserted and not open_edges:
        raise ValueError("empty scenario")
    table = EdgeTable.build(world.concept_count, max(1, len(known)), max_slots=max_slots, asserted=asserted + wrong,
                            candidates=open_edges, candidate_known_options=bool(known))
    cols = known if known else [None]
    return Scenario(f"seed-{kind}", table, names, list(known) if known else [None], _relation_prior(world, cols, seed),
                    [r for r in all_rel if r not in column],
                    {"asserted": len(asserted), "wrong": len(wrong), "open_candidates": len(open_edges)})


# =====================================================================================================
# model, context, evaluation helpers
# =====================================================================================================

def make_learn(config: dict[str, Any], **overrides: Any) -> LearnabilityConfig:
    return LearnabilityConfig(**{**config["learn"], **overrides})


def make_composer(world: OntologyWorld, scenario: Scenario, config: dict[str, Any], seed: int, *,
                  learn: LearnabilityConfig | None = None, assignment: str = "free", normalizer: str = "softmax",
                  relation_scaling: bool | None = None) -> LearnableOntologyComposer:
    model = config["model"]
    torch.manual_seed(seed + 100)
    output = world.target_dimension if world.target_dimension != world.dimension else None
    composer = LearnableOntologyComposer(
        scenario.table, world.atomic_count, world.dimension, learn=learn or make_learn(config),
        operator=model["operator"], mode="bundle",
        relation_scaling=model.get("relation_scaling", False) if relation_scaling is None else relation_scaling,
        atomic_prior=world.atomic_prior, relation_prior=scenario.relation_prior, slot_seed=seed,
        assignment=assignment, normalizer=normalizer, output_dimension=output,
        scale_free=model.get("scale_free", False))
    composer.seed_empty_concepts(float(config.get("discovery", {}).get("slot_init_mass", 0.1)))
    return composer


def make_context(world: OntologyWorld, scenario: Scenario, seed: int, *, heldout_split: str = "val") -> LearningContext:
    """Training targets and held-out (self-test) observations; never the audit observations."""
    train, val = world.splits["train"], world.splits["validation"]
    obs = world.observation_splits
    targets = world.observations[train][:, obs["train"]]
    held = torch.cat([train, val]).sort().values
    heldout = HeldOutData(held, world.observations[held][:, obs[heldout_split]], name=heldout_split)
    return LearningContext(train, targets, heldout, NodeMap(world.concept_node, world.filler_node),
                           scenario.relation_names, set(_framed(world)), seed)


def audit_data(world: OntologyWorld) -> HeldOutData:
    """Audit observations of every framed concept (evaluation only)."""
    concepts = torch.tensor(sorted(_framed(world)))
    return HeldOutData(concepts, world.observations[concepts][:, world.observation_splits["audit"]], name="audit")


def discovery_settings(config: dict[str, Any], **overrides: Any) -> DiscoverySettings:
    spec = dict(config.get("discovery", {}))
    hyp = HypothesisSettings(**spec.pop("hypotheses", {}))
    dream = DreamSettings(**{**config.get("dream_settings", {}), "lr": config["model"]["lr"]})
    spec.setdefault("lr", config["model"]["lr"])
    spec.update(overrides)
    return DiscoverySettings(hypotheses=hyp, dream=dream, **{k: v for k, v in spec.items() if k not in {"hypotheses"}})


@torch.no_grad()
def eval_fit(composer: LearnableOntologyComposer, data: HeldOutData, concepts: torch.Tensor) -> float:
    concepts = concepts[torch.tensor([data.has(int(c)) for c in concepts.tolist()], dtype=torch.bool)]
    if not concepts.numel():
        return float("nan")
    z = base_rows(composer, concepts)
    prediction = composer.predict_from_raw(z)
    return float(F.cosine_similarity(prediction[:, None, :], data.observations[data.rows(concepts)], dim=-1).mean())


def _auc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(scores)
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j] == order[i]:
            j += 1
        ranks[order[i]] = (i + j + 1) / 2
        i = j
    rank_sum = sum(ranks[s] for s in pos)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _prf(predicted: set, gold: set) -> dict[str, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else float("nan")
    r = tp / len(gold) if gold else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f, "predicted": len(predicted), "gold": len(gold)}


def _gold_columns(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer,
                  nodes: NodeMap) -> dict[int, str | None]:
    """Gold relation name of every column (seed labels by construction; slots by best Jaccard of their
    members' pairs against all gold relations restricted to framed heads). Evaluation only."""
    framed = _framed(world)
    gold = {name: world.pairs(name, heads=framed) for name in world.relation_names}
    out: dict[int, str | None] = {c: (world.relation_names[w] if w is not None else None)
                                  for c, w in enumerate(scenario.column_world)}
    for slot in composer.slot_active.nonzero().flatten().tolist():
        members = composer.slot_members(slot)
        pairs = edge_pairs(composer, members, nodes)
        name, score = best_match(pairs, gold)
        out[composer.slot_column(slot)] = name if score > 0 else None
    return out


def _alias(world: OntologyWorld, scenario: Scenario, columns_gold: dict[int, str | None], composer) -> dict[str, str]:
    alias = {name: world.relation_names[w] for name, w in zip(scenario.relation_names, scenario.column_world) if w is not None}
    for slot in composer.slot_active.nonzero().flatten().tolist():
        g = columns_gold.get(composer.slot_column(slot))
        if g is not None:
            alias[f"slot{slot}"] = g
    return alias


# =====================================================================================================
# part a — learnability ablation, part b — erasure & recovery
# =====================================================================================================

def _candidate_eval(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer,
                    threshold: float = 0.5) -> dict[str, Any]:
    """Recovery of erased edges by learned candidate masses (typed candidates)."""
    erased = set(scenario.info.get("erased", []))
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    columns = composer.schedule.relations.tolist(); masses = composer.edge_masses().detach().tolist()
    is_candidate = (composer.origin == ORIGIN["candidate"]).tolist()
    scores, labels, predicted = [], [], set()
    freq: dict[tuple[int, int], int] = defaultdict(int)
    for h, c, a, cand, m in zip(heads, columns, fillers, is_candidate, masses):
        if not cand and m > 0:
            freq[(c, a)] += 1
    freq_scores = []
    for h, c, a, cand, m in zip(heads, columns, fillers, is_candidate, masses):
        if not cand:
            continue
        edge = (h, scenario.column_world[c], a)
        scores.append(m); labels.append(edge in erased); freq_scores.append(freq[(c, a)])
        if m >= threshold:
            predicted.add(edge)
    gold = {e for e in erased}
    out = _prf(predicted, gold)
    k = len(gold)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    top_freq = sorted(range(len(freq_scores)), key=lambda i: (-freq_scores[i], i))[:k]
    out.update(auc=_auc(scores, labels), r_precision=sum(labels[i] for i in top) / max(1, k),
               prevalence=sum(labels) / max(1, len(labels)), candidates=len(labels),
               frequency_auc=_auc([float(s) for s in freq_scores], labels),
               frequency_r_precision=sum(labels[i] for i in top_freq) / max(1, k))
    spurious = set(scenario.info.get("spurious", []))
    if spurious:
        removed = kept = n_true = 0
        spurious_mass, true_mass = [], []
        for h, c, a, cand, m in zip(heads, columns, fillers, is_candidate, masses):
            if cand:
                continue
            edge = (h, scenario.column_world[c], a)
            if edge in spurious:
                removed += m < threshold; spurious_mass.append(m)
            else:
                n_true += 1; kept += m >= threshold; true_mass.append(m)
        out.update(spurious_removed=removed / len(spurious), true_asserted_kept=kept / max(1, n_true),
                   spurious_mass_mean=float(np.mean(spurious_mass)) if spurious_mass else float("nan"),
                   true_asserted_mass_mean=float(np.mean(true_mass)) if true_mass else float("nan"))
    return out


def _vector_recovery(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer) -> dict[str, float]:
    if "atomics" not in world.truth:
        return {}
    atoms = F.cosine_similarity(composer.atomic_vectors().detach(), world.truth["atomics"], dim=-1)
    prior = F.cosine_similarity(world.atomic_prior, world.truth["atomics"], dim=-1)
    known = [w for w in scenario.column_world if w is not None]
    roles = composer.relation_vectors().detach()[:len(known)]
    rel = F.cosine_similarity(F.normalize(roles, dim=-1), world.truth["roles"][known], dim=-1)
    return {"atomic_cosine": float(atoms.mean()), "atomic_prior_cosine": float(prior.mean()),
            "relation_cosine": float(rel.mean())}


def part_a(config: dict[str, Any], seed: int, chunk: list[tuple[str, ...]]) -> list[dict[str, Any]]:
    section = config["a"]
    world = build_world(config, seed)
    scenario = erasure_scenario(world, rate=section["erasure"], seed=seed, distractors=section["distractors"],
                                spurious=section["spurious"], erase_relations=section.get("erase_relations"))
    ctx = make_context(world, scenario, seed)
    audit = audit_data(world)
    rows = []
    for setting in chunk:
        learn = make_learn(config, **dict(zip(COMPONENTS, setting)))
        composer = make_composer(world, scenario, config, seed, learn=learn)
        start = time.time()
        result = train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["steps"], lr=config["model"]["lr"])
        rows.append({"part": "a", "seed": seed, **dict(zip(COMPONENTS, setting)), "train_fit": 1 - result.fit,
                     "val_fit": eval_fit(composer, ctx.heldout, world.splits["train"]),
                     "test_fit": eval_fit(composer, audit, world.splits["test"]),
                     **_candidate_eval(world, scenario, composer), **_vector_recovery(world, scenario, composer),
                     "seconds": time.time() - start})
    return rows


def _edge_acceptance_rows(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer,
                          ctx: LearningContext, config: dict[str, Any], seed: int, tag: str) -> list[dict[str, Any]]:
    """Self-tested acceptance of proposed edges (candidate masses ≥ propose threshold)."""
    st = config["selftest"]
    masses = composer.edge_masses().detach()
    proposals = ((composer.origin == ORIGIN["candidate"]) & (masses >= st["propose_mass"])).nonzero().flatten()
    if not proposals.numel():
        return []
    held = accept_edges(composer, proposals, ctx.heldout, resamples=st["resamples"], seed=seed, alpha=st["alpha"])
    obs = world.observation_splits
    train_data = HeldOutData(ctx.train_concepts, ctx.train_targets, name="train")
    insample = {d.key: d for d in accept_edges(composer, proposals, train_data, resamples=st["resamples"], seed=seed,
                                               alpha=st["alpha"])}
    audit = audit_data(world)
    audit_dec = {d.key: d for d in accept_edges(composer, proposals, audit, resamples=st["resamples"], seed=seed,
                                                alpha=st["alpha"])}
    gold = set(world.edge_list())
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    columns = composer.schedule.relations.tolist()
    artifacts = set(scenario.info.get("artifacts", []))
    rows = []
    for d in held:
        e = d.key
        edge = (heads[e], scenario.column_world[columns[e]], fillers[e])
        rows.append({"part": "d_edges", "tag": tag, "seed": seed, "edge": list(edge), "mass": float(masses[e]),
                     "gold": edge in gold, "artifact": edge in artifacts, "accept": bool(d.accept), "mean": d.mean,
                     "lower": d.lower, "insample_accept": bool(insample[e].accept),
                     "audit_mean": audit_dec[e].mean if e in audit_dec else float("nan")})
    return rows


def part_b(config: dict[str, Any], seed: int, rate: float) -> list[dict[str, Any]]:
    section = config["b"]
    world = build_world(config, seed)
    scenario = erasure_scenario(world, rate=rate, seed=seed, distractors=section["distractors"],
                                erase_relations=section.get("erase_relations"))
    ctx = make_context(world, scenario, seed)
    composer = make_composer(world, scenario, config, seed)
    result = train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["steps"], lr=config["model"]["lr"])
    audit = audit_data(world)
    row = {"part": "b", "seed": seed, "rate": rate, "principal_fixed": 1 - rate, "train_fit": 1 - result.fit,
           "val_fit": eval_fit(composer, ctx.heldout, world.splits["train"]),
           "test_fit": eval_fit(composer, audit, world.splits["test"]), **_candidate_eval(world, scenario, composer)}
    rows = [row]
    if rate in section.get("acceptance_rates", []):
        rows += _edge_acceptance_rows(world, scenario, composer, ctx, config, seed, f"erasure-{rate}")
    return rows


# =====================================================================================================
# part c — blank-relation discovery (+ e new words, riddle analysis, trajectories)
# =====================================================================================================

def _partition(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer, *, mode: str,
               labels_from: str = "slots", min_mass: float = 0.25) -> tuple[list, list, list]:
    """(gold label, predicted label, pair) per evaluated edge (evaluation only)."""
    gold_of = world.relations_of_pair()
    hidden = set(scenario.hidden)
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    masses = composer.edge_masses().detach().tolist()
    columns, confidence = composer.edge_options()
    columns, confidence = columns.tolist(), confidence.tolist()
    nominal = composer.schedule.relations.tolist()
    origin = composer.origin.tolist()
    gold_l, pred_l, pairs = [], [], []
    for e, (h, a) in enumerate(zip(heads, fillers)):
        if mode == "absent" and origin[e] != ORIGIN["candidate"]:
            continue
        if mode == "collapsed" and origin[e] != ORIGIN["open_asserted"]:
            continue
        rels = gold_of.get((h, a), set()) & hidden
        gold_l.append(world.relation_names[min(rels)] if rels else None)
        if labels_from == "relations":       # M3: the edge's (possibly split) relation id
            label = nominal[e] if masses[e] >= min_mass else None
        else:
            ok = masses[e] >= min_mass and confidence[e] >= 0.5 and columns[e] >= 0
            label = columns[e] if ok else None
            if mode == "absent" and label is not None and label < composer.base_relations:
                label = f"known:{label}"
        pred_l.append(label)
        pairs.append((world.concept_node[h], world.filler_node[a]))
    return gold_l, pred_l, pairs


def _slot_relation_recovery(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer,
                            nodes: NodeMap) -> dict[str, Any]:
    """Per hidden relation: best Jaccard of an accepted (crystallized) slot's full pair set, incl.
    rule-implied edges, against gold pairs over framed heads (evaluation only)."""
    framed = _framed(world)
    out = {}
    slots = composer.slot_frozen.nonzero().flatten().tolist()
    slot_pairs = {s: edge_pairs(composer, composer.slot_members(s, hard_only=True)[
        composer.edge_masses()[composer.slot_members(s, hard_only=True)] > 0], nodes) for s in slots}
    for r in scenario.hidden:
        name = world.relation_names[r]
        gold = world.pairs(name, heads=framed)
        best = max(((jaccard(p, gold), s, p) for s, p in slot_pairs.items()), default=(0.0, None, set()), key=lambda x: x[0])
        out[name] = {"jaccard": best[0], "slot": best[1],
                     "precision": len(best[2] & gold) / len(best[2]) if best[2] else float("nan"),
                     "recall": len(best[2] & gold) / len(gold) if gold else float("nan")}
    return out


def _riddle_eval(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer, rounds: list[dict],
                 nodes: NodeMap) -> list[dict[str, Any]]:
    """After-the-fact check of adopted structural hypotheses against gold (evaluation only)."""
    framed = _framed(world)
    gold_rel = {name: world.pairs(name, heads=None) for name in world.relation_names}
    columns_gold = _gold_columns(world, scenario, composer, nodes)
    alias = _alias(world, scenario, columns_gold, composer)
    out = []
    for record in rounds:
        pairs = set(map(tuple, record["pairs"]))
        offered = _offered_pairs(world, scenario)
        name, score = best_match(pairs, offered) if offered else best_match(pairs, {n: world.pairs(n, heads=framed) for n in world.relation_names})
        entry = {"slot": record["slot"], "accept": record["accept"], "best_relation": name, "best_jaccard": score,
                 "adopted": record["adopted"], "size": record["size"]}
        designed = world.gold_properties.get(name, ()) if name else ()
        entry["designed"] = list(designed)
        if record["adopted"] and name:
            kind, _, args = record["adopted"].partition(":")
            hyp = StructuralHypothesis(kind, tuple(args.split("∘")) if args else ())
            entry["adopted_true"] = property_holds(hyp, gold_rel[name], gold_rel, alias)
            mapped = kind if not args else f"{kind}:" + "∘".join(alias.get(a, a) for a in args.split("∘"))
            entry["adopted_mapped"] = mapped
            entry["adopted_designed"] = mapped in set(designed)
        hyps = []
        for h in record.get("hypotheses_full") or record["hypotheses"]:
            kind, _, args = h["name"].partition(":")
            hyp = StructuralHypothesis(kind, tuple(args.split("∘")) if args else ())
            mapped_h = kind if not args else f"{kind}:" + "∘".join(alias.get(a, a) for a in args.split("∘"))
            hyps.append({**h, "true": property_holds(hyp, gold_rel[name], gold_rel, alias) if name else False,
                         "designed": bool(name) and mapped_h in set(designed)})
        entry["hypotheses"] = hyps
        if record.get("predicted_by_adopted") and name:
            predicted = set(map(tuple, record["predicted_by_adopted"]))
            gold_all = gold_rel[name]
            entry["rule_prediction_precision"] = len(predicted & gold_all) / len(predicted) if predicted else float("nan")
        out.append(entry)
    return out


def _trajectory_eval(world: OntologyWorld, scenario: Scenario, trajectory: list[dict], rounds: list[dict]) -> list[dict]:
    """Human-likeness readout: per slot, size and gold purity of its members over training (eval only)."""
    framed = _framed(world)
    gold = {world.relation_names[r]: world.pairs(world.relation_names[r], heads=framed) for r in scenario.hidden}
    final = {r["slot"]: set(map(tuple, r["pairs"])) for r in rounds}
    target_of = {}
    for slot, pairs in final.items():
        name, _ = best_match(pairs, gold) if pairs else (None, 0)
        target_of[slot] = name
    out = []
    for entry in trajectory:
        pairs = set(map(tuple, entry["pairs"]))
        target = target_of.get(entry["slot"])
        relations_present = {n for n, g in gold.items() if pairs & g}
        in_target = len(pairs & gold[target]) if target else 0
        any_gold = sum(1 for p in pairs if any(p in g for g in gold.values()))
        soft = [(tuple(p), v) for p, v in entry.get("soft", [])]
        total = sum(v for _, v in soft)
        share = defaultdict(float)
        for p, v in soft:
            label = next((n for n, g in gold.items() if p in g), None)
            share[label] += v
        effective = (total ** 2 / sum(v * v for _, v in soft)) if soft else 0.0
        out.append({"step": entry["step"], "slot": entry["slot"], "members": entry["members"],
                    "assigned_mass": entry["assigned_mass"], "target": target,
                    "purity": in_target / len(pairs) if pairs else float("nan"),
                    "gold_share": any_gold / len(pairs) if pairs else float("nan"),
                    "relations_present": len(relations_present),
                    "target_recall": in_target / len(gold[target]) if target else float("nan"),
                    "soft_target_share": share.get(target, 0.0) / total if total and target else float("nan"),
                    "soft_other_relation_share": sum(v for k, v in share.items() if k not in {target, None}) / total if total else float("nan"),
                    "soft_distractor_share": share.get(None, 0.0) / total if total else float("nan"),
                    "soft_effective_size": effective})
    return out


def _m3_composer(world, scenario: Scenario, config, seed):
    """M3 baseline: open candidates become typed edges of one extra 'unknown' relation (absent mode),
    collapsed labels become ordinary relations; M3 then splits relation vectors."""
    t = scenario.table
    unknown = t.relation_count
    relations = t.relations.clone()
    relations[t.open & t.candidate] = unknown
    table = EdgeTable(t.concept_count, t.relation_count + 1, 0, t.heads, relations, t.fillers, t.prior, t.candidate,
                      torch.zeros_like(t.open), torch.zeros(t.heads.numel(), t.relation_count + 1, dtype=torch.bool),
                      torch.where(t.origin == ORIGIN["open_asserted"], torch.full_like(t.origin, ORIGIN["open_asserted"]), t.origin))
    prior = scenario.relation_prior
    if prior is not None:
        g = torch.Generator().manual_seed(seed + 5)
        prior = torch.cat([prior, F.normalize(torch.randn(1, prior.shape[1], generator=g), dim=-1)])
    m3_scenario = Scenario(scenario.name + "-m3", table, scenario.relation_names + ["unknown"],
                           scenario.column_world + [None], prior, scenario.hidden, scenario.info)
    composer = make_composer(world, m3_scenario, config, seed, relation_scaling=False)
    return composer, m3_scenario


def run_method(world: OntologyWorld, scenario: Scenario, config: dict[str, Any], seed: int, method: str, *,
               mode: str, matched_steps: int | None = None) -> tuple[LearnableOntologyComposer, dict[str, Any]]:
    section = config["c"]
    settings = discovery_settings(config)
    ctx = make_context(world, scenario, seed)
    steps = matched_steps or settings.steps_per_round * settings.max_rounds
    out: dict[str, Any] = {"method": method}
    if method in {"additive", "additive_norule"}:
        composer = make_composer(world, scenario, config, seed)
        s = discovery_settings(config, enforce_rule=method == "additive")
        composer, result = additive_curriculum(composer, ctx, s)
        out.update(result)
    elif method == "all_at_once":
        composer = make_composer(world, scenario, config, seed)
        composer, result = all_at_once(composer, ctx, settings, slots=section["all_at_once_slots"], steps=steps)
        out.update(result)
    elif method == "stem_cell":
        pool = section["stem_cell_pool"]
        t = scenario.table
        table = EdgeTable(t.concept_count, t.relation_count, pool, t.heads, t.relations, t.fillers, t.prior, t.candidate,
                          t.open, torch.cat([t.options[:, :t.relation_count],
                                             t.options[:, t.relation_count:t.relation_count + 1].expand(-1, pool)], 1),
                          t.origin)
        stem = Scenario(scenario.name, table, scenario.relation_names, scenario.column_world, scenario.relation_prior,
                        scenario.hidden, scenario.info)
        composer = make_composer(world, stem, config, seed, assignment="router", normalizer="sparsemax")
        for _ in range(pool):
            composer.add_blank_slot(init_mass=settings.slot_init_mass)
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=steps, lr=config["model"]["lr"])
        out.update(steps=steps, rounds=[])
    elif method == "m3":
        composer, m3_scenario = _m3_composer(world, scenario, config, seed)
        with torch.no_grad():
            unknown_edges = composer.candidate & (composer.schedule.relations == composer.base_relations - 1)
            composer.edge_mass[unknown_edges] = settings.slot_init_mass
        optimizer = torch.optim.Adam([p for p in composer.parameters() if p.requires_grad], lr=config["model"]["lr"])
        dev = DevelopmentalConfig(**section["m3"], target="relations", seed=seed)
        tracker = DevelopmentalDictionary(composer, optimizer, dev)
        warm = int(steps * section["m3_schedule"][0]); growth = int(steps * section["m3_schedule"][1])
        def after(step: int) -> None:
            if step == warm + growth - 1:
                tracker.consolidate()
            tracker.sync_parents()
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=steps, lr=config["model"]["lr"],
                       optimizer=optimizer, tracker=tracker, tracker_phase=lambda s: warm <= s < warm + growth,
                       after_step=after)
        out.update(steps=steps, rounds=[], m3_cards=[{k: v for k, v in c.items() if k != "direction"} for c in tracker.cards])
        scenario = m3_scenario
    elif method == "oracle":
        composer = make_composer(world, scenario, config, seed)
        hidden_names = [world.relation_names[r] for r in scenario.hidden]
        slots = {name: composer.add_blank_slot() for name in hidden_names}
        gold_of = world.relations_of_pair()
        heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
        for e in range(len(heads)):
            if not bool(composer.open[e]):
                continue
            rels = gold_of.get((heads[e], fillers[e]), set()) & set(scenario.hidden)
            if rels:
                composer.set_edge_option(e, composer.slot_column(slots[world.relation_names[min(rels)]]))
            elif mode == "collapsed":
                pass
        if mode == "collapsed":
            # oracle for sub-types: the first sub-type of each label keeps the label, the others own slots
            firsts = {world.relation_id(names[0]) for names in config["c"]["collapse"].values()}
            for e in range(len(heads)):
                if composer.origin[e] == ORIGIN["open_asserted"] and firsts & gold_of.get((heads[e], fillers[e]), set()):
                    composer.set_edge_option(e, int(composer.schedule.relations[e]))
        with torch.no_grad():
            seeded = composer.open & composer.candidate
            composer.edge_mass[seeded] = composer.edge_mass[seeded].clamp_min(settings.slot_init_mass)
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=steps, lr=config["model"]["lr"])
        out.update(steps=steps, rounds=[])
    else:
        raise ValueError(method)
    return composer, {**out, "scenario": scenario}


def eval_discovery(world, scenario, composer, ctx, result, *, mode: str, method: str) -> dict[str, Any]:
    labels_from = "relations" if method == "m3" else "slots"
    gold_l, pred_l, pairs = _partition(world, result["scenario"], composer, mode=mode, labels_from=labels_from)
    if method == "m3" and mode == "absent":
        unknown = result["scenario"].table.relation_count - 1
        pred_l = [p if p is None or p >= unknown else None for p in pred_l]
    scores = partition_scores(gold_l, pred_l, pairs, permutations=200, seed=ctx.seed)
    audit = audit_data(world)
    row = {"ari_all": scores.ari_all, "ari_gold_edges": scores.ari_gold_edges,
           "permuted_ari_mean": scores.permuted_ari_mean, "permuted_ari_p": scores.permuted_ari_p,
           "detection_precision": scores.detection_precision, "detection_recall": scores.detection_recall,
           "mean_best_jaccard": scores.mean_best_jaccard, "per_relation": scores.per_relation,
           "clusters": len({p for p in pred_l if p is not None}),
           "val_fit": eval_fit(composer, ctx.heldout, world.splits["train"]),
           "test_fit": eval_fit(composer, audit, world.splits["test"]),
           "validation_concept_fit": eval_fit(composer, audit, world.splits["validation"])}
    if method.startswith("additive") or method == "all_at_once":
        row["accepted_slots"] = sum(r["accept"] for r in result["rounds"])
        row["proposed_slots"] = len(result["rounds"])
        row["relation_recovery"] = _slot_relation_recovery(world, result["scenario"], composer, ctx.nodes)
    return row


def part_c(config: dict[str, Any], seed: int, mode: str) -> list[dict[str, Any]]:
    section = config["c"]
    world = build_world(config, seed)
    if mode == "absent":
        scenario = hidden_scenario(world, hidden=section["hidden"], seed=seed, coverage=section["coverage"],
                                   distractor_ratio=section["distractor_ratio"], max_slots=section["max_slots"])
    else:
        scenario = collapsed_scenario(world, collapse=section["collapse"], seed=seed, max_slots=section["max_slots"])
    ctx = make_context(world, scenario, seed)
    rows: list[dict[str, Any]] = []
    matched = None
    states = {}
    for method in section["methods"]:
        start = time.time()
        composer, result = run_method(world, scenario, config, seed, method, mode=mode, matched_steps=matched)
        if method == "additive":
            matched = result["steps"]
            states["additive"] = composer
        row = {"part": "c", "mode": mode, "seed": seed, "method": method, "steps": result["steps"],
               **eval_discovery(world, scenario, composer, ctx, result, mode=mode, method=method),
               "seconds": time.time() - start}
        if result.get("rounds"):
            row["rounds"] = [{k: v for k, v in r.items() if k not in {"pairs", "predicted_by_adopted"}} for r in result["rounds"]]
            row["riddle"] = _riddle_eval(world, scenario, composer, result["rounds"], ctx.nodes)
            row["decisions"] = _slot_decision_eval(world, scenario, result["rounds"])
        if result.get("trajectory"):
            row["trajectory"] = _trajectory_eval(world, scenario, result["trajectory"], result.get("rounds", []))
        if result.get("m3_cards"):
            row["m3_events"] = len(result["m3_cards"])
        rows.append(row)
    if mode == "absent" and "e" in config.get("parts", []) and "additive" in states:
        rows += part_e(world, scenario, states["additive"], config, seed)
    return rows


def _offered_pairs(world: OntologyWorld, scenario: Scenario) -> dict[str, set]:
    """Gold pairs of each hidden relation that were offered to the learner as open candidates."""
    gold_of = world.relations_of_pair()
    t = scenario.table
    out: dict[str, set] = defaultdict(set)
    for h, a, is_open, origin in zip(t.heads.tolist(), t.fillers.tolist(), t.open.tolist(), t.origin.tolist()):
        if not is_open:
            continue
        for r in gold_of.get((h, a), set()) & set(scenario.hidden):
            out[world.relation_names[r]].add((world.concept_node[h], world.filler_node[a]))
    return out


def _slot_decision_eval(world, scenario, rounds) -> list[dict[str, Any]]:
    """Gold verdict of each slot hypothesis (evaluation only). `real` = most of its pairs are gold
    edges of a hidden relation (precision ≥ 0.5); `complete` = Jaccard ≥ 0.5 with one hidden
    relation's offered pairs."""
    offered = _offered_pairs(world, scenario)
    out = []
    for r in rounds:
        pairs = set(map(tuple, r["pairs"]))
        name, score = best_match(pairs, offered) if pairs and offered else (None, 0.0)
        precision = sum(1 for p in pairs if any(p in g for g in offered.values())) / len(pairs) if pairs else float("nan")
        real = [p for p in pairs if any(p in g for g in offered.values())]
        dominant = max((sum(p in g for p in real) for g in offered.values()), default=0)
        purity = dominant / len(real) if real else float("nan")
        good = bool(precision == precision and precision >= 0.5 and purity >= 0.7)
        out.append({"slot": r["slot"], "accept": r["accept"], "gold_good": good, "complete": score >= 0.5, "best": name,
                    "jaccard": score, "precision": precision, "purity": purity, "lower": r["utility_lower"],
                    "size": len(pairs), "split_from": r.get("split_from")})
    return out


# =====================================================================================================
# part e — new-word frame inference
# =====================================================================================================

def part_e(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer, config: dict[str, Any],
           seed: int) -> list[dict[str, Any]]:
    section = config["e"]
    nodes = NodeMap(world.concept_node, world.filler_node)
    columns_gold = _gold_columns(world, scenario, composer, nodes)
    crystallized = [composer.slot_column(s) for s in composer.slot_frozen.nonzero().flatten().tolist()]
    known = list(range(composer.base_relations))
    words = world.splits["new_word"]
    gold_frames = {int(w): {(world.relation_names[r], a) for h, r, a in world.edge_list() if h == int(w)} for w in words}
    obs = world.observation_splits
    infer_obs = world.observations[words][:, obs["infer"]]
    eval_obs = world.observations[words][:, obs["eval"]]
    if section.get("filler_pool"):
        counts = torch.bincount(composer.schedule.fillers, minlength=world.atomic_count)
        # the most used fillers; gold edges with other fillers stay in the gold frames (count as misses)
        fillers = sorted(torch.argsort(counts, descending=True)[:int(section["filler_pool"])].tolist())
    else:
        fillers = list(range(world.atomic_count))
    project = composer.projector

    def bind_model(cols, fills):
        return composer.bind_options(cols, fills).detach()

    def bind_oracle(cols, fills):
        roles = world.truth["roles"][cols]
        from vsa_embed.algebra import HRRAlgebra
        return HRRAlgebra().bind(roles, world.truth["atomics"][fills])

    dictionaries = {
        "discovered": (known + crystallized, bind_model, lambda c: columns_gold.get(c)),
        "known_only": (known, bind_model, lambda c: columns_gold.get(c)),
    }
    if "roles" in world.truth:
        dictionaries["oracle"] = (list(range(len(world.relation_names))), bind_oracle, lambda c: world.relation_names[c])
    grids = {name: candidate_dictionary(bind, cols, fillers, project=project if name != "oracle" else None)
             for name, (cols, bind, _) in dictionaries.items()}
    # nearest-neighbour frames: training concepts' asserted / crystallized edges (model columns)
    train = world.splits["train"]
    with torch.no_grad():
        train_rows = composer.predict_from_raw(base_rows(composer, train))
    columns_now, conf = composer.edge_options()
    masses = composer.edge_masses().detach()
    heads = composer.edge_heads()
    frame_of: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e in range(heads.numel()):
        if masses[e] >= 0.25 and columns_now[e] >= 0 and conf[e] >= 0.5:
            frame_of[int(heads[e])].append((int(columns_now[e]), int(composer.schedule.fillers[e])))
    rng = random.Random(seed + 11)
    all_columns = known + crystallized
    rows = []
    levels = world.concept_level.tolist()
    for k in section["k"]:
        observations = infer_obs[:, :k]
        per = defaultdict(list)
        for name, (cols, bind, to_gold) in dictionaries.items():
            grid_c, grid_f, vectors = grids[name]
            frames = infer_frames(observations, grid_c, grid_f, vectors, max_edges=section["max_edges"],
                                  threshold=section["threshold"])
            for w, frame, ev in zip(words.tolist(), frames, eval_obs):
                predicted = {(to_gold(c), a) for c, a in zip(frame.columns, frame.fillers)}
                s = frame_scores(predicted, gold_frames[w])
                if name == "oracle":
                    fit = float("nan")
                else:
                    row_vec = compose_frame(bind_model, frame.columns, frame.fillers, project=project, dimension=world.dimension)
                    fit = float(F.cosine_similarity(row_vec[None], ev, dim=-1).mean())
                per[name].append({**s, "fit": fit, "level": levels[w]})
        for w, obs_k, ev in zip(words.tolist(), observations, eval_obs):
            mean_obs = F.normalize(obs_k.mean(0), dim=-1)
            nearest = int(train[int((F.normalize(train_rows, dim=-1) @ mean_obs).argmax())])
            nn_frame = frame_of.get(nearest, [])
            predicted = {(columns_gold.get(c), a) for c, a in nn_frame}
            s = frame_scores(predicted, gold_frames[w])
            vec = compose_frame(bind_model, [c for c, _ in nn_frame], [a for _, a in nn_frame], project=project,
                                dimension=world.dimension)
            per["nearest_neighbour"].append({**s, "fit": float(F.cosine_similarity(vec[None], ev, dim=-1).mean()), "level": levels[w]})
            degree = max(1, len(gold_frames[w]))
            rand = [(rng.choice(all_columns), rng.randrange(world.atomic_count)) for _ in range(degree)]
            s = frame_scores({(columns_gold.get(c), a) for c, a in rand}, gold_frames[w])
            vec = compose_frame(bind_model, [c for c, _ in rand], [a for _, a in rand], project=project, dimension=world.dimension)
            per["random"].append({**s, "fit": float(F.cosine_similarity(vec[None], ev, dim=-1).mean()), "level": levels[w]})
            per["context_mean"].append({"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                                        "fit": float(F.cosine_similarity(mean_obs[None], ev, dim=-1).mean()), "level": levels[w]})
            gold_cols = {g: c for c, g in columns_gold.items() if g is not None}
            gf = [(gold_cols[r], a) for r, a in gold_frames[w] if r in gold_cols]
            vec = compose_frame(bind_model, [c for c, _ in gf], [a for _, a in gf], project=project, dimension=world.dimension)
            per["gold_frame"].append({"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                                      "fit": float(F.cosine_similarity(vec[None], ev, dim=-1).mean()), "level": levels[w]})
        for name, values in per.items():
            mean = lambda key: float(np.nanmean([v[key] for v in values])) if values else float("nan")  # noqa: E731
            by_level = {}
            for lv in sorted({v["level"] for v in values}):
                sel = [v["f1"] for v in values if v["level"] == lv]
                by_level[str(lv)] = float(np.nanmean(sel)) if sel and not all(x != x for x in sel) else float("nan")
            rows.append({"part": "e", "seed": seed, "k": k, "method": name, "precision": mean("precision"),
                         "recall": mean("recall"), "f1": mean("f1"), "fit": mean("fit"), "f1_by_level": by_level,
                         "words": len(values), "crystallized_slots": len(crystallized)})
    return rows


# =====================================================================================================
# E10.4 seed ontologies
# =====================================================================================================

def _recovered_ontology(world: OntologyWorld, scenario: Scenario, composer: LearnableOntologyComposer,
                        nodes: NodeMap, min_mass: float = 0.25) -> dict[str, set]:
    """Final ontology by gold relation (columns mapped after the fact; evaluation only)."""
    columns_gold = _gold_columns(world, scenario, composer, nodes)
    columns, conf = composer.edge_options()
    masses = composer.edge_masses().detach()
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    out: dict[str, set] = defaultdict(set)
    for e, (h, a) in enumerate(zip(heads, fillers)):
        c = int(columns[e])
        if masses[e] < min_mass or c < 0 or conf[e] < 0.5:
            continue
        name = columns_gold.get(c)
        if name is not None:
            out[name].add((world.concept_node[h], world.filler_node[a]))
    return out


def _reasoning_probes(world: OntologyWorld, recovered: dict[str, set], seed: int) -> dict[str, float]:
    """Reasoning probes on an ontology (recovered or gold; evaluation only). Relation names come from
    `world.metadata["probes"]`: multi-hop is-a queries (is y an ancestor of x?) answered by the is-a
    closure of the ontology against gold ancestors (closure of the full gold is-a relation), with one
    level-matched non-ancestor per positive; closure ratio of the transitive relation; Jaccard of an
    inverse pair; symmetry of the symmetric relation."""
    names = world.metadata.get("probes", {"isa": "is_a", "transitive": "located_in", "inverse": ["has_part", "part_of"],
                                           "symmetric": "similar_to"})
    framed = _framed(world)
    framed_nodes = {world.concept_node[c] for c in framed}
    rng = random.Random(seed)
    out: dict[str, float] = {}

    def closure(pairs: set) -> dict:
        up = defaultdict(set)
        for h, t in pairs:
            up[h].add(t)
        cache: dict = {}

        def ancestors(x):
            if x in cache:
                return cache[x]
            seen, depth, frontier, d = set(), {}, [x], 0
            while frontier:
                d += 1
                nxt = []
                for node in frontier:
                    for p in up.get(node, ()):
                        if p not in seen:
                            seen.add(p); depth[p] = d; nxt.append(p)
                frontier = nxt
            cache[x] = depth
            return depth
        return ancestors

    if names.get("isa") and names["isa"] in world.relation_names:
        gold_up = closure(world.pairs(names["isa"]))
        rec_up = closure(recovered.get(names["isa"], set()))
        levels = world.concept_level.tolist()
        level_of = {world.concept_node[c]: levels[c] for c in range(world.concept_count)}
        pool = defaultdict(list)
        for node, lv in level_of.items():
            pool[lv].append(node)
        correct = {1: [], 2: [], 3: []}
        for c in sorted(framed):
            x = world.concept_node[c]
            gold = gold_up(x)
            if len(gold) < 2:
                continue
            rec = rec_up(x)
            for anc, distance in gold.items():
                bucket = min(distance, 3)
                correct[bucket].append(anc in rec)
                candidates = [y for y in pool.get(level_of.get(anc, -1), []) if y not in gold and y != x] or \
                    [y for y in level_of if y not in gold and y != x]
                negative = candidates[rng.randrange(len(candidates))]
                correct[bucket].append(negative not in rec)
        for d, values in correct.items():
            out[f"isa_hop{d}_accuracy"] = sum(values) / len(values) if values else float("nan")
        multi = correct[2] + correct[3]
        out["isa_multihop_accuracy"] = sum(multi) / len(multi) if multi else float("nan")
    if names.get("transitive"):
        out["located_in_transitivity"] = transitivity(recovered.get(names["transitive"], set()))[0]
    if names.get("inverse"):
        a, b = names["inverse"]
        hp, po = recovered.get(a, set()), recovered.get(b, set())
        out["part_inverse_consistency"] = jaccard(hp, inverse(po)) if (hp or po) else float("nan")
    if names.get("symmetric"):
        out["similar_symmetry"] = symmetry(recovered.get(names["symmetric"], set()))
    return out


def part_seed(config: dict[str, Any], seed: int, kind: str) -> list[dict[str, Any]]:
    section = config["seed_ontology"]
    world = build_world(config, seed)
    scenario = seed_scenario(world, kind, seed=seed, coverage=section["coverage"], distractor_ratio=section["distractor_ratio"],
                             max_slots=section["max_slots"], core_levels=section["core_levels"],
                             core_relations=tuple(section["core_relations"]))
    ctx = make_context(world, scenario, seed)
    composer = make_composer(world, scenario, config, seed)
    start = time.time()
    if kind == "full":
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["full_steps"], lr=config["model"]["lr"])
        result = {"rounds": [], "steps": section["full_steps"]}
    else:
        settings = discovery_settings(config, max_rounds=section["max_rounds"], patience=section["patience"])
        composer, result = additive_curriculum(composer, ctx, settings)
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["settle_steps"], lr=config["model"]["lr"])
        result["steps"] += section["settle_steps"]
    recovered = _recovered_ontology(world, scenario, composer, ctx.nodes)
    framed = _framed(world)
    train = set(world.splits["train"].tolist())
    gold_pairs = world.relations_of_pair()
    cand = ((composer.origin == ORIGIN["candidate"]) & composer.open).nonzero().flatten()
    if cand.numel():
        _, conf = composer.edge_options(cand)
        scores = (composer.edge_masses().detach()[cand] * conf).tolist()
        labels = [(int(h), int(a)) in gold_pairs for h, a in zip(composer.edge_heads()[cand].tolist(),
                                                                  composer.schedule.fillers[cand].tolist())]
        candidate_auc = _auc(scores, labels)
    else:
        candidate_auc = float("nan")
    gold_all = {(world.concept_node[h], world.relation_names[r], world.filler_node[a]) for h, r, a in world.edge_list() if h in framed}
    gold_train = {e for e in gold_all if e[0] in train}
    rec_edges = {(h, name, t) for name, pairs in recovered.items() for h, t in pairs}
    per_relation = {}
    for name in world.relation_names:
        g = world.pairs(name, heads=framed)
        per_relation[name] = jaccard(recovered.get(name, set()), g) if g else float("nan")
    # the learned part: candidate pairs sorted into relations, scored on the offered candidate universe
    t = scenario.table
    offered_all, offered_gold = set(), defaultdict(set)
    for h, a, o in zip(t.heads.tolist(), t.fillers.tolist(), t.open.tolist()):
        if o:
            pair = (world.concept_node[h], world.filler_node[a]); offered_all.add(pair)
            for r in gold_pairs.get((h, a), set()):
                offered_gold[world.relation_names[r]].add(pair)
    offered_jaccard = {name: jaccard(recovered.get(name, set()) & offered_all, g) for name, g in offered_gold.items() if g}
    audit = audit_data(world)
    row = {"part": "seed", "seed": seed, "condition": kind, "steps": result["steps"],
           "edges_all": _prf(rec_edges, gold_all), "edges_train": _prf({e for e in rec_edges if e[0] in train}, gold_train),
           "relation_jaccard": per_relation,
           "relations_recovered": sum(v >= 0.5 for v in per_relation.values() if v == v),
           "candidate_auc": candidate_auc, "offered_relation_jaccard": offered_jaccard,
           "offered_relation_jaccard_mean": float(np.mean(list(offered_jaccard.values()))) if offered_jaccard else float("nan"),
           "accepted_slots": sum(r["accept"] for r in result["rounds"]), "proposed_slots": len(result["rounds"]),
           "adopted": [r["adopted"] for r in result["rounds"] if r["accept"]],
           **_reasoning_probes(world, recovered, seed),
           "gold_probes": _reasoning_probes(world, {n: world.pairs(n, heads=framed) for n in world.relation_names}, seed),
           "val_fit": eval_fit(composer, ctx.heldout, world.splits["train"]),
           "test_fit": eval_fit(composer, audit, world.splits["test"]), "seconds": time.time() - start,
           "scenario": scenario.info}
    if result["rounds"]:
        row["riddle"] = _riddle_eval(world, scenario, composer, result["rounds"], ctx.nodes)
        row["decisions"] = _slot_decision_eval(world, Scenario(scenario.name, scenario.table, scenario.relation_names,
                                                               scenario.column_world, None,
                                                               list(range(len(world.relation_names)))), result["rounds"])
    return [row]


# =====================================================================================================
# E10.5 dreaming with injected corruptions
# =====================================================================================================

def _consolidated_state(world, scenario, config, seed, ctx) -> LearnableOntologyComposer:
    """Oracle-consolidated starting state: hidden relations' candidate edges trained in their own
    slots, then crystallized (a correct ontology to corrupt)."""
    composer, _ = run_method(world, scenario, config, seed, "oracle", mode="absent",
                             matched_steps=config["dream"]["consolidate_steps"])
    for slot in composer.slot_active.nonzero().flatten().tolist():
        composer.crystallize(slot, record={"hypothesis": "oracle"})
    return composer


def _inject(world, scenario, composer: LearnableOntologyComposer, ctx, seed, spec) -> dict[str, Any]:
    rng = random.Random(seed + 77)
    gold = set(world.edge_list())
    pools = _pools(world)
    train = sorted(world.splits["train"].tolist())
    injected: dict[str, Any] = {}
    # (i) wrong asserted edges among training heads
    n_known = int(((~composer.candidate) & ~composer.open).sum())
    wrong = []
    while len(wrong) < int(spec["wrong_edge_rate"] * n_known):
        h = rng.choice(train); c = rng.randrange(composer.base_relations); w = scenario.column_world[c]
        a = rng.choice(pools[w])
        if (h, w, a) not in gold:
            wrong.append((h, c, a))
    ids = composer.add_edges([h for h, _, _ in wrong], [c for _, c, _ in wrong], [a for _, _, a in wrong], mass=1.0,
                             origin="injected")
    injected["wrong_edges"] = ids
    # (ii) two crystallized relations merged into one slot
    slots = composer.slot_frozen.nonzero().flatten().tolist()
    if len(slots) >= 2:
        a_slot, b_slot = slots[0], slots[1]
        members_b = composer.slot_members(b_slot, hard_only=True)
        members_a = composer.slot_members(a_slot, hard_only=True)
        composer.reopen(a_slot); composer.reopen(b_slot)
        roles = composer.slot_role_vectors().detach()
        with torch.no_grad():
            composer.slot_roles[a_slot] = (roles[a_slot] + roles[b_slot]) / 2
        for e in torch.cat([members_a, members_b]).tolist():
            composer.set_edge_option(e, composer.slot_column(a_slot))
        composer.slot_active[b_slot] = False
        composer.crystallize(a_slot, min_mass=0.0, record={"hypothesis": "injected-merge"})
        injected["merged"] = {"slot": a_slot, "a_edges": members_a, "b_edges": members_b}
    else:
        a_slot = b_slot = -1
        injected["merged"] = None
    # (iii) a wrongly crystallized slot: random distractor candidates under a random operator
    distractors = [e for e in ((composer.origin == ORIGIN["candidate"]) & composer.open & (composer.assignment_fixed < 0)).nonzero().flatten().tolist()]
    rng.shuffle(distractors)
    gold_pairs = world.relations_of_pair()
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    junk = [e for e in distractors if (heads[e], fillers[e]) not in gold_pairs][:spec["wrong_slot_edges"]]
    slot = composer.add_blank_slot()
    for e in junk:
        composer.set_edge_option(e, composer.slot_column(slot))
    with torch.no_grad():
        composer.edge_mass[torch.tensor(junk)] = 1.0
    composer.crystallize(slot, min_mass=0.0, record={"hypothesis": "injected-wrong"})
    injected["wrong_slot"] = {"slot": slot, "edges": torch.tensor(junk)}
    injected["correct_slots"] = [s for s in slots if s not in {a_slot, b_slot}]
    return injected


def _dream_eval(world, composer: LearnableOntologyComposer, injected, ctx, before_correct: torch.Tensor) -> dict[str, Any]:
    masses = composer.edge_masses().detach()
    wrong = injected["wrong_edges"]
    out = {"wrong_edges_repaired": float((masses[wrong] < 0.25).float().mean())}
    correct = before_correct
    out["correct_edges_damaged"] = float((masses[correct] < 0.25).float().mean())
    columns, _ = composer.edge_options()
    merged = injected["merged"]
    if merged is not None:
        a, b = merged["a_edges"], merged["b_edges"]
        edges = torch.cat([a, b])
        gold = [0] * a.numel() + [1] * b.numel()
        pred = columns[edges].tolist()
        from vsa_embed.ontology_hypotheses import adjusted_rand_index
        out["merged_split_ari"] = adjusted_rand_index(gold, pred) if len(set(pred)) > 1 else 0.0
    else:
        out["merged_split_ari"] = float("nan")
    ws = injected["wrong_slot"]
    still = (columns[ws["edges"]] == composer.slot_column(ws["slot"])) & (masses[ws["edges"]] >= 0.25)
    out["wrong_slot_repaired"] = float(1 - still.float().mean())
    out["wrong_slot_reopened_or_removed"] = bool(not composer.slot_frozen[ws["slot"]] or float(still.float().mean()) < 0.5)
    correct_slots = injected["correct_slots"]
    out["correct_slot_disturbed"] = float(np.mean([not bool(composer.slot_frozen[s]) for s in correct_slots])) if correct_slots else float("nan")
    audit = audit_data(world)
    out["val_fit"] = eval_fit(composer, ctx.heldout, world.splits["train"])
    out["audit_fit_train"] = eval_fit(composer, audit, world.splits["train"])
    out["test_fit"] = eval_fit(composer, audit, world.splits["test"])
    return out


def part_dream(config: dict[str, Any], seed: int, every: int) -> list[dict[str, Any]]:
    section = config["dream"]
    world = build_world(config, seed)
    c = config["c"]
    scenario = hidden_scenario(world, hidden=c["hidden"], seed=seed, coverage=c["coverage"],
                               distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    ctx = make_context(world, scenario, seed)
    composer = _consolidated_state(world, scenario, config, seed, ctx)
    masses = composer.edge_masses().detach()
    correct = ((~composer.candidate) & (masses >= 0.25)).nonzero().flatten()
    injected = _inject(world, scenario, composer, ctx, seed, section)
    # indices may shift after add_edges inside _inject: recompute correct edges by key
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist(); origin = composer.origin.tolist()
    gold_pairs = world.relations_of_pair()
    current = composer.edge_masses().detach()
    correct = torch.tensor([e for e in range(len(heads)) if origin[e] in (ORIGIN["asserted"],) and (heads[e], fillers[e]) in gold_pairs
                            and float(current[e]) >= 0.25])
    before = _dream_eval(world, composer, injected, ctx, correct)
    total = section["post_steps"]
    settings = discovery_settings(config).dream
    passes = total // abs(every) if every else 0
    log = []
    start = time.time()
    if not every:
        extra = section["passes_for_off"] * settings.refit_steps
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=total + extra, lr=config["model"]["lr"])
    elif every < 0:
        # utility-threshold revisability only: reopen a crystallized slot whose held-out utility turned negative
        extra = section["passes_for_off"] * settings.refit_steps
        for p in range(max(1, total // -every)):
            train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=-every, lr=config["model"]["lr"])
            events = revisit_crystallized(composer, ctx)
            log.append({"pass": p, "revisions": [{"proposal": "revisit", **e} for e in events]})
            for e in events:
                if e["reopened"]:
                    train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=settings.refit_steps,
                                   lr=config["model"]["lr"])
                    composer.crystallize(e["slot"], record={"hypothesis": "revisited"})
        train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=extra, lr=config["model"]["lr"])
    else:
        for p in range(passes):
            train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=every, lr=config["model"]["lr"])
            composer, revisions = dream_pass(composer, ctx, settings, seed=seed + p)
            log.append({"pass": p, "revisions": [{k: v for k, v in r.items() if k not in {"edges", "partition"}} for r in revisions]})
    after = _dream_eval(world, composer, injected, ctx, correct)
    return [{"part": "dream", "seed": seed, "every": every, "passes": passes, "before": before, "after": after,
             "log": log, "seconds": time.time() - start}]


# =====================================================================================================
# E10.7 data variety vs self-confirmation; E10.8 continual additive learning
# =====================================================================================================

def part_variety(config: dict[str, Any], seed: int, views: int) -> list[dict[str, Any]]:
    section = config["variety"]
    world = build_world(config, seed, sources=section["sources"], sources_per_concept=views,
                        artifact_strength=section["artifact_strength"])
    obs = world.observation_splits
    attr = world.truth["source_attribute"]; rel = world.relation_id("has_attribute")
    train = world.splits["train"].tolist()
    artifacts = []
    for c in train:
        for s in sorted(set(world.observation_sources[c, obs["train"] + obs["val"]].tolist())):
            artifacts.append((c, rel, int(attr[s])))
    scenario = erasure_scenario(world, rate=section["erasure"], seed=seed, distractors=section["distractors"],
                                extra_candidates=artifacts)
    scenario.info["artifacts"] = artifacts
    ctx = make_context(world, scenario, seed)
    composer = make_composer(world, scenario, config, seed)
    train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["steps"], lr=config["model"]["lr"])
    rows = _edge_acceptance_rows(world, scenario, composer, ctx, config, seed, f"variety-{views}")
    for r in rows:
        r["part"] = "variety"; r["views"] = views
    summary = {"part": "variety_summary", "seed": seed, "views": views, **_candidate_eval(world, scenario, composer)}
    return rows + [summary]


def part_continual(config: dict[str, Any], seed: int, condition: str) -> list[dict[str, Any]]:
    section = config["continual"]
    c = config["c"]
    world = build_world(config, seed)
    scenario = hidden_scenario(world, hidden=c["hidden"], seed=seed, coverage=c["coverage"],
                               distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    ctx = make_context(world, scenario, seed)
    composer = make_composer(world, scenario, config, seed)
    order = [world.relation_id(n) for n in section["order"]]
    framed = _framed(world)
    gold_pairs = world.relations_of_pair()
    heads = composer.edge_heads().tolist(); fillers = composer.schedule.fillers.tolist()
    # Each open candidate arrives with the stage of its relation; distractors arrive with the stage of
    # the relation whose pool they were drawn for (same head, interleaved), approximated by head order.
    stage_of = {}
    open_edges = ((composer.origin == ORIGIN["candidate"]) & composer.open).nonzero().flatten().tolist()
    last_stage: dict[int, int] = {}
    for e in open_edges:
        rels = gold_pairs.get((heads[e], fillers[e]), set()) & set(order)
        if rels:
            stage_of[e] = order.index(min(rels)); last_stage[heads[e]] = stage_of[e]
        else:
            stage_of[e] = last_stage.get(heads[e], 0)
    gold_sets = {world.relation_names[r]: world.pairs(world.relation_names[r], heads=framed) for r in order}
    # Matching is scored on the offered candidate universe (comparable across conditions, whether or not
    # a condition adds rule-implied edges); the Jaccard against all framed heads is kept as `jaccard_framed`.
    offered_gold = _offered_pairs(world, scenario)
    offered_all = {(world.concept_node[h], world.filler_node[a]) for h, a, o in
                   zip(scenario.table.heads.tolist(), scenario.table.fillers.tolist(), scenario.table.open.tolist()) if o}
    settings = discovery_settings(config, max_rounds=1, patience=99)
    curve, history = [], []

    def snapshot(stage: int, comp: LearnableOntologyComposer) -> None:
        slots = comp.slot_frozen.nonzero().flatten().tolist() if condition != "plastic" else comp.slot_active.nonzero().flatten().tolist()
        pairs = {s: edge_pairs(comp, comp.slot_members(s), ctx.nodes) for s in slots}
        matched = {}
        for name, g in gold_sets.items():
            best = max(((jaccard(p & offered_all, offered_gold.get(name, set())), s) for s, p in pairs.items()),
                       default=(0.0, None))
            framed_best = max((jaccard(p, g) for p in pairs.values()), default=0.0)
            matched[name] = {"jaccard": best[0], "slot": best[1], "jaccard_framed": framed_best}
        curve.append({"stage": stage, "matched": sum(v["jaccard"] >= 0.5 for v in matched.values()), "per_relation": matched,
                      "val_fit": eval_fit(comp, ctx.heldout, world.splits["train"])})

    if condition == "all_at_once":
        composer, result = all_at_once(composer, ctx, discovery_settings(config), slots=c["all_at_once_slots"],
                                       steps=section["steps_per_stage"] * len(order))
        snapshot(len(order) - 1, composer)
    else:
        pending = torch.tensor([e for e in open_edges if stage_of[e] > 0])
        if pending.numel():
            composer.pin_masses(pending, 0.0)
        settings = discovery_settings(config, max_rounds=1, patience=99, steps_per_round=section["steps_per_stage"],
                                      dream_between_rounds=condition == "additive_dream")
        for stage in range(len(order)):
            arriving = torch.tensor([e for e in open_edges if stage_of[e] == stage])
            if stage > 0 and arriving.numel():
                composer.unpin(arriving)
            if condition == "plastic":
                composer.add_blank_slot(init_mass=settings.slot_init_mass)
                train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["steps_per_stage"],
                               lr=config["model"]["lr"])
            else:
                composer, result = additive_curriculum(composer, ctx, settings)
                history.append([{k: v for k, v in r.items() if k not in {"pairs", "predicted_by_adopted", "hypotheses"}}
                                for r in result["rounds"]])
            snapshot(stage, composer)
    final = curve[-1]["per_relation"]
    retention = {}
    for i, name in enumerate(gold_sets):
        at_stage = curve[min(i, len(curve) - 1)]["per_relation"][name]["jaccard"]
        retention[name] = {"at_arrival": at_stage, "final": final[name]["jaccard"]}
    return [{"part": "continual", "seed": seed, "condition": condition, "curve": curve, "retention": retention,
             "history": history}]


# =====================================================================================================
# orchestration
# =====================================================================================================

def _jobs(config: dict[str, Any]) -> list[tuple]:
    parts = config["parts"]
    seeds = [int(s) for s in config["seeds"]]
    jobs: list[tuple] = []
    for seed in seeds:
        if "a" in parts:
            grid = [tuple(c) for c in config["a"]["configs"]] if config["a"].get("configs") else \
                list(product(*[config["a"]["settings"]] * 4))
            size = config["a"].get("chunk", 27)
            jobs += [("a", seed, tuple(grid[i:i + size])) for i in range(0, len(grid), size)]
        if "b" in parts:
            jobs += [("b", seed, rate) for rate in config["b"]["rates"]]
        if "c" in parts:
            jobs += [("c", seed, mode) for mode in config["c"]["modes"]]
        if "seed" in parts:
            jobs += [("seed", seed, kind) for kind in config["seed_ontology"]["conditions"]]
        if "dream" in parts:
            jobs += [("dream", seed, every) for every in config["dream"]["every"]]
        if "variety" in parts:
            jobs += [("variety", seed, v) for v in config["variety"]["views"]]
        if "continual" in parts:
            jobs += [("continual", seed, cond) for cond in config["continual"]["conditions"]]
    return jobs


def _run_job(job: tuple, config: dict[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(int(config.get("num_threads", 1)))
    part, seed, arg = job
    start = time.time()
    if part == "a":
        rows = part_a(config, seed, list(arg))
    elif part == "b":
        rows = part_b(config, seed, float(arg))
    elif part == "c":
        rows = part_c(config, seed, arg)
    elif part == "seed":
        rows = part_seed(config, seed, arg)
    elif part == "dream":
        rows = part_dream(config, seed, int(arg))
    elif part == "variety":
        rows = part_variety(config, seed, int(arg))
    elif part == "continual":
        rows = part_continual(config, seed, arg)
    else:
        raise ValueError(part)
    for row in rows:
        row.setdefault("job_seconds", time.time() - start)
        row["job_part"] = part
        row["job_key"] = repr(arg) if not isinstance(arg, tuple) else f"chunk{len(arg)}:{arg[0]}"
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in (sorted(value, key=repr) if isinstance(value, set) else value)]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and value != value:
        return None
    return value


def run(config: dict[str, Any], output_dir: Path, *, workers: int | None = None) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    apply_thread_setting(config)
    jobs = _jobs(config)
    workers = workers or int(config.get("workers", 1))
    rows: list[dict[str, Any]] = []
    started = time.time()
    metrics = output_dir / "metrics.jsonl"
    if workers <= 1:
        for job in jobs:
            for row in _run_job(job, config):
                rows.append(row)
                with metrics.open("a") as handle:
                    handle.write(json.dumps(_jsonable(row)) + "\n")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_run_job, job, config): job for job in jobs}
            results = {}
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                print(f"[e10] done {futures[future][:2]} {futures[future][2] if not isinstance(futures[future][2], tuple) else len(futures[future][2])} "
                      f"({len(results)}/{len(jobs)}, {time.time() - started:.0f}s)", flush=True)
            for job in jobs:                     # write in job order: reproducible files
                for row in results[job]:
                    rows.append(row)
                    with metrics.open("a") as handle:
                        handle.write(json.dumps(_jsonable(row)) + "\n")
    from vsa_embed.experiments.e10_report import render_report, summarize
    timing = defaultdict(float)
    seen_jobs = set()
    for row in rows:
        key = (row.get("job_part"), row.get("seed"), row.get("job_key"))
        if key not in seen_jobs:
            seen_jobs.add(key); timing[row.get("job_part", row.get("part"))] += float(row.get("job_seconds", 0.0))
    summary = summarize([json.loads(json.dumps(_jsonable(r))) for r in rows], config)
    summary["job_cpu_seconds_by_part"] = dict(timing)
    summary["wall_seconds"] = time.time() - started
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, config))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", jobs=len(jobs), workers=workers,
                       wall_seconds=summary["wall_seconds"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.parts:
        config["parts"] = args.parts
    if args.seeds:
        config["seeds"] = args.seeds
    summary = run(config, args.output, workers=args.workers)
    print(json.dumps({"wall_seconds": summary.get("wall_seconds"), "verdicts": summary.get("verdicts")}, indent=2, default=str))


if __name__ == "__main__":
    main()
