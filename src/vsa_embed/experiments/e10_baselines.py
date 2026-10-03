"""E10 baselines for claim D (novelty check §5.5: D-B1, D-B2, D-B4, D-B5) on the E10 worlds.

The learnable ontology of E10 (`e10_self_semantics`) recovers erased edges, discovers hidden relations in
blank slots, names them by testing structural hypotheses on held-out data (the riddle step) and accepts its
own proposals by held-out self-tests. This module asks whether standard methods do as well on the *same*
worlds, scenarios and candidate pools:

- **recovery** (D-B2): erased-edge recovery on exactly the E10.0 (b) candidate pools (same scenario builder,
  same seeds) by AMIE-style rule closure, TransE, RotatE, ComplEx and an IterE-style loop (RotatE → operator
  axioms → closure triples injected → RotatE retrained). Every method is fit on 90% of the asserted edges;
  its decision threshold (for F1) is chosen on the other 10% plus distractors drawn by the scenario's rule.
  AUC / R-precision / F1 are compared with the learnable ontology's numbers of the committed E10.0 run
  (`runs/e10.0-v3/metrics.jsonl`, part b — the same candidate pools; the pool sizes are checked).
- **axioms** (D-B1): which Horn axioms hold, read from the observed graph (AMIE: PCA confidence) or from the
  relation operators (IterE-style: RotatE phases; the learnable ontology's own HRR role vectors), against the
  axioms that hold on the complete gold graph (standard confidence ≥ 0.9). Thresholds of the operator
  readings (and a tuned AMIE threshold) are chosen on the dev seeds only (7, 8, 9); AMIE's defaults are kept
  as the primary AMIE row.
- **discovery** (D-B1, D-B4, D-B5): the additive curriculum of E10.0 (c), exactly as configured, on
  (i) the absent and collapsed scenarios — the slots' captured pairs are then explained by AMIE rules,
  RotatE axioms and the slots' own operators, next to the riddle step's adopted hypothesis; (ii) two
  **null worlds** with no hidden relation (all relations asserted and only distractors offered; or the
  hidden relations absent and the offered pairs' tails permuted), where every accepted slot and every
  adopted rule is a false discovery; and (iii) a **fresh-holdout** control: each round's self-tests read
  freshly drawn validation observations instead of re-using one validation split.
- **edges** (D-B5): the (d) edge self-test (accept a proposed edge if removing it lowers held-out fit, lower
  bound > 0) with the re-used validation split vs a fresh draw per proposal vs Holm over all proposals
  (one-sided t-tests on the re-used split), on erasure 30% / 50% and on a **null edge world** (no edge
  erased: every proposal is a distractor).

Fresh observations are *simulated*: new noisy views of each concept's clean target, drawn exactly like the
world's observations (`observation_noise / √d` Gaussian noise, renormalized) from a separate generator. They
stand for "a new split of the same size"; in practice they would need more data.

Self-supervision: every baseline is fit on asserted edges (and, for the learnable ontology, training and
validation observations) only; gold edges, gold relation names and audit observations are read by the
evaluation code (`_gold*`, `eval_*`, the summary) alone.

    python -m vsa_embed.experiments.e10_baselines --config experiments/e10-self-semantics/e10-baselines.yaml \
        --output experiments/e10-self-semantics/runs/e10.0-baselines-v1 [--workers 2] [--parts recovery axioms edges discovery]
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from vsa_embed import kg_baselines as kg
from vsa_embed.experiments import e10_self_semantics as e10
from vsa_embed.learnable_ontology import ORIGIN, EdgeTable, train_composer
from vsa_embed.ontology_discovery import additive_curriculum
from vsa_embed.ontology_hypotheses import StructuralHypothesis, best_match, jaccard, property_holds
from vsa_embed.provenance import apply_thread_setting, prepare_output_dir, write_run_metadata
from vsa_embed.self_test import HeldOutData, base_rows, bootstrap_lower, edge_contributions, edit_effects
from vsa_embed.statistics import holm_adjust

PARTS = ("recovery", "axioms", "edges", "discovery")
RECOVERY_METHODS = ("prior_corr", "amie", "transe", "rotate", "complex", "itere")


# =====================================================================================================
# config and worlds
# =====================================================================================================

def resolve(config: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """The baseline config merged over its base E10 config (world, model, learn, selftest, discovery, c, …)."""
    base_path = Path(config["base_config"])
    if not base_path.is_absolute() and root is not None and not base_path.exists():
        base_path = root / base_path
    base = yaml.safe_load(base_path.read_text())
    merged = copy.deepcopy(base)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = copy.deepcopy(value)
    merged["base_experiment"] = base.get("experiment"); merged["base_stage"] = base.get("stage")
    return merged


def _entity_index(world: e10.OntologyWorld) -> dict[Hashable, int]:
    nodes = list(dict.fromkeys(list(world.concept_node) + list(world.filler_node)))
    return {n: i for i, n in enumerate(nodes)}


def _pairs_by_relation(edges: list[tuple[int, int, int]], world: e10.OntologyWorld, names: list[str]) -> dict[str, set]:
    out: dict[str, set] = {n: set() for n in names}
    for h, r, a in edges:
        out[names[r]].add((world.concept_node[h], world.filler_node[a]))
    return out


def _gold_relations(world: e10.OntologyWorld, heads: set[int] | None = None) -> dict[str, set]:
    """Gold pairs per relation (evaluation only)."""
    return {name: world.pairs(name, heads=heads) for name in world.relation_names}


def _distractors(world: e10.OntologyWorld, edges: list[tuple[int, int, int]], *, count: int, seed: int,
                 relation_confusion: float = 0.5, exclude: set | None = None,
                 relations: list[int] | None = None) -> list[tuple[int, int, int]]:
    """Distractors for `edges` drawn like `erasure_scenario`'s: a random filler of the same relation, or the
    same filler under another relation; never a gold edge (the gold set is read only to exclude it, as the
    scenario builder does)."""
    rng = random.Random(seed)
    gold = set(world.edge_list())
    pools = e10._pools(world)
    relations = relations if relations is not None else list(range(len(world.relation_names)))
    seen = set(exclude or set()) | set(edges)
    out = []
    for h, r, a in edges:
        for _ in range(count):
            for _attempt in range(20):
                if rng.random() < relation_confusion:
                    others = [q for q in relations if q != r and a in set(pools[q])]
                    if not others:
                        continue
                    edge = (h, rng.choice(others), a)
                else:
                    edge = (h, r, rng.choice(pools[r]))
                if edge not in gold and edge not in seen:
                    seen.add(edge); out.append(edge); break
    return out


def fresh_observations(world: e10.OntologyWorld, concepts: torch.Tensor, count: int, *, noise: float, seed: int) -> torch.Tensor:
    """`count` new noisy views of each concept's clean target (simulated fresh split; see the docstring)."""
    clean = world.truth["clean_targets"][concepts]
    d = clean.shape[-1]
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(clean.shape[0], count, d, generator=g) * noise / d**0.5
    return F.normalize(clean[:, None, :] + eps, dim=-1)


# =====================================================================================================
# recovery (D-B2) and axioms (D-B1)
# =====================================================================================================

def _candidate_pool(scenario: e10.Scenario) -> list[tuple[int, int, int]]:
    t = scenario.table
    rows = (t.candidate & ~t.open).nonzero().flatten().tolist()
    return [(int(t.heads[e]), scenario.column_world[int(t.relations[e])], int(t.fillers[e])) for e in rows]


def _threshold_metrics(scores: list[float], labels: list[bool], threshold: float) -> dict[str, float]:
    predicted = {i for i, s in enumerate(scores) if s >= threshold}
    gold = {i for i, y in enumerate(labels) if y}
    out = kg.prf(predicted, gold)
    k = len(gold)
    top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    out.update(auc=kg.auc(scores, labels), r_precision=sum(labels[i] for i in top) / max(1, k), threshold=threshold)
    return out


def _kge_triples(edges, entity, world) -> torch.Tensor:
    return torch.tensor([[entity[world.concept_node[h]], r, entity[world.filler_node[a]]] for h, r, a in edges],
                        dtype=torch.long).reshape(-1, 3)


def _rule_scores_on(rules_by_triple: dict, edges, world, names) -> list[float]:
    return [rules_by_triple.get((names[r], world.concept_node[h], world.filler_node[a]), 0.0) for h, r, a in edges]


def _itere(train_edges, world, names, entity, settings, *, threshold: float, iterations: int,
           seed: int, min_support: int = 2, max_axioms: int = 50) -> tuple[kg.KGEModel, list[dict[str, Any]]]:
    """IterE-style loop: RotatE → axioms read off its phases (score ≥ threshold) → closure triples (heads that
    are concepts) added to the training triples → RotatE retrained; returns the last model and per-iteration
    records. As in IterE, only axioms with instances in the observed graph are candidates (support ≥
    `min_support`), and at most `max_axioms` (the best-scoring ones) are injected per iteration."""
    observed = _pairs_by_relation(train_edges, world, names)
    supported = {s.rule for s in kg.mine_rules(observed, keep_all=True) if s.support >= min_support}
    universe = [r for r in kg.rule_universe(names, chains=True) if r.kind != "trivial" and r in supported]
    concepts = world_concepts(world)
    records = []
    triples = _kge_triples(train_edges, entity, world)
    added: dict[str, set] = defaultdict(set)
    model = None
    for it in range(iterations):
        model = kg.train_kge(triples, len(entity), len(names), kind="rotate", seed=seed + it, **settings)
        if it == iterations - 1:
            break
        phases = {name: model.phases()[i] for i, name in enumerate(names)}
        scored = {r: kg.rotate_axiom_score(r, phases) for r in universe}
        passing = [r for r in universe if scored[r] is not None and scored[r] >= threshold]
        keep = set(sorted(passing, key=lambda r: -scored[r])[:max_axioms])
        accepted = [r for r in passing if r in keep]          # graph order (the injected triples' order) kept
        closure = kg.axiom_closure(accepted, {n: observed[n] | added[n] for n in names})
        new_rows = []
        for name, pairs in closure.items():
            r = names.index(name)
            for h, t in sorted(pairs, key=repr):
                if h in concepts and h in entity and t in entity:
                    new_rows.append([entity[h], r, entity[t]]); added[name].add((h, t))
        records.append({"iteration": it, "axioms": [r.name for r in accepted], "closure_triples": len(new_rows)})
        if new_rows:
            triples = torch.cat([triples, torch.tensor(new_rows, dtype=torch.long)])
    return model, records


_CONCEPT_CACHE: dict[int, set] = {}


def world_concepts(world: e10.OntologyWorld) -> set:
    key = id(world)
    if key not in _CONCEPT_CACHE:
        _CONCEPT_CACHE[key] = set(world.concept_node)
    return _CONCEPT_CACHE[key]


def part_recovery(config: dict[str, Any], seed: int, rate: float) -> list[dict[str, Any]]:
    section = config["recovery"]
    world = e10.build_world(config, seed)
    scenario = e10.erasure_scenario(world, rate=rate, seed=seed, distractors=config["b"]["distractors"],
                                    erase_relations=config["b"].get("erase_relations"))
    names = list(world.relation_names)
    t = scenario.table
    asserted = [(int(h), int(r), int(a)) for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(),
                                                                 t.candidate.tolist()) if not c]
    rng = random.Random(seed + 11)
    order = list(range(len(asserted)))
    rng.shuffle(order)
    n_val = int(round(section["validation_fraction"] * len(asserted)))
    val_pos = [asserted[i] for i in sorted(order[:n_val])]
    train_edges = [asserted[i] for i in sorted(order[n_val:])]
    pool = _candidate_pool(scenario)
    erased = set(scenario.info["erased"])
    val_neg = _distractors(world, val_pos, count=config["b"]["distractors"], seed=seed + 13, exclude=set(pool))
    val_edges = val_pos + val_neg
    val_labels = [True] * len(val_pos) + [False] * len(val_neg)
    labels = [e in erased for e in pool]
    entity = _entity_index(world)
    observed = _pairs_by_relation(train_edges, world, names)
    amie = config["amie"]
    kge = dict(section["kge"])
    rows = []
    base = {"part": "recovery", "seed": seed, "rate": rate, "candidates": len(pool), "erased": len(erased),
            "train_edges": len(train_edges), "validation_edges": len(val_edges)}
    train_heads = set(world.splits["train"].tolist())
    for method in section["models"]:
        start = time.time()
        info: dict[str, Any] = {}
        method_val_labels = val_labels
        if method == "prior_corr":
            # No learning: cosine of a training concept's mean training observation with the candidate edge bound from
            # the *priors* (noisy stand-ins for pretrained vectors, as the learner starts from). Validation edges of
            # training heads only (the learner never reads other concepts' training observations).
            if world.atomic_prior is None or scenario.relation_prior is None or world.target_dimension != world.dimension:
                continue
            mean_obs = F.normalize(world.observations[:, world.observation_splits["train"]].mean(1), dim=-1)

            def correlation(edges: list[tuple[int, int, int]]) -> list[float]:
                if not edges:
                    return []
                h, r, a = (torch.tensor([e[i] for e in edges]) for i in range(3))
                bound = kg.circular_convolution(scenario.relation_prior[r], world.atomic_prior[a])
                return F.cosine_similarity(mean_obs[h], bound, dim=-1).tolist()
            keep = [i for i, e in enumerate(val_edges) if e[0] in train_heads]
            scores = correlation(pool)
            val_scores = correlation([val_edges[i] for i in keep])
            method_val_labels = [val_labels[i] for i in keep]
        elif method == "amie":
            rules = kg.mine_rules(observed, min_support=amie["min_support"], min_head_coverage=amie["min_head_coverage"],
                                  min_pca=amie["min_pca"])
            closure = kg.closure_scores(rules, observed)
            scores = _rule_scores_on(closure, pool, world, names)
            val_scores = _rule_scores_on(closure, val_edges, world, names)
            info = {"rules": len(rules), "top_rules": [(s.rule.name, round(s.pca_confidence, 3), s.support) for s in rules[:12]]}
        else:
            triples = _kge_triples(train_edges, entity, world)
            if method == "itere":
                model, records = _itere(train_edges, world, names, entity, kge, threshold=config["thresholds"]["rotate"],
                                        iterations=config["axioms"]["itere_iterations"], seed=seed,
                                        min_support=amie["min_support"],
                                        max_axioms=int(config["axioms"].get("itere_max_axioms", 50)))
                info = {"iterations": records}
            else:
                model = kg.train_kge(triples, len(entity), len(names), kind=method, seed=seed, **kge)
            scores = kg.kge_scores(model, _kge_triples(pool, entity, world)).tolist()
            val_scores = kg.kge_scores(model, _kge_triples(val_edges, entity, world)).tolist()
        threshold = kg.best_threshold(val_scores, method_val_labels)
        rows.append({**base, "method": method, **_threshold_metrics(scores, labels, threshold),
                     "validation_auc": kg.auc(val_scores, method_val_labels), "info": info, "seconds": time.time() - start})
    return rows


def _axiom_universe(names: list[str]) -> list[kg.Rule]:
    return [r for r in kg.rule_universe(names, chains=True) if r.kind != "trivial"]


def part_axioms(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Axiom scores on the 30%-erasure scenario's asserted graph: AMIE (PCA confidence) and RotatE phases,
    with the gold flag of every rule (evaluation only); thresholds are applied in the summary."""
    section = config["axioms"]
    world = e10.build_world(config, seed)
    scenario = e10.erasure_scenario(world, rate=section["erasure"], seed=seed, distractors=config["b"]["distractors"],
                                    erase_relations=config["b"].get("erase_relations"))
    names = list(world.relation_names)
    t = scenario.table
    asserted = [(int(h), int(r), int(a)) for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(),
                                                                 t.candidate.tolist()) if not c]
    observed = _pairs_by_relation(asserted, world, names)
    start = time.time()
    mined = {s.rule: s for s in kg.mine_rules(observed, keep_all=True)}
    entity = _entity_index(world)
    model = kg.train_kge(_kge_triples(asserted, entity, world), len(entity), len(names), kind="rotate", seed=seed,
                         **config["recovery"]["kge"])
    phases = {name: model.phases()[i] for i, name in enumerate(names)}
    universe = _axiom_universe(names)
    gold = kg.gold_axioms(_gold_relations(world), min_confidence=section["gold_confidence"], min_support=section["gold_support"])
    amie = config["amie"]
    row = {"part": "axioms", "seed": seed, "dev": seed in config.get("dev_seeds", []), "rules": [r.name for r in universe],
           "kinds": [r.kind for r in universe], "gold": [r in gold for r in universe],
           "amie_pca": [mined[r].pca_confidence if r in mined else 0.0 for r in universe],
           "amie_std": [mined[r].std_confidence if r in mined else 0.0 for r in universe],
           "amie_passes": [bool(r in mined and mined[r].passes(min_support=amie["min_support"],
                                                                 min_head_coverage=amie["min_head_coverage"],
                                                                 min_pca=amie["min_pca"])) for r in universe],
           "rotate": [kg.rotate_axiom_score(r, phases) for r in universe], "gold_axioms": sorted(r.name for r in gold),
           "seconds": time.time() - start}
    # closure recovery of the erased edges by each reading's accepted axioms (thresholds known for main seeds)
    thresholds = config.get("thresholds")
    if thresholds and not row["dev"]:
        erased = {(world.concept_node[h], names[r], world.filler_node[a]) for h, r, a in scenario.info["erased"]}
        gold_all = {(world.concept_node[h], names[r], world.filler_node[a]) for h, r, a in world.edge_list()}
        readings = {"amie_default": [r for r, p in zip(universe, row["amie_passes"]) if p],
                    "amie_dev": [r for r, s, p in zip(universe, row["amie_pca"], row["amie_std"])
                                 if s >= thresholds["amie"] and mined.get(r) is not None and mined[r].support >= amie["min_support"]],
                    # as in the IterE loop: supported candidates only, at most `itere_max_axioms` best-scoring ones
                    "rotate": [r for r, s in zip(universe, row["rotate"]) if s is not None and s >= thresholds["rotate"]
                               and mined.get(r) is not None and mined[r].support >= amie["min_support"]]}
        cap = int(section.get("itere_max_axioms", 50))
        if len(readings["rotate"]) > cap:
            score_of = dict(zip(universe, row["rotate"]))
            keep = set(sorted(readings["rotate"], key=lambda r: -score_of[r])[:cap])
            readings["rotate"] = [r for r in readings["rotate"] if r in keep]
        closures = {}
        for key, rules in readings.items():
            closure = kg.axiom_closure(rules, observed)
            predicted = {(h, name, t2) for name, pairs in closure.items() for h, t2 in pairs if h in world_concepts(world)}
            closures[key] = {"axioms": len(rules), "closure": len(predicted), "erased_recall": len(predicted & erased) / max(1, len(erased)),
                             "precision_vs_gold": len(predicted & gold_all) / len(predicted) if predicted else float("nan")}
        row["closures"] = closures
    return [row]


# =====================================================================================================
# edges (D-B5): re-used vs fresh vs Holm acceptance
# =====================================================================================================

def null_edge_scenario(world: e10.OntologyWorld, *, seed: int, fraction: float, distractors: int) -> e10.Scenario:
    """Every gold edge asserted; distractors (drawn like the erasure scenario's) for a `fraction` of the
    training heads' edges are the only candidates — every proposal is false."""
    rng = random.Random(seed + 17)
    train = set(world.splits["train"].tolist())
    framed = e10._framed(world)
    sampled = [e for e in world.edge_list() if e[0] in train and e[0] in framed and rng.random() < fraction]
    extra = _distractors(world, sampled, count=distractors, seed=seed + 19)
    scenario = e10.erasure_scenario(world, rate=0.0, seed=seed, distractors=distractors, extra_candidates=extra)
    scenario.name = "null-edges"
    return scenario


@torch.no_grad()
def _removal_utilities(composer, proposals: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    """Per proposal × observation: fit lost when the edge is removed (positive = the edge helps)."""
    heads = composer.edge_heads()[proposals]
    z = base_rows(composer, heads)
    delta = -edge_contributions(composer, proposals)
    base = F.cosine_similarity(composer.predict_from_raw(z)[:, None, :], observations, dim=-1)
    after = F.cosine_similarity(composer.predict_from_raw(z + delta)[:, None, :], observations, dim=-1)
    return base - after


def _t_pvalues(values: torch.Tensor) -> list[float]:
    from scipy.stats import t as student
    out = []
    for row in values.double():
        n = row.numel(); mean = float(row.mean()); sd = float(row.std(unbiased=True)) if n > 1 else 0.0
        if sd == 0:
            out.append(0.0 if mean > 0 else 1.0)
            continue
        out.append(float(student.sf(mean / (sd / n**0.5), df=n - 1)))
    return out


def part_edges(config: dict[str, Any], seed: int, kind: str) -> list[dict[str, Any]]:
    section, st = config["edges"], config["selftest"]
    world = e10.build_world(config, seed)
    if kind == "null":
        scenario = null_edge_scenario(world, seed=seed, fraction=section["null_fraction"], distractors=config["b"]["distractors"])
    else:
        scenario = e10.erasure_scenario(world, rate=float(kind), seed=seed, distractors=config["b"]["distractors"],
                                        erase_relations=config["b"].get("erase_relations"))
    ctx = e10.make_context(world, scenario, seed)
    composer = e10.make_composer(world, scenario, config, seed)
    start = time.time()
    train_composer(composer, ctx.train_concepts, ctx.train_targets, steps=section["steps"], lr=config["model"]["lr"])
    masses = composer.edge_masses().detach()
    proposals = ((composer.origin == ORIGIN["candidate"]) & (masses >= st["propose_mass"])).nonzero().flatten()
    heads = composer.edge_heads()[proposals]
    keep = torch.tensor([ctx.heldout.has(int(h)) for h in heads.tolist()], dtype=torch.bool)
    proposals, heads = proposals[keep], heads[keep]
    row: dict[str, Any] = {"part": "edges", "seed": seed, "scenario": kind, "dev": seed in config.get("dev_seeds", []),
                           "proposals": int(proposals.numel())}
    if proposals.numel():
        gold = set(world.edge_list())
        fillers = composer.schedule.fillers[proposals].tolist(); columns = composer.schedule.relations[proposals].tolist()
        labels = [(int(h), scenario.column_world[c], int(a)) in gold for h, c, a in zip(heads.tolist(), columns, fillers)]
        reused = -edit_effects(composer, ctx.heldout, heads, -edge_contributions(composer, proposals))
        mean_r, lower_r = bootstrap_lower(reused, resamples=st["resamples"], seed=seed, alpha=st["alpha"])
        pvalues = _t_pvalues(reused)
        holm = holm_adjust(pvalues)
        row.update(gold=labels, reused=(lower_r > 0).tolist(), holm=[p < st["alpha"] for p in holm],
                   ttest=[p < st["alpha"] for p in pvalues], reused_mean=mean_r.tolist())
        if "clean_targets" in world.truth:     # simulated fresh split (synthetic worlds only: needs the clean targets)
            fresh_obs = fresh_observations(world, heads, section["fresh_observations"],
                                           noise=float(config["world"]["observation_noise"]), seed=seed * 1000 + 7)
            fresh = _removal_utilities(composer, proposals, fresh_obs)
            mean_f, lower_f = bootstrap_lower(fresh, resamples=st["resamples"], seed=seed, alpha=st["alpha"])
            row.update(fresh=(lower_f > 0).tolist(), fresh_mean=mean_f.tolist())
    if kind == "0.3":
        names = list(scenario.relation_names)
        roles = composer.known_role_vectors().detach()
        role_of = {name: roles[i] for i, name in enumerate(names)}
        universe = _axiom_universe(names)
        gold_ax = kg.gold_axioms(_gold_relations(world), min_confidence=config["axioms"]["gold_confidence"],
                                 min_support=config["axioms"]["gold_support"])
        row.update(axiom_rules=[r.name for r in universe], axiom_kinds=[r.kind for r in universe],
                   axiom_gold=[r in gold_ax for r in universe], hrr=[kg.hrr_axiom_score(r, role_of) for r in universe])
    row["seconds"] = time.time() - start
    return [row]


# =====================================================================================================
# discovery (D-B1 identification, D-B4 null worlds, D-B5 fresh holdout)
# =====================================================================================================

def null_distractor_scenario(world: e10.OntologyWorld, *, hidden: list[str], seed: int, coverage: float,
                             distractor_ratio: float, max_slots: int) -> e10.Scenario:
    """All relations asserted; the open candidates are distractors only, drawn like the absent scenario's
    (same coverage, one more distractor per sampled edge instead of the true pair) — no hidden relation."""
    rng = random.Random(seed)
    hidden_ids = [world.relation_id(n) for n in hidden]
    framed, train = e10._framed(world), set(world.splits["train"].tolist())
    gold_pairs = set(world.relations_of_pair())
    pools = e10._pools(world)
    asserted, open_edges, seen = [], [], set()
    for h, r, a in world.edge_list():
        if h not in framed:
            continue
        asserted.append((h, r, a))
        if r in hidden_ids and h in train and rng.random() < coverage:
            count = 1 + int(distractor_ratio) + (rng.random() < distractor_ratio - int(distractor_ratio))
            for _ in range(count):
                for _attempt in range(20):
                    b = rng.choice(pools[r])
                    if (h, b) not in gold_pairs and (h, b) not in seen:
                        seen.add((h, b)); open_edges.append((h, -1, b)); break
    relations = list(range(len(world.relation_names)))
    table = EdgeTable.build(world.concept_count, len(relations), max_slots=max_slots, asserted=asserted, candidates=open_edges)
    return e10.Scenario("null-distractors", table, list(world.relation_names), relations,
                        e10._relation_prior(world, relations, seed), [], {"open_candidates": len(open_edges)})


def null_permuted_scenario(world: e10.OntologyWorld, *, hidden: list[str], seed: int, coverage: float,
                           distractor_ratio: float, max_slots: int) -> e10.Scenario:
    """The absent scenario with the offered pairs' tails permuted (no offered pair is a gold pair)."""
    base = e10.hidden_scenario(world, hidden=hidden, seed=seed, coverage=coverage, distractor_ratio=distractor_ratio,
                               max_slots=max_slots)
    t = base.table
    gold_pairs = set(world.relations_of_pair())
    asserted = [(int(h), int(r), int(a)) for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(),
                                                                 t.candidate.tolist()) if not c]
    opened = [(int(h), int(a)) for h, a, c in zip(t.heads.tolist(), t.fillers.tolist(), t.candidate.tolist()) if c]
    rng = random.Random(seed + 23)
    tails = [a for _, a in opened]
    rng.shuffle(tails)
    seen, open_edges = set(), []
    for (h, _), a in zip(opened, tails):
        for _attempt in range(50):
            if (h, a) not in gold_pairs and (h, a) not in seen:
                break
            a = rng.choice(tails)
        if (h, a) not in gold_pairs and (h, a) not in seen:
            seen.add((h, a)); open_edges.append((h, -1, a))
    table = EdgeTable.build(world.concept_count, t.relation_count, max_slots=max_slots, asserted=asserted, candidates=open_edges)
    return e10.Scenario("null-permuted", table, base.relation_names, base.column_world, base.relation_prior, base.hidden,
                        {"open_candidates": len(open_edges), "permuted_from": len(opened)})


def _seed_relations(scenario: e10.Scenario, world: e10.OntologyWorld) -> dict[str, set]:
    """The model's own seed relations at the start (asserted and open-asserted edges by nominal label)."""
    t = scenario.table
    out: dict[str, set] = defaultdict(set)
    for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(), t.candidate.tolist()):
        if not c:
            out[scenario.relation_names[r]].add((world.concept_node[h], world.filler_node[a]))
    return dict(out)


def _explanations(seed_names: list[str], slot: str = "slot") -> list[tuple[str, kg.Rule]]:
    """(E10.6 hypothesis name, rule) of every structural explanation of a slot that is a Horn rule."""
    out = [("symmetric", kg.Rule(slot, ((slot, True),))), ("transitive", kg.Rule(slot, ((slot, False), (slot, False))))]
    for p in seed_names:
        out.append((f"inverse_of:{p}", kg.Rule(slot, ((p, True),))))
        out.append((f"sub_relation_of:{p}", kg.Rule(p, ((slot, False),))))
        for q in seed_names:
            out.append((f"composition_of:{p}∘{q}", kg.Rule(slot, ((p, False), (q, False)))))
    return out


def _explain_slot(pairs: set, seeds: dict[str, set], world: e10.OntologyWorld, config: dict[str, Any], *,
                  slot_role: torch.Tensor | None, roles: dict[str, torch.Tensor], seed: int,
                  thresholds: dict[str, float]) -> dict[str, Any]:
    """AMIE, RotatE (IterE-style) and own-operator explanations of one slot's captured pairs (no gold)."""
    names = sorted(seeds)
    relations = {**{n: seeds[n] for n in names}, "slot": set(pairs)}
    candidates = _explanations(names)
    amie = config["amie"]
    out: dict[str, Any] = {}
    index = {r: {h for h, _ in p} for r, p in relations.items()}
    cache: dict = {}
    scored = []
    for hyp, rule in candidates:
        s = kg.score_rule(rule, relations, cache=cache, head_index=index)
        if s is not None and s.passes(min_support=amie["min_support"], min_head_coverage=amie["min_head_coverage"],
                                      min_pca=amie["min_pca"]):
            closure_type = rule.head == "slot"
            scored.append((s.pca_confidence, closure_type, s.support, hyp, s))
    scored.sort(key=lambda x: (-x[0], not x[1], -x[2], x[3]))
    out["amie"] = {"adopted": scored[0][3] if scored else None, "passing": len(scored),
                   "confidence": scored[0][0] if scored else None}
    if scored and scored[0][4].rule.head == "slot":
        predicted = kg.rule_predictions(scored[0][4].rule, relations) or set()
        out["amie"]["predicted"] = sorted(predicted - set(pairs), key=repr)
    # IterE-style: RotatE over the seed relations + the slot, axioms read off the phases
    nodes = list(dict.fromkeys([n for p in relations.values() for pair in p for n in pair]))
    entity = {n: i for i, n in enumerate(nodes)}
    rel_names = names + ["slot"]
    triples = torch.tensor([[entity[h], i, entity[t]] for i, n in enumerate(rel_names) for h, t in sorted(relations[n], key=repr)],
                           dtype=torch.long).reshape(-1, 3)
    settings = dict(config["recovery"]["kge"]); settings["epochs"] = int(config["discovery_baselines"]["rotate_epochs"])
    model = kg.train_kge(triples, len(entity), len(rel_names), kind="rotate", seed=seed, **settings)
    phases = {n: model.phases()[i] for i, n in enumerate(rel_names)}
    rot = sorted(((kg.rotate_axiom_score(rule, phases), hyp) for hyp, rule in candidates), key=lambda x: (-x[0], x[1]))
    out["rotate"] = {"adopted": rot[0][1] if rot and rot[0][0] >= thresholds["rotate"] else None, "best": rot[0][1],
                     "score": rot[0][0]}
    if slot_role is not None:
        own = {**roles, "slot": slot_role}
        hrr = sorted(((kg.hrr_axiom_score(rule, own), hyp) for hyp, rule in candidates
                      if kg.hrr_axiom_score(rule, own) is not None), key=lambda x: (-x[0], x[1]))
        out["hrr"] = {"adopted": hrr[0][1] if hrr and hrr[0][0] >= thresholds["hrr"] else None, "best": hrr[0][1] if hrr else None,
                      "score": hrr[0][0] if hrr else None}
    return out


def _judge(hypothesis: str | None, name: str | None, gold_rel: dict[str, set], alias: dict[str, str],
           designed: tuple[str, ...]) -> dict[str, Any]:
    """Is an adopted explanation true of the slot's best-matching gold relation, and its designed property?
    (evaluation only)."""
    if hypothesis is None or name is None:
        return {"adopted": hypothesis, "true": False, "designed": False}
    kind, _, args = hypothesis.partition(":")
    hyp = StructuralHypothesis(kind, tuple(args.split("∘")) if args else ())
    mapped = kind if not args else f"{kind}:" + "∘".join(alias.get(a, a) for a in args.split("∘"))
    return {"adopted": hypothesis, "mapped": mapped, "true": bool(property_holds(hyp, gold_rel[name], gold_rel, alias)),
            "designed": mapped in set(designed)}


def part_discovery(config: dict[str, Any], seed: int, arg: tuple[str, str]) -> list[dict[str, Any]]:
    world_kind, holdout = arg
    section = config["c"]
    world = e10.build_world(config, seed)
    common = dict(seed=seed, coverage=section["coverage"], distractor_ratio=section["distractor_ratio"], max_slots=section["max_slots"])
    if world_kind == "absent":
        scenario = e10.hidden_scenario(world, hidden=section["hidden"], **common)
    elif world_kind == "collapsed":
        scenario = e10.collapsed_scenario(world, collapse=section["collapse"], seed=seed, max_slots=section["max_slots"])
    elif world_kind == "null_distractors":
        scenario = null_distractor_scenario(world, hidden=section["hidden"], **common)
    elif world_kind == "null_permuted":
        scenario = null_permuted_scenario(world, hidden=section["hidden"], **common)
    else:
        raise ValueError(world_kind)
    ctx = e10.make_context(world, scenario, seed)
    settings = e10.discovery_settings(config)
    on_stage = None
    if holdout == "fresh":
        noise = float(config["world"]["observation_noise"])
        k = len(world.observation_splits["val"])

        def on_stage(round_index: int, composer):        # a fresh validation split for every round's tests
            obs = fresh_observations(world, ctx.heldout.concepts, k, noise=noise, seed=seed * 1000 + 101 + round_index)
            ctx.heldout = HeldOutData(ctx.heldout.concepts, obs, name=f"fresh-{round_index}")
            return None
    start = time.time()
    composer = e10.make_composer(world, scenario, config, seed)
    composer, result = additive_curriculum(composer, ctx, settings, on_stage=on_stage)
    learn_seconds = time.time() - start
    seeds = _seed_relations(scenario, world)
    roles = {name: composer.known_role_vectors().detach()[i] for i, name in enumerate(scenario.relation_names)}
    thresholds = config["thresholds"]
    # -------- evaluation only below --------
    framed = e10._framed(world)
    gold_rel = _gold_relations(world)
    gold_framed = _gold_relations(world, heads=framed)
    offered = e10._offered_pairs(world, scenario)
    columns_gold = e10._gold_columns(world, scenario, composer, ctx.nodes)
    alias = e10._alias(world, scenario, columns_gold, composer)
    riddle = {e["slot"]: e for e in e10._riddle_eval(world, scenario, composer, result["rounds"], ctx.nodes)} \
        if world_kind in {"absent", "collapsed"} else {}
    slots = []
    for record in result["rounds"]:
        pairs = set(map(tuple, record["pairs"]))
        column = composer.slot_column(record["slot"])
        slot_role = composer.option_roles()[column].detach()
        expl = _explain_slot(pairs, seeds, world, config, slot_role=slot_role, roles=roles, seed=seed + record["slot"],
                             thresholds=thresholds)
        if offered:
            name, score = best_match(pairs, offered) if pairs else (None, 0.0)
        else:
            name, score = (None, 0.0)
        designed = world.gold_properties.get(name, ()) if name else ()
        precision = (sum(1 for p in pairs if any(p in g for g in offered.values())) / len(pairs)) if pairs and offered else 0.0
        entry = {"slot": record["slot"], "accept": bool(record["accept"]), "size": len(pairs), "best_relation": name,
                 "best_jaccard": score, "gold_precision": precision, "designed": list(designed),
                 "riddle": _judge(record["adopted"], name, gold_rel, alias, designed),
                 "amie": _judge(expl["amie"]["adopted"], name, gold_rel, alias, designed),
                 "rotate": _judge(expl["rotate"]["adopted"], name, gold_rel, alias, designed),
                 "hrr": _judge(expl.get("hrr", {}).get("adopted"), name, gold_rel, alias, designed),
                 "amie_passing": expl["amie"]["passing"], "rotate_best": expl["rotate"]["best"],
                 "rotate_score": expl["rotate"]["score"], "hrr_best": expl.get("hrr", {}).get("best"),
                 "hrr_score": expl.get("hrr", {}).get("score")}
        if name and record["slot"] in riddle:
            entry["riddle_rule_precision"] = riddle[record["slot"]].get("rule_prediction_precision")
        if name and expl["amie"].get("predicted") is not None:
            predicted = {p for p in map(tuple, expl["amie"]["predicted"]) if p[0] in world_concepts(world)}
            entry["amie_rule_precision"] = len(predicted & gold_rel[name]) / len(predicted) if predicted else None
            entry["amie_closure_jaccard"] = jaccard(pairs | {p for p in predicted if p[0] in framed}, gold_framed[name])
        entry["slot_jaccard_framed"] = jaccard(pairs, gold_framed[name]) if name else None
        slots.append(entry)
    # AMIE on the offered candidates as one pseudo-relation: rules involving it (false in the null worlds)
    t = scenario.table
    offered_pairs = {(world.concept_node[h], world.filler_node[a]) for h, a, o in zip(t.heads.tolist(), t.fillers.tolist(),
                                                                                       t.open.tolist()) if o}
    pseudo = {**seeds, "offered": offered_pairs}
    amie = config["amie"]
    offered_rules = [s for s in kg.mine_rules(pseudo, min_support=amie["min_support"], min_head_coverage=amie["min_head_coverage"],
                                              min_pca=amie["min_pca"], chains=False)
                     if s.rule.head == "offered" or any(a[0] == "offered" for a in s.rule.body)]
    recovery = e10._slot_relation_recovery(world, scenario, composer, ctx.nodes) if scenario.hidden else {}
    row = {"part": "discovery", "seed": seed, "world": world_kind, "holdout": holdout, "slots": slots,
           "accepted_slots": sum(s["accept"] for s in slots), "proposed_slots": len(slots),
           "offered_rules": [(s.rule.name, round(s.pca_confidence, 3), s.support) for s in offered_rules],
           "relation_recovery": recovery, "open_candidates": int(t.open.sum()), "learn_seconds": learn_seconds,
           "seconds": time.time() - start}
    return [row]


# =====================================================================================================
# runner
# =====================================================================================================

def _jobs(config: dict[str, Any], *, dev: bool) -> list[tuple]:
    parts = config["parts"]
    seeds = [int(s) for s in (config["dev_seeds"] if dev else config["seeds"])]
    jobs: list[tuple] = []
    for seed in seeds:
        if dev:
            jobs += [("axioms", seed, None), ("edges", seed, "0.3")]
            continue
        if "recovery" in parts:
            jobs += [("recovery", seed, float(rate)) for rate in config["recovery"]["rates"]]
        if "axioms" in parts:
            jobs.append(("axioms", seed, None))
        if "edges" in parts:
            jobs += [("edges", seed, kind) for kind in config["edges"]["scenarios"]]
        if "discovery" in parts:
            for world_kind, holdouts in config["discovery_baselines"]["worlds"].items():
                jobs += [("discovery", seed, (world_kind, h)) for h in holdouts]
    return jobs


def _run_job(job: tuple, config: dict[str, Any]) -> list[dict[str, Any]]:
    torch.set_num_threads(int(config.get("num_threads", 1)))
    part, seed, arg = job
    start = time.time()
    if part == "recovery":
        rows = part_recovery(config, seed, arg)
    elif part == "axioms":
        rows = part_axioms(config, seed)
    elif part == "edges":
        rows = part_edges(config, seed, arg)
    elif part == "discovery":
        rows = part_discovery(config, seed, arg)
    else:
        raise ValueError(part)
    for row in rows:
        row["job_seconds"] = time.time() - start
        row["job_part"] = part
        row["job_key"] = repr(arg)
    return rows


def dev_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    """F1-optimal thresholds of the operator readings (and a tuned AMIE threshold) on the dev seeds, pooled."""
    axioms = [r for r in rows if r["part"] == "axioms"]
    edges = [r for r in rows if r["part"] == "edges" and r.get("hrr") is not None]
    pool = lambda key, src, gold_key: ([v if v is not None else -2.0 for r in src for v in r[key]],  # noqa: E731
                                       [g for r in src for g in r[gold_key]])
    rot_s, rot_g = pool("rotate", axioms, "gold")
    amie_s, amie_g = pool("amie_pca", axioms, "gold")
    hrr_s, hrr_g = pool("hrr", edges, "axiom_gold")
    return {"rotate": kg.best_threshold(rot_s, rot_g), "amie": kg.best_threshold(amie_s, amie_g),
            "hrr": kg.best_threshold(hrr_s, hrr_g) if hrr_s else float("inf")}


def _execute(jobs: list[tuple], config: dict[str, Any], workers: int, log) -> dict[tuple, list[dict[str, Any]]]:
    results: dict[tuple, list[dict[str, Any]]] = {}
    started = time.time()
    if workers <= 1:
        for job in jobs:
            results[job] = _run_job(job, config)
            log(f"[e10-baselines] done {job} ({len(results)}/{len(jobs)}, {time.time() - started:.0f}s)")
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_run_job, job, config): job for job in jobs}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            log(f"[e10-baselines] done {futures[future]} ({len(results)}/{len(jobs)}, {time.time() - started:.0f}s)")
    return results


def run(config: dict[str, Any], output_dir: Path, *, workers: int | None = None, log=print) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    apply_thread_setting(config)
    workers = workers or int(config.get("workers", 1))
    started = time.time()
    dev_jobs = _jobs(config, dev=True)
    dev_results = _execute(dev_jobs, config, workers, log)
    dev_rows = [row for job in dev_jobs for row in dev_results[job]]
    config = {**config, "thresholds": dev_thresholds(dev_rows)}
    log(f"[e10-baselines] dev thresholds {config['thresholds']}")
    jobs = _jobs(config, dev=False)
    results = _execute(jobs, config, workers, log)
    rows = dev_rows + [row for job in jobs for row in results[job]]
    with (output_dir / "metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(e10._jsonable(row)) + "\n")
    from vsa_embed.experiments.e10_report import render_baselines, summarize_baselines
    rows = [json.loads(json.dumps(e10._jsonable(r))) for r in rows]
    summary = summarize_baselines(rows, config)
    summary["thresholds"] = config["thresholds"]
    summary["wall_seconds"] = time.time() - started
    timing: dict[str, float] = defaultdict(float)
    for row in rows:
        timing[row["job_part"] + ("-dev" if row.get("dev") else "")] += float(row.get("job_seconds", 0.0))
    summary["job_cpu_seconds_by_part"] = dict(timing)
    (output_dir / "summary.json").write_text(json.dumps(e10._jsonable(summary), indent=2) + "\n")
    (output_dir / "report.md").write_text(render_baselines(summary, config))
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", jobs=len(dev_jobs) + len(jobs),
                       workers=workers, wall_seconds=summary["wall_seconds"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", default=None, choices=list(PARTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    config = resolve(yaml.safe_load(args.config.read_text()))
    if args.parts:
        config["parts"] = args.parts
    if args.seeds:
        config["seeds"] = args.seeds
    summary = run(config, args.output, workers=args.workers)
    print(json.dumps({"wall_seconds": summary.get("wall_seconds"), "thresholds": summary.get("thresholds")}, indent=2, default=str))


if __name__ == "__main__":
    main()
