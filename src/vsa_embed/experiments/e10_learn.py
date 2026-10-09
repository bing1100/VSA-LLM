"""E10.L — the *learn* tool, evaluated (decision 63, WP TK-L; pre-registration
`experiments/e10-self-semantics/e10-learn/preregistration.md`). Method: `vsa_embed.learn` (decompose, propose, verify;
the M3 null-calibrated acceptance test).

**Erased-edge recovery** (`erased`, `synthetic`). About 20% of the curated edges of seen concepts are erased (content
relations only; the same erased edges and probe concepts for every checkpoint seed of a track). The learner then sees:
- the store: the trained C5 composer's dictionary (atomics, operator) with typed candidates from the *erased* frames;
- a passive vector per concept — the C2 free-table row of the same term (`passive.kind: c2`, the primary), a host
  hidden state (`features`), or, as a positive control only, the trained C5 static store of the full frame (`c5full`);
- held-out observations per concept for the acceptance test — host hidden states at the term's occurrences in held-out
  (evaluation-split) documents (`evidence.kind: occurrences`, written by `extract`), never the passive vector itself.

Both are mapped into the store's space by cross-fitted ridge decoders (LRE-style; targets = the static stores of the
erased frames of other seen concepts, so no erased edge and no concept's own target enters its decoder). The decoded
passive vector is decomposed over the dictionary (`learn.decompose`, warm-started with the erased frame); rules mined on
the erased graph propose closure edges; every proposal passes the M3 test (held-out utility against same-type control
heads, one-sided Welch t, Holm over the run's proposals). Null worlds from the same data: `relabel` and `swap` decoys
under the identical test, a pipeline-level `permute` null (the proposer reads a same-type term's passive vector), and,
in the synthetic world, a `planted` null (observations generated without the erased edges). Baselines on one shared
candidate pool (each erased edge + 2 distractors): AMIE closure, TransE / RotatE / ComplEx (`kg_baselines`, the WP-PQ2
code), a no-learning prior readout (type-conditional filler frequency), the matched-filter `correlate` readout and a
per-relation LRE readout of the passive vector.

**Placement** (`placement`): the TK-H3L sets (`experiments/toolkit-learn/items/`): rank candidate parents of new terms
from their decoded passive vectors (one-edge decomposition = unbind + typed cleanup) against KGE (ridge into the KGE
entity space) and a text-only cosine baseline; MRR / Hits@k.

Outputs (`OUTPUT/`): `summary.json`, `proposals.jsonl.gz` (every proposal and decoy with its test record), `pool.npz`
(the shared pool: labels and every method's scores; concepts for the bootstrap), `report.md`, `resolved_config.yaml`,
`manifest.json`. `report` pools runs (concepts × seeds bootstrap, Holm) into the pre-registered endpoints.

    python -m vsa_embed.experiments.e10_learn synthetic --config experiments/e10-self-semantics/e10-learn/configs/synthetic.yaml --output DIR
    python -m vsa_embed.experiments.e10_learn extract --run RUN --ontology-run C5RUN --output RUN/learn-features [--device cuda]
    python -m vsa_embed.experiments.e10_learn erased --config CFG --output DIR [--set key=value ...]
    python -m vsa_embed.experiments.e10_learn placement --config CFG --output DIR
    python -m vsa_embed.experiments.e10_learn report --runs DIR [DIR ...] --output DIR
    python -m vsa_embed.experiments.e10_learn store-report --runs DIR [DIR ...] --reference-runs DIR [DIR ...] --output DIR
    python -m vsa_embed.experiments.e10_learn queue [--rf-store] [--dry-run]

**Fixed-operator store** (amendment 2, decision 64; secondary S9): the T4 / T5 erased runs repeated with C5rf's store (E9's
fixed random unitary operator, `random_fixed:unitary_hrr`, its atomics trained around it) — `store=` the C5rf run, every
other input identical — and `store-report` pairs them with the learned store's runs (same pool, row by row): learned −
fixed in decompose's recall at precision 0.8, Holm over the co-primary arms (`rf_store_plan`, `queue --rf-store`).
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Hashable, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from .. import learn as L

SCHEMA = "e10-learn/1"
ROOT = Path("experiments/e10-self-semantics/e10-learn")
E9_RUNS = Path("experiments/e9-retrofit/runs")
RESULT_FILES = ("summary.json", "proposals.jsonl.gz", "pool.npz", "per_concept.json", "report.md", "resolved_config.yaml",
                "manifest.json")
POOL_METHODS = ("decompose", "correlate", "prior", "lre", "amie", "transe", "rotate", "complex")
KGE_METHODS = ("transe", "rotate", "complex")

DEFAULTS: dict[str, Any] = {
    "experiment": "e10-learn",
    "label": None,                         # e.g. SMOKE / PILOT (recorded in every output)
    "seed": 0,                             # erasure / probe / null seed (track-level: identical for every checkpoint seed)
    "num_threads": 4,
    "erase": {"fraction": 0.2, "min_distinct_fillers": 10, "keep": 1, "probe_cap": 2000, "min_frequency": 10,
              "min_degree": 2, "probe_requires_evidence": True},
    "store": None,                         # C5 run folder (erased)
    "passive": {"kind": "c2", "run": None, "features": None},
    "evidence": {"kind": "occurrences", "features": None, "max_observations": 16, "layer": "middle"},
    "decoder": {"folds": 5},
    "sources": ["decompose", "closure"],
    "decompose": {"method": "omp", "max_new": 2, "threshold": 0.1, "min_relative": 0.0},
    "rules": {"min_pca": 0.8, "min_support": 2, "min_head_coverage": 0.01, "chains": True, "limit": 200000, "max_rules": 50,
              "concept_valued": 0.5},
    "test": {"alpha": 0.05, "correction": "holm+decoy", "control": "head", "controls": 8, "min_observations": 4,
             "decoy_statistic": "utility",
             "variants": ["holm", "bh", "knockoff", "decoy"]},
    "types": {"min_members": 5},
    "nulls": ["relabel", "swap", "permute"],
    "pool": {"distractors": 2},
    "baselines": ["prior", "correlate", "lre", "amie", "transe", "rotate", "complex"],
    "kge": {"dimension": 64, "epochs": 100, "negatives": 32, "batch_size": 4096, "lr": 0.01},
    "amie": {"min_pca": 0.1, "min_support": 2, "min_head_coverage": 0.01},
    "precision_target": 0.8,
    "placement": {"items": None, "files": None, "store": None, "fit_features": None, "features": None, "relation_map": {},
                  "layer": "middle", "methods": ["unbind", "correlate"], "kge": ["transe", "rotate", "complex"],
                  "groups": {"primary": {"split": "test"}, "dev": {"split": "dev"}}},
    "synthetic": {"seeds": [101, 202, 303], "dimension": 64, "observation_noise": 0.6, "dictionary": "prior",
                  "proposal_observations": [0, 1, 2, 3], "test_observations": [4, 5, 6, 7, 8, 9, 10, 11], "planted_null": True},
}


# ---------------------------------------------------------------- configuration and output


def merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def apply_sets(config: dict[str, Any], sets: Sequence[str]) -> dict[str, Any]:
    """`key.sub=value` overrides (YAML-parsed values)."""
    for item in sets:
        key, _, value = item.partition("=")
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def load_config(path: Path | None, sets: Sequence[str] = ()) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    return apply_sets(merge(DEFAULTS, raw or {}), sets)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(v) for v in (sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return json_ready(value.tolist())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(json_ready(value), indent=2) + "\n")


def start(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    from ..provenance import prepare_output_dir
    git = prepare_output_dir(Path(output))
    (Path(output) / "resolved_config.yaml").write_text(yaml.safe_dump(json_ready(config), sort_keys=False))
    return git


def finish(output: Path, config: dict[str, Any], git: dict[str, Any], **extra: Any) -> None:
    from ..provenance import write_run_metadata
    write_run_metadata(Path(output), json_ready(config), git_at_start=git, device="cpu", **json_ready(extra))


def write_records(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with gzip.open(path, "wt") as handle:
        for r in records:
            handle.write(json.dumps(json_ready(dict(r))) + "\n"); n += 1
    return n


def log(message: str) -> None:
    print(f"[e10-learn {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- inputs of one erased-edge-recovery run


@dataclass
class ErasureInputs:
    """What `run_erasure` reads. `frames` are the full curated frames (gold) of every concept; `seen` the concepts that
    may be erased, decoded and tested; `passive` / `observations` are in their own spaces (decoded here); `dictionary`
    builds the store's dictionary with typed candidates from the learner's (erased) frames."""

    name: str
    frames: dict[int, list[tuple[int, int]]]
    seen: list[int]
    relation_names: list[str]
    dictionary: Callable[[Sequence[Sequence[tuple[int, int]]]], L.Dictionary]
    passive: dict[int, Tensor]
    observations: dict[int, Tensor]
    atom_concept: Sequence[int]
    meta: dict[str, Any] = field(default_factory=dict)
    planted: Callable[[set[L.Edge]], tuple[dict[int, Tensor], dict[int, Tensor]]] | None = None


def content_relations(frames: Mapping[int, Sequence[tuple[int, int]]], concepts: Iterable[int], relation_count: int,
                      min_distinct: int) -> list[int]:
    """Relations with at least `min_distinct` distinct fillers in the concepts' frames (excludes value-like relations
    such as charge, branch, record class)."""
    fillers: dict[int, set[int]] = defaultdict(set)
    for c in concepts:
        for r, a in frames.get(int(c), ()):
            fillers[int(r)].add(int(a))
    return [r for r in range(relation_count) if len(fillers[r]) >= min_distinct]


def select_probes(erased: set[L.Edge], cap: int, seed: int, eligible: Collection[int] | None = None) -> list[int]:
    """Concepts with an erased edge (and, when given, among `eligible`: a passive vector and enough held-out evidence),
    a seeded sample of at most `cap`. The evidence's occurrences are fixed by the corpus, so the sample is the same for
    every checkpoint seed of a track."""
    concepts = sorted({e[0] for e in erased if eligible is None or e[0] in eligible})
    if cap and len(concepts) > cap:
        rng = np.random.default_rng(seed + 7)
        concepts = sorted(rng.choice(concepts, cap, replace=False).tolist())
    return [int(c) for c in concepts]


def decode_passive(passive: Mapping[int, Tensor], targets: Mapping[int, Tensor], *, folds: int, seed: int
                   ) -> tuple[dict[int, Tensor], dict[str, Any]]:
    """Out-of-fold decoded passive vectors of the concepts with both a passive vector and a target."""
    rows = sorted(c for c in passive if c in targets)
    x = torch.stack([torch.as_tensor(passive[c]).float() for c in rows])
    y = torch.stack([targets[c] for c in rows])
    oof, _, info = L.crossfit_decoder(x, y, folds=folds, seed=seed)
    return {c: oof[i] for i, c in enumerate(rows)}, info


def decode_observations(observations: Mapping[int, Tensor], targets: Mapping[int, Tensor], *, folds: int, seed: int
                        ) -> tuple[dict[int, Tensor], dict[str, Any]]:
    """Per concept its held-out observations mapped by the fold decoder that did not see the concept (fit on per-concept
    observation means → targets of the other folds)."""
    rows = sorted(c for c in observations if c in targets and torch.as_tensor(observations[c]).numel())
    x = torch.stack([torch.as_tensor(observations[c]).float().reshape(-1, torch.as_tensor(observations[c]).shape[-1]).mean(0)
                     for c in rows])
    y = torch.stack([targets[c] for c in rows])
    assignment = np.random.default_rng(seed + 1).permutation(len(rows)) % folds
    out: dict[int, Tensor] = {}
    alphas = []
    for k in range(folds):
        test = assignment == k
        if not test.any():
            continue
        predict, alpha = L.ridge_fit(x[torch.as_tensor(~test)], y[torch.as_tensor(~test)])
        alphas.append(alpha)
        for i in np.flatnonzero(test).tolist():
            obs = torch.as_tensor(observations[rows[i]]).float()
            out[rows[i]] = predict(obs.reshape(-1, obs.shape[-1]))
    mean_cos = float(np.mean([float(F.cosine_similarity(v.mean(0), targets[c], dim=0)) for c, v in out.items()])) if out else float("nan")
    return out, {"rows": len(rows), "fold_alphas": alphas, "mean_cosine_to_target": mean_cos,
                 "observations": int(sum(v.shape[0] for v in out.values()))}


# ---------------------------------------------------------------- pool and baselines


def build_pool(erased: set[L.Edge], probes: Sequence[int], gold: Mapping[int, Sequence[tuple[int, int]]],
               dictionary: L.Dictionary, *, distractors: int, seed: int) -> list[tuple[int, int, int, bool]]:
    """Each erased edge of a probe concept + `distractors` false candidates (E10.0's rule: the same relation with a
    random typed filler, and the same filler under another relation that admits it; neither a curated edge of the
    concept). Rows: (concept, relation, filler, label)."""
    rng = np.random.default_rng(seed + 11)
    wanted = set(int(c) for c in probes)
    rows: list[tuple[int, int, int, bool]] = []
    seen_rows: set[L.Edge] = set()
    gold_sets = {int(c): {(int(r), int(a)) for r, a in f} for c, f in gold.items()}
    for c, r, a in sorted(erased):
        if c not in wanted:
            continue
        rows.append((c, r, a, True)); seen_rows.add((c, r, a))
        made = 0
        for attempt in range(40):
            if made >= distractors:
                break
            if made % 2 == 0 or attempt > 20:
                options = dictionary.candidates[r].nonzero().flatten()
                b, s = int(options[int(rng.integers(options.numel()))]), r
            else:
                options = [x for x in dictionary.candidates[:, a].nonzero().flatten().tolist() if x != r]
                if not options:
                    continue
                s, b = int(options[int(rng.integers(len(options)))]), a
            if (s, b) in gold_sets.get(c, set()) or (c, s, b) in seen_rows:
                continue
            rows.append((c, s, b, False)); seen_rows.add((c, s, b)); made += 1
    return rows


def prior_scores(rows: Sequence[tuple[int, int, int, bool]], frames: Mapping[int, Sequence[tuple[int, int]]],
                 seen: Sequence[int], types: Mapping[int, Hashable]) -> np.ndarray:
    """No-learning prior readout: the share of same-type seen concepts whose (erased) frame has the candidate edge."""
    count: Counter = Counter()
    size: Counter = Counter()
    for c in seen:
        t = types.get(int(c), "other")
        size[t] += 1
        for r, a in set(frames.get(int(c), ())):
            count[(t, int(r), int(a))] += 1
    return np.asarray([count[(types.get(c, "other"), r, a)] / max(1, size[types.get(c, "other")]) for c, r, a, _ in rows])


def lre_scores(rows: Sequence[tuple[int, int, int, bool]], passive: Mapping[int, Tensor],
               frames: Mapping[int, Sequence[tuple[int, int]]], seen: Sequence[int], *, folds: int, seed: int) -> np.ndarray:
    """Probe-B-style direct readout: per relation a ridge map from the passive vector to the multi-hot set of its
    fillers in the (erased) frames, fitted on the seen concepts, out-of-fold for every scored concept; only the pool's
    filler columns are fitted (ridge outputs are independent)."""
    fit_rows = [int(c) for c in seen if int(c) in passive]
    index = {c: i for i, c in enumerate(fit_rows)}
    x = torch.stack([torch.as_tensor(passive[c]).float() for c in fit_rows])
    assignment = np.random.default_rng(seed + 3).permutation(len(fit_rows)) % folds
    out = np.zeros(len(rows))
    by_relation: dict[int, list[int]] = defaultdict(list)
    for i, (c, r, a, _) in enumerate(rows):
        by_relation[r].append(i)
    frame_sets = {c: set(frames.get(c, ())) for c in fit_rows}
    for r, items in by_relation.items():
        columns = sorted({rows[i][2] for i in items})
        col = {a: j for j, a in enumerate(columns)}
        y = torch.tensor([[1.0 if (r, a) in frame_sets[c] else 0.0 for a in columns] for c in fit_rows])
        scores = torch.zeros(len(fit_rows), len(columns))
        for k in range(folds):
            test = torch.as_tensor(assignment == k)
            if not bool(test.any()):
                continue
            predict, _ = L.ridge_fit(x[~test], y[~test])
            scores[test] = predict(x[test])
        for i in items:
            c, _, a, _ = rows[i]
            out[i] = float(scores[index[c], col[a]]) if c in index else 0.0
    return out


def kge_graph(frames: Mapping[int, Sequence[tuple[int, int]]], concepts: Iterable[int], graph: L.RuleGraph
              ) -> tuple[Tensor, dict[Hashable, int]]:
    nodes: dict[Hashable, int] = {}
    triples = []
    for c in sorted(int(x) for x in concepts):
        h = nodes.setdefault(graph.concept_node(c), len(nodes))
        for r, a in frames.get(c, ()):
            t = nodes.setdefault(graph.atom_node(int(a)), len(nodes))
            triples.append((h, int(r), t))
    return torch.tensor(triples, dtype=torch.long).reshape(-1, 3), nodes


def kge_scores(rows: Sequence[tuple[int, int, int, bool]], frames: Mapping[int, Sequence[tuple[int, int]]],
               seen: Sequence[int], graph: L.RuleGraph, relation_count: int, settings: Mapping[str, Any], *, seed: int
               ) -> dict[str, np.ndarray]:
    """TransE / RotatE / ComplEx (`kg_baselines`, the WP-PQ2 code) trained on the seen concepts' erased graph; pool rows
    whose head or tail never occurs in the graph score the minimum."""
    from ..kg_baselines import kge_scores as score_fn, train_kge
    triples, nodes = kge_graph(frames, seen, graph)
    for c, r, a, _ in rows:                                    # pool nodes absent from the graph get (untrained) ids
        nodes.setdefault(graph.concept_node(c), len(nodes)); nodes.setdefault(graph.atom_node(a), len(nodes))
    query = torch.tensor([(nodes[graph.concept_node(c)], r, nodes[graph.atom_node(a)]) for c, r, a, _ in rows],
                         dtype=torch.long).reshape(-1, 3)
    out = {}
    for kind in KGE_METHODS:
        started = time.monotonic()
        model = train_kge(triples, len(nodes), relation_count, kind=kind, dimension=int(settings["dimension"]),
                          epochs=int(settings["epochs"]), negatives=int(settings["negatives"]), lr=float(settings["lr"]),
                          seed=seed, batch_size=settings.get("batch_size"))
        out[kind] = score_fn(model, query).numpy().astype(np.float64)
        log(f"  {kind}: {triples.shape[0]} triples, {len(nodes)} nodes, {time.monotonic() - started:.1f} s")
    return out


def amie_relations(frames: Mapping[int, Sequence[tuple[int, int]]], concepts: Iterable[int], graph: L.RuleGraph,
                   share: float) -> dict[str, set]:
    """The observed graph for rule mining, restricted to concept-valued relations (≥ `share` of the edges' fillers name a
    concept: rules can only chain through concept nodes)."""
    relations = L.graph_relations({int(c): frames.get(int(c), ()) for c in concepts}, graph)
    keep = {}
    for name, pairs in relations.items():
        if pairs and sum(1 for _, t in pairs if t in graph.node_concept) / len(pairs) >= share:
            keep[name] = pairs
    return keep


def amie_scores(rows: Sequence[tuple[int, int, int, bool]], frames: Mapping[int, Sequence[tuple[int, int]]],
                seen: Sequence[int], graph: L.RuleGraph, settings: Mapping[str, Any], rule_settings: Mapping[str, Any]
                ) -> tuple[np.ndarray, int]:
    """AMIE closure score of each pool row: the best PCA confidence of a mined rule (AMIE defaults) predicting it."""
    from ..kg_baselines import closure_scores, mine_rules
    relations = amie_relations(frames, seen, graph, float(rule_settings["concept_valued"]))
    if not relations:
        return np.zeros(len(rows)), 0
    rules = mine_rules(relations, chains=bool(rule_settings["chains"]), min_support=int(settings["min_support"]),
                       min_head_coverage=float(settings["min_head_coverage"]), min_pca=float(settings["min_pca"]))
    table = closure_scores(rules, relations, limit=int(rule_settings["limit"]))
    names = graph.relation_names
    return np.asarray([table.get((names[r], graph.concept_node(c), graph.atom_node(a)), 0.0) for c, r, a, _ in rows]), len(rules)


# ---------------------------------------------------------------- one run


RULES = ("holm", "bh", "knockoff", "decoy", "holm+decoy", "none")


def _utility(dictionary: L.Dictionary, frames: Mapping[int, Sequence[tuple[int, int]]], observations: Mapping[int, Tensor],
             types: Mapping[int, Hashable], settings: Mapping[str, Any], *, seed: int) -> L.AcceptanceTest:
    """The held-out statistics of one world (the decision rule is applied afterwards, `apply_rule`)."""
    utility = L.VectorUtilityTest(dictionary, frames, observations, control=settings["control"],
                                  controls=int(settings["controls"]), types=types, seed=seed)
    return L.AcceptanceTest(utility, alpha=float(settings["alpha"]), correction="holm",
                            min_observations=int(settings["min_observations"]))


def half_of(concept: int, seed: int) -> int:
    """The calibration half of a concept (decoy thresholds are cross-fitted: a concept's threshold comes from the other
    half's proposals and decoys)."""
    return L._stable(seed, "half", int(concept)) % 2


def origin(record: Mapping[str, Any]) -> str:
    return str(record.get("origin", record["source"]))


def decoy_thresholds(real: Sequence[Mapping[str, Any]], decoys: Sequence[Mapping[str, Any]], alpha: float, seed: int, *,
                     statistic: str = "utility") -> dict[tuple[str, int], float]:
    """Per (source, half) the target–decoy threshold on `statistic` computed from the *other* half's real proposals and
    complete-null decoys of the same source (`learn.decoy_threshold`)."""
    out = {}
    for source in sorted({origin(r) for r in real} | {origin(r) for r in decoys}):
        for h in (0, 1):
            t = [L.stat_value(r, statistic) for r in real if r["testable"] and origin(r) == source and half_of(r["concept"], seed) != h]
            d = [L.stat_value(r, statistic) for r in decoys if r["testable"] and origin(r) == source and half_of(r["concept"], seed) != h]
            out[(source, h)] = L.decoy_threshold(t, d, alpha)
    return out


def apply_rule(records: Sequence[Mapping[str, Any]], rule: str, alpha: float, thresholds: Mapping[tuple[str, int], float],
               seed: int, knockoff_t: Sequence[float] | None = None, statistic: str = "utility") -> list[dict[str, Any]]:
    """Copies of `statistics` records decided by `rule` (decoy rules: the cross-fitted per-(source, half) threshold)."""
    copies = [dict(r) for r in records]
    threshold = (lambda r: thresholds.get((origin(r), half_of(r["concept"], seed)), math.inf)) if "decoy" in rule else None
    return L.decide(copies, rule, alpha, threshold=threshold, knockoff_t=knockoff_t, statistic=statistic)


def decomposition_proposals(decoded: Mapping[int, Tensor], probes: Sequence[int], frames: Mapping[int, Sequence[tuple[int, int]]],
                            dictionary: L.Dictionary, settings: L.DecomposeSettings
                            ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, float]]]]:
    have = [c for c in probes if c in decoded]
    if not have:
        return [], {}
    targets = torch.stack([decoded[c] for c in have])
    result = L.decompose(targets, [list(frames.get(c, ())) for c in have], dictionary, settings)
    proposals = [L.make_proposal(c, e["relation"], e["filler"], e["coefficient"], "decompose",
                                 meta={"correlation": e["correlation"], "order": e["order"], "relative": e.get("relative")})
                 for c, edges in zip(have, result) for e in edges]
    return proposals, dict(zip(have, result))


def world_proposals(decoded: Mapping[int, Tensor], probes: Sequence[int], frames: Mapping[int, Sequence[tuple[int, int]]],
                    dictionary: L.Dictionary, settings: L.DecomposeSettings, *, sources: Sequence[str], graph: L.RuleGraph,
                    seen: Sequence[int], content: Sequence[int], rules: Mapping[str, Any]
                    ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, float]]], list[dict[str, Any]]]:
    """The proposals of one world (decompose + closure, `learn.propose`'s order and de-duplication), never an edge of the
    world's frames; returns (proposals, the decomposition's selections, adopted rules)."""
    proposals, selected = decomposition_proposals(decoded, probes, frames, dictionary, settings) if "decompose" in sources else ([], {})
    rule_records: list[dict[str, Any]] = []
    if "closure" in sources:
        rule_frames = {c: frames[c] for c in seen}
        keep_names = set(amie_relations(rule_frames, seen, graph, float(rules["concept_valued"])))
        sub_frames = {c: [(r, a) for r, a in f if graph.relation_names[r] in keep_names] for c, f in rule_frames.items()}
        closure, rule_records = L.rule_closure(sub_frames, graph, concepts=probes, candidates=dictionary.candidates,
                                               settings=L.RuleSettings(**{k: v for k, v in rules.items() if k != "concept_valued"}))
        proposals = proposals + [p for p in closure if p["relation"] in set(content)]
    present = {(int(c), int(r), int(a)) for c in seen for r, a in frames[c]}
    return [p for p in L.dedupe(proposals) if L.edge_key(p) not in present], selected, rule_records


def decompose_pool_scores(rows: Sequence[tuple[int, int, int, bool]], decoded: Mapping[int, Tensor],
                          frames: Mapping[int, Sequence[tuple[int, int]]], selected: Mapping[int, list[dict[str, float]]],
                          dictionary: L.Dictionary) -> tuple[np.ndarray, np.ndarray]:
    """Pool scores of the decomposition (selected: 1 + c/(1+c) for coefficient c; otherwise the correlation with the
    final residual, < 1) and of the matched filter (`correlate`: correlation with the residual of the frame alone)."""
    by_concept: dict[int, list[int]] = defaultdict(list)
    for i, (c, _, _, _) in enumerate(rows):
        by_concept[c].append(i)
    dec = np.full(len(rows), -1.0)
    cor = np.full(len(rows), -1.0)
    for c, items in by_concept.items():
        if c not in decoded:
            continue
        y = F.normalize(decoded[c].float(), dim=-1)
        frame = list(frames.get(c, ()))
        chosen = selected.get(c, [])
        cand = [(rows[i][1], rows[i][2]) for i in items]
        units = F.normalize(dictionary.bound([r for r, _ in cand], [a for _, a in cand]), dim=-1)
        _, base_residual = L._fit(L._frame_design(dictionary, frame), y)
        extra = [(e["relation"], e["filler"]) for e in chosen]
        design = torch.cat([L._frame_design(dictionary, frame), dictionary.bound([r for r, _ in extra], [a for _, a in extra])])
        _, final_residual = L._fit(design, y)
        coef = {(int(e["relation"]), int(e["filler"])): float(e["coefficient"]) for e in chosen}
        corr_final = (units @ final_residual).tolist()
        corr_base = (units @ base_residual).tolist()
        for j, i in enumerate(items):
            key = (rows[i][1], rows[i][2])
            dec[i] = 1.0 + coef[key] / (1.0 + coef[key]) if key in coef else corr_final[j]
            cor[i] = corr_base[j]
    return dec, cor


def summarize_records(records: Sequence[Mapping[str, Any]], gold: set[L.Edge]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sources = sorted({r["source"] for r in records})
    for source in ["all", *sources]:
        chosen = [r for r in records if source == "all" or r["source"] == source]
        accepted = [r for r in chosen if r["accepted"]]
        m = L.edge_metrics(accepted, gold, proposals=chosen)
        m["testable"] = sum(bool(r.get("testable")) for r in chosen)
        out[source] = m
    return out


def run_erasure(inputs: ErasureInputs, config: Mapping[str, Any]) -> dict[str, Any]:
    """One erased-edge-recovery run (module docstring); returns the summary, the records and the pool."""
    timings: dict[str, float] = {}
    clock = time.monotonic()

    def tick(name: str) -> None:
        nonlocal clock
        timings[name] = round(time.monotonic() - clock, 2)
        clock = time.monotonic()

    seed = int(config["seed"])
    erase = config["erase"]
    test_settings = config["test"]
    alpha = float(test_settings["alpha"])
    primary = str(test_settings["correction"])
    rules_to_run = list(dict.fromkeys([primary, *test_settings.get("variants", [])]))
    relation_count = len(inputs.relation_names)
    seen = sorted(int(c) for c in inputs.seen)
    content = content_relations(inputs.frames, seen, relation_count, int(erase["min_distinct_fillers"]))
    frames, erased = L.erase_edges(inputs.frames, seen, fraction=float(erase["fraction"]), relations=set(content), seed=seed,
                                   keep=int(erase["keep"]))
    eligible = None
    if erase.get("probe_requires_evidence", True) and inputs.observations:
        need = int(test_settings["min_observations"])
        eligible = {c for c, v in inputs.observations.items() if torch.as_tensor(v).reshape(-1, torch.as_tensor(v).shape[-1]).shape[0] >= need
                    and c in inputs.passive}
    probes = select_probes(erased, int(erase["probe_cap"]), seed, eligible)
    gold = {e for e in erased if e[0] in set(probes)}
    learner_frames = {c: frames[c] for c in seen}
    dictionary = inputs.dictionary([frames[c] for c in sorted(frames)])
    types = L.concept_types(learner_frames, min_members=int(config["types"]["min_members"]))
    graph = L.RuleGraph.from_atom_concepts(inputs.relation_names, inputs.atom_concept)
    covered = sum(bool(dictionary.candidates[r, a]) for _, r, a in gold)
    full_gold = {(int(c), int(r), int(a)) for c, f in inputs.frames.items() for r, a in f}
    log(f"{inputs.name}: {len(seen)} seen, content relations {[inputs.relation_names[r] for r in content]}, "
        f"{len(erased)} erased ({len(gold)} on {len(probes)} probes; typed coverage {covered}/{len(gold)})")
    folds = int(config["decoder"]["folds"])
    settings = L.DecomposeSettings(**{**config["decompose"], "relations": tuple(content)})
    world_args = dict(sources=config["sources"], graph=graph, seen=seen, content=content, rules=config["rules"])

    def decoded_world(world_frames: Mapping[int, Sequence[tuple[int, int]]], world_dictionary: L.Dictionary,
                      passive: Mapping[int, Tensor], observations: Mapping[int, Tensor]):
        stores = world_dictionary.stores([world_frames[c] for c in seen])
        targets = {c: F.normalize(stores[i], dim=-1) for i, c in enumerate(seen)}
        dec, p_info = decode_passive({c: v for c, v in passive.items() if c in targets}, targets, folds=folds, seed=seed)
        obs, o_info = decode_observations({c: v for c, v in observations.items() if c in targets}, targets, folds=folds, seed=seed)
        return dec, obs, p_info, o_info

    tick("setup")
    decoded, observations, passive_info, evidence_info = decoded_world(learner_frames, dictionary, inputs.passive, inputs.observations)
    log(f"  decoders: passive oof cosine {passive_info['oof_cosine']:.3f} (R² {passive_info['oof_r2']:.3f}); "
        f"evidence {evidence_info['rows']} concepts, {evidence_info['observations']} observations, "
        f"mean cosine {evidence_info['mean_cosine_to_target']:.3f}")
    tick("decoders")
    proposals, selected, rule_records = world_proposals(decoded, probes, learner_frames, dictionary, settings, **world_args)
    log(f"  proposals: {dict(Counter(p['source'] for p in proposals))}")
    tick("propose")
    real = L.statistics(proposals, _utility(dictionary, learner_frames, observations, types, test_settings, seed=seed))
    tick("test")
    # the complete null world (nothing erased: every proposal beyond the curated frames counts as false) supplies the
    # decoys of the target–decoy rules; it is rebuilt from the same data with the full frames
    complete_frames = {c: list(inputs.frames[c]) for c in seen}
    complete_dictionary = inputs.dictionary([inputs.frames[c] for c in sorted(inputs.frames)])
    complete_types = L.concept_types(complete_frames, min_members=int(config["types"]["min_members"]))
    dec_c, obs_c, _, _ = decoded_world(complete_frames, complete_dictionary, inputs.passive, inputs.observations)
    complete_props, _, _ = world_proposals(dec_c, probes, complete_frames, complete_dictionary, settings, **world_args)
    complete = L.statistics(complete_props, _utility(complete_dictionary, complete_frames, obs_c, complete_types, test_settings,
                                                     seed=seed))
    statistic = str(test_settings.get("decoy_statistic", "utility"))
    thresholds = decoy_thresholds(real, complete, alpha, seed, statistic=statistic)
    tick("complete-null")
    knockoff_t = None
    if "knockoff" in rules_to_run:
        # one matched knockoff per proposal: its swap decoy, else its relabel decoy (decoys are seeded per edge)
        made: dict[str, dict[L.Edge, dict[str, Any]]] = {}
        for kind in ("swap", "relabel"):
            made[kind] = {tuple(d["null_of"]): d for d in L.null_proposals(proposals, kind, dictionary=dictionary,
                                                                           frames=learner_frames, types=types,
                                                                           exclude=full_gold, seed=seed)}
        knock = [made["swap"].get(L.edge_key(p)) or made["relabel"].get(L.edge_key(p)) for p in proposals]
        have = [k for k in knock if k is not None]
        k_stats = iter(L.statistics(have, _utility(dictionary, learner_frames, observations, types, test_settings, seed=seed)))
        knockoff_t = [next(k_stats)["t"] if k is not None else math.inf for k in knock]   # no decoy: never accepted
    decided = {rule: apply_rule(real, rule, alpha, thresholds, seed, knockoff_t=knockoff_t, statistic=statistic)
               for rule in rules_to_run}
    tick("decide")
    # null worlds, each judged by every rule at the real run's thresholds
    null_stats: dict[str, list[dict[str, Any]]] = {"complete": [dict(r, origin=r["source"]) for r in complete]}
    for kind in config["nulls"]:
        if kind in L.NULL_KINDS:
            decoys = []
            for source in sorted({p["source"] for p in proposals}):
                decoys += [dict(d, origin=source) for d in L.null_proposals([p for p in proposals if p["source"] == source], kind,
                                                                            dictionary=dictionary, frames=learner_frames,
                                                                            types=types, exclude=full_gold, seed=seed)]
        elif kind == "permute":
            mapping = L.permute_within_type([c for c in probes if c in decoded], types, seed=seed)
            fake = {c: decoded[mapping[c]] for c in mapping if mapping[c] != c}
            decoys, _ = decomposition_proposals(fake, sorted(fake), learner_frames, dictionary, settings)
            decoys = [dict(p, source="null:permute", origin="decompose") for p in decoys if L.edge_key(p) not in full_gold]
        else:
            raise ValueError(f"unknown null {kind!r}")
        null_stats[kind] = L.statistics(decoys, _utility(dictionary, learner_frames, observations, types, test_settings, seed=seed))
    if inputs.planted is not None and config.get("synthetic", {}).get("planted_null"):
        passive_null, observations_null = inputs.planted(erased)
        dec_null, obs_null, _, _ = decoded_world(learner_frames, dictionary, passive_null, observations_null)
        planted, _ = decomposition_proposals(dec_null, probes, learner_frames, dictionary, settings)
        null_stats["planted"] = L.statistics([dict(p, origin="decompose") for p in planted],
                                             _utility(dictionary, learner_frames, obs_null, types, test_settings, seed=seed))
    nulls: dict[str, dict[str, Any]] = {}
    null_records: list[dict[str, Any]] = []
    for kind, stats in null_stats.items():
        nulls[kind] = {}
        for rule in rules_to_run:
            if rule == "knockoff":
                continue
            recs = apply_rule(stats, rule, alpha, thresholds, seed, statistic=statistic)
            nulls[kind][rule] = L.false_acceptance(recs)
            if rule == primary:
                null_records += [dict(r, world=f"null:{kind}", rule=rule) for r in recs]
    tick("nulls")
    result: dict[str, Any] = {
        "name": inputs.name, "seen": len(seen), "probes": len(probes), "erased": len(erased), "gold": len(gold),
        "typed_coverage": covered / len(gold) if gold else float("nan"),
        "content_relations": [inputs.relation_names[r] for r in content], "types": len(set(types.values())),
        "decoders": {"passive": passive_info, "evidence": evidence_info}, "rule": primary,
        "thresholds": {f"{s}/{h}": v for (s, h), v in thresholds.items()},
        "decisions": {rule: summarize_records(recs, gold) for rule, recs in decided.items()},
        "nulls": nulls, "rules": rule_records[:20], "meta": inputs.meta}
    result["primary"] = result["decisions"][primary]
    result["null_far_max"] = {rule: max((v[rule]["rate"] for v in nulls.values() if rule in v and v[rule]["tested"]),
                                        default=float("nan")) for rule in rules_to_run if rule != "knockoff"}
    # the shared candidate pool and the baselines
    rows = build_pool(gold, probes, inputs.frames, dictionary, distractors=int(config["pool"]["distractors"]), seed=seed)
    scores: dict[str, np.ndarray] = {}
    scores["decompose"], scores["correlate"] = decompose_pool_scores(rows, decoded, learner_frames, selected, dictionary)
    baselines = set(config["baselines"])
    if "prior" in baselines:
        scores["prior"] = prior_scores(rows, learner_frames, seen, types)
    if "lre" in baselines:
        scores["lre"] = lre_scores(rows, {c: v for c, v in inputs.passive.items() if c in set(seen)}, learner_frames, seen,
                                   folds=folds, seed=seed)
    tick("pool-readouts")
    if "amie" in baselines:
        scores["amie"], result["amie_rules"] = amie_scores(rows, learner_frames, seen, graph, config["amie"], config["rules"])
        tick("amie")
    if baselines & set(KGE_METHODS):
        kge = kge_scores(rows, learner_frames, seen, graph, relation_count, config["kge"], seed=seed)
        scores.update({k: v for k, v in kge.items() if k in baselines})
        tick("kge")
    labels = np.asarray([r[3] for r in rows], dtype=bool)
    target = float(config["precision_target"])
    result["pool"] = {"rows": len(rows), "positives": int(labels.sum()),
                      "methods": {m: {"auc": L.roc_auc(s, labels), f"recall_at_{target:g}": L.recall_at_precision(s, labels, target)}
                                  for m, s in scores.items()}}
    pool = {"concept": np.asarray([r[0] for r in rows]), "relation": np.asarray([r[1] for r in rows]),
            "filler": np.asarray([r[2] for r in rows]), "label": labels, **{f"score_{m}": s for m, s in scores.items()}}
    counts = {rule: L.per_concept_counts([r for r in recs if r["accepted"] and r["source"] == "decompose"], gold, probes)
              for rule, recs in decided.items()}
    records = [dict(r, world="real", rule=primary, gold=L.edge_key(r) in gold,
                    decisions={rule: bool(decided[rule][i]["accepted"]) for rule in rules_to_run})
               for i, r in enumerate(decided[primary])]
    result["timings"] = timings
    return {"summary": result, "records": records + null_records, "pool": pool, "per_concept": counts}


# ---------------------------------------------------------------- the synthetic world


def synthetic_inputs(seed: int, settings: Mapping[str, Any]) -> ErasureInputs:
    """E10.0's planted ontology world (`synthetic_ontology.make_ontology_world`): seen = training + validation concepts;
    passive vector = the mean of the proposal observations; held-out observations = the test observations; dictionary =
    the noisy priors (cos 0.8 to the teacher; `dictionary: prior`) or the teacher (`truth`). The planted null regenerates
    every observation from the erased frames (same noise model), so nothing erased is in the data."""
    from ..algebra import HRRAlgebra
    from ..synthetic_ontology import make_ontology_world
    world = make_ontology_world(seed=seed, dimension=int(settings["dimension"]),
                                observation_noise=float(settings["observation_noise"]), **dict(settings.get("world") or {}))
    frames: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h, r, a in world.edge_list():
        frames[int(h)].append((int(r), int(a)))
    frames = {c: frames.get(c, []) for c in range(world.concept_count)}
    seen = sorted(set(world.splits["train"].tolist()) | set(world.splits["validation"].tolist()))
    seen = [c for c in seen if len(frames[c]) >= 2]
    if settings["dictionary"] == "truth":
        atomics, roles = world.truth["atomics"], world.truth["roles"]
    else:
        atomics, roles = world.atomic_prior, world.relation_prior
    proposal_obs = list(settings["proposal_observations"])
    test_obs = list(settings["test_observations"])
    passive = {c: world.observations[c, proposal_obs].mean(0) for c in seen}
    observations = {c: world.observations[c, test_obs] for c in seen}
    atom_concept = [a if a < world.concept_count else -1 for a in range(world.atomic_count)]

    def planted(erased: set[L.Edge]) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        gone = {(h, r, a) for h, r, a in erased}
        keep = torch.tensor([(int(h), int(r), int(a)) not in gone for h, r, a in world.edge_list()])
        heads, rels, fills, weights = world.heads[keep], world.relations[keep], world.fillers[keep], world.teacher_weights[keep]
        bound = HRRAlgebra().bind(world.truth["roles"][rels], world.truth["atomics"][fills])
        clean = F.normalize(torch.zeros(world.concept_count, world.target_dimension).index_add(0, heads, weights[:, None] * bound), dim=-1)
        g = torch.Generator().manual_seed(seed + 99991)
        k = world.observations.shape[1]
        noise = torch.randn(world.concept_count, k, world.target_dimension, generator=g) * float(settings["observation_noise"]) \
            / world.target_dimension ** 0.5
        obs = F.normalize(clean[:, None, :] + noise, dim=-1)
        return ({c: obs[c, proposal_obs].mean(0) for c in seen}, {c: obs[c, test_obs] for c in seen})

    return ErasureInputs(f"synthetic-s{seed}", frames, seen, list(world.relation_names),
                         lambda fr: L.Dictionary.from_roles(atomics, roles, L.typed_candidates(fr, len(world.relation_names),
                                                                                                world.atomic_count),
                                                            relation_names=world.relation_names),
                         passive, observations, atom_concept, meta={"world": world.metadata, "dictionary": settings["dictionary"]},
                         planted=planted)


def run_synthetic(config: dict[str, Any], output: Path) -> dict[str, Any]:
    git = start(output, config)
    torch.set_num_threads(int(config["num_threads"]))
    started = time.monotonic()
    summaries, records, per_seed = [], [], {}
    for s in config["synthetic"]["seeds"]:
        inputs = synthetic_inputs(int(s), config["synthetic"])
        cfg = dict(config, seed=int(s))
        out = run_erasure(inputs, cfg)
        summaries.append(out["summary"])
        records += [dict(r, seed=int(s)) for r in out["records"]]
        per_seed[int(s)] = out
        np.savez_compressed(Path(output) / f"pool-s{s}.npz", **out["pool"])
    summary = {"schema": SCHEMA, "mode": "synthetic", "label": config.get("label"), "runs": summaries,
               "pooled": pool_synthetic(summaries), "seconds": round(time.monotonic() - started, 1)}
    write_json(Path(output) / "summary.json", summary)
    write_records(Path(output) / "proposals.jsonl.gz", records)
    (Path(output) / "report.md").write_text(render_run_report(summary))
    finish(output, config, git, schema=SCHEMA, seconds=summary["seconds"])
    return summary


def pool_synthetic(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per rule: mean [95% t over seeds] of precision / recall / F1 of accepted decompose edges, and the pooled null
    false-acceptance rate (accepted / tested summed over seeds) of every null world; pool AUC and recall at 0.8."""
    from ..statistics import mean_confidence_interval

    def mean(values: list[float]) -> dict[str, Any]:
        values = [v for v in values if v is not None and math.isfinite(v)]
        return mean_confidence_interval(values) if values else {"mean": None}

    out: dict[str, Any] = {}
    rules = list(dict.fromkeys(r for s in summaries for r in s["decisions"]))
    for rule in rules:
        block: dict[str, Any] = {}
        for source in ("decompose", "all"):
            for key in ("precision", "recall", "f1", "accepted", "false_acceptance_nongold"):
                block[f"{source}.{key}"] = mean([s["decisions"][rule].get(source, {}).get(key, float("nan")) for s in summaries])
        for kind in dict.fromkeys(k for s in summaries for k in s["nulls"]):
            accepted = sum(s["nulls"][kind][rule]["accepted"] for s in summaries if rule in s["nulls"].get(kind, {}))
            tested = sum(s["nulls"][kind][rule]["tested"] for s in summaries if rule in s["nulls"].get(kind, {}))
            if tested:
                block[f"null.{kind}"] = {"accepted": accepted, "tested": tested, "rate": accepted / tested}
        out[rule] = block
    methods = {m for s in summaries for m in s["pool"]["methods"]}
    out["pool"] = {f"{m}.{key}": mean([s["pool"]["methods"].get(m, {}).get(key, float("nan")) for s in summaries])
                   for m in sorted(methods) for key in ("auc", "recall_at_0.8")}
    return out


# ---------------------------------------------------------------- real checkpoints


def entry_frames(ontology: Mapping[str, Any]) -> dict[int, list[tuple[int, int]]]:
    offsets = np.asarray(ontology["offsets"]); rel = np.asarray(ontology["relations"]); fil = np.asarray(ontology["fillers"])
    return {e: list(zip(rel[offsets[e]:offsets[e + 1]].tolist(), fil[offsets[e]:offsets[e + 1]].tolist()))
            for e in range(offsets.size - 1)}


def seen_entries(ontology: Mapping[str, Any], frames: Mapping[int, Sequence[tuple[int, int]]], *, min_frequency: int,
                 min_degree: int) -> list[int]:
    """Training frequency ≥ `min_frequency`, not held out, at least `min_degree` edges (E9's `seen` stratum)."""
    frequency = np.asarray(ontology.get("train_frequency") or np.zeros(int(ontology["entry_count"])))
    held = {int(e) for e in ontology.get("heldout_entries", ())}
    return [e for e in range(frequency.size) if frequency[e] >= min_frequency and e not in held and len(frames.get(e, ())) >= min_degree]


def atom_concepts(ontology: Mapping[str, Any]) -> np.ndarray:
    """Per atomic the entry of the concept it names, else −1 (`e9_binding_chain.atom_entries`)."""
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


def load_c2_rows(run_dir: Path) -> Tensor:
    """C2's free-table rows (entries × free width), read from `final.pt` without building the model."""
    state = torch.load(Path(run_dir) / "final.pt", weights_only=False, map_location="cpu", mmap=True)
    weight = state["model"].get("channel.table.weight")
    if weight is None:
        raise ValueError(f"{run_dir} has no free table (channel.table.weight)")
    return weight.float().clone()


def load_features(path: Path, *, layer: str, split_seed: int) -> tuple[dict[int, Tensor], dict[int, Tensor], dict[str, Any]]:
    """An `extract` file → (proposal-half means, test-half occurrences) per entry; occurrences are split by document
    (a seeded hash of the document id), so the two halves never share a document."""
    with np.load(path, allow_pickle=False) as data:
        entries = data["entry"].astype(np.int64)
        documents = data["document"].astype(np.int64)
        vectors = data[layer].astype(np.float32)
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    half = np.asarray([L._stable(split_seed, "doc", int(d)) % 2 for d in documents.tolist()])
    proposal: dict[int, Tensor] = {}
    test: dict[int, Tensor] = {}
    for e in np.unique(entries).tolist():
        rows = entries == e
        a, b = rows & (half == 0), rows & (half == 1)
        if a.any():
            proposal[int(e)] = torch.from_numpy(vectors[a].mean(0))
        if b.any():
            test[int(e)] = torch.from_numpy(vectors[b])
    return proposal, test, meta


def real_inputs(config: Mapping[str, Any]) -> ErasureInputs:
    from .e9_binding_chain import load_composer
    store = Path(config["store"])
    composer, run_config, ontology = load_composer(store)
    frames = entry_frames({"offsets": composer.schedule.offsets.cpu().numpy(), "relations": composer.schedule.relations.cpu().numpy(),
                           "fillers": composer.schedule.fillers.cpu().numpy()})
    erase = config["erase"]
    seen = seen_entries(ontology, frames, min_frequency=int(erase["min_frequency"]), min_degree=int(erase["min_degree"]))
    passive_cfg = config["passive"]
    split_seed = int(config["seed"])
    if passive_cfg["kind"] == "c2":
        rows = load_c2_rows(Path(passive_cfg["run"]))
        passive = {c: rows[c] for c in seen}
    elif passive_cfg["kind"] == "c5full":
        stores = L.Dictionary.from_composer(composer).stores([frames[c] for c in seen])
        passive = {c: stores[i] for i, c in enumerate(seen)}
    elif passive_cfg["kind"] == "features":
        passive, _, _ = load_features(Path(passive_cfg["features"]), layer=config["evidence"]["layer"], split_seed=split_seed)
        passive = {c: v for c, v in passive.items() if c in set(seen)}
    else:
        raise ValueError(f"unknown passive kind {passive_cfg['kind']!r}")
    evidence = config["evidence"]
    observations: dict[int, Tensor] = {}
    meta: dict[str, Any] = {"store": str(store), "operator": composer.operator, "passive": dict(passive_cfg)}
    if evidence["kind"] == "occurrences":
        _, observations, feature_meta = load_features(Path(evidence["features"]), layer=evidence["layer"], split_seed=split_seed)
        cap = int(evidence["max_observations"])
        observations = {c: v[:cap] for c, v in observations.items() if c in set(seen)}
        meta["evidence"] = {"features": str(evidence["features"]), **feature_meta}
    elif evidence["kind"] != "none":
        raise ValueError(f"unknown evidence kind {evidence['kind']!r}")
    names = list(ontology["relation_names"])
    return ErasureInputs(store.name, frames, seen, names,
                         lambda fr: L.Dictionary.from_composer(composer, fr, relation_names=names),
                         passive, observations, atom_concepts(ontology).tolist(), meta=meta)


def run_erased(config: dict[str, Any], output: Path) -> dict[str, Any]:
    git = start(output, config)
    torch.set_num_threads(int(config["num_threads"]))
    started = time.monotonic()
    inputs = real_inputs(config)
    out = run_erasure(inputs, config)
    summary = {"schema": SCHEMA, "mode": "erased", "label": config.get("label"), "runs": [out["summary"]],
               "seconds": round(time.monotonic() - started, 1)}
    write_json(Path(output) / "summary.json", summary)
    write_records(Path(output) / "proposals.jsonl.gz", out["records"])
    np.savez_compressed(Path(output) / "pool.npz", **out["pool"])
    write_json(Path(output) / "per_concept.json", {rule: {str(k): v for k, v in table.items()}
                                                   for rule, table in out["per_concept"].items()})
    (Path(output) / "report.md").write_text(render_run_report(summary))
    finish(output, config, git, schema=SCHEMA, seconds=summary["seconds"])
    return summary


# ---------------------------------------------------------------- feature extraction (host hidden states; GPU job)


def default_alias_table(run_dir: Path) -> Path | None:
    """`~/data/vsa-llm/e9/alias-tables/<track>.json` for an E9 run (`e9_track` in its config), when it exists."""
    try:
        track = yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text()).get("e9_track")
    except FileNotFoundError:
        return None
    path = Path("~/data/vsa-llm/e9/alias-tables").expanduser() / f"{track}.json"
    return path if track and path.is_file() else None


@torch.no_grad()
def extract_features(run_dir: Path, entries: Collection, *, window: int = 512, max_per_entry: int = 32, batch: int = 8,
                     splits: Sequence[tuple[str, int | None]] = (("eval", None),), device: str | None = None,
                     alias_table: Path | None = None) -> dict[str, Any]:
    """Hidden states of a run's host (no channel injection: the host's own representation) at the last subtoken of every
    occurrence of `entries`, at layers ⌊L/2⌋ (`middle`) and L (`final`), up to `max_per_entry` occurrences per entry over
    all `splits` together, with each occurrence's document (EOS-delimited; ids made unique across splits) and split.

    `splits` = (data split of the run's config, window cap) in order: an uncapped split is read in non-overlapping
    `window`-token windows in corpus order; a capped one in that many windows spread evenly over the corpus (the
    pre-registered evidence is the evaluation split, then up to 20,000 windows of the training split for entries still
    below the cap). The spans come from each corpus's own linking; `alias_table` only lets the run open (default: the
    track's table, `default_alias_table`)."""
    from ..data.corpus import TokenCorpus
    from .e5_common import open_run
    run = open_run(Path(run_dir), device=device, alias_table=alias_table or default_alias_table(Path(run_dir)))
    model = run.model.eval()
    wanted = {int(e) for e in entries}
    counts: Counter = Counter()
    rows_entry, rows_doc, rows_pos, rows_split, middle, final = [], [], [], [], [], []
    min_sub = int(run.config["data"]["min_subtokens"])
    layers = None
    info = []
    for index, (split, cap) in enumerate(splits):
        corpus = TokenCorpus.open(Path(run.config["data"][split]))
        eos = int(corpus.manifest.get("eos_id", 0))
        eos_positions = np.flatnonzero(np.asarray(corpus.tokens) == eos)
        last = max(1, len(corpus) - window)
        if cap:
            stride = max(window, last // int(cap))
            starts = list(range(0, last, stride))[:int(cap)]
        else:
            starts = list(range(0, last, window))
        scanned = found = 0
        for b in range(0, len(starts), batch):
            if wanted and all(counts[e] >= max_per_entry for e in wanted):
                break
            windows = [corpus.window(s, window, min_subtokens=min_sub) for s in starts[b:b + batch]]
            scanned += len(windows)
            picks = []
            for w, (_, spans) in enumerate(windows):
                for k, e in enumerate(spans["entry"].tolist()):
                    if e in wanted and counts[e] < max_per_entry:
                        counts[e] += 1
                        picks.append((w, int(spans["end"][k]), int(e), int(spans["start"][k]) + starts[b + w]))
            if not picks:
                continue
            ids = torch.from_numpy(np.stack([w[0] for w in windows])).to(run.device)
            with torch.autocast(run.device.type, dtype=torch.bfloat16, enabled=run.device.type == "cuda"):
                out = model.base(inputs_embeds=model.embed(ids, None), output_hidden_states=True)
            states = out.hidden_states
            layers = len(states) - 1
            w_idx = torch.tensor([p[0] for p in picks], device=run.device)
            t_idx = torch.tensor([p[1] for p in picks], device=run.device)
            middle.append(states[layers // 2][w_idx, t_idx].float().cpu().numpy().astype(np.float16))
            final.append(states[-1][w_idx, t_idx].float().cpu().numpy().astype(np.float16))
            rows_entry += [p[2] for p in picks]
            rows_pos += [p[3] for p in picks]
            rows_split += [index] * len(picks)
            documents = np.searchsorted(eos_positions, np.asarray([p[3] for p in picks]), side="right")
            rows_doc += (documents + index * 10 ** 10).tolist()
            found += len(picks)
        info.append({"split": split, "cap": cap, "windows": len(starts), "windows_scanned": scanned, "occurrences": found})
    width = model.model.get_input_embeddings().weight.shape[1]
    return {"entry": np.asarray(rows_entry, dtype=np.int64), "document": np.asarray(rows_doc, dtype=np.int64),
            "position": np.asarray(rows_pos, dtype=np.int64), "split": np.asarray(rows_split, dtype=np.int8),
            "middle": np.concatenate(middle) if middle else np.zeros((0, width), np.float16),
            "final": np.concatenate(final) if final else np.zeros((0, width), np.float16),
            "meta": json.dumps({"run": str(run_dir), "window": window, "max_per_entry": max_per_entry, "layers": layers,
                                "middle_layer": None if layers is None else layers // 2, "splits": info,
                                "entries_wanted": len(wanted), "entries_found": len(counts),
                                "entries_at_cap": sum(1 for v in counts.values() if v >= max_per_entry),
                                "occurrences": len(rows_entry)})}


# ---------------------------------------------------------------- placement (TK-H3L sets)

PLACEMENT_FILES = ("placement-dev.jsonl", "placement-test.jsonl")


def _placement_item(raw: Mapping[str, Any]) -> dict[str, Any]:
    """One TK-H3L placement item (`experiments/toolkit-learn/README.md`, `learn_data.validate_placement`) in the
    evaluator's form; the placement relation is the relation of the gold parents in `gold_relations` (MeSH descriptors,
    ICD, OET and taxonomies: `parent`; MeSH SCRs: `mapped_to`)."""
    meta = dict(raw.get("meta") or {})
    gold = raw.get("gold_parents", raw.get("gold", raw.get("parents")))
    if gold is None:
        raise ValueError(f"placement item {raw.get('id')!r} has no gold parents")
    gold = [str(g) for g in (gold if isinstance(gold, list) else [gold])]
    relation = raw.get("relation")
    if relation is None:
        relation = next((str(r) for r, target in raw.get("gold_relations") or () if str(target) in gold), "parent")
    contexts = raw.get("contexts") or meta.get("contexts")
    return {"id": str(raw["id"]), "set": str(raw.get("set", "")), "record": str(raw.get("record", raw["id"])),
            "term": str(raw.get("name", raw.get("term", raw["id"]))), "aliases": [str(a) for a in raw.get("aliases") or ()],
            "definition": raw.get("definition") or raw.get("text"), "gold": gold, "relation": str(relation),
            "candidates": [str(c) for c in raw["candidates"]] if raw.get("candidates") else None,
            "split": str(meta.get("split", raw.get("split", "test"))), "t7_group": meta.get("t7_group"),
            "kind": meta.get("kind"), "role": meta.get("role"), "primary": meta.get("primary"),
            "contexts": [str(c) if not isinstance(c, Mapping) else str(c.get("text", "")) for c in contexts] if contexts else None}


def load_placement_items(path: Path, files: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Placement items of a TK-H3L set: a JSONL file (optionally gzipped) or an item folder, whose `files` (default
    `placement-dev.jsonl` and `placement-test.jsonl`, the ones present) are read in order (`_placement_item`)."""
    path = Path(path)
    if path.is_dir():
        names = list(files) if files else [f for f in PLACEMENT_FILES if (path / f).exists()]
        if not names and (path / "items.jsonl").exists():
            names = ["items.jsonl"]
        paths = [path / f for f in names]
    else:
        paths = [path]
    items = []
    for p in paths:
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as handle:
            items += [_placement_item(json.loads(line)) for line in handle if line.strip()]
    return items


def placement_eval(items: Sequence[Mapping[str, Any]], term_vectors: Mapping[str, Tensor], dictionary: L.Dictionary,
                   relation_id: Mapping[str, int], node_atom: Mapping[str, int], *, method: str = "unbind",
                   baselines: Mapping[str, Callable[[Mapping[str, Any], list[int]], np.ndarray]] | None = None,
                   ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    """Rank of the best gold parent among each item's candidates — the item's list, or with `candidates: null` the store's
    typed candidates of the mapped relation (the store can only name atomics in its dictionary) — for the store
    (`learn.placement_scores` of the item's decoded passive vector, a one-edge decomposition) and every baseline
    (name → f(item, candidate atomics) → scores). An item whose relation is unmapped, whose gold parent is not among the
    candidates (`coverage`) or that has no vector is not placed (reciprocal rank 0). Returns overall metrics and per-item
    rows (with `split`, `t7_group`, `kind` for grouping)."""
    methods = ["store", *(baselines or {})]
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for item in items:
        row = {k: item.get(k) for k in ("id", "split", "t7_group", "kind", "relation")}
        r = relation_id.get(item["relation"])
        atoms: list[int] = []
        gold_cols: list[int] = []
        if r is None:
            skipped["relation"] += 1
        else:
            atoms = dictionary.candidates[r].nonzero().flatten().tolist() if item["candidates"] is None else \
                [node_atom[c] for c in item["candidates"] if c in node_atom]
            index = {a: i for i, a in enumerate(atoms)}
            gold_cols = sorted({index[node_atom[g]] for g in item["gold"] if g in node_atom and node_atom[g] in index})
            if not gold_cols:
                skipped["gold"] += 1
        row["covered"] = bool(gold_cols)
        row["candidates"] = len(atoms)
        for m in methods:
            rank = float("nan")
            if gold_cols:
                if m == "store":
                    vector = term_vectors.get(item["id"])
                    s = None if vector is None else L.placement_scores(torch.as_tensor(vector)[None], r, atoms, dictionary,
                                                                       method=method)
                else:
                    s = torch.as_tensor(np.asarray(baselines[m](item, atoms), dtype=np.float32))[None]
                if s is not None:
                    rank = float(L.gold_ranks(s, [gold_cols])[0])
            row[f"rank_{m}"] = rank
        rows.append(row)
    return {"items": len(items), "skipped": dict(skipped),
            "coverage": sum(r["covered"] for r in rows) / len(rows) if rows else float("nan"),
            "methods": {m: L.ranking_metrics([r[f"rank_{m}"] for r in rows], ks) for m in methods}, "rows": rows}


def placement_groups(rows: Sequence[Mapping[str, Any]], methods: Sequence[str], groups: Mapping[str, Mapping[str, Any]],
                     ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    """Metrics per named group of rows (`{"name": {field: value or [values]}}`; e.g. the primary T7 group
    `{"split": "test", "t7_group": "eval"}`); a field the rows do not carry (None) does not filter."""
    out = {}
    for name, spec in groups.items():
        def keep(row: Mapping[str, Any]) -> bool:
            for field_, wanted in spec.items():
                value = row.get(field_)
                if value is None:
                    continue
                if value not in (wanted if isinstance(wanted, (list, tuple)) else [wanted]):
                    return False
            return True
        chosen = [r for r in rows if keep(r)]
        out[name] = {"n": len(chosen), "coverage": sum(r["covered"] for r in chosen) / len(chosen) if chosen else float("nan"),
                     "methods": {m: L.ranking_metrics([r[f"rank_{m}"] for r in chosen], ks) for m in methods}}
    return out


# ---------------------------------------------------------------- reports


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_run_report(summary: Mapping[str, Any]) -> str:
    lines = [f"# E10.L {summary['mode']} ({summary.get('label') or 'run'})", ""]
    for run in summary["runs"]:
        lines += [f"## {run['name']}", "",
                  f"Seen {run['seen']}; probes {run['probes']}; erased {run['erased']} (gold on probes {run['gold']}; "
                  f"typed coverage {_fmt(run['typed_coverage'])}); content relations: {', '.join(run['content_relations'])}; "
                  f"{run['types']} concept types.", "",
                  f"Decoders: passive out-of-fold cosine {_fmt(run['decoders']['passive'].get('oof_cosine'))}; evidence "
                  f"{run['decoders']['evidence'].get('rows')} concepts / {run['decoders']['evidence'].get('observations')} observations "
                  f"(mean cosine {_fmt(run['decoders']['evidence'].get('mean_cosine_to_target'))}). Primary rule: `{run['rule']}`; "
                  f"decoy thresholds {({k: round(v, 2) if math.isfinite(v) else None for k, v in run['thresholds'].items()})}.", "",
                  "| rule | source | proposed | testable | accepted | precision | recall | F1 | proposal coverage | false acc. non-gold |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for rule, block in run["decisions"].items():
            for source, m in block.items():
                lines.append(f"| {rule} | {source} | {m.get('proposed')} | {m.get('testable')} | {m['accepted']} | "
                             f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m.get('proposal_coverage'))} | "
                             f"{_fmt(m.get('false_acceptance_nongold'))} |")
        rules = list(dict.fromkeys(r for v in run["nulls"].values() for r in v))
        lines += ["", "Null false-acceptance rate (accepted / tested) [Wilson 95%], each null world judged by the rule at the "
                  "real run's thresholds:", "", "| null world | tested | " + " | ".join(rules) + " |",
                  "|---|---|" + "---|" * len(rules)]
        for kind, by_rule in run["nulls"].items():
            tested = next(iter(by_rule.values()))["tested"] if by_rule else 0
            cells = [f"{_fmt(by_rule[r]['rate'], 4)} ({by_rule[r]['accepted']}) [{_fmt(by_rule[r]['ci_low'], 3)}, "
                     f"{_fmt(by_rule[r]['ci_high'], 3)}]" if r in by_rule else "—" for r in rules]
            lines.append(f"| {kind} | {tested} | " + " | ".join(cells) + " |")
        pool = run["pool"]
        lines += ["", f"Shared pool: {pool['rows']} candidates, {pool['positives']} erased edges.", "",
                  "| method | AUC | recall at precision 0.8 |", "|---|---|---|"]
        for m, v in pool["methods"].items():
            lines.append(f"| {m} | {_fmt(v['auc'])} | {_fmt(v.get('recall_at_0.8'))} |")
        lines += ["", f"Timings (s): {run['timings']}", ""]
    if "pooled" in summary:
        lines += ["## Pooled over seeds", "", "Mean [95% t over seeds] for precision / recall; null rates pooled (accepted / tested).", ""]
        for rule, block in summary["pooled"].items():
            if rule == "pool":
                continue
            d = lambda k: block.get(k, {})  # noqa: E731
            nulls = "; ".join(f"{k[5:]} {_fmt(v['rate'], 4)} ({v['accepted']}/{v['tested']})" for k, v in block.items()
                              if k.startswith("null."))
            lines.append(f"- **{rule}** decompose precision {_fmt(d('decompose.precision').get('mean'))} "
                         f"[{_fmt(d('decompose.precision').get('ci_low'))}, {_fmt(d('decompose.precision').get('ci_high'))}], recall "
                         f"{_fmt(d('decompose.recall').get('mean'))} [{_fmt(d('decompose.recall').get('ci_low'))}, "
                         f"{_fmt(d('decompose.recall').get('ci_high'))}], accepted {_fmt(d('decompose.accepted').get('mean'), 1)}; "
                         f"nulls: {nulls}")
        lines += ["", "Pool (mean over seeds): " + "; ".join(f"{k} {_fmt(v.get('mean'))}" for k, v in summary["pooled"]["pool"].items())]
    return "\n".join(lines) + "\n"


def weighted_recall_at_precision(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, precision: float) -> float:
    """`learn.recall_at_precision` with row weights (bootstrap multiplicities)."""
    order = np.argsort(-scores, kind="stable")
    s, y, w = scores[order], labels[order], weights[order]
    total = float((w * y).sum())
    if total <= 0:
        return float("nan")
    tp = np.cumsum(w * y)
    k = np.cumsum(w)
    last = np.r_[s[1:] != s[:-1], True] & (w > 0)
    ok = last & (k > 0) & (tp / np.maximum(k, 1e-12) >= precision - 1e-12)
    return float(tp[ok].max() / total) if ok.any() else 0.0


def pool_bootstrap(pools: Sequence[Mapping[str, np.ndarray]], method: str, baseline: str, *, precision: float,
                   resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Recall at `precision` of `method` − `baseline` on the shared pools of several seeds (same concepts): the pigeonhole
    bootstrap over concepts × seeds (`statistics.two_way_cluster_bootstrap`'s design: concept and seed multiplicities
    drawn independently; the statistic is recomputed on the weighted pool)."""
    concepts = sorted(set(np.concatenate([np.asarray(p["concept"]) for p in pools]).tolist()))
    index = {c: i for i, c in enumerate(concepts)}
    rng = np.random.default_rng(seed)

    def stat(cw: np.ndarray, sw: np.ndarray) -> float:
        diffs = []
        weights_total = 0.0
        for j, p in enumerate(pools):
            if sw[j] == 0:
                continue
            w = cw[np.asarray([index[int(c)] for c in p["concept"]])] * sw[j]
            a = weighted_recall_at_precision(np.asarray(p[f"score_{method}"]), np.asarray(p["label"], bool), w, precision)
            b = weighted_recall_at_precision(np.asarray(p[f"score_{baseline}"]), np.asarray(p["label"], bool), w, precision)
            diffs.append((a - b) * sw[j]); weights_total += sw[j]
        return float(np.sum(diffs) / weights_total) if weights_total else float("nan")

    point = stat(np.ones(len(concepts)), np.ones(len(pools)))
    draws = []
    for _ in range(resamples):
        cw = rng.multinomial(len(concepts), np.full(len(concepts), 1 / len(concepts))).astype(float)
        sw = rng.multinomial(len(pools), np.full(len(pools), 1 / len(pools))).astype(float)
        draws.append(stat(cw, sw))
    draws = np.asarray([d for d in draws if math.isfinite(d)])
    p = float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (draws.size + 1))) if draws.size else float("nan")
    return {"mean": point, "ci_low": float(np.quantile(draws, 0.025)) if draws.size else None,
            "ci_high": float(np.quantile(draws, 0.975)) if draws.size else None, "p_value": p, "resamples": int(draws.size)}


def precision_bootstrap(per_concept: Sequence[Mapping[str, Sequence[int]]], *, threshold: float = 0.8, resamples: int = 2000,
                        seed: int = 0) -> dict[str, Any]:
    """Pooled precision and recall of accepted edges over seeds (per-concept TP / FP / FN tables), with the pigeonhole
    bootstrap over concepts × seeds, and the one-sided bootstrap p-value of H0 "precision ≤ `threshold`":
    `(1 + #{draws ≤ threshold}) / (1 + #draws)` (1 when nothing is accepted)."""
    concepts = sorted({int(c) for table in per_concept for c in table})
    arr = np.zeros((len(concepts), len(per_concept), 3))
    for j, table in enumerate(per_concept):
        for i, c in enumerate(concepts):
            arr[i, j] = table.get(str(c), table.get(c, (0, 0, 0))) if isinstance(table, Mapping) else (0, 0, 0)
    rng = np.random.default_rng(seed)

    def stat(cw: np.ndarray, sw: np.ndarray) -> tuple[float, float]:
        tot = np.einsum("c,cjk,j->k", cw, arr, sw)
        tp, fp, fn = tot
        return (tp / (tp + fp) if tp + fp else float("nan")), (tp / (tp + fn) if tp + fn else float("nan"))

    precision, recall = stat(np.ones(len(concepts)), np.ones(len(per_concept)))
    draws = [stat(rng.multinomial(len(concepts), np.full(len(concepts), 1 / len(concepts))).astype(float),
                  rng.multinomial(len(per_concept), np.full(len(per_concept), 1 / len(per_concept))).astype(float))
             for _ in range(resamples)] if concepts else []
    pr = np.asarray([d[0] for d in draws if math.isfinite(d[0])])
    rc = np.asarray([d[1] for d in draws if math.isfinite(d[1])])
    totals = arr.sum((0, 1)) if concepts else np.zeros(3)
    p_value = float((1 + int((pr <= threshold).sum())) / (1 + pr.size)) if pr.size and math.isfinite(precision) else 1.0
    return {"precision": precision, "precision_ci": [float(np.quantile(pr, 0.025)), float(np.quantile(pr, 0.975))] if pr.size else None,
            "recall": recall, "recall_ci": [float(np.quantile(rc, 0.025)), float(np.quantile(rc, 0.975))] if rc.size else None,
            "tp": int(totals[0]), "fp": int(totals[1]), "fn": int(totals[2]), "accepted": int(totals[0] + totals[1]),
            "threshold": threshold, "p_precision": p_value}


PRIMARY_ARMS = ("c2", "features")          # co-primary decompose arms (amendment 1): one Holm family inside L4a and L4b


def run_arm(summary: Mapping[str, Any]) -> str:
    """The passive arm of an `erased` run (`c2`, `features`, `c5full`; `default` for runs without one, e.g. synthetic)."""
    return str(((summary["runs"][0].get("meta") or {}).get("passive") or {}).get("kind", "default"))


def arm_acceptance(runs: Sequence[Path], summaries: Sequence[Mapping[str, Any]], rule: str, *, precision: float,
                   min_accepted: int, far_ceiling: float, resamples: int) -> dict[str, Any]:
    """L4a's ingredients for one arm's runs (seeds): pooled precision / recall of accepted decompose edges with the
    one-sided bootstrap test of H0 "precision ≤ `precision`", every null's pooled false-acceptance rate (`calibrated`:
    all ≤ `far_ceiling`), `enough` (≥ `min_accepted` accepted) and the point criterion; no Holm (the caller's family)."""
    per_concept = [json.loads((Path(r) / "per_concept.json").read_text())[rule] for r in runs]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "tested": 0})
    for summary in summaries:
        for kind, by_rule in summary["runs"][0]["nulls"].items():
            counts[kind]["accepted"] += int(by_rule[rule]["accepted"]); counts[kind]["tested"] += int(by_rule[rule]["tested"])
    rates = {k: (v["accepted"] / v["tested"] if v["tested"] else float("nan")) for k, v in counts.items()}
    l4a = precision_bootstrap(per_concept, threshold=precision, resamples=resamples)
    l4a.update(rule=rule, null_rates=rates, null_counts=dict(counts),
               calibrated=bool(rates) and all(math.isfinite(r) and r <= far_ceiling for r in rates.values()),
               enough=l4a["accepted"] >= min_accepted, point_met=math.isfinite(l4a["precision"]) and l4a["precision"] >= precision)
    return l4a


def run_report(runs: Sequence[Path], output: Path, *, precision: float = 0.8, min_accepted: int = 30,
               far_ceiling: float = 0.05, resamples: int = 2000, alpha: float = 0.05) -> dict[str, Any]:
    """Pool the `erased` runs of one track over checkpoint seeds into the pre-registered endpoints (preregistration §4 and
    amendment 1). Runs are grouped by passive arm; the co-primary arms `c2` and `features` form one Holm family (any
    other arm, e.g. the `c5full` positive control, is reported descriptively, unadjusted; with no co-primary arm present
    every arm forms the family).

    - **L4a** per arm: ≥ `min_accepted` accepted decompose edges, every null's pooled false-acceptance rate ≤ 5%
      (inclusive), and H0 "precision ≤ 0.80" rejected by the one-sided pigeonhole bootstrap with Holm across the family's
      arms. Met iff met for at least one family arm.
    - **L4b** per arm: decompose − each of AMIE / TransE / RotatE / ComplEx in recall at precision 0.8 on the shared pool
      > 0 with the 95% bootstrap CI excluding 0, Holm over every comparison of the family (arms × 4). Met for an arm iff all
      four of its comparisons are supported; overall iff met for at least one family arm."""
    from ..statistics import holm_adjust
    summaries = [json.loads((Path(r) / "summary.json").read_text()) for r in runs]
    rules = {s["runs"][0]["rule"] for s in summaries}
    if len(rules) != 1:
        raise ValueError(f"runs disagree on the primary rule: {sorted(rules)}")
    rule = rules.pop()
    by_arm: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(summaries):
        by_arm[run_arm(s)].append(i)
    family = [a for a in PRIMARY_ARMS if a in by_arm] or sorted(by_arm)
    arms: dict[str, dict[str, Any]] = {}
    for arm, index in sorted(by_arm.items()):
        pools = [dict(np.load(Path(runs[i]) / "pool.npz")) for i in index]
        l4a = arm_acceptance([runs[i] for i in index], [summaries[i] for i in index], rule, precision=precision,
                             min_accepted=min_accepted, far_ceiling=far_ceiling, resamples=resamples)
        comparisons = {b: pool_bootstrap(pools, "decompose", b, precision=precision, resamples=resamples)
                       for b in ("amie", *KGE_METHODS) if all(f"score_{b}" in p for p in pools)}
        arms[arm] = {"runs": [str(runs[i]) for i in index], "family": arm in family, "L4a": l4a, "L4b": {"comparisons": comparisons}}
    for arm, adj in zip(family, holm_adjust([arms[a]["L4a"]["p_precision"] for a in family])):
        a = arms[arm]["L4a"]
        a["p_holm"] = adj
        a["met"] = bool(a["enough"] and a["calibrated"] and adj <= alpha)
    keys = [(arm, b) for arm in family for b in arms[arm]["L4b"]["comparisons"]]
    for (arm, b), adj in zip(keys, holm_adjust([arms[arm]["L4b"]["comparisons"][b]["p_value"] for arm, b in keys]) if keys else []):
        c = arms[arm]["L4b"]["comparisons"][b]
        c["p_holm"] = adj
        c["supported"] = bool(c["mean"] > 0 and c["ci_low"] is not None and c["ci_low"] > 0 and adj <= alpha)
    for arm in family:
        comps = arms[arm]["L4b"]["comparisons"]
        arms[arm]["L4b"]["met"] = bool(comps) and all(c["supported"] for c in comps.values())
    for arm, block in arms.items():                    # descriptive arms: unadjusted
        if not block["family"]:
            block["L4a"]["met"] = None
            block["L4b"]["met"] = None
    report = {"schema": SCHEMA, "mode": "report", "runs": [str(r) for r in runs], "rule": rule, "family": family,
              "arms": arms, "L4a_met": any(arms[a]["L4a"]["met"] for a in family),
              "L4b_met": any(arms[a]["L4b"]["met"] for a in family), "labels": sorted({str(s.get("label")) for s in summaries})}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    lines = ["# E10.L pooled endpoints", "", f"Runs: {len(runs)}; rule `{rule}`; labels {report['labels']}; Holm family "
             f"{family} (amendment 1).", ""]
    for arm, block in arms.items():
        a = block["L4a"]
        lines += [f"## Arm `{arm}`{'' if block['family'] else ' (descriptive, unadjusted)'}", "",
                  f"**L4a** precision {_fmt(a['precision'])} {a['precision_ci']}; recall {_fmt(a['recall'])} {a['recall_ci']}; "
                  f"accepted {a['accepted']}; p(precision ≤ {precision}) {_fmt(a['p_precision'], 4)}, Holm {_fmt(a.get('p_holm'), 4)}; "
                  f"null rates {({k: round(v, 4) for k, v in a['null_rates'].items()})} → met: {a['met']}", "",
                  "**L4b** (recall at precision 0.8, decompose − baseline):", ""]
        for name, c in block["L4b"]["comparisons"].items():
            lines.append(f"- vs {name}: {_fmt(c['mean'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}], p {_fmt(c['p_value'], 4)}, "
                         f"Holm {_fmt(c.get('p_holm'), 4)} → {c.get('supported')}")
        lines += ["", f"L4b met: {block['L4b']['met']}", ""]
    lines.append(f"**Overall:** L4a met {report['L4a_met']}; L4b met {report['L4b_met']}.")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def run_seed(summary: Mapping[str, Any], fallback: int) -> int:
    """The checkpoint seed of an `erased` run: the `-s<N>` suffix of its store run folder (`meta.store`), else `fallback`."""
    store = str(((summary["runs"][0].get("meta") or {}).get("store")) or "")
    found = re.search(r"-s(\d+)/?$", store)
    return int(found.group(1)) if found else int(fallback)


def run_store_report(runs: Sequence[Path], reference_runs: Sequence[Path], output: Path, *, precision: float = 0.8,
                     min_accepted: int = 30, far_ceiling: float = 0.05, resamples: int = 2000, alpha: float = 0.05
                     ) -> dict[str, Any]:
    """Amendment 2 (decision 64; secondary S9): does the **learned** operator matter for decompose-then-verify? `runs`
    decompose over the learned C5 store, `reference_runs` over a fixed-random-operator store (C5rf: the same recipe with
    `random_fixed:unitary_hrr`, never trained), with the same erasure, probes, passive vectors, evidence and candidate
    pool (checked row by row). Runs are paired by passive arm and checkpoint seed.

    Per arm: each store's L4a ingredients (`arm_acceptance`) and pool readouts, and **learned − fixed** in decompose's
    recall at precision 0.8 on the shared pool (pigeonhole bootstrap over concepts × seeds, as L4b); Holm over the
    co-primary arms (`c2`, `features`); `c5full` (each store's positive control) descriptive. Reading per family arm:
    `learned operator better` / `fixed operator better` (difference > 0 / < 0, 95% CI excluding 0, Holm p ≤ α) or `no
    difference detected`; `inconclusive (machinery)` when a store's c5full positive control misses precision 0.8."""
    from ..statistics import holm_adjust

    def load(paths: Sequence[Path]) -> dict[tuple[str, int], tuple[Path, dict[str, Any]]]:
        table: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
        for k, path in enumerate(paths):
            summary = json.loads((Path(path) / "summary.json").read_text())
            key = (run_arm(summary), run_seed(summary, k + 1))
            if key in table:
                raise ValueError(f"two runs for arm {key[0]!r} seed {key[1]}")
            table[key] = (Path(path), summary)
        return table

    stores = {"learned": load(runs), "fixed": load(reference_runs)}
    rules = {s["runs"][0]["rule"] for table in stores.values() for _, s in table.values()}
    if len(rules) != 1:
        raise ValueError(f"runs disagree on the primary rule: {sorted(rules)}")
    rule = rules.pop()
    present = sorted({a for a, _ in stores["learned"]} & {a for a, _ in stores["fixed"]})
    family = [a for a in PRIMARY_ARMS if a in present] or present
    arms: dict[str, dict[str, Any]] = {}
    for arm in present:
        seeds = sorted({s for a, s in stores["learned"] if a == arm} & {s for a, s in stores["fixed"] if a == arm})
        block: dict[str, Any] = {"seeds": seeds, "family": arm in family, "stores": {}}
        for name, table in stores.items():
            picked = [table[(arm, s)] for s in seeds]
            methods = [p[1]["runs"][0]["pool"]["methods"].get("decompose", {}) for p in picked]
            block["stores"][name] = {
                "runs": [str(p[0]) for p in picked],
                "operators": sorted({str((p[1]["runs"][0].get("meta") or {}).get("operator")) for p in picked}),
                "L4a": arm_acceptance([p[0] for p in picked], [p[1] for p in picked], rule, precision=precision,
                                      min_accepted=min_accepted, far_ceiling=far_ceiling, resamples=resamples),
                "pool_auc_mean": float(np.mean([m.get("auc", float("nan")) for m in methods])) if methods else None}
        merged, unpaired = [], []
        for s in seeds:
            a = np.load(stores["learned"][(arm, s)][0] / "pool.npz")
            b = np.load(stores["fixed"][(arm, s)][0] / "pool.npz")
            if not all(k in a.files and k in b.files and np.array_equal(a[k], b[k]) for k in ("concept", "relation", "filler", "label")):
                unpaired.append(s)
                continue
            merged.append({"concept": a["concept"], "label": a["label"], "score_learned": a["score_decompose"],
                           "score_fixed": b["score_decompose"]})
        if merged and not unpaired:
            block["contrast"] = {"available": True, "measure": f"decompose recall at precision {precision:g}, learned − fixed",
                                 **pool_bootstrap(merged, "learned", "fixed", precision=precision, resamples=resamples)}
        else:
            block["contrast"] = {"available": False, "unpaired_seeds": unpaired,
                                 "detail": "the two stores' pools differ (erasure, probes or candidates): no paired contrast"}
        arms[arm] = block
    tested = [a for a in family if arms[a]["contrast"].get("available")]
    for arm, adjusted in zip(tested, holm_adjust([arms[a]["contrast"]["p_value"] for a in tested]) if tested else []):
        c = arms[arm]["contrast"]
        c["p_holm"] = adjusted
        significant = adjusted <= alpha and c["ci_low"] is not None and c["ci_high"] is not None
        c["reading"] = ("learned operator better" if significant and c["mean"] > 0 and c["ci_low"] > 0 else
                        "fixed operator better" if significant and c["mean"] < 0 and c["ci_high"] < 0 else "no difference detected")
    control = arms.get("c5full")
    machinery = ({name: bool(control["stores"][name]["L4a"]["point_met"]) for name in ("learned", "fixed")} if control else None)
    readings = {a: arms[a]["contrast"].get("reading", "unavailable") for a in family}
    overall = ("inconclusive (machinery)" if machinery is not None and not all(machinery.values()) else
               "learned operator better" if "learned operator better" in readings.values() else
               "fixed operator better" if "fixed operator better" in readings.values() else
               "no difference detected" if readings and all(r == "no difference detected" for r in readings.values()) else
               "unavailable")
    report = {"schema": SCHEMA, "mode": "store-report", "rule": rule, "family": family, "arms": arms,
              "positive_control_met": machinery, "readings": readings, "overall": overall,
              "labels": sorted({str(s.get("label")) for table in stores.values() for _, s in table.values()})}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    lines = ["# E10.L — learned vs fixed-operator store (amendment 2, decision 64; secondary S9)", "",
             f"Rule `{rule}`; Holm family {family}; labels {report['labels']}. Positive difference = the learned operator "
             "decomposes better.", "",
             "| arm | seeds | learned: accepted / precision | fixed: accepted / precision | pool AUC learned / fixed | "
             "recall@0.8 learned − fixed | Holm p | reading |", "|---|---|---|---|---|---|---|---|"]
    for arm, block in arms.items():
        ls, fs, c = block["stores"]["learned"], block["stores"]["fixed"], block["contrast"]
        diff = (f"{_fmt(c['mean'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]" if c.get("available") else "n/a")
        lines.append(f"| {arm}{'' if block['family'] else ' (descriptive)'} | {block['seeds']} | {ls['L4a']['accepted']} / "
                     f"{_fmt(ls['L4a']['precision'])} | {fs['L4a']['accepted']} / {_fmt(fs['L4a']['precision'])} | "
                     f"{_fmt(ls['pool_auc_mean'])} / {_fmt(fs['pool_auc_mean'])} | {diff} | {_fmt(c.get('p_holm'), 4)} | "
                     f"{c.get('reading', '—')} |")
    lines += ["", f"Positive controls (c5full point precision ≥ {precision}): {machinery}. **Overall:** {overall}."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


# ---------------------------------------------------------------- queue plan


FEATURES = Path.home() / "data/vsa-llm/e10/learn-features"
LOCAL_OUT = Path.home() / "data/vsa-llm/e10/learn-placement"
HOST = "SmolLM2-360M"
# Evaluation-only jobs on existing checkpoints, decisive first, inside the 54.4995 slot (author: --priority 54.4995,
# sub-levels 54.4995x); T7 trains at 51–54, so its jobs sit behind its checkpoints in priority order.
PRIORITY = {"t7-features": 54.4995, "t7-erased": 54.49951, "t7-placement": 54.49952, "t1-placement": 54.49953,
            "t4t5-features": 54.49954, "t4t5-erased": 54.49955, "wordnet-placement": 54.49956, "report": 54.49959,
            # amendment 2 (decision 64): the fixed-random-operator store (C5rf) on T4 / T5, in the band's free slots
            "t4t5-erased-c5rf": 54.49957, "report-c5rf": 54.49958}
RF_STORE = "C5rf"                          # amendment 2: E9's fixed random unitary operator arm (`random_fixed:unitary_hrr`)
ERASED_ARMS = {"c2": {}, "features": {"passive.kind": "features"},          # co-primary (amendment 1): full baselines
               "c5full": {"passive.kind": "c5full", "baselines": "[prior,correlate]"}}
# idle-GPU / CPU hour estimates from the CPU smoke (preregistration §9): 360M forward ≈ 500 tokens/s on 4 CPU threads,
# taken as ≈ 30k tokens/s on the RTX 3090 (bf16, no gradients) plus one minute to load; erased runs scale the smoke's
# stage timings to the probe count and 100 KGE epochs.
SPLIT_WINDOWS = {"t7": 19550, "t4": 3850, "t5": 5799}
ERASED_CPU_H = {"c2": {"t7": 0.15, "t4": 0.3, "t5": 0.2}, "features": {"t7": 0.15, "t4": 0.3, "t5": 0.2},
                "c5full": {"t7": 0.05, "t4": 0.08, "t5": 0.06}}
PLACEMENT_SETS = {
    "mesh-2025-2026": {"items": "experiments/toolkit-learn/items/mesh-2025-2026-v1", "files": ["placement-dev.jsonl", "placement-test.jsonl"],
                       "extract_files": ["placement-min1-dev.jsonl", "placement-min1-test.jsonl"], "role": "primary"},
    "mesh-2025-2026-min1": {"items": "experiments/toolkit-learn/items/mesh-2025-2026-v1", "files": ["placement-min1-dev.jsonl", "placement-min1-test.jsonl"],
                            "features_of": "mesh-2025-2026", "role": "secondary"},
    "mesh-2024-2025": {"items": "experiments/toolkit-learn/items/mesh-2024-2025-v1", "files": ["placement-dev.jsonl", "placement-test.jsonl"],
                       "extract_files": ["placement-min1-dev.jsonl", "placement-min1-test.jsonl"], "role": "secondary"},
}
WORDNET_SETS = {name: {"items": str(Path.home() / "data/vsa-llm/toolkit-learn/taxonomy-expansion-v1" / name), "role": role}
                for name, role in (("taxoexpan-semeval-noun", "primary"), ("taxoexpan-semeval-verb", "primary"),
                                   ("tmn-wordnet-noun", "secondary"), ("tmn-wordnet-verb", "secondary"))}


def run_folder(track: str, model: str, seed: int) -> Path:
    mode = "frozen" if model == "P0" else "full"
    return E9_RUNS / track / f"{HOST}-{mode}-{model}-s{seed}"


def queue_plan(python: str = "python", *, seeds: Sequence[int] = (1, 2, 3)) -> list[dict[str, Any]]:
    """Every E10.L job (preregistration §9): name, priority, lane, command, estimated hours and what it reads. Nothing
    here touches the queue; `queue_jobs` adds them."""
    root = ROOT
    m = ["-m", "vsa_embed.experiments.e10_learn"]
    jobs: list[dict[str, Any]] = []

    def job(name: str, stage: str, lane: str, hours: float, args: list[str], reads: Sequence[Path]) -> None:
        jobs.append({"name": name, "priority": PRIORITY[stage], "lane": lane, "hours": round(hours, 2),
                     "command": [python, *m, *args], "reads": [str(p) for p in reads]})

    def features(track: str, seed: int) -> Path:
        return FEATURES / track / f"{HOST}-full-C0p-s{seed}"

    for track in ("t7", "t4", "t5"):
        stage = "t7" if track == "t7" else "t4t5"
        for seed in seeds:
            c0p, c5 = run_folder(track, "C0p", seed), run_folder(track, "C5", seed)
            tokens = (SPLIT_WINDOWS[track] + 20000) * 512
            job(f"e10l-{track}-extract-s{seed}", f"{stage}-features", "gpu", tokens / 30000 / 3600 + 1 / 60,
                ["extract", "--run", str(c0p), "--ontology-run", str(c5), "--output", str(features(track, seed)),
                 "--splits", "eval,train:20000", "--batch", "16", "--label", "E10L"], [c0p / "final.pt"])
        for seed in seeds:
            for arm, sets in ERASED_ARMS.items():
                args = ["erased", "--config", str(root / "configs" / f"erased-{track}.yaml"),
                        "--output", str(root / "runs" / f"{track}-{arm}-s{seed}"),
                        "--set", f"store={run_folder(track, 'C5', seed)}", "--set", f"passive.run={run_folder(track, 'C2', seed)}",
                        "--set", f"evidence.features={features(track, seed) / 'occurrences.npz'}",
                        "--set", f"passive.features={features(track, seed) / 'occurrences.npz'}"]
                for k, v in sets.items():
                    args += ["--set", f"{k}={v}"]
                job(f"e10l-{track}-erased-{arm}-s{seed}", f"{stage}-erased", "cpu", ERASED_CPU_H[arm][track], args,
                    [run_folder(track, "C5", seed) / "final.pt", run_folder(track, "C2", seed) / "final.pt",
                     features(track, seed) / "occurrences.npz"])
        # amendment 1: the co-primary arms (c2, features) form one Holm family in one report; c5full apart
        for label, arms in (("primary", PRIMARY_ARMS), ("c5full", ("c5full",))):
            job(f"e10l-{track}-erased-{label}-report", "report", "cpu", 0.05,
                ["report", "--runs", *[str(root / "runs" / f"{track}-{arm}-s{s}") for arm in arms for s in seeds],
                 "--output", str(root / "runs" / f"{track}-{label}-report")], [])
    # placement: MeSH on T7 (SCRs via mapped_to; t7_group eval primary) and on T1 (descriptors via parent; seed 1)
    for track, track_seeds in (("t7", seeds), ("t1", (1,))):
        stage = f"{track}-placement"
        for seed in track_seeds:
            c0p, c5 = run_folder(track, "C0p", seed), run_folder(track, "C5", seed)
            out = FEATURES / track / f"{HOST}-full-C0p-s{seed}"
            job(f"e10l-{track}-entries-s{seed}", "t7-features" if track == "t7" else stage, "gpu", 0.03,
                ["extract-items", "--run", str(c0p), "--store-entries", "--output", str(out / "entries"), "--label", "E10L"],
                [c0p / "final.pt"])
            for name, spec in PLACEMENT_SETS.items():
                if "extract_files" in spec:
                    job(f"e10l-{track}-items-{name}-s{seed}", "t7-features" if track == "t7" else stage, "gpu", 0.02,
                        ["extract-items", "--run", str(c0p), "--items", spec["items"], "--files", ",".join(spec["extract_files"]),
                         "--output", str(out / f"items-{name}"), "--label", "E10L"], [c0p / "final.pt"])
            for name, spec in PLACEMENT_SETS.items():
                terms = out / f"items-{spec.get('features_of', name)}" / "terms.npz"
                job(f"e10l-{track}-place-{name}-s{seed}", stage, "cpu", 0.15,
                    ["placement", "--config", str(root / "configs" / f"placement-mesh-{track}.yaml"),
                     "--output", str(root / "runs" / f"place-{name}-{track}-s{seed}"),
                     "--set", f"placement.items={spec['items']}", "--set", f"placement.files=[{','.join(spec['files'])}]",
                     "--set", f"placement.store={c5}", "--set", f"placement.fit_features={out / 'entries' / 'terms.npz'}",
                     "--set", f"placement.features={terms}", "--set", f"label=E10L-{spec['role']}"],
                    [c5 / "final.pt", out / "entries" / "terms.npz", terms])
        if len(track_seeds) > 1:
            for name in PLACEMENT_SETS:
                job(f"e10l-{track}-place-{name}-report", "report", "cpu", 0.02,
                    ["placement-report", "--runs", *[str(root / "runs" / f"place-{name}-{track}-s{s}") for s in track_seeds],
                     "--output", str(root / "runs" / f"place-{name}-{track}-report")], [])
    # placement: TaxoExpan (primary) and TMN (secondary) on the WordNet track, seed 1; items and outputs local
    c0p, c5 = run_folder("wordnet", "C0p", 1), run_folder("wordnet", "C5", 1)
    out = FEATURES / "wordnet" / f"{HOST}-full-C0p-s1"
    job("e10l-wordnet-entries-s1", "wordnet-placement", "gpu", 0.08,
        ["extract-items", "--run", str(c0p), "--store-entries", "--output", str(out / "entries"), "--label", "E10L"], [c0p / "final.pt"])
    for name, spec in WORDNET_SETS.items():
        job(f"e10l-wordnet-items-{name}-s1", "wordnet-placement", "gpu", 0.02,
            ["extract-items", "--run", str(c0p), "--items", spec["items"], "--output", str(out / f"items-{name}"), "--label", "E10L"],
            [c0p / "final.pt"])
        job(f"e10l-wordnet-place-{name}-s1", "wordnet-placement", "cpu", 0.5,
            ["placement", "--config", str(root / "configs" / "placement-taxonomy-wordnet.yaml"),
             "--output", str(LOCAL_OUT / f"place-{name}-wordnet-s1"), "--set", f"placement.items={spec['items']}",
             "--set", f"placement.store={c5}", "--set", f"placement.fit_features={out / 'entries' / 'terms.npz'}",
             "--set", f"placement.features={out / f'items-{name}' / 'terms.npz'}", "--set", f"label=E10L-{spec['role']}"],
            [c5 / "final.pt"])
    return jobs


def rf_store_plan(python: str = "python", *, seeds: Sequence[int] = (1, 2, 3)) -> list[dict[str, Any]]:
    """Amendment 2 (decision 64; secondary S9): every arm of the T4 / T5 erased runs again with C5rf's store (its trained
    atomics and its fixed random operator) in place of C5's — same erasure, passive runs (C2), extracted evidence (the
    C0′ `extract` files of `queue_plan`, reused, so no GPU job) and settings — then one pooled report of the C5rf runs
    (`report`) and one paired learned-vs-fixed comparison (`store-report`) per track. CPU only."""
    root = ROOT
    m = ["-m", "vsa_embed.experiments.e10_learn"]
    jobs: list[dict[str, Any]] = []

    def job(name: str, stage: str, hours: float, args: list[str], reads: Sequence[Path]) -> None:
        jobs.append({"name": name, "priority": PRIORITY[stage], "lane": "cpu", "hours": round(hours, 2),
                     "command": [python, *m, *args], "reads": [str(p) for p in reads]})

    for track in ("t4", "t5"):
        for seed in seeds:
            features = FEATURES / track / f"{HOST}-full-C0p-s{seed}" / "occurrences.npz"
            for arm, sets in ERASED_ARMS.items():
                args = ["erased", "--config", str(root / "configs" / f"erased-{track}.yaml"),
                        "--output", str(root / "runs" / f"{track}-{arm}-c5rf-s{seed}"),
                        "--set", f"store={run_folder(track, RF_STORE, seed)}", "--set", f"passive.run={run_folder(track, 'C2', seed)}",
                        "--set", f"evidence.features={features}", "--set", f"passive.features={features}"]
                for k, v in sets.items():
                    args += ["--set", f"{k}={v}"]
                job(f"e10l-{track}-erased-{arm}-c5rf-s{seed}", "t4t5-erased-c5rf", ERASED_CPU_H[arm][track], args,
                    [run_folder(track, RF_STORE, seed) / "final.pt", run_folder(track, "C2", seed) / "final.pt", features])
        rf_runs = [str(root / "runs" / f"{track}-{arm}-c5rf-s{s}") for arm in ERASED_ARMS for s in seeds]
        c5_runs = [str(root / "runs" / f"{track}-{arm}-s{s}") for arm in ERASED_ARMS for s in seeds]
        job(f"e10l-{track}-erased-c5rf-report", "report-c5rf", 0.05,
            ["report", "--runs", *rf_runs, "--output", str(root / "runs" / f"{track}-c5rf-report")], [])
        job(f"e10l-{track}-erased-store-report", "report-c5rf", 0.1,
            ["store-report", "--runs", *c5_runs, "--reference-runs", *rf_runs, "--output", str(root / "runs" / f"{track}-store-report")], [])
    return jobs


def queue_jobs(jobs: Sequence[Mapping[str, Any]], queue_dir: Path | None = None) -> list[str]:
    """Add the jobs to the GPU queue (idempotent: an existing job name is skipped). Run from the repository root."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add
    added = []
    for j in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, list(j["command"]), name=j["name"], priority=j["priority"], min_free_gb=5,
                env={"PYTHONPATH": "src", "OMP_NUM_THREADS": "4"}, resume_args=[], lane=j["lane"])
            added.append(j["name"])
        except FileExistsError:
            pass
    return added


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("synthetic", "erased", "placement"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=None); p.add_argument("--output", type=Path, required=True)
        p.add_argument("--set", action="append", default=[], help="key.sub=value overrides (YAML values)")
    ex = sub.add_parser("extract", help="host hidden states at entry occurrences in the evaluation split (GPU job)")
    ex.add_argument("--run", type=Path, required=True)
    ex.add_argument("--ontology-run", type=Path, default=None, help="the store run whose seen entries are wanted (default --run)")
    ex.add_argument("--output", type=Path, required=True); ex.add_argument("--window", type=int, default=512)
    ex.add_argument("--max-per-entry", type=int, default=32); ex.add_argument("--batch", type=int, default=8)
    ex.add_argument("--splits", default="eval,train:20000",
                    help="data splits in order, SPLIT[:WINDOW_CAP] (default: the whole evaluation split, then 20,000 "
                    "evenly spread training windows)")
    ex.add_argument("--device", default=None)
    ex.add_argument("--min-frequency", type=int, default=10); ex.add_argument("--label", default=None)
    ex.add_argument("--alias-table", type=Path, default=None, help="default: the track's E9 alias table")
    xi = sub.add_parser("extract-items", help="host hidden states of placement items' new terms (GPU job)")
    xi.add_argument("--run", type=Path, required=True); xi.add_argument("--items", type=Path, default=None)
    xi.add_argument("--output", type=Path, required=True); xi.add_argument("--device", default=None)
    xi.add_argument("--batch", type=int, default=16); xi.add_argument("--max-contexts", type=int, default=8)
    xi.add_argument("--alias-table", type=Path, default=None); xi.add_argument("--label", default=None)
    xi.add_argument("--files", default="", help="item files of the folder (comma-separated; default dev + test)")
    xi.add_argument("--store-entries", action="store_true",
                    help="the run's own entries by their canonical surfaces (the decoder's fit features) instead of items")
    xi.add_argument("--max-entries", type=int, default=None, help="store entries: a seeded sample of at most N")
    pr = sub.add_parser("placement-report", help="pool placement runs of one set over checkpoint seeds")
    pr.add_argument("--runs", type=Path, nargs="+", required=True); pr.add_argument("--output", type=Path, required=True)
    pr.add_argument("--group", default="primary"); pr.add_argument("--resamples", type=int, default=2000)
    qu = sub.add_parser("queue", help="add every E10.L job to the GPU queue (preregistration §9); --dry-run prints them")
    qu.add_argument("--dry-run", action="store_true"); qu.add_argument("--python", default=sys.executable)
    qu.add_argument("--rf-store", action="store_true",
                    help="only amendment 2's jobs: T4 / T5 erased runs on C5rf's fixed-random-operator store and their reports")
    rp = sub.add_parser("report", help="pool erased runs over seeds into the pre-registered endpoints")
    rp.add_argument("--runs", type=Path, nargs="+", required=True); rp.add_argument("--output", type=Path, required=True)
    rp.add_argument("--resamples", type=int, default=2000)
    sr = sub.add_parser("store-report", help="amendment 2: learned (C5) vs fixed-operator (C5rf) store on paired erased runs")
    sr.add_argument("--runs", type=Path, nargs="+", required=True, help="erased runs on the learned store")
    sr.add_argument("--reference-runs", type=Path, nargs="+", required=True, help="the same runs on the fixed-operator store")
    sr.add_argument("--output", type=Path, required=True); sr.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        summary = run_synthetic(load_config(args.config, args.set), args.output)
        print(json.dumps(json_ready(summary["pooled"]), indent=2))
    elif args.command == "erased":
        summary = run_erased(load_config(args.config, args.set), args.output)
        run = summary["runs"][0]
        print(json.dumps(json_ready({"primary": run["primary"], "nulls": run["nulls"], "pool": run["pool"],
                                     "timings": run["timings"], "seconds": summary["seconds"]}), indent=2))
    elif args.command == "placement":
        summary = run_placement(load_config(args.config, args.set), args.output)
        print(json.dumps(json_ready({k: v for k, v in summary.items() if k != "ranks"}), indent=2))
    elif args.command == "extract":
        started = time.monotonic()
        ontology = torch.load(yaml.safe_load((args.ontology_run or args.run).joinpath("resolved_config.yaml").read_text())
                              ["data"]["ontology"], weights_only=False)
        frames = entry_frames(ontology)
        wanted = seen_entries(ontology, frames, min_frequency=args.min_frequency, min_degree=1)
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / "occurrences.npz"
        if target.exists():
            raise FileExistsError(f"{target} exists")
        splits = [(part.split(":")[0], int(part.split(":")[1]) if ":" in part else None) for part in args.splits.split(",")]
        data = extract_features(args.run, wanted, window=args.window, max_per_entry=args.max_per_entry, batch=args.batch,
                                splits=splits, device=args.device, alias_table=args.alias_table)
        meta = json.loads(data["meta"]); meta.update(label=args.label, seconds=round(time.monotonic() - started, 1))
        data["meta"] = json.dumps(meta)
        np.savez(target, **data)
        print(json.dumps(meta, indent=2))
    elif args.command == "extract-items":
        started = time.monotonic()
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / "terms.npz"
        if target.exists():
            raise FileExistsError(f"{target} exists")
        items = [] if args.store_entries else load_placement_items(args.items, [f for f in args.files.split(",") if f] or None)
        data = extract_items(args.run, items, batch=args.batch, max_contexts=args.max_contexts, device=args.device,
                             alias_table=args.alias_table, store_entries=args.store_entries, max_entries=args.max_entries)
        meta = json.loads(data["meta"]); meta.update(label=args.label, seconds=round(time.monotonic() - started, 1))
        data["meta"] = json.dumps(meta)
        np.savez(target, **data)
        print(json.dumps(meta, indent=2))
    elif args.command == "queue":
        jobs = rf_store_plan(args.python) if args.rf_store else queue_plan(args.python)
        if args.dry_run:
            for j in jobs:
                print(f"{j['name']}  priority {j['priority']}  lane {j['lane']}  ≈ {j['hours']} {j['lane'].upper()}-h")
                print("    " + " ".join(j["command"]))
            print(f"{len(jobs)} jobs; ≈ {sum(j['hours'] for j in jobs if j['lane'] == 'gpu'):.2f} GPU-h, "
                  f"≈ {sum(j['hours'] for j in jobs if j['lane'] == 'cpu'):.2f} CPU-h")
        else:
            print(json.dumps(queue_jobs(jobs), indent=2))
    elif args.command == "placement-report":
        run_placement_report(args.runs, args.output, group=args.group, resamples=args.resamples)
        print((args.output / "report.md").read_text())
    elif args.command == "store-report":
        run_store_report(args.runs, args.reference_runs, args.output, resamples=args.resamples)
        print((args.output / "report.md").read_text())
    else:
        run_report(args.runs, args.output, resamples=args.resamples)
        print((args.output / "report.md").read_text())


# ---------------------------------------------------------------- placement runner


PLACEMENT_TEMPLATE = "We discussed {x}"


def mention_texts(item: Mapping[str, Any], *, template: str = PLACEMENT_TEMPLATE, max_contexts: int = 8
                  ) -> list[tuple[str, int, int]]:
    """(text, span start, span end) of an item's mentions: its own `contexts` that contain the term (case-insensitive),
    else the neutral mention of its name and aliases (the binding program's P1 template), at most `max_contexts`."""
    term = str(item["term"])
    found = []
    for context in (item.get("contexts") or [])[:max_contexts]:
        at = str(context).lower().find(term.lower())
        if at >= 0:
            found.append((str(context), at, at + len(term)))
    if not found:
        for surface in list(dict.fromkeys([term, *item.get("aliases", [])]))[:max_contexts]:
            prefix = template.split("{x}")[0]
            text = template.format(x=surface)
            found.append((text, len(prefix), len(prefix) + len(surface)))
    return found


@torch.no_grad()
def extract_items(run_dir: Path, items: Sequence[Mapping[str, Any]], *, template: str = PLACEMENT_TEMPLATE, batch: int = 16,
                  max_contexts: int = 8, device: str | None = None, alias_table: Path | None = None,
                  store_entries: bool = False, min_frequency: int = 1, max_entries: int | None = None) -> dict[str, Any]:
    """Host hidden states (no channel injection) at the term's last subtoken in each of its mentions (`mention_texts`),
    averaged; layers ⌊L/2⌋ (`middle`) and L (`final`). `store_entries`: instead of `items`, the run's own entries with
    training frequency ≥ `min_frequency` and a frame, each mentioned by its canonical surface (`e5_common.canonical_surfaces`)
    — the decoder's fit features, read the same way as the new terms."""
    from .e5_common import canonical_surfaces, open_run
    run = open_run(Path(run_dir), device=device, alias_table=alias_table or default_alias_table(Path(run_dir)))
    model, tokenizer = run.model.eval(), run.tokenizer
    if store_entries:
        frames = entry_frames(run.ontology)
        frequency = np.asarray(run.ontology.get("train_frequency") or np.zeros(int(run.ontology["entry_count"])))
        held = {int(e) for e in run.ontology.get("heldout_entries", ())}
        wanted = [e for e in range(frequency.size) if frequency[e] >= min_frequency and e not in held and frames.get(e)]
        if max_entries and len(wanted) > max_entries:                     # a seeded sample (smoke tests, cost caps)
            wanted = sorted(np.random.default_rng(0).choice(wanted, int(max_entries), replace=False).tolist())
        surfaces = canonical_surfaces(run.table, tokenizer, run.min_subtokens, wanted)
        items = [{"id": str(e), "term": surfaces[e]["surface"], "aliases": []} for e in wanted if e in surfaces]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts, spans, owner = [], [], []
    for i, item in enumerate(items):
        for text, start, end in mention_texts(item, template=template, max_contexts=max_contexts):
            texts.append(text); spans.append((start, end)); owner.append(i)
    width = model.model.get_input_embeddings().weight.shape[1]
    sums = {k: np.zeros((len(items), width), np.float64) for k in ("middle", "final")}
    counts = np.zeros(len(items))
    layers = None
    for b in range(0, len(texts), batch):
        part = texts[b:b + batch]
        encoded = tokenizer(part, add_special_tokens=False, padding=True, truncation=True, max_length=1024,
                            return_offsets_mapping=True, return_tensors="pt")
        ids = encoded["input_ids"].to(run.device)
        with torch.autocast(run.device.type, dtype=torch.bfloat16, enabled=run.device.type == "cuda"):
            out = model.base(inputs_embeds=model.embed(ids, None), attention_mask=encoded["attention_mask"].to(run.device),
                             output_hidden_states=True)
        layers = len(out.hidden_states) - 1
        for r in range(len(part)):
            start, end = spans[b + r]
            inside = [k for k, (s, e) in enumerate(encoded["offset_mapping"][r].tolist()) if e > s and s < end and e > start]
            if not inside:
                continue
            i = owner[b + r]
            sums["middle"][i] += out.hidden_states[layers // 2][r, inside[-1]].float().cpu().numpy()
            sums["final"][i] += out.hidden_states[-1][r, inside[-1]].float().cpu().numpy()
            counts[i] += 1
    keep = counts > 0
    return {"id": np.asarray([str(items[i]["id"]) for i in np.flatnonzero(keep)]),
            "middle": (sums["middle"][keep] / counts[keep, None]).astype(np.float32),
            "final": (sums["final"][keep] / counts[keep, None]).astype(np.float32),
            "mentions": counts[keep].astype(np.int64),
            "meta": json.dumps({"run": str(run_dir), "items": len(items), "placed": int(keep.sum()), "layers": layers,
                                "template": template, "max_contexts": max_contexts, "store_entries": store_entries})}


def kge_vector_scores(model: Any, heads: Tensor, relation: int, tails: Sequence[int]) -> Tensor:
    """`KGEModel.score` with head *vectors* (n, D) instead of ids, against tail ids: (n, m)."""
    et = model.entity.detach()[torch.as_tensor(list(tails), dtype=torch.long)][None]          # (1, m, D)
    eh = heads.float()[:, None, :]                                                             # (n, 1, D)
    rel = model.relation.detach()[int(relation)]
    if model.kind == "transe":
        return model.gamma - (eh + rel - et).abs().sum(-1)
    half = eh.shape[-1] // 2
    hr, hi, tr, ti = eh[..., :half], eh[..., half:], et[..., :half], et[..., half:]
    if model.kind == "rotate":
        c, s = torch.cos(rel), torch.sin(rel)
        dr, di = hr * c - hi * s - tr, hr * s + hi * c - ti
        return model.gamma - torch.sqrt(dr * dr + di * di + 1e-12).sum(-1)
    rr, ri = rel[:half], rel[half:]
    return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


def entry_means(path: Path, layer: str) -> dict[int, Tensor]:
    """Per store entry its vector: the mean of its occurrences in an `extract` file (`entry` column), or its row in an
    `extract-items --store-entries` file (`id` = entry)."""
    with np.load(path) as data:
        vectors = data[layer].astype(np.float32)
        if "entry" in data.files:
            entries = data["entry"].astype(np.int64)
            return {int(e): torch.from_numpy(vectors[entries == e].mean(0)) for e in np.unique(entries).tolist()}
        return {int(i): torch.from_numpy(v) for i, v in zip(data["id"].tolist(), vectors)}


def run_placement(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Placement on a TK-H3L set. Config `placement`: `items` (item folder or file; `files` to choose), `store` (C5 run),
    `fit_features` (store entries' vectors: `extract-items --store-entries`, or an `extract` file), `features` (the new
    terms: `extract-items`), `relation_map` (item relation → store relation name), `layer`, `methods` (store decoding:
    `unbind`, `correlate`; the first is primary), `kge` (graph baselines trained on the store's frames and reached through a
    ridge map from the host vectors), `groups` (named row filters; `primary` is the pre-registered one). The text-only
    baseline scores a candidate parent by the cosine of the new term's host vector with the centroid of the parent's known
    children (store entries with that edge) in the host's space. Atomic names are matched after `kind:`."""
    from ..kg_baselines import train_kge
    from .e9_binding_chain import load_composer
    git = start(output, config)
    torch.set_num_threads(int(config["num_threads"]))
    started = time.monotonic()
    settings = config["placement"]
    layer = settings.get("layer", "middle")
    composer, _, ontology = load_composer(Path(settings["store"]))
    frames = entry_frames({"offsets": composer.schedule.offsets.cpu().numpy(), "relations": composer.schedule.relations.cpu().numpy(),
                           "fillers": composer.schedule.fillers.cpu().numpy()})
    names = list(ontology["relation_names"])
    dictionary = L.Dictionary.from_composer(composer, relation_names=names)
    items = load_placement_items(Path(settings["items"]), settings.get("files"))
    fit = entry_means(Path(settings["fit_features"]), layer)
    fit_rows = sorted(c for c in fit if frames.get(c))
    stores = dictionary.stores([frames[c] for c in fit_rows])
    _, predict, info = L.crossfit_decoder(torch.stack([fit[c] for c in fit_rows]), F.normalize(stores, dim=-1),
                                          folds=int(config["decoder"]["folds"]), seed=int(config["seed"]))
    with np.load(settings["features"]) as data:
        terms = {str(i): torch.from_numpy(v.astype(np.float32)) for i, v in zip(data["id"].tolist(), data[layer])}
    decoded = {k: predict(v[None])[0] for k, v in terms.items()}
    relation_id = {item_rel: names.index(store_rel) for item_rel, store_rel in settings["relation_map"].items() if store_rel in names}
    node_atom = {name.split(":", 1)[-1]: a for a, name in enumerate(ontology["atomic_names"])}
    children: dict[tuple[int, int], list[int]] = defaultdict(list)
    for c in fit_rows:
        for r, a in frames[c]:
            children[(int(r), int(a))].append(c)
    centroids: dict[tuple[int, int], Tensor | None] = {}

    def centroid(r: int, a: int) -> Tensor | None:
        if (r, a) not in centroids:
            members = children.get((r, a))
            centroids[(r, a)] = F.normalize(torch.stack([fit[c] for c in members]).mean(0), dim=0) if members else None
        return centroids[(r, a)]

    def text_only(item: Mapping[str, Any], atoms: list[int]) -> np.ndarray:
        vector = terms.get(item["id"])
        if vector is None:
            return np.full(len(atoms), -1.0)
        v, r = F.normalize(vector, dim=0), relation_id[item["relation"]]
        return np.asarray([float(v @ c) if (c := centroid(r, a)) is not None else -1.0 for a in atoms])

    baselines: dict[str, Callable[[Mapping[str, Any], list[int]], np.ndarray]] = {"text": text_only}
    graph = L.RuleGraph.from_atom_concepts(names, atom_concepts(ontology).tolist())
    triples, nodes = kge_graph(frames, fit_rows, graph)
    kge_info = {}
    for kind in settings.get("kge", list(KGE_METHODS)):
        model = train_kge(triples, len(nodes), len(names), kind=kind, dimension=int(config["kge"]["dimension"]),
                          epochs=int(config["kge"]["epochs"]), negatives=int(config["kge"]["negatives"]),
                          lr=float(config["kge"]["lr"]), seed=int(config["seed"]), batch_size=config["kge"].get("batch_size"))
        rows = [c for c in fit_rows if graph.concept_node(c) in nodes]
        to_kge, _ = L.ridge_fit(torch.stack([fit[c] for c in rows]),
                                model.entity.detach()[torch.tensor([nodes[graph.concept_node(c)] for c in rows])])
        kge_info[kind] = {"triples": int(triples.shape[0]), "nodes": len(nodes), "fit_rows": len(rows)}

        def kge_baseline(item: Mapping[str, Any], atoms: list[int], model=model, to_kge=to_kge) -> np.ndarray:
            vector = terms.get(item["id"])
            out = np.full(len(atoms), -1e9)
            tails = [nodes.get(graph.atom_node(a)) for a in atoms]
            known = [i for i, t in enumerate(tails) if t is not None]
            if vector is not None and known:
                out[known] = kge_vector_scores(model, to_kge(vector[None]), relation_id[item["relation"]],
                                               [tails[i] for i in known])[0].numpy()
            return out

        baselines[kind] = kge_baseline
    methods = settings.get("methods", ["unbind"])
    groups = settings.get("groups") or {"all": {}}
    results = {}
    for i, method in enumerate(methods):
        r = placement_eval(items, decoded, dictionary, relation_id, node_atom, method=method, baselines=baselines if i == 0 else None)
        r["groups"] = placement_groups(r["rows"], list(r["methods"]), groups)
        results[method] = r
    summary = {"schema": SCHEMA, "mode": "placement", "label": config.get("label"), "items_path": str(settings["items"]),
               "decoder": info, "kge": kge_info, "items": len(items), "terms_with_vectors": len(terms),
               "relation_map": settings["relation_map"],
               "results": {m: {k: v for k, v in r.items() if k != "rows"} for m, r in results.items()},
               "seconds": round(time.monotonic() - started, 1)}
    write_json(Path(output) / "summary.json", summary)
    write_records(Path(output) / "rows.jsonl.gz", [dict(row, decoding=m) for m, r in results.items() for row in r["rows"]])
    lines = ["# E10.L placement", "", f"Items {len(items)} ({settings['items']}); terms with vectors {len(terms)}; decoder "
             f"out-of-fold cosine {_fmt(info.get('oof_cosine'))}; relation map {settings['relation_map']}.", "",
             "| store decoding | group | n | coverage | method | MRR | Hits@1 | Hits@5 | Hits@10 |", "|---|---|---|---|---|---|---|---|---|"]
    for decoding, r in results.items():
        for group, g in r["groups"].items():
            for m, v in g["methods"].items():
                lines.append(f"| {decoding} | {group} | {g['n']} | {_fmt(g['coverage'])} | {m} | {_fmt(v['mrr'])} | "
                             f"{_fmt(v.get('hits@1'))} | {_fmt(v.get('hits@5'))} | {_fmt(v.get('hits@10'))} |")
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n")
    finish(output, config, git, schema=SCHEMA, seconds=summary["seconds"])
    return summary


def run_placement_report(runs: Sequence[Path], output: Path, *, group: str = "primary", decoding: str | None = None,
                         resamples: int = 2000) -> dict[str, Any]:
    """Pool placement runs of one set over checkpoint seeds (same items): per method MRR / Hits@10 of the `group`, and the
    store − baseline difference in reciprocal rank with the pigeonhole bootstrap over items × seeds
    (`statistics.two_way_cluster_bootstrap`), Holm over the baselines."""
    from ..statistics import holm_adjust, two_way_cluster_bootstrap
    summaries = [json.loads((Path(r) / "summary.json").read_text()) for r in runs]
    decoding = decoding or next(iter(summaries[0]["results"]))           # the primary (first) store decoding
    spec = (yaml.safe_load((Path(runs[0]) / "resolved_config.yaml").read_text())["placement"].get("groups") or {}).get(group, {})

    def keep(row: Mapping[str, Any]) -> bool:
        return row["decoding"] == decoding and all(row.get(k) is None or row.get(k) in (v if isinstance(v, list) else [v])
                                                   for k, v in spec.items())

    by_id = []
    for r in runs:
        with gzip.open(Path(r) / "rows.jsonl.gz", "rt") as handle:
            by_id.append({row["id"]: row for row in map(json.loads, handle) if keep(row)})
    ids = sorted(set.intersection(*[set(t) for t in by_id])) if by_id else []
    methods = [k[5:] for k in by_id[0][ids[0]] if k.startswith("rank_")] if ids else []

    def reciprocal(value: Any) -> float:
        value = float("nan") if value is None else float(value)
        return 1.0 / value if math.isfinite(value) and value > 0 else 0.0

    rr = {m: np.asarray([[reciprocal(t[i][f"rank_{m}"]) for t in by_id] for i in ids]) for m in methods}
    out: dict[str, Any] = {"runs": [str(r) for r in runs], "group": group, "decoding": decoding, "items": len(ids),
                           "methods": {m: {"mrr": float(v.mean()) if v.size else None} for m, v in rr.items()}, "comparisons": {}}
    baselines = [m for m in methods if m != "store"]
    for m in baselines:
        out["comparisons"][m] = two_way_cluster_bootstrap(rr["store"] - rr[m], resamples=resamples) if ids else None
    valid = [m for m in baselines if out["comparisons"][m]]
    for m, adj in zip(valid, holm_adjust([out["comparisons"][m]["p_value"] for m in valid])):
        out["comparisons"][m]["p_holm"] = adj
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", out)
    lines = [f"# E10.L placement, pooled ({group}, {len(runs)} runs, {len(ids)} items)", ""]
    lines += [f"- {m}: MRR {_fmt(v['mrr'])}" for m, v in out["methods"].items()]
    lines += [f"- store − {m}: {_fmt(c['mean'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}], Holm p {_fmt(c.get('p_holm'), 4)}"
              for m, c in out["comparisons"].items() if c]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
