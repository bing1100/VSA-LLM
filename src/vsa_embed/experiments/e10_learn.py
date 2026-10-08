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
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
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
    "placement": {"items": None, "store": None, "fit_features": None, "features": None, "relation_map": {}, "layer": "middle",
                  "methods": ["unbind", "correlate"], "kge": ["transe", "rotate", "complex"]},
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


def load_placement_items(path: Path) -> list[dict[str, Any]]:
    """Placement items (JSONL, one per line, or a directory holding `items.jsonl`): the format of
    `experiments/toolkit-learn/README.md` as specified to TK-H3L, read tolerantly. Required: `id`, the new term (`term`
    or `name`), the gold parents (`gold`, `parents` or `gold_parents`: identifiers of existing nodes). Optional:
    `relation` (default `parent`), `candidates` (the item's candidate parents; default: every node of the relation),
    `split` (dev / test), `text` / `definition` / `contexts`."""
    path = Path(path)
    if path.is_dir():
        path = next(p for p in (path / "items.jsonl", path / "items.jsonl.gz") if p.exists())
    opener = gzip.open if path.suffix == ".gz" else open
    items = []
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            gold = raw.get("gold", raw.get("parents", raw.get("gold_parents")))
            if gold is None:
                raise ValueError(f"placement item {raw.get('id')!r} has no gold parents")
            items.append({"id": str(raw["id"]), "term": str(raw.get("term", raw.get("name", raw["id"]))),
                          "gold": [str(g) for g in (gold if isinstance(gold, list) else [gold])],
                          "relation": str(raw.get("relation", "parent")),
                          "candidates": [str(c) for c in raw["candidates"]] if raw.get("candidates") else None,
                          "split": str(raw.get("split", "test")), "text": raw.get("text") or raw.get("definition"),
                          "contexts": raw.get("contexts")})
    return items


def placement_eval(items: Sequence[Mapping[str, Any]], term_vectors: Mapping[str, Tensor], dictionary: L.Dictionary,
                   relation_id: Mapping[str, int], node_atom: Mapping[str, int], *, method: str = "unbind",
                   baselines: Mapping[str, Callable[[Mapping[str, Any], list[int]], np.ndarray]] | None = None,
                   ks: Sequence[int] = (1, 5, 10)) -> dict[str, Any]:
    """MRR / Hits@k of the gold parents among each item's candidates: the store method (`learn.placement_scores` on the
    item's decoded passive vector) and any `baselines` (name → f(item, candidate atoms) → scores). Items whose relation,
    vector or gold parent is unknown to the store are counted as not placed (rank NaN → reciprocal rank 0)."""
    methods = {"store": None, **(baselines or {})}
    ranks: dict[str, list[float]] = {m: [] for m in methods}
    skipped = Counter()
    for item in items:
        r = relation_id.get(item["relation"])
        cand_ids = item["candidates"] if item["candidates"] is not None else None
        if r is None:
            skipped["relation"] += 1
            for m in methods:
                ranks[m].append(float("nan"))
            continue
        if cand_ids is None:
            atoms = dictionary.candidates[r].nonzero().flatten().tolist()
        else:
            atoms = [node_atom[c] for c in cand_ids if c in node_atom]
        gold_cols = [atoms.index(node_atom[g]) for g in item["gold"] if g in node_atom and node_atom[g] in atoms]
        if not atoms or not gold_cols:
            skipped["gold"] += 1
            for m in methods:
                ranks[m].append(float("nan"))
            continue
        for m, fn in methods.items():
            if m == "store":
                vector = term_vectors.get(item["id"])
                if vector is None:
                    ranks[m].append(float("nan")); continue
                s = L.placement_scores(torch.as_tensor(vector)[None], r, atoms, dictionary, method=method)
            else:
                s = torch.as_tensor(np.asarray(fn(item, atoms), dtype=np.float32))[None]
            ranks[m].append(float(L.gold_ranks(s, [gold_cols])[0]))
    return {"items": len(items), "skipped": dict(skipped),
            "methods": {m: L.ranking_metrics(v, ks) for m, v in ranks.items()}, "ranks": ranks}


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


def precision_bootstrap(per_concept: Sequence[Mapping[str, Sequence[int]]], *, resamples: int = 2000, seed: int = 0
                        ) -> dict[str, Any]:
    """Pooled precision and recall of accepted edges over seeds (per-concept TP / FP / FN tables), with the pigeonhole
    bootstrap over concepts × seeds."""
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
             for _ in range(resamples)]
    pr = np.asarray([d[0] for d in draws if math.isfinite(d[0])])
    rc = np.asarray([d[1] for d in draws if math.isfinite(d[1])])
    totals = arr.sum((0, 1))
    return {"precision": precision, "precision_ci": [float(np.quantile(pr, 0.025)), float(np.quantile(pr, 0.975))] if pr.size else None,
            "recall": recall, "recall_ci": [float(np.quantile(rc, 0.025)), float(np.quantile(rc, 0.975))] if rc.size else None,
            "tp": int(totals[0]), "fp": int(totals[1]), "fn": int(totals[2]), "accepted": int(totals[0] + totals[1])}


def run_report(runs: Sequence[Path], output: Path, *, precision: float = 0.8, min_accepted: int = 30,
               far_ceiling: float = 0.05, resamples: int = 2000) -> dict[str, Any]:
    """Pool the `erased` runs of one track and passive source over checkpoint seeds into the pre-registered endpoints:
    L4a (precision of accepted decompose edges ≥ 0.8 with ≥ `min_accepted` accepted and every null false-acceptance rate
    ≤ 5%) and L4b (decompose − each of AMIE / TransE / RotatE / ComplEx in recall at matched precision on the shared
    pool > 0; Holm over the four)."""
    from ..statistics import holm_adjust
    summaries = [json.loads((Path(r) / "summary.json").read_text()) for r in runs]
    rules = {s["runs"][0]["rule"] for s in summaries}
    if len(rules) != 1:
        raise ValueError(f"runs disagree on the primary rule: {sorted(rules)}")
    rule = rules.pop()
    pools = [dict(np.load(Path(r) / "pool.npz")) for r in runs]
    per_concept = [json.loads((Path(r) / "per_concept.json").read_text())[rule] for r in runs]
    nulls: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "tested": 0})
    for s in summaries:
        for kind, by_rule in s["runs"][0]["nulls"].items():
            n = by_rule[rule]
            nulls[kind]["accepted"] += int(n["accepted"]); nulls[kind]["tested"] += int(n["tested"])
    null_rates = {k: (v["accepted"] / v["tested"] if v["tested"] else float("nan"), v) for k, v in nulls.items()}
    l4a = precision_bootstrap(per_concept, resamples=resamples)
    l4a["rule"] = rule
    l4a["null_rates"] = {k: r for k, (r, _) in null_rates.items()}
    l4a["null_counts"] = {k: v for k, (_, v) in null_rates.items()}
    l4a["met"] = bool(l4a["accepted"] >= min_accepted and math.isfinite(l4a["precision"]) and l4a["precision"] >= precision
                      and all(math.isfinite(r) and r <= far_ceiling for r, _ in null_rates.values()))
    comparisons = {}
    for baseline in ("amie", *KGE_METHODS):
        if all(f"score_{baseline}" in p for p in pools):
            comparisons[baseline] = pool_bootstrap(pools, "decompose", baseline, precision=precision, resamples=resamples)
    adjusted = holm_adjust([c["p_value"] for c in comparisons.values()]) if comparisons else []
    for (name, c), adj in zip(comparisons.items(), adjusted):
        c["p_holm"] = adj
        c["supported"] = bool(c["mean"] > 0 and c["ci_low"] is not None and c["ci_low"] > 0 and adj <= 0.05)
    l4b = {"comparisons": comparisons, "met": bool(comparisons) and all(c["supported"] for c in comparisons.values())}
    report = {"schema": SCHEMA, "mode": "report", "runs": [str(r) for r in runs], "L4a": l4a, "L4b": l4b,
              "labels": sorted({str(s.get("label")) for s in summaries})}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    lines = ["# E10.L pooled endpoints", "", f"Runs: {len(runs)}; labels {report['labels']}.", "",
             f"**L4a** precision {_fmt(l4a['precision'])} {l4a['precision_ci']}; recall {_fmt(l4a['recall'])} {l4a['recall_ci']}; "
             f"accepted {l4a['accepted']}; null rates {l4a['null_rates']} → met: {l4a['met']}", "",
             "**L4b** (recall at precision 0.8, decompose − baseline):", ""]
    for name, c in comparisons.items():
        lines.append(f"- vs {name}: {_fmt(c['mean'])} [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}], p {_fmt(c['p_value'], 4)}, "
                     f"Holm {_fmt(c['p_holm'], 4)} → {c['supported']}")
    lines.append(f"\nL4b met: {l4b['met']}")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


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
    xi.add_argument("--run", type=Path, required=True); xi.add_argument("--items", type=Path, required=True)
    xi.add_argument("--output", type=Path, required=True); xi.add_argument("--device", default=None)
    xi.add_argument("--batch", type=int, default=16); xi.add_argument("--max-contexts", type=int, default=8)
    xi.add_argument("--alias-table", type=Path, default=None); xi.add_argument("--label", default=None)
    rp = sub.add_parser("report", help="pool erased runs over seeds into the pre-registered endpoints")
    rp.add_argument("--runs", type=Path, nargs="+", required=True); rp.add_argument("--output", type=Path, required=True)
    rp.add_argument("--resamples", type=int, default=2000)
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
        data = extract_items(args.run, load_placement_items(args.items), batch=args.batch, max_contexts=args.max_contexts,
                             device=args.device, alias_table=args.alias_table)
        meta = json.loads(data["meta"]); meta.update(label=args.label, seconds=round(time.monotonic() - started, 1))
        data["meta"] = json.dumps(meta)
        np.savez(target, vector=data["middle"], **data)
        print(json.dumps(meta, indent=2))
    else:
        run_report(args.runs, args.output, resamples=args.resamples)
        print((args.output / "report.md").read_text())


# ---------------------------------------------------------------- placement runner


@torch.no_grad()
def extract_items(run_dir: Path, items: Sequence[Mapping[str, Any]], *, template: str = "We discussed {x}", batch: int = 16,
                  max_contexts: int = 8, device: str | None = None, alias_table: Path | None = None) -> dict[str, Any]:
    """Host hidden states (no channel injection) of placement items' new terms: at the term's last subtoken in each of
    the item's `contexts` that contains the term (case-insensitive; at most `max_contexts`), else in the neutral mention
    `template`, averaged over contexts; layers ⌊L/2⌋ (`middle`) and L (`final`)."""
    from .e5_common import open_run
    run = open_run(Path(run_dir), device=device, alias_table=alias_table or default_alias_table(Path(run_dir)))
    model, tokenizer = run.model.eval(), run.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts, spans, owner = [], [], []
    for i, item in enumerate(items):
        term = str(item["term"])
        found = []
        for context in (item.get("contexts") or [])[:max_contexts]:
            at = str(context).lower().find(term.lower())
            if at >= 0:
                found.append((str(context), at, at + len(term)))
        if not found:
            text = template.format(x=term)
            at = text.find(term)
            found = [(text, at, at + len(term))]
        for text, start, end in found:
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
            offsets = encoded["offset_mapping"][r].tolist()
            inside = [k for k, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
            if not inside:
                continue
            k = inside[-1]
            i = owner[b + r]
            sums["middle"][i] += out.hidden_states[layers // 2][r, k].float().cpu().numpy()
            sums["final"][i] += out.hidden_states[-1][r, k].float().cpu().numpy()
            counts[i] += 1
    keep = counts > 0
    return {"id": np.asarray([str(items[i]["id"]) for i in np.flatnonzero(keep)]),
            "middle": (sums["middle"][keep] / counts[keep, None]).astype(np.float32),
            "final": (sums["final"][keep] / counts[keep, None]).astype(np.float32),
            "contexts": counts[keep].astype(np.int64),
            "meta": json.dumps({"run": str(run_dir), "items": len(items), "placed": int(keep.sum()), "layers": layers,
                                "template": template, "max_contexts": max_contexts})}


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
    """Per entry the mean of its occurrence vectors in an `extract` file."""
    with np.load(path) as data:
        entries = data["entry"].astype(np.int64)
        vectors = data[layer].astype(np.float32)
    return {int(e): torch.from_numpy(vectors[entries == e].mean(0)) for e in np.unique(entries).tolist()}


def run_placement(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Placement on a TK-H3L set. Config `placement`: `items` (JSONL or a folder with `items.jsonl`), `store` (C5 run),
    `fit_features` (an `extract` file of the store's host: per-entry occurrence means fit the decoder and the baselines),
    `features` (an `extract-items` file: the new terms' vectors), `relation_map` (item relation → store relation name),
    `layer`, `methods` (store decoding: `unbind`, `correlate`), `kge` (models; trained on the store's frames).
    Node identifiers of the items are matched to the store's atomics by name (the part after `kind:`)."""
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
    dictionary = L.Dictionary.from_composer(composer, relation_names=ontology["relation_names"])
    items = load_placement_items(Path(settings["items"]))
    fit = entry_means(Path(settings["fit_features"]), layer)
    fit_rows = sorted(c for c in fit if frames.get(c))
    x_fit = torch.stack([fit[c] for c in fit_rows])
    stores = dictionary.stores([frames[c] for c in fit_rows])
    _, predict, info = L.crossfit_decoder(x_fit, F.normalize(stores, dim=-1), folds=int(config["decoder"]["folds"]),
                                          seed=int(config["seed"]))
    with np.load(settings["features"]) as data:
        terms = {str(i): torch.from_numpy(v.astype(np.float32)) for i, v in zip(data["id"].tolist(), data[layer])}
    decoded = {k: predict(v[None])[0] for k, v in terms.items()}
    names = list(ontology["relation_names"])
    relation_id = {item_rel: names.index(store_rel) for item_rel, store_rel in settings["relation_map"].items()
                   if store_rel in names}
    node_atom = {name.split(":", 1)[-1]: a for a, name in enumerate(ontology["atomic_names"])}
    # text-only baseline: the new term's host vector against the centroid of each candidate parent's known children
    children: dict[tuple[int, int], list[int]] = defaultdict(list)
    for c in fit_rows:
        for r, a in frames[c]:
            children[(int(r), int(a))].append(c)
    centroid_cache: dict[tuple[int, int], Tensor | None] = {}

    def centroid(r: int, a: int) -> Tensor | None:
        if (r, a) not in centroid_cache:
            members = children.get((r, a))
            centroid_cache[(r, a)] = F.normalize(torch.stack([fit[c] for c in members]).mean(0), dim=0) if members else None
        return centroid_cache[(r, a)]

    def text_only(item: Mapping[str, Any], atoms: list[int]) -> np.ndarray:
        vector = terms.get(item["id"])
        r = relation_id[item["relation"]]
        if vector is None:
            return np.zeros(len(atoms))
        v = F.normalize(vector, dim=0)
        return np.asarray([float(v @ c) if (c := centroid(r, a)) is not None else -1.0 for a in atoms])

    baselines: dict[str, Callable[[Mapping[str, Any], list[int]], np.ndarray]] = {"text": text_only}
    graph = L.RuleGraph.from_atom_concepts(names, atom_concepts(ontology).tolist())
    triples, nodes = kge_graph(frames, fit_rows, graph)
    kge_info = {}
    for kind in settings.get("kge", list(KGE_METHODS)):
        model = train_kge(triples, len(nodes), len(names), kind=kind, dimension=int(config["kge"]["dimension"]),
                          epochs=int(config["kge"]["epochs"]), negatives=int(config["kge"]["negatives"]), lr=float(config["kge"]["lr"]),
                          seed=int(config["seed"]), batch_size=config["kge"].get("batch_size"))
        rows = [c for c in fit_rows if graph.concept_node(c) in nodes]
        to_kge, _ = L.ridge_fit(torch.stack([fit[c] for c in rows]),
                                model.entity.detach()[torch.tensor([nodes[graph.concept_node(c)] for c in rows])])
        kge_info[kind] = {"triples": int(triples.shape[0]), "nodes": len(nodes), "fit_rows": len(rows)}

        def kge_baseline(item: Mapping[str, Any], atoms: list[int], model=model, to_kge=to_kge) -> np.ndarray:
            vector = terms.get(item["id"])
            tails = [nodes.get(graph.atom_node(a)) for a in atoms]
            if vector is None:
                return np.zeros(len(atoms))
            known = [i for i, t in enumerate(tails) if t is not None]
            out = np.full(len(atoms), -1e9)
            if known:
                s = kge_vector_scores(model, to_kge(vector[None]), relation_id[item["relation"]], [tails[i] for i in known])[0]
                out[known] = s.numpy()
            return out

        baselines[kind] = kge_baseline
    results = {}
    for method in settings.get("methods", ["unbind"]):
        results[method] = placement_eval(items, decoded, dictionary, relation_id, node_atom, method=method,
                                         baselines=baselines if method == settings.get("methods", ["unbind"])[0] else None)
    summary = {"schema": SCHEMA, "mode": "placement", "label": config.get("label"), "decoder": info, "kge": kge_info,
               "items": len(items), "terms_with_vectors": len(terms),
               "results": {m: {k: v for k, v in r.items() if k != "ranks"} for m, r in results.items()},
               "seconds": round(time.monotonic() - started, 1)}
    write_json(Path(output) / "summary.json", summary)
    write_json(Path(output) / "ranks.json", {m: r["ranks"] for m, r in results.items()})
    lines = ["# E10.L placement", "", f"Items {len(items)}; terms with vectors {len(terms)}; decoder out-of-fold cosine "
             f"{_fmt(info.get('oof_cosine'))}.", "", "| store decoding | method | n | MRR | Hits@1 | Hits@5 | Hits@10 |",
             "|---|---|---|---|---|---|---|"]
    for decoding, r in results.items():
        for m, v in r["methods"].items():
            lines.append(f"| {decoding} | {m} | {v['n']} | {_fmt(v['mrr'])} | {_fmt(v.get('hits@1'))} | {_fmt(v.get('hits@5'))} | "
                         f"{_fmt(v.get('hits@10'))} |")
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n")
    finish(output, config, git, schema=SCHEMA, seconds=summary["seconds"])
    return summary


if __name__ == "__main__":
    main()
