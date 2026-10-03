"""Reading a relation off its pairs: structural signatures, label-free hypotheses, alignment metrics.

A discovered relation (a blank slot) is a set of `(head, tail)` node pairs. Nothing in this module's
*learning-side* functions sees a relation name or a gold pair set:

- `pair_signatures` — logical properties of a pair set (symmetry, antisymmetry, transitivity,
  functionality, injectivity, overlap with the inverse of / composition of / containment in the
  model's own existing relations).
- `generate_hypotheses` — the riddle step of E10.6: candidate *structural* explanations generated
  from the captured pairs (abduction, in-sample): symmetric, antisymmetric, transitive, functional,
  one-to-one, inverse-of-X, composition-of-X-and-Y, sub-relation-of-X, where X, Y are relations the
  model already has (seed relations or earlier crystallized slots, by its own names for them).
- `StructuralHypothesis.predict` — what a hypothesis predicts about *unseen* pairs (pairs that should
  hold, pairs that should not); `self_test.score_hypotheses` tests those predictions on held-out data.
- `describe_pairs` — a short textual description of a pair set; the `namer` hook lets E10.3 replace it
  with a natural-language hypothesis written by a pretrained host.

Evaluation-only helpers (`jaccard`, `best_match`, `property_holds`, `adjusted_rand_index`) take gold
pair sets and are never called inside the learning loop.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable

Pair = tuple[Hashable, Hashable]


def _tails_by_head(pairs: Iterable[Pair]) -> dict[Hashable, set[Hashable]]:
    out: dict[Hashable, set[Hashable]] = defaultdict(set)
    for h, t in pairs:
        out[h].add(t)
    return out


def _heads_by_tail(pairs: Iterable[Pair]) -> dict[Hashable, set[Hashable]]:
    out: dict[Hashable, set[Hashable]] = defaultdict(set)
    for h, t in pairs:
        out[t].add(h)
    return out


def symmetry(pairs: set[Pair]) -> float:
    pairs = {p for p in pairs if p[0] != p[1]}
    return sum((t, h) in pairs for h, t in pairs) / len(pairs) if pairs else float("nan")


def transitivity(pairs: set[Pair]) -> tuple[float, int]:
    """Fraction of 2-paths `a→b→c` (a ≠ c) whose shortcut `a→c` is in the set, and the path count."""
    out = _tails_by_head(pairs)
    paths = closed = 0
    for a, mids in out.items():
        for b in mids:
            for c in out.get(b, ()):
                if c != a:
                    paths += 1
                    closed += (a, c) in pairs
    return (closed / paths if paths else float("nan")), paths


def functionality(pairs: set[Pair]) -> float:
    out = _tails_by_head(pairs)
    return sum(len(v) == 1 for v in out.values()) / len(out) if out else float("nan")


def injectivity(pairs: set[Pair]) -> float:
    inn = _heads_by_tail(pairs)
    return sum(len(v) == 1 for v in inn.values()) / len(inn) if inn else float("nan")


def compose_pairs(first: set[Pair], second: set[Pair], *, limit: int | None = None) -> set[Pair]:
    """`first ∘ second` = {(a, c) : (a, b) ∈ first, (b, c) ∈ second, a ≠ c}."""
    out = _tails_by_head(second)
    result: set[Pair] = set()
    for a, b in first:
        for c in out.get(b, ()):
            if c != a:
                result.add((a, c))
                if limit is not None and len(result) >= limit:
                    return result
    return result


def inverse(pairs: Iterable[Pair]) -> set[Pair]:
    return {(t, h) for h, t in pairs}


def pair_signatures(pairs: set[Pair], relations: dict[str, set[Pair]] | None = None, *,
                    composition_limit: int = 20000) -> dict[str, float | int | dict[str, float]]:
    """In-sample logical signature of a pair set against the model's own named relations."""
    relations = relations or {}
    trans, paths = transitivity(pairs)
    n = max(1, len(pairs))
    signature: dict[str, float | int | dict[str, float]] = {
        "pairs": len(pairs), "heads": len({h for h, _ in pairs}), "tails": len({t for _, t in pairs}),
        "symmetry": symmetry(pairs), "transitivity": trans, "two_paths": paths,
        "functionality": functionality(pairs), "injectivity": injectivity(pairs),
        "inverse_of": {name: len(pairs & inverse(other)) / n for name, other in relations.items()},
        "sub_relation_of": {name: len(pairs & other) / n for name, other in relations.items()},
    }
    compositions = {}
    for x, first in relations.items():
        for y, second in relations.items():
            composed = compose_pairs(first, second, limit=composition_limit)
            if composed:
                compositions[f"{x}∘{y}"] = len(pairs & composed) / n
    signature["composition_of"] = compositions
    return signature


@dataclass(frozen=True)
class StructuralHypothesis:
    """A label-free structural explanation of a pair set and the predictions it makes."""

    kind: str                      # symmetric | antisymmetric | transitive | functional | one_to_one |
    args: tuple[str, ...] = ()     # inverse_of | composition_of | sub_relation_of | unstructured
    support: float = 0.0           # in-sample support that generated it

    @property
    def name(self) -> str:
        return self.kind if not self.args else f"{self.kind}:{'∘'.join(self.args)}"

    def predict(self, pairs: set[Pair], relations: dict[str, set[Pair]], *, heads: set[Hashable],
                tails: set[Hashable], limit: int = 400, seed: int = 0) -> tuple[set[Pair], set[Pair]]:
        """Pairs predicted to hold and predicted not to hold, outside the captured set.

        `heads` are the nodes that can head an edge (concepts with a composable frame); `tails` the
        nodes that can be fillers.
        """
        rng = random.Random(seed)
        positive: set[Pair] = set(); negative: set[Pair] = set()
        if self.kind == "symmetric":
            positive = {(t, h) for h, t in pairs if t in heads and h in tails} - pairs
        elif self.kind == "antisymmetric":
            negative = {(t, h) for h, t in pairs if t in heads and h in tails} - pairs
        elif self.kind == "transitive":
            positive = {p for p in compose_pairs(pairs, pairs) if p[0] in heads} - pairs
        elif self.kind == "inverse_of":
            positive = {(t, h) for h, t in relations[self.args[0]] if t in heads and h in tails} - pairs
        elif self.kind == "composition_of":
            positive = {p for p in compose_pairs(relations[self.args[0]], relations[self.args[1]])
                        if p[0] in heads and p[1] in tails} - pairs
        elif self.kind == "sub_relation_of":
            # Pairs outside the parent relation should not hold under its sub-relation.
            parent = relations[self.args[0]]
            pool = sorted({t for _, t in pairs}, key=repr)
            for h in sorted({h for h, _ in pairs}, key=repr):
                for t in rng.sample(pool, min(2, len(pool))):
                    if (h, t) not in parent and (h, t) not in pairs:
                        negative.add((h, t))
        elif self.kind in {"functional", "one_to_one"}:
            by = _tails_by_head(pairs) if self.kind == "functional" else _heads_by_tail(pairs)
            pool = sorted({t for _, t in pairs} if self.kind == "functional" else {h for h, _ in pairs}, key=repr)
            for key in sorted(by, key=repr):
                for other in rng.sample(pool, min(2, len(pool))):
                    candidate = (key, other) if self.kind == "functional" else (other, key)
                    if candidate not in pairs and other not in by[key] and candidate[0] in heads:
                        negative.add(candidate)
        elif self.kind != "unstructured":
            raise ValueError(f"unknown hypothesis kind {self.kind!r}")
        if len(positive) > limit:
            positive = set(rng.sample(sorted(positive, key=repr), limit))
        if len(negative) > limit:
            negative = set(rng.sample(sorted(negative, key=repr), limit))
        return positive, negative

    def describe(self) -> str:
        return {
            "symmetric": "symmetric (if a→b then b→a)", "antisymmetric": "antisymmetric (a→b excludes b→a)",
            "transitive": "transitive (a→b, b→c ⇒ a→c)", "functional": "functional (one tail per head)",
            "one_to_one": "one-to-one (one head per tail)", "unstructured": "no structural rule",
        }.get(self.kind, {"inverse_of": "inverse of {0}", "composition_of": "composition {0} then {1}",
                          "sub_relation_of": "sub-relation of {0}"}.get(self.kind, self.kind).format(*self.args))


@dataclass
class HypothesisSettings:
    min_overlap: float = 0.2          # inverse_of / composition_of / sub_relation_of support
    min_symmetry: float = 0.1
    max_antisymmetry: float = 0.05
    min_transitivity: float = 0.2
    min_two_paths: int = 3
    min_functional: float = 0.8
    include_all: bool = False         # skip the in-sample generation filter (all kinds, all relations)


def generate_hypotheses(pairs: set[Pair], relations: dict[str, set[Pair]],
                        settings: HypothesisSettings | None = None) -> list[StructuralHypothesis]:
    """Candidate structural explanations of `pairs` from in-sample signatures (no labels)."""
    s = settings or HypothesisSettings()
    sig = pair_signatures(pairs, relations)
    out = [StructuralHypothesis("unstructured")]
    sym = sig["symmetry"]
    if s.include_all or (sym == sym and sym >= s.min_symmetry):
        out.append(StructuralHypothesis("symmetric", support=float(sym if sym == sym else 0)))
    if s.include_all or (sym == sym and sym <= s.max_antisymmetry):
        out.append(StructuralHypothesis("antisymmetric", support=float(1 - sym if sym == sym else 0)))
    trans = sig["transitivity"]
    if s.include_all or (trans == trans and trans >= s.min_transitivity and sig["two_paths"] >= s.min_two_paths):
        out.append(StructuralHypothesis("transitive", support=float(trans if trans == trans else 0)))
    for kind, key in (("functional", "functionality"), ("one_to_one", "injectivity")):
        value = sig[key]
        if s.include_all or (value == value and value >= s.min_functional):
            out.append(StructuralHypothesis(kind, support=float(value if value == value else 0)))
    for kind, key in (("inverse_of", "inverse_of"), ("sub_relation_of", "sub_relation_of")):
        for name, value in sig[key].items():
            if s.include_all or value >= s.min_overlap:
                out.append(StructuralHypothesis(kind, (name,), support=float(value)))
    for name, value in sig["composition_of"].items():
        if s.include_all or value >= s.min_overlap:
            out.append(StructuralHypothesis("composition_of", tuple(name.split("∘")), support=float(value)))
    return out


Namer = Callable[[dict], str]


def describe_pairs(pairs: set[Pair], hypothesis: StructuralHypothesis | None, signature: dict, *,
                   examples: int = 5, node_name: Callable[[Hashable], str] = str, namer: Namer | None = None) -> str:
    """Short description of a discovered relation; `namer` (E10.3 hook) may replace it with free text."""
    summary = {"pairs": sorted(pairs, key=repr)[:examples], "signature": signature,
               "hypothesis": hypothesis.name if hypothesis else None}
    if namer is not None:
        return namer(summary)
    shown = ", ".join(f"({node_name(h)}, {node_name(t)})" for h, t in summary["pairs"])
    rule = hypothesis.describe() if hypothesis else "no structural rule adopted"
    return f"relation over {len(pairs)} pairs; {rule}; e.g. {shown}"


# -- evaluation only (gold) -------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def best_match(pairs: set[Pair], gold: dict[str, set[Pair]]) -> tuple[str | None, float]:
    """Gold relation with the largest Jaccard overlap (evaluation only)."""
    if not gold or not pairs:
        return None, 0.0
    name = max(sorted(gold), key=lambda k: jaccard(pairs, gold[k]))
    return name, jaccard(pairs, gold[name])


def property_holds(hypothesis: StructuralHypothesis, gold_pairs: set[Pair], gold_relations: dict[str, set[Pair]],
                   relation_alias: dict[str, str], *, threshold: float = 0.9) -> bool:
    """Is a structural hypothesis true of a gold relation? `relation_alias` maps the model's names for
    existing relations (seed labels, slot ids) to gold names (evaluation only)."""
    if hypothesis.kind == "unstructured":
        return False
    if hypothesis.kind == "symmetric":
        return symmetry(gold_pairs) >= threshold
    if hypothesis.kind == "antisymmetric":
        return symmetry(gold_pairs) <= 1 - threshold
    if hypothesis.kind == "transitive":
        value, paths = transitivity(gold_pairs)
        return paths > 0 and value >= threshold
    if hypothesis.kind == "functional":
        return functionality(gold_pairs) >= threshold
    if hypothesis.kind == "one_to_one":
        return injectivity(gold_pairs) >= threshold
    names = [relation_alias.get(a) for a in hypothesis.args]
    if any(n is None or n not in gold_relations for n in names):
        return False
    n = max(1, len(gold_pairs))
    if hypothesis.kind == "inverse_of":
        return len(gold_pairs & inverse(gold_relations[names[0]])) / n >= threshold
    if hypothesis.kind == "sub_relation_of":
        return len(gold_pairs & gold_relations[names[0]]) / n >= threshold
    if hypothesis.kind == "composition_of":
        composed = compose_pairs(gold_relations[names[0]], gold_relations[names[1]])
        return len(gold_pairs & composed) / n >= threshold
    return False


def adjusted_rand_index(labels_true: list, labels_pred: list) -> float:
    """Adjusted Rand index of two flat clusterings (labels may be any hashables)."""
    from math import comb
    if len(labels_true) != len(labels_pred) or not labels_true:
        raise ValueError("labelings must be equal-length and non-empty")
    table: dict[tuple, int] = defaultdict(int)
    rows: dict = defaultdict(int); cols: dict = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        table[(t, p)] += 1; rows[t] += 1; cols[p] += 1
    index = sum(comb(n, 2) for n in table.values())
    row_sum = sum(comb(n, 2) for n in rows.values()); col_sum = sum(comb(n, 2) for n in cols.values())
    total = comb(len(labels_true), 2)
    expected = row_sum * col_sum / total if total else 0.0
    maximum = (row_sum + col_sum) / 2
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


@dataclass
class PartitionScores:
    ari_all: float
    ari_gold_edges: float
    detection_precision: float
    detection_recall: float
    mean_best_jaccard: float
    per_relation: dict[str, dict[str, float]] = field(default_factory=dict)
    permuted_ari_mean: float = float("nan")
    permuted_ari_p: float = float("nan")


def partition_scores(gold: list[Hashable | None], predicted: list[Hashable | None], pairs: list[Pair], *,
                     permutations: int = 200, seed: int = 0) -> PartitionScores:
    """Score a predicted edge partition against gold relation labels (None = distractor / rejected)."""
    g = ["∅" if x is None else x for x in gold]
    p = ["∅" if x is None else x for x in predicted]
    ari_all = adjusted_rand_index(g, p) if len(set(g)) > 1 or len(set(p)) > 1 else float("nan")
    gold_idx = [i for i, x in enumerate(gold) if x is not None]
    ari_gold = adjusted_rand_index([g[i] for i in gold_idx], [p[i] for i in gold_idx]) if len(gold_idx) > 1 else float("nan")
    detected = [i for i, x in enumerate(predicted) if x is not None]
    precision = sum(gold[i] is not None for i in detected) / len(detected) if detected else float("nan")
    recall = sum(predicted[i] is not None for i in gold_idx) / len(gold_idx) if gold_idx else float("nan")
    per_relation: dict[str, dict[str, float]] = {}
    clusters: dict[Hashable, set[Pair]] = defaultdict(set)
    for i, label in enumerate(predicted):
        if label is not None:
            clusters[label].add(pairs[i])
    gold_sets: dict[Hashable, set[Pair]] = defaultdict(set)
    for i, label in enumerate(gold):
        if label is not None:
            gold_sets[label].add(pairs[i])
    for name, members in gold_sets.items():
        best = max((jaccard(members, c) for c in clusters.values()), default=0.0)
        per_relation[str(name)] = {"best_jaccard": best, "size": len(members)}
    mean_best = sum(v["best_jaccard"] for v in per_relation.values()) / len(per_relation) if per_relation else float("nan")
    rng = random.Random(seed)
    null = []
    if ari_gold == ari_gold and permutations:
        labels = [p[i] for i in gold_idx]
        for _ in range(permutations):
            shuffled = labels[:]; rng.shuffle(shuffled)
            null.append(adjusted_rand_index([g[i] for i in gold_idx], shuffled))
    scores = PartitionScores(ari_all, ari_gold, precision, recall, mean_best, per_relation)
    if null:
        scores.permuted_ari_mean = sum(null) / len(null)
        scores.permuted_ari_p = (1 + sum(x >= ari_gold for x in null)) / (1 + len(null))
    return scores
