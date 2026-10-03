"""Knowledge-graph baselines for E10 (claim D; novelty check §5.5 D-B1, D-B2): rule mining, operator
axioms and link prediction over an observed relation graph.

Everything here reads an *observed* graph — relation name → set of `(head node, tail node)` pairs, i.e.
the asserted edges a learner is given — and nothing else. Gold graphs enter only through the same
functions when the caller evaluates (e.g. `gold_axioms` mines the complete gold graph).

- **Horn-rule mining, AMIE-style** (Galárraga, Teflioudi, Hose & Suchanek, *AMIE: Association Rule Mining
  under Incomplete Evidence in Ontological Knowledge Bases*, WWW 2013): closed rules with one or two body
  atoms over two or three variables — symmetric `r(x,y) ⇒ r(y,x)`, inverse `p(x,y) ⇒ q(y,x)`, sub-property
  `p(x,y) ⇒ q(x,y)`, chains `a(x,y) ∧ b(y,z) ⇒ s(x,z)` whose body atoms may be inverted (`r⁻¹`), which
  include transitivity `r ∘ r ⇒ r`. Each rule gets support, head coverage, standard confidence and PCA
  confidence (the partial-completeness denominator: body predictions `(x, y)` whose `x` has *some* tail
  under the head relation). AMIE's defaults are kept: head coverage ≥ 0.01, PCA confidence ≥ 0.1, plus a
  minimum support of 2 pairs (this graph is small; AMIE 3 uses an absolute support of 100 on large KBs).
- **Link prediction**: TransE (Bordes et al., NIPS 2013), RotatE (Sun, Deng, Nie & Tang, ICLR 2019,
  arXiv:1902.10197; with self-adversarial negative weighting) and ComplEx (Trouillon et al., ICML 2016,
  arXiv:1606.06357), full-batch, uniformly corrupted heads or tails, pure torch.
- **Operator axioms, IterE-style** (Zhang et al., *Iteratively Learning Embeddings and Rules for Knowledge
  Graph Reasoning*, WWW 2019, arXiv:1903.08948): an axiom (a rule of the kinds above) is read off the
  relation *operators* instead of the pairs. For RotatE (an element-wise rotation, i.e. unitary HRR binding
  in the Fourier domain) a rule `a ∘ b ⇒ s` holds when the phases compose, scored as the mean cosine of
  `±θ_a ± θ_b − θ_s` (−θ for an inverted atom); for HRR role vectors (the learnable ontology's own
  operators) it is `cos(a′ ⊛ b′, s)` with `r′ = r` or its involution `r*` (the approximate inverse), so
  symmetry is `cos(r*, r)`, an inverse pair `cos(p*, q)` and transitivity `cos(r ⊛ r, r)`.
  `iterate_axioms` adds the closure of the accepted axioms to the training triples and retrains (IterE's
  injection loop).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Pair = tuple[Hashable, Hashable]
Atom = tuple[str, bool]            # (relation name, inverted)


# =====================================================================================================
# rules
# =====================================================================================================

@dataclass(frozen=True)
class Rule:
    """A closed Horn rule `body ⇒ head(x, z)`; `body` holds one or two atoms `(relation, inverted)`."""

    head: str
    body: tuple[Atom, ...]

    @property
    def kind(self) -> str:
        if len(self.body) == 1:
            (r, inv), = self.body
            if r == self.head:
                return "symmetric" if inv else "trivial"
            return "inverse" if inv else "subproperty"
        (a, ia), (b, ib) = self.body
        return "transitive" if a == b == self.head and not ia and not ib else "chain"

    @property
    def name(self) -> str:
        atom = lambda r, inv: f"{r}⁻¹" if inv else r  # noqa: E731
        return " ∧ ".join(atom(*x) for x in self.body) + f" ⇒ {self.head}"

    def hypothesis_name(self) -> str | None:
        """The E10.6 structural-hypothesis name of this rule as an explanation of its head relation
        (`ontology_hypotheses.StructuralHypothesis.name`); None for kinds without one."""
        kind = self.kind
        if kind == "symmetric":
            return "symmetric"
        if kind == "transitive":
            return "transitive"
        if kind == "inverse":
            return f"inverse_of:{self.body[0][0]}"
        if kind == "chain" and not self.body[0][1] and not self.body[1][1]:
            return f"composition_of:{self.body[0][0]}∘{self.body[1][0]}"
        return None


@dataclass
class RuleScore:
    rule: Rule
    support: int
    predictions: int
    head_size: int
    head_coverage: float
    std_confidence: float
    pca_confidence: float

    def passes(self, *, min_support: int = 2, min_head_coverage: float = 0.01, min_pca: float = 0.1) -> bool:
        return self.support >= min_support and self.head_coverage >= min_head_coverage and self.pca_confidence >= min_pca


def inverse(pairs: Iterable[Pair]) -> set[Pair]:
    return {(t, h) for h, t in pairs}


def _by_head(pairs: Iterable[Pair]) -> dict[Hashable, set[Hashable]]:
    out: dict[Hashable, set[Hashable]] = defaultdict(set)
    for h, t in pairs:
        out[h].add(t)
    return out


def compose(first: set[Pair], second: set[Pair], *, limit: int | None = None) -> set[Pair] | None:
    """`first ∘ second` (x ≠ z); None if it exceeds `limit` pairs (the rule is skipped, as AMIE prunes)."""
    out = _by_head(second)
    result: set[Pair] = set()
    for a, b in first:
        for c in out.get(b, ()):
            if c != a:
                result.add((a, c))
        if limit is not None and len(result) > limit:
            return None
    return result


def atom_pairs(relations: dict[str, set[Pair]], atom: Atom) -> set[Pair]:
    r, inv = atom
    return inverse(relations[r]) if inv else relations[r]


def rule_universe(names: Sequence[str], *, chains: bool = True, heads: Sequence[str] | None = None) -> list[Rule]:
    """Every rule of the language bias over `names` (head relations restricted to `heads`)."""
    heads = list(heads) if heads is not None else list(names)
    atoms = [(r, inv) for r in names for inv in (False, True)]
    out: list[Rule] = []
    for s in heads:
        out.append(Rule(s, ((s, True),)))                                # symmetric
        for p in names:
            if p != s:
                out.append(Rule(s, ((p, True),)))                        # inverse
                out.append(Rule(s, ((p, False),)))                       # sub-property
        if chains:
            for a in atoms:
                for b in atoms:
                    out.append(Rule(s, (a, b)))
    return out


def rule_predictions(rule: Rule, relations: dict[str, set[Pair]], *, limit: int | None = 200_000,
                     cache: dict | None = None) -> set[Pair] | None:
    """Head pairs the body implies (x ≠ z); None if a chain is too large."""
    if len(rule.body) == 1:
        return {p for p in atom_pairs(relations, rule.body[0]) if p[0] != p[1]}
    key = rule.body
    if cache is not None and key in cache:
        return cache[key]
    first, second = (atom_pairs(relations, a) for a in rule.body)
    result = compose(first, second, limit=limit)
    if cache is not None:
        cache[key] = result
    return result


def score_rule(rule: Rule, relations: dict[str, set[Pair]], *, cache: dict | None = None,
               head_index: dict[str, set[Hashable]] | None = None) -> RuleScore | None:
    if rule.head not in relations or any(a[0] not in relations for a in rule.body):
        return None
    predicted = rule_predictions(rule, relations, cache=cache)
    if predicted is None:
        return None
    head = relations[rule.head]
    heads_with_tail = head_index[rule.head] if head_index is not None else {h for h, _ in head}
    support = len(predicted & head)
    pca_base = sum(1 for x, _ in predicted if x in heads_with_tail)
    return RuleScore(rule, support, len(predicted), len(head), support / len(head) if head else 0.0,
                     support / len(predicted) if predicted else 0.0, support / pca_base if pca_base else 0.0)


def mine_rules(relations: dict[str, set[Pair]], *, heads: Sequence[str] | None = None, chains: bool = True,
               min_support: int = 2, min_head_coverage: float = 0.01, min_pca: float = 0.1,
               keep_all: bool = False) -> list[RuleScore]:
    """AMIE-style mining of the language bias (`rule_universe`); rules passing the thresholds, best PCA
    confidence first (`keep_all`: every scored rule, for threshold sweeps)."""
    names = sorted(r for r, p in relations.items() if p)
    universe = rule_universe(names, chains=chains, heads=[h for h in (heads or names) if h in relations and relations[h]])
    cache: dict = {}
    index = {r: {h for h, _ in p} for r, p in relations.items()}
    out = []
    for rule in universe:
        if rule.kind == "trivial":
            continue
        score = score_rule(rule, relations, cache=cache, head_index=index)
        if score is None:
            continue
        if keep_all or score.passes(min_support=min_support, min_head_coverage=min_head_coverage, min_pca=min_pca):
            out.append(score)
    return sorted(out, key=lambda s: (-s.pca_confidence, -s.support, s.rule.name))


def gold_axioms(relations: dict[str, set[Pair]], *, min_confidence: float = 0.9, min_support: int = 3,
                chains: bool = True) -> set[Rule]:
    """Rules that hold on a complete graph (evaluation only): standard confidence ≥ `min_confidence`."""
    return {s.rule for s in mine_rules(relations, chains=chains, keep_all=True)
            if s.support >= min_support and s.std_confidence >= min_confidence}


def closure_scores(rules: Sequence[RuleScore], relations: dict[str, set[Pair]], *, value: str = "pca_confidence",
                   limit: int | None = 200_000) -> dict[tuple[str, Hashable, Hashable], float]:
    """(relation, head, tail) → best score of a rule predicting it (pairs already present included)."""
    out: dict[tuple[str, Hashable, Hashable], float] = {}
    cache: dict = {}
    for score in rules:
        predicted = rule_predictions(score.rule, relations, limit=limit, cache=cache)
        if not predicted:
            continue
        v = float(getattr(score, value))
        for h, t in predicted:
            key = (score.rule.head, h, t)
            if v > out.get(key, float("-inf")):
                out[key] = v
    return out


# =====================================================================================================
# link prediction
# =====================================================================================================

class KGEModel(nn.Module):
    """TransE / RotatE / ComplEx over integer entity and relation ids; `score` is higher for truer triples."""

    def __init__(self, kind: str, entities: int, relations: int, dimension: int, *, gamma: float = 6.0,
                 seed: int = 0) -> None:
        super().__init__()
        if kind not in {"transe", "rotate", "complex"}:
            raise ValueError(f"unknown KGE model {kind!r}")
        g = torch.Generator().manual_seed(seed)
        self.kind, self.gamma = kind, gamma
        bound = (gamma + 2.0) / dimension
        if kind == "transe":
            self.entity = nn.Parameter((torch.rand(entities, dimension, generator=g) * 2 - 1) * bound)
            self.relation = nn.Parameter((torch.rand(relations, dimension, generator=g) * 2 - 1) * bound)
        elif kind == "rotate":            # complex entities (re | im halves), relation phases
            half = dimension // 2
            self.entity = nn.Parameter((torch.rand(entities, 2 * half, generator=g) * 2 - 1) * bound)
            self.relation = nn.Parameter((torch.rand(relations, half, generator=g) * 2 - 1) * math.pi)
        else:
            half = dimension // 2
            self.entity = nn.Parameter(torch.randn(entities, 2 * half, generator=g) * 0.1)
            self.relation = nn.Parameter(torch.randn(relations, 2 * half, generator=g) * 0.1)

    def score(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        eh, et = self.entity[h], self.entity[t]
        if self.kind == "transe":
            return self.gamma - (eh + self.relation[r] - et).abs().sum(-1)
        half = eh.shape[-1] // 2
        hr, hi, tr, ti = eh[..., :half], eh[..., half:], et[..., :half], et[..., half:]
        if self.kind == "rotate":
            phase = self.relation[r]
            c, s = torch.cos(phase), torch.sin(phase)
            dr, di = hr * c - hi * s - tr, hr * s + hi * c - ti
            return self.gamma - torch.sqrt(dr * dr + di * di + 1e-12).sum(-1)
        rel = self.relation[r]
        rr, ri = rel[..., :half], rel[..., half:]
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    def phases(self) -> Tensor:
        if self.kind != "rotate":
            raise ValueError("phases are defined for RotatE")
        return self.relation.detach()


def train_kge(triples: Tensor, entities: int, relations: int, *, kind: str, dimension: int = 64, epochs: int = 400,
              negatives: int = 32, lr: float = 0.01, gamma: float = 6.0, adversarial: float = 1.0,
              regularization: float = 1e-3, seed: int = 0) -> KGEModel:
    """Full-batch training on `(h, r, t)` rows with `negatives` uniformly corrupted heads or tails per triple.
    TransE / RotatE: the negative-sampling loss of Sun et al. (self-adversarial weights at temperature
    `adversarial`; 0 = uniform); ComplEx: logistic loss with an L2 penalty on the used embeddings."""
    torch.manual_seed(seed)
    model = KGEModel(kind, entities, relations, dimension, gamma=gamma, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    n = triples.shape[0]
    for _ in range(epochs):
        corrupt = torch.randint(entities, (n, negatives), generator=g)
        tail_side = torch.rand(n, negatives, generator=g) < 0.5
        nh = torch.where(tail_side, h[:, None].expand(-1, negatives), corrupt)
        nt = torch.where(tail_side, corrupt, t[:, None].expand(-1, negatives))
        positive = model.score(h, r, t)
        negative = model.score(nh, r[:, None].expand(-1, negatives), nt)
        if kind == "complex":
            loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
            loss = loss + regularization * (model.entity[h].pow(2).sum(-1) + model.entity[t].pow(2).sum(-1)
                                            + model.relation[r].pow(2).sum(-1)).mean()
        else:
            weights = torch.softmax(adversarial * negative.detach(), -1) if adversarial > 0 else \
                torch.full_like(negative, 1.0 / negatives)
            loss = -F.logsigmoid(positive).mean() - (weights * F.logsigmoid(-negative)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def kge_scores(model: KGEModel, triples: Tensor) -> Tensor:
    return model.score(triples[:, 0], triples[:, 1], triples[:, 2]) if triples.numel() else torch.zeros(0)


# =====================================================================================================
# operator axioms (IterE-style)
# =====================================================================================================

def rotate_axiom_score(rule: Rule, phases: dict[str, Tensor]) -> float | None:
    """Mean cosine of the composed body phases against the head phase (1 = the axiom holds exactly)."""
    if rule.head not in phases or any(a[0] not in phases for a in rule.body):
        return None
    total = sum((-phases[r] if inv else phases[r]) for r, inv in rule.body)
    return float(torch.cos(total - phases[rule.head]).mean())


def involution(vector: Tensor) -> Tensor:
    """HRR involution `r*[i] = r[−i mod d]` (the approximate inverse under circular convolution)."""
    return torch.roll(torch.flip(vector, (-1,)), 1, -1)


def circular_convolution(a: Tensor, b: Tensor) -> Tensor:
    d = a.shape[-1]
    return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=d)


def hrr_axiom_score(rule: Rule, roles: dict[str, Tensor]) -> float | None:
    """`cos(a′ ⊛ b′, s)` (one body atom: `cos(a′, s)`) with `r′ = r*` for an inverted atom."""
    if rule.head not in roles or any(a[0] not in roles for a in rule.body):
        return None
    vectors = [involution(roles[r]) if inv else roles[r] for r, inv in rule.body]
    composed = vectors[0] if len(vectors) == 1 else circular_convolution(vectors[0], vectors[1])
    return float(F.cosine_similarity(composed, roles[rule.head], dim=-1))


def axiom_closure(rules: Sequence[Rule], relations: dict[str, set[Pair]], *, limit: int | None = 200_000) -> dict[str, set[Pair]]:
    """New pairs per head relation implied by `rules` (one application, pairs already present removed)."""
    out: dict[str, set[Pair]] = defaultdict(set)
    cache: dict = {}
    for rule in rules:
        predicted = rule_predictions(rule, relations, limit=limit, cache=cache) if rule.head in relations else None
        if predicted:
            out[rule.head] |= predicted - relations.get(rule.head, set())
    return dict(out)


def prf(predicted: set, gold: set) -> dict[str, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else float("nan")
    r = tp / len(gold) if gold else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f, "predicted": len(predicted), "gold": len(gold), "tp": tp}


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann–Whitney AUC with mid-rank ties."""
    pos = sum(bool(y) for y in labels); neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + j + 1) / 2
        i = j
    rank_sum = sum(rk for rk, y in zip(ranks, labels) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def best_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """The score threshold (predict ≥ threshold) with the largest F1; ties → the higher threshold."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    total = sum(bool(y) for _, y in pairs)
    if not total or not pairs:
        return float("inf")
    best, best_f1, tp = pairs[0][0], -1.0, 0
    for i, (s, y) in enumerate(pairs):
        tp += bool(y)
        if i + 1 < len(pairs) and pairs[i + 1][0] == s:
            continue
        f1 = 2 * tp / (i + 1 + total)
        if f1 > best_f1:
            best, best_f1 = s, f1
    return float(best)
