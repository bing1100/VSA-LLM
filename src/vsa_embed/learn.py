"""The *learn* tool (decision 63, WP TK-L): make passive learning explicit — **decompose, propose, verify**.

Design: `manuscript/toolkit-methodology-2026-10.md` §3.2 (decompose-then-verify) and M3 (a null-calibrated acceptance
test); pre-registration `experiments/e10-self-semantics/e10-learn/preregistration.md`; runner
`vsa_embed.experiments.e10_learn` (E10.L).

**Interface.** The `vsa_embed.concept_store.ConceptStore` facade (TK-E13) calls exactly two functions:

    proposals = propose(evidence)          # an `Evidence` → a list of proposal dicts
    accepted = accept(proposals, test)     # an `AcceptanceTest` → the accepted proposals

A *proposal* is a plain dict `{"concept": int, "relation": int, "filler": int, "score": float, "source": str}` —
concept = the store entry (term) the edge is for, relation = relation id, filler = atomic id, score = the proposer's
own score (higher = more confident; comparable only within one source), source ∈ `decompose`, `closure`, `author` (or a
baseline's name). Extra keys (e.g. `meta`) are carried through. `accept` returns *copies* of the accepted proposals with
the test's fields added: `utility` (mean held-out contrast), `n` (held-out observations), `t`, `p` (one-sided),
`p_adjusted` and `accepted` (True); `test_proposals` returns every proposal with these fields.

**Proposal sources** (`propose`, `Evidence.sources`):

1. `decompose` — a passively learned vector, already mapped into the store's space (`crossfit_decoder`: a ridge map from
   a C2 free-table row or a host hidden state to the static stores of seen concepts, LRE-style, as the binding
   program's probe B), is expressed as a sparse non-negative sum of role ⊛ filler bindings over the trained dictionary
   (`Dictionary`: the composer's atomics and relation operator, typed candidates). The concept's current frame is
   forced into the support (warm start), so what the decomposition adds beyond it are the proposals. Methods
   (`DecomposeSettings.method`): `omp` (non-negative orthogonal matching pursuit, `frame_inference.omp_frame` extended
   to a warm start and real dictionaries; the default), `lasso` (non-negative Lasso by ISTA over the atoms screened by
   correlation) and `unbind` (a resonator-style pursuit: unbind every relation from the residual with the operator's
   own unbinding, clean up over its typed fillers, keep the best pair, refit, repeat).
2. `closure` — AMIE-style Horn rules (`kg_baselines.mine_rules`) mined on the current graph and adopted above a PCA
   confidence; their closure edges are proposals (E10.9a's mechanism: rule-implied edges for concepts no observation
   covers).
3. `author` — self-authored frames (E7 authoring cards), an optional third source (`authoring_proposals`).

**Acceptance (M3).** Every proposal, whatever its source, passes the same test (`AcceptanceTest`):
- *utility*: the held-out effect of adding the edge to the concept's frame — the change of cosine fit of the static
  store to held-out observations of the concept (`VectorUtilityTest`: e.g. host hidden states at occurrences in held-out
  windows, decoded into the store space by a decoder fitted on other concepts) or the decrease of LM loss after the term
  on held-out windows (`LossUtilityTest`);
- *control*: the same edge added to same-type control concepts whose frames lack it (`head`, default — removes type-
  level effects of a shared decoder), or random typed fillers of the same relation on the same concept (`filler`);
- *decision*: one-sided t (Welch for `head`) per proposal, Holm over all proposals of a run (`correction="holm"`, the
  default: valid under any dependence between proposals, and FWER ≤ α bounds the FDR); `bh` (Benjamini–Hochberg) and
  `knockoff` (knockoff+ filter with one matched decoy per proposal) are the pre-registered variants.

**Null worlds** (calibration; built from the same data, every proposal false by construction, `null_proposals`):
`relabel` — each proposal's relation replaced by another relation of the same filler type (relation labels permuted
within type); `swap` — its filler replaced by the same relation's filler of another term of the same type (fillers
swapped between terms of the same type). Both exclude every curated edge. The *null false-acceptance rate* is accepted /
tested over the null proposals under the identical test; the pre-registered ceiling is 5%. The runner adds a pipeline-
level null (`permute_within_type`: the proposer reads a same-type term's evidence) and, on synthetic data, a planted
null (observations generated without the erased edges).

Everything here is CPU float32 and deterministic given its seeds; gold edges enter only `null_proposals` (to exclude
curated edges from decoys) and the metric functions.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Hashable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .algebra import HRRAlgebra
from .cleanup import relation_candidates
from .frame_inference import _nnls
from .statistics import holm_adjust

Frame = list[tuple[int, int]]
Edge = tuple[int, int, int]                      # (concept, relation, filler)
KEYS = ("concept", "relation", "filler", "score", "source")
SOURCES = ("decompose", "closure", "author")
METHODS = ("omp", "lasso", "unbind")
CORRECTIONS = ("holm", "bh", "knockoff", "decoy", "holm+decoy", "none")
CONTROLS = ("head", "filler", "none")
NULL_KINDS = ("relabel", "swap")
NEG = float("-inf")


# ---------------------------------------------------------------- proposals


def make_proposal(concept: int, relation: int, filler: int, score: float, source: str, **extra: Any) -> dict[str, Any]:
    return {"concept": int(concept), "relation": int(relation), "filler": int(filler), "score": float(score),
            "source": str(source), **extra}


def edge_key(item: Mapping[str, Any]) -> Edge:
    return int(item["concept"]), int(item["relation"]), int(item["filler"])


def dedupe(proposals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One proposal per edge: the first source wins; later sources are listed under `also`."""
    out: dict[Edge, dict[str, Any]] = {}
    for p in proposals:
        key = edge_key(p)
        if key in out:
            out[key].setdefault("also", []).append(p["source"])
        else:
            out[key] = dict(p)
    return list(out.values())


def _stable(*parts: Any) -> int:
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=8).digest(), "little")


# ---------------------------------------------------------------- the dictionary


def typed_candidates(frames: Iterable[Sequence[tuple[int, int]]], relation_count: int, atomic_count: int) -> Tensor:
    """`(relations, atomics)` bool: `a` may fill `r` when some frame has the edge `(r, a)` (`cleanup.relation_candidates`)."""
    rels, fills = [], []
    for frame in frames:
        for r, a in frame:
            rels.append(int(r)); fills.append(int(a))
    return relation_candidates(torch.tensor(rels, dtype=torch.long), torch.tensor(fills, dtype=torch.long),
                               relation_count, atomic_count)


class Dictionary:
    """A trained store's binding dictionary: atomics (as composed), the relation operator `bind(r, a_vectors) = T_r(a)`
    and its `unbind`, and the typed candidates (`candidates[r, a]`). The typed atoms `{T_r(a) : candidates[r, a]}` are
    built once (`atoms`)."""

    def __init__(self, atomics: Tensor, bind: Callable[[Tensor, Tensor], Tensor], candidates: Tensor, *,
                 unbind: Callable[[Tensor, Tensor], Tensor] | None = None, relation_names: Sequence[str] | None = None,
                 chunk: int = 8192) -> None:
        self.atomics = atomics.detach().float().cpu()
        self._bind, self._unbind = bind, unbind
        self.candidates = candidates.bool().cpu()
        self.relation_count, self.atomic_count = (int(s) for s in self.candidates.shape)
        if self.atomics.shape[0] != self.atomic_count:
            raise ValueError("candidates must be (relations, atomics)")
        self.dimension = int(self.atomics.shape[1])
        self.relation_names = list(relation_names) if relation_names else [str(r) for r in range(self.relation_count)]
        self.chunk = int(chunk)
        self._atoms: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        self._index: dict[tuple[int, int], int] | None = None

    @classmethod
    def from_roles(cls, atomics: Tensor, roles: Tensor, candidates: Tensor, **kwargs: Any) -> "Dictionary":
        """HRR roles (circular convolution; unbinding by circular correlation): the synthetic worlds."""
        algebra = HRRAlgebra()
        roles = roles.detach().float()
        return cls(atomics, lambda r, v: algebra.bind(roles[r], v), candidates,
                   unbind=lambda r, v: algebra.unbind(v, roles[r]), **kwargs)

    @classmethod
    def from_composer(cls, composer: Any, frames: Iterable[Sequence[tuple[int, int]]] | None = None, **kwargs: Any
                      ) -> "Dictionary":
        """A trained `FrameComposer`: its atomics as composed, its operator and its primary unbinding
        (`relations.readout_method`; a role-blind untyped store has none). Typed candidates from `frames` (the frames the
        learner may read — erased frames in an erasure experiment), else from the composer's own schedule."""
        from .relations import readout_method
        composer.eval()
        device = composer.atomics.device
        dtype = composer.atomics.dtype
        transform = composer.transform
        method = readout_method(transform)
        with torch.no_grad():
            atomics = composer.atomic_vectors().detach().float().cpu()
        if frames is None:
            schedule = composer.schedule
            candidates = relation_candidates(schedule.relations.cpu(), schedule.fillers.cpu(), int(composer.relation_count),
                                             atomics.shape[0])
        else:
            candidates = typed_candidates(frames, int(composer.relation_count), atomics.shape[0])

        def bind(r: Tensor, v: Tensor) -> Tensor:
            with torch.no_grad():
                return transform(r.to(device), v.to(device, dtype)).float().cpu()

        def unbind(r: Tensor, v: Tensor) -> Tensor:
            with torch.no_grad():
                return composer.unbind(r.to(device), v.to(device, dtype), method=method).float().cpu()

        return cls(atomics, bind, candidates, unbind=None if method == "bundle" else unbind, **kwargs)

    # -- vectors ---------------------------------------------------------------------------------------------------

    def bound(self, relations: Tensor | Sequence[int], fillers: Tensor | Sequence[int]) -> Tensor:
        """`(n, d)` bound vectors `T_r(a)`."""
        relations = torch.as_tensor(relations, dtype=torch.long).reshape(-1)
        fillers = torch.as_tensor(fillers, dtype=torch.long).reshape(-1)
        if not relations.numel():
            return torch.zeros(0, self.dimension)
        parts = [self._bind(relations[s:s + self.chunk], self.atomics[fillers[s:s + self.chunk]])
                 for s in range(0, relations.numel(), self.chunk)]
        return torch.cat(parts).float()

    def store(self, frame: Sequence[tuple[int, int]]) -> Tensor:
        """The static store `Σ_e T_{r_e}(a_e)` of a frame (zero for an empty frame)."""
        if not len(frame):
            return torch.zeros(self.dimension)
        return self.bound([r for r, _ in frame], [a for _, a in frame]).sum(0)

    def stores(self, frames: Sequence[Sequence[tuple[int, int]]]) -> Tensor:
        """`(n, d)` static stores, one batched bind."""
        owners = [i for i, f in enumerate(frames) for _ in f]
        out = torch.zeros(len(frames), self.dimension)
        if owners:
            vectors = self.bound([r for f in frames for r, _ in f], [a for f in frames for _, a in f])
            out.index_add_(0, torch.tensor(owners, dtype=torch.long), vectors)
        return out

    def unbind(self, relations: Tensor, vectors: Tensor) -> Tensor:
        if self._unbind is None:
            raise ValueError("this dictionary's operator has no unbinding (role-blind store)")
        return self._unbind(torch.as_tensor(relations, dtype=torch.long), vectors.float()).float()

    @property
    def can_unbind(self) -> bool:
        return self._unbind is not None

    def atoms(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The typed atoms: `(relations, fillers, unit vectors, norms)` of every `T_r(a)` with `candidates[r, a]`."""
        if self._atoms is None:
            rel, fil = self.candidates.nonzero(as_tuple=True)
            vectors = self.bound(rel, fil)
            norms = vectors.norm(dim=-1).clamp_min(1e-12)
            self._atoms = (rel, fil, vectors / norms[:, None], norms)
        return self._atoms

    def atom_index(self) -> dict[tuple[int, int], int]:
        if self._index is None:
            rel, fil, _, _ = self.atoms()
            self._index = {(int(r), int(a)): i for i, (r, a) in enumerate(zip(rel.tolist(), fil.tolist()))}
        return self._index


# ---------------------------------------------------------------- decoders (passive vector → store space)


def ridge_fit(x: Tensor, y: Tensor, alphas: Sequence[float] = tuple(10.0 ** p for p in range(-2, 5))
              ) -> tuple[Callable[[Tensor], Tensor], float]:
    """Multi-output ridge on standardized inputs with the strength chosen by generalized cross-validation (the binding
    program's probe B, `e9_binding_probe.ridge_fit`); returns (predict, alpha)."""
    x, y = x.double(), y.double()
    mean, std = x.mean(0), x.std(0, unbiased=False) + 1e-8
    y_mean = y.mean(0)
    u, s, vt = torch.linalg.svd((x - mean) / std, full_matrices=False)
    centered = y - y_mean
    uy = u.T @ centered
    best: tuple[float, float] | None = None
    for alpha in alphas:
        shrink = s ** 2 / (s ** 2 + alpha)
        residual = centered - u @ (shrink[:, None] * uy)
        denominator = max(1e-12, 1 - float(shrink.sum()) / x.shape[0]) ** 2
        score = float(residual.square().mean()) / denominator
        if best is None or score < best[0]:
            best = (score, float(alpha))
    assert best is not None
    w = vt.T @ ((s / (s ** 2 + best[1]))[:, None] * uy)
    return (lambda z: (((z.double() - mean) / std) @ w + y_mean).float()), best[1]


def crossfit_decoder(x: Tensor, y: Tensor, *, folds: int = 5, seed: int = 0,
                     alphas: Sequence[float] = tuple(10.0 ** p for p in range(-2, 5))
                     ) -> tuple[Tensor, Callable[[Tensor], Tensor], dict[str, Any]]:
    """LRE-style decoder from passive vectors `x` (n, p) to store vectors `y` (n, d), fitted on the rows with targets.

    Returns the out-of-fold predictions of every row (a row's own target never fits the map that decodes it), the map
    fitted on all rows (for vectors of concepts outside the fit set) and a record (alphas, held-out R²)."""
    n = x.shape[0]
    if n < max(folds, 3):
        raise ValueError("too few rows to cross-fit a decoder")
    assignment = torch.as_tensor(np.random.default_rng(seed).permutation(n) % folds)
    out = torch.zeros(n, y.shape[1])
    chosen = []
    for k in range(folds):
        test = assignment == k
        predict, alpha = ridge_fit(x[~test], y[~test], alphas)
        out[test] = predict(x[test])
        chosen.append(alpha)
    predict, alpha = ridge_fit(x, y, alphas)
    total = float((y - y.mean(0)).square().sum())
    r2 = 1 - float((y - out).square().sum()) / total if total > 0 else float("nan")
    cos = float(F.cosine_similarity(out, y, dim=-1).mean())
    return out, predict, {"folds": folds, "fold_alphas": chosen, "alpha": alpha, "oof_r2": r2, "oof_cosine": cos,
                          "rows": n, "input_dim": int(x.shape[1]), "output_dim": int(y.shape[1])}


# ---------------------------------------------------------------- decomposition


@dataclass
class DecomposeSettings:
    method: str = "omp"                       # omp | lasso | unbind
    max_new: int = 4                          # new edges per concept (beyond its frame)
    threshold: float = 0.1                    # stop when the best residual correlation falls below (omp, unbind)
    relations: tuple[int, ...] | None = None  # relations a new edge may use (None: every relation)
    lasso_penalty: float = 0.05               # λ on the new atoms (lasso; the frame's atoms are unpenalized)
    lasso_screen: int = 256                   # atoms screened by correlation before the Lasso
    iterations: int = 300                     # ISTA iterations (lasso)
    chunk: int = 256                          # concepts per batched correlation
    min_relative: float = 0.0                 # keep a new edge only if its coefficient ≥ this × the frame's median

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")


def _frame_design(dictionary: Dictionary, frame: Sequence[tuple[int, int]]) -> Tensor:
    return dictionary.bound([r for r, _ in frame], [a for _, a in frame]) if frame else torch.zeros(0, dictionary.dimension)


def _fit(design_rows: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Non-negative least squares of `target` on the rows of `design_rows` (s, d): (coefficients, residual)."""
    if not design_rows.shape[0]:
        return torch.zeros(0), target.clone()
    beta = _nnls(design_rows.T.double(), target.double()).float()
    return beta, target - design_rows.T @ beta


def _relative(beta: Tensor, frame_size: int) -> float:
    """The median coefficient of the frame's own edges in a fit (NaN for an empty frame): the scale against which a new
    edge's coefficient is read (a bound edge of the concept carries about the same mass as its other edges)."""
    base = beta[:frame_size]
    return float(base.median()) if base.numel() else float("nan")


def _keep(coefficient: float, median: float, settings: "DecomposeSettings") -> bool:
    if coefficient <= 1e-6:
        return False
    return not (settings.min_relative > 0 and math.isfinite(median) and coefficient < settings.min_relative * median)


def _allowed_atoms(dictionary: Dictionary, relations: Collection[int] | None) -> Tensor:
    rel, _, _, _ = dictionary.atoms()
    if relations is None:
        return torch.ones(rel.numel(), dtype=torch.bool)
    allowed = torch.zeros(dictionary.relation_count, dtype=torch.bool)
    allowed[torch.as_tensor(sorted(relations), dtype=torch.long)] = True
    return allowed[rel]


def _excluded(dictionary: Dictionary, frame: Sequence[tuple[int, int]]) -> list[int]:
    index = dictionary.atom_index()
    return [index[(int(r), int(a))] for r, a in frame if (int(r), int(a)) in index]


def decompose(targets: Tensor, frames: Sequence[Sequence[tuple[int, int]]], dictionary: Dictionary,
              settings: DecomposeSettings | None = None) -> list[list[dict[str, float]]]:
    """Sparse decomposition of each target (n, d) over the typed dictionary, the frame's own edges forced into the
    support (warm start). Returns per target the *new* edges, each `{"relation", "filler", "coefficient",
    "correlation", "order", "relative"}` (non-negative coefficients > 0; correlation = the residual's correlation with
    the unit atom when it was selected — for `lasso`, with the initial residual; relative = the coefficient over the
    median coefficient of the frame's edges in the same fit; `min_relative` drops edges below that share)."""
    settings = settings or DecomposeSettings()
    if len(frames) != targets.shape[0]:
        raise ValueError("one frame per target")
    y = F.normalize(targets.float(), dim=-1)
    if settings.method == "unbind":
        return _decompose_unbind(y, frames, dictionary, settings)
    rel, fil, units, norms = dictionary.atoms()
    allowed = _allowed_atoms(dictionary, settings.relations)
    out: list[list[dict[str, float]]] = []
    for start in range(0, y.shape[0], settings.chunk):
        part = range(start, min(y.shape[0], start + settings.chunk))
        designs = [_frame_design(dictionary, frames[i]) for i in part]
        residuals = torch.stack([_fit(designs[j], y[i])[1] for j, i in enumerate(part)])
        if settings.method == "lasso":
            out.extend(_lasso_part(y[list(part)], residuals, designs, [frames[i] for i in part], dictionary, allowed,
                                   settings))
            continue
        blocked = [set(_excluded(dictionary, frames[i])) for i in part]
        selected: list[list[int]] = [[] for _ in part]
        picked: list[list[float]] = [[] for _ in part]
        active = [True] * len(part)
        for _ in range(settings.max_new):
            if not any(active):
                break
            scores = residuals @ units.T                                          # (chunk, M)
            scores[:, ~allowed] = NEG
            for j in range(len(part)):
                if blocked[j] or selected[j]:
                    scores[j, list(blocked[j] | set(selected[j]))] = NEG
            best = scores.max(-1)
            for j, i in enumerate(part):
                if not active[j]:
                    continue
                value, index = float(best.values[j]), int(best.indices[j])
                if not math.isfinite(value) or value < settings.threshold:
                    active[j] = False
                    continue
                selected[j].append(index); picked[j].append(value)
                design = torch.cat([designs[j], units[selected[j]] * norms[selected[j], None]])
                _, residuals[j] = _fit(design, y[i])
        for j, i in enumerate(part):
            edges = []
            if selected[j]:
                design = torch.cat([designs[j], units[selected[j]] * norms[selected[j], None]])
                beta, _ = _fit(design, y[i])
                median = _relative(beta, designs[j].shape[0])
                new_beta = beta[designs[j].shape[0]:]
                for order, (index, corr, b) in enumerate(zip(selected[j], picked[j], new_beta.tolist())):
                    if _keep(b, median, settings):
                        edges.append({"relation": int(rel[index]), "filler": int(fil[index]), "coefficient": float(b),
                                      "correlation": float(corr), "order": order, "relative": b / median if median > 0 else float("nan")})
            out.append(edges)
    return out


def _lasso_part(y: Tensor, residuals: Tensor, designs: list[Tensor], frames: list[Sequence[tuple[int, int]]],
                dictionary: Dictionary, allowed: Tensor, settings: DecomposeSettings) -> list[list[dict[str, float]]]:
    """Non-negative Lasso (ISTA) per target over the atoms screened by correlation with the frame-fitted residual; the
    frame's own atoms are in the design without penalty."""
    rel, fil, units, norms = dictionary.atoms()
    scores = residuals @ units.T
    scores[:, ~allowed] = NEG
    out = []
    for j in range(y.shape[0]):
        row = scores[j].clone()
        blocked = _excluded(dictionary, frames[j])
        if blocked:
            row[blocked] = NEG
        k = min(settings.lasso_screen, int(torch.isfinite(row).sum()))
        if k <= 0:
            out.append([]); continue
        top = torch.topk(row, k).indices
        base = designs[j]
        design = torch.cat([base, units[top] * norms[top, None]]).double()        # (s, d)
        target = y[j].double()
        gram = design @ design.T
        step = 1.0 / float(torch.linalg.matrix_norm(gram, ord=2).clamp_min(1e-8))
        penalty = torch.cat([torch.zeros(base.shape[0]), torch.full((k,), settings.lasso_penalty)]).double()
        beta = torch.zeros(design.shape[0], dtype=torch.float64)
        rhs = design @ target
        for _ in range(settings.iterations):
            beta = (beta - step * (gram @ beta - rhs) - step * penalty).clamp_min(0)
        new = beta[base.shape[0]:]
        median = _relative(beta.float(), base.shape[0])
        order = torch.argsort(new, descending=True)[:settings.max_new]
        edges = []
        for rank, o in enumerate(order.tolist()):
            b = float(new[o])
            if not _keep(b, median, settings):
                continue
            index = int(top[o])
            edges.append({"relation": int(rel[index]), "filler": int(fil[index]), "coefficient": b,
                          "correlation": float(scores[j, index]), "order": rank, "relative": b / median if median > 0 else float("nan")})
        out.append(edges)
    return out


def _decompose_unbind(y: Tensor, frames: Sequence[Sequence[tuple[int, int]]], dictionary: Dictionary,
                      settings: DecomposeSettings) -> list[list[dict[str, float]]]:
    """Resonator-style pursuit: per step unbind every allowed relation from the residual with the operator's own
    unbinding, clean up against that relation's typed fillers (cosine), keep the best (relation, filler) over relations,
    refit non-negatively with the frame forced, repeat (explaining away)."""
    if not dictionary.can_unbind:
        raise ValueError("the unbind pursuit needs an operator with unbinding")
    relations = list(settings.relations) if settings.relations is not None else list(range(dictionary.relation_count))
    keys = F.normalize(dictionary.atomics, dim=-1)
    out: list[list[dict[str, float]]] = []
    for i in range(y.shape[0]):
        frame = list(frames[i])
        base = _frame_design(dictionary, frame)
        _, residual = _fit(base, y[i])
        taken = {(int(r), int(a)) for r, a in frame}
        chosen: list[tuple[int, int, float]] = []
        for _ in range(settings.max_new):
            rel_ids = torch.tensor(relations, dtype=torch.long)
            unbound = dictionary.unbind(rel_ids, residual[None].expand(len(relations), -1))
            cos = F.normalize(unbound, dim=-1) @ keys.T                              # (R', A)
            cos = cos.masked_fill(~dictionary.candidates[rel_ids], NEG)
            for (r, a) in taken:
                if r in relations:
                    cos[relations.index(r), a] = NEG
            flat = int(torch.argmax(cos))
            row, a = divmod(flat, cos.shape[1])
            value = float(cos[row, a])
            if not math.isfinite(value) or value < settings.threshold:
                break
            r = relations[row]
            chosen.append((r, a, value)); taken.add((r, a))
            design = torch.cat([base, dictionary.bound([c[0] for c in chosen], [c[1] for c in chosen])])
            _, residual = _fit(design, y[i])
        edges = []
        if chosen:
            design = torch.cat([base, dictionary.bound([c[0] for c in chosen], [c[1] for c in chosen])])
            beta, _ = _fit(design, y[i])
            median = _relative(beta, base.shape[0])
            for order, ((r, a, value), b) in enumerate(zip(chosen, beta[base.shape[0]:].tolist())):
                if _keep(b, median, settings):
                    edges.append({"relation": int(r), "filler": int(a), "coefficient": float(b), "correlation": value,
                                  "order": order, "relative": b / median if median > 0 else float("nan")})
        out.append(edges)
    return out


def residual_scores(targets: Tensor, frames: Sequence[Sequence[tuple[int, int]]], dictionary: Dictionary,
                    candidates: Sequence[Sequence[tuple[int, int]]]) -> list[list[float]]:
    """The matched-filter score of candidate edges: the correlation of each candidate's unit atom with the target's
    residual after the frame alone is fitted (the first pursuit step; no explaining away). The `correlate` readout."""
    y = F.normalize(targets.float(), dim=-1)
    out = []
    for i in range(y.shape[0]):
        _, residual = _fit(_frame_design(dictionary, frames[i]), y[i])
        cand = list(candidates[i])
        if not cand:
            out.append([]); continue
        vectors = F.normalize(dictionary.bound([r for r, _ in cand], [a for _, a in cand]), dim=-1)
        out.append((vectors @ residual).tolist())
    return out


# ---------------------------------------------------------------- rule closure


@dataclass
class RuleSettings:
    min_pca: float = 0.8                      # adopt a rule at PCA confidence ≥ this (E10's dev-tuned AMIE: 0.611)
    min_support: int = 2
    min_head_coverage: float = 0.01
    chains: bool = True
    limit: int = 200_000                      # skip a chain whose body predicts more pairs (AMIE-style pruning)
    max_rules: int = 50


@dataclass
class RuleGraph:
    """Node keys for rule mining: a concept's node, an atomic's node, and the inverse maps (an atomic that names a
    concept shares that concept's node, so rules can chain through it)."""

    relation_names: list[str]
    concept_node: Callable[[int], Hashable]
    atom_node: Callable[[int], Hashable]
    node_concept: Mapping[Hashable, int]
    node_atom: Mapping[Hashable, int]

    @classmethod
    def from_atom_concepts(cls, relation_names: Sequence[str], atom_concept: Sequence[int]) -> "RuleGraph":
        """`atom_concept[a]` = the concept an atomic names (−1: a plain value). Concepts are `("c", id)`, plain atomics
        `("a", id)`."""
        atom_concept = [int(c) for c in atom_concept]
        node_atom: dict[Hashable, int] = {}
        for a, c in enumerate(atom_concept):
            node_atom[("c", c) if c >= 0 else ("a", a)] = a
        node_concept: dict[Hashable, int] = _ConceptNodes()
        return cls(list(relation_names), lambda c: ("c", int(c)),
                   lambda a: ("c", atom_concept[int(a)]) if atom_concept[int(a)] >= 0 else ("a", int(a)),
                   node_concept, node_atom)


class _ConceptNodes(dict):
    """`("c", id) → id` for any id (a lazy inverse of `RuleGraph.concept_node`)."""

    def get(self, key: Hashable, default: Any = None) -> Any:          # type: ignore[override]
        return int(key[1]) if isinstance(key, tuple) and len(key) == 2 and key[0] == "c" else default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and len(key) == 2 and key[0] == "c"


def graph_relations(frames: Mapping[int, Sequence[tuple[int, int]]], graph: RuleGraph) -> dict[str, set[tuple[Hashable, Hashable]]]:
    """Relation name → set of (head node, tail node) pairs of the frames (the observed graph)."""
    out: dict[str, set[tuple[Hashable, Hashable]]] = {name: set() for name in graph.relation_names}
    for c, frame in frames.items():
        head = graph.concept_node(int(c))
        for r, a in frame:
            out[graph.relation_names[int(r)]].add((head, graph.atom_node(int(a))))
    return out


def rule_closure(frames: Mapping[int, Sequence[tuple[int, int]]], graph: RuleGraph, *, concepts: Collection[int] | None = None,
                 settings: RuleSettings | None = None, candidates: Tensor | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mine AMIE-style rules on the frames' graph, adopt those with PCA confidence ≥ `min_pca` (and AMIE's support and
    head-coverage filters), and propose their closure edges absent from the frames for `concepts` (default: every
    concept). Returns (proposals scored by the best adopting rule's PCA confidence, adopted-rule records)."""
    from .kg_baselines import mine_rules, rule_predictions
    settings = settings or RuleSettings()
    relations = graph_relations(frames, graph)
    rules = mine_rules({k: v for k, v in relations.items() if v}, chains=settings.chains, min_support=settings.min_support,
                       min_head_coverage=settings.min_head_coverage, min_pca=settings.min_pca)[:settings.max_rules]
    relation_id = {name: i for i, name in enumerate(graph.relation_names)}
    wanted = None if concepts is None else {int(c) for c in concepts}
    best: dict[Edge, tuple[float, str]] = {}
    cache: dict = {}
    for score in rules:
        predicted = rule_predictions(score.rule, relations, limit=settings.limit, cache=cache)
        if not predicted:
            continue
        r = relation_id[score.rule.head]
        present = relations[score.rule.head]
        for h, t in predicted - present:
            c = graph.node_concept.get(h)
            a = graph.node_atom.get(t)
            if c is None or a is None or (wanted is not None and c not in wanted):
                continue
            if candidates is not None and not bool(candidates[r, a]):
                continue
            key = (int(c), r, int(a))
            if key not in best or score.pca_confidence > best[key][0]:
                best[key] = (float(score.pca_confidence), score.rule.name)
    proposals = [make_proposal(c, r, a, s, "closure", meta={"rule": name}) for (c, r, a), (s, name) in sorted(best.items())]
    records = [{"rule": s.rule.name, "kind": s.rule.kind, "support": s.support, "pca_confidence": s.pca_confidence,
                "std_confidence": s.std_confidence, "head_coverage": s.head_coverage} for s in rules]
    return proposals, records


# ---------------------------------------------------------------- self-authoring (E7) as a source


def authoring_proposals(cards: Iterable[Mapping[str, Any]], concept_of: Callable[[str], int | None],
                        relation_id: Mapping[str, int], atom_id: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """E7 authoring cards (`authoring.card_from_record`: surface, relation, filler, atom, proposals, samples) as
    proposals: the concept the surface names (`concept_of`; unknown surfaces are dropped), the card's relation and atom,
    score = the self-consistency share (proposals / samples)."""
    out = []
    for card in cards:
        c = concept_of(str(card["surface"]))
        r = relation_id.get(str(card["relation"]))
        a = card.get("atom")
        if a is None and atom_id is not None:
            a = atom_id.get(str(card.get("filler")))
        if c is None or r is None or a is None or int(a) < 0:
            continue
        share = float(card.get("proposals", 0)) / max(1.0, float(card.get("samples", 1)))
        out.append(make_proposal(c, r, int(a), share, "author", meta={"surface": card["surface"]}))
    return dedupe(out)


# ---------------------------------------------------------------- propose


@dataclass
class Evidence:
    """What `propose` reads (the learner's view; no gold).

    `frames`: the concepts' current frames (the seed ontology). `vectors`: passive vectors already mapped into the
    store's space (`crossfit_decoder`), per concept. `graph`: node keys for rule closure. `cards` + `concept_of` +
    `relation_id`: E7 authoring cards."""

    dictionary: Dictionary
    frames: Mapping[int, Sequence[tuple[int, int]]]
    vectors: Mapping[int, Tensor] | None = None
    concepts: Sequence[int] | None = None
    sources: tuple[str, ...] = ("decompose",)
    decompose: DecomposeSettings = field(default_factory=DecomposeSettings)
    rules: RuleSettings = field(default_factory=RuleSettings)
    graph: RuleGraph | None = None
    cards: Sequence[Mapping[str, Any]] | None = None
    concept_of: Callable[[str], int | None] | None = None
    relation_id: Mapping[str, int] | None = None

    def targets(self) -> list[int]:
        if self.concepts is not None:
            return [int(c) for c in self.concepts]
        return sorted(int(c) for c in (self.vectors or {}))


def propose(evidence: Evidence) -> list[dict[str, Any]]:
    """Proposals of every requested source (`evidence.sources`), one per edge (`dedupe`: the first source listed wins),
    never an edge already in the concept's frame."""
    unknown = set(evidence.sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}; known: {SOURCES}")
    concepts = evidence.targets()
    out: list[dict[str, Any]] = []
    for source in evidence.sources:
        if source == "decompose":
            if not evidence.vectors:
                raise ValueError("the decompose source needs passive vectors in the store space")
            have = [c for c in concepts if c in evidence.vectors]
            if not have:
                continue
            targets = torch.stack([torch.as_tensor(evidence.vectors[c]).float() for c in have])
            frames = [list(evidence.frames.get(c, ())) for c in have]
            for c, edges in zip(have, decompose(targets, frames, evidence.dictionary, evidence.decompose)):
                for e in edges:
                    out.append(make_proposal(c, e["relation"], e["filler"], e["coefficient"], "decompose",
                                             meta={"correlation": e["correlation"], "order": e["order"],
                                                   "relative": e.get("relative")}))
        elif source == "closure":
            if evidence.graph is None:
                raise ValueError("the closure source needs a RuleGraph")
            proposals, _ = rule_closure(evidence.frames, evidence.graph, concepts=concepts, settings=evidence.rules,
                                        candidates=evidence.dictionary.candidates)
            out.extend(proposals)
        elif source == "author":
            if evidence.cards is None or evidence.concept_of is None or evidence.relation_id is None:
                raise ValueError("the author source needs cards, concept_of and relation_id")
            wanted = set(concepts) if evidence.concepts is not None else None
            out.extend(p for p in authoring_proposals(evidence.cards, evidence.concept_of, evidence.relation_id)
                       if wanted is None or p["concept"] in wanted)
    present = {(int(c), int(r), int(a)) for c, f in evidence.frames.items() for r, a in f}
    return [p for p in dedupe(out) if edge_key(p) not in present]


# ---------------------------------------------------------------- held-out utility


@dataclass
class Contrast:
    """A proposal's held-out evidence: `values` are per-observation utilities (paired controls already subtracted);
    `control` is an unpaired control sample (head control: one mean utility per control concept) or None."""

    values: np.ndarray
    control: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.values.size)

    @property
    def mean(self) -> float:
        base = float(self.values.mean()) if self.values.size else float("nan")
        return base - float(self.control.mean()) if self.control is not None and self.control.size else base


def one_sided(contrast: Contrast | None, *, min_observations: int = 4) -> tuple[float, float]:
    """(t, one-sided p) for "utility > control": one-sample t on paired values, Welch's t against an unpaired control.
    Untestable (fewer than `min_observations`, no control, zero variance with a non-positive mean): (0, 1)."""
    from scipy import stats
    if contrast is None or contrast.n < min_observations:
        return 0.0, 1.0
    x = contrast.values.astype(np.float64)
    if contrast.control is None:
        sd = float(x.std(ddof=1))
        if sd == 0:
            return (math.inf, 0.0) if float(x.mean()) > 0 else (0.0, 1.0)
        t = float(x.mean()) / (sd / math.sqrt(x.size))
        return t, float(stats.t.sf(t, x.size - 1))
    y = contrast.control.astype(np.float64)
    if y.size < 2:
        return 0.0, 1.0
    vx, vy = float(x.var(ddof=1)) / x.size, float(y.var(ddof=1)) / y.size
    se = math.sqrt(vx + vy)
    diff = float(x.mean() - y.mean())
    if se == 0:
        return (math.inf, 0.0) if diff > 0 else (0.0, 1.0)
    t = diff / se
    df = (vx + vy) ** 2 / ((vx ** 2 / (x.size - 1) if x.size > 1 else 0) + (vy ** 2 / (y.size - 1)) + 1e-300)
    return t, float(stats.t.sf(t, df))


class UtilityTest(Protocol):
    def contrasts(self, proposals: Sequence[Mapping[str, Any]]) -> list[Contrast | None]: ...


def concept_types(frames: Mapping[int, Sequence[tuple[int, int]]], *, type_relation: int | None = None,
                  min_members: int = 5) -> dict[int, Hashable]:
    """A concept's type from the learner's view: the sorted fillers of `type_relation` if given, else the frame's
    relation signature (sorted distinct relations). Types with fewer than `min_members` concepts merge into "other"."""
    raw: dict[int, Hashable] = {}
    for c, frame in frames.items():
        if type_relation is not None:
            raw[int(c)] = tuple(sorted({int(a) for r, a in frame if int(r) == type_relation})) or ("none",)
        else:
            raw[int(c)] = tuple(sorted({int(r) for r, _ in frame}))
    counts = Counter(raw.values())
    return {c: (t if counts[t] >= min_members else "other") for c, t in raw.items()}


@dataclass
class VectorUtilityTest:
    """Held-out utility in the store's space: adding edge `e` to concept `c` changes the cosine fit of its static store
    to each held-out observation `y_o` (decoded into the store space) by `Δ_o = cos(z_c + m·T_r(a), y_o) − cos(z_c,
    y_o)` (`z_c` the frame's store, `m` = `mass`). Controls (`control`): `head` — the same edge added to up to
    `controls` same-type concepts with observations whose frames lack it (Welch's t of `Δ_o` against their mean
    utilities); `filler` — `controls` random typed fillers of the same relation on the same concept (paired, one-sample
    t of `Δ_o(e) − mean Δ_o(e')`); `none`."""

    dictionary: Dictionary
    frames: Mapping[int, Sequence[tuple[int, int]]]
    observations: Mapping[int, Tensor]
    control: str = "head"
    controls: int = 8
    types: Mapping[int, Hashable] | None = None
    mass: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")
        self._z: dict[int, Tensor] = {}
        self._y: dict[int, Tensor] = {}
        self._base: dict[int, Tensor] = {}
        self._by_type: dict[Hashable, list[int]] = defaultdict(list)
        for c in sorted(self.observations):
            self._by_type[(self.types or {}).get(c, "all")].append(c)
        self._frame_sets = {int(c): {(int(r), int(a)) for r, a in f} for c, f in self.frames.items()}

    def _prepare(self, c: int) -> tuple[Tensor, Tensor, Tensor]:
        if c not in self._z:
            z = self.dictionary.store(list(self.frames.get(c, ())))
            y = F.normalize(torch.as_tensor(self.observations[c]).float().reshape(-1, self.dictionary.dimension), dim=-1)
            self._z[c], self._y[c] = z, y
            self._base[c] = y @ F.normalize(z, dim=-1) if float(z.norm()) > 0 else torch.zeros(y.shape[0])
        return self._z[c], self._y[c], self._base[c]

    def utilities(self, c: int, vectors: Tensor) -> Tensor:
        """`(m, k)` per-observation utilities of adding each of `vectors` (bound edges) to concept `c`."""
        z, y, base = self._prepare(c)
        after = F.normalize(z[None] + self.mass * vectors, dim=-1) @ y.T
        return after - base[None]

    def _control_heads(self, c: int, relation: int, filler: int) -> list[int]:
        pool = [h for h in self._by_type.get((self.types or {}).get(c, "all"), []) if h != c
                and (relation, filler) not in self._frame_sets.get(h, set())]
        if len(pool) <= self.controls:
            return pool
        rng = np.random.default_rng(_stable(self.seed, c, relation, filler))
        return [pool[i] for i in sorted(rng.choice(len(pool), self.controls, replace=False).tolist())]

    def _control_fillers(self, c: int, relation: int, filler: int) -> list[int]:
        options = self.dictionary.candidates[relation].nonzero().flatten().tolist()
        taken = self._frame_sets.get(c, set())
        pool = [a for a in options if a != filler and (relation, a) not in taken]
        if not pool:
            return []
        rng = np.random.default_rng(_stable(self.seed, "filler", c, relation, filler))
        return [pool[i] for i in rng.choice(len(pool), min(self.controls, len(pool)), replace=False).tolist()]

    def contrasts(self, proposals: Sequence[Mapping[str, Any]]) -> list[Contrast | None]:
        out: list[Contrast | None] = [None] * len(proposals)
        by_concept: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(proposals):
            if int(p["concept"]) in self.observations:
                by_concept[int(p["concept"])].append(i)
        for c, rows in by_concept.items():
            edges = [(int(proposals[i]["relation"]), int(proposals[i]["filler"])) for i in rows]
            vectors = self.dictionary.bound([r for r, _ in edges], [a for _, a in edges])
            delta = self.utilities(c, vectors)
            for j, (i, (r, a)) in enumerate(zip(rows, edges)):
                values = delta[j].numpy().astype(np.float64)
                if self.control == "none":
                    out[i] = Contrast(values)
                elif self.control == "filler":
                    fills = self._control_fillers(c, r, a)
                    if not fills:
                        continue
                    ctrl = self.utilities(c, self.dictionary.bound([r] * len(fills), fills)).mean(0)
                    out[i] = Contrast(values - ctrl.numpy().astype(np.float64))
                else:
                    heads = self._control_heads(c, r, a)
                    if len(heads) < 2:
                        continue
                    vector = vectors[j:j + 1]
                    means = [float(self.utilities(h, vector).mean()) for h in heads]
                    out[i] = Contrast(values, np.asarray(means, dtype=np.float64))
        return out


@dataclass
class LossUtilityTest:
    """Held-out utility as LM loss (the methodology's "loss change on held-out windows"): `losses(frames)` returns, for
    each concept of `frames` (concept → trial frame), the per-window losses after the term on that concept's held-out
    windows (the same windows, in the same order, on every call). Utility per window = loss(current frame) −
    loss(frame + e). Control: `filler` (paired: `controls` random typed fillers of the same relation; the default, since
    a concept's windows are its own) or `none`. Concepts' trial frames are evaluated together, one proposal per concept
    per call (as `authoring.verify_frames`)."""

    losses: Callable[[Mapping[int, list[tuple[int, int]]]], Mapping[int, np.ndarray]]
    frames: Mapping[int, Sequence[tuple[int, int]]]
    candidates: Tensor | None = None
    control: str = "filler"
    controls: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.control not in ("filler", "none"):
            raise ValueError("a loss test controls by filler or not at all")
        if self.control == "filler" and self.candidates is None:
            raise ValueError("the filler control needs typed candidates")

    def _phases(self, edges: list[tuple[int, int, int]]) -> dict[int, np.ndarray]:
        """Per edge index the per-window loss with the edge added (edges of one concept go to separate calls)."""
        out: dict[int, np.ndarray] = {}
        queue: dict[int, list[int]] = defaultdict(list)
        for i, (c, _, _) in enumerate(edges):
            queue[c].append(i)
        phase = 0
        while True:
            batch = {c: rows[phase] for c, rows in queue.items() if phase < len(rows)}
            if not batch:
                break
            trial = {c: list(self.frames.get(c, ())) + [(edges[i][1], edges[i][2])] for c, i in batch.items()}
            result = self.losses(trial)
            for c, i in batch.items():
                out[i] = np.asarray(result[c], dtype=np.float64)
            phase += 1
        return out

    def contrasts(self, proposals: Sequence[Mapping[str, Any]]) -> list[Contrast | None]:
        concepts = sorted({int(p["concept"]) for p in proposals})
        base = self.losses({c: list(self.frames.get(c, ())) for c in concepts})
        edges = [edge_key(p) for p in proposals]
        controls: list[list[int]] = [[] for _ in proposals]
        extra: list[Edge] = []
        if self.control == "filler":
            for i, (c, r, a) in enumerate(edges):
                taken = {(int(x), int(y)) for x, y in self.frames.get(c, ())}
                pool = [b for b in self.candidates[r].nonzero().flatten().tolist() if b != a and (r, b) not in taken]
                rng = np.random.default_rng(_stable(self.seed, "loss", c, r, a))
                for b in (rng.choice(len(pool), min(self.controls, len(pool)), replace=False).tolist() if pool else []):
                    controls[i].append(len(edges) + len(extra))
                    extra.append((c, r, pool[b]))
        after = self._phases(edges + extra)
        out: list[Contrast | None] = []
        for i, (c, _, _) in enumerate(edges):
            if c not in base or i not in after:
                out.append(None); continue
            values = np.asarray(base[c]) - after[i]
            if self.control == "filler":
                if not controls[i]:
                    out.append(None); continue
                values = values - np.mean([np.asarray(base[c]) - after[j] for j in controls[i]], axis=0)
            out.append(Contrast(values))
        return out


# ---------------------------------------------------------------- acceptance


@dataclass
class AcceptanceTest:
    """The M3 test: a utility (`VectorUtilityTest`, `LossUtilityTest`, or anything with `contrasts(proposals)`), the
    level `alpha`, the decision rule `correction` over all proposals of one call, and the minimum number of held-out
    observations a proposal needs to be testable (fewer: rejected).

    Rules: `holm` (one-sided t p-values, Holm; FWER ≤ α if the p-values are valid), `bh` (Benjamini–Hochberg),
    `knockoff` (knockoff+ with `knockoffs` = one matched null proposal per proposal, same order), `decoy` (target–decoy
    FDR ≤ α against `null_statistics` = the t statistics of proposals the same pipeline makes in a null world: valid
    whatever the p-values' calibration, as long as the null world reproduces the false proposals' statistics),
    `holm+decoy` (both must accept), `none`. `decoy_statistic` is the record field the decoy threshold reads (`utility`,
    the mean held-out contrast, by default: a concept-specific misfit of the decoded evidence to the dictionary is
    consistent across observations — a large t — but small; `t` is the alternative)."""

    utility: Any
    alpha: float = 0.05
    correction: str = "holm"
    min_observations: int = 4
    knockoffs: Sequence[Mapping[str, Any]] | None = None
    null_statistics: Sequence[float] | None = None
    decoy_statistic: str = "utility"
    decoy_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.correction not in CORRECTIONS:
            raise ValueError(f"correction must be one of {CORRECTIONS}")
        if "decoy" in self.correction and self.null_statistics is None:
            raise ValueError("a decoy rule needs null_statistics")


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    """Benjamini–Hochberg step-up adjusted p-values (monotone, capped at 1), in the input order."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i], reverse=True)
    adjusted, running = [1.0] * n, 1.0
    for k, index in enumerate(order):
        rank = n - k
        running = min(running, min(1.0, float(p_values[index]) * n / rank))
        adjusted[index] = running
    return adjusted


def knockoff_threshold(w: Sequence[float], q: float) -> float:
    """Knockoff+ threshold (Barber & Candès 2015): the smallest t > 0 with (1 + #{W ≤ −t}) / max(1, #{W ≥ t}) ≤ q
    (inf when none)."""
    values = np.asarray(w, dtype=np.float64)
    for t in np.sort(np.unique(np.abs(values[values != 0]))):
        if (1 + int((values <= -t).sum())) / max(1, int((values >= t).sum())) <= q:
            return float(t)
    return math.inf


def stat_value(record: Mapping[str, Any], statistic: str) -> float:
    """A record's decision statistic (`utility` or `t`); untestable or missing: −inf."""
    value = record.get(statistic)
    return float(value) if value is not None and record.get("testable", True) else -math.inf


def decoy_threshold(target: Sequence[float], decoy: Sequence[float], q: float, *, scale: float = 1.0) -> float:
    """Target–decoy threshold (Elias & Gygi 2007, with knockoff+'s +1): the smallest τ among the target statistics with
    estimated FDR(τ) = scale · (1 + #{decoy ≥ τ}) / #{target ≥ τ} ≤ q; inf when none. `decoy` holds the statistics of
    proposals the same pipeline makes in a null world (every one false). When that world runs the same proposer on the
    same concepts (the runner's complete null), the number of decoys above τ estimates the number of false targets above
    τ directly (`scale` = 1); otherwise `scale` = the target world's opportunities over the decoy world's (N_target /
    N_decoy gives the classical per-proposal normalization)."""
    t = np.sort(np.asarray([v for v in target if math.isfinite(v)], dtype=np.float64))[::-1]
    d = np.sort(np.asarray([v for v in decoy if math.isfinite(v)], dtype=np.float64))
    if not t.size:
        return math.inf
    best = math.inf
    for k, tau in enumerate(t, start=1):                         # k targets at or above tau (ties: the last of a run)
        if k < t.size and t[k] == tau:
            continue
        above = d.size - int(np.searchsorted(d, tau, side="left"))
        if scale * (1 + above) / k <= q:
            best = float(tau)
    return best


def statistics(proposals: Sequence[Mapping[str, Any]], test: AcceptanceTest) -> list[dict[str, Any]]:
    """Every proposal (a copy) with its held-out statistics: `utility`, `n`, `t`, `p`, `testable` (no decision yet)."""
    contrasts = test.utility.contrasts(proposals)
    out = []
    for proposal_, contrast in zip(proposals, contrasts):
        t, pv = one_sided(contrast, min_observations=test.min_observations)
        record = dict(proposal_)
        record.update(utility=None if contrast is None else contrast.mean, n=0 if contrast is None else contrast.n, t=t,
                      p=pv, testable=contrast is not None and contrast.n >= test.min_observations)
        out.append(record)
    return out


def decide(records: Sequence[dict[str, Any]], correction: str, alpha: float, *,
           threshold: float | Callable[[Mapping[str, Any]], float] | None = None,
           knockoff_t: Sequence[float] | None = None, statistic: str = "utility") -> list[dict[str, Any]]:
    """Apply a decision rule to `statistics` records (in place; returned): sets `p_adjusted`, `threshold` (decoy rules)
    and `accepted`. `threshold` is a number or a per-record function (cross-fitted decoy thresholds) on the record
    field `statistic`."""
    p = [float(r["p"]) for r in records]
    adjusted: list[float] = [float("nan")] * len(records)
    if correction in ("holm", "holm+decoy"):
        adjusted = holm_adjust(p) if p else []
    elif correction == "bh":
        adjusted = bh_adjust(p) if p else []
    elif correction == "none":
        adjusted = list(p)
    for i, r in enumerate(records):
        r["p_adjusted"] = adjusted[i]
        ok = adjusted[i] <= alpha if correction in ("holm", "bh", "none", "holm+decoy") else True
        if "decoy" in correction:
            tau = threshold(r) if callable(threshold) else threshold
            r["threshold"] = tau
            ok = ok and tau is not None and stat_value(r, statistic) >= float(tau) and bool(r["testable"])
        r["accepted"] = bool(ok and r["testable"])
    if correction == "knockoff":
        if knockoff_t is None or len(knockoff_t) != len(records):
            raise ValueError("the knockoff correction needs one matched knockoff statistic per proposal")
        clip = lambda v: max(-1e6, min(1e6, float(v)))  # noqa: E731
        w = [clip(r["t"]) - clip(k) for r, k in zip(records, knockoff_t)]
        tau = knockoff_threshold(w, alpha)
        for r, x in zip(records, w):
            r["threshold"] = tau
            r["accepted"] = bool(x >= tau and r["testable"])
    return list(records)


def test_proposals(proposals: Sequence[Mapping[str, Any]], test: AcceptanceTest) -> list[dict[str, Any]]:
    """Every proposal (a copy) with its test record: `utility`, `n`, `t`, `p`, `testable`, `p_adjusted`, `accepted`
    (and `threshold` for the decoy and knockoff rules)."""
    records = statistics(proposals, test)
    knockoff_t = None
    threshold = None
    if test.correction == "knockoff":
        if test.knockoffs is None or len(test.knockoffs) != len(proposals):
            raise ValueError("the knockoff correction needs one matched knockoff proposal per proposal")
        knockoff_t = [r["t"] for r in statistics(test.knockoffs, test)]
    if "decoy" in test.correction:
        threshold = decoy_threshold([stat_value(r, test.decoy_statistic) for r in records if r["testable"]],
                                    test.null_statistics or [], test.alpha, scale=test.decoy_scale)
    return decide(records, test.correction, test.alpha, threshold=threshold, knockoff_t=knockoff_t,
                  statistic=test.decoy_statistic)


def accept(proposals: Sequence[Mapping[str, Any]], test: AcceptanceTest) -> list[dict[str, Any]]:
    """The accepted proposals (copies with the test's fields; `test_proposals` keeps the rejected ones too)."""
    return [r for r in test_proposals(proposals, test) if r["accepted"]]


# ---------------------------------------------------------------- null worlds


def null_proposals(proposals: Sequence[Mapping[str, Any]], kind: str, *, dictionary: Dictionary,
                   frames: Mapping[int, Sequence[tuple[int, int]]], types: Mapping[int, Hashable] | None = None,
                   exclude: Collection[Edge] = frozenset(), seed: int = 0) -> list[dict[str, Any]]:
    """Decoys matched to `proposals`, false by construction (never an edge of `exclude` — the curated ontology — nor of
    the concept's frame), built from the same data:

    - `relabel`: `(c, r, a) → (c, r', a)`, `r' ≠ r` a relation that admits `a` (same filler type; relation labels
      permuted within type);
    - `swap`: `(c, r, a) → (c, r, a')`, `a'` the filler of `r` in the frame of another concept of `c`'s type (fillers
      swapped between terms of the same type; fallback: a random typed filler of `r`).

    A proposal without a valid decoy is dropped; each decoy keeps `null_of` (the original edge)."""
    if kind not in NULL_KINDS:
        raise ValueError(f"kind must be one of {NULL_KINDS}")
    excluded = set(exclude)
    for c, f in frames.items():
        excluded.update((int(c), int(r), int(a)) for r, a in f)
    type_of = lambda c: (types or {}).get(int(c), "all")  # noqa: E731
    # fillers of each relation within each type (the peers' frames), as a list with multiplicity: a draw picks a peer
    # edge uniformly, so frequent fillers are swapped in as often as they occur
    pool: dict[tuple[Hashable, int], list[int]] = defaultdict(list)
    if kind == "swap":
        for c in sorted(frames):
            for r, a in frames[c]:
                pool[(type_of(c), int(r))].append(int(a))
    out = []
    for p in proposals:
        c, r, a = edge_key(p)
        rng = np.random.default_rng(_stable(seed, kind, c, r, a))
        decoy: Edge | None = None
        if kind == "relabel":
            options = [s for s in dictionary.candidates[:, a].nonzero().flatten().tolist() if s != r and (c, s, a) not in excluded]
            if options:
                decoy = (c, int(options[int(rng.integers(len(options)))]), a)
        else:
            for fills in (pool.get((type_of(c), r), []), dictionary.candidates[r].nonzero().flatten().tolist()):
                for _ in range(min(64, 4 * len(fills))):           # rejection sampling: a filler not curated for c
                    b = int(fills[int(rng.integers(len(fills)))])
                    if (c, r, b) not in excluded:
                        decoy = (c, r, b)
                        break
                if decoy is not None:
                    break
        if decoy is not None:
            out.append(make_proposal(*decoy, p.get("score", 0.0), f"null:{kind}", null_of=[c, r, a]))
    return out


def permute_within_type(concepts: Sequence[int], types: Mapping[int, Hashable], *, seed: int = 0) -> dict[int, int]:
    """A derangement within each type (a cyclic shift of a seeded shuffle): concept → the concept whose evidence it reads
    in the pipeline-level null. Types with one member map to themselves (reported by the caller)."""
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for c in concepts:
        groups[types.get(int(c), "all")].append(int(c))
    rng = np.random.default_rng(seed)
    out = {}
    for _, members in sorted(groups.items(), key=lambda kv: repr(kv[0])):
        order = [members[i] for i in rng.permutation(len(members)).tolist()]
        for i, c in enumerate(order):
            out[c] = order[(i + 1) % len(order)]
    return out


def false_acceptance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Null false-acceptance rate: accepted / tested over null-world proposals (every one is false), with its Wilson
    interval; untestable proposals are counted separately (they are rejected, so they cannot inflate the rate)."""
    from .statistics import wilson_interval
    tested = [r for r in records if r.get("testable", True)]
    accepted = sum(bool(r["accepted"]) for r in records)
    n = len(tested)
    low, high = wilson_interval(accepted, n) if n else (float("nan"), float("nan"))
    return {"proposals": len(records), "tested": n, "accepted": int(accepted),
            "rate": accepted / n if n else float("nan"), "rate_all": accepted / len(records) if records else float("nan"),
            "ci_low": low, "ci_high": high}


# ---------------------------------------------------------------- metrics


def edge_metrics(accepted: Iterable[Mapping[str, Any]], gold: Collection[Edge], *,
                 proposals: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Precision / recall / F1 of accepted edges against the gold edges (erased curated edges); with `proposals`, also
    the false-acceptance rate among non-gold proposals and the gold coverage of the proposals (recall ceiling)."""
    gold_set = set(gold)
    acc = {edge_key(a) for a in accepted}
    tp = len(acc & gold_set)
    out: dict[str, Any] = {"accepted": len(acc), "gold": len(gold_set), "tp": tp,
                           "precision": tp / len(acc) if acc else float("nan"),
                           "recall": tp / len(gold_set) if gold_set else float("nan")}
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if acc and gold_set and p + r > 0 else 0.0
    if proposals is not None:
        prop = {edge_key(x) for x in proposals}
        non_gold = prop - gold_set
        out.update(proposed=len(prop), proposal_coverage=len(prop & gold_set) / len(gold_set) if gold_set else float("nan"),
                   false_acceptance_nongold=len(acc - gold_set) / len(non_gold) if non_gold else float("nan"))
    return out


def per_concept_counts(accepted: Iterable[Mapping[str, Any]], gold: Collection[Edge], concepts: Iterable[int]
                       ) -> dict[int, tuple[int, int, int]]:
    """concept → (true positives, false positives, false negatives) of the accepted edges (for cluster bootstraps)."""
    gold_by: dict[int, set[Edge]] = defaultdict(set)
    for e in gold:
        gold_by[e[0]].add(e)
    acc_by: dict[int, set[Edge]] = defaultdict(set)
    for a in accepted:
        acc_by[int(a["concept"])].add(edge_key(a))
    return {int(c): (len(acc_by[c] & gold_by[c]), len(acc_by[c] - gold_by[c]), len(gold_by[c] - acc_by[c]))
            for c in concepts}


def recall_at_precision(scores: Sequence[float], labels: Sequence[bool], precision: float = 0.8) -> float:
    """The largest recall over score thresholds (predict ≥ threshold, ties together) whose precision is ≥ `precision`;
    0 when no threshold reaches it."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    total = int(y.sum())
    if not total or not s.size:
        return float("nan") if not total else 0.0
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    k = np.arange(1, s.size + 1)
    last = np.r_[s[1:] != s[:-1], True]                      # the end of each run of tied scores
    ok = last & (tp / k >= precision - 1e-12)
    return float(tp[ok].max() / total) if ok.any() else 0.0


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann–Whitney AUC with mid-rank ties (vectorized `kg_baselines.auc`)."""
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ranking_metrics(ranks: Sequence[float], ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """MRR and Hits@k of 1-based ranks (NaN ranks = not placed: reciprocal rank 0)."""
    r = np.asarray(ranks, dtype=np.float64)
    rr = np.where(np.isfinite(r), 1.0 / np.maximum(r, 1.0), 0.0)
    out = {"n": int(r.size), "mrr": float(rr.mean()) if r.size else float("nan")}
    for k in ks:
        out[f"hits@{k}"] = float((np.isfinite(r) & (r <= k)).mean()) if r.size else float("nan")
    return out


def gold_ranks(scores: Tensor, gold: Sequence[Collection[int]]) -> np.ndarray:
    """Per row the best (filtered) rank of its gold columns: other gold columns of the row are removed, ties at half
    weight (`cleanup.filtered_ranks`); a row without gold gets NaN."""
    from .cleanup import filtered_ranks
    out = np.full(scores.shape[0], np.nan)
    for i, cols in enumerate(gold):
        cols = sorted({int(c) for c in cols})
        if not cols:
            continue
        row = scores[i:i + 1].float().expand(len(cols), -1)
        exclude = torch.zeros_like(row, dtype=torch.bool)
        exclude[:, cols] = True
        ranks = filtered_ranks(row, torch.tensor(cols), exclude=exclude)
        out[i] = float(ranks.min())
    return out


def placement_scores(vectors: Tensor, relation: int, candidates: Sequence[int], dictionary: Dictionary, *,
                     method: str = "unbind") -> Tensor:
    """Scores of candidate fillers (atomics) of `relation` for new terms whose passive vectors are already in the store
    space (n, d): `unbind` — cosine of the unbound `T_r⁻¹ y` with each candidate atomic (the read tool on a decoded
    vector, a one-edge decomposition); `correlate` — `cos(y, T_r(a))`."""
    cand = torch.as_tensor(list(candidates), dtype=torch.long)
    y = F.normalize(vectors.float(), dim=-1)
    if method == "unbind":
        unbound = dictionary.unbind(torch.full((y.shape[0],), int(relation), dtype=torch.long), y)
        return F.normalize(unbound, dim=-1) @ F.normalize(dictionary.atomics[cand], dim=-1).T
    if method == "correlate":
        bound = F.normalize(dictionary.bound([int(relation)] * cand.numel(), cand), dim=-1)
        return y @ bound.T
    raise ValueError("method must be 'unbind' or 'correlate'")


def erase_edges(frames: Mapping[int, Sequence[tuple[int, int]]], concepts: Iterable[int], *, fraction: float,
                relations: Collection[int] | None = None, seed: int = 0, keep: int = 1
                ) -> tuple[dict[int, list[tuple[int, int]]], set[Edge]]:
    """Erase each edge of `concepts` whose relation is in `relations` (default all) independently with probability
    `fraction`, keeping at least `keep` edges per concept (the last erasures are undone, in a seeded order). Returns the
    erased frames (every concept of `frames`; untouched ones copied) and the erased edges."""
    rng = np.random.default_rng(seed)
    out = {int(c): list(f) for c, f in frames.items()}
    erased: set[Edge] = set()
    for c in sorted(int(x) for x in concepts):
        frame = out.get(c, [])
        eligible = [i for i, (r, _) in enumerate(frame) if relations is None or int(r) in relations]
        draw = rng.random(len(eligible)) < fraction
        chosen = [eligible[i] for i in np.flatnonzero(draw).tolist()]
        while chosen and len(frame) - len(chosen) < keep:
            chosen.pop(int(rng.integers(len(chosen))))
        if chosen:
            gone = set(chosen)
            erased.update((c, int(frame[i][0]), int(frame[i][1])) for i in chosen)
            out[c] = [e for i, e in enumerate(frame) if i not in gone]
    return out, erased
