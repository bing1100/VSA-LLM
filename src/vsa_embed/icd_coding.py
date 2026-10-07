"""Clinical code assignment from notes with structured code vectors: frequency bias and never-trained codes (T1c-F).

HRRBERT (`resources/vsa-paper.md`) composed SNOMED CT HRR embeddings for ICD codes inside a BERT over MIMIC-IV code
sequences and showed (i) less frequency information in the code embeddings (t-SNE coloured by frequency), (ii) better
top-k masked-code accuracy for rare codes in 7 log-frequency bins and (iii) non-zero accuracy on codes never seen in
training. T1c-F asks the same of a *text* task: ICD-9 coding of MIMIC-III discharge summaries with a label-wise
attention head (LAAT / CAML style) on a frozen causal LM, where every label's query, output vector and bias come from a
**code vector** given by the condition — free, composed from SNOMED CT frames, TransE, the code title through the host,
random, or GRAM ancestor attention. The preregistration is `experiments/t1c-clinical/icd-frequency/preregistration.md`.

This module holds the licence-free parts (no file I/O on licensed data; tests use synthetic fixtures):

- frequency: HRRBERT's natural-log relative-frequency bins (`HRRBERT_EDGES`, width 2 from −14 to 0);
- the code-level holdout: hashed, stratified by frequency bin, digest-pinned (`choose_code_holdout`);
- ICD-9-CM hierarchy for GRAM (`icd9_ancestors`; chapter ranges are the public classification's);
- concept reduction for one-to-many ICD→SNOMED maps (`maximal_concepts`) and union frames;
- label sources (`FreeSource`, `FixedSource`, `ComposedSource`, `GramSource`, `SumSource`) and the head
  (`LabelAttentionHead`), joint training of several heads on identical batches (`train_heads`), scoring;
- metrics: per-code AUC (midrank), weighted per-code AUC for the two-way bootstrap (`WeightedAuc`), ranks and top-k;
- statistics: Holm, bootstrap p-values, OLS slope; probes: cross-validated ridge of log frequency, nearest-neighbour
  frequency agreement, an exact t-SNE (`tsne`).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compose import FrameComposer, FrameSchedule

# -- frequency bins ---------------------------------------------------------------------------------------------

#: HRRBERT's bins: natural log of a code's relative frequency, width 2 from −14 to 0 (7 bins).
HRRBERT_EDGES: tuple[float, ...] = (-14.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0)


def log_frequency(counts: Sequence[float] | np.ndarray, total: float) -> np.ndarray:
    """ln(count / total); −inf for a count of 0."""
    counts = np.asarray(counts, dtype=np.float64)
    out = np.full(counts.shape, -np.inf)
    seen = counts > 0
    out[seen] = np.log(counts[seen] / float(total))
    return out


def frequency_bins(logf: Sequence[float] | np.ndarray, edges: Sequence[float] = HRRBERT_EDGES) -> np.ndarray:
    """Bin index `i` with `edges[i] ≤ ln f < edges[i+1]`; below `edges[0]` → 0 (rarest), at or above `edges[-1]` → the
    last bin; −inf (never seen) → −1."""
    logf = np.asarray(logf, dtype=np.float64)
    index = np.searchsorted(np.asarray(edges, dtype=np.float64), logf, side="right") - 1
    index = np.clip(index, 0, len(edges) - 2)
    index[~np.isfinite(logf)] = -1
    return index.astype(np.int64)


def bin_name(index: int, edges: Sequence[float] = HRRBERT_EDGES) -> str:
    """`[lo,hi)`; HRRBERT names a bin by its upper edge (bin 0 = the most common codes, bin −12 the rarest)."""
    return f"[{edges[index]:g},{edges[index + 1]:g})"


# -- the code-level holdout -------------------------------------------------------------------------------------

def hash_key(code: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode()).hexdigest()


def holdout_digest(codes: Iterable[str]) -> str:
    """sha256 of the sorted codes, one per line (the pinned value; the list itself stays with the licensed data)."""
    return hashlib.sha256("\n".join(sorted(codes)).encode()).hexdigest()


def choose_code_holdout(codes: Sequence[str], counts: Mapping[str, int], total: float, *, eligible_bins: Sequence[float],
                        fraction: float, min_count: int, salt: str, edges: Sequence[float] = HRRBERT_EDGES) -> dict[str, Any]:
    """A frequency-stratified, hashed code holdout: within every eligible bin (given by its lower edge), the codes with
    at least `min_count` training occurrences are ranked by `sha256(salt:code)` and the first `round(fraction · n)`
    are held out. Deterministic, label-free beyond the training counts, and independent of any model."""
    codes = sorted(set(codes))
    bins = frequency_bins(log_frequency([counts.get(c, 0) for c in codes], total), edges)
    wanted = {float(e) for e in eligible_bins}
    chosen: list[str] = []
    per_bin: dict[str, dict[str, int]] = {}
    for index in range(len(edges) - 1):
        if float(edges[index]) not in wanted:
            continue
        pool = sorted((c for c, b in zip(codes, bins) if b == index and counts.get(c, 0) >= min_count),
                      key=lambda c: hash_key(c, salt))
        take = int(round(fraction * len(pool)))
        chosen += pool[:take]
        per_bin[bin_name(index, edges)] = {"eligible": len(pool), "chosen": take,
                                           "positives_removed": int(sum(counts.get(c, 0) for c in pool[:take]))}
    chosen = sorted(set(chosen))
    return {"codes": chosen, "per_bin": per_bin, "sha256": holdout_digest(chosen), "fraction": fraction,
            "min_count": min_count, "eligible_bins": sorted(wanted), "salt": salt}


# -- ICD-9-CM hierarchy (GRAM) ----------------------------------------------------------------------------------

#: ICD-9-CM chapters 1–17 by three-digit range (public classification structure); V and E codes are chapters V / E.
ICD9_CHAPTER_RANGES: tuple[tuple[int, int], ...] = (
    (1, 139), (140, 239), (240, 279), (280, 289), (290, 319), (320, 389), (390, 459), (460, 519), (520, 579),
    (580, 629), (630, 679), (680, 709), (710, 739), (740, 759), (760, 779), (780, 799), (800, 999))


def icd9_chapter(code: str) -> str:
    code = code.strip().upper()
    if code.startswith(("V", "E")):
        return f"icd9:chapter:{code[0]}"
    try:
        head = int(code[:3])
    except ValueError:
        return "icd9:chapter:unknown"
    for number, (lo, hi) in enumerate(ICD9_CHAPTER_RANGES, start=1):
        if lo <= head <= hi:
            return f"icd9:chapter:{number:02d}"
    return "icd9:chapter:unknown"


def icd9_ancestors(code: str) -> list[str]:
    """Ancestors of an ICD-9-CM diagnosis code written without the dot (MIMIC-III's form), general → specific, without
    the code itself: chapter, three-character category (four for E codes) and, for five-character V / numeric codes,
    the four-character subcategory."""
    code = code.strip().upper()
    chain = [icd9_chapter(code)]
    category = 4 if code.startswith("E") else 3
    if len(code) > category:
        chain.append(f"icd9:{code[:category]}")
    if not code.startswith("E") and len(code) > 4:
        chain.append(f"icd9:{code[:4]}")
    return chain


# -- concepts and frames ----------------------------------------------------------------------------------------

def maximal_concepts(concepts: Iterable[str], ancestors: Callable[[str], set[str]], cap: int) -> list[str]:
    """The concepts of a one-to-many map that have no ancestor in the mapped set (the most general mapped concepts),
    ordered by how many of the set they subsume (descending; ties by identifier), at most `cap`."""
    members = set(concepts)
    inside = {c: ancestors(c) & members for c in members}
    maximal = [c for c in members if not inside[c]]
    covered = {m: 1 + sum(1 for c in members if m in inside[c]) for m in maximal}
    maximal.sort(key=lambda m: (-covered[m], int(m) if m.isdigit() else 0, m))
    return maximal[:cap]


def union_frame(frames: Iterable[Iterable[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Union of frames, first occurrence order, duplicates removed (as `AliasTable.entry_schedule`)."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for frame in frames:
        for edge in frame:
            edge = (int(edge[0]), int(edge[1]))
            if edge not in seen:
                seen.add(edge); out.append(edge)
    return out


# -- label sources ----------------------------------------------------------------------------------------------

class LabelSource(nn.Module):
    """Code vectors for label ids (one row per id). `embedding_parameters` names parameters kept out of weight decay
    (tables whose rows a label never trained must stay at their initialization)."""

    dimension: int

    def forward(self, ids: Tensor) -> Tensor:          # pragma: no cover - interface
        raise NotImplementedError


class FreeSource(LabelSource):
    """A free learned vector per code (the unstructured control; BERT / HRRBERT initialization N(0, 0.02²)). Rows of
    codes outside the training label set receive no gradient and keep their initialization."""

    def __init__(self, count: int, dimension: int = 256, *, std: float = 0.02, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(count, dimension, generator=generator) * std)
        self.dimension = dimension

    def forward(self, ids: Tensor) -> Tensor:
        return self.weight[ids]


class FixedSource(LabelSource):
    """Frozen vectors (TransE, code title through the host, random, a trained composer's output)."""

    def __init__(self, vectors: Tensor) -> None:
        super().__init__()
        self.register_buffer("vectors", torch.as_tensor(vectors, dtype=torch.float32).clone())
        self.dimension = int(self.vectors.shape[1])

    def forward(self, ids: Tensor) -> Tensor:
        return self.vectors[ids]


def random_vectors(count: int, dimension: int = 256, *, seed: int = 0) -> Tensor:
    """Fixed random unit vectors (the random control)."""
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(count, dimension, generator=generator), dim=-1)


def standardize_vectors(vectors: Tensor, reference: Sequence[int] | Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """Per-dimension z-score over the reference rows (the trained labels), then unit L2 rows (as `row_sources`)."""
    vectors = torch.as_tensor(vectors, dtype=torch.float32)
    ref = vectors if reference is None else vectors[torch.as_tensor(reference, dtype=torch.long)]
    mean, std = ref.mean(0), ref.std(0).clamp_min(eps)
    return F.normalize((vectors - mean) / std, dim=-1)


class ComposedSource(LabelSource):
    """Code vectors composed from SNOMED CT frames by a `FrameComposer` (the C5 channel's composer: HRR binding,
    attentive bundling with an induced concept factor, so a code never trained gets its vector from the same rule).

    `composer=None` builds a fresh composer trained only by the coding loss (`composed_head`); a trained composer (the
    T1c C5 run's dictionary) receives the code frames by `add_concepts` and is used frozen (`composed_c5`)."""

    def __init__(self, frames: Sequence[Sequence[tuple[int, int]]], *, atomic_count: int, relation_count: int,
                 dimension: int = 256, operator: str = "hrr", mode: str = "attentive", concept_factor: str = "induced",
                 key_dimension: int = 8, composer: FrameComposer | None = None, trainable: bool = True, seed: int = 0) -> None:
        super().__init__()
        if any(len(f) == 0 for f in frames):
            raise ValueError("every code needs a non-empty frame")
        if composer is None:
            torch.manual_seed(seed)
            composer = FrameComposer(FrameSchedule.from_frames(frames), atomic_count, relation_count, dimension,
                                     operator=operator, mode=mode, concept_factor=concept_factor, key_dimension=key_dimension)
            self.offset = 0
        else:
            self.offset = int(composer.add_concepts(frames)[0])
        self.composer = composer
        self.trainable = trainable
        self.dimension = int(composer.atomics.shape[1])
        if not trainable:
            composer.requires_grad_(False)
            with torch.no_grad():
                ids = torch.arange(self.offset, self.offset + len(frames))
                cached = torch.cat([composer.compose(part) for part in ids.split(4096)])
            self.register_buffer("cached", cached)

    def forward(self, ids: Tensor) -> Tensor:
        if not self.trainable:
            return self.cached[ids]
        return self.composer.compose(ids + self.offset)


class GramSource(LabelSource):
    """GRAM (Choi et al., KDD 2017): a code's vector is an attention-weighted sum of the free embeddings of the code
    and its ancestors, `g_i = Σ_j α_ij e_j`, `α_ij = softmax_j(uᵀ tanh(W [e_i; e_j] + b))`. A never-trained code keeps
    its own initial embedding but borrows its trained ancestors'."""

    def __init__(self, paths: Sequence[Sequence[int]], node_count: int, dimension: int = 256, *, attention_dim: int = 128,
                 std: float = 0.02, seed: int = 0) -> None:
        super().__init__()
        width = max(len(p) for p in paths)
        index = torch.full((len(paths), width), -1, dtype=torch.long)
        for row, path in enumerate(paths):
            if not path:
                raise ValueError("every code needs at least itself in its GRAM path")
            index[row, :len(path)] = torch.as_tensor(path, dtype=torch.long)
        self.register_buffer("paths", index)
        generator = torch.Generator().manual_seed(seed)
        self.nodes = nn.Parameter(torch.randn(node_count, dimension, generator=generator) * std)
        self.attention = nn.Linear(2 * dimension, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        self.dimension = dimension

    def forward(self, ids: Tensor) -> Tensor:
        paths = self.paths[ids]                                   # (L, m); column 0 = the code itself
        mask = paths >= 0
        e = self.nodes[paths.clamp_min(0)]                        # (L, m, d)
        leaf = e[:, :1].expand_as(e)
        logits = self.score(torch.tanh(self.attention(torch.cat([leaf, e], -1)))).squeeze(-1)
        weights = logits.masked_fill(~mask, float("-inf")).softmax(-1)
        return (weights.unsqueeze(-1) * e).sum(1)


class SumSource(LabelSource):
    """Composed + free (HRRBERT's HRRAdd): the free part of a never-trained code stays at its small initialization."""

    def __init__(self, first: LabelSource, second: LabelSource) -> None:
        super().__init__()
        if first.dimension != second.dimension:
            raise ValueError("summed sources need one dimension")
        self.first, self.second, self.dimension = first, second, first.dimension

    def forward(self, ids: Tensor) -> Tensor:
        return self.first(ids) + self.second(ids)


# -- the head ---------------------------------------------------------------------------------------------------

class LabelAttentionHead(nn.Module):
    """Label-wise attention over pooled note segments (LAAT / CAML style) whose label parameters come from code vectors.

    Documents: `Z = tanh(W_d LN(H))` over segment states `H` (frozen host). Labels: `[q_l; o_l; b_l] = MLP(LN(u_l))` with
    `u_l` the condition's code vector. Logit: `s_l = Σ_i softmax_i(⟨Z_i, q_l⟩ / √a) ⟨Z_i, o_l⟩ + b_l + b_0`. Every label
    parameter, the per-label bias included, is a function of the code vector, so a never-trained code is scored by the
    same rule as a trained one and frequency can enter only through the code vector."""

    def __init__(self, source: LabelSource, input_dim: int, *, attention_dim: int = 256, hidden: int = 512,
                 prior: float = 0.002) -> None:
        super().__init__()
        self.source = source
        self.doc_norm = nn.LayerNorm(input_dim)
        self.doc_proj = nn.Linear(input_dim, attention_dim)
        self.label_norm = nn.LayerNorm(source.dimension, elementwise_affine=False)
        self.label_mlp = nn.Sequential(nn.Linear(source.dimension, hidden), nn.GELU(), nn.Linear(hidden, 2 * attention_dim + 1))
        prior = min(max(prior, 1e-6), 0.5)
        self.bias = nn.Parameter(torch.tensor(math.log(prior / (1 - prior))))
        self.attention_dim = attention_dim

    def label_parameters(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.label_mlp(self.label_norm(self.source(ids)))
        a = self.attention_dim
        return out[:, :a], out[:, a:2 * a], out[:, 2 * a]

    def documents(self, states: Tensor) -> Tensor:
        return torch.tanh(self.doc_proj(self.doc_norm(states)))

    def logits_from(self, z: Tensor, mask: Tensor, q: Tensor, o: Tensor, b: Tensor) -> Tensor:
        scores = torch.einsum("bna,la->bln", z, q) / math.sqrt(self.attention_dim)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        values = torch.einsum("bna,la->bln", z, o)
        return (scores.softmax(-1) * values).sum(-1) + b + self.bias

    def forward(self, states: Tensor, mask: Tensor, ids: Tensor) -> Tensor:
        q, o, b = self.label_parameters(ids)
        return self.logits_from(self.documents(states), mask, q, o, b)

    def parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Weight decay on the document projection and the label MLP's matrices only; none on label-source parameters
        (tables, composer, GRAM nodes), biases and norms."""
        decay, plain = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 and not name.startswith("source.") else plain).append(parameter)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}]


# -- batches over pooled segment states --------------------------------------------------------------------------

@dataclass
class SegmentStore:
    """Segment-pooled host states of every admission: `segments[offsets[a]:offsets[a+1]]` (float16 memmap)."""

    segments: np.ndarray
    offsets: np.ndarray

    @property
    def width(self) -> int:
        return int(self.segments.shape[1])

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def batch(self, admissions: Sequence[int], device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        lengths = [int(self.offsets[a + 1] - self.offsets[a]) for a in admissions]
        n = max(1, max(lengths))
        states = np.zeros((len(admissions), n, self.width), dtype=np.float32)
        mask = np.zeros((len(admissions), n), dtype=bool)
        for row, (a, length) in enumerate(zip(admissions, lengths)):
            if length:
                states[row, :length] = self.segments[self.offsets[a]:self.offsets[a + 1]]
                mask[row, :length] = True
            else:                                   # an empty document attends to one zero segment
                mask[row, 0] = True
        return torch.from_numpy(states).to(device, non_blocking=True), torch.from_numpy(mask).to(device)


def length_batches(lengths: np.ndarray, indices: np.ndarray, batch: int, rng: np.random.Generator, *,
                   pool: int = 50) -> list[np.ndarray]:
    """Shuffled batches of similar length: shuffle, sort pools of `pool · batch` by length, cut, shuffle batches."""
    order = rng.permutation(indices)
    batches = []
    for start in range(0, order.size, pool * batch):
        chunk = order[start:start + pool * batch]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch] for i in range(0, chunk.size, batch)]
    rng.shuffle(batches)
    return batches


def target_matrix(labels: Sequence[np.ndarray], admissions: Sequence[int], column: np.ndarray, width: int) -> Tensor:
    """Multi-hot targets: `labels[a]` are label ids; `column[label]` its position among the scored labels (−1: absent)."""
    out = torch.zeros(len(admissions), width)
    for row, a in enumerate(admissions):
        cols = column[labels[a]]
        cols = cols[cols >= 0]
        if cols.size:
            out[row, torch.as_tensor(cols, dtype=torch.long)] = 1.0
    return out


def train_heads(heads: Mapping[str, LabelAttentionHead], store: SegmentStore, labels: Sequence[np.ndarray], *,
                train_admissions: np.ndarray, dev_admissions: np.ndarray, train_labels: np.ndarray, label_count: int,
                epochs: int = 12, patience: int = 2, batch: int = 32, lr: float = 1e-3, weight_decay: float = 0.01,
                clip: float = 1.0, seed: int = 0, device: torch.device | str = "cpu", autocast: bool = True,
                max_steps: int | None = None, log: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train several heads on **identical batches** (the batch order is a function of `seed` only, so conditions are
    paired), each with its own AdamW; BCE over the training labels; early stopping per head on the dev loss (patience
    in epochs); the best state of each head is restored. Returns per-head histories."""
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    column = np.full(label_count, -1, dtype=np.int64)
    column[train_labels] = np.arange(train_labels.size)
    ids = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    lengths = store.lengths()
    use_amp = autocast and device.type == "cuda"
    optimizers = {name: torch.optim.AdamW(head.parameter_groups(weight_decay), lr=lr) for name, head in heads.items()}
    state = {name: {"best": math.inf, "best_epoch": -1, "bad": 0, "stopped": False, "history": [],
                    "best_state": None} for name in heads}
    for head in heads.values():
        head.to(device)
    steps = 0
    started = time.monotonic()
    for epoch in range(epochs):
        active = [n for n in heads if not state[n]["stopped"]]
        if not active:
            break
        sums = {n: 0.0 for n in active}
        batches = length_batches(lengths, train_admissions, batch, rng)
        for chunk in batches:
            states, mask = store.batch(chunk.tolist(), device)
            targets = target_matrix(labels, chunk.tolist(), column, train_labels.size).to(device)
            for name in active:
                head = heads[name]
                head.train()
                q, o, b = head.label_parameters(ids)            # label side in float32 (HRR binding uses the FFT)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = head.logits_from(head.documents(states), mask, q, o, b)
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_([p for g in optimizers[name].param_groups for p in g["params"]], clip)
                optimizers[name].step()
                sums[name] += float(loss.detach())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        dev = evaluate_loss({n: heads[n] for n in active}, store, labels, dev_admissions, train_labels, column,
                            batch=batch, device=device, autocast=use_amp)
        for name in active:
            record = state[name]
            entry = {"epoch": epoch + 1, "train_loss": sums[name] / max(1, len(batches)), "dev_loss": dev[name],
                     "seconds": round(time.monotonic() - started, 1)}
            record["history"].append(entry)
            if dev[name] < record["best"] - 1e-6:
                record.update(best=dev[name], best_epoch=epoch + 1, bad=0,
                              best_state={k: v.detach().to("cpu", copy=True) for k, v in heads[name].state_dict().items()})
            else:
                record["bad"] += 1
                if record["bad"] >= patience:
                    record["stopped"] = True
            if log:
                log({"head": name, **entry, "best_epoch": record["best_epoch"]})
        if max_steps is not None and steps >= max_steps:
            break
    out = {}
    for name, head in heads.items():
        record = state[name]
        if record["best_state"] is not None:
            head.load_state_dict(record["best_state"])
        out[name] = {"best_dev_loss": record["best"], "best_epoch": record["best_epoch"], "epochs": len(record["history"]),
                     "history": record["history"]}
    return out


@torch.no_grad()
def evaluate_loss(heads: Mapping[str, LabelAttentionHead], store: SegmentStore, labels: Sequence[np.ndarray],
                  admissions: np.ndarray, train_labels: np.ndarray, column: np.ndarray, *, batch: int = 32,
                  device: torch.device | str = "cpu", autocast: bool = False) -> dict[str, float]:
    """Mean BCE over `admissions` × training labels, per head."""
    device = torch.device(device)
    ids = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    totals = {n: 0.0 for n in heads}
    count = 0
    lengths = store.lengths()
    order = admissions[np.argsort(lengths[admissions], kind="stable")]
    params = {}
    for name, head in heads.items():
        head.eval()
        params[name] = head.label_parameters(ids)
    for start in range(0, order.size, batch):
        chunk = order[start:start + batch].tolist()
        states, mask = store.batch(chunk, device)
        targets = target_matrix(labels, chunk, column, train_labels.size).to(device)
        for name, head in heads.items():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
                logits = head.logits_from(head.documents(states), mask, *params[name])
            totals[name] += float(F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="sum"))
        count += len(chunk) * train_labels.size
    return {n: t / max(1, count) for n, t in totals.items()}


@torch.no_grad()
def score_admissions(head: LabelAttentionHead, store: SegmentStore, admissions: np.ndarray, label_ids: np.ndarray, *,
                     batch: int = 32, device: torch.device | str = "cpu", autocast: bool = False,
                     reduce: Callable[[np.ndarray, Tensor], None] | None = None) -> np.ndarray | None:
    """Logits (float32) of `admissions` × `label_ids`, in the given admission order; with `reduce(chunk, logits)` the
    logits of each batch are handed over instead of being collected (for admission sets too large to keep)."""
    device = torch.device(device)
    head.eval().to(device)
    ids = torch.as_tensor(label_ids, dtype=torch.long, device=device)
    q, o, b = head.label_parameters(ids)
    out = None if reduce else np.zeros((admissions.size, label_ids.size), dtype=np.float32)
    for start in range(0, admissions.size, batch):
        chunk = admissions[start:start + batch]
        states, mask = store.batch(chunk.tolist(), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
            logits = head.logits_from(head.documents(states), mask, q, o, b)
        logits = logits.float()
        if reduce:
            reduce(chunk, logits)
        else:
            out[start:start + chunk.size] = logits.cpu().numpy()
    return out


# -- metrics ----------------------------------------------------------------------------------------------------

def auc_columns(scores: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """Per-column ROC AUC (midranks for ties); NaN where a column has no positive or no negative."""
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    ranks = rankdata(scores, axis=0)
    npos = positives.sum(0).astype(np.float64)
    nneg = scores.shape[0] - npos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks * positives).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc


class WeightedAuc:
    """Per-column AUC under admission weights (the two-way bootstrap's admission resample), exact with ties counted
    half. The sort order of every column is computed once; a replicate costs one gather and one cumulative sum."""

    def __init__(self, scores: np.ndarray, positives: np.ndarray, device: torch.device | str = "cpu") -> None:
        device = torch.device(device)
        s = torch.as_tensor(np.asarray(scores, dtype=np.float32), device=device)
        y = torch.as_tensor(np.asarray(positives, dtype=bool), device=device)
        n, c = s.shape
        order = torch.argsort(s, dim=0, stable=True)
        s_sorted = torch.take_along_dim(s, order, 0)
        y_sorted = torch.take_along_dim(y, order, 0)
        positions = torch.arange(n, device=device)[:, None].expand(n, c)
        new = torch.ones(n, c, dtype=torch.bool, device=device)
        new[1:] = s_sorted[1:] != s_sorted[:-1]                    # first of a tie group
        first = torch.where(new, positions, torch.zeros_like(positions)).cummax(0).values
        last_marker = torch.ones(n, c, dtype=torch.bool, device=device)
        last_marker[:-1] = new[1:]                                  # last of a tie group
        last = torch.where(last_marker, positions, torch.full_like(positions, n)).flip(0).cummin(0).values.flip(0)
        pos_rows, pos_cols = torch.nonzero(y_sorted, as_tuple=True)
        self.order, self.negative = order, (~y_sorted).float()
        self.pos_cols = pos_cols
        self.pos_first = first[pos_rows, pos_cols]
        self.pos_last = last[pos_rows, pos_cols] + 1
        self.pos_admission = order[pos_rows, pos_cols]
        self.shape = (n, c)
        self.device = device

    def __call__(self, weights: np.ndarray | Tensor | None = None) -> Tensor:
        n, c = self.shape
        w = (torch.ones(n, device=self.device) if weights is None
             else torch.as_tensor(weights, dtype=torch.float32, device=self.device))
        negative = w[self.order] * self.negative
        cum = torch.zeros(n + 1, c, device=self.device)
        cum[1:] = negative.cumsum(0)
        below = cum[self.pos_first, self.pos_cols]
        equal = cum[self.pos_last, self.pos_cols] - below
        wp = w[self.pos_admission]
        numerator = torch.zeros(c, device=self.device).index_add_(0, self.pos_cols, wp * (below + 0.5 * equal))
        positive_weight = torch.zeros(c, device=self.device).index_add_(0, self.pos_cols, wp)
        negative_weight = cum[n]
        auc = numerator / (positive_weight * negative_weight)
        return torch.where((positive_weight > 0) & (negative_weight > 0), auc, torch.full_like(auc, float("nan")))


def positive_ranks(scores: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """1-based rank of `scores[r, c]` within row `r` (number of strictly higher scores + 1) for each positive pair."""
    out = np.empty(rows.size, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    rows_sorted, cols_sorted = rows[order], cols[order]
    bounds = np.flatnonzero(np.diff(rows_sorted)) + 1
    for part_rows, part_cols, part_index in zip(np.split(rows_sorted, bounds), np.split(cols_sorted, bounds), np.split(order, bounds)):
        if part_rows.size == 0:
            continue
        row = scores[part_rows[0]]
        out[part_index] = (row[None, :] > row[part_cols][:, None]).sum(1) + 1
    return out


def ranks_in_rows(logits: Tensor, rows: Tensor, cols: Tensor, subset: Tensor | None = None) -> Tensor:
    """Torch version on one batch: rank of `logits[rows, cols]` among the row's labels (or among `subset` columns)."""
    values = logits[rows, cols]
    pool = logits[rows] if subset is None else logits[rows][:, subset]
    return (pool > values[:, None]).sum(1) + 1


def topk_by_code(ranks: np.ndarray, codes: np.ndarray, code_count: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per code: hit rate of rank ≤ k over its positive pairs (NaN without positives) and its number of pairs."""
    hits = np.bincount(codes, weights=(ranks <= k).astype(np.float64), minlength=code_count)
    pairs = np.bincount(codes, minlength=code_count).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = hits / pairs
    rate[pairs == 0] = np.nan
    return rate, pairs


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan")
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def ols_slopes(x: Tensor, y: Tensor) -> Tensor:
    """Row-wise OLS slopes of `y` on `x` (both (R, n); NaN entries ignored per row)."""
    keep = torch.isfinite(x) & torch.isfinite(y)
    w = keep.float()
    count = w.sum(1).clamp_min(1)
    x0, y0 = torch.where(keep, x, torch.zeros_like(x)), torch.where(keep, y, torch.zeros_like(y))
    mx, my = x0.sum(1) / count, y0.sum(1) / count
    xc = (x0 - mx[:, None]) * w
    return (xc * (y0 - my[:, None])).sum(1) / (xc * xc).sum(1).clamp_min(1e-12)


# -- statistics -------------------------------------------------------------------------------------------------

def bootstrap_pvalue(deltas: np.ndarray) -> float:
    """Two-sided bootstrap p-value of a difference (share of replicates on the far side of 0, doubled; ≥ 1/B)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return float("nan")
    p = 2 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return float(min(1.0, max(p, 1.0 / deltas.size)))


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (NaN entries stay NaN and do not count)."""
    items = sorted(((p, k) for k, p in pvalues.items() if p == p), key=lambda t: t[0])
    m = len(items)
    adjusted: dict[str, float] = {k: float("nan") for k in pvalues}
    running = 0.0
    for rank, (p, key) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def dunnett(control: Sequence[float], treatments: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Two-sided Dunnett test of each treatment against the control over seeds (HRRBERT's per-bin test); NaN when a
    group has fewer than two finite values or no variance anywhere."""
    from scipy.stats import dunnett as scipy_dunnett
    names = [n for n, v in treatments.items() if np.isfinite(np.asarray(v, dtype=float)).sum() >= 2]
    ctrl = np.asarray([v for v in control if np.isfinite(v)], dtype=float)
    if ctrl.size < 2 or not names:
        return {n: float("nan") for n in treatments}
    samples = [np.asarray([v for v in treatments[n] if np.isfinite(v)], dtype=float) for n in names]
    if all(np.ptp(s) == 0 for s in samples) and np.ptp(ctrl) == 0:
        return {n: float("nan") for n in treatments}
    result = scipy_dunnett(*samples, control=ctrl)
    out = {n: float("nan") for n in treatments}
    out.update({n: float(p) for n, p in zip(names, result.pvalue)})
    return out


# -- probes of frequency information ----------------------------------------------------------------------------

def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).statistic)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean, std = x.mean(0), x.std(0) + 1e-8
    z = (x - mean) / std
    ym = y.mean()
    w = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - ym))
    return w, ym, mean, std


def _ridge_predict(model: tuple[np.ndarray, float, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    w, ym, mean, std = model
    return ((x - mean) / std) @ w + ym


def cv_ridge(x: np.ndarray, y: np.ndarray, *, folds: int = 5, alphas: Sequence[float] = (0.1, 1, 10, 100, 1000, 10000),
             seed: int = 0) -> dict[str, Any]:
    """How decodable `y` (log frequency) is from `x` (code vectors): `folds`-fold cross-validated ridge regression with
    the penalty chosen by an inner 4-fold CV on each training part; out-of-fold R² and Spearman ρ."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(y.size) % folds)
    predicted = np.zeros_like(y)
    chosen = []
    for f in range(folds):
        train, test = fold != f, fold == f
        inner = rng.permutation(np.arange(train.sum()) % 4)
        xt, yt = x[train], y[train]
        errors = []
        for alpha in alphas:
            err = 0.0
            for g in range(4):
                model = _ridge_fit(xt[inner != g], yt[inner != g], alpha)
                err += float(((_ridge_predict(model, xt[inner == g]) - yt[inner == g]) ** 2).sum())
            errors.append(err)
        alpha = alphas[int(np.argmin(errors))]
        chosen.append(alpha)
        predicted[test] = _ridge_predict(_ridge_fit(xt, yt, alpha), x[test])
    r2 = 1.0 - float(((predicted - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"r2": r2, "spearman": spearman(predicted, y), "alphas": chosen, "n": int(y.size), "dimension": int(x.shape[1])}


def neighbour_agreement(x: np.ndarray, y: np.ndarray, *, k: int = 10, device: torch.device | str = "cpu") -> float:
    """Spearman ρ between each code's log frequency and the mean log frequency of its `k` cosine nearest neighbours
    (HRRBERT's t-SNE observation as a number: high when codes cluster by frequency)."""
    v = F.normalize(torch.as_tensor(np.asarray(x, dtype=np.float32), device=device), dim=-1)
    target = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    means = []
    for part in torch.arange(v.shape[0], device=device).split(2048):
        sim = v[part] @ v.T
        sim[torch.arange(part.numel(), device=device), part] = -float("inf")
        means.append(target[sim.topk(k, dim=1).indices].mean(1))
    return spearman(torch.cat(means).cpu().numpy(), np.asarray(y, dtype=float))


def tsne(x: np.ndarray, *, perplexity: float = 30.0, iterations: int = 750, exaggeration: float = 12.0,
         exaggeration_iterations: int = 250, seed: int = 0, device: torch.device | str = "cpu") -> np.ndarray:
    """Exact t-SNE (van der Maaten & Hinton 2008; perplexity-calibrated affinities, early exaggeration, momentum and
    gains, learning rate max(n / exaggeration / 4, 50)) for a few thousand points."""
    device = torch.device(device)
    data = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    n = data.shape[0]
    d2 = torch.cdist(data, data).pow(2)
    d2 = d2 / d2[d2 > 0].mean().clamp_min(1e-12)
    target = math.log(perplexity)
    beta = torch.ones(n, device=device)
    lo, hi = torch.zeros(n, device=device), torch.full((n,), float("inf"), device=device)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    for _ in range(64):
        p = torch.exp(-d2 * beta[:, None]).masked_fill(eye, 0.0)
        total = p.sum(1).clamp_min(1e-12)
        entropy = torch.log(total) + beta * (d2 * p).sum(1) / total
        high = entropy > target
        lo = torch.where(high, beta, lo)
        hi = torch.where(high, hi, beta)
        beta = torch.where(torch.isinf(hi), beta * 2, (lo + hi) / 2)
    p = p / total[:, None]
    p = ((p + p.T) / (2 * n)).clamp_min(1e-12)
    generator = torch.Generator().manual_seed(seed)
    y = (torch.randn(n, 2, generator=generator) * 1e-4).to(device)
    velocity = torch.zeros_like(y)
    gains = torch.ones_like(y)
    lr = max(n / exaggeration / 4, 50.0)
    for it in range(iterations):
        factor = exaggeration if it < exaggeration_iterations else 1.0
        num = 1.0 / (1.0 + torch.cdist(y, y).pow(2))
        num = num.masked_fill(eye, 0.0)
        q = (num / num.sum()).clamp_min(1e-12)
        w = (factor * p - q) * num
        grad = 4.0 * (w.sum(1, keepdim=True) * y - w @ y)
        momentum = 0.5 if it < exaggeration_iterations else 0.8
        same = (grad > 0) == (velocity > 0)
        gains = torch.where(same, gains * 0.8, gains + 0.2).clamp_min(0.01)
        velocity = momentum * velocity - lr * gains * grad
        y = y + velocity
        y = y - y.mean(0)
    return y.cpu().numpy()
