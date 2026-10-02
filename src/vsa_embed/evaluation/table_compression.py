"""Compressed embedding tables at exact byte counts (E4.5; formulation §4.4).

A `CompressedTable` stores `n` rows of width `d` (here: the input-embedding rows of linked
single-token concepts, in a fixed order) and reconstructs any of them with `rows(index)`. Every
method

- fits the trained rows by reconstruction (`fit(target, steps=…, lr=…)`: a closed-form or
  heuristic initialization, then optionally Adam on the (weighted) squared error),
- reports its exact storage (`storage()`: bytes of each stored array; `nbytes` their sum), and
- can be sized to a byte budget (`for_budget(budget, …)`: the configuration with the most bytes
  that still fits, or None).

Parameters are float latents; `rows()` applies their storage format (FP16 rounding, or group-wise
round-to-nearest codes from `quantization.group_fake_quantize`) with a straight-through gradient,
so one module serves reconstruction, evaluation and a short LM finetune of the table alone.

Methods (`METHODS`):

- `int` — the rows themselves, group-wise RTN at `bits` with clip search (the PTQ table). With
  `high_rows > 0` the first `high_rows` rows of a priority order (frequency) get `high_bits`, and a
  1-bit-per-row precision map is stored: `for_budget` uses this to fill any budget exactly
  (mixed-precision PTQ).
- `albert` — factorized `U V`, `U` n×k, `V` k×d (ALBERT); truncated-SVD fit.
- `qr` — quotient–remainder embeddings (Shi et al. 2020): `Q[i // m] ⊙ R[i mod m]` (or `+`).
- `hashing` — the hashing trick (Weinberger et al. 2009): `s(v)·H[h(v)]` with universal hashes of
  the token id, least-squares bucket fit.
- `hash_embedding` — hash embeddings (Svenstrup et al. 2017): `Σ_j p_{i,j} H[h_j(v)]` with `k` hashes
  and per-row importance weights.
- `tt` — tensor-train table (TT-Rec / Hrinchuk et al.): three cores over factorized row and column
  indices, TT-SVD fit.
- `composed` — the ontology composition `P c_{e(i)} + b + δ_i` (formulation §4.4): a `FrameComposer`
  dictionary over the frames of the rows' entries, a projector `P`, and an optional residual `δ`
  stored at `delta_bits`. Bytes = dictionary + projector + frame schedule + row → entry map + `δ`.

`ReplacedRowsEmbedding` wraps a model's input embedding so the rows listed in `token_ids` come from
a compressed table while every other row — and the output head, even when tied — is unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..compose import FrameComposer, FrameSchedule
from .quantization import group_fake_quantize, group_quantized_bytes, index_bytes

_PRIME = 2_147_483_647      # universal hashing modulus (2^31 − 1)


def stored(x: Tensor, bits: int, group_size: int | None = None, *, clip_search: bool = False) -> Tensor:
    """`x` in its storage format, straight-through (identity gradient)."""
    return group_fake_quantize(x, bits, group_size, clip_search=clip_search, straight_through=True)


def stored_bytes(shape: tuple[int, ...] | torch.Size, bits: int, group_size: int | None = None) -> int:
    """Exact bytes of a tensor of `shape` stored at `bits` (groups along the last dimension)."""
    shape = tuple(shape) or (1,)
    return group_quantized_bytes(math.prod(shape[:-1]), shape[-1], bits, group_size)


class CompressedTable(nn.Module):
    method = "base"

    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        if n < 1 or d < 1:
            raise ValueError("n and d must be positive")
        self.n, self.d = int(n), int(d)

    # -- interface --------------------------------------------------------------------------------

    def rows(self, index: Tensor) -> Tensor:
        raise NotImplementedError

    def storage(self) -> dict[str, int]:
        raise NotImplementedError

    def settings(self) -> dict[str, Any]:
        raise NotImplementedError

    def initialize(self, target: Tensor) -> None:
        """Closed-form or heuristic fit (before any gradient refinement)."""

    @property
    def nbytes(self) -> int:
        return int(sum(self.storage().values()))

    @property
    def device(self) -> torch.device:
        for tensor in [*self.parameters(), *self.buffers()]:
            return tensor.device
        return torch.device("cpu")

    def dense(self) -> Tensor:
        return self.rows(torch.arange(self.n, device=self.device))

    def fit(self, target: Tensor, *, steps: int = 0, lr: float = 3e-3, weights: Tensor | None = None,
            batch_rows: int | None = None, seed: int = 0) -> dict[str, float]:
        """Initialize from `target` (n × d), then `steps` Adam steps (cosine-decayed `lr`) on the
        squared reconstruction error, weighted per row by `weights` (normalized to mean 1)."""
        if tuple(target.shape) != (self.n, self.d):
            raise ValueError(f"target must have shape ({self.n}, {self.d})")
        target = target.detach().float().to(self.device)
        with torch.no_grad():
            self.initialize(target)
        trainable = [p for p in self.parameters() if p.requires_grad]
        if steps and trainable:
            w = torch.ones(self.n, device=self.device) if weights is None else weights.float().to(self.device)
            w = w / w.mean()
            optimizer = torch.optim.Adam(trainable, lr=lr)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for step in range(steps):
                for group in optimizer.param_groups:
                    group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
                if batch_rows and batch_rows < self.n:
                    index = torch.randperm(self.n, generator=generator)[:batch_rows].to(self.device)
                else:
                    index = torch.arange(self.n, device=self.device)
                error = (self.rows(index) - target[index]).square().sum(-1)
                loss = (w[index] * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        return reconstruction_stats(self, target)


@torch.no_grad()
def reconstruction_stats(table: CompressedTable, target: Tensor) -> dict[str, float]:
    approx = torch.cat([table.rows(index) for index in torch.arange(table.n, device=table.device).split(16384)])
    target = target.to(approx)
    error = (approx - target).square().sum()
    # Relative to the rows' energy, and to their energy around the mean row (a trained table often
    # shares a large common direction, which any method with a bias captures for free).
    return {"relative_error": float(error / target.square().sum().clamp_min(1e-30)),
            "relative_error_centered": float(error / (target - target.mean(0)).square().sum().clamp_min(1e-30)),
            "mean_cosine": float(F.cosine_similarity(approx, target, dim=-1, eps=1e-12).mean())}


def _largest_fitting(lo: int, hi: int, cost) -> int | None:
    """Largest integer k in [lo, hi] with cost(k) feasible (cost monotone non-decreasing in k)."""
    if lo > hi or not cost(lo):
        return None
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if cost(middle):
            lo = middle
        else:
            hi = middle - 1
    return lo


# -- the PTQ table -----------------------------------------------------------------------------------


class QuantizedTable(CompressedTable):
    method = "int"

    def __init__(self, n: int, d: int, *, bits: int = 4, group_size: int | None = 64, high_bits: int | None = None,
                 high_rows: int = 0, priority: Tensor | None = None, clip_search: bool = True) -> None:
        super().__init__(n, d)
        if high_rows and (high_bits is None or high_bits <= bits):
            raise ValueError("mixed precision needs high_bits > bits")
        if not 0 <= high_rows <= n:
            raise ValueError("high_rows must be in [0, n]")
        self.bits, self.group_size, self.clip_search = int(bits), group_size, clip_search
        self.high_bits, self.high_rows = (int(high_bits) if high_bits else None), int(high_rows)
        self.latent = nn.Parameter(torch.zeros(n, d))
        order = torch.arange(n) if priority is None else torch.as_tensor(priority, dtype=torch.long)
        rank = torch.empty(n, dtype=torch.long)
        rank[order] = torch.arange(n)
        self.register_buffer("high", rank < self.high_rows)

    def initialize(self, target: Tensor) -> None:
        self.latent.copy_(target)

    def rows(self, index: Tensor) -> Tensor:
        x = self.latent[index]
        low = stored(x, self.bits, self.group_size, clip_search=self.clip_search)
        if not self.high_rows:
            return low
        high = stored(x, self.high_bits, self.group_size, clip_search=self.clip_search)
        return torch.where(self.high[index][:, None], high, low)

    @staticmethod
    def bytes_for(n: int, d: int, bits: int, group_size: int | None, high_bits: int | None = None, high_rows: int = 0) -> dict[str, int]:
        parts = {"rows": group_quantized_bytes(n - high_rows, d, bits, group_size)}
        if high_rows:
            parts["high_rows"] = group_quantized_bytes(high_rows, d, high_bits, group_size)
            parts["precision_map"] = math.ceil(n / 8)
        return parts

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.n, self.d, self.bits, self.group_size, self.high_bits, self.high_rows)

    def settings(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "high_bits": self.high_bits, "high_rows": self.high_rows,
                "clip_search": self.clip_search}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, group_size: int | None = 64, max_bits: int = 8,
                   mixed: bool = True, priority: Tensor | None = None, **_: Any) -> "QuantizedTable | None":
        """Most bits that fit; with `mixed`, the leftover bytes promote the highest-priority rows to
        one bit more (the precision map is counted). A budget that holds the FP16 table gets it."""
        fits = [b for b in [*range(1, max_bits + 1), 16] if sum(cls.bytes_for(n, d, b, group_size).values()) <= budget]
        if not fits:
            return None
        bits = max(fits)
        high_rows = 0
        if mixed and bits < max_bits:
            high_rows = _largest_fitting(0, n, lambda k: sum(cls.bytes_for(n, d, bits, group_size, bits + 1, k).values()) <= budget) or 0
            if high_rows == n:
                bits, high_rows = bits + 1, 0
        return cls(n, d, bits=bits, group_size=group_size, high_bits=bits + 1 if high_rows else None, high_rows=high_rows,
                   priority=priority)


# -- factorized, hashed and tensor-train tables -------------------------------------------------------


class FactorizedTable(CompressedTable):
    method = "albert"

    def __init__(self, n: int, d: int, *, rank: int, bits: int = 16, group_size: int | None = 64) -> None:
        super().__init__(n, d)
        self.rank, self.bits, self.group_size = int(rank), int(bits), group_size
        if not 1 <= self.rank <= min(n, d):
            raise ValueError("rank must be in [1, min(n, d)]")
        self.left = nn.Parameter(torch.zeros(n, self.rank))
        self.right = nn.Parameter(torch.zeros(self.rank, d))

    def initialize(self, target: Tensor) -> None:
        u, s, vh = torch.linalg.svd(target, full_matrices=False)
        root = s[:self.rank].sqrt()
        self.left.copy_(u[:, :self.rank] * root)
        self.right.copy_(root[:, None] * vh[:self.rank])

    def rows(self, index: Tensor) -> Tensor:
        return stored(self.left[index], self.bits, self.group_size) @ stored(self.right, self.bits, self.group_size)

    @staticmethod
    def bytes_for(n: int, d: int, rank: int, bits: int, group_size: int | None) -> dict[str, int]:
        return {"left": stored_bytes((n, rank), bits, group_size), "right": stored_bytes((rank, d), bits, group_size)}

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.n, self.d, self.rank, self.bits, self.group_size)

    def settings(self) -> dict[str, Any]:
        return {"rank": self.rank, "bits": self.bits, "group_size": self.group_size}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, bits: int = 16, group_size: int | None = 64, **_: Any):
        rank = _largest_fitting(1, min(n, d), lambda k: sum(cls.bytes_for(n, d, k, bits, group_size).values()) <= budget)
        return None if rank is None else cls(n, d, rank=rank, bits=bits, group_size=group_size)


class QRTable(CompressedTable):
    method = "qr"

    def __init__(self, n: int, d: int, *, collisions: int, operation: str = "mult", bits: int = 16,
                 group_size: int | None = 64) -> None:
        super().__init__(n, d)
        if operation not in {"mult", "add"}:
            raise ValueError("operation must be mult or add")
        if not 1 <= collisions <= n:
            raise ValueError("collisions must be in [1, n]")
        self.collisions, self.operation, self.bits, self.group_size = int(collisions), operation, int(bits), group_size
        self.quotient = nn.Parameter(torch.zeros(-(-n // self.collisions), d))
        self.remainder = nn.Parameter(torch.zeros(self.collisions, d))
        positions = torch.arange(n)
        self.register_buffer("q_index", positions // self.collisions)
        self.register_buffer("r_index", positions % self.collisions)

    def initialize(self, target: Tensor) -> None:
        def mean_by(index: Tensor, values: Tensor, count: int) -> Tensor:
            sums = values.new_zeros(count, values.shape[1]).index_add_(0, index, values)
            return sums / torch.bincount(index, minlength=count).clamp_min(1)[:, None].to(values)
        self.quotient.copy_(mean_by(self.q_index, target, self.quotient.shape[0]))
        if self.operation == "mult":
            self.remainder.fill_(1.0)
        else:
            self.remainder.copy_(mean_by(self.r_index, target - self.quotient[self.q_index], self.collisions))

    def rows(self, index: Tensor) -> Tensor:
        q = stored(self.quotient, self.bits, self.group_size)[self.q_index[index]]
        r = stored(self.remainder, self.bits, self.group_size)[self.r_index[index]]
        return q * r if self.operation == "mult" else q + r

    @staticmethod
    def bytes_for(n: int, d: int, collisions: int, bits: int, group_size: int | None) -> dict[str, int]:
        return {"quotient": stored_bytes((-(-n // collisions), d), bits, group_size),
                "remainder": stored_bytes((collisions, d), bits, group_size)}

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.n, self.d, self.collisions, self.bits, self.group_size)

    def settings(self) -> dict[str, Any]:
        return {"collisions": self.collisions, "operation": self.operation, "bits": self.bits, "group_size": self.group_size}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, operation: str = "mult", bits: int = 16,
                   group_size: int | None = 64, **_: Any):
        """Fewest collisions `m` (most rows) that fit; total rows `⌈n/m⌉ + m` is smallest near √n."""
        for m in range(1, math.isqrt(n) + 2):
            if sum(cls.bytes_for(n, d, m, bits, group_size).values()) <= budget:
                return cls(n, d, collisions=m, operation=operation, bits=bits, group_size=group_size)
        return None


def _universal_hash(keys: Tensor, buckets: int, seed: int, count: int) -> Tensor:
    """`count` universal hashes `((a·key + c) mod p) mod buckets` (keys × count); 2·count int32 seeds."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randint(1, _PRIME, (count,), generator=generator, dtype=torch.long)
    c = torch.randint(0, _PRIME, (count,), generator=generator, dtype=torch.long)
    return ((a * keys.long()[:, None] + c) % _PRIME) % buckets


class HashedTable(CompressedTable):
    method = "hashing"

    def __init__(self, n: int, d: int, *, buckets: int, keys: Tensor, signed: bool = True, seed: int = 0,
                 bits: int = 16, group_size: int | None = 64) -> None:
        super().__init__(n, d)
        if buckets < 1 or keys.numel() != n:
            raise ValueError("buckets must be positive and keys must hold one id per row")
        self.buckets, self.signed, self.seed, self.bits, self.group_size = int(buckets), signed, seed, int(bits), group_size
        hashes = _universal_hash(keys, self.buckets, seed, 2)
        self.register_buffer("bucket", hashes[:, 0])
        sign = 1.0 - 2.0 * (_universal_hash(keys, 2, seed + 1, 1)[:, 0].float()) if signed else torch.ones(n)
        self.register_buffer("sign", sign)
        self.table = nn.Parameter(torch.zeros(self.buckets, d))

    def initialize(self, target: Tensor) -> None:
        # Least squares: each bucket is the mean of its rows times their signs.
        sums = target.new_zeros(self.buckets, self.d).index_add_(0, self.bucket, self.sign[:, None] * target)
        self.table.copy_(sums / torch.bincount(self.bucket, minlength=self.buckets).clamp_min(1)[:, None].to(target))

    def rows(self, index: Tensor) -> Tensor:
        return self.sign[index, None] * stored(self.table, self.bits, self.group_size)[self.bucket[index]]

    @staticmethod
    def bytes_for(d: int, buckets: int, bits: int, group_size: int | None, signed: bool = True) -> dict[str, int]:
        return {"table": stored_bytes((buckets, d), bits, group_size), "hash_parameters": 8 * (2 if signed else 1)}

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.d, self.buckets, self.bits, self.group_size, self.signed)

    def settings(self) -> dict[str, Any]:
        return {"buckets": self.buckets, "signed": self.signed, "seed": self.seed, "bits": self.bits, "group_size": self.group_size}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, keys: Tensor, signed: bool = True, seed: int = 0, bits: int = 16,
                   group_size: int | None = 64, **_: Any):
        buckets = _largest_fitting(1, 4 * n, lambda b: sum(cls.bytes_for(d, b, bits, group_size, signed).values()) <= budget)
        return None if buckets is None else cls(n, d, buckets=buckets, keys=keys, signed=signed, seed=seed, bits=bits,
                                                group_size=group_size)


class HashEmbeddingTable(CompressedTable):
    method = "hash_embedding"

    def __init__(self, n: int, d: int, *, buckets: int, keys: Tensor, hashes: int = 2, seed: int = 0, bits: int = 16,
                 group_size: int | None = 64) -> None:
        super().__init__(n, d)
        if buckets < 1 or hashes < 1 or keys.numel() != n:
            raise ValueError("buckets and hashes must be positive and keys must hold one id per row")
        self.buckets, self.hashes, self.seed, self.bits, self.group_size = int(buckets), int(hashes), seed, int(bits), group_size
        self.register_buffer("bucket", _universal_hash(keys, self.buckets, seed, self.hashes))
        self.pool = nn.Parameter(torch.zeros(self.buckets, d))
        self.importance = nn.Parameter(torch.zeros(n, self.hashes))

    def initialize(self, target: Tensor) -> None:
        first = self.bucket[:, 0]
        sums = target.new_zeros(self.buckets, self.d).index_add_(0, first, target)
        self.pool.copy_(sums / torch.bincount(first, minlength=self.buckets).clamp_min(1)[:, None].to(target))
        self.importance.zero_()
        self.importance[:, 0] = 1.0

    def rows(self, index: Tensor) -> Tensor:
        weights = stored(self.importance[index], self.bits, self.group_size)
        vectors = stored(self.pool, self.bits, self.group_size)[self.bucket[index]]
        return (weights[..., None] * vectors).sum(1)

    @staticmethod
    def bytes_for(n: int, d: int, buckets: int, hashes: int, bits: int, group_size: int | None) -> dict[str, int]:
        return {"pool": stored_bytes((buckets, d), bits, group_size), "importance": stored_bytes((n, hashes), bits, group_size),
                "hash_parameters": 8 * hashes}

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.n, self.d, self.buckets, self.hashes, self.bits, self.group_size)

    def settings(self) -> dict[str, Any]:
        return {"buckets": self.buckets, "hashes": self.hashes, "seed": self.seed, "bits": self.bits, "group_size": self.group_size}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, keys: Tensor, hashes: int = 2, seed: int = 0, bits: int = 16,
                   group_size: int | None = 64, **_: Any):
        buckets = _largest_fitting(1, 4 * n, lambda b: sum(cls.bytes_for(n, d, b, hashes, bits, group_size).values()) <= budget)
        return None if buckets is None else cls(n, d, buckets=buckets, keys=keys, hashes=hashes, seed=seed, bits=bits,
                                                group_size=group_size)


def balanced_factors(value: int, parts: int = 3) -> tuple[int, ...]:
    """`parts` integer factors of `value` (product exactly `value`) as equal as possible."""
    best: tuple[int, ...] | None = None

    def search(remaining: int, left: int, prefix: tuple[int, ...]) -> None:
        nonlocal best
        if left == 1:
            candidate = tuple(sorted(prefix + (remaining,)))
            if best is None or max(candidate) / min(candidate) < max(best) / min(best):
                best = candidate
            return
        for factor in [f for f in range(1, remaining + 1) if remaining % f == 0]:
            if not prefix or factor >= prefix[-1]:
                search(remaining // factor, left - 1, prefix + (factor,))

    search(int(value), parts, ())
    return best


def row_factors(n: int) -> tuple[int, int, int]:
    """(s, s, ⌈n/s²⌉) with s = ⌈n^(1/3)⌉: covers `n` rows (the excess rows are padding)."""
    side = max(1, math.ceil(n ** (1 / 3) - 1e-9))
    return side, side, math.ceil(n / (side * side))


class TTTable(CompressedTable):
    method = "tt"

    def __init__(self, n: int, d: int, *, rank: int, rows: tuple[int, int, int] | None = None,
                 columns: tuple[int, int, int] | None = None, bits: int = 16, group_size: int | None = 64) -> None:
        super().__init__(n, d)
        self.row_shape = tuple(rows or row_factors(n))
        self.column_shape = tuple(columns or balanced_factors(d))
        if math.prod(self.row_shape) < n or math.prod(self.column_shape) != d:
            raise ValueError("row factors must cover n and column factors must multiply to d")
        (n1, n2, n3), (d1, d2, d3) = self.row_shape, self.column_shape
        self.rank, self.bits, self.group_size = int(rank), int(bits), group_size
        self.ranks = self.effective_ranks(self.row_shape, self.column_shape, self.rank)
        r1, r2 = self.ranks
        self.core1 = nn.Parameter(torch.zeros(n1, d1, r1))
        self.core2 = nn.Parameter(torch.zeros(r1, n2, d2, r2))
        self.core3 = nn.Parameter(torch.zeros(r2, n3, d3))

    @staticmethod
    def effective_ranks(row_shape, column_shape, rank: int) -> tuple[int, int]:
        (n1, n2, n3), (d1, d2, d3) = row_shape, column_shape
        r1 = min(rank, n1 * d1, n2 * d2 * n3 * d3)
        r2 = min(rank, n3 * d3, r1 * n2 * d2)
        return r1, r2

    def initialize(self, target: Tensor) -> None:
        """TT-SVD of the padded table reshaped to (n1·d1) × (n2·d2) × (n3·d3)."""
        (n1, n2, n3), (d1, d2, d3) = self.row_shape, self.column_shape
        r1, r2 = self.ranks
        padded = target.new_zeros(n1 * n2 * n3, self.d)
        padded[:self.n] = target
        tensor = padded.reshape(n1, n2, n3, d1, d2, d3).permute(0, 3, 1, 4, 2, 5)
        u, s, vh = torch.linalg.svd(tensor.reshape(n1 * d1, -1), full_matrices=False)
        self.core1.copy_(u[:, :r1].reshape(n1, d1, r1))
        rest = (s[:r1, None] * vh[:r1]).reshape(r1 * n2 * d2, n3 * d3)
        u, s, vh = torch.linalg.svd(rest, full_matrices=False)
        self.core2.copy_(u[:, :r2].reshape(r1, n2, d2, r2))
        self.core3.copy_((s[:r2, None] * vh[:r2]).reshape(r2, n3, d3))

    def rows(self, index: Tensor) -> Tensor:
        # Contract the cores into the padded table (n1·n2·n3 × d values, the size of the table itself)
        # and gather: gathering the middle core per row would materialize k × r1 × d2 × r2 values.
        g1 = stored(self.core1, self.bits, self.group_size)                              # n1 × d1 × r1
        g2 = stored(self.core2, self.bits, self.group_size)                              # r1 × n2 × d2 × r2
        g3 = stored(self.core3, self.bits, self.group_size)                              # r2 × n3 × d3
        prefix = torch.einsum("iar,rjbs->ijabs", g1, g2)
        full = torch.einsum("ijabs,skc->ijkabc", prefix, g3)
        return full.reshape(-1, self.d)[index]

    @classmethod
    def bytes_for(cls, row_shape, column_shape, rank: int, bits: int, group_size: int | None) -> dict[str, int]:
        (n1, n2, n3), (d1, d2, d3) = row_shape, column_shape
        r1, r2 = cls.effective_ranks(row_shape, column_shape, rank)
        return {"core1": stored_bytes((n1, d1, r1), bits, group_size), "core2": stored_bytes((r1, n2, d2, r2), bits, group_size),
                "core3": stored_bytes((r2, n3, d3), bits, group_size)}

    def storage(self) -> dict[str, int]:
        return self.bytes_for(self.row_shape, self.column_shape, self.rank, self.bits, self.group_size)

    def settings(self) -> dict[str, Any]:
        return {"rank": self.rank, "ranks": list(self.ranks), "row_shape": list(self.row_shape),
                "column_shape": list(self.column_shape), "bits": self.bits, "group_size": self.group_size}

    @classmethod
    def for_budget(cls, budget: int, n: int, d: int, *, bits: int = 16, group_size: int | None = 64, **_: Any):
        rows, columns = row_factors(n), balanced_factors(d)
        cap = max(cls.effective_ranks(rows, columns, n * d))
        rank = _largest_fitting(1, cap, lambda r: sum(cls.bytes_for(rows, columns, r, bits, group_size).values()) <= budget)
        return None if rank is None else cls(n, d, rank=rank, rows=rows, columns=columns, bits=bits, group_size=group_size)


# -- the ontology composition: P c + b + δ ------------------------------------------------------------


def entry_subschedule(offsets: Tensor, relations: Tensor, fillers: Tensor, entries: Tensor) -> tuple[FrameSchedule, Tensor]:
    """Frames of `entries` (global ids, in order) with fillers renumbered over the atomics they use;
    returns the local schedule and the global ids of those atomics (sorted)."""
    entries = torch.as_tensor(entries, dtype=torch.long)
    starts, ends = offsets[entries], offsets[entries + 1]
    degrees = ends - starts
    segment = torch.repeat_interleave(torch.arange(entries.numel()), degrees)
    edge = starts[segment] + torch.arange(int(degrees.sum())) - (torch.cumsum(degrees, 0) - degrees)[segment]
    atomics, local = torch.unique(fillers[edge], return_inverse=True)
    local_offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(degrees, 0)])
    return FrameSchedule(local_offsets, relations[edge].clone(), local), atomics


class ComposedTable(CompressedTable):
    method = "composed"

    def __init__(self, n: int, d: int, *, entries: Tensor, schedule: FrameSchedule, atomic_count: int, relation_count: int,
                 dimension: int, operator: str = "hrr", composition: str = "bundle", key_dimension: int = 8,
                 delta_bits: int = 0, group_size: int | None = 64, dictionary_bits: int = 16, fit_dictionary: bool = True) -> None:
        super().__init__(n, d)
        entries = torch.as_tensor(entries, dtype=torch.long)
        if entries.numel() != n or (n and int(entries.max()) >= schedule.concept_count):
            raise ValueError("entries must give one schedule entry per row")
        self.delta_bits, self.group_size, self.dictionary_bits = int(delta_bits), group_size, int(dictionary_bits)
        self.dimension, self.operator, self.composition = int(dimension), operator, composition
        self.composer = FrameComposer(schedule, atomic_count, relation_count, dimension, operator=operator, mode=composition,
                                      key_dimension=key_dimension)
        self.projector = nn.Linear(dimension, d)
        self.register_buffer("entries", entries)
        self.delta = nn.Parameter(torch.zeros(n, d)) if self.delta_bits else None
        self.fit_dictionary = fit_dictionary
        self.composer.requires_grad_(fit_dictionary)
        if operator.startswith("random_fixed"):
            self.composer.transform.requires_grad_(False)

    def load_dictionary(self, composer: FrameComposer, atomic_ids: Tensor) -> None:
        """Start from a trained composer (a channel's): its atomics `atomic_ids` (the global ids this
        table's schedule uses) and every other dictionary parameter, which must have the same shapes."""
        source = composer.state_dict()
        own = self.composer.state_dict()
        with torch.no_grad():
            for name, tensor in own.items():
                if name.startswith("frame_"):
                    continue
                value = source[name]
                if name == "atomics":
                    value = value[atomic_ids.to(value.device)]
                if tuple(value.shape) != tuple(tensor.shape):
                    raise ValueError(f"dictionary parameter {name}: {tuple(value.shape)} vs {tuple(tensor.shape)}")
                tensor.copy_(value)

    def _concepts(self, index: Tensor) -> Tensor:
        entries = self.entries[index]
        unique, inverse = torch.unique(entries, return_inverse=True)
        parameters = {name: stored(p, self.dictionary_bits if p.ndim > 1 else 16, self.group_size)
                      for name, p in self.composer.named_parameters()}
        rows = torch.func.functional_call(self.composer, parameters, (unique,))
        return rows[inverse]

    def base_rows(self, index: Tensor) -> Tensor:
        """`P c + b` (no residual)."""
        weight = stored(self.projector.weight, self.dictionary_bits, self.group_size)
        return F.linear(self._concepts(index), weight, stored(self.projector.bias, 16))

    def rows(self, index: Tensor) -> Tensor:
        out = self.base_rows(index)
        if self.delta is not None:
            out = out + stored(self.delta[index], self.delta_bits, self.group_size, clip_search=True)
        return out

    def initialize(self, target: Tensor) -> None:
        """Least-squares projector and bias for the current dictionary; residual δ = T − (P c + b)."""
        index = torch.arange(self.n, device=target.device)
        concepts = self._concepts(index)
        design = torch.cat([concepts, torch.ones_like(concepts[:, :1])], 1)
        solution = torch.linalg.lstsq(design.cpu().double(), target.cpu().double()).solution.to(target)
        self.projector.weight.copy_(solution[:-1].T)
        self.projector.bias.copy_(solution[-1])

    def fit(self, target: Tensor, *, steps: int = 0, **kwargs: Any) -> dict[str, float]:
        """Fit `P c + b ≈ T` alone (least squares, then `steps` of Adam on the dictionary and
        projector), then set δ to the residual (stored at `delta_bits`). Also returns the error of
        the base `P c + b` without δ."""
        delta, self.delta = self.delta, None
        try:
            base = super().fit(target, steps=steps, **kwargs)
        finally:
            self.delta = delta
        target = target.detach().float().to(self.device)
        if self.delta is not None:
            with torch.no_grad():
                self.delta.copy_(target - self.base_rows(torch.arange(self.n, device=self.device)))
        return {**reconstruction_stats(self, target), "base_relative_error": base["relative_error"],
                "base_relative_error_centered": base["relative_error_centered"], "base_mean_cosine": base["mean_cosine"]}

    def dictionary_storage(self) -> dict[str, int]:
        parts = {}
        for name, parameter in self.composer.named_parameters():
            shape = tuple(parameter.shape) or (1,)
            parts[f"dictionary.{name}"] = stored_bytes(shape, self.dictionary_bits if len(shape) > 1 else 16, self.group_size)
        schedule = self.composer.schedule
        edges, entries = schedule.relations.numel(), schedule.concept_count
        parts["schedule.degrees"] = index_bytes(int(schedule.degrees.max()) + 1, entries)
        parts["schedule.relations"] = index_bytes(self.composer.relation_count, edges)
        parts["schedule.fillers"] = index_bytes(self.composer.atomic_count, edges)
        return parts

    def storage(self) -> dict[str, int]:
        parts = self.dictionary_storage()
        parts["projector.weight"] = stored_bytes((self.d, self.dimension), self.dictionary_bits, self.group_size)
        parts["projector.bias"] = stored_bytes((self.d,), 16)
        parts["row_entries"] = index_bytes(self.composer.schedule.concept_count, self.n)
        if self.delta_bits:
            parts["delta"] = group_quantized_bytes(self.n, self.d, self.delta_bits, self.group_size)
        return parts

    def marginal_bytes(self) -> int:
        """Bytes beyond a dictionary and schedule the model already holds (a channel's, reused unchanged)."""
        return self.nbytes - sum(self.dictionary_storage().values())

    def settings(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "operator": self.operator, "composition": self.composition,
                "delta_bits": self.delta_bits, "dictionary_bits": self.dictionary_bits, "group_size": self.group_size,
                "atomics": self.composer.atomic_count, "entries": self.composer.schedule.concept_count,
                "edges": int(self.composer.schedule.relations.numel()), "fit_dictionary": self.fit_dictionary}


METHODS: dict[str, type[CompressedTable]] = {cls.method: cls for cls in (
    QuantizedTable, FactorizedTable, QRTable, HashedTable, HashEmbeddingTable, TTTable, ComposedTable)}


class ReplacedRowsEmbedding(nn.Module):
    """Input embedding whose rows `token_ids` come from `table` (row i of the table ↔ token_ids[i]).

    `weight` is the original, uncompressed table (still the output head's when tied). `cache()`
    stores the reconstructed rows for evaluation; any training step should call `clear_cache()`."""

    def __init__(self, base: nn.Embedding, token_ids: Tensor, table: CompressedTable) -> None:
        super().__init__()
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        if token_ids.numel() != table.n or token_ids.unique().numel() != token_ids.numel():
            raise ValueError("token_ids must list table.n distinct tokens")
        self.base, self.table = base, table
        slot = torch.full((base.num_embeddings,), -1, dtype=torch.long)
        slot[token_ids] = torch.arange(token_ids.numel())
        self.register_buffer("slot", slot.to(base.weight.device))
        self._cache: Tensor | None = None

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def num_embeddings(self) -> int:
        return self.base.num_embeddings

    @torch.no_grad()
    def cache(self) -> None:
        self._cache = torch.cat([self.table.rows(index) for index in torch.arange(self.table.n, device=self.slot.device).split(16384)])

    def clear_cache(self) -> None:
        self._cache = None

    def forward(self, input_ids: Tensor) -> Tensor:
        out = self.base(input_ids)
        slot = self.slot[input_ids]
        mask = slot >= 0
        if not bool(mask.any()):
            return out
        wanted = slot[mask]
        if self._cache is not None:
            rows = self._cache[wanted]
        else:
            unique, inverse = torch.unique(wanted, return_inverse=True)
            rows = self.table.rows(unique)[inverse]
        return out.index_put(tuple(mask.nonzero(as_tuple=True)), rows.to(out.dtype))
