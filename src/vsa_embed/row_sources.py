"""Files shared by the trainer and the E9 paper-quality tools (WP-PQ1): row-source tables and filler indexes.

**Row-source tables** (`channel.mode: source`, arms C6m / C6d / C6g of `e9_plan`): one frozen vector per link entry —
the mean of the host's input-embedding rows over the term's subtokens (FVT / Hewitt), the frozen host's mean-pooled
hidden state over the entry's verbalized frame (a definition encoder), or a knowledge-graph embedding of the entry
(TransE on the track ontology). `experiments.e9_rowsource` builds them; the trainer reads them through
`load_source_table`, which checks the entry count and the ontology frames they were built from.

**Filler indexes** (`eval.filler_strata`, `experiments.e9_rescore`): the token sequences that verbalize each atomic
(filler) of the track ontology, so the loss in the 8 tokens after a linked span can be split into tokens that belong
to an alias of a filler of that span's frame (`<stratum>_filler`) and the rest (`<stratum>_nonfiller`).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


def frames_digest(ontology: dict[str, Any]) -> str:
    """sha256 of an ontology's entry frames (offsets, relations, fillers): a table built for one frame set refuses
    another."""
    digest = hashlib.sha256()
    for key in ("offsets", "relations", "fillers"):
        digest.update(np.ascontiguousarray(np.asarray(ontology[key], dtype=np.int64)).tobytes())
    return digest.hexdigest()


def standardize_rows(rows: Tensor, reference: Sequence[int] | Tensor | None = None, eps: float = 1e-6) -> tuple[Tensor, dict[str, Tensor]]:
    """Whiten each dimension (mean and standard deviation over the `reference` rows: the entries that are not held
    out) and scale every row to unit L2 norm, like the composer's output; returns (rows, statistics)."""
    x = rows.double()
    ref = x if reference is None else x[torch.as_tensor(reference, dtype=torch.long)]
    mean, std = ref.mean(0), ref.std(0).clamp_min(eps)
    z = (x - mean) / std
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(eps)
    return z.float(), {"mean": mean.float(), "std": std.float()}


def save_source_table(path: Path, rows: Tensor, *, kind: str, ontology: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Write a row-source table (rows stored in FP16) with its provenance; returns the record without the rows."""
    record = {"kind": kind, "entry_count": int(rows.shape[0]), "dimension": int(rows.shape[1]),
              "frames_sha256": frames_digest(ontology), "meta": meta}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    torch.save({**record, "rows": rows.detach().to(torch.float16).cpu()}, temporary)
    temporary.replace(path)
    return record


def load_source_table(path: Path | str, ontology: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
    """(rows as float32, record) of a row-source table, checked against the run's ontology when given."""
    data = torch.load(Path(path).expanduser(), weights_only=False, map_location="cpu")
    rows = data["rows"].float()
    if ontology is not None:
        if int(data["entry_count"]) != int(ontology["entry_count"]):
            raise ValueError(f"{path}: {data['entry_count']} rows for an ontology of {ontology['entry_count']} entries")
        if data.get("frames_sha256") and data["frames_sha256"] != frames_digest(ontology):
            raise ValueError(f"{path} was built for other ontology frames")
    return rows, {k: v for k, v in data.items() if k != "rows"}


class FillerIndex:
    """Filler token sequences of each entry's frame (see the module docstring). `sequences[a]` holds the token-id
    tuples that verbalize atomic `a`; an entry's fillers are the atomics of its ontology frame."""

    def __init__(self, offsets: Sequence[int], fillers: Sequence[int], sequences: Sequence[Sequence[Sequence[int]]]) -> None:
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.fillers = np.asarray(fillers, dtype=np.int64)
        self.by_first: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
        for atomic, seqs in enumerate(sequences):
            for seq in {tuple(int(t) for t in s) for s in seqs if len(s)}:
                self.by_first.setdefault(seq[0], []).append((atomic, seq))
        self._atoms = lru_cache(maxsize=None)(self._entry_atomics)

    def _entry_atomics(self, entry: int) -> frozenset[int]:
        if entry + 1 >= self.offsets.size:
            return frozenset()           # an entry beyond the ontology (inserted at evaluation) has no known frame
        return frozenset(self.fillers[self.offsets[entry]:self.offsets[entry + 1]].tolist())

    def target_hits(self, row: Sequence[int], entry: int, first: int, last: int) -> list[int]:
        """Target positions `j ∈ [first, last]` covered by an occurrence of a filler sequence of `entry` that starts
        inside `[first, last]` (so it starts after the span; tokens of the span itself are never counted)."""
        atoms = self._atoms(int(entry))
        if not atoms:
            return []
        hits: set[int] = set()
        for p in range(max(first, 0), min(last, len(row) - 1) + 1):
            for atomic, seq in self.by_first.get(row[p], ()):
                if atomic in atoms and tuple(row[p:p + len(seq)]) == seq:
                    hits.update(range(p, min(p + len(seq) - 1, last) + 1))
        return sorted(hits)


def save_filler_table(path: Path, *, ontology: dict[str, Any], surfaces: Sequence[Sequence[str]],
                      sequences: Sequence[Sequence[Sequence[int]]], meta: dict[str, Any]) -> dict[str, Any]:
    if len(surfaces) != int(ontology["atomic_count"]) or len(sequences) != int(ontology["atomic_count"]):
        raise ValueError("one surface list and one sequence list per atomic")
    record = {"atomic_count": int(ontology["atomic_count"]), "frames_sha256": frames_digest(ontology), "meta": meta,
              "surfaces": [list(s) for s in surfaces], "sequences": [[list(map(int, q)) for q in s] for s in sequences]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(path)
    return {k: v for k, v in record.items() if k not in {"surfaces", "sequences"}}


def load_filler_index(path: Path | str, ontology: dict[str, Any]) -> FillerIndex:
    """The filler index of a filler table, checked against the run's ontology frames."""
    data = torch.load(Path(path).expanduser(), weights_only=False, map_location="cpu")
    if data.get("frames_sha256") != frames_digest(ontology):
        raise ValueError(f"{path} was built for other ontology frames")
    return FillerIndex(ontology["offsets"], ontology["fillers"], data["sequences"])
