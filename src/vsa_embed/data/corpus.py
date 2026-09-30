"""Linked token corpora: memmapped tokens plus span tables, and deterministic window sampling.

On-disk layout of a corpus directory:

    tokens.bin        uint16 (vocab < 65,536) or uint32 token stream; documents joined by EOS
    spans.npz         arrays start, end, inject, entry, length (int64) and confidence (float32),
                      global token positions, sorted by `inject`
    manifest.json     tokenizer, alias-table digest, linker settings, counts, document count

Sampling is a pure function of `(seed, step)`, so a resumed run sees the same batches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from ..span_channel import AliasTable, CausalLinker

SPAN_FIELDS = ("start", "end", "inject", "entry", "length")


_WORKER: dict[str, Any] = {}


def _init_worker(tokenizer_name: str, revision: str, table: AliasTable, boundary: str, min_subtokens: int) -> None:
    """Load the tokenizer and build the linker once per worker process."""
    from transformers import AutoTokenizer
    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision or None, local_files_only=True)
    _WORKER["linker"] = CausalLinker(table, boundary=boundary, min_subtokens=min_subtokens)


def _encode_batch(texts: list[str]) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    tokenizer, linker = _WORKER["tokenizer"], _WORKER["linker"]
    encoded = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
    tokens, spans = [], []
    for text, ids, offsets in zip(texts, encoded["input_ids"], encoded["offset_mapping"]):
        found = linker.link(text, offsets)
        tokens.append(np.asarray(ids, dtype=np.int64))
        spans.append({
            "start": np.asarray([s.start_token for s in found], dtype=np.int64),
            "end": np.asarray([s.end_token for s in found], dtype=np.int64),
            "inject": np.asarray([s.inject_token for s in found], dtype=np.int64),
            "entry": np.asarray([s.entry for s in found], dtype=np.int64),
            "length": np.asarray([s.length for s in found], dtype=np.int64),
            "confidence": np.asarray([s.confidence for s in found], dtype=np.float32),
        })
    return tokens, spans


def build_corpus(
    texts: Iterable[str], out_dir: Path, *, tokenizer_name: str, table: AliasTable, eos_id: int,
    max_tokens: int, revision: str | None = None, boundary: str = "prefix", min_subtokens: int = 1,
    batch_texts: int = 256, workers: int = 8, extra_manifest: dict[str, Any] | None = None,
    vocab_size: int = 50257,
) -> dict[str, Any]:
    """Tokenize and link documents in parallel until `max_tokens`; spans keep every length (≥ 1),
    so `ℓ_min` is applied at sampling time and one corpus serves every threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    token_path = out_dir / "tokens.bin"
    position, documents = 0, 0
    span_parts: dict[str, list[np.ndarray]] = {k: [] for k in (*SPAN_FIELDS, "confidence")}

    def batches() -> Iterator[list[str]]:
        chunk: list[str] = []
        for text in texts:
            chunk.append(text)
            if len(chunk) == batch_texts:
                yield chunk; chunk = []
        if chunk:
            yield chunk

    initargs = (tokenizer_name, revision or "", table, boundary, min_subtokens)
    with token_path.open("wb") as handle, get_context("spawn").Pool(workers, _init_worker, initargs) as pool:
        for tokens, spans in pool.imap(_encode_batch, batches()):
            for ids, doc_spans in zip(tokens, spans):
                if position >= max_tokens:
                    break
                if int(ids.max(initial=0)) >= vocab_size:
                    raise ValueError(f"token id ≥ vocab_size {vocab_size}")
                block = np.concatenate([ids, [eos_id]]).astype(dtype)
                handle.write(block.tobytes())
                for key in span_parts:
                    values = doc_spans[key]
                    span_parts[key].append(values + position if key in ("start", "end", "inject") else values)
                position += block.size; documents += 1
            if position >= max_tokens:
                pool.terminate()
                break
    arrays = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32 if k == "confidence" else np.int64))
              for k, v in span_parts.items()}
    order = np.argsort(arrays["inject"], kind="stable")
    # Compact on-disk types: positions and entries int32, lengths uint8, confidence float16.
    compact = {"start": np.int32, "end": np.int32, "inject": np.int32, "entry": np.int32,
               "length": np.uint8, "confidence": np.float16}
    if position >= 2**31:
        raise ValueError("corpus too large for int32 span positions")
    np.savez(out_dir / "spans.npz", **{k: v[order].astype(compact[k]) for k, v in arrays.items()})
    manifest = {"tokens": int(position), "documents": documents, "dtype": np.dtype(dtype).name, "tokenizer": tokenizer_name,
                "tokenizer_revision": revision, "alias_table_sha256": table.digest(), "boundary": boundary,
                "spans": int(arrays["inject"].size), "eos_id": eos_id, **(extra_manifest or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class TokenCorpus:
    tokens: np.ndarray
    spans: dict[str, np.ndarray]
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: Path) -> "TokenCorpus":
        manifest = json.loads((path / "manifest.json").read_text())
        tokens = np.memmap(path / "tokens.bin", dtype=np.uint16 if manifest["dtype"] == "uint16" else np.uint32, mode="r")
        with np.load(path / "spans.npz") as data:
            spans = {k: data[k] for k in data.files}
        return cls(tokens, spans, manifest)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def window(self, start: int, length: int, *, min_subtokens: int = 1,
               entry_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Tokens `[start, start + length)` and the spans lying fully inside, in local positions."""
        ids = np.asarray(self.tokens[start:start + length], dtype=np.int64)
        lo = np.searchsorted(self.spans["inject"], start, side="left")
        hi = np.searchsorted(self.spans["inject"], start + length, side="left")
        keep = (self.spans["start"][lo:hi] >= start) & (self.spans["length"][lo:hi] >= min_subtokens)
        if entry_mask is not None:
            keep &= entry_mask[self.spans["entry"][lo:hi]]
        local = {k: v[lo:hi][keep] for k, v in self.spans.items()}
        for key in ("start", "end", "inject"):
            local[key] = local[key] - start
        return ids, local


def collate_windows(windows: Sequence[tuple[np.ndarray, dict[str, np.ndarray]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ids = torch.from_numpy(np.stack([w[0] for w in windows]))
    parts = {k: [] for k in (*SPAN_FIELDS, "confidence", "batch")}
    for b, (_, spans) in enumerate(windows):
        for key in (*SPAN_FIELDS, "confidence"):
            parts[key].append(spans[key])
        parts["batch"].append(np.full(spans["entry"].shape[0], b, dtype=np.int64))
    spans_t = {k: torch.from_numpy(np.concatenate(v).astype(np.float32 if k == "confidence" else np.int64))
               for k, v in parts.items()}
    return ids, spans_t


def sample_batch(corpus: TokenCorpus, *, seed: int, step: int, micro_step: int, batch: int, length: int,
                 min_subtokens: int = 1, entry_mask: np.ndarray | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deterministic random windows for `(seed, step, micro_step)`."""
    generator = np.random.default_rng([seed, step, micro_step])
    starts = generator.integers(0, len(corpus) - length - 1, size=batch)
    return collate_windows([corpus.window(int(s), length, min_subtokens=min_subtokens, entry_mask=entry_mask)
                            for s in starts])


def eval_windows(corpus: TokenCorpus, *, count: int, length: int, seed: int = 12345) -> list[int]:
    """Fixed evaluation window starts (non-overlapping, evenly spread, then a seeded shuffle)."""
    stride = max(length, (len(corpus) - length - 1) // max(1, count))
    starts = list(range(0, len(corpus) - length - 1, stride))[:count]
    rng = np.random.default_rng(seed)
    rng.shuffle(starts)
    return sorted(starts)
