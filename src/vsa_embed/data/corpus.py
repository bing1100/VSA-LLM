"""Linked token corpora: memmapped tokens plus span tables, and deterministic window sampling.

On-disk layout of a corpus directory:

    tokens.bin        uint16 (vocab < 65,536) or uint32 token stream; documents joined by EOS
    spans.npz         arrays start, end, inject, entry, length (int64) and confidence (float32),
                      global token positions, sorted by `inject`
    manifest.json     tokenizer, alias-table digest, linker settings, counts, document count

Sampling is a pure function of `(seed, step)`, so a resumed run sees the same batches.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from ..span_channel import AliasTable, CausalLinker

SPAN_FIELDS = ("start", "end", "inject", "entry", "length")


_WORKER: dict[str, Any] = {}


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """sha256 of a fast tokenizer's full serialization (model, merges, normalizer, added tokens):
    equal fingerprints mean identical token ids, so one corpus serves every host that shares it."""
    return hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()


def _init_worker(tokenizer_name: str, revision: str, table_path: str, boundary: str, min_subtokens: int,
                 normalization: str = "") -> None:
    """Load the tokenizer and the alias table (from a file) once per spawned worker.

    The table is passed by path: pickling a ~150k-alias table into every spawn payload deadlocked
    the parent on a full pipe, and fork is unsafe in a multi-threaded (torch) parent.
    `normalization` (e.g. "NFC" for Qwen2.5, whose tokenizer normalizes its input) makes the
    decode round-trip check compare against the normalized text.
    """
    import os
    import pickle
    # Each worker tokenizes single-threaded (process-level parallelism only): the Rust tokenizer's
    # internal threads were associated with panics and corrupted ids on rare inputs.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    with open(table_path, "rb") as handle:
        table = pickle.load(handle)
    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision or None, local_files_only=True)
    _WORKER["linker"] = CausalLinker(table, boundary=boundary, min_subtokens=min_subtokens)
    _WORKER["normalization"] = normalization or None


def _encode_batch(texts: list[str]) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]], int]:
    tokenizer, linker = _WORKER["tokenizer"], _WORKER["linker"]
    try:
        encoded = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        pairs = list(zip(texts, encoded["input_ids"], encoded["offset_mapping"]))
    except BaseException as error:   # the Rust tokenizer can panic on rare inputs (pyo3 PanicException)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        pairs = []
        for text in texts:            # retry one document at a time and skip the ones that panic
            try:
                single = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                pairs.append((text, single["input_ids"], single["offset_mapping"]))
            except BaseException as inner:
                if isinstance(inner, (KeyboardInterrupt, SystemExit)):
                    raise
    tokens, spans = [], []
    vocabulary = len(tokenizer)
    normalization = _WORKER.get("normalization")
    checked = []
    for text, ids, offsets in pairs:
        # Byte-level BPE is lossless: a document whose ids are out of range or do not decode back
        # to the text was corrupted by the tokenizer and is dropped (counted as skipped).
        expected = unicodedata.normalize(normalization, text) if normalization else text
        if ids and (max(ids) >= vocabulary or tokenizer.decode(ids) != expected):
            continue
        checked.append((text, ids, offsets))
    for text, ids, offsets in checked:
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
    return tokens, spans, len(texts) - len(checked)


def build_corpus(
    texts: Iterable[str], out_dir: Path, *, tokenizer_name: str, table: AliasTable, eos_id: int,
    max_tokens: int, revision: str | None = None, boundary: str = "prefix", min_subtokens: int = 1,
    batch_texts: int = 256, workers: int = 8, extra_manifest: dict[str, Any] | None = None,
    vocab_size: int = 50257, reuse: bool = False, normalization: str | None = None,
) -> dict[str, Any]:
    """Tokenize and link documents in parallel until `max_tokens`; spans keep every length (≥ 1),
    so `ℓ_min` is applied at sampling time and one corpus serves every threshold.

    `vocab_size` sets the token dtype (uint16 up to 65,536 ids, else uint32) and the id range check;
    `normalization` is the Unicode form the tokenizer applies to its input (None = none), used by
    the decode round-trip check."""
    if reuse and (out_dir / "manifest.json").exists():
        existing = json.loads((out_dir / "manifest.json").read_text())
        if existing.get("alias_table_sha256") == table.digest():
            return existing          # completed earlier with the same alias table: resume past it
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    token_path = out_dir / "tokens.bin"
    position, documents, skipped_documents = 0, 0, 0
    span_parts: dict[str, list[np.ndarray]] = {k: [] for k in (*SPAN_FIELDS, "confidence")}

    def batches() -> Iterator[list[str]]:
        chunk: list[str] = []
        for text in texts:
            chunk.append(text)
            if len(chunk) == batch_texts:
                yield chunk; chunk = []
        if chunk:
            yield chunk

    import pickle
    table_path = out_dir / ".alias_table.pkl"
    with table_path.open("wb") as handle:
        pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    initargs = (tokenizer_name, revision or "", str(table_path), boundary, min_subtokens, normalization or "")

    def ordered_results(executor: ProcessPoolExecutor) -> Iterator[tuple]:
        # Bounded in-flight window, results in submission order. A crashed worker raises
        # BrokenProcessPool here instead of hanging the build (multiprocessing.Pool hangs).
        pending: deque = deque()
        for chunk in batches():
            pending.append(executor.submit(_encode_batch, chunk))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

    with token_path.open("wb") as handle, ProcessPoolExecutor(
            workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=initargs) as pool:
        for tokens, spans, skipped in ordered_results(pool):
            skipped_documents += skipped
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
                pool.shutdown(wait=False, cancel_futures=True)
                break
    arrays = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.float32 if k == "confidence" else np.int64))
              for k, v in span_parts.items()}
    table_path.unlink(missing_ok=True)
    order = np.argsort(arrays["inject"], kind="stable")
    # Compact on-disk types: positions and entries int32, lengths uint8, confidence float16.
    compact = {"start": np.int32, "end": np.int32, "inject": np.int32, "entry": np.int32,
               "length": np.uint8, "confidence": np.float16}
    if position >= 2**31:
        raise ValueError("corpus too large for int32 span positions")
    np.savez(out_dir / "spans.npz", **{k: v[order].astype(compact[k]) for k, v in arrays.items()})
    manifest = {"tokens": int(position), "documents": documents, "dtype": np.dtype(dtype).name, "tokenizer": tokenizer_name,
                "tokenizer_revision": revision, "alias_table_sha256": table.digest(), "boundary": boundary,
                "spans": int(arrays["inject"].size), "eos_id": eos_id, "skipped_documents": skipped_documents,
                **({"normalization": normalization} if normalization else {}), **(extra_manifest or {})}
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
