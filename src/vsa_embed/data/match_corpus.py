"""All-match corpora (E7): tokenize and scan documents once, then link them with many alias tables.

Self-authoring (formulation §5) trains the same reading text under many linkers — the curated
ontology, the self-authored one, the teacher's, random frames — and grows the alias set after
candidate discovery. Re-tokenizing and re-linking per linker would repeat the expensive part, so a
match corpus stores every alias match once, before the longest-match choice:

    tokens.bin      token stream (uint16/uint32), documents joined by EOS, as `corpus.build_corpus`
    documents.npy   start token of every document (int64)
    matches.npz     `token` (last subtoken), `first` (first subtoken), `string` (alias id), `chars`
                    (characters from the alias start to the token end) for every alias match covering
                    ≥ `min_subtokens` subtokens (`CausalLinker.all_matches`, prefix boundary)
    strings.json    alias strings (id → normalized alias); `add_matches` appends new ones
    texts.jsonl     the kept documents (optional; needed by `add_matches`)
    manifest.json

`select_longest` resolves a linker — an alias-string → entry map, −1 for inactive strings — exactly
as `CausalLinker.link` would: at each token the active match with the most characters wins and is
kept if it covers ≥ ℓ_min subtokens. `write_view` writes the result as a `TokenCorpus` directory
(tokens.bin symlinked, spans.npz, manifest.json) that the trainer reads unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import unicodedata
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..span_channel import LINKER_VERSION, AliasTable, CausalLinker

MATCH_FIELDS = ("token", "first", "string", "chars")
_WORKER: dict[str, Any] = {}


def strings_table(strings: Sequence[str]) -> AliasTable:
    """An alias table whose entries are the strings themselves (entry id = string id)."""
    if len(set(strings)) != len(strings):
        raise ValueError("alias strings must be unique")
    return AliasTable({s: i for i, s in enumerate(strings)}, [(i,) for i in range(len(strings))])


def map_digest(strings: Sequence[str], string_entry: np.ndarray) -> str:
    """sha256 of an active alias → entry map (the linker a view was resolved with)."""
    active = sorted((strings[i], int(e)) for i, e in enumerate(string_entry.tolist()) if e >= 0)
    return hashlib.sha256(json.dumps({"aliases": active, "version": LINKER_VERSION}).encode()).hexdigest()


def _init_worker(tokenizer_name: str, table_path: str, min_subtokens: int, normalization: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    with open(table_path, "rb") as handle:
        table = pickle.load(handle)
    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    _WORKER["linker"] = CausalLinker(table, min_subtokens=min_subtokens)
    _WORKER["normalization"] = normalization or None


def _scan_batch(texts: list[str]) -> list[tuple[list[int], list[tuple[int, int, int, int]]] | None]:
    """Per text: (token ids, all matches), or None when the tokenizer corrupted it (as `build_corpus`)."""
    tokenizer, linker, normalization = _WORKER["tokenizer"], _WORKER["linker"], _WORKER["normalization"]
    results: list[tuple[list[int], list[tuple[int, int, int, int]]] | None] = []
    vocabulary = len(tokenizer)
    for text in texts:
        try:
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        except BaseException as error:   # the Rust tokenizer can panic on rare inputs
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            results.append(None); continue
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        expected = unicodedata.normalize(normalization, text) if normalization else text
        if ids and (max(ids) >= vocabulary or tokenizer.decode(ids) != expected):
            results.append(None); continue
        results.append((ids, linker.all_matches(text, offsets)))
    return results


def _scan(texts: Iterable[str], *, tokenizer_name: str, table: AliasTable, min_subtokens: int, normalization: str | None,
          workers: int, batch_texts: int, scratch: Path) -> Iterator[tuple[str, tuple[list[int], list] | None]]:
    """(text, scan result) in input order, scanned by `workers` spawned processes."""
    scratch.mkdir(parents=True, exist_ok=True)
    table_path = scratch / ".match_table.pkl"
    with table_path.open("wb") as handle:
        pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def batches() -> Iterator[list[str]]:
        chunk: list[str] = []
        for text in texts:
            chunk.append(text)
            if len(chunk) == batch_texts:
                yield chunk; chunk = []
        if chunk:
            yield chunk

    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init_worker,
                                 initargs=(tokenizer_name, str(table_path), min_subtokens, normalization or "")) as pool:
            pending: deque = deque()
            for chunk in batches():
                pending.append((chunk, pool.submit(_scan_batch, chunk)))
                if len(pending) >= 2 * workers:
                    done, future = pending.popleft()
                    yield from zip(done, future.result())
            while pending:
                done, future = pending.popleft()
                yield from zip(done, future.result())
    finally:
        table_path.unlink(missing_ok=True)


def _save_matches(path: Path, parts: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    arrays = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in parts.items()}
    order = np.lexsort((arrays["string"], arrays["token"])) if arrays["token"].size else np.zeros(0, dtype=np.int64)
    compact = {k: arrays[k][order].astype(np.int32) for k in MATCH_FIELDS}
    np.savez(path, **compact)
    return compact


def build_match_corpus(texts: Iterable[str], out_dir: Path, *, tokenizer_name: str, strings: Sequence[str], eos_id: int,
                       max_tokens: int, min_subtokens: int = 2, vocab_size: int = 50257, normalization: str | None = None,
                       workers: int = 4, batch_texts: int = 128, keep_texts: bool = True,
                       extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tokenize documents until `max_tokens` and record every alias match of `strings`."""
    from transformers import AutoTokenizer
    from .corpus import tokenizer_fingerprint
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    table = strings_table(list(strings))
    parts: dict[str, list[np.ndarray]] = {k: [] for k in MATCH_FIELDS}
    documents: list[int] = []
    position = skipped = 0
    text_handle = (out_dir / "texts.jsonl").open("w") if keep_texts else None
    try:
        with (out_dir / "tokens.bin").open("wb") as handle:
            for text, result in _scan(texts, tokenizer_name=tokenizer_name, table=table, min_subtokens=min_subtokens,
                                      normalization=normalization, workers=workers, batch_texts=batch_texts, scratch=out_dir):
                if position >= max_tokens:
                    break
                if result is None:
                    skipped += 1; continue
                ids, matches = result
                block = np.asarray([*ids, eos_id], dtype=np.int64)
                if int(block.max(initial=0)) >= vocab_size:
                    raise ValueError(f"token id ≥ vocab_size {vocab_size}")
                handle.write(block.astype(dtype).tobytes())
                documents.append(position)
                if matches:
                    rows = np.asarray(matches, dtype=np.int64)
                    parts["token"].append(rows[:, 0] + position); parts["first"].append(rows[:, 1] + position)
                    parts["string"].append(rows[:, 2]); parts["chars"].append(rows[:, 3])
                if text_handle is not None:
                    text_handle.write(json.dumps({"text": text}) + "\n")
                position += block.size
    finally:
        if text_handle is not None:
            text_handle.close()
    if position >= 2**31:
        raise ValueError("corpus too large for int32 positions")
    matches = _save_matches(out_dir / "matches.npz", parts)
    np.save(out_dir / "documents.npy", np.asarray(documents, dtype=np.int64))
    (out_dir / "strings.json").write_text(json.dumps(list(strings)) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    manifest = {"tokens": int(position), "documents": len(documents), "skipped_documents": skipped, "dtype": np.dtype(dtype).name,
                "tokenizer": tokenizer_name, "tokenizer_sha256": tokenizer_fingerprint(tokenizer), "eos_id": eos_id,
                "min_subtokens": min_subtokens, "boundary": "prefix", "linker_version": LINKER_VERSION,
                "strings": len(strings), "matches": int(matches["token"].size), "texts": keep_texts,
                **({"normalization": normalization} if normalization else {}), **(extra_manifest or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class MatchCorpus:
    path: Path
    tokens: np.ndarray
    documents: np.ndarray
    matches: dict[str, np.ndarray]
    strings: list[str]
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: Path) -> "MatchCorpus":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        tokens = np.memmap(path / "tokens.bin", dtype=np.uint16 if manifest["dtype"] == "uint16" else np.uint32, mode="r")
        with np.load(path / "matches.npz") as data:
            matches = {k: data[k] for k in MATCH_FIELDS}
        return cls(path, tokens, np.load(path / "documents.npy"), matches, json.loads((path / "strings.json").read_text()), manifest)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def string_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.strings)}

    def texts(self) -> Iterator[str]:
        with (self.path / "texts.jsonl").open() as handle:
            for line in handle:
                yield json.loads(line)["text"]

    def document_of(self, positions: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.documents, positions, side="right") - 1


def add_matches(path: Path, new_strings: Sequence[str], *, workers: int = 4, batch_texts: int = 128) -> dict[str, int]:
    """Scan the kept texts again for `new_strings` (those not yet present) and append their matches;
    the re-tokenized ids must equal the stored tokens. Returns string → id for every requested string."""
    corpus = MatchCorpus.open(path)
    index = corpus.string_index()
    fresh = [s for s in dict.fromkeys(new_strings) if s not in index]
    if fresh:
        if not corpus.manifest.get("texts"):
            raise ValueError(f"{path} kept no texts; rebuild it with keep_texts=True")
        table = strings_table(fresh)
        offset = len(corpus.strings)
        parts: dict[str, list[np.ndarray]] = {k: [corpus.matches[k].astype(np.int64)] for k in MATCH_FIELDS}
        document = 0
        for _, result in _scan(corpus.texts(), tokenizer_name=corpus.manifest["tokenizer"], table=table,
                               min_subtokens=int(corpus.manifest["min_subtokens"]),
                               normalization=corpus.manifest.get("normalization"), workers=workers,
                               batch_texts=batch_texts, scratch=path):
            if result is None:
                raise ValueError(f"document {document} no longer tokenizes cleanly")
            start = int(corpus.documents[document])
            ids, matches = result
            stored = np.asarray(corpus.tokens[start:start + len(ids)], dtype=np.int64)
            if stored.size != len(ids) or not np.array_equal(stored, np.asarray(ids, dtype=np.int64)):
                raise ValueError(f"document {document} re-tokenizes differently from the stored tokens")
            if matches:
                rows = np.asarray(matches, dtype=np.int64)
                parts["token"].append(rows[:, 0] + start); parts["first"].append(rows[:, 1] + start)
                parts["string"].append(rows[:, 2] + offset); parts["chars"].append(rows[:, 3])
            document += 1
        if document != corpus.documents.size:
            raise ValueError(f"{path}: {document} texts for {corpus.documents.size} documents")
        matches = _save_matches(path / "matches.npz", parts)
        strings = corpus.strings + fresh
        (path / "strings.json").write_text(json.dumps(strings) + "\n")
        manifest = {**corpus.manifest, "strings": len(strings), "matches": int(matches["token"].size)}
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        index = {s: i for i, s in enumerate(strings)}
    return {s: index[s] for s in new_strings}


def select_longest(matches: dict[str, np.ndarray], string_entry: np.ndarray, *, min_subtokens: int = 2) -> dict[str, np.ndarray]:
    """Spans of the linker `string_entry` (string id → entry, −1 inactive), as `CausalLinker.link`
    (prefix boundary) would produce them: per token the active match with the most characters,
    kept when it covers ≥ `min_subtokens` subtokens. Sorted by injection position."""
    entry = np.asarray(string_entry, dtype=np.int64)[matches["string"]] if matches["string"].size else np.zeros(0, np.int64)
    keep = entry >= 0
    token, first, chars, entry = (matches["token"][keep].astype(np.int64), matches["first"][keep].astype(np.int64),
                                  matches["chars"][keep].astype(np.int64), entry[keep])
    order = np.lexsort((-chars, token))
    token, first, entry = token[order], first[order], entry[order]
    _, winners = np.unique(token, return_index=True)
    token, first, entry = token[winners], first[winners], entry[winners]
    length = token - first + 1
    ok = length >= min_subtokens
    return {"start": first[ok], "end": token[ok], "inject": token[ok], "entry": entry[ok], "length": length[ok]}


def write_view(corpus: MatchCorpus, out_dir: Path, string_entry: np.ndarray, entry_confidence: np.ndarray, *,
               min_subtokens: int = 2, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """A `TokenCorpus` directory for one linker: tokens.bin symlinked to the match corpus, resolved spans."""
    if min_subtokens < int(corpus.manifest["min_subtokens"]):
        raise ValueError(f"the match corpus stores matches of ≥ {corpus.manifest['min_subtokens']} subtokens only")
    out_dir.mkdir(parents=True, exist_ok=True)
    spans = select_longest(corpus.matches, string_entry, min_subtokens=min_subtokens)
    link = out_dir / "tokens.bin"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((corpus.path / "tokens.bin").resolve())
    confidence = np.asarray(entry_confidence, dtype=np.float32)[spans["entry"]] if spans["entry"].size else np.zeros(0, np.float32)
    compact = {"start": np.int32, "end": np.int32, "inject": np.int32, "entry": np.int32, "length": np.uint8}
    np.savez(out_dir / "spans.npz", **{k: spans[k].astype(t) for k, t in compact.items()},
             confidence=confidence.astype(np.float16))
    manifest = {"tokens": len(corpus), "documents": int(corpus.documents.size), "dtype": corpus.manifest["dtype"],
                "tokenizer": corpus.manifest["tokenizer"], "tokenizer_sha256": corpus.manifest.get("tokenizer_sha256"),
                "alias_table_sha256": map_digest(corpus.strings, np.asarray(string_entry)), "boundary": "prefix",
                "spans": int(spans["entry"].size), "eos_id": corpus.manifest["eos_id"], "min_subtokens_stored": min_subtokens,
                "match_corpus": str(corpus.path.resolve()), **(extra_manifest or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def concatenate(parts: Sequence[tuple[MatchCorpus, int, int]], out_dir: Path, *, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """A match corpus made of whole documents of others: `parts` = (corpus, documents to take, repeats).
    Every part must share the same alias strings (same ids)."""
    if not parts:
        raise ValueError("nothing to concatenate")
    strings = parts[0][0].strings
    if any(p[0].strings != strings for p in parts):
        raise ValueError("parts have different alias strings; add the same strings to every part first")
    dtype = parts[0][0].manifest["dtype"]
    if any(p[0].manifest["dtype"] != dtype or p[0].manifest["tokenizer"] != parts[0][0].manifest["tokenizer"] for p in parts):
        raise ValueError("parts use different tokenizers or token types")
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[np.ndarray]] = {k: [] for k in MATCH_FIELDS}
    documents: list[np.ndarray] = []
    position = 0
    sources = []
    with (out_dir / "tokens.bin").open("wb") as handle:
        for corpus, count, repeats in parts:
            count = min(int(count), int(corpus.documents.size))
            end = int(corpus.documents[count]) if count < corpus.documents.size else len(corpus)
            block = np.asarray(corpus.tokens[:end])
            inside = corpus.matches["token"] < end
            for _ in range(int(repeats)):
                handle.write(block.tobytes())
                documents.append(corpus.documents[:count] + position)
                for key in MATCH_FIELDS:
                    values = corpus.matches[key][inside].astype(np.int64)
                    out[key].append(values + position if key in ("token", "first") else values)
                position += end
            sources.append({"corpus": str(corpus.path.resolve()), "documents": count, "tokens": end, "repeats": int(repeats)})
    matches = _save_matches(out_dir / "matches.npz", out)
    np.save(out_dir / "documents.npy", np.concatenate(documents) if documents else np.zeros(0, np.int64))
    (out_dir / "strings.json").write_text(json.dumps(strings) + "\n")
    base = parts[0][0].manifest
    manifest = {**{k: base[k] for k in ("dtype", "tokenizer", "tokenizer_sha256", "eos_id", "min_subtokens", "boundary",
                                        "linker_version") if k in base},
                "tokens": position, "documents": int(sum(d.size for d in documents)), "strings": len(strings),
                "matches": int(matches["token"].size), "texts": False, "sources": sources, **(extra_manifest or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def documents_for_tokens(corpus: MatchCorpus, tokens: int) -> int:
    """The smallest number of leading documents holding at least `tokens` tokens (or all of them)."""
    ends = np.append(corpus.documents[1:], len(corpus))
    return int(min(corpus.documents.size, np.searchsorted(ends, tokens, side="left") + 1))
