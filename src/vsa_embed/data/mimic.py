"""MIMIC-III clinical notes (`NOTEEVENTS.csv.gz`, PhysioNet credentialed) for the licensed clinical track T1c.

**DUA.** MIMIC is credentialed data (decision 58): the extracted notes live only under `~/data/vsa-llm/t1c/`
(mode 700); nothing derived from them is committed, and nothing here prints note text or individual rows. The
statistics this module returns are aggregates (counts, characters, category histograms).

`extract_notes` reads the CSV once and writes Parquet shards (`notes-<split>-<shard>.parquet`):

- rows with `ISERROR` set are dropped, and notes that are empty after stripping;
- the **patient split**: `SUBJECT_ID`'s sha256 bucket < `eval_buckets` / 10,000 → `eval` (never trained on),
  else `train`, so no patient has notes on both sides;
- a deterministic shuffle: a note's shard and its order within the shard come from the sha256 of its `ROW_ID`,
  so any prefix of the stream (the holdout pre-sample, a capped evaluation set) is a uniform sample over note
  categories instead of the file's category-grouped order.

`iter_notes` streams the texts of one split in shard order.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

NOTE_COLUMNS = ("ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "ISERROR", "TEXT")
SPLITS = ("train", "eval")
BUCKETS = 10_000


def _hash64(value: int, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{int(value)}".encode()).digest()[:8], "big")


def subject_bucket(subject_id: int, buckets: int = BUCKETS) -> int:
    """Stable hash bucket of a patient (the train/eval split never depends on file order)."""
    return _hash64(subject_id, "subject") % buckets


def note_order(row_id: int) -> int:
    """64-bit shuffle key of a note."""
    return _hash64(row_id, "note")


def shard_path(out_dir: Path, split: str, shard: int) -> Path:
    return Path(out_dir) / f"notes-{split}-{shard:03d}.parquet"


def extract_notes(source: Path, out_dir: Path, *, eval_buckets: int, shards: int = 32, chunk_rows: int = 50_000,
                  limit_rows: int | None = None) -> dict[str, Any]:
    """CSV → shuffled Parquet shards per split; returns (and writes `extract.json`) aggregate statistics."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("row_id", pa.int64()), ("subject_id", pa.int64()), ("hadm_id", pa.int64()),
                        ("category", pa.string()), ("order", pa.uint64()), ("text", pa.large_string())])
    staging = {(split, s): out_dir / f".staging-{split}-{s:03d}.parquet" for split in SPLITS for s in range(shards)}
    writers: dict[tuple[str, int], Any] = {}
    stats: Counter[str] = Counter()
    categories: dict[str, Counter] = {split: Counter() for split in SPLITS}
    category_chars: dict[str, Counter] = {split: Counter() for split in SPLITS}
    subjects: dict[str, set[int]] = {split: set() for split in SPLITS}
    reader = pd.read_csv(source, usecols=list(NOTE_COLUMNS), chunksize=chunk_rows, dtype={"CATEGORY": str, "TEXT": str},
                         keep_default_na=True, nrows=limit_rows)
    try:
        for chunk in reader:
            stats["rows"] += len(chunk)
            error = chunk["ISERROR"].fillna(0).astype(float) != 0
            stats["dropped_iserror"] += int(error.sum())
            chunk = chunk[~error]
            text = chunk["TEXT"].fillna("").str.strip()
            empty = text.str.len() == 0
            stats["dropped_empty"] += int(empty.sum())
            chunk, text = chunk[~empty], text[~empty]
            rows: dict[tuple[str, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for row_id, subject, hadm, category, body in zip(chunk["ROW_ID"].tolist(), chunk["SUBJECT_ID"].tolist(),
                                                             chunk["HADM_ID"].tolist(), chunk["CATEGORY"].tolist(), text.tolist()):
                split = "eval" if subject_bucket(int(subject)) < eval_buckets else "train"
                order = note_order(int(row_id))
                key = (split, order % shards)
                part = rows[key]
                part["row_id"].append(int(row_id)); part["subject_id"].append(int(subject))
                part["hadm_id"].append(-1 if hadm != hadm else int(hadm))
                category = (category or "").strip() or "unknown"
                part["category"].append(category); part["order"].append(order); part["text"].append(body)
                categories[split][category] += 1
                category_chars[split][category] += len(body)
                subjects[split].add(int(subject))
            for key, part in rows.items():
                if key not in writers:
                    writers[key] = pq.ParquetWriter(staging[key], schema)
                writers[key].write_table(pa.table(dict(part), schema=schema))
    finally:
        for writer in writers.values():
            writer.close()
    # Order within each shard by the shuffle key.
    for (split, s), path in staging.items():
        target = shard_path(out_dir, split, s)
        if path.exists():
            table = pq.read_table(path)
            table = table.take(pc.sort_indices(table, sort_keys=[("order", "ascending")]))
            pq.write_table(table, target, row_group_size=4096)
            path.unlink()
        else:
            pq.write_table(pa.table({n: [] for n in schema.names}, schema=schema), target)
    overlap = len(subjects["train"] & subjects["eval"])
    if overlap:
        raise AssertionError(f"{overlap} patients on both sides of the split")
    result = {
        "source": str(source), "eval_buckets": eval_buckets, "buckets": BUCKETS, "shards": shards,
        "rows": stats["rows"], "dropped_iserror": stats["dropped_iserror"], "dropped_empty": stats["dropped_empty"],
        "notes": {split: sum(categories[split].values()) for split in SPLITS},
        "patients": {split: len(subjects[split]) for split in SPLITS}, "patients_on_both_sides": overlap,
        "characters": {split: sum(category_chars[split].values()) for split in SPLITS},
        "categories": {split: dict(sorted(categories[split].items())) for split in SPLITS},
        "category_characters": {split: dict(sorted(category_chars[split].items())) for split in SPLITS},
        "limit_rows": limit_rows,
    }
    (out_dir / "extract.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def iter_notes(notes_dir: Path, split: str, *, shards: int | None = None, limit: int | None = None,
               categories: frozenset[str] | None = None, exclude_subjects: frozenset[int] | None = None,
               admissions: frozenset[int] | None = None, keep: Callable[[str], bool] | None = None,
               log: Counter | None = None) -> Iterator[str]:
    """Texts of `split` in shard order (shard 0 first), optionally restricted to note categories.

    Opt-in filters (T1c-ROOD; all None leaves the stream unchanged): `exclude_subjects` drops every note of these
    patients (`SUBJECT_ID`, any admission or none), `admissions` keeps only notes of these `HADM_ID`s, `keep(text)`
    drops a note when it returns False (a mention filter). `log` (a Counter) receives aggregate counts of what each
    filter dropped and how many notes were yielded; `limit` counts yielded notes."""
    import pyarrow.parquet as pq

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    notes_dir = Path(notes_dir)
    if shards is None:
        shards = int(json.loads((notes_dir / "extract.json").read_text())["shards"])
    produced = 0
    columns = ["text"] + (["category"] if categories else []) + (["subject_id"] if exclude_subjects is not None else []) \
        + (["hadm_id"] if admissions is not None else [])
    for s in range(shards):
        parquet = pq.ParquetFile(shard_path(notes_dir, split, s))   # keep referenced while iterating
        for batch in parquet.iter_batches(columns=columns, batch_size=2048):
            data = batch.to_pydict()
            for i, text in enumerate(data["text"]):
                if categories and data["category"][i] not in categories:
                    continue
                if exclude_subjects is not None and int(data["subject_id"][i]) in exclude_subjects:
                    if log is not None:
                        log["dropped_excluded_subject"] += 1
                    continue
                if admissions is not None and int(data["hadm_id"][i]) not in admissions:
                    continue
                if keep is not None and not keep(text):
                    if log is not None:
                        log["dropped_by_filter"] += 1
                    continue
                if limit is not None and produced >= limit:
                    return
                produced += 1
                if log is not None:
                    log["yielded"] += 1
                yield text


def notes_signature(notes_dir: Path) -> str:
    """Fingerprint of an extraction (its statistics file): guards corpus reuse across extractions."""
    return hashlib.sha256((Path(notes_dir) / "extract.json").read_bytes()).hexdigest()
