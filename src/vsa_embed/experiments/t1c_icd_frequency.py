"""T1c-F: frequency bias and never-trained codes in ICD-9 coding of MIMIC-III discharge summaries.

HRRBERT's frequency-bias tests (7 log-frequency bins, top-k per bin, never-seen codes, t-SNE coloured by frequency)
carried to a text task on the T1c track: a label-wise attention head on a frozen host (T1c P0 now; C0′ and C5 once
trained) whose label parameters come from a **code vector** per condition. Preregistration:
`experiments/t1c-clinical/icd-frequency/preregistration.md`; library: `vsa_embed.icd_coding`.

**Licence and DUA (non-negotiable).** MIMIC-III is PhysioNet credentialed data; SNOMED CT is licensed; ICD code titles
belong to the credentialed context. Everything derived from them — code and holdout lists, frames, titles, tokens,
host states, scores, ranks, code vectors, t-SNE coordinates and figures — is written under `paths.data_root`
(`~/data/vsa-llm/t1c/icd-frequency-v1/`, mode 700). The committed run folders (`paths.runs`) receive aggregates only
(counts, digests, metrics per condition / bin, bootstrap summaries); no stage prints a note, a row, a code, a title or a
concept. Nothing is sent to any external service.

Stages (`python -m vsa_embed.experiments.t1c_icd_frequency <stage> --config experiments/t1c-clinical/icd-frequency/icd-frequency.yaml`):

- `prepare` (CPU, ≈ 2 min): discharge-summary admissions (T1c patient split; dev = training-side patients in
  `split.dev_buckets`), their ICD-9 diagnosis codes, the ICD-9 → SNOMED CT map (NLM 1-to-1, else the maximal concepts
  of the 1-to-many map), the code frames from the T1c ontology, HRRBERT's frequency bins, the frozen code-level
  holdout (hashed, stratified by bin; pinned by `holdout.expected_sha256`), GRAM paths and titles.
- `tokenize` (CPU, ≈ 15 min): SmolLM2 tokens of every admission's concatenated discharge summaries (first
  `text.max_tokens`), the E9 T1c evaluation linker's spans (used by the C5 encoder) and the code titles.
- `kge` (CPU, ≈ 10 min): TransE (E9's C6g trainer) on the frames of the label concepts and of the atomic concepts.
- `encode --encoder NAME (--pretrained HF_ID | --run E9_RUN_DIR [--channel on|off])` (GPU): segment-pooled
  last-layer states of every admission (`text.chunk_tokens` windows, `text.segment_tokens` segments) and the code
  titles' mean-pooled states, through the frozen host.
- `train --encoder NAME --seed S [--conditions ...] [--c5-run DIR]` (GPU): one head per condition, trained on
  identical batches; scores of the evaluation admissions × all labels, of all admissions × never-trained labels, ranks
  of never-trained positives, code vectors.
- `analyze --encoder NAME --seeds 1 2 3` (GPU or CPU): endpoints, per-bin tables, two-way bootstrap (admissions ×
  codes, seeds resampled), Holm and Dunnett, frequency probes and t-SNE.
- `plan`: the queue commands (printed, never queued).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

from vsa_embed import icd_coding as ic

SCTID = re.compile(r"^\d{6,18}$")
SPLITS = ("train", "dev", "eval")
DEFAULT_CONFIG = Path("experiments/t1c-clinical/icd-frequency/icd-frequency.yaml")
CONDITIONS = ("free", "composed_head", "composed_c5", "transe", "title", "random", "gram", "composed_free")
CONTROL = "free"
PRIMARY = "composed_head"


# -- configuration and paths ------------------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    from .t1_open_corpus import load_config as load_track_config
    config = yaml.safe_load(Path(path).read_text())
    config["_t1c"] = load_track_config(Path(config["paths"]["t1c_config"]))
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """The run config without the embedded T1c config (recorded by its path)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def private_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def data_root(config: dict[str, Any]) -> Path:
    return private_dir(Path(config["paths"]["data_root"]))


def licensed_root(config: dict[str, Any]) -> Path:
    return Path(config["_t1c"]["paths"]["licensed_root"]).expanduser()


def run_folder(config: dict[str, Any], name: str) -> Path:
    path = Path(config["paths"]["runs"]) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_native) + "\n")


def _native(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def finite(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def record_run(folder: Path, config: dict[str, Any], **extra: Any) -> None:
    from vsa_embed.provenance import write_run_metadata
    write_run_metadata(folder, public_config(config), **extra)


def split_of(bucket: int, eval_buckets: int, dev: Sequence[int]) -> str:
    if bucket < eval_buckets:
        return "eval"
    return "dev" if int(dev[0]) <= bucket < int(dev[1]) else "train"


# -- prepare ------------------------------------------------------------------------------------------------------

def read_icd9_maps(directory: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    """NLM ICD9CM_SNOMED_MAP: code (no dot) → SNOMED CT ids, from the 1-to-1 and 1-to-many files (columns by header)."""
    one: dict[str, set[str]] = defaultdict(set)
    many: dict[str, set[str]] = defaultdict(set)
    rows = Counter()
    for name, target in (("1TO1", one), ("1TOM", many)):
        paths = sorted(Path(directory).glob(f"ICD9CM_SNOMED_MAP_{name}_*.txt"))
        if not paths:
            raise FileNotFoundError(f"no ICD9CM_SNOMED_MAP_{name} file under {directory}")
        for path in paths:
            with open(path, encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                missing = {"ICD_CODE", "SNOMED_CID"} - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{path.name}: columns {sorted(missing)} missing")
                for row in reader:
                    code = (row["ICD_CODE"] or "").strip().replace(".", "")
                    concept = (row["SNOMED_CID"] or "").strip()
                    rows[name] += 1
                    if code and SCTID.match(concept):
                        target[code].add(concept)
    return dict(one), dict(many), {"rows": dict(rows), "codes_1to1": len(one), "codes_1tom": len(many),
                                   "codes_in_both": len(set(one) & set(many))}


def discharge_notes(notes_dir: Path, category: str) -> dict[int, dict[str, Any]]:
    """Admission → subject, note split (from the T1c extraction's shard name) and the discharge-summary note ids."""
    import pyarrow.parquet as pq
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(notes_dir).glob("notes-*-*.parquet")):
        split = path.name.split("-")[1]
        table = pq.read_table(path, columns=["row_id", "subject_id", "hadm_id", "category"]).to_pydict()
        for row_id, subject, hadm, kind in zip(table["row_id"], table["subject_id"], table["hadm_id"], table["category"]):
            if kind != category or hadm < 0:
                continue
            entry = out.setdefault(int(hadm), {"subject": int(subject), "note_split": split, "rows": []})
            if entry["subject"] != int(subject) or entry["note_split"] != split:
                raise ValueError("an admission's discharge summaries disagree on the patient")
            entry["rows"].append(int(row_id))
    for entry in out.values():
        entry["rows"].sort()
    return out


def read_diagnoses(path: Path, admissions: dict[int, dict[str, Any]]) -> tuple[dict[int, set[str]], dict[str, int]]:
    codes: dict[int, set[str]] = defaultdict(set)
    stats = Counter()
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            if not row["HADM_ID"] or not row["ICD9_CODE"]:
                stats["rows_without_admission_or_code"] += 1
                continue
            hadm = int(row["HADM_ID"])
            entry = admissions.get(hadm)
            if entry is None:
                continue
            if int(row["SUBJECT_ID"]) != entry["subject"]:
                stats["subject_mismatch"] += 1
                continue
            codes[hadm].add(row["ICD9_CODE"].strip())
            stats["rows_used"] += 1
    return dict(codes), dict(stats)


def read_titles(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("ICD9_CODE") or "").strip()
            title = (row.get("LONG_TITLE") or row.get("SHORT_TITLE") or "").strip()
            if code and title:
                titles[code] = title
    return titles


def ancestor_function(parents: dict[str, list[str]]):
    cache: dict[str, set[str]] = {}

    def ancestors(concept: str) -> set[str]:
        if concept in cache:
            return cache[concept]
        out: set[str] = set()
        stack = list(parents.get(concept, ()))
        while stack:
            parent = stack.pop()
            if parent not in out:
                out.add(parent)
                stack.extend(parents.get(parent, ()))
        cache[concept] = out
        return out
    return ancestors


def label_space(codes_by_admission: dict[int, set[str]], split: dict[int, str], mapping: dict[str, tuple[str, list[str]]],
                *, edges: Sequence[float], holdout_config: dict[str, Any]) -> dict[str, Any]:
    """Counts per split, frequency bins, the holdout and the label order (trained, held-out, naturally unseen) of the
    framed codes (codes with at least one mapped concept). Pure function of its inputs (unit-tested)."""
    counts = {s: Counter() for s in SPLITS}
    for hadm, codes in codes_by_admission.items():
        counts[split[hadm]].update(codes)
    total = int(sum(counts["train"].values()))
    framed = sorted(c for c in set().union(*[set(counts[s]) for s in SPLITS]) if c in mapping)
    holdout = ic.choose_code_holdout(framed, counts["train"], total, eligible_bins=holdout_config["bins"],
                                     fraction=float(holdout_config["fraction"]), min_count=int(holdout_config["min_count"]),
                                     salt=str(holdout_config["salt"]), edges=edges)
    held = set(holdout["codes"])
    trained = [c for c in framed if counts["train"].get(c, 0) >= 1 and c not in held]
    natural = [c for c in framed if counts["train"].get(c, 0) == 0 and c not in held]
    order = trained + sorted(held) + natural
    return {"counts": counts, "total_train": total, "framed": framed, "holdout": holdout, "order": order,
            "n_trained": len(trained), "n_heldout": len(held), "n_natural": len(natural)}


def run_prepare(config: dict[str, Any]) -> dict[str, Any]:
    from .t1c_corpus import build_track_ontology
    from vsa_embed.data.mimic import subject_bucket
    from vsa_embed.provenance import git_state
    started = time.monotonic()
    git_at_start = git_state()
    t1c = config["_t1c"]
    root, licensed = data_root(config), licensed_root(config)
    sources, labels_cfg = config["sources"], config["labels"]
    edges = [float(e) for e in labels_cfg["frequency_edges"]]
    eval_buckets = int(t1c["mimic"]["eval_buckets"])
    dev = config["split"]["dev_buckets"]

    notes = discharge_notes(Path(t1c["paths"]["notes_dir"]).expanduser(), labels_cfg["note_category"])
    codes_by_admission, diagnosis_stats = read_diagnoses(licensed / sources["diagnoses"], notes)
    admissions = sorted(codes_by_admission)                    # discharge-summary admissions with ≥ 1 diagnosis code
    split = {}
    for hadm in admissions:
        s = split_of(subject_bucket(notes[hadm]["subject"]), eval_buckets, dev)
        if (s == "eval") != (notes[hadm]["note_split"] == "eval"):
            raise AssertionError("the T1c note split and the patient bucket disagree")
        split[hadm] = s

    # Ontology (T1c's, rebuilt from RF2) — must match the E9 ontology.pt the C5 composer was trained on.
    ontology = build_track_ontology(t1c["ontology"])
    saved = torch.load(Path(config["paths"]["ontology_pt"]).expanduser(), weights_only=False)
    for key in ("concept_names", "atomic_names", "relation_names"):
        if list(getattr(ontology, key)) != list(saved[key]):
            raise AssertionError(f"T1c ontology {key} differ from ontology.pt: rebuild mismatch")
    index = ontology.concept_index
    parents = {c: list(p) for c, p in zip(ontology.concept_names, ontology.metadata["parents"])}
    ancestors = ancestor_function(parents)
    track = set(ontology.concept_names)

    one, many, map_info = read_icd9_maps(licensed / sources["icd9_map_dir"])
    cap = int(labels_cfg["max_concepts"])
    mapping: dict[str, tuple[str, list[str]]] = {}
    for code in set().union(*codes_by_admission.values()):
        direct = sorted(one.get(code, set()) & track, key=int)
        if direct:
            mapping[code] = ("1to1", direct)
            continue
        several = many.get(code, set()) & track
        if several:
            mapping[code] = ("1toM", ic.maximal_concepts(several, ancestors, cap))

    space = label_space(codes_by_admission, split, mapping, edges=edges, holdout_config=config["holdout"])
    expected = config["holdout"].get("expected_sha256")
    if expected and expected != space["holdout"]["sha256"]:
        raise ValueError(f"code holdout sha256 {space['holdout']['sha256']} != pinned {expected}")
    order = space["order"]
    n_trained, n_held = space["n_trained"], space["n_heldout"]
    position = {c: i for i, c in enumerate(order)}
    counts, total = space["counts"], space["total_train"]
    count = {s: np.array([counts[s].get(c, 0) for c in order], dtype=np.int64) for s in SPLITS}
    logf = ic.log_frequency(count["train"], total)
    bins = ic.frequency_bins(logf, edges)
    trained = np.zeros(len(order), dtype=bool); trained[:n_trained] = True
    heldout = np.zeros(len(order), dtype=bool); heldout[n_trained:n_trained + n_held] = True
    natural = ~(trained | heldout)

    members = [mapping[c][1] for c in order]
    frames = [ic.union_frame(ontology.frames[index[m]] for m in ms) for ms in members]
    if any(not f for f in frames):
        raise AssertionError("a framed code has an empty frame")
    t1c_held = set(Path(config["paths"]["t1c_holdout"]).expanduser().read_text().split())
    t1c_member = np.array([bool(set(ms) & t1c_held) for ms in members])
    trained_sets = {frozenset(ms) for ms, t in zip(members, trained) if t}
    shares = np.array([frozenset(ms) in trained_sets for ms in members])

    # GRAM: own node + ICD-9-CM ancestors (specific → general).
    node_names: list[str] = []
    node_index: dict[str, int] = {}

    def node(name: str) -> int:
        if name not in node_index:
            node_index[name] = len(node_names); node_names.append(name)
        return node_index[name]
    gram_paths = [[node(f"code:{c}")] + [node(a) for a in reversed(ic.icd9_ancestors(c))] for c in order]

    titles = read_titles(licensed / sources["titles"])
    title_list = [titles.get(c, "") for c in order]

    # Frames of every concept TransE needs: the label concepts and the concepts named by atomics.
    needed = {m for ms in members for m in ms}
    needed |= {a.split(":", 1)[1] for a in ontology.atomic_names if a.startswith("sct:") and a.split(":", 1)[1] in index}
    concept_frames = {c: list(ontology.frames[index[c]]) for c in sorted(needed, key=int)}

    labels = {"version": config["version"], "codes": order, "kind": [mapping[c][0] for c in order], "members": members,
              "frames": frames, "count": count, "total_train": total, "logf": logf, "bin": bins, "edges": edges,
              "trained": trained, "heldout": heldout, "natural_unseen": natural, "n_trained": n_trained,
              "gram_nodes": node_names, "gram_paths": gram_paths, "titles": title_list, "t1c_heldout_member": t1c_member,
              "shares_member_set_with_trained": shares, "holdout": space["holdout"], "concept_frames": concept_frames,
              "atomic_count": len(ontology.atomic_names), "relation_count": len(ontology.relation_names)}
    hadm = np.array(admissions, dtype=np.int64)
    adm_split = np.array([split[h] for h in admissions])
    adm_labels = [np.array(sorted(position[c] for c in codes_by_admission[h] if c in position), dtype=np.int64)
                  for h in admissions]
    adm = {"hadm": hadm, "split": adm_split, "notes": [notes[h]["rows"] for h in admissions], "labels": adm_labels,
           "codes_per_admission": np.array([len(codes_by_admission[h]) for h in admissions], dtype=np.int64)}
    torch.save(labels, root / "labels.pt")
    torch.save(adm, root / "admissions.pt")
    (root / "holdout_codes.txt").write_text("\n".join(space["holdout"]["codes"]) + "\n")

    # -- aggregates only --
    def bin_rows(mask: np.ndarray) -> dict[str, Any]:
        rows = {}
        for i in range(len(edges) - 1):
            sel = mask & (bins == i)
            rows[ic.bin_name(i, edges)] = {"codes": int(sel.sum()), "train_positives": int(count["train"][sel].sum()),
                                          "eval_positives": int(count["eval"][sel].sum()),
                                          "codes_with_eval_positive": int((sel & (count["eval"] > 0)).sum())}
        return rows
    positives_all = count["train"] + count["dev"] + count["eval"]
    kinds = Counter(labels["kind"])
    summary = {
        "track": "clinical (SNOMED CT + MIMIC-III)", "task": "ICD-9 diagnosis coding of discharge summaries",
        "admissions": {s: int((adm_split == s).sum()) for s in SPLITS},
        "admissions_without_framed_label": {s: int(sum(1 for l, sp in zip(adm_labels, adm_split) if sp == s and l.size == 0))
                                            for s in SPLITS},
        "notes_per_admission_mean": float(np.mean([len(r) for r in adm["notes"]])),
        "diagnoses": diagnosis_stats, "icd9_map": map_info,
        "distinct_codes": int(len(set().union(*codes_by_admission.values()))), "framed_codes": len(order),
        "mapping_kinds": dict(kinds), "members_per_code_mean": float(np.mean([len(m) for m in members])),
        "edges_per_frame_mean": float(np.mean([len(f) for f in frames])),
        "total_train_label_occurrences": total,
        "eval_label_occurrences_framed_share": float(count["eval"].sum() / max(1, sum(counts["eval"].values()))),
        "labels": {"trained": n_trained, "heldout": n_held, "natural_unseen": int(natural.sum())},
        "bins_trained": bin_rows(trained), "bins_heldout_natural_frequency": bin_rows(heldout),
        "holdout": {k: v for k, v in space["holdout"].items() if k != "codes"},
        "heldout_positives": {"all_admissions": int(positives_all[heldout].sum()), "eval": int(count["eval"][heldout].sum()),
                              "train_removed": int(count["train"][heldout].sum()),
                              "share_of_train_labels_removed": float(count["train"][heldout].sum() / max(1, total))},
        "natural_unseen_positives": {"dev": int(count["dev"][natural].sum()), "eval": int(count["eval"][natural].sum())},
        "t1c_holdout_interaction": {"labels_with_t1c_heldout_member": int(t1c_member.sum()),
                                    "heldout_labels_with_t1c_heldout_member": int((t1c_member & heldout).sum()),
                                    "trained_labels_with_t1c_heldout_member": int((t1c_member & trained).sum())},
        "heldout_sharing_member_set_with_trained": int((shares & heldout).sum()),
        "heldout_by_kind": dict(Counter(k for k, h in zip(labels["kind"], heldout) if h)),
        "titles_missing": {"all": int(sum(1 for t in title_list if not t)),
                           "heldout": int(sum(1 for t, h in zip(title_list, heldout) if h and not t))},
        "gram": {"nodes": len(node_names), "path_length_mean": float(np.mean([len(p) for p in gram_paths]))},
        "concept_frames_for_transe": len(concept_frames),
        "seconds": round(time.monotonic() - started, 1), "data_root": str(root),
    }
    folder = run_folder(config, f"prepare-{config['version']}")
    write_json(folder / "summary.json", summary)
    record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="prepare")
    return summary


# -- tokenize -----------------------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _tokenize_worker(item: tuple[int, str]) -> tuple[int, np.ndarray, int, np.ndarray]:
    index, text = item
    tokenizer, linker, max_tokens, chunk = _WORKER["tokenizer"], _WORKER["linker"], _WORKER["max_tokens"], _WORKER["chunk"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    full = len(ids)
    ids, offsets = ids[:max_tokens], offsets[:max_tokens]
    spans = np.zeros((0, 6), dtype=np.float64)
    if linker is not None and ids:
        found = linker.link(text[:offsets[-1][1]], offsets)
        rows = [(s.start_token, s.end_token, s.inject_token, s.entry, s.confidence, s.length) for s in found
                if s.start_token // chunk == s.inject_token // chunk]
        if rows:
            spans = np.asarray(rows, dtype=np.float64)
    return index, np.asarray(ids, dtype=np.uint16), full, spans


def tokenize_texts(texts: Sequence[str], *, tokenizer_name: str, linker: Any, max_tokens: int, chunk: int,
                   workers: int) -> dict[str, Any]:
    """Token ids (uint16, concatenated), offsets, untruncated lengths and linked spans (token positions within each
    text; spans crossing a chunk boundary dropped)."""
    import multiprocessing as mp
    from transformers import AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    if len(tokenizer) > 65535:
        raise ValueError("uint16 token storage needs a vocabulary below 65,536")
    _WORKER.update(tokenizer=tokenizer, linker=linker, max_tokens=max_tokens, chunk=chunk)
    results: list[Any] = [None] * len(texts)
    items = list(enumerate(texts))
    if workers > 1 and len(texts) > 64:
        with mp.get_context("fork").Pool(workers) as pool:
            for index, ids, full, spans in pool.imap_unordered(_tokenize_worker, items, chunksize=64):
                results[index] = (ids, full, spans)
    else:
        for item in items:
            index, ids, full, spans = _tokenize_worker(item)
            results[index] = (ids, full, spans)
    lengths = np.array([r[0].size for r in results], dtype=np.int64)
    offsets = np.zeros(len(texts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    span_rows = [np.column_stack([np.full(len(r[2]), i), r[2]]) for i, r in enumerate(results) if len(r[2])]
    spans = np.concatenate(span_rows) if span_rows else np.zeros((0, 7))
    return {"ids": np.concatenate([r[0] for r in results]) if results else np.zeros(0, np.uint16), "offsets": offsets,
            "full_lengths": np.array([r[1] for r in results], dtype=np.int64),
            "spans": {"text": spans[:, 0].astype(np.int64), "start": spans[:, 1].astype(np.int64),
                      "end": spans[:, 2].astype(np.int64), "inject": spans[:, 3].astype(np.int64),
                      "entry": spans[:, 4].astype(np.int64), "confidence": spans[:, 5].astype(np.float32),
                      "length": spans[:, 6].astype(np.int64)}}


def save_tokens(path: Path, tokens: dict[str, Any]) -> None:
    np.save(path.with_name(path.name + "-ids.npy"), tokens["ids"])
    np.savez(path.with_name(path.name + "-meta.npz"), offsets=tokens["offsets"], full_lengths=tokens["full_lengths"],
             **{f"span_{k}": v for k, v in tokens["spans"].items()})


def load_tokens(path: Path) -> dict[str, Any]:
    meta = np.load(path.with_name(path.name + "-meta.npz"))
    return {"ids": np.load(path.with_name(path.name + "-ids.npy"), mmap_mode="r"), "offsets": meta["offsets"],
            "full_lengths": meta["full_lengths"],
            "spans": {k[5:]: meta[k] for k in meta.files if k.startswith("span_")}}


def admission_texts(notes_dir: Path, rows: Sequence[Sequence[int]]) -> list[str]:
    import pyarrow.parquet as pq
    wanted = {r for group in rows for r in group}
    text: dict[int, str] = {}
    for path in sorted(Path(notes_dir).glob("notes-*-*.parquet")):
        table = pq.read_table(path, columns=["row_id", "text"])
        for row_id, body in zip(table.column("row_id").to_pylist(), table.column("text").to_pylist()):
            if row_id in wanted:
                text[row_id] = body
    missing = wanted - set(text)
    if missing:
        raise AssertionError(f"{len(missing)} discharge-summary notes missing from the extraction")
    return ["\n\n".join(text[r] for r in group) for group in rows]


def evaluation_linker(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from vsa_embed.evaluation.channel_probes import resolve_alias_table
    from vsa_embed.span_channel import CausalLinker
    ontology_path = Path(config["paths"]["ontology_pt"]).expanduser()
    ontology = torch.load(ontology_path, weights_only=False)
    table, info = resolve_alias_table(ontology, ontology_path, alias_table=Path(config["paths"]["alias_table"]).expanduser())
    linker = CausalLinker(table, boundary="prefix", min_subtokens=int(config["text"]["link_min_subtokens"]))
    return linker, {"alias_table_sha256": info["sha256"], "aliases": info["aliases"], "checks": info["checks"]}


def run_tokenize(config: dict[str, Any], *, limit: int | None = None) -> dict[str, Any]:
    from vsa_embed.provenance import git_state
    started = time.monotonic()
    git_at_start = git_state()
    root = data_root(config)
    text_cfg = config["text"]
    adm = torch.load(root / "admissions.pt", weights_only=False)
    labels = torch.load(root / "labels.pt", weights_only=False)
    rows = adm["notes"] if limit is None else adm["notes"][:limit]
    texts = admission_texts(Path(config["_t1c"]["paths"]["notes_dir"]).expanduser(), rows)
    linker, link_info = evaluation_linker(config)
    common = dict(tokenizer_name=text_cfg["tokenizer"], linker=linker, max_tokens=int(text_cfg["max_tokens"]),
                  chunk=int(text_cfg["chunk_tokens"]), workers=int(text_cfg["workers"]))
    notes = tokenize_texts(texts, **common)
    titles = tokenize_texts(labels["titles"], **{**common, "workers": 1})
    out = private_dir(root / ("tokens" if limit is None else f"tokens-limit{limit}"))
    save_tokens(out / "admissions", notes)
    save_tokens(out / "titles", titles)
    lengths = np.diff(notes["offsets"])
    chars = np.array([len(t) for t in texts], dtype=np.float64)
    summary = {
        "admissions": len(texts), "tokenizer": text_cfg["tokenizer"], "max_tokens": int(text_cfg["max_tokens"]),
        "tokens_kept": int(lengths.sum()), "tokens_full": int(notes["full_lengths"].sum()),
        "full_length_percentiles": {str(p): float(np.percentile(notes["full_lengths"], p)) for p in (10, 50, 90, 99)},
        "truncated_share": float((notes["full_lengths"] > int(text_cfg["max_tokens"])).mean()),
        "chars_per_token": float(chars.sum() / max(1, notes["full_lengths"].sum())),
        "spans": int(notes["spans"]["entry"].size), "spans_per_1k_tokens": float(1000 * notes["spans"]["entry"].size / max(1, lengths.sum())),
        "title_tokens_mean": float(np.diff(titles["offsets"]).mean()) if len(labels["titles"]) else 0.0,
        "title_spans": int(titles["spans"]["entry"].size), "linker": link_info,
        "seconds": round(time.monotonic() - started, 1), "limit": limit,
    }
    folder = run_folder(config, f"tokenize-{config['version']}" + ("" if limit is None else f"-limit{limit}"))
    write_json(folder / "summary.json", summary)
    record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="tokenize")
    return summary


# -- kge ----------------------------------------------------------------------------------------------------------

def kge_graph_for_labels(labels: dict[str, Any], atomic_names: Sequence[str]) -> tuple[dict[str, Any], dict[str, int]]:
    """TransE triples over concept frames: heads = the concepts with stored frames (label concepts and the concepts
    atomics name); an atomic `sct:X` is the entity of concept X when X has a frame, else an entity of its own."""
    frames = labels["concept_frames"]
    concepts = list(frames)
    entity = {c: i for i, c in enumerate(concepts)}
    atomic_entity, next_id, unified = [], len(concepts), 0
    for atom in atomic_names:
        value = atom.split(":", 1)[1] if atom.startswith("sct:") else None
        if value is not None and value in entity:
            atomic_entity.append(entity[value]); unified += 1
        else:
            atomic_entity.append(next_id); next_id += 1
    heads, relations, tails = [], [], []
    for c in concepts:
        for r, a in frames[c]:
            heads.append(entity[c]); relations.append(int(r)); tails.append(atomic_entity[int(a)])
    graph = {"heads": torch.tensor(heads), "relations": torch.tensor(relations), "tails": torch.tensor(tails),
             "entities": next_id, "unified_atomics": unified, "relation_count": int(labels["relation_count"])}
    return graph, entity


def run_kge(config: dict[str, Any], *, device: str = "cpu", steps: int | None = None) -> dict[str, Any]:
    from .e9_rowsource import train_transe
    from vsa_embed.provenance import git_state
    git_at_start = git_state()
    root = data_root(config)
    labels = torch.load(root / "labels.pt", weights_only=False)
    saved = torch.load(Path(config["paths"]["ontology_pt"]).expanduser(), weights_only=False)
    graph, entity = kge_graph_for_labels(labels, saved["atomic_names"])
    kge = config["kge"]
    steps = int(steps or kge["steps"])
    vectors, record = train_transe(graph, dimension=int(kge["dimension"]), batch=int(kge["batch"]),
                                   negatives=int(kge["negatives"]), min_steps=steps, max_steps=steps, seed=int(config["seed"]),
                                   device=device, log_every=0)
    label_vectors = torch.stack([vectors[[entity[m] for m in ms]].mean(0) for ms in labels["members"]])
    out = private_dir(root / "kge")
    torch.save({"vectors": label_vectors, "record": record}, out / f"transe-d{kge['dimension']}.pt")
    folder = run_folder(config, f"kge-{config['version']}")
    write_json(folder / "summary.json", {"transe": record, "labels": int(label_vectors.shape[0])})
    record_run(folder, config, git_at_start=git_at_start, device=device, stage="kge")
    return record


# -- encode -------------------------------------------------------------------------------------------------------

class HostEncoder:
    """A frozen host returning last-layer hidden states: a pretrained HF model (P0) or a trained E9 run (C0′, C5) with
    its span channel on or off."""

    def __init__(self, *, pretrained: str | None = None, run: Path | None = None, channel: bool = True,
                 device: torch.device | str = "cuda", dtype: str = "bfloat16", alias_table: Path | None = None) -> None:
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.channel = False
        if pretrained:
            from transformers import AutoModelForCausalLM
            from vsa_embed.training.lm import dtype_kwargs
            model = AutoModelForCausalLM.from_pretrained(pretrained, local_files_only=True, attn_implementation="sdpa",
                                                         **dtype_kwargs(self.dtype))
            self.base = model.model.to(self.device).eval()
            self.width = int(model.config.hidden_size)
            self.model = None
            self.info = {"pretrained": pretrained, "channel": False}
        else:
            from vsa_embed.evaluation.channel_probes import load_run
            # The explicit alias table: T1c's ontology.pt has no sidecar table (load_run would fall back to WordNet).
            adapter = load_run(Path(run), device=self.device, alias_table=alias_table)
            self.model = adapter.model.eval()
            self.base = None
            self.channel = bool(channel and self.model.channel is not None)
            self.width = int(self.model.model.get_input_embeddings().weight.shape[1])
            self.info = {"run": str(run), "channel": self.channel, "experiment": adapter.info.get("experiment"),
                         "channel_mode": adapter.info.get("channel")}
        self.info["width"] = self.width

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, spans: dict[str, torch.Tensor] | None) -> torch.Tensor:
        ids = ids.to(self.device)
        if self.base is not None:
            return self.base(input_ids=ids).last_hidden_state.float()
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            hidden = self.model.hidden_states(ids, spans=spans if self.channel else None)
        return hidden.float()


def chunk_plan(lengths: np.ndarray, chunk: int, segment: int) -> tuple[np.ndarray, np.ndarray]:
    """Segments per admission (Σ over its chunks of ⌈len / segment⌉) and the segment offsets."""
    counts = np.zeros(lengths.size, dtype=np.int64)
    for i, n in enumerate(lengths):
        full, rest = divmod(int(n), chunk)
        counts[i] = full * math.ceil(chunk / segment) + math.ceil(rest / segment)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return counts, offsets


def pool_segments(hidden: torch.Tensor, lengths: Sequence[int], segment: int) -> list[torch.Tensor]:
    """Mean over consecutive `segment`-token segments of each row's first `lengths[r]` positions."""
    b, t, d = hidden.shape
    pad = (-t) % segment
    if pad:
        hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad))
    mask = torch.arange(t + pad, device=hidden.device)[None, :] < torch.as_tensor(lengths, device=hidden.device)[:, None]
    sums = (hidden * mask[..., None]).view(b, -1, segment, d).sum(2)
    counts = mask.view(b, -1, segment).sum(2).clamp_min(1)
    means = sums / counts[..., None]
    return [means[r, :math.ceil(n / segment)] for r, n in enumerate(lengths)]


def span_index(spans: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Spans sorted by text, with per-text bounds, so a batch looks up only its own texts."""
    order = np.argsort(spans["text"], kind="stable")
    return {k: v[order] for k, v in spans.items()}


def _spans_of(indexed: dict[str, np.ndarray], text: int) -> dict[str, np.ndarray]:
    lo, hi = np.searchsorted(indexed["text"], [text, text + 1])
    return {k: v[lo:hi] for k, v in indexed.items()}


def encode_tokens(encoder: HostEncoder, tokens: dict[str, Any], *, chunk: int, segment: int, batch_chunks: int,
                  out: np.ndarray | None, seg_offsets: np.ndarray | None, start: int = 0, stop: int | None = None,
                  pooled: str = "segments", progress: Any = None) -> np.ndarray | None:
    """Run every text of `tokens[start:stop]` through the host in `chunk`-token windows (right-padded; a causal host's
    states at real positions do not depend on padding) and write segment means to `out` (or, `pooled="mean"`, return
    each text's mean state)."""
    offsets, ids_all = tokens["offsets"], tokens["ids"]
    indexed = span_index(tokens["spans"]) if encoder.channel else None
    stop = offsets.size - 1 if stop is None else stop
    means = np.zeros((stop - start, encoder.width), dtype=np.float32) if pooled == "mean" else None
    pending: list[tuple[int, int, int]] = []          # (text, chunk start, chunk end)

    def flush() -> None:
        if not pending:
            return
        lengths = [hi - lo for _, lo, hi in pending]
        width = max(lengths)
        batch = np.zeros((len(pending), width), dtype=np.int64)
        for row, (text, lo, hi) in enumerate(pending):
            base = offsets[text]
            batch[row, :hi - lo] = ids_all[base + lo:base + hi]
        spans = None
        if indexed is not None:
            spans = batch_spans_from_index(indexed, pending)
        hidden = encoder.hidden(torch.from_numpy(batch), spans)
        if pooled == "mean":
            for row, (text, lo, hi) in enumerate(pending):
                means[text - start] += hidden[row, :hi - lo].sum(0).cpu().numpy()
        else:
            per_chunk = math.ceil(chunk / segment)
            for row, part in enumerate(pool_segments(hidden, lengths, segment)):
                text, lo, _ = pending[row]
                first = seg_offsets[text] + (lo // chunk) * per_chunk
                out[first:first + part.shape[0]] = part.to(torch.float16).cpu().numpy()
        pending.clear()

    for text in range(start, stop):
        n = int(offsets[text + 1] - offsets[text])
        for lo in range(0, n, chunk):
            pending.append((text, lo, min(n, lo + chunk)))
            if len(pending) >= batch_chunks:
                flush()
        if progress is not None:
            progress(text)
    flush()
    if pooled == "mean":
        lengths = np.diff(offsets)[start:stop].astype(np.float32)
        nonzero = lengths > 0
        means[nonzero] /= lengths[nonzero, None]
        return means
    return None


def batch_spans_from_index(indexed: dict[str, np.ndarray], picks: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
    rows = []
    for row, (text, lo, hi) in enumerate(picks):
        own = _spans_of(indexed, text)
        sel = (own["start"] >= lo) & (own["inject"] < hi)
        for j in np.flatnonzero(sel):
            rows.append((row, own["start"][j] - lo, own["end"][j] - lo, own["inject"][j] - lo, own["entry"][j],
                         own["confidence"][j], own["length"][j]))
    if not rows:
        empty = torch.zeros(0, dtype=torch.long)
        return {"batch": empty, "start": empty, "end": empty, "inject": empty, "entry": empty,
                "confidence": torch.zeros(0), "length": empty}
    cols = list(zip(*rows))
    return {"batch": torch.tensor(cols[0]), "start": torch.tensor(cols[1]), "end": torch.tensor(cols[2]),
            "inject": torch.tensor(cols[3]), "entry": torch.tensor(cols[4]),
            "confidence": torch.tensor(cols[5], dtype=torch.float32), "length": torch.tensor(cols[6])}


def states_dir(config: dict[str, Any], encoder: str) -> Path:
    return data_root(config) / "states" / encoder


def open_store(config: dict[str, Any], encoder: str) -> ic.SegmentStore:
    folder = states_dir(config, encoder)
    meta = json.loads((folder / "meta.json").read_text())
    if not meta.get("complete"):
        raise RuntimeError(f"states of {encoder} are incomplete: run the encode stage (with --resume) first")
    offsets = np.load(folder / "seg_offsets.npy")
    segments = np.memmap(folder / "segments.f16", dtype=np.float16, mode="r", shape=(int(offsets[-1]), int(meta["width"])))
    return ic.SegmentStore(segments, offsets)


def run_encode(config: dict[str, Any], encoder_name: str, *, pretrained: str | None = None, run: Path | None = None,
               channel: bool = True, device: str = "cuda", limit: int | None = None, resume: bool = False,
               tokens_dir: str = "tokens") -> dict[str, Any]:
    from vsa_embed.provenance import git_state
    git_at_start = git_state()
    root = data_root(config)
    text_cfg, enc_cfg = config["text"], config["encode"]
    chunk, segment = int(text_cfg["chunk_tokens"]), int(text_cfg["segment_tokens"])
    tokens = load_tokens(root / tokens_dir / "admissions")
    titles = load_tokens(root / tokens_dir / "titles")
    n = tokens["offsets"].size - 1 if limit is None else min(limit, tokens["offsets"].size - 1)
    if limit is not None:
        tokens = {**tokens, "offsets": tokens["offsets"][:n + 1]}
    lengths = np.diff(tokens["offsets"])
    _, seg_offsets = chunk_plan(lengths, chunk, segment)
    folder = private_dir(states_dir(config, encoder_name))
    meta_path = folder / "meta.json"
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if previous and previous.get("complete") and not resume:
        raise FileExistsError(f"states of {encoder_name} exist; pass --resume to keep them")
    encoder = HostEncoder(pretrained=pretrained, run=run, channel=channel, device=device, dtype=enc_cfg["dtype"],
                          alias_table=Path(config["paths"]["alias_table"]).expanduser())
    shape = (int(seg_offsets[-1]), encoder.width)
    mode = "r+" if (resume and (folder / "segments.f16").exists()) else "w+"
    segments = np.memmap(folder / "segments.f16", dtype=np.float16, mode=mode, shape=shape)
    np.save(folder / "seg_offsets.npy", seg_offsets)
    done = int(previous.get("admissions_done", 0)) if (resume and previous) else 0
    meta = {"encoder": encoder_name, **encoder.info, "admissions": n, "segments": shape[0], "chunk_tokens": chunk,
            "segment_tokens": segment, "tokens": int(lengths.sum()), "admissions_done": done, "complete": False,
            "tokens_dir": tokens_dir}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 1024
    for begin in range(done, n, step):
        end = min(n, begin + step)
        encode_tokens(encoder, tokens, chunk=chunk, segment=segment, batch_chunks=int(enc_cfg["batch_chunks"]),
                      out=segments, seg_offsets=seg_offsets, start=begin, stop=end)
        segments.flush()
        meta["admissions_done"] = end
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        elapsed = time.monotonic() - started
        print(json.dumps({"admissions_done": end, "of": n, "seconds": round(elapsed, 1),
                          "tokens_per_second": round(float(lengths[done:end].sum()) / max(elapsed, 1e-9))}), flush=True)
    title_means = encode_tokens(encoder, titles, chunk=chunk, segment=segment, batch_chunks=64, out=None, seg_offsets=None,
                                pooled="mean")
    np.save(folder / "titles.npy", title_means)
    seconds = time.monotonic() - started
    meta.update(complete=True, seconds=round(seconds, 1),
                tokens_per_second=round(float(lengths[done:].sum()) / max(seconds, 1e-9), 1),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    summary = {k: v for k, v in meta.items() if k not in ("run",)}
    summary["run"] = str(run) if run else None
    out = run_folder(config, f"encode-{encoder_name}" + ("" if limit is None else f"-limit{limit}"))
    write_json(out / "summary.json", summary)
    record_run(out, config, git_at_start=git_at_start, device=device, stage="encode")
    return summary


# -- train --------------------------------------------------------------------------------------------------------

def load_c5_composer(run: Path, ontology_pt: Path):
    """The trained T1c C5 run's `FrameComposer` (its channel rebuilt from the run config and loaded from `final.pt`;
    the host weights are not needed)."""
    from vsa_embed.training.lm import build_channel, resolve_config
    from vsa_embed.evaluation.channel_probes import restore_composer_schedule
    state = torch.load(Path(run) / "final.pt", weights_only=False, map_location="cpu")
    cfg = resolve_config(state["config"])
    if cfg["channel"]["mode"] != "compose":
        raise ValueError(f"{run} is not a composition run")
    ontology = torch.load(ontology_pt, weights_only=False)
    width = int(state["model"]["model.model.embed_tokens.weight"].shape[1]) if "model.model.embed_tokens.weight" in state["model"] else 960
    channel, _ = build_channel(cfg, ontology, width)
    if state.get("composer_schedule"):
        restore_composer_schedule(channel.composer, state["composer_schedule"])
    own = {k[len("channel."):]: v for k, v in state["model"].items() if k.startswith("channel.")}
    missing, unexpected = channel.load_state_dict(own, strict=False)
    if [k for k in missing if k.startswith("composer.")] or unexpected:
        raise RuntimeError(f"C5 channel state does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return channel.composer.eval()


def build_sources(conditions: Sequence[str], labels: dict[str, Any], *, config: dict[str, Any], seed: int,
                  title_vectors: np.ndarray | None, transe_vectors: torch.Tensor | None, c5_composer: Any = None
                  ) -> dict[str, ic.LabelSource]:
    """One label source per condition (all `head.source_dim`-dimensional except the title encoder's host width)."""
    n, n_trained = len(labels["codes"]), int(labels["n_trained"])
    dim = int(config["head"]["source_dim"])
    comp = config["composer"]
    trained_ids = np.arange(n_trained)
    sources: dict[str, ic.LabelSource] = {}
    for offset, name in enumerate(conditions):
        local = seed * 1000 + offset
        if name == "free":
            sources[name] = ic.FreeSource(n, dim, std=float(config["head"]["free_std"]), seed=local)
        elif name == "composed_head":
            sources[name] = ic.ComposedSource(labels["frames"], atomic_count=int(labels["atomic_count"]),
                                              relation_count=int(labels["relation_count"]), dimension=int(comp["dimension"]),
                                              operator=comp["operator"], mode=comp["composition"],
                                              concept_factor=comp["concept_factor"], key_dimension=int(comp["key_dimension"]),
                                              seed=local)
        elif name == "composed_free":
            composed = ic.ComposedSource(labels["frames"], atomic_count=int(labels["atomic_count"]),
                                         relation_count=int(labels["relation_count"]), dimension=int(comp["dimension"]),
                                         operator=comp["operator"], mode=comp["composition"],
                                         concept_factor=comp["concept_factor"], key_dimension=int(comp["key_dimension"]), seed=local)
            sources[name] = ic.SumSource(composed, ic.FreeSource(n, int(comp["dimension"]), std=float(config["head"]["free_std"]),
                                                                 seed=local + 500))
        elif name == "composed_c5":
            if c5_composer is None:
                raise ValueError("composed_c5 needs --c5-run (the trained T1c C5 run)")
            sources[name] = ic.ComposedSource(labels["frames"], atomic_count=int(labels["atomic_count"]),
                                              relation_count=int(labels["relation_count"]), composer=c5_composer,
                                              trainable=False)
        elif name == "transe":
            if transe_vectors is None:
                raise ValueError("transe needs the kge stage's vectors")
            sources[name] = ic.FixedSource(ic.standardize_vectors(transe_vectors, trained_ids))
        elif name == "title":
            if title_vectors is None:
                raise ValueError("title needs the encoder's title states")
            vectors = torch.as_tensor(title_vectors, dtype=torch.float32)
            empty = vectors.abs().sum(1) == 0
            if bool(empty.any()):                    # a code without a title gets the mean title state of trained codes
                vectors[empty] = vectors[torch.as_tensor(trained_ids)][~empty[:n_trained]].mean(0)
            sources[name] = ic.FixedSource(ic.standardize_vectors(vectors, trained_ids))
        elif name == "random":
            sources[name] = ic.FixedSource(ic.random_vectors(n, dim, seed=local))
        elif name == "gram":
            sources[name] = ic.GramSource(labels["gram_paths"], len(labels["gram_nodes"]), dim,
                                          attention_dim=int(config["gram"]["attention_dim"]), std=float(config["head"]["free_std"]),
                                          seed=local)
        else:
            raise ValueError(f"unknown condition {name!r}; known: {CONDITIONS}")
    return sources


def heads_dir(config: dict[str, Any], encoder: str, seed: int, tag: str = "") -> Path:
    return data_root(config) / "heads" / f"{encoder}{tag}" / f"s{seed}"


def untrained_pairs(adm_labels: Sequence[np.ndarray], untrained: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(admission, label) pairs of never-trained labels over every admission."""
    rows, cols = [], []
    flag = np.zeros(int(untrained.max()) + 1 if untrained.size else 0, dtype=bool)
    flag[untrained] = True
    for a, labs in enumerate(adm_labels):
        for l in labs:
            if l < flag.size and flag[l]:
                rows.append(a); cols.append(int(l))
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def quick_metrics(scores_eval: np.ndarray, eval_labels: Sequence[np.ndarray], labels: dict[str, Any],
                  untrained_scores: np.ndarray, untrained_positive: np.ndarray) -> dict[str, Any]:
    """Aggregates for the run folder: seen-code macro AUC (evaluation admissions), held-out macro AUC (all admissions),
    micro-F1 at 0.5 and P@8 / P@15 over the trained labels."""
    n_trained = int(labels["n_trained"])
    positives = np.zeros(scores_eval.shape, dtype=bool)
    for row, labs in enumerate(eval_labels):
        positives[row, labs] = True
    seen = ic.auc_columns(scores_eval[:, :n_trained], positives[:, :n_trained])
    held_cols = np.flatnonzero(labels["heldout"][n_trained:])
    held = ic.auc_columns(untrained_scores[:, held_cols], untrained_positive[:, held_cols])
    s, y = scores_eval[:, :n_trained], positives[:, :n_trained]
    predicted = s > 0
    tp = float((predicted & y).sum())
    micro_f1 = 2 * tp / max(1.0, float(predicted.sum() + y.sum()))
    top = np.argsort(-s, axis=1)
    p_at = {k: float(np.take_along_axis(y, top[:, :k], 1).mean()) for k in (8, 15)}
    return {"seen_macro_auc_eval": finite(np.nanmean(seen)), "seen_codes_with_eval_positive": int(np.isfinite(seen).sum()),
            "heldout_macro_auc_all": finite(np.nanmean(held)), "heldout_codes_scored": int(np.isfinite(held).sum()),
            "micro_f1_at_0": micro_f1, "precision_at_8": p_at[8], "precision_at_15": p_at[15]}


def run_train(config: dict[str, Any], encoder: str, seed: int, *, conditions: Sequence[str], c5_run: Path | None = None,
              device: str = "cuda", limit_train: int | None = None, limit_eval: int | None = None,
              max_epochs: int | None = None, max_steps: int | None = None, tag: str = "", batch: int | None = None
              ) -> dict[str, Any]:
    from vsa_embed.provenance import git_state
    git_at_start = git_state()
    root = data_root(config)
    head_cfg = config["head"]
    labels = torch.load(root / "labels.pt", weights_only=False)
    adm = torch.load(root / "admissions.pt", weights_only=False)
    store = open_store(config, encoder)
    count = store.offsets.size - 1
    split = adm["split"][:count]
    adm_labels = adm["labels"][:count]
    rng = np.random.default_rng(int(config["seed"]))
    train_adm = np.flatnonzero(split == "train")
    dev_adm = np.flatnonzero(split == "dev")
    eval_adm = np.flatnonzero(split == "eval")
    if limit_train:
        train_adm = np.sort(rng.choice(train_adm, size=min(limit_train, train_adm.size), replace=False))
        dev_adm = dev_adm[:max(32, limit_train // 10)]
    if limit_eval:
        eval_adm = eval_adm[:limit_eval]
    n_labels, n_trained = len(labels["codes"]), int(labels["n_trained"])
    train_labels = np.arange(n_trained)
    positives_per_adm = np.mean([np.sum(l < n_trained) for l in (adm_labels[a] for a in train_adm)])
    prior = float(positives_per_adm / n_trained)
    folder_states = states_dir(config, encoder)
    titles = np.load(folder_states / "titles.npy") if "title" in conditions else None
    kge_path = root / "kge" / f"transe-d{config['kge']['dimension']}.pt"
    transe = torch.load(kge_path, weights_only=False)["vectors"] if "transe" in conditions else None
    composer = load_c5_composer(c5_run, Path(config["paths"]["ontology_pt"]).expanduser()) if "composed_c5" in conditions else None
    sources = build_sources(conditions, labels, config=config, seed=seed, title_vectors=titles, transe_vectors=transe,
                            c5_composer=composer)
    heads = {}
    for offset, (name, source) in enumerate(sources.items()):
        torch.manual_seed(seed * 1000 + 100 + offset)
        heads[name] = ic.LabelAttentionHead(source, store.width, attention_dim=int(head_cfg["attention_dim"]),
                                            hidden=int(head_cfg["label_hidden"]), prior=prior)
    with torch.no_grad():                         # code vectors before any gradient step (the init probe, §15)
        init_vectors = {name: head.source(torch.arange(n_labels)).float().cpu() for name, head in heads.items()}
    log_rows: list[dict[str, Any]] = []

    def log(row: dict[str, Any]) -> None:
        log_rows.append(row)
        print(json.dumps(row), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    histories = ic.train_heads(heads, store, adm_labels, train_admissions=train_adm, dev_admissions=dev_adm,
                               train_labels=train_labels, label_count=n_labels,
                               epochs=int(max_epochs or head_cfg["max_epochs"]), patience=int(head_cfg["patience"]),
                               batch=int(batch or head_cfg["batch"]), lr=float(head_cfg["lr"]),
                               weight_decay=float(head_cfg["weight_decay"]), seed=seed, device=device, max_steps=max_steps,
                               log=log)
    train_seconds = time.monotonic() - started

    # Scoring: evaluation admissions × every label; every admission × never-trained labels (+ ranks of their positives).
    out_dir = private_dir(heads_dir(config, encoder, seed, tag))
    untrained = np.arange(n_trained, n_labels)
    all_adm = np.arange(count) if not (limit_train or limit_eval) else np.sort(np.concatenate([train_adm, dev_adm, eval_adm]))
    pair_rows, pair_cols = untrained_pairs([adm_labels[a] for a in all_adm], untrained)
    pairs_by_row = defaultdict(list)
    for k, r in enumerate(pair_rows):
        pairs_by_row[int(r)].append(k)
    all_ids = np.arange(n_labels)
    metrics: dict[str, Any] = {}
    for name, head in heads.items():
        scores_eval = ic.score_admissions(head, store, eval_adm, all_ids, batch=32, device=device)
        untrained_scores = np.zeros((all_adm.size, untrained.size), dtype=np.float32)
        general = np.zeros(pair_rows.size, dtype=np.int64)
        zero_shot = np.zeros(pair_rows.size, dtype=np.int64)
        cursor = {"row": 0}
        subset = torch.as_tensor(untrained, device=device)

        def reduce(chunk: np.ndarray, logits: torch.Tensor) -> None:
            first = cursor["row"]
            untrained_scores[first:first + chunk.size] = logits[:, n_trained:].cpu().numpy()
            ks = [k for r in range(first, first + chunk.size) for k in pairs_by_row.get(r, ())]
            if ks:
                rows = torch.as_tensor(pair_rows[ks] - first, device=logits.device)
                cols = torch.as_tensor(pair_cols[ks], device=logits.device)
                general[ks] = ic.ranks_in_rows(logits, rows, cols).cpu().numpy()
                zero_shot[ks] = ic.ranks_in_rows(logits, rows, cols, subset).cpu().numpy()
            cursor["row"] += chunk.size
        ic.score_admissions(head, store, all_adm, all_ids, batch=32, device=device, reduce=reduce)
        cond_dir = private_dir(out_dir / name)
        np.save(cond_dir / "scores_eval.npy", scores_eval)
        np.save(cond_dir / "scores_untrained_all.npy", untrained_scores)
        np.savez(cond_dir / "ranks_untrained_all.npz", rows=pair_rows, cols=pair_cols, general=general, zero_shot=zero_shot)
        with torch.no_grad():
            ids = torch.arange(n_labels, device=device)
            head.eval()
            vectors = head.source(ids).float().cpu()
            q, o, b = head.label_parameters(ids)
            effective = torch.cat([q, o, b[:, None]], 1).float().cpu()
        torch.save({"source": vectors, "effective": effective, "source_init": init_vectors[name]}, cond_dir / "vectors.pt")
        untrained_positive = np.zeros(untrained_scores.shape, dtype=bool)
        untrained_positive[pair_rows, pair_cols - n_trained] = True
        metrics[name] = {**quick_metrics(scores_eval, [adm_labels[a] for a in eval_adm], labels, untrained_scores,
                                         untrained_positive),
                         "best_epoch": histories[name]["best_epoch"], "epochs": histories[name]["epochs"],
                         "best_dev_loss": histories[name]["best_dev_loss"],
                         "heldout_generalized_top100_all": float((general[labels["heldout"][pair_cols]] <= 100).mean())
                         if pair_cols.size else None}
    np.save(out_dir / "eval_admissions.npy", eval_adm)
    np.save(out_dir / "all_admissions.npy", all_adm)
    summary = {"encoder": encoder, "seed": seed, "conditions": list(conditions), "train_admissions": int(train_adm.size),
               "dev_admissions": int(dev_adm.size), "eval_admissions": int(eval_adm.size), "labels": n_labels,
               "trained_labels": n_trained, "prior": prior, "train_seconds": round(train_seconds, 1),
               "seconds": round(time.monotonic() - started, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
               "metrics": metrics, "history": {n: h["history"] for n, h in histories.items()},
               "c5_run": str(c5_run) if c5_run else None, "limits": {"train": limit_train, "eval": limit_eval,
                                                                       "max_epochs": max_epochs, "max_steps": max_steps}}
    # A job with another condition set than the config's (e.g. pass 2's composed_c5 on P0) gets its own folder.
    default = list(config["head"]["conditions"])
    folder = run_folder(config, f"train-{encoder}{tag}-s{seed}" + ("" if list(conditions) == default else "-" + "-".join(conditions)))
    write_json(folder / "metrics.json", summary)
    record_run(folder, config, git_at_start=git_at_start, device=device, stage="train")
    return summary


# -- analyze ------------------------------------------------------------------------------------------------------

def load_condition(config: dict[str, Any], encoder: str, seed: int, condition: str, tag: str = "") -> dict[str, Any] | None:
    folder = heads_dir(config, encoder, seed, tag) / condition
    if not (folder / "scores_eval.npy").exists():
        return None
    ranks = np.load(folder / "ranks_untrained_all.npz")
    return {"scores_eval": np.load(folder / "scores_eval.npy"), "untrained": np.load(folder / "scores_untrained_all.npy"),
            "ranks": {k: ranks[k] for k in ranks.files}, "vectors": torch.load(folder / "vectors.pt", weights_only=False)}


def run_analyze(config: dict[str, Any], encoder: str, seeds: Sequence[int], *, conditions: Sequence[str] | None = None,
                device: str = "cpu", bootstrap: int | None = None, tag: str = "", tsne_seed: int | None = None,
                label: str = "") -> dict[str, Any]:
    """Endpoints of the preregistration (§6–§7) for one encoder, with the two-way bootstrap."""
    from vsa_embed.provenance import git_state
    git_at_start = git_state()
    started = time.monotonic()

    def phase(name: str) -> None:
        print(json.dumps({"phase": name, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    root = data_root(config)
    an = config["analysis"]
    labels = torch.load(root / "labels.pt", weights_only=False)
    adm = torch.load(root / "admissions.pt", weights_only=False)
    n_trained, n_labels = int(labels["n_trained"]), len(labels["codes"])
    edges = labels["edges"]
    conditions = [c for c in (conditions or CONDITIONS)]
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for c in conditions:
        found = {s: load_condition(config, encoder, s, c, tag) for s in seeds}
        found = {s: v for s, v in found.items() if v is not None}
        if found:
            runs[c] = found
    if CONTROL not in runs:
        raise RuntimeError(f"the control condition {CONTROL!r} has no runs for {encoder}")
    first_seed = next(iter(runs[CONTROL]))
    eval_adm = np.load(heads_dir(config, encoder, first_seed, tag) / "eval_admissions.npy")
    all_adm = np.load(heads_dir(config, encoder, first_seed, tag) / "all_admissions.npy")
    eval_labels = [adm["labels"][a] for a in eval_adm]
    positives = np.zeros((eval_adm.size, n_labels), dtype=bool)
    for row, labs in enumerate(eval_labels):
        positives[row, labs] = True
    logf, bins = np.asarray(labels["logf"]), np.asarray(labels["bin"])
    heldout = np.asarray(labels["heldout"])
    seen_cols = np.flatnonzero(positives[:, :n_trained].any(0))             # trained codes with ≥ 1 eval positive
    held_cols_all = np.flatnonzero(heldout)                                  # label ids
    held_local = held_cols_all - n_trained                                   # columns of the untrained matrices
    pair = runs[CONTROL][first_seed]["ranks"]
    untrained_positive = np.zeros((all_adm.size, n_labels - n_trained), dtype=bool)
    untrained_positive[pair["rows"], pair["cols"] - n_trained] = True
    held_scored = held_local[untrained_positive[:, held_local].any(0)]
    eval_rows = np.searchsorted(all_adm, eval_adm)                          # all_adm is sorted (np.arange or sorted concat)
    held_eval_scored = held_local[untrained_positive[eval_rows][:, held_local].any(0)]
    ks = [int(k) for k in an["ks"]]

    phase("loaded")
    # -- point estimates per condition × seed --
    per: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    eval_pos_rows, eval_pos_cols = np.nonzero(positives)
    for c, by_seed in runs.items():
        for s, r in by_seed.items():
            seen_auc = ic.auc_columns(r["scores_eval"][:, seen_cols], positives[:, seen_cols])
            held_auc = ic.auc_columns(r["untrained"][:, held_scored], untrained_positive[:, held_scored])
            held_auc_eval = ic.auc_columns(r["untrained"][eval_rows][:, held_eval_scored],
                                           untrained_positive[eval_rows][:, held_eval_scored])
            ranks_eval = ic.positive_ranks(r["scores_eval"], eval_pos_rows, eval_pos_cols)
            topk = {k: ic.topk_by_code(ranks_eval, eval_pos_cols, n_labels, k)[0] for k in ks}
            hits = {k: ranks_eval <= k for k in ks}
            rk = r["ranks"]
            is_held_pair = heldout[rk["cols"]]
            held_general = {k: ic.topk_by_code(rk["general"][is_held_pair], rk["cols"][is_held_pair], n_labels, k)[0][held_cols_all]
                            for k in ks}
            held_zero = {k: ic.topk_by_code(rk["zero_shot"][is_held_pair], rk["cols"][is_held_pair], n_labels, k)[0][held_cols_all]
                         for k in ks}
            per[c][s] = {"seen_auc": seen_auc, "held_auc": held_auc, "held_auc_eval": held_auc_eval, "topk": topk,
                         "hits": hits, "held_general": held_general, "held_zero": held_zero}

    def mean_seeds(c: str, fn) -> float:
        vals = [fn(v) for v in per[c].values()]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    x_seen = logf[seen_cols]
    rare = x_seen < float(an["rare_below"])
    frequent = x_seen >= float(an["frequent_from"])
    endpoints: dict[str, Any] = {}
    for c in per:
        endpoints[c] = {
            "E1_heldout_macro_auc_all": mean_seeds(c, lambda v: np.nanmean(v["held_auc"])),
            "E2_slope_auc_on_logf": mean_seeds(c, lambda v: ic.ols_slope(x_seen, v["seen_auc"])),
            "heldout_macro_auc_eval": mean_seeds(c, lambda v: np.nanmean(v["held_auc_eval"])),
            "seen_macro_auc": mean_seeds(c, lambda v: np.nanmean(v["seen_auc"])),
            "rare_macro_auc": mean_seeds(c, lambda v: np.nanmean(v["seen_auc"][rare])),
            "frequent_macro_auc": mean_seeds(c, lambda v: np.nanmean(v["seen_auc"][frequent])),
            "gap_frequent_minus_rare": mean_seeds(c, lambda v: np.nanmean(v["seen_auc"][frequent]) - np.nanmean(v["seen_auc"][rare])),
            **{f"heldout_generalized_top{k}_all": mean_seeds(c, lambda v, k=k: np.nanmean(v["held_general"][k])) for k in ks},
            **{f"heldout_zero_shot_recall{k}_all": mean_seeds(c, lambda v, k=k: np.nanmean(v["held_zero"][k])) for k in ks},
            **{f"seen_top{k}_code_balanced": mean_seeds(c, lambda v, k=k: np.nanmean(v["topk"][k][seen_cols])) for k in ks},
            "seeds": sorted(per[c]),
        }

    # Per-bin tables (HRRBERT figure analogue): seen codes on evaluation admissions; held-out codes by natural bin.
    bin_table: dict[str, Any] = {}
    for i in range(len(edges) - 1):
        name = ic.bin_name(i, edges)
        in_bin = bins[seen_cols] == i
        held_in = bins[held_cols_all] == i
        row = {"seen_codes": int(in_bin.sum()), "heldout_codes": int(held_in.sum()), "conditions": {}}
        pair_in_bin = (bins[eval_pos_cols] == i) & (eval_pos_cols < n_trained)
        for c in per:
            entry = {"macro_auc": mean_seeds(c, lambda v: np.nanmean(v["seen_auc"][in_bin]) if in_bin.any() else float("nan"))}
            for k in ks:
                entry[f"top{k}_code_balanced"] = mean_seeds(c, lambda v, k=k: np.nanmean(v["topk"][k][seen_cols][in_bin]) if in_bin.any() else float("nan"))
                entry[f"recall{k}_pooled"] = mean_seeds(c, lambda v, k=k: float(v["hits"][k][pair_in_bin].mean()) if pair_in_bin.any() else float("nan"))
                entry[f"heldout_generalized_top{k}"] = mean_seeds(c, lambda v, k=k: np.nanmean(v["held_general"][k][held_in]) if held_in.any() else float("nan"))
            held_auc_bin = bins[held_scored + n_trained] == i
            entry["heldout_macro_auc_all"] = mean_seeds(c, lambda v: np.nanmean(v["held_auc"][held_auc_bin]) if held_auc_bin.any() else float("nan"))
            row["conditions"][c] = entry
        # HRRBERT-style Dunnett over seeds (descriptive; n = seeds per condition).
        row["dunnett"] = {}
        for k in ks:
            vals = {c: [float(np.nanmean(v["topk"][k][seen_cols][in_bin])) if in_bin.any() else float("nan")
                        for v in per[c].values()] for c in per}
            row["dunnett"][f"top{k}"] = ic.dunnett(vals[CONTROL], {c: v for c, v in vals.items() if c != CONTROL})
        bin_table[name] = row

    phase("point estimates")
    # -- two-way bootstrap (admissions × codes; seeds resampled within each replicate) --
    dev = torch.device(device)
    replicates = int(bootstrap or an["bootstrap_primary"])
    generator = np.random.default_rng(int(config["seed"]))
    weights_all = generator.poisson(1.0, size=(replicates, all_adm.size)).astype(np.float32)
    code_draw_seen = generator.integers(0, seen_cols.size, size=(replicates, seen_cols.size))
    code_draw_held = generator.integers(0, held_scored.size, size=(replicates, held_scored.size)) if held_scored.size else None
    seed_draw = {c: generator.integers(0, len(per[c]), size=(replicates, len(per[c]))) for c in per}
    boot: dict[str, dict[str, np.ndarray]] = {}
    x_seen_t = torch.as_tensor(x_seen, dtype=torch.float32, device=dev)
    for c, by_seed in runs.items():
        stats = {"E1": [], "E2": [], "rare": [], "frequent": [], "gap": [], "heldout_eval": []}
        seed_list = sorted(by_seed)
        per_seed = []
        for s in seed_list:
            r = by_seed[s]
            seen_w = ic.WeightedAuc(r["scores_eval"][:, seen_cols], positives[:, seen_cols], device=dev)
            held_w = ic.WeightedAuc(r["untrained"][:, held_scored], untrained_positive[:, held_scored], device=dev) if held_scored.size else None
            seen_b = torch.zeros(replicates, seen_cols.size, device=dev)
            held_b = torch.zeros(replicates, held_scored.size, device=dev)
            for b in range(replicates):
                w = torch.as_tensor(weights_all[b], device=dev)
                seen_b[b] = seen_w(w[eval_rows])
                if held_w is not None:
                    held_b[b] = held_w(w)
            per_seed.append((seen_b, held_b))
            del seen_w, held_w
        draws = seed_draw[c]
        idx_seen = torch.as_tensor(code_draw_seen, device=dev)
        for b in range(replicates):
            chosen = [per_seed[j] for j in draws[b]]
            seen = torch.stack([p[0][b] for p in chosen]).mean(0)[idx_seen[b]]
            x = x_seen_t[idx_seen[b]]
            stats["E2"].append(float(ic.ols_slopes(x[None], seen[None])[0]))
            r_mask, f_mask = x < float(an["rare_below"]), x >= float(an["frequent_from"])
            rare_v, freq_v = torch.nanmean(seen[r_mask]), torch.nanmean(seen[f_mask])
            stats["rare"].append(float(rare_v)); stats["frequent"].append(float(freq_v))
            stats["gap"].append(float(freq_v - rare_v))
            if code_draw_held is not None:
                held = torch.stack([p[1][b] for p in chosen]).mean(0)[torch.as_tensor(code_draw_held[b], device=dev)]
                stats["E1"].append(float(torch.nanmean(held)))
        boot[c] = {k: np.asarray(v, dtype=np.float64) for k, v in stats.items() if v}
        phase(f"bootstrap {c}")

    comparisons: dict[str, Any] = {}
    for metric, direction in (("E1", "higher"), ("E2", "lower"), ("rare", "higher"), ("frequent", "higher"), ("gap", "lower")):
        rows = {}
        for c in boot:
            if c == CONTROL or metric not in boot[c] or metric not in boot[CONTROL]:
                continue
            delta = boot[c][metric] - boot[CONTROL][metric]
            rows[c] = {"delta_bootstrap_mean": finite(np.nanmean(delta)), "ci95": [finite(np.nanpercentile(delta, 2.5)),
                                                                                   finite(np.nanpercentile(delta, 97.5))],
                       "p_two_sided": ic.bootstrap_pvalue(delta)}
        adjusted = ic.holm({c: v["p_two_sided"] for c, v in rows.items()})
        for c in rows:
            rows[c]["p_holm_vs_control"] = adjusted[c]
        comparisons[metric] = {"direction_of_benefit": direction, "vs_control": rows}
    # Specificity: the primary arm against each structured competitor (E1, E2), Holm over competitors.
    specificity = {}
    for metric in ("E1", "E2"):
        rows = {}
        for c in ("random", "title", "transe", "gram", "composed_c5", "composed_free"):
            if c in boot and PRIMARY in boot and metric in boot[c]:
                delta = boot[PRIMARY][metric] - boot[c][metric]
                rows[c] = {"delta_primary_minus_competitor": finite(np.nanmean(delta)),
                           "ci95": [finite(np.nanpercentile(delta, 2.5)), finite(np.nanpercentile(delta, 97.5))],
                           "p_two_sided": ic.bootstrap_pvalue(delta)}
        adjusted = ic.holm({c: v["p_two_sided"] for c, v in rows.items()})
        for c in rows:
            rows[c]["p_holm"] = adjusted[c]
        specificity[metric] = rows
    primary = {}
    if PRIMARY in boot:
        p = {m: comparisons[m]["vs_control"].get(PRIMARY, {}).get("p_two_sided", float("nan")) for m in ("E1", "E2")}
        adjusted = ic.holm(p)
        primary = {m: {**comparisons[m]["vs_control"][PRIMARY], "p_holm_primary": adjusted[m]} for m in ("E1", "E2")
                   if PRIMARY in comparisons[m]["vs_control"]}

    # -- frequency information in the code vectors (seed = the first; t-SNE written to the data root only) --
    probes = {}
    analysis_dir = private_dir(root / "analysis" / f"{encoder}{tag}{label}")
    tsne_seed = tsne_seed if tsne_seed is not None else min(seeds)
    trained_ids = np.arange(n_trained)
    y = logf[trained_ids]
    for c, by_seed in runs.items():
        rows = {}
        for s, r in by_seed.items():
            source = r["vectors"]["source"].numpy()[trained_ids]
            effective = r["vectors"]["effective"].numpy()[trained_ids]
            rows[s] = {"source_r2": ic.cv_ridge(source, y, folds=int(an["probe_folds"]), seed=s, device=dev)["r2"],
                       "effective_r2": ic.cv_ridge(effective, y, folds=int(an["probe_folds"]), seed=s, device=dev)["r2"],
                       "bias_spearman": ic.spearman(effective[:, -1], y),
                       "neighbour_agreement": ic.neighbour_agreement(source, y, k=int(an["neighbours"]), device=dev)}
            init = r["vectors"].get("source_init")
            if init is not None:                      # frequency implied by the content alone (before training)
                init = init.numpy()[trained_ids]
                rows[s]["source_r2_init"] = ic.cv_ridge(init, y, folds=int(an["probe_folds"]), seed=s, device=dev)["r2"]
                rows[s]["neighbour_agreement_init"] = ic.neighbour_agreement(init, y, k=int(an["neighbours"]), device=dev)
        probes[c] = {k: float(np.mean([v[k] for v in rows.values()])) for k in next(iter(rows.values()))}
        probes[c]["per_seed"] = rows
        if tsne_seed in by_seed:
            pick = np.random.default_rng(0).permutation(n_labels)[:int(an["tsne_points"])]
            coords = ic.tsne(by_seed[tsne_seed]["vectors"]["source"].numpy()[pick], iterations=int(an["tsne_iterations"]),
                             seed=0, device=dev)
            np.savez(analysis_dir / f"tsne-{c}-s{tsne_seed}.npz", coords=coords, labels=pick, logf=logf[pick],
                     heldout=heldout[pick])
            seen_pick = pick < n_trained
            probes[c]["tsne_neighbour_agreement"] = ic.neighbour_agreement(coords[seen_pick], logf[pick][seen_pick],
                                                                           k=int(an["neighbours"]), device=dev)
        phase(f"probes {c}")
    try:
        plot_tsne(analysis_dir, [c for c in runs], tsne_seed)
    except Exception as error:                     # the figure is a convenience; the numbers are the result
        probes["_figure_error"] = str(error)

    result = {"encoder": encoder, "seeds": list(seeds), "conditions": list(runs), "bootstrap_replicates": replicates,
              "codes": {"seen_with_eval_positive": int(seen_cols.size), "heldout_scored_all": int(held_scored.size),
                        "heldout_scored_eval": int(held_eval_scored.size), "rare_seen": int(rare.sum()),
                        "frequent_seen": int(frequent.sum())},
              "endpoints": endpoints, "primary": primary, "comparisons": comparisons, "specificity": specificity,
              "bins": bin_table, "probes": probes, "decision": decide(endpoints, primary, comparisons, specificity)}
    folder = run_folder(config, f"analysis-{encoder}{tag}{label}")
    write_json(folder / "endpoints.json", result)
    (folder / "report.md").write_text(render_report(result, edges))
    record_run(folder, config, git_at_start=git_at_start, device=device, stage="analyze")
    return result


def decide(endpoints: dict[str, Any], primary: dict[str, Any], comparisons: dict[str, Any],
           specificity: dict[str, Any]) -> dict[str, Any]:
    """The preregistered readings (§7) from the numbers; None where an input is missing."""
    if not primary:
        return {"available": False}
    e1, e2 = primary.get("E1", {}), primary.get("E2", {})
    rare = comparisons["rare"]["vs_control"].get(PRIMARY, {})
    freq = comparisons["frequent"]["vs_control"].get(PRIMARY, {})
    out = {"available": True}
    out["E1_supported"] = bool(e1 and e1["p_holm_primary"] < 0.05 and e1["delta_bootstrap_mean"] > 0)
    out["E1_refuted"] = bool(e1 and e1["ci95"][1] is not None and e1["ci95"][1] < 0.02)
    slope_down = bool(e2 and e2["p_holm_primary"] < 0.05 and e2["delta_bootstrap_mean"] < 0)
    rare_ok = bool(rare and rare["delta_bootstrap_mean"] is not None and rare["delta_bootstrap_mean"] >= 0
                   and rare["ci95"][0] is not None and rare["ci95"][0] > -0.01)
    frequent_damaged = bool(freq and freq["delta_bootstrap_mean"] is not None and freq["delta_bootstrap_mean"] < -0.01)
    out["E2_slope_reduced"] = slope_down
    out["E2_supported"] = slope_down and rare_ok
    out["E2_flattening_by_damage"] = slope_down and not rare_ok and frequent_damaged
    readings = {}
    for c, row in specificity.get("E1", {}).items():
        d, (lo, hi) = row["delta_primary_minus_competitor"], row["ci95"]
        if d is None or lo is None:
            continue
        readings[c] = ("primary better" if row["p_holm"] < 0.05 and d > 0 else
                       "competitor better" if row["p_holm"] < 0.05 and d < 0 else
                       "equivalent within ±0.01" if lo > -0.01 and hi < 0.01 else "inconclusive")
    out["E1_specificity"] = readings
    return out


def plot_tsne(folder: Path, conditions: Sequence[str], seed: int) -> None:
    """HRRBERT's figure: t-SNE of the code vectors coloured by log training frequency (held-out codes as crosses).
    Written to the licensed data root only (per-code points)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = [(c, folder / f"tsne-{c}-s{seed}.npz") for c in conditions if (folder / f"tsne-{c}-s{seed}.npz").exists()]
    if not files:
        return
    cols = min(4, len(files))
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax, (c, path) in zip(axes.flat, files):
        data = np.load(path)
        seen = np.isfinite(data["logf"])
        sc = ax.scatter(data["coords"][seen, 0], data["coords"][seen, 1], c=data["logf"][seen], cmap="coolwarm_r", s=3,
                        vmin=-14, vmax=-2)
        held = data["heldout"]
        ax.scatter(data["coords"][held, 0], data["coords"][held, 1], marker="x", c="black", s=6, linewidths=0.5)
        ax.set_title(c); ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(files):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="ln relative training frequency")
    fig.savefig(folder / f"tsne-s{seed}.png", dpi=150)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def render_report(result: dict[str, Any], edges: Sequence[float]) -> str:
    lines = [f"# T1c-F analysis — encoder {result['encoder']} (seeds {result['seeds']})", "",
             "Aggregates only (licensed data stay under the data root). Preregistration: "
             "`experiments/t1c-clinical/icd-frequency/preregistration.md`.", "",
             f"Codes: {result['codes']}. Bootstrap replicates: {result['bootstrap_replicates']}.", "",
             "## Endpoints (mean over seeds)", "",
             "| condition | E1 held-out macro-AUC (all adm.) | E2 slope AUC ~ ln f | seen macro-AUC | rare | frequent | gap | held-out top-100 (generalized) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, e in result["endpoints"].items():
        lines.append(f"| {c} | {_fmt(e['E1_heldout_macro_auc_all'])} | {_fmt(e['E2_slope_auc_on_logf'], 4)} | "
                     f"{_fmt(e['seen_macro_auc'])} | {_fmt(e['rare_macro_auc'])} | {_fmt(e['frequent_macro_auc'])} | "
                     f"{_fmt(e['gap_frequent_minus_rare'])} | {_fmt(e.get('heldout_generalized_top100_all'))} |")
    lines += ["", "## Primary comparisons (composed_head − free; Holm over E1, E2)", ""]
    for m, row in result["primary"].items():
        lines.append(f"- {m}: Δ = {_fmt(row['delta_bootstrap_mean'], 4)} [{_fmt(row['ci95'][0], 4)}, {_fmt(row['ci95'][1], 4)}], "
                     f"p = {_fmt(row['p_two_sided'], 4)}, Holm p = {_fmt(row['p_holm_primary'], 4)}")
    lines += ["", f"Decision: `{json.dumps(result['decision'])}`", "", "## Per bin (seen codes on evaluation admissions)", ""]
    conds = list(result["endpoints"])
    lines.append("| bin | seen / held-out codes | " + " | ".join(f"{c} AUC / top-10 / top-100" for c in conds) + " |")
    lines.append("|---|---|" + "---|" * len(conds))
    for name, row in result["bins"].items():
        cells = [f"{_fmt(row['conditions'][c]['macro_auc'])} / {_fmt(row['conditions'][c].get('top10_code_balanced'))} / "
                 f"{_fmt(row['conditions'][c].get('top100_code_balanced'))}" for c in conds]
        lines.append(f"| {name} | {row['seen_codes']} / {row['heldout_codes']} | " + " | ".join(cells) + " |")
    lines += ["", "## Frequency information in the code vectors", "",
              "| condition | ridge R² (code vector) | at init | ridge R² (label parameters) | ρ(bias, ln f) | kNN agreement | at init | t-SNE kNN agreement |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, p in result["probes"].items():
        if c.startswith("_"):
            continue
        lines.append(f"| {c} | {_fmt(p['source_r2'])} | {_fmt(p.get('source_r2_init'))} | {_fmt(p['effective_r2'])} | "
                     f"{_fmt(p['bias_spearman'])} | {_fmt(p['neighbour_agreement'])} | {_fmt(p.get('neighbour_agreement_init'))} | "
                     f"{_fmt(p.get('tsne_neighbour_agreement'))} |")
    return "\n".join(lines) + "\n"


# -- plan ---------------------------------------------------------------------------------------------------------

E9_RUNS = Path("experiments/e9-retrofit/runs/t1c")
CONFIG_ARG = "--config experiments/t1c-clinical/icd-frequency/icd-frequency.yaml"


def plan_commands(*, priority_now: int = 55, priority_after: int = 60, gpu_hours: dict[str, float] | None = None) -> str:
    """The exact queue commands (printed; never executed here). GPU-h: idle-GPU estimates scaled from the smoke test
    (§14 of the preregistration), which ran next to another training job."""
    h = {"encode_p0": 1.5, "encode_run": 1.7, "train_seed": 0.8, "train_c5_seed": 0.15, "analyze": 0.15, **(gpu_hours or {})}
    q = "PYTHONPATH=src $PY -m vsa_embed.jobqueue add"
    m = "$PY -m vsa_embed.experiments.t1c_icd_frequency"
    c5 = E9_RUNS / "SmolLM2-360M-full-C5-s1"
    all8 = "free composed_head composed_c5 transe title random gram composed_free"
    lines = ["#!/usr/bin/env bash",
             "# T1c-F (ICD frequency bias; preregistration.md in this folder): exact queue commands. NOT EXECUTED by the agent.",
             "# Printed by `python -m vsa_embed.experiments.t1c_icd_frequency plan`. Run from the main checkout's root after merging:",
             "#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python",
             "# The CPU stages (prepare, tokenize, kge) have run; their outputs are in ~/data/vsa-llm/t1c/icd-frequency-v1/",
             "# (shared by every checkout) and prepare re-checks the pinned holdout if re-run.",
             "# GPU-h: idle-GPU estimates from the smoke (§14); the smoke ran beside a training job (≈ 1.6× slower).",
             "# Memory: ≤ 1.5 GB per job measured (encode 1.2 GB, train 1.5 GB); disk: states ≈ 12 GB per encoder, heads ≈ 1.7 GB per job.",
             "set -euo pipefail", ': "${PY:?set PY to the pinned interpreter}"', "",
             f"# --- pass 1: P0 = frozen SmolLM2-360M (no dependency; could run now). Priority {priority_now}: after T1c seed 1 (51–54),",
             "# behind the 55 jobs already queued (WP-UB U1, Qwen3 seed 3; equal priority runs in creation order); a lower number",
             f"# would run it first. ≈ {h['encode_p0'] + 3 * h['train_seed'] + h['analyze']:.1f} GPU-h ---"]
    lines.append(f"{q} --name t1cf-encode-P0-360M --priority {priority_now} --min-free-gb 20 -- {m} encode {CONFIG_ARG} "
                 f"--encoder P0-360M --pretrained HuggingFaceTB/SmolLM2-360M   # ≈ {h['encode_p0']:.1f} GPU-h (resumable)")
    for s in (1, 2, 3):
        lines.append(f"{q} --name t1cf-train-P0-360M-s{s} --priority {priority_now} --min-free-gb 10 --no-resume -- {m} train "
                     f"{CONFIG_ARG} --encoder P0-360M --seed {s}   # 7 conditions ≈ {h['train_seed']:.1f} GPU-h")
    lines.append(f"{q} --name t1cf-analyze-P0-360M --priority {priority_now} --min-free-gb 2 --no-resume -- {m} analyze "
                 f"{CONFIG_ARG} --encoder P0-360M --seeds 1 2 3 --device cuda   # ≈ {h['analyze']:.2f} GPU-h")
    total2 = 2 * (h["encode_run"] + 3 * h["train_seed"] * 8 / 7) + 3 * h["train_c5_seed"] + 3 * h["analyze"]
    lines += ["", f"# --- pass 2: needs the T1c seed-1 runs SmolLM2-360M-full-C0p-s1 and -C5-s1 (queued at 51). Priority {priority_after}",
              f"# (analyses {priority_after + 1}): after WP-UB's 55–59 block, so the two frequency-bias packages do not interleave.",
              f"# ≈ {total2:.1f} GPU-h ---"]
    for enc, run, extra in (("C0p-360M", "SmolLM2-360M-full-C0p-s1", ""), ("C5-360M", "SmolLM2-360M-full-C5-s1", " --channel on")):
        lines.append(f"{q} --name t1cf-encode-{enc} --priority {priority_after} --min-free-gb 20 -- {m} encode {CONFIG_ARG} "
                     f"--encoder {enc} --run {E9_RUNS / run}{extra}   # ≈ {h['encode_run']:.1f} GPU-h (resumable)")
        for s in (1, 2, 3):
            lines.append(f"{q} --name t1cf-train-{enc}-s{s} --priority {priority_after} --min-free-gb 10 --no-resume -- {m} train "
                         f"{CONFIG_ARG} --encoder {enc} --seed {s} --conditions {all8} --c5-run {c5}   # 8 conditions ≈ "
                         f"{h['train_seed'] * 8 / 7:.1f} GPU-h")
    for s in (1, 2, 3):
        lines.append(f"{q} --name t1cf-train-P0-360M-c5dict-s{s} --priority {priority_after} --min-free-gb 10 --no-resume -- {m} train "
                     f"{CONFIG_ARG} --encoder P0-360M --seed {s} --conditions composed_c5 --c5-run {c5}"
                     f"   # ≈ {h['train_c5_seed']:.2f} GPU-h (same batch order as pass 1: paired)")
    for enc in ("P0-360M", "C0p-360M", "C5-360M"):
        lines.append(f"{q} --name t1cf-analyze-{enc}-pass2 --priority {priority_after + 1} --min-free-gb 2 --no-resume -- {m} analyze "
                     f"{CONFIG_ARG} --encoder {enc} --seeds 1 2 3 --device cuda --label -pass2   # ≈ {h['analyze']:.2f} GPU-h")
    lines += ["", "# optional (open decision): the C5 encoder with its channel off — reads whether a C5-encoder gain comes through the",
              "# injected rows (§6). Then train it like C5-360M (3 seeds) and analyze with --label -pass2:",
              f"# {q} --name t1cf-encode-C5off-360M --priority {priority_after + 2} --min-free-gb 20 -- {m} encode {CONFIG_ARG} "
              f"--encoder C5off-360M --run {c5} --channel off   # ≈ {h['encode_run']:.1f} GPU-h"]
    return "\n".join(lines) + "\n"


# -- CLI ----------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "tokenize", "kge", "encode", "train", "analyze", "plan"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--encoder", help="name of the encoder's state folder (e.g. P0-360M, C0p-360M, C5-360M)")
    parser.add_argument("--pretrained", help="encode: a frozen Hugging Face host (P0)")
    parser.add_argument("--run", type=Path, help="encode: a trained E9 run folder (C0′, C5)")
    parser.add_argument("--channel", choices=("on", "off"), default="on", help="encode: the C5 run's span channel")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--c5-run", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="tokenize / encode: first N admissions (smoke)")
    parser.add_argument("--tokens-dir", default="tokens", help="encode: token folder under the data root")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--kge-steps", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix of the head / analysis folders (smoke runs)")
    parser.add_argument("--label", default="", help="analyze: suffix of the analysis folders only (e.g. -pass2)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "plan":
        print(plan_commands(), end="")
        return
    config = load_config(args.config)
    if args.stage == "prepare":
        result = run_prepare(config)
    elif args.stage == "tokenize":
        result = run_tokenize(config, limit=args.limit)
    elif args.stage == "kge":
        result = run_kge(config, device=args.device, steps=args.kge_steps)
    elif args.stage == "encode":
        if not args.encoder or not (args.pretrained or args.run):
            parser.error("encode needs --encoder and --pretrained or --run")
        result = run_encode(config, args.encoder, pretrained=args.pretrained, run=args.run, channel=args.channel == "on",
                            device=args.device, limit=args.limit, resume=args.resume, tokens_dir=args.tokens_dir)
    elif args.stage == "train":
        if not args.encoder:
            parser.error("train needs --encoder")
        conditions = args.conditions or list(config["head"]["conditions"])
        result = run_train(config, args.encoder, args.seed, conditions=conditions, c5_run=args.c5_run, device=args.device,
                           limit_train=args.limit_train, limit_eval=args.limit_eval, max_epochs=args.max_epochs,
                           max_steps=args.max_steps, tag=args.tag, batch=args.batch)
        result = {k: v for k, v in result.items() if k != "history"}
    else:
        if not args.encoder:
            parser.error("analyze needs --encoder")
        result = run_analyze(config, args.encoder, args.seeds, conditions=args.conditions, device=args.device,
                             bootstrap=args.bootstrap, tag=args.tag, label=args.label)
        result = {k: result[k] for k in ("encoder", "seeds", "conditions", "codes", "primary", "decision")}
    print(json.dumps(result, indent=1, default=_native)[:6000])


if __name__ == "__main__":
    main()
