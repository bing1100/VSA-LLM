"""T1c proposal (a): how many ICD codes could get a zero-shot vector composed from SNOMED CT frames (counts only).

The design question behind proposal (a) — rare and unseen ICD code assignment from MIMIC-III discharge summaries,
with code vectors composed from the SNOMED CT frames of the codes' mapped concepts — needs to know, per code
frequency band, how many codes map to a concept of the T1c ontology. This module answers with aggregates only
(numbers of codes, admissions and label occurrences); no code, concept or note is printed or written.

- MIMIC-III (`DIAGNOSES_ICD`, ICD-9-CM, codes without dots) restricted to admissions with a discharge summary
  (`NOTEEVENTS` category, read from the T1c note extraction: hadm ids and categories only), split by the T1c
  patient split; frequency = training admissions carrying the code.
- ICD-9-CM → SNOMED CT: NLM's ICD9CM_SNOMED_MAP (1-to-1 and 1-to-many files); columns are located by pattern
  (ICD-9 code / SNOMED identifier), never by printing a row.
- MIMIC-IV (`diagnoses_icd`, ICD-10-CM; structured data only — no MIMIC-IV notes on this machine) against the
  SNOMED CT → ICD-10-CM map (US edition, human-readable TSV), reversed.

    PYTHONPATH=src python -m vsa_embed.experiments.t1c_icd --config experiments/t1c-clinical/t1c.yaml \
        --output experiments/t1c-clinical/runs/icd-coverage-v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vsa_embed.data.mimic import subject_bucket

ICD9 = re.compile(r"^(\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$")
SCTID = re.compile(r"^\d{6,18}$")
BANDS = (("unseen", 0, 0), ("1-9", 1, 9), ("10-49", 10, 49), ("50-499", 50, 499), ("500+", 500, 10**12))


def band(count: int) -> str:
    return next(name for name, lo, hi in BANDS if lo <= count <= hi)


def read_icd9_map(paths: list[Path]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """ICD-9-CM code (dots removed) → SNOMED CT concept ids, from tab-separated NLM map files whose code and concept
    columns are found by pattern (the column where most rows look like an ICD-9 code / a SNOMED identifier)."""
    rows: list[list[str]] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            rows += [line.rstrip("\n").rstrip("\r").split("\t") for line in handle]
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == width]
    icd_col = max(range(width), key=lambda c: sum(bool(ICD9.match(r[c].strip())) for r in rows))
    sct_col = max((c for c in range(width) if c != icd_col), key=lambda c: sum(bool(SCTID.match(r[c].strip())) for r in rows))
    mapping: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        code, concept = r[icd_col].strip(), r[sct_col].strip()
        if ICD9.match(code) and SCTID.match(concept):
            mapping[code.replace(".", "")].add(concept)
    info = {"rows": len(rows), "columns": width, "codes": len(mapping),
            "code_column_match": sum(bool(ICD9.match(r[icd_col].strip())) for r in rows) / max(1, len(rows)),
            "concept_column_match": sum(bool(SCTID.match(r[sct_col].strip())) for r in rows) / max(1, len(rows))}
    return dict(mapping), info


def read_icd10_map(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """ICD-10-CM code (dots removed) → SNOMED CT concepts, reversing the active rows of the SNOMED CT → ICD-10-CM map."""
    mapping: dict[str, set[str]] = defaultdict(set)
    rows = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            target = (row.get("mapTarget") or "").strip()
            if row.get("active") == "1" and target and SCTID.match(row.get("referencedComponentId") or ""):
                mapping[target.replace(".", "")].add(row["referencedComponentId"])
    return dict(mapping), {"rows": rows, "codes": len(mapping)}


def discharge_admissions(notes_dir: Path) -> dict[str, set[int]]:
    """Admission ids with a discharge summary, per split of the T1c extraction (no text read)."""
    import pyarrow.parquet as pq
    out: dict[str, set[int]] = {"train": set(), "eval": set()}
    for path in sorted(Path(notes_dir).glob("notes-*-*.parquet")):
        split = path.name.split("-")[1]
        table = pq.read_table(path, columns=["hadm_id", "category"]).to_pydict()
        out[split] |= {h for h, c in zip(table["hadm_id"], table["category"]) if c == "Discharge summary" and h >= 0}
    return out


def band_table(train_counts: Counter, eval_counts: Counter, mapping: dict[str, set[str]], track: set[str]) -> dict[str, Any]:
    """Per training-frequency band: codes (in either split), codes in evaluation, label occurrences in evaluation, and
    how many of each map to SNOMED CT and to a concept of the track ontology."""
    codes = set(train_counts) | set(eval_counts)
    out: dict[str, Any] = {}
    for name, _, _ in BANDS:
        chosen = [c for c in codes if band(train_counts.get(c, 0)) == name]
        in_eval = [c for c in chosen if eval_counts.get(c, 0)]
        mapped = [c for c in chosen if mapping.get(c)]
        in_track = [c for c in chosen if mapping.get(c, set()) & track]
        out[name] = {"codes": len(chosen), "mapped": len(mapped), "mapped_to_track": len(in_track),
                     "codes_in_eval": len(in_eval), "eval_codes_mapped_to_track": sum(1 for c in in_eval if mapping.get(c, set()) & track),
                     "eval_label_occurrences": sum(eval_counts[c] for c in in_eval),
                     "eval_label_occurrences_mapped_to_track": sum(eval_counts[c] for c in in_eval if mapping.get(c, set()) & track)}
    return out


def coverage(config: dict[str, Any]) -> dict[str, Any]:
    from vsa_embed.experiments.t1c_corpus import build_track_ontology
    licensed = Path(config["paths"]["licensed_root"]).expanduser()
    eval_buckets = int(config["mimic"]["eval_buckets"])
    ontology = build_track_ontology(config["ontology"])
    track = set(ontology.concept_names)
    result: dict[str, Any] = {"track_concepts": len(track)}

    # MIMIC-III, ICD-9-CM, admissions with a discharge summary.
    icd9_dir = licensed / "mimic-iv_data/hosp/Mapping/ICD9CM_DIAGNOSIS_MAP_202112"
    icd9_map, icd9_info = read_icd9_map(sorted(icd9_dir.glob("ICD9CM_SNOMED_MAP_*.txt")))
    admissions = discharge_admissions(Path(config["paths"]["notes_dir"]).expanduser())
    counts = {"train": Counter(), "eval": Counter()}
    labels = {"train": Counter(), "eval": Counter()}
    path = licensed / "physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz"
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            if not row["HADM_ID"] or not row["ICD9_CODE"]:
                continue
            split = "eval" if subject_bucket(int(row["SUBJECT_ID"])) < eval_buckets else "train"
            hadm = int(row["HADM_ID"])
            if hadm in admissions[split]:
                counts[split][row["ICD9_CODE"]] += 1
                labels[split][hadm] += 1
    result["mimic_iii_icd9_diagnoses"] = {
        "map": icd9_info, "discharge_admissions": {s: len(a) for s, a in admissions.items()},
        "admissions_with_codes": {s: len(labels[s]) for s in labels},
        "labels_per_admission_mean": {s: (sum(labels[s].values()) / max(1, len(labels[s]))) for s in labels},
        "distinct_codes": {s: len(counts[s]) for s in counts}, "distinct_codes_all": len(set(counts["train"]) | set(counts["eval"])),
        "codes_mapped_to_snomed": sum(1 for c in set(counts["train"]) | set(counts["eval"]) if icd9_map.get(c)),
        "codes_mapped_to_track": sum(1 for c in set(counts["train"]) | set(counts["eval"]) if icd9_map.get(c, set()) & track),
        "by_training_frequency": band_table(counts["train"], counts["eval"], icd9_map, track)}

    # MIMIC-IV, ICD-10-CM (structured only), split by the same patient hash.
    icd10_path = next(iter(sorted((licensed / "mimic-iv_data/hosp/Mapping").glob("SNOMED_CT_to_ICD-10-CM_Resources_*/SNOMED_CT_to_ICD-10-CM_Resources_*/tls_Icd10cmHumanReadableMap_*.tsv"))), None)
    if icd10_path is not None:
        icd10_map, icd10_info = read_icd10_map(icd10_path)
        counts10 = {"train": Counter(), "eval": Counter()}
        admissions10 = {"train": set(), "eval": set()}
        with gzip.open(licensed / "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz", "rt") as handle:
            for row in csv.DictReader(handle):
                if row["icd_version"] != "10":
                    continue
                split = "eval" if subject_bucket(int(row["subject_id"])) < eval_buckets else "train"
                counts10[split][row["icd_code"].strip()] += 1
                admissions10[split].add(row["hadm_id"])
        codes10 = set(counts10["train"]) | set(counts10["eval"])
        result["mimic_iv_icd10_diagnoses"] = {
            "map": icd10_info, "admissions": {s: len(a) for s, a in admissions10.items()},
            "distinct_codes_all": len(codes10), "codes_mapped_to_snomed": sum(1 for c in codes10 if icd10_map.get(c)),
            "codes_mapped_to_track": sum(1 for c in codes10 if icd10_map.get(c, set()) & track),
            "by_training_frequency": band_table(counts10["train"], counts10["eval"], icd10_map, track),
            "note": "MIMIC-IV notes are not on this machine: a text task on ICD-10 would need MIMIC-IV-Note"}
    return result


def main(argv: list[str] | None = None) -> None:
    from vsa_embed.experiments.t1_open_corpus import load_config
    from vsa_embed.provenance import prepare_output_dir, write_run_metadata
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    git_at_start = prepare_output_dir(args.output)
    result = coverage(config)
    (args.output / "icd_coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_metadata(args.output, config, git_at_start=git_at_start, device="cpu", stage="icd-coverage")
    print(json.dumps({k: v for k, v in result.items() if k != "track_concepts"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
