"""Inventory of the credentialed / licensed clinical files (decision 58): names, sizes, row counts, column headers.

Nothing here reads past a file's first line except to count lines, so no row content is ever returned:

- a header is reported only for groups whose first line is a documented header (RF2, MIMIC CSVs), or, for other
  files, when every first-line field is an identifier (`[A-Za-z_][A-Za-z0-9_]*`) and the second line has a
  numeric field where the first has none (header evidence; the second line is read but never returned);
  otherwise only the field count is reported;
- UMLS `.RRF` files have no header line; their columns come from `MRFILES.RRF` (the release's own file
  metadata, which lists file names and column names);
- row counts are line counts (`pigz -dc | wc -l` for `.gz`, a byte scan otherwise) minus the header line. For
  `NOTEEVENTS.csv.gz` the note text holds newlines, so its line count overstates its rows; the extraction
  (`data.mimic.extract_notes`) counts rows with a CSV parser.

    python -m vsa_embed.data.clinical_inventory --root /path/to/data --output inventory.json [--no-rows]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_COLUMN = re.compile(r"^[A-Za-z][A-Za-z0-9_ .()/-]{0,63}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_NUMBER = re.compile(r"^-?\d+([.:/-]\d+)*$")
TRUSTED_HEADERS = ("snomed_ct", "mimic_iii", "mimic_iv")
TABULAR = (".csv", ".csv.gz", ".txt", ".tsv", ".tsv.gz", ".rrf", ".rrf.gz")


def suffix(path: Path) -> str:
    name = path.name.lower()
    for ext in (".csv.gz", ".tsv.gz", ".txt.gz"):
        if name.endswith(ext):
            return ext
    if re.search(r"\.rrf(\.[a-z]{2})?\.gz$", name):
        return ".rrf.gz"
    return path.suffix.lower()


def _first_lines(path: Path, count: int = 2) -> list[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return [handle.readline().rstrip("\n").rstrip("\r") for _ in range(count)]


def separator_of(line: str) -> str:
    if "\t" in line and line.count("\t") >= line.count(","):
        return "\t"
    return "," if "," in line else "|" if "|" in line else "\t"


def looks_like_header(line: str, second: str | None = None, *, trusted: bool = False) -> tuple[bool, str]:
    """(is a header, separator). Trusted: every field column-like. Untrusted: every field an identifier and the
    second line numeric in a column where the first line is not (otherwise a data row could be mistaken for one)."""
    separator = separator_of(line)
    fields = [f.strip().strip('"') for f in line.split(separator)]
    if not fields or not all(fields):
        return False, separator
    if trusted:
        return all(_COLUMN.match(f) and len(f.split()) <= 4 for f in fields), separator
    if not all(_IDENTIFIER.match(f) for f in fields) or second is None:
        return False, separator
    values = [v.strip().strip('"') for v in second.split(separator)]
    evidence = len(values) == len(fields) and any(_NUMBER.match(v) for v in values)
    return evidence, separator


def count_lines(path: Path) -> int:
    if path.name.endswith(".gz"):
        tool = "pigz" if shutil.which("pigz") else "gzip"
        decompress = subprocess.Popen([tool, "-dc", str(path)], stdout=subprocess.PIPE)
        counted = subprocess.run(["wc", "-l"], stdin=decompress.stdout, capture_output=True, text=True, check=True)
        decompress.stdout.close()
        decompress.wait()
        return int(counted.stdout.split()[0])
    total = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            total += chunk.count(b"\n")
    return total


def umls_columns(meta_dirs: list[Path]) -> dict[str, list[str]]:
    """File name → column names from `MRFILES.RRF(.gz)` (FIL|DES|FMT|CLS|RWS|BTS)."""
    out: dict[str, list[str]] = {}
    for meta in meta_dirs:
        for path in sorted(meta.glob("MRFILES.RRF*")):
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("|")
                    if len(parts) >= 3:
                        out[parts[0]] = parts[2].split(",")
    return out


def describe(path: Path, root: Path, *, rows: bool, umls: dict[str, list[str]], trusted: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size}
    kind = suffix(path)
    if kind not in TABULAR:
        return entry
    if kind.startswith(".rrf"):
        base = re.sub(r"(\.[a-z]{2})?\.gz$", "", path.name)
        entry["columns"] = umls.get(base)
        entry["header_line"] = False
    else:
        try:
            line, second = _first_lines(path)
        except (OSError, UnicodeError, EOFError):
            return entry
        header, separator = looks_like_header(line, second, trusted=trusted)
        entry["fields"] = len(line.split(separator))
        if header:
            entry["columns"] = [f.strip().strip('"') for f in line.split(separator)]
        entry["header_line"] = header
    if rows:
        lines = count_lines(path)
        entry["lines"] = lines
        entry["rows"] = lines - (1 if entry.get("header_line") else 0)
    return entry


def inventory(root: Path, groups: dict[str, list[str]], *, rows: bool = True, workers: int = 2,
              skip_rows: tuple[str, ...] = ()) -> dict[str, Any]:
    """Per group (a list of glob patterns under `root`): every file's size, row count and header."""
    root = Path(root)
    meta_dirs = sorted({p.parent for p in root.glob("umls-*/**/META/MRFILES.RRF*")})
    umls = umls_columns(meta_dirs)
    out: dict[str, Any] = {}
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group, patterns in groups.items():
            files = sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})
            for path in files:
                count = rows and not any(re.search(s, path.name) for s in skip_rows)
                trusted = group in TRUSTED_HEADERS and (path.name.startswith(("sct2_", "der2_")) or suffix(path) in (".csv", ".csv.gz"))
                jobs.append((group, pool.submit(describe, path, root, rows=count, umls=umls, trusted=trusted)))
        for group, job in jobs:
            out.setdefault(group, []).append(job.result())
    totals = {g: {"files": len(v), "bytes": sum(e["bytes"] for e in v)} for g, v in out.items()}
    return {"root": str(root), "groups": out, "totals": totals}


DEFAULT_GROUPS = {
    "snomed_ct": ["SnomedCT_InternationalRF2_PRODUCTION_*/**/*"],
    "umls": ["umls-2022AB-full/**/META/*"],
    "mimic_iii": ["physionet.org/files/mimiciii/1.4/*"],
    "mimic_iv": ["mimic-iv-3.1/**/*"],
    "mimic_iv_group_data": ["mimic-iv_data/**/*"],
    "hrr_atomics": ["HRR_Atomics/**/*"],
    "omop": ["omop/**/*"],
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-rows", action="store_true", help="sizes and headers only (no line counts)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-rows", nargs="*", default=[], help="regexes of file names whose lines are not counted")
    args = parser.parse_args(argv)
    result = inventory(args.root, DEFAULT_GROUPS, rows=not args.no_rows, workers=args.workers, skip_rows=tuple(args.skip_rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
