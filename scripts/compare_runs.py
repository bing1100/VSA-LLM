"""Compare a re-run against a committed run, cell by cell.

Usage: python scripts/compare_runs.py COMMITTED_DIR RERUN_DIR [--tolerance 0] [--ignore REGEX]

Compares every metrics.csv, metrics.jsonl, results.json and summary.json present in both directories.
Numbers match when |a - b| <= tolerance (0 = bit-identical); keys matching --ignore (default: timings
and memory) are skipped. Exit code 0 when everything matches, 1 otherwise.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

FILES = ("metrics.csv", "metrics.jsonl", "results.json", "summary.json")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


def load(path):
    if path.suffix == ".csv":
        return [dict(row) for row in csv.DictReader(path.open())]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def same(a, b, tolerance):
    if a == b:
        return True
    try:
        a, b = float(a), float(b)
        return (a != a and b != b) or abs(a - b) <= tolerance   # NaN matches NaN
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("committed", type=Path); parser.add_argument("rerun", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--ignore", default=r"(time|seconds|elapsed|duration|spent_usd|cost|peak_mem|memory_gb)")
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()
    ignore = re.compile(args.ignore)
    failed = False
    for name in FILES:
        a_path, b_path = args.committed / name, args.rerun / name
        if not (a_path.exists() and b_path.exists()):
            continue
        a, b = dict(flatten(load(a_path))), dict(flatten(load(b_path)))
        keys = [k for k in a.keys() | b.keys() if not ignore.search(k)]
        diffs = sorted(k for k in keys if not same(a.get(k), b.get(k), args.tolerance))
        print(f"{name}: {len(diffs)} differing of {len(keys)} cells (tolerance {args.tolerance})")
        for k in diffs[:args.show]:
            print(f"  {k}: {a.get(k)!r} -> {b.get(k)!r}")
        failed |= bool(diffs)
    print("MATCH" if not failed else "DIFFER")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
