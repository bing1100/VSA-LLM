"""E7 gate G3, operational form (pre-registered 2026-10-20, before any D7.1 output existed).

    python -m vsa_embed.experiments.e7_g3 --run experiments/e7-self-authoring/runs/general-v1 [--stage d72|d73]

In the plan, G3 is E4's core-claim gate (125M x 500M tokens, 3 seeds). Decision 54 closed it, so it was never held.
For E7 it was to supply only two D7.2 settings, the host mode and the channel condition (decision 30), and E7 "runs
whatever G3 shows" (experiments.md, stop rules). The rule of
`experiments/e7-self-authoring/preregistration-2026-10-20-reschedule.md` §2 replaces it for E7. It reads the D7.1
outputs of the run folder and says whether D7.2 may be queued and in which mode. With `--stage d73`, it reads the D7.2
verification and says whether D7.3 may be queued.

Items for `d72` (thresholds are the module constants below):
- **complete:** every D7.1 output exists, and `quality.json` is newer than every proposals file;
- **power:** at least `MIN_VERIFIABLE` authoring-set candidates have at least `verification.min_validation`
  validation occurrences, and at least `MIN_MASKED_ENTRIES` masked gold entries are in the authoring set;
- **self / teacher (per author):** on masked gold concepts, the author's edge precision has a Wilson 95% lower bound
  above the upper bound of the degree-matched random frames, on at least `MIN_EDGES` edges. At least `MIN_COVERAGE`
  of the authoring-set candidates must also get one kept edge;
- **judge:** informational only. Calibration accuracy at least `JUDGE_MIN_CALIBRATION` and Fleiss kappa at least
  `JUDGE_MIN_KAPPA` mean the plausibility numbers are reliable; otherwise they are inconclusive.

The verdict:
- **PASS, mode `full`:** complete, power and self pass. D7.2 runs every condition, with the teacher arm whenever
  teacher proposals exist;
- **PASS, mode `teacher-only`:** complete, power and teacher pass but self fails. The plan's risk row applies ("E7
  reduces to teacher -> host"): the round-1 conditions are `TEACHER_ONLY` and D7.3 is not planned;
- **FAIL:** power fails, or both self and teacher fail. D7.2 and D7.3 are not queued;
- **INCOMPLETE:** inputs are missing.

`d73` needs the `d72` verdict PASS in mode `full`, the seed-`cross.seed` verification of the consumer, and the
materialized `self` ontology. Self must also have accepted edges for at least `MIN_ACCEPTED_CONCEPTS` candidates;
with fewer, the authored ontology is almost the visible one and E7.3 cannot test authoring.

Exit status: 0 PASS, 1 FAIL, 2 INCOMPLETE. The verdict is printed and written to `<run>/gate/g3-<stage>.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vsa_embed.experiments.e7_authoring import HOSTS, _json, data_root, kept_edges, load_track

MIN_VERIFIABLE = 100            # 10% of the 1,000-candidate authoring set (decision 33)
MIN_MASKED_ENTRIES = 30
MIN_EDGES = 30
MIN_COVERAGE = 0.10             # as the T7 host-read stage's gate (E9 addendum 2026-10-15 §7)
MIN_ACCEPTED_CONCEPTS = 30
JUDGE_MIN_CALIBRATION = 0.8
JUDGE_MIN_KAPPA = 0.4
TEACHER_ONLY = ("gold", "self", "selfnv", "teacher", "cm")
EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def required_outputs(track: dict[str, Any]) -> list[str]:
    """The D7.1 files the gate reads (the nine GPU jobs, link, the CPU baselines and quality)."""
    hosts = list(HOSTS)
    return ([f"discovery/{h}.json" for h in hosts] + ["authoring_set.json"] + [f"proposals/{h}.json" for h in hosts]
            + [f"proposals/direct-{h}.json" for h in hosts] + ["proposals/hearst.json", "proposals/random.json",
                                                              "quality/quality.json"])


def coverage(proposals: dict[str, Any], aset: dict[str, Any], *, min_share: float) -> float:
    """Share of the authoring-set candidates with at least one kept edge (host samples need `min_share` of the votes)."""
    threshold = min_share if proposals.get("kind") == "host" else 0.0
    surfaces = [c["surface"] for c in aset["candidates"]]
    kept = sum(bool(kept_edges(proposals["candidates"][s], min_share=threshold)) for s in surfaces if s in proposals["candidates"])
    return kept / max(1, len(surfaces))


def author_item(name: str, quality: dict[str, Any], proposals: dict[str, Any] | None, aset: dict[str, Any],
                *, min_share: float) -> dict[str, Any]:
    """The per-author test: precision on masked gold concepts above random frames, on enough edges, with coverage."""
    scored = quality["authors"].get(name)
    if proposals is None or scored is None:
        return {"ok": False, "present": False, "reason": "no proposals" if proposals is None else "not in quality.json"}
    precision = scored["precision"]
    random = quality["authors"].get("random", {}).get("precision") or {}
    random_high = float(random["high"]) if random.get("n") else 0.0
    share = coverage(proposals, aset, min_share=min_share)
    n = int(precision.get("n") or 0)
    low = precision.get("low")
    checks = {"edges": n >= MIN_EDGES, "above_random": low is not None and float(low) > random_high,
              "coverage": share >= MIN_COVERAGE}
    return {"ok": all(checks.values()), "present": True, "checks": checks, "precision": precision,
            "random_precision": random or None, "random_high": random_high, "coverage": share,
            "recall": scored.get("recall"), "f1": scored.get("f1"), "masked_concepts_authored": scored.get("masked_concepts_authored")}


def judge_item(run: Path) -> dict[str, Any]:
    path = run / "judge" / "results.json"
    if not path.exists():
        return {"status": "not run"}
    summary = _load(path)["summary"]
    calibration = (summary.get("calibration_accuracy") or {}).get("value")
    kappa = summary.get("fleiss_kappa")
    reliable = (calibration is not None and calibration >= JUDGE_MIN_CALIBRATION
                and kappa is not None and kappa == kappa and kappa >= JUDGE_MIN_KAPPA)
    budget = summary.get("budget") or {}
    return {"status": "reliable" if reliable else "inconclusive", "calibration_accuracy": calibration, "fleiss_kappa": kappa,
            "budget_exhausted": bool(budget.get("exhausted")), "spent_usd": summary.get("spent_usd")}


def d72_verdict(run: Path) -> dict[str, Any]:
    from vsa_embed.experiments.e7_round import CONDITIONS, round_settings
    run = Path(run)
    track = load_track(run)
    missing = [p for p in required_outputs(track) if not (run / p).exists()]
    out: dict[str, Any] = {"stage": "d72", "run": str(run), "thresholds": {
        "min_verifiable": MIN_VERIFIABLE, "min_masked_entries": MIN_MASKED_ENTRIES, "min_edges": MIN_EDGES,
        "min_coverage": MIN_COVERAGE, "judge_min_calibration": JUDGE_MIN_CALIBRATION, "judge_min_kappa": JUDGE_MIN_KAPPA}}
    if missing:
        return {**out, "verdict": "INCOMPLETE", "missing": missing}
    quality_time = (run / "quality" / "quality.json").stat().st_mtime
    stale = sorted(str(p.relative_to(run)) for p in (run / "proposals").glob("*.json") if p.stat().st_mtime > quality_time)
    if stale:
        return {**out, "verdict": "INCOMPLETE", "missing": [f"quality/quality.json is older than {stale}: re-run `quality`"]}
    quality, aset = _load(run / "quality" / "quality.json"), _load(run / "authoring_set.json")
    settings = round_settings(run)
    min_validation = int(settings["verification"]["min_validation"])
    min_share = float(track["settings"]["authoring"]["min_share"])
    verifiable = sum(len(c.get("validation") or []) >= min_validation for c in aset["candidates"])
    masked_entries = int((quality.get("authoring_set") or {}).get("masked_entries") or 0)
    power = {"ok": verifiable >= MIN_VERIFIABLE and masked_entries >= MIN_MASKED_ENTRIES, "verifiable_candidates": verifiable,
             "candidates": len(aset["candidates"]), "min_validation": min_validation, "masked_entries": masked_entries}
    consumer = track["consumer"]
    teacher_path = run / "proposals" / "teacher.json"
    self_item = author_item(consumer, quality, _load(run / "proposals" / f"{consumer}.json"), aset, min_share=min_share)
    teacher_item = author_item("teacher", quality, _load(teacher_path) if teacher_path.exists() else None, aset, min_share=min_share)
    descriptive = {name: {k: quality["authors"][name].get(k) for k in ("precision", "recall", "f1")}
                   for name in ("hearst", f"direct-{consumer}", "random") if name in quality["authors"]}
    items = {"complete": {"ok": True}, "power": power, "self": self_item, "teacher": teacher_item, "judge": judge_item(run),
             "discovery_recall_at_1000": {h: (d.get("@1000") or {}).get("value") for h, d in (quality.get("discovery") or {}).items()},
             "baselines": descriptive}
    teacher_arm = teacher_path.exists()
    if not power["ok"] or not (self_item["ok"] or teacher_item["ok"]):
        verdict, mode, conditions = "FAIL", None, []
    elif self_item["ok"]:
        verdict, mode = "PASS", "full"
        conditions = [c for c in CONDITIONS if teacher_arm or c != "teacher"]
    else:
        verdict, mode, conditions = "PASS", "teacher-only", list(TEACHER_ONLY)
    return {**out, "verdict": verdict, "mode": mode, "teacher_arm": teacher_arm if verdict == "PASS" else None,
            "conditions": conditions, "d73_planned": mode == "full", "items": items}


def d73_verdict(run: Path) -> dict[str, Any]:
    from vsa_embed.experiments.e7_round import round_settings
    run = Path(run)
    first = d72_verdict(run)
    out: dict[str, Any] = {"stage": "d73", "run": str(run), "d72": {k: first.get(k) for k in ("verdict", "mode")},
                           "thresholds": {"min_accepted_concepts": MIN_ACCEPTED_CONCEPTS}}
    if first["verdict"] != "PASS" or first["mode"] != "full":
        status = "INCOMPLETE" if first["verdict"] == "INCOMPLETE" else "FAIL"
        return {**out, "verdict": status, "reason": f"D7.3 needs the d72 verdict PASS in mode full (got {first['verdict']} {first.get('mode')})"}
    track = load_track(run)
    cross = round_settings(run)["cross"]
    seed, condition = int(cross["seed"]), cross["condition"]
    if condition != "self":
        return {**out, "verdict": "FAIL", "reason": f"cross.condition is {condition!r}; the pre-registered source is 'self'"}
    verify = run / "round1" / f"s{seed}" / f"verify-{track['consumer']}.json"
    ontology = data_root(track) / "round1" / f"s{seed}" / condition / "ontology.pt"
    missing = [str(p) for p in (verify, ontology) if not p.exists()]
    if missing:
        return {**out, "verdict": "INCOMPLETE", "missing": missing}
    summary = _load(verify)["summary"]
    accepted = int(summary.get("concepts_with_accepted_edges") or 0)
    verdict = "PASS" if accepted >= MIN_ACCEPTED_CONCEPTS else "FAIL"
    return {**out, "verdict": verdict, "seed": seed, "accepted_concepts": accepted, "accepted_edges": summary.get("accepted_edges"),
            "verifiable_candidates": summary.get("verifiable_candidates")}


def _rate(value: dict[str, Any] | None) -> str:
    if not value or value.get("value") is None:
        return "n/a"
    return f"{value['value']:.3f} [{value['low']:.3f}, {value['high']:.3f}] (n {value['n']})"


def render(result: dict[str, Any]) -> str:
    lines = [f"G3 (E7, stage {result['stage']}): {result['verdict']}"
             + (f", mode {result['mode']}" if result.get("mode") else "")]
    if result.get("missing"):
        lines += [f"  missing: {m}" for m in result["missing"]]
    if result.get("reason"):
        lines.append(f"  {result['reason']}")
    items = result.get("items") or {}
    if "power" in items:
        p = items["power"]
        lines.append(f"  power    {'ok' if p['ok'] else 'FAIL'}: verifiable candidates {p['verifiable_candidates']} / {p['candidates']} "
                     f"(>= {MIN_VERIFIABLE}); masked entries in the set {p['masked_entries']} (>= {MIN_MASKED_ENTRIES})")
    for name in ("self", "teacher"):
        a = items.get(name)
        if a is None:
            continue
        if not a["present"]:
            lines.append(f"  {name:8} FAIL: {a['reason']}")
            continue
        lines.append(f"  {name:8} {'ok' if a['ok'] else 'FAIL'}: precision {_rate(a['precision'])} vs random high "
                     f"{a['random_high']:.3f}; coverage {a['coverage']:.3f} (>= {MIN_COVERAGE}); checks {a['checks']}")
    if "judge" in items:
        lines.append(f"  judge    {items['judge']['status']} (informational)")
    if result.get("conditions"):
        lines.append(f"  D7.2 conditions: {' '.join(result['conditions'])}; teacher arm: {result['teacher_arm']}; "
                     f"D7.3 planned: {result['d73_planned']}")
    if result["stage"] == "d73" and "accepted_concepts" in result:
        lines.append(f"  self accepted edges for {result['accepted_concepts']} candidates (>= {MIN_ACCEPTED_CONCEPTS}), seed {result['seed']}")
    lines.append(result["verdict"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("experiments/e7-self-authoring/runs/general-v1"))
    parser.add_argument("--stage", choices=["d72", "d73"], default="d72")
    args = parser.parse_args(argv)
    result = d72_verdict(args.run) if args.stage == "d72" else d73_verdict(args.run)
    _json(Path(args.run) / "gate" / f"g3-{args.stage}.json", result)
    print(render(result))
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
