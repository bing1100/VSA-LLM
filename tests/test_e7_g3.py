"""E7 reschedule (2026-10-20): the operational G3 gate, fractional queue priorities and the D7.2 condition subset,
and the --max-usd cap / open-licence guard on the Claude teacher and judge."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vsa_embed import jobqueue
from vsa_embed.experiments import e7_authoring, e7_g3, e7_plan
from vsa_embed.experiments.e7_round import CONDITIONS, ROUND_DEFAULTS

CONSUMER = e7_authoring.CONSUMER


def _rate(successes: int, n: int, low: float, high: float) -> dict:
    return {"value": successes / n if n else None, "low": low if n else None, "high": high if n else None, "n": n,
            "successes": successes}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_run(root: Path, *, verifiable: int = 150, masked_entries: int = 40, self_precision=(60, 200, 0.24, 0.37),
             teacher_precision=(120, 200, 0.53, 0.67), random_precision=(6, 200, 0.014, 0.064), teacher: bool = True) -> Path:
    """A run folder holding the D7.1 outputs the gate reads (contents only as far as the gate looks at them)."""
    run = root / "run"
    track = {"kind": "general", "consumer": CONSUMER, "data_root": str(root / "data"), "entry_count": 10,
             "seed": 1, "settings": e7_authoring.DEFAULTS}
    _write(run / "track.json", track)
    _write(run / "round.json", ROUND_DEFAULTS)
    candidates = [{"index": k, "surface": f"term {k}", "entry": 10 + k, "authoring_documents": [k],
                   "validation": [[0, 1]] * (6 if k < verifiable else 2)} for k in range(1000)]
    _write(run / "authoring_set.json", {"candidates": candidates})
    for path in e7_g3.required_outputs(track):
        if not (run / path).exists():
            _write(run / path, {"author": Path(path).stem, "kind": "x", "candidates": {}})
    host_votes = {f"term {k}": {"votes": [["is_a", "thing", 3]], "samples": 3} for k in range(300)}
    _write(run / "proposals" / f"{CONSUMER}.json", {"author": CONSUMER, "kind": "host", "candidates": host_votes})
    if teacher:
        _write(run / "proposals" / "teacher.json", {"author": "teacher", "kind": "teacher", "candidates": {
            f"term {k}": {"votes": [["is_a", "thing", 1]], "samples": 1} for k in range(800)}})
    authors = {CONSUMER: {"precision": _rate(*self_precision)}, "random": {"precision": _rate(*random_precision)},
               "hearst": {"precision": _rate(3, 20, 0.05, 0.36)}}
    if teacher:
        authors["teacher"] = {"precision": _rate(*teacher_precision)}
    quality = run / "quality" / "quality.json"
    _write(quality, {"authoring_set": {"candidates": 1000, "masked": masked_entries, "masked_entries": masked_entries},
                     "authors": authors, "discovery": {CONSUMER: {"@1000": {"value": 0.1}}}})
    later = max(p.stat().st_mtime for p in (run / "proposals").glob("*.json")) + 5
    os.utime(quality, (later, later))
    return run


def test_g3_passes_in_full_mode_when_self_beats_random_with_power(tmp_path: Path) -> None:
    result = e7_g3.d72_verdict(make_run(tmp_path))
    assert result["verdict"] == "PASS" and result["mode"] == "full" and result["teacher_arm"] is True
    assert result["conditions"] == list(CONDITIONS) and result["d73_planned"]
    assert result["items"]["power"]["verifiable_candidates"] == 150
    assert result["items"]["self"]["coverage"] == pytest.approx(0.3)
    assert result["items"]["judge"]["status"] == "not run"


def test_g3_reduces_to_teacher_only_when_self_is_at_chance(tmp_path: Path) -> None:
    result = e7_g3.d72_verdict(make_run(tmp_path, self_precision=(8, 200, 0.02, 0.08)))
    assert result["verdict"] == "PASS" and result["mode"] == "teacher-only"
    assert result["conditions"] == list(e7_g3.TEACHER_ONLY) and not result["d73_planned"]
    assert result["items"]["self"]["checks"] == {"edges": True, "above_random": False, "coverage": True}


def test_g3_fails_without_power_or_without_any_author_above_chance(tmp_path: Path) -> None:
    assert e7_g3.d72_verdict(make_run(tmp_path / "a", verifiable=60))["verdict"] == "FAIL"
    assert e7_g3.d72_verdict(make_run(tmp_path / "b", masked_entries=12))["verdict"] == "FAIL"
    chance = e7_g3.d72_verdict(make_run(tmp_path / "c", self_precision=(8, 200, 0.02, 0.08), teacher=False))
    assert chance["verdict"] == "FAIL" and chance["items"]["teacher"]["reason"] == "no proposals"
    few_edges = e7_g3.d72_verdict(make_run(tmp_path / "d", self_precision=(10, 20, 0.3, 0.7), teacher=False))
    assert few_edges["verdict"] == "FAIL" and few_edges["items"]["self"]["checks"]["edges"] is False


def test_g3_is_incomplete_while_outputs_are_missing_or_quality_is_stale(tmp_path: Path, capsys) -> None:
    run = make_run(tmp_path)
    (run / "proposals" / "hearst.json").unlink()
    assert e7_g3.main(["--run", str(run)]) == 2
    assert "proposals/hearst.json" in capsys.readouterr().out
    _write(run / "proposals" / "hearst.json", {"author": "hearst", "candidates": {}})       # newer than quality.json
    later = (run / "quality" / "quality.json").stat().st_mtime + 10
    os.utime(run / "proposals" / "hearst.json", (later, later))
    result = e7_g3.d72_verdict(run)
    assert result["verdict"] == "INCOMPLETE" and "re-run `quality`" in result["missing"][0]


def test_g3_cli_prints_the_verdict_and_writes_it(tmp_path: Path, capsys) -> None:
    run = make_run(tmp_path)
    assert e7_g3.main(["--run", str(run), "--stage", "d72"]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "PASS" and "mode full" in out
    assert json.loads((run / "gate" / "g3-d72.json").read_text())["mode"] == "full"


def test_g3_d73_needs_the_self_verification_and_enough_accepted_concepts(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    assert e7_g3.d73_verdict(run)["verdict"] == "INCOMPLETE"
    _write(run / "round1" / "s1" / f"verify-{CONSUMER}.json", {"summary": {"concepts_with_accepted_edges": 12, "accepted_edges": 15}})
    ontology = tmp_path / "data" / "round1" / "s1" / "self" / "ontology.pt"
    ontology.parent.mkdir(parents=True); ontology.write_bytes(b"x")
    assert e7_g3.d73_verdict(run)["verdict"] == "FAIL"
    _write(run / "round1" / "s1" / f"verify-{CONSUMER}.json", {"summary": {"concepts_with_accepted_edges": 45, "accepted_edges": 70}})
    assert e7_g3.d73_verdict(run)["verdict"] == "PASS"
    teacher_only = make_run(tmp_path / "t", self_precision=(8, 200, 0.02, 0.08))
    assert e7_g3.d73_verdict(teacher_only)["verdict"] == "FAIL"


def test_plan_takes_fractional_priorities_and_a_condition_subset(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    jobs, _ = e7_plan.stage_jobs(run, "d72", seeds=[1, 2], conditions=e7_g3.TEACHER_ONLY)
    names = [j["name"] for j in jobs]
    assert not any(step in n for n in names for step in ("-entigraph", "-spa", "-notes"))
    assert sum("-round1-" in n for n in names) == 2 * len(e7_g3.TEACHER_ONLY)
    assert sum("-verify-" in n for n in names) == 2 * 2 and sum("-base-" in n for n in names) == 2
    full, _ = e7_plan.stage_jobs(run, "d72", seeds=[1])
    assert sum("-round1-" in j["name"] for j in full) == len(CONDITIONS) and any("-verify-notes-" in j["name"] for j in full)
    with pytest.raises(ValueError):
        e7_plan.stage_jobs(run, "d72", seeds=[1], conditions=["nonsense"])
    queue = tmp_path / "queue"
    e7_plan.queue(jobs[:2], priority=jobqueue._priority("54.4965"), queue_dir=queue)
    assert {j["priority"] for j in jobqueue.jobs(queue)} == {54.4965}
    _, then = e7_plan.stage_jobs(run, "d71", seeds=[1])
    assert all("--max-usd" in line for line in then if "teacher" in line or " judge " in line)


def test_claude_calls_need_a_cap_and_open_licence_text(tmp_path: Path) -> None:
    assert e7_authoring.require_open_licence({"kind": "general"}).startswith("FineWeb-Edu")
    with pytest.raises(PermissionError):
        e7_authoring.require_open_licence({"kind": "clinical"})
    with pytest.raises(ValueError):
        e7_authoring.spend_cap(tmp_path, None)


def test_judge_stops_at_the_cap_and_records_the_budget(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write(run / "track.json", {"kind": "general"})
    fields = {"context": "a granite bridge", "concept": "granite", "relation": "is_a (a more general kind)", "filler": "rock"}
    items = [{"id": f"i{k}", "fields": fields | {"filler": f"rock {k}"}} for k in range(2)]
    (run / "judge").mkdir(parents=True)
    (run / "judge" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (run / "judge" / "key.jsonl").write_text("".join(json.dumps({"id": i["id"], "system": "SmolLM2-360M"}) + "\n" for i in items))
    calls = []

    def fake(prompt, schema, model):
        calls.append(prompt)
        return {"structured_output": {"verdict": "true"}, "total_cost_usd": 0.05}

    summary = e7_authoring.judge(run, fake=fake, max_usd=0.16)     # 2 calls fit: 0.05 + 0.10 reserve ≤ 0.16
    assert len(calls) == 2 and summary["budget"]["exhausted"] and summary["licence"].startswith("FineWeb-Edu")
    assert summary["budget"]["spent_total_usd"] == pytest.approx(0.10)
    again = e7_authoring.judge(run, fake=fake, max_usd=1.0)          # raised cap: cached verdicts reused, the rest graded
    assert len(calls) == 6 and not again["budget"]["exhausted"] and again["graded"] == 2
