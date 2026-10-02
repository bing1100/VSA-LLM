import json
from pathlib import Path

import pytest

from vsa_embed.judging import JudgeClient, blind_options, calibration_agreement, fleiss_kappa, majority, schema_problem

SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}


def test_verdicts_are_cached_and_replayed(tmp_path: Path) -> None:
    calls = []
    def runner(prompt, schema, model):
        calls.append(prompt)
        return {"structured_output": {"score": len(calls)}, "total_cost_usd": 0.01, "modelUsage": {model: {}}}
    client = JudgeClient(tmp_path, calls=3, runner=runner)
    first = client.grade("item-1", ["p1", "p2"], SCHEMA)
    assert [r["verdict"]["score"] for r in first] == [1, 2, 3] and calls == ["p1", "p2", "p1"]
    again = JudgeClient(tmp_path, calls=3, runner=runner).grade("item-1", ["p1", "p2"], SCHEMA)
    assert [r["verdict"] for r in again] == [r["verdict"] for r in first] and len(calls) == 3


def test_malformed_output_is_retried_then_flagged(tmp_path: Path) -> None:
    attempts = []
    def runner(prompt, schema, model):
        attempts.append(1)
        return {"result": "not json", "is_error": False}
    record = JudgeClient(tmp_path, calls=1, retries=2, runner=runner).grade("x", ["p"], SCHEMA)[0]
    assert "error" in record and "verdict" not in record and len(attempts) == 3
    assert not list(tmp_path.glob("*.json"))   # failures are not cached


def test_blinding_is_seeded_and_reversible() -> None:
    options = {"channel": ["a", "b"], "baseline": ["c"], "random": ["d"]}
    shuffled, key = blind_options(options, seed=4)
    again, key_again = blind_options(options, seed=4)
    assert shuffled == again and key == key_again
    assert [options[key[letter]] for letter in sorted(key)] == shuffled


def test_fleiss_kappa_known_values() -> None:
    assert fleiss_kappa([[1, 1, 1], [2, 2, 2], [1, 1, 1]]) == pytest.approx(1.0)
    assert fleiss_kappa([[1, 2], [2, 1], [1, 2], [2, 1]]) < 0
    assert majority([2, 3, 2]) == 2
    assert calibration_agreement({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 1}) == {"items": 2, "accuracy": 0.5}


def test_file_exchange_posts_blinded_requests_then_ingests_valid_responses(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 3}},
              "required": ["score"]}
    client = JudgeClient(tmp_path / "cache", calls=2, exchange_dir=tmp_path / "x")
    first = client.grade("secret-item-id", ["p1", "p2"], schema)
    requests = sorted((tmp_path / "x" / "requests").glob("*.json"))
    assert [r["error"] for r in first] == ["pending", "pending"] and len(requests) == 2
    assert all("secret-item-id" not in r.read_text() for r in requests)
    posted = [json.loads(r.read_text()) for r in requests]
    by_call = {p["call"]: p for p in posted}
    Path(by_call[0]["response_path"]).write_text('{"score": 2}')
    Path(by_call[1]["response_path"]).write_text('{"score": 7}')        # out of range: stays outstanding
    second = JudgeClient(tmp_path / "cache", calls=2, exchange_dir=tmp_path / "x").grade("secret-item-id", ["p1", "p2"], schema)
    assert second[0]["verdict"] == {"score": 2} and "invalid response" in second[1]["error"]
    remaining = [json.loads(r.read_text()) for r in (tmp_path / "x" / "requests").glob("*.json")]
    assert [r["call"] for r in remaining] == [1] and "outside" in remaining[0]["last_error"]
    Path(by_call[1]["response_path"]).write_text('{"score": 1, "reason": "weak"}')
    third = JudgeClient(tmp_path / "cache", calls=2, exchange_dir=tmp_path / "x").grade("secret-item-id", ["p1", "p2"], schema)
    assert [r["verdict"]["score"] for r in third] == [2, 1]
    assert not list((tmp_path / "x" / "requests").glob("*.json"))


def test_schema_problem_rejects_wrong_types_and_values() -> None:
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["true", "false"]},
                                               "n": {"type": "integer"}}, "required": ["verdict"]}
    assert schema_problem({"verdict": "true"}, schema) is None
    assert "missing" in schema_problem({}, schema)
    assert "not in" in schema_problem({"verdict": "maybe"}, schema)
    assert "integer" in schema_problem({"verdict": "true", "n": True}, schema)
