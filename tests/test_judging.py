from pathlib import Path

import pytest

from vsa_embed.judging import JudgeClient, blind_options, calibration_agreement, fleiss_kappa, majority

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
