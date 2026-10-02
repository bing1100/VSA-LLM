"""WP-E7 baselines: Hearst patterns, OLLM/LLMs4OL-style direct prompting, random frames, EntiGraph
pairs and the cached teacher author."""

import json
import time
from pathlib import Path

import pytest

from vsa_embed.authoring_baselines import (
    TeacherAuthor, direct_prompt, entity_pairs, entigraph_prompt, hearst_extract, parse_direct, random_frames,
    teacher_schema,
)

FILLERS = {"mammal", "animal", "dog", "engine", "machine", "car", "steel", "team", "hot", "cold", "plant", "fruit"}


def resolve(key: str):
    key = key[:-1] if key.endswith("s") and key[:-1] in FILLERS else key
    return key if key in FILLERS else None


CANDIDATES = {"golden retriever": 0, "zorblax engine": 1, "glimmer fruit": 2, "tin can": 3, "striker": 4, "icy": 5}


@pytest.mark.parametrize("sentence, expected", [
    ("Large mammals such as the golden retriever and wolves roam here.", ("golden retriever", "is_a", "mammals")),
    ("Machines such as tractors, a zorblax engine or pumps break.", ("zorblax engine", "is_a", "machines")),
    ("We saw the zorblax engine and other machines.", ("zorblax engine", "is_a", "machines")),
    ("Many plants, including glimmer fruit, grow wild.", ("glimmer fruit", "is_a", "plants")),
    ("The golden retriever is a kind of dog.", ("golden retriever", "is_a", "dog")),
    ("A golden retriever is a friendly dog.", ("golden retriever", "is_a", "dog")),
    ("The zorblax engine is part of the car.", ("zorblax engine", "part_of", "car")),
    ("Each tin can is made of steel.", ("tin can", "made_of", "steel")),
    ("The striker is a member of the team.", ("striker", "member_of", "team")),
])
def test_hearst_patterns_find_the_relation(sentence: str, expected: tuple) -> None:
    assert expected[:2] + (resolve(expected[2]) or expected[2],) in [
        (c, r, resolve(f) or f) for c, r, f in hearst_extract(sentence, CANDIDATES, resolve)]


def test_hearst_needs_a_candidate_and_a_resolvable_filler() -> None:
    assert hearst_extract("Large mammals such as wolves roam here.", CANDIDATES, resolve) == []
    assert hearst_extract("The golden retriever is a kind of zorp.", CANDIDATES, resolve) == []
    assert hearst_extract("", CANDIDATES, resolve) == []


def test_hearst_is_fast_on_long_messy_text() -> None:
    text = ", ".join(["word"] * 400) + " such as " + ", ".join(["alpha beta"] * 200) + "."
    started = time.monotonic()
    hearst_extract(text * 3, CANDIDATES, resolve)
    assert time.monotonic() - started < 5.0


def test_direct_prompt_and_parse() -> None:
    prompt = direct_prompt("The zorblax engine powers the farm.", ["is_a", "part_of"],
                           [{"context": "A tabby purred.", "triples": [("tabby", "is_a", "cat")]}])
    assert prompt.endswith("Facts:\n") and "tabby | is_a | cat" in prompt
    parsed = parse_direct("Zorblax Engine | is_a | Machine\nfarm | owner | Bob\nbad line\nfarm | part_of | village\n\nText: x | is_a | y",
                          ["is_a", "part_of"])
    assert parsed == [("zorblax engine", "is_a", "machine"), ("farm", "part_of", "village")]


def test_random_frames_match_degrees_and_are_seeded() -> None:
    degrees = {"a": 3, "b": 1, "c": 0}
    frames = random_frames(degrees, ["is_a", "part_of"], list(range(50)), seed=3)
    assert {k: len(v) for k, v in frames.items()} == degrees
    assert all(len(set(v)) == len(v) for v in frames.values())
    assert frames == random_frames(degrees, ["is_a", "part_of"], list(range(50)), seed=3)
    assert frames != random_frames(degrees, ["is_a", "part_of"], list(range(50)), seed=4)


def test_entigraph_pairs_and_prompt() -> None:
    pairs = entity_pairs({0: ["a", "b", "c"], 1: ["a", "d"], 2: ["e"]}, per_entity=1, seed=0)
    assert all(x != y for x, y, _ in pairs) and {x for x, _, _ in pairs} == {"a", "b", "c", "d"}
    assert sum(x == "a" for x, _, _ in pairs) == 1
    assert "\"a\"" in entigraph_prompt("a", "b", "Some text.")


def _envelope(concepts):
    return {"structured_output": {"concepts": concepts}, "total_cost_usd": 0.01, "modelUsage": {"claude-opus-5-5": {}}}


def test_teacher_author_caches_batches_and_filters_relations(tmp_path: Path) -> None:
    calls = []

    def runner(prompt, schema, model):
        calls.append(prompt)
        assert schema == teacher_schema(["is_a", "part_of"]) and model == "claude-opus-5-5"
        return _envelope([{"id": "c0", "edges": [{"relation": "is_a", "filler": "Machine, engine"},
                                                 {"relation": "owner", "filler": "bob"}]},
                          {"id": "c1", "edges": [{"relation": "part_of", "filler": "car"}]}])

    teacher = TeacherAuthor(tmp_path / "cache", ["is_a", "part_of"], {"is_a": "kind of", "part_of": "part of"}, runner=runner)
    items = [{"id": "c0", "surface": "zorblax engine", "contexts": ["The zorblax engine failed."]},
             {"id": "c1", "surface": "spark plug", "contexts": ["A spark plug fires."]}]
    out = teacher.author(items, batch=8)
    assert out["c0"]["edges"] == [("is_a", "machine")] and out["c1"]["edges"] == [("part_of", "car")]
    assert out["c0"]["models_used"] == ["claude-opus-5-5"] and len(calls) == 1 and teacher.spent_usd == pytest.approx(0.01)
    replay = TeacherAuthor(tmp_path / "cache", ["is_a", "part_of"], {"is_a": "kind of", "part_of": "part of"},
                           runner=lambda *a: pytest.fail("cached responses must not call the CLI"))
    assert replay.author(items, batch=8) == out and replay.cached == 1
    assert "zorblax engine" in calls[0] and "The zorblax engine failed." in calls[0]


def test_teacher_author_retries_then_reports_errors(tmp_path: Path) -> None:
    attempts = []

    def flaky(prompt, schema, model):
        attempts.append(1)
        if len(attempts) < 2:
            return {"structured_output": None, "is_error": True}
        return _envelope([{"id": "c0", "edges": [{"relation": "is_a", "filler": "machine"}]}])

    teacher = TeacherAuthor(tmp_path / "a", ["is_a"], {}, runner=flaky, retries=2)
    assert teacher.author([{"id": "c0", "surface": "x", "contexts": ["x"]}])["c0"]["edges"] == [("is_a", "machine")]
    broken = TeacherAuthor(tmp_path / "b", ["is_a"], {}, runner=lambda *a: {"result": "oops"}, retries=1)
    out = broken.author([{"id": "c0", "surface": "x", "contexts": ["x"]}])
    assert out["c0"]["edges"] == [] and "no structured output" in out["c0"]["error"]
    assert broken.calls == 2 and not list((tmp_path / "b").glob("*.json"))
