from pathlib import Path

import pytest

from vsa_embed.judge_protocol import STUDIES, build_prompts, fisher_one_tailed, run_study
from vsa_embed.judging import JudgeClient


def test_every_study_has_two_paraphrases_that_format() -> None:
    fields = {"concept": "bank", "items": "river, shore", "sentence": "We sat on the bank.", "group_a": "x",
              "group_b": "y", "relation": "hypernym", "filler": "slope"}
    for study in STUDIES:
        prompts = build_prompts(study, fields)
        assert len(prompts) == 2 and all("bank" in p for p in prompts)


def test_run_study_blinds_and_aggregates(tmp_path: Path) -> None:
    runner = lambda prompt, schema, model: {"structured_output": {"score": 3 if "river" in prompt else 0}}
    client = JudgeClient(tmp_path, calls=3, runner=runner)
    items = [{"id": "a", "system": "channel", "fields": {"concept": "bank", "items": "river, shore"}},
             {"id": "b", "system": "baseline", "fields": {"concept": "bank", "items": "cat, sky"}}]
    result = run_study(client, "neighbours", items, seed=1)
    by_id = {r["id"]: r for r in result["items"]}
    assert by_id["a"]["majority"] == 3 and by_id["b"]["majority"] == 0
    assert result["fleiss_kappa"] == pytest.approx(1.0)


def test_fisher_matches_the_hrrbert_comparison_direction() -> None:
    assert fisher_one_tailed(28, 40, 4, 40) < 1e-6
    assert fisher_one_tailed(4, 40, 28, 40) > 0.99
