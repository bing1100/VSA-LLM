from pathlib import Path

from vsa_embed.experiments import e9_plan


def test_a_fractional_level_step_keeps_a_stage_inside_one_slot(tmp_path: Path) -> None:
    paths = sorted((e9_plan.ROOT / "configs" / "t5").glob("SmolLM2-360M-full-C5-s1.yaml"))
    assert paths, "the committed T5 config is needed"
    planned: list = []
    e9_plan.queue_jobs(paths, "zz", 54.496, track="t5", plan=planned, level_step=0.0001)
    levels = sorted({level for _, level, _ in planned})
    assert levels[0] == 54.496 and levels[-1] <= 54.4963 and all(54.496 <= x < 54.497 for x in levels)
    integer: list = []
    e9_plan.queue_jobs(paths, "zz", 54, track="t5", plan=integer)
    assert {level for _, level, _ in integer} <= {54, 55, 56, 57} and all(isinstance(x, int) for _, x, _ in integer)
