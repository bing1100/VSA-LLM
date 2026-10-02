import sys
from pathlib import Path

import pytest
import torch
import yaml

from vsa_embed import jobqueue
from vsa_embed.experiments import e4_plan
from vsa_embed.training.lm import build_channel, eval_token_schedule, resolve_config


def _ontology(path: Path, entries: int = 500, atomics: int = 120, relations: int = 6) -> None:
    """Synthetic channel ontology with the keys `c3_corpus` writes to `ontology.pt`."""
    generator = torch.Generator().manual_seed(0)
    degrees = torch.randint(1, 5, (entries,), generator=generator)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), degrees.cumsum(0)])
    edges = int(offsets[-1])
    torch.save({"entry_count": entries, "atomic_count": atomics, "relation_count": relations, "offsets": offsets,
                "relations": torch.randint(0, relations, (edges,), generator=generator),
                "fillers": torch.randint(0, atomics, (edges,), generator=generator),
                "heldout_entries": list(range(0, entries, 10)), "train_frequency": [5] * entries,
                "alias_table_sha256": "a" * 64, "holdout_sha256": "b" * 64,
                "relation_names": [f"r{i}" for i in range(relations)], "atomic_names": [f"a{i}" for i in range(atomics)],
                "entry_concepts": list(range(entries)), "concept_names": [f"c{i}" for i in range(entries)]}, path)


@pytest.fixture
def plan(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(e4_plan, "ROOT", tmp_path / "e4")
    monkeypatch.setattr(jobqueue, "DEFAULT_DIR", tmp_path / "jobs")
    data = tmp_path / "data"
    data.mkdir()
    _ontology(data / "ontology.pt")
    return {"data": data, "configs": tmp_path / "e4" / "configs", "jobs": tmp_path / "jobs"}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_pilot_preset_writes_the_s0_grid_and_queues_it_first(plan: dict[str, Path]) -> None:
    e4_plan.main(["--stage", "pilot", "--data-root", str(plan["data"]), "--queue"])
    names = sorted(p.stem for p in (plan["configs"] / "pilot").glob("*.yaml"))
    assert names == [f"50M-{c}-s{s}" for c in ("C0", "C1", "C2", "C3", "C5") for s in (1, 2)]
    config = _load(plan["configs"] / "pilot" / "50M-C5-s2.yaml")
    assert config["experiment"] == "e4-pilot-50M-C5-s2" and config["seed"] == 2
    assert config["train"]["total_tokens"] == 100_000_000 and config["train"]["warmup_tokens"] == 5_000_000
    assert config["train"]["micro_batch"] == 32 and config["train"]["lr"] == 1e-3 and config["train"]["checkpoint_minutes"] == 5
    assert config["eval"] == {"windows": 512, "batch": 32, "first_tokens": 10_000_000, "save_window_losses": True}
    assert config["channel"]["context_window"] == 8 and config["data"]["ontology"] == str(plan["data"] / "ontology.pt")
    resolved = resolve_config(config)
    per_step = 1024 * 32 * 8
    assert eval_token_schedule(10_000_000, (100_000_000 // per_step) * per_step) == [
        10_000_000, 20_000_000, 40_000_000, 80_000_000, 99_876_864]
    assert resolved["eval"]["save_window_losses"] is True
    queued = jobqueue.jobs(plan["jobs"])
    assert len(queued) == 10 and {j["priority"] for j in queued} == {10}
    job = next(j for j in queued if j["name"] == "pilot-50M-C5-s2")
    assert job["command"][0] == sys.executable and job["command"][-1].endswith("runs/pilot/50M-C5-s2")


def test_shakeout_adds_a_kill_and_resume_pair(plan: dict[str, Path]) -> None:
    e4_plan.main(["--stage", "shakeout", "--data-root", str(plan["data"]), "--queue"])
    names = sorted(p.stem for p in (plan["configs"] / "shakeout").glob("*.yaml"))
    assert len(names) == 9 and "50M-C5@resume-s1" in names
    stop = _load(plan["configs"] / "shakeout" / "50M-C5@resume-s1.yaml")
    twin = _load(plan["configs"] / "shakeout" / "50M-C5-s1.yaml")
    assert stop["train"].pop("stop_after_steps") == 381 == (200_000_000 // (1024 * 32 * 8)) // 2
    assert {k: v for k, v in stop.items() if k != "experiment"} == {k: v for k, v in twin.items() if k != "experiment"}
    queued = {j["name"]: j for j in jobqueue.jobs(plan["jobs"])}
    assert len(queued) == 10 and {j["priority"] for j in queued.values()} == {50}
    first, resume = queued["shakeout-50M-C5@resume-s1"], queued["shakeout-50M-C5@resume-s1-resume"]
    assert resume["command"] == [*first["command"], "--resume"] and first["created"] < resume["created"]


def test_baselines_use_the_committed_budgets_at_lowest_priority(plan: dict[str, Path]) -> None:
    e4_plan.main(["--stage", "baselines", "--data-root", str(plan["data"]), "--queue"])
    names = sorted(p.stem for p in (plan["configs"] / "baselines").glob("*.yaml"))
    assert names == [f"{size}-C0-s{s}" for size in ("125M", "50M") for s in (1, 2, 3)]
    assert _load(plan["configs"] / "baselines" / "125M-C0-s3.yaml")["train"]["total_tokens"] == 500_000_000
    small = _load(plan["configs"] / "baselines" / "50M-C0-s1.yaml")
    assert small["train"]["total_tokens"] == 300_000_000 and small["eval"]["windows"] == 1024
    assert {j["priority"] for j in jobqueue.jobs(plan["jobs"])} == {90}


def test_explicit_conditions_keep_the_old_behaviour(plan: dict[str, Path]) -> None:
    e4_plan.main(["--stage", "screen", "--conditions", "C0", "C1", "--seeds", "1", "--data-root", str(plan["data"]), "--queue"])
    config = _load(plan["configs"] / "screen" / "50M-C1-s1.yaml")
    assert config["eval"] == {"windows": 1024, "batch": 32, "first_tokens": 10_000_000}
    assert config["train"]["total_tokens"] == 300_000_000
    assert {j["priority"] for j in jobqueue.jobs(plan["jobs"])} == {50}
    with pytest.raises(ValueError, match="no preset"):
        e4_plan.stage_blocks("screen")
    blocks = e4_plan.stage_blocks("pilot", seeds=[3], overrides={"train": {"lr": 5e-4}})
    assert blocks[0]["seeds"] == [3] and blocks[0]["overrides"]["train"] == {
        "total_tokens": 100_000_000, "warmup_tokens": 5_000_000, "checkpoint_minutes": 5, "lr": 5e-4}


def test_matched_controls_have_the_composition_channels_parameters(plan: dict[str, Path]) -> None:
    ontology = torch.load(plan["data"] / "ontology.pt", weights_only=False)
    free_dim, buckets = e4_plan.matched_sizes(plan["data"] / "ontology.pt", 512, 256)
    table = e4_plan.conditions("hrr", 8, 256, free_dim, buckets, {})

    def channel_parameters(name: str) -> int:
        config = resolve_config(_merge({"model": {"size": "50M"}, "device": "cpu"}, table[name]))
        channel, context = build_channel(config, ontology, 512)
        return sum(p.numel() for p in channel.parameters()) + (sum(p.numel() for p in context.parameters()) if context else 0)

    c3 = channel_parameters("C3")
    for control in ("C2", "C1h"):
        assert channel_parameters(control) == pytest.approx(c3, rel=0.02)
    assert channel_parameters("C1") < 0.01 * c3            # fixed random table: only gate and scale train


def _merge(a: dict, b: dict) -> dict:
    return e4_plan._merge(a, b)
