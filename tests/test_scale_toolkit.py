from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.developmental import DevelopmentalConfig, DevelopmentalDictionary
from vsa_embed.experiments import scale_toolkit as st

OPSCREEN = Path("experiments/e4-small-lm/configs/opscreen")


def _diff(a, b, prefix=""):
    """Dotted paths where two nested configs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(p for k in set(a) | set(b) for p in _diff(a.get(k), b.get(k), f"{prefix}{k}."))
    return [] if a == b else [prefix.rstrip(".")]


def test_base_config_falls_back_to_the_opscreen_recipe_with_500M_tokens(tmp_path: Path) -> None:
    (tmp_path / "configs" / "opscreen").mkdir(parents=True)
    for name in ("C5", "C0"):
        (tmp_path / "configs" / "opscreen" / f"50M-{name}-s1.yaml").write_text((OPSCREEN / f"50M-{name}-s1.yaml").read_text())
    base, source = st.base_config(tmp_path)
    assert "opscreen" in source and base["train"]["total_tokens"] == 500_000_000
    assert base["eval"]["first_tokens"] == 5_000_000 and base["eval"]["save_window_losses"] is True
    opscreen = yaml.safe_load((OPSCREEN / "50M-C5-s1.yaml").read_text())
    assert _diff(base, opscreen) == ["eval.first_tokens", "train.total_tokens"]
    # Phase 1a's config wins once it exists (copied as is).
    (tmp_path / "configs" / "scale-v1").mkdir()
    scale = {**opscreen, "experiment": "e4-scale-v1-50M-C5-s1"}
    (tmp_path / "configs" / "scale-v1" / "50M-C5-s1.yaml").write_text(yaml.safe_dump(scale))
    assert st.base_config(tmp_path) == (scale, str(tmp_path / "configs" / "scale-v1" / "50M-C5-s1.yaml"))


def test_toolkit_arms_change_only_their_own_keys() -> None:
    base, _ = st.base_config()
    c0, _ = st.base_config(condition="C0")
    configs = st.toolkit_configs(base, c0=c0)
    assert set(configs) == {"C5dev", "C5teach", "C5teachF", "C5@teachref", "C0@teachref"}
    assert _diff(configs["C5dev"], base) == ["channel.developmental", "experiment"]
    assert _diff(configs["C5teach"], base) == ["data.ontology", "eval.reference_strata", "experiment"]
    assert _diff(configs["C5teachF"], base) == ["data.ontology", "experiment"]
    assert configs["C5teach"]["data"]["ontology"].endswith("ontology-hybrid.pt")
    assert configs["C5teachF"]["data"]["ontology"].endswith("ontology-full.pt")
    assert configs["C0@teachref"]["train"]["eval_only"] is True
    assert configs["C0@teachref"]["train"]["init_from"].endswith("runs/scale-v1/50M-C0-s1/final.pt")
    assert _diff(configs["C5@teachref"], base) == ["eval.reference_strata", "experiment", "train.eval_only", "train.init_from"]
    assert all(c["seed"] == 1 for c in configs.values())
    DevelopmentalConfig(**configs["C5dev"]["channel"]["developmental"])          # every key is a tracker setting
    with pytest.raises(ValueError):
        st.toolkit_configs(c0)


def test_queue_script_queues_c5dev_only() -> None:
    lines = st.queue_lines(overhead=0.05, teacher={"hybrid_usd": 73.0, "hybrid_hours": 1.1, "full_usd": 342.0, "full_hours": 5.3})
    live = [line for line in lines if line.startswith("PYTHONPATH")]
    assert len(live) == 1 and "--priority 54.301" in live[0] and "50M-C5dev-s1.yaml" in live[0]
    assert "≈ 2.19 GPU-h" in live[0]
    commented = [line for line in lines if line.startswith("# PYTHONPATH")]
    assert {n for n in ("C5teach-s1", "C5teachF-s1", "C5@teachref-s1", "C0@teachref-s1") if any(n in line for line in commented)} == \
        {"C5teach-s1", "C5teachF-s1", "C5@teachref-s1", "C0@teachref-s1"}
    assert "NOT EXECUTED" in lines[0] and "$73" in "\n".join(lines)


def _composer() -> FrameComposer:
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 0), (1, 2)], [(1, 0), (0, 2)], [(0, 1)]])
    torch.manual_seed(3)
    return FrameComposer(schedule, 3, 2, 12, operator="hrr", normalize_atomics=True)


def test_periodic_consolidation_waits_for_the_minimum_age_and_moves_parent_weight() -> None:
    composer = _composer()
    config = DevelopmentalConfig(growth_budget=1.0, route_unobserved="parent", screen_every=5, consolidate_every=5,
                                 merge_min_age=10, merge_cosine=0.9)
    tracker = DevelopmentalDictionary(composer, None, config)
    direction = F.normalize(torch.randn(12), dim=0)
    tracker.step = 5
    card = tracker._split(0, {"direction": direction, "labels": torch.tensor([0, 1, 2]),
                              "momenta": torch.stack([direction, -direction, direction]), "gain": 2.0, "p_value": 0.001,
                              "contributions": 30})
    first, second = card["children"]
    assert tracker.parent_links[0][1:3] == (first, second)
    tracker.step = 10                                       # 5 steps old: children still near-identical, but too young
    assert [e for e in tracker.grow() if e["event"] == "merge"] == []
    tracker.step = 15
    merges = [e for e in tracker.grow() if e["event"] == "merge"]
    assert merges == [{"event": "merge", "target": "atomics", "step": 15, "kept": first, "retired": second}]
    assert bool(tracker.frozen[second]) and int((composer.schedule.fillers == second).sum()) == 0
    assert tracker.parent_links[0][3:] == (3.0, 0.0)        # the parent now follows the kept child only
    tracker.sync_parents()
    torch.testing.assert_close(composer.atomics.detach()[0], composer.atomics.detach()[first])


def test_consolidation_is_off_by_default() -> None:
    assert DevelopmentalConfig().consolidate_every == 0
    composer = _composer()
    tracker = DevelopmentalDictionary(composer, None, DevelopmentalConfig(screen_every=1, merge_cosine=0.0))
    tracker.siblings.append((0, 1))
    tracker.step = 1
    assert tracker.grow() == [] and not bool(tracker.frozen[1])
