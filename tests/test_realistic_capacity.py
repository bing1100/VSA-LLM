from pathlib import Path

import pytest
import torch

from vsa_embed.atomics import correlated_hypervectors
from vsa_embed.experiments.realistic_capacity import Condition, run, sample_degree, simulate_seed


def test_correlated_atomics_reach_requested_mean_cosine() -> None:
    vectors = correlated_hypervectors(256, 1024, 0.3, generator=torch.Generator().manual_seed(4))
    similarity = vectors @ vectors.T
    off_diagonal = similarity[~torch.eye(256, dtype=torch.bool)]
    assert off_diagonal.mean().item() == pytest.approx(0.3, abs=0.03)


@pytest.mark.parametrize("profile", [
    {"kind": "fixed", "value": 10, "min": 2, "max": 32},
    {"kind": "poisson", "mean": 10, "min": 2, "max": 32},
    {"kind": "power_law", "alpha": 2.5, "min": 2, "max": 32},
])
def test_degree_profiles_obey_bounds(profile: dict) -> None:
    generator = torch.Generator().manual_seed(5)
    values = [sample_degree(profile, generator) for _ in range(100)]
    assert min(values) >= 2 and max(values) <= 32


def test_observable_margin_does_not_require_target_score() -> None:
    rows = simulate_seed(
        Condition("real_hrr", 64, 0.1, "fixed", 0.0),
        {"kind": "fixed", "value": 4, "min": 2, "max": 8},
        seed=1, split="evaluation", candidate_count=16, memories=2,
    )
    assert len(rows) == 8
    assert all(row["margin"] >= 0 for row in rows)


def test_realistic_runner_writes_calibrated_artifacts(tmp_path: Path) -> None:
    config = {
        "algebras": ["real_hrr"], "dimensions": [32], "correlations": [0.0],
        "degree_profiles": ["variable"], "noise_std": [0.0],
        "degree_profile_definitions": {"variable": {"kind": "poisson", "mean": 8, "min": 2, "max": 16}},
        "calibration_seeds": [1, 2], "evaluation_seeds": [3, 4],
        "candidate_count": 32, "memories_per_seed": 4,
        "target_accepted_accuracy": 0.8, "max_ece": 0.2,
    }
    result = run(config, tmp_path)
    assert result["conditions"] == 1 and result["observations"] > 0
    for name in ["observations.csv", "summary.csv", "capacity_model.json", "resolved_config.yaml", "report.md"]:
        assert (tmp_path / name).is_file()


def test_calibration_and_evaluation_seeds_must_be_disjoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        run({"calibration_seeds": [1], "evaluation_seeds": [1]}, tmp_path)
