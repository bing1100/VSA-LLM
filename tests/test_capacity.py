from pathlib import Path

import yaml

import pytest

from vsa_embed.experiments.capacity import Trial, _trial_grid, run, run_trial


def test_unitary_hrr_retrieves_small_bundle() -> None:
    trial = Trial("unitary_hrr", 256, 4, 0.0, "float32", 17)
    result = run_trial(trial, candidate_count=64, query_count=4, device="cpu")
    assert result["top1"] == 1.0
    assert result["mrr"] == 1.0


def test_trial_grid_is_cartesian_product() -> None:
    config = {
        "algebras": ["real_hrr", "map"],
        "dimensions": [64, 128],
        "bundle_sizes": [2],
        "noise_std": [0.0],
        "dtypes": ["float32"],
        "seeds": [1, 2],
    }
    assert len(_trial_grid(config)) == 8


def test_design_only_axes_are_not_silently_ignored() -> None:
    config = {
        "algebras": ["real_hrr"], "dimensions": [64], "bundle_sizes": [2],
        "noise_std": [0.0], "dtypes": ["float32"], "seeds": [1],
        "path_depths": [1, 2],
    }
    with pytest.raises(ValueError, match="not implemented"):
        _trial_grid(config)


def test_runner_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    config = {
        "algebras": ["unitary_hrr"],
        "dimensions": [64],
        "bundle_sizes": [2],
        "noise_std": [0.0],
        "dtypes": ["float32"],
        "seeds": [1],
        "candidate_count": 32,
        "query_count": 2,
    }
    results = run(config, tmp_path)
    assert len(results) == 1
    for name in ["metrics.csv", "resolved_config.yaml", "manifest.json", "report.md"]:
        assert (tmp_path / name).is_file()
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text()) == config
