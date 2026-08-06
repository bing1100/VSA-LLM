from pathlib import Path

import yaml

from vsa_embed.experiments.ontology_factorization import run, synthetic_ontology


def tiny_config() -> dict:
    return {
        "seeds": [1], "methods": ["typed_vsa", "train_mean"], "knn_k": 2,
        "synthetic": {"node_count": 40, "holdout_count": 10, "role_count": 3, "values_per_role": 4,
                      "teacher_vsa_dimension": 12, "host_dimension": 8, "teacher_algebra": "real_hrr", "target_noise_std": 0.0},
        "factorization": {"algebra": "real_hrr", "vsa_dimension": 12, "steps": 20,
                          "learning_rate": 0.03, "cosine_weight": 1.0},
        "acceptance": {"min_knn_gain": -1.0},
    }


def test_synthetic_split_has_atomic_support_and_no_overlap() -> None:
    config = tiny_config()["synthetic"]
    recipes, targets, holdout = synthetic_ontology(config, 5)
    train = set(range(recipes.shape[0])) - set(holdout.tolist())
    assert targets.shape == (40, 8)
    assert train.isdisjoint(holdout.tolist())
    for role in range(recipes.shape[1]):
        for value in recipes[holdout, role].unique().tolist():
            assert value in recipes[list(train), role].unique().tolist()
    assert len({tuple(row) for row in recipes.tolist()}) == recipes.shape[0]


def test_runner_writes_versioned_artifacts(tmp_path: Path) -> None:
    result = run(tiny_config(), tmp_path)
    assert result["rows"] == 6
    for name in ("metrics.csv", "factorizer.pt", "resolved_config.yaml", "manifest.json", "report.md"):
        assert (tmp_path / name).is_file()
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())["seeds"] == [1]