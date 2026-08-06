from pathlib import Path

import yaml

from vsa_embed.experiments.global_local_relations import run


def test_stage_a_runner_writes_complete_artifacts(tmp_path: Path) -> None:
    config = {
        "seeds": [1],
        "teachers": [{"family": "hrr"}, {"family": "diagonal"}],
        "learners": [{"family": "additive"}, {"family": "hrr"}, {"family": "diagonal"}],
        "data_budgets": [8, 16],
        "residual_dimensions": [0],
        "synthetic": {
            "relation_count": 2, "dimension": 8, "rank": 2, "test_per_relation": 12,
            "edge_feature_dimension": 4, "concept_feature_dimension": 5, "noise_std": 0.0,
        },
        "fit": {"steps": 80, "learning_rate": 0.04, "weight_supervision": 0.1, "threshold": 0.01},
        "acceptance": {"min_aligned_wins": 1},
    }
    result = run(config, tmp_path)
    assert result["conditions"] == 12
    for name in ("metrics.csv", "metrics.jsonl", "relation_transforms.pt", "resolved_config.yaml",
                 "manifest.json", "summary.json", "report.md"):
        assert (tmp_path / name).is_file()
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())["seeds"] == [1]
