"""Experiment 01b Stage A: synthetic relation teacher × learner matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from vsa_embed.global_local import (
    GlobalLocalRelationalModel,
    balanced_prefix,
    fit_global_local,
    make_synthetic_relations,
    relational_metrics,
)
from vsa_embed.provenance import prepare_output_dir, write_run_metadata


def _spec(value: str | dict[str, Any]) -> dict[str, Any]:
    return {"family": value} if isinstance(value, str) else value


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run nested learning curves and write reproducible Stage-A artifacts."""
    git_at_start = prepare_output_dir(output_dir)
    synthetic = config["synthetic"]
    fit_config = config["fit"]
    budgets = sorted(int(x) for x in config["data_budgets"])
    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}

    for seed in config["seeds"]:
        for teacher_value in config["teachers"]:
            teacher = _spec(teacher_value)
            for residual_dimension in config["residual_dimensions"]:
                full_train, test, hidden_teacher = make_synthetic_relations(
                    family=teacher["family"], relation_count=int(synthetic["relation_count"]),
                    dimension=int(synthetic["dimension"]), train_per_relation=max(budgets),
                    test_per_relation=int(synthetic["test_per_relation"]),
                    rank=int(teacher.get("rank", synthetic.get("rank", 8))),
                    edge_feature_dimension=int(synthetic["edge_feature_dimension"]),
                    concept_feature_dimension=int(synthetic["concept_feature_dimension"]),
                    residual_dimension=int(residual_dimension), noise_std=float(synthetic["noise_std"]),
                    seed=int(seed),
                )
                for budget in budgets:
                    train = balanced_prefix(full_train, budget, int(synthetic["relation_count"]))
                    for learner_value in config["learners"]:
                        learner = _spec(learner_value)
                        torch.manual_seed(int(seed) + 100_000 + budget)
                        model = GlobalLocalRelationalModel(
                            int(synthetic["relation_count"]), int(synthetic["dimension"]),
                            family=learner["family"], rank=int(learner.get("rank", synthetic.get("rank", 8))),
                            edge_feature_dimension=int(synthetic["edge_feature_dimension"]),
                            concept_feature_dimension=int(synthetic["concept_feature_dimension"]),
                            residual_dimension=int(residual_dimension),
                        )
                        fit = fit_global_local(
                            model, train, steps=int(fit_config["steps"]),
                            learning_rate=float(fit_config["learning_rate"]),
                            weight_supervision=float(fit_config["weight_supervision"]),
                            threshold=float(fit_config["threshold"]),
                        )
                        train_metrics = relational_metrics(model, train)
                        test_metrics = relational_metrics(model, test)
                        groups = model.parameter_groups()
                        row = {
                            "seed": int(seed), "teacher": teacher["family"], "learner": learner["family"],
                            "teacher_rank": int(teacher.get("rank", 0)), "learner_rank": int(learner.get("rank", 0)),
                            "data_per_relation": budget, "residual_dimension": int(residual_dimension),
                            "train_examples": len(train), "test_examples": len(test),
                            "parameters": _parameter_count(model), "initial_loss": fit.initial_loss,
                            "final_loss": fit.final_loss, "threshold_step": fit.threshold_step or -1,
                            **{f"train_{key}": value for key, value in train_metrics.items()},
                            **{f"test_{key}": value for key, value in test_metrics.items()},
                            "generalization_gap": test_metrics["mse"] - train_metrics["mse"], **groups,
                        }
                        rows.append(row)
                        if budget == max(budgets) and int(seed) == int(config["seeds"][0]):
                            key = f"{teacher['family']}-r{residual_dimension}-{learner['family']}"
                            checkpoints[key] = {"model": model.state_dict(), "parameters": groups}

    _write_csv(output_dir / "metrics.csv", rows)
    with (output_dir / "metrics.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    torch.save({"schema_version": 1, "models": checkpoints}, output_dir / "relation_transforms.pt")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", split="nested_relation_balanced_prefix_with_fixed_test",
        edge_specific_parameters=0, concept_specific_parameters=0,
    )
    summary = summarize(rows, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(render_report(summary, rows, config))
    return summary


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    max_budget = max(int(x) for x in config["data_budgets"])
    final = [row for row in rows if row["data_per_relation"] == max_budget]
    cells: dict[str, dict[str, float]] = {}
    for teacher in sorted({row["teacher"] for row in final}):
        cells[teacher] = {}
        for learner in sorted({row["learner"] for row in final}):
            selected = [row["test_mse"] for row in final if row["teacher"] == teacher and row["learner"] == learner]
            cells[teacher][learner] = sum(selected) / len(selected)
    winners = {teacher: min(scores, key=scores.get) for teacher, scores in cells.items()}
    aligned_wins = sum(winner == teacher for teacher, winner in winners.items() if teacher in cells[teacher])
    eligible = sum(teacher in cells[teacher] for teacher in cells)
    gate = aligned_wins >= int(config["acceptance"]["min_aligned_wins"])
    return {
        "conditions": len(rows), "max_budget": max_budget, "mean_test_mse": cells,
        "winner_by_teacher": winners, "aligned_wins": aligned_wins, "eligible_teachers": eligible,
        "gate_passed": gate,
    }


def render_report(summary: dict[str, Any], rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    learners = sorted({row["learner"] for row in rows})
    lines = [
        "# Experiment 01b Stage A — synthetic teacher × learner matrix", "",
        f"Conditions: **{summary['conditions']}**. Maximum data budget: **{summary['max_budget']} edges/relation**.", "",
        "## Mean held-out MSE at maximum budget", "",
        "| Teacher | " + " | ".join(learners) + " | Winner |",
        "|---|" + "---:|" * len(learners) + "---|",
    ]
    for teacher, scores in summary["mean_test_mse"].items():
        lines.append("| " + teacher + " | " + " | ".join(f"{scores[x]:.5f}" for x in learners) +
                     f" | {summary['winner_by_teacher'][teacher]} |")
    lines.extend([
        "", "## Stage-A code gate", "",
        f"Aligned-family wins: **{summary['aligned_wins']}/{summary['eligible_teachers']}**; "
        f"gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**.", "",
        "This is an identifiability and implementation gate, not evidence for pretrained-model benefit. "
        "Inspect `metrics.csv` for data-to-threshold, generalization gaps, salience recovery, and residual strata.", "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(yaml.safe_load(args.config.read_text()), args.output)
    print(f"Completed {result['conditions']} conditions; gate={'PASS' if result['gate_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
