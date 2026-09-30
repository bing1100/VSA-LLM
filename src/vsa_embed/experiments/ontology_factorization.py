"""Experiment 01 sandbox: strict held-out ontology factorization."""

from __future__ import annotations

import argparse
import csv
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from vsa_embed.factorization import (
    OntologyFactorizer,
    fit_factorizer,
    geometry_metrics,
    nearest_recipe_predictions,
)
from vsa_embed.provenance import prepare_output_dir, write_run_metadata


def synthetic_ontology(config: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create typed recipes and host rows from a hidden compositional teacher."""
    g = torch.Generator().manual_seed(seed)
    n = int(config["node_count"])
    role_count = int(config["role_count"])
    values_per_role = int(config["values_per_role"])
    value_count = values_per_role
    possible = values_per_role**role_count
    if n > possible:
        raise ValueError("node_count exceeds the number of unique recipes")
    # Values are deliberately shared across roles. A bag-of-factors therefore
    # cannot recover which value occupied which slot, while typed binding can.
    universe = torch.tensor(list(itertools.product(range(values_per_role), repeat=role_count)))
    recipes = universe[torch.randperm(possible, generator=g)[:n]]

    teacher = OntologyFactorizer(
        value_count, role_count, int(config["teacher_vsa_dimension"]),
        int(config["host_dimension"]), algebra=config["teacher_algebra"], typed=True,
    )
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=g) / max(1, parameter.shape[-1])**0.5)
        targets = teacher(recipes)
        targets += float(config.get("target_noise_std", 0.0)) * torch.randn(targets.shape, generator=g)

    holdout_count = int(config["holdout_count"])
    if n <= holdout_count:
        raise ValueError("holdout_count must be smaller than node_count")
    for _ in range(100):
        holdout = torch.randperm(n, generator=g)[:holdout_count]
        train_mask = torch.ones(n, dtype=torch.bool); train_mask[holdout] = False
        # Every (role, value) atomic required by held-out recipes must occur in
        # training, while complete recipes remain disjoint by construction.
        supported = all(
            set(recipes[holdout, role].tolist()) <= set(recipes[train_mask, role].tolist())
            for role in range(role_count)
        )
        if supported:
            return recipes, targets.detach(), holdout.sort().values
    raise RuntimeError("could not create an atomic-supported holdout split")


def _fit_method(
    method: str, recipes: torch.Tensor, targets: torch.Tensor, train: torch.Tensor,
    test: torch.Tensor, config: dict[str, Any], seed: int,
) -> tuple[torch.Tensor, dict[str, Any], OntologyFactorizer | None]:
    if method == "train_mean":
        return targets[train].mean(0).expand(test.numel(), -1), {}, None
    if method == "nearest_recipe":
        return nearest_recipe_predictions(recipes, targets, train, test), {}, None
    typed = method != "bag_of_factors"
    used_recipes = recipes
    if method == "shuffled_structure":
        g = torch.Generator().manual_seed(seed + 991)
        used_recipes = recipes.clone()
        for role in range(recipes.shape[1]):
            used_recipes[:, role] = used_recipes[torch.randperm(recipes.shape[0], generator=g), role]
    torch.manual_seed(seed + 100)
    model = OntologyFactorizer(
        int(recipes.max()) + 1, recipes.shape[1], int(config["vsa_dimension"]), targets.shape[1],
        algebra=config["algebra"], typed=typed,
    )
    fit = fit_factorizer(
        model, used_recipes, targets, train, steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]), cosine_weight=float(config["cosine_weight"]),
    )
    with torch.no_grad():
        predictions = model(used_recipes[test])
    return predictions, {"initial_loss": fit.initial_loss, "final_loss": fit.final_loss}, model


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    prepare_output_dir(output_dir)
    rows: list[dict[str, Any]] = []
    saved_model: OntologyFactorizer | None = None
    for seed in config["seeds"]:
        recipes, targets, test = synthetic_ontology(config["synthetic"], int(seed))
        mask = torch.ones(recipes.shape[0], dtype=torch.bool); mask[test] = False
        train = mask.nonzero().flatten()
        for method in config["methods"]:
            predicted, diagnostics, model = _fit_method(method, recipes, targets, train, test, config["factorization"], int(seed))
            metrics = geometry_metrics(predicted, targets[test], k=int(config["knn_k"]))
            degrees = (recipes[test].unsqueeze(1) == recipes.unsqueeze(0)).any(dim=2).sum(dim=1)
            low = degrees <= degrees.median()
            for band, selector in (("all", torch.ones_like(low, dtype=torch.bool)), ("low_degree", low), ("high_degree", ~low)):
                if selector.any():
                    band_metrics = geometry_metrics(predicted[selector], targets[test][selector], k=int(config["knn_k"]))
                    rows.append({"seed": seed, "method": method, "degree_band": band, "n": int(selector.sum()), **diagnostics, **band_metrics})
            if method == "typed_vsa" and saved_model is None:
                saved_model = model
    _write_csv(output_dir / "metrics.csv", rows)
    all_rows = [r for r in rows if r["degree_band"] == "all"]
    means = {method: sum(r["knn_overlap"] for r in all_rows if r["method"] == method) / len(config["seeds"]) for method in config["methods"]}
    strongest_control = max((score, method) for method, score in means.items() if method != "typed_vsa")
    delta = means["typed_vsa"] - strongest_control[0]
    decision = delta >= float(config["acceptance"]["min_knn_gain"])
    artifact = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "algebra": config["factorization"]["algebra"], "dimension": config["factorization"]["vsa_dimension"],
        "role_count": config["synthetic"]["role_count"], "value_count": config["synthetic"]["values_per_role"],
        "state_dict": saved_model.state_dict() if saved_model is not None else {},
    }
    torch.save(artifact, output_dir / "factorizer.pt")
    write_run_metadata(output_dir, config, device="cpu")
    report = [
        "# Experiment 01 synthetic factorization sandbox", "",
        "| Method | Mean held-out kNN overlap |", "|---|---:|",
        *[f"| {m} | {means[m]:.3f} |" for m in config["methods"]], "",
        f"Strongest control: **{strongest_control[1]} ({strongest_control[0]:.3f})**.",
        f"Typed-VSA gain: **{delta:+.3f}**; sandbox gate: **{'PASS' if decision else 'FAIL'}**.", "",
        "This synthetic result validates code direction only. It does not establish pretrained-LLM geometry or behavior.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")
    return {"rows": len(rows), "gate_passed": decision, "gain": delta}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(yaml.safe_load(args.config.read_text()), args.output)
    print(f"Completed {result['rows']} metric rows; gate={'PASS' if result['gate_passed'] else 'FAIL'}; gain={result['gain']:+.3f}")


if __name__ == "__main__":
    main()