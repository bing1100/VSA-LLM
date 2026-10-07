"""E9 dimension 3 (claim C): item × seed statistics and the power analysis behind the `-v2` items (decision 56).

**Unit and model.** A new word is one concept with several items (≈ 5 property items, 2 entailment items, ≈ 5
statements); an edit is one concept with an efficacy and a paraphrase item. Items of a concept are not independent, so
the unit is the concept: per concept and training seed the paired difference of the concept's mean item outcome
(candidate own rows − reference, e.g. C5 − C0′; seeds paired by index, P0 standing for every seed), giving a complete
concepts × seeds table. The table is analysed with a crossed random-effects model `d[c, s] = μ + a_c + b_s + e_cs`
(`statistics.crossed_components`: ANOVA = REML variance components for the balanced table, Satterthwaite t interval),
with a two-way cluster bootstrap that resamples concepts and seeds independently (`two_way_cluster_bootstrap`) as the
distribution-free check. The current R9 analysis (seed-averaged items, bootstrap over items) ignores both the seed
component and the clustering of items in concepts; it is reported next to the model for comparison.

**Power.** For a planned number of seeds `S` the mean's variance is `σ²_c / n + σ²_s / S + σ²_e / (n S)`; the concepts
needed for a two-sided test at `α / m` (the Bonferroni bound of Holm over the `m` tests of a family) to reach `power`
at a difference `Δ` follow from the estimated components (`statistics.required_clusters`; None when the seed component
alone exceeds the budget, i.e. more items cannot help). Families: new words — property, entailment, paraphrase,
statement accuracy (m = 4); edits — the log-odds move beyond the control edit (`d_after − d_control`) and the success
share beyond the control (`new-preferred after − control`) on efficacy and paraphrase prompts, seen and held-out terms
(m = 4 per metric); the WP-C7 track zero-shot items — property, entailment, paraphrase (m = 3).

    python -m vsa_embed.experiments.e9_power --runs experiments/e9-retrofit/runs/t5 experiments/e9-retrofit/runs/t5-qwen3
        --output experiments/e9-retrofit/power/claim-c-v1 [--folder edit] [--deltas 0.05 0.06] [--planned-seeds 3]
        [--power 0.8] [--alpha 0.05] [--targets 700] [--overwrite]

Output (run-folder contract): `power.json`, `report.md`, `resolved_config.yaml`, `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import crossed_components, holm_adjust, power_at, required_clusters, two_way_cluster_bootstrap
from . import e9_ontology_edit as edit

NEW_PRIMARY = ("property", "entailment", "paraphrase", "statement_accuracy")
EDIT_TESTS = ("efficacy", "paraphrase")
EDIT_SUBSETS = ("seen", "heldout")
EDIT_METRICS = ("log_odds", "success")         # d_after − d_control; new-preferred share after − control
ZS_TESTS = ("property", "entailment", "paraphrase")
NEW_WITHIN = ("random_frame", "none", "mean_row")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def edit_folder_name(variant: str = "bf16", items_version: str = "v1") -> str:
    """Folder of a run's ontology-editing evaluation: `edit` / `edit-int4` for the v1 items, `edit-v2` / `edit-v2-int4`
    for the v2 items (never the v1 folders, so v1 results are kept)."""
    base = "edit" if items_version == "v1" else f"edit-{items_version}"
    return base if variant == "bf16" else f"{base}-{variant}"


# ---------------------------------------------------------------- loaders: per concept values of one run


def _by_concept(values: dict[str, tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for concept, value in values.values():
        grouped[concept].append(value)
    return {c: float(np.mean(v)) for c, v in grouped.items()}


def new_word_items(folder: Path, source: str = "own") -> dict[str, dict[str, tuple[str, float]]] | None:
    """Per test: item id → (concept, value) of one source over the linked concepts (the values of
    `e9_ontology_edit.new_word_vectors`); None if the folder has no new-word evaluation or no such source."""
    path = Path(folder) / "summary.json"
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if "new_words" not in document:
        return None
    linked = {c for c, r in document["new_words"]["resolved"].items() if r["status"] == "linked"}
    rows = [r for r in _read_jsonl(Path(folder) / "predictions.jsonl") if r.get("part") == "new" and r.get("source") == source]
    if not rows:
        return None
    concept_of = {r["id"]: r["concept"] for r in rows}
    vectors = edit.new_word_vectors({"prompts": [r for r in rows if r["test"] in {"property", "entailment"}],
                                     "statements": [r for r in rows if r["test"] == "statement"]}, linked)
    return {test: {i: (concept_of[i], float(v)) for i, v in zip(ids, values.tolist())} for test, (ids, values) in vectors.items()}


def new_word_values(folder: Path, source: str = "own") -> dict[str, dict[str, float]] | None:
    """Per test: concept → mean of its linked items."""
    items = new_word_items(folder, source)
    return None if items is None else {test: _by_concept(values) for test, values in items.items()}


def edit_values(folder: Path) -> dict[str, dict[str, dict[str, dict[str, float]]]] | None:
    """subset (`all`, `seen`, `heldout`) → test (`efficacy`, `paraphrase`) → metric → edit concept → value over the
    linked edits: `log_odds` = d_after − d_control (the move toward the specified filler beyond the control edit, per
    edit: EM − EM_control), `magnitude` = d_after − d_before, `success` = new-preferred share after − control."""
    path = Path(folder) / "summary.json"
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    if "edits" not in document:
        return None
    resolved = document["edits"]["resolved"]
    by_item: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in _read_jsonl(Path(folder) / "predictions.jsonl"):
        if row.get("part") == "edit":
            by_item[row["id"]][row["condition"]] = row
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {s: {t: {m: {} for m in (*EDIT_METRICS, "magnitude")} for t in EDIT_TESTS}
                                                              for s in ("all", *EDIT_SUBSETS)}
    for item_id, conditions in by_item.items():
        concept, _, test = item_id.rpartition("-")
        if test not in EDIT_TESTS or concept not in resolved or resolved[concept]["status"] != "linked":
            continue
        if not all(c in conditions for c in edit.EDIT_CONDITIONS):
            continue
        before, after, control = (conditions[c] for c in edit.EDIT_CONDITIONS)
        values = {"log_odds": after["d_mean"] - control["d_mean"], "magnitude": after["d_mean"] - before["d_mean"],
                  "success": after["new_preferred"] - control["new_preferred"]}
        subset = "heldout" if resolved[concept].get("entry_status") == "heldout" else "seen"
        for s in ("all", subset):
            for metric, value in values.items():
                out[s][test][metric][concept] = float(value)
    return out


def zeroshot_items(folder: Path, source: str = "own", split: str | None = None) -> dict[str, dict[str, tuple[str, float]]] | None:
    """WP-C7 track zero-shot items (`e9_tracks zeroshot`): per test, item id → (term, value) over linked terms
    (optionally one split: `synthetic` or `heldout`)."""
    path = Path(folder) / "summary.json"
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    linked = {c for c, r in document["resolved"].items() if r["linked"] and r["status"] != "mislinked"}
    rows = [r for r in _read_jsonl(Path(folder) / "predictions.jsonl")
            if r.get("source") == source and r.get("concept") in linked and (split is None or r.get("split") == split)]
    if not rows:
        return None
    prop = [r for r in rows if r["test"] == "property"]
    pick = lambda chosen, key: {r["id"]: (r["concept"], float(r[key])) for r in chosen}
    return {"property": pick(prop, "correct"), "entailment": pick([r for r in rows if r["test"] == "entailment"], "correct"),
            "paraphrase": pick([r for r in prop if r.get("consistent") is not None], "consistent")}


# ---------------------------------------------------------------- concepts × seeds tables and comparisons


def paired_table(candidate: dict[int, dict[str, float]], reference: dict[int, dict[str, float]] | None = None
                 ) -> tuple[list[str], list[int], np.ndarray]:
    """(concepts, seeds, concepts × seeds table of candidate − reference) over the concepts every seed has; without a
    reference the candidate's own values (a one-sample table)."""
    seeds = sorted(s for s in candidate if reference is None or s in reference)
    if not seeds:
        return [], [], np.zeros((0, 0))
    common = set.intersection(*(set(candidate[s]) for s in seeds))
    if reference is not None:
        common &= set.intersection(*(set(reference[s]) for s in seeds))
    concepts = sorted(common)
    table = np.asarray([[candidate[s][c] - (reference[s][c] if reference is not None else 0.0) for s in seeds] for c in concepts],
                       dtype=np.float64).reshape(len(concepts), len(seeds))
    return concepts, seeds, table


def seed_averaged_items(candidate: dict[int, dict[str, tuple[str, float]]], reference: dict[int, dict[str, tuple[str, float]]] | None
                        ) -> np.ndarray:
    """Item-level paired differences averaged over the common seeds (the current R9 unit)."""
    seeds = sorted(s for s in candidate if reference is None or s in reference)
    common = set.intersection(*(set(candidate[s]) for s in seeds))
    if reference is not None:
        common &= set.intersection(*(set(reference[s]) for s in seeds))
    return np.asarray([np.mean([candidate[s][i][1] - (reference[s][i][1] if reference is not None else 0.0) for s in seeds])
                       for i in sorted(common)], dtype=np.float64)


def analyse_table(table: np.ndarray, *, resamples: int = 2000, seed: int = 0, bootstrap: bool = True) -> dict[str, Any]:
    """Crossed concept × seed model (and the two-way cluster bootstrap) of one paired table."""
    if table.shape[0] < 2:
        return {"available": False, "concepts": int(table.shape[0])}
    out = {"available": True, "model": crossed_components(table)}
    if bootstrap:
        out["bootstrap"] = two_way_cluster_bootstrap(table, resamples=resamples, seed=seed)
    return out


def plan(components: dict[str, Any], *, deltas: Sequence[float], planned_seeds: int, alpha: float, power: float, tests: int,
         targets: Sequence[int] = ()) -> dict[str, Any]:
    """Concepts needed per Δ for the planned seeds, the power at the current count and at each target, and the minimum
    detectable effect (80% power) at those counts."""
    from scipy import stats
    if components.get("seeds", 1) < 2 or components.get("var_cluster") is None:
        return {"available": False, "detail": "one seed: the seed component is not identifiable"}
    var = {k: float(components[k]) for k in ("var_cluster", "var_seed", "var_residual")}
    z = float(stats.norm.ppf(1 - alpha / (2 * tests)) + stats.norm.ppf(power))
    counts = sorted({int(components["clusters"]), *map(int, targets)})
    out: dict[str, Any] = {"available": True, "planned_seeds": planned_seeds, "tests": tests, "alpha": alpha, "power": power,
                           "required": {}, "power_at": {}, "mde": {}}
    for delta in deltas:
        n = required_clusters(delta, seeds=planned_seeds, alpha=alpha, power=power, tests=tests, **var)
        out["required"][f"{delta:g}"] = None if n is None else int(math.ceil(n))
        out["power_at"][f"{delta:g}"] = {str(c): power_at(c, delta, seeds=planned_seeds, alpha=alpha, tests=tests, **var) for c in counts}
    for c in counts:
        out["mde"][str(c)] = z * math.sqrt(var["var_cluster"] / c + var["var_seed"] / planned_seeds
                                           + var["var_residual"] / (c * planned_seeds))
    out["seed_floor"] = z * math.sqrt(var["var_seed"] / planned_seeds)     # MDE with infinitely many items
    return out


# ---------------------------------------------------------------- stage analysis


def _model_runs(group: Any, model: str, seeds: Sequence[int]) -> dict[int, Path]:
    return {s: run.path for s, run in group.paired(model, seeds).items()}


def comparison_block(candidate: dict[int, Path], reference: dict[int, Path] | None, *, loader: str, folder: str,
                     tests: Sequence[str], source_a: str = "own", source_b: str = "own", split: str | None = None,
                     resamples: int, seed: int, bootstrap: bool = True) -> dict[str, Any]:
    """Per test: the crossed model on the concepts × seeds table of candidate − reference (reference None: a one-sample
    table of the candidate's values), the seed-averaged item-level comparison of the current R9 analysis, and Holm over
    the tests (model p)."""
    def load(paths: dict[int, Path], source: str) -> dict[int, dict[str, dict[str, tuple[str, float]]]]:
        out = {}
        for s, path in paths.items():
            if loader == "new":
                found = new_word_items(path / folder, source)
            else:
                found = zeroshot_items(path / folder, source, split)
            if found is not None:
                out[s] = found
        return out

    a = load(candidate, source_a)
    b = load(reference if reference is not None else candidate, source_b)
    seeds = sorted(set(a) & set(b))
    result: dict[str, Any] = {"seeds": seeds, "tests": {}}
    if not seeds:
        return {**result, "available": False}
    for test in tests:
        ca = {s: _by_concept(a[s][test]) for s in seeds}
        cb = {s: _by_concept(b[s][test]) for s in seeds}
        concepts, _, table = paired_table(ca, cb)
        if len(concepts) < 2:
            continue
        entry = analyse_table(table, resamples=resamples, seed=seed, bootstrap=bootstrap)
        items = seed_averaged_items({s: a[s][test] for s in seeds}, {s: b[s][test] for s in seeds})
        entry.update(concepts=len(concepts), items=int(items.size), items_per_concept=items.size / len(concepts),
                     item_sd=float(items.std(ddof=1)) if items.size > 1 else None,
                     item_level_se=float(items.std(ddof=1) / math.sqrt(items.size)) if items.size > 1 else None)
        result["tests"][test] = entry
    tested = [e for e in result["tests"].values() if e.get("available")]
    for e, adjusted in zip(tested, holm_adjust([e["model"]["p_value"] for e in tested]) if tested else []):
        e["model"].update(p_holm=adjusted, significant=adjusted < 0.05)
    return {**result, "available": bool(tested)}


def edit_block(candidate: dict[int, Path], *, folder: str, resamples: int, seed: int, bootstrap: bool = True,
               metrics: Sequence[str] = EDIT_METRICS) -> dict[str, Any]:
    """The candidate's edits (one-sample: after − control per edit) per metric × subset × test; Holm over subsets × tests
    within a metric (model p)."""
    loaded = {s: v for s, v in ((s, edit_values(p / folder)) for s, p in candidate.items()) if v is not None}
    seeds = sorted(loaded)
    out: dict[str, Any] = {"seeds": seeds, "metrics": {}}
    if not seeds:
        return {**out, "available": False}
    for metric in metrics:
        block: dict[str, dict[str, Any]] = {}
        for subset in EDIT_SUBSETS:
            for test in EDIT_TESTS:
                concepts, _, table = paired_table({s: loaded[s][subset][test][metric] for s in seeds})
                if len(concepts) < 2:
                    continue
                block.setdefault(subset, {})[test] = {**analyse_table(table, resamples=resamples, seed=seed, bootstrap=bootstrap),
                                                      "concepts": len(concepts)}
        tested = [e for by_test in block.values() for e in by_test.values() if e.get("available")]
        for e, adjusted in zip(tested, holm_adjust([e["model"]["p_value"] for e in tested]) if tested else []):
            e["model"].update(p_holm=adjusted, significant=adjusted < 0.05)
        out["metrics"][metric] = block
    return {**out, "available": True}


def analyse_group(group: Any, *, candidate: str = "C5", folder: str = "edit", zeroshot_folder: str | None = "zeroshot",
                  references: Sequence[str] = ("C0'", "C2", "P0"), within: Sequence[str] = NEW_WITHIN, resamples: int = 2000,
                  seed: int = 0, bootstrap: bool = True) -> dict[str, Any]:
    """Item × seed analysis of one host group (`e9_report.Group`): new words (candidate own − each reference's own rows,
    and own − the candidate's control sources), edits (candidate, after − control) and the WP-C7 zero-shot items."""
    if candidate not in group.models:
        return {"available": False}
    seeds = group.seeds(candidate)
    mine = _model_runs(group, candidate, seeds)
    out: dict[str, Any] = {"available": True, "seeds": seeds, "new_words": {}, "zeroshot": {}}
    for reference in references:
        if reference == candidate or reference not in group.models:
            continue
        theirs = _model_runs(group, reference, seeds)
        common = sorted(set(mine) & set(theirs))
        block = comparison_block({s: mine[s] for s in common}, {s: theirs[s] for s in common}, loader="new", folder=folder,
                                 tests=NEW_PRIMARY, resamples=resamples, seed=seed, bootstrap=bootstrap)
        if block.get("available"):
            out["new_words"][f"own − {reference} own"] = block
        if zeroshot_folder:
            block = comparison_block({s: mine[s] for s in common}, {s: theirs[s] for s in common}, loader="zeroshot",
                                     folder=zeroshot_folder, tests=ZS_TESTS, split="synthetic", resamples=resamples, seed=seed,
                                     bootstrap=bootstrap)
            if block.get("available"):
                out["zeroshot"][f"own − {reference} own"] = block
    for source in within:
        block = comparison_block(mine, None, loader="new", folder=folder, tests=NEW_PRIMARY, source_b=source,
                                 resamples=resamples, seed=seed, bootstrap=bootstrap)
        if block.get("available"):
            out["new_words"][f"own − {source}"] = block
    edits = edit_block(mine, folder=folder, resamples=resamples, seed=seed, bootstrap=bootstrap)
    if edits.get("available"):
        out["edits"] = edits
    return out


def power_tables(analysis: dict[str, Any], *, deltas: Sequence[float], planned_seeds: int, alpha: float, power: float,
                 targets: Sequence[int]) -> dict[str, Any]:
    """`plan` for every comparison of `analyse_group` (new-word tests m = 4, zero-shot m = 3, edits m = 4 per metric)."""
    out: dict[str, Any] = {"new_words": {}, "zeroshot": {}, "edits": {}}
    for part, tests in (("new_words", len(NEW_PRIMARY)), ("zeroshot", len(ZS_TESTS))):
        for name, block in analysis.get(part, {}).items():
            out[part][name] = {test: plan(e["model"], deltas=deltas, planned_seeds=planned_seeds, alpha=alpha, power=power,
                                          tests=tests, targets=targets)
                               for test, e in block["tests"].items() if e.get("available")}
    for metric, block in (analysis.get("edits") or {}).get("metrics", {}).items():
        tests = sum(len(by_test) for by_test in block.values())
        out["edits"][metric] = {subset: {test: plan(e["model"], deltas=deltas, planned_seeds=planned_seeds, alpha=alpha,
                                                    power=power, tests=max(tests, 1), targets=[max(1, t // 2) for t in targets])
                                         for test, e in by_test.items() if e.get("available")}
                                for subset, by_test in block.items()}
    return out


# ---------------------------------------------------------------- rendering


def _f(value: Any, digits: int = 3, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _model_cell(entry: dict[str, Any], digits: int = 3) -> str:
    m = entry["model"]
    text = f"{_f(m['mean'], digits)} [{_f(m['ci_low'], digits)}, {_f(m['ci_high'], digits)}]"
    return text + ("*" if m.get("significant") else "")


def _boot_cell(entry: dict[str, Any], digits: int = 3) -> str:
    b = entry.get("bootstrap")
    return "—" if not b else f"[{_f(b['ci_low'], digits)}, {_f(b['ci_high'], digits)}]"


def render_item_seed(analysis: dict[str, Any], *, candidate: str = "C5", heading: str = "####") -> list[str]:
    """Markdown tables of `analyse_group`: per comparison and test the crossed-model estimate [95% t CI, Satterthwaite
    df] (`*` = Holm over the family), the two-way cluster bootstrap interval, the variance components as SDs and the
    current item-level interval for comparison."""
    lines: list[str] = []
    if not analysis.get("available"):
        return [f"No {candidate} evaluations.", ""]
    lines += [f"Seeds: {analysis['seeds']}. Unit: concept (a new word, an edit, a WP-C7 term) × seed; `model` = crossed "
              "random-effects estimate [95% t CI] with Satterthwaite df (`*` Holm over the family's tests); `two-way boot` = "
              "concepts and seeds resampled independently; `sd_c / sd_s / sd_e` = concept, seed and residual SDs of the "
              "paired difference; `item CI` = the current analysis (seed-averaged items, ±1.96 SE, no seed term).", ""]
    for part, title in (("new_words", "New words"), ("zeroshot", "WP-C7 track zero-shot items (synthetic terms)")):
        blocks = analysis.get(part) or {}
        if not blocks:
            continue
        lines += [f"{heading} {title}", "", "| Comparison | Test | concepts | model [95% CI] | df | two-way boot | sd_c / sd_s / sd_e | item CI |",
                  "|---|---|---:|---|---:|---|---|---|"]
        for name, block in blocks.items():
            for test, e in block["tests"].items():
                if not e.get("available"):
                    continue
                m = e["model"]
                sds = "/".join(_f(math.sqrt(m[k]), 3, False) if m.get(k) is not None else "n/a"
                               for k in ("var_cluster", "var_seed", "var_residual"))
                item_ci = (f"[{_f(m['mean'] - 1.96 * e['item_level_se'])}, {_f(m['mean'] + 1.96 * e['item_level_se'])}]"
                           if e.get("item_level_se") is not None else "n/a")
                lines.append(f"| {candidate} {name} | {test} | {e['concepts']} | {_model_cell(e)} | {_f(m['df'], 1, False)} | "
                             f"{_boot_cell(e)} | {sds} | {item_ci} |")
        lines.append("")
    edits = analysis.get("edits")
    if edits:
        lines += [f"{heading} Edits ({candidate}, after − control per edit)", "",
                  "| Metric | Subset | Test | edits | model [95% CI] | df | two-way boot | sd_c / sd_s / sd_e |", "|---|---|---|---:|---|---:|---|---|"]
        for metric, block in edits["metrics"].items():
            for subset, by_test in block.items():
                for test, e in by_test.items():
                    if not e.get("available"):
                        continue
                    m = e["model"]
                    sds = "/".join(_f(math.sqrt(m[k]), 3, False) if m.get(k) is not None else "n/a"
                                   for k in ("var_cluster", "var_seed", "var_residual"))
                    lines.append(f"| {metric} | {subset} | {test} | {e['concepts']} | {_model_cell(e)} | {_f(m['df'], 1, False)} | "
                                 f"{_boot_cell(e)} | {sds} |")
        lines += ["", "`log_odds` = d_after − d_control (nats; EM − EM_control per edit); `success` = share of prompts preferring "
                  "the new filler after the edit − after the control edit.", ""]
    return lines


def render_power(results: dict[str, Any], config: dict[str, Any]) -> str:
    deltas = [f"{d:g}" for d in config["deltas"]]
    lines = ["# Claim C power analysis (E9 dimension 3, decision 56)", "",
             f"Inputs: per-item outputs in `{config['folder']}` (and `{config['zeroshot_folder']}`) of the runs under "
             + ", ".join(f"`{r}`" for r in config["runs"]) + f". Planned seeds S = {config['planned_seeds']}; two-sided α = "
             f"{config['alpha']} with Holm over the family's m tests (Bonferroni bound α/m), power {config['power']}. "
             "Unit = concept (new word / edit / WP-C7 term); variance components from the crossed concept × seed model "
             "(module docstring). `n needed` = concepts per arm; `∞` = the seed component alone exceeds the budget. "
             "MDE = minimum detectable effect at 80% power.", ""]
    for label, entry in results["groups"].items():
        analysis, power = entry["analysis"], entry["power"]
        lines += [f"## {label}", "", f"Seeds: {analysis.get('seeds')}.", ""]
        for part, title, unit in (("new_words", "New words (m = 4)", "new words"), ("zeroshot", "WP-C7 zero-shot, synthetic terms (m = 3)", "terms")):
            blocks = analysis.get(part) or {}
            if not blocks:
                continue
            lines += [f"### {title}", "",
                      "| Comparison | Test | n now | Δ now [95% CI] | sd item | sd concept (seed-avg) | sd seed | items/concept | "
                      + " | ".join(f"n needed Δ={d}" for d in deltas) + " | " + " | ".join(f"power@{t} Δ={d}" for t in config["targets"] for d in deltas)
                      + " | MDE now | " + " | ".join(f"MDE@{t}" for t in config["targets"]) + " |",
                      "|---|---|---:|---|---:|---:|---:|---:|" + "---:|" * (len(deltas) * (1 + len(config["targets"])) + 1 + len(config["targets"]))]
            for name, block in blocks.items():
                for test, e in block["tests"].items():
                    p = power[part].get(name, {}).get(test, {})
                    if not e.get("available") or not p.get("available"):
                        continue
                    m = e["model"]
                    now = str(e["concepts"])
                    lines.append(
                        f"| {name} | {test} | {e['concepts']} | {_model_cell(e)} | {_f(e.get('item_sd'), 3, False)} | "
                        f"{_f(m['sd_cluster_mean'], 3, False)} | {_f(math.sqrt(m['var_seed']), 3, False)} | {_f(e['items_per_concept'], 2, False)} | "
                        + " | ".join("∞" if p["required"][d] is None else str(p["required"][d]) for d in deltas) + " | "
                        + " | ".join(_f(p["power_at"][d].get(str(t)), 2, False) for t in config["targets"] for d in deltas)
                        + f" | {_f(p['mde'].get(now), 3, False)} | " + " | ".join(_f(p["mde"].get(str(t)), 3, False) for t in config["targets"]) + " |")
            lines.append("")
        edits = analysis.get("edits")
        if edits:
            half = [max(1, t // 2) for t in config["targets"]]
            lines += ["### Edits (C5 after − control; m = 4 per metric: efficacy, paraphrase × seen, held-out)", "",
                      "| Metric | Subset | Test | n now | Δ now [95% CI] | sd concept (seed-avg) | sd seed | "
                      + " | ".join(f"n needed Δ={d}" for d in deltas) + " | MDE now | " + " | ".join(f"MDE@{h}" for h in half) + " |",
                      "|---|---|---|---:|---|---:|---:|" + "---:|" * (len(deltas) + 1 + len(half))]
            for metric, block in edits["metrics"].items():
                for subset, by_test in block.items():
                    for test, e in by_test.items():
                        p = power["edits"].get(metric, {}).get(subset, {}).get(test, {})
                        if not e.get("available") or not p.get("available"):
                            continue
                        m = e["model"]
                        lines.append(f"| {metric} | {subset} | {test} | {e['concepts']} | {_model_cell(e)} | "
                                     f"{_f(m['sd_cluster_mean'], 3, False)} | {_f(math.sqrt(m['var_seed']), 3, False)} | "
                                     + " | ".join("∞" if p["required"][d] is None else str(p["required"][d]) for d in deltas)
                                     + f" | {_f(p['mde'].get(str(e['concepts'])), 3, False)} | "
                                     + " | ".join(_f(p["mde"].get(str(h)), 3, False) for h in half) + " |")
            lines += ["", f"Edit targets are per subset (half of {config['targets']} edits each); `log_odds` in nats, `success` "
                      "as a share.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def run(args: argparse.Namespace) -> dict[str, Any]:
    from ..provenance import prepare_output_dir, write_run_metadata
    from .e4_report import discover
    from .e9_report import build_groups
    config = {"experiment": "e9-claim-c-power", "runs": [str(r) for r in args.runs], "folder": args.folder,
              "zeroshot_folder": args.zeroshot_folder, "deltas": list(args.deltas), "planned_seeds": args.planned_seeds,
              "alpha": args.alpha, "power": args.power, "targets": list(args.targets), "resamples": args.resamples,
              "seed": args.seed, "candidate": args.candidate}
    if args.overwrite and args.output.exists():
        for name in ("power.json", "report.md", "resolved_config.yaml", "manifest.json"):
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = prepare_output_dir(args.output)
    groups = build_groups(discover(args.runs))
    results: dict[str, Any] = {"config": config, "groups": {}}
    for group in groups:
        analysis = analyse_group(group, candidate=args.candidate, folder=args.folder, zeroshot_folder=args.zeroshot_folder,
                                 resamples=args.resamples, seed=args.seed)
        if not analysis.get("available"):
            continue
        results["groups"][group.label] = {
            "analysis": analysis,
            "power": power_tables(analysis, deltas=args.deltas, planned_seeds=args.planned_seeds, alpha=args.alpha,
                                  power=args.power, targets=args.targets)}
    (args.output / "power.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (args.output / "report.md").write_text(render_power(results, config))
    write_run_metadata(args.output, config, git_at_start=git_at_start, groups=list(results["groups"]))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folder", default="edit", help="per-run ontology-editing folder (edit, edit-int4, edit-v2, …)")
    parser.add_argument("--zeroshot-folder", default="zeroshot", help="per-run WP-C7 zero-shot folder ('' to skip)")
    parser.add_argument("--candidate", default="C5")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.06])
    parser.add_argument("--planned-seeds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05); parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--targets", type=int, nargs="+", default=[300, 700], help="concept counts to report power / MDE at")
    parser.add_argument("--resamples", type=int, default=2000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.zeroshot_folder = args.zeroshot_folder or None
    results = run(args)
    print(json.dumps({label: {"seeds": g["analysis"]["seeds"]} for label, g in results["groups"].items()}))


if __name__ == "__main__":
    main()
