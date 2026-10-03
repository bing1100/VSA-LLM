"""R9 report generator: retrofit × quantization (execution.md, E9; `reports/R9-retrofit-quantization.md`).

    python -m vsa_embed.experiments.e9_report --runs experiments/e9-retrofit/runs/<stage>
        [--quant experiments/e9-retrofit/quant/<stage>] --output <dir> [--candidate C5] [--resamples 10000] [--overwrite]

Runs are the E9 run folders (`e9_plan`; condition from `<host>-<mode>-<model>-s<seed>`, C0p read as C0′).
They are grouped per host and training mode; the host's P0 (the original model, evaluation only, no
training randomness) joins every group of its host and pairs with each seed of the other models.

**Dimension 1 — does the channel help on long-token, rare and out-of-distribution words?** Per stratum
of the fixed evaluation set (long-token words: `inside`, `after_len2`, `after_len3plus`; rare: `after_rare_seen`;
out of distribution: `after_unseen`, `after_heldout`; locality: `unlinked`, `all`), the final-evaluation loss
differences C5 − C0′, C5 − C2, C5 − P0 (token-weighted, pooled over common seeds; 95% cluster bootstrap over
evaluation windows, `e4_report.paired_difference`; Holm over the three within each stratum), with C2 − C0′ and
C0′ − P0 as context; probe differences per link-status subset (`channel_probes.compare_outputs`, per seed).

**Dimension 2 — is the gap larger under weight quantization?** From `e4_quant` (bf16 = `ref`, INT8, INT4;
variant A channel FP16, B channel quantized): the quantization damage per stratum (`variant − bf16`) for P0,
C0′, C2, C5, and the gap-under-quantization test, the difference in differences
`Δgain = (C5 − ref)@q − (C5 − ref)@bf16` for ref ∈ {C0′, C2, P0}, paired by window (cluster bootstrap,
`e4_quant.retention`; Holm over references × variants within each stratum). A negative `Δgain` means the
channel's advantage *grows* under quantization (the question's "yes"). Probe damage (INT4 − bf16 per model)
and the probe differences at INT4 complete it.

**Dimension 3 — zero-shot learning by editing the ontology** (`e9_ontology_edit`, `e5_zeroshot`): new words
(own row vs name only, mean row, random frame; and the candidate's own row vs the other models' own rows,
paired over items, seed-averaged per item) and edits (ROME/MEMIT-style efficacy, generalization,
specificity) at bf16 and INT4, plus the E5.4 contamination-free synthetic items.

Every result notes its seeds; a single seed is flagged (CIs then cover evaluation windows or items only).
Output (run-folder contract): `report.md`, `summary.json`, `figures/`, `resolved_config.yaml`, `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..evaluation import channel_probes as cp
from ..provenance import prepare_output_dir, write_run_metadata
from ..statistics import holm_adjust, paired_ratio_bootstrap
from . import e9_ontology_edit as edit
from .e4_quant import retention, run_id
from .e4_report import Run, _json_default, condition_order, discover, paired_difference
from .e5_zeroshot import paired_difference as item_difference

STRATUM_GROUPS: dict[str, tuple[str, ...]] = {
    "long-token words": ("inside", "after_len2", "after_len3plus"),
    "rare words": ("after_rare_seen",),
    "out of distribution": ("after_unseen", "after_heldout"),
    "locality": ("unlinked", "all"),
}
STRATA = tuple(s for group in STRATUM_GROUPS.values() for s in group)
REFERENCES = ("C0'", "C2", "P0")
CONTEXT = (("C2", "C0'"), ("C0'", "P0"))
MODELS = ("P0", "C0'", "C2", "C5")
PROBE_KEYS = (("lambada", "accuracy"), ("wic", "probe_accuracy"), ("card660", "spearman_tied"), ("rare_words", "spearman_tied"),
              ("wsd", "probe_f1"), ("bless", "pair_probe_macro_f1"), ("bless", "prompt_auc"), ("hyperlex", "cosine_spearman"))
PROBE_SUBSETS = ("all", "heldout", "seen", "unlinked")
ZS_TESTS = ("property", "entailment", "paraphrase")
ALPHA = 0.05
# Model colours (fixed per model, validated with the dataviz palette checks: blue / orange / aqua, all-pairs CVD
# ΔE ≥ 9.2); P0 is the neutral reference ink. Aqua is below 3:1 on the surface, so every figure has a table.
COLOURS = {"P0": "#8a8984", "C0'": "#2a78d6", "C2": "#eb6834", "C5": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


@dataclass
class Group:
    """One host × training mode: the models' runs by seed (P0 shared across the host's groups)."""

    host: str
    mode: str
    models: dict[str, dict[int, Run]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.host.split('/')[-1]} · {self.mode}"

    def seeds(self, model: str) -> list[int]:
        return sorted(self.models.get(model, {}))

    def paired(self, model: str, seeds: Sequence[int]) -> dict[int, Run]:
        """Runs of `model` for `seeds`; P0 (one evaluation-only run) stands for every seed."""
        runs = self.models.get(model, {})
        if model == "P0" and runs:
            only = runs[min(runs)]
            return {s: only for s in seeds}
        return {s: runs[s] for s in seeds if s in runs}


def model_name(condition: str) -> str:
    return "C0'" if condition in {"C0p", "C0'"} else condition


def build_groups(runs: Sequence[Run]) -> list[Group]:
    complete = [r for r in runs if r.complete and r.config.get("model", {}).get("pretrained")]
    groups: dict[tuple[str, str], Group] = {}
    p0: dict[str, dict[int, Run]] = {}
    for run in sorted(complete, key=lambda r: str(r.path)):
        host, mode, name = run.config["model"]["pretrained"], run.config["model"].get("host_mode", "train"), model_name(run.condition)
        if name == "P0":
            p0.setdefault(host, {}).setdefault(run.seed, run)
            continue
        group = groups.setdefault((host, mode), Group(host, mode))
        group.models.setdefault(name, {}).setdefault(run.seed, run)
    for (host, _), group in groups.items():
        if host in p0:
            group.models["P0"] = p0[host]
        group.models = {m: group.models[m] for m in sorted(group.models, key=lambda m: (m != "P0", condition_order(m)))}
    return [groups[k] for k in sorted(groups)]


def _flags(seeds: Sequence[int]) -> list[str]:
    return ["single seed: CIs cover evaluation windows or items only, not seed variance"] if len(seeds) == 1 else []


# ---------------------------------------------------------------- dimension 1


def dimension1(group: Group, *, candidate: str, resamples: int, seed: int) -> dict[str, Any]:
    if candidate not in group.models:
        return {"available": False, "detail": f"no {candidate} runs"}
    seeds = group.seeds(candidate)
    result: dict[str, Any] = {"available": True, "seeds": seeds, "flags": _flags(seeds), "strata": {}, "context": {}}
    present = {s for runs in group.models.values() for r in runs.values() for s in r.final}
    for stratum in [s for s in STRATA if s in present]:
        block = {}
        for reference in REFERENCES:
            if reference in group.models and reference != candidate:
                comparison = paired_difference(group.paired(candidate, seeds), group.paired(reference, seeds), stratum,
                                               resamples=resamples, seed=seed)
                if comparison is not None:
                    block[reference] = comparison
        tested = [c for c in block.values() if c.get("p_value") is not None]
        for comparison, adjusted in zip(tested, holm_adjust([c["p_value"] for c in tested])):
            comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
        result["strata"][stratum] = block
        context = {}
        for a, b in CONTEXT:
            if a in group.models and b in group.models:
                common = group.seeds(a) if b == "P0" else sorted(set(group.seeds(a)) & set(group.seeds(b)))
                comparison = paired_difference(group.paired(a, common), group.paired(b, common), stratum, resamples=resamples, seed=seed)
                if comparison is not None:
                    context[f"{a} − {b}"] = comparison
        result["context"][stratum] = context
    result["probes"] = probe_comparisons(group, candidate, "probes.json", resamples=min(resamples, 2000), seed=seed)
    return result


def probe_comparisons(group: Group, candidate: str, filename: str, *, resamples: int, seed: int,
                      references: Sequence[str] = REFERENCES) -> dict[str, Any]:
    """Per seed and reference: candidate − reference on the key probe metrics and link-status subsets."""
    out: dict[str, Any] = {}
    for reference in references:
        if reference not in group.models or reference == candidate:
            continue
        for s in group.seeds(candidate):
            a = group.models[candidate][s].path / filename
            ref_run = group.paired(reference, [s]).get(s)
            if ref_run is None or not a.exists() or not (ref_run.path / filename).exists():
                continue
            try:
                comparison = cp.compare_outputs(a, ref_run.path / filename, resamples=resamples, seed=seed)
            except ValueError as error:
                out.setdefault(reference, {})[s] = {"error": str(error)}
                continue
            out.setdefault(reference, {})[s] = _probe_rows(comparison)
    return out


def _probe_rows(comparison: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for table, metric in PROBE_KEYS:
        per_subset = comparison["tables"].get(table, {}).get(metric) or {}
        rows[f"{table}/{metric}"] = {subset: per_subset[subset] for subset in PROBE_SUBSETS if subset in per_subset}
    return rows


def probe_damage(group: Group, *, resamples: int, seed: int, quantized: str = "int4") -> dict[str, Any]:
    """Per model and seed: quantized (`probes-<quantized>.json`) − bf16 on the key probe metrics (paired over items)."""
    out: dict[str, Any] = {}
    for model, runs in group.models.items():
        for s, run in runs.items():
            a, b = run.path / f"probes-{quantized}.json", run.path / "probes.json"
            if a.exists() and b.exists():
                try:
                    out.setdefault(model, {})[s] = _probe_rows(cp.compare_outputs(a, b, resamples=resamples, seed=seed))
                except ValueError as error:
                    out.setdefault(model, {})[s] = {"error": str(error)}
    return out


# ---------------------------------------------------------------- dimension 2


def load_quant(quant_dir: Path | None) -> dict[str, dict[str, Any]]:
    """`e4_quant` output → {resolved run path: {"record", "windows"}}."""
    if quant_dir is None or not (Path(quant_dir) / "runs").is_dir():
        return {}
    out = {}
    for path in sorted((Path(quant_dir) / "runs").glob("*/quant.json")):
        record = json.loads(path.read_text())
        with np.load(path.parent / "windows.npz") as data:
            windows = {"strata": data["strata"].tolist(), "count": data["count"],
                       **{key[4:]: data[key] for key in data.files if key.startswith("sum_")}}
        out[str(Path(record["run"]).resolve())] = {"record": record, "windows": windows}
    return out


def _variant(windows: dict[str, Any], name: str) -> np.ndarray:
    if name in windows:
        return windows[name]
    if name.endswith("-B") and name[:-1] + "A" in windows:     # a model without a channel: B = A
        return windows[name[:-1] + "A"]
    raise KeyError(name)


def dimension2(group: Group, quant: dict[str, dict[str, Any]], *, candidate: str, resamples: int, seed: int) -> dict[str, Any]:
    entries = {m: {s: quant.get(str(r.path.resolve())) for s, r in runs.items()} for m, runs in group.models.items()}
    entries = {m: {s: e for s, e in by_seed.items() if e is not None} for m, by_seed in entries.items()}
    entries = {m: e for m, e in entries.items() if e}
    if not entries:
        return {"available": False, "detail": "no e4_quant output for these runs"}
    first = next(iter(next(iter(entries.values())).values()))["windows"]
    strata, counts = first["strata"], first["count"]
    for by_seed in entries.values():
        for e in by_seed.values():
            if e["windows"]["strata"] != strata or not np.array_equal(e["windows"]["count"], counts):
                return {"available": False, "detail": "the quantized evaluations do not share windows and targets"}
    variants = sorted({k for by_seed in entries.values() for e in by_seed.values() for k in e["windows"]
                       if k not in {"strata", "count", "ref"}}, key=lambda v: (-int(v[3:].split("-")[0]), v))
    keep = [s for s in STRATA if s in strata]

    def pooled(model: str, seeds: Sequence[int], name: str, i: int) -> np.ndarray:
        by_seed = entries[model]
        if model == "P0":
            only = by_seed[min(by_seed)]["windows"]
            return len(seeds) * _variant(only, name)[i]
        return sum(_variant(by_seed[s]["windows"], name)[i] for s in seeds)

    result: dict[str, Any] = {"available": True, "variants": variants, "strata": keep, "damage": {}, "gap": {},
                              "seeds": {m: sorted(e) for m, e in entries.items()}}
    for model, by_seed in entries.items():
        seeds = sorted(by_seed)
        for variant in variants:
            for stratum in keep:
                i = strata.index(stratum)
                n = len(seeds) * counts[i].astype(np.float64)
                if n.sum() == 0:
                    continue
                ref = pooled(model, seeds, "ref", i)
                change = paired_ratio_bootstrap(pooled(model, seeds, variant, i) - ref, n, ref, resamples=resamples, seed=seed)
                result["damage"].setdefault(model, {}).setdefault(variant, {})[stratum] = {
                    k: change.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative", "relative_ci_low", "relative_ci_high")}
    if candidate in entries:
        seeds = sorted(entries[candidate])
        for reference in REFERENCES:
            if reference not in entries or reference == candidate:
                continue
            common = seeds if reference == "P0" else sorted(set(seeds) & set(entries[reference]))
            if not common:
                continue
            for variant in variants:
                for stratum in keep:
                    i = strata.index(stratum)
                    n = len(common) * counts[i].astype(np.float64)
                    if n.sum() == 0:
                        continue
                    outcome = retention(pooled(candidate, common, "ref", i), pooled(reference, common, "ref", i),
                                        pooled(candidate, common, variant, i), pooled(reference, common, variant, i), n,
                                        resamples=resamples, seed=seed)
                    outcome["seeds"] = common
                    result["gap"].setdefault(reference, {}).setdefault(variant, {})[stratum] = outcome
        for stratum in keep:          # Holm over references × variants within each stratum
            family = [by_v[stratum]["gain_change"] for by_variant in result["gap"].values() for by_v in by_variant.values()
                      if stratum in by_v and by_v[stratum]["gain_change"].get("p_value") is not None]
            for test, adjusted in zip(family, holm_adjust([t["p_value"] for t in family])):
                test.update(holm_p=adjusted, significant=adjusted < ALPHA)
    return result


# ---------------------------------------------------------------- dimension 3


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _edit_folder(run: Run, variant: str) -> Path:
    return run.path / ("edit" if variant == "bf16" else f"edit-{variant}")


def _zeroshot_folder(run: Run, variant: str) -> Path:
    return run.path / ("zeroshot" if variant == "bf16" else f"zeroshot-{variant}")


def new_word_items(folder: Path, source: str = "own") -> tuple[dict[str, tuple[list[str], np.ndarray]], set[str]] | None:
    """Per test (ids, values) of one source from an edit evaluation's predictions, over linked concepts."""
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        return None
    document = json.loads(summary_path.read_text())
    if "new_words" not in document:
        return None
    linked = {c for c, r in document["new_words"]["resolved"].items() if r["status"] == "linked"}
    rows = [r for r in _read_jsonl(folder / "predictions.jsonl") if r.get("part") == "new" and r.get("source") == source]
    result = {"prompts": [r for r in rows if r["test"] in {"property", "entailment"}],
              "statements": [r for r in rows if r["test"] == "statement"]}
    return edit.new_word_vectors(result, linked), linked


def zeroshot_items(folder: Path, source: str = "own") -> dict[str, tuple[list[str], np.ndarray]] | None:
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        return None
    document = json.loads(summary_path.read_text())
    linked = {c for c, r in document["resolved"].items() if r["linked"] and r["status"] != "mislinked"}
    rows = [r for r in _read_jsonl(folder / "predictions.jsonl") if r.get("source") == source and r.get("concept") in linked]
    pick = lambda chosen, key: ([r["id"] for r in chosen], np.asarray([r[key] for r in chosen], dtype=float))
    prop = [r for r in rows if r["test"] == "property"]
    return {"property": pick(prop, "correct"), "entailment": pick([r for r in rows if r["test"] == "entailment"], "correct"),
            "paraphrase": pick([r for r in prop if r.get("consistent") is not None], "consistent")}


def _seed_average(vectors: list[dict[str, tuple[list[str], np.ndarray]]]) -> dict[str, dict[str, float]]:
    """Per test: item id → value averaged over the seeds that have it."""
    out: dict[str, dict[str, list[float]]] = {}
    for by_test in vectors:
        for test, (ids, values) in by_test.items():
            for i, v in zip(ids, values.tolist()):
                out.setdefault(test, {}).setdefault(i, []).append(v)
    return {t: {i: float(np.mean(v)) for i, v in by_id.items()} for t, by_id in out.items()}


def cross_model(group: Group, candidate: str, variant: str, loader, tests: Sequence[str], *, resamples: int, seed: int,
                lower_is_better: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Candidate's own rows − each reference model's own rows, paired over items (seed-averaged)."""
    def averaged(model: str) -> dict[str, dict[str, float]] | None:
        loaded = [loader(run, variant) for run in group.models.get(model, {}).values()]
        loaded = [x for x in loaded if x is not None]
        return _seed_average(loaded) if loaded else None

    mine = averaged(candidate)
    if mine is None:
        return {}
    out: dict[str, Any] = {}
    for reference in REFERENCES:
        if reference == candidate:
            continue
        theirs = averaged(reference)
        if theirs is None:
            continue
        block = []
        for test in tests:
            ids = sorted(set(mine.get(test, {})) & set(theirs.get(test, {})))
            if not ids:
                continue
            ci = item_difference(np.asarray([mine[test][i] for i in ids]), np.asarray([theirs[test][i] for i in ids]),
                                 resamples=resamples, seed=seed)
            block.append({"test": test, **ci, "better": bool(ci["mean"] < 0 if test in lower_is_better else ci["mean"] > 0)})
        for row, adjusted in zip(block, holm_adjust([r["p_value"] for r in block]) if block else []):
            row.update(p_holm=adjusted, significant=adjusted < ALPHA)
        out[reference] = block
    return out


def dimension3(group: Group, *, candidate: str, resamples: int, seed: int, variants: Sequence[str] = ("bf16", "int4")) -> dict[str, Any]:
    result: dict[str, Any] = {"variants": {}}
    for variant in variants:
        per_model: dict[str, Any] = {}
        for model, runs in group.models.items():
            for s, run in sorted(runs.items()):
                entry: dict[str, Any] = {}
                summary_path = _edit_folder(run, variant) / "summary.json"
                if summary_path.exists():
                    document = json.loads(summary_path.read_text())
                    if "new_words" in document:
                        entry["new_words"] = document["new_words"]["summary"]
                    if "edits" in document:
                        entry["edits"] = document["edits"]["summary"]
                zs_path = _zeroshot_folder(run, variant) / "summary.json"
                if zs_path.exists():
                    entry["zeroshot"] = json.loads(zs_path.read_text())["summary"]
                if entry:
                    per_model.setdefault(model, {})[s] = entry
        new_loader = lambda run, v: (lambda x: x[0] if x else None)(new_word_items(_edit_folder(run, v)))
        zs_loader = lambda run, v: zeroshot_items(_zeroshot_folder(run, v))
        result["variants"][variant] = {
            "models": per_model,
            "new_words_vs_models": cross_model(group, candidate, variant, new_loader, edit.NEW_TESTS, resamples=resamples, seed=seed,
                                               lower_is_better=frozenset(edit.LOWER_IS_BETTER)),
            "zeroshot_vs_models": cross_model(group, candidate, variant, zs_loader, ZS_TESTS, resamples=resamples, seed=seed)}
    return result


# ---------------------------------------------------------------- analysis and rendering


def analyze(runs: Sequence[Run], quant: dict[str, dict[str, Any]], *, candidate: str = "C5", resamples: int = 10_000,
            seed: int = 0, quantized: str = "int4") -> tuple[dict[str, Any], list[Group]]:
    """`quantized` names the per-run quantized evaluations (`probes-<q>.json`, `zeroshot-<q>`, `edit-<q>`)."""
    groups = build_groups(runs)
    summary: dict[str, Any] = {"candidate": candidate, "resamples": resamples, "seed": seed, "quantized": quantized, "groups": {},
                               "runs": [{"path": str(r.path), "condition": model_name(r.condition), "seed": r.seed,
                                         "complete": r.complete, "model": r.model} for r in runs]}
    for group in groups:
        summary["groups"][group.label] = {
            "host": group.host, "mode": group.mode, "seeds": {m: group.seeds(m) for m in group.models},
            "dimension1": dimension1(group, candidate=candidate, resamples=resamples, seed=seed),
            "dimension2": {**dimension2(group, quant, candidate=candidate, resamples=resamples, seed=seed),
                           "probe_damage": probe_damage(group, resamples=min(resamples, 2000), seed=seed, quantized=quantized),
                           "probes_quantized": probe_comparisons(group, candidate, f"probes-{quantized}.json",
                                                                 resamples=min(resamples, 2000), seed=seed)},
            "dimension3": dimension3(group, candidate=candidate, resamples=min(resamples, 10_000), seed=seed,
                                     variants=("bf16", quantized))}
    return summary, groups


def _fmt(value: float | None, digits: int = 4, signed: bool = True) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(entry: dict[str, Any] | None, key: str = "mean", digits: int = 4) -> str:
    if not entry or entry.get(key) is None:
        return "n/a"
    text = _fmt(entry[key], digits)
    if entry.get("ci_low") is not None:
        text += f" [{_fmt(entry['ci_low'], digits)}, {_fmt(entry['ci_high'], digits)}]"
    return text + ("*" if entry.get("significant") else "")


def _relative(entry: dict[str, Any] | None) -> str:
    if not entry or entry.get("relative") is None:
        return "n/a"
    text = f"{entry['relative'] * 100:+.2f}%"
    if entry.get("relative_ci_low") is not None:
        text += f" [{entry['relative_ci_low'] * 100:+.2f}, {entry['relative_ci_high'] * 100:+.2f}]"
    return text + ("*" if entry.get("significant") else "")


def _stratum_label(stratum: str) -> str:
    group = next(g for g, members in STRATUM_GROUPS.items() if stratum in members)
    return f"{stratum} ({group})"


def render(summary: dict[str, Any], figures: dict[str, dict[str, str]], *, title: str) -> str:
    candidate = summary["candidate"]
    lines = [f"# {title}", "",
             "E9 asks three questions on pretrained hosts, all on one fixed test set: (1) does a VSA ontology channel "
             "trained jointly with the host improve it on long-token, rare and out-of-distribution words; (2) is the gap "
             "larger under weight quantization; (3) after training, can the model learn new or changed words zero-shot "
             "by editing the ontology alone? Models: **P0** the original host (evaluation only), **C0′** continued "
             "training without the channel, **C2** the same with a capacity-matched free per-concept table, **C5** the "
             "same with the attentive VSA channel. Loss differences are condition − reference (negative = lower loss), "
             "token-weighted and pooled over common seeds, with 95% cluster bootstraps over evaluation windows "
             f"({summary['resamples']} resamples); `*` = Holm-adjusted p < 0.05 within the family named in each table. "
             "P0 has no training randomness and pairs with every seed.", ""]
    for label, g in summary["groups"].items():
        lines += ["", f"## {label}", "", "Seeds: " + "; ".join(f"{m} {s}" for m, s in g["seeds"].items()) + ".", ""]
        d1, d2, d3 = g["dimension1"], g["dimension2"], g["dimension3"]
        lines += [f"### Dimension 1 — {candidate} vs C0′, C2 and P0 (bf16)", ""]
        if not d1.get("available"):
            lines.append(f"Not available: {d1.get('detail')}.")
        else:
            lines += [f"> {flag}." for flag in d1["flags"]] + ([""] if d1["flags"] else [])
            lines += ["Relative loss difference [95% CI] (Holm over the three references within each stratum):", "",
                      f"| Stratum | Targets | {candidate} − C0′ | {candidate} − C2 | {candidate} − P0 | C2 − C0′ | C0′ − P0 |",
                      "|---|---:|---|---|---|---|---|"]
            for stratum, block in d1["strata"].items():
                tokens = next((c.get("tokens") for c in block.values() if c.get("tokens")), None)
                context = d1["context"].get(stratum, {})
                lines.append(f"| {_stratum_label(stratum)} | {tokens or 'n/a'} | " + " | ".join(
                    _relative(block.get(r)) for r in REFERENCES) + " | " + " | ".join(
                    _relative(context.get(f"{a} − {b}")) for a, b in CONTEXT) + " |")
            lines += ["", "Absolute differences (nats/token):", "", f"| Stratum | {candidate} − C0′ | {candidate} − C2 | {candidate} − P0 |",
                      "|---|---|---|---|"]
            for stratum, block in d1["strata"].items():
                lines.append(f"| {stratum} | " + " | ".join(_ci(block.get(r), "delta") for r in REFERENCES) + " |")
            lines += _render_probes(d1.get("probes", {}), candidate, "Probe differences (bf16)")
        lines += ["", "### Dimension 2 — quantization damage and the gap under quantization", ""]
        if not d2.get("available"):
            lines.append(f"Not available: {d2.get('detail')}.")
        else:
            variants = d2["variants"]
            lines += ["Quantization damage `variant − bf16`, relative [95% CI] (A = channel FP16, B = channel quantized; "
                      "models without a channel have B = A):", "",
                      "| Stratum | Model | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
            for stratum in d2["strata"]:
                for model, by_variant in d2["damage"].items():
                    lines.append(f"| {stratum} | {model} | " + " | ".join(_relative(by_variant.get(v, {}).get(stratum)) for v in variants) + " |")
            for reference, by_variant in d2["gap"].items():
                lines += ["", f"Gap under quantization vs {reference}: gain = {candidate} − {reference} at bf16 and quantized, "
                          "`Δgain` = difference in differences (negative = the channel's advantage grows under quantization; "
                          "Holm over references × variants within each stratum), retained = gain_q / gain_bf16:", "",
                          "| Stratum | Variant | gain bf16 | gain quantized | Δgain [95% CI] | retained |", "|---|---|---|---|---|---|"]
                for stratum in d2["strata"]:
                    for variant in variants:
                        r = by_variant.get(variant, {}).get(stratum)
                        if r is None:
                            continue
                        kept = r["retained"]
                        retained = "n/a" if kept["mean"] is None or not math.isfinite(kept["mean"]) else f"{kept['mean']:.2f}"
                        lines.append(f"| {stratum} | {variant} | {_ci(r['gain_bf16'])} | {_ci(r['gain_quantized'])} | "
                                     f"{_ci(r['gain_change'])} | {retained} |")
            if any(len(s) == 1 for s in d2["seeds"].values()):
                lines += ["", "> Single seed in at least one model: CIs cover evaluation windows only."]
        if d2.get("probe_damage"):
            lines += _render_probe_damage(d2["probe_damage"], summary["quantized"])
        if d2.get("probes_quantized"):
            lines += _render_probes(d2["probes_quantized"], candidate, f"Probe differences at {summary['quantized']} (variant A)")
        lines += ["", "### Dimension 3 — zero-shot learning by ontology editing (no weight update)", ""]
        if len(g["seeds"].get(candidate, [])) == 1:
            lines += ["> Single seed: intervals cover items only, not seed variance.", ""]
        lines += _render_dimension3(d3, candidate)
        if figures.get(label):
            lines += ["", "### Figures", ""]
            for caption, path in figures[label].items():
                lines += [f"![{caption}]({path})", ""]
    return "\n".join(lines) + "\n"


def _render_probes(probes: dict[str, Any], candidate: str, heading: str) -> list[str]:
    if not probes:
        return ["", f"{heading}: no probe outputs."]
    lines = ["", f"{heading}: {candidate} − reference per subset, per seed (paired bootstrap over items):", "",
             "| Reference | Seed | Probe | " + " | ".join(PROBE_SUBSETS) + " |", "|---|---:|---|" + "---|" * len(PROBE_SUBSETS)]
    for reference, by_seed in probes.items():
        for s, rows in sorted(by_seed.items()):
            if "error" in rows:
                lines.append(f"| {reference} | {s} | {rows['error']} |" + " |" * len(PROBE_SUBSETS)); continue
            for key, by_subset in rows.items():
                if not by_subset:              # probe not run for this pair
                    continue
                cells = [(_ci({"mean": v["difference"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]}, digits=3) + f" (n {v['n']})")
                         if (v := by_subset.get(subset)) and v.get("difference") is not None else "—" for subset in PROBE_SUBSETS]
                lines.append(f"| {reference} | {s} | {key} | " + " | ".join(cells) + " |")
    return lines


def _render_probe_damage(damage: dict[str, Any], quantized: str) -> list[str]:
    lines = ["", f"Probe damage {quantized} − bf16 per model (paired over items; all items / held-out / seen):", "",
             "| Model | Seed | Probe | all | heldout | seen |", "|---|---:|---|---|---|---|"]
    for model, by_seed in damage.items():
        for s, rows in sorted(by_seed.items()):
            if "error" in rows:
                lines.append(f"| {model} | {s} | {rows['error']} | | | |"); continue
            for key, by_subset in rows.items():
                if not by_subset:
                    continue
                cells = [_ci({"mean": v["difference"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]}, digits=3)
                         if (v := by_subset.get(subset)) and v.get("difference") is not None else "—" for subset in ("all", "heldout", "seen")]
                lines.append(f"| {model} | {s} | {key} | " + " | ".join(cells) + " |")
    return lines


def _render_dimension3(d3: dict[str, Any], candidate: str) -> list[str]:
    lines: list[str] = []
    tests = ("property", "property_new", "entailment", "paraphrase", "statement_accuracy", "statement_loss")
    for variant, block in d3["variants"].items():
        models = block["models"]
        if not models:
            lines += [f"{variant}: no evaluations.", ""]
            continue
        lines += [f"**{variant}** — new words (mean over linked items; `own` = the model's own rows: frame composed for "
                  f"{candidate}, fallback row for C2, nothing for C0′/P0):", "",
                  "| Model | Seed | Source | " + " | ".join(tests) + " |", "|---|---:|---|" + "---:|" * len(tests)]
        for model, by_seed in models.items():
            for s, entry in sorted(by_seed.items()):
                for source, by_test in (entry.get("new_words") or {}).get("sources", {}).items():
                    lines.append(f"| {model} | {s} | {source} | " + " | ".join(_fmt(by_test.get(t, {}).get("mean"), 3, signed=False) for t in tests) + " |")
        for reference, rows in block["new_words_vs_models"].items():
            if rows:
                lines += ["", f"{candidate} own − {reference} own (new words; paired over items, seed-averaged; Holm over tests):", "",
                          "| Test | Difference [95% CI] | n | p (Holm) |", "|---|---|---:|---:|"]
                lines += [f"| {r['test']} | {_ci(r, digits=3)} | {r['n']} | {_fmt(r.get('p_holm'), 4, signed=False)} |" for r in rows]
        lines += ["", f"**{variant}** — edits (ES/PS/NS after the edit; EM = change of log p(new) − log p(old)):", "",
                  "| Model | Seed | Subset | edits | ES before → after | EM [95% CI] | EM − control [95% CI] | PS | NS | score |",
                  "|---|---:|---|---:|---|---|---|---:|---:|---:|"]
        for model, by_seed in models.items():
            for s, entry in sorted(by_seed.items()):
                summary = entry.get("edits")
                if not summary:
                    continue
                for subset, e in summary["subsets"].items():
                    eff, par, nb = e.get("efficacy", {}), e.get("paraphrase", {}), e.get("neighbourhood", {})
                    if not eff.get("n"):
                        continue
                    shown = model if summary["edit_applicable"] else f"{model} (not editable)"
                    lines.append(f"| {shown} | {s} | {subset} | {e['edits']} | {_fmt(eff['success_before'], 3, False)} → "
                                 f"{_fmt(eff['success_after'], 3, False)} | {_ci(eff['magnitude'], digits=3)} | "
                                 f"{_ci(eff['target_minus_control'], digits=3)} | {_fmt(par.get('success_after'), 3, False)} | "
                                 f"{_fmt(nb.get('success_after'), 3, False)} | {_fmt(e.get('score'), 3, False)} |")
        zs_rows = [(model, s, entry["zeroshot"]) for model, by_seed in models.items() for s, entry in sorted(by_seed.items())
                   if entry.get("zeroshot")]
        if zs_rows:
            lines += ["", f"**{variant}** — E5.4 contamination-free synthetic items (`own` source, linked concepts):", "",
                      "| Model | Seed | property | entailment | paraphrase |", "|---|---:|---:|---:|---:|"]
            for model, s, z in zs_rows:
                own = z["sources"].get("own", {}).get("linked", {})
                lines.append(f"| {model} | {s} | " + " | ".join(_fmt(own.get(t, {}).get("mean"), 3, False) for t in ZS_TESTS) + " |")
            for reference, rows in block["zeroshot_vs_models"].items():
                if rows:
                    lines.append(f"\n{candidate} own − {reference} own (E5.4): " + "; ".join(
                        f"{r['test']} {_ci(r, digits=3)}" for r in rows))
        lines.append("")
    return lines


# ---------------------------------------------------------------- figures


def _axes(ax) -> None:
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _dot_panel(ax, rows: list[tuple[str, str, dict[str, Any] | None]], order: list[str], series: list[str], *, scale: float = 100.0,
               key: str = "relative", xlabel: str) -> None:
    """Dot-and-whisker rows: (row label, series, entry with key / key_ci_low / key_ci_high)."""
    offsets = {s: (i - (len(series) - 1) / 2) * 0.22 for i, s in enumerate(series)}
    for label, name, entry in rows:
        if not entry or entry.get(key) is None:
            continue
        y = order.index(label) + offsets[name]
        low = entry.get(f"{key}_ci_low") if key != "mean" else entry.get("ci_low")
        high = entry.get(f"{key}_ci_high") if key != "mean" else entry.get("ci_high")
        colour = COLOURS.get(name.split(" ")[-1], COLOURS.get(name, MUTED))
        if low is not None and high is not None:
            ax.plot([low * scale, high * scale], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot([entry[key] * scale], [y], "o", markersize=6, color=colour, markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                label=name)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    _axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if len(unique) > 1:              # outside the plot area, so it never covers a mark; one series needs no legend
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def plot_group(label: str, g: dict[str, Any], out: Path, slug: str, *, candidate: str) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures: dict[str, str] = {}
    d1 = g["dimension1"]
    if d1.get("available") and d1["strata"]:
        order = list(d1["strata"])
        rows = [(s, f"vs {r}", block.get(r)) for s, block in d1["strata"].items() for r in REFERENCES]
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.4))
        series = [f"vs {r}" for r in REFERENCES if any(r in b for b in d1["strata"].values())]
        _dot_panel(ax, [r for r in rows if r[1] in series], order, series, xlabel=f"{candidate} − reference, relative loss difference (%)")
        ax.set_title(f"Dimension 1 — {candidate} vs each reference per stratum ({label}; dots = estimate, bars = 95% CI)",
                     fontsize=9, color=INK, loc="left")
        fig.tight_layout(); name = f"figures/d1-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
        figures["Dimension 1: relative loss difference per stratum"] = name
    d2 = g["dimension2"]
    if d2.get("available") and d2["damage"]:
        int4 = next((v for v in d2["variants"] if v == "int4-A"), d2["variants"][0] if d2["variants"] else None)
        if int4:
            order = d2["strata"]
            models = list(d2["damage"])
            rows = [(s, m, d2["damage"][m].get(int4, {}).get(s)) for s in order for m in models]
            fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.4))
            _dot_panel(ax, rows, order, models, xlabel=f"{int4} − bf16, relative loss change (%)")
            ax.set_title(f"Dimension 2 — quantization damage per stratum ({label})", fontsize=9, color=INK, loc="left")
            fig.tight_layout(); name = f"figures/d2-damage-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
            figures["Dimension 2: quantization damage per stratum"] = name
        gap = d2["gap"].get("C0'")
        if gap:
            variants = [v for v in d2["variants"] if v in gap]
            fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 0.45 * len(d2["strata"]) + 1.6), squeeze=False)
            for ax, variant in zip(axes[0], variants):
                rows = [(s, candidate, (gap[variant].get(s) or {}).get("gain_change")) for s in d2["strata"]]
                _dot_panel(ax, rows, d2["strata"], [candidate], scale=1.0, key="mean", xlabel="Δgain (nats/token)")
                ax.set_title(variant, fontsize=9, color=INK, loc="left")
                if ax is not axes[0][0]:
                    ax.set_yticklabels([])
            fig.suptitle(f"Dimension 2 — ({candidate} − C0′)@quantized − ({candidate} − C0′)@bf16; negative = gap grows ({label})",
                         fontsize=9, x=0.01, ha="left", color=INK)
            fig.tight_layout(rect=(0, 0, 1, 0.93)); name = f"figures/d2-gap-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
            figures["Dimension 2: gap under quantization (difference in differences)"] = name
    d3 = g["dimension3"]["variants"]
    panels = [v for v in d3 if d3[v]["models"]]
    if panels:
        tests = ["property", "property_new", "entailment", "statement_accuracy"]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 2.8), squeeze=False, sharex=True)
        for ax, variant in zip(axes[0], panels):
            rows = []
            for model, by_seed in d3[variant]["models"].items():
                values = {t: [e.get("new_words", {}).get("sources", {}).get("own", {}).get(t, {}).get("mean") for e in by_seed.values()]
                          for t in tests}
                for t in tests:
                    v = [x for x in values[t] if x is not None]
                    if v:
                        rows.append((t, model, {"mean": float(np.mean(v)), "ci_low": min(v), "ci_high": max(v)}))
            _dot_panel(ax, rows, tests, [m for m in MODELS if m in d3[variant]["models"]], scale=1.0, key="mean",
                       xlabel="accuracy (own rows; bar = seed range)")
            ax.set_title(f"new words — {variant}", fontsize=9, color=INK, loc="left")
            if ax is not axes[0][0]:
                ax.set_yticklabels([])
            if ax is not axes[0][-1] and ax.get_legend() is not None:
                ax.get_legend().remove()          # one legend, on the last panel
        fig.suptitle(f"Dimension 3 — zero-shot new words by ontology editing ({label})", fontsize=9, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9)); name = f"figures/d3-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
        figures["Dimension 3: new-word accuracy per model"] = name
    return figures


# ---------------------------------------------------------------- CLI


def write_report(runs_dirs: Sequence[Path], output: Path, *, quant_dir: Path | None = None, candidate: str = "C5",
                 resamples: int = 10_000, seed: int = 0, title: str = "R9 — retrofit × quantization (E9)", figures: bool = True,
                 overwrite: bool = False, quantized: str = "int4") -> dict[str, Any]:
    config = {"runs": [str(p) for p in runs_dirs], "quant": str(quant_dir) if quant_dir else None, "candidate": candidate,
              "resamples": resamples, "seed": seed, "title": title, "quantized": quantized}
    if overwrite and output.exists():
        for path in [*output.glob("figures/*.png"), *(output / n for n in ("report.md", "summary.json", "resolved_config.yaml", "manifest.json"))]:
            if path.is_file():
                path.unlink()
        if (output / "figures").is_dir() and not any((output / "figures").iterdir()):
            (output / "figures").rmdir()
    git_at_start = prepare_output_dir(output)
    runs = discover(runs_dirs)
    if not runs:
        raise FileNotFoundError(f"no run folders (metrics.jsonl) under {', '.join(map(str, runs_dirs))}")
    summary, _ = analyze(runs, load_quant(quant_dir), candidate=candidate, resamples=resamples, seed=seed, quantized=quantized)
    plots: dict[str, dict[str, str]] = {}
    if figures:
        (output / "figures").mkdir(exist_ok=True)
        for i, (label, g) in enumerate(summary["groups"].items()):
            slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or f"group-{i}"
            plots[label] = plot_group(label, g, output, slug, candidate=candidate)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    (output / "report.md").write_text(render(summary, plots, title=title))
    write_run_metadata(output, config, git_at_start=git_at_start, runs=len(runs), groups=list(summary["groups"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--quant", type=Path, default=None, help="e4_quant output folder over these runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="C5")
    parser.add_argument("--resamples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default="R9 — retrofit × quantization (E9)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="regenerate into an existing report folder")
    parser.add_argument("--quantized", default="int4", help="label of the per-run quantized evaluations (probes-<q>.json, edit-<q>, …)")
    args = parser.parse_args(argv)
    summary = write_report(args.runs, args.output, quant_dir=args.quant, candidate=args.candidate, resamples=args.resamples,
                           seed=args.seed, title=args.title, figures=not args.no_figures, overwrite=args.overwrite,
                           quantized=args.quantized)
    print(json.dumps({label: {"seeds": g["seeds"]} for label, g in summary["groups"].items()}))


if __name__ == "__main__":
    main()
