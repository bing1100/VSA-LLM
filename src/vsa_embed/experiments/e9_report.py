"""R9 report generator: retrofit × quantization (execution.md, E9; `reports/R9-retrofit-quantization.md`).

    python -m vsa_embed.experiments.e9_report --runs experiments/e9-retrofit/runs/<stage>
        [--quant experiments/e9-retrofit/quant/<stage>] [--quant-general experiments/e9-retrofit/quant-general/<stage>]
        --output <dir> [--candidate C5] [--resamples 10000] [--overwrite]

`--quant-general` (tracks with a general-text corpus) adds the locality and quantization damage on general
text: the dimension-2 analysis on `e4_quant --eval-corpus <eval-general>`, whose bf16 gain is C5 − reference
on general text. Track zero-shot outputs (`e9_tracks zeroshot`, WP-C7 items) are read like E5.4 outputs.

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

**WP-PQ1 sections** (when their inputs exist; see the comment above `pq_sections`): the operator / specificity
ablation of C5 (C5rf, C5ut, C5tr, C5sh), the same-site row sources (C6m, C6d, C6g), the filler / non-filler split of
the after-span strata and the claim-B quantization controls (both from `e9_rescore`'s `RUN/rescore`).

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


# ---------------------------------------------------------------- WP-PQ1: ablation arms, filler split, claim-B controls
#
# Paper-quality controls after the 2026-10 novelty check (`e9_plan` arms, `e9_rescore`). Each section appears only when
# its inputs exist, so reports of stages without them are unchanged:
#
# - operator / specificity ablation: `C5 − arm` per stratum for C5rf (fixed random orthogonal operators), C5ut (no
#   binding), C5tr (translation), C5sh (shuffled frames) — negative = C5 better, i.e. the ablated part matters — and
#   `arm − C0′` (does the ablated channel still help); final-evaluation windows, pooled over common seeds, Holm over the
#   arms within each stratum;
# - same-site row sources: the same for C6m (subtoken mean), C6d (definition encoder), C6g (TransE), held-out terms
#   first among the strata of interest;
# - filler / non-filler split (`RUN/rescore`, variant `ref`): every model difference on `X_filler` and `X_nonfiller`
#   for the after-span strata, the share of targets that are filler tokens and the share of the gain carried by them;
# - claim-B controls (`RUN/rescore`): for each quantizer `q` the absolute loss of reference and candidate at bf16 and
#   at `q`, the gaps, the difference in differences `DiD = (C5 − r)@q − (C5 − r)@bf16` with paired window bootstraps
#   (`e4_quant.retention`), the same with the candidate's channel off (`DiD_off`: what the C5-trained host weights do by
#   themselves) and the channel's own contribution `(C5 − C5off)@q − (C5 − C5off)@bf16`; Holm over quantizers within
#   each stratum.

OPERATOR_ARMS = ("C5rf", "C5ut", "C5tr", "C5sh")
SOURCE_ARMS = ("C6m", "C6d", "C6g")
ARM_LABELS = {"C5rf": "fixed random orthogonal operators", "C5ut": "untyped (no binding)", "C5tr": "translation x + r",
              "C5sh": "shuffled frames", "C6m": "subtoken mean (FVT)", "C6d": "definition encoder", "C6g": "TransE KG embedding"}
FILLER_BASES = ("after", "after_heldout", "after_rare_seen", "after_unseen", "after_len3plus")
CONTROL_QUANTIZERS = ("int4-A", "int4-rtn", "int4-hqq", "int4-nf4", "int4-gptq", "int4-awq", "int4-A-emb", "int4-A-embhead")
CONTROL_STRATA = ("after_heldout", "after_rare_seen", "after_unseen", "after_len3plus", "inside", "unlinked", "all")


def arm_comparison(group: Group, arms: Sequence[str], *, candidate: str, resamples: int, seed: int) -> dict[str, Any]:
    """`candidate − arm` and `arm − C0′` per stratum for the arms present (Holm over arms within a stratum)."""
    present = [a for a in arms if a in group.models]
    if candidate not in group.models or not present:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "arms": present, "strata": {}, "seeds": {}}
    strata_present = {s for runs in group.models.values() for r in runs.values() for s in r.final}
    for stratum in [s for s in STRATA if s in strata_present]:
        block: dict[str, dict[str, Any]] = {}
        for arm in present:
            common = sorted(set(group.seeds(candidate)) & set(group.seeds(arm)))
            result["seeds"][arm] = common
            entry = {}
            versus = paired_difference(group.paired(candidate, common), group.paired(arm, common), stratum, resamples=resamples, seed=seed)
            if versus is not None:
                entry["candidate_minus_arm"] = versus
            if "C0'" in group.models:
                base = sorted(set(group.seeds(arm)) & set(group.seeds("C0'")))
                over = paired_difference(group.paired(arm, base), group.paired("C0'", base), stratum, resamples=resamples, seed=seed)
                if over is not None:
                    entry["arm_minus_baseline"] = over
            block[arm] = entry
        for key in ("candidate_minus_arm", "arm_minus_baseline"):
            tested = [e[key] for e in block.values() if e.get(key, {}).get("p_value") is not None]
            for comparison, adjusted in zip(tested, holm_adjust([c["p_value"] for c in tested])):
                comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
        result["strata"][stratum] = block
    return result


def load_rescores(group: Group) -> dict[str, dict[int, dict[str, Any]]]:
    """`RUN/rescore` outputs (`e9_rescore`) of the group's runs: model → seed → rescoring."""
    from .e9_rescore import load_rescore
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for model, runs in group.models.items():
        for s, run in runs.items():
            found = load_rescore(run.path / "rescore")
            if found is not None:
                out.setdefault(model, {})[s] = found
    return out


def _rescore_sums(rescores: dict[str, dict[int, dict[str, Any]]], model: str, seeds: Sequence[int], variant: str,
                  stratum: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Window sums and counts of `model` pooled over `seeds` (P0: its one run for every seed); a model without a
    channel answers `-off` with its plain variant. None if missing."""
    from .e9_rescore import plain_variant
    by_seed = rescores.get(model, {})
    picks = [by_seed[min(by_seed)]] * len(seeds) if model == "P0" and by_seed else [by_seed.get(s) for s in seeds]
    if not picks or any(p is None for p in picks):
        return None
    sums, counts = 0, 0
    for found in picks:
        if stratum not in found["strata"]:
            return None
        name = variant if variant in found["sums"] else plain_variant(variant)
        if name not in found["sums"] or (name != variant and found["record"].get("channel", "none") != "none"):
            return None
        i = found["strata"].index(stratum)
        sums = sums + found["sums"][name][i]
        counts = counts + found["count"][i].astype(np.float64)
    return sums, counts


def filler_split(group: Group, rescores: dict[str, dict[int, dict[str, Any]]], *, candidate: str, resamples: int,
                 seed: int) -> dict[str, Any]:
    """Candidate − reference on `X_filler` / `X_nonfiller` (bf16, `ref`), paired by window, pooled over common seeds."""
    if candidate not in rescores:
        return {"available": False}
    references = [m for m in (*REFERENCES, *OPERATOR_ARMS, *SOURCE_ARMS) if m in rescores and m != candidate]
    result: dict[str, Any] = {"available": True, "references": references, "rows": {}}
    for base in FILLER_BASES:
        for reference in references:
            common = (group.seeds(candidate) if reference == "P0"
                      else sorted(set(rescores[candidate]) & set(rescores[reference])))
            common = [s for s in common if s in rescores[candidate]]
            parts: dict[str, Any] = {}
            for part in ("", "_filler", "_nonfiller"):
                stratum = base + part
                a = _rescore_sums(rescores, candidate, common, "ref", stratum)
                b = _rescore_sums(rescores, reference, common, "ref", stratum)
                if a is None or b is None or not np.array_equal(a[1], b[1]) or a[1].sum() == 0:
                    continue
                boot = paired_ratio_bootstrap(a[0] - b[0], a[1], b[0], resamples=resamples, seed=seed)
                parts[part.lstrip("_") or "total"] = {**{k: boot.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative",
                                                                        "relative_ci_low", "relative_ci_high")},
                                          "targets": int(a[1].sum()), "seeds": common,
                                          "gain_sum": float((a[0] - b[0]).sum())}
            if "total" in parts and "filler" in parts:
                total = parts["total"]
                parts["filler_target_share"] = parts["filler"]["targets"] / total["targets"] if total["targets"] else None
                parts["filler_gain_share"] = parts["filler"]["gain_sum"] / total["gain_sum"] if total["gain_sum"] else None
            if parts:
                result["rows"].setdefault(base, {})[reference] = parts
    tested = [p[k] for by_ref in result["rows"].values() for p in by_ref.values() for k in ("filler", "nonfiller")
              if isinstance(p.get(k), dict) and p[k].get("p_value") is not None]
    for comparison, adjusted in zip(tested, holm_adjust([c["p_value"] for c in tested]) if tested else []):
        comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
    return result


def claim_b_controls(group: Group, rescores: dict[str, dict[int, dict[str, Any]]], *, candidate: str, resamples: int,
                     seed: int) -> dict[str, Any]:
    """Difference in differences of the candidate's gap per quantizer, with and without its channel (module comment)."""
    if candidate not in rescores:
        return {"available": False}
    variants = {v for found in rescores[candidate].values() for v in found["sums"]}
    quantizers = [q for q in CONTROL_QUANTIZERS if q in variants] + sorted(
        v for v in variants if v.startswith("int") and not v.endswith("-off") and v not in CONTROL_QUANTIZERS)
    if "ref" not in variants or not quantizers:
        return {"available": False}
    result: dict[str, Any] = {"available": True, "quantizers": quantizers, "references": {}}
    for reference in [r for r in REFERENCES if r in rescores and r != candidate]:
        seeds = (sorted(rescores[candidate]) if reference == "P0" else sorted(set(rescores[candidate]) & set(rescores[reference])))
        if not seeds:
            continue
        block: dict[str, dict[str, Any]] = {}
        for q in quantizers:
            for stratum in CONTROL_STRATA:
                c_ref, r_ref = _rescore_sums(rescores, candidate, seeds, "ref", stratum), _rescore_sums(rescores, reference, seeds, "ref", stratum)
                c_q, r_q = _rescore_sums(rescores, candidate, seeds, q, stratum), _rescore_sums(rescores, reference, seeds, q, stratum)
                if None in (c_ref, r_ref, c_q, r_q) or c_ref[1].sum() == 0:
                    continue
                n = c_ref[1]
                entry: dict[str, Any] = {
                    "cells": {"reference_bf16": float(r_ref[0].sum() / n.sum()), "candidate_bf16": float(c_ref[0].sum() / n.sum()),
                              "reference_q": float(r_q[0].sum() / n.sum()), "candidate_q": float(c_q[0].sum() / n.sum())},
                    "did": retention(c_ref[0], r_ref[0], c_q[0], r_q[0], n, resamples=resamples, seed=seed), "seeds": seeds,
                    "targets": int(n.sum())}
                c_off, c_q_off = (_rescore_sums(rescores, candidate, seeds, "ref-off", stratum),
                                  _rescore_sums(rescores, candidate, seeds, f"{q}-off", stratum))
                if c_off is not None and c_q_off is not None:
                    entry["cells"].update(candidate_off_bf16=float(c_off[0].sum() / n.sum()), candidate_off_q=float(c_q_off[0].sum() / n.sum()))
                    entry["did_off"] = retention(c_off[0], r_ref[0], c_q_off[0], r_q[0], n, resamples=resamples, seed=seed)
                    entry["channel"] = retention(c_ref[0], c_off[0], c_q[0], c_q_off[0], n, resamples=resamples, seed=seed)
                block.setdefault(q, {})[stratum] = entry
        for stratum in CONTROL_STRATA:
            tested = [block[q][stratum]["did"]["gain_change"] for q in quantizers
                      if stratum in block.get(q, {}) and block[q][stratum]["did"]["gain_change"].get("p_value") is not None]
            for test, adjusted in zip(tested, holm_adjust([t["p_value"] for t in tested]) if tested else []):
                test.update(holm_p=adjusted, significant=adjusted < ALPHA)
        result["references"][reference] = block
    return result


def pq_sections(group: Group, *, candidate: str, resamples: int, seed: int) -> dict[str, Any]:
    """The WP-PQ1 sections of one group (empty when none of their inputs exist)."""
    out: dict[str, Any] = {}
    operators = arm_comparison(group, OPERATOR_ARMS, candidate=candidate, resamples=resamples, seed=seed)
    if operators.get("available"):
        out["operator_ablation"] = operators
    sources = arm_comparison(group, SOURCE_ARMS, candidate=candidate, resamples=resamples, seed=seed)
    if sources.get("available"):
        out["row_sources"] = sources
    rescores = load_rescores(group)
    if rescores:
        split = filler_split(group, rescores, candidate=candidate, resamples=resamples, seed=seed)
        if split.get("available") and split["rows"]:
            out["filler_split"] = split
        controls = claim_b_controls(group, rescores, candidate=candidate, resamples=resamples, seed=seed)
        if controls.get("available") and controls["references"]:
            out["claim_b_controls"] = controls
        out["rescored"] = {m: sorted(by_seed) for m, by_seed in rescores.items()}
    return out


def _render_arms(block: dict[str, Any], candidate: str, title: str, note: str) -> list[str]:
    arms = block["arms"]
    lines = ["", f"### {title}", "", note, "",
             "Seeds: " + "; ".join(f"{a} {block['seeds'].get(a, [])}" for a in arms) + ".", "",
             f"Relative loss difference [95% CI] (`*` = Holm over the arms within the stratum; negative {candidate} − arm = "
             f"{candidate} better):", "",
             "| Stratum | " + " | ".join(f"{candidate} − {a}" for a in arms) + " | " + " | ".join(f"{a} − C0′" for a in arms) + " |",
             "|---|" + "---|" * (2 * len(arms))]
    for stratum, by_arm in block["strata"].items():
        lines.append(f"| {_stratum_label(stratum)} | " + " | ".join(_relative(by_arm.get(a, {}).get("candidate_minus_arm")) for a in arms)
                     + " | " + " | ".join(_relative(by_arm.get(a, {}).get("arm_minus_baseline")) for a in arms) + " |")
    lines += ["", f"Absolute {candidate} − arm (nats/token):", "", "| Stratum | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for stratum, by_arm in block["strata"].items():
        lines.append(f"| {stratum} | " + " | ".join(_ci(by_arm.get(a, {}).get("candidate_minus_arm"), "delta") for a in arms) + " |")
    if any(len(block["seeds"].get(a, [])) == 1 for a in arms):
        lines += ["", "> Single seed for at least one arm: CIs cover evaluation windows only."]
    return lines


def render_pq(pq: dict[str, Any], candidate: str) -> list[str]:
    """Markdown of `pq_sections`."""
    lines: list[str] = []
    if "operator_ablation" in pq:
        lines += _render_arms(pq["operator_ablation"], candidate, "WP-PQ1 — operator and specificity ablation of " + candidate,
                              "Arms: " + "; ".join(f"**{a}** {ARM_LABELS[a]}" for a in pq["operator_ablation"]["arms"])
                              + ". Same recipe, test set and parameters as the candidate otherwise.")
    if "row_sources" in pq:
        lines += _render_arms(pq["row_sources"], candidate, "WP-PQ1 — same-site row sources (vs " + candidate + ")",
                              "Arms: " + "; ".join(f"**{a}** {ARM_LABELS[a]}" for a in pq["row_sources"]["arms"])
                              + ": a frozen per-entry vector through a trained MLP projector (parameters matched to the "
                              "candidate's dictionary plus projector), at the same site and gate; held-out terms get their "
                              "vectors from the same source. `after_heldout` is the zero-shot comparison.")
    split = pq.get("filler_split")
    if split:
        lines += ["", "### WP-PQ1 — filler vs non-filler targets after a term (copy concern)", "",
                  "`X_filler` = targets in the 8-token window after a span that belong to an alias of a filler of that span's "
                  "ontology frame (starting after the span); `X_nonfiller` = the rest of `X` (`e9_rescore`, bf16). Relative "
                  f"loss difference {candidate} − reference [95% CI] (`*` = Holm over all filler / non-filler tests); "
                  "filler share = fraction of the stratum's targets that are filler tokens; gain share = fraction of the "
                  "summed loss difference carried by filler tokens.", "",
                  "| Stratum | Reference | targets | filler share | total | filler | non-filler | gain share on fillers |",
                  "|---|---|---:|---:|---|---|---|---:|"]
        for base, by_ref in split["rows"].items():
            for reference, parts in by_ref.items():
                total = parts.get("total") or {}
                lines.append(f"| {base} | {reference} | {total.get('targets', 'n/a')} | "
                             f"{_fmt(parts.get('filler_target_share'), 3, signed=False)} | {_relative(parts.get('total'))} | "
                             f"{_relative(parts.get('filler'))} | {_relative(parts.get('nonfiller'))} | "
                             f"{_fmt(parts.get('filler_gain_share'), 3, signed=False)} |")
    controls = pq.get("claim_b_controls")
    if controls:
        lines += ["", "### WP-PQ1 — claim-B controls (quantizers, channel off, quantized embedding)", "",
                  "Absolute nats/token in the four cells, gap = candidate − reference, `DiD` = gap@q − gap@bf16 (negative = "
                  "the advantage grows under quantization; `*` = Holm over quantizers within the stratum), `DiD off` = the "
                  "same with the candidate's channel switched off at both precisions (the C5-trained host weights alone), "
                  "`channel DiD` = (C5 − C5off)@q − (C5 − C5off)@bf16 (the channel's own contribution). `int4-A` = torchao "
                  "RTN; `int4-rtn/hqq/nf4/gptq/awq` simulated (`evaluation.quantization`); `-emb` = input embedding "
                  "quantized too. Paired window bootstraps, pooled over common seeds.", ""]
        for reference, block in controls["references"].items():
            lines += [f"**{candidate} vs {reference}**", "",
                      "| Stratum | n | q | ref bf16 | cand bf16 | ref q | cand q | gap bf16 | gap q | DiD [95% CI] | DiD off [95% CI] | channel DiD [95% CI] |",
                      "|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|"]
            for stratum in CONTROL_STRATA:
                for q in controls["quantizers"]:
                    entry = block.get(q, {}).get(stratum)
                    if entry is None:
                        continue
                    cells, did = entry["cells"], entry["did"]
                    lines.append(f"| {stratum} | {entry['targets']} | {q} | {cells['reference_bf16']:.4f} | {cells['candidate_bf16']:.4f} | "
                                 f"{cells['reference_q']:.4f} | {cells['candidate_q']:.4f} | {_fmt(did['gain_bf16']['mean'])} | "
                                 f"{_fmt(did['gain_quantized']['mean'])} | {_ci(did['gain_change'])} | "
                                 f"{_ci((entry.get('did_off') or {}).get('gain_change'))} | {_ci((entry.get('channel') or {}).get('gain_change'))} |")
            lines.append("")
    if pq.get("rescored"):
        lines += ["", "Rescored runs (`RUN/rescore`): " + "; ".join(f"{m} {s}" for m, s in pq["rescored"].items()) + "."]
    return lines


# ---------------------------------------------------------------- analysis and rendering


def analyze(runs: Sequence[Run], quant: dict[str, dict[str, Any]], *, candidate: str = "C5", resamples: int = 10_000,
            seed: int = 0, quantized: str = "int4", quant_general: dict[str, dict[str, Any]] | None = None
            ) -> tuple[dict[str, Any], list[Group]]:
    """`quantized` names the per-run quantized evaluations (`probes-<q>.json`, `zeroshot-<q>`, `edit-<q>`);
    `quant_general` is `e4_quant` on the track's general-text corpus (locality and general-text damage:
    the dimension-2 analysis on that corpus, whose `gain_bf16` is the bf16 locality difference)."""
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
        if quant_general:
            summary["groups"][group.label]["general_text"] = dimension2(group, quant_general, candidate=candidate,
                                                                       resamples=resamples, seed=seed)
        pq = pq_sections(group, candidate=candidate, resamples=resamples, seed=seed)      # WP-PQ1 (only when present)
        if pq:
            summary["groups"][group.label]["pq"] = pq
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
        general = g.get("general_text")
        if general is not None:
            lines += ["", "### General text (the track's `eval-general`: locality and general-text quantization damage)", ""]
            if not general.get("available"):
                lines.append(f"Not available: {general.get('detail')}.")
            else:
                keep = [s for s in ("all", "unlinked") if s in general["strata"]]
                lines += [f"{candidate} − reference at bf16 (locality, nats/token; positive = worse on general text) and its "
                          "change under quantization (Δgain; negative = the gap grows; Holm over references × variants):", "",
                          "| Stratum | Reference | Variant | gain bf16 [95% CI] | Δgain [95% CI] |", "|---|---|---|---|---|"]
                for reference, by_variant in general["gap"].items():
                    for variant in general["variants"]:
                        for stratum in keep:
                            r = by_variant.get(variant, {}).get(stratum)
                            if r is not None:
                                lines.append(f"| {stratum} | {reference} | {variant} | {_ci(r['gain_bf16'])} | {_ci(r['gain_change'])} |")
                lines += ["", "Quantization damage on general text (relative):", "",
                          "| Model | " + " | ".join(general["variants"]) + " |", "|---|" + "---|" * len(general["variants"])]
                for model, by_variant in general["damage"].items():
                    lines.append(f"| {model} | " + " | ".join(_relative(by_variant.get(v, {}).get("all")) for v in general["variants"]) + " |")
        lines += ["", "### Dimension 3 — zero-shot learning by ontology editing (no weight update)", ""]
        if len(g["seeds"].get(candidate, [])) == 1:
            lines += ["> Single seed: intervals cover items only, not seed variance.", ""]
        lines += _render_dimension3(d3, candidate)
        if g.get("pq"):
            lines += render_pq(g["pq"], candidate)
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
            series = [m for m in MODELS if m in d3[variant]["models"]] + [m for m in d3[variant]["models"] if m not in MODELS]
            _dot_panel(ax, rows, tests, series, scale=1.0, key="mean", xlabel="accuracy (own rows; bar = seed range)")
            ax.set_title(f"new words — {variant}", fontsize=9, color=INK, loc="left")
            if ax is not axes[0][0]:
                ax.set_yticklabels([])
            if ax is not axes[0][-1] and ax.get_legend() is not None:
                ax.get_legend().remove()          # one legend, on the last panel
        fig.suptitle(f"Dimension 3 — zero-shot new words by ontology editing ({label})", fontsize=9, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9)); name = f"figures/d3-{slug}.png"; fig.savefig(out / name, dpi=150); plt.close(fig)
        figures["Dimension 3: new-word accuracy per model"] = name
    return figures


# ---------------------------------------------------------------- dimension-3 baselines (WP-PQ2; opt-in `--dim3-baselines`)


def dimension3_baselines(group: Group, *, candidate: str, resamples: int, seed: int) -> dict[str, Any]:
    """The `e9_dim3_baselines` evaluations of the group's runs (`RUN/dim3-baselines`): in-context frames and IKE, ROME /
    MEMIT / AlphaEdit on the host weights, frame transplant, channel-off audit, intra-entity locality, row sources; and the
    candidate's ontology edit vs every other model × method, paired over edit items (`e9_dim3_baselines.stage_summary`)."""
    from . import e9_dim3_baselines as dim3
    runs = {model: {s: run.path for s, run in by_seed.items()} for model, by_seed in group.models.items()}
    return dim3.stage_summary(runs, candidate=candidate, resamples=resamples, seed=seed)


def render_dimension3_baselines(summary: dict[str, Any]) -> list[str]:
    from . import e9_dim3_baselines as dim3
    lines = ["", "## Dimension-3 baselines (claim C; novelty check §4.4)", ""]
    for label, g in summary["groups"].items():
        if "dimension3_baselines" in g:
            lines += [f"### {label}", ""] + dim3.render_stage(g["dimension3_baselines"], heading="####")
    return lines


# ---------------------------------------------------------------- CLI


def write_report(runs_dirs: Sequence[Path], output: Path, *, quant_dir: Path | None = None, candidate: str = "C5",
                 resamples: int = 10_000, seed: int = 0, title: str = "R9 — retrofit × quantization (E9)", figures: bool = True,
                 overwrite: bool = False, quantized: str = "int4", quant_general_dir: Path | None = None,
                 dim3_baselines: bool = False) -> dict[str, Any]:
    config = {"runs": [str(p) for p in runs_dirs], "quant": str(quant_dir) if quant_dir else None, "candidate": candidate,
              "resamples": resamples, "seed": seed, "title": title, "quantized": quantized,
              "quant_general": str(quant_general_dir) if quant_general_dir else None}
    if dim3_baselines:                                # opt-in key (configs of earlier reports unchanged)
        config["dim3_baselines"] = True
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
    summary, groups = analyze(runs, load_quant(quant_dir), candidate=candidate, resamples=resamples, seed=seed, quantized=quantized,
                              quant_general=load_quant(quant_general_dir) if quant_general_dir else None)
    if dim3_baselines:
        for group in groups:
            summary["groups"][group.label]["dimension3_baselines"] = dimension3_baselines(
                group, candidate=candidate, resamples=min(resamples, 10_000), seed=seed)
    plots: dict[str, dict[str, str]] = {}
    if figures:
        (output / "figures").mkdir(exist_ok=True)
        for i, (label, g) in enumerate(summary["groups"].items()):
            slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower() or f"group-{i}"
            plots[label] = plot_group(label, g, output, slug, candidate=candidate)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    text = render(summary, plots, title=title)
    if dim3_baselines:
        text += "\n".join(render_dimension3_baselines(summary)) + "\n"
    (output / "report.md").write_text(text)
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
    parser.add_argument("--quant-general", type=Path, default=None, help="e4_quant output on the track's general-text corpus")
    parser.add_argument("--dim3-baselines", action="store_true",
                        help="add the dimension-3 baselines section (RUN/dim3-baselines of e9_dim3_baselines; WP-PQ2)")
    args = parser.parse_args(argv)
    summary = write_report(args.runs, args.output, quant_dir=args.quant, candidate=args.candidate, resamples=args.resamples,
                           seed=args.seed, title=args.title, figures=not args.no_figures, overwrite=args.overwrite,
                           quantized=args.quantized, quant_general_dir=args.quant_general, dim3_baselines=args.dim3_baselines)
    print(json.dumps({label: {"seeds": g["seeds"]} for label, g in summary["groups"].items()}))


if __name__ == "__main__":
    main()
