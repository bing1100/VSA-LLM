"""E9 B-B1: the bit-matched unstructured side-module control for claim B (novelty check §3.4; pre-registration
`experiments/e9-retrofit/preregistration-addendum-2026-10-20-bb1.md`).

Claim B (R9, dimension 2) measures how C5's loss advantage over C0′ changes under 4-bit weight-only PTQ of the host. A
published mechanism (4-bit PTQ erases small full-fine-tuning deltas, while an update held in a separate module survives;
Zhang et al. ICLR 2025, Abitante et al. 2026) predicts the same pattern for *any* module outside the quantized linear
weights. The control arm **C0L** (`e9_plan.SIDE_ARMS`) is C0′ plus such a module with no structure: LoRA adapters on the
fully trained host's projections (`model.side_lora`), with as many parameters as C5's channel on the same track and host,
trained in the channel's parameter group, kept at the channel's precision (FP16) next to the INT4 host (variant A) or
quantized like the channel (variant B). C2 (a free per-term table matched to C5's parameters) is the keyed counterpart,
already in every stage.

This module reads the runs' claim-B rescorings (`RUN/rescore`, `e9_rescore`: `ref`, `ref-off`, `int4-A`, `int4-A-off`,
`int4-B`, `int4-hqq`, `int4-nf4`, `int4-gptq`, `int4-awq`; C0L with profile `side`, P0 / C0′ / C2 / C5 with `controls`) and,
for variant B of C2 / C5 (not in their rescorings), the stage's `e4_quant` windows (`quant/<stage>/runs/<stage>__<stem>/`;
same evaluation windows, checked). Window sums are pooled over the common seeds and compared by window (cluster bootstrap,
`e4_quant.retention`). For a candidate `c`, a reference `r` and a quantizer `q`:
`DiD = (c − r)@q − (c − r)@bf16` (positive = c's advantage over r shrinks under q), with the four absolute cells (B-B6).

Comparisons: `C5 vs C0L` (the primary: does the structured channel keep more of its advantage under INT4 than a bit-matched
unstructured module?), `C5 vs C0′` (R9's DiD, reproduced), `C0L vs C0′` (the side module's own INT4 robustness: the
mechanism), `C2 vs C0L` (keyed free table vs unkeyed module), `C5 vs C2`; with the modules off at both precisions
(`-off`): `C0L off vs C0′` (what the C0L-trained host weights do alone), the side module's own contribution
`(C0L − C0L off)@q − (…)@bf16`, and `C5 off vs C0L off` (the two co-trained hosts without their modules). Holm: the
primary family = `C5 vs C0L` under `int4-A` on the primary strata of every stage reported together; every other test is
adjusted over the quantizers within its (stage, comparison, stratum).

    python -m vsa_embed.experiments.e9_bb1 report --stages t5 t4 t7 [--host SmolLM2-360M] [--seeds 1 2 3]
        [--root experiments/e9-retrofit] --output experiments/e9-retrofit/report/bb1 [--resamples 10000]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ..provenance import git_state, write_run_metadata
from ..statistics import holm_adjust
from .e4_quant import retention
from .e9_rescore import load_rescore

ROOT = Path("experiments/e9-retrofit")
HOST = "SmolLM2-360M"
SIDE = "C0L"
MODELS = ("P0", "C0p", "C2", "C5", SIDE)
LABELS = {"P0": "P0", "C0p": "C0′", "C2": "C2", "C5": "C5", SIDE: "C0L", "C0L-off": "C0L off", "C5-off": "C5 off"}
QUANTIZERS = ("int4-A", "int4-B", "int4-hqq", "int4-nf4", "int4-gptq", "int4-awq")
PRIMARY_QUANTIZER = "int4-A"
PRIMARY_STRATA = ("after_heldout", "after_rare_seen", "after_len3plus")
STRATA = (*PRIMARY_STRATA, "after_unseen", "after", "inside", "unlinked", "all")
# (name, candidate, reference, candidate's channel / side module off, reference's off)
COMPARISONS = (
    ("C5 vs C0L", "C5", SIDE, False, False),
    ("C5 vs C0′", "C5", "C0p", False, False),
    ("C0L vs C0′", SIDE, "C0p", False, False),
    ("C2 vs C0L", "C2", SIDE, False, False),
    ("C5 vs C2", "C5", "C2", False, False),
    ("C0L off vs C0′", SIDE, "C0p", True, False),
    ("C0L vs C0L off", SIDE, SIDE, False, True),
    ("C5 off vs C0L off", "C5", SIDE, True, True),
)
OFF_QUANTIZERS = ("int4-A",)                 # the `-off` variants the rescorings hold
ALPHA = 0.05


def run_dir(root: Path, stage: str, host: str, model: str, seed: int) -> Path:
    mode = "frozen" if model == "P0" else "full"
    return Path(root) / "runs" / stage / f"{host}-{mode}-{model}-s{1 if model == 'P0' else seed}"


def _quant_windows(root: Path, stage: str, run: Path) -> dict[str, Any] | None:
    path = Path(root) / "quant" / stage / "runs" / f"{stage}__{run.name}" / "windows.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {"strata": data["strata"].tolist(), "starts": data["starts"], "count": data["count"],
                "sums": {k[4:]: data[k] for k in data.files if k.startswith("sum_")}}


class Stage:
    """One stage's rescorings (model → seed → windows), with the `e4_quant` windows as the source of variant B."""

    def __init__(self, root: Path, stage: str, host: str, seeds: Sequence[int]) -> None:
        self.root, self.name, self.host, self.seeds = Path(root), stage, host, list(seeds)
        self.rescores: dict[str, dict[int, dict[str, Any]]] = {}
        self.quant: dict[str, dict[int, dict[str, Any] | None]] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        for model in MODELS:
            for seed in ([1] if model == "P0" else self.seeds):
                run = run_dir(self.root, stage, host, model, seed)
                found = load_rescore(run / "rescore")
                if found is None:
                    continue
                self.rescores.setdefault(model, {})[seed] = found
                self.quant.setdefault(model, {})[seed] = _quant_windows(self.root, stage, run)
                config = run / "resolved_config.yaml"
                if model not in self.configs and config.exists():
                    self.configs[model] = yaml.safe_load(config.read_text())
        self.starts = next((found["starts"] for runs in self.rescores.values() for found in runs.values()), None)

    def common_seeds(self, *models: str) -> list[int]:
        sets = [set(self.rescores.get(m, {})) for m in models if m != "P0"]
        if any(not self.rescores.get(m) for m in models):
            return []
        return sorted(set.intersection(*sets)) if sets else [1]

    def has_module(self, model: str) -> bool:
        record = next(iter(self.rescores.get(model, {}).values()), {}).get("record", {})
        return record.get("channel", "none") != "none" or bool(record.get("side_lora"))

    def _one(self, model: str, seed: int, variant: str, stratum: str) -> tuple[np.ndarray, np.ndarray] | None:
        found = self.rescores[model][1 if model == "P0" else seed]
        name = variant
        if variant.endswith("-off") and not self.has_module(model):
            name = variant[:-4]                                     # no channel / side module: off = on
        if name == "int4-B" and not self.has_module(model):
            name = "int4-A"                                         # nothing besides the host to quantize: B = A
        source = found
        if name not in found["sums"]:
            source = (self.quant.get(model) or {}).get(1 if model == "P0" else seed)
            if source is None or name not in source["sums"] or not np.array_equal(source["starts"], found["starts"]):
                return None
        if stratum not in source["strata"] or stratum not in found["strata"]:
            return None
        i = source["strata"].index(stratum)
        counts = source["count"][i]
        if not np.array_equal(counts, found["count"][found["strata"].index(stratum)]):
            return None                                             # another window or stratum definition: never paired
        return source["sums"][name][i].astype(np.float64), counts.astype(np.float64)

    def pooled(self, model: str, seeds: Sequence[int], variant: str, stratum: str) -> tuple[np.ndarray, np.ndarray] | None:
        parts = [self._one(model, s, variant, stratum) for s in seeds]
        if not parts or any(p is None for p in parts):
            return None
        return sum(p[0] for p in parts), sum(p[1] for p in parts)


def compare(stage: Stage, candidate: str, reference: str, quantizer: str, stratum: str, *, candidate_off: bool,
            reference_off: bool, resamples: int, seed: int) -> dict[str, Any] | None:
    """DiD of `candidate − reference` from bf16 to `quantizer` on `stratum`, pooled over their common seeds."""
    seeds = stage.common_seeds(candidate, reference)
    if not seeds:
        return None
    suffix = lambda off: "-off" if off else ""                       # noqa: E731
    c_ref = stage.pooled(candidate, seeds, "ref" + suffix(candidate_off), stratum)
    r_ref = stage.pooled(reference, seeds, "ref" + suffix(reference_off), stratum)
    c_q = stage.pooled(candidate, seeds, quantizer + suffix(candidate_off), stratum)
    r_q = stage.pooled(reference, seeds, quantizer + suffix(reference_off), stratum)
    if None in (c_ref, r_ref, c_q, r_q) or c_ref[1].sum() == 0:
        return None
    n = c_ref[1]
    if not all(np.array_equal(n, x[1]) for x in (r_ref, c_q, r_q)):
        return None
    total = n.sum()
    return {"cells": {"reference_bf16": float(r_ref[0].sum() / total), "candidate_bf16": float(c_ref[0].sum() / total),
                      "reference_q": float(r_q[0].sum() / total), "candidate_q": float(c_q[0].sum() / total)},
            "did": retention(c_ref[0], r_ref[0], c_q[0], r_q[0], n, resamples=resamples, seed=seed),
            "seeds": seeds, "targets": int(total)}


def budget(stage: Stage) -> dict[str, Any]:
    """C0L's side-module pattern and parameters (its config) against C5's channel parameters (the matched budget)."""
    side = ((stage.configs.get(SIDE) or {}).get("model") or {}).get("side_lora")
    return {"side_lora": side, "c5_channel_parameters": (side or {}).get("matched_to")}


def analyze(root: Path, stages: Sequence[str], *, host: str = HOST, seeds: Sequence[int] = (1, 2, 3), resamples: int = 10_000,
            seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {"host": host, "seeds": list(seeds), "resamples": resamples, "primary": {
        "comparison": "C5 vs C0L", "quantizer": PRIMARY_QUANTIZER, "strata": list(PRIMARY_STRATA)}, "stages": {}}
    for name in stages:
        stage = Stage(root, name, host, seeds)
        block: dict[str, Any] = {"models": {m: sorted(r) for m, r in stage.rescores.items()}, "budget": budget(stage),
                                 "comparisons": {}}
        for label, candidate, reference, c_off, r_off in COMPARISONS:
            quantizers = OFF_QUANTIZERS if (c_off or r_off) else QUANTIZERS
            rows: dict[str, dict[str, Any]] = {}
            for q in quantizers:
                for stratum in STRATA:
                    found = compare(stage, candidate, reference, q, stratum, candidate_off=c_off, reference_off=r_off,
                                    resamples=resamples, seed=seed)
                    if found is not None:
                        rows.setdefault(q, {})[stratum] = found
            for stratum in STRATA:                                  # Holm over quantizers within (stage, comparison, stratum)
                tests = [rows[q][stratum]["did"]["gain_change"] for q in quantizers if stratum in rows.get(q, {})
                         and rows[q][stratum]["did"]["gain_change"].get("p_value") is not None]
                for test, adjusted in zip(tests, holm_adjust([t["p_value"] for t in tests]) if tests else []):
                    test.update(holm_p=adjusted, significant=adjusted < ALPHA)
            if rows:
                block["comparisons"][label] = rows
        out["stages"][name] = block
    family = [(name, stratum, out["stages"][name]["comparisons"]["C5 vs C0L"][PRIMARY_QUANTIZER][stratum]["did"]["gain_change"])
              for name in stages for stratum in PRIMARY_STRATA
              if stratum in out["stages"][name]["comparisons"].get("C5 vs C0L", {}).get(PRIMARY_QUANTIZER, {})
              and out["stages"][name]["comparisons"]["C5 vs C0L"][PRIMARY_QUANTIZER][stratum]["did"]["gain_change"].get("p_value") is not None]
    primary = []
    for (name, stratum, test), adjusted in zip(family, holm_adjust([t["p_value"] for _, _, t in family]) if family else []):
        reference = out["stages"][name]["comparisons"].get("C5 vs C0′", {}).get(PRIMARY_QUANTIZER, {}).get(stratum)
        entry = {"stage": name, "stratum": stratum, "did": test["mean"], "ci": [test["ci_low"], test["ci_high"]],
                 "p": test["p_value"], "holm_p_primary": adjusted, "significant": adjusted < ALPHA,
                 "did_vs_c0p": reference["did"]["gain_change"]["mean"] if reference else None,
                 "did_vs_c0p_ci": ([reference["did"]["gain_change"]["ci_low"], reference["did"]["gain_change"]["ci_high"]]
                                   if reference else None)}
        entry["reading"] = reading(entry)
        primary.append(entry)
    out["primary"]["tests"] = primary
    return out


def reading(entry: dict[str, Any]) -> str:
    """The pre-registered reading of one primary cell (addendum 2026-10-20 B-B1, §6)."""
    did, (low, high), significant = entry["did"], entry["ci"], entry["significant"]
    base = entry.get("did_vs_c0p")
    base_ci = entry.get("did_vs_c0p_ci") or [None, None]
    grows_vs_c0p = base is not None and base_ci[1] is not None and base_ci[1] < 0
    if significant and did < 0:
        return "structure: C5's advantage over the bit-matched side module grows under INT4"
    if significant and did > 0:
        return "shrinks: C5 keeps less of its advantage over the side module than the side module keeps over nothing"
    if grows_vs_c0p:
        return "separate module: the growth over C0′ is matched by the unstructured side module (no structure claim)"
    return "kept: no change of C5's advantage over the side module (CI includes 0)"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:+.{digits}f}"


def _did(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    change = entry["did"]["gain_change"]
    text = f"{_fmt(change['mean'])} [{_fmt(change['ci_low'])}, {_fmt(change['ci_high'])}]"
    return text + ("*" if change.get("significant") else "")


def render(summary: dict[str, Any]) -> str:
    lines = ["# E9 B-B1 — bit-matched unstructured side module under 4-bit PTQ", "",
             f"Host {summary['host']}, seeds {summary['seeds']}. C0L = C0′ plus LoRA adapters on the host's projections with as "
             "many parameters as C5's channel (no ontology, no composition, not keyed to terms), FP16 next to the INT4 host. "
             "DiD = (candidate − reference)@q − (candidate − reference)@bf16 in nats per token, pooled over common seeds, 95% "
             "cluster bootstrap over evaluation windows; positive = the candidate's advantage shrinks under q. `*` = Holm "
             "over the quantizers within the cell. Pre-registration: "
             "`experiments/e9-retrofit/preregistration-addendum-2026-10-20-bb1.md`.", "",
             f"## Primary: C5 vs C0L under {summary['primary']['quantizer']} (Holm over every stage × stratum below)", "",
             "| stage | stratum | DiD C5 vs C0L [95% CI] | Holm p | DiD C5 vs C0′ (R9) | reading |", "|---|---|---|---:|---|---|"]
    for t in summary["primary"]["tests"]:
        lines.append(f"| {t['stage']} | {t['stratum']} | {_fmt(t['did'])} [{_fmt(t['ci'][0])}, {_fmt(t['ci'][1])}]"
                     f"{'*' if t['significant'] else ''} | {t['holm_p_primary']:.4f} | {_fmt(t['did_vs_c0p'])} | {t['reading']} |")
    if not summary["primary"]["tests"]:
        lines.append("| — | — | no C0L rescoring yet | | | |")
    for name, block in summary["stages"].items():
        side = block["budget"].get("side_lora") or {}
        lines += ["", f"## {name}", "", f"Rescorings found: {block['models']}. Side module: rank {side.get('rank', '?')} "
                  f"(+1 in the first {side.get('extra_rank_layers', '?')} layers), {side.get('parameters', '?')} parameters for "
                  f"C5's {side.get('matched_to', '?')} (ratio {side.get('ratio', '?')}).", ""]
        for label, rows in block["comparisons"].items():
            quantizers = [q for q in QUANTIZERS if q in rows]
            lines += [f"### {label}", "", "| stratum | " + " | ".join(quantizers) + " |", "|---|" + "---|" * len(quantizers)]
            for stratum in STRATA:
                if any(stratum in rows[q] for q in quantizers):
                    lines.append(f"| {stratum} | " + " | ".join(_did(rows[q].get(stratum)) for q in quantizers) + " |")
            cells = rows.get(PRIMARY_QUANTIZER, {})
            if cells:
                lines += ["", f"Cells under {PRIMARY_QUANTIZER} (nats per token; reference bf16 / candidate bf16 / reference q / "
                          "candidate q; targets):", ""]
                for stratum in STRATA:
                    if stratum in cells:
                        c = cells[stratum]["cells"]
                        lines.append(f"- {stratum}: {c['reference_bf16']:.4f} / {c['candidate_bf16']:.4f} / {c['reference_q']:.4f} / "
                                     f"{c['candidate_q']:.4f} ({cells[stratum]['targets']:,})")
            lines.append("")
    return "\n".join(lines) + "\n"


def report(root: Path, stages: Sequence[str], output: Path, *, host: str = HOST, seeds: Sequence[int] = (1, 2, 3),
           resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    git_at_start = git_state()
    summary = analyze(root, stages, host=host, seeds=seeds, resamples=resamples, seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (output / "report.md").write_text(render(summary))
    write_run_metadata(output, {"experiment": "e9-bb1-report", "root": str(root), "stages": list(stages), "host": host,
                                "seeds": list(seeds), "resamples": resamples, "seed": seed}, git_at_start=git_at_start)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="the B-B1 claim-B table from the stages' rescorings")
    rep.add_argument("--stages", nargs="+", required=True); rep.add_argument("--host", default=HOST)
    rep.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3]); rep.add_argument("--root", type=Path, default=ROOT)
    rep.add_argument("--output", type=Path, required=True); rep.add_argument("--resamples", type=int, default=10_000)
    rep.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    summary = report(args.root, args.stages, args.output, host=args.host, seeds=args.seeds, resamples=args.resamples, seed=args.seed)
    print(json.dumps({"output": str(args.output), "primary": summary["primary"]["tests"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
