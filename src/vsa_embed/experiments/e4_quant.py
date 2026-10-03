"""E4.4 / D4.3: post-training weight-only quantization of trained ChannelLM runs (H-B).

    python -m vsa_embed.experiments.e4_quant --runs <run dir or root> [...] --output <dir>
        [--bits 8 4] [--variants A B] [--group-size auto] [--baseline C0] [--references C2 ...]
        [--resamples 10000] [--seed 0] [--windows N] [--eval-batch N] [--resume] [--report-only] [--eval-corpus DIR]

`--eval-corpus` (opt-in) evaluates every run on another token corpus than its `data.eval` — E9 uses a
track's general-text `eval-general` for locality and general-text quantization damage.

Every folder under `--runs` holding `final.pt` and `metrics.jsonl` is a trained run: from scratch,
or continued pretraining, whose `final.pt` holds only the trainable parameters (`load_final` reloads
the frozen host from the Hugging Face cache). Each run is evaluated on its own evaluation windows
(the trainer's `eval_windows` for its config) as

- `ref` — the run as trained (fp32 master weights under bf16 autocast) through the trainer's own
  `evaluate`; its per-window losses equal the run's `eval_windows.npz` at the final evaluation
  (checked when that file exists: `ref_check`);
- `int<b>-A` — the host's linear layers quantized with torchao weight-only INT8 (per-row) or INT4
  (group-wise, tile-packed layout; CUDA only), LoRA adapters merged into their base weights first.
  The input embedding and the output head (tied or not) stay 16-bit, as in standard GPTQ/AWQ
  practice; the span channel (dictionary, relation parameters, composer, projector, gate) and the
  P1 context stay FP16;
- `int<b>-B` — as A, with the channel and context quantized too, by the same scheme simulated with
  group-wise round-to-nearest (`fake_quantize_module_`: INT8 symmetric per row, INT4 asymmetric with
  the host's group size). A run without a channel has no B variant (its B is its A).

`--group-size auto` uses the largest of 128 / 64 / 32 that divides every linear layer's input width
(SmolLM2-135M's 576 needs 64), so no layer is skipped for its shape; skipped layers are listed.

Outputs (run-folder contract: `resolved_config.yaml`, `manifest.json`):

- `runs/<id>/quant.json` — per variant: stratified losses, perplexity, bytes of each part
  (`embedding`; `head`, 0 when tied; `linear`, torchao layouts measured from their inner tensors;
  `other` — positions, norms; `lora`; `channel`; `channel_schedule`, packed integer buffers), the
  quantized and skipped linear layers, timing; plus the run's condition, seed and cohort.
- `runs/<id>/windows.npz` — `strata`, `starts`, `count` (strata × windows target counts) and
  `sum_<variant>` (strata × windows loss sums): every variant pairs by window with the others and
  with the run's own `eval_windows.npz`.
- `quant.json` (every run and the analysis) and `report.md` — the R4 section, per cohort (as in
  `e4_report`): perplexity change and bytes per variant, and the retained gain of each condition
  over the baseline (C0, or C0'/C0p for continued pretraining) and over C2. For a condition `c`, a
  reference `r` and a quantized variant `q`, with window sums pooled over the common seeds:
  `gain_bf16 = (c − r)` at `ref`, `gain_q = (c − r)` under `q`, their difference
  `Δgain = gain_q − gain_bf16` (positive = the advantage shrinks under quantization; Holm over the
  cohort's comparisons within each stratum), and the retained fraction `gain_q / gain_bf16`, each
  with a cluster bootstrap over evaluation windows (`statistics.paired_ratio_bootstrap`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from ..data.corpus import TokenCorpus, eval_windows
from ..evaluation.quantization import (auto_group_size, conv1d_to_linear, fake_quantize_module_, group_quantized_bytes,
                                       index_bytes, is_quantized, merge_lora, quantize_weight_only, tensor_storage_bytes)
from ..provenance import git_state, prepare_output_dir, write_run_metadata
from ..statistics import _cluster_weights, holm_adjust, paired_ratio_bootstrap
from ..training.lm import evaluate, load_final, load_window_losses, resolve_config
from .e4_report import KEY_STRATA, _json_default, cohort_labels, condition_order, load_run

VARIANT_LABELS = {"A": "channel FP16", "B": "channel quantized"}
BASELINES = ("C0", "C0'", "C0p")
ALPHA = 0.05


def find_runs(roots: Sequence[Path], *, exclude: Path | None = None) -> list[Path]:
    """Run folders (holding `final.pt` and `metrics.jsonl`) under `roots`, sorted."""
    found = set()
    for root in roots:
        for final in Path(root).rglob("final.pt"):
            folder = final.parent.resolve()
            if (folder / "metrics.jsonl").exists() and not (exclude and exclude.resolve() in folder.parents):
                found.add(folder)
    return sorted(found)


def run_id(path: Path) -> str:
    return f"{path.parent.name}__{path.name}"


def variant_names(bits: Sequence[int], variants: Sequence[str]) -> list[str]:
    return ["ref"] + [f"int{b}-{v}" for b in bits for v in variants]


# ---------------------------------------------------------------- quantize and account


def quantize_host(model: nn.Module, bits: int, group_size: int) -> dict[str, Any]:
    """Merge LoRA, cast the host's linear layers to bf16 (INT4's layout needs it; under bf16 autocast
    this is the precision they run at anyway) and quantize them with torchao in place."""
    host = model.model
    merged = merge_lora(host)
    conv1d_to_linear(host)
    head = host.get_output_embeddings()
    tied = head.weight is host.get_input_embeddings().weight
    for module in host.modules():
        if isinstance(module, nn.Linear) and module is not head:
            module.to(torch.bfloat16)
    quantize_weight_only(host, bits, group_size=group_size)
    linear = [(name, m) for name, m in host.named_modules() if isinstance(m, nn.Linear) and m is not head]
    return {"lora_merged": merged, "head_tied": tied,
            "linear_quantized": sum(is_quantized(m.weight) for _, m in linear),
            "linear_skipped": [name for name, m in linear if not is_quantized(m.weight)]}


def _channel_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name in ("channel", "context") if (module := getattr(model, name, None)) is not None]


def quantize_channel(model: nn.Module, bits: int, group_size: int) -> dict[str, int]:
    """Variant B: the channel and context, simulated with the host's scheme."""
    sizes = {}
    for prefix, module in _channel_modules(model):
        symmetric = bits == 8
        for name, size in fake_quantize_module_(module, bits, None if symmetric else group_size, symmetric=symmetric).items():
            sizes[f"{prefix}.{name}"] = size
    return sizes


def _schedule_bytes(name: str, tensor: torch.Tensor, module: nn.Module) -> int:
    """Packed integer buffers: frame offsets as degrees, other indices at ⌈log2(max + 1)⌉ bits."""
    if tensor.dtype == torch.bool:
        return math.ceil(tensor.numel() / 8)
    if name.endswith("frame_offsets"):
        degrees = tensor[1:] - tensor[:-1]
        return index_bytes(int(degrees.max()) + 1 if degrees.numel() else 1, degrees.numel())
    return index_bytes(int(tensor.max()) + 1 if tensor.numel() else 1, tensor.numel())


def model_bytes(model: nn.Module, channel_sizes: dict[str, int] | None = None, *, bits: int | None = None,
                group_size: int | None = None) -> dict[str, int]:
    """Bytes of each part as deployed: plain floating tensors at 16 bits, torchao weights as measured
    from their inner tensors, the channel's quantized tensors as `channel_sizes` (variant B) and its
    integer buffers packed. torchao's tile-packed INT4 layout pads the input width to a multiple of
    1024 (a 512-wide INT4 layer takes as many bytes as INT8), so `linear_nominal` / `total_nominal`
    also count the quantized weights at their packed size (`group_quantized_bytes`: codes plus FP16
    scale and zero point per group, INT8 symmetric per row) without layout padding."""
    host = model.model
    embedding = host.get_input_embeddings().weight
    head_module = host.get_output_embeddings()
    parts = {"embedding": tensor_storage_bytes(embedding, float_bytes=2), "head": 0, "linear": 0, "lora": 0,
             "other": 0, "channel": 0, "channel_schedule": 0}
    seen = {id(embedding)}
    if id(head_module.weight) not in seen:
        parts["head"] = tensor_storage_bytes(head_module.weight, float_bytes=2)
        seen.add(id(head_module.weight))
    linear_ids = {id(p) for m in host.modules() if (isinstance(m, nn.Linear) or type(m).__name__ == "Conv1D")
                  and m is not head_module for p in m.parameters(recurse=False)}
    nominal = 0
    for name, parameter in host.named_parameters(remove_duplicate=True):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        part = "lora" if ("lora_a" in name or "lora_b" in name) else "linear" if id(parameter) in linear_ids else "other"
        size = tensor_storage_bytes(parameter, float_bytes=2)
        parts[part] += size
        if part == "linear":
            if is_quantized(parameter):
                if bits is None:
                    raise ValueError("quantized weights need `bits` for their nominal size")
                rows, columns = parameter.shape
                size = group_quantized_bytes(rows, columns, bits, None if bits == 8 else group_size, symmetric=bits == 8)
            nominal += size
    for prefix, module in _channel_modules(model):
        for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            key = f"{prefix}.{name}"
            if channel_sizes and key in channel_sizes:
                parts["channel"] += channel_sizes[key]
            elif tensor.is_floating_point():
                parts["channel"] += tensor.numel() * 2
            else:
                parts["channel_schedule"] += _schedule_bytes(name, tensor, module)
    parts["total"] = sum(parts.values())
    parts["linear_nominal"] = nominal
    parts["total_nominal"] = parts["total"] - parts["linear"] + nominal
    return parts


# ---------------------------------------------------------------- evaluation


def _strata_arrays(sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    strata = list(sink)
    sums = np.stack([np.concatenate(sink[s][0]) for s in strata])
    counts = np.stack([np.concatenate(sink[s][1]) for s in strata]).astype(np.int32)
    return strata, sums, counts


def evaluate_run(run_dir: Path, out_dir: Path, *, bits: Sequence[int], variants: Sequence[str], group_size: str | int,
                 device: torch.device, windows: int | None = None, eval_batch: int | None = None,
                 eval_corpus_path: Path | None = None) -> dict[str, Any]:
    """Evaluate one run as `ref` and every quantized variant; write `quant.json` and `windows.npz`.
    `eval_corpus_path` evaluates on another corpus than the run's `data.eval` (e.g. a track's
    general-text `eval-general`: locality and general-text quantization damage)."""
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")
    config = resolve_config(final["config"])
    del final
    if windows:
        config["eval"]["windows"] = int(windows)
    if eval_batch:
        config["eval"]["batch"] = int(eval_batch)
    if eval_corpus_path is not None:
        config["data"]["eval"] = str(eval_corpus_path)
    eval_corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    ontology = torch.load(config["data"]["ontology"], weights_only=False) if config["data"].get("ontology") else None
    heldout = set(ontology["heldout_entries"]) if ontology else set()
    frequency = np.asarray(ontology["train_frequency"]) if ontology else None
    del ontology
    starts = eval_windows(eval_corpus, count=config["eval"]["windows"], length=config["model"]["seq_len"])
    record: dict[str, Any] = {"run": str(run_dir), "id": run_id(run_dir), "windows": len(starts),
                              "channel": config["channel"]["mode"], "variants": {}}
    if eval_corpus_path is not None:                 # recorded only when it differs from the run's own corpus
        record["eval_corpus"] = str(eval_corpus_path)
    run = load_run(run_dir)
    record.update(condition=run.condition, seed=run.seed, model=run.model, cohort=list(run.cohort))
    arrays: dict[str, np.ndarray] = {}
    strata = counts = None
    has_channel = config["channel"]["mode"] != "none"
    for name in variant_names(bits, variants):
        if name != "ref" and name.endswith("-B") and not has_channel:
            continue                                 # B = A without a channel
        started = time.monotonic()
        model = load_final(run_dir / "final.pt", device)
        info: dict[str, Any] = {}
        channel_sizes = None
        if name != "ref":
            b = int(name[3:].split("-")[0])
            size = auto_group_size(model.model) if group_size == "auto" else int(group_size)
            info = {"bits": b, "group_size": size, **quantize_host(model, b, size)}
            if name.endswith("-B"):
                channel_sizes = quantize_channel(model, b, size)
        sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        results = evaluate(model, eval_corpus, starts, config, frequency, heldout, device, window_sink=sink)
        names, sums, these_counts = _strata_arrays(sink)
        if strata is None:
            strata, counts = names, these_counts
        elif names != strata or not np.array_equal(these_counts, counts):
            raise RuntimeError(f"{run_dir}: variant {name} produced different strata or target counts")
        arrays[f"sum_{name}"] = sums
        record["variants"][name] = {"strata": results, "ppl": math.exp(results["all"]["loss"]),
                                    "bytes": model_bytes(model, channel_sizes, bits=info.get("bits"), group_size=info.get("group_size")),
                                    **info,
                                    "seconds": round(time.monotonic() - started, 2)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    record["ref_check"] = reference_check(run_dir, strata, starts, arrays["sum_ref"], counts)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", strata=np.asarray(strata), starts=np.asarray(starts, dtype=np.int64),
                        count=counts, **arrays)
    (out_dir / "quant.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    return record


def reference_check(run_dir: Path, strata: list[str], starts: list[int], sums: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """Compare `ref` with the run's own per-window losses at its final evaluation."""
    path = run_dir / "eval_windows.npz"
    if not path.exists():
        return {"available": False}
    saved = load_window_losses(path)
    if not saved["evals"] or not np.array_equal(saved["starts"], np.asarray(starts)):
        return {"available": True, "paired": False, "detail": "different evaluation windows"}
    tokens = max(saved["evals"])
    run_sums, run_counts = saved["evals"][tokens]
    index = [saved["strata"].index(s) for s in strata if s in saved["strata"]]
    mine = [strata.index(s) for s in strata if s in saved["strata"]]
    return {"available": True, "paired": True, "final_tokens": tokens,
            "counts_equal": bool(np.array_equal(run_counts[index], counts[mine])),
            "max_abs_window_sum_diff": float(np.abs(run_sums[index] - sums[mine]).max()) if index else None}


# ---------------------------------------------------------------- analysis


def _load_windows(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    with np.load(output / "runs" / record["id"] / "windows.npz") as data:
        return {"strata": data["strata"].tolist(), "count": data["count"],
                **{key[4:]: data[key] for key in data.files if key.startswith("sum_")}}


def _variant(windows: dict[str, Any], name: str) -> np.ndarray:
    """Window sums of a variant; a run without a channel answers B with its A."""
    if name in windows:
        return windows[name]
    if name.endswith("-B") and name[:-1] + "A" in windows:
        return windows[name[:-1] + "A"]
    raise KeyError(f"variant {name} was not evaluated")


def retention(c_ref: np.ndarray, r_ref: np.ndarray, c_q: np.ndarray, r_q: np.ndarray, n: np.ndarray, *,
              resamples: int, seed: int) -> dict[str, Any]:
    """Gain (c − r) at bf16 and under quantization, its change, and the retained fraction, from
    window sums pooled over seeds (cluster bootstrap over windows, common resamples)."""
    gain_ref = paired_ratio_bootstrap(c_ref - r_ref, n, r_ref, resamples=resamples, seed=seed)
    gain_q = paired_ratio_bootstrap(c_q - r_q, n, r_q, resamples=resamples, seed=seed)
    change = paired_ratio_bootstrap((c_q - r_q) - (c_ref - r_ref), n, r_ref, resamples=resamples, seed=seed)
    weights = _cluster_weights(n.size, resamples, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = (weights @ (c_q - r_q)) / (weights @ (c_ref - r_ref))
        ratios = ratios[np.isfinite(ratios)]
        point = float((c_q - r_q).sum() / (c_ref - r_ref).sum()) if (c_ref - r_ref).sum() != 0 else float("nan")
    keys = ("mean", "ci_low", "ci_high", "p_value", "relative", "relative_ci_low", "relative_ci_high")
    return {"gain_bf16": {k: gain_ref.get(k) for k in keys}, "gain_quantized": {k: gain_q.get(k) for k in keys},
            "gain_change": {k: change.get(k) for k in keys},
            "retained": {"mean": point, "ci_low": float(np.quantile(ratios, 0.025)) if ratios.size else None,
                         "ci_high": float(np.quantile(ratios, 0.975)) if ratios.size else None},
            "windows": int((n > 0).sum()), "tokens": int(n.sum())}


def analyze(records: list[dict[str, Any]], output: Path, *, baseline: str = "C0", references: Sequence[str] = ("C2",),
            resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    by_cohort: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        by_cohort.setdefault(tuple(record["cohort"]), []).append(record)
    labels = cohort_labels(list(by_cohort))
    cohorts = {}
    for key, members in by_cohort.items():
        grid: dict[str, dict[int, dict[str, Any]]] = {}
        for record in sorted(members, key=lambda r: r["run"]):
            grid.setdefault(record["condition"], {}).setdefault(int(record["seed"]), record)
        grid = {c: grid[c] for c in sorted(grid, key=condition_order)}
        base = next((b for b in (baseline, *[x for x in BASELINES if x != baseline]) if b in grid), None)
        refs = [r for r in dict.fromkeys([base, *references]) if r in grid]
        windows = {(c, s): _load_windows(output, rec) for c, runs in grid.items() for s, rec in runs.items()}
        quantized = sorted({v for rec in members for v in rec["variants"] if v != "ref"},
                           key=lambda v: (-int(v[3:].split("-")[0]), v))
        strata = list(next(iter(windows.values()))["strata"]) if windows else []
        all_index = strata.index("all")
        # Perplexity change and bytes per condition and variant (pooled over seeds).
        conditions: dict[str, Any] = {}
        for condition, runs in grid.items():
            seeds = sorted(runs)
            entry: dict[str, Any] = {"seeds": seeds, "variants": {}}
            for variant in ["ref", *quantized]:
                ref_sums = sum(windows[(condition, s)]["ref"][all_index] for s in seeds)
                sums = sum(_variant(windows[(condition, s)], variant)[all_index] for s in seeds)
                n = sum(windows[(condition, s)]["count"][all_index].astype(np.float64) for s in seeds)
                loss = float(sums.sum() / n.sum())
                v_records = [runs[s]["variants"].get(variant) or runs[s]["variants"].get(variant[:-1] + "A") for s in seeds]
                row = {"loss": loss, "ppl": math.exp(loss),
                       "bytes": {k: float(np.mean([r["bytes"][k] for r in v_records])) for k in v_records[0]["bytes"]}}
                if variant != "ref":
                    change = paired_ratio_bootstrap(sums - ref_sums, n, ref_sums, resamples=resamples, seed=seed)
                    row.update(loss_change=change["mean"], loss_change_ci=[change["ci_low"], change["ci_high"]],
                               ppl_ratio=math.exp(change["mean"]),
                               linear_skipped=sorted({x for r in v_records for x in r.get("linear_skipped", [])}),
                               group_size=v_records[0].get("group_size"))
                entry["variants"][variant] = row
            conditions[condition] = entry
        # Retained gain of every condition over each reference.
        comparisons: dict[str, Any] = {}
        for reference in refs:
            for condition in grid:
                if condition == reference:
                    continue
                seeds = sorted(set(grid[condition]) & set(grid[reference]))
                if not seeds:
                    continue
                for variant in quantized:
                    for stratum in strata:
                        i = strata.index(stratum)
                        n = sum(windows[(condition, s)]["count"][i].astype(np.float64) for s in seeds)
                        if n.sum() == 0:
                            continue
                        def pooled(c: str, v: str) -> np.ndarray:
                            return sum(_variant(windows[(c, s)], v)[i] for s in seeds)
                        result = retention(pooled(condition, "ref"), pooled(reference, "ref"), pooled(condition, variant),
                                           pooled(reference, variant), n, resamples=resamples, seed=seed)
                        result["seeds"] = seeds
                        comparisons.setdefault(reference, {}).setdefault(condition, {}).setdefault(variant, {})[stratum] = result
        for stratum in strata:     # Holm over the cohort's gain-change tests within each stratum
            family = [by_v[stratum]["gain_change"] for by_c in comparisons.values() for by_variant in by_c.values()
                      for by_v in by_variant.values() if stratum in by_v and by_v[stratum]["gain_change"].get("p_value") is not None]
            for test, adjusted in zip(family, holm_adjust([t["p_value"] for t in family])):
                test.update(holm_p=adjusted, significant=adjusted < ALPHA)
        cohorts[labels[key]] = {"baseline": base, "references": refs, "strata": strata, "quantized_variants": quantized,
                                "grid": {c: sorted(r) for c, r in grid.items()}, "conditions": conditions,
                                "retention": comparisons,
                                "ref_checks": {rec["id"]: rec.get("ref_check") for rec in members}}
    return {"cohorts": cohorts, "resamples": resamples, "seed": seed}


# ---------------------------------------------------------------- rendering


def _ci(value: dict[str, Any] | None, scale: float = 1.0, digits: int = 4) -> str:
    if not value or value.get("mean") is None or not math.isfinite(value["mean"]):
        return "n/a"
    text = f"{value['mean'] * scale:+.{digits}f}"
    if value.get("ci_low") is not None:
        text += f" [{value['ci_low'] * scale:+.{digits}f}, {value['ci_high'] * scale:+.{digits}f}]"
    return text + ("*" if value.get("significant") else "")


def _mib(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 2**20:.2f}"


def render(summary: dict[str, Any], *, title: str, records: list[dict[str, Any]]) -> str:
    lines = [f"# {title}", "",
             "Post-training weight-only quantization (torchao; INT8 per-row, INT4 group-wise tile-packed) of the host's "
             "linear layers; input embedding and output head kept 16-bit. Variant **A** keeps the span channel "
             "(dictionary, relations, composer, projector, gate) and the P1 context in FP16; variant **B** quantizes them "
             "too (same scheme, simulated RTN). `ref` is the run as trained (bf16 autocast), evaluated on the run's own "
             "windows. Gains are loss differences `condition − reference` (negative = condition better), token-weighted "
             "and pooled over common seeds; `Δgain = gain_q − gain_bf16` (positive = the advantage shrinks under "
             "quantization; `*` = Holm-adjusted p < 0.05 within the stratum); retained = `gain_q / gain_bf16`. CIs are "
             f"95% cluster bootstraps over evaluation windows ({summary['resamples']} resamples).", "",
             "## Runs", "", "| Run | Condition | Seed | Windows | ref vs run windows | INT4 group | Linear quantized / skipped |",
             "|---|---|---:|---:|---|---:|---|"]
    for record in sorted(records, key=lambda r: (condition_order(r["condition"]), r["seed"], r["id"])):
        check = record.get("ref_check") or {}
        status = ("no run windows" if not check.get("available") else "windows differ" if not check.get("paired") else
                  f"max |Δ| {check['max_abs_window_sum_diff']:.2g}" + ("" if check.get("counts_equal") else " (counts differ)"))
        int4 = next((v for k, v in record["variants"].items() if k.startswith("int4")), None)
        quantized = next((v for k, v in record["variants"].items() if k != "ref"), None)
        layers = "n/a" if quantized is None else f"{quantized['linear_quantized']} / {len(quantized['linear_skipped'])}"
        lines.append(f"| `{record['id']}` | {record['condition']} | {record['seed']} | {record['windows']} | {status} | "
                     f"{int4['group_size'] if int4 else 'n/a'} | {layers} |")
    for label, cohort in summary["cohorts"].items():
        variants = cohort["quantized_variants"]
        lines += ["", f"## {label}", "", f"Baseline **{cohort['baseline']}**; references {', '.join(cohort['references'])}. "
                  "Seeds: " + "; ".join(f"{c} {s}" for c, s in cohort["grid"].items()) + ".", "",
                  "### Perplexity under quantization", "",
                  "| Condition | PPL ref | " + " | ".join(f"{v} PPL (Δloss [95% CI])" for v in variants) + " |",
                  "|---|---:|" + "---|" * len(variants)]
        for condition, entry in cohort["conditions"].items():
            ref = entry["variants"]["ref"]
            cells = []
            for v in variants:
                row = entry["variants"].get(v)
                cells.append("n/a" if row is None else
                             f"{row['ppl']:.3f} ({row['loss_change']:+.4f} [{row['loss_change_ci'][0]:+.4f}, {row['loss_change_ci'][1]:+.4f}])")
            lines.append(f"| {condition} | {ref['ppl']:.3f} | " + " | ".join(cells) + " |")
        lines += ["", "### Bytes (MiB, mean over seeds)", "",
                  "Nominal = quantized linear weights at their packed size (codes + FP16 scales/zeros); measured = torchao's "
                  "layout (tile-packed INT4 pads the input width to a multiple of 1024).", "",
                  "| Condition | Variant | Total nominal | Total measured | Embedding | Head | Linear nominal | Linear measured | "
                  "LoRA | Other | Channel | Channel schedule |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for condition, entry in cohort["conditions"].items():
            for variant in ["ref", *variants]:
                b = entry["variants"][variant]["bytes"]
                lines.append(f"| {condition} | {variant} | " + " | ".join(_mib(b.get(k)) for k in (
                    "total_nominal", "total", "embedding", "head", "linear_nominal", "linear", "lora", "other", "channel",
                    "channel_schedule")) + " |")
        for reference, by_condition in cohort["retention"].items():
            lines += ["", f"### Retained gain over {reference}", "",
                      "| Stratum | Condition | Variant | gain bf16 [95% CI] | gain quantized [95% CI] | Δgain [95% CI] | retained [95% CI] |",
                      "|---|---|---|---|---|---|---|"]
            for stratum in [s for s in KEY_STRATA if s in cohort["strata"]]:
                for condition, by_variant in by_condition.items():
                    for variant in variants:
                        result = by_variant.get(variant, {}).get(stratum)
                        if result is None:
                            continue
                        retained = result["retained"]
                        kept = ("n/a" if retained["mean"] is None or not math.isfinite(retained["mean"]) else
                                f"{retained['mean']:.2f}" + (f" [{retained['ci_low']:.2f}, {retained['ci_high']:.2f}]"
                                                              if retained["ci_low"] is not None else ""))
                        lines.append(f"| {stratum} | {condition} | {variant} | {_ci(result['gain_bf16'])} | "
                                     f"{_ci(result['gain_quantized'])} | {_ci(result['gain_change'])} | {kept} |")
        checks = [c for c in cohort["ref_checks"].values() if c and c.get("paired")]
        if checks:
            worst = max(c["max_abs_window_sum_diff"] or 0.0 for c in checks)
            lines += ["", f"`ref` reproduces the runs' own final per-window losses to max |Δ window sum| = {worst:.3g} "
                          f"({len(checks)} runs with saved windows)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI


def run(runs: Sequence[Path], output: Path, *, bits: Sequence[int] = (8, 4), variants: Sequence[str] = ("A", "B"),
        group_size: str | int = "auto", baseline: str = "C0", references: Sequence[str] = ("C2",), resamples: int = 10_000,
        seed: int = 0, windows: int | None = None, eval_batch: int | None = None, device: str = "cuda",
        resume: bool = False, report_only: bool = False, title: str = "D4.3 post-training quantization (E4.4)",
        eval_corpus: Path | None = None) -> dict[str, Any]:
    config = {"runs": [str(p) for p in runs], "bits": list(bits), "variants": list(variants), "group_size": group_size,
              "baseline": baseline, "references": list(references), "resamples": resamples, "seed": seed,
              "windows": windows, "eval_batch": eval_batch, "device": device, "title": title}
    if eval_corpus is not None:                      # opt-in: recorded only when used
        config["eval_corpus"] = str(eval_corpus)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    if 4 in bits and target.type != "cuda" and not report_only:
        raise RuntimeError("INT4 (tile-packed) needs CUDA; use --bits 8 on CPU")
    if report_only or resume:
        output.mkdir(parents=True, exist_ok=True)
        git_at_start = git_state()
    else:
        git_at_start = prepare_output_dir(output)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if not report_only:
        for path in find_runs(runs, exclude=output):
            out_dir = output / "runs" / run_id(path)
            if resume and (out_dir / "quant.json").exists():
                continue
            record = evaluate_run(path, out_dir, bits=bits, variants=variants, group_size=group_size, device=target,
                                  windows=windows, eval_batch=eval_batch, eval_corpus_path=eval_corpus)
            print(json.dumps({"run": record["id"], **{v: {"ppl": round(r["ppl"], 3), "seconds": r["seconds"]}
                                                      for v, r in record["variants"].items()}}), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*/quant.json"))]
    if not records:
        raise FileNotFoundError(f"no runs (final.pt + metrics.jsonl) under {', '.join(map(str, runs))}")
    summary = analyze(records, output, baseline=baseline, references=references, resamples=resamples, seed=seed)
    (output / "quant.json").write_text(json.dumps({"runs": records, **summary}, indent=2, default=_json_default) + "\n")
    (output / "report.md").write_text(render(summary, title=title, records=records))
    write_run_metadata(output, config, git_at_start=git_at_start, device=target, runs=len(records),
                       cohorts=list(summary["cohorts"]))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders or roots searched for final.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4], choices=[8, 4])
    parser.add_argument("--variants", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--group-size", default="auto", help="INT4 group size, or auto (largest of 128/64/32 dividing every layer)")
    parser.add_argument("--baseline", default="C0"); parser.add_argument("--references", nargs="*", default=["C2"])
    parser.add_argument("--resamples", type=int, default=10_000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", type=int, default=None, help="override the runs' evaluation window count (breaks pairing with the runs' own windows)")
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="reuse the output folder; skip runs already evaluated")
    parser.add_argument("--report-only", action="store_true", help="re-analyse the evaluated runs in --output")
    parser.add_argument("--title", default="D4.3 post-training quantization (E4.4)")
    parser.add_argument("--eval-corpus", type=Path, default=None,
                        help="evaluate every run on this corpus instead of its data.eval (e.g. a track's eval-general)")
    args = parser.parse_args(argv)
    group_size: str | int = args.group_size if args.group_size == "auto" else int(args.group_size)
    summary = run(args.runs, args.output, bits=args.bits, variants=args.variants, group_size=group_size, baseline=args.baseline,
                  references=args.references, resamples=args.resamples, seed=args.seed, windows=args.windows,
                  eval_batch=args.eval_batch, device=args.device, resume=args.resume, report_only=args.report_only,
                  title=args.title, eval_corpus=args.eval_corpus)
    print(json.dumps({label: {c: {v: round(row["ppl"], 3) for v, row in e["variants"].items()} for c, e in cohort["conditions"].items()}
                      for label, cohort in summary["cohorts"].items()}))


if __name__ == "__main__":
    main()
