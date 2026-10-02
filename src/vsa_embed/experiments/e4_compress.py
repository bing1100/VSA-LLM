"""E4.5 (E4c): equal-bytes compression of the embedding rows of linked single-token concepts (H-B).

    python -m vsa_embed.experiments.e4_compress --config <yaml> --output <dir>

For a trained run (`run`: a folder with `final.pt`, typically a 50M from-scratch model), the
replaced rows `R` are the input-embedding rows of tokens that the linker links as a whole concept
(ℓ = 1 spans: `start == end == inject`, `length == 1`). They are read from the spans of an
all-lengths linked corpus (`rows.link_corpus`, default the C3 `train-l1` slice next to the run's
training corpus; spans with `start < rows.link_tokens`); each token takes its majority entry
(conflicting tokens are counted). A token joins only if at least `rows.min_link_rate` of its
occurrences are such links (default 0.5): stray matches would otherwise pull very common tokens into
`R` (on C3, ',' is linked once in ~10^5 occurrences). Held-out entries are excluded unless
`rows.include_heldout`.

Each point replaces the rows `R` (input side only; the output head, tied or not, keeps the trained
full-precision table; all other rows are unchanged) by a compressed table fitted to the trained
rows by reconstruction (`evaluation.table_compression`), optionally followed by a short LM finetune
of that table alone (`finetune.tokens` ≤ 10M; the transformer is frozen; quantized storage uses
straight-through rounding). Points:

- `composed` (ours, formulation §4.4): `P c_e + b + δ` per spec — dictionary `dimension`, `delta_bits`
  (0 = no residual), `dictionary_bits`, `operator`, `composition`, `init` (`random`, or `run`: the
  run's own channel composer, which must be a compose channel of the same dimension), and
  `fit_dictionary`. Their byte counts are the budgets.
- `baselines` — every family in `table_compression.METHODS` other than `composed`, sized by
  `for_budget` to each composed point's bytes (the configuration with the most bytes ≤ the budget):
  `int` (group-wise RTN table, mixed precision filling the budget), `albert`, `qr`, `hashing`,
  `hash_embedding`, `tt`. Settings per family under `baselines.<family>` (e.g. `bits`, `operation`).
- `points` — extra fixed configurations for the curves (e.g. the INT2/INT3/INT4/INT8 tables).

Evaluation uses the run's evaluation windows and strata (`training.lm.stratum_masks`) plus the
input-row strata of this experiment: per target, the token *before* it (whose row is replaced)
is in `R` (`R_input`), split by its training frequency per million tokens (`rows.frequency_corpus`,
first `rows.frequency_tokens` tokens; `R_input_rare` < `bins_per_million[0]` ≤ `R_input_mid` <
`bins_per_million[1]` ≤ `R_input_frequent`), or not (`not_R_input`).

Outputs (run-folder contract: `resolved_config.yaml`, `manifest.json`): `compress.json` (rows,
every point's settings, exact bytes per stored array, reconstruction error, stratified losses,
timing; Δ vs the dense model and the equal-bytes comparisons), `windows.npz` (per-window loss sums
`sum_<point>` and the shared target counts `count`, strata × windows), `rows.npz` (token ids,
entries, frequencies), `report.md` and `figures/loss-vs-bytes.png`. Equal-bytes comparisons are
`baseline − composed` per stratum (positive = composed better) with a cluster bootstrap over
evaluation windows and Holm correction over all comparisons within a stratum.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..data.corpus import TokenCorpus, collate_windows, eval_windows, sample_batch
from ..evaluation.table_compression import (METHODS, ComposedTable, CompressedTable, ReplacedRowsEmbedding,
                                            entry_subschedule, reconstruction_stats)
from ..provenance import prepare_output_dir, write_run_metadata
from ..statistics import holm_adjust, paired_ratio_bootstrap
from ..training.lm import load_final, resolve_config, stratum_masks
from .e4_report import _PALETTE, _BASELINE_INK, _axes_style, _json_default

DEFAULTS: dict[str, Any] = {
    "run": None,
    "device": "cuda",
    "seed": 0,
    "rows": {"link_corpus": None, "link_tokens": 50_000_000, "include_heldout": False, "min_link_rate": 0.5, "min_links": 1,
             "frequency_corpus": None,
             "frequency_tokens": 300_000_000, "bins_per_million": [5.0, 100.0]},
    "eval": {"windows": None, "batch": None},
    "group_size": 64,
    "fit": {"steps": 1000, "lr": 0.003, "weighting": "uniform", "batch_rows": None},
    "finetune": {"tokens": 0, "lr": 1.0e-3, "micro_batch": 8, "seed": 4321, "methods": "all"},
    "composed": [{"dimension": 64, "delta_bits": 0}, {"dimension": 64, "delta_bits": 2}, {"dimension": 64, "delta_bits": 4}],
    "baselines": {"int": {"mixed": True}, "albert": {}, "qr": {}, "hashing": {}, "hash_embedding": {}, "tt": {}},
    "points": [{"method": "int", "bits": 2}, {"method": "int", "bits": 3}, {"method": "int", "bits": 4}, {"method": "int", "bits": 8}],
    "resamples": 10_000,
    "figures": True,
}
# Methods whose initialization is already the reconstruction optimum (or not gradient-fitted).
CLOSED_FORM = {"int", "albert", "hashing"}
R_STRATA = ("R_input", "R_input_rare", "R_input_mid", "R_input_frequent", "not_R_input")
REPORT_STRATA = ("all", "R_input", "R_input_rare", "R_input_mid", "R_input_frequent", "not_R_input", "after")
MAX_FINETUNE_TOKENS = 10_000_000
FAMILY_ORDER = ("composed", "int", "albert", "qr", "hashing", "hash_embedding", "tt")


def resolve(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(DEFAULTS)
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict) and key != "baselines":
            resolved[key].update(copy.deepcopy(value))
        else:
            resolved[key] = copy.deepcopy(value)
    if not resolved["run"]:
        raise ValueError("config needs `run` (a trained run folder)")
    if int(resolved["finetune"]["tokens"]) > MAX_FINETUNE_TOKENS:
        raise ValueError(f"finetune.tokens is capped at {MAX_FINETUNE_TOKENS:,} (E4.5 protocol)")
    return resolved


# ---------------------------------------------------------------- rows and frequencies


def single_token_rows(corpus: Path, *, max_tokens: int, heldout: set[int], include_heldout: bool = False,
                      min_link_rate: float = 0.5, min_links: int = 1) -> dict[str, Any]:
    """Tokens linked as a whole concept (ℓ = 1 spans) and their majority entry.

    A token joins only if at least `min_link_rate` of its occurrences (in the first `max_tokens`
    tokens) are ℓ = 1 links and it is linked at least `min_links` times: the linker treats it as a
    concept, not as a stray match (on C3, ',' is linked once in ~10^5 occurrences)."""
    with np.load(corpus / "spans.npz") as spans:
        length = spans["length"]
        keep = length == 1
        start = spans["start"][keep].astype(np.int64)
        entry = spans["entry"][keep].astype(np.int64)
    within = start < max_tokens
    start, entry = start[within], entry[within]
    tokens = TokenCorpus.open(corpus).tokens
    token = np.asarray(tokens[start], dtype=np.int64)
    if not include_heldout and heldout:
        mask = ~np.isin(entry, np.fromiter(heldout, dtype=np.int64))
        token, entry = token[mask], entry[mask]
    keys, counts = np.unique(token * (1 << 32) + entry, return_counts=True)
    pairs = np.stack([keys >> 32, keys & ((1 << 32) - 1)], 1)
    order = np.lexsort((-counts, pairs[:, 0]))            # by token, most frequent entry first
    pairs, counts = pairs[order], counts[order]
    first = np.r_[True, pairs[1:, 0] != pairs[:-1, 0]]
    token_ids, entries = pairs[first, 0], pairs[first, 1]
    links = np.add.reduceat(counts, np.flatnonzero(first))
    entries_per_token = np.bincount(np.searchsorted(token_ids, pairs[:, 0]), minlength=len(token_ids))
    occurrences, _ = token_frequency(corpus, max_tokens=max_tokens, vocab_size=int(token_ids.max()) + 1 if len(token_ids) else 1)
    rate = links / np.maximum(occurrences[token_ids], 1)
    keep = (rate >= min_link_rate) & (links >= min_links)
    return {"token_ids": token_ids[keep], "entries": entries[keep], "links": links[keep], "link_rate": rate[keep],
            "conflicting_tokens": int((entries_per_token[keep] > 1).sum()), "dropped_tokens": int((~keep).sum()),
            "dropped_links": int(links[~keep].sum()), "spans": int(token.size)}


def token_frequency(corpus: Path, *, max_tokens: int, vocab_size: int, chunk: int = 50_000_000) -> tuple[np.ndarray, int]:
    tokens = TokenCorpus.open(corpus).tokens
    total = min(int(max_tokens), len(tokens))
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, total, chunk):
        counts += np.bincount(np.asarray(tokens[start:min(total, start + chunk)]), minlength=vocab_size)[:vocab_size]
    return counts, total


def token_classes(token_ids: np.ndarray, per_million: np.ndarray, bins: list[float], vocab_size: int) -> np.ndarray:
    """−1 for tokens outside R, else 0 / 1 / 2 = rare / mid / frequent."""
    classes = np.full(vocab_size, -1, dtype=np.int8)
    classes[token_ids] = np.digitize(per_million[token_ids], bins).astype(np.int8)
    return classes


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_strata(model, corpus: TokenCorpus, starts: list[int], *, seq_len: int, batch: int, min_subtokens: int,
                    frequency: np.ndarray | None, heldout: set[int], classes: np.ndarray, device: torch.device
                    ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-window loss sums and target counts (strata × windows): the trainer's strata plus R strata."""
    model.eval()
    sums: dict[str, list[np.ndarray]] = {}
    counts: dict[str, list[np.ndarray]] = {}
    lookup = torch.from_numpy(classes.astype(np.int64))
    for i in range(0, len(starts), batch):
        windows = [corpus.window(s, seq_len, min_subtokens=min_subtokens) for s in starts[i:i + batch]]
        ids, spans = collate_windows(windows)
        ids_d = ids.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            per_token = model(ids_d, spans={k: v.to(device) for k, v in spans.items()} if model.channel else None,
                              labels=ids_d, reduction="none")["loss"].float().cpu()
        masks = stratum_masks(ids, spans, frequency, heldout)
        cls = lookup[ids[:, :-1]]
        masks.update(R_input=cls >= 0, R_input_rare=cls == 0, R_input_mid=cls == 1, R_input_frequent=cls == 2,
                     not_R_input=cls < 0)
        for name, mask in masks.items():
            sums.setdefault(name, []).append(per_token.double().masked_fill(~mask, 0.0).sum(1).numpy())
            counts.setdefault(name, []).append(mask.sum(1).numpy().astype(np.int32))
    strata = list(sums)
    return strata, np.stack([np.concatenate(sums[s]) for s in strata]), np.stack([np.concatenate(counts[s]) for s in strata])


def finetune_table(model, table: CompressedTable, corpus: TokenCorpus, *, tokens: int, lr: float, micro_batch: int,
                   seq_len: int, seed: int, min_subtokens: int, entry_mask: np.ndarray | None, device: torch.device) -> dict[str, Any]:
    """LM finetune of the compressed table only (transformer, channel and other rows frozen)."""
    frozen = [p for p in model.parameters() if p.requires_grad and not any(p is q for q in table.parameters())]
    for p in frozen:
        p.requires_grad_(False)
    trainable = [p for p in table.parameters() if p.requires_grad]
    steps = max(1, tokens // (micro_batch * seq_len))
    optimizer = torch.optim.Adam(trainable, lr=lr)
    losses = []
    model.eval()                     # no dropout: only the table changes
    try:
        for step in range(steps):
            for group in optimizer.param_groups:
                group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            ids, spans = sample_batch(corpus, seed=seed, step=step, micro_step=0, batch=micro_batch, length=seq_len,
                                      min_subtokens=min_subtokens, entry_mask=entry_mask)
            ids = ids.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(ids, spans={k: v.to(device) for k, v in spans.items()} if model.channel else None, labels=ids)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        for p in frozen:
            p.requires_grad_(True)
    return {"steps": steps, "tokens": steps * micro_batch * seq_len, "first_loss": losses[0], "last_loss": float(np.mean(losses[-10:]))}


# ---------------------------------------------------------------- building tables


def composed_name(spec: dict[str, Any], table: ComposedTable) -> str:
    if spec.get("name"):
        return str(spec["name"])
    name = f"composed-d{table.dimension}-delta{table.delta_bits}"
    if table.dictionary_bits != 16:
        name += f"-dict{table.dictionary_bits}"
    if table.composition != "bundle":
        name += f"-{table.composition}"
    if spec.get("init", "random") == "run":
        name += "-run" + ("" if table.fit_dictionary else "-frozen")
    return name


def point_name(spec: dict[str, Any]) -> str:
    if spec.get("name"):
        return str(spec["name"])
    settings = "-".join(f"{k}{v}" for k, v in spec.items() if k not in {"method", "name", "steps", "lr"})
    return f"{spec['method']}-{settings}" if settings else spec["method"]


def build_composed(spec: dict[str, Any], *, n: int, d: int, rows: dict[str, Any], ontology: dict[str, Any], model,
                   group_size: int) -> ComposedTable:
    unique, local = np.unique(rows["entries"], return_inverse=True)
    schedule, atomics = entry_subschedule(ontology["offsets"], ontology["relations"], ontology["fillers"], torch.from_numpy(unique))
    init = spec.get("init", "random")
    channel = getattr(model, "channel", None)
    source = channel.composer if init == "run" and channel is not None else None
    if init == "run" and source is None:
        raise ValueError("init: run needs a run with a compose channel")
    table = ComposedTable(n, d, entries=torch.from_numpy(local), schedule=schedule, atomic_count=atomics.numel(),
                          relation_count=int(ontology["relation_count"]),
                          dimension=int(spec.get("dimension", source.atomics.shape[1] if source is not None else 64)),
                          operator=spec.get("operator", source.operator if source is not None else "hrr"),
                          composition=spec.get("composition", source.mode if source is not None else "bundle"),
                          key_dimension=int(spec.get("key_dimension", source.key_dimension if source is not None else 8)),
                          delta_bits=int(spec.get("delta_bits", 0)), group_size=group_size,
                          dictionary_bits=int(spec.get("dictionary_bits", 16)), fit_dictionary=bool(spec.get("fit_dictionary", True)))
    if source is not None:
        table.load_dictionary(source, atomics)
    return table


def build_baseline(method: str, budget: int | None, settings: dict[str, Any], *, n: int, d: int, keys: torch.Tensor,
                   priority: torch.Tensor, group_size: int) -> CompressedTable | None:
    cls = METHODS[method]
    settings = {k: v for k, v in settings.items() if k not in {"steps", "lr", "name", "method"}}
    common = {"keys": keys} if method in {"hashing", "hash_embedding"} else {}
    if method == "int":
        common["priority"] = priority
    if budget is not None:
        return cls.for_budget(budget, n, d, group_size=group_size, **common, **settings)
    if method == "int":
        settings.pop("mixed", None)
        return cls(n, d, group_size=settings.pop("group_size", group_size), **common, **settings)
    return cls(n, d, **common, **settings)


# ---------------------------------------------------------------- analysis


def compare(sums: dict[str, np.ndarray], count: np.ndarray, strata: list[str], a: str, b: str, *, resamples: int,
            seed: int) -> dict[str, Any]:
    """`a − b` per stratum (token-weighted loss difference; cluster bootstrap over windows)."""
    result = {}
    for i, stratum in enumerate(strata):
        n = count[i].astype(np.float64)
        if n.sum() == 0:
            continue
        boot = paired_ratio_bootstrap(sums[a][i] - sums[b][i], n, sums[b][i], resamples=resamples, seed=seed)
        result[stratum] = {k: boot[k] for k in ("mean", "ci_low", "ci_high", "p_value", "relative")}
    return result


def analyze(points: dict[str, dict[str, Any]], sums: dict[str, np.ndarray], count: np.ndarray, strata: list[str], *,
            resamples: int, seed: int) -> dict[str, Any]:
    versus_dense = {name: compare(sums, count, strata, name, "dense", resamples=resamples, seed=seed)
                    for name in points if name != "dense"}
    equal_bytes: dict[str, dict[str, Any]] = {}
    for name, point in points.items():
        if point.get("budget_of") in sums and name in sums:
            reference = point["budget_of"]
            equal_bytes.setdefault(reference, {})[name] = {
                "method": point["method"], "bytes": point["bytes"], "budget": point["budget"],
                "difference": compare(sums, count, strata, name, reference, resamples=resamples, seed=seed)}
    for stratum in strata:     # Holm over all equal-bytes comparisons within the stratum
        family = [c["difference"][stratum] for by_ref in equal_bytes.values() for c in by_ref.values() if stratum in c["difference"]]
        for test, adjusted in zip(family, holm_adjust([t["p_value"] for t in family])):
            test.update(holm_p=adjusted, significant=adjusted < 0.05)
    return {"versus_dense": versus_dense, "equal_bytes": equal_bytes}


# ---------------------------------------------------------------- rendering


def _line(name: str, point: dict[str, Any]) -> str:
    """Curve a point belongs to: its method, and for composed points also its dictionary
    (dimension, bits, source) — δ bits vary along the curve."""
    return re.sub(r"-delta\d+", "", name) if point["method"] == "composed" else point["method"]


def plot(points: dict[str, dict[str, Any]], out: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strata = [s for s in ("all", "R_input", "R_input_rare") if s in points["dense"]["strata"]]
    fig, axes = plt.subplots(1, len(strata), figsize=(4.4 * len(strata), 3.6), squeeze=False)
    shown = {n: p for n, p in points.items() if n != "dense" and not n.endswith("+ft")}
    families = [f for f in FAMILY_ORDER if any(p["method"] == f for p in shown.values())]
    composed_lines = sorted({_line(n, p) for n, p in shown.items() if p["method"] == "composed"})
    for ax, stratum in zip(axes.flat, strata):
        dense = points["dense"]["strata"][stratum]["loss"]
        ax.axhline(dense, color=_BASELINE_INK, linewidth=1.0, linestyle="--")
        ax.annotate("dense", (0.0, dense), xycoords=("axes fraction", "data"), xytext=(3, 3), textcoords="offset points",
                    ha="left", fontsize=7, color=_BASELINE_INK)
        for slot, family in enumerate(families):
            lines = composed_lines if family == "composed" else [family]
            for variant, line in enumerate(lines):
                members = sorted((p["bytes"], p["strata"][stratum]["loss"]) for n, p in shown.items()
                                 if _line(n, p) == line and stratum in p["strata"])
                if not members:
                    continue
                x, y = zip(*members)
                ax.plot(x, y, color=_PALETTE[slot % len(_PALETTE)], linewidth=2.0 if family == "composed" else 1.4,
                        linestyle=("-", "--", ":", "-.")[variant % 4], marker="o",
                        markersize=4.5 if family == "composed" else 3.5, label=line)
        ax.set_xscale("log", base=2)
        ax.set_title(stratum, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("bytes of the replaced rows", fontsize=8, color="#52514e")
        ax.set_ylabel("loss (nats/token)", fontsize=8, color="#52514e")
        _axes_style(ax, token_axis=False)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 5), frameon=False, fontsize=8)
    fig.suptitle("Loss vs bytes of the linked single-token rows (reconstruction fit; composed: one curve per dictionary, "
                 "δ bits along it)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 + 0.04 * math.ceil(len(names) / 5), 1, 0.94))
    (out / "figures").mkdir(exist_ok=True)
    name = "figures/loss-vs-bytes.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def _kib(value: float) -> str:
    return f"{value / 1024:,.1f}"


def _delta(v: dict[str, Any] | None) -> str:
    if not v:
        return "–"
    return f"{v['mean']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" + ("*" if v.get("significant") else "")


def render(summary: dict[str, Any], figure: str | None) -> str:
    points, rows = summary["points"], summary["rows"]
    strata = [s for s in REPORT_STRATA if s in points["dense"]["strata"]]
    lines = ["# E4.5 table compression at equal bytes", "",
             f"Run `{summary['run']}`. Replaced rows R: **{rows['count']:,}** tokens linked as single-token concepts "
             f"({rows['entries']:,} entries; {rows['conflicting_tokens']:,} tokens with conflicting entries, majority kept; "
             f"held-out entries {'included' if rows['include_heldout'] else 'excluded'}; tokens linked in < "
             f"{rows['min_link_rate']:.0%} of their occurrences dropped: {rows['dropped_tokens']:,} tokens, "
             f"{rows['dropped_links']:,} links), width {rows['width']}; dense FP16 "
             f"bytes of R: {_kib(rows['dense_fp16_bytes'])} KiB. Only the input rows R are replaced; the output head and every "
             "other row keep the trained table. Frequency bins (per million training tokens): rare < "
             f"{rows['bins_per_million'][0]:g} ≤ mid < {rows['bins_per_million'][1]:g} ≤ frequent; R types per bin: "
             + ", ".join(f"{k} {v:,}" for k, v in rows["types_per_bin"].items()) + ".", "",
             "Equal-bytes rule: each baseline family is sized to the most bytes ≤ the composed point's bytes. Differences are "
             "`baseline − composed` (positive = composed better), token-weighted, 95% cluster bootstrap over evaluation windows; "
             "`*` = Holm-adjusted p < 0.05 over all equal-bytes comparisons within the stratum.", ""]
    if figure:
        lines += [f"![Loss vs bytes]({figure})", ""]
    lines += ["## Points", "", "Reconstruction error relative to the rows' energy / to their energy around the mean row "
              "(`+ft` points: before the finetune).", "",
              "| Point | Bytes (KiB) | Rel. recon. error | " + " | ".join(strata) + " | Fit s |",
              "|---|---:|---:|" + "---:|" * len(strata) + "---:|"]
    for name, p in sorted(points.items(), key=lambda kv: (kv[0] != "dense", FAMILY_ORDER.index(kv[1]["method"]) if kv[1]["method"] in FAMILY_ORDER else 99, kv[1]["bytes"])):
        recon = p.get("reconstruction", {})
        error = "–" if "relative_error" not in recon else f"{recon['relative_error']:.3f} / {recon['relative_error_centered']:.3f}"
        lines.append(f"| `{name}` | {_kib(p['bytes'])} | {error} | "
                     + " | ".join(f"{p['strata'][s]['loss']:.4f}" if s in p["strata"] else "–" for s in strata)
                     + f" | {p.get('seconds', {}).get('fit', 0):.1f} |")
    for reference, by_point in summary["analysis"]["equal_bytes"].items():
        ref_point = points[reference]
        lines += ["", f"## Equal bytes: `{reference}` ({_kib(ref_point['bytes'])} KiB)", "",
                  "| Baseline | Bytes (KiB) | " + " | ".join(strata) + " |", "|---|---:|" + "---|" * len(strata)]
        for name, comparison in sorted(by_point.items(), key=lambda kv: FAMILY_ORDER.index(kv[1]["method"]) if kv[1]["method"] in FAMILY_ORDER else 99):
            lines.append(f"| `{name}` | {_kib(comparison['bytes'])} | " + " | ".join(_delta(comparison["difference"].get(s)) for s in strata) + " |")
        missing = summary["infeasible"].get(reference, [])
        if missing:
            lines += ["", "No configuration fits this budget: " + ", ".join(missing) + "."]
    lines += ["", "## Δ vs the dense model (point − dense)", "", "| Point | " + " | ".join(strata) + " |", "|---|" + "---|" * len(strata)]
    for name, by_stratum in summary["analysis"]["versus_dense"].items():
        lines.append(f"| `{name}` | " + " | ".join(_delta(by_stratum.get(s)) for s in strata) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- driver


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    config = resolve(config)
    git_at_start = prepare_output_dir(output)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    run_dir = Path(config["run"]).expanduser()
    run_config = resolve_config(torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")["config"])
    model = load_final(run_dir / "final.pt", device)
    embedding = model.model.get_input_embeddings()
    vocab_size, width = embedding.weight.shape
    ontology = torch.load(run_config["data"]["ontology"], weights_only=False)
    heldout = set(ontology["heldout_entries"])
    frequency = np.asarray(ontology["train_frequency"])
    data_root = Path(run_config["data"]["train"]).parent
    link_corpus = Path(config["rows"]["link_corpus"] or data_root / "train-l1").expanduser()
    frequency_corpus = Path(config["rows"]["frequency_corpus"] or link_corpus).expanduser()

    # 1. replaced rows and their frequency classes
    rows = single_token_rows(link_corpus, max_tokens=int(config["rows"]["link_tokens"]), heldout=heldout,
                             include_heldout=bool(config["rows"]["include_heldout"]),
                             min_link_rate=float(config["rows"]["min_link_rate"]), min_links=int(config["rows"]["min_links"]))
    counts, counted = token_frequency(frequency_corpus, max_tokens=int(config["rows"]["frequency_tokens"]), vocab_size=vocab_size)
    per_million = counts / max(1, counted) * 1e6
    bins = [float(b) for b in config["rows"]["bins_per_million"]]
    classes = token_classes(rows["token_ids"], per_million, bins, vocab_size)
    token_ids = torch.from_numpy(rows["token_ids"])
    n = int(token_ids.numel())
    target = embedding.weight.detach()[token_ids.to(embedding.weight.device)].float()
    np.savez_compressed(output / "rows.npz", token_ids=rows["token_ids"], entries=rows["entries"],
                        links=rows["links"], link_rate=rows["link_rate"], frequency=counts[rows["token_ids"]],
                        classes=classes[rows["token_ids"]])
    priority = torch.from_numpy(np.argsort(-counts[rows["token_ids"]], kind="stable"))     # most frequent rows first
    weights = None
    if config["fit"]["weighting"] != "uniform":
        raw = torch.from_numpy(counts[rows["token_ids"]].astype(np.float64)).float() + 1.0
        weights = raw.sqrt() if config["fit"]["weighting"] == "sqrt_frequency" else raw

    # 2. evaluation setup
    eval_corpus = TokenCorpus.open(Path(run_config["data"]["eval"]))
    seq_len = int(run_config["model"]["seq_len"])
    starts = eval_windows(eval_corpus, count=int(config["eval"]["windows"] or run_config["eval"]["windows"]), length=seq_len)
    eval_kwargs = dict(seq_len=seq_len, batch=int(config["eval"]["batch"] or run_config["eval"]["batch"]),
                       min_subtokens=int(run_config["data"]["min_subtokens"]), frequency=frequency, heldout=heldout,
                       classes=classes, device=device)
    finetune = config["finetune"]
    train_corpus = TokenCorpus.open(Path(run_config["data"]["train"])) if int(finetune["tokens"]) else None
    entry_mask = np.ones(int(ontology["entry_count"]), dtype=bool)
    entry_mask[list(heldout)] = False
    group_size = int(config["group_size"])

    points: dict[str, dict[str, Any]] = {}
    sums: dict[str, np.ndarray] = {}
    infeasible: dict[str, list[str]] = {}
    started = time.monotonic()
    strata, dense_sums, count = evaluate_strata(model, eval_corpus, starts, **eval_kwargs)
    sums["dense"] = dense_sums
    points["dense"] = {"method": "dense", "bytes": 2 * n * width, "settings": {"bits": 16}, "storage": {"rows": 2 * n * width},
                       "strata": _strata_losses(strata, dense_sums, count), "seconds": {"eval": round(time.monotonic() - started, 2)}}

    def evaluate_point(name: str, table: CompressedTable, *, method: str, steps: int, lr: float,
                       budget: int | None = None, budget_of: str | None = None) -> None:
        if name in points:
            raise ValueError(f"duplicate point name {name!r}; give the spec a `name`")
        table.to(device)
        t0 = time.monotonic()
        reconstruction = table.fit(target, steps=steps, lr=lr, weights=weights, batch_rows=config["fit"]["batch_rows"],
                                   seed=int(config["seed"]))
        fit_seconds = time.monotonic() - t0
        replaced = ReplacedRowsEmbedding(embedding, token_ids, table).to(device)
        model.model.set_input_embeddings(replaced)
        try:
            variants = [(name, None)]
            wanted = finetune["methods"]
            if int(finetune["tokens"]) and (wanted == "all" or method in wanted):
                variants.append((f"{name}+ft", "finetune"))
            for label, mode in variants:
                record: dict[str, Any] = {"method": method, "settings": table.settings(), "bytes": table.nbytes,
                                          "storage": table.storage(), "reconstruction": reconstruction,
                                          "seconds": {"fit": round(fit_seconds, 2)}}
                if isinstance(table, ComposedTable):
                    record["marginal_bytes"] = table.marginal_bytes()
                if budget is not None:
                    record.update(budget=budget, budget_of=budget_of + ("+ft" if mode else ""))
                if mode == "finetune":
                    t1 = time.monotonic()
                    replaced.clear_cache()
                    record["finetune"] = finetune_table(model, table, train_corpus, tokens=int(finetune["tokens"]), lr=float(finetune["lr"]),
                                                        micro_batch=int(finetune["micro_batch"]), seq_len=seq_len, seed=int(finetune["seed"]),
                                                        min_subtokens=int(run_config["data"]["min_subtokens"]), entry_mask=entry_mask,
                                                        device=device)
                    record["seconds"]["finetune"] = round(time.monotonic() - t1, 2)
                    record["reconstruction_after_finetune"] = reconstruction_stats(table, target)
                t2 = time.monotonic()
                replaced.cache()
                _, point_sums, point_count = evaluate_strata(model, eval_corpus, starts, **eval_kwargs)
                if not np.array_equal(point_count, count):
                    raise RuntimeError(f"{label}: target counts differ from the dense evaluation")
                record["strata"] = _strata_losses(strata, point_sums, count)
                record["seconds"]["eval"] = round(time.monotonic() - t2, 2)
                points[label], sums[label] = record, point_sums
                print(json.dumps({"point": label, "bytes": record["bytes"], "relative_error": reconstruction["relative_error"],
                                  "loss_all": record["strata"]["all"]["loss"], "seconds": record["seconds"]}), flush=True)
        finally:
            model.model.set_input_embeddings(embedding)

    common = dict(n=n, d=width)
    # 3. composed points (they set the budgets)
    for spec in config["composed"]:
        table = build_composed(spec, **common, rows=rows, ontology=ontology, model=model, group_size=group_size)
        steps = int(spec.get("steps", config["fit"]["steps"])) if table.fit_dictionary else 0   # frozen: least squares only
        evaluate_point(composed_name(spec, table), table, method="composed", steps=steps, lr=float(spec.get("lr", config["fit"]["lr"])))
    budgets = {name: points[name]["bytes"] for name in list(points) if points[name]["method"] == "composed" and not name.endswith("+ft")}
    # 4. baselines at each budget
    for method, settings in (config["baselines"] or {}).items():
        settings = dict(settings or {})
        steps = 0 if method in CLOSED_FORM else int(settings.get("steps", config["fit"]["steps"]))
        for reference, budget in budgets.items():
            table = build_baseline(method, budget, settings, **common, keys=token_ids, priority=priority, group_size=group_size)
            if table is None:
                infeasible.setdefault(reference, []).append(method)
                continue
            evaluate_point(f"{method}@{reference}", table, method=method, steps=steps,
                           lr=float(settings.get("lr", config["fit"]["lr"])), budget=budget, budget_of=reference)
    # 5. extra fixed points for the curves
    for spec in config["points"] or []:
        spec = dict(spec)
        method = spec["method"]
        table = build_baseline(method, None, spec, **common, keys=token_ids, priority=priority, group_size=group_size)
        steps = 0 if method in CLOSED_FORM else int(spec.get("steps", config["fit"]["steps"]))
        evaluate_point(point_name(spec), table, method=method, steps=steps, lr=float(spec.get("lr", config["fit"]["lr"])))

    resamples = int(config["resamples"])
    analysis = analyze(points, sums, count, strata, resamples=resamples, seed=int(config["seed"]))
    type_bins = {name: int((classes[rows["token_ids"]] == k).sum()) for k, name in enumerate(("rare", "mid", "frequent"))}
    summary = {"run": str(run_dir), "rows": {"count": n, "entries": int(np.unique(rows["entries"]).size), "width": int(width),
                                             "conflicting_tokens": rows["conflicting_tokens"], "linked_spans": rows["spans"],
                                             "min_link_rate": float(config["rows"]["min_link_rate"]),
                                             "dropped_tokens": rows["dropped_tokens"], "dropped_links": rows["dropped_links"],
                                             "include_heldout": bool(config["rows"]["include_heldout"]), "link_corpus": str(link_corpus),
                                             "frequency_corpus": str(frequency_corpus), "frequency_tokens": counted,
                                             "bins_per_million": bins, "types_per_bin": type_bins,
                                             "dense_fp16_bytes": 2 * n * width},
               "strata": strata, "points": points, "infeasible": infeasible, "analysis": analysis, "resamples": resamples,
               "seconds": round(time.monotonic() - started, 1)}
    np.savez_compressed(output / "windows.npz", strata=np.asarray(strata), starts=np.asarray(starts, dtype=np.int64), count=count,
                        **{f"sum_{re.sub(r'[^A-Za-z0-9_.@+-]', '_', k)}": v for k, v in sums.items()})
    figure = plot(points, output) if config.get("figures", True) else None
    (output / "compress.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    (output / "report.md").write_text(render(summary, figure))
    write_run_metadata(output, config, git_at_start=git_at_start, device=device, points=len(points), rows=n,
                       source_run=str(run_dir))
    return summary


def _strata_losses(strata: list[str], sums: np.ndarray, count: np.ndarray) -> dict[str, dict[str, float]]:
    return {s: {"loss": float(sums[i].sum() / count[i].sum()) if count[i].sum() else float("nan"), "tokens": int(count[i].sum())}
            for i, s in enumerate(strata)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=None, help="override the config's run folder")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.run is not None:
        config["run"] = str(args.run)
    summary = run(config, args.output)
    print(json.dumps({name: {"bytes": p["bytes"], **{s: round(p["strata"][s]["loss"], 4) for s in ("all", "R_input") if s in p["strata"]}}
                      for name, p in summary["points"].items()}))


if __name__ == "__main__":
    main()
