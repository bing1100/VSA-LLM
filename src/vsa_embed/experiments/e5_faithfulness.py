"""E5.1 faithfulness of edge attention (D5.1; experiments.md E5.1, proposal H-F).

For every linked span in the run's evaluation windows (the same windows as the trainer's stratified
evaluation), the composer's edge weights are taken as the explanation and edges are erased:

- comprehensiveness (DeYoung et al. 2020): remove the top-`k` / random `k` / bottom-`k` edges of each
  occurrence; `Δloss = loss(ablated) − loss(full)` on the 8 tokens after the span (stratum `after`);
- sufficiency: keep only the top-/random/bottom-`k` edges; `Δloss = loss(kept) − loss(full)`.

`k ∈ {1, 2, 4}` by default; only occurrences with more than `k` edges are ablated (frames with ≤ `k`
edges would lose every edge under every policy) and only their after-tokens are measured, in the
strata `k<k>` (all such spans), `k<k>_heldout` and `k<k>_seen`. All spans of a window are ablated
at once (one forward pass per window and policy); random policies use `draws` independent draws,
averaged. Removing every edge (`remove_all`) gives a zero row, i.e. removes the channel; the run
checks that this equals the channel-off forward on the first batch. Prediction changes are also
reported: KL(full ‖ ablated) of the next-token distribution and the top-1 flip rate on the same
tokens. A static (bundle, C3) composer has uniform weights, so its "top-k" is a random subset by
construction (ties are broken at random) and top ≈ random is the expected reading.

Statistics: token-weighted differences with a cluster bootstrap over evaluation windows
(`statistics.paired_ratio_bootstrap`); contrasts top − random and random − bottom per `k` and
stratum, Holm-adjusted over the run's contrast grid. `--probes` additionally reruns the WP-probe
suite (`--fast` settings) with `remove_{top,random,bottom}<probe-k>` ablations active and compares
each with the full model item by item (`channel_probes.compare_outputs`).

    python -m vsa_embed.experiments.e5_faithfulness --run RUN --output OUT [--windows 1024] [--ks 1 2 4]
        [--draws 2] [--batch 8] [--no-kl] [--save-spans] [--probes wic,wsd --probe-k 2]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..data.corpus import TokenCorpus, collate_windows, eval_windows
from ..statistics import holm_adjust, paired_ratio_bootstrap
from ..training.lm import AFTER_WINDOW
from .e5_common import (EdgePolicy, E5Run, ablation_transform, edge_weight_hook, finish_output, fmt, fmt_ci,
                        forward_tokens, open_run, start_output, status_of, write_json)

ORDERS = ("top", "random", "bottom")


def policies_for(ks: Sequence[int], draws: int) -> list[EdgePolicy]:
    policies = [EdgePolicy("full"), EdgePolicy("remove_all")]
    for k in ks:
        for action in ("remove", "keep"):
            policies += [EdgePolicy(action, "top", k), EdgePolicy(action, "bottom", k)]
            policies += [EdgePolicy(action, "random", k, d) for d in range(draws)]
    return policies


def strata_for(ks: Sequence[int]) -> list[str]:
    return ["after"] + [f"k{k}{suffix}" for k in ks for suffix in ("", "_heldout", "_seen")]


def span_masks(ids_shape: tuple[int, int], spans: dict[str, torch.Tensor], degrees: torch.Tensor, heldout: set[int],
               ks: Sequence[int]) -> dict[str, torch.Tensor]:
    """After-token masks (batch × (T − 1)) per stratum: all spans, and spans with more than `k` edges."""
    batch, length = ids_shape
    shape = (batch, length - 1)
    masks = {name: torch.zeros(shape, dtype=torch.bool) for name in strata_for(ks)}
    for b, e, entry in zip(spans["batch"].tolist(), spans["end"].tolist(), spans["entry"].tolist()):
        lo, hi = e, min(length - 1, e + AFTER_WINDOW)
        if lo >= hi:
            continue
        masks["after"][b, lo:hi] = True
        degree = int(degrees[entry])
        status = "_heldout" if entry in heldout else "_seen"
        for k in ks:
            if degree > k:
                masks[f"k{k}"][b, lo:hi] = True
                masks[f"k{k}{status}"][b, lo:hi] = True
    return masks


@torch.no_grad()
def _log_probs(run: E5Run, vectors: torch.Tensor) -> torch.Tensor:
    """Next-token log-probabilities from final hidden states (rows × d)."""
    head = run.model.model.get_output_embeddings()
    bias = getattr(head, "bias", None)
    logits = F.linear(vectors.float(), head.weight.float(), None if bias is None else bias.float())
    return torch.log_softmax(logits, -1)


@torch.no_grad()
def prediction_change(run: E5Run, full: torch.Tensor, ablated: torch.Tensor, *, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Per position: KL(p_full ‖ p_ablated) and whether the top-1 prediction changes (CPU, float32)."""
    kls, flips = [], []
    for c0 in range(0, full.shape[0], chunk):
        p, q = _log_probs(run, full[c0:c0 + chunk]), _log_probs(run, ablated[c0:c0 + chunk])
        kls.append((p.exp() * (p - q)).sum(-1).clamp_min(0).cpu())
        flips.append((p.argmax(-1) != q.argmax(-1)).float().cpu())
    return (torch.cat(kls), torch.cat(flips)) if kls else (torch.zeros(0), torch.zeros(0))


@torch.no_grad()
def uniform_weight_counts(run: E5Run, ids: torch.Tensor, spans: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(occurrences with > 1 edge whose weights are all equal, occurrences with > 1 edge)."""
    from .e5_common import occurrence_weights, span_contexts
    entries = spans["entry"].to(run.device)
    if not entries.numel():
        return 0, 0
    context = span_contexts(run.model, ids.to(run.device), {k: v.to(run.device) for k, v in spans.items()})
    weights, _, segments = occurrence_weights(run.composer, entries, context)
    count = entries.numel()
    high = weights.new_full((count,), float("-inf")).scatter_reduce(0, segments, weights, "amax")
    low = weights.new_full((count,), float("inf")).scatter_reduce(0, segments, weights, "amin")
    multi = run.composer.schedule.degrees[entries] > 1
    return int((((high - low).abs() < 1e-6) & multi).sum()), int(multi.sum())


def faithfulness(run: E5Run, *, windows: int, batch: int, ks: Sequence[int], draws: int, seed: int,
                 kl: bool = True, save_spans: bool = False, log: Any = print) -> dict[str, Any]:
    """Window-level sums (policies × strata × windows) of after-token losses, KL and top-1 flips."""
    if run.composer is None:
        raise ValueError(f"E5.1 needs a composition channel; {run.path} has channel mode {run.mode!r}")
    config = run.config
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    seq_len, min_sub = int(config["model"]["seq_len"]), run.min_subtokens
    starts = eval_windows(corpus, count=windows, length=seq_len)
    policies, strata = policies_for(ks, draws), strata_for(ks)
    degrees = run.composer.schedule.degrees.cpu()
    heldout = set(run.adapter.heldout_entries)
    frequency = run.adapter.train_frequency
    shape = (len(policies), len(strata), len(starts))
    sums = np.zeros(shape)
    kls, flips = (np.zeros(shape), np.zeros(shape)) if kl else (None, None)
    counts = np.zeros(shape[1:])
    spans_out: dict[str, list[Any]] = {k: [] for k in ("window", "inject", "entry", "degree", "status", "count")}
    span_losses: list[np.ndarray] = []
    checks: dict[str, Any] = {}
    uniform = multi = 0
    started = time.monotonic()
    for b0 in range(0, len(starts), batch):
        chunk = [corpus.window(s, seq_len, min_subtokens=min_sub) for s in starts[b0:b0 + batch]]
        ids, spans = collate_windows(chunk)
        window = slice(b0, b0 + len(chunk))
        masks = span_masks(tuple(ids.shape), spans, degrees, heldout, ks)
        for s, name in enumerate(strata):
            counts[s, window] = masks[name].sum(1).numpy()
        positions = masks["after"].nonzero()
        full_vectors = None
        token_losses: list[torch.Tensor] = []
        for p, policy in enumerate(policies):
            transform = ablation_transform(policy, degrees, seed=seed * 1_000_003 + b0 * 101 + policy.draw)
            with edge_weight_hook(run.composer, transform):
                losses, hidden = forward_tokens(run, ids, spans)
            token_losses.append(losses)
            for s, name in enumerate(strata):
                sums[p, s, window] = (losses * masks[name]).sum(1).double().numpy()
            if kl and positions.numel():
                vectors = hidden[positions[:, 0].to(hidden.device), positions[:, 1].to(hidden.device)]
                if full_vectors is None:
                    full_vectors = vectors.clone()        # policy 0 is "full"
                kl_values, flip_values = prediction_change(run, full_vectors, vectors)
                kl_tokens, flip_tokens = torch.zeros(masks["after"].shape), torch.zeros(masks["after"].shape)
                kl_tokens[positions[:, 0], positions[:, 1]] = kl_values
                flip_tokens[positions[:, 0], positions[:, 1]] = flip_values
                for s, name in enumerate(strata):
                    kls[p, s, window] = (kl_tokens * masks[name]).sum(1).double().numpy()
                    flips[p, s, window] = (flip_tokens * masks[name]).sum(1).double().numpy()
            if b0 == 0 and policy.action == "remove_all":
                off, _ = forward_tokens(run, ids, None)
                checks["remove_all_matches_channel_off"] = bool(torch.allclose(losses, off, atol=1e-5, rtol=0))
                checks["remove_all_max_abs_difference"] = float((losses - off).abs().max())
        u, m = uniform_weight_counts(run, ids, spans)
        uniform, multi = uniform + u, multi + m
        if save_spans:
            for b, e, entry in zip(spans["batch"].tolist(), spans["end"].tolist(), spans["entry"].tolist()):
                lo, hi = e, min(ids.shape[1] - 1, e + AFTER_WINDOW)
                if lo >= hi:
                    continue
                spans_out["window"].append(b0 + b); spans_out["inject"].append(e); spans_out["entry"].append(entry)
                spans_out["degree"].append(int(degrees[entry]))
                spans_out["status"].append(status_of(entry, heldout, frequency))
                spans_out["count"].append(hi - lo)
                span_losses.append(np.asarray([float(t[b, lo:hi].sum()) for t in token_losses]))
        if log and (b0 // batch) % 16 == 0:
            log(f"  windows {b0 + len(chunk)}/{len(starts)} ({time.monotonic() - started:.0f}s)")
    checks["uniform_weight_fraction"] = uniform / multi if multi else None
    result = {"policies": [p.name for p in policies], "strata": strata, "starts": starts, "sums": sums, "counts": counts,
              "kl": kls, "flips": flips, "checks": checks, "seconds": time.monotonic() - started}
    if save_spans:
        result["spans"] = {k: np.asarray(v) for k, v in spans_out.items()}
        result["span_losses"] = np.stack(span_losses) if span_losses else np.zeros((0, len(policies)))
    return result


def _family_sums(result: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    """Policy family → (strata × windows) sums, random draws averaged."""
    families: dict[str, list[np.ndarray]] = {}
    for p, name in enumerate(result["policies"]):
        families.setdefault(EdgePolicy.parse(name).family, []).append(result[key][p])
    return {name: np.mean(arrays, axis=0) for name, arrays in families.items()}


def summarize(result: dict[str, Any], ks: Sequence[int], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    strata, counts = result["strata"], result["counts"]
    metrics = {"loss": _family_sums(result, "sums")}
    if result.get("kl") is not None:
        metrics["kl"] = _family_sums(result, "kl")
        metrics["flip"] = _family_sums(result, "flips")
    out: dict[str, Any] = {"strata": {}, "contrasts": [], "checks": result["checks"]}
    for s, stratum in enumerate(strata):
        n = counts[s]
        if n.sum() == 0:
            out["strata"][stratum] = {"tokens": 0}
            continue
        loss = metrics["loss"]
        entry: dict[str, Any] = {"tokens": int(n.sum()), "windows_with_tokens": int((n > 0).sum()),
                                 "loss_full": float(loss["full"][s].sum() / n.sum())}
        effect = paired_ratio_bootstrap(loss["remove_all"][s] - loss["full"][s], n, resamples=resamples, seed=seed)
        entry["channel_effect"] = effect
        ks_here = [int(stratum[1:].split("_")[0])] if stratum.startswith("k") else []
        for k in ks_here:
            for metric, table in metrics.items():
                block: dict[str, Any] = {}
                for action, label in (("remove", "comprehensiveness"), ("keep", "sufficiency")):
                    reference = table["full"][s] if metric == "loss" else 0.0
                    for order in ORDERS:
                        name = f"{action}_{order}{k}"
                        block[f"{label}_{order}"] = paired_ratio_bootstrap(table[name][s] - reference, n, resamples=resamples,
                                                                           seed=seed)
                    for a, b in (("top", "random"), ("random", "bottom")):
                        d = table[f"{action}_{a}{k}"][s] - table[f"{action}_{b}{k}"][s]
                        contrast = paired_ratio_bootstrap(d, n, resamples=resamples, seed=seed)
                        block[f"{label}_{a}_minus_{b}"] = contrast
                        out["contrasts"].append({"stratum": stratum, "k": k, "metric": metric, "measure": label,
                                                 "contrast": f"{a}-{b}", **{key: contrast[key] for key in
                                                                             ("mean", "ci_low", "ci_high", "p_value")}})
                if metric == "loss":
                    total = entry["channel_effect"]["mean"]
                    for order in ORDERS:
                        value = block[f"comprehensiveness_{order}"]["mean"]
                        block[f"normalized_comprehensiveness_{order}"] = value / total if total and abs(total) > 1e-12 else None
                entry[metric] = block
        out["strata"][stratum] = entry
    p_values = [c["p_value"] for c in out["contrasts"]]
    for contrast, adjusted in zip(out["contrasts"], holm_adjust(p_values) if p_values else []):
        contrast["p_holm"] = adjusted
        expected = 1 if contrast["measure"] == "comprehensiveness" else -1
        contrast["expected_direction"] = (contrast["mean"] or 0) * expected > 0
        contrast["significant"] = adjusted < 0.05 and contrast["expected_direction"]
    return out


def render(summary: dict[str, Any], header: dict[str, Any], ks: Sequence[int]) -> str:
    source = header["source"]
    lines = [f"# E5.1 faithfulness — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Run `{source['run']}`; channel `{source['channel_mode']}` "
             f"(composition `{(source.get('channel') or {}).get('composition')}`, operator `{(source.get('channel') or {}).get('operator')}`); "
             f"{header['windows']} evaluation windows; k ∈ {list(ks)}; {header['draws']} random draws.", "",
             f"Checks: remove_all = channel off: **{summary['checks'].get('remove_all_matches_channel_off')}**; "
             f"fraction of multi-edge occurrences with uniform weights: {fmt(summary['checks'].get('uniform_weight_fraction'), 3)}"
             + (" (static composer: top-k is a random subset by construction)." if (summary['checks'].get('uniform_weight_fraction') or 0) > 0.99 else "."),
             "", "Δloss on the 8 tokens after each ablated span (nats/token; cluster bootstrap over windows, 95% CI).", "",
             "| stratum | tokens | channel effect | comp. top | comp. random | comp. bottom | top − random | suff. top | suff. random | suff. bottom | random − top (suff.) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stratum, entry in summary["strata"].items():
        block = entry.get("loss")
        if not block:
            continue
        suff_gap = {"mean": -block["sufficiency_top_minus_random"]["mean"],
                    "ci_low": -block["sufficiency_top_minus_random"]["ci_high"],
                    "ci_high": -block["sufficiency_top_minus_random"]["ci_low"]}
        lines.append(f"| {stratum} | {entry['tokens']:,} | {fmt_ci(entry['channel_effect'])} | {fmt_ci(block['comprehensiveness_top'])} | "
                     f"{fmt_ci(block['comprehensiveness_random'])} | {fmt_ci(block['comprehensiveness_bottom'])} | "
                     f"{fmt_ci(block['comprehensiveness_top_minus_random'])} | {fmt_ci(block['sufficiency_top'])} | "
                     f"{fmt_ci(block['sufficiency_random'])} | {fmt_ci(block['sufficiency_bottom'])} | {fmt_ci(suff_gap)} |")
    if any("kl" in e for e in summary["strata"].values()):
        lines += ["", "Prediction change: KL(full ‖ ablated) per token and top-1 flip rate, removing k edges.", "",
                  "| stratum | KL top | KL random | KL bottom | flip top | flip random | flip bottom |", "|---|---:|---:|---:|---:|---:|---:|"]
        for stratum, entry in summary["strata"].items():
            if "kl" not in entry:
                continue
            kl, flip = entry["kl"], entry["flip"]
            lines.append(f"| {stratum} | " + " | ".join(fmt(kl[f"comprehensiveness_{o}"]["mean"]) for o in ORDERS) + " | "
                         + " | ".join(fmt(flip[f"comprehensiveness_{o}"]["mean"]) for o in ORDERS) + " |")
    significant = [c for c in summary["contrasts"] if c["significant"] and c["metric"] == "loss"]
    lines += ["", f"Holm-adjusted contrasts in the expected direction (loss): {len(significant)} of "
              f"{sum(c['metric'] == 'loss' for c in summary['contrasts'])}.", ""]
    return "\n".join(lines)


def probe_ablation(run: E5Run, output: Path, *, probes: Sequence[str], k: int, root: Path, seed: int) -> dict[str, Any]:
    """The WP-probe suite (fast settings) under remove_{top,random,bottom}<k>, compared with the full model."""
    from ..evaluation import channel_probes as cp
    settings = cp.ProbeSettings.fast()
    degrees = run.composer.schedule.degrees.cpu()
    out_dir = output / "probes"
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for policy in (EdgePolicy("full"), EdgePolicy("remove", "top", k), EdgePolicy("remove", "random", k),
                   EdgePolicy("remove", "bottom", k)):
        with edge_weight_hook(run.composer, ablation_transform(policy, degrees, seed=seed)):
            results, tables = cp.run_channel_probes(run.adapter, root, probes=probes, settings=settings)
        header = {"model": run.adapter.info, "policy": policy.name, "settings": {**asdict(settings), "probes": list(probes)}}
        paths[policy.family] = out_dir / f"{policy.family}.json"
        cp.write_outputs(paths[policy.family], results, tables, header)
    comparisons = {}
    for name in (f"remove_top{k}", f"remove_random{k}", f"remove_bottom{k}"):
        comparisons[f"{name}-vs-full"] = cp.compare_outputs(paths[name], paths["full"])
    comparisons[f"remove_top{k}-vs-remove_random{k}"] = cp.compare_outputs(paths[f"remove_top{k}"], paths[f"remove_random{k}"])
    write_json(out_dir / "comparisons.json", comparisons)
    return comparisons


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = {"experiment": "e5.1-faithfulness", "run": str(args.run), "checkpoint": args.checkpoint, "windows": args.windows,
              "ks": args.ks, "draws": args.draws, "batch": args.batch, "seed": args.seed, "kl": not args.no_kl,
              "resamples": args.resamples, "probes": args.probes, "probe_k": args.probe_k}
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device)
    result = faithfulness(run, windows=args.windows, batch=args.batch, ks=args.ks, draws=args.draws, seed=args.seed,
                          kl=not args.no_kl, save_spans=args.save_spans)
    summary = summarize(result, args.ks, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "windows": len(result["starts"]), "draws": args.draws}
    arrays = {"policies": np.asarray(result["policies"]), "strata": np.asarray(result["strata"]),
              "starts": np.asarray(result["starts"], dtype=np.int64), "sums": result["sums"].astype(np.float32),
              "counts": result["counts"].astype(np.int32)}
    if result["kl"] is not None:
        arrays.update(kl=result["kl"].astype(np.float32), flips=result["flips"].astype(np.float32))
    np.savez_compressed(args.output / "windows.npz", **arrays)
    if args.save_spans:
        np.savez_compressed(args.output / "spans.npz", losses=result["span_losses"].astype(np.float32),
                            **{k: v for k, v in result["spans"].items()})
    if args.probes:
        probes = [p.strip() for p in args.probes.split(",") if p.strip()]
        summary["probe_ablation"] = probe_ablation(run, args.output, probes=probes, k=args.probe_k, root=args.probes_root,
                                                   seed=args.seed)
    document = {**header, "summary": summary, "seconds": result["seconds"]}
    write_json(args.output / "summary.json", document)
    (args.output / "report.md").write_text(render(summary, header, args.ks))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--windows", type=int, default=1024); parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4]); parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--no-kl", action="store_true"); parser.add_argument("--save-spans", action="store_true")
    parser.add_argument("--probes", default="", help="comma-separated WP-probe names to rerun under ablation (fast settings)")
    parser.add_argument("--probe-k", type=int, default=2)
    parser.add_argument("--probes-root", type=Path, default=Path("~/data/vsa-llm/probes").expanduser())
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    document = run_experiment(args)
    print(json.dumps({k: v for k, v in document["summary"]["checks"].items()}, indent=2))


if __name__ == "__main__":
    main()
