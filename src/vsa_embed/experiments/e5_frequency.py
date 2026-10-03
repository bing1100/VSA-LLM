"""E5.5 frequency disentanglement (D5.5): how predictable is training frequency from embeddings?

The HRRBERT paper argued from t-SNE plots that structured code embeddings encode less frequency
than unstructured ones. Here that becomes a number: a ridge probe predicts `log(1 + training count)`
from a representation, with out-of-fold predictions (K folds; ridge strength chosen inside each
training fold by generalized cross-validation). Lower R² = frequency less linearly decodable = more
disentangled. Representations:

- `token_rows`: input-embedding rows of every token seen in the training stream (count from the
  whole stream; the trainer samples windows uniformly, so this is the expected exposure);
- `concept_surface`: mean input-embedding row of each linked entry's canonical alias (every
  condition, C0 included — what a model without a channel has for the concept);
- `concept_rows` / `concept_rows_unit`: the channel rows the model injects (composition, free table
  C2, random C1), raw and unit-normalized (direction only);
- `concept_rows_init` (composition runs): the same composer freshly initialized with the run's seed —
  the frequency that frame structure alone predicts before training (frames of frequent concepts
  differ, e.g. in degree), the reference against which learned rows are read.

Concept items are the non-held-out entries with training frequency ≥ 1 (the ontology's
`train_frequency`). Reported per representation: R², Spearman ρ of the out-of-fold predictions,
quartile AUC (top vs bottom quartile of true frequency, ranked by the prediction) and a bootstrap
interval for R² over items. For continued-pretraining hosts token rows mostly reflect pretraining,
so only the concept-level representations are meaningful there.

    python -m vsa_embed.experiments.e5_frequency --run RUN --output OUT [--folds 5] [--max-items N]
        [--token-counts PATH.npy]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from ..data.corpus import TokenCorpus
from ..evaluation.channel_probes import roc_auc
from .e5_common import E5Run, canonical_surfaces, entry_rows, finish_output, fmt, open_run, start_output, write_json

ALPHAS = tuple(10.0 ** p for p in range(-2, 6))


def token_counts(train_dir: Path, *, vocab: int, chunk: int = 50_000_000) -> np.ndarray:
    """Token counts over a whole training stream (read in chunks; nothing is written next to the corpus)."""
    corpus = TokenCorpus.open(Path(train_dir))
    counts = np.zeros(vocab, dtype=np.int64)
    for begin in range(0, len(corpus), chunk):
        part = np.asarray(corpus.tokens[begin:begin + chunk])
        counts += np.bincount(part, minlength=vocab)[:vocab]
    return counts


def ridge_gcv(x: np.ndarray, y: np.ndarray, alphas: Sequence[float] = ALPHAS):
    """Standardized ridge fit with the strength chosen by generalized cross-validation; returns (predict, alpha)."""
    mean, std = x.mean(0), x.std(0) + 1e-8
    xs = (x - mean) / std
    y_mean = y.mean()
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    uy = u.T @ (y - y_mean)
    n = len(y)
    best = None
    for alpha in alphas:
        shrink = s ** 2 / (s ** 2 + alpha)
        residual = (y - y_mean) - u @ (shrink * uy)
        score = float((residual ** 2).mean() / max(1e-12, (1 - shrink.sum() / n)) ** 2)
        if best is None or score < best[0]:
            best = (score, alpha)
    alpha = best[1]
    w = vt.T @ ((s / (s ** 2 + alpha)) * uy)

    def predict(z: np.ndarray) -> np.ndarray:
        return ((z - mean) / std) @ w + y_mean

    return predict, alpha


def probe(x: np.ndarray, y: np.ndarray, *, folds: int = 5, seed: int = 0, resamples: int = 1000) -> dict[str, Any]:
    """Out-of-fold ridge predictions and their R², Spearman ρ and quartile AUC."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(y)
    order = np.random.default_rng(seed).permutation(n)
    fold_of = np.empty(n, dtype=np.int64)
    fold_of[order] = np.arange(n) % folds
    prediction = np.zeros(n)
    alphas = []
    for f in range(folds):
        train, test = fold_of != f, fold_of == f
        predict, alpha = ridge_gcv(x[train], y[train])
        prediction[test] = predict(x[test]); alphas.append(alpha)
    errors = (prediction - y) ** 2
    deviations = (y - y.mean()) ** 2
    r2 = 1 - errors.sum() / deviations.sum()
    index = np.random.default_rng(seed + 1).integers(0, n, size=(resamples, n))
    boot = 1 - errors[index].sum(1) / np.maximum(((y[index] - y[index].mean(1, keepdims=True)) ** 2).sum(1), 1e-12)
    low, high = np.quantile(y, [0.25, 0.75])
    extreme = (y <= low) | (y >= high)
    auc = roc_auc(prediction[extreme], y[extreme] >= high) if extreme.any() and low < high else None
    return {"n": n, "r2": float(r2), "r2_ci_low": float(np.quantile(boot, 0.025)), "r2_ci_high": float(np.quantile(boot, 0.975)),
            "spearman": float(spearmanr(prediction, y).statistic) if n > 2 else None, "quartile_auc": auc,
            "alphas": alphas, "prediction": prediction, "target": y}


def representations(run: E5Run, counts: np.ndarray | None, *, max_items: int | None, seed: int) -> dict[str, dict[str, Any]]:
    """Representation name → item ids, features and log-frequency targets."""
    out: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(seed)
    wte = run.model.model.get_input_embeddings().weight.detach().float().cpu().numpy()

    def subsample(ids: np.ndarray) -> np.ndarray:
        if max_items is not None and len(ids) > max_items:
            ids = np.sort(rng.choice(ids, size=max_items, replace=False))
        return ids

    if counts is not None:
        ids = subsample(np.flatnonzero(counts[:wte.shape[0]] > 0))
        out["token_rows"] = {"ids": ids, "x": wte[ids], "y": np.log1p(counts[ids])}
    if run.ontology is not None and run.table is not None:
        frequency = np.asarray(run.ontology["train_frequency"])
        heldout = set(run.adapter.heldout_entries)
        entries = np.asarray([e for e in np.flatnonzero(frequency > 0) if e not in heldout], dtype=np.int64)
        entries = subsample(entries)
        surfaces = canonical_surfaces(run.table, run.tokenizer, run.min_subtokens, entries.tolist())
        entries = np.asarray([e for e in entries if e in surfaces], dtype=np.int64)
        encoded = run.tokenizer([" " + surfaces[int(e)]["surface"] for e in entries], add_special_tokens=False)["input_ids"]
        surface = np.stack([wte[ids].mean(0) for ids in encoded]) if len(entries) else np.zeros((0, wte.shape[1]))
        y = np.log1p(frequency[entries])
        out["concept_surface"] = {"ids": entries, "x": surface, "y": y}
        if run.channel is not None and run.mode != "hashed":
            rows = entry_rows(run.channel, torch.as_tensor(entries)).numpy()
            out["concept_rows"] = {"ids": entries, "x": rows, "y": y}
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            out["concept_rows_unit"] = {"ids": entries, "x": rows / np.maximum(norms, 1e-12), "y": y}
            out["concept_rows"]["norm_spearman"] = float(spearmanr(norms[:, 0], y).statistic) if len(y) > 2 else None
            if run.mode == "compose":
                # Reference: the same composer freshly initialized (run seed) — frequency that frame
                # structure alone predicts, before any training.
                from ..training.lm import build_channel
                torch.manual_seed(int(run.config["seed"]))
                fresh, _ = build_channel(run.config, run.ontology, run.channel.gate.in_features // 2,
                                         host=run.adapter.model.model)
                out["concept_rows_init"] = {"ids": entries, "x": entry_rows(fresh, torch.as_tensor(entries)).numpy(), "y": y}
    return out


def render(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E5.5 frequency disentanglement — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Run `{source['run']}`, channel `{source['channel_mode']}`. Ridge probe of log(1 + training count), "
             f"{header['folds']}-fold out-of-fold predictions; lower R² = less frequency information.", "",
             "| representation | items | R² | 95% CI | Spearman ρ | quartile AUC |", "|---|---:|---:|---|---:|---:|"]
    for name, r in summary.items():
        lines.append(f"| {name} | {r['n']:,} | {fmt(r['r2'], 3)} | [{fmt(r['r2_ci_low'], 3)}, {fmt(r['r2_ci_high'], 3)}] | "
                     f"{fmt(r['spearman'], 3)} | {fmt(r['quartile_auc'], 3)} |")
    norm = summary.get("concept_rows", {}).get("norm_spearman")
    if norm is not None:
        lines += ["", f"Spearman ρ between channel-row norm and log frequency: {fmt(norm, 3)}."]
    return "\n".join(lines + [""])


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config = {"experiment": "e5.5-frequency", "run": str(args.run), "checkpoint": args.checkpoint, "folds": args.folds,
              "max_items": args.max_items, "seed": args.seed, "resamples": args.resamples,
              "token_counts": str(args.token_counts) if args.token_counts else None}
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device)
    started = time.monotonic()
    vocab = run.model.model.get_input_embeddings().weight.shape[0]
    if args.token_counts and Path(args.token_counts).exists():
        counts = np.load(args.token_counts)[:vocab]
    elif run.config["data"].get("train") and Path(run.config["data"]["train"]).exists():
        counts = token_counts(Path(run.config["data"]["train"]), vocab=vocab)
        if args.token_counts:            # a path that does not exist yet: save the counts there for the next runs
            Path(args.token_counts).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.token_counts, counts)
    else:
        counts = None
    reps = representations(run, counts, max_items=args.max_items, seed=args.seed)
    summary, arrays = {}, {}
    for name, rep in reps.items():
        if len(rep["ids"]) < 2 * args.folds:
            continue
        result = probe(rep["x"], rep["y"], folds=args.folds, seed=args.seed, resamples=args.resamples)
        arrays[f"{name}_ids"] = rep["ids"].astype(np.int64)
        arrays[f"{name}_prediction"] = result.pop("prediction").astype(np.float32)
        arrays[f"{name}_target"] = result.pop("target").astype(np.float32)
        if "norm_spearman" in rep:
            result["norm_spearman"] = rep["norm_spearman"]
        summary[name] = result
    if args.save_predictions:
        np.savez_compressed(args.output / "predictions.npz", **arrays)
    header = {"source": run.describe(), "folds": args.folds}
    write_json(args.output / "summary.json", {**header, "summary": summary, "seconds": time.monotonic() - started})
    (args.output / "report.md").write_text(render(summary, header))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--folds", type=int, default=5); parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--token-counts", type=Path, default=None,
                        help="token counts of the training stream (.npy); computed and saved there if missing")
    parser.add_argument("--save-predictions", action="store_true", help="out-of-fold predictions per item (npz)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run_experiment(args)
    print(json.dumps({k: {m: v.get(m) for m in ("n", "r2", "spearman")} for k, v in result["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()
