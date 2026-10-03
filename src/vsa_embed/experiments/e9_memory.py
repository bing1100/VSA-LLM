"""E9 memory/throughput probe for the Qwen3 hosts (WP-Qwen; the B6 pattern of `host_memory`, on the real E9 run).

For each host × setting (host dtype, gradient checkpointing) × micro-batch (1, 2, 4, 8; ascending, stopping at
the first OOM): build the E9 C5 run exactly as the trainer does — the `e9_plan` config (LoRA r = 64, attentive
channel with the P1 context, `channel.scale_to_host`, the T5 Qwen3 ontology), `training.lm.build_model` /
`build_channel` / `wrap_host` and the trainer's AdamW (`build_optimizer`) — and run one warm-up plus `--steps`
optimizer steps of one micro-batch each on real T5 training batches (`sample_batch`, sequence 1024, bf16
autocast), recording peak allocated and reserved GiB and tokens/s. With the optimizer state still resident,
evaluation forwards (no gradient) at batches 2, 4, 8, 16 measure the evaluation batch.

Recommendation per host (`recommendation.json`, read by `e9_plan --memory-report`): among the settings whose
training peak (reserved) stays under `--budget-gib` (default: GPU memory − 1.5 GiB) and whose micro-batch
divides 64, the fastest with an fp32 host (the recipe of the smaller hosts and of SmolLM2); a bf16 host only
where no fp32 setting fits. Its evaluation batch is the largest measured one under the budget (at most 16).

C5 is the largest E9 model (C2's capacity-matched table is the same size; C0′ and P0 have no channel), so its
settings cover the stage. Runs as one GPU job that needs the whole GPU:

    python -m vsa_embed.experiments.e9_memory --output experiments/e9-retrofit/memory/qwen3-v1 \
        [--hosts Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base] [--steps 3] [--budget-gib 22.5]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vsa_embed.data.corpus import TokenCorpus, collate_windows, sample_batch
from vsa_embed.experiments.e4_plan import matched_sizes
from vsa_embed.experiments.e9_plan import CPT_HOSTS, QWEN3_HOSTS, estimate_hours, host_plan, run_config
from vsa_embed.experiments.e9_tracks import track_spec
from vsa_embed.provenance import prepare_output_dir, write_run_metadata

MICRO_BATCHES = (1, 2, 4, 8)
EVAL_BATCHES = (2, 4, 8, 16)
SEQUENCES_PER_STEP = 64
# (host dtype, gradient checkpointing) settings per host: checkpointing with and without for 4B (and 1.7B, whose
# fp32 activations may not fit beyond micro-batch 1); a bf16 host for 4B (fp32 weights alone are 16 GB).
SETTINGS = {"Qwen3-0.6B-Base": [("float32", False)],
            "Qwen3-1.7B-Base": [("float32", False), ("float32", True)],
            "Qwen3-4B-Base": [("float32", False), ("float32", True), ("bfloat16", False), ("bfloat16", True)]}


def probe_config(host: str, micro_batch: int, *, host_dtype: str = "float32", checkpointing: bool = False,
                 track: str = "t5", family: str = "qwen3", data_root: Path | None = None, lora_rank: int = 64) -> dict[str, Any]:
    """The E9 C5 config of `host` (as `e9_plan` writes it) at one micro-batch and memory setting."""
    from vsa_embed.training.lm import resolve_config
    spec = track_spec(track, family)
    root = Path(data_root) if data_root else spec.data_root
    free_dimension, _ = matched_sizes(root / "ontology.pt", CPT_HOSTS[host]["width"], 256)
    plan = host_plan(host, "lora")
    plan["micro"]["lora"] = int(micro_batch)
    plan["model"] = {**({"gradient_checkpointing": True, "checkpoint_use_reentrant": False} if checkpointing else {}),
                     **({"host_dtype": host_dtype} if host_dtype != "float32" else {})}
    _, config = run_config(stage="memory", host=host, mode="lora", model="C5", seed=1, data_root=root, tokens=50_000_000,
                           lora_rank=lora_rank, host_lr=None, gate_bias=0.0, free_dimension=free_dimension,
                           sequences_per_step=SEQUENCES_PER_STEP, windows=1024, eval_split=spec.eval_split, plan=plan,
                           channel_extra={"scale_to_host": True})
    return resolve_config(config)


def _gib(value: int) -> float:
    return value / 2**30


def _reset(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _peaks(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_gib": None, "peak_reserved_gib": None}
    return {"peak_gib": _gib(torch.cuda.max_memory_allocated()), "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved())}


def measure_setting(host: str, *, host_dtype: str, checkpointing: bool, device: torch.device, micro_batches=MICRO_BATCHES,
                    eval_batches=EVAL_BATCHES, steps: int = 3, data_root: Path | None = None,
                    log=print) -> list[dict[str, Any]]:
    """Rows for one (host, dtype, checkpointing): training steps per micro-batch, then evaluation batches."""
    from vsa_embed.training.lm import build_channel, build_model, build_optimizer, wrap_host
    rows: list[dict[str, Any]] = []
    base_row = {"host": host, "host_dtype": host_dtype, "checkpointing": checkpointing}
    config = probe_config(host, micro_batches[0], host_dtype=host_dtype, checkpointing=checkpointing, data_root=data_root)
    _reset(device)
    torch.manual_seed(config["seed"])
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    corpus = TokenCorpus.open(Path(config["data"]["train"]))
    entry_mask = np.ones(int(ontology["entry_count"]), dtype=bool)
    entry_mask[list(ontology["heldout_entries"])] = False
    started = time.monotonic()
    try:
        base = build_model(config)
        channel, context = build_channel(config, ontology, base.get_input_embeddings().weight.shape[1], host=base)
        channel.set_unseen(ontology["heldout_entries"])
        model = wrap_host(config, base, channel, context).to(device)
        optimizer = build_optimizer(model, config["train"], device)
    except torch.cuda.OutOfMemoryError:                # the host alone does not fit (another process on the GPU?)
        base = channel = context = model = None
        _reset(device)
        row = {**base_row, "kind": "train", "micro_batch": micro_batches[0], "ok": False, "tokens_per_s": None,
               "error": "out of memory while loading the host", **_peaks(device)}
        log(json.dumps(row))
        return [row]
    load_seconds = time.monotonic() - started
    parameters = {"host_parameters": sum(p.numel() for p in base.parameters() if not p.requires_grad),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "channel_host_scale": float(channel.host_scale), "load_seconds": load_seconds}
    length, autocast = config["model"]["seq_len"], device.type == "cuda"
    model.train()
    fitted = 0
    for micro in micro_batches:
        timings: list[float] = []
        row = {**base_row, "kind": "train", "micro_batch": micro, **parameters}
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(steps + 1):
                ids, spans = sample_batch(corpus, seed=config["data"]["seed"], step=step, micro_step=0, batch=micro, length=length,
                                          min_subtokens=config["data"]["min_subtokens"], entry_mask=entry_mask)
                ids = ids.to(device); spans = {k: v.to(device) for k, v in spans.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                    loss = model(ids, spans=spans, labels=ids)["loss"] + config["train"]["delta_weight"] * channel.composer.delta_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["train"]["grad_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if step:                                   # the first step allocates the optimizer state
                    timings.append(time.perf_counter() - tick)
            seconds = float(np.median(timings))
            row.update(ok=True, step_s=seconds, tokens_per_s=micro * length / seconds, **_peaks(device))
            fitted = micro
        except torch.cuda.OutOfMemoryError:
            row.update(ok=False, tokens_per_s=None, **_peaks(device))
        optimizer.zero_grad(set_to_none=True)
        loss = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rows.append(row); log(json.dumps(row))
        if not row["ok"]:
            break
    if fitted:
        eval_corpus = TokenCorpus.open(Path(config["data"]["eval"]))
        model.eval()
        for batch in eval_batches:
            row = {**base_row, "kind": "eval", "eval_batch": batch, "after_micro_batch": fitted}
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                windows = [eval_corpus.window(i * length, length, min_subtokens=config["data"]["min_subtokens"]) for i in range(batch)]
                ids, spans = collate_windows(windows)
                tick = time.perf_counter()
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                    model(ids.to(device), spans={k: v.to(device) for k, v in spans.items()}, labels=ids.to(device), reduction="none")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                row.update(ok=True, seconds=time.perf_counter() - tick, **_peaks(device))
            except torch.cuda.OutOfMemoryError:
                row.update(ok=False, **_peaks(device))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rows.append(row); log(json.dumps(row))
            if not row["ok"]:
                break
    del model, optimizer, channel, context, base
    _reset(device)
    return rows


def recommend(rows: list[dict[str, Any]], *, budget_gib: float, tokens: int = 50_000_000) -> dict[str, dict[str, Any]]:
    """host → {"lora": setting} (see the module docstring); hosts with no fitting setting are left out."""
    out: dict[str, dict[str, Any]] = {}
    for host in dict.fromkeys(r["host"] for r in rows):
        def fits(r: dict[str, Any]) -> bool:
            peak = r.get("peak_reserved_gib") if r.get("peak_reserved_gib") is not None else r.get("peak_gib")
            return bool(r.get("ok")) and (peak is None or peak <= budget_gib)
        train = [r for r in rows if r["host"] == host and r["kind"] == "train" and fits(r) and SEQUENCES_PER_STEP % r["micro_batch"] == 0]
        if not train:
            continue
        preferred = [r for r in train if r["host_dtype"] == "float32"] or train
        best = max(preferred, key=lambda r: (r["tokens_per_s"] or 0.0, -(r.get("peak_gib") or 0.0)))
        evals = [r for r in rows if r["host"] == host and r["kind"] == "eval" and fits(r)
                 and r["host_dtype"] == best["host_dtype"] and r["checkpointing"] == best["checkpointing"]]
        setting = {"micro_batch": best["micro_batch"], "gradient_checkpointing": best["checkpointing"],
                   "host_dtype": best["host_dtype"], "tokens_per_s": best["tokens_per_s"], "peak_gib": best.get("peak_gib"),
                   "peak_reserved_gib": best.get("peak_reserved_gib"),
                   "eval_batch": max((r["eval_batch"] for r in evals), default=None),
                   "gpu_hours_per_run": estimate_hours(host, tokens, tokens_per_s=best["tokens_per_s"]),
                   "rule": "fastest setting under the budget with an fp32 host (bf16 only if no fp32 setting fits)"}
        out[host] = {"lora": setting}
    return out


def render(rows: list[dict[str, Any]], recommendation: dict[str, Any], header: dict[str, Any]) -> str:
    lines = ["# E9 Qwen3 memory and throughput (C5, LoRA r = 64, sequence 1024, bf16 autocast)", "",
             f"GPU: {header['gpu']} ({header['total_gib']:.1f} GiB); budget {header['budget_gib']:.1f} GiB (peak reserved). "
             f"T5 Qwen3 batches; {header['steps']} timed optimizer steps per micro-batch after one warm-up step "
             "(fused AdamW, the trainer's parameter groups); channel injection scaled to the host (`scale_to_host`).", "",
             "| Host | dtype | checkpointing | micro-batch | peak GiB | reserved GiB | tokens/s |", "|---|---|---|---:|---:|---:|---:|"]
    fmt = lambda v, f: "—" if v is None else format(v, f)
    for r in rows:
        if r["kind"] == "train":
            lines.append(f"| {r['host']} | {r['host_dtype']} | {r['checkpointing']} | {r['micro_batch']} | {fmt(r.get('peak_gib'), '.1f')} | "
                         f"{fmt(r.get('peak_reserved_gib'), '.1f')} | " + (f"{r['tokens_per_s']:,.0f} |" if r["ok"] else "OOM |"))
    lines += ["", "Evaluation forwards (no gradient, optimizer state resident):", "",
              "| Host | dtype | checkpointing | eval batch | peak GiB | reserved GiB | ok |", "|---|---|---|---:|---:|---:|:-:|"]
    for r in rows:
        if r["kind"] == "eval":
            lines.append(f"| {r['host']} | {r['host_dtype']} | {r['checkpointing']} | {r['eval_batch']} | {fmt(r.get('peak_gib'), '.1f')} | "
                         f"{fmt(r.get('peak_reserved_gib'), '.1f')} | {'yes' if r['ok'] else 'OOM'} |")
    lines += ["", "## Recommendation (`recommendation.json`, read by `e9_plan`)", "",
              "| Host | micro-batch × accumulation | checkpointing | dtype | eval batch | tokens/s | GPU-h per 50M-token run |",
              "|---|---|---|---|---:|---:|---:|"]
    for host, by_mode in recommendation.items():
        s = by_mode["lora"]
        lines.append(f"| {host} | {s['micro_batch']} × {SEQUENCES_PER_STEP // s['micro_batch']} | {s['gradient_checkpointing']} | "
                     f"{s['host_dtype']} | {s['eval_batch']} | {s['tokens_per_s']:,.0f} | {s['gpu_hours_per_run']:.1f} |")
    missing = [h for h in header["hosts"] if h not in recommendation]
    if missing:
        lines += ["", f"No setting fits the budget for: {', '.join(missing)}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hosts", nargs="+", default=["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"], choices=list(QWEN3_HOSTS))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--budget-gib", type=float, default=None, help="default: GPU memory − 1.5 GiB")
    parser.add_argument("--data-root", type=Path, default=None, help="default: the T5 Qwen3 corpus")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    total = _gib(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0.0
    budget = args.budget_gib if args.budget_gib is not None else total - 1.5
    settings = {h: [{"host_dtype": d, "checkpointing": c} for d, c in SETTINGS[h]] for h in args.hosts}
    config = {"experiment": "e9-memory-qwen3", "hosts": args.hosts, "settings": settings, "micro_batches": list(MICRO_BATCHES), "eval_batches": list(EVAL_BATCHES), "steps": args.steps, "budget_gib": budget,
              "sequence": 1024, "model": "C5", "lora_rank": 64, "track": "t5", "family": "qwen3",
              "data_root": str(args.data_root) if args.data_root else None}
    git_at_start = prepare_output_dir(args.output)
    (args.output / "resolved_config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    with (args.output / "rows.jsonl").open("w") as sink:
        def log(line: str) -> None:
            print(line, flush=True); sink.write(line + "\n"); sink.flush()
        for host in args.hosts:
            for host_dtype, checkpointing in SETTINGS[host]:
                rows += measure_setting(host, host_dtype=host_dtype, checkpointing=checkpointing, device=device, steps=args.steps,
                                        data_root=args.data_root, log=log)
    recommendation = recommend(rows, budget_gib=budget)
    header = {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "total_gib": total, "budget_gib": budget,
              "steps": args.steps, "hosts": args.hosts}
    (args.output / "memory.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "recommendation.json").write_text(json.dumps({**header, "hosts": recommendation, "probed_hosts": args.hosts}, indent=2) + "\n")
    (args.output / "report.md").write_text(render(rows, recommendation, header))
    write_run_metadata(args.output, config, git_at_start=git_at_start, device=device)


if __name__ == "__main__":
    main()
