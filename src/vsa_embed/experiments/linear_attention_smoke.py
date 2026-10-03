"""Fast-path check of a hybrid linear-attention host (Qwen3.5 Gated DeltaNet; WP-Qwen35 environment).

    python -m vsa_embed.experiments.linear_attention_smoke --output DIR [--device cuda] [--host tiny | HF id]
        [--batch 2] [--length 512] [--reps 5] [--max-gib 2]

Loads the host through the trainer (`training.lm.build_model` + `wrap_host`: `use_cache` off, CUDA-only kernel
dispatch, LoRA on every projection of every layer with `model.lora_targets: linear_attention`) and runs the same
forward/backward step (bf16 autocast, as the trainer) with the reference implementation
(`linear_attention.reference_only`) and, on CUDA, with the fast kernels (flash-linear-attention chunked gated delta
rule, causal-conv1d). Records which implementation ran (calls per function), the median step time of each path,
their agreement (loss, final hidden states, LoRA gradients) and the peak memory. `--host tiny` (default) is a
randomly initialised 4-layer Qwen3.5 text model (3 linear-attention layers, 1 full-attention layer, as the 3:1 pattern
of the real hosts) small enough for a GPU shared with a running job; `--max-gib` caps this process's CUDA allocator.
Writes `smoke.json`, `report.md`, `manifest.json`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from vsa_embed.integrations import linear_attention as la
from vsa_embed.provenance import prepare_output_dir, write_run_metadata

TINY = {"vocab_size": 4096, "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64, "linear_num_key_heads": 4, "linear_num_value_heads": 4,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64, "max_position_embeddings": 4096, "tie_word_embeddings": True,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}


def tiny_host(path: Path, seed: int = 0, overrides: dict[str, Any] | None = None) -> Path:
    """Save a randomly initialised small Qwen3.5 text model (`TINY` updated by `overrides`) to `path`."""
    import transformers
    torch.manual_seed(seed)
    transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(**{**TINY, **(overrides or {})})).save_pretrained(path)
    return path


def build(host: str, *, lora_rank: int = 8) -> tuple[Any, dict[str, Any]]:
    import vsa_embed.training.lm as lm
    config = lm.resolve_config({"model": {"pretrained": host, "host_mode": "lora", "lora_rank": lora_rank,
                                          "lora_targets": "linear_attention"}, "device": "cpu"})
    base = lm.build_model(config)
    model = lm.wrap_host(config, base, None, None)
    with torch.no_grad():                       # non-zero adapters, so every adapter matrix gets a gradient
        generator = torch.Generator().manual_seed(1)
        for name, p in model.named_parameters():
            if name.endswith("lora_b"):
                p.copy_(torch.randn(p.shape, generator=generator) * 0.01)
    return model, config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def step(model: Any, ids: torch.Tensor, device: torch.device) -> tuple[float, torch.Tensor, list[torch.Tensor], float, float]:
    """One forward/backward (bf16 autocast on CUDA): (loss, final hidden states, LoRA gradients, forward s, total s)."""
    model.zero_grad(set_to_none=True)
    _sync(device)
    tick = time.perf_counter()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(ids, labels=ids)
    _sync(device)
    forward = time.perf_counter() - tick
    out["loss"].backward()
    _sync(device)
    total = time.perf_counter() - tick
    grads = [p.grad.detach().float().cpu().flatten() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    return float(out["loss"].detach()), out["hidden"].detach().float().cpu(), grads, forward, total


def measure(model: Any, ids: torch.Tensor, device: torch.device, *, reps: int, reference: bool) -> dict[str, Any]:
    la.kernel_calls(reset=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with (la.reference_only() if reference else contextlib.nullcontext()):
        warm = time.perf_counter()
        loss, hidden, grads, _, _ = step(model, ids, device)        # warm-up (Triton compiles and autotunes here)
        warmup = time.perf_counter() - warm
        timings = [step(model, ids, device)[3:] for _ in range(reps)]
    tokens = ids.numel()
    total = statistics.median(t for _, t in timings)
    return {"path": "reference" if reference else "fast", "loss": loss, "warmup_s": warmup,
            "forward_s": statistics.median(f for f, _ in timings), "step_s": total, "tokens_per_s": tokens / total,
            "calls": la.kernel_calls(reset=True),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
            "_hidden": hidden, "_grads": grads}


def agreement(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    ga, gb = torch.cat(a["_grads"]), torch.cat(b["_grads"])
    per = [float(torch.nn.functional.cosine_similarity(x, y, dim=0)) for x, y in zip(a["_grads"], b["_grads"]) if x.norm() > 0]
    hidden = (a["_hidden"] - b["_hidden"]).abs()
    return {"loss_abs_diff": abs(a["loss"] - b["loss"]),
            "hidden_max_abs_diff": float(hidden.max()), "hidden_rel_rms": float(hidden.pow(2).mean().sqrt() / b["_hidden"].pow(2).mean().sqrt()),
            "grad_cosine": float(torch.nn.functional.cosine_similarity(ga, gb, dim=0)), "grad_cosine_min": min(per) if per else float("nan"),
            "grad_rel_norm_diff": float((ga - gb).norm() / gb.norm())}


def render(result: dict[str, Any]) -> str:
    lines = [f"# Linear-attention fast-path smoke ({result['host']}, {result['device']})", "",
             f"Batch {result['batch']} × {result['length']} tokens, bf16 autocast on CUDA, fp32 weights, LoRA r = {result['lora_rank']} on "
             f"every projection (`lora_targets: linear_attention`, {result['lora_coverage']['adapters']} adapters, "
             f"uncovered token mixers: {result['lora_coverage']['uncovered_mixers'] or 'none'}); median of {result['reps']} steps after "
             "one warm-up step.", "",
             f"Bound by transformers: fast path {'yes' if result['kernels']['status'].get('modeling_qwen3_5', {}).get('fast_path_bound') else 'no'} "
             f"({result['kernels']['status'].get('modeling_qwen3_5', {}).get('packages')}).", "",
             "| path | forward s | forward+backward s | tokens/s | peak GiB | calls |", "|---|---:|---:|---:|---:|---|"]
    for row in result["paths"]:
        peak = "—" if row["peak_gib"] is None else f"{row['peak_gib']:.2f}"
        lines.append(f"| {row['path']} | {row['forward_s']:.4f} | {row['step_s']:.4f} | {row['tokens_per_s']:,.0f} | {peak} | {row['calls']} |")
    if result.get("speedup"):
        lines += ["", f"Speed-up of the fast path (forward+backward): **{result['speedup']:.1f}×** "
                      f"(forward only {result['forward_speedup']:.1f}×)."]
    if result.get("agreement"):
        a = result["agreement"]
        lines += ["", f"Agreement fast vs reference: |Δloss| {a['loss_abs_diff']:.2e}, final hidden max |Δ| {a['hidden_max_abs_diff']:.3e} "
                      f"(relative RMS {a['hidden_rel_rms']:.2e}), LoRA gradient cosine {a['grad_cosine']:.5f} (min per tensor "
                      f"{a['grad_cosine_min']:.5f}), relative gradient difference {a['grad_rel_norm_diff']:.2e}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="tiny", help="tiny (random 4-layer Qwen3.5 text model) or a cached HF id")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=512); parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=8); parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-gib", type=float, default=2.0, help="cap of this process's CUDA allocator")
    parser.add_argument("--tiny-overrides", default="{}", help="JSON merged into the tiny config (e.g. the real hosts' head shapes)")
    args = parser.parse_args(argv)
    overrides = json.loads(args.tiny_overrides)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gib / total), device)
    git_at_start = prepare_output_dir(args.output)
    with tempfile.TemporaryDirectory() as scratch:
        host = str(tiny_host(Path(scratch) / "tiny", overrides=overrides)) if args.host == "tiny" else args.host
        model, config = build(host, lora_rank=args.lora_rank)
        model = model.to(device).train()
        vocab = model.model.get_input_embeddings().weight.shape[0]
        ids = torch.randint(0, vocab, (args.batch, args.length), generator=torch.Generator().manual_seed(2)).to(device)
        paths = [measure(model, ids, device, reps=args.reps, reference=True)]
        if device.type == "cuda":
            paths.append(measure(model, ids, device, reps=args.reps, reference=False))
    result: dict[str, Any] = {"host": args.host, "device": str(device), "batch": args.batch, "length": args.length, "reps": args.reps,
                              "lora_rank": args.lora_rank, "tiny_config": {**TINY, **overrides} if args.host == "tiny" else None,
                              "kernels": {"status": la.kernel_status()}, "lora_coverage": la.lora_layer_coverage(model.model)}
    if len(paths) == 2:
        result["agreement"] = agreement(paths[1], paths[0])
        result["speedup"] = paths[0]["step_s"] / paths[1]["step_s"]
        result["forward_speedup"] = paths[0]["forward_s"] / paths[1]["forward_s"]
    result["paths"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in paths]
    (args.output / "smoke.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (args.output / "report.md").write_text(render(result))
    write_run_metadata(args.output, {"experiment": "linear-attention-smoke", **{k: v for k, v in vars(args).items() if k != "output"}},
                       git_at_start=git_at_start, device=device)
    print(render(result))
    return result


if __name__ == "__main__":
    main()
