"""B6 acceptance: one training step per host on the local GPU, with peak memory and tokens/s.

For each host × host mode × micro-batch (sequence length 1024, bf16 autocast), run one
forward/backward/optimizer step with a span channel attached (synthetic spans, ≈ 3% of tokens)
and record peak GPU memory and throughput. From-scratch GPT-2-style sizes are measured too.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.integrations.transformers import ChannelLM
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import SpanChannel

SCRATCH_SIZES = {
    "scratch-50M": dict(n_layer=8, n_embd=512, n_head=8),
    "scratch-125M": dict(n_layer=12, n_embd=768, n_head=12),
    "scratch-350M": dict(n_layer=24, n_embd=1024, n_head=16),
}


def build_host(name: str) -> torch.nn.Module:
    if name in SCRATCH_SIZES:
        config = GPT2Config(vocab_size=50257, n_positions=1024, **SCRATCH_SIZES[name])
        return GPT2LMHeadModel(config, attn_implementation="sdpa")
    return AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.float32,
                                                attn_implementation="sdpa")


def synthetic_spans(batch: int, length: int, entries: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    count = max(1, int(0.03 * batch * length))
    b = torch.randint(0, batch, (count,), generator=generator)
    end = torch.randint(2, length, (count,), generator=generator)
    return {"batch": b, "start": end - 1, "end": end, "inject": end,
            "entry": torch.randint(0, entries, (count,), generator=generator),
            "confidence": torch.ones(count), "length": torch.full((count,), 2)}


def measure(name: str, host_mode: str, micro_batch: int, *, length: int = 1024, checkpointing: bool = False,
            entries: int = 20000, atomics: int = 8192, dimension: int = 256) -> dict[str, Any]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model = build_host(name).to(device)
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    width = model.get_input_embeddings().weight.shape[1]
    generator = torch.Generator().manual_seed(0)
    frames = [[(int(torch.randint(0, 16, (), generator=generator)), int(torch.randint(0, atomics, (), generator=generator)))
               for _ in range(8)] for _ in range(entries)]
    composer = FrameComposer(FrameSchedule.from_frames(frames), atomics, 16, dimension, mode="attentive")
    wrapped = ChannelLM(model, SpanChannel(composer, width, entry_count=entries), host_mode=host_mode).to(device)
    optimizer = torch.optim.AdamW(wrapped.trainable_parameters(), lr=1e-4, fused=True)
    ids = torch.randint(0, model.get_input_embeddings().weight.shape[0], (micro_batch, length), device=device)
    spans = synthetic_spans(micro_batch, length, entries, generator)
    try:
        timings = []
        for _ in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = wrapped(ids, spans=spans, labels=ids)["loss"]
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); timings.append(time.perf_counter() - start)
        step = min(timings[1:])
        result = {"host": name, "host_mode": host_mode, "micro_batch": micro_batch, "checkpointing": checkpointing,
                  "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "step_s": step,
                  "tokens_per_s": micro_batch * length / step, "ok": True,
                  "host_parameters": sum(p.numel() for p in model.parameters())}
    except torch.cuda.OutOfMemoryError:
        result = {"host": name, "host_mode": host_mode, "micro_batch": micro_batch, "checkpointing": checkpointing,
                  "ok": False, "peak_gib": float("nan")}
    del wrapped, model, optimizer
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = {"length": 1024, "dtype": "bf16-autocast", "entries": 20000, "atomics": 8192, "dimension": 256}
    git_at_start = prepare_output_dir(args.output)
    plan = [
        ("scratch-50M", "train", [8, 16, 32], False), ("scratch-125M", "train", [8, 16], False),
        ("scratch-125M", "train", [16, 32], True), ("scratch-350M", "train", [4, 8, 16], True),
        ("HuggingFaceTB/SmolLM2-135M", "frozen", [8, 16], False), ("HuggingFaceTB/SmolLM2-135M", "lora", [8], False),
        ("HuggingFaceTB/SmolLM2-360M", "frozen", [4, 8], False), ("HuggingFaceTB/SmolLM2-360M", "frozen", [16], True),
        ("Qwen/Qwen2.5-0.5B", "frozen", [2, 4], False), ("Qwen/Qwen2.5-0.5B", "frozen", [8], True),
    ]
    rows = []
    for host, mode, batches, checkpointing in plan:
        for micro_batch in batches:
            row = measure(host, mode, micro_batch, checkpointing=checkpointing)
            rows.append(row); print(json.dumps(row), flush=True)
    (args.output / "memory.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Host memory and throughput on the local GPU (B6)", "",
             "Sequence length 1024, bf16 autocast, fused AdamW, attentive span channel (20k entries, 8,192 atomics, d=256).", "",
             "| Host | Mode | Checkpointing | Micro-batch | Peak GiB | Tokens/s |", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['host']} | {r['host_mode']} | {r['checkpointing']} | {r['micro_batch']} | "
                     + (f"{r['peak_gib']:.1f} | {r['tokens_per_s']:.0f} |" if r["ok"] else "OOM | — |"))
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    write_run_metadata(args.output, config, git_at_start=git_at_start, device="cuda")


if __name__ == "__main__":
    main()
