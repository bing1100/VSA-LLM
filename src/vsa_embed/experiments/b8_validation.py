"""B8 acceptance: probes on untouched hosts and quantized-perplexity retention.

GPT-2's LAMBADA-OpenAI last-word accuracy is a published reference (≈ 0.325 in lm-evaluation-harness);
SmolLM2-135M and 360M are reported for later comparison. Perplexity on LAMBADA passages is measured
in bf16, INT8 and INT4 weight-only form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.evaluation.probes import ModelAdapter, run_probes
from vsa_embed.evaluation.quantization import quantize_weight_only
from vsa_embed.provenance import prepare_output_dir, write_run_metadata


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], device: torch.device) -> float:
    total, count = 0.0, 0
    for text in texts:
        ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"][:512]], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(input_ids=ids, labels=ids).loss
        total += float(loss) * (ids.shape[1] - 1); count += ids.shape[1] - 1
    return float(torch.exp(torch.tensor(total / count)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes", type=Path, default=Path("~/data/vsa-llm/probes").expanduser())
    args = parser.parse_args(argv)
    git_at_start = prepare_output_dir(args.output)
    device = torch.device("cuda")
    lambada_path = args.probes / "lambada" / "data" / "lambada_test_en.jsonl"
    ppl_texts = [json.loads(l)["text"] for l in lambada_path.read_text().splitlines()[:300]]
    results = {}
    for name in ("gpt2", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"):
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
        adapter = ModelAdapter(model, tokenizer, device)
        entry = {"probes": run_probes(adapter, args.probes, lambada_limit=None), "ppl_bf16": perplexity(model, tokenizer, ppl_texts, device)}
        for bits in (8, 4):
            quantized = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
            try:
                quantize_weight_only(quantized, bits)
                entry[f"ppl_int{bits}"] = perplexity(quantized, tokenizer, ppl_texts, device)
            except Exception as error:  # report, do not hide
                entry[f"ppl_int{bits}"] = f"error: {type(error).__name__}: {str(error)[:200]}"
            del quantized; torch.cuda.empty_cache()
        results[name] = entry
        print(json.dumps({name: entry}, default=str), flush=True)
        del model; torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    lines = ["# B8 probe and quantization validation", "",
             "| Host | LAMBADA acc | WiC prompt | WiC probe | WiC majority | CARD-660 ρ | RW ρ | PPL bf16 | PPL INT8 | PPL INT4 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)[:30]
    for name, e in results.items():
        p = e["probes"]
        lines.append(f"| {name} | {fmt(p['lambada_accuracy'])} | {fmt(p['wic_prompt_accuracy'])} | {fmt(p['wic_probe_accuracy'])} | "
                     f"{fmt(p['wic_majority'])} | {fmt(p['card660']['spearman'])} | {fmt(p['rare_words']['spearman'])} | "
                     f"{fmt(e['ppl_bf16'])} | {fmt(e['ppl_int8'])} | {fmt(e['ppl_int4'])} |")
    lines += ["", "Reference: GPT-2 small LAMBADA-OpenAI accuracy ≈ 0.325 (lm-evaluation-harness).", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    write_run_metadata(args.output, {"probes_root": str(args.probes)}, git_at_start=git_at_start, device=device)


if __name__ == "__main__":
    main()
