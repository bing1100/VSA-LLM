"""WP-probe validation: the channel-probe suite on untouched hosts.

Runs `channel_probes.run_channel_probes` through the plain `ModelAdapter` (bf16 weights, exactly as
B8 loads hosts) on SmolLM2-135M and GPT-2. The probes B8 already ran (LAMBADA, WiC, CARD-660, Rare
Words) are compared with B8's committed `runs/v1/results.json` and, optionally, with B8 re-run on
this machine (`--machine-reference`, the R0 reproduction). WSD, BLESS and HyperLex are new and
reported as sanity references (published: SemCor MFS on Raganato ALL = 65.5 F1).

    python -m vsa_embed.experiments.probe_suite_validation --output experiments/wp-probe-validation/runs/v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from vsa_embed.evaluation.channel_probes import ProbeSettings, WORD_PREFIX, host_adapter, run_channel_probes, write_outputs
from vsa_embed.provenance import prepare_output_dir, write_run_metadata

B8_METRICS = {  # B8 results.json key → (probe, metric)
    "lambada_accuracy": ("lambada", "accuracy"), "lambada_word_loss": ("lambada", "word_loss"), "n": ("lambada", "n"),
    "wic_prompt_accuracy": ("wic", "prompt_accuracy"), "wic_probe_accuracy": ("wic", "probe_accuracy"),
    "wic_majority": ("wic", "majority"), "n_dev": ("wic", "n_dev"),
    "card660.spearman": ("card660", "spearman"), "rare_words.spearman": ("rare_words", "spearman"),
}


def _b8_value(probes: dict[str, Any], key: str) -> Any:
    value: Any = probes
    for part in key.split("."):
        value = value[part]
    return value


def compare_with_b8(host: str, results: dict[str, Any], references: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, (probe, metric) in B8_METRICS.items():
        ours = results[probe]["metrics"][metric]
        row: dict[str, Any] = {"host": host, "metric": key, "ours": ours}
        for name, reference in references.items():
            if host in reference and reference[host].get("probes"):
                value = _b8_value(reference[host]["probes"], key)
                row[name] = value; row[f"{name}_equal"] = ours == value; row[f"{name}_difference"] = ours - value
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probes-root", type=Path, default=Path("~/data/vsa-llm/probes").expanduser())
    parser.add_argument("--hosts", nargs="+", default=["HuggingFaceTB/SmolLM2-135M", "gpt2"])
    parser.add_argument("--reference", type=Path, default=Path("experiments/b8-probe-validation/runs/v1/results.json"))
    parser.add_argument("--machine-reference", type=Path, default=None,
                        help="B8 results re-run on this machine (R0), for bit-exact comparison")
    args = parser.parse_args(argv)
    git_at_start = prepare_output_dir(args.output)
    device = torch.device("cuda")
    settings = ProbeSettings()
    references = {"b8_committed": json.loads(args.reference.read_text())}
    if args.machine_reference:
        references["b8_rerun_this_machine"] = json.loads(args.machine_reference.read_text())
    summary: dict[str, Any] = {"hosts": {}, "b8_comparison": []}
    for host in args.hosts:
        started = time.monotonic()
        adapter = host_adapter(host, device)
        torch.cuda.reset_peak_memory_stats()
        results, tables = run_channel_probes(adapter, args.probes_root, settings=settings)
        seconds = round(time.monotonic() - started, 1)
        folder = args.output / host.replace("/", "__")
        write_outputs(folder / "probes.json", results, tables,
                      {"model": {"host": host, "weights": "bfloat16", "adapter": "ModelAdapter"},
                       "settings": {**settings.__dict__, "batch_size": adapter.batch_size, "max_length": adapter.max_length,
                                    "layer": adapter.layer, "word_prefix": WORD_PREFIX}})
        summary["hosts"][host] = {"seconds": seconds, "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                  "probe_seconds": {k: v["seconds"] for k, v in results.items()},
                                  "metrics": {k: v["metrics"] for k, v in results.items()}}
        summary["b8_comparison"] += compare_with_b8(host, results, references)
        print(json.dumps({host: summary["hosts"][host]["probe_seconds"]}), flush=True)
        del adapter; torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.output / "report.md").write_text(report(summary, references))
    write_run_metadata(args.output, {"probes_root": str(args.probes_root), "hosts": args.hosts, "settings": settings.__dict__,
                                     "references": {k: str(p) for k, p in (("b8_committed", args.reference),
                                                                           ("b8_rerun_this_machine", args.machine_reference)) if p}},
                       git_at_start=git_at_start, device=device)


def report(summary: dict[str, Any], references: dict[str, Any]) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)
    names = list(references)
    lines = ["# WP-probe validation: channel-probe suite on untouched hosts", "",
             "Plain `ModelAdapter`, bf16 weights, final layer, batch 32 (as B8). Full LAMBADA (5,153), WiC dev, "
             "Raganato ALL (7,253), BLESS noun pairs, HyperLex (2,616).", "",
             "## B8 probes", "", "| Host | Metric | This suite | " + " | ".join(f"{n} | equal" for n in names) + " |",
             "|---|---|---:|" + "---:|---|" * len(names)]
    for row in summary["b8_comparison"]:
        lines.append(f"| {row['host']} | {row['metric']} | {fmt(row['ours'])} | "
                     + " | ".join(f"{fmt(row.get(n))} | {'yes' if row.get(f'{n}_equal') else 'no'}" for n in names) + " |")
    if "b8_rerun_this_machine" in names:
        lines += ["", "`b8_committed` was produced on the replaced machine (faulty RAM); R0 found that its CARD-660/RW "
                  "Spearman and LAMBADA loss reproduce only to within 0.0035 even with B8's own code. "
                  "`b8_rerun_this_machine` is B8's code re-run here (R0); equality with it is bit-exact."]
    lines += ["", "## New probes", "",
              "| Host | WSD probe F1 | centroid F1 | MFS F1 | WN1 F1 | probe F1 (ambiguous) | MFS F1 (ambiguous) | "
              "BLESS probe acc | macro-F1 | relatum-only acc | pair probe acc | pair macro-F1 | majority | prompt MAP | "
              "prompt AUC | HyperLex cosine ρ | prompt ρ | probe ρ (test) |", "|---|" + "---:|" * 17]
    for host, entry in summary["hosts"].items():
        w, b, h = entry["metrics"]["wsd"], entry["metrics"]["bless"], entry["metrics"]["hyperlex"]
        lines.append(f"| {host} | {fmt(w['probe_f1'])} | {fmt(w['centroid_f1'])} | {fmt(w['mfs_f1'])} | {fmt(w['wn1_f1'])} | "
                     f"{fmt(w['ambiguous']['probe_f1'])} | {fmt(w['ambiguous']['mfs_f1'])} | {fmt(b['probe_accuracy'])} | "
                     f"{fmt(b['probe_macro_f1'])} | {fmt(b['relatum_only_accuracy'])} | {fmt(b['pair_probe_accuracy'])} | "
                     f"{fmt(b['pair_probe_macro_f1'])} | {fmt(b['majority_accuracy'])} | "
                     f"{fmt(b['prompt_map'])} | {fmt(b['prompt_auc'])} | {fmt(h['cosine_spearman'])} | "
                     f"{fmt(h['prompt_spearman'])} | {fmt(h['probe_spearman'])} |")
    lines += ["", "## Time", "", "| Host | total s | peak GiB | per probe (s) |", "|---|---:|---:|---|"]
    for host, entry in summary["hosts"].items():
        lines.append(f"| {host} | {entry['seconds']} | {entry['peak_gpu_gib']} | "
                     + ", ".join(f"{k} {v}" for k, v in entry["probe_seconds"].items()) + " |")
    lines += ["", "## Protocol", "",
              "- States: final layer, last subtoken of the word; isolated words are read after \"The word\" (as B8).",
              "- WSD: SemCor → Raganato ALL; per lemma#POS multinomial L2 logistic (l2 = 0.01, 200 L-BFGS steps; 400 gives the same predictions on GPT-2) on "
              "standardized states, at most 100 sampled SemCor occurrences per lemma#POS (seed 0); one seen sense → "
              "that sense; unseen lemma#POS → WordNet first sense (as MFS). Centroid = nearest sense centroid (cosine). "
              "MFS = most frequent SemCor sense, ties by WordNet order.",
              "- BLESS: noun pairs of {hyper, coord, mero, random-n}; 4-way probe on [h_x; h_y], concept-disjoint split "
              "(30% of the 200 concepts, seed 0); relatum-only control (h_y); pair probe on [|h_x − h_y|; h_x ⊙ h_y]; prompting = PMI of \"The x is a kind of y\" vs "
              "\"The thing is a kind of y\" (per-concept AP of hypernyms; pooled AUC hyper vs rest).",
              "- HyperLex: cosine and PMI (verbs: \"To x is a way to y\" vs \"To do …\") on all 2,616 pairs; ridge on "
              "[h_x; h_y] trained on the lexical split's train part, α from {1, …, 10⁴} on its dev part, Spearman on its test part.",
              "- Spearman for the new probes and all subset/paired statistics uses average ranks for ties; CARD-660/RW "
              "`spearman` keeps B8's ordinal-rank definition.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
