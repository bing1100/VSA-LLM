# R0 — Reproduction of committed results on `bhux-tiny`

**Date:** 2026-10-02. **Purpose:** the original experiment machine had faulty RAM (204 corrupted words in a 6 GiB user-space screen; 2,122 memtester failures; corrupted downloads and tokenizer panics, see [gates.md](../resources/plan-improvement/gates.md)). Before any committed result is used again, it was re-run on the replacement machine and compared cell by cell.

## Machine and environment

| Item | Value |
|---|---|
| Host | `bhux-tiny`: AMD Ryzen 5 8600G (12 threads), 60 GiB RAM, NVIDIA RTX 3090 24 GB |
| RAM screen | `scripts/memcheck.py 16 4` and `16 8`: **0 bad words** |
| Python env `vsa-repro` | Python 3.12.4, torch 2.11.0+cu128, transformers 4.54.0, numpy 2.2.6, nltk 3.8.1, PyYAML 6.0.2, safetensors 0.5.3, scipy 1.18.1, pyarrow 25.0.1, tokenizers 0.21.4, torchao 0.18.0 |
| Data | every probe set and corpus shard re-downloaded and checked against published counts and hashes (`~/data/vsa-llm/DATA_SOURCES.md`, `SHA256SUMS`) |

Re-runs were made at the commit recorded in each run's `manifest.json` (git worktrees), with outputs outside the repository, and compared with `scripts/compare_runs.py`.

## Results

| Run | Commit | Comparison | Outcome |
|---|---|---|---|
| E0 development (D0.1–D0.3, 3 seeds, CPU) | `2cc5d0b` | 8,100 metric cells + 154 summary cells, tolerance 1e-12 | **MATCH** (3 cells differ in the 17th digit at tolerance 0: float summation order) |
| D0.2 parent-sync (M3 default) | `9986d72` | 576 + 115 cells | **MATCH** |
| D0.2 parent-routing | `bf5f535` | 576 + 115 cells | **MATCH** |
| D0.2 parent-routing-v2 | `01e8090` (dirty) | 576 + 115 cells | **MATCH** |
| B6 host memory (3090) | HEAD | peak GiB per host × micro-batch (19 rows), tokens/s | **identical peak memory**, tokens/s within 1% |
| B8 probes (GPT-2, SmolLM2-135M/360M) | `e2078a4` | LAMBADA, WiC (prompt and probe), CARD-660 and RW Spearman | **within 0.0035** (LAMBADA and WiC identical) |
| B8 PTQ | HEAD | perplexity bf16 / INT8 / INT4 | **identical** |
| C5 judge calibration | v2 (new harness) | accuracy vs ontology gold, Fleiss κ | **30/30, κ 0.926** (v1: 30/30, κ 0.812) |

Two recorded commits had dirty trees whose fixes landed in later commits, so they were re-run at HEAD: B6 (`2ad45e5` crashes under transformers 4.54 because `GPT2LMHeadModel` does not accept `attn_implementation`) and B8 PTQ (`8e83d64` predates the full-precision output head of `4fae7cb`; at `8e83d64` GPT-2 INT4 perplexity is 2,874). This is a provenance gap in those two manifests, not a reproducibility failure: the committed numbers match the code that is now on `main`.

## Implications

- The G1 decision (mechanisms recover planted structure; M3 default `route_unobserved: parent`, `sync_parent: true`) rests on numbers that reproduce bit-for-bit on healthy hardware.
- The C4 throughput table used to plan the committed ≈ 400 GPU-hours holds on the new machine.
- The LLM judge is stable across harnesses (headless CLI vs Claude Code subagents): both reach 30/30 on the calibration set with κ ≥ 0.8. Individual verdicts are harness-specific, so each study records its harness.
- Partial E1/E2/E3/C3 outputs from the faulty machine are discarded; all are re-run under new run IDs.

**For the manuscript (reproducibility appendix):** single-threaded CPU experiments are bit-reproducible across machines; GPU measurements reproduce within 1% (throughput) and exactly (memory, PTQ perplexity); manifests record commit and dirty-tree state, and two dirty-tree manifests are disclosed here.
