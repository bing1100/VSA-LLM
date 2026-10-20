# 16 — Infrastructure and validation: the work that makes the results trustworthy

**Overview.** This file covers the supporting work behind the results rather than the results themselves. It has ten
explorations. 16.1 shows that committed results reproduce on the replacement machine after the first machine's memory
failed. 16.2 records the first sanity pilot. The pilot justified the GPU budget, but it used an under-trained recipe and
is superseded. 16.3–16.7 cover the benchmark and validation work packages: compute cost (B6, B10), the evaluation
harness (B8 and the probe-suite validation), the general corpus and its frozen holdout (C3), the LLM-judge calibration
(C5) and the developer-tools benchmark (C6). Most of these were built between 30 September and 2 October 2026 and are
done. 16.8–16.9 cover the two packages of the read–learn–write toolkit (decision 63, 8 October). Both are built, and
neither has a model result yet. 16.10 describes the job queue and the pre-registration practice that every experiment
runs under.

**Terms used throughout.**
- **Work-package codes** (B6, B8, B10, C3, C5, C6) come from `tasks.md`. Three of them collide with **condition codes**
  (the model arms). We therefore write "the C3 corpus", "the C5 judge calibration" and "the C6 benchmark" for work
  packages, and "condition C5" for a model arm.
- **Conditions:** C0 no channel; C0′ continued training without the channel; C1 random fixed span vectors; C2 a free
  per-concept lookup table; C3 a static vector-symbolic (VSA) bundle; C5 the attentive VSA channel (the method);
  P0 the untouched pretrained host.
- **Channel:** the module that adds a composed concept vector at the last subtoken of each linked multi-token term.
  **Frame:** a term's set of (relation, filler) facts. **HRR:** holographic reduced representations, the default
  binding operator.
- **Holdout:** concepts never linked in training, frozen and hashed (sha256) before any run that could see them.
  **ℓ_min:** the minimum number of subtokens for a term to be linked.
- **Manifest:** the `manifest.json` each run writes: commit hash, dirty-tree flag, package versions, device, threads,
  command line and config hash.
- **Statistics:** \* marks significance after Holm correction; brackets are 95% intervals.

---

# 16.1 — R0: committed results reproduce on the replacement machine

**Status:** done · **When:** 2026-09-30 (fault found) to 2026-10-02 (report) · **Sources:** `reports/R0-reproduction.md`;
`resources/plan-improvement/gates.md` (BLOCKER section); `resources/plan-improvement/runbook.md` Steps 1–5;
`manuscript/reproducibility.md` §5, §7

## Research gap
The first experiment machine had faulty RAM. A 6 GiB screen found 204 corrupted 64-bit words, and memtester reported
2,122 failures. The same days brought corrupted downloads, tokenizer panics and processes that died without a
traceback. Every result computed there was suspect, including the E0 numbers behind gate G1.

## Intuition
The corruption was intermittent: a single-threaded replay of 1,944 metric rows was bit-identical. Most results were
therefore probably right, but a gate cannot rest on "probably". Re-running each committed result on healthy hardware and
comparing it cell by cell either clears it or flags it.

## Methodology
- New machine `bhux-tiny`: AMD Ryzen 5 8600G, 60 GiB RAM, RTX 3090 24 GB. RAM screen (`memcheck.py 16 4` and `16 8`):
  0 bad words.
- We re-downloaded every probe set and corpus shard and checked it against published counts and hashes.
- We re-ran each result at the commit its manifest records (git worktrees, outputs outside the repository) and compared
  with `scripts/compare_runs.py`.
- Tolerances were fixed in the runbook beforehand: E0 and D0.2 at 1e-12; B6 peak memory within ±5%; B8 probes within
  ±0.005; post-training quantization (PTQ) perplexity within ±1%; judge accuracy ≥ 0.90 and Fleiss κ ≥ 0.6.

## Experiment
Eight comparisons: E0 development (3 seeds, CPU), three D0.2 variants, B6 host memory, B8 probes, B8 PTQ and the C5
judge calibration.

## Results
| Run | Compared | Outcome |
|---|---|---|
| E0 development (`2cc5d0b`) | 8,100 metric + 154 summary cells at 1e-12 | **MATCH** (3 cells differ in the 17th digit at tolerance 0: summation order) |
| D0.2 parent-sync, parent-routing, parent-routing-v2 | 576 + 115 cells each | **MATCH** |
| B6 host memory | peak GiB in 19 rows; tokens/s | identical memory; tokens/s within 1% |
| B8 probes (`e2078a4`) | LAMBADA, WiC, CARD-660, Rare Words | LAMBADA and WiC accuracy identical; Spearman within 0.0035 |
| B8 PTQ | perplexity bf16 / INT8 / INT4 | identical (at HEAD) |
| C5 judge calibration | accuracy; Fleiss κ | 30/30, κ 0.926 (v1: 30/30, κ 0.812) |

We lifted the blocker on 2026-10-02. Partial E1, E2, E3 and C3 outputs from the old machine were discarded and re-run
under new run IDs.

## Limitations
- Two manifests (B6 `2ad45e5`, B8 PTQ `8e83d64`) record dirty trees whose fixes landed later. At those commits B6
  crashes under transformers 4.54, and GPT-2 INT4 perplexity is 2,874. We re-ran both at HEAD. R0 calls this a
  provenance gap, not a reproducibility failure.
- GPU training is not bit-deterministic (index-add backward uses atomics), so GPU comparisons need tolerances.
- Individual judge verdicts differ between harnesses; only agreement statistics are comparable.
- R0's table calls LAMBADA "identical". The probe-suite report (16.4) shows that LAMBADA *accuracy* is identical but
  LAMBADA *loss* reproduces only within 0.0035.
- R0 covers results committed by 2 October. Everything later was produced on `bhux-tiny` and has not been re-run.

## Next steps
None for R0. `manuscript/reproducibility.md`, which summarizes R0, needs refreshing (16.10).

---

# 16.2 — R1: the first sanity pilot (superseded)

**Status:** done (superseded) · **When:** 2026-10-02 (report); superseded by the D4.0 recipe sweep, 2026-10-03/04 ·
**Sources:** `reports/R1-sanity-pilot.md`; `resources/plan-improvement/gates.md` (D4.0 sections);
`manuscript/update-2026-10-12.md` ("Reproduction and early pilots", E4-1); `manuscript/update-2026-10-15.md` §7;
`manuscript/paper-readiness.md`

## Research gap
Before spending the committed ≈ 400 GPU-hours, the author asked whether any signal appears at all (decision of
2026-10-02). The pilot was never a gate and is never pooled with gate runs.

## Intuition
A few GPU-hours on small models can show whether composed ontology vectors lower the loss where they should: on linked
text, and most for concepts the model never saw linked in training.

## Methodology
- **From scratch:** a GPT-2-style 50M model, 100M tokens of the C3 corpus, ℓ_min = 2, the frozen C3 holdout (5,900
  entries). Conditions C0, C1, C2, C3, C5; seeds 1 and 2.
- **Continued pretraining (CPT):** SmolLM2-135M with LoRA (low-rank adaptation) r = 16, 25M tokens; C0′, C2, C5;
  seed 1.
- Loss by stratum relative to C0, cluster bootstrap over 1,024-token windows (10,000 resamples), Holm within stratum.
  Seven probes, each split into held-out, seen and unlinked words.

## Experiment
13 runs on the RTX 3090: about 4.2 GPU-h from scratch and 1 GPU-h CPT.

## Results
From scratch, condition C5 relative to C0 (negative = lower loss):
- after a linked word −0.56% [−0.60, −0.51]\*; inside multi-token words −1.32% [−1.45, −1.19]\*;
- after held-out concepts −0.15% [−0.26, −0.04]\*; against C2 there −0.13% (not significant after Holm);
- on unlinked text +0.12%\* (C3: +0.28%\*), the locality cost;
- random span vectors (C1) raise the loss everywhere (+0.26–0.30%).

Probes (mean of 2 seeds):
- CARD-660 Spearman on held-out concepts (n = 36): C0 0.102, C2 0.092, C3 0.311, C5 0.326.
- BLESS pair-probe accuracy on held-out concepts: C0 0.621, C3 0.680, C5 0.684. Unlinked words do not change.

CPT: every difference is below 0.06%, but training barely moved the loss (2.798 → 2.791). R1 calls this
"uninformative, not negative".

## Limitations
- **Under-trained recipe.** The D4.0 sweep (2026-10-03) found the pilot recipe (262k tokens per step, lr 1e-3)
  1.3 nats worse than the best setting at the same tokens. On 2026-10-04 we froze 32,768 tokens per step and lr 2e-3
  (final loss 4.22 vs 5.63). gates.md: the pilot "is superseded as an effect-size estimate".
- C1 had about 1k trainable parameters, so it was not parameter-matched.
- Two seeds; every channel run used HRR, so binding was not tested.
- Later from-scratch runs did not reproduce the held-out loss signal cleanly. The 3-seed operator screen (50M,
  100M tokens) gave C5 − C0 −0.37%\* after held-out terms. The seed-1 scaling screen (500M tokens) gave +0.02%,
  −0.05% and +0.25% at 20M, 50M and 125M.
- update-2026-10-12 says the pilot helped most on "unseen concepts". R1's stratum is *held-out* concepts; later reports
  use "unseen" for a different stratum (terms with no training occurrence).

## Next steps
None: R1 stays labelled exploratory (paper-readiness: "ready (labelled exploratory)"). Its decisions table led to the
D4.0 recipe sweep (done), the E9 engagement check that replaced the CPT setup (done), CARD-660 as the primary
similarity probe (BLESS overlaps the WordNet hypernymy the channel injects), and a parameter-matched C1 for G2
(G2 not yet held).

---

# 16.3 — B6 and B10: the compute cost of hosts and of the channel

**Status:** done · **When:** B6 2026-09-30, re-run 2026-10-02; B10 2026-10-02 · **Sources:**
`experiments/b6-host-memory/runs/3090-v1/report.md` and `manifest.json`; `experiments/b10-benchmarks/generation.md`
and `generation.json`; `resources/plan-improvement/experiments.md` (C4 paragraph); `resources/plan-improvement/tasks.md`
(B6, B10, C4); `manuscript/claims.md` (B.1); `manuscript/paper-readiness.md`

## Research gap
Two cost questions. First, which hosts and micro-batches fit the 3090's ≈ 22 GB budget (the GPU also drives the
desktop), and how fast do they train? The plan's GPU-hours rested on an assumed 2.5e13 FLOP/s. Second, does the
channel's cost grow with the ontology? If every step had to materialize the whole concept table, large ontologies
would be unaffordable.

## Intuition
Measured tokens per second replace the planning guess. A batch needs only the rows of the concepts it links, so
composing only those ("sparse") should cost the same whatever the table size N.

## Methodology
- **B6** (with task C4, the throughput table): sequence 1,024, bf16 autocast, fused AdamW, attentive channel attached
  (20k entries, 8,192 atomics, d = 256). We record peak GiB and tokens/s per host × mode × micro-batch, with and without
  activation checkpointing.
- **B10:** sparse (compose the batch's linked entries) against full (compose all N entries, then gather). Both give the
  same rows to 1e-4. N ∈ {10k, 50k, 100k}; 256, 1,024 or 4,096 spans per batch, drawn Zipf(1.1); forward and train
  steps; CPU with 4 threads, GPU capped at 1.5 GiB. WP-B10 also added a CPU CI workflow and seeded property tests.

## Experiment
B6: 19 configurations over six hosts (scratch 50M, 125M, 350M; SmolLM2-135M, -360M; Qwen2.5-0.5B). B10: three modes
× three N × three batch sizes, on CPU and GPU.

## Results
B6 (selected rows):

| Host | Mode | Micro-batch | Peak GiB | Tokens/s |
|---|---|---:|---:|---:|
| scratch-50M | train | 32 | 18.8 | 71,220 |
| scratch-125M | train | 16 | 18.0 | 35,789 |
| scratch-125M | train, checkpointing | 32 | 9.9 | 28,784 |
| scratch-350M | train, checkpointing | 16 | 10.1 | 11,024 |
| SmolLM2-135M | frozen / LoRA | 16 / 8 | 12.8 / 9.5 | 36,430 / 26,450 |
| SmolLM2-360M | frozen | 8 | 11.5 | 18,564 |
| Qwen2.5-0.5B | frozen | 4 | 11.4 | 15,995 |

Every host fits at some micro-batch; the largest peak is 18.9 GiB. One configuration (SmolLM2-360M, frozen,
checkpointing, micro-batch 16) ran out of memory. From these numbers experiments.md derives run times: 50M × 300M
tokens ≈ 1.2 h, 125M × 500M ≈ 3.9 h, and ≈ 63 h per run for the 350M × 2.5B-token tier.

B10, CPU, N = 100,000, 1,024 spans per batch, sparse against full:

| Mode | Forward latency | Forward memory | Train latency | Train memory |
|---|---|---|---|---|
| bundle (M0) | 208.3× lower | 218.8× lower | 163.7× lower | 117.9× lower |
| attentive (M1) | 284.5× lower | 224.4× lower | 115.5× lower | 96.5× lower |

Sparse peak memory does not depend on N. On the GPU, the full step no longer fits the 1.5 GiB cap from N = 50,000,
while the sparse train step at N = 100,000 needs 41 MiB (bundle) and 58 MiB (attentive). Claim B.1 is "supported
(engineering)".

## Limitations
- B6's manifest records a dirty tree (`2ad45e5`); R0 re-ran it at HEAD with identical memory.
- B6 measures a 20k-entry channel on the hosts of September; later hosts (Qwen3, Qwen3.5) needed their own memory
  probes.
- B10's GPU latencies are qualitative: another job held 13.7 GiB of the GPU at 100% utilization.
- Sparse memory has an N-independent floor (8 MiB forward, 40 MiB train) because the whole atomic table is normalized
  before the gather. The report describes the fix but leaves `compose.py` unchanged.
- paper-readiness.md (2026-10-03) says B10's JSON "records no git state". The JSON's `machine` block does record commit
  `05073c3`, clean, and reproducibility.md agrees with the JSON.

## Next steps
None planned. If claim B.1 enters the paper, cite `generation.json` with its commit.

---

# 16.4 — B8 and the probe-suite validation: is the evaluation harness right?

**Status:** done · **When:** B8 runs 2026-09-30/10-01, reproduced 2026-10-02; probe-suite validation 2026-10-02 ·
**Sources:** `experiments/b8-probe-validation/runs/v1/report.md`, `v1/NOTE.md`, `v1-ptq/report.md`;
`experiments/wp-probe-validation/runs/v1/report.md` and `manifest.json`; `resources/plan-improvement/runbook.md` Step 3;
`resources/plan-improvement/queue.md` (row 24); `reports/R1-sanity-pilot.md`

## Research gap
Every behavioural claim and every quantization claim rests on the evaluation harness. Gate item H-A.7, for example,
needs gains on at least two of WiC, rare-word similarity, word-sense disambiguation (WSD) and a domain task. A probe
that mis-scores a known model, or a quantizer that breaks perplexity, would contaminate all of them.

## Intuition
Score untouched public models whose numbers are known. If the harness matches published values and gives sensible
above-chance scores, we can trust it on modified models.

## Methodology
- **B8:** GPT-2, SmolLM2-135M and SmolLM2-360M. LAMBADA accuracy (reference for GPT-2 ≈ 0.325,
  lm-evaluation-harness); WiC (Word-in-Context) as a prompt and as a linear probe; CARD-660 and Stanford Rare Words (RW)
  Spearman; torchao weight-only INT8 and INT4 PTQ with the output head in full precision.
- **Probe-suite validation (WP-probe):** the channel-aware suite re-implements the B8 probes, which must equal B8's
  re-run bit for bit. It adds WSD (trained on SemCor, tested on Raganato ALL), BLESS and HyperLex. States come from the
  final layer at the word's last subtoken, bf16.

## Experiment
B8 `v1` (probes) and `v1-ptq` (quantization). WP-probe `v1` on SmolLM2-135M and GPT-2 (103.5 s and 72.8 s per host).

## Results
B8 probes:

| Host | LAMBADA | WiC prompt | WiC probe | CARD-660 ρ | RW ρ |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 0.310 | 0.552 | 0.563 | 0.084 | 0.107 |
| SmolLM2-135M | 0.428 | 0.596 | 0.577 | 0.215 | 0.262 |
| SmolLM2-360M | 0.532 | 0.610 | 0.582 | 0.168 | 0.208 |

The WiC majority baseline is 0.500. PTQ perplexity, bf16 / INT8 / INT4: GPT-2 37.279 / 37.270 / 41.685; SmolLM2-135M
26.948 / 27.354 / 36.971; SmolLM2-360M 22.456 / 22.679 / 27.206. queue.md summarizes this as INT8 ≤ 1.5% and INT4
+12–37%.

WP-probe reproduces every B8 metric of the `bhux-tiny` re-run exactly. New probes, SmolLM2-135M / GPT-2:
- WSD probe F1 0.7055 / 0.7010, against 0.6546 for the most-frequent-sense baseline;
- BLESS 4-way probe 0.8905 / 0.9023; relatum-only control 0.8942 / 0.9012; majority 0.4640;
- HyperLex: cosine ρ 0.0118 / 0.0139, prompt ρ 0.3887 / 0.3168, probe ρ 0.3737 / 0.2754.

## Limitations
- GPT-2 LAMBADA is 0.310 against the ≈ 0.325 reference; no source explains the gap.
- B8 `v1`'s INT4 column failed (a missing `mslk` package), and its GPT-2 INT8 quantized the tied output head. `v1-ptq`
  supersedes both.
- Both B8 manifests record dirty trees. The committed `v1` probe numbers come from the faulty machine; where they
  differ we use the `bhux-tiny` re-run (SmolLM2-135M CARD-660: 0.215 committed, 0.2111 re-run).
- The BLESS relatum-only control scores as high as the full 4-way probe, so that probe mostly reads the relatum. R1
  separately makes CARD-660 the primary similarity probe because BLESS overlaps WordNet hypernymy.
- Validation covers untouched hosts and the final layer only.

## Next steps
None. The suite is the probe step of later runs: R1 used it, and E9 run folders carry its `probes.json`.

---

# 16.5 — C3: the general corpus and its frozen holdout

**Status:** done · **When:** 2026-09-30 (`v1`, faulty machine, stale); 2026-10-02 (`v2` and host corpora) ·
**Sources:** `experiments/c3-general-corpus/runs/v2/report.md` and `summary.json`; `runs/host-smollm2-v1/report.md`;
`runs/host-qwen2.5-v1/report.md`; `runs/v1/STALE.md`; `manuscript/reproducibility.md` §3–4;
`resources/plan-improvement/tasks.md` (C3, rules); `resources/plan-improvement/execution.md` (E9 engagement check;
known linker limitation)

## Research gap
Every from-scratch experiment and the general WordNet track need one shared linked corpus and one frozen list of
held-out concepts. Changing the holdout later would invalidate every earlier result (tasks.md rule 2). The corpus must
also cover the committed 300M- and 500M-token runs.

## Intuition
Link FineWeb-Edu text to WordNet once, with a frozen linker. Hash the holdout before any run. Then re-link the same
holdout under each host's tokenizer, so that every host sees the identical split.

## Methodology
- FineWeb-Edu `sample/10BT` shards 000 and 002, hash-verified; shard 001 failed its sha256 twice on the faulty machine.
- WordNet 3.0: 117,659 concepts become 101,500 linker entries, with 8,192 atomics and 16 relations.
- Holdout: 4,105 concepts / 5,900 entries (2,800 chosen from 27,973 eligible, plus closure), node- and alias-disjoint,
  sha256 `7f2462ed…`.
- Span-cardinality tables per tokenizer × ℓ_min ∈ {1, 2, 3, 4}.
- Host corpora for the SmolLM2 and Qwen2.5 tokenizers, checked against the C3 ontology, holdout and alias tables.

## Experiment
CPU builds on `bhux-tiny`: GPT-2 corpus `runs/v2` (manifest commit `724219a`), SmolLM2 (`57e4e23`) and Qwen2.5
(`ee184d2`).

## Results
- GPT-2: 1,100,000,492 training tokens (1,065,634 documents, 0 skipped), a 300,000,163-token slice, and 4,945,038
  evaluation tokens (5,000 documents) holding 576,226 held-out spans.
- SmolLM2 and Qwen2.5: 130,000,011 and 130,000,450 training tokens; 0 held-out spans in training; holdout and alias
  tables identical to C3.
- At ℓ_min = 2 (GPT-2, document sample): 24,549 linked entries, covered-token fraction 0.109, 23.9 distinct entries
  per 1,024 tokens. At ℓ_min = 3 the fraction falls to 0.036.
- The corpus fed R1, the from-scratch E4 screens and the E9 WordNet track; queue.md also lists it as an input of E1
  and the D4.0 recipe sweep.

## Limitations
- WordNet vocabulary is not new to pretrained hosts. In the E9 engagement check (SmolLM2-360M, 10M tokens), condition
  C5 ended identical to C0′ to four decimals, with inside-span loss 0.92 nats. WordNet therefore serves as the
  negative-control track for retrofit experiments.
- The shared span linker matches a name at the start of a longer word ("Javan" in "Javanese") and shifts matches after
  some Unicode characters (found by TK-B2, 2026-10-08). It is not fixed for existing tracks, because a fix would change
  the spans of finished runs. C5 − C0′ contrasts use identical spans in both arms.
- `v1`, from the faulty machine, is stale and never cited.

## Next steps
An opt-in strict-boundary linker for future builds is open (execution.md). Nothing else is planned.

---

# 16.6 — C5: calibrating the LLM judge

**Status:** done (calibration only) · **When:** v1 2026-09-30; v2 2026-10-02 · **Sources:**
`experiments/c5-judge-calibration/v1/summary.json`, `v2/summary.json`, `v2/manifest.json`, `v{1,2}/results.json`;
`src/vsa_embed/judge_protocol.py` (docstring); `resources/plan-improvement/runbook.md` Step 4;
`resources/plan-improvement/gates.md`; `manuscript/claims.md` (J.1); `resources/plan-improvement/execution.md`
(decision 3 of 2026-10-02; open decision 22); `manuscript/paper-readiness.md`

## Research gap
Two studies need human-style ratings: E5.3 (explanations) and E7.1 (self-authored edges). The plan replaced rater
recruiting with an LLM judge, Claude pinned to `claude-opus-5-5`, with a clinician study later on the same items. An
uncalibrated judge would make every rating uninterpretable.

## Intuition
Before it rates anything unknown, the judge grades items with known answers. If it agrees with the gold and with
itself across calls and paraphrases, its ratings are at least consistent.

## Methodology
- Rubric `neighbours`: how related is a list of candidate concepts to a concept, on a 0–3 scale?
- 30 WordNet items (seed 0) with ontology gold: 15 related (gold 3) and 15 random (gold 0).
- Three calls per item over two prompt paraphrases, blinded and position-randomized. A majority of ≥ 2 counts as
  "related".
- Pass rule: accuracy ≥ 0.90 (≥ 27/30) and Fleiss κ ≥ 0.6 (agreement across calls beyond chance).
- Two harnesses. v1 calls headless `claude -p`. v2 gives each of the 90 requests to a fresh Claude Code subagent; a
  request carries no item id, gold label or system name.

## Experiment
Two calibrations on identical items.

## Results
| Harness | Accuracy | Fleiss κ | Spend |
|---|---|---:|---|
| v1, headless `claude -p` | 30/30 | 0.812 | USD 2.60 |
| v2, Claude Code subagents | 30/30 | 0.926 | not recorded |

Both pass. Claim J.1 is "supported (calibration only; ratings are LLM-graded estimates)". Later studies may use either
harness, and each records which.

## Limitations
- The items are easy: gold is only 0 or 3, so the middle of the scale is untested.
- Only `neighbours` is calibrated. The protocol also defines `edge_explanation`, `split_card` and `authored_edge`, and
  its docstring plans an `authored_edge` calibration (gold against corrupted edges). None of these is committed.
- The plan asked for an author-labelled set of ≈ 30 items. All 30 items are ontology-derived.
- Individual verdicts are harness-specific; v1 has no manifest.
- Validation calls of the E5.3 rubrics cost $0.027–0.061 each (mean $0.039), about twice the planned $0.02.
- We found no committed result from E5.3 or E7.1, so the judge has not yet rated a real study.

## Next steps
The protocol requires a known-answer calibration before each study, so `authored_edge` and the E5.3 rubrics need one
before they run. The clinician study (W4) will rate the identical cached items.

---

# 16.7 — C6: the developer-tools benchmark

**Status:** done (data); no model results · **When:** 2026-09-30 · **Sources:**
`experiments/c6-devtools-benchmark/v1/summary.json`, `library.json`, `probes.jsonl`;
`src/vsa_embed/benchmarks/devtools.py` (docstring); `experiments/t2-developer-tools/README.md`;
`experiments/e5-explainability/preregistration.md`; `experiments/e7-self-authoring/runs/devtools-v1/report.md`;
`manuscript/reproducibility.md` §3–4; `manuscript/paper-readiness.md`; `resources/plan-improvement/queue.md`

## Research gap
Real vocabularies leak: a pretrained host may already know the terms. Zero-shot claims need vocabulary that no host can
have seen. The symbols of an invented software library are such vocabulary, and their API schemas are natural frames.

## Intuition
Generate a private library with pronounceable invented names. Write documentation that uses each symbol consistently
with its schema, and keep some symbols out of training. Any zero-shot skill on the held-out symbols must then come from
the composed schema.

## Methodology
- Generator (seed 20261001): modules, classes, functions, exceptions and constants; names checked against WordNet
  lemmas; frames over 8 relations (kind, belongs_to, returns, takes, raises, calls, inherits, category).
- A documentation corpus of API reference entries, tutorials, changelog notes and Q&A snippets. Held-out symbols appear
  only in evaluation documents and probes.
- Cloze probes whose answer is a schema field (module or return type).
- A leakage audit counts held-out symbols in training documents.

## Experiment
One CPU build, `v1`.

## Results
- 414 symbols in 12 modules; 2,950 training and 2,018 evaluation documents.
- 57 held-out symbols; leakage audit 0.
- 726 cloze probes, 101 of them on held-out symbols.
- Downstream uses so far: the E5.4 item set `c6-devtools-v1` (57 held-out symbols, 215 prompts); the E7 dev-tools
  reading corpus (D7.0: 64 masked concepts, the 57 held-out aliases excluded); and the T2 track's scaled generator
  (16 libraries, 11,750 private symbols), which still regenerates C6 v1 bit-identically.
- **No model has been evaluated on C6.** We found no committed result on its items.

## Limitations
- No manifest and no holdout hash; reproducibility.md lists the hash as an open item.
- C6 symbols are snake_case. The default alias normalization turns `_` into a space, so such aliases never match code.
  Until open decision 17 is resolved, only the CamelCase class symbols link (E5 pre-registration). T2 added an opt-in
  `identifier` normalization for this.
- It is small: 57 held-out symbols and 101 held-out probes.

## Next steps
Hash the holdout. Then the E5.4 zero-shot scenario `c6_devtools` and track D8.2 (T2), which wait for the D4.1 and G3
checkpoints.

---

# 16.8 — The toolkit's write benchmarks (toolkit-bench) and the methodology behind them

**Status:** queued (pre-registered; CPU smoke only) · **When:** methodology and pre-registration 2026-10-08; queued
2026-10-08; expected ≈ 2026-10-21–22 · **Sources:** `manuscript/toolkit-methodology-2026-10.md`;
`experiments/toolkit-bench/preregistration.md`; `experiments/toolkit-bench/items/*/manifest.json`;
`resources/plan-improvement/execution.md` (decision 63; "Toolkit implementation"); `manuscript/update-2026-10-15.md`
("Next experiments")

## Research gap
Decision 63 (author, 2026-10-08) frames the programme as a toolkit on one store of composed concept vectors: *read*
(decode a term's relations), *learn* (turn passive training into explicit relations), *write* (add a never-seen word
from its definition, with no gradient step) and a *meta* layer of tool calls. No public benchmark measured any tool.
Reviewers will ask how the store does on standard new-word benchmarks, and against the same information in the prompt.

## Intuition
Score public new-word benchmarks by log-likelihood ranking, which works for small hosts. Write the nonce word's frame
into the store; compare with no row, a random frame, and the same information as text in the context.

## Methodology
The methodology document (§2) sets six updates for every tool. M1 adds real-world holdouts, because on real tracks
held-out terms still occurred, unlinked, in the training text: H1 T7-ROOD (every training document that mentions a
held-out MeSH record is dropped), H2 T1c-ROOD (licensed clinical track) and H3 ICD-10-CM FY2027 codes (16.9). M2–M6:
one store API, a null-calibrated acceptance test (5% false-discovery ceiling), shared baselines that always include
the same information as text, a Wikidata-framed track (T8) and Qwen3-1.7B on real text.

The toolkit-bench pre-registration (work package TK-B1):
- **Sets:** COMPS-WUGS (13,896 minimal pairs, 4 nonce words) and the ranking forms of ALCUNA (5,000 items; 1,429 of
  3,554 entities map to WordNet).
- **Conditions:** no row (`none`); channel off; the store with a frame from one of six readers (oracle, parent, random,
  type prior, linker, joint linker); the frame or the definition in context; store plus definition.
- **Primary endpoints** (Holm): W1 = `store:oracle − none` and W2 = `store:linker − none`, COMPS-WUGS, SmolLM2-360M
  condition C5. Paired bootstrap over items; minimal detectable effect ≈ 0.012.
- 17 secondaries (S17 added before any run), decision readings (a)–(d) and refutation readings R1–R5, all pre-written.
- **Expected:** W1 and W2 near 0, because the checkpoints are the E9 WordNet track, where the channel was null.

## Experiment
16 GPU jobs (2 hosts × 4 models × 2 sets) on the E9 WordNet seed-1 checkpoints, ≈ 3.0 GPU-h, queued at priority 54.498,
with the report at 54.4981. A 40-item CPU smoke on SmolLM2-135M checked the pipeline.

## Results
No full run yet; no output folder exists as of 2026-10-20. The smoke (40 items, one seed, labelled "not evidence"):
- COMPS-WUGS: `none` 0.500; `store:oracle` 0.525 (+0.025 [−0.126, +0.176]); `store:linker` 0.412; frame in context
  0.700 (+0.200 [+0.025, +0.325]); definition in context 0.675 (+0.175 [+0.050, +0.325]).
- ALCUNA: `store:oracle` +0.000; definition in context +0.250 [+0.049, +0.450].
- The linker reader's edge precision was 0.05, against 0.85 for the type prior; this motivated S17.

The same package built the item sets for the T8 track: Entity Inferences 170, reversal 1,200, LRE relations 11,089,
BEAR 23,193 and PopQA 14,267 items.

## Limitations
- This is a negative-control track: even a clean null says nothing about new vocabulary (T5, T7, T8 and E13 cover that).
- One seed; the WordNet track has no seeds 2–3.
- 620 COMPS pairs (4.5%) get identical frames on both sides, and only 224 of 521 parent concepts (43%) map to an exact
  atomic; the rest use an ancestor. ALCUNA covers 40% of entities.
- The methodology proposed building only after two gating results (E12 recall and T7, each at 3 seeds). Decision 63
  instead queued every block once its harness passed tests and a smoke.

## Next steps
The jobs are expected ≈ 21–22 October (update-2026-10-15). The *read* and *meta* benchmarks run on T8 once its
training runs exist (T8 expected ≈ 23 October).

---

# 16.9 — The toolkit's learn data and holdout H3 (toolkit-learn)

**Status:** partly done (data built; no model run; evaluation queued) · **When:** 2026-10-08 · **Sources:**
`experiments/toolkit-learn/README.md` and `items/*/manifest.json`; `manuscript/toolkit-methodology-2026-10.md`
(§0, §2 M1, §3.2); `resources/plan-improvement/execution.md` (TK-H3L and TK-L rows)

## Research gap
*Learn* is the weakest tool. Synthetic recovery works, but rule mining (AMIE) matches it, its acceptance test accepts
structure in worlds that have none, and transfer to WordNet fails (R10). It also had no real-data gold: no set of
genuinely new concepts with a known place in an ontology.

## Intuition
Ontologies are versioned. A record that is new in a later release has a known parent, and a model built on the earlier
release has never seen it as an entry. Placing new records into the old release therefore gives real gold. Codes
created after every host's training data, such as ICD-10-CM (the US clinical modification of ICD-10) FY2027 codes
(effective 1 October 2026), form a holdout that no pretraining corpus can contain.

## Methodology
- **Placement items:** a new record with its aliases, definition and gold parents among the earlier release's nodes;
  scored by Hits@k and mean reciprocal rank.
- **Ranking items** (ICD code-to-title choice, MedConceptsQA) are scored by log-likelihood.
- A hash of the record id assigns 20% of records to dev, the rest to test, unless a benchmark has its own split (OET,
  TaxoExpan). We tune on dev and report test once. Primary and secondary sets are reported separately, never pooled.
- Builds are deterministic, and manifests pin every file's sha256. Licensed outputs stay under `~/data/vsa-llm/`
  (decision 58).
- On T7-trained models, the MeSH supplementary concept records (SCRs) that T7 trained with their gold frames are
  reported separately as "seen (sanity)".

## Experiment
CPU builds on 2026-10-08, with 23 tests on synthetic data.

## Results
| Set | Role | Items |
|---|---|---|
| MeSH 2025 → 2026 placement | primary | 179 (26 dev / 153 test): 120 descriptors, 59 SCRs |
| MeSH 2025 → 2026 new edges | primary | 584 |
| MeSH 2024 → 2025 placement | secondary | 212 (36 / 176) |
| ICD-10-CM FY2027 (H3) | primary | 238 placement (48 / 190); 476 choice (96 / 380) |
| MedConceptsQA ICD-10-CM | ranking reference | 264,256 test (local); 1,511-item frozen sample committed |
| OET (SNOMED CT time-split enrichment), 2014 → 2017; local, licensed | primary, concept-disjoint | `disease` 66 / 28; `cpp` 143 / 59 concepts |
| TaxoExpan SemEval noun / verb (local) | primary | 349 / 516; 51 / 84 |
| TMN protocol on WordNet, re-drawn (local) | secondary | 1,000 / 1,000 per part of speech |

- H3 has 190 new billable codes plus 48 new headers, 238 code strings in all; the FY2026 snapshot (98,505 nodes) is
  committed.
- The README states: "No model has been run on any of it."

## Limitations
- **Contamination:** most new MeSH names occur in 2025–26 PubMed text, and 93 new 2026 descriptors reuse 2025 SCR
  names. MeSH 2024–25 records may be in the hosts' pretraining. OET's contexts are 2017-era abstracts.
- **Leakage by design:** 79% (noun) and 69% (verb) of TaxoExpan test items have a training sibling. The re-drawn TMN
  split is not comparable to published TMN numbers.
- ICD frames must mostly come from the ICD hierarchy itself: of 41 gold-parent codes, only 8 reach a MeSH descriptor
  and 13 a SNOMED concept through UMLS.
- The primary test sets are small (153 MeSH and 190 ICD placements).

## Next steps
TK-L, the *learn* redesign that these sets evaluate, has 78 evaluation jobs queued since 2026-10-08 (≈ 1.85 GPU-h and
8.6 CPU-h; its T7 jobs wait for T7 training). ICD FY2027 and OET placement wait for the licensed clinical store. E13
uses the learn tool as stage 1 on T7-ROOD.

---

# 16.10 — The job queue and pre-registration practice

**Status:** done (in continuous use) · **When:** queue built 2026-09-30, extended 2026-10-04, 10-07, 10-08 and 10-09;
pre-registrations 2026-10-02 to 2026-10-17 · **Sources:** `src/vsa_embed/jobqueue.py` (docstring);
`resources/plan-improvement/execution.md` ("GPU order", "Rules", decisions 63–64); `resources/plan-improvement/queue.md`;
`resources/plan-improvement/tasks.md` ("Rules that keep the phases from invalidating each other");
`resources/plan-improvement/gates.md`; `manuscript/reproducibility.md`; `experiments/*/preregistration*.md`;
`manuscript/paper-readiness.md`; `manuscript/claims.md`; `manuscript/update-2026-10-12.md`

## Research gap
One RTX 3090 serves hundreds of jobs from several parallel agent sessions. Without order, jobs collide, crashed runs
are lost and no one knows which commit produced a number. And a programme that runs many analyses will find a "result"
somewhere unless it fixes its endpoints in advance.

## Intuition
Make every GPU job a file in a priority queue, run one at a time, and have every run record its provenance. Write
endpoints and readings, including how a failure would be read, before any run; date every later change.

## Methodology
**The job queue (B11).**
- Jobs are JSON files under `.jobs/` (git-ignored). The lowest priority number runs first, one GPU job at a time.
- A job left running by a dead runner is re-queued with resume arguments. A job starts only if enough disk is free.
- Added later: CPU and GPU lanes, so report jobs no longer hold the GPU (2026-10-04); one live job per lane, with
  blocked jobs retried first (2026-10-07); fractional priorities (2026-10-08), so blocks get non-overlapping bands
  (the toolkit blocks sit at 54.497–54.49984, with level steps of 0.00001); and `cancel`, which moves a job to
  `.jobs/cancelled/` with its reason (2026-10-09).

**Rules that bind every work package** (tasks.md, restated in execution.md):
- commit, then run; promotion-eligible runs refuse a dirty tree;
- one holdout per track, frozen and hashed before the first run that could see it;
- new names, not new semantics, so recorded runs replay bit for bit;
- the operator ablation in every report;
- MIMIC-derived data never committed;
- no GPU work beyond the committed ≈ 400 GPU-h without the author's approval at gate G6.

**Pre-registration.** A file is committed before any full run. It states the question, conditions, primary endpoints
with Holm correction, secondaries, the decision rule and pre-written refutation readings. A labelled smoke test may
follow ("SMOKE, not a result") and changes nothing. Later changes are dated amendments written before the affected
results are read; results produced earlier are reported under the design they ran with. Any development run that came
first is disclosed. Gate decisions apply the pre-registered rules and go to gates.md for the author's review.

## Experiment
This is a practice, not an experiment. Its footprint: 16 committed pre-registration files outside the clinical track,
from E5 (2026-10-02) to E14 (2026-10-17), plus 2 in the clinical track.

## Results
- **Failed predictions are reported.** Binding P1 predicted 0.55–0.65 for condition C5; all arms scored 0.48–0.50.
  T7's predicted gain (at least T4's) failed. E10.9a found no speed-up (H − P −0.003 [−0.019, +0.013]).
- **Holdouts stayed frozen.** When T1-open's slice and full builds produced different holdouts, we re-froze it from the
  full config before any training.
- **Amendments came before runs:** toolkit-bench §11 (three changes); E13 amendments 1–2; E10.L amendments 1–2.
  Decision 64 (2026-10-09) amended E13, E11, E12, E10.L and T1c-ROOD before any of them ran.
- **Withdrawals are recorded:** 12 T8 jobs (C5ut and C5tr, seeds 2–3) were cancelled with a reason.

## Limitations
- The G2 pre-registration of the E4 core (`experiments/e4-small-lm/preregistration.md`) does not exist, and queue.md
  still lists G2 as todo.
- queue.md's status table was last updated on 2026-10-04. reproducibility.md dates from 2026-10-02: it still lists the
  S0 pilot as running and E1 and E3 as queued, and its measured compute column is empty.
- Development runs preceded two pre-registrations (toolkit-bench and E10.9a); both are disclosed.
- gates.md cites commit `57e4e23` for E2 `runs/v2`; its manifest records `07a7315`. Two manifests record dirty trees
  (16.1).

## Next steps
- Write the G2 pre-registration before D4.1 starts.
- Refresh queue.md and reproducibility.md, and fill the compute ledger from the `.jobs/` logs.

---

# Sources

- Reports: `reports/R0-reproduction.md`, `reports/R1-sanity-pilot.md`.
- Benchmark and validation packages: `experiments/b6-host-memory/runs/3090-v1/{report.md,manifest.json}`;
  `experiments/b8-probe-validation/runs/v1/{report.md,NOTE.md,manifest.json}`, `runs/v1-ptq/{report.md,manifest.json}`;
  `experiments/b10-benchmarks/{generation.md,generation.json}`;
  `experiments/c3-general-corpus/runs/{v2,host-smollm2-v1,host-qwen2.5-v1}/report.md`, `runs/v2/summary.json`,
  `runs/v1/STALE.md`; `experiments/c5-judge-calibration/v{1,2}/{summary.json,results.json}`, `v2/manifest.json`;
  `experiments/c6-devtools-benchmark/v1/{summary.json,library.json,probes.jsonl}`;
  `experiments/wp-probe-validation/runs/v1/{report.md,manifest.json}`.
- Toolkit: `manuscript/toolkit-methodology-2026-10.md`; `experiments/toolkit-bench/preregistration.md`,
  `experiments/toolkit-bench/items/*/manifest.json`, `experiments/toolkit-bench/smoke/`;
  `experiments/toolkit-learn/README.md`.
- Related experiment files: `experiments/t2-developer-tools/README.md`; `experiments/e5-explainability/preregistration.md`;
  `experiments/e7-self-authoring/runs/devtools-v1/report.md`; `experiments/e13-learning-cycle/preregistration.md` (§9, §12).
- Plans and methodology: `resources/plan-improvement/{execution.md,queue.md,gates.md,tasks.md,runbook.md,experiments.md}`;
  `manuscript/reproducibility.md`.
- Manuscript status: `manuscript/update-2026-10-12.md`, `manuscript/update-2026-10-15.md`, `manuscript/claims.md`,
  `manuscript/paper-readiness.md`.
- Code docstrings: `src/vsa_embed/jobqueue.py`, `src/vsa_embed/judge_protocol.py`, `src/vsa_embed/benchmarks/devtools.py`.
- Dates: `git log --date=short` on each path above.
