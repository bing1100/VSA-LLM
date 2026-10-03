# R1 — S0 sanity pilot: is there any signal?

**Date:** 2026-10-02. **Status:** exploratory (author request of 2026-10-02: spend a few GPU-hours on an early sanity check before the committed ≈ 400 GPU-hours). Not a gate; never pooled with gate runs. **Machine:** `bhux-tiny`, RTX 3090. **Analysis:** `experiments/e4-small-lm/analysis/pilot-v2/` (`vsa_embed.experiments.e4_report`), probes per run in `runs/pilot/*/probes.json` (`vsa_embed.evaluation.channel_probes`).

## Setup

| Block | Model | Data | Conditions | Seeds | GPU |
|---|---|---|---|---|---|
| From scratch | GPT-2-style 50M, GPT-2 BPE | C3 FineWeb-Edu × WordNet (`runs/v2`), 100M tokens, ℓ_min = 2, frozen holdout `7f2462ed…` (5,900 entries never linked in training) | C0 none · C1 random fixed span vectors · C2 free per-concept table (capacity-matched to C3) · C3 static VSA bundle (HRR) · C5 attentive VSA with local-context query (HRR, induced mapping) | 1, 2 | ≈ 4.2 h |
| Continued pretraining | SmolLM2-135M, LoRA r = 16, host lr 2e-4, channel lr 1e-3 | SmolLM2 host corpus (same linker and holdout), 25M tokens | C0′ (LoRA only) · C2 · C5 | 1 | ≈ 1 h |

Evaluation: 512 (from scratch) / 1,024 (CPT) fixed 1,024-token windows of the C3 evaluation documents, stratified per experiments.md §0.3; paired differences use a cluster bootstrap over windows (10,000 resamples, seeds pooled per window), Holm-corrected within each stratum. Probes: WiC, WSD (Raganato ALL), CARD-660, Rare Words, BLESS, HyperLex, LAMBADA, each also split by whether the probed word is a held-out concept, a seen concept, or unlinked.

## Results

### From scratch: language-model loss

Relative difference to C0, (condition − C0)/C0, negative = better; `*` = significant after Holm.

| Stratum | targets | C1 random vectors | C2 free table | C3 static VSA | C5 attentive VSA |
|---|---:|---|---|---|---|
| tokens after a linked word (`after`) | 175,271 | +0.26% [+0.23, +0.28]* | −0.35% [−0.39, −0.32]* | −0.39% [−0.43, −0.35]* | **−0.56% [−0.60, −0.51]*** |
| inside multi-token words (`inside`) | 32,729 | +0.08%* | −0.87%* | −1.00%* | **−1.32% [−1.45, −1.19]*** |
| after 3+-subtoken words | 44,782 | +0.22%* | −0.49%* | −0.74%* | **−0.90% [−0.99, −0.80]*** |
| after held-out concepts (zero update) | 26,855 | +0.28%* | −0.02% [−0.08, +0.04] | −0.04% [−0.14, +0.06] | **−0.15% [−0.26, −0.04]*** |
| after rare concepts (train freq 1–9) | 335 | +0.40% | −0.38% | −0.37% | −0.53% [−1.45, +0.40] |
| unlinked text (locality) | 327,905 | +0.30%* | −0.01% | +0.28%* | +0.12% [+0.10, +0.14]* |

C5 vs the free table C2: `after` −0.20%*, `inside` −0.45%*, 3+-subtoken −0.41%*, held-out −0.13% [−0.24, −0.02] (not significant after Holm). C5 vs C3: better on every linked stratum (context query helps).

### From scratch: probes (mean of 2 seeds)

| Probe | C0 | C1 | C2 | C3 | C5 |
|---|---:|---:|---:|---:|---:|
| CARD-660 Spearman, all | 0.106 | 0.106 | 0.104 | **0.163** | **0.163** |
| — held-out concepts (n = 36) | 0.102 | 0.116 | 0.092 | **0.311** | **0.326** |
| — seen concepts (n = 321) | 0.138 | 0.142 | 0.146 | **0.255** | **0.264** |
| — unlinked words (n = 303) | 0.074 | 0.067 | 0.066 | 0.071 | 0.057 |
| BLESS pair probe accuracy, all | 0.717 | 0.717 | 0.713 | **0.759** | **0.759** |
| — held-out concepts (n = 128) | 0.621 | 0.629 | 0.594 | **0.680** | **0.684** |
| — seen concepts (n = 1,790) | 0.684 | 0.683 | 0.680 | **0.759** | **0.764** |
| — unlinked (n = 2,392) | 0.746 | 0.747 | 0.744 | 0.763 | 0.759 |
| Rare Words Spearman | 0.206 | 0.186 | 0.186 | 0.211 | 0.214 |
| WiC probe accuracy | 0.558 | 0.564 | 0.573 | 0.559 | 0.550 |
| WSD probe F1 | 0.650 | 0.655 | 0.655 | 0.655 | 0.655 |
| HyperLex probe Spearman | 0.326 | 0.328 | 0.332 | 0.327 | 0.332 |
| LAMBADA accuracy | ≈ 0 (models too early in training to score) | | | | |

Seed spread on the improved probes is small (CARD-660 ±0.004–0.02, BLESS ±0.003–0.01).

### Continued pretraining (SmolLM2-135M, LoRA, 25M tokens)

All differences between C0′, C2 and C5 are below 0.06% in every stratum. But training itself barely moved the model (all-token loss 2.798 → 2.791 for every condition): with a near-converged host, a closed initial gate (bias −2) and 190 optimizer steps, this setup cannot show an effect either way. **Uninformative, not negative.**

## Findings

1. **Ontology-structured composition gives a consistent, small advantage on the text it touches.** C5 is best on every linked stratum; it beats the capacity-matched free table on linked text by 0.2–0.45% and the random-vector control everywhere; C3 (static bundle) is second.
2. **The clearest signal is on concepts never seen in training.** On CARD-660 (independent human similarity judgements) the VSA channels triple the held-out correlation (0.10 → 0.31–0.33), while the free table, which has no row for unseen concepts, gains nothing. BLESS shows the same pattern (+6 points held-out). Unlinked words do not change. This is the zero-update composition signature the program is about (H-E).
3. **The loss-level held-out effect is small** (−0.15% for C5; not significant vs C2 after Holm). The rare stratum is too thin to measure (335 targets).
4. **Random span vectors hurt** (+0.26–0.30% everywhere), so span-boundary information alone does not explain the gains — but C1 here has only ≈ 1k trainable parameters (not parameter-matched), so this control must be fixed before the screen (G2).
5. **Locality cost:** the static bundle C3 raises unlinked-text loss by 0.28%, C5 by 0.12% (inside the 0.5% margin); the free table does not. Worth tracking.
6. **What the pilot cannot say:** whether *binding* matters (all channel runs used HRR; E2 found operators indistinguishable from `random_fixed` on frozen anchors), whether effects survive longer training, and anything about pretrained hosts.

## Implications and decisions

| Implication | Action |
|---|---|
| The signal is worth the committed budget, and it is strongest on zero-update concepts | proceed with the committed plan |
| At 262k tokens/step a 100M-token run is 381 optimizer steps and stays at loss ≈ 5.6 (very early training); the committed 300M/500M-token runs would also stay early | **D4.0 recipe sweep** queued (C0 at 50M × 100M: tokens/step {65k, 131k, 262k} × lr {1e-3, 2e-3, 4e-3}); the best setting is frozen for D4.8/D4.9/D4.1 |
| C1 is not parameter-matched | G2: give C1 a trainable projection of fixed random vectors matched to C3's parameters |
| The CPT setup does not move a pretrained host | **E9 retrofit × quantization** (author request): engagement check first (full fine-tuning vs LoRA-64; gate bias −2 vs 0; 10M tokens on SmolLM2-360M), then the main runs |
| "VSA" vs "compositional parameter sharing" is open | decided by the C8 operator ablation (now also adding a translation `x + r` operator, per the literature re-check) |
| BLESS overlaps WordNet hypernymy, which the channel injects | report CARD-660 (independent) as the primary similarity evidence; BLESS as a knowledge-injection probe |

## Reproduce

```bash
PYTHONPATH=src python -m vsa_embed.experiments.e4_plan --stage pilot --queue
PYTHONPATH=src python -m vsa_embed.experiments.cpt_plan --stage cpt-pilot --queue
PYTHONPATH=src python -m vsa_embed.jobqueue run
PYTHONPATH=src python -m vsa_embed.evaluation.channel_probes --run <run> --output <run>/probes.json
PYTHONPATH=src python -m vsa_embed.experiments.e4_report --runs experiments/e4-small-lm/runs/pilot experiments/e4-small-lm/runs/cpt-pilot --output experiments/e4-small-lm/analysis/pilot-v2
```

Runs were made at commits recorded in each `manifest.json` (from `9606bc8`, which contains the 1,600× batch-sampling speed-up; one C0 run started on the slow code was discarded and re-run).
