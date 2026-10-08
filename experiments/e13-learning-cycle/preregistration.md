# E13 — the learning cycle: pre-registration

**Date:** 8 October 2026. **Decision:** 63 (author, 2026-10-08; WP TK-E13). **Design source:**
[`manuscript/toolkit-methodology-2026-10.md`](../../manuscript/toolkit-methodology-2026-10.md) §4 and §2 (M2, M3).
**Code:** `vsa_embed.experiments.e13_cycle`, `vsa_embed.concept_store`, the opt-in trainer keys of
`vsa_embed.training.lm` (`train.init_mode: continue`, `model.host_quantization`, `channel.entry_rows`, `eval.points`).
**Status:** written and committed before any E13 run (the CPU smoke in `smoke/` is labelled SMOKE and is not a result).
Nothing is queued by this work package; `queue-commands.sh` lists the commands.

## 1. Question

Can a model trained passively with an ontology (1) learn new structure, (2) read in new words, (3) use them for
reasoning, and (4) then learn the following text faster? (5) Can it do so when its weights are frozen at 4 bits?
E13 is the one experiment that tests the read–learn–write toolkit as a whole, and the only one that can support a
training-efficiency claim.

## 2. Tracks and the round split

| track | role | round 1 (seen) | round 2 (new) | round-2 text |
|---|---|---|---|---|
| **T5** (enterprise glossary) | controlled upper bound; developed and run first | the 3,640 training terms | the **360 real held-out terms** (absent from every training text by construction; leakage audit 0) | new T5 documents about the round-2 terms, generated with the T5 generator under a fresh seed (`round2_documents`; ≈ 5M SmolLM2 tokens) + the same number of tokens of FineWeb-Edu documents no T5 build read (shard 000 from document 100,000) |
| **T7-ROOD** (H1) | main real-text track | MeSH 2026 records entered before the split date | records entered after it; every training document mentioning one is dropped | their PubMed 2025–26 abstracts (TK-H1) |

T7-ROOD's round split is built by TK-H1 (`experiments/t7-new-vocabulary/ROOD.md`). It did not exist when this was written;
`t7-rood.yaml` is written against the expected interface and marked TODO. Everything below applies to both tracks; the
T7-ROOD block is queued only after its files exist and its configuration is completed (no other change).

**Round-2 evaluation text** is the E9 evaluation corpus of the track (T5: the fixed 1,024 windows of 1,024 tokens
of `v1/eval`), which contains both round-1 and round-2 terms and is never trained on. Round-2 training text is generated
separately, so no evaluation window is trained on.

## 3. Hosts and seeds

- **Primary:** SmolLM2-360M, full fine-tuning (the E9 recipe), seeds 1, 2, 3.
- **Secondary:** Qwen3-1.7B-Base with LoRA r 64 (the E9 Qwen3 recipe), seed 1.
- Every arm of a host and seed starts from the same stage-0 run and sees the same round-2 batches (`data.seed` 4321);
  arms differ only in what they read (paired by window and batch).

## 4. Stages

| stage | what happens | settings |
|---|---|---|
| **0 passive round 1** | the E9 C5 run (attentive HRR channel, d 256, key 8, context 8; host lr 3e-5, channel lr 1e-3, gate bias 0; 50M tokens, 65,536 tokens per step) on the round-1 corpus with the **seed ontology** | seed ontology: round-2 entries have no frame; round-1 frames drop their edges to round-2 terms (716 edges on T5); two inverse relations are materialized (`owns` = owned_by⁻¹, 3,240 edges; `has_part` = part_of⁻¹, 690; E10.9b's design); then a seeded **20%** of the round-1 edges are erased (6,024 of 30,121; every frame keeps ≥ 2 edges): the **erased gold** of stage 1. C0′ and C2 read no frame: their E9 T5 runs (same recipe and corpus) are the stage-0 references |
| **1 learn** | `ConceptStore.propose` → `accept` | proposer: `rule_closure` (E10's rule closure: inverse / symmetric / transitive / composition rules mined in the store's own pairs, kept at support ≥ 10 and PCA confidence ≥ 0.6; their predicted new edges of round-1 entries, type-checked, no second filler of a functional relation). TK-L's decompose-then-verify proposer replaces it through `learn.proposer` *if it lands before stage 1 runs* (recorded as a deviation). Acceptance: the **LM held-out utility test** — per proposal, the loss over the 8 tokens after the entry's span minus the loss with the edge added, on ≤ 8 windows of 128 tokens of **fresh round-1 text** (`learn-validation`, never trained on), one-sided t-test, **Holm over every proposal of the call**, α 0.05; **null world** (M3): one filler-swapped proposal per real proposal (same relation, same filler type, frequency-weighted, never a true edge), tested in the same Holm family. Accepted edges are written into the store for every later stage |
| **2 write** | round-2 terms are read from their definitions; rows inserted; no gradient | definitions: E11's `prose` style (written from the gold frame in held-out wordings: no item or training template); reader: E11's pre-registered primary **`linker`** reader on the learned store; secondary readers `linker-joint`, `typeprior` (frame metrics only). The arms' frames (read, gold, random equal-degree) and the `fvt` rows are fixed here |
| **3 reason** | relation, reverse and two-hop questions about round-2 terms | items: the E9 understanding items of the round-2 anchors (`negation/affirm` = relation, `reverse`, `two_hop`); conditions `none` (no tool; the read rows in the channel), `recall:own` (the recall tool's text of the read store in context), `symbolic` (the gold frame as text), `definition` (the definition as text). E12's scoring (PMI argmax per template) |
| **4 passive round 2** | continue each stage-0 run on round-2 text | `train.init_mode: continue`, fresh AdamW with the stage-0 recipe (lr, host lr, decay, cosine to 0.1), warm-up 1M tokens, **10M tokens** (153 steps), evaluations at 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8 and 10M tokens on the fixed windows with the reference strata `ref_round2` (the 8 targets after a round-2 term) and `ref_round1` (after a round-1 term); general-text loss on 256 `eval-general` windows at the start and the end |
| **5 frozen 4-bit host** | stages 2–4 with the stage-0 host quantized and frozen | `model.host_quantization`: **RTN int4** (group 128, the AutoGPTQ grid; primary) and **NF4** (64-blocks, double-quantized absmax; QLoRA's format; secondary); simulated weight-only quantization (the dequantized values a 4-bit kernel computes with; output head and input embedding kept 16-bit, as E9's claim-B schemes). Same round-2 text, tokens and evaluations as stage 4 |

### Arms

| arm | stage | starts from | round-2 rows | trained | text |
|---|---|---|---|---|---|
| `read` | 4 | stage-0 C5 + learned edges | frames of the `linker` reader | host + channel | round 2 |
| `noread` | 4 | same | none (terms linked, no row) | host + channel | round 2 |
| `fvt` | 4 | same | subtoken mean of the host's input embeddings over the term's aliases, scaled to the mean channel row norm; **trainable** (standard vocabulary expansion) | host + channel + those rows | round 2 |
| `gold` | 4 | same | gold frames (oracle reader) | host + channel | round 2 |
| `defs` | 4 | same | none | host + channel | round 2 with each document preceded by the prose definitions of the round-2 terms it names; same training tokens (compute-matched); its general-text share kept at 50% |
| `random` | 4 | same | random frames of equal degree (gold relations, fillers frequency-weighted; content control) | host + channel | round 2 |
| `C0p`, `C2` | 4 (reference) | E9 C0′ / C2 run | none / C2's fallback row (trainable) | host (+ table) | round 2 |
| `q4-read` | 5 | stage-0 host, quantized, frozen | `read` frames | channel only | round 2 |
| `q4-noread` | 5 | same | none | channel only | round 2 |
| `qlora` | 5 | same + LoRA r 64 (host lr 2e-4), the in-repo `LoRALinear` | none | LoRA + channel | round 2 |
| `qlora-read` | 5 (RTN only) | same | `read` frames | LoRA + channel | round 2 |

Operator controls (untyped, translation) are not run (methodology §4: only if stage 3 is run on them). Qwen3-1.7B runs
`read`, `noread`, `gold` and the RTN `q4-read`, `q4-noread`, `qlora`.

## 5. Pre-registered endpoints and kill criteria (primary host SmolLM2-360M, track T5; T7-ROOD in its own report)

| | endpoint (methodology §4) | statistic | met iff | if not met |
|---|---|---|---|---|
| **L1 (write)** | read − no-read loss after round-2 terms at step 0 < 0, CI excluding 0 | `ref_round2` loss at token 0 of the stage-4 runs (the stage-0 model with the rows written, no gradient): per window the mean-loss difference `read − noread`; windows × seeds table, two-way (pigeonhole) cluster bootstrap, 2,000 resamples | Holm p < 0.05 and the upper CI < 0 | no zero-shot write claim on this track |
| **L2 (efficiency)** | tokens to criterion read / no-read < 1, CI excluding 1 | criterion = the `noread` arm's final `ref_round2` loss (10M tokens), per seed; tokens to criterion by linear interpolation between evaluation points (∞ if never reached); ratio = Σ_seeds TTC(read) / Σ_seeds TTC(noread); **read's tokens include its reading cost** (definition forward tokens ÷ 3, training-token equivalents); bootstrap: windows and seeds resampled together, curves recomputed per resample; p from the bootstrap distribution of log ratio | Holm p < 0.05 and the upper CI < 1 | **no training-efficiency claim** |
| **L3 (4-bit)** | INT4 host + read − INT4 host without reading < 0; non-inferior to QLoRA at matched tokens | RTN: `ref_round2` loss at 10M tokens, `q4-read − q4-noread`, as L1's statistic; non-inferiority: `q4-read − qlora` upper CI < 1% of `qlora`'s loss | Holm p < 0.05 and upper CI < 0 (L3); non-inferiority reported with it | if QLoRA wins clearly (`q4-read − qlora` lower CI > the margin): the claim is only "zero-gradient" |
| **L4 (learn)** | precision of accepted edges ≥ 0.8 with a null-world false rate ≤ 5% | pooled over seeds: precision = accepted ∩ erased gold / accepted; null false-acceptance = accepted null / null proposals (Wilson CIs reported); test for Holm: exact one-sided binomial p of precision > 0.8 | precision ≥ 0.80 and null rate ≤ 0.05 (point estimates) and Holm p < 0.05 | stage 1 is dropped from the cycle and reported as negative (stage 4's comparisons are unaffected: every arm carries the same accepted edges) |
| **L5 (reason)** | recall-tool accuracy on new terms − no tool > 0 | per round-2 anchor the mean over its relation / reverse / two-hop items of accuracy(`recall:own`) − accuracy(`none`); anchors × seeds, two-way cluster bootstrap | Holm p < 0.05 and the lower CI > 0 | the recall tool is not shown to help on new words |

**Multiplicity:** Holm across the five primary p-values (L1, L2, L3, L4, L5) at α = 0.05. Everything else is secondary.

## 6. Secondary analyses (reported, no claim without a primary)

- `gold`, `fvt`, `defs`, `random` vs `noread`: step-0 and final `ref_round2` differences and tokens to criterion;
  `read` vs `gold` (the reader's gap to the oracle), `read` vs `fvt` / `defs` (reading vs the standard alternatives).
- AULC (mean `ref_round2` loss over the token axis) of every arm vs `noread`.
- Forgetting: change of the `ref_round1` loss from token 0 to 10M per arm; locality: change of the general-text loss.
- Stage 1: proposal recall, accepted recall, utilities of accepted edges, rules mined. Stage 2: frame precision /
  recall / F1 of every reader against the gold frames, reading cost. Stage 3: `symbolic` and `definition` (the frame
  as text) against `recall:own`, decode accuracy of the recalled lines, per family.
- NF4 (L3's statistics), `qlora-read` vs `qlora`, bytes (nominal host bytes from the scheme record; trainable
  parameters) of every frozen arm.
- Qwen3-1.7B-Base seed 1: every endpoint's estimate with its window-bootstrap CI (one seed; descriptive).
- C0p / C2 references: round-2 learning curves of a model without a composing channel.

## 7. Expectations (from the evidence, methodology §4)

- E9 held-out terms composed from gold frames lower the loss after the term by 9–13% on T5: L1 large on T5, smaller
  with the linker reader (E11 frame F1 on T5 new words: linker 0.55, oracle 1.00).
- E10.9a predicts that stage 1 adds little speed but writes correct edges for unobserved concepts; on the T5 seed
  ontology rule closure proposes 1,302 edges, all of them erased gold (CPU check without a model), so L4 depends on
  the acceptance test's power and the null rate.
- On T4 the zero-shot gain was small; T7-ROOD's L1 / L2 depend on T7 showing a gain.

## 8. Analysis details

- All losses are token-weighted over the stratum's targets; a window enters a paired comparison only if both runs have
  ≥ 1 target in it (the same windows in every arm: the strata masks are fixed by `reference-1024x1024.npz`).
- Bootstrap: 2,000 resamples, percentile intervals, two-sided p from the bootstrap distribution
  (`statistics.two_way_cluster_bootstrap`; L2: `e13_cycle.efficiency`).
- A run that fails is re-run once from its checkpoint (`--resume`); a run that cannot finish is reported as missing,
  and an endpoint needing it is reported as not evaluable (never imputed).

## 9. Deviations policy

Any change after this commit — to the arms, endpoints, statistics, data, recipe or the proposer — is recorded as a dated
amendment at the end of this file *before* the affected results are read, with its reason. Changes forced by a crash or
a bug fix that does not change the design are recorded the same way. Results produced before an amendment are reported
under the design they ran with. Exploratory analyses are labelled as such.

## 10. Cost and queue

GPU-hours from the measured per-step costs of the E9 T5 runs (median `train_tokens_per_s`: SmolLM2-360M 12,767; Qwen3-1.7B
4,205) plus evaluation passes (24 s per 1,024 windows for 360M, scaled by parameters): **T5 ≈ 35.2 GPU-h** (70 jobs:
stage 0 with learn / write 8.5, stage 4 11.6, stage 5 10.6, stage-3 evaluations 4.3, data preparation 0.3; 360M ≈ 20.5,
Qwen3 ≈ 14.5). Priorities:
T5 at 54.4985 (stage 0 with data preparation, learn and write) / 54.4986 (stage 4) / 54.4987 (stage 5) / 54.4988
(stage 3 evaluations) / 54.4989 (report, CPU lane); T7-ROOD at 54.4996–54.4999 likewise. See `queue-commands.sh`.

## 11. Design decisions open for the author

1. The derived inverse relations in the seed ontology (they give the default rule-closure proposer something to close;
   without them it proposes nothing on T5's relations).
2. 20% erasure and 10M round-2 tokens (≈ 1 epoch of the round-2 stream; the E9 held-out gain suggests round-2 terms
   are learned within a few million tokens).
3. Reusing the E9 C0′ / C2 runs as stage-0 references (they read no frame, so the seed ontology does not change them).
4. Whether TK-L's proposer, if ready, replaces `rule_closure` as the stage-1 primary (then L4 measures it; the rule
   closure result stays as a baseline).
