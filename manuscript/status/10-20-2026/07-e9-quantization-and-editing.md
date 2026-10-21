# 07 — E9: 4-bit quantization and editing knowledge through the ontology

> **Update, 20 October (evening).** B-B1, the size-matched side-module control, was approved on 20 October. 22 jobs
> are queued after the toolkit band and should land about 4–6 November; see the digest, §5.

E9 asked two further questions about the method once it is trained into a pretrained host. First (claim B): does its advantage survive, or even grow, when the host is compressed to 4 bits? Second (claim C): can we change what the model knows by editing the ontology, with no weight update? For quantization the answer is a measurement, not a headline. The advantage roughly survives 4-bit post-training quantization, but how much survives depends on the track and on the quantizer. On the synthetic glossary it is largely kept, and fully under GPTQ. On chemistry it is partly lost under GPTQ and RTN and kept under AWQ. On 2026 MeSH names it grows under RTN and AWQ. The controls place each effect either in the host weights or in the method's own channel, never in "structure". A bug in our AWQ code was found, fixed and re-run on 12 October. For editing, writing an ontology entry for a new invented word improves simple property questions a little (0.189 → 0.259 at 360M; chance 0.20), and editing one relation shifts predictions on seen words. Neither works on real text. The standard weight-editing baselines (ROME, MEMIT, AlphaEdit, in-context editing) are queued but have not run. Quantization was taken out of the abstract on 12 October; claim C remains a proof of concept.

**Codes used in this file** (as in `manuscript/update-2026-10-12.md`, "How to read the numbers"):

| Code | Meaning |
|---|---|
| host | the pretrained language model: SmolLM2-135M or -360M (full fine-tuning), Qwen3-0.6B or -1.7B (LoRA r 64) |
| entry, frame | an ontology record; its frame is the set of its relation–filler edges |
| **C5** | the method: a vector composed from the entry's frame over a shared, jointly trained dictionary, added at the last subtoken of each linked multi-token term. Its binding is **HRR** (holographic reduced representation, circular convolution). The module that adds the vector is the "channel" |
| **P0** | the original host, not trained further |
| **C0′** | the host fine-tuned on the same text and tokens without the method; the fair baseline |
| **C2** | a free lookup table, one learned vector per entry, no ontology, matched to C5's parameters |
| held-out terms | kept out of the method's training; at test time their vector is composed from the frame alone. On T5 they never occur in the training text; on T4 and T7 they occur there, unlinked |
| unseen / rare / long terms | in the ontology but absent from the training text / seen 1–9 times / three or more subtokens |
| T1–T8 | tracks. **T1-open**: MeSH in PubMed; **T1c**: clinical notes (MIMIC-III) with SNOMED CT; **T4**: chemistry (ChEBI, PubMed); **T5**: synthetic company glossary; **T7**: PubMed 2025–26 with 2026 MeSH names; **T7-ROOD**: T7 with every training document naming a held-out entry removed; **T8**: Wikidata entities in web text; **WordNet**: general web text. (T2, T3 and T6 have no E9 runs; see 08.4) |
| INT8 / INT4, bf16 | 8- / 4-bit weight-only post-training quantization of the host's linear layers; bf16 = the model as trained |
| RTN, HQQ, NF4, GPTQ, AWQ | 4-bit quantizers: round-to-nearest (torchao, the simplest), half-quadratic, 4-bit NormalFloat, GPTQ (second-order error correction), AWQ (activation-aware scaling) |
| variant A / B | channel kept in FP16 / channel quantized too |
| channel off | the channel switched off at both precisions: what the C5-trained host weights do alone |
| gain, DiD, retained | gain = C5 − C0′ in nats per token (negative = C5 better); **DiD** ("difference in differences") = gain at 4-bit − gain at bf16. **Positive DiD = the advantage shrinks; negative = it grows.** retained = gain at 4-bit / gain at bf16 |

**Numbers.** Loss numbers are in nats per token unless marked %. Values pool 3 training seeds, with 95% cluster-bootstrap intervals over evaluation windows (loss) or items (editing). `*` = significant after Holm correction. "Seed 1" marks a single run. "Preliminary" marks numbers from generated reports that no committed summary has checked yet.

---

# 07.1 — 4-bit quantization (claim B)

**Status:** done on T5, T4 and T7 at 360M (3 seeds), T5 at 135M, Qwen3 on T5 (2 seeds); controls done; side-module control deferred · **When:** 2026-10-04 (T5, 3 seeds), 2026-10-06 (Qwen3), 2026-10-07 (T5 controls), 2026-10-11 (T4, T7 and T7 controls), 2026-10-12 (AWQ fix and re-run) · **Sources:** `reports/R9-retrofit-quantization.md` (dimension 2, "T5 controls", "T4 at 3 seeds", "T7", "Qwen3 on T5"); `manuscript/update-2026-10-12.md` (E9-9); `manuscript/claims.md` (E9.B1); `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §1; `experiments/e9-retrofit/report/{t5,t4,t7}-awqfix/`; `experiments/e9-retrofit/quant/{t5,t4,t7}/report.md`

## Research gap
The author's hypothesis (2026-10-02): the method's advantage grows after weight quantization. Published work predicts that 4-bit quantization erases small fine-tuning deltas, so knowledge in *any* separate module survives. We had to test whether the method does more than that.

## Intuition
The composed vector lives outside the quantized layers. If 4-bit noise damages the host's own fine-tuning, the channel should matter more, provided the quantized host can still read it.

## Methodology
- **Quantization:** torchao weight-only INT8 and INT4; output head and input embedding kept at 16 bit. INT4 group 64 on SmolLM2, 128 on Qwen3.
- **Controls:** variant B, the five quantizers, channel off, and the input embedding quantized too.
- **Metric:** DiD and retained per stratum; window bootstraps; Holm within stratum.
- **Pre-registration:**
  - Addendum 1, §1 (T7): (a) GPTQ removes most of RTN's growth; (b) with the channel off, the DiD keeps its sign.
  - Claims ledger (E9.B1): growth that vanishes under GPTQ or AWQ is an RTN artefact; growth the channel-off host also shows means "a separate module survives quantization", not structure.

## Experiment
SmolLM2-360M (seeds 1–3) on T5, T4 and T7; SmolLM2-135M on T5 (seeds 1–3), T4 (seed 1) and T7 (seeds 1–3, preliminary); Qwen3 on T5 (seeds 1–2). All 44 AWQ rescorings were redone after the fix.

## Results
**INT8 is free** (≤ 1.4% loss on T5; T7 gain retained 0.94–1.04×). **INT4** hits fine-tuned T5 hosts much harder than the original host (all tokens: C0′ +38% at 360M and +63% at 135M; P0 +5.6% and +7.7%). On T4 and T7 the trained hosts lose about what P0 loses.

DiD, C5 vs C0′, 360M (positive = the advantage shrinks):

| Track | Stratum | RTN | HQQ | NF4 | GPTQ | AWQ (after fix) | RTN, channel off |
|---|---|---|---|---|---|---|---|
| T5 | after held-out | +0.049* (retained 0.58×) | +0.055* | +0.033* | **−0.001** (fully kept) | +0.017* | −0.001 |
| T5 | all tokens | −0.005* | −0.003* | −0.002* | +0.002* | −0.006* | −0.012* |
| T4 | after held-out | +0.0020 (0.65×) | +0.0009 | +0.0059* | **+0.0040*** | −0.0010 (kept) | +0.0036 [+0.0021, +0.0051] |
| T4 | after rare | +0.0009 | +0.0002 | +0.0033* | +0.0036* | −0.0039* (grows) | — |
| T7 | after rare | **−0.0077*** (grows 1.65×) | +0.0006 | −0.0001 | −0.0022 | −0.0039* | −0.0003 |
| T7 | after long | −0.0080* | −0.0006 | −0.0034* | −0.0018* | −0.0043* | +0.0027 |
| T7 | after held-out (a cost) | −0.0062* (cost shrinks to 0.41×) | −0.0026* | −0.0039* | −0.0026* | −0.0073* | −0.0000 |

- **T5:** the held-out advantage is largely kept (0.58× at 360M; 1.14× at 135M, DiD −0.012 [−0.019, −0.005]*), and fully under GPTQ. With the channel off the held-out DiD vanishes, so the RTN loss is the quantized host reading the channel less well. The all-token robustness (1.26–1.32×) survives with the channel off (−0.012*): it sits in the C5-trained host weights.
- **T4:** INT4 shrinks the gain on every term stratum (0.65–0.94×), significantly only inside (+0.0033*) and after long terms (+0.0016*). GPTQ loses about 70% of the held-out advantage, NF4 all of it, AWQ none. The loss sits in the host weights (channel's own DiD −0.0016 [−0.0038, +0.0006]).
- **T7:** under RTN the gain after any linked term grows from −0.0125 to −0.0203 nats (1.63×), while C2's gain shrinks (0.83×). The growth is an RTN effect carried by the channel: GPTQ keeps the advantage without enlarging it, AWQ keeps about half the growth, HQQ none. **Prediction (a) held; (b) failed** (channel off: −0.0003, +0.0027).
- **T7 at 135M (3 seeds, preliminary):** under RTN the gain after long terms grows (1.34×, DiD −0.0093*). After held-out terms a non-significant cost (+0.0023) becomes an advantage (−0.0168; DiD −0.0192*). On all tokens the advantage turns into a small cost (DiD +0.0025*).
- **Qwen3 on T5 (2 seeds):** held-out retained 0.91× (0.6B) and 1.07× (1.7B, n.s.); all tokens 0.25× against 2.88×. No consistent direction.
- **The AWQ bug (fixed 2026-10-12, commit b6c0f0b).** Our code grouped layers that share an input by memory address. PyTorch reuses freed memory, so layers reading *different* inputs sometimes shared scales. T7's C2 lost +0.26 to +0.36 nats on every token; after the fix +0.094, like C0′ (+0.095). Two readings changed: AWQ now keeps T4's advantage and enlarges T7's. T5's all-token −0.047 was an artefact (now −0.006*).
- **The hypothesis is not supported as a general claim.** R9's wording: "survives INT4; the direction depends on the track (fully kept under GPTQ only on T5)". Quantization left the abstract on 2026-10-12.

## Limitations
- **The bit-matched unstructured side module (B-B1) has not run** (deferred by the author on 2026-10-11, still on 2026-10-15). Without it we cannot separate "a separate module survives" from anything specific to the method.
- Weight-only post-training quantization only. All controls are on SmolLM2-360M; Qwen3 is LoRA, 2 seeds, and mixed. Real-text effects are thousandths of a nat.
- Earlier documents gave the INT4 group size as 128 (it is 64 on SmolLM2; corrected 11 Oct). AWQ values before 12 Oct 05:00 are invalid.
- **Sources disagree; we follow the newest.** The claim text in `claims.md` (E9.B1) still says the advantage "does not grow on the target terms". It predates T7; the row's refutation column and R9 record the T7 growth.
- T1c quantization jobs finished, but no committed summary exists; we do not report them (licensed-data rule, 08.2).

## Next steps
- **Queued:** T7-ROOD (360M at 54.4972; 135M at 54.499937), T8 (54.4992), and Qwen3-1.7B on T4 (54.4962) and T7 (54.49752).
- **Deferred (author):** B-B1.
- **Proposed, not queued:** revert diagnostics (B-B4) and quantization-aware training or QLoRA (B-B5), P1 controls in `claims.md`. Write T7 135M into R9 once committed.

---

# 07.2 — Editing knowledge by editing the ontology (claim C)

**Status:** partly done (proof of concept on T5; null on T4; not frame-specific on T7); editing baselines queued · **When:** 2026-10-04 (T5, 3 seeds), 2026-10-06 (Qwen3, 2 seeds), 2026-10-05 (T4 seed 1), 2026-10-11 (T4 and T7, 3 seeds), 2026-10-18/19 (understanding items, preliminary) · **Sources:** `reports/R9-retrofit-quantization.md` (dimension 3, "T4 at 3 seeds", "T7", "Qwen3 on T5"); `manuscript/update-2026-10-12.md` (E9-10); `manuscript/claims.md` (E9.C1, E9.C2); `manuscript/outline.md` (E9 framing rules); `experiments/e9-retrofit/report/{t5,t4,t7}-understanding/report.md` (uncommitted)

## Research gap
After training, can we teach the model a new word, or change a fact, by writing to the ontology instead of updating weights? The general framing ("editing a symbolic store instead of weights") is already published (Facts as Experts, KBLaM, LMLM, REMEDI). Only this mechanism is new.

## Intuition
The composed vector is a function of the frame. If the trained host reads that vector, a new or edited frame should change its predictions at once, at zero gradient steps and zero prompt tokens.

## Methodology
- **New words:** 300 invented words per track, with frames built from existing dictionary entries and relations, added after training. Tests: five-way property selection (chance 0.20), entailment (0.50), paraphrase consistency, statement accuracy. Controls inside C5: own frame, random frame of equal degree, no frame, mean row.
- **Edits:** 200 one-relation edits per track. Measure: log-odds change toward the new fact beyond a same-pool control edit (ROME/MEMIT-style), plus neighbourhood specificity.
- **Statistics:** paired over items, seed-averaged, Holm over tests.
- **Fixed in advance (claims ledger E9.C1–C2; outline):** own frame ≈ random frame means the vector only signals "a term is here". If ROME, MEMIT or in-context editing do better at equal specificity, claim C stays a proof of concept of a zero-gradient, zero-prompt-token edit.

## Experiment
T5 on SmolLM2-135M and -360M (seeds 1–3) and Qwen3 (seeds 1–2); T4 and T7 on 360M (seeds 1–3); T1-open and WordNet at seed 1. Probes also ran at INT4.

## Results
**New words, C5 − C0′:**

| Track · host | Property selection | Entailment | Paraphrase |
|---|---|---|---|
| T5 · 360M | **+0.070 [+0.057, +0.083]*** (0.189 → 0.259) | +0.031 [+0.003, +0.058] (Holm n.s.) | **+0.084 [+0.061, +0.106]*** |
| T5 · 135M | +0.042 [+0.028, +0.055]* | +0.047 [+0.020, +0.073]* | +0.020 (n.s.) |
| T5 · Qwen3-1.7B (2 seeds) | +15.8 points* | +12.2* | — |
| T5 · Qwen3-0.6B (2 seeds) | +9.4 points* | +8.7* | — |
| T4 · 360M | +0.002 | +0.018 [+0.004, +0.032] (Holm p 0.07) | −0.000 |
| T7 · 360M | −0.004 | **+0.038 [+0.019, +0.057]*** | +0.022 (n.s.) |

- **T5: the frame's content matters.** At 135M the own frame scores 0.232–0.242 in every seed; random-frame, no-frame and mean-row controls score 0.190–0.196. The 360M gain stays at INT4 (property +0.060). On the generator's zero-shot items, property rises +0.125 (360M; 0.346 → 0.471).
- **T4: nothing moves,** and the probes sit far above floor (track property 0.58–0.63): a null, not a floor effect.
- **T7: the entailment rise is not the frame's content.** A random frame scores the same as the own frame (0.523–0.532 against 0.528–0.538 per seed; no injection 0.480–0.493), and the rise vanishes at INT4. **The refutation reading applies on T7.**
- **T1-open and WordNet** (seed 1): all n.s.

**Edits, log-odds change beyond the control edit (seed means):**

| Subset | T5 360M | T5 135M | T5 360M INT4 | T5 135M INT4 |
|---|---|---|---|---|
| seen terms | +0.25 | +0.47 | +0.28 | +0.41 |
| held-out terms | +0.03 | +0.31 | +0.02 | +0.31 |

- **T5:** edits work on seen terms at both sizes, and on held-out terms only at 135M. Absolute edit success stays low (0.40 → 0.455 on seen terms, 360M seed 1; neighbourhood specificity 0.55). Qwen3-1.7B (seed 1): seen +0.56 beyond the control, held-out +0.05.
- **T4:** −0.003 / −0.009 / +0.055 per seed, all n.s. **T7:** +0.010 / **+0.101 [+0.028, +0.170]** / +0.053; only seed 2 is significant.

**Understanding items on new words** (no option shares a word with the frame; preliminary; composite C5 − C0′): T5 360M +0.0081 [+0.0038, +0.0125]; T7 360M +0.0147 [+0.0069, +0.0219]; T4 360M +0.0027 [−0.0014, +0.0072]. Their own-vs-random-frame secondary is not read here.

## Limitations
- Absolute accuracy stays low (0.26 against chance 0.20 at 360M), and C0′ sits below chance: the probe is near its floor.
- No comparison with weight editors or in-context editing yet, so no superiority claim. No real-text track shows a content-specific effect.
- **Sources disagree; we follow the newest.** `claims.md` (E9.C2) says, from seed 1, that "edits do not transfer to terms unseen in training"; at 3 seeds that holds at 360M only (135M: +0.31). Its claim text (E9.C1) says "0.25 vs 0.19"; its evidence column and R9 give 0.259 against 0.189.

## Next steps
- **Queued:** the editing baselines (ROME, MEMIT, AlphaEdit, in-context editing) on the T5 items and hosts (28 jobs at priority 60, reports at 61; estimated ≈ 27–29 Oct); larger item sets (v2, about 700 per arm; 89 jobs at 54.6, 12 at 56); dimension 3 on T7-ROOD (54.4971) and T8 (54.4991).
- **Proposed, not queued (claims ledger C-B3 to C-B6):** same-site new-word vectors without gradient; relation-shuffled and filler-only frames; a channel-off audit and revert test; intra-entry locality. Also: read the understanding secondaries for new words on T5 and T7.

---

# Sources

- Plain-language summaries: `manuscript/update-2026-10-12.md` (E9-9, E9-10), `manuscript/update-2026-10-15.md` ("Also since the last update", "Still deferred").
- Ledgers: `manuscript/claims.md` (E9.B1, E9.C1, E9.C2), `manuscript/outline.md` (E9 framing rules).
- Report: `reports/R9-retrofit-quantization.md`.
- Pre-registration: `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §1.
- Generated reports (committed): `experiments/e9-retrofit/report/{t5,t4,t7,t5-qwen3}/report.md` and `report/{t5,t4,t7}-awqfix/` as of 2026-10-11/12; `experiments/e9-retrofit/quant/{t5,t4,t7}/report.md`, `quant-general/{t4,t7}/report.md`.
- Generated reports (**uncommitted, preliminary**): working-tree `experiments/e9-retrofit/report/t7/report.md` (135M quantization at 3 seeds); `experiments/e9-retrofit/report/{t5,t4,t7}-understanding/report.md`.
- Queue state: `python -m vsa_embed.jobqueue status`, read-only, 2026-10-20.
