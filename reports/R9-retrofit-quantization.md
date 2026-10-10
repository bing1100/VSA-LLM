# R9 — Retrofit × quantization × ontology editing (E9)

**Status (2026-10-09).**

- **T5 synthetic enterprise glossary:** SmolLM2-360M and SmolLM2-135M at **3 seeds** (1–3; P0 has no training randomness). The PQ1 controls are at 360M, 3 seeds (section "T5 controls"). Qwen3 hosts are at seeds 1–2 (section "Qwen3 on T5 at 2 seeds"); seed 3 is queued.
- **T4 chemistry (natural text):**
  - 360M dimension 1 and the PQ1 controls at 3 seeds (section "T4 chemistry — seeds 1–3"; preliminary report `report/t4-arms-prelim`).
  - T4 quantization, dimension 3 and 135M are still seed 1. The official 3-seed report is queued.
- **T1-open and the WordNet negative control:** seed 1, both null. The cross-track section shows the gain following the novelty of the vocabulary.
- **T7** (MeSH 2026 names, natural text): 360M at 3 seeds for P0, C0′, C2 and C5 (preliminary; section "T7"). After held-out terms the embedding hurts (+0.48%*).

Full generated T5 report with every table and figure: `experiments/e9-retrofit/report/t5/report.md` (frozen at commit `dd612dc`; later arm batches write their own folders). Intervals are 95% cluster bootstraps over evaluation windows (loss) or items (dimension 3), with paired differences pooled over seeds; the across-seed spread is given where it matters. The seed-1 version of this report (commit `ccb9204` and earlier) is superseded; where 3 seeds change a seed-1 conclusion this is said explicitly.

**Question (author request 2026-10-02).** (1) Does a VSA ontology channel trained jointly with a pretrained model improve it on long-token, rare and out-of-distribution words? (2) Is the gap larger after weight quantization? (3) After training, can the model learn new or changed words zero-shot by editing the ontology alone?

**Design.** P0 = original host (no training); C0′ = continued training on the same 50M tokens without the channel (full fine-tuning, host lr 3e-5); C2 = the same + a capacity-matched free per-concept table; C5 = the same + the attentive VSA channel (WordNet-style frames of the glossary ontology, HRR). T5 is generated internal-style documents over invented private terms (contamination-free: the host cannot have seen them); 360 held-out terms are never linked in training and are composed zero-shot at evaluation. Quantization: torchao weight-only INT8 / INT4 (RTN), output head in full precision; variant A keeps the channel in FP16, B quantizes it too (A ≈ B everywhere).

Why T5 first: on general WordNet vocabulary the pretrained host already models the words well, and an engagement check showed the channel was neutralized (C5 = C0′ to four decimals). The question only makes sense for vocabulary that is genuinely new to the host. T5 text is generated from the same ontology the channel reads, so effect sizes here are an upper bound; T4 (natural ChEBI text) and T1-open (PubMed) test that.

## Dimension 1 — rare and out-of-distribution words (bf16, 3 seeds)

Relative loss difference of C5, 95% CI, Holm over references within stratum (negative = better; `*` significant):

| Stratum | targets | 360M: C5 − C0′ | 360M: C5 − C2 | 135M: C5 − C0′ | 135M: C5 − C2 |
|---|---:|---|---|---|---|
| after held-out terms (never linked; zero-shot composed) | 150,369 | **−13.3% [−14.1, −12.4]*** | **−15.1% [−16.0, −14.3]*** | **−9.2% [−10.0, −8.4]*** | **−13.6% [−14.4, −12.8]*** |
| after unseen terms | 14,721 | −6.6% [−8.2, −5.1]* | −8.6%* | −3.8% [−5.4, −2.3]* | −8.0%* |
| after rare terms (train freq 1–9) | 108,909 | −6.0% [−6.6, −5.4]* | −7.6%* | −3.1% [−3.7, −2.5]* | −7.9%* |
| after 3+-subtoken terms | 1,447,566 | −7.6% [−7.8, −7.3]* | −7.5%* | −6.6% [−6.8, −6.4]* | −7.5%* |
| inside terms | 859,584 | +0.04% (n.s.) | +0.04% (n.s.) | +0.33% [+0.22, +0.45]* | +0.19%* |
| unlinked text in the domain (locality) | 1,142,448 | −0.03% (n.s.) | −0.08%* | −0.17%* | −0.22%* |
| general text (FineWeb-Edu `eval-general`), nats/token | — | −0.0004* | −0.0001* | −0.0007* | −0.0002* |

- Across seeds the held-out gain is −13.3% ± 1.7 (360M) and −9.2% ± 1.5 (135M) (mean ± SD of per-seed relative differences); every seed shows the gain on every term stratum (held-out per seed: −11.7 / −13.1 / −15.0% at 360M, −7.5 / −10.6 / −9.4% at 135M).
- The largest gain is on **held-out terms**, whose channel rows come only from composing the ontology. The free table cannot help there: C2 is *worse* than C0′ on held-out, unseen and rare terms (+4.6 to +5.2% at 135M), so the gain is from composition, not capacity.
- The gain grows with model size on held-out, unseen and rare terms (360M > 135M) and, with LoRA, is larger again on Qwen3 (addendum).
- Locality: no cost on ordinary or general text. The one cost is +0.33% *inside* terms at 135M (n.s. at 360M).
- General lexical probes (WiC, WSD, CARD-660, Rare Words, BLESS, HyperLex, LAMBADA) do not change, as expected: the channel was trained on glossary vocabulary only.

## Dimension 2 — the gap under quantization (3 seeds)

INT8 is nearly free for every model (≤ 1.4% loss). INT4 (RTN) costs the fine-tuned models far more than the original host on this domain (all tokens: C0′ +38% at 360M and +63% at 135M, vs +5.6% / +7.7% for the original host): the fine-tuned models sit at a very low domain loss (≈ 0.6 nats), where the same weight noise costs proportionally more.

C5's advantage (gain = C5 − C0′, nats/token) at bf16 vs INT4 (channel FP16; B is the same), Δgain = difference in differences with 95% CI (negative = the advantage grows):

| Stratum | 360M bf16 → INT4 | retained | Δgain | 135M bf16 → INT4 | retained | Δgain |
|---|---|---:|---|---|---:|---|
| after held-out terms | −0.115 → −0.066 | **0.58×** | +0.049 [+0.043, +0.054]* | −0.082 → −0.094 | **1.14×** | −0.012 [−0.019, −0.005]* |
| after unseen terms | −0.058 → −0.056 | 0.97× | +0.002 (n.s.) | −0.036 → −0.029 | 0.80× | +0.007 (n.s.) |
| after rare terms | −0.050 → −0.038 | 0.77× | +0.012* | −0.028 → −0.011 | **0.39×** | +0.017* |
| after 3+-subtoken terms | −0.045 → −0.039 | 0.85× | +0.007* | −0.043 → −0.040 | 0.93× | +0.003* |
| unlinked text in the domain | −0.000 → −0.020 | — | −0.020* | −0.001 → −0.012 | — | −0.011* |
| all tokens | −0.021 → −0.026 | 1.26× | −0.005* | −0.019 → −0.025 | 1.32× | −0.006* |

Against the free table C2 the term strata are the same or better retained (360M held-out 0.61×; 135M held-out 1.23×, rare 1.02×, unseen 0.99×).

- **The author's hypothesis ("the gap is bigger after quantization") is not supported on the target terms.** Under INT4 the channel keeps most of its advantage on new and rare terms (retained 0.4–1.1×, mostly 0.6–1.0×), and only 135M held-out terms grow (1.14×). Seed 1 had suggested growth at 135M on rare and long terms (1.09–1.39×); with 3 seeds this does not hold (rare 0.39×, long 0.93×).
- **What does grow is robustness on ordinary domain text:** at INT4 the channel models lose 2–4 points less on unlinked domain text than C0′ (360M: +47% vs +51%; 135M: +95% vs +97%), which drives the 1.26–1.32× overall gain. The channel does not touch these tokens, so the likely cause is that the host weights fine-tuned with a channel are less fragile. This is a hypothesis; the PQ1 controls (channel switched off at INT4; GPTQ/AWQ/HQQ/NF4; quantized embeddings) test it.
- General-text INT4 damage is the same for all trained models (Δgain ≤ 0.0013 nats): the channel neither protects nor hurts general text.

## Dimension 3 — zero-shot learning by editing the ontology (no weight update, 3 seeds)

**New words** (300 invented terms whose frames are new combinations of existing atomics and relations, added to the ontology after training). C5 own − C0′, paired over items, seed-averaged, Holm over tests (chance 0.20 for 5-way property selection, 0.50 for entailment):

| Test | 360M C0′ → C5 | Δ [95% CI] | 135M Δ [95% CI] |
|---|---|---|---|
| property selection | 0.189 → 0.259 | **+0.070 [+0.057, +0.083]*** | +0.042 [+0.028, +0.055]* |
| entailment | 0.503 → 0.534 | +0.031 [+0.003, +0.058] (Holm n.s.) | +0.047 [+0.020, +0.073]* |
| paraphrase consistency | 0.321 → 0.404 | **+0.084 [+0.061, +0.106]*** | +0.020 (n.s.) |
| statement accuracy | — | +0.027 [+0.014, +0.041]* | +0.036 [+0.022, +0.049]* |

Same against C2 (360M property +0.063*, paraphrase +0.068*). Controls inside C5: own frame > random frame of equal degree ≈ no frame ≈ mean row (135M property 0.232–0.242 vs 0.190–0.196 in every seed), so the gain comes from the frame content. At INT4 the 360M new-word gain stays (property +0.060, paraphrase +0.073).

**Track zero-shot items** (contamination-free synthetic terms from the T5 generator, E5.4 format), C5 − C0′ seed means: property **+0.125** (360M; 0.346 → 0.471, per-seed SD 0.005) and +0.123 (135M); entailment +0.096 / +0.102; paraphrase +0.020 / +0.061. At INT4: property +0.125 / +0.095.

**Edits** (change one relation–filler of an existing term; ROME/MEMIT-style metrics; only C5 is editable): log-odds change toward the new fact beyond a control edit, seed means:

| Subset | 360M bf16 | 360M INT4 | 135M bf16 | 135M INT4 |
|---|---:|---:|---:|---:|
| seen terms | +0.25 | +0.28 | +0.47 | +0.41 |
| held-out terms | +0.03 | +0.02 | +0.31 | +0.31 |

Editing works on seen terms at both sizes; on held-out terms it works at 135M but not at 360M. Absolute edit success stays low (ES ≈ 0.40 → 0.46 on seen terms at 360M seed 1); the ROME/MEMIT/AlphaEdit/IKE baselines on the same items are queued (dimension-3 baselines, priority 60–61).

## Findings and implications (updated 2026-10-09)

1. **On vocabulary that is genuinely new to the host, retrofitting works and replicates.**
   - **T5:** a jointly trained, ontology-composed embedding lowers a pretrained model's loss after new and held-out glossary terms by 3–13%, with no locality cost. Held-out terms: −13.3% at 360M, −9.2% at 135M (3 seeds; SD across seeds ≤ 1.7 points). On Qwen3 with LoRA the gain is larger: −17.7% and −20.2% (2 seeds).
   - **Natural text:** on T4 chemistry (360M, 3 seeds) the same direction holds, much smaller: held-out −0.40%, unseen −2.29%, rare −0.91%, 3+-subtoken −1.09%. On T7 (MeSH 2026 names, 3 seeds, preliminary) it helps after rare (−0.50%) and 3+-subtoken terms (−0.57%), but hurts after held-out terms (+0.48%*), whose names the host reads unlinked in training. On T1-open and WordNet, whose terms the host already models, there is no gain.
   - **Scope:** the gain follows how new the vocabulary is to the host. T5's sizes are a synthetic upper bound.
2. **The mechanism is compositional parameter sharing, not the binding operator.**
   - **T5:** untyped composition equals HRR, translation x + r beats it (−17.2% vs C0′), and a fixed random binding costs 5.2%.
   - **T4:** all four operators tie.
   - **Consequence:** by rule 7, E9 is not evidence that HRR binding is the source.
3. **What the composed vector adds beyond a generic domain vector differs between the tracks.**
   - **T5:** the term's own frame is necessary (shuffled frames are worse than no channel). Composing it beats subtoken-mean, definition-encoder and TransE rows at the same site by 10.8–15.3% on held-out terms.
   - **T4, held-out terms:** a shuffled frame keeps the whole gain, and the definition encoder is ahead (claim A's refutation reading is triggered). The frame swap explains why. These are frequent real terms that the host learns as unlinked text. Within the trained model their own composed vector beats another term's (+0.19%*), but injecting nothing beats both (−0.19%*). C5's held-out gain therefore comes from the channel-trained host weights, not from the zero-shot vector. On T7, whose held-out names occur unlinked in 22.7% of the training abstracts, the injection cost wins: C5 − C0′ +0.48%* after held-out terms.
   - **T4, unseen terms (no training occurrence):** the own composed vector helps and is term-specific (frame swap: empty +1.60%*, other +1.33%*), so zero-shot composition does work on natural text for terms absent from training. The same holds after rare seen terms.
   - **Against the alternatives (T4):** on the unseen and rare strata a frozen TransE row ties composition, the definition encoder ties it after rare and unseen terms, and only the subtoken mean is clearly beaten.
   - **Open question:** is the natural-text zero-shot benefit specific to composition? T7 and the decision-64 controls (C5sh and C6d on T7, T7-ROOD and T8) decide it.
4. **Copying explains part of T5, little of T4.**
   - **T5:** 86–90% of the rare- and unseen-term gain is on frame-filler tokens, but two thirds of the held-out gain is on other tokens.
   - **T4 (135M, seed 1):** 42% of the unseen-term gain and 2% of the held-out gain are on filler tokens. The 360M rescoring at 3 seeds is queued.
5. **Quantization: the advantage is largely preserved under INT4 and does not grow on the target terms.**
   - **T5 retention (held-out):** 0.58× at 360M and 1.14× at 135M. GPTQ keeps it fully (DiD −0.001).
   - **Channel off at both precisions:** the held-out DiD vanishes. So the RTN loss comes from the quantized host reading the channel less well.
   - **All-token robustness (1.26–1.32×):** a property of the C5-trained host weights, not of the channel.
   - **T4 (seed 1):** INT4 shrinks the gain on every term stratum, so "preserved" is a T5 result until the 3-seed T4 report says otherwise.
   - **Paper wording:** "largely preserved, fully under GPTQ", never "the gap grows".
6. **Zero-shot learning by ontology editing is real but modest on T5, and null on T4 (seed 1).**
   - **New words:** adding a frame for an unseen word improves property selection by 4–7 points on SmolLM2 and 9–16 points on Qwen3. It also improves paraphrase consistency by up to 8 points.
   - **Fact edits:** editing an existing fact moves seen terms reliably, and held-out terms only at 135M.
   - **Absolute accuracies stay low:** 0.26 vs chance 0.20 for new-word properties at 360M.

## Next

- **Official T4 report** (priority 54): quantization and dimension 3 at 3 seeds, and the 360M filler split.
- **The official T7 report, then T7-ROOD** (held-out documents removed from training: the clean zero-shot test), and the decision-64 controls (C5sh and C6d, 3 seeds, on T7, T7-ROOD, T8 and T1c-ROOD) with the frame swap on those tracks.
- **Remaining E9 batches:** dimension 3 on the v2 items (≈ 700 per arm), Qwen3 seed 3, the Qwen3.5 block (2B, 0.8B), and the dimension-3 knowledge-editing baselines.

## T5 controls (WP-PQ1; SmolLM2-360M, 3 seeds, 2026-10-07)

Full tables: `experiments/e9-retrofit/report/t5/report.md`, sections "WP-PQ1" (job `t5-report-s1-2-3-pq-SmolLM2-360M`). Same recipe, test set and parameters as C5; seeds 1–3 each; relative loss differences with 95% window bootstraps, Holm over arms within stratum.

**What the composed vector needs (operator and specificity ablation).** C5 − arm on held-out / rare / unseen / 3+-subtoken terms (negative = C5 better), and each arm − C0′ on held-out terms:

| Arm | C5 − arm | arm − C0′ (held-out) |
|---|---|---|
| C5rf: fixed random orthogonal binding | −5.2%* / −0.3%* / −0.3% / −1.1%* | −8.5% |
| C5ut: untyped (fillers only, no binding) | +0.2% / +0.1% / +0.4% / +0.1%* (equal) | −13.4% |
| C5tr: translation x + r | **+4.8%* / +2.9%* / +2.7%* / +2.1%*** (translation better) | **−17.2%** |
| C5sh: shuffled frames (another term's frame) | **−17.0%* / −8.2%* / −8.9%* / −7.6%*** | **+4.6%** (worse than no channel) |

- **The frame's content is necessary** on T5: with another term's frame the channel is worse than none. On T4 natural text (3 seeds, section "T4 chemistry — seeds 1–3 and the PQ1 controls") this holds after seen, rare and unseen terms but **not after held-out terms**, where shuffled frames keep C5's whole gain.
- **Binding is not**: dropping relation roles costs nothing, a random fixed binding costs 5% on held-out terms, and plain translation beats HRR on every term stratum. By rule 7 (outline), E9 is described as compositional parameter sharing over ontology frames; "HRR" names one operator in the ablation, the best one here is translation. The from-scratch operator screen (decision 54) decides the same question for training from scratch.

**Same-site baselines that also give an unseen term a vector without training on it** (frozen per-entry source vector through a trained projector, parameter-matched, same site and gate). C5 − arm on held-out / rare / unseen / 3+-subtoken, and arm − C0′ on held-out:

| Arm | C5 − arm | arm − C0′ (held-out) |
|---|---|---|
| C6m: subtoken mean (FVT / Hewitt-style new-token initialization) | −14.2%* / −6.9%* / −7.4%* / −7.5%* | +1.1% (no help) |
| C6d: definition encoder (the host encodes the verbalized frame) | −10.8%* / −2.8%* / −3.3%* / −3.8%* | −2.8% |
| C6g: TransE KG embedding of the entry | −15.3%* / −5.3%* / −5.9%* / −6.1%* | +2.4% (no help) |

The composed vector beats all three by wide margins (claim A's refutation reading "new-token initialization or a same-site frame-text vector matches the composed vector" is **not** triggered on T5). Encoding the same frame as text recovers only about a fifth of the gain. On T4 the reading **is** triggered after held-out terms: the definition encoder is ahead of C5 there and the other two tie it (section "T4 chemistry — seeds 1–3 and the PQ1 controls").

**Copy concern: filler vs non-filler continuation tokens** (`e9_rescore`; a filler target is a token in the 8 tokens after a term that belongs to an alias of a filler of that term's frame; 19–23% of targets). C5 − C0′:

| After… | total | on filler tokens | on other tokens | share of the gain on fillers |
|---|---|---|---|---|
| held-out terms | −13.3% | −8.1%* | **−19.4%*** | 0.33 |
| rare terms | −6.0% | −8.7%* | −2.1%* | 0.86 |
| unseen terms | −6.6% | −9.0%* | −1.9% | 0.90 |
| 3+-subtoken terms (≈ all linked) | −7.6% | −12.5%* | −3.5%* | 0.75 |

- For rare and unseen terms **most of the gain (86–90%) is predicting the frame's filler words**: the copy effect the novelty check warned about. For held-out terms, whose rows come only from composition, two thirds of the gain is on other tokens (−19.4%), consistent with the composed row telling the model what kind of term follows (in T5's templated text, the type-specific continuation) rather than which filler word comes next.
- Filler shares on T4 (first 256 windows; execution.md, WP-PQ1): 3.7% after any term, 0.8% after held-out, 10.3% after rare and 28.9% after unseen terms, against 18.6–23.9% on T5. T4's largest gain is after unseen terms (−2.3 / −3.7%), its stratum with the most filler continuations. The T4 rescoring at 135M (seed 1; `experiments/e9-retrofit/report/t4-arms-prelim/`, "filler vs non-filler") puts 42% of the gain after unseen terms on filler tokens (−4.5%* on fillers, −3.3%* on other tokens), 22% after rare terms, 10% after 3+-subtoken terms and 2% after held-out terms: on natural text most of the gain is **not** filler copying. The 360M rescoring at seeds 1–3 is queued (priority 52).

**Claim-B controls (360M, C5 vs C0′; DiD = change of the C5 − C0′ gap from bf16 to 4-bit, positive = the advantage shrinks):**

| Quantizer | held-out DiD | rare DiD | all-token DiD |
|---|---|---|---|
| RTN (torchao, group 128) | +0.049* | +0.012* | −0.005* |
| HQQ | +0.055* | +0.026* | −0.003* |
| NF4 | +0.033* | +0.004* | −0.002* |
| GPTQ | **−0.001 (n.s.)** | −0.005* | +0.002* |
| AWQ | +0.025* | +0.005* | −0.047* |
| RTN with the input embedding quantized too | +0.043* | +0.009* | −0.009* |

- With GPTQ the held-out advantage is fully kept; with the other quantizers 20–50% of it is lost. Variant B and a quantized embedding change little.
- **Channel off at both precisions**: the held-out DiD disappears (−0.001, n.s.), so the held-out loss under RTN comes from the channel's contribution being read less well by the quantized host. The all-token robustness of the C5-trained host is a property of its weights (DiD with the channel off −0.012*), not of the channel at inference: the "separate module survives PTQ" mechanism, not a structure effect.
- Claim B therefore stays at "the gain is largely preserved under 4-bit PTQ, fully under GPTQ", with the robustness on ordinary text attributed to the host weights.

## T4 chemistry — seeds 1–3 and the PQ1 controls (SmolLM2-360M, 2026-10-09)

Full tables: `experiments/e9-retrofit/report/t4-arms-prelim/report.md` (`e9_report` on `runs/t4` without quantization inputs, 10,000 resamples). It is preliminary: the official `t4-report-s1-2-3-pq-SmolLM2-360M` (priority 54) adds quantization, dimension 3 at 3 seeds and the 360M filler split, and should reproduce these numbers. Seeds 1–3 for C0′, C2, C5 and every arm; P0 is seed-independent. Relative loss differences, 95% window bootstraps pooled over seeds, Holm within stratum.

**Dimension 1 at 3 seeds** (C5 − C0′; the seed-1 values of the next section hold):

| After held-out | after unseen | after rare | after 3+-subtoken | inside | unlinked |
|---|---|---|---|---|---|
| −0.40% [−0.54, −0.25]* | −2.29% [−2.72, −1.90]* | −0.91% [−1.06, −0.77]* | −1.09% [−1.16, −1.01]* | −1.54% [−1.67, −1.41]* | −0.01%* |

C5 − C2 is negative on every term stratum (−0.22% after held-out to −2.06% after unseen).

**Arms** (C5 − arm after held-out / unseen / rare / 3+-subtoken terms, negative = C5 better; arm − C0′ after held-out):

| Arm | C5 − arm | arm − C0′ (held-out) |
|---|---|---|
| C5rf: fixed random orthogonal binding | −0.05% / +0.09% / +0.06% / +0.01% (all n.s.) | −0.35%* |
| C5ut: untyped (no binding) | −0.04% / −0.03% / +0.05% / −0.00% (all n.s.) | −0.36%* |
| C5tr: translation x + r | +0.06% / +0.14% / **+0.11%*** / −0.02% | −0.45%* |
| C5sh: shuffled frames | **+0.01% [−0.07, +0.10]** / **−0.82%*** / **−0.37%*** / **−0.20%*** | −0.41%* |
| C6m: subtoken mean | +0.03% / **−0.48%*** / **−0.21%*** / **−0.20%*** | −0.42%* |
| C6d: definition encoder | **+0.17% [+0.05, +0.29]*** / +0.19% / −0.00% / **−0.20%*** | **−0.56%*** |
| C6g: TransE KG embedding | +0.10% / −0.04% / +0.02% / +0.05%* | −0.50%* |

- **The binding operator does not matter on natural text either**: fixed random binding and untyped composition tie learned HRR on every stratum; translation is within ±0.14%.
- **After held-out terms the gain is not frame-specific.** In C5sh every entry, held-out entries included, reads another entry's frame (`training/lm.py`, a derangement over all entries), and C5sh keeps C5's whole held-out gain. On seen terms the own frame matters (after unseen −0.82%*, rare −0.37%*, 3+-subtoken −0.20%*), as on T5. The held-out gain on T4 therefore does not come from the term's relations. The frame swap (next section) locates it in the channel-trained host weights: within C5, injecting nothing at held-out terms beats their own vector.
- **Claim A's refutation reading is triggered after held-out terms on T4**: the definition encoder (the host encoding the same frame as text) beats C5 there (+0.17%*), and subtoken mean and TransE tie it.
- **Beyond held-out terms, composition beats only the subtoken mean.**
  - **Subtoken mean (C6m):** C5 is ahead after unseen (−0.48%*), rare (−0.21%*) and 3+-subtoken terms (−0.20%*).
  - **Definition encoder (C6d):** C5 is ahead only inside and after long terms (−0.50%* inside, −0.20%* after 3+-subtoken terms). It ties after rare (−0.00%) and unseen terms (+0.19%, n.s.).
  - **TransE row (C6g):** ties C5 on every stratum, and is slightly ahead after 3+-subtoken terms (+0.05%*). A frozen KG embedding of the entry, through a trained projector, does as well as composing the entry's frame.
- **T5 versus T4.** On T5 the same controls go the other way: C5sh is 4.6% worse than no channel, and C5 is ahead of every same-site vector by 10.8–15.3%.
  - "The frame's content is necessary" holds on T5, and on T4 after seen, rare and unseen terms.
  - "Composition beats same-site vectors" is a T5 result. On T4 it holds only against the subtoken mean, plus the definition encoder on long terms.

## Frame swap: does the trained model use a term's own frame? (decision 64, 2026-10-09)

**Method.**
- **Pre-registration:** `experiments/e9-retrofit/preregistration-frameswap.md`, committed before any run. Reports are under `experiments/e9-retrofit/frameswap/{t5,t4,t1,wordnet}/report/`.
- **What is swapped:** each finished run is re-scored on its own evaluation windows. Only a target set's rows change; every other span keeps its own row.
- **Variants:**
  - `other`: another target term's frame.
  - `other-any`: a random non-target term's frame.
  - `empty`: no injection.
  - `mean`: the mean non-target row.
- **Measure:** variant − own, relative to own's loss, so positive means the term's own row helps. Pooled over seeds per window, with a window bootstrap and Holm over the four variants.
- **Primary endpoint:** `other − own` after held-out terms (C5, 360M). The pre-registered reading rule counts a held-out gain as ontology-specific only if this is > 0 after Holm.

**SmolLM2-360M C5, 3 seeds, on the matched stratum of each target set:**

| Track · target set | other | other-any | empty | mean |
|---|---|---|---|---|
| T5 · held-out | **+24.4%*** | +23.8%* | **+28.7%*** | +21.2%* |
| T5 · unseen | +20.4%* | +23.0%* | +16.6%* | +14.3%* |
| T5 · rare seen | +21.5%* | +21.7%* | +16.9%* | +13.4%* |
| T4 · held-out | **+0.19% [+0.13, +0.26]*** | +0.30%* | **−0.19% [−0.25, −0.12]*** | −0.07% (n.s.) |
| T4 · unseen | +1.33%* | +1.25%* | +1.60%* | +1.07%* |
| T4 · rare seen | +0.75%* | +0.81%* | +0.79%* | +0.52%* |

**Other models after held-out terms (T4):**
- **C6d (360M, 3 seeds):** other +0.35%*, empty +0.01% (n.s.).
- **C5sh (360M, 3 seeds):** other +0.12%*, empty −0.11%*.
- **C5 at 135M (seed 1):** other +0.19%*, empty −0.55%*. On unseen terms at 135M: other +1.89%*, empty +2.90%*.

**T1-open and WordNet (C5, both hosts, seed 1):** the primary is null (−0.00% to +0.02%). The only significant effects are T1 135M empty −0.04% and mean −0.05% after held-out terms. The host ignores the vector, as the null C5 − C0′ predicts.

**Locality:** the `unlinked` stratum moves by ≤ 0.01% on T4 and by ≤ 0.37% on T5, against 16–29% on T5's matched strata. `own` replays each run's final evaluation (equal counts).

**Reading.**
- **T5:** the own frame carries a large part of the loss after every kind of term, held-out terms included (the vector is used and term-specific). The synthetic zero-shot result stands.
- **T4 held-out terms:** the primary endpoint passes, against the pre-registered prediction (≈ 0). Within the trained model a held-out term's own composed vector is better than another held-out term's. But injecting nothing is better still (−0.19%*, 135M −0.55%*), and a constant mean row is no worse. The own row is read term-specifically, but it is a net cost: it is only the least harmful row.
  - **Who the held-out terms are:** T4's real held-out terms (551 of 850; the other 299 are synthetic and absent from the text) were chosen among entries with ≥ 5 occurrences in a presample of the training documents (`t4.yaml`, `holdout_min_count: 5`). They appear in the training text, unlinked, so the host learns them as plain tokens and never receives an injection at their positions. At evaluation the injection is new to it.
  - **Where C5's held-out gain comes from:** C5's held-out gain over C0′ (−0.40%) is therefore not delivered by the zero-shot vector. It comes from the host weights trained alongside the channel. This fits C5sh keeping the gain (with its own empty −0.11%*) and the free table's −0.18% after held-out terms (C2 has no held-out rows).
  - **Definition encoder:** its held-out row costs nothing and is term-specific. So it wins there (C5 − C6d +0.17%*) because its row does no harm, not because it helps more.
- **T4 unseen terms:** these are linked entries with no training occurrence, and 68% of them share no concept with any trained entry. The own composed vector helps and is term-specific (own vs empty 1.60%, vs another unseen term's frame 1.33%). The same holds after rare seen terms. On natural text, then, composing a frame for a term absent from training does help. What does not help is injecting a vector for a term the host already learned from text without one.
- **What the frame swap does not settle:** on these strata a frozen TransE row or the definition encoder matches C5 (section "T4 chemistry — seeds 1–3"). The frame swap shows the vector is used; it does not show composition is better than those alternatives.

## T4 chemistry — natural text (SmolLM2-360M / 135M, seed 1, 2026-10-05)

*360M dimension 1 is now at 3 seeds (previous section; the seed-1 values below hold). Dimensions 2 and 3 and the 135M results are still seed 1.*

Full tables: `experiments/e9-retrofit/report/t4/report.md`. T4 links ChEBI entities (IUPAC and trivial names as multi-token aliases) in ChEBI entry texts and PubMed chemistry abstracts, half general text (FineWeb-Edu); same recipe, design and test protocol as T5. This is the test of the copy concern: T5 text is generated from the ontology the channel reads, T4 text is not (except the ChEBI entry texts, which are curated definitions). Seed 1 only; intervals are over windows/items.

**Dimension 1 (bf16), C5 − C0′ relative loss [95% CI], all Holm-significant:**

| Stratum | targets | 360M: C5 − C0′ | 360M: C5 − C2 | 135M: C5 − C0′ | 135M: C5 − C2 | T5 360M (3 seeds), for scale |
|---|---:|---|---|---|---|---|
| after held-out terms | 35,747 | −0.40% [−0.56, −0.24] | −0.15% | −0.63% [−0.85, −0.41] | −0.40% | −13.3% |
| after unseen terms | 9,792 | **−2.32% [−2.78, −1.90]** | −2.16% | **−3.70% [−4.17, −3.23]** | −3.63% | −6.6% |
| after rare terms | 31,458 | −0.87% [−1.03, −0.72] | −0.85% | −1.62% [−1.82, −1.43] | −1.56% | −6.0% |
| after 3+-subtoken terms | 185,109 | −1.08% [−1.16, −1.00] | −0.84% | −1.73% [−1.83, −1.62] | −1.46% | −7.6% |
| inside terms | 130,661 | −1.55% [−1.68, −1.42] | −1.13% | −2.03% [−2.19, −1.88] | −1.73% | +0.04% |
| unlinked text (locality) | 739,780 | −0.01% | −0.01% | −0.03% | −0.02% | −0.03% |

- **The direction replicates on natural text; the size does not.** C5 beats both C0′ and the free table on every term stratum with no locality cost, but by 0.4–3.7% instead of 6–13%. The held-out-term gain, the largest on T5, is the smallest on T4 (−0.4 / −0.6%). The PQ1 rescoring (section "T5 controls") shows the T4/T5 held-out gap is not filler copying: on T5 only a third of the held-out gain is on filler tokens, two thirds on other tokens (−19.4%), whereas filler copying carries 75–90% of the gain after rare, unseen and multi-subtoken terms. The held-out gap more likely reflects T5's templated, type-specific continuations, which a composed row can predict and natural text lacks; T4's filler share is also far smaller.
- **The gain is largest after unseen terms** (−2.3 / −3.7%) and **inside terms** (−1.6 / −2.0%; zero on T5). The inside gain is consistent with nested chemical names, where a shorter linked name ends inside a longer one and its injected vector helps predict the rest (not yet checked).
- **On T4 the smaller host gains more** (135M > 360M on every stratum), the reverse of T5.

**Dimension 2 (INT4 RTN, channel FP16):** the advantage shrinks under INT4 on every term stratum (retained 0.50–0.99×; DiD significant at 135M on inside, 3+-subtoken, rare and held-out). At 135M the held-out-term advantage reverses (C5 0.009 nats worse than C0′ at INT4). On natural text quantization does not enlarge the gap; it partly erases it.

**Dimension 3 (zero-shot by ontology editing): no effect on T4.** New invented words: property selection +0.003 [−0.011, +0.015] (135M) / +0.003 (360M), entailment +0.005 / +0.025 (Holm n.s.), paraphrase −0.020 / −0.006; the WP-C7 track zero-shot items do not move either (360M property 0.626 → 0.622). Edits: log-odds change beyond the control edit +0.01 to +0.10, none significant. Probes sit far above floor here (track property 0.58–0.63), so this is a null, not a floor effect.

**What T4 means for the paper (seed 1; seeds 2–3 are in the PQ1 block):**
1. Claim A survives in direction but must be stated at natural-text size: about 1–4% after unseen, rare and multi-subtoken terms, under 1% after held-out terms; T5 is a synthetic upper bound.
2. Claim B (quantization) is weaker still: the gain is partly lost under INT4 on natural text.
3. Claim C (zero-shot learning by editing the ontology) does not transfer to natural text at seed 1: it is a synthetic-glossary result (T5 SmolLM2 and Qwen3) until a natural-text track shows it.

## T7 — new MeSH 2026 vocabulary in PubMed 2025–26 (SmolLM2-360M, 3 seeds, 2026-10-10; preliminary)

**Sources and status.**
- **Point estimates:** from the runs' final evaluations (checked here).
- **Intervals:** from a scratchpad `e9_report` (2,000 resamples) run by the peer session. The official `t7` report (priority 54, with C5sh and C6d once trained) supersedes them.
- **Track:** T7 links MeSH 2026 supplementary-concept names in PubMed 2025–26 abstracts (decision 55). It has no unseen stratum.

| Stratum | C5 − C0′ | C2 − C0′ | C0′ − P0 (novelty) |
|---|---|---|---|
| after held-out | **+0.48% [+0.36, +0.61]*** (worse) | −0.02% | −10.2% [−10.6, −9.8] |
| after rare | −0.50% [−0.65, −0.36]* | −0.18% | −7.2% |
| after 3+-subtoken | −0.57%* | −0.46% | −9.9% |
| inside | −0.28%* | −0.25% | −34.1% |
| unlinked | −0.01% | −0.01% | −8.3% |

After held-out terms C5 is also worse than the free table (C5 − C2 +0.50%*).

**The recorded prediction fails on the primary stratum.** Before running, the prediction (decision 55; section "Across tracks") was a gain at least T4's and below T5's. After held-out terms the composed embedding makes the loss **worse**.

**Mechanism.**
- **The held-out names are in the training text.** They occur, unlinked, in 15,530 of the 68,264 domain training abstracts (22.7%; 6.76M of 57.27M tokens). Source: `experiments/t7-new-vocabulary/preregistration-rood.md` and `ROOD.md` §2–3, committed before these results. Held-out aliases leave the training linker (`AliasTable.without_holdout`), so the host learns the names as plain tokens and never receives an injection at their positions.
- **Injecting there costs.** On T4 the frame swap showed that injecting a vector at such positions is a net cost (no injection beats the own vector by 0.19%*). On T7 that cost outweighs any gain from the jointly trained host weights.

**The novelty premise was also wrong.** Continued training lowers the loss after T7's held-out names by only 10.2% (T4: 40.1%), close to the 8.3% on ordinary text. Only the names themselves are new to the host (inside −34.1%).
- By the cross-track ordering, a small gain was the expected size.
- The seen strata fit that: rare −0.50%, 3+-subtoken −0.57%, between T1's null and T4's −0.9% / −1.1%.

**The clean zero-shot test is T7-ROOD.** It drops the 15,530 abstracts and refills to the same budget (leakage audit 0). It is pre-registered with P1: C5 − C0′ < 0 after held-out terms.

## T1-open — MeSH terms in PubMed (SmolLM2-360M / 135M, seed 1, 2026-10-05)

Full tables: `experiments/e9-retrofit/report/t1/report.md`. MeSH descriptors linked in PubMed abstracts (open-clinical substitute for SNOMED CT / MIMIC), half general text; same recipe and protocol. Two of 30,915 MeSH entries have no frame edges and get no injection (`channel.skip_empty_frames`, opt-in; linking and strata unchanged).

**The channel does nothing on T1.** C5 − C0′ relative loss [95% CI]: held-out terms +0.02% [−0.01, +0.04] (360M) / +0.01% [−0.02, +0.04] (135M); unseen +0.00% / −0.15% (both n.s.; 664 targets); rare −0.01% / +0.02% (n.s.); 3+-subtoken −0.05% / −0.13% (significant but tiny); inside +0.05% / −0.02%; unlinked 0.00%. C2 is the same. INT4 changes nothing (DiD ≤ 0.001 nats). Dimension 3: new-word property +0.000 / +0.002, entailment +0.010 / −0.002, paraphrase −0.016 / +0.015, all n.s.

## WordNet negative control (SmolLM2-360M / 135M, seed 1, 2026-10-05)

Full tables: `experiments/e9-retrofit/report/wordnet/report.md`. General FineWeb-Edu text with WordNet multi-token words linked (C3 linker and holdout), the vocabulary the host already models. **Null, as predicted:** every term stratum within ±0.06% at both sizes (held-out −0.00% [−0.02, +0.01] at 360M); INT4 Δgain ≤ 0.0025 nats (n.s. on term strata); new-word property +0.001 / −0.003, entailment +0.003 / −0.001 (n.s.). The engagement check's "C5 = C0′" now holds at 50M tokens with intervals.

## Across tracks: the gain follows how new the vocabulary is to the host

A simple novelty index is how much plain continued training (C0′) lowers the loss after a track's terms relative to the original host (P0): large when the terms are new to the host, small when it already models them. Seed 1, relative change after held-out / unseen / rare / 3+-subtoken terms:

| Track | 360M novelty (C0′ − P0) | 360M channel gain (C5 − C0′) | 135M novelty | 135M channel gain |
|---|---|---|---|---|
| T5 invented glossary (synthetic) | −74 / −74 / −75 / −81% | −11.7 / −5.5 / −5.6 / −7.1% | −74 / −73 / −73 / −80% | −7.5 / −3.4 / −3.1 / −6.4% |
| T4 chemistry (ChEBI, PubMed) | −40 / −56 / −52 / −45% | −0.4 / −2.3 / −0.9 / −1.1% | −36 / −51 / −48 / −40% | −0.6 / −3.7 / −1.6 / −1.7% |
| T7 MeSH 2026 names (PubMed 2025–26; 3 seeds) | −10.2 / — / −7.2 / −9.9% | **+0.48** / — / −0.50 / −0.57% | not run | not run |
| T1-open (MeSH, PubMed) | −8 / −5 / −6 / −8% | +0.0 / +0.0 / −0.0 / −0.1% | −8 / −4 / −6 / −8% | +0.0 / −0.2 / +0.0 / −0.1% |
| WordNet general (negative control, 50M tokens) | -0.6 / -0.6 / -0.8 / -0.8% | -0.00 / +0.04 / +0.01 / +0.01% | -0.3 / -0.3 / -0.3 / -0.4% | -0.00 / +0.02 / -0.06 / -0.00% |

- **Ordering:** the gain is largest where the vocabulary is newest to the host (T5), smaller on T4, and zero where the host already models the terms (T1, WordNet). Within T4 the unseen stratum is both the most novel (−56%) and the most improved (−2.3%).
- **Implication for the paper:** the channel is a tool for vocabulary that is genuinely new to the host (invented, private or fast-changing terms), not for terms already well represented in pretraining. That is the defensible scope of claim A, and it matches the host learning to ignore the channel on known vocabulary (engagement check).
- **Caveat:** three tracks and one synthetic track are not a dose–response curve. T5's sizes also contain filler copying (75–90% of the gain after rare, unseen and multi-subtoken terms) and templated, type-specific continuations (most of the held-out gain; section "T5 controls"). T4 seeds 1–3 reproduce the seed-1 sizes at 360M (−0.40 / −2.29 / −0.91 / −1.09%). T7 (MeSH 2026 supplementary-concept names, chosen as new to the host) was predicted, before running, to show a gain at least T4's and below T5's. It failed: its novelty index is only −7 to −10%, and after held-out terms, whose names the host reads unlinked in training, the embedding hurts (+0.48%*; section "T7"). Its seen strata (−0.50 / −0.57%) sit where its novelty puts them. T7-ROOD is the clean test.

## Qwen3 on T5 at 2 seeds (2026-10-06)

Full tables: `experiments/e9-retrofit/report/t5-qwen3/report.md` (job `t5-qwen3-report-s2`; seeds 1–2 pooled, P0 seed 1). The seed-1 addendum below is superseded where the numbers differ.

| | Qwen3-0.6B (LoRA) | Qwen3-1.7B (LoRA) |
|---|---|---|
| **Dimension 1**, C5 − C0′ (bf16): after held-out terms | **−20.2% [−21.3, −19.1]** (per seed −20.3 / −19.9) | **−17.7% [−18.6, −16.8]** (−16.9 / −18.5) |
| after unseen / rare / 3+-subtoken terms | −8.9% / −8.7% / −9.4% | −10.6% / −11.2% / −9.4% |
| unlinked text | −0.02% (n.s.) | −0.01% (n.s.; +0.09% vs C2) |
| **Dimension 2**, gain retained at INT4 (held-out / rare / unseen / 3+-subtoken / all tokens) | 0.91× / 0.79× / 1.42× (n.s.) / 0.89× / 0.25× | 1.07× (n.s.) / 1.14× (n.s.) / 1.59× / **1.70×** / 2.88× |
| **Dimension 3**, new words, C5 − C0′: property / entailment / paraphrase / statement accuracy (points) | **+9.4 / +8.7 / +9.0 / +6.2** | **+15.8 / +12.2 / +7.9 / +10.8** |

- Dimensions 1 and 3 replicate across seeds on both Qwen3 sizes, and are larger than on SmolLM2 (held-out −18 to −20% vs −9 to −13%; new-word property +9 to +16 points vs +4 to +7). These are still T5 (synthetic) numbers, i.e. upper bounds (see T4 and "Across tracks").
- Dimension 2 stays mixed: at 1.7B the gain grows under INT4 on long terms (1.70×, DiD −0.037 [−0.040, −0.034]) and overall (2.88×), at 0.6B it shrinks overall (0.25×). Together with SmolLM2 (held-out 0.58× / 1.14×) there is no consistent direction: claim B remains "largely preserved on T5, partly lost on natural text (T4)".

## Addendum — Qwen3-1.7B-Base and Qwen3-0.6B-Base with LoRA (T5, seed 1)

Full tables: `experiments/e9-retrofit/report/t5-qwen3/report.md`. Recipe: LoRA r = 64 (host lr 2e-4), channel lr 1e-3, gate bias 0, channel scaled to the host's embedding-row norm, 50M tokens; same test set, holdout and items as SmolLM2.

**Dimension 1 (bf16), C5 − C0′ relative loss:** held-out terms **−16.9%** (1.7B) / **−20.3%** (0.6B); rare −10.8% / −8.7%; unseen −10.5% / −9.3%; 3+-subtoken −9.0% / −9.5%; ordinary domain text 0.0% / 0.0%. The free table C2 again gives nothing on held-out terms (C5 − C2 −16.9% / −21.0%). Larger than on SmolLM2 (−7.5% / −11.7% held-out), with LoRA instead of full fine-tuning.

**Dimension 3 (bf16), C5 − C0′:** new invented words — property selection **+16.1 points** (1.7B), entailment +12.3, paraphrase +7.7, statement accuracy +10.3 (all Holm-significant; SmolLM2-360M: +6.1 property); the generator's contamination-free zero-shot items — property **0.49 vs 0.32** (+17.4), entailment +16.0, paraphrase +5.7. Edits: seen terms move toward the edited fact (log-odds +1.12, +0.56 beyond the control edit; edit success 0.39 → 0.475), held-out terms barely (+0.05). At INT4 the new-word advantage shrinks but stays significant (property +6.2, entailment +9.2).

**Dimension 2 (INT4 RTN, channel FP16):** INT4 destroys much of every LoRA-adapted model's domain adaptation (loss +50% to +115% vs +5–12% for the original host; merged LoRA deltas are small relative to the quantization step, as the novelty check anticipated), but the adapted models stay far better than the original host. The channel's absolute advantage after held-out terms is **fully retained** (retained 1.00 at both sizes: 1.7B −0.142 → −0.143 nats; 0.6B −0.185 → −0.185), rare terms 0.78–0.98, long terms 0.97 (0.6B) to 1.75 (1.7B). So under INT4 the channel's held-out-term gain is preserved, not enlarged; the "gap grows" reading holds only for long terms at 1.7B and overall at 1.7B (3.1×), and reverses overall at 0.6B. Claim B remains a narrow measurement pending the PQ1 controls (channel off at INT4, GPTQ/AWQ, quantized embeddings).
