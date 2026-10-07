# R9 — Retrofit × quantization × ontology editing (E9)

**Status (2026-10-04):** T5 synthetic enterprise glossary, SmolLM2-360M and SmolLM2-135M, **3 seeds** (1–3; P0 has no training randomness). Qwen3 hosts: seeds 1–2 (section "Qwen3 on T5 at 2 seeds"). T4 chemistry (natural text) seed 1 is in (section "T4 chemistry" below: same direction, much smaller); T1-open seed 1 is in (null); the cross-track section shows the gain following the novelty of the vocabulary. The WordNet negative control is null, as predicted. Full generated report with every table and figure: `experiments/e9-retrofit/report/t5/report.md` (job `t5-report-s2-3`). Intervals are 95% cluster bootstraps over evaluation windows (loss) or items (dimension 3), with paired differences pooled over seeds; the across-seed spread is given where it matters. The seed-1 version of this report (commit `ccb9204` and earlier) is superseded; where 3 seeds change a seed-1 conclusion this is said explicitly.

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

## Findings and implications (3 seeds)

1. **Retrofitting works for genuinely new vocabulary, and it replicates.** A jointly trained VSA channel lowers a pretrained model's loss after new and held-out domain terms by 3–13% (held-out: −13.3% at 360M, −9.2% at 135M, SD across seeds ≤ 1.7 points) with no locality cost. The capacity-matched free table does not help held-out terms at all, so the gain comes from composition. It grows with model size and is larger on Qwen3 with LoRA (−17 to −20%, seed 1).
2. **Quantization: the channel's advantage survives INT4, but the gap does not grow on the target terms.** Held-out-term advantage retained 0.58× (360M) and 1.14× (135M), Qwen3 1.00× (seed 1). The consistent INT4 effect is extra robustness on ordinary domain text (overall advantage 1.26–1.32×), whose cause is untested. Paper wording: "the gain is preserved under 4-bit weight quantization", not "the gap grows". The narrow claim B stays pending the PQ1 controls.
3. **Zero-shot learning by ontology editing is real but modest, and replicates.** Adding a frame for an unseen word improves property selection by 4–7 points and paraphrase consistency by up to 8 points (12 points on the generator's own zero-shot items), attributable to the frame content and kept at INT4. Editing an existing fact moves seen terms reliably and held-out terms only at 135M.
4. **Caveats:** T5 text is generated from the ontology the channel reads, so these are upper bounds; T4 and T1-open (natural text) are the test of that, results due 2026-10-05. Absolute zero-shot accuracies are low (0.26 vs chance 0.20 for new-word properties at 360M).

## Next

T4 chemistry and T1-open (natural text with rare multi-token terms; running now), the WordNet negative control, Qwen3 seed 2, the PQ1 controls (channel off at INT4, GPTQ/AWQ/HQQ/NF4, quantized embeddings, the C5 arm variants, 3 seeds on T5 and T4), the Qwen3.5 block, and the dimension-3 knowledge-editing baselines.

## T5 controls (WP-PQ1; SmolLM2-360M, 3 seeds, 2026-10-07)

Full tables: `experiments/e9-retrofit/report/t5/report.md`, sections "WP-PQ1" (job `t5-report-s1-2-3-pq-SmolLM2-360M`). Same recipe, test set and parameters as C5; seeds 1–3 each; relative loss differences with 95% window bootstraps, Holm over arms within stratum.

**What the composed vector needs (operator and specificity ablation).** C5 − arm on held-out / rare / unseen / 3+-subtoken terms (negative = C5 better), and each arm − C0′ on held-out terms:

| Arm | C5 − arm | arm − C0′ (held-out) |
|---|---|---|
| C5rf: fixed random orthogonal binding | −5.2%* / −0.3%* / −0.3% / −1.1%* | −8.5% |
| C5ut: untyped (fillers only, no binding) | +0.2% / +0.1% / +0.4% / +0.1%* (equal) | −13.4% |
| C5tr: translation x + r | **+4.8%* / +2.9%* / +2.7%* / +2.1%*** (translation better) | **−17.2%** |
| C5sh: shuffled frames (another term's frame) | **−17.0%* / −8.2%* / −8.9%* / −7.6%*** | **+4.6%** (worse than no channel) |

- **The frame's content is necessary**: with another term's frame the channel is worse than none.
- **Binding is not**: dropping relation roles costs nothing, a random fixed binding costs 5% on held-out terms, and plain translation beats HRR on every term stratum. By rule 7 (outline), E9 is described as compositional parameter sharing over ontology frames; "HRR" names one operator in the ablation, the best one here is translation. The from-scratch operator screen (decision 54) decides the same question for training from scratch.

**Same-site baselines that also give an unseen term a vector without training on it** (frozen per-entry source vector through a trained projector, parameter-matched, same site and gate). C5 − arm on held-out / rare / unseen / 3+-subtoken, and arm − C0′ on held-out:

| Arm | C5 − arm | arm − C0′ (held-out) |
|---|---|---|
| C6m: subtoken mean (FVT / Hewitt-style new-token initialization) | −14.2%* / −6.9%* / −7.4%* / −7.5%* | +1.1% (no help) |
| C6d: definition encoder (the host encodes the verbalized frame) | −10.8%* / −2.8%* / −3.3%* / −3.8%* | −2.8% |
| C6g: TransE KG embedding of the entry | −15.3%* / −5.3%* / −5.9%* / −6.1%* | +2.4% (no help) |

The composed vector beats all three by wide margins (claim A's refutation reading "new-token initialization or a same-site frame-text vector matches the composed vector" is **not** triggered on T5). Encoding the same frame as text recovers only about a fifth of the gain.

**Copy concern: filler vs non-filler continuation tokens** (`e9_rescore`; a filler target is a token in the 8 tokens after a term that belongs to an alias of a filler of that term's frame; 19–23% of targets). C5 − C0′:

| After… | total | on filler tokens | on other tokens | share of the gain on fillers |
|---|---|---|---|---|
| held-out terms | −13.3% | −8.1%* | **−19.4%*** | 0.33 |
| rare terms | −6.0% | −8.7%* | −2.1%* | 0.86 |
| unseen terms | −6.6% | −9.0%* | −1.9% | 0.90 |
| 3+-subtoken terms (≈ all linked) | −7.6% | −12.5%* | −3.5%* | 0.75 |

- For rare and unseen terms **most of the gain (86–90%) is predicting the frame's filler words**: the copy effect the novelty check warned about. For held-out terms, whose rows come only from composition, two thirds of the gain is on other tokens (−19.4%), consistent with the composed row telling the model what kind of term follows (in T5's templated text, the type-specific continuation) rather than which filler word comes next.
- Filler shares on T4 (first 256 windows; execution.md, WP-PQ1): 3.7% after any term, 0.8% after held-out, 10.3% after rare and 28.9% after unseen terms, against 18.6–23.9% on T5. T4's largest gain is after unseen terms (−2.3 / −3.7%), its stratum with the most filler continuations, so filler copying plausibly carries much of T4's effect too; the T4 rescoring (queued after T4 seeds 2–3) measures it.

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

## T4 chemistry — natural text (SmolLM2-360M / 135M, seed 1, 2026-10-05)

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
| T1-open (MeSH, PubMed) | −8 / −5 / −6 / −8% | +0.0 / +0.0 / −0.0 / −0.1% | −8 / −4 / −6 / −8% | +0.0 / −0.2 / +0.0 / −0.1% |
| WordNet general (negative control, 50M tokens) | -0.6 / -0.6 / -0.8 / -0.8% | -0.00 / +0.04 / +0.01 / +0.01% | -0.3 / -0.3 / -0.3 / -0.4% | -0.00 / +0.02 / -0.06 / -0.00% |

- **Ordering:** the gain is largest where the vocabulary is newest to the host (T5), smaller on T4, and zero where the host already models the terms (T1, WordNet). Within T4 the unseen stratum is both the most novel (−56%) and the most improved (−2.3%).
- **Implication for the paper:** the channel is a tool for vocabulary that is genuinely new to the host (invented, private or fast-changing terms), not for terms already well represented in pretraining. That is the defensible scope of claim A, and it matches the host learning to ignore the channel on known vocabulary (engagement check).
- **Caveat:** three tracks and one synthetic track are not a dose–response curve. T5's sizes also contain filler copying (75–90% of the gain after rare, unseen and multi-subtoken terms) and templated, type-specific continuations (most of the held-out gain; section "T5 controls"). T4 seeds 2–3 sharpen this, and T7 (MeSH 2026 supplementary-concept names new to the host; queued, decision 55; prediction recorded before running: a gain at least T4's and below T5's) tests the ordering directly.

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
