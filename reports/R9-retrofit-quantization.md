# R9 — Retrofit × quantization × ontology editing (E9)

**Status:** first track complete — T5 synthetic enterprise glossary, SmolLM2-360M and SmolLM2-135M, **seed 1** (seeds 2–3, T4 chemistry, T1-open, WordNet control and Qwen3 hosts queued or planned). Full generated report with every table and figure: `experiments/e9-retrofit/report/t5/report.md`. Single seed: all intervals cover evaluation windows or items only, not seed variance.

**Question (author request 2026-10-02).** (1) Does a VSA ontology channel trained jointly with a pretrained model improve it on long-token, rare and out-of-distribution words? (2) Is the gap larger after weight quantization? (3) After training, can the model learn new or changed words zero-shot by editing the ontology alone?

**Design.** P0 = original host (no training); C0′ = continued training on the same 50M tokens without the channel (full fine-tuning, host lr 3e-5); C2 = the same + a capacity-matched free per-concept table; C5 = the same + the attentive VSA channel (WordNet-style frames of the glossary ontology, HRR). T5 is generated internal-style documents over invented private terms (contamination-free: the host cannot have seen them); 360 held-out terms are never linked in training and are composed zero-shot at evaluation. Quantization: torchao weight-only INT8 / INT4, output head in full precision; variant A keeps the channel in FP16, B quantizes it too (A ≈ B everywhere).

Why T5 first: on general WordNet vocabulary the pretrained host already models the words well, and an engagement check showed the channel was neutralized (C5 = C0′ to four decimals). The question only makes sense for vocabulary that is genuinely new to the host.

## Dimension 1 — rare and out-of-distribution words (bf16)

Relative loss difference of C5, 95% CI, Holm within stratum (negative = better):

| Stratum | targets | 360M: C5 − C0′ | 360M: C5 − C2 | 135M: C5 − C0′ | 135M: C5 − C2 |
|---|---:|---|---|---|---|
| after held-out terms (never linked; zero-shot composed) | 50,123 | **−11.7% [−12.5, −10.9]** | **−13.8%** | **−7.5% [−8.3, −6.8]** | **−12.1%** |
| after unseen terms | 4,907 | −5.5% [−7.2, −3.9] | −7.4% | −3.4% | −7.8% |
| after rare terms (train freq 1–9) | 36,303 | −5.6% [−6.2, −5.0] | −7.1% | −3.1% | −7.8% |
| after 3+-subtoken terms | 482,522 | −7.1% [−7.3, −6.9] | −7.1% | −6.4% | −7.4% |
| inside terms | 286,528 | +0.01% (n.s.) | −0.14% | +0.22% | −0.04% |
| unlinked text in the domain (locality) | 380,816 | −0.04% (n.s.) | −0.10% | −0.06% (n.s.) | −0.11% |
| general text (FineWeb-Edu `eval-general`) | — | −0.0004 nats (n.s.-sized) | | | |

- The VSA channel lowers loss after new terms by 3–12%, with no cost on other text. The largest gain is on **held-out terms**, whose channel rows come only from composing the ontology; the free table cannot help them (C2 is slightly *worse* than C0′ there, +2–5%).
- The gain grows with model size on the held-out and rare strata (360M > 135M).
- Probes on general lexical benchmarks (WiC, WSD, CARD-660, Rare Words, BLESS, HyperLex, LAMBADA) do not change (|Δ| ≤ 0.015), as expected: the channel was trained on glossary vocabulary only.

## Dimension 2 — the gap under quantization

INT8 is nearly free for every model (≤ 0.6% loss). INT4 costs the fine-tuned models far more than the original host on this domain (C0′ +38% all tokens vs P0 +7.7% on general text; the fine-tuned models sit at a very low loss of ≈ 0.6 nats, where the same weight noise costs proportionally more).

C5's advantage (gain = C5 − C0′, nats/token) at bf16 vs INT4 (channel FP16; B is the same):

| Stratum | 360M bf16 → INT4 | retained | 135M bf16 → INT4 | retained |
|---|---|---:|---|---:|
| after held-out terms | −0.101 → −0.048 | **0.47×** | −0.067 → −0.107 | **1.59×** |
| after 3+-subtoken terms | −0.043 → −0.038 | 0.89× | −0.042 → −0.058 | 1.39× |
| after rare terms | −0.047 → −0.039 | 0.84× | −0.028 → −0.031 | 1.09× |
| unlinked text in the domain | −0.000 → −0.028 | — | −0.000 → −0.048 | — |
| all tokens | −0.020 → −0.030 | **1.51×** | −0.019 → −0.052 | **2.80×** |

Against the free table C2 the pattern is the same and stronger at 135M (held-out 1.69×, rare 1.53×).

- **At 135M the gap clearly grows under INT4** on every stratum — the author's hypothesis holds: quantization hurts the plain model more on these terms and the structured channel recovers more of it, and the channel-equipped model is also more robust on the domain's ordinary text.
- **At 360M the picture is mixed:** the overall advantage grows (1.5×, driven by robustness on ordinary domain text), but the advantage on held-out terms halves under INT4.
- General-text quantization damage is identical for all four models (≈ +7.7% at INT4): the channel neither protects nor hurts general text.

## Dimension 3 — zero-shot learning by editing the ontology (no weight update)

**New words** (300 invented terms whose frames are new combinations of existing atomics and relations, added to the ontology after training). Accuracy (chance 0.20 for property selection with 5 options, 0.50 for entailment); C5 own − C0′ with Holm-adjusted CIs:

| Test | 360M C0′ | 360M C5 | Δ | 135M Δ |
|---|---:|---:|---|---|
| property selection | 0.188 | 0.249 | **+0.061 [+0.047, +0.076]** | +0.045 |
| entailment | 0.512 | 0.537 | +0.025 (n.s.) | +0.037 (n.s.) |
| paraphrase consistency | 0.319 | 0.387 | **+0.068** | — |
| statement accuracy | 0.231 | 0.256 | **+0.026** | — |

Controls inside C5 show the gain comes from the frame content: own frame 0.249 > random frame of equal degree 0.211 > no frame 0.195 ≈ mean row 0.196. The effect survives INT4 (property 0.269 vs 0.214).

**Track zero-shot items** (contamination-free synthetic terms from the T5 generator, E5.4 format): 360M C5 vs C0′: property **0.46 vs 0.35 (+0.112)**, entailment **0.67 vs 0.57 (+0.098)**, paraphrase +0.045 — all significant; same against C2.

**Edits** (change one relation–filler of an existing term; ROME/MEMIT-style metrics): only C5 is editable. Edits move the model toward the new fact for **seen** terms (log-odds change +0.69 [+0.34, +1.10]; +0.27 beyond a control edit) but not for held-out terms (+0.04, n.s.); edit success moves 0.40 → 0.455 on seen terms, specificity is preserved (NS 0.55). Editing works weakly and only where the model has learned to read the channel for that term.

## Findings and implications

1. **Retrofitting works for genuinely new vocabulary.** A jointly trained VSA channel gives a pretrained model a 6–12% lower loss after new and held-out domain terms with no locality cost; the free-table control does not, so the gain is from composition, not capacity. (On vocabulary the host already knows, it does nothing — R1/E9 check.)
2. **Quantization: the smaller model gains more from the channel after INT4.** At 135M the channel's advantage grows 1.1–1.6× on rare/held-out/long terms and 2.8× overall under INT4; at 360M only the overall advantage grows, while the held-out-term advantage halves. Size matters, and seeds 2–3 are needed before claiming either direction.
3. **Zero-shot learning by ontology editing is real but modest.** Adding a frame for an unseen word improves property selection and paraphrase consistency by 6–7 points (11 points on the generator's own zero-shot items) — attributable to the frame's content. Editing existing facts works only for terms seen in training.
4. **Caveats:** one seed; T5 is templated synthetic text, so effect sizes are likely larger than on natural text (T4 chemistry and T1 PubMed test that); absolute zero-shot accuracies are low (0.25 vs chance 0.20 for new-word properties).

## Next

Seeds 2–3 on T5; T4 chemistry and T1-open (natural text with rare multi-token terms); the WordNet negative control (quantization gap on known vocabulary); Qwen3-1.7B / 0.6B with LoRA (queued after the memory probe); Qwen3-4B and Qwen3.5 on the author's decision.
