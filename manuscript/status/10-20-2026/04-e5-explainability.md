# 04 — E5: explainability and zero-shot evaluations

**Overview.** E5 was to test whether the method explains itself and works zero-shot on the models E4 trains from
scratch. It has five parts:
- **E5.1:** is attention over a term's edges faithful?
- **E5.2:** does that attention pick out word senses?
- **E5.3:** does an LLM judge rate the method's neighbours and explanations higher?
- **E5.4:** do structure-only vectors for unseen terms beat simpler zero-shot vectors?
- **E5.5:** do the vectors separate meaning from word frequency?

The pre-registration, code and item sets exist (2 Oct), but **no E5 run has been made**. Every part needs the 125M
three-seed checkpoints of the original E4 core (D4.1), which were never trained (see file 03, exploration 03.5). Some of
E5's tools live on elsewhere. E9 uses the E5.4 zero-shot evaluator on pretrained hosts, E12's decoding-faithfulness test
reuses the E5.1 code, and E9's frequency-bias analysis uses the E5.5 probe. Exploration 04.1 records the plan and why it
stalled; 04.2 records where the tools were reused.

**Terms used throughout.**
- **C5:** the full method, a gated vector composed from the term's ontology frame (relation–filler edges) and added at
  the term's last subtoken. **HRR** (holographic reduced representation) is its binding operator.
- **C0 / C0′:** the plain baseline trained from scratch / the pretrained host fine-tuned without the method.
- **C2:** a free per-term lookup table with C5's parameter budget. For unseen terms it falls back to the mean row.
- **C1:** fixed random span vectors. **C3:** static bundling (uniform weights over edges). **C6:** C5 with a growing
  dictionary. **C8:** the operator sweep.
- **D4.1:** the planned 125M × 500M-token, three-seed E4 grid with all conditions.
- **Edge attention:** the composer's weights over a term's edges, computed with the run's own context query.
- **Judge:** Claude (`claude-opus-5-5`), called headless with `claude -p`. Judge results are labelled "LLM-graded
  estimate".
- **Signs:** for loss, negative = better. For accuracy, positive = better.

---

# 04.1 — The pre-registered E5 on the from-scratch models (E5.1–E5.5)

**Status:** not run (pre-registered, items built, never queued) · **When:** pre-registration and code 2026-10-02 ·
**Sources:** `experiments/e5-explainability/preregistration.md`; `experiments/e5-explainability/items/*/manifest.json`;
`resources/plan-improvement/experiments.md` (E5); `manuscript/paper-readiness.md`; `manuscript/claims.md` (H-F.1,
H-F.2, H-C.2, H-E.1, J.1); `manuscript/draft.md` §5.9

## Research gap
HRRBERT reported that clinicians judged the neighbours of composed rare-code embeddings strongly related far more often
than baseline neighbours (28/40 against 4/40). For a causal model with attentive composition, nobody had checked three
things:
- whether the edge weights explain its predictions;
- whether they track word senses;
- whether a row built only from an unseen concept's relations beats rows built from its surface form or definition.

## Intuition
If the composition is truly structured, three things should hold:
- removing the most-attended edges hurts more than removing random ones;
- attention favours the edges of the right sense;
- a new concept's structure-only row supports property and entailment judgements.

## Methodology
Pre-registered before any number:

| Part (hypothesis) | Measure | Primary endpoint | Refuted if |
|---|---|---|---|
| E5.1 faithfulness (H-F) | remove or keep the top, bottom or random k ∈ {1, 2, 4} edges; Δloss on the 8 tokens after the span, KL, top-1 flips | top − random at k = 2 on the best 125M composition model, > 0 with Holm p < 0.05 in all 3 seeds | top ≈ random |
| E5.2 sense alignment (H-C) | SemCor words whose entry contains the gold synset (about 7,800 instances) | attention accuracy − WordNet-first-sense accuracy, interval > 0 in all seeds | attention ≤ first sense |
| E5.3 rating study (H-F.2) | neighbours of rare and held-out terms, rare MeSH terms on T1-open (PubMed with MeSH), edge explanations, growth (split) cards; 40 pairs or items per system; 3 calls × 2 paraphrases | one-tailed Fisher test, reference > other | no improvement |
| E5.4 zero-shot insertion (H-E) | property selection, entailment, paraphrase agreement, after-span loss | `own` minus the strongest structure-only baseline on `c3-heldout` (best 125M model), Holm p < 0.05 in all seeds | definition mean ties `own` |
| E5.5 frequency | ridge probe of log(1 + training count) from the rows | R² of C5 rows vs C2 rows (lower = more disentangled) | — |

- **E5.4 sources.** The candidate is `own`, the term's frame. Structure-only baselines: none, random, mean row (the C2
  fallback), surface mean, graph projection. Text-evidence sources (definition mean, definition encoder, à la carte,
  CoLLEGe-style) are reported apart.
- **Checkpoints:**
  - D4.1 at 125M: C0, C1, C2, best of C3–C7, and C8 random-fixed and untyped, each with 3 seeds;
  - the 50M screen, every condition with 3 seeds;
  - the continued-pretraining hosts.
  - S0 pilot runs are for dry runs only.
- **Reporting rule for E5.3:** a study is reportable only if judge calibration accuracy ≥ 0.9 and Fleiss κ ≥ 0.6.
- **Budget:** about 21 GPU-h (27 with a 30% margin), plus about 2,000 judge calls (USD 60–80).

## Experiment
Nothing ran:
- No E5 job is in the queue, and no `experiments/e5-explainability/runs/` folder exists.
- The readiness audit (3 Oct) lists E5 as "not started (pre-registration present; needs D4.1 checkpoints)".
- D4.1 was never queued: decision 54 (7 Oct) replaced the E4 core with reduced screens.

Built and committed:
- **Item sets:**
  - `c3-heldout-gpt2-v1`: 400 held-out concepts, 1,735 prompts (967 property, 768 entailment). It uses real aliases,
    so it is deliberately not contamination-free.
  - `c3-synthetic-gpt2-v1`: the same 400 concepts and prompts under invented names; contamination-free.
  - `c3-synthetic-smollm2-v1` (1,743 prompts) and `c3-synthetic-qwen2.5-v1` (1,741), built for the pretrained hosts.
  - `c6-devtools-v1`: 57 held-out symbols, 215 prompts.
- **Judge calibration (J.1):** 30/30 correct, κ 0.812 (`claude -p`) and 0.926 (subagents). This meets E5.3's bar.

## Results
None. Claims H-F.1, H-F.2, H-C.2 and H-E.1 remain "pending" in the claims ledger.

Evidence from other experiments bears on them:
- **Sense alignment (H-C):** on frozen GPT-2 anchors (E1), attention's sense accuracy was below the most-frequent-sense
  baseline (−0.24 [−0.29, −0.20]; gate failed). E5.2 would have tested the jointly trained model instead.
- **Zero-shot (H-E), from scratch:** in loss terms, held-out terms gain nothing at 500M tokens. At 125M with 3 seeds,
  C5 − C0 after held-out terms is +0.18% [+0.04, +0.32], not significant (file 03, 03.3). At 50M × 100M tokens it was
  −0.37%* (03.1).

## Limitations
- The checkpoints E5 names do not exist. The scaling screen's 125M models are the nearest substitute: C0 and C5 with 3
  seeds; C2, HRRAdd and HRRCat with 1. They lack C1, C3–C7 and C8, so E5.1's controls (static C3, C8 operators) cannot
  run as pre-registered.
- The plans disagree on cost. The pre-registration says about 21 GPU-h (27 with margin), the readiness audit about 30,
  and experiments.md about 40 for "E5 + quantization evals".

## Next steps
- **Blocked:** E5 waits on D4.1, which the author left unqueued (decision 54).
- **Proposed (our suggestion; in no plan):** an amendment that runs E5.1, E5.2, E5.4 and E5.5 on the scale-v1 125M C5
  seeds 1–3, with C0 and C2 as references. The pre-registration's per-run cost at 125M is about 7 min for E5.1, under
  1 min for E5.2 and about 8 min per E5.4 scenario. This would need the author's approval and a written amendment before
  any run.

---

# 04.2 — Where E5's tools were reused

**Status:** done elsewhere (the results belong to E9 and E12) · **When:** 2026-10-02 to 2026-10-12 · **Sources:**
`reports/R9-retrofit-quantization.md` (dimension 3); `manuscript/update-2026-10-12.md` (E12-2);
`resources/plan-improvement/execution.md` (decision 62); `src/vsa_embed/experiments/{e9_plan, e9_freqbias}.py`

## Research gap
With the from-scratch checkpoints missing, E5's questions moved to the pretrained-host programme (E9) and the
explicit-recall programme (E12). The reader should know which E5 tools those results rest on.

## Intuition
The tools do not depend on the training regime. The zero-shot items, the faithfulness ablation and the frequency probe
work on any model that carries the method.

## Methodology
- **E9 zero-shot items (E5.4 format):** contamination-free invented terms from the T5 generator (a synthetic company
  glossary). `e5_zeroshot` runs as an E9 evaluation job on SmolLM2 hosts. The comparison is C5 vs C0′, 3 seeds, paired
  over items.
- **E12 decoding faithfulness** (decision 62): ablate or swap one decoded edge in the store and check that the matching
  prediction moves, and only that one. The decision says to reuse the E5.1 code where possible.
- **E9 frequency bias:** `e9_freqbias` reuses E5.5's ridge map to predict training frequency from each representation.

## Experiment
These ran inside E9 (T5, SmolLM2-135M and SmolLM2-360M, 3 seeds) and E12 (T5, SmolLM2-360M, 3 seeds). See those
programmes' reports for sizes and seeds.

## Results
- **E9, new words added to the ontology after training** (300 invented terms; C5 − C0′, paired over items, seed means;
  positive = better):
  - property selection +0.070 [+0.057, +0.083]* at 360M (0.189 → 0.259) and +0.042 [+0.028, +0.055]* at 135M;
  - inside C5, the term's own frame beats a random frame of equal degree, no frame and the mean row. At 135M, property
    is 0.232–0.242 with the own frame against 0.190–0.196 for the others, in every seed. The gain comes from the
    frame's content.
- **E9, track zero-shot items** (E5.4 format; C5 − C0′, seed means): property +0.125 at 360M (0.346 → 0.471, per-seed
  SD 0.005) and +0.123 at 135M; entailment +0.096 / +0.102; paraphrase +0.020 / +0.061.
- **E12, decoding faithfulness:** removing one decoded relation moves the answer as expected (faithfulness +0.36*).
  Binding makes this slightly weaker than a no-binding store (−0.044*), against a prediction of about 0.

These are E5.4- and E5.1-style results on pretrained hosts and synthetic terms. They are not E5's pre-registered
from-scratch endpoints.

## Limitations
- They do not test E5.4's structure-only claim on real held-out WordNet concepts in from-scratch models.
- They use the pretrained-host baseline C0′, not E5.4's full list of text-evidence baselines.
- We report them only as pointers. The E9 and E12 status files are the authority for their current values.

## Next steps
- **Done** in E9 and E12, and ongoing there.
- **Not planned:** a from-scratch E5 (see 04.1).

---

# Sources

- Pre-registration and items: `experiments/e5-explainability/preregistration.md`;
  `experiments/e5-explainability/items/{c3-heldout-gpt2-v1, c3-synthetic-gpt2-v1, c3-synthetic-smollm2-v1,
  c3-synthetic-qwen2.5-v1, c6-devtools-v1}/manifest.json`.
- Plan and decisions: `resources/plan-improvement/experiments.md` (E5, compute table); `resources/plan-improvement/execution.md`
  (decisions 54, 62); `resources/plan-improvement/gates.md` (D1, E1 gate).
- Ledger, draft and audit: `manuscript/claims.md` (H-C.2, H-E.1, H-F.1, H-F.2, J.1); `manuscript/draft.md` §5.9;
  `manuscript/paper-readiness.md`; `manuscript/reproducibility.md`.
- Reuse: `reports/R9-retrofit-quantization.md` (dimension 3); `manuscript/update-2026-10-12.md` (E12-2);
  `src/vsa_embed/experiments/e9_plan.py`, `e9_freqbias.py`.
- From-scratch held-out numbers: `experiments/e4-small-lm/analysis/{opscreen-v1, scale-v1-seeds}/`.
- Dates: `git log -- experiments/e5-explainability`; `.jobs/` (read only).
