# E5 — explainability and zero-shot: pre-registration and run plan (WP-E5)

Fixed before any E5 number is computed on a gate checkpoint. Code: `src/vsa_embed/experiments/e5_*.py`
(one module per sub-experiment, R5 report in `e5_report.py`). Every run writes a run folder
(`resolved_config.yaml`, `manifest.json` with git SHA and source run, `summary.json`, `report.md`,
per-example outputs). Outputs go to `experiments/e5-explainability/runs/<sub-experiment>/<source run>/`.
LLM-graded results are labelled **LLM-graded estimate**; text-evidence zero-shot generators are
reported apart from the structure-only claim (proposal §7).

## Checkpoints

From-scratch (GPT-2 BPE, C3 ontology `wordnet-gpt2-v1`): the D4.1 125M runs (C0, C1, C2, best of
C3–C7, C8 `random_fixed` and `untyped`; 3 seeds) and the D4.9 50M screen (every condition, 3 seeds).
Continued pretraining (D4.4): SmolLM2-135M/360M and Qwen2.5-0.5B × {C0', C2, best} × 3. T2 (D8.2)
runs for the C6 scenario. S0 pilot runs may be used for dry runs only; they are never reported as E5
results.

## E5.1 Faithfulness (H-F) — `e5_faithfulness`

- Spans: every linked span (ℓ ≥ ℓ_min) in the run's evaluation windows (the trainer's `eval.windows`
  windows; 1,024 at D4.1). Edge weights = the composer's weights with the run's own context query.
- Policies per `k ∈ {1, 2, 4}`: remove / keep the top-, bottom- and random-`k` edges (2 random
  draws, averaged); only occurrences with more than `k` edges; a removed edge is erased from the
  weighted sum (equal to renormalizing the softmax over the rest). Ties broken at random.
- Measure: Δloss on the 8 tokens after each ablated span; KL(full ‖ ablated) and top-1 flip rate on
  the same tokens. Cluster bootstrap over windows (2,000 resamples), Holm over the run's contrasts.
- **Primary endpoint:** comprehensiveness top − random at `k = 2` (stratum `k2`) on the 125M best
  composition condition, > 0 in all 3 seeds with Holm-adjusted p < 0.05 in each. *H-F refuted* if
  top ≈ random there. Controls: the static C3 composer (uniform weights: top ≈ random by
  construction) and C8 operators. Secondary: sufficiency (keep top-`k` loses less than keep random),
  held-out vs seen spans, probe-level ablation (`--probes wic,wsd --probe-k 2`, best condition only).

## E5.2 Sense alignment (H-C) — `e5_senses`

- Items: SemCor tagged words whose injected entry is a polysemous union entry containing the gold
  synset (≈ 7,800 instances with GPT-2 BPE at ℓ_min = 2). Per-concept frames rebuilt from WordNet and
  verified to reproduce the entry schedule exactly.
- Prediction: sense with the largest attention mass (shared edges split equally; ties → WordNet-
  earlier sense, the E1 rule). Baselines: WordNet first sense; SemCor MFS cross-fitted over document
  halves. Also the gold-mass lift over uniform weights.
- **Primary endpoint:** attention accuracy − WordNet-first-sense accuracy (paired bootstrap over
  items) for the best attentive condition (C5/C6), CI excluding zero in all seeds; gold-mass lift > 0
  as the tie-robust secondary. *H-C (alignment part) refuted* if attention does not beat WN1.

## E5.3 Rating study (LLM-graded estimate) — `e5_rating`

Rubrics are fixed by the prompt and schema text in `e5_rating.STUDIES` (SHA-256 of prompts + schema):
`neighbour_pair` `3ca1d37ddc4d389d…`, `neighbour_preference` `6418f54f0cbe6655…`, `edge_explanation`
`d750dfee737384dc…` and `split_card` `b73c21402c299d1c…` (the last two are the C5-calibrated rubrics of
`judge_protocol`). Each item: 3 calls over 2 paraphrases (preference paraphrases show the lists in
opposite positions), model `claude-opus-5-5` via headless `claude -p` (harness recorded), items
shuffled with seed 0, systems never shown. Calibration items with known answers (WordNet relations
vs random) are mixed in blind. Reported: per-system counts, one-tailed Fisher test (reference >
other, α = 0.05), Fleiss κ across calls, calibration accuracy. A study is reportable only if
calibration accuracy ≥ 0.9 and κ ≥ 0.6 (as C5); otherwise it is reported as inconclusive.

| Study | n (pre-registered) | Selection rule | Systems | Endpoint |
|---|---|---|---|---|
| neighbours (rare) | 10 concepts × top-4 = **40 pairs per system** | non-held-out single-concept entries with training frequency 1–9 and a linkable alias, frequency-weighted sampling without replacement (seed 0); neighbours in the `state` space among all seen linkable entries | best channel (reference), C0, C2 (125M seed 1) | "strongly related" pairs, Fisher one-tailed (HRRBERT: 28/40 vs 4/40) |
| neighbours (held-out) | 40 pairs per system | held-out single-concept entries, uniform (seed 0) | as above | as above |
| T1-open rare MeSH | 40 pairs per system | built by the T1 WP into `experiments/t1-open-clinical/items/` in this format | best, C0, C2 on T1 | as above |
| edge explanations | **40 occurrences** × 3 explanations | first eval occurrence per entry; polysemous union entries with ≥ 4 edges, seeded uniform, topped up with other entries with ≥ 4 edges | top-3 edges by context weight (reference), 3 random frame edges, 3 nearest neighbours | score 2 ("right") counts, Fisher top > random, top > neighbours |
| M3 cards | **40 splits** (all if fewer) × 2 | split cards of the C6 run whose two children each have ≥ 2 using entries, seeded uniform | real split (reference), random partition of the same usages | "distinct" counts, Fisher split > random |

## E5.4 Zero-shot insertion (H-E) — `e5_zeroshot`

- Items (committed, `items/`): `c3-heldout-gpt2-v1` (400 held-out single-concept noun/verb entries,
  1,735 prompts: 967 property, 768 entailment; real aliases — from-scratch models only),
  `c3-synthetic-gpt2-v1` (the same 400 concepts and prompts under invented names, checked against
  WordNet lemmas, alias words, the GPT-2 vocabulary, the evaluation corpus and the first 20M training
  tokens — contamination-free on every host), `c6-devtools-v1` (57 held-out C6 symbols, 215
  prompts; only the CamelCase class symbols link until open decision 17 is resolved). Pretrained
  hosts use `c3-synthetic-smollm2-v1` (1,743 prompts) and `c3-synthetic-qwen2.5-v1` (1,741), built
  with each host's tokenizer and ontology (the ℓ_min check changes which held-out concepts qualify,
  so the 400 concepts differ slightly from the GPT-2 set); their contamination checks used each
  host's evaluation corpus and first 20M training tokens.
- Sources: structure-only `own` (candidate), `none`, `random`, `mean_row` (C2 fallback in-model),
  `surface_mean`, `graph_projection`; text evidence `definition_mean`, `definition_encoder`,
  `alacarte`, `college` (4 example contexts from odd-numbered eval documents; corpus tests use
  even-numbered documents only). Maps fitted on 4,000 seen entries (frequency ≥ 10), seed 0.
- Tests: property selection and entailment (PMI against the null surface "thing", 2 paraphrases,
  accuracy averaged over paraphrases), paraphrase agreement, semantic-head rank (runs with
  `semantic_weight > 0`), after-span loss.
- **Primary endpoint:** property accuracy of `own` minus the strongest structure-only baseline on
  `c3-heldout` (from scratch, 125M best) with Holm-adjusted p < 0.05 in all seeds, and `own` (best)
  minus `own` (C2 run, the free-table fallback) item-level across seeds. On pretrained hosts the
  same endpoint on `c3-synthetic`. *H-E refuted* if `definition_mean` ties `own`
  (proposal H-E); text-evidence sources are never part of the structure-only claim.

## E5.5 Frequency disentanglement — `e5_frequency`

Ridge probe (5-fold out-of-fold, GCV strength) of log(1 + training count) from token rows, the mean
alias embedding (`concept_surface`), channel rows (raw, unit) and the freshly initialized composer
(`concept_rows_init`, the structure-only reference). **Endpoint:** R² of channel rows, C5/best vs
C2 free rows (and vs the C0 `concept_surface`), seed-level intervals; lower = more disentangled.
CPU only.

## Compute and judge budget (planning figures)

Extrapolated from the C4 throughput (50M ≈ 70k, 125M ≈ 36k training tokens/s; forward-only ≈ 3×) —
the GPU was busy while WP-E5 was built, so these are not yet measured on a gate checkpoint.

| Sub-experiment | per run, 50M | per run, 125M | runs | GPU-h |
|---|---:|---:|---|---:|
| E5.1 faithfulness (1,024 windows × 26 policies + KL) | ≈ 4 min | ≈ 7 min | 50M composition conditions × 3 (≈ 36), 125M best + C8 × 3 (9) | ≈ 3.5 |
| E5.1 probe ablation (fast suite × 4 policies) | — | ≈ 15 min | 125M best × 3 | ≈ 0.8 |
| E5.2 sense alignment (embeddings + composer only) | < 1 min | < 1 min | every composition run (≈ 45) | ≈ 0.8 |
| E5.4 zero-shot, per scenario (10 sources × ≈ 17k prompts + corpus tests) | ≈ 5 min | ≈ 8 min | 125M 18 runs × 2, 50M 24 runs × 2, CPT 27 runs × 1 (≈ 12 min), T2 21 runs × C6 (≈ 3 min) | ≈ 15 |
| E5.5 frequency | CPU | CPU | every run | 0 |
| E5.3 item builders (states of ≈ 89k aliases × 3 systems) | — | ≈ 3 min | 4 item sets | ≈ 0.5 |
| **Total** (+ 30% margin ≈ 27) | | | | **≈ 21** |

Judge: ≈ 1,550 calls for the WordNet-track studies (neighbours rare ≈ 450 and held-out ≈ 450 incl.
preferences and calibration, edges ≈ 390, cards ≈ 265) plus ≈ 450 for the T1-open neighbour set.
The rubric validation (10 `claude -p` calls) cost $0.027–0.061 per call (mean $0.039), so ≈ $60–80
in total (≈ $40 at the planned $0.02). No local GPU time.

## Commands (per source run; `$PY = ~/anaconda3/envs/vsa-repro/bin/python`, `PYTHONPATH=src`)

```bash
R=experiments/e4-small-lm/runs/<stage>/<size>-<cond>-s<seed>; O=experiments/e5-explainability/runs
$PY -m vsa_embed.experiments.e5_faithfulness --run $R --output $O/faithfulness/<run>          # + --probes wic,wsd for best
$PY -m vsa_embed.experiments.e5_senses       --run $R --output $O/senses/<run>
$PY -m vsa_embed.experiments.e5_frequency    --run $R --output $O/frequency/<run> --token-counts ~/data/vsa-llm/cache/gpt2-train-counts.npy
$PY -m vsa_embed.experiments.e5_zeroshot evaluate --run $R --items experiments/e5-explainability/items/c3-heldout-gpt2-v1 --output $O/zeroshot-c3h/<run>
$PY -m vsa_embed.experiments.e5_zeroshot evaluate --run $R --items experiments/e5-explainability/items/c3-synthetic-gpt2-v1 --output $O/zeroshot-c3s/<run>
$PY -m vsa_embed.experiments.e5_rating build neighbours --runs best=$BEST C0=$C0 C2=$C2 --output $O/rating/items-neighbours-rare
$PY -m vsa_embed.experiments.e5_rating build edges --run $BEST --output $O/rating/items-edges
$PY -m vsa_embed.experiments.e5_rating build cards --run $C6 --output $O/rating/items-cards
$PY -m vsa_embed.experiments.e5_rating grade --items $O/rating/items-neighbours-rare --output $O/rating/graded-neighbours-rare --cache $O/rating/judge-cache --max-new-calls 400
$PY -m vsa_embed.experiments.e5_report --inputs $O --output reports/R5-explainability
```

Host items: `e5_zeroshot items --scenario c3_synthetic --ontology ~/data/vsa-llm/c3/wordnet-smollm2-v1/ontology.pt
--tokenizer HuggingFaceTB/SmolLM2-135M --output experiments/e5-explainability/items/c3-synthetic-smollm2-v1`
(Qwen2.5 likewise).
