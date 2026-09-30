# Implementation audit: `vsa_embed` and experiments 00–01c

**Date:** 2026-09-30. **Scope:** [`src/vsa_embed/`](../../src/vsa_embed/), [`tests/`](../../tests/), the runners in [`src/vsa_embed/experiments/`](../../src/vsa_embed/experiments/), and every recorded run under [`experiments/*/runs/`](../../experiments/). **Method:** code reading, small reproductions on CPU, and re-execution of every recorded `resolved_config.yaml` into a scratch directory. No repository file was modified; `git status` is clean apart from this folder.

**Baseline:** `PYTHONPATH=src python -m pytest -q` → 66 passed (torch 2.11.0, Python 3.12.4, transformers 4.54.0, nltk 3.8.1).

**Reproduction:** re-running the recorded configs reproduces the recorded metrics bit-for-bit for 00 smoke/realistic, 01b Stage A smoke/reduced, 01b stage-b-reduced, 01c.3, 01c.4, 01c.5 and the 01 GPT-2 pilot. Three runs do not reproduce with the current code: 01 synthetic smoke (nondeterminism, conclusion unchanged), 01b stage-b-smoke and 01c development (code and configs changed after the runs; see F4).

## Summary

The algebra is correct and no training loop touches held-out targets. The defects are in **evaluation design**: model selection on the test split, splits that leak or discard data, a rank loss that penalizes correct answers, controls that are matched in parameter count but not in capacity, and metrics that are inflated by construction. Two recorded conclusions change (F1, F5a); one recorded PASS flips to FAIL under current code (F4); the headline 01b retrieval gain survives but with a confidence interval whose lower bound is below the gate's own margin (F6).

| # | Finding | Affects recorded conclusions? | Affects future runs? |
|---|---|---|---|
| F1 | 01c.5 "low-rank is best" comes from test parents seen in training | **yes** (01c README) | yes |
| F2 | Candidates selected on the test split; "confirmation" reuses development test nodes | yes (all 01b/01c gates are optimistic) | yes |
| F3 | Node-disjoint split discards ≈50% of edges, trains on ≈25% | yes (edge- vs node-disjoint comparisons) | yes |
| F4 | Stale or post-edited artifacts; stage-b-smoke PASS → FAIL under current code | yes (01b smoke, 01c development) | provenance |
| F5 | Metrics inflated by construction (relation accuracy, per-relation candidate sets, kNN chance, ties) | **yes** (01b README relation accuracy) | yes |
| F6 | Statistics: 3 seeds on one graph; CI lower bound below gate margin; 00 selective-accuracy gate at point estimates | qualifies 01b and 00 | yes |
| F7 | InfoNCE rank loss treats shared targets as negatives | 01c.4/01c.5 training | yes |
| F8 | Diagonal control is not capacity-matched (basis cancels); low-rank not parameter-matched; HRR init asymmetric | conservative for recorded negatives | yes |
| F9 | Shuffled controls missing or weak (up to 70% of labels unchanged) | yes (control strength) | yes |
| F10 | Gates in code do not implement the design gates; `promotion_eligible` defaults inconsistent | yes (gate semantics) | yes |
| F11 | Split hygiene: symmetric `attribute` not grouped; token sharing across node-disjoint splits | small | yes |
| F12 | Library nits: low-rank not identity at init, `map`≡`diagonal`, global RNG side effect, unitary roles on bipolar input, anchor truncation, GPU/determinism | no | yes |

## Confirmed findings

### F1. 01c.5 low-rank result is a leakage artifact — changes a recorded conclusion

`split_source_neighborhoods` ([wordnet_relations.py:62-82](../../src/vsa_embed/wordnet_relations.py#L62-L82)) holds out **sources** atomically but lets **targets** appear in both partitions. In the recorded `source-neighborhood-development` run, 21–28 of the ≈71 test candidate parents per seed are training endpoints. Recomputed twice, independently, from the saved checkpoints for seed 19 (the seed-19 split replays exactly from `splits.csv`; every recorded per-method MRR reproduces to four decimals; the regenerated anchor tensor differs from the manifest hash only at bit level on CPU):

| Method | Recorded MRR | Queries with a parent seen in training (15) | Queries with no seen parent (22) | Params |
|---|---:|---:|---:|---:|
| rank-4 low-rank | 0.3326 | **0.5704** | **0.1705** | 12,288 |
| basis-offset residual HRR | 0.2738 | 0.3000 | 0.2559 | 3,842 |
| basis-offset diagonal control | 0.3198 | 0.3255 | 0.3158 | 3,842 |
| offset | 0.2770 | 0.2858 | 0.2711 | 1,536 |

Low-rank is the best method on seen parents and the worst on unseen ones; it also has 3.2× the parameters of HRR/diagonal and no shuffled control. Script: `scratchpad/repro_01c5.py` (session scratch directory; not committed).

*Consequence:* the 01c README sentence "rank-4 low-rank transfer is best … capacity/calibration trade-off" is not supported. The HRR < diagonal verdict holds in both strata.

*Fix:* make the split target-disjoint as well (or hold out whole parent neighbourhoods), report seen/unseen strata, add an equal-parameter low-rank and a shuffled low-rank.

### F2. Selection on the test split; the "untouched" confirmation is not untouched

- [experiments/wordnet_relations.py:208-211](../../src/vsa_embed/experiments/wordnet_relations.py#L208-L211): `best_structured = max(structured, key=scores.get)` over **test** MRR, then the gain and paired wins are computed on the same rows. The recorded stage-b-reduced "Retrieval gate: PASS (+0.0194)" is the test-selected family; hrr and diagonal would pass if named in advance, low_rank and offset would fail.
- [experiments/wordnet_relations.py:267-270](../../src/vsa_embed/experiments/wordnet_relations.py#L267-L270): the 01c rescue filters by test MRR then maximizes test cosine.
- `split_edges` has no validation partition; the 01c README's "validation-selected candidate" is not implemented.
- `edges.csv` is byte-identical across 01b stage-b-reduced, 01c development and 01c.3 (same `selection_seed`, `max_edges`). `basis-confirmation.yaml` (the only config with `promotion_eligible: true`, never run) re-splits the same graph: its seeds 71/89 would reuse 211/253 and 212/251 test nodes already used in development test sets; 521 of the 560 edges have been in some test partition.

*Fix:* name candidates in config before running; add a validation split for selection; hold out confirmation nodes at graph-collection time.

### F3. Node-disjoint split discards about half the data

[wordnet_relations.py:171-181](../../src/vsa_embed/wordnet_relations.py#L171-L181) holds out `√f` of the nodes so that the within-test edge fraction is ≈ `f`, but train shrinks to ≈ `(1−√f)²` and crossing edges are discarded. Recorded runs: basis-confirmation-development 436 train / 434 test / 810 discarded; taxonomy 175 / 181 / 364 (54–66 training edges per seed). Node-disjoint is the primary split in every 01b/01c config, so edge-disjoint vs node-disjoint comparisons also compare ≈75% vs ≈25% training data.

*Fix:* log realized fractions; when comparing split modes, subsample edge-disjoint training to the same size; consider holding out nodes with a lower fraction and reporting the discard rate.

### F4. Stale, non-regenerable or post-edited artifacts

- **stage-b-smoke:** per-relation `target_mrr` in `metrics.csv` was computed against 3–11 candidates vs 41–42 in current code (inflated by +0.22 on average, max +0.62). Re-summarizing with the current code flips the recorded PASS to FAIL because the matched `shuffled_low_rank` control is missing.
- **stage-b YAMLs** gained `min_paired_wins` and `min_cosine_gain` after the run (`summary.json`/`report.md` re-rendered 30 s after the YAML edit; `resolved_config.yaml` lacks the keys). Outcome unchanged because defaults coincide.
- **01c smoke:** `smoke.yaml` was edited after both smoke runs (lr 0.02 → 0.01, steps 100 → 150, bounds added); the recorded correction norm 1.204 exceeds the current hard bound 0.25, so 01c.0 cannot be regenerated.
- **01c development:** with current code the basis-HRR MRR falls from 0.2585 to 0.2145; the README's "basis model on the Pareto frontier" is a pre-fix artifact. The selector now picks `residual_hrr` instead of `offset_residual_hrr`; the gate still fails.
- **Provenance:** all runs (2026-07-17) predate the only git commit (2026-08-05); manifests carry no git SHA, transformers/nltk versions, device or command line; the 00 realistic run writes no `manifest.json`; `resolved_config.yaml` is the input config, so code defaults (`max_residual_scale`, `min_paired_wins`) are unrecorded; the 01 `report.md` files were hand-edited after generation and a rerun overwrites them.

*Fix:* write git SHA, package versions, device and command line into every manifest; resolve defaults into `resolved_config.yaml`; never edit a config in place after a run (copy to a new name); regenerate or mark stale runs.

### F5. Metrics inflated by construction

- **(a) Relation accuracy** ([real_relations.py:143](../../src/vsa_embed/real_relations.py#L143)): for a parameter-free model all candidate scores are identical and `argmax` returns relation 0, so additive's "accuracy" (0.147/0.173/0.156) is exactly the hypernym share of each test set. The 01b README's "HRR 0.216–0.275 … above additive" is empty; offset (0.275–0.345), diagonal and low-rank score at least as high as HRR. **Changes a recorded statement.**
- **(b) Per-relation multi-positive metrics** ([experiments/wordnet_relations.py:169-172](../../src/vsa_embed/experiments/wordnet_relations.py#L169-L172)) restrict candidates to that relation's own targets (20–29 vs 47–49), contradicting the adjacent comment; `distribution_mrr` has no candidate parameter. In 01c.4 there are 1.02 targets per query, so the "distribution" metric is ordinary MRR (+0.001–0.003), and the README's per-relation numbers are inflated (hypernym 0.2609 vs exact 0.2517 with exactly one target per query).
- **(c) kNN overlap** in the 01 pilot is computed within test rows and within degree bands ([factorization.py:137-143](../../src/vsa_embed/factorization.py#L137-L143)); chance is 0.042 overall but 0.087–0.096 in the high-degree band, so "additive 0.202 among high-degree concepts" is partly a chance-floor effect (lift: 0.097 overall, 0.110 high, 0.060 low). The pilot gate still requires a raw-cosine gain over `train_mean` although raw cosine is dominated by the mean direction (train_mean scores 0.529, the best).
- **(d) Ties are ranked optimistically** (`1 + #(score > positive)`, [real_relations.py:162](../../src/vsa_embed/real_relations.py#L162), 184, 211): an all-zero or NaN prediction gets MRR 1.0. Ties are real on static tracks (identical anchors from shared tokens, up to 7 per ≈100–136 candidates); NaN was not observed.

*Fix:* report relation accuracy only for models with parameters (or add a tie rule); pass the shared candidate set to every metric; report kNN lift over chance; rank ties at mid-rank and assert finite predictions.

### F6. Statistics

- 01b Stage B: `n = 3` seeds that are re-splits of one graph sample. HRR − additive gains 0.0162/0.0168/0.0251 give a 95% t-CI of [0.007, 0.032]; the lower bound is below the gate's own `min_mrr_gain` of 0.01. Sign test p = 0.125.
- 00 realistic selective-accuracy gate is decided at point estimates: 19/36 Wilson CIs contain 0.95. The threshold is chosen at maximum coverage, so calibration accuracy sits at 0.950–0.954 and only 10/28 conditions where the threshold binds pass on held-out seeds (vs 7/8 where it does not bind). The "heavy-tailed degree is the failure regime" attribution is confounded with the threshold rule (heavy tails do have worse top-1 and ECE). Aggregate means hide individual failures (e.g. map/256/heavy-tail: ECE 0.112, selective accuracy 0.829).
- 00 smoke: `target_indices = arange % bundle` ([capacity.py:88](../../src/vsa_embed/experiments/capacity.py#L88)) means only `bundle_size` distinct queries at noise 0.

*Fix:* report CIs and effect sizes in every summary; choose the abstention threshold with a margin or a lower confidence bound; use more graph samples, not only re-splits.

### F7. InfoNCE rank loss treats shared targets as negatives

[real_relations.py:108-118](../../src/vsa_embed/real_relations.py#L108-L118): positives are rows with the same (source node, relation); a different source with the same **target node** is a negative, so a perfect predictor keeps a loss of `log k` on rows whose target has multiplicity `k` (reproduced: 1.099 = log 3). 22–38% of training edges in the taxonomy and source-neighbourhood runs are affected (multiplicity up to 6); hub parents receive less gradient.

*Fix:* mark a candidate positive if its target node is in the query's target set, or score against unique target nodes.

### F8. Controls are not matched in capacity

- `basis_offset_diagonal_control` ([residual_relations.py:61-77](../../src/vsa_embed/residual_relations.py#L61-L77)): with diagonal binding, `normalize(normalize(r ⊙ (x ⊙ s)) ⊙ s⁻¹) = normalize(r ⊙ x)`, so the basis cancels and only reparameterizes the offset (changing the basis moves the output by 6e-8 vs 1.4e-2 for HRR). The control has the same parameter count as HRR but strictly less capacity. Because it still matched or beat HRR (01c.4 −0.007, 01c.5 −0.031), the recorded negative conclusions are conservative — but a future HRR win over it would not isolate binding.
- Low-rank: 86,016 parameters vs 5,376 (01b Stage B) and 12,288 vs 3,842 (01c.5), same steps and LR.
- Initialization: `HRRRelation` roles are free and start as a random circulant with `cos(T(x), x) ≈ 0` ([relations.py:53](../../src/vsa_embed/relations.py#L53)); every other family starts at the identity. The "hrr" baseline confounds the family with its initialization.

*Fix:* a non-commuting control (fixed permutation or orthogonal mix inside `S⁻¹(·)S`); an equal-parameter low-rank; an identity-initialized HRR variant (unit role with spectrum ≈ 1).

### F9. Shuffled controls are missing or weak

- Missing: `shuffled_offset` everywhere (offset is counted as "structured" at [:208](../../src/vsa_embed/experiments/wordnet_relations.py#L208)), `shuffled_hrr` and `shuffled_low_rank` in 01c development, `shuffled_low_rank` in 01c.5. The READMEs say every family has one.
- Weak: 01b's global per-edge shuffle leaves 12–17% of labels unchanged; 01c.4's per-edge shuffle 41–52%; 01c.5's per-query shuffle ([hierarchy_neighborhoods.py:157-163](../../src/vsa_embed/experiments/hierarchy_neighborhoods.py#L157-L163)) 62–70%, with only 1–5 of 22 instance-hypernym queries keeping a changed label. With two relation types a label permutation is a coin flip.

*Fix:* derangement-style shuffles (force every label to change where possible) and a shuffled control for every family that is compared.

### F10. Gates in code do not implement the design gates

- 01c rescue `promotion_eligible` defaults to **True** ([experiments/wordnet_relations.py:312](../../src/vsa_embed/experiments/wordnet_relations.py#L312)) but False in [:389](../../src/vsa_embed/experiments/wordnet_relations.py#L389) and [hierarchy_neighborhoods.py:90-91](../../src/vsa_embed/experiments/hierarchy_neighborhoods.py#L90-L91); `test_01c_summary_requires_reconstruction_and_retrieval` depends on the True default. The rescue summarizer never checks relation-residual R², the equal-parameter control, relation-family breadth, a second ontology, paraphrase stability or locality (design items 2, 3, 6–9). Running `basis-confirmation.yaml` could print "final gate: PASS" although the diagonal control beat HRR on cosine in 01c.3.
- 01b `gate_passed: true` in stage-b-reduced is MRR gain plus paired wins only ([:243](../../src/vsa_embed/experiments/wordnet_relations.py#L243)); none of the seven design-gate items is checked.
- 00: "Decision: do not promote" and "No algebra dominates" are hard-coded strings ([realistic_capacity.py:197,202](../../src/vsa_embed/experiments/realistic_capacity.py#L197)); the design gate is not evaluated.
- Development configs allow 2/3 wins with no CI; the design asks for a CI or every split.

*Fix:* default `promotion_eligible` to False everywhere; rename partial checks "proxy criteria"; compute decisions from data and print which design items are unimplemented.

### F11. Split hygiene

- `attribute` is symmetric in WordNet but only `antonym` is grouped ([wordnet_relations.py:100-106](../../src/vsa_embed/wordnet_relations.py#L100-L106), [:143](../../src/vsa_embed/wordnet_relations.py#L143)): 6 reverse pairs; in edge-disjoint splits 3–4 of 15–20 attribute test edges have their exact reverse in train (node-disjoint unaffected).
- Token sharing: `aligned_lemma` maps different synsets to the same single token (217 of 966 synsets share a token; `abridge.v.01 → "cut"`), so a node-disjoint split is not representation-disjoint on static tracks (6–18 test nodes share a static anchor with a train node); 25 edges have source and target on the same token (17 of 80 `substance_meronym`, e.g. `oak.n.02 → oak.n.01`), which the identity baseline solves trivially.

*Fix:* group `attribute` like `antonym`; token-disjoint splits for static tracks; drop or separately report same-token edges.

### F12. Library nits (future runs only)

- `LowRankRelation` initializes both factors randomly ([relations.py:99-101](../../src/vsa_embed/relations.py#L99-L101)); the docstring says "identity plus update" but `‖T(x) − x‖` is 0.48 at d = 32. Initialize `left` to zero.
- `map` and `diagonal` are the same class and initialization ([relations.py:61-86](../../src/vsa_embed/relations.py#L61-L86)); the synthetic `orthogonal` teacher is near-identity (cos 0.998, [global_local.py:96-102](../../src/vsa_embed/global_local.py#L96-L102)).
- `make_synthetic_relations` calls global `torch.manual_seed` ([global_local.py:92](../../src/vsa_embed/global_local.py#L92)); use `torch.random.fork_rng()`. `test_restricted_residual_is_transferable` implicitly depends on this.
- Unitary roles from bipolar input: exactly-zero spectral bins get zero phase ([algebra.py:81](../../src/vsa_embed/algebra.py#L81)); guard with `torch.where(mag > eps, spectrum / mag, 1)`.
- Contextual anchors: `truncation=True, max_length=64` cuts from the right, so for 1 node per run (`david.n.03`, `church_father.n.01`) the anchor is read inside the gloss; a left-padding tokenizer would corrupt every anchor (GPT-2 pads right, so recorded runs are unaffected). Truncate the gloss only, assert `input_ids[pos] == token_id`, force right padding.
- Reporting: `mean_log_basis_scale_abs` reports the raw pre-tanh parameter (recorded 0.27–0.38 vs cap 0.20); regularizers penalize raw offset/log-scale, not effective ones; the no-parameter path returns `1 − cos` rather than the objective; `threshold_step` is one step early; `precision_threshold` falls back to 1.0 and uses a CPU `arange`.
- Determinism: no deterministic-algorithms flags; CUDA→CPU fallback is silent; `relation_models.pt` saves `split_modes[0]` (edge-disjoint), not the primary node-disjoint models ([experiments/wordnet_relations.py:156](../../src/vsa_embed/experiments/wordnet_relations.py#L156)).
- Schema drift: `capacity.py` rejects design-only axes but silently ignores `constraints`, `primary_metric`, `artifacts`; other runners validate no keys; `hierarchy_neighborhoods.py:152` ignores per-method `rank`; the 01c `seed.yaml` (seeds 11/22/33, 44/55, cosine gate 0.02) does not match the executed configs.
- Entry points and docs: no console script for `hierarchy_neighborhoods`; 01c README has no commands; 01b README command omits `PYTHONPATH=src`; 01b README says 798 nodes where every artifact says 966; the 01c.0 → 01c.1 improvement is attributed to bounding although LR and steps also changed (pure HRR moved 0.3753 → 0.3879).

## Suspected, not confirmed

- The optimism from test-set selection in 01b is small (HRR 0.2713 vs diagonal 0.2652) but structural.
- `OrthogonalRelation` determinant is +1 at init and could flip to −1 through a singular matrix during training.
- GPU reruns may drift from the recorded CPU-reproduced numbers (index-add backward uses atomics).
- 01b low-rank underperformance may be overfitting on 157–173 edges rather than evidence about capacity.

## Checked and correct

FFT HRR bind/unbind for odd and even `n`; unitary roles from Gaussian input (unit norm, exact round trip, real DC/Nyquist); MAP straight-through sign; row-wise normalization with zero-vector safety; residual correction bound after decode; orthogonal QR sign correction; swapped argument order raises; no held-out targets in any loss, centering mean (train endpoints only), relation mean or rank-loss candidates; contextual anchors extracted in eval mode, fp32, no grad, final layer after the last norm, batched = unbatched to 3e-7; `split_source_neighborhoods` is source-atomic and complete; 00 calibration fit on seeds 101–103 and evaluated on 201–203 with ECE/Brier/coverage recomputed from `observations.csv` to 3e-8; degree sequences identical across algebras (paired); all methods within a seed share split, init seed and shuffle permutation; antonym groups kept together; current `split_edges` reproduces all recorded splits; 01c.4 and 01c.5 fix the candidate in config; HRR and diagonal parameter-matched (11,527 / 3,842); report counts (560/966, 1,728, 576, 385/301/149, 39,360) consistent with CSVs; gloss lexical overlap (target lemma in source gloss for 144/560 edges) does not inflate results (those edges have lower identity MRR).

## Test gaps

No test covers: unitary roles from bipolar input; `unitary_hrr`/`map` paths in `OntologyFactorizer`; degenerate or tied predictions in retrieval metrics; realized node-disjoint fractions; symmetric `attribute` grouping; token-disjointness; centering from train only; duplicate targets in the rank loss; low-rank identity initialization (would fail today); diagonal-control basis invariance (would expose F8); `contextual_anchors` (always monkeypatched), `aligned_lemma`, `collect_source_neighborhood_graph`; the per-relation candidate set of `distribution_mrr`; orthogonal determinant; orthogonal and map synthetic teachers; duplicate `train_indices` weighting; target-disjointness of neighbourhood splits (would expose F1).

## Recommended order of fixes

1. **Evaluation validity (F1, F2, F5, F7):** target-disjoint neighbourhood splits with seen/unseen strata; pre-named candidates plus a validation partition; shared candidate sets and tie handling in every metric; fix the rank-loss positives. Re-run 01b Stage B, 01c.3–01c.5 and update the READMEs (01c.5's low-rank claim, 01b's relation-accuracy claim).
2. **Control fairness (F8, F9):** non-commuting diagonal control, equal-parameter low-rank, identity-initialized HRR, derangement shuffles for every family.
3. **Gates and provenance (F4, F10):** `promotion_eligible=False` everywhere; manifests with git SHA/versions/device/command; resolved defaults; no in-place config edits; regenerate stale runs or mark them.
4. **Statistics (F6):** CIs in every summary; threshold selection with a margin.
5. **Splits and nits (F3, F11, F12).**

Items 1–3 are prerequisites for the experiments in [experiments.md](experiments.md), which reuse these runners and metrics.
