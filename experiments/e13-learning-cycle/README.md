# E13 — the learning cycle (decision 63, WP TK-E13)

One model goes once around the read–learn–write cycle of
[`manuscript/toolkit-methodology-2026-10.md`](../../manuscript/toolkit-methodology-2026-10.md) §4: passive round 1 →
*learn* missing structure → *write* never-seen words by reading their definitions → *read* them for reasoning →
passive round 2, measured as tokens to criterion; then the same on a frozen 4-bit host against QLoRA.
Pre-registration (binding): [`preregistration.md`](preregistration.md). Nothing here is queued; the commands are in
[`queue-commands.sh`](queue-commands.sh).

## Files

| path | what |
|---|---|
| `t5.yaml` | the T5 configuration (round split, seed ontology, texts, hosts / seeds / arms, learn / write / round-2 settings, statistics, queue levels) |
| `t7-rood.yaml` | T7-ROOD on TK-H1's rounds build (`experiments/t7-new-vocabulary/ROOD.md` §5); pre-registration amendment 1 |
| `preregistration.md` | stages, arms, endpoints L1–L5 with kill criteria, statistics, deviations policy; §12 amendment 1 (T7-ROOD), amendment 2 (decision 64: L1's wrong-frame co-primary, definition comparators, L5's token cost) |
| `queue-commands.sh` | `jobqueue add` lines with GPU-h: T5 at 54.4985–54.4989 (queued 2026-10-08), T7-ROOD at 54.4996 / 54.49965 / 54.4997 / 54.49975 / 54.49979 (stage 0 + learn + write, stage 4, stage 5, reason, report) |
| `queue-commands-decision64.sh` | amendment 2's 10 new jobs (not queued; ≈ 3.6 GPU-h): Qwen3 `random` at 54.49865 / 54.49967, `context` at 54.49885 / 54.49977; cancels nothing (the queued command lines are unchanged) |
| `configs/t5/`, `configs/t7-rood/` | the trainer configs `plan` writes (stage 0 and every round-2 arm) |
| `items/understanding-t7rounds-{smollm2,qwen3}-v1` | relation / reverse / paraphrase items of T7-ROOD's 2,470 round-2 anchors (`e13_cycle items`; E9's understanding builders on the rounds ontology) |
| `smoke/`, `smoke-t7-rood/` | the CPU smokes (SMOKE; SmolLM2-135M, a few steps) — pipeline checks, not results |
| `smoke-amendment2/{t5,t7-rood}/` | the CPU smokes after amendment 2 (SMOKE): + the `random` arm, the `context` stage, stage-3 token accounting, the six-test Holm family |
| `runs/<track>/` | (created by the jobs) `stage0/`, `cycle/<host>-s<seed>/{learn,write,reason,context}/`, `round2/<host>-<arm>[-<scheme>]-s<seed>/` |
| `report/<track>/` | (created by the report job) `summary.json`, `report.md` |

Large data go to `~/data/vsa-llm/e13/<track>/<family>/`: `seed/ontology.pt` and `seed/erased.json` (the stage-1 gold),
`round2/train` and `round2-defs/train` (round-2 documents + FineWeb-Edu, 50/50), `learn-validation` (fresh round-1
text for the acceptance test), `reference-<windows>x<length>.npz` (the `ref_round2` / `ref_round1` strata of the
evaluation windows), `round2/<run>/` (each arm's ontology and entry rows); the generated texts under `text/`.

## Code

- `vsa_embed.concept_store` — methodology M2: `ConceptStore` with `read` (recall: frame, role, chain, reverse),
  `write` / `written` / `commit`, `propose` (proposers: `rule_closure` default, `reader`, any `"module:function"`) and
  `accept` (tests: `HeldOutUtilityTest` = LM held-out utility with Holm, `composer_self_test` = `self_test.accept_edges`);
  `null_proposals` / `false_acceptance_rate` (M3's null world); `call` exposes each verb as a tool.
- `vsa_embed.experiments.e13_cycle` — `prepare` (round split, corpora, reference strata), `learn`, `write` (frames,
  `fvt` rows, and the round-2 relation / property items scored by the shared TK-B1 harness `benchmarks.ranking` under
  `none` / `store:linker` / `store:oracle` / `store:random` / `definition-in-context`), `reason` (E12's recall-tool
  scorer: the harness has no recall condition and no PMI; amendment 2: the prompt tokens each condition adds, and on T7-ROOD
  `definition` = the read set's SCR note), `round2` (materialize an arm, train, locality), `context` (amendment 2: the
  noread arm's step-0 model without / with the round-2 definitions in context — L1's `read − definition`), `report` (L1's
  co-primaries read − noread and read − random, Holm over six; the definition comparators with token costs; L5's cost
  criterion), `plan`, `smoke`.
- Opt-in trainer keys (`vsa_embed.training.lm`; absent keys change nothing): `train.init_mode: continue` (+
  `init_merge_lora`), `model.host_quantization`, `channel.entry_rows`, `eval.points`.

## Commands

```bash
PY=~/anaconda3/envs/vsa-repro/bin/python
PYTHONPATH=src $PY -m vsa_embed.experiments.e13_cycle plan --config experiments/e13-learning-cycle/t5.yaml > /tmp/e13-t5.sh   # configs + commands
PYTHONPATH=src $PY -m vsa_embed.experiments.e13_cycle smoke --output experiments/e13-learning-cycle/smoke --threads 4      # CPU smoke
```

## Smoke (SMOKE — a pipeline check, not a result)

`smoke/smoke.json`, `smoke/report/` (2026-10-08, after merging main 35e9414; CPU, 4 threads, SmolLM2-135M;
`e13_cycle.smoke_config`: 128-token windows, 8 evaluation windows, stage 0 for 3 steps of 256 tokens, round 2 for 2
steps; ≈ 4 min in all, deterministic: a repeat gave the same numbers): every stage ran end to end — `prepare` (seed
ontology 6,024 of 30,121 round-1 edges erased, 3,240 `owns` / 690 `has_part` edges derived, 716 edges to round-2 terms
dropped; round-2 and `defs` streams 50/50 with general text), stage 0 (1.2 s per step), `learn` (24 rule-closure
proposals, all erased gold, + 24 null proposals; 6 testable on the small validation text; none accepted after 3
steps), `write` (the 4 round-2 terms present in the smoke windows read by `linker`: F1 0.55 against gold; their 54
relation / property items through the ranking harness: `definition-in-context` 0.72 vs `none` 0.43, the `store:*` rows
change the option scores but not yet the argmax), `reason` (4 anchors, 33 items, 4 conditions), the stage-4 arms
`read` / `noread` / `fvt` and the stage-5 RTN arms `q4-read` / `q4-noread` / `qlora` (the channel trains, the 4-bit host
stays frozen, `qlora` saves only its adapters), and `report` (L1–L5 with Holm; every verdict False, as expected after a
few steps on 8 windows). Read − no-read after round-2 terms at step 0: −0.009 nats (4 windows).

## Smoke after amendment 2 (SMOKE — a pipeline check, not a result)

`smoke-amendment2/{t5,t7-rood}/` (2026-10-09; CPU, 4 threads, SmolLM2-135M, `smoke_config`; ≈ 1.5 h each because the
machine's load was ≈ 45 on 12 cores — the unloaded smoke takes ≈ 4 min). Every stage ran end to end, plus the new parts:
- the `random` arm;
- `context` on the noread arm's step-0 model: T5 4 windows, 99 definition tokens per window; T7-ROOD 3 windows, 2 of them
  with an SCR note, 19 tokens;
- stage-3 token accounting: per item, T5 `recall:own` 45 / `definition` 140 / `symbolic` 158 tokens; T7-ROOD 60 / 14 / 116.
  T7-ROOD's `definition` scored 2 items, the one sampled anchor with a note;
- the report: the six-test Holm family, L1's gate, `read − definition`, `recall:own − definition` and the cost criterion.
  Every verdict is False, as expected after a few steps.

The report stage was re-run after a fix: `report`'s L3 block reused the variable name `margin` and had overwritten L5's δ.
`tests/test_e13_amendment2.py` covers it now.

At smoke scale, T7-ROOD's SCR notes cost fewer tokens than the recall text, so the token criterion may favour the
definition there.

## T7-ROOD (amendment 1)

- **Data.** TK-H1's `rounds-v1` (cut 2013-01-26; 2,470 round-2 entries; 0 round-1 training documents name one): `train`
  (round 1), `train-round2` (round 2's stream, linked as `round2/train`), `eval-round2` (every window: 2,359 / 2,356 for
  Qwen3), `eval-round1` (forgetting), `eval-general`. `prepare` adds the seed ontology, the rounds alias table (T7's aliases
  with round 2 held out: digest `90d54b…`, the build's), the `defs` stream and the reference strata.
- **Learn.** The seed ontology's erasure is E10.L's (`erase_like_tkl`: content relations `mapped_to`,
  `pharmacological_action`, `branch_second`; 1,966 of 15,594 edges of 2,933 seen entries; one seen set for both
  tokenizers), so TK-L's pipeline (`learn_tkl` → `e10_learn.run_erasure`, `holm+decoy`) reproduces exactly the edges the
  stage-0 store never saw (checked at run time). Its passive vectors and evidence are the stage-0 host's hidden states at
  round-1 terms (`eval-round1`, then round-1 training windows). Rule closure runs as the secondary (`learn-rule_closure`).
- **Smoke** (`smoke-t7-rood/`, SMOKE): every stage ran end to end on the rounds build; TK-L's run reproduced the 1,966
  erased edges (2 probes with the tiny smoke extraction, nothing accepted after 3 steps); rule closure proposes nothing on
  T7's relations.

## Design notes

- **Round split (T5).** Round 2 = the 360 real held-out terms, which no training text names (leakage audit 0). Their
  round-2 text is new T5 documents generated around them with the T5 generator under a fresh seed, so it shares no
  document with the evaluation corpus. The seed ontology does not know them (no frame; round-1 edges to them dropped).
- **Learn gold.** 20% of the round-1 edges are erased from the seed ontology; the text still states those facts, so a
  learner can recover them. Two inverse relations (`owns`, `has_part`) are materialized, as in the E10.9b design,
  which gives rule closure something to close: on the T5 seed it proposes 1,302 edges, every one of them erased gold
  (a CPU check without a model; the acceptance test then decides). TK-L's decompose-then-verify proposer plugs in
  through `learn.proposer`.
- **Continuation.** Round 2 continues the stage-0 run with a fresh AdamW and the same recipe
  (`train.init_mode: continue`): the frame table comes from the arm's ontology, so the arms differ only in the
  round-2 rows (and the `defs` arm in its text).
- **Frozen 4-bit host.** The in-repo E9 claim-B quantizers (`evaluation.quantization.simulate_weight_only_`: RTN, NF4,
  HQQ, GPTQ, AWQ) write the dequantized values into ordinary weights, i.e. exactly what a weight-only 4-bit kernel
  computes with; gradients reach the channel through them like through any frozen linear, so no dequantize-on-the-fly
  layer was needed. The bytes are reported nominally (`host_quantization.host_bytes`: quantized linears at the scheme's
  bits per weight, other host tensors at 16 bits); memory is not saved by the simulation. Stage-5 runs save
  trainable-only states; `training.lm.load_final` rebuilds their host (continuation + quantization).
  `evaluation.channel_probes.load_run` does not: open stage-5 runs with `load_final`.
