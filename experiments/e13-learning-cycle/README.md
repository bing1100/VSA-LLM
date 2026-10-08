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
| `t7-rood.yaml` | T7-ROOD, written against the interface TK-H1 is building (`experiments/t7-new-vocabulary/ROOD.md`); **TODO markers**: `prepare` refuses it until those files exist |
| `preregistration.md` | stages, arms, endpoints L1–L5 with kill criteria, statistics, deviations policy |
| `queue-commands.sh` | `jobqueue add` lines (T5: 54.4985–54.4989; T7-ROOD: 54.4996–54.4999 once its files exist) with GPU-h |
| `configs/t5/` | the trainer configs `plan` writes (stage 0 and every round-2 arm) |
| `smoke/` | the CPU smoke (SMOKE; SmolLM2-135M, a few steps) — a pipeline check, not a result |
| `runs/<track>/` | (created by the jobs) `stage0/`, `cycle/<host>-s<seed>/{learn,write,reason}/`, `round2/<host>-<arm>[-<scheme>]-s<seed>/` |
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
- `vsa_embed.experiments.e13_cycle` — `prepare` (round split, corpora, reference strata), `learn`, `write`, `reason`,
  `round2` (materialize an arm, train, locality), `report`, `plan`, `smoke`.
- Opt-in trainer keys (`vsa_embed.training.lm`; absent keys change nothing): `train.init_mode: continue` (+
  `init_merge_lora`), `model.host_quantization`, `channel.entry_rows`, `eval.points`.

## Commands

```bash
PY=~/anaconda3/envs/vsa-repro/bin/python
PYTHONPATH=src $PY -m vsa_embed.experiments.e13_cycle plan --config experiments/e13-learning-cycle/t5.yaml > /tmp/e13-t5.sh   # configs + commands
PYTHONPATH=src $PY -m vsa_embed.experiments.e13_cycle smoke --output experiments/e13-learning-cycle/smoke --threads 4      # CPU smoke
```

## Smoke (SMOKE — a pipeline check, not a result)

`smoke/smoke.json`, `smoke/report/` (2026-10-08, CPU, 4 threads, SmolLM2-135M; `e13_cycle.smoke_config`: 128-token
windows, 8 evaluation windows, stage 0 for 3 steps of 256 tokens, round 2 for 2 steps): every stage ran end to end —
`prepare` (27 s; seed ontology 6,024 of 30,121 round-1 edges erased, 3,240 `owns` / 690 `has_part` edges derived, 716 edges
to round-2 terms dropped; round-2 and `defs` streams 50/50 with general text), stage 0 (2.2 s per step), `learn`
(24 rule-closure proposals, all erased gold, + 24 null proposals; 6 testable on the small validation text; none accepted
after 3 steps), `write` (the 4 round-2 terms present in the smoke windows read by `linker`: F1 0.55 against gold;
5.2k definition forward tokens per term), `reason` (4 anchors, 33 items, 4 conditions), the stage-4 arms `read` /
`noread` / `fvt` and the stage-5 RTN arms `q4-read` / `q4-noread` / `qlora` (the channel trains, the 4-bit host stays
frozen, `qlora` saves only its adapters), and `report` (L1–L5 with Holm; every verdict False, as expected after a few
steps on 8 windows). Read − no-read after round-2 terms at step 0: −0.009 nats (4 windows).

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
