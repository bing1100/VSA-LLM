# Experiment 00 implementation

This is the executable seed for the research specification at [`../../resources/vsa-understanding/experiments/00-capacity-and-algebra/`](../../resources/vsa-understanding/experiments/00-capacity-and-algebra/).

## Current scope

- differentiable FFT real HRR and unitary-role HRR;
- elementwise MAP binding as a control;
- weighted, row-normalized bundling;
- circular-correlation unbinding and exhaustive cleanup retrieval;
- dimension/load/noise/seed sweeps;
- CSV metrics, resolved config, environment manifest, and Markdown report;
- forward and gradient equivalence against naive circular convolution.

Generalized HRR, correlated atomics, path depth, sharding, learned cleanup, calibrated abstention, and large-scale timing remain subsequent experiment-00 milestones. The design config in `resources/` is broader than the currently executable schema, so use `smoke.yaml` first.

## Run

From `VSA-LLM/`, without installing the package:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m vsa_embed.experiments.capacity \
  --config experiments/00-capacity-and-algebra/smoke.yaml \
  --output experiments/00-capacity-and-algebra/runs/smoke
PYTHONPATH=src python -m vsa_embed.experiments.realistic_capacity \
  --config experiments/00-capacity-and-algebra/realistic.yaml \
  --output experiments/00-capacity-and-algebra/runs/realistic
```

For editable installation, use `python -m pip install -e '.[dev]'` in an isolated environment. No installation is needed for the commands above.

## Interpretation

The benchmark stores one role–filler pair per bundle item, superposes bound pairs, unbinds each queried role, and ranks the true filler among distractor atomics. `top1`, MRR, rank, cosine margin, latency, and storage are recorded per trial. The smoke run validates direction and tooling; it does not establish the proposal's realistic capacity acceptance gate.

## Preliminary smoke result

The corrected 72-trial CPU run (2 dimensions × 3 loads × 2 noise levels × 3 algebras × 2 seeds) shows the expected capacity frontier:

- all three algebras achieve 100% top-1 at dimension 512 and bundle size 16, except unitary HRR at 96.9%;
- at dimension 128 and bundle size 16, top-1 falls to roughly 45–56%;
- at bundle size 64, no backend reaches 50% top-1 even at dimension 512;
- the small `noise_std=0.1` perturbation is not the dominant failure mode for HRR; superposition interference is;
- unitary roles improve exact single-pair inversion, but do **not** automatically improve bundled cleanup capacity over ordinary real HRR in this isotropic setting.

These are diagnostic—not confirmatory—results because two seeds and 32 queries per condition are insufficient for uncertainty estimates. The next experiment-00 increment should add correlated atomics and degree distributions, then confidence/abstention calibration. Only after those pass should sharding or a learned cleanup model be evaluated.

The canonical smoke artifacts are under `runs/smoke/`: `metrics.csv`, `resolved_config.yaml`, `manifest.json`, and `report.md`.

## Milestone 2 — correlated atomics, degree skew, and calibration

`realistic.yaml` adds controlled mean candidate cosine (0.0/0.3), fixed/Poisson/heavy-tailed degree profiles, 36 graph-memory conditions, and disjoint calibration/evaluation seeds. It records every query and fits a Platt calibrator from the **observable** top-1/top-2 similarity gap. The generated `capacity_model.json` stores condition-specific calibrators and abstention thresholds.

The study completed **39,360 query observations**. Main findings:

- D=512 materially outperforms D=256 (mean top-1 0.894 vs 0.632; mean ECE 0.019 vs 0.046).
- Correlation 0.3 lowers mean top-1 from 0.799 to 0.726 and accepted coverage from 0.748 to 0.598.
- Heavy-tailed degree is the central failure regime (mean top-1 0.622; mean ECE 0.051; selective accuracy 0.924).
- 29/36 conditions meet ECE ≤0.05, but only 17/36 transfer a calibration-selected ≥95% accuracy threshold to held-out seeds; only 15/36 pass both.
- No algebra dominates consistently: mean top-1 is 0.755 real HRR, 0.766 unitary HRR, and 0.767 MAP.

**Decision:** do not promote one universal backend or abstention policy. The next change should be degree-aware calibration and typed/community sharding, with the same held-out protocol. Results are under `runs/realistic/` (`summary.csv`, `observations.csv`, `capacity_model.json`, `resolved_config.yaml`, and `report.md`).
