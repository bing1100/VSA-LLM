# E10.L — the *learn* tool (decision 63, WP TK-L)

The learn tool turns passive learning into explicit structure: **decompose, propose, verify**. Pre-registration:
[`preregistration.md`](preregistration.md). Design: `manuscript/toolkit-methodology-2026-10.md` §1.2, M3, §3.2, §4.

## Code

- `src/vsa_embed/learn.py`, the method. Its interface is what TK-E13's `ConceptStore` calls:
  - `propose(Evidence) -> list[dict]`; each proposal is `{"concept", "relation", "filler", "score", "source"}`;
  - `accept(proposals, AcceptanceTest) -> list[dict]`: the accepted copies, with `utility`, `n`, `t`, `p`, `p_adjusted`
    and `accepted`.

  Building blocks:
  - `Dictionary`: a trained composer's atomics, operator and unbinding, with typed candidates;
  - `decompose`: warm-start non-negative OMP, Lasso, or the unbind pursuit;
  - `crossfit_decoder`;
  - `rule_closure`: AMIE rules;
  - `authoring_proposals`: E7 cards;
  - `VectorUtilityTest` and `LossUtilityTest`;
  - decision rules `holm`, `bh`, `knockoff`, `decoy` and `holm+decoy`;
  - `null_proposals` (relabel / swap) and `permute_within_type`;
  - metrics.
- `src/vsa_embed/experiments/e10_learn.py`, the runner:

  | Command | What it does |
  |---|---|
  | `synthetic` | the E10.0 planted world |
  | `erased` | trained real-track stores |
  | `extract` | host hidden states at linked occurrences (GPU) |
  | `extract-items` | new-term or store-entry mentions (GPU) |
  | `placement` | the TK-H3L sets |
  | `report` | L4a / L4b pooled over seeds |
  | `placement-report` | placement pooled over seeds |
  | `queue` | the job plan; `--dry-run` prints it |

- `kg_baselines.train_kge` gained an opt-in `batch_size`; the full-batch path is unchanged.
- Tests: `tests/test_learn.py`, `tests/test_e10_learn.py` (synthetic fixtures, null-world calibration, the committed
  MeSH items' format).

## Runs

```bash
PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python
# synthetic calibration (CPU, ≈ 10 min): evaluation seeds 101/202/303, both dictionaries
PYTHONPATH=src $PY -m vsa_embed.experiments.e10_learn synthetic --config experiments/e10-self-semantics/e10-learn/configs/synthetic.yaml \
  --output experiments/e10-self-semantics/e10-learn/runs/synthetic-v1/prior
PYTHONPATH=src $PY -m vsa_embed.experiments.e10_learn synthetic --config experiments/e10-self-semantics/e10-learn/configs/synthetic.yaml \
  --set synthetic.dictionary=truth --output experiments/e10-self-semantics/e10-learn/runs/synthetic-v1/truth
# real tracks: the queue plan (GPU extraction, then CPU evaluation, then reports)
bash experiments/e10-self-semantics/e10-learn/queue-commands.sh
```

| Folder | Contents |
|---|---|
| `runs/synthetic-v1/` | synthetic calibration (committed) |
| `runs/<track>-<arm>-s<seed>/` | erased-edge runs: `summary.json`, `proposals.jsonl.gz`, `pool.npz`, `per_concept.json`, `report.md`, `resolved_config.yaml`, `manifest.json` |
| `runs/place-<set>-<track>-s<seed>/` | MeSH placement: `summary.json`, `rows.jsonl.gz`, `report.md` |
| `smoke/` | labelled CPU smokes (pipeline checks, not results) |
| `~/data/vsa-llm/e10/learn-features/` | extracted hidden states (not committed) |
| `~/data/vsa-llm/e10/learn-placement/` | TaxoExpan / TMN outputs (local data) |

## Data rules

- Every run here is evaluation-only, on existing checkpoints.
- T7 / T4 / T5 / T1 / WordNet are open tracks.
- Nothing licensed is read: no MIMIC, no SNOMED, no UMLS. OET and ICD-10-CM H3 need the licensed T1c store and are not
  queued (decision 58).
- The TaxoExpan and TMN items stay under `~/data/vsa-llm/toolkit-learn/`, and their placement outputs under
  `~/data/vsa-llm/e10/learn-placement/`.
