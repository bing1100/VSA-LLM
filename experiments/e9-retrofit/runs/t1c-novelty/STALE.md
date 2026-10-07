# Stale: evaluated against the superseded T1c holdout

This P0 (evaluation-only) run read `~/data/vsa-llm/t1c/snomed-mimic3-smollm2-v1` (holdout `fa738430…`, fraction 0.10).
The holdout was re-frozen at fraction 0.30 before any training run (T1c-1); the same evaluation on the v2 corpora is
`runs/t1c-novelty-v2`. The documents are identical, so `all`, `unlinked` and `inside` match; the held-out, rare and
unseen strata differ.
