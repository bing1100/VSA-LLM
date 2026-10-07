# Stale: holdout superseded before any run (T1c-1)

This build froze holdout `fa738430…` at fraction 0.10 (1,672 held-out entries). Its held-out stratum fell short of
decision 17's entries bar (153 entries with ≥ 5 occurrences on the whole `eval-mimic` split), so the author re-froze the
holdout at fraction 0.30 from the same pre-sample on 2026-10-07 (T1c-1), before any training or E9 run used either
holdout: `runs/v2` (sha256 `7369ec27…`, data root `~/data/vsa-llm/t1c/snomed-mimic3-smollm2-v2`). The documents,
linker and alias policy are unchanged. Use this folder only as a record of the first freeze.
