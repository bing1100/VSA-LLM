# Stale: holdout not the track's frozen holdout

This slice froze holdout `fd4244ea…`, but its config capped `eval_general_docs` at 1,000, and those documents feed the
chars-per-token calibration of the PubMed/general mix. The training stream and therefore the frequency pre-sample of the
full config differ by one document, which changes the holdout. The track's holdout was re-frozen on 2026-10-02 from the
full config (`runs/v1`, sha256 `1c477afd…`) before any training run saw either. Use this folder only as a pipeline test.
