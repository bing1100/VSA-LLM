# Provenance note

This run was launched at about 22:58 UTC on 2026-09-30 from the clean commit `5762dca`, with `num_threads: 1`, and imported all of its code at launch. Its `manifest.json` reports `git_dirty: true` because manifest schema v2 then recorded the git state at the **end** of the run, after unrelated files (`compose.py`, the plan queue) had been edited in the working tree. The code this process ran is `5762dca` as committed. Later runners record the git state at run start (`git_state_recorded_at`).
