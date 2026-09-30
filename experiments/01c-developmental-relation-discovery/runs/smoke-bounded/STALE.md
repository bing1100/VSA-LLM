# Stale run

`stale: true` — marked 2026-09-30 by the implementation audit ([audit.md](../../../../resources/plan-improvement/audit.md), finding F4).

`smoke.yaml` was edited after this run; the recorded correction norm 1.204 exceeds the current hard bound 0.25, so 01c.0/01c.1 cannot be regenerated. The 01c.0 → 01c.1 improvement is also confounded: learning rate and steps changed together with the bound.

Keep for provenance only; do not cite these numbers.
