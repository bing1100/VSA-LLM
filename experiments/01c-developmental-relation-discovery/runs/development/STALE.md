# Stale run

`stale: true` — marked 2026-09-30 by the implementation audit ([audit.md](../../../../resources/plan-improvement/audit.md), finding F4).

With the current code the basis-HRR MRR falls from 0.2585 to 0.2145; the README's "basis model on the Pareto frontier" is a pre-fix artifact. The selector now picks `residual_hrr` instead of `offset_residual_hrr` (the gate still fails).

Keep for provenance only; do not cite these numbers.
