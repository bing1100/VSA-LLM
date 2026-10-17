# E14 RD-11 — multi-round active reading with relation discovery (t1, 120 definitions, style `scope`)

Hidden at the start: none. Teacher $5.40 (74 calls, 0 cached).

| round | relations | facts/def (single → consolidated) | NEW share | unresolved | visible P / R (single) | visible P / R (consolidated) | gold recall, all relations | ceiling | accepted next |
|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 0 | 5 | 2.26 → 2.04 | 0.720 | 0.310 | 0.492 / 0.054 | 0.529 / 0.050 | 0.050 | 0.140 | has_manifestation (30), affects_site (13), transports (11), leads_to (12) |
| 1 | 9 | 2.34 → 2.18 | 0.409 | 0.306 | 0.491 / 0.049 | 0.520 / 0.049 | 0.049 | 0.140 | acts_on_substrate (9), has_member (9) |
| 2 | 11 | 2.76 → 2.40 | 0.323 | 0.326 | 0.484 / 0.058 | 0.527 / 0.054 | 0.054 | 0.140 | — |

## Re-analysis (discovered relations = every relation outside the round-0 visible set)

| round | visible P / R | gold recall, all relations | discovered relations (edges) |
|---:|---|---:|---|
| 0 | 0.529 / 0.059 | 0.059 |  |
| 1 | 0.520 / 0.056 | 0.056 | has_manifestation (24), affects_site (22), leads_to (10), transports (8) |
| 2 | 0.527 / 0.063 | 0.063 | has_manifestation (25), affects_site (22), leads_to (13), transports (8), has_member (7), acts_on_substrate (6) |
