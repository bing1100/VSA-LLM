# E14 LG-P — search policies on `HuggingFaceTB/SmolLM2-360M` (800 problems)

| policy | budget | accuracy | proofs found (provable) | soundness | steps (found) | model calls |
|---|---:|---:|---:|---:|---:|---:|
| bfs | 10 | 0.858 | 0.721 | 1.000 | 4.374 | 0.0 |
| bfs | 30 | 0.914 | 0.831 | 1.000 | 6.522 | 0.0 |
| ceiling | 30 | 1.000 | 1.000 | 1.000 | 5.338 | 0.0 |
| random | 10 | 0.876 | 0.757 | 1.000 | 4.440 | 0.0 |
| random | 30 | 0.959 | 0.919 | 1.000 | 6.787 | 0.0 |
| score | 10 | 0.874 | 0.752 | 1.000 | 4.420 | 1.9 |
| score | 30 | 0.931 | 0.865 | 1.000 | 6.337 | 3.0 |
| sos | 10 | 0.925 | 0.853 | 1.000 | 3.991 | 0.0 |
| sos | 30 | 1.000 | 1.000 | 1.000 | 5.338 | 0.0 |
