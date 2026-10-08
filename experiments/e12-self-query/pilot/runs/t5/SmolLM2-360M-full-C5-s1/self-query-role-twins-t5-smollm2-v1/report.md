# E12 self-query, phase A (twins) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Items `experiments/e9-retrofit/items/role-twins-t5-smollm2-v1`; 600 of 600 concepts link. Conditions: `none`, `recall:own`. Stores: `own` = /home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1 (hrr, correlation).

| condition | kind | units | contrast | item | decode | all slots decoded |
|---|---|---:|---:|---:|---:|---:|
| none | choice | 300 | 0.507 | 0.506 | — | — |
| recall:own | choice | 300 | 0.790 | 0.539 | 0.959 | 0.857 |

`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the gold fillers the condition's context states (twins: the four critical slots of a pair).
