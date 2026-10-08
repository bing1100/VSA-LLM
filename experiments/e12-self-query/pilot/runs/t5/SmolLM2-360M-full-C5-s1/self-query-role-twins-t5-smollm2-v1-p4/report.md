# E12 self-query, phase A (twins) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Items `experiments/e9-retrofit/items/role-twins-t5-smollm2-v1`; 600 of 600 concepts link. Conditions: `definition`, `recall:C5tr`. Stores: `C5tr` = /home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5tr-s1 (translation, subtract).

| condition | kind | units | contrast | item | decode | all slots decoded |
|---|---|---:|---:|---:|---:|---:|
| definition | choice | 300 | 0.748 | 0.533 | — | — |
| recall:C5tr | choice | 300 | 0.497 | 0.500 | 0.443 | 0.000 |

`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the gold fillers the condition's context states (twins: the four critical slots of a pair).
