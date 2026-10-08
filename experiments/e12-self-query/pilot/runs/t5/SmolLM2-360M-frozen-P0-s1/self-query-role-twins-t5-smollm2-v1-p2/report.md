# E12 self-query, phase A (twins) — P0 seed 1 (HuggingFaceTB/SmolLM2-360M/frozen) — PILOT

Items `experiments/e9-retrofit/items/role-twins-t5-smollm2-v1`; 600 of 600 concepts link. Conditions: `symbolic`, `wrong:C5@1`. Stores: `C5@1` = /home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1 (hrr, correlation).

| condition | kind | units | contrast | item | decode | all slots decoded |
|---|---|---:|---:|---:|---:|---:|
| symbolic | choice | 300 | 0.730 | 0.515 | 1.000 | 1.000 |
| wrong:C5@1 | choice | 300 | 0.311 | 0.485 | 0.000 | 0.000 |

`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the gold fillers the condition's context states (twins: the four critical slots of a pair).
