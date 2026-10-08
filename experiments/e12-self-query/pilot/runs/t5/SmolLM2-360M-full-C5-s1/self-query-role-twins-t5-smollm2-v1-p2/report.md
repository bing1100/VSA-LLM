# E12 self-query, phase A (twins) — C5 seed 1 (HuggingFaceTB/SmolLM2-360M/train) — PILOT

Items `experiments/e9-retrofit/items/role-twins-t5-smollm2-v1`; 600 of 600 concepts link. Conditions: `recall:C5ut`, `symbolic`. Stores: `C5ut` = /home/bhux/workplace/VSA-LLM/experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5ut-s1 (additive, bundle).

| condition | kind | units | contrast | item | decode | all slots decoded |
|---|---|---:|---:|---:|---:|---:|
| recall:C5ut | choice | 300 | 0.495 | 0.501 | 0.897 | — |
| symbolic | choice | 300 | 0.863 | 0.546 | 1.000 | 1.000 |

`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the gold fillers the condition's context states (twins: the four critical slots of a pair).
