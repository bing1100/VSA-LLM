# E11 read-to-learn — pooled report

Primary endpoints (Holm over both):

| set | host | model | contrast | test | difference [95% CI] | seeds | p (Holm) |
|---|---|---|---|---|---|---:|---:|

Secondary contrasts (unadjusted p):

| set | host | model | contrast | test | difference [95% CI] | seeds | p |
|---|---|---|---|---|---|---:|---:|
| t7rood heldout | HuggingFaceTB/SmolLM2-135M/lora | C5 | linker − none | loss after term | +0.00% [+0.00, +0.00] | 1 | 1.0000 |
| t7rood heldout | HuggingFaceTB/SmolLM2-135M/lora | C5 | oracle − none | loss after term | -0.07% [-0.09, +0.58] | 1 | 0.5586 |
| t7rood heldout | HuggingFaceTB/SmolLM2-135M/lora | C5 | linker − random | loss after term | -0.04% [-0.04, +0.02] | 1 | 0.5586 |
| t7rood heldout | HuggingFaceTB/SmolLM2-135M/lora | C5 | gradient×1.0 − none | loss after term | -0.01% [-0.06, +1.84] | 1 | 0.7682 |
