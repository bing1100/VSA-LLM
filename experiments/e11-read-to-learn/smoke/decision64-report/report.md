**SMOKE TEST — not a result** (CPU, SmolLM2-360M seed 1, 8 T4-H terms / 5 windows / 7 occurrences, 5 T5-N words; preregistration §16.6).

# E11 read-to-learn — pooled report

Primary endpoints (Holm over P1 and P2; P2's p is the intersection–union p of its specificity components, §16.1):

| set | host | model | contrast | test | difference [95% CI] | seeds | p | p (Holm) |
|---|---|---|---|---|---|---:|---:|---:|
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | linker − none | loss after term | +1.17% [+0.02, +2.57] | 1 | 0.0103 | 0.4205 |

**Reading (b), P2 as amended (§16.1):** not supported — linker − none +1.17% (p 0.0103), linker − random +0.11% (p 0.4205), linker − linker-random +1.27% (p 0.1641); p (IUT) 0.4205, p (Holm) 0.4205.

Specificity of the loss gain on every held-out set (linker against no frame and against frames of wrong content; §16.1; outside T4-H unadjusted):

| set | host | model | linker − none | linker − random | linker − linker-random | p (IUT) | specific |
|---|---|---|---|---|---|---:|---|
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | +1.17% [+0.02, +2.57] | +0.11% [-0.08, +0.26] | +1.27% [-0.01, +2.89] | 0.4205 | no |

Definition-encoder competitor (§16.2; C6d with the read definition's encoding against C5 with the linker's frame; paired by seed on the same items and windows; unadjusted):

| set | host | contrast | test | difference [95% CI] | seeds | p |
|---|---|---|---|---|---:|---:|
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d encoder − C5 linker | loss after term | -1.13% [-2.01, -0.31] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | (C6d encoder − C6d none) − (C5 linker − C5 none) | loss after term | -0.54% [-1.36, +0.16] | 1 | 0.2564 |

Cost per learned word (seed means, primary style): write = forward tokens of the reading (linker: 1 + candidates passes; encoder: one pass) or training tokens (gradient ×1; one training token ≈ 3 forward tokens); use = context tokens (frames and encoder: 0).

| set | host | model | linker write | linker passes | encoder write | gradient train | context / query | context / window occurrence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | 3513 | 34.8 | — | — | — | 70.6 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | — | — | 65.1 | — | — | — |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | — | — | 76.6 | — | — | — |

Secondary contrasts (unadjusted p):

| set | host | model | contrast | test | difference [95% CI] | seeds | p |
|---|---|---|---|---|---|---:|---:|
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | oracle − none | loss after term | -0.00% [-0.21, +1.13] | 1 | 0.8410 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | linker − typeprior | loss after term | +1.81% [+0.10, +4.22] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | linker − random | loss after term | +0.11% [-0.08, +0.26] | 1 | 0.4205 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | linker − linker-random | loss after term | +1.27% [-0.01, +2.89] | 1 | 0.1641 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | context − none | loss after term | -2.13% [-4.68, -0.05] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | context − linker | loss after term | -3.26% [-7.07, -0.07] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C5 | context+linker − context | loss after term | +1.26% [+0.02, +2.85] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | property | +0.0238 [+0.0000, +0.0714] | 1 | 0.6368 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | property | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | property | +0.0238 [+0.0000, +0.0714] | 1 | 0.6368 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | paraphrase | +0.1429 [+0.0000, +0.3571] | 1 | 0.2090 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | paraphrase | +0.0714 [+0.0000, +0.2143] | 1 | 0.7164 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | paraphrase | +0.0714 [+0.0000, +0.2143] | 1 | 0.6368 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | entailment | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | entailment | -0.0714 [-0.2143, +0.0000] | 1 | 0.6070 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | entailment | +0.0714 [+0.0000, +0.2143] | 1 | 0.6070 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | loss after term | +0.64% [-0.27, +1.38] | 1 | 0.1436 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | loss after term | +0.78% [+0.54, +1.03] | 1 | 0.0103 |
| t4 heldout | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | loss after term | -0.14% [-1.17, +0.34] | 1 | 0.5436 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | property | +0.0000 [-0.0625, +0.0625] | 1 | 1.0000 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | property | -0.0417 [-0.1458, +0.0417] | 1 | 0.5373 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | property | +0.0417 [-0.0417, +0.1250] | 1 | 0.4577 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | entailment | -0.1000 [-0.3000, +0.0000] | 1 | 0.7463 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | entailment | -0.1000 [-0.3000, +0.0000] | 1 | 0.5771 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | entailment | +0.0000 [-0.2025, +0.3000] | 1 | 1.0000 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | paraphrase | -0.0833 [-0.2083, +0.0000] | 1 | 0.2587 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | paraphrase | -0.1250 [-0.2917, +0.0417] | 1 | 0.2687 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | paraphrase | +0.0417 [-0.0844, +0.1667] | 1 | 0.8557 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − none | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | oracle − none | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
| t5 new | HuggingFaceTB/SmolLM2-360M/train | C6d | encoder − oracle | statement_accuracy | +0.0000 [+0.0000, +0.0000] | 1 | 1.0000 |
