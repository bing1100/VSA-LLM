# E12 3b — learning from tool-using traces — SMOKE

Pre-registration: `experiments/e12-self-query/preregistration.md` §12, amendments 16.3 and 16.5. Contrasts: twin pairs (or items) × seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm over B1's two contrasts.

**SMOKE: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-135M

Arms × seeds: {'T@C0p': [1], 'I': [1], 'L': [1], 'S': [1], 'T': [1], 'base': [1]}

### B1 — internalization (twin contrast without the tool, choice)

| contrast | estimate |
|---|---|
| B1: I − L | +0.0000 [+0.0000, +0.0000] (p 1, Holm 1; 3 units × 1 seeds) |
| B1: S − L | +0.0833 [-0.2752, +0.4419] (p 0.423, Holm 0.845; 3 units × 1 seeds) |

Reading: not shown: neither I − L nor S − L is significant

### B2 — tool use learned (agentic twins, first 100 pairs)

Test T − I: —

| arm | format_ok | relevant | answered | twin contrast agent | twin contrast no_tool | twin contrast fixed |
|---|---:|---:|---:|---:|---:|---:|
| T | 0.750 | 0.250 | 1.000 | 1.000 | 1.000 | 0.000 |
| I | — | — | — | — | — | — |
| base | — | — | — | — | — | — |
| T@C0p | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 |

### Means by arm (seed-averaged)

| arm | twins none choice | twins none cloze | twins recall:own choice | twins question format | new words none | two-hop none | two-hop recall:own | agent twin contrast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 0.333 | 0.333 | 1.000 | 0.667 | 0.200 | 0.158 | 0.921 | — |
| L | 0.333 | 0.333 | 1.000 | 1.000 | 0.200 | 0.132 | 0.921 | — |
| S | 0.417 | 0.333 | 0.917 | 0.667 | 0.200 | 0.158 | 0.921 | — |
| T | 0.333 | 0.333 | 0.917 | 1.000 | 0.200 | 0.158 | 0.947 | 1.000 |
| T@C0p | 0.583 | 0.333 | 0.917 | 0.667 | 0.167 | 0.000 | 0.812 | 1.000 |
| base | 0.333 | 0.333 | 1.000 | 0.667 | 0.233 | 0.132 | 0.921 | — |

### Secondaries (family SB; reported, never promoted)

| contrast | estimate |
|---|---|
| T − L (twins, no tool) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| T − I (twins, no tool): do the traces add to the answers? | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| S − I (twins, no tool) | +0.0833 [-0.2752, +0.4419] (p 0.423; 3 units × 1 seeds) |
| I − L (twins cloze, never-trained wording) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| I − L (twins, trained question format, no tool) | -0.3333 [-1.7676, +1.1009] (p 0.423; 3 units × 1 seeds) |
| I − L (twins, recall:own) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| I − L (new words, no tool; accuracy) | +0.0000 [+0.0000, +0.0000] (p 1; 15 units × 1 seeds) |
| I − L (two-hop, no tool; accuracy) | +0.0263 [-0.0290, +0.0816] (p 0.331; 19 units × 1 seeds) |
| I − L (two-hop, chained recall; accuracy) | +0.0000 [+0.0000, +0.0000] (p 1; 19 units × 1 seeds) |
| S − L (twins cloze, never-trained wording) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| S − L (twins, trained question format, no tool) | -0.3333 [-1.7676, +1.1009] (p 0.423; 3 units × 1 seeds) |
| S − L (twins, recall:own) | -0.0833 [-0.4419, +0.2752] (p 0.423; 3 units × 1 seeds) |
| S − L (new words, no tool; accuracy) | +0.0000 [+0.0000, +0.0000] (p 1; 15 units × 1 seeds) |
| S − L (two-hop, no tool; accuracy) | +0.0263 [-0.0290, +0.0816] (p 0.331; 19 units × 1 seeds) |
| S − L (two-hop, chained recall; accuracy) | +0.0000 [+0.0000, +0.0000] (p 1; 19 units × 1 seeds) |
| T − L (twins cloze, never-trained wording) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| T − L (twins, trained question format, no tool) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |
| T − L (twins, recall:own) | -0.0833 [-0.4419, +0.2752] (p 0.423; 3 units × 1 seeds) |
| T − L (new words, no tool; accuracy) | +0.0000 [+0.0000, +0.0000] (p 1; 15 units × 1 seeds) |
| T − L (two-hop, no tool; accuracy) | +0.0263 [-0.0290, +0.0816] (p 0.331; 19 units × 1 seeds) |
| T − L (two-hop, chained recall; accuracy) | +0.0263 [-0.0290, +0.0816] (p 0.331; 19 units × 1 seeds) |
| L − base (twins, no tool): the size-matched update itself | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) |

### Decision 64 (amendment 16.5): does LoRA learn to use the decoded store, whatever the host or binding?

Equivalence: two one-sided t tests at α = 0.05 (90% CI inside ±margin; ±0.05 with the decoded store in context, ±0.075 for B1's I − L); secondaries, never promoted.

| contrast | estimate | 90% CI | reading |
|---|---|---|---|
| C0′ − C5 host, arm T, twins recall:own (the tool reads C5's store) | +0.0000 [+0.0000, +0.0000] (p 1; 3 units × 1 seeds) | [+0.0000, +0.0000] | ≈ (90% CI within ±0.05) |
| C0′ − C5 host, arm T, agentic twins (the tool reads C5's store) | — | — | not available |

| control | estimate | reading |
|---|---|---|
| T@C0p − 0.5 (twins, no tool; no channel: ≈ 0) | +0.0833 [-0.2752, +0.4419] (p 0.423; 3 units × 1 seeds) | no difference shown |

Predictions (§12, 16.5): I twins none ≤ 0.65: yes, S twins none ≤ 0.65: yes, S ≥ I: yes, T well-formed calls ≥ 0.9: no, 16.5: C0′ host ≈ C5 host with the decoded store (arm T, ±0.05): yes, 16.5: T@C0p without the tool ≈ 0.5 (CI includes 0.5): yes

## SmolLM2-360M

Arms × seeds: {'T-rf': [1]}

### B1 — internalization (twin contrast without the tool, choice)

| contrast | estimate |
|---|---|

Reading: not available

### B2 — tool use learned (agentic twins, first 100 pairs)

Test T − I: —

| arm | format_ok | relevant | answered | twin contrast agent | twin contrast no_tool | twin contrast fixed |
|---|---:|---:|---:|---:|---:|---:|
| T-rf | 1.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |

### Means by arm (seed-averaged)

| arm | twins none choice | twins none cloze | twins recall:own choice | twins question format | new words none | two-hop none | two-hop recall:own | agent twin contrast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T-rf | 0.500 | 0.250 | 0.875 | 0.500 | — | — | — | 1.000 |

### Secondaries (family SB; reported, never promoted)

| contrast | estimate |
|---|---|

Predictions (§12, 16.5): 

