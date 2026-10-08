# E12 3b — learning from tool-using traces — SMOKE

Pre-registration: `experiments/e12-self-query/preregistration.md` §12 and amendment 16.3. Contrasts: twin pairs (or items) × seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm over B1's two contrasts.

**SMOKE: these numbers do not count toward the pre-registered endpoints.**

## SmolLM2-135M

Arms × seeds: {'I': [1], 'L': [1], 'S': [1], 'T': [1], 'base': [1]}

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

### Means by arm (seed-averaged)

| arm | twins none choice | twins none cloze | twins recall:own choice | twins question format | new words none | two-hop none | two-hop recall:own | agent twin contrast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 0.333 | 0.333 | 1.000 | 0.667 | 0.200 | 0.158 | 0.921 | — |
| L | 0.333 | 0.333 | 1.000 | 1.000 | 0.200 | 0.132 | 0.921 | — |
| S | 0.417 | 0.333 | 0.917 | 0.667 | 0.200 | 0.158 | 0.921 | — |
| T | 0.333 | 0.333 | 0.917 | 1.000 | 0.200 | 0.158 | 0.947 | 1.000 |
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

Predictions (§12): I twins none ≤ 0.65: yes, S twins none ≤ 0.65: yes, S ≥ I: yes, T well-formed calls ≥ 0.9: no

