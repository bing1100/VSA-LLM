**SMOKE TEST — not a result.**

# E14 LG — scores on `HuggingFaceTB/SmolLM2-135M`

| set · format · condition | all | by depth |
|---|---:|---|
| composition|mc|exec-nobind | 0.150 (20) | 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.00, 6: 0.20 |
| composition|mc|exec-sym | 0.550 (20) | 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.67, 6: 0.60 |
| composition|mc|exec-vsa | 0.550 (20) | 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.67, 6: 0.60 |
| composition|mc|facts | 0.150 (20) | 2: 0.00, 3: 0.00, 4: 0.25, 5: 0.33, 6: 0.20 |
| composition|mc|none | 0.150 (20) | 2: 0.00, 3: 0.25, 4: 0.25, 5: 0.00, 6: 0.20 |
| composition|mc|text | 0.200 (20) | 2: 0.00, 3: 0.00, 4: 0.50, 5: 0.33, 6: 0.20 |
| edits|tf|exec-stale | 0.650 (20) | 1: 0.60, 2: 0.67, 3: 0.67 |
| edits|tf|exec-vsa | 0.650 (20) | 1: 0.60, 2: 0.67, 3: 0.67 |
| edits|tf|none | 1.000 (20) | 1: 1.00, 2: 1.00, 3: 1.00 |
| edits|tf|text-edited | 0.800 (20) | 1: 0.80, 2: 0.83, 3: 0.78 |
| fewshot|mc|exec-vsa | 0.250 (20) | 1: 0.20, 10: 0.20, 3: 0.40, 5: 0.20 |
| fewshot|mc|icl | 0.150 (20) | 1: 0.40, 10: 0.20, 3: 0.00, 5: 0.00 |
| fewshot|mc|none | 0.100 (20) | 1: 0.00, 10: 0.20, 3: 0.00, 5: 0.20 |
| prop-lp|tf|exec-shuffled | 0.500 (20) | 1: 0.67, 2: 0.80, 3: 0.25, 4: 0.50, 5: 0.00 |
| prop-lp|tf|exec-sym | 0.500 (20) | 1: 0.83, 2: 0.60, 3: 0.25, 4: 0.50, 5: 0.00 |
| prop-lp|tf|exec-vsa | 0.550 (20) | 1: 0.83, 2: 0.80, 3: 0.25, 4: 0.50, 5: 0.00 |
| prop-lp|tf|none | 0.350 (20) | 1: 0.50, 2: 0.40, 3: 0.25, 4: 0.50, 5: 0.00 |
| prop-lp|tf|text | 0.350 (20) | 1: 0.50, 2: 0.40, 3: 0.25, 4: 0.50, 5: 0.00 |
| prop-rp|tf|exec-shuffled | 0.300 (20) | 1: 0.17, 2: 0.33, 3: 0.33, 4: 0.50, 5: 0.25 |
| prop-rp|tf|exec-sym | 0.400 (20) | 1: 0.50, 2: 0.00, 3: 0.67, 4: 0.50, 5: 0.25 |
| prop-rp|tf|exec-vsa | 0.400 (20) | 1: 0.50, 2: 0.00, 3: 0.67, 4: 0.50, 5: 0.25 |
| prop-rp|tf|none | 0.250 (20) | 1: 0.50, 2: 0.00, 3: 0.00, 4: 0.25, 5: 0.25 |
| prop-rp|tf|text | 0.250 (20) | 1: 0.50, 2: 0.00, 3: 0.00, 4: 0.25, 5: 0.25 |
| rulebase|tf|exec-vsa | 0.650 (20) | 1: 0.90, 3: 0.40, 5: 0.40 |
| rulebase|tf|text | 0.600 (20) | 1: 0.70, 3: 0.60, 5: 0.40 |
| rulebase|tf|text-retrieved | 0.550 (20) | 1: 0.70, 3: 0.60, 5: 0.20 |
| taxonomy|cloze|exec-sym | 1.000 (7) | 0: 1.00, 1: 1.00 |
| taxonomy|cloze|exec-vsa | 1.000 (7) | 0: 1.00, 1: 1.00 |
| taxonomy|cloze|none | 0.143 (7) | 0: 0.20, 1: 0.00 |
| taxonomy|cloze|text | 0.571 (7) | 0: 0.80, 1: 0.00 |
| taxonomy|tf|exec-nobind | 0.615 (13) | 0: 0.50, 1: 1.00 |
| taxonomy|tf|exec-shuffled | 0.923 (13) | 0: 0.90, 1: 1.00 |
| taxonomy|tf|exec-sym | 0.538 (13) | 0: 0.40, 1: 1.00 |
| taxonomy|tf|exec-sym@0.1 | 0.615 (13) | 0: 0.50, 1: 1.00 |
| taxonomy|tf|exec-sym@0.3 | 0.615 (13) | 0: 0.50, 1: 1.00 |
| taxonomy|tf|exec-vsa | 0.846 (13) | 0: 0.80, 1: 1.00 |
| taxonomy|tf|exec-vsa-conclusion | 0.538 (13) | 0: 0.50, 1: 0.67 |
| taxonomy|tf|exec-vsa-noconf | 0.538 (13) | 0: 0.50, 1: 0.67 |
| taxonomy|tf|exec-vsa-stressed | 0.692 (13) | 0: 0.70, 1: 0.67 |
| taxonomy|tf|exec-vsa@0.1 | 0.923 (13) | 0: 0.90, 1: 1.00 |
| taxonomy|tf|exec-vsa@0.3 | 0.846 (13) | 0: 0.80, 1: 1.00 |
| taxonomy|tf|facts | 0.538 (13) | 0: 0.50, 1: 0.67 |
| taxonomy|tf|none | 0.538 (13) | 0: 0.50, 1: 0.67 |
| taxonomy|tf|text | 0.462 (13) | 0: 0.50, 1: 0.33 |
| taxonomy|tf|text+demo | 0.538 (13) | 0: 0.60, 1: 0.33 |
| taxonomy|tf|text-retrieved | 0.615 (13) | 0: 0.60, 1: 0.67 |
| trees|mc|leaf-vsa | 1.000 (20) | 2: 1.00, 3: 1.00, 4: 1.00 |
| trees|mc|none | 0.200 (20) | 2: 0.14, 3: 0.14, 4: 0.33 |
| trees|mc|path-shuffled | 0.300 (20) | 2: 0.43, 3: 0.00, 4: 0.50 |
| trees|mc|path-sym | 1.000 (20) | 2: 1.00, 3: 1.00, 4: 1.00 |
| trees|mc|path-vsa | 1.000 (20) | 2: 1.00, 3: 1.00, 4: 1.00 |
| trees|mc|tree-bullets | 0.250 (20) | 2: 0.57, 3: 0.14, 4: 0.00 |
| trees|mc|tree-text | 0.250 (20) | 2: 0.57, 3: 0.14, 4: 0.00 |
