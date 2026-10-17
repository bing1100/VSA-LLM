# E14 RD-11 — multi-round active reading with relation discovery (t4, 160 definitions, style `chebi`)

Hidden at the start: has_functional_parent, is_conjugate_acid_of, is_conjugate_base_of, has_part. Teacher $5.82 (98 calls, 0 cached).

| round | relations | facts/def (single → consolidated) | NEW share | unresolved | visible P / R (single) | visible P / R (consolidated) | gold recall, all relations | ceiling | accepted next |
|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 0 | 6 | 1.90 → 1.71 | 0.447 | 0.237 | 0.743 / 0.180 | 0.745 / 0.177 | 0.155 | 0.232 | has_functional_parent (26), found_in_organism (10), is_metabolite_of (9), has_part (22) |
| 1 | 10 | 2.04 → 1.82 | 0.098 | 0.252 | 0.516 / 0.189 | 0.529 / 0.182 | 0.176 | 0.232 | — |
| 2 | 10 | 1.98 → 1.83 | 0.082 | 0.256 | 0.509 / 0.182 | 0.527 / 0.182 | 0.176 | 0.232 | — |

Hidden gold relations — best-matching discovered relation (edge Jaccard) by round:

- has_functional_parent: r0: — (0.00); r1: is_metabolite_of (0.00); r2: is_metabolite_of (0.00)
- is_conjugate_acid_of: r0: — (0.00); r1: is_metabolite_of (0.00); r2: is_metabolite_of (0.00)
- is_conjugate_base_of: r0: — (0.00); r1: is_metabolite_of (0.00); r2: is_metabolite_of (0.00)
- has_part: r0: — (0.00); r1: is_metabolite_of (0.00); r2: is_metabolite_of (0.00)

## Re-analysis (discovered relations = every relation outside the round-0 visible set)

| round | visible P / R | gold recall, all relations | discovered relations (edges) |
|---:|---|---:|---|
| 0 | 0.745 / 0.180 | 0.157 |  |
| 1 | 0.706 / 0.185 | 0.162 | has_functional_parent (34), has_part (17), is_metabolite_of (5) |
| 2 | 0.720 / 0.185 | 0.162 | has_functional_parent (39), has_part (16), is_metabolite_of (6) |

Hidden gold relations — best discovered relation by round (Jaccard; precision; recall):

- has_functional_parent (27 gold edges): r0: — (J 0.00; P 0.00; R 0.00); r1: has_functional_parent (J 0.17; P 0.26; R 0.33); r2: has_functional_parent (J 0.16; P 0.23; R 0.33)
- is_conjugate_acid_of (25 gold edges): r0: — (J 0.00; P 0.00; R 0.00); r1: has_functional_parent (J 0.00; P 0.00; R 0.00); r2: has_functional_parent (J 0.00; P 0.00; R 0.00)
- is_conjugate_base_of (18 gold edges): r0: — (J 0.00; P 0.00; R 0.00); r1: has_functional_parent (J 0.00; P 0.00; R 0.00); r2: has_functional_parent (J 0.00; P 0.00; R 0.00)
- has_part (15 gold edges): r0: — (J 0.00; P 0.00; R 0.00); r1: has_part (J 0.07; P 0.12; R 0.13); r2: has_part (J 0.07; P 0.12; R 0.13)
