# E14 RD-13 — retrieved text for t4 heldout: what it states

332 of 341 concepts have a Wikipedia page (331 plausible); text cut to 300 words (mean 259.901 words; whole extract mean 1435.741).

## Stated rate (share of gold edges whose partner the finder found)

| style | concepts | with a stated edge | gold edges | stated |
|---|---:|---:|---:|---:|
| chebi | 332 | 240 | 2700 | 0.116 |
| wiki | 332 | 164 | 2700 | 0.087 |
| wiki-full | 332 | 199 | 2700 | 0.120 |

| relation | chebi | wiki | wiki-full |
|---|---:|---:|---:|
| branch | 0.000 | 0.000 | 0.000 |
| charge | 0.000 | 0.000 | 0.000 |
| contains_element | 0.000 | 0.000 | 0.000 |
| has_functional_parent | 0.531 | 0.388 | 0.469 |
| has_parent_hydride | 0.200 | 0.000 | 0.200 |
| has_part | 0.174 | 0.174 | 0.174 |
| has_role | 0.050 | 0.090 | 0.150 |
| is_a | 0.346 | 0.184 | 0.236 |
| is_conjugate_acid_of | 0.054 | 0.125 | 0.179 |
| is_conjugate_base_of | 0.128 | 0.205 | 0.256 |
| is_enantiomer_of | 0.200 | 0.400 | 0.467 |
| is_substituent_group_from | 1.000 | 0.500 | 0.500 |
| is_tautomer_of | 0.444 | 0.278 | 0.444 |

## Readers (micro precision / recall / F1 against the gold frame)

| style | reader | frames | empty | edges | precision | recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| chebi | oracle | 332 | 0 | 8.13 | 1.000 | 1.000 | 1.000 |
| chebi | stated | 332 | 92 | 0.94 | 1.000 | 0.116 | 0.208 |
| chebi | typeprior | 332 | 20 | 2.50 | 0.313 | 0.096 | 0.147 |
| wiki | oracle | 332 | 0 | 8.13 | 1.000 | 1.000 | 1.000 |
| wiki | stated | 332 | 168 | 0.70 | 1.000 | 0.087 | 0.160 |
| wiki | typeprior | 332 | 2 | 10.71 | 0.051 | 0.067 | 0.058 |
| wiki-full | oracle | 332 | 0 | 8.13 | 1.000 | 1.000 | 1.000 |
| wiki-full | stated | 332 | 133 | 0.98 | 1.000 | 0.120 | 0.215 |
