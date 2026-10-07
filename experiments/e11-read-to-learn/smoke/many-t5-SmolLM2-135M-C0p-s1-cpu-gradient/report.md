**SMOKE TEST — not a result.**

# E11-M many terms — t5 new — C0p seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Sizes [25]; routes ['frames', 'gradient']. Frames are written for every term at once; in-context prompts carry the first N definitions up to the budget; retrieval puts BM25's top-k definitions in the prompt; the gradient route learns the definitions one after another.

## Property accuracy by number of terms learned

| condition | N=25 |
|---|---:|
| gradient | 0.227 |
| none | 0.156 |

## Loss after new-term mentions (passages)

| condition | N=25 |
|---|---:|
| gradient | 3.3587 |
| none | 2.3987 |

Gradient route, property accuracy of the first terms as more are learned: N=25: 0.227

Timings (s): `{"frames": 40.7, "context": 0.0, "rag": 0.0, "gradient": 118.5}`; total 160s.