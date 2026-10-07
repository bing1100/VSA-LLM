**SMOKE TEST — not a result.**

# E11-M many terms — t5 new — C5 seed 1 (HuggingFaceTB/SmolLM2-135M/train)

Sizes [25]; routes ['frames', 'context', 'rag']. Frames are written for every term at once; in-context prompts carry the first N definitions up to the budget; retrieval puts BM25's top-k definitions in the prompt; the gradient route learns the definitions one after another.

## Property accuracy by number of terms learned

| condition | N=25 |
|---|---:|
| context:B2048 | 0.391 |
| frame:linker | 0.180 |
| frame:oracle | 0.184 |
| frame:typeprior | 0.195 |
| none | 0.148 |
| rag:k1 | 0.629 |

Definitions that fit the budget: context:B2048|N25: 18/25 (2035 tokens)

Retrieval recall of the term's own definition: rag:k1|N25: 1.000

## Loss after new-term mentions (passages)

| condition | N=25 |
|---|---:|
| context:B2048 | 1.8548 |
| frame:linker | 2.4975 |
| frame:oracle | 2.4259 |
| frame:typeprior | 2.4325 |
| none | 2.5351 |
| rag:k1 | 1.3774 |

Frame interference check: `{"items": 306, "max_abs_margin_change": 0.0}`.

Timings (s): `{"frames": 228.9, "context": 231.3, "rag": 81.6, "gradient": 0.0}`; total 542s.