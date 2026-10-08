# E12 3a — agentic self-query, feasibility pilot — P0 seed 1 (Qwen/Qwen3-0.6B-Base/frozen) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 50 | 1.00 | 0.82 | 0.00 | 0.76 | 0.88 | 1.38 | 0.72 | 0.58 | 0.92 |
| twins | 20 | 1.00 | 0.95 | 0.00 | 0.95 | 0.70 | 1.60 | 0.80 | 0.55 | 0.95 |
| two_hop | 15 | 1.00 | 0.80 | 0.00 | 0.60 | 1.00 | 1.47 | 0.53 | 0.40 | 0.87 |
| reverse | 15 | 1.00 | 0.67 | 0.00 | 0.67 | 1.00 | 1.00 | 0.80 | 0.80 | 0.93 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

Twin contrast accuracy over 10 pairs: agent 1.00, no tool 0.60, fixed 0.90.
