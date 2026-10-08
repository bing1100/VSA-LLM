# E12 3a — agentic self-query, feasibility pilot — P0 seed 1 (Qwen/Qwen3-1.7B-Base/frozen) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.96 | 0.84 | 0.00 | 0.84 | 0.92 | 1.34 | 0.76 | 0.58 | 0.88 |
| twins | 20 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.55 | 0.85 |
| two_hop | 15 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 2.00 | 0.60 | 0.33 | 0.80 |
| reverse | 15 | 0.87 | 0.47 | 0.00 | 0.47 | 0.73 | 1.13 | 0.73 | 0.87 | 1.00 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

Twin contrast accuracy over 10 pairs: agent 1.00, no tool 0.50, fixed 1.00.
