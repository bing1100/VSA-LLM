# E12 3a — agentic self-query, feasibility pilot — C5 seed 1 (Qwen/Qwen3-1.7B-Base/lora) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 50 | 0.64 | 0.02 | 0.30 | 0.02 | 0.64 | 0.64 | 0.46 | 0.48 | 0.64 |
| twins | 20 | 0.95 | 0.00 | 0.45 | 0.00 | 0.95 | 0.95 | 0.35 | 0.50 | 0.50 |
| two_hop | 15 | 0.47 | 0.07 | 0.40 | 0.07 | 0.47 | 0.47 | 0.47 | 0.20 | 0.67 |
| reverse | 15 | 0.40 | 0.00 | 0.00 | 0.00 | 0.40 | 0.40 | 0.60 | 0.73 | 0.80 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

Twin contrast accuracy over 10 pairs: agent 0.30, no tool 0.30, fixed 0.80.
