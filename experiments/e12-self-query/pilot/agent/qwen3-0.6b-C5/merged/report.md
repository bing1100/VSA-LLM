# E12 3a — agentic self-query, feasibility pilot — C5 seed 1 (Qwen/Qwen3-0.6B-Base/lora) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 50 | 1.00 | 0.52 | 0.24 | 0.44 | 0.52 | 1.00 | 0.60 | 0.52 | 0.62 |
| twins | 20 | 1.00 | 0.40 | 0.55 | 0.25 | 0.10 | 1.00 | 0.55 | 0.45 | 0.55 |
| two_hop | 15 | 1.00 | 0.60 | 0.00 | 0.53 | 0.93 | 1.00 | 0.33 | 0.20 | 0.40 |
| reverse | 15 | 1.00 | 0.60 | 0.07 | 0.60 | 0.67 | 1.00 | 0.93 | 0.93 | 0.93 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

Twin contrast accuracy over 10 pairs: agent 0.40, no tool 0.50, fixed 0.80.
