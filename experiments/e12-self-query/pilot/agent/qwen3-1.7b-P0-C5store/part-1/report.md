# E12 3a — agentic self-query, feasibility pilot — P0 seed 1 (Qwen/Qwen3-1.7B-Base/frozen) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 17 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.82 | 0.59 | 0.41 | 0.76 |
| twins | 3 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 |
| two_hop | 14 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 2.00 | 0.57 | 0.36 | 0.79 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

Twin contrast accuracy over 1 pairs: agent 1.00, no tool 0.00, fixed 1.00.

## Demonstrations (tool)

```
Question: Bailkzhun Hub depends on ___? Options: Bairktraim Pipeline | Kailkzound Plourb Catalog
Thought: I need what Bailkzhun Hub depends on.
Action: recall[Bailkzhun Hub, depends on]
Observation: recall(Bailkzhun Hub, depends on):
- Bailkzhun Hub depends on Kailkzound Plourb Catalog. (0.50)
Thought: The recall says Kailkzound Plourb Catalog.
Answer: Kailkzound Plourb Catalog

Question: Bailkda Veinzheish Review is owned by a team that reports to ___? Options: the Kleindstou Office | the Grashsa Division | the Saindbem Pirbbam Office
Thought: First I need the team that owns Bailkda Veinzheish Review.
Action: recall[Bailkda Veinzheish Review, owned by]
Observation: recall(Bailkda Veinzheish Review, owned by):
- Bailkda Veinzheish Review is owned by the Sulkpurk Pod. (0.25)
Thought: Now I need what Sulkpurk Pod reports to.
Action: recall[Sulkpurk Pod, reports to]
Observation: recall(Sulkpurk Pod, reports to):
- Sulkpurk Pod reports to the Saindbem Pirbbam Office. (0.36)
Thought: The second recall says the Saindbem Pirbbam Office.
Answer: the Saindbem Pirbbam Office

Question: Of Bailkda Veinzheish Review and Torkgain Extract, the one owned by the Rainploulk Squad is ___? Options: Bailkda Veinzheish Review | Torkgain Extract
Thought: I should look up what the Rainploulk Squad owns.
Action: lookup[owned by, the Rainploulk Squad]
Observation: lookup(owned by, the Rainploulk Squad):
- Dreirkbrax Index is owned by the Rainploulk Squad. (0.48)
- Flaltgon Rate is owned by the Rainploulk Squad. (0.44)
- Fleilk Kurbfult Playbook is owned by the Rainploulk Squad. (0.38)
- Torkgain Extract is owned by the Rainploulk Squad. (0.38)
- doulkzaixpouxkeirb is owned by the Rainploulk Squad. (0.38)
Thought: Torkgain Extract is in the lookup and Bailkda Veinzheish Review is not.
Answer: Torkgain Extract
```
