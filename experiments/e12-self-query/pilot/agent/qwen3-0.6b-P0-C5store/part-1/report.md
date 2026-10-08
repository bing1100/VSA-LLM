# E12 3a — agentic self-query, feasibility pilot — P0 seed 1 (Qwen/Qwen3-0.6B-Base/frozen) — PILOT

Host dtype: bfloat16. Greedy decoding; at most 3 calls, 40 tokens per segment. Answers read as forced choice after `Answer:`.

| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 25 | 1.00 | 0.68 | 0.00 | 0.60 | 1.00 | 1.16 | 0.72 | 0.64 | 0.88 |
| two_hop | 10 | 1.00 | 0.70 | 0.00 | 0.50 | 1.00 | 1.40 | 0.60 | 0.40 | 0.80 |
| reverse | 15 | 1.00 | 0.67 | 0.00 | 0.67 | 1.00 | 1.00 | 0.80 | 0.80 | 0.93 |

`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other tool's name (recorded, not repaired); `relevant`: the first call is about the item.

## Demonstrations (tool)

```
Question: Bailkzhun Hub depends on ___? Options: Bairktraim Pipeline | Kailkzound Plourb Catalog
Thought: I need what Bailkzhun Hub depends on.
Action: recall[Bailkzhun Hub, depends on]
Observation: recall(Bailkzhun Hub, depends on):
- Bailkzhun Hub depends on Kailkzound Plourb Catalog. (0.48)
Thought: The recall says Kailkzound Plourb Catalog.
Answer: Kailkzound Plourb Catalog

Question: Bailkda Veinzheish Review is owned by a team that reports to ___? Options: the Kleindstou Office | the Grashsa Division | the Saindbem Pirbbam Office
Thought: First I need the team that owns Bailkda Veinzheish Review.
Action: recall[Bailkda Veinzheish Review, owned by]
Observation: recall(Bailkda Veinzheish Review, owned by):
- Bailkda Veinzheish Review is owned by the Sulkpurk Pod. (0.26)
Thought: Now I need what Sulkpurk Pod reports to.
Action: recall[Sulkpurk Pod, reports to]
Observation: recall(Sulkpurk Pod, reports to):
- Sulkpurk Pod reports to the Saindbem Pirbbam Office. (0.42)
Thought: The second recall says the Saindbem Pirbbam Office.
Answer: the Saindbem Pirbbam Office

Question: Of Bailkda Veinzheish Review and Felkzoux Klenalk Dataset, the one owned by the Rainploulk Squad is ___? Options: Bailkda Veinzheish Review | Felkzoux Klenalk Dataset
Thought: I should look up what the Rainploulk Squad owns.
Action: lookup[owned by, the Rainploulk Squad]
Observation: lookup(owned by, the Rainploulk Squad):
- Dreirkbrax Index is owned by the Rainploulk Squad. (0.50)
- Fleilk Kurbfult Playbook is owned by the Rainploulk Squad. (0.48)
- skashtrirkkirkgraish is owned by the Rainploulk Squad. (0.44)
- Flaltgon Rate is owned by the Rainploulk Squad. (0.44)
- Felkzoux Klenalk Dataset is owned by the Rainploulk Squad. (0.43)
Thought: Felkzoux Klenalk Dataset is in the lookup and Bailkda Veinzheish Review is not.
Answer: Felkzoux Klenalk Dataset
```
