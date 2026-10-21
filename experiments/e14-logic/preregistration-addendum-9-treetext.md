# E14 pre-registration — addendum 9 (2026-10-20, before any queue entry): decision trees read from prose (tree-text)

Companion to `experiments/e14-logic/preregistration.md` (LG-D §5, LG-D2 §19). Written and committed before the host runs
were queued. The author approved the test on 2026-10-20 ("Do the tree test and see whether it is worth queuing more
experiments for"). Code: `src/vsa_embed/experiments/e14_treetext.py`; items: `experiments/e14-logic/items/trees-text-v1/`.

## Question

Every LG-D / LG-D2 tree so far is synthetic and handed to the exact executor directly. The untested step that decides
whether "trees decoded from text, then used zero-shot" deserves further experiments is: **can a frozen language model read a
decision tree described in prose and emit it in the executor's structured form, accurately enough that executing the
decoded tree reproduces the gold tree's outcomes?** If reading fails on the small hosts of E14, the rest of that line is
not worth more queue time at this scale.

## Items (`e14_treetext build`, set `trees-text-v1`, seed 1)

- **Trees:** LG-D's generator (`world.random_tree`: 8 invented attributes, categorical with 3–5 values or numeric 0–100,
  tests "is v" and "at most t"; 6 outcomes "send it to the X desk"), under its own seed key `trees-text:1` and namer
  `Namer(51)`; depths 2, 3, 4 × 24 trees = **72 trees**, half with numeric attributes (odd indices, as in LG-D). The
  generator may test one attribute twice on a path or give both branches the same outcome; such trees are kept (they are
  LG-D's population).
- **Test records:** 24 per tree, every attribute drawn from its prior; gold leaf from `reference.run_tree`. The gold
  tree's majority leaf covers 51% of a tree's records on average (the ceiling of a constant reader); the first outcome
  mentioned covers 23% (`first-leaf`).
- **Prose, two styles per tree** (no language model is used to write them):
  - `templated`: a numbered pre-order step list, one fixed sentence per test, "Step k: if its A is v, <action>;
    otherwise, <action>." where an action is an outcome or "go to step j". Unambiguous; the regex reader
    (`regex_reader`) parses it exactly (leaf accuracy 1.000 on all 72 trees, recorded in the items' manifest).
  - `paraphrased`: a small grammar — steps after the first in random order; 6 sentence frames with varied connectives
    ("unless … in which case", "as long as … failing that", "when …; when …"); half the frames state the negated test
    and give the no-branch first ("when its A exceeds 40, …; when its A is 40 or less, …"); 5 phrasings per test and per
    outcome, 5 for "go to step j"; and, with probability 1/2 per step, a child test whose two branches are outcomes is
    not given its own step but inlined as a nested clause at the end of the parent's sentence ("… Otherwise it then
    comes down to whether its B is w: if that holds, …; if it fails, …"). Seeded per tree (`trees-text-para:1:<id>`).
- **Worked examples:** two trees (depth 2 categorical, depth 3 numeric) from a separate seed key and namer, rendered in
  both styles; they share no attribute or outcome name with the 72 items and are not scored.
- **Files:** `items.jsonl`, `demos.json`, `manifest.json` (counts, SHA-256 of both files, baseline scores).

## Structured form and parser

The target is the executor's own text form of a tree (`world.tree_text`): "If its <attr> is <value>:" or "If its <attr>
is at most <n>:", the yes branch indented, "Otherwise:", the no branch, leaves as "then <action>.". `parse_tree` is
tolerant: indentation, case, trailing punctuation and a few spellings ("else", "<=", a leaf without "then") are accepted;
a negated test ("If its A is not v:", "is above t", "exceeds t") is accepted and normalised to the executor's form by
swapping its two branches, so a tree that is semantically right but keeps the prose's negation is not penalised (added
after the CPU smoke showed SmolLM2-360M copying such phrasings; before any queue entry); what is parsed is the if /
otherwise / then grammar; text after the first complete tree is ignored; anything else is an unparsed output. A decoded attribute name is resolved to the item's attribute by exact match or unique containment
(so "plonk" for "plonk score" counts); an outcome is canonicalised to "send it to the <desk> desk" when a "the <desk>
desk" phrase is present. The parser round-trips the gold form exactly on all 72 trees.

## Hosts, prompt, decoding

- Hosts: **SmolLM2-360M** (`H-360`), **Qwen3-0.6B-Base** (`H-q06`), **Qwen3-1.7B-Base** (`H-q17`), frozen, bfloat16 on
  the GPU (the E14 hosts; the base checkpoints, as in every E14 run — no instruct variants are in the local cache).
- One fixed prompt per style: an instruction sentence, the two worked examples in that style (policy → decision tree,
  separated by "###"), then the item's policy and "Decision tree:". Prompts are 0.9–1.5k tokens.
- Greedy decoding, stopped at a blank line or "###". Token budget per depth group: 1.5 × the longest gold target in the
  group + 16 (depth 2: 175, depth 3: 360, depth 4: 724 tokens on SmolLM2's tokenizer), so a correct tree is never cut;
  an output that reaches the budget is recorded as truncated.
- Runs: `e14_treetext read --host <host>`, both styles, all 72 trees → `experiments/e14-logic/runs/treetext/<host>/`
  (`generations.jsonl`, `summary.json`, `report.md`).

## Metrics (per tree; means over trees; 2,000-resample percentile bootstrap over trees)

1. **parse rate**: the output parses to a tree;
2. **exact tree match**: canonical decoded tree equals the gold tree (tests and leaves);
3. **node accuracy**: over the gold tree's internal nodes, the share whose decoded node at the same yes/no path carries the
   same test (attribute, operator, value); an unparsed output scores 0;
4. **leaf accuracy (primary)**: the decoded tree executed on the tree's 24 records, the share of records whose leaf
   equals the gold tree's leaf; a record whose test cannot be evaluated (unknown attribute, non-numeric threshold) and
   an unparsed output score 0;
5. everything by style, by depth and by categorical / numeric.

Baselines (no model): `regex:templated` (the format's upper bound for the templated style; 1.0 by construction),
`first-leaf` (a reader that copies only the first outcome; 0.23), gold (1.0). The constant-reader ceiling (0.51) is noted
above for reference.

## Contrasts

- **Primary:** leaf accuracy, host × style — three hosts' means with intervals on each style; per host the paired
  (over trees) difference `templated − paraphrased`; pairwise host differences on each style (paired over trees). No
  Holm correction: the decision below is read off one pre-specified cell, not off significance tests.
- Secondary: exact match, node accuracy, parse rate; the same by depth; the truncation count.

## Predictions (written before any host output was read)

- Qwen3-1.7B, templated: leaf accuracy > 0.9 (depth 2 near 1.0; depth 4 the weakest, ≥ 0.8); exact match ≥ 0.6.
- Qwen3-1.7B, paraphrased: clearly lower, 0.6–0.8; the negated-and-swapped sentences and the inlined clauses are where the
  errors are (node accuracy drops faster than parse rate); depth 4 ≤ 0.6.
- Qwen3-0.6B: templated 0.7–0.9, paraphrased 0.4–0.65.
- SmolLM2-360M: low — templated ≤ 0.6, paraphrased ≤ 0.4; parse rate still high (the form is easy to imitate), so the
  failures are wrong tests and wrong leaves, not malformed output.
- Parse rate ≥ 0.9 on every host and style at depth ≤ 3; some truncation at depth 4 on SmolLM2-360M.
- `templated − paraphrased` positive on every host.
- Numeric trees slightly worse than categorical on the paraphrased style (the "exceeds / at most" swaps).

## Decision rule (for the author)

**Worth queuing more experiments on this line iff the best host's leaf accuracy on the paraphrased style is ≥ 0.80 with
the bootstrap interval's lower bound ≥ 0.70.** Rationale: the paraphrased style is the one that resembles text a store
would actually be built from; 0.8 of records routed correctly is the point at which a decoded tree beats the hosts'
own tree-text reading in LG-D (0.37 on clean records, §5) by a wide enough margin that a "read the tree once, execute it
exactly" pipeline would be worth testing on real policy text; below it, the errors are of the kind an executor cannot
repair. The templated style is diagnostic only: a pass there with a fail on the paraphrased style means the hosts can
imitate the format but not read the logic, and that is a "not worth more" outcome. `report` prints the verdict
(`decision.worth_queuing_more` in `report.json`); it is withheld when any run is a smoke run.

If the rule fails but Qwen3-1.7B's templated leaf accuracy is ≥ 0.9, the one follow-up I would propose (not queued by
this addendum) is a single run on a larger base host already in the cache (Qwen3-4B-Base) to see whether the paraphrased
gap closes with scale; that would need its own addendum.

## Queue and cost

- GPU lane, priority 54.4959997: `e14-treetext-SmolLM2-360M`, `e14-treetext-Qwen3-0.6B`, `e14-treetext-Qwen3-1.7B`
  (one `read` each, batch 8, `--no-resume`).
- CPU lane, priority 54.49599985: `e14-treetext-report` → `experiments/e14-logic/report/treetext/` (`report.md`,
  `report.json`).
- Commands: `experiments/e14-logic/queue-commands-addendum9-treetext.sh`.
- **Cost estimate:** per host 144 generations; at most 2 × 24 × (175 + 360 + 724) ≈ 60k generated tokens, batched 8 wide
  with 0.9–1.5k-token prompts: about 3–10 minutes per host on the RTX 3090 (longest for Qwen3-1.7B), i.e. **0.15–0.5 GPU-h
  in total**, well under 1 GPU-h. The report is seconds on the CPU.

## Smoke

A CPU-only pipeline check (SmolLM2-360M, float32, three trees: two of depth 2 and one of depth 4, both styles) runs in the
session's scratchpad (`…/scratchpad/treetext/smoke-h360-d2`, `…/smoke-h360-d4`), labelled SMOKE in its report; it checks
that generation, parsing, scoring and the report run end to end and that the token budget holds. Its numbers are not
results and are not used to adjust the predictions above, which were fixed before its output was read.

## Not changed

`preregistration.md`, the LG-D / LG-D2 item sets, `world.trees`, `world.noisy_trees`, the tier-P runs and reports.
