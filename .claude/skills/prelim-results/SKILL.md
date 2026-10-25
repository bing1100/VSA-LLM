---
name: prelim-results
description: Produce a human-readable preliminary-results digest for the VSA-LLM experiment programme — what the job queue is doing, which pre-registered reports have landed, early reads of finished runs whose official report has not run yet, and how each number sits against its pre-registered decision rule. Use this whenever the author asks "what are prelim results", "what landed", "status", "how are the experiments going", "any results yet", or asks about a specific block (E15, E13, T7-ROOD, …) before its report exists — even when they type it casually ("what r prelim results"). Also use it before writing a findings file, so the write-up starts from the same checks.
---

# Preliminary results for a human reviewer

The author checks in several times a day with "what r prelim results". They want to know, in a few minutes of
reading, where the research stands *right now*: what finished, what the numbers say against the plan, what is still
guesswork, and what decision (if any) is waiting for them. The dangers are specific: quoting a CPU estimate as if it
were the official number, moving a pre-registered bar after seeing data, or reporting a job's loss curve as a result
when its control has not run. This skill exists to make the status fast to produce and hard to over-read.

## 1. Gather (deterministic first)

Run the bundled script from the repository root with the pinned environment:

```bash
cd /home/bhux/workplace/VSA-LLM && PYTHONPATH=src /home/bhux/anaconda3/envs/vsa-repro/bin/python \
  .claude/skills/prelim-results/scripts/prelim_status.py --since 36
```

It prints: queue counts by block, running and failed jobs, the next jobs in line, every `report.md` modified in the
window, finished runs whose block has no report yet (with their final evaluation losses), and disk space. Widen
`--since` if the author has been away longer; narrow it with `--block` when they ask about one experiment.

If the script's block table has an "other" row that is large, a new experiment was queued without a label: add its
prefix to `BLOCKS` in the script before continuing, so the next check is clean.

## 2. Read what landed — against the registration, not from memory

For every new `report.md`, open it and extract three things:

1. **The primaries** with intervals and Holm-adjusted p-values, as the report prints them.
2. **The decision-rule reading** the report prints (most reports have a "Decision rule" line). Quote it verbatim.
3. **The pre-registered predictions** and whether each held. Find the pre-registration file the report names (always
   cited in its header) and score the predictions there, even when the report does not.

Do not reconstruct a number from memory of an earlier conversation; numbers in the digest come from a file read in
this pass. If two sources disagree (a findings file and the report), the report governs and the disagreement is worth
one line.

## 3. Early reads of finished runs (clearly labelled)

When a block's training jobs are done but its evaluations or report have not run, the author usually wants an early
read anyway. Three sources are cheap and honest:

- **Final evaluation losses** from `metrics.jsonl` (the script prints them). Compare only against a baseline *at the
  same number of tokens*; a 200M-token arm against a 50M baseline confounds the objective with length (E15 stage 1b
  learned this the hard way — the matched-length control had to be added afterwards).
- **Training-loss trajectories** of auxiliary objectives (e.g. `read_loss`, `kl_loss`): is the objective still
  falling at the end, or has it plateaued? That is the question "would more tokens help", and it needs no GPU.
- **Reduced CPU evaluations** on a finished checkpoint, when an evaluation CLI exists (`e15_eval read --device cpu
  --limit 60`, `e14_reading recall --limit N`). Always `CUDA_VISIBLE_DEVICES=` (the GPU belongs to the queue runner),
  a `timeout`, `--limit` so one run takes minutes, output to the session scratchpad, never to `experiments/`. A 360M
  model on CPU reads 60 terms in 3–5 minutes.

Label every such number **preliminary** and say what makes it so (one seed; 60 of 285 items; CPU; no Holm). Expect
the official number to differ: in E15 stage 1b the 60-term CPU read accuracy was 0.72 where the full split gave 0.76.

## 4. Write the digest

Structure it so a reader who skims the bold text gets the state, and a reader who reads everything can act. Order
blocks by what changed since the last check, then by what the author is waiting on.

```markdown
## <Block> — <what question it answers>, <status: N of M jobs; report landed / not yet>
| | result | pre-registered bar |          ← one table when there is a rule to compare against
**What the controls say** — 2–4 bullets; this is where the meaning lives.
**Verdict by the rule:** quote it. If a prelim number would change it, say "would read as … if it holds".
**Caveats:** seeds, partial items, CPU, unadjusted, confounds.
```

Then a short **"What this leaves"** section: what stops, what continues, and the one or two decisions that are the
author's (with the cost of each in GPU-hours and what the registered rule says). Close with the next landings and when.

Keep the author's conventions: accuracy differences positive = better; loss differences positive = worse; `*` only
for Holm-significant primaries; secondaries carry raw p-values and are called unadjusted; they/them for people.

## 5. Things that go wrong, and why they matter

- **Moving the bar.** If a prelim number falls short of a registered threshold, the honest move is to record the stage
  as registered and *register the next stage* (longer, different objective) before its data exist — never to reinterpret
  the current one. The author approves re-registration readily; they do not want a verdict quietly softened.
- **Length confounds.** Any arm trained longer than its baseline needs a matched-length control before its loss or
  QA numbers mean anything. Flag the missing control as a decision, not as a result.
- **Truncation and other harness artefacts.** If a number is implausibly bad or exactly equals a trivial baseline
  (e.g. accuracy equal to the share of items whose gold is candidate 0), suspect the harness before the model. The
  256-token prompt truncation in the probe adapter hid for a week this way.
- **Peer ownership.** Some blocks belong to another session (R9, claims.md and draft.md are workplace-9a's). Read
  their reports and report the numbers; do not edit their files, and send them anything that changes a claims row.
- **Reading another session's scratch as results.** Only `experiments/**/report/` and `runs/**/metrics.jsonl` are
  results; `/tmp/claude-*/…/scratchpad` holds smoke runs.

## 6. After the digest

If a report landed, the findings file and the status digest (`manuscript/status/<date>/`) need updating; the author
has asked for that every time so far. Offer it in one line rather than assuming, unless they already said "write
these up". Record any new decision or stage in the memory notes.
