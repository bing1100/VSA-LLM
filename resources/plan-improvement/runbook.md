# Runbook — resuming experiments on `bhux-tiny`

Step-by-step plan for a Claude Code session in tmux. The experiment machine with faulty RAM ([gates.md](gates.md#blocker--faulty-ram-on-the-experiment-machine-2026-09-30)) has been replaced by `bhux-tiny` (Ryzen 5 8600G, 60 GiB RAM, RTX 3090). Work proceeds in this order: re-run the committed results and check they reproduce, then lift the blocker, then resume [queue.md](queue.md).

**How to run it.**

```bash
tmux new -s vsa            # later: tmux attach -t vsa
cd ~/workplace/VSA-LLM
claude
```

Then tell the session: *"Follow resources/plan-improvement/runbook.md from the first unchecked step."* The session ticks each box (`- [x]`) when a step is done and writes the outcome on the line below it, so a later session can pick up where this one stopped. Each step ends with a check that decides whether to continue.

**Conventions for every step.**

- Python: `PY=~/anaconda3/envs/vsa-repro/bin/python`. This env matches the committed manifests: Python 3.12.4, torch 2.11.0+cu128, transformers 4.54.0, numpy 2.2.6, nltk 3.8.1, PyYAML 6.0.2, safetensors 0.5.3. scipy and torchao were not recorded in the manifests; record the installed versions in gates.md when a result depends on them.
- Data root: `~/data/vsa-llm/` (the configs point there). WordNet is in `~/nltk_data`.
- Reproductions run at the commit recorded in the committed `manifest.json`. Create a worktree with `git worktree add --detach ~/workplace/vsa-worktrees/<sha> <sha>`, run with `PYTHONPATH=~/workplace/vsa-worktrees/<sha>/src` from inside the worktree, and write outputs to `~/workplace/vsa-repro-runs/<name>`, outside the repo so the tree stays clean.
- Compare a re-run with `python scripts/compare_runs.py <committed run dir> <re-run dir> [--tolerance T]`. CPU runs use `--tolerance 1e-12`: summaries can differ in the last digit when numpy differs. GPU runs use `--tolerance 1e-3` unless a step says otherwise.
- Long jobs run detached (`nohup … > log 2>&1 &`, or the B11 queue `$PY -m vsa_embed.jobqueue add/run`). Don't hold the session on them; check the log.
- Commit after each step that changes the repo, ending messages with the attribution line already used in this repo's history.

---

## Step 0 — environment and machine checks (done 2026-10-02)

- [x] RAM screen: `python scripts/memcheck.py 16 4` → **0 bad words** in 16 GiB × 4 passes (the old machine had 204 at 6 GiB).
- [x] `vsa-repro` env created; `pip install -e .` (no deps), plus `scipy`, `pytest` and `pyarrow` (C3 reads parquet). WordNet downloaded. `pytest` passes; 15 tests are skipped because the GPT-2 tokenizer isn't cached yet (it is downloaded in Step 2).
- [x] Before any long run, re-run `$PY scripts/memcheck.py 20 8` and record the total here. Continue only if it is 0.
  2026-10-02: ran `memcheck.py 16 8` instead (20 GiB needs ≈ 60 GiB with the index and temporaries, the whole of this machine's RAM): **0 bad words** in 16 GiB × 8 passes.

## Step 1 — E0 reproduction (CPU, single-threaded)

- [x] **D0.2 parent-sync** (`runs/d02-parent-sync`, commit `9986d72`, `--parts d02`, config `e0-parent-routing.yaml`): metrics.csv **0 of 576 cells differ**. summary.json matches at 1e-12; one mean differs in the last digit (the re-run used numpy 2.4).
- [x] **E0 development** (2026-10-02, output in `~/workplace/vsa-repro-runs/e0-development`): **MATCH at 1e-12**, metrics.csv 8,100 cells and summary.json 154 cells; G1 passes again. At tolerance 0, three `ari_heldout` cells differ in the 17th digit. That run used numpy 2.4 rather than the pinned 2.2.6, so the metric's float summation explains it. Command used (this commit has no `--parts`, so all parts run in one process):

  ```bash
  git worktree add --detach ~/workplace/vsa-worktrees/2cc5d0b 2cc5d0b
  cd ~/workplace/vsa-worktrees/2cc5d0b
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nohup $PY src/vsa_embed/experiments/e0_identifiability.py \
      --config experiments/e0-synthetic-identifiability/e0.yaml \
      --output ~/workplace/vsa-repro-runs/e0-development > ~/workplace/vsa-repro-runs/e0-development.log 2>&1 &
  ```

  Check: `python scripts/compare_runs.py experiments/e0-synthetic-identifiability/runs/e0-development ~/workplace/vsa-repro-runs/e0-development --tolerance 1e-12` prints `MATCH`.
- [x] **D0.2 parent-routing** (`runs/d02-parent-routing` and `runs/d02-parent-routing-v2`): read each `manifest.json` for its sha, argv and config, run them the same way, and compare.
  2026-10-02: both **MATCH at 1e-12** (metrics.csv 0 of 576 cells, summary.json 0 of 115), at `bf5f535` and `01e8090` respectively (v2's manifest records a dirty tree; it reproduces anyway).
- If any comparison prints `DIFFER` beyond rounding: stop, write the differing cells into gates.md, and tell the author. G1 rests on these numbers.

## Step 2 — model and probe data downloads

- [x] Models into the HF cache (2026-10-02, all four):

  ```bash
  for m in gpt2 HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
      ~/anaconda3/envs/vsa-repro/bin/huggingface-cli download $m; done
  ```

  Then `$PY -m pytest -q`: the 15 GPT-2 tests now run and should pass.
- [x] Probe data under `~/data/vsa-llm/probes/` (2026-10-02; every count matches; sources, sha256 and checks in `~/data/vsa-llm/DATA_SOURCES.md` and `SHA256SUMS`; BLESS from the Wayback copy of the GEMS zip, HyperLex from github.com/cambridgeltl/hyperlex `ccb41619`; MeSH `desc2026.gz` and both FineWeb shards fetched too, shards' sha256 match `c3.yaml`). The loaders in `src/vsa_embed/evaluation/probes.py` expect this layout; find each official source and check the counts before using it:

  | Path | Source | Check |
  |---|---|---|
  | `lambada/data/lambada_test_en.jsonl` | HF dataset `EleutherAI/lambada_openai` (the `data/` file) | 5,153 lines |
  | `wic/{train,dev}/{split}.data.txt`, `{split}.gold.txt` | WiC dataset zip (Pilehvar & Camacho-Collados) | dev 638, train 5,428 |
  | `card660.tsv` | CARD-660 (Pilehvar et al. 2018) | 660 pairs |
  | `rw/rw/rw.txt` | Stanford Rare Words (Luong et al. 2013) | 2,034 pairs |
  | `wsd/WSD_Evaluation_Framework/` | Raganato et al. 2017 (SemCor + ALL), needed for E1 | — |

- [x] `pip install torchao` into `vsa-repro` (needed for B8 PTQ) and record the version here: **torchao 0.18.0** (its Hopper-only `_C_mxfp8` / `_C_cutlass_90a` extensions fail to load on the 3090; harmless).

## Step 3 — B6 and B8 reproduction (GPU)

- [x] **B6 host memory** (`b6-host-memory/runs/3090-v1`, commit `2ad45e5`): `$PY src/vsa_embed/experiments/host_memory.py --output ~/workplace/vsa-repro-runs/b6-3090`. This is a hardware measurement, so compare `report.md` against the committed one by eye. Peak memory per host × micro-batch within ±5% and the same pass/fail against 22 GB counts as reproduced.
  2026-10-02: **reproduced** — peak memory identical in all 19 rows (the SmolLM2-360M micro-batch-16 row is `nan` in both), tokens/s within 1%. The recorded commit `2ad45e5` (dirty tree) crashes with transformers 4.54 (`GPT2LMHeadModel(..., attn_implementation=…)`); the fix landed later, so this ran at HEAD `724219a`.
- [ ] **B8 probes** (`b8-probe-validation/runs/v1`, commit `e2078a4`): `$PY src/vsa_embed/experiments/b8_validation.py --output ~/workplace/vsa-repro-runs/b8-v1`. Compare the probe columns only; v1's quantization columns are superseded (see its NOTE.md). Accept LAMBADA, WiC and Spearman within ±0.005.
- [ ] **B8 PTQ** (`runs/v1-ptq`, commit `8e83d64`, `--quantization-only`): compare PPL bf16, INT8 and INT4 within ±1%.

## Step 4 — C5 judge calibration through this Claude Code session

The judge no longer shells out to `claude -p`. `calibrate` writes one request file per (item, call); this session answers them with fresh subagents and writes the verdicts to files; re-running `calibrate` validates and caches the verdicts and writes the results. Requests carry only the prompt, the schema and the response path: no item id, gold label or system name.

- [x] **4a. Post requests.**

  ```bash
  $PY -m vsa_embed.judge_protocol calibrate --output experiments/c5-judge-calibration/v2
  ```

  Expected: `90 requests outstanding … in experiments/c5-judge-calibration/v2/exchange/requests` and exit code 3. That is 30 WordNet items (seed 0, the same items as v1) × 3 calls.
- [x] **4b. Judge each request in a fresh subagent.** Calls must stay independent, as separate `claude -p` calls were in v1, so:
  - Use **one new subagent per request file** (Agent tool, `model: opus`, which is the pinned `claude-opus-5-5`), about 10 in parallel per message.
  - Never answer requests in this main session. Never give one subagent two requests, and never pass on another call's answer.
  - Subagent prompt, verbatim apart from the path:

    > Read the JSON file `<request path>`. Act as the judge: answer its `prompt` field on its own merits. Then write ONLY a JSON object that conforms to its `schema` field (for example `{"score": 2, "reason": "…"}`) to the path in its `response_path` field. Do not read or list any other file. Reply with the object you wrote.

  - Don't open `results.json`, `summary.json` or `cache/` in `v2/` while judging is in progress, and keep subagents away from `experiments/c5-judge-calibration/v1/`.
- [x] **4c. Ingest.** Re-run the 4a command. Exit 0 prints the summary. Exit 3 lists what is still outstanding: each remaining request's `last_error` says whether it is `pending` or why its response was rejected (`invalid response: …`). Re-judge those with new subagents (4b) and re-run until it exits 0.
  2026-10-02: 90 subagents, all responses valid on the first pass; exit 0.
- [x] **4d. Check against v1** (accuracy 1.0, i.e. 30/30; Fleiss κ 0.81). Accuracy is binary: majority ≥ 2 counts as related, gold 3 is related and gold 0 is not. Pass if accuracy ≥ 0.90 (≥ 27/30) and κ ≥ 0.6. Otherwise, record the disagreeing items in gates.md and tell the author before D5.3 or D7.1 use the judge.
  2026-10-02: **PASS — accuracy 1.0 (30/30), Fleiss κ 0.926** (v1: 1.0, 0.812).
- [x] **4e. Commit** `v2/results.json`, `summary.json`, `manifest.json`, `resolved_config.yaml` and `cache/`. The cache lets the study be replayed and lets the later clinician study rate identical items. Don't commit `exchange/`: answered requests are deleted and the responses duplicate the cache. In gates.md, note the harness change: v1 used lean `claude -p --tools ""`, while v2 uses Claude Code subagents with the default system prompt. Agreement numbers are therefore comparable, but individual verdicts are not.
  2026-10-02: committed in `724219a`; `exchange/` is git-ignored.

Other judge studies (`edge_explanation`, `split_card`, `authored_edge` in D5.3 and D7.1) use the same loop: build a `JudgeClient(..., exchange_dir=<run>/exchange)`, run the study, answer the requests as in 4b, and re-run.

## Step 5 — lift the blocker

- [ ] When Steps 1, 3 and 4 pass, add a dated section to gates.md under the BLOCKER heading, "Resolved: new machine `bhux-tiny`". List memcheck, each comparison and its outcome, and the env versions.
- [ ] In queue.md, remove the BLOCKED banner and set C3, D1, D2 and D3 back to `todo`. Commit.

## Step 6 — resume the queue (in [queue.md](queue.md) order)

Partial E2/E3/C3 outputs committed from the old machine (`e2…/runs/v1`, `e3…/runs/hrr-v1`, `c3…/runs/v1`) are untrusted. Use new run IDs and never append to those directories. `prepare_output_dir` refuses non-empty directories anyway.

- [ ] **C3 corpus.** Download FineWeb-Edu `sample/10BT` shards `000_00000.parquet` and `002_00000.parquet` into `~/data/vsa-llm/fineweb-edu/sample/10BT/`. The run refuses shards whose sha256 doesn't match the prefixes in `c3.yaml`. Then run `$PY -m vsa_embed.experiments.c3_corpus --config experiments/c3-general-corpus/c3.yaml --output experiments/c3-general-corpus/runs/v2`, detached. It reuses completed stages if interrupted. Check that the manifest records 0 skipped/panicked documents; the old machine's tokenizer panics were RAM corruption.
- [ ] **D1 E1** `e1_contextual --config experiments/e1-contextual-composition/e1-gpt2.yaml --output …/runs/v1` (needs C3 and the WSD framework).
- [ ] **D2 E2** `e2_frontier --config experiments/e2-mapping-operator-frontier/e2.yaml --output …/runs/v2`. Needs MeSH `desc2026.gz` in `~/data/vsa-llm/mesh/`.
- [ ] **D3 E3** `e3_developmental --config experiments/e3-developmental-wordnet/e3.yaml --output …/runs/v2`. M3 default: `route_unobserved: parent`, `sync_parent: true` (G1).
- [ ] GPU jobs go through the B11 queue, one at a time: `$PY -m vsa_embed.jobqueue add --name <id> --priority <p> -- <command>`, then `$PY -m vsa_embed.jobqueue run` in its own tmux window.
- [ ] Then continue with queue.md from B10. Apply the pre-registered gate rules in tasks.md and record decisions in gates.md, as before. C2 stays blocked until the author supplies SNOMED CT, UMLS and MIMIC files. G6 escalation waits for the author.
