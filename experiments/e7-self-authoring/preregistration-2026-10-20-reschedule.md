# E7: reschedule, the operational G3 gate, and the D7.2/D7.3 plan (2026-10-20)

**Status:** pre-registration. Written on 2026-10-20, before any D7.1 output exists. No E7 job has run. The CPU smokes
in §7 check plumbing only and change nothing below. Later changes go in §8 with their date and reason.

**Origin:** author, 2026-10-20. Digest item 5.4 asked to "confirm retirement or reschedule". The author answered
"approve all … run them and queue". E7 is rescheduled, with no change to its science. Decision 63 keeps E7 as one
proposal source of the toolkit's *learn* step: the authoring cards that D7.2 writes (`round1/s*/cards-*.jsonl`) are an
input `vsa_embed.learn` already lists.

**Code (E7 only):**
- new `vsa_embed.experiments.e7_g3`: the G3 check (§2);
- `e7_plan`: fractional `--priority`, and `--conditions` for the teacher-only mode;
- `e7_authoring`: `--max-usd` hard caps (E5's `SpendCap` ledger) and an open-licence guard on the Claude teacher and
  judge;
- test `tests/test_e7_g3.py`;
- scripts `requeue-d71.sh` and `queue-commands-d72-d73.sh`. Neither has been run.

## 1. Reschedule (D7.1)

- **The 9 pending jobs:** `.jobs/e7-general-v1-{discover,author,direct}-{SmolLM2-360M,SmolLM2-135M,Qwen2.5-0.5B}`, at
  priority 60 since 2 Oct.
- **They are current.** The E7 modules have not changed since `376b6a4` (3 Oct). The dependencies that did change
  (`compose`, `span_channel`, `data/corpus`, `training/lm`, `e4_plan`) changed in additive ways. The E7 tests pass (53
  + 9 new). A CPU smoke of each command type, run against today's code and the real 2 Oct data, passed (§7). So only
  `priority` changes: `requeue-d71.sh` sets it to `PRIO_D71` (a placeholder for the author) on the jobs still
  `pending`.
- **Order:** the jobs keep their `created` stamps, so they run in plan order. The SmolLM2-360M discovery runs first,
  because the first author job runs `link` on its output.
- **Follow-ups** (`queue-commands-d72-d73.sh followups`, CPU): Hearst, random frames, the Claude teacher
  (`--max-usd`), `quality`, judge items, then G3. The Claude judge (`… judge`, `--max-usd`) may finish later, because it
  is informational for G3.

## 2. G3, operational

**What the plan says.** G3 is E4's core-claim gate, held at 125M × 500M tokens with 3 seeds (tasks.md G3;
experiments.md, timeline). Decision 54 closed it ("the full E4 core stays unqueued"), so it was never held. For E7 it
had two roles:
- it was to supply two D7.2 settings: the host mode and the channel condition, "the G3 best" (decision 30);
- the stop rules say "E7 and E8 run whatever G3 shows".

**No D7.1 number enters the planned G3.** The rule below is new. It is fixed today, before any D7.1 output, and gates
on what D7.2 needs. Its thresholds are new, except where cited.

| Item | Read from | Passes when |
|---|---|---|
| complete | the run folder | all 3 discovery files, `authoring_set.json`, proposals of the 3 hosts, the 3 `direct-*`, `hearst` and `random`, and `quality/quality.json` exist; `quality.json` is newer than every proposals file |
| power | `authoring_set.json`, `quality.json` | ≥ **100** candidates with ≥ 6 validation occurrences (`verification.min_validation`, decision 33; 100 = 10% of the 1,000-candidate set) **and** ≥ **30** masked gold entries in the authoring set |
| self (SmolLM2-360M) | `quality.json`, proposals | Wilson 95% lower bound of edge precision on masked gold concepts **>** the Wilson upper bound for the degree-matched random frames, on ≥ **30** edges; **and** ≥ **10%** of the authoring-set candidates keep ≥ 1 edge at `min_share` 0.3 (the 10% host-read gate of E9 addendum 2026-10-15 §7) |
| teacher (Claude) | the same | the same test for `proposals/teacher.json` |
| judge | `judge/results.json` | informational: calibration accuracy ≥ 0.8 and Fleiss κ ≥ 0.4 make the plausibility numbers *reliable*; otherwise they are *inconclusive* (tasks.md risk row). Never gates. |

**Verdict** (`python -m vsa_embed.experiments.e7_g3 --run …`; exit 0 PASS / 1 FAIL / 2 INCOMPLETE; written to
`runs/general-v1/gate/g3-d72.json`):
- **PASS, full:** complete, power and self pass. D7.2 runs every condition. The teacher arm runs when
  `proposals/teacher.json` exists. D7.3 is planned.
- **PASS, teacher-only:** complete, power and teacher pass but self fails. This is the tasks.md risk row "E7 reduces
  to teacher → host". D7.2 runs base and verification (self and teacher) and the round-1 conditions gold, self,
  selfnv, teacher and cm. The primary contrast becomes teacher − cm. D7.3 is not planned: its source ontology would be
  self's. The nearest test is C5teach, which awaits the author's approval of about USD 73.
- **FAIL:** power fails, or neither self nor teacher passes. Neither D7.2 nor D7.3 is queued. D7.1 is reported as E7's
  result: H-G.3, plus the authoring quality per host.

**D7.3 gate** (`--stage d73`, after D7.2's `verify-SmolLM2-360M-s1` and `round1-self-s1`): it needs the d72 verdict
PASS in mode full. Self's seed-1 verification must also accept edges for ≥ **30** candidates. With fewer, the authored
ontology is almost the visible one, and E7.3 cannot test authoring.

## 3. D7.2 / D7.3 if G3 passes

### 3.1 Settings

`queue-commands-d72-d73.sh d72` writes these to `round.json` (`update_round_settings`) before queueing:
- `host_mode: lora`, per decision 30. LoRA lets the text controls change the host, whereas a frozen host would favour
  the channel arms.
- `channel_condition: C5`, `operator: hrr` stand in for "the G3 best". No arm beat C5 with a CI excluding 0 in
  decision 54's screen, where all operators tied. C5 is the arm used on pretrained hosts in E9, E13 and E14.
- `config_overrides: {model: {lora_rank: 64}, channel: {gate_bias: 0.0}}`: the LoRA arm of the E9 engagement check,
  with gate bias 0 (decision 36). Otherwise E7 would run on the trainer's defaults (rank 16, gate bias −2), the
  regime the S0 pilot found "barely moved the loss".
- `cross.config_overrides: {train: {micro_batch: 32, grad_accum: 1, lr: 0.002, warmup_tokens: 10000000}}`: the frozen
  50M recipe (gates.md D4.0: 32,768 tokens per step, peak lr 2e-3, cosine to 0.1×), with warmup 10M as in scale-v1.
  The E7.3 code still uses the pre-D4.0 `SIZES["50M"]` recipe (262k tokens per step, lr 1e-3), which D4.0 found
  under-trained. The override fixes that.

Everything else is as written on 2 Oct (`round.json`; decisions 31–34): 50M base tokens, 25M round tokens, the
verification settings, cm's FLOPs accounting, and the text-control budgets.

### 3.2 Jobs (`e7_plan --stage d72`, priority `PRIO_D72`, in plan order)

| Mode | Jobs |
|---|---|
| full, with teacher | 3 generation (entigraph, spa, notes) + 3 base + 9 verification (self, teacher, notes × 3 seeds) + 30 round-1 (10 conditions × 3) = **45** |
| full, no teacher | 3 + 3 + 6 + 27 = **39** |
| teacher-only | 3 base + 6 verification + 15 round-1 (gold, self, selfnv, teacher, cm × 3) = **24** |
| D7.3 (`--stage d73`, `PRIO_D73`, after the CPU step `cross-prepare`) | none / curated / authored × 3 seeds = **9** |

**Endpoints and analysis** are unchanged (experiments.md E7, G5; `e7_report`):
- **primary:** `ref_authored_new_self` and `ref_masked`, self against cm and against random, with a paired window
  bootstrap pooled over seeds and Holm correction;
- **locality:** `ref_unlinked` within 0.5% of cm;
- verification against selfnv on precision.

In teacher-only mode the same report reads teacher − cm as the primary contrast.

**Interim look.** Seed 1 finishes first. Its gold − cm gap is reported descriptively. It has no stopping force.

## 4. Predictions (updated)

The plan's E7 predictions were written before the evidence below.

- **D7.1.**
  - **Self:** SmolLM2-360M edge precision on masked gold concepts ≤ 0.10, and its interval overlaps random frames'.
    So the G3 *self* item fails (probability about 0.7). The evidence: the 360M host added 0 facts from 4 MeSH notes;
    on ChEBI definitions the host reader recovered 8.0% of gold edges against 9.7% for a type prior (RD-1a); and the
    host's own relation choices were worse than a type-frequency table (E11-1).
  - **Teacher:** precision clearly above random but below its gloss agreement (0.78 on all edges, 0.59 on pointer
    edges), because it reads 400-character running-text excerpts, not glosses. Expected range 0.3–0.6. The *teacher*
    item passes.
  - **Most likely G3 verdict:** PASS, teacher-only.
  - **Hearst:** precision possibly high on is_a, but recall well below the teacher's.
  - **Direct (OLLM-style):** at or below the host's own authoring.
  - **Power:** likely passes, but masked entries in the authoring set are uncertain (expected 30–150). The excess
    surprisal screen favours rare jargon over WordNet words the hosts already know (contamination caveat, 05.1).
- **D7.2.** On WordNet general text with SmolLM2-360M, the E9 engagement check found C5 equal to C0′ to four decimals
  in every stratum, under both LoRA-64 and full fine-tuning (execution.md, E9 engagement check, 2026-10-02).
  - **gold − cm on `ref_masked`:** ≈ 0 (|Δ| < 0.2%).
  - **self − cm and teacher − cm:** ≈ 0. Any gain should sit on `ref_authored_new_*`, the high-surprisal terms the
    host lacks, where E9 saw C5 help (T4 unseen, long terms).
  - **Locality:** passes.
  - **G5:** predicted to fail. The refutation reading then applies: "self ≈ compute-matched".
- **D7.3.** At 50M from scratch the channel lowered loss on all text (decision 65: C5 − C0 −0.44%*). So:
  - curated − none and authored − none: both negative on `all`;
  - authored − curated: ≈ 0 on `all` and `unlinked`, and worse on `ref_masked`, because the authored ontology lacks
    the masked frames.

## 5. Compute and Claude spend

| Block | GPU-h | Basis |
|---|---:|---|
| D7.1, 9 jobs | ≈ 1.1 | plan (`plan/d71.json`). These jobs have never run on the GPU; their generation rates are assumptions. `link` adds about 1 CPU-minute inside the first author job. |
| D7.2 full + teacher | ≈ 44 (34–50) | 1.31B training tokens at the measured SmolLM2-360M LoRA-64 rate (e9-check: median 10.8k train tokens/s; 9.6k effective with evaluations), plus evaluations (≈ 4 h) and verification and generation (1.5 h). The plan's 24 h assumed frozen-host speed (18.5k tokens/s). |
| D7.2 teacher-only | ≈ 21 (16–24) | 0.61B training tokens, same basis |
| D7.3 | ≈ 11.5 | 9 × 300M tokens at 50M. Measured: opscreen 0.42 h per 100M tokens, scale-v1 2.06 h per 500M. Plus `cross-prepare`, 1–2 CPU-hours. |
| **E7 total** | **≈ 57** (full) / ≈ 23 (teacher-only, no D7.3) | the plan committed ≈ 40 |

**Claude** (open-licence text only, §6; hard caps through `--max-usd`):
- **teacher:** about 125 calls; cap USD 15 (`MAX_USD_TEACHER`). Estimate USD 8–13, from the E4 pilot's USD 0.073 per
  call.
- **judge:** up to 390 items × 3 calls; cap USD 50 (`MAX_USD_JUDGE`). Estimate about USD 46 at E5's USD 0.039 per
  call. It runs sequentially and may take many hours.

A run cut by its cap writes no teacher proposals, or leaves judge items ungraded. A re-run with a raised cap resumes
from the cache.

## 6. Data and licences

- **The Claude teacher and judge:**
  - read only FineWeb-Edu excerpts (ODC-By 1.0, an open licence) and WordNet names;
  - are refused on any track not in `OPEN_LICENCE_TRACKS`; the clinical track is never sent (decision 58);
  - receive no gold frames: the gold is read only by `quality`, the gold condition and the report.
- **Clinical data:** none is involved. E7 has no clinical corpus (05.1), and nothing here reads MIMIC, SNOMED CT or
  UMLS.

## 7. Smokes (2026-10-20, CPU, `CUDA_VISIBLE_DEVICES=`; plumbing checks, not results)

Every smoke wrote only to scratch copies of the run folder and match corpora. `~/data/vsa-llm/e7/general-v1` and the
run folder are untouched.

| Command (job form + tiny overrides) | Outcome |
|---|---|
| `discover --host SmolLM2-360M` (`limit_texts` 80, `keep` 6) on the real D_read | exit 0, 837 s, 3.8 GB peak; 117,618 forward tokens; 6 candidates |
| `author --author SmolLM2-135M` (`link` on a copy of the real match corpora; `samples` 2, `max_new_tokens` 24) | exit 0, 138 s, 2.1 GB; 6 candidates authored |
| `author --author direct-SmolLM2-135M` (`documents` 3) | exit 0, 46 s, 1.8 GB |
| D7.2 on the 2 Oct tiny smoke data (round host SmolLM2-135M) with §3.1's settings: `base`, `verify` (self, teacher), `train self` | all ran: base 413 s, verify 3 s each, self 340 s. The resolved config shows host_mode lora, lora_rank 64, gate_bias 0.0, host lr 2e-4, channel lr 1e-3; self `init_from` is the round-0 state. Self accepted 0 edges (0 verifiable candidates on the tiny slice). |
| `train cm` (same data) | resolved as intended (LoRA 64, gate 0, 63,765 tokens = round + matched extra). Stopped by hand: about 15 tokens/s with LoRA on CPU |
| D7.3 `cross-prepare` and `cross-train authored` with §3.1's cross keys (tiny token counts) | exit 0, 119 s, 3.8 GB. The resolved train config shows grad_accum 1, lr 2e-3, min_lr_ratio 0.1; channel compose/hrr |

The E7 tests (`test_e7_*`, `test_authoring*`, 53 tests) and the new `test_e7_g3.py` (9 tests) pass with today's code.
The D7.0–D7.3 chain also runs end to end in `test_e7_pipeline.py`.

## 8. Changes after pre-registration

None yet.
