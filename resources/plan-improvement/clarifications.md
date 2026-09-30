# Clarifications needed

Questions whose answers change [tasks.md](tasks.md) or the proposal. Each states the default that the schedule assumes if no answer is given, so work can start without blocking. Answer inline under each item.

**Status (2026-09-30): all 30 answered and applied** (1–24 in the table below; 25–30 at the end). The table records where each answer landed; follow-up questions raised by the answers are at the end, each with the default the plan now assumes.

| # | Answer, in short | Applied in |
|---|---|---|
| 1 | from-scratch core; continued pretraining also required; pretrained models may author their own ontologies | new M5 ([formulation.md](formulation.md) §5), H-G and E7; E4.6 made a required track; D4.4, D7 |
| 2 | explore every application area; self-authoring in each | E8 with tracks T1–T7 and a common recipe; proposal §6 decision; C7, D8 |
| 3 | 50M / 125M / 350M, both regimes | E4.0, E4.6; 350M top-4, last in the GPU queue |
| 4 | no deadline | schedule is GPU-queue driven, ≈ 42 weeks |
| 5 | keep CG-VSA | unchanged |
| 6 | use this computer's GPU | one RTX 3090 (24 GB), 12 cores, 30 GB RAM, ≈ 125 GB free disk; compute re-planned at ≈ 3,200 GPU-h; 350M at 2.5B tokens; checkpoint/resume and job queue (B11); experiments §0.9 |
| 7 | FineWeb-Edu subset | C3 |
| 8 | GPT-2 BPE | E4.0, B7; 32k-BPE option removed |
| 9 | gpt2, Qwen2.5-0.5B, SmolLM2-135M, add SmolLM2-360M; respect local limits | B6 memory table, chunked cross-entropy, frozen/LoRA hosts |
| 10 | three seeds | unchanged; C8 to two seeds only if the queue slips, reported |
| 11 | SNOMED CT / UMLS / MIMIC available | clinical track T1 unconditional and the flagship (D4.5); MIMIC text stays local |
| 12 | grade with an LLM for now, clinicians later | LLM-judge protocol (experiments §0.12), `judging.py` (B13), C5; clinician study W4 |
| 13 | Claude implements | staffing line in tasks.md |
| 14 | fixes in `vsa_embed` with new names | Phase 0 unchanged |
| 15 | new run IDs, old runs marked stale; errata appended; index corrected | A6 |
| 16 | BERTHA reference only | repository policy in tasks.md |
| 17 | commit automatically, push to `main` | repository policy and rule 8 in tasks.md; branch-per-phase dropped |
| 18 | `preregistration.md` | G2 |
| 19 | prompting and linear probes | experiments §0.11, B8 |
| 20 | `torchao` weight-only INT8/INT4 | E4.4, B8 |
| 21 | ≥2 subtokens; also track feasibility by cardinality of ≥ `ℓ` subtokens | span-cardinality report (formulation §4.1, B5), E4.7 / D4.7, per-track feasibility verdicts in E8 |
| 22 | minimal stem-cell baseline | D3.2 unchanged |
| 23 | web search, standalone `related-work.md` | C1, extended to M5 prior art |
| 24 | "compositional parameter sharing" framing | experiments E4 gate, proposal §10, tasks rule 7 |

## Scope and priorities

1. **Which regime is the core claim?** From-scratch small LMs (E4 as written) or retrofitting existing small models (continued pretraining, E4.6)? *Default:* from-scratch at 125M is the pre-registered core; continued pretraining on SmolLM2/Qwen2.5-0.5B is the secondary track.

Default sounds good here - but we do need to test both as possible tasks - the interesting part is if the model is pre-trained - it has the possibility of self-improving by reading text and generating self ontologies that can improve its own weights

2. **Application tracks.** Confirm the recommendation: WordNet general track as the scientific core, developer-tools track first (contamination-free, unlicensed), biomedical as the flagship only if licences are in hand. Is there a track you would drop or add (product catalogues, chemistry, enterprise glossaries)? *Default:* WordNet + developer tools; bio via MeSH + PubMed unless C2 succeeds.

Yes these are good - for each other these, we want to explore all application areas - as well as the possibility of self ontology authoring (by reading text and then generalizing them in terms of an ontology - if the model is a pre-trained model)

3. **Model sizes.** 50M / 125M / 350M as planned, or include a ~1B run? *Default:* as planned; 350M only for the best four conditions.

As planned - default for now - using both pre-trained and from scratch

4. **Timeline and deadline.** Is ~36 weeks acceptable, and is there a target venue or date that should pull G3 (the core-claim decision) earlier? *Default:* no external deadline; the schedule is dependency-driven.

No deadline - just do the work

5. **Working name.** "CG-VSA" is a placeholder. Any preference, or keep it until the write-up? *Default:* keep the placeholder.

keep place holder

## Compute and data

6. **Hardware.** How many GPUs, which class (A100/H100/consumer), and for which weeks? The plan assumes one A100 for weeks 1–11 and a 4-GPU window for weeks 12–22 (~650 A100-hours for E4 plus ~200 for the smaller experiments). If only consumer GPUs are available, E4 caps at 125M and 350M is dropped. *Default:* as assumed.

Use the GPU in this computer - update the experiment to run locally as needed

7. **Corpus.** Is a FineWeb-Edu-class public subset acceptable for the general track, or is there an in-house corpus that must be used? Any storage or download constraints (7B tokens ≈ tens of GB)? *Default:* FineWeb-Edu subset.

default is good

8. **Tokenizer.** Use GPT-2 BPE for the from-scratch models (keeps the existing single-token lemma tooling and matches the cached GPT-2 host) or train a 32k BPE on the corpus? *Default:* GPT-2 BPE.

GPT-2 BPE

9. **Hosts.** The local Hugging Face cache already holds `gpt2`, `Qwen/Qwen2.5-0.5B` and `HuggingFaceTB/SmolLM2-135M`. Are these the intended hosts, and may the plan add SmolLM2-360M? *Default:* yes to both.

Yes to both - also consider the limitation of the local hardware

10. **Seeds.** Three seeds minimum for promoted results, per the dossier. More for E4 if compute allows? *Default:* three; two for the C8 operator ablation only if compute is short (reported as such).

default answer is good

## Licences and people

11. **SNOMED CT / UMLS / MIMIC.** Does the team already hold these (the BERTHA work suggests so), and can they be used for this project? If yes, the bio track becomes the flagship application and D4.5 is unconditional. *Default:* treat as not available until confirmed; prepare MeSH + PubMed.

Yes all done

12. **Expert raters for E5.3.** Are clinicians or other domain experts available for a ~40-item rating study around weeks 20–28, and is any ethics/consent process required? *Default:* two raters, no PHI, no formal review needed; recruiting starts week 8.

Grade using an LLM for now - we want to estimate for now - we will bring in clinicians after

13. **Team.** Who implements, and at what fraction of time? The schedule assumes one full-time research engineer plus a half-time researcher. *Default:* as assumed.

as assumed - claude will implent

## Code and repository policy

14. **May the audit fixes modify `src/vsa_embed` and its tests?** Some recorded conclusions change (audit F1, F5a) and some tests encode the old behaviour (e.g. the `promotion_eligible=True` default). *Default:* yes, on a branch, with new option/family names so recorded runs still replay.

default answers is good

15. **Recorded runs and dossier text.** After the fixes, should 01b Stage B and 01c.3–01c.5 be re-run into new run IDs with the old runs kept and marked stale (default), or should the old runs be deleted? Should the dossier READMEs under `resources/vsa-understanding/` be edited in place, or receive appended errata sections? *Default:* new run IDs, old runs kept and marked; errata sections appended, plus a one-line status correction in the program index.

default is good

16. **`resources/models-main` (BERTHA).** Keep as an untouched reference (default) or port the normalization and accumulation fixes there too? *Default:* reference only; the fixes live in `vsa_embed`.

reference only - fixes live in vsa_embed

17. **Branching and commits.** One long-lived branch per phase with PRs, or feature branches per task? Commit messages per the repository's convention? *Default:* one branch per phase (`phase-0-audit-fixes`, `phase-1-infra`, …), PR per phase, no commits without your go-ahead.

automatically commit as needed - push directly to main as needed

18. **Pre-registration location.** A `preregistration.md` in the experiment folder with the holdout hash and frozen configs, committed before D4.1 starts? *Default:* yes.

yes

## Evaluation choices

19. **Probes for small LMs.** Prefer prompting-style evaluations, linear probes on hidden states, or both? Loss-based stratified metrics are the primary endpoint either way. *Default:* both, with linear probes as the reported secondary metrics because prompting is unreliable below 1B.

both

20. **Quantization tooling and target.** GPTQ/AWQ-style weight-only PTQ via `torchao`/`auto-gptq` (default), `bitsandbytes`, or an export target such as llama.cpp that dictates the format? *Default:* torchao weight-only INT8/INT4.

default answer is good

21. **Minimum span length.** The channel links spans of ≥2 subtokens by default, with a single-token rare-word track. Should single-token words be included in the main condition instead? *Default:* ≥2 subtokens in the main condition; single-token rare words as a variant.

default is good - we also want to track the feasibility based on the cardinality of >= subtoken 

22. **The 01d "stem-cell" baseline.** Implement a minimal version of the designed entmax-routed expert pool as a comparison in E3 (default, ~3 pd), or drop it because the design was never run? *Default:* implement minimally, relations only.

default

## Literature and claims

23. **External literature search.** May the search use web tools from this environment, and should it produce a standalone `related-work.md` (default) or be folded into the proposal? *Default:* web search allowed; standalone note, finished before G0.

default is good

24. **Claims discipline.** If the operator ablation shows `random_fixed ≈ hrr`, the program reports "compositional parameter sharing" rather than "symbolic semantics". Confirm that this framing is acceptable for the write-up. *Default:* yes.

yes

## Follow-up questions raised by the answers

Each has a default the plan already assumes; answer only to change it.

25. **API judge.** Which API model should be the external LLM judge for ontology-only items (codes, preferred terms, neighbours, edges)? *Default:* a current frontier model with its version pinned in the protocol file, plus a local ≈ 7B instruct model on the 3090 as the second judge.
- just use a frontier model - call claude code itself to do the judging
26. **MIMIC and external services.** Does your MIMIC data-use agreement allow any MIMIC-derived text to reach an external LLM service? *Default:* no; MIMIC-derived items are graded by the local judge only, and the harness enforces it.
yes it can reach external llm services
27. **Compute budget.** ≈ 3,200 GPU-hours is ≈ 23 weeks of the 3090 at realistic utilization. Is that acceptable, and if something must be cut, is the 350M phase the first cut? *Default:* keep everything, run 350M last; cut order as in the tasks.md risk table.
Lets reduce so its alot fewer gpu hours - we instead want to see the speed that convergence occurs and generate evidence in which we can justofy additional gpu hours
28. **Self-authoring relation vocabulary.** Closed (the target ontology's relation labels) as the main condition and open relations as an ablation? *Default:* yes.
yes
29. **Desktop GPU.** Long jobs will share the GPU that drives the desktop (≈ 22 GB usable, some UI lag possible). Acceptable, or should jobs pause while you are working? *Default:* run continuously; jobs checkpoint every 30 min and can be paused from the queue.
default answer is good
30. **SmolLM2-360M download.** It is not in the local cache yet (≈ 0.7 GB). *Default:* download when B6 starts.
default is good

### How 25–30 were applied (2026-09-30)

| # | Answer, in short | Applied in |
|---|---|---|
| 25 | frontier model only; call Claude Code itself | judge = headless `claude -p` with the model pinned, ≥3 calls with paraphrases, self-agreement instead of inter-judge κ (experiments §0.12, B13); the same client is the M5 teacher author, so no local 7B model is needed |
| 26 | MIMIC-derived text may reach external LLM services | local-only judging rule removed (experiments §0.12, E7, E8-T1; tasks C2, rule 9); MIMIC text is still never committed or published |
| 27 | far fewer GPU hours; measure convergence speed and build evidence for more | committed budget ≈ 400 GPU-h (was ≈ 3,200): 50M at 300M tokens, 125M at 500M tokens, continued pretraining at 100M tokens, E7 round 1, short E8 runs; convergence metrics (data multiplier, fitted curves, projections) and a pre-registered escalation rule (experiments §0.13); larger runs moved to escalation tiers X1–X4 requested in the W3 evidence report at G6; new `convergence.py` (B14); calendar ≈ 22 weeks |
| 28 | closed relation vocabulary, open as ablation | unchanged (formulation §5.5) |
| 29 | run continuously | unchanged (B11) |
| 30 | download SmolLM2-360M at B6 | unchanged (B6) |
