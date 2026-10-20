# 12 — E12: explicit recall — decoding the store into the prompt

**Overview.** The method stores a term's ontology frame in one composed vector, with each relation bound to its
partner. The binding program showed that the roles are in the vector (algebraic recovery 0.96 on held-out concepts) but
that the model does not use them: it cannot tell role-swapped twins apart. E12 decodes the store into a few lines of
text and puts them before the question. We call this "explicit recall". Exploration 12.1 tests the twins (Q1,
pre-registered, passed). 12.2 asks whether the decoded edges are the ones the model uses through its vector (F1: the
partners yes, the roles no). 12.3 moves to real chemistry at three seeds. 12.4 covers the tool layer: an agent that
calls recall itself, training on tool traces, and a self-critique loop. In short, explicit recall lifts twin accuracy
from 0.50 to 0.78 on T5 and role assignment from 0.55 to 0.88 on T4, within 0.015 of the gold relations as text. Any
host can read the store. Binding is needed for this, but learning the binding operator is not. A prose definition does
as well or slightly better on T4, and on T5 the decoded text is longer than a definition (183 against 116 tokens per
item). Read as text, recall's advantage is that it needs no curated text, not that it saves prompt tokens. Only a rule loop
that uses the decode without putting it in the prompt saves tokens (12.4). Only the T7-ROOD part is still queued.

**Terms used throughout.**
- **Tracks.** **T5:** a synthetic company glossary of invented words. **T4:** real chemistry (ChEBI entries and PubMed
  text). **T7-ROOD:** 2026 MeSH names in 2025–26 PubMed, retrained with every abstract that names a held-out record
  removed.
- **Hosts.** SmolLM2-135M and -360M (full fine-tuning); Qwen3-0.6B-Base and -1.7B-Base (LoRA).
  - **C5:** fine-tuned with the method, which adds the composed vector at each linked term;
  - **C0′:** fine-tuned on the same text without it; **C2:** a free per-term table; **P0:** the untouched model.
- **Store arms** (how the vector binds a partner to its relation):
  - **C5:** learned HRR binding; **C5rf:** a fixed random operator; **C5tr:** translation (partner + relation vector);
  - **C5ut:** untyped, a plain sum of partners with no roles; **C5sh:** each term trained on another term's frame.
- **Recall tool.** It unbinds each relation from the term's vector, snaps the result to the nearest known concept of
  the right type ("cleanup"), and writes one line per partner, with the cleanup cosine as a confidence:
  `- X depends on Y. (0.58)`.
- **Routes.** **Implicit:** the frame acts only through the composed vector. **Explicit:** the decoded frame is written
  as text before the question.
- **Conditions** (what precedes the question):
  - `none`;
  - `recall:own`: the host's own store; `recall:<arm>`: another arm's store;
  - `roleless`: the same decoded partners without their relations;
  - `wrong`: the partner twin's recall under this twin's name;
  - `symbolic`: the gold relations in the recall format;
  - `definition`: a prose definition written from the gold frame (E11's writer).
- **Readers** (E14 cross-references only): **oracle** = the gold frame; **type prior** = the most frequent relation for
  each found partner; **linker** = the trained model's own choice of relation (E11).
- **Twins:** 300 pairs of new T5 words with swapped roles (A *depends on* X and *is part of* Y; B the reverse). **Twin
  contrast accuracy:** the share of a pair's 2 relations × 2 templates in which the model prefers X over Y more for the
  twin that truly has X in that role than for its partner; chance 0.50.
- **PMI scoring** (property items): a candidate's log-probability after the prompt, minus its log-probability after a
  context-free null prompt in which the term is replaced by "this". The twin contrast needs no null.
- **Sign convention:** differences are a − b in accuracy, **positive = better**.
- **Statistics:** pairs (or terms, items) × seeds crossed random-effects model, Satterthwaite t, two-sided, 95% CI; Holm
  within Q1 and within F1. Secondaries unadjusted.

---

# 12.1 — Twins: does recall let the model use stored roles? (Q1, E12-1)

**Status:** done · **When:** pre-registered 2026-10-08 (commit 6638c39); pilot and 3-seed result 2026-10-08; report
regenerated 2026-10-17 · **Sources:** `experiments/e12-self-query/preregistration.md` §§0–4, 15; `experiments/e12-self-query/report/t5-q1-f1/report.md`;
`experiments/e12-self-query/report/t5-qwen3-e12/report.md`; `manuscript/status-2026-10-08.md` §3.4;
`manuscript/update-2026-10-12.md` (E12-1, E12-4)

## Research gap
Each part exists in the literature: tool use (Toolformer, ReAct), editable knowledge stores (Facts as Experts, KBLaM),
models verbalizing their own states (Patchscopes, SelfIE), and VSA decoders of neural states. The pre-registration's
check found no work that does all three of the following:
1. decodes the very vector the model computes with, by closed-form unbinding;
2. validates the decode by interventions on the store;
3. returns it to the same model as a tool, on items the model fails without it.

## Intuition
To answer a role question, the host would have to learn unbinding in its weights, and next-token loss never asks for
that. The algebra can unbind for the model. Reading text is something the model already does.

## Methodology
- **Items:** the 300 twin pairs, `choice` format (primary) and `cloze` format.
- **Hosts and stores:** SmolLM2-360M C5, C5ut, C5tr, C5rf, C5sh, C2 and C0′, seeds 1–3, plus P0 reading each seed's C5
  store. Replications: SmolLM2-135M (3 seeds) and Qwen3 (2 seeds).
- **Endpoints:** Q1a `recall:own − none`; Q1b `recall:own − recall:C5ut`; Holm over the two.
- **Decision rule:** reading (a), "the self-query reads the stored roles", needs both contrasts positive. It is labelled
  "text-driven" if `wrong:own` < 0.5.
- **Power:** detects 0.04 at 80%; the predicted effect was ≥ 0.2.

## Experiment
Every queued run finished. The pooled report covers both SmolLM2 sizes; a separate report covers Qwen3.

## Results
**Twin contrast accuracy, SmolLM2-360M C5 host, `choice`, seeds 1–3:**

| Context before the question | Accuracy | Prediction |
|---|---:|---|
| none | 0.496 | ≈ 0.5: passed |
| **recall:own** (HRR store) | **0.776** | 0.75–0.9: passed |
| recall:C5ut (no binding) | 0.501 | ≈ 0.5: passed |
| recall:C5tr (translation) | 0.494 | ≈ 0.5: passed |
| roleless:own | 0.506 | ≈ 0.5: passed |
| wrong:own | 0.242 | ≈ 1 − recall: passed |
| symbolic (gold relations) | 0.858 | 0.85–1.0: passed |
| definition | 0.741 | below symbolic: passed |

- **Both primaries pass:** Q1a +0.2794 [+0.2550, +0.3039] (Holm p 1.0e-26); Q1b +0.2744 [+0.2448, +0.3041] (Holm p
  1.3e-10). This is reading (a), text-driven.
- **The roles live in the store, not the host.** C0′ scores 0.785 and C2 0.790 reading C5's store; C5 − C0′ is −0.0092
  [−0.0206, +0.0023]. The untouched P0 reads it less well (0.681–0.701; C5 − P0 +0.0853 [+0.0591, +0.1114]).
- **The gain tracks the decode:** 0.836 on the 244 pairs decoded without error, 0.509 on the 55 with an error.
- **Costs of the text:** symbolic − recall:own +0.0825 [+0.0470, +0.1180] (decode errors); definition − symbolic −0.1172
  [−0.1415, −0.0929] (prose reads worse than statements).
- **Formats** (seed 1): no confidences 0.785, all-atom cleanup 0.757, decoding without known slots 0.675, "relation:
  partner" fields 0.659.
- **Replications:**
  - SmolLM2-135M: 0.806 against 0.491 (+0.3144 [+0.2918, +0.3371]);
  - Qwen3-0.6B: 0.792 against 0.494 (+0.2983 [+0.2235, +0.3732]);
  - Qwen3-1.7B: 0.785 against 0.503 (+0.2821 [+0.1620, +0.4022]).

  Their reports print "(b) partial" only because these hosts have no C5ut store, so Q1b is not computed.
- **Other item sets** (360M C5; secondary predictions):
  - two-hop chained recall 0.193 → 0.800 (gold 0.844), predicted far above none: passed;
  - reverse lookup 0.522 → 0.541 (gold 0.595), predicted above none: passed, barely;
  - new-word property 0.260 → 0.805 (gold 0.819), predicted ≈ gold: passed.

## Limitations
- **Storage, not learning.** The stores are built from gold frames, so E12 tests storage and read-out.
- **Q1b is near-guaranteed by design:** the twins' no-binding stores are identical.
- **Synthetic items.** The twins are templated, and the definition is generated from the gold frame. P0 has one run.

## Next steps
- **Done.**
- **Queued:** the T7-ROOD understanding report (54.49795).
- **Related:** models trained to unbind (the binding program's readout arms) still score 0.49–0.50 on the twins
  (preliminary, 18 Oct).
- **Open:** the claims ledger has no E12 row yet.

---

# 12.2 — Faithfulness: does the vector route use the decoded edges? (F1, E12-2)

**Status:** done · **When:** pre-registered 2026-10-08; result 2026-10-08; report regenerated 2026-10-17 · **Sources:**
`experiments/e12-self-query/preregistration.md` §5; `experiments/e12-self-query/report/t5-q1-f1/report.md`;
`experiments/e12-self-query/report/t5-qwen3-e12/report.md`; `manuscript/update-2026-10-12.md` (E12-2)

## Research gap
What a probe can decode, a model need not use (amnesic probing). A decode counts as an explanation only if intervening
on the decoded edge changes behaviour.

## Intuition
Suppose removing a decoded edge from the store lowers the model's preference for that edge's partner more than removing
another edge. Then the decoded edge is a cause of the behaviour. The twins test the same idea for roles.

## Methodology
- **Items:** 300 new T5 words inserted with their gold frames, and their property items. Nothing is in context, so only
  the implicit route acts.
- **Interventions:** remove one edge (its weight in the composer's sum set to 0), or keep only that edge.
- **Measure:** the comprehensiveness gap, i.e. the partner's margin after removing another edge minus its margin after
  removing this edge. Its per-word net share counts edges moved beyond τ = 0.05 nats (range −1 to +1).
- **Endpoints:** F1a, C5's net share > 0; F1b, C5 − C5ut. Holm over the two.
- **Role specificity** (twins): remove twin A's (relation 1, X). It is positive when X drops more under its own role.

## Experiment
SmolLM2-360M C5 and C5ut, seeds 1–3; SmolLM2-135M C5, seeds 1–3; Qwen3 C5, seeds 1–2.

## Results
| Measure (SmolLM2-360M, 3 seeds) | Estimate [95% CI] | Prediction |
|---|---|---|
| **F1a:** net share, C5 | **+0.3618** [+0.3303, +0.3933], Holm p 2.0e-66 | 0.1–0.4: **passed** |
| **F1b:** C5 − C5ut | **−0.0438** [−0.0738, −0.0138], Holm p 0.0043 | ≈ 0: **failed** (negative) |
| Role specificity, C5 | −0.0444 [−0.0838, −0.0051], p 0.029 | ≈ 0: failed narrowly (negative) |
| Role specificity, C5 − C5ut | −0.0300 [−0.0787, +0.0187] | — |

- **Secondaries:**
  - 0.6255 of decoded edges move beyond τ;
  - sufficiency +0.2749; specificity 0.2429;
  - robust to τ: +0.3494 to +0.3703 for τ from 0 to 0.1.
- **Replications of F1a:** 135M +0.4023 [+0.3658, +0.4387]; Qwen3-0.6B +0.5555 [+0.4272, +0.6837]; Qwen3-1.7B +0.6210
  [+0.5787, +0.6633].
- **Role specificity elsewhere is ≈ 0:** 135M −0.0033 [−0.0389, +0.0323]; Qwen3 −0.0362 and +0.0050 (both n.s.).
- **The pre-registered joint reading holds:** the decoded partners explain the vector route; the decoded roles act only
  when read back as text (12.1). Binding makes partner-level faithfulness slightly weaker, not stronger.

## Limitations
- Synthetic words and gold frames.
- Removing an edge renormalizes the others, and attention is not recomputed.
- Too few undecoded edges to test whether decoding tracks use (refutation reading F-R2).
- The 12 Oct update gives two −0.044 values (F1b −0.0438; role specificity −0.0444). They are different quantities.

## Next steps
Closed for T5. The readout arms tested training-time unbinding; their reading R1 looks null (preliminary, 18 Oct).

---

# 12.3 — Real chemistry: explicit recall on T4 at three seeds (E12-3)

**Status:** done (360M at 3 seeds; 135M, P0, two-hop and reverse at seed 1) · **When:** seed 1 2026-10-08; seeds 1–3
2026-10-17 · **Sources:** `experiments/e12-self-query/report/t4-s1-2-3/{report.md,analysis.json}`;
`experiments/e12-self-query/report/t4-s1/report.md`; `manuscript/findings/2026-10-17-e12-t4-explicit-recall.md`;
`manuscript/status-2026-10-08.md` §3.4; `manuscript/update-2026-10-12.md` (E12-3)

## Research gap
The twins are synthetic. We did not know whether recall resolves roles in a real ontology, where partners are confusable
and frames reach 24 edges.

## Intuition
A ChEBI chemical often relates to two others by confusable roles: *is a* against *has functional parent*, or a
conjugate acid against a conjugate base. If only the recall lets the model tell which partner fills which role, the
store is useful on real text.

## Methodology
- **Items:** 715 anchors × 4 templates. Each anchor has two relations to two chemicals, and each chemical is attested
  elsewhere under the *other* relation.
- **Subsets:** 300 seen, 300 rare, and 115 held-out terms (vectors composed zero-shot).
- **Conditions:** as on T5; `choice` (primary) and `cloze`.
- **Statistics:** the official report gives means only. The intervals below were derived by workplace-fd (paired
  bootstrap over anchors, seed-averaged, 4,000 resamples). There is no confirmatory endpoint.

## Experiment
SmolLM2-360M C5, C5ut, C5tr, C5rf, C5sh, C2, C0′ and the C6 row-source arms, seeds 1–3; SmolLM2-135M and P0 at seed 1.

## Results
**Role-contrast accuracy, SmolLM2-360M, `choice`, seeds 1–3:**

| Host | Context | Accuracy | Decode accuracy |
|---|---|---:|---:|
| C5 | none | 0.554 | — |
| C5 | **own store** | **0.882** | 0.908 |
| C5 | gold relations | 0.897 | 1.000 |
| C5 | prose definition | 0.915 | — |
| C5 | partners without roles | 0.479 | 0.909 |
| C5 | no-binding store (C5ut) | 0.500 | 0.882 |
| C5 | translation store (C5tr) | 0.511 | 0.625 |
| C5rf | own fixed-random store | 0.881 | 0.960 |
| C5sh | own shuffled store | 0.559 | 0.439 |
| C0′ | C5's store | 0.877 | 0.908 |

- **Derived contrasts:**
  - own store − none +0.328 [+0.302, +0.355];
  - own − no-binding store +0.382 [+0.357, +0.407];
  - own − gold −0.015 [−0.023, −0.006]; own − definition −0.033 [−0.049, −0.017];
  - C0′ − C5 −0.005 [−0.011, +0.001].
- **The seeds agree** (0.880–0.884).
- **Zero-shot vectors read back as well as gold:** held-out terms score 0.925 against gold 0.929 (none 0.707).
- **Capacity shows at large frames.** The gap to gold is +0.003 at 0–7 edges (n = 176), −0.014 at 8–12 (n = 397) and
  −0.037 at 13–24 (n = 142).
- **Cloze:** 0.540 → 0.794 (gold 0.822, definition 0.856).
- **Seed-1 hosts:** SmolLM2-135M 0.565 → 0.894 (definition 0.875); P0 store 0.857, gold 0.887, definition 0.822.
- **Multi-step (seed 1):**
  - two-hop 0.230 → 0.623 (gold 0.843); the prediction "far above none" passed;
  - reverse lookup 0.554 → 0.477 (gold 0.766); the prediction "above none" failed.
- **Cross-reference (E14, file 14):** E14 used different items, E11's T4 property questions.
  - Whole-frame recall lowered accuracy there (−0.103\*), about two-thirds of it from a PMI null that also read the
    recall.
  - Recall of only the asked relation raised it by +0.248\* (RD-4b).
  - E12's scorer uses a context-free null, so that artefact does not apply here.

## Limitations
- **Storage, not discovery.** The stores come from the curated ontology. E11 shows that reading facts from real
  definitions does not work yet.
- **Gold-derived definition.** The "definition" is written from the gold frame, not taken from ChEBI.
- **Unofficial intervals.** The intervals are derived, and the report code should recompute them before anyone cites
  them.
- **Superseded wording.** The 8 Oct status (§3.4) said the store "costs fewer tokens" than a definition. That does not
  hold for decoded text in the prompt (12.4).

## Next steps
- **Proposed:** add natural-item contrasts with intervals to `e12_report`.
- **Queued:** the T7-ROOD understanding report (54.49795); the training of the width-512 arm C5w512 on T4 (ontology
  richness, stage 1; ≈ Oct 27–28).
- **Proposed:** run E12 on C5w512 to test capacity at large frames.
- **Open:** fold the result into the claims ledger and the draft.

---

# 12.4 — The tool layer: an agent, training on tool traces, and self-critique (3a, 3b, 3c)

**Status:** done (runs and reports; 3a has no pooled endpoint report) · **When:** pre-registered 2026-10-08, amended
2026-10-08 and 2026-10-09 (before any run); 3a runs 2026-10-16; 3b and 3c reports 2026-10-17 · **Sources:**
`experiments/e12-self-query/preregistration.md` §§10, 12, 13, 16.2–16.5; `experiments/e12-self-query/agent/*/report.md`;
`experiments/e12-self-query/report/t5-3b/report.md`; `experiments/e12-self-query/report/t5-3c/report.md`

## Research gap
12.1 uses a fixed pipeline. Three questions were open:
- **3a:** can a model decide when and how to call recall?
- **3b:** does practice with recalled roles teach it to read roles without the tool?
- **3c:** can it check its decoded beliefs and reject false structure, where E10 failed?

## Intuition
- Base models follow few-shot tool formats.
- Reading a role from an injected vector is a linear map, which LoRA could learn.
- Calibrated confidence, plus outside evidence, should flag false beliefs.

## Methodology
- **3a:** untouched Qwen3-0.6B- and 1.7B-Base agents with tools on the Qwen3 C5 stores (seeds 1–2). The self-hosted
  LoRA C5 runs (seed 1) are a secondary. 900 questions per run: 600 twin, 150 two-hop and 150 reverse. Endpoints: A1,
  agent − no tool; A2, agent − fixed pipeline.
- **3b:** SmolLM2-360M C5 × 3 seeds; rank-16 LoRA, with the host and the channel frozen. Arms:
  - **T:** tool traces; **I:** answers only; **S:** the tool removed step by step;
  - **L:** a token-matched control; **I-ut:** a leak check on the no-binding store.

  B1: the twins without the tool, I − L and S − L.
- **3c:** a rule loop answers with the decoded belief when its calibrated confidence is ≥ θ, else with the no-tool
  answer; disagreements are flagged. A null world uses a corrupted store. Endpoints, at 80% coverage:
  - **K1:** accuracy, loop − no tool;
  - **K2:** false-belief adoption, loop − always adopting the recall;
  - **K1b:** loop(store) − loop(definition), non-inferiority at −0.05.

## Experiment
All six 3a runs; 3b's 11 arms × 3 seeds; 3c on C5 and C5ut × 3 seeds, plus a Qwen3-1.7B-Base model loop × 2 seeds.

## Results
**3a, twin contrast accuracy over 300 pairs** (descriptive; no pooled test):

| Agent (store) | Well-formed first call | Agent | No tool | Fixed pipeline |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Base (C5 store), seeds 1 / 2 | 0.81 / 0.89 | 0.94 / 0.95 | 0.50 / 0.50 | 0.89 / 0.85 |
| Qwen3-0.6B-Base (C5 store), seeds 1 / 2 | 0.57 / 0.56 | 0.76 / 0.76 | 0.46 / 0.46 | 0.97 / 0.92 |
| Qwen3-1.7B LoRA C5 (own store), seed 1 | 0.06 | 0.51 | 0.50 | 0.72 |
| Qwen3-0.6B LoRA C5 (own store), seed 1 | 0.57 | 0.63 | 0.45 | 0.94 |

- The 1.7B-Base agent meets A1's precondition (well-formed calls ≥ 0.8). It beats the fixed pipeline, against the
  predicted A2 < 0.
- Fine-tuning on the T5 corpus degrades the protocol, as the pilot found: the 1.7B LoRA run makes almost no well-formed
  call, and the 0.6B run often swaps the two tools' names (0.15).

**3b:**
- **B1:** I − L +0.0156 [−0.0049, +0.0360] (Holm p 0.132); S − L +0.0289 [+0.0077, +0.0501] (Holm p 0.0153). The report
  reads this as "internalization shown by S − L", though S scores only 0.520 without the tool.
- **Predictions:** B1 ≤ 0.65 passed; S ≥ I passed; I-ut ≈ 0.5 passed (+0.0033 [−0.0107, +0.0174]).
- **Failed predictions:**
  - arm T's well-formed calls ≥ 0.9: it made **none** (0.000), while arm I made 0.972. B2's test, T − I on agentic
    twins, is −0.2967 [−0.3838, −0.2095];
  - decision 64's equivalences (C0′ host ≈ C5 host; C5rf ≈ C5) are not shown. Only B1's contrast falls within its
    margin; arm T's contrasts are inconclusive.

**3c** (SmolLM2-360M C5, 1,790 test items × 3 seeds):

| Loop (real world) | Accuracy at 80% coverage | Prompt tokens added per item |
|---|---:|---:|
| no tool | 0.432 | 0 |
| **rule loop** (store, read without the host) | **0.985** | 0 |
| host reads the decoded store (no loop) | 0.944 | 182.9 |
| loop, definition in the prompt | 0.917 | 115.6 |
| loop, gold relations as text | 0.902 | 182.9 |
| loop, decoded store as text | 0.893 | 182.9 |

- **K1:** +0.5531 [+0.5278, +0.5783], Holm p ≈ 2e-277; predicted > 0: **passed**.
- **K2:** −0.0144 [−0.0236, −0.0052], Holm p 0.0021. It is significant but tiny: in the null world the loop adopts the
  false belief 0.986 of the time, against 1.000. Predicted < 0 on held-out terms: passed (−0.0104 [−0.0192, −0.0015]).
  Predicted ≈ 0 on new words: failed, marginally (−0.0197 [−0.0380, −0.0014]). Revising with evidence (a secondary)
  cuts held-out adoption by 0.6907.
- **K1b:** +0.0680 [+0.0384, +0.0976], non-inferior and superior: **passed**. The store loop adds 116 fewer prompt tokens
  per item.
- **The decode as text is longer than the definition** (182.9 against 115.6 tokens). Read by the host, it is slightly
  worse than the definition (−0.0247 [−0.0347, −0.0147]; non-inferior at −0.05).
- **Qwen3-1.7B-Base model loop:** K1 +0.1901 [+0.1322, +0.2481]; K2 −0.8632 [−0.9149, −0.8114].

## Limitations
- **3a:** per-run outputs only, two seeds, and A1/A2 untested. The per-run headers still say "feasibility pilot",
  though the runs are full size.
- **3b:** arm T's zero well-formed calls is unexplained; it may be a training or a harness problem. It blocks B2.
- **3c:**
  - the beliefs come from a gold-built store, so K1 measures storage plus calibration;
  - θ came out at 0 in every seed, so the loop always answers with a belief when it has one;
  - without evidence, the loop barely resists false structure.
- **Not yet in any manuscript summary.**

## Next steps
- **Proposed:** a pooled 3a report (A1, A2); a diagnosis of arm T; a write-up of 3b and 3c.
- **Closed:** LoRA internalization is small (+0.03). Explicit recall stays the route for using stored roles; E13
  amendment 3 builds it into the learning cycle.

---

# Sources

- `experiments/e12-self-query/preregistration.md` (§§0–16)
- Reports: `experiments/e12-self-query/report/{t5-q1-f1,t5-qwen3-e12,t4-s1,t4-s1-2-3,t5-3b,t5-3c}/report.md`; agent runs
  `experiments/e12-self-query/agent/{qwen3-0.6B-base-C5store-s1,-s2,qwen3-1.7B-base-C5store-s1,-s2,qwen3-0.6B-C5-s1,qwen3-1.7B-C5-s1}/report.md`
- `manuscript/findings/2026-10-17-e12-t4-explicit-recall.md`
- `manuscript/status-2026-10-08.md` §3.4; `manuscript/update-2026-10-12.md` (E12-1 to E12-4)
- Cross-references: `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §1;
  `manuscript/findings/2026-10-20-e14-tier-p-gpu.md` §1; `manuscript/findings/2026-10-20-e14-followups.md` §1;
  `manuscript/status/10-20-2026/14-e14-reading-real-text.md` (14.4–14.5)
- Dates: `git log -- experiments/e12-self-query` (6638c39 2026-10-08 … 0f23531 2026-10-17); report file times. Only
  `report/t4-s1-2-3` is committed; the other report folders are untracked.
