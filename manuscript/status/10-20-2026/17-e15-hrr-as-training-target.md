# 17 — E15: the HRR vector as a training target

**Overview.** Every earlier experiment fed the composed vector to the model and hoped the language-modelling loss would
teach the model to read it. It did not: the model used the vector as a per-term identifier and ignored its structure
(files 03, 06, 09). E15, approved on 22 October at high priority, puts the HRR into the loss. It asks two questions in
sequence: can the model be trained to *read* a term's facts out of the vector, including for words it has never seen;
and, if so, does it then *use* them without being told them in text? Stages 1, 1b and 1c have run (22–25 October);
stage 1d (500M tokens) is queued. Full numbers: `manuscript/findings/2026-10-25-e15-stages-1-1c.md`.

**Terms used throughout.**
- **Write loss:** the model's hidden state at layer 11, through a linear map, must reproduce the term's HRR bundle.
- **Read loss:** each of the term's fillers must be recovered from that state by unbinding with its role and matching
  against the table of atom codes (Plate's cleanup memory, made an objective).
- **Read accuracy:** the share of a word's facts recovered that way; chance is 1 in 3,493.
- **Implicit QA:** multiple-choice questions about a word answered with only its vector injected, no text about it.
  **Explicit recall:** the same facts written as text before the question (E14's route).
- **Compositional held-out:** 285 invented words (T5) whose every relation and partner was seen in training, but
  whose combination is new. A model that memorises cannot answer these; one that reads the algebra can.
- **C5+WR, C5+R:** the method plus write-and-read or read-only losses. **C2+R:** a free per-term vector (no
  structure) plus the read loss. **C5sh+WR:** the method with the facts shuffled. **C5-long:** the method alone at the
  same length as the long runs. **RD:** route distillation (the model imitates its own behaviour when given the facts as
  text). **FV:** fact verbalisation (a cloze objective on the facts in token space).

---

# 17.1 — Stage 1: do write and read losses teach the model the algebra? (50M tokens)

**Status:** done · **When:** registered and queued 2026-10-22, report 2026-10-22 · **Sources:**
`experiments/e15-hrr-target/report/t5/report.md`; `experiments/e15-hrr-target/preregistration.md` (stage 1)

## Research gap
After the binding programme (file 09) we knew the model does not read roles from the vector when nothing in the loss
rewards it. We did not know whether it *could*, given a loss that does.

## Intuition
Unbinding by a fixed role is a linear operation, and transformers learn linear maps easily. If the hidden state is
asked to carry the bundle and give back each filler by role, the model should learn the algebra, and a word it has
never seen should then be readable from its vector alone.

## Methodology
- **Arms (3 seeds):** C5+WR, C5+R, C2+R (the control that can memorise but not generalise), C5sh+WR (same algebra,
  wrong content). Existing C5, C2 and C5sh runs as baselines.
- **Losses:** L = L_LM + 0.20 L_write + 0.035 L_read, the weights fixed from a CPU smoke so the added terms were about
  16% of the LM loss at step 0.
- **Primaries (Holm):** implicit QA on compositional held-out words, C5+WR − C5; C5+WR − C2+R; read accuracy of C5+WR.
- **Decision rule:** QA gain ≥ +0.15 and read ≥ 0.8 → the algebra is learnable; read < 0.6 → closed at this scale.

## Experiment
Twelve training runs, 42 evaluations, one report; about 20 GPU-h; 22 October.

## Results
- **Read accuracy on held-out words 0.48** (C5+R 0.40; C2+R 0.045; C5sh+WR 0.45). The structure is what is read:
  the free table cannot, and shuffled facts are read as well as true ones.
- **Implicit QA 0.294 against 0.264 for C5: +0.030\***. Explicit recall of the same facts: 0.82. The shuffled arm
  gets no QA gain (0.200), so the small gain is the content.
- **Loss after held-out words:** C5+WR −1.9%\* and C5+R −2.7%\* against C5, but C5sh+WR **+8.7%** and C2+R **+17.6%**.
  Forced to carry the vector, the model's behaviour follows what is in it — correct facts help, wrong ones hurt — the
  first such result in the programme.
- Unseen filler atoms decode at 0.54, above the compositional words: the mechanism is "carry the vector to layer 11
  intact and unbind", not a learned decoder per relation.

## Limitations
One host size; the read loss was still falling at the end of training (undertrained); the 0.8 bar was set from first
principles, not from any measurement.

## Next steps
Stage 1b (length) and stage 1c (aligned objectives), both registered before any run.

---

# 17.2 — Stage 1b: more tokens (200M), and a matched-length control

**Status:** done · **When:** registered 2026-10-22, control added 2026-10-23 before any evaluation, report 2026-10-24 ·
**Sources:** `report/t5-stage1b/report.md`; pre-registration §16

## Research gap
Stage 1's read loss had not converged. Is 0.48 a ceiling or a point on a curve?

## Intuition
If the mechanism is linear and the loss is still falling, more tokens should raise read accuracy; whether use follows
is the real question.

## Methodology
C5+WR at 200M tokens, 3 seeds, evaluated at 50/100/150/200M checkpoints; C2+R at 200M (seed 1); one seed with the read
weight tripled; and, added before any evaluation ran, **C5-long**, the method alone at 200M, because comparing a 200M
arm with a 50M baseline confounds the objective with the tokens. Primaries: QA at the checkpoint chosen by read
accuracy; read accuracy at 200M; the 150→200M increment. Rule: read 0.6–0.8 and still rising → register a longer stage.

## Experiment
Five long runs, 29 evaluations, one report; about 30 GPU-h; 22–24 October.

## Results

| tokens | read accuracy, held-out | implicit QA (C5 0.264) |
|---:|---:|---:|
| 50M | 0.59 | 0.330 |
| 100M | 0.72 | 0.357 |
| 150M | 0.75 | 0.369 |
| 200M | **0.76** | **0.375 (+0.112\*)** |

- Read accuracy is still rising at 200M (+0.010 per 50M, CI excludes 0).
- **Half the QA gain is length:** C5-long reaches 0.329 on its own; the objective adds **+0.046\*** at matched length.
- C2+R-long stays at 0.04 read and 0.21 QA: four times the tokens do not let a structureless vector generalise.
- **The loss gain fades at matched length:** −0.27% (n.s.) after held-out words against C5-long, where it was −1.9%
  against C5 at 50M.
- Tripling the read weight: +0.03 read accuracy, +1.2% loss. Not worth it.

## Limitations
One matched-length control seed; the long runs' learning-rate schedule differs from stage 1's at equal tokens (the
50M checkpoint reads 0.59, not 0.48, for that reason).

## Next steps
By the rule: stage 1d at 500M, one seed of C5+WR and of C5 (about 25 GPU-h, queued 25 October; seeds 2–3 conditional).
Registered predictions: read 0.80–0.83, objective's QA share +0.04–0.06, i.e. no change of conclusion.

---

# 17.3 — Stage 1c: objectives aligned with the test (route distillation, fact verbalisation)

**Status:** done · **When:** registered 2026-10-23, report 2026-10-25 · **Sources:** `report/t5-stage1c/report.md`;
pre-registration §17

## Research gap
Stage 1 showed the model can decode the vector but barely uses what it decodes. The read loss lives in atom space and
is scored by an external head; the test is a judgement in token space by the LM head. Nothing connects them.

## Intuition
Make the objective the test. Route distillation: the model already uses the facts when they are text (0.82), so make
the vector-only pass imitate the text pass. Fact verbalisation: make the model say each fact in words from the vector.

## Methodology
- **C5+RD** (3 seeds): a teacher pass of the same model with the gold frame as recall lines prepended; the student pass
  sees only the vector; KL between their next-token distributions over the 8 tokens after the term; weight 1.0.
- **C5+FV** (3 seeds): cloze prompts "term: relation: ___" with the filler's name as target, templates disjoint from
  every question template; weight 0.05.
- **Controls:** C2+RD, C2+FV (free table + the same objectives, seed 1).
- **Rule:** C5+RD ≥ 0.50 on held-out words and ≥ +0.15 over C2+RD → the model learns to use composed facts when asked;
  < 0.35 → aligned objectives do not close the gap; implicit route closed.

## Experiment
Eight training runs, 16 evaluations, one report; about 13 GPU-h; 23–25 October.

## Results
- **Route distillation made things worse:** implicit QA 0.247 (**−0.017\*** against C5), loss after held-out words
  **+8.6%**, overall +9.2%. The student matched its teacher (KL 0.38 → 0.017); the teacher, with recall text
  prepended, is a worse predictor of ordinary text than the student, and the student learned that. One real effect in
  the wrong place: explicit recall on the distilled model rose from 0.84 to **0.93** — it learned to use recalled text
  better.
- **Fact verbalisation:** implicit QA 0.278 (+0.014\*); the best loss of any E15 arm (after held-out words −4.7%\*,
  overall −4.8%\*); the cloze task was learned (5.7 → 0.16 nats) without transferring to question answering.
- Controls at 0.18–0.19, as predicted. **The rule's "closed" branch fired.**

## Limitations
One host size; one teacher design (full gold frame prepended; KL over all eight following tokens rather than the
fact-bearing ones); 50M tokens.

## Next steps
Stopped. The write-up keeps the loss-domain result (17.1) as the structural-use finding and the explicit route as the
usable one.

---

# 17.4 — What E15 says so far

- **Learnable:** a 360M host can be trained to read an HRR vector for words it has never seen — 0.76 of their facts
  at 200M tokens and rising — and only the structure makes this possible (free tables stay at chance).
- **Not used:** the same model answers questions through the vector at 0.375 (0.329 of it from length alone), against
  0.82 with the facts as text. Token-space objectives did not close that gap; the one that aligned best with the test
  (distillation) harmed both routes.
- **Behaviour follows content when the model is made to carry the vector** (C5+WR −1.9% vs shuffled +8.7% at 50M),
  but the benefit shrinks to nothing at matched length.
- **Pending:** stage 1d at 500M (does reading cross 0.8; does the objective's share grow). The registered prediction
  is that it does not change the conclusion.
- **For the manuscript:** the strong hypothesis is closed in every form tried; the narrow positive — trainable
  decoding of composed facts, structure-dependent, without use through the weights — is new and specific.

# Sources
`experiments/e15-hrr-target/preregistration.md`; `experiments/e15-hrr-target/report/{t5, t5-stage1b, t5-stage1c}/`;
`manuscript/findings/2026-10-25-e15-stages-1-1c.md`; `manuscript/proposals/2026-10-21-executor-distillation.md` (the
idea's origin).
