# Proposal (deferred): learning to use the store from the executor's own labels — "executor distillation"

**Status:** idea recorded 2026-10-21 for possible later work. Not pre-registered, not queued, no code. Author's
decision: "document this as something we may explore later."

**Origin.** LG-CF (pre-registration addendum 11; findings 2026-10-21 §4) showed that an exact executor over the store
keeps a chain of what-if edits consistent (99.7%) where the frozen hosts, given the same tree in the prompt, answer at
chance. The model never touches the store there: the executor's output is written into the prompt as text. The author
asked whether the executor's labels could instead train the model to use the vector construction itself, and drew the
analogy with how clinicians use decision trees: they consult them (lookup), they learn an intuition by working with
them in hand and then without (scaffolding), and the tree lets them learn a new procedure faster (efficiency).

## 1. The idea

The executor supplies unlimited, exact, free labels: sample a tree and a record, the executor returns the leaf and the
path. Train the host on those labels with the tree given **only through the embedding channel** (the store's vectors),
so that the only way to answer is to read the construction. Because the answer is a forced choice, the signal is
supervised (cross-entropy on the executor's leaf); reinforcement learning is needed only if the model is to produce a
free-form reasoning chain that the executor verifies step by step (RLVR with the executor as verifier), which would
train "run trees in text", a different question.

Three uses of a tree, in the analogy and in the model:

| Clinician | Model | Status |
|---|---|---|
| consults the flowchart each time | `chain-relay`: the executor runs, the model copies | measured (LG-CF) |
| learns with the chart in hand, then without | train with the tree and the executor's path visible, test with them withdrawn | **proposed arm (b)** |
| learns a new procedure faster because the chart exists | tokens to criterion with the tree available vs without | **proposed arm (c)** |

## 2. What the programme already says

- **The channel is learnable under dense, consistent supervision:** E0's planted structure (gate G1 passed); T5's
  invented words (implicit route −9 to −13% loss, +0.07 property). The executor's labels are that kind of signal, so
  learning to decode this tree family from the vector is the expected result.
- **The failures are of transfer, not of learnability:** readout arms trained under the LM loss on natural text did
  not learn to unbind roles (binding programme 09.4); nothing learned on synthetic structure has transferred to real
  text (E10.2; E11 and E9, T5 against T4). The analogy's "intuition" rests on the clinician already knowing what the
  attributes mean; our tree attributes are invented symbols with no meaning outside the task, so the *procedure* can
  transfer and the *content* cannot.
- **A tree is not one vector.** The pre-order layout needs about 2,500 dimensions at depth 8 (X8b), against a 960-wide
  channel. Injecting a tree means a sequence of slot vectors, one per node, in effect a soft prompt. Transformers learn
  to read soft prompts, so the interesting question is compositional: trained on depths 2–4, does the model run depth
  6, and does it handle unseen attribute names, where HRR unbinding is the same operation whatever the content? That is
  where the algebra could buy something a learned soft prompt would not.
- **What a trained reader would be for:** speed (one forward pass) and graceful behaviour on noisy records (LG-D2:
  the executor's only weakness is corrupted fields), at the price of exactness. Its honest role is a prior, e.g. a
  better search policy for the prover (LG-P: Qwen3-1.7B already guides search untrained), not a replacement for the
  executor.

## 3. Design sketch (about 3 GPU-h)

- **Host:** SmolLM2-360M, LoRA. **Data:** about 50,000 (tree, record) pairs from `world.random_tree`, depths 2–4,
  labelled by `reference.run_tree`; a fresh seed key; the LG-D / LG-CF item sets untouched.
- **Arms:** (i) channel only; (ii) tree as text only; (iii) *scaffolded*: channel plus the executor's path during
  training, both withdrawn at test; (iv) shuffled-vector control (the same training with the tree vectors permuted
  across items, to show the content is used); (v) no tree (chance).
- **Held out:** new trees at the trained depths; depth 6 (extrapolation); unseen attribute names; the LG-CF chains for
  consistency and the follow rate against the executor.
- **Primaries (Holm):** (a) channel-only leaf accuracy on held-out same-depth trees, rule ≥ 0.90 = the channel
  carries executable structure the model can learn to read; (b) scaffolded-then-withdrawn minus never-scaffolded at
  matched training steps (internalisation); (c) tokens to criterion with the tree available minus without (learning
  faster), the TTC metric of E13.
- **Secondaries:** depth-6 extrapolation, unseen names, LG-CF consistency; and, only if (a) passes, a transfer test:
  does the trained model's *implicit* use of T5 and T4 frames improve (RD-4's explicit − implicit contrast)?
- **Predictions, written 2026-10-21:** (a) passes; (b) the scaffold helps early and the gap closes — a curriculum,
  not a different endpoint; (c) the tree cuts tokens to criterion substantially (the most likely positive); depth-6
  extrapolation drops sharply for both routes; transfer to T4 is about zero.
- **Controls that must be in any registration:** the shuffled-vector arm; matched steps and tokens across arms; the
  executor's path never present at test; a text-only arm so that "learns the procedure" is not credited to the vector.

## 4. Why deferred, and when to pick it up

Deferred because the result the author cares about — "intuition" that transfers to real facts — is the part the
evidence predicts against, and because the queue holds the decisive real-text tests (T7-ROOD, E13) first. Pick it up
if (1) RD-4e shows the store's realistic pipeline matches the definition text on both tracks, so that a learned
reader has a store worth reading; or (2) a real-attribute tree set exists (clinical or chemical decision rules over
grounded attributes), where the analogy's intuition could in principle form. Register as an E14 addendum before any
run; the design above is the starting point, not a pre-registration.
