# The VSA-LLM programme in plain terms: the gap, how we test, what we ran, what it means

**Written 2026-10-22.** A non-technical account of the whole research programme as it stands. The detailed digest is
`manuscript/status/10-20-2026/00-digest.md`; every number here comes from an official report cited there.

---

## 1. The gap we set out to close

Language models know what they were trained on. When the world adds a word — a new drug, a new company term, a
chemical nobody wrote about before the model's training cut-off — the model has three bad options: retrain (slow,
expensive, and it forgets other things), be told the fact in every prompt (costly, and the prompt fills up), or guess.

Knowledge bases have the opposite strengths. A dictionary entry or an ontology record says exactly what a term is
related to: *penicillin is an antibiotic; it has the role of a drug; it contains the elements C, H, N, O, S.* Facts
like these are precise, editable and auditable, but a language model cannot read a database row.

**The programme asks: can we hand a language model structured facts about a term in a form it can use — a single
vector built from the term's facts — so that it learns new words from their definitions, keeps its knowledge
editable, and can even run simple logic over the facts?**

## 2. What we built, in plain words

- **A fact store made of vectors.** Each relation ("is a", "has role", "contains element") and each partner concept
  gets a random code. A fact is the two codes *bound* together (a fixed mathematical operation, circular convolution);
  a term's whole record is the sum of its facts. The construction is called a vector-symbolic architecture (VSA), or
  holographic reduced representations (HRR). Its two useful properties: you can *unbind* to read a fact back out, and
  one fixed-size vector holds many facts.
- **The method ("C5").** During training, wherever a known multi-word term appears in the text, we add its record
  vector at the term's last token. The hope is that the model learns to read these vectors the way it reads words.
- **Routes for using a fact.** *Implicit*: the vector is added inside the model. *Explicit* (recall): the facts are
  decoded back into short text lines and placed before the question, where the model can read them.
- **An executor.** A small program that runs logic (rules, deductions, decision trees) over the store exactly, and
  writes its conclusion in text.
- **Readers.** Programs, or the model itself, that turn a definition into facts for the store.

## 3. How we test (the methodology)

- **Everything is pre-registered.** Before a run, we write down the question, the conditions, the exact comparison,
  the prediction and a decision rule ("if X, we do Y; if not, we stop"). Results are read against those rules. We also
  record when a prediction fails.
- **Controls that matter.** The main one is a *free table*: a model given one freely learned vector per term, with no
  structure in it. If the structured vector does no better than the free table, the structure is not what helps.
  Others: shuffled facts (same vector size, wrong content), no vector at all, the definition text in the prompt.
- **Three kinds of test words.**
  - **T5, invented words:** a made-up company glossary. The model has never seen these words, so anything it knows
    comes from us. Clean, but artificial.
  - **T4, real chemistry:** ChEBI entries and PubMed text. Real facts, but the model has seen much of the vocabulary.
  - **T7 / T7-ROOD, new medical terms:** 2026 MeSH names in 2025–26 abstracts. T7-ROOD removes every training
    abstract that mentions a held-out name, so those names are truly unseen. This is the decisive real-world test.
- **What we measure.** Mostly *loss after the term*: how well the model predicts the next few words right after a
  term it was given facts about; lower is better, reported as a percentage change against the control. For reading
  tests, accuracy on multiple-choice questions ("which of these is penicillin a kind of?"). Differences come with
  confidence intervals from resampling, and corrections for multiple comparisons.
- **Scale.** One machine with one GPU. Models from 50M to 1.7B parameters. About 4,000 jobs run through a queue.

## 4. What we ran, and what we found

### 4.1 Training small models from scratch with the method (E4)
Models of 20M–125M parameters trained on 500M tokens, with and without the vector. **Result:** a small, real gain
(about 0.3–0.4% lower loss overall, 1% after long terms), which does not grow with model size and does not appear for
held-out terms. Equal to or barely above the free table. A Claude-written ontology (C5teach) changed nothing. **The
method is a modest, non-scaling improvement for seen terms when training from scratch.**

### 4.2 Adding the method to pretrained models (E9)
Fine-tune SmolLM2 (135M, 360M) and Qwen3 (0.6B, 1.7B) with the vectors. **On invented words the effect is large**
(9–13% lower loss after the term). **On real new words it is 1–2%,** and it does not help after a word the model has
truly never seen (T7-ROOD, preliminary: +0.1%, i.e. nothing). Shuffling the facts costs almost nothing, so the model
is using "a vector for this term", not the facts inside it. The gain survives 4-bit quantization, and editing a fact
changes behaviour on invented words. **The vector works as a per-term memory; the model does not read its structure.**

### 4.3 Does the model read the algebra? (the binding programme)
Direct tests of whether the model unbinds roles from the vector, including models trained with an extra readout head.
**It does not,** even when trained to. The algebra behaves as designed in the store; the model treats the vector as
opaque. Editing frames does change behaviour, but no better than for a vector without binding.

### 4.4 Learning a new word by reading its definition (E11, E14)
The full pipeline: a reader turns a definition into facts, the facts go into the store, the model is tested with the
definition gone. We traced where facts are lost:
- **Definitions state only about 12% of the facts an ontology holds.** No reader can recover what the text does not
  say. Wikipedia pages are no richer (12.0% vs 11.6%).
- **Simple statistics beat smart readers.** A frequency table of "which relation does this kind of partner usually
  fill" beats a step-by-step reader on the model, and Claude is only 3 points better than the table.
- **The model cannot use facts hidden in the vector, but it can use them written as text.** Recalling the facts into
  the prompt lifts accuracy on real chemistry from 0.61 to 0.87 with correct facts (0.72 with facts a realistic reader
  produces, about equal to having the definition itself), and on invented words from 0.19 to 0.85.
- **Wrong facts hurt more than missing ones.** Facts inferred by analogy (55% precision) do not help; neither does
  more text per term.
- **Repeated reading adds a few facts; training on them adds nothing.** The cycle of "read, train, read again" does
  not build knowledge beyond what the explicit recall already gives.
- **A correction found on 21 October:** an evaluation setting cut long prompts at 256 tokens, which made recalling a
  term's *whole* record look harmful. Re-measured, whole-record recall is as good as recalling only the asked fact, so
  no "routing" step is needed for records of normal size.

**The method's genuine use for new words is as an editable fact store read through the prompt, not through the
weights.**

### 4.5 Logic in the store (E14, part 2)
Can the vector store run rules and decision trees, and can a model use the result?
- **The store is an exact executor** of rules and trees (up to 100,000 clauses), but it is never better than ordinary
  symbolic logic. Its advantage is uniformity: one memory for facts and rules, decoded to text.
- **Small models cannot run a decision tree written in prose** (at chance), and cannot read one from paraphrased text
  into structure (0.24 accuracy). They *can* copy an executor's conclusion from a short, structured context.
- **Consistency under "what if" edits.** Across chains of 8 edits to a tree or its input, the executor stays right
  (99.7% of chains fully consistent) while the model answering in context is at chance throughout (3–5%). The honest
  wording: the executor supplies the computation and the model relays it; this is not evidence that models drift,
  because they never ran the tree in the first place.
- **The model can guide a prover.** Qwen3-1.7B, asked to break ties in a symbolic proof search, finds proofs faster
  than either alone (0.20 → 0.34 within a small budget). All proofs remain sound.

### 4.6 Self-authoring and self-learning (E7, E10)
Can a model write its own ontology, or repair one? **No.** Frames written by a small model, or by Claude, on general
WordNet text are no more precise than degree-matched random frames; a learnable ontology recovers planted structure
in synthetic worlds but nothing transfers to real WordNet.

### 4.7 Explainability (E5)
At 125M: a structure-only vector for an unseen concept does lift question accuracy (0.52 → 0.67), but attention over
facts is only weakly faithful, sense alignment fails, and the vectors carry *more* word-frequency information than the
baseline.

## 5. What it means (implications)

1. **Composing facts into a vector does not teach a language model those facts.** Across every scale and host we
   tested, the model uses the vector as an opaque per-term memory; it never reads the structure. Gains on real text
   are 1–2% after seen terms and zero after truly new ones. This closes the strong version of the hypothesis.
2. **The structure is still useful — outside the weights.** Decoding the stored facts into the prompt gives large,
   reliable gains for new words (+0.25 real chemistry, +0.66 invented words), with every used fact visible, editable
   and deletable, and without retraining. Whole-record recall needs no routing step.
3. **Text is the ceiling for reading.** Definitions and encyclopaedia text state about one fact in eight. Better
   readers, bigger dictionaries, more rounds and more text all failed to move this. Precision of what is stored matters
   more than quantity.
4. **An exact executor gives small models a capability they lack entirely**: running a rule or decision tree
   consistently as facts change. But the model only relays; it does not reason over the vector.
5. **Models cannot write their own knowledge bases** at these sizes, and a frontier model's frames are not more
   precise than the ontology's own statistics.
6. **For the manuscript:** the claims that survive are modest and specific — an editable, auditable fact store read
   through the prompt; an exact executor with a model-readable interface; a measurable but non-scaling training gain
   for seen terms; and a set of carefully controlled negatives that other groups can build on.

## 6. What is still running

- **T7-ROOD** (the clean new-word test): 14 of 253 jobs done; preliminary numbers above; full report about 23 Oct.
- **E13, the whole read–learn–write cycle** on invented words (about 25–26 Oct) and on T7-ROOD (about 30 Oct). It is
  the only experiment that tests the toolkit as a whole and could support a "learns faster" claim.
- **T8 (Wikidata), T2 (developer tools), B-B1 (a size-matched control for the quantization result), T1c-ROOD
  (clinical, aggregates only)**: late October to early November.

## 7. Terms in one line each

- **VSA / HRR:** a way to pack many (relation, partner) facts into one vector so that each can be read back out.
- **Frame / record:** a term's set of facts. **Ontology:** all frames.
- **C5:** the method (the record vector added to the term). **C0′:** the same model trained without it. **C2:** a free
  vector per term with no structure. **C5sh:** the method with the facts shuffled. **P0:** the untouched pretrained model.
- **Implicit route:** the vector inside the model. **Explicit route / recall:** the facts written as text before the
  question.
- **Loss after the term:** how surprised the model is by the words that follow a term; lower is better.
- **T4, T5, T7, T7-ROOD:** real chemistry; invented words; new medical terms; the same with every mention of the
  held-out terms removed from training.
- **Pre-registration:** the written plan, predictions and decision rules fixed before a run.
