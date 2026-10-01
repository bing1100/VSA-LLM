# Formulation: contextual, factored and growable VSA composition

Working name for the mechanism set: **CG-VSA** (contextual, growable VSA). Each mechanism is written so that switching it off recovers the previous one exactly; that is what makes the ablations in [experiments.md](experiments.md) clean. Section numbers M1–M5 are referenced from [proposal.md](proposal.md).

## Notation

```text
C            concept set (ontology nodes: synsets, SNOMED concepts, API symbols, ...)
A, R         atomic and relation dictionaries; a_k, r_k ∈ ℝ^d
F_i          frame of concept i: list of edges e = (r_e, a_e); |F_i| = degree
T_r          relation transform for r (family choice, §0.2); T_r^⊤ its adjoint
v_e          bound edge vector  v_e = T_{r_e}(a_e)
N(x)         L2 row normalization x / ‖x‖
P            projector ℝ^d → ℝ^{d_m} into the host model width (identity if d = d_m)
q            context query (∅ for static composition)
E[x], p_t    host token embedding and position embedding
```

## 0. Base composition and the operator ladder

### 0.1 Static bundle (M0, what exists)

```text
c_i = N( Σ_{e ∈ F_i} v_e )                      HRRBERT eq. 2 with v_e = r_e ⊛ a_e
```

This is the gather → bind → scatter-add of `FastCVGen` ([vsa_utils.py](../models-main/medair_models/bertha/vsa_utils.py)) and `OntologyFactorizer.compose` ([factorization.py](../../src/vsa_embed/factorization.py)). Two fixes are carried into every mechanism below: normalize rows (`dim=1`), and accumulate with `index_add_`/segment reductions rather than indexed `+=`, which drops duplicate destinations.

### 0.2 Relation transform families (the operator is an ablation axis)

| Family | `T_r(x)` | Params / relation | Adjoint `T_r^⊤` | Status in repo |
|---|---|---:|---|---|
| `hrr` | `r ⊛ x` (circulant) | d | circular correlation (`HRRAlgebra.unbind`) | implemented |
| `unitary_hrr` | as above, unit-magnitude spectrum | d | exact inverse | implemented |
| `diagonal` / `map` | `r ⊙ x` | d | same diagonal | implemented |
| `low_rank` | `x + L_r R_r x` | 2·d·k | `x + R_r^⊤ L_r^⊤ x` | implemented |
| `bounded_residual` | `x + o_r + c_r · N(r̂ ⊛ (x ⊙ s)) ⊙ s⁻¹` | ~2d+2 | via autograd | implemented (01c) |
| `orthogonal` | `Q_r x` | d² (QR) | `Q_r^⊤` | implemented |
| `random_fixed` | any of the above, frozen at init | 0 trainable | — | control |
| `untyped` | `x` | 0 | identity | control (01a additive) |

The 01c result says fixed circular convolution added nothing over a bounded relation-conditioned correction for one-hop reconstruction of frozen GPT-2 space. Nothing below depends on which family wins; every experiment reports the family as an ablation.

## 1. M1 — Attentive composition (query-conditioned mapping)

**From idea.md:** the composition `Σ (relation × mapping) ⊛ atomic` "is similar to attention"; a query should enter the formulation so context can modify the mapping.

### 1.1 Definition

```text
k_e        = W_K^r r_e + W_K^a a_e                 two-part edge key, ℝ^{d_k}
c̄_i        = Σ_{e ∈ F_i} v_e                        the concept's own static bundle
ū_i        = W_Q c̄_i                                frame query (no per-concept parameters)
u_i(q)     = ū_i + W_C q                            context-conditioned query
s_{i,e}(q) = ⟨u_i(q), k_e⟩ / √d_k
w_{i,e}(q) = |F_i| · softmax_{e ∈ F_i}( s_{i,e}(q) / τ )      mass-preserving weights
c_i(q)     = N( Σ_{e ∈ F_i} w_{i,e}(q) · v_e )
```

Concept `i`'s frame is a small key–value memory whose values are the bound edges and whose keys are built from the edge's relation and filler. The query is the concept's own bundle plus a projection of the context.

### 1.2 Properties

- **Exact recovery of M0.** `τ → ∞` gives `w = 1` and `c_i(q) = c_i`. With random near-orthogonal atomics, `⟨c̄_i, v_e⟩ ≈ const` for all `e`, so even at finite `τ` the weights start uniform: training departs from the VSA prior rather than replacing it. A temperature curriculum (large τ → learned τ) is the default.
- **Static learned mapping (P0).** With `q = ∅` the mapping is learned but has **zero per-concept parameters**, so it applies unchanged to a concept never seen in training. This is the answer to "make the mapping itself learnable" that stays compatible with zero-shot composition.
- **Bundling, unbinding and attention are the same operation.** Uniform attention is bundling. Querying with a relation probe `W_K^r r` selects edges of type `r`; unbinding their weighted sum with `T_r^⊤` and cleaning up against `A` reads out the filler. The readout head of experiment E6 is this mechanism with a relation probe instead of a context query.
- **Explanation.** `w_{i,·}(q)` is a distribution over ontology edges for this occurrence: "in this sentence, `bank` is 0.7 `is_a → financial_institution`, 0.2 `part_of → river`".
- **Multi-sense frames.** For a surface string with senses `{i_1..i_m}`, use the union frame `F = ∪_j F_{i_j}`. The query selects; sense attribution is the attention mass per sub-frame. Splits in M3 create new sub-frames automatically.
- **Variants for ablation.** Relation-level keys (`W_K^a = 0`); independent sigmoid gates `w = 2σ(s)` instead of the softmax; `H ∈ {1, 2, 4}` heads, averaged.

### 1.3 Where the query comes from

| Placement | `q_t` | Cost | Use |
|---|---|---|---|
| P0 static | `∅` | none | zero-shot rows, table compression |
| P1 embedding-layer local context | causal depthwise conv or mean over `E[x_{t−w..t}]`, `w ≈ 8–16` | negligible | default for the span channel (§4) |
| P2 mid-network sidecar | `h_t^{(ℓ)}`, injected into the residual stream at layer `ℓ+1` at the span's last position | one extra cross-attention head | sense selection that needs syntax; experiment E6 |

### 1.4 Cost

Per linked occurrence: `O(|F_i| (d + d_k))` plus one FFT per edge if binding is not cached. With mean degree ≈ 10 and `d = 512` this is ≈ 10⁴ multiply-adds, against ≈ 2N per token in an N-parameter transformer.

## 2. M2 — Factored ("LoRA-style") mapping

**From idea.md:** make the mapping learnable by an SVD/LoRA-like factorization `A × B`.

### 2.1 Matrix view

The ontology defines a binary mapping `M ∈ {0,1}^{|C| × |E|}` over global edge types `E = R × A`. M0 uses `M` as-is. A dense learnable `M` is impossible at scale and cannot transfer to a new concept (its row would be untrained). The rank-`k` factorization on the ontology support is

```text
W_{i,e} = ⟨u_i, k_e⟩ · M_{i,e}          U ∈ ℝ^{|C| × k},   K ∈ ℝ^{|E| × k}
```

`K` never needs `|E|` rows: `k_e` is composed from `r_e` and `a_e` (§1.1), so its parameters are `W_K^r, W_K^a` plus the dictionaries. The concept factor has three options:

| Option | `u_i` | Transfers to unseen concept `j`? | Role |
|---|---|---|---|
| (a) free | learnable row per training concept | no (`u_j` untrained) | memorization upper bound, LoRA on the row |
| (b) induced | `ū_i = W_Q c̄_i` (§1) | yes | default |
| (c) hybrid | `ū_i + δ_i`, `‖δ_i‖²` penalized, `δ_j = 0` for unseen `j` | yes, without `δ` | the "restricted concept residual" of experiment 01b |

Weights are passed through the mass-preserving softmax of §1 (default) or used raw (ablation). Rank sweep `k ∈ {4, 8, 16, 32}`.

### 2.2 Parameter accounting

```text
global      W_Q, W_K^r, W_K^a, W_C          O(d · d_k)
dictionary  (|A| + |R|) · d  +  Σ_r params(T_r)
concept     |C_train| · k                   options (a)/(c) only
```

Reported separately, as [shared-protocols.md](../vsa-understanding/experiments/shared-protocols.md) requires.

## 3. M3 — Developmental dictionary (split / merge / allocate)

**From idea.md:** track the momentum of the gradient and the mean absolute gradient per vector; if momentum ≈ 0 while the mean absolute gradient is high, gradients pull in different directions and the vector should be split into two copies that start identical and drift apart.

The 01d design ([01c README §"developmental"](../vsa-understanding/experiments/01c-developmental-relation-discovery/README.md)) specifies routing, merging and phases but states no split statistic and was never implemented. This section supplies the statistic, its justification, and the assignment rule that makes a split actually break symmetry.

### 3.1 Per-usage gradients are free

Let `z_i = Σ_e w_{i,e} v_e` (before normalization) and `g_i = ∂L/∂z_i`, which autograd produces anyway (it is the gradient of the generated row). Ignoring the path through the weights,

```text
atomic a used by concept i through edge e:   ∂L/∂a |_i = w_{i,e} · T_{r_e}^⊤ g_i
relation r used on edge e = (r, a_e) of i:   ∂L/∂r |_e = J_{T}(a_e)^⊤ g_i        (hrr: correlation with a_e)
```

For `hrr`, an atomic's gradient from a concept is the concept's gradient **unbound** by the relation. Usage-level gradients therefore cost one FFT per edge, batched with the forward schedule; no per-sample gradient machinery is needed.

### 3.2 Statistics

For dictionary vector `θ` with usage set `U(θ)` (concepts for atomics, edges for relations):

```text
g_t^{(u)}   per-usage gradient at step t (0 if u absent from the batch)
m^{(u)}     = EMA_β( g^{(u)} )                          usage momentum
m           = Σ_u m^{(u)}                               total momentum (Adam's first moment of θ, same β)
A           = Σ_u |m^{(u)}|      (elementwise)          usage-summed absolute momentum
a           = EMA_β( |g_t| )     (elementwise)          idea.md's mean absolute gradient (usage-agnostic)
κ           = ‖m‖₁ / ‖a‖₁  ∈ [0, 1]                     coherence
α           = ‖a‖₁                                      activity
```

**Screening (idea.md's criterion, zero extra cost).** `θ` is a candidate if `α` is above the `p`-th percentile of active vectors and `κ < τ_κ`. Both numbers can be read from Adam's state: `κ_j ≈ |m̂_j| / √v̂_j` is Adam's per-coordinate signal-to-noise ratio, so a candidate is precisely a vector Adam cannot move despite strong pulls.

**Noise floor.** For zero-mean i.i.d. gradients, `E[κ] ≈ √((1−β)/(1+β))`: 0.23 at `β = 0.9`, 0.07 at `β = 0.99`. A converged parameter under minibatch noise looks the same as a conflicted one on this statistic. Screening alone therefore cannot decide; it only bounds the bookkeeping of §3.3 to a small candidate set.

### 3.3 Split gain: confirmation and direction

Take a unit direction `v` and a step of size `ε`.

```text
shared vector, best sign:            loss decrease  ε · |⟨m, v⟩|  =  ε · |Σ_u ⟨m^{(u)}, v⟩|
two children, usages split by sign:  loss decrease  ε · Σ_u |⟨m^{(u)}, v⟩|
split gain                           G(v) = Σ_u |⟨m^{(u)}, v⟩| − |Σ_u ⟨m^{(u)}, v⟩|   ≥ 0
```

`G(v) = 0` iff every usage agrees in sign along `v`. Summing `G` over the coordinate axes gives `‖A‖₁ − ‖m‖₁`: **idea.md's "mean absolute gradient minus momentum" is the coordinate-wise split gain**, and replacing `A` by the per-step `a` is its usage-agnostic approximation. The refined criterion chooses one direction:

```text
S   = Σ_u m^{(u)} m^{(u)⊤} − (1/|U|) m m^⊤        between-usage scatter
v*  = top eigenvector of S,   G* = G(v*)
```

and accepts the split only if `G*` exceeds the 99th percentile of a permutation null (usage labels shuffled over the accumulated per-step contributions) and the budget of §3.6 allows. A converged vector has isotropic between-usage scatter and fails the null; a conflicted one does not.

**Children.** `θ_± = θ ± ε v*`; usage `u` goes to the child matching `sign⟨m^{(u)}, v*⟩`; keys, optimizer state and frames are duplicated then partitioned. Starting both children at exactly `θ` with the same usages, as idea.md suggests, would give them identical gradients forever; the partition and the `±ε v*` offset are what break the symmetry.

A second-order alternative (splitting along the minimum eigenvector of a Hessian-like splitting matrix) is a comparison baseline in E0.2, not the default, because it needs Hessian-vector products per candidate.

**Credit.** Offsetting two copies along a direction is the Linde–Buzo–Gray codebook split and the neuron split of Splitting Steepest Descent (Liu et al.); deciding a split by a test along a principal direction is G-means (Hamerly & Elkan 2003). What M3 adds is the usage-level first-order statistic read from optimizer state, the partition of *usages* (not data points) by sign, and the permutation null; see [related-work.md](related-work.md) §4. An Anderson–Darling variant of the test is an E0.2 comparison.

### 3.4 Usages without gradient signal

Rare, held-out and new concepts have no `m^{(u)}` and are not assigned by sign. They route by their frame context: the mean of the concept's other fillers (the split atomic left out) is compared with the same feature averaged over each child's observed usages, and the nearer child wins (implemented; E0.2 showed that routing by the concept's full composed vector fails, because trained concepts' compositions are fitted to targets that already contain the sense). With attentive composition the same comparison can use the frame query of §1, `p(θ_s | i) = softmax_s ⟨ū_i, k_{θ_s}⟩`, soft during a commitment window, then hard. This keeps zero-shot composition working after splits and reuses the 01d routing idea with a bounded sibling set instead of a global expert pool.

### 3.5 Merge, prune, allocate

- **Merge.** `cos(θ_1, θ_2) > τ_m` and `G` over the union of usages ≈ 0 (sharing costs nothing to first order) ⇒ `θ ← ` usage-weighted mean; budget refunded.
- **Prune / freeze.** `α` below the floor for `T` steps ⇒ freeze, keep for provenance, drop optimizer state.
- **Allocate.** A frame element with no dictionary entry (new filler or relation from an ontology update or a text-to-frame compiler) ⇒ new vector initialized from an anchor (definition or subword mean through `P⁺`, or nearest dictionary entry), flagged provisional until `α` rises above the floor.

### 3.6 Budget and schedule

MDL-style cap: a split costs `d` parameters plus reassignment bits; at most `B` splits per screening round, total growth ≤ `γ` (default 25%) of the initial dictionary, cooldown `T_c` steps for new children, splits accepted in descending `G*`. Phases: warm-up (no splits) → growth (screen every `T_s` steps) → consolidation (merges only) → freeze. Every event is logged as a **card** (parent, children, `v*`, `G*`, p-value, usage partition, step). The cards are the explainability artifact for discovered senses and relation sub-types.

### 3.7 Pseudo-code

```text
every T_s steps during growth:
  cand ← { θ : α_θ ≥ pct_p(α), κ_θ < τ_κ }              # from Adam state
  for θ in cand:                                          # usage-level EMAs kept only for cand
      S ← between-usage scatter;  v* ← top eigenvector;  G* ← G(v*);  p ← permutation null
  accepted ← top-B by G* with p < 0.01, within budget and cooldown
  for θ in accepted:
      split θ along v*; partition observed usages by sign; route the rest by §3.4
      duplicate keys and optimizer state; log card
during consolidation:
  merge pairs with cos > τ_m and G_union ≈ 0; freeze vectors with α below floor
```

## 4. M4 — Span-level semantic channel

**From idea.md:** add the VSA vector to the sequence like a position embedding, and create it only for words the tokenizer fragments into several tokens.

### 4.1 Linking

Deterministic longest-match over alias tables (WordNet lemmas, SNOMED descriptions, API symbol tables), minimum span length `ℓ_min` (default 2 subtokens), optional single-token track for words below a frequency threshold. Polysemous strings link to the union frame (§1.2). Output per span: `(s, e, concept-or-lemma, confidence)`. The linker is fixed, not learned; a learned span proposer is a later extension; the self-authoring loop of §5 adds aliases to it between rounds.

Matching runs on **character offsets**, so one linked corpus serves every tokenizer (GPT-2 BPE for the from-scratch models, the SmolLM2 and Qwen2.5 tokenizers for the pretrained hosts); span length in subtokens, and therefore membership in the `ℓ ≥ ℓ_min` set, is computed per tokenizer.

**Span cardinality (feasibility).** For each tokenizer × ontology × corpus and each threshold `ℓ_min ∈ {1, 2, 3, 4+}`, the linker reports: distinct linkable concepts, distinct linked concepts in the corpus, span occurrences, fraction of corpus tokens inside linked spans, the concept-frequency histogram (occurrences per concept), and the implied channel cost (concepts generated per batch, schedule size). This is the feasibility table for the channel: a threshold that leaves too few concepts or too little covered text cannot produce a measurable effect, and one that covers most tokens is no longer a rare-word channel. Gains in E4 are reported against the same `ℓ` strata.

### 4.2 Injection (causal-safe)

```text
h_t^{(0)} = E[x_t] + p_t + 1[t = e] · g_t · P c_j(q_t)
g_t       = σ( w_g · [ E[x_t] ; P c_j(q_t) ; confidence ] + b_g )
```

**Last-subtoken rule, made prefix-causal (correction found in implementation, B5).** At position `e` every subtoken of the word is visible, but *deciding* that a linked word ends at `e` is not: whether "bank" is a word or the start of "banking", and whether "new york" or "new york city" is the longest alias, depends on token `e+1`. Injecting only when the word ends would therefore reveal the next token through the presence of an injection. The implemented linker (`span_channel.py`) is prefix-causal: an alias is linked at `e` if the text up to the end of `e` ends with it and it starts at a word boundary, taking the longest alias ending at `e` (looking back only); the right word boundary is not checked, so "bank" is also linked inside "banking" and the gate must learn to discount such links. A second mode checks the right boundary and injects at `e+1`, one step later. With either rule the injected vector at position `e` is a function of `x_{≤e}` only (unit-tested). Injecting at `s`, or on all span tokens, would reveal `x_{s+1..e}` and is forbidden for causal LMs (it is fine for encoders, where BERTHA injects on all tokens). The gate lets the model discount bad links. Secondary tracks: a virtual extra token after the span (changes positions and lengths) and the P2 sidecar.

### 4.3 Output side (optional, E4b)

```text
L_sem = InfoNCE( W_o h_{s−1}, c_j ; negatives = other linked concepts in the batch )
```

applied at positions that precede a linked span: predict the meaning before spelling it. It gives the LM a semantic target for multi-token words instead of only subword targets, and yields a concept-level generation metric (rank of `c_j` among concepts). Native generation of a *new* token remains the reserved-token track of dossier experiment 02.

### 4.4 Table-compression variant (E4c)

For single-token concepts `x` that have a frame: `E[x] ← P c_x + δ_x`, with `δ_x` stored at low precision or omitted. Bytes = dictionary + schedule + residual, compared against `|V| · d · bits` at equal quality.

### 4.5 Parameter and byte accounting

```text
channel            (|A| + |R|) · d + Σ_r params(T_r) + 3 · d · d_k + d · d_m + gate
free-entity control |C_train| · d_m                 (KnowBERT / LUKE-style span vectors)
```

For orientation: in Qwen2.5-0.5B the tied embedding table is 151,936 × 896 ≈ 136M of ≈ 494M parameters (≈ 27%); in SmolLM2-135M it is 49,152 × 576 ≈ 28M (≈ 21%). Verify against the checkpoint configs before quoting.

## 5. M5 — Self-authored ontologies (pretrained regime)

**From the clarifications:** a pretrained model can read text, generalize what it reads into an ontology (its own frames), and use those frames to improve its own weights. M1–M4 consume a curated ontology; M5 lets the host write, verify and extend it.

This regime needs a model that can already generate sensible text, so it applies to the pretrained hosts (SmolLM2-135M/360M, Qwen2.5-0.5B). From-scratch small models are **consumers** of authored ontologies (cross-authoring, §5.4), not authors.

### 5.1 The loop

```text
round r = 0, 1, …, R_max
  1 discover   candidates J_r: spans with ℓ ≥ ℓ_min subtokens, frequency ≥ f_min in the reading corpus D_read,
               not linked (or linked below a confidence floor), ranked by excess surprisal
               Δ_j = mean over occurrences of Σ_{t ∈ span} −log p(x_t | x_<t)  −  frequency-matched baseline
  2 author     for each j, n authoring contexts from D_read; the host proposes edges (r, a) with a constrained template:
               r ∈ R (closed vocabulary; "new relation" in the open ablation), a ∈ aliases(A) ∪ {new filler span};
               K samples per context
  3 verify     keep edge e for concept j only if
               (i) self-consistency: proposed in ≥ ρ of samples pooled over contexts,
               (ii) held-out utility: U_{j,e} (§5.2) on validation contexts of j (disjoint from the authoring
                    contexts) has a lower confidence bound > 0,
               (iii) no gold leakage: gold-audit sets are never used for acceptance
  4 integrate  accepted frames enter the linker as provisional (surface forms observed in D_read become aliases);
               new fillers and relations are allocated through M3 §3.5
  5 train      channel parameters (and optionally a LoRA on the host) on D_read plus a replay of general text;
               M3 active, so authored atomics that conflict are split rather than averaged
  6 audit      locality (general-text loss), gold-edge precision/recall where a gold ontology exists, drift of
               earlier frames; stop when the median U of newly accepted edges ≤ 0 or locality fails
```

### 5.2 Why the utility test is cheap

Adding a candidate edge `e` to concept `j` with a small mass `β` changes the pre-normalization row to `z_j + β v_e`. To first order its effect on the loss is

```text
U_{j,e} = −∂L_val/∂β |_{β=0} = −Σ_{occurrences of j in validation contexts} ⟨ g_j, v_e ⟩        (g_j = ∂L/∂z_j, §3.1)
```

`g_j` is what the backward pass of §3.1 already produces, and `v_e` is one bind, so every candidate edge of every candidate concept is scored from a single validation pass; a bootstrap over occurrences gives the confidence bound. *(Implementation note, B12: a brand-new concept starts with an empty frame, where `z_j = 0` and the expansion through the row normalisation is undefined, so `authoring.py` scores edges by the exact loss difference on the validation occurrences — with and without the edge — rather than by this first-order estimate.)* No per-edge retraining is needed. Edges that pass enter the frame with the normal M1 weights and are trained. Edges already in a frame are re-scored the same way (utility of their current contribution) each round; negative utility in two consecutive rounds removes the edge.

### 5.3 Guards against self-confirmation

A model that writes its own training signal can reinforce its own errors. The guards: authoring and validation contexts are disjoint; acceptance uses held-out loss, never the model's own confidence alone; general-text replay and the locality gate bound drift; gold-audit edges (WordNet, SNOMED CT, API schemas) are hidden from the loop and used only to measure it; every accepted edge is logged as an **authoring card** (concept, edge, source contexts, samples, `U` with CI, round), which is also the explainability artifact for self-authored knowledge.

### 5.4 Regimes and controls

| Author → consumer | Meaning |
|---|---|
| gold → host | curated ontology (M1–M4 as in E4); upper reference |
| self → self | the host authors for its own channel (the claim) |
| self, no verification | step 3 reduced to self-consistency; measures what verification buys |
| teacher → host | a frontier model (Claude Code, headless) authors; upper bound on authoring quality |
| self → from-scratch 125M | cross-authoring: does a pretrained model's ontology help a model trained from scratch? |
| random frames → host | matched edge counts and degree distribution; isolates capacity |
| none, compute-matched | continued pretraining on `D_read` with extra tokens equal to the FLOPs of authoring + verification |

The compute-matched control is what makes "self-improvement" a claim about the ontology rather than about extra passes over the text.

### 5.5 Open vs closed relation vocabulary

Default closed: relation labels come from the target ontology (WordNet pointers, SNOMED attributes, API schema fields), so authored frames compose with curated ones and can be scored against gold. Open ablation: the host may name a new relation, which is allocated as a new relation vector keyed by its surface string and is subject to M3 merge (duplicate relations with `G_union ≈ 0` merge back).

### 5.6 Cost

Authoring: `n · K` generations of ≈ 32 tokens per candidate, ≈ `2 N` FLOPs per generated token; verification: one forward/backward over validation contexts. Both are counted in the compute-matched control.

## 6. Training objective

```text
L = L_LM + λ_sem L_sem + λ_δ Σ_i ‖δ_i‖² + λ_τ R_τ + (growth budget of §3.6)
```

`R_τ` is an entropy regularizer that keeps attention from collapsing onto one edge early. Two regimes, both tested: **joint** (transformer and channel trained from scratch together, BERTHA's regime, where the only positive VSA evidence exists; the pre-registered core claim) and **continued pretraining** (pretrained host frozen or LoRA-adapted, channel trained; the regime in which M5 self-authoring is possible). The frozen-host, no-training regime of 01a–01c is not a target of this proposal.

## 7. Efficiency

Generate only the concepts present in the batch (unique linked spans) with a CSR schedule: gather atomics, bind in the FFT domain, segment-softmax the weights, segment-sum into concept rows. Full-vocabulary generation is needed only for the tied-output and compression variants; there the FFT of the dictionary is cached and invalidated when atomics change. Per-usage gradient statistics (§3.1) reuse the same schedule in reverse.

## 8. What this formulation does not solve

Flat one-hop frames (no paths, negation, time, cardinality); authored-frame correctness beyond what held-out utility and gold audits can detect (a frame can lower loss and still be wrong); superposition capacity at high degree (experiment 00: cap `|F_i|` by salience or shard); linker errors and coverage; and the operator question, which the joint regime must re-test rather than assume.
