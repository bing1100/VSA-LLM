# 06 — E9 core: adding the method to pretrained language models

E9 asks whether a pretrained language model, fine-tuned together with the method, predicts text better after multi-token domain terms. On invented words it clearly does: loss after held-out terms falls by 9–13% on SmolLM2 and 18–20% on Qwen3. On real text the gain shrinks to about 0.4–2.3%, and it appears only where the words are new to the model. The mechanism analyses narrow what we can claim. The binding operator is not the source. On real text a TransE vector or an encoded definition does as well as composition, and a wide enough free table matches it after long terms. Composition keeps two advantages on real text: it wins clearly for unseen and rare terms, and it needs about 7× fewer parameters than the wide table. Real-text gains are not copying. The method adds on top of a definition placed in the prompt. Smaller hosts gain more on real text. Injecting a vector for a word the model already learned without one costs a little, and link dropout does not remove that cost. Several results here are new since 15 October and **preliminary**: they come from generated reports that are not yet committed or summarized in R9. These are the hybrid (H5add), random hashed sharing (C1h), T7 at 135M with 3 seeds, and the strict non-copy and understanding tests. The clean zero-shot test (T7-ROOD) and the richer-ontology runs are queued; none has started.

**Codes used in this file** (as in `manuscript/update-2026-10-12.md`, "How to read the numbers", and `src/vsa_embed/experiments/e9_plan.py`):

| Code | Meaning |
|---|---|
| host | the pretrained language model being fine-tuned: SmolLM2-135M or -360M (full fine-tuning), Qwen3-0.6B or -1.7B (LoRA r 64) |
| entry, frame | an ontology record (e.g. a ChEBI compound); its frame is the set of its relation–filler edges (*is-a*, *contains element*, …) |
| **C5** | the method: a vector composed from the entry's frame over a small shared dictionary trained with the host, added at the last subtoken of each linked multi-token term. Default binding: **HRR** (holographic reduced representation, circular convolution) |
| **P0** | the original host, not trained further |
| **C0′** | the host fine-tuned on the same text and tokens without the method; the fair baseline |
| **C2** | a free lookup table, one learned vector per entry, no ontology, matched to C5's parameter budget (rows 34 wide on T4, 119 on T7) |
| C5rf / C5ut / C5tr | C5 with a fixed random orthogonal binding / no binding (a plain sum of fillers) / translation (relation added, x + r) |
| C5sh | C5 with shuffled frames: every entry reads another entry's frame |
| C6m / C6d / C6g / C6gi | a frozen per-entry vector through a trained, parameter-matched projector at C5's position: subtoken mean / definition encoder (the frozen host's encoding of the frame written as text) / TransE embedding trained with the held-out entries in its graph / TransE trained without them, held-out entries folded in afterwards |
| C2w | C2 with rows at C5's width (256); **not** parameter-matched |
| H5add | hybrid (HRRBERT's "compose-add"): a half-width composed vector plus a free per-entry row, matched to C5's parameters |
| C1h | hashed memory: terms share rows at random, matched to C5's parameters |
| C5ld | C5 with link dropout: in training, each linked term gets no vector with probability 0.3 |
| C5isa, C5comp, C5half, C5w512, C5mesh, … | ontology-richness arms, defined in 06.12 |
| held-out terms | kept out of the method's training on purpose; at test time their vector is composed from the frame alone ("zero-shot"). On T5 they never occur in the training text; on T4, T7 and T1c they still occur there, unlinked |
| unseen / rare / long terms | in the ontology but absent from the training text / seen 1–9 times / three or more subtokens |
| T1–T8 | test domains ("tracks"). **T1-open**: MeSH terms in PubMed; **T1c**: clinical notes (MIMIC-III, licensed) with SNOMED CT; **T2**: developer tools; **T3**: product catalogues; **T4**: chemistry (ChEBI, PubMed abstracts); **T5**: synthetic company glossary (invented words, templated text); **T6**: legal (EuroVoc, EU law); **T7**: PubMed 2025–26 abstracts with 2026 MeSH supplementary names; **T7-ROOD**: T7 with every training document that names a held-out entry removed; **T8**: Wikidata entities in web text; **WordNet**: general web text (negative control) |
| novelty index | C0′ − P0 after a track's terms: how much plain fine-tuning lowers the loss. Large = the words are new to the host |

**Numbers.** Unless stated, a number is the relative change in next-token loss on the 8 tokens after a term, C5 − C0′. **Negative means better.** Values pool 3 training seeds, with 95% cluster-bootstrap intervals over evaluation windows in brackets. `*` = significant after Holm correction within the stratum. "Seed 1" marks a single run. "Preliminary" marks numbers from generated reports that no committed summary has checked yet.

---

# 06.1 — Invented words (T5)

**Status:** done (SmolLM2 at 3 seeds; Qwen3 at 2 seeds, seed 3 queued) · **When:** 2026-10-04 (SmolLM2, 3 seeds), 2026-10-06 (Qwen3, 2 seeds), 2026-10-07 (controls) · **Sources:** `reports/R9-retrofit-quantization.md` (dimension 1, "Qwen3 on T5"); `experiments/e9-retrofit/report/t5/report.md`; `experiments/e9-retrofit/report/t5-qwen3/report.md`; `manuscript/update-2026-10-12.md` (E9-1); `manuscript/claims.md` (E9.A1–A3)

## Research gap
We needed the method's best case: words that no pretrained host can know, so that contamination cannot explain a gain. Without it we could not tell a weak mechanism from a domain the host already knows.

## Intuition
If the composed vector carries what a term means, it should help most where the host has nothing else to go on. Held-out terms are the sharpest test, because their vector comes from the ontology alone.

## Methodology
- **Data:** T5, a seeded synthetic glossary of an invented organisation; templated internal documents use its terms, mixed 50/50 with FineWeb-Edu.
- **Arms:** P0, C0′, C2 and C5 on the same 50M training tokens (SmolLM2: host lr 3e-5, method lr 1e-3; Qwen3: LoRA r 64).
- **Strata:** 360 held-out terms (never linked, never in the text), unseen, rare, long and inside terms, unlinked domain text, and general text (FineWeb-Edu).
- **Statistics:** as in "Numbers" above.
- **Pre-registration:** the design is R9's (author request, 2026-10-02). The claims ledger (E9.A1) fixed the refutation readings: if a same-site new-word vector matched C5, no composition-specific claim.

## Experiment
SmolLM2-135M and -360M at seeds 1–3; Qwen3-0.6B and -1.7B at seeds 1–2. The held-out stratum has 150,369 target tokens.

## Results

| Host | after held-out | after unseen | after rare | after long |
|---|---|---|---|---|
| SmolLM2-360M | **−13.3% [−14.1, −12.4]*** | −6.6% [−8.2, −5.1]* | −6.0% [−6.6, −5.4]* | −7.6% [−7.8, −7.3]* |
| SmolLM2-135M | **−9.2% [−10.0, −8.4]*** | −3.8% [−5.4, −2.3]* | −3.1% [−3.7, −2.5]* | −6.6% [−6.8, −6.4]* |
| Qwen3-1.7B (2 seeds) | −17.7% [−18.6, −16.8]* | −10.6% | −11.2% | −9.4% |
| Qwen3-0.6B (2 seeds) | −20.2% [−21.3, −19.1]* | −8.9% | −8.7% | −9.4% |

- **Every seed shows the gain.** After held-out terms: −11.7 / −13.1 / −15.0% (360M) and −7.5 / −10.6 / −9.4% (135M).
- **The free table does not explain it.** C5 − C2 after held-out terms is −15.1% [−16.0, −14.3]* (360M). C2 is *worse* than C0′ on held-out, unseen and rare terms (+4.6 to +5.2% at 135M).
- **Locality holds.** Unlinked domain text moves −0.03% (n.s., 360M); general text −0.0004 nats/token. The one cost is inside terms at 135M: +0.33% [+0.22, +0.45]*.
- **The refutation reading is not triggered on T5.** C5 beats every same-site alternative by 10.8–15.3% (06.4).

## Limitations
- T5's text is generated from the ontology the method reads, so these sizes are an upper bound. Copying and templated continuations inflate them (06.7).
- Qwen3 uses LoRA and 2 seeds, so host size and fine-tuning regime are confounded.
- T5's held-out terms never occur in the text, unlike T4's and T7's. "Held-out" therefore means something stronger here (06.9).

## Next steps
- **Done:** SmolLM2 at 3 seeds, with the controls of 06.3–06.4.
- **Queued:** Qwen3 seed 3 (priority 55); Qwen3-4B on T5 (priority 56); a Qwen3.5 memory probe (54.5).
- **Closed:** no further SmolLM2 seeds.

---

# 06.2 — Real text, and how the gain follows novelty (T4, T7, T1-open, WordNet)

**Status:** done (T4 and T7 at 360M, 3 seeds; T1-open and WordNet seed 1) · **When:** 2026-10-05 (T4 seed 1, T1-open, WordNet), 2026-10-09 to 10-11 (T4 3 seeds, official report; T7 3 seeds) · **Sources:** `reports/R9-retrofit-quantization.md` ("T4 chemistry — seeds 1–3", "T7", "T1-open", "WordNet", "Across tracks"); `experiments/e9-retrofit/report/{t4,t7,t1,wordnet}/report.md`; `manuscript/update-2026-10-12.md` (E9-2); `experiments/t4-chemistry/README.md`

## Research gap
T5 favours the method by construction. We needed natural text, not generated from the ontology, to size the effect a user would see.

## Intuition
The method should fill a gap only where the host lacks the word. Where the host already knows a term, plain fine-tuning should do the job and the vector should add nothing.

## Methodology
- **T4:** ChEBI release 255 (62,418 entities; frames of up to 24 edges), linked by IUPAC and trivial names in ChEBI entry texts and PubMed chemistry abstracts, mixed with FineWeb-Edu (100M tokens). Its 551 real held-out entries were chosen among entries with ≥ 5 occurrences, so they stay in the text, unlinked; 299 more are synthetic.
- **T7:** see 08.1. **T1-open:** MeSH descriptors in PubMed. **WordNet:** WordNet multi-word terms in FineWeb-Edu (negative control).
- **Novelty index:** C0′ − P0 after the terms.
- **Recipe, arms and statistics:** as in 06.1.
- **Pre-registration:** T7 had a recorded prediction (decision 55): a gain at least T4's and below T5's. It failed (06.9, 08.1).

## Experiment
T4 at 360M, seeds 1–3, with the full arm battery (135M in 06.11); T7 at 360M, seeds 1–3; T1-open and WordNet at 135M and 360M, seed 1.

## Results
SmolLM2-360M:

| Track | Novelty (C0′ − P0) | after held-out | after unseen | after rare | after long |
|---|---|---|---|---|---|
| T5 | −74 to −81% | −13.3%* | −6.6%* | −6.0%* | −7.6%* |
| T4 (3 seeds) | −40 to −56% | −0.40% [−0.54, −0.25]* | **−2.29% [−2.72, −1.90]*** | −0.91% [−1.06, −0.77]* | −1.09% [−1.16, −1.01]* |
| T7 (3 seeds) | −7 to −10% | **+0.48% [+0.35, +0.62]*** (worse) | — | −0.50% [−0.65, −0.36]* | −0.57% [−0.62, −0.51]* |
| T1-open (seed 1) | −5 to −8% | +0.02% [−0.01, +0.04] | +0.00% | −0.01% | −0.05%* |
| WordNet (seed 1) | < 1% | −0.00% [−0.02, +0.01] | every term stratum within ±0.06% | | |

- **The gain follows novelty.** It is largest on T5, smaller on T4, small on T7's seen terms and zero on T1-open and WordNet.
- **Within T4, the most novel stratum gains most:** unseen terms (novelty −56%) gain −2.29%.
- **On T4, C5 beats the free table on every term stratum** (C5 − C2 from −0.22% after held-out to −2.06% after unseen). T4 also gains inside terms (−1.54% [−1.67, −1.41]*), where T5 gains nothing.
- **T1c shows the same shape:** a gain after long terms only (08.2).

## Limitations
- Five tracks do not make a dose–response curve. Novelty also moves with ontology richness (06.12) and with how held-out terms were chosen.
- T1-open and WordNet are seed 1.
- "Held-out" differs across tracks: on T4 and T7 the host reads those names unlinked during fine-tuning.
- `status-2026-10-08.md` gives T7's held-out interval as [+0.36, +0.61] (preliminary, 10 Oct); we use the official report's [+0.35, +0.62].

## Next steps
- **Done:** T4 and T7 at 3 seeds.
- **Queued:** T7-ROOD (priority 54.497); T8 (54.499); Qwen3-1.7B on T4 (training done, evaluations at 54.4961–54.4963) and on T7 (54.4975).
- **Closed:** no further T1-open or WordNet seeds are queued.

---

# 06.3 — The binding operator is not what helps

**Status:** done · **When:** 2026-10-07 (T5), 2026-10-09 (T4) · **Sources:** `reports/R9-retrofit-quantization.md` ("T5 controls", "T4 chemistry — seeds 1–3"); `experiments/e9-retrofit/report/t5/report.md`; `experiments/e9-retrofit/report/t4-addendum2/report.md` (operator table); `manuscript/update-2026-10-12.md` (E9-3); `manuscript/outline.md` (rule 7)

## Research gap
We frame the method as vector-symbolic. If HRR binding is not what lowers the loss, the paper must not say it is (project rule 7).

## Intuition
If roles matter, dropping them (C5ut) or freezing a random operator (C5rf) should hurt. If translation (C5tr) does as well, the gain lies in the shared parts, not in the binding algebra.

## Methodology
- **Arms:** C5rf, C5ut and C5tr against C5. Everything else is identical, including parameters.
- **Tracks and hosts:** T5 and T4, SmolLM2-360M, seeds 1–3.
- **Rule 7 (outline):** unless HRR separates from fixed random binding and from no binding, describe the mechanism as "compositional parameter sharing".

## Experiment
Nine trained arm runs per track (3 arms × 3 seeds), scored on the base runs' windows.

## Results
C5 − arm (negative = C5 better; positive = the arm is better):

| Track | Arm | after held-out | after rare | after unseen | after long | arm − C0′, after held-out |
|---|---|---|---|---|---|---|
| T5 | C5rf | −5.2%* | −0.3%* | −0.3% | −1.1%* | −8.5% |
| T5 | C5ut | +0.2% | +0.1% | +0.4% | +0.1%* | −13.4% |
| T5 | C5tr | **+4.8%*** | +2.9%* | +2.7%* | +2.1%* | **−17.2%** |
| T4 | C5rf | −0.05% | +0.06% | +0.09% | +0.01% | −0.35%* |
| T4 | C5ut | −0.04% | +0.05% | −0.03% | −0.00% | −0.36%* |
| T4 | C5tr | +0.06% | +0.11%* | +0.14% | −0.02% | −0.45%* |

- **T5:** no binding equals HRR; translation beats it on every term stratum; only the fixed random operator costs (5.2% after held-out terms).
- **T4:** all four operators tie within ±0.14%.
- **Rule 7 applies.** The wording is "compositional parameter sharing", not symbolic binding.
- **Other programmes agree** (reported in their own files): from scratch all operators tie (E4-2); models trained to unbind stay at chance on role-swapped twins (`manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §1, preliminary). Binding beats no binding only in explicit read-out with ambiguous roles (update-2026-10-12, B-3).

## Limitations
- Loss endpoints only. Binding may still matter where the store is decoded explicitly (E12, other files).
- T7 has no operator arms. Why translation wins on T5 is not explained.

## Next steps
- **Closed** for language-model loss.
- **Queued:** T8 keeps C5ut and C5tr at seed 1 as descriptive replications (54.499). Their seeds 2–3 were withdrawn (decision 64).

---

# 06.4 — Composition versus other per-term vectors

**Status:** done on T5, T4 and T7; hybrid preliminary; fair TransE re-test (C6gi) queued · **When:** 2026-10-07 (T5), 2026-10-09 to 10-11 (T4, T7), 2026-10-20 (H5add, preliminary) · **Sources:** `reports/R9-retrofit-quantization.md` ("T5 controls", "T4 chemistry — seeds 1–3", "T7"); `experiments/e9-retrofit/report/{t4,t7}-addendum2/report.md` ("same-site row sources", generated 2026-10-20, uncommitted); `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §4; `manuscript/outline.md` (E9 framing rules)

## Research gap
Reviewers will ask first whether composition is needed at all. Any vector derived from the same entry, placed at the same position, might do as well (novelty check A-B1/A-B2). HRRBERT's hybrid design had not been tried in pretrained hosts.

## Intuition
If composition is special, it should beat other vectors built from the same entry, above all for held-out terms, where nothing was trained.

## Methodology
- **Arms:** C6m, C6d and C6g, each a frozen per-entry vector read through a trained, parameter-matched projector at C5's site and gate. Held-out terms get their vectors from the same source. H5add is matched to C5's parameters (T4 2.33M against 2.35M; T7 1.10M against 1.10M).
- **Hosts:** SmolLM2-360M, seeds 1–3; T5, T4 and T7 (of these arms, T7 has only C6d and H5add).
- **Framing rule (outline):** if a same-site vector matches C5 after held-out terms, the claim becomes "frame information at the term's position helps", with no composition-specific claim.
- **Pre-registration of H5add** (addendum 2, §4): H5add − C5 < 0 after held-out terms; its interval includes 0 after long terms. Reading: if H5add is nowhere worse and better after held-out terms, it replaces C5 as the recommended design.

## Experiment
Three seeds per arm on each track, scored on the base windows.

## Results
C5 − arm (negative = C5 better):

| Track | Arm | after held-out | after rare | after unseen | after long |
|---|---|---|---|---|---|
| T5 | C6m | −14.2%* | −6.9%* | −7.4%* | −7.5%* |
| T5 | C6d | −10.8%* | −2.8%* | −3.3%* | −3.8%* |
| T5 | C6g | −15.3%* | −5.3%* | −5.9%* | −6.1%* |
| T4 | C6m | +0.03% | −0.21%* | −0.48%* | −0.20%* |
| T4 | C6d | **+0.17% [+0.05, +0.29]*** | −0.00% | +0.19% | −0.20%* |
| T4 | C6g | +0.10% | +0.02% | −0.04% | +0.05%* |
| T4 | H5add (prelim.) | +0.01% | −0.46%* | −0.96%* | −0.35%* |
| T7 | C6d | +0.22%* | −0.15%* | — | −0.12%* |
| T7 | H5add (prelim.) | **+0.22%*** | −0.13%* | — | +0.01% |

- **T5:** composition beats all three alternatives by wide margins.
- **T4:** TransE ties C5 everywhere; the definition encoder is ahead after held-out terms; only the subtoken mean is clearly beaten. **The framing rule is triggered on real text.**
- **H5add (preliminary):**
  - On T4 the prediction fails: it ties C5 after held-out terms and is behind after long terms (−0.35%*).
  - On T7 it holds: the hybrid roughly halves the held-out cost (H5add − C0′ +0.26%* against C5's +0.48%*) and ties after long terms. It is behind after rare terms (−0.13%*).
  - The replacement reading is met on neither track, so C5 stays the recommended design.

## Limitations
- C6g's TransE saw the held-out entries' triples during its training, which favours it after held-out terms.
- H5add numbers come from uncommitted generated reports and are not yet in R9.
- T7 lacks C6m and C6g.

## Next steps
- **Queued:** C6gi on T7-ROOD (54.49705). Primary: C5 − C6gi after held-out terms. Prediction: C5 ahead or tied.
- **Queued:** reading rule (b), "better than standard new-word vectors" only if C5 − C6d < 0 after held-out terms, on T7-ROOD and T8.
- **Proposed:** write H5add into R9 once the reports are committed.

---

# 06.5 — Shared building blocks

**Status:** partly done (C5sh, C2 and C2w done; C1h preliminary) · **When:** 2026-10-10/11 (shuffled-frame reading), 2026-10-14/15 (C2w), 2026-10-20 (C1h, preliminary) · **Sources:** `reports/R9-retrofit-quantization.md` ("Linked terms"); `manuscript/update-2026-10-12.md` (E9-5); `manuscript/update-2026-10-15.md` §2; `experiments/e9-retrofit/report/{t4,t7}-addendum2/report.md` (generated 2026-10-20, uncommitted); `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §2; `-2026-10-15.md` §5

## Research gap
On real text the frame's truth matters little: C5sh keeps most of C5's gain. We had to find what carries the gain instead. Candidates: sharing dictionary vectors across terms, any sharing at all, or simply enough room per term.

## Intuition
In C5sh each term gets a consistent vector built from shared parts, but from the wrong entry. If C5sh beats a free table, sharing rather than truth carries the gain. Random sharing (C1h) then tests whether the ontology's *pattern* of sharing matters.

## Methodology
- **Arms:** C5, C5sh, C2, C2w (06.6) and C1h. C1h has one hash bucket per term's token sequence, sized to C5's budget (T4: 2,444 buckets of 960 numbers = 2.35M parameters; T7: 1,137 buckets).
- **Hosts:** SmolLM2-360M, seeds 1–3, T4 and T7.
- **Pre-registration:**
  - Addendum 1, §2: C5sh − C2w < 0 after long terms on T4 (06.6).
  - Addendum 2, §5: C5sh − C1h < 0 after long terms on T4 (Holm). Equality would mean any sharing explains the gain.

## Experiment
Three seeds per arm on T4 and T7.

## Results
After long terms, arm − C0′:

| Track | C5 | C5sh | C2w | C1h (prelim.) | C2 |
|---|---|---|---|---|---|
| T4 | −1.09% | −0.89% | −0.95% | −0.97% | −0.23% |
| T7 | −0.57% | −0.56% | −0.60% | −0.31% | −0.46% |

- **The frame's content adds little on real text.** C5 − C5sh after long terms is −0.20%* on T4 and −0.00% on T7. On T4 it is larger after unseen (−0.82%*) and rare terms (−0.37%*), and zero after held-out terms (+0.01% [−0.07, +0.10]). On T5, by contrast, shuffling destroys the gain (+4.6% against C0′ after held-out terms).
- **The "sharing" explanation was withdrawn on 15 October.** The wide free table C2w matches C5sh after long terms (06.6).
- **C1h (preliminary):**
  - On T4, random sharing at C5's parameter count does as well as shuffled-ontology sharing after long terms (−0.97% against −0.89%). The report prints C5 − C1h (−0.12%*), not the pre-registered C5sh − C1h. On the point estimates, the C1h prediction fails on T4.
  - On T7, C1h is well behind (−0.31% against −0.56%; C5 − C1h −0.26%*). Ontology-shaped sharing helps there.
  - Composition beats C1h clearly after unseen (−1.03%*) and rare terms (−0.53%*) on T4.

## Limitations
- C1h is preliminary, and its primary contrast is not printed. The comparison above uses arm − C0′ point estimates.
- C1h's buckets are 960 wide (the host's width). On our reading, C1h may therefore also gain from wide rows, not only from sharing.
- Only two real tracks.
- **Sources disagree; we follow the newest.** `status-2026-10-08.md` says scrambling the relations "destroys the gain", and `update-2026-10-12.md` (E9-5) says sharing building blocks is what helps on real text. The first holds only on T5. The second was withdrawn by `update-2026-10-15.md` §2.

## Next steps
- **Done:** C5sh, C2 and C2w at 3 seeds.
- **Proposed:** commit the addendum-2 reports and compute C5sh − C1h as pre-registered.
- **Queued:** richness stage 1 (06.12), which varies the frames at a fixed text.

---

# 06.6 — Per-term vector size

**Status:** partly done (C2w done; C5w512 queued) · **When:** 2026-10-14/15 · **Sources:** `manuscript/update-2026-10-15.md` §2; `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §2; `experiments/e9-retrofit/report/{t4,t7}-addendum/report.md` (uncommitted); `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §6; `manuscript/update-2026-10-12.md` (B-3)

## Research gap
The free table that C5sh beat on T4 has rows only 34 numbers wide, against 256 for the composed vector. C2 may have lost on width, not on lack of sharing. Separately, 256 numbers may be too few for entries with many edges.

## Intuition
If width is the bottleneck, a free table at full width should catch up with C5sh. If capacity limits the composed vector, widening it should help.

## Methodology
- **C2w:** C2's recipe with 256-wide rows. It is *not* parameter-matched: 17.3M parameters on T4 (C5: 2.35M) and 2.1M on T7 (C5: 1.09M). It answers "is it width?", not "is it budget?".
- **Hosts:** SmolLM2-360M, seeds 1–3, T4 and T7.
- **Pre-registration (addendum 1, §2):** C2w beats C2 but stays behind C5sh on T4 (C5sh − C2w < 0 after long terms). If the interval includes 0 or C2w is better, the wording becomes "a per-term vector of sufficient width", and the sharing claim is withdrawn for natural text.

## Experiment
Three C2w seeds per track, with the full evaluation chain.

## Results

| arm − C0′ | C5 | C5sh | C2w | C2 |
|---|---|---|---|---|
| T4, after long | −1.09% | −0.89% | **−0.95%** | −0.23% |
| T4, after unseen | −2.29% | −1.48% | −1.08% | — |
| T4, after rare | −0.91% | −0.54% | −0.37% | — |
| T4, after held-out | −0.40% | −0.41% | −0.61% | — |
| T7, after long | −0.57% | −0.56% | −0.60% | −0.46% |

- **The prediction failed.** C2w matches C5sh after long terms on T4, so the sharing claim is withdrawn for natural text. The wording is "a per-term vector of sufficient size".
- **Composition keeps two advantages:**
  - It needs about 7.5× fewer parameters on T4 for the same result. C5 − C2w after long terms is still −0.14%* in C5's favour.
  - It wins where a free table has little to learn from: C5 − C2w after unseen terms −1.22%*, after rare terms −0.55%*.
- **C2w is better after held-out terms** (T4 C5 − C2w +0.21%*; T7 +0.50%*). This fits 06.9: injecting a composed vector for a word the host already learned from text slightly hurts.
- **Capacity:** in a random-vector sweep at 256 dimensions, recovery is 0.98 with 8 relations per entry, 0.66 with 16 and 0.24 with 32. T4 entries average 8.08 edges and reach 24.

## Limitations
Width and parameter budget are confounded by design. Whether 256 dimensions limit C5 itself is untested until C5w512 runs.

## Next steps
- **Queued:** C5w512 (T4, 135M and 360M, 3 seeds, priority 54.49992). Prediction: C5w512 ≈ C5, i.e. capacity is not binding at about 8 edges per entry.
- **Proposed (update-2026-10-15):** a wide table built from a small shared basis at C5's budget, to separate width from sharing at equal parameters. C1h (06.5) answers part of this.

---

# 06.7 — Copying analysis

**Status:** done (filler split); strict non-copy and understanding tests ran but are preliminary · **When:** 2026-10-07 (T5), 2026-10-11 (T4 360M, T7), 2026-10-18/19 (strict and understanding reports, uncommitted) · **Sources:** `reports/R9-retrofit-quantization.md` ("T5 controls", "T4 at 3 seeds", "T7"); `manuscript/update-2026-10-12.md` (E9-8); `experiments/e9-retrofit/preregistration-understanding.md`; `experiments/e9-retrofit/report/{t5,t4,t7}-understanding/report.md` (generated 2026-10-18/19, uncommitted)

## Research gap
The novelty check warned that the gain might be the host copying words of the entry's frame that follow the term, especially in templated text. A gain of that kind would not be a "domain terms" result.

## Intuition
Split the tokens after a term into **filler tokens** (part of a name of one of the term's frame fillers) and the rest. If the gain sits on filler tokens, it is copying. A stricter split also removes fillers two hops away and relation words. Test items whose answer is no filler word then probe knowledge beyond copying.

## Methodology
- **Filler split** (`e9_rescore`, 8-token window after each term).
- **Strict stratum (B2):** keeps only target tokens that belong to no name of the term, of any filler within two hops, or of any relation wording.
- **Understanding items (B1):** six families (two-hop, affordance, paraphrase, reverse, comparison, negation). No option shares a content word with the anchor's verbalizations within two hops. Each option is scored by pointwise mutual information against a null prompt; the composite is correct − chance, averaged over families. Analysis: an anchors × seeds crossed model.
- **Pre-registration (2026-10-07, before any evaluation):**
  - B2: strict loss after held-out terms (T5, 360M) is < 0, but smaller than R9's −19.4% on non-filler tokens.
  - B1: held-out composite C5 − C0′ > 0. Readings (a)–(c) need the primary > 0; (d) "no evidence beyond copying" if its interval includes 0.

## Experiment
SmolLM2-360M, seeds 1–3, on T5, T4 and T7, plus 135M replications. T5's B1 set has 48,786 items over 560 held-out, 300 rare, 300 seen and 700 new anchors.

## Results
**Share of the gain on filler tokens:**

| Track | after held-out | after unseen | after rare | after long |
|---|---|---|---|---|
| T5 | 33% | 90% | 86% | 75% |
| T4 (360M) | 1% | 51% | 25% | 12% |
| T7 | ≈ 0% (fillers are 0.1–0.5% of targets) | — | ≈ 0% | ≈ 0% |

On T5, two thirds of the held-out gain falls on non-filler tokens (−19.4%*), which fits templated, type-specific continuations.

**Strict non-copy loss (B2; preliminary):**

| Track | after held-out | after unseen | after rare | after long |
|---|---|---|---|---|
| T5 | **−27.18% [−28.96, −25.42]*** (58% of targets kept) | −4.03%* | −3.80%* | −4.86%* |
| T4 | −0.39% [−0.54, −0.25]* | −1.85% [−2.27, −1.45]* | −0.66%* | −0.92%* |
| T7 | +0.46%* (a cost) | — | −0.51%* | −0.56%* |

The B2 direction passed. The size prediction failed in the method's favour: −27.2%, not less than −19.4%.

**Understanding items (B1; preliminary):**
- **T5, held-out:** composite C5 − C0′ = **−0.0240 [−0.0300, −0.0184]**. That is significant and opposite to the prediction.
  - Affordance drives it: −0.149*.
  - Paraphrase improves (+0.039*), while reverse (−0.024*) and negation (−0.011*) worsen.
  - None of readings (a)–(d) fits a significantly negative primary.
- **T5, new words:** +0.0081 [+0.0038, +0.0125].
- **T4, held-out:** −0.0001 [−0.0029, +0.0030]. **T7, held-out:** +0.0062 [+0.0024, +0.0099].

## Limitations
- The B1 and B2 numbers come from uncommitted generated reports. No committed document interprets them yet. The secondaries that the B1 readings need (own vs random frame, C5 − C5sh) are not read here.
- B1 scores sit near chance.

## Next steps
- **Done:** filler split on T5, T4 and T7.
- **Queued:** T7-ROOD understanding (54.49795) and Qwen3 understanding (59).
- **Proposed:** commit the understanding reports, read the B1 decision rule in full, and explain the T5 affordance drop.

---

# 06.8 — A definition in the prompt

**Status:** done (1,024-token glossary on T4, T5 and T7; uncapped glossary on T4 and T5); T7-ROOD queued · **When:** 2026-10-14 (capped), 2026-10-18 (uncapped; report regenerated 2026-10-20) · **Sources:** `manuscript/update-2026-10-15.md` §1; `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` §2; `experiments/e9-retrofit/incontext/report/report.md`; `experiments/e9-retrofit/incontext-full/report/report.md`; `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §3; `-2026-10-15.md` §1; `manuscript/outline.md` (rule A-B3)

## Research gap
The obvious objection is "why not put the definition in the prompt?" The novelty review lists it as a required baseline (A-B3). No loss-endpoint version existed.

## Intuition
If the method only carries what the written frame says, a glossary in front of the text should do as well, and the method should add nothing on top.

## Methodology
- **Glossary:** a `Glossary:` prefix with one line `<name>: <frame written as sentences>` per entry linked in the window (the C6d texts). It was capped at 1,024 tokens, then at 4,096. A control glossary lists the same number of *other* entries.
- **Hosts:** P0 (seed 1), C0′ and C5 (seeds 1–3), SmolLM2-360M. Evaluation only.
- **Primary:** "method on top of the glossary", C5 with glossary − C0′ with glossary.
- **Pre-registered predictions:**
  - Addendum 1: (a) the glossary helps C0′ at least as much as the method does; (b) the method on top is small (|·| < 0.5% on T4/T7, smaller than C5 − C0′ on T5); (c) the right glossary beats the control glossary.
  - Addendum 2: the glossary's gain grows with coverage; the method on top stays negative but shrinks.

## Experiment
Glossary cost per window: capped T4 910 tokens (45.3% of entries covered), T5 969 (79.2%), T7 162 (100%); uncapped T4 2,025 (98.8%), T5 1,231 (100%). The method costs no prompt tokens.

## Results

| Track, terms | Glossary alone (capped → uncapped) | Method alone | Method on top (capped → uncapped) |
|---|---|---|---|
| T5, held-out | −12.0%* → −9.89%* | −13.3%* | −10.8%* → −10.93%* |
| T4, unseen | −0.8%* → −0.20% (n.s.) | −2.3%* | −2.2%* → −2.26%* |
| T4, rare | −0.2% → +0.18% (n.s.) | −0.9%* | −0.8%* → −0.83%* |
| T4, long | −0.5%* → −0.14% (n.s.) | −1.1%* | −1.0%* → −1.01%* |
| T4, held-out | −0.7%* → −0.25% (n.s.) | −0.4%* | −0.4%* → −0.36%* |
| T7, long | +0.10%* (worse) | −0.57%* | −0.49%* |
| T7, held-out | +0.10% (n.s.) | +0.48%* | +0.52%* |

- **The method adds on top of the definition** on every track and term type tested. On T4 it beats the definition for unseen, rare and long terms.
- **The glossary information is real:** the right glossary beats the control glossary everywhere (e.g. T4 uncapped, held-out −0.95%*).
- **The untouched host gains more from the glossary** (P0, T4 held-out −1.65%* uncapped).
- **The glossary costs on text without terms:** +0.66%* on T4 (uncapped, C0′) and +0.59%* on T7.
- **Predictions:**
  - (a) failed on every track.
  - (b) failed on T4. By its letter it held on T5 (−10.8% on top, against −13.3% alone) and on T7 (−0.49%, inside the 0.5% bound).
  - (c) held.
  - Addendum 2: "glossary gain grows with coverage" failed; it shrank on both tracks. "Method on top stays negative" held; "shrinks" failed (unchanged).
- **Framing rule A-B3** ("embedding only a zero-token alternative") is not triggered.

## Limitations
- The glossary lines are written from the same frame the method reads, not taken from real dictionary definitions.
- Only the 360M host, only loss endpoints, and P0 at seed 1.
- The uncapped numbers are from the report regenerated on 2026-10-20. The 18 Oct finding leaves T5's "glossary for the plain model" cell empty; we take −9.89%* from the report.

## Next steps
- **Queued:** the same comparison on T7-ROOD (54.49736).
- **Open gap (not proposed in any source):** a glossary of real definitions rather than written-out frames.

---

# 06.9 — The held-out cost

**Status:** diagnosis done; clean test (T7-ROOD) queued · **When:** 2026-10-09 to 10-11 (frame swap, T7) · **Sources:** `reports/R9-retrofit-quantization.md` ("Frame swap", "Linked terms", "T7"); `experiments/e9-retrofit/preregistration-frameswap.md`; `experiments/e9-retrofit/frameswap/{t5,t4,t7,t1,wordnet}/report/`; `manuscript/update-2026-10-12.md` (E9-6, E9-7)

## Research gap
On T4 the held-out gain is not frame-specific: C5sh keeps all of it. On T7, C5 makes the loss *worse* after held-out names (+0.48%*). We had to find out why before trusting any zero-shot claim on real text.

## Intuition
T4's and T7's held-out terms still occur in the training text, unlinked. The host learns them as plain tokens, then at test time suddenly receives a vector it never saw for them.

## Methodology
- **Frame swap** (pre-registered, committed before any run): at evaluation only, a trained C5 gets, for one target set, another target term's vector (other), a random non-target term's vector (other-any), no vector (empty) or the mean row (mean).
- **Measure:** variant − own, relative to own. Positive = the term's own vector helps.
- **Primary:** other − own after held-out terms (360M, 3 seeds). A held-out gain counts as frame-specific only if it is > 0 after Holm.
- **Predictions:** T5 > 0; T4 other − own ≈ 0 and empty − own > 0; T1 and WordNet ≈ 0; T7 none registered.
- **Amendment 1:** the same over every linked term (after long terms).

## Experiment
C5 (and C5sh, C6d) at 360M, seeds 1–3, on T5, T4 and T7; T1-open and WordNet at seed 1; T4 at 135M, seed 1.

## Results

| Track · terms | other − own | empty − own |
|---|---|---|
| T5 · held-out | +24.4%* | +28.7%* |
| T4 · unseen | +1.33%* | +1.60%* |
| T4 · rare | +0.75%* | +0.79%* |
| T4 · held-out | +0.19% [+0.13, +0.26]* | **−0.19% [−0.25, −0.12]*** (no vector is better) |
| T7 · held-out | −0.01% [−0.09, +0.08] | **−0.44% [−0.57, −0.30]*** (no vector is better) |
| T7 · rare | +0.50%* | +0.66%* |

- **Predictions:** T5 held. On T4 both failed: other − own was +0.19%* (predicted ≈ 0) and empty − own −0.19%* (predicted > 0). T1-open and WordNet were null, as predicted.
- **Mechanism:** T7's held-out names occur unlinked in 22.7% of training abstracts (15,530 of 68,264). Every arm that injects a term-specific vector there is worse than C0′ (C5sh +0.31%*, C6d +0.26%*). The free table, which injects its mean row, is neutral (−0.02%).
- **C5's −0.40% held-out gain on T4 comes from the host weights trained alongside the method,** not from the zero-shot vector.
- **Zero-shot composition does help terms absent from training** (T4 unseen).
- **Removal measures dependence, not benefit.** Removing every linked vector costs more than C5's whole gain (T4 +1.95% against −1.09%), because the host co-adapts.
- **Size matters:** at 135M, T7 shows no held-out cost (+0.10% [−0.08, +0.27], 3 seeds, preliminary; 06.11).
- **Deployment rule:** inject only for terms the host has not already learned from text.

## Limitations
- The frame swap measures use inside one model, not benefit over C0′.
- T4's and T7's held-out terms were chosen among frequent terms, which built the confound into both tracks.

## Next steps
- **Done, failed:** link dropout as a training fix (06.10).
- **Queued:** T7-ROOD (54.497; P1: C5 − C0′ < 0 after held-out terms) with its frame swap (54.4972), and T8 and T1c-ROOD with the same document-exclusion holdout (08.2, 08.3). The updates dated T7-ROOD ≈ 19–21 Oct. On 2026-10-20 none of its 253 jobs has started; the E14 band (reading and logic experiments, priority 54.4959981) runs first.

---

# 06.10 — Link dropout

**Status:** done; closed · **When:** 2026-10-14/15 · **Sources:** `manuscript/update-2026-10-15.md` §3; `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §6; `experiments/e9-retrofit/report/t7-addendum2/report.md` (operator and specificity table; uncommitted); `experiments/e9-retrofit/frameswap/{t7,t7-linked}/report-addendum/report.md` (uncommitted)

## Research gap
On T7 the method costs +0.48%* after held-out names, because the host learned those names unlinked and then meets a vector it never saw (06.9). Could a change to training remove that cost, so that the deployment rule ("inject only for terms the host has not learned") becomes unnecessary?

## Intuition
If the host also meets known terms *without* their vector during training, it learns that a name can come with or without one. An unexpected vector at test time should then disturb it less.

## Methodology
- **C5ld:** C5 with `link_dropout` 0.3. Each linked term in a training batch gets no vector with probability 0.3; evaluation links every term.
- **Host and track:** T7, SmolLM2-360M, seeds 1–3; frame swap on the held-out and linked sets.
- **Pre-registration (addendum 1, §6; low confidence):** C5ld − C5 < 0 after held-out terms, at a seen-term price under 0.1%. "No held-out change" would mean the cost is not a train/test linking mismatch of this kind.

## Experiment
Three C5ld runs with the full evaluation chain and both frame swaps.

## Results

| | C5ld − C0′ | C5 − C0′ | C5 − C5ld |
|---|---|---|---|
| after held-out | +0.46% [+0.34, +0.57]* | +0.48%* | +0.03% [−0.01, +0.06] |
| after long | −0.46%* | −0.57%* | −0.10% [−0.12, −0.09]* |
| after rare | −0.44%* | −0.50%* | −0.06% [−0.10, −0.03]* |

- **The prediction failed.** The held-out cost is unchanged, and link dropout costs a little on seen terms (0.10% after long terms).
- **Link dropout lowers dependence on the vectors:** removing them costs +0.61% instead of +0.97% after long terms. Removing the vector after held-out names still helps by the same amount (−0.44%).
- **Reading:** the held-out cost is not caused by the host always seeing known terms with their vectors.

## Limitations
One dropout rate (0.3), one track, one host size.

## Next steps
**Closed:** link dropout is dropped. The deployment rule of 06.9 stands, and T7-ROOD is the clean test.

---

# 06.11 — Smaller hosts (SmolLM2-135M)

**Status:** done on T4 (3 seeds); T7 seeds 2–3 finished (preliminary report); T5 and T1c at both sizes · **When:** 2026-10-04 (T5), 2026-10-13/15 (T4 3 seeds), 2026-10-15 (T7 seed 1), 2026-10-20 (T7 3 seeds, preliminary) · **Sources:** `manuscript/update-2026-10-15.md` §§4–6; `experiments/e9-retrofit/report/t4-addendum2/report.md` and `experiments/e9-retrofit/report/t7/report.md` (135M sections; working-tree versions, uncommitted); `reports/R9-retrofit-quantization.md` (dimension 1); `experiments/e9-retrofit/preregistration-addendum-2026-10-11.md` §5; `-2026-10-15.md` §2

## Research gap
At seed 1 the smaller host gained more on T4. That needed confirming at 3 seeds and on a second real domain.

## Intuition
The method should help most where the host's own capacity to learn a new term is weakest.

## Methodology
- **Comparison:** the same recipe at 135M and 360M, 3 seeds each.
- **Pre-registration:**
  - Addendum 1, §5 (T4): 135M gains more after unseen, rare and long terms. The size result enters R9 only if the 135M − 360M intervals exclude 0.
  - Addendum 2, §2 (T7): seed 1 replicates, 135M gains more after rare and long terms, and 135M shows no held-out cost.

## Experiment
T4 135M seeds 2–3 and T7 135M seeds 2–3 (C0′, C2, C5) were added to the existing seed-1 runs. T1c had both sizes at 3 seeds.

## Results

| Stratum | T4 135M | T4 360M | T7 135M (prelim.) | T7 360M | T1c 135M | T1c 360M |
|---|---|---|---|---|---|---|
| after unseen | **−3.62%*** | −2.29%* | — | — | +0.02% | −0.48%* |
| after long | **−1.72%*** | −1.09%* | **−1.09% [−1.16, −1.02]*** | −0.57%* | −1.84%* | −0.94%* |
| after rare | **−1.58%*** | −0.91%* | **−1.04% [−1.22, −0.87]*** | −0.50%* | +0.04% | −0.02% |
| inside | −1.98%* | −1.54%* | −0.50%* | −0.28%* | −1.23%* | −0.53%* |
| after held-out | −0.54%* | −0.40%* | +0.10% [−0.08, +0.27] | **+0.48%*** | −0.13% | −0.02% |

- **On real text the smaller host gains more,** roughly twice as much on T7. T4's intervals for the two sizes do not overlap on unseen, long, rare and inside terms; they overlap on held-out terms.
- **The T7 held-out cost disappears at 135M** at 3 seeds (preliminary).
- **Predictions:** both pass on non-overlapping intervals. The pre-registered paired contrast 135M − 360M is not printed in either report.
- **T5 goes the other way:** 360M gains more than 135M (−13.3% against −9.2% after held-out terms), and Qwen3 with LoRA more again (06.1).

## Limitations
- Only two SmolLM2 sizes. Qwen3 differs in fine-tuning regime (LoRA).
- T1c's 3-seed numbers come without intervals in the manuscript.
- T7's 3-seed numbers come from an uncommitted report.
- **Sources disagree; we follow the newest.** R9 states that "the gain grows with model size (360M > 135M)"; that holds for T5 only. `claims.md` (E9.A1) still quotes T4 135M at seed 1 (held-out −0.63%, unseen −3.7%); the 3-seed values above supersede it.

## Next steps
- **Queued:** Qwen3-1.7B on T4 (training done; evaluations 54.4961–54.4963), T7 (54.4975), T7-ROOD (54.4977) and T8 (54.4994).
- **Queued:** T7-ROOD base runs at 135M (54.499935); richness stage 1 at both sizes (06.12).
- **Proposed:** write T7 135M into R9 once the report is committed.

---

# 06.12 — The richer-ontology idea

**Status:** queued (stages 1–2 pre-registered; stage 3 gated) · **When:** proposed and pre-registered 2026-10-15; runs estimated ≈ 27–29 Oct · **Sources:** `manuscript/update-2026-10-15.md` ("Idea under evaluation", "Queued on 15 October"); `experiments/e9-retrofit/preregistration-addendum-2026-10-15.md` §§6–7; `src/vsa_embed/experiments/e9_plan.py`; `src/vsa_embed/experiments/t7_enrich.py`

## Research gap
Does the ontology's content cause the gain? Our evidence cannot say. Across tracks, novelty and ontology richness move together, and host-size runs keep the ontology fixed. (Author's idea, 15 Oct.)

## Intuition
Keep the host and the text fixed and vary only the frames. If the method draws on the ontology's knowledge, more true relations should help more, and shuffled relations should not.

## Methodology
**Stage 1: thinner T4 ontologies** (135M and 360M, 3 seeds; C5sh added at 135M):

| Arm | Frames | Edges per entry |
|---|---|---|
| C5isa | *is-a* only | 1.95 |
| C5comp | *is-a*, elements, charge, branch | 6.44 |
| C5half | half of every entry's edges, all 13 relation types | 4.08 |
| C5 | full ontology | 8.08 |
| C5w512 | full ontology at width 512 (4.70M parameters) | 8.08 |

- **Primary:** dose–response of arm − C0′ over C5isa, C5comp and C5 after long, unseen and rare terms; C5 − C5isa with Holm.
- **Predictions:** (a) monotone, with C5 − C5isa < 0 on at least two strata; (b) C5w512 ≈ C5; (c) no direction for the size interaction.

**Stages 2–3: richer T7-ROOD ontologies:**
- **C5mesh / C5meshsh:** heading facts from MeSH itself. 8,160 of 8,184 entries gain edges; frames grow from 5.15 to 9.47 edges per entry.
- **C5tch / C5tchsh:** facts Claude reads from each record's MeSH note. Only 4,011 entries have a usable note; spend is capped at $60.
- **C5host / C5hostsh:** the trained host reads the same notes. This stage is gated: it runs only if ≥ 10% of entries gain an edge. A smoke test added 0 edges.
- **Primary:** C5<variant> − C5 after held-out and long terms. Predictions: (a) C5mesh − C5 < 0 after long terms; (b) C5tch − C5 small.

## Experiment
Nothing has run. Queue (2026-10-20): 111 T4 richness jobs at 54.49992–54.49993; T7-ROOD enrichment jobs at 54.49995–54.499961; reports at 54.49993 and 54.49999. All of them sit behind T7-ROOD and T8.

## Results
No results yet. Prior evidence fits the idea but is confounded with novelty:

| Track | Relation types | Edges per term | Content effect (C5 vs C5sh) |
|---|---|---|---|
| T5 | 23 | 7.4 | large: shuffling destroys the gain |
| T4 | 13 | 8.1 | small: −0.20% after long terms |
| T7 | 7 | 5.2 (75% tree or record-class codes) | none |
| T1-open | 5 | 4.4 | none (no gain) |

- **From scratch** (50M, seed 1), correct relations beat shuffled ones by 0.50% after long terms.
- **A teacher pilot** wrote 400 WordNet frames for $2.41 at precision 0.78 and recall 0.75.

## Limitations
- "Richer" mixes edge count, relation types and correctness. Teacher frames may distil Claude's knowledge.
- Without C5w512, "richer does not help" could mean "the vector is full".
- Stage 2 depends on T7-ROOD's base runs, which have not started.

## Next steps
- **Queued:** stages 1–2.
- **Gated:** stage 3, expected to skip.
- **Proposed, not queued:** one level pair on Qwen3-1.7B (update-2026-10-15).

---

# Sources

- Plain-language summaries: `manuscript/update-2026-10-12.md` (E9-1 … E9-11), `manuscript/update-2026-10-15.md` (§§1–6, richness idea), `manuscript/status-2026-10-08.md`, `manuscript/findings/2026-10-18-prelim-readout-incontext-scale.md` (§§1–2).
- Ledgers: `manuscript/claims.md` (E9.A1–A3), `manuscript/outline.md` (rule 7, E9 framing rules).
- Report: `reports/R9-retrofit-quantization.md`.
- Pre-registrations: `experiments/e9-retrofit/preregistration-frameswap.md`, `preregistration-understanding.md`, `preregistration-addendum-2026-10-11.md`, `preregistration-addendum-2026-10-15.md`; `experiments/t7-new-vocabulary/preregistration-rood.md`.
- Generated reports (committed): `experiments/e9-retrofit/report/{t5,t5-qwen3,t4,t7,t7-pq,t1,wordnet}/report.md` as of 2026-10-11/12; `experiments/e9-retrofit/incontext/report/report.md`, `incontext-full/report/report.md`; `experiments/e9-retrofit/frameswap/*/report/`.
- Generated reports (**uncommitted, preliminary**): `experiments/e9-retrofit/report/{t4-addendum,t7-addendum,t4-addendum2,t7-addendum2}/report.md`; working-tree `report/t4/report.md` and `report/t7/report.md` (135M at 3 seeds); `report/{t5,t4,t7}-understanding/report.md`; `frameswap/{t7,t7-linked}/report-addendum/report.md`.
- Track design: `experiments/t4-chemistry/README.md`.
- Arm definitions: `src/vsa_embed/experiments/e9_plan.py`.
- Queue state: `python -m vsa_embed.jobqueue status`, read-only, 2026-10-20.
