# E7.1 judge rubric: authored edges without a gold counterpart (fixed before grading)

Study `authored_edge_in_context` (`vsa_embed.judge_protocol`). Each item shows one usage of a concept
from the reading corpus, the concept, and one authored fact `concept — relation — related concept`.
The judge answers **true**, **partly** or **false**:

- **true** — the fact is correct for the concept in the sense used, and specific enough to be useful
  (e.g. `golden retriever — is_a — dog`).
- **partly** — roughly right but too vague (`— is_a — thing`), only sometimes true, or right for
  another sense of the phrase.
- **false** — wrong, unsupported, or the phrase is not a meaningful concept (a fragment such as
  "number of the").

Relations are shown with their descriptions (`is_a: a more general kind of thing it is`, ...).
Protocol (experiments.md §0.12): items blinded (no author, no gold flag) and shuffled; each item
graded in ≥ 3 independent calls cycling over 2 prompt paraphrases; model pinned (`claude-opus-5-5`)
and recorded with every verdict; verdicts cached by model and prompt hash. Reported: majority
verdict per item, plausibility (true, and true + partly) per author with Wilson intervals, Fleiss κ
across calls, and accuracy on the calibration items (visible-ontology edges vs corrupted edges,
`label_source: ontology`; author-labelled items may be added with `label_source: author`).
