"""LLM-judge protocol (task C5): rubrics, prompts, schemas and study runner for E5.3 and E7.1.

Four studies, each item graded by `JudgeClient` (Claude Code headless) in ≥ 3 calls over 2 prompt
paraphrases, blinded and position-randomised:

- `neighbours` — are a rare concept's nearest neighbours (from one system) related to it? Score
  0–3 per neighbour list; "strongly related" = 3 (the HRRBERT study's endpoint, Fisher test).
- `edge_explanation` — do the top-weighted frame edges explain the concept's meaning in this
  sentence? 0–2.
- `split_card` — do the two usage groups of an M3 split correspond to distinct meanings? yes/no,
  plus a short label for each group.
- `authored_edge` — is an authored `relation: filler` edge true of the concept? true / partly / false.

Calibration: before any study, the judge grades ontology-derived items with known answers
(WordNet hypernym pairs vs random pairs for `neighbours`; gold edges vs corrupted edges for
`authored_edge`); accuracy is reported. Author-labelled calibration items can be added to the same
files (`label_source: author`) and are reported separately.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from .judging import JudgeClient, blind_options, fleiss_kappa, majority

STUDIES: dict[str, dict[str, Any]] = {
    "neighbours": {
        "schema": {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 3},
                                                     "reason": {"type": "string"}}, "required": ["score"]},
        "prompts": [
            "You are rating semantic relatedness for a terminology study.\nConcept: {concept}\nCandidate related concepts: {items}\n"
            "Score how related the candidates are to the concept overall: 0 unrelated, 1 weakly, 2 related, 3 strongly related. "
            "Answer with a JSON object {{\"score\": <0-3>, \"reason\": \"...\"}}.",
            "Here is a concept and a list of other concepts a system says are similar.\nConcept: {concept}\nList: {items}\n"
            "On a 0-3 scale (0 = not related, 3 = strongly related), how related is the list to the concept?",
        ],
    },
    "edge_explanation": {
        "schema": {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 2},
                                                     "reason": {"type": "string"}}, "required": ["score"]},
        "prompts": [
            "Sentence: {sentence}\nWord: {concept}\nA system explains the word's meaning here with these facts: {items}\n"
            "Score the explanation: 0 wrong for this sentence, 1 partly right, 2 right.",
            "In the sentence \"{sentence}\", the word \"{concept}\" is explained as: {items}. Is this explanation correct for "
            "this use of the word? 0 = no, 1 = partly, 2 = yes.",
        ],
    },
    "split_card": {
        "schema": {"type": "object", "properties": {"distinct": {"type": "boolean"}, "label_a": {"type": "string"},
                                                     "label_b": {"type": "string"}}, "required": ["distinct"]},
        "prompts": [
            "A vocabulary item \"{concept}\" was split into two groups by how it is used.\nGroup A uses: {group_a}\nGroup B uses: {group_b}\n"
            "Do the groups correspond to two distinct meanings of the item? Give a short label for each group.",
            "Two sets of concepts both contain \"{concept}\".\nSet A: {group_a}\nSet B: {group_b}\n"
            "Does \"{concept}\" mean something different in set A than in set B? Label each meaning.",
        ],
    },
    "authored_edge": {
        "schema": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["true", "partly", "false"]},
                                                     "reason": {"type": "string"}}, "required": ["verdict"]},
        "prompts": [
            "Concept: {concept}\nClaimed fact: {relation} → {filler}\nIs the claimed fact true of the concept? Answer true, partly or false.",
            "Is it correct that \"{concept}\" has the relation \"{relation}\" to \"{filler}\"? Reply true, partly, or false.",
        ],
    },
}


def build_prompts(study: str, fields: dict[str, Any]) -> list[str]:
    return [template.format(**fields) for template in STUDIES[study]["prompts"]]


def run_study(client: JudgeClient, study: str, items: Sequence[dict[str, Any]], *, seed: int = 0,
              score_of: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Grade items; each item has `id`, `fields` (prompt fields) and optional `system` / `gold`.
    Items are shuffled (blinded order); the system label is never shown to the judge."""
    order = list(items)
    random.Random(seed).shuffle(order)
    key = score_of or (lambda verdict: next(iter(verdict.values())))
    results, ratings = [], []
    for item in order:
        records = client.grade(item["id"], build_prompts(study, item["fields"]), STUDIES[study]["schema"])
        scores = [key(r["verdict"]) for r in records if "verdict" in r]
        if scores:
            ratings.append(scores)
        results.append({"id": item["id"], "system": item.get("system"), "gold": item.get("gold"),
                        "scores": scores, "majority": majority(scores) if scores else None})
    return {"study": study, "items": results, "fleiss_kappa": fleiss_kappa(ratings) if ratings else float("nan"),
            "spent_usd": client.spent_usd}


def fisher_one_tailed(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """P(X ≥ a_success) for the first group under the hypergeometric null (one-tailed Fisher)."""
    from math import comb
    total_success, total = a_success + b_success, a_total + b_total
    denominator = comb(total, a_total)
    return sum(comb(total_success, k) * comb(total - total_success, a_total - k)
               for k in range(a_success, min(total_success, a_total) + 1)) / denominator


def wordnet_calibration_items(wordnet: Any, *, count: int = 30, seed: int = 0) -> list[dict[str, Any]]:
    """Neighbour-study calibration: hypernym/hyponym lists (gold 3) vs random lists (gold 0)."""
    rng = random.Random(seed)
    nouns = [s for s in wordnet.all_synsets("n") if s.hyponyms() and s.hypernyms()]
    rng.shuffle(nouns)
    items = []
    for i, synset in enumerate(nouns[:count]):
        name = synset.lemma_names()[0].replace("_", " ")
        if i % 2 == 0:
            related = [h.lemma_names()[0] for h in (synset.hypernyms() + synset.hyponyms())[:5]]
            gold = 3
        else:
            related = [rng.choice(nouns).lemma_names()[0] for _ in range(5)]
            gold = 0
        items.append({"id": f"calib-neighbours-{i}", "gold": gold,
                      "fields": {"concept": name, "items": ", ".join(r.replace("_", " ") for r in related)}})
    return items
