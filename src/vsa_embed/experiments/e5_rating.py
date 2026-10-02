"""E5.3 rating study, LLM-graded now and clinician-rated later on the identical items (D5.3).

Item builders (each writes an item directory, `e5-rating-items/1`):

- `neighbours` — the HRRBERT neighbour study (28/40 vs 4/40 "strongly related" for rare codes, one-
  tailed Fisher test) replicated: `n_concepts` = 10 rare concepts × top-4 neighbours = **n = 40 pairs
  per system** (pre-registered). Rare = non-held-out single-concept entries with training frequency
  1–9 and a linkable alias, drawn by frequency-weighted sampling without replacement (the paper's
  "drawn randomly by weighted frequency"); `--set heldout` draws 10 held-out entries uniformly
  instead. Neighbours by cosine among the shared candidate pool (non-held-out entries seen in
  training), in the model's `state` space (final-layer state of "The word <alias>", channel live —
  primary) or `row` space (channel rows; C0: mean subtoken embedding). Each (concept, candidate)
  pair is one item, pooled over systems and graded once (systems are never shown; order is
  shuffled), rated "strongly related" / "less related" / "unrelated". A secondary blinded
  side-by-side preference (reference system vs each other) is counterbalanced: the two paraphrases
  show the lists in opposite positions.
- `edges` — edge-level explanations: 40 eval-corpus occurrences (one per entry; polysemous union
  entries with ≥ 4 edges first, seeded) explained by the top-3 edges by the run's context weight
  (ties broken at random), by 3 random frame edges, and by the 3 nearest neighbours (channel-row
  space) — rated with the C5 `edge_explanation` rubric (0 wrong, 1 partly, 2 right).
- `cards` — M3 split cards of a developmental run: up to 40 splits (seeded sample) with the usage
  groups of the two children vs a random partition of the same usages into groups of the same
  sizes — rated with the C5 `split_card` rubric (distinct meanings yes/no).

Every study mixes in calibration items with known answers (WordNet relations vs random), graded
blind like the rest. `grade` runs the judge (`JudgeClient`: headless `claude -p` with the model
pinned, the Claude Code file exchange, or a fake runner for tests): ≥ 3 calls per item over ≥ 2
paraphrases, verdicts cached by (model, prompt, schema, call) so a later clinician study rates the
identical items. Reported: per-system counts with the one-tailed Fisher test (reference system vs
each other), sign test for preferences, Fleiss' κ across calls, calibration accuracy, harness,
model, calls and spend. Results are labelled "LLM-graded estimate".

Item directory format (`e5-rating-items/1`): `items.jsonl`, one object per item with `id`, `study`
(`neighbour_pair` | `neighbour_preference` | `edge_explanation` | `split_card`), `calibration` (bool),
`systems` (the systems that produced it; never shown to the judge), `fields` (exactly the prompt
fields of the study: `concept`, `candidate` | `list_a`, `list_b` | `sentence`, `concept`, `items` |
`concept`, `group_a`, `group_b`), `gold` for calibration items and `order` (the systems behind lists
A and B) for preference items; `manifest.json` with `study`, `reference` (the system tested against
each other with Fisher), `systems`, `n_per_system` (neighbour pairs), `selection` and `endpoint`.
The T1-open track feeds rare-MeSH neighbour items into `experiments/t1-open-clinical/items/<set>/`
in this format (`grade --t1` grades every item directory found there). MIMIC-derived text is never
written to an item file.

    python -m vsa_embed.experiments.e5_rating build neighbours --runs channel=RUN_C5 C0=RUN_C0 C2=RUN_C2 --output DIR
    python -m vsa_embed.experiments.e5_rating build edges --run RUN_C5 --output DIR
    python -m vsa_embed.experiments.e5_rating build cards --run RUN_C6 --output DIR
    python -m vsa_embed.experiments.e5_rating grade --items DIR --output OUT [--runner cli|exchange] [--max-new-calls N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..data.corpus import TokenCorpus
from ..evaluation import channel_probes as cp
from ..judge_protocol import STUDIES as BASE_STUDIES
from ..judge_protocol import fisher_one_tailed
from ..judging import JudgeClient, calibration_agreement, claude_cli_runner, fleiss_kappa, majority
from .e5_common import (E5Run, canonical_surfaces, edge_text, entry_rows, eos_positions, fmt, json_ready, open_run,
                        readable_atomic, sha256_text, span_contexts, write_json)

SCHEMA = "e5-rating-items/1"
T1_ITEMS = Path("experiments/t1-open-clinical/items")
PAIR_LABELS = ("strongly related", "less related", "unrelated")
PAIR_ORDINAL = {"strongly related": 2, "less related": 1, "unrelated": 0}
STUDIES: dict[str, dict[str, Any]] = {
    "neighbour_pair": {
        "schema": {"type": "object", "properties": {"rating": {"type": "string", "enum": list(PAIR_LABELS)},
                                                     "reason": {"type": "string"}}, "required": ["rating"]},
        "prompts": [
            "You are rating ontological relatedness for a terminology study.\nConcept: {concept}\nCandidate: {candidate}\n"
            "Rate the candidate: \"strongly related\" (a synonym, a direct kind or parent, a close sibling of the same kind, or a "
            "part or whole of the concept), \"less related\" (same broad domain or a looser association) or \"unrelated\". "
            "Answer with a JSON object {{\"rating\": \"...\", \"reason\": \"...\"}}.",
            "Concept A: {concept}\nConcept B: {candidate}\nHow closely related are A and B in meaning? Choose one: strongly related "
            "(synonym, parent or child kind, close sibling, or part/whole), less related (same general area or a loose "
            "association), unrelated.",
        ],
    },
    "neighbour_preference": {
        "schema": {"type": "object", "properties": {"choice": {"type": "string", "enum": ["A", "B", "tie"]},
                                                     "reason": {"type": "string"}}, "required": ["choice"]},
        "prompts": [
            "Two systems listed the terms they find most similar to a concept.\nConcept: {concept}\nList A: {list_a}\n"
            "List B: {list_b}\nWhich list contains terms more closely related in meaning to the concept (synonyms, kinds, "
            "parts, close relatives)? Answer \"A\", \"B\" or \"tie\".",
            "Concept: {concept}\nCandidates from system A: {list_a}\nCandidates from system B: {list_b}\n"
            "Which system's candidates are more strongly related to the concept? Reply A, B, or tie.",
        ],
    },
    "edge_explanation": BASE_STUDIES["edge_explanation"],
    "split_card": BASE_STUDIES["split_card"],
}
VERDICT_KEY = {"neighbour_pair": "rating", "neighbour_preference": "choice", "edge_explanation": "score", "split_card": "distinct"}
SUCCESS = {"neighbour_pair": lambda v: v == "strongly related", "edge_explanation": lambda v: v == 2,
           "split_card": lambda v: v is True}


def rubric_sha256(study: str) -> str:
    return sha256_text(json.dumps({"prompts": STUDIES[study]["prompts"], "schema": STUDIES[study]["schema"]}, sort_keys=True))


def write_items(out_dir: Path, manifest: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "items.jsonl").write_text("".join(json.dumps(json_ready(i)) + "\n" for i in items))
    studies = Counter(i["study"] for i in items)
    manifest = {"schema": SCHEMA, **manifest, "counts": {"items": len(items), "calibration": sum(i["calibration"] for i in items),
                                                          **studies},
                "rubrics": {s: rubric_sha256(s) for s in studies}}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_rating_items(items_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((Path(items_dir) / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{items_dir} is not an {SCHEMA} item directory")
    items = [json.loads(line) for line in (Path(items_dir) / "items.jsonl").read_text().splitlines() if line.strip()]
    for item in items:
        if item["study"] not in STUDIES:
            raise ValueError(f"unknown study {item['study']} in {item['id']}")
    return manifest, items


# -- calibration items (known answers) --------------------------------------------------------------------

def _lemma(synset: Any) -> str:
    return synset.lemma_names()[0].replace("_", " ")


def calibration_items(study: str, wordnet: Any, *, count: int, seed: int) -> list[dict[str, Any]]:
    """`count` WordNet items with known answers for `study` (alternately positive and negative),
    graded blind among the study's items."""
    rng = random.Random(seed + 991)
    nouns = [s for s in wordnet.all_synsets("n") if s.hypernyms() and s.hyponyms()]
    rng.shuffle(nouns)
    items: list[dict[str, Any]] = []
    for synset in nouns:
        if len(items) >= count:
            break
        i, positive = len(items), len(items) % 2 == 0
        related = synset.hypernyms() + synset.hyponyms()
        other = rng.choice(nouns)
        if study == "neighbour_pair":
            candidate = _lemma(rng.choice(related)) if positive else _lemma(rng.choice(other.hyponyms()))
            items.append({"id": f"calib-pair-{i}", "study": study, "calibration": True, "systems": [],
                          "gold": "strongly related" if positive else "unrelated",
                          "fields": {"concept": _lemma(synset), "candidate": candidate}})
        elif study == "neighbour_preference":
            if len(related) < 3:
                continue
            good = ", ".join(_lemma(s) for s in related[:4])
            bad = ", ".join(_lemma(rng.choice(nouns)) for _ in range(4))
            items.append({"id": f"calib-preference-{i}", "study": study, "calibration": True, "systems": [],
                          "gold": "A" if positive else "B", "order": ["good", "bad"] if positive else ["bad", "good"],
                          "fields": {"concept": _lemma(synset), "list_a": good if positive else bad,
                                     "list_b": bad if positive else good}})
        elif study == "edge_explanation":
            examples = [e for e in synset.examples() if _lemma(synset).split()[0].lower() in e.lower()]
            if not examples or other == synset:
                continue
            source = synset if positive else other
            facts = [f"is a kind of {_lemma(h)}" for h in source.hypernyms()[:2]] + [f"has category {source.lexname().split('.')[-1]}"]
            items.append({"id": f"calib-edges-{i}", "study": study, "calibration": True, "systems": [], "gold": 2 if positive else 0,
                          "fields": {"sentence": examples[0], "concept": _lemma(synset), "items": "; ".join(facts)}})
        elif study == "split_card":
            word = _lemma(synset).split()[0]
            if positive:
                senses = [s for s in wordnet.synsets(word, "n") if len(s.hyponyms()) >= 3]
                if len(senses) < 2 or senses[0].lexname() == senses[1].lexname():
                    continue
                group_a = ", ".join(_lemma(h) for h in senses[0].hyponyms()[:5])
                group_b = ", ".join(_lemma(h) for h in senses[1].hyponyms()[:5])
            else:
                pool = [_lemma(h) for h in synset.hyponyms()][:10]
                if len(pool) < 4:
                    continue
                rng.shuffle(pool)
                half = len(pool) // 2
                group_a, group_b = ", ".join(pool[:half]), ", ".join(pool[half:])
            items.append({"id": f"calib-card-{i}", "study": study, "calibration": True, "systems": [], "gold": positive,
                          "fields": {"concept": word, "group_a": group_a, "group_b": group_b}})
    return items


# -- neighbours ---------------------------------------------------------------------------------------------------

def select_concepts(ontology: dict[str, Any], table: Any, surfaces: dict[int, dict[str, Any]], *, concept_set: str,
                    count: int, seed: int) -> list[int]:
    """Pre-registered selection: rare (frequency 1–9, frequency-weighted sampling without replacement)
    or held-out (uniform), single-concept entries with a linkable alias."""
    frequency = np.asarray(ontology["train_frequency"])
    heldout = {int(e) for e in ontology["heldout_entries"]}
    rng = np.random.default_rng(seed)
    single = [e for e in range(len(table.entry_concepts)) if len(table.entry_concepts[e]) == 1
              and e in surfaces and surfaces[e]["linkable"]]
    if concept_set == "rare":
        eligible = np.asarray([e for e in single if e not in heldout and 1 <= frequency[e] <= 9], dtype=np.int64)
        weights = frequency[eligible].astype(float)
        if len(eligible) <= count:
            return sorted(eligible.tolist())
        return sorted(rng.choice(eligible, size=count, replace=False, p=weights / weights.sum()).tolist())
    if concept_set == "heldout":
        eligible = np.asarray([e for e in single if e in heldout], dtype=np.int64)
        return sorted(rng.choice(eligible, size=min(count, len(eligible)), replace=False).tolist())
    raise ValueError("concept_set must be rare or heldout")


@torch.no_grad()
def representations(run: E5Run, entries: Sequence[int], surfaces: dict[int, dict[str, Any]], space: str) -> torch.Tensor:
    """Unit vectors of `entries` in the run's `state` or `row` space."""
    if space == "state":
        states, _ = cp.word_states(run.adapter, [surfaces[e]["surface"] for e in entries])
        return F.normalize(states.float(), dim=-1)
    if space == "row":
        if run.channel is not None and run.mode in {"compose", "free"}:
            return F.normalize(entry_rows(run.channel, torch.as_tensor(list(entries))), dim=-1)
        wte = run.model.model.get_input_embeddings().weight.detach().float().cpu()
        encoded = run.tokenizer([" " + surfaces[e]["surface"] for e in entries], add_special_tokens=False)["input_ids"]
        return F.normalize(torch.stack([wte[ids].mean(0) for ids in encoded]), dim=-1)
    raise ValueError("space must be state or row")


def nearest(run: E5Run, targets: Sequence[int], pool: Sequence[int], surfaces: dict[int, dict[str, Any]], *, space: str,
            top: int, table: Any) -> dict[int, list[int]]:
    """Top-`top` neighbours of each target in `pool` (excluding the target, its surface and shared concepts)."""
    vectors = representations(run, list(targets) + list(pool), surfaces, space)
    t, p = vectors[:len(targets)], vectors[len(targets):]
    scores = t @ p.T
    out = {}
    for i, target in enumerate(targets):
        concepts = set(table.entry_concepts[target])
        chosen = []
        for j in torch.argsort(scores[i], descending=True, stable=True).tolist():
            candidate = pool[j]
            if candidate == target or surfaces[candidate]["surface"] == surfaces[target]["surface"] \
                    or concepts & set(table.entry_concepts[candidate]):
                continue
            chosen.append(candidate)
            if len(chosen) == top:
                break
        out[target] = chosen
    return out


def build_neighbour_items(runs: dict[str, E5Run], out_dir: Path, *, concept_set: str = "rare", n_concepts: int = 10, top: int = 4,
                          space: str = "state", pool_size: int | None = None, seed: int = 0, calibration: int = 10,
                          wordnet: Any = None) -> dict[str, Any]:
    if wordnet is None:
        from nltk.corpus import wordnet
    names = list(runs)
    reference = runs[names[0]]
    digests = {name: run.table.digest() for name, run in runs.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"the runs use different alias tables: {digests}")
    table, ontology = reference.table, reference.ontology
    surfaces = canonical_surfaces(table, reference.tokenizer, reference.min_subtokens)
    targets = select_concepts(ontology, table, surfaces, concept_set=concept_set, count=n_concepts, seed=seed)
    frequency = np.asarray(ontology["train_frequency"])
    heldout = {int(e) for e in ontology["heldout_entries"]}
    pool = [e for e in range(len(table.entry_concepts)) if e not in heldout and frequency[e] >= 1 and e in surfaces
            and surfaces[e]["linkable"] and e not in targets]
    if pool_size is not None and len(pool) > pool_size:
        pool = sorted(np.random.default_rng(seed + 1).choice(pool, size=pool_size, replace=False).tolist())
    neighbours = {name: nearest(run, targets, pool, surfaces, space=space, top=top, table=table) for name, run in runs.items()}
    items: list[dict[str, Any]] = []
    for target in targets:
        listed: dict[int, dict[str, int]] = defaultdict(dict)
        for name in names:
            for rank, candidate in enumerate(neighbours[name][target], 1):
                listed[candidate][name] = rank
        for candidate, ranks in sorted(listed.items()):
            items.append({"id": f"pair-{target}-{candidate}", "study": "neighbour_pair", "calibration": False, "group": target,
                          "systems": sorted(ranks), "ranks": ranks, "entry": target, "candidate_entry": candidate,
                          "fields": {"concept": surfaces[target]["surface"], "candidate": surfaces[candidate]["surface"]}})
        rng = random.Random(seed * 7919 + target)
        for other in names[1:]:
            pair = [names[0], other]
            rng.shuffle(pair)
            lists = {name: ", ".join(surfaces[c]["surface"] for c in neighbours[name][target]) for name in pair}
            items.append({"id": f"preference-{target}-{other}", "study": "neighbour_preference", "calibration": False,
                          "group": target, "systems": pair, "order": pair, "entry": target,
                          "fields": {"concept": surfaces[target]["surface"], "list_a": lists[pair[0]], "list_b": lists[pair[1]]}})
    items += calibration_items("neighbour_pair", wordnet, count=calibration, seed=seed)
    items += calibration_items("neighbour_preference", wordnet, count=max(2, calibration // 2), seed=seed)
    manifest = {"study": "neighbours", "concept_set": concept_set, "space": space, "n_concepts": len(targets), "top": top,
                "n_per_system": len(targets) * top, "reference": names[0], "systems": names,
                "runs": {name: run.describe() for name, run in runs.items()}, "pool": len(pool), "seed": seed,
                "concepts": {int(t): {"surface": surfaces[t]["surface"], "frequency": int(frequency[t])} for t in targets},
                "neighbours": {name: {int(t): [int(c) for c in v] for t, v in n.items()} for name, n in neighbours.items()},
                "selection": ("rare: non-held-out single-concept entries with training frequency 1–9 and a linkable alias, "
                              "frequency-weighted sampling without replacement" if concept_set == "rare" else
                              "held-out single-concept entries with a linkable alias, uniform sampling"),
                "endpoint": "per system: pairs whose majority rating is 'strongly related' out of n_concepts × top; "
                            "one-tailed Fisher test, reference > other"}
    return write_items(out_dir, manifest, items)


# -- edge explanations -------------------------------------------------------------------------------------------

def _occurrence_window(corpus: TokenCorpus, boundaries: np.ndarray, start: int, end: int, *, before: int = 40,
                       after: int = 20) -> tuple[int, int]:
    doc = int(np.searchsorted(boundaries, start, side="left"))
    doc_start = int(boundaries[doc - 1]) + 1 if doc > 0 else 0
    doc_end = int(boundaries[doc]) if doc < len(boundaries) else len(corpus)
    return max(doc_start, start - before), min(doc_end, end + 1 + after)


@torch.no_grad()
def build_edge_items(run: E5Run, out_dir: Path, *, n: int = 40, top: int = 3, seed: int = 0, calibration: int = 10,
                     wordnet: Any = None) -> dict[str, Any]:
    if wordnet is None:
        from nltk.corpus import wordnet
    composer = run.composer
    if composer is None:
        raise ValueError("edge explanations need a composition channel")
    corpus = TokenCorpus.open(Path(run.config["data"]["eval"]))
    boundaries = eos_positions(corpus)
    schedule = composer.schedule
    degrees = schedule.degrees.cpu()
    spans = corpus.spans
    keep = spans["length"] >= run.min_subtokens
    first: dict[int, tuple[int, int]] = {}
    for s, e, entry in zip(spans["start"][keep].tolist(), spans["end"][keep].tolist(), spans["entry"][keep].tolist()):
        if entry not in first and degrees[entry] > top:
            first[entry] = (s, e)
    table, ontology = run.table, run.ontology
    polysemous = sorted(e for e in first if len(table.entry_concepts[e]) >= 2 and degrees[e] >= 4)
    rest = sorted(e for e in first if e not in set(polysemous) and degrees[e] >= 4)
    rng = random.Random(seed)
    chosen = rng.sample(polysemous, min(n, len(polysemous)))
    if len(chosen) < n:
        chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    frequency = np.asarray(ontology["train_frequency"])
    heldout = set(run.adapter.heldout_entries)
    surfaces = canonical_surfaces(table, run.tokenizer, run.min_subtokens)
    pool = [e for e in range(len(table.entry_concepts)) if frequency[e] >= 10 and e not in heldout and e in surfaces]
    pool_rows = F.normalize(entry_rows(run.channel, torch.as_tensor(pool)), dim=-1) if pool else None
    relations, atoms = ontology["relation_names"], ontology["atomic_names"]
    lineage = card_lineage(json.loads((run.path / "cards.json").read_text())) if (run.path / "cards.json").exists() else {}
    readable_relation = lambda r: base_name(r, relations, lineage, "relations")
    readable_atom = lambda a: base_name(a, atoms, lineage, "atomics")
    items = []
    for entry in sorted(chosen):
        s, e = first[entry]
        lo, hi = _occurrence_window(corpus, boundaries, s, e)
        ids = torch.as_tensor(np.asarray(corpus.tokens[lo:hi], dtype=np.int64))[None].to(run.device)
        local = {"batch": torch.zeros(1, dtype=torch.long, device=run.device),
                 "inject": torch.tensor([e - lo], device=run.device), "entry": torch.tensor([entry], device=run.device)}
        context = span_contexts(run.model, ids, local)
        explained = composer.explain(entry, None if context is None else context[0])
        tiebreak = random.Random(seed * 31 + entry)
        keyed = sorted(explained, key=lambda x: (-round(x["weight"], 6), tiebreak.random()))
        edges = {"top_edges": keyed[:top], "random_edges": random.Random(seed * 37 + entry).sample(explained, top)}
        text = lambda chosen_edges: "; ".join(edge_text(readable_relation(x["relation"]), readable_atom(x["filler"])) for x in chosen_edges)
        sentence = run.tokenizer.decode(np.asarray(corpus.tokens[lo:hi]).tolist()).strip()
        word = run.tokenizer.decode(np.asarray(corpus.tokens[s:e + 1]).tolist()).strip()
        explanations = {name: text(chosen_edges) for name, chosen_edges in edges.items()}
        if pool_rows is not None:
            own = F.normalize(entry_rows(run.channel, torch.tensor([entry])), dim=-1)[0]
            order = torch.argsort(pool_rows @ own, descending=True).tolist()
            similar = [surfaces[pool[j]]["surface"] for j in order if pool[j] != entry
                       and surfaces[pool[j]]["surface"] != word.lower()][:top]
            explanations["neighbours"] = "similar to " + ", ".join(similar)
        for name, explanation in explanations.items():
            items.append({"id": f"edges-{entry}-{name}", "study": "edge_explanation", "calibration": False, "group": entry,
                          "systems": [name], "entry": entry, "position": s,
                          "weights": [round(x["weight"], 4) for x in edges.get(name, [])],
                          "fields": {"sentence": sentence, "concept": word, "items": explanation}})
    items += calibration_items("edge_explanation", wordnet, count=calibration, seed=seed)
    manifest = {"study": "edges", "n": len(chosen), "top": top, "systems": ["top_edges", "random_edges", "neighbours"],
                "reference": "top_edges", "run": run.describe(), "seed": seed,
                "selection": "first eval-corpus occurrence per entry; polysemous union entries with ≥ 4 edges sampled "
                             "uniformly (seeded), topped up with other entries with ≥ 4 edges",
                "endpoint": "per system: items whose majority score is 2 (right) out of n; one-tailed Fisher test, "
                            "top_edges > random_edges and top_edges > neighbours"}
    return write_items(out_dir, manifest, items)


# -- M3 cards ---------------------------------------------------------------------------------------------------

def card_lineage(cards: Sequence[dict[str, Any]]) -> dict[tuple[str, int], int]:
    """(target, grown id) → the id it was split from."""
    lineage: dict[tuple[str, int], int] = {}
    for card in cards:
        if card.get("event") == "split":
            for child in card["children"]:
                if int(child) != int(card["parent"]):
                    lineage.setdefault((card.get("target", "atomics"), int(child)), int(card["parent"]))
    return lineage


def base_name(index: int, names: Sequence[str], lineage: dict[tuple[str, int], int], target: str) -> str:
    """Name of a dictionary vector, following splits back to an original (named) vector."""
    seen = set()
    while index >= len(names) and (target, index) in lineage and index not in seen:
        seen.add(index)
        index = lineage[(target, index)]
    return names[index] if index < len(names) else f"{target}#{index}"


def card_groups(cards: Sequence[dict[str, Any]], schedule: Any, *, target: str) -> list[dict[str, Any]]:
    """For each split card of `target`: per child, its uses `(entry, filler, relation)` in the final
    schedule — the child's own id and every id split from it later."""
    splits = sorted((c for c in cards if c.get("event") == "split" and c.get("target", "atomics") == target),
                    key=lambda c: int(c["step"]))

    def descendants(node: int, after: int) -> set[int]:
        out = {node}
        for later in splits:
            if int(later["step"]) > after and int(later["parent"]) in out:
                out |= {int(x) for x in later["children"]}
        return out

    relations, fillers, offsets = schedule.relations.tolist(), schedule.fillers.tolist(), schedule.offsets.tolist()
    owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    column = fillers if target == "atomics" else relations
    groups = []
    for card in splits:
        sets = []
        for child in card["children"]:
            ids = descendants(int(child), int(card["step"]))
            sets.append([(int(owner[i]), int(fillers[i]), int(relations[i])) for i, value in enumerate(column) if value in ids])
        groups.append({"card": card, "groups": sets})
    return groups


def build_card_items(run: E5Run, out_dir: Path, *, n: int = 40, members: int = 8, seed: int = 0, calibration: int = 10,
                     wordnet: Any = None, cards: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    if wordnet is None:
        from nltk.corpus import wordnet
    if cards is None:
        path = run.path / "cards.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (developmental runs write cards.json)")
        cards = json.loads(path.read_text())
    table, ontology = run.table, run.ontology
    frequency = np.asarray(ontology["train_frequency"])
    surfaces = canonical_surfaces(table, run.tokenizer, run.min_subtokens)
    atoms, relations = list(ontology["atomic_names"]), list(ontology["relation_names"])
    schedule = run.composer.schedule
    lineage = card_lineage(cards)

    items: list[dict[str, Any]] = []
    eligible = []
    for target, names in (("atomics", atoms), ("relations", relations)):
        for entry in card_groups(cards, schedule, target=target):
            sizes = [len({u[0] for u in g}) for g in entry["groups"]]
            if len(sizes) == 2 and min(sizes) >= 2:
                eligible.append((target, names, entry))
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(n, len(eligible)))

    def describe(uses: list[tuple[int, int, int]], target: str, concept: str) -> str:
        ranked = sorted(set(uses), key=lambda u: (-int(frequency[u[0]]), u))
        shown, seen = [], set()
        for owner, filler, relation in ranked:
            if owner in seen or owner not in surfaces:
                continue
            seen.add(owner)
            relation_name = base_name(relation, relations, lineage, "relations")
            if target == "atomics":
                shown.append(f"{surfaces[owner]['surface']} ({edge_text(relation_name, 'x:' + concept)})")
            else:
                shown.append(f"{surfaces[owner]['surface']} → {readable_atomic(base_name(filler, atoms, lineage, 'atomics'))}")
            if len(shown) == members:
                break
        return ", ".join(shown)

    for index, (target, names, entry) in enumerate(chosen):
        card = entry["card"]
        parent = base_name(int(card["parent"]), names, lineage, target)
        concept = readable_atomic(parent) if target == "atomics" else parent.replace("_", " ")
        group_a, group_b = entry["groups"]
        all_uses = group_a + group_b
        shuffled = list(all_uses)
        random.Random(seed * 13 + index).shuffle(shuffled)
        random_a, random_b = shuffled[:len(group_a)], shuffled[len(group_a):]
        for system, (a, b) in (("split", (group_a, group_b)), ("random", (random_a, random_b))):
            items.append({"id": f"card-{target}-{card['parent']}-{card['step']}-{system}", "study": "split_card", "calibration": False,
                          "group": f"{target}-{card['parent']}-{card['step']}", "systems": [system],
                          "card": {k: card.get(k) for k in ("event", "target", "step", "parent", "children", "gain", "p_value")},
                          "fields": {"concept": concept, "group_a": describe(a, target, concept), "group_b": describe(b, target, concept)}})
    items += calibration_items("split_card", wordnet, count=calibration, seed=seed)
    manifest = {"study": "cards", "n": len(chosen), "eligible_splits": len(eligible), "systems": ["split", "random"],
                "reference": "split", "run": run.describe(), "seed": seed, "members_shown": members,
                "selection": "split cards whose two children each have ≥ 2 using entries; seeded uniform sample of n",
                "endpoint": "per system: cards judged distinct (majority) out of n; one-tailed Fisher test, split > random"}
    return write_items(out_dir, manifest, items)


# -- grading ---------------------------------------------------------------------------------------------------------

def item_prompts(item: dict[str, Any]) -> list[str]:
    """Prompt paraphrases of one item; preference items are counterbalanced (paraphrase 2 swaps A and B)."""
    templates = STUDIES[item["study"]]["prompts"]
    fields = dict(item["fields"])
    if item["study"] == "neighbour_preference":
        swapped = {**fields, "list_a": fields["list_b"], "list_b": fields["list_a"]}
        return [templates[0].format(**fields), templates[1].format(**swapped)]
    return [template.format(**fields) for template in templates]


def call_value(item: dict[str, Any], call: int, verdict: dict[str, Any]) -> Any:
    """The verdict value of one call, mapped back to the unswapped order for preference items."""
    value = verdict.get(VERDICT_KEY[item["study"]])
    if item["study"] == "neighbour_preference" and call % 2 == 1 and value in {"A", "B"}:
        value = "B" if value == "A" else "A"
    return value


def consensus(study: str, values: Sequence[Any]) -> Any:
    """Per-item judgement: median of the ordinal pair rating; majority otherwise (preference ties → tie)."""
    if not values:
        return None
    if study == "neighbour_pair":
        ordinal = sorted(PAIR_ORDINAL[v] for v in values)
        middle = ordinal[len(ordinal) // 2]
        return next(label for label, value in PAIR_ORDINAL.items() if value == middle)
    if study == "neighbour_preference":
        counts = Counter(values)
        best = max(counts.values())
        winners = [v for v, c in counts.items() if c == best]
        return winners[0] if len(winners) == 1 else "tie"
    return majority(list(values))


def pending_calls(client: JudgeClient, items: Sequence[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        prompts = item_prompts(item)
        schema = STUDIES[item["study"]]["schema"]
        for call in range(client.calls):
            if not (client.cache_dir / f"{client._key(prompts[call % len(prompts)], schema, call)}.json").exists():
                count += 1
    return count


def grade_items(client: JudgeClient, items: Sequence[dict[str, Any]], *, seed: int = 0) -> list[dict[str, Any]]:
    order = list(items)
    random.Random(seed).shuffle(order)          # blinded, randomized order; systems never reach the prompt
    graded = []
    for item in order:
        records = client.grade(item["id"], item_prompts(item), STUDIES[item["study"]]["schema"])
        values = [call_value(item, r["call"], r["verdict"]) for r in records if "verdict" in r]
        graded.append({**item, "values": values, "consensus": consensus(item["study"], values),
                       "errors": [r.get("error") for r in records if "verdict" not in r]})
    return graded


def analyze(manifest: dict[str, Any], graded: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per system with one-tailed Fisher tests, preferences, κ and calibration accuracy."""
    out: dict[str, Any] = {"label": "LLM-graded estimate", "studies": {}}
    by_study: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graded:
        by_study[item["study"]].append(item)
    reference = manifest.get("reference")
    for study, rows in by_study.items():
        real = [r for r in rows if not r["calibration"]]
        calib = [r for r in rows if r["calibration"]]
        complete = [r["values"] for r in rows if len(r["values"]) == max((len(x["values"]) for x in rows), default=0) and r["values"]]
        entry: dict[str, Any] = {"items": len(real), "calibration_items": len(calib),
                                 "fleiss_kappa": fleiss_kappa(complete) if complete else None,
                                 "ungraded": sum(1 for r in rows if r["consensus"] is None)}
        if calib:
            judged = {r["id"]: r["consensus"] for r in calib if r["consensus"] is not None}
            entry["calibration"] = calibration_agreement(judged, {r["id"]: r["gold"] for r in calib})
        if study in SUCCESS:
            counts: dict[str, dict[str, int]] = {}
            if study == "neighbour_pair":
                totals = {s: int(manifest.get("n_per_system") or 0) for s in manifest.get("systems", [])}
                for r in real:
                    for system in r["systems"]:
                        counts.setdefault(system, {"success": 0, "graded": 0, "total": totals.get(system, 0)})
                        counts[system]["graded"] += r["consensus"] is not None
                        counts[system]["success"] += bool(r["consensus"] is not None and SUCCESS[study](r["consensus"]))
                for system, c in counts.items():
                    c["total"] = c["total"] or c["graded"]
            else:
                for r in real:
                    for system in r["systems"]:
                        c = counts.setdefault(system, {"success": 0, "graded": 0, "total": 0})
                        c["total"] += 1
                        c["graded"] += r["consensus"] is not None
                        c["success"] += bool(r["consensus"] is not None and SUCCESS[study](r["consensus"]))
            entry["counts"] = counts
            if study == "edge_explanation":
                entry["mean_score"] = {s: float(np.mean([r["consensus"] for r in real if s in r["systems"] and r["consensus"] is not None]))
                                       for s in counts if any(s in r["systems"] and r["consensus"] is not None for r in real)}
            ref = reference if reference in counts else None
            if ref is not None:
                entry["fisher"] = {other: {"reference": ref, "p_one_tailed": fisher_one_tailed(
                    counts[ref]["success"], counts[ref]["total"], c["success"], c["total"]),
                    "reference_success": counts[ref]["success"], "other_success": c["success"],
                    "reference_total": counts[ref]["total"], "other_total": c["total"]}
                    for other, c in counts.items() if other != ref}
        if study == "neighbour_preference":
            wins: dict[str, Counter] = defaultdict(Counter)
            for r in real:
                if r["consensus"] is None:
                    continue
                pair = r["order"]
                other = pair[1] if pair[0] == reference else pair[0]
                if r["consensus"] == "tie":
                    wins[other]["tie"] += 1
                else:
                    chosen = pair[0] if r["consensus"] == "A" else pair[1]
                    wins[other]["reference" if chosen == reference else "other"] += 1
            entry["preference"] = {}
            for other, c in wins.items():
                decided = c["reference"] + c["other"]
                p = sum(math.comb(decided, k) for k in range(c["reference"], decided + 1)) / 2 ** decided if decided else None
                entry["preference"][other] = {**c, "p_sign_one_tailed": p}
        out["studies"][study] = entry
    return out


def render(analysis: dict[str, Any], header: dict[str, Any]) -> str:
    manifest = header["manifest"]
    lines = [f"# E5.3 rating study — {manifest.get('study')} ({analysis['label']})", "",
             f"Items `{header['items']}`; judge `{header['judge']['model']}` via {header['judge']['harness']}, "
             f"{header['judge']['calls']} calls per item over {header['judge']['paraphrases']} paraphrases; "
             f"new calls this run: {header['judge']['new_calls']}, spend ${header['judge']['spent_usd']:.2f}.", ""]
    if manifest.get("selection"):
        lines += [f"Selection (pre-registered): {manifest['selection']}.", f"Endpoint: {manifest.get('endpoint')}.", ""]
    for study, entry in analysis["studies"].items():
        lines += [f"## {study}", "", f"Items {entry['items']} (+ {entry['calibration_items']} calibration); Fleiss κ across calls "
                  f"{fmt(entry['fleiss_kappa'], 3)}; calibration accuracy "
                  f"{fmt((entry.get('calibration') or {}).get('accuracy'), 3)} on {(entry.get('calibration') or {}).get('items', 0)} items; "
                  f"ungraded {entry['ungraded']}.", ""]
        if entry.get("counts"):
            lines += ["| system | success | of | rate |", "|---|---:|---:|---:|"]
            for system, c in entry["counts"].items():
                lines.append(f"| {system} | {c['success']} | {c['total']} | {fmt(c['success'] / c['total'] if c['total'] else None, 3)} |")
            for other, f in (entry.get("fisher") or {}).items():
                lines.append(f"\nOne-tailed Fisher, {f['reference']} ({f['reference_success']}/{f['reference_total']}) > {other} "
                             f"({f['other_success']}/{f['other_total']}): p = {f['p_one_tailed']:.3g}.")
            lines.append("")
        for other, p in (entry.get("preference") or {}).items():
            lines.append(f"Preference vs {other}: reference {p.get('reference', 0)}, other {p.get('other', 0)}, tie {p.get('tie', 0)}; "
                         f"one-tailed sign test p = {fmt(p['p_sign_one_tailed'], 4)}.")
        lines.append("")
    return "\n".join(lines)


def fake_runner(prompt: str, schema: dict[str, Any], model: str) -> dict[str, Any]:
    """Deterministic stand-in judge for tests (no API calls)."""
    properties = schema["properties"]
    digest = int(sha256_text(prompt), 16)
    if "rating" in properties:
        verdict = {"rating": PAIR_LABELS[digest % 3]}
    elif "choice" in properties:
        verdict = {"choice": "AB"[digest % 2]}
    elif "score" in properties:
        verdict = {"score": digest % (properties["score"]["maximum"] + 1)}
    else:
        verdict = {"distinct": bool(digest % 2), "label_a": "a", "label_b": "b"}
    return {"structured_output": verdict, "total_cost_usd": 0.0, "modelUsage": {model: {}}}


def run_grade(items_dir: Path, output: Path, *, runner: str = "cli", calls: int = 3, model: str = "claude-opus-5-5", seed: int = 0,
              max_new_calls: int | None = None, cache_dir: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    from ..provenance import write_run_metadata
    manifest, items = load_rating_items(items_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    harness = {"cli": "claude -p headless (judging.claude_cli_runner)", "exchange": "Claude Code session (file exchange)",
               "fake": "fake runner (tests)"}[runner]
    client = JudgeClient(cache_dir or output / "cache", model=model, calls=calls,
                         runner={"cli": claude_cli_runner, "fake": fake_runner}.get(runner),
                         exchange_dir=output / "exchange" if runner == "exchange" else None)
    new_calls = pending_calls(client, items)
    estimate = {"items": len(items), "calls_per_item": calls, "new_calls": new_calls, "estimated_usd_at_0.02": round(0.02 * new_calls, 2)}
    if dry_run:
        return {"dry_run": estimate}
    if max_new_calls is not None and new_calls > max_new_calls and runner != "fake":
        raise RuntimeError(f"{new_calls} uncached judge calls needed, more than --max-new-calls {max_new_calls}")
    started = time.monotonic()
    graded = grade_items(client, items, seed=seed)
    if runner == "exchange":
        outstanding = sorted((output / "exchange" / "requests").glob("*.json"))
        if outstanding:
            return {"pending": len(outstanding), "requests": str(output / "exchange" / "requests")}
    analysis = analyze(manifest, graded)
    header = {"items": str(items_dir), "manifest": manifest,
              "judge": {"model": model, "harness": harness, "calls": calls, "paraphrases": 2, "new_calls": new_calls,
                        "spent_usd": client.spent_usd, "cache": str(client.cache_dir)}}
    with (output / "verdicts.jsonl").open("w") as handle:
        for item in graded:
            handle.write(json.dumps(json_ready({k: item.get(k) for k in ("id", "study", "calibration", "systems", "gold", "values",
                                                                         "consensus", "errors")})) + "\n")
    write_json(output / "summary.json", {**header, "analysis": analysis, "seconds": time.monotonic() - started})
    (output / "report.md").write_text(render(analysis, header))
    write_run_metadata(output, {"experiment": "e5.3-rating", "items": str(items_dir), "runner": runner, "calls": calls,
                                "model": model, "seed": seed}, device="cpu", judge=header["judge"])
    return {**header, "analysis": analysis}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("study", choices=["neighbours", "edges", "cards"])
    build.add_argument("--runs", nargs="+", default=[], help="name=RUN pairs; the first is the reference (neighbours)")
    build.add_argument("--run", type=Path, help="run folder (edges, cards)")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--set", dest="concept_set", default="rare", choices=["rare", "heldout"])
    build.add_argument("--space", default="state", choices=["state", "row"])
    build.add_argument("--concepts", type=int, default=10); build.add_argument("--top", type=int, default=4)
    build.add_argument("--n", type=int, default=40); build.add_argument("--pool", type=int, default=None)
    build.add_argument("--calibration", type=int, default=10); build.add_argument("--seed", type=int, default=0)
    build.add_argument("--device", default=None)
    grade = sub.add_parser("grade")
    target = grade.add_mutually_exclusive_group(required=True)
    target.add_argument("--items", type=Path); target.add_argument("--t1", action="store_true", help=f"every item dir under {T1_ITEMS}")
    grade.add_argument("--output", type=Path, required=True)
    grade.add_argument("--runner", default="cli", choices=["cli", "exchange", "fake"])
    grade.add_argument("--calls", type=int, default=3); grade.add_argument("--model", default="claude-opus-5-5")
    grade.add_argument("--seed", type=int, default=0); grade.add_argument("--max-new-calls", type=int, default=None)
    grade.add_argument("--cache", type=Path, default=None, help="shared verdict cache (default <output>/cache)")
    grade.add_argument("--dry-run", action="store_true", help="count uncached calls and the estimated cost only")
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.study == "neighbours":
            pairs = [r.split("=", 1) for r in args.runs]
            runs = {name: open_run(Path(path), device=args.device) for name, path in pairs}
            result = build_neighbour_items(runs, args.output, concept_set=args.concept_set, n_concepts=args.concepts, top=args.top,
                                           space=args.space, pool_size=args.pool, seed=args.seed, calibration=args.calibration)
        elif args.study == "edges":
            result = build_edge_items(open_run(args.run, device=args.device), args.output, n=args.n, seed=args.seed,
                                      calibration=args.calibration)
        else:
            result = build_card_items(open_run(args.run, device=args.device), args.output, n=args.n, seed=args.seed,
                                      calibration=args.calibration)
        print(json.dumps(result.get("counts"), indent=2))
        return 0
    directories = sorted(p.parent for p in T1_ITEMS.glob("*/manifest.json")) if args.t1 else [args.items]
    if not directories:
        print(f"no item directories under {T1_ITEMS} (the T1 WP writes them)")
        return 0
    status = 0
    for directory in directories:
        output = args.output / directory.name if args.t1 else args.output
        result = run_grade(directory, output, runner=args.runner, calls=args.calls, model=args.model, seed=args.seed,
                           max_new_calls=args.max_new_calls, cache_dir=args.cache, dry_run=args.dry_run)
        if "pending" in result:
            print(f"{result['pending']} requests outstanding in {result['requests']}; answer them, then re-run")
            status = 3
        else:
            print(json.dumps(result.get("dry_run") or {s: {k: v for k, v in e.items() if k in ("counts", "fisher", "fleiss_kappa")}
                                                       for s, e in result["analysis"]["studies"].items()}, indent=2, default=str))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
