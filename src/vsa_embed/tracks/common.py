"""Shared pieces of the application tracks (E8 T3–T6): synthetic concepts, task items, hashing.

Every track exposes the same item formats so one probe harness serves all of them:

- **multiple-choice cloze** (`choices` + `label`): score each choice continuation of `prompt` by
  its log-likelihood; no generation. Items of one `(concept, relation)` share a `group` and differ
  in `paraphrase`, which gives the E5.4 paraphrase-consistency test for free.
- **entailment** (`statement` + `label` ∈ {0, 1}): a true frame fact and a corrupted one (same
  relation, wrong filler); score by likelihood or probe.
- **linear probe** (`text` + `span` + `label`): a mention of the concept in context; a linear probe
  reads the hidden state at the span's last subtoken and predicts `label` (a frame filler class).

`split` is `train` (a concept linked in training), `heldout` (frozen linker holdout: in the
evaluation corpus only, never linked in training) or `synthetic` (invented, contamination-free
name with a frame and no text anywhere; the E5.4 zero-shot scenario).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_WORD = re.compile(r"[a-z]+")


@dataclass
class SyntheticConcept:
    """An invented concept defined only by its frame (E5.4 zero-shot scenario)."""

    name: str
    aliases: list[str]
    frame: list[tuple[str, str]]                       # (relation name, atomic name)
    facts: dict[str, list[str]] = field(default_factory=dict)   # relation → readable filler names
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "aliases": self.aliases, "frame": [list(edge) for edge in self.frame],
                "facts": self.facts, "meta": self.meta}


def wordnet_forbidden() -> set[str]:
    """Lower-cased WordNet lemma words (empty if WordNet data are not installed)."""
    try:
        from nltk.corpus import wordnet as wn
        return {part for lemma in wn.all_lemma_names() for part in lemma.lower().replace("-", "_").split("_") if part}
    except LookupError:
        return set()


def text_vocabulary(texts: Iterable[str]) -> set[str]:
    """Lower-cased alphabetic words of a text sample (used to keep invented names novel)."""
    words: set[str] = set()
    for text in texts:
        words.update(_WORD.findall(text.lower()))
    return words


def occurrences(aliases: Sequence[str], texts: Iterable[str]) -> dict[str, int]:
    """Case-insensitive counts of each alias starting at a word boundary in a text sample — the
    linker's matching rule (prefix-causal: the right boundary is not checked) — for the
    contamination check of synthetic names."""
    joined = "\n".join(t.lower() for t in texts)
    counts = {}
    for alias in aliases:
        key = " ".join(alias.lower().split())
        counts[alias] = sum(1 for _ in re.finditer(r"(?<![\w])" + re.escape(key), joined)) if key else 0
    return counts


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write rows (sorted keys) and return `{rows, sha256}` of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    payload = ("\n".join(lines) + "\n").encode() if lines else b""
    path.write_bytes(payload)
    return {"rows": len(lines), "sha256": hashlib.sha256(payload).hexdigest()}


def names_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


@dataclass
class RelationTemplates:
    """Prompt paraphrases, an entailment statement and the answer format of one relation."""

    prompts: list[str]                 # "{x} is owned by"; the choice is appended
    statement: str                     # "{x} is owned by {y}."
    answer: str = " {y}"               # continuation format of a choice


def _relations(concept: dict[str, Any], templates: dict[str, Any], limit: int | None, rng: random.Random) -> list[str]:
    usable = [r for r, fillers in sorted(concept["facts"].items()) if r in templates and fillers]
    return usable if limit is None or len(usable) <= limit else sorted(rng.sample(usable, limit))


def choice_items(concepts: Sequence[dict[str, Any]], templates: dict[str, RelationTemplates],
                 pools: dict[str, Sequence[str]], *, track: str, task: str, seed: int,
                 choices: int = 4, max_paraphrases: int = 3, max_relations: int | None = None) -> list[dict[str, Any]]:
    """Multiple-choice cloze items from frame facts.

    `concepts`: dicts with `concept` (id), `surface` (name used in the prompt), `split` and `facts`
    (relation → true filler names). Distractors are drawn from `pools[relation]` minus every true
    filler of that concept; a relation with too small a pool is skipped. `max_relations` keeps a
    seeded sample of each concept's relations."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for concept in concepts:
        for relation in _relations(concept, templates, max_relations, rng):
            fillers = concept["facts"][relation]
            pool = sorted(set(pools.get(relation, ())) - set(fillers))
            if len(pool) < choices - 1:
                continue
            truth = rng.choice(sorted(fillers))
            options = [truth] + rng.sample(pool, choices - 1)
            rng.shuffle(options)
            spec = templates[relation]
            group = f"{track}-{task}-{len(items):06d}"
            for paraphrase, prompt in enumerate(spec.prompts[:max_paraphrases]):
                items.append({
                    "id": f"{group}-p{paraphrase}", "group": group, "track": track, "task": task,
                    "split": concept["split"], "concept": concept["concept"], "surface": concept["surface"],
                    "relation": relation, "paraphrase": paraphrase, "prompt": prompt.format(x=concept["surface"]),
                    "choices": [spec.answer.format(y=o) for o in options], "label": options.index(truth),
                })
    return items


def entailment_items(concepts: Sequence[dict[str, Any]], templates: dict[str, RelationTemplates],
                     pools: dict[str, Sequence[str]], *, track: str, task: str, seed: int,
                     max_relations: int | None = None) -> list[dict[str, Any]]:
    """One true and one corrupted (same relation, wrong filler) statement per concept × relation."""
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for concept in concepts:
        for relation in _relations(concept, templates, max_relations, rng):
            fillers = concept["facts"][relation]
            pool = sorted(set(pools.get(relation, ())) - set(fillers))
            if not pool:
                continue
            pair = f"{track}-{task}-{len(items) // 2:06d}"
            for label, filler in ((1, rng.choice(sorted(fillers))), (0, rng.choice(pool))):
                items.append({"id": f"{pair}-{label}", "pair": pair, "track": track, "task": task,
                              "split": concept["split"], "concept": concept["concept"], "surface": concept["surface"],
                              "relation": relation, "statement": templates[relation].statement.format(x=concept["surface"], y=filler),
                              "label": label})
    return items


def mention_item(text: str, surface: str, **fields: Any) -> dict[str, Any] | None:
    """A linear-probe item: `text` with the char span of the first case-insensitive `surface`."""
    start = text.lower().find(surface.lower())
    if start < 0:
        return None
    return {**fields, "text": text, "span": [start, start + len(surface)], "surface": surface}


def split_counts(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item["split"]] = counts.get(item["split"], 0) + 1
    return dict(sorted(counts.items()))
