"""Authoring baselines and the teacher author for E7 (WP-E7; formulation §5.4, related-work §8).

- **Hearst / lexico-syntactic patterns** (Hearst 1992 and the usual part–whole and copula
  extensions): `hearst_extract` finds `X such as A, B and C`, `A and other X`, `X, including A`,
  `A is a kind of X`, `A is part of X`, `A is made of X`, … around candidate concepts. Patterns emit
  generic labels (`is_a`, `part_of`, `has_part`, `member_of`, `made_of`, `opposite_of`) that a track
  maps to its closed relation vocabulary. Without a POS tagger, the other noun phrase is the longest
  1–3-word sub-span (next to the pattern) that resolves to a dictionary filler.
- **OLLM / LLMs4OL-style direct prompting** (no candidate screen): `direct_prompt` asks the host to
  list concepts of a passage with their relations; `parse_direct` reads `concept | relation | filler`.
- **Random frames with matched degree**: `random_frames`.
- **EntiGraph-style synthetic text** (Yang et al. 2025): `entigraph_prompt` asks the writer for a
  passage relating two entities of a document, grounded in the document excerpt.
- **Teacher author**: `TeacherAuthor` calls Claude Code headless (`judging.claude_cli_runner`, model
  pinned) for batches of candidates under the same closed vocabulary, caches every response by
  (model, prompt, schema) and records the model ids and cost; a fake runner serves tests.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .authoring import STOPWORDS, parse_proposals
from .span_channel import normalize_alias

# ---------------------------------------------------------------------------------------------
# Hearst patterns

GENERIC_RELATIONS = ("is_a", "part_of", "has_part", "member_of", "made_of", "opposite_of")
_NP = r"(?:[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})"
_LIST = r"(?:[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})(?:\s*,\s*(?:and\s+|or\s+)?[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})*(?:,?\s+(?:and|or)\s+[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})?"
_ART = r"(?:a|an|the|one)\s+"
# (regex, relation of the hyponym/part side, which group holds the concept side, which the other NP;
#  "list" groups hold enumerations whose every item is a concept side)
HEARST_PATTERNS: list[tuple[str, re.Pattern, str, str, str]] = [
    ("such_as", re.compile(rf"(?P<other>{_NP})\s*,?\s+such\s+as\s+(?P<list>{_LIST})", re.I), "is_a", "list", "other"),
    ("such_x_as", re.compile(rf"\bsuch\s+(?P<other>{_NP})\s+as\s+(?P<list>{_LIST})", re.I), "is_a", "list", "other"),
    ("and_other", re.compile(rf"(?P<list>{_LIST})\s*,?\s+(?:and|or)\s+other\s+(?P<other>{_NP})", re.I), "is_a", "list", "other"),
    ("including", re.compile(rf"(?P<other>{_NP})\s*,\s*(?:including|especially|particularly)\s+(?P<list>{_LIST})", re.I), "is_a", "list", "other"),
    ("kind_of", re.compile(rf"(?P<concept>{_NP})\s+(?:is|are|was|were)\s+(?:{_ART})?(?:kind|type|form|sort|variety|class|species|genus)\s+of\s+(?:{_ART})?(?P<other>{_NP})", re.I), "is_a", "concept", "other"),
    ("copula", re.compile(rf"(?P<concept>{_NP})\s+(?:is|was)\s+{_ART}(?P<other>{_NP})", re.I), "is_a", "concept", "other"),
    ("part_of", re.compile(rf"(?P<concept>{_NP})\s+(?:is|are|was|were|forms?)\s+(?:{_ART})?part\s+of\s+(?:{_ART})?(?P<other>{_NP})", re.I), "part_of", "concept", "other"),
    ("consists_of", re.compile(rf"(?P<concept>{_NP})\s+(?:consists?\s+of|is\s+made\s+up\s+of|is\s+composed\s+of|comprises)\s+(?P<list>{_LIST})", re.I), "has_part", "concept", "list"),
    ("made_of", re.compile(rf"(?P<concept>{_NP})\s+(?:is|are|was|were)\s+made\s+(?:of|from)\s+(?P<other>{_NP})", re.I), "made_of", "concept", "other"),
    ("member_of", re.compile(rf"(?P<concept>{_NP})\s+(?:is|are|was|were)\s+(?:{_ART})?members?\s+of\s+(?:{_ART})?(?P<other>{_NP})", re.I), "member_of", "concept", "other"),
    ("opposite_of", re.compile(rf"(?P<concept>{_NP})\s+(?:is|are)\s+the\s+opposite\s+of\s+(?P<other>{_NP})", re.I), "opposite_of", "concept", "other"),
]


MAX_SENTENCE = 400          # longer "sentences" (lists, tables, run-ons) are skipped: bounded regex backtracking
_TRIGGER = re.compile(r"\b(?:such|other|including|especially|particularly|kind|type|form|sort|variety|class|species|"
                      r"genus|is|was|part|parts|consists?|made|composed|comprises|members?|opposite)\b", re.I)


def _split_list(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*,\s*|\s+(?:and|or)\s+", text) if p.strip()]


def _subspans(words: list[str], *, from_end: bool) -> Iterable[str]:
    """Contiguous sub-spans of ≤ 3 words, longest first, anchored at the end (head noun) or start."""
    for n in range(min(3, len(words)), 0, -1):
        starts = range(len(words) - n, -1, -1) if from_end else range(0, len(words) - n + 1)
        for i in starts:
            piece = words[i:i + n]
            if piece[0].lower() in STOPWORDS or piece[-1].lower() in STOPWORDS:
                continue
            yield " ".join(piece)


def _concept_in(phrase: str, candidates: dict[str, Any], *, anchor_end: bool) -> str | None:
    """The longest candidate among the phrase's 1–3-word sub-spans, nearest the given anchor first
    (the NP slots are up to four words, so determiners and stray words may surround the concept)."""
    words = phrase.split()
    for n in range(min(3, len(words)), 0, -1):
        starts = range(len(words) - n, -1, -1) if anchor_end else range(0, len(words) - n + 1)
        for i in starts:
            key = normalize_alias(" ".join(words[i:i + n]))
            if key in candidates:
                return key
    return None


def hearst_extract(text: str, candidates: dict[str, Any], resolve: Callable[[str], Any | None]) -> list[tuple[str, str, str]]:
    """`(candidate, generic relation, filler)` triples found by the patterns in `text`.

    The concept side must end with (hyponym/part side before the pattern) or start with (list items
    after it) a candidate surface (normalized keys of `candidates`); the other NP is the longest
    1–3-word sub-span — nearest the pattern — for which `resolve` returns a filler."""
    found: list[tuple[str, str, str]] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if len(sentence) > MAX_SENTENCE or not _TRIGGER.search(sentence):
            continue
        for _, pattern, relation, concept_group, other_group in HEARST_PATTERNS:
            for match in pattern.finditer(sentence):
                if concept_group == "list":
                    list_first = match.start("list") < match.start("other")
                    concepts = [_concept_in(item, candidates, anchor_end=list_first) for item in _split_list(match["list"])]
                    others = [match["other"]]
                elif other_group == "list":
                    concepts = [_concept_in(match["concept"], candidates, anchor_end=True)]
                    others = _split_list(match["list"])
                else:
                    concepts = [_concept_in(match["concept"], candidates, anchor_end=True)]
                    others = [match["other"]]
                for concept in filter(None, concepts):
                    for other in others:
                        # the other NP's head is at its end when it precedes the pattern, else at its start
                        before = match.start("other") < match.start(concept_group) if other_group == "other" else False
                        for piece in _subspans(other.split(), from_end=before):
                            key = normalize_alias(piece)
                            if key != concept and resolve(key) is not None:
                                found.append((concept, relation, key))
                                break
    return found


# ---------------------------------------------------------------------------------------------
# OLLM / LLMs4OL-style direct prompting (no candidate screen)

def direct_prompt(passage: str, relations: Sequence[str], demonstrations: Sequence[dict[str, Any]] = ()) -> str:
    """Ask for the concepts of a passage and their facts, `concept | relation | related concept` per line."""
    blocks = [f"Each example lists important concepts mentioned in a text with facts about them, one per line as "
              f"`concept | relation | related concept`, using only these relations: {', '.join(relations)}."]
    for demo in demonstrations:
        lines = "\n".join(f"{c} | {r} | {f}" for c, r, f in demo["triples"])
        blocks.append(f"Text: {' '.join(demo['context'].split())}\nFacts:\n{lines}")
    blocks.append(f"Text: {' '.join(passage.split())}\nFacts:\n")
    return "\n\n".join(blocks)


def parse_direct(text: str, relations: Sequence[str]) -> list[tuple[str, str, str]]:
    """`(concept, relation, filler)` from `concept | relation | filler` lines; closed vocabulary only."""
    out = []
    for line in text.split("\n\n")[0].splitlines():
        parts = [p.strip(" -*`\t.") for p in line.split("|")]
        if len(parts) != 3:
            continue
        concept = normalize_alias(parts[0])
        for relation, filler in parse_proposals(f"{parts[1]}: {parts[2]}", relations):
            if concept and len(concept) < 60:
                out.append((concept, relation, filler))
    return out


# ---------------------------------------------------------------------------------------------
# Random frames with matched degree

def random_frames(degrees: dict[Any, int], relations: Sequence[str], fillers: Sequence[Any], *, seed: int,
                  relation_weights: Sequence[float] | None = None) -> dict[Any, list[tuple[str, Any]]]:
    """One random frame per key with the given degree: relations drawn from `relation_weights` (default
    uniform), fillers uniformly from `fillers`, no duplicate edge within a frame."""
    rng = random.Random(seed)
    frames = {}
    for key in sorted(degrees, key=str):
        frame: list[tuple[str, Any]] = []
        seen = set()
        for _ in range(100 * max(1, degrees[key])):
            if len(frame) >= degrees[key]:
                break
            edge = (rng.choices(list(relations), weights=relation_weights)[0], rng.choice(list(fillers)))
            if edge not in seen:
                seen.add(edge); frame.append(edge)
        frames[key] = frame
    return frames


# ---------------------------------------------------------------------------------------------
# EntiGraph-style synthetic text

def entigraph_prompt(entity: str, other: str, context: str) -> str:
    """A writing prompt relating two entities of one document (EntiGraph's pair-relation step)."""
    return (f"Text: {' '.join(context.split())}\n\n"
            f"Write a short, factual paragraph that explains what \"{entity}\" is and how it relates to \"{other}\", "
            f"based on the text above.\nParagraph:")


def entity_pairs(entities_by_document: dict[int, Sequence[str]], *, per_entity: int, seed: int) -> list[tuple[str, str, int]]:
    """Up to `per_entity` (entity, other entity, document) pairs per entity, others drawn from the same document."""
    rng = random.Random(seed)
    used: dict[str, int] = {}
    pairs = []
    for document in sorted(entities_by_document):
        names = sorted(set(entities_by_document[document]))
        for name in names:
            others = [o for o in names if o != name]
            if not others or used.get(name, 0) >= per_entity:
                continue
            pairs.append((name, rng.choice(others), document))
            used[name] = used.get(name, 0) + 1
    return pairs


# ---------------------------------------------------------------------------------------------
# Teacher author (Claude Code headless)

def teacher_schema(relations: Sequence[str]) -> dict[str, Any]:
    edge = {"type": "object", "properties": {"relation": {"type": "string", "enum": list(relations)},
                                             "filler": {"type": "string"}}, "required": ["relation", "filler"]}
    concept = {"type": "object", "properties": {"id": {"type": "string"}, "edges": {"type": "array", "items": edge}},
               "required": ["id", "edges"]}
    return {"type": "object", "properties": {"concepts": {"type": "array", "items": concept}}, "required": ["concepts"]}


def teacher_prompt(items: Sequence[dict[str, Any]], relations: Sequence[str], descriptions: dict[str, str], *,
                   filler_hint: str, max_edges: int = 6) -> str:
    """One request for several candidates: each item has `id`, `surface` and `contexts` (texts)."""
    vocabulary = "\n".join(f"- {r}: {descriptions.get(r, r)}" for r in relations)
    lines = ["You are building an ontology from text. For each concept below, read its contexts and list facts "
             "about the concept as (relation, related concept) pairs, using ONLY these relations:", vocabulary, "",
             f"Each related concept must be {filler_hint} Give at most {max_edges} facts per concept, only facts that "
             "are true of the concept as it is used in the contexts, and none if the phrase is not a meaningful concept. "
             "Answer with JSON matching the schema; use each concept's id.", ""]
    for item in items:
        lines.append(f"Concept {item['id']}: \"{item['surface']}\"")
        for k, context in enumerate(item["contexts"], 1):
            lines.append(f"  Context {k}: {' '.join(context.split())}")
        lines.append("")
    return "\n".join(lines)


def _default_runner(prompt: str, schema: dict[str, Any], model: str) -> dict[str, Any]:
    from .judging import claude_cli_runner
    return claude_cli_runner(prompt, schema, model)


@dataclass
class TeacherAuthor:
    """Cached teacher calls: one JSON response per (model, prompt, schema), stored under `cache_dir`."""

    cache_dir: Path
    relations: Sequence[str]
    descriptions: dict[str, str]
    filler_hint: str = "a short, general, lowercase noun phrase (1-3 words, singular), preferably a common dictionary word."
    model: str = "claude-opus-5-5"
    retries: int = 2
    runner: Callable[[str, dict[str, Any], str], dict[str, Any]] | None = None
    spent_usd: float = field(default=0.0, init=False)
    calls: int = field(default=0, init=False)
    cached: int = field(default=0, init=False)

    def _key(self, prompt: str, schema: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps({"model": self.model, "prompt": prompt, "schema": schema}, sort_keys=True).encode()).hexdigest()

    def request(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Edges for a batch of items; returns the cached record (`verdict` = {"concepts": [...]}) or one with `error`."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prompt = teacher_prompt(items, self.relations, self.descriptions, filler_hint=self.filler_hint)
        schema = teacher_schema(self.relations)
        key = self._key(prompt, schema)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text())
        runner = self.runner or _default_runner
        record: dict[str, Any] = {"model": self.model, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                  "items": [i["id"] for i in items]}
        for attempt in range(self.retries + 1):
            try:
                self.calls += 1
                envelope = runner(prompt, schema, self.model)
                verdict = envelope.get("structured_output")
                if not isinstance(verdict, dict) or envelope.get("is_error") or not isinstance(verdict.get("concepts"), list):
                    raise ValueError("no structured output")
                record.update(verdict=verdict, cost_usd=envelope.get("total_cost_usd"),
                              models_used=sorted((envelope.get("modelUsage") or {}).keys()), attempts=attempt + 1)
                self.spent_usd += float(envelope.get("total_cost_usd") or 0)
                break
            except (ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
                record.update(error=str(error)[:300], attempts=attempt + 1)
        if "verdict" in record:
            record.pop("error", None)
            path.write_text(json.dumps(record, indent=2) + "\n")
        return record

    def author(self, items: Sequence[dict[str, Any]], *, batch: int = 8) -> dict[str, dict[str, Any]]:
        """Per item id: `edges` [(relation, filler)], or `error`; items are sent `batch` at a time."""
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            record = self.request(chunk)
            if "verdict" not in record:
                for item in chunk:
                    out[item["id"]] = {"edges": [], "error": record.get("error")}
                continue
            answered = {c.get("id"): c.get("edges", []) for c in record["verdict"]["concepts"] if isinstance(c, dict)}
            for item in chunk:
                edges = []
                for edge in answered.get(item["id"], []):
                    if isinstance(edge, dict) and edge.get("relation") in self.relations:
                        filler = normalize_alias(str(edge.get("filler", "")).split(",")[0].split(";")[0])
                        if filler and len(filler) < 60:
                            edges.append((edge["relation"], filler))
                out[item["id"]] = {"edges": list(dict.fromkeys(edges)), "models_used": record.get("models_used"),
                                   "prompt_sha256": record["prompt_sha256"]}
        return out
