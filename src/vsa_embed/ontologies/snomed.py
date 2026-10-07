"""SNOMED CT International (RF2 snapshot) as a frame ontology — the licensed clinical track T1c.

**Licence.** SNOMED CT is licensed content (decision 58): this module reads the release files in place and
returns in-memory structures; nothing it produces (concept names, aliases, frames) is ever committed. The
only SNOMED identifiers in the code are the public RF2 metadata constants and the concept-model attribute
types named in `ATTRIBUTE_NAMES`.

Concept = an active concept in one of the selected top-level hierarchies (`HIERARCHIES`, chosen for clinical
notes). Frame edges, from the active **inferred** relationships (`sct2_Relationship_Snapshot`; relationship
groups are flattened):

- `hierarchy`: the top-level hierarchy atom (`top:<label>`; like MeSH's `branch_top`);
- `semantic_tag`: the FSN's semantic tag (`tag:<tag>`, e.g. disorder vs finding; like `branch_second`);
- `is_a`: the inferred parents;
- one relation per attribute type with at least `min_relation_edges` edges among the selected concepts
  (finding site, associated morphology, causative agent, procedure site, method, …; `ATTRIBUTE_NAMES`, other
  types `attr:<id>`).

Fillers come from a bounded atomic dictionary: the `max_atomics` concepts most used as fillers (plus the
hierarchy and tag atoms); an out-of-dictionary filler falls back to its nearest in-dictionary ancestor
(breadth-first over inferred parents), as in the MeSH and WordNet adapters. Edges are ordered hierarchy, tag,
is-a, then attributes by `relation priority` (the frequency order), and capped at `max_degree`.

Aliases (`SnomedAliasPolicy`, the decision-19 lesson for a case-insensitive linker): active English
descriptions accepted in the US or GB English language refset — the FSN without its semantic tag and every
synonym. Dropped: English function words; all-caps abbreviations (no lowercase letter, at most
`max_abbreviation_chars` characters without spaces, e.g. a bare acronym — the linker lowercases text, so an
acronym would link every homograph); aliases shorter than `min_chars`; and aliases with more than `max_words`
words (long post-coordinated FSN wordings never occur verbatim in notes and only grow the linker's trie).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..span_channel import normalize_alias
from .mesh import FUNCTION_WORDS
from .wordnet import FrameOntology

# RF2 metadata constants (public RF2 specification).
ROOT = "138875005"
IS_A = "116680003"
FSN = "900000000000003001"
SYNONYM = "900000000000013009"
INFERRED = "900000000000011006"
US_ENGLISH = "900000000000509007"
GB_ENGLISH = "900000000000508004"
PREFERRED = "900000000000548007"
ACCEPTABLE = "900000000000549004"

# Top-level hierarchies (children of the root) → labels; the default selection covers what clinical notes talk about.
HIERARCHIES: dict[str, str] = {
    "404684003": "clinical_finding", "71388002": "procedure", "123037004": "body_structure", "105590001": "substance",
    "373873005": "product", "410607006": "organism", "363787002": "observable_entity",
    "243796009": "situation", "272379006": "event", "123038009": "specimen", "260787004": "physical_object",
    "362981000": "qualifier_value", "254291000": "staging_scale", "48176007": "social_context",
    "308916002": "environment_location", "78621006": "physical_force", "370115009": "special_concept",
    "419891008": "record_artifact", "900000000000441003": "model_component",
}
DEFAULT_HIERARCHIES = ("clinical_finding", "procedure", "body_structure", "substance", "product", "organism",
                       "observable_entity")
# Concept-model attribute types → relation names (other attribute types are named `attr:<id>`).
ATTRIBUTE_NAMES: dict[str, str] = {
    "363698007": "finding_site", "116676008": "associated_morphology", "246075003": "causative_agent",
    "363704007": "procedure_site", "405813007": "procedure_site_direct", "405814001": "procedure_site_indirect",
    "260686004": "method", "127489000": "has_active_ingredient", "762949000": "has_precise_active_ingredient",
    "42752001": "due_to", "370135005": "pathological_process", "363714003": "interprets",
    "363713009": "has_interpretation", "246454002": "occurrence", "263502005": "clinical_course",
    "246112005": "severity", "255234002": "after", "47429007": "associated_with", "363701004": "direct_substance",
    "363700003": "direct_morphology", "424226004": "using_device", "363699004": "direct_device",
    "260507000": "access", "363703001": "has_intent", "272741003": "laterality", "123005000": "part_of",
    "733928003": "all_or_part_of", "411116001": "has_manufactured_dose_form", "246093002": "component", "370130000": "property",
    "370132008": "scale_type", "704319004": "inheres_in", "704321009": "characterizes", "370134009": "time_aspect",
    "704327008": "direct_site", "726542003": "has_disposition", "766939001": "plays_role",
    "738774007": "is_modification_of", "116686009": "has_specimen", "418775008": "finding_method",
    "419066007": "finding_informer", "408729009": "finding_context", "246090004": "associated_finding",
    "408731000": "temporal_context", "408732007": "subject_relationship_context", "363589002": "associated_procedure",
    "408730004": "procedure_context", "424876005": "surgical_approach", "363702006": "has_focus",
    "424361007": "using_substance", "405816004": "procedure_morphology", "370129005": "measurement_method",
    "1142135004": "has_basis_of_strength_substance", "763032000": "has_unit_of_presentation",
}
_TAG = re.compile(r"\s*\(([^()]*)\)\s*$")
_LETTERS = re.compile(r"[A-Za-z]")


def split_semantic_tag(fsn: str) -> tuple[str, str | None]:
    """("Term (tag)") → ("Term", "tag"); no trailing parenthesis → (fsn, None)."""
    match = _TAG.search(fsn)
    if not match:
        return fsn.strip(), None
    return fsn[:match.start()].strip(), match.group(1).strip()


def _rows(path: Path) -> Iterator[list[str]]:
    """Tab-separated RF2 rows after the header (RF2 is unquoted: `csv` would misread quote characters)."""
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            yield line.rstrip("\n").rstrip("\r").split("\t")


@dataclass(frozen=True)
class SnomedFiles:
    """The RF2 snapshot files of one release directory (`Snapshot/...`)."""

    concept: Path
    description: Path
    relationship: Path
    language: Path
    text_definition: Path | None = None

    @classmethod
    def find(cls, release: Path) -> "SnomedFiles":
        snapshot = Path(release).expanduser() / "Snapshot"

        def one(pattern: str, required: bool = True) -> Path | None:
            found = sorted(snapshot.glob(pattern))
            if not found:
                if required:
                    raise FileNotFoundError(f"no {pattern} under {snapshot}")
                return None
            return found[0]
        return cls(one("Terminology/sct2_Concept_Snapshot_*.txt"), one("Terminology/sct2_Description_Snapshot-en_*.txt"),
                   one("Terminology/sct2_Relationship_Snapshot_*.txt"),
                   one("Refset/Language/der2_cRefset_LanguageSnapshot-en_*.txt"),
                   one("Terminology/sct2_TextDefinition_Snapshot-en_*.txt", required=False))


@dataclass
class SnomedRelease:
    """The parts of a release the adapter needs (active content only)."""

    active: set[str]
    parents: dict[str, list[str]]                       # inferred is-a
    attributes: dict[str, list[tuple[str, str]]]        # source → [(type, destination)] (inferred, groups flattened)
    descriptions: dict[str, list[tuple[str, str, str]]]  # concept → [(type, term, case significance)] (accepted, English)
    definitions: dict[str, int] = field(default_factory=dict)   # concept → number of active text definitions


def read_release(files: SnomedFiles, *, languages: tuple[str, ...] = (US_ENGLISH, GB_ENGLISH)) -> SnomedRelease:
    active = {r[0] for r in _rows(files.concept) if r[2] == "1"}
    accepted: set[str] = set()
    for r in _rows(files.language):
        if r[2] == "1" and r[4] in languages and r[6] in (PREFERRED, ACCEPTABLE):
            accepted.add(r[5])
    descriptions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for r in _rows(files.description):
        if r[2] == "1" and r[5] == "en" and r[6] in (FSN, SYNONYM) and r[4] in active and r[0] in accepted:
            descriptions[r[4]].append((r[6], r[7], r[8]))
    parents: dict[str, list[str]] = defaultdict(list)
    attributes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in _rows(files.relationship):
        if r[2] != "1" or r[8] != INFERRED or r[4] not in active or r[5] not in active:
            continue
        if r[7] == IS_A:
            parents[r[4]].append(r[5])
        else:
            attributes[r[4]].append((r[7], r[5]))
    definitions: Counter[str] = Counter()
    if files.text_definition is not None:
        for r in _rows(files.text_definition):
            if r[2] == "1" and r[4] in active:
                definitions[r[4]] += 1
    return SnomedRelease(active, {k: sorted(set(v)) for k, v in parents.items()},
                         {k: sorted(set(v)) for k, v in attributes.items()}, dict(descriptions), dict(definitions))


def top_hierarchies(release: SnomedRelease) -> dict[str, set[str]]:
    """concept → the labels of the top-level hierarchies it descends from (a concept may sit in two)."""
    children: dict[str, list[str]] = defaultdict(list)
    for child, parents in release.parents.items():
        for parent in parents:
            children[parent].append(child)
    out: dict[str, set[str]] = defaultdict(set)
    for top, label in HIERARCHIES.items():
        if top not in release.active:
            continue
        queue = deque([top])
        seen = {top}
        while queue:
            node = queue.popleft()
            out[node].add(label)
            for child in children.get(node, ()):
                if child not in seen:
                    seen.add(child); queue.append(child)
    return out


# -- aliases ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SnomedAliasPolicy:
    """Which descriptions become linker aliases (T1c). Defaults are the T1c policy (decision 19's lesson)."""

    function_words: frozenset[str] = field(default=FUNCTION_WORDS)
    drop_abbreviations: bool = True
    max_abbreviation_chars: int = 8
    min_chars: int = 3
    max_words: int = 8

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SnomedAliasPolicy":
        config = dict(config or {})
        return cls(drop_abbreviations=bool(config.get("drop_abbreviations", True)),
                   max_abbreviation_chars=int(config.get("max_abbreviation_chars", 8)),
                   min_chars=int(config.get("min_chars", 3)), max_words=int(config.get("max_words", 8)))

    def as_dict(self) -> dict[str, Any]:
        return {"drop_abbreviations": self.drop_abbreviations, "max_abbreviation_chars": self.max_abbreviation_chars,
                "min_chars": self.min_chars, "max_words": self.max_words, "function_words": len(self.function_words)}


def is_abbreviation(term: str, max_chars: int = 8) -> bool:
    """A bare all-caps abbreviation: has a letter, no lowercase letter, no space, at most `max_chars` characters."""
    stripped = term.strip()
    return (bool(_LETTERS.search(stripped)) and not any(c.islower() for c in stripped) and " " not in stripped
            and len(stripped) <= max_chars)


def alias_decision(term: str, policy: SnomedAliasPolicy) -> str:
    """'keep' or the reason a description is not an alias."""
    key = normalize_alias(term)
    if not key:
        return "dropped_empty"
    if key in policy.function_words:
        return "dropped_function_word"
    if policy.drop_abbreviations and is_abbreviation(term, policy.max_abbreviation_chars):
        return "dropped_abbreviation"
    if len(key) < policy.min_chars:
        return "dropped_short"
    if len(key.split()) > policy.max_words:
        return "dropped_long"
    return "keep"


def concept_aliases(descriptions: list[tuple[str, str, str]], policy: SnomedAliasPolicy,
                    stats: Counter | None = None) -> tuple[list[str], str | None, str | None]:
    """(kept alias strings, preferred name, semantic tag) of one concept: the FSN without its tag, then synonyms."""
    stats = stats if stats is not None else Counter()
    kept: dict[str, str] = {}
    tag = preferred = None
    for kind, term, _ in sorted(descriptions, key=lambda d: (d[0] != FSN, d[1])):
        if kind == FSN:
            term, tag = split_semantic_tag(term)
            preferred = preferred or term
        stats["descriptions"] += 1
        decision = alias_decision(term, policy)
        stats[decision] += 1
        if decision == "keep":
            kept.setdefault(normalize_alias(term), term)
    return sorted(kept.values()), preferred, tag


# -- frames ----------------------------------------------------------------------------------------------------

def relation_name(type_id: str) -> str:
    return ATTRIBUTE_NAMES.get(type_id, f"attr:{type_id}")


def build_snomed_ontology(release_dir: Path, *, hierarchies: tuple[str, ...] = DEFAULT_HIERARCHIES,
                          max_atomics: int = 8192, max_degree: int = 16, min_relation_edges: int = 200,
                          policy: SnomedAliasPolicy = SnomedAliasPolicy(), release: SnomedRelease | None = None
                          ) -> FrameOntology:
    """The T1c frame ontology (concept names = SCTIDs, sorted). Metadata: preferred names (`headings`, the FSN without
    its tag), semantic tags, hierarchy labels, parents, alias-policy counts and per-relation edge counts."""
    unknown = sorted(set(hierarchies) - set(HIERARCHIES.values()))
    if unknown:
        raise ValueError(f"unknown hierarchies {unknown}; known: {sorted(HIERARCHIES.values())}")
    files = SnomedFiles.find(release_dir)
    release = release or read_release(files)
    tops = top_hierarchies(release)
    wanted = set(hierarchies)
    concepts = sorted((c for c in release.active if tops.get(c, set()) & wanted and c not in HIERARCHIES), key=int)
    selected = set(concepts)

    stats: Counter[str] = Counter()
    aliases, headings, tags = [], [], []
    for concept in concepts:
        kept, preferred, tag = concept_aliases(release.descriptions.get(concept, []), policy, stats)
        aliases.append(kept); headings.append(preferred or concept); tags.append(tag or "none")
        if not kept:
            stats["concepts_without_alias"] += 1

    # Relation types: is-a and every attribute type frequent enough among the selected sources.
    type_counts: Counter[str] = Counter(t for c in concepts for t, _ in release.attributes.get(c, ()))
    attribute_types = [t for t, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], int(kv[0]))) if n >= min_relation_edges]
    relation_names = ["hierarchy", "semantic_tag", "is_a", *(relation_name(t) for t in attribute_types)]
    rel = {name: i for i, name in enumerate(relation_names)}
    type_relation = {t: rel[relation_name(t)] for t in attribute_types}

    # Atomic dictionary: hierarchy and tag atoms, then the most-used filler concepts.
    use: Counter[str] = Counter()
    for concept in concepts:
        use.update(release.parents.get(concept, ()))
        use.update(d for t, d in release.attributes.get(concept, ()) if t in type_relation)
    category_atoms = [f"top:{label}" for label in hierarchies] + [f"tag:{t}" for t in sorted(set(tags))]
    budget = max(0, max_atomics - len(category_atoms))
    fillers = [c for c, _ in sorted(use.items(), key=lambda kv: (-kv[1], int(kv[0])))[:budget]]
    atomic_names = category_atoms + [f"sct:{c}" for c in fillers]
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    cache: dict[str, int | None] = {}

    def in_dictionary(concept: str) -> int | None:
        """The concept's atom, else its nearest in-dictionary ancestor (BFS over inferred parents)."""
        if concept in cache:
            return cache[concept]
        found, queue, seen = None, deque([concept]), {concept}
        while queue and found is None:
            node = queue.popleft()
            found = atomic_index.get(f"sct:{node}")
            if found is None:
                for parent in release.parents.get(node, ()):
                    if parent not in seen:
                        seen.add(parent); queue.append(parent)
        cache[concept] = found
        return found

    frames: list[list[tuple[int, int]]] = []
    relation_edges: Counter[str] = Counter()
    truncated = 0
    for concept, tag in zip(concepts, tags):
        frame: list[tuple[int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        candidates: list[tuple[int, int | None]] = []
        for label in sorted(tops[concept] & wanted):
            candidates.append((rel["hierarchy"], atomic_index.get(f"top:{label}")))
        candidates.append((rel["semantic_tag"], atomic_index.get(f"tag:{tag}")))
        candidates += [(rel["is_a"], in_dictionary(p)) for p in release.parents.get(concept, ())]
        attributes = sorted(((type_relation[t], d) for t, d in release.attributes.get(concept, ()) if t in type_relation),
                            key=lambda e: (e[0], int(e[1])))
        candidates += [(r, in_dictionary(d)) for r, d in attributes]
        for r, atom in candidates:
            if atom is None or (r, atom) in seen_edges:
                continue
            if len(frame) >= max_degree:
                truncated += 1
                break
            seen_edges.add((r, atom)); frame.append((r, atom)); relation_edges[relation_names[r]] += 1
        frames.append(frame)
    alias_pairs = [(alias, i) for i, kept in enumerate(aliases) for alias in kept]
    ontology = FrameOntology("snomed", concepts, relation_names, atomic_names, frames, alias_pairs,
                             {"source": str(Path(release_dir).expanduser()), "concepts": len(concepts)})
    hierarchy_counts = Counter(label for c in concepts for label in tops[c] & wanted)
    ontology.metadata.update({
        "headings": headings, "semantic_tags": tags, "hierarchies": list(hierarchies),
        "hierarchy_counts": dict(sorted(hierarchy_counts.items())),
        "concept_hierarchies": [sorted(tops[c] & wanted) for c in concepts],
        "parents": [release.parents.get(c, []) for c in concepts],
        "alias_policy": policy.as_dict(), "alias_stats": dict(sorted(stats.items())),
        "relation_edges": dict(sorted(relation_edges.items())), "attribute_types": attribute_types,
        "frames_truncated": truncated, "max_atomics": max_atomics, "max_degree": max_degree,
        "min_relation_edges": min_relation_edges, "definitions": [release.definitions.get(c, 0) for c in concepts],
        "selected": len(selected),
    })
    return ontology
