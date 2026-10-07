"""MeSH descriptors as a frame ontology (open biomedical ontology; second family for E2, T1-open track).

Concept = descriptor. Frame edges:
- `parent`: descriptors whose tree number is the immediate prefix of one of this descriptor's tree numbers;
- `branch`: the top-level tree category letter(s) (A anatomy, C diseases, D chemicals, …) and the
  second-level tree node (e.g. C04);
- `pharmacological_action`: listed pharmacological action descriptors;
- `see_also`: "see related" descriptors.
Fillers come from a bounded atomic dictionary (most-used filler descriptors plus branch atoms);
out-of-dictionary parents fall back to their nearest in-dictionary ancestor.

Two alias policies:
- `build_mesh_ontology` (E2): the preferred heading and every entry term of every concept, verbatim.
- `build_mesh_track_ontology` (T1-open linker): curated by `MeshAliasPolicy` (see `mesh_track_aliases`):
  inverted "Head, Modifier" forms whose natural word order is itself a listed term are dropped (the
  listed natural form already links), unresolved comma forms are kept as listed (no form is ever
  invented), bare abbreviations/acronyms (lexical tags ABB/ACR) are dropped because the linker is
  case-insensitive ("ALL" would link every "all"), as are publication-type descriptors and aliases
  that are English function words.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..span_channel import normalize_alias
from .wordnet import FrameOntology

# English function words never used as aliases (MeSH 2026 has "back", "who" (WHO), "will" (Volition)).
FUNCTION_WORDS = frozenset("""
a about above across after again against all almost alone along already also although always am among an and
another any anyone anything are around as at back be became because become been before being below between both
but by can cannot could did do does doing done down during each either else enough even ever every few for from
further get gets got had has have having he her here hers herself him himself his how however i if in into is it
its itself just least less like made make many may me might more most much must my myself neither never no nor
not now of off often on once one only or other others our ours out over own per perhaps rather same see seem
several she should since so some still such than that the their theirs them themselves then there these they
this those though through thus to too toward under until up upon us use used very was we well were what when
where whether which while who whom whose why will with within without would yet you your yours
""".split())


def _parse(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != "DescriptorRecord":
                continue
            ui = element.findtext("DescriptorUI")
            name = element.findtext("DescriptorName/String")
            trees = [t.text for t in element.findall("TreeNumberList/TreeNumber") if t.text]
            actions = [a.findtext("DescriptorReferredTo/DescriptorUI") for a in element.findall("PharmacologicalActionList/PharmacologicalAction")]
            related = [r.findtext("DescriptorReferredTo/DescriptorUI") for r in element.findall("SeeRelatedList/SeeRelatedDescriptor")]
            terms = {t.text for t in element.findall("ConceptList/Concept/TermList/Term/String") if t.text}
            note = element.findtext("ConceptList/Concept[@PreferredConceptYN='Y']/ScopeNote") or ""
            term_info = [{"string": t.findtext("String"), "permuted": t.get("IsPermutedTermYN") == "Y",
                          "lexical_tag": t.get("LexicalTag") or "NON",
                          "record_preferred": t.get("RecordPreferredTermYN") == "Y"}
                         for t in element.findall("ConceptList/Concept/TermList/Term") if t.findtext("String")]
            introduced = element.findtext("DateIntroduced/Year") or element.findtext("DateCreated/Year")
            records.append({"ui": ui, "name": name, "trees": trees, "actions": [a for a in actions if a],
                            "related": [r for r in related if r], "terms": sorted(terms), "note": " ".join(note.split()),
                            "term_info": term_info, "descriptor_class": element.get("DescriptorClass") or "",
                            "introduced": int(introduced) if introduced else None})
            element.clear()
    return records


def _parents(records: list[dict]) -> dict[str, list[str]]:
    tree_owner = {tree: r["ui"] for r in records for tree in r["trees"]}
    parents: dict[str, list[str]] = {}
    for r in records:
        ps = []
        for tree in r["trees"]:
            if "." in tree:
                owner = tree_owner.get(tree.rsplit(".", 1)[0])
                if owner and owner not in ps:
                    ps.append(owner)
        parents[r["ui"]] = ps
    return parents


def _frame_ontology(records: list[dict], path: Path, *, max_atomics: int, max_degree: int) -> FrameOntology:
    """Frames over a bounded atomic dictionary; aliases = heading + every entry term (E2 policy)."""
    parents = _parents(records)
    use: Counter[str] = Counter()
    for r in records:
        use.update(parents[r["ui"]]); use.update(r["actions"]); use.update(r["related"])
    branches = sorted({t.split(".")[0] for r in records for t in r["trees"]} | {t[0] for r in records for t in r["trees"]})
    budget = max(0, max_atomics - len(branches))
    filler_uis = [ui for ui, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = [f"mesh:{ui}" for ui in filler_uis] + [f"branch:{b}" for b in branches]
    atomic_index = {n: i for i, n in enumerate(atomic_names)}
    relation_names = ["parent", "pharmacological_action", "see_also", "branch_top", "branch_second"]
    rel = {n: i for i, n in enumerate(relation_names)}
    cache: dict[str, int | None] = {}

    def in_dictionary(ui: str, depth: int = 0) -> int | None:
        if f"mesh:{ui}" in atomic_index:
            return atomic_index[f"mesh:{ui}"]
        if ui in cache:
            return cache[ui]
        result = None
        if depth < 12:
            for parent in parents.get(ui, []):
                result = in_dictionary(parent, depth + 1)
                if result is not None:
                    break
        cache[ui] = result
        return result

    frames, names, aliases = [], [], []
    for i, r in enumerate(records):
        frame: list[tuple[int, int]] = []
        seen = set()
        def push(relation: str, atom: int | None) -> None:
            if atom is not None and (rel[relation], atom) not in seen and len(frame) < max_degree:
                seen.add((rel[relation], atom)); frame.append((rel[relation], atom))
        for tree in r["trees"]:
            push("branch_top", atomic_index.get(f"branch:{tree[0]}"))
            push("branch_second", atomic_index.get(f"branch:{tree.split('.')[0]}"))
        for parent in parents[r["ui"]]:
            push("parent", in_dictionary(parent))
        for action in r["actions"]:
            push("pharmacological_action", in_dictionary(action))
        for other in r["related"]:
            push("see_also", in_dictionary(other))
        frames.append(frame); names.append(r["ui"])
        aliases += [(term, i) for term in {r["name"], *r["terms"]}]
    ontology = FrameOntology("mesh", names, relation_names, atomic_names, frames, aliases,
                             {"source": str(path), "descriptors": len(records)})
    ontology.metadata["headings"] = [r["name"] for r in records]
    ontology.metadata["scope_notes"] = [r["note"] for r in records]
    return ontology


def build_mesh_ontology(path: Path, *, max_atomics: int = 8192, max_degree: int = 16) -> FrameOntology:
    records = sorted(_parse(path), key=lambda r: r["ui"])
    return _frame_ontology(records, path, max_atomics=max_atomics, max_degree=max_degree)


# -- T1-open linker aliases ---------------------------------------------------------------------

@dataclass(frozen=True)
class MeshAliasPolicy:
    """Which MeSH terms become linker aliases (T1-open). Defaults are the pre-registered T1 policy."""

    exclude_classes: tuple[str, ...] = ("2",)                 # DescriptorClass 2 = publication types
    drop_lexical_tags: tuple[str, ...] = ("ABB", "ACR")       # bare abbreviations / acronyms (not headings)
    drop_resolved_inverted: bool = True
    function_words: frozenset[str] = field(default=FUNCTION_WORDS)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "MeshAliasPolicy":
        config = dict(config or {})
        return cls(exclude_classes=tuple(str(c) for c in config.get("exclude_classes", cls.exclude_classes)),
                   drop_lexical_tags=tuple(config.get("drop_lexical_tags", cls.drop_lexical_tags)),
                   drop_resolved_inverted=bool(config.get("drop_resolved_inverted", True)))


def natural_order_candidates(term: str) -> set[str]:
    """Natural-word-order readings of an inverted "Head, Modifier[, Modifier]" term, normalized:
    the reversed parts ("Leukemia, Lymphoblastic, Acute" → "acute lymphoblastic leukemia") and every
    rotation ("Receptors, Opioid, mu" → "opioid mu receptors", …). Empty if the term has no ", "."""
    parts = [p.strip() for p in term.split(", ")]
    if len(parts) < 2 or not all(parts):
        return set()
    readings = {" ".join(reversed(parts))}
    readings |= {" ".join(parts[k:] + parts[:k]) for k in range(1, len(parts))}
    return {normalize_alias(r) for r in readings}


def mesh_track_aliases(records: list[dict], policy: MeshAliasPolicy = MeshAliasPolicy()) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    """(alias, descriptor index) pairs under `policy`, plus counts of every decision.

    A comma term is *resolved inverted* when one of its natural-order readings is itself a listed
    term (any concept, permuted or not) of the same descriptor; it is then redundant and dropped.
    Otherwise it is kept as listed ("Hand, Foot, Mouth Disease"). MeSH's own permuted terms
    (`IsPermutedTermYN="Y"`) are listed terms, so natural-order permutations ("Acute Lymphoblastic
    Leukemia") are kept; permuted comma forms follow the same inversion rule.
    """
    stats: Counter[str] = Counter()
    pairs: list[tuple[str, int]] = []
    for index, record in enumerate(records):
        terms = record["term_info"] or [{"string": record["name"], "permuted": False, "lexical_tag": "NON",
                                         "record_preferred": True}]
        stats["terms"] += len(terms)
        if record["descriptor_class"] in policy.exclude_classes:
            stats["dropped_excluded_class"] += len(terms); stats["descriptors_excluded_class"] += 1
            continue
        listed = {normalize_alias(t["string"]) for t in terms}
        kept: set[str] = set()
        for term in terms:
            text, key = term["string"], normalize_alias(term["string"])
            if key in policy.function_words:
                stats["dropped_function_word"] += 1; continue
            if term["lexical_tag"] in policy.drop_lexical_tags and not term["record_preferred"]:
                stats["dropped_abbreviation"] += 1; continue
            readings = natural_order_candidates(text)
            if readings:
                if policy.drop_resolved_inverted and (readings - {key}) & listed:
                    stats["dropped_resolved_inverted"] += 1; continue
                stats["kept_unresolved_comma"] += 1
            if term["permuted"]:
                stats["kept_permuted"] += 1
            kept.add(text)
        if not kept:
            stats["descriptors_without_alias"] += 1
        stats["kept_terms"] += len(kept)
        pairs += [(text, index) for text in sorted(kept)]
    return pairs, dict(stats)


def build_mesh_track_ontology(path: Path, *, max_atomics: int = 8192, max_degree: int = 16,
                              policy: MeshAliasPolicy = MeshAliasPolicy()) -> FrameOntology:
    """MeSH frames (as `build_mesh_ontology`) with the curated T1-open alias table.

    Metadata adds per-descriptor tree numbers, parent UIs, descriptor class and the alias-policy counts,
    which the track tasks (tree-category probe, neighbour items) read.
    """
    records = sorted(_parse(path), key=lambda r: r["ui"])
    ontology = _frame_ontology(records, path, max_atomics=max_atomics, max_degree=max_degree)
    pairs, stats = mesh_track_aliases(records, policy)
    ontology.alias_pairs = pairs
    parents = _parents(records)
    ontology.metadata.update({
        "alias_policy": {"exclude_classes": list(policy.exclude_classes), "drop_lexical_tags": list(policy.drop_lexical_tags),
                         "drop_resolved_inverted": policy.drop_resolved_inverted},
        "alias_stats": stats, "trees": [r["trees"] for r in records], "parents": [parents[r["ui"]] for r in records],
        "descriptor_class": [r["descriptor_class"] for r in records],
        "related": [r["related"] for r in records], "actions": [r["actions"] for r in records],
        "max_atomics": max_atomics, "max_degree": max_degree,
    })
    return ontology
