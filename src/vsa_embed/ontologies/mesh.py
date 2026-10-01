"""MeSH descriptors as a frame ontology (open biomedical ontology; second family for E2, T1 fallback).

Concept = descriptor. Frame edges:
- `parent`: descriptors whose tree number is the immediate prefix of one of this descriptor's tree numbers;
- `branch`: the top-level tree category letter(s) (A anatomy, C diseases, D chemicals, …) and the
  second-level tree node (e.g. C04);
- `pharmacological_action`: listed pharmacological action descriptors;
- `see_also`: "see related" descriptors.
Fillers come from a bounded atomic dictionary (most-used filler descriptors plus branch atoms);
out-of-dictionary parents fall back to their nearest in-dictionary ancestor. Aliases are the
preferred heading and every entry term of every concept.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .wordnet import FrameOntology


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
            records.append({"ui": ui, "name": name, "trees": trees, "actions": [a for a in actions if a],
                            "related": [r for r in related if r], "terms": sorted(terms), "note": " ".join(note.split())})
            element.clear()
    return records


def build_mesh_ontology(path: Path, *, max_atomics: int = 8192, max_degree: int = 16) -> FrameOntology:
    records = sorted(_parse(path), key=lambda r: r["ui"])
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
