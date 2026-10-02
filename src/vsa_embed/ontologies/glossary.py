"""A generated private glossary (track T5, `benchmarks/glossary.py`) as a frame ontology.

Concept = glossary term (training and held-out terms; zero-shot terms are added by the track builder
as synthetic concepts). Frame edges are the term's typed relations (`term_frame`). Fillers come from
a bounded atomic dictionary: the fixed attribute atoms (types, areas, purposes, statuses, cadences,
tiers) plus the terms most used as fillers (by any term, zero-shot terms included); a term filler outside the dictionary is replaced by its
type atom (the coarser filler, as WordNet/MeSH fall back to an ancestor). Aliases are the term name
and its acronym, if any.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..benchmarks.glossary import AREAS, CADENCES, RELATIONS, STATUSES, TIERS, TYPES, term_frame
from .wordnet import FrameOntology


def fixed_atoms() -> list[str]:
    return ([f"type:{t}" for t in TYPES] + [f"area:{a}" for a in AREAS]
            + [f"purpose:{p}" for purposes in AREAS.values() for p in purposes]
            + [f"status:{s}" for s in STATUSES] + [f"cadence:{c}" for c in CADENCES] + [f"tier:{t}" for t in TIERS])


def build_glossary_ontology(glossary: dict[str, Any], *, max_atomics: int = 8192, max_degree: int = 16) -> FrameOntology:
    terms = [t for t in glossary["terms"] if t["split"] != "zeroshot"]
    types = {t["name"]: t["type"] for t in glossary["terms"]}
    frames_by_name = [term_frame(t) for t in terms]
    # Fillers of zero-shot terms count too, so their frames (added later as synthetic concepts) resolve.
    use: Counter[str] = Counter(atom for t in glossary["terms"] for _, atom in term_frame(t) if atom.startswith("term:"))
    fixed = fixed_atoms()
    budget = max(0, max_atomics - len(fixed))
    term_atoms = [atom for atom, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = fixed + term_atoms
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_index = {name: i for i, name in enumerate(RELATIONS)}
    frames, replaced = [], 0
    for frame in frames_by_name:
        edges: list[tuple[int, int]] = []
        for relation, atom in frame:
            if atom not in atomic_index:
                atom = f"type:{types[atom.split(':', 1)[1]]}"; replaced += 1
            edge = (relation_index[relation], atomic_index[atom])
            if edge not in edges and len(edges) < max_degree:
                edges.append(edge)
        frames.append(edges)
    aliases = [(alias, i) for i, t in enumerate(terms) for alias in t["aliases"]]
    ontology = FrameOntology("glossary", [t["name"] for t in terms], list(RELATIONS), atomic_names, frames, aliases,
                             {"seed": glossary["seed"], "terms": len(terms), "max_atomics": max_atomics,
                              "max_degree": max_degree, "fillers_replaced_by_type": replaced})
    ontology.metadata["splits"] = [t["split"] for t in terms]
    return ontology
