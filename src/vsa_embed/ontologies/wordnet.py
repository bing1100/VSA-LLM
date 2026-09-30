"""WordNet as a frame ontology for the span channel (general track T0).

Concepts are synsets. A concept's frame lists `(relation, filler)` edges:
typed WordNet pointers to other synsets plus its lexicographer file and part of speech.
Fillers come from a bounded atomic dictionary: the `max_atomics` synsets most used as fillers,
the 45 lexicographer files and the POS tags. A pointer to a synset outside the dictionary is
replaced by its nearest in-dictionary hypernym (so the edge keeps a meaningful, coarser
filler) or dropped if none exists. Every lemma name is an alias of its synset; polysemous
surface forms become union entries in the `AliasTable`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

POINTERS: dict[str, Callable[[Any], Iterable[Any]]] = {
    "hypernym": lambda s: s.hypernyms(),
    "instance_hypernym": lambda s: s.instance_hypernyms(),
    "part_meronym": lambda s: s.part_meronyms(),
    "member_meronym": lambda s: s.member_meronyms(),
    "substance_meronym": lambda s: s.substance_meronyms(),
    "part_holonym": lambda s: s.part_holonyms(),
    "member_holonym": lambda s: s.member_holonyms(),
    "substance_holonym": lambda s: s.substance_holonyms(),
    "attribute": lambda s: s.attributes(),
    "similar_to": lambda s: s.similar_tos(),
    "topic_domain": lambda s: s.topic_domains(),
    "entailment": lambda s: s.entailments(),
    "cause": lambda s: s.causes(),
    "antonym": lambda s: [a.synset() for lemma in s.lemmas() for a in lemma.antonyms()],
}


@dataclass
class FrameOntology:
    """Concepts with frames over a bounded atomic dictionary, plus alias pairs."""

    name: str
    concept_names: list[str]
    relation_names: list[str]
    atomic_names: list[str]
    frames: list[list[tuple[int, int]]]
    alias_pairs: list[tuple[str, int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def concept_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.concept_names)}


def build_wordnet_ontology(wordnet: Any, *, max_atomics: int = 8192, max_degree: int = 16,
                           pointers: Iterable[str] = tuple(POINTERS)) -> FrameOntology:
    """Build the WordNet frame ontology (deterministic for a given WordNet version)."""
    pointers = list(pointers)
    synsets = sorted(wordnet.all_synsets(), key=lambda s: s.name())
    names = [s.name() for s in synsets]
    use: Counter[str] = Counter()
    for synset in synsets:
        for pointer in pointers:
            use.update(t.name() for t in POINTERS[pointer](synset))
    lexnames = sorted({s.lexname() for s in synsets})
    poses = sorted({s.pos() for s in synsets})
    budget = max(0, max_atomics - len(lexnames) - len(poses))
    filler_synsets = [name for name, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = [f"synset:{n}" for n in filler_synsets] + [f"lexname:{n}" for n in lexnames] + [f"pos:{p}" for p in poses]
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_names = pointers + ["lexname", "pos"]
    relation_index = {name: i for i, name in enumerate(relation_names)}
    by_name = {s.name(): s for s in synsets}
    ancestor_cache: dict[str, int | None] = {}

    def in_dictionary(synset: Any, depth: int = 0) -> int | None:
        name = synset.name()
        if f"synset:{name}" in atomic_index:
            return atomic_index[f"synset:{name}"]
        if name in ancestor_cache:
            return ancestor_cache[name]
        result = None
        if depth < 12:
            for parent in sorted(synset.hypernyms() + synset.instance_hypernyms(), key=lambda s: s.name()):
                result = in_dictionary(parent, depth + 1)
                if result is not None:
                    break
        ancestor_cache[name] = result
        return result

    frames: list[list[tuple[int, int]]] = []
    replaced = dropped = 0
    for synset in synsets:
        frame: list[tuple[int, int]] = [
            (relation_index["lexname"], atomic_index[f"lexname:{synset.lexname()}"]),
            (relation_index["pos"], atomic_index[f"pos:{synset.pos()}"]),
        ]
        seen = set(frame)
        for pointer in pointers:
            for target in sorted(POINTERS[pointer](synset), key=lambda s: s.name()):
                filler = in_dictionary(target)
                if filler is None:
                    dropped += 1; continue
                if atomic_names[filler] != f"synset:{target.name()}":
                    replaced += 1
                edge = (relation_index[pointer], filler)
                if edge not in seen and len(frame) < max_degree:
                    seen.add(edge); frame.append(edge)
        frames.append(frame)
    alias_pairs = [(lemma.replace("_", " "), i) for i, synset in enumerate(synsets) for lemma in synset.lemma_names()]
    return FrameOntology(
        "wordnet", names, relation_names, atomic_names, frames, alias_pairs,
        {"wordnet_version": wordnet.get_version(), "max_atomics": max_atomics, "max_degree": max_degree,
         "fillers_replaced_by_ancestor": replaced, "fillers_dropped": dropped},
    )
