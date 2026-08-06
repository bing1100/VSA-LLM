"""Heterogeneous WordNet graph and frozen-host anchor utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


RELATION_ACCESSORS: dict[str, Callable[[Any], Iterable[Any]]] = {
    "hypernym": lambda synset: synset.hypernyms(),
    "instance_hypernym": lambda synset: synset.instance_hypernyms(),
    "member_meronym": lambda synset: synset.member_meronyms(),
    "substance_meronym": lambda synset: synset.substance_meronyms(),
    "part_meronym": lambda synset: synset.part_meronyms(),
    "attribute": lambda synset: synset.attributes(),
    "antonym": lambda synset: [item.synset() for lemma in synset.lemmas() for item in lemma.antonyms()],
}


def collect_source_neighborhood_graph(
    wordnet: Any, tokenizer: Any, relation_types: list[str], *,
    max_sources_per_relation: int, min_targets_per_source: int, seed: int,
) -> tuple[list[SenseNode], list[RelationEdge]]:
    """Sample source queries while retaining every aligned target in each neighborhood."""
    unknown = set(relation_types) - set(RELATION_ACCESSORS)
    if unknown:
        raise ValueError(f"unsupported relations: {sorted(unknown)}")
    aligned: dict[str, SenseNode] = {}
    synsets: dict[str, Any] = {}
    for synset in wordnet.all_synsets():
        match = aligned_lemma(synset, tokenizer)
        if match is not None:
            lemma, token_id = match
            aligned[synset.name()] = SenseNode(
                synset.name(), lemma, token_id, synset.definition(), synset.pos(),
            )
            synsets[synset.name()] = synset
    groups: dict[str, list[list[RelationEdge]]] = {relation: [] for relation in relation_types}
    for source in sorted(aligned):
        for relation in relation_types:
            targets = sorted({
                target.name() for target in RELATION_ACCESSORS[relation](synsets[source])
                if target.name() in aligned and target.name() != source
            })
            if len(targets) >= min_targets_per_source:
                groups[relation].append([RelationEdge(source, relation, target) for target in targets])
    generator = torch.Generator().manual_seed(seed)
    edges: list[RelationEdge] = []
    for relation in relation_types:
        neighborhoods = groups[relation]
        order = torch.randperm(len(neighborhoods), generator=generator).tolist()
        for index in order[:max_sources_per_relation]:
            edges.extend(neighborhoods[index])
    used = {edge.source for edge in edges} | {edge.target for edge in edges}
    return [aligned[name] for name in sorted(used)], edges


def split_source_neighborhoods(
    edges: list[RelationEdge], *, test_fraction: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Split source IDs atomically, stratified by their relation signature."""
    groups: dict[str, list[int]] = {}
    for index, edge in enumerate(edges):
        groups.setdefault(edge.source, []).append(index)
    by_signature: dict[tuple[str, ...], list[str]] = {}
    for source, indices in groups.items():
        signature = tuple(sorted({edges[index].relation for index in indices}))
        by_signature.setdefault(signature, []).append(source)
    generator = torch.Generator().manual_seed(seed)
    test_sources: set[str] = set()
    for sources in by_signature.values():
        sources = sorted(sources)
        order = torch.randperm(len(sources), generator=generator).tolist()
        count = max(1, round(len(sources) * test_fraction))
        test_sources.update(sources[index] for index in order[:count])
    test = [index for source in test_sources for index in groups[source]]
    train = [index for source in groups if source not in test_sources for index in groups[source]]
    return sorted(train), sorted(test)


@dataclass(frozen=True)
class SenseNode:
    synset: str
    lemma: str
    token_id: int
    gloss: str
    pos: str


@dataclass(frozen=True)
class RelationEdge:
    source: str
    relation: str
    target: str

    @property
    def group(self) -> tuple[str, str, str]:
        # Antonym edges are symmetric and reverse duplicates must never cross splits.
        if self.relation == "antonym":
            left, right = sorted((self.source, self.target))
            return left, self.relation, right
        return self.source, self.relation, self.target


def aligned_lemma(synset: Any, tokenizer: Any) -> tuple[str, int] | None:
    """Choose a deterministic alphabetic lemma represented by one host token."""
    for raw in sorted(synset.lemma_names(), key=lambda value: (len(value), value)):
        lemma = raw.lower()
        token_ids = tokenizer.encode(" " + lemma, add_special_tokens=False)
        if lemma.isalpha() and len(token_ids) == 1 and tokenizer.decode(token_ids).strip().lower() == lemma:
            return lemma, int(token_ids[0])
    return None


def collect_wordnet_graph(
    wordnet: Any, tokenizer: Any, relation_types: list[str], *, max_edges_per_relation: int,
    seed: int,
) -> tuple[list[SenseNode], list[RelationEdge]]:
    """Collect a deterministic, relation-balanced graph of token-aligned senses."""
    unknown = set(relation_types) - set(RELATION_ACCESSORS)
    if unknown:
        raise ValueError(f"unsupported relations: {sorted(unknown)}")
    aligned: dict[str, SenseNode] = {}
    synsets: dict[str, Any] = {}
    for synset in wordnet.all_synsets():
        match = aligned_lemma(synset, tokenizer)
        if match is not None:
            lemma, token_id = match
            aligned[synset.name()] = SenseNode(synset.name(), lemma, token_id, synset.definition(), synset.pos())
            synsets[synset.name()] = synset
    by_relation: dict[str, list[RelationEdge]] = {relation: [] for relation in relation_types}
    seen: set[tuple[str, str, str]] = set()
    for source in sorted(aligned):
        synset = synsets[source]
        for relation in relation_types:
            for target_synset in RELATION_ACCESSORS[relation](synset):
                target = target_synset.name()
                edge = RelationEdge(source, relation, target)
                if target in aligned and target != source and edge.group not in seen:
                    by_relation[relation].append(edge); seen.add(edge.group)
    generator = torch.Generator().manual_seed(seed)
    edges: list[RelationEdge] = []
    for relation in relation_types:
        candidates = by_relation[relation]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        edges.extend(candidates[index] for index in order[:max_edges_per_relation])
    used = {edge.source for edge in edges} | {edge.target for edge in edges}
    return [aligned[name] for name in sorted(used)], edges


def split_edges(
    edges: list[RelationEdge], *, mode: str, test_fraction: float, seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return train, test, and intentionally discarded crossing-edge indices."""
    generator = torch.Generator().manual_seed(seed)
    if mode == "edge_disjoint":
        groups: dict[tuple[str, str, str], list[int]] = {}
        for index, edge in enumerate(edges): groups.setdefault(edge.group, []).append(index)
        keys = list(groups); order = torch.randperm(len(keys), generator=generator).tolist()
        test_keys = {keys[i] for i in order[:max(1, round(len(keys) * test_fraction))]}
        test = [i for key in test_keys for i in groups[key]]
        train = [i for key in keys if key not in test_keys for i in groups[key]]
        return sorted(train), sorted(test), []
    if mode != "node_disjoint":
        raise ValueError("mode must be edge_disjoint or node_disjoint")
    nodes = sorted({edge.source for edge in edges} | {edge.target for edge in edges})
    # sqrt fraction gives approximately the requested fraction of within-test edges.
    node_fraction = min(0.8, test_fraction**0.5)
    order = torch.randperm(len(nodes), generator=generator).tolist()
    test_nodes = {nodes[i] for i in order[:max(1, round(len(nodes) * node_fraction))]}
    train, test, discarded = [], [], []
    for index, edge in enumerate(edges):
        source_test, target_test = edge.source in test_nodes, edge.target in test_nodes
        if source_test and target_test: test.append(index)
        elif not source_test and not target_test: train.append(index)
        else: discarded.append(index)
    return train, test, discarded


def static_anchors(host: Any, nodes: list[SenseNode]) -> Tensor:
    token_ids = torch.tensor([node.token_id for node in nodes])
    return F.normalize(host.get_input_embeddings()(token_ids).detach().float(), dim=-1)


def contextual_anchors(
    host: Any, tokenizer: Any, nodes: list[SenseNode], *, batch_size: int, max_length: int,
    device: torch.device,
) -> Tensor:
    """Encode `Definition: gloss. Term: lemma`, reading the final lemma state."""
    tokenizer.pad_token = tokenizer.eos_token
    host = host.to(device).eval()
    outputs = []
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        prompts = [f"Definition: {node.gloss} Term: {node.lemma}" for node in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(): hidden = host(**encoded, output_hidden_states=True).hidden_states[-1]
        positions = encoded["attention_mask"].sum(1) - 1
        outputs.append(hidden[torch.arange(len(batch), device=device), positions].float().cpu())
    return F.normalize(torch.cat(outputs), dim=-1)
