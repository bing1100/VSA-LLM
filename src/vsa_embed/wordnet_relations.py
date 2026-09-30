"""Heterogeneous WordNet graph and frozen-host anchor utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        return edge_group(self, LEGACY_SYMMETRIC_RELATIONS)


# `attribute` is also symmetric in WordNet (noun ↔ adjective); the legacy grouping kept only
# `antonym`, so reverse `attribute` pairs could straddle an edge-disjoint split.
LEGACY_SYMMETRIC_RELATIONS: tuple[str, ...] = ("antonym",)
SYMMETRIC_RELATIONS: tuple[str, ...] = ("antonym", "attribute")


def edge_group(edge: RelationEdge, symmetric_relations: Iterable[str]) -> tuple[str, str, str]:
    """Group key under which reverse duplicates of symmetric relations coincide."""
    if edge.relation in set(symmetric_relations):
        left, right = sorted((edge.source, edge.target))
        return left, edge.relation, right
    return edge.source, edge.relation, edge.target


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
    seed: int, symmetric_relations: Iterable[str] = LEGACY_SYMMETRIC_RELATIONS,
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
                group = edge_group(edge, symmetric_relations)
                if target in aligned and target != source and group not in seen:
                    by_relation[relation].append(edge); seen.add(group)
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


@dataclass
class EdgeSplit:
    """Train / validation / test / discarded edge indices plus realized fractions."""

    train: list[int]
    validation: list[int]
    test: list[int]
    discarded: list[int]
    stats: dict[str, float | int] = field(default_factory=dict)

    def partitions(self) -> dict[str, list[int]]:
        return {"train": self.train, "validation": self.validation, "test": self.test,
                "discarded": self.discarded}


def _split_stats(total: int, split: EdgeSplit) -> dict[str, float | int]:
    stats: dict[str, float | int] = {"edges": total}
    for name, indices in split.partitions().items():
        stats[f"{name}_edges"] = len(indices)
        stats[f"{name}_fraction"] = len(indices) / total if total else 0.0
    return stats


def _choose(keys: list[Any], fraction: float, generator: torch.Generator) -> set[Any]:
    if fraction <= 0 or not keys:
        return set()
    order = torch.randperm(len(keys), generator=generator).tolist()
    return {keys[i] for i in order[:max(1, round(len(keys) * fraction))]}


def split_edge_partitions(
    edges: list[RelationEdge], *, mode: str, test_fraction: float, seed: int,
    validation_fraction: float = 0.0,
    symmetric_relations: Iterable[str] = LEGACY_SYMMETRIC_RELATIONS,
    node_groups: dict[str, Any] | None = None,
) -> EdgeSplit:
    """Split edges into train / validation / test for model selection without test reuse.

    The test partition is chosen exactly as `split_edges` chooses it (same generator stream),
    so with the legacy grouping and no validation the result equals `split_edges`. Validation
    is carved from the remaining data with its own generator and is disjoint from train in the
    same sense as test: grouped edges for `edge_disjoint`; nodes for `node_disjoint`; host
    tokens for `token_disjoint` (nodes that share a token move together, `node_groups` maps
    node → token). Edges crossing partitions in the node/token modes are discarded; realized
    fractions are recorded in `stats`.
    """
    generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1_000_003)
    if mode == "edge_disjoint":
        groups: dict[tuple[str, str, str], list[int]] = {}
        for index, edge in enumerate(edges):
            groups.setdefault(edge_group(edge, symmetric_relations), []).append(index)
        keys = list(groups); order = torch.randperm(len(keys), generator=generator).tolist()
        test_keys = {keys[i] for i in order[:max(1, round(len(keys) * test_fraction))]}
        rest = [key for key in keys if key not in test_keys]
        validation_keys = _choose(rest, validation_fraction / max(1e-12, 1 - test_fraction), validation_generator)
        split = EdgeSplit(
            train=sorted(i for key in rest if key not in validation_keys for i in groups[key]),
            validation=sorted(i for key in validation_keys for i in groups[key]),
            test=sorted(i for key in test_keys for i in groups[key]),
            discarded=[],
        )
    elif mode in {"node_disjoint", "token_disjoint"}:
        if mode == "token_disjoint" and node_groups is None:
            raise ValueError("token_disjoint requires node_groups (node → host token)")
        key_of = (lambda node: node) if mode == "node_disjoint" else (lambda node: node_groups[node])
        nodes = sorted({edge.source for edge in edges} | {edge.target for edge in edges})
        units = sorted({key_of(node) for node in nodes}, key=str)
        # sqrt fraction gives approximately the requested fraction of within-test edges.
        node_fraction = min(0.8, test_fraction**0.5)
        order = torch.randperm(len(units), generator=generator).tolist()
        test_units = {units[i] for i in order[:max(1, round(len(units) * node_fraction))]}
        rest = [unit for unit in units if unit not in test_units]
        validation_units = _choose(
            rest, min(0.8, validation_fraction**0.5) if validation_fraction > 0 else 0.0,
            validation_generator,
        )
        train, validation, test, discarded = [], [], [], []
        for index, edge in enumerate(edges):
            ends = {key_of(edge.source), key_of(edge.target)}
            if ends <= test_units: test.append(index)
            elif ends <= validation_units: validation.append(index)
            elif not ends & (test_units | validation_units): train.append(index)
            else: discarded.append(index)
        split = EdgeSplit(train, validation, test, discarded)
    else:
        raise ValueError("mode must be edge_disjoint, node_disjoint or token_disjoint")
    split.stats = _split_stats(len(edges), split)
    return split


def split_neighborhood_partitions(
    edges: list[RelationEdge], *, test_fraction: float, seed: int,
    validation_fraction: float = 0.0, target_disjoint: bool = True,
) -> EdgeSplit:
    """Hold out whole source neighbourhoods, optionally with their targets.

    Test sources are chosen exactly as `split_source_neighborhoods` chooses them. With
    `target_disjoint=True` every node of a held-out neighbourhood (its source and all its
    targets) is also removed from training: training and validation edges touching a test
    node are discarded, and training edges touching a validation node are discarded, so no
    test target is ever a training endpoint. This fixes the leakage in which held-out
    queries' parents were seen in training (audit F1). Discards are recorded in `stats`.
    """
    train_legacy, test = split_source_neighborhoods(edges, test_fraction=test_fraction, seed=seed)
    test_set = set(test)
    by_source: dict[str, list[int]] = {}
    for index in train_legacy:
        by_source.setdefault(edges[index].source, []).append(index)
    sources = sorted(by_source)
    validation_sources = _choose(
        sources, validation_fraction / max(1e-12, 1 - test_fraction),
        torch.Generator().manual_seed(seed + 1_000_003),
    )
    validation = sorted(i for source in validation_sources for i in by_source[source])
    train = [i for source in sources if source not in validation_sources for i in by_source[source]]
    discarded: list[int] = []
    if target_disjoint:
        test_nodes = {node for i in test_set for node in (edges[i].source, edges[i].target)}
        kept_validation = [i for i in validation
                           if not {edges[i].source, edges[i].target} & test_nodes]
        discarded += sorted(set(validation) - set(kept_validation))
        validation = kept_validation
        held_out = test_nodes | {node for i in validation for node in (edges[i].source, edges[i].target)}
        kept_train = [i for i in train if not {edges[i].source, edges[i].target} & held_out]
        discarded += sorted(set(train) - set(kept_train))
        train = kept_train
    split = EdgeSplit(sorted(train), sorted(validation), sorted(test), sorted(discarded))
    split.stats = _split_stats(len(edges), split)
    return split


def static_anchors(host: Any, nodes: list[SenseNode]) -> Tensor:
    token_ids = torch.tensor([node.token_id for node in nodes])
    return F.normalize(host.get_input_embeddings()(token_ids).detach().float(), dim=-1)


def _gloss_truncated_prompt(tokenizer: Any, node: SenseNode, max_length: int) -> str:
    """Shorten only the gloss so that the prompt, ending in the lemma, fits `max_length`."""
    prefix, suffix = "Definition: ", f" Term: {node.lemma}"
    budget = max_length - len(tokenizer.encode(prefix + suffix, add_special_tokens=False))
    if budget < 1:
        raise ValueError(f"max_length {max_length} leaves no room for the gloss of {node.synset}")
    gloss_ids = tokenizer.encode(node.gloss, add_special_tokens=False)
    gloss = node.gloss if len(gloss_ids) <= budget else tokenizer.decode(gloss_ids[:budget])
    return prefix + gloss + suffix


def contextual_anchors(
    host: Any, tokenizer: Any, nodes: list[SenseNode], *, batch_size: int, max_length: int,
    device: torch.device, truncation: str = "right",
) -> Tensor:
    """Encode `Definition: gloss Term: lemma`, reading the final lemma state.

    `truncation="right"` (historical) truncates the whole prompt from the right, which for
    over-long glosses reads the anchor inside the gloss. `truncation="gloss"` shortens only the
    gloss and asserts that the read position holds the lemma's token. Padding is forced to the
    right in both modes, because the read position assumes it.
    """
    if truncation not in {"right", "gloss"}:
        raise ValueError("truncation must be 'right' or 'gloss'")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    host = host.to(device).eval()
    outputs = []
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        if truncation == "gloss":
            prompts = [_gloss_truncated_prompt(tokenizer, node, max_length) for node in batch]
        else:
            prompts = [f"Definition: {node.gloss} Term: {node.lemma}" for node in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        positions = encoded["attention_mask"].sum(1) - 1
        if truncation == "gloss":
            read = encoded["input_ids"][torch.arange(len(batch), device=device), positions].tolist()
            expected = [node.token_id for node in batch]
            if read != expected:
                raise AssertionError(f"anchor position does not hold the lemma token: {read} != {expected}")
        with torch.no_grad(): hidden = host(**encoded, output_hidden_states=True).hidden_states[-1]
        outputs.append(hidden[torch.arange(len(batch), device=device), positions].float().cpu())
    return F.normalize(torch.cat(outputs), dim=-1)
