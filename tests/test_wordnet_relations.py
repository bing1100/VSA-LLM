import torch

from vsa_embed.wordnet_relations import RelationEdge, split_edges, split_source_neighborhoods


def test_antonym_reverse_edges_share_a_group() -> None:
    left = RelationEdge("hot.a.01", "antonym", "cold.a.01")
    right = RelationEdge("cold.a.01", "antonym", "hot.a.01")
    assert left.group == right.group
    train, test, _ = split_edges([left, right, RelationEdge("dog.n.01", "hypernym", "animal.n.01")],
                                 mode="edge_disjoint", test_fraction=0.5, seed=3)
    assert ({0, 1} <= set(train)) or ({0, 1} <= set(test))


def test_node_disjoint_split_has_no_endpoint_overlap() -> None:
    edges = [RelationEdge(f"n{i}", "hypernym", f"n{(i + 1) % 20}") for i in range(20)]
    train, test, discarded = split_edges(edges, mode="node_disjoint", test_fraction=0.25, seed=7)
    train_nodes = {node for i in train for node in (edges[i].source, edges[i].target)}
    test_nodes = {node for i in test for node in (edges[i].source, edges[i].target)}
    assert train_nodes.isdisjoint(test_nodes)
    assert sorted(train + test + discarded) == list(range(len(edges)))


def test_edge_split_is_complete_and_disjoint() -> None:
    edges = [RelationEdge(f"a{i}", "part_meronym", f"b{i}") for i in range(20)]
    train, test, discarded = split_edges(edges, mode="edge_disjoint", test_fraction=0.2, seed=4)
    assert not discarded and set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(20))


def test_source_neighborhood_split_never_fragments_a_target_set() -> None:
    edges = [
        RelationEdge(f"source{i}", relation, f"target{i}-{j}")
        for relation in ("hypernym", "instance_hypernym")
        for i in range(10) for j in range(2)
    ]
    train, test = split_source_neighborhoods(edges, test_fraction=0.2, seed=5)
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(len(edges)))
    train_sources = {edges[index].source for index in train}
    test_sources = {edges[index].source for index in test}
    assert train_sources.isdisjoint(test_sources)
    for source in {edge.source for edge in edges}:
        for relation in {edge.relation for edge in edges}:
            indices = {i for i, edge in enumerate(edges)
                       if edge.source == source and edge.relation == relation}
            assert not indices or indices <= set(train) or indices <= set(test)
