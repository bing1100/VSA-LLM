import pytest

from vsa_embed.span_channel import AliasTable


def test_wordnet_ontology_frames_and_aliases() -> None:
    wordnet = pytest.importorskip("nltk.corpus").wordnet
    try:
        wordnet.ensure_loaded()
    except LookupError:
        pytest.skip("WordNet data not installed")
    from vsa_embed.ontologies.wordnet import build_wordnet_ontology
    onto = build_wordnet_ontology(wordnet, max_atomics=2048)
    assert len(onto.atomic_names) <= 2048
    assert len(onto.frames) == len(onto.concept_names)
    assert all(frame and len(frame) <= 16 for frame in onto.frames)
    assert all(0 <= a < len(onto.atomic_names) for frame in onto.frames for _, a in frame)
    dog = onto.concept_index["dog.n.01"]
    relations = {onto.relation_names[r] for r, _ in onto.frames[dog]}
    assert {"hypernym", "lexname", "pos"} <= relations
    table = AliasTable.from_pairs(onto.alias_pairs)
    assert dog in table.entry_concepts[table.alias_to_entry["dog"]]
    assert "new york" in table.alias_to_entry
