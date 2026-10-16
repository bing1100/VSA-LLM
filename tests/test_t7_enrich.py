"""t7_enrich: richer T7-ROOD ontologies (2026-10-15 addendum 2, richness stages 2–3)."""

from pathlib import Path

import pytest
import torch

from vsa_embed.experiments import t7_enrich


def _ontology() -> dict:
    # two SCR entries: C1 maps to D2 (whose parent is D1, action D9, related D3, tree A01.100.200); C2 maps to an unknown heading
    relation_names = ["mapped_to", "parent", "pharmacological_action", "see_also", "record_class", "branch_top", "branch_second"]
    atomic_names = ["mesh:D2", "class:chemical", "branch:A", "branch:A01"]
    frames = [[(0, 0), (4, 1), (5, 2), (6, 3)], [(4, 1)]]
    offsets, rel, fil = [0], [], []
    for f in frames:
        for r, a in f:
            rel.append(r); fil.append(a)
        offsets.append(len(rel))
    return {"entry_count": 2, "concept_names": ["C1", "C2"], "entry_concepts": [(0,), (1,)], "relation_names": relation_names,
            "relation_count": len(relation_names), "atomic_names": atomic_names, "atomic_count": len(atomic_names),
            "offsets": torch.tensor(offsets), "relations": torch.tensor(rel), "fillers": torch.tensor(fil), "heldout_entries": [1],
            "alias_table_sha256": "x", "train_frequency": [5, 0]}


DESCRIPTORS = [
    {"ui": "D1", "name": "Root", "trees": ["A01.100"], "actions": [], "related": []},
    {"ui": "D2", "name": "Thing", "trees": ["A01.100.200"], "actions": ["D9"], "related": ["D3"]},
    {"ui": "D3", "name": "Other", "trees": ["B01"], "actions": [], "related": []},
    {"ui": "D9", "name": "Action", "trees": ["D27.505"], "actions": [], "related": []},
]
SCRS = [{"ui": "C1", "name": "newdrug", "mapped": ["D2"], "actions": [], "scr_class": 1},
        {"ui": "C2", "name": "orphan", "mapped": ["D404"], "actions": [], "scr_class": 1}]


def test_mesh_enrichment_appends_inherited_heading_facts() -> None:
    base = _ontology()
    rich = t7_enrich.mesh_enrichment(base, DESCRIPTORS, SCRS)
    names = rich["relation_names"]
    assert names[:7] == base["relation_names"] and names[7:] == list(t7_enrich.MESH_RELATIONS)
    assert rich["atomic_names"][:4] == base["atomic_names"]                         # indices kept, new atomics appended
    frames = t7_enrich.base_frames(rich)
    assert frames[0][:4] == t7_enrich.base_frames(base)[0]                          # base edges first, unchanged
    added = {(names[r], rich["atomic_names"][a]) for r, a in frames[0][4:]}
    assert added == {("heading_parent", "mesh:D1"), ("heading_action", "mesh:D9"), ("heading_see_also", "mesh:D3"),
                     ("branch_third", "branch:A01.100")}
    assert frames[1] == t7_enrich.base_frames(base)[1]                              # an unknown heading adds nothing
    assert rich["enrichment"]["entries_enriched"] == 1 and rich["enrichment"]["edges_added"] == 4
    assert rich["heldout_entries"] == [1] and rich["alias_table_sha256"] == "x"     # entries, holdout and links unchanged


def test_merge_cap_and_index_guard(tmp_path) -> None:
    assert t7_enrich.merged([(0, 1)], [(0, 1), (1, 2), (2, 3)], max_degree=2) == [(0, 1), (1, 2)]
    base = _ontology()
    with pytest.raises(ValueError, match="append only"):
        t7_enrich.with_frames(base, t7_enrich.base_frames(base), relation_names=base["relation_names"][::-1],
                              atomic_names=base["atomic_names"], enrichment={})
    root = tmp_path / "base"
    (root / "train").mkdir(parents=True); (root / "eval").mkdir(); torch.save(base, root / "ontology.pt")
    summary = t7_enrich.write_variant(t7_enrich.mesh_enrichment(base, DESCRIPTORS, SCRS), tmp_path / "rich", root)
    assert sorted(summary["linked"]) == ["eval", "train"] and (tmp_path / "rich" / "train").is_symlink()
    loaded = torch.load(tmp_path / "rich" / "ontology.pt", weights_only=False)
    assert loaded["relation_count"] == 11 and loaded["enrichment"]["kind"] == "mesh"
    with pytest.raises(FileExistsError):
        t7_enrich.write_variant(loaded, tmp_path / "rich", root)


def test_queue_arms_respects_the_enrichment_threshold(tmp_path, monkeypatch) -> None:
    from vsa_embed.experiments import e9_plan, e9_tracks
    corpus = tmp_path / "tracks" / "rood-v1"; corpus.mkdir(parents=True)
    variant = tmp_path / "tracks" / "rood-v1-hostread"; variant.mkdir()
    torch.save({"entry_count": 100}, variant / "ontology.pt")
    monkeypatch.setattr(e9_tracks, "track_spec", lambda *a, **k: type("S", (), {"ontology": corpus / "ontology.pt"})())
    calls = []
    monkeypatch.setattr(e9_plan, "write_stage", lambda *a, **k: calls.append(k["models"]) or [Path("x.yaml")])
    monkeypatch.setattr(e9_plan, "queue_jobs", lambda *a, **k: ["queued-job"])
    (variant / "enrichment.json").write_text('{"entries_enriched": 5}')
    low = t7_enrich.queue_arms("rood-v1-hostread", ["C5host", "C5hostsh"], hosts=["SmolLM2-360M"], seeds=[1], priority=54.5, min_enriched=0.1)
    assert low["queued"] == [] and "skipped" in low and not calls
    (variant / "enrichment.json").write_text('{"entries_enriched": 40}')
    high = t7_enrich.queue_arms("rood-v1-hostread", ["C5host", "C5hostsh"], hosts=["SmolLM2-360M"], seeds=[1], priority=54.5, min_enriched=0.1)
    assert high["queued"] == ["queued-job"] and calls == [["C5host", "C5hostsh"]]
    with pytest.raises(ValueError, match="not ontology arms"):
        t7_enrich.queue_arms("rood-v1-hostread", ["C5mesh"], hosts=["SmolLM2-360M"], seeds=[1], priority=54.5)
