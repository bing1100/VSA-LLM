"""T1c — the licensed clinical track (SNOMED CT × MIMIC-III; decision 58).

Every fixture here is synthetic: invented concept ids, terms and note texts in the RF2 / NOTEEVENTS layouts, never
licensed content.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from vsa_embed.data import mimic
from vsa_embed.ontologies import snomed as S


# -- MIMIC-III notes ---------------------------------------------------------------------------------------------

def _write_noteevents(path: Path, rows: list[dict]) -> None:
    import csv
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME",
                                                    "CATEGORY", "DESCRIPTION", "CGID", "ISERROR", "TEXT"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"CHARTDATE": "", "CHARTTIME": "", "STORETIME": "", "DESCRIPTION": "Report", "CGID": "", **row})


def _fake_notes(n: int = 400) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({"ROW_ID": i + 1, "SUBJECT_ID": 1000 + i % 97, "HADM_ID": "" if i % 11 == 0 else 5000 + i % 50,
                     "CATEGORY": ["Radiology", "Nursing", "Discharge summary"][i % 3],
                     "ISERROR": 1 if i % 50 == 7 else "",
                     "TEXT": "" if i % 60 == 5 else f"synthetic note {i}\nline two, with \"quotes\", commas\n\nend {i}"})
    return rows


def test_extract_notes_split_shuffle_and_counts(tmp_path: Path) -> None:
    source = tmp_path / "NOTEEVENTS.csv.gz"
    rows = _fake_notes()
    _write_noteevents(source, rows)
    out = tmp_path / "notes"
    stats = mimic.extract_notes(source, out, eval_buckets=2500, shards=4, chunk_rows=37)
    errors = sum(1 for r in rows if r["ISERROR"] == 1)
    empty = sum(1 for r in rows if r["ISERROR"] != 1 and not r["TEXT"])
    assert stats["rows"] == len(rows)
    assert stats["dropped_iserror"] == errors and stats["dropped_empty"] == empty
    assert sum(stats["notes"].values()) == len(rows) - errors - empty
    assert stats["patients_on_both_sides"] == 0 and stats["patients"]["eval"] > 0 and stats["patients"]["train"] > 0
    # Every note of a patient is on one side, decided by the patient hash.
    import pyarrow.parquet as pq
    seen: dict[str, set[int]] = {s: set() for s in mimic.SPLITS}
    for split in mimic.SPLITS:
        for shard in range(4):
            table = pq.read_table(mimic.shard_path(out, split, shard)).to_pydict()
            assert table["order"] == sorted(table["order"])                      # shuffled order within the shard
            assert all(o % 4 == shard for o in table["order"])
            seen[split] |= set(table["subject_id"])
            assert all((mimic.subject_bucket(s) < 2500) == (split == "eval") for s in table["subject_id"])
    assert not (seen["train"] & seen["eval"])
    texts = list(mimic.iter_notes(out, "train")) + list(mimic.iter_notes(out, "eval"))
    assert len(texts) == sum(stats["notes"].values())
    assert all("\n" in t and t == t.strip() for t in texts)                     # multi-line text survives the CSV
    assert len(list(mimic.iter_notes(out, "train", limit=5))) == 5
    only = list(mimic.iter_notes(out, "train", categories=frozenset({"Radiology"})))
    assert 0 < len(only) < stats["notes"]["train"]
    assert stats["categories"]["train"]["Radiology"] == len(only)
    assert mimic.notes_signature(out) == mimic.notes_signature(out)


def test_subject_bucket_is_stable() -> None:
    assert mimic.subject_bucket(123) == mimic.subject_bucket(123)
    shares = Counter(mimic.subject_bucket(s) < 1000 for s in range(20_000))
    assert 0.08 < shares[True] / 20_000 < 0.12


# -- SNOMED CT adapter (synthetic RF2 snapshot) -------------------------------------------------------------------

FINDING, BODY, ORGANISM, QUALIFIER = "404684003", "123037004", "410607006", "362981000"
SITE, AGENT, RARE_ATTR = "363698007", "246075003", "999000001"
STATED = "900000000000010007"


def _rf2(path: Path, header: str, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "".join("\t".join(r) + "\n" for r in rows))


@pytest.fixture
def snomed_release(tmp_path: Path) -> Path:
    """Invented concepts (ids 1001+) under the real top-level hierarchy ids; terms are made up."""
    root = tmp_path / "SnomedCT_Test"
    terminology, language = root / "Snapshot" / "Terminology", root / "Snapshot" / "Refset" / "Language"
    concepts = {S.ROOT: "Root (root)", FINDING: "Clinical finding (finding)", BODY: "Body structure (body structure)",
                ORGANISM: "Organism (organism)", QUALIFIER: "Qualifier value (qualifier value)",
                "1001": "Zorbic disorder (disorder)", "1002": "Left zorbic disorder (disorder)",
                "1003": "Plinth structure (body structure)", "1004": "Left plinth structure (body structure)",
                "1005": "Quorbacter (organism)", "1006": "Severe (qualifier value)", "1007": "Retired thing (disorder)",
                "1008": "Zorbic infection by quorbacter of the left plinth with many extra words (disorder)",
                S.IS_A: "Is a (attribute)", SITE: "Finding site (attribute)", AGENT: "Causative agent (attribute)",
                RARE_ATTR: "Rare attribute (attribute)"}
    active = {c: ("0" if c == "1007" else "1") for c in concepts}
    _rf2(terminology / "sct2_Concept_Snapshot_INT_20990101.txt", "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId",
         [[c, "20990101", active[c], "m", "d"] for c in concepts])
    descriptions, refset = [], []
    did = 5000
    def describe(concept: str, term: str, kind: str, accepted: bool = True, desc_active: str = "1") -> None:
        nonlocal did
        did += 1
        descriptions.append([str(did), "20990101", desc_active, "m", concept, "en", kind, term, "c"])
        if accepted:
            refset.append([f"r{did}", "20990101", "1", "m", S.US_ENGLISH, str(did), S.PREFERRED])
    for concept, fsn in concepts.items():
        describe(concept, fsn, S.FSN)
    describe("1001", "Zorbosis", S.SYNONYM)
    describe("1001", "ZD", S.SYNONYM)                       # bare acronym: dropped
    describe("1001", "Zorbic thing", S.SYNONYM, accepted=False)   # not in a language refset: ignored
    describe("1001", "Zorbic old", S.SYNONYM, desc_active="0")    # inactive description: ignored
    describe("1002", "the", S.SYNONYM)                      # function word: dropped
    describe("1003", "Plinth", S.SYNONYM)
    describe("1005", "Qb", S.SYNONYM)                       # mixed case, 2 characters: dropped as too short
    _rf2(terminology / "sct2_Description_Snapshot-en_INT_20990101.txt",
         "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId", descriptions)
    _rf2(language / "der2_cRefset_LanguageSnapshot-en_INT_20990101.txt",
         "id\teffectiveTime\tactive\tmoduleId\trefsetId\treferencedComponentId\tacceptabilityId", refset)
    rel = []
    def edge(source: str, destination: str, kind: str = S.IS_A, characteristic: str = S.INFERRED, group: str = "0") -> None:
        rel.append([f"e{len(rel)}", "20990101", "1", "m", source, destination, group, kind, characteristic, "x"])
    for top in (FINDING, BODY, ORGANISM, QUALIFIER):
        edge(top, S.ROOT)
    edge("1001", FINDING); edge("1002", "1001"); edge("1008", "1001"); edge("1007", FINDING)
    edge("1003", BODY); edge("1004", "1003"); edge("1005", ORGANISM); edge("1006", QUALIFIER)
    edge("1001", "1003", SITE); edge("1002", "1004", SITE); edge("1008", "1004", SITE, group="1")
    edge("1008", "1005", AGENT, group="1"); edge("1002", "1006", RARE_ATTR)
    edge("1002", "1005", AGENT, characteristic=STATED)     # stated, not inferred: ignored
    _rf2(terminology / "sct2_Relationship_Snapshot_INT_20990101.txt",
         "id\teffectiveTime\tactive\tmoduleId\tsourceId\tdestinationId\trelationshipGroup\ttypeId\tcharacteristicTypeId\tmodifierId", rel)
    _rf2(terminology / "sct2_TextDefinition_Snapshot-en_INT_20990101.txt",
         "id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId",
         [["t1", "20990101", "1", "m", "1001", "en", "def", "A made-up definition.", "c"]])
    return root


def test_split_semantic_tag_and_alias_decisions() -> None:
    assert S.split_semantic_tag("Zorbic disorder (disorder)") == ("Zorbic disorder", "disorder")
    assert S.split_semantic_tag("Zorbic (left) disorder") == ("Zorbic (left) disorder", None)
    assert S.split_semantic_tag("Plain term") == ("Plain term", None)
    policy = S.SnomedAliasPolicy(max_words=4)
    assert S.alias_decision("ZD", policy) == "dropped_abbreviation"
    assert S.alias_decision("COPD2", policy) == "dropped_abbreviation"
    assert S.alias_decision("Zorbic disorder", policy) == "keep"
    assert S.alias_decision("The", policy) == "dropped_function_word"
    assert S.alias_decision("Qb", policy) == "dropped_short"
    assert S.alias_decision("one two three four five", policy) == "dropped_long"
    assert S.alias_decision("Type II zorbosis", policy) == "keep"             # an all-caps word inside a phrase is kept
    assert S.alias_decision("ZDX", S.SnomedAliasPolicy(drop_abbreviations=False)) == "keep"


def test_snomed_ontology_frames_aliases_and_hierarchies(snomed_release: Path) -> None:
    onto = S.build_snomed_ontology(snomed_release, hierarchies=("clinical_finding", "body_structure", "organism"),
                                   max_atomics=64, max_degree=16, min_relation_edges=1, policy=S.SnomedAliasPolicy(max_words=6))
    names = onto.concept_names
    assert names == sorted(names, key=int)
    assert set(names) == {"1001", "1002", "1003", "1004", "1005", "1008"}       # inactive 1007 and qualifier 1006 excluded
    index = onto.concept_index
    aliases = {}
    for alias, concept in onto.alias_pairs:
        aliases.setdefault(names[concept], set()).add(alias)
    assert aliases["1001"] == {"Zorbic disorder", "Zorbosis"}                   # FSN without tag + synonym; ZD dropped
    assert aliases["1002"] == {"Left zorbic disorder"}                          # "the" dropped
    assert "1008" not in aliases                                               # its only term is too long
    assert aliases["1003"] == {"Plinth structure", "Plinth"}
    meta = onto.metadata
    assert meta["headings"][index["1001"]] == "Zorbic disorder" and meta["semantic_tags"][index["1003"]] == "body structure"
    assert meta["alias_stats"]["dropped_abbreviation"] == 1 and meta["alias_stats"]["dropped_function_word"] == 1
    assert meta["alias_stats"]["dropped_long"] == 1 and meta["alias_stats"]["dropped_short"] == 1
    assert meta["definitions"][index["1001"]] == 1
    rel = {name: i for i, name in enumerate(onto.relation_names)}
    assert {"hierarchy", "semantic_tag", "is_a", "finding_site", "causative_agent"} <= set(rel)
    atoms = onto.atomic_names
    frame = {(onto.relation_names[r], atoms[a]) for r, a in onto.frames[index["1008"]]}
    assert ("hierarchy", "top:clinical_finding") in frame and ("semantic_tag", "tag:disorder") in frame
    assert ("is_a", "sct:1001") in frame and ("finding_site", "sct:1004") in frame and ("causative_agent", "sct:1005") in frame
    frame2 = {(onto.relation_names[r], atoms[a]) for r, a in onto.frames[index["1002"]]}
    assert not any(r == "causative_agent" for r, _ in frame2)                   # the stated edge is ignored
    assert any(r.startswith("attr:") for r, _ in frame2)                        # unnamed attribute type, kept at min_relation_edges 1


def test_snomed_atomic_fallback_and_relation_threshold(snomed_release: Path) -> None:
    # Fillers in use: 1001 ×2 (is-a of 1002, 1008), 1004 ×2, 1003 ×1+1, 1005, tops...; a tiny dictionary forces fallbacks.
    onto = S.build_snomed_ontology(snomed_release, hierarchies=("clinical_finding", "body_structure", "organism"),
                                   max_atomics=8, max_degree=16, min_relation_edges=2)
    assert "causative_agent" not in onto.relation_names                         # 1 inferred edge < 2
    assert not any(r.startswith("attr:") for r in onto.relation_names)
    atoms = onto.atomic_names
    assert len(atoms) == 8 and atoms[:3] == ["top:clinical_finding", "top:body_structure", "top:organism"]
    index = onto.concept_index
    sites = {atoms[a] for r, a in onto.frames[index["1002"]] if onto.relation_names[r] == "finding_site"}
    assert sites and all(a.startswith("sct:") for a in sites)
    capped = S.build_snomed_ontology(snomed_release, hierarchies=("clinical_finding",), max_atomics=64, max_degree=3,
                                     min_relation_edges=1)
    assert max(len(f) for f in capped.frames) <= 3 and capped.metadata["frames_truncated"] >= 1
    with pytest.raises(ValueError):
        S.build_snomed_ontology(snomed_release, hierarchies=("no_such_hierarchy",))


# -- T1c corpus helpers ------------------------------------------------------------------------------------------

def test_clinical_documents_mix_and_signature(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vsa_embed.experiments.t1c_corpus import ClinicalDocuments
    source = tmp_path / "NOTEEVENTS.csv.gz"
    _write_noteevents(source, _fake_notes(300))
    notes = tmp_path / "notes"
    mimic.extract_notes(source, notes, eval_buckets=3000, shards=2)
    shard = tmp_path / "general.parquet"
    pq.write_table(pa.table({"text": [f"general document {i} " * 20 for i in range(200)]}), shard)
    docs = ClinicalDocuments(notes_dir=notes, general_shards=[str(shard)], general_skip=20, eval_general_docs=10,
                             eval_domain_docs=5, calibration=(3.5, 4.6))
    assert len(list(docs.eval_domain())) == 5 and len(list(docs.eval_general())) == 10
    log: list[int] = []
    mixed = list(docs.train(log))
    assert set(log) == {0, 1} and len(mixed) == len(log)
    domain_chars = sum(len(t) for t, s in zip(mixed, log) if s == 0) / 3.5
    general_chars = sum(len(t) for t, s in zip(mixed, log) if s == 1) / 4.6
    assert 0.3 < domain_chars / (domain_chars + general_chars) < 0.7
    assert docs.signature("train") != docs.signature("eval-mimic") and docs.signature("train") == docs.signature("train")
    other = ClinicalDocuments(notes_dir=notes, general_shards=[str(shard)], general_skip=20, eval_general_docs=10,
                              eval_domain_docs=6, calibration=(3.5, 4.6))
    assert other.signature("eval-mimic") != docs.signature("eval-mimic") and other.signature("train") == docs.signature("train")
    assert ClinicalDocuments.domain_split == "eval-mimic" and ClinicalDocuments.sources == ("mimic", "general")


def test_stratum_table_and_verdicts() -> None:
    import numpy as np
    from vsa_embed.experiments.t1c_corpus import stratum_table, stratum_verdict
    # entries: 0 held out, 1 unseen (freq 0), 2 rare-seen (freq 3), 3 frequent (freq 50)
    spans = {"entry": np.array([0, 0, 1, 2, 3, 3, 0, 1]), "length": np.array([2, 3, 2, 4, 1, 3, 2, 2])}
    heldout = np.array([True, False, False, False])
    frequency = np.array([0, 0, 3, 50])
    table = stratum_table(spans, threshold=2, heldout=heldout, frequency=frequency)
    assert table["after_heldout"] == {"occurrences": 3, "entries": 1, "entries_5plus": 0}
    assert table["after_unseen"]["occurrences"] == 2 and table["after_rare_seen"]["occurrences"] == 1
    assert table["after_len3plus"]["occurrences"] == 3                          # lengths 3, 4, 3 (≥ 2 kept first)
    assert table["all"]["occurrences"] == 7
    criteria = {"min_occurrences": 2000, "min_entries": 300, "exploratory_occurrences": 1000, "exploratory_entries": 50}
    assert stratum_verdict({"occurrences": 2500, "entries": 400, "entries_5plus": 310}, criteria) == "feasible"
    assert stratum_verdict({"occurrences": 2500, "entries": 400, "entries_5plus": 100}, criteria).startswith("feasible (")
    assert stratum_verdict({"occurrences": 1200, "entries": 60, "entries_5plus": 10}, criteria) == "exploratory-feasible"
    assert stratum_verdict({"occurrences": 100, "entries": 10, "entries_5plus": 1}, criteria) == "infeasible"


# -- inventory header safety ---------------------------------------------------------------------------------------

def test_inventory_never_reports_a_data_row_as_header(tmp_path: Path) -> None:
    from vsa_embed.data.clinical_inventory import describe, looks_like_header
    assert looks_like_header("ROW_ID,SUBJECT_ID,TEXT", "1,2,abc", trusted=True)[0]
    assert looks_like_header("subject_id,hadm_id", "10,20")[0]
    assert not looks_like_header("A00\tCholera", "A01\tTyphoid")[0]             # a headerless code table
    assert not looks_like_header("Some free text, with words", "more text, here")[0]
    data = tmp_path / "codes.txt"
    data.write_text("X00\tMadeup name\nX01\tOther name\n")
    entry = describe(data, tmp_path, rows=True, umls={})
    assert "columns" not in entry and entry["fields"] == 2 and entry["rows"] == 2
    csv = tmp_path / "table.csv"
    csv.write_text("code,count\nX00,5\n")
    entry = describe(csv, tmp_path, rows=True, umls={})
    assert entry["columns"] == ["code", "count"] and entry["rows"] == 1


# -- E9 registration -----------------------------------------------------------------------------------------------

def test_t1c_track_registration_keeps_licensed_paths_outside_the_repo() -> None:
    from vsa_embed.experiments import e9_plan, e9_tracks
    spec = e9_tracks.track_spec("t1c")
    assert spec.licensed and spec.eval_split == "eval-mimic" and spec.windows == 2048
    repo = Path.cwd().resolve()
    for path in (spec.data_root, spec.holdout_names, spec.alias_table_path, spec.items_dir, *e9_plan.dimension3_items("t1c")):
        assert not Path(path).resolve().is_relative_to(repo), path
    qwen = e9_tracks.track_spec("t1c", "qwen3")
    assert qwen.data_root == spec.data_root / "hosts" / "qwen3" and qwen.alias_table_path == spec.alias_table_path
    assert all(r in e9_tracks.T1C_TEMPLATES for r in spec.edit_relations)
    # the other tracks are unchanged
    assert e9_plan.dimension3_items("t5")[0] == e9_plan.ITEMS / "new-words-t5-smollm2-v1"
    assert e9_tracks.track_spec("t1").alias_table_path == e9_tracks.ALIAS_TABLE_DIR / "t1.json"


# -- exact source shares when the builder skips documents ------------------------------------------------------------

def test_source_shares_stay_exact_when_documents_are_skipped(tmp_path: Path) -> None:
    import transformers
    from vsa_embed.data.corpus import build_corpus
    from vsa_embed.experiments.t1_open_corpus import source_token_shares
    from vsa_embed.span_channel import AliasTable
    qwen = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    texts = ["first plain document about kangaroos.", "Le café de l'école.",      # not NFC: skipped
             "a general text " * 5, "another plain document.", "tail text that is never written " * 50]
    log = [0, 0, 1, 0, 1]
    table = AliasTable.from_pairs([("kangaroo", 0)])
    lengths = [len(qwen(t, add_special_tokens=False)["input_ids"]) + 1 for t in texts]
    budget = lengths[0] + lengths[2] + lengths[3]                    # stops after the fourth input document
    manifest = build_corpus(texts, tmp_path / "c", tokenizer_name="Qwen/Qwen2.5-0.5B", table=table, eos_id=qwen.eos_token_id,
                            max_tokens=budget, workers=1, vocab_size=len(qwen), batch_texts=2)
    assert manifest["skipped_documents"] == 1 and manifest["skipped_document_indices"] == [1] and manifest["documents"] == 3
    shares = source_token_shares(tmp_path / "c", log, qwen.eos_token_id, ("mimic", "general"))
    assert shares["exact"]
    assert shares["tokens"] == {"mimic": lengths[0] + lengths[3], "general": lengths[2]}
    assert shares["documents"] == {"mimic": 2, "general": 1}


def test_zeroshot_concepts_use_own_aliases_and_track_fillers(snomed_release: Path) -> None:
    from vsa_embed.experiments.e9_tracks import T1C_TEMPLATES
    from vsa_embed.experiments.t1c_corpus import zeroshot_concepts
    from vsa_embed.tracks.common import choice_items, entailment_items
    onto = S.build_snomed_ontology(snomed_release, hierarchies=("clinical_finding", "body_structure", "organism"),
                                   max_atomics=64, max_degree=16, min_relation_edges=1)
    concepts, pools = zeroshot_concepts(onto, {"1002", "1008"}, templates=T1C_TEMPLATES)
    by_id = {c["concept"]: c for c in concepts}
    assert set(by_id) == {"1002"}                         # 1008 has no alias of its own
    assert by_id["1002"]["surface"] == "Left zorbic disorder" and by_id["1002"]["split"] == "heldout"
    assert by_id["1002"]["facts"] == {"is_a": ["Zorbic disorder"], "finding_site": ["Left plinth structure"]}
    assert pools["finding_site"] == {"Plinth structure", "Left plinth structure"}
    items = choice_items(concepts, T1C_TEMPLATES, {r: sorted(v) for r, v in pools.items()}, track="t1c", task="p", seed=0,
                         choices=2)
    assert items and all(i["surface"] in i["prompt"] for i in items)
    statements = entailment_items(concepts, T1C_TEMPLATES, {r: sorted(v) for r, v in pools.items()}, track="t1c", task="e", seed=0)
    labels = Counter(i["label"] for i in statements)
    assert labels[0] == labels[1] >= 1


def test_e9_plan_dry_run_plans_t1c_without_touching_the_queue(tmp_path: Path) -> None:
    import torch
    from vsa_embed.experiments import e9_plan, e9_tracks
    torch.save({"entry_count": 295000, "atomic_count": 8192, "relation_count": 62}, tmp_path / "counts.pt")
    paths = e9_plan.write_stage("t1c", hosts=["SmolLM2-360M", "SmolLM2-135M"], models=["P0", "C0p", "C2", "C5"], seeds=[1, 2],
                                track="t1c", data_root=tmp_path / "data", counts_ontology=tmp_path / "counts.pt", root=tmp_path / "e9")
    configs = {p.stem: __import__("yaml").safe_load(p.read_text()) for p in paths}
    assert len(paths) == 2 * (1 + 3 * 2)
    c5 = configs["SmolLM2-360M-full-C5-s1"]
    assert c5["eval"]["windows"] == 2048 and c5["data"]["eval"].endswith("eval-mimic") and c5["e9_track"] == "t1c"
    queue = tmp_path / "jobs"
    planned: list = []
    names = e9_plan.queue_jobs(paths, "t1c", 62, track="t1c", root=tmp_path / "e9", queue_dir=queue, plan=planned)
    assert not queue.exists() and len(names) == len(planned)
    by_name = {n: (level, command) for n, level, command in planned}
    level, command = by_name["t1c-SmolLM2-360M-full-C5-s1-edit"]
    assert level == 63 and str(e9_plan.dimension3_items("t1c")[0]) in command
    hours = {n: e9_plan.job_estimate_hours(n, c, configs={str(p): configs[p.stem] for p in paths}) for n, _, c in planned}
    assert abs(hours["t1c-SmolLM2-360M-full-C5-s1"] - 67 / 60) < 1e-6 and abs(hours["t1c-SmolLM2-135M-full-C0p-s2"] - 31 / 60) < 1e-6
    assert hours["t1c-SmolLM2-360M-frozen-P0-s1"] < 0.1 and all(h > 0 for h in hours.values())
    assert by_name["t1c-quant-s1-2"][0] == 64 and by_name["t1c-report-s1-2"][0] == 65


def test_rrf_lines_join_lines_split_across_gzip_parts(tmp_path: Path) -> None:
    from vsa_embed.experiments.t1c_corpus import rrf_lines
    text = "C1|ENG|a|\nC2|ENG|bb|\nC3|ENG|ccc|\n"
    cut = text.index("bb") + 1                      # the split falls inside the second line
    for name, part in (("X.RRF.aa.gz", text[:cut]), ("X.RRF.ab.gz", text[cut:])):
        with gzip.open(tmp_path / name, "wt") as handle:
            handle.write(part)
    assert list(rrf_lines(sorted(tmp_path.glob("X.RRF.*gz")))) == ["C1|ENG|a|", "C2|ENG|bb|", "C3|ENG|ccc|"]
