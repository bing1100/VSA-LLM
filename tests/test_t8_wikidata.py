"""T8 (Wikidata entities, decision 63): the Wikidata client cache and per-entity stores, the case-aware mention trie and
FineWeb counts, the alias policy and concept selection, the frame ontology, the FineWeb streams with the ROOD document
exclusion, the ROOD holdout eligibility, the benchmark name resolution and the item builders. No network."""

import gzip
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vsa_embed.experiments import t8_benchmarks as tb
from vsa_embed.experiments.t8_wikidata_corpus import (EntityDocuments, exclusion_names, iter_records, read_counts,
                                                      rood_eligibility, write_counts)
from vsa_embed.ontologies import wikidata as wd
from vsa_embed.span_channel import AliasTable


# -- client cache and stores ------------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.headers = payload, status, {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Answers claims queries from a fixed table; counts requests."""

    def __init__(self, claims):
        self.claims, self.requests, self.headers = claims, 0, {}

    def request(self, method, url, timeout=None, data=None, params=None, headers=None):
        self.requests += 1
        if params is not None:                                   # wbsearchentities
            return FakeResponse({"search": [{"id": "Q2"}, {"id": "Q3"}]})
        query = data["query"]
        rows = []
        for qid, props in self.claims.items():
            if f"wd:{qid} " in query + " ":
                for prop, values in props.items():
                    rows += [{"item": {"value": wd.ENTITY_PREFIX + qid}, "p": {"value": "http://www.wikidata.org/prop/direct/" + prop},
                              "value": {"value": wd.ENTITY_PREFIX + v}} for v in values]
        return FakeResponse({"results": {"bindings": rows}})


def test_client_caches_and_store_reuses_entities(tmp_path):
    session = FakeSession({"Q1": {"P31": ["Q5"], "P27": ["Q30", "Q145"]}, "Q2": {"P31": ["Q515"]}})
    client = wd.WikidataClient(tmp_path, session=session, min_interval=0, api_interval=0)
    first = wd.fetch_claims(client, ["Q1", "Q2"], ["P31", "P27"], batch=1)
    assert first == {"Q1": {"P27": ["Q30", "Q145"], "P31": ["Q5"]}, "Q2": {"P31": ["Q515"]}}
    assert session.requests == 2
    # A different entity set reuses the stored entities and queries only the new one.
    again = wd.fetch_claims(wd.WikidataClient(tmp_path, session=session, min_interval=0), ["Q2", "Q3"], ["P31", "P27"], batch=5)
    assert again["Q2"] == {"P31": ["Q515"]} and again["Q3"] == {} and session.requests == 3
    # Offline: everything comes from the store and the response cache.
    offline = wd.WikidataClient(tmp_path, offline=True)
    assert wd.fetch_claims(offline, ["Q1", "Q3"], ["P31", "P27"]) == {"Q1": first["Q1"], "Q3": {}}
    assert client.search("Paris") == [{"id": "Q2"}, {"id": "Q3"}]
    assert wd.WikidataClient(tmp_path, offline=True).search("Paris") == [{"id": "Q2"}, {"id": "Q3"}]
    with pytest.raises(FileNotFoundError):
        wd.WikidataClient(tmp_path, offline=True).search("Lyon")


def test_records_roundtrip_is_pinned(tmp_path):
    records = {"Q10": {"label": "B", "claims": {}}, "Q2": {"label": "A", "claims": {"P31": ["Q5"]}}}
    digest = wd.write_records(tmp_path / "r.jsonl.gz", records)
    assert wd.write_records(tmp_path / "s.jsonl.gz", records) == digest          # deterministic bytes (gzip mtime 0)
    assert wd.read_records(tmp_path / "r.jsonl.gz", expected_sha256=digest) == records
    with pytest.raises(ValueError):
        wd.read_records(tmp_path / "r.jsonl.gz", expected_sha256="0" * 64)


# -- mention counting ---------------------------------------------------------------------------------------------------

def test_key_trie_counts_every_occurrence_with_case():
    keys = [wd.entity_key(k) for k in ("New York", "New York City", "apple", "AC/DC")]
    trie = wd.KeyTrie(keys)
    found = Counter((keys[k], cased) for k, cased in trie.matches("New York City is in new york. An apple, Apple and AC/DC."))
    assert found[("new york", True)] == 1 and found[("new york", False)] == 1        # nested in "New York City" too
    assert found[("new york city", True)] == 1
    assert found[("apple", False)] == 1 and found[("apple", True)] == 1
    assert found[("ac / dc", True)] == 1
    assert trie.first("nothing here") is False and trie.first("AC/DC rocks")
    # With the names' capitals: "the American" is not "The American"; "the United States" is "the United States".
    names = ["The American", "the United States"]
    keys = [wd.entity_key(n) for n in names]
    masked = wd.KeyTrie(keys, [wd.capital_positions(n) for n in names])
    found = Counter((keys[k], cased) for k, cased in masked.matches("The American said the American way of the United States."))
    assert found == {("the american", True): 1, ("the american", False): 1, ("the united states", True): 1}
    records = {"Q1": {"label": "The American"}, "Q3": {"label": "the American"}, "Q2": {"label": "the United States"}}
    all_keys, by_record, masks = wd.candidate_keys(records)
    assert all_keys == ["the american", "the united states"] and masks == [(1,), (1, 2)]


def _shard(path: Path, rows: list[tuple[str, str]]) -> str:
    pq.write_table(pa.table({"id": [r[0] for r in rows], "text": [r[1] for r in rows]}), path, row_group_size=2)
    return str(path)


def test_count_fineweb_mentions_sides_and_skip(tmp_path):
    rows = [(f"<urn:{i}>", text) for i, text in enumerate(
        ["Paris is big.", "paris is a name.", "Paris and Paris.", "Lyon.", "Paris again.", "nothing"])]
    shards = [_shard(tmp_path / "a.parquet", rows[:3]), _shard(tmp_path / "b.parquet", rows[3:])]
    keys = ["paris", "lyon"]
    buckets = {doc_id: wd.doc_bucket(doc_id) for doc_id, _ in rows}
    eval_buckets = sorted(buckets.values())[2] + 1                   # some documents fall on the evaluation side
    counts = wd.count_fineweb_mentions(shards, keys, skip=1, eval_buckets=eval_buckets, workers=1, rows_per_job=2)
    expected = np.zeros((2, 2), np.int64)
    lower = np.zeros((2, 2), np.int64)
    for doc_id, text in rows[1:]:
        side = 1 if buckets[doc_id] < eval_buckets else 0
        for k, cased in wd.KeyTrie(keys).matches(text):
            expected[side, k] += 1
            lower[side, k] += 0 if cased else 1
    assert (counts.occurrences == expected).all() and (counts.lowercase == lower).all()
    assert counts.read.sum() == 5


# -- aliases, selection, ontology -------------------------------------------------------------------------------------------

def _records():
    return {
        "Q1": {"label": "Kyriakos Mitsotakis", "aliases": ["Mitsotakis", "PM"], "sitelinks": 90, "description": "Greek PM",
               "claims": {"P31": ["Q5"], "P27": ["Q41"], "P106": ["Q82955", "Q40348"]}, "kind": "benchmark"},
        "Q2": {"label": "Apple", "aliases": ["Apple Inc."], "sitelinks": 200, "claims": {"P31": ["Q4830453"], "P17": ["Q30"]},
               "kind": "benchmark"},
        "Q3": {"label": "Georgia", "aliases": [], "sitelinks": 150, "claims": {"P31": ["Q6256"]}, "kind": "benchmark"},
        "Q4": {"label": "Georgia", "aliases": [], "sitelinks": 120, "claims": {"P31": ["Q35657"], "P17": ["Q30"]}, "kind": "benchmark"},
        "Q41": {"label": "Greece", "aliases": ["Hellas"], "sitelinks": 300, "claims": {"P31": ["Q6256"]}, "kind": "filler1"},
        "Q30": {"label": "United States", "aliases": ["USA"], "sitelinks": 400, "claims": {"P31": ["Q6256"]}, "kind": "filler1"},
        "Q5": {"label": "human", "aliases": [], "sitelinks": 10, "claims": {}, "kind": "label_only"},
        "Q6256": {"label": "country", "aliases": [], "claims": {}, "kind": "label_only"},
        "Q82955": {"label": "politician", "aliases": [], "claims": {}, "kind": "label_only"},
        "Q40348": {"label": None, "aliases": [], "claims": {}, "kind": "label_only"},
    }


def test_alias_policy_case_and_sharing():
    policy = wd.WikidataAliasPolicy()
    assert wd.alias_decision("Kyriakos Mitsotakis", "kyriakos mitsotakis", is_label=True, owners=1, train=40, lowercase=0,
                             policy=policy) == "keep"
    assert wd.alias_decision("Apple", "apple", is_label=True, owners=1, train=1000, lowercase=700, policy=policy) == "lowercase_usage"
    assert wd.alias_decision("Georgia", "georgia", is_label=True, owners=2, train=50, lowercase=0, policy=policy) == "shared"
    assert wd.alias_decision("PM", "pm", is_label=False, owners=1, train=50, lowercase=0, policy=policy) == "short"
    assert wd.alias_decision("Hellas", "hellas", is_label=False, owners=1, train=40, lowercase=0, policy=policy) == "single_word_alias"
    assert wd.alias_decision("Hellenic Republic", "hellenic republic", is_label=False, owners=1, train=0, lowercase=0,
                             policy=policy) == "unseen_alias"
    assert wd.alias_decision("Zorblax", "zorblax", is_label=True, owners=1, train=0, lowercase=0, policy=policy) == "keep"
    assert wd.alias_decision("march", "march", is_label=True, owners=1, train=900, lowercase=10, policy=policy) == "no_capital"
    # "November" (a minor work) shares its label with the month: a homonym with more sitelinks.
    assert wd.alias_decision("November", "november", is_label=True, owners=1, train=900, lowercase=0, policy=policy,
                             sitelinks=4, homonym_sitelinks=180) == "homonym"
    assert wd.alias_decision("Paris", "paris", is_label=True, owners=1, train=900, lowercase=0, policy=policy,
                             sitelinks=250, homonym_sitelinks=40) == "keep"


def test_select_concepts_round_robin_and_cap():
    records = _records()
    counts = {"kyriakos mitsotakis": {"train": 40, "eval": 5, "train_documents": 30, "lowercase": 0},
              "mitsotakis": {"train": 60, "eval": 4, "train_documents": 50, "lowercase": 1},
              "apple": {"train": 1000, "eval": 90, "train_documents": 900, "lowercase": 700},
              "apple inc .": {"train": 30, "eval": 2, "train_documents": 25, "lowercase": 0},
              "georgia": {"train": 80, "eval": 9, "train_documents": 70, "lowercase": 0},
              "greece": {"train": 500, "eval": 40, "train_documents": 400, "lowercase": 0},
              "united states": {"train": 900, "eval": 80, "train_documents": 800, "lowercase": 2}}
    sources = {"Q1": ["bear:subject"], "Q2": ["lre:subject"], "Q3": ["popqa:object"], "Q4": ["bear:object"]}
    pool = [q for q, r in records.items() if r["label"] and r["kind"] != "label_only"]
    rows, stats = wd.select_concepts(records, counts, sources, policy=wd.WikidataAliasPolicy(), max_concepts=3,
                                     min_entity_mentions=5, source_order=["lre", "bear", "popqa", "filler"], pool=pool)
    chosen = {r["qid"] for r in rows}
    # Georgia is shared (both dropped); Apple keeps only "Apple Inc."; single-word aliases ("Mitsotakis", "USA") do not
    # link; one per source in turn, frequent first.
    assert chosen == {"Q1", "Q2", "Q30"}
    assert {r["alias"] for r in rows if r["qid"] == "Q2"} == {"Apple Inc."}
    assert {r["alias"] for r in rows if r["qid"] == "Q1"} == {"Kyriakos Mitsotakis"}
    assert stats["alias_shared"] == 2 and stats["selected"] == 3


def test_build_wikidata_ontology_frames(tmp_path):
    records = _records()
    selection = [{"qid": "Q1", "alias": "Kyriakos Mitsotakis"}, {"qid": "Q2", "alias": "Apple Inc."},
                 {"qid": "Q41", "alias": "Greece"}]
    ontology = wd.build_wikidata_ontology(records, selection, properties=["P31", "P27", "P106", "P17"], max_atomics=3,
                                          max_degree=16, max_values=3)
    assert ontology.concept_names == ["Q1", "Q2", "Q41"]
    assert ontology.relation_names == ["instance_of", "country_of_citizenship", "occupation", "country"]
    # Q40348 has no label and Q4830453 no record (both skipped); fillers used once each: Q5, Q30, Q41, Q6256, Q82955 → the
    # first 3 by QID number.
    assert ontology.atomic_names == ["wd:Q5", "wd:Q30", "wd:Q41"]
    frames = {q: [(ontology.relation_names[r], ontology.atomic_names[a]) for r, a in f]
              for q, f in zip(ontology.concept_names, ontology.frames)}
    assert frames["Q1"] == [("instance_of", "wd:Q5"), ("country_of_citizenship", "wd:Q41")]
    assert frames["Q2"] == [("country", "wd:Q30")]
    assert ontology.metadata["dropped_edges"]["filler_without_label"] == 2
    assert ontology.metadata["filler_types"]["wd:Q41".split(":")[1]] == "Q6256"
    assert ontology.alias_pairs == [("Kyriakos Mitsotakis", 0), ("Apple Inc.", 1), ("Greece", 2)]
    text = wd.verbalize_frame(ontology.frames[0], ontology.relation_names, ontology.atomic_names, ontology.metadata["filler_headings"])
    assert text == [["instance_of", "human"], ["country_of_citizenship", "Greece"]]
    templates = wd.relation_templates()
    assert set(templates) == {name for _, name, _ in wd.PROPERTIES}
    for name, spec in templates.items():                 # the statement's prefix is a held-out wording, never a prompt
        assert not spec.statement.startswith("{y}") and spec.statement.split("{y}")[0].rstrip() not in spec.prompts


def test_selection_file_pin(tmp_path):
    rows = [{"qid": "Q1", "alias": "A b", "key": "a b", "label": "A b", "train": 3, "eval": 1, "train_documents": 2,
             "lowercase": 0, "sources": "bear:subject"}]
    digest = wd.write_selection(tmp_path / "s.tsv", rows)
    assert wd.read_selection(tmp_path / "s.tsv", expected_sha256=digest) == rows
    with pytest.raises(ValueError):
        wd.read_selection(tmp_path / "s.tsv", expected_sha256="1" * 64)


# -- FineWeb streams with ROOD ------------------------------------------------------------------------------------------------

def test_entity_documents_rood_streams(tmp_path):
    texts = ["Greece is sunny.", "A plain text.", "Kyriakos Mitsotakis spoke.", "Another plain one.", "Greece and Hellas.",
             "Mitsotakis again.", "More plain words.", "Greece once more.", "Plain.", "Hellas only."] * 3
    rows = [(f"<urn:doc-{i}>", t) for i, t in enumerate(texts)]
    shards = [_shard(tmp_path / "a.parquet", rows[:15]), _shard(tmp_path / "b.parquet", rows[15:])]
    buckets = sorted(wd.doc_bucket(d) for d, _ in rows[2:])
    documents = EntityDocuments(shards=shards, skip=2, eval_buckets=buckets[6] + 1, eval_general_docs=2,
                                mention_keys=["greece", "kyriakos mitsotakis"])
    assert list(iter_records(shards, skip=2))[0] == rows[2]
    before = list(documents.train_records("entities"))
    rood = documents.with_exclusion(["kyriakos mitsotakis", "mitsotakis"])
    train = list(rood.train_records("entities")) + list(rood.train_records("general"))
    evaluation = list(rood.eval_records())
    train_ids, eval_ids = {d for d, _ in train}, {d for d, _ in evaluation}
    assert not train_ids & eval_ids
    assert all("Mitsotakis" not in t for _, t in train)                      # every document naming the held-out entity dropped
    assert {d for d, t in rows[2:] if "Mitsotakis" in t} <= eval_ids          # … and evaluated instead
    assert all(wd.doc_bucket(d) >= documents.eval_buckets for d in train_ids)
    assert len(before) > len(list(rood.train_records("entities")))
    general = list(rood.train_records("general"))
    assert all("Greece" not in t for _, t in general) and general
    mixed = list(rood.train_ids())
    assert [t for _, t, _ in mixed] == list(rood.train())                     # the audit re-reads exactly the training mix
    assert rood.signature("train") != documents.signature("train")
    assert list(rood.eval_general()) == [rows[0][1], rows[1][1]]


def test_rood_eligibility_node_disjoint_and_cost(tmp_path):
    records = _records()
    selection = [{"qid": "Q1", "alias": "Kyriakos Mitsotakis"}, {"qid": "Q41", "alias": "Greece"},
                 {"qid": "Q2", "alias": "Apple Inc."}]
    ontology = wd.build_wikidata_ontology(records, selection, properties=["P31", "P27"], max_atomics=10)
    table = AliasTable.from_pairs(ontology.alias_pairs)
    counts = {"kyriakos mitsotakis": {"train_documents": 3}, "mitsotakis": {"train_documents": 4}, "pm": {"train_documents": 900},
              "apple inc .": {"train_documents": 2}, "apple": {"train_documents": 5000}, "greece": {"train_documents": 50}}
    assert exclusion_names(records["Q1"], min_chars=3) == ["kyriakos mitsotakis", "mitsotakis"]
    result = rood_eligibility(table, ontology, records, counts, min_chars=3, max_documents=100)
    names = {ontology.concept_names[table.entry_concepts[e][0]] for e in result["eligible"]}
    assert names == {"Q1"}                                  # Greece fills Q1's frame; Apple's names cost 5,002 documents
    assert result["reasons"] == {"frame_filler": 1, "exclusion_cost": 1}
    path = tmp_path / "counts.tsv.gz"
    mention = wd.MentionCounts.zeros(2)
    mention.occurrences[0] = [5, 1]; mention.documents[0] = [4, 1]; mention.lowercase[0] = [1, 0]
    write_counts(path, ["greece", "hellas"], mention)
    assert read_counts(path)["greece"] == {"train": 5, "eval": 0, "train_documents": 4, "eval_documents": 0, "lowercase": 1}


# -- benchmarks ------------------------------------------------------------------------------------------------------------------

def test_choose_candidate_methods():
    claims = {"Q1": {"P31": ["Q5"], "P106": ["Q33999"]}, "Q2": {"P31": ["Q5"], "P106": ["Q82955"]}, "Q3": {"P31": ["Q4167410"]}}
    assert tb.choose_candidate(["Q3", "Q1", "Q2"], claims=claims, properties=["P106"], objects=["Q82955"]) == ("Q2", "verified")
    assert tb.choose_candidate(["Q3", "Q1", "Q2"], claims=claims, types=["Q5"]) == ("Q1", "typed")
    assert tb.choose_candidate(["Q3", "Q1"], claims=claims) == ("Q3", "top")
    assert tb.choose_candidate(["Q3", "Q1"], claims=claims, titles={"Q1": "Drake_(musician)"}, title="Drake (musician)") == ("Q1", "sitelink")
    assert tb.choose_candidate([], claims=claims) == (None, "unresolved")


def _benchmark_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmarks"
    bear = root / "bear" / "raw-github" / "BEAR"
    bear.mkdir(parents=True)
    meta = {}
    for relation in tb.BEAR_RELATIONS:
        meta[relation] = {"templates": ["[X] is a citizen of [Y].", "[Y] is where [X] lives."],
                          "answer_space_labels": ["Greece", "France"], "answer_space_ids": ["Q41", "Q142"]}
        (bear / f"{relation}.jsonl").write_text("" if relation != "P27" else json.dumps(
            {"sub_id": "Q1", "sub_label": "Kyriakos Mitsotakis", "sub_aliases": [], "obj_id": "Q41", "obj_label": "Greece",
             "answer_idx": 0}) + "\n")
    (bear / "metadata_relations.json").write_text(json.dumps(meta))
    (root / "popqa" / "raw").mkdir(parents=True)
    header = ["id", "subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "s_aliases", "o_aliases", "s_uri", "o_uri",
              "s_wiki_title", "o_wiki_title", "s_pop", "o_pop", "question", "possible_answers"]
    lines = ["\t".join(header)]
    for i, (s, sq, o, oq) in enumerate([("Kyriakos Mitsotakis", "Q1", "politician", "Q82955"), ("Apple Inc.", "Q2", "lawyer", "Q40348")]):
        lines.append("\t".join([str(i), s, "occupation", o, "1", "22", "2", "[]", "[]", wd.ENTITY_PREFIX + sq, wd.ENTITY_PREFIX + oq,
                                s, o, "10", "20", f"What is {s}'s occupation?", json.dumps([o])]))
    (root / "popqa" / "raw" / "test.tsv").write_text("\n".join(lines) + "\n")
    (root / "twohopfact" / "raw").mkdir(parents=True)
    cols = ["uid", "fact_comp_type", "e1.value", "e1.wikidata_qid", "e2.value", "e2.wikidata_qid", "e3.value", "e3.wikidata_qid",
            "r1.category", "r2.category", "r1(e1).prompt", "r2(e2).prompt", "r2(r1(e1)).prompt"]
    row = ["0", "x", "Apple Inc.", "Q2", "United States", "Q30", "Washington", "Q61", "orgz-hqcntry", "cntry-capital",
           "Apple Inc. is based in", "The capital of United States is", "The capital of the country of Apple Inc. is"]
    (root / "twohopfact" / "raw" / "TwoHopFact.csv").write_text(",".join(cols) + "\n" + ",".join(row) + "\n")
    lre = root / "lre" / "raw" / "data" / "factual"
    lre.mkdir(parents=True)
    for relation in tb.LRE_FACTUAL:
        samples = [{"subject": "Kyriakos Mitsotakis", "object": "Greece"}, {"subject": "Zorblax", "object": "France"}] \
            if relation == "city_in_country" else []
        (lre / f"{relation}.json").write_text(json.dumps({"prompt_templates": ["{} is part of"], "samples": samples}))
    for folder, names in (("entity_inferences", tb.EI_FILES), ("ecbd", tb.ECBD_FILES)):
        (root / "entity-inferences" / "raw" / "data" / folder).mkdir(parents=True)
        for name in names:
            row = {"ex_id": "Apple Inc._1_0_0" if folder == "ecbd" else "tv_dev_0", "ent_str": "Apple Inc.",
                   "category": "tv_show", "qid": "television series (Q5398426)",
                   "definition": "Apple Inc. is an American <extra_id_0> company.", "def_target": "<extra_id_0> technology <extra_id_1>",
                   "attribute": "x", "label": "funny",
                   "probe_sentences": {"template_0": {"probe_sentence": "Apple Inc. is very <extra_id_0> today.",
                                                      "labels": ["<extra_id_0> funny <extra_id_1>", "<extra_id_0> sad <extra_id_1>"]}}}
            (root / "entity-inferences" / "raw" / "data" / folder / f"{name}.json").write_text(json.dumps(row) + "\n")
    return root


def test_entity_references_and_items(tmp_path):
    root = _benchmark_root(tmp_path)
    refs = tb.entity_references(root)
    by = Counter((r["benchmark"], r["role"]) for r in refs)
    assert by[("bear", "subject")] == 1 and by[("popqa", "object")] == 2 and by[("twohopfact", "bridge")] == 1
    assert by[("lre", "subject")] == 2 and by[("entity_inferences", "subject")] == len(tb.EI_FILES)
    assert tb.search_names(refs) == ["Apple Inc.", "France", "Greece", "Kyriakos Mitsotakis", "Zorblax"]
    records = _records()
    ontology = wd.build_wikidata_ontology(records, [{"qid": "Q1", "alias": "Kyriakos Mitsotakis"}, {"qid": "Q2", "alias": "Apple Inc."},
                                                    {"qid": "Q41", "alias": "Greece"}], properties=["P31", "P27", "P17"])
    frames = [wd.verbalize_frame(f, ontology.relation_names, ontology.atomic_names, ontology.metadata["filler_headings"])
              for f in ontology.frames]
    view = tb.TrackView({q: i for i, q in enumerate(ontology.concept_names)}, {0: 0, 1: 1, 2: 2}, {"Q2"}, np.array([3, 0, 9]),
                        frames, [{ontology.atomic_names[a].split(":")[1] for _, a in f} for f in ontology.frames],
                        ["Greek PM", "", ""], {"kyriakos mitsotakis": 0, "apple inc.": 1, "greece": 2}, records)
    bear = list(tb.bear_items(view, root, 0))
    assert len(bear) == 2 and bear[0]["context"] == "Kyriakos Mitsotakis is a citizen of" and bear[0]["options"] == [" Greece.", " France."]
    assert bear[1]["context"] == "" and bear[1]["options"][0] == "Greece is where Kyriakos Mitsotakis lives."
    assert bear[0]["meta"]["t8"]["answer_in_frame"] and bear[0]["meta"]["t8"]["linked"] and not bear[0]["meta"]["t8"]["heldout"]
    assert bear[0]["terms"][0]["frame"] == [["instance_of", "human"], ["country_of_citizenship", "Greece"]]
    popqa = list(tb.popqa_items(view, root, 0))
    assert popqa[1]["meta"]["t8"]["heldout"] and popqa[0]["options"][popqa[0]["answer"]] == " politician"
    two = list(tb.twohop_items(view, root, 0))
    assert [i["set"] for i in two] == ["twohopfact-2hop", "twohopfact-hop1", "twohopfact-hop2"]
    assert two[0]["meta"]["t8"]["subject"] == "Q2" and len(two[0]["terms"]) == 2
    refs_by = {("entity_inferences", "subject", "tv_show", "Apple Inc."): {"qid": "Q2", "method": "typed"},
               ("lre", "subject", "city_in_country", "Kyriakos Mitsotakis"): {"qid": "Q1"},
               ("lre", "object", "city_in_country", "Greece"): {"qid": "Q41"}}
    ei = list(tb.ei_items(view, root, refs_by))
    assert ei[0]["context"] == "Apple Inc. is very" and ei[0]["options"] == [" funny today.", " sad today."] and ei[0]["answer"] == 0
    assert ei[0]["terms"][0]["definition"] == "Apple Inc. is an American technology company."
    lre = list(tb.lre_items(view, root, refs_by, 0))
    assert lre[0]["context"] == "Kyriakos Mitsotakis is part of" and lre[0]["meta"]["t8"]["answer_in_frame"]
    assert not lre[1]["meta"]["t8"]["in_track"]
    coverage = tb._coverage(bear + popqa)
    assert coverage["items"] == 4 and coverage["items_subject_heldout"] == 1 and coverage["subjects_in_track"] == 2


def test_track_registered():
    from vsa_embed.experiments.e9_tracks import TRACKS, track_spec
    spec = track_spec("t8", "qwen3")
    assert spec.data_root.name == "qwen3" and spec.eval_split == "eval-entities" and TRACKS["t8"].windows >= 1024
