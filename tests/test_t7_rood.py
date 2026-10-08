"""T7-ROOD (decision 63, holdout H1) and the E13 round split: the opt-in config section, the word-boundary name matcher,
the excluding training stream (drop, refill, document budget), the date split, the first-mention fallback, the leakage
audit, the `t7rood` E9 track and the T7 v1 defaults."""

import gzip
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vsa_embed.data.corpus import build_corpus
from vsa_embed.data.pubmed import pmid_bucket
from vsa_embed.experiments import t7_rood as rood
from vsa_embed.experiments.t1_open_corpus import TrackDocuments, load_config, save_if_changed
from vsa_embed.ontologies.mesh_novel import MentionCounter, parse_supplementary
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.span_channel import AliasTable

torch.set_num_threads(1)
T7 = Path("experiments/t7-new-vocabulary")


def _parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path, row_group_size=2)
    return path


# -- the opt-in section: T7 v1 is unchanged ------------------------------------------------------------------------------


def test_rood_is_opt_in_and_t7_keeps_its_build() -> None:
    t7 = load_config(T7 / "t7.yaml")
    assert "holdout" not in t7 and rood.RoodSettings.from_config(t7) is None          # T7 v1 builds as before
    settings = rood.RoodSettings.from_config(load_config(T7 / "t7-rood.yaml"))
    assert settings.split == "frequency" and settings.names == "candidates" and settings.match_train_tokens == 57_268_320
    loaded = load_config(T7 / "t7-rood.yaml")
    # the same frozen selection and holdout as T7 v1, a fresh data root
    assert loaded["data"]["expected_holdout_sha256"] == t7["data"]["expected_holdout_sha256"]
    assert loaded["ontology"] == t7["ontology"] and loaded["paths"]["data_root"] != t7["paths"]["data_root"]
    qwen = load_config(T7 / "t7-qwen3.yaml")
    assert "holdout" not in qwen and qwen["paths"] == t7["paths"] and qwen["hosts"][0]["name"] == "qwen3"
    rounds = rood.RoodSettings.from_config(load_config(T7 / "t7-rounds.yaml"))
    assert rounds.split == "date" and rounds.round2_fraction == pytest.approx(0.30)
    with pytest.raises(ValueError, match="exclude_training_documents"):
        rood.RoodSettings.from_config({"holdout": {"split": "date"}})
    with pytest.raises(ValueError, match="match_train_tokens"):
        rood.RoodSettings.from_config({"holdout": {"exclude_training_documents": True}})
    with pytest.raises(ValueError, match="unknown"):
        rood.RoodSettings.from_config({"holdout": {"exclude_training_documents": True, "top_up": "none", "x": 1}})


def test_save_if_changed_keeps_equal_files(tmp_path: Path) -> None:
    path = tmp_path / "ontology.pt"
    save_if_changed({"a": [1, 2]}, path)
    inode = path.stat().st_ino
    save_if_changed({"a": [1, 2]}, path)                 # equal bytes: the file is not touched
    assert path.stat().st_ino == inode
    save_if_changed({"a": [1, 3]}, path)                 # other content: atomically replaced
    assert path.stat().st_ino != inode and torch.load(path)["a"] == [1, 3]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ontology.pt"]


def test_supplementary_dates(tmp_path: Path) -> None:
    def record(ui: str, date: str) -> str:
        return (f'<SupplementalRecord SCRClass="1"><SupplementalRecordUI>{ui}</SupplementalRecordUI><SupplementalRecordName>'
                f"<String>{ui.lower()}</String></SupplementalRecordName>{date}<ConceptList/></SupplementalRecord>")
    xml = ("<SupplementalRecordSet>" + record("C1", "<DateIntroduced><Year>2013</Year><Month>02</Month><Day>01</Day></DateIntroduced>")
           + record("C2", "<DateIntroduced><Year>1974</Year></DateIntroduced>") + record("C3", "") + "</SupplementalRecordSet>")
    path = tmp_path / "supp.xml.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(xml)
    dates = {r["ui"]: (r["introduced"], r["introduced_date"]) for r in parse_supplementary(path)}
    assert dates == {"C1": (2013, "2013-02-01"), "C2": (1974, "1974-01-01"), "C3": (None, None)}


# -- the matcher ---------------------------------------------------------------------------------------------------------


def test_name_matcher_is_case_insensitive_at_word_boundaries() -> None:
    matcher = rood.NameMatcher(["Venetoclax", "fascin", "N,N-dimethyltryptamine", "  serial   C  "])
    assert matcher("Resistance to VENETOCLAX-based therapy") and matcher("(venetoclax).")
    assert matcher("Fascin bundles actin") and not matcher("a fascinating story")      # a prefix of another word is not a mention
    assert not matcher("venetoclaxs") and not matcher("pre-venetoclax2")
    assert matcher("n, n - dimethyltryptamine levels")                               # the screen's punctuation-spacing rule
    assert matcher("the serial c port") and not matcher("serial comm.")
    assert matcher.found("Venetoclax and fascin") == ["fascin", "venetoclax"]
    assert rood.NameMatcher(["fascin", "Venetoclax"]).digest() == rood.NameMatcher(["VENETOCLAX", "fascin"]).digest()


# -- the excluding stream ------------------------------------------------------------------------------------------------


def _documents(tmp_path: Path) -> TrackDocuments:
    train = [p for p in range(1000, 2000) if pmid_bucket(p) >= 800][:8]
    evaluation = [p for p in range(1000, 2000) if pmid_bucket(p) < 800][:3]
    texts = ["olezarsen lowers triglycerides in patients", "teclistamab in relapsed myeloma",
             "a plain abstract about sleep", "Tecvayli (teclistamab) dosing schedule", "olezarsen versus placebo trial",
             "another plain abstract about bones", "a TECVAYLI note without a selected name", "a third plain abstract"]
    rows = [{"pmid": p, "text": t} for p, t in zip(train, texts)]
    rows += [{"pmid": evaluation[0], "text": "teclistamab evaluation abstract"},
             {"pmid": evaluation[1], "text": "olezarsen evaluation abstract"}]
    path = _parquet(tmp_path / "pubmed.parquet", rows)
    # the general side outlasts the domain side, so the domain stream ends the mix (as in T7)
    general = ["general eval one", "general one about cooking", "general teclistamab gossip"]
    general += [f"general document number {i} about travel and music" for i in range(12)]
    general = _parquet(tmp_path / "general.parquet", [{"text": t} for t in general])
    return TrackDocuments(pubmed_paths=[path], eval_buckets=800, general_shards=[str(general)], general_skip=1,
                          eval_general_docs=1, mention_filter=MentionCounter(["olezarsen", "teclistamab"]),
                          mention_digest="test", calibration=(1.0, 1.0))


def test_excluding_stream_drops_refills_and_fixes_a_document_count(tmp_path: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    documents = _documents(tmp_path)
    matcher = rood.NameMatcher(["teclistamab", "Tecvayli"])
    stream = rood.excluding(documents, matcher, "domain")
    full = list(stream.train())
    assert not any(matcher(t) for t in full)
    domain = [t for t in full if not t.startswith("general")]
    # kept mention-bearing abstracts first, then the refill (no selected name) in file order; the TECVAYLI note is dropped
    assert domain == ["olezarsen lowers triglycerides in patients", "olezarsen versus placebo trial",
                      "a plain abstract about sleep", "another plain abstract about bones", "a third plain abstract"]
    assert "general teclistamab gossip" not in full and full.count("general one about cooking") == 1
    lengths = [len(tokenizer(t)["input_ids"]) + 1 for t in full]
    whole = rood.budget_pass(stream, tokenizer, sum(lengths) - 1)          # crossed by the last document, which is kept
    assert whole["train_documents"] == len(full) and whole["tokens"] == sum(lengths) and whole["reached_target"]
    tally = whole["tally"]
    assert (tally["pubmed_kept"], tally["pubmed_dropped"], tally["refill_kept"], tally["refill_dropped"]) == (2, 2, 3, 1)
    assert tally["general_dropped"] == 1 and tally["general_kept"] == sum(t.startswith("general") for t in full)
    assert whole["dropped_tokens"] == {"pubmed": sum(len(tokenizer(t)["input_ids"]) + 1 for t in (
        "teclistamab in relapsed myeloma", "Tecvayli (teclistamab) dosing schedule")),
        "refill": len(tokenizer("a TECVAYLI note without a selected name")["input_ids"]) + 1,
        "general": len(tokenizer("general teclistamab gossip")["input_ids"]) + 1}
    budget = rood.budget_pass(stream, tokenizer, sum(lengths[:5]) - 1)
    assert budget["train_documents"] == 5 and budget["tokens"] == sum(lengths[:5]) and len(budget["log"]) == 5
    assert budget["raw_general_read"] == budget["tally"].get("general_kept", 0) + budget["tally"].get("general_dropped", 0)
    assert budget["tally"]["general_kept"] == sum(t.startswith("general") for t in full[:5])
    assert not rood.budget_pass(stream, tokenizer, sum(lengths) + 10)["reached_target"]
    fixed = replace(stream, train_documents=5, tally=Counter(), dropped=None)
    log: list[int] = []
    assert list(fixed.train(log)) == full[:5] and log == budget["log"]
    # only the training stream's signature changes; evaluation streams are T7's
    for name in ("eval", "eval-pubmed", "eval-general"):
        assert fixed.signature(name) == documents.signature(name)
    assert fixed.signature("train") != documents.signature("train") != stream.signature("train")
    assert list(fixed.eval_pubmed()) == list(documents.eval_pubmed())
    assert list(fixed.flagged_pubmed()) == ["teclistamab in relapsed myeloma", "Tecvayli (teclistamab) dosing schedule"]
    # E13: round-2 streams — the flagged training abstracts with the general documents round 1 did not read
    streams = rood.round_streams(fixed, raw_general_read=budget["raw_general_read"])
    round2 = list(streams["train-round2"][0]([]))
    assert {t for t in round2 if not t.startswith("general")} == set(fixed.flagged_pubmed())
    assert not set(round2) & set(full[:5])
    assert list(streams["eval-round2"][0]([])) == ["teclistamab evaluation abstract"]
    assert list(streams["eval-round1"][0]([])) == ["olezarsen evaluation abstract"]


# -- the date split ------------------------------------------------------------------------------------------------------


def test_date_holdout_takes_the_latest_records_and_closes_containment() -> None:
    dates = ["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01", "2005-01-01", "2006-01-01", "2010-05-05",
             "2010-05-05", "2020-01-01", "2021-01-01"]
    names = [f"C{i}" for i in range(10)]
    aliases = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa iota"]
    pairs = [(a, i) for i, a in enumerate(aliases)]
    ontology = FrameOntology("t", names, ["r"], ["x"], [[] for _ in names], pairs, {})
    records = [{"ui": n, "introduced_date": d} for n, d in zip(names, dates)]
    table = AliasTable.from_pairs(pairs)
    out = rood.date_holdout({"holdout": {"round2_fraction": 0.3}}, ontology, table, records=records)
    by_concept = {table.entry_concepts[e][0] for e in out["heldout_entries"]}
    # 7 of 10 stay in round 1; the cut (2010-05-05) is shared by C6 and C7, which stay; C9's alias contains C8's
    assert out["split"]["cut_date"] == "2010-05-05"
    assert by_concept == {8, 9} and {table.entry_concepts[e][0] for e in out["chosen_entries"]} == {8, 9}
    assert out["names"] == ["C8", "C9"] and out["split"]["date_sources"] == {"DateIntroduced": 10}
    # a round-1 record whose alias contains a round-2 alias moves to round 2 (the linker's alias-disjoint closure)
    pairs[5] = ("zeta kappa iota", 5)
    ontology = FrameOntology("t", names, ["r"], ["x"], [[] for _ in names], pairs, {})
    table = AliasTable.from_pairs(pairs)
    out = rood.date_holdout({"holdout": {"round2_fraction": 0.2}}, ontology, table, records=records)
    assert {table.entry_concepts[e][0] for e in out["closure_entries"]} == {5}
    assert {table.entry_concepts[e][0] for e in out["heldout_entries"]} == {5, 8, 9}


def test_first_mention_fallback(tmp_path: Path) -> None:
    train = [p for p in range(3000, 4000) if pmid_bucket(p) >= 800][:3]
    path = _parquet(tmp_path / "p.parquet", [{"pmid": train[2], "year": 2026, "text": "zorbamab works"},
                                            {"pmid": train[0], "year": 2025, "text": "early zorbamab report"},
                                            {"pmid": train[1], "year": 2024, "text": "unrelated"}])
    found = rood.first_mention_dates([path], {0: ["Zorbamab"], 1: ["never seen"]}, eval_buckets=800)
    assert found == {0: "2025-12-31"}


# -- the leakage audit ---------------------------------------------------------------------------------------------------


def test_audit_counts_leaks_composition_and_overlap(tmp_path: Path) -> None:
    table = AliasTable.from_pairs([("olezarsen", 0)])
    kwargs = dict(tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=10_000, workers=1, batch_texts=2)
    clean = ["olezarsen trial one", "a refill abstract", "general text here"]
    build_corpus(iter(clean), tmp_path / "clean", **kwargs)
    build_corpus(iter(clean[:1] + ["teclistamab slipped through", "general text here", "older general doc"]),
                 tmp_path / "reference", **kwargs)
    matcher = rood.NameMatcher(["teclistamab"])
    audit = rood.audit_training(tmp_path / "clean", matcher, reference=tmp_path / "reference", log=[0, 0, 1], kept_mentions=1)
    assert audit["leaked_documents"] == 0 and audit["exact"] and audit["pieces"] == 3
    comp = audit["composition"]
    assert comp["mention_documents"] == comp["refill_documents"] == comp["general_documents"] == 1
    assert comp["mention_tokens"] + comp["refill_tokens"] + comp["general_tokens"] == audit["tokens"]
    ref = audit["reference"]
    assert ref["shared"]["documents"] == 2 and ref["reference_only"] == {"documents": 2, "tokens": ref["reference_only"]["tokens"],
                                                                         "flagged": 1}
    assert ref["this_only"]["documents"] == 1
    leaky = rood.audit_training(tmp_path / "reference", matcher)
    assert leaky["leaked_documents"] == 1 and leaky["leak_examples"] == [{"document": 1, "names": ["teclistamab"]}]
    same = rood.compare_corpora(tmp_path, tmp_path, names=["clean"])
    assert same["clean"] == {"tokens.bin": True, "spans": True, "manifest": True}


# -- the E9 track ----------------------------------------------------------------------------------------------------------


def test_t7rood_track_and_its_e9_plan(tmp_path: Path) -> None:
    import yaml
    from vsa_embed.experiments import e9_plan
    from vsa_embed.experiments.e9_tracks import QWEN3_ROOTS, TRACKS, track_spec
    t7, spec = TRACKS["t7"], TRACKS["t7rood"]
    for key in ("eval_split", "windows", "category_relations", "kept_relations", "edit_relations", "general_split"):
        assert getattr(spec, key) == getattr(t7, key)
    assert spec.config == T7 / "t7-rood.yaml" and spec.alias_table_path.name == "t7rood.json"
    assert track_spec("t7rood", "qwen3").data_root == QWEN3_ROOTS["t7rood"] == spec.data_root / "hosts" / "qwen3"
    assert QWEN3_ROOTS["t7"] == t7.data_root / "hosts" / "qwen3"
    assert e9_plan.dimension3_items("t7rood", "qwen3")[1].name == "edits-t7rood-qwen3-v1"
    torch.save({"entry_count": 8184, "atomic_count": 3295, "relation_count": 7}, tmp_path / "counts.pt")
    paths = e9_plan.write_stage("t7rood", hosts=["SmolLM2-360M"], models=list(e9_plan.MODELS), seeds=[1, 2, 3], track="t7rood",
                                data_root=tmp_path / "rood", counts_ontology=tmp_path / "counts.pt", root=tmp_path / "e9")
    c5 = yaml.safe_load(next(p for p in paths if p.stem == "SmolLM2-360M-full-C5-s3").read_text())
    assert c5["e9_track"] == "t7rood" and c5["eval"]["windows"] == 4096 and c5["data"]["eval"].endswith("eval-pubmed")
    planned: list = []
    e9_plan.queue_jobs(paths, "t7rood", 54.497, track="t7rood", root=tmp_path / "e9", queue_dir=tmp_path / "jobs", plan=planned,
                       level_step=0.0001)
    levels = {level for _, level, _ in planned}
    assert not (tmp_path / "jobs").exists() and min(levels) == 54.497 and max(levels) == pytest.approx(54.4973)
    edit = next(command for name, _, command in planned if name == "t7rood-SmolLM2-360M-full-C5-s1-edit")
    assert edit[edit.index("--new-items") + 1].endswith("new-words-t7rood-smollm2-v1")
