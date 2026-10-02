"""WP-host: per-host corpora share C3's entry ids and frozen holdout under every tokenizer."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.experiments import c3_corpus, host_corpus
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.span_channel import AliasTable

TOKENIZERS = ["gpt2", "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B"]


def _cached() -> bool:
    try:
        for name in TOKENIZERS:
            transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
            transformers.AutoConfig.from_pretrained(name, local_files_only=True)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _cached(), reason="host tokenizers not cached")

LEMMAS = ["acetaminophen", "hydroxychloroquine", "new york", "ibuprofen", "cat", "photosynthesis", "mitochondrion",
          "bank", "bank", "chlorophyll", "electroencephalography", "kangaroo", "thermodynamics", "paracetamol"]


def fake_ontology(*_args, **_kwargs) -> FrameOntology:
    """A small frame ontology standing in for WordNet (two senses of "bank" share one entry)."""
    names = [f"{lemma.replace(' ', '_')}.n.{i:02d}" for i, lemma in enumerate(LEMMAS)]
    frames = [[(0, i % 6), (1, (i + 1) % 6)] for i in range(len(LEMMAS))]
    pairs = [(lemma, i) for i, lemma in enumerate(LEMMAS)] + [("tylenol", 0), ("paracetamol", 0)]
    return FrameOntology("fake", names, ["hypernym", "pos"], [f"a{i}" for i in range(6)], frames, pairs,
                         {"wordnet_version": "fake"})


def documents(count: int, offset: int) -> list[str]:
    rng = np.random.default_rng(offset)
    words = ["the", "a", "study", "of", "with", "and", "in", "showed", "that", "river", "people", "data"]
    texts = []
    for i in range(count):
        body = []
        for _ in range(40):
            body.append(LEMMAS[rng.integers(len(LEMMAS))] if rng.random() < 0.3 else words[rng.integers(len(words))])
        texts.append(f"Document {offset + i}. " + " ".join(body).capitalize() + ".")
    return texts


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("host")
    shards = []
    for k, (count, offset) in enumerate([(60, 0), (120, 1000)]):
        path = root / f"shard{k}.parquet"
        pq.write_table(pa.table({"text": documents(count, offset)}), path)
        shards.append(str(path))
    c3_config = {
        "experiment": "c3-test", "seed": 7, "tokenizer": "gpt2", "workers": 2,
        "paths": {"data_root": str(root / "c3-data"), "shards": shards,
                  "shard_sha256": {Path(shards[0]).name: host_corpus._sha256_file(Path(shards[0]))[:16]}},
        "ontology": {"max_atomics": 16, "max_degree": 4},
        "data": {"eval_docs": 20, "presample_docs": 40, "holdout_fraction": 0.3, "holdout_min_count": 1, "min_subtokens": 2,
                 "train_tokens": 6000, "train_min_subtokens": 2, "l1_slice_tokens": 1000, "cardinality_docs": 10},
        "cardinality_tokenizers": ["gpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"],
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c3_corpus, "build_wordnet_ontology", fake_ontology)
        patch.setattr(host_corpus, "build_wordnet_ontology", fake_ontology)
        c3_corpus.run(c3_config, root / "c3-run")
        hosts = {}
        for host in ("smollm2", "qwen2.5"):
            config = {"host": host, "c3_run": str(root / "c3-run"), "workers": 2,
                      "paths": {"data_root": str(root / f"{host}-data")}, "data": {"train_tokens": 6000}}
            hosts[host] = {"summary": host_corpus.run(config, root / f"{host}-run"), "data": root / f"{host}-data",
                           "run": root / f"{host}-run"}
    return {"root": root, "c3_data": root / "c3-data", "c3_run": root / "c3-run", "hosts": hosts, "shards": shards}


def test_host_corpora_share_entry_ids_and_holdout_with_c3(built) -> None:
    c3 = torch.load(built["c3_data"] / "ontology.pt", weights_only=False)
    assert c3["heldout_entries"], "the synthetic C3 run must hold out something"
    for host, expected in (("smollm2", ("HuggingFaceTB/SmolLM2-135M", 49152, "uint16")),
                           ("qwen2.5", ("Qwen/Qwen2.5-0.5B", 151665, "uint32"))):
        onto = torch.load(built["hosts"][host]["data"] / "ontology.pt", weights_only=False)
        for key in ("entry_count", "entry_concepts", "heldout_entries", "alias_table_sha256", "holdout_sha256",
                    "concept_names", "atomic_names", "relation_names"):
            assert onto[key] == c3[key], (host, key)
        for key in ("offsets", "relations", "fillers"):
            assert torch.equal(onto[key], c3[key])
        assert (onto["tokenizer"], onto["vocab_size"]) == expected[:2]
        summary = built["hosts"][host]["summary"]
        assert summary["token_dtype"] == expected[2]
        assert "c3 ontology.pt" in summary["verification"]["holdout_sha256"]["verified_against"]
        assert "c3 eval/manifest.json" in summary["verification"]["alias_table_sha256"]["verified_against"]
        assert "c3 train/manifest.json" in summary["verification"]["train_alias_table_sha256"]["verified_against"]
        assert summary["eval_documents"]["same_documents"]
        cardinality = json.loads((built["hosts"][host]["run"] / "cardinality.json").read_text())
        assert cardinality["sample_matches_c3"] is True
        assert len(onto["train_frequency"]) == onto["entry_count"]
        for name in ("manifest.json", "resolved_config.yaml", "report.md", "summary.json"):
            assert (built["hosts"][host]["run"] / name).is_file()
    smol = torch.load(built["hosts"]["smollm2"]["data"] / "ontology.pt", weights_only=False)
    assert smol["models"] == ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"]


def test_heldout_aliases_are_absent_from_host_training_spans(built) -> None:
    c3 = torch.load(built["c3_data"] / "ontology.pt", weights_only=False)
    held = set(c3["heldout_entries"])
    heldout_names = {c3["concept_names"][c] for e in held for c in c3["entry_concepts"][e]}
    heldout_aliases = {lemma for i, lemma in enumerate(LEMMAS) if fake_ontology().concept_names[i] in heldout_names}
    for host in ("smollm2", "qwen2.5"):
        data = built["hosts"][host]["data"]
        train, evaluation = TokenCorpus.open(data / "train"), TokenCorpus.open(data / "eval")
        assert not set(train.spans["entry"].tolist()) & held
        assert bool((train.spans["length"] >= 2).all())
        assert set(evaluation.spans["entry"].tolist()) & held        # linked at evaluation
        tokenizer = transformers.AutoTokenizer.from_pretrained(host_corpus.HOSTS[host]["tokenizer"], local_files_only=True)
        text = tokenizer.decode(np.asarray(train.tokens).tolist()).lower()
        assert any(alias in text for alias in heldout_aliases)        # present in the text, never linked
        for start, end in zip(train.spans["start"][:50], train.spans["end"][:50]):
            surface = tokenizer.decode(np.asarray(train.tokens[start:end + 1]).tolist()).strip().lower()
            assert not any(surface.endswith(alias) for alias in heldout_aliases)


def test_qwen_token_ids_above_uint16_round_trip(built, tmp_path: Path) -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    texts = ["The kangaroo (袋鼠) lives in 澳大利亚; hydroxychloroquine 🦘 is a drug.", "超长的中文句子用于测试词表 ids."] * 3
    table = AliasTable.from_pairs([("kangaroo", 0), ("hydroxychloroquine", 1)])
    manifest = build_corpus(texts, tmp_path / "q", tokenizer_name="Qwen/Qwen2.5-0.5B", table=table,
                            eos_id=tokenizer.eos_token_id, max_tokens=10_000, batch_texts=2, workers=1,
                            vocab_size=len(tokenizer), normalization="NFC")
    corpus = TokenCorpus.open(tmp_path / "q")
    expected = []
    for text in texts:
        expected += tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    assert manifest["dtype"] == "uint32" and corpus.tokens.dtype == np.uint32
    assert max(expected) > 65535
    assert np.asarray(corpus.tokens).tolist() == expected
    ids, spans = corpus.window(0, 20)
    assert ids.dtype == np.int64 and int(ids.max()) > 65535
    host_tokens = TokenCorpus.open(built["hosts"]["qwen2.5"]["data"] / "eval").tokens
    assert host_tokens.dtype == np.uint32


def test_normalizing_tokenizer_keeps_non_nfc_documents(tmp_path: Path) -> None:
    qwen = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    smol = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=True)
    assert host_corpus.tokenizer_normalization(qwen) == "NFC"
    assert host_corpus.tokenizer_normalization(smol) is None
    texts = ["Le café de l'école and the kangaroo."]        # decomposed accents (not NFC)
    table = AliasTable.from_pairs([("kangaroo", 0)])
    common = dict(tokenizer_name="Qwen/Qwen2.5-0.5B", table=table, eos_id=qwen.eos_token_id, max_tokens=1000,
                  workers=1, vocab_size=len(qwen))
    assert build_corpus(texts, tmp_path / "raw", **common)["skipped_documents"] == 1
    kept = build_corpus(texts, tmp_path / "nfc", normalization="NFC", **common)
    assert kept["skipped_documents"] == 0 and kept["normalization"] == "NFC" and kept["spans"] == 1


def test_verification_rejects_another_holdout(built, monkeypatch) -> None:
    monkeypatch.setattr(host_corpus, "build_wordnet_ontology", fake_ontology)
    record = host_corpus.load_c3_record(built["c3_run"])
    tables = host_corpus.rebuild_c3_tables(record["config"], record["holdout_names"])
    assert host_corpus.verify_against_c3(tables, record)["alias_table_sha256"]["value"] == tables["full"].digest()
    others = host_corpus.rebuild_c3_tables(record["config"], record["holdout_names"][1:] or ["cat.n.04"])
    with pytest.raises(ValueError, match="holdout_sha256"):
        host_corpus.verify_against_c3(others, record)
    bare = {**record, "ontology": None, "summary": None, "manifests": {}}
    with pytest.raises(ValueError, match="no C3 record"):
        host_corpus.verify_against_c3(tables, bare)
    assert host_corpus.verify_against_c3(tables, bare, require_record=False)["holdout_sha256"]["verified_against"] == []


def test_hosts_sharing_a_corpus_share_the_tokenizer_and_fit_their_vocabulary() -> None:
    for host, settings in host_corpus.HOSTS.items():
        tokenizer = transformers.AutoTokenizer.from_pretrained(settings["tokenizer"], local_files_only=True)
        fingerprint = host_corpus.tokenizer_fingerprint(tokenizer)
        for model in settings["models"]:
            other = transformers.AutoTokenizer.from_pretrained(model, local_files_only=True)
            assert host_corpus.tokenizer_fingerprint(other) == fingerprint, (host, model)
            assert transformers.AutoConfig.from_pretrained(model, local_files_only=True).vocab_size >= len(tokenizer)


def test_covered_tokens_counts_the_union_of_spans() -> None:
    start, end = np.array([0, 2, 3, 10, 12]), np.array([4, 3, 6, 10, 13])
    assert host_corpus.covered_tokens(start, end) == len({*range(0, 7), 10, 12, 13})
    assert host_corpus.covered_tokens(np.array([], dtype=np.int32), np.array([], dtype=np.int32)) == 0
