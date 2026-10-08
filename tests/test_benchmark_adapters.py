"""Benchmark adapters (`vsa_embed.benchmarks.adapters`) and raw-file records (`benchmarks.sources`) on tiny fixtures
(no network): WordNet is-a frames, COMPS-WUGS minimal pairs, ALCUNA ranking forms, Entity Inferences, the reversal
set, LRE, BEAR and PopQA items, deterministic builds and manifests."""

import csv
import gzip
import json
from pathlib import Path

import pytest
import torch

from vsa_embed.benchmarks import adapters as ad
from vsa_embed.benchmarks import ranking as rk
from vsa_embed.benchmarks import sources as src


def _wordnet():
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("dog")
        return wordnet
    except LookupError:
        return None


WN = _wordnet()
needs_wordnet = pytest.mark.skipif(WN is None, reason="WordNet not available")


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    yield


def _ontology() -> dict:
    atoms = ["synset:canine.n.02", "synset:feline.n.01", "synset:bird.n.01", "synset:carnivore.n.01", "synset:vine.n.01",
             "lexname:noun.animal", "lexname:noun.plant", "pos:n"]
    relations = ["hypernym", "instance_hypernym", "lexname", "pos"]
    # one entry: dog → hypernym canine, lexname animal, pos n
    return {"atomic_names": atoms, "relation_names": relations, "offsets": torch.tensor([0, 3]),
            "relations": torch.tensor([0, 2, 3]), "fillers": torch.tensor([0, 5, 7])}


@needs_wordnet
def test_wordnet_frames_map_to_the_nearest_atomic() -> None:
    frames = ad.WordNetFrames(_ontology(), WN)
    dog = WN.synset("dog.n.01")
    frame, info = frames.isa_frame(dog)
    assert info["depth"] == 1 and frame[0] == ["hypernym", "synset:canine.n.02"]
    assert ["lexname", "lexname:noun.animal"] in frame and ["pos", "pos:n"] in frame
    exact, info = frames.isa_frame(WN.synset("canine.n.02"))
    assert info["depth"] == 0 and exact[0] == ["hypernym", "synset:canine.n.02"]
    assert frames.nearest_atomic(WN.synset("idea.n.01")) == (None, None)
    kind, synset = ad.taxon_synset("Bougainvillea spinosa", WN)
    assert kind in {"genus", "full"} and synset.lexname() == "noun.plant"
    assert ad.taxon_synset("Xyzzy plughia", WN) == (None, None)


def _comps_raw(root: Path) -> Path:
    raw = root / "raw"
    (raw / "data/comps").mkdir(parents=True)
    rows = [{"id": 1, "base_id": 7, "nonsense_words": ["wug", "fep"], "property": "can bark", "acceptable_concept": "dog",
             "unacceptable_concept": "cat", "prefix_acceptable": "A wug is a dog. Therefore, a wug",
             "prefix_unacceptable": "A wug is a cat. Therefore, a wug", "property_phrase": "can bark.",
             "negative_sample_type": "taxonomic", "similarity": 0.5, "distraction_type": "undistracted"},
            {"id": 2, "base_id": 8, "nonsense_words": ["dax", "fep"], "property": "can fly", "acceptable_concept": "robin",
             "unacceptable_concept": "dog", "prefix_acceptable": "A dax is a robin. Therefore, a dax",
             "prefix_unacceptable": "A dax is a dog. Therefore, a dax", "property_phrase": "can fly.",
             "negative_sample_type": "random", "similarity": 0.1, "distraction_type": "undistracted"}]
    (raw / "data/comps/comps_wugs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with open(raw / "data/concept_senses.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["category", "concept", "sensekey", "article"])
        for c, key in (("dog", "dog%1:05:00::"), ("cat", "cat%1:05:00::"), ("robin", "robin%1:05:02::")):
            w.writerow(["animal", c, key, f"a {c}"])
    return raw


@needs_wordnet
def test_comps_wugs_items_are_minimal_pairs(tmp_path) -> None:
    raw = _comps_raw(tmp_path)
    items, coverage = ad.build_comps_wugs(raw, ad.WordNetFrames(_ontology(), WN))
    assert len(items) == 2 and coverage["concepts"] == 3 and coverage["pairs"] == 2
    first = items[0]
    assert first.context == "Therefore, a wug" and first.options == [" can bark.", " can bark."] and first.answer == 0
    assert first.joiner == " " and [t[0].definition for t in first.option_terms] == ["A wug is a dog.", "A wug is a cat."]
    assert first.option_terms[0][0].frame[0] == ("hypernym", "synset:canine.n.02")
    assert first.option_terms[1][0].frame[0] == ("hypernym", "synset:feline.n.01")
    assert first.option_terms[0][0].insert and not first.meta["frames_equal"] and first.meta["exact_both"] is False
    # the in-context route rebuilds the original COMPS prompt exactly
    assert first.option_terms[0][0].definition + first.joiner + first.context == "A wug is a dog. Therefore, a wug"


def _alcuna_raw(root: Path) -> Path:
    raw = root / "raw"
    (raw / "dataset").mkdir(parents=True)
    meta = [{"artificial_entity": {"id": -5, "name": "Bainvillea spina", "rank": "species",
                                   "properties": [{"name": "habitat", "type": "attribute", "values": ["terrestrial"]},
                                                  {"name": "eat", "type": "relation", "values": ["Cleridae"]}]},
             "parent_entity": {"id": 5, "name": "Bougainvillea spinosa", "rank": "species", "properties": []}, "difference": {}},
            {"artificial_entity": {"id": -6, "name": "Zorbix", "rank": "genus", "properties": []},
             "parent_entity": {"id": 6, "name": "Xyzzy plughia", "rank": "species", "properties": []}, "difference": {}}]
    (raw / "dataset/meta_data.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    questions = {"-5": [{"question": "What is the habitat of Bainvillea spina?\n0. marine\n\n1. terrestrial\n\n2. freshwater\n\n3. aerial",
                         "answers": [1], "form": "multi-choice", "type": "Knowledge Understanding", "meta_data": {"difference": "heredity"}},
                        {"question": "Is the habitat of Bainvillea spina terrestrial?", "answers": ["Yes"], "form": "boolean",
                         "type": "Knowledge Understanding", "meta_data": {"difference": "heredity"}},
                        {"question": "Does Bainvillea spina fly?", "answers": ["I don't know"], "form": "boolean",
                         "type": "Knowledge Differentiation", "meta_data": {}},
                        {"question": "What does Bainvillea spina eat?", "answers": ["Cleridae"], "form": "fill-in-blank",
                         "type": "Knowledge Understanding", "meta_data": {}}],
                 "-6": [{"question": "Is Zorbix big?", "answers": ["No"], "form": "boolean", "type": "Knowledge Understanding",
                         "meta_data": {}}]}
    (raw / "dataset/id2question.json").write_text(json.dumps(questions))
    return raw


@needs_wordnet
def test_alcuna_ranking_forms(tmp_path) -> None:
    raw = _alcuna_raw(tmp_path)
    ontology = _ontology()
    items, coverage = ad.build_alcuna(raw, ad.WordNetFrames(ontology, WN))
    assert coverage["entities"] == 2 and coverage["mapped"] == 1 and coverage["skipped"]["boolean_not_yes_no"] == 1
    by_set = {i.set: i for i in items}
    mc, yes_no = by_set["alcuna-mc"], by_set["alcuna-bool"]
    assert mc.options == [" marine", " terrestrial", " freshwater", " aerial"] and mc.answer == 1
    assert mc.context == "Question: What is the habitat of Bainvillea spina?\nAnswer:"
    assert yes_no.options == [" Yes", " No"] and yes_no.answer == 0
    term = mc.terms[0]
    assert term.surface == "Bainvillea spina" and term.frame[0] == ("hypernym", "synset:vine.n.01")
    assert "The habitat of Bainvillea spina: terrestrial." in term.definition and "Bainvillea spina eat Cleridae." in term.definition
    assert ad.alcuna_choice("Which?\n0. a\n\n1. b") == ("Which?", ["a", "b"]) and ad.alcuna_choice("No options here") is None


def test_entity_inferences_reversal_lre_bear_popqa(tmp_path) -> None:
    ei = tmp_path / "ei" / "data/entity_inferences"
    ei.mkdir(parents=True)
    row = {"ex_id": "x0", "definition": "Cyclone Niran hit northern <extra_id_0> in 2021.", "def_target": "<extra_id_0> Australia <extra_id_1>",
           "category": "disaster", "qid": "cyclone (Q79602)", "ent_str": "Cyclone Niran", "attribute": "Australia", "label": "Australia",
           "probe_sentences": {"template_0": {"probe_sentence": "Cyclone Niran left damage in <extra_id_0>.",
                                              "labels": ["<extra_id_0> Italy <extra_id_1>", "<extra_id_0> Australia <extra_id_1>"]}}}
    (ei / "disaster_explicit.json").write_text(json.dumps(row) + "\n")
    items, info = ad.build_entity_inferences(tmp_path / "ei")
    assert info["items"] == 1 and items[0].context == "Cyclone Niran left damage in" and items[0].options == [" Italy.", " Australia."]
    assert items[0].answer == 1 and items[0].terms[0].definition == "Cyclone Niran hit northern Australia in 2021."
    assert items[0].meta["type_qid"] == "Q79602" and items[0].terms[0].frame is None

    base = tmp_path / "rev" / "name_description_dataset"
    base.mkdir(parents=True)
    people = [("Ann Zed", "the inventor of the cloud harp."), ("Bo Yarrow", "the first mayor of Moonbase."), ("Cy Wex", "the painter of the Glass Sea.")]
    w = lambda name, rows: (base / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))  # noqa: E731
    w("p2d_prompts_train", [{"prompt": f"{n}, known for being", "completion": f" {d}"} for n, d in people])
    w("d2p_prompts_train", [{"prompt": f"Known as {d},", "completion": f" {n}"} for n, d in people])
    w("p2d_reverse_prompts_test", [{"prompt": f"Famous as {d},", "completion": f" {n}"} for n, d in people])
    w("d2p_prompts_test", [{"prompt": f"Famous as {d},", "completion": f" {n}"} for n, d in people])
    w("p2d_prompts_test", [{"prompt": f"The trailblazer {n} was once", "completion": f" {d}"} for n, d in people])
    w("d2p_reverse_prompts_test", [{"prompt": f"The trailblazer {n} was once", "completion": f" {d}"} for n, d in people])
    items, info = ad.build_reversal(tmp_path / "rev", distractors=2)
    assert info["by_set"] == {"reversal-p2d-reverse": 3, "reversal-d2p-forward": 3, "reversal-p2d-forward": 3, "reversal-d2p-reverse": 3}
    reverse = next(i for i in items if i.set == "reversal-p2d-reverse")
    assert len(reverse.options) == 3 and reverse.options[reverse.answer] == " Ann Zed"
    assert [t[0].surface for t in reverse.option_terms] == [o.strip() for o in reverse.options]
    forward = next(i for i in items if i.set == "reversal-d2p-reverse")
    assert forward.terms[0].surface == "Ann Zed" and forward.terms[0].definition.startswith("Known as")
    again, _ = ad.build_reversal(tmp_path / "rev", distractors=2)
    assert [i.to_json() for i in again] == [i.to_json() for i in items]                         # seeded

    lre = tmp_path / "lre" / "data"
    for kind, names in src.LRE_RELATIONS.items():
        (lre / kind).mkdir(parents=True, exist_ok=True)
        for name in names:
            (lre / kind / f"{name}.json").write_text(json.dumps({
                "prompt_templates": ["The capital of {} is"], "properties": {"domain_name": "country", "range_name": "city"},
                "samples": [{"subject": "France", "object": "Paris"}, {"subject": "Peru", "object": "Lima"},
                            {"subject": "Chile", "object": "Santiago"}]}))
    items, info = ad.build_lre(tmp_path / "lre", distractors=2)
    first = items[0]
    assert info["relations"] == 47 and first.context == "The capital of France is" and first.options[first.answer] == " Paris"
    assert not first.terms[0].insert

    import pandas as pd
    (tmp_path / "bear" / "BEAR").mkdir(parents=True)
    pd.DataFrame([{"composite_id": "P36/0/0", "relation": "P36", "item": 0, "template_index": 0, "template": "The capital of [X] is [Y].",
                   "subject": "France", "answer_options": ["Paris", "Rome"], "correct": 0, "text_options": []},
                  {"composite_id": "P36/0/1", "relation": "P36", "item": 0, "template_index": 1, "template": "[Y] is the capital of [X].",
                   "subject": "France", "answer_options": ["Paris", "Rome"], "correct": 0, "text_options": []}]
                 ).to_parquet(tmp_path / "bear" / "BEAR" / "test-00000-of-00001.parquet")
    items, _ = ad.build_bear(tmp_path / "bear")
    assert items[0].context == "The capital of France is" and items[0].options == [" Paris.", " Rome."]
    assert items[1].context == "" and items[1].options == ["Paris is the capital of France.", "Rome is the capital of France."]

    (tmp_path / "popqa").mkdir()
    pd.DataFrame([{"id": i, "subj": s, "prop": "capital", "obj": o, "subj_id": i, "prop_id": 1, "obj_id": i, "s_aliases": "[]",
                   "o_aliases": "[]", "s_uri": f"http://www.wikidata.org/entity/Q{i}", "o_uri": f"http://www.wikidata.org/entity/Q{9 + i}",
                   "s_wiki_title": s, "o_wiki_title": o, "s_pop": 10 * i, "o_pop": 5, "question": f"What is the capital of {s}?",
                   "possible_answers": json.dumps([o])} for i, (s, o) in enumerate([("France", "Paris"), ("Peru", "Lima"), ("Chile", "Santiago")])]
                 ).to_csv(tmp_path / "popqa" / "test.tsv", sep="\t", index=False)
    items, _ = ad.build_popqa(tmp_path / "popqa", distractors=2)
    assert items[0].context == "Q: What is the capital of France? A:" and items[0].options[items[0].answer] == " Paris"
    assert items[0].terms[0].concept == "Q0" and not items[0].terms[0].insert and items[0].meta["s_pop"] == 0


def test_sources_fetch_records_digests_without_network(tmp_path, monkeypatch) -> None:
    source = src.Source("toy", "Toy", "cite", "MIT", "LICENSE", "abc", {"a.txt": "https://example.invalid/a", "d/b.txt": "https://example.invalid/b"})
    monkeypatch.setitem(src.SOURCES, "toy", source)

    def fake(url: str, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(url)
        return len(url)
    monkeypatch.setattr(src, "_download", fake)
    info = src.fetch("toy", tmp_path)
    assert [f["path"] for f in info["files"]] == ["raw/a.txt", "raw/d/b.txt"] and info["licence"] == "MIT"
    assert src.verify("toy", tmp_path) == []
    (tmp_path / "toy/raw/a.txt").write_text("changed")
    assert src.verify("toy", tmp_path) == ["raw/a.txt"]
    with pytest.raises(FileNotFoundError):
        src.raw_dir("absent", tmp_path)


@needs_wordnet
def test_build_writes_manifest_and_commit_copy(tmp_path, monkeypatch) -> None:
    root = tmp_path / "benchmarks"
    raw = _comps_raw(root / "comps")
    (root / "comps" / "source.json").write_text(json.dumps({"files": [{"path": "raw/x", "sha256": "0" * 64, "bytes": 1}]}))
    ontology_path = tmp_path / "ontology.pt"
    torch.save(_ontology(), ontology_path)
    frames = ad.WordNetFrames(_ontology(), WN)
    manifest = ad.build("comps-wugs", tmp_path / "out", ontology=ontology_path, root=root, commit_dir=tmp_path / "commit", frames=frames)
    assert manifest["counts"]["items"] == 2 and manifest["source"]["licence"] == "Apache-2.0" and manifest["committed_items"]
    assert (tmp_path / "commit" / "comps-wugs-wordnet-v1" / "items.jsonl.gz").exists()
    items = rk.load_items(tmp_path / "out" / "items.jsonl.gz")
    assert len(items) == 2 and raw.exists()
    with gzip.open(tmp_path / "out" / "items.jsonl.gz", "rt") as handle:
        assert json.loads(handle.readline())["id"] == "comps-wugs-1"
    with pytest.raises(FileExistsError):
        ad.build("comps-wugs", tmp_path / "out", ontology=ontology_path, root=root, frames=frames)


def test_write_bench_plan_and_contrasts(tmp_path) -> None:
    from vsa_embed.benchmarks import write_bench as wb
    smoke = {f"{s}|{m}": {"padded_tokens_per_item": 100.0, "cpu_seconds_per_item": 0.01} for s in wb.SETS for m in wb.MODELS}
    jobs = wb.plan_jobs(smoke=smoke, item_counts={"comps-wugs": 1000, "alcuna": 500})
    gpu = [j for j in jobs if j["lane"] == "gpu"]
    assert len(gpu) == 16 and all(j["priority"] == 54.498 for j in gpu) and all(j["hours"] > 0 for j in gpu)
    report = jobs[-1]
    assert report["priority"] == 54.4981 and "-report" in report["name"] and report["lane"] == "cpu"
    assert len({j["name"] for j in jobs}) == len(jobs)
    c5 = next(j for j in gpu if j["name"] == "tk-bench-comps-wugs-SmolLM2-360M-C5-s1")
    assert "store:linker" in c5["command"][c5["command"].index("--conditions") + 1]
    assert c5["command"][c5["command"].index("--output") + 1].endswith("SmolLM2-360M-full-C5-s1/toolkit-bench-comps-wugs")
    text = wb.render_commands(jobs)
    assert "write_bench queue" in text and "54.498" in text and "GPU-h" in text
    spec = wb.contrast_spec()
    assert [c["name"] for c in spec["primary"]] == ["W1", "W2"]
    labels = {rk.model_label({"experiment": f"e9-wordnet-{h}-{'frozen' if m == 'P0' else 'full'}-{m}-s1"})
              for h in wb.HOSTS for m in wb.MODELS}
    used = {c[k]["model"] for fam in spec.values() for c in fam for k in ("a", "b")}
    assert used <= labels
    conditions = {c[k]["condition"] for fam in spec.values() for c in fam for k in ("a", "b")}
    assert conditions <= {c for m in wb.MODELS for c in wb.CONDITIONS[m]}
    added = wb.queue(jobs, queue_dir=tmp_path / "q")
    assert len(added) == len(jobs) and wb.queue(jobs, queue_dir=tmp_path / "q") == []
    stored = json.loads((tmp_path / "q" / f"{report['name']}.json").read_text())
    assert stored["priority"] == 54.4981 and stored["lane"] == "cpu"
