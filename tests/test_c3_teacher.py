import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nltk.corpus").wordnet.ensure_loaded()

from vsa_embed.experiments import c3_teacher as ct  # noqa: E402

RELATIONS = ["hypernym", "instance_hypernym", "part_meronym", "member_meronym", "substance_meronym", "part_holonym",
             "member_holonym", "substance_holonym", "attribute", "similar_to", "topic_domain", "entailment", "cause",
             "antonym", "lexname", "pos"]
ATOMS = ["synset:canine.n.02", "synset:carnivore.n.01", "synset:dog.n.01", "synset:domestic_animal.n.01",
         "synset:large.a.01", "lexname:noun.Tops", "lexname:noun.animal", "pos:a", "pos:n", "pos:r", "pos:s", "pos:v"]
A = {name: i for i, name in enumerate(ATOMS)}
R = {name: i for i, name in enumerate(RELATIONS)}


@pytest.fixture(scope="module")
def mapper() -> ct.FillerMapper:
    return ct.FillerMapper(ATOMS, RELATIONS)


def test_synset_fillers_map_exactly_or_to_the_nearest_dictionary_hypernym(mapper) -> None:
    assert mapper.map("hypernym", "canine.n.02", own="puppy.n.01") == (A["synset:canine.n.02"], "exact")
    assert mapper.map("hypernym", "Domestic_Dog.n.01", own="puppy.n.01") == (A["synset:dog.n.01"], "exact")   # a synonym lemma
    assert mapper.map("hypernym", "wolf.n.01", own="puppy.n.01") == (A["synset:canine.n.02"], "ancestor")      # the builder's rule
    assert mapper.map("hypernym", "`canine.n.2`", own="puppy.n.01") == (A["synset:canine.n.02"], "exact")


def test_wrong_sense_numbers_and_bare_words_fall_back_to_the_word(mapper) -> None:
    assert mapper.map("hypernym", "canine.n.09", own="puppy.n.01") == (A["synset:canine.n.02"], "lemma")   # n.01 is a tooth
    assert mapper.map("hypernym", "carnivore", own="puppy.n.01") == (A["synset:carnivore.n.01"], "lemma")
    assert mapper.map("hypernym", "wolves", own="puppy.n.01") == (A["synset:canine.n.02"], "lemma_ancestor")
    assert mapper.map("hypernym", "zzqxv.n.01", own="puppy.n.01") == (None, "unknown_lemma")
    assert mapper.map("similar_to", "large", own="big.s.01") == (A["synset:large.a.01"], "lemma")


def test_closed_vocabulary_relations_and_unmapped_reasons(mapper) -> None:
    assert mapper.map("lexname", "noun.tops", own="entity.n.01") == (A["lexname:noun.Tops"], "lexname")
    assert mapper.map("lexname", "noun.food", own="dog.n.01") == (None, "bad_lexname")
    assert mapper.map("pos", "noun", own="dog.n.01") == (A["pos:n"], "pos")
    assert mapper.map("pos", "adjective (satellite)", own="big.s.01") == (A["pos:s"], "pos")
    assert mapper.map("pos", "x", own="dog.n.01") == (None, "bad_pos")
    assert mapper.map("hypernym", "noun.animal", own="dog.n.01") == (None, "ill_typed")
    assert mapper.map("hypernym", "dog.n.01", own="dog.n.01") == (None, "self")
    assert mapper.map("owner", "dog.n.01", own="puppy.n.01") == (None, "unknown_relation")


def test_frame_keeps_builder_order_one_lexname_and_pos_and_the_cap(mapper) -> None:
    edges = [("hypernym", "canine.n.02"), ("pos", "n"), ("lexname", "noun.animal"), ("lexname", "noun.Tops"),
             ("hypernym", "domestic_animal.n.01"), ("hypernym", "canine.n.02"), ("antonym", "zzqxv")]
    frame, records = mapper.frame("dog.n.01", edges)
    assert frame == [(R["lexname"], A["lexname:noun.animal"]), (R["pos"], A["pos:n"]), (R["hypernym"], A["synset:canine.n.02"]),
                     (R["hypernym"], A["synset:domestic_animal.n.01"])]
    assert [r["how"] for r in records][-1] == "unknown_lemma" and len(records) == len(edges)
    capped, _ = mapper.frame("dog.n.01", edges, max_degree=3)
    assert len(capped) == 3


def test_answers_of_reports_missing_items_and_deduplicates() -> None:
    items = [{"id": "s1"}, {"id": "s2"}]
    record = {"verdict": {"concepts": [{"id": "s1", "edges": [{"relation": "hypernym", "filler": " canine.n.02 "},
                                                               {"relation": "hypernym", "filler": "canine.n.02"},
                                                               {"relation": "pos", "filler": ""}]}]}}
    out = ct.answers_of(record, items)
    assert out["s1"] == {"edges": [("hypernym", "canine.n.02")], "error": None}
    assert out["s2"]["error"] == "missing from the answer"
    assert ct.answers_of({"error": "boom"}, items)["s1"] == {"edges": [], "error": "boom"}


def _fake_runner(cost: float, calls: list):
    def runner(prompt, schema, model, **options):
        calls.append(options)
        ids = re.findall(r"^(s\d+) \|", prompt, flags=re.M)
        concepts = [{"id": i, "edges": [{"relation": "hypernym", "filler": "canine.n.02"}, {"relation": "pos", "filler": "n"}]}
                    for i in ids]
        return {"structured_output": {"concepts": concepts}, "total_cost_usd": cost, "modelUsage": {model: {}},
                "usage": {"output_tokens": 10}}
    return runner


def test_teach_resumes_from_the_store_and_never_passes_the_cap(tmp_path: Path) -> None:
    synsets = ["dog.n.01", "puppy.n.01", "wolf.n.01", "cat.n.01", "lion.n.01", "fox.n.01", "bear.n.01"]
    calls: list = []
    kwargs = dict(store=tmp_path, batch=2, workers=2, relations=RELATIONS, lexnames=["noun.animal"], log=lambda _: None)
    result = ct.teach(synsets, max_usd=0.25, runner=_fake_runner(0.05, calls), **kwargs)
    # Reservations: 0.30 up front is above the cap, so nothing may start.
    assert result["batches"] == 0 and result["stopped"] and not calls
    result = ct.teach(synsets[:4], max_usd=1.0, runner=_fake_runner(0.05, calls), **kwargs)
    assert result["answered"] == 4 and result["spent_usd_total"] == pytest.approx(0.10)
    assert all(c["max_budget_usd"] > 0 and c["effort"] == "medium" and c["system_prompt"] for c in calls)
    before = len(calls)
    result = ct.teach(synsets, max_usd=1.0, runner=_fake_runner(0.05, calls), **kwargs)
    assert result["todo"] == 3 and result["answered"] == 3 and len(calls) == before + 2
    answers = ct.load_answers(tmp_path)
    assert set(answers) == set(synsets) and answers["wolf.n.01"]["edges"] == [["hypernym", "canine.n.02"], ["pos", "n"]]
    ledger = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert sum(r["usd"] for r in ledger) == pytest.approx(0.20)
    # The cap is cumulative over the ledger: a new teacher starts from the recorded spend.
    teacher = ct.CappedTeacher(tmp_path / "calls", RELATIONS, {}, lexnames=["noun.animal"], max_usd=0.3, ledger=tmp_path / "ledger.jsonl",
                               runner=_fake_runner(0.05, calls))
    assert teacher.spent_usd == pytest.approx(0.20)
    with pytest.raises(ct.BudgetReached):
        teacher.request([{"id": "s1", "surface": "x", "pos": "noun", "contexts": ["x"]}])


def test_failed_calls_are_charged(tmp_path: Path) -> None:
    def failing(prompt, schema, model, **options):
        return {"is_error": True, "subtype": "error_max_budget_usd", "total_cost_usd": 0.07}

    def silent(prompt, schema, model, **options):
        raise RuntimeError("no envelope")

    teacher = ct.CappedTeacher(tmp_path / "a", RELATIONS, {}, lexnames=["noun.animal"], runner=failing, retries=1,
                               ledger=tmp_path / "a.jsonl")
    record = teacher.request([{"id": "s1", "surface": "x", "pos": "noun", "contexts": ["x"]}])
    assert "error" in record and teacher.spent_usd == pytest.approx(0.14) and not list((tmp_path / "a").glob("*.json"))
    teacher = ct.CappedTeacher(tmp_path / "b", RELATIONS, {}, lexnames=["noun.animal"], runner=silent, retries=0, reserve_usd=0.2)
    teacher.request([{"id": "s1", "surface": "x", "pos": "noun", "contexts": ["x"]}])
    assert teacher.spent_usd == pytest.approx(0.2)              # unknown cost: the whole reservation


def _toy_ontology() -> dict:
    from vsa_embed.compose import FrameSchedule
    frames = [[(R["lexname"], A["lexname:noun.animal"]), (R["pos"], A["pos:n"]), (R["hypernym"], A["synset:carnivore.n.01"])],
              [(R["pos"], A["pos:n"])], [(R["pos"], A["pos:a"])]]
    schedule = FrameSchedule.from_frames(frames)
    return {"entry_count": 3, "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
            "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": [2],
            "train_frequency": [5, 0, 0], "relation_names": RELATIONS, "atomic_names": ATOMS,
            "entry_concepts": [(0, 1), (1,), (2,)], "concept_names": ["dog.n.01", "puppy.n.01", "large.a.01"]}


def test_teacher_ontology_replaces_only_the_chosen_entries_frames() -> None:
    onto = _toy_ontology()
    frames = {"dog.n.01": [(R["pos"], A["pos:n"]), (R["hypernym"], A["synset:canine.n.02"])],
              "puppy.n.01": [(R["pos"], A["pos:n"]), (R["hypernym"], A["synset:dog.n.01"])]}
    assert ct.union_frame(onto, 0, frames) == [(R["pos"], A["pos:n"]), (R["hypernym"], A["synset:canine.n.02"]),
                                               (R["hypernym"], A["synset:dog.n.01"])]
    out = ct.teacher_ontology(onto, frames, entries=[0, 1], provenance={"scope": "test"})
    assert ct.entry_frame(out, 0) == ct.union_frame(onto, 0, frames)
    assert ct.entry_frame(out, 2) == ct.entry_frame(onto, 2)            # WordNet frame kept
    assert out["heldout_entries"] == onto["heldout_entries"] and out["atomic_names"] is onto["atomic_names"]
    assert out["teacher"]["teacher_entries"] == 2
    with pytest.raises(ValueError, match="unanswered"):
        ct.teacher_ontology(onto, frames, entries=None, provenance={})


def test_synset_frames_fall_back_to_the_shown_part_of_speech(mapper) -> None:
    answers = {"dog.n.01": {"edges": [["hypernym", "zzqxv"]], "error": None, "pos": "n"},
               "puppy.n.01": {"edges": [["hypernym", "canine.n.02"]], "error": None, "pos": "n"},
               "large.a.01": {"edges": [], "error": "missing from the answer", "pos": "a"}}
    frames, stats = ct.synset_frames(answers, mapper, ["dog.n.01", "puppy.n.01", "large.a.01"])
    assert frames["dog.n.01"] == [(R["pos"], A["pos:n"])] and stats["pos_fallback"] == 1
    assert stats["unanswered"] == 1 and stats["parse_rate"] == pytest.approx(0.5) and "large.a.01" not in frames


def test_agreement_counts_edges_and_fillers() -> None:
    pairs = [([(0, 1), (0, 2), (14, 5)], [(0, 1), (14, 5), (13, 3)]), ([(0, 4)], [(2, 4)])]
    a = ct.agreement(pairs)
    assert a["precision"] == pytest.approx(2 / 4) and a["recall"] == pytest.approx(2 / 4)
    assert a["filler_precision"] == pytest.approx(3 / 4)
    b = ct.agreement(pairs, skip={14})
    assert b["precision"] == pytest.approx(1 / 3) and b["teacher_edges_per_entry"] == pytest.approx(1.5)


def test_sampling_and_hybrid_sets_are_stratified_and_disjoint() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    held = sorted(rng.choice(n, size=200, replace=False).tolist())
    onto = {"entry_count": n, "heldout_entries": held, "train_frequency": rng.integers(0, 3000, size=n).tolist(),
            "entry_concepts": [tuple(range(int(k))) for k in rng.integers(1, 6, size=n)]}
    counts = rng.integers(0, 60, size=n)
    sample = ct.pilot_sample(onto, counts, per_group=50)
    assert len(sample) == 100 and {s["group"] for s in sample} == {"heldout", "trained"}
    assert all((s["entry"] in held) == (s["group"] == "heldout") for s in sample)
    assert sample == ct.pilot_sample(onto, counts, per_group=50)
    groups = ct.hybrid_entries(onto, counts)
    assert groups["heldout"] == held and len(groups["trained"]) == len(held) and not set(groups["trained"]) & set(held)
    assert all(onto["train_frequency"][e] > 0 for e in groups["trained"])


def test_prompt_shows_word_part_of_speech_and_gloss_but_not_the_synset() -> None:
    wn = ct.wordnet()
    item = ct.sense_item(wn.synset("dog.n.01"), "s1")
    prompt = ct.c3_prompt([item], RELATIONS, ["noun.animal", "noun.Tops"])
    assert "s1 | dog | noun | a member of the genus Canis" in prompt and '"the dog barked all night"' in prompt
    assert "dog.n.01" not in prompt and "domestic dog" not in prompt and "noun.animal, noun.Tops" in prompt
