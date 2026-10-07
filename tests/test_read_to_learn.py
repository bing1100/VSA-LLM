"""E11 read-to-learn, model-free parts: definition writers, the concept finder, relation typing, the readers (with fake
scorers / generators / teachers) and frame metrics."""

import random

import numpy as np
import pytest

from vsa_embed import read_to_learn as rtl

RELATIONS = ["is_a", "area", "owned_by", "depends_on", "uses", "status"]
ATOMS = ["type:tool", "type:system", "area:finance", "area:sales", "term:Brix Squad", "term:Flam Platform", "term:Kord Hub",
         "status:active", "status:deprecated"]
R = {n: i for i, n in enumerate(RELATIONS)}
A = {n: i for i, n in enumerate(ATOMS)}
TYPES = {"term:Brix Squad": "term:team", "term:Flam Platform": "term:system", "term:Kord Hub": "term:system"}
TEXTS = {"type:tool": "tool", "type:system": "system", "area:finance": "finance", "area:sales": "sales",
         "term:Brix Squad": "the Brix Squad", "term:Flam Platform": "Flam Platform", "term:Kord Hub": "Kord Hub",
         "status:active": "active", "status:deprecated": "deprecated"}


def _ontology() -> dict:
    frames = [
        [("is_a", "type:tool"), ("area", "area:finance"), ("owned_by", "term:Brix Squad"), ("depends_on", "term:Flam Platform")],
        [("is_a", "type:system"), ("area", "area:sales"), ("owned_by", "term:Brix Squad"), ("uses", "term:Kord Hub"),
         ("status", "status:active")],
        [("is_a", "type:tool"), ("area", "area:sales"), ("owned_by", "term:Brix Squad"), ("depends_on", "term:Flam Platform"),
         ("depends_on", "term:Kord Hub"), ("status", "status:deprecated")],
        [("is_a", "type:system"), ("area", "area:finance"), ("owned_by", "term:Brix Squad"), ("uses", "term:Flam Platform")],
    ]
    offsets, relations, fillers = [0], [], []
    for frame in frames:
        relations += [R[r] for r, _ in frame]; fillers += [A[a] for _, a in frame]
        offsets.append(len(relations))
    return {"relation_names": RELATIONS, "atomic_names": ATOMS, "offsets": offsets, "relations": relations, "fillers": fillers,
            "concept_names": ["Brix Squad", "Flam Platform", "Kord Hub", "Zed Tool"], "heldout_entries": []}


def _typing() -> rtl.RelationTyping:
    return rtl.RelationTyping(_ontology(), lambda atom: TYPES.get(atom, atom.partition(":")[0]))


def _lexicon() -> rtl.FillerLexicon:
    return rtl.build_filler_lexicon(_ontology(), None, TEXTS.get)


def test_strip_markup_and_headword_span() -> None:
    raw = "The (<i>R</i>)-enantiomer of lipoic acid. A C<small><sub>8</sub></small> thia fatty acid&#160;with anti-oxidant properties.It is"
    assert rtl.strip_markup(raw) == "The (R)-enantiomer of lipoic acid. A C8 thia fatty acid with anti-oxidant properties. It is"
    assert rtl.headword_span("A note: Zork tool is new.", "zork tool") == (8, 17)
    assert rtl.headword_span("Zorktool", "zork") is None


def test_t5_definitions_state_every_fact_and_prose_avoids_templates() -> None:
    from vsa_embed.benchmarks.glossary import FACT_TEMPLATES
    from vsa_embed.tracks.glossary import TEMPLATES
    facts = {"is_a": ["tool"], "area": ["finance"], "purpose": ["plans headcount"], "status": ["deprecated"], "tier": ["tier 3"],
             "owned_by": ["the Bairksai Squad"], "depends_on": ["Flamstoulk Platform", "Kord Hub"], "cadence": ["weekly"]}
    for style in rtl.T5_STYLES:
        text = rtl.t5_definition("steindsox", facts, style, random.Random(0))
        assert text.lower().startswith("steindsox") or text.startswith("In our organisation, steindsox")
        for values in facts.values():
            assert all(v in text for v in values), (style, values, text)
    every = {**facts, **{r: [f"the Thing {i}"] for i, r in enumerate(rtl.T5_ORDER) if r not in facts}}
    prose = rtl.t5_definition("steindsox", every, "prose", random.Random(1))
    assert all(f"the Thing {i}".lower() in prose.lower() for i, r in enumerate(rtl.T5_ORDER) if r not in facts)
    templates = {t for spec in TEMPLATES.values() for t in spec.prompts}
    assert rtl.template_overlaps(prose, "steindsox", templates) == []
    # The training fact templates' "<subject> <verb phrase>" (e.g. "steindsox is owned by") never occur either.
    verbs = {t.split("{s}", 1)[1].split("{", 1)[0] for ts in FACT_TEMPLATES.values() for t in ts if t.startswith("{s}")}
    assert not [v for v in verbs if len(v.strip()) > 4 and f"steindsox{v}".lower() in prose.lower()]
    glossary = rtl.t5_definition("steindsox", facts, "glossary", random.Random(0))
    assert glossary.startswith("steindsox: a finance tool that plans headcount, owned by the Bairksai Squad.")
    with pytest.raises(ValueError):
        rtl.t5_definition("x", facts, "poem", random.Random(0))


def test_t4_definitions() -> None:
    facts = {"is_a": ["benzoic acids", "phenols"], "has_functional_parent": ["benzoic acid"], "has_role": ["antioxidant"],
             "is_conjugate_acid_of": ["4-hydroxybenzoate"], "contains_element": ["C"]}
    chebi = rtl.t4_definition("4-zork acid", facts, "chebi", random.Random(0))
    assert chebi.startswith("4-zork acid: A member of the class of benzoic acids that is functionally related to benzoic acid.")
    assert "It has a role as an antioxidant." in chebi and "It is a conjugate acid of 4-hydroxybenzoate." in chebi
    assert "C." not in chebi.split("benzoate.")[-1]                   # elements are never stated
    prose = rtl.t4_definition("4-zork acid", facts, "prose", random.Random(0))
    assert "member of the class" not in prose and "antioxidant" in prose and "4-hydroxybenzoate" in prose


def test_filler_lexicon_longest_match_headword_and_resolve() -> None:
    lexicon = _lexicon()
    text = "Zed Tool: a tool owned by the Brix Squad that depends on Flam Platform and Kord Hub; Zed Tool is active."
    mentions = lexicon.find(text, [(0, 8), (86, 94)])
    surfaces = [m.surface for m in mentions]
    assert surfaces == ["tool", "the Brix Squad", "Flam Platform", "Kord Hub", "active"]
    assert mentions[1].atoms == (A["term:Brix Squad"],)
    assert lexicon.resolve("Brix Squad") == (A["term:Brix Squad"],) and lexicon.resolve("the brix squad.") == (A["term:Brix Squad"],)
    assert lexicon.resolve("a nice Kord Hub thing") == (A["term:Kord Hub"],) and lexicon.resolve("nothing here") == ()


def test_relation_typing_candidates_prior_functional() -> None:
    typing = _typing()
    assert typing.candidates(A["term:Kord Hub"]) == typing.candidates(A["term:Flam Platform"])
    assert set(typing.candidates(A["term:Kord Hub"])) == {R["depends_on"], R["uses"]}           # type level (term:system)
    assert typing.candidates(A["term:Brix Squad"]) == [R["owned_by"]]
    assert typing.prior(A["term:Flam Platform"]) == R["depends_on"]                              # 2 × depends_on, 1 × uses
    assert typing.prior(A["term:Kord Hub"]) in {R["depends_on"], R["uses"]}
    assert R["is_a"] in typing.functional and R["owned_by"] in typing.functional and R["depends_on"] not in typing.functional
    assert rtl.functional_dedupe([(R["is_a"], 0, 0.1), (R["is_a"], 1, 0.5), (R["depends_on"], 5, 0.2), (R["depends_on"], 6, 0.3)],
                                 typing.functional) == [(R["is_a"], 1), (R["depends_on"], 6), (R["depends_on"], 5)]


def _task(text: str, gold: list[tuple[str, str]], headword: str = "Zed Tool") -> rtl.ReadTask:
    return rtl.ReadTask("c1", "prose", text, headword, [(R[r], A[a]) for r, a in gold],
                        random_frame=[(R["is_a"], A["type:system"])])


def test_static_readers_and_metrics() -> None:
    typing, lexicon = _typing(), _lexicon()
    gold = [("is_a", "type:tool"), ("owned_by", "term:Brix Squad"), ("uses", "term:Kord Hub"), ("area", "area:sales")]
    task = _task("Zed Tool is a tool. The Brix Squad looks after Zed Tool, whose work leans on Kord Hub.", gold)
    mentions = rtl.mentions_of(task, lexicon)
    assert [m.surface for m in mentions] == ["tool", "The Brix Squad", "Kord Hub"]
    assert rtl.read_oracle(task).frame == task.gold and rtl.read_none(task).frame is None
    assert rtl.read_random(task).frame == [(R["is_a"], A["type:system"])]
    stated = rtl.read_stated(task, mentions).frame
    assert set(stated) == {(R["is_a"], A["type:tool"]), (R["owned_by"], A["term:Brix Squad"]), (R["uses"], A["term:Kord Hub"])}
    prior = rtl.read_typeprior(task, mentions, typing).frame
    assert (R["is_a"], A["type:tool"]) in prior and (R["owned_by"], A["term:Brix Squad"]) in prior
    found = {a for m in mentions for a in m.atoms}
    metrics = rtl.frame_metrics(stated, task.gold, found_atoms=found, ambiguous_atoms={A["term:Kord Hub"]})
    assert metrics["precision"] == 1.0 and metrics["recall"] == 0.75 and metrics["filler_recall"] == 0.75
    assert metrics["stated_recall"] == 1.0 and metrics["ambiguous_edges"] == 1 and metrics["ambiguous_relation_hits"] == 1
    empty = rtl.frame_metrics(None, task.gold)
    assert empty["edges"] == 0 and empty["recall"] == 0.0 and empty["precision"] is None


def test_pattern_reader_uses_hearst_and_relation_name_cues() -> None:
    typing, lexicon = _typing(), _lexicon()
    task = _task("Zed Tool is a kind of tool. It is owned by the Brix Squad and depends on Flam Platform. Kord Hub is nearby.",
                 [("is_a", "type:tool")])
    frame = rtl.read_pattern(task, rtl.mentions_of(task, lexicon), typing, lexicon).frame
    assert set(frame) == {(R["is_a"], A["type:tool"]), (R["owned_by"], A["term:Brix Squad"]), (R["depends_on"], A["term:Flam Platform"])}
    assert rtl.cue_phrase("has_functional_parent") == "functional parent" and rtl.cue_phrase("is_a") is None
    assert rtl.cue_phrase("is_conjugate_acid_of") == "conjugate acid of"


def test_linker_reader_chooses_by_gain_and_self_tests() -> None:
    typing, lexicon = _typing(), _lexicon()
    task = _task("Zed Tool: a tool; the Brix Squad; Kord Hub; active.", [])
    mentions = rtl.mentions_of(task, lexicon)
    candidates = rtl.linker_candidates(task, mentions, typing)
    assert len(candidates) == 1 + 1 + 2 + 1                      # tool: is_a; squad: owned_by; Kord Hub: depends_on/uses; active: status
    seen_variants = []

    def scorer(tasks, variants):
        seen_variants.extend(variants)
        gains = {(R["is_a"], A["type:tool"]): 2.0, (R["owned_by"], A["term:Brix Squad"]): 0.5,
                 (R["depends_on"], A["term:Kord Hub"]): 0.2, (R["uses"], A["term:Kord Hub"]): 1.0,
                 (R["status"], A["status:active"]): -0.3}
        return [np.asarray([-10.0] + [-10.0 + gains[v[0]] for v in vs[1:]]) for vs in variants]

    kept, every = rtl.read_linker([task], [mentions], typing, scorer)
    assert seen_variants[0][0] is None and len(seen_variants[0]) == len(candidates) + 1
    assert set(kept[0].frame) == {(R["is_a"], A["type:tool"]), (R["owned_by"], A["term:Brix Squad"]), (R["uses"], A["term:Kord Hub"])}
    assert (R["status"], A["status:active"]) in every[0].frame and (R["status"], A["status:active"]) not in kept[0].frame
    assert kept[0].cost["forward_passes"] == len(candidates) + 1 and kept[0].reader == "linker" and every[0].reader == "linker-all"
    nothing = _task("Zed Tool has no known fillers here.", [])
    kept, _ = rtl.read_linker([nothing], [rtl.mentions_of(nothing, lexicon)], typing, scorer)
    assert kept[0].frame is None


def test_host_reader_parses_resolves_and_drops_ill_typed_edges() -> None:
    typing, lexicon = _typing(), _lexicon()
    task = _task("Zed Tool: a tool owned by the Brix Squad.", [])
    prompts_seen = []

    def generate(prompts):
        prompts_seen.extend(prompts)
        return (["is_a: tool\nowned_by: Brix Squad\nuses: finance\narea: moon\n\nText: next", ], 50, 12)

    demos = lambda t: [{"context": "Kord Hub: a system.", "surface": "Kord Hub", "edges": [("is_a", "system")]}]
    result = rtl.read_host([task], lexicon, typing, generate, demos)[0]
    assert "Concept: Zed Tool" in prompts_seen[0] and "Concept: Kord Hub\nis_a: system" in prompts_seen[0]
    assert set(result.frame) == {(R["is_a"], A["type:tool"]), (R["owned_by"], A["term:Brix Squad"])}
    assert result.details["ill_typed"] == 1 and result.details["unresolved"] == 1 and result.cost["generated_tokens"] == 12


def test_teacher_reader_with_a_fake_runner(tmp_path) -> None:
    from vsa_embed.authoring_baselines import TeacherAuthor
    typing, lexicon = _typing(), _lexicon()
    task = _task("Zed Tool: a tool owned by the Brix Squad.", [])

    def runner(prompt, schema, model):
        assert "Zed Tool" in prompt and model == "claude-opus-5-5"
        return {"structured_output": {"concepts": [{"id": "c1|prose", "edges": [{"relation": "is_a", "filler": "tool"},
                                                                                {"relation": "owned_by", "filler": "the Brix Squad"}]}]},
                "total_cost_usd": 0.04, "modelUsage": {"claude-opus-5-5": {}}}

    teacher = TeacherAuthor(tmp_path, RELATIONS, {r: r for r in RELATIONS}, runner=runner)
    result = rtl.read_teacher([task], lexicon, typing, teacher)[0]
    assert set(result.frame) == {(R["is_a"], A["type:tool"]), (R["owned_by"], A["term:Brix Squad"])}
    assert result.cost["usd"] == pytest.approx(0.04)
    again = rtl.read_teacher([task], lexicon, typing, teacher)[0]               # cached: no new spend
    assert again.frame == result.frame and again.cost["usd"] == 0.0
