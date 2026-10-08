"""E12 self-query (decision 62): phase A (`e12_self_query`), the faithfulness harness (`e12_faithfulness`) and the stage
report (`e12_report`) on the toy E9 world of `test_e9_binding` (CPU, tiny fake host)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from test_e9_binding import ATOMS, RELATIONS, TERMS, NoWordNet, _frame, _lexicon, _ok, item_dirs, world  # noqa: E402,F401
from vsa_embed.compose import FrameComposer, FrameSchedule  # noqa: E402
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e9_understanding as und  # noqa: E402
from vsa_embed.experiments import e12_faithfulness as faith  # noqa: E402
from vsa_embed.experiments import e12_report as report  # noqa: E402
from vsa_embed.experiments import e12_self_query as sqx  # noqa: E402
from vsa_embed.span_channel import normalize_alias  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")


@pytest.fixture(autouse=True)
def toy_lexicon(monkeypatch):
    monkeypatch.setattr(sqx, "lexicon_for_track", lambda track, family, ontology: _lexicon())
    monkeypatch.setitem(sqx.DEFINITION_STYLE, "toy", "prose")


@pytest.fixture(scope="module")
def stores(world) -> dict:
    return {label: sqx.load_store(world["runs"][model], label) for label, model in
            (("own", "C5"), ("C5ut", "C5ut"), ("C5tr", "C5tr"), ("C5rf", "C5rf"))}


def _open(world, name: str) -> common.E5Run:
    # the fake host has 64 positions: scored contexts are cut to 1 recalled line (`max_lines=1`)
    return common.open_run(world["runs"][name], device="cpu", batch_size=8, max_length=64, alias_table=world["alias"])


def _names(term: str) -> list[list[str]]:
    return [[RELATIONS[r], ATOMS[f]] for r, f in _frame(term)]


@pytest.fixture(scope="module")
def understanding_dir(world, tmp_path_factory) -> Path:
    """A hand-built WP-UB set: one two-hop item (owned_by > area) and one reverse item (owned by the Zash Team)."""
    root = tmp_path_factory.mktemp("und") / "understanding-toy-v1"
    root.mkdir()
    entry = {t: world["table"].alias_to_entry[normalize_alias(t)] for t in TERMS}       # entries are numbered by sorted alias
    concepts = [{"concept": "u-bl", "subset": "seen", "surface": "Brightwater Ledger", "entry": entry["Brightwater Ledger"],
                 "frame": _names("Brightwater Ledger"), "random_frame": None, "source": "Brightwater Ledger", "frequency": 10},
                {"concept": "u-gc", "subset": "seen", "surface": "Grosh Console", "entry": entry["Grosh Console"],
                 "frame": _names("Grosh Console"), "random_frame": None, "source": "Grosh Console", "frequency": 10},
                {"concept": "u-bridge-zash", "subset": "bridge", "surface": "Zash Team", "entry": entry["Zash Team"],
                 "frame": _names("Zash Team"), "random_frame": None, "source": "Zash Team", "frequency": 10, "role": "bridge"}]
    items = [{"id": "u-bl-two_hop-owned_by-area", "family": "two_hop", "test": "two_hop", "subset": "seen", "anchor": "u-bl",
              "slots": {"x": "u-bl"}, "text": {}, "null": {"x": "this"}, "templates": ["{x} is owned by a team in the"],
              "candidates": [" finance area", " sales area"], "gold": 0, "relation": "owned_by>area", "pair": None, "chance": 0.5,
              "meta": {"path": ["owned_by", "area"], "bridge": "Zash Team", "answer": "area:finance"}},
             {"id": "u-bl-reverse-owned_by", "family": "reverse", "test": "reverse", "subset": "seen", "anchor": "u-bl",
              "slots": {"x": "u-bl", "y": "u-gc"}, "text": {"c": "owned by the Zash Team"}, "null": {"c": "we mean"},
              "templates": ["Of {x} and {y}, the one {c} is"], "candidates": [" {x}", " {y}"], "gold": 0, "relation": "owned_by",
              "pair": None, "chance": 0.5, "meta": {"filler": "term:Zash Team", "partner": "u-gc"}}]
    (root / "concepts.jsonl").write_text("".join(json.dumps(c) + "\n" for c in concepts))
    und.write_jsonl_gz(root / "items.jsonl.gz", items)
    (root / "manifest.json").write_text(json.dumps({"schema": und.SCHEMA, "track": "toy", "family": "gpt2"}))
    return root


@pytest.fixture(scope="module")
def new_dir(world, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("new") / "new-words-toy-v1"
    edit.build_new_word_items(world["root"] / "ontology.pt", out, tokenizer_name="gpt2", count=3, distractors=1, wordnet=NoWordNet(),
                              alias_table=world["alias"], lexicon=_lexicon(), min_subtokens=1, reserved_names=TERMS)
    return out


# -- conditions and stores --------------------------------------------------------------------------------------------------

def test_condition_grammar() -> None:
    assert sqx.Condition.parse("recall:own").name == "recall:own" and sqx.Condition.parse("none").store is None
    assert [c.name for c in sqx.parse_conditions("none, symbolic,recall:C5@2")] == ["none", "symbolic", "recall:C5@2"]
    for bad in ("recall", "symbolic:own", "nonsense:own"):
        with pytest.raises(ValueError):
            sqx.Condition.parse(bad)
    with pytest.raises(ValueError):
        sqx.parse_conditions("none,none")


def test_store_resolution_and_defaults(world) -> None:
    c5 = world["runs"]["C5"]
    assert sqx.resolve_store(c5, "own") == c5 and sqx.resolve_store(c5, "C5ut") == world["runs"]["C5ut"]
    assert sqx.resolve_store(world["runs"]["P0"], "C5@1") == c5
    with pytest.raises(FileNotFoundError):
        sqx.resolve_store(c5, "C5@7")
    stage = ["C5", "C5ut", "C5tr", "C0p", "P0"]
    assert sqx.default_conditions("C5", "twins", stage_models=stage) == ["none", "recall:own", "symbolic", "definition", "roleless:own",
                                                                         "wrong:own", "recall:C5ut", "recall:C5tr"]
    assert sqx.default_conditions("C0p", "twins", stage_models=stage)[:3] == ["none", "recall:C5", "symbolic"]
    assert sqx.default_conditions("P0", "twins", stage_models=stage, store_seeds=[1, 2])[:3] == ["none", "recall:C5@1", "recall:C5@2"]
    assert "wrong:own" not in sqx.default_conditions("C5", "understanding", stage_models=stage)
    assert sqx.default_conditions("C5", "twins", stage_models=stage, core=True) == ["none", "recall:own", "symbolic", "roleless:own",
                                                                                    "recall:C5ut", "recall:C5tr"]
    assert sqx.default_conditions("C0p", "twins", stage_models=stage, core=True) == ["none", "recall:C5", "symbolic", "roleless:C5"]


# -- phase A on the twins ----------------------------------------------------------------------------------------------------

def _builder(world, items: Path, stores: dict) -> sqx.ContextBuilder:
    item_set = sqx.load_item_set(items)
    return sqx.ContextBuilder(item_set, world["ontology"], _lexicon(), stores)


def test_twin_contexts_state_roles_only_from_a_bound_store(world, item_dirs, stores) -> None:
    builder = _builder(world, item_dirs["twins"], stores)
    concepts = builder.concepts
    own, records = builder.build(sqx.Condition("recall", "own"))
    a, b = "tw-0000-A", "tw-0000-B"
    first = lambda cid: next(p.id for p in builder.items.prompts if p.concept == cid)
    text_a = own[first(a)]
    assert text_a.splitlines()[0] == f"recall({concepts[a]['surface']}):" and " depends on " in text_a and " uses " in text_a
    assert all(line.startswith("- ") and line.endswith(")") for line in text_a.splitlines()[1:])
    assert {r["concept"] for r in records} == set(concepts) and all(set(r["slots"]) <= set(RELATIONS) for r in records)
    blind, blind_records = builder.build(sqx.Condition("recall", "C5ut"))
    strip = lambda text, cid: text.replace(concepts[cid]["surface"], "X")
    assert "associated with" in blind[first(a)] and strip(blind[first(a)], a) == strip(blind[first(b)], b)
    translation, _ = builder.build(sqx.Condition("recall", "C5tr"))
    assert strip(translation[first(a)], a) == strip(translation[first(b)], b)          # the same bag of fillers and offsets
    symbolic, sym_records = builder.build(sqx.Condition("symbolic"))
    assert "(1.00)" in symbolic[first(a)] and all(v == 1.0 for r in sym_records for v in r["slots"].values())
    wrong, _ = builder.build(sqx.Condition("wrong", "own"))
    assert strip(wrong[first(a)], a) == strip(own[first(b)], b)                      # A reads B's recall under its own name
    roleless, _ = builder.build(sqx.Condition("roleless", "own"))
    body = roleless[first(a)].splitlines()[1:]
    assert len(body) == 1 and "associated with" in body[0] and "(" not in body[0]    # no role, no role-derived confidence
    definition, _ = builder.build(sqx.Condition("definition"))
    assert concepts[a]["surface"] in definition[first(a)]


def test_scoring_prepends_the_context_and_keeps_a_context_free_null(world, item_dirs, stores) -> None:
    run = _open(world, "C5")
    item_set = sqx.load_item_set(item_dirs["twins"], limit=2)
    builder = sqx.ContextBuilder(item_set, run.ontology, _lexicon(), stores, max_lines=1)
    contexts, _ = builder.build(sqx.Condition("recall", "own"))
    prefixes, _ = sqx.prompt_texts(item_set.prompts, contexts)
    assert prefixes[0].startswith("recall(") and "\n" in prefixes[0]
    with sqx.host_view(run, item_set) as (adapter, ids):
        rows, cache, longest = sqx.score_prompts(adapter, item_set.prompts, contexts)
        plain, cache2, _ = sqx.score_prompts(adapter, item_set.prompts, {}, cache)
    assert longest > 10 and all("recall(" not in key[0] for key in cache) and cache2 == cache
    assert len(rows) == len(item_set.prompts) and np.asarray(rows[0]["s"]).shape == (len(item_set.prompts[0].templates), 2)
    assert not np.allclose(rows[0]["s"], plain[0]["s"])                            # the context changes the scores
    pmi_minus_s = np.asarray(rows[0]["s"]) - np.asarray(rows[0]["pmi"])
    assert np.allclose(pmi_minus_s, np.asarray(plain[0]["s"]) - np.asarray(plain[0]["pmi"]))   # the same (context-free) null
    with pytest.raises(ValueError):
        sqx.check_lengths(adapter.tokenizer, ["word " * 40], [" end"], 16)


def test_evaluate_twins_reports_contrast_and_decode_per_condition(world, item_dirs, stores) -> None:
    run = _open(world, "C5")
    item_set = sqx.load_item_set(item_dirs["twins"])
    # (role-blind lines list every filler: too long for the fake host's 64 positions; their text is tested above)
    conditions = sqx.parse_conditions("none,recall:own,symbolic,wrong:own")
    evaluation = sqx.evaluate(run, item_set, conditions, stores, max_lines=1, log=lambda *_: None)
    summary = sqx.summarize(item_set, evaluation["results"], evaluation["records"], evaluation["resolved"])
    assert list(summary["conditions"]) == [c.name for c in conditions]
    choice = {name: block["choice"] for name, block in summary["conditions"].items()}
    assert all(0 <= b["mean"]["contrast"] <= 1 for b in choice.values() if b["mean"]["n"])
    assert choice["symbolic"]["mean"]["decode"] == 1.0 and choice["symbolic"]["mean"]["decode_all"] == 1.0
    assert "decode" not in choice["none"]["mean"]
    assert 0 <= choice["recall:own"]["mean"]["decode"] <= 1 and evaluation["longest_tokens"]["recall:own"] > evaluation["longest_tokens"]["none"]


# -- chained recall and reverse lookup ------------------------------------------------------------------------------------------

def test_chain_and_reverse_contexts(world, understanding_dir, stores) -> None:
    builder = _builder(world, understanding_dir, stores)
    symbolic, records = builder.build(sqx.Condition("symbolic"))
    chain = symbolic["u-bl-two_hop-owned_by-area"].splitlines()
    assert chain[0] == "recall(Brightwater Ledger, owned by):" and chain[2] == "recall(Zash Team, area):"
    assert chain[1] == "- Brightwater Ledger is owned by the Zash Team. (1.00)" and "finance area" in chain[3]
    by_item = {r["item"]: r for r in records}
    assert by_item["u-bl-two_hop-owned_by-area"]["hop2_correct"] == 1.0 and by_item["u-bl-two_hop-owned_by-area"]["bridge_correct"] == 1.0
    lookup = symbolic["u-bl-reverse-owned_by"]
    assert lookup.startswith("lookup(owned by, the Zash Team):") and "Brightwater Ledger is owned by the Zash Team." in lookup
    assert "Grosh Console" not in lookup                                         # the partner is no holder
    own, own_records = builder.build(sqx.Condition("recall", "own"))
    reverse = next(r for r in own_records if r["item"] == "u-bl-reverse-owned_by")
    assert len(own["u-bl-reverse-owned_by"].splitlines()) == 1 + sqx.REVERSE_K and reverse["pair_correct"] in (0.0, 0.5, 1.0)
    assert {"hop1_correct", "bridge_correct", "hop2_correct"} <= set(next(r for r in own_records if r["item"].endswith("area")))
    with pytest.raises(ValueError):
        builder.build(sqx.Condition("wrong", "own"))


def test_evaluate_understanding_and_new_words(world, understanding_dir, new_dir, stores) -> None:
    run = _open(world, "C5")
    # scored: the reverse item (a two-block chain overflows the fake host's 64 positions; its text is tested above)
    item_set = sqx.load_item_set(understanding_dir, families=("reverse",))
    evaluation = sqx.evaluate(run, item_set, sqx.parse_conditions("none,recall:own,symbolic"), stores, max_lines=1, log=lambda *_: None)
    summary = sqx.summarize(item_set, evaluation["results"], evaluation["records"], evaluation["resolved"])
    units = summary["conditions"]["symbolic"]["items"]["units"]
    assert units["u-bl-reverse-owned_by"]["pair_correct"] == 1.0 and "reverse/all" in summary["conditions"]["symbolic"]["items"]["by"]
    assert "pair_correct" in summary["conditions"]["recall:own"]["items"]["units"]["u-bl-reverse-owned_by"]
    definitions, _ = _builder(world, understanding_dir, stores).build(sqx.Condition("definition"))
    assert set(definitions) == {"u-bl-two_hop-owned_by-area"}                     # a lookup has no definition: not scored
    assert "Brightwater Ledger" in definitions["u-bl-two_hop-owned_by-area"] and "Zash Team" in definitions["u-bl-two_hop-owned_by-area"]
    new = sqx.load_item_set(new_dir)
    assert new.kind == "new" and new.prompts
    evaluation = sqx.evaluate(run, new, sqx.parse_conditions("none,recall:own,symbolic"), stores, max_lines=1, log=lambda *_: None)
    summary = sqx.summarize(new, evaluation["results"], evaluation["records"], evaluation["resolved"])
    assert summary["conditions"]["symbolic"]["property"]["mean"]["decode"] == 1.0
    assert summary["conditions"]["none"]["property"]["mean"]["n"] == len(new.prompts)


# -- faithfulness -------------------------------------------------------------------------------------------------------------------

def test_drop_mask_and_decision_24_semantics() -> None:
    offsets = torch.tensor([0, 3, 5])
    assert faith.drop_mask(offsets, [0, 1], remove=1).nonzero().flatten().tolist() == [1, 4]
    assert faith.drop_mask(offsets, [0, 1], remove=2).nonzero().flatten().tolist() == [2]       # entry 1 has only 2 edges
    assert faith.drop_mask(offsets, [0, 1], keep=0).nonzero().flatten().tolist() == [1, 2, 4]
    # static composer: zeroing an edge's weight = composing the frame without the edge (the bundle is normalized)
    torch.manual_seed(0)
    frames = [[(0, 1), (1, 2), (2, 3)], [(0, 1), (2, 3)]]
    composer = FrameComposer(FrameSchedule.from_frames(frames), 6, 3, 32, operator="hrr", mode="bundle")
    with faith.dropped_edges(composer, faith.drop_mask(composer.schedule.offsets, [0], remove=1)):
        dropped = composer.compose(torch.tensor([0]))
    assert torch.allclose(dropped, composer.compose(torch.tensor([1])), atol=1e-6)


def test_text_cache_scores_as_the_standard_scorer(world, item_dirs) -> None:
    from vsa_embed.experiments import e9_binding_items as role_items
    run = _open(world, "C5")
    _, concepts, items = role_items.load_items(item_dirs["natural"])          # existing entries: linked, nothing inserted
    surface = {c["concept"]: c["surface"] for c in concepts}
    pairs = [(und.render(t, {"x": surface[i["concept"]]}), c) for i in items for t in i["templates"] for c in i["candidates"]]
    expected = role_items.continuation_scores(run.adapter, [p for p, _ in pairs], [c for _, c in pairs])
    cache = faith.TextCache(run.adapter)
    got = cache.scores(pairs)
    assert np.allclose(got[0], expected[0], atol=1e-4) and np.array_equal(got[1], expected[1])
    again = cache.scores(pairs[::-1])                                            # reused, in any order
    assert np.allclose(again[0][::-1], got[0], atol=1e-5)


def test_faithfulness_on_new_words_and_twins(world, new_dir, item_dirs, stores) -> None:
    for model, label in (("C5", "own"), ("C5ut", "C5ut")):
        run = _open(world, model)
        loaded = stores[label]
        result = faith.faithfulness_new_words(run, loaded.store, loaded.ontology, new_dir, log=lambda *_: None)
        edges = result["edges"]
        assert edges and all(abs(e["gap"] - (e["control"] - e["delta"])) < 1e-9 for e in edges)
        assert {e["concept"] for e in edges} <= {c["concept"] for c in edit.load_item_dir(new_dir, edit.SCHEMA_NEW)[1]}
        assert any(abs(e["delta"]) > 0 for e in edges)                             # removing an edge moves something
        twins = faith.role_specificity(run, loaded.store, loaded.ontology, item_dirs["twins"], log=lambda *_: None)
        assert twins["rows"] and all(abs(r["rs"] - (r["delta_other"] - r["delta_own"])) < 1e-9 for r in twins["rows"])
        assert set(twins["swap"]) and all(0 <= v["swap"] <= 1 for v in twins["swap"].values())
        summary = faith.summarize({"new_words": result, "twins": twins})
        assert -1 <= summary["new_words"]["mean"]["comprehensiveness"] <= 1 and summary["twins"]["mean"]["rs"] is not None
    # the untyped store keeps no role: removing (r1, X) or (r2, X) leaves the same store, so its decodes are role-free
    blind = stores["C5ut"].store
    assert blind.role_blind and all(e["decoded"] in (True, False) for e in faith.decoded_edges(blind, _frame("Plurb Gateway"),
                                                                                               blind.frame_vector(_frame("Plurb Gateway"))))


def test_cli_outputs_and_the_report(world, item_dirs, new_dir, tmp_path) -> None:
    common_args = ["--items", str(item_dirs["twins"]), "--alias-table", str(world["alias"]), "--device", "cpu", "--batch-size", "8",
                   "--max-length", "64", "--max-lines", "1", "--overwrite", "--label", "SMOKE"]
    # C5 in two parts (conditions split over jobs: the report merges `self-query-<items>` and `self-query-<items>-<tag>`)
    sqx.main(["evaluate", "--run", str(world["runs"]["C5"]), "--conditions", "none,recall:own", *common_args])
    sqx.main(["evaluate", "--run", str(world["runs"]["C5"]), "--conditions", "symbolic", "--tag", "part2", *common_args])
    sqx.main(["evaluate", "--run", str(world["runs"]["C5ut"]), "--conditions", "none,recall:C5,symbolic", *common_args])
    faith_args = ["--new-items", str(new_dir), "--twins", str(item_dirs["twins"]), "--alias-table", str(world["alias"]), "--device", "cpu",
                  "--batch-size", "8", "--overwrite"]
    for model in ("C5", "C5ut"):            # F1 in two term ranges; the twins once
        faith.main(["evaluate", "--run", str(world["runs"][model]), "--limit", "2", *faith_args])
        faith.main(["evaluate", "--run", str(world["runs"][model]), "--limit", "2", "--offset", "2", "--skip-twins", "--tag", "part2",
                    *faith_args])
    folder = world["runs"]["C5"] / f"{sqx.OUTPUT_PREFIX}{item_dirs['twins'].name}"
    for name in ("summary.json", "predictions.jsonl.gz", "recalls.jsonl.gz", "report.md", "manifest.json"):
        assert (folder / name).exists()
    assert {r["condition"] for r in und.read_jsonl(folder / "recalls.jsonl")} == {"recall:own"}
    assert {r["condition"] for r in und.read_jsonl(folder.with_name(folder.name + "-part2") / "recalls.jsonl")} == {"symbolic"}
    analysis = report.analyse(world["runs"]["C5"].parent, twins=item_dirs["twins"].name)
    host = analysis["hosts"]["fake"]
    assert set(host["q1"]["contrasts"]) == {"recall:own − none"} and host["q1"]["reading"]
    assert host["f1"]["available"] and "F1b: C5 − C5ut" in host["f1"]["contrasts"]
    assert any(r["model"] == "C5ut" and r["condition"] == "recall:C5" for r in host["twins_table"]["choice"])
    assert any(r["model"] == "C5" and r["condition"] == "symbolic" for r in host["twins_table"]["choice"])      # the merged part
    merged = report.discover(world["runs"]["C5"].parent, sets=[item_dirs["twins"].name])["fake"]["C5"][1]["faithfulness"]["summary"]
    assert len(merged["new_words"]["terms"]) > len(json.loads((world["runs"]["C5"] / faith.OUTPUT / "summary.json").read_text())
                                                   ["summary"]["new_words"]["terms"])          # both term ranges
    text = report.render(analysis, title="toy", label="SMOKE")
    assert "Q1" in text and "F1" in text and "SMOKE" in text
    with pytest.raises(ValueError):
        report.main(["--runs", str(world["runs"]["C5"].parent), "--output", str(tmp_path / world["runs"]["C5"].parent.name)])


def _doc(conditions: dict[str, dict[str, float]]) -> dict:
    return {"summary": {"conditions": {c: {"choice": {"units": {u: {"contrast": v, "decode_all": 1.0} for u, v in units.items()}}}
                                       for c, units in conditions.items()}}}


def test_report_statistics_on_synthetic_units() -> None:
    rng = np.random.default_rng(0)
    pairs = [str(p) for p in range(60)]
    jitter = lambda: {p: float(np.clip(rng.normal(0, 0.1), -0.4, 0.4)) for p in pairs}
    models = {"C5": {s: {"twins": _doc({"recall:own": {p: 0.9 + j for p, j in jitter().items()}, "none": {p: 0.5 + j for p, j in jitter().items()},
                                        "recall:C5ut": {p: 0.5 + j for p, j in jitter().items()}, "symbolic": {p: 0.95 for p in pairs}})}
                     for s in (1, 2, 3)},
              "P0": {1: {"twins": _doc({"none": {p: 0.5 for p in pairs}, "recall:C5@1": {p: 0.8 for p in pairs},
                                        "recall:C5@2": {p: 0.7 for p in pairs}})}}}
    q = report.q1(models, "twins")
    assert set(q["contrasts"]) == {"recall:own − none", "recall:own − recall:C5ut"}
    assert all(r["p_holm"] < 0.05 and r["model"]["mean"] > 0.3 and r["seeds"] == [1, 2, 3] for r in q["contrasts"].values())
    assert q["reading"].startswith("(a)") and q["means"]["symbolic"] == pytest.approx(0.95)
    p0 = report.seeds_of(models, "P0", "twins", "recall:C5")            # P0's per-seed store conditions become its seeds
    assert sorted(p0) == [1, 2] and p0[2]["0"] == pytest.approx(0.7)
    blind = {s: {"twins": _doc({"recall:own": {p: 0.5 for p in pairs}, "none": {p: 0.5 for p in pairs}})} for s in (1, 2)}
    assert report.q1({"C5": blind}, "twins")["reading"].startswith("(c)")
    faith_doc = lambda value: {"faithfulness": {"summary": {"new_words": {"terms": {f"t{i}": {"comprehensiveness": value + 0.05 * (i % 3)}
                                                                                     for i in range(30)}, "undecoded": {}}}}}
    f = report.f1({"C5": {1: faith_doc(0.4), 2: faith_doc(0.35)}, "C5ut": {1: faith_doc(0.4), 2: faith_doc(0.35)}})
    assert f["contrasts"]["F1a: C5 − 0"]["p_holm"] < 0.05 and f["reading"]["F1a"].startswith("(a)")
    assert f["contrasts"]["F1b: C5 − C5ut"]["model"]["mean"] == pytest.approx(0.0) and f["reading"]["F1b"].startswith("no difference")


def test_queue_dry_runs(world, item_dirs, tmp_path) -> None:
    root = tmp_path / "root"
    (root / "configs" / "toy").mkdir(parents=True)
    for model in ("P0", "C0p", "C5", "C5ut"):
        run = world["runs"][model]
        (root / "configs" / "toy" / f"{run.name}.yaml").write_text((run / "resolved_config.yaml").read_text())
    jobs = sqx.queue_stage("toy", item_dirs["twins"], root=root, dry_run=True)
    by_model = {j["model"]: j for j in jobs}
    assert set(by_model) == {"P0", "C0p", "C5", "C5ut"} and all(j["priority"] == 50 for j in jobs)
    command = lambda m: " ".join(by_model[m]["command"])
    assert "recall:C5@1" in command("P0") and "recall:C5," in command("C0p") and "recall:C5ut" in command("C5")
    assert all(j["name"].endswith(f"{sqx.OUTPUT_PREFIX}{item_dirs['twins'].name}") for j in jobs)
    f_jobs = faith.queue_stage("toy", root=root, dry_run=True)
    assert {j["model"] for j in f_jobs} == {"C5", "C5ut"} and all(j["name"].endswith(faith.OUTPUT) for j in f_jobs)
    formats = sqx.queue_stage("toy", item_dirs["twins"], root=root, models=["C5"], override=["recall-all:own", "fields:own"], tag="formats",
                              dry_run=True)
    assert len(formats) == 1 and formats[0]["name"].endswith("-formats") and "recall-all:own,fields:own" in " ".join(formats[0]["command"])
    assert formats[0]["command"][-2:] == ["--tag", "formats"]


# -- 3a: the agent harness (scripted generation: the fake host has 64 positions) ------------------------------------------------

from vsa_embed.experiments import e12_agent as agent  # noqa: E402


def _toolbox(world, item_dirs, understanding_dir, stores):
    run = _open(world, "C5")
    twins, und_set = sqx.load_item_set(item_dirs["twins"]), sqx.load_item_set(understanding_dir)
    combined = sqx.ItemSet("agent", twins.path, {"track": "toy", "family": "gpt2"},
                           list({c["concept"]: c for c in twins.concepts + und_set.concepts}.values()), twins.prompts + und_set.prompts)
    builder = sqx.ContextBuilder(combined, run.ontology, _lexicon(), {"own": stores["own"]})
    return run, twins, und_set, combined, builder, agent.Toolbox(builder, stores["own"], run.table)


def test_agent_tools_parse_and_answer(world, item_dirs, understanding_dir, stores) -> None:
    run, twins, und_set, combined, builder, tools = _toolbox(world, item_dirs, understanding_dir, stores)
    text, record = tools.call("recall", "Brightwater Ledger")
    assert text.startswith("recall(Brightwater Ledger):") and record["parsed"] and len(text.splitlines()) == 1 + len(_frame("Brightwater Ledger"))
    text, record = tools.call("recall", "Brightwater Ledger, depends on")
    assert text.splitlines()[0] == "recall(Brightwater Ledger, depends on):" and len(text.splitlines()) == 2 and record["relation"] == "depends_on"
    twin = combined.concepts[0]
    assert tools.call("recall", twin["surface"])[1]["parsed"]                       # a new term resolves by its surface
    assert tools.call("recall", "the Zash Team")[1]["parsed"]                       # "the …" names resolve too
    assert not tools.call("recall", "Nonexistent Thing")[1]["parsed"]
    assert not tools.call("recall", "Brightwater Ledger, likes")[1]["parsed"]
    text, record = tools.call("lookup", "owned by, the Zash Team")
    assert text.startswith("lookup(owned by, the Zash Team):") and record == {"parsed": True, "relation": "owned_by", "value": "term:Zash Team"}
    assert len(text.splitlines()) == 6
    assert not tools.call("lookup", "owned by")[1]["parsed"]
    qs = agent.questions(twins, und_set, twin_pairs=2, two_hop=1, reverse=1)
    assert [q.kind for q in qs].count("twins") == 4 and {q.kind for q in qs} == {"twins", "two_hop", "reverse"}
    assert qs[0].text().startswith("Question: ") and " ___? Options: " in qs[0].text()
    demos = agent.demonstrations(builder, tools, set())
    assert "Action: recall[" in demos and "Observation: recall(" in demos and "Answer:" in demos
    assert "Action:" not in agent.demonstrations(builder, tools, set(), tool=False)


def test_agent_episodes_insert_observations_and_stop_at_the_answer(world, item_dirs, understanding_dir, stores, monkeypatch) -> None:
    run, twins, und_set, combined, builder, tools = _toolbox(world, item_dirs, understanding_dir, stores)
    tokenizer = run.adapter.tokenizer
    scripts = {"called": " I need it.\nAction: recall[Brightwater Ledger, depends on]\n Done.\nAnswer: Grosh Console",
               "direct": " I know.\nAnswer: x", "rambling": " " + "word " * 60}
    queues = {k: tokenizer.encode(v) for k, v in scripts.items()}

    def scripted(adapter, texts):
        out = []
        for text in texts:
            key = text.split("<<")[1].split(">>")[0]
            out.append(queues[key].pop(0) if queues[key] else tokenizer.eos_token_id)
        return out

    monkeypatch.setattr(agent, "next_tokens", scripted)
    q = agent.questions(twins, None, twin_pairs=1, two_hop=0, reverse=0)[0]
    episodes = agent.run_episodes(run.adapter, tools, [(q, f"<<{k}>>\nThought:") for k in scripts], batch=2, log=lambda *_: None)
    called, direct, rambling = episodes
    assert called.answered and called.text.endswith("Answer:") and len(called.actions) == 1 and called.actions[0]["parsed"]
    assert "Observation: recall(Brightwater Ledger, depends on):" in called.text
    assert direct.answered and not direct.actions
    assert not rambling.answered and rambling.text.endswith("\nAnswer:") and rambling.tokens <= agent.MAX_SEGMENT + 1
    rows = [{"kind": "twins", "gold": 0, "meta": {"pair": 0, "twin": t}, "actions": [], "answered": True, "called": c, "format_ok": c,
             "relevant": c, "scores": {k: ([0.0, -1.0] if t == "A" else [-1.0, 0.0]) for k in ("agent", "no_tool", "fixed")},
             "correct": {k: 1.0 for k in ("agent", "no_tool", "fixed")}} for t, c in (("A", True), ("B", False))]
    rows[1]["gold"] = 1
    summary = agent.summarize(rows)
    assert summary["all"]["called"] == 0.5 and summary["twins_contrast"] == {"agent": 1.0, "no_tool": 1.0, "fixed": 1.0, "pairs": 1}


def test_chunked_head_matches_the_full_head() -> None:
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=True)
    hidden, targets = torch.randn(7, 16), torch.randint(0, 50, (7,))
    full = faith.head_logprobs(head, hidden, targets)
    assert torch.allclose(faith.head_logprobs(head, hidden, targets, chunk=8), full, atol=1e-5)
    assert torch.equal(faith.head_argmax(head, hidden, chunk=8), faith.head_argmax(head, hidden))


def test_agent_parts_merge(tmp_path) -> None:
    def part(folder, rows):
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps({"source": {"condition": "P0", "seed": 1, "size": "x"}, "store": "s",
                                                         "host_dtype": "bfloat16", "label": "PILOT", "seconds": 10.0, "peak_gb": 1.5}))
        und.write_jsonl_gz(folder / "episodes.jsonl.gz", rows)

    def row(i, twin):
        return {"id": f"q{i}{twin}", "kind": "twins", "gold": 0 if twin == "A" else 1, "meta": {"pair": i, "twin": twin}, "actions": [],
                "answered": True, "called": True, "format_ok": True, "swapped": False, "relevant": True,
                "scores": {k: ([0.0, -1.0] if twin == "A" else [-1.0, 0.0]) for k in ("agent", "no_tool", "fixed")},
                "correct": {k: 1.0 for k in ("agent", "no_tool", "fixed")}}

    part(tmp_path / "p0", [row(0, "A"), row(0, "B")])
    part(tmp_path / "p1", [row(1, "A"), row(1, "B")])
    merged = agent.merge([tmp_path / "p0", tmp_path / "p1"], tmp_path / "merged")
    assert merged["summary"]["all"]["questions"] == 4 and merged["summary"]["twins_contrast"]["pairs"] == 2
    assert json.loads((tmp_path / "merged" / "summary.json").read_text())["seconds"] == 20.0 and (tmp_path / "merged" / "report.md").exists()
