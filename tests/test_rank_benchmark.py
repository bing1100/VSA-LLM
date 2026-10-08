"""The log-probability ranking harness (`vsa_embed.benchmarks.ranking`, decision 63 TK-B1) on the tiny E9 world (CPU):
the item format, conditions, the scorer against the shared E9/E11 scorers, store rows against E9's `own` / `none`,
exact ties, channel surgery restored, the readers, the CLI run folder and the pooled report with Holm."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401  (module fixtures re-used)

from vsa_embed.benchmarks import ranking as rk  # noqa: E402
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")
QUIET = lambda message: None  # noqa: E731


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)
    yield


def _term(surface, frame=None, definition=None, concept=None):
    return {"surface": surface, "concept": concept, "frame": frame, "definition": definition}


def _new_word_items(items_dir: Path) -> tuple[list[rk.Item], list[dict]]:
    """The E9 new-word property items as rank items (one per template): the term is the invented word with its frame."""
    _, concepts, prompts = edit.load_item_dir(items_dir, edit.SCHEMA_NEW)
    by = {c["concept"]: c for c in concepts}
    out = []
    for p in prompts:
        if p["test"] != "property":
            continue
        c = by[p["concept"]]
        for t, template in enumerate(p["templates"]):
            definition = f"{c['surface']} is a kind of thing."
            out.append(rk.Item.from_json({"id": f"{p['id']}-{t}", "set": "toy", "context": template.format(x=c["surface"]),
                                          "options": p["candidates"], "answer": p["gold"],
                                          "terms": [_term(c["surface"], c["frame"], definition)], "meta": {"concept": c["concept"]}}))
    return out, concepts


def test_item_format_round_trip_and_validation(tmp_path) -> None:
    row = {"id": "a", "set": "s", "kind": "rank", "context": "A wug", "options": [" can bark.", " can bark."], "answer": 0,
           "terms": [], "option_terms": [[_term("wug", [["hypernym", "synset:dog.n.01"]], "A wug is a dog.")],
                                         [_term("wug", [["hypernym", "synset:cat.n.01"]], "A wug is a cat.")]],
           "joiner": " ", "meta": {"pair": 1}}
    item = rk.Item.from_json(row)
    assert item.option_terms[0][0].insert and item.terms_of(1)[0].frame == (("hypernym", "synset:cat.n.01"),)
    info = rk.write_items(tmp_path / "x.jsonl.gz", [item])
    again = rk.load_items(tmp_path / "x.jsonl.gz")
    assert again[0].to_json() == item.to_json() and info["items"] == 1
    assert rk.write_items(tmp_path / "y.jsonl.gz", [item])["file_sha256"] == info["file_sha256"]     # deterministic gzip
    with pytest.raises(ValueError):
        rk.Item.from_json({**row, "answer": 5})
    with pytest.raises(ValueError):
        rk.Item.from_json({**row, "kind": "generate"})
    known = rk.Term.from_json(_term("dog", None, None, concept="synset:dog.n.01"))
    assert not known.insert and rk.Term.from_json({**_term("dog", concept="Q144"), "insert": True}).insert
    c = rk.parse_condition("store:oracle+definition-in-context")
    assert c.rows == "oracle" and c.context == ("definition",) and c.channel
    assert rk.parse_condition("channel-off").channel is False and rk.parse_condition("none").rows is None
    assert rk.parse_condition("frame-in-context:linker").readers == ["linker"]
    with pytest.raises(ValueError):
        rk.parse_condition("store:magic")


def test_scorer_matches_the_shared_continuation_scorers(world) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=4)
    prefixes = ["The house cat chased the", "A glass jar and a", "He used a claw hammer on the"]
    continuations = [" domestic dog", " tea kettle stood", " motorcar"]
    expected, counts = e11.continuation_scores(run.adapter, prefixes, continuations)
    scorer = rk.RankScorer(run, [], batch_size=2, token_budget=64)
    got = scorer.score([rk.Request(p + c, len(p)) for p, c in zip(prefixes, continuations)])
    np.testing.assert_allclose(got["sum"], expected, atol=1e-5)
    np.testing.assert_array_equal(got["tokens"], counts)
    sums, _ = edit.continuation_stats(run.adapter, prefixes, continuations)
    np.testing.assert_allclose(got["sum"], sums, atol=1e-4)
    # channel-off equals the adapter with no spans
    off = scorer.score([rk.Request(p + c, len(p), channel=False) for p, c in zip(prefixes, continuations)])
    saved = run.adapter.spans_fn
    run.adapter.spans_fn = None
    try:
        plain, _ = e11.continuation_scores(run.adapter, prefixes, continuations)
    finally:
        run.adapter.spans_fn = saved
    np.testing.assert_allclose(off["sum"], plain, atol=1e-5)
    assert not np.allclose(off["sum"], got["sum"])
    # left truncation keeps the scored continuation
    short = rk.RankScorer(run, [], max_length=4).score([rk.Request(prefixes[0] + continuations[0], len(prefixes[0]))])
    assert short["truncated"][0] and short["tokens"][0] == counts[0]


def test_store_rows_reproduce_e9_own_and_none(world, items) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    rank_items, concepts = _new_word_items(items["new"])
    rows_before = common.entry_rows(run.channel, torch.arange(run.channel.entry_count))
    evaluation = rk.Evaluation(run, rank_items, batch_size=8, log=QUIET)
    result = evaluation.evaluate([rk.parse_condition(c) for c in ("none", "store:oracle", "store:random", "channel-off")])
    torch.testing.assert_close(common.entry_rows(run.channel, torch.arange(run.channel.entry_count)), rows_before, rtol=0, atol=0)
    assert run.channel.entry_count == world["ontology"]["entry_count"]
    # E9's own scoring of the same texts: new names inserted with their frames (own) or as zero rows (none).
    o = world["ontology"]
    relation_id = {n: i for i, n in enumerate(o["relation_names"])}
    atom_id = {n: i for i, n in enumerate(o["atomic_names"])}
    base = int(o["entry_count"])
    entry_of = {c["concept"]: base + i for i, c in enumerate(concepts)}
    adapter = edit.extended_adapter(run, {c["surface"]: entry_of[c["concept"]] for c in concepts})
    frames = [edit.resolve_frame(c["frame"], relation_id, atom_id) for c in concepts]
    prefixes = [i.context for i in rank_items for _ in i.options]
    continuations = [o_ for i in rank_items for o_ in i.options]
    with edit.inserted_entries(run.channel, len(concepts), frames):
        own, _ = e11.continuation_scores(adapter, prefixes, continuations)
        zero = {entry_of[c["concept"]]: torch.zeros(run.channel.gate.in_features // 2) for c in concepts}
        with common.override_rows(run.channel, zero):
            none, _ = e11.continuation_scores(adapter, prefixes, continuations)
    ours = {c: np.concatenate([result["items"][i.id]["conditions"][c]["sum"] for i in rank_items]) for c in ("none", "store:oracle")}
    np.testing.assert_allclose(ours["store:oracle"], own, atol=1e-5)
    np.testing.assert_allclose(ours["none"], none, atol=1e-5)
    assert not np.allclose(ours["store:oracle"], ours["none"])
    linked = [result["items"][i.id]["conditions"]["store:oracle"]["linked"] for i in rank_items]
    assert min(linked) == 1.0
    random_sums = np.concatenate([result["items"][i.id]["conditions"]["store:random"]["sum"] for i in rank_items])
    assert not np.allclose(random_sums, ours["store:oracle"])
    for item in rank_items:
        gold, rand = evaluation.gold(item.terms[0]), evaluation.frames("random")[item.terms[0].key]
        assert [r for r, _ in rand] == [r for r, _ in gold]                  # equal degree, same relations


def test_minimal_pairs_tie_exactly_without_information(world) -> None:
    """COMPS-style: the same continuation under two frames of a nonce term. No row → exact tie (0.5); identical frames →
    exact tie; different frames → a decision."""
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=4)
    dog = [["hypernym", "synset:canine.n.02"], ["lexname", "lexname:noun.animal"], ["pos", "pos:n"]]
    cat = [["hypernym", "synset:feline.n.01"], ["lexname", "lexname:noun.animal"], ["pos", "pos:n"]]
    rows = []
    for k, (fa, fb) in enumerate([(dog, cat), (dog, dog)]):
        rows.append(rk.Item.from_json({"id": f"p{k}", "set": "pairs", "context": "Therefore, a blorptak", "options": [" has a tail."] * 2,
                                       "answer": 0, "terms": [], "joiner": " ",
                                       "option_terms": [[_term("blorptak", fa, "A blorptak is a dog.")],
                                                        [_term("blorptak", fb, "A blorptak is a cat.")]]}))
    evaluation = rk.Evaluation(run, rows, log=QUIET)
    names = ["none", "store:oracle", "definition-in-context", "frame-in-context:oracle", "channel-off"]
    result = evaluation.evaluate([rk.parse_condition(c) for c in names])["items"]
    assert result["p0"]["conditions"]["none"]["correct"]["sum"] == 0.5
    assert result["p1"]["conditions"]["store:oracle"]["correct"]["sum"] == 0.5          # same frame on both sides
    assert result["p0"]["conditions"]["store:oracle"]["correct"]["sum"] in (0.0, 1.0)
    sums = result["p0"]["conditions"]["store:oracle"]["sum"]
    assert sums[0] != sums[1]
    d = result["p0"]["conditions"]["definition-in-context"]
    assert d["context_tokens"] > 0 and d["sum"][0] != d["sum"][1]
    prefix, added = evaluation._prefix(rows[0], 0, rk.parse_condition("definition-in-context"))
    assert prefix == "A blorptak is a dog. Therefore, a blorptak" and added == "A blorptak is a dog."
    prefix, added = evaluation._prefix(rows[0], 1, rk.parse_condition("frame-in-context:oracle"))
    assert "blorptak is a kind of feline" in added and "pos" not in added
    summary = rk.summarize(result, names, resamples=50)
    contrasts = {(c["a"], c["b"]) for c in summary["sets"]["all"]["contrasts"]}
    assert ("store:oracle", "none") in contrasts and summary["chance"] == 0.5


def test_conditions_by_channel_mode_and_shadowed_aliases(world) -> None:
    p0 = common.open_run(world["runs"]["P0"], device="cpu")
    kept, skipped = rk.available_conditions(p0, ["none", "channel-off", "store:oracle", "definition-in-context"])
    assert [c.name for c in kept] == ["none", "definition-in-context"] and skipped == ["channel-off", "store:oracle"]
    c2 = common.open_run(world["runs"]["C2"], device="cpu")
    kept, skipped = rk.available_conditions(c2, ["store:oracle", "store:linker"])
    assert [c.name for c in kept] == ["store:oracle"] and skipped == ["store:linker"]
    c5 = common.open_run(world["runs"]["C5"], device="cpu")
    scorer = rk.RankScorer(c5, ["dog", "flurbix"])
    assert scorer.shadowed == ["dog"] and scorer.placeholder_of("Flurbix") is not None
    table = c5.adapter.linker.table
    assert scorer.placeholder_of("dog") >= len(table.entry_concepts)             # the new term wins the surface


def test_readers_on_definitions(world, items) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    rank_items, _ = _new_word_items(items["new"])
    rank_items = rank_items[:4]
    for item in rank_items:                       # definitions that name an atom the concept finder knows
        t = item.terms[0]
        item.terms[0] = rk.Term(t.surface, None, t.frame, f"{t.surface} is a kind of canine with a tail.")
    evaluation = rk.Evaluation(run, rank_items, log=QUIET)
    for reader in ("typeprior", "linker", "linker-joint", "pattern"):
        frames = evaluation.frames(reader)
        assert set(frames) == set(evaluation.terms)
    assert any(evaluation.frames("typeprior").values())
    summary = evaluation.reader_summary()
    assert summary["linker"]["cost_per_term"]["forward_passes"] > 1 and "linker-all" in evaluation._frames
    result = evaluation.evaluate([rk.parse_condition("store:linker"), rk.parse_condition("frame-in-context:typeprior")])
    assert all("store:linker" in r["conditions"] for r in result["items"].values())


def test_cli_evaluate_and_pooled_report(world, items, tmp_path) -> None:
    rank_items, _ = _new_word_items(items["new"])
    path = tmp_path / "items.jsonl.gz"
    rk.write_items(path, rank_items[:6])
    outputs = []
    for model in ("C5", "C0p"):
        out = tmp_path / f"out-{model}"
        rk.main(["evaluate", "--run", str(world["runs"][model]), "--items", str(path), "--output", str(out), "--device", "cpu",
                 "--threads", "1", "--conditions", "none,store:oracle,definition-in-context", "--resamples", "50", "--smoke"])
        summary = json.loads((out / "summary.json").read_text())
        assert summary["smoke"] and (out / "report.md").read_text().startswith("**SMOKE TEST")
        with gzip.open(out / "items.jsonl.gz", "rt") as handle:
            assert len(handle.readlines()) == 6
        outputs.append(out)
    assert json.loads((outputs[1] / "summary.json").read_text())["skipped_conditions"] == ["store:oracle"]
    label = rk.model_label(json.loads((outputs[0] / "summary.json").read_text())["source"])
    other = rk.model_label(json.loads((outputs[1] / "summary.json").read_text())["source"])
    spec = {"primary": [{"name": "W1", "a": {"model": label, "condition": "store:oracle"}, "b": {"model": label, "condition": "none"}},
                        {"name": "X", "a": {"model": label, "condition": "none"}, "b": {"model": other, "condition": "none"}}],
            "secondary": [{"name": "S", "a": {"model": label, "condition": "definition-in-context"}, "b": {"model": label, "condition": "none"}}]}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    rk.main(["report", "--inputs", *map(str, outputs), str(tmp_path / "missing"), "--output", str(tmp_path / "report"),
             "--contrasts", str(tmp_path / "spec.json"), "--resamples", "50"])
    report = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert [c["name"] for c in report["primary"]] == ["W1", "X"] and all("holm_p" in c for c in report["primary"])
    assert report["primary"][0]["holm_p"] >= report["primary"][0]["result"]["p_value"]
    assert report["sources"][-1]["missing"]
