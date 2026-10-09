"""E11 amendment of decision 64 (preregistration §16) on the tiny E9 world (CPU): the shape-matched content control
`linker-random`, the in-context route on the evaluation windows (token accounting; an empty prefix reproduces the plain
windows), the definition-encoder route on a trained row-source (C6d) run (the re-encoded table matches the stored one,
`oracle` is the run's own row, `none` is no row, the table is restored), the amended P2 rule (intersection–union test,
Holm over P1 and P2), the cross-model encoder contrasts and costs in the pooled report, the plan and the decision-64
queue script."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _config, _ok, fake_host, items, world  # noqa: E402,F401  (module fixtures re-used)
from test_e11_read_to_learn import QUIET, _heldout_set  # noqa: E402

from vsa_embed import read_to_learn as rtl  # noqa: E402
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e9_rowsource  # noqa: E402
from vsa_embed.experiments import e9_tracks as tracks  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402
import vsa_embed.training.lm as lm  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")


@pytest.fixture(scope="module")
def sets(world, items, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e11-d64")
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    e11.build_new_set(items["new"], root / "new", ctx)
    return {"new": root / "new", "heldout": _heldout_set(world, root)}


@pytest.fixture(scope="module")
def c6d(world, tmp_path_factory) -> Path:
    """A trained C6d arm of the tiny world: the definition table of the fake host, then a short run reading it."""
    from transformers import AutoTokenizer
    root = tmp_path_factory.mktemp("e11-c6d")
    table = root / "definition.pt"
    e9_rowsource.build_table("definition", ontology=world["ontology"], output=table, table=world["table"],
                             lexicon=edit.WordNetLexicon(), model=fake_host(None),
                             tokenizer=AutoTokenizer.from_pretrained("gpt2", local_files_only=True),
                             verbalization_path=root / "verbalized.jsonl.gz")
    channel = {"mode": "source", "source_kind": "definition", "source_table": str(table), "source_hidden": 8}
    folder = world["root"] / "runs" / "fake-lora-C6d-s1"
    lm.train(_config(world["root"], "lora-C6d", channel, "lora"), folder)
    return folder


def test_shape_random_keeps_the_shape_and_changes_the_content() -> None:
    pools = {0: Counter({1: 5, 2: 3, 3: 1}), 1: Counter({4: 2, 5: 2, 8: 1}), 2: Counter({6: 1})}
    task = rtl.ReadTask("c", "prose", "c: a thing.", "c", [(0, 1), (1, 4)])
    base = rtl.ReadResult("c", "prose", "linker", [(0, 2), (1, 5), (2, 6)])
    out = rtl.read_shape_random(task, base, pools, random.Random(0))
    # Never the linker's filler nor a gold filler of the relation; relation 2 has no other filler: its edge is dropped.
    assert out.reader == "linker-random" and out.frame == [(0, 3), (1, 8)] and out.details["dropped"] == 1
    assert rtl.read_shape_random(task, rtl.ReadResult("c", "prose", "linker", None), pools, random.Random(0)).frame is None
    big = {0: Counter({a: 1 for a in range(10, 40)})}
    frames = [rtl.read_shape_random(task, rtl.ReadResult("c", "prose", "linker", [(0, 10)] * 3), big, random.Random(s)).frame
              for s in range(20)]
    assert all(len(f) == 3 and all(r == 0 and a != 10 for r, a in f) for f in frames)
    assert "linker-random" in rtl.READERS and "linker-random" in rtl.MODEL_READERS
    assert tracks.relation_pools({"relations": [0, 0, 1], "fillers": [7, 7, 9]}) == {0: Counter({7: 2}), 1: Counter({9: 1})}


def test_encoder_text_removes_the_headword() -> None:
    assert e11.encoder_text("aardvark: a kind of mammal that eats ants.", "aardvark") == "a kind of mammal that eats ants."
    assert e11.encoder_text("Glorbix is owned by the Zash Team. Glorbix uses X.", "glorbix") == "It is owned by the Zash Team. It uses X."
    assert e11.encoder_text("aardvark:", "aardvark") == "aardvark:"                        # nothing left: the text is kept


def test_linker_random_and_the_in_context_route_on_windows(world, sets) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["heldout"])
    schedule_before = run.composer.schedule
    result = e11.evaluate(run, read_set, methods=["frames", "persistence", "context", "windows"],
                          readers=["oracle", "linker", "random", "linker-random", "none"], styles=["prose"], primary_style="prose",
                          resamples=50, log=QUIET)
    assert torch.equal(run.composer.schedule.fillers, schedule_before.fillers) and not run.channel.skip_empty_frames
    frames = {(r["reader"], r["concept"]): r["frame"] for r in result["frames"]}
    assert any(frames[("linker", c["concept"])] for c in read_set.concepts)
    for c in read_set.concepts:                          # the same relations and edge count as the linker's, other fillers
        linker, shaped = frames[("linker", c["concept"])], frames[("linker-random", c["concept"])]
        assert (linker is None) == (shaped is None)
        if linker:
            assert sorted(r for r, _ in shaped) == sorted(r for r, _ in linker) and not set(map(tuple, shaped)) & set(map(tuple, linker))
    document = result["document"]
    assert "prose|linker-random" in document["items"]["means"]
    conditions = result["windows"]["conditions"]
    assert {"prose|linker-random", "prose|context", "prose|context+linker"} <= set(conditions)
    references = {(c["condition"], c["reference"]) for c in document["windows"]["contrasts"]}
    assert {("prose|linker", "random"), ("prose|linker", "linker-random"), ("prose|context", "linker"),
            ("prose|context+linker", "context"), ("prose|context", "none")} <= references
    tokens = document["windows"]["context_tokens"]
    limit = run.model.model.config.n_positions
    assert 0 < tokens["prefix_tokens_mean"] <= tokens["budget"] <= limit - 32 and tokens["occurrences_with_definition"] > 0
    np.testing.assert_array_equal(conditions["prose|context"]["count_other"], conditions["prose|none"]["count_other"])
    assert not np.allclose(conditions["prose|context"]["sum_other"], conditions["prose|none"]["sum_other"])
    # An empty prefix (no definition) reproduces the plain windows exactly: the prefix path adds nothing else.
    entry_of = e11.link_entry_of(read_set, run)
    wset = e11.build_window_set(run, [entry_of[c["concept"]] for c in read_set.concepts],
                                {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts})
    corpus = Path(run.config["data"]["eval"])
    none = {c["concept"]: None for c in read_set.concepts}
    with e11.term_condition(run, entry_of, none):
        plain = e11.window_term_losses(run, corpus, wset)
        empty, record = e11.window_context_losses(run, corpus, wset, [None] * len(read_set.concepts))
    np.testing.assert_allclose(empty["sum_other"], plain["sum_other"], atol=1e-4)
    np.testing.assert_allclose(empty["sum_own"], plain["sum_own"], atol=1e-4)
    assert record["prefix_tokens_total"] == 0 and record["occurrences_with_definition"] == 0
    np.testing.assert_allclose(conditions["prose|none"]["sum_other"], plain["sum_other"], atol=1e-5)


def test_text_route_model_reads_windows_in_context(world, sets) -> None:
    run = common.open_run(world["runs"]["C0p"], device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["heldout"])
    result = e11.evaluate(run, read_set, methods=["persistence", "context", "windows", "encoder"], readers=["linker", "none"],
                          styles=["prose"], primary_style="prose", resamples=50, log=QUIET)
    assert set(result["windows"]["conditions"]) == {"prose|none", "prose|context"}
    assert result["document"]["encoder"] == {"applicable": False, "reason": "needs a definition row-source run (C6d)"}


def test_encoder_route_on_a_c6d_run(world, sets, c6d, monkeypatch) -> None:
    monkeypatch.setattr(e11, "encoder_host", lambda run, record: fake_host(None).eval())
    run = common.open_run(c6d, device="cpu", batch_size=8)
    assert run.channel.mode == "source" and run.condition == "C6d"
    count = int(run.ontology["entry_count"])
    table_before = run.channel.source_rows[:count].clone()
    state_before = {k: v.clone() for k, v in run.model.state_dict().items()}
    read_set = e11.load_read_set(sets["heldout"])
    result = e11.evaluate(run, read_set, methods=["encoder", "persistence", "windows"], readers=["none"], styles=["prose"],
                          primary_style="prose", resamples=50, log=QUIET)
    document = result["document"]
    check = document["encoder"]["check"]
    assert check["cosine_to_table_mean"] > 0.999 and check["texts"] == "the table's stored verbalizations"
    cost = document["encoder"]["cost_per_word"]["prose"]
    assert cost["terms"] == len(read_set.concepts) and cost["forward_tokens"] > 0 and cost["context_tokens_per_use"] == 0.0
    assert {"prose|none", "prose|oracle", "prose|encoder"} <= set(result["item_rows"])
    torch.testing.assert_close(run.channel.source_rows[:count], table_before, rtol=0, atol=0)        # the table is restored
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)
    # `oracle` is the run's own row (its table holds the held-out terms' gold-frame encodings); `none` is no row.
    ctx = e11.run_context(run)
    scorer = e11.ItemScorer(run, read_set, ctx)
    own = scorer.score({c: ctx.ids(next(x for x in read_set.concepts if x["concept"] == c)["frame"]) for c in scorer.concepts})
    values = lambda rows: e11.test_values(rows, "heldout")["property"]  # noqa: E731
    assert values(own) == values(result["item_rows"]["prose|oracle"])
    entry_of = e11.link_entry_of(read_set, run)
    entries = sorted(entry_of.values())
    with e11.term_condition(run, entry_of, {c: None for c in entry_of}):
        assert common.entry_rows(run.channel, entries).abs().max() == 0
    conditions = result["windows"]["conditions"]
    assert {"prose|none", "prose|oracle", "prose|encoder"} <= set(conditions)
    wset = e11.build_window_set(run, [entry_of[c["concept"]] for c in read_set.concepts],
                                {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts})
    plain = e11.window_term_losses(run, Path(run.config["data"]["eval"]), wset)
    np.testing.assert_allclose(conditions["prose|oracle"]["sum_other"], plain["sum_other"], atol=1e-5)
    assert not np.allclose(conditions["prose|none"]["sum_other"], plain["sum_other"])
    assert any(c["reference"] == "oracle" and c["condition"] == "prose|encoder" for c in document["windows"]["contrasts"])
    # The encoder writes the gold-frame text of a held-out term back onto its table row (same host, texts and statistics).
    encoder = e11.DefinitionEncoder(run, ctx)
    held = entries[0]
    vector = encoder([e9_rowsource.verbalize_frame(ctx.frame(held), run.ontology, edit.WordNetLexicon())])[0]
    assert torch.nn.functional.cosine_similarity(vector, table_before[held].float(), dim=0) > 0.999


def test_encoder_route_on_new_words(world, sets, c6d, monkeypatch) -> None:
    monkeypatch.setattr(e11, "encoder_host", lambda run, record: fake_host(None).eval())
    run = common.open_run(c6d, device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["new"])
    result = e11.evaluate(run, read_set, methods=["encoder"], readers=["none"], styles=["dictionary"], primary_style="dictionary",
                          resamples=50, log=QUIET)
    means = result["document"]["items"]["means"]
    assert {"dictionary|none", "dictionary|oracle", "dictionary|encoder"} <= set(means)
    assert run.channel.entry_count == world["ontology"]["entry_count"]
    contrasts = {(c["a"], c["b"]) for c in result["document"]["items"]["contrasts"]}
    assert ("encoder", "none") in contrasts and ("encoder", "oracle") in contrasts


# -- the pooled report -------------------------------------------------------------------------------------------------------

def _folder(root: Path, name: str, *, track: str, kind: str, model: str, seed: int, items: dict[str, list[float]] | None = None,
            windows: dict[str, float] | None = None, summary: dict | None = None) -> Path:
    """A synthetic E11 output folder: per condition item correctness (property) and per-term window sums (30 terms × 8 targets)."""
    out = root / name
    out.mkdir(parents=True)
    style = e11.PRIMARY_STYLE[track]
    with (out / "predictions.jsonl").open("w") as handle:
        for cond, values in (items or {}).items():
            for i, v in enumerate(values):
                handle.write(json.dumps({"condition": f"{style}|{cond}", "concept": f"c{i}", "id": f"i{i}", "test": "property",
                                         "correct": v}) + "\n")
    if windows:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.01, 30)
        arrays = {}
        for cond, loss in windows.items():
            arrays[f"{style}|{cond}::sum_other"] = 8 * (loss + noise)
            arrays[f"{style}|{cond}::count_other"] = np.full(30, 8.0)
            arrays[f"{style}|{cond}::sum_own"] = np.zeros(30)
            arrays[f"{style}|{cond}::count_own"] = np.zeros(30)
            arrays[f"{style}|{cond}::unlinked"] = np.zeros((4, 2))
        np.savez_compressed(out / "windows.npz", **arrays)
    (out / "summary.json").write_text(json.dumps({"set": {"track": track, "kind": kind},
                                                  "source": {"size": "HuggingFaceTB/SmolLM2-360M/train", "condition": model, "seed": seed},
                                                  **(summary or {})}))
    return out


def test_report_amended_p2_competitor_and_costs(tmp_path, capsys) -> None:
    runs = []
    for seed in (1, 2):
        runs.append(_folder(tmp_path, f"t5-C5-s{seed}", track="t5", kind="new", model="C5", seed=seed,
                            items={"linker": [1.0] * 30 + [0.0] * 20, "none": [1.0] * 10 + [0.0] * 40}))
        runs.append(_folder(tmp_path, f"t4-C5-s{seed}", track="t4", kind="heldout", model="C5", seed=seed,
                            windows={"none": 2.0, "linker": 1.94, "random": 1.97, "linker-random": 1.93, "oracle": 1.9,
                                     "context": 1.8},
                            summary={"frames": {"chebi": {"linker": {"cost_per_word": {"forward_tokens": 2600.0, "forward_passes": 33.0}}}},
                                     "windows": {"context_tokens": {"prefix_tokens_per_occurrence": 21.5}}}))
        runs.append(_folder(tmp_path, f"t4-C6d-s{seed}", track="t4", kind="heldout", model="C6d", seed=seed,
                            windows={"none": 2.01, "oracle": 1.95, "encoder": 1.92},
                            summary={"encoder": {"cost_per_word": {"chebi": {"forward_tokens": 70.0}}}}))
        runs.append(_folder(tmp_path, f"t7rood-C5-s{seed}", track="t7rood", kind="heldout", model="C5", seed=seed,
                            windows={"none": 2.0, "linker": 1.99, "random": 1.995}))
    out = e11.run_report(type("A", (), {"runs": runs, "output": tmp_path / "report", "resamples": 200})())
    decision = out["decision"]
    assert decision["P1"]["positive"] and decision["P2"]["complete"]
    p2 = decision["P2"]
    # linker − linker-random is a loss *increase* (the shape-matched control does better): P2 is not supported.
    assert p2["components"]["linker-random"]["relative"] > 0 and not p2["reductions"] and not p2["supported"]
    assert p2["p_iut"] == max(c["p_value"] for c in p2["components"].values())
    endpoints = {r["kind"]: r for r in out["endpoints"]}
    assert endpoints["heldout"]["p_iut"] == p2["p_iut"] and endpoints["heldout"]["p_holm"] >= p2["p_iut"]
    t7 = next(s for s in out["specificity"] if s["track"] == "t7rood")
    assert t7["missing"] == ["linker-random"] and not t7["complete"] and not t7["specific_unadjusted"]
    competitor = {(r["contrast"], r["test"]): r for r in out["competitor"]}
    absolute = competitor[("C6d encoder − C5 linker", "loss after term")]
    assert absolute["seeds"] == 2 and absolute["relative"] == pytest.approx((1.92 - 1.94) / 1.94, abs=2e-3)
    gains = competitor[("(C6d encoder − C6d none) − (C5 linker − C5 none)", "loss after term")]
    assert gains["relative"] == pytest.approx(((1.92 - 2.01) - (1.94 - 2.0)) / 2.0, abs=2e-3)
    costs = {(c["track"], c["model"]): c for c in out["costs"]}
    assert costs[("t4", "C5")]["linker_forward_tokens"] == 2600.0 and costs[("t4", "C6d")]["encoder_forward_tokens"] == 70.0
    assert costs[("t4", "C5")]["context_tokens_per_occurrence"] == 21.5
    contexts = {(r["model"], r["a"], r["b"]) for r in out["secondary"]}
    assert ("C5", "context", "linker") in contexts and ("C6d", "encoder", "oracle") in contexts
    text = (tmp_path / "report" / "report.md").read_text()
    assert "P2 as amended (§16.1):** not supported" in text and "Definition-encoder competitor" in text
    # The same data with the shape-matched control no better than the linker: P2 is supported.
    for seed in (1, 2):
        folder = tmp_path / f"t4-C5-s{seed}"
        with np.load(folder / "windows.npz") as data:
            arrays = {k: data[k] for k in data.files}
        arrays["chebi|linker-random::sum_other"] = arrays["chebi|random::sum_other"]
        np.savez_compressed(folder / "windows.npz", **arrays)
    out = e11.run_report(type("A", (), {"runs": runs, "output": tmp_path / "report2", "resamples": 200})())
    assert out["decision"]["P2"]["reductions"] and out["decision"]["P2"]["supported"]
    capsys.readouterr()


def test_plan_runs_the_encoder_competitor_and_the_shape_control() -> None:
    jobs = e11.plan_jobs(tracks=["t5", "t4"], hosts=["SmolLM2-360M"], seeds=[1, 2, 3])
    by_name = {j["name"]: j for j in jobs}
    c6d = by_name["e11-t4-heldout-SmolLM2-360M-C6d-s2"]["command"]
    assert c6d[c6d.index("--methods") + 1] == "encoder,windows" and c6d[c6d.index("--styles") + 1] == "chebi"
    new = by_name["e11-t5-new-SmolLM2-360M-C6d-s1"]
    assert new["command"][new["command"].index("--methods") + 1] == "encoder" and new["tier"] == 1 and new["hours"] > 0
    assert new["command"][new["command"].index("--styles") + 1] == "prose"
    assert not any("C6d" in n for n in by_name if "t5-heldout" in n or "t4-new" in n)
    c5 = by_name["e11-t4-heldout-SmolLM2-360M-C5-s1"]["command"]
    assert "linker-random" in c5[c5.index("--readers") + 1].split(",")
    assert by_name["e11-t4-heldout-SmolLM2-360M-C6d-s1"]["command"][-1] in by_name["e11-report"]["command"]
    # The decision-64 script: every replacement keeps its name and priority; new jobs only for C6d; reports after them.
    cancel, adds = e11.decision64_jobs()
    added = {j["name"]: j for j in adds}
    assert len(cancel) == 15 and len(adds) == 24 and {c["name"] for c in cancel} <= set(added)
    for c in cancel:
        if not c["name"].endswith("report"):
            assert added[c["name"]]["priority"] == c["priority"]
    new = [n for n in added if n not in {c["name"] for c in cancel}]
    assert len(new) == 9 and all("-C6d-" in n for n in new)
    assert added["e11-t7rood-report"]["priority"] > max(added[n]["priority"] for n in new if "t7rood" in n)
    assert added["e11-t4-heldout-SmolLM2-360M-C6d-s1"]["priority"] == 52.5 and added["e11-report"]["priority"] == 54
    t7 = added["e11-t7-heldout-SmolLM2-360M-C5-s1"]["command"]
    assert "linker-random" in t7[t7.index("--readers") + 1] and "context" not in t7[t7.index("--methods") + 1]
    text = e11.render_decision64()
    assert text.count("# CANCEL: ") == 15 and text.count("jobqueue add") == 24 and "NOT EXECUTED" in text
