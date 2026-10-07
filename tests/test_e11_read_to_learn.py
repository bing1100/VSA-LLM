"""E11 read-to-learn on the tiny E9 world (CPU): item sets, the definition scorer against a direct recomputation, the
evaluation methods (frames, persistence, in-context, gradient, natural-text windows, locality) and their invariants,
consistency with the E9 dimension-3 evaluation, the CLI run folder, the pooled report and the plan."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401  (module fixtures re-used)

from vsa_embed import read_to_learn as rtl  # noqa: E402
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e5_zeroshot as zs  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e9_tracks as tracks  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")
QUIET = lambda message: None  # noqa: E731


def _heldout_set(world: dict, root: Path) -> Path:
    """A `heldout` read set over the world's held-out entries: a prose definition from the gold frame, property items."""
    ontology, table = world["ontology"], world["table"]
    ctx = e11.make_context("wordnet", ontology, table, edit.WordNetLexicon())
    entries = sorted(table.heldout_entries())
    aliases = common.entry_aliases(table)
    random_frames = tracks.random_frames(ontology, entries, seed=0)
    concepts, definitions, item_rows = [], [], []
    pool = sorted({t for e in range(len(table.entry_concepts)) for r, a in ctx.frame(e)
                   if ontology["relation_names"][r] == "hypernym" and (t := ctx.text(a))})
    for e in entries:
        surface = sorted(aliases[e], key=len)[-1]
        frame = ctx.frame(e)
        facts = ctx.facts(frame)
        hypernym = (facts.get("hypernym") or ["thing"])[0]
        parts = facts.get("part_meronym", [])
        text = f"{surface}: a kind of {hypernym}" + (f" that has a {parts[0]}" if parts else "") + "."
        cid = f"toyh-{e}"
        concepts.append({"concept": cid, "surface": surface, "item_surface": surface, "entry": e, "frame": ctx.names(frame),
                         "random_frame": ctx.names(random_frames[e]), "own_atom": None, "split": "heldout", "degree": len(frame)})
        definitions.append({"concept": cid, "style": "prose", "text": text, "headword": surface, "source": "test", "licence": "test"})
        wrong = [t for t in pool if t != hypernym][:3]
        candidates = [f" {hypernym}"] + [f" {w}" for w in wrong]
        item_rows.append({"id": f"{cid}-property-hypernym", "concept": cid, "test": "property", "relation": "hypernym",
                          "templates": ["{x} is a kind of", "Every {x} is a type of"], "null": "this", "candidates": candidates, "gold": 0})
    out = root / "heldout"
    out.mkdir()
    e11._write_jsonl(out / "concepts.jsonl", concepts)
    e11._write_jsonl(out / "definitions.jsonl", definitions)
    e11._write_jsonl(out / "items.jsonl", item_rows)
    e11.write_json(out / "manifest.json", e11._manifest("heldout", "wordnet", ["prose"], concepts=len(concepts)))
    return out


@pytest.fixture(scope="module")
def sets(world, items, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e11")
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    manifest = e11.build_new_set(items["new"], root / "new", ctx)
    return {"new": root / "new", "heldout": _heldout_set(world, root), "manifest": manifest, "ctx": ctx}


def _same(a, b) -> bool:
    return all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("offsets", "relations", "fillers"))


def _fake_generator(prompts):
    return ["hypernym: canine\npart_meronym: tail\n" for _ in prompts], 10 * len(prompts), 5 * len(prompts)


def test_new_set_definitions(sets) -> None:
    read_set = e11.load_read_set(sets["new"])
    assert read_set.kind == "new" and read_set.styles == list(rtl.T5_STYLES) and len(read_set.concepts) == 6
    assert len(read_set.definitions) == 18 and sets["manifest"]["definitions"] == 18
    for d in read_set.definitions:
        assert rtl.headword_span(d["text"], d["headword"]) is not None
    assert {i["concept"] for i in read_set.items} <= {c["concept"] for c in read_set.concepts}
    assert read_set.limit(2).concepts == read_set.concepts[:2]


def test_definition_scorer_matches_direct_scoring(world, sets) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=4)
    read_set = e11.load_read_set(sets["heldout"])
    ctx = e11.run_context(run)
    task = e11.read_tasks(read_set, ctx, ["prose"])[0]
    schedule_before = run.composer.schedule
    count_before = run.channel.entry_count
    scorer = e11.DefinitionScorer(run, read_set)
    scores = scorer([task], [[None, task.gold, task.gold, [task.gold[0]]]])[0]
    assert run.channel.entry_count == count_before and _same(run.composer.schedule, schedule_before)
    assert scores[1] == pytest.approx(scores[2], abs=1e-6) and scores[0] != pytest.approx(scores[1], abs=1e-6)
    # The gold-frame variant equals the run's own reading of the text (the headword links to its entry, gold frame).
    logprobs, offsets = run.adapter.token_logprobs([task.text])
    direct = sum(float(logprobs[0][t - 1]) for t in range(1, len(offsets[0])) if offsets[0][t][0] >= task.span[1])
    assert scores[1] == pytest.approx(direct, abs=1e-4)
    assert scorer.unlinked_rows == 0 and scorer.rows == 4


def test_new_word_evaluation_matches_e9_and_restores_everything(world, items, sets, monkeypatch) -> None:
    monkeypatch.setattr(e11, "host_generator", lambda run, **kw: _fake_generator)
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["new"])
    state_before = {k: v.clone() for k, v in run.model.state_dict().items()}
    rows_before = common.entry_rows(run.channel, torch.arange(run.channel.entry_count))
    result = e11.evaluate(run, read_set, methods=list(e11.METHODS), readers=["oracle", "stated", "typeprior", "pattern", "linker", "linker-joint",
                                                                             "linker-all", "host", "random", "none"],
                          styles=["dictionary"], primary_style="dictionary", gradient_lr=1e-3, gradient_factors=(1.0,),
                          gradient_optimizer="sgd", resamples=50, log=QUIET)
    document = result["document"]
    # Weights, rows and the entry count are exactly as before.
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)
    torch.testing.assert_close(common.entry_rows(run.channel, torch.arange(run.channel.entry_count)), rows_before, rtol=0, atol=0)
    assert run.channel.entry_count == world["ontology"]["entry_count"]
    frames = document["frames"]["dictionary"]
    assert frames["oracle"]["f1"] == 1.0 and frames["none"]["empty"] == 6 and frames["linker"]["cost_per_word"]["forward_passes"] > 1
    assert frames["host"]["cost_per_word"]["generated_tokens"] == 5 and frames["linker-joint"]["frames"] == 6
    means = document["items"]["means"]
    for cond in ("dictionary|oracle", "dictionary|none", "dictionary|linker", "dictionary|context", "dictionary|context+linker",
                 "dictionary|gradient×1.0"):
        assert cond in means, cond
    # The oracle and none conditions reproduce the E9 dimension-3 evaluation's `own` and `none` sources.
    e9 = edit.evaluate_new_words(run, items["new"], sources=["own", "none"], fit_entries=20, log=QUIET)
    e9_values = {s: e11.test_values({"all": r["prompts"] + r["statements"]}, "new") for s, r in e9["results"].items()}
    ours = {c: e11.test_values(result["item_rows"][f"dictionary|{c}"], "new") for c in ("oracle", "none")}
    for source, cond in (("own", "oracle"), ("none", "none")):
        for test in ("property", "entailment", "statement_accuracy"):
            assert ours[cond][test] == pytest.approx(e9_values[source][test], abs=1e-6), (cond, test)
    assert document["locality"]["max_abs_row_change"] == 0.0
    gradient = document["gradient"]["×1.0"]
    assert gradient["steps_mean"] >= 1 and np.isfinite(gradient["general_delta_mean"])
    contrasts = {(c["a"], c["b"], c["test"]) for c in document["items"]["contrasts"]}
    assert ("linker", "none", "property") in contrasts and ("context", "linker", "property") in contrasts


def test_heldout_windows_and_frame_swap(world, sets) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["heldout"])
    schedule_before = run.composer.schedule
    result = e11.evaluate(run, read_set, methods=["frames", "persistence", "windows", "locality"], readers=["oracle", "typeprior",
                          "random", "none"], styles=["prose"], primary_style="prose", resamples=50, log=QUIET)
    assert _same(run.composer.schedule, schedule_before) and not run.channel.skip_empty_frames
    windows = result["document"]["windows"]
    assert windows["windows"] > 0 and windows["occurrences"] > 0
    conditions = result["windows"]["conditions"]
    # The oracle frame is what the trained model reads anyway: its after-term loss equals the plain evaluation.
    entry_of = e11.link_entry_of(read_set, run)
    wset = e11.build_window_set(run, [entry_of[c["concept"]] for c in read_set.concepts],
                                {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts})
    plain = e11.window_term_losses(run, Path(run.config["data"]["eval"]), wset)
    np.testing.assert_allclose(conditions["prose|oracle"]["sum_other"], plain["sum_other"], atol=1e-5)
    assert not np.allclose(conditions["prose|none"]["sum_other"], plain["sum_other"])
    np.testing.assert_allclose(conditions["prose|none"]["count_other"], plain["count_other"])
    assert {c["condition"] for c in windows["contrasts"]} >= {"prose|oracle", "prose|random"}
    assert result["document"]["locality"]["max_abs_row_change"] == 0.0
    # Per-term selection counts only the chosen term's occurrences.
    first = next(t for t in range(len(wset.terms)) if plain["count_other"][t] + plain["count_own"][t] > 0)
    only = e11.window_term_losses(run, Path(run.config["data"]["eval"]), wset, select=lambda w, t: t == first)
    assert only["count_other"][first] == plain["count_other"][first] and only["count_other"].sum() == plain["count_other"][first]


def test_text_routes_on_a_model_without_a_channel(world, sets) -> None:
    run = common.open_run(world["runs"]["C0p"], device="cpu", batch_size=8)
    read_set = e11.load_read_set(sets["new"])
    result = e11.evaluate(run, read_set, methods=["persistence", "context", "gradient"], readers=["oracle", "linker", "none"],
                          styles=["dictionary"], primary_style="dictionary", gradient_lr=1e-3, gradient_factors=(1.0,), gradient_optimizer="sgd",
                          resamples=50, log=QUIET)
    means = result["document"]["items"]["means"]
    assert set(means) == {"dictionary|none", "dictionary|context", "dictionary|gradient×1.0"}
    assert result["document"]["readers"] == ["none"] and not result["document"]["compose"]


def test_cli_run_folder_report_and_plan(world, sets, tmp_path, capsys) -> None:
    outputs = []
    for seed in (1, 2):
        out = tmp_path / f"s{seed}"
        e11.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(sets["new"]), "--output", str(out),
                  "--methods", "frames,persistence", "--readers", "oracle,typeprior,random,none", "--styles", "prose",
                  "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--smoke", "--seed", str(seed)])
        for name in ("resolved_config.yaml", "manifest.json", "summary.json", "report.md", "predictions.jsonl", "frames.jsonl"):
            assert (out / name).exists(), name
        assert (out / "report.md").read_text().startswith("**SMOKE TEST")
        outputs.append(out)
    capsys.readouterr()
    e11.main(["report", "--runs", *map(str, outputs), str(tmp_path / "missing"), "--output", str(tmp_path / "report"),
              "--resamples", "50"])
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["missing"] == [str(tmp_path / "missing")] and report["endpoints"] == []
    assert not any(r["a"] == "linker" for r in report["secondary"])          # no linker in these runs
    jobs = e11.plan_jobs(tracks=["t5", "t4"], hosts=["SmolLM2-360M"], seeds=[1, 2, 3])
    names = [j["name"] for j in jobs]
    assert names[0] == "e11-gradient-dev-SmolLM2-360M" and names[-1] == "e11-report"
    assert "e11-t5-new-SmolLM2-360M-C5-s3" in names and "e11-t4-heldout-SmolLM2-360M-P0-s1" in names
    pending = {j["name"] for j in jobs if j.get("after_training")}
    assert "e11-t4-heldout-SmolLM2-360M-C5-s2" in pending and "e11-t5-new-SmolLM2-360M-C5-s2" not in pending
    c5 = next(j for j in jobs if j["name"] == "e11-t4-heldout-SmolLM2-360M-C5-s1")
    command = c5["command"]
    assert "windows" in command[command.index("--methods") + 1] and "linker-joint" in command[command.index("--readers") + 1]
    assert "--window-gradient" in command and c5["tier"] == 1 and c5["priority"] == e11.PLAN_PRIORITY + 1
    new = next(j for j in jobs if j["name"] == "e11-t5-new-SmolLM2-360M-C0p-s1")["command"]
    assert "windows" not in new[new.index("--methods") + 1] and new[new.index("--readers") + 1] == "none"
    assert new[new.index("--styles") + 1] == "glossary,dictionary,prose"
    c2 = next(j for j in jobs if j["name"] == "e11-t5-heldout-SmolLM2-360M-C2-s1")["command"]
    assert c2[c2.index("--methods") + 1] == "persistence,context,windows" and "--gradient-lr-from" not in c2
    assert c2[c2.index("--styles") + 1] == "prose"
    assert all(j["hours"] > 0 for j in jobs if j["name"].startswith("e11-t")) and jobs[-1]["priority"] == e11.PLAN_PRIORITY + 4


def test_fast_continuation_scores_equal_the_shared_scorer(world, items) -> None:
    from vsa_embed.evaluation import channel_probes as cp
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=3)
    prefixes = ["The house cat is a kind of", "A tea kettle has a", "Paracetamol is", "The bike"]
    continuations = [" feline", " lid and a handle", " an analgesic drug", " wheel"]
    sums, counts = e11.continuation_scores(run.adapter, prefixes, continuations)
    np.testing.assert_allclose(sums, cp.continuation_logprob(run.adapter, prefixes, continuations), atol=1e-5)
    reference_sums, reference_counts = edit.continuation_stats(run.adapter, prefixes, continuations)
    np.testing.assert_allclose(sums, reference_sums, atol=1e-5) and np.testing.assert_array_equal(counts, reference_counts)
    with e11.fast_continuations():
        assert cp.continuation_logprob is not None and edit.continuation_stats is e11.continuation_scores
    assert edit.continuation_stats is not e11.continuation_scores                # restored


def test_headwords_are_link_checked(world) -> None:
    from transformers import AutoTokenizer
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    aliases = common.entry_aliases(world["table"])
    entries = sorted(world["table"].heldout_entries())
    names = {entries[0]: sorted(aliases[entries[0]])[0], entries[1]: "not an alias at all", entries[2]: sorted(aliases[entries[1]])[0]}
    chosen = e11.linkable_headwords(ctx, names, tokenizer, min_subtokens=1)
    assert chosen[entries[0]] == (names[entries[0]], "name")
    assert chosen[entries[1]][1] == "alias" and chosen[entries[1]][0] in aliases[entries[1]]
    assert chosen[entries[2]][1] == "alias" and chosen[entries[2]][0] in aliases[entries[2]]   # its name links to another entry


def test_gradient_dev_writes_the_lr_that_evaluate_reads(world, sets, tmp_path, capsys) -> None:
    e11.main(["gradient-dev", "--run", str(world["runs"]["C0p"]), "--items", str(sets["new"]), "--output", str(tmp_path / "dev"),
              "--primary-style", "dictionary", "--lrs", "1e-4", "1e-3", "--gradient-optimizer", "sgd", "--device", "cpu",
              "--batch-size", "8"])
    chosen = json.loads((tmp_path / "dev" / "gradient_lr.json").read_text())
    assert chosen["lr"] in {1e-4, 1e-3} and set(chosen["scores"]) == {"0.0001", "0.001"} and "property" in chosen["no_update"]
    capsys.readouterr()
    e11.main(["evaluate", "--run", str(world["runs"]["C0p"]), "--items", str(sets["new"]), "--output", str(tmp_path / "eval"),
              "--methods", "gradient", "--readers", "none", "--styles", "dictionary", "--gradient-lr-from",
              str(tmp_path / "dev" / "gradient_lr.json"), "--gradient-factors", "1.0", "--gradient-optimizer", "sgd",
              "--device", "cpu", "--batch-size", "8", "--resamples", "50"])
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert summary["gradient"]["lr"] == chosen["lr"] and (tmp_path / "eval" / "gradient_records.json").exists()


def test_teacher_reads_open_licence_text_only(tmp_path) -> None:
    def read_set(licence: str) -> e11.ReadSet:
        return e11.ReadSet(tmp_path, {"licence": [licence]}, [], [{"concept": "c", "licence": licence}], [])
    assert e11.teacher_allowed(read_set(e11.LICENCES["chebi"])) and e11.teacher_allowed(read_set(e11.LICENCES["mesh"]))
    assert e11.teacher_allowed(read_set(e11.LICENCES["t5"]))
    for closed in (e11.LICENCES["openstax"], "SNOMED CT International (credentialed)", "MIMIC-IV notes", "UMLS 2022AB"):
        assert not e11.teacher_allowed(read_set(closed))
    assert not e11.teacher_allowed(e11.ReadSet(tmp_path, {}, [], [], []))
