"""E11-M (many terms read once each) on the tiny E9 world (CPU): the cached context prefix against joint scoring, the
budgeted definitions context, BM25 retrieval, passages and their after-mention losses, every route end to end with its
invariants (frame rows independent of N, weights restored after the gradient pass), the pooled report and the plan."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401
from test_e11_read_to_learn import sets  # noqa: E402,F401

from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e11_many as many  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")
QUIET = lambda message: None  # noqa: E731


def test_prefix_cache_equals_joint_scoring(world) -> None:
    run = common.open_run(world["runs"]["C0p"], device="cpu", batch_size=3)
    context = "Glossary: the tea kettle is a container.\n"
    prompts, continuations = ["The house cat is a kind of", "A bike has a"], [" feline", " wheel"]
    cache = e11.PrefixCache(run, run.adapter, context)
    cached, counts = e11.continuation_scores(run.adapter, prompts, continuations, prefix=cache)
    joint, joint_counts = e11.continuation_scores(run.adapter, [context + p for p in prompts], continuations)
    np.testing.assert_allclose(cached, joint, atol=1e-4) and np.testing.assert_array_equal(counts, joint_counts)
    empty = e11.PrefixCache(run, run.adapter, "")
    np.testing.assert_allclose(e11.continuation_scores(run.adapter, prompts, continuations, prefix=empty)[0],
                               e11.continuation_scores(run.adapter, prompts, continuations)[0], atol=1e-5)
    c5 = common.open_run(world["runs"]["C5"], device="cpu", batch_size=3)        # with channel spans: close, not exact
    cached5, _ = e11.continuation_scores(c5.adapter, prompts, continuations, prefix=e11.PrefixCache(c5, c5.adapter, context))
    joint5, _ = e11.continuation_scores(c5.adapter, [context + p for p in prompts], continuations)
    np.testing.assert_allclose(cached5, joint5, atol=0.3)


def test_definitions_context_and_bm25() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    defs = [(f"c{i}", f"term{i}zork: a kind of thing number {i}.") for i in range(10)]
    small = many.definitions_context(defs, tokenizer, 40)
    assert small["text"].startswith(many.HEADER) and small["tokens"] <= 40 and 0 < len(small["present"]) < 10
    full = many.definitions_context(defs, tokenizer, 10_000)
    assert sorted(full["present"]) == sorted(c for c, _ in defs) and full["tokens"] == full["all_tokens"]
    assert many.definitions_context(defs, tokenizer, 40)["present"] == small["present"]          # seeded order
    retriever = many.BM25([text for _, text in defs])
    assert retriever.top("What is term7zork?", 1) == [7]
    context, got = many.retrieved_context(retriever, [c for c, _ in defs], [t for _, t in defs], "term3zork and term5zork", 2)
    assert set(got) == {"c3", "c5"} and context.count("\n") == 3


def test_t5_passages_mention_every_term() -> None:
    terms = [{"concept": "a", "surface": "Zorbix Hub"}, {"concept": "b", "surface": "Quelt Feed"}]
    facts = {"a": {"is_a": ["system"], "owned_by": ["the Brix Squad"], "status": ["active"]},
             "b": {"is_a": ["dataset"], "area": ["finance"], "tier": ["tier 2"]}}
    text, mentions = many.t5_passage(terms, facts, random.Random(0))
    assert {m[0] for m in mentions} == {"a", "b"} and all(text[s:e].lower() in {"zorbix hub", "quelt feed"} for _, s, e in mentions)


@pytest.fixture(scope="module")
def episodes(world, sets, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e11m")
    new = many.build_many_set(sets["new"], root / "new", sizes=[2, 4, 6], style="dictionary")
    held = many.build_many_set(sets["heldout"], root / "held", sizes=[2, 3], style="prose")
    return {"new": root / "new", "held": root / "held", "new_manifest": new, "held_manifest": held}


def test_build_many_sets(episodes) -> None:
    m = episodes["new_manifest"]
    assert m["sizes"] == [2, 4, 6] and m["kind"] == "new" and m["passages"] == 0          # passages: T5 sets only
    loaded = many.load_many(episodes["new"])
    assert len(loaded.order) == 6 and [c["concept"] for c in loaded.read_set.concepts] == loaded.order
    assert {d["style"] for d in loaded.read_set.definitions} == {"dictionary"}
    held = many.load_many(episodes["held"])
    assert held.kind == "heldout" and len(held.order) == 3 and episodes["held_manifest"]["sizes"] == [2, 3]


def test_every_route_end_to_end(world, episodes) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=4)
    state = {k: v.clone() for k, v in run.model.state_dict().items()}
    m = many.load_many(episodes["new"])
    for d in m.read_set.definitions:                 # the tiny host has 64 positions: short definitions keep contexts inside them
        d["text"] = f"{d['headword']}: a short note."
    m.passages = [{"id": "p", "size": 4, "terms": m.order[:2], "text": f"{m.read_set.concepts[0]['surface']} is new. "
                   f"{m.read_set.concepts[1]['surface']} is old too.", "mentions": []}]
    for c in m.read_set.concepts[:2]:
        start = m.passages[0]["text"].lower().index(c["surface"].lower())
        m.passages[0]["mentions"].append([c["concept"], start, start + len(c["surface"])])
    result = many.evaluate_many(run, m, routes=list(many.ROUTES), readers=["oracle", "typeprior", "linker", "none"], budgets=[30],
                                rag_k=[1], gradient_lr=1e-3, gradient_optimizer="sgd", log=QUIET)
    for key, value in run.model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert run.channel.entry_count == world["ontology"]["entry_count"]
    assert set(result["items"]) >= {"frame:oracle", "frame:typeprior", "frame:linker", "none"}
    assert {("context:B30", n) for n in (2, 4, 6)} <= set(result["items_n"]) and ("gradient", 6) in result["items_n"]
    assert result["costs"]["interference"]["max_abs_margin_change"] < 1e-5                    # rows do not depend on N
    assert result["presence"]["context:B30|N6"]["tokens"] <= 30 and result["recall"]["rag:k1|N6"] is not None
    assert ("none", 4) in result["passages"] and ("gradient", 4) in result["passages"] and ("rag:k1", 4) in result["passages"]
    assert all(c > 0 for p in result["passages"][("none", 4)] for _, c in p)
    rows = result["items_n"][("context:B30", 6)]
    assert any(r.get("definition_in_context") for rs in rows.values() for r in rs)
    summary = many.summarize(result, m, resamples=50)
    assert summary["means"]["frame:oracle"]["6"]["property"] is not None and summary["forgetting_first_terms_property"]
    assert any(c["a"] == "frame:oracle" and c["b"] == "none" for c in summary["contrasts"])
    text = many.render({"source": run.describe(), "track": "wordnet", "kind": "new", "routes": list(many.ROUTES)}, summary, result)
    assert "Property accuracy by number of terms" in text


def test_natural_windows_route(world, episodes) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=4)
    m = many.load_many(episodes["held"])
    result = many.evaluate_many(run, m, routes=["frames", "context"], readers=["oracle", "none"], budgets=[24], rag_k=[1],
                                gradient_lr=None, log=QUIET)
    assert all((c, n) in result["windows"] for c in ("frame:oracle", "none", "context:B24") for n in m.sizes)
    for data in result["windows"].values():
        assert data["sum_other"].shape == (len(m.order),)


def test_cli_report_and_plan(world, episodes, tmp_path, capsys) -> None:
    outputs = []
    for seed in (1, 2):
        out = tmp_path / f"s{seed}"
        many.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(episodes["new"]), "--output", str(out),
                   "--routes", "frames,context", "--readers", "oracle,none", "--budgets", "30", "--device", "cpu", "--batch-size", "4",
                   "--resamples", "50", "--smoke"])
        assert (out / "report.md").read_text().startswith("**SMOKE TEST") and (out / "predictions.jsonl").exists()
        outputs.append(out)
    capsys.readouterr()
    many.main(["report", "--runs", *map(str, outputs), str(tmp_path / "gone"), "--output", str(tmp_path / "report"), "--resamples", "50"])
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["missing"] == [str(tmp_path / "gone")] and any(r["a"] == "frame:oracle" and r["seeds"] == 2 for r in report["rows"])
    jobs = many.plan_jobs(seeds=[1, 2, 3])
    names = [j["name"] for j in jobs]
    assert "e11-many-t5-SmolLM2-360M-C5-s1" in names and names[-1] == "e11-many-report"
    assert {j["name"] for j in jobs if j.get("after_training")} == {f"e11-many-t4-SmolLM2-360M-{m}-s{s}" for m in ("C5", "C0p") for s in (2, 3)}
    c0 = next(j for j in jobs if j["name"] == "e11-many-t5-SmolLM2-360M-C0p-s1")["command"]
    assert c0[c0.index("--readers") + 1] == "none"
