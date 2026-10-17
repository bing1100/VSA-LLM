"""E14 logic (CPU): world generators against the exact reference, the VSA executors against the reference, item sets,
the scorer and report on a tiny host, and the prover's soundness and completeness."""

import collections
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vsa_embed.logic import reference as R  # noqa: E402
from vsa_embed.logic import store as S  # noqa: E402
from vsa_embed.logic import world as W  # noqa: E402
from vsa_embed.experiments import e14_items as items_mod  # noqa: E402
from vsa_embed.experiments import e14_prover as prover  # noqa: E402

NAMER = lambda seed=1: W.Namer(seed, forbidden=set())  # noqa: E731


def test_reference_prover_cases() -> None:
    mp = [{"kind": "fact", "cls": "a", "neg": False}, {"kind": "rule", "body": "a", "head": "b", "neg": False}]
    assert R.prove(mp, {"cls": "b", "neg": False})["proved"] and not R.prove(mp, {"cls": "c", "neg": False})["proved"]
    cases = [{"kind": "or", "cls": ["a", "b"]}, {"kind": "rule", "body": "a", "head": "g", "neg": False},
             {"kind": "rule", "body": "b", "head": "g", "neg": False}]
    assert R.prove(cases, {"cls": "g", "neg": False})["proved"] and not R.prove(cases[:2], {"cls": "g", "neg": False})["proved"]
    tollens = [{"kind": "rule", "body": "a", "head": "b", "neg": False}, {"kind": "fact", "cls": "b", "neg": True}]
    assert R.prove(tollens, {"cls": "a", "neg": True})["proved"]
    db = R.datalog({"p": {(1, 2), (2, 3)}}, [{"shape": "chain", "body": ["p", "p"], "head": "q"},
                                             {"shape": "inv", "body": ["p"], "head": "r"}])
    assert db["q"] == {(1, 3)} and db["r"] == {(2, 1), (3, 2)}


def test_generators_agree_with_the_reference() -> None:
    tx = W.taxonomy(1, questions_per_depth=6, namer=NAMER())["items"]
    assert all(R.taxonomy_answer(i) == i["label"] for i in tx)
    assert collections.Counter(i["label"] for i in tx)[True] == len(tx) // 2
    fo = W.fol(1, per_depth=6, namer=NAMER())["items"]
    designed = [(i["kind"], (int(i["id"].rsplit("-", 1)[1]) // 3) % 2 == 0) for i in fo]
    assert all(i["label"] == provable for i, (_, provable) in zip(fo, designed))
    for smp in ("rp", "lp"):
        pr = W.propositional(1, sampler=smp, items_per_depth=6, namer=NAMER())["items"]
        assert pr and all((i["query"] in W._forward(set(i["facts"]), [(tuple(b), h) for b, h in i["rules"]])) == i["label"] for i in pr)


def test_executors_match_the_reference_at_working_width() -> None:
    nm = NAMER(2)
    tx = W.taxonomy(2, questions_per_depth=6, namer=nm)["items"]
    assert all(S.TaxonomyStore(i, width=1024).execute().answer == i["label"] for i in tx)
    assert all(S.taxonomy_symbolic(i).answer == i["label"] for i in tx)
    shuffled = sum(S.TaxonomyStore(i, width=1024, shuffle=True).execute().answer == i["label"] for i in tx) / len(tx)
    assert shuffled < 0.9
    cw = W.composition(2, items_per_k=4, namer=nm)
    assert all(S.CompositionStore(cw, i, width=1024).execute().answer == i["answer"] for i in cw["items"])
    pr = W.propositional(2, sampler="rp", items_per_depth=4, namer=nm)["items"]
    assert all(S.PropositionStore(i, width=1024).execute().answer == i["label"] for i in pr)
    tr = W.trees(2, depths=(2, 3), inputs_per_depth=4, list_lengths=(3,), inputs_per_list=4, edits=2, namer=nm)["items"]
    assert all(S.TreeStore(i, width=1024).execute().answer == i["answer"] for i in tr)
    assert all(S.tree_symbolic(i).answer == i["answer"] for i in tr)
    edit = next(i for i in tr if i["kind"] == "edit")
    assert R.run_tree(edit["base_tree"], edit["record"])[0] == edit["before"] != edit["answer"]
    nb = S.TaxonomyStore(tx[0], width=1024, binding=False).execute()
    assert nb.answer is None and all("associated with" in s for s, _ in nb.lines)


def test_item_sets_are_well_formed(tmp_path) -> None:
    for name, builder in items_mod.BUILDERS.items():
        rows = builder(3, scale=0.03, width=512, namer=NAMER(3))
        assert rows, name
        for r in rows:
            assert 0 <= r["gold"] < len(r["options"]) and r["question"] and r["contexts"], name
            assert all(isinstance(v, str) for v in r["contexts"].values())
    path = tmp_path / "x.jsonl.gz"
    d1 = items_mod.write_set(rows, path)
    assert items_mod.read_set(path) == [json.loads(json.dumps(r, sort_keys=True)) for r in rows]
    assert items_mod.write_set(rows, tmp_path / "y.jsonl.gz") == d1                                       # deterministic


def test_prover_policies_are_sound_and_the_ceiling_is_complete() -> None:
    fo = W.fol(4, per_depth=6, namer=NAMER(4))["items"]
    for item in fo:
        problem = prover.Problem(item)
        for name in ("random", "sos", "bfs"):
            r = prover.run_policy(problem, name, 30, seed=0)
            assert not (r["proved"] and not item["label"]), (name, item["id"])                          # sound
        ceiling = prover.run_policy(problem, "ceiling", 30)
        assert ceiling["proved"] == item["label"], item["id"]                                             # complete here
    accept = prover.Problem(next(i for i in fo if not i["label"]))
    s = prover.Search(budget=5)
    assert prover.dfs(accept, accept.goal, frozenset(), lambda p, g, a, st, se: [len(st) - 1], s, 0, frozenset(), allow_accept=True)
    assert s.accepted                                                                                     # unsound accept closes goals


def test_score_and_report_on_a_tiny_host(tmp_path) -> None:
    pytest.importorskip("transformers")
    from test_e14_reading import _ok  # noqa: F401
    import transformers
    from vsa_embed.experiments import e14_score
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not available")
    host_dir = tmp_path / "host"
    tok.save_pretrained(host_dir)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=1024, n_embd=32, n_layer=1, n_head=2)).save_pretrained(host_dir)
    items_dir = tmp_path / "items"
    for name in ("taxonomy", "trees"):
        rows = items_mod.BUILDERS[name](1, scale=0.02, width=512, namer=NAMER(5))
        items_mod.write_set(rows, items_dir / f"{name}-s1.jsonl.gz")
    out = tmp_path / "run"
    e14_score.main(["score", "--host", str(host_dir), "--items", str(items_dir), "--output", str(out), "--data", str(tmp_path / "data"),
                    "--sets", "taxonomy,trees", "--device", "cpu", "--batch-size", "8", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert any(k.startswith("taxonomy|tf|exec-vsa") for k in summary["accuracy"])
    e14_score.main(["report", "--runs", str(out), "--output", str(tmp_path / "report"), "--resamples", "50"])
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["hosts"] and "B1_primary" in next(iter(report["hosts"].values()))
