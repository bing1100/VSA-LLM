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


def test_algebra_experiments_quick() -> None:
    from vsa_embed.experiments import e14_algebra as A
    order = {r["encoding"]: r["accuracy"] for r in A.a4(1, quick=True) if r["test"] == "order" and r["width"] == 1024}
    assert order["perm"] > 0.95 and 0.35 < order["conv"] < 0.65                 # convolution is commutative; a permutation is not
    assert all(r["auc"] > 0.9 for r in A.a5(1, quick=True) if r["width"] == 1024)
    assert all(r["le_decode_compare"] > 0.95 for r in A.a6(1, quick=True))
    c4 = A.c4(1, quick=True)
    assert all(r["accuracy"] == 1.0 for r in c4 if r["method"] == "amie-mining" and r["k"] >= 3)
    facts = A.a3_world(1, n_ent=200)
    rules = A.a3_rules("chain", 2, 8, __import__("random").Random(0))
    assert A.apply_rules(facts, rules)["h0"] == R.datalog(facts, rules)["h0"]
    p, r, f = A.prf({1, 2}, {2, 3})
    assert (p, r, f) == (0.5, 0.5, 0.5)


# -- addendum 8 (pre-registration §19): LG-D2 noisy records, X8b irregular trees, X7b statistics, LG-P3b retrieval at scale --

def test_missing_split_and_soft_reference() -> None:
    attrs = [{"name": "hue", "values": ["a", "b", "c", "d"]}, {"name": "size score", "values": None}]
    p, yes, no = R.missing_split({"attr": "hue", "op": "is", "value": "b"}, attrs, {})
    assert p == 0.25 and yes["hue"] == {"b"} and no["hue"] == {"a", "c", "d"}
    p2, _, _ = R.missing_split({"attr": "hue", "op": "is", "value": "b"}, attrs, no)               # already excluded on this path
    assert p2 == 0.0
    q, yes, no = R.missing_split({"attr": "size score", "op": "le", "value": 40}, attrs, {})
    assert abs(q - 41 / 101) < 1e-12 and yes["size score"] == (0, 40) and no["size score"] == (41, 100)
    tree = {"test": {"attr": "hue", "op": "is", "value": "b"}, "yes": {"leaf": "X"},
            "no": {"test": {"attr": "size score", "op": "le", "value": 40}, "yes": {"leaf": "Y"}, "no": {"leaf": "X"}}}
    dist = R.tree_leaf_distribution(tree, {"hue": None, "size score": 70}, attrs)
    assert abs(dist["X"] - 1.0) < 1e-12                                                            # both branches end in X here
    dist = R.tree_leaf_distribution(tree, {"hue": None, "size score": None}, attrs)
    assert abs(sum(dist.values()) - 1) < 1e-12 and abs(dist["Y"] - 0.75 * 41 / 101) < 1e-12
    assert R.run_tree_missing(tree, {"hue": None, "size score": 10})[0] == "Y"                      # missing: the otherwise branch


def test_noisy_trees_items_and_stores() -> None:
    world = W.noisy_trees(2, depths=(2, 4), inputs_per_depth=6, namer=NAMER(42))
    for item in world["items"]:
        assert R.run_tree(item["tree"], item["record"])[0] == item["answer"]
        lo, hi = item["noisy"]["0.1"]["corrupted"], item["noisy"]["0.3"]["corrupted"]
        assert set(lo) <= set(hi) and all(hi[a] == k for a, k in lo.items())                     # coupled noise levels
        for p, nz in item["noisy"].items():
            for a, kind in nz["corrupted"].items():
                assert (nz["record"][a] is None) == (kind == "missing") and nz["record"][a] != item["record"][a]
            store = S.NoisyTreeStore(item, nz["record"], width=1024, seed=1)
            assert store.execute().answer == R.run_tree_missing(item["tree"], nz["record"])[0]     # presence detection works
            _, soft, dist = store.execute_soft(item["entity"])
            exact = R.tree_leaf_distribution(item["tree"], nz["record"], item["attributes"])
            assert abs(exact.get(soft, 0.0) - max(exact.values())) < 1e-6 and abs(sum(dist.values()) - 1) < 1e-6
    rows, stats = items_mod.noisy_trees_set(2, scale=0.02, width=512, namer=NAMER(42))
    assert rows and stats["ceilings"]["path-vsa"]["accuracy"] >= 0.9
    for r in rows:
        assert len(r["options"]) == 6 and 0 <= r["gold"] < 6 and len(r["contexts"]) == 12
        assert all(0 <= v < 6 for p in r["noise"] for v in r["noise"][p]["exec"].values())
    assert "trees-noisy" not in items_mod.BUILDERS                                                  # `build`'s default sets unchanged


def test_stable_codebook_is_process_independent() -> None:
    import subprocess
    code = "from vsa_embed.logic.store import StableCodebook, Codebook; print(StableCodebook(64, 1).filler('abc')[:3].tolist())"
    outs = set()
    for salt in ("1", "2"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": salt, "CUDA_VISIBLE_DEVICES": ""}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
    assert len(outs) == 1


def test_x8b_layouts_quick() -> None:
    from vsa_embed.experiments import e14_concat as C
    import random
    rng = random.Random(0)
    for fam in C.X8B_FAMILIES:
        tree = C.x8b_tree(rng, fam, 8)
        shape = C.x8b_shape(tree)
        assert shape["depth"] == 8 and shape["leaves"] == shape["internal"] + 1
        nodes, sizes, _ = C._preorder(tree)
        assert sizes[0] == len(nodes) and all(sizes[i] == 1 for i, n in enumerate(nodes) if "leaf" in n)
        xs, classes, lengths = C.x8b_inputs(rng, tree, 20)
        assert len(xs) == 20 and max(lengths) <= 8
    rows = C.x8b(1, depths=(8,), families=("spine", "gw"), trees_per_cell=1, inputs_per_tree=20)
    acc = {(r["family"], r["layout"], r["block_width"]): r["accuracy"] for r in rows}
    assert all(acc[(f, lay, w)] == 1.0 for f in ("spine", "gw") for lay in ("preorder-concat", "heap-concat") for w in (64, 256))
    assert all(acc[(f, "preorder-vsa", 256)] >= 0.95 for f in ("spine", "gw"))


def test_x7b_cluster_bootstrap() -> None:
    import numpy as np
    from vsa_embed.experiments import e14_concat as C
    zero = C.cluster_bootstrap(np.zeros(10), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], resamples=200)
    assert zero["mean"] == 0 and zero["p_value"] == 1.0 and zero["n_entries"] == 5
    pos = C.cluster_bootstrap(np.ones(40) * 0.1, list(range(20)) * 2, resamples=200)
    assert abs(pos["mean"] - 0.1) < 1e-12 and pos["ci_low"] > 0 and pos["p_value"] < 0.05


def test_p3b_names_and_retrieval_small() -> None:
    assert "abd" in prover.edits1("abcd") and "abxcd" in prover.edits1("abcd") and "abcd" not in prover.edits1("abcd")
    for regime in prover.P3B_NAMES:
        names = prover.p3b_class_names(1, 40, regime)
        assert len(names) == len(set(names)) == 40
    nd = prover.p3b_class_names(1, 40, "near-dup")
    assert all(v in prover.edits1(nd[i - i % 4]) for i, v in enumerate(nd) if i % 4)                # clusters of one-edit variants
    rows = prover.retrieval_scale_study(1, sizes=(50,), widths=(512,), queries=20, log=lambda *a: None)
    get = {(r["names"], r["codes"], r["query"], r["method"]): r for r in rows}
    for regime in prover.P3B_NAMES:
        assert get[(regime, "-", "exact", "exact")]["recall_at_5"] == 1.0
        assert get[(regime, "-", "typo", "exact")]["recall_at_5"] == 0.0
        assert get[(regime, "random", "exact", "vsa")]["recall_at_5"] >= 0.95
    assert get[("independent", "-", "typo", "exact+edit1")]["recall_at_5"] > 0.9
