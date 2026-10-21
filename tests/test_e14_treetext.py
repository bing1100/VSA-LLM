"""E14 addendum 9 (tree-text, CPU): the prose builder against the exact reference, the template-regex reader and the tolerant
parser, the scorer, and the read / report commands on a tiny random host."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from vsa_embed.logic import reference as R  # noqa: E402
from vsa_embed.logic import world as W  # noqa: E402
from vsa_embed.experiments import e14_treetext as T  # noqa: E402

NAMER = lambda seed=1: W.Namer(seed, forbidden=set())  # noqa: E731


def _small():
    return T.build_items(1, per_depth=4, records=6, namer=NAMER(7))


def test_builder_is_deterministic_and_matches_the_reference() -> None:
    items, demos = _small()
    again, _ = _small()
    assert json.dumps(items, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert len(items) == 12 and sorted({i["depth"] for i in items}) == [2, 3, 4] and len(demos) == 2
    for it in items:
        assert all(R.run_tree(it["tree"], r["record"])[0] == r["gold"] for r in it["records"])
        assert it["text"]["templated"] != it["text"]["paraphrased"]
        for style in T.STYLES:                                                # every outcome's desk is named in the prose
            text = it["text"][style]
            assert all(T._DESK.search(leaf).group(1) in text for leaf in T.leaves(it["tree"]))
    assert {d["id"] for d in demos}.isdisjoint({i["id"] for i in items})


def test_regex_reader_and_parser_are_exact_on_their_formats() -> None:
    items, _ = _small()
    for it in items:
        assert T.score_tree(T.regex_reader(it["text"]["templated"]), it)["exact"] == 1
        assert T.score_tree(T.parse_tree(it["target"]), it)["exact"] == 1
        assert T.regex_reader(it["text"]["paraphrased"]) is None or T.score_tree(T.regex_reader(it["text"]["paraphrased"]), it)["exact"] == 0
    bl = T.baseline_scores(items)
    assert bl["regex:templated"]["leaf_accuracy"] == 1.0 and bl["gold"]["exact"] == 1.0 and bl["first-leaf"]["exact"] == 0.0


def test_parser_tolerance_and_failures() -> None:
    loose = "if plonk <= 40:\nSend it to the varn desk\nelse:\n  IF the glim is zork:\n    then route it to the Pim desk.\n  Otherwise:\n    then send it to the zed desk.\n"
    tree = T.parse_tree(loose)
    assert tree["test"] == {"attr": "plonk", "op": "le", "value": 40} and tree["yes"]["leaf"] == "Send it to the varn desk"
    assert tree["no"]["test"] == {"attr": "glim", "op": "is", "value": "zork"} and T.canon_leaf(tree["no"]["yes"]["leaf"]) == "send it to the pim desk"
    neg = T.parse_tree("If its plonk score is above 40:\n  then a.\nOtherwise:\n  If its glim is not zork:\n    then b.\n  Otherwise:\n    then c.\n")
    assert neg["test"] == {"attr": "plonk score", "op": "le", "value": 40} and neg["yes"]["test"] == {"attr": "glim", "op": "is", "value": "zork"}
    assert neg["no"]["leaf"] == "a" and neg["yes"]["yes"]["leaf"] == "c" and neg["yes"]["no"]["leaf"] == "b"
    assert T.parse_tree("") is None
    assert T.parse_tree("If its plonk is varn:\n  then send it to the a desk.\n") is None       # missing otherwise
    assert T.parse_tree("Otherwise:\n  then x.") is None
    assert T.parse_tree("then send it to the a desk.\nIf its p is q:\n garbage") == {"leaf": "send it to the a desk"}   # trailing text ignored


def test_scorer_aligns_by_path_and_executes_the_decoded_tree() -> None:
    items, _ = _small()
    it = next(i for i in items if i["depth"] == 2)
    wrong = W._copy(it["tree"])
    wrong["no"]["no"] = {"leaf": "send it to the nowhere desk"}
    s = T.score_tree(wrong, it)
    assert s["parsed"] == 1 and s["exact"] == 0 and s["node_accuracy"] == 1.0 and s["leaf_node_accuracy"] == 0.75
    expected = sum(int(R.run_tree(it["tree"], r["record"])[0] == R.run_tree(wrong, r["record"])[0]) for r in it["records"]) / len(it["records"])
    assert s["leaf_accuracy"] == expected
    unknown = {"test": {"attr": "no such attribute", "op": "is", "value": "x"}, "yes": {"leaf": "a"}, "no": {"leaf": "b"}}
    assert T.score_tree(unknown, it)["leaf_accuracy"] == 0.0 and T.score_tree(None, it) == {**T.score_tree(None, it), "parsed": 0, "leaf_accuracy": 0.0}
    numeric = next(i for i in items if i["numeric"] and any(a["values"] is None for a in i["attributes"]))
    attr = next(a["name"] for a in numeric["attributes"] if a["values"] is None)
    short = {"test": {"attr": attr.replace(" score", ""), "op": "le", "value": 50}, "yes": {"leaf": "a"}, "no": {"leaf": "b"}}
    assert T.resolve_attr(attr.replace(" score", ""), numeric["attributes"]) == attr
    assert T.run_decoded(short, numeric["records"][0]["record"], numeric["attributes"]) in {"a", "b"}


def test_read_and_report_on_a_tiny_host(tmp_path) -> None:
    pytest.importorskip("transformers")
    import transformers
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not available")
    host_dir = tmp_path / "host"
    tok.save_pretrained(host_dir)
    transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=4096, n_embd=32, n_layer=1, n_head=2)).save_pretrained(host_dir)
    items, demos = _small()
    items_dir = tmp_path / "items"
    T.write_items(items, demos, items_dir, seed=1, config={})
    manifest = json.loads((items_dir / "manifest.json").read_text())
    assert manifest["items"] == 12 and set(manifest["files"]) == {"items.jsonl", "demos.json"}
    out = tmp_path / "run"
    T.main(["read", "--host", str(host_dir), "--items", str(items_dir), "--output", str(out), "--depths", "2", "--limit", "2",
            "--device", "cpu", "--batch-size", "2", "--margin", "0.3", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["items"] == 2 and set(summary["accuracy"]) == set(T.STYLES)
    rows = [json.loads(ln) for ln in (out / "generations.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(0.0 <= r["leaf_accuracy"] <= 1.0 for r in rows)
    T.main(["report", "--runs", str(out), "--items", str(items_dir), "--output", str(tmp_path / "report"), "--resamples", "50"])
    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["decision"]["smoke"] is True and report["decision"]["worth_queuing_more"] is False
    assert report["baselines"]["regex:templated"]["leaf_accuracy"]["mean"] == 1.0
    assert "templated_minus_paraphrased" in next(iter(report["hosts"].values()))["contrasts"]
