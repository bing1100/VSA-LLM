"""E14 RD-4e (pre-registration addendum 12, CPU): the router hold-out guarantee (paraphrase demonstrations never show the
routed wording or a near-duplicate), the size-rule assembly, the abstention threshold rule, prefix-cached few-shot scoring
against the brute path, and a labelled smoke of `--family rd4e` on the tiny E9 world."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401  (module fixtures re-used)
from test_e14_reading import heldout  # noqa: E402,F401

from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402
from vsa_embed.experiments import e14_reading as rd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T4_ITEMS = REPO / "experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1/items.jsonl"
T5_ITEMS = REPO / "experiments/e9-retrofit/items/new-words-t5-smollm2-v1/items.jsonl"


def _bank(path: Path) -> dict[str, list[str]]:
    bank: dict[str, list[str]] = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("relation"):
            for t in row["templates"]:
                bank.setdefault(row["relation"], [])
                if t not in bank[row["relation"]]:
                    bank[row["relation"]].append(t)
    return bank


# -- the hold-out guarantee ---------------------------------------------------------------------------------------------

def test_near_duplicate_rule() -> None:
    assert rd.near_duplicate("{x} has a role as", "{x} has a role as a")
    assert rd.near_duplicate("{x} has a role as a", "{x} has a role as an")
    assert rd.near_duplicate("{x} is part of", "{x} forms part of")                      # one content word more
    assert not rd.near_duplicate("{x} is owned by", "{x} is run by")                     # different content words
    assert not rd.near_duplicate("{x} is a member of the class of", "Chemically, {x} is classified among the")
    assert rd.wording_content("{x} is a") == frozenset()


@pytest.mark.parametrize("path", [T4_ITEMS, T5_ITEMS])
def test_demonstrations_never_hold_the_routed_wording(path: Path) -> None:
    bank = _bank(path)
    seen = 0
    for relation, templates in bank.items():
        if relation not in rd.ROUTE_PARAPHRASES:
            continue
        paraphrases = rd.ROUTE_PARAPHRASES[relation]
        assert 4 <= len(paraphrases) <= 6
        for t in templates:
            for p in paraphrases:                      # hand-written wordings are never near-duplicates of an item template
                assert not rd.near_duplicate(p, t), (relation, p, t)
        for q in templates:
            demos = rd.demonstration_bank(relation, templates, q)
            assert q not in demos and all(not rd.near_duplicate(d, q) for d in demos)
            assert len({rd.wording_content(d) for d in demos}) == len(demos)             # no repeated content
            assert len(demos) >= 5 and set(paraphrases) <= set(demos)
            for key in ("item-a", "item-b", "item-c"):
                shots = rd.rotate_shots(demos, key)
                assert len(shots) == rd.PARA_SHOTS and q not in shots and len(set(shots)) == len(shots)
            seen += 1
    assert seen >= 9
    # every wording of the bank is shown for some item: the rotation covers the whole bank
    relation = "owned_by" if "owned_by" in bank else "has_role"
    demos = rd.demonstration_bank(relation, bank[relation], "")
    covered = {s for i in range(50) for s in rd.rotate_shots(demos, f"e9n-{i:04d}-property-owned_by")}
    assert covered == set(demos)


def test_grammar_paraphrases_and_prompt_shape() -> None:
    bank = rd.demonstration_bank("hypernym", ["{x} is a kind of", "Every {x} is a type of"], "{x} is a kind of")
    assert "{x} is a kind of" not in bank and "Every {x} is a type of" in bank and len(bank) == 6
    head, tail = rd.fewshot_prompt([("{x} has a role as", "has role")], "{x} is a kind of", "limonene")
    assert head.startswith("Each question asks for one kind of fact.\nQuestion: X has a role as ...\nFact: X has role\n")
    assert tail == "Question: limonene is a kind of ...\nFact: limonene"


# -- abstention and the size rule ------------------------------------------------------------------------------------------

def test_abstain_threshold_is_the_quantile_of_correct_margins() -> None:
    margins = [0.1, 0.2, 0.3, 0.4, 0.5, 5.0]
    correct = [True, True, True, True, True, False]
    assert rd.abstain_threshold(margins, correct) == pytest.approx(np.quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.2))
    assert rd.abstain_threshold([0.3], [False]) is None
    assert rd.abstain_threshold([float("inf"), 0.3], [True, True]) == pytest.approx(0.3)
    assert rd.ranked({3: 0.1, 1: 0.5, 2: 0.5}) == [1, 2, 3] and rd.margin_of({1: 0.5, 2: 0.2}) == pytest.approx(0.3)
    assert rd.margin_of({1: 0.5}) == float("inf")


class _Scorer:
    def __init__(self) -> None:
        self.concepts = ["c1", "c2", "c3"]
        self.items_by_concept = {"c1": [{"id": "c1-a"}, {"id": "c1-b"}], "c2": [{"id": "c2-a"}], "c3": [{"id": "c3-a"}]}


def test_size_rule_assembly_from_scored_rows() -> None:
    scorer = _Scorer()
    frames = {"oracle": {"c1": [(0, 1), (1, 2)], "c2": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], "c3": None}}
    sym = {c: [{"id": i["id"], "correct": 1.0, "src": "sym"} for i in scorer.items_by_concept[c]] for c in scorer.concepts}
    routed = {c: [{"id": i["id"], "correct": 0.0, "src": "route"} for i in scorer.items_by_concept[c]] for c in scorer.concepts}
    rows = {"explicit-sym:oracle": sym, "explicit-route:fewshot:oracle": routed}
    out, info = rd.derived_size_rows("derived-size:6:fewshot:oracle", rows, scorer, frames)
    assert [r["src"] for r in out["c1"]] == ["sym", "sym"] and [r["src"] for r in out["c2"]] == ["route"] and out["c3"][0]["src"] == "sym"
    assert all(r["derived"] for rows_ in out.values() for r in rows_)
    assert info == {"size_limit": 6, "whole_frame_share": 0.75}
    out2, info2 = rd.derived_size_rows("derived-size:2:fewshot:oracle", rows, scorer, frames)
    assert info2["whole_frame_share"] == 0.75 and out2["c1"][0]["src"] == "sym"
    out1, _ = rd.derived_size_rows("derived-size:1:fewshot:oracle", rows, scorer, frames)
    assert out1["c1"][0]["src"] == "route" and out1["c3"][0]["src"] == "sym"
    with pytest.raises(KeyError):
        rd.derived_size_rows("derived-size:6:lex:oracle", rows, scorer, frames)
    assert rd.SIZE_STAR in rd.SIZE_LIMITS and f"explicit-size:{rd.SIZE_STAR}:fewshot:oracle" in rd.RD4E_CONDITIONS
    assert all(c.rsplit(":", 1)[1] in rd.RD4E_READERS for c in rd.RD4E_DERIVED)


# -- the few-shot scoring paths and a labelled smoke on the tiny world ---------------------------------------------------------

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")


def _routed_set(world: dict, heldout: dict, root: Path) -> Path:
    """The `heldout` toy set with a second relation per concept (lexname) and a part item, so routers have ≥ 2 candidates."""
    src = Path(heldout["items"])
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    concepts = e11._read_jsonl(src / "concepts.jsonl")
    rows = e11._read_jsonl(src / "items.jsonl")
    names = world["ontology"]["relation_names"]
    lex_atoms = sorted({a for c in concepts for r, a in ctx.ids(c["frame"]) if names[r] == "lexname"})
    for c in concepts:
        frame = ctx.ids(c["frame"])
        facts = {names[r]: a for r, a in frame}
        if "lexname" in facts:
            gold, pool = facts["lexname"], [a for a in lex_atoms if a != facts["lexname"]]
            cands = [" " + ctx.text(gold)] + [" " + ctx.text(a) for a in pool[:3]]
            rows.append({"id": f"{c['concept']}-property-lexname", "concept": c["concept"], "test": "property", "relation": "lexname",
                         "templates": ["The word {x} refers to a kind of", "In general, the {x} is a kind of"], "null": "this",
                         "candidates": cands, "gold": 0})
        if "part_meronym" in facts:
            rows.append({"id": f"{c['concept']}-property-part_meronym", "concept": c["concept"], "test": "property", "relation": "part_meronym",
                         "templates": ["The {x} has a part called the", "One part of the {x} is the"], "null": "this",
                         "candidates": [" " + ctx.text(facts["part_meronym"]), " wheel", " lid"], "gold": 0})
    out = root / "routed"
    out.mkdir()
    for name in ("concepts.jsonl", "definitions.jsonl", "manifest.json"):
        (out / name).write_text((src / name).read_text())
    e11._write_jsonl(out / "items.jsonl", rows)
    return out


def test_prefix_and_brute_fewshot_scores_agree(world, heldout, tmp_path) -> None:
    run = common.open_run(world["runs"]["C0p"], device="cpu", batch_size=4)
    read_set = e11.load_read_set(_routed_set(world, heldout, tmp_path))
    scorer = e11.ItemScorer(run, read_set, e11.run_context(run))
    queries = [(*rd.fewshot_prompt([("{x} has a part called the", "part meronym")], "{x} is a kind of", "aardvark"),
                [(0, "hypernym"), (1, "part meronym"), (3, "lexname")])]                   # short: the toy host has 64 positions
    stats: dict[str, int] = {}
    prefix, brute = rd.fewshot_scores(scorer, queries, scoring="prefix", stats=stats), rd.fewshot_scores(scorer, queries, scoring="brute")
    assert set(prefix[0]) == {0, 1, 3} and all(np.isfinite(v) for v in prefix[0].values()) and not stats
    for r in prefix[0]:
        assert prefix[0][r] == pytest.approx(brute[0][r], abs=1e-3)
    head, trimmed = rd.trim_head("Each question asks for one kind of fact.\nQuestion: a ...\nFact: X b\nQuestion: c ...\nFact: X d\n",
                                 run.tokenizer, 20)
    assert trimmed and head.startswith("Each question") and head.count("Question:") == 1
    long = [(*rd.fewshot_prompt([("{x} has a part called the", "part meronym")] * 12, "{x} is a kind of", "aardvark"), [(0, "hypernym"), (1, "part meronym")])]
    rd.fewshot_scores(scorer, long, scoring="prefix", stats=stats)
    assert stats["trimmed"] == 1


def test_rd4e_family_smoke_on_the_toy_world(world, heldout, tmp_path) -> None:
    items_dir = _routed_set(world, heldout, tmp_path)
    out = tmp_path / "rd4e-c5"
    frames = heldout["e11"] / "frames.jsonl"
    rd.main(["recall", "--run", str(world["runs"]["C5"]), "--items", str(items_dir), "--output", str(out), "--frames", str(frames),
             "--style", "prose", "--family", "rd4e", "--null", "plain", "--max-length", "200", "--device", "cpu", "--batch-size", "8",
             "--resamples", "50", "--smoke", "--calibration-items", str(items_dir), "--calibration-frames", str(frames),
             "--calibration-style", "prose"])
    summary = json.loads((out / "summary.json").read_text())
    conds = summary["conditions"]
    assert set(rd.RD4E_CONDITIONS) <= set(conds) and set(rd.RD4E_DERIVED) <= set(conds) and summary["max_length"] == 200
    assert summary["routing"]["scoring"] == "prefix" and summary["routing"]["calibration_queries"] > 0
    diag = summary["diagnostics"]
    assert diag["explicit-route:fewshot-abstain:oracle"]["threshold"] == summary["routing"]["threshold"]
    assert diag["explicit-route:fewshot-top2:oracle"]["lines_per_item"] >= diag["explicit-route:fewshot:oracle"]["lines_per_item"]
    assert diag["explicit-route:fewshot-para:oracle"]["route_accuracy"] is not None
    assert diag[f"explicit-size:{rd.SIZE_STAR}:fewshot:oracle"]["whole_frame_share"] == diag[f"derived-size:{rd.SIZE_STAR}:fewshot:oracle"]["whole_frame_share"]
    assert all("truncated_share" in (d or {}) for d in diag.values() if d is not None and "size_limit" not in (d or {}))
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["id"]] = r
    real, derived = by[f"explicit-size:{rd.SIZE_STAR}:fewshot:oracle"], by[f"derived-size:{rd.SIZE_STAR}:fewshot:oracle"]
    assert set(real) == set(derived) and all(derived[i]["derived"] for i in derived)
    assert all(real[i]["correct"] == pytest.approx(derived[i]["correct"], abs=1e-6) for i in real)   # the assembly check
    routing = [json.loads(line) for line in (out / "routing.jsonl").read_text().splitlines()]
    calibration = [r for r in routing if r.get("calibration")]
    routed = [r for r in routing if not r.get("calibration")]
    assert calibration and routed and all(set(r["fewshot"]) == set(r["options"]) == set(r["fewshot_para"]) for r in routed)
    assert {r["reader"] for r in routed} == {"oracle", "typeprior"}
    assert (out / "report.md").read_text().startswith("**SMOKE TEST")
    report = tmp_path / "report"
    rd.main(["recall-report", "--family", "rd4e", "--runs", str(out), "--output", str(report), "--resamples", "50"])
    data = json.loads((report / "report.json").read_text())
    g = data["groups"][0]
    assert g["size_rule_check"]["agreement"] == 1.0 and "decision" in g and g["abstention"]
    assert {(r["a"], r["b"]) for r in g["primary"]} == set(rd.PRIMARY_E)
    text = (report / "report.md").read_text()
    assert "Decision rule" in text and "Size-rule check" in text and "addendum 12" in text


def test_size_rule_offline_from_run_folders(world, heldout, tmp_path) -> None:
    items_dir = _routed_set(world, heldout, tmp_path)
    out = tmp_path / "rd4e-c0p"
    frames = heldout["e11"] / "frames.jsonl"
    conditions = "explicit-sym:oracle,explicit-rel:oracle,explicit-route:lex:oracle"
    rd.main(["recall", "--run", str(world["runs"]["C0p"]), "--items", str(items_dir), "--output", str(out), "--frames", str(frames),
             "--style", "prose", "--conditions", conditions, "--null", "plain", "--device", "cpu", "--batch-size", "8",
             "--resamples", "50", "--smoke"])
    report = tmp_path / "offline"
    rd.main(["size-rule-offline", "--sym-runs", str(out), "--route-runs", str(out), "--output", str(report), "--routers", "lex",
             "--limits", "1,3,1000", "--resamples", "50"])
    data = json.loads((report / "offline-size-rule.json").read_text())
    rows = {(r["L"], r["test"]): r for r in data["rows"]}
    assert rows[(1000, "property")]["whole_frame_share"] == 1.0 and rows[(1000, "property")]["vs_route"]["n"] > 0
    assert rows[(1, "property")]["vs_route"]["mean"] == 0.0                              # pure routing below the smallest frame
    assert "hybrid mean" in (report / "offline-size-rule.md").read_text()
