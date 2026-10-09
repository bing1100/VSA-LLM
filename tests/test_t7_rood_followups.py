"""T7-ROOD follow-ups (decision 63, TK-H1): the `t7rood` track in the understanding, E11 and E12 modules, E12's opt-in
relation families (on the toy E9 world of `test_e9_binding`), the paired comparison T7-ROOD vs T7 v1 and the queue plan."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from test_e9_binding import ATOMS, RELATIONS, TERMS, _frame, _lexicon, _ok, world  # noqa: E402,F401
from vsa_embed.experiments import e9_understanding as und  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402
from vsa_embed.experiments import e12_self_query as sqx  # noqa: E402
from vsa_embed.experiments import t7_rood_compare as cmp  # noqa: E402
from vsa_embed.experiments import t7_rood_queue as plan  # noqa: E402
from vsa_embed.span_channel import normalize_alias  # noqa: E402

torch.set_num_threads(1)


def test_t7rood_is_a_t7_track_for_understanding_and_e11() -> None:
    assert und.TRACK_SPECS["t7rood"] is und.TRACK_SPECS["t7"]
    assert e11.PRIMARY_STYLE["t7rood"] == "scr" and e11.EXCLUDED_KINDS["t7rood"] == e11.EXCLUDED_KINDS["t7"]
    for name in ("understanding-t7rood-smollm2-v1", "understanding-t7rood-qwen3-v1"):
        manifest = json.loads((Path("experiments/e9-retrofit/items") / name / "manifest.json").read_text())
        assert manifest["track"] == "t7rood" and manifest["anchors"]["heldout"] == 908
    read = json.loads(Path("experiments/e11-read-to-learn/items/t7rood-heldout-smollm2-v1/manifest.json").read_text())
    assert read["track"] == "t7rood" and read["styles"] == ["scr"] and read["held_out_real"] == 908


# -- E12: the opt-in relation families -------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def toy_lexicon(monkeypatch):
    monkeypatch.setattr(sqx, "lexicon_for_track", lambda track, family, ontology: _lexicon())


@pytest.fixture(scope="module")
def relation_dir(world, tmp_path_factory) -> Path:
    """A hand-built WP-UB set: a paraphrase item (area), a negation pair (owned_by) and a reverse item."""
    root = tmp_path_factory.mktemp("rel") / "understanding-toy-rel-v1"
    root.mkdir()
    entry = {t: world["table"].alias_to_entry[normalize_alias(t)] for t in TERMS}
    names = lambda term: [[RELATIONS[r], ATOMS[f]] for r, f in _frame(term)]  # noqa: E731
    concepts = [{"concept": "u-bl", "subset": "heldout", "surface": "Brightwater Ledger", "entry": entry["Brightwater Ledger"],
                 "frame": names("Brightwater Ledger"), "random_frame": None, "source": "Brightwater Ledger", "frequency": 0},
                {"concept": "u-gc", "subset": "seen", "surface": "Grosh Console", "entry": entry["Grosh Console"],
                 "frame": names("Grosh Console"), "random_frame": None, "source": "Grosh Console", "frequency": 10}]
    base = {"subset": "heldout", "anchor": "u-bl", "slots": {"x": "u-bl"}, "text": {}, "null": {"x": "this"}, "pair": None, "meta": {}}
    items = [{**base, "id": "u-bl-paraphrase-area", "family": "paraphrase", "test": "paraphrase", "relation": "area",
              "templates": ["{x} sits in the"], "candidates": [" money side", " selling side"], "gold": 0, "chance": 0.5},
             {**base, "id": "u-bl-negation-owned_by-affirm", "family": "negation", "test": "affirm", "relation": "owned_by",
              "templates": ["{x} is owned by"], "candidates": [" the Zash Team", " nobody"], "gold": 0, "chance": 0.5, "pair": "n1"},
             {**base, "id": "u-bl-negation-owned_by-negate", "family": "negation", "test": "negate", "relation": "owned_by",
              "templates": ["{x} is not owned by"], "candidates": [" nobody", " the Zash Team"], "gold": 0, "chance": 0.5, "pair": "n1"},
             {**base, "id": "u-bl-reverse-owned_by", "family": "reverse", "test": "reverse", "relation": "owned_by",
              "slots": {"x": "u-bl", "y": "u-gc"}, "text": {"c": "owned by the Zash Team"}, "null": {"c": "we mean"},
              "templates": ["Of {x} and {y}, the one {c} is"], "candidates": [" {x}", " {y}"], "gold": 0, "chance": 0.5,
              "meta": {"filler": "term:Zash Team", "partner": "u-gc"}}]
    (root / "concepts.jsonl").write_text("".join(json.dumps(c) + "\n" for c in concepts))
    und.write_jsonl_gz(root / "items.jsonl.gz", items)
    (root / "manifest.json").write_text(json.dumps({"schema": und.SCHEMA, "track": "toy", "family": "gpt2"}))
    return root


def test_relation_families_are_opt_in(relation_dir) -> None:
    assert [p.id for p in sqx.load_item_set(relation_dir).prompts] == ["u-bl-reverse-owned_by"]      # the default is unchanged
    families = ("reverse", *sqx.RELATION_FAMILIES)
    assert len(sqx.load_item_set(relation_dir, families=families).prompts) == 4
    args = SimpleNamespace(families="reverse,paraphrase")
    assert sqx._families(args) == ("reverse", "paraphrase") and sqx._families(SimpleNamespace()) == sqx.UNDERSTANDING_FAMILIES
    command = sqx.evaluate_command(Path("RUN"), relation_dir, families=["reverse", "negation"])
    assert command[command.index("--families") + 1] == "reverse,negation"


def test_relation_family_contexts_recall_the_anchors_whole_frame(world, relation_dir) -> None:
    stores = {"own": sqx.load_store(world["runs"]["C5"], "own")}
    item_set = sqx.load_item_set(relation_dir, families=("reverse", *sqx.RELATION_FAMILIES))
    builder = sqx.ContextBuilder(item_set, world["ontology"], _lexicon(), stores)
    symbolic, records = builder.build(sqx.Condition("symbolic"))
    whole = symbolic["u-bl-paraphrase-area"]
    assert whole == symbolic["u-bl-negation-owned_by-affirm"] == symbolic["u-bl-negation-owned_by-negate"]   # one recall per anchor
    assert "Zash Team" in whole and "finance" in whole and not whole.startswith("lookup(")
    by_item = {r["item"]: r for r in records}
    assert by_item["u-bl-paraphrase-area"]["slot_correct"] == 1.0 and by_item["u-bl-negation-owned_by-affirm"]["slot_correct"] == 1.0
    assert symbolic["u-bl-reverse-owned_by"].startswith("lookup(owned by")
    own, own_records = builder.build(sqx.Condition("recall", "own"))
    assert set(own) == set(symbolic) and all(r["slot_correct"] in (0.0, 1.0) for r in own_records if "slot_correct" in r)
    definitions, _ = builder.build(sqx.Condition("definition"))
    assert not definitions                                                    # no definition writer: relation items not scored


# -- the paired comparison ---------------------------------------------------------------------------------------------------

def test_difference_of_relative_differences_and_readings() -> None:
    rng = np.random.default_rng(0)
    reference = rng.uniform(2.0, 3.0, 200) * 10
    same = cmp.relative_difference_of_differences((reference * 0.97, reference), (reference * 0.97, reference), resamples=300, seed=1)
    assert same["difference"]["mean"] == pytest.approx(0.0) and same["difference"]["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert same["rood"]["mean"] == pytest.approx(-0.03)
    better = cmp.relative_difference_of_differences((reference * 0.90, reference), (reference * 0.98, reference), resamples=300, seed=1)
    assert better["difference"]["mean"] == pytest.approx(-0.08) and better["difference"]["ci_high"] < 0
    assert cmp.reading(better["difference"], -1) == "confirmed" and cmp.reading(better["difference"], +1) == "contradicted"
    assert cmp.reading({"mean": -0.01, "ci_low": -0.02, "ci_high": 0.01}, -1).startswith("holds")
    with pytest.raises(ValueError):
        cmp.relative_difference_of_differences((reference, reference), (reference[:10], reference[:10]), resamples=10, seed=0)


def _group(losses: dict[str, float], starts: np.ndarray, counts: np.ndarray, seeds=(1, 2)) -> SimpleNamespace:
    """A fake `e9_report.Group`: per model and seed a run whose window sums are loss × counts."""
    runs = {m: {s: SimpleNamespace(path=Path(f"{m}-s{s}"), starts=starts, windows={cmp.STRATUM: (counts * loss, counts)})
                for s in ((1,) if m == "P0" else seeds)} for m, loss in losses.items()}
    return SimpleNamespace(seeds=lambda m: sorted(runs.get(m, {})),
                           paired=lambda m, seeds: {s: runs[m][min(runs[m])] for s in seeds} if m == "P0" else
                           {s: runs[m][s] for s in seeds if s in runs[m]})


def test_compare_pairs_windows_and_reads_p2_p3(monkeypatch) -> None:
    starts, counts = np.arange(0, 4096 * 50, 4096), np.random.default_rng(1).integers(1, 9, 50).astype(float)
    groups = {"rood": _group({"P0": 4.0, "C0'": 3.9, "C5": 3.5}, starts, counts),
              "v1": _group({"P0": 4.0, "C0'": 3.0, "C5": 2.9}, starts, counts)}
    monkeypatch.setattr(cmp, "stage_group", lambda root, host: groups[str(root)])
    out = cmp.compare(Path("rood"), Path("v1"), resamples=200)
    p2, p3 = out["strata"][cmp.STRATUM]["P2"], out["strata"][cmp.STRATUM]["P3"]
    assert out["seeds"] == [1, 2] and p2["rood"]["mean"] == pytest.approx(3.5 / 3.9 - 1) and p2["reference"]["mean"] == pytest.approx(2.9 / 3.0 - 1)
    assert p2["reading"] == "confirmed" and p3["reading"] == "confirmed"           # a constant shift: CIs collapse on the point
    text = cmp.render(out, "T")
    assert "| P2 |" in text and "confirmed" in text
    groups["v1"] = _group({"P0": 4.0, "C0'": 3.0, "C5": 2.9}, starts + 1, counts)
    with pytest.raises(ValueError, match="do not pair"):
        cmp.compare(Path("rood"), Path("v1"), resamples=50)


# -- the queue plan ----------------------------------------------------------------------------------------------------------

def test_follow_up_plan_priorities_names_and_quoting() -> None:
    compare = plan.compare_jobs()
    assert compare[0]["name"] == "t7rood-vs-t7-report" and compare[0]["priority"] == 54.4974 and "-report" in compare[1]["name"]
    jobs = plan.e11_jobs()
    assert jobs[0]["name"].startswith("e11-gradient-dev") and jobs[-1]["name"] == "e11-t7rood-report"
    assert {j["priority"] for j in jobs} == {54.4979, 54.49791, 54.49792} and len(jobs) == 1 + 9 + 1
    assert jobs[-1]["priority"] == max(j["priority"] for j in jobs)                 # the report after every evaluation
    c5 = next(j["command"] for j in jobs if j["name"].endswith("C5-s2"))
    assert c5[c5.index("--items") + 1].endswith("t7rood-heldout-smollm2-v1") and c5[c5.index("--alias-table") + 1].endswith("t7rood.json")
    assert c5[c5.index("--gradient-lr-from") + 1].startswith(str(plan.E11_DEV))
    # Decision 64: the in-context route and the shape-matched control on C5; the C6d encoder route, reported with them.
    assert "context" in c5[c5.index("--methods") + 1].split(",") and "linker-random" in c5[c5.index("--readers") + 1].split(",")
    c6d = next(j for j in jobs if j["name"].endswith("C6d-s3"))
    assert c6d["priority"] == 54.49791 and c6d["command"][c6d["command"].index("--methods") + 1] == "encoder,windows"
    assert "--gradient-lr-from" not in c6d["command"] and c6d["command"][-1] in jobs[-1]["command"]
    e12 = plan.e12_jobs()
    evals = [j for j in e12 if "self-query" in j["name"]]
    assert len(evals) == 10 and {j["priority"] for j in e12} == {54.49795} and e12[-1]["name"].endswith("-report")
    c5 = next(j["command"] for j in evals if "-C5-s1-" in j["name"])
    assert c5[c5.index("--conditions") + 1] == "none,recall:own,symbolic"
    assert c5[c5.index("--families") + 1] == "reverse,paraphrase,negation,affordance"
    text = plan.render(compare + jobs + e12)
    assert "'E12 — recall on T7-ROOD (WP-UB reverse and relation families)'" in text and " $PY -m " in text
