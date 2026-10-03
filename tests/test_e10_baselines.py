"""E10 baselines for claim D (WP-PQ2): rule mining, KGE, operator axioms, null worlds, fresh holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from vsa_embed import kg_baselines as kg
from vsa_embed.experiments import e10_baselines as b
from vsa_embed.experiments import e10_report as report
from vsa_embed.experiments import e10_self_semantics as e10

ROOT = Path(__file__).resolve().parents[1]


def _toy_relations() -> dict[str, set]:
    sym = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}
    part = {(10, 20), (11, 20), (12, 21), (13, 21)}
    has = {(20, 10), (20, 11), (21, 12), (21, 13)}
    loc = {(30, 31), (31, 32), (30, 32), (33, 31), (33, 32)}
    return {"similar": sym, "part_of": part, "has_part": has, "located_in": loc}


def test_rule_mining_finds_the_planted_horn_rules() -> None:
    rules = {s.rule.name: s for s in kg.mine_rules(_toy_relations())}
    assert rules["similar⁻¹ ⇒ similar"].pca_confidence == 1.0
    assert rules["part_of⁻¹ ⇒ has_part"].std_confidence == 1.0
    assert rules["has_part⁻¹ ⇒ part_of"].support == 4
    trans = rules["located_in ∧ located_in ⇒ located_in"]
    assert trans.rule.kind == "transitive" and trans.std_confidence == 1.0
    assert kg.Rule("x", (("y", True),)).hypothesis_name() == "inverse_of:y"
    gold = kg.gold_axioms(_toy_relations(), min_support=2)
    assert kg.Rule("similar", (("similar", True),)) in gold
    assert kg.Rule("located_in", (("located_in", False), ("located_in", False))) in gold


def test_pca_confidence_uses_partial_completeness() -> None:
    # body predicts (x, y) for x without any head tail: excluded from the PCA denominator
    relations = {"p": {(1, 2), (3, 4)}, "q": {(1, 2)}}
    score = kg.score_rule(kg.Rule("q", (("p", False),)), relations)
    assert score.std_confidence == 0.5 and score.pca_confidence == 1.0


def test_operator_axiom_scores() -> None:
    phases = {"sym": torch.full((8,), torch.pi), "a": torch.linspace(0, 1, 8), "b": -torch.linspace(0, 1, 8)}
    assert kg.rotate_axiom_score(kg.Rule("sym", (("sym", True),)), phases) == pytest.approx(1.0)
    assert kg.rotate_axiom_score(kg.Rule("b", (("a", True),)), phases) == pytest.approx(1.0)
    g = torch.Generator().manual_seed(0)
    r = torch.nn.functional.normalize(torch.randn(64, generator=g), dim=0)
    # involution is the HRR approximate inverse: r* ⊛ r ≈ δ for unitary r; exact identity of the index map
    star = kg.involution(r)
    assert torch.allclose(star[0], r[0]) and torch.allclose(star[1], r[-1])
    roles = {"r": r, "s": star}
    assert kg.hrr_axiom_score(kg.Rule("s", (("r", True),)), roles) == pytest.approx(1.0)


def test_kge_models_learn_a_tiny_graph() -> None:
    triples = torch.tensor([[i, 0, i + 1] for i in range(10)] + [[i + 1, 1, i] for i in range(10)])
    false = torch.tensor([[i, 0, (i + 5) % 11] for i in range(10)])
    for kind in ("transe", "rotate", "complex"):
        model = kg.train_kge(triples, 11, 2, kind=kind, dimension=16, epochs=300, negatives=8, seed=0)
        true_s, false_s = kg.kge_scores(model, triples[:10]), kg.kge_scores(model, false)
        labels = [True] * 10 + [False] * 10
        assert kg.auc(true_s.tolist() + false_s.tolist(), labels) > 0.8, kind


def test_best_threshold_and_auc() -> None:
    assert kg.auc([0.1, 0.2, 0.9, 0.8], [False, False, True, True]) == 1.0
    assert kg.auc([0.5, 0.5], [True, False]) == 0.5
    assert kg.best_threshold([0.9, 0.8, 0.3, 0.1], [True, True, False, False]) == 0.8


@pytest.fixture(scope="module")
def tiny_config() -> dict:
    config = b.resolve(yaml.safe_load((ROOT / "experiments/e10-self-semantics/e10-baselines.yaml").read_text()), root=ROOT)
    config["world"].update(branching=[3, 4], roots=2, wholes=6, parts=10, groups=3, members=8, attributes=10, regions=2,
                           countries_per_region=2, cities_per_country=2, dimension=32)
    config["seeds"], config["dev_seeds"] = [3], [4]
    config["recovery"].update(rates=[0.3], kge={"dimension": 16, "epochs": 30, "negatives": 4, "lr": 0.01, "gamma": 6.0,
                                                "adversarial": 1.0}, reference=None)
    config["edges"].update(steps=60)
    config["discovery"].update(steps_per_round=60, max_rounds=2, resamples=100, riddle_full=False)
    config["selftest"]["resamples"] = 100
    config["discovery_baselines"] = {"rotate_epochs": 20, "worlds": {"absent": ["reused", "fresh"], "null_distractors": ["reused"],
                                                                     "null_permuted": ["reused"]}}
    return config


def test_null_scenarios_offer_no_gold_pair(tiny_config) -> None:
    world = e10.build_world(tiny_config, 3)
    gold_pairs = set(world.relations_of_pair())
    for build in (b.null_distractor_scenario, b.null_permuted_scenario):
        scenario = build(world, hidden=tiny_config["c"]["hidden"], seed=3, coverage=0.9, distractor_ratio=1.0, max_slots=4)
        t = scenario.table
        offered = [(h, a) for h, a, o in zip(t.heads.tolist(), t.fillers.tolist(), t.open.tolist()) if o]
        assert offered and not (set(offered) & gold_pairs)
    null_edges = b.null_edge_scenario(world, seed=3, fraction=0.5, distractors=2)
    assert not null_edges.info["erased"]
    t = null_edges.table
    candidates = {(h, r, a) for h, r, a, c in zip(t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(), t.candidate.tolist()) if c}
    assert candidates and not (candidates & set(world.edge_list()))


def test_fresh_observations_follow_the_world_noise(tiny_config) -> None:
    world = e10.build_world(tiny_config, 3)
    concepts = torch.arange(5)
    fresh = b.fresh_observations(world, concepts, 64, noise=float(tiny_config["world"]["observation_noise"]), seed=1)
    real = world.observations[concepts]
    clean = world.truth["clean_targets"][concepts]
    cos = lambda x: torch.nn.functional.cosine_similarity(x, clean[:, None], dim=-1).mean()  # noqa: E731
    assert fresh.shape == (5, 64, world.target_dimension)
    assert abs(float(cos(fresh)) - float(cos(real))) < 0.05
    assert not torch.equal(fresh[:, :real.shape[1]], real)


def test_baseline_runner_end_to_end(tiny_config, tmp_path) -> None:
    summary = b.run(tiny_config, tmp_path / "run", workers=1, log=lambda *_: None)
    run = tmp_path / "run"
    for name in ("metrics.jsonl", "summary.json", "report.md", "resolved_config.yaml", "manifest.json"):
        assert (run / name).exists(), name
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    parts = {r["part"] for r in rows}
    assert parts == {"recovery", "axioms", "edges", "discovery"}
    assert set(summary["thresholds"]) == {"rotate", "amie", "hrr"}
    rec = summary["recovery"]["0.3"]["methods"]
    assert set(rec) >= {"amie", "transe", "rotate", "complex", "itere"}
    assert 0.0 <= rec["amie"]["auc"]["mean"] <= 1.0
    assert {"absent/reused", "absent/fresh", "null_distractors/reused", "null_permuted/reused"} <= set(summary["discovery"])
    null = summary["discovery"]["null_distractors/reused"]
    assert null["false_accepted_slots"]["mean"] == null["accepted_slots"]["mean"]
    text = (run / "report.md").read_text()
    assert "D-B2" in text and "D-B5" in text and "null_permuted/reused" in text
    # the CLI re-renders the same report from the rows
    out = tmp_path / "again.md"
    report.main(["--baselines", str(run), "--output", str(out)])
    assert out.read_text() == text
