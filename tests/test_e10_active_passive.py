"""E10.9a: passive learning paired with active self-reflection (graded worlds, budgets, arms, isolation)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from vsa_embed.experiments import e10_active_passive as A
from vsa_embed.experiments import e10_baselines as b
from vsa_embed.experiments import e10_self_semantics as e10
from vsa_embed.learnable_ontology import ORIGIN, train_composer
from vsa_embed.ontology_discovery import enforce_rule, interpret_slot, own_relations
from vsa_embed.ontology_hypotheses import StructuralHypothesis

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    cfg = A.resolve(yaml.safe_load((ROOT / "experiments/e10-self-semantics/e10.9-active-passive.yaml").read_text()), root=ROOT)
    cfg["world"].update(roots=2, branching=[2, 4], attributes=10, wholes=4, parts=8, groups=2, members=5,
                        similar_pairs_per_family=1, regions=1, countries_per_region=2, cities_per_country=2,
                        dimension=32, split_fractions=[0.15, 0.15, 0.15])
    cfg["c"].update(max_slots=8)
    cfg["discovery"].update(resamples=200)
    cfg["dream_settings"].update(refit_steps=10)
    cfg["schedule"].update(total_steps=120, reflect_every=40, eval_every=40)
    cfg["seeds"], cfg["budgets"], cfg["rhos"] = [3], [0.5, 1.0], [1.0, 0.0]
    cfg["arms"] = {"P": {}, "H": {}, "H+R": {}, "P+compute": {"rho": [1.0]}, "H-rand": {"rho": [1.0]},
                   "H-AMIE": {"rho": [1.0]}, "P-split": {}, "A-first": {"f": [1.0]}, "A-first-forced": {"f": [1.0]}}
    cfg["extra_cells"] = []
    cfg["k_sensitivity"] = {"K": [60], "arms": ["P", "H"], "rho": [1.0], "f": [1.0]}
    cfg["statistics"]["bootstrap_resamples"] = 200
    cfg["workers"] = 1
    return cfg


def _setup(config, rho=1.0, f=1.0, seed=3):
    c = config["c"]
    world = e10.build_world(config, seed)
    scenario = A.graded_scenario(world, rho=rho, hidden=c["hidden"], seed=seed, coverage=c["coverage"],
                                 distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    data = A.budget_data(world, scenario, A.observation_pool(world, seed), f, seed)
    return world, scenario, data


def _table(scenario):
    t = scenario.table
    return [t.heads.tolist(), t.relations.tolist(), t.fillers.tolist(), t.open.tolist(), t.candidate.tolist()]


def test_graded_world_runs_from_the_absent_scenario_to_the_null_world() -> None:
    full = A.resolve(yaml.safe_load((ROOT / "experiments/e10-self-semantics/e10.9-active-passive.yaml").read_text()), root=ROOT)
    c = full["c"]
    world = e10.build_world(full, 5)                     # the E10.0 world (≈ 160 offered gold pairs)
    common = dict(hidden=c["hidden"], seed=5, coverage=c["coverage"], distractor_ratio=c["distractor_ratio"], max_slots=8)
    assert _table(A.graded_scenario(world, rho=1.0, **common)) == _table(e10.hidden_scenario(world, **common))
    assert _table(A.graded_scenario(world, rho=0.0, **common)) == _table(b.null_permuted_scenario(world, **common))
    for rho in (0.5, 0.25):
        info = A.graded_scenario(world, rho=rho, **common).info
        assert abs(info["structure_fraction"] - rho) < 0.12, info
    assert A.graded_scenario(world, rho=0.0, **common).info["offered_gold_after"] == 0


def test_budgets_are_nested_prefixes_without_audit_observations(config) -> None:
    world = e10.build_world(config, 3)
    pool = A.observation_pool(world, 3)
    audit = set(world.observation_splits["audit"])
    assert not {o for _, o in pool} & audit
    assert len(pool) == len(set(pool))
    small, full = (A.budget_data(world, e10.hidden_scenario(world, hidden=config["c"]["hidden"], seed=3, coverage=0.7,
                                                            distractor_ratio=1.0, max_slots=8), pool, f, 3) for f in (0.3, 1.0))
    assert small.counts["prefix_observations"] < full.counts["prefix_observations"] == len(pool)
    # full budget = E10.0's data: training observations of training concepts; held-out = val observations
    ctx = e10.make_context(world, e10.hidden_scenario(world, hidden=config["c"]["hidden"], seed=3, coverage=0.7,
                                                      distractor_ratio=1.0, max_slots=8), 3)
    assert torch.equal(full.ctx.train_concepts, ctx.train_concepts)
    assert torch.equal(full.ctx.heldout.concepts, ctx.heldout.concepts)
    # lcm repetition keeps each concept's own mean of its observations
    torch.testing.assert_close(full.ctx.heldout.observations.mean(1), ctx.heldout.observations.mean(1))
    assert full.counts["passive_concepts"] == full.counts["heldout_concepts"]


def test_masked_training_equals_train_composer_with_a_full_mask(config) -> None:
    world, scenario, data = _setup(config)
    a = e10.make_composer(world, scenario, config, 3)
    b_ = copy.deepcopy(a)
    concepts, targets, mask = data.hybrid
    assert bool((mask == 1).all())
    train_composer(a, concepts, targets, steps=15, lr=0.02)
    A.train_masked(b_, concepts, targets, mask, steps=15, lr=0.02)
    torch.testing.assert_close(a.edge_masses(), b_.edge_masses(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(a.atomics, b_.atomics, atol=1e-5, rtol=1e-4)


def _learning_trace(config, world, arm, evaluate):
    c = config["c"]
    scenario = A.graded_scenario(world, rho=1.0, hidden=c["hidden"], seed=3, coverage=c["coverage"],
                                 distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    data = A.budget_data(world, scenario, A.observation_pool(world, 3), 1.0, 3)
    composer, log = A.run_arm(arm, world, scenario, data, config, 3, reflect_every=40, evaluate=evaluate)
    decisions = [[(s["slot"], s["accept"], s["commit"], s["adopted"], s["size"], s["rule_added"], s["rule_pruned"])
                  for s in r["slots"]] for r in log["reflections"]]
    revisions = [[(v.get("proposal"), v.get("accepted")) for v in r["revisions"]] for r in log["reflections"]]
    return decisions, revisions, composer.edge_masses().detach().clone(), composer.slot_role_vectors().detach().clone()


def test_learning_never_reads_the_audit_split_or_the_evaluation(config) -> None:
    world = e10.build_world(config, 3)
    poisoned = copy.deepcopy(world)
    poisoned.observations[:, world.observation_splits["audit"]] = float("nan")
    for arm in ("H+R", "H-rand"):
        clean = _learning_trace(config, world, arm, evaluate=False)
        assert any(any(s[2] for s in r) for r in clean[0]), "the tiny world should commit something"
        for other in (_learning_trace(config, poisoned, arm, evaluate=False), _learning_trace(config, world, arm, evaluate=True)):
            assert clean[0] == other[0] and clean[1] == other[1]
            torch.testing.assert_close(clean[2], other[2])
            torch.testing.assert_close(clean[3], other[3])


def test_passive_arm_never_commits_and_reflection_is_counted(config) -> None:
    world, scenario, data = _setup(config)
    passive, log_p = A.run_arm("P", world, scenario, data, config, 3, reflect_every=40, evaluate=False)
    assert not bool(passive.slot_frozen.any()) and not bool((passive.origin == ORIGIN["rule"]).any())
    assert not log_p["reflections"] and int(passive.slot_active.sum()) == 1 + len(log_p["rounds"])
    hybrid, log_h = A.run_arm("H", world, scenario, data, config, 3, reflect_every=40, evaluate=False)
    assert log_h["reflection_units"] and any(u > 0 for u in log_h["reflection_units"])
    assert int(hybrid.slot_active.sum()) == int(passive.slot_active.sum())        # same slot schedule
    with A.counting() as box:
        e10.eval_fit(hybrid, e10.audit_data(world), world.splits["test"])
    assert box["nograd_rows"] > 0 and box["grad_rows"] == 0
    extra, log_c = A.run_arm("P+compute", world, scenario, data, config, 3, reflect_every=40, evaluate=False,
                             compute_plan=[10_000] * len(log_h["rounds"]))
    assert log_c["extra_steps"] > 0 and not bool(extra.slot_frozen.any())


def test_random_matched_content_writes_as_much_as_the_rule(config) -> None:
    world, scenario, data = _setup(config)
    composer, _ = e10.run_method(world, scenario, config, 3, "oracle", mode="absent", matched_steps=60)
    settings = e10.discovery_settings(config)
    similar = world.relation_id("similar_to")
    gold = world.relations_of_pair()
    sizes = {}
    for slot in composer.slot_active.nonzero().flatten().tolist():
        members = composer.slot_members(slot, min_mass=0.0)
        heads, fillers = composer.edge_heads()[members].tolist(), composer.schedule.fillers[members].tolist()
        sizes[slot] = sum(similar in gold.get((h, a), set()) for h, a in zip(heads, fillers))
    slot = max(sizes, key=sizes.get)
    record = interpret_slot(composer, slot, data.ctx, A.dataclasses.replace(settings, min_mass=0.0))
    relations = own_relations(composer, data.ctx, exclude_slot=slot)
    composer.crystallize(slot, min_mass=0.0)
    hyp = StructuralHypothesis("symmetric")
    rule_copy = copy.deepcopy(composer)
    expected = enforce_rule(rule_copy, slot, hyp, data.ctx, settings, mass=0.5, pairs=record["pairs"], relations=relations)
    assert expected["rule_added"] > 0
    before = int((composer.origin == ORIGIN["rule"]).sum())
    got = A.enforce_random_matched(composer, slot, hyp, data.ctx, settings, mass=0.5, pairs=record["pairs"],
                                   relations=relations, seed=1)
    assert got == expected
    assert int((composer.origin == ORIGIN["rule"]).sum()) - before == expected["rule_added"]


def test_untestable_predictions_are_skipped_under_budgets_and_nothing_changes_at_full_data() -> None:
    """Regression (first E10.9 launch): at f < 1 a captured head may lack held-out observations, and the
    `sub_relation_of` negatives are headed by captured heads; `testable_only` skips them (opt-in)."""
    full = A.resolve(yaml.safe_load((ROOT / "experiments/e10-self-semantics/e10.9-active-passive.yaml").read_text()), root=ROOT)
    assert full["discovery"]["testable_only"] is True
    c = full["c"]
    world = e10.build_world(full, 303)
    scenario = A.graded_scenario(world, rho=1.0, hidden=c["hidden"], seed=303, coverage=c["coverage"],
                                 distractor_ratio=c["distractor_ratio"], max_slots=c["max_slots"])
    pool = A.observation_pool(world, 303)
    off = copy.deepcopy(full)
    off["discovery"]["testable_only"] = False
    budget = A.budget_data(world, scenario, pool, 0.4, 303)
    with pytest.raises(ValueError, match="no held-out observations"):
        A.run_arm("H", world, scenario, budget, off, 303, reflect_every=800, evaluate=False)
    A.run_arm("H", world, scenario, budget, full, 303, reflect_every=800, evaluate=False)
    whole = A.budget_data(world, scenario, pool, 1.0, 303)
    logs = [A.run_arm("H", world, scenario, whole, cfg, 303, reflect_every=800, evaluate=False)[1] for cfg in (full, off)]
    strip = lambda log: [[{k: v for k, v in s.items() if k != "pairs"} for s in r["slots"]] for r in log["reflections"]]  # noqa: E731
    assert json.dumps(strip(logs[0])) == json.dumps(strip(logs[1]))          # NaN-aware comparison


def test_amie_explanation_reads_symmetry_and_inverses() -> None:
    pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5)}
    assert A.amie_explanation(pairs, {"part_of": {(10, 20)}}, {"min_support": 2, "min_head_coverage": 0.01,
                                                                  "min_pca": 0.1}).name == "symmetric"
    inv = {(20, 10), (21, 11), (22, 12)}
    got = A.amie_explanation(inv, {"part_of": {(10, 20), (11, 21), (12, 22)}},
                             {"min_support": 2, "min_head_coverage": 0.01, "min_pca": 0.1})
    assert got.name == "inverse_of:part_of"


def test_learning_curve_statistics() -> None:
    assert A.to_criterion([0.1, 0.2, 0.4], [0.5, 0.6, 0.8], 0.7) == pytest.approx(0.2 * 2 ** 0.5)   # midway on ln f
    assert A.to_criterion([0.1, 0.2], [0.5, 0.6], 0.9) == float("inf")
    assert A.to_criterion([0.1, 0.2], [0.95, 0.96], 0.9) == 0.1
    assert A.aulc_linear([0.0, 1.0], [0.2, 0.4]) == pytest.approx(0.3)
    assert A.aulc([0.25, 1.0], [0.5, 0.5]) == pytest.approx(0.5)


def test_end_to_end_run_writes_the_run_folder(config, tmp_path) -> None:
    out = tmp_path / "run"
    summary = A.run(config, out, workers=1, log=lambda *_: None)
    for name in ("metrics.jsonl", "summary.json", "report.md", "resolved_config.yaml", "manifest.json"):
        assert (out / name).exists(), name
    assert not (out / "metrics.partial.jsonl").exists()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    arms = {(r["arm"], r["rho"], r["f"], r["K"]) for r in rows}
    assert ("P+compute", 1.0, 1.0, 40) in arms and ("H+R", 0.0, 0.5, 40) in arms and ("H", 1.0, 1.0, 60) in arms
    assert ("H-rand", 0.0, 1.0, 40) not in arms
    pre = summary["preregistered"]
    assert set(pre["e10_9b_gate"]) == {"aulc_H_minus_P_ci_excludes_0", "aulc_H_minus_Pcompute_ci_excludes_0", "positive"}
    text = (out / "report.md").read_text()
    assert "Pre-registered readings" in text and "E10.9b gate" in text and "K sensitivity" in text
    A.main(["--report", str(out)])
    assert (out / "report.md").read_text() == text
