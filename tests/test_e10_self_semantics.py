import copy
import json

import torch
import yaml

from vsa_embed.experiments import e10_self_semantics as E
from vsa_embed.synthetic_ontology import make_ontology_world


def _config() -> dict:
    config = yaml.safe_load(open("experiments/e10-self-semantics/e10-synthetic.yaml"))
    config["seeds"] = [3]
    config["workers"] = 1
    config["world"].update(roots=2, branching=[2, 4], attributes=10, wholes=4, parts=8, groups=2, members=5,
                           similar_pairs_per_family=1, regions=1, countries_per_region=2, cities_per_country=2,
                           dimension=32, split_fractions=[0.15, 0.15, 0.15])
    config["discovery"].update(steps_per_round=60, max_rounds=2, resamples=200)
    config["selftest"]["resamples"] = 200
    config["a"].update(settings=["fixed", "l2"], steps=40, chunk=16)
    config["b"].update(rates=[0.3], steps=60, acceptance_rates=[0.3])
    config["c"].update(modes=["absent"], methods=["additive", "all_at_once"], max_slots=6, all_at_once_slots=2)
    config["e"].update(k=[1, 4])
    config["seed_ontology"].update(conditions=["core"], max_rounds=2, settle_steps=40, max_slots=6)
    config["dream"].update(every=[0, 60], post_steps=60, consolidate_steps=60, passes_for_off=1, wrong_slot_edges=4)
    config["dream_settings"].update(refit_steps=20)
    config["variety"].update(views=[1], steps=60, sources=6)
    config["continual"].update(conditions=["additive"], steps_per_stage=60)
    return config


def test_synthetic_world_has_planted_logic() -> None:
    world = make_ontology_world(seed=0)
    has_part, part_of = world.pairs("has_part"), world.pairs("part_of")
    assert has_part == {(t, h) for h, t in part_of}
    similar = world.pairs("similar_to")
    assert all((t, h) in similar for h, t in similar)
    located = world.pairs("located_in")
    assert all((a, c) in located for a, b in located for b2, c in located if b == b2)
    obs = world.observation_splits
    assert set(obs["train"]) | set(obs["val"]) | set(obs["audit"]) == set(range(world.observations.shape[1]))
    assert not set(obs["audit"]) & (set(obs["train"]) | set(obs["val"]))


def test_learning_never_reads_the_audit_split() -> None:
    config = _config()
    world = E.build_world(config, 3)
    poisoned = copy.deepcopy(world)
    poisoned.observations[:, world.observation_splits["audit"]] = float("nan")
    results = []
    for w in (world, poisoned):
        scenario = E.hidden_scenario(w, hidden=config["c"]["hidden"], seed=3, coverage=0.7, distractor_ratio=1.0,
                                     max_slots=6)
        ctx = E.make_context(w, scenario, 3)
        assert torch.isfinite(ctx.train_targets).all() and torch.isfinite(ctx.heldout.observations).all()
        composer, result = E.run_method(w, scenario, config, 3, "additive", mode="absent")
        results.append(([(r["slot"], r["accept"], r["adopted"], r["size"]) for r in result["rounds"]],
                        composer.edge_masses().detach().clone(), composer.slot_role_vectors().detach().clone()))
    assert results[0][0] == results[1][0]
    torch.testing.assert_close(results[0][1], results[1][1])
    torch.testing.assert_close(results[0][2], results[1][2])


def test_corrupted_crystallized_slot_is_reopened_by_the_utility_revisit() -> None:
    from vsa_embed.ontology_discovery import revisit_crystallized
    config = _config()
    world = E.build_world(config, 3)
    scenario = E.hidden_scenario(world, hidden=config["c"]["hidden"], seed=3, coverage=1.0, distractor_ratio=0.5,
                                 max_slots=6)
    ctx = E.make_context(world, scenario, 3)
    composer, _ = E.run_method(world, scenario, config, 3, "oracle", mode="absent", matched_steps=200)
    slots = composer.slot_active.nonzero().flatten().tolist()
    for slot in slots:
        composer.crystallize(slot, min_mass=0.0)
    sizes = {s: int(composer.slot_members(s, hard_only=True).numel()) for s in slots}
    target = max(sizes, key=sizes.get)
    with torch.no_grad():                                   # a false belief: the operator no longer fits its pairs
        composer.slot_frozen_roles[target] = torch.nn.functional.normalize(torch.randn(world.dimension), dim=0)
        composer.edge_mass[composer.slot_members(target, hard_only=True)] = 1.0
    events = {e["slot"]: e for e in revisit_crystallized(composer, ctx)}
    assert events[target]["reopened"] and not bool(composer.slot_frozen[target])


def test_end_to_end_run_writes_the_run_folder(tmp_path) -> None:
    config = _config()
    out = tmp_path / "run"
    summary = E.run(config, out, workers=1)
    for name in ("metrics.jsonl", "summary.json", "report.md", "resolved_config.yaml", "manifest.json"):
        assert (out / name).exists()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    parts = {r["part"] for r in rows}
    assert {"a", "b", "c", "e", "seed", "dream", "variety", "continual", "d_edges"} <= parts
    assert "verdicts" in summary and "## (c)" in (out / "report.md").read_text()
