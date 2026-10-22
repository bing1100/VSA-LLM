"""E15 stage 1b (`experiments/e15-hrr-target/preregistration.md`, "Stage 1b"): the five long configs differ from their
stage-1 bases only in length, named checkpoints and (lr0.10) λ_r; `train.save_at_tokens` writes `checkpoint-<tokens>.pt`
files that every evaluation loads, and changes nothing when absent; `report --stage 1b` computes the curve, the three
primaries and the decision reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from test_e15 import TARGET, _config, world  # noqa: E402,F401  (the toy world fixture and config of the stage-1 tests)
from vsa_embed.experiments import e15_eval  # noqa: E402
from vsa_embed.training import lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
E15 = REPO / "experiments" / "e15-hrr-target"
CHECKPOINTS = [50_000_000, 100_000_000, 150_000_000, 200_000_000]


def test_the_five_long_configs_differ_from_their_stage1_base_only_in_length_checkpoints_and_lambda_r() -> None:
    folder = E15 / "configs" / "t5"
    names = sorted(p.name for p in folder.glob("*-long-*.yaml"))
    assert names == [f"SmolLM2-360M-full-{a}.yaml" for a in ("C2R-long-s1", "C5WR-long-lr0.10-s1", "C5WR-long-s1", "C5WR-long-s2", "C5WR-long-s3")]
    for name in names:
        config = yaml.safe_load((folder / name).read_text())
        base = yaml.safe_load((folder / config["e15_base_config"]).read_text())
        assert config["e15_stage"] == "1b" and config["experiment"] == f"e15-t5-{name[:-5]}" and name.endswith(f"-s{config['seed']}.yaml")
        assert base["seed"] == config["seed"] and base["e15_arm"] == config["e15_arm"].split("-long")[0]
        for key in ("experiment", "e15_arm", "e15_stage", "e15_base_config"):
            config.pop(key)
        for key in ("experiment", "e15_arm"):
            base.pop(key)
        assert config["train"].pop("save_at_tokens") == CHECKPOINTS
        assert config["train"]["total_tokens"] == 200_000_000 and base["train"]["total_tokens"] == 50_000_000
        config["train"]["total_tokens"] = base["train"]["total_tokens"]
        if "lr0.10" in name:
            assert config["channel"]["target"]["read_weight"] == 0.10 and base["channel"]["target"]["read_weight"] == 0.035
            config["channel"]["target"]["read_weight"] = base["channel"]["target"]["read_weight"]
        assert config == base, name                 # everything else: λ_w, τ, layer, data, splits-independent recipe, seeds
        resolved = lm.resolve_config(yaml.safe_load((folder / name).read_text()))
        assert resolved["train"]["save_at_tokens"] == CHECKPOINTS and resolved["train"]["warmup_tokens"] == 2_500_000
    # The schedule at 200M: warm-up unchanged (38 steps of 65,536 tokens), the cosine stretched over 3,051 steps.
    tokens_per_step = 1024 * 4 * 16
    assert 200_000_000 // tokens_per_step == 3051 and 2_500_000 // tokens_per_step == 38 and 50_000_000 // tokens_per_step == 762
    assert lm._lr(762, 762, 38, 1.0, 0.1) == pytest.approx(0.1, abs=1e-6)             # stage 1 ends at the floor
    assert 0.87 < lm._lr(763, 3051, 38, 1.0, 0.1) < 0.89                               # the long run at the 50M checkpoint


def test_save_at_tokens_writes_named_checkpoints_that_load_and_changes_nothing_when_absent(world, tmp_path: Path) -> None:
    from vsa_embed.experiments import e11_read_to_learn as e11
    from vsa_embed.experiments.e5_common import open_run
    tokens_per_step = 48 * 2                                       # seq_len × micro_batch × grad_accum of the toy config
    config = _config(world["root"], "compose", TARGET)
    config["train"]["save_at_tokens"] = [2 * tokens_per_step, 5 * tokens_per_step + 1]   # step 2; and one the run never reaches
    named = tmp_path / "named"
    assert lm.train(config, named)["steps"] == 5
    first, beyond = named / "checkpoint-192.pt", named / "checkpoint-481.pt"
    assert first.exists() and beyond.exists() and not (named / "checkpoint.pt").exists()
    state = torch.load(first, weights_only=False)
    assert state["step"] == 2 and state["tokens"] == 192 and state["nominal_tokens"] == 192 and set(state["model"]) == set(torch.load(named / "final.pt", weights_only=False)["model"])
    rows = [json.loads(l) for l in (named / "metrics.jsonl").read_text().splitlines() if '"checkpoint"' in l]
    assert [(r["nominal_tokens"], r["step"], r.get("final", False)) for r in rows] == [(192, 2, False), (481, 5, True)]
    final_bytes = (named / "final.pt").read_bytes()
    assert beyond.read_bytes() == final_bytes                       # the count beyond the end: the final state (a hard link)
    assert beyond.stat().st_ino == (named / "final.pt").stat().st_ino or beyond.stat().st_size == len(final_bytes)
    early = lm.load_final(first)
    late = lm.load_final(named / "final.pt")
    assert not torch.equal(early.channel.target.write.weight, late.channel.target.write.weight)   # an earlier state, not a copy
    # Every evaluation CLI loads it through `--checkpoint`: open_run + read accuracy on the named file.
    run = open_run(named, checkpoint="checkpoint-192.pt", device="cpu", batch_size=4, alias_table=world["root"] / "alias_table.json")
    assert run.describe()["checkpoint"] == "checkpoint-192.pt"
    torch.testing.assert_close(run.adapter.model.channel.target.write.weight, early.channel.target.write.weight)
    new = [{"concept": "e15n-0", "surface": "glimmerstack", "frame": [["is_a", "type:tool"], ["area", "area:health"], ["status", "status:pilot"]]}]
    read_set = e11.ReadSet(tmp_path, {"kind": "new", "track": "toy", "styles": ["prose"]}, new, [], [])
    splits = e15_eval.build_splits(world["ontology"], new, seen_count=3, seed=0)
    result = e15_eval.read_accuracy(run, splits, read_set, resamples=10, log=lambda m: None)
    assert 0.0 <= result["groups"]["compositional"]["trained"]["accuracy"] <= 1.0
    # Regression: without the key nothing is written, no rows are logged, and the training itself is unchanged.
    plain = tmp_path / "plain"
    lm.train(_config(world["root"], "compose", TARGET), plain)
    assert not list(plain.glob("checkpoint-*.pt"))
    assert not any('"checkpoint"' in l for l in (plain / "metrics.jsonl").read_text().splitlines())
    a, b = torch.load(plain / "final.pt", weights_only=False)["model"], torch.load(named / "final.pt", weights_only=False)["model"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)                 # saving mid-run perturbs nothing


def _write_read(folder: Path, per_concept: dict[str, list[float]], chance: float = 1 / 3493) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    groups = {name: {"concepts": len(v), "linked_share": 1.0, "trained": {"accuracy": float(np.mean(v)), "concept_accuracy": float(np.mean(v)),
                                                                           "per_concept": v, "edges": 3 * len(v)}} for name, v in per_concept.items()}
    (folder / "read.json").write_text(json.dumps({"layer": 11, "atoms": 3493, "chance": chance, "groups": groups}))


def _write_qa(folder: Path, concepts: list[str], correct: dict[str, float], conditions: tuple[str, ...], rng: np.random.Generator) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "predictions.jsonl").open("w") as handle:
        for c in concepts:
            for cond in conditions:
                for j in range(5):
                    handle.write(json.dumps({"id": f"{c}-p{j}", "concept": c, "condition": cond, "test": "property",
                                             "correct": float(rng.random() < correct.get(cond, 0.2)), "consistent": 1.0}) + "\n")
                handle.write(json.dumps({"id": f"{c}-e", "concept": c, "condition": cond, "test": "entailment", "correct": 1.0}) + "\n")
    (folder / "summary.json").write_text(json.dumps({"conditions": list(conditions)}))


def _write_windows(folder: Path, offset: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    starts = np.arange(8)
    np.savez_compressed(folder / "eval_windows.npz", strata=np.array(["all", "unlinked", "after", "after_heldout"]), starts=starts,
                        sum_50=np.full((4, 8), 10.0 + offset), count_50=np.full((4, 8), 10.0))


def test_report_stage_1b_computes_the_curve_the_primaries_and_the_decision(tmp_path: Path) -> None:
    splits_path = E15 / "splits" / "t5-v1" / "splits.json"
    splits = e15_eval.load_splits(splits_path)
    concepts = [r["concept"] for r in splits["splits"]["compositional"]]
    strict = [r["concept"] for r in splits["splits"]["compositional_strict"]]
    unseen = [r["concept"] for r in splits["splits"]["unseen_filler"] if r["kind"] == "new"]
    rng = np.random.default_rng(0)
    host, root = "SmolLM2-360M", tmp_path
    runs, base, qa, rd = root / "runs", root / "base", root / "qa", root / "read"
    levels = {50_000_000: 0.45, 100_000_000: 0.55, 150_000_000: 0.62, 200_000_000: 0.68}   # rising, inside [0.6, 0.8) at the end
    for k in (1, 2, 3):
        for t, level in levels.items():
            acc = {"compositional": [float(min(1.0, max(0.0, level + rng.normal(0, 0.05)))) for _ in concepts],
                   "compositional_strict": [level] * len(strict), "unseen_filler": [level - 0.1] * len(unseen), "seen": [0.9] * 300}
            _write_read(rd / f"{host}-full-C5WR-long-s{k}" / f"ckpt-{t}", acc)
            _write_qa(qa / f"{host}-full-C5WR-long-s{k}" / f"ckpt-{t}", concepts, {"implicit:oracle": 0.2 + level / 4, "none": 0.2, "explicit-rel:oracle": 0.8},
                      ("none", "implicit:oracle", "explicit-rel:oracle"), rng)
        _write_qa(qa / f"{host}-full-C5-s{k}", concepts, {"implicit:oracle": 0.25, "none": 0.2, "explicit-rel:oracle": 0.8},
                  ("none", "implicit:oracle", "explicit-rel:oracle"), rng)
        _write_read(rd / f"{host}-full-C5WR-s{k}", {"compositional": [0.47] * len(concepts)})
        _write_windows(runs / f"{host}-full-C5WR-long-s{k}", 0.0); _write_windows(base / f"{host}-full-C5-s{k}", 0.2)
        _write_windows(runs / f"{host}-full-C5WR-s{k}", 0.1)
    _write_read(rd / f"{host}-full-C2R-long-s1" / "ckpt-200000000", {"compositional": [0.1] * len(concepts)})
    _write_qa(qa / f"{host}-full-C2R-long-s1" / "ckpt-200000000", concepts, {"none": 0.2, "explicit-rel:oracle": 0.8}, ("none", "explicit-rel:oracle"), rng)
    _write_read(rd / f"{host}-full-C5WR-long-lr0.10-s1" / "ckpt-200000000", {"compositional": [0.75] * len(concepts)})
    _write_windows(runs / f"{host}-full-C2R-long-s1", 0.0); _write_windows(runs / f"{host}-full-C5WR-long-lr0.10-s1", 0.3)
    out_dir = root / "report"
    e15_eval.main(["report", "--stage", "1b", "--runs-root", str(runs), "--baseline-root", str(base), "--qa-root", str(qa), "--read-root", str(rd),
                   "--splits", str(splits_path), "--output", str(out_dir), "--host", host, "--resamples", "200"])
    out = json.loads((out_dir / "report.json").read_text())
    assert out["stage"] == "1b" and out["missing"] == [] and out["best_checkpoint_by_read"] == 200_000_000
    assert [r["name"][:2] for r in out["primary"]] == ["P1", "P2", "P3"] and all("p_holm" in r for r in out["primary"])
    assert out["primary"][0]["checkpoint"] == 200_000_000 and out["primary"][0]["n"] == 3 * 5 * len(concepts)
    assert 0.6 <= out["primary"][1]["mean_value"] < 0.8 and out["primary"][2]["mean_value"] > 0.05      # ≈ +0.077 per 50M
    assert out["increment"]["still_rising"] and out["increment"]["from"] == 150_000_000
    assert "stage 1c" in out["decision"] and "stage 2" not in out["decision"]                          # QA gain ≈ +0.12 < 0.15
    assert list(out["curve"]) == [str(t) for t in CHECKPOINTS] and out["consistency_50M_vs_stage1"]["n"] == 3 * len(concepts)
    assert out["losses"]["C5WR-long − C5"]["after_heldout"]["relative"] < 0 and "lr0.10 − C5WR-long s1" in out["losses"]
    names = [r["name"] for r in out["secondary"]]
    assert any("C2R-long − C5WR-long" in n for n in names) and any("lr0.10 − C5WR-long" in n for n in names) and any("unseen-filler" in n for n in names)
    text = (out_dir / "report.md").read_text()
    assert "## The curve" in text and "still rising" in text and "Decision rule reading" in text
    # A flat, low curve reads as the closed route; a high one as learnable to the original standard.
    for k in (1, 2, 3):
        for t in levels:
            _write_read(rd / f"{host}-full-C5WR-long-s{k}" / f"ckpt-{t}", {"compositional": [float(0.3 + rng.normal(0, 0.05)) for _ in concepts]})
    flat = e15_eval.run_report_1b(argparse.Namespace(runs_root=runs, baseline_root=base, qa_root=qa, read_root=rd, splits=splits_path,
                                                     output=root / "flat", host=host, resamples=200, seed=0, checkpoints=",".join(map(str, CHECKPOINTS))))
    assert "implicit route is closed" in flat["decision"]
