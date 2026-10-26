"""E15 stage 1e (`experiments/e15-hrr-target/preregistration.md`, "Stage 1e: read, then inject"): the injection path
(shapes, gradients to W / P / gate / host from the LM-side loss, the read loss still trains W, the gate near-closed at
initialisation, the fillers never read), the gate-closed evaluation equals the model without its hook, the regression
that a stage-1 config is untouched, the six configs, tiny end-to-end training with the injection on compose and free
channels, the `--gate-closed` flag, and `report --stage 1e` on synthetic folders with the three decision readings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from test_e15 import ATOMS, TARGET, _config, _store, world  # noqa: E402,F401  (the toy world and store of the stage-1 tests)
from test_e15_stage1b import _write_qa, _write_read, _write_windows  # noqa: E402
from vsa_embed.data.corpus import TokenCorpus, sample_batch  # noqa: E402
from vsa_embed.experiments import e15_eval  # noqa: E402
from vsa_embed.hrr_target import FactInjection, HRRTarget, injection_closed, injection_record, set_injection  # noqa: E402
from vsa_embed.training import lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
E15 = REPO / "experiments" / "e15-hrr-target"
INJECT = {**TARGET, "layer": 0, "inject": {"gate_bias": -4.0}}      # the toy host has 2 layers: capture at 0, inject at 1
STAGE1E = {"C5WRI": ("C5WR", (1, 2, 3), {"gate_bias": -4.0}), "C5WRI-nogate": ("C5WR", (1,), {"gate": False}),
           "C2RI": ("C2R", (1,), {"gate_bias": -4.0}), "C5shWRI": ("C5shWR", (1,), {"gate_bias": -4.0})}


# -- the injection path on a toy store ----------------------------------------------------------------------------------------

def test_injection_shapes_gradients_gate_and_content() -> None:
    store = _store(atoms=40, relations=3, dimension=64, concepts=6, degree=3, seed=0)
    target = HRRTarget(store, 24, layer=1, write_weight=1.0, read_weight=1.0)
    assert target.inject is None and target.injection(torch.zeros(1, 2, 24)) is None
    target.inject = FactInjection(64, 24, gate_bias=-4.0)
    assert target.inject.gate_value() == pytest.approx(1 / (1 + math.exp(4)), rel=1e-5)     # σ(−4) = 0.018: near-closed
    assert float(target.inject.project.weight.detach().abs().sum()) == 0.0                  # P zero-initialised
    hidden = torch.randn(2, 7, 24, requires_grad=True)
    spans = {"batch": torch.tensor([0, 0, 1]), "inject": torch.tensor([2, 5, 3]), "entry": torch.tensor([0, 3, 5])}
    target.current = spans
    target.capture(hidden)
    assert target.captured["rows"].tolist() == [0, 0, 1] and target.captured["cols"].tolist() == [2, 5, 3]
    later = torch.randn(2, 7, 24)                                                            # the input of layer ℓ + 1
    assert float(target.injection(later).detach().abs().sum()) == 0.0                        # at initialisation nothing is added
    with torch.no_grad():
        target.inject.project.weight.normal_()
    addition = target.injection(later)
    assert addition.shape == later.shape and addition.dtype == later.dtype
    assert (addition.abs().sum(-1) > 0).nonzero().tolist() == [[0, 2], [0, 5], [1, 3]]      # only the linked positions
    # Every parameter of the path receives gradient from a loss on the addition (the LM loss's role) — W, P, b, the host (h)
    # and the store; the read / write losses still reach W and keep the store detached.
    (addition.pow(2).sum()).backward(retain_graph=True)
    assert target.write.weight.grad is not None and target.inject.project.weight.grad.abs().sum() > 0
    assert target.inject.gate_logit.grad is not None and hidden.grad is not None and hidden.grad[0, 2].abs().sum() > 0
    assert hidden.grad[0, 0].abs().sum() == 0 and store.atomics.grad is not None
    store.zero_grad(); target.write.weight.grad = None
    parts = target.losses()
    (parts["write"] + parts["read"]).backward()
    assert target.write.weight.grad.abs().sum() > 0 and store.atomics.grad is None            # read loss trains W; store detached
    assert all(p.requires_grad for p in store.parameters())
    # Closed gate: nothing is added; `nogate`: g ≡ 1 and no gate parameter.
    target.current = spans; target.capture(hidden.detach())
    target.inject.enabled = False
    assert target.injection(later) is None
    target.inject.enabled = True
    nogate = FactInjection(64, 24, gated=False)
    assert nogate.gate_value() == 1.0 and [n for n, _ in nogate.named_parameters()] == ["project.weight"]
    assert nogate.record()["gated"] is False
    with pytest.raises(ValueError):
        FactInjection(64, 24, gate_bias=float("nan"))


def test_decoded_content_is_the_soft_cleanup_summed_over_the_frames_relations_and_never_reads_the_fillers() -> None:
    store = _store(atoms=40, relations=3, dimension=128, concepts=8, degree=3, seed=1)
    target = HRRTarget(store, 128, layer=1, read_weight=1.0)
    target.inject = FactInjection(128, 8)
    with torch.no_grad():
        target.write.weight.copy_(torch.eye(128))                                            # W h = the bundle
    entries = torch.arange(8)
    states = target.bundles(entries)
    content = target.decoded_content(states, entries)
    assert content.shape == (8, 128)
    atoms = torch.nn.functional.normalize(store.atomic_vectors(), dim=-1)
    schedule = store.schedule
    for e in range(8):
        frame = list(zip(schedule.relations[schedule.offsets[e]:schedule.offsets[e + 1]].tolist(),
                         schedule.fillers[schedule.offsets[e]:schedule.offsets[e + 1]].tolist()))
        manual = torch.zeros(128)
        for r, _ in frame:                                                                   # the relations only
            logits = target.read_logits(states[e:e + 1], torch.tensor([r]), torch.tensor([0]))
            manual += (torch.softmax(logits, -1) @ atoms)[0]
        torch.testing.assert_close(content[e], manual, atol=1e-4, rtol=1e-4)
        planted = [f for _, f in frame]
        cos = torch.nn.functional.cosine_similarity(content[e][None].detach(), atoms[planted].detach())
        others = torch.nn.functional.cosine_similarity(content[e][None].detach(), atoms[[i for i in range(40) if i not in planted]].detach())
        assert float(cos.min()) > float(others.max())                                        # a perfect decoder injects its fillers
    # Changing the fillers (not the relations) of the schedule changes the loss but not the injected content: the fillers are not read.
    shuffled = torch.roll(schedule.fillers, 1)
    store.set_schedule(type(schedule)(schedule.offsets, schedule.relations, shuffled))
    torch.testing.assert_close(target.decoded_content(states, entries), content)
    # An entry outside the schedule (zero-shot insertion of a free channel) gets a zero row; no span → None.
    assert float(target.decoded_content(states[:2], torch.tensor([0, 11])).detach().abs().sum(-1)[1]) == 0.0
    target.current = {"batch": torch.zeros(0, dtype=torch.long), "inject": torch.zeros(0, dtype=torch.long), "entry": torch.zeros(0, dtype=torch.long)}
    target.capture(torch.randn(1, 4, 128))
    assert target.injection(torch.randn(1, 4, 128)) is None


# -- integration: the hook, the regression, gate-closed evaluation --------------------------------------------------------------

def _build(config: dict, ontology: dict):
    torch.manual_seed(config["seed"])
    base = lm.build_model(config)
    channel, context = lm.build_channel(config, ontology, base.get_input_embeddings().weight.shape[1], host=base)
    channel.set_unseen(ontology["heldout_entries"])
    return lm.wrap_host(config, base, channel, context)


def test_a_stage1_config_is_unchanged_and_the_injection_only_adds_its_two_parameters(world) -> None:
    ontology = torch.load(world["root"] / "ontology.pt", weights_only=False)
    plain = _build(lm.resolve_config(_config(world["root"], "compose", TARGET)), ontology)
    with_inject = _build(lm.resolve_config(_config(world["root"], "compose", INJECT)), ontology)
    assert plain.channel.target.inject is None and plain._inject_hook is None and injection_record(plain) is None
    assert isinstance(with_inject.channel.target.inject, FactInjection) and with_inject._inject_hook is not None
    assert injection_record(with_inject) == {"gate": pytest.approx(1 / (1 + math.exp(4))), "gate_bias": -4.0, "gated": True, "enabled": True,
                                             "layer": with_inject.channel.target.layer + 1}
    sa, sb = plain.state_dict(), with_inject.state_dict()
    assert set(sb) - set(sa) == {"channel.target.inject.project.weight", "channel.target.inject.gate_logit"}
    for key, value in sa.items():
        torch.testing.assert_close(value, sb[key], msg=key)                                  # W and every other parameter identical
    corpus = TokenCorpus.open(world["root"] / "train")
    ids, spans = sample_batch(corpus, seed=1234, step=0, micro_step=0, batch=2, length=48, min_subtokens=1)
    plain.eval(); with_inject.eval()
    with torch.no_grad():
        la, lb = plain(ids, spans=spans, labels=ids)["loss"], with_inject(ids, spans=spans, labels=ids)["loss"]
    torch.testing.assert_close(la, lb)                                                       # P = 0: the forward is unchanged at initialisation
    # `set_injection` / `injection_closed` only touch a run that has the path.
    assert not set_injection(plain, False)
    with injection_closed(plain) as had:
        assert not had
    with injection_closed(with_inject) as had:
        assert had and not with_inject.channel.target.inject.enabled
    assert with_inject.channel.target.inject.enabled


@pytest.mark.parametrize("mode,extra,inject", [("compose", {}, {"gate_bias": -4.0}), ("compose", {}, {"gate": False}),
                                               ("free", {"free_dimension": 6}, {"gate_bias": -4.0})])
def test_training_with_the_injection_logs_the_gate_saves_the_path_and_reloads(world, tmp_path: Path, mode: str, extra: dict, inject: dict) -> None:
    target = {**INJECT, "inject": inject, **({"write_weight": 0.0, "dimension": 16} if mode == "free" else {})}
    folder = tmp_path / f"{mode}-{'nogate' if inject.get('gate') is False else 'gate'}"
    assert lm.train(_config(world["root"], mode, target, **extra), folder)["steps"] == 5
    rows = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    assert rows and all({"lm_loss", "write_loss", "read_loss", "gate"} <= set(r) for r in rows)
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["read_loss"]) and r["read_loss"] > 0 for r in rows)
    first = rows[0]
    assert first["loss"] == pytest.approx(first["lm_loss"] + target["write_weight"] * first["write_loss"] + target["read_weight"] * first["read_loss"], rel=1e-4)
    if inject.get("gate") is False:
        assert all(r["gate"] == 1.0 for r in rows)
    else:
        assert first["gate"] == pytest.approx(1 / (1 + math.exp(4)), rel=1e-3) and all(0.0 < r["gate"] < 0.1 for r in rows)
    state = torch.load(folder / "final.pt", weights_only=False)["model"]
    assert "channel.target.inject.project.weight" in state and state["channel.target.inject.project.weight"].shape == (64, 16)
    assert ("channel.target.inject.gate_logit" in state) == (inject.get("gate") is not False)
    assert float(state["channel.target.inject.project.weight"].abs().sum()) > 0                 # P has moved from 0: the LM loss reaches it
    model = lm.load_final(folder / "final.pt")
    assert isinstance(model.channel.target.inject, FactInjection) and model._inject_hook is not None and model._target_hook is not None
    assert model.channel.target.inject.gated == (inject.get("gate") is not False)
    torch.testing.assert_close(model.channel.target.inject.project.weight, state["channel.target.inject.project.weight"])
    # Gate-closed evaluation equals the model without its injection hook, and differs from the model with it.
    corpus = TokenCorpus.open(world["root"] / "train")
    ids, spans = sample_batch(corpus, seed=1234, step=1, micro_step=0, batch=2, length=48, min_subtokens=1)
    assert spans["entry"].numel() > 0
    with torch.no_grad():
        on = model(ids, spans=spans, labels=ids)["loss"]
        with injection_closed(model):
            off = model(ids, spans=spans, labels=ids)["loss"]
        model._inject_hook.remove(); model._inject_hook = None
        without = model(ids, spans=spans, labels=ids)["loss"]
    torch.testing.assert_close(off, without)
    assert not torch.allclose(on, off)


def test_read_with_gate_closed_records_the_flag_and_the_injection(world, tmp_path: Path, monkeypatch) -> None:
    from vsa_embed.experiments import e11_read_to_learn as e11
    folder = tmp_path / "c5wri"
    lm.train(_config(world["root"], "compose", INJECT), folder)
    new = [{"concept": "e15n-0", "surface": "glimmerstack", "frame": [["is_a", "type:tool"], ["area", "area:health"], ["status", "status:pilot"]]}]
    items = tmp_path / "items"                   # the toy "new" read set lives in memory (its on-disk form needs E9 base items)
    read_set = e11.ReadSet(items, {"kind": "new", "track": "toy", "styles": ["prose"]}, new, [], [])
    monkeypatch.setattr(e11, "load_read_set", lambda path: read_set)
    splits = e15_eval.build_splits(world["ontology"], new, seen_count=3, seed=0)
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    for name, flag in (("open", []), ("closed", ["--gate-closed"])):
        e15_eval.main(["read", "--run", str(folder), "--splits", str(tmp_path / "splits.json"), "--new-items", str(items), "--output", str(tmp_path / name),
                       "--alias-table", str(world["root"] / "alias_table.json"), "--device", "cpu", "--batch-size", "4", "--resamples", "10", "--smoke", *flag])
    a, b = (json.loads((tmp_path / name / "read.json").read_text()) for name in ("open", "closed"))
    assert a["gate_closed"] is False and b["gate_closed"] is True
    assert a["injection"]["enabled"] is True and b["injection"]["enabled"] is False and a["injection"]["layer"] == a["layer"] + 1
    assert a["injection"]["gate"] == b["injection"]["gate"] and 0.0 < a["injection"]["gate"] < 0.1
    assert a["groups"]["compositional"]["trained"]["per_concept"] == b["groups"]["compositional"]["trained"]["per_concept"]   # the read layer is pre-injection
    assert yaml.safe_load((tmp_path / "closed" / "resolved_config.yaml").read_text())["gate_closed"] is True
    assert e15_eval.gate_of_run(folder, tmp_path / "open")["from"] == "read.json" and e15_eval.gate_of_run(folder)["from"] == "metrics.jsonl"
    assert e15_eval.gate_of_run(folder, tmp_path / "open")["gate"] == pytest.approx(e15_eval.gate_of_run(folder)["gate"], abs=1e-6)
    # A run without the path refuses the flag.
    plain = tmp_path / "c5wr"
    lm.train(_config(world["root"], "compose", TARGET), plain)
    with pytest.raises(SystemExit):
        e15_eval.main(["read", "--run", str(plain), "--splits", str(tmp_path / "splits.json"), "--new-items", str(items), "--output", str(tmp_path / "refused"),
                       "--alias-table", str(world["root"] / "alias_table.json"), "--device", "cpu", "--resamples", "10", "--smoke", "--gate-closed"])


# -- configs -----------------------------------------------------------------------------------------------------------------

def test_the_six_stage1e_configs_differ_from_their_stage1_base_only_by_the_inject_block() -> None:
    folder = E15 / "configs" / "t5"
    names = sorted(p.name for p in folder.glob("*.yaml") if yaml.safe_load(p.read_text()).get("e15_stage") == "1e")
    assert names == sorted(f"SmolLM2-360M-full-{arm}-s{k}.yaml" for arm, (_, seeds, _) in STAGE1E.items() for k in seeds) and len(names) == 6
    for arm, (base_arm, seeds, inject) in STAGE1E.items():
        for k in seeds:
            name = f"SmolLM2-360M-full-{arm}-s{k}.yaml"
            config = yaml.safe_load((folder / name).read_text())
            base = yaml.safe_load((folder / config["e15_base_config"]).read_text())
            assert config["e15_base_config"] == f"SmolLM2-360M-full-{base_arm}-s{k}.yaml" and "e15_stage" not in base   # a stage-1 base
            assert config["e15_arm"] == arm and config["seed"] == base["seed"] == k and config["experiment"] == f"e15-t5-{name[:-5]}"
            assert config["e15_base_arm"] == base["e15_base_arm"]
            for key in ("experiment", "e15_arm", "e15_stage", "e15_base_config"):
                config.pop(key)
            for key in ("experiment", "e15_arm"):
                base.pop(key)
            assert config["channel"]["target"].pop("inject") == inject
            assert config == base, name                 # everything else: λ_w, λ_r, τ, layer, detach, data, recipe, 50M tokens
            assert config["train"]["total_tokens"] == 50_000_000 and config["channel"]["target"]["read_weight"] == 0.035
            resolved = lm.resolve_config(yaml.safe_load((folder / name).read_text()))
            assert resolved["channel"]["target"]["inject"] == inject
    # The stage-1 config test still sees exactly its twelve files (they carry no `e15_stage`).
    assert len([p for p in folder.glob("*.yaml") if "e15_stage" not in yaml.safe_load(p.read_text())]) == 12
    assert e15_eval.STAGE1E_SEEDS == {arm: seeds for arm, (_, seeds, _) in STAGE1E.items()}


# -- the stage-1e report -----------------------------------------------------------------------------------------------------

def _write_metrics(folder: Path, gate_end: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "train", "step": s, "tokens": s * 65536, "loss": 3.0, "lm_loss": 2.8, "write_loss": 0.1, "read_loss": 3.0,
             "gate": 0.018 + (gate_end - 0.018) * s / 99} for s in range(1, 100)]
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_stage_1e_computes_the_primaries_the_gates_and_the_decision(tmp_path: Path) -> None:
    splits_path = E15 / "splits" / "t5-v1" / "splits.json"
    splits = e15_eval.load_splits(splits_path)
    concepts = [r["concept"] for r in splits["splits"]["compositional"]]
    rng = np.random.default_rng(0)
    host, root = "SmolLM2-360M", tmp_path
    runs, base, qa, rd = root / "runs", root / "base", root / "qa", root / "read"
    compose = ("none", "implicit:oracle", "explicit-rel:oracle")
    free = ("none", "explicit-rel:oracle")
    reads = lambda level: {"compositional": [level] * len(concepts), "seen": [0.75] * 300, "unseen_filler": [level] * 164}  # noqa: E731

    def write_c5wri(level: float, closed: float) -> None:
        for k in (1, 2, 3):
            _write_qa(qa / f"{host}-full-C5WRI-s{k}", concepts, {"implicit:oracle": level, "none": 0.2, "explicit-rel:oracle": 0.82}, compose, rng)
            _write_qa(qa / f"{host}-full-C5WRI-s{k}" / "gate-closed", concepts, {"implicit:oracle": closed, "none": 0.2, "explicit-rel:oracle": 0.82}, compose, rng)

    write_c5wri(0.56, 0.30)
    for k in (1, 2, 3):
        _write_qa(qa / f"{host}-full-C5WR-s{k}", concepts, {"implicit:oracle": 0.29, "none": 0.2, "explicit-rel:oracle": 0.82}, compose, rng)
        _write_qa(qa / f"{host}-full-C5-s{k}", concepts, {"implicit:oracle": 0.26, "none": 0.19, "explicit-rel:oracle": 0.84}, compose, rng)
        _write_read(rd / f"{host}-full-C5WRI-s{k}", reads(0.5)); _write_read(rd / f"{host}-full-C5WR-s{k}", reads(0.48))
        _write_windows(runs / f"{host}-full-C5WRI-s{k}", -0.2); _write_windows(runs / f"{host}-full-C5WR-s{k}", -0.1); _write_windows(base / f"{host}-full-C5-s{k}", 0.0)
        _write_metrics(runs / f"{host}-full-C5WRI-s{k}", 0.45)
    for arm, level, conds in (("C2RI", 0.2, free), ("C5shWRI", 0.22, compose), ("C5WRI-nogate", 0.5, compose), ("C2R", 0.2, free), ("C5shWR", 0.2, compose)):
        _write_qa(qa / f"{host}-full-{arm}-s1", concepts, {"implicit:oracle": level, "none": 0.2, "explicit-rel:oracle": 0.8}, conds, rng)
        _write_read(rd / f"{host}-full-{arm}-s1", reads(0.1 if arm.startswith("C2") else 0.45))
        _write_windows(runs / f"{host}-full-{arm}-s1", 0.3 if "sh" in arm else 0.0)
    _write_metrics(runs / f"{host}-full-C2RI-s1", 0.05); _write_metrics(runs / f"{host}-full-C5shWRI-s1", 0.3)
    (rd / f"{host}-full-C5WRI-s1" / "read.json").write_text(json.dumps({**json.loads((rd / f"{host}-full-C5WRI-s1" / "read.json").read_text()),
                                                                        "injection": {"gate": 0.52, "gate_bias": -4.0, "gated": True, "enabled": True, "layer": 12}}))
    out_dir = root / "report"
    e15_eval.main(["report", "--stage", "1e", "--runs-root", str(runs), "--baseline-root", str(base), "--qa-root", str(qa), "--read-root", str(rd),
                   "--splits", str(splits_path), "--output", str(out_dir), "--host", host, "--resamples", "200"])
    out = json.loads((out_dir / "report.json").read_text())
    assert out["stage"] == "1e" and out["missing"] == []
    assert [r["name"][:2] for r in out["primary"]] == ["P1", "P2", "P3"] and all("p_holm" in r for r in out["primary"])
    p1, p2, p3 = out["primary"]
    assert p1["n"] == 3 * 5 * len(concepts) and p1["mean"] > 0.2 and p1["b"] == "C5WR"
    assert p2["n"] == 5 * len(concepts) and p2["b_condition"] == "none" and p2["mean"] > 0.3
    assert p3["n"] == 3 * 5 * len(concepts) and p3["b"] == "C5WRI[gate closed]" and p3["mean"] > 0.2 and p3["ci_low"] > 0
    assert out["path_used"] is True and "stage 2" in out["decision"] and "WRI" in out["decision"]
    assert out["gates"]["C5WRI"]["s1"] == {"gate": 0.52, "gate_bias": -4.0, "gated": True, "from": "read.json"}
    assert out["gates"]["C5WRI"]["s2"]["from"] == "metrics.jsonl" and out["gates"]["C5WRI"]["s2"]["gate"] == pytest.approx(0.45, abs=0.01)
    assert "C5WRI-nogate" in out["gates"] and out["gates"]["C5WRI-nogate"] == {}                 # no metrics written for it: nothing to read
    assert out["read"]["C5WRI"]["compositional/trained"]["mean_value"] == pytest.approx(0.5)
    names = [r["name"] for r in out["secondary"]]
    assert any("read accuracy (trained W), compositional: C5WRI − C5WR" in n for n in names)
    assert next(r for r in out["secondary"] if r["name"].startswith("read accuracy (trained W), compositional: C5WRI − C5WR"))["mean"] == pytest.approx(0.02)
    assert any("C5shWRI − C5WRI" in n for n in names) and any("C5WRI-nogate − C5WRI" in n for n in names) and any("strict compositional" in n for n in names)
    assert any("explicit-rel:oracle" in n for n in names) and any("C5WRI[gate closed] − C5WR" in n for n in names)
    assert out["losses"]["C5WRI − C5"]["after_heldout"]["relative"] < 0 and out["losses"]["C5WRI − C5WR"]["after_heldout"]["relative"] < 0
    assert out["losses"]["C5shWRI − C5WRI (seed 1)"]["after_heldout"]["relative"] > 0 and "C5shWRI − C5shWR (seed 1)" in out["losses"]
    assert out["implicit_property_compositional"]["C2RI"] == pytest.approx(0.2, abs=0.05) and out["implicit_condition"]["C2RI"] == "none"
    text = (out_dir / "report.md").read_text()
    assert "## Primary" in text and "Decision rule reading" in text and "## Learned gate" in text and "0.5200" in text
    # The partial reading, and the closed reading (the gate-closed model matches the open one: P3 ≈ 0).
    ns = lambda word: argparse.Namespace(runs_root=runs, baseline_root=base, qa_root=qa, read_root=rd, splits=splits_path, output=root / word,  # noqa: E731
                                         host=host, resamples=200, seed=0)
    write_c5wri(0.42, 0.30)
    partial = e15_eval.run_report_1e(ns("partial"))
    assert partial["decision"].startswith("partial") and partial["path_used"] is True
    write_c5wri(0.30, 0.30)
    closed = e15_eval.run_report_1e(ns("closed"))
    assert closed["path_used"] is False and "not used" in closed["decision"] and "model size" in closed["decision"]
    # P3 > 0 but C5WRI ≤ C5WR also closes; a high level with a weak memorisation contrast has no pre-registered reading.
    write_c5wri(0.28, 0.15)
    assert "not used" in e15_eval.run_report_1e(ns("closed2"))["decision"]
    write_c5wri(0.56, 0.30)
    _write_qa(qa / f"{host}-full-C2RI-s1", concepts, {"none": 0.5, "explicit-rel:oracle": 0.8}, free, rng)
    assert "no pre-registered reading" in e15_eval.run_report_1e(ns("memo"))["decision"]


def test_decoded_content_under_bf16_autocast_as_on_the_gpu() -> None:
    """Regression for the 2026-10-26 queue failure: under autocast the cleanup matmul `softmax(logits) @ atoms` comes out
    bf16 while `content` (built from the fp32 codes) is fp32; `index_add` refuses mixed dtypes unless the cleanup is cast.
    CPU autocast reproduces the dtype mix the GPU runs hit."""
    store = _store(atoms=40, relations=3, dimension=64, concepts=6, degree=3, seed=2)
    target = HRRTarget(store, 64, layer=1, read_weight=1.0)
    target.inject = FactInjection(64, 8)
    entries = torch.arange(6)
    states = target.bundles(entries)
    plain = target.decoded_content(states, entries)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = target.decoded_content(states, entries)
    assert mixed.shape == (6, 64) and float(mixed.float().abs().sum()) > 0
    torch.testing.assert_close(mixed.float(), plain.float(), atol=5e-2, rtol=5e-2)
