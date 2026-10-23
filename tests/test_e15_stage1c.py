"""E15 stage 1c (`experiments/e15-hrr-target/preregistration.md`, "Stage 1c"): route distillation and fact verbalisation
(`aligned_targets`), the C5-long control of stage 1b, the configs, the regression that a config without the blocks is
unchanged, and `report --stage 1c`."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from test_e15 import ATOMS, FRAMES, RELATIONS, _config, world  # noqa: E402,F401  (the toy world of the stage-1 tests)
from test_e15_stage1b import CHECKPOINTS, _write_qa, _write_read, _write_windows  # noqa: E402
from vsa_embed import aligned_targets as at  # noqa: E402
from vsa_embed.data.corpus import TokenCorpus, sample_batch  # noqa: E402
from vsa_embed.experiments import e15_eval  # noqa: E402
from vsa_embed.training import lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
E15 = REPO / "experiments" / "e15-hrr-target"
DISTILL = {"weight": 0.5, "k": 8, "max_entries": 8, "max_prefix_tokens": 384}
VERBALIZE = {"weight": 0.5, "max_facts": 6, "max_prompts": 64}


# -- templates ---------------------------------------------------------------------------------------------------------

def _qa_templates() -> tuple[set[str], set[str]]:
    """(every E14 question-item template of the T5 item sets + the T5 lexicon's prompts, the RD-4e router wordings)."""
    from vsa_embed.experiments import e9_tracks as tracks
    from vsa_embed.experiments.e14_reading import ROUTE_PARAPHRASES
    qa: set[str] = set()
    files = glob.glob(str(REPO / "experiments/e9-retrofit/items/*t5*/items.jsonl")) + glob.glob(str(REPO / "experiments/e11-read-to-learn/items/t5-*/items.jsonl"))
    assert files
    for f in files:
        for line in open(f):
            qa.update(json.loads(line).get("templates", []))
    spec = tracks.track_spec("t5")
    if Path(spec.ontology).exists():
        ontology = torch.load(spec.ontology, weights_only=False)
        for s in tracks.lexicon_for(spec, ontology).templates.values():
            qa.update(s.prompts)
    return qa, {t for ts in ROUTE_PARAPHRASES.values() for t in ts}


def test_fv_templates_are_disjoint_from_the_e14_qa_item_templates() -> None:
    from vsa_embed.experiments.e14_reading import near_duplicate, wording_content
    qa, route = _qa_templates()
    assert len(qa) >= 40
    t5_relations = ["is_a", "area", "purpose", "status", "cadence", "tier", "subtype_of", "owned_by", "depends_on", "part_of", "measured_by",
                    "governed_by", "replaces", "uses", "produces", "reports_to", "computed_from", "reported_in", "approved_by", "applies_to",
                    "describes", "sponsored_by", "produced_by"]
    assert set(at.FV_TEMPLATES) == set(t5_relations)
    for relation, templates in at.FV_TEMPLATES.items():
        assert len(templates) >= 2 and len(set(templates)) == len(templates)
        for t in templates:
            assert t.startswith("{x}: ") and t.count("{x}") == 1 and len(wording_content(t)) >= 2, t
            assert t not in qa and t not in route
            clashes = [q for q in qa | route if near_duplicate(t, q)]
            assert not clashes, (relation, t, clashes)
    assert at.fv_templates("is_a") == at.FV_TEMPLATES["is_a"] and at.fv_templates("has_role") == ["{x}: has role field:", "{x}: recorded has role:"]


# -- the two objectives on the toy world ---------------------------------------------------------------------------------

def _build(world, mode: str, distill: dict | None, verbalize: dict | None, **extra):
    config = lm.resolve_config(_config(world["root"], mode, None, **extra))
    config["model"]["max_positions"] = 512                     # the toy GPT-2 must have room for the teacher's prefix
    if distill is not None:
        config["channel"]["distill"] = distill
    if verbalize is not None:
        config["channel"]["verbalize"] = verbalize
    ontology = torch.load(world["root"] / "ontology.pt", weights_only=False)
    torch.manual_seed(config["seed"])
    base = lm.build_model(config)
    channel, context = lm.build_channel(config, ontology, base.get_input_embeddings().weight.shape[1], host=base)
    channel.set_unseen(ontology["heldout_entries"])
    model = lm.wrap_host(config, base, channel, context)
    aligned = at.aligned_from_config(config, channel, ontology, tokenizer_name="gpt2", host=base) if (distill or verbalize) else None
    return config, model, aligned


def _batch(world, step: int = 0):
    corpus = TokenCorpus.open(world["root"] / "train")
    for s in range(step, step + 50):
        ids, spans = sample_batch(corpus, seed=1234, step=s, micro_step=0, batch=2, length=48, min_subtokens=1)
        if spans["entry"].numel() >= 2:
            return ids, spans
    raise AssertionError("no linked batch")


def test_the_teacher_sees_recall_lines_before_the_window_and_the_student_input_is_untouched(world) -> None:
    _, model, aligned = _build(world, "compose", DISTILL, None)
    model.eval()
    ids, spans = _batch(world)
    original = ids.clone()
    batch = aligned.teacher(model, ids, spans)
    assert batch is not None and batch.logprobs is not None
    # No leak: the student's tensor is untouched and holds no recall text; the teacher's input is prefix + window.
    assert torch.equal(ids, original)
    for b in range(ids.shape[0]):
        p = batch.prefix_lengths[b]
        assert torch.equal(batch.ids[b, p:p + ids.shape[1]], ids[b])
        student_text = aligned.tokenizer.decode(ids[b].tolist())
        assert "recall(" not in student_text and "(1.00)" not in student_text
        if p:
            prefix_text = aligned.tokenizer.decode(batch.ids[b, :p].tolist())
            assert prefix_text.startswith("recall(") and "(1.00)" in prefix_text and prefix_text == batch.prefix_texts[b] + "\n\n"
            gold = at.FrameTable(world["ontology"]["offsets"], world["ontology"]["relations"], world["ontology"]["fillers"])
            for entry in batch.covered_entries[b]:
                assert len(gold.frame(entry)) == 3 and gold.frame(entry) in FRAMES.values()        # entries are the table's order
                for r, f in gold.frame(entry):
                    assert ATOMS[f].split(":")[1] in prefix_text                      # every gold filler is in the recall lines
            assert "(1.00)" in prefix_text and sum(l.startswith("- ") for l in prefix_text.splitlines()) == 3 * len(batch.covered_entries[b])
    # Spans shift by the prefix; the distilled positions are the K after each prefixed term, inside the window.
    shift = torch.tensor(batch.prefix_lengths)[spans["batch"]]
    for key in ("start", "end", "inject"):
        assert torch.equal(batch.spans[key], spans[key] + shift)
    assert torch.equal(batch.spans["entry"], spans["entry"])
    assert batch.rows.shape == batch.student_cols.shape == batch.teacher_cols.shape
    assert batch.rows.numel() <= DISTILL["k"] * spans["entry"].numel() and batch.rows.numel() > 0
    assert torch.equal(batch.teacher_cols - batch.student_cols, torch.tensor(batch.prefix_lengths)[batch.rows])
    ends = {(int(b), int(e)) for b, e in zip(spans["batch"], spans["end"])}
    for b, c in zip(batch.rows.tolist(), batch.student_cols.tolist()):
        assert any(bb == b and e <= c < e + DISTILL["k"] for bb, e in ends) and c <= ids.shape[1] - 2
    # KL shapes and values: (positions, vocab) log-probabilities; KL ≥ 0; a student equal to the teacher gives 0.
    vocab = model.model.get_output_embeddings().weight.shape[0]
    assert batch.logprobs.shape == (batch.rows.numel(), vocab) and torch.allclose(batch.logprobs.exp().sum(-1), torch.ones(batch.rows.numel()), atol=1e-4)
    hidden = model(ids, spans=spans)["hidden"]
    kl, n = aligned.distill_loss(model, hidden, batch)
    assert n == batch.rows.numel() and kl.ndim == 0 and torch.isfinite(kl) and float(kl) >= 0
    teacher_hidden = model.hidden_states(batch.ids, spans=batch.spans)
    same, _ = aligned.distill_loss(model, teacher_hidden[:, :ids.shape[1]].clone().index_put_(
        (batch.rows, batch.student_cols), teacher_hidden[batch.rows, batch.teacher_cols]), batch)
    assert float(same) == pytest.approx(0.0, abs=1e-5)
    # Gradient reaches the host and the channel through the student only.
    model.train()
    kl, _ = aligned.distill_loss(model, model(ids, spans=spans)["hidden"], batch)
    kl.backward()
    assert model.channel.composer.atomics.grad is not None and any(p.grad is not None for p in model.model.parameters())
    # Without linked spans there is nothing to distil.
    assert aligned.teacher(model, ids, {k: v[:0] for k, v in spans.items()}) is None
    assert aligned.distill_loss(model, hidden, None)[1] == 0


def test_cloze_prompts_hold_the_term_tokens_a_template_suffix_and_the_filler_name_as_target(world) -> None:
    _, model, aligned = _build(world, "compose", None, {**VERBALIZE, "max_facts": 2})
    ids, spans = _batch(world)
    batch = aligned.cloze_batch(ids, spans, step=3, micro_step=1)
    assert batch is not None and 0 < len(batch.prompts) <= 2 * spans["entry"].numel()
    per_occurrence: dict[tuple[int, int], int] = {}
    for j, p in enumerate(batch.prompts):
        n = p["term_tokens"]
        assert torch.equal(batch.ids[j, :n], ids[p["row"], p["start"]:p["end"] + 1])                 # the term's own tokens
        suffix = p["template"].split("{x}", 1)[1]
        text = aligned.tokenizer.decode(batch.ids[j, :n + len(aligned.encode(suffix))].tolist())
        assert text.endswith(suffix) and p["template"] in at.fv_templates(RELATIONS[p["relation"]])
        target = batch.labels[j][batch.labels[j] != -100]
        assert aligned.tokenizer.decode(target.tolist()) == " " + ATOMS[p["filler"]].split(":")[1]  # the filler's name
        assert (batch.labels[j, :n + len(aligned.encode(suffix))] == -100).all() and target.numel() == p["target_tokens"]
        assert (p["relation"], p["filler"]) in aligned.injected.frame(p["entry"]) and aligned.injected.frame(p["entry"]) in FRAMES.values()
        per_occurrence[(p["row"], p["start"])] = per_occurrence.get((p["row"], p["start"]), 0) + 1
    assert max(per_occurrence.values()) <= 2                                                        # the cap on facts per occurrence
    assert torch.equal(batch.spans["inject"], batch.spans["end"]) and (batch.spans["start"] == 0).all()
    assert torch.equal(batch.spans["length"], batch.spans["end"] + 1) and torch.equal(batch.spans["batch"], torch.arange(len(batch.prompts)))
    # Deterministic per (seed, step, micro-step); a different step may draw other templates / facts.
    again = aligned.cloze_batch(ids, spans, step=3, micro_step=1)
    assert torch.equal(again.ids, batch.ids) and torch.equal(again.labels, batch.labels)
    loss, n = aligned.cloze_loss(model, batch, ids)
    assert n == len(batch.prompts) and loss.ndim == 0 and torch.isfinite(loss) and float(loss) > 0
    # The prompt cap is a seeded subset.
    small = at.AlignedObjectives(aligned.tokenizer, aligned.ontology, gold=aligned.gold, injected=aligned.injected, writer=aligned.writer,
                                 distill=None, verbalize={**VERBALIZE, "max_prompts": 3}, seed=0)
    assert len(small.cloze_batch(ids, spans, step=0, micro_step=0).prompts) == 3
    assert aligned.cloze_batch(ids, {k: v[:0] for k, v in spans.items()}, step=0, micro_step=0) is None


@pytest.mark.parametrize("mode,extra,distill,verbalize", [("compose", {}, DISTILL, None), ("compose", {}, None, VERBALIZE),
                                                           ("free", {"free_dimension": 6}, DISTILL, None), ("free", {"free_dimension": 6}, None, VERBALIZE),
                                                           ("compose", {}, DISTILL, VERBALIZE)])
def test_training_with_each_block_logs_its_loss_and_reloads(world, tmp_path: Path, mode: str, extra: dict, distill, verbalize) -> None:
    config = _config(world["root"], mode, None, **extra)
    config["model"]["max_positions"] = 512
    if distill:
        config["channel"]["distill"] = distill
    if verbalize:
        config["channel"]["verbalize"] = verbalize
    folder = tmp_path / "run"
    assert lm.train(config, folder)["steps"] == 5
    rows = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    assert rows and all("lm_loss" in r and np.isfinite(r["loss"]) for r in rows)
    assert all(("kl_loss" in r) == bool(distill) and ("cloze_loss" in r) == bool(verbalize) for r in rows)
    assert not any("write_loss" in r for r in rows)
    for r in rows:
        total = r["lm_loss"] + (distill["weight"] * r["kl_loss"] if distill else 0.0) + (verbalize["weight"] * r["cloze_loss"] if verbalize else 0.0)
        assert r["loss"] == pytest.approx(total, rel=1e-4, abs=1e-6)
    if distill:
        assert any(r["kl_loss"] > 0 for r in rows)
    if verbalize:
        assert all(r["cloze_loss"] > 0 for r in rows)
    state = torch.load(folder / "final.pt", weights_only=False)["model"]
    assert not any("aligned" in k or "target" in k for k in state)                 # no parameters added
    model = lm.load_final(folder / "final.pt")
    assert model.channel is not None and not hasattr(model.channel, "target")


def test_a_config_without_the_blocks_is_unchanged(world, tmp_path: Path) -> None:
    """Regression: same parameters, same LM loss on the same batch, no extra metric columns, no objectives built."""
    _, plain, none = _build(world, "compose", None, None)
    _, both, aligned = _build(world, "compose", DISTILL, VERBALIZE)
    assert none is None and aligned is not None
    sa, sb = plain.state_dict(), both.state_dict()
    assert set(sa) == set(sb)
    for key, value in sa.items():
        torch.testing.assert_close(value, sb[key], msg=key)
    ids, spans = _batch(world)
    plain.eval(); both.eval()
    with torch.no_grad():
        torch.testing.assert_close(plain(ids, spans=spans, labels=ids)["loss"], both(ids, spans=spans, labels=ids)["loss"])
    folder = tmp_path / "plain"
    lm.train(_config(world["root"], "compose", None), folder)
    rows = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    assert rows and not any(k in r for r in rows for k in ("lm_loss", "kl_loss", "cloze_loss"))
    # The regression on the toy host: two plain runs are byte-identical (the loop is a pure function of the config).
    again = tmp_path / "again"
    lm.train(_config(world["root"], "compose", None), again)
    a, b = torch.load(folder / "final.pt", weights_only=False)["model"], torch.load(again / "final.pt", weights_only=False)["model"]
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert [r["loss"] for r in rows] == [json.loads(l)["loss"] for l in (again / "metrics.jsonl").read_text().splitlines() if '"train"' in l]


def test_the_block_validation_rejects_bad_weights(world) -> None:
    with pytest.raises(ValueError):
        _build(world, "compose", {"weight": 0.0}, None)
    with pytest.raises(ValueError):
        _build(world, "compose", None, {"weight": 0.3, "max_facts": 0})


# -- the configs -------------------------------------------------------------------------------------------------------

E9 = REPO / "experiments" / "e9-retrofit" / "configs" / "t5"
STAGE1C = {"C5RD": ("C5", "distill", (1, 2, 3)), "C5FV": ("C5", "verbalize", (1, 2, 3)), "C2RD": ("C2", "distill", (1,)), "C2FV": ("C2", "verbalize", (1,))}


def _strip_labels(config: dict, keys=("experiment", "e15_arm", "e15_base_arm", "e15_stage", "e15_base_config")) -> dict:
    return {k: v for k, v in config.items() if k not in keys}


def test_the_eight_stage1c_configs_differ_from_their_e9_base_only_by_the_block() -> None:
    folder = E15 / "configs" / "t5"
    weights: dict[str, set[float]] = {"distill": set(), "verbalize": set()}
    for arm, (base_arm, block, seeds) in STAGE1C.items():
        for k in seeds:
            name = f"SmolLM2-360M-full-{arm}-s{k}.yaml"
            config = yaml.safe_load((folder / name).read_text())
            base = yaml.safe_load((E9 / f"SmolLM2-360M-full-{base_arm}-s{k}.yaml").read_text())
            assert config["e15_stage"] == "1c" and config["e15_arm"] == arm and config["e15_base_arm"] == base_arm and config["seed"] == k
            assert config["experiment"] == f"e15-t5-SmolLM2-360M-full-{arm}-s{k}" and Path(config["e15_base_config"]) == (E9 / f"SmolLM2-360M-full-{base_arm}-s{k}.yaml").relative_to(REPO)
            other = "verbalize" if block == "distill" else "distill"
            assert other not in config["channel"] and "target" not in config["channel"]
            spec = config["channel"].pop(block)
            weights[block].add(float(spec["weight"]))
            assert spec["weight"] > 0
            if block == "distill":
                assert spec == {"weight": spec["weight"], "k": 8, "max_entries": 8, "max_prefix_tokens": 384}
            else:
                assert spec == {"weight": spec["weight"], "max_facts": 6, "max_prompts": 64}
            assert _strip_labels(config) == _strip_labels(base), name        # same recipe: 50M tokens, lrs, data, seeds, channel
            assert config["train"]["total_tokens"] == 50_000_000
    assert len(weights["distill"]) == 1 and len(weights["verbalize"]) == 1    # one λ_rd and one λ_fv for every arm and seed


def test_the_c5_long_control_config_differs_from_e9_c5_only_by_length_and_checkpoints() -> None:
    config = yaml.safe_load((E15 / "configs" / "t5" / "SmolLM2-360M-full-C5-long-s1.yaml").read_text())
    base = yaml.safe_load((E9 / "SmolLM2-360M-full-C5-s1.yaml").read_text())
    assert config["e15_stage"] == "1b" and config["e15_arm"] == "C5-long" and config["e15_base_arm"] == "C5" and config["seed"] == 1
    assert config["experiment"] == "e15-t5-SmolLM2-360M-full-C5-long-s1"
    assert config["train"].pop("save_at_tokens") == CHECKPOINTS
    assert config["train"]["total_tokens"] == 200_000_000 and base["train"]["total_tokens"] == 50_000_000
    config["train"]["total_tokens"] = base["train"]["total_tokens"]
    assert "target" not in config["channel"] and "distill" not in config["channel"] and "verbalize" not in config["channel"]
    assert _strip_labels(config) == _strip_labels(base)


# -- the stage-1c report -----------------------------------------------------------------------------------------------

def _write_metrics(folder: Path, kl_start: float, kl_end: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "train", "step": s, "tokens": s * 65536, "loss": 3.0, "lm_loss": 2.8, "kl_loss": kl_start + (kl_end - kl_start) * s / 99} for s in range(1, 100)]
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_stage_1c_computes_the_primaries_the_kl_ratio_and_the_decision(tmp_path: Path) -> None:
    splits_path = E15 / "splits" / "t5-v1" / "splits.json"
    splits = e15_eval.load_splits(splits_path)
    concepts = [r["concept"] for r in splits["splits"]["compositional"]]
    rng = np.random.default_rng(0)
    host, root = "SmolLM2-360M", tmp_path
    runs, base, qa, rd = root / "runs", root / "base", root / "qa", root / "read"
    compose = ("none", "implicit:oracle", "explicit-rel:oracle")
    probe = lambda level: {"compositional": [level] * len(concepts), "seen": [0.9] * 300}  # noqa: E731
    for k in (1, 2, 3):
        _write_qa(qa / f"{host}-full-C5RD-s{k}", concepts, {"implicit:oracle": 0.56, "none": 0.2, "explicit-rel:oracle": 0.8}, compose, rng)
        _write_qa(qa / f"{host}-full-C5FV-s{k}", concepts, {"implicit:oracle": 0.42, "none": 0.2, "explicit-rel:oracle": 0.8}, compose, rng)
        _write_qa(qa / f"{host}-full-C5-s{k}", concepts, {"implicit:oracle": 0.26, "none": 0.19, "explicit-rel:oracle": 0.84}, compose, rng)
        _write_qa(qa / f"{host}-full-C5WR-s{k}", concepts, {"implicit:oracle": 0.29, "none": 0.2, "explicit-rel:oracle": 0.82}, compose, rng)
        for arm in ("C5RD", "C5FV", "C5"):
            (rd / f"{host}-full-{arm}-s{k}").mkdir(parents=True, exist_ok=True)
            groups = {name: {"concepts": len(v), "probe": {"accuracy": float(np.mean(v)), "per_concept": v}} for name, v in probe(0.3 if arm != "C5" else 0.16).items()}
            (rd / f"{host}-full-{arm}-s{k}" / "read.json").write_text(json.dumps({"layer": 11, "atoms": 3493, "chance": 1 / 3493, "trained_map": False, "groups": groups}))
        _write_windows(runs / f"{host}-full-C5RD-s{k}", -0.3); _write_windows(runs / f"{host}-full-C5FV-s{k}", 0.1); _write_windows(base / f"{host}-full-C5-s{k}", 0.0)
        _write_metrics(runs / f"{host}-full-C5RD-s{k}", 0.8, 0.1)
    for arm in ("C2RD", "C2FV"):
        _write_qa(qa / f"{host}-full-{arm}-s1", concepts, {"none": 0.2, "explicit-rel:oracle": 0.8}, ("none", "explicit-rel:oracle"), rng)
        _write_windows(runs / f"{host}-full-{arm}-s1", 0.5)
    _write_qa(qa / f"{host}-full-C2-s1", concepts, {"none": 0.2, "explicit-rel:oracle": 0.8}, ("none", "explicit-rel:oracle"), rng)
    _write_windows(base / f"{host}-full-C2-s1", 0.5)
    _write_metrics(runs / f"{host}-full-C2RD-s1", 0.8, 0.5)
    out_dir = root / "report"
    e15_eval.main(["report", "--stage", "1c", "--runs-root", str(runs), "--baseline-root", str(base), "--qa-root", str(qa), "--read-root", str(rd),
                   "--splits", str(splits_path), "--output", str(out_dir), "--host", host, "--resamples", "200"])
    out = json.loads((out_dir / "report.json").read_text())
    assert out["stage"] == "1c" and all(not m.endswith("C5RD-s1") for m in out["missing"])
    assert [r["name"][:2] for r in out["primary"]] == ["P1", "P2", "P3"] and all("p_holm" in r for r in out["primary"])
    assert out["primary"][0]["mean"] > 0.25 and out["primary"][1]["mean"] > 0.1 and out["primary"][2]["mean"] > 0.3
    assert out["primary"][2]["n"] == 5 * len(concepts) and out["primary"][2]["b_condition"] == "none"
    assert out["implicit_property_compositional"]["C5RD"] > 0.5 and "stage 2" in out["decision"]
    assert out["kl_ratio"]["C5RD"]["ratio"] == pytest.approx(0.1 / 0.8, rel=0.25) and out["kl_ratio"]["C2RD"]["ratio"] > 0.5   # 5%-of-rows means
    assert out["losses_vs_C5"]["C5RD"]["after_heldout"]["relative"] < 0 and "C2FV" in out["losses_vs_C5"]
    names = [r["name"] for r in out["secondary"]]
    assert any("C5RD − C5WR" in n for n in names) and any("explicit-rel:oracle" in n for n in names) and any("strict compositional" in n for n in names)
    assert any("probe read accuracy" in n for n in names) and not any("unseen-filler" in n for n in names)   # no unseen-filler items in the synthetic QA
    assert out["read"]["C5RD"]["compositional/probe"]["mean_value"] == pytest.approx(0.3) and "compositional/trained" not in out["read"]["C5RD"]
    text = (out_dir / "report.md").read_text()
    assert "## Primary" in text and "Decision rule reading" in text and "teacher" in text.lower()
    # A partial and a closed reading.
    for level, word in ((0.42, "partial"), (0.30, "closed")):
        for k in (1, 2, 3):
            _write_qa(qa / f"{host}-full-C5RD-s{k}", concepts, {"implicit:oracle": level, "none": 0.2, "explicit-rel:oracle": 0.8}, compose, rng)
        again = e15_eval.run_report_1c(argparse.Namespace(runs_root=runs, baseline_root=base, qa_root=qa, read_root=rd, splits=splits_path,
                                                          output=root / word, host=host, resamples=200, seed=0))
        assert word in again["decision"]
