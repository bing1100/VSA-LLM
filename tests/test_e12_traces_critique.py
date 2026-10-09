"""E12 3b (`e12_traces`: training questions, the arms' texts, LoRA training, tests, report) and 3c (`e12_critique`:
calibration, evidence, null frames, the rule loop, K1 / K2, the model loop) on the toy E9 world of `test_e9_binding` (CPU,
tiny fake host, torch threads 1)."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from test_e9_binding import ATOMS, TERMS, NoWordNet, _frame, _lexicon, _ok, item_dirs, world  # noqa: E402,F401
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_binding_items as bi  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e9_understanding as und  # noqa: E402
from vsa_embed.experiments import e12_critique as crit  # noqa: E402
from vsa_embed.experiments import e12_self_query as sqx  # noqa: E402
from vsa_embed.experiments import e12_traces as traces  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")


@pytest.fixture(autouse=True)
def toy_lexicon(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(sqx, "lexicon_for_track", lambda track, family, ontology: _lexicon())


def _open(world, name: str) -> common.E5Run:
    return common.open_run(world["runs"][name], device="cpu", batch_size=8, max_length=64, alias_table=world["alias"])


@pytest.fixture(scope="module")
def new_dir(world, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("new") / "new-words-toy-v1"
    edit.build_new_word_items(world["root"] / "ontology.pt", out, tokenizer_name="gpt2", count=3, distractors=1, wordnet=NoWordNet(),
                              alias_table=world["alias"], lexicon=_lexicon(), min_subtokens=1, reserved_names=TERMS)
    return out


@pytest.fixture(scope="module")
def training_items(world, item_dirs, new_dir, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("refs")
    for path in (item_dirs["twins"], item_dirs["natural"], new_dir):
        shutil.copytree(path, root / path.name)
    out = tmp_path_factory.mktemp("traces") / "traces-toy-v1"
    manifest = traces.build_training_items(world["ctx"], out, test_twins=item_dirs["twins"], items_roots=[root], pairs=2, seed=1,
                                           name_seed=29, wordnet=NoWordNet(), min_shared=2)
    return {"dir": out, "manifest": manifest, "refs": traces.item_references([root], "toy")}


# -- 3b: training questions ----------------------------------------------------------------------------------------------------

def test_build_twins_exclusions_leave_the_default_draw_unchanged(world, item_dirs, tmp_path) -> None:
    again = tmp_path / "again"
    bi.build_twins(world["ctx"], again, count=4, seed=0, min_shared=2, wordnet=NoWordNet(), reserved_names=set(TERMS),
                   exclude_frames=[], exclude_fillers=[])
    for name in ("concepts.jsonl", "items.jsonl.gz"):
        assert (again / name).read_bytes() == (item_dirs["twins"] / name).read_bytes()
    assert "excluded" not in json.loads((again / "manifest.json").read_text())
    _, concepts, _ = bi.load_items(item_dirs["twins"])
    view = world["ctx"].view
    aid = {n: i for i, n in enumerate(view.atomic_names)}
    frames = [[(view.relation_id[r], aid[a]) for r, a in c["frame"]] for c in concepts]
    pairs = [(aid[c["fillers"]["X"]], aid[c["fillers"]["Y"]]) for c in concepts if c["twin"] == "A"]
    other = tmp_path / "other"
    manifest = bi.build_twins(world["ctx"], other, count=1, seed=3, min_shared=2, wordnet=NoWordNet(), reserved_names=set(TERMS),
                              exclude_frames=frames, exclude_fillers=pairs)
    _, built, _ = bi.load_items(other)
    assert manifest["excluded"]["filler_pairs"] == len({frozenset(p) for p in pairs})
    assert not {frozenset(map(tuple, c["frame"])) for c in built} & {frozenset(map(tuple, c["frame"])) for c in concepts}
    assert not {frozenset((c["fillers"]["X"], c["fillers"]["Y"])) for c in built} & {frozenset((c["fillers"]["X"], c["fillers"]["Y"])) for c in concepts}


def test_training_items_are_disjoint_from_every_test_set(world, item_dirs, training_items) -> None:
    manifest, refs = training_items["manifest"], training_items["refs"]
    assert manifest["schema"] == traces.SCHEMA and not any(manifest["checks"].values())
    _, questions, twins = traces.load_questions(training_items["dir"])
    sources = {q["source"] for q in questions}
    assert sources == {"twins", "seen"} and len([q for q in questions if q["source"] == "twins"]) == 2 * 2 * 2   # pairs × twins × relations
    assert {c["surface"].lower() for c in twins.concepts}.isdisjoint(refs["surfaces"])
    held = set(world["ontology"]["heldout_entries"])
    lexicon = _lexicon()
    for q in questions:
        assert q["template"] in lexicon.prompts("*", q["relation"]) and len(q["candidates"]) == 2
        if q["source"] == "seen":
            assert q["entry"] not in refs["entries"] and q["entry"] not in held
            gold = q["options"][q["gold"]]
            assert q["candidates"][q["gold"]] == lexicon.answer(q["relation"], lexicon.text(gold))
    assert len({(q["concept"], q["relation"]) for q in questions}) == len(questions)          # one question per (term, relation)
    again = traces.load_questions(training_items["dir"], limit=2)[1]
    assert len([q for q in again if q["source"] == "twins"]) == 2 and len([q for q in again if q["source"] == "seen"]) == 2


# -- 3b: the arms' texts and loss ----------------------------------------------------------------------------------------------

def _q() -> dict:
    return {"id": "q", "concept": "seen-3", "surface": "Brightwater Ledger", "relation": "depends_on", "template": "{x} depends on",
            "candidates": [" Grosh Console", " Quill Engine"], "gold": 0}


def test_arm_formats_put_the_loss_on_the_model_turns_only() -> None:
    q = _q()
    seq = traces.render(q, "I")
    assert seq.text == "Question: Brightwater Ledger depends on ___? Options: Grosh Console | Quill Engine\nAnswer: Grosh Console\n"
    assert [seq.text[a:b] for a, b in seq.spans] == [" Grosh Console"]
    obs = "recall(Brightwater Ledger, depends on):\n- Brightwater Ledger depends on Grosh Console. (0.91)"
    full = traces.render(q, "T", observation=obs, said="Grosh Console")
    turns = [full.text[a:b] for a, b in full.spans]
    assert turns == [" I should recall Brightwater Ledger, depends on.\nAction: recall[Brightwater Ledger, depends on]",
                     " The recall says Grosh Console.\nAnswer: Grosh Console"]
    assert "Observation: " + obs in full.text and not any("Observation" in t or "Question" in t for t in turns)
    without = traces.render(q, "T-obs", said="Grosh Console")
    assert "Observation" not in without.text and [without.text[a:b] for a, b in without.spans] == turns
    assert "neither option" in traces.render(q, "T", observation=obs, said=None).text
    with pytest.raises(ValueError):
        traces.render(q, "T")


def test_s_arm_is_t_then_t_without_observation_then_i(world, training_items) -> None:
    run = _open(world, "C5")
    _, questions, twins = traces.load_questions(training_items["dir"], limit=2)
    store = sqx.load_store(world["runs"]["C5"], "own")
    recaller = traces.Recaller(run, store, twins, _lexicon())
    epochs, info = traces.arm_epochs("S", questions, recaller)
    assert [e[0].meta["form"] for e in epochs] == ["T", "T-obs", "I"] and sum(info["recall_says"].values()) == len(questions)
    assert "Observation: recall(" in epochs[0][0].text and "Observation" not in epochs[1][0].text
    i_epochs, _ = traces.arm_epochs("I", questions)
    assert len(i_epochs) == traces.EPOCHS and i_epochs[0][0].text == epochs[2][0].text
    with sqx.host_view(run, twins) as (adapter, _):
        encoded = traces.encode(adapter, i_epochs[0][0])
    targets = [i for i, l in zip(encoded.ids, encoded.labels) if l != -100]
    assert adapter.tokenizer.decode(targets).strip() == questions[0]["candidates"][questions[0]["gold"]].strip()
    assert encoded.spans is not None and encoded.spans["entry"].numel() >= 1                     # the term is linked


def test_lora_training_moves_only_the_new_adapters(world, training_items) -> None:
    run = _open(world, "C5")
    _, questions, twins = traces.load_questions(training_items["dir"])
    text = "Brightwater Ledger depends on Grosh Console."
    adapter = run.adapter

    def logits() -> torch.Tensor:
        with torch.no_grad():
            ids = adapter.tokenizer(text, return_tensors="pt")["input_ids"]
            return run.model(ids)["hidden"]

    before = logits()
    channel = {k: v.clone() for k, v in run.channel.state_dict().items()}
    with sqx.host_view(run, twins) as (view, _):
        data = [[traces.encode(view, s) for s in epoch] for epoch in traces.arm_epochs("I", questions)[0]]
        params = traces.attach_lora(run.model)
        assert run.model.merged_lora > 0 and all(p.requires_grad for p in params)          # the toy run's own adapters merged
        assert torch.allclose(logits(), before, atol=1e-4)                                   # merged + zero-initialized: same function
        host = {n: p.clone() for n, p in run.model.model.named_parameters() if "lora_" not in n}
        trainable = [p for p in run.model.parameters() if p.requires_grad]
        assert len(trainable) == len(params)
        run.model.model.__dict__.pop("_lora_attached", None)
        result = _train_without_reattach(view, data, params)
    assert result["steps"] == 3 and result["loss_tokens"] > 0
    assert all(torch.equal(p, host[n]) for n, p in run.model.model.named_parameters() if n in host)
    assert all(torch.equal(v, channel[k]) for k, v in run.channel.state_dict().items() if k in channel)
    assert any(float(p.detach().abs().sum()) > 0 for n, p in run.model.named_parameters() if n.endswith("lora_b"))
    state = traces.lora_state(run.model)
    assert state and all(".lora_" in k for k in state)


def _train_without_reattach(adapter, data, params):
    """`traces.train` with `attach_lora` already applied (the test inspects the state in between)."""
    original = traces.attach_lora
    traces.attach_lora = lambda model, **_: params
    try:
        return traces.train(adapter, data, seed=1, lr=5e-3, per_step=4, micro=2, max_steps=3)
    finally:
        traces.attach_lora = original


def test_corpus_windows_match_the_requested_size(world) -> None:
    run = _open(world, "C5")
    epochs = traces.corpus_epochs(run, sequences=5, length=20, epochs=2, seed=7)
    assert [len(e) for e in epochs] == [5, 5] and all(len(r.ids) == 20 and r.labels == r.ids for e in epochs for r in e)
    assert [r.ids for r in epochs[0]] != [r.ids for r in epochs[1]]                       # fresh windows each epoch
    held = set(world["ontology"]["heldout_entries"])
    assert all(not (set(r.spans["entry"].tolist()) & held) for e in epochs for r in e)


def _arm_args(world, training_items, item_dirs, new_dir, model: str, arm: str) -> argparse.Namespace:
    parser_args = ["run", "--run", str(world["runs"][model]), "--arm", arm, "--train-items", str(training_items["dir"]),
                   "--twins", str(item_dirs["twins"]), "--new-words", str(new_dir), "--alias-table", str(world["alias"]), "--device", "cpu",
                   "--batch-size", "8", "--max-length", "64", "--micro-batch", "4", "--max-steps", "2", "--twin-limit", "2",
                   "--new-limit", "3", "--agent-pairs", "0", "--max-lines", "1", "--overwrite", "--label", "SMOKE"]
    return parser_args


def test_arm_jobs_and_the_b1_report(world, training_items, item_dirs, new_dir, tmp_path) -> None:
    # (the role-blind store's "associated with" line overflows the fake host's 64 positions: the C5ut arm is a copy of C5's,
    # which tests the report's arm labels; its training path is C5's)
    for model, arm in (("C5", "I"), ("C5", "L"), ("C5", "base")):
        traces.main(_arm_args(world, training_items, item_dirs, new_dir, model, arm))
        folder = world["runs"][model] / f"{traces.OUTPUT_PREFIX}{arm}"
        summary = json.loads((folder / "summary.json").read_text())
        assert {"twins", "new_words", "question_format"} <= set(summary["tests"])
        assert set(summary["tests"]["twins"]["summary"]["conditions"]) == {"none", "recall:own"}
        assert (folder / "adapters.pt").exists() == (arm != "base")
        if arm == "L":
            t = summary["training"]
            assert t["matched_sequences"] == t["sequences_per_epoch"][0] and t["window_length"] <= 64
        if arm != "base":
            assert summary["training"]["steps"] == 2 and len((folder / "training.jsonl").read_text().splitlines()) == 2
    shutil.copytree(world["runs"]["C5"] / f"{traces.OUTPUT_PREFIX}I", world["runs"]["C5ut"] / f"{traces.OUTPUT_PREFIX}I", dirs_exist_ok=True)
    analysis = traces.analyse(world["runs"]["C5"].parent)
    host = analysis["hosts"]["fake"]
    assert set(host["arms"]) >= {"I", "L", "base", "I-ut"} and "B1: I − L" in host["b1"]
    assert host["b1_reading"] and "I-ut − 0.5 (twins, no tool; > 0 = a leak)" in host["secondaries"]
    text = traces.render_report(analysis, title="toy", label="SMOKE")
    assert "B1" in text and "SMOKE" in text


def _doc(units: dict[str, float], cloze: dict[str, float] | None = None) -> dict:
    block = lambda u: {"units": {k: {"contrast": v} for k, v in u.items()}}
    return {"tests": {"twins": {"summary": {"conditions": {"none": {"choice": block(units), "cloze": block(cloze or units)}}}}}}


def test_b1_statistics_on_synthetic_units() -> None:
    rng = np.random.default_rng(0)
    pairs = [str(p) for p in range(80)]
    noisy = lambda m: {p: float(np.clip(m + rng.normal(0, 0.1), 0, 1)) for p in pairs}
    arms = {"I": {s: _doc(noisy(0.62), noisy(0.5)) for s in (1, 2, 3)}, "S": {s: _doc(noisy(0.5)) for s in (1, 2, 3)},
            "L": {s: _doc(noisy(0.5)) for s in (1, 2, 3)}, "I-ut": {s: _doc(noisy(0.5)) for s in (1, 2, 3)}}
    from vsa_embed.experiments.e12_report import _holm, contrast
    rows = {f"B1: {a} − L": contrast(traces.units(arms, a), traces.units(arms, "L")) for a in ("I", "S")}
    _holm(rows)
    assert rows["B1: I − L"]["p_holm"] < 0.05 and rows["B1: I − L"]["model"]["mean"] > 0.08
    assert rows["B1: S − L"]["p_holm"] > 0.05
    assert traces.b1_reading(rows).startswith("internalization shown by I − L")
    cloze = contrast(traces.units(arms, "I", kind="cloze"), traces.units(arms, "L", kind="cloze"))
    assert abs(cloze["model"]["mean"]) < 0.05                       # the refutation reading: learned the trained wording only
    assert traces.pair_contrast([{"gold": 0, "meta": {"pair": 0, "twin": "A"}, "scores": {"k": [0.0, -1.0]}},
                                 {"gold": 1, "meta": {"pair": 0, "twin": "B"}, "scores": {"k": [-1.0, 0.0]}}], "k") == {"0": 1.0}


# -- decision 64 (amendment 16.5): the C0′ host and the C5rf store ------------------------------------------------------------

def test_margin_tests_read_equivalence_and_noninferiority() -> None:
    from vsa_embed.experiments.e12_report import contrast, margin_reading, margin_test
    rng = np.random.default_rng(1)
    units = lambda m, sd=0.05: {s: {str(k): float(m + rng.normal(0, sd)) for k in range(200)} for s in (1, 2, 3)}
    near = margin_test(contrast(units(0.80), units(0.80)), 0.05, kind="equivalence")
    assert near["margin"]["shown"] and near["margin"]["ci90_low"] > -0.05 and margin_reading(near).startswith("≈")
    far = margin_test(contrast(units(0.80), units(0.65)), 0.05, kind="equivalence")
    assert not far["margin"]["shown"] and margin_reading(far).startswith("differs")
    ni = margin_test(contrast(units(0.78), units(0.80)), 0.05, kind="noninferiority")
    assert ni["margin"]["shown"] and ni["margin"]["alpha"] == 0.025 and margin_reading(ni).startswith("non-inferior at −0.05")
    worse = margin_test(contrast(units(0.60), units(0.80)), 0.05, kind="noninferiority")
    assert not worse["margin"]["shown"] and margin_reading(worse).startswith("inferior by more than the margin")
    with pytest.raises(ValueError):
        margin_test(contrast(units(0.8), units(0.8)), 0.05, kind="superiority")


def _d64_doc(none: dict[str, float], recall: dict[str, float], agent: dict[str, float] | None = None) -> dict:
    block = lambda u: {"units": {k: {"contrast": v} for k, v in u.items()}}
    tests = {"twins": {"summary": {"conditions": {"none": {"choice": block(none), "cloze": block(none)},
                                                  "recall:own": {"choice": block(recall)}}}}}
    if agent is not None:
        tests["agent"] = {"units": {k: {"contrast": v} for k, v in agent.items()}, "summary": {"all": {"format_ok": 0.95}}}
    return {"tests": tests}


def test_decision64_contrasts_and_predictions_on_synthetic_units() -> None:
    rng = np.random.default_rng(0)
    pairs = [str(p) for p in range(150)]
    noisy = lambda m: {p: float(np.clip(m + rng.normal(0, 0.03), 0, 1)) for p in pairs}
    arms = {"T": {s: _d64_doc(noisy(0.55), noisy(0.85), noisy(0.9)) for s in (1, 2, 3)},
            "base": {s: _d64_doc(noisy(0.5), noisy(0.8), noisy(0.6)) for s in (1, 2, 3)},
            "I": {s: _d64_doc(noisy(0.60), noisy(0.8)) for s in (1, 2, 3)}, "L": {s: _d64_doc(noisy(0.5), noisy(0.8)) for s in (1, 2, 3)},
            "T@C0p": {s: _d64_doc(noisy(0.5), noisy(0.85), noisy(0.9)) for s in (1, 2, 3)},
            "base@C0p": {s: _d64_doc(noisy(0.5), noisy(0.8), noisy(0.6)) for s in (1, 2, 3)},
            "T-rf": {s: _d64_doc(noisy(0.55), noisy(0.85), noisy(0.9)) for s in (1, 2, 3)},
            "I-rf": {s: _d64_doc(noisy(0.60), noisy(0.8)) for s in (1, 2, 3)}, "L-rf": {s: _d64_doc(noisy(0.5), noisy(0.8)) for s in (1, 2, 3)}}
    d64 = traces.decision64(arms, resamples=50)
    rows = d64["contrasts"]
    assert len(rows) == 5 and all(r["available"] and r["margin"]["kind"] == "equivalence" for r in rows.values())
    assert all(r["margin"]["shown"] and r["reading"].startswith("≈") for r in rows.values())
    b1 = rows["C5rf − C5, B1's contrast I − L (twins, no tool)"]
    assert b1["margin"]["margin"] == traces.B1_MARGIN and abs(b1["model"]["mean"]) < 0.02
    controls = d64["controls"]
    assert controls["T@C0p − 0.5 (twins, no tool; no channel: ≈ 0)"]["model"]["ci_low"] <= 0 <= controls["T@C0p − 0.5 (twins, no tool; no channel: ≈ 0)"]["model"]["ci_high"]
    assert controls["B1 on C5rf: I-rf − L-rf (twins, no tool)"]["model"]["mean"] > 0.08
    teach = controls["T@C0p − base@C0p (agentic twin contrast: the traces teach the C0′ host the protocol)"]
    assert teach["model"]["mean"] > 0.25
    predicted = traces.predictions({"means": {}, "secondaries": {}, "b2": {"descriptive": {}}, "decision64": d64})
    assert predicted["16.5: C0′ host ≈ C5 host with the decoded store (arm T, ±0.05)"] and predicted["16.5: C5rf ≈ C5 (B1's I − L ±0.075; arm T ±0.05)"]
    assert predicted["16.5: T@C0p without the tool ≈ 0.5 (CI includes 0.5)"]
    arms["T-rf"] = {s: _d64_doc(noisy(0.55), noisy(0.70), noisy(0.9)) for s in (1, 2, 3)}             # the fixed binding reads worse
    worse = traces.decision64(arms, resamples=50)["contrasts"]["C5rf − C5, arm T, twins recall:own (each reads its own store)"]
    assert not worse["margin"]["shown"] and worse["reading"].startswith("differs")


def test_c0p_and_c5rf_arms_on_the_toy_world(world, training_items, item_dirs, new_dir) -> None:
    with pytest.raises(ValueError, match="--store"):
        traces.main(_arm_args(world, training_items, item_dirs, new_dir, "C0p", "base"))
    for model, arm, store in (("C0p", "I", "C5"), ("C0p", "base", "C5"), ("C5rf", "I", None)):
        traces.main(_arm_args(world, training_items, item_dirs, new_dir, model, arm) + (["--store", str(world["runs"][store])] if store else []))
        summary = json.loads((world["runs"][model] / f"{traces.OUTPUT_PREFIX}{arm}" / "summary.json").read_text())
        assert summary["store"]["run"] == str(world["runs"][store or model])
        assert set(summary["tests"]["twins"]["summary"]["conditions"]) == {"none", "recall:own"}
        if arm != "base":
            assert summary["training"]["steps"] == 2 and summary["training"]["merged_run_lora"] > 0
    assert json.loads((world["runs"]["C5rf"] / f"{traces.OUTPUT_PREFIX}I" / "summary.json").read_text())["store"]["operator"] != \
        sqx.load_store(world["runs"]["C5"], "own").operator
    found = traces.discover(world["runs"]["C5"].parent)["fake"]
    assert {"I@C0p", "base@C0p", "I-rf"} <= set(found)


def test_decision64_jobs() -> None:
    jobs = traces.decision64_jobs()
    assert len(jobs) == 3 * 2 + 3 * 3 and not {j["name"] for j in jobs} & {j["name"] for j in traces.job_commands()}
    c0p = [j for j in jobs if "-C0p-" in j["name"]]
    assert {j["name"].rsplit("-", 1)[-1] for j in c0p} == {"T", "base"} and all(j["priority"] == 54.44971 for j in c0p)
    for j in c0p:
        seed = j["name"].split("-C0p-s", 1)[1].split("-", 1)[0]
        assert j["command"][j["command"].index("--store") + 1].endswith(f"SmolLM2-360M-full-C5-s{seed}")
    rf = [j for j in jobs if "-C5rf-" in j["name"]]
    assert {j["name"].rsplit("-", 1)[-1] for j in rf} == {"T", "I", "L"} and all(j["priority"] == 54.44972 for j in rf)
    assert all("--store" not in j["command"] for j in rf)
    assert all(("--agent-pairs" in j["command"]) == j["name"].endswith("-I") for j in rf)
    assert sum(j["gpu_h"] for j in jobs) == pytest.approx(4.89, abs=0.01)


def test_job_commands_follow_the_preregistered_design() -> None:
    jobs = traces.job_commands()
    assert len(jobs) == 3 * 5 + 3 and all(j["priority"] == 54.4497 and j["lane"] == "gpu" for j in jobs)
    assert {j["name"].rsplit("-", 1)[-1] for j in jobs if "C5ut" in j["name"]} == {"I"}
    assert all("--overwrite" in j["command"] and "self-query-traces-" in j["name"] for j in jobs)
    critique = crit.job_commands()
    assert len(critique) == 6 + 2 and all(j["priority"] == 54.4498 for j in critique)
    assert sum("critique-loop" in j["name"] for j in critique) == 2


# -- 3c: calibration ------------------------------------------------------------------------------------------------------------

def test_calibrator_is_monotone_and_falls_back_to_the_pooled_model() -> None:
    rng = np.random.default_rng(0)
    n = 600
    cos = rng.uniform(0, 1, n); margin = rng.uniform(0, 0.5, n); degree = rng.integers(2, 10, n)
    y = (rng.uniform(0, 1, n) < cos).astype(float)
    x = np.stack([cos, margin, np.log(degree)], 1)
    relations = np.where(np.arange(n) < 560, 0, 1)                   # relation 1: 40 edges → the pooled model
    cal = crit.Calibrator().fit(x, y, relations)
    assert cal.info["own_models"] == [0] and cal.info["pooled_relations"] == [1]
    grid = np.stack([np.linspace(0, 1, 50), np.full(50, 0.2), np.full(50, np.log(5))], 1)
    p = cal.predict(grid, np.zeros(50, int))
    assert np.all(np.diff(p) >= -1e-12) and p[0] < 0.25 and p[-1] > 0.75 and 0 <= p.min() and p.max() <= 1
    fitted = cal.predict(x, relations)
    assert crit.ece(fitted, y) < 0.06 and crit.auroc(fitted, y) > 0.7 and crit.calibration_metrics(fitted, y)["brier"] < 0.25
    assert crit.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert crit.ece(np.array([0.95] * 20), np.array([0.0] * 20)) == pytest.approx(0.95)
    constant = crit.Calibrator().fit(x[:60], np.ones(60), np.zeros(60, int))
    assert np.allclose(constant.predict(x[:3], np.zeros(3, int)), 1.0)


def test_calibration_on_a_toy_store_and_decoded_edges(world) -> None:
    store = sqx.load_store(world["runs"]["C5"], "own")
    cal, seen = crit.fit_calibrator(store, cap=50)
    assert len(seen["x"]) == len(seen["store"]) > 0 and seen["x"].shape[1] == 3
    frame = _frame("Plurb Gateway")
    found = crit.decoded(store.store, store.store.frame_vector(frame)[None].expand(2, -1), [3, 4], [1, 1])
    line = store.store.decode_role(store.store.frame_vector(frame), 3)
    assert found[0][0][0] == line.fillers[0].atom and found[0][0][2] >= 0         # the slot's read-back and a margin ≥ 0


# -- 3c: evidence, options, null frames ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heldout_files(world, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("wpc7")
    rows = []
    specs = [("Quarn Portal", "owned_by", ["{x} is owned by", "Questions about {x} go to", "Ownership of {x} lies with"],
              [" the Varkt Crew", " the Zash Team"]),
             ("Dribbet Sync", "area", ["{x} belongs to the", "In the org chart, {x} sits in the", "{x} belongs to the"], [" sales area", " finance area"]),
             ("Terb Template", "depends_on", ["{x} depends on", "{x} cannot run without", "{x} has a hard dependency on"],
              [" Grosh Console", " Quill Engine"])]
    for k, (term, relation, templates, choices) in enumerate(specs):
        for j, t in enumerate(templates):
            rows.append({"choices": choices, "concept": term, "group": f"toy-zs-{k}", "id": f"toy-zs-{k}-p{j}", "label": 0, "paraphrase": j,
                         "prompt": t.replace("{x}", term), "relation": relation, "split": "heldout", "surface": term, "task": "zeroshot_property"})
    (root / "zeroshot_property.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with gzip.open(root / "docs.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"text": "Quarn Portal sits with the Varkt Crew. Nothing else.\nDribbet Sync is in the sales area, not the finance area."}) + "\n")
    item_set = crit.wpc7_heldout(root / "zeroshot_property.jsonl", world["ontology"], world["table"])
    evidence_dir = root / "evidence"
    evidence_dir.mkdir()
    und.write_jsonl_gz(evidence_dir / "evidence.jsonl.gz", crit.evidence_sentences(item_set, root / "docs.jsonl.gz"))
    return {"items": root / "zeroshot_property.jsonl", "evidence": evidence_dir, "set": item_set}


def test_heldout_items_and_evidence(world, heldout_files) -> None:
    item_set = heldout_files["set"]
    held = set(world["ontology"]["heldout_entries"])
    assert item_set.kind == "heldout" and len(item_set.prompts) == 3 and all(c["entry"] in held for c in item_set.concepts)
    assert all(len(p.templates) == 3 and all("{x}" in t for t in p.templates) for p in item_set.prompts)
    evidence = crit.load_evidence(heldout_files["evidence"])
    assert evidence["toy-zs-0"]["sentence"] == "Quarn Portal sits with the Varkt Crew." and evidence["toy-zs-0"]["option"] == 0
    assert evidence["toy-zs-1"]["sentence"] is None                  # two options in one sentence: ambiguous, not used
    assert evidence["toy-zs-2"]["sentence"] is None                  # the term is in no sentence


def test_null_frames_make_the_tested_edge_a_distractor(world) -> None:
    import random
    from collections import Counter
    frame = [(0, 1), (3, 10), (3, 11), (4, 12)]
    base = [(0, 2), (3, 10), (3, 13), (4, 14)]
    counts = {3: Counter({10: 5, 11: 3, 13: 1, 15: 2}), 4: Counter({12: 1, 14: 1})}
    for seed in range(20):
        out = crit.null_frame(frame, base, {3: (10, [15, 13])}, counts, random.Random(seed))
        assert [r for r, _ in out] == [r for r, _ in frame]
        assert out[1][1] in {15, 13} and out[2][1] != 10 and out[0][1] == 2 and out[3][1] == 14
    with pytest.raises(ValueError):
        crit.null_frame(frame, [(1, 2)] + base[1:], {}, counts, random.Random(0))
    options = crit.Options(world["ontology"], _lexicon())
    assert options.atoms("owned_by", [" the Varkt Crew", " nobody"]) == [ATOMS.index("term:Varkt Crew"), None]
    assert options.atoms("area", [" sales area"]) == [ATOMS.index("area:sales")]


# -- 3c: the rule loop and its endpoints -----------------------------------------------------------------------------------------

def _item(i: int, *, gold: int, none: list, belief: tuple, null_belief: tuple, null_option: int | None, evidence: int | None = None,
          split: str = "test", kind: str = "new") -> dict:
    b, p = belief
    nb, np_ = null_belief
    scores = {"none": none}
    if evidence is not None:
        scores["evidence"] = [5.0 if k == evidence else 0.0 for k in range(len(none))]
    return {"id": f"i{i}", "set": kind, "split": split, "gold": gold, "options": len(none), "scores": scores,
            "belief": {"real": {"answer": b, "p": p}, "null": {"answer": nb, "p": np_}}, "null_option": null_option,
            "evidence": {"present": evidence is not None, "option": evidence}}


def test_selective_answering_and_k_units() -> None:
    answers, confidence = [0, 1, 0, 1, 0], [0.9, 0.1, 0.5, 0.7, 0.2]
    answered = crit.selective(answers, confidence, coverage=0.8)
    assert answered.tolist() == [True, False, True, True, True]
    flagged = crit.selective(answers, confidence, coverage=0.8, flagged=[True, False, False, False, False])
    assert flagged.tolist() == [False, True, True, True, True]            # a flagged item is abstained first
    items = [{"gold": 0, "null_option": 1}] * 5
    s = crit.scored(items, {"answer": answers, "confidence": confidence, "flagged": [False] * 5})
    assert np.mean(s["k1"]) == pytest.approx(s["accuracy_at"]) and s["accuracy_at"] == pytest.approx(3 / 4)
    assert np.mean(s["k2"]) == pytest.approx(s["adoption_at"]) == pytest.approx(1 / 4)


def test_rule_loop_uses_belief_behaviour_and_evidence() -> None:
    weak = [0.0, 0.1, 0.0]
    items = [_item(i, gold=0, none=weak if i % 2 else [0.0, 2.0, 0.0], belief=(0, 0.95), null_belief=(2, 0.95), null_option=2,
                   evidence=0 if i < 6 else None, kind="heldout" if i < 6 else "new") for i in range(10)]
    ms = crit.methods(items, world="real", theta=0.5, behaviour_cal=None)
    assert ms["loop"]["answer"] == [0] * 10 and ms["no_tool"]["answer"][0] == 1
    null = crit.methods(items, world="null", theta=0.5, behaviour_cal=None)
    assert null["naive"]["answer"] == [2] * 10 and null["loop"]["answer"] == [2] * 10
    assert null["loop"]["flagged"] == [True] * 10                         # belief ≠ behaviour (and ≠ evidence where present)
    assert null["loop_revise"]["answer"][:6] == [0] * 6                   # secondary: the evidence revises the answer
    low = crit.methods(items, world="null", theta=0.99, behaviour_cal=None)
    assert low["loop"]["answer"] == null["no_tool"]["answer"]             # p < θ: the behaviour answers
    dev = [_item(i, gold=0, none=[0.0, 1.0, 0.0], belief=(0, 0.3 + 0.05 * i), null_belief=(2, 0.9), null_option=2, split="dev")
           for i in range(10)]
    theta, table = crit.choose_theta(dev, None)
    assert theta == 0.0 and len(table) == len(crit.THETAS)                # every belief is right: the smallest θ wins the tie
    analysis = crit.analyse_run(items + dev)
    pooled = analysis["pools"]["pooled"]
    assert pooled["items"] == 10 and pooled["real"]["loop"]["accuracy_at"] == 1.0
    assert pooled["null"]["naive"]["adoption_at"] == 1.0 and len(pooled["real"]["loop"]["curve"]) == 17


# -- 3c: the GPU job, the report and the model loop on the toy world -----------------------------------------------------------

def test_critique_job_and_report(world, item_dirs, new_dir, heldout_files, tmp_path) -> None:
    for model, sets in (("C5", "twins,new,heldout"),):          # (C5ut's role-blind recall overflows the fake host's 64 positions)
        crit.main(["evaluate", "--run", str(world["runs"][model]), "--sets", sets, "--twins", str(item_dirs["twins"]), "--new-words", str(new_dir),
                   "--heldout", str(heldout_files["items"]), "--evidence", str(heldout_files["evidence"]), "--alias-table", str(world["alias"]),
                   "--device", "cpu", "--batch-size", "8", "--max-length", "64", "--max-lines", "1", "--twin-limit", "2", "--seen-cap", "50",
                   "--competitors", "off", "--overwrite", "--label", "SMOKE"])        # (the competitors: test_text_competitors_…)
    folder = world["runs"]["C5"] / crit.OUTPUT
    items = und.read_jsonl(folder / "items.jsonl")
    assert {i["set"] for i in items} == {"twins", "new", "heldout"}
    held = [i for i in items if i["set"] == "heldout"]
    assert {"none", "recall:own", "null", "evidence"} == set(held[0]["scores"]) | {"evidence"} and any("evidence" in i["scores"] for i in held)
    assert all(i["null_option"] != i["gold"] for i in items if i["null_option"] is not None)
    twins = [i for i in items if i["set"] == "twins"]
    assert all(i["null_option"] == 1 - i["gold"] for i in twins)          # the partner's frame: the other option
    new = [i for i in items if i["set"] == "new"]
    assert {i["split"] for i in new} == {"dev", "test"}
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["seen"]["n"] > 0 and set(summary["sets"]["heldout"]["calibration_edges"]) == {"real", "null"}
    analysis = crit.analyse(world["runs"]["C5"].parent)
    c5 = analysis["hosts"]["fake"]["models"]["C5"]
    assert any(n.startswith("K1") for n in c5["primary"]) and any(n.startswith("K2") for n in c5["primary"])
    assert "heldout real vs world" in next(iter(c5["calibration"].values()))
    text = crit.render_report(analysis, title="toy", label="SMOKE")
    assert "K1" in text and "ECE" in text


# -- 3c, decision 64 (amendment 16.5): text competitors in the same loop, token accounting, K1b -----------------------------

def _text_item(i: int, *, split: str, kind: str, gold: int = 0, none=(0.0, 1.0, 0.0), recall=(3.0, 0.0, 0.0), symbolic=(3.0, 0.0, 0.0),
               definition=(2.0, 0.0, 0.0), belief=(0, 0.95)) -> dict:
    item = _item(i, gold=gold, none=list(none), belief=belief, null_belief=(2, 0.95), null_option=2, split=split, kind=kind)
    item["scores"].update({"recall:own": list(recall), "symbolic": list(symbolic), "definition": list(definition), "null": [0.0, 0.0, 3.0]})
    item["tokens"] = {"none": 0, "recall:own": 60, "null": 58, "symbolic": 55, "definition": 80}
    return item


def test_text_loop_reads_the_text_in_the_prompt() -> None:
    items = [_text_item(0, split="test", kind="new"), _text_item(1, split="test", kind="new", definition=(0.0, 0.0, 3.0))]
    loop = crit.text_loop(items, "definition", theta=0.5, behaviour_cal=None, context_cal=None)
    assert loop["answer"] == [0, 2] and loop["tokens"] == [80, 80] and loop["flagged"] == [True, True]   # behaviour says 1
    low = crit.text_loop(items, "definition", theta=1.01, behaviour_cal=None, context_cal=None)
    assert low["answer"] == [1, 1]                                                     # below θ: the behaviour answers
    missing = crit.text_loop([_item(5, gold=0, none=[0.0, 1.0], belief=(0, 0.9), null_belief=(1, 0.9), null_option=1)], "definition",
                             theta=0.0, behaviour_cal=None, context_cal=None)
    assert missing["answer"] == [1] and missing["tokens"] == [0]


def test_k1b_noninferiority_with_token_accounting() -> None:
    rng = np.random.default_rng(3)
    per_seed = {}
    for s in (1, 2, 3):
        items = []
        for i in range(240):
            split, kind = ("dev", "new") if i < 60 else ("test", "new" if i < 150 else "heldout")
            wrong_def = rng.uniform() < 0.1                                             # the definition reading errs on 10%
            items.append(_text_item(i, split=split, kind=kind, definition=(0.0, 3.0, 0.0) if wrong_def else (3.0, 0.0, 0.0),
                                    belief=(0, float(rng.uniform(0.8, 1.0)))))
        per_seed[s] = crit.analyse_run(items)
    first = per_seed[1]
    assert set(first["text_loops"]) == set(crit.TEXT_LOOPS) and all(0.0 <= v["theta"] <= 1.0 for v in first["text_loops"].values())
    real, null = first["pools"]["pooled"]["real"], first["pools"]["pooled"]["null"]
    assert {"loop_definition", "loop_symbolic", "loop_recall_text"} <= set(real) and not {"loop_definition", "loop_symbolic"} & set(null)
    assert real["loop"]["tokens"] == 0.0 and real["loop_definition"]["tokens"] == 80.0 and real["recall_context"]["tokens"] == 60.0
    assert first["pools"]["pooled"]["tokens_evidence"] == 0.0
    block = crit.k1b(per_seed, resamples=50)
    assert block["K1b"]["available"] and block["K1b"]["margin"]["kind"] == "noninferiority" and block["K1b"]["margin"]["shown"]
    assert block["K1b"]["model"]["mean"] > 0                                            # the store loop is right where the definition errs
    assert block["tokens"]["loop"]["tokens_per_item"] == 0.0 and block["tokens"]["loop_definition"]["tokens_per_item"] == 80.0
    assert "80 fewer prompt tokens" in block["reading"] and block["reading"].startswith("non-inferior")
    assert {"K1b (new)", "K1b (heldout)", "K1 of loop_definition: loop_definition − no tool (pooled)"} <= set(block["secondaries"])
    assert not crit.k1b({1: crit.analyse_run([_item(i, gold=0, none=[0.0, 1.0], belief=(0, 0.9), null_belief=(1, 0.9), null_option=1,
                                                    split="test") for i in range(5)])})["K1b"]["available"]


def test_text_competitors_in_the_critique_job(world, item_dirs, new_dir, heldout_files, tmp_path, monkeypatch) -> None:
    store = sqx.load_store(world["runs"]["C5"], "own")
    assert crit.competitors_enabled(store) and not crit.competitors_enabled(sqx.load_store(world["runs"]["C5ut"], "own"))
    assert crit.competitors_enabled(store, "off") is False
    # a short definition: the prose writer's text overflows the toy host's 64 positions
    monkeypatch.setattr(sqx.ContextBuilder, "_definition", lambda self, concept: f"{self.concepts[concept]['surface']} is a toy term.")
    output = tmp_path / "runs" / "fake-lora-C5-s1" / crit.OUTPUT
    crit.main(["evaluate", "--run", str(world["runs"]["C5"]), "--output", str(output), "--sets", "twins,new,heldout", "--twins", str(item_dirs["twins"]),
               "--new-words", str(new_dir), "--heldout", str(heldout_files["items"]), "--evidence", str(heldout_files["evidence"]),
               "--alias-table", str(world["alias"]), "--device", "cpu", "--batch-size", "8", "--max-length", "64", "--max-lines", "1",
               "--twin-limit", "2", "--seen-cap", "50", "--overwrite", "--label", "SMOKE"])
    items = und.read_jsonl(output / "items.jsonl")
    for item in items:
        expected = {"symbolic", "definition"} if item["set"] in crit.COMPETITOR_SETS else set()
        assert set(item["scores"]) & {"symbolic", "definition"} == expected
        assert set(item["tokens"]) == set(item["scores"]) and item["tokens"]["none"] == 0
        assert all(item["tokens"][k] > 0 for k in item["tokens"] if k != "none")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sets"]["new"]["competitors"] == ["symbolic", "definition"] and summary["sets"]["twins"]["competitors"] == []
    assert summary["sets"]["heldout"]["tokens_added"]["definition"] > 0
    assert "Prompt tokens each context adds" in (output / "report.md").read_text()
    analysis = crit.analyse(tmp_path / "runs", resamples=20)
    block = analysis["hosts"]["fake"]["models"]["C5"]
    assert "k1b" in block and "loop_definition" in block["k1b"]["tokens"]
    assert "K1b (amendment 16.5)" in crit.render_report(analysis, title="toy", label="SMOKE")


def test_model_loop_on_the_toy_world(world, new_dir, heldout_files) -> None:
    crit.main(["loop", "--run", str(world["runs"]["C5"]), "--new-words", str(new_dir), "--heldout", str(heldout_files["items"]),
               "--evidence", str(heldout_files["evidence"]), "--alias-table", str(world["alias"]), "--device", "cpu", "--batch-size", "4",
               "--max-length", "64", "--max-lines", "1", "--items", "2", "--seen-cap", "50", "--overwrite", "--label", "SMOKE"])
    folder = world["runs"]["C5"] / crit.LOOP_OUTPUT
    rows = und.read_jsonl(folder / "items.jsonl")
    assert rows and all(set(r["worlds"]) == {"real", "null"} for r in rows)
    assert all(r["worlds"][w]["decision"] in {"Keep", "Revise", "Unsure"} for r in rows for w in ("real", "null"))
    summary = json.loads((folder / "summary.json").read_text())
    assert "Decision:" in summary["demonstrations"] and summary["loop"]["items"] == len(rows)
    units = crit.loop_units(rows)
    assert set(next(iter(units.values()))) == {"k1_loop", "k1_no_tool", "k2_loop", "k2_naive"}
    case = crit.critique_case("Question: X depends on ___? Options: A | B\n", "A", "X depends on B. (p = 0.91)", None)
    assert case.endswith("Evidence: none\nDecision:")
