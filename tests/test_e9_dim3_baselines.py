"""E9 dimension-3 baselines (WP-PQ2): weight editors, in-context frames and IKE, frame transplant, channel-off audit,
intra-entity locality, row sources, the plan's job types and the report section."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed import knowledge_editing as ke  # noqa: E402
from vsa_embed.compose import FrameSchedule  # noqa: E402
from vsa_embed.evaluation.probes import ModelAdapter  # noqa: E402
from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_dim3_baselines as dim3  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e9_plan, e9_report  # noqa: E402

from test_e9 import _ok, _plan, items, plan_root, world  # noqa: E402,F401  (module fixtures re-used)


def _tiny_llama() -> ModelAdapter:
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(config).eval()
    return ModelAdapter(model, tokenizer, torch.device("cpu"), batch_size=8, max_length=64)


TEXTS = ["The cat sat on the mat while the dog ran to the river bank.", "A claw hammer and a glass jar stood on the shelf.",
         "Paracetamol is a common analgesic sold in every pharmacy.", "The motortruck passed the bike near the old bridge."] * 12
REQUEST = ke.EditRequest("{x} is owned by", "Plorktorb Ledger", " the Zeimrirb Guild", "e1")


def test_default_layers_scale_with_depth() -> None:
    assert ke.default_layers("rome", 32) == (5,) and ke.default_layers("memit", 32) == (4, 5, 6, 7, 8)
    assert ke.default_layers("rome", 30) == (5,) and ke.default_layers("alphaedit", 28) == tuple(range(4, 8))
    assert ke.default_layers("rome", 4) == (1,) and ke.default_layers("memit", 2) == (0,)
    with pytest.raises(ValueError):
        ke.EditorSettings(method="mend")
    with pytest.raises(ValueError):
        ke.EditorSettings(context_prefixes=("Note. {}",))


@pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")
def test_rome_writes_the_target_value_at_the_subject_key_and_restores() -> None:
    adapter = _tiny_llama()
    host = ke.EditableHost(adapter)
    settings = ke.EditorSettings(method="rome", v_steps=6, layers=(1,))
    moments = ke.second_moments(host, TEXTS, [1])
    assert moments[1].shape == (64, 64) and torch.allclose(moments[1], moments[1].T)
    saved = host.snapshot([1])
    target, init, _ = ke.optimize_delta(host, REQUEST, settings, layer=1, at_residual=False)       # deterministic on the CPU
    plain = ke._build("{}", REQUEST.prompt, REQUEST.subject)
    key, before = ke._site_io(host, 1, [plain])
    assert torch.allclose(before[0], init, atol=1e-5)
    record = ke.apply_rome(host, REQUEST, settings, moments)
    _, after = ke._site_io(host, 1, [plain])
    assert torch.allclose(after[0], target, atol=1e-4 * max(1.0, float(target.norm())))   # W' k* = v* (rank-one property)
    delta = host.weight(1)[0] - saved[1]
    assert torch.linalg.matrix_rank(delta.double(), tol=1e-6 * float(delta.abs().max())) == 1
    assert record["layer"] == 1 and record["delta_norm"] <= settings.clamp_norm_factor * record["init_norm"] + 1e-5
    host.restore(saved)
    assert torch.equal(host.weight(1)[0], saved[1])


@pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")
def test_memit_moves_the_residual_and_alphaedit_stays_in_the_null_space() -> None:
    adapter = _tiny_llama()
    host = ke.EditableHost(adapter)
    requests = [REQUEST, ke.EditRequest("{x} belongs to", "Brammulk Registry", " the Briltbei Squad", "e2")]
    layers = (1, 2)
    moments = ke.second_moments(host, TEXTS, layers)
    saved = host.snapshot(layers)
    memit = ke.EditorSettings(method="memit", layers=layers, v_steps=4, mom2_update_weight=1.0)
    info = ke.apply_memit(host, requests, memit, moments)
    assert info["edits"] == 2 and [p["layer"] for p in info["per_layer"]] == list(layers)
    # The z error left for the second layer is smaller than for the first (the first layer's update did part of the work).
    assert info["per_layer"][1]["z_error"] < info["per_layer"][0]["z_error"]
    host.restore(saved)
    alpha = ke.EditorSettings(method="alphaedit", layers=layers, v_steps=4, nullspace_relative=0.05)
    info = ke.apply_memit(host, requests, alpha, moments)
    for layer, record in zip(layers, info["per_layer"]):
        update = (host.weight(layer)[0] - saved[layer]).double()        # (d, d_in): LLaMA Linear orientation
        values, vectors = torch.linalg.eigh(moments[layer].double())
        preserved = vectors[:, values >= 0.05 * float(values.max())]
        assert 0 < record["null_dim"] < 64
        assert float((update @ preserved).norm()) <= 1e-5 * max(1.0, float(update.norm())) + 1e-7
    host.restore(saved)


def test_transplant_schedule_moves_whole_frames() -> None:
    schedule = FrameSchedule(torch.tensor([0, 2, 5, 6]), torch.tensor([0, 1, 0, 1, 2, 0]), torch.tensor([10, 11, 12, 13, 14, 15]))
    moved = dim3.transplant_schedule(schedule, {0: 1})
    assert moved.offsets.tolist() == [0, 3, 6, 7]
    assert moved.fillers.tolist() == [12, 13, 14, 12, 13, 14, 15] and moved.relations.tolist()[:3] == [0, 1, 2]
    assert dim3.transplant_schedule(schedule, {}).fillers.tolist() == schedule.fillers.tolist()


def test_verbalize_and_contexts() -> None:
    text = dim3.verbalize({"hypernym": ["container"], "owned_by": ["the Pod"]}, "plork")
    assert text == "Definition of plork: is a kind of container; owned by the Pod." and dim3.verbalize({"area": ["x"]}) == "area x."
    item = {"id": "a", "templates": ["The {x} is", "A {x} was"], "concept": "c"}
    out = dim3.with_context([item, {**item, "id": "b"}], {"a": "Note {braces}. "})
    assert out[0]["templates"][0].format(x="cat") == "Note {braces}. The cat is" and out[1] is not out[0] and out[1]["templates"] == item["templates"]
    assert dim3.formatted(None, "owned_by", "the Crew", like=" the Guild.", like_text="the Guild") == " the Crew."


@pytest.fixture(scope="module")
def evaluations(world, items) -> dict:
    """The baselines on the tiny C5 and C0′ runs (CPU)."""
    torch.set_num_threads(2)
    out = {}
    for name, methods in (("C5", ["context", "ike", "transplant", "channel_off", "intra", "rows"]),
                          ("C0p", ["context", "ike", "rome", "memit", "alphaedit", "intra", "transplant"])):
        run = common.open_run(world["runs"][name], device="cpu", batch_size=8, max_length=64)     # the tiny host's positions
        settings = {m: ke.EditorSettings(method=m, v_steps=3, nullspace_relative=0.05 if m == "alphaedit" else None)
                    for m in ke.METHODS}
        out[name] = (run, dim3.evaluate(run, new_items=items["new"], edit_items=items["edits"], methods=methods, resamples=50,
                                        fit_entries=20, cov_tokens=2000, edit_limit=4, new_limit=4, settings=settings, ike_demos=1,
                                        log=lambda _: None))
    return out


def test_channel_model_audits(evaluations) -> None:
    run, out = evaluations["C5"]
    document = out["document"]
    assert set(document["methods"]) == {"context", "ike", "transplant", "channel_off", "intra", "rows"} and not document["skipped"]
    context = document["new_words"]["context"]
    assert set(context["sources"]) == {"own", "context_frame", "context_other", "context_frame_none"}
    assert context["context_tokens"] > 3 and set(context["raw_correct"]) >= {"own", "context_frame"}
    assert {"own", "surface_mean"} <= set(document["new_words"]["rows"]["sources"])
    edits = document["edits"]
    off = edits["channel_off"]
    assert off["revert_max_abs"] == 0.0                         # the frame edit cannot act without the channel
    assert edits["ontology"]["edit_applicable"] and edits["intra_items"] > 0
    assert edits["ontology"]["intra"]["all"]["n"] > 0 and edits["ike"]["context_tokens"] > 5
    transplant = edits["transplant"]
    assert transplant["items"] >= 1 and transplant["subsets"]["all"]["all_relations"]["n"] >= 1
    # Every prediction row is attributed; the model is unchanged afterwards (schedule restored).
    rows = out["predictions"]
    assert {r["method"] for r in rows} >= {"context", "rows", "ontology", "ike", "channel_off", "transplant"}
    assert run.adapter.spans_fn is not None


def test_weight_editors_on_the_host(evaluations) -> None:
    run, out = evaluations["C0p"]
    document = out["document"]
    assert document["skipped"] == {"transplant": "needs composed rows (C5)"}
    edits = document["edits"]
    for method in ("rome", "memit", "alphaedit"):
        block = edits[method]
        assert block["subsets"]["all"]["efficacy"]["n"] >= 1 and "intra" in block
        assert block["records"]["after"]["seconds"] >= 0 and block["records"]["settings"]["method"] == method
    assert edits["second_moments"]["texts"] > 0
    rome = {r["condition"] for r in out["predictions"] if r["method"] == "rome"}
    assert rome == {"before", "after", "control"} and "ontology" not in edits and "channel_off" not in edits


def test_weights_restored_after_editing(world, items, evaluations) -> None:
    run, out = evaluations["C0p"]
    _, concepts, edit_items = edit.load_item_dir(items["edits"], edit.SCHEMA_EDITS)
    keep = {c["concept"] for c in concepts if c.get("role") == "edited"}
    recorded = {r["id"]: r["d_mean"] for r in out["predictions"] if r["method"] == "rome" and r["condition"] == "before"}
    fresh = edit.score_pairs(run.adapter, [i for i in edit_items if i["id"] in recorded and i["edit"] in keep],
                             {c["concept"]: c["surface"] for c in concepts})
    assert fresh and all(abs(fresh[i]["d_mean"] - recorded[i]) < 1e-6 for i in fresh)


def test_cli_run_folder_and_report_section(world, items, tmp_path) -> None:
    folder = world["runs"]["C5"] / dim3.FOLDER
    dim3.main(["evaluate", "--run", str(world["runs"]["C5"]), "--new-items", str(items["new"]), "--edit-items", str(items["edits"]),
               "--methods", "context,ike,channel_off,intra", "--output", str(folder), "--device", "cpu", "--resamples", "50",
               "--edit-limit", "4", "--new-limit", "3", "--batch-size", "8", "--ike-demos", "1", "--max-length", "64"])
    assert {"report.md", "summary.json", "manifest.json", "resolved_config.yaml", "predictions.jsonl"} <= {p.name for p in folder.iterdir()}
    assert "Channel-off audit" in (folder / "report.md").read_text()
    c0 = world["runs"]["C0p"] / dim3.FOLDER
    dim3.main(["evaluate", "--run", str(world["runs"]["C0p"]), "--edit-items", str(items["edits"]), "--methods", "ike,rome,intra",
               "--output", str(c0), "--device", "cpu", "--resamples", "50", "--edit-limit", "4", "--cov-tokens", "2000", "--ike-demos", "1",
               "--max-length", "64"])
    stage = dim3.stage_summary({"C5": {1: world["runs"]["C5"]}, "C0'": {1: world["runs"]["C0p"]}}, resamples=50)
    assert set(stage["models"]) == {"C5", "C0'"}
    references = {r["reference"] for r in stage["comparisons"]["all"]}
    assert {"C0' rome", "C0' ike", "C5 ike"} <= references
    out = tmp_path / "report"
    summary = e9_report.write_report([world["root"] / "runs"], out, resamples=100, figures=False, dim3_baselines=True)
    group = next(iter(summary["groups"].values()))
    assert "dimension3_baselines" in group and set(group["dimension3_baselines"]["models"]) == {"C5", "C0'"}
    report = (out / "report.md").read_text()
    assert "Dimension-3 baselines" in report and "Edits: ontology edit vs IKE" in report
    plain = e9_report.write_report([world["root"] / "runs"], tmp_path / "plain", resamples=100, figures=False)
    assert "dimension3_baselines" not in next(iter(plain["groups"].values()))
    assert "Dimension-3 baselines" not in (tmp_path / "plain" / "report.md").read_text()


def test_plan_queues_evaluation_only_dim3_jobs(plan_root, tmp_path) -> None:
    _plan(plan_root, hosts=["SmolLM2-135M"], seeds=[1, 2])
    queue = tmp_path / "jobs"
    queued, planned = e9_plan.queue_dim3_baselines("main", track="wordnet", root=plan_root["root"], queue_dir=queue)
    names = {name for name, _, _ in planned}
    assert len(planned) == 1 + 3 * 2 + 1 and "main-report-dim3" in names and set(queued) == names
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    c0 = jobs["main-SmolLM2-135M-full-C0p-s2-dim3-baselines"]
    command = c0["command"]
    assert c0["priority"] == e9_plan.DIM3_PRIORITY and command[2] == "vsa_embed.experiments.e9_dim3_baselines"
    assert command[command.index("--methods") + 1] == "context,ike,rome,memit,alphaedit,intra"
    assert command[command.index("--output") + 1].endswith("SmolLM2-135M-full-C0p-s2/dim3-baselines") and c0["resume_args"] == ["--overwrite"]
    c5 = jobs["main-SmolLM2-135M-full-C5-s1-dim3-baselines"]["command"]
    assert c5[c5.index("--methods") + 1] == "context,ike,transplant,channel_off,intra,rows"
    report = jobs["main-report-dim3"]
    assert report["priority"] == e9_plan.DIM3_PRIORITY + 1 and "--dim3-baselines" in report["command"]
    assert not any("training.lm" in " ".join(j["command"]) for j in jobs.values())          # evaluation only
    again, _ = e9_plan.queue_dim3_baselines("main", track="wordnet", root=plan_root["root"], queue_dir=queue)
    assert again == []                                                                     # idempotent
    _, only = e9_plan.queue_dim3_baselines("main", track="wordnet", root=plan_root["root"], queue=False, seeds=[1], models=["C5"],
                                           weights_on_candidate=True)
    assert [n for n, _, _ in only] == ["main-SmolLM2-135M-full-C5-s1-dim3-baselines", "main-report-dim3"]
    assert only[0][1][only[0][1].index("--methods") + 1].endswith(",rome,memit,alphaedit")
    hours = e9_plan.dim3_estimate_hours("SmolLM2-360M", "C0p")
    assert hours > e9_plan.dim3_estimate_hours("SmolLM2-360M", "C2") > 0
    assert e9_plan.dim3_estimate_hours("SmolLM2-135M", "C0p") == pytest.approx(hours * e9_plan.DIM3_SCALE_FLOOR)
    assert e9_plan.dim3_estimate_hours("Qwen3-1.7B-Base", "C0p") > 2 * hours
    with pytest.raises(FileNotFoundError):
        e9_plan.queue_dim3_baselines("absent", root=plan_root["root"], queue=False)
