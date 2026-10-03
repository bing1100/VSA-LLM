"""WP-Qwen35: hybrid linear-attention hosts (Qwen3.5) — opt-in LoRA targets and loss chunks, the CUDA-only dispatch of the
fast kernels, the Qwen3.5 host tables and interpreter, the T5 Qwen3.5 corpus and items, the E9 plan and the memory probe.

Runs in both environments: the pinned one (transformers 4.54, no `qwen3_5`; the Qwen3.5 model tests skip) and the
Qwen3.5 one (transformers 5.18 with flash-linear-attention and causal-conv1d, CPU here: the dispatch must route CPU
tensors to the reference implementation)."""

import functools
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

import vsa_embed.training.lm as lm
from vsa_embed.integrations import linear_attention as la
from vsa_embed.integrations.transformers import (LINEAR_ATTENTION_TARGETS, LORA_TARGETS, ChannelLM, LoRALinear,
                                                 chunked_causal_lm_loss, lora_targets)

HAS_QWEN35 = hasattr(transformers, "Qwen3_5TextConfig")
REPO = Path(__file__).resolve().parents[1]


def _hub_config(repo: str) -> dict | None:
    """The cached `config.json` of a hub repo, read as JSON (the pinned transformers cannot parse `qwen3_5`)."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(repo, "config.json")
    return json.loads(Path(path).read_text()) if isinstance(path, str) else None


# -- 1. opt-in LoRA targets and loss chunks (both environments) ---------------------------------------------------

def test_lora_target_sets_resolve_and_default_is_unchanged() -> None:
    assert lora_targets(None) == LORA_TARGETS == lora_targets("default")
    assert lora_targets("linear_attention") == LORA_TARGETS + LINEAR_ATTENTION_TARGETS
    assert set(LINEAR_ATTENTION_TARGETS) == {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    assert lora_targets(["q_proj", "out_proj"]) == ("q_proj", "out_proj")
    for bad in ("everything", [], [""]):
        with pytest.raises(ValueError):
            lora_targets(bad)


@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen3-35")
    config = transformers.Qwen3Config(vocab_size=300, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def test_wrap_host_reads_the_opt_in_keys_and_leaves_default_configs_alone(tiny_qwen3) -> None:
    plain = lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_mode": "lora", "lora_rank": 2}, "device": "cpu"})
    model = lm.wrap_host(plain, lm.build_model(plain), None, None)
    assert model.loss_chunk == 2048 and sum(isinstance(m, LoRALinear) for m in model.modules()) == 7 * 2
    assert lm.host_records(plain, model) == {}                    # no new manifest keys for a default run
    chosen = lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_mode": "lora", "lora_rank": 2,
                                          "lora_targets": ["q_proj", "v_proj"], "loss_chunk": 5}, "device": "cpu"})
    narrow = lm.wrap_host(chosen, lm.build_model(chosen), None, None)
    found = {n.rsplit(".", 1)[-1] for n, m in narrow.named_modules() if isinstance(m, LoRALinear)}
    assert found == {"q_proj", "v_proj"} and narrow.loss_chunk == 5
    records = lm.host_records(chosen, narrow)
    assert set(records) == {"lora_coverage"} and records["lora_coverage"]["adapters"] == 4
    ids = torch.randint(0, 300, (2, 13), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(narrow(ids, labels=ids)["loss"], lm.wrap_host(chosen, lm.build_model(chosen), None, None)(ids, labels=ids)["loss"])


def test_chunked_loss_does_not_depend_on_the_chunk() -> None:
    torch.manual_seed(0)
    hidden, weight = torch.randn(2, 9, 6), torch.randn(40, 6)
    labels = torch.randint(0, 40, (2, 9)); labels[0, 3] = -100
    reference = chunked_causal_lm_loss(hidden, weight, labels, chunk=10_000, reduction="none")
    for chunk in (1, 3, 7):
        torch.testing.assert_close(chunked_causal_lm_loss(hidden, weight, labels, chunk=chunk, reduction="none"), reference)


def test_dtype_argument_follows_the_transformers_major() -> None:
    major = int(transformers.__version__.split(".")[0])
    assert lm.dtype_kwargs(torch.bfloat16) == ({"dtype": torch.bfloat16} if major >= 5 else {"torch_dtype": torch.bfloat16})


# -- 2. the CUDA-only dispatch of the fast kernels (a stand-in modeling module) ------------------------------------

def _hub_style(implementation, torch_function):
    """The shape of transformers' `use_kernel_func_from_hub_with_fallback` wrapper (closure + functools.wraps)."""
    @functools.wraps(torch_function)
    def wrapped(*args, **kwargs):
        return implementation(*args, **kwargs)
    return wrapped


def test_dispatch_routes_cpu_tensors_to_the_reference_and_counts_calls(monkeypatch) -> None:
    calls = []
    fast = lambda x, **kw: calls.append("fast") or x + 1
    def reference(x, **kw):
        calls.append("reference")
        return x + 1
    module = types.ModuleType("fake_modeling_qwen3_5")
    module.torch_chunk_gated_delta_rule = _hub_style(fast, reference)
    module.causal_conv1d_fn = _hub_style(reference, reference)       # bound to its reference (package absent)
    monkeypatch.setitem(sys.modules, "fake_modeling_qwen3_5", module)
    monkeypatch.setattr(la, "MODELING_MODULES", ("fake_modeling_qwen3_5",))
    status = la.install_device_dispatch()["fake_modeling_qwen3_5"]
    assert status["functions"]["torch_chunk_gated_delta_rule"]["bound"] == "fast" and status["functions"]["torch_chunk_gated_delta_rule"]["dispatch"]
    assert status["functions"]["causal_conv1d_fn"]["bound"] == "reference" and not status["functions"]["causal_conv1d_fn"]["dispatch"]
    assert not status["fast_path_bound"]
    dispatcher = module.torch_chunk_gated_delta_rule
    la.install_device_dispatch()
    assert module.torch_chunk_gated_delta_rule is dispatcher           # idempotent
    la.kernel_calls(reset=True)
    assert float(dispatcher(torch.zeros(1))) == 1.0 and calls == ["reference"]
    class Cuda:                                                        # a stand-in for a CUDA tensor
        is_cuda = True
    monkeypatch.setattr(la, "_first_tensor", lambda args, kwargs: Cuda())
    dispatcher(torch.zeros(1))
    with la.reference_only():
        dispatcher(torch.zeros(1))
    assert calls == ["reference", "fast", "reference"]
    assert la.kernel_calls(reset=True) == {"torch_chunk_gated_delta_rule": {"fast": 1, "reference": 2}}


# -- 3. a tiny Qwen3.5 text model (transformers ≥ 5 only) ----------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_qwen35(tmp_path_factory) -> Path:
    if not HAS_QWEN35:
        pytest.skip("transformers without qwen3_5 (the pinned environment)")
    from vsa_embed.experiments.linear_attention_smoke import tiny_host
    return tiny_host(tmp_path_factory.mktemp("tiny-qwen35") / "host", overrides={"vocab_size": 512, "hidden_size": 64,
                                                                                    "intermediate_size": 96, "head_dim": 16,
                                                                                    "linear_key_head_dim": 16, "linear_value_head_dim": 16})


def test_qwen35_host_loads_without_cache_with_dispatch_and_every_layer_adapted(tiny_qwen35) -> None:
    torch.set_num_threads(2)
    config = lm.resolve_config({"model": {"pretrained": str(tiny_qwen35), "host_mode": "lora", "lora_rank": 4,
                                          "lora_targets": "linear_attention", "loss_chunk": 7}, "device": "cpu"})
    base = lm.build_model(config)
    assert la.is_linear_attention_host(base) and base.config.use_cache is False
    assert "modeling_qwen3_5" in base.linear_attention_kernels
    model = lm.wrap_host(config, base, None, None)
    coverage = la.lora_layer_coverage(model.model)
    assert coverage["uncovered_mixers"] == [] and coverage["by_type"]["linear_attention"] == {"layers": 3, "adapters": 24, "per_layer": [8]}
    assert coverage["by_type"]["full_attention"] == {"layers": 1, "adapters": 7, "per_layer": [7]}
    # the default targets leave every Gated DeltaNet token mixer without adapters (the reason for the opt-in key)
    default = lm.resolve_config({"model": {"pretrained": str(tiny_qwen35), "host_mode": "lora", "lora_rank": 4}, "device": "cpu"})
    assert la.lora_layer_coverage(lm.wrap_host(default, lm.build_model(default), None, None).model)["uncovered_mixers"] == [0, 1, 2]
    ids = torch.randint(0, 512, (2, 21), generator=torch.Generator().manual_seed(3))
    la.kernel_calls(reset=True)
    loss = model(ids, labels=ids)["loss"]
    with torch.no_grad():
        torch.testing.assert_close(loss.detach(), base(input_ids=ids, labels=ids).loss, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_b.normal_(std=0.01)
    model(ids, labels=ids)["loss"].backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora" in n]
    assert len(grads) == 2 * 31 and all(g is not None and float(g.abs().sum()) > 0 for g in grads)
    calls = la.kernel_calls(reset=True)
    assert calls and all(set(kinds) == {"reference"} for kinds in calls.values())     # CPU: the reference path only
    records = lm.host_records(config, model)
    assert records["lora_coverage"]["adapters"] == 31 and "linear_attention_kernels" in records


def test_qwen35_trains_saves_and_reloads_through_the_trainer(tiny_qwen35, tmp_path) -> None:
    """A two-step LoRA C5 run on a tiny Qwen3.5 and a tiny corpus (GPT-2 ids < 512 are not needed: random ids)."""
    from vsa_embed.data.corpus import build_corpus
    from vsa_embed.span_channel import AliasTable
    torch.set_num_threads(2)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True) if _cached("gpt2") else None
    if tokenizer is None:
        pytest.skip("gpt2 tokenizer not cached")
    full = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2)], holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the hydroxychloroquine in New York and acetaminophen." for i in range(40)]
    for part, table in (("train", full.without_holdout()), ("eval", full)):
        build_corpus(texts, tmp_path / part, tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=20_000, batch_texts=8, workers=1)
    ontology = {"entry_count": len(full.entry_concepts), "atomic_count": 4, "relation_count": 2, "offsets": torch.tensor([0, 2, 4, 6]),
                "relations": torch.tensor([0, 1, 0, 1, 0, 1]), "fillers": torch.tensor([0, 1, 1, 2, 3, 0]),
                "heldout_entries": sorted(full.heldout_entries()), "train_frequency": [40, 40, 0]}
    torch.save(ontology, tmp_path / "ontology.pt")
    # gpt2 ids exceed the tiny host's 512 rows: remap the stored token streams into its range
    import numpy as np
    for part in ("train", "eval"):
        tokens = np.memmap(tmp_path / part / "tokens.bin", dtype=np.uint16, mode="r+")
        tokens[:] = tokens % 512
        tokens.flush()
    config = {"seed": 0, "device": "cpu",
              "model": {"size": "pretrained", "seq_len": 32, "pretrained": str(tiny_qwen35), "host_mode": "lora", "lora_rank": 2,
                        "lora_targets": "linear_attention", "loss_chunk": 16},
              "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 2, "lr": 3e-3, "warmup_tokens": 64, "log_every": 1,
                        "save_trainable_only": True, "host_lr": 1e-3},
              "data": {"train": str(tmp_path / "train"), "eval": str(tmp_path / "eval"), "ontology": str(tmp_path / "ontology.pt"),
                       "min_subtokens": 1},
              "eval": {"windows": 2, "batch": 2, "first_tokens": 64},
              "channel": {"mode": "compose", "dimension": 16, "key_dimension": 8, "context_window": 4, "gate_bias": 0.0, "scale_to_host": True}}
    lm.train(config, tmp_path / "run")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["lora_coverage"]["uncovered_mixers"] == [] and manifest["lora_coverage"]["adapters"] == 31
    kernels = manifest["linear_attention_kernels"]
    assert kernels["python"] == sys.executable and set(kernels["status"]) == {"modeling_qwen3_5"}
    assert kernels["calls"] and all(set(kinds) == {"reference"} for kinds in kernels["calls"].values())
    final = torch.load(tmp_path / "run" / "final.pt", weights_only=False)
    assert final["trainable_only"] and any("in_proj_qkv.lora_a" in k for k in final["model"])
    reloaded = lm.load_final(tmp_path / "run" / "final.pt")
    torch.testing.assert_close(reloaded.state_dict()["channel.host_scale"], final["model"]["channel.host_scale"])


def _cached(name: str) -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        return True
    except OSError:
        return False


# -- 4. host tables, tracks, plan and probe (both environments) ----------------------------------------------------

def test_qwen35_hosts_in_the_tables_match_their_checkpoints() -> None:
    from vsa_embed.experiments import cpt_plan, host_corpus
    for host in ("Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base"):
        settings = cpt_plan.HOSTS[host]
        assert settings["corpus"] == "qwen3_5" and settings["python"] == cpt_plan.QWEN35_PYTHON == cpt_plan.host_python(host)
        assert settings["model"] == {"lora_targets": "linear_attention", "loss_chunk": 1024}
        assert settings["pretrained"] in host_corpus.HOSTS["qwen3_5"]["models"]
        hub = _hub_config(settings["pretrained"])
        if hub is None:
            continue
        text = hub["text_config"]
        assert (hub["model_type"], text["hidden_size"], text["vocab_size"]) == ("qwen3_5", settings["width"], settings["vocab_size"])
        assert text["layer_types"].count("linear_attention") == 18 and len(text["layer_types"]) == 24
        assert hub.get("tie_word_embeddings", text.get("tie_word_embeddings")) is True
    for host in ("SmolLM2-360M", "Qwen3-1.7B-Base"):
        assert "python" not in cpt_plan.HOSTS[host] and cpt_plan.host_python(host) == cpt_plan.pinned_python()


def test_track_specs_for_the_qwen35_family() -> None:
    from vsa_embed.experiments import e9_tracks
    smol, qwen35 = e9_tracks.track_spec("t5"), e9_tracks.track_spec("t5", "qwen3_5")
    assert qwen35.family == "qwen3_5" and qwen35.data_root == e9_tracks.QWEN35_ROOTS["t5"] != e9_tracks.QWEN3_ROOTS["t5"]
    assert qwen35.alias_table_path == smol.alias_table_path and qwen35.items_dir == smol.items_dir
    assert e9_tracks.track_spec("t1", "qwen3_5").eval_corpus == e9_tracks.QWEN35_ROOTS["t1"] / "eval-pubmed"
    assert e9_tracks.FAMILY_TOKENIZERS["qwen3_5"] == "Qwen/Qwen3.5-0.8B-Base"


@pytest.fixture()
def plan_root(tmp_path: Path) -> dict:
    torch.save({"entry_count": 4200, "atomic_count": 3493, "relation_count": 23}, tmp_path / "counts.pt")
    return {"root": tmp_path / "e9", "counts": tmp_path / "counts.pt"}


def test_e9_plan_writes_and_queues_the_qwen35_block_with_the_venv_interpreter(plan_root, tmp_path) -> None:
    from vsa_embed.experiments import cpt_plan, e9_plan, e9_tracks
    hosts = list(e9_plan.QWEN35_HOSTS)
    assert hosts == ["Qwen3.5-2B-Base", "Qwen3.5-0.8B-Base"]
    paths = e9_plan.write_stage("t5-qwen35", hosts=hosts, models=list(e9_plan.MODELS), seeds=[1], track="t5", lora_rank=64,
                                counts_ontology=plan_root["counts"], root=plan_root["root"])
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    assert len(paths) == 8 and "Qwen3.5-2B-Base-lora-C5-s1" in configs and "Qwen3.5-0.8B-Base-frozen-P0-s1" in configs
    for stem, c in configs.items():
        resolved = lm.resolve_config(c)
        assert c["e9_family"] == "qwen3_5" and resolved["data"]["train"] == str(e9_tracks.QWEN35_ROOTS["t5"] / "train")
        assert resolved["model"]["lora_targets"] == "linear_attention" and resolved["model"]["loss_chunk"] == 1024
        assert resolved["model"]["lora_rank"] == 64 and resolved["train"]["micro_batch"] * resolved["train"]["grad_accum"] == 64
        assert bool(resolved["channel"].get("scale_to_host")) == stem.endswith(("C2-s1", "C5-s1"))
        if "-P0-" not in stem:
            assert resolved["model"]["host_mode"] == "lora" and resolved["train"]["host_lr"] == 2e-4
    assert configs["Qwen3.5-0.8B-Base-lora-C2-s1"]["channel"]["free_dimension"] == 222        # matched at width 1024, as Qwen3-0.6B
    queue = tmp_path / "jobs"
    queued = e9_plan.queue_jobs(paths, "t5-qwen35", track="t5", root=plan_root["root"], queue_dir=queue, alias_table=tmp_path / "t5.json")
    assert len(queued) == 8 + 8 * 6 + 3
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    assert {j["command"][0] for j in jobs.values()} == {cpt_plan.QWEN35_PYTHON}            # every job: the venv interpreter
    assert jobs["t5-qwen35-Qwen3.5-2B-Base-lora-C5-s1"]["priority"] == 52
    assert jobs["t5-qwen35-Qwen3.5-2B-Base-lora-C5-s1-probes-int4"]["priority"] == 53
    assert jobs["t5-qwen35-quant-s1"]["priority"] == 54 and jobs["t5-qwen35-quant-general-s1"]["priority"] == 54
    assert jobs["t5-qwen35-report-s1"]["priority"] == 55
    edit = jobs["t5-qwen35-Qwen3.5-0.8B-Base-lora-C5-s1-edit"]["command"]
    assert edit[edit.index("--new-items") + 1].endswith("new-words-t5-qwen3_5-v1") and edit[edit.index("--batch-size") + 1] == "16"
    general = jobs["t5-qwen35-quant-general-s1"]["command"]
    assert general[general.index("--eval-corpus") + 1].endswith("t5-glossary/v1-qwen35/eval-general")
    # the committed dimension-3 items of the block exist
    for folder in e9_plan.dimension3_items("t5", "qwen3_5"):
        assert (REPO / folder / "manifest.json").exists()
    # other families keep this interpreter; one family (and interpreter) per stage
    smol = e9_plan.write_stage("t5", hosts=["SmolLM2-135M"], models=["C5"], seeds=[1], counts_ontology=plan_root["counts"],
                               root=plan_root["root"], data_root=tmp_path / "smol")
    e9_plan.queue_jobs(smol, "t5", track="t5", root=plan_root["root"], queue_dir=tmp_path / "jobs-smol", alias_table=tmp_path / "t5.json")
    assert {json.loads(p.read_text())["command"][0] for p in (tmp_path / "jobs-smol").glob("*.json")} == {cpt_plan.pinned_python()}
    assert cpt_plan.pinned_python() != cpt_plan.QWEN35_PYTHON          # also when planned from the Qwen3.5 environment
    with pytest.raises(ValueError, match="one E9 stage per host tokenizer family"):
        e9_plan.write_stage("x", hosts=["Qwen3-0.6B-Base", "Qwen3.5-0.8B-Base"], models=["C5"], seeds=[1],
                            counts_ontology=plan_root["counts"], root=plan_root["root"])
    with pytest.raises(ValueError, match="one interpreter"):
        e9_plan.stage_python(["Qwen3-0.6B-Base", "Qwen3.5-0.8B-Base"])
    assert e9_plan.FAMILIES["qwen3_5"]["suffix"] == "-qwen35" and e9_plan.FAMILIES["qwen3_5"]["memory_report"] == e9_plan.QWEN35_MEMORY_REPORT


def test_e9_plan_main_defaults_for_qwen35(monkeypatch, tmp_path) -> None:
    from vsa_embed.experiments import e9_plan
    seen = {}
    monkeypatch.setattr(e9_plan, "write_stage", lambda stage, **kw: seen.update(stage=stage, **kw) or [])
    monkeypatch.setattr(e9_plan, "load_memory_report", lambda path: seen.update(memory_report=path) or None)
    e9_plan.main(["--track", "t5", "--hosts", "Qwen3.5-2B-Base", "Qwen3.5-0.8B-Base", "--host-mode", "lora", "--lora-rank", "64"])
    assert seen["stage"] == "t5-qwen35" and seen["memory_report"] == e9_plan.QWEN35_MEMORY_REPORT and seen["lora_rank"] == 64
    e9_plan.main(["--track", "t5", "--hosts", "Qwen3-0.6B-Base"])
    assert seen["stage"] == "t5-qwen3" and seen["memory_report"] == e9_plan.MEMORY_REPORT


def test_memory_probe_covers_the_qwen35_hosts(tmp_path, monkeypatch) -> None:
    from vsa_embed.experiments import e9_memory, e9_plan, e9_tracks
    if (e9_tracks.QWEN35_ROOTS["t5"] / "ontology.pt").exists():
        config = e9_memory.probe_config("Qwen3.5-2B-Base", 2, checkpointing=True)
        assert config["data"]["train"] == str(e9_tracks.QWEN35_ROOTS["t5"] / "train")
        assert config["model"]["lora_targets"] == "linear_attention" and config["model"]["loss_chunk"] == 1024
        assert config["model"]["gradient_checkpointing"] and config["model"]["checkpoint_use_reentrant"] is False
        assert (config["train"]["micro_batch"], config["train"]["grad_accum"]) == (2, 32)
    seen = {}

    def fake_measure(host, *, host_dtype, checkpointing, device, steps, data_root, log, micro_batches=e9_memory.MICRO_BATCHES):
        seen[(host, checkpointing)] = tuple(micro_batches)
        rows = [{"host": host, "host_dtype": host_dtype, "checkpointing": checkpointing, "kind": "train", "micro_batch": m, "ok": True,
                 "tokens_per_s": 1000.0 * m * (0.7 if checkpointing else 1.0), "peak_gib": None, "peak_reserved_gib": None,
                 "kernel_calls": {"torch_chunk_gated_delta_rule": {"fast": 4}}} for m in micro_batches]
        if not checkpointing:
            rows.append({**rows[0], "kind": "train-reference", "tokens_per_s": 100.0,
                         "kernel_calls": {"torch_chunk_gated_delta_rule": {"reference": 4}}})
        rows.append({"host": host, "host_dtype": host_dtype, "checkpointing": checkpointing, "kind": "eval", "eval_batch": 8, "ok": True,
                     "peak_gib": None, "peak_reserved_gib": None})
        return rows

    monkeypatch.setattr(e9_memory, "measure_setting", fake_measure)
    out = tmp_path / "memory"
    e9_memory.main(["--output", str(out), "--hosts", "Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base", "--device", "cpu", "--budget-gib", "22"])
    assert seen == {(h, c): (1, 2, 4) for h in ("Qwen3.5-0.8B-Base", "Qwen3.5-2B-Base") for c in (False, True)}
    memory = e9_plan.load_memory_report(out / "recommendation.json")
    assert memory["Qwen3.5-2B-Base"]["lora"]["micro_batch"] == 4 and not memory["Qwen3.5-2B-Base"]["lora"]["gradient_checkpointing"]
    report = (out / "report.md").read_text()
    assert report.startswith("# E9 Qwen3.5 memory") and "## Linear-attention kernels" in report and "speed-up 10.00×" in report
    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert resolved["family"] == "qwen3_5" and resolved["experiment"] == "e9-memory-qwen3_5"
    assert resolved["host_micro_batches"] == {"Qwen3.5-0.8B-Base": [1, 2, 4], "Qwen3.5-2B-Base": [1, 2, 4]}
    plan = e9_plan.host_plan("Qwen3.5-2B-Base", "lora", memory=memory)
    assert plan["micro"]["lora"] == 4 and plan["model"]["lora_targets"] == "linear_attention"
    with pytest.raises(SystemExit):
        e9_memory.main(["--output", str(tmp_path / "mixed"), "--hosts", "Qwen3-0.6B-Base", "Qwen3.5-0.8B-Base", "--device", "cpu"])


# -- 5. the T5 Qwen3.5 build and its items (where built) -----------------------------------------------------------

def test_t5_qwen35_build_keeps_the_ontology_holdout_and_items() -> None:
    from vsa_embed.experiments import e9_tracks
    root = e9_tracks.QWEN35_ROOTS["t5"]
    if not (root / "ontology.pt").exists():
        pytest.skip("T5 Qwen3.5 corpus not built")
    smol = torch.load(e9_tracks.track_spec("t5").ontology, weights_only=False)
    qwen35 = torch.load(root / "ontology.pt", weights_only=False)
    for key in ("alias_table_sha256", "holdout_sha256", "synthetic_sha256", "heldout_entries", "entry_concepts", "concept_names"):
        assert qwen35[key] == smol[key], key
    for key in ("offsets", "relations", "fillers"):
        assert torch.equal(torch.as_tensor(qwen35[key]), torch.as_tensor(smol[key]))
    assert qwen35["tokenizer"] == "Qwen/Qwen3.5-0.8B-Base"
    run = REPO / "experiments/t5-enterprise-glossary/runs/v1-qwen35"
    check = json.loads((run / "crosscheck.json").read_text())
    assert check["qwen3_5"]["docs_sha256"] == check["smollm2"]["docs_sha256"]
    for track_items in ("new-words", "edits"):
        a = (REPO / f"experiments/e9-retrofit/items/{track_items}-t5-qwen3_5-v1/items.jsonl").read_bytes()
        assert a == (REPO / f"experiments/e9-retrofit/items/{track_items}-t5-smollm2-v1/items.jsonl").read_bytes()
