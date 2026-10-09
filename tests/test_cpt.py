"""Continued pretraining (E4.6): trainer options for pretrained hosts and the CPT run plan."""

import json
from pathlib import Path

import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.span_channel import AliasTable
import vsa_embed.training.lm as lm


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("cpt")
    full = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3)],
                                 holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the cat saw hydroxychloroquine in New York and acetaminophen too." for i in range(60)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    build_corpus(texts[:20], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    ontology = {"entry_count": len(full.entry_concepts), "atomic_count": 6, "relation_count": 2,
                "offsets": torch.tensor([0, 2, 4, 6, 8]), "relations": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                "fillers": torch.tensor([0, 1, 1, 2, 3, 4, 4, 5]), "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": [60, 60, 0, 60]}
    torch.save(ontology, root / "ontology.pt")
    return {"root": root}


def tiny_host(config) -> torch.nn.Module:
    """A deterministic stand-in for a pretrained host (same weights at every call)."""
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=1, n_head=2))


def config(root: Path, mode: str, host_mode: str, **extra) -> dict:
    return {"seed": 0, "device": "cpu",
            "model": {"size": "tiny", "seq_len": 32, "pretrained": "fake-host", "host_mode": host_mode, "lora_rank": 2},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 6, "lr": 3e-3, "warmup_tokens": 64,
                      "log_every": 2, **extra.pop("train", {})},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                     "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 64},
            "channel": {"mode": mode, "dimension": 16, "key_dimension": 8, **extra.pop("channel", {})}}


def test_from_scratch_configs_resolve_as_before() -> None:
    resolved = lm.resolve_config({})
    assert not {"eval_only", "host_lr", "save_trainable_only"} & set(resolved["train"])


def test_frozen_host_without_channel_is_evaluated_once_at_every_point(setup, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lm, "build_model", tiny_host)
    with pytest.raises(ValueError, match="eval_only"):
        lm.train(config(setup["root"], "none", "frozen"), tmp_path / "plain")
    cfg = config(setup["root"], "none", "frozen", train={"eval_only": True})
    assert lm.train(cfg, tmp_path / "c0p") == {"steps": 0, "tokens": 0, "eval_only": True}
    rows = [json.loads(line) for line in (tmp_path / "c0p" / "metrics.jsonl").read_text().splitlines()]
    steps = sorted({r["step"] for r in rows})                              # 64 tokens per step
    assert steps == [0, *(t // 64 for t in lm.eval_token_schedule(64, 6 * 64))]
    for stratum in ("all", "after_heldout"):
        losses = {r["loss"] for r in rows if r["stratum"] == stratum}
        assert len(losses) == 1 and all(r["eval_only"] for r in rows)
    final = torch.load(tmp_path / "c0p" / "final.pt", weights_only=False)
    assert final["trainable_only"] and final["model"] == {}
    assert (tmp_path / "c0p" / "manifest.json").is_file()
    # A queued re-run (with --resume) of a finished evaluation does nothing.
    assert lm.train(cfg, tmp_path / "c0p", resume=True)["eval_only"]
    assert len((tmp_path / "c0p" / "metrics.jsonl").read_text().splitlines()) == len(rows)


def test_trainable_only_checkpoints_resume_and_reload(setup, tmp_path: Path, monkeypatch) -> None:
    torch.set_num_threads(1)
    monkeypatch.setattr(lm, "build_model", tiny_host)
    extra = {"train": {"save_trainable_only": True, "host_lr": 1e-3}, "channel": {"context_window": 4}}
    lm.train(config(setup["root"], "compose", "lora", **json.loads(json.dumps(extra))), tmp_path / "straight")
    stop = json.loads(json.dumps(extra)); stop["train"]["stop_after_steps"] = 3
    assert lm.train(config(setup["root"], "compose", "lora", **stop), tmp_path / "resumed").get("interrupted")
    checkpoint = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)
    assert checkpoint["trainable_only"]
    scales = sorted(g.get("lr_scale", 1.0) for g in checkpoint["optimizer"]["param_groups"])
    assert scales[0] == pytest.approx(1e-3 / 3e-3) and scales[-1] == 1.0
    lm.train(config(setup["root"], "compose", "lora", **json.loads(json.dumps(extra))), tmp_path / "resumed", resume=True)
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)
    assert a["trainable_only"] and set(a["model"]) == set(b["model"])
    assert any("lora_a" in k for k in a["model"]) and any(k.startswith("channel.") for k in a["model"])
    assert not any(k.startswith("model.") and "lora" not in k for k in a["model"])     # no frozen host weights
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key])
    model = lm.load_final(tmp_path / "straight" / "final.pt")
    state = model.state_dict()
    for key, value in a["model"].items():
        torch.testing.assert_close(state[key], value)
    host = tiny_host(None).state_dict()
    torch.testing.assert_close(state["model.transformer.wte.weight"], host["transformer.wte.weight"])


def test_probe_loader_reads_trainable_only_checkpoints(setup, tmp_path: Path, monkeypatch) -> None:
    """`channel_probes.load_run` on a LoRA run saved trainable-only (the failure of the cpt-pilot probe jobs:
    "Missing key(s) … model.model.embed_tokens.weight") and on an evaluation-only C0' (an empty state)."""
    from vsa_embed.evaluation import channel_probes as cp
    torch.set_num_threads(1)
    monkeypatch.setattr(lm, "build_model", tiny_host)
    table = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3)],
                                  holdout=[2], include_holdout=True)
    cp.save_alias_table(table, tmp_path / "alias_table.json")
    extra = {"train": {"save_trainable_only": True, "host_lr": 1e-3}, "channel": {"context_window": 4}}
    lm.train(config(setup["root"], "compose", "lora", **extra), tmp_path / "lora")
    adapter = cp.load_run(tmp_path / "lora", device="cpu", alias_table=tmp_path / "alias_table.json")
    expected = lm.load_final(tmp_path / "lora" / "final.pt").state_dict()
    assert adapter.info["trainable_only_checkpoint"] and set(adapter.model.state_dict()) == set(expected)
    for key, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    lm.train(config(setup["root"], "none", "frozen", train={"eval_only": True}), tmp_path / "c0p")
    assert cp.load_run(tmp_path / "c0p", device="cpu", alias_table=tmp_path / "alias_table.json").model.channel is None


def test_full_state_runs_are_unchanged_by_the_new_options(setup, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lm, "build_model", tiny_host)
    lm.train(config(setup["root"], "free", "lora", train={"keep_checkpoint": True}), tmp_path / "full")
    final = torch.load(tmp_path / "full" / "final.pt", weights_only=False)
    assert "trainable_only" not in final and "model.transformer.wte.weight" in final["model"]
    groups = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)["optimizer"]["param_groups"]
    assert len(groups) == 2 and not any("lr_scale" in g for g in groups)


def test_host_corpora_must_match_the_host(setup, tmp_path: Path) -> None:
    corpus = TokenCorpus.open(setup["root"] / "eval")
    small = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=1000, n_positions=64, n_embd=8, n_layer=1, n_head=2))
    with pytest.raises(ValueError, match="embedding rows"):
        lm.check_host_corpora({"model": {"pretrained": "fake-host"}}, small, (corpus,))
    lm.check_host_corpora({"model": {"pretrained": "fake-host"}}, tiny_host(None), (corpus,))
    corpus.manifest["tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the tokenizer"):
        lm.check_host_corpora({"model": {"pretrained": "gpt2"}}, tiny_host(None), (corpus,))
    gpt2 = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    corpus.manifest["tokenizer_sha256"] = lm.tokenizer_fingerprint(gpt2)
    lm.check_host_corpora({"model": {"pretrained": "gpt2"}}, tiny_host(None), (corpus,))


# -- the CPT plan ------------------------------------------------------------------------------

@pytest.fixture()
def plan_root(tmp_path: Path) -> dict:
    counts = {"entry_count": 101500, "atomic_count": 8192, "relation_count": 16}
    torch.save(counts, tmp_path / "counts.pt")
    roots = {"smollm2": tmp_path / "smollm2", "qwen2.5": tmp_path / "qwen2.5"}
    return {"root": tmp_path / "e4", "counts": tmp_path / "counts.pt", "roots": roots}


def _write(stage: str, plan_root: dict, **kwargs) -> list[Path]:
    from vsa_embed.experiments import cpt_plan
    spec = cpt_plan.STAGES[stage]
    hosts = kwargs.pop("hosts", spec["hosts"])
    return cpt_plan.write_stage(stage, hosts=hosts, modes=cpt_plan.stage_modes(stage, hosts, **kwargs.pop("mode_args", {})),
                                names=kwargs.pop("names", spec["conditions"]), seeds=spec["seeds"],
                                total_tokens=spec["total_tokens"], data_roots=plan_root["roots"],
                                counts_ontology=plan_root["counts"], root=plan_root["root"], **kwargs)


def test_cpt_pilot_configs(plan_root) -> None:
    paths = _write("cpt-pilot", plan_root)
    assert sorted(p.stem for p in paths) == ["SmolLM2-135M-lora-C0p-s1", "SmolLM2-135M-lora-C2-s1", "SmolLM2-135M-lora-C5-s1"]
    configs = {p.stem.split("-")[3]: yaml.safe_load(p.read_text()) for p in paths}
    for c in configs.values():
        resolved = lm.resolve_config(c)
        assert resolved["model"]["pretrained"] == "HuggingFaceTB/SmolLM2-135M" and resolved["model"]["host_mode"] == "lora"
        assert resolved["model"]["lora_rank"] == 16 and resolved["train"]["total_tokens"] == 25_000_000
        assert resolved["eval"]["first_tokens"] == 2_500_000 and resolved["eval"]["save_window_losses"] is True
        assert resolved["train"]["micro_batch"] * resolved["train"]["grad_accum"] == 128
        assert resolved["train"]["host_lr"] == 2e-4 and resolved["train"]["save_trainable_only"]
        assert not resolved["train"].get("eval_only")
        assert resolved["data"]["train"] == str(plan_root["roots"]["smollm2"] / "train")
        assert resolved["seed"] == 1
    assert configs["C0p"]["channel"]["mode"] == "none"                    # LoRA only
    assert configs["C2"]["channel"]["mode"] == "free" and configs["C2"]["channel"]["free_dimension"] > 0
    c5 = configs["C5"]["channel"]
    assert (c5["mode"], c5["composition"], c5["operator"], c5["context_window"]) == ("compose", "attentive", "hrr", 8)


def test_cpt_grid_with_and_without_best(plan_root) -> None:
    paths = _write("cpt", plan_root)                                       # `best` pending G3
    assert len(paths) == 3 * 3 * 3 + 3 * 3                                 # frozen hosts + SmolLM2-135M LoRA check
    assert not any("best" in p.stem for p in paths)
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    assert configs["Qwen2.5-0.5B-frozen-C0p-s2"]["train"]["eval_only"] is True
    assert "eval_only" not in configs["SmolLM2-135M-lora-C0p-s2"]["train"]
    micro = {name: c["train"]["micro_batch"] for name, c in configs.items() if name.endswith("C1-s1")}
    assert micro == {"SmolLM2-135M-frozen-C1-s1": 16, "SmolLM2-360M-frozen-C1-s1": 8, "Qwen2.5-0.5B-frozen-C1-s1": 4,
                     "SmolLM2-135M-lora-C1-s1": 8}
    assert all(c["train"]["micro_batch"] * c["train"]["grad_accum"] == 128 and c["train"]["total_tokens"] == 100_000_000
               for c in configs.values())
    assert configs["Qwen2.5-0.5B-frozen-C2-s1"]["data"]["ontology"] == str(plan_root["roots"]["qwen2.5"] / "ontology.pt")
    assert configs["SmolLM2-360M-frozen-C2-s1"]["data"]["train"] == str(plan_root["roots"]["smollm2"] / "train")
    with_best = _write("cpt", plan_root, best="C5")
    assert len(with_best) == 48 and sum("best-C5" in p.stem for p in with_best) == 12
    larger = _write("cpt", plan_root, best="C5", mode_args={"large_host_mode": "lora"})
    assert {p.stem.split("-")[2] for p in larger if p.stem.startswith("Qwen")} == {"lora"}


def test_cpt_queue_jobs(plan_root, tmp_path: Path, monkeypatch) -> None:
    import vsa_embed.jobqueue as jobqueue
    from vsa_embed.experiments import cpt_plan
    monkeypatch.setattr(jobqueue, "DEFAULT_DIR", tmp_path / "jobs")
    paths = _write("cpt-pilot", plan_root)
    assert len(cpt_plan.queue_jobs(paths, "cpt-pilot", 11, root=plan_root["root"])) == 3
    assert cpt_plan.queue_jobs(paths, "cpt-pilot", 11, root=plan_root["root"]) == []       # idempotent
    job = json.loads((tmp_path / "jobs" / "cpt-pilot-SmolLM2-135M-lora-C5-s1.json").read_text())
    assert job["priority"] == 11 and job["command"][2] == "vsa_embed.training.lm"
    assert job["command"][-1] == str(plan_root["root"] / "runs" / "cpt-pilot" / "SmolLM2-135M-lora-C5-s1")


def test_cpt_hosts_match_their_checkpoints() -> None:
    from vsa_embed.experiments import cpt_plan
    for host, settings in cpt_plan.HOSTS.items():
        try:
            # the text config (a multimodal Qwen3.5 checkpoint keeps width and rows there; other configs return themselves)
            hf = transformers.AutoConfig.from_pretrained(settings["pretrained"], local_files_only=True).get_text_config()
        except OSError:
            pytest.skip(f"{settings['pretrained']} not cached")
        except ValueError:          # an architecture this transformers does not know: only hosts with their own environment
            assert settings.get("python"), host
            continue
        assert (hf.hidden_size, hf.vocab_size) == (settings["width"], settings["vocab_size"]), host
