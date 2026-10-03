"""WP-Qwen: the channel scale to the host (open decision 1), host dtype and non-reentrant checkpointing options,
Qwen3 host tables and LoRA targets, Qwen3 track corpora, the E9 plan on Qwen3 hosts and the memory probe."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus, tokenizer_fingerprint
from vsa_embed.integrations.transformers import LORA_TARGETS, ChannelLM, LoRALinear
from vsa_embed.span_channel import HOST_SCALE_FRACTION, AliasTable, SpanChannel
import vsa_embed.training.lm as lm


def _cached(name: str) -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        return True
    except OSError:
        return False


GPT2 = _cached("gpt2")
QWEN3 = _cached("Qwen/Qwen3-0.6B-Base")
ONTOLOGY = {"entry_count": 5, "atomic_count": 6, "relation_count": 2, "offsets": torch.tensor([0, 2, 4, 6, 8, 10]),
            "relations": torch.tensor([0, 1] * 5), "fillers": torch.tensor([0, 1, 1, 2, 3, 4, 4, 5, 0, 5]),
            "heldout_entries": [2], "train_frequency": [60, 60, 0, 60, 3]}


def toy_host(width: int, scale: float, seed: int = 1234) -> torch.nn.Module:
    """A tiny causal LM whose embedding rows are `scale` times GPT-2's initialization."""
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=500, n_positions=64, n_embd=width, n_layer=1, n_head=2))
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(scale)
    return model


def channel_config(mode: str, *, scale: bool, **channel) -> dict:
    spec = {"mode": mode, "dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}
    if mode == "free":
        spec["free_dimension"] = 4
    if scale:
        spec["scale_to_host"] = True
    return lm.resolve_config({"seed": 1, "channel": spec})


def injected_fraction(host: torch.nn.Module, channel: SpanChannel, context=None) -> float:
    """Mean norm of what the channel adds at the injection positions, over the host's mean embedding row norm."""
    wrapped = ChannelLM(host, channel, context=context, host_mode="frozen").eval()
    ids = torch.randint(0, 500, (2, 12), generator=torch.Generator().manual_seed(0))
    # every entry once, so the mean injected norm is the gate times the scaled mean row norm exactly
    spans = {"batch": torch.tensor([0, 0, 1, 1, 1]), "start": torch.tensor([1, 4, 2, 6, 8]), "end": torch.tensor([2, 5, 3, 7, 9]),
             "inject": torch.tensor([2, 5, 3, 7, 9]), "entry": torch.tensor([0, 1, 2, 3, 4]), "confidence": torch.ones(5),
             "length": torch.full((5,), 2)}
    with torch.no_grad():
        added = wrapped.embed(ids, spans) - wrapped.embed(ids, None)
    norms = added[spans["batch"], spans["inject"]].norm(dim=-1)
    return float(norms.mean()) / lm.host_row_norm(host)


# -- 1. channel.scale_to_host ------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["compose", "free"])
def test_scale_to_host_gives_the_same_fraction_of_an_embedding_row_on_every_host(mode) -> None:
    hosts = [toy_host(32, 0.1), toy_host(48, 7.0)]                 # different widths and embedding scales
    assert lm.host_row_norm(hosts[1]) / lm.host_row_norm(hosts[0]) > 50
    fractions, unscaled = [], []
    for host in hosts:
        width = host.get_input_embeddings().weight.shape[1]
        extra = {"composition": "attentive"} if mode == "compose" else {}
        torch.manual_seed(1)
        channel, context = lm.build_channel(channel_config(mode, scale=True, **extra), ONTOLOGY, width, host=host)
        fractions.append(injected_fraction(host, channel, context))
        torch.manual_seed(1)
        plain, plain_context = lm.build_channel(channel_config(mode, scale=False, **extra), ONTOLOGY, width)
        unscaled.append(injected_fraction(host, plain, plain_context))
    # the gate starts at σ(0) = 0.5, so the injection is 0.5 · ρ of a mean embedding row on both hosts
    assert fractions[0] == pytest.approx(fractions[1], rel=1e-4) == pytest.approx(0.5 * HOST_SCALE_FRACTION, rel=1e-4)
    assert unscaled[0] / unscaled[1] > 50                           # without the key the fraction follows the host's scale


def test_scale_to_host_needs_the_host_draws_no_randomness_and_keeps_default_state_keys() -> None:
    host = toy_host(32, 3.0)
    with pytest.raises(ValueError, match="scale_to_host"):
        lm.build_channel(channel_config("compose", scale=True), ONTOLOGY, 32)
    torch.manual_seed(5)
    scaled, _ = lm.build_channel(channel_config("compose", scale=True), ONTOLOGY, 32, host=host)
    after_scaled = torch.rand(1)
    torch.manual_seed(5)
    plain, _ = lm.build_channel(channel_config("compose", scale=False), ONTOLOGY, 32)
    assert torch.equal(torch.rand(1), after_scaled)                 # the measurement consumes no random numbers
    state, reference = scaled.state_dict(), plain.state_dict()
    assert set(state) - set(reference) == {"host_scale"} and "host_scale" not in reference
    for key in reference:
        torch.testing.assert_close(state[key], reference[key])
    record = scaled.host_scale_record
    assert record["scale"] == pytest.approx(HOST_SCALE_FRACTION * lm.host_row_norm(host) / plain.mean_row_norm(), rel=1e-6)
    custom, _ = lm.build_channel(channel_config("compose", scale=True, host_scale_fraction=0.1), ONTOLOGY, 32, host=host)
    assert custom.host_scale_record["fraction"] == 0.1 and float(custom.host_scale) < float(scaled.host_scale)
    rows = scaled.rows({"entry": torch.arange(5)})
    torch.testing.assert_close(rows, plain.rows({"entry": torch.arange(5)}) * scaled.host_scale)


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    if not GPT2:
        pytest.skip("gpt2 tokenizer not cached")
    root = tmp_path_factory.mktemp("qwen-hosts")
    full = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3)],
                                 holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the cat saw hydroxychloroquine in New York and acetaminophen too." for i in range(60)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    build_corpus(texts[:20], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=50_000, batch_texts=8,
                 workers=2)
    ontology = {"entry_count": len(full.entry_concepts), "atomic_count": 6, "relation_count": 2,
                "offsets": torch.tensor([0, 2, 4, 6, 8]), "relations": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                "fillers": torch.tensor([0, 1, 1, 2, 3, 4, 4, 5]), "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": [60, 60, 0, 60]}
    torch.save(ontology, root / "ontology.pt")
    from vsa_embed.evaluation import channel_probes as cp
    cp.save_alias_table(full, root / "alias_table.json")
    return root


def scaled_host(config) -> torch.nn.Module:
    """A deterministic stand-in for a pretrained host with Qwen-like small embedding rows."""
    torch.manual_seed(1234)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=1, n_head=2))
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(0.05)
    return model


def run_config(root: Path, **channel) -> dict:
    return {"seed": 0, "device": "cpu",
            "model": {"size": "tiny", "seq_len": 32, "pretrained": "fake-host", "host_mode": "lora", "lora_rank": 2},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 4, "lr": 3e-3, "warmup_tokens": 64, "log_every": 2,
                      "save_trainable_only": True, "host_lr": 1e-3},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 64},
            "channel": {"mode": "compose", "dimension": 16, "key_dimension": 8, "context_window": 4, "gate_bias": 0.0, **channel}}


def test_trained_scale_is_recorded_and_restored_by_every_rebuild(tiny_corpus, tmp_path, monkeypatch) -> None:
    from vsa_embed.evaluation import channel_probes as cp
    torch.set_num_threads(1)
    monkeypatch.setattr(lm, "build_model", scaled_host)
    lm.train(run_config(tiny_corpus, scale_to_host=True), tmp_path / "scaled")
    final = torch.load(tmp_path / "scaled" / "final.pt", weights_only=False)
    trained = final["model"]["channel.host_scale"]
    record = json.loads((tmp_path / "scaled" / "channel_scale.json").read_text())
    manifest = json.loads((tmp_path / "scaled" / "manifest.json").read_text())
    assert record["scale"] == pytest.approx(float(trained)) and manifest["channel_host_scale"]["scale"] == pytest.approx(float(trained))
    assert record["host_row_norm"] == pytest.approx(lm.host_row_norm(scaled_host(None)))
    # rebuilds (another RNG state, so another fresh channel) restore the trained value from the state
    torch.manual_seed(99)
    assert float(lm.load_final(tmp_path / "scaled" / "final.pt").channel.host_scale) == float(trained)
    adapter = cp.load_run(tmp_path / "scaled", device="cpu", alias_table=tiny_corpus / "alias_table.json")
    assert float(adapter.model.channel.host_scale) == float(trained) and adapter.info["channel_host_scale"] == float(trained)
    # a state without the buffer does not fit a scaled channel (and a default run's state has no such key)
    lm.train(run_config(tiny_corpus), tmp_path / "plain")
    plain = torch.load(tmp_path / "plain" / "final.pt", weights_only=False)
    assert "channel.host_scale" not in plain["model"] and not (tmp_path / "plain" / "channel_scale.json").exists()
    assert "channel_host_scale" not in json.loads((tmp_path / "plain" / "manifest.json").read_text())
    model = lm.load_final(tmp_path / "scaled" / "final.pt")
    with pytest.raises(RuntimeError, match="host_scale"):
        lm.load_model_state(model, plain["model"], trainable_only=True)


# -- 2. host dtype, LoRA targets and checkpointing on a (tiny) Qwen3 ---------------------------------------------

@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory) -> Path:
    """A randomly initialised two-layer Qwen3 saved locally (the `qwen3` architecture of the Qwen3 base hosts)."""
    path = tmp_path_factory.mktemp("tiny-qwen3")
    config = transformers.Qwen3Config(vocab_size=600, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, head_dim=16, max_position_embeddings=128, tie_word_embeddings=True)
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def test_qwen3_lora_targets_host_dtype_and_non_reentrant_checkpointing(tiny_qwen3) -> None:
    config = lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_mode": "lora", "lora_rank": 4}, "device": "cpu"})
    base = lm.build_model(config)
    assert next(base.parameters()).dtype == torch.float32 and lm.lora_adapter_dtype(config) is None
    wrapped = lm.wrap_host(config, base, None, None)
    found = {name.rsplit(".", 1)[-1] for name, m in wrapped.named_modules() if isinstance(m, LoRALinear)}
    assert found == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= set(LORA_TARGETS)
    assert sum(isinstance(m, LoRALinear) for m in wrapped.modules()) == 7 * 2
    half = lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_mode": "lora", "lora_rank": 4, "host_dtype": "bfloat16"}})
    host = lm.build_model(half)
    adapters = [p for n, p in lm.wrap_host(half, host, None, None).named_parameters() if p.requires_grad]
    assert next(host.parameters()).dtype == torch.bfloat16 and {p.dtype for p in adapters} == {torch.float32}
    with pytest.raises(ValueError, match="host_dtype"):
        lm.build_model(lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_dtype": "float8"}}))
    # C0′ with gradient checkpointing: the embeddings carry no gradient; the non-reentrant form still trains the adapters
    ckpt = lm.resolve_config({"model": {"pretrained": str(tiny_qwen3), "host_mode": "lora", "lora_rank": 4,
                                        "gradient_checkpointing": True, "checkpoint_use_reentrant": False}})
    torch.manual_seed(0)
    model = lm.wrap_host(ckpt, lm.build_model(ckpt), None, None).train()
    assert model.model.is_gradient_checkpointing
    ids = torch.randint(0, 600, (2, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_b.normal_(std=0.01)                         # so that every adapter matrix gets a gradient
    model(ids, labels=ids)["loss"].backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora" in n]
    assert grads and all(g is not None and float(g.abs().sum()) > 0 for g in grads)


@pytest.mark.skipif(not QWEN3, reason="Qwen3 tokenizer not cached")
def test_qwen3_hosts_in_the_tables_match_their_checkpoints() -> None:
    from vsa_embed.experiments import cpt_plan, host_corpus
    for host in ("Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"):
        settings = cpt_plan.HOSTS[host]
        try:
            hf = transformers.AutoConfig.from_pretrained(settings["pretrained"], local_files_only=True)
        except OSError:
            pytest.skip(f"{settings['pretrained']} not cached")
        assert (hf.model_type, hf.hidden_size, hf.vocab_size, hf.tie_word_embeddings) == ("qwen3", settings["width"], settings["vocab_size"], True)
        assert settings["corpus"] == "qwen3" and settings["pretrained"] in host_corpus.HOSTS["qwen3"]["models"]
    assert cpt_plan.HOSTS["Qwen3-4B-Base"]["model"] == {"gradient_checkpointing": True, "checkpoint_use_reentrant": False}
    # one tokenizer for the three sizes; Qwen2.5's plus four added tokens, so not interchangeable with Qwen2.5 corpora
    prints = {tokenizer_fingerprint(transformers.AutoTokenizer.from_pretrained(name, local_files_only=True))
              for name in host_corpus.HOSTS["qwen3"]["models"] if _cached(name)}
    assert len(prints) == 1
    if _cached("Qwen/Qwen2.5-0.5B"):
        qwen25 = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
        qwen3 = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
        assert tokenizer_fingerprint(qwen25) not in prints and len(qwen3) - len(qwen25) == 4
        text = "The Brainsh Ledger is owned by the Data Platform team; café naïve."
        assert qwen3(text)["input_ids"] == qwen25(text)["input_ids"]


@pytest.mark.skipif(not (QWEN3 and GPT2), reason="Qwen3 or gpt2 tokenizer not cached")
def test_track_corpus_with_the_qwen3_tokenizer_keeps_the_holdout(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vsa_embed.experiments.track_corpus import run
    rng = random.Random(0)
    words = "the river city school garden market history music science water light energy family story café".split()
    pq.write_table(pa.table({"text": [" ".join(rng.choice(words) for _ in range(120)) + "." for _ in range(300)]}),
                   tmp_path / "general.parquet")

    def config(tokenizer: str, name: str) -> dict:
        return {"experiment": f"t5-tiny-{name}", "track": "t5", "seed": 3, "tokenizer": tokenizer, "workers": 1,
                "items_dir": str(tmp_path / name / "items"),
                "paths": {"data_root": str(tmp_path / name / "data"), "general_shards": [str(tmp_path / "general.parquet")]},
                "glossary": {"terms": 200, "zero_shot": 12, "heldout_fraction": 0.1, "zipf": 1.5, "eval_uniform_focus": 0.5,
                             "acronym_fraction": 0.3, "train_chars": 150_000, "eval_chars": 60_000, "forbidden_general_docs": 40},
                "ontology": {"max_atomics": 8192, "max_degree": 16},
                "data": {"eval_tokens": 100_000, "contamination_docs": 40, "presample_tokens": 0, "holdout_fraction": 0.1,
                         "holdout_min_count": 5, "general_skip_docs": 0, "general_eval_docs": 20, "train_domain_tokens": 30_000,
                         "train_total_tokens": 45_000, "min_subtokens": 2, "cardinality_docs": 30},
                "items": {"train_terms_per_frequency_bin": 5, "relations_per_term": 2}, "cardinality_tokenizers": [tokenizer]}

    gpt2 = run(config("gpt2", "gpt2"), tmp_path / "gpt2" / "run")
    qwen_config = config("Qwen/Qwen3-0.6B-Base", "qwen3")
    qwen_config["data"]["expected_holdout_sha256"] = gpt2["holdout"]["sha256"]
    qwen = run(qwen_config, tmp_path / "qwen3" / "run")
    a, b = (torch.load(tmp_path / n / "data" / "ontology.pt", weights_only=False) for n in ("gpt2", "qwen3"))
    assert a["alias_table_sha256"] == b["alias_table_sha256"] and a["heldout_entries"] == b["heldout_entries"]
    assert qwen["holdout"]["sha256"] == gpt2["holdout"]["sha256"]
    fingerprint = tokenizer_fingerprint(transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True))
    for part in ("train", "eval", "eval-general"):
        corpus = TokenCorpus.open(tmp_path / "qwen3" / "data" / part)
        assert corpus.manifest["dtype"] == "uint32" and corpus.manifest["tokenizer_sha256"] == fingerprint
        assert int(corpus.tokens.max()) < 151669
    assert TokenCorpus.open(tmp_path / "qwen3" / "data" / "eval").manifest["normalization"] == "NFC"
    assert "normalization" not in TokenCorpus.open(tmp_path / "gpt2" / "data" / "eval").manifest   # GPT-2 builds as before


# -- 3. the E9 plan on Qwen3 hosts -------------------------------------------------------------------------------

@pytest.fixture()
def plan_root(tmp_path: Path) -> dict:
    torch.save({"entry_count": 4200, "atomic_count": 3493, "relation_count": 23}, tmp_path / "counts.pt")
    return {"root": tmp_path / "e9", "counts": tmp_path / "counts.pt", "data": tmp_path / "qwen3"}


def test_e9_plan_writes_the_qwen3_block(plan_root, tmp_path) -> None:
    from vsa_embed.experiments import e9_plan, e9_tracks
    hosts = ["Qwen3-1.7B-Base", "Qwen3-0.6B-Base", "Qwen3-4B-Base"]
    paths = e9_plan.write_stage("t5-qwen3", hosts=hosts, models=list(e9_plan.MODELS), seeds=[1], track="t5",
                                counts_ontology=plan_root["counts"], root=plan_root["root"])
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    assert len(paths) == 3 * 4 and "Qwen3-1.7B-Base-lora-C5-s1" in configs and "Qwen3-4B-Base-frozen-P0-s1" in configs
    qwen3_root = e9_tracks.QWEN3_ROOTS["t5"]
    for stem, c in configs.items():
        resolved = lm.resolve_config(c)
        assert c["e9_family"] == "qwen3" and c["e9_track"] == "t5" and resolved["data"]["train"] == str(qwen3_root / "train")
        assert resolved["train"]["micro_batch"] * resolved["train"]["grad_accum"] == 64 and resolved["train"]["total_tokens"] == 50_000_000
        trained = "-P0-" not in stem
        assert resolved["model"]["host_mode"] == ("lora" if trained else "frozen") and resolved["model"]["lora_rank"] == 64
        assert bool(resolved["channel"].get("scale_to_host")) == (stem.endswith(("C2-s1", "C5-s1")))
        if trained:
            assert resolved["train"]["host_lr"] == 2e-4 and resolved["train"]["lr"] == 1e-3 and resolved["train"]["save_trainable_only"]
        if stem.endswith(("C2-s1", "C5-s1")):
            assert resolved["channel"]["gate_bias"] == 0.0
    big = configs["Qwen3-4B-Base-lora-C5-s1"]["model"]
    assert big["gradient_checkpointing"] is True and big["checkpoint_use_reentrant"] is False
    assert configs["Qwen3-0.6B-Base-lora-C5-s1"]["model"]["gradient_checkpointing"] is False
    assert configs["Qwen3-0.6B-Base-lora-C2-s1"]["channel"]["free_dimension"] == 222       # matched to C5 at width 1024
    # the memory probe's recommendation, then an explicit override
    memory = {"Qwen3-1.7B-Base": {"lora": {"micro_batch": 4, "gradient_checkpointing": True, "host_dtype": "bfloat16",
                                           "eval_batch": 6, "tokens_per_s": 3000.0}}}
    probed = e9_plan.write_stage("t5-qwen3", hosts=["Qwen3-1.7B-Base"], models=["P0", "C5"], seeds=[1], track="t5",
                                 counts_ontology=plan_root["counts"], root=plan_root["root"], memory=memory)
    c5 = yaml.safe_load(probed[1].read_text())
    p0 = yaml.safe_load(probed[0].read_text())
    assert (c5["train"]["micro_batch"], c5["train"]["grad_accum"], c5["eval"]["batch"]) == (4, 16, 6)
    assert c5["model"]["host_dtype"] == "bfloat16" == p0["model"]["host_dtype"] and c5["model"]["checkpoint_use_reentrant"] is False
    forced = e9_plan.write_stage("t5-qwen3", hosts=["Qwen3-1.7B-Base"], models=["C5"], seeds=[1], track="t5", memory=memory,
                                 micro_batch=2, counts_ontology=plan_root["counts"], root=plan_root["root"])
    assert yaml.safe_load(forced[0].read_text())["train"]["micro_batch"] == 2
    assert e9_plan.host_plan("Qwen3-1.7B-Base", "lora", memory=memory)["tokens_per_s"] == 3000.0
    with pytest.raises(ValueError, match="one E9 stage per host tokenizer family"):
        e9_plan.write_stage("x", hosts=["SmolLM2-360M", "Qwen3-0.6B-Base"], models=["C5"], seeds=[1], counts_ontology=plan_root["counts"],
                            root=plan_root["root"])
    with pytest.raises(ValueError, match="host-mode lora"):
        e9_plan.write_stage("x", hosts=["Qwen3-0.6B-Base"], models=["C5"], seeds=[1], mode="train", counts_ontology=plan_root["counts"],
                            root=plan_root["root"])
    off = e9_plan.write_stage("t5-qwen3-noscale", hosts=["Qwen3-0.6B-Base"], models=["C5"], seeds=[1], channel_scale="off",
                              counts_ontology=plan_root["counts"], root=plan_root["root"])
    assert "scale_to_host" not in yaml.safe_load(off[0].read_text())["channel"]
    smol = e9_plan.write_stage("t5-scaled", hosts=["SmolLM2-135M"], models=["C5"], seeds=[1], channel_scale="on",
                               counts_ontology=plan_root["counts"], root=plan_root["root"], data_root=tmp_path / "smol")
    smol_config = yaml.safe_load(smol[0].read_text())
    assert smol_config["channel"]["scale_to_host"] is True and "e9_family" not in smol_config
    # GPU estimates: the 6N model before the probe, measured tokens/s after it
    assert 1.0 < e9_plan.estimate_hours("Qwen3-0.6B-Base", 50_000_000) < e9_plan.estimate_hours("Qwen3-1.7B-Base", 50_000_000)
    assert e9_plan.estimate_hours("Qwen3-4B-Base", 50_000_000, checkpointing=True) > e9_plan.estimate_hours("Qwen3-4B-Base", 50_000_000)
    assert e9_plan.estimate_hours("Qwen3-1.7B-Base", 36_000_000, tokens_per_s=1000.0) == pytest.approx(10.0)


def test_e9_plan_queues_the_qwen3_block_with_its_items_batches_and_priority(plan_root, tmp_path) -> None:
    from vsa_embed.experiments import e9_plan
    paths = e9_plan.write_stage("t5-qwen3", hosts=["Qwen3-1.7B-Base", "Qwen3-0.6B-Base"], models=list(e9_plan.MODELS), seeds=[1],
                                track="t5", counts_ontology=plan_root["counts"], root=plan_root["root"])
    queue = tmp_path / "jobs"
    queued = e9_plan.queue_jobs(paths, "t5-qwen3", track="t5", root=plan_root["root"], queue_dir=queue, alias_table=tmp_path / "t5.json")
    assert len(queued) == 8 + 8 * 6 + 3
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    assert jobs["t5-qwen3-Qwen3-1.7B-Base-lora-C5-s1"]["priority"] == 26
    probes = jobs["t5-qwen3-Qwen3-1.7B-Base-lora-C5-s1-probes-int4"]
    assert probes["priority"] == 27 and probes["command"][probes["command"].index("--probes") + 1] == e9_plan.INT4_PROBES_NO_WSD
    for suffix, batch in (("edit", "8"), ("zeroshot-int4", "8"), ("probes", "8")):
        command = jobs[f"t5-qwen3-Qwen3-1.7B-Base-lora-C5-s1-{suffix}"]["command"]
        assert command[command.index("--batch-size") + 1] == batch
    small = jobs["t5-qwen3-Qwen3-0.6B-Base-lora-C2-s1-edit"]["command"]
    assert small[small.index("--batch-size") + 1] == "16"
    assert small[small.index("--new-items") + 1].endswith("new-words-t5-qwen3-v1")
    assert small[small.index("--edit-items") + 1].endswith("edits-t5-qwen3-v1")
    general = jobs["t5-qwen3-quant-general-s1"]["command"]
    assert general[general.index("--eval-corpus") + 1].endswith("t5-glossary/v1-qwen3/eval-general") and jobs["t5-qwen3-quant-s1"]["priority"] == 28
    assert jobs["t5-qwen3-report-s1"]["priority"] == 29
    # the committed dimension-3 items of the Qwen3 T5 block exist
    for folder in e9_plan.dimension3_items("t5", "qwen3"):
        assert (Path(__file__).resolve().parents[1] / folder / "manifest.json").exists()


def test_track_specs_per_host_family() -> None:
    from vsa_embed.experiments import e9_tracks
    smol, qwen = e9_tracks.track_spec("t5"), e9_tracks.track_spec("t5", "qwen3")
    assert smol.family == "smollm2" and qwen.family == "qwen3" and qwen.data_root == e9_tracks.QWEN3_ROOTS["t5"]
    assert qwen.alias_table_path == smol.alias_table_path and qwen.items_dir == smol.items_dir and qwen.holdout_names == smol.holdout_names
    assert qwen.general_corpus == e9_tracks.QWEN3_ROOTS["t5"] / "eval-general"
    assert e9_tracks.track_spec("t1", "qwen3").eval_corpus == e9_tracks.QWEN3_ROOTS["t1"] / "eval-pubmed"
    with pytest.raises(ValueError, match="no gpt2 corpora"):
        e9_tracks.track_spec("t5", "gpt2")


# -- 4. the memory probe -----------------------------------------------------------------------------------------

def test_memory_probe_recommendation_prefers_the_fastest_fitting_fp32_setting() -> None:
    from vsa_embed.experiments import e9_memory
    def row(host, dtype, ckpt, micro, peak, tps, kind="train", ok=True):
        base = {"host": host, "host_dtype": dtype, "checkpointing": ckpt, "kind": kind, "ok": ok, "peak_gib": peak,
                "peak_reserved_gib": None if peak is None else peak + 0.5}
        return {**base, "micro_batch": micro, "tokens_per_s": tps} if kind == "train" else {**base, "eval_batch": micro}
    rows = [row("Qwen3-1.7B-Base", "float32", False, 1, 15.0, 2000.0), row("Qwen3-1.7B-Base", "float32", False, 2, 21.0, 2300.0),
            row("Qwen3-1.7B-Base", "float32", False, 4, 30.0, None, ok=False),
            row("Qwen3-1.7B-Base", "float32", True, 4, 17.0, 2100.0), row("Qwen3-1.7B-Base", "float32", True, 8, 23.5, 2400.0),
            row("Qwen3-1.7B-Base", "float32", False, 4, 18.0, None, kind="eval"), row("Qwen3-1.7B-Base", "float32", False, 8, 22.4, None, kind="eval"),
            row("Qwen3-4B-Base", "float32", False, 1, 25.0, None, ok=False), row("Qwen3-4B-Base", "float32", True, 1, 23.0, 900.0),
            row("Qwen3-4B-Base", "bfloat16", True, 4, 14.0, 1200.0), row("Qwen3-4B-Base", "bfloat16", True, 2, 12.0, None, kind="eval")]
    best = e9_memory.recommend(rows, budget_gib=22.5)
    small = best["Qwen3-1.7B-Base"]["lora"]                         # fp32 micro-batch 2 beats checkpointing at 4 (8 is over budget)
    assert (small["micro_batch"], small["gradient_checkpointing"], small["host_dtype"], small["eval_batch"]) == (2, False, "float32", 4)
    four = best["Qwen3-4B-Base"]["lora"]                            # fp32 does not fit (23.5 reserved > 22.5): the bf16 host
    assert (four["host_dtype"], four["micro_batch"], four["gradient_checkpointing"], four["eval_batch"]) == ("bfloat16", 4, True, 2)
    assert four["gpu_hours_per_run"] == pytest.approx(50_000_000 / 1200.0 / 3600)
    assert "Qwen3-4B-Base" not in e9_memory.recommend(rows, budget_gib=10.0)


@pytest.mark.skipif(not QWEN3, reason="Qwen3 tokenizer not cached")
def test_memory_probe_config_is_the_e9_c5_run() -> None:
    from vsa_embed.experiments import e9_memory, e9_tracks
    if not (e9_tracks.QWEN3_ROOTS["t5"] / "ontology.pt").exists():
        pytest.skip("T5 Qwen3 corpus not built")
    config = e9_memory.probe_config("Qwen3-4B-Base", 2, host_dtype="bfloat16", checkpointing=True)
    assert config["model"]["pretrained"] == "Qwen/Qwen3-4B-Base" and config["model"]["lora_rank"] == 64
    assert config["model"]["host_dtype"] == "bfloat16" and config["model"]["checkpoint_use_reentrant"] is False
    assert config["channel"]["mode"] == "compose" and config["channel"]["scale_to_host"] and config["channel"]["context_window"] == 8
    assert (config["train"]["micro_batch"], config["train"]["grad_accum"], config["model"]["seq_len"]) == (2, 32, 1024)
    plain = e9_memory.probe_config("Qwen3-0.6B-Base", 1)
    assert not plain["model"]["gradient_checkpointing"] and "host_dtype" not in plain["model"]
