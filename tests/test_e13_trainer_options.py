"""E13 trainer options (opt-in; absent keys change nothing): `train.init_mode: continue`, `model.host_quantization`,
`channel.entry_rows` (`SpanChannel.set_entry_rows`) and `eval.points`."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.data.corpus import build_corpus
from vsa_embed.span_channel import AliasTable, SpanChannel
from vsa_embed.training.lm import (eval_token_schedule, evaluation_schedule, load_continuation_state, load_final, resolve_config,
                                   train)

torch.set_num_threads(1)


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e13trainer")
    full = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3)],
                                 holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the cat saw hydroxychloroquine in New York and acetaminophen too." for i in range(60)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=50_000, batch_texts=8, workers=2)
    build_corpus(texts[:20], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=50_000, batch_texts=8, workers=2)
    ontology = {"entry_count": len(full.entry_concepts), "atomic_count": 6, "relation_count": 2,
                "offsets": torch.tensor([0, 2, 4, 6, 8]), "relations": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                "fillers": torch.tensor([0, 1, 1, 2, 3, 4, 4, 5]), "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": [60, 60, 0, 60]}
    torch.save(ontology, root / "ontology.pt")
    grown = {**ontology, "atomic_count": 7, "relation_count": 3, "offsets": torch.tensor([0, 3, 5, 7, 9]),
             "relations": torch.tensor([0, 1, 2, 0, 1, 0, 1, 0, 1]), "fillers": torch.tensor([0, 1, 6, 1, 2, 3, 4, 4, 5]),
             "heldout_entries": []}
    torch.save(grown, root / "grown.pt")
    host = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, n_positions=64))
    host.save_pretrained(root / "tiny-host")
    return {"root": root}


def config(root: Path, **extra) -> dict:
    model = {"size": "tiny", "seq_len": 32, **extra.pop("model", {})}
    return {"seed": 0, "device": "cpu", "model": model,
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 4, "lr": 3e-3, "warmup_tokens": 64,
                      "log_every": 2, **extra.pop("train", {})},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(extra.pop("ontology", root / "ontology.pt")),
                     "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 64, **extra.pop("eval", {})},
            "channel": {"mode": "compose", "dimension": 16, "key_dimension": 8, **extra.pop("channel", {})}}


def _eval_rows(run: Path) -> list[dict]:
    return [r for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines()) if r["type"] == "eval"]


# -- defaults unchanged ------------------------------------------------------------------------------------------------


def test_configs_without_the_new_keys_resolve_and_schedule_as_before(setup, tmp_path: Path) -> None:
    resolved = resolve_config(config(setup["root"]))
    for section, key in (("train", "init_mode"), ("train", "init_merge_lora"), ("model", "host_quantization"),
                         ("channel", "entry_rows"), ("eval", "points")):
        assert key not in resolved[section]
    assert evaluation_schedule(resolved, 256) == eval_token_schedule(64, 256) == [64, 128, 256]
    assert evaluation_schedule({"eval": {"points": [300, 10, 64, 10_000]}}, 256) == [10, 64, 256]
    train(config(setup["root"]), tmp_path / "plain")
    manifest = json.loads((tmp_path / "plain" / "manifest.json").read_text())
    assert "host_quantization" not in manifest
    assert sorted({r["tokens"] for r in _eval_rows(tmp_path / "plain")}) == [0, 64, 128, 256]


def test_a_channel_without_entry_rows_keeps_its_state_and_rows() -> None:
    torch.manual_seed(0)
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 2)]])
    channel = SpanChannel(FrameComposer(schedule, 3, 2, 8), 12, entry_count=2)
    keys = set(channel.state_dict())
    rows = channel.rows({"entry": torch.tensor([0, 1])}).detach().clone()
    assert not any(k.startswith("entry_row") for k in keys)
    channel.set_entry_rows([1], torch.ones(1, 12))
    assert set(channel.state_dict()) == keys | {"entry_row_ids", "entry_row_table"}
    after = channel.rows({"entry": torch.tensor([0, 1, 1])})
    assert torch.equal(after[0], rows[0]) and torch.equal(after[1], torch.ones(12)) and torch.equal(after[2], torch.ones(12))
    after.sum().backward()
    assert channel.entry_row_table.grad is not None and float(channel.entry_row_table.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        channel.set_entry_rows([0, 0], torch.zeros(2, 12))


# -- the new keys -------------------------------------------------------------------------------------------------------


def test_entry_rows_train_and_survive_load_final(setup, tmp_path: Path) -> None:
    root = setup["root"]
    torch.save({"entries": torch.tensor([2]), "rows": torch.full((1, 64), 0.5)}, tmp_path / "rows.pt")
    cfg = config(root, channel={"entry_rows": {"path": str(tmp_path / "rows.pt")}}, eval={"points": [64]})
    train(cfg, tmp_path / "run")
    assert sorted({r["tokens"] for r in _eval_rows(tmp_path / "run")}) == [0, 64, 256]
    final = torch.load(tmp_path / "run" / "final.pt", weights_only=False)
    table = final["model"]["channel.entry_row_table"]
    assert table.shape == (1, 64) and not torch.allclose(table, torch.full((1, 64), 0.5))     # trained
    model = load_final(tmp_path / "run" / "final.pt")
    assert torch.equal(model.channel.entry_row_table.detach(), table)


def test_continue_takes_the_config_frames_and_grows_the_dictionary(setup, tmp_path: Path) -> None:
    root = setup["root"]
    train(config(root), tmp_path / "first")
    first = torch.load(tmp_path / "first" / "final.pt", weights_only=False)["model"]
    with pytest.raises(RuntimeError):                       # exact loading refuses the grown dictionary
        train(config(root, ontology=root / "grown.pt", train={"init_from": str(tmp_path / "first" / "final.pt")}), tmp_path / "exact")
    cfg = config(root, ontology=root / "grown.pt", train={"init_from": str(tmp_path / "first" / "final.pt"), "init_mode": "continue",
                                                          "total_tokens": 64})
    train(cfg, tmp_path / "second")
    rows = {r["stratum"]: r["loss"] for r in _eval_rows(tmp_path / "second") if r["tokens"] == 0}
    assert np.isfinite(rows["after"])
    model = load_final(tmp_path / "second" / "final.pt")
    composer = model.channel.composer
    assert composer.atomics.shape[0] == 7 and composer.relation_count == 3
    assert composer.schedule.offsets.tolist() == [0, 3, 5, 7, 9]
    # the continued run started from the first run's rows (step-0 evaluation of the shared entries reads them)
    fresh = config(root, ontology=root / "grown.pt", train={"total_tokens": 64})
    train(fresh, tmp_path / "fresh")
    zero = lambda p: {r["stratum"]: r["loss"] for r in _eval_rows(p) if r["tokens"] == 0}      # noqa: E731
    assert zero(tmp_path / "second")["all"] != pytest.approx(zero(tmp_path / "fresh")["all"], abs=1e-6)
    assert first["channel.composer.atomics"].shape[0] == 6


def test_continue_maps_plain_weights_into_lora_and_merges_adapters(setup, tmp_path: Path) -> None:
    root = setup["root"]
    host = {"pretrained": str(root / "tiny-host"), "host_mode": "train"}
    train(config(root, model=host), tmp_path / "full")
    lora = config(root, model={**host, "host_mode": "lora", "lora_rank": 4},
                  train={"init_from": str(tmp_path / "full" / "final.pt"), "init_mode": "continue", "total_tokens": 0})
    from vsa_embed.training.lm import build_channel, build_model, wrap_host
    resolved = resolve_config(lora)
    ontology = torch.load(root / "ontology.pt", weights_only=False)
    base = build_model(resolved)
    channel, context = build_channel(resolved, ontology, 64, host=base)
    model = wrap_host(resolved, base, channel, context)
    state = torch.load(tmp_path / "full" / "final.pt", weights_only=False)
    record = load_continuation_state(model, state)
    assert record["lora_mapped"] > 0
    name = "model.transformer.h.0.attn.c_attn"
    assert torch.equal(model.state_dict()[f"{name}.base.weight"], state["model"][f"{name}.weight"])
    # merging: adapters of a LoRA state are folded into the weights of a plain (frozen) host
    with torch.no_grad():
        model.model.transformer.h[0].attn.c_attn.lora_b.fill_(0.01)
    lora_state = {"model": {k: v for k, v in model.state_dict().items()}, "trainable_only": False}
    lora_state["model"] = {k.replace(".base.", "."): v for k, v in lora_state["model"].items()}
    frozen = resolve_config(config(root, model={**host, "host_mode": "frozen"}))
    base2 = build_model(frozen)
    channel2, context2 = build_channel(frozen, ontology, 64, host=base2)
    plain = wrap_host(frozen, base2, channel2, context2)
    merged = load_continuation_state(plain, lora_state, merge_lora=True)
    assert merged["merged_adapters"] > 0
    x = torch.randn(3, 64)
    wrapped = model.model.transformer.h[0].attn.c_attn
    assert torch.allclose(plain.model.transformer.h[0].attn.c_attn(x), wrapped(x), atol=1e-5)


def test_quantized_frozen_host_trains_only_the_channel_and_reloads(setup, tmp_path: Path) -> None:
    root = setup["root"]
    host = {"pretrained": str(root / "tiny-host"), "host_mode": "train"}
    train(config(root, model=host), tmp_path / "full")
    cfg = config(root, model={**host, "host_mode": "frozen", "host_quantization": {"scheme": "rtn", "group_size": 32}},
                 train={"init_from": str(tmp_path / "full" / "final.pt"), "init_mode": "continue", "save_trainable_only": True})
    train(cfg, tmp_path / "q4")
    manifest = json.loads((tmp_path / "q4" / "manifest.json").read_text())
    q = manifest["host_quantization"]
    assert q["scheme"] == "rtn" and q["bits_per_weight"] == pytest.approx(5.0) and q["host_bytes"] > 0
    final = torch.load(tmp_path / "q4" / "final.pt", weights_only=False)
    assert final.get("trainable_only") and not any(k.startswith("model.") for k in final["model"])
    model = load_final(tmp_path / "q4" / "final.pt")
    weight = model.model.transformer.h[0].mlp.c_fc.weight.detach()            # converted to nn.Linear and quantized
    groups = weight.reshape(weight.shape[0], -1, 32)
    assert int(max(len(torch.unique(g)) for g in groups.reshape(-1, 32)[:20])) <= 16
    full = torch.load(tmp_path / "full" / "final.pt", weights_only=False)["model"]
    assert not torch.equal(model.model.transformer.wte.weight, torch.zeros_like(model.model.transformer.wte.weight))
    assert torch.equal(model.model.transformer.wte.weight, full["model.transformer.wte.weight"])     # embedding kept 16/32-bit
    # load_final rebuilds the quantized host: its end-of-run loss equals the logged one
    from vsa_embed.data.corpus import TokenCorpus, eval_windows
    from vsa_embed.training.lm import evaluate
    resolved = resolve_config(cfg)
    corpus = TokenCorpus.open(root / "eval")
    result = evaluate(model, corpus, eval_windows(corpus, count=4, length=32), resolved, None, set(), torch.device("cpu"))
    logged = {r["stratum"]: r["loss"] for r in _eval_rows(tmp_path / "q4") if r["tokens"] == 256}
    assert result["all"]["loss"] == pytest.approx(logged["all"], abs=1e-5)


def test_host_quantization_refuses_a_trained_host(setup, tmp_path: Path) -> None:
    root = setup["root"]
    cfg = config(root, model={"pretrained": str(root / "tiny-host"), "host_mode": "train", "host_quantization": {"scheme": "rtn"}})
    with pytest.raises(ValueError):
        train(cfg, tmp_path / "bad")
