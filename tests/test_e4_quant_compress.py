"""D4.3 PTQ (e4_quant) and E4.5 table compression (e4_compress) on tiny trained runs (CPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import build_corpus
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import load_final, train

SINGLE = ["cat", "dog", "tree", "house", "water", "music", "garden", "river", "bread", "stone"]
MULTI = ["hydroxychloroquine", "new york", "acetaminophen"]


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("wpq")
    words = SINGLE + MULTI
    full = AliasTable.from_pairs([(w, i) for i, w in enumerate(words)], holdout=[words.index("acetaminophen")], include_holdout=True)
    rng = np.random.default_rng(0)
    texts = []
    for i in range(80):
        order = rng.permutation(len(SINGLE))
        texts.append(f"Doc {i}: the " + " and the ".join(SINGLE[j] for j in order[:6]) + " near hydroxychloroquine in New York, "
                     "with acetaminophen too.")
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=100_000,
                 batch_texts=16, workers=2)
    build_corpus(texts[:24], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=100_000,
                 batch_texts=8, workers=2)
    entries = len(full.entry_concepts)
    frames = [[(e % 3, (e * 5 + k) % 12) for k in range(2 + e % 2)] for e in range(entries)]
    offsets = np.cumsum([0] + [len(f) for f in frames])
    ontology = {"entry_count": entries, "atomic_count": 12, "relation_count": 3, "offsets": torch.tensor(offsets),
                "relations": torch.tensor([r for f in frames for r, _ in f]), "fillers": torch.tensor([a for f in frames for _, a in f]),
                "heldout_entries": sorted(full.heldout_entries()), "train_frequency": [50] * entries}
    torch.save(ontology, root / "ontology.pt")
    channels = {"C0": {"mode": "none"}, "C2": {"mode": "free", "free_dimension": 8},
                "C3": {"mode": "compose", "composition": "bundle", "dimension": 16}}
    for name, channel in channels.items():
        for seed in (1, 2):
            config = {"seed": seed, "device": "cpu", "experiment": f"e4-test-tiny-{name}-s{seed}",
                      "model": {"size": "tiny", "seq_len": 32},
                      "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 6, "lr": 3e-3, "warmup_tokens": 64,
                                "log_every": 2},
                      "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                               "min_subtokens": 1},
                      "eval": {"windows": 6, "batch": 3, "first_tokens": 128, "save_window_losses": True},
                      "channel": channel}
            train(config, root / "runs" / f"tiny-{name}-s{seed}")
    return {"root": root, "runs": root / "runs"}


# -- D4.3: post-training quantization --------------------------------------------------------------------


def test_quantized_channel_lm_forward_runs(runs: dict) -> None:
    from vsa_embed.experiments.e4_quant import model_bytes, quantize_channel, quantize_host
    model = load_final(runs["runs"] / "tiny-C3-s1" / "final.pt")
    ids = torch.randint(0, 50257, (2, 32))
    spans = {"batch": torch.tensor([0, 1]), "start": torch.tensor([3, 5]), "end": torch.tensor([3, 6]), "inject": torch.tensor([3, 6]),
             "entry": torch.tensor([0, 1]), "confidence": torch.ones(2), "length": torch.tensor([1, 2])}
    with torch.no_grad():
        reference = float(model(ids, spans=spans, labels=ids)["loss"])
    before = model_bytes(model)
    info = quantize_host(model, 8, 32)
    sizes = quantize_channel(model, 8, 32)
    with torch.no_grad():
        quantized = float(model(ids, spans=spans, labels=ids)["loss"])
    assert np.isfinite(quantized) and abs(quantized - reference) < 0.05
    assert info["linear_quantized"] == 8 and info["linear_skipped"] == [] and info["head_tied"]
    after = model_bytes(model, sizes, bits=8, group_size=32)
    assert after["embedding"] == before["embedding"] == 50257 * 64 * 2 and after["head"] == 0
    assert after["linear"] < before["linear"] and after["channel"] < before["channel"]
    assert before["linear_nominal"] == before["linear"] and after["linear_nominal"] < before["linear"]
    weights = 2 * (64 * 192 + 64 * 64 + 64 * 256 + 256 * 64)          # two tiny GPT-2 blocks
    biases = 2 * 2 * (192 + 64 + 256 + 64)
    assert after["linear_nominal"] == weights + 2 * 2 * (192 + 64 + 256 + 64) + biases     # int8 codes, FP16 scale per row
    assert after["total_nominal"] == after["total"] - after["linear"] + after["linear_nominal"]
    with pytest.raises(ValueError, match="bits"):
        model_bytes(model, sizes)
    assert "channel.composer.atomics" in sizes and "channel.projector.weight" in sizes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="INT4 tile-packed layout needs CUDA")
def test_int4_quantized_channel_lm_forward_runs_on_cuda(runs: dict) -> None:
    from vsa_embed.evaluation.quantization import auto_group_size
    from vsa_embed.experiments.e4_quant import quantize_host
    try:
        free = torch.cuda.mem_get_info()[0]
    except RuntimeError:            # a full GPU (shared job queue) cannot even open a context
        free = 0
    if free < 1 << 30:
        pytest.skip("less than 1 GiB of free GPU memory")
    model = load_final(runs["runs"] / "tiny-C0-s1" / "final.pt", "cuda")
    ids = torch.randint(0, 50257, (2, 32), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = float(model(ids, labels=ids)["loss"])
        info = quantize_host(model, 4, auto_group_size(model.model))
        quantized = float(model(ids, labels=ids)["loss"])
    assert info["linear_quantized"] == 8 and abs(quantized - reference) < 0.5


def test_merged_lora_gives_the_same_outputs() -> None:
    from vsa_embed.evaluation.quantization import merge_lora
    from vsa_embed.integrations.transformers import LoRALinear, add_lora
    torch.manual_seed(0)
    host = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=100, n_positions=16, n_embd=32, n_layer=1, n_head=2)).eval()
    add_lora(host, rank=2)
    for module in host.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_b, std=0.1)
    ids = torch.randint(0, 100, (2, 12))
    with torch.no_grad():
        before = host(ids).logits
        assert merge_lora(host) == 4
        after = host(ids).logits
    assert not any(isinstance(m, LoRALinear) for m in host.modules())
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)


def test_e4_quant_cli_end_to_end(runs: dict, tmp_path: Path) -> None:
    from vsa_embed.experiments.e4_quant import main
    out = tmp_path / "quant"
    main(["--runs", str(runs["runs"]), "--output", str(out), "--bits", "8", "--device", "cpu", "--resamples", "200"])
    for name in ("quant.json", "report.md", "manifest.json", "resolved_config.yaml"):
        assert (out / name).is_file()
    result = json.loads((out / "quant.json").read_text())
    records = {r["id"]: r for r in result["runs"]}
    assert len(records) == 6
    c0, c3 = records["runs__tiny-C0-s1"], records["runs__tiny-C3-s1"]
    assert set(c0["variants"]) == {"ref", "int8-A"} and set(c3["variants"]) == {"ref", "int8-A", "int8-B"}
    # `ref` is the trainer's own evaluation: it reproduces the run's saved final per-window losses.
    assert c3["ref_check"]["paired"] and c3["ref_check"]["counts_equal"] and c3["ref_check"]["max_abs_window_sum_diff"] < 1e-6
    assert c3["variants"]["int8-B"]["bytes"]["channel"] < c3["variants"]["int8-A"]["bytes"]["channel"]
    assert c3["variants"]["int8-A"]["bytes"]["linear"] < c3["variants"]["ref"]["bytes"]["linear"]
    cohort = next(iter(result["cohorts"].values()))
    assert cohort["baseline"] == "C0" and cohort["references"] == ["C0", "C2"]
    retained = cohort["retention"]["C0"]["C3"]["int8-A"]["all"]
    # The bf16 gain is the pooled, token-weighted (C3 − C0) loss difference at `ref`.
    loss = lambda rec, v: rec["variants"][v]["strata"]["all"]
    pooled = lambda cond, v: sum(loss(records[f"runs__tiny-{cond}-s{s}"], v)["loss"] * loss(records[f"runs__tiny-{cond}-s{s}"], v)["tokens"]
                                 for s in (1, 2)) / sum(loss(records[f"runs__tiny-{cond}-s{s}"], v)["tokens"] for s in (1, 2))
    assert retained["gain_bf16"]["mean"] == pytest.approx(pooled("C3", "ref") - pooled("C0", "ref"), abs=1e-5)
    assert retained["gain_quantized"]["mean"] == pytest.approx(pooled("C3", "int8-A") - pooled("C0", "int8-A"), abs=1e-5)
    assert retained["gain_change"]["mean"] == pytest.approx(retained["gain_quantized"]["mean"] - retained["gain_bf16"]["mean"], abs=1e-9)
    assert "int8-B" in cohort["retention"]["C0"]["C3"] and "C3" in cohort["retention"]["C2"]
    report = (out / "report.md").read_text()
    assert "Retained gain over C0" in report and "Retained gain over C2" in report and "Bytes (MiB" in report
    # Re-analysis without re-evaluation gives the same numbers.
    main(["--runs", str(runs["runs"]), "--output", str(out), "--bits", "8", "--device", "cpu", "--resamples", "200", "--report-only"])
    again = json.loads((out / "quant.json").read_text())
    assert again["cohorts"] == result["cohorts"]


def test_e4_quant_refuses_int4_on_cpu(runs: dict, tmp_path: Path) -> None:
    from vsa_embed.experiments.e4_quant import run
    with pytest.raises(RuntimeError, match="INT4"):
        run([runs["runs"]], tmp_path / "q4", bits=(4,), device="cpu")


# -- E4.5: table compression -----------------------------------------------------------------------------


def test_e4_compress_end_to_end(runs: dict, tmp_path: Path) -> None:
    from vsa_embed.experiments.e4_compress import main, single_token_rows
    root = runs["root"]
    config = {"run": str(runs["runs"] / "tiny-C3-s1"), "device": "cpu", "resamples": 200,
              "rows": {"link_corpus": str(root / "train"), "frequency_corpus": str(root / "train"), "bins_per_million": [20_000, 60_000]},
              "group_size": 32, "fit": {"steps": 10, "lr": 0.01},
              "finetune": {"tokens": 128, "micro_batch": 2, "methods": ["composed", "int"]},
              "composed": [{"dimension": 8, "delta_bits": 0}, {"dimension": 8, "delta_bits": 4}, {"init": "run", "delta_bits": 2}],
              "points": [{"method": "int", "bits": 4}, {"method": "int", "bits": 32}, {"method": "albert", "rank": 4}]}
    (tmp_path / "compress.yaml").write_text(yaml.safe_dump(config))
    out = tmp_path / "compress"
    main(["--config", str(tmp_path / "compress.yaml"), "--output", str(out)])
    for name in ("compress.json", "report.md", "windows.npz", "rows.npz", "manifest.json", "resolved_config.yaml",
                 "figures/loss-vs-bytes.png"):
        assert (out / name).is_file(), name
    summary = json.loads((out / "compress.json").read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    singles = {tokenizer.encode(" " + w)[0] for w in SINGLE}
    rows = np.load(out / "rows.npz")
    assert set(rows["token_ids"].tolist()) <= singles | {tokenizer.encode(w)[0] for w in ("Cat", "Dog", "cat")}
    assert len(singles & set(rows["token_ids"].tolist())) == len(SINGLE)
    points = summary["points"]
    budgets = {name: p["bytes"] for name, p in points.items() if p["method"] == "composed" and not name.endswith("+ft")}
    assert set(budgets) == {"composed-d8-delta0", "composed-d8-delta4", "composed-d16-delta2-run"}
    for name, point in points.items():
        if point.get("budget_of"):
            assert point["bytes"] <= point["budget"] == budgets[point["budget_of"].removesuffix("+ft")], name
            assert point["method"] in {"int", "albert", "qr", "hashing", "hash_embedding", "tt"}
    # bits → ∞: the compressed forward equals the dense forward exactly.
    full = points["int-bits32"]
    for stratum, value in points["dense"]["strata"].items():
        if value["tokens"]:
            assert full["strata"][stratum]["loss"] == pytest.approx(value["loss"], abs=1e-6), stratum
    assert points["dense"]["strata"]["R_input"]["tokens"] > 0
    assert any(name.endswith("+ft") and "finetune" in p for name, p in points.items())
    run_point = next(p for name, p in points.items() if name.endswith("-run") and not name.endswith("+ft"))
    assert run_point["settings"]["dimension"] == 16 and run_point["marginal_bytes"] < run_point["bytes"]
    equal = summary["analysis"]["equal_bytes"]
    assert set(equal) >= set(budgets)
    first = next(iter(equal.values()))
    assert all("all" in comparison["difference"] for comparison in first.values())
    assert "Equal bytes" in (out / "report.md").read_text()
    again = single_token_rows(root / "train", max_tokens=10**9, heldout=set())
    np.testing.assert_array_equal(again["token_ids"], rows["token_ids"])


def test_committed_compress_configs_resolve() -> None:
    from vsa_embed.evaluation.table_compression import METHODS
    from vsa_embed.experiments.e4_compress import MAX_FINETUNE_TOKENS, resolve
    paths = sorted((Path(__file__).parents[1] / "experiments/e4-small-lm/configs/compress").glob("*.yaml"))
    assert paths
    for path in paths:
        config = resolve(yaml.safe_load(path.read_text()))
        assert set(config["baselines"]) <= set(METHODS) - {"composed"} and config["composed"]
        assert all(p["method"] in METHODS for p in config["points"])
        assert int(config["finetune"]["tokens"]) <= MAX_FINETUNE_TOKENS
    with pytest.raises(ValueError, match="capped"):
        resolve({"run": "x", "finetune": {"tokens": MAX_FINETUNE_TOKENS + 1}})


def test_single_token_rows_keep_tokens_the_linker_treats_as_concepts(tmp_path: Path) -> None:
    from vsa_embed.experiments.e4_compress import single_token_rows
    # Token 5 occurs 4 times but is linked once (a stray match); token 7 is linked at both
    # occurrences, mostly to entry 2; token 9 is linked inside a 2-token span only.
    tokens = np.array([5, 5, 5, 5, 7, 7, 9, 9, 7], dtype=np.uint16)
    tokens.tofile(tmp_path / "tokens.bin")
    np.savez(tmp_path / "spans.npz", start=np.array([0, 4, 5, 6, 8], dtype=np.int32), end=np.array([0, 4, 5, 7, 8], dtype=np.int32),
             inject=np.array([0, 4, 5, 7, 8], dtype=np.int32), entry=np.array([1, 2, 2, 4, 3], dtype=np.int32),
             length=np.array([1, 1, 1, 2, 1], dtype=np.uint8), confidence=np.ones(5, dtype=np.float16))
    (tmp_path / "manifest.json").write_text(json.dumps({"dtype": "uint16", "tokens": 9}))
    rows = single_token_rows(tmp_path, max_tokens=100, heldout=set())
    assert rows["token_ids"].tolist() == [7] and rows["entries"].tolist() == [2] and rows["links"].tolist() == [3]
    assert rows["conflicting_tokens"] == 1 and rows["dropped_tokens"] == 1 and rows["dropped_links"] == 1
    loose = single_token_rows(tmp_path, max_tokens=100, heldout=set(), min_link_rate=0.0)
    assert loose["token_ids"].tolist() == [5, 7]
    held = single_token_rows(tmp_path, max_tokens=100, heldout={2}, min_link_rate=0.0)       # entry 3 remains for token 7
    assert held["token_ids"].tolist() == [5, 7] and held["entries"].tolist() == [1, 3]
