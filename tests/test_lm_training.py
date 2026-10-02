import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import build_corpus
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import stratum_masks, train


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("lm")
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


def config(root: Path, mode: str, **extra) -> dict:
    return {"seed": 0, "device": "cpu",
            "model": {"size": "tiny", "seq_len": 32},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 6, "lr": 3e-3, "warmup_tokens": 64,
                      "log_every": 2, **extra.pop("train", {})},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                     "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 64},
            "channel": {"mode": mode, "dimension": 16, "key_dimension": 8, **extra.pop("channel", {})}}


@pytest.mark.parametrize("mode", ["none", "compose", "free", "random", "hashed"])
def test_every_condition_trains_and_logs_strata(setup, tmp_path: Path, mode: str) -> None:
    extra = {"channel": {"hashed_buckets": 64}} if mode == "hashed" else {}
    result = train(config(setup["root"], mode, **extra), tmp_path / mode)
    assert result["steps"] == 6
    rows = [json.loads(line) for line in (tmp_path / mode / "metrics.jsonl").read_text().splitlines()]
    strata = {r["stratum"] for r in rows if r["type"] == "eval"}
    assert {"all", "unlinked", "after", "after_heldout"} <= strata
    assert (tmp_path / mode / "final.pt").is_file() and (tmp_path / mode / "manifest.json").is_file()


def test_attentive_context_semantic_head_and_growth_run(setup, tmp_path: Path) -> None:
    cfg = config(setup["root"], "compose", train={"semantic_weight": 0.1},
                 channel={"context_window": 4, "developmental": {"screen_every": 2, "min_usages": 1,
                                                                 "min_contributions": 1, "permutations": 9}})
    assert train(cfg, tmp_path / "rich")["steps"] == 6


def test_interrupted_run_resumes_to_the_same_result(setup, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    train(config(setup["root"], "compose"), tmp_path / "straight")
    first = train(config(setup["root"], "compose", train={"stop_after_steps": 3}), tmp_path / "resumed")
    assert first.get("interrupted")
    train(config(setup["root"], "compose"), tmp_path / "resumed", resume=True)
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_stratum_masks_place_after_and_inside_targets() -> None:
    ids = torch.zeros(1, 20, dtype=torch.long)
    spans = {"batch": torch.tensor([0]), "start": torch.tensor([3]), "end": torch.tensor([5]),
             "entry": torch.tensor([2]), "length": torch.tensor([3])}
    masks = stratum_masks(ids, spans, None, {2})
    assert masks["inside"][0].nonzero().flatten().tolist() == [3, 4]
    assert masks["after_heldout"][0].nonzero().flatten().tolist() == list(range(5, 13))
    assert not bool((masks["unlinked"] & masks["after"]).any())


def test_pretrained_host_lora_trains_only_adapters_and_channel(setup, tmp_path: Path, monkeypatch) -> None:
    import vsa_embed.training.lm as lm_module
    tiny = lambda config: transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=1, n_head=2))
    monkeypatch.setattr(lm_module, "build_model", tiny)
    cfg = config(setup["root"], "compose")
    cfg["model"].update(pretrained="fake-host", host_mode="lora", lora_rank=2)
    result = train(cfg, tmp_path / "lora")
    assert result["steps"] == 6
    state = torch.load(tmp_path / "lora" / "final.pt", weights_only=False)["model"]
    assert any("lora_a" in k for k in state)


def test_eval_rows_carry_training_tokens_and_stratum_counts(setup, tmp_path: Path) -> None:
    from vsa_embed.convergence import load_curves
    train(config(setup["root"], "none"), tmp_path / "rows")
    rows = [json.loads(line) for line in (tmp_path / "rows" / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if r["type"] == "eval"]
    assert sorted({r["tokens"] for r in evals}) == [0, 64, 128, 256, 384]
    assert all(r["stratum_tokens"] >= 0 for r in evals)
    assert [t for t, _ in load_curves(tmp_path / "rows")["all"]] == [64.0, 128.0, 256.0, 384.0]


def test_window_losses_are_opt_in_and_sum_to_the_logged_strata(setup, tmp_path: Path) -> None:
    import numpy as np
    import yaml
    from vsa_embed.training.lm import load_window_losses
    train(config(setup["root"], "compose"), tmp_path / "off")
    assert not (tmp_path / "off" / "eval_windows.npz").exists()
    assert "save_window_losses" not in yaml.safe_load((tmp_path / "off" / "resolved_config.yaml").read_text())["eval"]
    cfg = config(setup["root"], "compose")
    cfg["eval"]["save_window_losses"] = True
    train(cfg, tmp_path / "on")
    data = load_window_losses(tmp_path / "on" / "eval_windows.npz")
    rows = [json.loads(line) for line in (tmp_path / "on" / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if r["type"] == "eval"]
    assert set(data["evals"]) == {r["tokens"] for r in evals}
    assert data["starts"].shape == (4,)
    for row in evals:
        sums, counts = data["evals"][row["tokens"]]
        assert sums.shape == counts.shape == (len(data["strata"]), 4)
        index = data["strata"].index(row["stratum"])
        assert int(counts[index].sum()) == row["stratum_tokens"]
        if row["stratum_tokens"]:
            assert float(sums[index].sum() / counts[index].sum()) == pytest.approx(row["loss"], rel=1e-5)
    # Windows are fixed by the eval corpus, so another condition sees the same windows and counts.
    other = config(setup["root"], "free")
    other["eval"]["save_window_losses"] = True
    train(other, tmp_path / "free")
    free = load_window_losses(tmp_path / "free" / "eval_windows.npz")
    np.testing.assert_array_equal(free["starts"], data["starts"])
    np.testing.assert_array_equal(free["evals"][384][1], data["evals"][384][1])


def test_resume_without_a_checkpoint_starts_over_and_keeps_the_partial_files(setup, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cfg = config(setup["root"], "compose")
    cfg["eval"]["save_window_losses"] = True
    train(cfg, tmp_path / "straight")
    run = tmp_path / "crashed"
    train(cfg, run)
    for name in ("checkpoint.pt", "final.pt", "manifest.json"):     # as if it died before its first checkpoint
        (run / name).unlink()
    partial = (run / "metrics.jsonl").read_text()
    train(cfg, run, resume=True)                                    # what the job queue does on a retry
    assert (run / "metrics.aborted-1.jsonl").read_text() == partial
    assert (run / "eval_windows.aborted-1.npz").is_file() and (run / "resolved_config.aborted-1.yaml").is_file()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert sum(r["type"] == "eval" and r["tokens"] == 0 and r["stratum"] == "all" for r in rows) == 1
    a = torch.load(tmp_path / "straight" / "final.pt", weights_only=False)["model"]
    b = torch.load(run / "final.pt", weights_only=False)["model"]
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    (run / "checkpoint.pt").unlink()                                # a finished run is never restarted
    with pytest.raises(FileExistsError):
        train(cfg, run, resume=True)


def test_window_losses_survive_an_interrupted_run(setup, tmp_path: Path) -> None:
    from vsa_embed.training.lm import load_window_losses
    torch.set_num_threads(1)
    cfg = config(setup["root"], "none", train={"stop_after_steps": 3})
    cfg["eval"]["save_window_losses"] = True
    assert train(cfg, tmp_path / "run").get("interrupted")
    assert set(load_window_losses(tmp_path / "run" / "eval_windows.npz")["evals"]) == {0, 64, 128}
    train(cfg, tmp_path / "run", resume=True)
    assert set(load_window_losses(tmp_path / "run" / "eval_windows.npz")["evals"]) == {0, 64, 128, 256, 384}
