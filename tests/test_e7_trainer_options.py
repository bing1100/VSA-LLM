"""WP-E7 trainer options: `train.init_from` and `eval.reference_strata` (opt-in, absent keys change nothing)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import load_final, load_reference_strata, load_window_losses, save_reference_strata, train


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e7trainer")
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


def config(root: Path, **extra) -> dict:
    return {"seed": 0, "device": "cpu", "model": {"size": "tiny", "seq_len": 32},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 4, "lr": 3e-3, "warmup_tokens": 64,
                      "log_every": 2, **extra.pop("train", {})},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                     "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 64, **extra.pop("eval", {})},
            "channel": {"mode": "compose", "dimension": 16, "key_dimension": 8}}


def test_init_from_starts_training_from_a_saved_state(setup, tmp_path: Path) -> None:
    root = setup["root"]
    train(config(root), tmp_path / "first")
    first = torch.load(tmp_path / "first" / "final.pt", weights_only=False)
    fresh = train(config(root, train={"total_tokens": 0}), tmp_path / "fresh")
    started = train(config(root, train={"total_tokens": 0, "init_from": str(tmp_path / "first" / "final.pt")}),
                    tmp_path / "started")
    assert fresh["steps"] == started["steps"]
    rows = lambda p: {r["stratum"]: r["loss"] for r in map(json.loads, (p / "metrics.jsonl").read_text().splitlines())
                      if r["type"] == "eval" and r["tokens"] == 0}
    trained_end = {r["stratum"]: r["loss"] for r in map(json.loads, (tmp_path / "first" / "metrics.jsonl").read_text().splitlines())
                   if r["type"] == "eval" and r["tokens"] == max(json.loads(l)["tokens"] for l in (tmp_path / "first" / "metrics.jsonl").read_text().splitlines())}
    assert rows(tmp_path / "started")["after"] == pytest.approx(trained_end["after"], abs=1e-5)
    assert rows(tmp_path / "fresh")["after"] != pytest.approx(trained_end["after"], abs=1e-5)
    assert load_final(tmp_path / "started" / "final.pt").channel.composer.atomics.shape == first["model"]["channel.composer.atomics"].shape


def test_reference_strata_are_logged_and_paired_by_window(setup, tmp_path: Path) -> None:
    root = setup["root"]
    corpus = TokenCorpus.open(root / "eval")
    starts = eval_windows(corpus, count=4, length=32)
    masks = {"early": np.zeros((4, 31), dtype=bool), "none": np.zeros((4, 31), dtype=bool)}
    masks["early"][:, :5] = True
    save_reference_strata(tmp_path / "ref.npz", starts, masks, 32)
    loaded = load_reference_strata(tmp_path / "ref.npz", starts, 32)
    assert set(loaded) == {"ref_early", "ref_none"} and loaded["ref_early"].sum() == 20
    with pytest.raises(ValueError):
        load_reference_strata(tmp_path / "ref.npz", [s + 1 for s in starts], 32)
    train(config(root, eval={"reference_strata": str(tmp_path / "ref.npz"), "save_window_losses": True}), tmp_path / "run")
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    early = [r for r in rows if r.get("stratum") == "ref_early"]
    assert early and all(r["stratum_tokens"] == 20 for r in early)
    windows = load_window_losses(tmp_path / "run" / "eval_windows.npz")
    index = windows["strata"].index("ref_early")
    sums, counts = windows["evals"][max(windows["evals"])]
    assert counts[index].tolist() == [5, 5, 5, 5]
    resolved = yaml.safe_load((tmp_path / "run" / "resolved_config.yaml").read_text())
    assert "init_from" not in resolved["train"]


def test_configs_without_the_new_keys_resolve_and_log_as_before(setup, tmp_path: Path) -> None:
    root = setup["root"]
    train(config(root), tmp_path / "plain")
    rows = [json.loads(l) for l in (tmp_path / "plain" / "metrics.jsonl").read_text().splitlines()]
    assert not any(str(r.get("stratum", "")).startswith("ref_") for r in rows)
    resolved = yaml.safe_load((tmp_path / "plain" / "resolved_config.yaml").read_text())
    assert "reference_strata" not in resolved["eval"] and "init_from" not in resolved["train"]
