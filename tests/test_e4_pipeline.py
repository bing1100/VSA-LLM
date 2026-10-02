"""Trainer → E4 report end to end on the tiny model (real run folders, kill-and-resume check)."""

import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import build_corpus
from vsa_embed.experiments.e4_report import write_report
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import train


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


def test_trainer_runs_feed_the_report(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    full = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3)],
                                 holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the cat saw hydroxychloroquine in New York and acetaminophen too." for i in range(60)]
    build_corpus(texts, tmp_path / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    build_corpus(texts[:20], tmp_path / "eval", tokenizer_name="gpt2", table=full, eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    torch.save({"entry_count": len(full.entry_concepts), "atomic_count": 6, "relation_count": 2,
                "offsets": torch.tensor([0, 2, 4, 6, 8]), "relations": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
                "fillers": torch.tensor([0, 1, 1, 2, 3, 4, 4, 5]), "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": [60, 60, 0, 60]}, tmp_path / "ontology.pt")
    channels = {"C0": {"mode": "none"}, "C1": {"mode": "random"}, "C2": {"mode": "free", "free_dimension": 8},
                "C5": {"mode": "compose", "context_window": 4}}
    runs = tmp_path / "runs"

    def config(name: str, seed: int, **train_extra) -> dict:
        return {"seed": seed, "device": "cpu", "experiment": f"e4-test-tiny-{name}-s{seed}",
                "model": {"size": "tiny", "seq_len": 32},
                "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 32 * 2 * 8, "lr": 3e-3, "warmup_tokens": 64,
                          "log_every": 2, **train_extra},
                "data": {"train": str(tmp_path / "train"), "eval": str(tmp_path / "eval"),
                         "ontology": str(tmp_path / "ontology.pt"), "min_subtokens": 1},
                "eval": {"windows": 6, "batch": 2, "first_tokens": 64, "save_window_losses": True},
                "channel": {"dimension": 16, "key_dimension": 8, **channels[name.split("@")[0]]}}

    for name in channels:
        for seed in (1, 2):
            train(config(name, seed), runs / f"tiny-{name}-s{seed}")
    resumed = config("C5@resume", 1, stop_after_steps=3)
    assert train(resumed, runs / "tiny-C5@resume-s1").get("interrupted")
    train(resumed, runs / "tiny-C5@resume-s1", resume=True)

    summary = write_report([runs], tmp_path / "report", resamples=500)
    (cohort,) = summary["cohorts"].values()
    assert cohort["conditions"] == ["C0", "C1", "C2", "C5"] and cohort["seeds"]["C5"] == [1, 2]
    assert cohort["resume_checks"][0]["verdict"] == "identical"
    comparison = cohort["paired"]["C0"]["C5"]["all"]
    assert comparison["method"] == "window bootstrap" and comparison["windows"] == 6
    assert comparison["tokens"] == 2 * cohort["final_loss"]["all"]["tokens"]
    assert set(cohort["gate"]["C5"]["items"]) == {"1", "2", "3", "4"}
    assert cohort["throughput"]["C5"]["channel_parameters"] > 0
    assert "after_heldout" in cohort["convergence"] and (tmp_path / "report" / "report.md").is_file()
    assert json.loads((tmp_path / "report" / "manifest.json").read_text())["complete_runs"] == 9
