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
