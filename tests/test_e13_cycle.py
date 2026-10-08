"""E13 learning cycle: round split, round-2 text, reference strata, tokens-to-criterion statistics, configs and plan."""

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from vsa_embed.experiments import e13_cycle as e13

torch.set_num_threads(1)


def toy_ontology() -> dict:
    """Entries: 0–3 round 1 (2, 3 teams), 4 round 2 (held out), 5 synthetic. Atoms name concepts as `term:<name>`."""
    names = ["Alpha Hub", "Beta Feed", "Red Team", "Blue Team", "Gamma Tool", "Zero Bot"]
    atoms = ["type:system", "type:dataset", "type:team", "term:Red Team", "term:Blue Team", "term:Gamma Tool", "term:Beta Feed"]
    frames = [[(0, 0), (1, 3), (2, 6), (2, 5)], [(0, 1), (1, 4)], [(0, 2)], [(0, 2)], [(0, 0), (1, 3)], [(0, 0), (1, 4)]]
    ontology = {"entry_count": 6, "relation_names": ["is_a", "owned_by", "depends_on"], "atomic_names": atoms,
                "concept_names": names, "entry_concepts": [(i,) for i in range(6)], "heldout_entries": [4, 5],
                "heldout_real_entries": [4], "synthetic_entries": [5], "train_frequency": [5, 5, 5, 5, 0, 0]}
    return e13.with_frames(ontology, frames)


def test_seed_ontology_empties_round2_drops_its_fillers_derives_inverses_and_erases() -> None:
    ontology = toy_ontology()
    seeded, erased = e13.seed_ontology(ontology, [4], erase_fraction=0.0, min_keep=1, derived_inverses={"owns": "owned_by"}, seed=0)
    frames = e13.frames_of(seeded)
    assert frames[4] == [] and frames[5] == e13.frames_of(ontology)[5]
    assert (2, 5) not in frames[0]                                  # Alpha Hub depends_on Gamma Tool (round 2): dropped
    owns = seeded["relation_names"].index("owns")
    alpha = seeded["atomic_names"].index("term:Alpha Hub")         # a new atom naming Alpha Hub
    assert (owns, alpha) in frames[2] and (owns, seeded["atomic_names"].index("term:Beta Feed")) in frames[3]
    assert seeded["relation_count"] == 4 and seeded["atomic_count"] == len(ontology["atomic_names"]) + 1
    assert erased == [] and seeded["heldout_entries"] == [4, 5] and seeded["e13"]["round2_entries"] == [4]
    seeded, erased = e13.seed_ontology(ontology, [4], erase_fraction=0.5, min_keep=1, derived_inverses={"owns": "owned_by"}, seed=3)
    frames = e13.frames_of(seeded)
    assert erased and all(e in (0, 1, 2, 3) for e, _, _ in erased)
    assert all(len(frames[e]) >= 1 for e in range(4))
    assert all((r, a) not in frames[e] for e, r, a in erased)
    assert seeded["e13"]["erased"] == len(erased) <= round(0.5 * seeded["e13"]["edges_round1"])


def test_round2_documents_focus_on_round2_terms_and_hide_zero_shot_ones() -> None:
    from vsa_embed.benchmarks.glossary import generate_glossary
    glossary = generate_glossary(seed=3, terms=80, zero_shot=4, heldout_fraction=0.2, zipf=1.5)
    heldout = [t["name"] for t in glossary["terms"] if t["split"] == "heldout"]
    zero = [t["name"] for t in glossary["terms"] if t["split"] == "zeroshot"]
    texts = list(e13.round2_documents(glossary, heldout, seed=7, max_chars=40_000))
    again = list(e13.round2_documents(glossary, heldout, seed=7, max_chars=40_000))
    assert texts == again and sum(map(len, texts)) >= 40_000
    mentioned = Counter(name for name in heldout for text in texts if name in text)
    assert len(mentioned) >= 0.8 * len(heldout)
    assert not any(name in text for name in zero for text in texts)
    prepended = list(e13.definitions_prepended(texts[:20], {heldout[0]: f"{heldout[0]}: a defined term."}))
    for original, new in zip(texts[:20], prepended):
        assert new == (f"{heldout[0]}: a defined term.\n\n{original}" if heldout[0] in original else original)


def test_tokens_to_criterion_and_efficiency() -> None:
    tokens = np.array([0, 10, 20, 40])
    assert e13.tokens_to_criterion(tokens, np.array([3.0, 2.0, 1.5, 1.0]), 1.5) == 20
    assert e13.tokens_to_criterion(tokens, np.array([3.0, 2.0, 1.5, 1.0]), 1.75) == pytest.approx(15)
    assert e13.tokens_to_criterion(tokens, np.array([1.0, 0.9, 0.8, 0.7]), 2.0) == 0
    assert math.isinf(e13.tokens_to_criterion(tokens, np.array([3.0, 2.9, 2.8, 2.7]), 1.0))
    rng = np.random.default_rng(0)
    windows, counts = 40, np.full((4, 40), 8.0)

    def curve(levels):
        noise = rng.normal(0, 0.05, size=(4, windows))
        return tokens, (np.asarray(levels)[:, None] + noise) * counts, counts
    fast = [curve([2.0, 1.2, 1.0, 0.9]) for _ in range(3)]
    slow = [curve([2.0, 1.8, 1.4, 1.0]) for _ in range(3)]
    result = e13.efficiency({"read": fast, "noread": slow}, "read", "noread", resamples=300)
    assert result["ratio"] < 0.6 and result["ci_high"] < 1 and result["aulc_difference"] < 0 and result["p_value"] < 0.05
    shifted = e13.efficiency({"read": fast, "noread": slow}, "read", "noread", offset=100.0, resamples=50)
    assert shifted["ratio"] > 1
    table = e13.window_table(fast, slow, -1)
    assert table.shape == (windows, 3) and abs(table.mean()) < 0.1


transformers = pytest.importorskip("transformers")


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


@pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")
def test_reference_masks_follow_the_trainer_after_span_rule(tmp_path: Path) -> None:
    from vsa_embed.data.corpus import TokenCorpus, build_corpus, collate_windows, eval_windows
    from vsa_embed.span_channel import AliasTable
    from vsa_embed.training.lm import stratum_masks
    table = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1)])
    texts = [f"Doc {i}: hydroxychloroquine in New York and more words after it here." for i in range(30)]
    build_corpus(texts, tmp_path / "c", tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=50_000, batch_texts=8, workers=2)
    corpus = TokenCorpus.open(tmp_path / "c")
    starts = eval_windows(corpus, count=4, length=32)
    masks = e13.reference_masks(corpus, starts, 32, 1, {"first": {0}, "both": {0, 1}})
    ids, spans = collate_windows([corpus.window(s, 32, min_subtokens=1) for s in starts])
    after = stratum_masks(ids, spans, None, set())["after"].numpy()
    assert np.array_equal(masks["both"], after) and masks["first"].sum() < masks["both"].sum()


def test_round2_configs_and_the_plan(tmp_path: Path) -> None:
    config = e13.load_config(e13.ROOT / "t5.yaml")
    read = e13.round2_config(config, "SmolLM2-360M", "read", 2)
    assert read["train"]["init_mode"] == "continue" and read["train"]["init_from"].endswith("stage0/SmolLM2-360M-full-C5-s2/final.pt")
    assert read["train"]["total_tokens"] == 10_000_000 and read["data"]["seed"] == 4321
    assert read["data"]["train"].endswith("smollm2/round2/train") and read["channel"]["skip_empty_frames"]
    assert read["eval"]["reference_strata"].endswith("reference-1024x1024.npz") and read["e13"]["frames"] == "read"
    assert e13.round2_config(config, "SmolLM2-360M", "defs", 1)["data"]["train"].endswith("round2-defs/train")
    fvt = e13.round2_config(config, "SmolLM2-360M", "fvt", 1)
    assert fvt["channel"]["entry_rows"]["path"].endswith("entry_rows.pt") and fvt["e13"]["frames"] is None
    c0p = e13.round2_config(config, "SmolLM2-360M", "C0p", 1)
    assert c0p["channel"]["mode"] == "none" and c0p["train"]["init_from"].endswith("t5/SmolLM2-360M-full-C0p-s1/final.pt")
    q4 = e13.round2_config(config, "SmolLM2-360M", "q4-read", 1, scheme="rtn")
    assert q4["model"]["host_mode"] == "frozen" and q4["model"]["host_quantization"]["scheme"] == "rtn"
    assert q4["train"]["save_trainable_only"] and "host_lr" not in q4["train"]
    qlora = e13.round2_config(config, "Qwen3-1.7B-Base", "qlora", 1, scheme="rtn")
    assert qlora["model"]["host_mode"] == "lora" and qlora["train"]["init_merge_lora"] and qlora["train"]["host_lr"] == 2e-4
    stage0 = e13.stage0_config(config, "SmolLM2-360M", 1)
    assert stage0["data"]["ontology"].endswith("smollm2/seed/ontology.pt") and stage0["train"]["total_tokens"] == 50_000_000
    jobs = e13.plan(config, write_configs=False)
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names))
    assert [n for n in names if "-report" in n] == ["e13-t5-report"]
    assert {j["priority"] for j in jobs} <= {54.4985, 54.4986, 54.4987, 54.4988, 54.4989}
    by = {j["name"]: j for j in jobs}
    assert by["e13-t5-SmolLM2-360M-read-s1"]["priority"] == 54.4986 and by["e13-t5-SmolLM2-360M-q4-read-rtn-s1"]["priority"] == 54.4987
    assert by["e13-t5-SmolLM2-360M-s1-reason"]["priority"] == 54.4988 and by["e13-t5-report"]["priority"] == 54.4989
    assert names.index("e13-t5-SmolLM2-360M-s1-learn") > names.index("e13-t5-SmolLM2-360M-full-C5-s3")
    assert all(j["hours"] >= 0 for j in jobs) and 15 < sum(j["hours"] for j in jobs) < 60
    lines = e13.queue_lines(jobs[:2])
    assert lines[0].startswith("PYTHONPATH=src $PY -m vsa_embed.experiments.e13_cycle enqueue --name e13-t5-prep-")


def test_enqueue_adds_a_job_with_a_fractional_priority(tmp_path: Path) -> None:
    import json
    e13.main(["enqueue", "--queue", str(tmp_path / "q"), "--name", "e13-x", "--priority", "54.4986", "--no-resume", "--",
              "python", "-m", "x", "--flag"])
    job = json.loads((tmp_path / "q" / "e13-x.json").read_text())
    assert job["priority"] == 54.4986 and job["command"] == ["python", "-m", "x", "--flag"] and job["resume_args"] == []
    assert job["env"] == {"PYTHONPATH": "src"} and job["status"] == "pending"
