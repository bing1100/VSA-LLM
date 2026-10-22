"""E15 — HRR as a training target (`experiments/e15-hrr-target/preregistration.md`): the two losses, the read
accuracy, the compositional split check, the configs, and the regression that a config without the block is unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.compose import FrameComposer, FrameSchedule  # noqa: E402
from vsa_embed.data.corpus import TokenCorpus, build_corpus  # noqa: E402
from vsa_embed.evaluation import channel_probes as cp  # noqa: E402
from vsa_embed.experiments import e15_eval  # noqa: E402
from vsa_embed.hrr_target import HRRTarget, ontology_decoder, target_from_config  # noqa: E402
from vsa_embed.span_channel import AliasTable  # noqa: E402
from vsa_embed.training import lm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")


# -- the losses on a toy store ---------------------------------------------------------------------------------------------

def _store(atoms: int = 40, relations: int = 3, dimension: int = 64, concepts: int = 6, degree: int = 3, seed: int = 0) -> FrameComposer:
    torch.manual_seed(seed)
    frames = [[(r, int(torch.randint(0, atoms, ()))) for r in range(degree)] for _ in range(concepts)]
    return FrameComposer(FrameSchedule.from_frames(frames), atoms, relations, dimension, operator="hrr", mode="bundle")


def test_losses_have_the_right_shapes_and_gradients_flow_to_the_host_and_w_only() -> None:
    store = _store()
    target = HRRTarget(store, 24, layer=1, write_weight=1.0, read_weight=1.0)
    hidden = torch.randn(2, 7, 24, requires_grad=True)
    spans = {"batch": torch.tensor([0, 0, 1]), "inject": torch.tensor([2, 5, 3]), "entry": torch.tensor([0, 3, 5])}
    target.current = spans
    target.capture(hidden)
    assert target.captured["states"].shape == (3, 24) and target.captured["entries"].tolist() == [0, 3, 5]
    parts = target.losses()
    assert target.captured is None and parts["occurrences"] == 3 and parts["edges"] == 9
    assert 0.0 <= float(parts["write"]) <= 2.0 and abs(float(parts["read"]) - np.log(40)) < 1.5     # ≈ chance at initialization
    (parts["write"] + parts["read"]).backward()
    assert target.write.weight.grad is not None and hidden.grad is not None and hidden.grad[0, 2].abs().sum() > 0
    assert hidden.grad[0, 0].abs().sum() == 0                                           # only the linked positions
    assert store.atomics.grad is None and store.transform.roles.grad is None             # the store is the target (detached)
    assert all(p.requires_grad for p in store.parameters())                              # flags restored
    # Without detaching, the store would learn from the losses too.
    loose = HRRTarget(store, 24, layer=1, write_weight=1.0, read_weight=1.0, detach_target=False)
    loose.current = spans; loose.capture(torch.randn(2, 7, 24))
    p = loose.losses(); (p["write"] + p["read"]).backward()
    assert store.atomics.grad is not None
    # No linked span: zero losses, nothing to count.
    target.current = {"batch": torch.zeros(0, dtype=torch.long), "inject": torch.zeros(0, dtype=torch.long), "entry": torch.zeros(0, dtype=torch.long)}
    target.capture(torch.randn(1, 4, 24))
    assert target.losses()["edges"] == 0


def test_a_perfect_write_reads_every_planted_filler() -> None:
    store = _store(atoms=40, dimension=128, concepts=8, degree=3, seed=1)
    target = HRRTarget(store, 128, layer=1, read_weight=1.0)
    with torch.no_grad():
        target.write.weight.copy_(torch.eye(128))                                        # W = identity: W h = the bundle
    entries = torch.arange(8)
    states = target.bundles(entries)                                                     # h carries the store exactly
    frames = [list(zip(store.schedule.relations[store.schedule.offsets[e]:store.schedule.offsets[e + 1]].tolist(),
                       store.schedule.fillers[store.schedule.offsets[e]:store.schedule.offsets[e + 1]].tolist())) for e in range(8)]
    result = target.read_frames(states, frames)
    assert result["filler"].numel() == 24 and bool((result["top1"] == result["filler"]).all()) and bool((result["rank"] == 1).all())
    summary = e15_eval.summarize_edges(result, 8, resamples=50, seed=0)
    assert summary["accuracy"] == 1.0 and summary["mrr"] == 1.0 and summary["concept_accuracy"] == 1.0
    # The read loss of a perfect write is small, and the write loss is zero.
    target.write_weight = 1.0
    parts = target._losses(states, entries)
    assert float(parts["write"]) < 1e-5 and float(parts["read"]) < 1.0
    # A random map reads at chance.
    torch.manual_seed(3)
    noise = target.read_frames(torch.randn(8, 128), frames)
    assert float((noise["top1"] == noise["filler"]).float().mean()) < 0.3


def test_the_ontology_decoder_is_frozen_random_and_seeded() -> None:
    ontology = {"offsets": torch.tensor([0, 2, 4]), "relations": torch.tensor([0, 1, 0, 1]), "fillers": torch.tensor([0, 1, 2, 3]),
                "atomic_count": 5, "relation_count": 2}
    a, b, c = (ontology_decoder(ontology, 16, seed=s) for s in (1, 1, 2))
    assert not any(p.requires_grad for p in a.parameters())
    torch.testing.assert_close(a.atomics, b.atomics)
    assert not torch.allclose(a.atomics, c.atomics)


# -- the compositional split check -------------------------------------------------------------------------------------------

def test_compositional_split_check_flags_atoms_edges_and_frames() -> None:
    train = [{(0, 1), (1, 2)}, {(0, 1), (1, 3)}, {(0, 4), (1, 2)}]
    checks = e15_eval.split_check([{(0, 1), (1, 2)}, {(0, 4), (1, 3)}, {(0, 1), (1, 9)}, {(0, 2), (1, 3)}], train)
    assert [c["frame_new"] for c in checks] == [False, True, True, True]
    assert [c["atoms_seen"] for c in checks] == [True, True, False, True]        # 9 never a filler; 2 is a filler (of role 1)
    assert [c["edges_seen"] for c in checks] == [True, True, False, False]       # (0, 2) is a new combination of known atoms
    ontology = {"entry_count": 4, "atomic_count": 10, "relation_count": 2, "offsets": torch.tensor([0, 2, 4, 6, 8]),
                "relations": torch.tensor([0, 1] * 4), "fillers": torch.tensor([1, 2, 1, 3, 4, 2, 5, 6]), "heldout_entries": [3],
                "train_frequency": [12, 30, 3, 0], "concept_names": ["a", "b", "c", "d"], "relation_names": ["is_a", "area"],
                "atomic_names": [f"x{i}" for i in range(10)]}
    new = [{"concept": "n0", "surface": "foo", "frame": [["is_a", "x4"], ["area", "x3"]]},      # recombination of seen edges
           {"concept": "n1", "surface": "bar", "frame": [["is_a", "x1"], ["area", "x9"]]}]      # x9 unseen
    splits = e15_eval.build_splits(ontology, new, seen_count=5, seed=0)
    s = splits["splits"]
    assert [r["entry"] for r in s["seen"]] == [0, 1] and [r["concept"] for r in s["compositional"]] == ["n0"]
    assert [r["concept"] for r in s["compositional_strict"]] == ["n0"]
    assert {r.get("concept", r.get("entry")) for r in s["unseen_filler"]} == {"n1", 3}      # entry 3's filler 6 never trained
    assert splits["stats"]["new_all_atoms_seen"] == 1 and len(s["heldout_ontology"]) == 1


def test_the_committed_t5_splits_match_their_hash_and_sizes() -> None:
    folder = REPO / "experiments" / "e15-hrr-target" / "splits" / "t5-v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert e15_eval.sha256_file(folder / "splits.json") == manifest["sha256"]
    splits = e15_eval.load_splits(folder / "splits.json")
    assert {k: len(v) for k, v in splits["splits"].items()} == manifest["sizes"]
    assert manifest["sizes"]["compositional"] == 285 and manifest["sizes"]["compositional_strict"] == 264
    assert all(r["atoms_seen"] and r["frame_new"] for r in splits["splits"]["compositional"])
    assert all(not r["atoms_seen"] for r in splits["splits"]["unseen_filler"])


# -- configs -----------------------------------------------------------------------------------------------------------------

def test_the_twelve_configs_differ_from_their_e9_base_only_by_the_target_block() -> None:
    folder = REPO / "experiments" / "e15-hrr-target" / "configs" / "t5"
    base_folder = REPO / "experiments" / "e9-retrofit" / "configs" / "t5"
    names = sorted(p.name for p in folder.glob("*.yaml"))
    assert len(names) == 12
    for name in names:
        config = yaml.safe_load((folder / name).read_text())
        arm, base = config["e15_arm"], config["e15_base_arm"]
        reference = yaml.safe_load((base_folder / name.replace(arm, base)).read_text())
        target = config["channel"].pop("target")
        for key in ("experiment", "e15_arm", "e15_base_arm"):
            config.pop(key)
        reference.pop("experiment")
        assert config == reference, name
        assert target["layer"] == "third" and target["tau"] == 0.1 and target["detach_target"] is True
        assert (target["write_weight"] > 0) == arm.endswith("WR") and target["read_weight"] > 0
        resolved = lm.resolve_config(yaml.safe_load((folder / name).read_text()))
        assert resolved["channel"]["target"] == target
    weights = {yaml.safe_load((folder / n).read_text())["e15_arm"]: yaml.safe_load((folder / n).read_text())["channel"]["target"] for n in names}
    assert weights["C5WR"]["read_weight"] == weights["C5R"]["read_weight"] == weights["C2R"]["read_weight"] == weights["C5shWR"]["read_weight"]
    assert weights["C5WR"]["write_weight"] == weights["C5shWR"]["write_weight"] and weights["C2R"]["dimension"] == 256


# -- training integration on a tiny from-scratch host ------------------------------------------------------------------------

TERMS = ["hydroxychloroquine", "new york", "acetaminophen", "zorbital flux", "quantum ledger"]
ATOMS = ["type:drug", "type:place", "type:tool", "area:health", "area:travel", "area:finance", "status:active", "status:pilot"]
RELATIONS = ["is_a", "area", "status"]
FRAMES = {"hydroxychloroquine": [(0, 0), (1, 3), (2, 6)], "new york": [(0, 1), (1, 4), (2, 6)], "acetaminophen": [(0, 0), (1, 3), (2, 7)],
          "zorbital flux": [(0, 2), (1, 5), (2, 7)], "quantum ledger": [(0, 2), (1, 5), (2, 6)]}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("e15")
    full = AliasTable.from_pairs([(t, i) for i, t in enumerate(TERMS)], holdout=[2], include_holdout=True)
    texts = [f"Doc {i}: the hydroxychloroquine note and the new york memo; the zorbital flux tool and the quantum ledger report "
             f"and acetaminophen too." for i in range(80)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=60_000,
                 batch_texts=8, workers=1)
    build_corpus(texts[:20], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=60_000, batch_texts=8, workers=1)
    schedule = full.entry_schedule([FRAMES[t] for t in TERMS])                 # entries are the table's, not TERMS' order
    frequency = np.bincount(TokenCorpus.open(root / "train").spans["entry"], minlength=len(TERMS))
    ontology = {"entry_count": len(TERMS), "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": TERMS, "relation_names": RELATIONS, "atomic_names": ATOMS}   # indexed by concept (E11)
    torch.save(ontology, root / "ontology.pt")
    cp.save_alias_table(full, root / "alias_table.json")
    return {"root": root, "ontology": ontology}


def _config(root: Path, mode: str, target: dict | None, **channel_extra) -> dict:
    channel = {"mode": mode, "dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel_extra}
    if target is not None:
        channel["target"] = target
    return {"seed": 0, "device": "cpu", "model": {"size": "tiny", "seq_len": 48},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 48 * 2 * 5, "lr": 3e-3, "warmup_tokens": 96, "log_every": 1},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
            "eval": {"windows": 4, "batch": 2, "first_tokens": 96, "save_window_losses": True}, "channel": channel}


TARGET = {"layer": "third", "tau": 0.1, "write_weight": 0.5, "read_weight": 0.25}


@pytest.mark.parametrize("mode,extra", [("compose", {}), ("free", {"free_dimension": 6})])
def test_training_with_the_target_logs_its_losses_saves_w_and_reloads(world, tmp_path: Path, mode: str, extra: dict) -> None:
    target = {**TARGET, **({"write_weight": 0.0, "dimension": 16} if mode == "free" else {})}
    folder = tmp_path / mode
    assert lm.train(_config(world["root"], mode, target, **extra), folder)["steps"] == 5
    rows = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    assert rows and all({"lm_loss", "write_loss", "read_loss"} <= set(r) for r in rows)
    assert all(r["read_loss"] > 0 for r in rows) and all(np.isfinite(r["loss"]) for r in rows)
    first = rows[0]
    assert first["loss"] == pytest.approx(first["lm_loss"] + target["write_weight"] * first["write_loss"] + target["read_weight"] * first["read_loss"], rel=1e-4)
    state = torch.load(folder / "final.pt", weights_only=False)["model"]
    assert "channel.target.write.weight" in state and state["channel.target.write.weight"].shape == (16, 64)
    if mode == "free":
        assert "channel.target.decoder_module.atomics" in state                     # the frozen code travels with the run
    model = lm.load_final(folder / "final.pt")
    assert isinstance(model.channel.target, HRRTarget) and model.channel.target.layer == 1 and model._target_hook is not None
    torch.testing.assert_close(model.channel.target.write.weight, state["channel.target.write.weight"])
    if mode == "compose":
        assert model.channel.target.decoder is model.channel.composer
    else:
        assert not any(p.requires_grad for p in model.channel.target.decoder.parameters())


def test_a_config_without_the_block_is_unchanged(world, tmp_path: Path) -> None:
    """Regression: no target attribute, no extra metric keys, identical initialization and identical LM loss."""
    plain = lm.resolve_config(_config(world["root"], "compose", None))
    with_target = lm.resolve_config(_config(world["root"], "compose", TARGET))
    ontology = torch.load(world["root"] / "ontology.pt", weights_only=False)

    def build(config):
        torch.manual_seed(config["seed"])
        base = lm.build_model(config)
        channel, context = lm.build_channel(config, ontology, base.get_input_embeddings().weight.shape[1], host=base)
        channel.set_unseen(ontology["heldout_entries"])
        return lm.wrap_host(config, base, channel, context)

    a, b = build(plain), build(with_target)
    assert not hasattr(a.channel, "target") and a._target_hook is None and isinstance(b.channel.target, HRRTarget)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"channel.target.write.weight"}
    for key, value in sa.items():
        torch.testing.assert_close(value, sb[key], msg=key)                             # the target draws from its own generator
    corpus = TokenCorpus.open(world["root"] / "train")
    from vsa_embed.data.corpus import sample_batch
    ids, spans = sample_batch(corpus, seed=1234, step=0, micro_step=0, batch=2, length=48, min_subtokens=1)
    a.eval(); b.eval()                                                                   # same dropout-free forward
    with torch.no_grad():
        la, lb = a(ids, spans=spans, labels=ids)["loss"], b(ids, spans=spans, labels=ids)["loss"]
    torch.testing.assert_close(la, lb)                                                   # the hook changes nothing in the forward
    assert b.channel.target.captured is not None and b.channel.target.captured["entries"].numel() == spans["entry"].numel()
    folder = tmp_path / "plain"
    lm.train(_config(world["root"], "compose", None), folder)
    rows = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    assert rows and not any("write_loss" in r or "lm_loss" in r for r in rows)


def test_read_accuracy_runs_on_a_toy_run_and_the_probe_is_fitted(world, tmp_path: Path) -> None:
    from vsa_embed.experiments import e11_read_to_learn as e11
    from vsa_embed.experiments.e5_common import open_run
    folder = tmp_path / "c5wr"
    lm.train(_config(world["root"], "compose", TARGET), folder)
    run = open_run(folder, device="cpu", batch_size=4, alias_table=world["root"] / "alias_table.json")
    new = [{"concept": "e15n-0", "surface": "glimmerstack", "frame": [["is_a", "type:tool"], ["area", "area:health"], ["status", "status:pilot"]]},
           {"concept": "e15n-1", "surface": "voltrane", "frame": [["is_a", "type:place"], ["area", "area:finance"], ["status", "status:active"]]}]
    read_set = e11.ReadSet(tmp_path, {"kind": "new", "track": "toy", "styles": ["prose"]}, new, [], [])
    splits = e15_eval.build_splits(world["ontology"], new, seen_count=3, seed=0)
    assert [r["concept"] for r in splits["splits"]["compositional"]] == ["e15n-0", "e15n-1"]
    result = e15_eval.read_accuracy(run, splits, read_set, resamples=20, log=lambda m: None)
    groups = result["groups"]
    assert {"seen", "compositional", "compositional_strict", "heldout_ontology"} <= set(groups) and result["trained_map"]
    for name, g in groups.items():
        assert g["linked_share"] == 1.0, name                                            # every term was injected where it is read
        assert 0.0 <= g["trained"]["accuracy"] <= 1.0 and g["trained"]["edges"] == 3 * g["concepts"]
        assert 0.0 <= g["probe"]["accuracy"] <= 1.0 and "typed_accuracy" in g["trained"]
    assert result["chance"] == pytest.approx(1 / len(ATOMS)) and result["probe"]["rows"] == 3
    text = e15_eval.render_read({"source": run.describe(), **result}, smoke=True)
    assert "SMOKE" in text and "compositional" in text
