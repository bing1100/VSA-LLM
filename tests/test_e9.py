"""E9 (retrofit × quantization) on tiny models: the probe loader for trainable-only checkpoints, quantized
probe / zero-shot / editing evaluations, channel surgery (inserted entries, edits), the dimension-3 item
builders and evaluation, the plan with job chaining, and the R9 report (on runs and on synthetic inputs)."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.evaluation import channel_probes as cp
from vsa_embed.experiments import e5_common as common
from vsa_embed.experiments import e5_zeroshot as zs
from vsa_embed.experiments import e9_ontology_edit as edit
from vsa_embed.experiments import e9_plan, e9_report
from vsa_embed.span_channel import AliasTable, SpanChannel
import vsa_embed.training.lm as lm


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        from nltk.corpus import wordnet
        wordnet.synsets("bank")
        return True
    except (OSError, LookupError):
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")

CONCEPTS = ["cat.n.01", "dog.n.01", "wolf.n.01", "hammer.n.02", "bucket.n.01", "car.n.01", "bicycle.n.01", "bank.n.01",
            "depository_financial_institution.n.01", "aardvark.n.01", "acetaminophen.n.01", "sprint.v.01", "jar.n.01",
            "truck.n.01", "kettle.n.01"]
ALIASES = [("cat", 0), ("house cat", 0), ("dog", 1), ("domestic dog", 1), ("wolf", 2), ("timber wolf", 2), ("hammer", 3),
           ("claw hammer", 3), ("bucket", 4), ("pail", 4), ("car", 5), ("motorcar", 5), ("bicycle", 6), ("bike", 6),
           ("bank", 7), ("river bank", 7), ("banking company", 8), ("aardvark", 9), ("ant bear", 9), ("acetaminophen", 10),
           ("paracetamol", 10), ("sprint", 11), ("jar", 12), ("glass jar", 12), ("truck", 13), ("motortruck", 13),
           ("kettle", 14), ("tea kettle", 14)]
HELDOUT = [9, 10, 11]
RELATIONS = ["hypernym", "part_meronym", "part_holonym", "lexname", "pos"]
ATOMS = ["synset:feline.n.01", "synset:canine.n.02", "synset:mammal.n.01", "synset:hand_tool.n.01", "synset:container.n.01",
         "synset:vehicle.n.01", "synset:tail.n.01", "synset:paw.n.01", "synset:handle.n.01", "synset:wheel.n.01",
         "synset:lid.n.01", "synset:engine.n.01", "synset:stream.n.01", "synset:financial_institution.n.01",
         "synset:slope.n.01", "synset:analgesic.n.01", "synset:run.v.01", "lexname:noun.animal", "lexname:noun.artifact",
         "lexname:noun.object", "lexname:noun.group", "lexname:verb.motion", "pos:n", "pos:v"]
A = {name: i for i, name in enumerate(ATOMS)}
R = {name: i for i, name in enumerate(RELATIONS)}


def _frame(hypernym, *parts, lexname="noun.artifact", pos="n", holonym=None):
    edges = [("hypernym", f"synset:{hypernym}")] + [("part_meronym", f"synset:{p}") for p in parts]
    if holonym:
        edges.append(("part_holonym", f"synset:{holonym}"))
    edges += [("lexname", f"lexname:{lexname}"), ("pos", f"pos:{pos}")]
    return [(R[r], A[a]) for r, a in edges]


FRAMES = [
    _frame("feline.n.01", "tail.n.01", "paw.n.01", lexname="noun.animal"),
    _frame("canine.n.02", "tail.n.01", "paw.n.01", lexname="noun.animal"),
    _frame("canine.n.02", "tail.n.01", lexname="noun.animal"),
    _frame("hand_tool.n.01", "handle.n.01"),
    _frame("container.n.01", "handle.n.01", "lid.n.01"),
    _frame("vehicle.n.01", "wheel.n.01", "engine.n.01"),
    _frame("vehicle.n.01", "wheel.n.01", "handle.n.01"),
    _frame("slope.n.01", lexname="noun.object", holonym="stream.n.01"),
    _frame("financial_institution.n.01", lexname="noun.group"),
    _frame("mammal.n.01", "tail.n.01", lexname="noun.animal"),
    _frame("analgesic.n.01"),
    _frame("run.v.01", lexname="verb.motion", pos="v"),
    _frame("container.n.01", "lid.n.01"),
    _frame("vehicle.n.01", "wheel.n.01", "engine.n.01", "lid.n.01"),
    _frame("container.n.01", "handle.n.01", "lid.n.01", holonym="stream.n.01"),
]
SENTENCES = ["The house cat chased the domestic dog to the river bank.", "She put money in the banking company and bought a claw hammer.",
             "An ant bear ate ants while the timber wolf slept.", "Take paracetamol if you sprint too far.",
             "The motorcar and the motortruck passed the bike.", "A glass jar and a tea kettle stood near the pail.",
             "The timber wolf and the domestic dog met near the river bank.", "He used a claw hammer on the motorcar.",
             "Paracetamol is also called acetaminophen.", "The ant bear dug near the river bank with a bike."]
SEQ = 32


def fake_host(config) -> torch.nn.Module:
    """A deterministic stand-in for a pretrained host (same weights at every call)."""
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=1, n_head=2))


def _config(root: Path, name: str, channel: dict, host_mode: str, **train) -> dict:
    return {"seed": 1, "device": "cpu", "experiment": f"e9-test-fake-{name}-s1",
            "model": {"size": "pretrained", "seq_len": SEQ, "pretrained": "fake-host", "host_mode": host_mode, "lora_rank": 2},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": SEQ * 2 * 6, "lr": 3e-3, "warmup_tokens": 64, "log_every": 2,
                      "save_trainable_only": True, "host_lr": 1e-3, **train},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
            "eval": {"windows": 6, "batch": 3, "first_tokens": 128, "save_window_losses": True},
            "channel": {"dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    patch = pytest.MonkeyPatch()
    patch.setattr(lm, "build_model", fake_host)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("e9")
    full = AliasTable.from_pairs(ALIASES, holdout=HELDOUT, include_holdout=True)
    rng = random.Random(0)
    texts = [f"Doc {i}: " + " ".join(rng.sample(SENTENCES, 5)) for i in range(80)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=80_000,
                 batch_texts=8, workers=2)
    build_corpus(texts[:40], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=80_000, batch_texts=8,
                 workers=2)
    schedule = full.entry_schedule(FRAMES)
    entries = len(full.entry_concepts)
    frequency = np.bincount(TokenCorpus.open(root / "train").spans["entry"], minlength=entries)
    ontology = {"entry_count": entries, "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": CONCEPTS, "relation_names": RELATIONS, "atomic_names": ATOMS}
    torch.save(ontology, root / "ontology.pt")
    cp.save_alias_table(full, root / "alias_table.json")
    runs = {}
    specs = {"P0": ({"mode": "none"}, "frozen", {"eval_only": True}), "C0p": ({"mode": "none"}, "lora", {}),
             "C2": ({"mode": "free", "free_dimension": 8}, "lora", {}),
             "C5": ({"mode": "compose", "composition": "attentive", "context_window": 4}, "lora", {})}
    for name, (channel, host_mode, train) in specs.items():
        label = "frozen" if host_mode == "frozen" else "lora"
        folder = root / "runs" / f"fake-{label}-{name}-s1"
        lm.train(_config(root, f"{label}-{name}", channel, host_mode, **train), folder)
        runs[name] = folder
    probes = root / "probes"
    probes.mkdir()
    pairs = [("cat", "dog", 3.0), ("wolf", "dog", 3.5), ("bank", "car", 1.0), ("truck", "car", 3.2), ("hammer", "cat", 0.2),
             ("aardvark", "wolf", 1.0), ("kettle", "jar", 2.5), ("bike", "motorcar", 2.0)]
    (probes / "card660.tsv").write_text("".join(f"{a}\t{b}\t{v}\n" for a, b, v in pairs))
    yield {"root": root, "runs": runs, "table": full, "ontology": ontology, "probes": probes}
    patch.undo()


@pytest.fixture(scope="module")
def items(world, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e9-items")
    reserved = root / "reserved"
    zs.SCENARIOS["c3_synthetic"](ontology_path=world["root"] / "ontology.pt", out_dir=reserved, tokenizer_name="gpt2",
                                 min_subtokens=1, max_concepts=10, seed=0, contamination_texts=SENTENCES)
    new = edit.build_new_word_items(world["root"] / "ontology.pt", root / "new", tokenizer_name="gpt2", count=6, min_subtokens=1,
                                    contamination_texts=SENTENCES, reserved_names=edit._reserved_names([reserved]))
    edits = edit.build_edit_items(world["root"] / "ontology.pt", root / "edits", tokenizer_name="gpt2", count=6, min_subtokens=1,
                                  neighbors=2)
    return {"new": root / "new", "edits": root / "edits", "reserved": reserved, "new_manifest": new, "edit_manifest": edits}


# -- 1. probe loader for trainable-only checkpoints ------------------------------------------------------------------

def test_probe_loader_reads_trainable_only_checkpoints(world) -> None:
    final = torch.load(world["runs"]["C5"] / "final.pt", weights_only=False)
    assert final["trainable_only"] and not any(k.startswith("model.") and "lora" not in k for k in final["model"])
    adapter = cp.load_run(world["runs"]["C5"], device="cpu")
    assert adapter.info["trainable_only_checkpoint"] and "quantization" not in adapter.info
    reference = lm.load_final(world["runs"]["C5"] / "final.pt").state_dict()
    for key, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, reference[key])
    torch.testing.assert_close(adapter.model.state_dict()["model.transformer.wte.weight"], fake_host(None).state_dict()["transformer.wte.weight"])
    p0 = cp.load_run(world["runs"]["P0"], device="cpu")              # evaluation-only C0' on a frozen host: no state at all
    assert p0.model.channel is None and p0.info["trainable_only_checkpoint"]
    plain = cp.load_run(world["runs"]["C2"], device="cpu", batch_size=4)
    assert plain.word_state(["The word bike"], [(9, 13)]).shape == (1, 32)


# -- 2. quantized evaluation --------------------------------------------------------------------------------------------

def test_quantized_probes_record_the_variant_and_change_outputs(world, tmp_path) -> None:
    base = ["--run", str(world["runs"]["C5"]), "--probes", "card660", "--probes-root", str(world["probes"]), "--device", "cpu",
            "--batch-size", "4"]
    cp.main([*base, "--output", str(tmp_path / "bf16.json")])
    cp.main([*base, "--quantize", "int8", "--output", str(tmp_path / "int8.json")])
    cp.main([*base, "--quantize", "int8", "--quantize-channel", "--output", str(tmp_path / "int8b.json")])
    documents = {n: json.loads((tmp_path / f"{n}.json").read_text()) for n in ("bf16", "int8", "int8b")}
    assert "quantization" not in documents["bf16"]["model"]
    assert documents["int8"]["model"]["quantization"]["variant"] == "int8-A"
    assert documents["int8b"]["model"]["quantization"]["variant"] == "int8-B"
    assert documents["int8b"]["model"]["quantization"]["channel_tensors"]
    assert documents["int8"]["settings"] == documents["bf16"]["settings"]           # so INT8 − bf16 pairs by item
    damage = cp.compare_outputs(tmp_path / "int8.json", tmp_path / "bf16.json", resamples=50)
    assert damage["tables"]["card660"]["spearman_tied"]["all"]["n"] == 8
    a, b = (cp.load_items(tmp_path / f"{n}.json")["card660"]["cosine"] for n in ("bf16", "int8"))
    assert a != b and np.allclose(a, b, atol=0.05)
    with pytest.raises(RuntimeError, match="INT4"):
        cp.load_run(world["runs"]["C5"], device="cpu", quantize="int4")
    with pytest.raises(SystemExit):
        cp.main(["--host", "gpt2", "--quantize", "int8", "--output", str(tmp_path / "x.json")])


def test_quantized_zero_shot_evaluation(world, items, tmp_path) -> None:
    out = tmp_path / "zs-int8"
    zs.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(items["reserved"]), "--output", str(out),
             "--sources", "own,none,mean_row", "--resamples", "50", "--device", "cpu", "--quantize", "int8"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["source"]["quantization"]["variant"] == "int8-A"
    config = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert config["quantize"] == "int8" and "int8-A" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        zs.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(items["reserved"]), "--output", str(out),
                 "--sources", "own", "--resamples", "50", "--device", "cpu"])
    zs.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(items["reserved"]), "--output", str(out),
             "--sources", "own", "--resamples", "50", "--device", "cpu", "--overwrite"])
    assert "quantize" not in yaml.safe_load((out / "resolved_config.yaml").read_text())


# -- 3. channel surgery ---------------------------------------------------------------------------------------------------

def _composer(schedule: FrameSchedule, **kw) -> FrameComposer:
    torch.manual_seed(0)
    return FrameComposer(schedule, 6, 3, 16, mode="attentive", key_dimension=4, context_dimension=4, **kw)


FRAMES_SMALL = [[(0, 1), (1, 2)], [(2, 3)], [(0, 4), (2, 5), (1, 1)]]
NEW_FRAMES = [[(1, 0), (2, 2)], [(0, 5)]]


@pytest.mark.parametrize("factor", ["induced", "hybrid"])
def test_inserted_concepts_change_rows_only_for_their_entries(factor) -> None:
    schedule = FrameSchedule.from_frames(FRAMES_SMALL)
    composer = _composer(schedule, concept_factor=factor)
    if factor == "hybrid":
        with torch.no_grad():
            composer.delta.normal_()
    channel = SpanChannel(composer, 8, entry_count=3)
    context = torch.randn(3, 4)
    old = torch.arange(3)
    before = channel.rows({"entry": old}, context=context).detach()
    with edit.inserted_entries(channel, 2, NEW_FRAMES) as ids:
        assert ids.tolist() == [3, 4] and channel.entry_count == 5
        after = channel.rows({"entry": old}, context=context).detach()
        new_rows = channel.rows({"entry": ids}, context=context[:2]).detach()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        reference = _composer(FrameSchedule.from_frames(FRAMES_SMALL + NEW_FRAMES), concept_factor=factor)
        reference.load_state_dict({k: v for k, v in composer.state_dict().items()}, strict=False)
        if factor == "hybrid":
            assert torch.equal(composer.delta[3:], torch.zeros(2, 4))      # never-trained concepts have δ = 0
        expected = channel.projector(reference.compose(ids, context[:2])).detach()
        torch.testing.assert_close(new_rows, expected)
    assert channel.entry_count == 3 and composer.schedule.concept_count == 3
    torch.testing.assert_close(channel.rows({"entry": old}, context=context).detach(), before, rtol=0, atol=0)
    free = _composer(schedule, concept_factor="free")
    with pytest.raises(ValueError, match="free concept factor"):
        free.add_concepts(NEW_FRAMES)


def test_free_and_random_channels_grow_without_touching_old_rows() -> None:
    torch.manual_seed(0)
    free = SpanChannel(None, 8, entry_count=4, mode="free", free_dimension=3)
    free.set_unseen([3])
    old = torch.arange(4)
    before = free.rows({"entry": old}).detach()
    with edit.inserted_entries(free, 2) as ids:
        torch.testing.assert_close(free.rows({"entry": old}).detach(), before, rtol=0, atol=0)
        fallback = free.rows({"entry": ids}).detach()
        torch.testing.assert_close(fallback[0], before[3])                 # an unseen entry gets the C2 fallback row
    assert free.table.weight.shape[0] == 4 and free.unseen.numel() == 4
    rand = SpanChannel(None, 8, entry_count=4, mode="random")
    before = rand.rows({"entry": old}).detach()
    with edit.inserted_entries(rand, 3, seed=5) as ids:
        torch.testing.assert_close(rand.rows({"entry": old}).detach(), before, rtol=0, atol=0)
        assert rand.rows({"entry": ids}).shape == (3, 8)
    assert rand.table_fixed.shape[0] == 4
    with pytest.raises(ValueError):
        SpanChannel(_composer(FrameSchedule.from_frames(FRAMES_SMALL)), 8, entry_count=3).add_entries(1)


def test_edits_change_only_the_edited_entry_row() -> None:
    composer = _composer(FrameSchedule.from_frames(FRAMES_SMALL))
    channel = SpanChannel(composer, 8, entry_count=3)
    entries = torch.arange(3)
    context = torch.randn(3, 4)
    before = channel.rows({"entry": entries}, context=context).detach()
    with edit.applied_edits(channel, [(2, 2, 5, 3)]) as applied:
        assert applied
        after = channel.rows({"entry": entries}, context=context).detach()
        assert composer.schedule.fillers.tolist() == [1, 2, 3, 4, 3, 1]
    torch.testing.assert_close(after[:2], before[:2], rtol=0, atol=0)
    assert not torch.allclose(after[2], before[2])
    torch.testing.assert_close(channel.rows({"entry": entries}, context=context).detach(), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no edge"):
        edit.edited_schedule(composer.schedule, [(0, 2, 5, 3)])
    with edit.applied_edits(SpanChannel(None, 8, entry_count=3, mode="free"), [(2, 2, 5, 3)]) as applied:
        assert not applied


# -- 4. dimension 3 items and evaluation ---------------------------------------------------------------------------

def test_new_word_items_are_new_combinations(world, items) -> None:
    from nltk.corpus import wordnet
    manifest, concepts, prompts = edit.load_item_dir(items["new"], edit.SCHEMA_NEW)
    assert manifest["counts"]["concepts"] == 6 and manifest["checks"]["reserved_clashes"] >= 0
    view = edit.OntologyView(world["ontology"], world["table"])
    index = view.edge_index()
    lemmas = {l.lower() for s in wordnet.all_synsets() for l in s.lemma_names()}
    reserved = edit._reserved_names([items["reserved"]])
    alias_words = {w for a in world["table"].alias_to_entry for w in a.split()}
    for c in concepts:
        assert c["surface"] not in lemmas and c["surface"] not in reserved and c["surface"] not in alias_words
        assert not any(c["surface"] in s.lower() for s in SENTENCES)
        frame = edit.resolve_frame(c["frame"], view.relation_id, view.atomic_id)
        donor = view.frame(c["donor_entry"])
        assert len(frame) == c["degree"] == len(donor) and [r for r, _ in frame] == [r for r, _ in donor]
        assert c["donor_entry"] not in view.heldout and c["resampled"]
        category = (view.relation_id["hypernym"], view.atomic_id[c["category"]])
        for r, f in edit.resolve_frame(c["resampled"], view.relation_id, view.atomic_id):
            assert not (index[category] & index.get((r, f), set()))          # the combination exists nowhere
        random_frame = edit.resolve_frame(c["random_frame"], view.relation_id, view.atomic_id)
        assert [r for r, _ in random_frame] == [r for r, _ in frame]
    tests = {p["test"] for p in prompts}
    assert tests == {"property", "statement", "entailment"}
    for p in prompts:
        assert 0 <= p["gold"] < len(p["candidates"]) and len(p["templates"]) == 2
        if p["test"] == "statement":
            assert not set(p["templates"]) & {t for ts in zs.TEMPLATES.values() for t in ts}
    assert any(p.get("edge_kind") == "resampled" for p in prompts if p["test"] == "property")


def test_edit_items_pick_plausible_alternatives(world, items) -> None:
    from nltk.corpus import wordnet
    manifest, concepts, prompts = edit.load_item_dir(items["edits"], edit.SCHEMA_EDITS)
    edited = [c for c in concepts if c["role"] == "edited"]
    neighbours = {c["concept"]: c for c in concepts if c["role"] == "neighbour"}
    assert edited and {c["status"] for c in edited} & {"heldout"} and set(manifest["edited_status"]) >= {"heldout"}
    for c in edited:
        new, old = (wordnet.synset(c[k].partition(":")[2]) for k in ("new", "old"))
        assert new.lexname() == old.lexname() and c["new"] != c["old"] != c["control"] != c["new"]
        assert new.name() not in zs._ancestors(wordnet.synset(c["source_concept"]))
        assert old.name() not in zs._ancestors(new)
        assert all(neighbours[n]["entry"] not in {e["entry"] for e in edited} for n in c["neighbours"])
    by_test = {t: [p for p in prompts if p["test"] == t] for t in ("efficacy", "paraphrase", "neighbourhood")}
    assert len(by_test["efficacy"]) == len(by_test["paraphrase"]) == len(edited) and by_test["neighbourhood"]
    assert all(len(p["candidates"]) == 2 for p in prompts)


def test_new_word_evaluation_sources_and_invariants(world, items) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    before = common.entry_rows(run.channel, torch.arange(run.channel.entry_count))
    evaluation = edit.evaluate_new_words(run, items["new"], fit_entries=20, log=lambda m: None)
    assert evaluation["sources"] == ["own", "none", "mean_row", "random_frame"]
    assert all(r["status"] == "linked" for r in evaluation["resolved"].values())
    torch.testing.assert_close(common.entry_rows(run.channel, torch.arange(run.channel.entry_count)), before, rtol=0, atol=0)
    assert run.channel.entry_count == world["ontology"]["entry_count"]
    own, none, random_frame = (evaluation["results"][s] for s in ("own", "none", "random_frame"))
    flat = lambda rows: [v for r in rows["prompts"] for t in r["pmi"] for v in t]
    assert flat(own) != flat(none) and flat(own) != flat(random_frame)
    # `none` (zero rows) is exactly "the names are not linked": the plain adapter does not know them.
    _, concepts, prompts = edit.load_item_dir(items["new"], edit.SCHEMA_NEW)
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    plain, _ = zs.score_prompts(run.adapter, [p for p in prompts if p["test"] in {"property", "entailment"}], surfaces)
    assert flat({"prompts": plain}) == pytest.approx(flat(none), abs=1e-4)
    summary = edit.summarize_new_words(evaluation, resamples=50)
    assert summary["linked_concepts"] == 6 and {c["baseline"] for c in summary["comparisons"]} == {"none", "mean_row", "random_frame"}
    c0 = edit.evaluate_new_words(common.open_run(world["runs"]["C0p"], device="cpu"), items["new"], log=lambda m: None)
    assert c0["sources"] == ["own"] and len(c0["results"]["own"]["statements"]) == len(own["statements"])
    c2 =edit.evaluate_new_words(common.open_run(world["runs"]["C2"], device="cpu"), items["new"], fit_entries=20, log=lambda m: None)
    assert c2["sources"] == ["own", "none", "mean_row"]


def test_edit_evaluation_moves_only_the_edited_concepts(world, items) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    evaluation = edit.evaluate_edits(run, items["edits"], log=lambda m: None)
    assert evaluation["edit_applicable"]
    _, concepts, prompts = edit.load_item_dir(items["edits"], edit.SCHEMA_EDITS)
    results = evaluation["results"]
    efficacy = [p["id"] for p in prompts if p["test"] == "efficacy"]
    neighbourhood = [p["id"] for p in prompts if p["test"] == "neighbourhood"]
    assert any(abs(results["after"][i]["d_mean"] - results["before"][i]["d_mean"]) > 1e-6 for i in efficacy)
    for i in neighbourhood:            # composed rows are independent: unedited concepts are untouched
        assert results["after"][i]["d"] == pytest.approx(results["before"][i]["d"], abs=1e-5)
    summary = edit.summarize_edits(evaluation, prompts, concepts, resamples=50)
    assert summary["subsets"]["all"]["neighbourhood"]["max_abs_change"] < 1e-4
    assert summary["subsets"]["all"]["efficacy"]["n"] == len(efficacy)
    c2 = edit.evaluate_edits(common.open_run(world["runs"]["C2"], device="cpu"), items["edits"], log=lambda m: None)
    assert not c2["edit_applicable"] and c2["results"]["after"] is c2["results"]["before"]


def test_ontology_edit_cli_writes_a_run_folder(world, items, tmp_path) -> None:
    out = tmp_path / "edit-int8"
    args = ["evaluate", "--run", str(world["runs"]["C5"]), "--new-items", str(items["new"]), "--edit-items", str(items["edits"]),
            "--output", str(out), "--device", "cpu", "--resamples", "50", "--fit-entries", "20"]
    edit.main([*args, "--quantize", "int8"])
    for name in ("summary.json", "report.md", "predictions.jsonl", "manifest.json", "resolved_config.yaml"):
        assert (out / name).is_file()
    document = json.loads((out / "summary.json").read_text())
    assert document["source"]["quantization"]["variant"] == "int8-A"
    assert set(document["new_words"]["summary"]["sources"]) == {"own", "none", "mean_row", "random_frame"}
    assert "## (a) New words" in (out / "report.md").read_text() and "## (b) Edited words" in (out / "report.md").read_text()
    with pytest.raises(FileExistsError):
        edit.main(args)
    edit.main([*args, "--overwrite"])
    assert "quantization" not in json.loads((out / "summary.json").read_text())["source"]


# -- 5. the plan --------------------------------------------------------------------------------------------------------

@pytest.fixture()
def plan_root(tmp_path: Path) -> dict:
    torch.save({"entry_count": 101500, "atomic_count": 8192, "relation_count": 16}, tmp_path / "counts.pt")
    return {"root": tmp_path / "e9", "counts": tmp_path / "counts.pt", "data": tmp_path / "smollm2"}


def _plan(plan_root: dict, **kw) -> list[Path]:
    return e9_plan.write_stage("main", hosts=kw.pop("hosts", list(e9_plan.E9_HOSTS)), models=kw.pop("models", list(e9_plan.MODELS)),
                               seeds=kw.pop("seeds", [1]), data_root=plan_root["data"], counts_ontology=plan_root["counts"],
                               root=plan_root["root"], track=kw.pop("track", "wordnet"), **kw)


def test_e9_plan_grid(plan_root) -> None:
    paths = _plan(plan_root, seeds=[1, 2])
    stems = sorted(p.stem for p in paths)
    assert len(paths) == 2 * (1 + 3 * 2)                                   # P0 once per host; C0p, C2, C5 × 2 seeds
    assert "SmolLM2-360M-frozen-P0-s1" in stems and "SmolLM2-360M-frozen-P0-s2" not in stems
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    for stem, c in configs.items():
        resolved = lm.resolve_config(c)
        assert resolved["eval"]["windows"] >= 1024 and resolved["eval"]["save_window_losses"]
        assert resolved["train"]["total_tokens"] == 50_000_000 and resolved["train"]["micro_batch"] * resolved["train"]["grad_accum"] == 64
        assert resolved["experiment"] == f"e9-main-{stem}" and resolved["data"]["ontology"] == str(plan_root["data"] / "ontology.pt")
    p0 = configs["SmolLM2-360M-frozen-P0-s1"]
    assert p0["model"]["host_mode"] == "frozen" and p0["train"]["eval_only"] and p0["channel"]["mode"] == "none"
    c0 = configs["SmolLM2-360M-full-C0p-s2"]
    assert c0["model"]["host_mode"] == "train" and c0["train"]["host_lr"] == 3e-5 and not c0["train"]["save_trainable_only"]
    c2, c5 = configs["SmolLM2-135M-full-C2-s1"]["channel"], configs["SmolLM2-360M-full-C5-s1"]["channel"]
    assert c2["mode"] == "free" and c2["free_dimension"] > 0 and c2["gate_bias"] == 0.0          # the engagement-check recipe
    assert (c5["mode"], c5["composition"], c5["context_window"], c5["gate_bias"]) == ("compose", "attentive", 8, 0.0)
    assert c0["train"]["lr"] == 1e-3 and configs["SmolLM2-360M-full-C5-s1"]["e9_track"] == "wordnet"
    lora = _plan(plan_root, mode="lora", gate_bias=0.0, lora_rank=64, tokens=10_000_000, models=["C5"], hosts=["SmolLM2-360M"])
    c = yaml.safe_load(lora[0].read_text())
    assert lora[0].stem == "SmolLM2-360M-lora-C5-s1" and c["model"]["lora_rank"] == 64 and c["train"]["host_lr"] == 2e-4
    assert c["train"]["save_trainable_only"] and c["channel"]["gate_bias"] == 0.0 and c["train"]["total_tokens"] == 10_000_000
    check_path = Path(__file__).resolve().parents[1] / "experiments/e4-small-lm/configs/e9-check/SmolLM2-360M-lora-C5@g0-s1.yaml"
    check = yaml.safe_load(check_path.read_text())
    for section in ("model", "channel"):
        assert {k: v for k, v in c[section].items()} == check[section]
    with pytest.raises(ValueError, match="1024"):
        _plan(plan_root, windows=512)


def test_e9_plan_chains_evaluations_after_training(plan_root, tmp_path) -> None:
    paths = _plan(plan_root, hosts=["SmolLM2-135M"])
    queue = tmp_path / "jobs"
    queued = e9_plan.queue_jobs(paths, "main", 22, root=plan_root["root"], queue_dir=queue, track="wordnet")
    assert len(queued) == 4 + 4 * 6 + 2
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    train = jobs["main-SmolLM2-135M-full-C5-s1"]
    from vsa_embed.experiments.cpt_plan import pinned_python     # this interpreter (the pinned one under the Qwen3.5 env)
    assert pinned_python() == sys.executable or "vsa-qwen35" in sys.prefix
    assert train["priority"] == 22 and train["command"][:3] == [pinned_python(), "-m", "vsa_embed.training.lm"]
    assert train["resume_args"] == ["--resume"] and train["env"] == {"PYTHONPATH": "src"}
    run_dir = str(plan_root["root"] / "runs" / "main" / "SmolLM2-135M-full-C5-s1")
    for suffix, module in (("probes", "vsa_embed.evaluation.channel_probes"), ("zeroshot-int4", "vsa_embed.experiments.e5_zeroshot"),
                           ("edit-int4", "vsa_embed.experiments.e9_ontology_edit")):
        job = jobs[f"main-SmolLM2-135M-full-C5-s1-{suffix}"]
        assert job["priority"] == 23 and job["command"][0] == pinned_python() and job["command"][2] == module
        assert run_dir in job["command"] and job["resume_args"] == ["--overwrite"]
        assert ("--quantize" in job["command"]) == suffix.endswith("int4")
    assert jobs["main-SmolLM2-135M-frozen-P0-s1-edit"]["priority"] == 23
    quant = jobs["main-quant-s1"]
    assert quant["priority"] == 24 and quant["command"][2] == "vsa_embed.experiments.e4_quant" and "--resume" in quant["command"]
    assert sum(1 for x in quant["command"] if x.endswith("-s1") and "runs" in x) == 4
    assert jobs["main-report-s1"]["priority"] == 25 and jobs["main-report-s1"]["command"][2] == "vsa_embed.experiments.e9_report"
    assert e9_plan.queue_jobs(paths, "main", 22, root=plan_root["root"], queue_dir=queue, track="wordnet") == []      # idempotent
    seeds = _plan(plan_root, hosts=["SmolLM2-135M"], seeds=[2, 3])
    again = e9_plan.queue_jobs(seeds, "main", 22, root=plan_root["root"], queue_dir=queue, int4_probes="card660,wic",
                               track="wordnet")
    assert "main-quant-s2-3" in again and not any("P0" in name for name in again)
    subset = json.loads((queue / "main-SmolLM2-135M-full-C5-s2-probes-int4.json").read_text())["command"]
    assert subset[subset.index("--probes") + 1] == "card660,wic"
    assert "--probes" not in json.loads((queue / "main-SmolLM2-135M-full-C5-s2-probes.json").read_text())["command"]


# -- 6. the R9 report ----------------------------------------------------------------------------------------------------

def test_r9_report_on_tiny_runs(world, items, tmp_path) -> None:
    from vsa_embed.experiments import e4_quant
    root = world["root"]
    for name, run in world["runs"].items():
        for quantize, suffix in ((None, ""), ("int8", "-int8")):
            output = run / f"probes{suffix}.json"
            if not output.exists():
                cp.main(["--run", str(run), "--probes", "card660", "--probes-root", str(world["probes"]), "--device", "cpu",
                         "--batch-size", "4", "--output", str(output), *(["--quantize", quantize] if quantize else [])])
            folder = run / f"edit{suffix}"
            if not folder.exists():
                edit.main(["evaluate", "--run", str(run), "--new-items", str(items["new"]), "--edit-items", str(items["edits"]),
                           "--output", str(folder), "--device", "cpu", "--resamples", "50", "--fit-entries", "20",
                           *(["--quantize", quantize] if quantize else [])])
    quant = tmp_path / "quant"
    e4_quant.run([root / "runs"], quant, bits=(8,), variants=("A", "B"), baseline="C0'", device="cpu", resamples=200)
    out = tmp_path / "report"
    summary = e9_report.write_report([root / "runs"], out, quant_dir=quant, resamples=200, quantized="int8")
    assert {"report.md", "summary.json", "manifest.json", "resolved_config.yaml"} <= {p.name for p in out.iterdir()}
    group = summary["groups"]["fake-host · lora"]
    assert group["seeds"] == {"P0": [1], "C0'": [1], "C2": [1], "C5": [1]}
    d1 = group["dimension1"]
    assert d1["flags"] and set(d1["strata"]["all"]) == {"C0'", "C2", "P0"}
    # C5 − P0 at bf16 equals the difference of the two runs' own final losses (token-weighted).
    runs = {r.condition: r for r in e9_report.discover([root / "runs"])}
    c5, p0 = runs["C5"].final["all"]["loss"], runs["P0"].final["all"]["loss"]
    assert d1["strata"]["all"]["P0"]["delta"] == pytest.approx(c5 - p0, abs=1e-6)
    assert d1["probes"]["C0'"][1]["card660/spearman_tied"]["all"]["n"] == 8
    d2 = group["dimension2"]
    assert d2["available"] and d2["variants"] == ["int8-A", "int8-B"] and set(d2["damage"]) == {"P0", "C0'", "C2", "C5"}
    gap = d2["gap"]["C0'"]["int8-A"]["all"]
    assert gap["gain_change"]["mean"] == pytest.approx(gap["gain_quantized"]["mean"] - gap["gain_bf16"]["mean"], abs=1e-9)
    assert "holm_p" in gap["gain_change"] and d2["probe_damage"]["C5"][1]
    d3 = group["dimension3"]["variants"]
    assert set(d3) == {"bf16", "int8"} and d3["bf16"]["models"]["C5"][1]["edits"]["edit_applicable"]
    assert {r["test"] for r in d3["bf16"]["new_words_vs_models"]["C0'"]} >= {"property", "statement_loss"}
    report = (out / "report.md").read_text()
    for heading in ("Dimension 1", "Dimension 2", "Dimension 3", "single seed", "Gap under quantization vs C0′".replace("′", "'")):
        assert heading in report
    assert (out / "figures").is_dir() and len(list((out / "figures").glob("*.png"))) >= 3
    with pytest.raises(FileExistsError):
        e9_report.write_report([root / "runs"], out, quant_dir=quant, resamples=200, quantized="int8")
    e9_report.main(["--runs", str(root / "runs"), "--quant", str(quant), "--output", str(out), "--resamples", "200",
                    "--quantized", "int8", "--overwrite", "--no-figures"])


def _synthetic_run(root: Path, name: str, mode: str, seed: int, sums: np.ndarray, counts: np.ndarray, strata: list[str]) -> Path:
    folder = root / f"host-{mode}-{name}-s{seed}"
    folder.mkdir(parents=True)
    config = {"experiment": f"e9-syn-host-{mode}-{name}-s{seed}", "seed": seed,
              "model": {"pretrained": "org/host", "host_mode": "frozen" if name == "P0" else mode, "seq_len": 8},
              "train": {"total_tokens": 1000, **({"eval_only": True} if name == "P0" else {})},
              "eval": {"windows": sums.shape[1]}, "data": {"eval": "e", "ontology": "o", "min_subtokens": 2}}
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (folder / "manifest.json").write_text("{}")
    rows = [{"type": "eval", "step": 1, "tokens": 1000, "stratum": s, "loss": float(sums[i].sum() / counts[i].sum()),
             "stratum_tokens": int(counts[i].sum())} for i, s in enumerate(strata)]
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lm.save_window_losses(folder / "eval_windows.npz", 1000, list(range(sums.shape[1])),
                          {s: ([sums[i]], [counts[i]]) for i, s in enumerate(strata)})
    return folder


def test_r9_report_statistics_on_synthetic_inputs(tmp_path) -> None:
    rng = np.random.default_rng(0)
    strata = ["all", "unlinked", "inside", "after_rare_seen", "after_heldout"]
    windows = 40
    counts = rng.integers(5, 20, size=(len(strata), windows)).astype(np.int32)
    base = 3.0 * counts
    # P0 3.0 nats; C0' 2.9; C2 2.85; C5 2.8 on linked strata (2.9 on unlinked). INT8 adds 0.2 to every model
    # except C5's linked strata (+0.1): the gap C5 − C0' grows by 0.1 under quantization on linked strata.
    level = {"P0": 3.0, "C0'": 2.9, "C2": 2.85, "C5": 2.8}
    runs_root, quant = tmp_path / "runs", tmp_path / "quant"
    for name, value in level.items():
        noise = rng.normal(0, 0.01, size=counts.shape) * counts
        unlinked = np.asarray(strata)[:, None] == "unlinked"
        loss = np.where(unlinked, 2.9, value) * np.ones(counts.shape) if name == "C5" else np.full(counts.shape, value)
        sums = loss * counts + noise
        folder = _synthetic_run(runs_root, "C0p" if name == "C0'" else name, "full", 1, sums, counts, strata)
        damage = np.full(counts.shape, 0.2)
        if name == "C5":
            damage[[i for i, s in enumerate(strata) if s not in {"all", "unlinked"}]] = 0.1
        out = quant / "runs" / e9_report.run_id(folder)
        out.mkdir(parents=True)
        arrays = {"sum_ref": sums, "sum_int8-A": sums + damage * counts}
        if name in {"C2", "C5"}:
            arrays["sum_int8-B"] = sums + (damage + 0.05) * counts
        np.savez_compressed(out / "windows.npz", strata=np.asarray(strata), starts=np.arange(windows), count=counts, **arrays)
        (out / "quant.json").write_text(json.dumps({"run": str(folder.resolve()), "id": e9_report.run_id(folder)}))
    summary = e9_report.write_report([runs_root], tmp_path / "report", quant_dir=quant, resamples=500, figures=True)
    group = summary["groups"]["host · full"]
    d1 = group["dimension1"]["strata"]
    assert d1["after_heldout"]["C0'"]["delta"] == pytest.approx(-0.1, abs=0.01) and d1["after_heldout"]["C0'"]["significant"]
    assert d1["after_heldout"]["P0"]["delta"] == pytest.approx(-0.2, abs=0.01)
    assert d1["unlinked"]["C0'"]["delta"] == pytest.approx(0.0, abs=0.01)
    assert group["dimension1"]["context"]["after_heldout"]["C0' − P0"]["delta"] == pytest.approx(-0.1, abs=0.01)
    d2 = group["dimension2"]
    assert d2["damage"]["C0'"]["int8-A"]["after_heldout"]["mean"] == pytest.approx(0.2, abs=1e-9)
    assert d2["damage"]["C5"]["int8-A"]["after_heldout"]["mean"] == pytest.approx(0.1, abs=1e-9)
    assert d2["damage"]["C0'"]["int8-B"]["all"]["mean"] == pytest.approx(0.2, abs=1e-9)      # no channel: B = A
    gap = d2["gap"]["C0'"]["int8-A"]
    assert gap["after_heldout"]["gain_change"]["mean"] == pytest.approx(-0.1, abs=1e-9) and gap["after_heldout"]["gain_change"]["significant"]
    assert gap["unlinked"]["gain_change"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert d2["gap"]["P0"]["int8-B"]["after_heldout"]["gain_change"]["mean"] == pytest.approx(-0.05, abs=1e-9)
    report = (tmp_path / "report" / "report.md").read_text()
    assert "negative = the channel's advantage grows" in report and "single seed" in report


# -- 7. tracks (e9_tracks): wording, WP-C7 zero-shot items, plan and quantization on another corpus ------------------

from vsa_embed.experiments import e9_tracks as tracks  # noqa: E402
from vsa_embed.tracks.common import RelationTemplates  # noqa: E402

TRACK_TEMPLATES = {
    "hypernym": RelationTemplates(["{x} is a kind of", "Every {x} is a type of", "Asked what {x} is, we say: a"], "{x} is a {y}."),
    "part_meronym": RelationTemplates(["{x} has a", "One part of {x} is the"], "{x} comes with a {y} inside."),
}


def _toy_lexicon() -> "tracks.TrackLexicon":
    texts = {a: a.partition(":")[2].split(".")[0].replace("_", " ") for a in ATOMS if a.startswith("synset:")}
    return tracks.TrackLexicon("toy", TRACK_TEMPLATES, texts, category_relations=("hypernym",),
                               kept_relations=frozenset({"hypernym", "lexname", "pos"}), edit_relations=("part_meronym", "hypernym"),
                               article_relations=frozenset({"part_meronym"}))


def test_track_lexicon_wording() -> None:
    lex = _toy_lexicon()
    assert lex.prompts("*", "hypernym") == ["{x} is a kind of", "Every {x} is a type of"]
    assert lex.statements("*", "hypernym") == ["Asked what {x} is, we say: a"]            # third paraphrase, answer format
    assert lex.statement_answer("hypernym", "canine") == " canine"
    assert lex.statements("*", "part_meronym") == ["{x} comes with a"]                     # statement prefix, its suffix kept
    assert lex.answer("part_meronym", "engine") == " an engine"
    assert lex.statement_answer("part_meronym", "lid") == " a lid inside."
    assert lex.prompts("*", "lexname") is None and lex.hierarchy("synset:feline.n.01") is None
    assert lex.display_surface("glass jar", "Glass Jar") == "Glass Jar" and lex.display_surface("jar", "Glass Jar") == "jar"
    assert lex.plausible_edit("x", "synset:tail.n.01", "synset:wheel.n.01") and not lex.plausible_edit("x", "synset:tail.n.01", "pos:n")


def test_track_items_on_a_toy_track(world, tmp_path) -> None:
    lex = _toy_lexicon()
    new = edit.build_new_word_items(world["root"] / "ontology.pt", tmp_path / "new", tokenizer_name="gpt2", count=4, min_subtokens=1,
                                    contamination_texts=SENTENCES, lexicon=lex)
    assert new["lexicon"] == "toy" and "all resampled edges" in new["construction"]["rule"]
    _, concepts, prompts = edit.load_item_dir(tmp_path / "new", edit.SCHEMA_NEW)
    view = edit.OntologyView(world["ontology"], world["table"], lex)
    index = view.edge_index()
    for c in concepts:
        category = (view.relation_id["hypernym"], view.atomic_id[c["category"]])
        together = set(index[category])
        for r, f in edit.resolve_frame(c["resampled"], view.relation_id, view.atomic_id):
            together &= index.get((r, f), set())
        assert not together and c["pos"] == "*"                         # no entry states the whole combination
    entail = [p for p in prompts if p["test"] == "entailment"]
    assert entail and all(p["relation"].startswith("corrupted_") and len(p["candidates"]) == 2 for p in entail)
    assert all(p["null"] == "this" for p in prompts)
    statements = [p for p in prompts if p["test"] == "statement" and p["relation"] == "part_meronym"]
    assert statements and all(c.endswith(" inside.") and c.startswith((" a ", " an ")) for p in statements for c in p["candidates"])
    edits = edit.build_edit_items(world["root"] / "ontology.pt", tmp_path / "edits", tokenizer_name="gpt2", count=6, min_subtokens=1,
                                  lexicon=lex)
    assert edits["lexicon"] == "toy"
    _, concepts, prompts = edit.load_item_dir(tmp_path / "edits", edit.SCHEMA_EDITS)
    edited = [c for c in concepts if c["role"] == "edited"]
    assert edited and all(c["relation"] in {"part_meronym", "hypernym"} and c["new"].split(":")[0] == c["old"].split(":")[0]
                          for c in edited)


def _write_wpc7(path: Path) -> None:
    rows, entail = [], []
    facts = [("aardvark", "hypernym", ["mammal", "canine", "feline", "vehicle"], 0), ("acetaminophen", "hypernym",
             ["vehicle", "analgesic", "canine", "container"], 1), ("ant bear", "part_meronym", ["wheel", "lid", "tail", "engine"], 2),
             ("timber wolf", "hypernym", ["canine", "feline", "vehicle", "mammal"], 0)]
    for g, (surface, relation, choices, label) in enumerate(facts):
        split = "train" if surface == "timber wolf" else "heldout"
        for k, prompt in enumerate(["{x} is a kind of", "Every {x} is a type of", "What is {x}? A"]):
            rows.append({"choices": [" " + c for c in choices], "concept": surface, "group": f"t-zp-{g:06d}", "id": f"t-zp-{g:06d}-p{k}",
                         "label": label, "paraphrase": k, "prompt": prompt.format(x=surface), "relation": relation, "split": split,
                         "surface": surface, "task": "zeroshot_property"})
        right, wrong = choices[label], choices[(label + 1) % 4]
        for value, filler in ((1, right), (0, wrong)):
            entail.append({"concept": surface, "id": f"t-ze-{g:06d}-{value}", "label": value, "pair": f"t-ze-{g:06d}",
                           "relation": relation, "split": split, "statement": f"{surface} is a kind of {filler}.",
                           "surface": surface, "task": "zeroshot_entailment"})
    path.mkdir(parents=True, exist_ok=True)
    (path / "zeroshot_property.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "zeroshot_entailment.jsonl").write_text("".join(json.dumps(r) + "\n" for r in entail))


def test_wpc7_items_and_track_zero_shot(world, tmp_path) -> None:
    _write_wpc7(tmp_path / "wpc7")
    concepts, items = tracks.convert_wpc7_items(tmp_path / "wpc7")
    assert len(concepts) == 4 and len(items) == 8
    prop = next(i for i in items if i["test"] == "property" and i["concept"] == "wpc7-acetaminophen")
    assert prop["templates"] == ["{x} is a kind of", "Every {x} is a type of", "What is {x}? A"] and prop["gold"] == 1
    pair = next(i for i in items if i["test"] == "entailment" and i["concept"] == "wpc7-ant bear")
    assert pair["templates"] == ["{x} is a kind of"] and pair["candidates"] == [" tail.", " engine."] and pair["gold"] == 0
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=8)
    fillers = run.composer.schedule.fillers.clone()
    evaluation = tracks.evaluate_track_zeroshot(run, tmp_path / "wpc7", fit_entries=20, log=lambda m: None)
    assert evaluation["sources"] == ["own", "none", "mean_row", "random_frame"]
    assert torch.equal(run.composer.schedule.fillers, fillers)                          # frames restored
    assert {r["split"] for r in evaluation["resolved"].values()} == {"heldout", "train"}
    assert all(r["linked"] for r in evaluation["resolved"].values())
    flat = lambda source: [v for r in evaluation["results"][source]["prompts"] for t in r["pmi"] for v in t]
    assert flat("own") != flat("random_frame")
    summary = tracks.summarize_track_zeroshot(evaluation, resamples=50)
    assert summary["split_counts"]["heldout"] == 3 and set(summary["sources"]["own"]) == {"linked", "synthetic", "heldout"}
    assert summary["sources"]["own"]["heldout"]["property"]["n"] == 3
    offsets, relations = world["ontology"]["offsets"], world["ontology"]["relations"]
    frames = tracks.random_frames(world["ontology"], [0, 2], seed=1)
    assert [r for r, _ in frames[0]] == relations[offsets[0]:offsets[1]].tolist()
    with tracks.replaced_frames(run.channel, frames) as applied:
        assert applied and run.composer.schedule.fillers[offsets[0]:offsets[1]].tolist() == [f for _, f in frames[0]]
    assert torch.equal(run.composer.schedule.fillers, fillers)
    with pytest.raises(ValueError, match="relations"):
        with tracks.replaced_frames(run.channel, {0: [(0, 1)]}):
            pass
    c0 = tracks.evaluate_track_zeroshot(common.open_run(world["runs"]["C0p"], device="cpu"), tmp_path / "wpc7", log=lambda m: None)
    assert c0["sources"] == ["own"]
    out = tmp_path / "zs"
    tracks.main(["zeroshot", "--run", str(world["runs"]["C5"]), "--track", "t5", "--items", str(tmp_path / "wpc7"), "--alias-table",
                 str(world["root"] / "alias_table.json"), "--output", str(out), "--device", "cpu", "--resamples", "50",
                 "--fit-entries", "20", "--quantize", "int8"])
    document = json.loads((out / "summary.json").read_text())
    assert document["track"] == "t5" and document["source"]["quantization"]["variant"] == "int8-A"
    vectors = e9_report.zeroshot_items(out)                                           # the report reads it like E5.4 output
    assert vectors["property"][1].size == 4 and "# E9 track zero-shot" in (out / "report.md").read_text()


def test_e4_quant_on_another_corpus(world, tmp_path) -> None:
    from vsa_embed.experiments import e4_quant
    out = tmp_path / "quant-general"
    e4_quant.main(["--runs", str(world["runs"]["C5"]), str(world["runs"]["C0p"]), "--output", str(out), "--bits", "8", "--device", "cpu",
                   "--resamples", "100", "--baseline", "C0'", "--eval-corpus", str(world["root"] / "train")])
    result = json.loads((out / "quant.json").read_text())
    assert all(r["eval_corpus"] == str(world["root"] / "train") for r in result["runs"])
    assert yaml.safe_load((out / "resolved_config.yaml").read_text())["eval_corpus"] == str(world["root"] / "train")
    assert all(r["ref_check"].get("paired") is False for r in result["runs"])           # other windows than the runs' own


def test_e9_plan_tracks(plan_root, tmp_path) -> None:
    t5 = _plan(plan_root, track="t5", hosts=["SmolLM2-135M"], channel_lr=5e-4)
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in t5}
    c5 = configs["SmolLM2-135M-full-C5-s1"]
    assert c5["e9_track"] == "t5" and c5["data"]["eval"] == str(plan_root["data"] / "eval") and c5["train"]["lr"] == 5e-4
    t1 = _plan(plan_root, track="t1", hosts=["SmolLM2-135M"], models=["C5"])
    c = yaml.safe_load(t1[0].read_text())
    assert c["data"]["eval"] == str(plan_root["data"] / "eval-pubmed") and c["eval"]["windows"] == 2048
    queue = tmp_path / "jobs"
    alias = tmp_path / "t5.json"
    queued = e9_plan.queue_jobs(t5, "t5", 22, track="t5", root=plan_root["root"], queue_dir=queue, alias_table=alias)
    assert "t5-quant-general-s1" in queued and len(queued) == 4 + 4 * 6 + 3
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    for suffix in ("probes", "edit-int4", "zeroshot"):
        command = jobs[f"t5-SmolLM2-135M-full-C5-s1-{suffix}"]["command"]
        assert command[command.index("--alias-table") + 1] == str(alias)
    assert jobs["t5-SmolLM2-135M-full-C5-s1-zeroshot"]["command"][2] == "vsa_embed.experiments.e9_tracks"
    edit_command = jobs["t5-SmolLM2-135M-full-C5-s1-edit"]["command"]
    assert edit_command[edit_command.index("--new-items") + 1].endswith("new-words-t5-smollm2-v1")
    general = jobs["t5-quant-general-s1"]["command"]
    assert general[general.index("--eval-corpus") + 1].endswith("t5-glossary/v1/eval-general")
    assert general[general.index("--output") + 1].endswith("quant-general/t5") and "--quant-general" in jobs["t5-report-s1"]["command"]
    t1_jobs = e9_plan.evaluation_jobs(Path("run"), tracks.TRACKS["t1"], alias_table=alias)
    assert {s for s, _, _ in t1_jobs} == {"probes", "probes-int4", "edit", "edit-int4"}            # T1 has no zero-shot items
    for track in ("t5", "t4", "t1", "wordnet"):
        for folder in e9_plan.dimension3_items(track):
            assert (Path(__file__).resolve().parents[1] / folder / "manifest.json").exists()


def test_r9_report_with_general_text(tmp_path) -> None:
    test_r9_report_statistics_on_synthetic_inputs(tmp_path)               # writes runs/ and quant/ under tmp_path
    summary = e9_report.write_report([tmp_path / "runs"], tmp_path / "report2", quant_dir=tmp_path / "quant", resamples=200,
                                     figures=False, quant_general_dir=tmp_path / "quant")
    general = summary["groups"]["host · full"]["general_text"]
    assert general["available"] and general["gap"]["C0'"]["int8-A"]["all"]["gain_bf16"]["mean"] == pytest.approx(-0.1, abs=0.01)
    assert "### General text" in (tmp_path / "report2" / "report.md").read_text()


def _t5_available() -> bool:
    spec = tracks.TRACKS["t5"]
    root = Path(__file__).resolve().parents[1]
    return spec.ontology.exists() and (root / spec.config).exists() and (spec.data_root / "docs" / "glossary.json").exists()


@pytest.mark.skipif(not _t5_available(), reason="T5 corpus not on this machine")
def test_t5_alias_table_replays_exactly(monkeypatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    table = tracks.track_alias_table(tracks.TRACKS["t5"])
    ontology = torch.load(tracks.TRACKS["t5"].ontology, weights_only=False)
    assert table.digest() == ontology["alias_table_sha256"] and len(table.entry_concepts) == ontology["entry_count"]
    lexicon = tracks.track_lexicon(tracks.TRACKS["t5"], ontology)
    assert lexicon.text("type:process") == "process" and any(t.startswith("the ") for t in lexicon.texts.values())


# -- 8. item versions (decision 56): v2 ⊃ v1, opt-in jobs and report ------------------------------------------------------

def _frame_key(frame) -> frozenset:
    return frozenset(tuple(edge) for edge in frame)


def test_items_v2_extend_new_words_keeps_the_base_first(world, items, tmp_path) -> None:
    ontology = world["root"] / "ontology.pt"
    reserved = edit._reserved_names([items["reserved"]])
    base = tmp_path / "new-v1"
    edit.build_new_word_items(ontology, base, tokenizer_name="gpt2", count=3, min_subtokens=1, contamination_texts=SENTENCES,
                              reserved_names=reserved)
    out = tmp_path / "new-v2"
    manifest = edit.extend_new_word_items(base, ontology, out, tokenizer_name="gpt2", count=6, contamination_texts=SENTENCES,
                                          reserved_names=reserved)
    for name in ("concepts.jsonl", "items.jsonl"):
        assert (out / name).read_bytes().startswith((base / name).read_bytes())
        assert len((out / name).read_bytes()) > len((base / name).read_bytes())
    _, concepts, new_items = edit.load_item_dir(out, edit.SCHEMA_NEW)
    assert [c["concept"] for c in concepts] == [f"e9n-{i:04d}" for i in range(6)] and manifest["counts"]["concepts"] == 6
    assert len({c["surface"].lower() for c in concepts}) == 6
    # The names continue the base's sequence (a longer build has the same names); extension frames are new combinations.
    longer = edit.build_new_word_items(ontology, tmp_path / "new-6", tokenizer_name="gpt2", count=6, min_subtokens=1,
                                       contamination_texts=SENTENCES, reserved_names=reserved)
    assert [c["surface"] for c in concepts] == [json.loads(line)["surface"]
                                                for line in (tmp_path / "new-6" / "concepts.jsonl").read_text().splitlines()]
    existing = {_frame_key([[RELATIONS[r], ATOMS[f]] for r, f in frame]) for frame in FRAMES}
    base_frames = {_frame_key(c["frame"]) for c in concepts[:3]}
    for concept in concepts[3:]:
        assert _frame_key(concept["frame"]) not in existing | base_frames
        assert concept["donor_entry"] not in {c["donor_entry"] for c in concepts[:3]}
        assert any(i["concept"] == concept["concept"] for i in new_items)
    assert manifest["base"]["concepts"] == 3 and manifest["extension"]["first_concept"] == "e9n-0003"
    assert manifest["base"]["sha256"]["items.jsonl"] == edit._sha256(base / "items.jsonl") and manifest["seed"] == 0
    assert manifest["checks"]["names"] == 6 + 64 and manifest["contamination_free"] and longer["count"] == 6
    assert manifest["sha256"]["items.jsonl"] == edit._sha256(out / "items.jsonl")
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("concepts.jsonl", "manifest.json"):
        (tampered / name).write_bytes((base / name).read_bytes())
    lines = (base / "items.jsonl").read_text().splitlines(keepends=True)
    row = json.loads(lines[0])
    row["templates"] = row["templates"] + ["The {x} is"]
    (tampered / "items.jsonl").write_text(json.dumps(row) + "\n" + "".join(lines[1:]))
    with pytest.raises(ValueError, match="not reproduced"):
        edit.extend_new_word_items(tampered, ontology, tmp_path / "bad", tokenizer_name="gpt2", count=6,
                                   contamination_texts=SENTENCES, reserved_names=reserved)
    with pytest.raises(ValueError, match="must exceed"):
        edit.extend_new_word_items(base, ontology, tmp_path / "bad2", tokenizer_name="gpt2", count=3,
                                   contamination_texts=SENTENCES, reserved_names=reserved)


def test_items_v2_extend_edits_never_touch_base_concepts(world, tmp_path) -> None:
    ontology = world["root"] / "ontology.pt"
    base = tmp_path / "edits-v1"
    lexicon = _toy_lexicon()                      # same-type fillers (the toy's WordNet pools are too small to edit within a lexname)
    edit.build_edit_items(ontology, base, tokenizer_name="gpt2", count=2, min_subtokens=1, neighbors=1, lexicon=lexicon)
    pool = edit.edit_pool(ontology, tokenizer_name="gpt2", min_subtokens=1, lexicon=lexicon)
    out = tmp_path / "edits-v2"
    manifest = edit.extend_edit_items(base, ontology, out, tokenizer_name="gpt2", count=5, lexicon=lexicon)
    for name in ("concepts.jsonl", "items.jsonl"):
        assert (out / name).read_bytes().startswith((base / name).read_bytes())
    _, base_concepts, _ = edit.load_item_dir(base, edit.SCHEMA_EDITS)
    _, concepts, all_items = edit.load_item_dir(out, edit.SCHEMA_EDITS)
    base_entries = {c["entry"] for c in base_concepts}                       # edited and neighbour concepts of v1
    extension = [c for c in concepts[len(base_concepts):] if c["role"] == "edited"]
    edited = {c["entry"] for c in concepts if c["role"] == "edited"}
    assert extension and not ({c["entry"] for c in extension} & base_entries)
    assert not ({c["entry"] for c in concepts if c["role"] == "neighbour"} & edited)
    assert len({c["concept"] for c in concepts}) == len(concepts) and len({i["id"] for i in all_items}) == len(all_items)
    assert manifest["pool"]["heldout_eligible"] == pool["heldout"] and manifest["pool"]["seen_eligible"] == pool["seen"]
    assert manifest["base"]["edits"] == 2 and manifest["extension"]["edits"] == len(extension)
    assert manifest["counts"]["efficacy"] == 2 + len(extension) and isinstance(manifest["pool"]["heldout_capped"], bool)


def test_e9_plan_item_versions_are_opt_in(plan_root, tmp_path) -> None:
    spec = tracks.TRACKS["t5"]
    run = Path("runs/t5/SmolLM2-360M-full-C5-s1")
    default = e9_plan.evaluation_jobs(run, spec, alias_table=Path("t5.json"))
    assert [s for s, _, _ in default] == ["probes", "zeroshot", "edit", "probes-int4", "zeroshot-int4", "edit-int4"]
    both = {s: c for s, c, _ in e9_plan.evaluation_jobs(run, spec, alias_table=Path("t5.json"), item_versions=("v1", "v2"))}
    assert both["edit"] == {s: c for s, c, _ in default}["edit"]                          # v1 unchanged
    v2 = both["edit-v2-int4"]
    assert v2[v2.index("--new-items") + 1].endswith("new-words-t5-smollm2-v2")
    assert v2[v2.index("--edit-items") + 1].endswith("edits-t5-smollm2-v2")
    assert v2[v2.index("--output") + 1] == str(run / "edit-v2-int4") and v2[v2.index("--quantize") + 1] == "int4"
    pq = {s: c for s, c, _ in e9_plan.evaluation_jobs(run, spec, profile="pq", model="C5sh", item_versions=("v2",))}
    assert set(pq) == {"zeroshot", "edit-v2"} and "--edit-items" not in pq["edit-v2"]      # C5sh: new words only
    with pytest.raises(ValueError):
        e9_plan.dimension3_items("t5", "smollm2", "v3")
    assert e9_plan.edit_folder("v1", "int4") == "edit-int4" and e9_plan.edit_folder("v2") == "edit-v2"
    counts = {v: e9_plan.item_counts("t5", "smollm2", v) for v in ("v1", "v2")}
    assert counts["v1"]["new_concepts"] == 300 and counts["v2"]["new_concepts"] == 700 and counts["v2"]["edits"] == 700
    new_ratio, edit_ratio = e9_plan.item_scale("t5", "smollm2", "v2")
    assert new_ratio == pytest.approx(counts["v2"]["new_items"] / counts["v1"]["new_items"]) and edit_ratio == pytest.approx(3.5)
    # Evaluation-only re-runs on v2 for an existing stage: core models at bf16 and INT4, arms at bf16, report on v2.
    _plan(plan_root, hosts=["SmolLM2-135M"], seeds=[1, 2], track="t5")
    e9_plan.write_stage("main", hosts=["SmolLM2-135M"], models=["C5sh"], seeds=[1], data_root=plan_root["data"],
                        counts_ontology=plan_root["counts"], root=plan_root["root"], track="t5")
    measured = plan_root["root"] / "runs" / "main" / "SmolLM2-135M-full-C5-s1" / "edit"
    measured.mkdir(parents=True)
    (measured / "summary.json").write_text(json.dumps({"new_words": {"seconds": 50.0}, "edits": {"seconds": 6.0}}))
    queue = tmp_path / "jobs"
    queued, planned = e9_plan.queue_item_evaluations("main", track="t5", root=plan_root["root"], queue_dir=queue,
                                                     alias_table=tmp_path / "t5.json")
    names = [n for n, _, _, _ in planned]
    assert len(planned) == 1 * 2 + 3 * 2 * 2 + 1 + 1 and set(queued) == set(names) and names[-1] == "main-report-items-v2"
    assert "main-SmolLM2-135M-full-C5sh-s1-edit-v2" in names and "main-SmolLM2-135M-full-C5sh-s1-edit-v2-int4" not in names
    hours = {n: (h, b) for n, _, h, b in planned}
    overhead = e9_plan.EDIT_JOB_OVERHEAD_S["SmolLM2-135M"]
    assert hours["main-SmolLM2-135M-full-C5-s1-edit-v2"] == (pytest.approx((overhead + 50 * new_ratio + 6 * edit_ratio) / 3600), "measured")
    assert hours["main-SmolLM2-135M-full-C5-s2-edit-v2"][1].startswith("sibling")
    assert hours["main-SmolLM2-135M-full-C5sh-s1-edit-v2"][0] == pytest.approx((overhead + 50 * new_ratio) / 3600)
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    assert jobs["main-SmolLM2-135M-full-C2-s2-edit-v2-int4"]["priority"] == e9_plan.ITEM_EVAL_PRIORITY
    report = jobs["main-report-items-v2"]
    assert report["priority"] == e9_plan.ITEM_EVAL_PRIORITY + 1 and "--item-seed" in report["command"]
    assert report["command"][report["command"].index("--items-version") + 1] == "v2"
    assert not any("training.lm" in " ".join(j["command"]) for j in jobs.values())          # evaluation only
    assert e9_plan.queue_item_evaluations("main", track="t5", root=plan_root["root"], queue_dir=queue,
                                          alias_table=tmp_path / "t5.json")[0] == []      # idempotent
    _, only = e9_plan.queue_item_evaluations("main", track="t5", root=plan_root["root"], queue=False, models=["C5"], seeds=[2],
                                             report=False)
    assert [n for n, _, _, _ in only] == ["main-SmolLM2-135M-full-C5-s2-edit-v2", "main-SmolLM2-135M-full-C5-s2-edit-v2-int4"]
    with pytest.raises(ValueError):
        e9_plan.queue_item_evaluations("main", track="t5", root=plan_root["root"], queue=False, version="v1")
    # The dimension-3 baselines on v2: own folders, names and report; estimates scaled by the item counts.
    _, dim3 = e9_plan.queue_dim3_baselines("main", track="t5", root=plan_root["root"], queue=False, version="v2", models=["C5"],
                                           seeds=[1])
    name, command, hours_v2 = dim3[0]
    assert name == "main-SmolLM2-135M-full-C5-s1-dim3-baselines-v2"
    assert command[command.index("--output") + 1].endswith("/dim3-baselines-v2")
    assert command[command.index("--edit-items") + 1].endswith("edits-t5-smollm2-v2") and dim3[-1][0] == "main-report-dim3-v2"
    assert "--items-version" in dim3[-1][1] and dim3[-1][1][dim3[-1][1].index("--output") + 1].endswith("main-dim3-v2")
    assert hours_v2 > e9_plan.dim3_estimate_hours("SmolLM2-135M", "C5") * 2
    _, dim3_v1 = e9_plan.queue_dim3_baselines("main", track="t5", root=plan_root["root"], queue=False, models=["C5"], seeds=[1])
    assert dim3_v1[0][0] == "main-SmolLM2-135M-full-C5-s1-dim3-baselines" and "--items-version" not in dim3_v1[-1][1]


def test_r9_report_items_v2_and_item_seed(world, items, tmp_path) -> None:
    ontology = world["root"] / "ontology.pt"
    reserved = edit._reserved_names([items["reserved"]])
    new_v2, edits_v2 = tmp_path / "new-v2", tmp_path / "edits-v2"
    edit.extend_new_word_items(items["new"], ontology, new_v2, tokenizer_name="gpt2", count=7, contamination_texts=SENTENCES,
                               reserved_names=reserved)
    edit.extend_edit_items(items["edits"], ontology, edits_v2, tokenizer_name="gpt2", count=7)
    for name in ("C0p", "C5"):
        run = world["runs"][name]
        edit.main(["evaluate", "--run", str(run), "--new-items", str(new_v2), "--edit-items", str(edits_v2),
                   "--output", str(run / "edit-v2"), "--device", "cpu", "--resamples", "50", "--fit-entries", "20", "--overwrite"])
    out = tmp_path / "report"
    summary = e9_report.write_report([world["root"] / "runs"], out, resamples=100, figures=False, items_version="v2", item_seed=True)
    group = summary["groups"]["fake-host · lora"]
    assert summary["items_version"] == "v2"
    d3 = group["dimension3"]["variants"]["bf16"]["models"]
    assert set(d3) == {"C0'", "C5"} and d3["C5"][1]["new_words"]["concepts"] == 7
    item_seed = group["dimension3_item_seed"]["variants"]["bf16"]
    block = item_seed["new_words"]["own − C0' own"]["tests"]["property"]
    assert block["concepts"] == 7 and block["model"]["seeds"] == 1 and "bootstrap" in block
    assert item_seed["edits"]["metrics"]["log_odds"]
    report = (out / "report.md").read_text()
    assert "Items **v2**" in report and "item × seed analysis" in report
    config = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert config["items_version"] == "v2" and config["item_seed"] is True
    plain = e9_report.write_report([world["root"] / "runs"], tmp_path / "plain", resamples=100, figures=False)
    assert "items_version" not in plain and "dimension3_item_seed" not in next(iter(plain["groups"].values()))
    assert "item_seed" not in yaml.safe_load((tmp_path / "plain" / "resolved_config.yaml").read_text())
