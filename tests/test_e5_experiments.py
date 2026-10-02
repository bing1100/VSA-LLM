"""E5 (WP-E5) metrics on tiny synthetic models: faithfulness, sense alignment, frequency,
zero-shot insertion, rating-study items and the R5 report."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.evaluation import channel_probes as cp
from vsa_embed.experiments import e5_common as common
from vsa_embed.experiments import e5_faithfulness as faith
from vsa_embed.experiments import e5_frequency as freq
from vsa_embed.experiments import e5_rating as rating
from vsa_embed.experiments import e5_report as report
from vsa_embed.experiments import e5_senses as senses
from vsa_embed.experiments import e5_zeroshot as zs
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import train


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        from nltk.corpus import wordnet
        wordnet.synsets("bank")
        return True
    except (OSError, LookupError):
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")

CONCEPTS = ["cat.n.01", "dog.n.01", "bank.n.01", "depository_financial_institution.n.01", "aardvark.n.01", "hammer.n.02",
            "acetaminophen.n.01", "river.n.01", "money.n.01", "wolf.n.01", "sprint.v.01"]
ALIASES = [("cat", 0), ("house cat", 0), ("dog", 1), ("domestic dog", 1), ("bank", 2), ("bank", 3), ("banking company", 3),
           ("aardvark", 4), ("ant bear", 4), ("hammer", 5), ("acetaminophen", 6), ("paracetamol", 6), ("river", 7),
           ("money", 8), ("wolf", 9), ("sprint", 10)]
HELDOUT = [4, 6, 10]
RELATIONS = ["hypernym", "part_meronym", "part_holonym", "lexname", "pos"]
ATOMS = ["synset:feline.n.01", "synset:canine.n.02", "synset:slope.n.01", "synset:financial_institution.n.01",
         "synset:mammal.n.01", "synset:hand_tool.n.01", "synset:analgesic.n.01", "synset:stream.n.01",
         "synset:medium_of_exchange.n.01", "synset:tail.n.01", "synset:paw.n.01", "synset:handle.n.01", "synset:run.v.01",
         "lexname:noun.animal", "lexname:noun.object", "lexname:noun.group", "lexname:noun.artifact", "lexname:noun.possession",
         "lexname:verb.motion", "pos:n", "pos:v"]
A = {name: i for i, name in enumerate(ATOMS)}
R = {name: i for i, name in enumerate(RELATIONS)}


def _frame(*edges):
    return [(R[r], A[a]) for r, a in edges]


FRAMES = [
    _frame(("hypernym", "synset:feline.n.01"), ("part_meronym", "synset:tail.n.01"), ("part_meronym", "synset:paw.n.01"),
           ("lexname", "lexname:noun.animal"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:canine.n.02"), ("part_meronym", "synset:tail.n.01"), ("part_meronym", "synset:paw.n.01"),
           ("lexname", "lexname:noun.animal"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:slope.n.01"), ("part_holonym", "synset:stream.n.01"), ("lexname", "lexname:noun.object"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:financial_institution.n.01"), ("part_meronym", "synset:handle.n.01"), ("lexname", "lexname:noun.group"),
           ("pos", "pos:n")),
    _frame(("hypernym", "synset:mammal.n.01"), ("part_meronym", "synset:tail.n.01"), ("lexname", "lexname:noun.animal"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:hand_tool.n.01"), ("part_meronym", "synset:handle.n.01"), ("lexname", "lexname:noun.artifact"),
           ("pos", "pos:n")),
    _frame(("hypernym", "synset:analgesic.n.01"), ("lexname", "lexname:noun.artifact"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:stream.n.01"), ("lexname", "lexname:noun.object"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:medium_of_exchange.n.01"), ("lexname", "lexname:noun.possession"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:canine.n.02"), ("part_meronym", "synset:tail.n.01"), ("part_meronym", "synset:paw.n.01"),
           ("lexname", "lexname:noun.animal"), ("pos", "pos:n")),
    _frame(("hypernym", "synset:run.v.01"), ("lexname", "lexname:verb.motion"), ("pos", "pos:v")),
]
SENTENCES = ["The cat chased the dog to the river bank.", "She put money in the bank and bought a hammer.",
             "An aardvark ate ants while the wolf slept.", "Take acetaminophen if you sprint too far.",
             "The house cat and the domestic dog sat by the river.", "The banking company kept the money safe.",
             "A wolf and a dog met near the river.", "He used a hammer on the bank of the river.",
             "Paracetamol is also called acetaminophen.", "The ant bear dug near the bank."]


def _write_semcor(root: Path) -> None:
    sentences = []
    for i in range(6):
        sentences.append([("We", None), ("sat", None), ("on", None), ("the", None), ("river", None), ("bank", "bank%1:17:01::")])
        sentences.append([("She", None), ("paid", None), ("the", None), ("bank", "bank%1:14:00::"), ("today", None)])
    xml = ['<?xml version="1.0" encoding="UTF-8" ?>', '<corpus lang="en">']
    keys = []
    for s, tokens in enumerate(sentences):
        xml.append(f'<text id="d{s:03d}">'); xml.append(f'<sentence id="d{s:03d}.s000">')
        for t, (word, key) in enumerate(tokens):
            if key:
                iid = f"d{s:03d}.s000.t{t:03d}"
                xml.append(f'<instance id="{iid}" lemma="{word.lower()}" pos="NOUN">{word}</instance>'); keys.append(f"{iid} {key}")
            else:
                xml.append(f'<wf lemma="{word.lower()}" pos="X">{word}</wf>')
        xml += ["</sentence>", "</text>"]
    xml.append("</corpus>")
    base = root / "wsd" / "WSD_Evaluation_Framework" / "Training_Corpora" / "SemCor"
    base.mkdir(parents=True)
    (base / "semcor.data.xml").write_text("\n".join(xml) + "\n"); (base / "semcor.gold.key.txt").write_text("\n".join(keys) + "\n")


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e5")
    full = AliasTable.from_pairs(ALIASES, holdout=HELDOUT, include_holdout=True)
    rng = random.Random(0)
    texts = []
    for i in range(80):
        chosen = rng.sample(SENTENCES, 5)
        texts.append(f"Doc {i}: " + " ".join(chosen))
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256,
                 max_tokens=80_000, batch_texts=8, workers=2)
    build_corpus(texts[:40], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=80_000,
                 batch_texts=8, workers=2)
    schedule = full.entry_schedule(FRAMES)
    entries = len(full.entry_concepts)
    train_corpus = TokenCorpus.open(root / "train")
    frequency = np.bincount(train_corpus.spans["entry"], minlength=entries)
    names = {cs: i for i, cs in enumerate(full.entry_concepts)}
    for concept, value in ((5, 5), (8, 3), (7, 7)):      # hammer, money, river become "rare"
        frequency[names[(concept,)]] = value
    ontology = {"entry_count": entries, "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": CONCEPTS, "relation_names": RELATIONS, "atomic_names": ATOMS, "concept_frames": FRAMES}
    torch.save(ontology, root / "ontology.pt")
    cp.save_alias_table(full, root / "alias_table.json")
    runs = {}
    specs = {"C0": {"mode": "none"}, "C1": {"mode": "random"}, "C2": {"mode": "free", "free_dimension": 8},
             "C3": {"mode": "compose", "composition": "bundle"},
             "C5": {"mode": "compose", "composition": "attentive", "context_window": 4}}
    for name, channel in specs.items():
        config = {"seed": 0, "device": "cpu", "experiment": f"e4-test-tiny-{name}-s1", "model": {"size": "tiny", "seq_len": 64},
                  "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 64 * 2 * 6, "lr": 3e-3, "warmup_tokens": 64,
                            "log_every": 2, "semantic_weight": 0.1 if name == "C5" else 0.0},
                  "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                           "min_subtokens": 1},
                  "eval": {"windows": 6, "batch": 2, "first_tokens": 128},
                  "channel": {"dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}}
        train(config, root / f"tiny-{name}-s1")
        runs[name] = root / f"tiny-{name}-s1"
    probes = root / "probes"
    _write_semcor(probes)
    pairs = [("cat", "dog", 3.0), ("wolf", "dog", 3.5), ("bank", "money", 2.0), ("river", "bank", 2.5), ("hammer", "cat", 0.2),
             ("aardvark", "wolf", 1.0)]
    (probes / "card660.tsv").write_text("".join(f"{a}\t{b}\t{v}\n" for a, b, v in pairs))
    return {"root": root, "runs": runs, "table": full, "ontology": ontology, "probes": probes}


@pytest.fixture(scope="module")
def opened(world) -> dict:
    return {name: common.open_run(path, device="cpu", batch_size=8) for name, path in world["runs"].items()}


# -- common ---------------------------------------------------------------------------------------------------------

def test_segment_ranks_match_a_loop() -> None:
    torch.manual_seed(0)
    segments = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    keys = torch.tensor([0.5, 0.2, 0.5, 1.0, 3.0, 0.1, 0.1, 0.1, 0.4])
    tiebreak = torch.rand(9)
    ranks = common.segment_ranks(keys, segments, 3, descending=True, tiebreak=tiebreak)
    for s in range(3):
        idx = (segments == s).nonzero().flatten().tolist()
        expected = sorted(idx, key=lambda i: (-float(keys[i]), float(tiebreak[i])))
        assert [int(ranks[i]) for i in expected] == list(range(len(idx)))


def test_ablation_policies_on_known_weights() -> None:
    degrees = torch.tensor([3, 1])
    weights = torch.tensor([0.2, 2.5, 0.3, 1.0])
    segments = torch.tensor([0, 0, 0, 1]); concepts = torch.tensor([0, 1]); edges = torch.arange(4)
    apply = lambda name: common.ablation_transform(common.EdgePolicy.parse(name), degrees, seed=1)(weights, edges, segments, concepts)
    assert apply("remove_top1").tolist() == pytest.approx([0.2, 0.0, 0.3, 1.0])        # the single-edge frame is untouched
    assert apply("remove_bottom1").tolist() == pytest.approx([0.0, 2.5, 0.3, 1.0])
    assert apply("keep_top1").tolist() == pytest.approx([0.0, 2.5, 0.0, 1.0])
    assert apply("remove_all").tolist() == [0.0] * 4
    assert apply("full").tolist() == pytest.approx(weights.tolist())
    random_a, random_b = apply("remove_random2_d0"), apply("remove_random2_d0")
    assert torch.equal(random_a, random_b) and int((random_a[:3] == 0).sum()) == 2
    assert common.EdgePolicy.parse("keep_random4_d1").family == "keep_random4"


def test_uniform_weights_make_top_k_a_random_draw() -> None:
    degrees = torch.tensor([6])
    weights = torch.ones(6); segments = torch.zeros(6, dtype=torch.long); concepts = torch.tensor([0])
    top = common.ablation_transform(common.EdgePolicy("remove", "top", 2), degrees, seed=5)(weights, torch.arange(6), segments, concepts)
    rnd = common.ablation_transform(common.EdgePolicy("remove", "random", 2), degrees, seed=5)(weights, torch.arange(6), segments, concepts)
    assert torch.equal(top, rnd)


def test_override_rows_and_zero_rows(opened) -> None:
    run = opened["C3"]
    table = run.table
    entry = table.alias_to_entry["aardvark"]
    texts = ["An aardvark ate ants near the river bank today."]
    base, _ = run.adapter.token_logprobs(texts)
    own = {entry: common.entry_rows(run.channel, [entry])[0]}
    with common.override_rows(run.channel, own):          # a static composer's own row reproduces the model
        same, _ = run.adapter.token_logprobs(texts)
    assert torch.allclose(base[0], same[0], atol=1e-5)
    with common.override_rows(run.channel, {entry: torch.zeros(own[entry].shape[0])}):
        zero, _ = run.adapter.token_logprobs(texts)
    encoded = run.tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    offsets = [[tuple(o) for o in encoded["offset_mapping"][0].tolist()]]
    spans = run.adapter.link_spans(texts, offsets)
    keep = spans["entry"] != entry
    spans = {k: v[keep] for k, v in spans.items()}
    with torch.no_grad():
        out = run.model(encoded["input_ids"], spans=spans, labels=encoded["input_ids"], reduction="none")
    assert torch.allclose(-out["loss"][0], zero[0], atol=1e-5)        # zero row = no injection
    assert "rows" not in run.channel.__dict__                          # the override is removed afterwards


def test_readable_names_and_surfaces(opened) -> None:
    assert common.readable_atomic("synset:hand_tool.n.01") == "hand tool"
    assert common.readable_atomic("lexname:noun.animal") == "animal"
    assert common.readable_atomic("pos:n") == "noun" and common.readable_atomic("type:int") == "int"
    assert common.edge_text("part_holonym", "synset:stream.n.01") == "is part of stream"
    run = opened["C5"]
    surfaces = common.canonical_surfaces(run.table, run.tokenizer, 2)
    aardvark = run.table.alias_to_entry["aardvark"]
    assert surfaces[aardvark]["surface"] in {"aardvark", "ant bear"} and surfaces[aardvark]["subtokens"] >= 2
    one = common.canonical_surfaces(run.table, run.tokenizer, 1)
    assert one[run.table.alias_to_entry["cat"]] == {"surface": "cat", "subtokens": 1, "linkable": True}
    assert opened["C5"].condition == "C5" and opened["C5"].seed == 1


# -- E5.1 ---------------------------------------------------------------------------------------------------------------

def test_faithfulness_remove_all_is_channel_off_and_shapes(opened) -> None:
    result = faith.faithfulness(opened["C5"], windows=4, batch=2, ks=[1, 2, 4], draws=2, seed=0, save_spans=True, log=None)
    assert result["checks"]["remove_all_matches_channel_off"]
    assert result["sums"].shape == (len(result["policies"]), len(result["strata"]), 4)
    counts = dict(zip(result["strata"], result["counts"].sum(1)))
    assert counts["after"] > 0 and counts["k1"] >= counts["k2"] >= counts["k4"] > 0
    full = result["policies"].index("full")
    assert np.allclose(result["kl"][full], 0) and np.allclose(result["flips"][full], 0)
    summary = faith.summarize(result, [1, 2, 4], resamples=200)
    block = summary["strata"]["k1"]["loss"]
    assert set(block) >= {"comprehensiveness_top", "comprehensiveness_random", "comprehensiveness_bottom",
                          "sufficiency_top_minus_random", "normalized_comprehensiveness_top"}
    assert all("p_holm" in c for c in summary["contrasts"])
    assert result["span_losses"].shape[0] == len(result["spans"]["entry"])


def test_static_composer_comprehensiveness_equals_random(opened) -> None:
    result = faith.faithfulness(opened["C3"], windows=4, batch=2, ks=[1, 2], draws=1, seed=3, log=None)
    assert result["checks"]["uniform_weight_fraction"] == pytest.approx(1.0)
    summary = faith.summarize(result, [1, 2], resamples=200)
    for k in (1, 2):
        block = summary["strata"][f"k{k}"]["loss"]
        assert block["comprehensiveness_top_minus_random"]["mean"] == pytest.approx(0.0, abs=1e-9)
        assert block["sufficiency_top_minus_random"]["mean"] == pytest.approx(0.0, abs=1e-9)
    noisy = faith.summarize(faith.faithfulness(opened["C3"], windows=4, batch=2, ks=[2], draws=3, seed=4, log=None), [2], resamples=500)
    contrast = noisy["strata"]["k2"]["loss"]["comprehensiveness_top_minus_random"]
    assert contrast["ci_low"] <= 1e-9 and contrast["ci_high"] >= -1e-9   # within noise


def test_probe_predictions_under_ablation(world, opened, tmp_path) -> None:
    comparisons = faith.probe_ablation(opened["C5"], tmp_path, probes=["card660"], k=1, root=world["probes"], seed=0)
    assert set(comparisons) == {"remove_top1-vs-full", "remove_random1-vs-full", "remove_bottom1-vs-full",
                                "remove_top1-vs-remove_random1"}
    assert "card660" in comparisons["remove_top1-vs-full"]["tables"]
    assert (tmp_path / "probes" / "full.json").exists() and "edge_weights" not in opened["C5"].composer.__dict__


def test_faithfulness_cli_writes_a_run_folder(world, tmp_path) -> None:
    out = tmp_path / "faith"
    faith.main(["--run", str(world["runs"]["C5"]), "--output", str(out), "--windows", "4", "--batch", "2", "--ks", "1", "2",
                "--draws", "1", "--resamples", "100", "--device", "cpu"])
    for name in ("summary.json", "report.md", "manifest.json", "resolved_config.yaml", "windows.npz"):
        assert (out / name).exists()
    with pytest.raises(ValueError, match="composition channel"):
        faith.faithfulness(common.open_run(world["runs"]["C2"], device="cpu"), windows=2, batch=2, ks=[1], draws=1, seed=0, log=None)


# -- E5.2 ----------------------------------------------------------------------------------------------------------------

def test_sense_ownership_and_masses(world) -> None:
    table = world["table"]
    schedule = table.entry_schedule(FRAMES)
    bank = table.alias_to_entry["bank"]
    owners = senses.edge_owners(schedule, bank, [2, 3], FRAMES)
    assert sum(len(o) == 2 for o in owners) == 1          # only the shared POS edge has two owners
    mass = senses.sense_masses([1.0] * len(owners), owners)
    assert mass[2] == pytest.approx(3.5) and mass[3] == pytest.approx(3.5)


def test_concept_frames_must_reproduce_the_schedule(world) -> None:
    ontology = dict(world["ontology"])
    assert len(senses.concept_frames(ontology, world["table"])) == len(CONCEPTS)
    broken = dict(ontology, concept_frames=[f[:-1] for f in FRAMES])
    with pytest.raises(ValueError, match="entry schedule"):
        senses.concept_frames(broken, world["table"])


def test_score_items_tie_rule_and_cross_fitted_mfs() -> None:
    rank = lambda lemma, pos, synset: {"a.n.01": 0, "b.n.01": 1}.get(synset, 5)
    names = ["a.n.01", "b.n.01"]
    items = [{"concepts": [0, 1], "gold": 1, "lemma": "x", "pos": "NOUN", "fold": f, "entry": 0, "mass": m}
             for f, m in ((0, {0: 1.0, 1: 1.0}), (1, {0: 0.2, 1: 2.0}), (1, {0: 0.1, 1: 1.0}))]
    senses.score_items(items, rank, names, set(), None)
    assert items[0]["tie"] and items[0]["predicted"] == 0 and items[0]["wn1"] == 0
    assert items[1]["predicted_correct"] == 1
    assert items[0]["semcor_mfs"] == 1          # counts come from fold 1, where gold is b
    assert items[1]["semcor_mfs"] == 1          # fold-1 items count fold 0, where the gold is also b
    summary = senses.summarize(items, resamples=100)
    assert summary["all"]["n"] == 3 and 0 <= summary["all"]["tie_rate"] <= 1


def test_sense_alignment_end_to_end(world, tmp_path) -> None:
    args = argparse.Namespace(run=world["runs"]["C5"], output=tmp_path / "senses", checkpoint="final.pt", dataset="semcor", limit=None,
                              max_degree=16, probes_root=world["probes"], seed=0, resamples=200, device="cpu")
    result = senses.run_experiment(args)
    assert result["linked"] == 12 and result["summary"]["all"]["n"] == 12
    rows = [json.loads(line) for line in (tmp_path / "senses" / "items.jsonl").read_text().splitlines()]
    assert all(0 <= r["gold_share"] <= 1 for r in rows)
    static = senses.run_experiment(argparse.Namespace(**{**vars(args), "run": world["runs"]["C3"], "output": tmp_path / "static"}))
    assert static["summary"]["all"]["tie_rate"] == pytest.approx(1.0)   # equal edge counts per sense → every item ties


# -- E5.5 -----------------------------------------------------------------------------------------------------------------

def test_ridge_probe_recovers_planted_frequency() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 12))
    y = x @ rng.standard_normal(12) + 0.05 * rng.standard_normal(400)
    good = freq.probe(x, y, folds=5, resamples=100)
    noise = freq.probe(rng.standard_normal((400, 12)), y, folds=5, resamples=100)
    assert good["r2"] > 0.95 and good["quartile_auc"] > 0.95
    assert noise["r2"] < 0.1


def test_frequency_representations_per_condition(world, opened, tmp_path) -> None:
    counts = freq.token_counts(world["root"] / "train", vocab=50257)
    assert counts.sum() == len(TokenCorpus.open(world["root"] / "train"))
    c0 = freq.representations(opened["C0"], counts, max_items=None, seed=0)
    c5 = freq.representations(opened["C5"], counts, max_items=None, seed=0)
    assert set(c0) == {"token_rows", "concept_surface"}
    assert set(c5) == {"token_rows", "concept_surface", "concept_rows", "concept_rows_unit", "concept_rows_init"}
    assert not np.allclose(c5["concept_rows"]["x"], c5["concept_rows_init"]["x"])          # trained vs fresh composer
    heldout = set(opened["C5"].adapter.heldout_entries)
    assert not set(c5["concept_rows"]["ids"].tolist()) & heldout
    args = argparse.Namespace(run=world["runs"]["C2"], output=tmp_path / "freq", checkpoint="final.pt", folds=2, max_items=None, seed=0,
                              resamples=50, token_counts=None, save_predictions=True, device="cpu")
    result = freq.run_experiment(args)
    assert "concept_rows" in result["summary"] and (tmp_path / "freq" / "predictions.npz").exists()


# -- E5.4 ----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def items(world, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("items")
    out = {}
    for scenario in ("c3_heldout", "c3_synthetic"):
        zs.SCENARIOS[scenario](ontology_path=world["root"] / "ontology.pt", out_dir=root / scenario, tokenizer_name="gpt2",
                               min_subtokens=1, max_concepts=10, seed=0, contamination_texts=SENTENCES)
        out[scenario] = root / scenario
    return out


def test_c3_items_are_well_formed_and_paired(items) -> None:
    manifest, concepts, prompts = zs.load_items(items["c3_heldout"])
    _, s_concepts, s_prompts = zs.load_items(items["c3_synthetic"])
    assert manifest["counts"]["concepts"] == 3 and {c["source_concept"] for c in concepts} == {"aardvark.n.01", "acetaminophen.n.01", "sprint.v.01"}
    assert [c["entry"] for c in concepts] == [c["entry"] for c in s_concepts]          # same reserved concepts, invented names
    from nltk.corpus import wordnet
    lemmas = {l.lower() for s in wordnet.all_synsets() for l in s.lemma_names()}
    assert all(c["synthetic"] and c["surface"] not in lemmas for c in s_concepts)
    assert all(len(p["templates"]) >= 2 for p in prompts)
    assert all(not any(c["definition"] in t for t in p["templates"]) for p in prompts for c in concepts)
    hyper = next(p for p in prompts if p["id"] == f"c3h-{concepts[0]['entry']}-property-hypernym")
    assert hyper["candidates"][hyper["gold"]] == " mammal" and len(hyper["candidates"]) == 5
    assert any(p["test"] == "entailment" for p in prompts)
    assert [(p["test"], p["relation"], p["candidates"], p["gold"]) for p in prompts] == \
           [(p["test"], p["relation"], p["candidates"], p["gold"]) for p in s_prompts]


def test_synthetic_names_link_to_the_reserved_entry(opened, items) -> None:
    _, concepts, prompts = zs.load_items(items["c3_synthetic"])
    run = opened["C5"]
    adapter = zs.scenario_adapter(run, concepts)
    assert adapter is not run.adapter and adapter.linker.table.alias_to_entry != run.table.alias_to_entry
    resolved = zs.resolve_entries(adapter, concepts, prompts, set(run.adapter.heldout_entries), run.adapter.train_frequency)
    assert all(r["linked"] and r["status"] == "heldout" for r in resolved.values())
    plain = zs.resolve_entries(run.adapter, concepts, prompts, set(), None)
    assert all(not r["linked"] for r in plain.values())


def test_zero_shot_evaluation_sources_and_invariants(opened, items) -> None:
    run = opened["C5"]
    evaluation = zs.evaluate(run, items["c3_heldout"], fit_entries=50, contexts=2, generator_steps=20, seed=0, windows=30,
                             log=lambda m: None)
    assert evaluation["sources"][0] == "own"
    assert set(evaluation["sources"]) >= {"own", "none", "random", "mean_row", "surface_mean", "graph_projection",
                                          "definition_mean", "definition_encoder", "alacarte", "college"}
    n_items = len(zs.load_items(items["c3_heldout"])[2])
    for source, result in evaluation["results"].items():
        assert len(result["prompts"]) == n_items
        assert result["corpus"] and all("rank" in r for r in result["corpus"] if r["start"] > 0)  # semantic head
    assert 0 < evaluation["info"]["alacarte_coverage"] <= 1 and evaluation["info"]["random_coverage"] == 1
    reserved = {info["entry"] for info in evaluation["resolved"].values()}
    summary = zs.summarize(evaluation, resamples=100)
    assert summary["linked_concepts"] == 3 and len(reserved) == 3
    families = {c["family"] for c in summary["comparisons"]}
    assert families == {"structure", "text_evidence"}
    c0 = zs.evaluate(opened["C0"], items["c3_heldout"], windows=30, log=lambda m: None)
    assert c0["sources"] == ["own"] and not c0["results"]["own"]["corpus"][0].get("rank")


def test_none_source_equals_dropping_the_span(opened, items) -> None:
    run = opened["C3"]
    _, concepts, prompts = zs.load_items(items["c3_heldout"])
    resolved = zs.resolve_entries(run.adapter, concepts, prompts, set(), None)
    entries = {r["entry"] for r in resolved.values()}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    zero = {e: torch.zeros(run.channel.gate.in_features // 2) for e in entries}
    with common.override_rows(run.channel, zero):
        with_none, _ = zs.score_prompts(run.adapter, prompts, surfaces)
    linker = run.adapter.linker
    table = AliasTable({a: e for a, e in linker.table.alias_to_entry.items() if e not in entries}, linker.table.entry_concepts,
                       linker.table.holdout)
    import dataclasses
    from vsa_embed.span_channel import CausalLinker
    dropped = dataclasses.replace(run.adapter, linker=CausalLinker(table, min_subtokens=linker.min_subtokens), spans_fn=None)
    without, _ = zs.score_prompts(dropped, prompts, surfaces)
    flat = lambda rows: [v for r in rows for template in r["pmi"] for v in template]
    assert flat(with_none) == pytest.approx(flat(without), abs=1e-4)


def test_ridge_map_and_graph_features(world) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 6)); w = rng.standard_normal((6, 4))
    predict, _ = zs.ridge_map(x, x @ w)
    assert np.allclose(predict(x[:5]), x[:5] @ w, atol=0.05)
    embed = zs.graph_features(world["ontology"], fit_entries=list(range(8)), rank=4)
    assert embed([0, 1]).shape == (2, 4)
    cat, wolf, money = (world["table"].alias_to_entry[a] for a in ("cat", "wolf", "money"))
    v = embed([cat, wolf, money])
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    assert cos(v[0], v[1]) > cos(v[0], v[2])           # shared frame edges → close in the graph embedding
    diff = zs.paired_difference(np.ones(20), np.zeros(20), resamples=100)
    assert diff["mean"] == 1.0 and diff["p_value"] < 0.05
    assert zs.paired_difference(np.ones(5), np.ones(5), resamples=50)["p_value"] == 1.0


def test_c6_items_and_cli(tmp_path) -> None:
    from vsa_embed.benchmarks.devtools import build
    bench = tmp_path / "bench"
    build(bench, seed=1)
    manifest = zs.build_c6_items(bench, tmp_path / "c6")
    _, concepts, prompts = zs.load_items(tmp_path / "c6")
    assert manifest["contamination_free"] and manifest["counts"]["concepts"] == len(concepts) > 0
    assert {p["relation"] for p in prompts} >= {"module", "kind", "category"}
    assert all(c["entry"] is None for c in concepts)


def test_zero_shot_cli(world, items, tmp_path) -> None:
    out = tmp_path / "zs"
    zs.main(["evaluate", "--run", str(world["runs"]["C2"]), "--items", str(items["c3_synthetic"]), "--output", str(out),
             "--sources", "own,none,random,surface_mean", "--resamples", "50", "--device", "cpu"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["summary"]["sources"]) == {"own", "none", "random", "surface_mean"}
    assert (out / "predictions.jsonl").exists() and (out / "report.md").read_text().startswith("# E5.4")


# -- E5.3 -----------------------------------------------------------------------------------------------------------------

def test_fisher_reproduces_the_paper() -> None:
    from vsa_embed.judge_protocol import fisher_one_tailed
    assert fisher_one_tailed(28, 40, 4, 40) == pytest.approx(2.44e-8, rel=0.02)


def test_neighbour_items_and_fake_grading(opened, tmp_path) -> None:
    runs = {"channel": opened["C5"], "C0": opened["C0"], "C2": opened["C2"]}
    manifest = rating.build_neighbour_items(runs, tmp_path / "items", n_concepts=2, top=2, calibration=4, seed=0)
    assert manifest["n_per_system"] == 4 and manifest["counts"]["calibration"] >= 4
    loaded, rows = rating.load_rating_items(tmp_path / "items")
    pairs = [r for r in rows if r["study"] == "neighbour_pair" and not r["calibration"]]
    assert all(set(r["systems"]) <= {"channel", "C0", "C2"} for r in pairs)
    for name in runs:
        assert sum(name in r["systems"] for r in pairs) == 4
    assert all("channel" not in p for r in rows if not r["calibration"] for p in rating.item_prompts(r))   # systems stay hidden
    result = rating.run_grade(tmp_path / "items", tmp_path / "graded", runner="fake", calls=3)
    pair = result["analysis"]["studies"]["neighbour_pair"]
    assert set(pair["fisher"]) == {"C0", "C2"} and pair["counts"]["channel"]["total"] == 4
    assert result["judge"]["new_calls"] == 3 * len(rows)
    again = rating.run_grade(tmp_path / "items", tmp_path / "graded", runner="fake", calls=3, dry_run=True)
    assert again["dry_run"]["new_calls"] == 0                                         # verdicts are cached
    with pytest.raises(RuntimeError, match="max-new-calls"):
        rating.run_grade(tmp_path / "items", tmp_path / "other", runner="cli", calls=3, max_new_calls=1)
    assert (tmp_path / "graded" / "verdicts.jsonl").exists() and "LLM-graded" in (tmp_path / "graded" / "report.md").read_text()


def test_preference_counterbalancing_and_consensus() -> None:
    item = {"study": "neighbour_preference", "fields": {"concept": "x", "list_a": "one", "list_b": "two"}}
    first, second = rating.item_prompts(item)
    assert first.index("one") < first.index("two") and second.index("two") < second.index("one")
    assert rating.call_value(item, 1, {"choice": "A"}) == "B" and rating.call_value(item, 0, {"choice": "A"}) == "A"
    assert rating.consensus("neighbour_pair", ["strongly related", "unrelated", "less related"]) == "less related"
    assert rating.consensus("neighbour_preference", ["A", "B", "tie"]) == "tie"


def test_edge_and_card_items(world, opened, tmp_path) -> None:
    manifest = rating.build_edge_items(opened["C5"], tmp_path / "edges", n=3, calibration=4)
    _, rows = rating.load_rating_items(tmp_path / "edges")
    real = [r for r in rows if not r["calibration"]]
    assert manifest["n"] == 3 and {r["systems"][0] for r in real} == {"top_edges", "random_edges", "neighbours"}
    top = [r for r in real if r["systems"] == ["top_edges"]]
    assert all(r["weights"] == sorted(r["weights"], reverse=True) for r in top)
    tail = A["synset:tail.n.01"]; paw = A["synset:paw.n.01"]
    cards = [{"event": "split", "target": "atomics", "step": 3, "parent": tail, "children": [tail, paw], "gain": 1.0, "p_value": 0.01}]
    groups = rating.card_groups(cards, opened["C5"].composer.schedule, target="atomics")
    assert len(groups) == 1 and all(len({u[0] for u in g}) >= 2 for g in groups[0]["groups"])
    card_manifest = rating.build_card_items(opened["C5"], tmp_path / "cards", n=5, cards=cards, calibration=4)
    _, card_rows = rating.load_rating_items(tmp_path / "cards")
    real_cards = [r for r in card_rows if not r["calibration"]]
    assert card_manifest["n"] == 1 and {r["systems"][0] for r in real_cards} == {"split", "random"}
    assert "tail" in real_cards[0]["fields"]["concept"]
    graded = rating.run_grade(tmp_path / "cards", tmp_path / "cards-graded", runner="fake", calls=3)
    assert "fisher" in graded["analysis"]["studies"]["split_card"]


# -- R5 report --------------------------------------------------------------------------------------------------------------

def test_r5_report_aggregates_every_kind(world, items, tmp_path) -> None:
    runs = tmp_path / "runs"
    for name in ("C3", "C5"):
        faith.main(["--run", str(world["runs"][name]), "--output", str(runs / f"faith-{name}"), "--windows", "4", "--batch", "2",
                    "--ks", "1", "2", "--draws", "1", "--resamples", "50", "--no-kl", "--device", "cpu"])
        zs.main(["evaluate", "--run", str(world["runs"][name]), "--items", str(items["c3_heldout"]), "--output", str(runs / f"zs-{name}"),
                 "--sources", "own,none,random", "--resamples", "50", "--device", "cpu"])
    senses.run_experiment(argparse.Namespace(run=world["runs"]["C5"], output=runs / "senses-C5", checkpoint="final.pt", dataset="semcor",
                                             limit=None, max_degree=16, probes_root=world["probes"], seed=0, resamples=50, device="cpu"))
    freq.run_experiment(argparse.Namespace(run=world["runs"]["C5"], output=runs / "freq-C5", checkpoint="final.pt", folds=2, max_items=None,
                                           seed=0, resamples=50, token_counts=None, save_predictions=False, device="cpu"))
    rating.run_grade(_neighbour_items(world, tmp_path), runs / "rating", runner="fake", calls=3)
    summary = report.write_report([runs], tmp_path / "R5")
    text = (tmp_path / "R5" / "report.md").read_text()
    for heading in ("E5.1 Faithfulness", "E5.2 Sense alignment", "E5.3 Rating study", "E5.4 Zero-shot", "E5.5 Frequency"):
        assert heading in text
    assert {r["condition"] for r in summary["faithfulness"]} == {"C3", "C5"}
    assert summary["zeroshot"]["c3_heldout"]["cross_condition"]
    assert (tmp_path / "R5" / "figures" / "faithfulness.png").exists()


def _neighbour_items(world, tmp_path) -> Path:
    runs = {"channel": common.open_run(world["runs"]["C5"], device="cpu"), "C0": common.open_run(world["runs"]["C0"], device="cpu")}
    rating.build_neighbour_items(runs, tmp_path / "nitems", n_concepts=2, top=2, calibration=2)
    return tmp_path / "nitems"
