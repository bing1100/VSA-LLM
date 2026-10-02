import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.data.semcor import load_wsd_instances
from vsa_embed.evaluation import channel_probes as cp
from vsa_embed.evaluation.probes import ModelAdapter, lambada, wic, word_similarity, load_card660
from vsa_embed.span_channel import AliasTable
from vsa_embed.training.lm import train


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


def _wordnet_ok() -> bool:
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("bank"); return True
    except LookupError:
        return False


pytestmark = pytest.mark.skipif(not (_gpt2_ok() and _wordnet_ok()), reason="gpt2 tokenizer or WordNet not available")

ALIASES = [("hydroxychloroquine", 0), ("new york", 1), ("acetaminophen", 2), ("cat", 3), ("river bank", 4)]
HELDOUT_CONCEPTS = [2]


def _write_wsd(root: Path) -> None:
    def document(sentences: list[tuple[str, list[tuple[str, str, str, str]]]], prefix: str) -> tuple[str, str]:
        xml, keys = ['<?xml version="1.0" encoding="UTF-8" ?>', '<corpus lang="en">', f'<text id="{prefix}.d000">'], []
        for s, (sid, tokens) in enumerate(sentences):
            xml.append(f'<sentence id="{prefix}.d000.s{s:03d}">')
            for t, (word, lemma, pos, key) in enumerate(tokens):
                if key:
                    iid = f"{prefix}.d000.s{s:03d}.t{t:03d}"
                    xml.append(f'<instance id="{iid}" lemma="{lemma}" pos="{pos}">{word}</instance>'); keys.append(f"{iid} {key}")
                else:
                    xml.append(f'<wf lemma="{lemma}" pos="{pos}">{word}</wf>')
            xml.append("</sentence>")
        xml += ["</text>", "</corpus>"]
        return "\n".join(xml) + "\n", "\n".join(keys) + "\n"

    w = lambda word: (word, word.lower(), "X", None)
    river = lambda: ("bank", "bank", "NOUN", "bank%1:17:01::")
    money = lambda: ("bank", "bank", "NOUN", "bank%1:14:00::")
    cat = ("cat", "cat", "NOUN", "cat%1:05:00::")
    drug = ("acetaminophen", "acetaminophen", "NOUN", "acetaminophen%1:06:00::")
    train_sentences = []
    for i in range(6):
        train_sentences.append((i, [w("We"), w("sat"), w("on"), w("the"), w("river"), river(), w("today")]))
        train_sentences.append((i, [w("She"), w("put"), w("money"), w("in"), w("the"), money(), w("today")]))
        train_sentences.append((i, [w("The"), cat, w("took"), drug, w("today")]))
    test_sentences = [(0, [w("They"), w("fished"), w("from"), w("the"), w("river"), river()]),
                      (1, [w("He"), w("robbed"), w("the"), money(), w("and"), w("the"), cat]),
                      (2, [w("A"), ("dog", "dog", "NOUN", "dog%1:05:00::"), w("ate"), drug])]
    semcor = root / "wsd" / "WSD_Evaluation_Framework" / "Training_Corpora" / "SemCor"
    every = root / "wsd" / "WSD_Evaluation_Framework" / "Evaluation_Datasets" / "ALL"
    semcor.mkdir(parents=True); every.mkdir(parents=True)
    xml, keys = document(train_sentences, "semcor")
    (semcor / "semcor.data.xml").write_text(xml); (semcor / "semcor.gold.key.txt").write_text(keys)
    xml, keys = document(test_sentences, "semeval2007")
    (every / "ALL.data.xml").write_text(xml); (every / "ALL.gold.key.txt").write_text(keys)


def _write_probe_data(root: Path) -> None:
    (root / "lambada" / "data").mkdir(parents=True)
    passages = ["the cat saw hydroxychloroquine in New York and then the cat",
                "acetaminophen was sold in New York and people bought acetaminophen",
                "we walked along the river bank to the river",
                "the cat sat on the mat near the cat"] * 2
    (root / "lambada" / "data" / "lambada_test_en.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in passages))
    wic_rows = [("bank", "N", "3-1", "We sat by the bank today", "The bank lent money", "F"),
                ("cat", "N", "1-1", "The cat slept", "A cat purred", "T"),
                ("run", "V", "1-1", "They run fast", "We run home", "T"),
                ("acetaminophen", "N", "1-0", "Take acetaminophen now", "acetaminophen helps", "T"),
                ("bank", "N", "1-1", "The bank closed", "A bank opened", "T"),
                ("play", "V", "1-1", "Kids play outside", "Actors play roles", "F")]
    for split, rows in (("train", wic_rows * 2), ("dev", wic_rows)):
        (root / "wic" / split).mkdir(parents=True)
        (root / "wic" / split / f"{split}.data.txt").write_text("".join("\t".join(r[:5]) + "\n" for r in rows))
        (root / "wic" / split / f"{split}.gold.txt").write_text("".join(r[5] + "\n" for r in rows))
    pairs = [("cat", "dog", 3.0), ("acetaminophen", "hydroxychloroquine", 2.5), ("bank", "river", 1.0),
             ("new york", "city", 2.0), ("cat", "acetaminophen", 0.1), ("money", "bank", 2.2)]
    (root / "card660.tsv").write_text("".join(f"{a}\t{b}\t{s}\n" for a, b, s in pairs))
    (root / "rw" / "rw").mkdir(parents=True)
    (root / "rw" / "rw" / "rw.txt").write_text("".join(f"{a}\t{b}\t{s * 2}\t1\t2\n" for a, b, s in pairs))
    bless = []
    for concept, hyper, coord, mero, random in (("cat", "animal", "dog", "tail", "spoon"), ("dog", "animal", "wolf", "paw", "chair"),
                                                ("hammer", "tool", "saw", "handle", "river"), ("apple", "fruit", "pear", "core", "car"),
                                                ("acetaminophen", "drug", "aspirin", "tablet", "cloud"), ("car", "vehicle", "truck", "wheel", "apple"),
                                                ("robin", "bird", "crow", "wing", "brick"), ("oak", "tree", "pine", "leaf", "phone")):
        for relation, relatum in (("hyper", hyper), ("coord", coord), ("mero", mero), ("random-n", random), ("hyper", "object")):
            bless.append(f"{concept}-n\tclass\t{relation}\t{relatum}-n")
        bless.append(f"{concept}-n\tclass\tattri\tbig-j")
    (root / "bless" / "bless-gems").mkdir(parents=True)
    (root / "bless" / "bless-gems" / "BLESS.txt").write_text("\n".join(bless) + "\n")
    header = "WORD1 WORD2 POS TYPE AVG_SCORE AVG_SCORE_0_10 STD SCORES..\n"
    hyperlex = [("cat", "animal", "N", "hyp-1", 5.8), ("dog", "animal", "N", "hyp-1", 5.9), ("cat", "dog", "N", "cohyp", 0.5),
                ("acetaminophen", "drug", "N", "hyp-1", 5.5), ("car", "vehicle", "N", "hyp-1", 5.7), ("apple", "fruit", "N", "hyp-1", 5.9),
                ("oak", "tree", "N", "hyp-1", 5.6), ("tree", "oak", "N", "r-hyp-1", 1.0), ("run", "move", "V", "hyp-1", 4.5),
                ("walk", "move", "V", "hyp-1", 4.8), ("eat", "run", "V", "no-rel", 0.2), ("bank", "river", "N", "no-rel", 0.3)]
    line = lambda r: f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]:.2f} {r[4] * 10 / 6:.2f} 1.0 5 5\n"
    (root / "hyperlex" / "splits" / "lexical").mkdir(parents=True)
    (root / "hyperlex" / "hyperlex-all.txt").write_text(header + "".join(line(r) for r in hyperlex))
    for name, rows in (("training", hyperlex[:6]), ("dev", hyperlex[6:9]), ("test", hyperlex[9:])):
        (root / "hyperlex" / "splits" / "lexical" / f"hyperlex_{name}_all_lexical.txt").write_text(header + "".join(line(r) for r in rows))
    _write_wsd(root)


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("channel-probes")
    full = AliasTable.from_pairs(ALIASES, holdout=HELDOUT_CONCEPTS, include_holdout=True)
    texts = [f"Doc {i}: the cat saw hydroxychloroquine on the river bank in New York and acetaminophen too." for i in range(60)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    build_corpus(texts[:20], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256,
                 max_tokens=50_000, batch_texts=8, workers=2)
    entries = len(full.entry_concepts)
    frames = [[(0, e % 6), (1, (e + 1) % 6)] for e in range(entries)]
    offsets = torch.tensor([0] + [2 * (e + 1) for e in range(entries)])
    ontology = {"entry_count": entries, "atomic_count": 6, "relation_count": 2, "offsets": offsets,
                "relations": torch.tensor([r for f in frames for r, _ in f]), "fillers": torch.tensor([a for f in frames for _, a in f]),
                "heldout_entries": sorted(full.heldout_entries()), "train_frequency": [60] * entries,
                "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest()}
    torch.save(ontology, root / "ontology.pt")
    cp.save_alias_table(full, root / "alias_table.json")
    probes = root / "probes"
    _write_probe_data(probes)
    runs = {}
    for mode in ("compose", "none"):
        config = {"seed": 0, "device": "cpu", "model": {"size": "tiny", "seq_len": 64},
                  "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": 64 * 2 * 6, "lr": 3e-3, "warmup_tokens": 64,
                            "log_every": 2},
                  "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"),
                           "min_subtokens": 1},
                  "eval": {"windows": 2, "batch": 2, "first_tokens": 128},
                  "channel": {"mode": mode, "dimension": 16, "key_dimension": 8, "gate_bias": 0.0}}
        train(config, root / mode)
        runs[mode] = root / mode
    return {"root": root, "probes": probes, "runs": runs, "table": full}


def test_alias_table_is_resolved_from_the_sidecar_and_verified(setup) -> None:
    ontology = torch.load(setup["root"] / "ontology.pt", weights_only=False)
    table, info = cp.resolve_alias_table(ontology, setup["root"] / "ontology.pt")
    assert table.alias_to_entry == setup["table"].alias_to_entry
    assert info["checks"] == {"entry_concepts": True, "digest": True, "heldout_entries": True}
    tampered = dict(ontology, alias_table_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        cp.resolve_alias_table(tampered, setup["root"] / "ontology.pt")


def test_wordnet_rebuild_matches_the_ontology_alias_pairs(tmp_path: Path) -> None:
    from nltk.corpus import wordnet
    names = sorted(["bank.n.01", "cat.n.01", "dog.n.01", "depository_financial_institution.n.01"])
    pairs = [(lemma.replace("_", " "), i) for i, n in enumerate(names) for lemma in wordnet.synset(n).lemma_names()]
    expected = AliasTable.from_pairs(pairs, holdout=[1], include_holdout=True)
    (tmp_path / "holdout_concepts.txt").write_text(names[1] + "\n")
    import hashlib
    ontology = {"concept_names": names, "entry_count": len(expected.entry_concepts), "entry_concepts": expected.entry_concepts,
                "alias_table_sha256": expected.digest(), "heldout_entries": sorted(expected.heldout_entries()),
                "holdout_sha256": hashlib.sha256(names[1].encode()).hexdigest()}
    table, info = cp.resolve_alias_table(ontology, tmp_path / "ontology.pt")
    assert info["source"] == "wordnet" and info["checks"]["digest"] and table.digest() == expected.digest()


def test_c0_run_behaves_exactly_like_the_plain_model(setup) -> None:
    adapter = cp.load_run(setup["runs"]["none"], device="cpu")
    assert adapter.spans_fn is None and adapter.linker is not None
    plain = ModelAdapter(adapter.model.model, adapter.tokenizer, torch.device("cpu"))
    texts = ["the cat saw hydroxychloroquine", "acetaminophen in New York today"]
    spans = [(4, 7), (0, 13)]
    assert torch.equal(adapter.word_state(texts, spans), plain.word_state(texts, spans))
    for a, b in zip(adapter.token_logprobs(texts)[0], plain.token_logprobs(texts)[0]):
        assert torch.equal(a, b)


def test_channel_changes_states_only_from_linked_positions(setup) -> None:
    adapter = cp.load_run(setup["runs"]["compose"], device="cpu")
    assert adapter.spans_fn is not None and adapter.info["min_subtokens"] == 1
    on, _, _, _ = adapter._forward(["we saw the cat and the dog", "nothing linked here at all"])
    adapter_off = cp.load_run(setup["runs"]["compose"], device="cpu"); adapter_off.spans_fn = None
    off, offsets, _, _ = adapter_off._forward(["we saw the cat and the dog", "nothing linked here at all"])
    cat = cp.read_index(offsets[0], 11, 14)
    assert torch.equal(on[0][:cat], off[0][:cat])                       # before the first injection: unchanged
    assert not torch.allclose(on[0][cat], off[0][cat])                  # at the linked span: changed
    assert torch.equal(on[1], off[1])                                   # no linked span: unchanged


def test_heldout_entries_are_linked_and_composed_at_probe_time(setup) -> None:
    adapter = cp.load_run(setup["runs"]["compose"], device="cpu")
    held = next(iter(adapter.heldout_entries))
    trained = TokenCorpus.open(setup["root"] / "train").spans["entry"]
    assert held not in set(trained.tolist())                            # never linked in training
    text = "they sold acetaminophen"
    assert adapter.link_targets([text], [(10, 23)]) == [[held]]
    assert adapter.statuses([text, "they sold the cat"], [(10, 23), (14, 17)]) == ["heldout", "mid"]
    off = cp.load_run(setup["runs"]["compose"], device="cpu"); off.spans_fn = None
    assert not torch.allclose(adapter.word_state([text], [(10, 23)]), off.word_state([text], [(10, 23)]))


def test_reused_probes_reproduce_the_b8_functions(setup) -> None:
    adapter = cp.load_run(setup["runs"]["compose"], device="cpu")
    root, settings = setup["probes"], cp.ProbeSettings()
    metrics, tables, _ = cp.probe_lambada(adapter, root, settings)
    reference = lambada(adapter, root / "lambada" / "data" / "lambada_test_en.jsonl")
    assert (metrics["accuracy"], metrics["word_loss"], metrics["n"]) == (reference["lambada_accuracy"], reference["lambada_word_loss"], reference["n"])
    metrics, _, _ = cp.probe_wic(adapter, root, settings)
    reference = wic(adapter, root / "wic")
    assert (metrics["prompt_accuracy"], metrics["probe_accuracy"]) == (reference["wic_prompt_accuracy"], reference["wic_probe_accuracy"])
    metrics, _, _ = cp.probe_card660(adapter, root, settings)
    assert metrics["spearman"] == word_similarity(adapter, load_card660(root / "card660.tsv"))["spearman"]


def test_token_logprobs_agree_with_full_logits(setup) -> None:
    adapter = cp.load_run(setup["runs"]["compose"], device="cpu")
    texts = ["the cat saw the river bank", "acetaminophen"]
    scores, _ = adapter.token_logprobs(texts)
    adapter.need_logits = True
    _, _, logits, ids = adapter._forward(texts)
    for score, logit, tokens in zip(scores, logits, ids):
        expected = torch.log_softmax(logit[:-1], -1)[torch.arange(len(tokens) - 1), torch.tensor(tokens[1:])]
        torch.testing.assert_close(score, expected)


def test_states_at_reads_the_rows_word_state_reads(setup) -> None:
    adapter = cp.load_run(setup["runs"]["compose"], device="cpu")
    texts = ["the cat saw the river bank", "acetaminophen in New York"]
    multi = cp.states_at(adapter, texts, [[(4, 7), (16, 26)], [(0, 13)]])
    single = adapter.word_state([texts[0], texts[0], texts[1]], [(4, 7), (16, 26), (0, 13)])
    torch.testing.assert_close(torch.cat(multi), single, atol=1e-5, rtol=1e-5)


def test_full_suite_is_deterministic_and_reports_heldout_subsets(setup, tmp_path: Path) -> None:
    outputs = []
    for name in ("a", "b"):
        output = tmp_path / name / "probes.json"
        cp.main(["--run", str(setup["runs"]["compose"]), "--probes-root", str(setup["probes"]), "--device", "cpu",
                 "--output", str(output)])
        outputs.append(output)
    first, second = (json.loads(o.read_text()) for o in outputs)
    assert first["summary"] == second["summary"] and first["probes"] == {**second["probes"], **{
        k: {**v, "seconds": first["probes"][k]["seconds"]} for k, v in second["probes"].items()}}
    assert cp.items_path(outputs[0]).read_bytes() == cp.items_path(outputs[1]).read_bytes()
    assert set(first["probes"]) == set(cp.PROBES)
    assert first["model"]["alias_table"]["checks"]["digest"] is True
    tables = cp.load_items(outputs[0])
    assert "heldout" in tables["wsd"]["status"] and "heldout" in tables["bless"]["status"]
    heldout = first["probes"]["wsd"]["subsets"]["wsd"]["heldout"]
    assert heldout["probe_f1"]["n"] == tables["wsd"]["status"].count("heldout")
    assert first["probes"]["wsd"]["metrics"]["backoff_items"] == 1        # "dog" never occurs in the SemCor fixture
    assert first["probes"]["wsd"]["metrics"]["mfs_f1"] == 0.8            # 6/6 "bank" tie → WordNet order → river sense
    with pytest.raises(FileExistsError):
        cp.main(["--run", str(setup["runs"]["compose"]), "--probes", "card660", "--probes-root", str(setup["probes"]),
                 "--device", "cpu", "--output", str(outputs[0])])


def test_paired_comparison_between_runs_with_identical_items(setup, tmp_path: Path) -> None:
    outputs = {}
    for mode in ("compose", "none"):
        outputs[mode] = tmp_path / mode / "probes.json"
        cp.main(["--run", str(setup["runs"][mode]), "--probes", "card660,bless,hyperlex,lambada",
                 "--probes-root", str(setup["probes"]), "--device", "cpu", "--output", str(outputs[mode])])
    same = cp.compare_outputs(outputs["compose"], outputs["compose"], resamples=200)
    assert same["tables"]["lambada"]["accuracy"]["all"]["difference"] == 0.0
    assert same["tables"]["card660"]["spearman_tied"]["all"]["ci_low"] == 0.0
    diff = cp.compare_outputs(outputs["compose"], outputs["none"], resamples=200)
    entry = diff["tables"]["hyperlex"]["cosine_spearman"]["all"]
    assert entry["ci_low"] <= entry["difference"] <= entry["ci_high"]
    assert "heldout" in diff["tables"]["bless"]["prompt_auc"] and "heldout" in diff["tables"]["lambada"]["accuracy"]
    other = tmp_path / "other" / "probes.json"
    cp.write_outputs(other, {}, {"card660": {"id": [0], "cosine": [0.1], "gold": [1.0], "status": [None]}}, {})
    with pytest.raises(ValueError, match="same items"):
        cp.compare_outputs(outputs["compose"], other)


def test_restore_composer_schedule_regrows_a_grown_dictionary() -> None:
    from vsa_embed.compose import FrameComposer, FrameSchedule
    from vsa_embed.span_channel import SpanChannel
    from vsa_embed.training.lm import _schedule_state
    schedule = FrameSchedule.from_frames([[(0, 0), (1, 1)], [(0, 2)]])
    build = lambda: SpanChannel(FrameComposer(schedule, 3, 2, 8, mode="attentive", key_dimension=4), 16, entry_count=2)
    torch.manual_seed(0)
    grown = build()
    grown.composer.add_atomics(torch.randn(2, 8)); grown.composer.add_relation_copies(torch.tensor([1]))
    grown.composer.set_schedule(FrameSchedule.from_frames([[(0, 3), (2, 1)], [(0, 4)]]))
    fresh = build()
    cp.restore_composer_schedule(fresh.composer, _schedule_state(grown))
    fresh.load_state_dict(grown.state_dict())
    ids = torch.tensor([0, 1])
    assert torch.equal(fresh.composer.compose(ids), grown.composer.compose(ids))


def test_grouped_softmax_equals_separate_fits() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(60, 5, generator=generator)
    groups = [i % 3 for i in range(60)]
    labels = [int(x[i, g] > 0) + (int(x[i, 4] > 0.5) if g == 2 else 0) for i, g in enumerate(groups)]
    counts = [2, 2, 3]
    joint = cp.fit_grouped_softmax(x, groups, labels, counts, l2=1e-2, max_elements=10**9)
    chunked = cp.fit_grouped_softmax(x, groups, labels, counts, l2=1e-2, max_elements=1)   # one group per chunk
    for (wa, ba), (wb, bb), g in zip(joint, chunked, range(3)):
        rows = [i for i in range(60) if groups[i] == g]
        pa, pb = (x[rows] @ wa.T + ba).argmax(-1), (x[rows] @ wb.T + bb).argmax(-1)
        assert torch.equal(pa, pb)
        torch.testing.assert_close(wa, wb, atol=1e-3, rtol=1e-3)
        assert (pa == torch.tensor([labels[i] for i in rows])).float().mean() > 0.8


def test_loaders_and_metrics(setup) -> None:
    rows = cp.load_bless(setup["probes"] / "bless" / "bless-gems" / "BLESS.txt")
    assert rows[0] == {"concept": "cat", "class": "class", "relation": "hyper", "relatum": "animal", "pos": "n"}
    hyperlex = cp.load_hyperlex(setup["probes"] / "hyperlex" / "hyperlex-all.txt")
    assert len(hyperlex) == 12 and hyperlex[0]["score"] == 5.8
    split = cp.hyperlex_split(setup["probes"] / "hyperlex")
    assert split[("cat", "animal", "N")] == "train" and split[("bank", "river", "N")] == "test"
    base = setup["probes"] / "wsd" / "WSD_Evaluation_Framework" / "Evaluation_Datasets" / "ALL"
    instances = load_wsd_instances(base / "ALL.data.xml", base / "ALL.gold.key.txt")
    assert instances[0].text[instances[0].start:instances[0].end] == "bank" and instances[0].keys == ("bank%1:17:01::",)
    assert cp.macro_f1(["a", "b", "a"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert cp.roc_auc([0.9, 0.1, 0.5], [True, False, False]) == 1.0
    assert cp.average_precision([0.9, 0.8, 0.1], [False, True, True]) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert cp.spearman_tied([1, 2, 3, 4], [10, 20, 20, 40]) == pytest.approx(0.9486833)
    assert cp.combine_status("unlinked", "rare", None) == "rare" and cp.combine_status("frequent", "heldout") == "heldout"
    assert cp.entry_status([5], {5}, None) == "heldout" and cp.entry_status([], set(), None) == "unlinked"
    assert cp.entry_status([1], set(), np.array([0, 150])) == "frequent"


def test_items_file_is_gzip_json_with_recorded_hash(tmp_path: Path) -> None:
    output = tmp_path / "probes.json"
    document = cp.write_outputs(output, {"x": {"metrics": {"m": 1.0}}}, {"card660": {"id": [0, 1]}}, {"model": {}})
    assert document["summary"] == {"x/m": 1.0}
    raw = cp.items_path(output).read_bytes()
    assert json.loads(gzip.decompress(raw))["tables"]["card660"]["id"] == [0, 1]
    cp.items_path(output).write_bytes(gzip.compress(b'{"tables": {}}'))
    with pytest.raises(ValueError, match="hash"):
        cp.load_items(output)
