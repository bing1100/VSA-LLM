"""WP-E7 end to end on a tiny synthetic C3 / host corpus: D7.0 → D7.1 → D7.2 (round 0 → 1) → D7.3 → report.

The host is a 1-layer GPT-2 (gpt2 tokenizer); generations are scripted, the teacher and the judge use
fake runners. The gold directory is hidden while discovery, linking, authoring and verification run."""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from vsa_embed.data.corpus import TokenCorpus
from vsa_embed.data.match_corpus import MatchCorpus
from vsa_embed.experiments import c3_corpus, cpt_plan, e7_authoring, e7_plan, e7_report, e7_round, host_corpus
from vsa_embed.ontologies.wordnet import FrameOntology
import vsa_embed.training.lm as lm


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")

LEMMAS = ["acetaminophen", "hydroxychloroquine", "electroencephalography", "photosynthesis", "mitochondrion", "chlorophyll",
          "thermodynamics", "kangaroo", "new york", "paracetamol", "ibuprofen", "nitroglycerin", "cat", "bank"]
LEXICON = {"machine": [0], "animal": [1], "drug": [2], "fruit": [3], "city": [4], "plant": [5]}
FILLER_OF = {i: next(k for k, v in LEXICON.items() if v == [i % 6]) for i in range(len(LEMMAS))}
NEW_TERMS = ["zorblax engine", "quintar valve", "glimmer fruit", "vorpal crank"]
WORDS = ["the", "a", "study", "of", "with", "and", "in", "showed", "that", "river", "people", "data"]


def fake_ontology(*_args, **_kwargs) -> FrameOntology:
    names = [f"{lemma.replace(' ', '_')}.n.{i:02d}" for i, lemma in enumerate(LEMMAS)]
    frames = [[(0, i % 6), (1, (i + 1) % 6)] for i in range(len(LEMMAS))]
    pairs = [(lemma, i) for i, lemma in enumerate(LEMMAS)]
    return FrameOntology("fake", names, ["hypernym", "pos"], [f"a{i}" for i in range(6)], frames, pairs, {"wordnet_version": "fake"})


def documents(count: int, offset: int) -> list[str]:
    rng = np.random.default_rng(offset)
    texts = []
    for i in range(count):
        body = []
        for _ in range(45):
            r = rng.random()
            body.append(LEMMAS[rng.integers(len(LEMMAS))] if r < 0.25 else
                        NEW_TERMS[rng.integers(len(NEW_TERMS))] if r < 0.4 else WORDS[rng.integers(len(WORDS))])
        extra = ["Machines such as the zorblax engine and pumps fail.", "Plants, including glimmer fruit, grow wild.",
                 "The quintar valve is a kind of machine."][i % 3]
        texts.append(f"Document {offset + i}. " + " ".join(body).capitalize() + ". " + extra)
    return texts


def scripted_generate(model, tokenizer, prompts, device, *, samples, temperature, top_p, max_new_tokens, batch, seed):
    """Host completions: the gold edge for ontology lemmas, a shared new filler for new terms, some noise."""
    out = []
    for prompt in prompts:
        if prompt.rstrip().endswith("Facts:"):
            out.append(["zorblax engine | is_a | machine\nacetaminophen | is_a | machine\nphotosynthesis | part_of | city"])
            continue
        if prompt.rstrip().endswith("Q:"):
            out.append([" What fails on farms?\nA: The zorblax engine.\nQ: What is a quintar valve?\nA: A kind of machine."])
            continue
        note_of = re.findall(r'Notes on "([^"]+)":$', prompt.rstrip())
        if note_of:
            text = f" {note_of[0].capitalize()} is a machine that farmers repair, according to the text above."
            out.append([text] * (samples if temperature > 0 else 1))
            continue
        if prompt.rstrip().endswith("Paragraph:"):
            out.append(["The zorblax engine is a machine used on farms; it drives the quintar valve and the river pumps.\n\nMore."])
            continue
        surface = re.findall(r"Concept: (.+)\n", prompt)[-1].strip()
        if surface in LEMMAS:
            text = f"is_a: {FILLER_OF[LEMMAS.index(surface)]}\npart_of: city"
        elif surface in NEW_TERMS:
            text = "is_a: machine\nis_a: gadget"
        else:
            text = "is_a: animal"
        out.append([text] * (samples if temperature > 0 else 1))
    return out, 100 * len(prompts) * samples, 10 * len(prompts) * samples


def teacher_runner(prompt, schema, model):
    concepts = []
    for cid, surface in re.findall(r'Concept (c\d+): "([^"]+)"', prompt):
        filler = FILLER_OF[LEMMAS.index(surface)] if surface in LEMMAS else "machine"
        concepts.append({"id": cid, "edges": [{"relation": "is_a", "filler": filler}]})
    return {"structured_output": {"concepts": concepts}, "total_cost_usd": 0.0, "modelUsage": {"fake-teacher": {}}}


def judge_runner(prompt, schema, model):
    return {"structured_output": {"verdict": "true" if "machine" in prompt else "partly"}, "total_cost_usd": 0.0}


def tiny_host(config=None) -> torch.nn.Module:
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=1, n_head=2))


ROUND = {"host": "tiny", "seq_len": 64, "base_tokens": 64 * 2 * 4, "round_tokens": 64 * 2 * 3, "sequences_per_step": 2,
         "channel_dimension": 16, "key_dimension": 8,
         "verification": {"window": 32, "batch": 4, "min_validation": 2, "resamples": 200},
         "entigraph": {"writer": "tiny", "per_entity": 2, "max_new_tokens": 16},
         "config_overrides": {"train": {"checkpoint_minutes": 30}},
         "cross": {"train_tokens": 64 * 2 * 3, "workers": 1, "seeds": [1],
                   "config_overrides": {"model": {"size": "tiny"}, "train": {"micro_batch": 2, "grad_accum": 1, "warmup_tokens": 64},
                                        "eval": {"batch": 2, "first_tokens": 128}}}}


@pytest.fixture(scope="module")
def e7(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e7")
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(c3_corpus, "build_wordnet_ontology", fake_ontology)
        patch.setattr(host_corpus, "build_wordnet_ontology", fake_ontology)
        patch.setattr(e7_authoring, "HOSTS", {"tiny": "gpt2"})
        patch.setattr(e7_authoring, "CONSUMER", "tiny")
        patch.setattr(e7_plan, "HOSTS", {"tiny": "gpt2"})
        patch.setitem(e7_plan.TRAIN, "tiny", 1000); patch.setitem(e7_plan.GENERATION, "tiny", 100)
        patch.setattr(e7_authoring, "wordnet_lexicon", lambda names: LEXICON)
        patch.setattr(e7_authoring, "generate_samples", scripted_generate)
        patch.setitem(cpt_plan.HOSTS, "tiny", {"pretrained": "gpt2", "corpus": "tiny", "width": 32, "vocab_size": 50257,
                                               "micro_batch": {"frozen": 2, "lora": 2}, "eval_batch": 2})
        patch.setattr(lm, "build_model", tiny_host)
        shards = []
        for k, (count, offset) in enumerate([(60, 0), (140, 1000)]):
            path = root / f"shard{k}.parquet"
            pq.write_table(pa.table({"text": documents(count, offset)}), path)
            shards.append(str(path))
        c3_config = {"experiment": "c3-test", "seed": 7, "tokenizer": "gpt2", "workers": 1,
                     "paths": {"data_root": str(root / "c3-data"), "shards": shards},
                     "ontology": {"max_atomics": 16, "max_degree": 4},
                     "data": {"eval_docs": 10, "presample_docs": 30, "holdout_fraction": 0.2, "holdout_min_count": 1,
                              "min_subtokens": 2, "train_tokens": 4000, "train_min_subtokens": 2, "l1_slice_tokens": 500,
                              "cardinality_docs": 5},
                     "cardinality_tokenizers": ["gpt2"]}
        c3_corpus.run(c3_config, root / "c3-run")
        host_corpus.run({"host": "tiny", "tokenizer": "gpt2", "models": ["gpt2"], "c3_run": str(root / "c3-run"), "workers": 1,
                         "paths": {"data_root": str(root / "host")},
                         "data": {"train_tokens": 6000, "min_subtokens": 2, "eval_min_subtokens": 1}}, root / "host-run")
        run = root / "e7-run"
        config = {"c3_run": str(root / "c3-run"), "host_root": str(root / "host"), "data_root": str(root / "e7-data"),
                  "read_start": 100, "mask": {"fraction": 0.5, "min_count": 1},
                  "corpora": {"general_tokens": 3000, "replay_tokens": 800, "read_tokens": 3500, "test_tokens": 2500, "workers": 1},
                  "discovery": {"min_count": 3, "keep": 400, "max_occurrences": 6},
                  "authoring": {"candidates": 400, "contexts": 2, "samples": 2, "max_validation": 6},
                  "direct": {"documents": 6}}
        summary = e7_authoring.prepare("general", e7_authoring._merge(e7_authoring.DEFAULTS, config), run)
        track = e7_authoring.load_track(run)
        gold_dir = Path(track["data_root"]) / "gold"
        hidden = gold_dir.with_name("gold.hidden")
        gold_dir.rename(hidden)                                       # gold-audit files out of reach
        model, tokenizer = tiny_host().eval(), transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        cpu = torch.device("cpu")
        e7_authoring.discover(run, "tiny", device=cpu, model=model, tokenizer=tokenizer)
        e7_authoring.link(run)
        for name in ("tiny", "hearst", "random", "direct-tiny"):
            e7_authoring.author(run, name, device=cpu, model=model, tokenizer=tokenizer)
        e7_authoring.author(run, "teacher", device=cpu, runner=teacher_runner)
        e7_round.update_round_settings(run, ROUND)
        hidden.rename(gold_dir)          # round 0 is training: its evaluation strata may use the gold linker
        e7_round.train_base(run, 1)
        gold_dir.rename(hidden)
        verified = {name: e7_round.verify(run, 1, name, device=cpu) for name in ("tiny", "teacher")}
        hidden.rename(gold_dir)
        quality = e7_authoring.quality(run)
        e7_authoring.judge_items(run, per_author=5, calibration=4)
        judged = e7_authoring.judge(run, fake=judge_runner)
        e7_round.entigraph(run, device=cpu, model=model, tokenizer=tokenizer)
        e7_round.spa(run, device=cpu, model=model, tokenizer=tokenizer)
        e7_round.notes(run, device=cpu, model=model, tokenizer=tokenizer)
        notes_verified = e7_round.verify_notes(run, 1, device=cpu)
        for condition in e7_round.CONDITIONS + e7_round.OPTIONAL_CONDITIONS:
            e7_round.train_condition(run, 1, condition)
        e7_round.cross_prepare(run)
        for condition in e7_round.CROSS_CONDITIONS:
            e7_round.cross_train(run, 1, condition)
        report = e7_report.write_report(run, resamples=200)
        yield {"root": root, "run": run, "track": track, "summary": summary, "quality": quality, "judged": judged,
               "verified": verified, "notes": notes_verified, "report": report, "patch": patch}
    finally:
        patch.undo()


def test_prepare_masks_concepts_and_keeps_their_frames_out_of_the_visible_ontology(e7) -> None:
    track, data = e7["track"], Path(e7["track"]["data_root"])
    hidden = json.loads((data / "gold" / "gold.json").read_text())
    names = (e7["run"] / "masked_concepts.txt").read_text().split()
    import hashlib
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == hidden["sha256"] == track["masked_sha256"]
    assert hidden["masked_entries"]
    visible_ontology = torch.load(data / "visible" / "ontology.pt", weights_only=False)
    offsets = visible_ontology["offsets"]
    assert all(int(offsets[e + 1]) == int(offsets[e]) for e in hidden["masked_entries"])
    view = json.loads((data / "visible" / "visible.json").read_text())
    assert not set(view["base"]) & set(hidden["masked_strings"])
    assert all(e not in hidden["masked_entries"] for e in view["base"].values())
    ranges = track["documents"]
    assert ranges["general"][1] <= ranges["replay"][0] <= ranges["replay"][1] <= ranges["read"][0] == 100 <= ranges["read"][1] <= ranges["test"][0]
    strings = [MatchCorpus.open(data / "match" / s).strings for s in ("read", "test", "replay")]
    assert strings[0] == strings[1] == strings[2]


def test_gold_is_never_read_by_discovery_authoring_or_acceptance(e7) -> None:
    """The fixture ran discovery, linking, every author and verification with gold/ renamed away."""
    run = e7["run"]
    for name in ("tiny", "hearst", "random", "direct-tiny", "teacher"):
        assert (run / "proposals" / f"{name}.json").exists()
    assert set(e7["verified"]) == {"tiny", "teacher"}
    hidden = json.loads((Path(e7["track"]["data_root"]) / "gold" / "gold.json").read_text())
    candidates = {c["surface"] for c in json.loads((run / "authoring_set.json").read_text())["candidates"]}
    assert candidates & set(hidden["masked_strings"]), "some masked concepts should be discovered"
    prompt = json.loads((run / "proposals" / "tiny.json").read_text())
    assert "examples" in prompt and all("hypernym" not in ex["prompt"] for ex in prompt["examples"])


def test_authoring_and_validation_contexts_are_disjoint(e7) -> None:
    run, data = e7["run"], Path(e7["track"]["data_root"])
    read = MatchCorpus.open(data / "match" / "read")
    aset = json.loads((run / "authoring_set.json").read_text())
    checked = 0
    for item in aset["candidates"]:
        authoring = set(item["authoring_documents"])
        documents = read.document_of(np.asarray([end for _, end in item["validation"]], dtype=np.int64))
        assert not authoring & set(documents.tolist())
        checked += len(item["validation"])
    assert checked > 0
    proposals = json.loads((run / "proposals" / "tiny.json").read_text())
    for item in aset["candidates"]:
        assert set(proposals["candidates"][item["surface"]]["contexts"]) <= set(item["authoring_documents"])


def test_quality_scores_authors_against_the_hidden_gold(e7) -> None:
    quality = e7["quality"]
    assert {"tiny", "teacher", "hearst", "random", "direct-tiny"} <= set(quality["authors"])
    tiny = quality["authors"]["tiny"]
    assert tiny["masked_concepts_authored"] > 0 and tiny["precision"]["value"] == pytest.approx(0.5)   # gold edge + one wrong
    assert tiny["recall"]["value"] == pytest.approx(1.0)            # the hypernym; `pos` is not an authoring relation
    assert quality["authors"]["teacher"]["precision"]["value"] == pytest.approx(1.0)
    assert quality["discovery"]["tiny"]["discoverable"] > 0
    judged = e7["judged"]
    assert judged["items"] > 0 and judged["fleiss_kappa"] == pytest.approx(1.0)
    assert (e7["run"] / "judge" / "rubric.md").exists()
    items = (e7["run"] / "judge" / "items.jsonl").read_text()
    assert '"system"' not in items and "tiny" not in json.loads(items.splitlines()[0])["fields"].values()


def test_verification_cards_and_new_atoms(e7) -> None:
    verified = e7["verified"]["tiny"]
    assert verified["new_atoms"] == ["gadget"]                      # proposed for ≥ 2 new terms → allocated (M3)
    assert verified["summary"]["proposed_edges"] > 0
    cards = (e7["run"] / "round1" / "s1" / "cards-tiny.jsonl").read_text().splitlines()
    assert cards and {"surface", "relation", "filler", "utility_mean", "utility_low", "accepted", "round"} <= set(json.loads(cards[0]))
    assert all(len(frame) <= len(verified["noverify"][k]) for k, frame in verified["accepted"].items())


def test_conditions_materialize_their_linkers_frames_and_initial_state(e7) -> None:
    data = Path(e7["track"]["data_root"]) / "round1" / "s1"
    hidden = json.loads((Path(e7["track"]["data_root"]) / "gold" / "gold.json").read_text())
    entry_base = e7["track"]["entry_count"]
    ontologies = {c: torch.load(data / c / "ontology.pt", weights_only=False) for c in e7_round.CONDITIONS}
    gold_onto = ontologies["gold"]
    for entry in hidden["masked_entries"]:
        assert int(gold_onto["offsets"][entry + 1]) > int(gold_onto["offsets"][entry])
    for condition in ("cm", "entigraph"):
        assert all(int(ontologies[condition]["offsets"][e + 1]) == int(ontologies[condition]["offsets"][e])
                   for e in range(entry_base, ontologies[condition]["entry_count"]))
    self_degrees = np.diff(ontologies["self"]["offsets"].numpy())[entry_base:]
    random_degrees = np.diff(ontologies["random"]["offsets"].numpy())[entry_base:]
    assert np.array_equal(self_degrees, random_degrees)
    spans = {c: TokenCorpus.open(data / c / "train").spans for c in e7_round.CONDITIONS}
    linked_new = lambda c: int((spans[c]["entry"] >= entry_base).sum())
    assert linked_new("cm") == 0 and linked_new("gold") == 0 and linked_new("selfnv") >= linked_new("self")
    assert int(np.isin(spans["gold"]["entry"], hidden["masked_entries"]).sum()) > 0
    assert int(np.isin(spans["self"]["entry"], hidden["masked_entries"]).sum()) == 0
    init = torch.load(data / "self" / "init_state.pt", weights_only=False)
    assert init["model"]["channel.composer.atomics"].shape[0] == ontologies["self"]["atomic_count"]
    assert ontologies["self"]["atomic_names"][-1] == "authored:gadget"


def test_compute_matched_control_adds_the_authoring_flops_as_tokens(e7) -> None:
    data = Path(e7["track"]["data_root"]) / "round1" / "s1"
    cm = json.loads((data / "cm" / "materialized.json").read_text())["compute_matched"]
    flops = sum(e["flops"] for e in cm["ledger"] if e["stage"] in cm["stages"])
    parameters = next(e["parameters"] for e in cm["ledger"] if e["stage"] == "verification")
    assert cm["extra_tokens"] == int(np.ceil(flops / (4 * parameters)))                 # frozen host: 4N per training token
    assert cm["total_tokens"] == ROUND["round_tokens"] + cm["extra_tokens"]
    run = e7["run"] / "train"
    config = lambda c: __import__("yaml").safe_load((run / f"tiny-{c}-s1" / "resolved_config.yaml").read_text())
    for condition in ("cm", "entigraph", "notes", "spa", "verbal"):
        assert config(condition)["train"]["total_tokens"] == cm["total_tokens"]          # text controls at matched tokens
    assert config("self")["train"]["total_tokens"] == config("selfrand")["train"]["total_tokens"] == ROUND["round_tokens"]
    budget = json.loads((data / "notes" / "materialized.json").read_text())["text_budget"]
    assert {e["stage"] for e in budget["ledger"]} == {"discovery", "notes", "verification"} and budget["side_flops"] > 0
    assert json.loads((data / "verbal" / "materialized.json").read_text())["text_budget"]["side_flops"] == pytest.approx(flops)


def test_text_controls_carry_their_synthetic_text_and_no_new_frames(e7) -> None:
    data = Path(e7["track"]["data_root"]) / "round1"
    entry_base = e7["track"]["entry_count"]
    notes = e7["notes"]
    assert notes["notes"] > 0 and 0 <= notes["kept"] <= notes["notes"] and len(notes["records"]) == notes["notes"]
    assert all(r["accepted"] == (r["low"] > 0) for r in notes["records"])
    verbal = json.loads((e7["run"] / "round1" / "s1" / "verbal.json").read_text())
    cards = [json.loads(line) for line in (e7["run"] / "round1" / "s1" / "cards-tiny.jsonl").read_text().splitlines()]
    assert verbal["sentences"] == sum(c["accepted"] for c in cards)
    for condition in ("entigraph", "spa", "verbal", "notes"):
        info = json.loads((data / "s1" / condition / "materialized.json").read_text())
        produced = {"verbal": verbal["tokens"], "notes": notes["tokens"]}.get(condition, 1)
        if produced:
            assert info["synthetic"] and info["mix"][-1]["repeats"] >= 1 and info["entries_with_new_frames"] == 0
        else:                                   # nothing verified: the control trains on plain text at the same tokens
            assert info["synthetic"] is None and info["synthetic_empty"] and info["text_budget"]["total_tokens"] > 0
        spans = TokenCorpus.open(data / "s1" / condition / "train").spans
        assert int((spans["entry"] >= entry_base).sum()) == 0
    selfrand = json.loads((data / "s1" / "selfrand" / "materialized.json").read_text())
    assert selfrand["selfrand"]["accepted_edges"] == sum(len(f) for f in e7["verified"]["tiny"]["accepted"].values())


def test_round_runs_share_reference_strata_and_report(e7) -> None:
    run = e7["run"] / "train"
    counts = {}
    for condition in e7_round.CONDITIONS:
        windows = lm.load_window_losses(run / f"tiny-{condition}-s1" / "eval_windows.npz")
        sums, count = windows["evals"][max(windows["evals"])]
        counts[condition] = {s: count[i].tolist() for i, s in enumerate(windows["strata"]) if s.startswith("ref_")}
    reference = counts["cm"]
    assert {"ref_masked", "ref_unlinked", "ref_authored_new_self", "ref_authored_new_teacher"} <= set(reference)
    assert all(counts[c] == reference for c in counts)
    report = e7["report"]
    assert report["loss_gaps"] and report["gate"] is not None and report["cross_gaps"]
    assert {"self_beats_cm", "self_beats_random", "locality", "verification_improves_precision"} <= set(report["gate"])
    text = (e7["run"] / "report-r6" / "report.md").read_text()
    assert "D7.1 authoring quality" in text and "Compute-matched control" in text and "D7.3 cross-authoring" in text


def test_command_lines_reuse_finished_steps(e7, capsys) -> None:
    run = str(e7["run"])
    assert e7_authoring.main(["quality", "--run", run, "--device", "cpu"]) == 0
    assert e7_authoring.main(["train", "--run", run, "--device", "cpu", "--seed", "1", "--condition", "self"]) == 0
    assert "finished" in capsys.readouterr().out
    assert e7_round.main(["verify", "--run", run, "--seed", "1", "--author", "teacher", "--device", "cpu"]) == 0
    assert e7_plan.main(["--run", run, "--stage", "d73", "--seeds", "1"]) == 0
    plan = json.loads((e7["run"] / "plan" / "d73.json").read_text())
    assert plan["priority"] == 60 and len(plan["jobs"]) == 3 and plan["gpu_hours"] > 0
    assert e7_round.main(["materialize", "--run", run, "--seed", "1", "--condition", "cm", "--set", '{"note": "x"}']) == 0
    assert json.loads((e7["run"] / "round.json").read_text())["note"] == "x"


def test_plan_lists_and_queues_every_stage(e7, tmp_path: Path) -> None:
    run = e7["run"]
    for stage in ("d71", "d72", "d73"):
        jobs, then = e7_plan.stage_jobs(run, stage, seeds=[1, 2, 3])
        assert jobs and all(j["gpu_hours"] >= 0 for j in jobs)
        if stage == "d72":
            names = [j["name"] for j in jobs]
            assert names.index(f"e7-{run.name}-base-s1") < names.index(f"e7-{run.name}-verify-teacher-s1") \
                < names.index(f"e7-{run.name}-round1-self-s1")
            assert len([n for n in names if "-round1-" in n]) == 30 and f"e7-{run.name}-verify-notes-s3" in names
    jobs, _ = e7_plan.stage_jobs(run, "d73", seeds=[1, 2, 3])
    queued = e7_plan.queue(jobs, queue_dir=tmp_path / "jobs")
    record = json.loads((tmp_path / "jobs" / f"{queued[0]}.json").read_text())
    assert len(queued) == 9 and record["priority"] == 60 and record["env"] == {"PYTHONPATH": "src"}
