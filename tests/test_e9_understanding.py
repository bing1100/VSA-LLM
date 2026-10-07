"""E9 frequency bias (`e9_freqbias`: frequency bins, the strict non-copy stratum, exact replay of the trainer's strata,
slope / gap analysis, the row probe) and the understanding items (`e9_understanding`: builders, evaluation, analysis) on a
T5-like toy glossary with a tiny fake host (CPU)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

transformers = pytest.importorskip("transformers")

from vsa_embed.data.corpus import TokenCorpus, build_corpus
from vsa_embed.evaluation import channel_probes as cp
from vsa_embed.experiments import e9_freqbias as fb
from vsa_embed.experiments import e9_ontology_edit as edit
from vsa_embed.experiments import e9_report
from vsa_embed.experiments import e9_rescore
from vsa_embed.experiments import e9_understanding as und
from vsa_embed.experiments import e5_common as common
from vsa_embed.experiments.e9_tracks import TrackLexicon
from vsa_embed.row_sources import FillerIndex, frames_digest
from vsa_embed.span_channel import AliasTable
from vsa_embed.tracks.common import RelationTemplates
import vsa_embed.training.lm as lm


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")

# name → (type, area, owner, uses, reports_to)
GLOSSARY = {
    "Zash Team": ("team", "finance", None, None, "Brim Division"),
    "Varkt Crew": ("team", "sales", None, None, "Tolk Office"),
    "Quill Squad": ("team", "finance", None, None, "Morv Group"),
    "Brim Division": ("department", "finance", None, None, None),
    "Tolk Office": ("department", "sales", None, None, None),
    "Morv Group": ("department", "finance", None, None, None),
    "Brightwater Ledger": ("system", "finance", "Zash Team", "Grosh Console", None),
    "Grosh Console": ("system", "sales", "Varkt Crew", None, None),
    "Plurb Standard": ("policy", "sales", "Varkt Crew", None, None),
    "Skolt Handoff": ("process", "finance", "Quill Squad", "Grosh Console", None),
    "Noxel Index": ("metric", "finance", "Zash Team", None, None),
    "Morbel Feed": ("dataset", "sales", "Zash Team", None, None),
    "Quarn Portal": ("system", "finance", "Varkt Crew", "Brightwater Ledger", None),
    "Dribbet Sync": ("process", "sales", "Zash Team", None, None),
    "Terb Template": ("document", "finance", "Quill Squad", None, None),
}
TERMS = list(GLOSSARY)
HELDOUT = [TERMS.index(t) for t in ("Quarn Portal", "Dribbet Sync", "Terb Template")]
RELATIONS = ["is_a", "area", "owned_by", "uses", "reports_to"]
TYPES = ["system", "process", "policy", "metric", "document", "dataset", "team", "department"]
ATOMS = [f"type:{t}" for t in TYPES] + ["area:finance", "area:sales"] + [f"term:{t}" for t in TERMS]
A = {a: i for i, a in enumerate(ATOMS)}
ARTICLE = {"team", "department"}
SEQ = 48
TEMPLATES = {
    "owned_by": RelationTemplates(["{x} is owned by", "Questions about {x} go to"], "{x} is owned by {y}."),
    "reports_to": RelationTemplates(["{x} reports to", "{x} sits within"], "{x} reports to {y}."),
    "area": RelationTemplates(["{x} belongs to the", "In the org chart, {x} sits in the"], "{x} belongs to the {y} area.", " {y} area"),
    "is_a": RelationTemplates(["The kind of thing {x} is: a", "In the glossary, {x} is filed under the type"], "{x} is a {y}."),
    "uses": RelationTemplates(["{x} uses", "{x} relies on"], "{x} uses {y}."),
}
TOY_SPEC = {
    "anchor": {"relation": "is_a", "values": [f"type:{t}" for t in TYPES[:6]]},
    "two_hop": [{"path": ("owned_by", "reports_to"), "distractors": 1,
                 "templates": ["{x} is owned by a team that reports to", "The team that owns {x} sits within"]}],
    "affordance": {"relation": "is_a", "items": {"use": {"templates": ["Most people interact with {x} by"], "options": {
        "type:system": " logging in to it", "type:process": " carrying out its steps", "type:policy": " complying with it",
        "type:metric": " checking its value", "type:document": " reading it", "type:dataset": " querying its rows"}}}},
    "paraphrase": {"area": {"templates": ["{x} is mainly concerned with"], "k": 2, "options": {
        "area:finance": [" money and accounting"], "area:sales": [" selling to clients"]}}},
    "reverse": {"owned_by": {"cue": "owned by {t}", "match": None}},
    "reverse_templates": ["Of {x} and {y}, the one {c} is", "Between {y} and {x}, the one {c} is"],
    "comparison": {"owned_by": {"templates": ["{x} and {y} are owned by"], "candidates": [" the same team", " two different teams"]}},
    "negation": {"owned_by": {"affirm": ["{x} is owned by"], "negate": ["{x} is not owned by"]}},
}


def _display(name: str) -> str:
    return f"the {name}" if GLOSSARY[name][0] in ARTICLE else name


def _frame(term: str) -> list[tuple[int, int]]:
    kind, area, owner, uses, reports = GLOSSARY[term]
    edges = [(0, A[f"type:{kind}"]), (1, A[f"area:{area}"])]
    edges += [(2, A[f"term:{owner}"])] if owner else []
    edges += [(3, A[f"term:{uses}"])] if uses else []
    edges += [(4, A[f"term:{reports}"])] if reports else []
    return edges


def _sentences(term: str) -> list[str]:
    kind, area, owner, uses, reports = GLOSSARY[term]
    out = [f"{term}: a {area} {kind}" + (f", owned by {_display(owner)}" if owner else "") + "."]
    out += [f"{_display(term)[:1].upper() + _display(term)[1:]} reports to {_display(reports)}."] if reports else []
    out += [f"{term} uses {uses}."] if uses else []
    return out


def fake_host(config=None) -> torch.nn.Module:
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=1, n_head=2))


def _config(root: Path, name: str, channel: dict, host_mode: str = "lora", **train) -> dict:
    config = {"seed": 1, "device": "cpu", "experiment": f"e9-u-fake-{host_mode}-{name}-s1",
              "model": {"size": "pretrained", "seq_len": SEQ, "pretrained": "fake-host", "host_mode": host_mode, "lora_rank": 2},
              "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": SEQ * 2 * 4, "lr": 3e-3, "warmup_tokens": 64,
                        "log_every": 2, "save_trainable_only": True, "host_lr": 1e-3, **train},
              "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
              "eval": {"windows": 6, "batch": 3, "first_tokens": 128, "save_window_losses": True},
              "channel": {"dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}}
    if host_mode == "frozen":
        config["train"]["eval_only"] = True
    return config


def _lexicon() -> TrackLexicon:
    texts = {f"term:{t}": _display(t) for t in TERMS}
    texts.update({a: a.split(":", 1)[1] for a in ATOMS if not a.startswith("term:")})
    types = {f"term:{t}": f"term:{GLOSSARY[t][0]}" for t in TERMS}
    return TrackLexicon("toy", TEMPLATES, texts, types, category_relations=("is_a",), kept_relations=frozenset({"is_a"}),
                        edit_relations=("owned_by",))


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    patch = pytest.MonkeyPatch()
    patch.setattr(lm, "build_model", fake_host)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("e9u")
    full = AliasTable.from_pairs([(t, i) for i, t in enumerate(TERMS)], holdout=HELDOUT, include_holdout=True)
    rng = random.Random(0)
    weights = [8 if t in {"Brightwater Ledger", "Zash Team", "Grosh Console"} else 1 for t in TERMS]
    texts = [" ".join(s for t in rng.choices(TERMS, weights, k=5) for s in _sentences(t)) for _ in range(140)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=60_000,
                 batch_texts=8, workers=1)
    build_corpus(texts[:50], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=60_000, batch_texts=8, workers=1)
    schedule = full.entry_schedule([_frame(t) for t in TERMS])
    entries = len(full.entry_concepts)
    frequency = np.bincount(TokenCorpus.open(root / "train").spans["entry"], minlength=entries)
    ontology = {"entry_count": entries, "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": TERMS, "relation_names": RELATIONS, "atomic_names": ATOMS}
    torch.save(ontology, root / "ontology.pt")
    cp.save_alias_table(full, root / "alias_table.json")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    lexicon = _lexicon()
    e9_rescore.build_filler_table(ontology, full, tokenizer, root / "fillers.pt", lexicon=lexicon)
    fb.build_exclusion_table(ontology, full, tokenizer, root / "exclusions.pt", lexicon=lexicon, track="toy")
    runs = {}
    specs = {"P0": ({"mode": "none"}, "frozen"), "C0p": ({"mode": "none"}, "lora"), "C2": ({"mode": "free", "free_dimension": 8}, "lora"),
             "C5": ({"mode": "compose", "composition": "attentive", "context_window": 4}, "lora")}
    for name, (channel, mode) in specs.items():
        folder = root / "runs" / f"fake-{mode}-{name}-s1"
        lm.train(_config(root, name, channel, mode), folder)
        runs[name] = folder
    yield {"root": root, "table": full, "ontology": ontology, "tokenizer": tokenizer, "lexicon": lexicon, "runs": runs,
           "fillers": root / "fillers.pt", "exclusions": root / "exclusions.pt"}
    patch.undo()


# -- 1. bins, words, the strict stratum ------------------------------------------------------------------------------------

def test_bins_nest_the_trainer_strata() -> None:
    bins = fb.scheme_bins("halfdecade")
    assert [b for b, _, _ in bins][:3] == ["after_f1_3", "after_f4_9", "after_f10_31"] and bins[-1] == ("after_f3163plus", 3163, None)
    assert fb.nesting(bins) == {"after_rare_seen": ["after_f1_3", "after_f4_9"], "after_mid": ["after_f10_31", "after_f32_99"],
                                "after_frequent": ["after_f100_316", "after_f317_999", "after_f1000_3162", "after_f3163plus"]}
    assert all(not v for v in fb.nesting(fb.scheme_bins("log2")).values())
    assert fb.bin_index(0, bins) is None and bins[fb.bin_index(9, bins)][0] == "after_f4_9" and bins[fb.bin_index(5000, bins)][1] == 3163
    assert fb.content_words("the Zash Team, tier 1") == ["Zash", "Team", "tier"]
    assert und.stem("running") == und.stem("runs") == "run" and und.stem("weekly") == und.stem("weeks")
    assert und.overlaps(" checking its weekly value", und.word_stems(["weeks"])) and not und.overlaps(" the anode", {"cathode"})


def _toy_strict() -> tuple[fb.StrictIndex, dict]:
    # entries 0 (frequency 2) and 1 (held out); entry 0's frame holds atomic 0, whose concept is entry 2 (frame: atomic 1).
    ontology = {"offsets": np.array([0, 1, 2, 3]), "relations": np.array([0, 0, 0]), "fillers": np.array([0, 2, 1])}
    data = {"frames_sha256": frames_digest(ontology), "sequences": [[[7, 8]], [[9]], [[11]], [[12]], [[50]]],
            "atomic_surfaces": [[0], [1], [4]], "entry_surfaces": [[3], [], []], "relation_surfaces": [2], "atomic_entries": [[2], [], []]}
    return fb.StrictIndex(data, ontology), ontology


def test_strict_index_closure_and_masks() -> None:
    strict, ontology = _toy_strict()
    assert strict.closure(0) == {0, 1} and strict.closure(1) == {2}                 # two hops: 0 → entry 2 → atomic 1
    ids = torch.tensor([[1, 2, 7, 8, 7, 8, 5, 9, 11, 12, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4]])
    spans = {"batch": torch.tensor([0, 0]), "start": torch.tensor([1, 6]), "end": torch.tensor([3, 6]), "inject": torch.tensor([3, 6]),
             "entry": torch.tensor([0, 1]), "length": torch.tensor([3, 1]), "confidence": torch.ones(2)}
    fillers = FillerIndex(ontology["offsets"], ontology["fillers"], [[[7, 8]], [[9]], [[50]]])
    bins = fb.scheme_bins("halfdecade")
    masks, terms, checks = fb.frequency_masks(ids, spans, np.array([2, 0, 0]), {1}, fillers=fillers, strict=strict, bins=bins)
    targets = lambda name: sorted((masks[name][0].nonzero().flatten() + 1).tolist())
    assert checks == {"replay": True, "nested": True}
    # span A (entry 0): targets 4..11; hop 1 (7, 8) at 4–5, hop 2 (9) at 7, relation wording (11) at 8, own alias (12) at 9
    assert targets("after_rare_seen") == list(range(4, 12)) == targets("after_f1_3")
    assert targets("after_rare_seen_strict") == [6, 10, 11] == targets("after_f1_3_strict")
    assert targets("after_f1_3_filler") == [4, 5]                                  # the filler rule counts hop 1 only
    # span B (held out, entry 1): targets 7..14; only relation wording (11) and its own hop-1 atomic (50: absent) count
    assert targets("after_heldout_strict") == [7, 9, 10, 11, 12, 13, 14]
    assert targets("after_strict") == [6, 10, 11, 12, 13, 14]                      # union rule: excluded if excluded for any span
    assert targets("after_f4_9") == [] and "after_f4_9_strict" in masks
    own = {t["entry"]: sorted((t["own"].nonzero().flatten() + 1).tolist()) for t in terms}
    assert own == {0: list(range(4, 12)), 1: list(range(7, 15))}
    plain = lm.stratum_masks(ids, spans, np.array([2, 0, 0]), {1}, fillers)
    assert all(torch.equal(plain[k], masks[k]) for k in plain)                     # the trainer's strata, unchanged


def test_exclusion_table_of_the_toy_glossary(world) -> None:
    data = torch.load(world["exclusions"], weights_only=False)
    strict = fb.StrictIndex(data, world["ontology"])
    ledger = world["table"].alias_to_entry["brightwater ledger"]                   # entries are the alias table's, not concepts
    words = {data["surfaces"][i] for i in strict.surface_ids(ledger)}
    assert {"Brightwater Ledger", "Zash Team", "Zash", "Brim Division", "Brim", "Grosh Console", "finance", "system"} <= words
    assert "owned" in words and "Tolk" not in words                    # relation wording; Varkt Crew's department is 3 hops away
    assert "Varkt Crew" in words                                       # Grosh Console (hop 1) is owned by the Varkt Crew (hop 2)


# -- 2. rescoring a run: exact replay, terms, analysis ------------------------------------------------------------------------

def test_score_run_replays_the_trainer_and_writes_terms(world, tmp_path) -> None:
    for name in ("C5", "C0p"):
        record = fb.score_run(world["runs"][name], tmp_path / name, variants=["ref", "ref-off"], fillers=world["fillers"],
                              exclusions=world["exclusions"], device="cpu")
        check = record["ref_check"]
        assert check["paired"] and check["counts_equal"] and check["max_abs_window_sum_diff"] == 0.0
        assert all(v["nested"] for v in record["variants"].values())
    c5, c0 = fb.load_scores(tmp_path / "C5"), fb.load_scores(tmp_path / "C0p")
    assert sorted(c5["sums"]) == ["ref", "ref-off"] and sorted(c0["sums"]) == ["ref"]
    assert c5["strata"] == c0["strata"] and np.array_equal(c5["count"], c0["count"])            # model-independent masks
    for stratum in ("after_heldout_strict", "after_strict", "after_f1_3", "after_f10_31_strict", "after_rare_seen_filler"):
        assert stratum in c5["strata"]
    i = c5["strata"].index
    assert (c5["count"][i("after_strict")] <= c5["count"][i("after_nonfiller")]).all()          # strict ⊆ non-filler
    terms = c5["terms"]
    assert np.array_equal(terms["window"], c0["terms"]["window"]) and (terms["strict_count"] <= terms["count"]).all()
    # every window's entry-level own targets cover its `after` targets (overlaps are counted once per entry)
    per_window = np.bincount(terms["window"], weights=terms["count"], minlength=c5["count"].shape[1])
    assert (per_window >= c5["count"][i("after")]).all()
    assert c5["record"]["bin_support"]["after_heldout"]["targets"] == int(c5["count"][i("after_heldout")].sum())
    with pytest.raises(FileExistsError):
        fb.score_run(world["runs"]["C0p"], tmp_path / "C0p", fillers=world["fillers"], exclusions=world["exclusions"], device="cpu")
    again = fb.score_run(world["runs"]["C0p"], tmp_path / "C0p", fillers=world["fillers"], exclusions=world["exclusions"], device="cpu",
                         resume=True)
    assert sorted(again["variants"]) == ["ref"] and (tmp_path / "C0p" / "manifest.json").exists()


def _synthetic_scores(losses: dict[str, float], windows: int = 40, seed: int = 0) -> dict:
    """A `load_scores`-shaped rescoring whose bins have the given mean losses (counts shared by every model)."""
    rng = np.random.default_rng(seed)
    bins = fb.scheme_bins("halfdecade")
    strata = [b for b, _, _ in bins] + ["after_unseen", "after_heldout", "after_rare_seen", "after_frequent"]
    counts = np.random.default_rng(99).integers(20, 60, size=(len(strata), windows)).astype(np.int32)
    counts[strata.index("after_rare_seen")] = counts[0] + counts[1]
    counts[strata.index("after_frequent")] = counts[4:8].sum(0)
    value = np.asarray([losses[s] for s in strata])[:, None]
    sums = value * counts + rng.normal(0, 0.01, size=counts.shape) * counts
    for coarse, parts in (("after_rare_seen", [0, 1]), ("after_frequent", [4, 5, 6, 7])):
        sums[strata.index(coarse)] = sums[parts].sum(0)
    entries = np.arange(80)
    frequency = np.r_[np.full(10, 2), np.full(10, 6), np.full(10, 20), np.full(10, 50), np.full(10, 200), np.full(10, 500),
                      np.full(10, 2000), np.full(5, 5000), np.full(5, 0)]
    heldout = np.r_[np.zeros(75, bool), np.ones(5, bool)]
    window = np.repeat(np.arange(windows), 2)[:80]
    loss_of = lambda f, h: losses["after_heldout"] if h else losses[bins[fb.bin_index(int(f), bins)][0]] if f else losses["after_unseen"]
    term_sum = np.asarray([loss_of(f, h) * 10 for f, h in zip(frequency, heldout)])
    terms = {"window": window.astype(np.int32), "entry": entries, "count": np.full(80, 10, np.int32), "frequency": frequency,
             "heldout": heldout, "sum_ref": term_sum + rng.normal(0, 0.05, 80), "filler_count": np.zeros(80, np.int32),
             "strict_count": np.zeros(80, np.int32)}
    support = {b: {"targets": int(counts[k].sum()), "entries": 10, "mean_log2_frequency": float(np.log2(max(lo, 1)) + 0.7)}
               for k, (b, lo, _) in enumerate(bins)}
    record = {"bins": [{"name": b, "lo": lo, "hi": hi} for b, lo, hi in bins], "bin_support": support, "scheme": "halfdecade",
              "channel": "compose"}
    return {"strata": strata, "count": counts, "sums": {"ref": sums}, "terms": terms, "record": record}


def test_frequency_analysis_on_synthetic_scores() -> None:
    bins = [b for b, _, _ in fb.scheme_bins("halfdecade")]
    base = dict(zip(bins, [3.0, 2.8, 2.5, 2.2, 1.9, 1.6, 1.3, 1.0])) | {"after_unseen": 3.2, "after_heldout": 3.4}
    base["after_rare_seen"], base["after_frequent"] = 2.9, 1.5
    flat_gain = {k: v - 0.1 for k, v in base.items()}                               # the same gain at every frequency
    rare_gain = {k: v - (0.4 if k in {"after_f1_3", "after_f4_9", "after_rare_seen"} else 0.05) for k, v in base.items()}
    biased = {k: v + (0.1 if k in {"after_f1_3", "after_f4_9", "after_rare_seen", "after_unseen", "after_heldout"} else -0.1)
              for k, v in base.items()}
    scores = {"C0'": {1: _synthetic_scores(base, seed=1)}, "C5": {1: _synthetic_scores(rare_gain, seed=2)},
              "C5ut": {1: _synthetic_scores(flat_gain, seed=3)}, "C2": {1: _synthetic_scores(biased, seed=4)},
              "P0": {1: _synthetic_scores({k: v + 1 for k, v in base.items()}, seed=5)}}
    result = fb.frequency_analysis(scores, candidates=["C5", "C2", "C5ut", "P0"], resamples=500, min_targets=10, min_entries=5)
    assert result["available"] and len(result["slope_bins"]) == 8
    assert result["bias"]["C0'"]["slope"]["mean"] < 0                               # rare terms are worse: frequency bias
    c5, c2, flat = result["changes"]["C5 − C0'"], result["changes"]["C2 − C0'"], result["changes"]["C5ut − C0'"]
    assert c5["gap"]["mean"] == pytest.approx(-0.35, abs=0.02) and c5["gap"]["ci_high"] < 0 and c5["gap"]["primary"]
    assert c5["gap_cut"]["mean"] == pytest.approx(0.35 / 1.4, abs=0.02) and c5["slope"]["mean"] > 0
    assert c2["gap"]["mean"] == pytest.approx(0.2, abs=0.02) and c2["gap"]["ci_low"] > 0      # the free table adds bias
    assert abs(flat["gap"]["mean"]) < 0.02 and flat["gap"]["ci_low"] < 0 < flat["gap"]["ci_high"]
    assert "term" in c5 and c5["term"]["gap"]["mean"] < 0
    per_bin = result["comparisons"]["C5 − C0'"]["bins"]
    assert per_bin["after_f1_3"]["mean"] == pytest.approx(-0.4, abs=0.01) and per_bin["after_f1_3"]["significant"]
    assert per_bin["after_f1_3"]["term"]["entries"] == 10 and "holm_p" in per_bin["after_heldout"]
    assert result["seeds"]["P0"] == [1] and result["losses"]["C2"]["after_f1_3"] == pytest.approx(3.1, abs=0.01)


def test_strict_analysis_on_the_toy_runs(world, tmp_path) -> None:
    scores = {}
    for name, label in (("C5", "C5"), ("C0p", "C0'")):
        fb.score_run(world["runs"][name], tmp_path / name, fillers=world["fillers"], exclusions=world["exclusions"], device="cpu")
        scores[label] = {1: fb.load_scores(tmp_path / name)}
    result = fb.strict_analysis(scores, candidate="C5", references=["C0'"], resamples=200)
    row = result["rows"]["after"]["C0'"]
    assert {"total", "filler", "nonfiller", "strict"} <= set(row) and 0 < row["strict_share"] < 1 and "holm_p" in row["strict"]


# -- 3. A2: the row probe ------------------------------------------------------------------------------------------------------

def test_ridge_r2_and_tsne() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    y = x @ rng.normal(size=8) + rng.normal(0, 0.1, 300)
    assert fb.r2(y, fb.ridge_cv(x, y)) > 0.95
    noise = rng.normal(size=300)
    assert fb.r2(noise, fb.ridge_cv(x, noise)) < 0.05
    weights = rng.multinomial(300, np.full(300, 1 / 300), size=50).astype(float)
    assert fb.r2(y, fb.ridge_cv(x, y), weights).shape == (50,)
    clusters = np.r_[rng.normal(0, 0.1, (40, 5)), rng.normal(3, 0.1, (40, 5))]
    coords = fb.tsne(clusters, perplexity=10, iterations=750)
    assert coords.shape == (80, 2)
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    assert ((distance.argmin(1) < 40) == (np.arange(80) < 40)).all()                  # every nearest neighbour in its cluster
    projection = coords @ (coords[40:].mean(0) - coords[:40].mean(0))
    assert projection[:40].max() < projection[40:].min()                              # the clusters are separated
    pick = fb.tsne_sample(np.arange(100), np.log2(np.arange(1, 101)), size=20)
    assert pick.size == 20 and len(set(np.floor(np.log2(np.arange(1, 101)))[pick].astype(int))) >= 5


def test_probe_run_and_analysis(world, tmp_path) -> None:
    records = {}
    for name in ("C2", "C5", "C0p"):
        records[name] = fb.probe_run(world["runs"][name], tmp_path / name, folds=3, tsne_points=10, tsne_iterations=50,
                                     alias_table=world["root"] / "alias_table.json", tokenizer_name="gpt2")
    assert set(records["C5"]["representations"]) == {"table", "row", "host_subtoken_mean", "degree", "frame_graph"}
    assert set(records["C0p"]["representations"]) == {"host_subtoken_mean", "degree", "frame_graph"}
    assert (tmp_path / "C5" / "tsne.png").exists() and (tmp_path / "C5" / "oof.npz").exists()
    probes = {"C2": {1: fb.load_probe(tmp_path / "C2")}, "C5": {1: fb.load_probe(tmp_path / "C5")}, "C0'": {1: fb.load_probe(tmp_path / "C0p")}}
    result = fb.probe_analysis(probes, resamples=100)
    assert result["available"] and "C2 table − C5 table" in result["contrasts"] and result["contrasts"]["C2 table − C5 table"]["primary"]
    assert set(result["r2"]["C5"]) >= {"table", "row"} and result["entries"] > 5


def test_queue_dry_runs(tmp_path) -> None:
    configs = tmp_path / "configs" / "t5"
    configs.mkdir(parents=True)
    for stem in ("SmolLM2-360M-full-C5-s1", "SmolLM2-360M-full-C0p-s2", "SmolLM2-360M-frozen-P0-s1"):
        (configs / f"{stem}.yaml").write_text(yaml.safe_dump({"model": {"pretrained": "HuggingFaceTB/SmolLM2-360M"}}))
    jobs = fb.queue_stage("t5", priority=55, root=tmp_path, dry_run=True, models=["C5", "C0p"])
    assert [j["name"] for j in jobs] == ["t5-SmolLM2-360M-full-C0p-s2-freqbias", "t5-SmolLM2-360M-full-C0p-s2-freqrows",
                                         "t5-SmolLM2-360M-full-C5-s1-freqbias", "t5-SmolLM2-360M-full-C5-s1-freqrows"]
    assert jobs[1]["lane"] == "cpu" and jobs[1]["priority"] == 56 and "--no-tsne" not in jobs[1]["command"]
    assert "--no-tsne" in fb.queue_stage("t5", root=tmp_path, dry_run=True, licensed=True, kinds=["rows"])[0]["command"]
    understanding = und.queue_stage("t5", tmp_path / "items" / "understanding-t5-smollm2-v1", priority=56, root=tmp_path, dry_run=True,
                                    seeds=[1])
    assert [j["name"] for j in understanding] == ["t5-SmolLM2-360M-frozen-P0-s1-understanding-t5-smollm2-v1",
                                                  "t5-SmolLM2-360M-full-C5-s1-understanding-t5-smollm2-v1"]


# -- 4. understanding items ------------------------------------------------------------------------------------------------------

def _context(world, spec=TOY_SPEC) -> und.BuildContext:
    return und.BuildContext(track="toy", family="gpt2", ontology=world["ontology"], table=world["table"], lexicon=world["lexicon"],
                            tokenizer=world["tokenizer"], tokenizer_name="gpt2", exclusions=torch.load(world["exclusions"], weights_only=False),
                            families_spec=spec, min_subtokens=1, ontology_path=world["root"] / "ontology.pt",
                            alias_table_path=world["root"] / "alias_table.json")


@pytest.fixture(scope="module")
def items(world, tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e9u-items")
    new = edit.build_new_word_items(world["root"] / "ontology.pt", root / "new", tokenizer_name="gpt2", count=3, min_subtokens=1,
                                    alias_table=world["root"] / "alias_table.json", lexicon=world["lexicon"])
    assert new["count"] == 3
    und.build_items_from_context(_context(world), root / "understanding", new_items=root / "new", counts={"seen": 20, "rare": 20})
    return root / "understanding"


def test_items_answers_are_not_fillers(world, items) -> None:
    manifest, concepts, rows = und.load_items(items)
    ctx = _context(world)
    by_id = {c["concept"]: c for c in concepts}
    assert manifest["schema"] == und.SCHEMA and {"seen", "heldout", "new"} <= set(manifest["anchors"])
    tests = {r["test"] for r in rows}
    assert {"two_hop", "bridge", "hop1", "affordance", "paraphrase", "reverse", "comparison", "affirm", "negate"} <= tests
    for r in rows:
        anchor = by_id[r["anchor"]]
        frame_texts = {ctx.text(ctx.view.atomic_id[a]) for _, a in anchor["frame"]}
        if r["test"] == "two_hop":
            owner = next(a for rel, a in anchor["frame"] if rel == "owned_by").split(":", 1)[1]
            department = GLOSSARY[owner][4]
            assert r["candidates"][r["gold"]] == " " + _display(department)          # the owner's department
            assert all(c.strip() not in frame_texts for c in r["candidates"])         # no option is a filler of the frame
        if r["test"] in {"affordance", "paraphrase"}:
            forbidden = ctx.closure_words([(ctx.view.relation_id[rel], ctx.view.atomic_id[a]) for rel, a in anchor["frame"]],
                                          anchor["entry"])
            assert not any(und.overlaps(c, forbidden) for c in r["candidates"])
        if r["test"] == "reverse":
            assert r["candidates"] == [" {x}", " {y}"] and r["gold"] == 0 and r["null"] == {"c": und.NULL_CUE}
            assert by_id[r["slots"]["y"]]["subset"] == anchor["subset"]
    pairs = {}
    for r in rows:
        if r["family"] == "negation":
            pairs.setdefault(r["pair"], {})[r["test"]] = r
    assert pairs and all(p["affirm"]["candidates"] == p["negate"]["candidates"] and p["affirm"]["gold"] != p["negate"]["gold"]
                         for p in pairs.values())
    labels = [r["meta"]["label"] for r in rows if r["test"] == "comparison"]
    assert labels.count("same") == labels.count("different") > 0                   # balanced
    bridges = [c for c in concepts if c.get("role") == "bridge"]
    assert bridges and all(c["subset"] == "bridge" for c in bridges)
    new = [c for c in concepts if c["subset"] == "new"]
    assert new and all(c["entry"] is None and c["random_frame"] for c in new)


def test_evaluation_sources_and_restoration(world, items) -> None:
    run = common.open_run(world["runs"]["C5"], device="cpu", batch_size=16, alias_table=world["root"] / "alias_table.json")
    fillers = run.composer.schedule.fillers.clone()
    entries = run.channel.entry_count
    evaluation = und.evaluate(run, items, log=lambda m: None)
    assert evaluation["sources"] == ["own", "none", "random_frame"]
    assert torch.equal(run.composer.schedule.fillers, fillers) and run.channel.entry_count == entries     # restored
    assert all(r["status"] == "linked" for r in evaluation["resolved"].values())
    own, none = evaluation["results"]["own"], evaluation["results"]["none"]
    assert {r["test"] for r in own} >= {"hop1", "bridge"} and not {r["test"] for r in none} & {"hop1", "bridge"}   # references: own only
    own_main = [r for r in own if r["test"] not in und.REFERENCE_TESTS]
    assert [r["id"] for r in own_main] == [r["id"] for r in none] and any(a["margin"] != b["margin"] for a, b in zip(own_main, none))
    summary = und.summarize(evaluation, resamples=50)
    assert "composite" in summary["sources"]["own"]["heldout"] and summary["comparisons"]
    scores = und.concept_family_scores(own)
    assert set(scores) >= {"composite", "two_hop", "negation", "affirm", "negate"}
    assert all(-0.25 <= v <= 0.75 for v in scores["negation"].values())
    c0 = und.evaluate(common.open_run(world["runs"]["C0p"], device="cpu", alias_table=world["root"] / "alias_table.json"), items,
                      log=lambda m: None)
    assert c0["sources"] == ["own"]
    c2 = und.evaluate(common.open_run(world["runs"]["C2"], device="cpu", alias_table=world["root"] / "alias_table.json"), items,
                      log=lambda m: None)
    assert c2["sources"] == ["own", "none"]


def test_understanding_cli_and_cross_model(world, items, tmp_path) -> None:
    folders = {}
    for name in ("C5", "C0p", "P0"):
        out = tmp_path / name
        und.main(["evaluate", "--run", str(world["runs"][name]), "--items", str(items), "--output", str(out), "--device", "cpu",
                  "--alias-table", str(world["root"] / "alias_table.json"), "--resamples", "50", "--batch-size", "16"])
        folders[name] = out
        assert (out / "report.md").exists() and (out / "manifest.json").exists()
    evaluations = {"C5": {1: und.load_evaluation(folders["C5"]), 2: und.load_evaluation(folders["C5"])},
                   "C0'": {1: und.load_evaluation(folders["C0p"]), 2: und.load_evaluation(folders["C0p"])},
                   "P0": {1: und.load_evaluation(folders["P0"])}}
    result = und.cross_model(evaluations, candidate="C5", references=["C0'", "P0"], resamples=50, item_seed=True,
                             sources=[("C5", "none")])
    labels = {r["comparison"] for r in result["comparisons"]}
    assert labels == {"C5 − C0'", "C5 − P0", "C5 own − none"}
    composite = [r for r in result["comparisons"] if r["test"] == "composite" and r["subset"] == "heldout" and r["comparison"] == "C5 − C0'"]
    assert composite and composite[0]["seeds"] == [1, 2] and composite[0]["item_seed"]["available"]
    assert "composite" in result["means"]["C5"]["heldout"]


# -- 5. the report sections ---------------------------------------------------------------------------------------------------------

def _report_run(root: Path, name: str, seed: int, scores: dict) -> Path:
    folder = root / f"host-full-{name}-s{seed}"
    folder.mkdir(parents=True)
    config = {"experiment": f"e9-syn-host-full-{name}-s{seed}", "seed": seed,
              "model": {"pretrained": "org/host", "host_mode": "frozen" if name == "P0" else "train", "seq_len": 8},
              "train": {"total_tokens": 1000, **({"eval_only": True} if name == "P0" else {})},
              "eval": {"windows": scores["count"].shape[1]}, "data": {"eval": "e", "ontology": "o", "min_subtokens": 2}}
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (folder / "manifest.json").write_text("{}")
    strata, counts, sums = scores["strata"], scores["count"], scores["sums"]["ref"]
    rows = [{"type": "eval", "step": 1, "tokens": 1000, "stratum": s, "loss": float(sums[i].sum() / counts[i].sum()),
             "stratum_tokens": int(counts[i].sum())} for i, s in enumerate(strata)]
    (folder / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lm.save_window_losses(folder / "eval_windows.npz", 1000, list(range(counts.shape[1])),
                          {s: ([sums[i]], [counts[i]]) for i, s in enumerate(strata)})
    out = folder / fb.SCORE_DIR
    out.mkdir()
    np.savez_compressed(out / "windows.npz", strata=np.asarray(strata), starts=np.arange(counts.shape[1]), count=counts, sum_ref=sums)
    np.savez_compressed(out / "terms.npz", **scores["terms"])
    (out / "freqbias.json").write_text(json.dumps(scores["record"]))
    return folder


def test_report_freqbias_and_understanding_sections(world, items, tmp_path) -> None:
    bins = [b for b, _, _ in fb.scheme_bins("halfdecade")]
    base = dict(zip(bins, [3.0, 2.8, 2.5, 2.2, 1.9, 1.6, 1.3, 1.0])) | {"after_unseen": 3.2, "after_heldout": 3.4,
                                                                         "after_rare_seen": 2.9, "after_frequent": 1.5}
    runs = tmp_path / "runs"
    for name, shift in (("C0p", 0.0), ("C5", -0.2), ("C2", 0.05), ("P0", 0.8)):
        for seed in ((1,) if name == "P0" else (1, 2)):
            folder = _report_run(runs, name, seed, _synthetic_scores({k: v + shift for k, v in base.items()}, seed=seed + len(name)))
            if name in {"C5", "C0p", "P0"}:
                und.main(["evaluate", "--run", str(world["runs"][name]), "--items", str(items), "--output",
                          str(und.output_folder(folder, items)), "--device", "cpu", "--resamples", "30",
                          "--alias-table", str(world["root"] / "alias_table.json"), "--batch-size", "16"])
    summary = e9_report.write_report([runs], tmp_path / "report", resamples=200, figures=True, freqbias=True,
                                     understanding=items.name, item_seed=True)
    group = summary["groups"]["host · train"]
    freq = group["freqbias"]
    assert freq["loss"]["available"] and "C5 − C0'" in freq["loss"]["changes"]
    assert freq["loss"]["changes"]["C5 − C0'"]["gap"]["mean"] == pytest.approx(0.0, abs=0.05)
    assert freq["strict"]["available"] is False or "rows" in freq["strict"]
    understanding = group["understanding"]
    assert understanding["available"] and any(r["comparison"] == "C5 − C0'" for r in understanding["comparisons"])
    report = (tmp_path / "report" / "report.md").read_text()
    for heading in ("Frequency bias of the loss", "Understanding items"):
        assert heading in report
    assert any(p.name.startswith("freqbias") for p in (tmp_path / "report" / "figures").iterdir())
