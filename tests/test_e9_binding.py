"""E9 binding and unbinding program, step 1 (decision 60): the decodability probe (`e9_binding_probe`), the role items
(`e9_binding_items`: role-swap twins, natural role-ambiguous items) and the stage analysis (`e9_binding_report`), on a
T5-like toy glossary whose `depends_on` and `uses` share fillers, with a tiny fake host (CPU)."""

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
from vsa_embed.experiments import e9_binding_items as bi
from vsa_embed.experiments import e9_binding_probe as bp
from vsa_embed.experiments import e9_binding_report as br
from vsa_embed.experiments import e9_freqbias as fb
from vsa_embed.experiments import e9_understanding as und
from vsa_embed.experiments import e5_common as common
from vsa_embed.experiments.e9_tracks import TrackLexicon
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

# name → (type, area, owner, depends_on, uses)
GLOSSARY = {
    "Zash Team": ("team", "finance", None, None, None),
    "Varkt Crew": ("team", "sales", None, None, None),
    "Grosh Console": ("system", "sales", "Varkt Crew", None, None),
    "Brightwater Ledger": ("system", "finance", "Zash Team", "Grosh Console", None),
    "Quill Engine": ("system", "finance", "Zash Team", None, "Grosh Console"),
    "Plurb Gateway": ("system", "sales", "Varkt Crew", "Brightwater Ledger", "Quill Engine"),
    "Skolt Handoff": ("process", "finance", "Zash Team", "Quill Engine", "Brightwater Ledger"),
    "Noxel Index": ("metric", "finance", "Zash Team", "Grosh Console", "Plurb Gateway"),
    "Morbel Feed": ("dataset", "sales", "Varkt Crew", "Plurb Gateway", "Grosh Console"),
    "Quarn Portal": ("system", "finance", "Varkt Crew", "Quill Engine", "Grosh Console"),
    "Dribbet Sync": ("process", "sales", "Zash Team", "Brightwater Ledger", "Plurb Gateway"),
    "Terb Template": ("document", "finance", "Varkt Crew", "Grosh Console", "Quill Engine"),
}
TERMS = list(GLOSSARY)
HELDOUT = [TERMS.index(t) for t in ("Quarn Portal", "Dribbet Sync", "Terb Template")]
RELATIONS = ["is_a", "area", "owned_by", "depends_on", "uses"]
TYPES = ["system", "process", "metric", "document", "dataset", "team"]
ATOMS = [f"type:{t}" for t in TYPES] + ["area:finance", "area:sales"] + [f"term:{t}" for t in TERMS]
A = {a: i for i, a in enumerate(ATOMS)}
ARTICLE = {"team"}
SEQ = 48
TEMPLATES = {
    "owned_by": RelationTemplates(["{x} is owned by", "Questions about {x} go to", "Ownership of {x} lies with"], "{x} is owned by {y}."),
    "area": RelationTemplates(["{x} belongs to the", "In the org chart, {x} sits in the"], "{x} belongs to the {y} area.", " {y} area"),
    "is_a": RelationTemplates(["The kind of thing {x} is: a", "In the glossary, {x} is filed under the type"], "{x} is a {y}."),
    "depends_on": RelationTemplates(["{x} depends on", "{x} cannot run without", "{x} has a hard dependency on"], "{x} depends on {y}."),
    "uses": RelationTemplates(["{x} uses", "{x} relies on", "During {x} the team works in"], "{x} uses {y}."),
}
SPEC = {"anchor": {"relation": "is_a", "values": [f"type:{t}" for t in TYPES[:5]]}, "two_hop": [], "affordance": {"relation": "is_a", "items": {}},
        "paraphrase": {}, "reverse": {}, "reverse_templates": [], "comparison": {}, "negation": {}}


class NoWordNet:
    @staticmethod
    def all_synsets():
        return []


def _display(name: str) -> str:
    return f"the {name}" if GLOSSARY[name][0] in ARTICLE else name


def _frame(term: str) -> list[tuple[int, int]]:
    kind, area, owner, depends, uses = GLOSSARY[term]
    edges = [(0, A[f"type:{kind}"]), (1, A[f"area:{area}"])]
    edges += [(2, A[f"term:{owner}"])] if owner else []
    edges += [(3, A[f"term:{depends}"])] if depends else []
    edges += [(4, A[f"term:{uses}"])] if uses else []
    return edges


def _sentences(term: str) -> list[str]:
    kind, area, owner, depends, uses = GLOSSARY[term]
    out = [f"{term}: a {area} {kind}" + (f", owned by {_display(owner)}" if owner else "") + "."]
    out += [f"{term} depends on {depends}."] if depends else []
    out += [f"{term} uses {uses}."] if uses else []
    return out


def fake_host(config=None) -> torch.nn.Module:
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=2))


def _config(root: Path, name: str, channel: dict, host_mode: str = "lora") -> dict:
    config = {"seed": 1, "device": "cpu", "experiment": f"e9-b-fake-{host_mode}-{name}-s1",
              "model": {"size": "pretrained", "seq_len": SEQ, "pretrained": "fake-host", "host_mode": host_mode, "lora_rank": 2},
              "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": SEQ * 2 * 4, "lr": 3e-3, "warmup_tokens": 64,
                        "log_every": 2, "save_trainable_only": True, "host_lr": 1e-3},
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


COMPOSE = {"mode": "compose", "composition": "attentive", "context_window": 4}
SPECS = {"P0": ({"mode": "none"}, "frozen"), "C0p": ({"mode": "none"}, "lora"), "C2": ({"mode": "free", "free_dimension": 8}, "lora"),
         "C5": ({**COMPOSE, "operator": "hrr"}, "lora"), "C5ut": ({**COMPOSE, "operator": "untyped"}, "lora"),
         "C5tr": ({**COMPOSE, "operator": "translation"}, "lora"), "C5rf": ({**COMPOSE, "operator": "random_fixed:unitary_hrr"}, "lora")}
# Step 2's readout arms (decision 60 step 2, decision 61): C5's channel plus the unbinding readout.
READOUT = {"layer": "third", "window": 8, "gate_bias": 0.0, "beta": 16.0}
READOUT_SPECS = {"U5": ({**COMPOSE, "operator": "hrr", "readout": READOUT}, "lora"),
                 "U5ut": ({**COMPOSE, "operator": "untyped", "readout": READOUT}, "lora"),
                 "U5sl": ({**COMPOSE, "operator": "slotted_unitary", "slots": 3, "readout": READOUT}, "lora"),
                 "U5bu": ({**COMPOSE, "operator": "block_unitary", "readout": READOUT}, "lora"),
                 # decision 64 (§13.1): a fixed random unitary operator under the readout, as C5rf
                 "U5rf": ({**COMPOSE, "operator": "random_fixed:unitary_hrr", "readout": READOUT}, "lora")}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    patch = pytest.MonkeyPatch()
    patch.setattr(lm, "build_model", fake_host)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("e9b")
    full = AliasTable.from_pairs([(t, i) for i, t in enumerate(TERMS)], holdout=HELDOUT, include_holdout=True)
    rng = random.Random(0)
    texts = [" ".join(s for t in rng.choices(TERMS, k=5) for s in _sentences(t)) for _ in range(140)]
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
    fb.build_exclusion_table(ontology, full, tokenizer, root / "exclusions.pt", lexicon=lexicon, track="toy")
    from vsa_embed.experiments import e9_rescore
    e9_rescore.build_filler_table(ontology, full, tokenizer, root / "fillers.pt", lexicon=lexicon)
    runs = {}
    for name, (channel, mode) in {**SPECS, **READOUT_SPECS}.items():
        folder = root / "runs" / "toy" / f"fake-{mode}-{name}-s1"
        lm.train(_config(root, name, channel, mode), folder)
        runs[name] = folder
    ctx = und.BuildContext(track="toy", family="gpt2", ontology=ontology, table=full, lexicon=lexicon, tokenizer=tokenizer,
                           tokenizer_name="gpt2", exclusions=torch.load(root / "exclusions.pt", weights_only=False), families_spec=SPEC,
                           min_subtokens=1, ontology_path=root / "ontology.pt", alias_table_path=root / "alias_table.json")
    yield {"root": root, "table": full, "ontology": ontology, "runs": runs, "ctx": ctx, "alias": root / "alias_table.json",
           "fillers": root / "fillers.pt"}
    patch.undo()


def _open(world, name: str) -> common.E5Run:
    return common.open_run(world["runs"][name], device="cpu", batch_size=8, alias_table=world["alias"])


# -- edges, flags and exclusions -------------------------------------------------------------------------------------------------

def test_frame_edges_mark_role_ambiguity(world) -> None:
    o = world["ontology"]
    entries = np.arange(len(TERMS))
    edges = bp.frame_edges(np.asarray(o["offsets"]), np.asarray(o["relations"]), np.asarray(o["fillers"]), entries)
    plurb = TERMS.index("Plurb Gateway")
    mine = edges["entry"] == plurb
    by_relation = dict(zip(edges["relation"][mine].tolist(), edges["ambiguous"][mine].tolist()))
    assert by_relation[3] and by_relation[4]                        # Brightwater Ledger and Quill Engine serve both roles
    assert not by_relation[0]                                        # a type filler occurs under is_a only
    assert not edges["reuse"].any() and not edges["multi"].any()
    assert (edges["degree"][mine] == 5).all()


def test_within_frame_exclusions_pair_other_golds() -> None:
    segments = torch.tensor([0, 0, 0, 1, 1])
    relations = torch.tensor([1, 1, 2, 1, 2])
    fillers = torch.tensor([7, 8, 7, 5, 5])
    (fr, fa), (rr, ra) = bp.within_frame_exclusions(segments, relations, fillers)
    assert sorted(zip(fr.tolist(), fa.tolist())) == [(0, 8), (1, 7)]              # the other filler of relation 1
    assert sorted(zip(rr.tolist(), ra.tolist())) == [(0, 2), (2, 1), (3, 2), (4, 1)]  # the same filler under another role


def test_selection_is_seeded_capped_and_skips_empty_frames() -> None:
    ontology = {"entry_count": 9, "train_frequency": [20, 20, 20, 3, 0, 0, 50, 2, 1], "heldout_entries": [4, 8]}
    degrees = np.array([3, 3, 3, 2, 2, 0, 4, 1, 2])
    labels = bp.entry_subsets(ontology)
    assert labels.tolist() == ["seen", "seen", "seen", "rare", "heldout", "unseen", "seen", "rare", "heldout"]
    picked = bp.select_entries(ontology, degrees, cap=2, seed=0)
    assert picked.tolist() == bp.select_entries(ontology, degrees, cap=2, seed=0).tolist()
    assert sum(labels[picked] == "seen") == 2 and 5 not in picked.tolist() and {3, 7, 4, 8} <= set(picked.tolist())


def test_chance_reciprocal_rank() -> None:
    assert bp.chance_reciprocal(torch.tensor([1, 2, 4])).tolist() == pytest.approx([1.0, 0.75, 25 / 48], abs=1e-6)


# -- probe A / B on the toy runs -------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def probes(world, tmp_path_factory) -> dict:
    out = {}
    for name in ("C5", "C5ut", "C5tr", "C5rf", "C2", "P0"):
        folder = world["runs"][name] / bp.OUTPUT
        out[name] = bp.probe_run(world["runs"][name], folder, device="cpu", alias_table=world["alias"],
                                 settings=bp.ProbeSettings(cap=50, contexts=2, folds=2, min_train=3, batch_size=8), log=lambda *_: None)
    return out


def test_probe_a_reads_every_condition_and_method(probes) -> None:
    c5 = probes["C5"]
    assert c5["algebraic"]["methods"] == ["correlation", "inverse"] and c5["same_edges"]
    assert set(c5["algebraic"]["conditions"]) == {"static", "nocontext", "neutral", "context"}
    keys = c5["summary"]["algebraic"]
    assert {"a.static.correlation.filler.typed", "a.neutral.inverse.filler.all", "a.context.correlation.filler.typed",
            "a.static.match.role.all", "b.frequency.filler.typed", "b.chance.filler.typed"} <= set(keys)
    block = keys["a.static.correlation.filler.typed"]
    assert {"seen", "heldout", "all"} <= set(block["subsets"]) and block["subsets"]["all"]["edges"] > 30
    assert 0 < block["subsets"]["all"]["mrr"] <= 1 and "relations" in block and "depends_on" in block["relations"]
    assert probes["C5ut"]["algebraic"]["methods"] == ["bundle"] and probes["C5tr"]["algebraic"]["methods"] == ["subtract"]
    assert probes["C5rf"]["algebraic"]["methods"] == ["conjugate"]
    assert "algebraic" not in probes["C2"] and "algebraic" not in probes["P0"]


def test_untyped_role_recovery_is_exactly_chance(world, probes) -> None:
    edges = bp.load_probe(world["runs"]["C5ut"] / bp.OUTPUT)["edges"]
    hits = edges["a.static.match.role.all.hit"].astype(np.float64)
    assert np.allclose(hits, 1 / len(RELATIONS), atol=1e-3)         # every relation ties: the bundle keeps no role
    typed = edges["a.static.match.role.typed.hit"].astype(np.float64)
    ambiguous = edges["ae_ambiguous"]
    assert np.allclose(typed[ambiguous], 0.5, atol=1e-3)           # depends_on vs uses: a coin flip
    assert np.allclose(typed[~ambiguous], 1.0)                      # one candidate role: no choice to make


def test_probe_b_representations_per_model(probes) -> None:
    assert {"l.composed_static", "l.composed", "l.injected_row", "l.hidden_middle", "l.hidden_final"} <= set(probes["C5"]["summary"]["learned"])
    assert {"l.free_row", "l.injected_row", "l.hidden_final"} <= set(probes["C2"]["summary"]["learned"])
    assert set(probes["P0"]["summary"]["learned"]) == {"l.hidden_middle", "l.hidden_final"}
    assert "l.hidden_final" not in probes["C5ut"]["summary"]["learned"]          # hidden states: P0, C0′, C2, C5 only
    assert probes["C5"]["hidden"]["linked"] == len(TERMS) and probes["C5"]["hidden"]["middle_layer"] == 1


def test_learned_probe_recovers_a_linear_code_and_not_noise() -> None:
    rng = np.random.default_rng(0)
    entries = np.arange(120)
    fillers = rng.integers(0, 6, size=120)
    edges = {"edge": entries.copy(), "entry": entries, "relation": np.zeros(120, dtype=np.int64), "filler": fillers,
             "degree": np.ones(120, dtype=np.int64), "ambiguous": np.zeros(120, bool), "reuse": np.zeros(120, bool), "multi": np.zeros(120, bool)}
    subsets = np.asarray(["seen"] * 100 + ["heldout"] * 20)
    code = np.eye(6)[fillers] + 0.05 * rng.standard_normal((120, 6))
    noise = rng.standard_normal((120, 6))
    values, info = bp.learned_probe({"code": code, "noise": noise}, entries, subsets, edges,
                                    bp.ProbeSettings(folds=3, min_train=5), torch.device("cpu"), log=lambda *_: None)
    held = subsets == "heldout"
    assert values["l.code.rr"][held].mean() > 0.95 and values["l.code.covered"].all()
    assert values["l.noise.rr"][held].mean() < 0.7
    assert info["code"]["relations"]["0"]["classes"] == 6


def test_licensed_probe_writes_aggregates_only(world, tmp_path) -> None:
    record = bp.probe_run(world["runs"]["C5"], tmp_path / "lic", device="cpu", alias_table=world["alias"], licensed=True, hidden="none",
                          settings=bp.ProbeSettings(cap=50, contexts=1, folds=2, min_train=3, batch_size=8), log=lambda *_: None)
    assert not (tmp_path / "lic" / "edges.npz").exists() and (tmp_path / "lic" / "summary.json").exists()
    relations = record["summary"]["algebraic"]["a.static.correlation.filler.typed"]["relations"]
    assert set(relations) <= {f"r{i}" for i in range(len(RELATIONS))}           # relations by index, never by name
    assert "depends_on" not in (tmp_path / "lic" / "summary.json").read_text()


# -- role items ------------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def item_dirs(world, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("roleitems")
    ctx = world["ctx"]
    twins = bi.build_twins(ctx, root / "role-twins-toy-v1", count=4, seed=0, min_shared=2, wordnet=NoWordNet(), reserved_names=set(TERMS))
    natural = bi.build_natural(ctx, root / "role-natural-toy-v1", counts={"seen": 5, "rare": 5, "heldout": 5}, seed=0)
    return {"twins": root / "role-twins-toy-v1", "natural": root / "role-natural-toy-v1", "twins_manifest": twins, "natural_manifest": natural}


def test_twins_swap_two_roles_of_shared_fillers(world, item_dirs) -> None:
    manifest, concepts, items = bi.load_items(item_dirs["twins"])
    assert manifest["kind"] == "twins" and manifest["pairs"] == 4
    assert [(p["r1"], p["r2"]) for p in manifest["relation_pairs"]] == [("depends_on", "uses")]
    by_id = {c["concept"]: c for c in concepts}
    shared = {ATOMS[a] for a in bi.shared_fillers(world["ctx"], "depends_on", "uses")}
    for c in concepts:
        if c["twin"] != "A":
            continue
        partner = by_id[c["partner"]]
        frame_a, frame_b = {tuple(e) for e in c["frame"]}, {tuple(e) for e in partner["frame"]}
        x, y = c["fillers"]["X"], c["fillers"]["Y"]
        assert {x, y} <= shared and x != y
        assert ("depends_on", x) in frame_a and ("uses", y) in frame_a and ("depends_on", y) in frame_b and ("uses", x) in frame_b
        assert sorted(f for _, f in c["frame"]) == sorted(f for _, f in partner["frame"])        # the same filler multiset
        assert frame_a - {("depends_on", x), ("uses", y)} == frame_b - {("depends_on", y), ("uses", x)}
    assert len({c["surface"] for c in concepts}) == 8 and not {c["surface"] for c in concepts} & set(TERMS)
    kinds = {(i["kind"], i["meta"]["twin"], i["meta"]["role"]) for i in items}
    assert len(kinds) == 8 and all(i["candidates"][0].strip() and len(i["candidates"]) == 2 for i in items)
    golds = {(i["concept"], i["relation"], i["kind"]): i["gold"] for i in items}
    assert golds[("tw-0000-A", "depends_on", "choice")] == 0 and golds[("tw-0000-B", "depends_on", "choice")] == 1
    assert golds[("tw-0000-A", "uses", "cloze")] == 1


def test_natural_items_pick_fillers_plausible_in_both_roles(world, item_dirs) -> None:
    manifest, concepts, items = bi.load_items(item_dirs["natural"])
    assert manifest["kind"] == "natural" and concepts
    ctx = world["ctx"]
    for c in concepts:
        r1, r2 = c["relations"]
        f, g = A[c["fillers"]["F"]], A[c["fillers"]["G"]]
        assert f in ctx.pools[r2] and g in ctx.pools[r1] and f != g
    assert {i["gold"] for i in items if i["meta"]["role"] == "r1"} == {0} and {i["gold"] for i in items if i["meta"]["role"] == "r2"} == {1}


def _rows(pairs: int, *, bind: bool) -> list[dict]:
    """Synthetic twin rows: with binding A prefers its own filler by +1; without, A and B score alike."""
    rows = []
    for p in range(pairs):
        for twin in ("A", "B"):
            for relation, role in (("depends_on", "r1"), ("uses", "r2")):
                gold = 0 if (role == "r1") == (twin == "A") else 1
                s = [[1.0, 0.0], [0.5, 0.2]]
                if bind and gold == 1:
                    s = [[0.0, 1.0], [0.2, 0.5]]
                rows.append({"kind": "choice", "concept": f"tw-{p:04d}-{twin}", "relation": relation, "gold": gold, "s": s, "pmi": s,
                             "correct": [float(np.argmax(r) == gold) for r in s], "meta": {"pair": p, "twin": twin, "role": role}})
    return rows


def test_twin_contrast_is_one_with_binding_and_one_half_without() -> None:
    bound = bi.twin_scores(_rows(3, bind=True))
    blind = bi.twin_scores(_rows(3, bind=False))
    assert all(v["contrast"] == 1.0 and v["margin"] > 0 for v in bound.values())
    assert all(v["contrast"] == 0.5 and v["margin"] == 0 and v["item"] == 0.5 for v in blind.values())


def test_role_items_evaluate_on_toy_runs(world, item_dirs) -> None:
    for name, expected in (("C5", ["own", "none", "swap"]), ("C2", ["own", "none"]), ("C0p", ["own"])):
        run = _open(world, name)
        evaluation = bi.evaluate(run, item_dirs["twins"], log=lambda *_: None)
        assert evaluation["sources"] == expected
        assert all(r["status"] == "linked" for r in evaluation["resolved"].values()) or name == "C0p"
        units = bi.unit_scores(evaluation, "own", "choice")
        if name != "C0p":
            assert sorted(units) == [0, 1, 2, 3] and all(0 <= v["contrast"] <= 1 for v in units.values())
        summary = bi.summarize(evaluation, resamples=50)
        assert "choice" in summary["sources"]["own"] and "cloze" in summary["sources"]["own"]
    run = _open(world, "C5")
    natural = bi.evaluate(run, item_dirs["natural"], log=lambda *_: None)
    assert natural["sources"] == ["own", "none"] and bi.unit_scores(natural, "own", "choice")
    zero = bi.evaluate(_open(world, "C5ut"), item_dirs["twins"], sources=["none"], log=lambda *_: None)
    blind = bi.unit_scores(zero, "none", "choice")
    names_only = bi.unit_scores(bi.evaluate(_open(world, "C0p"), item_dirs["twins"], log=lambda *_: None), "own", "choice")
    assert blind and names_only                                         # rows zeroed: what the host does from names alone


def test_role_items_cli_writes_outputs_and_the_report_reads_them(world, item_dirs, probes, tmp_path) -> None:
    for name in ("C5", "C5ut", "C5tr", "C0p"):
        bi.main(["evaluate", "--run", str(world["runs"][name]), "--items", str(item_dirs["twins"]), "--alias-table", str(world["alias"]),
                 "--device", "cpu", "--batch-size", "8", "--resamples", "50"])
        folder = bi.output_folder(world["runs"][name], item_dirs["twins"])
        assert (folder / "summary.json").exists() and (folder / "predictions.jsonl.gz").exists()
        loaded = bi.load_evaluation(folder)
        assert loaded["kind"] == "twins" and bi.unit_scores(loaded, "own", "choice") or name == "C0p"
    analysis = br.analyse(world["root"] / "runs" / "toy", twins=item_dirs["twins"].name, resamples=50)
    block = analysis["hosts"]["fake"]
    assert set(block["P1"]["contrasts"]) == {"C5 − C5ut", "C5 − C5tr"}
    assert all(v["available"] and "p_holm" in v for v in block["P1"]["contrasts"].values())
    assert block["P1"]["reading"]
    assert block["P2"]["available"] and {"C5 − frequency", "C5 − chance", "C5 − C5rf"} <= set(block["P2"]["contrasts"])
    assert "C5ut" in block["probes"]["algebraic"] and "C5" in block["probes"]["learned"]
    br.main(["--runs", str(world["root"] / "runs" / "toy"), "--output", str(tmp_path / "report"), "--twins", str(item_dirs["twins"]),
             "--resamples", "50"])
    text = (tmp_path / "report" / "report.md").read_text()
    assert "P1 — role-swap twins" in text and "P2 — algebraic filler recovery" in text


def test_queue_dry_runs(world, item_dirs, tmp_path) -> None:
    configs = tmp_path / "configs" / "toy"
    configs.mkdir(parents=True)
    for name, (channel, mode) in SPECS.items():
        config = _config(world["root"], name, channel, mode)
        (configs / f"SmolLM2-360M-{'frozen' if mode == 'frozen' else 'full'}-{name}-s1.yaml").write_text(yaml.safe_dump(config))
    jobs = bp.queue_stage("toy", root=tmp_path, priority=50, dry_run=True, python="python")
    assert len(jobs) == len(SPECS) and all(j["name"].endswith("-binding-probe") and j["priority"] == 50 for j in jobs)
    assert jobs[0]["command"][:4] == ["python", "-m", "vsa_embed.experiments.e9_binding_probe", "probe"]
    licensed = bp.queue_stage("toy", root=tmp_path, licensed=True, dry_run=True, python="python", models=["C5"])
    assert len(licensed) == 1 and "--licensed" in licensed[0]["command"]
    items = bi.queue_stage("toy", item_dirs["twins"], root=tmp_path, dry_run=True, python="python")
    sources = {j["model"]: j["command"][j["command"].index("--sources") + 1] for j in items}
    assert sources == {"P0": "own", "C0p": "own", "C2": "own,none", "C5": "own,none,swap", "C5ut": "own,none,swap",
                       "C5tr": "own,none,swap", "C5rf": "own,none,swap"}
    readout = bi.queue_stage("toy", item_dirs["twins"], root=tmp_path, dry_run=True, python="python", models=["U5"])
    assert readout == []                                                   # no U5 config written in this test
    assert all(j["name"].endswith("-role-twins-toy-v1") for j in items)


# -- step 2: the unbinding readout arms ------------------------------------------------------------------------------------------

def test_readout_runs_train_save_and_reload(world) -> None:
    from vsa_embed.readout import UnbindingReadout
    for name in READOUT_SPECS:
        final = torch.load(world["runs"][name] / "final.pt", weights_only=False)
        assert any(k.startswith("channel.readout.") for k in final["model"])
        model = lm.load_final(world["runs"][name] / "final.pt", "cpu")
        readout = model.channel.readout
        assert isinstance(readout, UnbindingReadout) and model._readout_hook is not None and readout.layer == 1
        assert readout.method == {"U5": "correlation", "U5ut": "bundle", "U5sl": "conjugate", "U5bu": "transpose",
                                  "U5rf": "conjugate"}[name]
        torch.testing.assert_close(readout.query.weight, final["model"]["channel.readout.query.weight"])
    # U5rf: the operator is saved, reloaded frozen, and equals C5rf's (same seed; the readout draws from its own generator,
    # and training never moves a fixed operator)
    phases = [k for k in torch.load(world["runs"]["U5rf"] / "final.pt", weights_only=False)["model"] if k.endswith("transform.phases")]
    assert phases
    fixed = {name: torch.load(world["runs"][name] / "final.pt", weights_only=False)["model"][phases[0]] for name in ("U5rf", "C5rf")}
    assert torch.equal(fixed["U5rf"], fixed["C5rf"])
    reloaded = lm.load_final(world["runs"]["U5rf"] / "final.pt", "cpu")
    assert not any(p.requires_grad for p in reloaded.channel.composer.transform.parameters())
    slots = json.loads((world["runs"]["U5sl"] / "slots.json").read_text())
    assert slots["groups"] == 3 and sum(slots["load"]) == int(world["ontology"]["offsets"][-1])
    assert sorted(r for group in slots["relations"].values() for r in group) == sorted(RELATIONS)
    assert not (world["runs"]["U5"] / "slots.json").exists()


def test_readout_initialization_leaves_the_rest_of_the_run_unchanged(world) -> None:
    config = lm.resolve_config(_config(world["root"], "C5", {**COMPOSE, "operator": "hrr"}))
    with_readout = lm.resolve_config(_config(world["root"], "U5", READOUT_SPECS["U5"][0]))
    torch.manual_seed(1)
    host = fake_host()
    torch.manual_seed(1)
    plain, _ = lm.build_channel(config, world["ontology"], 32, host=host)
    after_plain = torch.rand(3)
    torch.manual_seed(1)
    readout, _ = lm.build_channel(with_readout, world["ontology"], 32, host=host)
    after_readout = torch.rand(3)
    assert torch.equal(after_plain, after_readout)                       # the readout draws from its own generator
    shared = {k: v for k, v in readout.state_dict().items() if not k.startswith("readout.")}
    assert all(torch.equal(v, plain.state_dict()[k]) for k, v in shared.items())


def test_readout_evaluation_on_and_off_with_diagnostics(world, tmp_path) -> None:
    from vsa_embed.experiments import e9_binding_readout as ro
    record = ro.evaluate_run(world["runs"]["U5"], tmp_path / "readout", device="cpu", fillers=world["fillers"])
    assert set(record["variants"]) == {"on", "off"} and record["ref_check"]["replay_ok"]
    on, off = record["variants"]["on"], record["variants"]["off"]
    assert on["after"]["targets"] == off["after"]["targets"] and on["after"]["loss"] != off["after"]["loss"]
    block = record["diagnostics"]["subsets"]["all"]
    assert block["positions"] > 10 and 0 <= block["role_accuracy"] <= 1 and 0 < block["oracle_mrr"] <= 1
    assert record["diagnostics"]["chance"] == pytest.approx(1 / len(RELATIONS))
    loaded = ro.load_evaluation(tmp_path / "readout")
    assert loaded["windows"]["sum_on"].shape == loaded["windows"]["sum_off"].shape
    assert loaded["positions"]["correct"].size == block["positions"]
    with pytest.raises(ValueError, match="no unbinding readout"):
        ro.evaluate_run(world["runs"]["C5"], tmp_path / "none", device="cpu", fillers=world["fillers"])


def test_row_overrides_switch_the_readout_off_for_those_entries(world) -> None:
    run = _open(world, "U5")
    readout = run.channel.readout
    assert readout.blocked is None
    with common.override_rows(run.channel, {0: torch.zeros(32), 4: torch.zeros(32)}):
        assert readout.blocked.tolist() == [0, 4]
        spans = {"batch": torch.tensor([0, 0]), "inject": torch.tensor([2, 6]), "entry": torch.tensor([0, 5])}
        rows, cols, span = readout.locate(1, 12, spans, torch.device("cpu"))
        assert set(span.tolist()) == {1} and cols.min() == 6                # the blocked entry is not read out
    assert readout.blocked is None


def test_probe_and_twins_read_the_readout_arms(world, item_dirs, tmp_path) -> None:
    settings = bp.ProbeSettings(cap=50, contexts=1, folds=2, min_train=3, batch_size=8)
    slotted = bp.probe_run(world["runs"]["U5sl"], tmp_path / "sl", device="cpu", alias_table=world["alias"], settings=settings,
                           log=lambda *_: None)
    assert slotted["algebraic"]["methods"] == ["conjugate"] and "a.static.conjugate.filler.typed" in slotted["summary"]["algebraic"]
    single = bp.probe_run(world["runs"]["U5bu"], tmp_path / "bu", device="cpu", alias_table=world["alias"], settings=settings,
                          log=lambda *_: None)
    assert single["algebraic"]["methods"] == ["transpose"]
    evaluation = bi.evaluate(_open(world, "U5"), item_dirs["twins"], log=lambda *_: None)
    assert evaluation["sources"] == ["own", "none", "swap"] and bi.unit_scores(evaluation, "none", "choice")


def test_plan_wires_the_readout_arms() -> None:
    from vsa_embed.experiments import e9_plan
    from vsa_embed.experiments.e9_tracks import track_spec
    for model, operator in (("U5", "hrr"), ("U5u", "unitary_hrr"), ("U5sb", "spectral_bounded"), ("U5bu", "block_unitary"),
                            ("U5sl", "slotted_unitary"), ("U5tr", "translation"), ("U5ut", "untyped"),
                            ("U5rf", "random_fixed:unitary_hrr")):
        spec = e9_plan.model_spec(model, free_dimension=8, gate_bias=0.0)["channel"]
        assert spec["mode"] == "compose" and spec["operator"] == operator and spec["readout"] == e9_plan.READOUT
        assert spec["composition"] == "attentive" and spec["context_window"] == 8 and spec["gate_bias"] == 0.0
        assert model in e9_plan.ARM_LIKE and model in e9_plan.ALL_MODELS and model not in e9_plan.ARMS
    assert e9_plan.model_spec("U5sl", free_dimension=8, gate_bias=0.0)["channel"]["slots"] == 3
    assert "readout" not in e9_plan.model_spec("C5", free_dimension=8, gate_bias=0.0)["channel"]       # C5 unchanged
    jobs = e9_plan.readout_jobs(Path("runs/x"), track_spec("t5"), python="python")
    assert [s for s, _, _ in jobs] == ["edit-v2", "understanding-t5-smollm2-v1", "role-twins-t5-smollm2-v1", "freqbias", "readout",
                                       "binding-probe"]
    twins = next(c for s, c, _ in jobs if s.startswith("role-twins"))
    assert twins[twins.index("--sources") + 1] == "own,none,swap"


def test_step2_analysis_reads_the_readout_arms(world, item_dirs, tmp_path) -> None:
    from vsa_embed.experiments import e9_binding_readout as ro
    for name in ("U5", "U5ut", "C5", "U5rf"):
        bi.main(["evaluate", "--run", str(world["runs"][name]), "--items", str(item_dirs["twins"]), "--alias-table", str(world["alias"]),
                 "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--overwrite"])
    for name in ("U5", "U5ut"):
        ro.main(["evaluate", "--run", str(world["runs"][name]), "--device", "cpu", "--fillers", str(world["fillers"]), "--overwrite"])
        assert (world["runs"][name] / "readout" / "summary.json").exists()
    analysis = br.analyse(world["root"] / "runs" / "toy", twins=item_dirs["twins"].name, resamples=50)
    step = analysis["hosts"]["fake"]["step2"]
    assert step["available"] and set(step["R1"]["contrasts"]) == {"U5 − U5ut"}            # U5tr is not trained in the toy
    assert step["R1"]["contrasts"]["U5 − U5ut"]["available"] and "p_holm" in step["R1"]["contrasts"]["U5 − U5ut"]
    r2 = step["R2"]
    assert r2["available"] and r2["stratum"] == "after_heldout" and r2["seeds"] == [1] and r2["relative_ci_low"] <= r2["relative_ci_high"]
    assert set(step["off_minus_on"]) == {"U5", "U5ut"} and step["off_minus_on"]["U5"]["after"]["available"]
    assert "U5 − C5" in step["twins_secondary"] and {"U5", "U5ut"} <= set(step["diagnostics"])
    assert step["reading"]["R1"] == "incomplete" and "R2" in step["reading"]
    # §13.1 (decision 64): R1's secondary U5 − U5rf, its own family (R1's Holm and S2.1 unchanged)
    rf = step["R1_operator"]["U5 − U5rf"]
    assert rf["available"] and "p_holm" not in rf and "U5 − U5rf" not in step["twins_secondary"]
    text = br.render(analysis, title="toy")
    assert "Step 2 — readout arms" in text and "| R2 | U5 − C5" in text and "| R1 secondary (§13.1) | U5 − U5rf |" in text


# -- step 3: chained two-hop, reverse lookup, capacity, path order ------------------------------------------------------------

CHAIN_SPEC = {**SPEC, "two_hop": [{"path": ("depends_on", "owned_by"), "distractors": 0,
                                   "templates": ["{x} depends on something that is owned by", "Something {x} depends on is owned by"]}],
              "reverse": {"owned_by": {"cue": "owned by {t}", "match": None}},
              "reverse_templates": ["Of {x} and {y}, the one {c} is", "Between {y} and {x}, the one {c} is"]}


@pytest.fixture(scope="module")
def chain_items(world, tmp_path_factory) -> Path:
    ctx = world["ctx"]
    previous = ctx.families_spec
    ctx.families_spec = CHAIN_SPEC
    try:
        out = tmp_path_factory.mktemp("chainitems") / "understanding-toy-v1"
        und.build_items_from_context(ctx, out, counts={"seen": 20, "rare": 20}, families=("two_hop", "reverse"))
    finally:
        ctx.families_spec = previous
    return out


def test_chain_on_toy_composers(world, chain_items, tmp_path, monkeypatch) -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    monkeypatch.setattr(ch, "_item_atoms", lambda ontology, track, family: _toy_item_atoms(world))
    settings = ch.ChainSettings(cap=50, paths=50, reverse=40, memory_targets=20, global_paths=20, path_order=200)
    record = ch.chain_run(world["runs"]["C5"], tmp_path / "c5", items=chain_items, settings=settings, log=lambda *_: None)
    assert record["fidelity"]["methods"] == ["correlation", "inverse", "exact"]
    assert set(record["fidelity"]["results"]["exact"]["by_degree"]) and record["two_hop"]["paths"] > 0
    two = record["two_hop"]["all"]
    assert {"hop1_hit", "chain_hit", "oracle_hit", "soft_hit", "bridge_found"} <= set(two) and 0 <= two["chain_hit"] <= 1
    assert record["two_hop_global"]["memory_concepts"] == len(TERMS) and record["reverse"]["queries"] > 0
    sizes = record["capacity"]["sizes"]
    assert "1" in sizes and str(len(TERMS)) in sizes
    assert record["items"]["two_hop"]["items"] > 0 and record["items"]["reverse"]["items"] > 0
    assert (tmp_path / "c5" / "items.jsonl.gz").exists() and (tmp_path / "c5" / "report.md").exists()
    untyped = ch.chain_run(world["runs"]["C5ut"], tmp_path / "ut", settings=settings, log=lambda *_: None)
    assert untyped["fidelity"]["methods"] == ["bundle"] and untyped["path_order"]["by_distractors"]["0"]["accuracy"] == pytest.approx(0.5)
    blocks = ch.chain_run(world["runs"]["U5bu"], tmp_path / "bu", settings=settings, log=lambda *_: None)
    assert blocks["path_order"]["by_distractors"]["0"]["accuracy"] > 0.9                    # non-commutative keeps the order
    slotted = ch.chain_run(world["runs"]["U5sl"], tmp_path / "sl", settings=settings, licensed=True, log=lambda *_: None)
    assert not (tmp_path / "sl" / "items.jsonl.gz").exists() and slotted["method"] == "conjugate"


def _toy_item_atoms(world):
    lexicon = _lexicon()
    pools = {}
    o = world["ontology"]
    for r, f in zip(np.asarray(o["relations"]).tolist(), np.asarray(o["fillers"]).tolist()):
        pools.setdefault(RELATIONS[r], set()).add(f)
    def answer_atom(relation, text):
        for a in sorted(pools.get(relation, ())):
            atom_text = lexicon.text(ATOMS[a])
            if atom_text and lexicon.answer(relation, atom_text) == text:
                return a
        return None
    return answer_atom, {t: world["table"].alias_to_entry[t.lower()] for t in TERMS}


def test_global_memory_of_one_concept_is_the_local_store(world) -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    composer, _, ontology = ch.load_composer(world["runs"]["C5rf"])
    store = ch.Store(composer)
    atom_entry = ch.atom_entries(ontology)
    assert atom_entry[A["term:Zash Team"]] == world["table"].alias_to_entry["zash team"] and atom_entry[A["type:system"]] == -1
    keys = ch.memory_keys(store, atom_entry, seed=0)
    spectrum = torch.fft.rfft(keys).abs()
    torch.testing.assert_close(spectrum, torch.ones_like(spectrum), atol=1e-4, rtol=0)        # unitary keys
    memory = ch.global_memory(store, torch.tensor([3]), keys)
    recovered = ch.HRRAlgebra().unbind(memory[None], keys[[3]])[0]
    torch.testing.assert_close(recovered, torch.nn.functional.normalize(store.stores()[3], dim=0), atol=1e-4, rtol=1e-4)


def test_reverse_lookup_of_an_untyped_store_ignores_the_relation(world) -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    composer, _, _ = ch.load_composer(world["runs"]["C5ut"])
    store = ch.Store(composer)
    filler = torch.tensor([A["term:Grosh Console"]] * 2)
    scores = ch.reverse_scores(store, torch.tensor([3, 4]), filler)                       # depends_on vs uses
    torch.testing.assert_close(scores[0], scores[1])
    bound, _, _ = ch.load_composer(world["runs"]["C5"])
    typed = ch.reverse_scores(ch.Store(bound), torch.tensor([3, 4]), filler)
    assert not torch.allclose(typed[0], typed[1])


def test_synthetic_sweep_runs_and_orders_the_families() -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    result = ch.sweep(atoms=256, relations=8, trials=8, dimensions=(64,), loads=(1, 4), families=("unitary_hrr", "hrr_exact", "additive"))
    assert result["local"]["unitary_hrr"]["64"]["1"] == 1.0 and result["local"]["additive"]["64"]["4"] < 0.6
    assert result["path_order"]["block_unitary"]["by_distractors"]["0"]["accuracy"] > result["path_order"]["unitary_hrr"]["by_distractors"]["0"]["accuracy"]
    assert result["global"]["unitary_hrr"]["1"] > result["global"]["unitary_hrr"]["2048"]
    assert "Path order" in ch.render_sweep(result)


def test_chain_queue_takes_composing_configs_only(world, tmp_path) -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    configs = tmp_path / "configs" / "toy"
    configs.mkdir(parents=True)
    for name, (channel, mode) in {**SPECS, **READOUT_SPECS}.items():
        (configs / f"SmolLM2-360M-{'frozen' if mode == 'frozen' else 'full'}-{name}-s1.yaml").write_text(
            yaml.safe_dump(_config(world["root"], name, channel, mode)))
    jobs = ch.queue_stage("toy", root=tmp_path, dry_run=True, python="python", items=Path("items/u"))
    assert sorted(j["model"] for j in jobs) == sorted(["C5", "C5ut", "C5tr", "C5rf", *READOUT_SPECS])
    assert all(j["name"].endswith("-binding-chain") and "--items" in j["command"] for j in jobs)


def test_no_arm_batch_writes_the_stage_report_folder(tmp_path) -> None:
    """An arm batch (WP-PQ1 or readout arms, alone or with base models) writes its R9 report to report/<stage>-<tag>;
    only a base batch writes report/<stage>, the folder R9, the claims ledger and the draft cite."""
    from vsa_embed.experiments import e9_plan
    batches = {"base": ["P0", "C0p", "C2", "C5"], "pq": ["C5rf", "C5ut", "C5tr", "C5sh"], "readout": list(e9_plan.READOUT_ARMS),
               "mixed": ["C5", "U5", "U5ut"], "mixed_pq": ["C0p", "C5ut"]}
    expected = {"base": None, "pq": "pq", "readout": "readout", "mixed": "readout", "mixed_pq": "pq"}
    for name, models in batches.items():
        folder = tmp_path / name
        folder.mkdir()
        paths = []
        for model in models:
            stem, config = e9_plan.run_config(stage="toy", host="SmolLM2-360M", mode="train", model=model, seed=1,
                                              data_root=tmp_path / "corpus", tokens=1_000_000, lora_rank=64, host_lr=None,
                                              gate_bias=0.0, free_dimension=8)
            config["e9_track"] = "t5"
            path = folder / f"{stem}.yaml"
            path.write_text(yaml.safe_dump(config))
            paths.append(path)
        planned: list = []
        e9_plan.queue_jobs(paths, "toy", 50, track="t5", root=Path("experiments/e9-retrofit"), plan=planned)
        reports = [(job, command) for job, _, command in planned if command[2] == "vsa_embed.experiments.e9_report"]
        assert len(reports) == 1
        job, command = reports[0]
        output = Path(command[command.index("--output") + 1])
        tag = expected[name]
        assert e9_plan.batch_tag(models) == tag
        if tag is None:
            assert output == Path("experiments/e9-retrofit/report/toy") and "-pq-" not in job and "-readout-" not in job
        else:
            assert output == Path(f"experiments/e9-retrofit/report/toy-{tag}") and f"-{tag}-" in job
            assert output.name != "toy"                                          # never the stage's base report


def test_step3_analysis_reads_the_chain_outputs(world, chain_items, tmp_path, monkeypatch) -> None:
    from vsa_embed.experiments import e9_binding_chain as ch
    monkeypatch.setattr(ch, "_item_atoms", lambda ontology, track, family: _toy_item_atoms(world))
    settings = ch.ChainSettings(cap=50, paths=50, reverse=60, memory_targets=20, global_paths=20, path_order=100)
    for name in ("C5", "C5ut", "C5tr"):
        ch.chain_run(world["runs"][name], items=chain_items, settings=settings, overwrite=True, log=lambda *_: None)
        arrays = ch.load_chain(world["runs"][name] / ch.OUTPUT)["paths"]
        assert arrays["global_chain_hit"].size == min(20, arrays["path_chain_hit"].size) and arrays["query_rr"].size > 0
    analysis = br.analyse(world["root"] / "runs" / "toy", resamples=50)
    step = analysis["hosts"]["fake"]["step3"]
    assert step["available"] and step["C1"]["available"] and step["C1"]["seeds"] == [1]
    assert set(step["C2"]) == {"C5 − C5ut", "C5 − C5tr"} and set(step["C3"]) == {"C5 − C5ut", "C5 − C5tr"}
    assert all("p_holm" in v for v in step["C3"].values() if v.get("available"))
    assert {"C5", "C5ut", "C5tr"} <= set(step["descriptive"]) and "exact" in step["descriptive"]["C5"]["fidelity_mrr"]
    text = br.render(analysis, title="toy")
    assert "Step 3 — chains, reverse lookup, capacity" in text and "| C1 | C5: local − global" in text


def test_the_binding_report_refuses_the_stage_base_report_folder(world, tmp_path) -> None:
    base = tmp_path / "report" / "toy"
    base.mkdir(parents=True)
    (base / "keep.md").write_text("R9")
    with pytest.raises(SystemExit):
        br.main(["--runs", str(world["root"] / "runs" / "toy"), "--output", str(base), "--overwrite"])
    assert (base / "keep.md").read_text() == "R9"
