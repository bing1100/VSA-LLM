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
    runs = {}
    for name, (channel, mode) in SPECS.items():
        folder = root / "runs" / "toy" / f"fake-{mode}-{name}-s1"
        lm.train(_config(root, name, channel, mode), folder)
        runs[name] = folder
    ctx = und.BuildContext(track="toy", family="gpt2", ontology=ontology, table=full, lexicon=lexicon, tokenizer=tokenizer,
                           tokenizer_name="gpt2", exclusions=torch.load(root / "exclusions.pt", weights_only=False), families_spec=SPEC,
                           min_subtokens=1, ontology_path=root / "ontology.pt", alias_table_path=root / "alias_table.json")
    yield {"root": root, "table": full, "ontology": ontology, "runs": runs, "ctx": ctx, "alias": root / "alias_table.json"}
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
    assert all(j["name"].endswith("-role-twins-toy-v1") for j in items)
