"""WP-PQ1 (E9 paper-quality controls, dimensions 1–2) on tiny models: the new operator families, the shuffled-frame
and row-source arms (each builds and trains a step), the row-source tables (subtoken means, definition encodings,
TransE), the filler / non-filler strata (toy masks, the trainer's opt-in key, exact replay by `e9_rescore`), the
simulated quantizers and controls (`e9_rescore` variants on the CPU), the plan wiring and the new report sections."""

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
from vsa_embed.evaluation import quantization as quant
from vsa_embed.experiments import e9_plan, e9_report, e9_rescore, e9_rowsource
from vsa_embed.relations import create_composition_operator, create_relation_transform
from vsa_embed.row_sources import FillerIndex, frames_digest, load_source_table, standardize_rows
from vsa_embed.span_channel import AliasTable
import vsa_embed.training.lm as lm


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")

# A T5-like toy glossary: terms with a type, an area, an owner and a dependency; documents verbalize the frames.
TERMS = ["Brightwater Ledger", "Plurb Standard", "Zash Team", "Grosh Console", "Varkt Crew", "Terb Template", "Skolt Handoff",
         "Noxel Index", "Quarn Portal", "Dribbet Sync"]
HELDOUT = [7, 8]
RELATIONS = ["is_a", "area", "owned_by", "uses"]
ATOMS = ["type:system", "type:policy", "type:team", "type:metric", "area:finance", "area:sales"] + [f"term:{t}" for t in TERMS]
A = {name: i for i, name in enumerate(ATOMS)}
FRAMES_BY_TERM = {
    "Brightwater Ledger": ("system", "finance", "Zash Team", "Grosh Console"),
    "Plurb Standard": ("policy", "sales", "Varkt Crew", "Terb Template"),
    "Zash Team": ("team", "finance", None, None),
    "Grosh Console": ("system", "sales", "Varkt Crew", "Dribbet Sync"),
    "Varkt Crew": ("team", "sales", None, None),
    "Terb Template": ("policy", "finance", "Zash Team", None),
    "Skolt Handoff": ("metric", "sales", "Varkt Crew", "Quarn Portal"),
    "Noxel Index": ("metric", "finance", "Zash Team", "Brightwater Ledger"),
    "Quarn Portal": ("system", "finance", "Varkt Crew", "Grosh Console"),
    "Dribbet Sync": ("system", "sales", "Zash Team", "Plurb Standard"),
}
SEQ = 32


def _frame(term: str) -> list[tuple[int, int]]:
    kind, area, owner, uses = FRAMES_BY_TERM[term]
    edges = [(0, A[f"type:{kind}"]), (1, A[f"area:{area}"])]
    if owner:
        edges.append((2, A[f"term:{owner}"]))
    if uses:
        edges.append((3, A[f"term:{uses}"]))
    return edges


def _sentence(term: str) -> str:
    kind, area, owner, uses = FRAMES_BY_TERM[term]
    text = f"{term}: a {area} {kind}" + (f" owned by the {owner}" if owner else "") + "."
    return text + (f" {term} uses {uses}." if uses else "")


def fake_host(config=None) -> torch.nn.Module:
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=2))


def _config(root: Path, name: str, channel: dict, host_mode: str = "lora", **extra) -> dict:
    train = {"micro_batch": 2, "grad_accum": 1, "total_tokens": SEQ * 2 * 4, "lr": 3e-3, "warmup_tokens": 64, "log_every": 2,
             "save_trainable_only": True, "host_lr": 1e-3, **extra.pop("train", {})}
    config = {"seed": 1, "device": "cpu", "experiment": f"e9-pq-fake-{host_mode}-{name}-s1",
              "model": {"size": "pretrained", "seq_len": SEQ, "pretrained": "fake-host", "host_mode": host_mode, "lora_rank": 2},
              "train": train,
              "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
              "eval": {"windows": 6, "batch": 3, "first_tokens": 128, "save_window_losses": True, **extra.pop("eval", {})},
              "channel": {"dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}}
    if host_mode == "frozen":
        config["train"]["eval_only"] = True
    return config


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    patch = pytest.MonkeyPatch()
    patch.setattr(lm, "build_model", fake_host)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("e9pq")
    full = AliasTable.from_pairs([(t, i) for i, t in enumerate(TERMS)], holdout=HELDOUT, include_holdout=True)
    rng = random.Random(0)
    texts = [" ".join(_sentence(rng.choice(TERMS)) for _ in range(6)) for _ in range(120)]
    build_corpus(texts, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=60_000,
                 batch_texts=8, workers=1)
    build_corpus(texts[:40], root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=60_000, batch_texts=8,
                 workers=1)
    schedule = full.entry_schedule([_frame(t) for t in TERMS])
    entries = len(full.entry_concepts)
    frequency = np.bincount(TokenCorpus.open(root / "train").spans["entry"], minlength=entries)
    ontology = {"entry_count": entries, "atomic_count": len(ATOMS), "relation_count": len(RELATIONS), "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": TERMS, "relation_names": RELATIONS, "atomic_names": ATOMS}
    torch.save(ontology, root / "ontology.pt")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    fillers = root / "fillers.pt"
    e9_rescore.build_filler_table(ontology, full, tokenizer, fillers)
    tables = {}
    for kind in e9_rowsource.KINDS:
        tables[kind] = root / f"{kind}.pt"
        e9_rowsource.build_table(kind, ontology=ontology, output=tables[kind], table=full, model=fake_host(), tokenizer=tokenizer,
                                 **({"min_steps": 40, "max_steps": 40, "log_every": 0} if kind == "kge" else {}))
    yield {"root": root, "table": full, "ontology": ontology, "tokenizer": tokenizer, "fillers": fillers, "tables": tables}
    patch.undo()


# -- 1. operators and frames ---------------------------------------------------------------------------------------------

def test_unitary_and_translation_operators() -> None:
    for dimension in (16, 15):
        transform = create_relation_transform("unitary_hrr", 3, dimension)
        x, ids = torch.randn(6, dimension), torch.tensor([0, 1, 2, 0, 1, 2])
        y = transform(ids, x)
        assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)          # orthogonal: norm-preserving
        assert torch.allclose(transform.adjoint(ids, y), x, atol=1e-5)            # adjoint = inverse
    translation = create_composition_operator("translation", 2, 8)
    x = torch.randn(3, 8)
    assert torch.allclose(translation(torch.tensor([1, 1, 0]), x) - x, translation.offsets[torch.tensor([1, 1, 0])])
    assert torch.equal(translation.adjoint(torch.tensor([1, 1, 0]), x), x)
    composer = FrameComposer(FrameSchedule.from_frames([[(0, 1), (1, 2)]]), 4, 2, 16, operator="random_fixed:unitary_hrr")
    assert composer.transform.family == "unitary_hrr" and not any(p.requires_grad for p in composer.transform.parameters())
    assert composer.parameter_groups()["relation"] == 0


def test_shuffled_frames_permute_the_frames(world) -> None:
    onto = world["ontology"]
    schedule = FrameSchedule(onto["offsets"], onto["relations"], onto["fillers"])
    shuffled = lm.frame_variant(schedule, "shuffled", onto["relation_count"], seed=1)
    frame = lambda s, i: list(zip(s.relations[s.offsets[i]:s.offsets[i + 1]].tolist(), s.fillers[s.offsets[i]:s.offsets[i + 1]].tolist()))
    originals = [frame(schedule, i) for i in range(schedule.concept_count)]
    news = [frame(shuffled, i) for i in range(schedule.concept_count)]
    assert sorted(map(tuple, originals)) == sorted(map(tuple, news))                   # the same frames, permuted
    sources = [originals.index(f) for f in news]
    assert sorted(sources) == list(range(len(sources))) and all(s != i for i, s in enumerate(sources))   # a derangement


# -- 2. row-source tables ----------------------------------------------------------------------------------------------

def test_subtoken_mean_rows_are_subtoken_means(world) -> None:
    table, tokenizer, onto = world["table"], world["tokenizer"], world["ontology"]
    weight = fake_host().get_input_embeddings().weight.detach()
    surfaces = e9_rowsource.entry_surfaces(table, onto)
    assert surfaces[0] == ["Brightwater Ledger"]                                        # display casing of the alias
    raw, info = e9_rowsource.subtoken_mean_rows(weight, tokenizer, surfaces)
    for entry, names in enumerate(surfaces):
        expected = torch.stack([weight[tokenizer.encode(" " + n)].mean(0) for n in names]).mean(0)
        assert torch.allclose(raw[entry], expected, atol=1e-6)
    rows, record = load_source_table(world["tables"]["subtoken_mean"], onto)
    reference = [e for e in range(onto["entry_count"]) if e not in set(onto["heldout_entries"])]
    standardized, _ = standardize_rows(raw, reference)
    assert torch.allclose(rows, standardized, atol=2e-3) and torch.allclose(rows.norm(dim=-1), torch.ones(len(rows)), atol=2e-3)
    assert record["frames_sha256"] == frames_digest(onto) and record["kind"] == "subtoken_mean"


def test_definition_and_kge_tables(world) -> None:
    onto = world["ontology"]
    text = e9_rowsource.verbalize_frame(_frame("Brightwater Ledger"), onto, None)
    assert text == "It is a system. It is area finance. It is owned by Zash Team. It uses Grosh Console."
    assert "Brightwater" not in text                                                    # the name is not in the encoded text
    graph = e9_rowsource.kge_graph(onto)
    assert graph["unified_atomics"] == len(TERMS) and graph["entities"] == onto["entry_count"] + 6
    first = int(onto["offsets"][world["table"].alias_to_entry["brightwater ledger"]])
    assert graph["tails"][first + 2] == world["table"].alias_to_entry["zash team"]       # term fillers are the terms' entities
    for kind in ("definition", "kge"):
        rows, record = load_source_table(world["tables"][kind], onto)
        assert rows.shape[0] == onto["entry_count"] and torch.isfinite(rows).all()
    assert load_source_table(world["tables"]["kge"])[1]["meta"]["kge"]["steps"] == 40
    bad = dict(onto, fillers=onto["fillers"].flip(0))
    with pytest.raises(ValueError, match="other ontology frames"):
        load_source_table(world["tables"]["kge"], bad)


# -- 3. every arm builds and trains a step ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", e9_plan.ARMS)
def test_each_arm_builds_and_trains(world, arm, tmp_path) -> None:
    root = world["root"]
    source = None
    if arm in e9_plan.ROW_SOURCE_ARMS:
        kind = e9_plan.ROW_SOURCE_ARMS[arm]
        source = {"source_kind": kind, "source_table": str(world["tables"][kind]),
                  "source_hidden": e9_rowsource.matched_hidden(e9_plan.channel_budget(world["ontology"], 32, 16),
                                                               32 if kind != "kge" else 256, 32)}
    spec = e9_plan.model_spec(arm, free_dimension=8, gate_bias=0.0, source=source)["channel"]
    config = _config(root, arm, {**spec, "dimension": 16})
    lm.train(config, tmp_path / arm)
    model = lm.load_final(tmp_path / arm / "final.pt")
    channel = model.channel
    rows = channel.rows({"entry": torch.arange(world["ontology"]["entry_count"])})
    assert rows.shape == (world["ontology"]["entry_count"], 32) and torch.isfinite(rows).all()
    if arm in e9_plan.ROW_SOURCE_ARMS:
        table, _ = load_source_table(world["tables"][e9_plan.ROW_SOURCE_ARMS[arm]])
        assert channel.mode == "source" and torch.equal(channel.source_rows, table)
        assert not any("source_rows" in k for k in torch.load(tmp_path / arm / "final.pt", weights_only=False)["model"])
        assert channel.source_projector[0].out_features == source["source_hidden"]
        projector = sum(p.numel() for p in channel.source_projector.parameters())
        if source["source_hidden"] > 8:                                                      # (the minimum width aside)
            assert abs(projector - e9_plan.channel_budget(world["ontology"], 32, 16)) <= (table.shape[1] + 32) / 2
        new = channel.add_entries(2)                                                         # new words: mean source row
        assert torch.allclose(channel.source_rows[new[0]], table.mean(0), atol=1e-6)
        return
    composer = channel.composer
    family = {"C5rf": "unitary_hrr", "C5ut": "additive", "C5tr": "translation", "C5sh": "hrr"}[arm]
    assert composer.transform.family == family
    if arm == "C5rf":
        assert not any(p.requires_grad for p in composer.transform.parameters())
    onto = world["ontology"]
    same = torch.equal(composer.schedule.fillers.cpu(), onto["fillers"]) and torch.equal(composer.schedule.offsets.cpu(), onto["offsets"])
    assert same == (arm != "C5sh")


# -- 4. filler strata -----------------------------------------------------------------------------------------------------

def test_filler_masks_on_a_toy_example() -> None:
    # Entry 0's frame has fillers {0, 1}; entry 1 (held out) has {1}. Atomic 0 is the sequence (7, 8), atomic 1 is (9,).
    index = FillerIndex([0, 2, 3], [0, 1, 1], [[[7, 8]], [[9]], [[50]]])
    ids = torch.tensor([[1, 2, 7, 8, 7, 8, 5, 9, 9, 6, 7, 8, 9, 4, 4, 4, 4, 4, 4, 4]])
    # span A: tokens 1..3 (entry 0), injected at 3; its window holds targets 4..11. Span B: token 6 (entry 1), targets 7..14.
    spans = {"batch": torch.tensor([0, 0]), "start": torch.tensor([1, 6]), "end": torch.tensor([3, 6]), "inject": torch.tensor([3, 6]),
             "entry": torch.tensor([0, 1]), "length": torch.tensor([3, 1]), "confidence": torch.ones(2)}
    masks = lm.stratum_masks(ids, spans, np.array([5, 0]), {1}, index)
    targets = lambda name: sorted((masks[name][0].nonzero().flatten() + 1).tolist())
    # A: (7, 8) at 4–5 and at 10–11 (the occurrence at 2–3 starts inside the span: not counted), (9) at 7, 8.
    assert targets("after_rare_seen") == list(range(4, 12))
    assert targets("after_rare_seen_filler") == [4, 5, 7, 8, 10, 11]
    assert targets("after_rare_seen_nonfiller") == [6, 9]
    # B (held out): only (9) counts — at 7, 8, 12; (7, 8) at 10–11 is not a filler of entry 1.
    assert targets("after_heldout_filler") == [7, 8, 12] and targets("after_heldout_nonfiller") == [9, 10, 11, 13, 14]
    assert targets("after_filler") == [4, 5, 7, 8, 10, 11, 12]                     # the union over spans
    assert set(targets("after_filler")) | set(targets("after_nonfiller")) == set(targets("after"))
    assert not set(targets("after_filler")) & set(targets("after_nonfiller"))
    plain = lm.stratum_masks(ids, spans, np.array([5, 0]), {1})
    assert list(plain) == list(masks)[:len(plain)] and all(torch.equal(plain[k], masks[k]) for k in plain)


def test_filler_table_surfaces(world) -> None:
    data = torch.load(world["fillers"], weights_only=False)
    surfaces = dict(zip(ATOMS, data["surfaces"]))
    assert surfaces["term:Zash Team"] == ["Zash Team"] and surfaces["type:system"] == ["system"]
    tokenizer = world["tokenizer"]
    sequences = {tuple(s) for s in data["sequences"][A["term:Zash Team"]]}
    assert tuple(tokenizer.encode(" Zash Team")) in sequences and tuple(tokenizer.encode("zash team")) in sequences


def test_trainer_filler_strata_and_exact_rescoring(world, tmp_path) -> None:
    root = world["root"]
    opt_in = _config(root, "C5", {"mode": "compose", "composition": "attentive", "context_window": 4},
                     eval={"filler_strata": str(world["fillers"])})
    lm.train(opt_in, tmp_path / "c5")
    plain = _config(root, "C0p", {"mode": "none"})
    lm.train(plain, tmp_path / "c0")
    trained = lm.load_window_losses(tmp_path / "c5" / "eval_windows.npz")
    assert "after_filler" in trained["strata"] and trained["strata"][:len(lm.AFTER_STRATA) + 3][-1] == "unlinked"
    assert "after_filler" not in lm.load_window_losses(tmp_path / "c0" / "eval_windows.npz")["strata"]
    final_tokens = max(trained["evals"])
    for run in ("c5", "c0"):
        record = e9_rescore.score_run(tmp_path / run, variants=["ref"], fillers=world["fillers"], device="cpu")
        check = record["ref_check"]
        assert check["paired"] and check["counts_equal"] and check["max_abs_window_sum_diff"] == 0.0
    found = e9_rescore.load_rescore(tmp_path / "c5" / "rescore")
    sums, counts = trained["evals"][final_tokens]
    for name in ("after_filler", "after_nonfiller", "after_heldout_filler"):
        i, j = trained["strata"].index(name), found["strata"].index(name)
        assert np.array_equal(counts[i], found["count"][j]) and np.array_equal(sums[i], found["sums"]["ref"][j])
    assert (tmp_path / "c5" / "rescore" / "manifest.json").exists() and (tmp_path / "c5" / "rescore" / "resolved_config.yaml").exists()
    c0 = e9_rescore.load_rescore(tmp_path / "c0" / "rescore")
    assert np.array_equal(c0["count"], found["count"]) and c0["strata"] == found["strata"]   # filler masks are model-independent


def test_rescore_variants_and_quantizers_on_cpu(world, tmp_path) -> None:
    root = world["root"]
    lm.train(_config(root, "C5", {"mode": "compose", "composition": "attentive", "context_window": 4}), tmp_path / "c5")
    lm.train(_config(root, "C0p", {"mode": "none"}), tmp_path / "c0")
    variants = ["ref", "ref-off", "int4-rtn", "int4-hqq", "int4-nf4", "int4-gptq", "int4-awq", "int4-rtn-emb", "int4-gptq-off"]
    record = e9_rescore.score_run(tmp_path / "c5", variants=variants, fillers=world["fillers"], device="cpu", group_size=16,
                                  calibration_windows=4)
    found = e9_rescore.load_rescore(tmp_path / "c5" / "rescore")
    assert sorted(found["sums"]) == sorted(variants)
    losses = {v: record["variants"][v]["strata"]["all"]["loss"] for v in variants}
    assert all(np.isfinite(x) for x in losses.values()) and len({round(x, 6) for x in losses.values()}) == len(variants)
    assert record["variants"]["ref-off"]["channel_off"] and record["variants"]["int4-gptq"]["quantization"]["scheme"] == "gptq"
    assert record["variants"]["int4-gptq"]["quantization"]["calibration"]["windows"] == 4
    assert record["variants"]["int4-rtn-emb"]["quantization"]["head_tied"]
    again =e9_rescore.score_run(tmp_path / "c5", variants=[*variants, "int4-awq-off"], fillers=world["fillers"], device="cpu",
                                 group_size=16, calibration_windows=4, resume=True)
    assert sorted(again["variants"]) == sorted([*variants, "int4-awq-off"])
    plain = e9_rescore.score_run(tmp_path / "c0", variants=["ref", "ref-off", "int4-nf4"], fillers=world["fillers"], device="cpu")
    assert sorted(plain["variants"]) == ["int4-nf4", "ref"] and "off_variants" in plain
    with pytest.raises(FileExistsError):
        e9_rescore.score_run(tmp_path / "c0", variants=["ref"], fillers=world["fillers"], device="cpu")
    with pytest.raises(ValueError, match="unknown variant"):
        e9_rescore.parse_variant("int3-A")
    assert e9_rescore.parse_variant("int4-A-emb-off") == {"base": "int4-A", "emb": "emb", "off": True, "name": "int4-A-emb-off"}
    assert e9_rescore.profile_variants("auto", "C5")[:2] == ["ref", "ref-off"] and e9_rescore.profile_variants("auto", "C5rf") == ["ref", "int4-A"]


def test_quantizer_numerics() -> None:
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import NF4Tensor
    torch.manual_seed(0)
    w = torch.randn(256, 512)
    assert torch.equal(quant.nf4_fake_quantize(w), NF4Tensor.from_tensor(w.bfloat16(), 64, 256).get_original_weight().float())
    assert torch.isfinite(quant.nf4_fake_quantize(torch.randn(48, 576))).all()          # shapes torchao refuses
    x = torch.randn(1024, 64) * torch.linspace(0.1, 3.0, 64)
    weight = torch.randn(48, 64)
    hessian = 2 * x.T @ x / len(x)
    error = lambda q: float(((x @ q.T - x @ weight.T) ** 2).mean())
    assert error(quant.gptq_quantize_weight(weight, hessian, 32)) < error(quant.int4_fake_quantize(weight, 32))
    codes = quant.int4_fake_quantize(weight, 32).reshape(48, 2, 32)
    assert max(len(torch.unique(g)) for g in codes.reshape(-1, 32)) <= 16                   # 4-bit grid per group
    hqq = quant.hqq_fake_quantize(weight, 32)
    assert float((hqq - weight).abs().mean()) <= float((quant.int4_fake_quantize(weight, 32) - weight).abs().mean()) * 1.05


# -- 5. the plan ------------------------------------------------------------------------------------------------------------

@pytest.fixture()
def plan_root(tmp_path: Path) -> dict:
    torch.save({"entry_count": 4200, "atomic_count": 3493, "relation_count": 23}, tmp_path / "counts.pt")
    return {"root": tmp_path / "e9", "counts": tmp_path / "counts.pt", "data": tmp_path / "smollm2"}


def _plan(plan_root: dict, **kw) -> list[Path]:
    return e9_plan.write_stage("main", hosts=kw.pop("hosts", ["SmolLM2-360M"]), models=kw.pop("models", list(e9_plan.ARMS)),
                               seeds=kw.pop("seeds", [1]), data_root=plan_root["data"], counts_ontology=plan_root["counts"],
                               root=plan_root["root"], track="t5", **kw)


def test_e9_plan_arms_configs(plan_root, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(e9_rowsource, "ROOT", tmp_path / "row-sources")
    paths = _plan(plan_root, models=["C5", *e9_plan.ARMS], seeds=[1, 2])
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    c5 = configs["SmolLM2-360M-full-C5-s1"]
    for arm, change in e9_plan.C5_ABLATIONS.items():
        config = configs[f"SmolLM2-360M-full-{arm}-s1"]
        assert {k: v for k, v in config["channel"].items() if k not in change} == {k: v for k, v in c5["channel"].items() if k not in change}
        assert all(config["channel"][k] == v for k, v in change.items())
        assert {k: v for k, v in config.items() if k not in {"channel", "experiment"}} == {k: v for k, v in c5.items() if k not in {"channel", "experiment"}}
    budget = (3493 + 23) * 256 + 256 * 960
    assert budget == e9_plan.channel_budget(plan_root["counts"], 960)
    for arm, kind in e9_plan.ROW_SOURCE_ARMS.items():
        channel = configs[f"SmolLM2-360M-full-{arm}-s2"]["channel"]
        source_width = 256 if kind == "kge" else 960
        assert channel["mode"] == "source" and channel["source_kind"] == kind and channel["gate_bias"] == 0.0
        assert channel["source_table"] == str(e9_rowsource.table_path("t5", "smollm2", kind, "SmolLM2-360M"))
        assert abs(channel["source_hidden"] * (source_width + 960) - budget) <= (source_width + 960) / 2
    with pytest.raises(ValueError, match="row-source table"):
        e9_plan.model_spec("C6m", free_dimension=8, gate_bias=0.0)
    qwen = e9_plan.run_config(stage="s", host="SmolLM2-360M", mode="train", model="C5tr", seed=1, data_root=plan_root["data"],
                              tokens=1000, lora_rank=64, host_lr=None, gate_bias=0.0, free_dimension=8, channel_extra={"scale_to_host": True})[1]
    assert qwen["channel"]["scale_to_host"] and qwen["channel"]["operator"] == "translation"


def test_e9_plan_arms_queue(plan_root, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(e9_rowsource, "ROOT", tmp_path / "row-sources")               # no table exists yet
    paths = _plan(plan_root, seeds=[1])
    queue = tmp_path / "jobs"
    queued = e9_plan.queue_jobs(paths, "main", 51, root=plan_root["root"], queue_dir=queue, track="t5", alias_table=Path("t5.json"))
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    order = [j["name"] for j in sorted(jobs.values(), key=lambda j: (j["priority"], j["created"]))]
    builds = [n for n in order if "rowsource" in n]
    assert builds and order[:len(builds)] == builds and all(jobs[n]["priority"] == 51 for n in builds)   # tables first
    stem = "SmolLM2-360M-full-C5rf-s1"
    assert {n for n in jobs if n.startswith(f"main-{stem}-")} == {f"main-{stem}-{s}" for s in ("zeroshot", "edit", "rescore")}
    rescore = jobs[f"main-{stem}-rescore"]["command"]
    assert rescore[2] == "vsa_embed.experiments.e9_rescore" and rescore[rescore.index("--variants") + 1:][:2] == ["ref", "int4-A"]
    assert jobs[f"main-{stem}-rescore"]["priority"] == 52 and "--quantize" not in jobs[f"main-{stem}-edit"]["command"]
    shuffled = jobs["main-SmolLM2-360M-full-C5sh-s1-zeroshot"]["command"]
    assert shuffled[shuffled.index("--sources") + 1] == "own,none,mean_row"
    assert "--edit-items" not in jobs["main-SmolLM2-360M-full-C5sh-s1-edit"]["command"]
    assert "--edit-items" in jobs[f"main-{stem}-edit"]["command"]
    assert not any("quant" in n for n in jobs) and jobs["main-report-s1-pq"]["priority"] == 54   # no e4_quant for arms
    base = _plan(plan_root, models=list(e9_plan.MODELS), seeds=[2])
    more = e9_plan.queue_jobs(base, "main", 51, root=plan_root["root"], queue_dir=queue, track="t5", alias_table=Path("t5.json"),
                              rescore="all")
    assert "main-SmolLM2-360M-full-C5-s2-rescore" in more and "main-quant-s2" in more
    command = json.loads((queue / "main-SmolLM2-360M-full-C5-s2-rescore.json").read_text())["command"]
    assert "int4-gptq" in command and "ref-off" in command
    default = e9_plan.queue_jobs(_plan(plan_root, models=list(e9_plan.MODELS), seeds=[3]), "main", 51, root=plan_root["root"],
                                 queue_dir=tmp_path / "jobs-default", track="t5", alias_table=Path("t5.json"))
    assert not any("rescore" in n or "rowsource" in n for n in default)                 # defaults unchanged


def test_rescore_queue_for_a_stage(plan_root, tmp_path) -> None:
    _plan(plan_root, models=["P0", "C0p", "C5", "C5rf"], seeds=[1])
    queued = e9_rescore.queue_stage("main", priority=51, root=plan_root["root"], queue_dir=tmp_path / "jobs")
    assert sorted(queued) == sorted(f"main-SmolLM2-360M-{m}-s1-rescore" for m in ("frozen-P0", "full-C0p", "full-C5", "full-C5rf"))
    job = json.loads((tmp_path / "jobs" / "main-SmolLM2-360M-frozen-P0-s1-rescore.json").read_text())
    assert job["priority"] == 51 and "int4-awq" in job["command"] and job["command"][0] == sys.executable
    assert e9_rescore.queue_stage("main", root=plan_root["root"], queue_dir=tmp_path / "jobs") == []          # idempotent


# -- 6. the report sections --------------------------------------------------------------------------------------------------

def _synthetic(root: Path, name: str, seed: int, sums: np.ndarray, counts: np.ndarray, strata: list[str]) -> Path:
    folder = root / f"host-full-{name}-s{seed}"
    folder.mkdir(parents=True)
    config = {"experiment": f"e9-syn-host-full-{name}-s{seed}", "seed": seed,
              "model": {"pretrained": "org/host", "host_mode": "frozen" if name == "P0" else "train", "seq_len": 8},
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


def test_report_pq_sections_on_synthetic_inputs(tmp_path) -> None:
    rng = np.random.default_rng(0)
    strata = ["all", "unlinked", "inside", "after", "after_filler", "after_nonfiller", "after_heldout", "after_heldout_filler",
              "after_heldout_nonfiller", "after_rare_seen"]
    windows = 30
    counts = rng.integers(5, 20, size=(len(strata), windows)).astype(np.int32)
    # filler / non-filler partition `after` and `after_heldout`
    for base in ("after", "after_heldout"):
        counts[strata.index(base)] = counts[strata.index(base + "_filler")] + counts[strata.index(base + "_nonfiller")]
    is_filler = np.array([s.endswith("_filler") for s in strata])[:, None]
    linked = np.array([s not in {"all", "unlinked"} for s in strata])[:, None]
    # bf16: C0' 2.9 everywhere; C5 0.3 lower on filler tokens only; arms in between; quantization (int4-A) adds 0.2 to
    # C0' and 0.1 to C5; C5 with the channel off: 2.9 at bf16 and +0.15 at int4.
    level = {"P0": (3.0, 0.0), "C0p": (2.9, 0.0), "C2": (2.88, 0.0), "C5": (2.9, -0.3), "C5rf": (2.9, -0.25), "C6m": (2.9, -0.1)}
    runs = tmp_path / "runs"
    for name, (value, filler_gain) in level.items():
        loss = np.full(counts.shape, value) + np.where(is_filler, filler_gain, 0.0)
        for base in ("after", "after_heldout"):   # totals are the token-weighted mixtures of their parts
            f, nf = strata.index(base + "_filler"), strata.index(base + "_nonfiller")
            loss[strata.index(base)] = (loss[f] * counts[f] + loss[nf] * counts[nf]) / counts[strata.index(base)]
        sums = loss * counts + rng.normal(0, 0.002, size=counts.shape) * counts
        folder = _synthetic(runs, name, 1, sums, counts, strata)
        damage = 0.1 if name == "C5" else 0.2
        arrays = {"ref": sums, "int4-A": sums + damage * counts, "int4-hqq": sums + damage * counts}
        if name in {"C5", "C2"}:
            off = np.full(counts.shape, value) * counts
            arrays.update({"ref-off": off, "int4-A-off": off + 0.15 * counts})
        rescore = folder / "rescore"
        rescore.mkdir()
        e9_rescore._write_windows(rescore / "windows.npz", strata, list(range(windows)), counts, arrays)
        (rescore / "rescore.json").write_text(json.dumps({"channel": "none" if name in {"P0", "C0p"} else "compose"}))
    summary = e9_report.write_report([runs], tmp_path / "report", resamples=400, figures=False)
    pq = summary["groups"]["host · train"]["pq"]
    ablation = pq["operator_ablation"]["strata"]["after_heldout"]["C5rf"]["candidate_minus_arm"]
    assert ablation["delta"] < 0                                                          # C5 better than its ablation
    assert pq["row_sources"]["arms"] == ["C6m"]
    split = pq["filler_split"]["rows"]["after"]["C0'"]
    assert split["filler"]["mean"] == pytest.approx(-0.3, abs=0.01) and split["nonfiller"]["mean"] == pytest.approx(0.0, abs=0.01)
    assert split["filler_gain_share"] == pytest.approx(1.0, abs=0.05) and 0 < split["filler_target_share"] < 1
    controls = pq["claim_b_controls"]["references"]["C0'"]["int4-A"]["after_heldout"]
    assert controls["did"]["gain_change"]["mean"] == pytest.approx(-0.1, abs=1e-3)
    assert controls["did_off"]["gain_change"]["mean"] == pytest.approx(-0.05, abs=1e-3)
    assert controls["channel"]["gain_change"]["mean"] == pytest.approx(-0.05, abs=1e-3)
    assert controls["cells"]["reference_q"] == pytest.approx(3.1, abs=0.01)
    assert "int4-hqq" in pq["claim_b_controls"]["quantizers"] and "did_off" not in pq["claim_b_controls"]["references"]["C0'"]["int4-hqq"]["all"]
    report = (tmp_path / "report" / "report.md").read_text()
    for heading in ("operator and specificity ablation", "same-site row sources", "filler vs non-filler", "claim-B controls"):
        assert heading in report
