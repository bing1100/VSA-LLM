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
                                 **({"min_steps": 40, "max_steps": 40, "log_every": 0} if kind in e9_rowsource.KGE_KINDS else {}))
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
    source = hybrid = buckets = None
    if arm in e9_plan.HYBRID_ARMS:
        from vsa_embed.experiments.e4_plan import matched_hybrid_widths
        hybrid = {**matched_hybrid_widths(world["ontology"], 32, 16, 8)[e9_plan.HYBRID_ARMS[arm]]["channel"]}
    if arm in e9_plan.HASHED_ARMS:
        buckets = 64
    if arm in e9_plan.ROW_SOURCE_ARMS:
        kind = e9_plan.ROW_SOURCE_ARMS[arm]
        source = {"source_kind": kind, "source_table": str(world["tables"][kind]),
                  "source_hidden": e9_rowsource.matched_hidden(e9_plan.channel_budget(world["ontology"], 32, 16),
                                                               32 if kind not in e9_rowsource.KGE_KINDS else 256, 32)}
    spec = e9_plan.model_spec(arm, free_dimension=8, gate_bias=0.0, source=source, hybrid=hybrid, buckets=buckets)["channel"]
    if str(spec.get("frames", "")).startswith("relations:"):                     # T4 relation names → the toy world's
        spec["frames"] = "relations:" + ",".join(n for n in spec["frames"].split(":", 1)[1].split(",") if n in RELATIONS) + \
                         ("" if arm == "C5isa" else ",area")
    config = _config(root, arm, {**spec, **({} if arm in e9_plan.HYBRID_ARMS else {"dimension": 16})})
    lm.train(config, tmp_path / arm)
    model = lm.load_final(tmp_path / arm / "final.pt")
    channel = model.channel
    if arm in e9_plan.HASHED_ARMS:                                                           # C1h: rows keyed by token sequence
        assert channel.mode == "hashed" and channel.hashed_buckets == 64
        return
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
    if arm in e9_plan.HYBRID_ARMS:                                                           # H5add: composed + free row
        assert channel.mode == "compose_add" and channel.composer is not None
        return
    if arm in e9_plan.FREE_ARMS:                                                             # C2w: a free table at C5's width
        assert channel.mode == "free" and channel.table.weight.shape[1] == e9_plan.FREE_ARMS[arm]["free_dimension"]
        return
    composer = channel.composer
    family = {"C5rf": "unitary_hrr", "C5ut": "additive", "C5tr": "translation", "C5sh": "hrr", "C5ld": "hrr", "C5isa": "hrr",
              "C5comp": "hrr", "C5half": "hrr", "C5w512": "hrr", **{a: "hrr" for a in e9_plan.ONTOLOGY_ARMS}}[arm]
    assert composer.transform.family == family
    if arm == "C5rf":
        assert not any(p.requires_grad for p in composer.transform.parameters())
    onto = world["ontology"]
    same = torch.equal(composer.schedule.fillers.cpu(), onto["fillers"]) and torch.equal(composer.schedule.offsets.cpu(), onto["offsets"])
    shuffled = arm == "C5sh" or (arm in e9_plan.ONTOLOGY_ARMS and e9_plan.ONTOLOGY_ARMS[arm][1] == "C5sh")
    assert same == (not shuffled and arm not in {"C5isa", "C5comp", "C5half"})


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
    # --redo replaces a variant in place (the 2026-10-12 AWQ grouping fix) and records it; the others are kept
    before = e9_rescore.load_rescore(tmp_path / "c5" / "rescore")
    redone = e9_rescore.score_run(tmp_path / "c5", variants=["int4-awq"], fillers=world["fillers"], device="cpu", group_size=16,
                                  calibration_windows=4, resume=True, redo=["int4-awq"])
    after = e9_rescore.load_rescore(tmp_path / "c5" / "rescore")
    assert "int4-awq" in redone["redone"] and redone["redone"]["int4-awq"]["previous_all"] == pytest.approx(losses["int4-awq"])
    assert sorted(after["sums"]) == sorted(before["sums"]) and np.array_equal(after["sums"]["int4-gptq"], before["sums"]["int4-gptq"])
    assert redone["variants"]["int4-awq"]["quantization"]["awq_grouping"].startswith("input tensor object")
    with pytest.raises(ValueError, match="redo needs resume"):
        e9_rescore.score_run(tmp_path / "c5", variants=["int4-awq"], fillers=world["fillers"], device="cpu", redo=["int4-awq"])
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
    # calibration statistics are float32 even inside the forward's bf16 autocast region (a bf16 Gram matrix is not PSD)
    layer = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 8))

    def forward(step):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            step(lambda: layer(x[:256]))

    stats = quant._collect([("1", layer[1])], forward, "gptq", 64, 0)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = layer[0](x[:256])
    hidden = hidden.float()
    assert torch.allclose(stats["1"]["H"], hidden.T @ hidden, rtol=1e-5, atol=1e-4)            # float32, not bf16
    assert float(torch.linalg.eigvalsh(stats["1"]["H"].double()).min()) > -1e-6 * float(stats["1"]["H"].diag().max())
    singular = torch.zeros(16, 16)
    singular[0, 0] = 1.0                                                                   # rank 1: needs more damping
    assert torch.isfinite(quant._damped_inverse_cholesky(singular, 0.0 + 1e-12)).all()
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
    arms = [a for a in e9_plan.ARMS if a not in e9_plan.HYBRID_ARMS]       # hybrids need a full ontology (frames) to match widths
    paths = _plan(plan_root, models=["C5", *arms], seeds=[1, 2])
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
        source_width = 256 if kind in e9_rowsource.KGE_KINDS else 960
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
    paths = _plan(plan_root, models=[a for a in e9_plan.ARMS if a not in e9_plan.HYBRID_ARMS], seeds=[1])
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
    assert {n for n in jobs if n.startswith("main-SmolLM2-360M-full-C1h-s1-")} == {"main-SmolLM2-360M-full-C1h-s1-rescore"}  # no frames
    shuffled = jobs["main-SmolLM2-360M-full-C5sh-s1-zeroshot"]["command"]
    assert shuffled[shuffled.index("--sources") + 1] == "own,none,mean_row"
    assert "--edit-items" not in jobs["main-SmolLM2-360M-full-C5sh-s1-edit"]["command"]
    assert "--edit-items" in jobs[f"main-{stem}-edit"]["command"]
    assert not any("quant" in n for n in jobs) and jobs["main-report-s1-pq-SmolLM2-360M"]["priority"] == 54   # no e4_quant for arms
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


# -- 2026-10-11 addendum: link dropout and the inductive TransE table -------------------------------------------------

def test_drop_links_is_deterministic_and_drops_whole_spans() -> None:
    n = 4000
    spans = {"batch": torch.arange(n) % 8, "start": torch.arange(n), "end": torch.arange(n) + 1, "entry": torch.arange(n) % 50,
             "confidence": torch.ones(n)}
    a = lm.drop_links(spans, 0.3, seed=1, step=5, micro_step=2)
    b = lm.drop_links(spans, 0.3, seed=1, step=5, micro_step=2)
    c = lm.drop_links(spans, 0.3, seed=2, step=5, micro_step=2)
    assert all(torch.equal(a[k], b[k]) for k in spans) and not torch.equal(a["start"], c["start"])
    assert set(a) == set(spans) and len({v.shape[0] for v in a.values()}) == 1          # every field filtered alike
    assert torch.equal(a["end"], a["start"] + 1)                                         # rows stay aligned
    assert abs(a["entry"].shape[0] / n - 0.7) < 0.03
    assert lm.drop_links({k: v[:0] for k, v in spans.items()}, 0.3, seed=1, step=0, micro_step=0)["entry"].numel() == 0
    with pytest.raises(ValueError):
        lm.drop_links(spans, 1.0, seed=1, step=0, micro_step=0)


def test_kge_inductive_excludes_heldout_and_folds_in(world) -> None:
    onto = world["ontology"]
    heldout = set(int(e) for e in onto["heldout_entries"])
    full, inductive = e9_rowsource.kge_graph(onto), e9_rowsource.kge_graph(onto, exclude=heldout)
    out = torch.as_tensor(sorted(heldout))
    assert not torch.isin(inductive["heads"], out).any() and not torch.isin(inductive["tails"], out).any()
    assert inductive["excluded_triples"] == full["heads"].numel() - inductive["heads"].numel() > 0
    # the fold-in is the exact L1 fit: the per-dimension median of t - r over the entry's triples
    entity, relation = torch.randn(full["entities"], 6), torch.randn(onto["relation_count"], 6)
    reference = [e for e in range(onto["entry_count"]) if e not in heldout]
    folded, record = e9_rowsource.fold_in_transe(entity, relation, full, sorted(heldout), reference)
    for e in sorted(heldout):
        rows = (full["heads"] == e) & ~torch.isin(full["tails"], out)
        if rows.any():
            want = (entity[full["tails"][rows]] - relation[full["relations"][rows]]).median(0).values
            assert torch.allclose(folded[e], want)
    keep = [e for e in range(full["entities"]) if e not in heldout]
    assert torch.equal(folded[keep], entity[keep]) and record["folded_entries"] == len(heldout)
    rows, meta = load_source_table(world["tables"]["kge_inductive"], onto)
    assert rows.shape[0] == onto["entry_count"] and torch.isfinite(rows).all()
    assert meta["meta"]["fold_in"]["folded_entries"] == len(heldout) and meta["meta"]["excluded_triples"] > 0
    assert e9_rowsource.table_path("t7rood", "smollm2", "kge_inductive").name == "kge-transe-inductive-d256.pt"
    assert e9_plan.row_source("C6gi", track="t7rood", family="smollm2", host="SmolLM2-360M", budget=10_000)["source_kind"] == "kge_inductive"


def test_addendum_arm_specs() -> None:
    assert e9_plan.model_spec("C2w", free_dimension=34, gate_bias=0.0)["channel"] == {"mode": "free", "free_dimension": 256,
                                                                                     "gate_bias": 0.0}
    c5ld = e9_plan.model_spec("C5ld", free_dimension=34, gate_bias=0.0)["channel"]
    assert c5ld["link_dropout"] == 0.3 and c5ld["mode"] == "compose" and c5ld["operator"] == "hrr"
    assert {"C2w", "C5ld", "C6gi"} <= set(e9_plan.ARMS) and e9_plan.batch_tag(["C2w"]) == "pq"


def test_incontext_glossary_scoring_and_report(world, tmp_path) -> None:
    import gzip
    from vsa_embed.experiments import e9_incontext
    root, onto = world["root"], world["ontology"]
    glossary = tmp_path / "verbalized.jsonl.gz"
    with gzip.open(glossary, "wt") as handle:
        for e, term in enumerate(TERMS):
            handle.write(json.dumps({"entry": e, "name": term, "heldout": e in HELDOUT, "text": f"It is item {e} of the glossary."}) + "\n")
    c5 = e9_plan.model_spec("C5", free_dimension=8, gate_bias=0.0)["channel"]
    runs = {}
    for name, channel in (("C5", {**c5, "dimension": 16}), ("C0p", {"mode": "none"})):
        for seed in (1, 2):
            run = tmp_path / "runs" / "toy" / f"fake-lora-{name}-s{seed}"
            config = _config(root, name, channel)
            config["seed"] = seed
            lm.train(config, run)
            runs[(name, seed)] = run
    outputs = []
    for (name, seed), run in runs.items():
        out = tmp_path / "incontext" / "toy" / run.name
        summary = e9_incontext.score_run(run, out, max_prefix=32, device="cpu", glossary=glossary, tokenizer=world["tokenizer"], log=lambda *_: None)
        check = summary["own_check"]
        assert check["paired"] and check["counts_equal"] and check["max_abs_window_sum_diff"] < 1e-3   # `none` replays the run
        assert summary["prefix"]["frame"]["tokens_mean"] > 0 and 0 < summary["prefix"]["frame"]["entry_coverage"] <= 1
        assert summary["prefix"]["frame"]["tokens_max"] <= 32 and summary["prefix"]["none"]["tokens_mean"] == 0
        data = e9_incontext.load_incontext(out)
        assert not np.array_equal(data["sums"]["frame"], data["sums"]["none"]) and np.array_equal(data["count"], data["count"])
        outputs.append(out)
    # positions shift by the row's prefix; nothing else moves
    spans = {"batch": torch.tensor([0, 1, 1]), "start": torch.tensor([2, 3, 9]), "end": torch.tensor([4, 5, 11]),
             "inject": torch.tensor([4, 5, 11]), "entry": torch.tensor([7, 1, 2]), "length": torch.tensor([3, 3, 3])}
    shifted = e9_incontext.shift_spans(spans, torch.tensor([10, 20]))
    assert shifted["start"].tolist() == [12, 23, 29] and shifted["inject"].tolist() == [14, 25, 31] and torch.equal(shifted["entry"], spans["entry"])
    with pytest.raises(ValueError, match="positions"):
        e9_incontext.score_run(runs[("C0p", 1)], tmp_path / "too-long", max_prefix=40, device="cpu", glossary=glossary,
                               tokenizer=world["tokenizer"], log=lambda *_: None)
    # the glossary keeps whole lines under the cap, in order of first occurrence
    ids, used = e9_incontext.build_prefix(["a: x", "b: y", "c: " + "z " * 200], world["tokenizer"], 40)
    assert used == 2 and len(ids) <= 40
    summary = e9_incontext.write_report([tmp_path / "incontext"], tmp_path / "report", resamples=200)
    group = next(iter(summary["groups"].values()))
    assert "channel (C5 − C0′)" in group["strata"]["all"] and "in-context, C0′ (C0′ frame − C0′)" in group["strata"]["all"]
    assert group["strata"]["all"]["channel (C5 − C0′)"]["seeds"] == [1, 2]
    assert (tmp_path / "report" / "report.md").read_text().startswith("# E9 in-context frame baseline")


# -- 2026-10-12: AWQ shared-input grouping by tensor identity, not by memory address -----------------------------------

class _ReuseBlock(torch.nn.Module):
    """`lin_a` reads a tensor that is freed before a same-shaped one is made for `lin_b` / `lin_c` (which share it)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin_a, self.lin_b, self.lin_c = (torch.nn.Linear(width, width) for _ in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * 2.0
        a = self.lin_a(h)
        del h
        h2 = a + 1.0                      # same shape as h: the allocator may hand it h's memory
        return self.lin_b(h2) + self.lin_c(h2)


class _ReuseHost(torch.nn.Module):
    def __init__(self, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_ReuseBlock(width) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_awq_groups_linears_by_input_tensor_not_address() -> None:
    torch.manual_seed(0)
    host = _ReuseHost(width=64)
    # (8, 256, 64) under no_grad: the CPU allocator hands h2 the freed h's address in about 9 of 10 calls (measured
    # 2026-10-12), so the old `data_ptr` key grouped lin_a with lin_b / lin_c
    batches = [torch.randn(8, 256, 64) for _ in range(3)]

    def forward(step):
        for x in batches:
            step(lambda x=x: host(x))
    info = quant.simulate_weight_only_(host, "awq", group_size=16, forward=forward, samples=64)
    assert info["awq_groups"] == {"lin_a": 3, "lin_b+lin_c": 3}            # never lin_a+lin_b+lin_c, whatever the addresses


def test_awq_groups_on_a_llama_block() -> None:
    config = transformers.LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    host = transformers.LlamaForCausalLM(config).eval()
    ids = [torch.randint(0, 128, (2, 32)) for _ in range(3)]

    def forward(step):
        for x in ids:
            step(lambda x=x: host(x))
    info = quant.simulate_weight_only_(host, "awq", group_size=32, forward=forward, samples=64)
    assert info["awq_groups"] == {"down_proj": 2, "gate_proj+up_proj": 2, "o_proj": 2, "q_proj+k_proj+v_proj": 2}


def test_thinned_frame_variants() -> None:
    frames = [[(0, 5), (1, 6), (2, 7), (0, 8)], [(1, 9)], [], [(2, 1), (2, 2), (2, 3)]]
    schedule = FrameSchedule.from_frames(frames)
    names = ["is_a", "has_role", "contains_element"]
    def as_frames(s):
        o, r, f = s.offsets.tolist(), s.relations.tolist(), s.fillers.tolist()
        return [list(zip(r[a:b], f[a:b])) for a, b in zip(o[:-1], o[1:])]
    kept = as_frames(lm.frame_variant(schedule, "relations:is_a,contains_element", 3, seed=1, relation_names=names))
    assert kept == [[(0, 5), (2, 7), (0, 8)], [], [], [(2, 1), (2, 2), (2, 3)]]
    half = as_frames(lm.frame_variant(schedule, "edge_fraction:0.5", 3, seed=1, relation_names=names))
    again = as_frames(lm.frame_variant(schedule, "edge_fraction:0.5", 3, seed=99, relation_names=names))
    assert half == again                                            # fixed thinning, independent of the run seed
    assert [len(f) for f in half] == [2, 1, 0, 2] and all(set(h) <= set(f) for h, f in zip(half, frames))
    with pytest.raises(ValueError, match="unknown relation"):
        lm.frame_variant(schedule, "relations:charge", 3, seed=1, relation_names=names)
    with pytest.raises(ValueError, match="edge_fraction"):
        lm.frame_variant(schedule, "edge_fraction:0", 3, seed=1, relation_names=names)


def test_ontology_arms_read_the_enriched_copy(plan_root) -> None:
    data = plan_root["data"]
    _, config = e9_plan.run_config(stage="s", host="SmolLM2-360M", mode="train", model="C5mesh", seed=1, data_root=data, tokens=1000,
                                   lora_rank=64, host_lr=None, gate_bias=0.0, free_dimension=8)
    assert config["data"]["ontology"] == str(data.parent / "rood-v1-meshrich" / "ontology.pt")
    assert config["data"]["train"] == str(data / "train") and config["channel"]["mode"] == "compose"
    _, shuffled = e9_plan.run_config(stage="s", host="SmolLM2-360M", mode="train", model="C5tchsh", seed=1, data_root=data, tokens=1000,
                                     lora_rank=64, host_lr=None, gate_bias=0.0, free_dimension=8)
    assert shuffled["channel"]["frames"] == "shuffled" and shuffled["data"]["ontology"].endswith("rood-v1-teacher/ontology.pt")


# -- 2026-10-20 addendum: B-B1, the bit-matched unstructured side module (C0L) ------------------------------------------

def test_side_lora_plan_adapters_merge_and_off() -> None:
    from vsa_embed.integrations.transformers import (LoRALinear, add_side_lora, lora_units, side_adapters, side_lora_off,
                                                     side_lora_plan)
    host = fake_host()
    units = lora_units(host)
    assert {layer for _, layer, _ in units} == {0, 1} and all(u > 0 for _, _, u in units)
    per_rank = sum(u for _, _, u in units)
    layer0 = sum(u for _, layer, u in units if layer == 0)
    plan = side_lora_plan(units, 2 * per_rank + layer0 + 3)                          # rank 2, +1 in the first layer
    assert (plan["rank"], plan["extra_rank_layers"], plan["parameters"]) == (2, 1, 2 * per_rank + layer0)
    assert side_lora_plan(units, 10)["rank"] == 1                                  # never below rank 1
    record = add_side_lora(host, rank=2, extra_rank_layers=1)
    adapters = side_adapters(host)
    assert record["parameters"] == plan["parameters"] == sum(m.lora_a.numel() + m.lora_b.numel() for _, m in adapters)
    assert {m.rank for name, m in adapters if ".h.0." in name} == {3} and {m.rank for name, m in adapters if ".h.1." in name} == {2}
    assert all(m.scale == 1.0 and m.side_module for _, m in adapters)             # α = rank
    for _, m in adapters:
        torch.nn.init.normal_(m.lora_b, std=0.05)
    ids = torch.arange(12).view(1, 12)
    host.eval()                                                                      # no dropout
    with torch.no_grad():
        on = host(ids).logits
        with side_lora_off(host) as count:
            off = host(ids).logits
        again = host(ids).logits
    assert count == len(adapters) and not torch.allclose(on, off) and torch.equal(on, again)
    assert quant.merge_lora(host) == 0 and len(side_adapters(host)) == len(adapters)     # side adapters are never merged
    lora_host = fake_host()                                                         # host-mode LoRA is still merged
    from vsa_embed.integrations.transformers import add_lora
    wrapped = len(add_lora(lora_host, rank=2)) // 2
    assert quant.merge_lora(lora_host) == wrapped > 0 and not any(isinstance(m, LoRALinear) for m in lora_host.modules())
    assert quant.merge_lora(host, include_side=True) == len(adapters) and not side_adapters(host)


def test_side_arm_trains_rescores_and_quantizes_on_cpu(world, tmp_path) -> None:
    from vsa_embed.experiments import e4_quant
    from vsa_embed.integrations.transformers import side_adapters
    root = world["root"]
    config = _config(root, "C0L", {"mode": "none"}, host_mode="train", train={"save_trainable_only": False, "host_lr": 1e-4})
    config["model"]["side_lora"] = {"rank": 2, "extra_rank_layers": 1}
    lm.train(config, tmp_path / "c0l")
    manifest = json.loads((tmp_path / "c0l" / "manifest.json").read_text())
    assert manifest["side_lora"]["rank"] == 2 and manifest["side_lora"]["adapters"] == 8           # 4 projections × 2 blocks
    model = lm.load_final(tmp_path / "c0l" / "final.pt")
    adapters = side_adapters(model.model)
    assert adapters and any(float(m.lora_b.detach().abs().sum()) > 0 for _, m in adapters)   # the side module trained
    optimizer = lm.build_optimizer(model, lm.resolve_config(config)["train"], torch.device("cpu"))
    side_ids = {id(p) for _, m in adapters for p in (m.lora_a, m.lora_b)}
    for group in optimizer.param_groups:
        ids = {id(p) for p in group["params"]}
        if ids & side_ids:
            assert ids <= side_ids | {id(p) for p in model.parameters() if not p.requires_grad} and "lr_scale" not in group
        else:
            assert group.get("lr_scale") == pytest.approx(1e-4 / 3e-3)                  # host parameters at host_lr
    with pytest.raises(ValueError, match="fully trained host"):
        lm.wrap_host({**config, "model": {**config["model"], "host_mode": "lora"}}, fake_host(), None, None)
    # rescoring: `-off` switches the side module off; the simulated schemes quantize the host, never the adapters
    variants = ["ref", "ref-off", "int4-rtn", "int4-rtn-off", "int4-gptq"]
    record = e9_rescore.score_run(tmp_path / "c0l", variants=variants, fillers=world["fillers"], device="cpu", group_size=16,
                                  calibration_windows=4)
    losses = {v: record["variants"][v]["strata"]["all"]["loss"] for v in variants}
    assert losses["ref"] != losses["ref-off"] and losses["int4-rtn"] != losses["int4-rtn-off"] and losses["ref"] != losses["int4-rtn"]
    assert all(np.isfinite(x) for x in losses.values()) and record["side_lora"]["rank"] == 2 and record["ref_check"]["paired"]
    assert record["variants"]["int4-rtn"]["quantization"]["lora_merged"] == 0 and "off_variants" not in record
    assert e9_rescore.profile_variants("auto", "C0L") == list(e9_rescore.PROFILES["side"])
    # variant B quantizes the side module's own tensors (not the host weight inside the adapter); bytes follow
    model = lm.load_final(tmp_path / "c0l" / "final.pt")
    name, adapter = side_adapters(model.model)[0]
    base_before, a_before = adapter.base.weight.detach().clone(), adapter.lora_a.detach().clone()
    sizes = e4_quant.quantize_channel(model, 4, 16)
    assert f"side.{name}.lora_a" in sizes and torch.equal(adapter.base.weight, base_before)
    assert not torch.equal(adapter.lora_a, a_before) and len(torch.unique(adapter.lora_a[0, :16])) <= 16
    parts = e4_quant.model_bytes(model, sizes)
    assert parts["lora"] == sum(v for k, v in sizes.items() if k.startswith("side."))


def test_e9_plan_side_arm(world, tmp_path, monkeypatch) -> None:
    from vsa_embed.integrations.transformers import lora_units
    units = tuple(lora_units(fake_host()))
    monkeypatch.setattr(e9_plan, "host_lora_units", lambda host: units)
    root = tmp_path / "e9"
    paths = e9_plan.write_stage("main", hosts=["SmolLM2-360M"], models=["C0p", "C5", "C0L"], seeds=[1, 2], data_root=world["root"],
                                root=root, track="t5")
    configs = {p.stem: yaml.safe_load(p.read_text()) for p in paths}
    side, c0p = configs["SmolLM2-360M-full-C0L-s1"], configs["SmolLM2-360M-full-C0p-s1"]
    budget = e9_plan.channel_parameters(configs["SmolLM2-360M-full-C5-s1"], world["ontology"])
    assert side["model"]["side_lora"]["matched_to"] == budget and side["channel"] == {"mode": "none"}
    from vsa_embed.integrations.transformers import side_lora_plan
    assert side["model"]["side_lora"]["parameters"] == side_lora_plan(units, budget)["parameters"]
    strip = lambda c: {k: v for k, v in c.items() if k not in {"experiment"}}            # noqa: E731
    assert strip({**side, "model": {k: v for k, v in side["model"].items() if k != "side_lora"}}) == strip(c0p)
    assert budget > e9_plan.channel_parameters(configs["SmolLM2-360M-full-C0p-s1"], world["ontology"]) == 0
    with pytest.raises(ValueError, match="fully trained host"):
        e9_plan.run_config(stage="s", host="SmolLM2-360M", mode="lora", model="C0L", seed=1, data_root=world["root"], tokens=1000,
                           lora_rank=64, host_lr=None, gate_bias=0.0, free_dimension=8, side={"rank": 1})
    assert e9_plan.batch_tag(["C0L"]) == "bb1" and "C0L" in e9_plan.ARM_LIKE and "C0L" not in e9_plan.ARMS
    queue = tmp_path / "jobs"
    side_paths = [p for p in paths if "-C0L-" in p.stem]
    e9_plan.queue_jobs(side_paths, "main", 51, root=root, queue_dir=queue, track="t5", alias_table=Path("t5.json"))
    jobs = {p.stem: json.loads(p.read_text()) for p in queue.glob("*.json")}
    assert set(jobs) == {"main-SmolLM2-360M-full-C0L-s1", "main-SmolLM2-360M-full-C0L-s2", "main-SmolLM2-360M-full-C0L-s1-rescore",
                         "main-SmolLM2-360M-full-C0L-s2-rescore", "main-report-s1-2-bb1-SmolLM2-360M"}
    rescore = jobs["main-SmolLM2-360M-full-C0L-s1-rescore"]["command"]
    assert rescore[rescore.index("--variants") + 1:rescore.index("--output")] == list(e9_rescore.PROFILES["side"])
    report = jobs["main-report-s1-2-bb1-SmolLM2-360M"]
    assert report["command"][2] == "vsa_embed.experiments.e9_bb1" and report["priority"] == 54 and "-report" in report["name"]
    hours = e9_plan.job_estimate_hours("main-SmolLM2-360M-full-C0L-s1", jobs["main-SmolLM2-360M-full-C0L-s1"]["command"])
    assert hours == pytest.approx(e9_plan.MEASURED_TRAINING_MINUTES["SmolLM2-360M"] * e9_plan.SIDE_TRAINING_OVERHEAD
                                  * side["train"]["total_tokens"] / 50_000_000 / 60)
    assert e9_plan.job_estimate_hours(report["name"], report["command"]) == 0.0


def test_bb1_report_on_synthetic_rescores(tmp_path) -> None:
    from vsa_embed.experiments import e9_bb1
    rng = np.random.default_rng(1)
    strata = ["all", "unlinked", "after", "after_heldout", "after_rare_seen", "after_len3plus", "after_heldout_filler"]
    windows = 40
    counts = rng.integers(4, 12, size=(len(strata), windows)).astype(np.int32)
    root, stage = tmp_path / "e9", "toy"
    # bf16 levels; INT4 damage per model: C0′ +0.30, C0L +0.20 (its side module survives), C2 +0.25, C5 +0.10; with the module
    # off: C0L 2.95 (+0.28 at INT4), C5 2.92 (+0.27).
    level = {"P0": 3.2, "C0p": 3.0, "C0L": 2.97, "C2": 2.98, "C5": 2.90}
    damage = {"P0": 0.05, "C0p": 0.30, "C0L": 0.20, "C2": 0.25, "C5": 0.10}
    for model in e9_bb1.MODELS:
        for seed in ([1] if model == "P0" else [1, 2]):
            run = e9_bb1.run_dir(root, stage, "host", model, seed)
            noise = rng.normal(0, 0.001, size=counts.shape) * counts
            ref = level[model] * counts + noise
            arrays = {"ref": ref, "int4-A": ref + damage[model] * counts, "int4-gptq": ref + damage[model] / 2 * counts}
            if model in {"C0L", "C5", "C2"}:
                off = {"C0L": 2.95, "C5": 2.92, "C2": 2.99}[model] * counts + noise
                arrays.update({"ref-off": off, "int4-A-off": off + {"C0L": 0.28, "C5": 0.27, "C2": 0.29}[model] * counts})
            if model == "C0L":
                arrays["int4-B"] = ref + 0.22 * counts
            (run / "rescore").mkdir(parents=True)
            e9_rescore._write_windows(run / "rescore" / "windows.npz", strata, list(range(windows)), counts, arrays)
            record = {"channel": "none" if model in {"P0", "C0p", "C0L"} else "compose",
                      **({"side_lora": {"rank": 2}} if model == "C0L" else {})}
            (run / "rescore" / "rescore.json").write_text(json.dumps(record))
            if model in {"C5", "C2"}:                              # variant B lives in the stage's e4_quant windows
                quant_dir = root / "quant" / stage / "runs" / f"{stage}__{run.name}"
                quant_dir.mkdir(parents=True)
                np.savez_compressed(quant_dir / "windows.npz", strata=np.asarray(strata[:-1]), starts=np.arange(windows), count=counts[:-1],
                                    sum_ref=ref[:-1], **{"sum_int4-B": (ref + (damage[model] + 0.02) * counts)[:-1]})
            if model == "C0L":
                (run / "resolved_config.yaml").write_text(yaml.safe_dump({"model": {"side_lora": {"rank": 2, "parameters": 100,
                                                                                                "matched_to": 101}}}))
    summary = e9_bb1.report(root, [stage], tmp_path / "out", host="host", seeds=[1, 2], resamples=300)
    block = summary["stages"][stage]["comparisons"]
    primary = block["C5 vs C0L"]["int4-A"]["after_heldout"]
    assert primary["did"]["gain_change"]["mean"] == pytest.approx(0.10 - 0.20, abs=2e-3)       # C5's advantage grows vs C0L
    assert primary["cells"]["reference_q"] == pytest.approx(3.17, abs=5e-3) and primary["seeds"] == [1, 2]
    assert block["C5 vs C0′"]["int4-A"]["all"]["did"]["gain_change"]["mean"] == pytest.approx(-0.20, abs=2e-3)
    assert block["C0L vs C0′"]["int4-A"]["all"]["did"]["gain_change"]["mean"] == pytest.approx(-0.10, abs=2e-3)
    assert block["C0L off vs C0′"]["int4-A"]["all"]["did"]["gain_change"]["mean"] == pytest.approx(0.28 - 0.30, abs=2e-3)
    assert block["C0L vs C0L off"]["int4-A"]["all"]["did"]["gain_change"]["mean"] == pytest.approx(0.20 - 0.28, abs=2e-3)
    assert block["C5 vs C0L"]["int4-B"]["all"]["did"]["gain_change"]["mean"] == pytest.approx(0.12 - 0.22, abs=2e-3)
    assert "after_heldout_filler" not in block["C5 vs C0L"]["int4-B"]                      # not in the e4_quant windows
    assert "int4-B" not in block["C5 vs C0′"] or block["C5 vs C0′"]["int4-B"]["all"]["cells"]["reference_q"] == pytest.approx(3.30, abs=5e-3)
    tests = summary["primary"]["tests"]
    assert [t["stratum"] for t in tests] == list(e9_bb1.PRIMARY_STRATA) and all(t["significant"] for t in tests)
    assert tests[0]["reading"].startswith("structure")
    assert summary["stages"][stage]["budget"]["side_lora"]["matched_to"] == 101
    text = (tmp_path / "out" / "report.md").read_text()
    assert "## Primary: C5 vs C0L under int4-A" in text and "### C0L vs C0′" in text
    assert (tmp_path / "out" / "manifest.json").exists() and (tmp_path / "out" / "summary.json").exists()
