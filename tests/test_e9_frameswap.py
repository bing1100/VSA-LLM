"""E9 frame swap (decision 64, `e9_frameswap`) on a toy glossary with a fake host (CPU): target sets and derangements,
`own` replaying the trainer's final evaluation exactly, untouched windows unchanged, `empty` = the channel off on target
spans only, the entry remap = a frame swap (compose) and a source-row swap (C6), refusals and warnings, the report's
pooled bootstrap and Holm math, and the queue-command lines."""

import contextlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.authoring import replace_frames
from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows
from vsa_embed.experiments import e9_frameswap as fs
from vsa_embed.experiments import e9_rescore, e9_rowsource
from vsa_embed.span_channel import AliasTable
from vsa_embed.statistics import holm_adjust, paired_ratio_bootstrap
import vsa_embed.training.lm as lm


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer not available")

COMMON = ["Brightwater Ledger", "Plurb Standard", "Zash Team", "Grosh Console", "Varkt Crew", "Terb Template", "Skolt Handoff"]
RARE = ["Noxel Index", "Quarn Portal"]
UNSEEN = ["Dribbet Sync", "Morlin Gauge"]
HELDOUT = ["Pelto Vault", "Siddle Queue", "Wexa Ring"]
TERMS = COMMON + RARE + UNSEEN + HELDOUT
KINDS = ["system", "policy", "team", "metric"]
ATOMS = [f"type:{k}" for k in KINDS] + ["area:finance", "area:sales"] + [f"term:{t}" for t in COMMON]
A = {name: i for i, name in enumerate(ATOMS)}
SEQ = 32


def _frame(i: int) -> list[tuple[int, int]]:
    edges = [(0, A[f"type:{KINDS[i % 4]}"]), (1, A["area:finance" if i % 2 else "area:sales"])]
    return edges + [(2, A[f"term:{COMMON[(3 * i + 1) % len(COMMON)]}"])] if i % 3 else edges


def _sentence(i: int) -> str:
    term = TERMS[i]
    text = f"{term}: a {'finance' if i % 2 else 'sales'} {KINDS[i % 4]}"
    return text + (f" owned by the {COMMON[(3 * i + 1) % len(COMMON)]}." if i % 3 else ".")


def fake_host(config=None) -> torch.nn.Module:
    torch.manual_seed(1234)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=2))


def _config(root: Path, name: str, channel: dict) -> dict:
    return {"seed": 1, "device": "cpu", "experiment": f"e9-frameswap-fake-lora-{name}-s1",
            "model": {"size": "pretrained", "seq_len": SEQ, "pretrained": "fake-host", "host_mode": "lora", "lora_rank": 2},
            "train": {"micro_batch": 2, "grad_accum": 1, "total_tokens": SEQ * 2 * 6, "lr": 3e-3, "warmup_tokens": 64, "log_every": 2,
                      "save_trainable_only": True, "host_lr": 1e-3},
            "data": {"train": str(root / "train"), "eval": str(root / "eval"), "ontology": str(root / "ontology.pt"), "min_subtokens": 1},
            "eval": {"windows": 10, "batch": 3, "first_tokens": 128, "save_window_losses": True},
            "channel": {"dimension": 16, "key_dimension": 8, "gate_bias": 0.0, **channel}}


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    patch = pytest.MonkeyPatch()
    patch.setattr(lm, "build_model", fake_host)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("frameswap")
    full = AliasTable.from_pairs([(t, i) for i, t in enumerate(TERMS)], holdout=[TERMS.index(t) for t in HELDOUT], include_holdout=True)
    rng = random.Random(0)
    common = [TERMS.index(t) for t in COMMON]
    train = [" ".join(_sentence(rng.choice(common)) for _ in range(6)) for _ in range(100)]
    train += [_sentence(TERMS.index(t)) + " " + _sentence(common[0]) for t in RARE for _ in range(2)]
    evaluation = [" ".join(_sentence(rng.randrange(len(TERMS))) for _ in range(6)) for _ in range(40)]
    build_corpus(train, root / "train", tokenizer_name="gpt2", table=full.without_holdout(), eos_id=50256, max_tokens=60_000,
                 batch_texts=8, workers=1)
    build_corpus(evaluation, root / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=60_000, batch_texts=8, workers=1)
    schedule = full.entry_schedule([_frame(i) for i in range(len(TERMS))])
    entries = len(full.entry_concepts)
    frequency = np.bincount(TokenCorpus.open(root / "train").spans["entry"], minlength=entries)
    ontology = {"entry_count": entries, "atomic_count": len(ATOMS), "relation_count": 3, "offsets": schedule.offsets,
                "relations": schedule.relations, "fillers": schedule.fillers, "heldout_entries": sorted(full.heldout_entries()),
                "train_frequency": frequency.tolist(), "entry_concepts": full.entry_concepts, "alias_table_sha256": full.digest(),
                "concept_names": TERMS, "relation_names": ["is_a", "area", "owned_by"], "atomic_names": ATOMS}
    torch.save(ontology, root / "ontology.pt")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    fillers = root / "fillers.pt"
    e9_rescore.build_filler_table(ontology, full, tokenizer, fillers)
    source = root / "subtoken_mean.pt"
    e9_rowsource.build_table("subtoken_mean", ontology=ontology, output=source, table=full, model=fake_host(), tokenizer=tokenizer)
    runs = {}
    for name, channel in {"C5": {"mode": "compose", "composition": "attentive", "context_window": 4},
                          "C6m": {"mode": "source", "source_kind": "subtoken_mean", "source_table": str(source), "source_hidden": 0},
                          "C2": {"mode": "free"}, "C0p": {"mode": "none"}}.items():
        runs[name] = root / "toy" / f"Fake-lora-{name}-s1"
        lm.train(_config(root, name, channel), runs[name])
    yield {"root": root, "table": full, "ontology": ontology, "fillers": fillers, "runs": runs}
    patch.undo()


def _evaluate(model, config: dict, ontology: dict, starts: list[int], fillers=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    sink: dict = {}
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    lm.evaluate(model, corpus, starts, config, np.asarray(ontology["train_frequency"]), set(ontology["heldout_entries"]),
                torch.device("cpu"), window_sink=sink, fillers=fillers)
    return e9_rescore._sink_arrays(sink)


@contextlib.contextmanager
def _dropped(model, targets) -> None:
    """The channel off on target spans only: those spans never reach the model."""
    original, drop = model.embed, torch.as_tensor(np.asarray(targets), dtype=torch.long)

    def embed(ids, spans):
        keep = ~torch.isin(spans["entry"], drop.to(spans["entry"].device))
        return original(ids, {k: v[keep] for k, v in spans.items()})
    model.embed = embed
    try:
        yield
    finally:
        del model.embed


# -- target sets and swap plans -----------------------------------------------------------------------------------------

def test_target_sets_derangement_and_draws(world) -> None:
    onto = world["ontology"]
    frequency = np.asarray(onto["train_frequency"])
    heldout = fs.target_entries(onto, "heldout")
    unseen, rare = fs.target_entries(onto, "unseen"), fs.target_entries(onto, "rare_seen")
    assert heldout.tolist() == sorted(onto["heldout_entries"]) and len(heldout) == 3
    assert len(unseen) == 2 and (frequency[unseen] == 0).all() and not set(unseen) & set(heldout)
    assert len(rare) == 2 and ((frequency[rare] >= 1) & (frequency[rare] <= 9)).all()
    for size in range(2, 40):
        entries = np.sort(np.random.default_rng(size).choice(1000, size, replace=False))
        mapping = fs.derangement(entries, np.random.default_rng([0, size]))
        assert sorted(mapping) == entries.tolist() and sorted(mapping.values()) == entries.tolist()
        assert all(k != v for k, v in mapping.items())                                # no fixed point, within the set
        assert mapping == fs.derangement(entries, np.random.default_rng([0, size]))   # seeded
    assert fs.derangement(np.array([5]), np.random.default_rng(0)) == {}
    pool = np.arange(100)
    drawn = fs.draw_others(np.array([3, 4, 5]), pool, np.random.default_rng(1))
    assert sorted(drawn) == [3, 4, 5] and len(set(drawn.values())) == 3 and not set(drawn.values()) & {3, 4, 5}
    small = fs.draw_others(np.arange(10), np.arange(12), np.random.default_rng(2))     # pool of 2: with replacement
    assert set(small.values()) <= {10, 11} and len(small) == 10


# -- scoring --------------------------------------------------------------------------------------------------------------

def test_score_own_replays_the_run_and_variants_touch_only_target_windows(world, tmp_path) -> None:
    run = world["runs"]["C5"]
    out = tmp_path / "c5"
    summary = fs.score_run(run, out, entries=["heldout", "unseen", "rare_seen"], fillers=world["fillers"], device="cpu", log=lambda _: None)
    check = summary["own_check"]
    assert check["paired"] and check["counts_equal"] and check["max_abs_window_sum_diff"] == 0.0 and check["windows"] == 10
    found = fs.load_frameswap(out)
    keys = {"own"} | {f"{v}@{k}" for v in fs.SWAPS for k in ("heldout", "unseen", "rare_seen")}
    assert set(found["sums"]) == keys and "after_heldout_filler" in found["strata"]
    for kind in ("heldout", "unseen", "rare_seen"):
        hit = found["touched"][kind]
        assert hit.any() and summary["targets"][kind]["windows_touched"] == int(hit.sum())
        for variant in fs.SWAPS:
            values = found["sums"][f"{variant}@{kind}"]
            assert np.array_equal(values[:, ~hit], found["sums"]["own"][:, ~hit])        # untouched windows: own's sums
    i = found["strata"].index("after_heldout")
    for variant in fs.SWAPS:                                                              # each swap moves the held-out loss
        assert not np.allclose(found["sums"][f"{variant}@heldout"][i], found["sums"]["own"][i])
    assert summary["differences"]["empty@heldout"]["after_heldout"]["delta"] == pytest.approx(
        (found["sums"]["empty@heldout"][i].sum() - found["sums"]["own"][i].sum()) / found["count"][i].sum())
    # Every window under every variant (no subsetting) agrees: the premise of copying own's sums holds.
    every = fs.score_run(run, tmp_path / "all", entries=["heldout"], fillers=world["fillers"], device="cpu", all_windows=True,
                         log=lambda _: None)
    assert every["all_windows"]
    again = fs.load_frameswap(tmp_path / "all")
    for variant in fs.SWAPS:
        np.testing.assert_allclose(again["sums"][f"{variant}@heldout"], found["sums"][f"{variant}@heldout"], rtol=0, atol=1e-4)
    # No entry id or name in the outputs: target sets are counts.
    text = (out / "summary.json").read_text()
    assert set(summary["targets"]["heldout"]) == {"entries", "dropped_empty_frames", "pool", "stratum", "windows_touched",
                                                  "stratum_tokens", "skipped", "other_any_with_replacement"}
    assert not any(name in text for name in TERMS)
    with pytest.raises(FileExistsError):
        fs.score_run(run, out, entries=["heldout"], fillers=world["fillers"], device="cpu", log=lambda _: None)
    resumed = fs.score_run(run, out, entries=["heldout", "unseen", "rare_seen"], fillers=world["fillers"], device="cpu", resume=True,
                           log=lambda _: None)
    assert resumed["seconds"] == summary["seconds"] and (out / "manifest.json").exists() and (out / "resolved_config.yaml").exists()


def test_empty_is_the_channel_off_on_target_spans_only(world, tmp_path) -> None:
    run = world["runs"]["C5"]
    fs.score_run(run, tmp_path / "e", entries=["heldout"], variants=["empty"], fillers=None, device="cpu", log=lambda _: None)
    found = fs.load_frameswap(tmp_path / "e")
    model = lm.load_final(run / "final.pt")
    config = lm.resolve_config(torch.load(run / "final.pt", weights_only=False)["config"])
    starts = found["starts"].tolist()
    with _dropped(model, fs.target_entries(world["ontology"], "heldout")):
        strata, sums, counts = _evaluate(model, config, world["ontology"], starts)
    assert strata == found["strata"] and np.array_equal(counts, found["count"])
    np.testing.assert_allclose(sums, found["sums"]["empty@heldout"], rtol=0, atol=1e-5)
    plain = _evaluate(model, config, world["ontology"], starts)[1]
    np.testing.assert_allclose(plain, found["sums"]["own"], rtol=0, atol=1e-5)


def test_remap_is_a_frame_swap_and_a_source_row_swap(world) -> None:
    onto = world["ontology"]
    targets = fs.target_entries(onto, "heldout")
    mapping = fs.derangement(targets, np.random.default_rng(3))
    starts = None
    for name in ("C5", "C6m"):
        run = world["runs"][name]
        model = lm.load_final(run / "final.pt")
        config = lm.resolve_config(torch.load(run / "final.pt", weights_only=False)["config"])
        starts = starts or eval_windows(TokenCorpus.open(Path(config["data"]["eval"])), count=10, length=SEQ)
        with fs.remapped(model, mapping):
            swapped = _evaluate(model, config, onto, starts)[1]
        channel = model.channel
        if name == "C5":                         # the other entry's frame written into the target's slot of the schedule
            schedule = channel.composer.schedule
            frame = lambda e: list(zip(schedule.relations[schedule.offsets[e]:schedule.offsets[e + 1]].tolist(),
                                       schedule.fillers[schedule.offsets[e]:schedule.offsets[e + 1]].tolist()))
            channel.composer.set_schedule(replace_frames(schedule, {e: frame(o) for e, o in mapping.items()}))
            direct = _evaluate(model, config, onto, starts)[1]
            channel.composer.set_schedule(schedule)
        else:                                    # the other entry's frozen source vector in the target's row
            original = channel.source_rows.clone()
            channel.source_rows[torch.tensor(list(mapping))] = original[torch.tensor(list(mapping.values()))]
            direct = _evaluate(model, config, onto, starts)[1]
            channel.source_rows.copy_(original)
        np.testing.assert_allclose(swapped, direct, rtol=0, atol=1e-5)
        assert not np.allclose(swapped, _evaluate(model, config, onto, starts)[1])


def test_source_arm_scores_c2_warns_and_runs_without_rows_are_refused(world, tmp_path) -> None:
    c6 = fs.score_run(world["runs"]["C6m"], tmp_path / "c6", entries=["heldout"], fillers=None, device="cpu", log=lambda _: None)
    assert c6["channel"]["mode"] == "source" and c6["targets"]["heldout"]["entries"] == 3 and not c6["warnings"]
    messages = []
    c2 = fs.score_run(world["runs"]["C2"], tmp_path / "c2", entries=["heldout"], fillers=None, device="cpu", log=messages.append)
    assert c2["warnings"] and any("warning" in m for m in messages)
    found = fs.load_frameswap(tmp_path / "c2")
    np.testing.assert_allclose(found["sums"]["other@heldout"], found["sums"]["own"], rtol=0, atol=1e-5)   # one fallback row
    with pytest.raises(ValueError, match="per-entry rows"):
        fs.score_run(world["runs"]["C0p"], tmp_path / "c0", fillers=None, device="cpu")
    with pytest.raises(ValueError, match="unknown"):
        fs.score_run(world["runs"]["C5"], tmp_path / "x", variants=["own", "swap"], device="cpu")
    smoke = fs.score_run(world["runs"]["C5"], tmp_path / "smoke", entries=["heldout"], limit_windows=4, fillers=None, device="cpu",
                         log=lambda _: None)
    assert smoke["smoke"] and smoke["windows"] == 4 and smoke["own_check"]["paired"] and smoke["own_check"]["max_abs_window_sum_diff"] == 0.0


# -- report ----------------------------------------------------------------------------------------------------------------

def _fake_output(folder: Path, seed: int, rng: np.random.Generator, shift: float) -> dict:
    strata = ["all", "after", "after_heldout", "unlinked"]
    windows = 50
    count = rng.integers(0, 30, size=(len(strata), windows)).astype(np.int32)
    own = count * rng.uniform(1.5, 2.5, size=count.shape)
    sums = {"own": own}
    for k, variant in enumerate(fs.SWAPS):
        sums[f"{variant}@heldout"] = own + count * (shift * (k + 1) + rng.normal(0, 0.05, size=count.shape))
    folder.mkdir(parents=True)
    np.savez_compressed(folder / "windows.npz", strata=np.asarray(strata), starts=np.arange(windows) * 100, count=count,
                        **{f"sum_{k}": v for k, v in sums.items()}, touched_heldout=np.ones(windows, dtype=bool))
    summary = {"kind": fs.KIND, "id": f"toy__Fake-full-C5-s{seed}", "stage": "toy", "stem": f"Fake-full-C5-s{seed}", "host": "Fake",
               "host_mode": "full", "model": "C5", "run_seed": seed, "entries": ["heldout"], "smoke": False, "warnings": [],
               "targets": {"heldout": {"entries": 7, "dropped_empty_frames": 0, "windows_touched": windows}}, "own_check": {"available": False}}
    (folder / "summary.json").write_text(json.dumps(summary))
    return {"count": count, "sums": sums, "strata": strata}


def test_report_pools_seeds_by_window_with_holm_over_variants(tmp_path) -> None:
    rng = np.random.default_rng(0)
    made = {s: _fake_output(tmp_path / "in" / f"Fake-full-C5-s{s}", s, rng, 0.02) for s in (1, 2, 3)}
    summary = fs.write_report([tmp_path / "in"], tmp_path / "in" / "report", resamples=400)
    group = summary["groups"]["toy · Fake · full · C5"]
    assert group["seeds"] == [1, 2, 3]
    i = 2                                                                            # after_heldout
    rows = group["sets"]["heldout"]["strata"]["after_heldout"]
    d = sum(made[s]["sums"]["other@heldout"][i] - made[s]["sums"]["own"][i] for s in made)
    n = sum(made[s]["count"][i].astype(float) for s in made)
    b = sum(made[s]["sums"]["own"][i] for s in made)
    boot = paired_ratio_bootstrap(d, n, b, resamples=400, seed=0)
    assert rows["other"]["delta"] == pytest.approx(d.sum() / n.sum()) and rows["other"]["relative"] == pytest.approx(d.sum() / b.sum())
    assert rows["other"]["ci_low"] == pytest.approx(boot["ci_low"]) and rows["other"]["windows"] == boot["nonempty_clusters"]
    assert [rows[v]["holm_p"] for v in fs.SWAPS] == pytest.approx(holm_adjust([rows[v]["p_value"] for v in fs.SWAPS]))
    assert group["primary"]["ontology_specific"] and group["primary"]["delta"] > 0
    seed2 = group["sets"]["heldout"]["per_seed"][2]["other"]
    assert seed2["delta"] == pytest.approx((made[2]["sums"]["other@heldout"][i] - made[2]["sums"]["own"][i]).sum() / made[2]["count"][i].sum())
    text = (tmp_path / "in" / "report" / "report.md").read_text()
    assert "**Primary**" in text and "ontology-specific" in text and "SMOKE" not in text
    assert set(group["sets"]["heldout"]["strata"]) == {"after_heldout", "after", "unlinked", "all"}
    with pytest.raises(FileExistsError):
        fs.write_report([tmp_path / "in"], tmp_path / "in" / "report", resamples=50)
    fs.write_report([tmp_path / "in"], tmp_path / "in" / "report", resamples=50, overwrite=True)       # the report folder is skipped
    null = tmp_path / "null"
    for s in (1, 2):
        _fake_output(null / f"Fake-full-C5-s{s}", s, rng, 0.0)
    flat = fs.write_report([null], null / "report", resamples=400)["groups"]["toy · Fake · full · C5"]["primary"]
    assert not flat["ontology_specific"]


def test_queue_lines(tmp_path) -> None:
    root = tmp_path / "e9"
    for stage, stem in (("t4", "SmolLM2-360M-full-C5-s1"), ("t5", "SmolLM2-135M-full-C5-s2")):
        (root / "runs" / stage / stem).mkdir(parents=True)
        (root / "runs" / stage / stem / "final.pt").write_bytes(b"")
    for stage in ("t7", "t1c-rood"):
        (root / "configs" / stage).mkdir(parents=True)
        (root / "configs" / stage / "SmolLM2-360M-full-C5sh-s2.yaml").write_text("{}")
    lines = fs.queue_lines(fs.ROOT, check_root=root)
    jobs = [line for line in lines if line.startswith("PYTHONPATH")]
    names = [line.split("--name ")[1].split()[0] for line in jobs]
    assert "t4-SmolLM2-360M-full-C5-s1-frameswap" in names and "t5-SmolLM2-135M-full-C5-s2-frameswap" in names
    assert "t7-SmolLM2-360M-full-C5sh-s2-frameswap" in names and "t1c-rood-SmolLM2-360M-full-C5sh-s2-frameswap" in names
    assert {n for n in names if n.endswith("-report")} == {f"{s}-frameswap-report" for s in ("t4", "t5", "t1", "wordnet", "t7", "t7rood",
                                                                                             "t8", "t1c-rood")}
    rood = [line for line in jobs if "t1c-rood" in line]
    assert all("--priority 54.4998" in line for line in rood)
    for line in rood:
        outputs = [line.split(flag)[1].split()[0] for flag in (" --output ", " --inputs ") if flag in line]
        assert outputs and all(o.startswith("experiments/e9-retrofit/frameswap/t1c-rood") for o in outputs)
        assert "-report" in line or "--fillers none" in line
    assert any("--priority 51.5 " in line for line in jobs) and any("--priority 52 " in line for line in jobs)
    assert any(line.startswith("# skipped t4/SmolLM2-360M-full-C5-s2") for line in lines)
    assert all("GPU-h" in line for line in jobs) and not any(" add " in line and "--queue" in line for line in jobs)
