"""T1c-F — ICD frequency bias with structured code vectors (`vsa_embed.icd_coding`, `experiments.t1c_icd_frequency`).

Every fixture here is synthetic: invented codes ("Q…", "W…"), invented concept ids and random states — never MIMIC,
SNOMED CT or ICD titles. The ICD-9-CM hierarchy test parses a few textbook code strings of the public classification
(format examples, not drawn from MIMIC; no titles).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from vsa_embed import icd_coding as ic
from vsa_embed.experiments import t1c_icd_frequency as tf


# -- frequency bins and the holdout -------------------------------------------------------------------------------

def test_hrrbert_bins_cover_minus_14_to_0_in_width_2() -> None:
    assert len(ic.HRRBERT_EDGES) == 8 and np.allclose(np.diff(ic.HRRBERT_EDGES), 2)
    logf = ic.log_frequency([0, 1, 3, 30, 3000, 400_000], 1_000_000)
    assert np.isneginf(logf[0])
    bins = ic.frequency_bins(logf)
    assert bins[0] == -1                                   # never seen
    assert bins[1] == 0                                    # ln(1e-6) = −13.8 → [−14, −12)
    assert bins[2] == 0                                    # ln(3e-6) = −12.7 → [−14, −12)
    assert bins[3:].tolist() == [1, 4, 6]                  # −10.4, −5.8, −0.9
    assert ic.bin_name(0) == "[-14,-12)" and ic.bin_name(6) == "[-2,0)"
    assert ic.frequency_bins([-20.0, -0.1, 0.0, 5.0]).tolist() == [0, 6, 6, 6]


def test_bins_are_half_open_on_the_upper_edge() -> None:
    assert ic.frequency_bins([-12.0, -12.0 - 1e-9, -10.0]).tolist() == [1, 0, 2]


def _counts(n_codes: int = 400, seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    return {f"Q{i:04d}": int(c) for i, c in enumerate(np.exp(rng.uniform(0, 9, n_codes)).astype(int))}


def test_code_holdout_is_hashed_stratified_and_pinned() -> None:
    counts = _counts()
    total = sum(counts.values())
    codes = sorted(counts)
    first = ic.choose_code_holdout(codes, counts, total, eligible_bins=[-12, -10, -8], fraction=0.2, min_count=5, salt="s")
    again = ic.choose_code_holdout(list(reversed(codes)), counts, total, eligible_bins=[-12, -10, -8], fraction=0.2,
                                   min_count=5, salt="s")
    assert first["codes"] == again["codes"] and first["sha256"] == again["sha256"]
    assert first["sha256"] == ic.holdout_digest(first["codes"])
    bins = ic.frequency_bins(ic.log_frequency([counts[c] for c in first["codes"]], total))
    assert set(bins.tolist()) <= {1, 2, 3}                 # only the eligible bins
    assert all(counts[c] >= 5 for c in first["codes"])
    for name, row in first["per_bin"].items():
        assert row["chosen"] == round(0.2 * row["eligible"])
        assert row["positives_removed"] == sum(counts[c] for c in first["codes"]
                                               if ic.bin_name(int(ic.frequency_bins(ic.log_frequency([counts[c]], total))[0])) == name)
    other = ic.choose_code_holdout(codes, counts, total, eligible_bins=[-12, -10, -8], fraction=0.2, min_count=5, salt="t")
    assert other["codes"] != first["codes"]                # the salt decides, nothing else


def test_label_space_orders_trained_heldout_natural_and_keeps_heldout_out_of_training() -> None:
    rng = np.random.default_rng(1)
    codes = [f"W{i:03d}" for i in range(120)]
    mapping = {c: ("1to1", [str(10_000 + i)]) for i, c in enumerate(codes[:100])}      # 20 codes unframed
    weights = np.exp(rng.uniform(0, 5, len(codes)))
    weights /= weights.sum()
    by_admission, split = {}, {}
    for a in range(3000):
        chosen = set(rng.choice(codes, size=4, replace=False, p=weights).tolist())
        by_admission[a] = chosen
        split[a] = "eval" if a % 10 == 0 else ("dev" if a % 10 == 1 else "train")
    by_admission[99_999] = {"W099"} ; split[99_999] = "eval"                             # possibly unseen in training
    space = tf.label_space(by_admission, split, mapping, edges=ic.HRRBERT_EDGES,
                           holdout_config={"bins": [-8, -6, -4], "fraction": 0.3, "min_count": 5, "salt": "x"})
    order, held = space["order"], set(space["holdout"]["codes"])
    trained = order[:space["n_trained"]]
    assert set(order) <= set(mapping) and len(order) == len(set(order))
    assert not held & set(trained)
    assert order[space["n_trained"]:space["n_trained"] + space["n_heldout"]] == sorted(held)
    assert all(space["counts"]["train"][c] >= 1 for c in trained)
    assert all(space["counts"]["train"][c] == 0 for c in order[space["n_trained"] + space["n_heldout"]:])
    assert space["total_train"] == sum(space["counts"]["train"].values())


# -- ICD-9-CM hierarchy, concept reduction, frames ----------------------------------------------------------------

def test_icd9_ancestors_for_numeric_v_and_e_codes() -> None:
    assert ic.icd9_ancestors("4280") == ["icd9:chapter:07", "icd9:428"]
    assert ic.icd9_ancestors("25000") == ["icd9:chapter:03", "icd9:250", "icd9:2500"]
    assert ic.icd9_ancestors("0389") == ["icd9:chapter:01", "icd9:038"]
    assert ic.icd9_ancestors("V5861") == ["icd9:chapter:V", "icd9:V58", "icd9:V586"]
    assert ic.icd9_ancestors("E8798") == ["icd9:chapter:E", "icd9:E879"]
    assert ic.icd9_ancestors("999") == ["icd9:chapter:17"]
    assert ic.icd9_chapter("X12") == "icd9:chapter:unknown"


def test_maximal_concepts_keep_the_most_general_mapped_concepts() -> None:
    parents = {"2": ["1"], "3": ["2"], "4": ["1"], "6": ["5"], "7": ["5"], "8": []}
    ancestors = tf.ancestor_function(parents)
    assert ancestors("3") == {"1", "2"}
    # {2, 3, 4, 6, 7, 8}: 3 is under 2; the maximal ones are 2, 4, 6, 7, 8; 2 subsumes 3 → first.
    assert ic.maximal_concepts({"2", "3", "4", "6", "7", "8"}, ancestors, cap=10) == ["2", "4", "6", "7", "8"]
    assert ic.maximal_concepts({"2", "3", "4", "6", "7", "8"}, ancestors, cap=2) == ["2", "4"]
    assert ic.union_frame([[(0, 1), (1, 2)], [(1, 2), (2, 3)]]) == [(0, 1), (1, 2), (2, 3)]


# -- label sources and the head -----------------------------------------------------------------------------------

def _frames(n: int, atoms: int = 30, relations: int = 4, seed: int = 0) -> list[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    return [[(int(rng.integers(relations)), int(rng.integers(atoms))) for _ in range(int(rng.integers(2, 5)))] for _ in range(n)]


def test_composed_source_composes_untrained_codes_by_the_trained_rule() -> None:
    frames = _frames(12)
    frames[11] = list(frames[3])                            # an untrained code with a trained code's frame
    source = ic.ComposedSource(frames, atomic_count=30, relation_count=4, dimension=32, key_dimension=4, seed=0)
    vectors = source(torch.arange(12))
    assert vectors.shape == (12, 32) and torch.allclose(vectors.norm(dim=1), torch.ones(12), atol=1e-5)
    assert torch.allclose(vectors[11], vectors[3], atol=1e-6)
    # A frozen, already-trained composer receives the frames by add_concepts (zero-shot insertion).
    from vsa_embed.compose import FrameComposer, FrameSchedule
    trained = FrameComposer(FrameSchedule.from_frames(_frames(50, seed=3)), 30, 4, 32, mode="attentive", key_dimension=4)
    frozen = ic.ComposedSource(frames, atomic_count=30, relation_count=4, composer=trained, trainable=False)
    assert frozen.offset == 50 and not any(p.requires_grad for p in frozen.parameters())
    expected = trained.compose(torch.arange(50, 62))
    assert torch.allclose(frozen(torch.arange(12)), expected, atol=1e-6)


def test_gram_source_attends_over_the_code_and_its_ancestors() -> None:
    paths = [[0, 3, 4], [1, 3, 4], [2, 5]]
    source = ic.GramSource(paths, node_count=6, dimension=8, attention_dim=4, seed=0)
    out = source(torch.arange(3))
    assert out.shape == (3, 8)
    # Code 1's vector is a convex combination of nodes 1, 3, 4 (padding never attended).
    with torch.no_grad():
        e = source.nodes[[1, 3, 4]]
        coef = torch.linalg.lstsq(e.T, out[1:2].T).solution.squeeze()
    assert torch.allclose(coef.sum(), torch.tensor(1.0), atol=1e-4) and bool((coef > -1e-4).all())


def test_head_ignores_padding_segments_and_scores_every_label() -> None:
    torch.manual_seed(0)
    head = ic.LabelAttentionHead(ic.FreeSource(7, 16), input_dim=12, attention_dim=8, hidden=16, prior=0.1)
    states = torch.randn(2, 5, 12)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    logits = head(states, mask, torch.arange(7))
    padded = states.clone()
    padded[1, 3:] = 100.0                                    # garbage in the padding
    assert logits.shape == (2, 7)
    assert torch.allclose(head(padded, mask, torch.arange(7)), logits, atol=1e-5)
    assert torch.allclose(head(states[1:, :3], mask[1:, :3], torch.arange(7)), logits[1:], atol=1e-5)


def _synthetic_store(n_adm: int = 240, n_labels: int = 10, width: int = 16, seed: int = 0):
    """Admissions whose segments carry label-specific directions: a learnable coding task."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_labels, width)).astype(np.float32)
    labels, rows, offsets = [], [], [0]
    for a in range(n_adm):
        k = int(rng.integers(1, 3))
        labs = np.sort(rng.choice(n_labels, size=k, replace=False, p=np.linspace(2, 1, n_labels) / np.linspace(2, 1, n_labels).sum()))
        n = int(rng.integers(3, 9))
        seg = rng.normal(scale=0.5, size=(n, width)).astype(np.float32)
        for j, l in enumerate(labs):
            seg[j % n] += 2.0 * directions[l]
        rows.append(seg); offsets.append(offsets[-1] + n); labels.append(labs.astype(np.int64))
    store = ic.SegmentStore(np.concatenate(rows).astype(np.float16), np.asarray(offsets))
    return store, labels


def test_train_heads_learns_pairs_batches_and_leaves_untrained_free_rows_untouched() -> None:
    store, labels = _synthetic_store()
    n_labels, trained = 10, np.arange(8)                     # labels 8 and 9 never trained
    idx = np.arange(len(labels))
    heads = {}
    for name in ("a", "b"):
        torch.manual_seed(0)
        heads[name] = ic.LabelAttentionHead(ic.FreeSource(n_labels, 16, seed=1), store.width, attention_dim=16, hidden=32,
                                            prior=0.15)
    untrained_before = heads["a"].source.weight[8:].detach().clone()
    history = ic.train_heads(heads, store, labels, train_admissions=idx[:200], dev_admissions=idx[200:], train_labels=trained,
                             label_count=n_labels, epochs=6, patience=10, batch=16, lr=3e-3, seed=0)
    assert history["a"]["history"][-1]["dev_loss"] < history["a"]["history"][0]["dev_loss"]
    # identical heads on identical batches stay identical (the batch order is a function of the seed only)
    for (k, va), vb in zip(heads["a"].state_dict().items(), heads["b"].state_dict().values()):
        assert torch.allclose(va, vb), k
    assert torch.equal(heads["a"].source.weight[8:].detach(), untrained_before)
    scores = ic.score_admissions(heads["a"], store, idx[200:], np.arange(n_labels))
    positives = np.zeros(scores.shape, dtype=bool)
    for r, a in enumerate(idx[200:]):
        positives[r, labels[a]] = True
    auc = ic.auc_columns(scores, positives)
    assert np.nanmean(auc[:8]) > 0.8


def test_composed_head_gives_never_trained_codes_signal_through_shared_atomics() -> None:
    store, labels = _synthetic_store(n_labels=10, seed=2)
    # Code 9 is never trained but shares its frame with trained code 0 → after training it scores like code 0.
    frames = _frames(10, atoms=12, relations=2, seed=5)
    frames[9] = list(frames[0])
    torch.manual_seed(0)
    head = ic.LabelAttentionHead(ic.ComposedSource(frames, atomic_count=12, relation_count=2, dimension=16, key_dimension=4),
                                 store.width, attention_dim=16, hidden=32, prior=0.15)
    idx = np.arange(len(labels))
    ic.train_heads({"c": head}, store, labels, train_admissions=idx[:200], dev_admissions=idx[200:],
                   train_labels=np.arange(9), label_count=10, epochs=3, patience=5, batch=16, lr=3e-3, seed=0)
    scores = ic.score_admissions(head, store, idx, np.arange(10))
    assert np.allclose(scores[:, 9], scores[:, 0], atol=1e-4)


# -- metrics and statistics ---------------------------------------------------------------------------------------

def test_weighted_auc_matches_midrank_auc_and_explicit_resampling() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(60, 7)).astype(np.float32)
    scores[:, 3] = np.round(scores[:, 3])                   # many ties
    positives = rng.random((60, 7)) < 0.3
    positives[:, 6] = False                                  # no positive → NaN
    expected = ic.auc_columns(scores, positives)
    weighted = ic.WeightedAuc(scores, positives)
    assert np.allclose(weighted().numpy(), expected, equal_nan=True, atol=1e-6)
    counts = rng.poisson(1.0, size=60)
    rows = np.repeat(np.arange(60), counts)
    resampled = ic.auc_columns(scores[rows], positives[rows])
    assert np.allclose(weighted(counts.astype(np.float32)).numpy(), resampled, equal_nan=True, atol=1e-6)


def test_ranks_and_topk() -> None:
    scores = np.array([[0.1, 0.9, 0.5, 0.3], [0.8, 0.2, 0.7, 0.9]], dtype=np.float32)
    rows, cols = np.array([0, 0, 1, 1]), np.array([1, 3, 0, 1])
    ranks = ic.positive_ranks(scores, rows, cols)
    assert ranks.tolist() == [1, 3, 2, 4]
    torch_ranks = ic.ranks_in_rows(torch.as_tensor(scores), torch.as_tensor(rows), torch.as_tensor(cols))
    assert torch_ranks.tolist() == ranks.tolist()
    sub = ic.ranks_in_rows(torch.as_tensor(scores), torch.as_tensor(rows), torch.as_tensor(cols), torch.tensor([1, 3]))
    assert sub.tolist() == [1, 2, 2, 2]
    rate, pairs = ic.topk_by_code(ranks, cols, 4, k=2)
    assert pairs.tolist() == [1, 2, 0, 1] and rate[1] == 0.5 and rate[0] == 1.0 and math.isnan(rate[2])


def test_slopes_holm_bootstrap_and_dunnett() -> None:
    x = np.array([1.0, 2, 3, 4, np.nan])
    assert ic.ols_slope(x, 2 * x + 1) == pytest.approx(2.0)
    xs = torch.tensor([[1.0, 2, 3, 4], [1, 2, 3, float("nan")]])
    ys = torch.tensor([[2.0, 4, 6, 8], [3, 2, 1, 0]])
    assert torch.allclose(ic.ols_slopes(xs, ys), torch.tensor([2.0, -1.0]))
    adjusted = ic.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["c"] == pytest.approx(0.06) and adjusted["b"] == pytest.approx(0.06)
    assert math.isnan(adjusted["d"])
    assert ic.bootstrap_pvalue(np.full(100, 0.5)) == pytest.approx(0.01)
    assert ic.bootstrap_pvalue(np.linspace(-1, 1, 101)) == pytest.approx(1.0)
    p = ic.dunnett([0.5, 0.52, 0.51], {"x": [0.9, 0.91, 0.92], "y": [0.5, 0.51, 0.52]})
    assert p["x"] < 0.01 < p["y"]
    assert math.isnan(ic.dunnett([0.5], {"x": [0.9, 0.9]})["x"])


def test_frequency_probes_and_tsne() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(-14, -2, 400)
    informative = np.column_stack([y + rng.normal(scale=0.3, size=400), rng.normal(size=(400, 15))])
    noise = rng.normal(size=(400, 16))
    assert ic.cv_ridge(informative, y)["r2"] > 0.9
    assert ic.cv_ridge(noise, y)["r2"] < 0.1
    theta = (y + 14) / 12 * np.pi / 2                         # frequency written in the direction (cosine kNN)
    directional = np.column_stack([np.cos(theta), np.sin(theta), 0.01 * rng.normal(size=(400, 6))])
    assert ic.neighbour_agreement(directional, y, k=5) > 0.9
    assert abs(ic.neighbour_agreement(noise, y, k=5)) < 0.3
    clusters = np.concatenate([rng.normal(size=(60, 10)), rng.normal(size=(60, 10)) + 8])
    coords = ic.tsne(clusters, perplexity=15, iterations=300, seed=0)
    group = np.repeat([0, 1], 60)
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    assert (group[dist.argmin(1)] == group).mean() > 0.95


# -- tokens, chunks, segments ---------------------------------------------------------------------------------------

def test_chunk_plan_and_segment_pooling_agree() -> None:
    lengths = np.array([0, 5, 32, 33, 70])
    counts, offsets = tf.chunk_plan(lengths, chunk=40, segment=8)
    # 5 → 1; 32 → 4; 33 → 5; 70 = 40 + 30 → 5 + 4
    assert counts.tolist() == [0, 1, 4, 5, 9] and offsets[-1] == 19
    hidden = torch.arange(2 * 12 * 2, dtype=torch.float32).view(2, 12, 2)
    pooled = tf.pool_segments(hidden, [12, 5], segment=4)
    assert [p.shape[0] for p in pooled] == [3, 2]
    assert torch.allclose(pooled[1][1], hidden[1, 4])        # a one-token last segment is that token


class _StubEncoder:
    """A 'causal host' whose state at t is the running mean of token ids up to t (padding-invariant to the right)."""

    width = 3
    channel = False

    def hidden(self, ids: torch.Tensor, spans):
        x = ids.float()
        run = x.cumsum(1) / torch.arange(1, x.shape[1] + 1)
        return torch.stack([run, x, torch.ones_like(x)], -1)


def test_encode_tokens_writes_chunked_segment_means_in_place() -> None:
    lengths = [7, 0, 13]
    ids = np.concatenate([np.arange(1, n + 1) for n in lengths]).astype(np.uint16)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    tokens = {"ids": ids, "offsets": offsets, "spans": {}}
    counts, seg_offsets = tf.chunk_plan(np.array(lengths), chunk=5, segment=2)
    out = np.zeros((int(seg_offsets[-1]), 3), dtype=np.float16)
    tf.encode_tokens(_StubEncoder(), tokens, chunk=5, segment=2, batch_chunks=2, out=out, seg_offsets=seg_offsets)
    # text 2, chunk 2 (tokens 6..10 → ids 6..10, a fresh window): its first segment is ids 6, 7 → running means 6, 6.5
    first = seg_offsets[2] + 5 // 2 + 1
    assert out[first, 0] == pytest.approx(6.25) and out[first, 1] == pytest.approx(6.5)
    means = tf.encode_tokens(_StubEncoder(), tokens, chunk=5, segment=2, batch_chunks=3, out=None, seg_offsets=None,
                             pooled="mean")
    assert means.shape == (3, 3) and means[1].tolist() == [0, 0, 0] and means[0, 1] == pytest.approx(4.0)


def test_tokenize_texts_truncates_and_keeps_spans_inside_chunks() -> None:
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M", local_files_only=True)
    except Exception:
        pytest.skip("SmolLM2 tokenizer not cached")
    from vsa_embed.span_channel import AliasTable, CausalLinker
    table = AliasTable.from_pairs([("zorbic plinth syndrome", 0), ("quorbacter", 1)])
    linker = CausalLinker(table, min_subtokens=2)
    texts = ["the zorbic plinth syndrome was seen. " * 30, "quorbacter quorbacter and nothing else", ""]
    out = tf.tokenize_texts(texts, tokenizer_name="HuggingFaceTB/SmolLM2-360M", linker=linker, max_tokens=64, chunk=16,
                            workers=1)
    lengths = np.diff(out["offsets"])
    assert lengths.tolist()[0] == 64 and lengths[2] == 0 and out["full_lengths"][0] > 64
    spans = out["spans"]
    assert spans["entry"].size > 0
    assert np.all(spans["start"] // 16 == spans["inject"] // 16) and np.all(spans["inject"] < 64)


# -- TransE graph, plan, and the licence guard of an end-to-end synthetic run -------------------------------------

def test_kge_graph_unifies_atomics_with_their_concepts() -> None:
    labels = {"concept_frames": {"111111": [(0, 0), (1, 1)], "222222": [(0, 2)]}, "relation_count": 2}
    graph, entity = tf.kge_graph_for_labels(labels, ["top:x", "sct:222222", "sct:999999"])
    assert entity == {"111111": 0, "222222": 1}
    assert graph["tails"].tolist() == [2, 1, 3] and graph["entities"] == 4 and graph["unified_atomics"] == 1


def test_plan_places_jobs_after_t1c_seed_1_and_queues_nothing() -> None:
    text = tf.plan_commands()
    priorities = [int(p) for p in __import__("re").findall(r"--priority (\d+)", text)]
    assert priorities and min(priorities) >= 55
    assert "t1cf-encode-P0-360M" in text and "--c5-run" in text and "GPU-h" in text


def _synthetic_experiment(tmp_path: Path) -> dict:
    """A tiny data root in the layout of the prepare / encode stages, with invented codes."""
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    root.mkdir()
    n_trained, n_held, n_nat = 14, 4, 2
    n = n_trained + n_held + n_nat
    codes = [f"Q{9000 + i}X" for i in range(n)]
    frames = _frames(n, atoms=20, relations=3, seed=1)
    count_train = np.concatenate([np.sort(rng.integers(1, 400, n_trained))[::-1], rng.integers(10, 80, n_held), np.zeros(n_nat, int)])
    count_train[n_trained - 4:n_trained] = [3, 2, 2, 1]                       # a few rare trained codes
    total = int(count_train.sum())
    logf = ic.log_frequency(count_train, total)
    store, adm_labels = _synthetic_store(n_adm=300, n_labels=n, width=16, seed=4)
    split = np.array(["eval" if a % 5 == 0 else ("dev" if a % 5 == 1 else "train") for a in range(300)])
    labels = {"codes": codes, "frames": frames, "n_trained": n_trained, "logf": logf, "bin": ic.frequency_bins(logf),
              "edges": list(ic.HRRBERT_EDGES), "heldout": np.array([False] * n_trained + [True] * n_held + [False] * n_nat),
              "trained": np.array([True] * n_trained + [False] * (n_held + n_nat)), "atomic_count": 20, "relation_count": 3,
              "gram_paths": [[i, n + (i % 3)] for i in range(n)], "gram_nodes": [f"n{i}" for i in range(n + 3)],
              "members": [[str(100000 + i)] for i in range(n)]}
    torch.save(labels, root / "labels.pt")
    torch.save({"split": split, "labels": adm_labels}, root / "admissions.pt")
    states = root / "states" / "stub"
    states.mkdir(parents=True)
    segments = np.memmap(states / "segments.f16", dtype=np.float16, mode="w+", shape=store.segments.shape)
    segments[:] = store.segments
    segments.flush()
    np.save(states / "seg_offsets.npy", store.offsets)
    np.save(states / "titles.npy", rng.normal(size=(n, 16)).astype(np.float32))
    (states / "meta.json").write_text(json.dumps({"width": 16, "complete": True}))
    (root / "kge").mkdir()
    torch.save({"vectors": torch.randn(n, 8)}, root / "kge" / "transe-d8.pt")
    return {"experiment": "t", "version": "test", "seed": 3,
            "paths": {"data_root": str(root), "runs": str(tmp_path / "runs"), "ontology_pt": str(root / "none.pt")},
            "head": {"source_dim": 16, "attention_dim": 16, "label_hidden": 32, "batch": 16, "lr": 3e-3, "weight_decay": 0.01,
                     "max_epochs": 3, "patience": 2, "free_std": 0.02,
                     "conditions": ["free", "composed_head", "transe", "title", "random", "gram", "composed_free"]},
            "composer": {"operator": "hrr", "dimension": 16, "composition": "attentive", "concept_factor": "induced",
                         "key_dimension": 4},
            "gram": {"attention_dim": 8}, "kge": {"dimension": 8},
            "analysis": {"ks": [2, 5], "bootstrap_primary": 30, "bootstrap_secondary": 30, "rare_below": -6,
                         "frequent_from": -4, "probe_folds": 3, "neighbours": 3, "tsne_points": 20, "tsne_iterations": 50}}


def test_end_to_end_synthetic_run_writes_only_aggregates_to_the_run_folder(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    conditions = config["head"]["conditions"]
    for seed in (1, 2):
        summary = tf.run_train(config, "stub", seed, conditions=conditions, device="cpu")
        assert set(summary["metrics"]) == set(conditions)
    result = tf.run_analyze(config, "stub", [1, 2], device="cpu", bootstrap=20)
    assert set(result["endpoints"]) == set(conditions)
    assert {"E1", "E2"} <= set(result["primary"])
    assert result["decision"]["available"]
    free = result["endpoints"]["free"]["E1_heldout_macro_auc_all"]
    assert 0.2 < free < 0.8                                # never-trained free rows: chance
    # Licence guard: nothing in the committed run folders names a code (synthetic codes stand in for ICD codes).
    written = "".join(p.read_text() for p in (tmp_path / "runs").rglob("*") if p.is_file())
    assert "Q90" not in written and "100000" not in written
    # Per-item outputs went to the data root only.
    assert not list((tmp_path / "runs").rglob("*.npy")) and not list((tmp_path / "runs").rglob("*.npz"))
    assert list((tmp_path / "data" / "heads" / "stub" / "s1" / "free").glob("scores_eval.npy"))
