"""T1c-ROOD — HRRBERT's really-out-of-distribution protocol on clinical text (`experiments.t1c_rood`).

Every fixture here is synthetic: invented codes ("Q…"), invented patient / admission / concept ids, invented note texts
in the NOTEEVENTS layout and random states — never MIMIC, SNOMED CT or ICD titles.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vsa_embed import icd_coding as ic
from vsa_embed.data import mimic
from vsa_embed.experiments import t1c_icd_frequency as tf
from vsa_embed.experiments import t1c_rood as tr


# -- selection -------------------------------------------------------------------------------------------------------

def _codes(n: int = 300) -> list[str]:
    return [f"Q{7000 + i}" for i in range(n)]


def test_choose_rood_codes_is_hashed_stratified_deterministic_and_pinned() -> None:
    codes = _codes()
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 6, size=len(codes))
    eligible = rng.random(len(codes)) < 0.8
    edges = list(ic.HRRBERT_EDGES)
    first = tr.choose_rood_codes(codes, eligible=eligible, bins=bins, edges=edges, bin_lower_edges=[-12, -10, -8],
                                 quotas=[6, 4, 2], salt="s")
    again = tr.choose_rood_codes(codes, eligible=eligible, bins=bins, edges=edges, bin_lower_edges=[-12, -10, -8],
                                 quotas=[6, 4, 2], salt="s")
    assert first == again and len(first["codes"]) == 12
    index = {c: i for i, c in enumerate(codes)}
    chosen = np.array([index[c] for c in first["codes"]])
    assert eligible[chosen].all()
    assert sorted(Counter_(bins[chosen]).items()) == [(1, 6), (2, 4), (3, 2)]          # bins of lower edge −12, −10, −8
    assert first["sha256"] == ic.holdout_digest(first["codes"])
    # the first `quota` by sha256(salt:code) within each bin
    pool = sorted((codes[i] for i in np.flatnonzero(eligible & (bins == 2))), key=lambda c: ic.hash_key(c, "s"))
    assert set(pool[:4]) <= set(first["codes"]) and not set(pool[4:]) & set(first["codes"])
    other = tr.choose_rood_codes(codes, eligible=eligible, bins=bins, edges=edges, bin_lower_edges=[-12, -10, -8],
                                 quotas=[6, 4, 2], salt="t")
    assert other["codes"] != first["codes"]
    with pytest.raises(ValueError):
        tr.choose_rood_codes(codes, eligible=eligible, bins=bins, edges=edges, bin_lower_edges=[-12], quotas=[10_000], salt="s")


def Counter_(values):            # noqa: N802 - a tiny Counter over numpy values
    from collections import Counter
    return Counter(int(v) for v in values)


def test_concept_disjoint_and_hub_free_masks() -> None:
    members = [["1"], ["2", "3"], ["3"], ["4"], ["5", "6"]]
    assert tr.concept_disjoint_mask(members).tolist() == [True, False, False, True, True]
    entries_of = {"1": [0], "2": [1], "3": [2], "4": [3, 4], "5": [5]}          # "6" has no alias (no entry)
    containing = {0: {7}, 3: {8, 9, 10}, 4: set(), 5: {11}}
    assert tr.hub_free_mask(members, entries_of, containing, 2).tolist() == [True, True, True, False, True]


def test_read_all_diagnoses_and_rood_patients(tmp_path: Path) -> None:
    import csv
    path = tmp_path / "DIAGNOSES_ICD.csv.gz"
    rows = [(1, 10, 100, 1, "Q1"), (2, 10, 100, 2, "Q2"), (3, 10, 101, 1, "Q3"), (4, 11, 102, 1, "Q2"), (5, 12, 103, 1, "Q4"),
            (6, 12, "", 1, "Q1"), (7, 13, 104, 1, "")]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        writer.writerows(rows)
    codes, subject, primary, stats = tr.read_all_diagnoses(path)
    assert codes == {100: {"Q1", "Q2"}, 101: {"Q3"}, 102: {"Q2"}, 103: {"Q4"}}
    assert primary == {100: "Q1", 101: "Q3", 102: "Q2", 103: "Q4"}
    assert stats["rows_without_admission_or_code"] == 2
    # patient 10 has the ROOD code on admission 100 only; both of its admissions leave training (patient-level)
    assert tr.rood_patients(codes, subject, {"Q2"}) == {10, 11}
    assert tr.rood_patients(codes, subject, {"Q9"}) == set()


# -- the ROOD label space (coding) -----------------------------------------------------------------------------------

def _base_label_space(rng: np.random.Generator, n_trained: int = 16, n_held: int = 3, n_nat: int = 2, n_adm: int = 400):
    n = n_trained + n_held + n_nat
    codes = [f"Q{8000 + i}X" for i in range(n)]
    trained = np.array([True] * n_trained + [False] * (n_held + n_nat))
    held = np.array([False] * n_trained + [True] * n_held + [False] * n_nat)
    natural = ~(trained | held)
    adm_labels, split, cpa = [], [], []
    for a in range(n_adm):
        k = int(rng.integers(1, 4))
        labs = np.sort(rng.choice(n, size=k, replace=False))
        adm_labels.append(labs.astype(np.int64))
        split.append("eval" if a % 7 == 0 else ("dev" if a % 7 == 1 else "train"))
        cpa.append(k + int(rng.integers(0, 3)))               # plus unframed codes
    split = np.array(split)
    count = {s: np.zeros(n, dtype=np.int64) for s in ("train", "dev", "eval")}
    for labs, s in zip(adm_labels, split):
        count[s][labs] += 1
    # T1c-F's invariant: naturally unseen codes have no training admission
    for a, labs in enumerate(adm_labels):
        if split[a] == "train" and natural[labs].any():
            split[a] = "eval"
    count = {s: np.zeros(n, dtype=np.int64) for s in ("train", "dev", "eval")}
    for labs, s in zip(adm_labels, split):
        count[s][labs] += 1
    logf = ic.log_frequency(np.where(trained, np.maximum(count["train"], 1), 0), 1000)
    labels = {"codes": codes, "kind": ["1to1"] * n, "members": [[str(100000 + i)] for i in range(n)],
              "frames": [[(0, i % 5)] for i in range(n)], "titles": [f"title {i}" for i in range(n)],
              "gram_paths": [[i, n + (i % 3)] for i in range(n)], "gram_nodes": [f"g{i}" for i in range(n + 3)],
              "t1c_heldout_member": np.zeros(n, dtype=bool), "count": count, "total_train": 1000, "logf": logf,
              "bin": ic.frequency_bins(logf), "edges": list(ic.HRRBERT_EDGES), "trained": trained, "heldout": held,
              "natural_unseen": natural, "n_trained": n_trained, "holdout": {"sha256": "x"}, "concept_frames": {},
              "atomic_count": 5, "relation_count": 1, "version": "v1", "shares_member_set_with_trained": np.zeros(n, dtype=bool)}
    adm = {"hadm": np.arange(5000, 5000 + n_adm), "split": split, "labels": adm_labels, "notes": [[a] for a in range(n_adm)],
           "codes_per_admission": np.array(cpa)}
    return labels, adm


def test_rood_label_space_removes_every_rood_patient_from_training() -> None:
    rng = np.random.default_rng(1)
    labels, adm = _base_label_space(rng)
    rood_codes = ["Q8002X", "Q8007X"]
    rood_ids_base = [labels["codes"].index(c) for c in rood_codes]
    subject = np.array([a // 3 for a in range(len(adm["split"]))])             # 3 admissions per synthetic patient
    has_rood = np.array([bool(set(l.tolist()) & set(rood_ids_base)) for l in adm["labels"]])
    rood_subjects = set(subject[has_rood].tolist())
    rood_patient = np.isin(subject, list(rood_subjects))
    assert rood_patient.sum() > has_rood.sum()                                  # patient-level: more than the coded admissions
    primary = has_rood & (np.arange(len(subject)) % 2 == 0)
    new, new_adm, info = tr.rood_label_space(labels, adm, rood_codes, rood_patient, primary, majority_share=0.5)

    n_trained = new["n_trained"]
    rood_ids = np.flatnonzero(new["rood"])
    assert [new["codes"][i] for i in rood_ids] == sorted(rood_codes, key=lambda c: labels["codes"].index(c))
    assert (rood_ids >= n_trained).all() and not new["trained"][rood_ids].any()
    # the label order is trained, T1c-F held out, ROOD, never trained, and a permutation of the base labels
    order = np.asarray(new["source_index"])
    assert sorted(order.tolist()) == list(range(len(labels["codes"])))
    assert [labels["codes"][i] for i in order] == new["codes"]
    assert [labels["titles"][i] for i in order] == new["titles"]
    assert new["heldout"].sum() == labels["heldout"].sum() and (np.flatnonzero(new["heldout"]) < rood_ids.min()).all()
    # admissions keep their order; labels name the same codes
    for a in range(len(adm["labels"])):
        assert sorted(labels["codes"][i] for i in adm["labels"][a]) == sorted(new["codes"][i] for i in new_adm["labels"][a])
    split = np.asarray(new_adm["split"])
    fit = np.isin(split, ("train", "dev"))
    assert not (fit & rood_patient).any()
    assert (split[rood_patient] == "rood").all()
    assert all(not set(new_adm["labels"][a].tolist()) & set(rood_ids.tolist()) for a in np.flatnonzero(fit))
    assert tr.leakage_audit(new, new_adm) == {"train_dev_admissions_with_rood_code": 0, "train_dev_admissions_of_rood_patients": 0,
                                              "rood_labels_in_trained_range": 0, "rood_labels_with_training_positive": 0}
    # a trained label keeps ≥ 1 training admission; frequency is recomputed on the remaining training admissions
    assert (new["count"]["train"][:n_trained] > 0).all()
    assert new["total_train"] == int(np.asarray(adm["codes_per_admission"])[split == "train"].sum())
    assert info["labels"]["rood"] == 2
    # subsets
    assert (new_adm["rood_any"] == has_rood).all()
    assert (new_adm["rood_primary"] == primary).all()
    share = np.array([np.isin(l, rood_ids).mean() for l in new_adm["labels"]])
    assert (new_adm["rood_majority"] == (has_rood & (share >= 0.5))).all()
    # a ROOD code must be a trained T1c-F code
    with pytest.raises(ValueError):
        tr.rood_label_space(labels, adm, [labels["codes"][-1]], rood_patient, primary)


def test_a_trained_code_that_loses_every_training_admission_becomes_never_trained() -> None:
    rng = np.random.default_rng(2)
    labels, adm = _base_label_space(rng)
    target = labels["codes"].index("Q8005X")
    holders = np.array([target in l for l in adm["labels"]])
    rood_patient = holders.copy()                                               # every admission with the code is excluded
    rood_patient |= np.array([labels["codes"].index("Q8001X") in l for l in adm["labels"]])
    new, new_adm, info = tr.rood_label_space(labels, adm, ["Q8001X"], rood_patient, np.zeros(len(holders), bool))
    position = new["codes"].index("Q8005X")
    assert position >= new["n_trained"] and new["natural_unseen"][position]
    assert info["labels"]["trained_lost_by_exclusion"] >= 1


# -- free head: no trained row for a ROOD code; fallbacks ------------------------------------------------------------

def test_fallback_source_replaces_only_never_trained_rows() -> None:
    base = ic.FreeSource(10, 8, seed=0)
    ids = torch.arange(10)
    mean = ic.FallbackSource(base, 7, "mean")(ids)
    zero = ic.FallbackSource(base, 7, "zero")(ids)
    init = ic.FallbackSource(base, 7, "init")(ids)
    assert torch.equal(mean[:7], base.weight[:7]) and torch.equal(zero[:7], base.weight[:7])
    assert torch.allclose(mean[7:], base.weight[:7].mean(0).expand(3, 8))
    assert torch.equal(zero[7:], torch.zeros(3, 8)) and torch.equal(init, base.weight)
    with pytest.raises(ValueError):
        ic.FallbackSource(base, 7, "nearest")
    head = ic.LabelAttentionHead(base, 8, attention_dim=8, hidden=16)
    derived = ic.with_fallback(head, 7, "mean")
    assert derived.source.base is not head.source and head.source is base                  # the trained head is untouched
    states, mask = torch.randn(3, 4, 8), torch.ones(3, 4, dtype=torch.bool)
    logits = derived(states, mask, ids)
    assert torch.allclose(logits[:, :7], head(states, mask, ids)[:, :7], atol=1e-6)
    assert torch.allclose(logits[:, 7:], logits[:, 7:8].expand(3, 3))                      # one score for every unseen code


def test_rood_codes_receive_no_gradient_in_the_free_head() -> None:
    rng = np.random.default_rng(3)
    width, n = 8, 12
    rows, offsets, labels = [], [0], []
    for a in range(120):
        labs = np.sort(rng.choice(n, size=2, replace=False))
        seg = rng.normal(size=(4, width)).astype(np.float32)
        rows.append(seg); offsets.append(offsets[-1] + 4); labels.append(labs)
    store = ic.SegmentStore(np.concatenate(rows).astype(np.float16), np.asarray(offsets))
    head = ic.LabelAttentionHead(ic.FreeSource(n, 8, seed=1), width, attention_dim=8, hidden=16)
    before = head.source.weight.detach().clone()
    ic.train_heads({"free": head}, store, labels, train_admissions=np.arange(100), dev_admissions=np.arange(100, 120),
                   train_labels=np.arange(9), label_count=n, epochs=2, patience=5, batch=16, lr=1e-2, seed=0)
    assert torch.equal(head.source.weight[9:].detach(), before[9:])            # labels 9–11 (the "ROOD" ids): no trained row
    assert not torch.equal(head.source.weight[:9].detach(), before[:9])


def test_tie_aware_ranks_and_strict_default() -> None:
    logits = torch.tensor([[3.0, 1.0, 1.0, 1.0, 0.0]])
    rows, cols = torch.tensor([0]), torch.tensor([2])
    assert int(ic.ranks_in_rows(logits, rows, cols)) == 2                       # strict: ties do not count (unchanged)
    assert float(ic.ranks_in_rows(logits, rows, cols, ties="half")) == pytest.approx(3.0)   # 1 above + ½·2 ties + 1
    subset = torch.tensor([1, 2, 3])
    assert float(ic.ranks_in_rows(logits, rows, cols, subset, ties="half")) == pytest.approx(2.0)
    outside = torch.tensor([1, 3, 4])                                           # the positive is not in the subset
    assert float(ic.ranks_in_rows(logits, rows, cols, outside, ties="half")) == pytest.approx(2.0)
    scores = np.array([[0.5, 0.5, 0.5], [0.9, 0.1, 0.5]])
    assert tr.zero_shot_midranks(scores, np.array([0, 1]), np.array([1, 2])).tolist() == [2.0, 2.0]


# -- mention filter and document streams ---------------------------------------------------------------------------

def test_mention_filter_matches_whole_words_case_and_punctuation_insensitively() -> None:
    flt = tr.MentionFilter(["zorblax syndrome", "quim's palsy", "vex"])
    assert len(flt) == 3
    assert flt.count("History of Zorblax  syndrome, resolved.") == 1
    assert flt.count("quim s palsy and QUIM'S PALSY") == 2
    assert flt("zorblaxsyndrome and vexing") is True                          # partial words are not mentions
    assert flt("a vex.") is False
    assert flt("") is True
    assert tr.MentionFilter(["a b"]).digest != tr.MentionFilter(["a c"]).digest


def _write_noteevents(path: Path, rows: list[dict]) -> None:
    import csv
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "STORETIME",
                                                    "CATEGORY", "DESCRIPTION", "CGID", "ISERROR", "TEXT"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"CHARTDATE": "", "CHARTTIME": "", "STORETIME": "", "DESCRIPTION": "Report", "CGID": "", "ISERROR": "",
                             **row})


def _notes(tmp_path: Path, n: int = 360) -> Path:
    rows = []
    for i in range(n):
        subject = 2000 + i % 60
        text = f"synthetic note {i} for patient {subject}" + (" mentions zorblax syndrome" if i % 13 == 0 else "")
        rows.append({"ROW_ID": i + 1, "SUBJECT_ID": subject, "HADM_ID": "" if i % 17 == 0 else 9000 + subject * 2 + i % 2,
                     "CATEGORY": ["Radiology", "Nursing", "Discharge summary"][i % 3], "TEXT": text})
    source = tmp_path / "NOTEEVENTS.csv.gz"
    _write_noteevents(source, rows)
    out = tmp_path / "notes"
    mimic.extract_notes(source, out, eval_buckets=3000, shards=3)
    return out


def test_iter_notes_filters_are_opt_in(tmp_path: Path) -> None:
    from collections import Counter
    import pyarrow.parquet as pq
    notes = _notes(tmp_path)
    plain = list(mimic.iter_notes(notes, "train"))
    assert plain == list(mimic.iter_notes(notes, "train", exclude_subjects=None, admissions=None, keep=None))
    excluded = frozenset(range(2000, 2015))
    log: Counter = Counter()
    kept = list(mimic.iter_notes(notes, "train", exclude_subjects=excluded, log=log))
    assert all(int(t.split("patient ")[1].split()[0]) not in excluded for t in kept)
    assert log["dropped_excluded_subject"] + log["yielded"] == len(plain) and log["dropped_excluded_subject"] > 0
    flt = tr.MentionFilter(["zorblax syndrome"])
    filtered = list(mimic.iter_notes(notes, "train", keep=flt))
    assert filtered and all("zorblax" not in t for t in filtered) and len(filtered) < len(plain)
    wanted = set()
    for s in range(3):
        table = pq.read_table(mimic.shard_path(notes, "train", s), columns=["hadm_id"]).column("hadm_id").to_pylist()
        wanted |= {h for h in table if h >= 0 and h % 4 == 0}
    some = list(mimic.iter_notes(notes, "train", admissions=frozenset(wanted)))
    assert 0 < len(some) < len(plain)


def test_rood_documents_drop_rood_patients_and_mentions_and_evaluate_on_rood_admissions(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    notes = _notes(tmp_path)
    shard = tmp_path / "general.parquet"
    pq.write_table(pa.table({"text": [f"general document {i} " * 10 + ("zorblax syndrome" if i % 9 == 0 else "")
                                      for i in range(240)]}), shard)
    excluded = frozenset(range(2000, 2020))
    rood_adm = frozenset(9000 + s * 2 for s in range(2000, 2060))
    RoodDocuments = tr._rood_documents_class()
    docs = RoodDocuments(notes_dir=notes, general_shards=[str(shard)], general_skip=20, eval_general_docs=10,
                         eval_domain_docs=None, calibration=(3.5, 4.6), exclude_subjects=excluded, rood_admissions=rood_adm,
                         mention_filter=tr.MentionFilter(["zorblax syndrome"]), rood_digest="d")
    train = list(docs.train([]))
    assert train and all("zorblax" not in t for t in train)
    assert all(int(t.split("patient ")[1].split()[0]) not in excluded for t in train if t.startswith("synthetic"))
    assert docs.counters["train_notes"]["dropped_excluded_subject"] > 0 and docs.counters["train_general"]["dropped_by_filter"] > 0
    evaluation = list(docs.eval_domain())
    # discharge summaries (i % 3 == 2) of the ROOD admissions, from both sides of the patient split
    expected = 0
    for split in mimic.SPLITS:
        for s in range(3):
            t = pq.read_table(mimic.shard_path(notes, split, s), columns=["hadm_id", "category"]).to_pydict()
            expected += sum(1 for h, c in zip(t["hadm_id"], t["category"]) if h in rood_adm and c == "Discharge summary")
    assert len(evaluation) == expected > 0
    assert RoodDocuments.domain_split == "eval-rood"
    dense = RoodDocuments(notes_dir=notes, general_shards=[str(shard)], general_skip=20, eval_general_docs=10,
                          eval_domain_docs=None, calibration=(3.5, 4.6), exclude_subjects=excluded, rood_admissions=rood_adm,
                          mention_filter=tr.MentionFilter(["zorblax syndrome"]), eval_mentions_only=True, rood_digest="d")
    mentioning = list(dense.eval_domain())
    assert mentioning and all("zorblax syndrome" in t for t in mentioning)
    assert len(mentioning) == sum("zorblax syndrome" in t for t in evaluation)
    assert dense.signature("eval-rood") != docs.signature("eval-rood") and dense.signature("train") == docs.signature("train")
    other = RoodDocuments(notes_dir=notes, general_shards=[str(shard)], general_skip=20, eval_general_docs=10,
                          eval_domain_docs=None, calibration=(3.5, 4.6), exclude_subjects=excluded, rood_admissions=rood_adm,
                          mention_filter=tr.MentionFilter(["other words"]), rood_digest="d")
    assert other.signature("train") != docs.signature("train") and other.signature("eval-general") == docs.signature("eval-general")


def test_document_bounds_split_on_eos() -> None:
    tokens = np.array([5, 6, 0, 7, 0, 0, 8, 9])
    assert tr.document_bounds(tokens, 0) == [(0, 2), (3, 4), (6, 8)]


# -- default-off behaviour of the new keys ---------------------------------------------------------------------------

def test_existing_t1c_and_t1cf_configs_do_not_use_the_new_keys() -> None:
    t1cf = yaml.safe_load(Path("experiments/t1c-clinical/icd-frequency/icd-frequency.yaml").read_text())
    assert "base_root" not in t1cf["paths"] and "free_fallbacks" not in t1cf["head"] and "midranks" not in t1cf["analysis"]
    assert "rood" not in t1cf
    t1c = yaml.safe_load(Path("experiments/t1c-clinical/t1c.yaml").read_text())
    assert "rood" not in t1c
    from vsa_embed.experiments.e9_tracks import TRACKS
    assert TRACKS["t1c"].data_root.name == "snomed-mimic3-smollm2-v2" and TRACKS["t1c"].eval_split == "eval-mimic"
    assert TRACKS["t1c-rood"].eval_split == "eval-rood" and TRACKS["t1c-rood"].licensed
    rood = yaml.safe_load(Path("experiments/t1c-clinical/rood/rood.yaml").read_text())
    for key in ("text", "encode", "composer", "gram", "kge"):
        assert rood[key] == t1cf[key] or key == "text"                          # text.workers differs (≤ 6 on the shared CPU)
    assert {k: v for k, v in rood["head"].items() if k != "free_fallbacks"} == t1cf["head"]
    assert rood["holdout"] == t1cf["holdout"] and rood["seed"] == t1cf["seed"]
    assert rood["rood"]["expected_sha256"] and rood["rood"]["expected_concept_holdout_sha256"]


def test_base_root_is_opt_in(tmp_path: Path) -> None:
    config = {"paths": {"data_root": str(tmp_path / "own")}}
    assert tf.base_root(config) == tf.data_root(config) == tmp_path / "own"
    assert tf.resolve_states_dir(config, "enc") == tmp_path / "own" / "states" / "enc"
    config["paths"]["base_root"] = str(tmp_path / "base")
    assert tf.base_root(config) == tmp_path / "base"
    # never a silent fallback: the base root's states are read only through a verified link (`encode --reuse-base`)
    assert tf.resolve_states_dir(config, "enc") == tmp_path / "own" / "states" / "enc"
    (tmp_path / "own" / "states" / "enc").mkdir(parents=True)
    (tmp_path / "own" / "states" / "enc" / "meta.json").write_text("{}")
    assert tf.resolve_states_dir(config, "enc") == tmp_path / "own" / "states" / "enc"


# -- the frozen-host encode: reuse T1c-F's states only on an exact manifest match -------------------------------------

def _encode_roots(tmp_path: Path, *, n_adm: int = 12, length: int = 160, meta_extra: dict | None = None,
                  complete: bool = True) -> tuple[dict, Path]:
    """A base root with a token store, admissions and complete frozen-host states, and a ROOD-like data root on top."""
    base, own = tmp_path / "base", tmp_path / "own"
    for root in (base, own):
        root.mkdir()
    lengths = np.full(n_adm, length, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    empty_spans = {k: np.zeros(0, dtype=np.int64) for k in ("text", "start", "end", "inject", "entry", "length")}
    empty_spans["confidence"] = np.zeros(0, dtype=np.float32)
    (base / "tokens").mkdir()
    tf.save_tokens(base / "tokens" / "admissions", {"ids": np.ones(int(offsets[-1]), dtype=np.uint16), "offsets": offsets,
                                                    "full_lengths": lengths, "spans": empty_spans})
    tf.save_tokens(base / "tokens" / "titles", {"ids": np.ones(6, dtype=np.uint16), "offsets": np.array([0, 3, 6]),
                                                "full_lengths": np.array([3, 3]), "spans": empty_spans})
    hadm = np.arange(500, 500 + n_adm)
    torch.save({"hadm": hadm, "split": np.array(["train"] * n_adm)}, base / "admissions.pt")
    torch.save({"hadm": hadm, "split": np.array(["rood"] * n_adm)}, own / "admissions.pt")
    config = {"paths": {"data_root": str(own), "base_root": str(base), "runs": str(tmp_path / "runs"),
                        "alias_table": str(tmp_path / "none.json")},
              "text": {"chunk_tokens": 1024, "segment_tokens": 32, "max_tokens": 8192, "tokenizer": "stub-tokenizer"},
              "encode": {"dtype": "bfloat16", "batch_chunks": 8}}
    _, seg_offsets = tf.chunk_plan(lengths, 1024, 32)
    states = base / "states" / "P0"
    states.mkdir(parents=True)
    np.save(states / "seg_offsets.npy", seg_offsets)
    np.memmap(states / "segments.f16", dtype=np.float16, mode="w+", shape=(int(seg_offsets[-1]), 8)).flush()
    np.save(states / "titles.npy", np.zeros((2, 8), dtype=np.float32))
    meta = {**tf.encode_spec(config, "P0", pretrained="stub/host", tokens_dir="tokens", admissions=n_adm,
                             tokens=int(lengths.sum())), "width": 8, "segments": int(seg_offsets[-1]),
            "admissions_done": n_adm if complete else n_adm // 2, "complete": complete, **(meta_extra or {})}
    (states / "meta.json").write_text(json.dumps(meta))
    return config, states


def _digest(folder: Path) -> dict[str, str]:
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_frozen_host_encode_reuses_matching_base_states_without_writing_them(tmp_path: Path) -> None:
    config, base_states = _encode_roots(tmp_path)
    before = _digest(base_states)
    summary = tf.run_encode(config, "P0", pretrained="stub/host", device="cpu", reuse_base=True)
    assert summary["reuse"]["decision"] == "reuse"
    own = tmp_path / "own" / "states" / "P0"
    assert sorted(p.name for p in own.iterdir()) == [tf.REUSE_FILE]           # only the link: no states of its own
    assert tf.resolve_states_dir(config, "P0") == base_states
    store = tf.open_store(config, "P0")
    assert store.offsets.size == 13
    again = tf.run_encode(config, "P0", pretrained="stub/host", device="cpu", reuse_base=True)    # idempotent
    assert again["reuse"]["decision"] == "reused"
    assert _digest(base_states) == before                                      # the base (T1c-F's) states are never written
    # a later change of the base manifest is refused rather than read silently
    meta = json.loads((base_states / "meta.json").read_text())
    (base_states / "meta.json").write_text(json.dumps({**meta, "segments": meta["segments"] + 1}))
    with pytest.raises(RuntimeError):
        tf.resolve_states_dir(config, "P0")


@pytest.mark.parametrize("field, value", [("pretrained", "other/host"), ("segment_tokens", 64), ("max_tokens", 4096),
                                          ("dtype", "float32"), ("tokenizer", "other")])
def test_frozen_host_encode_fails_loudly_on_a_mismatched_base(tmp_path: Path, field: str, value) -> None:
    config, base_states = _encode_roots(tmp_path, meta_extra={field: value})
    before = _digest(base_states)
    with pytest.raises(ValueError):
        tf.run_encode(config, "P0", pretrained="stub/host", device="cpu", reuse_base=True)
    assert not (tmp_path / "own" / "states" / "P0" / tf.REUSE_FILE).exists()
    assert _digest(base_states) == before


def test_reuse_decision_rules() -> None:
    spec = {"encoder": "P0", "pretrained": "h", "channel": False, "tokens": 10, "dtype": "bfloat16"}
    meta = {**spec, "complete": True, "admissions": 3, "admissions_done": 3}
    assert tf.reuse_decision(spec, meta, offsets_equal=True, admissions_equal=True) == ("reuse", [])
    assert tf.reuse_decision(spec, None, offsets_equal=False, admissions_equal=True)[0] == "encode"        # absent
    partial = {**meta, "complete": False, "admissions_done": 1}
    assert tf.reuse_decision(spec, partial, offsets_equal=True, admissions_equal=True)[0] == "encode"      # incomplete
    older = {k: v for k, v in meta.items() if k != "dtype"}                    # written before the field was recorded
    assert tf.reuse_decision(spec, older, offsets_equal=True, admissions_equal=True)[0] == "encode"
    assert tf.reuse_decision(spec, {**meta, "tokens": 11}, offsets_equal=True, admissions_equal=True)[0] == "fail"
    assert tf.reuse_decision(spec, meta, offsets_equal=False, admissions_equal=True) == ("fail", ["seg_offsets"])
    assert tf.reuse_decision(spec, meta, offsets_equal=True, admissions_equal=False) == ("fail", ["admission order"])


def test_reuse_is_opt_in_and_only_for_a_full_frozen_host_encode(tmp_path: Path) -> None:
    config, _ = _encode_roots(tmp_path)
    with pytest.raises(ValueError):
        tf.run_encode(config, "P0", pretrained="stub/host", device="cpu", reuse_base=True, limit=4)
    with pytest.raises(ValueError):
        tf.run_encode(config, "P0", run=tmp_path / "run", device="cpu", reuse_base=True)


# -- end to end on a synthetic data root: training (T1c-F's stage) and the ROOD analysis -----------------------------

def _synthetic_rood_experiment(tmp_path: Path) -> dict:
    """A tiny ROOD data root (labels / admissions in the `prepare` layout) whose states live in a base root."""
    rng = np.random.default_rng(5)
    labels, adm = _base_label_space(rng, n_trained=14, n_held=3, n_nat=2, n_adm=360)
    n = len(labels["codes"])
    width = 16
    directions = rng.normal(size=(n, width)).astype(np.float32)
    rows, offsets = [], [0]
    for labs in adm["labels"]:
        seg = rng.normal(scale=0.5, size=(5, width)).astype(np.float32)
        for j, l in enumerate(labs):
            seg[j % 5] += 2.0 * directions[l]
        rows.append(seg); offsets.append(offsets[-1] + 5)
    base = tmp_path / "base"
    states = base / "states" / "stub"
    states.mkdir(parents=True)
    segments = np.memmap(states / "segments.f16", dtype=np.float16, mode="w+", shape=(offsets[-1], width))
    segments[:] = np.concatenate(rows)
    segments.flush()
    np.save(states / "seg_offsets.npy", np.asarray(offsets))
    np.save(states / "titles.npy", rng.normal(size=(n, width)).astype(np.float32))
    empty_spans = {k: np.zeros(0, dtype=np.int64) for k in ("text", "start", "end", "inject", "entry", "length")}
    empty_spans["confidence"] = np.zeros(0, dtype=np.float32)
    lengths = np.full(len(adm["labels"]), 160, dtype=np.int64)               # 5 segments of 32 tokens each
    (base / "tokens").mkdir()
    tf.save_tokens(base / "tokens" / "admissions", {"ids": np.ones(int(lengths.sum()), dtype=np.uint16),
                                                    "offsets": np.concatenate([[0], np.cumsum(lengths)]),
                                                    "full_lengths": lengths, "spans": empty_spans})
    tf.save_tokens(base / "tokens" / "titles", {"ids": np.ones(n, dtype=np.uint16), "offsets": np.arange(n + 1),
                                                "full_lengths": np.ones(n, dtype=np.int64), "spans": empty_spans})
    text_cfg = {"chunk_tokens": 1024, "segment_tokens": 32, "max_tokens": 8192, "tokenizer": "stub-tokenizer"}
    meta = {"encoder": "stub", "pretrained": "stub/host", "channel": False, "chunk_tokens": 1024, "segment_tokens": 32,
            "tokens_dir": "tokens", "admissions": len(lengths), "tokens": int(lengths.sum()), "dtype": "bfloat16",
            "max_tokens": 8192, "tokenizer": "stub-tokenizer", "width": width, "segments": int(offsets[-1]),
            "admissions_done": len(lengths), "complete": True}
    (states / "meta.json").write_text(json.dumps(meta))
    torch.save(adm, base / "admissions.pt")
    (base / "kge").mkdir()
    torch.save({"vectors": torch.randn(n, 8)}, base / "kge" / "transe-d8.pt")
    labels["frames"] = [[(r, int(a)) for r, a in zip(rng.integers(0, 3, 3), rng.integers(0, 20, 3))] for _ in range(n)]
    labels["atomic_count"], labels["relation_count"] = 20, 3
    rood_codes = [labels["codes"][2], labels["codes"][9]]
    subject = np.arange(len(adm["split"])) // 2
    ids = [labels["codes"].index(c) for c in rood_codes]
    has = np.array([bool(set(l.tolist()) & set(ids)) for l in adm["labels"]])
    rood_patient = np.isin(subject, subject[has])
    new, new_adm, _ = tr.rood_label_space(labels, adm, rood_codes, rood_patient, has & (subject % 2 == 0))
    root = tmp_path / "data"
    root.mkdir()
    torch.save(new, root / "labels.pt")
    torch.save(new_adm, root / "admissions.pt")
    return {"experiment": "t", "version": "test", "seed": 3, "text": text_cfg, "encode": {"dtype": "bfloat16", "batch_chunks": 8},
            "paths": {"data_root": str(root), "base_root": str(base), "runs": str(tmp_path / "runs"), "ontology_pt": str(root / "none.pt"),
                      "alias_table": str(root / "none.json")},
            "head": {"source_dim": 16, "attention_dim": 16, "label_hidden": 32, "batch": 16, "lr": 3e-3, "weight_decay": 0.01,
                     "max_epochs": 3, "patience": 2, "free_std": 0.02, "free_fallbacks": ["mean", "zero"],
                     "conditions": ["free", "composed_head", "transe", "title", "random", "gram", "composed_free"]},
            "composer": {"operator": "hrr", "dimension": 16, "composition": "attentive", "concept_factor": "induced",
                         "key_dimension": 4},
            "gram": {"attention_dim": 8}, "kge": {"dimension": 8},
            "analysis": {"ks": [2, 5], "bootstrap_primary": 25, "bootstrap_secondary": 25, "rare_below": -6,
                         "frequent_from": -4, "probe_folds": 3, "neighbours": 3, "tsne_points": 20, "tsne_iterations": 50,
                         "midranks": True},
            "rood": {"primary": "composed_head", "control": "free_mean"}}


def test_end_to_end_synthetic_rood_run_writes_only_aggregates(tmp_path: Path) -> None:
    config = _synthetic_rood_experiment(tmp_path)
    conditions = config["head"]["conditions"]
    linked = tf.run_encode(config, "stub", pretrained="stub/host", device="cpu", reuse_base=True)   # the P0 job's first step
    assert linked["reuse"]["decision"] == "reuse"
    for seed in (1, 2):
        summary = tf.run_train(config, "stub", seed, conditions=conditions, device="cpu")
        assert set(summary["metrics"]) == set(conditions) | {"free_mean", "free_zero"}
        assert summary["train_admissions"] == int((np.asarray(torch.load(Path(config["paths"]["data_root"]) / "admissions.pt",
                                                                         weights_only=False)["split"]) == "train").sum())
    heads = tmp_path / "data" / "heads" / "stub" / "s1"
    ranks = np.load(heads / "free_mean" / "ranks_untrained_all.npz")
    assert {"general_mid", "zero_shot_mid"} <= set(ranks.files)
    labels = torch.load(tmp_path / "data" / "labels.pt", weights_only=False)
    n_trained = int(labels["n_trained"])
    local = np.flatnonzero(labels["rood"]) - n_trained
    mean_scores = np.load(heads / "free_mean" / "scores_untrained_all.npy")[:, local]
    assert np.allclose(mean_scores, mean_scores[:, :1])                          # the mean row: one score per admission
    free_vectors = torch.load(heads / "free" / "vectors.pt", weights_only=False)
    rood_rows = torch.as_tensor(np.flatnonzero(labels["rood"]))
    assert torch.equal(free_vectors["source"][rood_rows], free_vectors["source_init"][rood_rows])   # never trained

    result = tr.run_analyze(config, "stub", [1, 2], device="cpu", bootstrap=25)
    assert {"free", "free_mean", "free_zero", "composed_head"} <= set(result["endpoints"])
    assert result["primary_contrast"] is not None and result["decision"]["available"]
    assert result["pool"]["rood_codes"] == 2 and result["pool"]["positives"] > 0
    for subset in tr.SUBSETS:
        assert f"R2_{subset}" in result["endpoints"]["composed_head"]
    # T1c-F's own analysis still runs on the ROOD label space (descriptive endpoints)
    tf.run_analyze(config, "stub", [1, 2], device="cpu", bootstrap=10)
    # Licence guard: no code string and no per-item array in the committed run folders.
    written = "".join(p.read_text() for p in (tmp_path / "runs").rglob("*") if p.is_file())
    import re
    leaked = re.findall(r"\bQ80\d\dX\b|\b1000\d\d\b", written)                  # synthetic codes / concept ids as tokens
    assert not leaked, leaked[:5]
    assert not list((tmp_path / "runs").rglob("*.npy")) and not list((tmp_path / "runs").rglob("*.npz"))
    assert list((tmp_path / "data" / "analysis").rglob("per_code_auc.npz"))


def test_lm_endpoint_pairs_windows_and_pools_seeds(tmp_path: Path) -> None:
    quant = tmp_path / "quant"
    strata = np.array(["all", "after_heldout"])
    starts = np.arange(0, 40, 4)
    counts = np.vstack([np.full(10, 100), np.arange(10) % 3 + 1])
    for condition, seed, shift in (("C0'", 1, 0.0), ("C0'", 2, 0.0), ("C5", 1, -0.2), ("C5", 2, -0.1), ("P0", 1, 0.3)):
        folder = quant / "runs" / f"stage__{condition}-s{seed}"
        folder.mkdir(parents=True)
        sums = counts * (3.0 + shift)
        np.savez(folder / "windows.npz", strata=strata, starts=starts, count=counts, sum_ref=sums)
        (folder / "quant.json").write_text(json.dumps({"id": folder.name, "condition": condition, "seed": seed}))
    result = tr.lm_endpoint(quant, resamples=200)
    row = result["C5-C0'"]
    assert row["available"] and row["seeds"] == [1, 2] and row["mean"] == pytest.approx(-0.15)
    assert result["C5-P0"]["mean"] == pytest.approx(-0.45)
    assert "C5-C2" not in result


def test_plan_uses_the_decision_63_slot_and_queues_nothing() -> None:
    import re
    text = tr.plan_commands()
    priorities = {float(p) for p in re.findall(r"--priority ([0-9.]+)", text)}
    assert priorities == {54.4998, 54.49981, 54.49982, 54.49983, 54.49984}
    assert "-report" in text and "GPU-h" in text and "e9_plan --track t1c-rood" in text and "--no-evals" in text
    lines = [line for line in text.splitlines() if "--priority" in line]
    level = {re.search(r"--name (\S+)", line).group(1) if "--name" in line else "e9_plan": float(re.search(r"--priority ([0-9.]+)", line).group(1))
             for line in lines}
    # decisive first: the frozen-host encode, its coding and R1 before E9; the encode reuses T1c-F's states (never --resume)
    assert {level[k] for k in ("t1crood-encode-P0-360M", "t1crood-train-P0-360M-s1", "t1crood-analyze-P0-360M")} == {54.4998}
    assert level["e9_plan"] == 54.49981 and level["t1crood-encode-C5-ROOD-360M"] == 54.49982
    assert level["t1crood-quant-rood"] == 54.49982 and level["t1crood-analyze-C5-ROOD-360M"] == 54.49983
    encode_p0 = next(line for line in lines if "t1crood-encode-P0-360M" in line)
    assert "--reuse-base" in encode_p0 and "rood/rood.yaml" in encode_p0 and "--resume" not in encode_p0
    report_lines = [line for line in lines if "--priority 54.49984" in line]
    assert report_lines and all("-report" in line for line in report_lines)
    assert all("-report" not in line for line in lines if "--priority 54.49984" not in line)
