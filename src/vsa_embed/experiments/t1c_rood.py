"""T1c-ROOD: HRRBERT's "really out of distribution" protocol on clinical text (decision 63, holdout H2, design (A)).

HRRBERT (`resources/vsa-paper.md`, "Really-Out-Of-Distribution") withheld 32 ICD codes entirely from pre-training and
fine-tuning, took every MIMIC-IV patient with any of them as the test set, and compared code embeddings composed from
SNOMED CT (HRRBase) with unstructured ones on those patients. T1c-ROOD does the same on MIMIC-III **text**:

- a frozen, hashed, frequency-stratified set of framed ICD-9 diagnosis codes (the ROOD codes; `select`);
- every admission of every patient with any ROOD code leaves all training data: the coding head's train / dev
  admissions (T1c-F's task, `prepare`) and the T1c language-model corpus (`build`), where every training document
  that mentions a ROOD concept's alias is dropped too and the ROOD concepts are never linked;
- the test set is the ROOD admissions; ROOD codes get code vectors composed zero-shot from SNOMED CT frames, the free
  table has no trained row for them (`head.free_fallbacks`), and the endpoints are per-code AUC and ranks (`analyze`)
  and the loss after ROOD-concept mentions in the ROOD admissions' discharge summaries (E9 on track `t1c-rood`).

Preregistration: `experiments/t1c-clinical/rood/preregistration.md`. Configs: `rood.yaml` (selection, coding; a T1c-F
config with opt-in keys) and `t1c-rood.yaml` (the LM corpus; extends `t1c.yaml`).

**Licence and DUA (decision 58; non-negotiable).** MIMIC-III is credentialed, SNOMED CT and UMLS are licensed. The
ROOD code list, patient and admission identifiers, the concept closure, label tables, corpora, scores and per-code
results are written under `~/data/vsa-llm/t1c/rood-v1/` (mode 700). The committed run folders receive aggregates only
(counts, shares, digests, macro metrics, bootstrap summaries); no stage prints a note, a row, a code, a title or a
concept, and nothing is sent to any external service.

Stages (`python -m vsa_embed.experiments.t1c_rood <stage> --config experiments/t1c-clinical/rood/rood.yaml`):

- `select` (CPU, minutes): the ROOD codes (rule in §3 of the preregistration), the ROOD patients and admissions (all
  of MIMIC-III's `DIAGNOSES_ICD`), the ROOD concepts and their alias-disjoint closure; digests checked against the pins.
- `prepare` (CPU, seconds): the ROOD label space and admission splits for the coding head (T1c-F's admissions, tokens
  and states are reused; labels reordered: trained, T1c-F held-out, ROOD, never trained); the leakage audit.
- `build` (CPU, ≈ 30–60 min): the ROOD LM corpora for SmolLM2 (`train`, `eval-rood`, `eval-mimic`, `eval-general`,
  `ontology.pt`) through T1c's builder, the feasibility of the held-out (= ROOD) stratum and the leakage audit (no
  ROOD-patient note, no ROOD span, no ROOD mention in the decoded training corpus).
- `alias-table`: the E9 evaluation alias table of track `t1c-rood` (for the C5-ROOD encoder and E9 evaluations).
- `analyze --encoder NAME --seeds 1 2 3` (CPU or GPU): the ROOD coding endpoints (R1, R2, subsets, bootstrap).
- `report`: the ROOD endpoints of every encoder and the LM endpoint (L1, from `e4_quant` on `eval-rood`).
- `plan`: the queue commands (printed; never queued here).

The encode and train stages are T1c-F's (`t1c_icd_frequency encode|train --config experiments/t1c-clinical/rood/rood.yaml`); the
frozen P0 encode runs with `--reuse-base`, which links T1c-F's P0 states only on an exact manifest match.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Iterator, Sequence

import numpy as np
import torch

from vsa_embed import icd_coding as ic
from vsa_embed.experiments import t1c_icd_frequency as tf

DEFAULT_CONFIG = Path("experiments/t1c-clinical/rood/rood.yaml")
TRACK_LABEL = "clinical ROOD (SNOMED CT + MIMIC-III; ROOD patients and mentions excluded)"
FALLBACK_CONDITIONS = ("free_mean", "free_zero")
ROOD_CONDITIONS = (*tf.CONDITIONS, *FALLBACK_CONDITIONS)
SUBSETS = ("rood_any", "rood_majority", "rood_primary")
POOL_SPLITS = ("eval", "rood")            # admissions never used to train or stop a head: the ROOD evaluation pool
LM_STRATUM = "after_heldout"              # on track t1c-rood the held-out entries are exactly the ROOD closure
ZERO_SHOT_KS = (1, 5, 10)
WORDS = re.compile(r"\w+")


# -- configuration ------------------------------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """The ROOD config (a T1c-F config with the `rood` block) with the T1c config (`_t1c`) and the LM config (`_lm`)."""
    from .t1_open_corpus import load_config as load_track_config
    config = tf.load_config(Path(path))
    config["_lm"] = load_track_config(Path(config["rood"]["lm_config"]))
    return config


def rood_root(config: dict[str, Any]) -> Path:
    return tf.private_dir(Path(config["rood"]["root"]))


def lm_root(config: dict[str, Any]) -> Path:
    return tf.private_dir(Path(config["_lm"]["paths"]["data_root"]))


def digest_lines(values: Iterable[str]) -> str:
    """sha256 of the sorted values, one per line (`icd_coding.holdout_digest`'s rule)."""
    return ic.holdout_digest(values)


def check_pin(value: str, expected: str | None, what: str, *, required: bool) -> None:
    if expected:
        if value != expected:
            raise ValueError(f"{what} sha256 {value} != pinned {expected}: the frozen set changed")
    elif required:
        raise ValueError(f"{what} is not pinned: run `select`, pin its sha256 in the config, then continue")


def read_lines(path: Path) -> list[str]:
    return [line for line in Path(path).read_text().splitlines() if line]


# -- selection (pure parts, unit-tested) ---------------------------------------------------------------------------

def concept_disjoint_mask(members: Sequence[Sequence[str]]) -> np.ndarray:
    """True for a code none of whose member concepts belongs to another code (its composed vector and its concepts are
    then not shared with, or trained through, any other label)."""
    users: dict[str, set[int]] = defaultdict(set)
    for i, ms in enumerate(members):
        for m in ms:
            users[m].add(i)
    return np.array([all(users[m] == {i} for m in ms) for i, ms in enumerate(members)], dtype=bool)


def hub_free_mask(members: Sequence[Sequence[str]], entries_of: dict[str, Sequence[int]], containing: dict[int, set[int]],
                  max_containing: int) -> np.ndarray:
    """True for a code whose member concepts' linker entries are each contained in at most `max_containing` other
    entries' aliases (T1c's hub rule, so the alias-disjoint closure stays local)."""
    return np.array([all(len(containing.get(e, ())) <= max_containing for m in ms for e in entries_of.get(m, ()))
                     for ms in members], dtype=bool)


def choose_rood_codes(codes: Sequence[str], *, eligible: np.ndarray, bins: np.ndarray, edges: Sequence[float],
                      bin_lower_edges: Sequence[float], quotas: Sequence[int], salt: str) -> dict[str, Any]:
    """Within each listed natural frequency bin (lower edge), the eligible codes ranked by `sha256(salt:code)`; the
    first `quota` of each are the ROOD codes. Deterministic and independent of any model or outcome."""
    if len(bin_lower_edges) != len(quotas):
        raise ValueError("one quota per bin")
    edges = [float(e) for e in edges]
    chosen: list[str] = []
    per_bin: dict[str, dict[str, int]] = {}
    for lower, quota in zip(bin_lower_edges, quotas):
        index = edges.index(float(lower))
        pool = sorted((codes[i] for i in np.flatnonzero(eligible & (np.asarray(bins) == index))),
                      key=lambda c: ic.hash_key(c, salt))
        if len(pool) < int(quota):
            raise ValueError(f"bin {ic.bin_name(index, edges)} has {len(pool)} eligible codes, fewer than its quota {quota}")
        chosen += pool[:int(quota)]
        per_bin[ic.bin_name(index, edges)] = {"eligible": len(pool), "chosen": int(quota)}
    chosen = sorted(set(chosen))
    return {"codes": chosen, "per_bin": per_bin, "sha256": digest_lines(chosen), "salt": salt,
            "quotas": [int(q) for q in quotas], "bins": [float(b) for b in bin_lower_edges]}


def read_all_diagnoses(path: Path) -> tuple[dict[int, set[str]], dict[int, int], dict[int, str], dict[str, int]]:
    """MIMIC-III `DIAGNOSES_ICD` (every admission, not only those with a discharge summary): admission → codes,
    admission → patient, admission → the code with `SEQ_NUM` 1 (the principal diagnosis); plus aggregate counts."""
    codes: dict[int, set[str]] = defaultdict(set)
    subject: dict[int, int] = {}
    primary: dict[int, str] = {}
    stats: Counter = Counter()
    with gzip.open(path, "rt") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            if not row["HADM_ID"] or not row["ICD9_CODE"]:
                stats["rows_without_admission_or_code"] += 1
                continue
            hadm = int(row["HADM_ID"])
            code = row["ICD9_CODE"].strip()
            codes[hadm].add(code)
            if subject.setdefault(hadm, int(row["SUBJECT_ID"])) != int(row["SUBJECT_ID"]):
                raise ValueError("an admission with two patients in DIAGNOSES_ICD")
            if (row.get("SEQ_NUM") or "").strip() == "1":
                primary[hadm] = code
    return dict(codes), subject, primary, dict(stats)


def rood_patients(admission_codes: dict[int, set[str]], subject_of: dict[int, int], rood: set[str]) -> set[int]:
    """Patients with any admission carrying a ROOD code."""
    return {subject_of[h] for h, cs in admission_codes.items() if cs & rood}


# -- the ROOD label space for the coding head (pure, unit-tested) ---------------------------------------------------

PER_CODE_KEYS = ("codes", "kind", "members", "frames", "titles", "gram_paths", "t1c_heldout_member")


def rood_label_space(labels: dict[str, Any], adm: dict[str, Any], rood_codes: Sequence[str], rood_patient: np.ndarray,
                     primary_rood: np.ndarray, *, majority_share: float = 0.5) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """T1c-F's label tables with the ROOD exclusion: every admission of a ROOD patient leaves train / dev (split
    `rood`); labels reordered as trained (≥ 1 remaining training admission, not ROOD, not T1c-F held out), T1c-F's
    held-out codes, the ROOD codes, never trained (T1c-F's naturally unseen and codes that lost every training
    admission). `source_index` maps every new label to its T1c-F id (titles, TransE rows). Training frequencies, bins
    and the label-occurrence denominator are recomputed on the remaining training admissions; `base_bin` keeps the
    natural T1c-F bin (the ROOD strata). Admissions keep T1c-F's order (the state stores are indexed by it)."""
    codes = list(labels["codes"])
    n = len(codes)
    index = {c: i for i, c in enumerate(codes)}
    rood = set(rood_codes)
    if rood - set(codes):
        raise ValueError(f"{len(rood - set(codes))} ROOD codes are not framed labels")
    base_trained = np.asarray(labels["trained"], dtype=bool)
    base_held = np.asarray(labels["heldout"], dtype=bool)
    is_rood = np.zeros(n, dtype=bool)
    is_rood[[index[c] for c in rood]] = True
    if (is_rood & (base_held | ~base_trained)).any():
        raise ValueError("ROOD codes must be T1c-F trained codes (not held out, not naturally unseen)")
    rood_patient = np.asarray(rood_patient, dtype=bool)
    split = np.asarray(adm["split"]).astype(object)
    if rood_patient.shape != split.shape:
        raise ValueError("rood_patient must have one flag per admission")
    new_split = np.where(rood_patient, "rood", split).astype(object)
    adm_labels = adm["labels"]
    count = {s: np.zeros(n, dtype=np.int64) for s in ("train", "dev", "eval", "rood")}
    for a, labs in enumerate(adm_labels):
        count[new_split[a]][labs] += 1
    trained = base_trained & ~is_rood & (count["train"] > 0)
    lost = base_trained & ~is_rood & (count["train"] == 0)
    natural = np.asarray(labels["natural_unseen"], dtype=bool) | lost
    order = np.concatenate([np.flatnonzero(trained), np.flatnonzero(base_held), np.flatnonzero(is_rood),
                            np.flatnonzero(natural)])
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise AssertionError("the ROOD label order is not a permutation of T1c-F's labels")
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    n_trained, n_held, n_rood = int(trained.sum()), int(base_held.sum()), int(is_rood.sum())
    train_rows = np.flatnonzero(new_split == "train")
    total = int(np.asarray(adm["codes_per_admission"])[train_rows].sum())   # all codes, framed or not (T1c-F's rule)
    edges = labels["edges"]
    logf = ic.log_frequency(count["train"][order], total)
    out = {k: v for k, v in labels.items() if k not in PER_CODE_KEYS}
    for key in PER_CODE_KEYS:
        values = labels[key]
        out[key] = values[order] if isinstance(values, np.ndarray) else [values[i] for i in order]
    mask = lambda lo, hi: np.arange(n) >= lo if hi is None else (np.arange(n) >= lo) & (np.arange(n) < hi)  # noqa: E731
    new_trained = mask(0, n_trained)
    members = out["members"]
    trained_sets = {frozenset(ms) for ms, t in zip(members, new_trained) if t}
    out.update(
        count={s: v[order] for s, v in count.items()}, total_train=total, logf=logf, bin=ic.frequency_bins(logf, edges),
        trained=new_trained, heldout=mask(n_trained, n_trained + n_held),
        rood=mask(n_trained + n_held, n_trained + n_held + n_rood), natural_unseen=mask(n_trained + n_held + n_rood, None),
        n_trained=n_trained, source_index=order, base_bin=np.asarray(labels["bin"])[order],
        base_logf=np.asarray(labels["logf"])[order], base_count={s: np.asarray(v)[order] for s, v in labels["count"].items()},
        shares_member_set_with_trained=np.array([frozenset(ms) in trained_sets for ms in members]),
        rood_codes=sorted(rood), rood_sha256=digest_lines(rood), version=f"{labels.get('version', 'v1')}-rood")
    rood_ids = set(range(n_trained + n_held, n_trained + n_held + n_rood))
    new_labels = [np.sort(position[np.asarray(l, dtype=np.int64)]) for l in adm_labels]
    has_rood = np.array([bool(set(l.tolist()) & rood_ids) for l in new_labels])
    share = np.array([np.isin(l, list(rood_ids)).mean() if l.size else 0.0 for l in new_labels])
    adm_out = {**{k: v for k, v in adm.items() if k not in ("split", "labels")},
               "split": new_split.astype(str), "labels": new_labels, "rood_patient": rood_patient,
               "rood_any": has_rood, "rood_majority": has_rood & (share >= float(majority_share)),
               "rood_primary": has_rood & np.asarray(primary_rood, dtype=bool)}
    if (has_rood & ~rood_patient).any():
        raise AssertionError("an admission with a ROOD code is not a ROOD patient's")
    info = {"labels": {"trained": n_trained, "t1cf_heldout": n_held, "rood": n_rood, "never_trained": int(natural.sum()),
                       "trained_lost_by_exclusion": int(lost.sum())},
            "total_train_label_occurrences": total}
    return out, adm_out, info


def leakage_audit(labels: dict[str, Any], adm: dict[str, Any]) -> dict[str, int]:
    """Counts that must be 0: training / dev admissions with a ROOD label or of a ROOD patient, ROOD labels inside the
    trained range, ROOD labels with a training positive."""
    n_trained = int(labels["n_trained"])
    rood_ids = np.flatnonzero(labels["rood"])
    split = np.asarray(adm["split"])
    fit = np.isin(split, ("train", "dev"))
    rood_set = set(rood_ids.tolist())
    return {"train_dev_admissions_with_rood_code": int(sum(1 for a in np.flatnonzero(fit) if set(adm["labels"][a].tolist()) & rood_set)),
            "train_dev_admissions_of_rood_patients": int((fit & np.asarray(adm["rood_patient"], dtype=bool)).sum()),
            "rood_labels_in_trained_range": int((rood_ids < n_trained).sum()),
            "rood_labels_with_training_positive": int((np.asarray(labels["count"]["train"])[rood_ids] > 0).sum())}


# -- mention filter (document exclusion; unit-tested) ----------------------------------------------------------------

class MentionFilter:
    """Whole-word mentions of a set of aliases: an alias is its lower-cased sequence of `\\w+` words, so punctuation and
    spacing between words are ignored (a little stricter than the linker, which needs the exact characters). Called on a
    text it returns True when the text mentions none of them (the `keep` rule of `iter_notes`)."""

    def __init__(self, aliases: Iterable[str]) -> None:
        self.by_last: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for alias in aliases:
            words = tuple(WORDS.findall(alias.lower()))
            if words:
                self.by_last[words[-1]].add(words)
        self.last = frozenset(self.by_last)
        self.digest = hashlib.sha256("\n".join(sorted(" ".join(w) for ws in self.by_last.values() for w in ws)).encode()).hexdigest()

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_last.values())

    def count(self, text: str) -> int:
        words = WORDS.findall(text.lower())
        if self.last.isdisjoint(words):
            return 0
        hits = 0
        for i, word in enumerate(words):
            if word in self.last:
                for alias in self.by_last[word]:
                    n = len(alias)
                    if i + 1 >= n and tuple(words[i + 1 - n:i + 1]) == alias:
                        hits += 1
        return hits

    def __call__(self, text: str) -> bool:
        return self.count(text) == 0


# -- the ROOD document streams --------------------------------------------------------------------------------------

def _rood_documents_class():
    from .t1c_corpus import ClinicalDocuments

    @dataclass
    class RoodDocuments(ClinicalDocuments):
        """T1c's streams with the ROOD exclusion: training notes without any ROOD patient's note, training notes and
        general documents without a mention of a held-out alias; `eval-rood` = the discharge summaries of the ROOD
        admissions (both sides of the patient split) — with `eval_mentions_only`, only those that mention a held-out
        alias (the stratum exists only there; the selection reads the text, never an outcome)."""

        exclude_subjects: frozenset[int] = frozenset()
        rood_admissions: frozenset[int] = frozenset()
        mention_filter: MentionFilter | None = None
        eval_categories: tuple[str, ...] = ("Discharge summary",)
        eval_mentions_only: bool = False
        rood_digest: str = ""
        counters: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
        domain_split: ClassVar[str] = "eval-rood"
        track_label: ClassVar[str] = TRACK_LABEL

        def signature(self, stream: str) -> str:
            spec = {"t1c": super().signature(stream), "rood": self.rood_digest,
                    "filter": self.mention_filter.digest if (self.mention_filter and stream == "train") else None,
                    "eval_categories": list(self.eval_categories) if stream == "eval-rood" else None}
            if stream == "eval-rood" and self.eval_mentions_only:          # new key only when set (earlier builds keep theirs)
                spec["eval_mentions_only"] = self.mention_filter.digest if self.mention_filter else None
            return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

        def train_domain(self) -> Iterator[str]:
            from vsa_embed.data.mimic import iter_notes
            return iter_notes(self.notes_dir, "train", categories=self._category_filter(), exclude_subjects=self.exclude_subjects,
                              keep=self.mention_filter, log=self.counters["train_notes"])

        def train_general(self) -> Iterator[str]:
            counter = self.counters["train_general"]
            for text in super().train_general():
                if self.mention_filter is not None and not self.mention_filter(text):
                    counter["dropped_by_filter"] += 1
                    continue
                counter["yielded"] += 1
                yield text

        def eval_domain(self) -> Iterator[str]:
            from vsa_embed.data.mimic import iter_notes
            categories = frozenset(self.eval_categories) if self.eval_categories else None
            texts = chain.from_iterable(iter_notes(self.notes_dir, split, categories=categories, admissions=self.rood_admissions)
                                        for split in ("train", "eval"))
            if not (self.eval_mentions_only and self.mention_filter is not None):
                return texts
            return (t for t in texts if not self.mention_filter(t))

    return RoodDocuments


# -- stages: select -----------------------------------------------------------------------------------------------

def linker_index(ontology: Any) -> tuple[Any, dict[int, set[int]], dict[str, list[int]]]:
    """The track's full alias table (no holdout), its containment index and concept id → entries."""
    from vsa_embed.span_channel import AliasTable
    from .t1_open_corpus import containment_index
    table = AliasTable.from_pairs(ontology.alias_pairs)
    containing = containment_index(table)
    entries_of: dict[str, list[int]] = defaultdict(list)
    for entry, concepts in enumerate(table.entry_concepts):
        for c in concepts:
            entries_of[ontology.concept_names[c]].append(entry)
    return table, containing, dict(entries_of)


def run_select(config: dict[str, Any]) -> dict[str, Any]:
    from .t1c_corpus import build_track_ontology
    from .t1_open_corpus import holdout_closure
    from vsa_embed.provenance import git_state
    started, git_at_start = time.monotonic(), git_state()
    rcfg, t1c = config["rood"], config["_t1c"]
    base = tf.base_root(config)
    labels = torch.load(base / "labels.pt", weights_only=False)
    adm = torch.load(base / "admissions.pt", weights_only=False)
    if labels["holdout"]["sha256"] != config["holdout"]["expected_sha256"]:
        raise ValueError("the base label space is not T1c-F's pinned v1")
    codes, members = list(labels["codes"]), list(labels["members"])
    count = labels["count"]
    total = np.asarray(count["train"]) + np.asarray(count["dev"]) + np.asarray(count["eval"])
    trained, held = np.asarray(labels["trained"], dtype=bool), np.asarray(labels["heldout"], dtype=bool)

    ontology = build_track_ontology(t1c["ontology"])
    table, containing, entries_of = linker_index(ontology)
    base_ok = trained & ~held
    enough = total >= int(rcfg["min_admissions"])
    disjoint = concept_disjoint_mask(members) if rcfg.get("concept_disjoint", True) else np.ones(len(codes), dtype=bool)
    hub_ok = hub_free_mask(members, entries_of, containing, int(rcfg["max_containing"]))
    eligible = base_ok & enough & disjoint & hub_ok
    bins, edges = np.asarray(labels["bin"]), labels["edges"]
    chosen = choose_rood_codes(codes, eligible=eligible, bins=bins, edges=edges, bin_lower_edges=rcfg["bins"],
                               quotas=rcfg["quotas"], salt=rcfg["salt"])
    check_pin(chosen["sha256"], rcfg.get("expected_sha256"), "ROOD code set", required=False)
    rood = set(chosen["codes"])
    index = {c: i for i, c in enumerate(codes)}
    rood_idx = np.array(sorted(index[c] for c in rood))

    licensed = tf.licensed_root(config)
    admission_codes, subject_of, primary, diagnosis_stats = read_all_diagnoses(licensed / config["sources"]["diagnoses"])
    patients = rood_patients(admission_codes, subject_of, rood)
    rood_adm_all = {h for h, cs in admission_codes.items() if cs & rood}
    excluded_adm_all = {h for h, s in subject_of.items() if s in patients}

    member_concepts = sorted({m for c in rood for m in members[index[c]]}, key=int)
    seeds = {e for m in member_concepts for e in entries_of.get(m, ())}
    closure = holdout_closure(seeds, table, containing)
    # The member concepts (an alias-less one cannot be linked, but is recorded as held out) and the closure's concepts.
    held_concepts = sorted(set(member_concepts) | {ontology.concept_names[c] for e in closure for c in table.entry_concepts[e]},
                           key=int)
    concept_sha = digest_lines(held_concepts)
    check_pin(concept_sha, rcfg.get("expected_concept_holdout_sha256"), "ROOD concept holdout", required=False)

    root = rood_root(config)
    (root / "rood_codes.txt").write_text("\n".join(chosen["codes"]) + "\n")
    (root / "rood_subjects.txt").write_text("\n".join(map(str, sorted(patients))) + "\n")
    (root / "rood_admissions_all.txt").write_text("\n".join(map(str, sorted(rood_adm_all))) + "\n")
    (root / "rood_member_concepts.txt").write_text("\n".join(member_concepts) + "\n")
    (root / "rood_holdout_concepts.txt").write_text("\n".join(held_concepts) + "\n")

    # -- aggregates only --
    hadm = np.asarray(adm["hadm"])
    split = np.asarray(adm["split"])
    t1cf_rood_patient = np.array([subject_of.get(int(h)) in patients for h in hadm])
    t1cf_rood_any = np.array([bool(admission_codes.get(int(h), set()) & rood) for h in hadm])
    all_patients = set(subject_of.values())
    train_side_patients = {subject_of[int(h)] for h, s in zip(hadm, split) if s in ("train", "dev") and int(h) in subject_of}

    def by_bin(mask: np.ndarray) -> dict[str, int]:
        return {ic.bin_name(int(b), edges): int((mask & (bins == b)).sum()) for b in sorted(set(bins[mask].tolist())) if b >= 0}
    summary = {
        "track": TRACK_LABEL, "rule": {k: rcfg[k] for k in ("salt", "bins", "quotas", "min_admissions", "concept_disjoint",
                                                              "max_containing", "exclusion")},
        "eligibility": {"framed_codes": len(codes), "t1cf_trained_not_heldout": int(base_ok.sum()),
                        f"with_at_least_{rcfg['min_admissions']}_admissions": int((base_ok & enough).sum()),
                        "and_concept_disjoint": int((base_ok & enough & disjoint).sum()),
                        "and_hub_free": int(eligible.sum()), "eligible_by_bin": by_bin(eligible)},
        "chosen": {"codes": len(rood), "per_bin": chosen["per_bin"], "sha256": chosen["sha256"],
                   "admissions_per_code": {"min": int(total[rood_idx].min()), "median": float(np.median(total[rood_idx])),
                                           "max": int(total[rood_idx].max()), "sum": int(total[rood_idx].sum())},
                   "t1cf_training_positives_of_rood_codes": int(np.asarray(count["train"])[rood_idx].sum()),
                   "one_to_many_maps": int(sum(1 for i in rood_idx if labels["kind"][i] == "1toM")),
                   "member_concepts": len(member_concepts)},
        "exclusion_mimic3": {"admissions_with_diagnoses": len(admission_codes), "patients": len(all_patients),
                             "rood_patients": len(patients), "rood_patient_share": len(patients) / max(1, len(all_patients)),
                             "admissions_with_rood_code": len(rood_adm_all), "admissions_of_rood_patients": len(excluded_adm_all),
                             "rood_admissions_with_rood_principal_diagnosis": int(sum(1 for h in rood_adm_all if primary.get(h) in rood))},
        "exclusion_t1cf": {"admissions_of_rood_patients": {s: int((t1cf_rood_patient & (split == s)).sum()) for s in ("train", "dev", "eval")},
                           "admissions_with_rood_code": {s: int((t1cf_rood_any & (split == s)).sum()) for s in ("train", "dev", "eval")},
                           "train_dev_admissions_removed_share": float((t1cf_rood_patient & np.isin(split, ("train", "dev"))).sum()
                                                                       / max(1, np.isin(split, ("train", "dev")).sum())),
                           "train_side_patients_removed_share": len(patients & train_side_patients) / max(1, len(train_side_patients))},
        "concept_holdout": {"member_concepts": len(member_concepts),
                            "member_concepts_without_alias": int(sum(1 for m in member_concepts if not entries_of.get(m))),
                            "seed_entries": len(seeds), "closure_entries": len(closure), "held_concepts": len(held_concepts),
                            "sha256": concept_sha},
        "diagnoses": diagnosis_stats, "seconds": round(time.monotonic() - started, 1), "data_root": str(root),
    }
    folder = tf.run_folder(config, f"select-{config['version']}")
    tf.write_json(folder / "summary.json", summary)
    tf.record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="select")
    return summary


# -- stages: prepare (coding) --------------------------------------------------------------------------------------

def frozen_rood(config: dict[str, Any]) -> tuple[list[str], set[int]]:
    """The pinned ROOD codes and patients written by `select` (the code list is checked against its pin)."""
    root = rood_root(config)
    codes = read_lines(root / "rood_codes.txt")
    check_pin(digest_lines(codes), config["rood"].get("expected_sha256"), "ROOD code set", required=True)
    return codes, {int(s) for s in read_lines(root / "rood_subjects.txt")}


def run_prepare(config: dict[str, Any]) -> dict[str, Any]:
    from vsa_embed.provenance import git_state
    started, git_at_start = time.monotonic(), git_state()
    base = tf.base_root(config)
    labels = torch.load(base / "labels.pt", weights_only=False)
    adm = torch.load(base / "admissions.pt", weights_only=False)
    codes, patients = frozen_rood(config)
    admission_codes, subject_of, primary, _ = read_all_diagnoses(tf.licensed_root(config) / config["sources"]["diagnoses"])
    if rood_patients(admission_codes, subject_of, set(codes)) != patients:
        raise AssertionError("the ROOD patients differ from select's")
    hadm = np.asarray(adm["hadm"])
    rood_patient = np.array([subject_of[int(h)] in patients for h in hadm])
    primary_rood = np.array([primary.get(int(h)) in set(codes) for h in hadm])
    new_labels, new_adm, info = rood_label_space(labels, adm, codes, rood_patient, primary_rood,
                                                 majority_share=float(config["rood"]["majority_share"]))
    audit = leakage_audit(new_labels, new_adm)
    if any(audit.values()):
        raise AssertionError(f"ROOD leakage: {audit}")
    root = tf.data_root(config)
    torch.save(new_labels, root / "labels.pt")
    torch.save(new_adm, root / "admissions.pt")
    split = np.asarray(new_adm["split"])
    rood_ids = np.flatnonzero(new_labels["rood"])
    pool = np.isin(split, POOL_SPLITS)
    per_admission = np.array([np.isin(l, rood_ids).sum() for l in new_adm["labels"]])
    summary = {
        "admissions": {s: int((split == s).sum()) for s in ("train", "dev", "eval", "rood")},
        "evaluation_pool": int(pool.sum()), **info,
        "subsets": {s: int(np.asarray(new_adm[s]).sum()) for s in SUBSETS},
        "subsets_by_patient_side": {s: {"train_side": int((np.asarray(new_adm[s]) & np.isin(np.asarray(adm["split"]), ("train", "dev"))).sum()),
                                        "eval_side": int((np.asarray(new_adm[s]) & (np.asarray(adm["split"]) == "eval")).sum())}
                                    for s in SUBSETS},
        "rood_codes_per_rood_admission": {"mean": float(per_admission[np.asarray(new_adm["rood_any"])].mean()),
                                          "max": int(per_admission.max())},
        "rood_positives": {"pool": int(per_admission[pool].sum()), "codes_with_10plus_positives":
                           int((np.asarray(new_labels["count"]["rood"])[rood_ids] + np.asarray(new_labels["count"]["eval"])[rood_ids] >= 10).sum())},
        "rood_codes_by_natural_bin": {ic.bin_name(int(b), new_labels["edges"]): int((np.asarray(new_labels["base_bin"])[rood_ids] == b).sum())
                                      for b in sorted(set(np.asarray(new_labels["base_bin"])[rood_ids].tolist()))},
        "leakage_audit": audit, "rood_sha256": new_labels["rood_sha256"], "seconds": round(time.monotonic() - started, 1),
        "data_root": str(root),
    }
    folder = tf.run_folder(config, f"prepare-{config['version']}")
    tf.write_json(folder / "summary.json", summary)
    tf.record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="prepare")
    return summary


# -- stages: build (LM corpora) ------------------------------------------------------------------------------------

_AUDIT: dict[str, Any] = {}


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:20]


def _audit_documents(job: tuple[str, list[tuple[int, int]], str]) -> tuple[int, int, int, list[str]]:
    """Worker: decode documents (token bounds) of a corpus; count those mentioning a held-out alias; hash every text."""
    corpus_dir, bounds, tokenizer_name = job
    from transformers import AutoTokenizer
    from vsa_embed.data.corpus import TokenCorpus
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    corpus = TokenCorpus.open(Path(corpus_dir))
    flt = _AUDIT.get("filter")
    mentioning = mentions = 0
    hashes = []
    for lo, hi in bounds:
        text = tokenizer.decode(np.asarray(corpus.tokens[lo:hi], dtype=np.int64).tolist())
        hits = flt.count(text) if flt is not None else 0
        mentioning += int(hits > 0)
        mentions += hits
        hashes.append(text_hash(text))
    return len(bounds), mentioning, mentions, hashes


def document_bounds(tokens: np.ndarray, eos: int) -> list[tuple[int, int]]:
    """Token bounds of the EOS-separated documents of a corpus (the EOS itself excluded)."""
    ends = np.flatnonzero(np.asarray(tokens) == eos)
    starts = np.concatenate([[0], ends[:-1] + 1]) if ends.size else np.zeros(0, dtype=np.int64)
    bounds = [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]
    tail = int(ends[-1]) + 1 if ends.size else 0
    if tail < len(tokens):
        bounds.append((tail, len(tokens)))
    return bounds


def corpus_audit(corpus_dir: Path, mention_filter: MentionFilter | None, *, tokenizer_name: str, workers: int = 4,
                 chunk: int = 2000) -> tuple[dict[str, int], set[str]]:
    """Decode every document of a token corpus (byte-level BPE round-trips exactly: `build_corpus` drops a document
    that does not) and count the documents that mention a held-out alias; also the set of document text hashes (for
    the note-identity audit). Aggregates only."""
    import multiprocessing as mp
    from vsa_embed.data.corpus import TokenCorpus
    corpus = TokenCorpus.open(Path(corpus_dir))
    bounds = document_bounds(corpus.tokens, int(corpus.manifest["eos_id"]))
    parts = [bounds[i:i + chunk] for i in range(0, len(bounds), chunk)]
    _AUDIT["filter"] = mention_filter
    jobs = [(str(corpus_dir), part, tokenizer_name) for part in parts]
    totals = np.zeros(3, dtype=np.int64)
    hashes: set[str] = set()

    def absorb(result: tuple[int, int, int, list[str]]) -> None:
        totals[:] += np.asarray(result[:3])
        hashes.update(result[3])
    if workers > 1 and len(parts) > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for result in pool.imap_unordered(_audit_documents, jobs):
                absorb(result)
    else:
        for job in jobs:
            absorb(_audit_documents(job))
    return {"documents": int(totals[0]), "documents_mentioning": int(totals[1]), "mentions": int(totals[2])}, hashes


def note_identity_audit(notes_dir: Path, patients: set[int], training_hashes: set[str]) -> dict[str, int]:
    """Do any ROOD patient's note texts occur among the decoded training documents? Hashes of every training-side
    note, split into ROOD patients' and the others'; a ROOD note text that also belongs to another patient (templated
    reports) cannot be attributed and is counted apart. Positive control: the other patients' notes found in training."""
    import pyarrow.parquet as pq
    from vsa_embed.data.mimic import shard_path
    shards = int(json.loads((Path(notes_dir) / "extract.json").read_text())["shards"])
    rood_hashes: set[str] = set()
    other_hashes: set[str] = set()
    rood_notes = 0
    for s in range(shards):
        parquet = pq.ParquetFile(shard_path(notes_dir, "train", s))
        for batch in parquet.iter_batches(columns=["subject_id", "text"], batch_size=4096):
            data = batch.to_pydict()
            for subject, text in zip(data["subject_id"], data["text"]):
                if int(subject) in patients:
                    rood_hashes.add(text_hash(text)); rood_notes += 1
                else:
                    other_hashes.add(text_hash(text))
    unique_rood = rood_hashes - other_hashes
    return {"rood_patient_training_side_notes": rood_notes, "rood_note_texts": len(rood_hashes),
            "rood_note_texts_shared_with_other_patients": len(rood_hashes & other_hashes),
            "rood_only_note_texts_in_training": len(unique_rood & training_hashes),
            "shared_note_texts_in_training": len(rood_hashes & other_hashes & training_hashes),
            "other_patients_note_texts_in_training": len(other_hashes & training_hashes)}


def run_build(config: dict[str, Any], *, text_audit: bool = True) -> dict[str, Any]:
    """The ROOD LM corpora (SmolLM2), their feasibility and the leakage audit."""
    from itertools import islice
    from transformers import AutoTokenizer
    from vsa_embed.data.corpus import TokenCorpus
    from vsa_embed.span_channel import AliasTable
    from vsa_embed.provenance import git_state
    from .t1c_corpus import build_track_ontology, feasibility_by_stratum, sanitize_manifests, track_documents
    from .t1_open_corpus import (assert_alias_disjoint, chars_per_token, general_settings, relink_for_host, train_frequency,
                                 verify_shards)
    started, git_at_start = time.monotonic(), git_state()
    lm, rcfg = config["_lm"], config["rood"]
    lm_rood = lm.get("rood") or {}
    root = rood_root(config)
    out = lm_root(config)
    codes, patients = frozen_rood(config)
    held_names = read_lines(root / "rood_holdout_concepts.txt")
    concept_sha = digest_lines(held_names)
    check_pin(concept_sha, rcfg.get("expected_concept_holdout_sha256"), "ROOD concept holdout", required=True)
    rood_adm_all = frozenset(int(h) for h in read_lines(root / "rood_admissions_all.txt"))

    ontology = build_track_ontology(lm["ontology"])
    index = ontology.concept_index
    held_idx = [index[n] for n in held_names]
    full = AliasTable.from_pairs(ontology.alias_pairs, holdout=held_idx, include_holdout=True)
    train_table = full.without_holdout()
    assert_alias_disjoint(full, train_table)
    held_entries = sorted(full.heldout_entries())
    held_set = set(held_entries)
    held_aliases = [a for a, e in full.alias_to_entry.items() if e in held_set]
    mention_filter = MentionFilter(held_aliases) if lm_rood.get("mention_filter", True) else None

    tokenizer_name, data = lm["tokenizer"], lm["data"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    general = general_settings(lm)
    verify_shards(general["shards"], general["shard_sha256"])
    plain = track_documents(lm)                                   # T1c's streams (calibration, eval-mimic)
    calibration_docs = int(lm["mix"]["calibration_docs"])
    calibration = (chars_per_token(list(islice(plain.eval_domain(), calibration_docs)), tokenizer),
                   chars_per_token(list(islice(plain.eval_general(), calibration_docs)), tokenizer))
    plain.calibration = calibration
    RoodDocuments = _rood_documents_class()
    documents = RoodDocuments(notes_dir=plain.notes_dir, general_shards=plain.general_shards, general_skip=plain.general_skip,
                              eval_general_docs=plain.eval_general_docs, eval_domain_docs=None, domain_share=plain.domain_share,
                              calibration=calibration, categories=plain.categories, exclude_subjects=frozenset(patients),
                              rood_admissions=rood_adm_all, mention_filter=mention_filter,
                              eval_categories=tuple(lm_rood.get("eval_categories") or ("Discharge summary",)),
                              eval_mentions_only=bool(lm_rood.get("eval_mentions_only", False)),
                              rood_digest=digest_lines(codes) + ":" + concept_sha)
    wanted = list(lm_rood.get("corpora") or ["train", "eval-rood", "eval-mimic", "eval-general"])
    shared = dict(full=full, train_table=train_table, ontology=ontology, holdout_sha256=concept_sha,
                  train_min_subtokens=int(data["train_min_subtokens"]), min_subtokens=int(data["min_subtokens"]),
                  workers=int(lm["workers"]), train_tokens=int(data["train_tokens"]), eval_mix_tokens=data.get("eval_mix_tokens"))
    info = relink_for_host(tokenizer_name, out, documents=documents,
                           corpora=tuple(c for c in wanted if c in ("train", "eval-rood", "eval-general")), **shared)
    if "eval-mimic" in wanted:
        extra = relink_for_host(tokenizer_name, out, documents=plain, corpora=("eval-mimic",), **shared)
        info["corpora"].update(extra["corpora"])
    built = info["corpora"]["train"]["tokens"]
    if built < 0.99 * int(data["train_tokens"]):
        raise ValueError(f"training corpus has {built:,} of {int(data['train_tokens']):,} tokens (the stream ran out)")
    # Licensed: the held-out concept names stay in the data root (`e9_tracks` replays the alias table from them).
    (out / "holdout_concepts.txt").write_text("\n".join(held_names) + "\n")

    # -- feasibility of the held-out (= ROOD) stratum on eval-rood and eval-mimic --
    criteria = lm["feasibility"]
    feasibility = {}
    for split_name in ("eval-rood", "eval-mimic"):
        if (out / split_name / "manifest.json").exists():
            feasibility[split_name] = [feasibility_by_stratum(out / split_name, heldout_entries=held_entries,
                                                              frequency=train_frequency(out / "train", len(full.entry_concepts), t),
                                                              criteria=criteria, threshold=t) for t in (2, 3)]

    # -- leakage audit (aggregates) --
    train_spans = TokenCorpus.open(out / "train").spans
    held_mask = np.zeros(len(full.entry_concepts), dtype=bool)
    held_mask[held_entries] = True
    audit: dict[str, Any] = {
        "training_spans_of_heldout_entries": int(held_mask[train_spans["entry"].astype(np.int64)].sum()),
        "stream_counts": {k: dict(v) for k, v in documents.counters.items()},
    }
    failures = {"spans": audit["training_spans_of_heldout_entries"]}
    if text_audit:
        decoded, hashes = corpus_audit(out / "train", mention_filter, tokenizer_name=tokenizer_name, workers=int(lm["workers"]))
        audit["decoded_training_corpus"] = decoded
        audit["note_identity"] = note_identity_audit(plain.notes_dir, patients, hashes)
        audit["decoded_eval_rood"] = corpus_audit(out / "eval-rood", mention_filter, tokenizer_name=tokenizer_name,
                                                  workers=int(lm["workers"]))[0]
        failures.update(mentions=decoded["documents_mentioning"] if mention_filter is not None else 0,
                        rood_notes=audit["note_identity"]["rood_only_note_texts_in_training"])
    failures = {k: v for k, v in failures.items() if v}
    sanitize_manifests({"x": info})
    summary = {
        "track": TRACK_LABEL, "tokenizer": tokenizer_name, "data_root": str(out),
        "holdout": {"concepts": len(held_names), "entries": len(held_entries), "aliases": len(held_aliases), "sha256": concept_sha,
                    "mention_filter_aliases": len(mention_filter) if mention_filter else 0,
                    "mention_filter_sha256": mention_filter.digest if mention_filter else None},
        "alias_table_sha256": full.digest(), "mix": {"chars_per_token": {"mimic": calibration[0], "general": calibration[1]}},
        "corpora": {name: {k: m.get(k) for k in ("tokens", "documents", "spans", "skipped_documents", "alias_table_sha256")}
                    for name, m in info["corpora"].items()},
        "source_shares": info.get("source_shares"), "ontology_sha256": info.get("ontology_sha256"),
        "linked_entries_in_train": info.get("linked_entries_in_train_at_min_subtokens"),
        "feasibility": {"criteria": criteria, "by_split": feasibility}, "leakage_audit": audit,
        "leakage_failures": failures, "seconds": round(time.monotonic() - started, 1),
    }
    folder = tf.run_folder(config, f"build-{config['version']}")
    tf.write_json(folder / "summary.json", summary)
    (folder / "report.md").write_text(render_build_report(summary))
    tf.record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="build")
    if failures:
        raise AssertionError(f"ROOD LM leakage audit failed: {failures}")
    return summary


def render_build_report(summary: dict[str, Any]) -> str:
    lines = [f"# T1c-ROOD LM corpus — {summary['track']}", "",
             "Aggregates only (licensed data stay under the data root). Preregistration: "
             "`experiments/t1c-clinical/rood/preregistration.md`.", "",
             f"Concept holdout (the ROOD closure): {summary['holdout']['concepts']:,} concepts, {summary['holdout']['entries']:,} "
             f"entries, {summary['holdout']['aliases']:,} aliases; sha256 `{summary['holdout']['sha256']}`.", "",
             "| corpus | tokens | documents | spans |", "|---|---:|---:|---:|"]
    for name, m in summary["corpora"].items():
        lines.append(f"| {name} | {m['tokens']:,} | {m['documents']:,} | {m['spans']:,} |")
    lines += ["", "## Feasibility of the held-out (ROOD) stratum", "",
              "| split | ℓ_min | occurrences (split) | entries | entries ≥ 5 | verdict (split) | E9 windows: occ. / entries ≥ 5 | verdict | windows needed |",
              "|---|---:|---:|---:|---:|---|---|---|---:|"]
    for split, blocks in summary["feasibility"]["by_split"].items():
        for block in blocks:
            row = block["strata"]["after_heldout"]
            s, w = row["split"], row["trainer_windows"]
            lines.append(f"| {split} | {block['min_subtokens']} | {s['occurrences']:,} | {s['entries']:,} | {s['entries_5plus']:,} | "
                         f"{row['verdict_split']} | {w['occurrences']:,} / {w['entries_5plus']:,} ({w['windows']:,}) | "
                         f"{row['verdict_trainer_windows']} | {row['windows_needed'] or 'not reached'} |")
    audit = summary["leakage_audit"]
    lines += ["", "## Leakage audit", "",
              f"- training spans of held-out entries: {audit['training_spans_of_heldout_entries']}",
              f"- stream counts: `{json.dumps(audit['stream_counts'])}`"]
    if "decoded_training_corpus" in audit:
        lines.append(f"- decoded training corpus: `{json.dumps(audit['decoded_training_corpus'])}`")
        lines.append(f"- note identity (training-side notes vs decoded training documents): `{json.dumps(audit['note_identity'])}`")
        lines.append(f"- decoded eval-rood: `{json.dumps(audit['decoded_eval_rood'])}`")
    lines.append(f"- failures: `{json.dumps(summary['leakage_failures'])}`")
    return "\n".join(lines) + "\n"


def run_alias_table(config: dict[str, Any]) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    spec = track_spec("t1c-rood")
    path = ensure_alias_table(spec, Path(config["paths"]["alias_table"]).expanduser())
    return {"alias_table": str(path), "exists": Path(path).exists()}


# -- stages: analyze (coding endpoints) ----------------------------------------------------------------------------

def zero_shot_midranks(scores: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Midrank (ties with other candidates counted half) of `scores[r, c]` among the columns of `scores`."""
    values = scores[rows, cols]
    above = (scores[rows] > values[:, None]).sum(1)
    equal = (scores[rows] == values[:, None]).sum(1) - 1
    return above + 0.5 * equal + 1.0


def rood_point_estimates(scores: np.ndarray, positives: np.ndarray, subsets: dict[str, np.ndarray],
                         general_ranks: dict[str, np.ndarray] | None = None, ks: Sequence[int] = (10, 100)) -> dict[str, Any]:
    """ROOD endpoints of one condition × seed on the evaluation pool: per-code AUC (R1's terms) and, per admission
    subset, the zero-shot midrank of each ROOD positive among the ROOD codes (R2: MRR, recall@k) and, when given, the
    generalized midranks among all labels (`general_ranks[subset]`, recall@k)."""
    auc = ic.auc_columns(scores, positives)
    out: dict[str, Any] = {"auc": auc, "R1": float(np.nanmean(auc)), "subsets": {}}
    for name, mask in subsets.items():
        rows, cols = np.nonzero(positives & mask[:, None])
        if rows.size == 0:
            out["subsets"][name] = {"pairs": 0}
            continue
        ranks = zero_shot_midranks(scores, rows, cols)
        row = {"pairs": int(rows.size), "admissions": int(np.unique(rows).size), "mrr": float(np.mean(1.0 / ranks)),
               **{f"recall{k}": float(np.mean(ranks <= k)) for k in ZERO_SHOT_KS}, "rows": rows, "rr": 1.0 / ranks}
        if general_ranks is not None and name in general_ranks and general_ranks[name].size:
            g = general_ranks[name]
            row.update({f"generalized_recall{k}": float(np.mean(g <= k)) for k in ks})
            row["generalized_median_rank"] = float(np.median(g))
        out["subsets"][name] = row
    return out


def load_rood_condition(config: dict[str, Any], encoder: str, seed: int, condition: str, tag: str = "") -> dict[str, Any] | None:
    folder = tf.heads_dir(config, encoder, seed, tag) / condition
    if not (folder / "scores_untrained_all.npy").exists():
        return None
    ranks = np.load(folder / "ranks_untrained_all.npz")
    return {"untrained": np.load(folder / "scores_untrained_all.npy", mmap_mode="r"), "ranks": {k: ranks[k] for k in ranks.files}}


def run_analyze(config: dict[str, Any], encoder: str, seeds: Sequence[int], *, conditions: Sequence[str] | None = None,
                device: str = "cpu", bootstrap: int | None = None, tag: str = "", label: str = "") -> dict[str, Any]:
    """The ROOD coding endpoints of one encoder (preregistration §5–§7), with the two-way bootstrap."""
    from vsa_embed.provenance import git_state
    git_at_start, started = git_state(), time.monotonic()
    root = tf.data_root(config)
    rcfg, an = config["rood"], config["analysis"]
    labels = torch.load(root / "labels.pt", weights_only=False)
    adm = torch.load(root / "admissions.pt", weights_only=False)
    n_trained = int(labels["n_trained"])
    rood_ids = np.flatnonzero(labels["rood"])
    local = rood_ids - n_trained
    primary, control = rcfg["primary"], rcfg["control"]
    wanted = list(conditions or ROOD_CONDITIONS)
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for c in wanted:
        found = {s: load_rood_condition(config, encoder, s, c, tag) for s in seeds}
        found = {s: v for s, v in found.items() if v is not None}
        if found:
            runs[c] = found
    if control not in runs:
        raise RuntimeError(f"the control {control!r} has no runs for {encoder}")
    first = next(iter(runs[control]))
    all_adm = np.load(tf.heads_dir(config, encoder, first, tag) / "all_admissions.npy")
    split = np.asarray(adm["split"])[all_adm]
    pool = np.flatnonzero(np.isin(split, POOL_SPLITS))
    rood_set = {int(r): j for j, r in enumerate(rood_ids)}
    positives = np.zeros((pool.size, rood_ids.size), dtype=bool)
    for row, a in enumerate(all_adm[pool]):
        for l in adm["labels"][a]:
            j = rood_set.get(int(l))
            if j is not None:
                positives[row, j] = True
    subsets = {s: np.asarray(adm[s], dtype=bool)[all_adm[pool]] for s in SUBSETS}
    pool_index = {int(r): i for i, r in enumerate(pool)}
    ks = [int(k) for k in an["ks"]]

    def general_ranks(r: dict[str, Any]) -> dict[str, np.ndarray]:
        rk = r["ranks"]
        key = "general_mid" if "general_mid" in rk else "general"
        is_rood = np.isin(rk["cols"], rood_ids)
        out = {}
        for name, mask in subsets.items():
            keep = is_rood & np.array([pool_index.get(int(x), -1) >= 0 and mask[pool_index[int(x)]] for x in rk["rows"]], dtype=bool)
            out[name] = np.asarray(rk[key])[keep].astype(np.float64)
        return out

    per: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for c, by_seed in runs.items():
        for s, r in by_seed.items():
            scores = np.asarray(r["untrained"][pool][:, local], dtype=np.float32)
            per[c][s] = rood_point_estimates(scores, positives, subsets, general_ranks(r), ks)
            per[c][s]["scores"] = scores

    base_bin = np.asarray(labels["base_bin"])[rood_ids]
    edges = labels["edges"]
    bins_present = sorted(set(base_bin.tolist()))

    def mean_seeds(c: str, fn: Callable[[dict[str, Any]], float]) -> float:
        vals = [fn(v) for v in per[c].values()]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")
    endpoints: dict[str, Any] = {}
    for c in per:
        row: dict[str, Any] = {"R1_rood_macro_auc": mean_seeds(c, lambda v: v["R1"]), "seeds": sorted(per[c]),
                               "R1_by_natural_bin": {ic.bin_name(int(b), edges): mean_seeds(c, lambda v, b=b: float(np.nanmean(v["auc"][base_bin == b])))
                                                     for b in bins_present}}
        for name in SUBSETS:
            sub = {"pairs": next(iter(per[c].values()))["subsets"][name].get("pairs", 0)}
            for key in ("mrr", *[f"recall{k}" for k in ZERO_SHOT_KS], *[f"generalized_recall{k}" for k in ks], "generalized_median_rank"):
                sub[key] = mean_seeds(c, lambda v, key=key, name=name: v["subsets"][name].get(key, float("nan")))
            row[f"R2_{name}"] = sub
        endpoints[c] = row

    # -- two-way bootstrap of R1 (pool admissions: Poisson weights; ROOD codes resampled; seeds resampled) --
    dev = torch.device(device)
    replicates = int(bootstrap or an["bootstrap_primary"])
    generator = np.random.default_rng(int(config["seed"]))
    weights = generator.poisson(1.0, size=(replicates, pool.size)).astype(np.float32)
    code_draw = generator.integers(0, rood_ids.size, size=(replicates, rood_ids.size))
    # Seeds are drawn per condition from a generator keyed by its name, so a contrast does not depend on which other
    # conditions exist (a later analysis with composed_c5 reproduces the primary contrast exactly).
    seed_draw = {c: np.random.default_rng([int(config["seed"]), zlib.crc32(c.encode())]).integers(0, len(per[c]), size=(replicates, len(per[c])))
                 for c in per}
    boot: dict[str, dict[str, np.ndarray]] = {}
    for c in per:
        seed_list = sorted(per[c])
        per_seed_auc = []
        per_seed_mrr = {name: [] for name in SUBSETS}
        for s in seed_list:
            v = per[c][s]
            auc = ic.WeightedAuc(v["scores"], positives, device=dev)
            draws = torch.stack([auc(torch.as_tensor(weights[b], device=dev)) for b in range(replicates)])
            per_seed_auc.append(draws.cpu().numpy())
            for name in SUBSETS:
                sub = v["subsets"][name]
                if sub.get("pairs", 0):
                    w = weights[:, sub["rows"]]
                    per_seed_mrr[name].append((w @ sub["rr"]) / np.maximum(w.sum(1), 1e-12))
        stats = {"R1": np.array([np.nanmean(np.mean([per_seed_auc[j][b] for j in seed_draw[c][b]], axis=0)[code_draw[b]])
                                 for b in range(replicates)])}
        for name in SUBSETS:
            if per_seed_mrr[name]:
                stats[f"MRR_{name}"] = np.array([np.mean([per_seed_mrr[name][j][b] for j in seed_draw[c][b]]) for b in range(replicates)])
        boot[c] = stats

    def contrast(a: str, b: str, metric: str) -> dict[str, Any] | None:
        if metric not in boot.get(a, {}) or metric not in boot.get(b, {}):
            return None
        delta = boot[a][metric] - boot[b][metric]
        return {"delta_bootstrap_mean": tf.finite(np.nanmean(delta)), "ci95": [tf.finite(np.nanpercentile(delta, 2.5)),
                                                                              tf.finite(np.nanpercentile(delta, 97.5))],
                "p_two_sided": ic.bootstrap_pvalue(delta)}
    primary_result = contrast(primary, control, "R1") if primary in boot else None
    vs_control = {}
    for metric in ("R1", *[f"MRR_{s}" for s in SUBSETS]):
        rows = {c: contrast(c, control, metric) for c in boot if c != control}
        rows = {c: v for c, v in rows.items() if v}
        adjusted = ic.holm({c: v["p_two_sided"] for c, v in rows.items()})
        for c in rows:
            rows[c]["p_holm"] = adjusted[c]
        vs_control[metric] = rows
    specificity = {}
    for metric in ("R1", "MRR_rood_any"):
        rows = {c: contrast(primary, c, metric) for c in boot if c not in (primary,)}
        rows = {c: v for c, v in rows.items() if v}
        adjusted = ic.holm({c: v["p_two_sided"] for c, v in rows.items()})
        for c in rows:
            rows[c]["p_holm"] = adjusted[c]
        specificity[metric] = rows
    decision = decide(primary_result, specificity.get("R1", {}))

    # per-code AUCs (per item) stay in the data root
    analysis_dir = tf.private_dir(root / "analysis" / f"rood-{encoder}{tag}{label}")
    np.savez(analysis_dir / "per_code_auc.npz", rood_ids=rood_ids,
             **{f"{c}_s{s}": v["auc"] for c in per for s, v in per[c].items()})
    result = {"encoder": encoder, "seeds": list(seeds), "conditions": list(per), "bootstrap_replicates": replicates,
              "primary": primary, "control": control,
              "pool": {"admissions": int(pool.size), "rood_codes": int(rood_ids.size),
                       "codes_with_positive": int(positives.any(0).sum()), "positives": int(positives.sum()),
                       **{f"{s}_admissions": int(subsets[s].sum()) for s in SUBSETS}},
              "endpoints": endpoints, "primary_contrast": primary_result, "vs_control": vs_control,
              "specificity": specificity, "decision": decision, "seconds": round(time.monotonic() - started, 1)}
    folder = tf.run_folder(config, f"analysis-rood-{encoder}{tag}{label}")
    tf.write_json(folder / "endpoints.json", result)
    (folder / "report.md").write_text(render_analysis_report(result))
    tf.record_run(folder, config, git_at_start=git_at_start, device=device, stage="analyze")
    return result


def decide(primary: dict[str, Any] | None, specificity: dict[str, Any]) -> dict[str, Any]:
    """The preregistered readings (§7) from the numbers."""
    if not primary:
        return {"available": False}
    d, hi, p = primary["delta_bootstrap_mean"], primary["ci95"][1], primary["p_two_sided"]
    readings = {}
    for c, row in specificity.items():
        dd, (l2, h2) = row["delta_bootstrap_mean"], row["ci95"]
        if dd is None or l2 is None:
            continue
        readings[c] = ("primary better" if row["p_holm"] < 0.05 and dd > 0 else
                       "competitor better" if row["p_holm"] < 0.05 and dd < 0 else
                       "equivalent within ±0.01" if l2 > -0.01 and h2 < 0.01 else "inconclusive")
    return {"available": True, "R1_supported": bool(d is not None and d > 0 and p < 0.05),
            "R1_refuted": bool(hi is not None and hi < 0.02), "R1_specificity": readings}


def _f(value: Any, digits: int = 3) -> str:
    return tf._fmt(value, digits)


def render_analysis_report(result: dict[str, Any]) -> str:
    lines = [f"# T1c-ROOD coding — encoder {result['encoder']} (seeds {result['seeds']})", "",
             "Aggregates only (per-code values stay under the data root). Preregistration: "
             "`experiments/t1c-clinical/rood/preregistration.md`.", "",
             f"Pool: `{json.dumps(result['pool'])}`. Bootstrap replicates: {result['bootstrap_replicates']}.", "",
             f"## R1 — ROOD macro-AUC on the evaluation pool (primary {result['primary']} − control {result['control']})", "",
             "| condition | R1 | " + " | ".join(next(iter(result["endpoints"].values()))["R1_by_natural_bin"]) + " |",
             "|---|---:|" + "---:|" * len(next(iter(result["endpoints"].values()))["R1_by_natural_bin"])]
    for c, e in result["endpoints"].items():
        lines.append(f"| {c} | {_f(e['R1_rood_macro_auc'])} | " + " | ".join(_f(v) for v in e["R1_by_natural_bin"].values()) + " |")
    p = result["primary_contrast"]
    if p:
        lines += ["", f"Primary: Δ = {_f(p['delta_bootstrap_mean'], 4)} [{_f(p['ci95'][0], 4)}, {_f(p['ci95'][1], 4)}], "
                      f"p = {_f(p['p_two_sided'], 4)}. Decision: `{json.dumps(result['decision'])}`"]
    lines += ["", "## R2 — ROOD positives ranked among the ROOD codes (zero-shot) and among all labels (generalized)", "",
              "| condition | subset | pairs | MRR | recall@1 | recall@5 | recall@10 | gen. recall@10 | gen. recall@100 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c, e in result["endpoints"].items():
        for s in SUBSETS:
            r = e[f"R2_{s}"]
            lines.append(f"| {c} | {s} | {r['pairs']} | {_f(r['mrr'])} | {_f(r['recall1'])} | {_f(r['recall5'])} | {_f(r['recall10'])} | "
                         f"{_f(r.get('generalized_recall10'))} | {_f(r.get('generalized_recall100'))} |")
    lines += ["", "## Contrasts (bootstrap; Holm within each metric)", ""]
    for metric, rows in result["specificity"].items():
        for c, row in rows.items():
            lines.append(f"- {metric}: {result['primary']} − {c}: Δ = {_f(row['delta_bootstrap_mean'], 4)} "
                         f"[{_f(row['ci95'][0], 4)}, {_f(row['ci95'][1], 4)}], Holm p = {_f(row['p_holm'], 4)}")
    return "\n".join(lines) + "\n"


# -- stages: report (coding + LM) ---------------------------------------------------------------------------------

def lm_endpoint(quant_dir: Path, *, stratum: str = LM_STRATUM, candidate: str = "C5", references: Sequence[str] = ("C0'", "C2", "P0"),
                resamples: int = 2000, variant: str = "ref") -> dict[str, Any]:
    """L1 (and its references): candidate − reference loss on `stratum` from `e4_quant` window sums, pooled over the
    common seeds (P0, evaluation only, pairs with every seed), cluster bootstrap over evaluation windows."""
    from vsa_embed.statistics import paired_ratio_bootstrap
    records = [json.loads(p.read_text()) for p in sorted((Path(quant_dir) / "runs").glob("*/quant.json"))]
    runs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        with np.load(Path(quant_dir) / "runs" / record["id"] / "windows.npz") as data:
            strata = data["strata"].tolist()
            if stratum not in strata:
                continue
            i = strata.index(stratum)
            runs[str(record["condition"])][int(record.get("seed") or 1)] = {
                "sum": data[f"sum_{variant}"][i].astype(np.float64), "count": data["count"][i].astype(np.float64),
                "starts": data["starts"]}
    out: dict[str, Any] = {"stratum": stratum, "variant": variant, "conditions": {c: sorted(v) for c, v in runs.items()}}
    if candidate not in runs:
        return {**out, "available": False}
    for reference in references:
        if reference not in runs:
            continue
        seeds = sorted(runs[candidate]) if reference == "P0" else sorted(set(runs[candidate]) & set(runs[reference]))
        pairs = [(runs[candidate][s], runs[reference][1 if reference == "P0" else s]) for s in seeds]
        if not pairs or any(not np.array_equal(a["starts"], b["starts"]) or not np.array_equal(a["count"], b["count"]) for a, b in pairs):
            out[f"{candidate}-{reference}"] = {"available": False, "note": "windows or target counts do not pair"}
            continue
        d = sum(a["sum"] - b["sum"] for a, b in pairs)
        n = sum(a["count"] for a, _ in pairs)
        base = sum(b["sum"] for _, b in pairs)
        if n.sum() <= 0:
            out[f"{candidate}-{reference}"] = {"available": False, "note": "no targets in the stratum"}
            continue
        boot = paired_ratio_bootstrap(d, n, base, resamples=resamples, seed=0)
        out[f"{candidate}-{reference}"] = {"available": True, "seeds": seeds, "targets": int(n.sum()), **boot}
    return out


def run_report(config: dict[str, Any], *, encoders: Sequence[str], quant_dirs: Sequence[Path], label: str = "") -> dict[str, Any]:
    """The ROOD endpoints in one place: R1 / R2 per encoder (from the analysis folders) and L1 per quant folder."""
    from vsa_embed.provenance import git_state
    git_at_start = git_state()
    coding = {}
    for encoder in encoders:
        path = Path(config["paths"]["runs"]) / f"analysis-rood-{encoder}" / "endpoints.json"
        if path.exists():
            result = json.loads(path.read_text())
            coding[encoder] = {k: result[k] for k in ("primary_contrast", "decision", "pool")}
            coding[encoder]["R1"] = {c: e["R1_rood_macro_auc"] for c, e in result["endpoints"].items()}
    lm = {str(q): lm_endpoint(Path(q)) for q in quant_dirs if (Path(q) / "runs").exists()}
    result = {"coding": coding, "lm": lm}
    folder = tf.run_folder(config, f"report-{config['version']}{label}")
    tf.write_json(folder / "endpoints.json", result)
    lines = ["# T1c-ROOD — endpoints", "", "Aggregates only. Preregistration: `experiments/t1c-clinical/rood/preregistration.md`.", "",
             "## Coding (R1: ROOD macro-AUC, primary composed_head − free_mean)", ""]
    for encoder, row in coding.items():
        p = row["primary_contrast"] or {}
        lines.append(f"- {encoder}: R1 {json.dumps({c: round(v, 4) for c, v in row['R1'].items() if v == v})}; "
                     f"Δ = {_f(p.get('delta_bootstrap_mean'), 4)} [{_f((p.get('ci95') or [None, None])[0], 4)}, "
                     f"{_f((p.get('ci95') or [None, None])[1], 4)}]; decision `{json.dumps(row['decision'])}`")
    lines += ["", f"## LM (L1: loss after ROOD-concept mentions in eval-rood, `{LM_STRATUM}`)", ""]
    for q, row in lm.items():
        for key, value in row.items():
            if isinstance(value, dict) and value.get("available"):
                lines.append(f"- {q} {key}: Δ = {_f(value['mean'], 4)} [{_f(value['ci_low'], 4)}, {_f(value['ci_high'], 4)}] nats, "
                             f"relative {_f(value.get('relative'), 4)}, p = {_f(value['p_value'], 4)}, seeds {value['seeds']}, "
                             f"targets {value['targets']:,}")
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    tf.record_run(folder, config, git_at_start=git_at_start, device="cpu", stage="report")
    return result


# -- plan -----------------------------------------------------------------------------------------------------------

ROOD_CONFIG_ARG = "--config experiments/t1c-clinical/rood/rood.yaml"
T1CF_CONFIG_ARG = "--config experiments/t1c-clinical/icd-frequency/icd-frequency.yaml"
E9_STAGE = "t1c-rood"
E9_RUNS = Path("experiments/e9-retrofit/runs") / E9_STAGE
# Decision 63's band, decisive first (author, 2026-10-08): frozen-host coding + R1, E9 training, ROOD-trained encoders and
# rescoring, the remaining analyses, reports.
LEVELS = (54.4998, 54.49981, 54.49982, 54.49983, 54.49984)


BUILD_SUMMARY = Path("experiments/t1c-clinical/rood/runs/build-v1/summary.json")
EVAL_GENERAL = Path("~/data/vsa-llm/t1c/rood-v1/lm/eval-general")
# e4_quant on SmolLM2-360M: ≈ 0.25 GPU-h per pass of 21,827 windows (T1c-2's whole-split estimate; ref + INT8-A = 2 passes).
QUANT_HOURS_PER_WINDOW_PASS = 0.25 / 21827


def whole_split_windows(summary_path: Path = BUILD_SUMMARY, split: str = "eval-rood") -> int | None:
    """Windows of the whole `eval-rood` split (the build summary's feasibility table), for `e4_quant --windows`."""
    if not Path(summary_path).exists():
        return None
    summary = json.loads(Path(summary_path).read_text())
    blocks = summary.get("feasibility", {}).get("by_split", {}).get(split) or []
    return int(blocks[0]["windows_for_whole_split"]) if blocks else None


def plan_commands(*, levels: Sequence[float] = LEVELS, hours: dict[str, float] | None = None,
                  windows: int | None = None) -> str:
    """The queue commands of T1c-ROOD (printed; never executed here), decisive block first (author, 2026-10-08): the
    frozen-host coding and its R1 analysis, then E9, then the ROOD-trained encoders and the rescoring, then the other
    analyses, then the reports. GPU-h from T1c / T1c-F's measured costs."""
    windows = windows or whole_split_windows()
    h = {"e9_train": 1.16, "e9_p0": 0.03, "encode_p0": 1.5, "encode_run": 1.7, "train_seed": 0.75, "train_seed_c5": 0.85,
         "train_c5_only": 0.15, "analyze": 0.1, "report": 0.4, **(hours or {})}
    quant_rood = 2 * (windows or 21827) * QUANT_HOURS_PER_WINDOW_PASS
    quant_general = 2 * 2048 * QUANT_HOURS_PER_WINDOW_PASS
    decisive, training, evaluations, analyses, report = levels
    q = "PYTHONPATH=src $PY -m vsa_embed.jobqueue add"
    m = "$PY -m vsa_embed.experiments.t1c_icd_frequency"
    r = "$PY -m vsa_embed.experiments.t1c_rood"
    c5 = E9_RUNS / "SmolLM2-360M-full-C5-s1"
    c0 = E9_RUNS / "SmolLM2-360M-full-C0p-s1"
    eight = "free composed_head composed_c5 transe title random gram composed_free"
    p0_hours = h["encode_p0"] + 3 * h["train_seed"] + h["analyze"]
    lm_hours = 9 * h["e9_train"] + h["e9_p0"]
    run_hours = 2 * h["encode_run"] + 6 * h["train_seed_c5"] + 3 * h["train_c5_only"]
    eval_hours = 10 * (quant_rood + quant_general)
    analysis_hours = 5 * h["analyze"]
    total = lm_hours + p0_hours + run_hours + eval_hours + analysis_hours
    whole = windows or "WHOLE"
    lines = ["#!/usr/bin/env bash",
             "# T1c-ROOD (decision 63, H2 design (A); preregistration.md in this folder): exact queue commands. NOT EXECUTED by",
             "# the agent. Printed by `python -m vsa_embed.experiments.t1c_rood plan`. Run from the main checkout's root after merging:",
             "#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python",
             "# The CPU stages (select, prepare, build, alias-table) have run; their outputs are under ~/data/vsa-llm/t1c/rood-v1/",
             "# and ~/data/vsa-llm/t1c/e9/t1c-rood.json (shared by every checkout).",
             f"# Run order (author, 2026-10-08; decisive first): {decisive} frozen-host (P0-360M) encode, coding and R1 analysis;",
             f"# {training} E9 training (seeds 1–3); {evaluations} coding on the ROOD-trained C0' / C5 and the rescoring;",
             f"# {analyses} the remaining analyses; {report} reports. Equal priorities run in creation order: run this file top to bottom.",
             "# GPU-h: idle-GPU estimates from measured costs — E9 SmolLM2-360M ≈ 67 min per 50M-token run plus its 2,048-window",
             "# evaluations (e9_plan --dry-run); T1c-F's smoke-scaled encode 1.5 / 1.7 GPU-h and head training ≈ 0.8 GPU-h per",
             "# 7-condition seed (ROOD trains on 84% of T1c-F's admissions; free_mean / free_zero add two scoring passes); e4_quant",
             f"# ≈ 0.25 GPU-h per pass of 21,827 windows (T1c-2). Total ≈ {total:.1f} GPU-h.",
             "set -euo pipefail", ': "${PY:?set PY to the pinned interpreter}"', "",
             f"# --- 1. at {decisive}: coding on P0 (frozen SmolLM2-360M, never saw MIMIC), the primary encoder, and R1. The encode",
             "# reuses T1c-F's P0 states (t1cf-encode-P0-360M, 54.41) only when their manifest matches exactly (host, token store",
             "# and truncation, chunking, precision, admission order): it then writes only REUSED.json under rood-v1/coding and",
             "# costs nothing. Absent, incomplete or older (unrecorded fields) states → it encodes into rood-v1/coding/states;",
             "# a definite mismatch fails loudly. It never writes T1c-F's states, and T1c-F's job never reads ROOD's. "
             f"≈ {p0_hours:.1f} GPU-h (≈ {p0_hours - h['encode_p0']:.1f} on reuse) ---",
             f"{q} --name t1crood-encode-P0-360M --priority {decisive} --min-free-gb 20 -- {m} encode {ROOD_CONFIG_ARG} "
             f"--encoder P0-360M --pretrained HuggingFaceTB/SmolLM2-360M --reuse-base   # 0 on reuse, else ≈ {h['encode_p0']:.1f} GPU-h (resumable)"]
    for seed in (1, 2, 3):
        lines.append(f"{q} --name t1crood-train-P0-360M-s{seed} --priority {decisive} --min-free-gb 10 --no-resume -- {m} train "
                     f"{ROOD_CONFIG_ARG} --encoder P0-360M --seed {seed}   # 7 conditions + free_mean / free_zero ≈ {h['train_seed']:.2f} GPU-h")
    lines.append(f"{q} --name t1crood-analyze-P0-360M --priority {decisive} --min-free-gb 2 --no-resume -- {r} analyze "
                 f"{ROOD_CONFIG_ARG} --encoder P0-360M --seeds 1 2 3 --device cuda   # R1 (primary) ≈ {h['analyze']:.2f} GPU-h")
    lines += ["", f"# --- 2. at {training}: E9 on the ROOD corpus (track t1c-rood): SmolLM2-360M P0 / C0' / C2 / C5 × seeds 1–3, training",
              f"# only (no per-run probe / item evaluations: T1c's items are built on its own holdout, trained here). ≈ {lm_hours:.1f} GPU-h ---",
              f"PYTHONPATH=src $PY -m vsa_embed.experiments.e9_plan --track t1c-rood --stage {E9_STAGE} --hosts SmolLM2-360M "
              f"--models P0 C0p C2 C5 --seeds 1 2 3 --no-evals --priority {training} --level-step 0.00001 --queue", "",
              f"# --- 3. at {evaluations}: coding on the ROOD-trained encoders (E9 seed 1: {c0.name}, {c5.name}): their encodes,",
              "# 8 conditions (composed_c5 = the C5-ROOD composer; ROOD concepts never linked in its training) and composed_c5 on",
              f"# P0 (≈ {run_hours:.1f} GPU-h); L1 on the whole eval-rood split ({whole} windows; e4_quant ref + INT8-A; P0, C0', C2, C5",
              f"# × 3 seeds) and the locality check on eval-general (C3's documents, the runs' 2,048 windows) (≈ {eval_hours:.1f} GPU-h) ---"]
    for enc, run, extra in (("C0p-ROOD-360M", c0, ""), ("C5-ROOD-360M", c5, " --channel on")):
        lines.append(f"{q} --name t1crood-encode-{enc} --priority {evaluations} --min-free-gb 20 -- {m} encode {ROOD_CONFIG_ARG} "
                     f"--encoder {enc} --run {run}{extra}   # ≈ {h['encode_run']:.1f} GPU-h (resumable)")
        for seed in (1, 2, 3):
            lines.append(f"{q} --name t1crood-train-{enc}-s{seed} --priority {evaluations} --min-free-gb 10 --no-resume -- {m} train "
                         f"{ROOD_CONFIG_ARG} --encoder {enc} --seed {seed} --conditions {eight} --c5-run {c5}   # ≈ {h['train_seed_c5']:.2f} GPU-h")
    for seed in (1, 2, 3):
        lines.append(f"{q} --name t1crood-train-P0-360M-c5dict-s{seed} --priority {evaluations} --min-free-gb 10 --no-resume -- {m} train "
                     f"{ROOD_CONFIG_ARG} --encoder P0-360M --seed {seed} --conditions composed_c5 --c5-run {c5}   # ≈ {h['train_c5_only']:.2f} GPU-h")
    quant_runs = " ".join(str(E9_RUNS / f"SmolLM2-360M-{x}") for x in
                          ["frozen-P0-s1", *[f"full-{model}-s{seed}" for model in ("C0p", "C2", "C5") for seed in (1, 2, 3)]])
    lines += [f"{q} --name t1crood-quant-rood --priority {evaluations} --min-free-gb 5 -- $PY -m vsa_embed.experiments.e4_quant --runs {quant_runs} "
              f"--output experiments/e9-retrofit/quant-full/{E9_STAGE} --bits 8 --variants A --baseline \"C0'\" --references C2 "
              f"--windows {whole} --resume --title \"E9 T1c-ROOD on the whole eval-rood split\"",
              f"{q} --name t1crood-quant-general --priority {evaluations} --min-free-gb 5 -- $PY -m vsa_embed.experiments.e4_quant --runs {quant_runs} "
              f"--output experiments/e9-retrofit/quant-general/{E9_STAGE} --bits 8 --variants A --baseline \"C0'\" --references C2 "
              f"--eval-corpus {EVAL_GENERAL.expanduser()} --resume --title \"E9 T1c-ROOD on general text\"", "",
              f"# --- 4. at {analyses}: the remaining analyses (the bootstrap on the GPU): P0 again with composed_c5 (-pass2; the primary",
              "# contrast is unchanged: its draws do not depend on the other conditions), C0' / C5-ROOD, and T1c-F's endpoints under",
              f"# ROOD (descriptive). ≈ {analysis_hours:.1f} GPU-h ---",
              f"{q} --name t1crood-analyze-P0-360M-pass2 --priority {analyses} --min-free-gb 2 --no-resume -- {r} analyze "
              f"{ROOD_CONFIG_ARG} --encoder P0-360M --seeds 1 2 3 --device cuda --label -pass2   # ≈ {h['analyze']:.2f} GPU-h"]
    for enc in ("C0p-ROOD-360M", "C5-ROOD-360M"):
        lines.append(f"{q} --name t1crood-analyze-{enc} --priority {analyses} --min-free-gb 2 --no-resume -- {r} analyze "
                     f"{ROOD_CONFIG_ARG} --encoder {enc} --seeds 1 2 3 --device cuda   # ≈ {h['analyze']:.2f} GPU-h")
    for enc in ("P0-360M", "C0p-ROOD-360M", "C5-ROOD-360M"):
        lines.append(f"{q} --name t1crood-analyze-t1cf-{enc} --priority {analyses} --min-free-gb 2 --no-resume -- {m} analyze "
                     f"{ROOD_CONFIG_ARG} --encoder {enc} --seeds 1 2 3 --device cuda   # T1c-F's endpoints under ROOD (descriptive)")
    lines += ["", f"# --- 5. at {report} (CPU lane: names contain -report): R9 for the stage, then the ROOD endpoints ---",
              f"{q} --name t1crood-e9-report --priority {report} --min-free-gb 1 --no-resume -- $PY -m vsa_embed.experiments.e9_report "
              f"--runs {E9_RUNS} --quant experiments/e9-retrofit/quant-full/{E9_STAGE} --quant-general experiments/e9-retrofit/quant-general/{E9_STAGE} "
              f"--output experiments/e9-retrofit/report/{E9_STAGE} --overwrite   # ≈ {h['report']:.1f} h",
              f"{q} --name t1crood-report --priority {report} --min-free-gb 1 --no-resume -- {r} report {ROOD_CONFIG_ARG} "
              f"--encoders P0-360M P0-360M-pass2 C0p-ROOD-360M C5-ROOD-360M --quant experiments/e9-retrofit/quant-full/{E9_STAGE}   # minutes"]
    return "\n".join(lines) + "\n"


# -- CLI ------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select", "prepare", "build", "alias-table", "analyze", "report", "plan"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--encoder")
    parser.add_argument("--encoders", nargs="+", default=["P0-360M", "P0-360M-pass2", "C0p-ROOD-360M", "C5-ROOD-360M"])
    parser.add_argument("--quant", type=Path, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-text-audit", action="store_true", help="build: skip decoding the training corpus")
    args = parser.parse_args(argv)
    if args.stage == "plan":
        print(plan_commands(), end="")
        return
    config = load_config(args.config)
    if args.stage == "select":
        result = run_select(config)
    elif args.stage == "prepare":
        result = run_prepare(config)
    elif args.stage == "build":
        result = run_build(config, text_audit=not args.no_text_audit)
        result = {k: result[k] for k in ("holdout", "corpora", "leakage_failures", "seconds")}
    elif args.stage == "alias-table":
        result = run_alias_table(config)
    elif args.stage == "analyze":
        if not args.encoder:
            parser.error("analyze needs --encoder")
        result = run_analyze(config, args.encoder, args.seeds, conditions=args.conditions, device=args.device,
                             bootstrap=args.bootstrap, tag=args.tag, label=args.label)
        result = {k: result[k] for k in ("encoder", "pool", "primary_contrast", "decision")}
    else:
        result = run_report(config, encoders=args.encoders, quant_dirs=args.quant, label=args.label)
    print(json.dumps(result, indent=1, default=tf._native)[:6000])


if __name__ == "__main__":
    main()
