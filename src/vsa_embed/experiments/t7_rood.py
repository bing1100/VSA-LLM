"""T7-ROOD: the HRRBERT-style "really out of distribution" holdout of T7 (decision 63, holdout H1), and the date split of
T7's records into the two learning rounds of E13.

T7 v1 holds 908 entries out of the linker only: their channel rows are never trained, but their names still occur,
unlinked, in the training text, so the host can learn the word from text and only the channel is zero-shot
(`experiments/t4-chemistry/README.md`, holdout). HRRBERT kept its held-out codes out of every training document. Here
every **training** document that mentions a held-out record is dropped; evaluation documents keep them.

- **Names** (`heldout_names`): every candidate name of a held-out record under the track's selection policy
  (`mesh_novel.candidate_aliases`: all its MeSH terms except bare abbreviations and names under 4 characters), linked
  or not, plus its linked aliases.
- **Matcher** (`NameMatcher`): the T7 screen's `MentionCounter`, i.e. case-insensitive whole alphanumeric tokens, so a
  name matches at word boundaries ("venetoclax-resistant" mentions venetoclax; "fascinating" does not mention fascin).
- **Budget** (`ExcludingDocuments`, `budget_pass`): the T7 training stream without those documents. The domain tokens
  they held are refilled (`top_up: domain`) from the training-side PubMed abstracts of the same files that mention no
  selected name, until the reference tokenizer's corpus reaches `match_train_tokens` (T7 v1: 57,268,320; as
  `build_corpus` cuts, the document that crosses the budget is kept). The stream is then fixed as a document count
  (`train_documents`), so every host tokenizer reads the same documents.
- **Holdout.** `split: frequency` keeps the track's own holdout (same presample, same choice, pinned by
  `data.expected_holdout_sha256`), so T7 and T7-ROOD differ only in the exclusion. `split: date` (E13) holds out the
  latest `round2_fraction` of the records by MeSH `DateIntroduced` (the record-level date that replaced `DateCreated`
  in the 2025+ XML; fallback: the publication year of the first training-side PubMed mention), plus the containment
  closure: round 2 (`date_holdout`).
- **Audit** (`audit_training`): every written training document of every tokenizer is decoded and matched again; the
  build fails unless none contains a held-out name. With `reference_root` (T7 v1) the evaluation corpora are compared
  byte for byte and the training documents are hashed against the reference's (kept, removed, added).
- **E13 rounds** (`split: date`; `build_round_corpora`): `train` is round 1. `train-round2` holds the training-side
  abstracts round 1 dropped (they mention a round-2 name), mixed 50/50 with the general documents that follow round 1's.
  `eval-round2` / `eval-round1` hold the evaluation abstracts that do / do not mention a round-2 name.
  `ontology-round2.pt` is the channel ontology with nothing held out. The run folder gets the round split, the round-2
  records (gold frames, SCR notes) and an E11-format held-out set (`e11_read_to_learn.build_heldout_set`).

Enabled by the opt-in `holdout:` section of a `t1_open_corpus` config (`exclude_training_documents: true`); without it
every build is unchanged. The E13 interface is documented in `experiments/t7-new-vocabulary/ROOD.md`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from itertools import chain, islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from vsa_embed.data.corpus import TokenCorpus, build_corpus, tokenizer_fingerprint
from vsa_embed.data.pubmed import file_digest, iter_pubmed, pmid_bucket
from vsa_embed.ontologies.mesh_novel import MentionCounter, mention_key
from vsa_embed.span_channel import AliasTable

from .t1_open_corpus import (TrackDocuments, channel_ontology, containment_index, feasibility_report, guard_reuse,
                             holdout_closure, pubmed_documents, record_shares, save_if_changed, train_frequency)

SPLITS = ("frequency", "date")
NAME_SETS = ("candidates", "linked")
TOP_UPS = ("domain", "none")
EVAL_CORPORA = ("eval", "eval-pubmed", "eval-general")
ROUND_CORPORA = ("train-round2", "eval-round2", "eval-round1")
SIDECAR = "rood.json"              # next to each training corpus: the document budget, stream tally and audit
EXAMPLES = 20


@dataclass(frozen=True)
class RoodSettings:
    """The config's `holdout:` section (opt-in; `from_config` returns None without `exclude_training_documents: true`)."""

    split: str = "frequency"                 # frequency: the track's holdout; date: round 2 = the latest records (E13)
    names: str = "candidates"                # candidates: every candidate name of a held-out record; linked: its aliases
    top_up: str = "domain"                   # refill the dropped domain tokens from abstracts mentioning no selected name
    match_train_tokens: int | None = None    # the reference tokenizer's training budget (T7 v1's corpus size)
    round2_fraction: float = 0.30            # split: date
    reference_root: str | None = None        # a build to compare with (T7 v1's data root)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RoodSettings | None":
        spec = dict(config.get("holdout") or {})
        if not spec.pop("exclude_training_documents", False):
            if spec:
                raise ValueError("a holdout: section needs exclude_training_documents: true")
            return None
        unknown = set(spec) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown holdout keys {sorted(unknown)}")
        settings = cls(**spec)
        for name, value, allowed in (("split", settings.split, SPLITS), ("names", settings.names, NAME_SETS),
                                     ("top_up", settings.top_up, TOP_UPS)):
            if value not in allowed:
                raise ValueError(f"holdout.{name} must be one of {allowed} (got {value!r})")
        if settings.top_up == "domain" and not settings.match_train_tokens:
            raise ValueError("holdout.top_up: domain needs match_train_tokens (the refill pool is far larger than the budget)")
        if not 0 < settings.round2_fraction < 1:
            raise ValueError("holdout.round2_fraction must lie in (0, 1)")
        return settings


# -- the held-out names and their matcher ----------------------------------------------------------------------------

class NameMatcher:
    """Whether a text mentions any of `names`: `MentionCounter` keys (case-insensitive whole alphanumeric tokens and
    single punctuation marks, as the T7 screen counted them), i.e. a match at word boundaries."""

    def __init__(self, names: Iterable[str]) -> None:
        self.keys = sorted({key for key in (mention_key(n) for n in names) if key})
        self.counter = MentionCounter(self.keys)

    def __call__(self, text: str) -> bool:
        return self.counter.mentions(text)

    def found(self, text: str) -> list[str]:
        return sorted(self.counter.count(text))

    def digest(self) -> str:
        return hashlib.sha256("\n".join(self.keys).encode()).hexdigest()


def heldout_names(config: dict[str, Any], ontology: Any, table: AliasTable, settings: RoodSettings,
                  records: Sequence[dict[str, Any]] | None = None) -> tuple[list[str], dict[str, int]]:
    """(names, counts): the linked aliases of the held-out entries and, for `names: candidates`, every candidate name of
    the held-out records (`mesh_novel.candidate_aliases` under the track's policy, selected for linking or not).
    `records`: the MeSH records (descriptors + SCRs) if already parsed."""
    held = table.heldout_entries()
    linked = sorted(alias for alias, entry in table.alias_to_entry.items() if entry in held)
    if settings.names == "linked":
        return linked, {"linked_aliases": len(linked), "candidate_names": 0, "names": len(linked), "records": len(table.holdout)}
    from vsa_embed.ontologies.mesh_novel import NovelVocabularyPolicy, candidate_aliases
    spec = config["ontology"]
    if spec.get("adapter") != "mesh_novel":
        raise ValueError("holdout.names: candidates reads MeSH records (mesh_novel adapter); use names: linked")
    uis = {ontology.concept_names[c] for c in table.holdout}
    records = records if records is not None else mesh_records(config)
    held_records = [r for r in records if r["ui"] in uis]
    # The records are chosen by UI, so the policy's class and year filters must not drop any of them.
    policy = replace(NovelVocabularyPolicy.from_config(spec.get("policy")), include_descriptors=True,
                     descriptor_min_year=None, scr_min_year=None, exclude_descriptor_classes=(),
                     scr_classes=tuple(sorted({r.get("scr_class", "") for r in held_records if "scr_class" in r})))
    pairs, _ = candidate_aliases([r for r in held_records if "scr_class" not in r],
                                 [r for r in held_records if "scr_class" in r], policy)
    names = sorted({" ".join(a.split()) for a, _ in pairs} | set(linked))
    return names, {"linked_aliases": len(linked), "candidate_names": len({a for a, _ in pairs}), "names": len(names),
                   "records": len(held_records)}


def mesh_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every MeSH record of the track's files (descriptors, then SCRs; `mesh_novel.load_mesh_records`)."""
    from vsa_embed.ontologies.mesh_novel import load_mesh_records
    spec = config["ontology"]
    descriptors, scrs = load_mesh_records(Path(spec["path"]).expanduser(), Path(spec["supplementary_path"]).expanduser())
    return [*descriptors, *scrs]


# -- the training stream without the held-out documents ----------------------------------------------------------------

@dataclass
class ExcludingDocuments(TrackDocuments):
    """`TrackDocuments` whose training side drops every document `matcher` flags (PubMed and general alike), refills the
    domain side from abstracts that mention no selected name (`top_up: domain`) and ends after `train_documents`
    documents. Evaluation streams are unchanged."""

    matcher: Any = None
    top_up: str = "domain"
    train_documents: int | None = None
    tally: Counter = field(default_factory=Counter, compare=False, repr=False)   # documents read while streaming
    dropped: list | None = field(default=None, compare=False, repr=False)        # (source, text) of dropped documents

    def signature(self, stream: str) -> str:
        base = super().signature(stream)
        if stream != "train":
            return base
        spec = {"base": base, "exclude": self.matcher.digest(), "top_up": self.top_up, "train_documents": self.train_documents}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def _drop(self, texts: Iterator[str], source: str) -> Iterator[str]:
        for text in texts:
            if self.matcher(text):
                self.tally[f"{source}_dropped"] += 1
                self.tally[f"{source}_dropped_chars"] += len(text)
                if self.dropped is not None:
                    self.dropped.append((source, text))
                continue
            self.tally[f"{source}_kept"] += 1
            yield text

    def refill_pubmed(self) -> Iterator[str]:
        """Training-side abstracts of the same files that mention no selected name, in stream order (none without a
        mention filter: then the domain stream is not mention-limited)."""
        if self.top_up == "none" or self.mention_filter is None:
            return iter(())
        unmentioned = lambda text: not self.mention_filter.mentions(text)  # noqa: E731
        return pubmed_documents(self.pubmed_paths, eval_buckets=self.eval_buckets, split="train", exclude=self.exclude_pmids,
                                keep=unmentioned, min_pmid=self.min_pmid)

    def train_pubmed(self) -> Iterator[str]:
        return chain(self._drop(super().train_pubmed(), "pubmed"), self._drop(self.refill_pubmed(), "refill"))

    def train_general(self) -> Iterator[str]:
        return self._drop(super().train_general(), "general")

    def train(self, log: list[int] | None = None) -> Iterator[str]:
        stream = super().train(log)
        return islice(stream, self.train_documents) if self.train_documents is not None else stream

    def base(self) -> TrackDocuments:
        """The same streams without the exclusion."""
        return TrackDocuments(**{f.name: getattr(self, f.name) for f in fields(TrackDocuments)})

    def flagged_pubmed(self) -> Iterator[str]:
        """Every training-side mention-bearing abstract the matcher flags (E13: the round-2 domain text), in stream order."""
        return (text for text in self.base().train_pubmed() if self.matcher(text))


def excluding(documents: TrackDocuments, matcher: NameMatcher, top_up: str) -> ExcludingDocuments:
    return ExcludingDocuments(**{f.name: getattr(documents, f.name) for f in fields(TrackDocuments)}, matcher=matcher,
                              top_up=top_up)


def budget_pass(documents: ExcludingDocuments, tokenizer: Any, target: int | None) -> dict[str, Any]:
    """One pass over the training stream, no corpus written: the fewest documents whose `tokenizer` tokens (+ one EOS
    each) reach `target` (as `build_corpus` cuts at `max_tokens`: the document that crosses the budget is kept; the whole
    stream without a target), each document's source (0 PubMed, 1 general), the documents read, dropped and refilled
    per source, and the tokens of the dropped documents. Documents are tokenized one at a time, so the pass stops
    exactly at the budget and the tally is that of the stream the corpora read."""
    probe = replace(documents, train_documents=None, tally=Counter(), dropped=[])
    count = lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 1  # noqa: E731
    log: list[int] = []
    tokens = documents_read = 0
    for text in probe.train(log):
        tokens += count(text)
        documents_read += 1
        if target is not None and tokens >= target:
            break
    dropped_tokens: Counter = Counter()
    for source, text in probe.dropped:
        dropped_tokens[source] += count(text)
    return {"train_documents": documents_read, "tokens": tokens, "target_tokens": target, "log": log,
            "reached_target": target is None or tokens >= target, "tally": dict(probe.tally),
            "dropped_tokens": dict(dropped_tokens), "raw_general_read": probe.tally["general_kept"] + probe.tally["general_dropped"]}


# -- the date split (E13 rounds) ---------------------------------------------------------------------------------------

def record_dates(config: dict[str, Any], ontology: Any, records: Sequence[dict[str, Any]] | None = None
                 ) -> dict[int, tuple[str, str]]:
    """concept → (date "YYYY-MM-DD", source): the SCR's `DateIntroduced` (a descriptor's year introduced as YYYY-01-01);
    records without one get the publication year of their first training-side PubMed mention (`first_mention_dates`)."""
    records = records if records is not None else mesh_records(config)
    by_ui = {r["ui"]: r for r in records}
    out: dict[int, tuple[str, str]] = {}
    missing: dict[int, list[str]] = {}
    for concept, ui in enumerate(ontology.concept_names):
        record = by_ui.get(ui) or {}
        date = record.get("introduced_date") or (f"{record['introduced']:04d}-01-01" if record.get("introduced") else None)
        if date:
            out[concept] = (date, "DateIntroduced")
        else:
            missing[concept] = [a for a, c in ontology.alias_pairs if c == concept]
    if missing:
        from .t1_open_corpus import pubmed_paths, pubmedqa_pmids
        found = first_mention_dates(pubmed_paths(config), missing, eval_buckets=int(config["pubmed"]["eval_buckets"]),
                                    exclude=pubmedqa_pmids(config), min_pmid=config["pubmed"].get("min_pmid"))
        for concept in missing:
            if concept not in found:
                raise ValueError(f"{ontology.concept_names[concept]}: no MeSH date and no training-side mention")
            out[concept] = (found[concept], "first PubMed mention")
    return out


def first_mention_dates(paths: Sequence[Path], names: dict[int, list[str]], *, eval_buckets: int,
                        exclude: frozenset[int] = frozenset(), min_pmid: int | None = None) -> dict[int, str]:
    """concept → "YYYY-12-31" for the publication year of its earliest training-side PubMed mention (smallest PMID; the
    last day of that year, the latest date the mention vouches for)."""
    owners: dict[str, set[int]] = {}
    for concept, aliases in names.items():
        for alias in aliases:
            owners.setdefault(mention_key(alias), set()).add(concept)
    counter = MentionCounter(owners)
    first: dict[int, tuple[int, int]] = {}
    for record in iter_pubmed(list(paths), columns=("pmid", "year", "text")):
        pmid = int(record["pmid"])
        if pmid in exclude or (min_pmid is not None and pmid < int(min_pmid)) or pmid_bucket(pmid) < eval_buckets:
            continue
        for key in counter.count(record["text"]):
            for concept in owners[key]:
                if concept not in first or pmid < first[concept][0]:
                    first[concept] = (pmid, int(record.get("year") or 0))
    return {concept: f"{year:04d}-12-31" for concept, (_, year) in first.items() if year}


def date_holdout(config: dict[str, Any], ontology: Any, table: AliasTable, *,
                 records: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """`choose_track_holdout`'s result for the date split: the entries whose date is later than the cut that leaves
    1 − `round2_fraction` of the entries in round 1 (an entry's date is its latest concept's; ties at the cut stay in
    round 1), then the holdout closure (containment of a held-out alias, shared concepts)."""
    fraction = float(config["holdout"].get("round2_fraction", RoodSettings.round2_fraction))
    dates = record_dates(config, ontology, records)
    entry_date = [max(dates[c][0] for c in concepts) for concepts in table.entry_concepts]
    order = sorted(range(len(entry_date)), key=lambda e: (entry_date[e], e))
    keep = len(order) - int(round(len(order) * fraction))
    cut = entry_date[order[keep - 1]]
    chosen = sorted(e for e, date in enumerate(entry_date) if date > cut)
    held = holdout_closure(set(chosen), table, containment_index(table))
    concepts = sorted({c for e in held for c in table.entry_concepts[e]})
    names = sorted(ontology.concept_names[c] for c in concepts)
    return {"chosen_entries": chosen, "closure_entries": sorted(held - set(chosen)), "heldout_entries": sorted(held),
            "concepts": concepts, "names": names, "sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "eligible_entries": len(order), "excluded_hub_entries": 0,
            "split": {"by": "DateIntroduced", "cut_date": cut, "round2_fraction_requested": fraction,
                      "round2_entries": len(held), "round2_fraction": len(held) / max(1, len(order)),
                      "date_sources": dict(Counter(source for _, source in dates.values())),
                      "entry_dates": entry_date, "concept_dates": {c: d for c, d in dates.items()}}}


# -- the leakage audit -------------------------------------------------------------------------------------------------

def decoded_documents(corpus_dir: Path) -> tuple[Iterator[tuple[str, int]], dict[str, Any]]:
    """((text, tokens incl. EOS) per document in corpus order, info): documents are split at the EOS id and decoded with
    the corpus tokenizer (byte-level BPE is lossless: `build_corpus` dropped every document that did not decode back)."""
    from transformers import AutoTokenizer
    corpus = TokenCorpus.open(corpus_dir)
    tokenizer = AutoTokenizer.from_pretrained(corpus.manifest["tokenizer"], local_files_only=True)
    tokens = np.asarray(corpus.tokens)
    ends = np.flatnonzero(tokens == int(corpus.manifest["eos_id"]))
    info = {"documents": int(corpus.manifest["documents"]), "pieces": int(ends.size), "tokens": int(tokens.size),
            "exact": int(ends.size) == int(corpus.manifest["documents"])}

    def documents() -> Iterator[tuple[str, int]]:
        start = 0
        for end in ends.tolist():
            yield tokenizer.decode(tokens[start:end].tolist()), end + 1 - start
            start = end + 1
    return documents(), info


def _hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=10).digest()


def audit_training(corpus_dir: Path, matcher: NameMatcher, *, reference: Path | None = None,
                   log: Sequence[int] | None = None, kept_mentions: int | None = None) -> dict[str, Any]:
    """Decode every document of a training corpus and match the held-out names again (`leaked_documents` must be 0).
    With the stream's source `log` and its count of kept mention-bearing abstracts: exact tokens of the mention-bearing,
    refill and general documents. With a `reference` training corpus of the same tokenizer: documents (and tokens) both
    hold, only the reference holds (and how many of those the matcher flags) and only this one holds, by text hash."""
    documents, info = decoded_documents(corpus_dir)
    leaked, examples = 0, []
    hashes: dict[bytes, int] = {}
    categories: Counter = Counter()
    pubmed_seen = 0
    for index, (text, length) in enumerate(documents):
        if matcher(text):
            leaked += 1
            if len(examples) < EXAMPLES:
                examples.append({"document": index, "names": matcher.found(text)[:5]})
        hashes[_hash(text)] = length
        if log is not None and info["exact"] and index < len(log):
            if log[index] == 0:
                kind = "mention" if kept_mentions is None or pubmed_seen < kept_mentions else "refill"
                pubmed_seen += 1
            else:
                kind = "general"
            categories[f"{kind}_documents"] += 1
            categories[f"{kind}_tokens"] += length
    out = {"corpus": str(corpus_dir), **info, "names": len(matcher.keys), "names_sha256": matcher.digest(),
           "leaked_documents": leaked, "leak_examples": examples, "composition": dict(categories) or None}
    if reference is not None and (reference / "manifest.json").exists():
        ref_documents, ref_info = decoded_documents(reference)
        shared = Counter()
        ref_only = Counter()
        for text, length in ref_documents:
            key = _hash(text)
            if key in hashes:
                shared["documents"] += 1; shared["tokens"] += length
            else:
                ref_only["documents"] += 1; ref_only["tokens"] += length
                ref_only["flagged"] += int(matcher(text))
        this_only = {"documents": len(hashes) - shared["documents"], "tokens": sum(hashes.values()) - shared["tokens"]}
        out["reference"] = {"corpus": str(reference), "tokens": ref_info["tokens"], "documents": ref_info["pieces"],
                            "shared": dict(shared), "reference_only": dict(ref_only), "this_only": this_only,
                            "duplicate_texts": info["pieces"] - len(hashes)}
    return out


def compare_corpora(root: Path, reference: Path, names: Sequence[str] = EVAL_CORPORA) -> dict[str, Any]:
    """Per corpus: whether `tokens.bin` equals the reference build's byte for byte, its span arrays element for element
    (`spans.npz` is a zip whose member timestamps differ between builds) and its manifest key for key (except the
    `track` label)."""
    out = {}
    for name in names:
        mine, theirs = root / name, reference / name
        if not (theirs / "tokens.bin").exists():
            out[name] = None
            continue
        with np.load(mine / "spans.npz") as a, np.load(theirs / "spans.npz") as b:
            spans = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        manifests = [{k: v for k, v in json.loads((d / "manifest.json").read_text()).items() if k != "track"} for d in (mine, theirs)]
        out[name] = {"tokens.bin": file_digest(mine / "tokens.bin") == file_digest(theirs / "tokens.bin"), "spans": bool(spans),
                     "manifest": manifests[0] == manifests[1]}
    return out


# -- E13 rounds: the round-2 corpora --------------------------------------------------------------------------------------

def round_streams(documents: ExcludingDocuments, raw_general_read: int) -> dict[str, tuple[Callable[[list[int]], Iterator[str]], str]]:
    """(stream factory, signature) of the round corpora: `train-round2` = the flagged training-side abstracts mixed 50/50
    with the general documents after the `raw_general_read` round 1 read; `eval-round2` / `eval-round1` = the evaluation
    abstracts the matcher flags / does not flag."""
    base = documents.base()
    general = replace(base, general_skip=base.general_skip + int(raw_general_read))
    flagged = documents.matcher

    def signature(kind: str, stream: str, extra: dict[str, Any] | None = None) -> str:
        spec = {"round": kind, "base": base.signature(stream), "names": flagged.digest(), **(extra or {})}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    return {
        "train-round2": (lambda log: base.mixed(documents.flagged_pubmed(), general.train_general(), log),
                         signature("train-round2", "train", {"general_skip": general.general_skip})),
        "eval-round2": (lambda log: (t for t in base.eval_pubmed() if flagged(t)), signature("eval-round2", "eval-pubmed")),
        "eval-round1": (lambda log: (t for t in base.eval_pubmed() if not flagged(t)), signature("eval-round1", "eval-pubmed")),
    }


def build_round_corpora(tokenizer_name: str, root: Path, *, documents: ExcludingDocuments, raw_general_read: int,
                        full: AliasTable, ontology: Any, holdout_sha256: str, min_subtokens: int, workers: int,
                        heldout_entries: Sequence[int], criteria: dict[str, Any]) -> dict[str, Any]:
    """The round corpora of one tokenizer (linked with the full table: round-2 names are linked) and
    `ontology-round2.pt` (nothing held out; `round2_entries` lists them; `train_frequency` = rounds 1 + 2, with each
    round's own), and the round-2 feasibility on `eval-round2`."""
    from transformers import AutoTokenizer
    from .host_corpus import tokenizer_normalization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    eos, vocab = tokenizer.eos_token_id, len(tokenizer)
    normalization = tokenizer_normalization(tokenizer)
    extra = {"normalization": normalization} if normalization else {}
    fingerprint = {"tokenizer_sha256": tokenizer_fingerprint(tokenizer)} if normalization else {}
    manifests, shares = {}, {}
    for name, (stream, signature) in round_streams(documents, raw_general_read).items():
        guard_reuse(root / name, full, 10**12, signature)
        log: list[int] = []
        manifests[name] = build_corpus(stream(log), root / name, tokenizer_name=tokenizer_name, table=full, eos_id=eos,
                                       max_tokens=10**12, min_subtokens=1, workers=workers, vocab_size=vocab, reuse=True,
                                       extra_manifest={"track": documents.track_label, "max_tokens_requested": 10**12,
                                                       "stream_signature": signature, "round_corpus": name, **fingerprint},
                                       **extra)
        if name == "train-round2":
            shares[name] = record_shares(root / name, log, eos, documents.sources)
    count = len(full.entry_concepts)
    round1 = train_frequency(root / "train", count, min_subtokens)
    round2 = train_frequency(root / "train-round2", count, min_subtokens)
    onto = channel_ontology(full, ontology, round1 + round2, holdout_sha256)
    onto.update(heldout_entries=[], round2_entries=sorted(int(e) for e in heldout_entries), round="2",
                train_frequency_round1=round1.tolist(), train_frequency_round2=round2.tolist())
    save_if_changed(onto, root / "ontology-round2.pt")
    frequencies = {t: (train_frequency(root / "train", count, t), "measured") for t in (1, 2, 3, 4)}
    feasibility = feasibility_report(root / "eval-round2", heldout_entries=list(heldout_entries), entry_count=count,
                                     frequencies=frequencies, criteria=criteria)
    held = np.zeros(count, dtype=bool)
    held[list(heldout_entries)] = True
    return {"tokenizer": tokenizer_name, "corpora": manifests, "source_shares": shares,
            "round2_entries_in_train_round2": int(((round2 > 0) & held).sum()),
            "round2_spans_in_train_round2": int(round2[held].sum()), "feasibility_eval_round2": feasibility}


def write_round_records(output_dir: Path, *, ontology: Any, full: AliasTable, holdout: dict[str, Any],
                        records: Sequence[dict[str, Any]], channel: dict[str, Any], occurrences: dict[str, Counter]) -> dict[str, Any]:
    """`round_split.tsv` (every entry: UI, name, date, round, reason) and `round2_records.jsonl` (every round-2 entry: its
    linked aliases, gold frame, SCR note and the definition read from it, occurrences in the round corpora)."""
    from .e11_read_to_learn import scr_definition
    by_ui = {r["ui"]: r for r in records}
    split = holdout["split"]
    chosen, closure = set(holdout["chosen_entries"]), set(holdout["closure_entries"])
    aliases: dict[int, list[str]] = {}
    for alias, entry in full.alias_to_entry.items():
        aliases.setdefault(entry, []).append(alias)
    rows = ["entry\tui\tname\tintroduced\tdate_source\tround\treason"]
    for entry, concepts in enumerate(full.entry_concepts):
        ui = ontology.concept_names[concepts[0]]
        date, source = split["concept_dates"][concepts[0]]
        reason = "date" if entry in chosen else "closure" if entry in closure else ""
        rows.append(f"{entry}\t{ui}\t{by_ui.get(ui, {}).get('name', '')}\t{date}\t{source}\t{2 if reason else 1}\t{reason}")
    (output_dir / "round_split.tsv").write_text("\n".join(rows) + "\n")
    relations, atoms = channel["relation_names"], channel["atomic_names"]
    offsets, rel, fil = (np.asarray(channel[k]) for k in ("offsets", "relations", "fillers"))
    fillers = ontology.metadata.get("filler_headings", {})
    with (output_dir / "round2_records.jsonl").open("w") as handle:
        for entry in holdout["heldout_entries"]:
            concept = full.entry_concepts[entry][0]
            ui = ontology.concept_names[concept]
            record = by_ui.get(ui, {})
            lo, hi = int(offsets[entry]), int(offsets[entry + 1])
            frame = [[relations[r], atoms[a], fillers.get(atoms[a].partition(":")[2], atoms[a].partition(":")[2])]
                     for r, a in zip(rel[lo:hi].tolist(), fil[lo:hi].tolist())]
            handle.write(json.dumps({
                "entry": entry, "ui": ui, "name": record.get("name"), "introduced": split["concept_dates"][concept][0],
                "reason": "date" if entry in chosen else "closure", "aliases": sorted(aliases.get(entry, [])), "frame": frame,
                "note": record.get("note") or None, "definition": scr_definition(record.get("note") or ""),
                **{f"occurrences_{name}": int(counts.get(entry, 0)) for name, counts in occurrences.items()}}) + "\n")
    return {"round_split": str(output_dir / "round_split.tsv"), "round2_records": str(output_dir / "round2_records.jsonl"),
            "sha256": {name: file_digest(output_dir / name) for name in ("round_split.tsv", "round2_records.jsonl")}}


def span_counts(corpus_dir: Path, entries: Sequence[int], min_subtokens: int) -> Counter:
    corpus = TokenCorpus.open(corpus_dir)
    keep = (corpus.spans["length"] >= min_subtokens) & np.isin(corpus.spans["entry"], np.asarray(list(entries)))
    return Counter(int(e) for e in corpus.spans["entry"][keep])


def build_e11_items(output_dir: Path, *, root: Path, full: AliasTable, min_subtokens: int) -> dict[str, Any]:
    """The round-2 records as an E11 held-out set (`e11_read_to_learn.build_heldout_set` on the round-1 ontology, whose
    held-out entries are round 2): SCR notes read as '<headword>: <note>', gold and random frames, occurrences in
    `eval-round2`. Written to `output_dir`."""
    import torch
    from transformers import AutoTokenizer
    from . import e9_tracks as tracks
    from .e11_read_to_learn import build_heldout_set, make_context
    onto = torch.load(root / "ontology.pt", weights_only=False)
    ctx = make_context("t7", onto, full, tracks.lexicon_for(tracks.track_spec("t7"), onto))
    tokenizer = AutoTokenizer.from_pretrained(tracks.FAMILY_TOKENIZERS["smollm2"], local_files_only=True)
    return build_heldout_set("t7", output_dir, ctx, eval_corpus=root / "eval-round2", min_subtokens=min_subtokens,
                             tokenizer=tokenizer)


# -- the build hooks ------------------------------------------------------------------------------------------------------

@dataclass
class Rood:
    """State `t1_open_corpus.run` carries between `prepare` (before the corpora) and `finish` (after them)."""

    config: dict[str, Any]
    settings: RoodSettings
    matcher: NameMatcher
    name_counts: dict[str, int]
    documents: ExcludingDocuments
    budget: dict[str, Any]
    holdout: dict[str, Any]
    records: list[dict[str, Any]] | None = None

    @property
    def data_root(self) -> Path:
        return Path(self.config["paths"]["data_root"]).expanduser()

    def reference_for(self, root: Path) -> Path | None:
        if not self.settings.reference_root:
            return None
        return Path(self.settings.reference_root).expanduser() / root.relative_to(self.data_root)

    def finish(self, builds: Sequence[tuple[str, Path, dict[str, Any], int]], *, output_dir: Path, full: AliasTable,
               ontology: Any, holdout_sha256: str, min_subtokens: int, workers: int, **_: Any) -> dict[str, Any]:
        """Audit every training corpus (fails on a leak), compare with the reference build, write the sidecars and, for
        the date split, build the round corpora, records and E11 items. Returns the summary's `rood` section. The
        keyword arguments are `t1_open_corpus.run`'s shared relink settings."""
        tally = self.budget["tally"]
        audits, comparisons = {}, {}
        for label, root, _, _ in builds:
            reference = self.reference_for(root)
            audits[label] = audit_training(root / "train", self.matcher, reference=(reference / "train") if reference else None,
                                           log=self.budget["log"], kept_mentions=tally.get("pubmed_kept"))
            if reference is not None:
                comparisons[label] = compare_corpora(root, reference)
            sidecar = {"train_documents": self.budget["train_documents"], "budget_tokens": self.budget["tokens"],
                       "target_tokens": self.budget["target_tokens"], "tally": tally, "dropped_tokens": self.budget["dropped_tokens"],
                       "names_sha256": self.matcher.digest(), "audit": {k: v for k, v in audits[label].items() if k != "leak_examples"}}
            (root / "train" / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n")
        leaks = {label: a["leaked_documents"] for label, a in audits.items() if a["leaked_documents"]}
        summary: dict[str, Any] = {
            "settings": {f.name: getattr(self.settings, f.name) for f in fields(self.settings)},
            "names": {**self.name_counts, "matcher_keys": len(self.matcher.keys)}, "names_sha256": self.matcher.digest(),
            "matcher": "mesh_novel.MentionCounter: case-insensitive whole alphanumeric tokens (word boundaries)",
            "stream": {k: v for k, v in self.budget.items() if k != "log"},
            "audit": audits, "reference_comparison": comparisons or None,
            "leakage": {"training_documents_with_a_heldout_name": sum(a["leaked_documents"] for a in audits.values()),
                        "passed": not leaks},
        }
        if leaks:
            (output_dir / "rood_audit_failed.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
            raise AssertionError(f"held-out names in training documents: {leaks}")
        if self.settings.split == "date":
            summary["rounds"] = self.finish_rounds(builds, output_dir=output_dir, full=full, ontology=ontology,
                                                   holdout_sha256=holdout_sha256, min_subtokens=min_subtokens, workers=workers)
        return summary

    def finish_rounds(self, builds: Sequence[tuple[str, Path, dict[str, Any], int]], *, output_dir: Path, full: AliasTable,
                      ontology: Any, holdout_sha256: str, min_subtokens: int, workers: int) -> dict[str, Any]:
        import torch
        held = self.holdout["heldout_entries"]
        per_tokenizer = {}
        for label, root, _, _ in builds:
            per_tokenizer[label] = build_round_corpora(
                label, root, documents=self.documents, raw_general_read=self.budget["raw_general_read"], full=full,
                ontology=ontology, holdout_sha256=holdout_sha256, min_subtokens=min_subtokens, workers=workers,
                heldout_entries=held, criteria=self.config["feasibility"])
        reference_root = builds[0][1]
        occurrences = {name: span_counts(reference_root / name, held, min_subtokens) for name in ("train-round2", "eval-round2")}
        channel = torch.load(reference_root / "ontology.pt", weights_only=False)
        files = write_round_records(output_dir, ontology=ontology, full=full, holdout=self.holdout,
                                    records=self.records or mesh_records(self.config), channel=channel, occurrences=occurrences)
        items = build_e11_items(output_dir / "round2-items-smollm2-v1", root=reference_root, full=full,
                                min_subtokens=min_subtokens)
        split = {k: v for k, v in self.holdout["split"].items() if k not in ("entry_dates", "concept_dates")}
        rounds = Counter(2 if e in set(held) else 1 for e in range(len(full.entry_concepts)))
        return {"split": split, "entries": {"round1": rounds[1], "round2": rounds[2],
                                            "round2_by_date": len(self.holdout["chosen_entries"]),
                                            "round2_by_closure": len(self.holdout["closure_entries"])},
                "corpora": per_tokenizer, "files": files,
                "e11_items": {k: items.get(k) for k in ("concepts", "definitions", "terms_with_eval_occurrences",
                                                        "eval_occurrences", "headwords", "filters")}}

    def report(self, summary: dict[str, Any]) -> str:
        """The report section of the exclusion (appended to the build's `report.md`)."""
        s, budget = summary["settings"], summary["stream"]
        tally = budget["tally"]
        lines = ["", "## Held-out documents excluded from training (T7-ROOD, decision 63 H1)", "",
                 f"Split `{s['split']}`; names: `{s['names']}` ({summary['names'].get('candidate_names', 0):,} candidate names "
                 f"and {summary['names'].get('linked_aliases', 0):,} linked aliases of {summary['names'].get('records', 0):,} records: "
                 f"{summary['names'].get('matcher_keys', 0):,} matcher keys, sha256 `{summary['names_sha256'][:16]}…`). "
                 f"Matcher: {summary['matcher']}.", "",
                 f"Training stream: {budget['train_documents']:,} documents, {budget['tokens']:,} reference tokens "
                 f"(target {budget['target_tokens'] or 'none'}; reached: {budget['reached_target']}). "
                 f"Dropped: {tally.get('pubmed_dropped', 0):,} mention-bearing abstracts "
                 f"({budget['dropped_tokens'].get('pubmed', 0):,} tokens), {tally.get('general_dropped', 0):,} general "
                 f"documents, {tally.get('refill_dropped', 0):,} refill abstracts. Refill: {tally.get('refill_kept', 0):,} "
                 f"abstracts without a selected name (`top_up: {s['top_up']}`).", "",
                 "| tokenizer | documents | tokens | leaked documents | mention-bearing tokens | refill tokens | general tokens | "
                 "shared with reference (docs) | reference only (docs, flagged) | this only (docs) |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for label, audit in summary["audit"].items():
            comp = audit.get("composition") or {}
            ref = audit.get("reference") or {}
            only = ref.get("reference_only", {})
            lines.append(f"| {label} | {audit['pieces']:,} | {audit['tokens']:,} | {audit['leaked_documents']} | "
                         f"{comp.get('mention_tokens', 0):,} | {comp.get('refill_tokens', 0):,} | {comp.get('general_tokens', 0):,} | "
                         f"{ref.get('shared', {}).get('documents', 0):,} | {only.get('documents', 0):,} ({only.get('flagged', 0):,}) | "
                         f"{ref.get('this_only', {}).get('documents', 0):,} |")
        if summary.get("reference_comparison"):
            lines += ["", "Evaluation corpora against the reference build (identical `tokens.bin` bytes, span arrays and manifests): "
                      + "; ".join(f"{label}: " + ", ".join(f"{name} {'yes' if v and all(v.values()) else 'no'}" for name, v in comp.items())
                                  for label, comp in summary["reference_comparison"].items()) + "."]
        lines += ["", f"**Leakage audit:** {summary['leakage']['training_documents_with_a_heldout_name']} training documents "
                  "contain a held-out name (every written document decoded and matched again)."]
        rounds = summary.get("rounds")
        if rounds:
            split, entries = rounds["split"], rounds["entries"]
            lines += ["", "## E13 rounds", "",
                      f"Round 2 = entries introduced after {split['cut_date']} (MeSH DateIntroduced; {split['date_sources']}): "
                      f"{entries['round2']:,} entries ({entries['round2_by_date']:,} by date, {entries['round2_by_closure']:,} by the "
                      f"closure), round 1 {entries['round1']:,}.", "",
                      "| tokenizer | corpus | tokens | documents | spans | PubMed share |", "|---|---|---:|---:|---:|---:|"]
            for label, info in rounds["corpora"].items():
                for name, manifest in info["corpora"].items():
                    share = (info["source_shares"].get(name) or {}).get("share", {}).get("pubmed")
                    lines.append(f"| {label} | {name} | {manifest['tokens']:,} | {manifest['documents']:,} | {manifest['spans']:,} | "
                                 f"{'—' if share is None else f'{share:.3f}'} |")
            lines += ["", "Round-2 feasibility on `eval-round2` (ℓ_min 2): " + "; ".join(
                f"{label}: {row['split']['heldout_occurrences']:,} occurrences, {row['split']['heldout_entries_5plus']:,} entries ≥ 5, "
                f"{row['verdict']}, windows for the whole split {row['windows_for_whole_split']:,}"
                for label, info in rounds["corpora"].items() for row in info["feasibility_eval_round2"] if row["min_subtokens"] == 2) + "."]
        return "\n".join(lines) + "\n"


def prepare(config: dict[str, Any], *, documents: TrackDocuments, full: AliasTable, ontology: Any, holdout: dict[str, Any],
            tokenizer: Any) -> Rood:
    """Before the corpora: the held-out names, the excluding stream and its document budget (one pass)."""
    settings = RoodSettings.from_config(config)
    if settings is None:
        raise ValueError("the holdout: section needs exclude_training_documents: true")
    if (settings.split == "date") != ("split" in holdout):
        raise AssertionError("the holdout was not chosen by the configured split")
    records =mesh_records(config) if settings.names == "candidates" or settings.split == "date" else None
    names, counts = heldout_names(config, ontology, full, settings, records)
    matcher = NameMatcher(names)
    stream = excluding(documents, matcher, settings.top_up)
    budget = budget_pass(stream, tokenizer, settings.match_train_tokens)
    if not budget["reached_target"]:
        raise ValueError(f"the excluding stream ends at {budget['tokens']:,} tokens, below the target {settings.match_train_tokens:,}")
    stream = replace(stream, train_documents=budget["train_documents"], tally=Counter(), dropped=None)
    return Rood(config, settings, matcher, counts, stream, budget, holdout, records)
