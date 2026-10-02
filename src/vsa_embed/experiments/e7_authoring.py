"""E7 self-authored ontologies (WP-E7): reading corpora and hidden gold (D7.0), authoring quality (D7.1).

    python -m vsa_embed.experiments.e7_authoring <command> --run <run folder> [options]

D7.0 / D7.1 commands (this module):

- `prepare --track general|devtools|clinical` — reading corpora and the hidden gold. General: 20% of the
  WordNet concepts linked in the SmolLM2 host corpus are masked (seed-fixed, stratified by frequency,
  names hashed); their aliases are removed from every linker the hosts see, their frames are kept only
  in `<data>/gold/` (read by `quality`, the gold → host condition and the report, never by discovery,
  authoring or acceptance). Corpora (FineWeb-Edu documents in C3's shard order, SmolLM2 tokenizer):
  `general` (base-channel training) and `replay` from the SmolLM2 host-corpus documents, `read` (D_read,
  25M tokens) and `test` (held-out contexts) from the documents after the GPT-2 C3 training corpus, so
  they are disjoint from every C3/host training corpus. Dev-tools: the C6 v1 documentation corpus with
  20% of its symbols masked (schemas are the gold). Clinical: the hook for T1-open (WP-T1).
- `discover --host H` — candidates by excess surprisal under one host (GPU).
- `link` — the authoring set (top candidates of the consumer host, SmolLM2-360M), their matches in the
  corpora, and per candidate disjoint authoring and validation documents.
- `author --author A` — `SmolLM2-135M`, `SmolLM2-360M`, `Qwen2.5-0.5B` (local generation, constrained
  few-shot template), `teacher` (Claude Code headless, cached), `hearst`, `random` (matched degree),
  `direct-<host>` (OLLM/LLMs4OL-style prompting without the candidate screen).
- `quality` — edge precision/recall/F1 and relation-type accuracy against the hidden gold, discovery
  recall per host (Wilson intervals).
- `judge-items` / `judge` — blinded items for the LLM-judged plausibility of edges without gold
  (experiments §0.12: ≥ 3 calls over 2 paraphrases, Fleiss κ, calibration items).

The round (D7.2), cross-authoring (D7.3), plan and report commands are dispatched to `e7_round`,
`e7_plan` and `e7_report`. Every step writes JSON next to the run folder's `track.json`; large
artifacts go to the data root recorded there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import yaml

from vsa_embed.authoring import (
    IDENTIFIER, STOPWORDS, WORD, ComputeLedger, authoring_prompt_fewshot, cut_completion, discover_candidates, excerpt,
    pool_proposals, self_consistent, split_contexts, stable_seed,
)
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.span_channel import normalize_alias
from vsa_embed.statistics import wilson_interval

ROOT = Path("experiments/e7-self-authoring")
DATA_ROOT = Path("~/data/vsa-llm/e7")
HOSTS = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
         "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B"}
CONSUMER = "SmolLM2-360M"

# Closed relation vocabularies (formulation §5.5): authoring label, ontology relation, description.
WORDNET_RELATIONS = [
    ("is_a", "hypernym", "a more general kind of thing it is"),
    ("instance_of", "instance_hypernym", "the class a named thing is an instance of"),
    ("has_part", "part_meronym", "a part it has"),
    ("part_of", "part_holonym", "a whole it is part of"),
    ("has_member", "member_meronym", "a member of this group"),
    ("member_of", "member_holonym", "a group it belongs to"),
    ("made_of", "substance_meronym", "a substance it is made of"),
    ("substance_of", "substance_holonym", "something it is a substance of"),
    ("attribute", "attribute", "a property it has"),
    ("similar_to", "similar_to", "a similar concept"),
    ("domain", "topic_domain", "the field or topic it belongs to"),
    ("entails", "entailment", "an action it implies"),
    ("causes", "cause", "something it causes"),
    ("opposite_of", "antonym", "its opposite"),
]
DEVTOOLS_RELATIONS = [
    ("kind", "kind", "function, class, method, exception or constant"),
    ("belongs_to", "belongs_to", "the module or class it belongs to"),
    ("returns", "returns", "the type it returns"),
    ("takes", "takes", "a parameter type it takes"),
    ("raises", "raises", "an exception it raises"),
    ("calls", "calls", "another function it calls"),
    ("inherits", "inherits", "the class it inherits from"),
    ("category", "category", "the kind of work it is used for"),
]
HEARST_MAPS = {"general": {"is_a": "is_a", "part_of": "part_of", "has_part": "has_part", "member_of": "member_of",
                           "made_of": "made_of", "opposite_of": "opposite_of"},
               "devtools": {"is_a": "kind", "part_of": "belongs_to", "member_of": "belongs_to"}}
# Hand-written demonstrations; `prepare` keeps the first two whose surface is not a masked alias.
WORDNET_DEMOS = [
    {"surface": "golden retriever", "context": "Our golden retriever loves to swim and fetch sticks in the lake.",
     "edges": [("is_a", "dog")]},
    {"surface": "carburetor", "context": "The mechanic cleaned the carburetor so that the engine got the right mix of air and fuel.",
     "edges": [("is_a", "device"), ("part_of", "engine")]},
    {"surface": "granite", "context": "The old bridge was built from blocks of grey granite cut in a nearby quarry.",
     "edges": [("is_a", "rock")]},
    {"surface": "violinist", "context": "The violinist tuned her instrument before the orchestra began to play.",
     "edges": [("is_a", "musician"), ("member_of", "orchestra")]},
    {"surface": "photosynthesis", "context": "Through photosynthesis, plants turn sunlight, water and carbon dioxide into sugar.",
     "edges": [("is_a", "process"), ("domain", "biology")]},
]
DEVTOOLS_DEMOS = [
    {"surface": "plorv_quandle", "context": "API reference. `plorv_quandle` is a function in the `zestin` module for parsing work. "
                                            "It takes str and returns dict. It raises KrumbleError on invalid input.",
     "edges": [("kind", "function"), ("belongs_to", "zestin"), ("returns", "dict"), ("takes", "str"), ("raises", "krumbleerror")]},
    {"surface": "VashMorrel", "context": "`VashMorrel` is a class in the `quindo` module used for caching.",
     "edges": [("kind", "class"), ("belongs_to", "quindo"), ("category", "caching")]},
]
DEFAULTS: dict[str, Any] = {
    "seed": 20261002,
    "mask": {"fraction": 0.2, "min_count": 1},
    "corpora": {"general_tokens": 60_000_000, "replay_tokens": 6_250_000, "read_tokens": 25_000_000,
                "test_tokens": 4_000_000, "min_subtokens": 2, "workers": 4},
    "discovery": {"min_count": 8, "max_words": 3, "max_occurrences": 32, "keep": 5000, "batch": 8},
    "authoring": {"candidates": 1000, "contexts": 4, "samples": 3, "temperature": 0.5, "top_p": 0.95,
                  "max_new_tokens": 48, "min_share": 0.3, "batch": 16, "max_validation": 24, "context_chars": 400},
    "direct": {"documents": 1500, "passage_chars": 600, "max_new_tokens": 96, "batch": 16},
    "teacher": {"model": "claude-opus-5-5", "batch": 8},
}


# ---------------------------------------------------------------------------------------------
# helpers

def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=_default) + "\n")


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_track(run: Path) -> dict[str, Any]:
    return json.loads((Path(run) / "track.json").read_text())


def data_root(track: dict[str, Any]) -> Path:
    return Path(track["data_root"]).expanduser()


def visible(track: dict[str, Any]) -> dict[str, Any]:
    """What authoring and acceptance may see: base aliases → entries, excluded aliases, lexicon, ontology names."""
    return json.loads((data_root(track) / "visible" / "visible.json").read_text())


def gold(track: dict[str, Any]) -> dict[str, Any]:
    """The hidden gold-audit subgraph (masked entries, their aliases and frames). Audit-only."""
    return json.loads((data_root(track) / "gold" / "gold.json").read_text())


def iter_documents(shards: Sequence[str], start: int, *, column: str = "text") -> Iterator[str]:
    """Documents from global index `start` on, in the shard order, skipping whole row groups."""
    import pyarrow.parquet as pq
    index = 0
    for path in shards:
        parquet = pq.ParquetFile(str(Path(path).expanduser()))
        for group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            if index + rows <= start:
                index += rows
                continue
            for text in parquet.read_row_group(group, columns=[column]).column(0).to_pylist():
                if index >= start:
                    yield text
                index += 1


def occurrence_regex(surface: str, *, identifier: bool = False) -> re.Pattern:
    words = [re.escape(w) for w in surface.split()]
    body = r"[\s_]+".join(words) if not identifier else re.escape(surface)
    return re.compile(rf"(?<![\w]){body}(?![\w])", re.I)


class FillerResolver:
    """Filler string → dictionary atom: exact lexicon match, else a WordNet morphological base form
    (plural → singular, also of the head word of a phrase). `candidates` lists every atom sense."""

    def __init__(self, lexicon: dict[str, list[int]], *, morphology: bool = True, normalize_key: Callable[[str], str] = normalize_alias) -> None:
        self.lexicon, self.morphology, self.normalize_key = lexicon, morphology, normalize_key
        self._morphy = None
        if morphology:
            try:
                from nltk.corpus import wordnet as wn
                wn.ensure_loaded()
                self._morphy = wn.morphy
            except (LookupError, ImportError):
                self._morphy = None

    def variants(self, filler: str) -> list[str]:
        key = self.normalize_key(filler)
        out = [key]
        if self._morphy is not None and key:
            words = key.split()
            whole = self._morphy(key.replace(" ", "_"), "n")
            if whole:
                out.append(normalize_alias(whole))
            head = self._morphy(words[-1], "n")
            if head:
                out.append(" ".join(words[:-1] + [head]))
        return list(dict.fromkeys(v for v in out if v))

    def candidates(self, filler: str) -> list[int]:
        for variant in self.variants(filler):
            if variant in self.lexicon:
                return list(self.lexicon[variant])
        return []

    def __call__(self, filler: str) -> int | None:
        found = self.candidates(filler)
        return found[0] if found else None


def resolver_for(track: dict[str, Any], view: dict[str, Any]) -> FillerResolver:
    if track["kind"] == "devtools":
        return FillerResolver(view["lexicon"], morphology=False, normalize_key=lambda s: s.strip(" `").lower())
    return FillerResolver(view["lexicon"])


def atom_surface(name: str) -> str:
    """A readable filler for a dictionary atom: `synset:golden_retriever.n.01` → `golden retriever`."""
    kind, _, rest = name.partition(":")
    if kind == "synset":
        return rest.rsplit(".", 2)[0].replace("_", " ")
    return (rest or kind).lower()


def relation_labels(track: dict[str, Any]) -> list[str]:
    return [label for label, _, _ in track["relations"]]


def relation_descriptions(track: dict[str, Any]) -> dict[str, str]:
    return {label: description for label, _, description in track["relations"]}


# ---------------------------------------------------------------------------------------------
# D7.0 prepare

def choose_masked(frequency: np.ndarray, entry_concepts: Sequence[Sequence[int]], excluded_entries: set[int], *,
                  fraction: float, min_count: int, seed: int) -> dict[str, Any]:
    """`fraction` of the eligible entries (training frequency ≥ `min_count`, not excluded), stratified by
    log2 frequency; every entry sharing a concept with a chosen one is masked too (an alias that can
    mean a masked concept would expose it)."""
    eligible = [e for e in range(len(frequency)) if frequency[e] >= min_count and e not in excluded_entries]
    rng = np.random.default_rng(seed)
    by_bin: dict[int, list[int]] = defaultdict(list)
    for entry in eligible:
        by_bin[int(math.log2(max(1, frequency[entry])))].append(entry)
    chosen: list[int] = []
    for _, entries in sorted(by_bin.items()):
        k = int(round(len(entries) * fraction))
        if k:
            chosen += sorted(rng.choice(entries, size=min(k, len(entries)), replace=False).tolist())
    concepts = sorted({c for e in chosen for c in entry_concepts[e]})
    concept_set = set(concepts)
    masked = sorted(e for e, cs in enumerate(entry_concepts) if set(cs) & concept_set and e not in excluded_entries)
    return {"chosen_entries": sorted(chosen), "concepts": concepts, "masked_entries": masked, "eligible_entries": len(eligible)}


def wordnet_lexicon(atomic_names: Sequence[str]) -> dict[str, list[int]]:
    from nltk.corpus import wordnet as wn
    lexicon: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(atomic_names):
        if name.startswith("synset:"):
            for lemma in wn.synset(name[len("synset:"):]).lemma_names():
                key = normalize_alias(lemma)
                if index not in lexicon[key]:
                    lexicon[key].append(index)
    return dict(lexicon)


def _frames_of(offsets: Sequence[int], relations: Sequence[int], fillers: Sequence[int], entry: int) -> list[tuple[int, int]]:
    return list(zip(relations[offsets[entry]:offsets[entry + 1]], fillers[offsets[entry]:offsets[entry + 1]]))


def _demos(pool: list[dict[str, Any]], hidden: set[str], count: int = 2) -> list[dict[str, Any]]:
    chosen = [d for d in pool if normalize_alias(d["surface"]) not in hidden and d["surface"].lower() not in hidden][:count]
    if len(chosen) < count:
        raise ValueError("not enough demonstrations outside the masked set")
    return chosen


def _build_split(name: str, texts: Iterator[str], out: Path, *, tokenizer: str, strings: list[str], eos: int, tokens: int,
                 settings: dict[str, Any], keep_texts: bool, first_document: int, vocab_size: int) -> dict[str, Any]:
    from vsa_embed.data.match_corpus import build_match_corpus
    manifest = build_match_corpus(texts, out, tokenizer_name=tokenizer, strings=strings, eos_id=eos, max_tokens=tokens,
                                  min_subtokens=int(settings["min_subtokens"]), vocab_size=vocab_size,
                                  workers=int(settings["workers"]), keep_texts=keep_texts,
                                  extra_manifest={"split": name, "first_document": first_document})
    manifest["last_document"] = first_document + manifest["documents"] + manifest["skipped_documents"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_general(config: dict[str, Any], run: Path) -> dict[str, Any]:
    """D7.0 general track (see the module docstring)."""
    from transformers import AutoTokenizer
    from vsa_embed.experiments import host_corpus
    c3_run = Path(config["c3_run"])
    record = host_corpus.load_c3_record(c3_run)
    c3 = record["config"]
    tables = host_corpus.rebuild_c3_tables(c3, record["holdout_names"])
    full, ontology = tables["full"], tables["ontology"]
    host = torch.load(Path(config["host_root"]).expanduser() / "ontology.pt", weights_only=False)
    if host["alias_table_sha256"] != full.digest():
        raise ValueError("the rebuilt C3 alias table differs from the host corpus's")
    heldout = set(int(e) for e in host["heldout_entries"])
    frequency = np.asarray(host["train_frequency"])
    mask = choose_masked(frequency, full.entry_concepts, heldout, fraction=float(config["mask"]["fraction"]),
                         min_count=int(config["mask"]["min_count"]), seed=int(config["seed"]))
    masked = set(mask["masked_entries"])
    names = sorted(ontology.concept_names[c] for e in mask["masked_entries"] for c in full.entry_concepts[e])
    names = sorted(set(names))
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    (run / "masked_concepts.txt").write_text("\n".join(names) + "\n")
    base = {a: e for a, e in full.alias_to_entry.items() if e not in heldout and e not in masked}
    masked_aliases = {a: e for a, e in full.alias_to_entry.items() if e in masked}
    excluded = sorted(a for a, e in full.alias_to_entry.items() if e in heldout)
    strings = sorted(base) + sorted(masked_aliases)
    offsets, relations, fillers = (host["offsets"].tolist(), host["relations"].tolist(), host["fillers"].tolist())
    root = Path(config["data_root"]).expanduser()
    (root / "visible").mkdir(parents=True, exist_ok=True); (root / "gold").mkdir(parents=True, exist_ok=True)
    hidden_strings = set(masked_aliases)
    demos = _demos(WORDNET_DEMOS, hidden_strings)
    lexicon = wordnet_lexicon(host["atomic_names"])
    visible_frames = {e: [] for e in sorted(masked | heldout)}
    from vsa_embed.authoring import replace_frames
    from vsa_embed.compose import FrameSchedule
    schedule = replace_frames(FrameSchedule(host["offsets"], host["relations"], host["fillers"]), visible_frames)
    entry_count = len(full.entry_concepts)
    torch.save({"entry_count": entry_count, "atomic_count": len(host["atomic_names"]), "relation_count": len(host["relation_names"]),
                "offsets": schedule.offsets, "relations": schedule.relations, "fillers": schedule.fillers,
                "relation_names": host["relation_names"], "atomic_names": host["atomic_names"],
                "entry_concepts": full.entry_concepts, "base_entries": sorted(set(base.values())),
                "c3_heldout_entries": sorted(heldout), "visible": True}, root / "visible" / "ontology.pt")
    _json(root / "visible" / "visible.json", {
        "base": base, "excluded": excluded, "lexicon": lexicon, "entry_count": entry_count,
        "relation_names": host["relation_names"], "atomic_names": host["atomic_names"],
        "entry_confidence": [1.0 / len(c) for c in full.entry_concepts]})
    gold_entries = {str(e): {"aliases": sorted(a for a, x in masked_aliases.items() if x == e),
                             "concepts": [ontology.concept_names[c] for c in full.entry_concepts[e]],
                             "frame": [[host["relation_names"][r], host["atomic_names"][a]] for r, a in _frames_of(offsets, relations, fillers, e)],
                             "frame_ids": _frames_of(offsets, relations, fillers, e)}
                    for e in sorted(masked)}
    _json(root / "gold" / "gold.json", {"masked_entries": sorted(masked), "chosen_entries": mask["chosen_entries"],
                                        "entries": gold_entries, "masked_strings": masked_aliases, "sha256": digest,
                                        "concepts": len(names)})
    # corpora
    settings = config["corpora"]
    tokenizer_name = HOSTS[CONSUMER]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    shards = [str(Path(p).expanduser()) for p in c3["paths"]["shards"]]
    eval_docs = int(c3["data"]["eval_docs"])
    gpt2_train = record["manifests"].get("train") or (record["summary"] or {}).get("train_corpus")
    read_start = int(config.get("read_start") or (eval_docs + int(gpt2_train["documents"]) + int(gpt2_train.get("skipped_documents", 0))))
    manifests = {}
    plan = [("general", eval_docs, int(settings["general_tokens"]), False), ("replay", None, int(settings["replay_tokens"]), True),
            ("read", read_start, int(settings["read_tokens"]), True), ("test", None, int(settings["test_tokens"]), True)]
    previous_end = None
    for name, start, tokens, keep in plan:
        first = start if start is not None else previous_end
        manifests[name] = _build_split(name, iter_documents(shards, first), root / "match" / name, tokenizer=tokenizer_name,
                                       strings=strings, eos=tokenizer.eos_token_id, tokens=tokens, settings=settings,
                                       keep_texts=keep, first_document=first, vocab_size=len(tokenizer))
        previous_end = manifests[name]["last_document"]
    track = {"kind": "general", "data_root": str(root), "seed": int(config["seed"]), "consumer": CONSUMER,
             "tokenizer": tokenizer_name, "relations": WORDNET_RELATIONS, "hearst_map": HEARST_MAPS["general"],
             "demonstrations": demos, "direct_demonstrations": [
                 {"context": d["context"], "triples": [(d["surface"], r, f) for r, f in d["edges"]]} for d in demos],
             "pattern": "word", "max_words": int(config["discovery"]["max_words"]), "entry_count": entry_count,
             "settings": config, "masked_sha256": digest, "c3_run": str(c3_run), "host_root": config["host_root"],
             "documents": {k: [m["first_document"], m["last_document"]] for k, m in manifests.items()}}
    summary = {"masked": {"concepts": len(names), "entries": len(masked), "chosen_entries": len(mask["chosen_entries"]),
                          "eligible_entries": mask["eligible_entries"], "aliases": len(masked_aliases), "sha256": digest},
               "base": {"aliases": len(base), "entries": len(set(base.values()))}, "excluded_aliases": len(excluded),
               "c3_heldout_entries": len(heldout), "corpora": manifests, "lexicon_keys": len(lexicon),
               "demonstrations": [d["surface"] for d in demos]}
    return {"track": track, "summary": summary}


def prepare_devtools(config: dict[str, Any], run: Path) -> dict[str, Any]:
    """D7.0 dev-tools track: the C6 v1 library; documentation docs are D_read; masked symbols' schemas are the gold."""
    source = Path(config["devtools_root"])
    frames = json.loads((source / "frames.json").read_text())
    docs = [json.loads(line)["text"] for line in (source / "train.jsonl").read_text().splitlines() if line.strip()]
    heldout = set(frames["heldout"])
    names = frames["concepts"]
    joined = "\n".join(docs)
    counts = {i: len(occurrence_regex(n, identifier=True).findall(joined)) for i, n in enumerate(names) if i not in heldout}
    eligible = sorted(i for i, c in counts.items() if c >= int(config["mask"]["min_count"]))
    rng = np.random.default_rng(int(config["seed"]))
    chosen = sorted(rng.choice(eligible, size=int(round(len(eligible) * float(config["mask"]["fraction"]))), replace=False).tolist())
    masked = set(chosen)
    masked_names = sorted(names[i] for i in chosen)
    digest = hashlib.sha256("\n".join(masked_names).encode()).hexdigest()
    (run / "masked_concepts.txt").write_text("\n".join(masked_names) + "\n")
    key = lambda s: s.strip(" `").lower()
    base = {key(names[i]): i for i in range(len(names)) if i not in masked and i not in heldout}
    excluded = sorted(key(names[i]) for i in heldout)
    lexicon: dict[str, list[int]] = defaultdict(list)
    for index, atom in enumerate(frames["atoms"]):
        lexicon[key(atom.split(":", 1)[1])].append(index)
    root = Path(config["data_root"]).expanduser()
    _json(root / "visible" / "visible.json", {"base": base, "excluded": excluded, "lexicon": dict(lexicon),
                                              "entry_count": len(names), "relation_names": frames["relations"],
                                              "atomic_names": frames["atoms"], "entry_confidence": [1.0] * len(names)})
    _json(root / "gold" / "gold.json", {
        "masked_entries": sorted(masked), "masked_strings": {key(names[i]): i for i in sorted(masked)}, "sha256": digest,
        "entries": {str(i): {"aliases": [key(names[i])], "concepts": [names[i]],
                             "frame": [[frames["relations"][r], frames["atoms"][a]] for r, a in frames["frames"][i]],
                             "frame_ids": frames["frames"][i]} for i in sorted(masked)}, "concepts": len(masked)})
    (root / "read").mkdir(parents=True, exist_ok=True)
    (root / "read" / "texts.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in docs))
    demos = _demos(DEVTOOLS_DEMOS, {key(n) for n in masked_names})
    track = {"kind": "devtools", "data_root": str(root), "seed": int(config["seed"]), "consumer": CONSUMER,
             "tokenizer": HOSTS[CONSUMER], "relations": DEVTOOLS_RELATIONS, "hearst_map": HEARST_MAPS["devtools"],
             "demonstrations": demos, "direct_demonstrations": [
                 {"context": d["context"], "triples": [(d["surface"], r, f) for r, f in d["edges"]]} for d in demos],
             "pattern": "identifier", "max_words": 1, "entry_count": len(names), "settings": config,
             "masked_sha256": digest, "source": str(source)}
    summary = {"masked": {"concepts": len(masked), "eligible": len(eligible), "sha256": digest},
               "base": {"aliases": len(base)}, "excluded_aliases": len(excluded), "read_documents": len(docs)}
    return {"track": track, "summary": summary}


CLINICAL_EXPECTED = ("ontology.pt (entries, frames, names as C3's)", "aliases.json (alias → entry, held-out aliases included)",
                     "holdout.json (held-out MeSH subgraph: entry ids)", "reading texts (PubMed abstracts, jsonl or parquet)")


def prepare_clinical(config: dict[str, Any], run: Path) -> dict[str, Any]:
    """Hook for T1-open (MeSH + PubMed, WP-T1): needs the T1 artifacts; until they exist it stops with
    the list of what is missing. With them it is the general track with MeSH frames as gold."""
    root = Path(config.get("t1_root") or "~/data/vsa-llm/t1/mesh-pubmed-v1").expanduser()
    missing = [name for name in ("ontology.pt", "aliases.json", "holdout.json") if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"T1-open artifacts not found under {root} (missing {missing}); WP-T1 must provide: "
                                + "; ".join(CLINICAL_EXPECTED))
    raise NotImplementedError("T1-open artifacts exist: wire their adapter into prepare_general (see e7 clinical.yaml)")


def prepare(track_kind: str, config: dict[str, Any], run: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(run)
    config = _merge(DEFAULTS, config)
    started = time.monotonic()
    result = {"general": prepare_general, "devtools": prepare_devtools, "clinical": prepare_clinical}[track_kind](config, run)
    track, summary = result["track"], result["summary"]
    _json(run / "track.json", track)
    summary["seconds"] = time.monotonic() - started
    _json(run / "summary.json", summary)
    (run / "report.md").write_text(_prepare_report(track, summary))
    write_run_metadata(run, config, git_at_start=git_at_start, device="cpu", step="prepare", track=track_kind)
    return summary


def _prepare_report(track: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [f"# E7 {track['kind']} track: reading corpora and hidden gold (D7.0)", "",
             f"Masked: **{summary['masked']['concepts']:,}** concepts (sha256 `{summary['masked']['sha256'][:16]}…`, "
             f"list in `masked_concepts.txt`); their frames are kept only in `{track['data_root']}/gold/`.",
             f"Visible (base) aliases: {summary['base']['aliases']:,}; excluded aliases (C3 holdout / C6 held-out): "
             f"{summary['excluded_aliases']:,}.", "", f"Demonstrations: {', '.join(d['surface'] for d in track['demonstrations'])}.", ""]
    if "corpora" in summary:
        lines += ["| corpus | documents (global range) | tokens | matches |", "|---|---|---:|---:|"]
        for name, m in summary["corpora"].items():
            lines.append(f"| {name} | {m['first_document']:,}–{m['last_document']:,} | {m['tokens']:,} | {m['matches']:,} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# reading texts

def read_texts(track: dict[str, Any]) -> list[str]:
    root = data_root(track)
    path = root / "match" / "read" / "texts.jsonl" if track["kind"] == "general" else root / "read" / "texts.jsonl"
    with path.open() as handle:
        return [json.loads(line)["text"] for line in handle]


def _span_settings(track: dict[str, Any]) -> dict[str, Any]:
    if track["pattern"] == "identifier":
        return {"pattern": IDENTIFIER, "max_words": 1, "normalize_key": lambda s: s.strip(" `").lower(),
                "stopwords": STOPWORDS | {"function", "class", "method", "module", "returns", "takes", "raises"}}
    return {"pattern": WORD, "max_words": int(track["max_words"]), "normalize_key": normalize_alias, "stopwords": STOPWORDS}


def load_host(host: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = HOSTS.get(host, host)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval(), tokenizer


def parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------------------------
# D7.1 discovery

def discover(run: Path, host: str, *, device: torch.device, overrides: dict[str, Any] | None = None, model: Any = None,
             tokenizer: Any = None) -> dict[str, Any]:
    track = load_track(run)
    out = run / "discovery" / f"{host}.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = _merge(track["settings"]["discovery"], overrides or {})
    view = visible(track)
    exclude = set(view["base"]) | set(view["excluded"])
    texts = read_texts(track)
    if model is None:
        model, tokenizer = load_host(host, device)
    started = time.monotonic()
    candidates, stats = discover_candidates(
        model, tokenizer, texts, device, exclude=exclude, min_count=int(settings["min_count"]),
        min_subtokens=int(track["settings"]["corpora"]["min_subtokens"]),
        max_candidates=int(settings["keep"]), max_occurrences=int(settings["max_occurrences"]),
        batch=int(settings["batch"]), **_span_settings(track))
    ledger = ComputeLedger()
    ledger.add("discovery", parameters=parameter_count(model), forward_tokens=stats["forward_tokens"],
               seconds=time.monotonic() - started, note=host)
    result = {"host": host, "model": HOSTS.get(host, host), "settings": settings, "stats": stats, "ledger": ledger.to_json(),
              "candidates": [{"rank": i, "surface": c.surface, "count": c.count, "subtokens": c.subtokens,
                              "surprisal": round(c.surprisal, 4), "excess": round(c.excess, 4)} for i, c in enumerate(candidates)]}
    _json(out, result)
    return result


# ---------------------------------------------------------------------------------------------
# D7.1 authoring set and contexts

def link(run: Path, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authoring set: the consumer host's top candidates; candidate matches; disjoint authoring/validation docs."""
    track = load_track(run)
    out = run / "authoring_set.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = _merge(track["settings"]["authoring"], overrides or {})
    discovery = json.loads((run / "discovery" / f"{track['consumer']}.json").read_text())
    chosen = discovery["candidates"][:int(settings["candidates"])]
    surfaces = [c["surface"] for c in chosen]
    seed = int(track["seed"])
    entry_base = int(track["entry_count"])
    items = []
    if track["kind"] == "general":
        from vsa_embed.data.match_corpus import MatchCorpus, add_matches
        root = data_root(track)
        ids = None
        for split in ("read", "test", "replay"):
            found = add_matches(root / "match" / split, surfaces, workers=int(track["settings"]["corpora"]["workers"]))
            if ids is not None and found != ids:
                raise ValueError(f"candidate string ids differ in {split}")
            ids = found
        corpora = {s: MatchCorpus.open(root / "match" / s) for s in ("read", "test", "replay")}
        if not corpora["read"].strings == corpora["test"].strings == corpora["replay"].strings:
            raise ValueError("match corpora disagree on alias strings")
        read = corpora["read"]
        view = visible(track)
        index = read.string_index()
        base_ids = np.asarray([index[a] for a in view["base"] if a in index], dtype=np.int64)
        best = _base_best_chars(read.matches, base_ids)
        for k, (candidate, surface) in enumerate(zip(chosen, surfaces)):
            sid = ids[surface]
            rows = np.nonzero(read.matches["string"] == sid)[0]
            tokens, firsts, chars = (read.matches[key][rows].astype(np.int64) for key in ("token", "first", "chars"))
            valid = chars > best(tokens)
            tokens, firsts = tokens[valid], firsts[valid]
            documents = read.document_of(tokens)
            authoring_docs, validation_docs = split_contexts(documents.tolist(), authoring=int(settings["contexts"]),
                                                             key=surface, seed=seed)
            in_validation = np.isin(documents, validation_docs)
            choice = np.nonzero(in_validation)[0]
            if choice.size > int(settings["max_validation"]):
                rng = np.random.default_rng(stable_seed(seed, "validation", surface))
                choice = np.sort(rng.choice(choice, size=int(settings["max_validation"]), replace=False))
            items.append({"index": k, "surface": surface, "string": int(sid), "entry": entry_base + k, "count": candidate["count"],
                          "excess": candidate["excess"], "authoring_documents": authoring_docs,
                          "validation_documents": len(validation_docs),
                          "validation": [[int(firsts[i]), int(tokens[i])] for i in choice.tolist()],
                          "linked_occurrences": int(tokens.size)})
        strings_digest = hashlib.sha256(json.dumps(read.strings).encode()).hexdigest()
    else:
        texts = read_texts(track)
        identifier = track["pattern"] == "identifier"
        for k, (candidate, surface) in enumerate(zip(chosen, surfaces)):
            pattern = occurrence_regex(surface, identifier=identifier)
            documents = [d for d, text in enumerate(texts) if pattern.search(text)]
            authoring_docs, validation_docs = split_contexts(documents, authoring=int(settings["contexts"]), key=surface, seed=seed)
            items.append({"index": k, "surface": surface, "entry": entry_base + k, "count": candidate["count"],
                          "excess": candidate["excess"], "authoring_documents": authoring_docs,
                          "validation_documents": len(validation_docs), "validation": []})
        strings_digest = None
    result = {"consumer": track["consumer"], "settings": settings, "entry_base": entry_base, "candidates": items,
              "strings_sha256": strings_digest}
    _json(out, result)
    return result


def _base_best_chars(matches: dict[str, np.ndarray], base_ids: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """For token positions: the most characters of a base-alias match ending there (0 if none)."""
    keep = np.isin(matches["string"], base_ids)
    tokens, chars = matches["token"][keep].astype(np.int64), matches["chars"][keep].astype(np.int64)
    order = np.lexsort((-chars, tokens))
    tokens, chars = tokens[order], chars[order]
    unique, first = np.unique(tokens, return_index=True)
    best = chars[first]

    def lookup(positions: np.ndarray) -> np.ndarray:
        if unique.size == 0:
            return np.zeros(positions.shape, dtype=np.int64)
        at = np.clip(np.searchsorted(unique, positions), 0, unique.size - 1)
        return np.where(unique[at] == positions, best[at], 0)
    return lookup


def authoring_contexts(track: dict[str, Any], aset: dict[str, Any], texts: Sequence[str]) -> dict[int, list[tuple[int, str]]]:
    """Per candidate: (document, excerpt) for each authoring document (the first occurrence in it)."""
    identifier = track["pattern"] == "identifier"
    chars = int(track["settings"]["authoring"]["context_chars"])
    out: dict[int, list[tuple[int, str]]] = {}
    for item in aset["candidates"]:
        pattern = occurrence_regex(item["surface"], identifier=identifier)
        contexts = []
        for document in item["authoring_documents"]:
            match = pattern.search(texts[document])
            if match:
                contexts.append((document, excerpt(texts[document], match.start(), match.end(), chars=chars)))
        out[item["index"]] = contexts
    return out


# ---------------------------------------------------------------------------------------------
# D7.1 authors

@torch.no_grad()
def generate_samples(model: Any, tokenizer: Any, prompts: Sequence[str], device: torch.device, *, samples: int,
                     temperature: float, top_p: float, max_new_tokens: int, batch: int, seed: int) -> tuple[list[list[str]], int, int]:
    """`samples` completions per prompt (greedy when temperature is 0); returns them with the prompt
    tokens (counted once per sample) and generated tokens."""
    tokenizer.padding_side = "left"
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out: list[list[str]] = []
    prompt_tokens = generated = 0
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: dict[int, list[str]] = {}
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        encoded = tokenizer([prompts[i] for i in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        torch.manual_seed(stable_seed(seed, start))
        sampling = temperature > 0
        output = model.generate(**encoded, do_sample=sampling, temperature=temperature if sampling else None,
                                top_p=top_p if sampling else None, top_k=None, max_new_tokens=max_new_tokens,
                                num_return_sequences=samples if sampling else 1, pad_token_id=pad)
        new = output[:, encoded["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        repeats = samples if sampling else 1
        prompt_tokens += int(encoded["attention_mask"].sum()) * repeats
        generated += int((new != pad).sum())
        for row, i in enumerate(chunk):
            results[i] = texts[row * repeats:(row + 1) * repeats]       # one completion when greedy
    out = [results[i] for i in range(len(prompts))]
    return out, prompt_tokens, generated


def _proposal_record(votes: Counter, samples: int, contexts: Sequence[int]) -> dict[str, Any]:
    return {"votes": [[r, f, n] for (r, f), n in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))],
            "samples": samples, "contexts": list(contexts)}


def author(run: Path, name: str, *, device: torch.device, overrides: dict[str, Any] | None = None, model: Any = None,
           tokenizer: Any = None, runner: Callable | None = None) -> dict[str, Any]:
    """Proposals of one author for the authoring set (or, for `direct-*`, for the concepts it names)."""
    track = load_track(run)
    out = run / "proposals" / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    if not (run / "authoring_set.json").exists() and not name.startswith("direct-"):
        link(run)
    settings = _merge(track["settings"]["authoring"], overrides or {})
    relations = relation_labels(track)
    texts = read_texts(track)
    ledger = ComputeLedger()
    started = time.monotonic()
    result: dict[str, Any] = {"author": name, "settings": settings}
    if name.startswith("direct-"):
        result.update(_direct(run, track, name[len("direct-"):], texts, device, ledger, model, tokenizer,
                              _merge(track["settings"]["direct"], overrides or {})))
    else:
        aset = json.loads((run / "authoring_set.json").read_text())
        contexts = authoring_contexts(track, aset, texts)
        surfaces = {item["index"]: item["surface"] for item in aset["candidates"]}
        if name in HOSTS or name.startswith("host:"):
            host = name.split(":", 1)[1] if name.startswith("host:") else name
            if model is None:
                model, tokenizer = load_host(host, device)
            prompts, owners = [], []
            for k, items in contexts.items():
                for document, context in items:
                    prompts.append(authoring_prompt_fewshot(surfaces[k], context, relations, track["demonstrations"]))
                    owners.append((k, document))
            completions, prompt_tokens, generated = generate_samples(
                model, tokenizer, prompts, device, samples=int(settings["samples"]), temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]), max_new_tokens=int(settings["max_new_tokens"]), batch=int(settings["batch"]),
                seed=int(track["seed"]))
            grouped: dict[int, list[list[str]]] = defaultdict(list)
            for (k, _), samples in zip(owners, completions):
                grouped[k].append(samples)
            candidates = {}
            for k in surfaces:
                votes, samples = pool_proposals(grouped.get(k, []), relations)
                candidates[surfaces[k]] = _proposal_record(votes, samples, [d for d, _ in contexts[k]])
            ledger.add("authoring", parameters=parameter_count(model), prompt_tokens=prompt_tokens, generated_tokens=generated,
                       seconds=time.monotonic() - started, note=host)
            examples = [{"prompt": prompts[i], "completion": completions[i][0]} for i in range(min(3, len(prompts)))]
            result.update(kind="host", model=HOSTS.get(host, host), candidates=candidates, examples=examples)
        elif name == "teacher":
            from vsa_embed.authoring_baselines import TeacherAuthor
            teacher_settings = _merge(track["settings"]["teacher"], overrides or {})
            teacher = TeacherAuthor(run / "proposals" / "teacher_cache", relations, relation_descriptions(track),
                                    model=teacher_settings["model"], runner=runner,
                                    **({"filler_hint": "a module name, type name, kind, category or symbol name, as written in the "
                                                       "documentation (lowercase)."} if track["kind"] == "devtools" else {}))
            items = [{"id": f"c{k}", "surface": surfaces[k], "contexts": [c for _, c in contexts[k]]}
                     for k in sorted(surfaces) if contexts[k]]
            limit = teacher_settings.get("limit")
            items = items[:int(limit)] if limit else items
            answers = teacher.author(items, batch=int(teacher_settings["batch"]))
            candidates = {}
            for k in sorted(surfaces):
                answer = answers.get(f"c{k}")
                if answer is None:
                    continue
                votes = Counter({edge: 1 for edge in answer["edges"]})
                record = _proposal_record(votes, 1, [d for d, _ in contexts[k]])
                if answer.get("error"):
                    record["error"] = answer["error"]
                candidates[surfaces[k]] = record
            models_used = sorted({m for a in answers.values() for m in (a.get("models_used") or [])})
            result.update(kind="teacher", model=teacher_settings["model"], models_used=models_used, candidates=candidates,
                          calls=teacher.calls, cached_calls=teacher.cached, spent_usd=teacher.spent_usd)
        elif name == "hearst":
            from vsa_embed.authoring_baselines import hearst_extract
            resolve = resolver_for(track, visible(track))
            mapping = track["hearst_map"]
            lookup = {surfaces[k]: k for k in surfaces}
            votes: dict[str, Counter] = defaultdict(Counter)
            for text in texts:
                for concept, relation, filler in hearst_extract(text, lookup, resolve):
                    if relation in mapping:
                        votes[concept][(mapping[relation], filler)] += 1
            candidates = {surfaces[k]: _proposal_record(votes.get(surfaces[k], Counter()),
                                                        sum(votes.get(surfaces[k], Counter()).values()), [])
                          for k in sorted(surfaces)}
            result.update(kind="hearst", candidates=candidates)
        elif name == "random":
            from vsa_embed.authoring_baselines import random_frames
            source = json.loads((run / "proposals" / f"{track['consumer']}.json").read_text())
            view = visible(track)
            resolve = resolver_for(track, view)
            kept = {s: [(r, f) for r, f, n in rec["votes"] if rec["samples"] and n / rec["samples"] >= float(settings["min_share"])
                        and resolve(f) is not None] for s, rec in source["candidates"].items()}
            relation_counts = Counter(r for edges in kept.values() for r, _ in edges)
            weights = [relation_counts.get(r, 0) for r in relations]
            canonical = {}
            for key, atoms in sorted(view["lexicon"].items()):
                canonical.setdefault(atoms[0], key)
            fillers = [canonical[a] for a in sorted(canonical)]
            frames = random_frames({s: len(e) for s, e in kept.items()}, relations, fillers, seed=int(track["seed"]),
                                   relation_weights=weights if sum(weights) else None)
            candidates = {s: _proposal_record(Counter({edge: 1 for edge in frame}), 1, []) for s, frame in frames.items()}
            result.update(kind="random", matched_to=track["consumer"], candidates=candidates)
        else:
            raise ValueError(f"unknown author {name!r}")
    result["ledger"] = ledger.to_json()
    result["seconds"] = time.monotonic() - started
    _json(out, result)
    return result


def _direct(run: Path, track: dict[str, Any], host: str, texts: Sequence[str], device: torch.device, ledger: ComputeLedger,
            model: Any, tokenizer: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """OLLM/LLMs4OL-style: the host names concepts of passages and their facts; no candidate screen."""
    from vsa_embed.authoring_baselines import direct_prompt, parse_direct
    if model is None:
        model, tokenizer = load_host(host, device)
    relations = relation_labels(track)
    rng = random.Random(stable_seed(track["seed"], "direct"))
    documents = sorted(rng.sample(range(len(texts)), k=min(int(settings["documents"]), len(texts))))
    passages = [excerpt(texts[d], 0, 0, chars=2 * int(settings["passage_chars"])) for d in documents]
    prompts = [direct_prompt(p, relations, track["direct_demonstrations"]) for p in passages]
    completions, prompt_tokens, generated = generate_samples(model, tokenizer, prompts, device, samples=1, temperature=0.0,
                                                             top_p=1.0, max_new_tokens=int(settings["max_new_tokens"]),
                                                             batch=int(settings["batch"]), seed=int(track["seed"]))
    view = visible(track)
    exclude = set(view["base"]) | set(view["excluded"])
    key = _span_settings(track)["normalize_key"]
    votes: dict[str, Counter] = defaultdict(Counter)
    docs_of: dict[str, set[int]] = defaultdict(set)
    for document, (completion,) in zip(documents, completions):
        for concept, relation, filler in parse_direct(completion, relations):
            concept = key(concept)
            if concept and concept not in exclude:
                votes[concept][(relation, filler)] += 1
                docs_of[concept].add(document)
    ledger.add("direct", parameters=parameter_count(model), prompt_tokens=prompt_tokens, generated_tokens=generated, note=host)
    candidates = {c: _proposal_record(v, len(docs_of[c]), sorted(docs_of[c])) for c, v in sorted(votes.items())}
    return {"kind": "direct", "model": HOSTS.get(host, host), "documents": len(documents), "candidates": candidates,
            "examples": [{"prompt": prompts[i], "completion": completions[i][0]} for i in range(min(2, len(prompts)))]}


# ---------------------------------------------------------------------------------------------
# D7.1 quality (audit: the only D7.1 step that reads the gold)

def kept_edges(record: dict[str, Any], *, min_share: float) -> list[tuple[str, str]]:
    samples = max(1, int(record.get("samples") or 1))
    return [(r, f) for r, f, n in record["votes"] if n / samples >= min_share]


def edge_matches(edge: tuple[str, str], gold_edges: Sequence[tuple[str, set[str]]], resolver: FillerResolver) -> tuple[bool, bool]:
    """(exact match: same relation and a filler naming the gold atom; filler match: any relation)."""
    relation, filler = edge
    variants = set(resolver.variants(filler))
    filler_hit = [r for r, lemmas in gold_edges if variants & lemmas]
    return relation in filler_hit, bool(filler_hit)


def gold_edge_sets(track: dict[str, Any], gold_data: dict[str, Any], view: dict[str, Any]) -> dict[int, list[tuple[str, set[str]]]]:
    """Per masked entry: (authoring label, filler lemma set) for gold edges whose relation is in the vocabulary."""
    label_of = {relation: label for label, relation, _ in track["relations"]}
    lemmas_of: dict[int, set[str]] = defaultdict(set)
    for key, atoms in view["lexicon"].items():
        for atom in atoms:
            lemmas_of[atom].add(key)
    atom_index = {name: i for i, name in enumerate(view["atomic_names"])}
    out = {}
    for entry, info in gold_data["entries"].items():
        edges = []
        for relation, atom in info["frame"]:
            if relation in label_of and atom in atom_index and lemmas_of.get(atom_index[atom]):
                edges.append((label_of[relation], lemmas_of[atom_index[atom]]))
        out[int(entry)] = edges
    return out


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"value": None, "low": None, "high": None, "n": 0}
    low, high = wilson_interval(successes, total)
    return {"value": successes / total, "low": low, "high": high, "n": total, "successes": successes}


def score_author(proposals: dict[str, Any], gold_edges: dict[int, list[tuple[str, set[str]]]], masked_strings: dict[str, int],
                 resolver: FillerResolver, *, min_share: float) -> dict[str, Any]:
    """Edge precision/recall/F1 and relation-type accuracy on authored concepts that are masked gold concepts."""
    authored = matched = filler_matched = relation_right = 0
    gold_total = gold_found = 0
    concepts = with_edges = 0
    per_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for surface, record in proposals["candidates"].items():
        if surface not in masked_strings:
            continue
        entry = int(masked_strings[surface])
        edges = kept_edges(record, min_share=min_share)
        reference = gold_edges.get(entry, [])
        concepts += 1
        with_edges += bool(edges)
        found = set()
        for edge in edges:
            exact, filler = edge_matches(edge, reference, resolver)
            authored += 1
            matched += exact
            per_relation[edge[0]][0] += 1; per_relation[edge[0]][1] += exact
            if filler:
                filler_matched += 1
                relation_right += exact
            if exact:
                variants = set(resolver.variants(edge[1]))
                found |= {i for i, (r, lemmas) in enumerate(reference) if r == edge[0] and variants & lemmas}
        gold_total += len(reference)
        gold_found += len(found)
    precision, recall = _rate(matched, authored), _rate(gold_found, gold_total)
    p, r = precision["value"], recall["value"]
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"masked_concepts_authored": concepts, "with_edges": with_edges, "edges": authored, "precision": precision,
            "recall": recall, "f1": f1, "relation_accuracy": _rate(relation_right, filler_matched),
            "per_relation_precision": {k: _rate(v[1], v[0]) for k, v in sorted(per_relation.items())}}


def discovery_recall(track: dict[str, Any], gold_data: dict[str, Any], ranked: Sequence[str], *, discoverable: set[int],
                     cutoffs: Sequence[int] = (100, 250, 500, 1000, 2500, 5000)) -> dict[str, Any]:
    """Fraction of discoverable masked entries with an alias among the top-N candidates."""
    masked_strings = gold_data["masked_strings"]
    first_rank: dict[int, int] = {}
    for rank, surface in enumerate(ranked):
        entry = masked_strings.get(surface)
        if entry is not None and int(entry) not in first_rank:
            first_rank[int(entry)] = rank
    total = len(discoverable)
    out = {"discoverable": total}
    for cutoff in cutoffs:
        hits = sum(1 for e in discoverable if first_rank.get(e, 10**12) < cutoff)
        out[f"@{cutoff}"] = _rate(hits, total)
    out["candidates_that_are_masked"] = sum(1 for s in ranked if s in masked_strings)
    return out


def discoverable_entries(track: dict[str, Any], gold_data: dict[str, Any], min_count: int) -> set[int]:
    """Masked entries with an alias occurring ≥ `min_count` times in D_read as a span of ≥ ℓ_min subtokens."""
    masked_strings = gold_data["masked_strings"]
    if track["kind"] == "general":
        from vsa_embed.data.match_corpus import MatchCorpus
        read = MatchCorpus.open(data_root(track) / "match" / "read")
        index = read.string_index()
        counts = np.bincount(read.matches["string"].astype(np.int64), minlength=len(read.strings))
        out = set()
        for alias, entry in masked_strings.items():
            if alias in index and counts[index[alias]] >= min_count:
                out.add(int(entry))
        return out
    joined = "\n".join(read_texts(track))
    return {int(e) for a, e in masked_strings.items() if len(occurrence_regex(a, identifier=True).findall(joined)) >= min_count}


def quality(run: Path, *, min_share: float | None = None) -> dict[str, Any]:
    track = load_track(run)
    gold_data = gold(track)
    view = visible(track)
    resolver = resolver_for(track, view)
    gold_edges = gold_edge_sets(track, gold_data, view) if track["kind"] == "general" else _devtools_gold_edges(track, gold_data, view)
    share = float(track["settings"]["authoring"]["min_share"] if min_share is None else min_share)
    masked_strings = {s: int(e) for s, e in gold_data["masked_strings"].items()}
    discoverable = discoverable_entries(track, gold_data, int(track["settings"]["discovery"]["min_count"]))
    out: dict[str, Any] = {"min_share": share, "discoverable_masked_entries": len(discoverable), "authors": {}, "discovery": {}}
    for path in sorted((run / "discovery").glob("*.json")):
        disc = json.loads(path.read_text())
        out["discovery"][disc["host"]] = discovery_recall(track, gold_data, [c["surface"] for c in disc["candidates"]],
                                                          discoverable=discoverable)
    aset = json.loads((run / "authoring_set.json").read_text()) if (run / "authoring_set.json").exists() else None
    if aset:
        in_set = {item["surface"] for item in aset["candidates"]}
        out["authoring_set"] = {"candidates": len(in_set), "masked": sum(s in masked_strings for s in in_set),
                                "masked_entries": len({masked_strings[s] for s in in_set if s in masked_strings})}
    for path in sorted((run / "proposals").glob("*.json")):
        proposals = json.loads(path.read_text())
        name = proposals["author"]
        threshold = share if proposals.get("kind") == "host" else 0.0
        scored = score_author(proposals, gold_edges, masked_strings, resolver, min_share=threshold)
        if proposals.get("kind") == "direct":
            named = {masked_strings[s] for s in proposals["candidates"] if s in masked_strings}
            scored["discovery"] = _rate(len(named & discoverable), len(discoverable))
        scored["kind"] = proposals.get("kind")
        scored["min_share"] = threshold
        out["authors"][name] = scored
    _json(run / "quality" / "quality.json", out)
    return out


def _devtools_gold_edges(track: dict[str, Any], gold_data: dict[str, Any], view: dict[str, Any]) -> dict[int, list[tuple[str, set[str]]]]:
    label_of = {relation: label for label, relation, _ in track["relations"]}
    out = {}
    for entry, info in gold_data["entries"].items():
        out[int(entry)] = [(label_of[r], {atom.split(":", 1)[1].lower()}) for r, atom in info["frame"] if r in label_of]
    return out


# ---------------------------------------------------------------------------------------------
# D7.1 LLM-judged plausibility of edges without gold (§0.12)

RUBRIC = """# E7.1 judge rubric: authored edges without a gold counterpart (fixed before grading)

Study `authored_edge_in_context` (`vsa_embed.judge_protocol`). Each item shows one usage of a concept
from the reading corpus, the concept, and one authored fact `concept — relation — related concept`.
The judge answers **true**, **partly** or **false**:

- **true** — the fact is correct for the concept in the sense used, and specific enough to be useful
  (e.g. `golden retriever — is_a — dog`).
- **partly** — roughly right but too vague (`— is_a — thing`), only sometimes true, or right for
  another sense of the phrase.
- **false** — wrong, unsupported, or the phrase is not a meaningful concept (a fragment such as
  "number of the").

Relations are shown with their descriptions (`is_a: a more general kind of thing it is`, ...).
Protocol (experiments.md §0.12): items blinded (no author, no gold flag) and shuffled; each item
graded in ≥ 3 independent calls cycling over 2 prompt paraphrases; model pinned (`claude-opus-5-5`)
and recorded with every verdict; verdicts cached by model and prompt hash. Reported: majority
verdict per item, plausibility (true, and true + partly) per author with Wilson intervals, Fleiss κ
across calls, and accuracy on the calibration items (visible-ontology edges vs corrupted edges,
`label_source: ontology`; author-labelled items may be added with `label_source: author`).
"""


def judge_items(run: Path, *, per_author: int = 40, calibration: int = 30, seed: int | None = None) -> dict[str, Any]:
    """Blinded items for edges without gold: edges of new concepts and non-matching edges of masked ones."""
    track = load_track(run)
    gold_data = gold(track)
    view = visible(track)
    resolver = resolver_for(track, view)
    gold_edges = gold_edge_sets(track, gold_data, view) if track["kind"] == "general" else _devtools_gold_edges(track, gold_data, view)
    masked_strings = {s: int(e) for s, e in gold_data["masked_strings"].items()}
    descriptions = relation_descriptions(track)
    seed = int(track["seed"] if seed is None else seed)
    texts = read_texts(track)
    share = float(track["settings"]["authoring"]["min_share"])
    items, key = [], []
    for path in sorted((run / "proposals").glob("*.json")):
        proposals = json.loads(path.read_text())
        name = proposals["author"]
        threshold = share if proposals.get("kind") == "host" else 0.0
        pool = []
        for surface, record in sorted(proposals["candidates"].items()):
            for edge in kept_edges(record, min_share=threshold):
                if surface in masked_strings and edge_matches(edge, gold_edges.get(masked_strings[surface], []), resolver)[0]:
                    continue
                context = _first_context(track, surface, record, texts)
                if context:
                    pool.append((surface, edge, context, surface in masked_strings))
        rng = random.Random(stable_seed(seed, "judge", name))
        for surface, (relation, filler), context, is_masked in rng.sample(pool, k=min(per_author, len(pool))):
            item_id = hashlib.sha256(f"{name}|{surface}|{relation}|{filler}".encode()).hexdigest()[:16]
            items.append({"id": item_id, "fields": {"context": context, "concept": surface,
                                                    "relation": f"{relation} ({descriptions.get(relation, relation)})", "filler": filler}})
            key.append({"id": item_id, "system": name, "masked_concept": is_masked, "label_source": None})
    items += _calibration_items(track, view, texts, count=calibration, seed=seed, key=key)
    random.Random(seed).shuffle(items)
    (run / "judge").mkdir(parents=True, exist_ok=True)
    (run / "judge" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (run / "judge" / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in sorted(key, key=lambda k: k["id"])))
    (run / "judge" / "rubric.md").write_text(RUBRIC)
    return {"items": len(items), "calibration": sum(k["label_source"] == "ontology" for k in key)}


def _first_context(track: dict[str, Any], surface: str, record: dict[str, Any], texts: Sequence[str]) -> str | None:
    pattern = occurrence_regex(surface, identifier=track["pattern"] == "identifier")
    for document in record.get("contexts") or []:
        match = pattern.search(texts[document])
        if match:
            return excerpt(texts[document], match.start(), match.end(), chars=300)
    for text in texts[:2000]:
        match = pattern.search(text)
        if match:
            return excerpt(text, match.start(), match.end(), chars=300)
    return None


def _calibration_items(track: dict[str, Any], view: dict[str, Any], texts: Sequence[str], *, count: int, seed: int,
                       key: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Visible-ontology edges (gold true) and the same with a random filler (gold false), in context."""
    if track["kind"] != "general" or count <= 0:
        return []
    ontology = torch.load(data_root(track) / "visible" / "ontology.pt", weights_only=False)
    label_of = {relation: label for label, relation, _ in track["relations"]}
    names = {}
    for k, atoms in view["lexicon"].items():
        names.setdefault(atoms[0], k)
    rng = random.Random(stable_seed(seed, "calibration"))
    aliases = sorted((a, e) for a, e in view["base"].items() if " " in a)
    rng.shuffle(aliases)
    offsets, relations, fillers = ontology["offsets"], ontology["relations"], ontology["fillers"]
    joined_texts = texts[:3000]
    items = []
    for alias, entry in aliases:
        if len(items) >= count:
            break
        frame = [(ontology["relation_names"][int(r)], int(a)) for r, a in zip(relations[offsets[entry]:offsets[entry + 1]],
                                                                              fillers[offsets[entry]:offsets[entry + 1]])]
        frame = [(label_of[r], a) for r, a in frame if r in label_of and a in names]
        if not frame:
            continue
        pattern = occurrence_regex(alias)
        context = next((excerpt(t, m.start(), m.end(), chars=300) for t in joined_texts if (m := pattern.search(t))), None)
        if context is None:
            continue
        relation, atom = frame[0]
        true_item = len(items) % 2 == 0
        filler = names[atom] if true_item else names[rng.choice(sorted(names))]
        item_id = hashlib.sha256(f"calibration|{alias}|{relation}|{filler}".encode()).hexdigest()[:16]
        items.append({"id": item_id, "fields": {"context": context, "concept": alias,
                                                "relation": f"{relation} ({relation_descriptions(track).get(relation, relation)})",
                                                "filler": filler}})
        key.append({"id": item_id, "system": "calibration", "label_source": "ontology", "gold": "true" if true_item else "false"})
    return items


def judge(run: Path, *, runner: str = "cli", calls: int = 3, model: str = "claude-opus-5-5", seed: int = 0,
          fake: Callable | None = None) -> dict[str, Any]:
    """Grade the items (≥ 3 calls over 2 paraphrases) and summarise plausibility per author, κ and calibration."""
    from vsa_embed.judge_protocol import run_study
    from vsa_embed.judging import JudgeClient, fleiss_kappa
    items = [json.loads(line) for line in (run / "judge" / "items.jsonl").read_text().splitlines() if line.strip()]
    key = {k["id"]: k for k in map(json.loads, (run / "judge" / "key.jsonl").read_text().splitlines()) if k}
    client = JudgeClient(run / "judge" / "cache", model=model, calls=calls, runner=fake,
                         exchange_dir=run / "judge" / "exchange" if runner == "exchange" else None)
    result = run_study(client, "authored_edge_in_context", items, seed=seed)
    by_system: dict[str, list[str]] = defaultdict(list)
    calibration = []
    for item in result["items"]:
        info = key[item["id"]]
        if item["majority"] is None:
            continue
        if info.get("label_source") == "ontology":
            calibration.append(item["majority"] == info["gold"])
        else:
            by_system[info["system"]].append(item["majority"])
    summary = {"model": model, "calls": calls, "items": len(items), "graded": sum(bool(i["scores"]) for i in result["items"]),
               "fleiss_kappa": result["fleiss_kappa"], "spent_usd": result["spent_usd"],
               "calibration_accuracy": _rate(sum(calibration), len(calibration)),
               "plausibility": {s: {"true": _rate(v.count("true"), len(v)),
                                    "true_or_partly": _rate(v.count("true") + v.count("partly"), len(v))}
                                for s, v in sorted(by_system.items())},
               "outstanding": len(list((run / "judge" / "exchange" / "requests").glob("*.json"))) if runner == "exchange" else 0}
    _json(run / "judge" / "results.json", {"summary": summary, "items": result["items"]})
    return summary


# ---------------------------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command")
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "general-v1")
    parser.add_argument("--track", default="general", choices=["general", "devtools", "clinical"])
    parser.add_argument("--config", type=Path, default=None, help="YAML merged over the defaults (prepare)")
    parser.add_argument("--host", default=None); parser.add_argument("--author", default=None)
    parser.add_argument("--overrides", default="{}", help="JSON merged into the step's settings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runner", default="cli", choices=["cli", "exchange"])
    parser.add_argument("--calls", type=int, default=3); parser.add_argument("--per-author", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="accepted for the job queue; every step is idempotent")
    args, rest = parser.parse_known_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    overrides = json.loads(args.overrides)
    if args.command == "prepare":
        config = yaml.safe_load(args.config.read_text()) if args.config else {}
        config = _merge(config, overrides)
        defaults = {"general": {"c3_run": "experiments/c3-general-corpus/runs/v2",
                                "host_root": "~/data/vsa-llm/c3/wordnet-smollm2-v1",
                                "data_root": str(DATA_ROOT / "general-v1")},
                    "devtools": {"devtools_root": "experiments/c6-devtools-benchmark/v1", "data_root": str(DATA_ROOT / "devtools-v1")},
                    "clinical": {"data_root": str(DATA_ROOT / "clinical-v1")}}[args.track]
        print(json.dumps(prepare(args.track, _merge(defaults, config), args.run), indent=2, default=_default)[:4000])
    elif args.command == "discover":
        result = discover(args.run, args.host or CONSUMER, device=device, overrides=overrides)
        print(json.dumps(result["stats"], indent=2))
    elif args.command == "link":
        result = link(args.run, overrides=overrides)
        print(f"{len(result['candidates'])} candidates in the authoring set")
    elif args.command == "author":
        result = author(args.run, args.author, device=device, overrides=overrides)
        print(f"{args.author}: {len(result['candidates'])} concepts")
    elif args.command == "quality":
        print(json.dumps(quality(args.run), indent=1, default=_default)[:6000])
    elif args.command == "judge-items":
        print(judge_items(args.run, per_author=args.per_author))
    elif args.command == "judge":
        print(json.dumps(judge(args.run, runner=args.runner, calls=args.calls), indent=1, default=_default))
    else:
        from vsa_embed.experiments import e7_round
        return e7_round.main([args.command, "--run", str(args.run), "--device", args.device,
                              *(["--resume"] if args.resume else []), *rest])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
