"""T2 developer tools: private libraries (scaled C6 generator) + real API documentation, as a track.

Domain documents (`prepare`, written once to `devtools.docs_dir` and shared by every tokenizer's build):

- **Private libraries** (`benchmarks/devtools_libraries.py`): contamination-free API symbols and their
  documentation in Python and TypeScript styles. Held-out symbols appear only in evaluation documents;
  zero-shot symbols ("new API symbols", E5.4) appear in no document.
- **Real libraries** (`data/api_docs.py`, a frozen snapshot under `real_libraries.snapshot_dir`): API
  symbols and documentation units of installed permissively licensed Python packages, the standard
  library (CPython 3.12 docs) and Node.js v20. A unit goes to evaluation by a hash bucket of its id
  (`eval_buckets` / 10,000) or when it documents a held-out real symbol. Real held-out symbols (leaf
  symbols — no members, never a base, owner, raised exception or replacement of another symbol — with
  ≥ `holdout_min_mentions` alias occurrences in the units and aliases that are neither a prefix of
  another alias nor have one as a prefix, stratified by log2 occurrences) never occur in training text
  either: training units drop every line that contains one of their aliases.

Both streams are interleaved by characters so every prefix of the training (and evaluation) file has
the same private/real composition; the trainer's `train_domain_tokens` then take a prefix. While the
files are written every alias occurrence is counted (the linker's word-start rule), which gives
tokenizer-free mention counts per symbol (`mentions.json`, used to stratify items) and the leakage
audit: no held-out or zero-shot alias, and no held-out private stem as a word, in training text.

Items (all tokenizer-free, so the SmolLM2 and GPT-2 builds share them; `source` is `private` or `real`):
`signature_probe` (symbol → returns / takes / has_param / raises / member_of / kind / inherits, multiple
choice, 2 paraphrases), `doc_qa_cloze` (question-answer form: purpose, library, since, status,
deprecated_by, overrides, calls, raises, returns), `property_probe` (linear probe on a neutral mention:
kind, library, language, status, category), `zeroshot_property` and `zeroshot_entailment` (E5.4 on the
zero-shot symbols and the held-out symbols). Training symbols are sampled per source and mention bin
(1–9, 10–99, ≥ 100 training-document mentions); every held-out symbol is included.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ..benchmarks.devtools_libraries import (AliasMatcher, generate_private_libraries, hidden_names, library_documents,
                                             qualname, symbol_facts)
from ..data.api_docs import build_snapshot, read_jsonl_gz, snapshot_sha256
from ..ontologies.devtools import _RealIndex, build_devtools_ontology, private_concept, real_canonical, real_concept
from ..ontologies.wordnet import FrameOntology
from . import Track
from .common import (RelationTemplates, SyntheticConcept, choice_items, entailment_items, mention_item, text_vocabulary,
                     wordnet_forbidden)

SIGNATURE = {
    "returns": RelationTemplates(["`{x}` returns", "The return type of `{x}` is", "Calling `{x}` gives you a value of type"],
                                 "`{x}` returns {y}."),
    "takes": RelationTemplates(["`{x}` takes an argument of type", "One parameter of `{x}` has the type",
                                "You pass `{x}` a value of type"], "`{x}` takes an argument of type {y}."),
    "has_param": RelationTemplates(["`{x}` accepts a parameter named", "One argument of `{x}` is called",
                                    "The signature of `{x}` includes the parameter"], "`{x}` has a parameter named {y}."),
    "raises": RelationTemplates(["`{x}` can raise", "On invalid input, `{x}` raises", "Callers of `{x}` should catch"],
                                "`{x}` raises {y}."),
    "member_of": RelationTemplates(["`{x}` is defined in", "`{x}` is a member of", "You can find `{x}` in"],
                                   "`{x}` is defined in {y}."),
    "kind": RelationTemplates(["`{x}` is a", "In the API reference, `{x}` is listed as a", "The kind of symbol `{x}` is:"],
                              "`{x}` is a {y}."),
    "inherits": RelationTemplates(["`{x}` is a subclass of", "`{x}` inherits from", "The base class of `{x}` is"],
                                  "`{x}` inherits from {y}."),
}
DOC_QA = {
    "purpose": RelationTemplates(["Q: What does `{x}` do?\nA: It", "Q: What is `{x}` for?\nA: It",
                                  "Q: Summarize `{x}` in one line.\nA: It"], "`{x}` {y}."),
    "category": RelationTemplates(["Q: Which area of the library does `{x}` belong to?\nA: It is used for",
                                   "Q: What kind of work is `{x}` used for?\nA:", "Q: In which part of the docs is `{x}` described?\nA: Under"],
                                  "`{x}` is used for {y}."),
    "library": RelationTemplates(["Q: Which library provides `{x}`?\nA: It is provided by", "Q: What package do I install to use `{x}`?\nA:",
                                  "Q: `{x}` comes from which library?\nA: From"], "`{x}` is provided by {y}."),
    "since": RelationTemplates(["Q: In which version was `{x}` added?\nA: In version", "Q: Since when is `{x}` available?\nA: Since version",
                                "Q: What is the first release with `{x}`?\nA: Version"], "`{x}` was added in version {y}."),
    "status": RelationTemplates(["Q: Is `{x}` safe to use?\nA: It is", "Q: What is the API status of `{x}`?\nA: It is",
                                 "Q: Can I rely on `{x}`?\nA: It is marked"], "`{x}` is {y}."),
    "deprecated_by": RelationTemplates(["Q: `{x}` is deprecated. What should I use instead?\nA: Use",
                                        "Q: What replaces `{x}`?\nA:", "Q: Which API supersedes `{x}`?\nA:"],
                                       "`{x}` is deprecated in favour of {y}."),
    "overrides": RelationTemplates(["Q: Which base method does `{x}` override?\nA: It overrides",
                                    "Q: `{x}` is an override of which method?\nA:", "Q: What does `{x}` re-implement?\nA:"],
                                   "`{x}` overrides {y}."),
    "calls": RelationTemplates(["Q: Which function does `{x}` call internally?\nA: It calls",
                                "Q: What does `{x}` delegate to?\nA:", "Q: Which helper is used inside `{x}`?\nA:"],
                               "`{x}` calls {y}."),
    "raises": RelationTemplates(["Q: Which exception does `{x}` raise?\nA: It raises",
                                 "Q: What should I catch around `{x}`?\nA:", "Q: What error can `{x}` throw?\nA:"],
                                "`{x}` raises {y}."),
    "returns": RelationTemplates(["Q: What does `{x}` return?\nA: It returns", "Q: What is the result type of `{x}`?\nA:",
                                  "Q: What do I get back from `{x}`?\nA:"], "`{x}` returns {y}."),
}
ZEROSHOT = {**DOC_QA, **SIGNATURE}
MENTION_CONTEXTS = ["We call `{x}` in the import step.", "The latest diff touches `{x}`.", "See the docs for `{x}`.",
                    "Our service depends on `{x}`.", "A question came up about `{x}` in code review."]
MENTION_BINS = ((1, 10), (10, 100), (100, 10**12))


def unit_bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10_000


def interleave(primary: Iterator[str], secondary: list[str], share: float, budget: int) -> Iterator[tuple[int, str]]:
    """(source, text): `secondary` documents (each once, in order) whenever their share of the characters
    so far is below `share`, else `primary`, until `budget` characters."""
    produced = secondary_chars = 0
    i = 0
    while produced < budget:
        if i < len(secondary) and secondary_chars < share * (produced + 1):
            text, source = secondary[i], 1
            i += 1
            secondary_chars += len(text)
        else:
            text, source = next(primary), 0
        produced += len(text)
        yield source, text


def choose_real_holdout(records: list[dict[str, Any]], mentions: dict[str, int], *, fraction: float, min_mentions: int,
                        seed: int) -> list[str]:
    """Paths of held-out real symbols (see the module docstring), stratified by log2 mentions."""
    referenced: set[str] = set()
    index = _RealIndex(records)
    for r in records:
        for name in [r.get("owner"), r.get("overrides"), r.get("deprecated_by")] + list(r.get("bases", [])) + list(r.get("raises", [])):
            found = index.find(name)
            if found is not None:
                referenced.add(found["path"])
    from ..experiments.t1_open_corpus import whole_word_substrings
    owner = {a.lower(): r["path"] for r in records for a in r["aliases"]}
    aliases = sorted(owner)
    contained = {part for a, path in owner.items() for part in whole_word_substrings(a)
                 if part in owner and owner[part] != path}

    def isolated(alias: str) -> bool:
        """Not a whole-word part of another symbol's alias, and neither a prefix of another alias nor
        extending one (prefix-causal linking)."""
        a = alias.lower()
        if a in contained:
            return False
        j = bisect_left(aliases, a)
        if j + 1 < len(aliases) and aliases[j + 1].startswith(a):
            return False
        return not any(a[:n] in owner for n in range(1, len(a)))

    eligible = sorted(r["path"] for r in records
                      if r["kind"] in ("function", "method", "class", "property") and r["path"] not in referenced
                      and mentions.get(r["path"], 0) >= min_mentions and all(isolated(a) for a in r["aliases"]))
    rng = random.Random(seed)
    bins: dict[int, list[str]] = {}
    for path in eligible:
        bins.setdefault(int(math.log2(mentions[path])), []).append(path)
    chosen: list[str] = []
    for _, paths in sorted(bins.items()):
        chosen += rng.sample(paths, max(1, round(len(paths) * fraction)))
    return sorted(chosen)


def _repetition(texts: list[str], n: int = 8) -> dict[str, Any]:
    """Distinct word n-grams / all word n-grams on a sample (lower = more repetitive)."""
    grams: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        words = re.findall(r"\w+", text.lower())
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i:i + n]))
            total += 1
    return {"n": n, "ngrams": total, "distinct_ratio": len(grams) / max(1, total)}


class DevToolsTrack(Track):
    name = "t2"

    def __init__(self, config: dict[str, Any], data_root: Path) -> None:
        super().__init__(config, data_root)
        docs = config.get("devtools", {}).get("docs_dir")
        if docs:
            self.docs_dir = Path(docs).expanduser()

    # sources --------------------------------------------------------------------------------------
    def _settings(self) -> dict[str, Any]:
        return self.config["devtools"]

    def _real_settings(self) -> dict[str, Any]:
        return self.config["real_libraries"]

    def snapshot_dir(self) -> Path:
        return Path(self._real_settings()["snapshot_dir"]).expanduser()

    def ensure_snapshot(self) -> dict[str, Any]:
        """Build the real-library snapshot if absent; check the pinned hash if the config has one."""
        cfg = self._real_settings()
        out = self.snapshot_dir()
        if not (out / "manifest.json").exists():
            raw = Path(cfg["raw_dir"]).expanduser()
            lib = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
            from nltk.corpus import wordnet as wn
            build_snapshot(out, packages=list(cfg["python_packages"]),
                           site_packages=Path(cfg.get("site_packages") or lib / "site-packages").expanduser(),
                           stdlib_dir=Path(cfg.get("stdlib_dir") or lib).expanduser(),
                           cpython_docs=raw / cfg["cpython_docs"], inventory=raw / cfg["inventory"], node_api=raw / cfg["node_api"],
                           english_words={l.lower() for l in wn.all_lemma_names()})
        digest = snapshot_sha256(out)
        expected = cfg.get("snapshot_sha256")
        if expected and digest != expected:
            raise ValueError(f"real-library snapshot {out} has sha256 {digest}, config pins {expected}")
        return {**json.loads((out / "manifest.json").read_text()), "sha256": digest}

    def private(self) -> dict[str, Any]:
        if not hasattr(self, "_private"):
            self._private = json.loads((self.docs_dir / "libraries.json").read_text())
        return self._private

    def real_records(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_real"):
            self._real = list(read_jsonl_gz(self.snapshot_dir() / "symbols.jsonl.gz"))
        return self._real

    def real_heldout(self) -> list[str]:
        return json.loads((self.docs_dir / "real_holdout.json").read_text())["paths"]

    def mentions(self) -> dict[str, list[int]]:
        return json.loads((self.docs_dir / "mentions.json").read_text())

    def _request(self) -> dict[str, Any]:
        real = {k: v for k, v in self._real_settings().items() if k not in ("snapshot_dir", "raw_dir")}
        return {"seed": int(self.config["seed"]), "devtools": {k: v for k, v in self._settings().items() if k != "docs_dir"},
                "real_libraries": real, "general_shards": [Path(p).name for p in self.config["paths"]["general_shards"]]}

    # documents ------------------------------------------------------------------------------------
    def prepare(self) -> dict[str, Any]:
        summary_path = self.docs_dir / "documents_summary.json"
        request = self._request()
        if summary_path.exists() and (self.docs_dir / "train.jsonl.gz").exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("request") != request:
                raise ValueError(f"{self.docs_dir} holds documents for a different request; use a fresh devtools.docs_dir")
            return summary
        settings, real_cfg, seed = self._settings(), self._real_settings(), int(self.config["seed"])
        snapshot = self.ensure_snapshot()
        records = self.real_records()
        units = list(read_jsonl_gz(self.snapshot_dir() / "units.jsonl.gz"))
        from ..experiments.c3_corpus import iter_texts
        shards = [str(Path(p).expanduser()) for p in self.config["paths"]["general_shards"]]
        forbidden = wordnet_forbidden() | text_vocabulary(iter_texts(shards, limit=int(settings["forbidden_general_docs"]))) \
            | text_vocabulary(u["text"] for u in units) | {w for r in records for a in r["aliases"] for w in re.findall(r"[a-z]+", a.lower())}
        generator = {k: tuple(v) if isinstance(v, list) else v for k, v in settings.items() if k in (
            "libraries", "typescript_fraction", "modules", "classes", "functions", "methods", "exceptions", "constants",
            "heldout_fraction", "zero_shot", "zipf", "zipf_offset", "deprecated_fraction", "experimental_fraction")}
        private = generate_private_libraries(seed=seed, forbidden=forbidden, **generator)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        (self.docs_dir / "libraries.json").write_text(json.dumps(private, indent=1) + "\n")
        self._private = private

        # real holdout from alias occurrences in the units
        alias_owner = {a.lower(): r["path"] for r in records for a in r["aliases"]}
        real_matcher = AliasMatcher(alias_owner)
        unit_counts: Counter = Counter()
        for u in units:
            real_matcher.count(u["text"], unit_counts)
        real_mentions = Counter()
        for alias, n in unit_counts.items():
            real_mentions[alias_owner[alias]] += n
        held_real = choose_real_holdout(records, real_mentions, fraction=float(real_cfg["heldout_fraction"]),
                                        min_mentions=int(real_cfg["holdout_min_mentions"]), seed=seed)
        (self.docs_dir / "real_holdout.json").write_text(json.dumps({"paths": held_real}, indent=1) + "\n")
        held_set = set(held_real)
        held_aliases = {a.lower() for r in records if r["path"] in held_set for a in r["aliases"]}
        hidden_real = AliasMatcher(held_aliases) if held_aliases else None

        # real pages: units grouped by page up to max_page_chars, train lines with held-out aliases dropped
        max_chars, buckets = int(real_cfg["max_page_chars"]), int(real_cfg["eval_buckets"])
        pages: dict[tuple[str, str, int], list[str]] = {}
        redacted_lines = 0
        for u in units:
            held_unit = u["symbol"] in held_set
            split = "eval" if held_unit or unit_bucket(u["unit"]) < buckets else "train"
            text = u["text"]
            if split == "train" and hidden_real is not None:
                kept = hidden_real.lines_without(text)
                redacted_lines += text.count("\n") - kept.count("\n")
                text = kept.strip()
                if not text:
                    continue
            key = (split, u["page"], 0 if held_unit else 1)
            chunks = pages.setdefault(key, [""])
            if chunks[-1] and len(chunks[-1]) + len(text) > max_chars:
                chunks.append("")
            chunks[-1] = (chunks[-1] + "\n\n" + text) if chunks[-1] else text
        real_docs: dict[str, list[str]] = {"train": [], "eval": []}
        ordered = sorted(((split, priority, hashlib.sha256(f"{page}#{i}".encode()).hexdigest(), chunk)
                          for (split, page, priority), chunks in pages.items() for i, chunk in enumerate(chunks)))
        for split, _, _, chunk in ordered:            # evaluation pages of held-out symbols come first
            real_docs[split].append(chunk)

        # write both streams, counting alias occurrences (mentions, audit)
        private_aliases = {s["name"].lower() for s in private["symbols"]}
        matcher = AliasMatcher(private_aliases | set(alias_owner))
        stats: dict[str, Any] = {}
        counts = {"train": Counter(), "eval": Counter()}
        train_words: set[str] = set()
        samples: dict[str, list[str]] = {"private": [], "real": []}
        sample_chars = Counter()
        for split, budget_key in (("train", "train_chars"), ("eval", "eval_chars")):
            budget = int(settings[budget_key])
            real_chars = sum(map(len, real_docs[split]))
            share = min(float(real_cfg.get(f"max_{split}_share", 1.0)), real_chars / budget)
            stream = library_documents(private, split=split, seed=seed, max_chars=10**15,
                                       uniform_focus=float(settings["eval_uniform_focus"]) if split == "eval" else 0.0)
            docs = chars = 0
            by_source = Counter()
            with gzip.open(self.docs_dir / f"{split}.jsonl.gz.part", "wt", encoding="utf-8") as handle:
                for source, text in interleave(stream, real_docs[split], share, budget):
                    runs = Counter(re.findall(r"[a-z0-9_.$]+", text.lower()))
                    if split == "train" and hidden_real is not None and any(hidden_real.in_run(r) for r in runs):
                        # a held-out real alias in any training document (e.g. `Buffer.from` in a private
                        # TypeScript example): drop the lines that contain it
                        kept = hidden_real.lines_without(text)
                        redacted_lines += text.count("\n") - kept.count("\n")
                        text = kept
                        runs = Counter(re.findall(r"[a-z0-9_.$]+", text.lower()))
                    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    docs += 1; chars += len(text)
                    by_source["real" if source else "private"] += len(text)
                    by_source["real_docs" if source else "private_docs"] += 1
                    for run, n in runs.items():
                        for alias in matcher.in_run(run):
                            counts[split][alias] += n
                    if split == "train":
                        for run in runs:
                            train_words.update(w for w in re.split(r"[_.$]", run) if w)
                        name = "real" if source else "private"
                        if sample_chars[name] < 3_000_000:
                            samples[name].append(text)
                            sample_chars[name] += len(text)
            Path(str(self.docs_dir / f"{split}.jsonl.gz.part")).rename(self.docs_dir / f"{split}.jsonl.gz")
            stats[split] = {"documents": docs, "chars": chars, "private_chars": by_source["private"],
                            "real_chars": by_source["real"], "private_documents": by_source["private_docs"],
                            "real_documents": by_source["real_docs"], "real_share_target": share,
                            "real_documents_available": len(real_docs[split]), "real_chars_available": real_chars}

        # leakage audit: hidden aliases (linker rule) and held-out private stems (whole words)
        hidden = hidden_names(private, "train")
        secret_private = [s for s in private["symbols"] if (s["library"], s["name"]) in hidden]
        leaked = sorted({qualname(s) for s in secret_private if counts["train"][s["name"].lower()]}
                        | {qualname(s) for s in secret_private for stem in s["stems"] if stem.lower() in train_words}
                        | {f"real:{alias_owner[a]}" for a in held_aliases if counts["train"][a]}
                        | {f"eval:{qualname(s)}" for s in private["symbols"]          # zero-shot: in no document at all
                           if s["split"] == "zeroshot" and counts["eval"][s["name"].lower()]})
        audit = {"heldout_private": sum(s["split"] == "heldout" for s in private["symbols"]),
                 "zeroshot_private": sum(s["split"] == "zeroshot" for s in private["symbols"]),
                 "heldout_real": len(held_real), "redacted_real_lines": redacted_lines, "leaked_into_train": len(leaked)}
        if leaked:
            raise RuntimeError(f"held-out/zero-shot symbols leaked into training documents: {leaked[:5]}")
        mentions: dict[str, list[int]] = {}
        for s in private["symbols"]:
            mentions[qualname(s)] = [counts["train"][s["name"].lower()], counts["eval"][s["name"].lower()]]
        for r in records:
            mentions[r["path"]] = [sum(counts["train"][a.lower()] for a in r["aliases"]),
                                   sum(counts["eval"][a.lower()] for a in r["aliases"])]
        (self.docs_dir / "mentions.json").write_text(json.dumps(mentions, sort_keys=True) + "\n")
        general_sample = list(iter_texts(shards, limit=500))
        symbols = private["symbols"]
        summary = {
            "request": request, "code_sha256": _code_digest(), **stats, "audit": audit, "forbidden_words": len(forbidden),
            "private": {"libraries": len(private["libraries"]),
                        "languages": dict(Counter(l["language"] for l in private["libraries"])),
                        "symbols": len(symbols), "by_kind": dict(sorted(Counter(s["kind"] for s in symbols).items())),
                        "by_split": dict(sorted(Counter(s["split"] for s in symbols).items())),
                        "train_mention_bins": _bins([mentions[qualname(s)][0] for s in symbols if s["split"] == "train"])},
            "real": {"snapshot_sha256": snapshot["sha256"], "symbols": len(records), "units": len(units),
                     "unit_chars": snapshot["unit_chars"], "by_library": snapshot["by_library"], "heldout": len(held_real),
                     "train_mention_bins": _bins([mentions[r["path"]][0] for r in records if r["path"] not in held_set])},
            "repetition_8gram": {"private": _repetition(samples["private"]), "real": _repetition(samples["real"]),
                                 "general": _repetition(general_sample)},
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # ontology -------------------------------------------------------------------------------------
    def ontology(self) -> FrameOntology:
        if not hasattr(self, "_ontology"):
            self._ontology = build_devtools_ontology(self.private(), self.real_records(), real_heldout=self.real_heldout(),
                                                     max_atomics=int(self.config["ontology"]["max_atomics"]),
                                                     max_degree=int(self.config["ontology"]["max_degree"]))
        return self._ontology

    def fixed_holdout(self, ontology: FrameOntology) -> list[int]:
        """Private held-out symbols (generator) and real held-out symbols (`prepare`); checked to be
        alias-disjoint from the training linker (no held-out alias is, or is a whole-word part of, a
        training alias)."""
        from ..experiments.t1_open_corpus import assert_alias_disjoint
        from ..span_channel import AliasTable
        held = [i for i, split in enumerate(ontology.metadata["splits"]) if split == "heldout"]
        normalization = (self.config.get("linker") or {}).get("alias_normalization", "default")
        full = AliasTable.from_pairs(ontology.alias_pairs, holdout=held, include_holdout=True, normalization=normalization)
        assert_alias_disjoint(full, full.without_holdout())
        return held

    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        frames = ontology.metadata["zeroshot_frames"]
        return [SyntheticConcept(qualname(s), [s["name"]], frames[qualname(s)], symbol_facts(s),
                                 {"kind": s["kind"], "library": s["library"], "language": s["language"],
                                  "scenario": "new API symbol (E5.4)"})
                for s in self.private()["symbols"] if s["split"] == "zeroshot"]

    # items ----------------------------------------------------------------------------------------
    def records(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Item records (concept, surface, split, source, group, facts, mentions) of every symbol."""
        mentions = self.mentions()
        kept = {c.name for c in context["synthetic"]}
        out = []
        for s in self.private()["symbols"]:
            name = qualname(s)
            if s["split"] == "zeroshot" and name not in kept:
                continue
            split = {"train": "train", "heldout": "heldout", "zeroshot": "synthetic"}[s["split"]]
            out.append({"concept": f"synthetic:{name}" if split == "synthetic" else private_concept(s), "surface": s["name"],
                        "split": split, "source": "private", "group": s["library"], "language": s["language"],
                        "facts": symbol_facts(s), "mentions": mentions[name][0], "kind": s["kind"],
                        "labels": {"kind": s["kind"], "library": s["library"], "language": s["language"],
                                   "status": s["status"], "category": s["category"]}})
        records = self.real_records()
        index = _RealIndex(records)
        held = set(self.real_heldout())
        for r in records:
            out.append({"concept": real_concept(r), "surface": index.canonical[r["path"]],
                        "split": "heldout" if r["path"] in held else "train", "source": "real", "group": r["library"],
                        "language": r["language"], "facts": real_facts(r, index), "mentions": mentions[r["path"]][0],
                        "kind": r["kind"], "labels": {"kind": r["kind"], "library": r["library"], "language": r["language"],
                                                      "status": r["status"]}})
        return out

    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        cfg = self.config["items"]
        seed = int(self.config["seed"])
        rng = random.Random(seed + 5)
        records = self.records(context)
        per_bin, k = int(cfg["train_per_bin"]), int(cfg["relations_per_symbol"])
        train_sample = []
        for source in ("private", "real"):
            for low, high in MENTION_BINS:
                members = sorted((r for r in records if r["source"] == source and r["split"] == "train"
                                  and low <= r["mentions"] < high), key=lambda r: r["concept"])
                train_sample += rng.sample(members, min(per_bin, len(members)))
        heldout = [r for r in records if r["split"] == "heldout"]
        synthetic = [r for r in records if r["split"] == "synthetic"]
        pools = _pools(records)
        signature = _grouped(choice_items, train_sample + heldout, SIGNATURE, pools, task="signature_probe", seed=seed,
                             max_paraphrases=2, max_relations=k)
        doc_qa = _grouped(choice_items, train_sample + heldout, DOC_QA, pools, task="doc_qa_cloze", seed=seed + 1,
                          max_paraphrases=2, max_relations=k)
        zero_property = _grouped(choice_items, synthetic + heldout, ZEROSHOT, pools, task="zeroshot_property",
                                 seed=seed + 2, max_relations=k)
        zero_entail = _grouped(entailment_items, synthetic + heldout, ZEROSHOT, pools, task="zeroshot_entailment",
                               seed=seed + 3, max_relations=k)
        probe = []
        for r in sorted(train_sample, key=lambda r: r["concept"]) + heldout + synthetic:
            text = rng.choice(MENTION_CONTEXTS).format(x=r["surface"])
            labels = {key: value for key, value in r["labels"].items() if value}
            item = mention_item(text, r["surface"], id=f"t2-property_probe-{len(probe):06d}", track="t2", task="property_probe",
                                split=r["split"], source=r["source"], concept=r["concept"], labels=labels, mentions=r["mentions"])
            if item:
                probe.append(item)
        return {"signature_probe": signature, "doc_qa_cloze": doc_qa, "property_probe": probe,
                "zeroshot_property": zero_property, "zeroshot_entailment": zero_entail}


def _code_digest() -> str:
    """sha256 over the sources that generate the documents (provenance in `documents_summary.json`)."""
    from ..benchmarks import devtools, devtools_libraries
    from ..data import api_docs
    digest = hashlib.sha256()
    for module in (devtools, devtools_libraries, api_docs, sys.modules[__name__]):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def _bins(values: list[int]) -> dict[str, int]:
    out = Counter()
    for v in values:
        out["0" if v == 0 else "1-9" if v < 10 else "10-99" if v < 100 else "100+"] += 1
    return {k: out[k] for k in ("0", "1-9", "10-99", "100+")}


def real_facts(record: dict[str, Any], index: _RealIndex) -> dict[str, list[str]]:
    """Relation → readable filler names of a real symbol (canonical names for known symbols)."""
    def show(name: str | None) -> str | None:
        if not name:
            return None
        found = index.find(name)
        if found is not None:
            return index.canonical[found["path"]]
        head = re.split(r"[\[|, ]", name.strip().strip("'\""))[0]
        return head.split(".")[-1] or None
    r = record
    canonical = index.canonical[r["path"]]
    facts: dict[str, list[str]] = {"kind": [r["kind"]], "library": [r["library"]], "language": [r["language"]],
                                   "status": [r["status"]]}
    owner = index.find(r.get("owner"))
    module = canonical.rpartition(".")[0]
    if owner is not None:
        facts["member_of"] = [index.canonical[owner["path"]]]
    elif module:
        facts["member_of"] = [module]
    if r.get("since"):
        facts["since"] = [r["since"]]
    if r["kind"] in ("function", "method", "property") and show(r.get("returns")) not in (None, "Any", "object"):
        facts["returns"] = [show(r["returns"])]
    takes = sorted({t for t in (show(p.get("annotation")) for p in r.get("params", [])) if t and t not in ("Any", "object")})
    if takes:
        facts["takes"] = takes
    params = [p["name"].lstrip("*") for p in r.get("params", []) if p["name"].lstrip("*")]
    if params:
        facts["has_param"] = params
    for relation, values in (("raises", r.get("raises", [])), ("inherits", r.get("bases", [])),
                             ("overrides", [r.get("overrides")]), ("deprecated_by", [r.get("deprecated_by")])):
        shown = sorted({v for v in (show(x) for x in values) if v and v not in ("object", "Generic", "Protocol")})
        if shown:
            facts[relation] = shown
    return facts


def _pools(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, list[str]]]:
    """Distractor pools per (source, group) and relation."""
    pools: dict[tuple[str, str], dict[str, set[str]]] = {}
    for r in records:
        target = pools.setdefault((r["source"], r["group"]), {})
        for relation, fillers in r["facts"].items():
            target.setdefault(relation, set()).update(fillers)
    return {key: {rel: sorted(v) for rel, v in rels.items()} for key, rels in pools.items()}


def _grouped(builder: Any, records: list[dict[str, Any]], templates: dict[str, RelationTemplates],
             pools: dict[tuple[str, str], dict[str, list[str]]], *, task: str, seed: int, **kwargs: Any) -> list[dict[str, Any]]:
    """Run an item builder per (source, group) with that group's distractor pools; renumber ids."""
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        by_group.setdefault((r["source"], r["group"]), []).append(r)
    items: list[dict[str, Any]] = []
    counter = 0
    for offset, (key, members) in enumerate(sorted(by_group.items())):
        rows = builder(members, templates, pools[key], track="t2", task=task, seed=seed + 7919 * offset, **kwargs)
        meta = {r["concept"]: r for r in members}
        renumber: dict[str, str] = {}
        for row in rows:
            link = "group" if "group" in row else "pair"
            if row[link] not in renumber:
                renumber[row[link]] = f"t2-{task}-{counter:06d}"
                counter += 1
            row[link] = renumber[row[link]]
            row["id"] = f"{row[link]}-p{row['paraphrase']}" if link == "group" else f"{row[link]}-{row['label']}"
            row["source"], row["library"] = key
            row["mentions"] = meta[row["concept"]]["mentions"]
        items += rows
    return items


def freeze(config: dict[str, Any]) -> dict[str, Any]:
    """Write the domain documents and return the holdout hash exactly as `track_corpus` computes it,
    so `data.expected_holdout_sha256` can pin it in every config before any build."""
    from . import load_track
    from .common import names_sha256
    track = load_track(config["track"], config, Path(config["paths"]["data_root"]).expanduser())
    summary = track.prepare()
    ontology = track.ontology()
    held = track.fixed_holdout(ontology)
    return {"holdout_sha256": names_sha256(ontology.concept_names[c] for c in held), "heldout_concepts": len(held),
            "audit": summary["audit"], "train_chars": summary["train"]["chars"], "eval_chars": summary["eval"]["chars"]}


def main(argv: list[str] | None = None) -> None:
    """`--stage snapshot`: build (or check) the real-library snapshot and print its sha256;
    `--stage freeze`: write the domain documents and print the holdout sha256 (see `freeze`)."""
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["snapshot", "freeze"], required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.stage == "snapshot":
        track = DevToolsTrack(config, Path(config["paths"]["data_root"]).expanduser())
        info = track.ensure_snapshot()
        print(json.dumps({k: info[k] for k in ("sha256", "symbols", "units", "unit_chars", "by_library", "licences")}, indent=2))
    else:
        print(json.dumps(freeze(config), indent=2))


if __name__ == "__main__":
    main()
