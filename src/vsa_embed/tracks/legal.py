"""T6 legal/regulatory: EuroVoc thesaurus; corpus = MultiEURLEX English EU legislation.

Domain documents (`prepare`): the English texts of MultiEURLEX (Chalkidis et al. 2021, CC BY-SA 4.0;
streamed from the hub archive, sha256-verified, English kept). Evaluation documents: the test split;
training documents: train + development splits; both in a seeded order (hash of the CELEX id).

Synthetic concepts (E5.4 "new EuroVoc descriptors"): invented descriptor names under an existing
descriptor — `invented` (two invented stems) or `headed` (invented stem + the head word of the broader
descriptor: a surface cue). Frames: broader, top term, microthesaurus, domain, 1–2 related
descriptors of the same microthesaurus.

Items:
- `eurovoc_probe` (EuroVoc descriptor classification, document level, linear probe): the first
  1,200 characters of MultiEURLEX documents (probe-train from its train split, probe-test from its
  test split) with their gold labels — domains, microthesauri and assigned descriptors (names);
  `split` is `heldout` when an assigned descriptor is held out;
- `defined_term_cloze` (multiple choice): definitions extracted from the acts ("‘X’ means …"), the
  true definition opening against three definitions of other terms; `split` by whether the
  defined term links a held-out / training descriptor;
- `zeroshot_property`, `zeroshot_entailment`: broader / microthesaurus / domain facts of synthetic
  and held-out descriptors.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

from ..benchmarks.devtools import NameMaker
from ..ontologies.eurovoc import build_eurovoc_ontology
from ..ontologies.wordnet import FrameOntology
from ..span_channel import CausalLinker
from . import Track
from .common import RelationTemplates, SyntheticConcept, choice_items, entailment_items

TEMPLATES = {
    "broader": RelationTemplates(["In EuroVoc, {x} is a narrower term of", "{x} is a specific case of",
                                  "The broader EuroVoc term for {x} is"], "In EuroVoc, {x} is a narrower term of {y}."),
    "microthesaurus": RelationTemplates(["{x} is filed in the EuroVoc microthesaurus", "In EuroVoc, {x} belongs to the microthesaurus",
                                         "The EuroVoc subject area of {x} is"], "{x} is filed in the EuroVoc microthesaurus {y}."),
    "domain": RelationTemplates(["{x} belongs to the EuroVoc domain", "The EuroVoc domain of {x} is",
                                 "In EuroVoc, {x} is classified under the domain"], "{x} belongs to the EuroVoc domain {y}."),
}
DEFINITION = re.compile(r"[‘'\"“]([^’'\"”\n]{2,80})[’'\"”]\s+(?:shall\s+)?means?\s+([^;\n]{20,400})")
_TOKENS = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[a-z0-9)\]])\.\s+(?=[A-Z(])")


def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def read_eurlex(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def extract_definitions(text: str) -> list[tuple[str, str]]:
    """("term", "definition") pairs of the "‘X’ means …" pattern of EU legislation."""
    out = []
    for match in DEFINITION.finditer(text):
        term, definition = " ".join(match.group(1).split()), " ".join(match.group(2).split())
        definition = _SENTENCE_END.split(definition, 1)[0]      # stop at the end of the sentence
        if any(ch.isalpha() for ch in term) and len(term.split()) <= 8 and len(definition) >= 20:
            out.append((term, definition.rstrip(",. ")))
    return out


class LegalTrack(Track):
    name = "t6"

    def _raw(self, key: str) -> Path:
        return Path(self.config["legal"][key]).expanduser()

    def ontology(self) -> FrameOntology:
        if not hasattr(self, "_ontology_cache"):
            self._ontology_cache = build_eurovoc_ontology(self._raw("eurovoc"), max_atomics=int(self.config["ontology"]["max_atomics"]),
                                                          max_degree=int(self.config["ontology"]["max_degree"]))
        c = self._ontology_cache
        return FrameOntology(c.name, list(c.concept_names), list(c.relation_names), list(c.atomic_names),
                             [list(f) for f in c.frames], list(c.alias_pairs), dict(c.metadata))

    def _split_files(self, split: str) -> list[Path]:
        root = self._raw("multi_eurlex")
        return [root / "test.en.jsonl.gz"] if split == "eval" else [root / "train.en.jsonl.gz", root / "dev.en.jsonl.gz"]

    def prepare(self) -> dict[str, Any]:
        summary_path = self.docs_dir / "documents_summary.json"
        if summary_path.exists() and (self.docs_dir / "train.jsonl.gz").exists():
            return json.loads(summary_path.read_text())
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for split in ("eval", "train"):
            rows = [r for path in self._split_files(split) for r in read_eurlex(path)]
            rows.sort(key=lambda r: _bucket(r["celex_id"]))
            with gzip.open(self.docs_dir / f"{split}.jsonl.gz.part", "wt", encoding="utf-8") as out:
                for r in rows:
                    out.write(json.dumps({"text": r["text"], "celex_id": r["celex_id"]}, ensure_ascii=False) + "\n")
            Path(self.docs_dir / f"{split}.jsonl.gz.part").rename(self.docs_dir / f"{split}.jsonl.gz")
            summary[split] = {"documents": len(rows), "chars": sum(len(r["text"]) for r in rows)}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # -- synthetic descriptors -------------------------------------------------------------------
    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        rng = random.Random(int(self.config["seed"]) + 11)
        names = NameMaker(rng, {w.lower() for w in forbidden})
        records = ontology.metadata["records"]
        atoms = set(ontology.atomic_names)
        candidates = sorted(c for c, r in records.items() if f"{c}" in atoms and r["schemes"])
        by_scheme: dict[str, list[str]] = {}
        for cid, r in records.items():
            for s in r["schemes"]:
                by_scheme.setdefault(s, []).append(cid)
        mt_notation, mt_labels = ontology.metadata["mt_notation"], ontology.metadata["mt_labels"]
        domains = ontology.metadata["domain_labels"]
        existing = {a.lower() for a, _ in ontology.alias_pairs}
        out = []
        for k in range(int(self.config["synthetic"]["count"])):
            kind = "invented" if k % 2 == 0 else "headed"
            parent = rng.choice(candidates)
            r = records[parent]
            head = r["label"].split(" (")[0].split()[-1]
            name = (f"{names.stem()} {names.stem()}" if kind == "invented" else f"{names.stem()} {head}").lower()
            if name in existing:
                continue
            scheme = r["schemes"][0]
            notation = mt_notation[scheme]
            frame = [("broader", parent)]
            if r["top_term"]:
                top_uri = next((c for c, rr in records.items() if rr["label"] == r["top_term"]), None)
                if top_uri and top_uri in atoms:
                    frame.append(("top_term", top_uri))
            frame += [("microthesaurus", f"mt:{notation}"), ("domain", f"domain:{notation[:2]}")]
            related = [c for c in rng.sample(by_scheme[scheme], min(3, len(by_scheme[scheme]))) if c != parent and c in atoms][:2]
            frame += [("related", c) for c in related]
            facts = {"broader": [r["label"]], "microthesaurus": [mt_labels[scheme]], "domain": [domains[notation[:2]]]}
            out.append(SyntheticConcept(name, [name], [e for e in frame if e[1] in atoms], facts,
                                        {"kind": kind, "broader": parent, "surface_cue": kind == "headed"}))
        return out

    # -- items -------------------------------------------------------------------------------------
    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        ontology: FrameOntology = context["ontology"]
        table = context["table"]
        seed = int(self.config["seed"])
        rng = random.Random(seed + 5)
        settings = self.config["items"]
        records = ontology.metadata["records"]
        held = {ontology.concept_names[c] for c in context["holdout_concepts"]}
        mt_labels = ontology.metadata["mt_labels"]
        domain_uris = ontology.metadata["domain_uris"]
        linker = CausalLinker(table, min_subtokens=1)

        def linked(text: str) -> set[str]:
            offsets = [(m.start(), m.end()) for m in _TOKENS.finditer(text)]
            return {ontology.concept_names[c] for s in linker.link(text, offsets) for c in table.entry_concepts[s.entry]
                    if (offsets[s.end_token][1] - offsets[s.start_token][0]) >= 6}

        # 1. document-level EuroVoc classification.
        probe = []
        for source, cap in (("train", int(settings["probe_train_docs"])), ("test", int(settings["probe_test_docs"]))):
            rows = list(read_eurlex(self._raw("multi_eurlex") / f"{source}.en.jsonl.gz"))
            for r in rng.sample(rows, min(cap, len(rows))):
                labels = r["eurovoc_concepts"]
                descriptors = [f"eurovoc:{d}" for d in labels.get("all_levels", [])]
                probe.append({
                    "id": f"t6-eurovoc_probe-{len(probe):06d}", "track": "t6", "task": "eurovoc_probe",
                    "split": "heldout" if set(descriptors) & held else "train", "probe_split": source,
                    "celex_id": r["celex_id"], "text": r["text"][:1200],
                    "labels": {"domains": sorted(domain_uris.get(f"http://eurovoc.europa.eu/{d}", d) for d in labels.get("level_1", [])),
                               "microthesauri": sorted(mt_labels.get(f"http://eurovoc.europa.eu/{d}", d) for d in labels.get("level_2", [])),
                               "descriptors": sorted(records[d]["label"] for d in descriptors if d in records)},
                    "heldout_descriptors": sorted(records[d]["label"] for d in set(descriptors) & held if d in records)})
            del rows

        # 2. defined-term cloze.
        definitions: dict[str, tuple[str, str, str]] = {}
        for source in ("test", "train"):
            for r in read_eurlex(self._raw("multi_eurlex") / f"{source}.en.jsonl.gz"):
                for term, definition in extract_definitions(r["text"]):
                    definitions.setdefault(term.lower(), (term, definition, f"{source}:{r['celex_id']}"))
        keys = sorted(definitions)
        rng.shuffle(keys)
        openings = {k: " ".join(definitions[k][1].split()[:12]) for k in keys}
        cloze = []
        for key in keys[:int(settings["defined_terms"])]:
            term, _, origin = definitions[key]
            others = rng.sample([k for k in keys if openings[k] != openings[key]], 3)
            options = [openings[key]] + [openings[o] for o in others]
            rng.shuffle(options)
            concepts = linked(term)
            split = "heldout" if concepts & held else "train" if concepts else "unlinked"
            group = f"t6-defined_term_cloze-{len(cloze):06d}"
            for paraphrase, prompt in enumerate([f"For the purposes of this Regulation, ‘{term}’ means",
                                                 f"In this Directive, the term ‘{term}’ means"]):
                cloze.append({"id": f"{group}-p{paraphrase}", "group": group, "track": "t6", "task": "defined_term_cloze",
                              "split": split, "concept": term, "surface": term, "source": origin, "paraphrase": paraphrase,
                              "prompt": prompt, "choices": [" " + o for o in options], "label": options.index(openings[key]),
                              "linked_concepts": sorted(concepts)})

        # 3. zero-shot frame facts.
        def facts(cid: str) -> dict[str, list[str]]:
            r = records[cid]
            out = {"broader": r["broader"], "microthesaurus": r["microthesauri"], "domain": r["domains"]}
            return {k: v for k, v in out.items() if v}
        concepts = [{"concept": f"synthetic:{s.name}", "surface": s.name, "split": "synthetic", "facts": s.facts}
                    for s in context["synthetic"]]
        concepts += [{"concept": cid, "surface": records[cid]["label"], "split": "heldout", "facts": facts(cid)}
                     for cid in sorted(held) if cid in records]
        pools: dict[str, set[str]] = {}
        for cid in records:
            for rel, fillers in facts(cid).items():
                pools.setdefault(rel, set()).update(fillers)
        pools_sorted = {k: sorted(v) for k, v in pools.items()}
        zero_property = choice_items(concepts, TEMPLATES, pools_sorted, track="t6", task="zeroshot_property", seed=seed + 1)
        zero_entail = entailment_items(concepts, TEMPLATES, pools_sorted, track="t6", task="zeroshot_entailment", seed=seed + 2)
        return {"eurovoc_probe": probe, "defined_term_cloze": cloze, "zeroshot_property": zero_property,
                "zeroshot_entailment": zero_entail}
