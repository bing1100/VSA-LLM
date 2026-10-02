"""T5 enterprise glossaries: the synthetic private glossary (`benchmarks/glossary.py`) as a track.

Contamination-free by construction: every name is invented and checked against WordNet and the
vocabulary of a general-text sample. The holdout is fixed by the generator (held-out terms appear
only in evaluation documents; `leakage_audit` enforces it) and the zero-shot terms ("new terms
defined only by frames") appear in no document; they are the track's synthetic concepts.

Items: `glossary_cloze` (multiple choice over frame facts, 2 paraphrases each; train terms stratified
by training frequency plus all held-out terms), `term_relation_probe` (linear probe on a neutral
mention; labels type, area, owner, status), `zeroshot_property` and `zeroshot_entailment` (E5.4 on
zero-shot and held-out terms).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..benchmarks.glossary import ARTICLE_TYPES, generate_glossary, term_frame, write_documents
from ..ontologies.glossary import build_glossary_ontology
from ..ontologies.wordnet import FrameOntology
from . import Track
from .common import (RelationTemplates, SyntheticConcept, choice_items, entailment_items, mention_item,
                     text_vocabulary, wordnet_forbidden)

TEMPLATES = {
    "owned_by": RelationTemplates(["{x} is owned by", "Questions about {x} go to", "Ownership of {x} lies with"],
                                  "{x} is owned by {y}."),
    "is_a": RelationTemplates(["In the glossary, {x} is filed under the type", "The kind of thing {x} is: a",
                               "Asked what {x} is, the glossary answers: a"], "{x} is a {y}."),
    "area": RelationTemplates(["{x} belongs to the", "In the org chart, {x} sits in the", "{x} is part of the work of the"],
                              "{x} belongs to the {y} area.", " {y} area"),
    "purpose": RelationTemplates(["The job of {x} is that it", "In short, {x}", "What {x} does: it"], "{x} {y}."),
    "status": RelationTemplates(["{x} is currently", "Status of {x}:", "At the moment, {x} is"], "{x} is currently {y}."),
    "cadence": RelationTemplates(["{x} runs", "{x} happens", "We do {x}"], "{x} runs {y}."),
    "tier": RelationTemplates(["{x} is classified as", "Support level for {x}:", "{x} has support level"],
                              "{x} is classified as {y}."),
    "depends_on": RelationTemplates(["{x} depends on", "{x} cannot run without", "{x} has a hard dependency on"],
                                    "{x} depends on {y}."),
    "uses": RelationTemplates(["{x} uses", "During {x} the team works in", "{x} relies on"], "{x} uses {y}."),
    "governed_by": RelationTemplates(["{x} is governed by", "Changes to {x} must follow", "The rulebook for {x} is"],
                                     "{x} is governed by {y}."),
    "part_of": RelationTemplates(["{x} is part of", "{x} was delivered as part of", "{x} belongs to"], "{x} is part of {y}."),
    "reports_to": RelationTemplates(["{x} reports to", "{x} sits within", "The parent organisation of {x} is"],
                                    "{x} reports to {y}."),
    "measured_by": RelationTemplates(["{x} is measured by", "We track the health of {x} with", "The headline number for {x} is"],
                                     "{x} is measured by {y}."),
    "produces": RelationTemplates(["{x} produces", "The output of {x} is", "Running {x} generates"], "{x} produces {y}."),
    "approved_by": RelationTemplates(["{x} is approved by", "Sign-off for {x} comes from", "Any change to {x} needs approval from"],
                                     "{x} is approved by {y}."),
}
MENTION_CONTEXTS = ["Please send questions about {x} to the usual channel.", "We discussed {x} in today's sync.",
                    "There is a new page about {x} on the wiki.", "The latest update on {x} is below."]


class GlossaryTrack(Track):
    name = "t5"

    def _settings(self) -> dict[str, Any]:
        return self.config["glossary"]

    def glossary(self) -> dict[str, Any]:
        if not hasattr(self, "_glossary"):
            self._glossary = json.loads((self.docs_dir / "glossary.json").read_text())
        return self._glossary

    def prepare(self) -> dict[str, Any]:
        summary_path = self.docs_dir / "documents_summary.json"
        if summary_path.exists() and (self.docs_dir / "train.jsonl.gz").exists():
            return json.loads(summary_path.read_text())
        settings = self._settings()
        from ..experiments.c3_corpus import iter_texts
        shards = [str(Path(p).expanduser()) for p in self.config["paths"]["general_shards"]]
        forbidden = wordnet_forbidden() | text_vocabulary(iter_texts(shards, limit=int(settings["forbidden_general_docs"])))
        glossary = generate_glossary(seed=int(self.config["seed"]), terms=int(settings["terms"]),
                                     zero_shot=int(settings["zero_shot"]), heldout_fraction=float(settings["heldout_fraction"]),
                                     zipf=float(settings["zipf"]), acronym_fraction=float(settings["acronym_fraction"]),
                                     forbidden=forbidden)
        summary = write_documents(glossary, self.docs_dir, seed=int(self.config["seed"]),
                                  train_chars=int(settings["train_chars"]), eval_chars=int(settings["eval_chars"]),
                                  eval_uniform_focus=float(settings["eval_uniform_focus"]))
        summary["forbidden_words"] = len(forbidden)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def ontology(self) -> FrameOntology:
        return build_glossary_ontology(self.glossary(), max_atomics=int(self.config["ontology"]["max_atomics"]),
                                       max_degree=int(self.config["ontology"]["max_degree"]))

    def fixed_holdout(self, ontology: FrameOntology) -> list[int]:
        return [i for i, split in enumerate(ontology.metadata["splits"]) if split == "heldout"]

    def _facts(self, term: dict[str, Any]) -> dict[str, list[str]]:
        types = {t["name"]: t["type"] for t in self.glossary()["terms"]}
        def show(name: str) -> str:
            return f"the {name}" if types[name] in ARTICLE_TYPES else name
        facts = {"is_a": [term["type"]], "area": [term["area"]], "purpose": [term["purpose"]]}
        facts.update({k: [v] for k, v in term["attributes"].items()})
        facts.update({k: [show(f) for f in v] for k, v in term["relations"].items()})
        return facts

    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        return [SyntheticConcept(t["name"], list(t["aliases"]), term_frame(t), self._facts(t),
                                 {"type": t["type"], "kind": "zero-shot glossary term"})
                for t in self.glossary()["terms"] if t["split"] == "zeroshot"]

    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        rng = random.Random(int(self.config["seed"]) + 5)
        terms = self.glossary()["terms"]
        frequency = context["concept_frequency"]          # concept name → training span count
        def record(term: dict[str, Any]) -> dict[str, Any]:
            split = {"train": "train", "heldout": "heldout", "zeroshot": "synthetic"}[term["split"]]
            return {"concept": term["name"], "surface": term["name"], "split": split, "facts": self._facts(term),
                    "frequency": frequency.get(term["name"], 0)}
        kept = {c.name for c in context["synthetic"]}      # synthetic concepts that passed the contamination check
        records = [record(t) for t in terms if t["split"] != "zeroshot" or t["name"] in kept]
        pools: dict[str, set[str]] = {}
        for r in records:
            for relation, fillers in r["facts"].items():
                pools.setdefault(relation, set()).update(fillers)
        pools_sorted = {k: sorted(v) for k, v in pools.items()}
        per_bin = int(self.config["items"]["train_terms_per_frequency_bin"])
        train_sample = []
        for low, high in ((1, 10), (10, 100), (100, 10**12)):
            members = sorted((r for r in records if r["split"] == "train" and low <= r["frequency"] < high),
                             key=lambda r: r["concept"])
            train_sample += rng.sample(members, min(per_bin, len(members)))
        heldout = [r for r in records if r["split"] == "heldout"]
        synthetic = [r for r in records if r["split"] == "synthetic"]
        seed = int(self.config["seed"])
        k = int(self.config["items"]["relations_per_term"])
        cloze = choice_items(train_sample + heldout, TEMPLATES, pools_sorted, track="t5", task="glossary_cloze", seed=seed,
                             max_paraphrases=2, max_relations=k)
        probe = []
        for r in records:
            text = rng.choice(MENTION_CONTEXTS).format(x=r["surface"])
            labels = {k: r["facts"][k][0] for k in ("is_a", "area", "owned_by", "status") if k in r["facts"]}
            item = mention_item(text, r["surface"], id=f"t5-term_relation_probe-{len(probe):06d}", track="t5",
                                task="term_relation_probe", split=r["split"], concept=r["concept"], labels=labels,
                                frequency=r["frequency"])
            if item:
                probe.append(item)
        zero_property = choice_items(synthetic + heldout, TEMPLATES, pools_sorted, track="t5", task="zeroshot_property",
                                     seed=seed + 1, max_relations=k)
        zero_entail = entailment_items(synthetic + heldout, TEMPLATES, pools_sorted, track="t5",
                                       task="zeroshot_entailment", seed=seed + 2, max_relations=k)
        return {"glossary_cloze": cloze, "term_relation_probe": probe, "zeroshot_property": zero_property,
                "zeroshot_entailment": zero_entail}
