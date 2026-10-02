"""T4 chemistry: ChEBI ontology; corpus = ChEBI entry texts + chemistry abstracts from PubMed.

Domain documents (`prepare`):
- **ChEBI entries** (3-star entities with a definition): name, definition and PubChem-style relation
  sentences ("It has a role as …", "It is functionally related to …", "It is a conjugate acid of …")
  generated from the ChEBI relationships — the form in which PubChem shows ChEBI-sourced compound
  descriptions. 10% of entities (by a hash of the ChEBI id) go to evaluation.
- **PubMed chemistry abstracts** from NLM baseline files already on disk (shared with the T1 track;
  md5-verified against NCBI's companion files, never modified): title + abstract, kept if the
  citation has a ChemicalList or the abstract links ≥ `min_links` distinct ChEBI entities through
  aliases of ≥ 8 characters or several words. 3% of PMIDs (by hash) go to evaluation.

Synthetic concepts (E5.4 "new compounds by IUPAC name"): ring-substituted derivatives of aromatic ChEBI
parent compounds (`PARENTS`, each with the locants its name allows) that have a family class named
after them ("benzoic acid" → "benzoic acids", "phenol" → "phenols"), named by IUPAC substitutive
nomenclature ("3-bromo-5-ethoxybenzoic acid") and absent
from every ChEBI name and synonym (all stars) and from the text samples. Frames: the family class and
substituent classes (organochlorine compound, …) as `is_a`, `has_functional_parent`, elements,
charge, branch, and one invented role (`has_role`, marked `fictional_role`), so role questions can
only be answered from the frame. Substitution keeps membership of the family class, so the class
facts are chemically right; PubChem could not be searched offline, so a name may exist there.

Items: `class_role_probe` (IUPAC name in a neutral sentence → coarse chemical class and frequent roles;
linear probe), `property_cloze` (multiple choice over is_a / has_role / has_functional_parent /
conjugate / parent-hydride / enantiomer facts), `zeroshot_property` and `zeroshot_entailment`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..data.pubmed import iter_abstracts, verify_md5
from ..ontologies.chebi import build_chebi_ontology, formula_elements
from ..ontologies.wordnet import FrameOntology
from ..span_channel import AliasTable, CausalLinker
from . import Track, write_jsonl_texts
from .common import RelationTemplates, SyntheticConcept, choice_items, entailment_items, mention_item

SUBSTITUENTS = {   # prefix: (elements added, ChEBI class name for the substituent, if any)
    "chloro": (["Cl"], "organochlorine compound"), "bromo": (["Br"], "organobromine compound"),
    "fluoro": (["F"], "organofluorine compound"), "iodo": (["I"], "organoiodine compound"),
    "nitro": (["N", "O"], "C-nitro compound"), "cyano": (["N"], "nitrile"), "amino": (["N"], None),
    "hydroxy": (["O"], None), "methoxy": (["O"], "aromatic ether"), "ethoxy": (["O"], "aromatic ether"),
    "methyl": ([], None), "ethyl": ([], None), "propyl": ([], None), "acetyl": (["O"], None),
    "formyl": (["O"], "aldehyde"), "sulfanyl": (["S"], "thiol"), "trifluoromethyl": (["F"], "organofluorine compound"),
}
# Aromatic parent compounds and the ring locants that take a substituent in their names; substituents
# that would rename the parent ("3-hydroxyphenol" is resorcinol) are excluded per parent.
PARENTS = {
    "benzoic acid": [2, 3, 4, 5, 6], "phenol": [2, 3, 4, 5, 6], "aniline": [2, 3, 4, 5, 6], "benzamide": [2, 3, 4, 5, 6],
    "benzaldehyde": [2, 3, 4, 5, 6], "benzonitrile": [2, 3, 4, 5, 6], "anisole": [2, 3, 4], "pyridine": [2, 3, 4],
    "nicotinic acid": [2, 4, 5, 6], "quinoline": [2, 3, 4, 5, 6, 7, 8], "indole": [4, 5, 6, 7], "tryptamine": [4, 5, 6, 7],
    "coumarin": [5, 6, 7, 8], "chromone": [5, 6, 7, 8], "catechol": [3, 4], "resorcinol": [2, 4, 5], "hydroquinone": [2],
    "cinnamic acid": [2, 3, 4], "phenylacetic acid": [2, 3, 4], "salicylic acid": [3, 4, 5, 6], "acetophenone": [2, 3, 4],
}
EXCLUDED = {"phenol": {"hydroxy"}, "catechol": {"hydroxy"}, "resorcinol": {"hydroxy"}, "hydroquinone": {"hydroxy"},
            "salicylic acid": {"hydroxy"}, "aniline": {"amino"}, "benzonitrile": {"cyano"}, "benzaldehyde": {"formyl"},
            "anisole": {"methoxy"}, "acetophenone": {"acetyl"}}
# Six-membered rings numbered from the principal group: positions 2↔6 and 3↔5 are equivalent, so the
# lower locant set is used (lowest-locants rule: "2-acetylphenol", not "6-acetylphenol").
SYMMETRIC = {"benzoic acid", "phenol", "aniline", "benzamide", "benzaldehyde", "benzonitrile", "anisole", "pyridine",
             "cinnamic acid", "phenylacetic acid", "acetophenone"}
FICTIONAL_ROLES = ["antifungal agent", "herbicide", "antibacterial agent", "antioxidant", "plant growth regulator",
                   "anti-inflammatory agent", "insecticide", "antineoplastic agent", "fluorescent probe", "fungicide",
                   "enzyme inhibitor", "antiviral agent", "anticonvulsant", "food preservative", "dye"]
TEMPLATES = {
    "is_a": RelationTemplates(["{x} is a member of the class of", "Chemically, {x} is classified among the",
                               "{x} belongs to the chemical class of"], "{x} is a member of the class of {y}."),
    "has_role": RelationTemplates(["{x} has a role as", "One role of {x} is that of", "{x} is used or acts as"],
                                  "{x} has a role as {y}."),
    "has_functional_parent": RelationTemplates(["{x} is functionally related to", "The functional parent of {x} is",
                                                "{x} is derived from the parent compound"], "{x} is functionally related to {y}."),
    "is_conjugate_acid_of": RelationTemplates(["{x} is the conjugate acid of", "Deprotonating {x} gives",
                                               "The conjugate base of {x} is"], "{x} is the conjugate acid of {y}."),
    "is_conjugate_base_of": RelationTemplates(["{x} is the conjugate base of", "Protonating {x} gives",
                                               "The conjugate acid of {x} is"], "{x} is the conjugate base of {y}."),
    "has_parent_hydride": RelationTemplates(["{x} has the parent hydride", "The parent hydride of {x} is",
                                             "{x} is formally derived from the hydride"], "{x} has the parent hydride {y}."),
    "is_enantiomer_of": RelationTemplates(["{x} is the enantiomer of", "The mirror-image form of {x} is",
                                           "{x} and its enantiomer"], "{x} is the enantiomer of {y}."),
}
PROBE_CONTEXTS = ["{x} was tested in this study.", "We measured the concentration of {x} in each sample.",
                  "The synthesis of {x} is described below.", "Samples were treated with {x}."]
_TOKENS = re.compile(r"\w+|[^\w\s]")


def _a(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class ChemistryTrack(Track):
    name = "t4"

    def _obo(self) -> Path:
        return Path(self.config["chebi"]["obo"]).expanduser()

    def ontology(self) -> FrameOntology:
        if not hasattr(self, "_ontology_cache"):
            settings = self.config["chebi"]
            self._ontology_cache = build_chebi_ontology(
                self._obo(), min_star=int(settings["min_star"]), max_atomics=int(self.config["ontology"]["max_atomics"]),
                max_degree=int(self.config["ontology"]["max_degree"]), role_budget=int(settings["role_budget"]),
                max_alias_chars=int(settings["max_alias_chars"]))
        # A fresh copy each call: the builder appends synthetic concepts to the returned object.
        cached = self._ontology_cache
        return FrameOntology(cached.name, list(cached.concept_names), list(cached.relation_names), list(cached.atomic_names),
                             [list(f) for f in cached.frames], list(cached.alias_pairs), dict(cached.metadata))

    # -- documents ---------------------------------------------------------------------------------
    def entry_text(self, record: dict[str, Any], names: dict[str, str]) -> str:
        parts = [f"{record['name']}: {record['definition']}"]
        roles = [names[t] for rel, t in record["relations"] if rel == "has_role" and t in names]
        if roles:
            parts.append(f"It has a role as {_join([_a(r) for r in roles])}.")
        classes = [names[p] for p in record["is_a"] if p in names]
        if classes:
            parts.append(f"It is a member of {_join(classes)}.")
        for relation, phrase in (("has_functional_parent", "It is functionally related to"),
                                 ("is_conjugate_acid_of", "It is a conjugate acid of"),
                                 ("is_conjugate_base_of", "It is a conjugate base of"),
                                 ("has_parent_hydride", "It derives from a hydride of"),
                                 ("is_tautomer_of", "It is a tautomer of"), ("is_enantiomer_of", "It is an enantiomer of")):
            targets = [names[t] for rel, t in record["relations"] if rel == relation and t in names]
            if targets:
                parts.append(f"{phrase} {_join(targets)}.")
        return " ".join(parts)

    def prepare(self) -> dict[str, Any]:
        summary_path = self.docs_dir / "documents_summary.json"
        if summary_path.exists() and (self.docs_dir / "train.jsonl.gz").exists():
            return json.loads(summary_path.read_text())
        ontology = self.ontology()
        records, names = ontology.metadata["records"], ontology.metadata["all_names"]
        eval_percent = int(self.config["chebi"]["eval_percent"])
        chebi_docs = {"eval": [], "train": []}
        for cid, record in sorted(records.items()):
            if record.get("definition"):
                chebi_docs["eval" if _bucket(cid) < eval_percent else "train"].append(self.entry_text(record, names))
        table = AliasTable.from_pairs(ontology.alias_pairs)
        linker = CausalLinker(table, min_subtokens=1)
        settings = self.config["pubmed"]
        pubmed_dir = Path(settings["dir"]).expanduser()
        pubmed_docs = {"eval": [], "train": []}
        files = {}
        for name in settings["files"]:
            path = pubmed_dir / name
            if not verify_md5(path):
                raise ValueError(f"{path} failed its md5 check (or has no .md5 companion)")
            kept = total = 0
            for record in iter_abstracts(path):
                total += 1
                text = f"{record['title']}\n{record['abstract']}"
                offsets = [(m.start(), m.end()) for m in _TOKENS.finditer(text)]
                lowered = text.lower()
                entries = {s.entry for s in linker.link(text, offsets)
                           if (lambda alias: len(alias) >= 8 or " " in alias)(lowered[offsets[s.start_token][0]:offsets[s.end_token][1]])}
                if record["chemicals"] or len(entries) >= int(settings["min_links"]):
                    kept += 1
                    pubmed_docs["eval" if _bucket(record["pmid"] or text) < int(settings["eval_percent"]) else "train"].append(text)
            files[name] = {"abstracts": total, "kept": kept}
        rng = random.Random(int(self.config["seed"]))
        summary: dict[str, Any] = {"pubmed_files": files}
        for split in ("eval", "train"):
            docs = chebi_docs[split] + pubmed_docs[split]
            rng.shuffle(docs)
            summary[split] = {**write_jsonl_texts(self.docs_dir / f"{split}.jsonl.gz", docs),
                              "chebi_entries": len(chebi_docs[split]), "pubmed_abstracts": len(pubmed_docs[split])}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # -- synthetic compounds ---------------------------------------------------------------------
    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        settings = self.config["synthetic"]
        rng = random.Random(int(self.config["seed"]) + 11)
        records, names = ontology.metadata["records"], ontology.metadata["all_names"]
        by_name = {}
        for cid, name in names.items():
            by_name.setdefault(name.lower(), cid)
        atoms = set(ontology.atomic_names)
        existing = {a.lower() for a, _ in ontology.alias_pairs} | ontology.metadata["all_surface_forms"]
        family = {}       # parent compound → its family class ("benzoic acid" → "benzoic acids")
        for name in PARENTS:
            cid = by_name.get(name)
            plural = name[:-4] + "acids" if name.endswith(" acid") else name + "s"
            cls = by_name.get(plural)
            if cid in records and f"chebi:{cid}" in atoms and cls and f"chebi:{cls}" in atoms:
                family[cid] = cls
        parents = sorted(family)
        roles = [by_name[r] for r in FICTIONAL_ROLES if r in by_name and f"chebi:{by_name[r]}" in atoms]
        out: list[SyntheticConcept] = []
        seen: set[str] = set()
        attempts = 0
        while len(out) < int(settings["count"]) and attempts < 100 * int(settings["count"]):
            attempts += 1
            parent = records[rng.choice(parents)]
            allowed_subs = sorted(set(SUBSTITUENTS) - EXCLUDED.get(parent["name"], set()))
            allowed_locants = PARENTS[parent["name"]]
            chosen = rng.sample(allowed_subs, min(len(allowed_locants), rng.randint(1, int(settings["max_substituents"]))))
            locants = rng.sample(allowed_locants, len(chosen))
            if parent["name"] in SYMMETRIC and sorted(8 - l for l in locants) < sorted(locants):
                locants = [8 - l for l in locants]
            prefix = "-".join(f"{loc}-{sub}" for sub, loc in sorted(zip(chosen, locants)))
            name = f"{prefix}{parent['name']}"
            if name in seen or name.lower() in existing:
                continue
            seen.add(name)
            frame: list[tuple[str, str]] = []
            facts: dict[str, list[str]] = {"is_a": [], "has_functional_parent": [parent["name"]]}
            cls = family[parent["id"]]
            frame.append(("is_a", f"chebi:{cls}")); facts["is_a"].append(names[cls])
            for sub in chosen:
                cls_name = SUBSTITUENTS[sub][1]
                cls = by_name.get(cls_name) if cls_name else None
                if cls and f"chebi:{cls}" in atoms and ("is_a", f"chebi:{cls}") not in frame:
                    frame.append(("is_a", f"chebi:{cls}")); facts["is_a"].append(names[cls])
            frame.append(("has_functional_parent", f"chebi:{parent['id']}"))
            role = rng.choice(roles) if roles else None
            if role:
                frame.append(("has_role", f"chebi:{role}")); facts["has_role"] = [_a(names[role])]
            elements = sorted(set(formula_elements(parent.get("formula"))) | {e for s in chosen for e in SUBSTITUENTS[s][0]})
            frame += [("contains_element", f"element:{e}") for e in elements if f"element:{e}" in atoms]
            frame += [("charge", "charge:neutral"), ("branch", "branch:chemical entity")]
            if not facts["is_a"]:
                del facts["is_a"]
            out.append(SyntheticConcept(name, [name], frame, facts,
                                        {"parent": parent["id"], "substituents": chosen, "fictional_role": True}))
        return out

    # -- items -------------------------------------------------------------------------------------
    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        ontology: FrameOntology = context["ontology"]
        records, names = ontology.metadata["records"], ontology.metadata["all_names"]
        seed = int(self.config["seed"])
        rng = random.Random(seed + 5)
        heldout_ids = {ontology.concept_names[c] for c in context["holdout_concepts"]}
        frequency = context["concept_frequency"]

        def facts(record: dict[str, Any]) -> dict[str, list[str]]:
            out: dict[str, list[str]] = {}
            if record["is_a"]:
                out["is_a"] = [names[p] for p in record["is_a"] if p in names]
            for rel, target in record["relations"]:
                if rel in TEMPLATES and target in names:
                    out.setdefault(rel, []).append(_a(names[target]) if rel == "has_role" else names[target])
            return out

        concepts = []
        for cid, record in sorted(records.items()):
            split = "heldout" if cid in heldout_ids else "train"
            concepts.append({"concept": cid, "surface": record["name"], "split": split, "facts": facts(record),
                             "frequency": frequency.get(cid, 0), "record": record})
        pools: dict[str, set[str]] = {}
        for c in concepts:
            for rel, fillers in c["facts"].items():
                pools.setdefault(rel, set()).update(fillers)
        synthetic = [{"concept": f"synthetic:{s.name}", "surface": s.name, "split": "synthetic", "facts": s.facts}
                     for s in context["synthetic"]]
        for s in synthetic:
            for rel, fillers in s["facts"].items():
                pools.setdefault(rel, set()).update(fillers)
        pools_sorted = {k: sorted(v) for k, v in pools.items()}
        per_bin = int(self.config["items"]["train_concepts_per_frequency_bin"])
        k = int(self.config["items"]["relations_per_concept"])
        train_sample = []
        for low, high in ((1, 10), (10, 100), (100, 10**12)):
            members = [c for c in concepts if c["split"] == "train" and low <= c["frequency"] < high and c["facts"]]
            train_sample += rng.sample(members, min(per_bin, len(members)))
        heldout = [c for c in concepts if c["split"] == "heldout" and c["facts"]]
        cloze = choice_items(train_sample + heldout, TEMPLATES, pools_sorted, track="t4", task="property_cloze",
                             seed=seed, max_paraphrases=2, max_relations=k)
        # class/role probe on IUPAC names: coarse class = a frequent direct superclass, roles = frequent roles.
        class_use = Counter(p for c in concepts for p in c["record"]["is_a"])
        role_use = Counter(t for c in concepts for rel, t in c["record"]["relations"] if rel == "has_role")
        top_classes = {cid for cid, _ in class_use.most_common(int(self.config["items"]["probe_classes"]))}
        top_roles = {cid for cid, _ in role_use.most_common(int(self.config["items"]["probe_roles"]))}
        probe = []
        probe_concepts = [c for c in concepts if c["split"] == "heldout"]
        train_pool = [c for c in concepts if c["split"] == "train"]
        probe_concepts += rng.sample(train_pool, min(int(self.config["items"]["probe_train_concepts"]), len(train_pool)))
        for c in probe_concepts:
            iupac = [text for text, _, kind in c["record"]["synonyms"] if kind.startswith("IUPAC")]
            if not iupac:
                continue
            classes = sorted(names[p] for p in c["record"]["is_a"] if p in top_classes)
            roles = sorted({names[t] for rel, t in c["record"]["relations"] if rel == "has_role" and t in top_roles})
            if not classes and not roles:
                continue
            item = mention_item(rng.choice(PROBE_CONTEXTS).format(x=iupac[0]), iupac[0],
                                id=f"t4-class_role_probe-{len(probe):06d}", track="t4", task="class_role_probe",
                                split=c["split"], concept=c["concept"], labels={"classes": classes, "roles": roles},
                                frequency=c["frequency"])
            if item:
                probe.append(item)
        for s in context["synthetic"]:
            classes = sorted(f for f in s.facts.get("is_a", []))
            item = mention_item(rng.choice(PROBE_CONTEXTS).format(x=s.name), s.name, id=f"t4-class_role_probe-{len(probe):06d}",
                                track="t4", task="class_role_probe", split="synthetic", concept=f"synthetic:{s.name}",
                                labels={"classes": classes, "roles": [r.split(" ", 1)[1] for r in s.facts.get("has_role", [])]},
                                frequency=0)
            if item:
                probe.append(item)
        zero_templates = {k: TEMPLATES[k] for k in ("is_a", "has_role", "has_functional_parent")}
        zero_property = choice_items(synthetic + heldout, zero_templates, pools_sorted, track="t4", task="zeroshot_property",
                                     seed=seed + 1, max_relations=k)
        zero_entail = entailment_items(synthetic + heldout, zero_templates, pools_sorted, track="t4",
                                       task="zeroshot_entailment", seed=seed + 2, max_relations=k)
        return {"property_cloze": cloze, "class_role_probe": probe, "zeroshot_property": zero_property,
                "zeroshot_entailment": zero_entail}
