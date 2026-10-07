"""E9 understanding items: tests whose correct answer is **not** a filler word of the term's frame, so copying a filler
cannot solve them (author request 2026-10-07; pre-registration `experiments/e9-retrofit/preregistration-understanding.md`,
family B1). Evaluation only, on finished runs; gold answers come from the track ontology (exact on T5's generator).

**Families** (track specs below; every option is checked against the term's verbalizations within two hops — its own
aliases, its frame's fillers and the fillers of the frames of concepts that fill its frame, as words — and an item whose
option shares a content word with them is dropped, except where a family says otherwise):

- `two_hop` — the answer is a property of a filler, not of the term: "X is owned by a team that reports to" → the
  owner's department (T5), "The functional parent of X has a role as" → the parent's role (T4). The answer atom is not in
  X's frame and no distractor is a filler of X's frame. References (not in the composite): `hop1` (X's own filler, a copy
  item) and `bridge` (the filler's property asked of the filler itself: does the host know the second hop?).
- `affordance` — a consequence of the term's type, never a filler: "Most people interact with X by" → " logging in to it"
  (system) vs the other types' phrases (T5); the electrode an ion migrates to (T4, from its charge); what is done with a
  chemical / disease / organism (T7).
- `paraphrase` — property selection whose options describe each filler by a definition with no lexical overlap with any
  verbalization of the term ("Right now, X is" → " being phased out" for status deprecated; element symbols → element
  names; MeSH tree categories by name).
- `reverse` — given the filler, pick the term, both candidates in context: "Of X and Y, the one owned by the Zash Team is"
  → " X" (both orders; PMI against "the one we mean"), the reversal direction through the channel.
- `comparison` — two terms of the same subset: "X and Y are owned by" → " the same team" / " two different teams"
  (balanced same / different pairs).
- `negation` — "X is not owned by" → the false filler; scored as pair consistency with the affirmative item (both
  correct); a model that copies the filler fails the negated item.

**Subsets:** `seen` (training frequency ≥ 10), `rare` (1–9), `heldout` (held-out and generator zero-shot terms: rows
composed zero-shot) and `new` (the E9 new words, `items/new-words-<track>-<family>-v2` for T5 — decision 56 — else v1;
their frames are inserted at evaluation as in dimension 3). Anchors: single-concept entries whose canonical alias links
(≥ ℓ_min subtokens) and, on T5, of a non-structural type.

**Scoring:** as E5.4 — PMI of each option against a null prompt (the term replaced by "this"/"that"; for `reverse` the
cue replaced by "we mean"), argmax over options, per template; an item's score is the mean over its templates.
Sources: `own` (the run's rows; new words: their inserted frames), `none` (the anchors' rows zeroed: what C0′/P0 see),
`random_frame` (composing channels: the anchors' frames replaced by random frames of equal degree). Item record format:
`items.jsonl.gz` rows (gzip with mtime 0, so the sha256 in the manifest is reproducible) {`id`, `family`, `test`, `subset`,
`anchor`, `slots` (template slot → concept), `text` (fixed slots),
`null` (slot overrides of the null prompt), `templates`, `candidates`, `gold`, `relation`, `pair`, `meta`}.

    python -m vsa_embed.experiments.e9_understanding items --track t5 --output experiments/e9-retrofit/items/understanding-t5-smollm2-v1
    python -m vsa_embed.experiments.e9_understanding evaluate --run RUN --items DIR [--sources own,none,random_frame] [--output OUT]
    python -m vsa_embed.experiments.e9_understanding queue --stage t5 --items DIR --priority 56 [--models …] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import yaml

from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable, normalize_alias
from ..statistics import holm_adjust
from . import e5_zeroshot as zs
from . import e9_ontology_edit as edit
from .e5_common import E5Run, clear_output, finish_output, json_ready, open_run, override_rows, start_output, write_json

SCHEMA = "e9-understanding/1"
FAMILIES = ("two_hop", "affordance", "paraphrase", "reverse", "comparison", "negation")
REFERENCE_TESTS = ("hop1", "bridge")
SUBSETS = ("seen", "rare", "heldout", "new")
SOURCES = ("own", "none", "random_frame")
ROOT = Path("experiments/e9-retrofit")
ITEMS_ROOT = ROOT / "items"
OUTPUT_PREFIX = "understanding"
NULL_X, NULL_Y, NULL_CUE = "this", "that", "we mean"
STEM_SUFFIXES = ("ings", "ing", "ed", "es", "s", "ly")


# ---------------------------------------------------------------- files


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Deterministic gzip (mtime 0), so the file's sha256 is a function of its rows."""
    import gzip
    import io
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(json_ready(row)) + "\n").encode())
    Path(path).write_bytes(buffer.getvalue())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of `path` or of `path` + `.gz`."""
    import gzip
    path = Path(path)
    if path.exists():
        text = path.read_text()
    elif path.with_name(path.name + ".gz").exists():
        text = gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode()
    else:
        raise FileNotFoundError(path)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------- lexical overlap

def stem(word: str) -> str:
    """A crude, fixed stemmer for the overlap rule (lower case; -ing/-ed/-es/-s/-ly removed when ≥ 3 letters remain; a
    doubled final consonant collapsed): "running" = "runs" = "run", "weekly" = "weeks" = "week"."""
    w = word.lower()
    for suffix in STEM_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def word_stems(texts: Iterable[str]) -> set[str]:
    from .e9_freqbias import content_words
    return {stem(w) for t in texts for w in content_words(t)}


def overlaps(option: str, forbidden: set[str]) -> bool:
    return bool(word_stems([option]) & forbidden)


# ---------------------------------------------------------------- track specs

T5_TYPES = ("system", "process", "metric", "policy", "document", "project", "dataset", "tool")
T5_SPEC: dict[str, Any] = {
    "anchor": {"relation": "is_a", "values": [f"type:{t}" for t in T5_TYPES]},
    "two_hop": [
        {"path": ("owned_by", "reports_to"), "templates": ["{x} is owned by a team that reports to", "The team that owns {x} sits within"]},
        {"path": ("governed_by", "approved_by"), "templates": ["{x} is governed by a policy that is approved by",
                                                               "The policy that governs {x} needs sign-off from"]},
        {"path": ("governed_by", "owned_by"), "templates": ["{x} is governed by a policy that is owned by",
                                                            "The policy that governs {x} is owned by"]},
        {"path": ("part_of", "sponsored_by"), "templates": ["{x} is part of a project that is sponsored by",
                                                            "The project that {x} belongs to is sponsored by"]},
        {"path": ("measured_by", "owned_by"), "templates": ["{x} is measured by a metric that is owned by",
                                                            "The metric that measures {x} is owned by"]},
        {"path": ("depends_on", "owned_by"), "templates": ["{x} depends on something that is owned by",
                                                           "Something that {x} depends on is owned by"]},
        {"path": ("uses", "owned_by"), "templates": ["{x} uses something that is owned by", "Something that {x} uses is owned by"]},
    ],
    "affordance": {"relation": "is_a", "items": {
        "use": {"templates": ["Most people interact with {x} by", "The usual way to work with {x} is by"],
                "options": {"type:system": " logging in to it", "type:process": " carrying out its steps in order",
                            "type:metric": " checking whether its value went up or down", "type:policy": " making sure they comply with it",
                            "type:document": " reading it from start to finish", "type:project": " helping it finish before its deadline",
                            "type:dataset": " querying its rows", "type:tool": " installing it on their laptop"}},
        "event": {"templates": ["Last week, {x}", "According to the latest update, {x}"],
                  "options": {"type:system": " crashed and had to be restarted", "type:process": " was carried out late again",
                              "type:metric": " dropped by three percent", "type:policy": " was violated twice",
                              "type:document": " was edited and signed again", "type:project": " slipped past its deadline",
                              "type:dataset": " was downloaded by an analyst", "type:tool": " was installed on every laptop"}}}},
    "paraphrase": {
        "is_a": {"templates": ["{x} is best described as", "In plain words, {x} is"], "k": 5, "options": {
            "type:system": [" a piece of software infrastructure", " a computer platform that other software connects to"],
            "type:process": [" a sequence of steps that people carry out", " a routine that staff follow"],
            "type:metric": [" a number that is watched over time", " a quantity that is tracked as a figure"],
            "type:policy": [" a set of rules everyone must obey", " a binding rule set"],
            "type:document": [" a written text you can read", " a piece of writing"],
            "type:project": [" a temporary effort with a goal and an end date", " a time-limited undertaking"],
            "type:dataset": [" a collection of stored records", " a table of stored entries"],
            "type:tool": [" a small utility that staff use", " a handy application for employees"]}},
        "area": {"templates": ["{x} is mainly concerned with", "The work around {x} is about"], "k": 5, "options": {
            "area:finance": [" money, budgets and accounting", " the company's money"],
            "area:logistics": [" moving and storing goods", " getting goods from place to place"],
            "area:security": [" protecting the company from attackers", " keeping intruders out"],
            "area:people operations": [" hiring and looking after employees", " staff and their careers"],
            "area:sales": [" selling to clients", " closing deals with buyers"],
            "area:compliance": [" following laws and regulations", " obeying regulators"],
            "area:engineering": [" building and shipping software", " writing code"],
            "area:customer support": [" helping users who have problems", " answering complaints from users"],
            "area:procurement": [" buying what the company needs from outside firms", " purchasing from outside firms"],
            "area:data": [" organising information so it can be analysed", " analytics and information management"]}},
        "status": {"templates": ["Right now, {x} is", "These days, {x} is"], "k": 4, "options": {
            "status:active": [" in everyday use", " fully in service"],
            "status:pilot": [" being trialled by a small group", " in an early trial"],
            "status:deprecated": [" being phased out", " on its way out"],
            "status:planned": [" not yet built", " still only an idea"]}},
        "cadence": {"templates": ["{x} takes place", "{x} is scheduled to happen"], "k": 4, "options": {
            "cadence:daily": [" every working morning", " each morning"],
            "cadence:weekly": [" once every seven days", " every seven days"],
            "cadence:monthly": [" twelve times a year", " once in each calendar period of about thirty days"],
            "cadence:quarterly": [" four times a year", " every three months"]}},
        "tier": {"templates": ["{x} is supported", "When {x} breaks, it is supported"], "k": 3, "options": {
            "tier:tier 1": [" with the highest priority", " first, before anything else"],
            "tier:tier 2": [" with medium priority", " after the most critical services"],
            "tier:tier 3": [" with the lowest priority", " last, when there is time"]}}},
    "reverse": {
        "owned_by": {"cue": "owned by {t}", "match": "is_a"},
        "area": {"cue": "in the {t} area", "match": "is_a"}},
    "reverse_templates": ["Of {x} and {y}, the one {c} is", "Between {y} and {x}, the one {c} is"],
    "comparison": {
        "owned_by": {"templates": ["{x} and {y} are owned by", "Ownership of {x} and {y} lies with"],
                     "candidates": [" the same team", " two different teams"]},
        "area": {"templates": ["{x} and {y} belong to", "In the org chart, {x} and {y} sit in"],
                 "candidates": [" the same area", " two different areas"]},
        "is_a": {"templates": ["{x} and {y} are", "In the glossary, {x} and {y} are filed as"],
                 "candidates": [" the same kind of thing", " two different kinds of things"]}},
    "negation": {
        "owned_by": {"affirm": ["{x} is owned by", "Ownership of {x} lies with"],
                     "negate": ["{x} is not owned by", "Ownership of {x} does not lie with"]},
        "area": {"affirm": ["{x} belongs to the", "In the org chart, {x} sits in the"],
                 "negate": ["{x} does not belong to the", "In the org chart, {x} does not sit in the"]},
        "status": {"affirm": ["{x} is currently", "At the moment, {x} is"], "negate": ["{x} is not currently", "At the moment, {x} is not"]},
        "is_a": {"affirm": ["The kind of thing {x} is: a", "In the glossary, {x} is filed under the type"],
                 "negate": ["The kind of thing {x} is not: a", "In the glossary, {x} is not filed under the type"]}},
}

ELEMENT_NAMES = {"H": "hydrogen", "B": "boron", "C": "carbon", "N": "nitrogen", "O": "oxygen", "F": "fluorine", "Na": "sodium",
                 "Mg": "magnesium", "Al": "aluminium", "Si": "silicon", "P": "phosphorus", "S": "sulfur", "Cl": "chlorine",
                 "K": "potassium", "Ca": "calcium", "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel", "Cu": "copper",
                 "Zn": "zinc", "As": "arsenic", "Se": "selenium", "Br": "bromine", "Ag": "silver", "Sn": "tin", "I": "iodine",
                 "Pt": "platinum", "Au": "gold", "Hg": "mercury", "Pb": "lead", "Li": "lithium", "Gd": "gadolinium", "Tc": "technetium",
                 "Ga": "gallium", "Cr": "chromium", "Mo": "molybdenum", "Bi": "bismuth", "Sb": "antimony", "Ba": "barium"}
T4_SPEC: dict[str, Any] = {
    "anchor": {"relation": "branch", "values": ["branch:chemical entity"]},
    "two_hop": [
        {"path": ("has_functional_parent", "has_role"), "templates": ["{x} is derived from a parent compound that has a role as",
                                                                      "The functional parent of {x} has a role as"]},
        {"path": ("is_conjugate_acid_of", "has_role"), "templates": ["The conjugate base of {x} has a role as",
                                                                     "Deprotonating {x} gives a compound that has a role as"]},
        {"path": ("is_conjugate_base_of", "has_role"), "templates": ["The conjugate acid of {x} has a role as",
                                                                     "Protonating {x} gives a compound that has a role as"]},
        {"path": ("is_a", "has_role"), "templates": ["{x} belongs to a chemical class that has a role as",
                                                     "A class of compounds that includes {x} has a role as"]},
    ],
    "affordance": {"relation": "charge", "items": {
        "electrophoresis": {"templates": ["In electrophoresis, {x} migrates toward", "In an electric field, {x} drifts toward"],
                            "options": {"charge:negative": " the anode", "charge:positive": " the cathode",
                                        "charge:neutral": " neither electrode"}}}},
    "paraphrase": {
        "contains_element": {"templates": ["{x} contains atoms of", "Among the elements in {x} is"], "k": 5, "skip": ["element:C", "element:H"],
                             "options": {f"element:{s}": [" " + n] for s, n in ELEMENT_NAMES.items()}},
        "charge": {"templates": ["Overall, {x} has", "Each particle of {x} has"], "k": 3, "options": {
            "charge:negative": [" more electrons than protons"], "charge:positive": [" fewer electrons than protons"],
            "charge:neutral": [" as many electrons as protons"]}}},
    "reverse": {
        "has_role": {"cue": "that has a role as {t}", "match": None, "article": True},
        "is_a": {"cue": "that is classified as {t}", "match": None}},
    "reverse_templates": ["Of {x} and {y}, the one {c} is", "Between {y} and {x}, the one {c} is"],
    "comparison": {
        "is_a": {"templates": ["{x} and {y} belong to", "Chemically, {x} and {y} fall into"],
                 "candidates": [" the same chemical class", " two different chemical classes"]},
        "charge": {"templates": ["{x} and {y} carry", "Overall, {x} and {y} have"],
                   "candidates": [" the same net charge", " different net charges"]}},
    "negation": {
        "has_role": {"affirm": ["{x} has a role as", "One role of {x} is that of"],
                     "negate": ["{x} does not have a role as", "One role that {x} lacks is that of"]},
        "is_a": {"affirm": ["{x} is a member of the class of", "Chemically, {x} is classified among the"],
                 "negate": ["{x} is not a member of the class of", "Chemically, {x} is not classified among the"]}},
}

# MeSH 2026 tree categories (second level) for T7's `branch_second` atoms, which carry no text of their own.
MESH_CATEGORIES = {
    "B01": "eukaryotes", "B02": "archaea", "B03": "bacteria", "B04": "viruses", "B05": "organism forms",
    "C01": "infections", "C04": "neoplasms", "C05": "musculoskeletal diseases", "C06": "digestive system diseases",
    "C07": "stomatognathic diseases", "C08": "respiratory tract diseases", "C09": "otorhinolaryngologic diseases",
    "C10": "nervous system diseases", "C11": "eye diseases", "C12": "urogenital diseases", "C14": "cardiovascular diseases",
    "C15": "hemic and lymphatic diseases", "C16": "congenital, hereditary, and neonatal diseases and abnormalities",
    "C17": "skin and connective tissue diseases", "C18": "nutritional and metabolic diseases", "C19": "endocrine system diseases",
    "C20": "immune system diseases", "C21": "disorders of environmental origin", "C22": "animal diseases",
    "C23": "pathological conditions, signs and symptoms", "C24": "occupational diseases", "C25": "chemically-induced disorders",
    "C26": "wounds and injuries",
    "D01": "inorganic chemicals", "D02": "organic chemicals", "D03": "heterocyclic compounds", "D04": "polycyclic compounds",
    "D05": "macromolecular substances", "D06": "hormones, hormone substitutes, and hormone antagonists", "D08": "enzymes and coenzymes",
    "D09": "carbohydrates", "D10": "lipids", "D12": "amino acids, peptides, and proteins",
    "D13": "nucleic acids, nucleotides, and nucleosides", "D20": "complex mixtures", "D23": "biological factors",
    "D25": "biomedical and dental materials", "D26": "pharmaceutical preparations", "D27": "chemical actions and uses"}
T7_SPEC: dict[str, Any] = {
    "anchor": {"relation": "record_class", "values": ["class:chemical", "class:disease", "class:organism"]},
    "two_hop": [],                 # T7 fillers (MeSH headings) have no frames in the track ontology
    "affordance": {"relation": "record_class", "items": {
        "handling": {"templates": ["In practice, {x} is usually", "Researchers report that {x} is usually"],
                     "options": {"class:chemical": " synthesized and given in measured doses",
                                 "class:disease": " diagnosed in patients and then treated",
                                 "class:organism": " grown in culture and sequenced"}}}},
    "paraphrase": {
        "branch_second": {"templates": ["In the MeSH tree, {x} falls under", "{x} is catalogued among the"], "k": 5,
                          "options": {f"branch:{code}": [" " + name] for code, name in MESH_CATEGORIES.items()}, "same_prefix": True},
        "record_class": {"templates": ["{x} is", "Put simply, {x} is"], "k": 3, "options": {
            "class:chemical": [" a substance with a defined molecular structure"], "class:disease": [" an illness that affects patients"],
            "class:organism": [" a living species"]}}},
    "reverse": {"mapped_to": {"cue": "indexed under {t}", "match": "record_class"},
                "pharmacological_action": {"cue": "that acts as {t}", "match": "record_class"}},
    "reverse_templates": ["Of {x} and {y}, the one {c} is", "Between {y} and {x}, the one {c} is"],
    "comparison": {
        "mapped_to": {"templates": ["{x} and {y} are indexed under", "In MeSH, {x} and {y} share"],
                      "candidates": [" the same heading", " different headings"]},
        "record_class": {"templates": ["{x} and {y} are", "In MeSH, {x} and {y} are recorded as"],
                         "candidates": [" the same kind of entity", " different kinds of entities"]}},
    "negation": {
        "mapped_to": {"affirm": ["{x} is a kind of", "In MeSH, {x} is indexed under"],
                      "negate": ["{x} is not a kind of", "In MeSH, {x} is not indexed under"]},
        "pharmacological_action": {"affirm": ["The pharmacological action of {x} is", "{x} is classed among the"],
                                   "negate": ["The pharmacological action of {x} is not", "{x} is not classed among the"]}},
}
TRACK_SPECS = {"t5": T5_SPEC, "t4": T4_SPEC, "t7": T7_SPEC}


# ---------------------------------------------------------------- build context


class BuildContext:
    """Ontology views for the item builders: frames by relation name, the entry an atomic names, filler text pools, the
    two-hop verbalization words of a frame (the exclusion table) and the canonical surfaces of entries."""

    def __init__(self, *, track: str, family: str, ontology: dict[str, Any], table: AliasTable, lexicon: Any, tokenizer: Any,
                 tokenizer_name: str, exclusions: dict[str, Any], families_spec: dict[str, Any], min_subtokens: int = 2,
                 ontology_path: Path | None = None, alias_table_path: Path | None = None) -> None:
        from .e5_common import canonical_surfaces
        self.track, self.family, self.families_spec = track, family, families_spec
        self.ontology, self.table, self.lexicon, self.tokenizer = ontology, table, lexicon, tokenizer
        self.tokenizer_name, self.exclusions = tokenizer_name, exclusions
        self.ontology_path, self.alias_table_path = ontology_path, alias_table_path
        self.view = view = edit.OntologyView(ontology, table, lexicon)
        self.names = [str(n) for n in ontology["concept_names"]]
        concept_of = {n: c for c, n in enumerate(self.names)}
        entries_of: dict[int, list[int]] = defaultdict(list)
        for e, concepts in enumerate(self.table.entry_concepts):
            for c in concepts:
                entries_of[int(c)].append(e)
        self.entry_of_atom: dict[int, int] = {}
        self.atoms_of_concept: dict[str, list[int]] = defaultdict(list)
        for a, atom in enumerate(view.atomic_names):
            value = atom.split(":", 1)[1] if ":" in atom else atom
            if value in concept_of:
                self.atoms_of_concept[value].append(a)
            found = entries_of.get(concept_of.get(value, -1), [])
            if len(found) == 1:
                self.entry_of_atom[a] = found[0]
        self.pools: dict[str, Counter] = defaultdict(Counter)
        for r, f in zip(view.relations.tolist(), view.fillers.tolist()):
            self.pools[view.relation_names[r]][f] += 1
        single = [e for e in range(view.entry_count) if len(self.table.entry_concepts[e]) == 1]
        self.surfaces = canonical_surfaces(self.table, self.tokenizer, min_subtokens, single)
        self.min_subtokens = min_subtokens
        self.synthetic = {int(e) for e in ontology.get("synthetic_entries", ())}

    @classmethod
    def for_track(cls, track: str, family: str = "smollm2", *, min_subtokens: int = 2) -> "BuildContext":
        from transformers import AutoTokenizer

        from .e9_freqbias import ensure_exclusion_table
        from .e9_tracks import FAMILY_TOKENIZERS, ensure_alias_table, lexicon_for, track_spec
        if track not in TRACK_SPECS:
            raise ValueError(f"no understanding spec for {track!r} (have {sorted(TRACK_SPECS)})")
        spec = track_spec(track, family)
        ontology = torch.load(spec.ontology, weights_only=False)
        alias_path = ensure_alias_table(spec)
        return cls(track=track, family=family, ontology=ontology, table=cp.load_alias_table(alias_path),
                   lexicon=lexicon_for(spec, ontology), tokenizer=AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True),
                   tokenizer_name=FAMILY_TOKENIZERS[family], exclusions=torch.load(ensure_exclusion_table(track, family), weights_only=False),
                   families_spec=TRACK_SPECS[track], min_subtokens=min_subtokens, ontology_path=spec.ontology, alias_table_path=alias_path)

    def shown(self, entry: int) -> str:
        """The surface shown in prompts: the concept's own name (without a `synthetic:` prefix, in its casing) when it is
        an alias of the entry with ≥ ℓ_min subtokens; else the canonical alias (`e5_common.canonical_surfaces`)."""
        name = self.names[self.table.entry_concepts[entry][0]]
        bare = name.split(":", 1)[1] if name.startswith("synthetic:") else name
        options = [bare] + [self.view.text(a) for a in self.atoms_of_concept.get(name, ()) if self.view.text(a)]
        for option in options:            # the name, then the lexicon text of an atom naming the concept (T4: the ChEBI name)
            if self.table.alias_to_entry.get(normalize_alias(option)) == entry and \
                    len(self.tokenizer.encode(" " + option, add_special_tokens=False)) >= self.min_subtokens:
                return option
        surface = self.surfaces[entry]["surface"]
        return self.lexicon.display_surface(surface, name) if hasattr(self.lexicon, "display_surface") else surface

    def frame_by_relation(self, frame: Sequence[tuple[int, int]]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for r, f in frame:
            out[self.view.relation_names[r]].append(f)
        return {k: sorted(v) for k, v in out.items()}

    def closure_words(self, frame: Sequence[tuple[int, int]], entry: int | None) -> set[str]:
        """Stems of the content words of every verbalization within two hops of a frame (and of the entry's own aliases)."""
        data = self.exclusions
        atoms = {f for _, f in frame}
        for a in list(atoms):
            if a in self.entry_of_atom:
                atoms.update(f for _, f in self.view.frame(self.entry_of_atom[a]))
        ids = {i for a in atoms for i in data["atomic_surfaces"][a]}
        if entry is not None:
            ids.update(data["entry_surfaces"][entry])
        return word_stems(data["surfaces"][i] for i in ids)

    def text(self, atom: int) -> str | None:
        return self.view.text(atom)


def _new_word_concepts(ctx: BuildContext, items_dir: Path) -> list[dict[str, Any]]:
    """The E9 new words of a dimension-3 item directory as anchors (frames by name, resolved against the ontology)."""
    _, concepts, _ = edit.load_item_dir(items_dir, edit.SCHEMA_NEW)
    out = []
    for c in concepts:
        frame = edit.resolve_frame(c["frame"], ctx.view.relation_id, ctx.view.atomic_id)
        random_frame = edit.resolve_frame(c["random_frame"], ctx.view.relation_id, ctx.view.atomic_id)
        out.append({"concept": c["concept"], "subset": "new", "surface": c["surface"], "entry": None, "frame": frame,
                    "random_frame": random_frame, "source": c["concept"], "frequency": 0})
    return out


def select_concepts(ctx: BuildContext, *, counts: dict[str, int], new_items: Path | None, seed: int) -> list[dict[str, Any]]:
    """Anchor concepts per subset (module docstring), seeded."""
    view, spec = ctx.view, ctx.families_spec
    rng = random.Random(seed)
    anchor_rel = view.relation_id.get(spec["anchor"]["relation"])
    anchor_values = {view.atomic_id[a] for a in spec["anchor"]["values"] if a in view.atomic_id}
    frequency = view.frequency if view.frequency is not None else np.zeros(view.entry_count)
    pools: dict[str, list[int]] = defaultdict(list)
    for e, info in sorted(ctx.surfaces.items()):
        if not info["linkable"]:
            continue
        frame = view.frame(e)
        if anchor_rel is None or not any(r == anchor_rel and f in anchor_values for r, f in frame):
            continue
        if e in view.heldout:
            pools["heldout"].append(e)
        elif frequency[e] >= 10:
            pools["seen"].append(e)
        elif frequency[e] >= 1:
            pools["rare"].append(e)
    concepts = []
    for subset in ("seen", "rare", "heldout"):
        pool = pools[subset]
        k = counts.get(subset, len(pool))
        chosen = sorted(rng.sample(pool, min(k, len(pool))))
        for e in chosen:
            concepts.append({"concept": f"u-{ctx.track}-{e}", "subset": subset, "surface": ctx.shown(e), "entry": e,
                             "frame": view.frame(e), "random_frame": None, "source": ctx.names[ctx.table.entry_concepts[e][0]],
                             "frequency": int(frequency[e]), "split": "synthetic" if e in ctx.synthetic else subset})
    if new_items is not None and counts.get("new", 1):
        new = _new_word_concepts(ctx, new_items)
        k = counts.get("new", len(new))
        concepts += new[:k]
    return concepts


# ---------------------------------------------------------------- family builders


def _answer(ctx: BuildContext, relation: str, atom: int) -> str:
    text = ctx.text(atom)
    return ctx.lexicon.answer(relation, text) if relation in getattr(ctx.lexicon, "templates", {}) else " " + text


def _item(**fields: Any) -> dict[str, Any]:
    base = {"slots": {}, "text": {}, "null": {}, "pair": None, "meta": {}}
    base.update(fields)
    base["chance"] = 1.0 / len(base["candidates"])
    return base


def _distractor_texts(ctx: BuildContext, relation: str, exclude_atoms: set[int], exclude_texts: set[str], k: int,
                      rng: random.Random) -> list[int] | None:
    pool = sorted(a for a in ctx.pools.get(relation, {}) if a not in exclude_atoms and ctx.text(a) and ctx.text(a) not in exclude_texts)
    seen_texts, unique = set(), []
    for a in pool:
        if ctx.text(a) not in seen_texts:
            seen_texts.add(ctx.text(a)); unique.append(a)
    return rng.sample(unique, k) if len(unique) >= k else None


def build_two_hop(ctx: BuildContext, c: dict[str, Any], rng: random.Random, bridges: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    by_rel = ctx.frame_by_relation(c["frame"])
    frame_atoms = {f for _, f in c["frame"]}
    frame_texts = {ctx.text(f) for f in frame_atoms if ctx.text(f)}
    for path in ctx.families_spec["two_hop"]:
        r1, r2 = path["path"]
        for f1 in by_rel.get(r1, []):
            if f1 not in ctx.entry_of_atom:
                continue
            bridge_entry = ctx.entry_of_atom[f1]
            answers = [a for a in ctx.frame_by_relation(ctx.view.frame(bridge_entry)).get(r2, [])
                       if a not in frame_atoms and ctx.text(a) and ctx.text(a) not in frame_texts]
            if not answers:
                continue
            answer = answers[0]
            k = int(path.get("distractors", 4))
            wrong = _distractor_texts(ctx, r2, frame_atoms | {answer}, frame_texts | {ctx.text(answer)}, k, rng)
            if wrong is None:
                continue
            options = [answer] + wrong
            rng.shuffle(options)
            candidates = [_answer(ctx, r2, a) for a in options]
            gold = options.index(answer)
            key = f"{c['concept']}-two_hop-{r1}-{r2}"
            meta = {"path": [r1, r2], "bridge": ctx.names[ctx.table.entry_concepts[bridge_entry][0]], "answer": ctx.view.atomic_names[answer]}
            items.append(_item(id=key, family="two_hop", test="two_hop", subset=c["subset"], anchor=c["concept"], relation=f"{r1}>{r2}",
                               slots={"x": c["concept"]}, templates=path["templates"], null={"x": NULL_X}, candidates=candidates,
                               gold=gold, meta=meta))
            prompts = ctx.lexicon.prompts("*", r2)
            info = ctx.surfaces.get(bridge_entry)
            if prompts and info and info["linkable"]:
                bridge_id = f"u-{ctx.track}-bridge-{bridge_entry}"
                bridges.setdefault(bridge_entry, {"concept": bridge_id, "subset": "bridge", "surface": ctx.shown(bridge_entry),
                                                  "entry": bridge_entry, "frame": ctx.view.frame(bridge_entry), "random_frame": None,
                                                  "source": meta["bridge"], "frequency": int(ctx.view.frequency[bridge_entry])
                                                  if ctx.view.frequency is not None else 0, "role": "bridge"})
                items.append(_item(id=key.replace("-two_hop-", "-bridge-"), family="two_hop", test="bridge", subset=c["subset"],
                                   anchor=c["concept"], relation=r2, slots={"x": bridge_id}, templates=prompts, null={"x": NULL_X},
                                   candidates=candidates, gold=gold, meta=meta))
            hop1 = ctx.lexicon.prompts("*", r1)
            others = _distractor_texts(ctx, r1, frame_atoms, frame_texts, k, rng)
            if hop1 and others:
                options1 = [f1] + others
                rng.shuffle(options1)
                items.append(_item(id=key.replace("-two_hop-", "-hop1-"), family="two_hop", test="hop1", subset=c["subset"],
                                   anchor=c["concept"], relation=r1, slots={"x": c["concept"]}, templates=hop1, null={"x": NULL_X},
                                   candidates=[_answer(ctx, r1, a) for a in options1], gold=options1.index(f1), meta=meta))
            break                          # one item per path
    return items


def build_affordance(ctx: BuildContext, c: dict[str, Any], forbidden: set[str]) -> list[dict[str, Any]]:
    spec = ctx.families_spec["affordance"]
    by_rel = ctx.frame_by_relation(c["frame"])
    values = [ctx.view.atomic_names[a] for a in by_rel.get(spec["relation"], [])]
    items = []
    for name, block in spec["items"].items():
        keys = sorted(block["options"])
        hits = [v for v in values if v in block["options"]]
        if len(hits) != 1:
            continue
        candidates = [block["options"][k] for k in keys]
        if any(overlaps(o, forbidden) for o in candidates):
            continue
        items.append(_item(id=f"{c['concept']}-affordance-{name}", family="affordance", test="affordance", subset=c["subset"],
                           anchor=c["concept"], relation=f"{spec['relation']}:{name}", slots={"x": c["concept"]},
                           templates=block["templates"], null={"x": NULL_X}, candidates=candidates, gold=keys.index(hits[0]),
                           meta={"category": hits[0]}))
    return items


def build_paraphrase(ctx: BuildContext, c: dict[str, Any], forbidden: set[str], rng: random.Random) -> list[dict[str, Any]]:
    items = []
    by_rel = ctx.frame_by_relation(c["frame"])
    for relation, block in ctx.families_spec["paraphrase"].items():
        names = [ctx.view.atomic_names[a] for a in by_rel.get(relation, [])]
        mine = [n for n in names if n in block["options"] and n not in block.get("skip", [])]
        if not mine:
            continue
        gold_name = sorted(mine)[0] if len(mine) == 1 else rng.choice(sorted(mine))

        def phrase(value: str) -> str | None:
            return next((p for p in block["options"][value] if not overlaps(p, forbidden)), None)

        right = phrase(gold_name)
        if right is None:
            continue
        others = [v for v in sorted(block["options"]) if v not in names]
        if block.get("same_prefix"):
            prefix = gold_name.split(":", 1)[1][:1]
            others = [v for v in others if v.split(":", 1)[1][:1] == prefix]
        usable = [(v, phrase(v)) for v in others]
        usable = [(v, p) for v, p in usable if p is not None and p != right]
        k = block["k"] - 1
        if len(usable) < min(k, 2):
            continue
        chosen = rng.sample(usable, min(k, len(usable)))
        options = [(gold_name, right)] + chosen
        rng.shuffle(options)
        items.append(_item(id=f"{c['concept']}-paraphrase-{relation}", family="paraphrase", test="paraphrase", subset=c["subset"],
                           anchor=c["concept"], relation=relation, slots={"x": c["concept"]}, templates=block["templates"],
                           null={"x": NULL_X}, candidates=[p for _, p in options], gold=[v for v, _ in options].index(gold_name),
                           meta={"value": gold_name}))
    return items


def _partners(ctx: BuildContext, c: dict[str, Any], pool: Sequence[dict[str, Any]], relation: str, *, same: bool,
              match: str | None, rng: random.Random) -> dict[str, Any] | None:
    mine = set(ctx.frame_by_relation(c["frame"]).get(relation, []))
    if not mine:
        return None
    kind = set(ctx.frame_by_relation(c["frame"]).get(match, [])) if match else set()
    options = []
    for other in pool:
        if other["concept"] == c["concept"] or other["surface"].lower() == c["surface"].lower():
            continue
        theirs = set(ctx.frame_by_relation(other["frame"]).get(relation, []))
        if not theirs:
            continue
        if match and set(ctx.frame_by_relation(other["frame"]).get(match, [])) != kind:
            continue
        if (same and theirs & mine) or (not same and not theirs & mine):
            options.append(other)
    return rng.choice(options) if options else None


def build_reverse(ctx: BuildContext, c: dict[str, Any], pool: Sequence[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    spec = ctx.families_spec
    items = []
    by_rel = ctx.frame_by_relation(c["frame"])
    for relation, block in spec["reverse"].items():
        if not by_rel.get(relation):
            continue
        other = _partners(ctx, c, pool, relation, same=False, match=block.get("match"), rng=rng)
        if other is None:
            continue
        atom = rng.choice(by_rel[relation])
        text = ctx.text(atom)
        if not text:
            continue
        if block.get("article"):
            text = ("an " if text[:1].lower() in "aeiou" else "a ") + text
        cue = block["cue"].format(t=text)
        items.append(_item(id=f"{c['concept']}-reverse-{relation}", family="reverse", test="reverse", subset=c["subset"],
                           anchor=c["concept"], relation=relation, slots={"x": c["concept"], "y": other["concept"]},
                           text={"c": cue}, null={"c": NULL_CUE}, templates=spec["reverse_templates"], candidates=[" {x}", " {y}"],
                           gold=0, meta={"filler": ctx.view.atomic_names[atom], "partner": other["concept"]}))
    return items


def build_comparison(ctx: BuildContext, c: dict[str, Any], pool: Sequence[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    items = []
    for relation, block in ctx.families_spec["comparison"].items():
        same = _partners(ctx, c, pool, relation, same=True, match=None, rng=rng)
        different = _partners(ctx, c, pool, relation, same=False, match=None, rng=rng)
        if same is None or different is None:               # balanced: both or neither
            continue
        for label, other, gold in (("same", same, 0), ("different", different, 1)):
            items.append(_item(id=f"{c['concept']}-comparison-{relation}-{label}", family="comparison", test="comparison",
                               subset=c["subset"], anchor=c["concept"], relation=relation, slots={"x": c["concept"], "y": other["concept"]},
                               null={"x": NULL_X, "y": NULL_Y}, templates=block["templates"], candidates=block["candidates"], gold=gold,
                               meta={"partner": other["concept"], "label": label}))
    return items


def build_negation(ctx: BuildContext, c: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    items = []
    by_rel = ctx.frame_by_relation(c["frame"])
    frame_atoms = {f for _, f in c["frame"]}
    frame_texts = {ctx.text(f) for f in frame_atoms if ctx.text(f)}
    for relation, block in ctx.families_spec["negation"].items():
        true = [a for a in by_rel.get(relation, []) if ctx.text(a)]
        if not true:
            continue
        right = rng.choice(true)
        wrong = _distractor_texts(ctx, relation, frame_atoms, frame_texts, 1, rng)
        if wrong is None:
            continue
        options = [right, wrong[0]]
        rng.shuffle(options)
        candidates = [_answer(ctx, relation, a) for a in options]
        pair = f"{c['concept']}-negation-{relation}"
        meta = {"true": ctx.view.atomic_names[right], "false": ctx.view.atomic_names[wrong[0]]}
        for test, templates, gold in (("affirm", block["affirm"], options.index(right)), ("negate", block["negate"], options.index(wrong[0]))):
            items.append(_item(id=f"{pair}-{test}", family="negation", test=test, subset=c["subset"], anchor=c["concept"],
                               relation=relation, slots={"x": c["concept"]}, null={"x": NULL_X}, templates=templates,
                               candidates=candidates, gold=gold, pair=pair, meta=meta))
    return items


def build_items(track: str, output: Path, *, family: str = "smollm2", counts: dict[str, int] | None = None,
                new_items: Path | None | str = "auto", seed: int = 0, families: Sequence[str] = FAMILIES) -> dict[str, Any]:
    """The understanding item directory of a track (module docstring)."""
    if track not in TRACK_SPECS:
        raise ValueError(f"no understanding spec for {track!r} (have {sorted(TRACK_SPECS)})")
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError(f"{output} is not empty")
    if new_items == "auto":
        candidates = [ITEMS_ROOT / f"new-words-{track}-{family}-v2", ITEMS_ROOT / f"new-words-{track}-{family}-v1"]
        new_items = next((p for p in candidates if (p / "manifest.json").exists()), None)
    return build_items_from_context(BuildContext.for_track(track, family), output, counts=counts,
                                    new_items=Path(new_items) if new_items else None, seed=seed, families=families)


def build_items_from_context(ctx: BuildContext, output: Path, *, counts: dict[str, int] | None = None, new_items: Path | None = None,
                             seed: int = 0, families: Sequence[str] = FAMILIES) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    counts = {"seen": 300, "rare": 300, **(counts or {})}
    track, family = ctx.track, ctx.family
    concepts = select_concepts(ctx, counts=counts, new_items=new_items, seed=seed)
    rng = random.Random(seed + 1)
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in concepts:
        by_subset[c["subset"]].append(c)
    items: list[dict[str, Any]] = []
    bridges: dict[int, dict[str, Any]] = {}
    for c in concepts:
        forbidden = ctx.closure_words(c["frame"], c["entry"])
        pool = by_subset[c["subset"]]
        if "two_hop" in families:
            items += build_two_hop(ctx, c, rng, bridges)
        if "affordance" in families:
            items += build_affordance(ctx, c, forbidden)
        if "paraphrase" in families:
            items += build_paraphrase(ctx, c, forbidden, rng)
        if "reverse" in families:
            items += build_reverse(ctx, c, pool, rng)
        if "comparison" in families:
            items += build_comparison(ctx, c, pool, rng)
        if "negation" in families:
            items += build_negation(ctx, c, rng)
    used = {cid for item in items for cid in item["slots"].values()}
    records = [c for c in concepts if c["concept"] in used] + [b for b in bridges.values() if b["concept"] in used]
    for record in records:
        record["frame"] = ctx.view.readable(record["frame"])
        if record.get("random_frame") is not None:
            record["random_frame"] = ctx.view.readable(record["random_frame"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "concepts.jsonl").write_text("".join(json.dumps(json_ready(r)) + "\n" for r in records))
    write_jsonl_gz(output / "items.jsonl.gz", items)
    table = Counter((i["subset"], i["family"], i["test"]) for i in items)
    anchors = Counter((c["subset"]) for c in records if c.get("role") != "bridge")
    manifest = {"schema": SCHEMA, "track": track, "family": family, "tokenizer": ctx.tokenizer_name, "seed": seed,
                "ontology": str(ctx.ontology_path), "ontology_alias_sha256": ctx.table.digest(), "alias_table": str(ctx.alias_table_path),
                "new_items": str(new_items) if new_items else None, "counts_requested": counts, "anchors": dict(sorted(anchors.items())),
                "families": list(families), "items": len(items),
                "items_by": {f"{s}/{f}/{t}": n for (s, f, t), n in sorted(table.items())},
                "rules": {"overlap": "an option sharing a content-word stem (≥ 3 letters, not a stopword) with any verbalization within "
                                     "two hops of the term (exclusion table) is not used",
                          "two_hop": "answer = the first sorted r2 filler of the r1 filler's frame that is not in the term's frame (atom "
                                     "or text); distractors = other r2 fillers not in the term's frame",
                          "scoring": "PMI against the null prompt (term → 'this'/'that'; reverse: cue → 'we mean'); argmax per template"},
                "spec": json_ready(ctx.families_spec)}
    import hashlib
    manifest["sha256"] = {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in ("concepts.jsonl", "items.jsonl.gz")}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_items(items_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    items_dir = Path(items_dir)
    manifest = json.loads((items_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{items_dir} is not an {SCHEMA} item directory")
    concepts, items = read_jsonl(items_dir / "concepts.jsonl"), read_jsonl(items_dir / "items.jsonl")
    known = {c["concept"] for c in concepts}
    for item in items:
        if any(cid not in known for cid in item["slots"].values()):
            raise ValueError(f"item {item['id']} refers to an unknown concept")
        if not 0 <= int(item["gold"]) < len(item["candidates"]):
            raise ValueError(f"item {item['id']} is malformed")
    return manifest, concepts, items


# ---------------------------------------------------------------- evaluation


SLOT = re.compile(r"\{(x|y|c)\}")


def render(template: str, fills: dict[str, str]) -> str:
    """Fill the slots `{x}`, `{y}`, `{c}`; every other brace is literal (chemical names hold braces)."""
    return SLOT.sub(lambda m: fills[m.group(1)], template)


def item_prompts(item: dict[str, Any], surfaces: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(real, null) (prefix, continuation) pairs, template-major then candidate."""
    fills = {slot: surfaces[cid] for slot, cid in item["slots"].items()} | dict(item["text"])
    null_fills = fills | dict(item["null"])
    real, null = [], []
    for template in item["templates"]:
        for candidate in item["candidates"]:
            continuation = render(candidate, fills)
            real.append((render(template, fills), continuation))
            null.append((render(template, null_fills), continuation))
    return real, null


def score_items(adapter: cp.ModelAdapter, items: Sequence[dict[str, Any]], surfaces: dict[str, str],
                null_cache: dict[tuple[str, str], float] | None = None) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    """Per item: PMI per template × candidate, correctness (mean over templates), all-templates-correct and agreement."""
    real, null, shapes = [], [], []
    for item in items:
        r, n = item_prompts(item, surfaces)
        real += r; null += n
        shapes.append((len(item["templates"]), len(item["candidates"])))
    raw = cp.continuation_logprob(adapter, [p for p, _ in real], [c for _, c in real]) if real else np.zeros(0)
    cache = dict(null_cache or {})
    missing = sorted(set(null) - set(cache))
    if missing:
        cache.update(zip(missing, cp.continuation_logprob(adapter, [p for p, _ in missing], [c for _, c in missing]).tolist()))
    pmi = raw - np.asarray([cache[pair] for pair in null])
    out, cursor = [], 0
    for item, (n_t, k) in zip(items, shapes):
        scores = pmi[cursor:cursor + n_t * k].reshape(n_t, k)
        cursor += n_t * k
        gold = int(item["gold"])
        argmax = scores.argmax(1)
        correct = (argmax == gold).astype(float)
        out.append({"id": item["id"], "family": item["family"], "test": item["test"], "subset": item["subset"], "anchor": item["anchor"],
                    "relation": item["relation"], "pair": item.get("pair"), "chance": item["chance"],
                    "correct": float(correct.mean()), "per_template": correct.astype(int).tolist(),
                    "consistent": int(len(set(argmax.tolist())) == 1) if n_t > 1 else None,
                    "margin": float(np.mean([s[gold] - np.max(np.delete(s, gold)) for s in scores]))})
    return out, cache


def _entry_ids(run: E5Run, concepts: Sequence[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Entry id per concept: existing entries as recorded; new words get the next ids (inserted at evaluation)."""
    base = int(run.ontology["entry_count"])
    new = [c for c in concepts if c["entry"] is None]
    ids = {c["concept"]: int(c["entry"]) for c in concepts if c["entry"] is not None}
    ids.update({c["concept"]: base + i for i, c in enumerate(new)})
    return ids, new


def check_links(adapter: cp.ChannelModelAdapter, concepts: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]],
                ids: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Per concept: does its surface inject its entry where it stands in the first prompt that uses it?"""
    first: dict[str, tuple[str, int, int]] = {}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    for item in items:
        for slot, cid in item["slots"].items():
            if cid in first:
                continue
            fills = {s: surfaces[c] for s, c in item["slots"].items()} | dict(item["text"])
            template = item["templates"][0]
            marker = "\x00"
            text = render(template, {**fills, slot: marker})
            start = text.index(marker)
            first[cid] = (render(template, fills), start, start + len(surfaces[cid]))
    texts = [first[c["concept"]][0] if c["concept"] in first else c["surface"] for c in concepts]
    spans = [first[c["concept"]][1:] if c["concept"] in first else (0, len(c["surface"])) for c in concepts]
    found = adapter.link_targets(texts, spans) if adapter.linker is not None else [[] for _ in concepts]
    out = {}
    for c, entries in zip(concepts, found):
        want = ids[c["concept"]]
        status = "linked" if entries == [want] else "unlinked" if not entries else "mislinked"
        out[c["concept"]] = {"entry": want, "found": entries, "status": status}
    return out


def available_sources(run: E5Run, requested: Sequence[str] | None) -> list[str]:
    mode = run.mode
    allowed = ["own"] if mode in {"none", "hashed"} else ["own", "none"] + (["random_frame"] if mode == "compose" else [])
    return [s for s in (requested or SOURCES) if s in allowed]


def evaluate(run: E5Run, items_dir: Path, *, sources: Sequence[str] | None = None, seed: int = 0, limit_anchors: int | None = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score every item under each source (module docstring). `limit_anchors` (smoke tests only) keeps the items of the
    first N anchors of each subset."""
    from .e9_tracks import random_frames, replaced_frames
    manifest, concepts, items = load_items(items_dir)
    if limit_anchors:
        firsts: dict[str, list[str]] = defaultdict(list)
        for item in items:
            if item["anchor"] not in firsts[item["subset"]] and len(firsts[item["subset"]]) < limit_anchors:
                firsts[item["subset"]].append(item["anchor"])
        keep = {a for anchors in firsts.values() for a in anchors}
        items = [i for i in items if i["anchor"] in keep]
        used = {cid for i in items for cid in i["slots"].values()}
        concepts = [c for c in concepts if c["concept"] in used]
    ontology = run.ontology
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    ids, new = _entry_ids(run, concepts)
    frames = [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in new]
    new_random = [edit.resolve_frame(c["random_frame"], relation_id, atomic_id) for c in new]
    adapter = edit.extended_adapter(run, {c["surface"]: ids[c["concept"]] for c in new}) if new else run.adapter
    resolved = check_links(adapter, concepts, items, ids)
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    roles = {c["concept"]: c.get("role") for c in concepts}
    anchors = sorted({ids[cid] for item in items for cid in item["slots"].values() if roles[cid] != "bridge"})
    existing_anchors = [e for e in anchors if e < int(ontology["entry_count"])]
    sources = available_sources(run, sources)
    channel = run.channel
    started = time.monotonic()
    results: dict[str, list[dict[str, Any]]] = {}
    null_cache: dict[tuple[str, str], float] | None = None
    for source in sources:
        log(f"  understanding: source {source}")
        insert = new_random if source == "random_frame" else frames
        replace = random_frames(ontology, existing_anchors, seed=seed) if source == "random_frame" else {}
        zero = None
        if source == "none" and channel is not None:
            width = channel.gate.in_features // 2
            zero = {e: torch.zeros(width) for e in anchors}
        with edit.inserted_entries(channel, len(new), insert, seed=seed) if new else contextlib.nullcontext(), \
                replaced_frames(channel, replace), (override_rows(channel, zero) if zero else contextlib.nullcontext()):
            # reference tests (hop1, bridge) only under `own`: the bridge does not involve the anchor at all
            chosen = items if source == "own" else [i for i in items if i["test"] not in REFERENCE_TESTS]
            rows, null_cache = score_items(adapter, chosen, surfaces, null_cache)
        results[source] = rows
    return {"manifest": manifest, "resolved": resolved, "sources": sources, "results": results,
            "seconds": time.monotonic() - started}


# ---------------------------------------------------------------- per-concept scores and summaries


def concept_family_scores(rows: Sequence[dict[str, Any]], keep: set[str] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """test → anchor → score: mean of (correct − chance) over the anchor's items of that test; `negation` = pair
    consistency (affirm and negate correct under the same template) − 0.25; `composite` = mean over the six families."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if keep is not None and r["anchor"] not in keep:
            continue
        if r["family"] == "negation":
            pairs[r["pair"]][r["test"]] = r
            by[r["test"]][r["anchor"]].append(r["correct"] - r["chance"])
            continue
        by[r["test"]][r["anchor"]].append(r["correct"] - r["chance"])
    for pair, parts in pairs.items():
        if "affirm" in parts and "negate" in parts:
            both = np.mean([a * b for a, b in zip(parts["affirm"]["per_template"], parts["negate"]["per_template"])])
            by["negation"][parts["affirm"]["anchor"]].append(float(both) - 0.25)
    out = {test: {a: float(np.mean(v)) for a, v in anchors.items()} for test, anchors in by.items()}
    families = [f for f in FAMILIES if f in out]
    composite: dict[str, list[float]] = defaultdict(list)
    for f in families:
        for a, v in out[f].items():
            composite[a].append(v)
    out["composite"] = {a: float(np.mean(v)) for a, v in composite.items()}
    return out


SUMMARY_TESTS = ("composite", *FAMILIES, "affirm", "negate", *REFERENCE_TESTS)


def summarize(evaluation: dict[str, Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per source × subset × test the mean score (correct − chance) and accuracy; `own` − other sources (paired over
    anchors; Holm over the six families within a subset)."""
    resolved = evaluation["resolved"]
    linked = {c for c, r in resolved.items() if r["status"] == "linked"}
    out: dict[str, Any] = {"concepts": len(resolved), "linked": len(linked),
                           "status_counts": dict(sorted(Counter(r["status"] for r in resolved.values()).items())), "sources": {},
                           "comparisons": []}
    subsets = sorted({r["subset"] for rows in evaluation["results"].values() for r in rows})
    for source, rows in evaluation["results"].items():
        good = [r for r in rows if _linked_item(r, linked, evaluation)]
        block = {}
        for subset in subsets:
            sub = [r for r in good if r["subset"] == subset]
            scores = concept_family_scores(sub)
            block[subset] = {t: {"mean": float(np.mean(list(scores[t].values()))) if scores.get(t) else None,
                                 "anchors": len(scores.get(t, {})),
                                 "accuracy": float(np.mean([r["correct"] for r in sub if r["test"] == t])) if any(r["test"] == t for r in sub) else None,
                                 "items": sum(r["test"] == t for r in sub)}
                             for t in SUMMARY_TESTS if scores.get(t)}
        out["sources"][source] = block
    if "own" in evaluation["results"]:
        for source in evaluation["sources"]:
            if source == "own":
                continue
            for subset in subsets:
                a = concept_family_scores([r for r in evaluation["results"]["own"] if r["subset"] == subset and _linked_item(r, linked, evaluation)])
                b = concept_family_scores([r for r in evaluation["results"][source] if r["subset"] == subset and _linked_item(r, linked, evaluation)])
                block = []
                for test in ("composite", *FAMILIES):
                    common = sorted(set(a.get(test, {})) & set(b.get(test, {})))
                    if len(common) < 2:
                        continue
                    ci = zs.paired_difference(np.asarray([a[test][x] for x in common]), np.asarray([b[test][x] for x in common]),
                                              resamples=resamples, seed=seed)
                    block.append({"subset": subset, "test": test, "baseline": source, **ci})
                families = [r for r in block if r["test"] != "composite"]
                for row, adjusted in zip(families, holm_adjust([r["p_value"] for r in families]) if families else []):
                    row.update(p_holm=adjusted, significant=adjusted < 0.05)
                out["comparisons"] += block
    return out


def _linked_item(row: dict[str, Any], linked: set[str], evaluation: dict[str, Any]) -> bool:
    return row["anchor"] in linked


def render_report(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E9 understanding items — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Items `{header['items']}`: {summary['linked']} of {summary['concepts']} concepts link ({summary['status_counts']}). "
             "Score = correct − chance (mean over an anchor's items, then over anchors); `negation` = affirm-and-negate pair "
             "consistency − 0.25; `composite` = mean of the six families; accuracy = raw share correct.", ""]
    for name, block in summary["sources"].items():
        lines += [f"## source `{name}`", "", "| subset | test | anchors | items | score | accuracy |", "|---|---|---:|---:|---:|---:|"]
        for subset, tests in block.items():
            for test, v in tests.items():
                acc = "n/a" if v["accuracy"] is None else f"{v['accuracy']:.3f}"
                lines.append(f"| {subset} | {test} | {v['anchors']} | {v['items']} | {v['mean']:+.3f} | {acc} |")
        lines.append("")
    if summary["comparisons"]:
        lines += ["## `own` − source (paired over anchors; Holm over the six families within a subset)", "",
                  "| subset | test | baseline | difference [95% CI] | n | p (Holm) |", "|---|---|---|---|---:|---:|"]
        for c in summary["comparisons"]:
            holm = c.get("p_holm")
            lines.append(f"| {c['subset']} | {c['test']} | {c['baseline']} | {c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                         f"{c['n']} | {'n/a' if holm is None else f'{holm:.4f}'} |")
    return "\n".join(lines) + "\n"


def output_folder(run_dir: Path, items: Path | str) -> Path:
    """`RUN/<items directory name>` (prefixed with `understanding-` unless the name has it)."""
    name = Path(items).name
    return Path(run_dir) / (name if name.startswith(OUTPUT_PREFIX) else f"{OUTPUT_PREFIX}-{name}")


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    manifest = json.loads((Path(args.items) / "manifest.json").read_text())
    requested = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    output = args.output or output_folder(args.run, args.items)
    alias_table = args.alias_table or ensure_alias_table(track_spec(manifest["track"], manifest["family"]))
    config = {"experiment": "e9-understanding", "run": str(args.run), "items": str(args.items), "sources": requested, "seed": args.seed,
              "limit_anchors": args.limit_anchors,
              "resamples": args.resamples, "alias_table": str(alias_table) if alias_table else None, "batch_size": args.batch_size}
    if args.overwrite:
        clear_output(output)
    git_at_start = start_output(output, config)
    run = open_run(args.run, device=args.device, batch_size=args.batch_size, alias_table=alias_table)
    evaluation = evaluate(run, args.items, sources=requested, seed=args.seed, limit_anchors=args.limit_anchors)
    summary = summarize(evaluation, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "items": str(args.items), "track": manifest["track"]}
    write_jsonl_gz(output / "predictions.jsonl.gz", ({"source": source, **{k: v for k, v in row.items() if k != "relation"}}
                                                     for source, rows in evaluation["results"].items() for row in rows))
    write_json(output / "summary.json", {**header, "resolved": evaluation["resolved"], "summary": summary,
                                         "seconds": evaluation["seconds"], "items_manifest_sha256": manifest.get("sha256")})
    (output / "report.md").write_text(render_report(summary, header))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary, "output": str(output)}


# ---------------------------------------------------------------- analysis across runs (e9_report --understanding)


def load_evaluation(folder: Path) -> dict[str, Any] | None:
    folder = Path(folder)
    if not (folder / "summary.json").exists() or not any((folder / n).exists() for n in ("predictions.jsonl", "predictions.jsonl.gz")):
        return None
    document = json.loads((folder / "summary.json").read_text())
    rows = read_jsonl(folder / "predictions.jsonl")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    linked = {c for c, r in document["resolved"].items() if r["status"] == "linked"}
    return {"rows": dict(by_source), "linked": linked, "document": document}


def run_scores(found: dict[str, Any], source: str = "own") -> dict[str, dict[str, dict[str, float]]] | None:
    """subset → test → anchor → score of one evaluation (linked anchors only)."""
    rows = found["rows"].get(source)
    if rows is None:
        return None
    out = {}
    for subset in sorted({r["subset"] for r in rows}):
        out[subset] = concept_family_scores([r for r in rows if r["subset"] == subset], keep=found["linked"])
    return out


def cross_model(evaluations: dict[str, dict[int, dict[str, Any]]], *, candidate: str = "C5", references: Sequence[str] = (),
                resamples: int = 2000, seed: int = 0, item_seed: bool = False, sources: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Candidate − reference per subset × test: anchors as the unit, scores seed-averaged per anchor (paired bootstrap
    over anchors; Holm over the six families within subset × reference), and with `item_seed` the concepts × seeds
    crossed model (`e9_power.analyse_table`). `sources` adds within-candidate contrasts (own − none / random_frame)."""
    from .e9_power import analyse_table, paired_table
    out: dict[str, Any] = {"available": candidate in evaluations, "comparisons": [], "means": {}}
    if candidate not in evaluations:
        return out
    scores = {m: {s: run_scores(e) for s, e in by_seed.items() if e is not None} for m, by_seed in evaluations.items()}
    for model, by_seed in scores.items():
        by_seed = {s: v for s, v in by_seed.items() if v}
        out["means"][model] = {}
        for subset in sorted({k for v in by_seed.values() for k in v}):
            out["means"][model][subset] = {t: float(np.mean([np.mean(list(v[subset][t].values())) for v in by_seed.values()
                                                             if subset in v and v[subset].get(t)]))
                                           for t in ("composite", *FAMILIES, "affirm", "negate", *REFERENCE_TESTS)
                                           if any(subset in v and v[subset].get(t) for v in by_seed.values())}
    contrasts = [(candidate, "own", r, "own") for r in references if r in scores and r != candidate]
    contrasts += [(candidate, "own", candidate, s) for _, s in sources]
    for model_a, source_a, model_b, source_b in contrasts:
        if source_b != "own":
            b_scores = {s: run_scores(e, source_b) for s, e in evaluations[model_b].items() if e is not None}
        else:
            b_scores = scores[model_b]
        a_scores = scores[model_a]
        if len(b_scores) == 1 and len(a_scores) > 1:                       # P0: one run stands for every seed
            only = next(iter(b_scores.values()))
            b_scores = {s: only for s in a_scores}
        seeds = sorted(s for s in set(a_scores) & set(b_scores) if a_scores[s] and b_scores[s])
        if not seeds:
            continue
        label = f"{model_a} − {model_b}" if source_b == "own" else f"{model_a} own − {source_b}"
        block = []
        for subset in sorted(set.intersection(*(set(a_scores[s]) & set(b_scores[s]) for s in seeds))):
            for test in ("composite", *FAMILIES):
                ca = {s: a_scores[s][subset].get(test, {}) for s in seeds}
                cb = {s: b_scores[s][subset].get(test, {}) for s in seeds}
                anchors, _, table = paired_table(ca, cb)
                if len(anchors) < 2:
                    continue
                averaged = table.mean(1)
                ci = zs.paired_difference(averaged, np.zeros_like(averaged), resamples=resamples, seed=seed)
                row = {"comparison": label, "subset": subset, "test": test, "seeds": seeds, "anchors": len(anchors), **ci}
                if item_seed:
                    row["item_seed"] = analyse_table(table, resamples=resamples, seed=seed)
                block.append(row)
        for subset in {r["subset"] for r in block}:
            fam = [r for r in block if r["subset"] == subset and r["test"] != "composite"]
            for row, adjusted in zip(fam, holm_adjust([r["p_value"] for r in fam]) if fam else []):
                row.update(p_holm=adjusted, significant=adjusted < 0.05)
        out["comparisons"] += block
    return out


# ---------------------------------------------------------------- queueing


def evaluate_command(run_dir: Path, items_dir: Path, *, python: str = sys.executable, batch_size: int | None = None,
                     sources: Sequence[str] | None = None) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_understanding", "evaluate", "--run", str(run_dir), "--items", str(items_dir),
            "--overwrite", *(["--batch-size", str(batch_size)] if batch_size else []),
            *(["--sources", ",".join(sources)] if sources else [])]


def queue_stage(stage: str, items_dir: Path, *, priority: int = 56, models: Sequence[str] | None = None,
                seeds: Sequence[int] | None = None, root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None,
                dry_run: bool = False, candidate: str = "C5", candidate_sources: Sequence[str] = SOURCES) -> list[dict[str, Any]]:
    """One evaluation job per config of `stage` (idempotent; named `<stage>-<stem>-<output folder>`). The candidate is scored
    under `candidate_sources` (its within-model controls); every other model under `own` only (pre-registration B1)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import EVAL_JOB_BATCH, _config_host, stage_python
    from .e9_rescore import model_of
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        model = model_of(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        config = yaml.safe_load(path.read_text())
        host = _config_host(config) or ""
        batch = {"Qwen3-0.6B-Base": 16, "Qwen3-1.7B-Base": 8, "Qwen3-4B-Base": 4}.get(host, 32 if "360M" in host else 64)
        run_dir = Path(root) / "runs" / stage / path.stem
        sources = list(candidate_sources) if model == candidate else ["own"]
        jobs.append({"name": f"{stage}-{path.stem}-{output_folder(run_dir, items_dir).name}", "priority": priority,
                     "command": evaluate_command(run_dir, items_dir, python=python, batch_size=EVAL_JOB_BATCH.get(host, batch),
                                                 sources=sources)})
    if dry_run:
        return jobs
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"], min_free_gb=5,
                env={"PYTHONPATH": "src"}, resume_args=[])
            queued.append(job)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("items", help="build a track's understanding items")
    build.add_argument("--track", required=True, choices=sorted(TRACK_SPECS)); build.add_argument("--family", default="smollm2")
    build.add_argument("--output", type=Path, required=True); build.add_argument("--seed", type=int, default=0)
    build.add_argument("--new-items", default="auto", help="dimension-3 new-word item directory, 'auto' (v2, else v1) or 'none'")
    build.add_argument("--count", action="append", default=[], help="subset=N anchors (defaults: seen=300 rare=300, heldout and new: all)")
    build.add_argument("--families", default=",".join(FAMILIES))
    ev = sub.add_parser("evaluate", help="score a finished run on an understanding item directory")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--output", type=Path, default=None); ev.add_argument("--alias-table", type=Path, default=None)
    ev.add_argument("--sources", default=""); ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--resamples", type=int, default=2000); ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--device", default=None); ev.add_argument("--overwrite", action="store_true")
    ev.add_argument("--limit-anchors", type=int, default=None, help="smoke tests only: the items of the first N anchors per subset")
    queue = sub.add_parser("queue", help="queue one evaluation per config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--items", type=Path, required=True)
    queue.add_argument("--priority", type=int, default=56); queue.add_argument("--models", nargs="*", default=None)
    queue.add_argument("--seeds", type=int, nargs="*", default=None); queue.add_argument("--root", type=Path, default=ROOT)
    queue.add_argument("--candidate-sources", default=",".join(SOURCES), help="sources of C5's job (other models: own)")
    queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "items":
        counts = {k: int(v) for k, v in (c.split("=", 1) for c in args.count)}
        new_items: Any = None if args.new_items == "none" else ("auto" if args.new_items == "auto" else Path(args.new_items))
        manifest = build_items(args.track, args.output, family=args.family, counts=counts, new_items=new_items, seed=args.seed,
                               families=[f.strip() for f in args.families.split(",") if f.strip()])
        print(json.dumps({k: manifest[k] for k in ("anchors", "items", "items_by")}, indent=2))
    elif args.command == "evaluate":
        result = run_evaluate(args)
        print(json.dumps({"output": result["output"], "composite": {s: {sub: v.get("composite", {}).get("mean") for sub, v in block.items()}
                                                                    for s, block in result["summary"]["sources"].items()}}, indent=2))
    else:
        jobs = queue_stage(args.stage, args.items, priority=args.priority, models=args.models, seeds=args.seeds, root=args.root,
                           dry_run=args.dry_run, candidate_sources=[x for x in args.candidate_sources.split(",") if x])
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
