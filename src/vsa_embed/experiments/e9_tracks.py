"""E9 on vocabulary that is new or rare for the host: the application-track corpora (orchestrator decision
after the engagement check, 2026-10-02: SmolLM2 already models general WordNet words — inside-span loss
0.92 — so C5 matched C0′ there; WordNet stays as the secondary / negative-control corpus).

Tracks (SmolLM2-tokenized corpora; order of priority):

- `t5` — T5 enterprise glossary (WP-C7): invented organisation terms, contamination-free; held-out terms
  and 200 zero-shot terms ("defined only by frames", in no document) are entries of the ontology.
- `t4` — T4 chemistry (WP-C7): ChEBI names (long, rare IUPAC-style names) and synthetic compounds.
- `t1` — T1-open (WP-T1): MeSH descriptors on PubMed; evaluation on `eval-pubmed` with 2,048 windows
  (open decision 12); no synthetic concepts.
- `t1c` — T1c clinical (decision 58): SNOMED CT International 2022-05-31 on MIMIC-III notes (`t1c_corpus`);
  evaluation on `eval-mimic` (held-out patients) with 2,048 windows. **Licensed**: its alias table, holdout
  list and dimension-3 items live under `~/data/vsa-llm/t1c/` (`TrackSpec.alias_table_dir`, `items_root`), never
  in the repository; WP-C7-format zero-shot items on held-out terms (`t1c_corpus --stage zeroshot-items`).
- `wordnet` — the C3 WordNet host corpus (the original E9 design).
- `t7` — T7 new biomedical vocabulary (author decision 55): MeSH supplementary concept records and descriptors whose
  names are frequent in PubMed 2025–26 and absent from general web text, linked in the abstracts that mention them
  (`ontologies/mesh_novel.py`, built by `t1_open_corpus` with the `mesh_novel` adapter); evaluation on
  `eval-pubmed`; no WP-C7 zero-shot items (dimension 3 = invented new words + edits, as T1).

Host tokenizer families (WP-Qwen): every track is built for SmolLM2 (`data_root`) and, relinked with the same
ontology, alias table and holdout, for the Qwen3 base tokenizer (`TrackSpec.for_family("qwen3")`, `QWEN3_ROOTS`)
and the Qwen3.5 one (`"qwen3_5"`, `QWEN35_ROOTS`; WP-Qwen35, built and read with the Qwen3.5 environment);
the WP-C7 zero-shot items and the alias table are tokenizer-independent and shared.

Each track gives the trainer its corpora (`data.train`, `data.eval`, `data.ontology`), a general-text
corpus for locality and general-text quantization damage (`eval-general`; none for WordNet, whose
evaluation corpus is general text), the evaluation alias table, the wording of the dimension-3 items
(`TrackLexicon`: the track's own relation templates and filler names) and its WP-C7 zero-shot items.

**Alias tables.** `channel_probes.load_run` needs the full evaluation alias table; the track corpora do
not store it, so `track_alias_table` replays the track's ontology adapter (`tracks.load_track(...).ontology()`,
plus the synthetic concepts of `items/synthetic_concepts.jsonl` and the frozen holdout of
`holdout_concepts.txt`; for T1 `t1_open_corpus.build_track_ontology`) and refuses a table whose digest
differs from the ontology's `alias_table_sha256` (all three replay exactly). Tables are written once to
`~/data/vsa-llm/e9/alias-tables/<track>.json` (T4 and T1 have 10⁵ aliases; not committed).

**WP-C7 zero-shot items** (`zeroshot_property`: multiple choice, 3 paraphrases per group;
`zeroshot_entailment`: a true and a corrupted frame statement) are scored as E5.4 items (PMI against
the null surface "this"; a statement pair becomes a 2-choice item on the statements' common prefix)
on the synthetic (zero-shot) and held-out terms, under the dimension-3 sources: `own` (the composed
frame; C2 fallback; nothing for C0′/P0), `none` (name only), `mean_row` and `random_frame` (the entries'
frames replaced by random frames of equal degree, same relations, fillers drawn frequency-weighted
from each relation's fillers).

    python -m vsa_embed.experiments.e9_tracks alias-table --track t5 [--output PATH]
    python -m vsa_embed.experiments.e9_tracks items --track t5 --kind new|edits [--family qwen3] --output DIR
    python -m vsa_embed.experiments.e9_tracks zeroshot --run RUN --track t5 --output OUT [--quantize int4]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from ..compose import FrameSchedule
from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable, normalize_alias
from ..statistics import holm_adjust
from ..tracks.common import RelationTemplates
from . import e5_zeroshot as zs
from . import e9_ontology_edit as edit
from .e5_common import E5Run, clear_output, finish_output, fmt, json_ready, open_run, override_rows, start_output, write_json

ALIAS_TABLE_DIR = Path("~/data/vsa-llm/e9/alias-tables").expanduser()
TRACK_SOURCES = ("own", "none", "mean_row", "random_frame")
SPLITS = ("synthetic", "heldout")
TESTS = ("property", "paraphrase", "entailment")
NULL_SURFACE = "this"

# T1-open has no relation templates of its own (WP-T1 built probes, not cloze items); these word the
# dimension-3 items on MeSH frames (two property paraphrases and a statement each).
# T1c (SNOMED CT) relation wordings for the dimension-3 items: two property paraphrases and a statement each, for the
# attribute relations with readable body-structure / organism / substance / procedure fillers. Category relations
# (`hierarchy`, `semantic_tag`) are kept, never templated.
T1C_TEMPLATES = {
    "is_a": RelationTemplates(["{x} is a kind of", "In SNOMED CT, {x} is classified under"], "{x} is a type of {y}."),
    "finding_site": RelationTemplates(["The finding site of {x} is", "{x} is found in"], "{x} has the finding site {y}."),
    "associated_morphology": RelationTemplates(["The associated morphology of {x} is", "{x} involves the morphology"],
                                               "{x} has the associated morphology {y}."),
    "causative_agent": RelationTemplates(["The causative agent of {x} is", "{x} is caused by"], "{x} has the causative agent {y}."),
    "procedure_site_direct": RelationTemplates(["The direct procedure site of {x} is", "{x} is performed directly on"],
                                               "{x} has the direct procedure site {y}."),
    "procedure_site_indirect": RelationTemplates(["The indirect procedure site of {x} is", "{x} is performed indirectly on"],
                                                 "{x} has the indirect procedure site {y}."),
    "procedure_site": RelationTemplates(["The procedure site of {x} is", "{x} is performed on"], "{x} has the procedure site {y}."),
    "method": RelationTemplates(["The method of {x} is", "{x} is carried out by"], "{x} has the method {y}."),
    "has_active_ingredient": RelationTemplates(["The active ingredient of {x} is", "{x} contains the active ingredient"],
                                               "{x} has the active ingredient {y}."),
    "has_precise_active_ingredient": RelationTemplates(["The precise active ingredient of {x} is", "{x} contains precisely"],
                                                       "{x} has the precise active ingredient {y}."),
    "due_to": RelationTemplates(["{x} is due to", "The underlying cause of {x} is"], "{x} is a consequence of {y}."),
    "pathological_process": RelationTemplates(["The pathological process of {x} is", "{x} arises through"],
                                              "{x} has the pathological process {y}."),
    "interprets": RelationTemplates(["{x} interprets", "{x} is an assessment of"], "{x} is an interpretation of {y}."),
    "has_interpretation": RelationTemplates(["The interpretation of {x} is", "{x} is assessed as"], "{x} has the interpretation {y}."),
    "occurrence": RelationTemplates(["The occurrence of {x} is", "{x} typically begins in"], "{x} has the occurrence {y}."),
    "clinical_course": RelationTemplates(["The clinical course of {x} is", "The course of {x} is"], "{x} has the clinical course {y}."),
    "part_of": RelationTemplates(["{x} is part of", "{x} is a component of"], "{x} is a part of {y}."),
    "plays_role": RelationTemplates(["{x} plays the role of", "{x} acts as"], "{x} has the role {y}."),
    "has_manufactured_dose_form": RelationTemplates(["The dose form of {x} is", "{x} is supplied as"], "{x} has the dose form {y}."),
    "component": RelationTemplates(["The component of {x} is", "{x} measures"], "{x} has the component {y}."),
    "direct_substance": RelationTemplates(["The direct substance of {x} is", "{x} acts directly on"], "{x} has the direct substance {y}."),
    "direct_morphology": RelationTemplates(["The direct morphology of {x} is", "{x} acts directly on the lesion"],
                                           "{x} has the direct morphology {y}."),
    "has_disposition": RelationTemplates(["The disposition of {x} is", "{x} has the disposition of"], "{x} is disposed to act as {y}."),
}

T1_TEMPLATES = {
    "parent": RelationTemplates(["{x} is a kind of", "In the MeSH tree, {x} is filed under"], "{x} is a type of {y}."),
    "pharmacological_action": RelationTemplates(["The pharmacological action of {x} is", "{x} is classed among the"],
                                                "{x} is used as {y}."),
    "see_also": RelationTemplates(["{x} is related to", "See also, for {x}:"], "{x} is closely related to {y}."),
}
# T7 (MeSH SCRs and descriptors): the T1 wordings plus the SCR relations (the heading an SCR is indexed under, and
# its record class).
T7_TEMPLATES = {
    **T1_TEMPLATES,
    "mapped_to": RelationTemplates(["{x} is a kind of", "In MeSH, {x} is indexed under"], "{x} is a type of {y}."),
    "record_class": RelationTemplates(["{x} is classified as a", "In MeSH, {x} is recorded as a"], "{x} is a {y}."),
}


@dataclass(frozen=True)
class TrackSpec:
    name: str
    label: str
    data_root: Path
    eval_split: str = "eval"
    general_split: str | None = "eval-general"
    windows: int = 1024
    config: Path | None = None               # the track's corpus build config (alias-table replay)
    items_dir: Path | None = None            # WP-C7 items (zeroshot_property / zeroshot_entailment)
    holdout_names: Path | None = None
    category_relations: tuple[str, ...] = ("is_a",)
    kept_relations: tuple[str, ...] = ("is_a",)
    edit_relations: tuple[str, ...] = ("is_a",)
    family: str = "smollm2"                  # host tokenizer family of `data_root` (`cpt_plan.HOSTS[...]["corpus"]`)
    family_roots: dict[str, Path] = field(default_factory=dict, compare=False)   # other families' corpus roots
    # Licensed tracks (T1c): their alias table and dimension-3 items live outside the repository.
    alias_table_dir: Path | None = None      # default ALIAS_TABLE_DIR
    items_root: Path | None = None           # dimension-3 items (default `e9_plan.ITEMS`, in the repository)
    licensed: bool = False

    def for_family(self, family: str) -> "TrackSpec":
        """The track on another host tokenizer family: the same ontology, alias table, holdout, WP-C7 items and
        relation choices, with the corpora (`data_root`) relinked for that tokenizer (e.g. `qwen3`)."""
        if family == self.family:
            return self
        if family not in self.family_roots:
            raise ValueError(f"{self.name} has no {family} corpora (known: {', '.join([self.family, *self.family_roots])})")
        return replace(self, family=family, data_root=self.family_roots[family])

    @property
    def ontology(self) -> Path:
        return self.data_root / "ontology.pt"

    @property
    def eval_corpus(self) -> Path:
        return self.data_root / self.eval_split

    @property
    def general_corpus(self) -> Path | None:
        return self.data_root / self.general_split if self.general_split else None

    @property
    def alias_table_path(self) -> Path | None:
        return None if self.name == "wordnet" else (self.alias_table_dir or ALIAS_TABLE_DIR) / f"{self.name}.json"

    @property
    def zeroshot_items(self) -> Path | None:
        return self.items_dir if self.items_dir and (self.items_dir / "zeroshot_property.jsonl").exists() else None


DATA = Path("~/data/vsa-llm").expanduser()
# Qwen3 corpora of each track (WP-Qwen): the same builders with the Qwen3 base tokenizer (`Qwen/Qwen3-0.6B-Base`,
# shared by the 0.6B/1.7B/4B hosts). T5 is built (`experiments/t5-enterprise-glossary/t5-qwen3.yaml`); the others
# are built by the commands of `resources/plan-improvement/execution.md` (E9 on Qwen3).
# T1c (licensed; decision 58): the SmolLM2 corpora are the reference build; the Qwen3 relink is `hosts/qwen3`
# (`experiments/t1c-clinical/t1c-qwen3.yaml`).
T1C_ROOT = DATA / "t1c/snomed-mimic3-smollm2-v2"     # v2: holdout re-frozen at fraction 0.30 (T1c-1); v1 is stale
# T1c-ROOD (decision 63, H2 design (A); `experiments/t1c-clinical/rood/`): T1c with every ROOD patient's notes and every
# training document mentioning a ROOD concept dropped; held-out = the ROOD concepts' closure; evaluation on `eval-rood`.
T1C_ROOD_ROOT = DATA / "t1c/rood-v1/lm"
# Tracks built from SNOMED CT on MIMIC-III notes by `t1c_corpus` (same ontology adapter and lexicon).
T1C_TRACKS = ("t1c", "t1c-rood")
QWEN3_ROOTS = {"t5": DATA / "tracks/t5-glossary/v1-qwen3", "t4": DATA / "tracks/t4-chemistry/v1-qwen3",
               "t1": DATA / "t1/mesh-pubmed-gpt2-v1/hosts/qwen3", "wordnet": DATA / "c3/wordnet-qwen3-v1",
               "t1c": T1C_ROOT / "hosts/qwen3", "t7": DATA / "tracks/t7-newvocab/v1/hosts/qwen3"}
# Qwen3.5 corpora (WP-Qwen35): the same builders with the Qwen3.5 base tokenizer (`Qwen/Qwen3.5-0.8B-Base`, shared by the
# 0.8B/2B hosts), run with the Qwen3.5 environment. T5 is built (`experiments/t5-enterprise-glossary/t5-qwen35.yaml`).
QWEN35_ROOTS = {"t5": DATA / "tracks/t5-glossary/v1-qwen35", "t4": DATA / "tracks/t4-chemistry/v1-qwen35",
                "t1": DATA / "t1/mesh-pubmed-gpt2-v1/hosts/qwen3_5", "wordnet": DATA / "c3/wordnet-qwen3_5-v1",
                "t7": DATA / "tracks/t7-newvocab/v1/hosts/qwen3_5"}
TRACKS: dict[str, TrackSpec] = {
    "t5": TrackSpec("t5", "T5 enterprise glossary", DATA / "tracks/t5-glossary/v1",
                    config=Path("experiments/t5-enterprise-glossary/t5.yaml"), items_dir=Path("experiments/t5-enterprise-glossary/items"),
                    holdout_names=Path("experiments/t5-enterprise-glossary/items/holdout_concepts.txt"),
                    category_relations=("is_a",), kept_relations=("is_a",), edit_relations=("owned_by", "area", "status"),
                    family_roots={"qwen3": QWEN3_ROOTS["t5"], "qwen3_5": QWEN35_ROOTS["t5"]}),
    "t4": TrackSpec("t4", "T4 chemistry", DATA / "tracks/t4-chemistry/v1", config=Path("experiments/t4-chemistry/t4.yaml"),
                    items_dir=Path("experiments/t4-chemistry/items"),
                    holdout_names=Path("experiments/t4-chemistry/items/holdout_concepts.txt"),
                    category_relations=("is_a",), kept_relations=("is_a", "branch", "charge", "contains_element"),
                    edit_relations=("has_functional_parent", "has_role", "is_a"), family_roots={"qwen3": QWEN3_ROOTS["t4"], "qwen3_5": QWEN35_ROOTS["t4"]}),
    "t1": TrackSpec("t1", "T1-open (MeSH + PubMed)", DATA / "t1/mesh-pubmed-gpt2-v1/hosts/smollm2", eval_split="eval-pubmed",
                    windows=2048, config=Path("experiments/t1-open-clinical/t1.yaml"),
                    holdout_names=Path("experiments/t1-open-clinical/runs/v1/holdout_concepts.txt"),
                    category_relations=("parent",), kept_relations=("parent", "branch_top", "branch_second"),
                    edit_relations=("parent", "pharmacological_action"), family_roots={"qwen3": QWEN3_ROOTS["t1"], "qwen3_5": QWEN35_ROOTS["t1"]}),
    "t1c": TrackSpec("t1c", "T1c clinical (SNOMED CT + MIMIC-III)", T1C_ROOT, eval_split="eval-mimic", windows=2048,
                     config=Path("experiments/t1c-clinical/t1c.yaml"), holdout_names=T1C_ROOT / "holdout_concepts.txt",
                     category_relations=("is_a",), kept_relations=("is_a", "hierarchy", "semantic_tag"),
                     edit_relations=("finding_site", "causative_agent", "associated_morphology"),
                     family_roots={"qwen3": QWEN3_ROOTS["t1c"]}, alias_table_dir=DATA / "t1c/e9",
                     items_root=DATA / "t1c/items", items_dir=DATA / "t1c/items/zeroshot-t1c-v1", licensed=True),
    # No dimension-3 items: T1c's are built on its own holdout (trained here); the ROOD endpoints are the loss strata
    # on `eval-rood` and the coding task (`t1c_rood`). SmolLM2 only.
    "t1c-rood": TrackSpec("t1c-rood", "T1c-ROOD clinical (SNOMED CT + MIMIC-III, ROOD excluded)", T1C_ROOD_ROOT,
                          eval_split="eval-rood", windows=2048, config=Path("experiments/t1c-clinical/rood/t1c-rood.yaml"),
                          holdout_names=T1C_ROOD_ROOT / "holdout_concepts.txt", category_relations=("is_a",),
                          kept_relations=("is_a", "hierarchy", "semantic_tag"),
                          edit_relations=("finding_site", "causative_agent", "associated_morphology"),
                          alias_table_dir=DATA / "t1c/e9", items_root=DATA / "t1c/rood-v1/items", licensed=True),
    "wordnet": TrackSpec("wordnet", "WordNet general (C3)", DATA / "c3/wordnet-smollm2-v1", general_split=None,
                         category_relations=edit.CATEGORY_RELATIONS, kept_relations=tuple(sorted(edit.KEPT_RELATIONS)),
                         edit_relations=edit.CATEGORY_RELATIONS, family_roots={"qwen3": QWEN3_ROOTS["wordnet"], "qwen3_5": QWEN35_ROOTS["wordnet"]}),
    # T7 (decision 55): the SmolLM2 corpus is the build's reference; the Qwen relinks would be `hosts` of the same config
    # (not built). Category = the class a record is indexed under (SCR `mapped_to`) or filed under (descriptor `parent`);
    # edits change the pharmacological action where an entry has exactly one, else that class.
    "t7": TrackSpec("t7", "T7 new biomedical vocabulary (MeSH SCR + PubMed 2025-26)", DATA / "tracks/t7-newvocab/v1",
                    # 4,096 windows: the fewest that meet the held-out and rare criteria at ℓ_min 2 (runs/v1/feasibility.json)
                    eval_split="eval-pubmed", windows=4096, config=Path("experiments/t7-new-vocabulary/t7.yaml"),
                    holdout_names=Path("experiments/t7-new-vocabulary/runs/v1/holdout_concepts.txt"),
                    category_relations=("mapped_to", "parent"),
                    kept_relations=("mapped_to", "parent", "branch_top", "branch_second", "record_class"),
                    edit_relations=("pharmacological_action", "mapped_to", "parent"),
                    family_roots={"qwen3": QWEN3_ROOTS["t7"], "qwen3_5": QWEN35_ROOTS["t7"]}),
}
FAMILY_TOKENIZERS = {"smollm2": "HuggingFaceTB/SmolLM2-135M", "qwen3": "Qwen/Qwen3-0.6B-Base", "qwen3_5": "Qwen/Qwen3.5-0.8B-Base"}


def track_spec(name: str, family: str = "smollm2") -> TrackSpec:
    """A track's spec; `family` selects the host tokenizer family's corpora (default: the SmolLM2 ones)."""
    if name not in TRACKS:
        raise ValueError(f"unknown track {name!r}; choose from {', '.join(TRACKS)}")
    return TRACKS[name].for_family(family)


# -- the track ontology replay and its alias table ------------------------------------------------------------

def _load_config(spec: TrackSpec) -> dict[str, Any]:
    from .t1_open_corpus import load_config            # resolves `extends:`
    return load_config(Path(spec.config))


def track_frame_ontology(spec: TrackSpec) -> tuple[Any, list[int], dict[str, Any]]:
    """(FrameOntology with the synthetic concepts appended, their concept indices, track config) — the
    state `track_corpus.run` / `t1_open_corpus` had when it built the alias table."""
    config = _load_config(spec)
    if spec.name in ("t1", "t7"):              # built by t1_open_corpus (adapters `mesh`, `mesh_novel`)
        from .t1_open_corpus import build_track_ontology
        return build_track_ontology(config["ontology"]), [], config
    if spec.name in T1C_TRACKS:
        from .t1c_corpus import build_track_ontology as build_t1c_ontology
        return build_t1c_ontology(config["ontology"]), [], config
    from ..tracks import load_track
    from ..tracks.common import SyntheticConcept
    from .track_corpus import add_synthetic
    track = load_track(config["track"], config, Path(config["paths"]["data_root"]).expanduser())
    ontology = track.ontology()
    rows = [json.loads(line) for line in (spec.items_dir / "synthetic_concepts.jsonl").read_text().splitlines() if line.strip()]
    synthetic = [SyntheticConcept(r["name"], list(r["aliases"]), [tuple(e) for e in r["frame"]], r.get("facts", {}), r.get("meta", {}))
                 for r in rows]
    return ontology, add_synthetic(ontology, synthetic), config


def track_alias_table(spec: TrackSpec, *, ontology: dict[str, Any] | None = None) -> AliasTable:
    """The track's full evaluation alias table, checked against the ontology's digest and held-out entries."""
    frame_ontology, synthetic, config = track_frame_ontology(spec)
    index = {name: i for i, name in enumerate(frame_ontology.concept_names)}
    real = [index[n] for n in Path(spec.holdout_names).read_text().splitlines() if n]
    normalization = (config.get("linker") or {}).get("alias_normalization", "default")
    table = AliasTable.from_pairs(frame_ontology.alias_pairs, holdout=real + synthetic, include_holdout=True,
                                  normalization=normalization)
    ontology = ontology if ontology is not None else torch.load(spec.ontology, weights_only=False)
    if table.digest() != ontology["alias_table_sha256"]:
        raise ValueError(f"{spec.name}: the replayed alias table does not reproduce the ontology's alias_table_sha256")
    if sorted(table.heldout_entries()) != sorted(int(e) for e in ontology["heldout_entries"]):
        raise ValueError(f"{spec.name}: the replayed alias table has other held-out entries than the ontology")
    return table


def ensure_alias_table(spec: TrackSpec, path: Path | None = None) -> Path | None:
    """Write the track's evaluation alias table once (None for WordNet, rebuilt by `channel_probes`)."""
    path = Path(path) if path else spec.alias_table_path
    if path is None:
        return None
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cp.save_alias_table(track_alias_table(spec), path)
    return path


# -- the lexicon of a track (dimension-3 item wording) -------------------------------------------------------------

def _article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


@dataclass
class TrackLexicon:
    """`e9_ontology_edit` lexicon for a track ontology: the track's `RelationTemplates` (two property
    paraphrases; the third paraphrase, or the statement's prefix, as the held-out wording), readable
    filler names, and edits within the same filler type."""

    name: str
    templates: dict[str, RelationTemplates]
    texts: dict[str, str]
    types: dict[str, str] = field(default_factory=dict)
    category_relations: tuple[str, ...] = ("is_a",)
    kept_relations: frozenset[str] = frozenset({"is_a"})
    edit_relations: tuple[str, ...] = ("is_a",)
    article_relations: frozenset[str] = frozenset()
    null_surface: str = NULL_SURFACE
    donor_pos = frozenset({"*"})
    combination = "frame"           # no entry holds the category edge with all resampled edges together

    @property
    def edit_rule(self) -> str:
        return (f"edited relation: the first of {list(self.edit_relations)} with exactly one templated edge; new filler: same "
                "relation, same filler type, not already in the frame, another name (frequency-weighted); control: another "
                "filler drawn by the same rule")

    def entry_pos(self, ontology: dict[str, Any], table: AliasTable) -> list[str]:
        return ["*"] * len(table.entry_concepts)

    def display_surface(self, alias: str, concept: str) -> str:
        """The concept name's own casing when it is this alias (the linker matches case-insensitively)."""
        return concept if normalize_alias(concept) == alias else alias

    def text(self, atom: str) -> str | None:
        return self.texts.get(atom)

    def prompts(self, pos: str, relation: str) -> list[str] | None:
        spec = self.templates.get(relation)
        return list(spec.prompts[:2]) if spec else None

    def _statement(self, relation: str) -> tuple[str, str] | None:
        """(template, answer kind): the third paraphrase if there is one, else the statement's prefix."""
        spec = self.templates.get(relation)
        if spec is None:
            return None
        if len(spec.prompts) > 2:
            return spec.prompts[2], "answer"
        if "{y}" in spec.statement and not spec.statement.startswith("{y}"):
            prefix = spec.statement[:spec.statement.index("{y}")].rstrip()
            if prefix not in spec.prompts[:2]:
                return prefix, "statement"
        return None

    def statements(self, pos: str, relation: str) -> list[str] | None:
        found = self._statement(relation)
        return [found[0]] if found else None

    def _filler(self, relation: str, text: str) -> str:
        return _article(text) if relation in self.article_relations else text

    def answer(self, relation: str, text: str) -> str:
        return self.templates[relation].answer.format(y=self._filler(relation, text))

    def statement_answer(self, relation: str, text: str) -> str:
        template, kind = self._statement(relation)
        if kind == "answer":
            return self.answer(relation, text)
        statement = self.templates[relation].statement
        return " " + self._filler(relation, text) + statement[statement.index("{y}") + 3:]

    def hierarchy(self, atom: str) -> None:
        return None

    def hierarchy_text(self, name: str) -> str:
        return name

    def edit_edge(self, view: Any, entry: int) -> tuple[int, int] | None:
        frame = view.frame(entry)
        for relation in self.edit_relations:
            rid = view.relation_id.get(relation)
            found = [(r, f) for r, f in frame if r == rid]
            if len(found) == 1 and relation in self.templates and view.text(found[0][1]):
                return found[0]
        return None

    def atom_type(self, atom: str) -> str:
        return self.types.get(atom, atom.partition(":")[0])

    def plausible_edit(self, concept: str, old: str, new: str) -> bool:
        return self.atom_type(new) == self.atom_type(old)


def track_lexicon(spec: TrackSpec, ontology: dict[str, Any] | None = None) -> TrackLexicon:
    """The lexicon of a track: its relation templates and readable filler names for the ontology's atomics."""
    ontology = ontology if ontology is not None else torch.load(spec.ontology, weights_only=False)
    atoms = list(ontology["atomic_names"])
    texts: dict[str, str] = {}
    types: dict[str, str] = {}
    article: frozenset[str] = frozenset()
    if spec.name == "t5":
        from ..benchmarks.glossary import ARTICLE_TYPES
        from ..tracks.glossary import TEMPLATES
        from ..tracks import load_track
        config = _load_config(spec)
        track = load_track(config["track"], config, Path(config["paths"]["data_root"]).expanduser())
        term_type = {t["name"]: t["type"] for t in track.glossary()["terms"]}
        for atom in atoms:
            kind, _, value = atom.partition(":")
            if kind == "term":
                texts[atom] = f"the {value}" if term_type.get(value) in ARTICLE_TYPES else value
                types[atom] = f"term:{term_type.get(value, '?')}"
            else:
                texts[atom] = value
        templates = TEMPLATES
    elif spec.name == "t4":
        from ..tracks.chemistry import TEMPLATES
        frame_ontology, _, _ = track_frame_ontology(spec)
        names = frame_ontology.metadata["all_names"]
        for atom in atoms:
            kind, _, value = atom.partition(":")
            if kind == "chebi" and value in names:
                texts[atom] = names[value]
            elif kind in {"element", "charge", "branch"}:
                texts[atom] = value
        templates, article = TEMPLATES, frozenset({"has_role"})
    elif spec.name == "t1":
        frame_ontology, _, _ = track_frame_ontology(spec)
        heading = dict(zip(frame_ontology.concept_names, frame_ontology.metadata["headings"]))
        for atom in atoms:
            kind, _, value = atom.partition(":")
            if kind == "mesh" and value in heading:
                texts[atom] = heading[value]
        templates = T1_TEMPLATES
    elif spec.name == "t7":
        frame_ontology, _, _ = track_frame_ontology(spec)
        fillers = frame_ontology.metadata["filler_headings"]
        for atom in atoms:
            kind, _, value = atom.partition(":")
            if kind == "mesh" and value in fillers:
                texts[atom] = fillers[value]
            elif kind == "class":
                texts[atom] = value
        templates = T7_TEMPLATES
    elif spec.name in T1C_TRACKS:
        # Filler names are the concepts' preferred terms (licensed: they only reach item files under `items_root`);
        # the filler type is its top-level hierarchy, so an edit keeps e.g. a body structure a body structure.
        frame_ontology, _, _ = track_frame_ontology(spec)
        meta = frame_ontology.metadata
        heading = dict(zip(frame_ontology.concept_names, meta["headings"]))
        hierarchy = {c: (h[0] if h else "other") for c, h in zip(frame_ontology.concept_names, meta["concept_hierarchies"])}
        for atom in atoms:
            kind, _, value = atom.partition(":")
            if kind == "sct":
                if value in heading:
                    texts[atom] = heading[value]
                types[atom] = f"sct:{hierarchy.get(value, 'other')}"
            elif kind in {"top", "tag"}:
                texts[atom] = value.replace("_", " ")
        templates = T1C_TEMPLATES
    else:
        raise ValueError("WordNet uses e9_ontology_edit.WordNetLexicon")
    return TrackLexicon(spec.name, templates, texts, types, category_relations=spec.category_relations,
                        kept_relations=frozenset(spec.kept_relations), edit_relations=spec.edit_relations,
                        article_relations=article)


def lexicon_for(spec: TrackSpec, ontology: dict[str, Any] | None = None) -> Any:
    return edit.WordNetLexicon() if spec.name == "wordnet" else track_lexicon(spec, ontology)


# -- WP-C7 zero-shot items ---------------------------------------------------------------------------------------

def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def _template(text: str, surface: str) -> str | None:
    return text.replace(surface, "{x}", 1) if surface in text else None


def convert_wpc7_items(items_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """WP-C7 `zeroshot_property` / `zeroshot_entailment` rows → E5.4-format (concepts, items)."""
    concepts: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read(Path(items_dir) / "zeroshot_property.jsonl"):
        groups[row["group"]].append(row)
    for group, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["paraphrase"])
        first = rows[0]
        templates = [t for t in (_template(r["prompt"], r["surface"]) for r in rows) if t]
        if not templates or any(r["choices"] != first["choices"] or r["label"] != first["label"] for r in rows):
            continue
        cid = f"wpc7-{first['surface']}"
        concepts.setdefault(cid, {"concept": cid, "surface": first["surface"], "split": first["split"], "entry": None})
        items.append({"id": group, "concept": cid, "test": "property", "relation": first["relation"], "split": first["split"],
                      "templates": templates, "null": NULL_SURFACE, "candidates": list(first["choices"]), "gold": int(first["label"])})
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read(Path(items_dir) / "zeroshot_entailment.jsonl"):
        pairs[row["pair"]].append(row)
    for pair, rows in pairs.items():
        if sorted(r["label"] for r in rows) != [0, 1]:
            continue
        a, b = rows[0]["statement"], rows[1]["statement"]
        common = 0
        while common < min(len(a), len(b)) and a[common] == b[common]:
            common += 1
        cut = a.rfind(" ", 0, common + 1)              # the continuations start at a word, with their space
        prefix = a[:cut] if cut > 0 else a[:common]
        template = _template(prefix, rows[0]["surface"])
        if template is None or not template.strip():
            continue
        cid = f"wpc7-{rows[0]['surface']}"
        concepts.setdefault(cid, {"concept": cid, "surface": rows[0]["surface"], "split": rows[0]["split"], "entry": None})
        items.append({"id": pair, "concept": cid, "test": "entailment", "relation": rows[0]["relation"], "split": rows[0]["split"],
                      "templates": [template], "null": NULL_SURFACE, "candidates": [r["statement"][len(prefix):] for r in rows],
                      "gold": [r["label"] for r in rows].index(1)})
    return list(concepts.values()), items


@contextlib.contextmanager
def replaced_frames(channel: Any, frames: dict[int, list[tuple[int, int]]]) -> Iterator[bool]:
    """Within the block, the listed entries read the given frames (same relations and degree, other
    fillers); yields False when the channel does not compose."""
    if channel is None or channel.mode != "compose" or not frames:
        yield False
        return
    composer = channel.composer
    original = composer.schedule
    fillers = original.fillers.clone()
    for entry, frame in frames.items():
        lo, hi = int(original.offsets[entry]), int(original.offsets[entry + 1])
        if [r for r, _ in frame] != original.relations[lo:hi].tolist():
            raise ValueError(f"entry {entry}: a replacement frame must keep the relations")
        fillers[lo:hi] = torch.tensor([f for _, f in frame], dtype=fillers.dtype)
    composer.set_schedule(FrameSchedule(original.offsets, original.relations, fillers))
    try:
        yield True
    finally:
        composer.set_schedule(original)


def random_frames(ontology: dict[str, Any], entries: Sequence[int], *, seed: int) -> dict[int, list[tuple[int, int]]]:
    """Per entry: its relations with fillers drawn frequency-weighted from each relation's fillers."""
    offsets = np.asarray(ontology["offsets"]); relations = np.asarray(ontology["relations"]); fillers = np.asarray(ontology["fillers"])
    pools: dict[int, Counter] = defaultdict(Counter)
    for r, f in zip(relations.tolist(), fillers.tolist()):
        pools[r][f] += 1
    rng = random.Random(seed)
    out = {}
    for e in sorted(set(int(x) for x in entries)):
        out[e] = [(int(r), edit._weighted_choice(rng, pools[int(r)], set())) for r in relations[offsets[e]:offsets[e + 1]].tolist()]
    return out


def evaluate_track_zeroshot(run: E5Run, items_dir: Path, *, sources: Sequence[str] | None = None, fit_entries: int = 4000,
                            seed: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    concepts, items = convert_wpc7_items(items_dir)
    adapter = run.adapter
    found = adapter.link_targets([c["surface"] for c in concepts], [(0, len(c["surface"])) for c in concepts]) \
        if adapter.linker is not None else [[] for _ in concepts]
    resolved = {}
    for c, entries in zip(concepts, found):
        entry = entries[0] if len(entries) == 1 else None
        status = cp.entry_status([entry], adapter.heldout_entries, adapter.train_frequency) if entry is not None else "unlinked"
        resolved[c["concept"]] = {"entry": entry, "linked": entry is not None, "status": status, "split": c["split"]}
        c["entry"] = entry
    entries = {cid: r["entry"] for cid, r in resolved.items() if r["linked"]}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    mode = run.mode
    allowed = (["own"] if mode in {"none", "hashed"} else ["own", "none"] if mode == "random"
               else ["own", "none", "mean_row"] + (["random_frame"] if mode == "compose" else []))
    sources = [s for s in (sources or TRACK_SOURCES) if s in allowed]
    rows: dict[str, dict[int, torch.Tensor]] = {}
    info: dict[str, Any] = {"items": {t: sum(i["test"] == t for i in items) for t in ("property", "entailment")}}
    if run.channel is not None and entries and {"none", "mean_row"} & set(sources):
        rows, fit_info = zs.baseline_rows(run, adapter, concepts, entries, [s for s in ("none", "mean_row") if s in sources],
                                          manifest={}, items_dir=Path(items_dir), fit_entries=fit_entries, contexts=0, seed=seed,
                                          log=log)
        info.update(fit_info)
    sources = [s for s in sources if s not in {"none", "mean_row"} or s in rows]
    shuffled = random_frames(run.ontology, entries.values(), seed=seed) if "random_frame" in sources else {}
    started = time.monotonic()
    null_scores = None
    results: dict[str, Any] = {}
    for source in sources:
        log(f"  track zero-shot: source {source}")
        with replaced_frames(run.channel, shuffled if source == "random_frame" else {}), \
                (override_rows(run.channel, rows.get(source)) if run.channel is not None else contextlib.nullcontext()):
            prompts, null_scores = zs.score_prompts(adapter, items, surfaces, null_scores)
        split = {i["id"]: i["split"] for i in items}
        for row in prompts:
            row["split"] = split[row["id"]]
        results[source] = {"prompts": prompts, "corpus": []}
    return {"resolved": resolved, "sources": sources, "results": results, "info": info, "seconds": time.monotonic() - started}


def _vectors(result: dict[str, Any], keep: set[str]) -> dict[str, tuple[list[str], np.ndarray]]:
    rows = [r for r in result["prompts"] if r["concept"] in keep]
    pick = lambda chosen, key: ([r["id"] for r in chosen], np.asarray([r[key] for r in chosen], dtype=float))
    prop = [r for r in rows if r["test"] == "property"]
    return {"property": pick(prop, "correct"), "paraphrase": pick([r for r in prop if r["consistent"] is not None], "consistent"),
            "entailment": pick([r for r in rows if r["test"] == "entailment"], "correct")}


def summarize_track_zeroshot(evaluation: dict[str, Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per source and subset (`linked` = every linked concept; `synthetic`, `heldout`) the mean of each
    test, and `own` − baseline on the linked concepts (paired bootstrap; Holm within each test)."""
    resolved = evaluation["resolved"]
    linked = {c for c, r in resolved.items() if r["linked"]}
    subsets = {"linked": linked, **{s: {c for c in linked if resolved[c]["split"] == s} for s in SPLITS}}
    out: dict[str, Any] = {"concepts": len(resolved), "linked_concepts": len(linked),
                           "status_counts": dict(sorted(Counter(r["status"] for r in resolved.values()).items())),
                           "split_counts": {s: len(v) for s, v in subsets.items()}, "sources": {}, "comparisons": []}
    for source, result in evaluation["results"].items():
        out["sources"][source] = {name: {t: {"mean": float(v.mean()) if v.size else None, "n": int(v.size)}
                                         for t, (_, v) in _vectors(result, keep).items()} for name, keep in subsets.items()}
    if "own" in evaluation["results"]:
        for subset in ("linked", *SPLITS):
            own = _vectors(evaluation["results"]["own"], subsets[subset])
            for test in TESTS:
                block = []
                ids_a, a = own[test]
                for source in evaluation["sources"]:
                    if source == "own" or not a.size:
                        continue
                    ids_b, b = _vectors(evaluation["results"][source], subsets[subset])[test]
                    if ids_a != ids_b:
                        continue
                    ci = zs.paired_difference(a, b, resamples=resamples, seed=seed)
                    block.append({"subset": subset, "test": test, "baseline": source, "difference": ci["mean"], "ci_low": ci["ci_low"],
                                  "ci_high": ci["ci_high"], "n": ci["n"], "p_value": ci["p_value"], "better": ci["mean"] > 0})
                for row, adjusted in zip(block, holm_adjust([r["p_value"] for r in block]) if block else []):
                    row.update(p_holm=adjusted, significant=adjusted < 0.05)
                out["comparisons"] += block
    return out


def render_zeroshot(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    quantized = f", {source['quantization']['variant']}" if source.get("quantization") else ""
    lines = [f"# E9 track zero-shot ({header['track']}) — {source['condition']} seed {source['seed']} ({source['size']}{quantized})", "",
             f"WP-C7 items `{header['items']}`: {summary['linked_concepts']} of {summary['concepts']} terms link "
             f"({summary['status_counts']}; {summary['split_counts']}). Property: PMI argmax among the choices (null surface "
             f"\"{NULL_SURFACE}\"); paraphrase: argmax agreement over the 3 paraphrases; entailment: the true vs the corrupted "
             "frame statement.", ""]
    for subset in ("linked", *SPLITS):
        lines += [f"## {subset}", "", "| source | " + " | ".join(TESTS) + " |", "|---|" + "---:|" * len(TESTS)]
        for name, by_subset in summary["sources"].items():
            lines.append(f"| {name} | " + " | ".join(fmt(by_subset[subset][t]["mean"], 4) for t in TESTS) + " |")
        rows = [c for c in summary["comparisons"] if c["subset"] == subset]
        if rows:
            lines += ["", "| test | baseline | own − baseline [95% CI] | n | p (Holm) |", "|---|---|---|---:|---:|"]
            lines += [f"| {c['test']} | {c['baseline']} | {c['difference']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                      f"{c['n']} | {fmt(c.get('p_holm'), 4)} |" for c in rows]
        lines.append("")
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------------------------------------------

def run_zeroshot(args: argparse.Namespace) -> dict[str, Any]:
    spec = track_spec(args.track)
    items_dir = args.items or spec.zeroshot_items
    if items_dir is None:
        raise ValueError(f"{spec.name} has no WP-C7 zero-shot items")
    requested = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    alias_table = args.alias_table or ensure_alias_table(spec)
    config = {"experiment": "e9-track-zeroshot", "track": spec.name, "run": str(args.run), "checkpoint": args.checkpoint,
              "items": str(items_dir), "alias_table": str(alias_table) if alias_table else None, "sources": requested,
              "fit_entries": args.fit_entries, "seed": args.seed, "resamples": args.resamples, "quantize": args.quantize,
              "quantize_channel": bool(args.quantize_channel), "group_size": args.group_size}
    if args.overwrite:
        clear_output(args.output)
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=alias_table,
                   quantize=args.quantize, quantize_channel=args.quantize_channel, group_size=args.group_size)
    evaluation = evaluate_track_zeroshot(run, items_dir, sources=requested, fit_entries=args.fit_entries, seed=args.seed)
    summary = summarize_track_zeroshot(evaluation, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "items": str(items_dir), "track": spec.name}
    with (args.output / "predictions.jsonl").open("w") as handle:
        for source, result in evaluation["results"].items():
            for row in result["prompts"]:
                handle.write(json.dumps(json_ready({"source": source, **{k: v for k, v in row.items() if k != "pmi"}})) + "\n")
    write_json(args.output / "summary.json", {**header, "resolved": evaluation["resolved"], "info": evaluation["info"],
                                              "summary": summary, "seconds": evaluation["seconds"]})
    (args.output / "report.md").write_text(render_zeroshot(summary, header))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary}


def run_items(args: argparse.Namespace) -> dict[str, Any]:
    family = getattr(args, "family", "smollm2")
    spec = track_spec(args.track, family)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")
    ontology = torch.load(spec.ontology, weights_only=False)
    alias_table = ensure_alias_table(spec)
    lexicon = lexicon_for(spec, ontology)
    tokenizer = args.tokenizer or FAMILY_TOKENIZERS[family]
    extend = getattr(args, "extend", None)
    if extend is not None and not args.count:
        raise ValueError("--extend needs --count (the total of the superset)")
    if args.kind == "new":
        texts = edit._contamination_texts([spec.data_root / "train"], tokenizer, args.contamination_tokens)
        reserved = edit._reserved_names(args.reserved_names)
        if spec.items_dir is not None:          # the track's own synthetic names are taken too
            reserved |= {a.lower() for r in _read(spec.items_dir / "synthetic_concepts.jsonl") for a in r.get("aliases", [])}
        if extend is not None:                  # items -v2: a strict superset of -v1 (decision 56)
            return edit.extend_new_word_items(extend, spec.ontology, args.output, tokenizer_name=tokenizer, count=args.count,
                                              extension_seed=args.extension_seed, alias_table=alias_table,
                                              contamination_texts=texts, reserved_names=reserved, lexicon=lexicon)
        return edit.build_new_word_items(spec.ontology, args.output, tokenizer_name=tokenizer, count=args.count or 300,
                                         seed=args.seed, min_subtokens=args.min_subtokens, alias_table=alias_table,
                                         contamination_texts=texts, reserved_names=reserved, name_seed=args.name_seed,
                                         lexicon=lexicon)
    if extend is not None:
        return edit.extend_edit_items(extend, spec.ontology, args.output, tokenizer_name=tokenizer, count=args.count,
                                      extension_seed=args.extension_seed, alias_table=alias_table, lexicon=lexicon)
    return edit.build_edit_items(spec.ontology, args.output, tokenizer_name=tokenizer, count=args.count or 200, seed=args.seed,
                                 min_subtokens=args.min_subtokens, alias_table=alias_table, lexicon=lexicon)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    table = sub.add_parser("alias-table", help="write a track's evaluation alias table (digest-checked)")
    table.add_argument("--track", required=True, choices=sorted(TRACKS)); table.add_argument("--output", type=Path, default=None)
    items = sub.add_parser("items", help="dimension-3 items (new words or edits) on a track ontology")
    items.add_argument("--track", required=True, choices=sorted(TRACKS)); items.add_argument("--kind", required=True, choices=["new", "edits"])
    items.add_argument("--output", type=Path, required=True)
    items.add_argument("--family", default="smollm2", choices=sorted(FAMILY_TOKENIZERS),
                       help="host tokenizer family: its corpora (ontology, contamination text) and default tokenizer")
    items.add_argument("--tokenizer", default=None, help="default: the family's (SmolLM2-135M; Qwen3-0.6B-Base; Qwen3.5-0.8B-Base)")
    items.add_argument("--count", type=int, default=None); items.add_argument("--seed", type=int, default=0)
    items.add_argument("--min-subtokens", type=int, default=2); items.add_argument("--name-seed", type=int, default=11)
    items.add_argument("--contamination-tokens", type=int, default=20_000_000)
    items.add_argument("--reserved-names", type=Path, nargs="*", default=[])
    items.add_argument("--extend", type=Path, default=None,
                       help="build a strict superset of this item directory (e.g. -v2 from -v1: its rows first and "
                            "byte-identical, its seeds reused; --count = the new total)")
    items.add_argument("--extension-seed", type=int, default=1, help="random stream of the extension's rows (with --extend)")
    ev = sub.add_parser("zeroshot", help="score the track's WP-C7 zero-shot items on a trained run")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--track", required=True, choices=sorted(TRACKS))
    ev.add_argument("--output", type=Path, required=True); ev.add_argument("--items", type=Path, default=None)
    ev.add_argument("--alias-table", type=Path, default=None); ev.add_argument("--checkpoint", default="final.pt")
    ev.add_argument("--sources", default="", help="comma-separated subset of " + ",".join(TRACK_SOURCES))
    ev.add_argument("--fit-entries", type=int, default=4000); ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--batch-size", type=int, default=64); ev.add_argument("--device", default=None)
    ev.add_argument("--quantize", choices=["int8", "int4"], default=None)
    ev.add_argument("--quantize-channel", action="store_true"); ev.add_argument("--group-size", default="auto")
    ev.add_argument("--overwrite", action="store_true", help="replace the result files of a previous evaluation in --output")
    args = parser.parse_args(argv)
    if args.command == "alias-table":
        spec = track_spec(args.track)
        path = ensure_alias_table(spec, args.output)
        print(json.dumps({"track": spec.name, "alias_table": str(path) if path else None}))
    elif args.command == "items":
        print(json.dumps(run_items(args), indent=2, default=str))
    else:
        if args.quantize_channel and not args.quantize:
            parser.error("--quantize-channel needs --quantize")
        result = run_zeroshot(args)
        print(json.dumps({s: v["linked"] for s, v in result["summary"]["sources"].items()}, indent=2))


if __name__ == "__main__":
    main()
