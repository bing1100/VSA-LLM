"""E11 — read-to-learn: one-shot vocabulary from reading a definition (author decision 59).

Pre-registration: `experiments/e11-read-to-learn/preregistration.md` (written and committed before any run). A trained
E9 model with the ontology-composed span channel reads a term's definition; a reader turns it into a frame over the
existing atomics and relations; the frame is written into the ontology (new words: `SpanChannel.add_entries`; held-out
real terms: their frame replaced, `authoring.replace_frames`); the channel composes the term's row; the model is then
tested on the term **without the definition in context and with no gradient step**. Evaluation only — nothing trains.

Item sets (`experiments/e11-read-to-learn/items/<name>/`, schema `e11-read/1`):
- `new` — the E9 invented new words (`e9_ontology_edit` items) with definitions written from their gold frames
  (T5: `glossary` / `dictionary` / `prose`; T4: `chebi` / `prose`); tests = the E9 dimension-3 tests (property,
  entailment, paraphrase, statement) on the E9 items.
- `heldout` — held-out real terms of a track (never linked in training): T5 glossary terms (definitions written from
  their frames), T4 ChEBI entities (their ChEBI definition, CC BY 4.0), T1 MeSH descriptors (their scope note);
  tests = the WP-C7 zero-shot items of those terms and the loss after the term in the track's evaluation text.
- `swap` — a textbook glossary (OpenStax Chemistry 2e) mapped onto existing ChEBI entries: the read frame replaces the
  trained one (exploratory; coverage and frame quality).

Methods of `evaluate` (`--methods`): `frames` (readers → frames, precision / recall, cost; always), `persistence`
(item tests with each reader's frame, no definition in context), `context` (the definition prepended: IKE-style
in-context reading), `gradient` (a one-shot, compute-matched gradient update of the host on the definition, weights
restored after every term), `windows` (loss after the read terms in the evaluation corpus, per reader; `heldout` and
`swap` sets), `locality` (rows of other entries unchanged; unlinked-token loss in the read terms' windows).

    python -m vsa_embed.experiments.e11_read_to_learn items --kind new --track t5 --new-items DIR --output DIR
    python -m vsa_embed.experiments.e11_read_to_learn items --kind heldout --track t5|t4|t1|t7|t7rood --output DIR [--limit N]
    python -m vsa_embed.experiments.e11_read_to_learn fetch-openstax --output ~/data/vsa-llm/e11/raw/openstax-chemistry-2e
    python -m vsa_embed.experiments.e11_read_to_learn items --kind swap --track t4 --glossary GLOSSARY.jsonl --output DIR
    python -m vsa_embed.experiments.e11_read_to_learn teacher --items DIR [--dry-run]          (optional; claude -p)
    python -m vsa_embed.experiments.e11_read_to_learn evaluate --run RUN --items DIR --output OUT [--methods ...]
        [--readers ...] [--styles ...] [--primary-style prose] [--gradient-lr 1e-4 | --gradient-lr-from JSON] [--limit N]
    python -m vsa_embed.experiments.e11_read_to_learn gradient-dev --run RUN --items DEV_DIR --output OUT
    python -m vsa_embed.experiments.e11_read_to_learn report --runs RUN_E11_DIR ... --output DIR
    python -m vsa_embed.experiments.e11_read_to_learn plan [--tracks t5 t4] [--seeds 1 2 3]     (prints queue commands)
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import dataclasses
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from .. import read_to_learn as rtl
from ..authoring import replace_frames
from ..compose import FrameSchedule
from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable, normalize_alias
from ..statistics import holm_adjust, paired_ratio_bootstrap
from . import e5_zeroshot as zs
from . import e9_ontology_edit as edit
from .e5_common import (E5Run, clear_output, entry_rows, finish_output, fmt, json_ready, open_run, override_rows,
                        start_output, write_json)

SCHEMA = "e11-read/1"
ROOT = Path("experiments/e11-read-to-learn")
ITEMS = ROOT / "items"
DATA = Path("~/data/vsa-llm/e11").expanduser()
FOLDER = "e11"                                     # per-run output folder name (RUN/e11-<set>)
METHODS = ("frames", "persistence", "context", "gradient", "windows", "locality")
PRIMARY_STYLE = {"t5": "prose", "t4": "chebi", "t1": "scope", "t7": "scr", "wordnet": "prose"}
EXCLUDED_KINDS = {"t4": ("element", "charge", "branch"), "t1": ("branch",), "t7": ("branch",), "wordnet": ("lexname", "pos"),
                  "t5": ()}
# T7-ROOD (decision 63, H1; `e9_tracks.TRACKS["t7rood"]`) reads as T7: the same records, notes and lexicon.
PRIMARY_STYLE["t7rood"], EXCLUDED_KINDS["t7rood"] = PRIMARY_STYLE["t7"], EXCLUDED_KINDS["t7"]
# MeSH SCR notes are partly bibliographic ("structure given in first source", "RN given refers to parent cpd"): those
# `;`-separated parts are dropped; a note is read only if ≥ 3 words remain (§15).
SCR_BOILERPLATE = re.compile(r"^(rn given|structure|mf given|in first source|for .* see|see also|no structure|mixture of|"
                             r"also see|isomer|.*\bfirst source\b|.*\bgiven in\b.*source)", re.I)


def scr_definition(note: str, *, min_words: int = 3) -> str | None:
    """The informative part of a MeSH supplementary-record note, or None."""
    kept = [p.strip() for p in (note or "").split(";")]
    kept = [p for p in kept if p and not SCR_BOILERPLATE.match(p) and len(p.split()) >= 2]
    text = "; ".join(kept)
    return text if len(text.split()) >= min_words else None
LICENCES = {"t5": "project-generated (synthetic T5 glossary)", "chebi": "CC BY 4.0 (ChEBI, EMBL-EBI, release 255)",
            "mesh": "public domain (MeSH 2026, courtesy of the U.S. National Library of Medicine)",
            "openstax": "CC BY-NC-SA 4.0 (OpenStax Chemistry 2e; research use, text kept under ~/data, not committed)"}
NEW_TESTS = ("property", "property_new", "entailment", "paraphrase", "statement_accuracy", "statement_loss")
HELDOUT_TESTS = ("property", "paraphrase", "entailment")
AFTER_WINDOW = 8                                   # targets e+1..e+8 after a span ending at e (the trainer's strata)


# -- track context ---------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class TrackContext:
    track: str
    ontology: dict[str, Any]
    table: AliasTable
    lexicon: Any
    fillers: rtl.FillerLexicon
    typing: rtl.RelationTyping

    @property
    def atom_id(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.ontology["atomic_names"])}

    @property
    def relation_id(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.ontology["relation_names"])}

    def text(self, atom: int | str) -> str | None:
        name = self.ontology["atomic_names"][atom] if isinstance(atom, int) else atom
        return self.lexicon.text(name)

    def frame(self, entry: int) -> rtl.Frame:
        o = self.ontology
        lo, hi = int(o["offsets"][entry]), int(o["offsets"][entry + 1])
        return list(zip(np.asarray(o["relations"][lo:hi]).tolist(), np.asarray(o["fillers"][lo:hi]).tolist()))

    def names(self, frame: rtl.Frame | None) -> list[list[str]] | None:
        if frame is None:
            return None
        return [[self.ontology["relation_names"][r], self.ontology["atomic_names"][a]] for r, a in frame]

    def ids(self, frame: Sequence[Sequence[str]] | None) -> rtl.Frame | None:
        return None if frame is None else edit.resolve_frame(frame, self.relation_id, self.atom_id)

    def facts(self, frame: rtl.Frame) -> dict[str, list[str]]:
        """Relation name → readable filler texts of a frame (atoms without a text are skipped)."""
        out: dict[str, list[str]] = defaultdict(list)
        for r, a in frame:
            text = self.text(a)
            if text:
                out[self.ontology["relation_names"][r]].append(text)
        return dict(out)

    def own_atom(self, concept_name: str | None) -> int | None:
        if not concept_name:
            return None
        by_value = {n.partition(":")[2]: i for i, n in enumerate(self.ontology["atomic_names"])}
        return by_value.get(concept_name)


def atom_type_function(lexicon: Any) -> Callable[[str], str]:
    return lexicon.atom_type if hasattr(lexicon, "atom_type") else (lambda atom: atom.partition(":")[0])


def make_context(track: str, ontology: dict[str, Any], table: AliasTable, lexicon: Any) -> TrackContext:
    fillers = rtl.build_filler_lexicon(ontology, table, lexicon.text, exclude_kinds=EXCLUDED_KINDS.get(track, ()))
    return TrackContext(track, ontology, table, lexicon, fillers, rtl.RelationTyping(ontology, atom_type_function(lexicon)))


def track_context(track: str, family: str = "smollm2") -> TrackContext:
    from . import e9_tracks as tracks
    spec = tracks.track_spec(track, family)
    ontology = torch.load(spec.ontology, weights_only=False)
    table_path = tracks.ensure_alias_table(spec)
    table = cp.load_alias_table(table_path) if table_path else cp.resolve_alias_table(ontology, spec.ontology)[0]
    return make_context(track, ontology, table, tracks.lexicon_for(spec, ontology))


def run_context(run: E5Run) -> TrackContext:
    from .e9_dim3_baselines import lexicon_for_run
    lexicon, error = lexicon_for_run(run)
    if lexicon is None:
        raise ValueError(f"no lexicon for this run: {error}")
    return make_context(run.config.get("e9_track") or "wordnet", run.ontology, run.table, lexicon)


# -- item sets -------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ReadSet:
    path: Path
    manifest: dict[str, Any]
    concepts: list[dict[str, Any]]
    definitions: list[dict[str, Any]]
    items: list[dict[str, Any]]

    @property
    def kind(self) -> str:
        return self.manifest["kind"]

    @property
    def styles(self) -> list[str]:
        return list(self.manifest["styles"])

    def limit(self, count: int | None) -> "ReadSet":
        if not count:
            return self
        keep = {c["concept"] for c in self.concepts[:count]}
        return dataclasses.replace(self, concepts=[c for c in self.concepts if c["concept"] in keep],
                                   definitions=[d for d in self.definitions if d["concept"] in keep],
                                   items=[i for i in self.items if i["concept"] in keep])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(json_ready(r)) + "\n" for r in rows))


def load_read_set(path: Path) -> ReadSet:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not an {SCHEMA} item directory")
    concepts = _read_jsonl(path / "concepts.jsonl")
    definitions = _read_jsonl(path / "definitions.jsonl")
    if manifest["kind"] == "new":
        base = Path(manifest["base_items"])
        _, base_concepts, items = edit.load_item_dir(base, edit.SCHEMA_NEW)
        if [c["concept"] for c in base_concepts] != [c["concept"] for c in concepts]:
            raise ValueError(f"{path}: concepts differ from the base items {base}")
    else:
        items = _read_jsonl(path / "items.jsonl")
    known = {c["concept"] for c in concepts}
    if any(d["concept"] not in known for d in definitions) or any(i["concept"] not in known for i in items):
        raise ValueError(f"{path}: a definition or item refers to an unknown concept")
    return ReadSet(path, manifest, concepts, definitions, items)


def _manifest(kind: str, track: str, styles: Sequence[str], **extra: Any) -> dict[str, Any]:
    return {"schema": SCHEMA, "kind": kind, "track": track, "styles": list(styles), "created": time.strftime("%Y-%m-%d"),
            **extra}


def build_new_set(new_items: Path, out_dir: Path, ctx: TrackContext, *, styles: Sequence[str] | None = None,
                  seed: int = 0) -> dict[str, Any]:
    """Definitions for the E9 invented new words, written from each word's gold frame in every style."""
    manifest, concepts, items = edit.load_item_dir(new_items, edit.SCHEMA_NEW)
    writer, default_styles = (rtl.t4_definition, rtl.T4_STYLES) if ctx.track == "t4" else (rtl.t5_definition, rtl.T5_STYLES)
    styles = list(styles or default_styles)
    templates: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["test"] in {"property", "statement"}:
            templates[item["concept"]] |= set(item["templates"])
    out_concepts, definitions = [], []
    overlaps = Counter()
    for c in concepts:
        frame = ctx.ids(c["frame"])
        facts = ctx.facts(frame)
        out_concepts.append({"concept": c["concept"], "surface": c["surface"], "entry": None, "frame": c["frame"],
                             "random_frame": c["random_frame"], "own_atom": None, "split": "new", "degree": len(frame)})
        for style in styles:
            rng = random.Random(f"{seed}|{c['concept']}|{style}")
            text = writer(c["surface"], facts, style, rng)
            found = rtl.template_overlaps(text, c["surface"], templates[c["concept"]])
            overlaps[style] += bool(found)
            licence = "project-generated (invented compound; ChEBI names CC BY 4.0)" if ctx.track == "t4" else LICENCES["t5"]
            definitions.append({"concept": c["concept"], "style": style, "text": text, "headword": c["surface"],
                                "source": "written from the gold frame", "licence": licence, "template_overlaps": found})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "concepts.jsonl", out_concepts)
    _write_jsonl(out_dir / "definitions.jsonl", definitions)
    result = _manifest("new", ctx.track, styles, base_items=str(new_items), base_manifest_counts=manifest.get("counts"),
                       concepts=len(out_concepts), definitions=len(definitions), seed=seed,
                       definitions_with_template_overlap=dict(overlaps),
                       writer="read_to_learn.t4_definition" if ctx.track == "t4" else "read_to_learn.t5_definition")
    write_json(out_dir / "manifest.json", result)
    return result


def _entry_concept_name(ctx: TrackContext, entry: int) -> str:
    return ctx.ontology["concept_names"][ctx.table.entry_concepts[entry][0]]


def linkable_headwords(ctx: TrackContext, names: dict[int, str], tokenizer: Any, min_subtokens: int
                       ) -> dict[int, tuple[str, str]]:
    """Per entry the headword a definition is read under: its display name if the track linker links it to **this**
    entry (in '<headword>: …'), else the entry's canonical linkable alias (`canonical_surfaces`); entries with neither
    are left out (the channel could not read them). Values: (headword, "name" | "alias")."""
    from ..span_channel import CausalLinker
    from .e5_common import canonical_surfaces
    linker = CausalLinker(ctx.table, min_subtokens=min_subtokens)
    canonical = canonical_surfaces(ctx.table, tokenizer, min_subtokens, list(names))

    def links(text: str, entry: int) -> bool:
        offsets = [tuple(o) for o in tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]]
        return any(span.entry == entry for span in linker.link(text, offsets))

    out: dict[int, tuple[str, str]] = {}
    for entry, name in names.items():
        alias = canonical.get(entry, {})
        for source, headword in (("name", name), ("alias", alias.get("surface") if alias.get("linkable") else None)):
            if headword and links(f"{headword}: x", entry):
                out[entry] = (headword, source)
                break
    return out


def heldout_definitions(track: str, ctx: TrackContext, entries: Sequence[int], styles: Sequence[str], *, seed: int = 0,
                        tokenizer: Any = None, min_subtokens: int = 2
                        ) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """(headword per entry, definitions per entry, source info) of held-out real terms. With a `tokenizer`, headwords are
    link-checked (`linkable_headwords`): a term whose headword the linker would not link to its entry is left out."""
    from . import e9_tracks as tracks
    names: dict[int, str] = {}
    texts: dict[int, Any] = {}
    info: dict[str, Any] = {}
    if track == "t5":
        for e in entries:
            names[e] = _entry_concept_name(ctx, e).removeprefix("synthetic:")
        info["source"] = "T5 generator glossary (gold frames)"
    else:
        spec = tracks.track_spec(track)
        frame_ontology, _, _ = tracks.track_frame_ontology(spec)
        if track == "t4":
            records, all_names = frame_ontology.metadata["records"], frame_ontology.metadata["all_names"]
            for e in entries:
                cid = _entry_concept_name(ctx, e)
                record = records.get(cid) or {}
                definition = rtl.strip_markup(record.get("definition") or "")
                if definition:
                    names[e] = all_names.get(cid, record.get("name") or cid)
                    texts[e] = (definition, f"ChEBI release 255 definition ({cid})", LICENCES["chebi"], "chebi")
            info["source"] = "ChEBI 255 OBO `def:` (HTML stripped), read as '<headword>: <definition>'"
        elif track == "t1":
            notes = dict(zip(frame_ontology.concept_names, frame_ontology.metadata.get("scope_notes", [])))
            headings = dict(zip(frame_ontology.concept_names, frame_ontology.metadata["headings"]))
            for e in entries:
                cid = _entry_concept_name(ctx, e)
                note = " ".join((notes.get(cid) or "").split())
                if note:
                    names[e] = headings.get(cid, cid)
                    texts[e] = (note, f"MeSH 2026 scope note ({cid})", LICENCES["mesh"], "scope")
            info["source"] = "MeSH 2026 ScopeNote of the preferred concept, read as '<headword>: <note>'"
        elif track in ("t7", "t7rood"):
            from ..ontologies.mesh_novel import parse_supplementary
            config = tracks._load_config(spec)["ontology"]
            records = {r["ui"]: r for r in parse_supplementary(Path(config["supplementary_path"]).expanduser())}
            for e in entries:
                record = records.get(_entry_concept_name(ctx, e)) or {}
                note = scr_definition(record.get("note") or "")
                if note and record.get("name"):
                    names[e] = record["name"]
                    texts[e] = (note, f"MeSH 2026 supplementary record note ({record['ui']}, introduced {record.get('introduced')})",
                                LICENCES["mesh"], "scr")
            info["source"] = ("MeSH 2026 supplementary concept record <Note>, bibliographic parts removed, read as "
                              "'<headword>: <note>'")
        else:
            raise ValueError(f"held-out definitions are not defined for track {track!r}")
    chosen = linkable_headwords(ctx, names, tokenizer, min_subtokens) if tokenizer is not None else {e: (n, "name") for e, n in names.items()}
    info["headwords"] = dict(Counter(source for _, source in chosen.values()))
    info["headwords"]["not linkable (left out)"] = len(names) - len(chosen)
    headwords = {e: h for e, (h, _) in chosen.items()}
    out: dict[int, list[dict[str, Any]]] = {}
    for e, headword in headwords.items():
        if track == "t5":
            facts = ctx.facts(ctx.frame(e))
            out[e] = [{"style": s, "text": rtl.t5_definition(headword, facts, s, random.Random(f"{seed}|{headword}|{s}")),
                       "source": "written from the gold frame", "licence": LICENCES["t5"]} for s in styles]
        else:
            body, source, licence, style = texts[e]
            out[e] = [{"style": style, "text": f"{headword}: {body}", "source": source, "licence": licence,
                       "display_name": names[e]}]
    return headwords, out, info


def eval_occurrences(corpus_path: Path, entries: Sequence[int], min_subtokens: int) -> Counter:
    from ..data.corpus import TokenCorpus
    corpus = TokenCorpus.open(Path(corpus_path))
    keep = (corpus.spans["length"] >= min_subtokens) & np.isin(corpus.spans["entry"], np.asarray(list(entries)))
    return Counter(int(e) for e in corpus.spans["entry"][keep])


def build_heldout_set(track: str, out_dir: Path, ctx: TrackContext, *, styles: Sequence[str] | None = None, seed: int = 0,
                      limit: int | None = None, include_synthetic: bool = True, wpc7_dir: Path | None = None,
                      eval_corpus: Path | None = None, min_subtokens: int = 2, tokenizer: Any = None) -> dict[str, Any]:
    """Held-out real terms of a track (never linked in training) with their definitions, gold frames, WP-C7 items and
    evaluation-text occurrence counts. T5 also reads its zero-shot synthetic terms (in no document: items only)."""
    from . import e9_tracks as tracks
    o = ctx.ontology
    styles = list(styles or (rtl.T5_STYLES if track == "t5" else [PRIMARY_STYLE[track]]))
    real = sorted(int(e) for e in o.get("heldout_real_entries", o["heldout_entries"]))
    synthetic = sorted(int(e) for e in o.get("synthetic_entries", ())) if include_synthetic and track == "t5" else []
    entries = [e for e in real + synthetic if len(ctx.table.entry_concepts[e]) == 1 and ctx.frame(e)]
    filters = {"held_out": len(real) + len(synthetic), "single_concept_with_frame": len(entries)}
    headwords, definitions, info = heldout_definitions(track, ctx, entries, styles, seed=seed, tokenizer=tokenizer,
                                                       min_subtokens=min_subtokens)
    entries = [e for e in entries if e in definitions]
    filters["with_definition_and_linkable_headword"] = len(entries)
    info["filters"] = filters
    if limit:
        entries = entries[:limit]
    random_frames = tracks.random_frames(o, entries, seed=seed)
    occurrences = eval_occurrences(eval_corpus, entries, min_subtokens) if eval_corpus else Counter()
    concepts, rows = [], []
    for e in entries:
        cid = f"{track}h-{e}"
        name = _entry_concept_name(ctx, e)
        concepts.append({"concept": cid, "surface": headwords[e], "entry": e, "frame": ctx.names(ctx.frame(e)),
                         "random_frame": ctx.names(random_frames[e]), "own_atom": ctx.own_atom(name), "source_concept": name,
                         "split": "synthetic" if e in synthetic else "heldout", "degree": len(ctx.frame(e)),
                         "eval_occurrences": int(occurrences.get(e, 0))})
        for d in definitions[e]:
            rows.append({"concept": cid, "headword": headwords[e], **d})
    by_entry = {c["entry"]: c["concept"] for c in concepts}
    items: list[dict[str, Any]] = []
    item_surface: dict[str, str] = {}
    if wpc7_dir is not None and (Path(wpc7_dir) / "zeroshot_property.jsonl").exists():
        wpc7_concepts, wpc7_items = tracks.convert_wpc7_items(Path(wpc7_dir))
        mapped: dict[str, tuple[int, str]] = {}
        for c in wpc7_concepts:
            entry = ctx.table.alias_to_entry.get(normalize_alias(c["surface"]))
            if entry in by_entry:
                mapped[c["concept"]] = (entry, c["surface"])
        for item in wpc7_items:
            if item["concept"] in mapped:
                entry, surface = mapped[item["concept"]]
                items.append({**item, "concept": by_entry[entry]})
                item_surface.setdefault(by_entry[entry], surface)
    # The WP-C7 items name the term by their own surface (prompts use it); the definition uses the headword.
    for c in concepts:
        c["item_surface"] = item_surface.get(c["concept"], c["surface"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "concepts.jsonl", concepts)
    _write_jsonl(out_dir / "definitions.jsonl", rows)
    _write_jsonl(out_dir / "items.jsonl", items)
    counts = Counter(c["split"] for c in concepts)
    result = _manifest("heldout", track, styles, concepts=len(concepts), splits=dict(counts), definitions=len(rows),
                       items=dict(Counter(i["test"] for i in items)), items_source=str(wpc7_dir) if wpc7_dir else None,
                       eval_corpus=str(eval_corpus) if eval_corpus else None,
                       terms_with_eval_occurrences=sum(1 for c in concepts if c["eval_occurrences"]),
                       eval_occurrences=sum(c["eval_occurrences"] for c in concepts), seed=seed,
                       held_out_real=len(real), held_out_synthetic=len(synthetic), **info,
                       licence=sorted({d["licence"] for d in rows}))
    write_json(out_dir / "manifest.json", result)
    return result


# -- textbook glossary (OpenStax) ---------------------------------------------------------------------------------------

OPENSTAX_REPO = "openstax/osbooks-chemistry-bundle"
OPENSTAX_COMMIT = "db0a8e6027100ce082e67fc8879faab86f9a58a7"          # main, 2026-09-24
CNXML = "{http://cnx.rice.edu/cnxml}"


def fetch_openstax(out_dir: Path, *, collection: str = "chemistry-2e", repo: str = OPENSTAX_REPO,
                   commit: str = OPENSTAX_COMMIT) -> dict[str, Any]:
    """Download the collection file and its modules' `index.cnxml` at a pinned commit (raw.githubusercontent.com), write
    `glossary.jsonl` (term, meaning, module) and `files.sha256`. Licence as stated in the collection metadata."""
    import urllib.request
    import xml.etree.ElementTree as ET
    out_dir = Path(out_dir).expanduser()
    (out_dir / "modules").mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{repo}/{commit}"

    def get(rel: str) -> bytes:
        path = out_dir / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{base}/{rel}", timeout=60) as response:
                path.write_bytes(response.read())
        return path.read_bytes()

    collection_xml = get(f"collections/{collection}.collection.xml").decode()
    licence = re.search(r'<md:license url="([^"]+)">([^<]+)<', collection_xml)
    modules = re.findall(r'<col:module document="(m\d+)"', collection_xml)
    rows, hashes = [], {}
    for module in modules:
        rel = f"modules/{module}/index.cnxml"
        data = get(rel)
        hashes[rel] = hashlib.sha256(data).hexdigest()
        root = ET.fromstring(data)
        title = root.findtext(f".//{CNXML}title") or ""
        for definition in root.iter(f"{CNXML}definition"):
            term = " ".join("".join(definition.find(f"{CNXML}term").itertext()).split()) if definition.find(f"{CNXML}term") is not None else ""
            meaning = definition.find(f"{CNXML}meaning")
            text = " ".join("".join(meaning.itertext()).split()) if meaning is not None else ""
            if term and text:
                rows.append({"term": term, "meaning": text, "module": module, "module_title": title})
    hashes[f"collections/{collection}.collection.xml"] = hashlib.sha256(collection_xml.encode()).hexdigest()
    _write_jsonl(out_dir / "glossary.jsonl", rows)
    (out_dir / "files.sha256").write_text("".join(f"{h}  {p}\n" for p, h in sorted(hashes.items())))
    info = {"repo": repo, "commit": commit, "collection": collection, "modules": len(modules), "glossary_terms": len(rows),
            "unique_terms": len({r["term"].lower() for r in rows}), "licence_url": licence[1] if licence else None,
            "licence": licence[2] if licence else None, "glossary_sha256": hashlib.sha256((out_dir / "glossary.jsonl").read_bytes()).hexdigest(),
            "retrieved": time.strftime("%Y-%m-%d")}
    write_json(out_dir / "source.json", info)
    return info


def _singular_variants(term: str) -> list[str]:
    key = normalize_alias(term)
    out = [key]
    if key.endswith("ies"):
        out.append(key[:-3] + "y")
    if key.endswith("es"):
        out.append(key[:-2])
    if key.endswith("s"):
        out.append(key[:-1])
    out.append(key + "s")
    return list(dict.fromkeys(out))


def build_swap_set(glossary: Path, out_dir: Path, ctx: TrackContext, *, source: str = "openstax", style: str = "glossary"
                   ) -> dict[str, Any]:
    """Textbook glossary terms mapped onto existing entries by alias (exact, then singular/plural variants); every mapped
    term is read as '<term>: <meaning>' and its read frame replaces the trained one (frame swap). Coverage is recorded."""
    rows = _read_jsonl(Path(glossary).expanduser())
    seen_terms: set[str] = set()
    concepts, definitions = [], []
    status = Counter()
    frequency = np.asarray(ctx.ontology.get("train_frequency") or np.zeros(len(ctx.table.entry_concepts)))
    heldout = {int(e) for e in ctx.ontology["heldout_entries"]}
    for row in rows:
        key = normalize_alias(row["term"])
        if key in seen_terms:
            status["duplicate term"] += 1
            continue
        seen_terms.add(key)
        entry = next((ctx.table.alias_to_entry[v] for v in _singular_variants(row["term"]) if v in ctx.table.alias_to_entry), None)
        if entry is None:
            status["unmapped"] += 1
            continue
        frame = ctx.frame(entry)
        if not frame or len(ctx.table.entry_concepts[entry]) != 1:
            status["mapped, no single-concept frame"] += 1
            continue
        status["mapped"] += 1
        name = _entry_concept_name(ctx, entry)
        cid = f"{ctx.track}s-{entry}"
        if any(c["concept"] == cid for c in concepts):
            status["entry already read"] += 1
            continue
        concepts.append({"concept": cid, "surface": row["term"], "item_surface": row["term"], "entry": entry,
                         "frame": ctx.names(frame), "random_frame": None, "own_atom": ctx.own_atom(name),
                         "source_concept": name, "split": "heldout" if entry in heldout else "seen", "degree": len(frame),
                         "train_frequency": int(frequency[entry]) if entry < len(frequency) else None})
        definitions.append({"concept": cid, "style": style, "text": f"{row['term']}: {row['meaning']}", "headword": row["term"],
                            "source": f"{source} glossary ({row.get('module')})", "licence": LICENCES.get(source, source)})
    from . import e9_tracks as tracks
    random_frames = tracks.random_frames(ctx.ontology, [c["entry"] for c in concepts], seed=0)
    for c in concepts:
        c["random_frame"] = ctx.names(random_frames[c["entry"]])
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "concepts.jsonl", concepts)
    _write_jsonl(out_dir / "definitions.jsonl", definitions)
    _write_jsonl(out_dir / "items.jsonl", [])
    result = _manifest("swap", ctx.track, [style], concepts=len(concepts), glossary=str(glossary), glossary_terms=len(rows),
                       coverage=dict(status), coverage_share=status["mapped"] / max(1, len(seen_terms)),
                       mapped_heldout=sum(c["split"] == "heldout" for c in concepts), licence=[LICENCES.get(source, source)])
    write_json(out_dir / "manifest.json", result)
    return result


# -- readers on a run -------------------------------------------------------------------------------------------------------

def read_tasks(read_set: ReadSet, ctx: TrackContext, styles: Sequence[str]) -> list[rtl.ReadTask]:
    concepts = {c["concept"]: c for c in read_set.concepts}
    tasks = []
    for d in read_set.definitions:
        if d["style"] not in styles:
            continue
        c = concepts[d["concept"]]
        tasks.append(rtl.ReadTask(d["concept"], d["style"], d["text"], d["headword"], ctx.ids(c["frame"]),
                                  c.get("own_atom"), ctx.ids(c.get("random_frame")) if c.get("random_frame") else None))
    return tasks


def link_entry_of(read_set: ReadSet, run: E5Run) -> dict[str, int]:
    """Entry each concept's headword links to: new words get fresh ids after the ontology's entries."""
    if read_set.kind == "new":
        base = int(run.ontology["entry_count"])
        return {c["concept"]: base + i for i, c in enumerate(read_set.concepts)}
    return {c["concept"]: int(c["entry"]) for c in read_set.concepts}


class DefinitionScorer:
    """`read_to_learn.Scorer` on a trained run: summed log-probability of a definition's tokens after the headword's
    first linked occurrence, with the headword's entry remapped to a variant entry that holds a candidate frame (new
    words) or replaced in the schedule (existing entries), or dropped (no row)."""

    def __init__(self, run: E5Run, read_set: ReadSet, *, batch_size: int | None = None) -> None:
        self.run, self.read_set = run, read_set
        self.entry_of = link_entry_of(read_set, run)
        self.adapter = edit.extended_adapter(run, {c["surface"]: self.entry_of[c["concept"]] for c in read_set.concepts}) \
            if read_set.kind == "new" else run.adapter
        self.batch_size = batch_size or max(4, run.adapter.batch_size // 2)
        self.forward_tokens = 0
        self.rows = 0
        self.unlinked_rows = 0                   # texts whose headword the linker did not link (all variants then tie)

    def token_count(self, text: str) -> int:
        return len(self.run.tokenizer(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def _logprobs(self, texts: Sequence[str], headword_spans: Sequence[tuple[int, int] | None], link: Sequence[int],
                  remap: Sequence[int | None]) -> np.ndarray:
        """Per text: Σ log p over tokens starting at or after the headword's end; `link[i]` is the entry the headword
        links to, `remap[i]` the entry injected instead (None: the headword's spans are dropped)."""
        tokenizer, device, model = self.run.tokenizer, self.run.device, self.run.model
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        out = np.zeros(len(texts))
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False, padding=True, truncation=True,
                                max_length=self.adapter.max_length, return_tensors="pt")
            ids = encoded["input_ids"]
            mask = encoded["attention_mask"]
            offsets = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in zip(encoded["offset_mapping"], mask)]
            spans = None
            if self.adapter.spans_fn is not None:
                spans = self.adapter.spans_fn(batch, offsets)
                keep = torch.ones(spans["entry"].numel(), dtype=torch.bool)
                entry = spans["entry"].clone()
                linked_rows = set()
                for k, (b, e) in enumerate(zip(spans["batch"].tolist(), spans["entry"].tolist())):
                    row = start + b
                    if e == link[row]:
                        linked_rows.add(b)
                        if remap[row] is None:
                            keep[k] = False
                        else:
                            entry[k] = int(remap[row])
                self.rows += len(batch)
                self.unlinked_rows += len(batch) - len(linked_rows)
                spans = {key: (entry if key == "entry" else value)[keep] for key, value in spans.items()}
                spans = {k: v.to(device) for k, v in spans.items()}
            labels = ids.masked_fill(mask == 0, -100).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses = model(ids.to(device), attention_mask=mask.to(device), spans=spans, labels=labels,
                               reduction="none")["loss"].float().cpu()
            self.forward_tokens += int(mask.sum())
            for b in range(len(batch)):
                span = headword_spans[start + b]
                end = span[1] if span else 0
                targets = [t for t in range(1, len(offsets[b])) if offsets[b][t][0] >= end]
                out[start + b] = -float(sum(losses[b, t - 1] for t in targets))
        return out

    def __call__(self, tasks: Sequence[rtl.ReadTask], variants: Sequence[Sequence[rtl.Frame | None]]) -> list[np.ndarray]:
        channel = self.run.channel
        rows = [(i, j, frame) for i, vs in enumerate(variants) for j, frame in enumerate(vs)]
        framed = [(i, j, frame) for i, j, frame in rows if frame]
        texts = [tasks[i].text for i, _, _ in rows]
        spans = [tasks[i].span for i, _, _ in rows]
        link = [self.entry_of[tasks[i].concept] for i, _, _ in rows]
        base = int(self.run.ontology["entry_count"]) + (len(self.read_set.concepts) if self.read_set.kind == "new" else 0)
        variant_id = {(i, j): base + k for k, (i, j, _) in enumerate(framed)}
        remap = [variant_id.get((i, j)) for i, j, _ in rows]
        dummy = [tasks[0].gold or framed[0][2]] * (len(self.read_set.concepts) if self.read_set.kind == "new" else 0)
        with contextlib.ExitStack() as stack:
            if self.read_set.kind == "new":
                stack.enter_context(edit.inserted_entries(channel, len(dummy), dummy))
            stack.enter_context(edit.inserted_entries(channel, len(framed), [f for _, _, f in framed]))
            values = self._logprobs(texts, spans, link, remap)
        out = [np.zeros(len(vs)) for vs in variants]
        for (i, j, _), v in zip(rows, values.tolist()):
            out[i][j] = v
        return out


def host_generator(run: E5Run, *, max_new_tokens: int = 96, batch: int = 8) -> rtl.Generator:
    """Greedy completions from the run's host weights (the channel is not used: it reads no spans here)."""
    from .e7_authoring import generate_samples
    host = run.model.model

    def generate(prompts: Sequence[str]) -> tuple[list[str], int, int]:
        completions, prompt_tokens, generated = generate_samples(host, run.tokenizer, list(prompts), run.device, samples=1,
                                                                 temperature=0.0, top_p=1.0, max_new_tokens=max_new_tokens,
                                                                 batch=batch, seed=0)
        return [c[0] for c in completions], prompt_tokens, generated
    return generate


def demonstrations_for(read_set: ReadSet, ctx: TrackContext, *, count: int = 2, seed: int = 0
                       ) -> Callable[[rtl.ReadTask], list[dict[str, Any]]]:
    """Style-matched demonstrations for the host reader: seen training entries (never a read term) with a definition
    written in the task's style from their gold frame (T5, T4-new) or, for natural definitions, a ChEBI / MeSH entry's
    own definition with its stated gold edges (edges whose filler the concept finder sees)."""
    o = ctx.ontology
    frequency = np.asarray(o.get("train_frequency") or np.zeros(len(ctx.table.entry_concepts)))
    heldout = {int(e) for e in o["heldout_entries"]} | {int(e) for e in o.get("synthetic_entries", ())}
    read_entries = {c.get("entry") for c in read_set.concepts}
    rng = random.Random(seed)
    pool = [e for e in range(len(ctx.table.entry_concepts)) if e not in heldout and e not in read_entries
            and len(ctx.table.entry_concepts[e]) == 1 and frequency[e] >= 10 and 3 <= len(ctx.frame(e)) <= 10]
    rng.shuffle(pool)
    chosen = pool[:64]
    natural: list[dict[str, Any]] = []
    if read_set.kind != "new" and ctx.track in {"t4", "t1"}:
        _, definitions, _ = heldout_definitions(ctx.track, ctx, chosen, [PRIMARY_STYLE[ctx.track]])
        for e in chosen:
            if e not in definitions:
                continue
            text = definitions[e][0]["text"]
            name = text.split(":", 1)[0]
            task = rtl.ReadTask("demo", "", text, name, ctx.frame(e))
            seen = {a for m in rtl.mentions_of(task, ctx.fillers) for a in m.atoms}
            edges = [(o["relation_names"][r], ctx.text(a)) for r, a in ctx.frame(e) if a in seen and ctx.text(a)]
            if edges:
                natural.append({"context": text, "surface": name, "edges": edges})
            if len(natural) >= count:
                break
        return lambda task: natural

    writer, styles = (rtl.t4_definition, rtl.T4_STYLES) if ctx.track == "t4" else (rtl.t5_definition, rtl.T5_STYLES)

    def display(e: int) -> str | None:
        """T5: the term's name; other tracks: the readable text of the atom the entry names (else skipped)."""
        name = _entry_concept_name(ctx, e)
        if ctx.track == "t5":
            return name
        atom = ctx.own_atom(name)
        return ctx.text(atom) if atom is not None else None

    def demos(task: rtl.ReadTask) -> list[dict[str, Any]]:
        style = task.style if task.style in styles else styles[0]
        out = []
        for e in chosen:
            name = display(e)
            facts = ctx.facts(ctx.frame(e))
            if ctx.track == "t4":                       # only what the ChEBI-style writer states
                facts = {r: ts for r, ts in facts.items() if r in {"is_a", "has_functional_parent", "has_role"}}
            if not name or not facts:
                continue
            text = writer(name, facts, style, random.Random(f"demo|{e}|{style}"))
            out.append({"context": text, "surface": name, "edges": [(r, t) for r, ts in facts.items() for t in ts]})
            if len(out) >= count:
                break
        return out
    return demos


def run_readers(run: E5Run | None, read_set: ReadSet, ctx: TrackContext, readers: Sequence[str], styles: Sequence[str], *,
                scorer: rtl.Scorer | None = None, generator: rtl.Generator | None = None, teacher_frames: Path | None = None,
                log: Callable[[str], None] = print) -> tuple[list[rtl.ReadResult], dict[tuple[str, str], list[rtl.Mention]]]:
    """Every requested reader on every (concept, style) definition; model readers need `scorer` / `generator`."""
    tasks = read_tasks(read_set, ctx, styles)
    mentions = {(t.concept, t.style): rtl.mentions_of(t, ctx.fillers) for t in tasks}
    results: list[rtl.ReadResult] = []
    for name in readers:
        if name in {"linker-all"}:
            continue
        log(f"  reader {name}")
        if name == "oracle":
            results += [rtl.read_oracle(t) for t in tasks]
        elif name == "stated":
            results += [rtl.read_stated(t, mentions[(t.concept, t.style)]) for t in tasks]
        elif name == "typeprior":
            results += [rtl.read_typeprior(t, mentions[(t.concept, t.style)], ctx.typing) for t in tasks]
        elif name == "pattern":
            results += [rtl.read_pattern(t, mentions[(t.concept, t.style)], ctx.typing, ctx.fillers) for t in tasks]
        elif name == "random":
            results += [rtl.read_random(t) for t in tasks]
        elif name == "none":
            results += [rtl.read_none(t) for t in tasks]
        elif name == "linker":
            if scorer is None:
                continue
            token_count = getattr(scorer, "token_count", None)
            kept, every = rtl.read_linker(tasks, [mentions[(t.concept, t.style)] for t in tasks], ctx.typing, scorer,
                                          token_count=token_count)
            results += kept + (every if "linker-all" in readers else [])
        elif name == "linker-joint":
            if scorer is None:
                continue
            results += rtl.read_linker_joint(tasks, [mentions[(t.concept, t.style)] for t in tasks], ctx.typing, scorer,
                                             token_count=getattr(scorer, "token_count", None))
        elif name == "host":
            if generator is None:
                continue
            demos = demonstrations_for(read_set, ctx)
            results += rtl.read_host(tasks, ctx.fillers, ctx.typing, generator, demos)
        elif name == "teacher":
            cached = {(r["concept"], r["style"]): r for r in _read_jsonl(teacher_frames)} if teacher_frames else {}
            for t in tasks:
                row = cached.get((t.concept, t.style))
                if row is not None:
                    results.append(rtl.ReadResult(t.concept, t.style, "teacher", ctx.ids(row["frame"]), row.get("cost", {}),
                                                  row.get("details", {})))
        else:
            raise ValueError(f"unknown reader {name!r}")
    return results, mentions


def frame_rows(results: Sequence[rtl.ReadResult], read_set: ReadSet, ctx: TrackContext,
               mentions: dict[tuple[str, str], list[rtl.Mention]]) -> list[dict[str, Any]]:
    gold = {c["concept"]: ctx.ids(c["frame"]) for c in read_set.concepts}
    rows = []
    ambiguous_cache: dict[int, bool] = {}

    def ambiguous(atom: int) -> bool:
        if atom not in ambiguous_cache:
            ambiguous_cache[atom] = len(ctx.typing.candidates(atom)) > 1
        return ambiguous_cache[atom]

    for r in results:
        found = {a for m in mentions.get((r.concept, r.style), []) for a in m.atoms}
        ambiguous_atoms = {a for _, a in gold[r.concept] if ambiguous(a)}
        rows.append({"concept": r.concept, "style": r.style, "reader": r.reader, "frame": ctx.names(r.frame),
                     "metrics": rtl.frame_metrics(r.frame, gold[r.concept], found_atoms=found, ambiguous_atoms=ambiguous_atoms),
                     "cost": r.cost, "details": r.details})
    return rows


def summarize_frames(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per style and reader: micro precision / recall / F1 (pooled edges), mean filler recall, relation accuracy, empty
    frames, mean edges and mean cost."""
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["style"], r["reader"])].append(r)
    for (style, reader), members in sorted(groups.items()):
        m = [x["metrics"] for x in members]
        hits, pred, gold = sum(x["hits"] for x in m), sum(x["edges"] for x in m), sum(x["gold_edges"] for x in m)
        stated = sum(x["stated_edges"] for x in m)
        stated_hits = sum((x["stated_recall"] or 0) * x["stated_edges"] for x in m)
        precision, recall = (hits / pred if pred else None), (hits / gold if gold else None)
        mean = lambda key: float(np.mean([x[key] for x in m if x[key] is not None])) if any(x[key] is not None for x in m) else None
        costs: dict[str, float] = defaultdict(float)
        for x in members:
            for k, v in (x["cost"] or {}).items():
                costs[k] += float(v or 0)
        out.setdefault(style, {})[reader] = {
            "frames": len(members), "empty": sum(1 for x in members if not x["frame"]), "edges_mean": pred / max(1, len(members)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0 if pred else None,
            "stated_recall": stated_hits / stated if stated else None, "filler_recall": mean("filler_recall"),
            "relation_accuracy": mean("relation_accuracy"),
            "ambiguous_relation_accuracy": (sum(x["ambiguous_relation_hits"] for x in m) / sum(x["ambiguous_edges"] for x in m)
                                            if sum(x["ambiguous_edges"] for x in m) else None),
            "ambiguous_edges": sum(x["ambiguous_edges"] for x in m),
            "cost_per_word": {k: v / max(1, len(members)) for k, v in sorted(costs.items())}}
    return out


# -- persistence: item tests with each reader's frame (no definition in context) ---------------------------------------------

@contextlib.contextmanager
def swapped_frames(channel: Any, frames: dict[int, rtl.Frame | None]) -> Iterator[bool]:
    """Within the block existing entries read the given frames (None or [] = no row: `skip_empty_frames`)."""
    if channel is None or channel.mode != "compose" or not frames:
        yield False
        return
    composer = channel.composer
    original, skip = composer.schedule, channel.skip_empty_frames
    composer.set_schedule(replace_frames(original, {e: list(f or []) for e, f in frames.items()}))
    channel.skip_empty_frames = True
    try:
        yield True
    finally:
        composer.set_schedule(original)
        channel.skip_empty_frames = skip


class ItemScorer:
    """Scores a read set's items under a condition: per concept a frame (or None) and an optional context prepended to
    every prompt. New words are inserted as fresh entries (`inserted_entries`, rows of `None` concepts zeroed); held-out
    terms get their frames swapped. Results of the `new` kind are cached per (concept, frame, context) — rows of
    different entries are independent and an invented name never appears in another concept's prompt."""

    def __init__(self, run: E5Run, read_set: ReadSet, ctx: TrackContext) -> None:
        self.run, self.read_set, self.ctx = run, read_set, ctx
        self.entry_of = link_entry_of(read_set, run)
        self.concepts = [c["concept"] for c in read_set.concepts]
        self.surface = {c["concept"]: c.get("item_surface") or c["surface"] for c in read_set.concepts}
        self.adapter = edit.extended_adapter(run, {c["surface"]: self.entry_of[c["concept"]] for c in read_set.concepts}) \
            if read_set.kind == "new" else run.adapter
        self.items_by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in read_set.items:
            self.items_by_concept[item["concept"]].append(item)
        self.gold = {c["concept"]: ctx.ids(c["frame"]) for c in read_set.concepts}
        self.cache: dict[tuple, list[dict[str, Any]]] = {}
        self.passes = 0

    @staticmethod
    def _key(frame: rtl.Frame | None) -> tuple:
        return tuple(sorted(frame)) if frame else ()

    @contextlib.contextmanager
    def condition(self, frames: dict[str, rtl.Frame | None]) -> Iterator[None]:
        channel = self.run.channel
        compose = channel is not None and channel.mode == "compose"
        with contextlib.ExitStack() as stack:
            if self.read_set.kind == "new":
                inserted = [frames.get(c) or self.gold[c] for c in self.concepts]
                stack.enter_context(edit.inserted_entries(channel, len(self.concepts), inserted))
                zero = [self.entry_of[c] for c in self.concepts if not frames.get(c)]
                if channel is not None and zero:
                    width = channel.gate.in_features // 2
                    stack.enter_context(override_rows(channel, {e: torch.zeros(width) for e in zero}))
            elif compose:
                stack.enter_context(swapped_frames(channel, {self.entry_of[c]: frames.get(c) for c in self.concepts}))
            yield

    def score(self, frames: dict[str, rtl.Frame | None], contexts: dict[str, str] | None = None,
              concepts: Sequence[str] | None = None, *, prefix: "PrefixCache | str | None" = None, tag: str | None = None
              ) -> dict[str, list[dict[str, Any]]]:
        """Per concept: result rows of its items (prompts: PMI correctness, paraphrase consistency; statements). With a
        cached `prefix` every prompt (and its null) is read after it; `tag` names the prefix in the result cache."""
        concepts = list(concepts or self.concepts)
        cacheable = self.read_set.kind == "new" and (prefix is None or tag is not None)
        key_of = {c: (c, self._key(frames.get(c)), (contexts or {}).get(c), tag) for c in concepts}
        todo = [c for c in concepts if not (cacheable and key_of[c] in self.cache)]
        if todo:
            items = [i for c in todo for i in self.items_by_concept[c]]
            if contexts:
                from .e9_dim3_baselines import with_context
                items = with_context(items, {i["id"]: contexts[i["concept"]] + "\n" for i in items if i["concept"] in contexts})
            prompt_items = [i for i in items if i["test"] in {"property", "entailment"}]
            statement_items = [i for i in items if i["test"] == "statement"]
            # Prompts with a definition in front are ≈ 3–5× longer: half the batch keeps the peak memory of a plain pass.
            with self.condition(frames), _smaller(self.adapter, 2 if contexts or prefix is not None else 1):
                # A prefix given as text is read here, inside the condition: its new names link to inserted entries.
                cache = PrefixCache(self.run, self.adapter, prefix) if isinstance(prefix, str) else prefix
                with prefixed_continuations(cache) if cache is not None else fast_continuations():
                    prompts, _ = zs.score_prompts(self.adapter, prompt_items, self.surface) if prompt_items else ([], None)
                    statements = edit.score_statements(self.adapter, statement_items, self.surface) if statement_items else []
            self.passes += 1
            kinds = {i["id"]: i.get("edge_kind") for i in items}
            fresh: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in prompts + statements:
                row = {k: v for k, v in row.items() if k != "pmi"}
                row["edge_kind"] = kinds.get(row["id"], row.get("edge_kind"))
                row.setdefault("consistent", None)
                fresh[row["concept"]].append(row)
            for c in todo:
                self.cache[key_of[c]] = fresh.get(c, [])
        return {c: self.cache[key_of[c]] for c in concepts}


class PrefixCache:
    """A context prefix (e.g. definitions read in context) run once through the model, its key/value cache reused for
    every text scored after it (E11-M, preregistration §14). Texts are tokenized on their own and placed at positions
    P … P+T−1 after the P prefix tokens; their spans (channel injections) are linked within the text. One difference
    from scoring `prefix + text` jointly: the channel's causal context query (`context_window` tokens) does not reach
    back into the prefix for injections in a text's first few tokens (tested close; recorded)."""

    def __init__(self, run: E5Run, adapter: Any, text: str | None = None, *, ids: torch.Tensor | None = None,
                 spans: dict[str, torch.Tensor] | None = None) -> None:
        self.run, self.adapter = run, adapter
        model, device, tokenizer = run.model, run.device, run.tokenizer
        if ids is None:
            encoded = tokenizer([text or ""], return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
            ids = encoded["input_ids"]
            if getattr(adapter, "spans_fn", None) is not None and model.channel is not None and ids.shape[1]:
                spans = adapter.spans_fn([text], [[tuple(o) for o in encoded["offset_mapping"][0].tolist()]])
        self.length = int(ids.shape[1])
        self.legacy = None
        if self.length:
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                embeddings = model.embed(ids.to(device), None if spans is None else {k: v.to(device) for k, v in spans.items()})
                out = model.base(inputs_embeds=embeddings, use_cache=True)
            cache = out.past_key_values
            self.legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)

    def _cache(self, batch: int) -> Any:
        from transformers import DynamicCache
        return DynamicCache.from_legacy_cache(tuple((k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in self.legacy))

    @torch.no_grad()
    def hidden(self, ids: torch.Tensor, attention: torch.Tensor, spans: dict[str, torch.Tensor] | None) -> torch.Tensor:
        """Final hidden states (B × T × d) of `ids` read after the prefix."""
        model, device = self.run.model, self.run.device
        ids, attention = ids.to(device), attention.to(device)
        spans = None if spans is None else {k: v.to(device) for k, v in spans.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            embeddings = model.embed(ids, spans)
            if not self.length:
                return model.base(inputs_embeds=embeddings, attention_mask=attention).last_hidden_state
            mask = torch.cat([torch.ones(ids.shape[0], self.length, dtype=attention.dtype, device=device), attention], 1)
            out = model.base(inputs_embeds=embeddings, attention_mask=mask, past_key_values=self._cache(ids.shape[0]), use_cache=True)
        return out.last_hidden_state


@torch.no_grad()
def continuation_scores(adapter: Any, prefixes: Sequence[str], continuations: Sequence[str], *,
                        prefix: PrefixCache | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Σ log p(continuation | prefix) and the number of continuation tokens, as `channel_probes.continuation_logprob` /
    `e9_ontology_edit.continuation_stats` compute them (same batch forward, spans, autocast and output head), but with
    output logits only at the positions that predict a continuation token and with texts batched by length. The shared
    adapter materializes full-vocabulary log-probabilities at every position, which dominates the cost of prompts with a
    definition in front (tested equal on the CPU)."""
    from torch.nn import functional as F
    texts = [p + c for p, c in zip(prefixes, continuations)]
    sums, counts = np.zeros(len(texts)), np.zeros(len(texts))
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), adapter.batch_size):
        chunk = order[start:start + adapter.batch_size]
        if prefix is None:
            encoded, out, head = adapter._run_batch([texts[i] for i in chunk])
            hidden = out.hidden_states[-1]
        else:
            encoded, hidden, head = _prefixed_batch(adapter, prefix, [texts[i] for i in chunk])
        mask = encoded["attention_mask"]
        rows, cols, targets, owners = [], [], [], []
        for r, i in enumerate(chunk):
            n = int(mask[r].sum())
            offsets = encoded["offset_mapping"][r, :n].tolist()
            for t in range(1, n):
                if offsets[t][1] > len(prefixes[i]):
                    rows.append(r); cols.append(t - 1); targets.append(int(encoded["input_ids"][r, t])); owners.append(i)
        if not rows:
            continue
        with adapter._autocast():
            logits = F.linear(hidden[torch.tensor(rows, device=hidden.device), torch.tensor(cols, device=hidden.device)].float(),
                              head.weight.float())
        picked = torch.log_softmax(logits.float(), -1).gather(-1, torch.tensor(targets, device=logits.device)[:, None]).squeeze(-1)
        np.add.at(sums, owners, picked.cpu().double().numpy())
        np.add.at(counts, owners, 1.0)
    return sums, counts


def _prefixed_batch(adapter: Any, prefix: PrefixCache, batch: list[str]) -> tuple[Any, torch.Tensor, Any]:
    """(encoding, final hidden states, output head) of a right-padded batch read after `prefix` (`_run_batch`'s contract)."""
    tokenizer = adapter.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False, padding=True, truncation=True,
                        max_length=adapter.max_length, return_tensors="pt")
    spans = None
    if adapter.spans_fn is not None:
        offsets = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in
                   zip(encoded["offset_mapping"], encoded["attention_mask"])]
        spans = adapter.spans_fn(batch, offsets)
    hidden = prefix.hidden(encoded["input_ids"], encoded["attention_mask"], spans)
    return encoded, hidden, prefix.run.model.model.get_output_embeddings()


@contextlib.contextmanager
def patched_continuations(scorer: Callable[..., tuple[np.ndarray, np.ndarray]]) -> Iterator[None]:
    """Within the block the E5.4 / E9 scorers (`zs.score_prompts`, `edit.score_statements`) use `scorer`."""
    saved = cp.continuation_logprob, edit.continuation_stats
    cp.continuation_logprob = lambda adapter, prefixes, continuations: scorer(adapter, prefixes, continuations)[0]
    edit.continuation_stats = scorer
    try:
        yield
    finally:
        cp.continuation_logprob, edit.continuation_stats = saved


def fast_continuations() -> contextlib.AbstractContextManager:
    """`patched_continuations(continuation_scores)`: the E9 scorers with logits only at continuation positions."""
    return patched_continuations(continuation_scores)


def prefixed_continuations(prefix: PrefixCache) -> contextlib.AbstractContextManager:
    """The E9 scorers reading every prompt after a cached context `prefix` (definitions in context, E11-M)."""
    return patched_continuations(lambda adapter, p, c: continuation_scores(adapter, p, c, prefix=prefix))


@contextlib.contextmanager
def _smaller(adapter: Any, factor: int) -> Iterator[None]:
    saved = adapter.batch_size
    adapter.batch_size = max(1, saved // factor)
    try:
        yield
    finally:
        adapter.batch_size = saved


def test_values(rows_by_concept: dict[str, list[dict[str, Any]]], kind: str) -> dict[str, dict[str, float]]:
    """Per test: item id → value (the E9 dimension-3 tests for new words; property / paraphrase / entailment otherwise)."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for rows in rows_by_concept.values():
        for r in rows:
            if r["test"] == "property":
                out["property"][r["id"]] = r["correct"]
                if r.get("edge_kind") == "resampled":
                    out["property_new"][r["id"]] = r["correct"]
                if r.get("consistent") is not None:
                    out["paraphrase"][r["id"]] = float(r["consistent"])
            elif r["test"] == "entailment":
                out["entailment"][r["id"]] = r["correct"]
            elif r["test"] == "statement":
                out["statement_accuracy"][r["id"]] = r["correct"]
                out["statement_loss"][r["id"]] = r["gold_loss"]
    return out


def compare(a: dict[str, float], b: dict[str, float], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return zs.paired_difference(np.asarray([a[i] for i in ids]), np.asarray([b[i] for i in ids]), resamples=resamples, seed=seed)


# -- gradient baseline ----------------------------------------------------------------------------------------------------

def host_parameters(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if n.startswith("model.")]


@contextlib.contextmanager
def host_snapshot(model: Any, *, backup: str = "gpu") -> Iterator[Callable[[], None]]:
    """Within the block host parameters are trainable (channel and context frozen); `restore()` puts the saved weights
    back; on exit the weights and every `requires_grad` flag are restored."""
    params = host_parameters(model)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    device = "cpu" if backup == "cpu" else None
    saved = [p.detach().to(device, copy=True) if device else p.detach().clone() for _, p in params]
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("model."))

    @torch.no_grad()
    def restore() -> None:
        for (_, p), s in zip(params, saved):
            p.copy_(s.to(p.device, non_blocking=True))
            p.grad = None
    try:
        yield restore
    finally:
        restore()
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])


def gradient_steps(run: E5Run, adapter: Any, text: str, headword_entry: int | None, *, steps: int, lr: float,
                   optimizer: str = "adam", clip: float = 1.0) -> float:
    """`steps` updates of the host on the language-model loss of `text` (the headword's own row dropped: the host learns
    from the text alone); returns the last loss."""
    model, tokenizer, device = run.model, run.tokenizer, run.device
    params = [p for _, p in host_parameters(model)]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95)) if optimizer == "adam" else torch.optim.SGD(params, lr=lr)
    encoded = tokenizer([text], return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=adapter.max_length,
                        return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    spans = None
    if getattr(adapter, "spans_fn", None) is not None and model.channel is not None:
        offsets = [[tuple(o) for o in encoded["offset_mapping"][0].tolist()]]
        spans = adapter.spans_fn([text], offsets)
        keep = spans["entry"] != (headword_entry if headword_entry is not None else -1)
        spans = {k: v[keep].to(device) for k, v in spans.items()}
    loss_value = float("nan")
    model.eval()
    for _ in range(steps):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(ids, spans=spans, labels=ids)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
    del opt
    return loss_value


@torch.no_grad()
def general_loss(run: E5Run, windows: Sequence[np.ndarray]) -> float:
    """Mean token loss of fixed general-text windows (no spans): the gradient update's locality probe."""
    if not windows:
        return float("nan")
    device = run.device
    ids = torch.as_tensor(np.stack(windows), dtype=torch.long, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = run.model(ids, spans=None, labels=ids)["loss"]
    return float(loss)


def general_windows(run: E5Run, *, count: int = 8, length: int = 512, seed: int = 0) -> list[np.ndarray]:
    from ..data.corpus import TokenCorpus
    from . import e9_tracks as tracks
    try:
        spec = tracks.track_spec(run.config.get("e9_track") or "wordnet", run.config.get("e9_family") or "smollm2")
        path = spec.general_corpus or Path(run.config["data"]["eval"])
        corpus = TokenCorpus.open(Path(path))
    except Exception:                                  # noqa: BLE001 — tests and runs without a track: the run's eval text
        corpus = TokenCorpus.open(Path(run.config["data"]["eval"]))
    length = min(length, int(run.config["model"]["seq_len"]), len(corpus) - 2)
    rng = np.random.default_rng(seed)
    starts = sorted(rng.choice(max(1, len(corpus) - length - 1), size=min(count, max(1, len(corpus) - length - 1)), replace=False).tolist())
    return [np.asarray(corpus.tokens[s:s + length], dtype=np.int64) for s in starts]


def matched_steps(task_candidates: int, *, factor: float = 1.0) -> int:
    """Compute-matched gradient steps: the linker reader scores (1 + candidates) forward passes over the definition; one
    training step costs ≈ 3 forward passes of the same text."""
    return max(1, int(round(factor * (task_candidates + 1) / 3.0)))


def evaluate_gradient(run: E5Run, read_set: ReadSet, ctx: TrackContext, scorer: ItemScorer, *, style: str, lr: float,
                      factors: Sequence[float] = (1.0, 4.0), optimizer: str = "adam", backup: str = "gpu",
                      locality_windows: int = 8, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Per term: k compute-matched host updates on its definition, its items scored (no definition in context, no frame),
    the general-text loss change, weights restored. Returns per factor the item rows and per-term records."""
    tasks = {t.concept: t for t in read_tasks(read_set, ctx, [style])}
    general = general_windows(run, count=locality_windows)
    before_general = general_loss(run, general)
    entry_of = link_entry_of(read_set, run)
    out: dict[str, Any] = {"lr": lr, "optimizer": optimizer, "style": style, "general_before": before_general, "factors": {}}
    for factor in factors:
        rows: dict[str, list[dict[str, Any]]] = {}
        records = []
        started = time.monotonic()
        with host_snapshot(run.model, backup=backup) as restore:
            for concept, task in tasks.items():
                k = matched_steps(len(rtl.linker_candidates(task, rtl.mentions_of(task, ctx.fillers), ctx.typing)), factor=factor)
                # The headword's own spans are dropped during the update (no entry needs to exist for it).
                loss = gradient_steps(run, scorer.adapter, task.text, entry_of[concept], steps=k, lr=lr, optimizer=optimizer)
                rows[concept] = _score_uncached(scorer, concept)
                general_after = general_loss(run, general) if general else float("nan")
                tokens = scorer.adapter.tokenizer(task.text, add_special_tokens=False)["input_ids"]
                records.append({"concept": concept, "steps": k, "last_loss": loss, "definition_tokens": len(tokens),
                                "train_tokens": k * len(tokens), "general_delta": general_after - before_general})
                restore()
        out["factors"][str(factor)] = {"rows": rows, "records": records, "seconds": time.monotonic() - started}
        log(f"  gradient ×{factor}: {len(records)} terms, {time.monotonic() - started:.0f}s")
    return out


def _score_uncached(scorer: ItemScorer, concept: str) -> list[dict[str, Any]]:
    """One concept's items scored now (bypassing the frame cache: the weights changed), no frame, no context."""
    saved = scorer.cache
    scorer.cache = {}
    try:
        return scorer.score({concept: None}, None, [concept])[concept]
    finally:
        scorer.cache = saved


# -- natural-text use: loss after the read terms -------------------------------------------------------------------------------

@dataclasses.dataclass
class WindowSet:
    starts: list[int]
    length: int
    spans: list[list[tuple[int, int, int, bool]]]        # per window: (span end token, entry, term index, own-document)
    terms: list[int]                                     # entries, in term-index order
    excluded_own: int = 0


def build_window_set(run: E5Run, entries: Sequence[int], headwords: dict[int, str], *, length: int = 1024,
                     max_windows: int | None = None, corpus_path: Path | None = None) -> WindowSet:
    """Consecutive `length`-token windows (at most the run's sequence length) of the run's evaluation corpus that hold
    ≥ 1 linked occurrence of a read term (spans fully inside the window, ≥ ℓ_min subtokens). An occurrence inside the
    term's own definition document (a document containing '<headword>:', e.g. its ChEBI entry text or glossary page) is
    flagged `own`."""
    from ..data.corpus import TokenCorpus
    corpus = TokenCorpus.open(Path(corpus_path or run.config["data"]["eval"]))
    length = min(length, int(run.config["model"]["seq_len"]))
    min_subtokens = int(run.config["data"]["min_subtokens"])
    index = {int(e): i for i, e in enumerate(entries)}
    eos = run.tokenizer.eos_token_id
    starts, spans_all, own_count = [], [], 0
    markers = {e: normalize_alias(h) + ":" for e, h in headwords.items()}
    for start in range(0, len(corpus) - length - 1, length):
        ids, spans = corpus.window(start, length, min_subtokens=min_subtokens)
        hits = [(int(e), int(en)) for e, en in zip(spans["entry"].tolist(), spans["end"].tolist()) if int(e) in index]
        if not hits:
            continue
        boundaries = [-1] + [i for i, t in enumerate(ids.tolist()) if t == eos] + [len(ids)]
        decoded: dict[int, str] = {}
        row = []
        for entry, end in hits:
            doc = bisect.bisect_left(boundaries, end) - 1          # the document whose tokens hold the span's end
            if doc not in decoded:
                decoded[doc] = normalize_alias(run.tokenizer.decode(ids[boundaries[doc] + 1:boundaries[doc + 1]].tolist()))
            own = markers[entry] in decoded[doc]
            own_count += own
            row.append((end, entry, index[entry], own))
        starts.append(start)
        spans_all.append(row)
        if max_windows and len(starts) >= max_windows:
            break
    return WindowSet(starts, length, spans_all, list(entries), own_count)


@torch.no_grad()
def window_term_losses(run: E5Run, corpus_path: Path, windows: WindowSet, *, batch: int = 8,
                       select: Callable[[int, int], bool] | None = None) -> dict[str, np.ndarray]:
    """Per term: summed loss and target count over the `AFTER_WINDOW` targets after its occurrences (split own / other
    document), plus per-window unlinked-token loss (targets neither inside nor after any span) — under whatever
    frames the channel currently reads. `select(window index, term index)` restricts the counted occurrences."""
    from ..data.corpus import TokenCorpus, collate_windows
    from ..training.lm import stratum_masks
    corpus = TokenCorpus.open(Path(corpus_path))
    min_subtokens = int(run.config["data"]["min_subtokens"])
    n_terms = len(windows.terms)
    sums = {k: np.zeros(n_terms) for k in ("other", "own")}
    counts = {k: np.zeros(n_terms) for k in ("other", "own")}
    unlinked = np.zeros((len(windows.starts), 2))
    device = run.device
    order = list(range(len(windows.starts)))
    if select is not None:
        order = [w for w in order if any(select(w, t) for _, _, t, _ in windows.spans[w])]
    for b0 in range(0, len(order), batch):
        chunk = order[b0:b0 + batch]
        parts = [corpus.window(windows.starts[w], windows.length, min_subtokens=min_subtokens) for w in chunk]
        ids, spans = collate_windows(parts)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = run.model(ids.to(device), spans={k: v.to(device) for k, v in spans.items()} if run.model.channel else None,
                               labels=ids.to(device), reduction="none")["loss"].float().cpu().numpy()
        masks = stratum_masks(ids, spans, None, set())
        for row, w in enumerate(chunk):
            un = masks["unlinked"][row].numpy()
            unlinked[w] = (losses[row][un].sum(), un.sum())
            for end, _, t, own in windows.spans[w]:
                if select is not None and not select(w, t):
                    continue
                lo, hi = end, min(windows.length - 1, end + AFTER_WINDOW)
                if lo >= hi:
                    continue
                key = "own" if own else "other"
                sums[key][t] += float(losses[row, lo:hi].sum())
                counts[key][t] += hi - lo
    return {"sum_other": sums["other"], "count_other": counts["other"], "sum_own": sums["own"], "count_own": counts["own"],
            "unlinked": unlinked}


def relative_change(cond: dict[str, np.ndarray], ref: dict[str, np.ndarray], *, part: str = "other", resamples: int = 2000,
                    seed: int = 0) -> dict[str, Any] | None:
    """Token-weighted loss change after the read terms (cluster bootstrap over terms), condition − reference."""
    n = ref[f"count_{part}"]
    if n.sum() <= 0:
        return None
    d = cond[f"sum_{part}"] - ref[f"sum_{part}"]
    return paired_ratio_bootstrap(d, n, ref[f"sum_{part}"], resamples=resamples, seed=seed)


# -- locality ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def row_locality(run: E5Run, read_set: ReadSet, frames: dict[str, rtl.Frame | None], *, sample: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Max |Δ| of the context-free rows of `sample` other entries when the read frames are written (exactly 0 on the
    CPU; bf16 / CUDA summation order can add ≈ 1e-6 noise)."""
    channel = run.channel
    if channel is None or channel.mode != "compose":
        return {"applicable": False}
    read_entries = np.asarray(sorted(set(link_entry_of(read_set, run).values())))
    rng = np.random.default_rng(seed)
    count = int(run.ontology["entry_count"])
    degrees = run.composer.schedule.degrees[:count].cpu().numpy()
    usable = (degrees > 0) & ~np.isin(np.arange(count), read_entries)
    pool = np.flatnonzero(usable)
    others = torch.as_tensor(np.sort(rng.choice(pool, size=min(sample, len(pool)), replace=False)))
    before = entry_rows(channel, others)
    with contextlib.ExitStack() as stack:
        if read_set.kind == "new":
            relation_id = {n: i for i, n in enumerate(run.ontology["relation_names"])}
            atom_id = {n: i for i, n in enumerate(run.ontology["atomic_names"])}
            inserted = [frames.get(c["concept"]) or edit.resolve_frame(c["frame"], relation_id, atom_id) for c in read_set.concepts]
            stack.enter_context(edit.inserted_entries(channel, len(inserted), inserted))
        else:
            entry_of = link_entry_of(read_set, run)
            stack.enter_context(swapped_frames(channel, {entry_of[c]: f for c, f in frames.items()}))
        after = entry_rows(channel, others)
    return {"applicable": True, "entries": int(others.numel()), "max_abs_row_change": float((after - before).abs().max())}


# -- evaluate ----------------------------------------------------------------------------------------------------------------

def frames_by_condition(results: Sequence[rtl.ReadResult]) -> dict[tuple[str, str], dict[str, rtl.Frame | None]]:
    out: dict[tuple[str, str], dict[str, rtl.Frame | None]] = defaultdict(dict)
    for r in results:
        out[(r.style, r.reader)][r.concept] = r.frame
    return out


def evaluate(run: E5Run, read_set: ReadSet, *, methods: Sequence[str], readers: Sequence[str], styles: Sequence[str],
             primary_style: str, gradient_lr: float | None = None, gradient_factors: Sequence[float] = (1.0, 4.0),
             gradient_optimizer: str = "adam", gradient_backup: str = "gpu", teacher_frames: Path | None = None,
             max_windows: int | None = None, window_gradient: bool = False, resamples: int = 2000, seed: int = 0,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    started = time.monotonic()
    ctx = run_context(run)
    compose = run.channel is not None and run.channel.mode == "compose"
    styles = [s for s in styles if s in read_set.styles]
    model_readers = [r for r in readers if r in rtl.MODEL_READERS]
    if not compose:                                      # frames are only consumed by a composing channel
        readers = [r for r in readers if r in {"none"}] or ["none"]
        model_readers = []
    scorer = DefinitionScorer(run, read_set) if compose and {"linker", "linker-all", "linker-joint"} & set(readers) else None
    generator = host_generator(run) if compose and "host" in readers else None
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    results, mentions = run_readers(run, read_set, ctx, readers, styles, scorer=scorer, generator=generator,
                                    teacher_frames=teacher_frames, log=log)
    timings["readers"] = time.monotonic() - t0
    frames = frame_rows(results, read_set, ctx, mentions)
    document: dict[str, Any] = {"set": {"path": str(read_set.path), "kind": read_set.kind, "track": read_set.manifest["track"],
                                        "concepts": len(read_set.concepts), "items": len(read_set.items)},
                                "styles": styles, "primary_style": primary_style, "readers": sorted({r.reader for r in results}),
                                "model_readers": model_readers, "compose": compose,
                                "frames": summarize_frames(frames) if "frames" in methods else {},
                                "definition_forward_tokens": scorer.forward_tokens if scorer else 0,
                                "definition_rows": {"scored": scorer.rows, "headword_unlinked": scorer.unlinked_rows} if scorer else None}
    conditions = frames_by_condition(results)
    item_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    item_scorer = ItemScorer(run, read_set, ctx) if read_set.items and {"persistence", "context", "gradient"} & set(methods) else None
    if item_scorer is not None and "persistence" in methods:
        t0 = time.monotonic()
        for (style, reader), by_concept in sorted(conditions.items()):
            if reader != "none" and not compose:
                continue
            log(f"  persistence: {style} / {reader}")
            item_rows[f"{style}|{reader}"] = item_scorer.score(by_concept)
        timings["persistence"] = time.monotonic() - t0
    if item_scorer is not None and "context" in methods:
        t0 = time.monotonic()
        definitions = {(d["concept"], d["style"]): d["text"] for d in read_set.definitions}
        context_styles = styles if compose else [primary_style]
        for style in context_styles:
            contexts = {c: definitions[(c, style)] for c in item_scorer.concepts if (c, style) in definitions}
            log(f"  context: {style}")
            item_rows[f"{style}|context"] = item_scorer.score({c: None for c in item_scorer.concepts}, contexts)
            if compose and (style, "linker") in conditions:
                item_rows[f"{style}|context+linker"] = item_scorer.score(conditions[(style, "linker")], contexts)
            if compose and (style, "oracle") in conditions and style == primary_style:
                item_rows[f"{style}|context+oracle"] = item_scorer.score(conditions[(style, "oracle")], contexts)
        document["context_tokens"] = {s: float(np.mean([len(run.tokenizer(d["text"], add_special_tokens=False)["input_ids"]) + 1
                                                        for d in read_set.definitions if d["style"] == s])) for s in context_styles}
        timings["context"] = time.monotonic() - t0
    gradient_records: dict[str, Any] = {}
    if item_scorer is not None and "gradient" in methods:
        if gradient_lr is None:
            raise ValueError("the gradient method needs --gradient-lr or --gradient-lr-from")
        t0 = time.monotonic()
        result = evaluate_gradient(run, read_set, ctx, item_scorer, style=primary_style, lr=gradient_lr, factors=gradient_factors,
                                   optimizer=gradient_optimizer, backup=gradient_backup, log=log)
        for factor, block in result["factors"].items():
            item_rows[f"{primary_style}|gradient×{factor}"] = block["rows"]
            gradient_records[factor] = block["records"]
        document["gradient"] = {"lr": gradient_lr, "optimizer": gradient_optimizer, "general_before": result["general_before"],
                                **{f"×{f}": {"steps_mean": float(np.mean([r["steps"] for r in b["records"]])),
                                             "train_tokens_mean": float(np.mean([r["train_tokens"] for r in b["records"]])),
                                             "general_delta_mean": float(np.mean([r["general_delta"] for r in b["records"]])),
                                             "general_delta_max": float(np.max([r["general_delta"] for r in b["records"]])),
                                             "seconds": b["seconds"]} for f, b in result["factors"].items()}}
        timings["gradient"] = time.monotonic() - t0
    document["items"] = summarize_items(item_rows, read_set.kind, primary_style, resamples=resamples, seed=seed)
    windows_out: dict[str, Any] = {}
    if "windows" in methods and read_set.kind in {"heldout", "swap"}:
        t0 = time.monotonic()
        entry_of = link_entry_of(read_set, run)
        headwords = {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts}
        terms = [entry_of[c["concept"]] for c in read_set.concepts]
        corpus_path = Path(run.config["data"]["eval"])
        wset = build_window_set(run, terms, headwords, max_windows=max_windows, corpus_path=corpus_path)
        log(f"  windows: {len(wset.starts)} windows, {sum(len(s) for s in wset.spans)} occurrences ({wset.excluded_own} in own documents)")
        per_condition: dict[str, dict[str, np.ndarray]] = {}
        window_conditions = [(primary_style, r) for r in readers if (primary_style, r) in conditions]
        for style, reader in window_conditions:
            if reader != "none" and not compose:
                continue
            concept_frames = conditions[(style, reader)]
            with swapped_frames(run.channel, {entry_of[c]: concept_frames.get(c) for c in concept_frames}) if compose else contextlib.nullcontext():
                per_condition[f"{style}|{reader}"] = window_term_losses(run, corpus_path, wset)
        if window_gradient and "gradient" in methods and gradient_lr is not None and read_set.kind == "heldout":
            per_condition[f"{primary_style}|gradient×1.0"] = window_gradient_losses(run, read_set, ctx, wset, corpus_path, style=primary_style,
                                                                              lr=gradient_lr, optimizer=gradient_optimizer,
                                                                              backup=gradient_backup)
        windows_out = {"windows": len(wset.starts), "occurrences": sum(len(s) for s in wset.spans), "own_occurrences": wset.excluded_own,
                       "terms_with_occurrences": int(sum(1 for t in range(len(terms)) if any(x[2] == t for s in wset.spans for x in s))),
                       "conditions": per_condition}
        document["windows"] = summarize_windows(per_condition, primary_style, resamples=resamples, seed=seed)
        document["windows"].update({k: v for k, v in windows_out.items() if k != "conditions"})
        timings["windows"] = time.monotonic() - t0
    if "locality" in methods and compose:
        key = (primary_style, "linker") if (primary_style, "linker") in conditions else (primary_style, "oracle")
        if key in conditions:
            document["locality"] = row_locality(run, read_set, conditions[key])
    document["timings"] = timings
    document["seconds"] = time.monotonic() - started
    return {"document": document, "frames": frames, "item_rows": item_rows, "windows": windows_out, "gradient": gradient_records}


def window_gradient_losses(run: E5Run, read_set: ReadSet, ctx: TrackContext, wset: WindowSet, corpus_path: Path, *, style: str, lr: float,
                    optimizer: str, backup: str) -> dict[str, np.ndarray]:
    """Per term: compute-matched host updates on its definition, then the after-term loss of its own occurrences only."""
    tasks = {t.concept: t for t in read_tasks(read_set, ctx, [style])}
    entry_of = link_entry_of(read_set, run)
    total: dict[str, np.ndarray] | None = None
    with host_snapshot(run.model, backup=backup) as restore, swapped_frames(run.channel, {entry_of[c]: None for c in tasks}):
        for t_index, concept in enumerate([c["concept"] for c in read_set.concepts]):
            task = tasks.get(concept)
            if task is None or not any(x[2] == t_index for s in wset.spans for x in s):
                continue
            k = matched_steps(len(rtl.linker_candidates(task, rtl.mentions_of(task, ctx.fillers), ctx.typing)))
            gradient_steps(run, run.adapter, task.text, entry_of[concept], steps=k, lr=lr, optimizer=optimizer)
            part = window_term_losses(run, corpus_path, wset, select=lambda w, t, i=t_index: t == i)
            if total is None:
                total = {k2: np.zeros_like(v) for k2, v in part.items()}
            for k2 in ("sum_other", "count_other", "sum_own", "count_own"):
                total[k2] += part[k2]
            restore()
    if total is None:
        n = len(wset.terms)
        total = {"sum_other": np.zeros(n), "count_other": np.zeros(n), "sum_own": np.zeros(n), "count_own": np.zeros(n),
                 "unlinked": np.zeros((len(wset.starts), 2))}
    total["unlinked"] = np.full((len(wset.starts), 2), np.nan)
    return total


def summarize_items(item_rows: dict[str, dict[str, list[dict[str, Any]]]], kind: str, primary_style: str, *,
                    resamples: int, seed: int) -> dict[str, Any]:
    """Per condition and test: the mean; and the pre-registered within-run contrasts (paired over items)."""
    values = {cond: test_values(rows, kind) for cond, rows in item_rows.items()}
    means = {cond: {t: (float(np.mean(list(v.values()))) if v else None, len(v)) for t, v in by_test.items()}
             for cond, by_test in values.items()}
    contrasts = []
    tests = NEW_TESTS if kind == "new" else HELDOUT_TESTS
    styles = sorted({c.split("|")[0] for c in values})
    for style in styles:
        pairs = [("linker", "none"), ("oracle", "none"), ("linker", "oracle"), ("linker", "typeprior"), ("linker", "random"),
                 ("linker", "pattern"), ("linker", "host"), ("linker", "linker-all"), ("teacher", "none"), ("stated", "none"),
                 ("linker-joint", "none"), ("linker-joint", "typeprior"), ("linker-joint", "oracle"), ("linker-joint", "linker"),
                 ("context", "linker"), ("context+linker", "context"), ("context", "none")]
        pairs += [(f"gradient×{f}", "linker") for f in ("1.0", "4.0")] + [(f"gradient×{f}", "none") for f in ("1.0", "4.0")]
        for a, b in pairs:
            ka, kb = f"{style}|{a}", f"{style}|{b}"
            if ka not in values or kb not in values:
                continue
            block = []
            for test in tests:
                result = compare(values[ka].get(test, {}), values[kb].get(test, {}), resamples=resamples, seed=seed)
                if result is not None:
                    block.append({"style": style, "a": a, "b": b, "test": test, **result})
            for row, p in zip(block, holm_adjust([r["p_value"] for r in block]) if block else []):
                row["p_holm"] = p
            contrasts += block
    return {"means": means, "contrasts": contrasts, "primary_style": primary_style}


def summarize_windows(per_condition: dict[str, dict[str, np.ndarray]], primary_style: str, *, resamples: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"loss": {}, "contrasts": []}
    for cond, data in per_condition.items():
        for part in ("other", "own"):
            n = data[f"count_{part}"].sum()
            out["loss"].setdefault(cond, {})[part] = {"loss": float(data[f"sum_{part}"].sum() / n) if n else None, "targets": int(n)}
        un = data["unlinked"]
        if np.isfinite(un).all() and un[:, 1].sum():
            out["loss"][cond]["unlinked"] = float(un[:, 0].sum() / un[:, 1].sum())
    ref = per_condition.get(f"{primary_style}|none")
    if ref is not None:
        for cond, data in per_condition.items():
            if cond.endswith("|none"):
                continue
            for part in ("other", "own"):
                result = relative_change(data, ref, part=part, resamples=resamples, seed=seed)
                if result is not None:
                    out["contrasts"].append({"condition": cond, "reference": "none", "part": part, **result})
        oracle = per_condition.get(f"{primary_style}|oracle")
        linker = per_condition.get(f"{primary_style}|linker")
        if oracle is not None and linker is not None:
            gain_oracle = (oracle["sum_other"] - ref["sum_other"]).sum()
            gain_linker = (linker["sum_other"] - ref["sum_other"]).sum()
            out["recovered_share"] = float(gain_linker / gain_oracle) if gain_oracle else None
    return out


# -- report of one evaluation ----------------------------------------------------------------------------------------------------

def render(header: dict[str, Any], document: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E11 read-to-learn — {document['set']['track']} {document['set']['kind']} — {source['condition']} seed {source['seed']} "
             f"({source['size']})", "",
             f"Item set `{document['set']['path']}` ({document['set']['concepts']} terms, {document['set']['items']} items); "
             f"styles {document['styles']} (primary `{document['primary_style']}`); channel composes: {document['compose']}. "
             "No weight is updated except inside the gradient baseline, whose weights are restored after every term.", ""]
    if document.get("frames"):
        lines += ["## Frames written by each reader", "",
                  "| style | reader | frames | empty | edges | precision | recall | F1 | stated recall | filler recall | relation acc. | "
                  "relation acc. (ambiguous fillers) | cost / word |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for style, by_reader in document["frames"].items():
            for reader, m in by_reader.items():
                cost = ", ".join(f"{k} {v:.1f}" for k, v in m["cost_per_word"].items() if v)
                lines.append(f"| {style} | {reader} | {m['frames']} | {m['empty']} | {m['edges_mean']:.2f} | {fmt(m['precision'], 3)} | "
                             f"{fmt(m['recall'], 3)} | {fmt(m['f1'], 3)} | {fmt(m['stated_recall'], 3)} | {fmt(m['filler_recall'], 3)} | "
                             f"{fmt(m['relation_accuracy'], 3)} | {fmt(m.get('ambiguous_relation_accuracy'), 3)} "
                             f"({m.get('ambiguous_edges', 0)}) | {cost or '—'} |")
        lines.append("")
    items = document.get("items") or {}
    if items.get("means"):
        tests = NEW_TESTS if document["set"]["kind"] == "new" else HELDOUT_TESTS
        lines += ["## Item tests (no definition in context unless the condition says `context`)", "",
                  "| condition | " + " | ".join(tests) + " |", "|---|" + "---:|" * len(tests)]
        for cond, by_test in sorted(items["means"].items()):
            lines.append(f"| {cond} | " + " | ".join(fmt(by_test.get(t, (None, 0))[0], 4) for t in tests) + " |")
        if items.get("contrasts"):
            lines += ["", "| style | a − b | test | difference [95% CI] | n | p (Holm within pair) |", "|---|---|---|---|---:|---:|"]
            for c in items["contrasts"]:
                lines.append(f"| {c['style']} | {c['a']} − {c['b']} | {c['test']} | {c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                             f"{c['n']} | {fmt(c.get('p_holm'), 4)} |")
        lines.append("")
    windows = document.get("windows")
    if windows:
        lines += ["## Loss after the read terms in the evaluation text", "",
                  f"{windows.get('windows')} windows, {windows.get('occurrences')} occurrences ({windows.get('own_occurrences')} inside the "
                  "term's own definition document, reported separately).", "",
                  "| condition | loss (other documents) | targets | loss (own document) | unlinked-token loss |", "|---|---:|---:|---:|---:|"]
        for cond, parts in windows["loss"].items():
            lines.append(f"| {cond} | {fmt(parts['other']['loss'], 4)} | {parts['other']['targets']} | {fmt(parts['own']['loss'], 4)} | "
                         f"{fmt(parts.get('unlinked'), 4)} |")
        if windows.get("contrasts"):
            lines += ["", "| condition − none | part | relative change [95% CI] | Δ nats/token | p |", "|---|---|---|---:|---:|"]
            for c in windows["contrasts"]:
                lines.append(f"| {c['condition']} | {c['part']} | {100 * c['relative']:+.2f}% [{100 * c['relative_ci_low']:+.2f}, "
                             f"{100 * c['relative_ci_high']:+.2f}] | {c['mean']:+.4f} | {fmt(c['p_value'], 4)} |")
        if windows.get("recovered_share") is not None:
            lines.append(f"\nShare of the oracle frame's loss gain recovered by the linker reader: {windows['recovered_share']:.3f}.")
        lines.append("")
    if document.get("gradient"):
        g = document["gradient"]
        lines += ["## Gradient baseline", "", f"lr {g['lr']}, {g['optimizer']}; general-text loss before {fmt(g['general_before'], 4)}.", ""]
        for key, block in g.items():
            if key.startswith("×"):
                lines.append(f"- {key}: {block['steps_mean']:.1f} steps / term ({block['train_tokens_mean']:.0f} training tokens), general-text "
                             f"Δ loss mean {block['general_delta_mean']:+.4f} (max {block['general_delta_max']:+.4f}), {block['seconds']:.0f}s")
        lines.append("")
    if document.get("locality"):
        lines += ["## Locality", "", f"`{json.dumps(document['locality'])}`", ""]
    lines += [f"Timings (s): `{json.dumps({k: round(v, 1) for k, v in document.get('timings', {}).items()})}`; total {document['seconds']:.0f}s."]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    read_set = load_read_set(args.items).limit(args.limit)
    track = read_set.manifest["track"]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()] if args.styles else read_set.styles
    primary = args.primary_style or (PRIMARY_STYLE.get(track) if PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    readers = [r.strip() for r in args.readers.split(",") if r.strip()]
    lr = args.gradient_lr
    if args.gradient_lr_from:
        lr = float(json.loads(Path(args.gradient_lr_from).read_text())["lr"])
    config = {"experiment": "e11-read-to-learn", "run": str(args.run), "items": str(args.items), "limit": args.limit,
              "methods": methods, "readers": readers, "styles": styles, "primary_style": primary, "gradient_lr": lr,
              "gradient_factors": args.gradient_factors, "gradient_optimizer": args.gradient_optimizer,
              "gradient_backup": args.gradient_backup, "max_windows": args.max_windows, "window_gradient": args.window_gradient,
              "seed": args.seed,
              "resamples": args.resamples, "alias_table": str(args.alias_table) if args.alias_table else None,
              "teacher_frames": str(args.teacher_frames) if args.teacher_frames else None, "smoke": bool(args.smoke)}
    if args.overwrite:
        clear_output(args.output)
        for extra in ("frames.jsonl", "windows.npz", "gradient_records.json"):
            (Path(args.output) / extra).unlink(missing_ok=True)
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    teacher = args.teacher_frames or (read_set.path / "teacher-frames.jsonl" if (read_set.path / "teacher-frames.jsonl").exists() else None)
    result = evaluate(run, read_set, methods=methods, readers=readers, styles=styles, primary_style=primary, gradient_lr=lr,
                      gradient_factors=args.gradient_factors, gradient_optimizer=args.gradient_optimizer,
                      gradient_backup=args.gradient_backup, teacher_frames=teacher, max_windows=args.max_windows,
                      window_gradient=args.window_gradient, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "smoke": bool(args.smoke)}
    document = {**header, **result["document"]}
    _write_jsonl(args.output / "frames.jsonl", result["frames"])
    with (args.output / "predictions.jsonl").open("w") as handle:
        for cond, by_concept in result["item_rows"].items():
            for concept, rows in by_concept.items():
                for row in rows:
                    handle.write(json.dumps(json_ready({"condition": cond, **row})) + "\n")
    if result["windows"].get("conditions"):
        np.savez_compressed(args.output / "windows.npz", **{f"{cond}::{k}": v for cond, data in result["windows"]["conditions"].items()
                                                            for k, v in data.items()})
    if result["gradient"]:
        write_json(args.output / "gradient_records.json", result["gradient"])
    write_json(args.output / "summary.json", document)
    (args.output / "report.md").write_text(("**SMOKE TEST — not a result.**\n\n" if args.smoke else "") + render(header, document))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


# -- learning-rate selection for the gradient baseline (dev words) ----------------------------------------------------------------

def run_gradient_dev(args: argparse.Namespace) -> dict[str, Any]:
    """lr ∈ `--lrs` on the dev words (primary style, ×1 compute): the lr with the best mean property accuracy is written
    to `OUTPUT/gradient_lr.json` (ties → the smaller lr)."""
    read_set = load_read_set(args.items)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    ctx = run_context(run)
    primary = args.primary_style or PRIMARY_STYLE.get(read_set.manifest["track"], read_set.styles[0])
    scorer = ItemScorer(run, read_set, ctx)
    scores = {}
    base = test_values(scorer.score({c: None for c in scorer.concepts}), read_set.kind)
    for lr in args.lrs:
        result = evaluate_gradient(run, read_set, ctx, scorer, style=primary, lr=lr, factors=(1.0,), optimizer=args.gradient_optimizer,
                                   backup=args.gradient_backup, locality_windows=0)
        values = test_values(result["factors"]["1.0"]["rows"], read_set.kind)
        scores[str(lr)] = {t: float(np.mean(list(v.values()))) for t, v in values.items() if v}
    best = min(args.lrs, key=lambda lr: (-scores[str(lr)].get("property", 0.0), lr))
    out = {"lr": best, "scores": scores, "no_update": {t: float(np.mean(list(v.values()))) for t, v in base.items() if v},
           "run": str(args.run), "items": str(args.items), "style": primary, "optimizer": args.gradient_optimizer,
           "rule": "highest mean property accuracy at ×1 compute on the dev words; ties → smaller lr"}
    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output) / "gradient_lr.json", out)
    return out


# -- teacher frames (optional; claude -p) -------------------------------------------------------------------------------------

RESTRICTED = ("NC", "MIMIC", "SNOMED", "UMLS", "credentialed", "DUA")


def teacher_allowed(read_set: ReadSet) -> bool:
    """An external reader (Claude) may read a set only if every licence is open (CC BY, public domain, project-generated):
    decision 58 forbids sending credentialed text out; non-commercial text is kept local too."""
    licences = list(read_set.manifest.get("licence") or []) + [d.get("licence", "") for d in read_set.definitions]
    return bool(licences) and not any(word in str(lic) for lic in licences for word in RESTRICTED)


def run_teacher(args: argparse.Namespace) -> dict[str, Any]:
    """Claude teacher frames for every definition of an item set (open-licence text only), cached per prompt; writes
    `ITEMS/teacher-frames.jsonl`. `--dry-run` prints the number of calls and the cost estimate only."""
    from ..authoring_baselines import TeacherAuthor
    read_set = load_read_set(args.items)
    if not teacher_allowed(read_set):
        raise ValueError("teacher reading is restricted to open-licence text (CC BY / public domain / project-generated); "
                         "credentialed (MIMIC, SNOMED CT, UMLS) and non-commercial text is read by local readers only")
    ctx = track_context(read_set.manifest["track"])
    tasks = read_tasks(read_set, ctx, read_set.styles)
    calls = math.ceil(len(tasks) / args.batch)
    estimate = {"definitions": len(tasks), "calls": calls, "usd_low": 0.03 * calls, "usd_high": 0.10 * calls}
    if args.dry_run:
        return estimate
    descriptions = {r: r.replace("_", " ") for r in ctx.typing.relation_names}
    teacher = TeacherAuthor(Path(args.cache), ctx.typing.relation_names, descriptions,
                            filler_hint="a concept named in the entry text, copied exactly as it is written there.", model=args.model)
    results = rtl.read_teacher(tasks, ctx.fillers, ctx.typing, teacher, batch=args.batch)
    _write_jsonl(read_set.path / "teacher-frames.jsonl", [{"concept": r.concept, "style": r.style, "frame": ctx.names(r.frame),
                                                            "cost": r.cost, "details": r.details} for r in results])
    return {**estimate, "spent_usd": teacher.spent_usd, "calls_made": teacher.calls, "cached": teacher.cached}


# -- cross-run report (pre-registered endpoints) ------------------------------------------------------------------------------

def _load_predictions(folder: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows_by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_jsonl(Path(folder) / "predictions.jsonl"):
        rows_by[row["condition"]][row["concept"]].append(row)
    summary = json.loads((Path(folder) / "summary.json").read_text())
    return {cond: test_values(rows, summary["set"]["kind"]) for cond, rows in rows_by.items()}


def _load_windows(folder: Path) -> dict[str, dict[str, np.ndarray]]:
    path = Path(folder) / "windows.npz"
    if not path.exists():
        return {}
    out: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    with np.load(path) as data:
        for key in data.files:
            cond, name = key.split("::")
            out[cond][name] = data[key]
    return out


def pooled_item_contrast(folders: Sequence[Path], a: str, b: str, test: str, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """a − b on one test, paired over items, item differences of every seed pooled (the R9 convention)."""
    diffs = []
    for folder in folders:
        values = _load_predictions(folder)
        va, vb = values.get(a, {}).get(test, {}), values.get(b, {}).get(test, {})
        diffs += [va[i] - vb[i] for i in sorted(set(va) & set(vb))]
    if not diffs:
        return None
    return {**zs.paired_difference(np.asarray(diffs), np.zeros(len(diffs)), resamples=resamples, seed=seed), "seeds": len(folders)}


def pooled_window_contrast(folders: Sequence[Path], a: str, b: str, *, part: str = "other", resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """a − b relative loss after the read terms; clusters = terms, every seed's sums added per term."""
    d = n = base = None
    for folder in folders:
        data = _load_windows(folder)
        if a not in data or b not in data:
            continue
        dd = data[a][f"sum_{part}"] - data[b][f"sum_{part}"]
        nn, bb = data[b][f"count_{part}"], data[b][f"sum_{part}"]
        d, n, base = (dd, nn, bb) if d is None else (d + dd, n + nn, base + bb)
    if d is None or n.sum() <= 0:
        return None
    return {**paired_ratio_bootstrap(d, n, base, resamples=resamples, seed=seed), "seeds": len(folders)}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """Primary endpoints P1 (T5 new words, prose, property, linker − none) and P2 (T4 held-out ChEBI terms, loss after the
    term in other documents, linker − none), Holm over both; the key secondaries per set, seeds pooled."""
    groups: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    missing = [str(f) for f in args.runs if not (Path(f) / "summary.json").exists()]
    for folder in args.runs:
        if str(folder) in missing:
            continue
        summary = json.loads((Path(folder) / "summary.json").read_text())
        source = summary["source"]
        groups[(summary["set"]["track"], summary["set"]["kind"], source["size"], source["condition"])].append(Path(folder))
    endpoints, secondary = [], []
    for (track, kind, size, condition), folders in sorted(groups.items()):
        primary = PRIMARY_STYLE.get(track, "prose")
        if condition.startswith("C5"):
            if kind == "new":
                for test in ("property", "entailment", "paraphrase", "statement_accuracy"):
                    for a, b in (("linker", "none"), ("oracle", "none"), ("linker", "typeprior"), ("linker", "random"),
                                 ("linker-joint", "none"), ("linker-joint", "typeprior"),
                                 ("linker", "oracle"), ("context", "linker"), ("context+linker", "context"), ("gradient×1.0", "linker")):
                        result = pooled_item_contrast(folders, f"{primary}|{a}", f"{primary}|{b}", test, resamples=args.resamples)
                        if result:
                            row = {"track": track, "kind": kind, "size": size, "model": condition, "a": a, "b": b, "test": test, **result}
                            (endpoints if (track == "t5" and test == "property" and (a, b) == ("linker", "none") and "360M" in size)
                             else secondary).append(row)
            if kind == "heldout":
                for test in HELDOUT_TESTS:
                    for a, b in (("linker", "none"), ("oracle", "none"), ("linker", "typeprior"), ("context", "linker"),
                                 ("linker-joint", "none"), ("linker-joint", "typeprior")):
                        result = pooled_item_contrast(folders, f"{primary}|{a}", f"{primary}|{b}", test, resamples=args.resamples)
                        if result:
                            secondary.append({"track": track, "kind": kind, "size": size, "model": condition, "a": a, "b": b,
                                              "test": test, **result})
                for a, b in (("linker", "none"), ("oracle", "none"), ("linker", "typeprior"), ("linker", "random"), ("gradient×1.0", "none"),
                             ("linker-joint", "none"), ("linker-joint", "typeprior")):
                    result = pooled_window_contrast(folders, f"{primary}|{a}", f"{primary}|{b}", resamples=args.resamples)
                    if result:
                        row = {"track": track, "kind": kind, "size": size, "model": condition, "a": a, "b": b, "test": "loss after term", **result}
                        (endpoints if (track == "t4" and (a, b) == ("linker", "none") and "360M" in size) else secondary).append(row)
    for row, p in zip(endpoints, holm_adjust([r["p_value"] for r in endpoints]) if endpoints else []):
        row["p_holm"] = p
    out = {"endpoints": endpoints, "secondary": secondary, "runs": [str(f) for f in args.runs], "missing": missing}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", out)
    lines = ["# E11 read-to-learn — pooled report", "", "Primary endpoints (Holm over both):", "",
             "| set | host | model | contrast | test | difference [95% CI] | seeds | p (Holm) |", "|---|---|---|---|---|---|---:|---:|"]
    for r in endpoints:
        value = (f"{100 * r['relative']:+.2f}% [{100 * r['relative_ci_low']:+.2f}, {100 * r['relative_ci_high']:+.2f}]" if "relative" in r
                 else f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        lines.append(f"| {r['track']} {r['kind']} | {r['size']} | {r['model']} | {r['a']} − {r['b']} | {r['test']} | {value} | {r['seeds']} | "
                     f"{fmt(r.get('p_holm'), 4)} |")
    lines += ["", "Secondary contrasts (unadjusted p):", "", "| set | host | model | contrast | test | difference [95% CI] | seeds | p |",
              "|---|---|---|---|---|---|---:|---:|"]
    for r in secondary:
        value = (f"{100 * r['relative']:+.2f}% [{100 * r['relative_ci_low']:+.2f}, {100 * r['relative_ci_high']:+.2f}]" if "relative" in r
                 else f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        lines.append(f"| {r['track']} {r['kind']} | {r['size']} | {r['model']} | {r['a']} − {r['b']} | {r['test']} | {value} | {r['seeds']} | "
                     f"{fmt(r.get('p_value'), 4)} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return out


# -- plan (queue commands; printed, never submitted here) ------------------------------------------------------------------------

E9_RUNS = Path("experiments/e9-retrofit/runs")
PLAN_PRIORITY = 62
# Per-concept seconds at SmolLM2-360M, measured in the GPU smoke test (preregistration §12; 2026-10-07) while a training job
# held the GPU at 100%: model readers per definition, one persistence condition, one in-context condition, the gradient
# baseline (×1, ×4 incl. scoring and restore), and the per-term gradient on windows (T4-H); `windows` = windows per run.
# `CONTENTION` converts them to an idle GPU (the queue runs one job at a time); `SCALE_135M` for the smaller host.
SMOKE_SECONDS = {
    "t5-new": {"reader": 1.3, "persist": 0.55, "context": 2.3, "grad1": 2.4, "grad4": 4.6},
    "t5-heldout": {"reader": 1.3, "persist": 0.28, "context": 1.15, "grad1": 1.6, "grad4": 3.8, "windows": 2000},
    "t4-heldout": {"reader": 1.5, "persist": 0.26, "context": 0.65, "grad1": 2.0, "grad4": 7.0, "windows": 1000, "wgrad": 2.5},
    "t4-new": {"reader": 2.5, "persist": 0.3, "context": 1.2, "grad1": 3.0, "grad4": 9.0},
    "t1-heldout": {"reader": 1.3, "persist": 0.0, "context": 0.0, "grad1": 0.0, "grad4": 0.0, "windows": 2048},
}
WINDOW_SECONDS = 0.06                 # one 1,024-token window under one condition (smoke)
CONTENTION, SCALE_135M = 2.0, 0.55
SETS = {("t5", "new"): ("t5-new-smollm2-v1", 300, ["glossary", "dictionary", "prose"]),
        ("t5", "heldout"): ("t5-heldout-smollm2-v1", 560, ["prose"]),
        ("t4", "heldout"): ("t4-heldout-smollm2-v1", 341, ["chebi"]),
        ("t4", "new"): ("t4-new-smollm2-v1", 300, ["chebi"]),
        ("t1", "heldout"): ("t1-heldout-smollm2-v1", 1972, ["scope"])}
C5_READERS = "oracle,stated,typeprior,pattern,linker,linker-all,linker-joint,host,teacher,random,none"
READER_CONDITIONS = 10                # persistence / window conditions of a C5 run (teacher only if its frames exist)


def job_hours(track: str, kind: str, model: str, host: str, styles: Sequence[str]) -> float:
    """GPU hours of one evaluation job on an idle GPU (`SMOKE_SECONDS` / `CONTENTION`)."""
    _, concepts, _ = SETS[(track, kind)]
    c = SMOKE_SECONDS[f"{track}-{kind}"]
    n_styles = len(styles)
    windows = c.get("windows", 0) * WINDOW_SECONDS
    if model == "C5":
        persist_conditions = READER_CONDITIONS * (1 + 0.6 * (n_styles - 1) if kind == "new" else n_styles)
        seconds = concepts * (c["reader"] * n_styles + c["persist"] * persist_conditions + c["context"] * (2 * n_styles + 1)
                              + c["grad1"] + c["grad4"] + c.get("wgrad", 0.0)) + windows * READER_CONDITIONS
    elif model == "C0p":
        seconds = concepts * (c["persist"] + c["context"] + c["grad1"] + c["grad4"] + c.get("wgrad", 0.0)) + windows
    else:
        seconds = concepts * (c["persist"] + c["context"]) + windows
    return seconds / CONTENTION / 3600 * (1.0 if "360M" in host else SCALE_135M)


def tier_of(track: str, kind: str, host: str, model: str, seed: int) -> int:
    """1: the primary endpoints and their key comparators (T5-N 360M C5/C0′ seeds 1–3; T4-H 360M C5/C0′, seeds 2–3 once
    trained); 2: replications and controls; 3: T5-H and T4-N on 135M. Priority = PLAN_PRIORITY + tier."""
    if "360M" in host and model in {"C5", "C0p"} and (track, kind) in {("t5", "new"), ("t4", "heldout")}:
        return 1
    if (track, kind) in {("t5", "heldout"), ("t4", "new")} and "135M" in host:
        return 3
    return 2


def plan_jobs(*, tracks: Sequence[str] = ("t5", "t4", "t1"), hosts: Sequence[str] = ("SmolLM2-360M", "SmolLM2-135M"),
              seeds: Sequence[int] = (1, 2, 3), python: str = "$PY", priority: int = PLAN_PRIORITY,
              runs_root: Path = E9_RUNS) -> list[dict[str, Any]]:
    """The E11 evaluation jobs (nothing is submitted): the gradient lr choice on the dev words first (`priority`), then
    one job per (set, run) at `priority + tier` (`tier_of`), the pooled report last (`priority + 4`).

    Scope (preregistration §§5, 8, 13.5): C5 runs every reader and method; C0′ the text routes (no frame, in context,
    gradient); C2 and P0 no frame and in context only (the gradient comparator is C0′). T5-N reads all three styles; the
    secondary sets read their primary style. T4 runs past seed 1 wait for their training (`after_training`). T1-H
    (negative control) runs SmolLM2-360M seed 1 (frames and loss after the term; no items exist), windows capped at
    2,048."""
    jobs: list[dict[str, Any]] = []
    for host in hosts:
        jobs.append({"name": f"e11-gradient-dev-{host}", "priority": priority, "min_free_gb": 10, "tier": 0,
                     "hours": round(40 * 3 * 2.4 / CONTENTION / 3600 * (1.0 if "360M" in host else SCALE_135M), 3),
                     "command": [python, "-m", "vsa_embed.experiments.e11_read_to_learn", "gradient-dev", "--run",
                                 str(runs_root / "t5" / f"{host}-full-C0p-s1"), "--items", str(ITEMS / "t5-dev-smollm2-v1"),
                                 "--output", str(ROOT / "dev" / host)]})
    for (track, kind), (folder, _, styles) in SETS.items():
        if track not in tracks:
            continue
        for host in hosts:
            if track == "t1" and "360M" not in host:
                continue
            for model in ("C5", "C0p", "C2", "P0"):
                if track == "t1" and model not in {"C5", "C0p"}:
                    continue
                run_seeds = (1,) if model == "P0" or track == "t1" else tuple(seeds)
                for seed in run_seeds:
                    mode = "frozen" if model == "P0" else "full"
                    run = runs_root / track / f"{host}-{mode}-{model}-s{seed}"
                    if model == "C5":
                        methods, readers = "frames,persistence,context,gradient,windows,locality", C5_READERS
                    elif model == "C0p":
                        methods, readers = "persistence,context,gradient,windows", "none"
                    else:
                        methods, readers = "persistence,context,windows", "none"
                    if kind == "new":
                        methods = methods.replace(",windows", "")
                    if track == "t1":
                        methods = ",".join(m for m in methods.split(",") if m in {"frames", "windows", "locality"})
                    command = [python, "-m", "vsa_embed.experiments.e11_read_to_learn", "evaluate", "--run", str(run),
                               "--items", str(ITEMS / folder), "--methods", methods, "--readers", readers,
                               "--styles", ",".join(styles), "--alias-table",
                               str(Path("~/data/vsa-llm/e9/alias-tables").expanduser() / f"{track}.json")]
                    if "gradient" in methods:
                        command += ["--gradient-lr-from", str(ROOT / "dev" / host / "gradient_lr.json")]
                    if (track, kind) == ("t4", "heldout") and "gradient" in methods:
                        command.append("--window-gradient")
                    if track == "t1":
                        command += ["--max-windows", "2048"]
                    command += ["--output", str(run / f"{FOLDER}-{track}-{kind}")]
                    tier = tier_of(track, kind, host, model, seed)
                    hours = job_hours(track, kind, model, host, styles) if track != "t1" or model == "C5" else \
                        2048 * WINDOW_SECONDS / CONTENTION / 3600
                    jobs.append({"name": f"e11-{track}-{kind}-{host}-{model}-s{seed}", "priority": priority + tier, "tier": tier,
                                 "min_free_gb": 10, "after_training": track == "t4" and seed > 1, "hours": round(hours, 3),
                                 "command": command})
    outputs = [j["command"][-1] for j in jobs if j["name"].startswith("e11-t")]
    jobs.append({"name": "e11-report", "priority": priority + 4, "min_free_gb": 1, "hours": 0.0, "tier": 4,
                 "command": [python, "-m", "vsa_embed.experiments.e11_read_to_learn", "report", "--runs", *outputs,
                             "--output", str(ROOT / "report")]})
    return jobs


def run_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = plan_jobs(tracks=args.tracks, hosts=args.hosts, seeds=args.seeds, priority=args.priority)
    totals: dict[str, float] = defaultdict(float)
    for job in jobs:
        pending = job.get("after_training")
        if pending and not args.include_pending:
            totals["pending: T4 seeds 2-3 (not printed; --include-pending)"] += job["hours"]
            continue
        totals[f"tier {job['tier']}" + (" (after T4 seeds 2-3 train)" if pending else "")] += job["hours"]
        command = " ".join(job["command"])
        print(f"PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name {job['name']} --priority {job['priority']} "
              f"--min-free-gb {job['min_free_gb']} --no-resume -- {command}   # ≈ {job['hours']:.2f} GPU-h")
    for key, hours in sorted(totals.items()):
        print(f"# {key}: ≈ {hours:.1f} GPU-h")
    print(f"# total printed ≈ {sum(h for k, h in totals.items() if not k.startswith('pending')):.1f} GPU-h (idle-GPU estimate)")
    return jobs


# -- CLI ----------------------------------------------------------------------------------------------------------------------

def run_items(args: argparse.Namespace) -> dict[str, Any]:
    from . import e9_tracks as tracks
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")
    ctx = track_context(args.track)
    if args.kind == "new":
        return build_new_set(args.new_items, args.output, ctx, styles=args.styles, seed=args.seed)
    if args.kind == "heldout":
        from transformers import AutoTokenizer
        spec = tracks.track_spec(args.track)
        tokenizer = AutoTokenizer.from_pretrained(tracks.FAMILY_TOKENIZERS["smollm2"], local_files_only=True)
        return build_heldout_set(args.track, args.output, ctx, styles=args.styles, seed=args.seed, limit=args.limit,
                                 wpc7_dir=spec.zeroshot_items, eval_corpus=spec.eval_corpus,
                                 min_subtokens=int(ctx.ontology.get("min_subtokens", 2)), tokenizer=tokenizer)
    return build_swap_set(args.glossary, args.output, ctx)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    items = sub.add_parser("items", help="build a read-to-learn item set")
    items.add_argument("--kind", required=True, choices=["new", "heldout", "swap"])
    items.add_argument("--track", required=True, choices=["t5", "t4", "t1", "t7", "t7rood"])
    items.add_argument("--output", type=Path, required=True); items.add_argument("--new-items", type=Path, default=None)
    items.add_argument("--glossary", type=Path, default=None); items.add_argument("--styles", nargs="*", default=None)
    items.add_argument("--seed", type=int, default=0); items.add_argument("--limit", type=int, default=None)
    fetch = sub.add_parser("fetch-openstax", help="download the OpenStax glossary (pinned commit)")
    fetch.add_argument("--output", type=Path, required=True); fetch.add_argument("--collection", default="chemistry-2e")
    teacher = sub.add_parser("teacher", help="Claude teacher frames for an item set (optional)")
    teacher.add_argument("--items", type=Path, required=True); teacher.add_argument("--dry-run", action="store_true")
    teacher.add_argument("--batch", type=int, default=8); teacher.add_argument("--model", default="claude-opus-5-5")
    teacher.add_argument("--cache", type=Path, default=ROOT / "teacher-cache")
    for name in ("evaluate", "gradient-dev"):
        ev = sub.add_parser(name)
        ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
        ev.add_argument("--output", type=Path, required=True); ev.add_argument("--checkpoint", default="final.pt")
        ev.add_argument("--alias-table", type=Path, default=None); ev.add_argument("--batch-size", type=int, default=32)
        ev.add_argument("--device", default=None); ev.add_argument("--primary-style", default=None)
        ev.add_argument("--gradient-optimizer", choices=["adam", "sgd"], default="adam")
        ev.add_argument("--gradient-backup", choices=["gpu", "cpu"], default="gpu")
        if name == "evaluate":
            ev.add_argument("--methods", default=",".join(METHODS)); ev.add_argument("--readers", default=",".join(rtl.READERS))
            ev.add_argument("--styles", default=""); ev.add_argument("--limit", type=int, default=None)
            ev.add_argument("--gradient-lr", type=float, default=None); ev.add_argument("--gradient-lr-from", type=Path, default=None)
            ev.add_argument("--gradient-factors", type=float, nargs="*", default=[1.0, 4.0])
            ev.add_argument("--teacher-frames", type=Path, default=None); ev.add_argument("--max-windows", type=int, default=None)
            ev.add_argument("--window-gradient", action="store_true",
                            help="with gradient on a held-out set: also the per-term update's loss after the term (T4-H)")
            ev.add_argument("--seed", type=int, default=0); ev.add_argument("--resamples", type=int, default=2000)
            ev.add_argument("--overwrite", action="store_true"); ev.add_argument("--smoke", action="store_true",
                                                                                   help="label the outputs as a smoke test")
        else:
            ev.add_argument("--lrs", type=float, nargs="*", default=[1e-5, 1e-4, 1e-3])
    report = sub.add_parser("report", help="pooled pre-registered endpoints over runs")
    report.add_argument("--runs", type=Path, nargs="+", required=True); report.add_argument("--output", type=Path, required=True)
    report.add_argument("--resamples", type=int, default=2000)
    plan = sub.add_parser("plan", help="print the queue commands (nothing is submitted)")
    plan.add_argument("--tracks", nargs="*", default=["t5", "t4", "t1"]); plan.add_argument("--hosts", nargs="*", default=["SmolLM2-360M", "SmolLM2-135M"])
    plan.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3]); plan.add_argument("--priority", type=int, default=PLAN_PRIORITY)
    plan.add_argument("--include-pending", action="store_true", help="also print jobs whose runs are still training (T4 seeds 2–3)")
    args = parser.parse_args(argv)
    if args.command == "items":
        print(json.dumps(run_items(args), indent=2, default=str))
    elif args.command == "fetch-openstax":
        print(json.dumps(fetch_openstax(args.output, collection=args.collection), indent=2))
    elif args.command == "teacher":
        print(json.dumps(run_teacher(args), indent=2, default=str))
    elif args.command == "evaluate":
        document = run_evaluate(args)
        print(json.dumps({"frames": {s: {r: m["f1"] for r, m in by.items()} for s, by in document.get("frames", {}).items()},
                          "seconds": document["seconds"]}, indent=2, default=str))
    elif args.command == "gradient-dev":
        print(json.dumps(run_gradient_dev(args), indent=2, default=str))
    elif args.command == "report":
        print(json.dumps(run_report(args)["endpoints"], indent=2, default=str))
    else:
        run_plan(args)


if __name__ == "__main__":
    main()
