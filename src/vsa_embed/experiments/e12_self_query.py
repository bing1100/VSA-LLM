"""E12 — self-query, phase A: the recall tool as a fixed pipeline (author decision 62; pre-registration
`experiments/e12-self-query/preregistration.md`). Evaluation only, on finished E9 runs.

A run's concept store (the E9 span channel's composer: a static frame bundle per term) is read back with the recall tool
(`vsa_embed.self_query`: unbind with the operator's own primary method, typed cleanup against the trained atomics) and the
recalled frame is put **in context, as text, before the question**. The host then answers by reading — it does not have to
unbind anything in its weights. Each item is scored under several conditions (the context differs, the question does not):

| condition | context before the prompt |
|---|---|
| `none` | nothing (the step-1 `own` source: the channel's injection only) |
| `recall:<store>` | the recall of the item's term from a store: `own` (the host run's own composer) or another run of the same stage, host and seed (`C5ut`: the untyped, role-blind store; `C5@2`: seed 2's store) |
| `roleless:<store>` | the same recall with its roles removed (one "associated with" line, fillers in a role-independent order) |
| `wrong:<store>` | twins only: the partner twin's recall, written with this twin's name (a model that follows the recalled text flips) |
| `symbolic` | the gold frame in the recall format (confidence 1.00): the upper bound of a perfect store |
| `definition` | the E11 prose definition written from the gold frame (`read_to_learn.t5_definition` / `t4_definition`, style `prose`) |
| `recall-all:<store>`, `recall-free:<store>`, `fields:<store>`, `noconf:<store>` | secondaries: all-atom cleanup; slot-free decoding (presence thresholds calibrated on the store's seen entries); the `fields` wording; no confidences |

**Item sets** (the E9 item directories; recall per set):
- role-swap twins (`e9-role-items/1`, kind `twins`) and natural role-ambiguous items (kind `natural`): the term's whole frame,
  slot-aware (the store's relation keys of the term; the fillers decoded from the vector);
- WP-UB understanding items (`e9-understanding/1`): `two_hop` items get **chained recall** along the item's path (hop 1 from the
  anchor's store, hop 2 from the store of the entry the recovered filler names); `reverse` items get **reverse lookup** (every
  store of the track — and the item set's new words — scanned for the item's relation and filler; top 5);
- new words (`e9-new-words/1`, v2): `property` items, the inserted frame's whole recall.

**Scores:** per item and template × candidate the summed log-probability `s` of the candidate after the (context +) prompt
(cloze: per token), as `e9_binding_items.continuation_scores` (output head in float32), and PMI against the context-free null
prompt (cached once per item set; the twin contrast does not use it). Twins: the twin contrast accuracy of step 1
(`e9_binding_items.twin_scores`); natural: the role contrast; understanding and new words: PMI-argmax accuracy. Every condition
that reads a store also records the **decode accuracy** of what it put in context (per slot; twins: the four critical slots of
a pair; two-hop: hop 1, bridge, hop 2; reverse: anchor above partner and anchor in the top 5).

Outputs (`RUN/self-query-<items>/`): `summary.json` (per condition the per-unit scores and decode accuracies), `predictions.jsonl.gz`,
`recalls.jsonl.gz` (every recalled text with its decoded edges: what the store says, inspectable), `report.md`,
`resolved_config.yaml`, `manifest.json`.

    python -m vsa_embed.experiments.e12_self_query evaluate --run RUN --items DIR [--conditions none,recall:own,symbolic,...]
    python -m vsa_embed.experiments.e12_self_query recall --store RUN --items DIR [--limit N]       (CPU; print recalled texts)
    python -m vsa_embed.experiments.e12_self_query queue --stage t5 --items DIR --priority 50 [--models …] [--seeds …] [--dry-run]
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import yaml

from .. import read_to_learn as rtl
from .. import self_query as sq
from . import e9_binding_items as role_items
from . import e9_ontology_edit as edit
from . import e9_understanding as und
from .e5_common import E5Run, finish_output, json_ready, open_run, start_output, write_json

SCHEMA = "e12-self-query/1"
ROOT = Path("experiments/e9-retrofit")
E12_ROOT = Path("experiments/e12-self-query")
OUTPUT_PREFIX = "self-query-"
RESULT_FILES = ("summary.json", "predictions.jsonl.gz", "recalls.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json")
RUN_NAME = re.compile(r"^(?P<host>.+)-(?P<mode>full|lora|frozen)-(?P<model>[^-]+)-s(?P<seed>\d+)$")
KINDS = ("none", "recall", "roleless", "wrong", "symbolic", "definition", "recall-all", "recall-free", "fields", "noconf")
STORE_KINDS = frozenset({"recall", "roleless", "wrong", "recall-all", "recall-free", "fields", "noconf"})
UNDERSTANDING_FAMILIES = ("two_hop", "reverse")
REVERSE_K = 5
SYMBOLIC_HOLDERS = 5
DEFINITION_STYLE = {"t5": "prose", "t4": "prose"}
SLOT_IN_TEXT = re.compile(r"\{(x|y|c)\}")
COMPOSING = ("C5", "C5rf", "C5ut", "C5tr", "C5sh", "U5", "U5u", "U5sb", "U5bu", "U5sl", "U5tr", "U5ut")


# ---------------------------------------------------------------- conditions and stores


@dataclass(frozen=True)
class Condition:
    kind: str
    store: str | None = None          # store label (`own`, `C5ut`, `C5@2`) for the store kinds

    @property
    def name(self) -> str:
        return self.kind if self.store is None else f"{self.kind}:{self.store}"

    @classmethod
    def parse(cls, text: str) -> "Condition":
        kind, _, store = text.strip().partition(":")
        if kind not in KINDS:
            raise ValueError(f"unknown condition kind {kind!r}; known: {KINDS}")
        if (kind in STORE_KINDS) != bool(store):
            raise ValueError(f"condition {text!r}: {'a store is required' if kind in STORE_KINDS else 'takes no store'}")
        return cls(kind, store or None)


def parse_conditions(text: str | Sequence[str]) -> list[Condition]:
    parts = text.split(",") if isinstance(text, str) else list(text)
    out = [Condition.parse(p) for p in parts if p.strip()]
    if len({c.name for c in out}) != len(out):
        raise ValueError("duplicate conditions")
    return out


def resolve_store(host_run: Path, label: str) -> Path:
    """The run folder of a store label: `own` → the host run; `MODEL[@SEED]` → the run of the same stage folder, host and
    seed (or SEED) whose model is MODEL (any host mode)."""
    host_run = Path(host_run)
    if label == "own":
        return host_run
    model, _, seed_text = label.partition("@")
    match = RUN_NAME.match(host_run.name)
    if match is None:
        candidate = Path(label)
        if candidate.exists():
            return candidate
        raise ValueError(f"cannot resolve store {label!r} next to {host_run}")
    seed = int(seed_text) if seed_text else int(match["seed"])
    found = [p for p in sorted(host_run.parent.glob(f"{match['host']}-*-{model}-s{seed}"))
             if (m := RUN_NAME.match(p.name)) and m["host"] == match["host"] and m["model"] == model and int(m["seed"]) == seed]
    if not found:
        raise FileNotFoundError(f"no {model} run with seed {seed} for host {match['host']} in {host_run.parent}")
    return found[0]


@dataclass
class LoadedStore:
    label: str
    run: Path
    store: sq.RecallStore
    ontology: dict[str, Any]
    operator: str
    relation_id: dict[str, int] = field(default_factory=dict)
    atomic_id: dict[str, int] = field(default_factory=dict)
    thresholds: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "run": str(self.run), "operator": self.operator, "method": self.store.method,
                "role_blind": self.store.role_blind, "family": self.store.composer.transform.family}


def load_store(run_dir: Path, label: str, *, loader: Callable[[Path], tuple[Any, dict[str, Any], dict[str, Any]]] | None = None
               ) -> LoadedStore:
    """A run's composer read from `final.pt` (the host is not loaded; `e9_binding_chain.load_composer`)."""
    if loader is None:
        from .e9_binding_chain import load_composer as loader
    composer, _, ontology = loader(Path(run_dir))
    return LoadedStore(label, Path(run_dir), sq.RecallStore(composer), ontology, composer.operator,
                       {n: i for i, n in enumerate(ontology["relation_names"])}, {n: i for i, n in enumerate(ontology["atomic_names"])})


def seen_entries(ontology: dict[str, Any], *, cap: int = 3000, seed: int = 0) -> list[int]:
    """Entries with training frequency ≥ 10 that are not held out (a seeded sample of at most `cap`): the presence calibration set."""
    frequency = np.asarray(ontology.get("train_frequency") or np.zeros(int(ontology["entry_count"])))
    held = {int(e) for e in ontology.get("heldout_entries", ())}
    degrees = np.diff(np.asarray(ontology["offsets"]))
    pool = [e for e in np.flatnonzero((frequency >= 10) & (degrees > 0)).tolist() if e not in held]
    if len(pool) > cap:
        pool = sorted(random.Random(seed).sample(pool, cap))
    return pool


# ---------------------------------------------------------------- item sets


@dataclass
class Prompt:
    """One item, normalized: templates with `{x}` / `{y}` / `{c}` slots, the real and null fills, rendered candidates."""

    id: str
    concept: str                       # the term whose recall goes in context (understanding: the anchor)
    templates: list[str]
    fills: dict[str, str]
    null_fills: dict[str, str]
    candidates: list[str]
    gold: int
    per_token: bool
    row: dict[str, Any]
    item: dict[str, Any]


@dataclass
class ItemSet:
    kind: str                          # twins | natural | understanding | new
    path: Path
    manifest: dict[str, Any]
    concepts: list[dict[str, Any]]
    prompts: list[Prompt]

    @property
    def track(self) -> str:
        return str(self.manifest.get("track") or self.manifest.get("lexicon") or "")

    @property
    def by_concept(self) -> dict[str, dict[str, Any]]:
        return {c["concept"]: c for c in self.concepts}

    def raw_items(self) -> list[dict[str, Any]]:
        return [p.item for p in self.prompts]


def _render(template: str, fills: dict[str, str]) -> str:
    return und.render(template, fills)


def load_item_set(path: Path, *, limit: int | None = None, families: Sequence[str] = UNDERSTANDING_FAMILIES) -> ItemSet:
    """Any of the four E9 item sets, normalized (module docstring). `limit` (smoke tests only): the first N concepts
    (twins: pairs; understanding: anchors per subset)."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    schema = manifest.get("schema")
    prompts: list[Prompt] = []
    if schema == role_items.SCHEMA:
        manifest, concepts, items = role_items.load_items(path)
        kind = manifest["kind"]
        if limit:
            keep = ({c["concept"] for c in concepts if c["pair"] < limit} if kind == "twins" else {c["concept"] for c in concepts[:limit]})
            concepts = [c for c in concepts if c["concept"] in keep]
            items = [i for i in items if i["concept"] in keep]
        surfaces = {c["concept"]: c["surface"] for c in concepts}
        for item in items:
            prompts.append(Prompt(item["id"], item["concept"], list(item["templates"]), {"x": surfaces[item["concept"]]},
                                  {"x": item["null"]}, list(item["candidates"]), int(item["gold"]), item["kind"] == "cloze",
                                  {"kind": item["kind"], "concept": item["concept"], "relation": item["relation"],
                                   "gold": int(item["gold"]), "meta": item.get("meta", {})}, item))
    elif schema == und.SCHEMA:
        kind = "understanding"
        manifest, concepts, items = und.load_items(path)
        items = [i for i in items if i["family"] in families and i["test"] in families]
        if limit:
            firsts: dict[str, list[str]] = defaultdict(list)
            for item in items:
                if item["anchor"] not in firsts[item["subset"]] and len(firsts[item["subset"]]) < limit:
                    firsts[item["subset"]].append(item["anchor"])
            keep = {a for anchors in firsts.values() for a in anchors}
            items = [i for i in items if i["anchor"] in keep]
        used = {cid for i in items for cid in i["slots"].values()} | {c["concept"] for c in concepts if c.get("role") == "bridge"}
        concepts = [c for c in concepts if c["concept"] in used]
        surfaces = {c["concept"]: c["surface"] for c in concepts}
        for item in items:
            fills = {slot: surfaces[cid] for slot, cid in item["slots"].items()} | dict(item["text"])
            prompts.append(Prompt(item["id"], item["anchor"], list(item["templates"]), fills, fills | dict(item["null"]),
                                  [_render(c, fills) for c in item["candidates"]], int(item["gold"]), False,
                                  {"family": item["family"], "test": item["test"], "subset": item["subset"], "anchor": item["anchor"],
                                   "relation": item["relation"], "pair": item.get("pair"), "chance": item["chance"],
                                   "meta": item.get("meta", {})}, item))
    elif schema == edit.SCHEMA_NEW:
        kind = "new"
        manifest, concepts, items = edit.load_item_dir(path, edit.SCHEMA_NEW)
        if limit:
            keep = {c["concept"] for c in concepts[:limit]}
            concepts = [c for c in concepts if c["concept"] in keep]
            items = [i for i in items if i["concept"] in keep]
        surfaces = {c["concept"]: c["surface"] for c in concepts}
        for item in items:
            if item["test"] != "property":
                continue
            prompts.append(Prompt(item["id"], item["concept"], list(item["templates"]), {"x": surfaces[item["concept"]]},
                                  {"x": item.get("null", "this")}, list(item["candidates"]), int(item["gold"]), False,
                                  {"concept": item["concept"], "test": item["test"], "relation": item["relation"],
                                   "edge_kind": item.get("edge_kind")}, item))
    else:
        raise ValueError(f"{path}: unsupported item schema {schema!r}")
    return ItemSet(kind, path, manifest, concepts, prompts)


# ---------------------------------------------------------------- recall contexts


def frame_ids(frame: Sequence[Sequence[str]], relation_id: dict[str, int], atomic_id: dict[str, int]) -> list[tuple[int, int]]:
    return edit.resolve_frame(frame, relation_id, atomic_id)


def same_entries(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two ontologies with the same entries (entry-based items may read either's stores)."""
    keys = ("entry_count", "alias_table_sha256")
    return all(a.get(k) == b.get(k) for k in keys) and list(a["atomic_names"]) == list(b["atomic_names"]) \
        and list(a["relation_names"]) == list(b["relation_names"])


def lexicon_for_track(track: str, family: str, ontology: dict[str, Any]) -> Any:
    from .e9_tracks import lexicon_for, track_spec
    return lexicon_for(track_spec(track, family), ontology)


def home_relations(ontology: dict[str, Any]) -> list[int]:
    """Per atomic the relation it is most often a filler of (−1: never a filler)."""
    relations = np.asarray(ontology["relations"], dtype=np.int64)
    fillers = np.asarray(ontology["fillers"], dtype=np.int64)
    counts = np.zeros((len(ontology["relation_names"]), len(ontology["atomic_names"])), dtype=np.int64)
    np.add.at(counts, (relations, fillers), 1)
    return np.where(counts.sum(0) > 0, counts.argmax(0), -1).tolist()


def concept_names(ontology: dict[str, Any]) -> list[str]:
    """Per entry a readable name: its single concept's name (without a `synthetic:` prefix), else the first one's."""
    names = [str(n) for n in ontology.get("concept_names") or []]
    members = ontology.get("entry_concepts") or [(e,) for e in range(int(ontology["entry_count"]))]
    out = []
    for e in range(int(ontology["entry_count"])):
        name = names[members[e][0]] if names and members[e] else f"entry {e}"
        out.append(name.split(":", 1)[1] if name.startswith("synthetic:") else name)
    return out


class ContextBuilder:
    """The text each condition puts before an item's prompts, and what it decoded (module docstring)."""

    def __init__(self, items: ItemSet, ontology: dict[str, Any], lexicon: Any, stores: dict[str, LoadedStore], *,
                 seed: int = 0, cleanup: str = "typed", max_lines: int | None = 32) -> None:
        self.items, self.ontology, self.lexicon, self.stores = items, ontology, lexicon, stores
        self.seed, self.cleanup, self.max_lines = seed, cleanup, max_lines
        self.relation_names = list(ontology["relation_names"])
        self.atomic_names = list(ontology["atomic_names"])
        self.relation_id = {n: i for i, n in enumerate(self.relation_names)}
        self.atomic_id = {n: i for i, n in enumerate(self.atomic_names)}
        self.home = home_relations(ontology)
        self.concepts = items.by_concept
        # gold frames: recorded by name (twins, new words, understanding anchors), else the entry's ontology frame (natural items)
        offsets = np.asarray(ontology["offsets"])
        relations, fillers = np.asarray(ontology["relations"]), np.asarray(ontology["fillers"])
        self.frames = {}
        for cid, c in self.concepts.items():
            if c.get("frame"):
                self.frames[cid] = frame_ids(c["frame"], self.relation_id, self.atomic_id)
            elif c.get("entry") is not None:
                lo, hi = int(offsets[int(c["entry"])]), int(offsets[int(c["entry"]) + 1])
                self.frames[cid] = list(zip(relations[lo:hi].tolist(), fillers[lo:hi].tolist()))
        self._atom_entry: np.ndarray | None = None
        self._names: list[str] | None = None
        self._pools: dict[str, tuple[torch.Tensor, list[str], list[str]]] = {}
        bridges = [c for c in self.concepts.values() if c.get("role") == "bridge" and c.get("entry") is not None]
        self.bridge_surface = {int(c["entry"]): c["surface"] for c in bridges}
        self.bridge_concept = {int(c["entry"]): c["concept"] for c in bridges}
        self.bridge_entry_of = {c.get("source"): int(c["entry"]) for c in bridges}
        self._owners = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))

    # -- shared pieces --------------------------------------------------------------------------------------------

    def writer(self, *, style: str = "statements", confidence: bool = True) -> sq.RecallWriter:
        return sq.RecallWriter(self.relation_names, self.atomic_names, self.lexicon, style=style, confidence=confidence, home=self.home,
                               max_lines=self.max_lines)

    @property
    def atom_entry(self) -> np.ndarray:
        if self._atom_entry is None:
            from .e9_binding_chain import atom_entries
            self._atom_entry = atom_entries(self.ontology)
        return self._atom_entry

    @property
    def names(self) -> list[str]:
        if self._names is None:
            self._names = concept_names(self.ontology)
        return self._names

    def store(self, label: str) -> LoadedStore:
        if label not in self.stores:
            raise KeyError(f"store {label!r} was not loaded")
        return self.stores[label]

    def _check_entries(self, store: LoadedStore) -> None:
        if not same_entries(store.ontology, self.ontology):
            raise ValueError(f"store {store.label} ({store.run}) has other entries than the host's ontology: entry-based items "
                             "cannot read it")

    def vector(self, store: LoadedStore, concept: str) -> torch.Tensor:
        """The concept's static store in `store`: its entry's bundle, or (new words, twins) its frame composed from the
        store's trained atomics and relations."""
        c = self.concepts[concept]
        if c.get("entry") is not None:
            self._check_entries(store)
            return store.store.entry_vectors()[int(c["entry"])]
        return store.store.frame_vector(frame_ids(c["frame"], store.relation_id, store.atomic_id))

    def thresholds(self, store: LoadedStore) -> torch.Tensor:
        if store.thresholds is None:
            store.thresholds = store.store.presence_thresholds(seen_entries(store.ontology, seed=self.seed), cleanup=self.cleanup)
        return store.thresholds

    def _named(self, lines: Sequence[sq.RecallLine]) -> list[dict[str, Any]]:
        return [{"subject": line.subject, "relation": None if line.relation is None else self.relation_names[line.relation],
                 "fillers": [[self.atomic_names[f.atom], round(f.score, 4)] for f in line.fillers]} for line in lines]

    def _frame_lines(self, condition: Condition, concept: str, *, vector_of: str | None = None) -> list[sq.RecallLine]:
        """The recalled lines of a term's whole frame (slot-aware; `vector_of`: read another concept's store — `wrong`)."""
        frame = self.frames[concept]
        if condition.kind == "symbolic":
            return sq.symbolic_lines(frame)
        store = self.store(condition.store)
        vector = self.vector(store, vector_of or concept)
        slots = [r for r, _ in frame]
        if condition.kind == "recall-free":
            return store.store.decode_free(vector, self.thresholds(store), cleanup=self.cleanup)
        cleanup = "all" if condition.kind == "recall-all" else self.cleanup
        lines = store.store.decode_slots(vector, slots, cleanup=cleanup)
        return sq.roleless(lines) if condition.kind == "roleless" else lines

    def _definition(self, concept: str) -> str | None:
        track = self.items.track
        style = DEFINITION_STYLE.get(track)
        if style is None:
            return None
        c = self.concepts[concept]
        facts: dict[str, list[str]] = defaultdict(list)
        for r, a in self.frames[concept]:
            text = self.lexicon.text(self.atomic_names[a])
            if text:
                facts[self.relation_names[r]].append(text)
        rng = random.Random(f"{self.seed}|{concept}|{style}")
        writer = rtl.t4_definition if track == "t4" else rtl.t5_definition
        return writer(c["surface"], dict(facts), style, rng)

    # -- per item set ---------------------------------------------------------------------------------------------

    def build(self, condition: Condition) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """(context text per item id, decode records) of one condition."""
        if condition.kind == "none":
            return {}, []
        if self.items.kind == "understanding":
            return self._understanding(condition)
        return self._frames(condition)

    def _frames(self, condition: Condition) -> tuple[dict[str, str], list[dict[str, Any]]]:
        style = "fields" if condition.kind == "fields" else "statements"
        # `roleless` prints no confidences: a filler's cleanup score comes from unbinding its role
        writer = self.writer(style=style, confidence=condition.kind not in {"noconf", "roleless"})
        texts, records = {}, []
        for concept, c in self.concepts.items():
            if concept not in self.frames:
                continue
            if condition.kind == "definition":
                text = self._definition(concept)
                if text is None:
                    raise ValueError(f"no definition writer for track {self.items.track!r}")
                texts[concept] = text
                records.append({"condition": condition.name, "concept": concept, "text": text})
                continue
            if condition.kind == "wrong":
                if self.items.kind != "twins":
                    raise ValueError("the `wrong` condition needs role-swap twins")
                lines = self._frame_lines(Condition("recall", condition.store), concept, vector_of=c["partner"])
            else:
                lines = self._frame_lines(condition, concept)
            text = writer.render(c["surface"], lines)
            texts[concept] = text
            accuracy = sq.slot_accuracy(lines, self.frames[concept])
            records.append({"condition": condition.name, "concept": concept, "store": condition.store, "text": text,
                            "lines": self._named(lines), "slots": {self.relation_names[r]: v for r, v in accuracy.items()},
                            "roles": any(line.relation is not None for line in lines)})
        return {p.id: texts[p.concept] for p in self.items.prompts if p.concept in texts}, records

    def _bridge_surface(self, item: dict[str, Any], bridge: int, atom: int, writer: sq.RecallWriter) -> str:
        return self.bridge_surface.get(int(bridge)) or writer.atom_text(atom)

    def _gold_hop1(self, anchor: str, r1: int, bridge: int | None) -> int | None:
        options = [f for r, f in self.frames.get(anchor, []) if r == r1]
        if bridge is not None:
            named = [f for f in options if f < self.atom_entry.size and int(self.atom_entry[f]) == bridge]
            if named:
                return named[0]
        return options[0] if options else None

    def _bridge_entry(self, item: dict[str, Any]) -> int | None:
        return self.bridge_entry_of.get(item["meta"].get("bridge"))

    def _pool(self, store: LoadedStore) -> tuple[torch.Tensor, list[str], list[str]]:
        """Reverse-lookup pool of a store: every entry's store plus the item set's new words (frames composed): vectors,
        surfaces and keys (`entry:<id>` / the concept id)."""
        if store.label not in self._pools:
            self._check_entries(store)
            vectors = [store.store.entry_vectors()]
            surfaces = list(self.names)
            keys = [f"entry:{e}" for e in range(len(surfaces))]
            for c in self.concepts.values():
                if c.get("entry") is not None:
                    surfaces[int(c["entry"])] = c["surface"]
            new = [c for c in self.concepts.values() if c.get("entry") is None and c.get("frame")]
            if new:
                vectors.append(store.store.frame_vectors([frame_ids(c["frame"], store.relation_id, store.atomic_id) for c in new]))
                surfaces += [c["surface"] for c in new]
                keys += [c["concept"] for c in new]
            self._pools[store.label] = (torch.cat(vectors), surfaces, keys)
        return self._pools[store.label]

    def _holders(self, relation: int, filler: int) -> list[str]:
        """Keys of every entry and new word whose gold frame holds (relation, filler)."""
        o = self.ontology
        hit = (np.asarray(o["relations"]) == relation) & (np.asarray(o["fillers"]) == filler)
        keys = [f"entry:{e}" for e in sorted(set(self._owners[hit].tolist()))]
        keys += [cid for cid, c in self.concepts.items() if c.get("entry") is None and (relation, filler) in self.frames.get(cid, [])]
        return keys

    def _key(self, concept: str) -> str:
        c = self.concepts[concept]
        return f"entry:{c['entry']}" if c.get("entry") is not None else concept

    def _understanding(self, condition: Condition) -> tuple[dict[str, str], list[dict[str, Any]]]:
        if condition.kind in {"roleless", "wrong", "recall-free"}:
            raise ValueError(f"condition {condition.kind!r} is not defined for the understanding items")
        style = "fields" if condition.kind == "fields" else "statements"
        writer = self.writer(style=style, confidence=condition.kind != "noconf")
        cleanup = "all" if condition.kind == "recall-all" else self.cleanup
        texts, records = {}, []
        for prompt in self.items.prompts:
            item = prompt.item
            anchor = self.concepts[item["anchor"]]
            if item["family"] == "two_hop":
                r1, r2 = (self.relation_id[r] for r in item["meta"]["path"])
                bridge = self._bridge_entry(item)
                answer = self.atomic_id.get(item["meta"]["answer"])
                gold1 = self._gold_hop1(item["anchor"], r1, bridge)
                if condition.kind == "definition":
                    bridge_concept = self.bridge_concept.get(bridge) if bridge is not None else None
                    parts = [self._definition(item["anchor"])] + ([self._definition(bridge_concept)] if bridge_concept else [])
                    if any(p is None for p in parts):
                        raise ValueError(f"no definition writer for track {self.items.track!r}")
                    texts[prompt.id] = "\n".join(parts)
                    records.append({"condition": condition.name, "item": prompt.id, "text": texts[prompt.id]})
                    continue
                if condition.kind == "symbolic":
                    if gold1 is None or answer is None:
                        continue
                    hop1 = sq.RecallLine(r1, [sq.Filler(gold1, 1.0)])
                    hop2 = sq.RecallLine(r2, [sq.Filler(answer, 1.0)])
                    meta = {"bridge": bridge if bridge is not None else -1, "hop1": gold1, "hop2": answer}
                else:
                    store = self.store(condition.store)
                    self._check_entries(store)                 # the bridge's store is read by entry id
                    (lines, meta) = store.store.chain(self.vector(store, item["anchor"]), r1, r2, self.atom_entry, cleanup=cleanup)
                    hop1, hop2 = lines[0], (lines[1] if len(lines) > 1 else None)
                blocks = [(anchor["surface"], f"recall({anchor['surface']}, {sq.relation_phrase(self.relation_names[r1])}):", [hop1])]
                if hop2 is not None and meta["bridge"] >= 0:
                    who = self._bridge_surface(item, meta["bridge"], meta["hop1"], writer)
                    blocks.append((who, f"recall({who}, {sq.relation_phrase(self.relation_names[r2])}):", [hop2]))
                texts[prompt.id] = writer.render_blocks(blocks)
                records.append({"condition": condition.name, "item": prompt.id, "store": condition.store, "text": texts[prompt.id],
                                "hop1_correct": float(meta["hop1"] == gold1) if gold1 is not None else None,
                                "bridge_correct": float(bridge is not None and meta["bridge"] == bridge),
                                "hop2_correct": float(answer is not None and meta["hop2"] == answer)})
            elif item["family"] == "reverse":
                relation = self.relation_id[item["relation"]]
                filler = self.atomic_id.get(item["meta"]["filler"])
                if filler is None or condition.kind == "definition":          # no definition for a lookup: not scored
                    continue
                phrase = sq.relation_phrase(item["relation"])
                filler_text = writer.filler_text(relation, filler)
                anchor_key, partner_key = self._key(item["slots"]["x"]), self._key(item["slots"]["y"])
                if condition.kind == "symbolic":
                    holders = [k for k in self._holders(relation, filler) if k != anchor_key]
                    rng = random.Random(f"{self.seed}|{prompt.id}|symbolic")
                    chosen = [anchor_key] + rng.sample(holders, min(len(holders), SYMBOLIC_HOLDERS - 1))
                    rng.shuffle(chosen)
                    _, surfaces, keys = self._pool_surfaces()
                    index = {k: i for i, k in enumerate(keys)}
                    lines = [sq.RecallLine(relation, [sq.Filler(filler, 1.0)], subject=surfaces[index[k]]) for k in chosen if k in index]
                    record = {"anchor_rank": 1 + chosen.index(anchor_key), "pair_correct": 1.0, "anchor_in_top": 1.0}
                else:
                    store = self.store(condition.store)
                    pool, surfaces, keys = self._pool(store)
                    scores = store.store.reverse_scores(relation, filler, pool)
                    index = {k: i for i, k in enumerate(keys)}
                    top = torch.topk(scores, min(REVERSE_K, scores.numel()))
                    lines = [sq.RecallLine(relation, [sq.Filler(filler, float(s))], subject=surfaces[i])
                             for s, i in zip(top.values.tolist(), top.indices.tolist())]
                    a, p = index.get(anchor_key), index.get(partner_key)
                    sa = float(scores[a]) if a is not None else float("-inf")
                    sp = float(scores[p]) if p is not None else float("-inf")
                    record = {"anchor_rank": int((scores > sa).sum()) + 1 if a is not None else None,
                              "pair_correct": 0.5 if abs(sa - sp) <= 1e-7 else float(sa > sp),
                              "anchor_in_top": float(a is not None and a in top.indices.tolist())}
                texts[prompt.id] = writer.render(filler_text, lines, call=f"lookup({phrase}, {filler_text}):")
                records.append({"condition": condition.name, "item": prompt.id, "store": condition.store, "text": texts[prompt.id], **record})
        return texts, records

    def _pool_surfaces(self) -> tuple[None, list[str], list[str]]:
        """Surfaces and keys of the reverse pool without vectors (symbolic lookup)."""
        surfaces = list(self.names)
        keys = [f"entry:{e}" for e in range(len(surfaces))]
        for c in self.concepts.values():
            if c.get("entry") is not None:
                surfaces[int(c["entry"])] = c["surface"]
        new = [c for c in self.concepts.values() if c.get("entry") is None and c.get("frame")]
        surfaces += [c["surface"] for c in new]
        keys += [c["concept"] for c in new]
        return None, surfaces, keys


# ---------------------------------------------------------------- scoring


def _guard(context: str) -> str:
    """A context never holds a template slot (`{x}`, `{y}`, `{c}`): such a pattern (a chemical name) is spaced out."""
    return SLOT_IN_TEXT.sub(lambda m: "{ " + m.group(1) + "}", context)


def prompt_texts(prompts: Sequence[Prompt], contexts: dict[str, str]) -> tuple[list[str], list[str]]:
    prefixes, continuations = [], []
    for p in prompts:
        context = contexts.get(p.id)
        for template in p.templates:
            prefix = _render((_guard(context) + "\n" if context else "") + template, p.fills)
            for candidate in p.candidates:
                prefixes.append(prefix); continuations.append(candidate)
    return prefixes, continuations


def check_lengths(tokenizer: Any, prefixes: Sequence[str], continuations: Sequence[str], max_length: int) -> int:
    """The longest text in tokens; raises when one would be truncated (truncation would cut the continuation)."""
    longest = 0
    for start in range(0, len(prefixes), 2048):
        texts = [p + c for p, c in zip(prefixes[start:start + 2048], continuations[start:start + 2048])]
        lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
        longest = max([longest, *lengths])
    if longest > max_length:
        raise ValueError(f"a prompt has {longest} tokens, more than the adapter's max_length {max_length}: raise --max-length")
    return longest


def score_prompts(adapter: Any, prompts: Sequence[Prompt], contexts: dict[str, str],
                  null_cache: dict[tuple[str, str], tuple[float, float]] | None = None, *, check: bool = True
                  ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[float, float]], int]:
    """Per item: `s` (template × candidate summed log-probability after the context and prompt; per token for cloze items),
    PMI against the context-free null prompt, PMI-argmax correctness per template (ties 0.5). Returns (rows, null cache,
    longest text in tokens)."""
    prefixes, continuations = prompt_texts(prompts, contexts)
    longest = check_lengths(adapter.tokenizer, prefixes, continuations, adapter.max_length) if check and prefixes else 0
    sums, counts = role_items.continuation_scores(adapter, prefixes, continuations) if prefixes else (np.zeros(0), np.zeros(0))
    cache = dict(null_cache or {})
    null_pairs = [(_render(t, p.null_fills), c) for p in prompts for t in p.templates for c in p.candidates]
    missing = sorted(set(null_pairs) - set(cache))
    if missing:
        ms, mc = role_items.continuation_scores(adapter, [m[0] for m in missing], [m[1] for m in missing])
        cache.update(zip(missing, zip(ms.tolist(), mc.tolist())))
    rows, cursor = [], 0
    for p in prompts:
        n_t, k = len(p.templates), len(p.candidates)
        span = slice(cursor, cursor + n_t * k)
        nulls = null_pairs[span]
        cursor += n_t * k
        raw_sum, raw_count = sums[span], counts[span]
        null_sum = np.asarray([cache[n][0] for n in nulls]); null_count = np.asarray([cache[n][1] for n in nulls])
        if p.per_token:
            s = (raw_sum / np.maximum(raw_count, 1)).reshape(n_t, k)
            s0 = (null_sum / np.maximum(null_count, 1)).reshape(n_t, k)
        else:
            s, s0 = raw_sum.reshape(n_t, k), null_sum.reshape(n_t, k)
        pmi = s - s0
        correct = [(1.0 if int(np.argmax(row)) == p.gold else 0.0) if np.ptp(row) > 0 else 1.0 / k for row in pmi]
        rows.append({"id": p.id, **p.row, "s": s.round(5).tolist(), "pmi": pmi.round(5).tolist(), "correct": correct})
    return rows, cache, longest


# ---------------------------------------------------------------- per-unit scores


def twin_units(rows: Sequence[dict[str, Any]], kind: str = "choice") -> dict[str, dict[str, float]]:
    return {str(k): v for k, v in role_items.twin_scores(rows, kind).items()}


def natural_units(rows: Sequence[dict[str, Any]], kind: str = "choice") -> dict[str, dict[str, float]]:
    return role_items.natural_scores(rows, kind)


def understanding_units(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per item: accuracy (mean over templates) and accuracy − chance."""
    return {r["id"]: {"accuracy": float(np.mean(r["correct"])), "score": float(np.mean(r["correct"])) - float(r["chance"]),
                      "family": r["family"], "subset": r["subset"]} for r in rows}


def new_word_units(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {r["id"]: {"accuracy": float(np.mean(r["correct"])), "edge_kind": r.get("edge_kind"), "relation": r["relation"]}
            for r in rows}


def twin_decode(records: Sequence[dict[str, Any]], concepts: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per twin pair: the share of its four critical slots (both twins × r1, r2) whose gold filler the recall put in
    context (`decode`) and whether all four were, in their roles (`decode_all`). A role-blind context (no role stated)
    counts a filler without its role in `decode` and has no `decode_all`."""
    by_pair: dict[str, list[float]] = defaultdict(list)
    roles: dict[str, bool] = defaultdict(lambda: True)
    for record in records:
        c = concepts.get(record.get("concept"))
        if c is None or "slots" not in record:
            continue
        for relation in c["relations"]:
            by_pair[str(c["pair"])].append(float(record["slots"].get(relation, 0.0)))
        roles[str(c["pair"])] &= bool(record.get("roles", True))
    out = {}
    for p, v in by_pair.items():
        if len(v) == 4:
            out[p] = {"decode": float(np.mean(v))}
            if roles[p]:
                out[p]["decode_all"] = float(all(x == 1.0 for x in v))
    return out


def concept_decode(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per concept: the mean slot accuracy of its recall."""
    return {r["concept"]: float(np.mean(list(r["slots"].values()))) for r in records if r.get("slots")}


def summarize(item_set: ItemSet, results: dict[str, list[dict[str, Any]]], records: dict[str, list[dict[str, Any]]],
              resolved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per condition: per-unit scores (and the decode accuracy of what was put in context) and their means."""
    linked = {c for c, r in resolved.items() if r.get("status") in ("linked", "no entry")}
    out: dict[str, Any] = {"kind": item_set.kind, "conditions": {}, "linked": len(linked), "concepts": len(resolved)}
    concepts = item_set.by_concept
    for name, rows in results.items():
        block: dict[str, Any] = {}
        recs = records.get(name, [])
        if item_set.kind == "twins":
            complete = {str(c["pair"]) for c in concepts.values() if c["concept"] in linked and c["partner"] in linked}
            decode = twin_decode(recs, concepts)
            for kind in role_items.ITEM_KINDS:
                units = {k: v for k, v in twin_units(rows, kind).items() if k in complete}
                for k, v in units.items():
                    if k in decode:
                        v.update(decode[k])
                block[kind] = {"units": units, "mean": _means(units)}
        elif item_set.kind == "natural":
            decode = concept_decode(recs)
            for kind in role_items.ITEM_KINDS:
                units = {k: v for k, v in natural_units(rows, kind).items() if k in linked}
                for k, v in units.items():
                    if k in decode:
                        v["decode"] = decode[k]
                block[kind] = {"units": units, "mean": _means(units)}
        elif item_set.kind == "understanding":
            by_item = {r.get("item"): r for r in recs}
            units = understanding_units([r for r in rows if r["anchor"] in linked])
            for k, v in units.items():
                rec = by_item.get(k, {})
                for key in ("hop1_correct", "bridge_correct", "hop2_correct", "pair_correct", "anchor_in_top"):
                    if rec.get(key) is not None:
                        v[key] = rec[key]
            block["items"] = {"units": units, "by": _grouped(units, ("family", "subset"))}
        else:
            decode = concept_decode(recs)
            units = new_word_units([r for r in rows if r["concept"] in linked])
            for k, v in units.items():
                concept = k.split("-property-")[0]
                if concept in decode:
                    v["decode"] = decode[concept]
            block["property"] = {"units": units, "mean": _means(units)}
        out["conditions"][name] = block
    return out


def _means(units: dict[str, dict[str, Any]]) -> dict[str, float]:
    keys = sorted({k for v in units.values() for k, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)})
    return {k: float(np.mean([v[k] for v in units.values() if isinstance(v.get(k), (int, float))])) for k in keys} | {"n": len(units)}


def _grouped(units: dict[str, dict[str, Any]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for k, v in units.items():
        groups["/".join(str(v.get(key)) for key in keys)][k] = v
        groups[str(v.get(keys[0])) + "/all"][k] = v
    return {g: _means(members) for g, members in sorted(groups.items())}


# ---------------------------------------------------------------- the evaluation


@contextlib.contextmanager
def host_view(run: E5Run, item_set: ItemSet) -> Iterator[tuple[Any, dict[str, int]]]:
    """The host's adapter for the item set, with every new concept (twins, new words, new-word anchors) inserted with its
    gold frame and linked (`e9_ontology_edit.inserted_entries`, `extended_adapter`); yields (adapter, concept → entry)."""
    ontology = run.ontology
    if ontology is None:
        raise ValueError(f"{run.path} has no ontology")
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    new = [c for c in item_set.concepts if c.get("entry") is None and c.get("frame")]
    base = int(ontology["entry_count"])
    ids = {c["concept"]: int(c["entry"]) for c in item_set.concepts if c.get("entry") is not None}
    ids.update({c["concept"]: base + i for i, c in enumerate(new)})
    if not new:
        yield run.adapter, ids
        return
    frames = [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in new]
    adapter = edit.extended_adapter(run, {c["surface"]: ids[c["concept"]] for c in new})
    with edit.inserted_entries(run.channel, len(new), frames):
        yield adapter, ids


def link_status(adapter: Any, item_set: ItemSet, ids: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Per concept: does its surface inject its entry where its first prompt reads it (the evaluators' own checks)?"""
    if item_set.kind == "understanding":
        return und.check_links(adapter, item_set.concepts, item_set.raw_items(), ids)
    return edit.link_check(adapter, item_set.concepts, item_set.raw_items(), ids)


@contextlib.contextmanager
def smaller_batches(adapter: Any, factor: int) -> Iterator[None]:
    saved = adapter.batch_size
    adapter.batch_size = max(1, saved // max(1, factor))
    try:
        yield
    finally:
        adapter.batch_size = saved


def evaluate(run: E5Run, item_set: ItemSet, conditions: Sequence[Condition], stores: dict[str, LoadedStore], *, seed: int = 0,
             cleanup: str = "typed", context_batch_factor: int = 2, max_lines: int | None = 32,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score the item set under each condition (module docstring)."""
    track = item_set.track
    family = item_set.manifest.get("family") or run.config.get("e9_family") or "smollm2"
    lexicon = lexicon_for_track(track, family, run.ontology)
    builder = ContextBuilder(item_set, run.ontology, lexicon, stores, seed=seed, cleanup=cleanup, max_lines=max_lines)
    started = time.monotonic()
    results: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    longest: dict[str, int] = {}
    with host_view(run, item_set) as (adapter, ids):
        resolved = link_status(adapter, item_set, ids)
        cache = None
        for condition in conditions:
            t0 = time.monotonic()
            contexts, recs = builder.build(condition)
            prompts = [p for p in item_set.prompts if condition.kind == "none" or p.id in contexts]
            log(f"  self-query ({item_set.kind}): {condition.name} — {len(prompts)} items")
            with smaller_batches(adapter, context_batch_factor if contexts else 1):
                rows, cache, longest[condition.name] = score_prompts(adapter, prompts, contexts, cache)
            results[condition.name], records[condition.name] = rows, recs
            timings[condition.name] = round(time.monotonic() - t0, 1)
    return {"kind": item_set.kind, "resolved": resolved, "conditions": [c.name for c in conditions], "results": results,
            "records": records, "timings": timings, "longest_tokens": longest, "seconds": time.monotonic() - started,
            "stores": {label: s.describe() for label, s in stores.items()}}


def default_conditions(model: str, kind: str, *, stage_models: Sequence[str] = (), store_seeds: Sequence[int] = ()) -> list[str]:
    """The pre-registered conditions of a host model on an item set kind (pre-registration §5). Composing hosts read their
    own store; hosts without one (C0′, C2, C6*) read the C5 store of their seed; P0 (one run) reads the C5 stores of
    `store_seeds`."""
    if model == "P0":
        stores = [f"C5@{s}" for s in store_seeds] or ["C5"]
    else:
        stores = ["own" if model in COMPOSING else "C5"]
    primary = stores[0]
    out = ["none", *(f"recall:{s}" for s in stores), "symbolic", "definition"]
    if kind in {"twins", "natural", "new"}:
        out.append(f"roleless:{primary}")
    if kind == "twins":
        out.append(f"wrong:{primary}")
    if model == "C5":                                  # the role-blind stores of the same seed (decision 60's controls)
        out += [f"recall:{m}" for m in ("C5ut", "C5tr") if m in stage_models and (kind != "new" or m == "C5ut")]
    if model == "C5ut" and "C5" in stage_models:
        out.append("recall:C5")
    return list(dict.fromkeys(out))


def output_folder(run_dir: Path, items: Path | str, tag: str | None = None) -> Path:
    return Path(run_dir) / (OUTPUT_PREFIX + Path(items).name + (f"-{tag}" if tag else ""))


def render_report(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    label = f" — {header['label']}" if header.get("label") else ""
    lines = [f"# E12 self-query, phase A ({summary['kind']}) — {source['condition']} seed {source['seed']} ({source['size']}){label}", "",
             f"Items `{header['items']}`; {summary['linked']} of {summary['concepts']} concepts link. Conditions: "
             + ", ".join(f"`{c}`" for c in summary["conditions"]) + ". Stores: "
             + ", ".join(f"`{k}` = {v['run']} ({v['family']}, {v['method']})" for k, v in header.get("stores", {}).items()) + ".", ""]
    if summary["kind"] in {"twins", "natural"}:
        lines += ["| condition | kind | units | contrast | item | decode | all slots decoded |", "|---|---|---:|---:|---:|---:|---:|"]
        for name, block in summary["conditions"].items():
            for kind in role_items.ITEM_KINDS:
                m = block.get(kind, {}).get("mean", {})
                if not m.get("n"):
                    continue
                dec = f"{m['decode']:.3f}" if "decode" in m else "—"
                full = f"{m['decode_all']:.3f}" if "decode_all" in m else "—"
                lines.append(f"| {name} | {kind} | {m['n']} | {m.get('contrast', float('nan')):.3f} | {m.get('item', float('nan')):.3f} | {dec} | {full} |")
    elif summary["kind"] == "understanding":
        lines += ["| condition | family / subset | items | accuracy | − chance | hop 1 | bridge | hop 2 | pair (reverse) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, block in summary["conditions"].items():
            for group, m in block["items"]["by"].items():
                cells = [f"{m[k]:.3f}" if k in m else "—" for k in ("hop1_correct", "bridge_correct", "hop2_correct", "pair_correct")]
                lines.append(f"| {name} | {group} | {m['n']} | {m['accuracy']:.3f} | {m['score']:+.3f} | " + " | ".join(cells) + " |")
    else:
        lines += ["| condition | items | property accuracy | decode |", "|---|---:|---:|---:|"]
        for name, block in summary["conditions"].items():
            m = block["property"]["mean"]
            lines.append(f"| {name} | {m['n']} | {m.get('accuracy', float('nan')):.3f} | " + (f"{m['decode']:.3f}" if "decode" in m else "—") + " |")
    lines += ["", "`contrast`: twin / role contrast accuracy (chance 0.5); `item`: PMI-argmax accuracy; `decode`: share of the "
              "gold fillers the condition's context states (twins: the four critical slots of a pair).", ""]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    item_set = load_item_set(args.items, limit=args.limit)
    run_dir = Path(args.run)
    match = RUN_NAME.match(run_dir.name)
    model = match["model"] if match else run_dir.name
    stage_models = sorted({m["model"] for p in run_dir.parent.iterdir() if (m := RUN_NAME.match(p.name))}) if run_dir.parent.exists() else []
    conditions = parse_conditions(args.conditions) if args.conditions else parse_conditions(default_conditions(model, item_set.kind,
                                                                                                                stage_models=stage_models))
    labels = sorted({c.store for c in conditions if c.store})
    output = Path(args.output or output_folder(run_dir, args.items, args.tag))
    family = item_set.manifest.get("family") or "smollm2"
    alias_table = args.alias_table or ensure_alias_table(track_spec(item_set.track, family))
    config = {"experiment": "e12-self-query", "phase": "A", "run": str(run_dir), "items": str(args.items),
              "conditions": [c.name for c in conditions], "stores": {l: str(resolve_store(run_dir, l)) for l in labels},
              "cleanup": args.cleanup, "seed": args.seed, "limit": args.limit, "batch_size": args.batch_size,
              "max_length": args.max_length, "max_lines": args.max_lines, "alias_table": str(alias_table) if alias_table else None,
              "label": args.label}
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    stores = {label: load_store(resolve_store(run_dir, label), label) for label in labels}
    run = open_run(run_dir, device=args.device, batch_size=args.batch_size, max_length=args.max_length, alias_table=alias_table)
    evaluation = evaluate(run, item_set, conditions, stores, seed=args.seed, cleanup=args.cleanup, max_lines=args.max_lines)
    summary = summarize(item_set, evaluation["results"], evaluation["records"], evaluation["resolved"])
    summary["conditions_order"] = evaluation["conditions"]
    header = {"source": run.describe(), "items": str(args.items), "kind": item_set.kind, "label": args.label,
              "stores": evaluation["stores"]}
    und.write_jsonl_gz(output / "predictions.jsonl.gz", ({"condition": c, **row} for c, rows in evaluation["results"].items() for row in rows))
    und.write_jsonl_gz(output / "recalls.jsonl.gz", (r for recs in evaluation["records"].values() for r in recs))
    write_json(output / "summary.json", {**header, "summary": summary, "resolved": evaluation["resolved"], "timings": evaluation["timings"],
                                         "longest_tokens": evaluation["longest_tokens"], "seconds": evaluation["seconds"],
                                         "items_manifest_sha256": item_set.manifest.get("sha256")})
    (output / "report.md").write_text(render_report(summary, header))
    peak =torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {**header, "summary": summary, "output": str(output), "timings": evaluation["timings"], "seconds": evaluation["seconds"],
            "peak_gb": peak}


def load_evaluation(folder: Path) -> dict[str, Any] | None:
    folder = Path(folder)
    if not (folder / "summary.json").exists():
        return None
    return json.loads((folder / "summary.json").read_text())


# ---------------------------------------------------------------- the recall command (CPU)


def run_recall(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Print (and optionally write) the recalled texts of an item set from one store: no host, CPU only."""
    item_set = load_item_set(args.items, limit=args.limit)
    store = load_store(Path(args.store), "own")
    family = item_set.manifest.get("family") or "smollm2"
    lexicon = lexicon_for_track(item_set.track, family, store.ontology)
    builder = ContextBuilder(item_set, store.ontology, lexicon, {"own": store}, seed=args.seed, cleanup=args.cleanup)
    _, records = builder.build(Condition.parse(args.condition))
    if args.output:
        und.write_jsonl_gz(Path(args.output), records)
    return records


# ---------------------------------------------------------------- queue


def evaluate_command(run_dir: Path, items: Path, *, python: str = sys.executable, batch_size: int = 16, conditions: Sequence[str] | None = None,
                     max_length: int = 768, limit: int | None = None, tag: str | None = None) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e12_self_query", "evaluate", "--run", str(run_dir), "--items", str(items), "--overwrite",
            "--batch-size", str(batch_size), "--max-length", str(max_length), *(["--limit", str(limit)] if limit else []),
            *(["--conditions", ",".join(conditions)] if conditions else []), *(["--tag", tag] if tag else [])]


CONTEXT_BATCH = {"SmolLM2-135M": 32, "SmolLM2-360M": 16, "Qwen3-0.6B-Base": 8, "Qwen3-1.7B-Base": 4}
# Pre-registered subsets (preregistration §3): the first 300 new words; at most 150 WP-UB anchors per subset. T4's natural items
# hold frames of up to 24 edges with long chemical names (contexts up to ≈ 1,250 tokens).
SET_LIMIT = {"new": 300, "understanding": 150}
SET_MAX_LENGTH = {"natural": 1536}


def queue_stage(stage: str, items: Path, *, priority: int = 50, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                hosts: Sequence[str] | None = None, root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None,
                dry_run: bool = False) -> list[dict[str, Any]]:
    """One GPU-lane job per run of `stage` (finished or not: a job waits in the queue for its run's training), named
    `<stage>-<stem>-<output folder>` (idempotent). P0 reads the C5 stores of seeds 1–3 (its single run stands for every seed)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import _config_host, stage_python
    kind = load_item_set(items, limit=1).kind
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    parsed = [(p, RUN_NAME.match(p.stem)) for p in configs]
    stage_models = sorted({m["model"] for _, m in parsed if m})
    host_names = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(host_names) if host_names else pinned_python())
    jobs = []
    for path, match in parsed:
        if match is None:
            continue
        model, seed, host = match["model"], int(match["seed"]), match["host"]
        if (models and model not in models) or (seeds and seed not in seeds) or (hosts and host not in hosts):
            continue
        c5_seeds = sorted(int(m["seed"]) for _, m in parsed if m and m["host"] == host and m["model"] == "C5"
                          and (not seeds or int(m["seed"]) in seeds))
        conditions = default_conditions(model, kind, stage_models=[m["model"] for _, m in parsed if m and m["host"] == host],
                                        store_seeds=c5_seeds)
        run_dir = Path(root) / "runs" / stage / path.stem
        batch = CONTEXT_BATCH.get(host, 8) // (2 if kind == "natural" else 1)
        jobs.append({"name": f"{stage}-{path.stem}-{output_folder(run_dir, items).name}", "priority": int(priority), "model": model,
                     "command": evaluate_command(run_dir, items, python=python, batch_size=max(1, batch), conditions=conditions,
                                                 max_length=SET_MAX_LENGTH.get(kind, 768), limit=SET_LIMIT.get(kind))})
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
    ev = sub.add_parser("evaluate", help="score an item set with recalled text in context, per condition (GPU)")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--conditions", default="", help="comma-separated; default: the pre-registered set of the run's model")
    ev.add_argument("--output", type=Path, default=None); ev.add_argument("--tag", default=None)
    ev.add_argument("--alias-table", type=Path, default=None); ev.add_argument("--cleanup", default="typed", choices=sq.CLEANUPS)
    ev.add_argument("--seed", type=int, default=0); ev.add_argument("--batch-size", type=int, default=16)
    ev.add_argument("--max-length", type=int, default=768); ev.add_argument("--device", default=None)
    ev.add_argument("--max-lines", type=int, default=32, help="recalled lines per call (the rest is cut; recorded)")
    ev.add_argument("--limit", type=int, default=None, help="the first N concepts (twins: pairs; understanding: anchors per subset): "
                    "the pre-registered subsets (new words 300, understanding 150; `queue` passes them), else pilots and smoke tests")
    ev.add_argument("--label", default=None, help="a free label recorded in the outputs (e.g. PILOT)")
    ev.add_argument("--overwrite", action="store_true")
    rc = sub.add_parser("recall", help="print the recalled texts of an item set from one store (CPU, no host)")
    rc.add_argument("--store", type=Path, required=True); rc.add_argument("--items", type=Path, required=True)
    rc.add_argument("--condition", default="recall:own"); rc.add_argument("--cleanup", default="typed", choices=sq.CLEANUPS)
    rc.add_argument("--limit", type=int, default=3); rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--output", type=Path, default=None)
    qu = sub.add_parser("queue", help="queue one phase-A job per run of a stage")
    qu.add_argument("--stage", required=True); qu.add_argument("--items", type=Path, required=True)
    qu.add_argument("--priority", type=int, default=50); qu.add_argument("--models", nargs="*", default=None)
    qu.add_argument("--seeds", type=int, nargs="*", default=None); qu.add_argument("--hosts", nargs="*", default=None)
    qu.add_argument("--root", type=Path, default=ROOT); qu.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = run_evaluate(args)
        print(json.dumps({"output": result["output"], "seconds": round(result["seconds"], 1), "timings": result["timings"],
                          "peak_gb": result["peak_gb"]}, indent=2))
    elif args.command == "recall":
        for record in run_recall(args):
            print(record.get("text", ""), "\n")
    else:
        jobs = queue_stage(args.stage, args.items, priority=args.priority, models=args.models, seeds=args.seeds, hosts=args.hosts,
                           root=args.root, dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
