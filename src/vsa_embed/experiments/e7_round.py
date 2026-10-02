"""E7.2 self-improvement round 0 → 1 and E7.3 cross-authoring (WP-E7; formulation §5.1, §5.4).

    python -m vsa_embed.experiments.e7_round <command> --run <E7 run folder> [--seed S] [--condition C]

Commands (each idempotent; `--resume` continues an interrupted training run):

- `base --seed S` — round 0: the consumer host (SmolLM2-360M, `host_mode` frozen unless G3 chose LoRA)
  with the best channel condition (`C5` until G3) trained on the general corpus with the *visible*
  ontology (curated WordNet minus the masked concepts). Entry space = C3's entries + one entry per
  authoring-set candidate (empty frames, never linked in round 0).
- `verify --seed S --author A` — held-out utility of A's edges (A = SmolLM2-360M for self → self, or
  `teacher`) with the round-0 channel of seed S: edges whose filler resolves to a dictionary atom, plus
  new atoms (M3 `allocate`) for unresolved fillers proposed for ≥ 2 candidates; greedy sequential
  acceptance with a one-sided 95% bootstrap bound > 0 on validation windows from documents disjoint
  from the authoring documents. Writes accepted frames, authoring cards and the compute ledger.
- `entigraph` — EntiGraph-style synthetic text: the writer host relates each candidate to another
  entity of a D_read document it occurs in (small, documented budget).
- `notes` / `verify-notes --seed S` — self-authored *unstructured* notes (Active Reading / SEAL-style,
  no RL): the consumer host writes a free-text note about each candidate from each authoring context;
  a note is kept when its held-out utility has a bootstrap lower bound > 0, measured in context on
  the same validation windows as the frames (the note replaces the most distant context tokens of the
  window; ≥ `verification.window` real tokens stay), with the round-0 model of seed S.
- `spa` — SPA / synthetic-QA study material: the consumer host writes question–answer pairs about D_read
  passages (self-generated, unverified).
- `train --seed S --condition C` — materializes C (ontology, linked train/test views, initial state =
  the round-0 channel, with new atoms appended) and trains on D_read + general replay:
  `gold` (masked concepts get their curated frames), `self` (verified self-authored frames), `selfnv`
  (self-consistency only), `teacher`, `random` (self's accepted entries, same degrees, random edges),
  `cm` (no new frames; round tokens + the FLOPs of discovery + authoring + verification as extra
  tokens), and the text controls — no new frames, D_read + replay + their synthetic text at a fixed
  share of the training windows, the same training tokens as `cm` (`text.budget: tokens`; with
  `compute`, each subtracts its own generation/verification FLOPs so total compute equals `self`'s):
  `entigraph`, `notes` (verified unstructured notes), `spa` (synthetic QA), `verbal` (the same verified
  edges as `self`, rendered as sentences — "X is a kind of Y." — so the knowledge reaches the model as
  text, not through the channel). Optional: `selfrand` (self's proposals accepted at random at the
  verified rate; what utility verification buys). With a frozen host the text controls can only
  train the channel's shared parameters; a LoRA host (`host_mode: lora`) lets them change the host.
  Every run of seed S evaluates the same test windows with the same reference strata
  (`eval.reference_strata`): masked concepts, authored-only concepts per author, unlinked text.
- `cross-prepare`, `cross-train --seed S --condition none|curated|authored` — E7.3: a from-scratch 50M
  GPT-2 model with the authored ontology (the base WordNet part plus the self-authored frames that
  replace the masked part) as its linker ontology, vs the curated ontology and no channel.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from vsa_embed.authoring import (
    ComputeLedger, ValidationWindow, card_from_record, compute_matched_tokens, excerpt, replace_frames, stable_seed,
    verify_frames,
)
from vsa_embed.compose import FrameSchedule
from vsa_embed.data.match_corpus import MatchCorpus, concatenate, select_longest, write_view
from vsa_embed.experiments.e7_authoring import (
    HOSTS, _default, _json, _merge, atom_surface, data_root, gold, kept_edges, load_host, load_track, occurrence_regex, parameter_count,
    read_texts, resolver_for, visible,
)

CONDITIONS = ("gold", "self", "selfnv", "teacher", "random", "cm", "entigraph", "notes", "spa", "verbal")
OPTIONAL_CONDITIONS = ("selfrand",)
TEXT_CONDITIONS = ("entigraph", "notes", "spa", "verbal")     # no new frames; synthetic text in the training mix
CROSS_CONDITIONS = ("none", "curated", "authored")
VERBAL_TEMPLATES = {
    "is_a": "{x} is a kind of {y}.", "instance_of": "{x} is an instance of {y}.", "has_part": "{x} has {y} as a part.",
    "part_of": "{x} is part of {y}.", "has_member": "{x} has {y} as a member.", "member_of": "{x} is a member of {y}.",
    "made_of": "{x} is made of {y}.", "substance_of": "{x} is a substance of {y}.", "attribute": "{x} is {y}.",
    "similar_to": "{x} is similar to {y}.", "domain": "{x} belongs to the field of {y}.", "entails": "{x} entails {y}.",
    "causes": "{x} causes {y}.", "opposite_of": "{x} is the opposite of {y}.",
    "kind": "{x} is a {y}.", "belongs_to": "{x} belongs to {y}.", "returns": "{x} returns {y}.", "takes": "{x} takes a {y}.",
    "raises": "{x} raises {y}.", "calls": "{x} calls {y}.", "inherits": "{x} inherits from {y}.", "category": "{x} is used for {y}.",
}
ROUND_DEFAULTS: dict[str, Any] = {
    "host": "SmolLM2-360M", "host_mode": "frozen", "channel_condition": "C5", "operator": "hrr", "key_dimension": 8,
    "channel_dimension": 256, "base_tokens": 50_000_000, "round_tokens": 25_000_000, "sequences_per_step": 128,
    "seq_len": 1024,
    "verification": {"window": 128, "after": 8, "min_validation": 4, "quantile": 0.05, "resamples": 1000, "batch": 32,
                     "max_edges": 8},
    "new_fillers": {"min_concepts": 2},
    "cm_stages": ["discovery", "authoring", "verification"],
    "teacher_verified": True,
    "teacher_condition": True,
    "entigraph": {"writer": "SmolLM2-360M", "per_entity": 4, "max_new_tokens": 192, "temperature": 0.7, "top_p": 0.95,
                  "batch": 16, "share": 0.1, "context_chars": 600},
    "notes": {"writer": None, "samples": 1, "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 64, "batch": 16,
              "share": 0.1},
    "spa": {"writer": None, "documents": 2000, "passage_chars": 1200, "max_new_tokens": 160, "temperature": 0.7,
            "top_p": 0.95, "batch": 16, "share": 0.1},
    "verbal": {"share": 0.02},
    "text": {"budget": "tokens"},          # tokens: same training tokens as cm; compute: minus own generation/verification FLOPs
    "eval": {"first_fraction": 0.5},
    "cross": {"seed": 1, "condition": "self", "train_tokens": 300_000_000, "run_tokens": None, "size": "50M", "channel_condition": "C5",
              "seeds": [1, 2, 3], "workers": 4},
}
AUTHOR = {"self": "consumer", "selfnv": "consumer", "teacher": "teacher", "random": "consumer", "selfrand": "consumer"}


def round_settings(run: Path) -> dict[str, Any]:
    """`round.json` in the run folder (written with the defaults on first use; change it only with
    `update_round_settings`, i.e. `--set` on the command line, so every step of a run sees the same values)."""
    path = Path(run) / "round.json"
    stored = json.loads(path.read_text()) if path.exists() else None
    settings = _merge(ROUND_DEFAULTS, stored or {})
    if stored is None:
        _json(path, settings)
    return settings


def update_round_settings(run: Path, values: dict[str, Any]) -> dict[str, Any]:
    """Persist changes to `round.json` (e.g. the teacher condition dropped by the plan)."""
    settings = _merge(round_settings(run), values)
    _json(Path(run) / "round.json", settings)
    return settings


def _slug(host: str) -> str:
    return host.split("-")[-1]


def run_dir(run: Path, settings: dict[str, Any], condition: str, seed: int) -> Path:
    return Path(run) / "train" / f"{settings['host']}-{condition}-s{seed}"


def round_root(track: dict[str, Any]) -> Path:
    return data_root(track) / "round1"


def _authoring_set(run: Path) -> dict[str, Any]:
    return json.loads((Path(run) / "authoring_set.json").read_text())


def author_name(track: dict[str, Any], condition: str) -> str:
    return track["consumer"] if AUTHOR[condition] == "consumer" else AUTHOR[condition]


# ---------------------------------------------------------------------------------------------
# linkers over the superset entry space (C3 entries + one per authoring-set candidate)

def string_map(strings: Sequence[str], *, base: dict[str, int], masked: dict[str, int] | None = None,
               candidates: dict[str, int] | None = None) -> np.ndarray:
    """Alias string id → entry (−1 inactive): base aliases, optionally masked aliases (gold), then
    candidate surfaces (which take precedence over a masked alias with the same string)."""
    out = np.full(len(strings), -1, dtype=np.int64)
    index = {s: i for i, s in enumerate(strings)}
    for table in (base, masked or {}, candidates or {}):
        for alias, entry in table.items():
            if alias in index:
                out[index[alias]] = int(entry)
    return out


def confidence_of(view: dict[str, Any], entry_count: int) -> np.ndarray:
    values = np.ones(entry_count, dtype=np.float32)
    known = np.asarray(view["entry_confidence"], dtype=np.float32)
    values[:known.size] = known
    return values


def base_ontology(run: Path) -> dict[str, Any]:
    """Round-0 ontology: visible frames for C3 entries, empty frames for the candidates."""
    track = load_track(run)
    root = round_root(track) / "base"
    path = root / "ontology.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    aset = _authoring_set(run)
    visible_ontology = torch.load(data_root(track) / "visible" / "ontology.pt", weights_only=False)
    count = len(aset["candidates"])
    offsets = torch.cat([visible_ontology["offsets"], visible_ontology["offsets"][-1:].repeat(count)])
    total = int(visible_ontology["entry_count"]) + count
    base_entries = set(visible_ontology["base_entries"])
    ontology = {"entry_count": total, "atomic_count": int(visible_ontology["atomic_count"]),
                "relation_count": int(visible_ontology["relation_count"]), "offsets": offsets,
                "relations": visible_ontology["relations"], "fillers": visible_ontology["fillers"],
                "heldout_entries": [e for e in range(total) if e not in base_entries],
                "relation_names": visible_ontology["relation_names"], "atomic_names": list(visible_ontology["atomic_names"]),
                "candidate_surfaces": [c["surface"] for c in aset["candidates"]], "c3_entries": int(visible_ontology["entry_count"])}
    root.mkdir(parents=True, exist_ok=True)
    view = visible(track)
    general = MatchCorpus.open(data_root(track) / "match" / "general")
    manifest = write_view(general, root / "general", string_map(general.strings, base=view["base"]),
                          confidence_of(view, total), extra_manifest={"condition": "base"})
    spans = np.load(root / "general" / "spans.npz")
    ontology["train_frequency"] = np.bincount(spans["entry"].astype(np.int64), minlength=total).tolist()
    test = MatchCorpus.open(data_root(track) / "match" / "test")
    write_view(test, root / "test", string_map(test.strings, base=view["base"]), confidence_of(view, total),
               extra_manifest={"condition": "base"})
    ontology["views"] = {"general": manifest}
    torch.save(ontology, path)
    write_reference(run, None)
    return ontology


# ---------------------------------------------------------------------------------------------
# reference strata (fixed target masks over the test windows, shared by every condition of a seed)

def test_windows(track: dict[str, Any], length: int = 1024) -> list[int]:
    from vsa_embed.data.corpus import eval_windows
    test = MatchCorpus.open(data_root(track) / "match" / "test")
    return eval_windows(test, count=max(1, (len(test) - length - 1) // length), length=length)


def _after_masks(starts: Sequence[int], spans: dict[str, np.ndarray], keep: np.ndarray, length: int, after: int = 8,
                 inside: bool = False) -> np.ndarray:
    """Targets after (or inside) the kept spans, per window, as in `training.lm.stratum_masks`."""
    mask = np.zeros((len(starts), length - 1), dtype=bool)
    inject, first = spans["inject"][keep], spans["start"][keep]
    for w, start in enumerate(starts):
        lo, hi = np.searchsorted(inject, start), np.searchsorted(inject, start + length)
        for s, e in zip(first[lo:hi] - start, inject[lo:hi] - start):
            if s < 0:
                continue
            if inside:
                if e > s:
                    mask[w, s:e] = True
            else:
                mask[w, e:min(length - 1, e + after)] = True
    return mask


def reference_masks(track: dict[str, Any], *, accepted: dict[str, dict[str, int]], length: int = 1024) -> tuple[list[int], dict[str, np.ndarray]]:
    """Strata over the test windows: `masked` (after masked gold concepts, gold linker), `new_all` (after
    any authoring-set candidate that is not a WordNet alias), per author `authored_new_<a>` /
    `authored_masked_<a>` (after its accepted candidates), `base_after`, and `unlinked` (outside and not
    after any span of the union linker)."""
    test = MatchCorpus.open(data_root(track) / "match" / "test")
    view, hidden = visible(track), gold(track)
    masked = {a: int(e) for a, e in hidden["masked_strings"].items()}
    starts = test_windows(track, length)
    entry_base = int(track["entry_count"])
    masks: dict[str, np.ndarray] = {}

    def spans(candidates: dict[str, int] | None, with_masked: bool) -> dict[str, np.ndarray]:
        return select_longest(test.matches, string_map(test.strings, base=view["base"], masked=masked if with_masked else None,
                                                       candidates=candidates))

    gold_spans = spans(None, True)
    masked_entries = set(masked.values())
    masks["masked"] = _after_masks(starts, gold_spans, np.isin(gold_spans["entry"], list(masked_entries)), length)
    base_spans = spans(None, False)
    masks["base_after"] = _after_masks(starts, base_spans, np.ones(base_spans["entry"].shape, bool), length)
    union: dict[str, int] = {}
    for name, table in accepted.items():
        union.update(table)
        found = spans(table, False)
        is_new = found["entry"] >= entry_base
        masked_alias = np.asarray([False] * found["entry"].size)
        if found["entry"].size:
            surfaces = {e: s for s, e in table.items()}
            masked_alias = np.asarray([surfaces.get(int(e)) in masked for e in found["entry"]])
        masks[f"authored_new_{name}"] = _after_masks(starts, found, is_new & ~masked_alias, length)
        masks[f"authored_masked_{name}"] = _after_masks(starts, found, is_new & masked_alias, length)
    all_candidates = accepted.get("all", {})
    if all_candidates:
        found = spans(all_candidates, False)
        surfaces = {e: s for s, e in all_candidates.items()}
        new = np.asarray([int(e) >= entry_base and surfaces.get(int(e)) not in masked for e in found["entry"]], dtype=bool)
        masks["new_all"] = _after_masks(starts, found, new, length)
    every = spans(union or None, True)
    linked = (_after_masks(starts, every, np.ones(every["entry"].shape, bool), length)
              | _after_masks(starts, every, np.ones(every["entry"].shape, bool), length, inside=True))
    masks["unlinked"] = ~linked
    return starts, masks


def write_reference(run: Path, seed: int | None) -> Path:
    """Reference strata of the base runs (seed None) or of one seed's conditions."""
    from vsa_embed.training.lm import save_reference_strata
    track = load_track(run)
    aset = _authoring_set(run)
    all_candidates = {c["surface"]: c["entry"] for c in aset["candidates"]}
    accepted: dict[str, dict[str, int]] = {"all": all_candidates}
    if seed is None:
        path = round_root(track) / "base" / "reference.npz"
    else:
        path = round_root(track) / f"s{seed}" / "reference.npz"
        if path.exists():
            return path
        settings = round_settings(run)
        for condition in ("self", "selfnv", "teacher"):
            if condition == "teacher" and not settings["teacher_condition"]:
                continue
            verified = _verification(run, seed, author_name(track, condition))
            if verified is None:
                raise FileNotFoundError(f"verify --seed {seed} --author {author_name(track, condition)} must run before "
                                        f"any round-1 training of seed {seed} (its reference strata need every author)")
            frames = verified["accepted"] if condition != "selfnv" else verified["noverify"]
            accepted[condition] = {aset["candidates"][int(k)]["surface"]: aset["candidates"][int(k)]["entry"]
                                   for k, frame in frames.items() if frame}
    if path.exists():
        return path
    length = int(round_settings(run)["seq_len"])
    starts, masks = reference_masks(track, accepted=accepted, length=length)
    masks.pop("authored_new_all", None); masks.pop("authored_masked_all", None)
    save_reference_strata(path, starts, masks, length)
    return path


# ---------------------------------------------------------------------------------------------
# training configs

def train_config(settings: dict[str, Any], *, condition: str, seed: int, ontology: Path, train: Path, test: Path,
                 reference: Path, total_tokens: int, init_from: Path | None, experiment: str) -> dict[str, Any]:
    from vsa_embed.experiments import cpt_plan
    from vsa_embed.experiments.e4_plan import conditions
    table = conditions(settings["operator"], int(settings["key_dimension"]), int(settings["channel_dimension"]), 0, 0, {})
    spec = table[settings["channel_condition"]]
    config = cpt_plan.run_config(stage="e7", host=settings["host"], mode=settings["host_mode"], label=condition, spec=spec,
                                 seed=seed, data_root=Path("/unused"), total_tokens=int(total_tokens),
                                 sequences_per_step=int(settings["sequences_per_step"]), save_window_losses=True,
                                 overrides=settings.get("config_overrides"))
    from vsa_embed.data.corpus import TokenCorpus
    config["model"]["seq_len"] = length = int(settings["seq_len"])
    count = max(1, (len(TokenCorpus.open(test)) - length - 1) // length)
    tokens_per_step = length * int(config["train"]["micro_batch"]) * int(config["train"]["grad_accum"])
    first = max(tokens_per_step, int(int(total_tokens) * float(settings["eval"]["first_fraction"])))
    config["data"].update(train=str(train), eval=str(test), ontology=str(ontology))
    config["eval"].update(windows=count, first_tokens=first, reference_strata=str(reference))
    config["train"]["warmup_tokens"] = min(int(config["train"]["warmup_tokens"]), max(tokens_per_step, int(total_tokens) // 10))
    if init_from is not None:
        config["train"]["init_from"] = str(init_from)
    config["experiment"] = experiment
    return config


def _train(config: dict[str, Any], out: Path, resume: bool) -> dict[str, Any]:
    from vsa_embed.training.lm import train
    if (out / "final.pt").exists() and (out / "manifest.json").exists():
        return {"skipped": "finished", "run": str(out)}
    return train(config, out, resume=resume and out.exists())


def train_base(run: Path, seed: int, *, resume: bool = False) -> dict[str, Any]:
    track = load_track(run)
    settings = round_settings(run)
    base_ontology(run)
    root = round_root(track) / "base"
    config = train_config(settings, condition="base", seed=seed, ontology=root / "ontology.pt", train=root / "general",
                          test=root / "test", reference=root / "reference.npz", total_tokens=int(settings["base_tokens"]),
                          init_from=None, experiment=f"e7-round0-{_slug(settings['host'])}-base-s{seed}")
    return _train(config, run_dir(run, settings, "base", seed), resume)


# ---------------------------------------------------------------------------------------------
# verification

def _verification(run: Path, seed: int, name: str) -> dict[str, Any] | None:
    path = Path(run) / "round1" / f"s{seed}" / f"verify-{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def proposal_ids(track: dict[str, Any], view: dict[str, Any], proposals: dict[str, Any], aset: dict[str, Any], *,
                 min_share: float, max_edges: int, new_min_concepts: int, atomic_count: int) -> tuple[dict[int, list[tuple[int, int]]], list[str], dict[str, int]]:
    """Per candidate index: deduplicated (relation id, atom id) edges, most votes first; new atoms for
    unresolved fillers proposed for ≥ `new_min_concepts` candidates (M3 allocation)."""
    resolver = resolver_for(track, view)
    relation_index = {name: i for i, name in enumerate(view["relation_names"])}
    relation_id = {label: relation_index[relation] for label, relation, _ in track["relations"] if relation in relation_index}
    threshold = min_share if proposals.get("kind") == "host" else 0.0
    kept: dict[int, list[tuple[str, str]]] = {}
    unresolved: Counter = Counter()
    for item in aset["candidates"]:
        record = proposals["candidates"].get(item["surface"])
        edges = kept_edges(record, min_share=threshold) if record else []
        kept[item["index"]] = [(r, f) for r, f in edges if r in relation_id]
        unresolved.update({f for _, f in kept[item["index"]] if resolver(f) is None})
    new_atoms = sorted(f for f, n in unresolved.items() if n >= new_min_concepts)
    new_index = {f: atomic_count + i for i, f in enumerate(new_atoms)}
    out: dict[int, list[tuple[int, int]]] = {}
    sources: dict[int, dict[tuple[int, int], tuple[str, str]]] = {}
    stats = {"edges": 0, "resolved": 0, "new_atom_edges": 0, "dropped_unresolved": 0}
    for k, edges in kept.items():
        frame: list[tuple[int, int]] = []
        sources[k] = {}
        for relation, filler in edges:
            stats["edges"] += 1
            atom = resolver(filler)
            if atom is None and filler in new_index:
                atom = new_index[filler]; stats["new_atom_edges"] += 1
            elif atom is None:
                stats["dropped_unresolved"] += 1; continue
            else:
                stats["resolved"] += 1
            edge = (relation_id[relation], int(atom))
            if edge not in frame and len(frame) < max_edges:
                frame.append(edge)
                sources[k][edge] = (relation, filler)
        out[k] = frame
    stats["sources"] = sources
    return out, new_atoms, stats


def validation_windows(track: dict[str, Any], aset: dict[str, Any], *, window: int, after: int) -> dict[int, list[ValidationWindow]]:
    """Windows of the validation occurrences (documents disjoint from the authoring documents), with the
    base linker's spans inside each window."""
    read = MatchCorpus.open(data_root(track) / "match" / "read")
    view = visible(track)
    spans = select_longest(read.matches, string_map(read.strings, base=view["base"]))
    confidence = confidence_of(view, int(track["entry_count"]))
    out: dict[int, list[ValidationWindow]] = {}
    for item in aset["candidates"]:
        windows = []
        authoring = set(item["authoring_documents"])
        for first, end in item["validation"]:
            stop, begin = end + after + 1, end + after + 1 - window
            if begin < 0 or stop > len(read) or first < begin:
                continue
            document = int(read.document_of(np.asarray([end]))[0])
            if document in authoring:
                raise AssertionError("a validation occurrence lies in an authoring document")
            lo, hi = np.searchsorted(spans["inject"], begin), np.searchsorted(spans["inject"], stop)
            keep = spans["start"][lo:hi] >= begin
            local = {key: spans[key][lo:hi][keep] - (begin if key in ("start", "end", "inject") else 0)
                     for key in ("start", "end", "inject", "entry", "length")}
            local["confidence"] = confidence[local["entry"]] if local["entry"].size else np.zeros(0, np.float32)
            windows.append(ValidationWindow(np.asarray(read.tokens[begin:stop], dtype=np.int64), local, item["index"],
                                            first - begin, end - begin, document))
        out[item["index"]] = windows
    return out


def verify(run: Path, seed: int, name: str, *, device: torch.device) -> dict[str, Any]:
    """Held-out utility of author `name`'s edges with the round-0 channel of `seed` (see module docstring)."""
    from vsa_embed.training.lm import load_final
    track = load_track(run)
    out = Path(run) / "round1" / f"s{seed}" / f"verify-{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = round_settings(run)
    checks = settings["verification"]
    aset = _authoring_set(run)
    view = visible(track)
    proposals = json.loads((Path(run) / "proposals" / f"{name}.json").read_text())
    final = run_dir(run, settings, "base", seed) / "final.pt"
    lm = load_final(final, device)
    composer = lm.channel.composer
    atomic_count = int(composer.atomics.shape[0])
    frames, new_atoms, stats = proposal_ids(track, view, proposals, aset, min_share=float(track["settings"]["authoring"]["min_share"]),
                                            max_edges=int(checks["max_edges"]),
                                            new_min_concepts=int(settings["new_fillers"]["min_concepts"]), atomic_count=atomic_count)
    sources = stats.pop("sources")
    vectors = torch.stack([torch.randn(composer.atomics.shape[1], generator=torch.Generator().manual_seed(stable_seed(seed, name, f)))
                           / math.sqrt(composer.atomics.shape[1]) for f in new_atoms]) if new_atoms else torch.zeros(0, composer.atomics.shape[1])
    if new_atoms:
        composer.add_atomics(vectors.to(composer.atomics.device))
    windows = validation_windows(track, aset, window=int(checks["window"]), after=int(checks["after"]))
    entry_of = {item["index"]: item["entry"] for item in aset["candidates"]}
    started = time.monotonic()
    records, forwarded = verify_frames(lm, windows, frames, entry_of, device, min_validation=int(checks["min_validation"]),
                                       quantile=float(checks["quantile"]), resamples=int(checks["resamples"]),
                                       seed=stable_seed(seed, name) % 2**31, after=int(checks["after"]), batch=int(checks["batch"]))
    ledger = ComputeLedger()
    ledger.add("verification", parameters=parameter_count(lm.model), forward_tokens=forwarded, seconds=time.monotonic() - started,
               note=f"{name} s{seed}")
    accepted = {str(k): [[r["relation"], r["atom"]] for r in rs if r["accepted"]] for k, rs in records.items()}
    atom_names = list(view["atomic_names"]) + [f"authored:{f}" for f in new_atoms]
    cards = []
    for item in aset["candidates"]:
        record = proposals["candidates"].get(item["surface"], {})
        votes = {(r, f): n for r, f, n in record.get("votes", [])}
        for r in records.get(item["index"], []):
            label, filler = sources[item["index"]][(r["relation"], r["atom"])]
            card = card_from_record(item["surface"], label, filler, r, proposals=int(votes.get((label, filler), 0)),
                                    samples=int(record.get("samples", 0)), round_index=1, contexts=item["authoring_documents"])
            cards.append({**card, "atom_name": atom_surface(atom_names[r["atom"]])})
    root = Path(run) / "round1" / f"s{seed}"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"cards-{name}.jsonl").write_text("".join(json.dumps(c, default=_default) + "\n" for c in cards))
    torch.save({"names": new_atoms, "vectors": vectors.cpu()}, root / f"new_atoms-{name}.pt")
    utilities = [r["mean"] for rs in records.values() for r in rs if r["accepted"]]
    result = {"author": name, "seed": seed, "settings": checks, "stats": stats, "new_atoms": new_atoms,
              "atomic_count": atomic_count, "accepted": accepted,
              "noverify": {str(k): [list(e) for e in v] for k, v in frames.items()},
              "records": {str(k): v for k, v in records.items()}, "ledger": ledger.to_json(),
              "summary": {"candidates": len(frames), "proposed_edges": sum(len(v) for v in frames.values()),
                          "accepted_edges": sum(len(v) for v in accepted.values()),
                          "concepts_with_accepted_edges": sum(bool(v) for v in accepted.values()),
                          "verifiable_candidates": sum(len(windows[k]) >= int(checks["min_validation"]) for k in windows),
                          "median_accepted_utility": float(np.median(utilities)) if utilities else None}}
    _json(out, result)
    return result


# ---------------------------------------------------------------------------------------------
# Synthetic-text controls: EntiGraph-style, verified notes, SPA / synthetic QA, verbalized edges

def text_root(track: dict[str, Any], name: str) -> Path:
    return round_root(track) / "text" / name


def _writer(track: dict[str, Any], section: dict[str, Any]) -> str:
    return section.get("writer") or track["consumer"]


def _build_synthetic(track: dict[str, Any], name: str, texts: Sequence[str]) -> dict[str, Any]:
    """Tokenize and scan synthetic texts with the reading corpus's alias strings (so they concatenate)."""
    from transformers import AutoTokenizer
    from vsa_embed.data.match_corpus import build_match_corpus
    read = MatchCorpus.open(data_root(track) / "match" / "read")
    tokenizer = AutoTokenizer.from_pretrained(read.manifest["tokenizer"], local_files_only=True)
    if not texts:
        raise ValueError(f"no synthetic text for {name}")
    return build_match_corpus(list(texts), text_root(track, name) / "synthetic", tokenizer_name=read.manifest["tokenizer"],
                              strings=read.strings, eos_id=int(read.manifest["eos_id"]), max_tokens=10**12,
                              min_subtokens=int(read.manifest["min_subtokens"]), vocab_size=len(tokenizer),
                              workers=int(track["settings"]["corpora"]["workers"]), keep_texts=True,
                              extra_manifest={"synthetic": name})


def entigraph(run: Path, *, device: torch.device, model: Any = None,
              tokenizer: Any = None) -> dict[str, Any]:
    from vsa_embed.authoring_baselines import entigraph_prompt
    from vsa_embed.experiments.e7_authoring import generate_samples
    track = load_track(run)
    out = Path(run) / "round1" / "entigraph.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = round_settings(run)["entigraph"]
    aset = _authoring_set(run)
    read = MatchCorpus.open(data_root(track) / "match" / "read")
    texts = read_texts(track)
    by_document: dict[int, set[str]] = defaultdict(set)
    string_of = {c["string"]: c["surface"] for c in aset["candidates"]}
    rows = np.isin(read.matches["string"], list(string_of))
    for s, document in zip(read.matches["string"][rows].tolist(), read.document_of(read.matches["token"][rows]).tolist()):
        by_document[int(document)].add(string_of[int(s)])
    rng = random.Random(stable_seed(track["seed"], "entigraph"))
    used: Counter = Counter()
    prompts, pairs = [], []
    for document in sorted(by_document):
        names = sorted(by_document[document])
        for name in names:
            if used[name] >= int(settings["per_entity"]):
                continue
            others = [o for o in names if o != name]
            other = rng.choice(others) if others else None
            match = occurrence_regex(name).search(texts[document])
            if match is None:
                continue
            if other is None:
                words = [w for w in texts[document][max(0, match.start() - 300):match.end() + 300].split() if len(w) > 6 and w.isalpha()]
                if not words:
                    continue
                other = rng.choice(words).lower()
            context = excerpt(texts[document], match.start(), match.end(), chars=int(settings["context_chars"]))
            prompts.append(entigraph_prompt(name, other, context))
            pairs.append((name, other, document))
            used[name] += 1
    writer = _writer(track, settings)
    if model is None:
        model, tokenizer = load_host(writer, device)
    started = time.monotonic()
    completions, prompt_tokens, generated = generate_samples(
        model, tokenizer, prompts, device, samples=1, temperature=float(settings["temperature"]), top_p=float(settings["top_p"]),
        max_new_tokens=int(settings["max_new_tokens"]), batch=int(settings["batch"]), seed=int(track["seed"]))
    synthetic = [" ".join(c[0].split("\n\n")[0].split()) for c in completions]
    synthetic = [t for t in synthetic if len(t.split()) >= 8]
    ledger = ComputeLedger()
    ledger.add("entigraph", parameters=parameter_count(model), prompt_tokens=prompt_tokens, generated_tokens=generated,
               seconds=time.monotonic() - started, note=writer)
    manifest = _build_synthetic(track, "entigraph", synthetic)
    result = {"writer": writer, "settings": settings, "pairs": len(pairs), "documents": len(synthetic),
              "tokens": manifest["tokens"], "ledger": ledger.to_json(),
              "examples": [{"pair": pairs[i][:2], "text": synthetic[i]} for i in range(min(5, len(synthetic)))]}
    _json(out, result)
    return result


def note_prompt(surface: str, context: str, demonstrations: Sequence[dict[str, Any]]) -> str:
    """Few-shot prompt for an unstructured note about one concept (the host continues after the colon)."""
    blocks = ["Each example quotes a text and gives short notes that explain a concept from it."]
    for demo in demonstrations:
        if demo.get("note"):
            blocks.append(f"Text: {' '.join(demo['context'].split())}\nNotes on \"{demo['surface']}\": {demo['note']}")
    blocks.append(f"Text: {' '.join(context.split())}\nNotes on \"{surface}\":")
    return "\n\n".join(blocks)


def notes(run: Path, *, device: torch.device, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    """Self-authored unstructured notes: one prompt per authoring context, `samples` notes each."""
    from vsa_embed.experiments.e7_authoring import authoring_contexts, generate_samples
    track = load_track(run)
    out = Path(run) / "round1" / "notes.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = round_settings(run)["notes"]
    aset = _authoring_set(run)
    contexts = authoring_contexts(track, aset, read_texts(track))
    surfaces = {item["index"]: item["surface"] for item in aset["candidates"]}
    prompts, owners = [], []
    for k, items in contexts.items():
        for document, context in items:
            prompts.append(note_prompt(surfaces[k], context, track["demonstrations"]))
            owners.append((k, document))
    writer = _writer(track, settings)
    if model is None:
        model, tokenizer = load_host(writer, device)
    started = time.monotonic()
    completions, prompt_tokens, generated = generate_samples(
        model, tokenizer, prompts, device, samples=int(settings["samples"]), temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]), max_new_tokens=int(settings["max_new_tokens"]), batch=int(settings["batch"]),
        seed=stable_seed(track["seed"], "notes") % 2**31)
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (k, document), samples in zip(owners, completions):
        for sample in samples:
            text = " ".join(sample.split("\n\n")[0].split("\nText:")[0].split())
            if len(text.split()) >= 5 and text not in {n["text"] for n in by_candidate[k]}:
                by_candidate[k].append({"text": text, "document": document})
    ledger = ComputeLedger()
    ledger.add("notes", parameters=parameter_count(model), prompt_tokens=prompt_tokens, generated_tokens=generated,
               seconds=time.monotonic() - started, note=writer)
    result = {"writer": writer, "settings": settings, "ledger": ledger.to_json(),
              "notes": {str(k): v for k, v in sorted(by_candidate.items())},
              "count": sum(len(v) for v in by_candidate.values())}
    _json(out, result)
    return result


def verify_notes(run: Path, seed: int, *, device: torch.device) -> dict[str, Any]:
    """Keep notes whose in-context held-out utility has a bootstrap lower bound > 0 (round-0 model of `seed`)."""
    from transformers import AutoTokenizer
    from vsa_embed.authoring import bootstrap_lower, window_losses
    from vsa_embed.training.lm import load_final
    track = load_track(run)
    out = Path(run) / "round1" / f"s{seed}" / "verify-notes.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = round_settings(run)
    checks = settings["verification"]
    written = json.loads((Path(run) / "round1" / "notes.json").read_text())
    aset = _authoring_set(run)
    room = int(settings["notes"]["max_new_tokens"]) + 8
    windows = validation_windows(track, aset, window=int(checks["window"]) + room, after=int(checks["after"]))
    tokenizer = AutoTokenizer.from_pretrained(track["tokenizer"], local_files_only=True)
    lm = load_final(run_dir(run, settings, "base", seed) / "final.pt", device)
    flat_plain = [(k, w) for k, ws in windows.items() if len(ws) >= int(checks["min_validation"]) for w in ws]
    plain, forwarded = window_losses(lm, [w for _, w in flat_plain], [None] * len(flat_plain), device,
                                     after=int(checks["after"]), batch=int(checks["batch"]))
    without: dict[int, list[float]] = defaultdict(list)
    for (k, _), value in zip(flat_plain, plain):
        without[k].append(float(value))
    trials, owners = [], []
    for key, items in written["notes"].items():
        k = int(key)
        if k not in without:
            continue
        for n, note in enumerate(items):
            ids = np.asarray(tokenizer(note["text"] + "\n\n", add_special_tokens=False)["input_ids"][:room], dtype=np.int64)
            for w in windows[k]:
                # the note replaces the most distant context tokens; spans that started there are dropped
                keep = w.spans["start"] >= ids.size
                spans = {name: value[keep] for name, value in w.spans.items()}
                trials.append(ValidationWindow(np.concatenate([ids, w.ids[ids.size:]]), spans, k, w.start, w.end, w.document))
            owners.append((k, n, len(windows[k])))
    with_note, more = window_losses(lm, trials, [None] * len(trials), device, after=int(checks["after"]), batch=int(checks["batch"]))
    forwarded += more
    kept, records, position = [], [], 0
    for k, n, count in owners:
        deltas = np.asarray(without[k]) - with_note[position:position + count]
        position += count
        low = bootstrap_lower(deltas.tolist(), resamples=int(checks["resamples"]), seed=stable_seed(seed, "notes", k, n) % 2**31,
                              quantile=float(checks["quantile"]))
        accepted = bool(low > 0)
        records.append({"candidate": k, "note": n, "mean": float(deltas.mean()), "low": float(low), "n": int(deltas.size),
                        "accepted": accepted})
        if accepted:
            kept.append(written["notes"][str(k)][n]["text"])
    ledger = ComputeLedger()
    ledger.add("verification", parameters=parameter_count(lm.model), forward_tokens=forwarded, note=f"notes s{seed}")
    manifest = _build_synthetic(track, f"notes-s{seed}", kept) if kept else None
    utilities = [r["mean"] for r in records if r["accepted"]]
    result = {"seed": seed, "records": records, "kept": len(kept), "notes": len(records), "ledger": ledger.to_json(),
              "tokens": manifest["tokens"] if manifest else 0,
              "median_accepted_utility": float(np.median(utilities)) if utilities else None,
              "examples": kept[:5]}
    _json(out, result)
    return result


def spa(run: Path, *, device: torch.device, model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    """SPA / synthetic-QA study material: the consumer host writes question–answer pairs about D_read passages."""
    from vsa_embed.experiments.e7_authoring import QA_DEMO, generate_samples
    track = load_track(run)
    out = Path(run) / "round1" / "spa.json"
    if out.exists():
        return json.loads(out.read_text())
    settings = round_settings(run)["spa"]
    texts = read_texts(track)
    rng = random.Random(stable_seed(track["seed"], "spa"))
    documents = sorted(rng.sample(range(len(texts)), k=min(int(settings["documents"]), len(texts))))
    demo = f"Text: {QA_DEMO['context']}\nQuestions and answers about this text:\n{QA_DEMO['qa']}"
    prompts = [f"{demo}\n\nText: {excerpt(texts[d], 0, 0, chars=2 * int(settings['passage_chars']))}\n"
               f"Questions and answers about this text:\nQ:" for d in documents]
    writer = _writer(track, settings)
    if model is None:
        model, tokenizer = load_host(writer, device)
    started = time.monotonic()
    completions, prompt_tokens, generated = generate_samples(
        model, tokenizer, prompts, device, samples=1, temperature=float(settings["temperature"]), top_p=float(settings["top_p"]),
        max_new_tokens=int(settings["max_new_tokens"]), batch=int(settings["batch"]), seed=stable_seed(track["seed"], "spa") % 2**31)
    synthetic = []
    for (completion,) in completions:
        text = "Q:" + completion.split("\n\n")[0].split("\nText:")[0]
        if "A:" in text:
            synthetic.append(text.strip())
    ledger = ComputeLedger()
    ledger.add("spa", parameters=parameter_count(model), prompt_tokens=prompt_tokens, generated_tokens=generated,
               seconds=time.monotonic() - started, note=writer)
    manifest = _build_synthetic(track, "spa", synthetic)
    result = {"writer": writer, "settings": settings, "documents": len(documents), "texts": len(synthetic),
              "tokens": manifest["tokens"], "ledger": ledger.to_json(), "examples": synthetic[:3]}
    _json(out, result)
    return result


def verbal(run: Path, seed: int) -> dict[str, Any]:
    """The verified edges of `self` at seed S rendered as sentences (one document per concept)."""
    track = load_track(run)
    out = Path(run) / "round1" / f"s{seed}" / "verbal.json"
    if out.exists():
        return json.loads(out.read_text())
    cards = [json.loads(line) for line in (Path(run) / "round1" / f"s{seed}" / f"cards-{track['consumer']}.jsonl").read_text().splitlines()
             if line.strip()]
    by_concept: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        if card["accepted"]:
            template = VERBAL_TEMPLATES.get(card["relation"], "{x} " + card["relation"].replace("_", " ") + " {y}.")
            sentence = template.format(x=card["surface"], y=card["filler"])
            by_concept[card["surface"]].append(sentence[0].upper() + sentence[1:])
    texts = [" ".join(sentences) for _, sentences in sorted(by_concept.items())]
    manifest = _build_synthetic(track, f"verbal-s{seed}", texts) if texts else None
    result = {"seed": seed, "concepts": len(texts), "sentences": sum(len(s) for s in by_concept.values()),
              "tokens": manifest["tokens"] if manifest else 0, "examples": texts[:5]}
    _json(out, result)
    return result


def synthetic_name(condition: str, seed: int) -> str:
    return {"entigraph": "entigraph", "spa": "spa", "notes": f"notes-s{seed}", "verbal": f"verbal-s{seed}"}[condition]


# ---------------------------------------------------------------------------------------------
# conditions

def _mix(track: dict[str, Any], *, synthetic: str | None = None, share: float | None = None) -> MatchCorpus:
    """D_read + general replay (+ a synthetic-text corpus repeated to `share` of the tokens) as one match corpus."""
    read = MatchCorpus.open(data_root(track) / "match" / "read")
    replay = MatchCorpus.open(data_root(track) / "match" / "replay")
    if synthetic is None:
        path = round_root(track) / "mix"
        if not (path / "manifest.json").exists():
            concatenate([(read, read.documents.size, 1), (replay, replay.documents.size, 1)], path)
        return MatchCorpus.open(path)
    path = text_root(track, synthetic) / "mix"
    if not (path / "manifest.json").exists():
        corpus = MatchCorpus.open(text_root(track, synthetic) / "synthetic")
        base = len(read) + len(replay)
        repeats = max(1, round(share * base / ((1 - share) * max(1, len(corpus)))))
        concatenate([(read, read.documents.size, 1), (replay, replay.documents.size, 1),
                     (corpus, corpus.documents.size, repeats)], path,
                     extra_manifest={"synthetic": synthetic, "synthetic_share": share, "synthetic_repeats": repeats})
    return MatchCorpus.open(path)


def _ledger_of(path: Path) -> ComputeLedger:
    return ComputeLedger.from_json(json.loads(path.read_text())["ledger"]) if path.exists() else ComputeLedger()


def compute_matched(run: Path, seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    """Tokens of the compute-matched control for seed S: round tokens + (discovery + authoring of the
    consumer host + its verification at S) FLOPs / training FLOPs per token."""
    track = load_track(run)
    consumer = track["consumer"]
    ledger = ComputeLedger()
    ledger.extend(_ledger_of(Path(run) / "discovery" / f"{consumer}.json"))
    ledger.extend(_ledger_of(Path(run) / "proposals" / f"{consumer}.json"))
    verified = _verification(run, seed, consumer)
    if verified is None:
        raise FileNotFoundError(f"verify --seed {seed} --author {consumer} must run before the compute-matched control")
    ledger.extend(ComputeLedger.from_json(verified["ledger"]))
    parameters = verified["ledger"][0]["parameters"]            # the round host (it ran the verification)
    trainable = 0.0
    if settings["host_mode"] == "lora":
        state = torch.load(run_dir(run, settings, "base", seed) / "final.pt", weights_only=False, map_location="cpu")["model"]
        trainable = float(sum(v.numel() for k, v in state.items() if "lora_" in k))
    result = compute_matched_tokens(int(settings["round_tokens"]), ledger, parameters=parameters, host_mode=settings["host_mode"],
                                    stages=settings["cm_stages"], trainable=trainable)
    result["ledger"] = ledger.to_json()
    result["parameters"], result["trainable"] = parameters, trainable
    return result


def text_budget(run: Path, seed: int, condition: str, settings: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    """Training tokens of a synthetic-text control and the compute it spent outside training.

    `text.budget: tokens` (default): the compute-matched control's tokens. `compute`: those minus the
    condition's own generation/verification FLOPs (and the discovery/authoring it reuses), so that its
    total compute equals `self`'s (`verbal` reuses the whole self pipeline and gets the round tokens)."""
    track = load_track(run)
    consumer = track["consumer"]
    side = ComputeLedger()
    if condition in ("entigraph", "notes"):
        side.extend(_ledger_of(Path(run) / "discovery" / f"{consumer}.json"))
    if condition == "entigraph":
        side.extend(_ledger_of(Path(run) / "round1" / "entigraph.json"))
    elif condition == "notes":
        side.extend(_ledger_of(Path(run) / "round1" / "notes.json"))
        side.extend(_ledger_of(Path(run) / "round1" / f"s{seed}" / "verify-notes.json"))
    elif condition == "spa":
        side.extend(_ledger_of(Path(run) / "round1" / "spa.json"))
    elif condition == "verbal":
        side = ComputeLedger.from_json(matched["ledger"])
    per_token = matched["flops_per_training_token"]
    if settings["text"]["budget"] == "compute":
        total = max(int(settings["round_tokens"]) // 10, int(matched["total_tokens"] - math.ceil(side.flops() / per_token)))
    else:
        total = int(matched["total_tokens"])
    return {"budget": settings["text"]["budget"], "total_tokens": total, "side_flops": side.flops(),
            "training_flops": total * per_token, "total_flops": side.flops() + total * per_token, "ledger": side.to_json()}


def materialize(run: Path, seed: int, condition: str) -> dict[str, Any]:
    """Ontology, linked views and initial state of one (condition, seed)."""
    track = load_track(run)
    root = round_root(track) / f"s{seed}" / condition
    done = root / "materialized.json"
    if done.exists():
        return json.loads(done.read_text())
    settings = round_settings(run)
    aset = _authoring_set(run)
    view = visible(track)
    base = base_ontology(run)
    final = torch.load(run_dir(run, settings, "base", seed) / "final.pt", weights_only=False, map_location="cpu")
    saved = final["composer_schedule"]
    schedule = FrameSchedule(saved["offsets"], saved["relations"], saved["fillers"])
    atomic_count, relation_count = int(saved["atomics"]), int(saved["relations_count"])
    surfaces = {item["index"]: item["surface"] for item in aset["candidates"]}
    entry_of = {item["index"]: item["entry"] for item in aset["candidates"]}
    frames: dict[int, list[tuple[int, int]]] = {}
    candidates: dict[str, int] = {}
    masked = None
    new_names: list[str] = []
    new_vectors = torch.zeros(0, final["model"]["channel.composer.atomics"].shape[1])
    info: dict[str, Any] = {"condition": condition, "seed": seed}
    synthetic = None
    if condition == "gold":
        hidden = gold(track)
        masked = {a: int(e) for a, e in hidden["masked_strings"].items()}
        frames = {int(e): [tuple(edge) for edge in data["frame_ids"]] for e, data in hidden["entries"].items()}
    elif condition in ("self", "selfnv", "teacher", "random", "selfrand"):
        name = author_name(track, condition)
        if condition == "teacher" and not settings["teacher_verified"]:
            raise ValueError("unverified teacher frames are not a planned condition")
        verified = _verification(run, seed, name)
        if verified is None:
            raise FileNotFoundError(f"verify --seed {seed} --author {name} must run before {condition}")
        chosen = verified["noverify"] if condition in ("selfnv", "selfrand") else verified["accepted"]
        if condition == "random":
            relation_pool = [r for frame in chosen.values() for r, _ in frame]
            atoms = sorted({a for atoms in view["lexicon"].values() for a in atoms})
            rng = random.Random(stable_seed(seed, "random-frames"))
            randomized = {}
            for k, frame in sorted(chosen.items()):
                edges: list[tuple[int, int]] = []
                for _ in range(100 * max(1, len(frame))):
                    if len(edges) == len(frame):
                        break
                    edge = (rng.choice(relation_pool), rng.choice(atoms))
                    if edge not in edges:
                        edges.append(edge)
                randomized[k] = edges
            chosen = randomized
            info["random"] = {"relations_from": "self-accepted edges of this seed", "fillers": "uniform over dictionary synsets",
                              "degrees": "equal to the self-accepted frames"}
        if condition == "selfrand":
            # the proposals of `selfnv`, accepted at random at the verified acceptance count
            pool = [(k, tuple(edge)) for k, frame in sorted(chosen.items()) for edge in frame]
            target = sum(len(frame) for frame in verified["accepted"].values())
            picked = set(random.Random(stable_seed(seed, "selfrand")).sample(range(len(pool)), k=min(target, len(pool))))
            subset: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for i, (k, edge) in enumerate(pool):
                if i in picked:
                    subset[k].append(edge)
            chosen = dict(subset)
            info["selfrand"] = {"accepted_edges": target, "pool": len(pool)}
        if condition != "random":
            atoms_file = torch.load(Path(run) / "round1" / f"s{seed}" / f"new_atoms-{name}.pt", weights_only=False)
            new_names, new_vectors = list(atoms_file["names"]), atoms_file["vectors"]
            if int(verified["atomic_count"]) != atomic_count:
                raise ValueError("the verification ran on another dictionary size than the round-0 checkpoint")
        for k, frame in chosen.items():
            frame = [tuple(int(x) for x in edge) for edge in frame]
            if frame:
                frames[entry_of[int(k)]] = list(dict.fromkeys(frame))
                candidates[surfaces[int(k)]] = entry_of[int(k)]
        info["author"] = name
    elif condition in TEXT_CONDITIONS:
        synthetic = synthetic_name(condition, seed)
        if condition == "verbal":
            verbal(run, seed)
        if not (text_root(track, synthetic) / "synthetic" / "manifest.json").exists():
            step = {"entigraph": "entigraph", "spa": "spa", "notes": f"notes, then verify-notes --seed {seed}",
                    "verbal": f"verify --seed {seed} (self has no accepted edges?)"}[condition]
            raise FileNotFoundError(f"{condition} needs its synthetic text first: run `{step}`")
    elif condition != "cm":
        raise ValueError(f"unknown condition {condition!r}")
    schedule = replace_frames(schedule, frames)
    total = int(base["entry_count"])
    active = set(view["base"].values()) | set(frames)
    share = float(settings[condition]["share"]) if synthetic else None
    mix = _mix(track, synthetic=synthetic, share=share)
    test = MatchCorpus.open(data_root(track) / "match" / "test")
    confidence = confidence_of(view, total)
    root.mkdir(parents=True, exist_ok=True)
    write_view(mix, root / "train", string_map(mix.strings, base=view["base"], masked=masked, candidates=candidates), confidence,
               extra_manifest={"condition": condition, "seed": seed})
    write_view(test, root / "test", string_map(test.strings, base=view["base"], masked=masked, candidates=candidates), confidence,
               extra_manifest={"condition": condition, "seed": seed})
    spans = np.load(root / "train" / "spans.npz")
    ontology = {**{k: v for k, v in base.items() if k not in ("views",)},
                "atomic_count": atomic_count + len(new_names), "relation_count": relation_count,
                "offsets": schedule.offsets, "relations": schedule.relations, "fillers": schedule.fillers,
                "heldout_entries": [e for e in range(total) if e not in active],
                "train_frequency": np.bincount(spans["entry"].astype(np.int64), minlength=total).tolist(),
                "condition": condition, "seed": seed}
    names = list(base["atomic_names"])[:atomic_count]
    names += [f"grown:{i}" for i in range(len(names), atomic_count)]        # rows added by M3 splits in round 0
    ontology["atomic_names"] = names + [f"authored:{n}" for n in new_names]
    torch.save(ontology, root / "ontology.pt")
    state = dict(final["model"])
    if new_names:
        state["channel.composer.atomics"] = torch.cat([state["channel.composer.atomics"], new_vectors.to(state["channel.composer.atomics"])])
    # the frame table is a set of buffers: the initial state carries this condition's frames
    state.update({"channel.composer.frame_offsets": schedule.offsets.clone(), "channel.composer.frame_relations": schedule.relations.clone(),
                  "channel.composer.frame_fillers": schedule.fillers.clone()})
    torch.save({"model": state, "trainable_only": bool(final.get("trainable_only", False)), "source": str(run_dir(run, settings, "base", seed))},
               root / "init_state.pt")
    info.update(entries_with_new_frames=len(frames), linked_candidates=len(candidates), new_atoms=len(new_names),
                train_spans=int(spans["entry"].size), train_tokens_available=len(mix), synthetic=synthetic,
                synthetic_share=share, mix=mix.manifest.get("sources"))
    if condition == "cm" or synthetic:
        info["compute_matched"] = compute_matched(run, seed, settings)
    if synthetic:
        info["text_budget"] = text_budget(run, seed, condition, settings, info["compute_matched"])
    write_reference(run, seed)
    _json(done, info)
    return info


def training_tokens(info: dict[str, Any], settings: dict[str, Any]) -> int:
    if "text_budget" in info:
        return int(info["text_budget"]["total_tokens"])
    if info["condition"] == "cm":
        return int(info["compute_matched"]["total_tokens"])
    return int(settings["round_tokens"])


def train_condition(run: Path, seed: int, condition: str, *, resume: bool = False) -> dict[str, Any]:
    track = load_track(run)
    settings = round_settings(run)
    info = materialize(run, seed, condition)
    root = round_root(track) / f"s{seed}" / condition
    config = train_config(settings, condition=condition, seed=seed, ontology=root / "ontology.pt", train=root / "train",
                          test=root / "test", reference=round_root(track) / f"s{seed}" / "reference.npz",
                          total_tokens=training_tokens(info, settings), init_from=root / "init_state.pt",
                          experiment=f"e7-round1-{_slug(settings['host'])}-{condition}-s{seed}")
    return _train(config, run_dir(run, settings, condition, seed), resume)


# ---------------------------------------------------------------------------------------------
# E7.3 cross-authoring

def cross_root(track: dict[str, Any]) -> Path:
    return data_root(track) / "cross"


def cross_prepare(run: Path) -> dict[str, Any]:
    """Authored ontology → GPT-2 linker ontology; curated and authored GPT-2 corpora over the same documents."""
    from transformers import AutoTokenizer
    from vsa_embed.data.corpus import TokenCorpus, build_corpus
    from vsa_embed.experiments import host_corpus
    from vsa_embed.experiments.c3_corpus import iter_texts
    from vsa_embed.span_channel import AliasTable
    track = load_track(run)
    settings = round_settings(run)["cross"]
    root = cross_root(track)
    done = root / "prepared.json"
    if done.exists():
        return json.loads(done.read_text())
    seed, condition = int(settings["seed"]), settings["condition"]
    source = round_root(track) / f"s{seed}" / condition
    authored = torch.load(source / "ontology.pt", weights_only=False)
    record = host_corpus.load_c3_record(Path(track["c3_run"]))
    c3 = record["config"]
    tables = host_corpus.rebuild_c3_tables(c3, record["holdout_names"])
    curated_table = tables["train_table"]
    view = visible(track)
    aset = _authoring_set(run)
    active = set(range(int(authored["entry_count"]))) - set(authored["heldout_entries"])
    concept_count = len(tables["ontology"].concept_names)
    alias_to_entry = {a: e for a, e in view["base"].items()}
    alias_to_entry.update({c["surface"]: c["entry"] for c in aset["candidates"] if c["entry"] in active})
    entry_concepts = [tuple(c) for c in tables["full"].entry_concepts] + [(concept_count + c["index"],) for c in aset["candidates"]]
    authored_table = AliasTable(alias_to_entry, entry_concepts)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    shards = [str(Path(p).expanduser()) for p in c3["paths"]["shards"]]
    eval_docs = int(c3["data"]["eval_docs"])
    test_texts = [json.loads(line)["text"] for line in (data_root(track) / "match" / "test" / "texts.jsonl").read_text().splitlines()]
    manifests = {}
    for name, table in (("curated", curated_table), ("authored", authored_table)):
        manifests[name] = {
            "train": build_corpus(iter_texts(shards, skip=eval_docs), root / name / "train", tokenizer_name="gpt2", table=table,
                                  eos_id=tokenizer.eos_token_id, max_tokens=int(settings["train_tokens"]), workers=int(settings["workers"]),
                                  min_subtokens=2, reuse=True, extra_manifest={"e7_cross": name}),
            "eval": build_corpus(test_texts, root / name / "eval", tokenizer_name="gpt2", table=table, eos_id=tokenizer.eos_token_id,
                                 max_tokens=10**12, workers=int(settings["workers"]), min_subtokens=2, reuse=True,
                                 extra_manifest={"e7_cross": name, "documents": "E7 test"})}
    curated_ontology = torch.load(Path(c3["paths"]["data_root"]).expanduser() / "ontology.pt", weights_only=False)
    for name, ontology in (("curated", dict(curated_ontology)), ("authored", dict(authored))):
        spans = TokenCorpus.open(root / name / "train").spans
        ontology["train_frequency"] = np.bincount(spans["entry"][spans["length"] >= 2].astype(np.int64),
                                                  minlength=int(ontology["entry_count"])).tolist()
        if name == "authored":
            ontology["heldout_entries"] = sorted(set(range(int(ontology["entry_count"]))) - active)
        torch.save(ontology, root / name / "ontology.pt")
    _cross_reference(track, root, int(round_settings(run)["seq_len"]))
    info = {"source": str(source), "seed": seed, "condition": condition, "manifests": manifests,
            "authored_entries": len(active), "new_entries": len([c for c in aset["candidates"] if c["entry"] in active])}
    _json(done, info)
    return info


def _cross_reference(track: dict[str, Any], root: Path, length: int) -> None:
    """Strata over the GPT-2 test windows: after masked gold concepts (curated linker), after authored
    new concepts (authored linker), and text linked by neither."""
    from vsa_embed.data.corpus import TokenCorpus, eval_windows
    from vsa_embed.training.lm import save_reference_strata
    curated, authored = TokenCorpus.open(root / "curated" / "eval"), TokenCorpus.open(root / "authored" / "eval")
    if not np.array_equal(np.asarray(curated.tokens), np.asarray(authored.tokens)):
        raise ValueError("curated and authored evaluation corpora must hold the same tokens")
    starts = eval_windows(curated, count=max(1, (len(curated) - length - 1) // length), length=length)
    masked_entries = list({int(e) for e in gold(track)["masked_strings"].values()})
    entry_base = int(track["entry_count"])
    c_spans, a_spans = curated.spans, authored.spans
    masks = {"masked": _after_masks(starts, c_spans, np.isin(c_spans["entry"], masked_entries), length),
             "authored_new": _after_masks(starts, a_spans, a_spans["entry"] >= entry_base, length)}
    linked = np.zeros_like(masks["masked"])
    for spans in (c_spans, a_spans):
        everything = np.ones(spans["entry"].shape, bool)
        linked |= _after_masks(starts, spans, everything, length) | _after_masks(starts, spans, everything, length, inside=True)
    masks["unlinked"] = ~linked
    save_reference_strata(root / "reference.npz", starts, masks, length)


def cross_config(run: Path, condition: str, seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    from vsa_embed.data.corpus import TokenCorpus
    from vsa_embed.experiments.e4_plan import BASE, SIZES, conditions
    track = load_track(run)
    root = cross_root(track)
    corpus = "authored" if condition == "authored" else "curated"
    channel = conditions(settings["operator"], int(settings["key_dimension"]), int(settings["channel_dimension"]), 0, 0, {})
    spec = channel["C0"] if condition == "none" else channel[settings["cross"]["channel_condition"]]
    config = copy.deepcopy(BASE)
    for part in (SIZES[settings["cross"]["size"]], spec, settings["cross"].get("config_overrides") or {}):
        for key, value in part.items():
            config.setdefault(key, {})
            if isinstance(value, dict):
                config[key].update(copy.deepcopy(value))
            else:
                config[key] = value
    config["model"]["seq_len"] = length = int(settings["seq_len"])
    count = max(1, (len(TokenCorpus.open(root / corpus / "eval")) - length - 1) // length)
    config["train"]["total_tokens"] = int(settings["cross"].get("run_tokens") or settings["cross"]["train_tokens"])
    config["train"]["checkpoint_minutes"] = 10
    config["eval"].update(windows=count, save_window_losses=True, reference_strata=str(root / "reference.npz"))
    config["seed"] = seed
    config["data"].update(train=str(root / corpus / "train"), eval=str(root / corpus / "eval"), ontology=str(root / corpus / "ontology.pt"))
    config["experiment"] = f"e7-cross-{settings['cross']['size']}-{condition}-s{seed}"
    return config


def cross_train(run: Path, seed: int, condition: str, *, resume: bool = False) -> dict[str, Any]:
    settings = round_settings(run)
    cross_prepare(run)
    config = cross_config(run, condition, seed, settings)
    return _train(config, Path(run) / "cross" / f"{settings['cross']['size']}-{condition}-s{seed}", resume)


# ---------------------------------------------------------------------------------------------
# CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["base", "verify", "entigraph", "notes", "verify-notes", "spa", "verbal", "materialize",
                                            "train", "cross-prepare", "cross-train", "plan", "report"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1); parser.add_argument("--condition", default=None)
    parser.add_argument("--author", default=None); parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", default="{}", help="JSON merged into round.json before the step (persisted)")
    parser.add_argument("--resume", action="store_true")
    args, rest = parser.parse_known_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    changes = json.loads(args.set)
    if changes:
        update_round_settings(args.run, changes)
    if args.command == "base":
        print(json.dumps(train_base(args.run, args.seed, resume=args.resume), default=_default))
    elif args.command == "verify":
        result = verify(args.run, args.seed, args.author or load_track(args.run)["consumer"], device=device)
        print(json.dumps(result["summary"], indent=1))
    elif args.command == "entigraph":
        result = entigraph(args.run, device=device)
        print(json.dumps({k: result[k] for k in ("pairs", "documents", "tokens")}))
    elif args.command == "notes":
        print(json.dumps({"notes": notes(args.run, device=device)["count"]}))
    elif args.command == "verify-notes":
        result = verify_notes(args.run, args.seed, device=device)
        print(json.dumps({k: result[k] for k in ("notes", "kept", "tokens", "median_accepted_utility")}))
    elif args.command == "spa":
        result = spa(args.run, device=device)
        print(json.dumps({k: result[k] for k in ("documents", "texts", "tokens")}))
    elif args.command == "verbal":
        result = verbal(args.run, args.seed)
        print(json.dumps({k: result[k] for k in ("concepts", "sentences", "tokens")}))
    elif args.command == "materialize":
        print(json.dumps(materialize(args.run, args.seed, args.condition), indent=1, default=_default)[:3000])
    elif args.command == "train":
        print(json.dumps(train_condition(args.run, args.seed, args.condition, resume=args.resume), default=_default))
    elif args.command == "cross-prepare":
        print(json.dumps(cross_prepare(args.run), indent=1, default=_default)[:3000])
    elif args.command == "cross-train":
        print(json.dumps(cross_train(args.run, args.seed, args.condition, resume=args.resume), default=_default))
    elif args.command == "plan":
        from vsa_embed.experiments import e7_plan
        return e7_plan.main(["--run", str(args.run), *rest])
    else:
        from vsa_embed.experiments import e7_report
        return e7_report.main(["--run", str(args.run), *rest])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
