"""E9 dimension-3 baselines (claim C; manuscript/novelty-check-2026-10.md §4.4, C-B1–C-B6): evaluation only.

Runs on a trained E9 run (any model: P0, C0′, C2, C5) with **the same new-word and edit items** as
`e9_ontology_edit` and **the same metric code** (`e9_ontology_edit.summarize_new_words` /
`summarize_edits`: property / entailment / paraphrase / statement for new words; ES, EM, PS, PM, NS,
score and the control edit for edits). No model is trained. Methods (`--methods`; each one is skipped,
with the reason recorded, where it does not apply to the run):

- **`context`** — in-context upper bound for new words (**text evidence**, reported separately from the
  structure-only rows): the verbalized frame (`Definition of <name>: is a tool; area finance; …`, the
  frame's relation phrases and filler names — never the property or statement templates) precedes every
  prompt. Sources: `context_frame`; `context_other` (the frame of another new word: the copy/distraction
  control); for channel models also `context_frame_none` (the new entries' rows zeroed: the text path
  alone). Scores are PMI against the same context with the null surface (the subject's contribution
  given the context) and raw accuracy (`raw_correct`, which credits copying); the context's token cost
  is recorded. Onoe et al. (ACL 2023) and Padmanabhan et al. (NeurIPS 2023) for the protocol.
- **`ike`** — in-context knowledge editing (IKE; Zheng et al., EMNLP 2023, arXiv:2305.12740): `--ike-demos`
  demonstrations built from *other* edit items, each of the three IKE kinds (copy: the new fact's own
  prompt; update: its paraphrase; retain: a neighbour's prompt with its true filler), then
  `New Fact: <edit prompt> <new filler>` and `Prompt: <test prompt>`; the control writes the control
  filler into the new fact.
- **`rome`, `memit`, `alphaedit`** — weight editing (`vsa_embed.knowledge_editing`) on the run's host,
  rewrite prompt = the efficacy item's first template, target = its new filler: ROME one edit at a time
  (the edited layer restored after each edit's items are scored), MEMIT and AlphaEdit with all edits as
  one batch (as all ontology edits are applied at once); the control edits write the control fillers.
  Second moments from `--cov-tokens` tokens of the run's training stream.
- **`transplant`** (compose channels) — frame transplant: term A reads the whole frame of term B (same
  edited relation, a different filler), for every edited concept. Items: A's templated relations where A's
  and B's single fillers differ, candidates [A's filler, B's filler]; `follow` = share preferring B's after
  the transplant (edited relation and all relations), against a control transplant of a third term C's frame.
- **`channel_off`** (channel models) — the ontology edit scored with the channel switched off (no spans)
  before and after the edit: the edit must revert exactly (`max |Δd|`), and `d` with the channel off vs on
  shows how much of the old fact the host stores outside the channel.
- **`intra`** — intra-entity locality: per edited concept up to `--intra-relations` of its *other*
  templated single-filler relations (true filler vs a same-relation distractor); `locality` = share still
  preferring the true filler after the edit, and the mean / max `|Δd|`, for every edit method of this run
  (the ontology edit on compose channels, IKE, the weight editors).
- **`rows`** (channel models; WP-PQ1 trains the token-level arms) — zero-shot rows for new words at the
  channel's site, evaluation-time sources next to `own`: `surface_mean` (FVT / Hewitt: the mean input
  embedding of the name's subtokens, scaled to the mean trained row norm; Gee et al. 2022) and
  `definition_encoder` (the host's mean-pooled final state over the verbalized frame without the name,
  ridge-mapped to the rows of `--fit-entries` seen entries' verbalized frames; Bahdanau et al. 2017).

    python -m vsa_embed.experiments.e9_dim3_baselines evaluate --run RUN --new-items DIR --edit-items DIR --output OUT
        [--methods context,ike,rome,memit,alphaedit,transplant,channel_off,intra,rows] [--alias-table JSON]
        [--cov-tokens 200000] [--edit-limit N] [--new-limit N] [--device cuda] [--batch-size 32]
    python -m vsa_embed.experiments.e9_dim3_baselines toy --model HuggingFaceTB/SmolLM2-135M --output OUT   (known-fact check)

Output (run-folder contract): `resolved_config.yaml`, `manifest.json`, `summary.json`, `report.md`,
`predictions.jsonl` (one row per method × condition × item).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

from .. import knowledge_editing as ke
from ..compose import FrameSchedule
from ..evaluation import channel_probes as cp
from ..statistics import holm_adjust
from . import e5_zeroshot as zs
from . import e9_ontology_edit as edit
from .e5_common import (E5Run, RELATION_PHRASES, clear_output, entry_rows, finish_output, fmt, fmt_ci, json_ready, open_run,
                        override_rows, readable_atomic, start_output, write_json)

METHODS = ("context", "ike", "rome", "memit", "alphaedit", "transplant", "channel_off", "intra", "rows")
WEIGHT_METHODS = ("rome", "memit", "alphaedit")
EDIT_METHODS = ("ontology", "ike") + WEIGHT_METHODS
CONDITIONS = edit.EDIT_CONDITIONS          # before, after, control
CONTEXT_SOURCES = ("context_frame", "context_other", "context_frame_none")
ROW_SOURCES = ("surface_mean", "definition_encoder")


# -- shared helpers ------------------------------------------------------------------------------------------

def relation_phrase(relation: str) -> str:
    return RELATION_PHRASES.get(relation, relation.replace("_", " "))


def verbalize(gold: dict[str, Sequence[str]], subject: str | None = None) -> str:
    """`Definition of <subject>: <relation phrase> <filler>; …` (frame order of `gold`); without a subject
    the bare definition (the definition encoder's input)."""
    facts = "; ".join(f"{relation_phrase(r)} {t}" for r, texts in gold.items() for t in texts)
    return f"Definition of {subject}: {facts}." if subject else f"{facts}."


def lexicon_for_run(run: E5Run) -> tuple[Any | None, str | None]:
    """(lexicon, error): the run's track lexicon (`e9_tracks.lexicon_for`), WordNet by default."""
    from . import e9_tracks as tracks
    try:
        spec = tracks.track_spec(run.config.get("e9_track") or "wordnet", run.config.get("e9_family") or "smollm2")
        return tracks.lexicon_for(spec, run.ontology), None
    except Exception as error:                      # noqa: BLE001 — recorded; the methods needing it are skipped
        return None, f"{type(error).__name__}: {error}"


@contextlib.contextmanager
def channel_off(run: E5Run) -> Iterator[None]:
    """Within the block the adapter links nothing, so the model runs without its channel."""
    adapter = run.adapter
    saved = adapter.spans_fn
    adapter.spans_fn = None
    try:
        yield
    finally:
        adapter.spans_fn = saved


def with_context(items: Sequence[dict[str, Any]], contexts: dict[str, str]) -> list[dict[str, Any]]:
    """Items whose templates are preceded by `contexts[item id]` (items without a context unchanged).
    Braces in the context are escaped (templates are formatted with `{x}`)."""
    out = []
    for item in items:
        context = contexts.get(item["id"])
        if context is None:
            out.append(item)
            continue
        safe = context.replace("{", "{{").replace("}", "}}")
        out.append({**item, "templates": [safe + t for t in item["templates"]]})
    return out


@contextlib.contextmanager
def smaller_batches(adapter: Any, factor: int = 4) -> Iterator[None]:
    """Within the block the adapter scores `factor` times fewer texts per batch (prompts with a long context: the
    adapter's log-softmax materializes full-vocabulary logits per batch)."""
    saved = adapter.batch_size
    adapter.batch_size = max(1, saved // factor)
    try:
        yield
    finally:
        adapter.batch_size = saved


def token_count(tokenizer: Any, texts: Sequence[str]) -> float:
    return float(np.mean([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts])) if texts else 0.0


# -- new words ----------------------------------------------------------------------------------------------------

def _new_word_setup(run: E5Run, items_dir: Path, limit: int | None) -> dict[str, Any]:
    manifest, concepts, items = edit.load_item_dir(items_dir, edit.SCHEMA_NEW)
    if limit:
        keep = {c["concept"] for c in concepts[:limit]}
        concepts = [c for c in concepts if c["concept"] in keep]
        items = [i for i in items if i["concept"] in keep]
    ontology = run.ontology
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    base = int(ontology["entry_count"])
    entry_of = {c["concept"]: base + i for i, c in enumerate(concepts)}
    for c in concepts:
        c["entry"] = entry_of[c["concept"]]
    adapter = edit.extended_adapter(run, {c["surface"]: entry_of[c["concept"]] for c in concepts})
    return {"manifest": manifest, "concepts": concepts, "items": items, "entry_of": entry_of, "adapter": adapter,
            "frames": [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in concepts],
            "resolved": edit.link_check(adapter, concepts, items, entry_of),
            "surfaces": {c["concept"]: c["surface"] for c in concepts}}


def _score_new(run: E5Run, setup: dict[str, Any], items: Sequence[dict[str, Any]], rows: dict[int, torch.Tensor] | None
               ) -> dict[str, Any]:
    """Prompt (PMI) and statement scores of `items` with the new entries inserted (own frames) and `rows` overriding."""
    channel = run.channel
    prompt_items = [i for i in items if i["test"] in {"property", "entailment"}]
    statement_items = [i for i in items if i["test"] == "statement"]
    with edit.inserted_entries(channel, len(setup["concepts"]), setup["frames"]), \
            (override_rows(channel, rows) if channel is not None and rows else contextlib.nullcontext()):
        # The adapter is swapped in for the extended one (new aliases) only for this scoring.
        prompts, _ = zs.score_prompts(setup["adapter"], prompt_items, setup["surfaces"])
        statements = edit.score_statements(setup["adapter"], statement_items, setup["surfaces"])
    kinds = {i["id"]: i.get("edge_kind") for i in items}
    for row in prompts:
        row["edge_kind"] = kinds.get(row["id"])
    return {"prompts": prompts, "statements": statements}


def evaluate_context(run: E5Run, setup: dict[str, Any], *, seed: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """New words with the verbalized frame in context (`context_frame`), another new word's frame
    (`context_other`) and, for channel models, the frame with the new entries' rows zeroed."""
    concepts, items = setup["concepts"], setup["items"]
    own = {c["concept"]: verbalize(c["gold"], c["surface"]) for c in concepts}
    order = [c["concept"] for c in concepts]
    rng = random.Random(seed)
    shifted = order[:]
    while len(shifted) > 1 and any(a == b for a, b in zip(order, shifted)):
        rng.shuffle(shifted)
    other_of = dict(zip(order, shifted))
    gold_of = {c["concept"]: c["gold"] for c in concepts}
    other = {c["concept"]: verbalize(gold_of[other_of[c["concept"]]], c["surface"]) for c in concepts}
    sources = {"own": {}, "context_frame": own, "context_other": other}
    if run.channel is not None:
        sources["context_frame_none"] = own
    width = run.channel.gate.in_features // 2 if run.channel is not None else 0
    zero_rows = {e: torch.zeros(width) for e in setup["entry_of"].values()} if run.channel is not None else None
    results = {}
    for source, contexts in sources.items():
        log(f"  context: {source}")
        scored_items = with_context(items, {i["id"]: contexts[i["concept"]] + "\n" for i in items if i["concept"] in contexts})
        with smaller_batches(setup["adapter"], 1 if source == "own" else 2):
            results[source] = _score_new(run, setup, scored_items, zero_rows if source == "context_frame_none" else None)
    evaluation = {"resolved": setup["resolved"], "sources": list(results), "results": results}
    summary = edit.summarize_new_words(evaluation)
    summary["raw_correct"] = {s: float(np.mean([r["raw_correct"] for r in res["prompts"] if r["test"] == "property"]))
                              for s, res in results.items() if res["prompts"]}
    summary["context_tokens"] = token_count(run.tokenizer, list(own.values()))
    summary["note"] = ("text evidence (verbalized frame in the prompt), reported separately from the structure-only rows; "
                       "`own` = no context (the model's own rows); comparisons are own − source")
    return {"summary": summary, "results": results}


def definition_rows(run: E5Run, setup: dict[str, Any], *, lexicon: Any, fit_entries: int, seed: int) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Definition-encoder rows for the new entries: the host's mean-pooled final state over the verbalized
    frame (no name), ridge-mapped to the rows of seen entries' verbalized frames."""
    channel = run.channel
    ontology = run.ontology
    view = edit.OntologyView(ontology, run.table, lexicon)
    frequency = np.asarray(ontology["train_frequency"])
    heldout = set(run.adapter.heldout_entries) | set(setup["entry_of"].values())
    rng = np.random.default_rng(seed)
    seen = np.asarray([e for e in np.flatnonzero(frequency >= 10) if e not in heldout and view.frame(int(e))], dtype=np.int64)
    fit = np.sort(rng.choice(seen, size=min(fit_entries, len(seen)), replace=False)) if len(seen) else seen
    if len(fit) < 3:
        return {}, {"definition_encoder": "fewer than 3 seen entries"}

    def text(atom_id: int) -> str:
        return view.text(atom_id) or readable_atomic(view.atomic_names[atom_id])

    fit_texts = []
    for e in fit.tolist():
        gold: dict[str, list[str]] = defaultdict(list)
        for r, f in view.frame(e):
            gold[view.relation_names[r]].append(text(f))
        fit_texts.append(verbalize(gold))
    new_texts = [verbalize(c["gold"]) for c in setup["concepts"]]
    targets = entry_rows(channel, torch.as_tensor(fit)).numpy()
    norm = float(np.linalg.norm(targets, axis=1).mean())
    x_fit = zs.pooled_states(run.adapter, channel, fit_texts, [None] * len(fit_texts), heldout)
    predict, alpha = zs.ridge_map(x_fit, targets)
    x_new = zs.pooled_states(run.adapter, channel, new_texts, [None] * len(new_texts), heldout)
    rows = {setup["entry_of"][c["concept"]]: zs._scaled(v, norm) for c, v in zip(setup["concepts"], predict(x_new))}
    return rows, {"definition_fit_entries": int(len(fit)), "definition_alpha": alpha, "mean_row_norm": norm}


def evaluate_rows(run: E5Run, setup: dict[str, Any], *, lexicon: Any, fit_entries: int, seed: int,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """`own` vs `surface_mean` and `definition_encoder` rows at the channel's site (channel models)."""
    channel = run.channel
    linked = {c["concept"]: setup["entry_of"][c["concept"]] for c in setup["concepts"]}
    info: dict[str, Any] = {}
    with edit.inserted_entries(channel, len(setup["concepts"]), setup["frames"]):
        rows, fit_info = zs.baseline_rows(run, setup["adapter"], setup["concepts"], linked, ["surface_mean"],
                                          manifest=setup["manifest"], items_dir=Path("."), fit_entries=fit_entries, contexts=0,
                                          seed=seed, log=log)
        info.update(fit_info)
        if lexicon is not None and channel.mode == "compose":
            definition, definition_info = definition_rows(run, setup, lexicon=lexicon, fit_entries=fit_entries, seed=seed)
            info.update(definition_info)
            if definition:
                rows["definition_encoder"] = definition
        elif lexicon is None:
            info["definition_encoder"] = "no lexicon for this run"
    results = {"own": _score_new(run, setup, setup["items"], None)}
    for source in ROW_SOURCES:
        if source in rows:
            log(f"  rows: {source}")
            results[source] = _score_new(run, setup, setup["items"], rows[source])
    evaluation = {"resolved": setup["resolved"], "sources": list(results), "results": results}
    summary = edit.summarize_new_words(evaluation)
    summary["info"] = info
    return {"summary": summary, "results": results}


# -- edits: items -----------------------------------------------------------------------------------------------

def _edit_setup(run: E5Run, items_dir: Path, limit: int | None) -> dict[str, Any]:
    manifest, concepts, items = edit.load_item_dir(items_dir, edit.SCHEMA_EDITS)
    edited = [c for c in concepts if c.get("role") == "edited"]
    if limit:
        edited = edited[:limit]
        keep = {c["concept"] for c in edited}
        items = [i for i in items if i["edit"] in keep]
        used = {i["concept"] for i in items} | keep
        concepts = [c for c in concepts if c["concept"] in used]
    adapter = run.adapter
    resolved = edit.link_check(adapter, concepts, items, {c["concept"]: int(c["entry"]) for c in concepts})
    for c in concepts:
        resolved[c["concept"]]["entry_status"] = cp.entry_status([int(c["entry"])], adapter.heldout_entries, adapter.train_frequency)
    return {"manifest": manifest, "concepts": concepts, "edited": edited, "items": items, "resolved": resolved,
            "surfaces": {c["concept"]: c["surface"] for c in concepts}}


def formatted(lexicon: Any, relation: str, text: str, *, like: str, like_text: str) -> str:
    """A filler formatted as the items format answers (the lexicon's, else by substitution into `like`)."""
    if lexicon is not None:
        try:
            return lexicon.answer(relation, text)
        except Exception:                            # noqa: BLE001
            pass
    return like.replace(like_text, text) if like_text in like else " " + text


def intra_entity_items(run: E5Run, edited: Sequence[dict[str, Any]], lexicon: Any, *, per_concept: int = 3,
                       seed: int = 0) -> list[dict[str, Any]]:
    """Per edited concept: up to `per_concept` of its other templated single-filler relations, candidates
    [true filler, a same-relation distractor] (gold 0; d = log p(distractor) − log p(true))."""
    view = edit.OntologyView(run.ontology, run.table, lexicon)
    pools = view.filler_pools()
    rng = random.Random(seed)
    items = []
    for c in edited:
        entry = int(c["entry"])
        pos = view.pos[entry]
        by_relation: dict[int, list[int]] = defaultdict(list)
        for r, f in view.frame(entry):
            by_relation[r].append(f)
        usable = []
        for r, fillers in sorted(by_relation.items()):
            name = view.relation_names[r]
            if name == c["relation"] or len(fillers) != 1 or not lexicon.prompts(pos, name) or not view.text(fillers[0]):
                continue
            true = view.text(fillers[0])
            options = sorted({view.text(f) for f in pools[(pos, r)] if f not in fillers and view.text(f) and view.text(f) != true})
            if options:
                usable.append((name, true, options))
        for name, true, options in rng.sample(usable, min(per_concept, len(usable))):
            wrong = rng.choice(options)
            items.append({"id": f"{c['concept']}-intra-{name}", "concept": c["concept"], "edit": c["concept"], "test": "intra",
                          "relation": name, "templates": lexicon.prompts(pos, name), "null": lexicon.null_surface,
                          "candidates": [lexicon.answer(name, true), lexicon.answer(name, wrong)], "gold": 0})
    return items


def summarize_intra(results: dict[str, dict[str, dict[str, Any]]], items: Sequence[dict[str, Any]], keep: set[str], *,
                    resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    ids = [i["id"] for i in items if i["test"] == "intra" and i["edit"] in keep]
    if not ids or any(i not in results.get("after", {}) for i in ids):
        return {"n": 0}
    d = {c: np.asarray([results[c][i]["d_mean"] for i in ids]) for c in CONDITIONS if c in results}
    keep_true = {c: 1 - np.asarray([results[c][i]["new_preferred"] for i in ids]) for c in CONDITIONS if c in results}
    change = np.abs(d["after"] - d["before"])
    return {"n": len(ids), "locality_before": float(keep_true["before"].mean()), "locality_after": float(keep_true["after"].mean()),
            **({"locality_control": float(keep_true["control"].mean())} if "control" in keep_true else {}),
            "shift": zs.paired_difference(d["after"], d["before"], resamples=resamples, seed=seed),
            "mean_abs_change": float(change.mean()), "max_abs_change": float(change.max())}


def _edit_subsets(setup: dict[str, Any]) -> dict[str, set[str]]:
    resolved = setup["resolved"]
    linked = {c for c, r in resolved.items() if r["status"] == "linked"}
    status = {c["concept"]: ("heldout" if resolved[c["concept"]]["entry_status"] == "heldout" else "seen") for c in setup["edited"]}
    return {s: {c for c, v in status.items() if c in linked and (s == "all" or v == s)} for s in ("all", "seen", "heldout")}


def summarize_method(setup: dict[str, Any], results: dict[str, dict[str, dict[str, Any]]], intra: Sequence[dict[str, Any]], *,
                     applicable: bool = True, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    evaluation = {"resolved": setup["resolved"], "results": results, "edit_applicable": applicable}
    summary = edit.summarize_edits(evaluation, setup["items"], setup["concepts"], resamples=resamples, seed=seed)
    if intra:
        summary["intra"] = {s: summarize_intra(results, intra, keep, resamples=resamples, seed=seed)
                            for s, keep in _edit_subsets(setup).items()}
    return summary


# -- edits: methods ----------------------------------------------------------------------------------------------------

def _pairs(run: E5Run, items: Sequence[dict[str, Any]], surfaces: dict[str, str]) -> dict[str, dict[str, Any]]:
    return edit.score_pairs(run.adapter, items, surfaces) if items else {}


def ontology_edit_results(run: E5Run, setup: dict[str, Any], scored: Sequence[dict[str, Any]], before: dict[str, Any]
                          ) -> dict[str, dict[str, Any]]:
    """The `e9_ontology_edit` conditions on `scored` (edit + intra items): after / control = the frame edit."""
    relation_id = {n: i for i, n in enumerate(run.ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(run.ontology["atomic_names"])}
    results = {"before": before}
    for condition, target in (("after", "new"), ("control", "control")):
        edits = [(int(c["entry"]), relation_id[c["relation"]], atomic_id[c["old"]], atomic_id[c[target]]) for c in setup["edited"]]
        with edit.applied_edits(run.channel, edits):
            results[condition] = _pairs(run, scored, setup["surfaces"])
    return results


def ike_contexts(setup: dict[str, Any], lexicon: Any, *, demos: int, target: str, seed: int) -> dict[str, str]:
    """Per item id: the IKE context (demonstrations from other edits, then the new fact)."""
    by_edit: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in setup["items"]:
        by_edit[item["edit"]][item["test"]].append(item)
    surfaces = setup["surfaces"]
    rng = random.Random(seed)
    edited = setup["edited"]

    def fact(c: dict[str, Any], which: str) -> str:
        efficacy = by_edit[c["concept"]]["efficacy"][0]
        answer = efficacy["candidates"][1] if which == "new" else formatted(
            lexicon, c["relation"], c["control_text"], like=efficacy["candidates"][1], like_text=c["new_text"])
        return efficacy["templates"][0].format(x=c["surface"]) + answer

    contexts: dict[str, str] = {}
    kinds = ("copy", "update", "retain", "retain")
    for c in edited:
        others = [o for o in edited if o["concept"] != c["concept"]]
        lines = []
        for k, o in enumerate(rng.sample(others, min(demos, len(others)))):
            kind = kinds[k % len(kinds)]
            new_fact = fact(o, "new")
            if kind == "update" and by_edit[o["concept"]]["paraphrase"]:
                para = by_edit[o["concept"]]["paraphrase"][0]
                prompt = para["templates"][0].format(x=o["surface"]) + para["candidates"][1]
            elif kind == "retain" and by_edit[o["concept"]]["neighbourhood"]:
                nb = by_edit[o["concept"]]["neighbourhood"][0]
                prompt = nb["templates"][0].format(x=surfaces[nb["concept"]]) + nb["candidates"][0]
            else:
                prompt = new_fact
            lines.append(f"New Fact: {new_fact}\nPrompt: {prompt}\n")
        head = "\n".join(lines) + f"\nNew Fact: {fact(c, target)}\nPrompt: "
        for item in setup["items"] + setup.get("intra", []):
            if item["edit"] == c["concept"]:
                contexts[item["id"]] = head
    return contexts


def ike_results(run: E5Run, setup: dict[str, Any], scored: Sequence[dict[str, Any]], before: dict[str, Any], lexicon: Any, *,
                demos: int, seed: int) -> tuple[dict[str, dict[str, Any]], float]:
    results = {"before": before}
    tokens = 0.0
    for condition, target in (("after", "new"), ("control", "control")):
        contexts = ike_contexts(setup, lexicon, demos=demos, target=target, seed=seed)
        if condition == "after":
            tokens = token_count(run.tokenizer, sorted(set(contexts.values())))
        with smaller_batches(run.adapter, 4):
            results[condition] = _pairs(run, with_context(scored, contexts), setup["surfaces"])
    return results, tokens


def corpus_texts(run: E5Run, tokens: int, *, window: int = 256, seed: int = 0) -> list[str]:
    """About `tokens` tokens of the run's training stream: evenly spread windows, cut at end-of-text, decoded."""
    from ..data.corpus import TokenCorpus
    corpus = TokenCorpus.open(Path(run.config["data"]["train"]))
    eos = int(corpus.manifest.get("eos_id", -1))
    window = min(window, int(run.adapter.max_length))       # the host's positions (re-tokenized text may grow slightly)
    count = max(1, tokens // window)
    starts = np.linspace(0, max(0, len(corpus) - window - 1), count).astype(np.int64)
    texts = []
    for start in starts.tolist():
        ids = np.asarray(corpus.tokens[start:start + window], dtype=np.int64)
        for piece in np.split(ids, np.flatnonzero(ids == eos)):
            piece = piece[piece != eos]
            if len(piece) >= 8:
                texts.append(run.tokenizer.decode(piece.tolist()))
    return texts


def weight_edit_requests(setup: dict[str, Any], lexicon: Any, target: str) -> list[ke.EditRequest]:
    efficacy = {i["edit"]: i for i in setup["items"] if i["test"] == "efficacy"}
    requests = []
    for c in setup["edited"]:
        item = efficacy.get(c["concept"])
        if item is None:
            continue
        answer = item["candidates"][1] if target == "new" else formatted(
            lexicon, c["relation"], c["control_text"], like=item["candidates"][1], like_text=c["new_text"])
        requests.append(ke.EditRequest(item["templates"][0], c["surface"], answer, c["concept"]))
    return requests


def weight_edit_results(run: E5Run, setup: dict[str, Any], scored: Sequence[dict[str, Any]], before: dict[str, Any], lexicon: Any,
                        method: str, moments: dict[int, torch.Tensor], settings: ke.EditorSettings, *,
                        log: Callable[[str], None] = print) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    host = ke.EditableHost(run.adapter)
    layers = ke.edit_layers(host, settings)
    saved = host.snapshot(layers)
    results: dict[str, dict[str, Any]] = {"before": before}
    records: dict[str, Any] = {"settings": settings.record(layers)}
    by_edit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_edit[item["edit"]].append(item)
    try:
        for condition, target in (("after", "new"), ("control", "control")):
            requests = weight_edit_requests(setup, lexicon, target)
            started = time.monotonic()
            if method == "rome":
                results[condition], per_edit = {}, []
                for k, request in enumerate(requests):
                    per_edit.append(ke.apply_rome(host, request, settings, moments))
                    results[condition].update(_pairs(run, by_edit[request.key], setup["surfaces"]))
                    host.restore(saved)
                    if (k + 1) % 25 == 0:
                        log(f"  {method} {condition}: {k + 1}/{len(requests)} edits")
                records[condition] = {"edits": len(requests), "nll_mean": float(np.mean([r["nll"] for r in per_edit])) if per_edit else None,
                                      "delta_norm_mean": float(np.mean([r["delta_norm"] for r in per_edit])) if per_edit else None}
            else:
                info = ke.apply_memit(host, requests, settings, moments)
                results[condition] = _pairs(run, scored, setup["surfaces"])
                host.restore(saved)
                records[condition] = info
            records[condition]["seconds"] = time.monotonic() - started
    finally:
        host.restore(saved)
    return results, records


def transplant_schedule(schedule: FrameSchedule, mapping: dict[int, int]) -> FrameSchedule:
    """Entry `a` reads the whole frame of `mapping[a]` (every other entry unchanged)."""
    offsets = schedule.offsets.tolist()
    relations, fillers, new_offsets = [], [], [0]
    for e in range(len(offsets) - 1):
        source = mapping.get(e, e)
        lo, hi = offsets[source], offsets[source + 1]
        relations.append(schedule.relations[lo:hi]); fillers.append(schedule.fillers[lo:hi])
        new_offsets.append(new_offsets[-1] + hi - lo)
    return FrameSchedule(torch.tensor(new_offsets, dtype=schedule.offsets.dtype), torch.cat(relations), torch.cat(fillers))


@contextlib.contextmanager
def transplanted(channel: Any, mapping: dict[int, int]) -> Iterator[None]:
    composer = channel.composer
    original = composer.schedule
    composer.set_schedule(transplant_schedule(original, mapping))
    try:
        yield
    finally:
        composer.set_schedule(original)


def transplant_items(run: E5Run, setup: dict[str, Any], lexicon: Any, *, seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """(items, A → B entries, A → C entries): each edited concept A gets the frame of B (same edited relation,
    another old filler); C is the control donor. Items: A's templated single-filler relations whose A and B
    fillers differ, candidates [A's, B's] (d = log p(B's) − log p(A's))."""
    view = edit.OntologyView(run.ontology, run.table, lexicon)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in setup["edited"]:
        groups[c["relation"]].append(c)
    donor, control = {}, {}
    for members in groups.values():
        for c in members:
            options = [o for o in members if o["concept"] != c["concept"] and o["old_text"] != c["old_text"]]
            if len(options) < 2:
                continue
            b, x = rng.sample(options, 2)
            donor[c["concept"]], control[c["concept"]] = int(b["entry"]), int(x["entry"])

    def single(entry: int) -> dict[str, int]:
        found: dict[str, list[int]] = defaultdict(list)
        for r, f in view.frame(entry):
            found[view.relation_names[r]].append(f)
        return {r: fs[0] for r, fs in found.items() if len(fs) == 1}

    items = []
    for c in setup["edited"]:
        if c["concept"] not in donor:
            continue
        mine, theirs = single(int(c["entry"])), single(donor[c["concept"]])
        pos = view.pos[int(c["entry"])]
        for relation in sorted(set(mine) & set(theirs)):
            a_text, b_text = view.text(mine[relation]), view.text(theirs[relation])
            templates = lexicon.prompts(pos, relation)
            if not templates or not a_text or not b_text or a_text == b_text:
                continue
            items.append({"id": f"{c['concept']}-transplant-{relation}", "concept": c["concept"], "edit": c["concept"],
                          "test": "transplant", "relation": relation, "edited_relation": relation == c["relation"],
                          "templates": templates, "null": lexicon.null_surface,
                          "candidates": [lexicon.answer(relation, a_text), lexicon.answer(relation, b_text)], "gold": 0})
    return items, donor, control


def evaluate_transplant(run: E5Run, setup: dict[str, Any], lexicon: Any, *, seed: int, resamples: int) -> dict[str, Any]:
    items, donor, control = transplant_items(run, setup, lexicon, seed=seed)
    entry_of = {c["concept"]: int(c["entry"]) for c in setup["edited"]}
    results = {"before": _pairs(run, items, setup["surfaces"])}
    for condition, mapping in (("after", donor), ("control", control)):
        with transplanted(run.channel, {entry_of[c]: e for c, e in mapping.items()}):
            results[condition] = _pairs(run, items, setup["surfaces"])
    subsets = _edit_subsets(setup)
    summary: dict[str, Any] = {"pairs": len(donor), "items": len(items), "subsets": {}}
    for subset, keep in subsets.items():
        block = {}
        for scope, chosen in (("edited_relation", [i for i in items if i["edited_relation"]]), ("all_relations", items)):
            ids = [i["id"] for i in chosen if i["edit"] in keep]
            if not ids:
                block[scope] = {"n": 0}
                continue
            d = {c: np.asarray([results[c][i]["d_mean"] for i in ids]) for c in CONDITIONS}
            follow = {c: np.asarray([results[c][i]["new_preferred"] for i in ids]) for c in CONDITIONS}
            block[scope] = {"n": len(ids), **{f"follow_{c}": float(follow[c].mean()) for c in CONDITIONS},
                            "magnitude": zs.paired_difference(d["after"], d["before"], resamples=resamples, seed=seed),
                            "target_minus_control": zs.paired_difference(d["after"], d["control"], resamples=resamples, seed=seed)}
        summary["subsets"][subset] = block
    return {"summary": summary, "results": results, "items": items}


# -- the evaluation ------------------------------------------------------------------------------------------------

def applicable_methods(run: E5Run, requested: Sequence[str], lexicon: Any) -> tuple[list[str], dict[str, str]]:
    mode = run.mode
    skipped: dict[str, str] = {}
    methods = []
    for method in requested:
        if method in {"transplant"} and mode != "compose":
            skipped[method] = "needs composed rows (C5)"
        elif method == "channel_off" and mode != "compose":
            skipped[method] = "needs composed rows (C5): only an ontology edit can be audited"
        elif method == "rows" and mode in {"none", "hashed"}:
            skipped[method] = "needs a channel with per-entry rows"
        elif method in {"transplant", "intra"} and lexicon is None:
            skipped[method] = "no lexicon for this run's ontology"
        elif method in WEIGHT_METHODS and run.adapter.info.get("quantization"):
            skipped[method] = "weight editing runs on the unquantized model"
        else:
            methods.append(method)
    return methods, skipped


def evaluate(run: E5Run, *, new_items: Path | None, edit_items: Path | None, methods: Sequence[str], seed: int = 0,
             resamples: int = 2000, fit_entries: int = 4000, cov_tokens: int = 200_000, ike_demos: int = 4,
             intra_relations: int = 3, edit_limit: int | None = None, new_limit: int | None = None,
             settings: dict[str, ke.EditorSettings] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    lexicon, lexicon_error = lexicon_for_run(run)
    methods, skipped = applicable_methods(run, methods, lexicon)
    document: dict[str, Any] = {"methods": methods, "skipped": skipped, "lexicon": getattr(lexicon, "name", None),
                                "lexicon_error": lexicon_error, "timing": {}}
    predictions: list[dict[str, Any]] = []

    def timed(name: str, function: Callable[[], Any]) -> Any:
        started = time.monotonic()
        value = function()
        document["timing"][name] = time.monotonic() - started
        return value

    if new_items and {"context", "rows"} & set(methods):
        setup = _new_word_setup(run, new_items, new_limit)
        document["new_words"] = {"concepts": len(setup["concepts"]), "items": len(setup["items"])}
        if "context" in methods:
            out = timed("context", lambda: evaluate_context(run, setup, seed=seed, log=log))
            document["new_words"]["context"] = out["summary"]
            predictions += _new_rows("context", out["results"])
        if "rows" in methods:
            out = timed("rows", lambda: evaluate_rows(run, setup, lexicon=lexicon, fit_entries=fit_entries, seed=seed, log=log))
            document["new_words"]["rows"] = out["summary"]
            predictions += _new_rows("rows", out["results"])
    edit_methods = [m for m in methods if m in {"ike", "transplant", "channel_off", "intra", *WEIGHT_METHODS}]
    if edit_items and edit_methods:
        setup = _edit_setup(run, edit_items, edit_limit)
        intra = intra_entity_items(run, setup["edited"], lexicon, per_concept=intra_relations, seed=seed) \
            if "intra" in methods and lexicon is not None else []
        setup["intra"] = intra
        scored = setup["items"] + intra
        block: dict[str, Any] = {"edited": len(setup["edited"]), "items": len(setup["items"]), "intra_items": len(intra),
                                 "linked_edits": {s: sorted(keep) for s, keep in _edit_subsets(setup).items()}}
        log("  edits: before")
        before = timed("before", lambda: _pairs(run, scored, setup["surfaces"]))
        if run.mode == "compose" and ("intra" in methods or "channel_off" in methods):
            results = timed("ontology", lambda: ontology_edit_results(run, setup, scored, before))
            block["ontology"] = summarize_method(setup, results, intra, resamples=resamples, seed=seed)
            predictions += _edit_rows("ontology", results)
        if "channel_off" in methods and run.mode == "compose":
            with channel_off(run):
                off_before = _pairs(run, scored, setup["surfaces"])
                off = timed("channel_off", lambda: ontology_edit_results(run, setup, scored, off_before))
            on = results
            ids = [i["id"] for i in setup["items"] if i["test"] in {"efficacy", "paraphrase"} and i["edit"] in _edit_subsets(setup)["all"]]
            revert = np.abs(np.asarray([off["after"][i]["d_mean"] - off["before"][i]["d_mean"] for i in ids])) if ids else np.zeros(0)
            block["channel_off"] = {
                "summary": summarize_method(setup, off, intra, resamples=resamples, seed=seed),
                "revert_max_abs": float(revert.max()) if revert.size else None,
                "old_fact_on": float(np.mean([1 - on["before"][i]["new_preferred"] for i in ids])) if ids else None,
                "old_fact_off": float(np.mean([1 - off["before"][i]["new_preferred"] for i in ids])) if ids else None,
                "d_before_on_minus_off": zs.paired_difference(
                    np.asarray([on["before"][i]["d_mean"] for i in ids]), np.asarray([off["before"][i]["d_mean"] for i in ids]),
                    resamples=resamples, seed=seed) if ids else None}
            predictions += _edit_rows("channel_off", off)
        if "ike" in methods:
            log("  edits: IKE")
            results, tokens = timed("ike", lambda: ike_results(run, setup, scored, before, lexicon, demos=ike_demos, seed=seed))
            block["ike"] = {**summarize_method(setup, results, intra, resamples=resamples, seed=seed), "context_tokens": tokens,
                            "demonstrations": ike_demos}
            predictions += _edit_rows("ike", results)
        weight = [m for m in WEIGHT_METHODS if m in methods]
        if weight:
            host = ke.EditableHost(run.adapter)
            chosen = {m: (settings or {}).get(m) or ke.EditorSettings(method=m) for m in weight}
            layers = sorted({l for m in weight for l in ke.edit_layers(host, chosen[m])})
            log(f"  second moments: layers {layers}, {cov_tokens} tokens")
            texts = corpus_texts(run, cov_tokens, seed=seed)
            moments = timed("second_moments", lambda: ke.second_moments(host, texts, layers))
            block["second_moments"] = {"layers": layers, "texts": len(texts), "tokens_requested": cov_tokens}
            for method in weight:
                log(f"  edits: {method}")
                results, records = timed(method, lambda: weight_edit_results(run, setup, scored, before, lexicon, method, moments,
                                                                           chosen[method], log=log))
                block[method] = {**summarize_method(setup, results, intra, resamples=resamples, seed=seed), "records": records}
                predictions += _edit_rows(method, results)
        if "transplant" in methods:
            log("  frame transplant")
            out = timed("transplant", lambda: evaluate_transplant(run, setup, lexicon, seed=seed, resamples=resamples))
            block["transplant"] = out["summary"]
            predictions += _edit_rows("transplant", out["results"])
        document["edits"] = block
    return {"document": document, "predictions": predictions}


def _new_rows(method: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"method": method, "source": source, **{k: v for k, v in row.items() if k != "pmi"}}
            for source, result in results.items() for row in result["prompts"] + result["statements"]]


def _edit_rows(method: str, results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"method": method, "condition": condition, "id": item_id, **row}
            for condition, by_item in results.items() for item_id, row in by_item.items()]


# -- report --------------------------------------------------------------------------------------------------------

def _edit_table(block: dict[str, Any]) -> list[str]:
    lines = ["| method | subset | edits | ES before → after (control) | EM [95% CI] | EM − control [95% CI] | PS after | NS after | "
             "score | intra locality before → after | intra mean |Δd| |", "|---|---|---:|---|---|---|---:|---:|---:|---|---:|"]
    for method in ("ontology", "ike", *WEIGHT_METHODS):
        summary = block.get(method)
        if not summary:
            continue
        for subset, e in summary["subsets"].items():
            eff, par, nb = e.get("efficacy", {}), e.get("paraphrase", {}), e.get("neighbourhood", {})
            if not eff.get("n"):
                continue
            intra = (summary.get("intra") or {}).get(subset, {})
            lines.append(f"| {method} | {subset} | {e['edits']} | {fmt(eff['success_before'], 3)} → {fmt(eff['success_after'], 3)} "
                         f"({fmt(eff['success_control'], 3)}) | {fmt_ci(eff['magnitude'], 3)} | {fmt_ci(eff['target_minus_control'], 3)} | "
                         f"{fmt(par.get('success_after'), 3)} | {fmt(nb.get('success_after'), 3)} | {fmt(e.get('score'), 3)} | "
                         f"{fmt(intra.get('locality_before'), 3)} → {fmt(intra.get('locality_after'), 3)} | {fmt(intra.get('mean_abs_change'), 3)} |")
    return lines


def render(header: dict[str, Any], document: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E9 dimension-3 baselines — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Run `{source['run']}`, channel `{source['channel_mode']}`. Methods: {', '.join(document['methods']) or 'none'}"
             + (f"; skipped: {document['skipped']}" if document["skipped"] else "") + ". Same items and metric code as "
             "`e9_ontology_edit` (manuscript/novelty-check-2026-10.md §4.4).", ""]
    new = document.get("new_words") or {}
    for key, title, note in (("context", "New words — in-context verbalized frame (text evidence, not structure-only)",
                              "own = no context; comparisons are own − source (negative: the context helps)."),
                             ("rows", "New words — zero-shot rows at the channel site", "comparisons are own − source")):
        summary = new.get(key)
        if not summary:
            continue
        tests = ("property", "property_new", "entailment", "paraphrase", "statement_accuracy", "statement_loss")
        lines += [f"## {title}", "", f"{summary['linked_concepts']} of {summary['concepts']} new words link. {note}", "",
                  "| source | " + " | ".join(tests) + " |", "|---|" + "---:|" * len(tests)]
        for name, by_test in summary["sources"].items():
            lines.append(f"| {name} | " + " | ".join(fmt(by_test[t]["mean"], 3) for t in tests) + " |")
        if key == "context":
            lines += ["", f"Raw (non-PMI) property accuracy, which credits copying: "
                      + ", ".join(f"{s} {fmt(v, 3)}" for s, v in summary["raw_correct"].items())
                      + f". Mean context length {fmt(summary['context_tokens'], 1)} tokens."]
        rows = [c for c in summary["comparisons"] if c["test"] in tests]
        if rows:
            lines += ["", "| test | source | own − source [95% CI] | p (Holm) |", "|---|---|---|---:|"]
            lines += [f"| {c['test']} | {c['baseline']} | {fmt_ci({'mean': c['difference'], 'ci_low': c['ci_low'], 'ci_high': c['ci_high']}, 3)} "
                      f"| {fmt(c.get('p_holm'), 4)} |" for c in rows]
        lines.append("")
    block = document.get("edits")
    if block:
        lines += ["## Edits — ontology edit vs IKE vs weight editors", "",
                  f"{block['edited']} edited concepts, {block['items']} items, {block['intra_items']} intra-entity items. "
                  "ES/PS: share preferring the new filler; NS: neighbours keeping their filler; intra locality: the edited "
                  "term's other relations keeping their true filler.", ""]
        lines += _edit_table(block)
        for method in ("ike", *WEIGHT_METHODS):
            if block.get(method, {}).get("context_tokens"):
                lines.append(f"\nIKE context: {fmt(block[method]['context_tokens'], 1)} tokens, {block[method]['demonstrations']} demonstrations.")
        off = block.get("channel_off")
        if off:
            lines += ["", "**Channel-off audit** (the ontology edit with the channel switched off): max |Δd| after − before "
                      f"{fmt(off['revert_max_abs'], 5)} (0 = reverts exactly); old fact preferred with the channel on "
                      f"{fmt(off['old_fact_on'], 3)} vs off {fmt(off['old_fact_off'], 3)}; d(on) − d(off) before the edit "
                      f"{fmt_ci(off['d_before_on_minus_off'], 3)}."]
        transplant = block.get("transplant")
        if transplant:
            lines += ["", f"**Frame transplant** (A reads B's whole frame; {transplant['pairs']} pairs, {transplant['items']} items):", "",
                      "| subset | scope | n | follow B before → after (control C) | Δd [95% CI] | after − control [95% CI] |",
                      "|---|---|---:|---|---|---|"]
            for subset, scopes in transplant["subsets"].items():
                for scope, e in scopes.items():
                    if e.get("n"):
                        lines.append(f"| {subset} | {scope} | {e['n']} | {fmt(e['follow_before'], 3)} → {fmt(e['follow_after'], 3)} "
                                     f"({fmt(e['follow_control'], 3)}) | {fmt_ci(e['magnitude'], 3)} | {fmt_ci(e['target_minus_control'], 3)} |")
        lines.append("")
    lines += ["Timing (s): " + ", ".join(f"{k} {v:.0f}" for k, v in document["timing"].items()), ""]
    return "\n".join(lines)


# -- stage-level aggregation (read by `e9_report --dim3-baselines`) -----------------------------------------------------

FOLDER = "dim3-baselines"
STAGE_TESTS = ("efficacy", "paraphrase", "neighbourhood")


def _edit_item_values(run_dir: Path, folder: str = FOLDER) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, list[str]], list[dict[str, Any]]]:
    """((method → item id → {before, after, success}), linked edits per subset, edit items) of one run's evaluation."""
    path = Path(run_dir) / folder
    summary = json.loads((path / "summary.json").read_text())
    block = summary.get("edits") or {}
    values: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    if not block or not (path / "predictions.jsonl").exists():
        return {}, {}, []
    for line in (path / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if not row or "condition" not in row or row.get("method") not in EDIT_METHODS:
            continue
        values[row["method"]][row["id"]][row["condition"]] = row["d_mean"]
        if row["condition"] == "after":
            values[row["method"]][row["id"]]["success"] = row["new_preferred"]
    edit_items = summary.get("edit_items_path")
    items = edit.load_item_dir(Path(edit_items), edit.SCHEMA_EDITS)[2] if edit_items and Path(edit_items).exists() else []
    return {m: dict(v) for m, v in values.items()}, block.get("linked_edits", {}), items


def stage_summary(runs: dict[str, dict[int, Path]], *, candidate: str = "C5", resamples: int = 2000, seed: int = 0,
                  folder: str = FOLDER) -> dict[str, Any]:
    """Per model × seed: the run's `dim3-baselines` summary; and the candidate's ontology edit vs every other model ×
    edit method, paired over edit items (values averaged over each model's seeds; Holm over comparisons per test)."""
    per_model: dict[str, dict[int, Any]] = {}
    item_values: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    linked: dict[str, set[str]] = {}
    items: list[dict[str, Any]] = []
    for model, by_seed in runs.items():
        for s, run_dir in sorted(by_seed.items()):
            path = Path(run_dir) / folder / "summary.json"
            if not path.exists():
                continue
            per_model.setdefault(model, {})[s] = json.loads(path.read_text())
            values, linked_now, items_now = _edit_item_values(Path(run_dir), folder)
            items = items or items_now
            for subset, ids in linked_now.items():
                linked[subset] = (linked[subset] & set(ids)) if subset in linked else set(ids)
            for method, by_item in values.items():
                for item_id, v in by_item.items():
                    for key in ("before", "after", "control", "success"):
                        if key in v:
                            item_values[(model, method)][item_id][key].append(v[key])
    comparisons: dict[str, list[dict[str, Any]]] = {}
    mine = item_values.get((candidate, "ontology"))
    if mine and items:
        averaged = {key: {i: {k: float(np.mean(x)) for k, x in v.items()} for i, v in by_item.items()} for key, by_item in item_values.items()}
        for subset in ("all", "seen", "heldout"):
            keep = linked.get(subset, set())
            block = []
            for test in STAGE_TESTS:
                ids = [i["id"] for i in items if i["test"] == test and i["edit"] in keep]
                for key in sorted(averaged):
                    if key == (candidate, "ontology"):
                        continue
                    common = [i for i in ids if i in averaged[(candidate, "ontology")] and i in averaged[key]
                              and "after" in averaged[key][i] and "before" in averaged[key][i]]
                    if not common:
                        continue
                    a, b = averaged[(candidate, "ontology")], averaged[key]
                    sign = -1.0 if test == "neighbourhood" else 1.0      # neighbourhood: success = keeping the true filler
                    success = zs.paired_difference(np.asarray([sign * (a[i]["success"] - b[i]["success"]) for i in common]),
                                                   np.zeros(len(common)), resamples=resamples, seed=seed)
                    magnitude = zs.paired_difference(np.asarray([a[i]["after"] - a[i]["before"] for i in common]),
                                                     np.asarray([b[i]["after"] - b[i]["before"] for i in common]),
                                                     resamples=resamples, seed=seed)
                    block.append({"test": test, "reference": f"{key[0]} {key[1]}", "n": len(common), "success": success,
                                  "magnitude": magnitude})
            for row, adjusted in zip(block, holm_adjust([r["success"]["p_value"] for r in block]) if block else []):
                row.update(p_holm=adjusted, significant=adjusted < 0.05)
            comparisons[subset] = block
    return {"models": per_model, "candidate": candidate, "comparisons": comparisons}


def render_stage(stage: dict[str, Any], *, heading: str = "###") -> list[str]:
    """Report section of `stage_summary` (e9_report's dimension-3 baselines block)."""
    models = stage.get("models") or {}
    if not models:
        return ["No dimension-3 baseline evaluations (`e9_plan --dim3-baselines`).", ""]
    lines = ["Text evidence (in-context frames, IKE) is reported apart from the structure-only rows; weight editors run on the "
             "host weights (C0′, P0). Seeds are listed per row; intervals cover items only.", ""]
    rows = []
    for model, by_seed in models.items():
        for s, document in sorted(by_seed.items()):
            block = document.get("edits") or {}
            for method in EDIT_METHODS:
                e = (block.get(method) or {}).get("subsets", {}).get("all")
                if not e or not e.get("efficacy", {}).get("n"):
                    continue
                intra = ((block.get(method) or {}).get("intra") or {}).get("all", {})
                eff, par, nb = e["efficacy"], e.get("paraphrase", {}), e.get("neighbourhood", {})
                rows.append(f"| {model} | {s} | {method} | {e['edits']} | {fmt(eff['success_before'], 3)} → {fmt(eff['success_after'], 3)} | "
                            f"{fmt_ci(eff['magnitude'], 3)} | {fmt_ci(eff['target_minus_control'], 3)} | {fmt(par.get('success_after'), 3)} | "
                            f"{fmt(nb.get('success_after'), 3)} | {fmt(e.get('score'), 3)} | {fmt(intra.get('locality_after'), 3)} |")
    if rows:
        lines += [f"{heading} Edits: ontology edit vs IKE vs ROME / MEMIT / AlphaEdit (all linked edits)", "",
                  "| Model | Seed | Method | edits | ES before → after | EM [95% CI] | EM − control [95% CI] | PS | NS | score | intra locality |",
                  "|---|---:|---|---:|---|---|---|---:|---:|---:|---:|"] + rows + [""]
    for subset, block in (stage.get("comparisons") or {}).items():
        if block:
            lines += [f"{stage['candidate']} ontology edit − reference, {subset} edits (paired over items, seed-averaged; success = share "
                      "preferring the new filler, for neighbourhood keeping the true one; Holm over the rows):", "",
                      "| Test | Reference | n | Δ success [95% CI] | Δ EM [95% CI] | p (Holm) |", "|---|---|---:|---|---|---:|"]
            lines += [f"| {r['test']} | {r['reference']} | {r['n']} | {fmt_ci(r['success'], 3)} | {fmt_ci(r['magnitude'], 3)} | "
                      f"{fmt(r.get('p_holm'), 4)} |" for r in block] + [""]
    context_rows, row_rows, other = [], [], []
    for model, by_seed in models.items():
        for s, document in sorted(by_seed.items()):
            new = document.get("new_words") or {}
            for key, target in (("context", context_rows), ("rows", row_rows)):
                summary = new.get(key)
                if not summary:
                    continue
                for source, by_test in summary["sources"].items():
                    raw = (summary.get("raw_correct") or {}).get(source)
                    target.append(f"| {model} | {s} | {source} | " + " | ".join(
                        fmt(by_test.get(t, {}).get("mean"), 3) for t in ("property", "property_new", "entailment", "statement_accuracy"))
                        + (f" | {fmt(raw, 3)} |" if key == "context" else " |"))
            block = document.get("edits") or {}
            if block.get("transplant") or block.get("channel_off"):
                t = ((block.get("transplant") or {}).get("subsets") or {}).get("all", {})
                edited, everything = t.get("edited_relation", {}), t.get("all_relations", {})
                off = block.get("channel_off") or {}
                other.append(f"| {model} | {s} | {fmt(edited.get('follow_before'), 3)} → {fmt(edited.get('follow_after'), 3)} "
                             f"({fmt(edited.get('follow_control'), 3)}) | {fmt(everything.get('follow_before'), 3)} → "
                             f"{fmt(everything.get('follow_after'), 3)} | {fmt(off.get('revert_max_abs'), 5)} | "
                             f"{fmt(off.get('old_fact_on'), 3)} / {fmt(off.get('old_fact_off'), 3)} |")
    if context_rows:
        lines += [f"{heading} New words with the verbalized frame in context (text evidence)", "",
                  "| Model | Seed | Source | property | property_new | entailment | statement acc | raw property |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"] + context_rows + [""]
    if row_rows:
        lines += [f"{heading} New words: zero-shot rows at the channel site", "",
                  "| Model | Seed | Source | property | property_new | entailment | statement acc |", "|---|---:|---|---:|---:|---:|---:|"] + row_rows + [""]
    if other:
        lines += [f"{heading} Frame transplant and channel-off audit", "",
                  "| Model | Seed | follow B, edited relation: before → after (control) | follow B, all relations | channel-off revert max |Δd| | "
                  "old fact preferred, channel on / off |", "|---|---:|---|---|---:|---|"] + other + [""]
    return lines


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else list(METHODS)
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {', '.join(METHODS)}")
    config = {"experiment": "e9-dim3-baselines", "run": str(args.run), "checkpoint": args.checkpoint,
              "new_items": str(args.new_items) if args.new_items else None, "edit_items": str(args.edit_items) if args.edit_items else None,
              "methods": methods, "seed": args.seed, "resamples": args.resamples, "fit_entries": args.fit_entries,
              "cov_tokens": args.cov_tokens, "ike_demos": args.ike_demos, "intra_relations": args.intra_relations,
              "edit_limit": args.edit_limit, "new_limit": args.new_limit, "alias_table": str(args.alias_table) if args.alias_table else None,
              "nullspace_relative": args.nullspace_relative, "max_length": args.max_length}
    if args.overwrite:
        clear_output(args.output)
    git_at_start = start_output(args.output, config)
    torch.manual_seed(args.seed)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table,
                   max_length=args.max_length)
    settings = {"alphaedit": ke.EditorSettings(method="alphaedit", nullspace_relative=args.nullspace_relative)} \
        if args.nullspace_relative is not None else None
    out = evaluate(run, new_items=args.new_items, edit_items=args.edit_items, methods=methods, seed=args.seed, resamples=args.resamples,
                   fit_entries=args.fit_entries, cov_tokens=args.cov_tokens, ike_demos=args.ike_demos,
                   intra_relations=args.intra_relations, edit_limit=args.edit_limit, new_limit=args.new_limit, settings=settings)
    header = {"source": run.describe()}
    document = {**header, "new_items_path": config["new_items"], "edit_items_path": config["edit_items"], **out["document"]}
    with (args.output / "predictions.jsonl").open("w") as handle:
        for row in out["predictions"]:
            handle.write(json.dumps(json_ready(row)) + "\n")
    write_json(args.output / "summary.json", document)
    (args.output / "report.md").write_text(render(header, out["document"]))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


# -- toy verification on a known fact ---------------------------------------------------------------------------------

TOY_FACTS = (
    # (rewrite prompt, subject, true target, new target, paraphrase prompt, neighbour prompt with the same true target)
    ("{x} is located in the city of", "The Eiffel Tower", " Paris", " Rome", "{x} can be found in the city of",
     "The Louvre is located in the city of"),
    ("{x} is located in the country of", "The Great Wall", " China", " Brazil", "You can visit {x} in the country of",
     "The Forbidden City is located in the country of"),
    ("The mother tongue of {x} is", "Danielle Darrieux", " French", " English", "{x} grew up speaking",
     "The mother tongue of Léon Blum is"),
    ("{x} was developed by", "The iPhone", " Apple", " Nokia", "The company behind {x} is", "The iPad was developed by"),
)


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    """ROME / MEMIT / AlphaEdit on a pretrained host for well-known facts (CounterFact-style): the host must
    prefer the true target before, the new one after (efficacy, paraphrase), and keep the neighbour's."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..evaluation.probes import ModelAdapter
    config = {"experiment": "e9-dim3-toy", "model": args.model, "cov_texts": args.cov_texts, "methods": list(ke.METHODS),
              "nullspace_relative": args.nullspace_relative, "facts": [list(f) for f in TOY_FACTS]}
    git_at_start = start_output(args.output, config)
    device = torch.device(args.device or "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, attn_implementation="sdpa").to(device).eval()
    adapter = ModelAdapter(model, tokenizer, device, batch_size=8, max_length=128)
    host = ke.EditableHost(adapter)
    sample = [f"{s} {t.strip()}." for _, s, t, _, _, _ in TOY_FACTS]
    texts = (list(args.cov_text) if args.cov_text else []) + _toy_corpus(args.cov_texts)
    rows = []
    for method in ke.METHODS:
        settings = ke.EditorSettings(method=method, nullspace_relative=args.nullspace_relative if method == "alphaedit" else None)
        layers = ke.edit_layers(host, settings)
        moments = ke.second_moments(host, texts, layers)
        saved = host.snapshot(layers)
        requests = [ke.EditRequest(p, s, new, s) for p, s, _, new, _, _ in TOY_FACTS]

        def margins() -> list[dict[str, float]]:
            out = []
            for prompt, subject, true, new, para, neighbour in TOY_FACTS:
                margin = lambda text: ke.sequence_logprob(host, text, new) - ke.sequence_logprob(host, text, true)
                out.append({"efficacy": margin(prompt.format(x=subject)), "paraphrase": margin(para.format(x=subject)),
                            "neighbour": margin(neighbour)})
            return out

        before = margins()
        started = time.monotonic()
        info = (ke.apply_edits(host, requests, settings, moments) if method != "rome" else None)
        if method == "rome":
            after = []
            for request in requests:
                ke.apply_rome(host, request, settings, moments)
                after.append(margins()[requests.index(request)])
                host.restore(saved)
        else:
            after = margins()
            host.restore(saved)
        seconds = time.monotonic() - started
        for fact, b, a in zip(TOY_FACTS, before, after):
            rows.append({"method": method, "subject": fact[1], "true": fact[2], "new": fact[3], "layers": list(layers),
                         **{f"{k}_before": v for k, v in b.items()}, **{f"{k}_after": v for k, v in a.items()},
                         "seconds": seconds, **({"null_dims": [l.get("null_dim") for l in info["per_layer"]]} if info and method == "alphaedit" else {})})
    summary = {"rows": rows, "sample": sample,
               "checks": {m: {"knew_true_before": all(r["efficacy_before"] < 0 for r in rows if r["method"] == m),
                              "efficacy": float(np.mean([r["efficacy_after"] > 0 for r in rows if r["method"] == m])),
                              "paraphrase": float(np.mean([r["paraphrase_after"] > 0 for r in rows if r["method"] == m])),
                              "neighbour_kept": float(np.mean([r["neighbour_after"] < 0 for r in rows if r["method"] == m]))}
                          for m in ke.METHODS}}
    write_json(args.output / "summary.json", summary)
    lines = [f"# Weight-editing toy check — {args.model}", "",
             "Margins are log p(new) − log p(true) (summed over target tokens); efficacy/paraphrase should turn positive after "
             "the edit, the neighbour should stay negative. ROME edits one fact at a time; MEMIT/AlphaEdit edit all four at once.", "",
             "| method | subject | true → new | efficacy before → after | paraphrase before → after | neighbour before → after |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['subject']} | {r['true'].strip()} → {r['new'].strip()} | {r['efficacy_before']:+.2f} → "
                     f"{r['efficacy_after']:+.2f} | {r['paraphrase_before']:+.2f} → {r['paraphrase_after']:+.2f} | "
                     f"{r['neighbour_before']:+.2f} → {r['neighbour_after']:+.2f} |")
    lines += ["", "| method | knew the true facts | efficacy | paraphrase | neighbour kept |", "|---|---|---:|---:|---:|"]
    for m, c in summary["checks"].items():
        lines.append(f"| {m} | {c['knew_true_before']} | {c['efficacy']:.2f} | {c['paraphrase']:.2f} | {c['neighbour_kept']:.2f} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    finish_output(args.output, config, git_at_start=git_at_start, device=device)
    return summary


def _toy_corpus(count: int) -> list[str]:
    """`count` general-English sentences for the toy's second moments (WordNet glosses, deterministic)."""
    from nltk.corpus import wordnet
    out = []
    for synset in wordnet.all_synsets():
        gloss = synset.definition()
        examples = synset.examples()
        out.append(f"{gloss.capitalize()}. {' '.join(examples)}".strip())
        if len(out) >= count:
            break
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="dimension-3 baselines on a trained E9 run")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--new-items", type=Path, default=None); ev.add_argument("--edit-items", type=Path, default=None)
    ev.add_argument("--methods", default="", help="comma-separated subset of " + ",".join(METHODS) + " (default: all)")
    ev.add_argument("--checkpoint", default="final.pt"); ev.add_argument("--alias-table", type=Path, default=None)
    ev.add_argument("--seed", type=int, default=0); ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--fit-entries", type=int, default=4000)
    ev.add_argument("--cov-tokens", type=int, default=200_000, help="tokens of the training stream for the editors' second moments")
    ev.add_argument("--ike-demos", type=int, default=4); ev.add_argument("--intra-relations", type=int, default=3)
    ev.add_argument("--edit-limit", type=int, default=None, help="first N edited concepts only (smoke tests)")
    ev.add_argument("--new-limit", type=int, default=None, help="first N new words only (smoke tests)")
    ev.add_argument("--nullspace-relative", type=float, default=None,
                    help="AlphaEdit: null space = eigenvalues below this fraction of the largest (default: the absolute 2e-2)")
    ev.add_argument("--batch-size", type=int, default=32); ev.add_argument("--device", default=None)
    ev.add_argument("--max-length", type=int, default=256, help="longest scored text in tokens (IKE contexts are ≈ 200)")
    ev.add_argument("--overwrite", action="store_true", help="replace the result files of a previous evaluation in --output")
    toy = sub.add_parser("toy", help="weight editors on well-known facts of a pretrained host (implementation check)")
    toy.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M"); toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--cov-texts", type=int, default=2000, help="WordNet gloss sentences for the second moments")
    toy.add_argument("--cov-text", nargs="*", default=None, help=argparse.SUPPRESS)
    toy.add_argument("--nullspace-relative", type=float, default=None); toy.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "toy":
        summary = run_toy(args)
        print(json.dumps(summary["checks"], indent=2))
        return
    document = run_evaluate(args)
    print(json.dumps({"methods": document["methods"], "skipped": document["skipped"], "timing": document["timing"]}, indent=2))


if __name__ == "__main__":
    main()
