"""E14 RD — reading facts from real text (pre-registration `experiments/e14-logic/preregistration.md`; plan
`manuscript/proposals/2026-10-17-vsa-logic-experiment-plan.md` §8; proposal §9).

`recall` — **RD-4 / RD-7, the route by which read facts are used.** E11 composed a reader's frame into the term's vector
(the *implicit* route) and found that even the gold frame does not help on real ChEBI terms. Here the same frames are
also given *explicitly*: written before every item as E12 recall lines (`explicit-sym`, confidence 1.00), or composed in
the run's own store and decoded back (`explicit-vsa`, E12's recall). Conditions (`--conditions`, default all that the
run supports):

- `none` — no frame, no text (E11's `none`);
- `implicit:<reader>` — the reader's frame composed into the term's row (compose channels only);
- `explicit-sym:<reader>` — no frame; the reader's frame as recall lines in the prompt;
- `explicit-vsa:<reader>` — no frame; the frame composed in the run's store and decoded (slot-aware, typed clean-up);
- `both:<reader>` — the frame composed and its recall lines in the prompt;
- `definition` — no frame; the definition in the prompt (E11's `context`);
- `noisy-explicit:<p>` / `noisy-implicit:<p>` (RD-7) — the gold frame with a fraction `p` of its edges given a wrong filler
  observed under the same relation (seeded per concept), explicit or implicit.

Readers' frames come from the run's E11 output (`frames.jsonl`, default `RUN/e11-<kind>-<track>/`; `--frames` for a host
without its own, e.g. C0′ reading C5's). Items and tests are E11's (`ItemScorer`, `test_values`), so values are directly
comparable to E11's. `recall-report` pools seeds (the R9 convention: item differences of every seed pooled).

    python -m vsa_embed.experiments.e14_reading recall --run RUN --items DIR --output OUT [--frames FRAMES.jsonl]
        [--style chebi] [--conditions ...] [--noise 0.1,0.3,0.5] [--alias-table T.json] [--limit N] [--smoke]
    python -m vsa_embed.experiments.e14_reading recall-report --runs OUT ... --output DIR
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import self_query as sq
from ..statistics import holm_adjust
from . import e11_read_to_learn as e11
from .e5_common import clear_output, finish_output, json_ready, open_run, start_output, write_json

READERS = ("oracle", "stated", "typeprior", "linker")
VSA_READERS = ("oracle", "stated", "linker")
NOISE = (0.1, 0.3, 0.5)
PRIMARY = (("explicit-sym:oracle", "none"), ("explicit-sym:oracle", "implicit:oracle"), ("explicit-sym:stated", "implicit:stated"))
SECONDARY = (("implicit:oracle", "none"), ("implicit:linker", "none"), ("explicit-sym:linker", "none"),
             ("explicit-sym:typeprior", "none"), ("explicit-sym:stated", "none"), ("explicit-vsa:oracle", "explicit-sym:oracle"),
             ("explicit-vsa:oracle", "none"), ("explicit-vsa:linker", "none"), ("both:oracle", "explicit-sym:oracle"),
             ("both:oracle", "implicit:oracle"), ("definition", "explicit-sym:oracle"), ("definition", "none"),
             ("explicit-sym:linker", "implicit:linker"), ("explicit-sym:typeprior", "implicit:typeprior"))
PRIMARY_TESTS = ("property", "paraphrase", "entailment")


# -- frames ---------------------------------------------------------------------------------------------------------

def load_reader_frames(path: Path, style: str, readers: Sequence[str], ctx: e11.TrackContext) -> dict[str, dict[str, list | None]]:
    """reader → concept → frame (relation id, atom id) pairs, from an E11 `frames.jsonl` (rows of the given style)."""
    out: dict[str, dict[str, list | None]] = defaultdict(dict)
    for row in e11._read_jsonl(Path(path)):
        if row.get("style") != style or row.get("reader") not in readers:
            continue
        out[row["reader"]][row["concept"]] = ctx.ids(row["frame"]) if row.get("frame") is not None else None
    return out


def filler_pools(ontology: dict[str, Any]) -> dict[int, np.ndarray]:
    """relation → the atomics observed as its fillers in the ontology frames (the wrong-filler pool of RD-7)."""
    relations, fillers = np.asarray(ontology["relations"]), np.asarray(ontology["fillers"])
    return {int(r): np.unique(fillers[relations == r]) for r in np.unique(relations)}


def noisy_frame(frame: Sequence[tuple[int, int]], p: float, pools: dict[int, np.ndarray], *, seed: int, key: str) -> list:
    """`frame` with each edge's filler replaced, with probability `p`, by another filler of the same relation (seeded per
    concept `key`, so every host sees the same corruption). An edge whose relation has no other filler is kept."""
    rng = random.Random(f"{seed}:{key}:{p}")
    out = []
    for r, a in frame:
        pool = pools.get(int(r))
        if rng.random() < p and pool is not None and len(pool) > 1:
            choice = int(pool[rng.randrange(len(pool))])
            while choice == int(a):
                choice = int(pool[rng.randrange(len(pool))])
            out.append((int(r), choice))
        else:
            out.append((int(r), int(a)))
    return out


# -- recall text ----------------------------------------------------------------------------------------------------

class RecallText:
    """Writes a frame as E12 recall lines: symbolic (the frame itself, confidence 1.00) or through the run's store (composed,
    decoded with the slot list, typed clean-up). `store` is a `self_query.RecallStore` or None (symbolic only)."""

    def __init__(self, ontology: dict[str, Any], lexicon: Any, store: sq.RecallStore | None = None) -> None:
        self.store = store
        home = store.home.tolist() if store is not None else None
        self.writer = sq.RecallWriter(ontology["relation_names"], ontology["atomic_names"], lexicon, style="statements",
                                      confidence=True, home=home)

    def symbolic(self, subject: str, frame: Sequence[tuple[int, int]] | None) -> str:
        return self.writer.render(subject, sq.symbolic_lines(list(frame or [])))

    def decoded(self, subject: str, frame: Sequence[tuple[int, int]] | None) -> tuple[str, float | None]:
        """Recall text of the frame's store and the share of the frame's edges recovered (None for an empty frame)."""
        if self.store is None:
            raise ValueError("explicit-vsa needs the run's store")
        frame = list(frame or [])
        if not frame:
            return self.writer.render(subject, []), None
        vector = self.store.frame_vector(frame)
        lines = self.store.decode_slots(vector, [r for r, _ in frame])
        got = {(line.relation, f.atom) for line in lines for f in line.fillers}
        return self.writer.render(subject, lines), sum((r, a) in got for r, a in set(frame)) / len(set(frame))


# -- conditions -------------------------------------------------------------------------------------------------------

def default_conditions(compose: bool, vsa: bool, noise: Sequence[float]) -> list[str]:
    out = ["none"] + [f"explicit-sym:{r}" for r in READERS] + ["definition"] + [f"noisy-explicit:{p:g}" for p in noise]
    if compose:
        out += [f"implicit:{r}" for r in READERS] + ["both:oracle"] + [f"noisy-implicit:{p:g}" for p in noise]
    if vsa:
        out += [f"explicit-vsa:{r}" for r in VSA_READERS]
    return out


def condition_inputs(name: str, concepts: Sequence[str], frames: dict[str, dict[str, list | None]], surface: dict[str, str],
                     text: RecallText, definitions: dict[str, str], pools: dict[int, np.ndarray], *, seed: int
                     ) -> tuple[dict[str, list | None], dict[str, str] | None, dict[str, Any]]:
    """(frames for the term rows, prompt contexts, diagnostics) of one condition."""
    kind, _, arg = name.partition(":")
    none = {c: None for c in concepts}
    info: dict[str, Any] = {}
    if kind == "none":
        return none, None, info
    if kind == "definition":
        return none, {c: definitions[c] for c in concepts if c in definitions}, info
    if kind in {"noisy-explicit", "noisy-implicit"}:
        p = float(arg)
        gold = frames["oracle"]
        noisy = {c: noisy_frame(gold[c], p, pools, seed=seed, key=c) if gold.get(c) else None for c in concepts}
        changed = [sum(x != y for x, y in zip(gold[c], noisy[c])) / len(gold[c]) for c in concepts if gold.get(c)]
        info["corrupted_share"] = float(np.mean(changed)) if changed else None
        if kind == "noisy-implicit":
            return noisy, None, info
        return none, {c: text.symbolic(surface[c], noisy[c]) for c in concepts}, info
    reader = frames.get(arg)
    if reader is None:
        raise KeyError(f"condition {name!r}: no {arg!r} frames")
    own = {c: reader.get(c) for c in concepts}
    if kind == "implicit":
        return own, None, info
    if kind == "explicit-sym":
        return none, {c: text.symbolic(surface[c], own[c]) for c in concepts}, info
    if kind == "both":
        return own, {c: text.symbolic(surface[c], own[c]) for c in concepts}, info
    if kind == "explicit-vsa":
        contexts, recovered = {}, []
        for c in concepts:
            contexts[c], share = text.decoded(surface[c], own[c])
            if share is not None:
                recovered.append(share)
        info["decode_recovered"] = float(np.mean(recovered)) if recovered else None
        return none, contexts, info
    raise ValueError(f"unknown condition {name!r}")


# -- RD-4b (pre-registration §13): recall of the item's relation only, and a null prompt without the context ------------

RD4B_CONDITIONS = ("none", "explicit-sym:oracle", "explicit-rel:oracle", "explicit-rel:stated", "explicit-rel:typeprior",
                   "explicit-rel:linker", "explicit-rel-wrong:oracle", "definition")
RD4B_COMPOSE = ("implicit:oracle",)
PRIMARY_B = (("explicit-rel:oracle", "none"), ("explicit-rel:oracle", "explicit-sym:oracle"),
             ("explicit-rel:oracle", "implicit:oracle"))
SECONDARY_B = (("explicit-sym:oracle", "none"), ("explicit-rel:stated", "none"), ("explicit-rel:typeprior", "none"),
               ("explicit-rel:linker", "none"), ("explicit-rel-wrong:oracle", "none"),
               ("explicit-rel:oracle", "explicit-rel-wrong:oracle"), ("definition", "explicit-rel:oracle"), ("definition", "none"))
STATED_B = (("explicit-rel:stated", "none"), ("definition", "none"), ("explicit-rel:stated", "definition"))

# RD-4c (pre-registration §16): routing without the item's label, Claude's frames, larger hosts.
RD4C_CONDITIONS = ("none", "explicit-sym:oracle", "explicit-rel:oracle", "explicit-rel:typeprior", "explicit-rel-wrong:oracle",
                   *(f"explicit-route:{r}:{reader}" for reader in ("oracle", "typeprior") for r in ("lex", "lm", "top2")),
                   "definition")
RD4C_TEACHER = ("explicit-rel:teacher", "explicit-route:lm:teacher")
PRIMARY_C = (("explicit-route:lm:typeprior", "none"), ("explicit-route:lm:oracle", "explicit-rel:oracle"))
SECONDARY_C = (("explicit-route:lex:oracle", "explicit-rel:oracle"), ("explicit-route:top2:oracle", "explicit-rel:oracle"),
               ("explicit-route:lm:oracle", "explicit-sym:oracle"), ("explicit-route:lm:typeprior", "explicit-rel:typeprior"),
               ("explicit-route:lex:typeprior", "none"), ("explicit-route:top2:typeprior", "none"),
               ("explicit-route:lm:typeprior", "definition"), ("explicit-rel:teacher", "explicit-rel:typeprior"),
               ("explicit-rel:teacher", "none"), ("explicit-route:lm:teacher", "none"), ("explicit-route:lm:teacher", "definition"))
PRIMARY_HOST = (("explicit-rel:oracle", "none"), ("explicit-rel:oracle", "explicit-sym:oracle"))
# RD-12c (pre-registration §17): the RD-12 cycle's read frames under selective recall (the route that works).
RD12C_CONDITIONS = ("none", "explicit-rel:oracle", "explicit-rel:typeprior", "explicit-rel:rd12-single-r2",
                    "explicit-rel:rd12-cycle-r2", "explicit-sym:rd12-single-r2", "explicit-sym:rd12-cycle-r2", "definition")
RD12C_READERS = ("rd12-single-r2", "rd12-cycle-r0", "rd12-cycle-r2", "rd12-gold-r2")
PRIMARY_D = (("explicit-rel:rd12-cycle-r2", "explicit-rel:rd12-single-r2"),)
SECONDARY_D = (("explicit-rel:rd12-cycle-r2", "none"), ("explicit-rel:rd12-single-r2", "none"),
               ("explicit-rel:rd12-cycle-r2", "explicit-rel:typeprior"), ("explicit-rel:oracle", "explicit-rel:rd12-cycle-r2"),
               ("definition", "explicit-rel:rd12-cycle-r2"), ("explicit-sym:rd12-cycle-r2", "explicit-sym:rd12-single-r2"))
SECONDARY_HOST = (("explicit-route:lm:typeprior", "none"), ("explicit-route:lm:typeprior", "definition"),
                  ("explicit-rel:typeprior", "none"), ("explicit-rel-wrong:oracle", "none"), ("definition", "none"))


def _norm_answer(text: str | None) -> str:
    t = (text or "").strip().lower().rstrip(".").strip()
    for article in ("a ", "an ", "the "):
        if t.startswith(article):
            t = t[len(article):]
    return t


def gold_flags(items: Sequence[dict[str, Any]], frames: dict[str, dict[str, list | None]], ctx: e11.TrackContext
               ) -> dict[str, dict[str, bool]]:
    """item id → {reader: the item's gold answer is a partner of the item's relation in that reader's frame} (property and
    entailment items; answers compared as lower-case text without article or full stop)."""
    rid = ctx.relation_id
    out: dict[str, dict[str, bool]] = {}
    for item in items:
        if item["test"] not in {"property", "entailment"} or item.get("relation") not in rid:
            continue
        r, gold = rid[item["relation"]], _norm_answer(item["candidates"][int(item["gold"])])
        out[item["id"]] = {reader: any(int(e[0]) == r and _norm_answer(ctx.text(int(e[1]))) == gold for e in (by.get(item["concept"]) or []))
                           for reader, by in frames.items()}
    return out


def item_condition_inputs(name: str, scorer: Any, frames: dict[str, dict[str, list | None]], text: RecallText,
                          definitions: dict[str, str], pools: dict[int, np.ndarray], ctx: e11.TrackContext, *, seed: int
                          ) -> tuple[dict[str, list | None], dict[str, str] | None, dict[str, Any]]:
    """(frames for the term rows, per-item prompt contexts, diagnostics). `explicit-rel:<reader>`: the reader's edges of the
    item's own relation as recall lines (an item without a relation gets the whole frame); `explicit-rel-wrong:oracle`: the
    same lines with every partner replaced by another filler of that relation (RD-7's pool, p = 1). Other conditions: the
    concept's context (`condition_inputs`) for each of its items."""
    concepts = scorer.concepts
    kind, _, arg = name.partition(":")
    if kind == "explicit-route":                        # RD-4c: `explicit-route:<router>:<reader>`
        router, _, reader_name = arg.partition(":")
        if router not in ROUTERS:
            raise ValueError(f"condition {name!r}: router must be one of {ROUTERS}")
        reader = frames.get(reader_name)
        if reader is None:
            raise KeyError(f"condition {name!r}: no {reader_name!r} frames")
        chosen = route_relations(scorer, reader, ctx, router)
        rid = ctx.relation_id
        out, lines, hits = {}, [], []
        for c in concepts:
            frame = list(reader.get(c) or [])
            held = {int(r) for r, _ in frame}
            for item in scorer.items_by_concept[c]:
                pick = set(chosen.get(item["id"], []))
                edges = [e for e in frame if int(e[0]) in pick]
                lines.append(len(edges))
                asked = rid.get(item.get("relation"))
                if asked is not None and asked in held:
                    hits.append(float(asked in pick))
                out[item["id"]] = text.symbolic(scorer.surface[c], edges)
        return {c: None for c in concepts}, out, {"lines_per_item": float(np.mean(lines)) if lines else 0.0,
                                                  "route_accuracy": float(np.mean(hits)) if hits else None,
                                                  "routable_items": len(hits)}
    if kind not in {"explicit-rel", "explicit-rel-wrong"}:
        row_frames, contexts, info = condition_inputs(name, concepts, frames, scorer.surface, text, definitions, pools, seed=seed)
        if not contexts:
            return row_frames, None, info
        return row_frames, {i["id"]: contexts[c] for c in concepts if c in contexts for i in scorer.items_by_concept[c]}, info
    reader = frames.get(arg)
    if reader is None:
        raise KeyError(f"condition {name!r}: no {arg!r} frames")
    rid = ctx.relation_id
    out, lines, changed = {}, [], []
    for c in concepts:
        frame = list(reader.get(c) or [])
        for item in scorer.items_by_concept[c]:
            rel = rid.get(item.get("relation"))
            edges = [e for e in frame if rel is None or int(e[0]) == rel]
            if kind == "explicit-rel-wrong":
                wrong = noisy_frame(edges, 1.0, pools, seed=seed, key=f"{c}:{item['id']}")
                changed += [float(x != y) for x, y in zip(edges, wrong)]
                edges = wrong
            lines.append(len(edges))
            out[item["id"]] = text.symbolic(scorer.surface[c], edges)
    info: dict[str, Any] = {"lines_per_item": float(np.mean(lines)) if lines else 0.0,
                            "items_without_lines": int(sum(n == 0 for n in lines))}
    if changed:
        info["corrupted_share"] = float(np.mean(changed))
    return {c: None for c in concepts}, out, info


# -- RD-4c (pre-registration §16): choosing which relation to recall, without the item's label ---------------------------

ROUTERS = ("lex", "lm", "top2")


def _route_phrase(name: str) -> str:
    from .. import self_query as sqm
    return sqm.statement_phrase(name)


def route_relations(scorer: Any, frames: dict[str, list | None], ctx: e11.TrackContext, router: str) -> dict[str, list[int]]:
    """item id → the relation ids to recall, chosen from the relations the term's frame holds, without the item's
    relation label. `lex`: the relation whose phrase (`statement_phrase`, e.g. "has role") shares most words with the item's
    templates; `lm`: the host's choice — the summed length-normalised log-probability over the item's templates of
    " <phrase>" after "Question: <template with the term> ...\\nThe fact needed is that <term>"; `top2`: `lm`'s best two."""
    from ..evaluation import channel_probes as cp
    names = list(ctx.ontology["relation_names"])
    out: dict[str, list[int]] = {}
    jobs: list[tuple[str, int, str, str]] = []
    for c in scorer.concepts:
        options = sorted({int(r) for r, _ in (frames.get(c) or [])})
        for item in scorer.items_by_concept[c]:
            if len(options) <= 1:
                out[item["id"]] = options
                continue
            if router == "lex":
                words = set(re.findall(r"[a-z]+", " ".join(item["templates"]).lower()))
                overlap = [len(words & set(re.findall(r"[a-z]+", _route_phrase(names[r]).lower()))) for r in options]
                out[item["id"]] = [options[int(np.argmax(overlap))]]
                continue
            surface = scorer.surface[c]
            for t in item["templates"]:
                question = t.replace("{x}", surface)
                for r in options:
                    jobs.append((item["id"], r, f"Question: {question} ...\nThe fact needed is that {surface}", " " + _route_phrase(names[r])))
    if jobs:
        with scorer.condition({c: None for c in scorer.concepts}), e11.fast_continuations():
            values = cp.continuation_logprob(scorer.adapter, [j[2] for j in jobs], [j[3] for j in jobs])
        lengths = [max(1, len(scorer.run.tokenizer(j[3], add_special_tokens=False)["input_ids"])) for j in jobs]
        totals: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for (iid, r, _, _), v, n in zip(jobs, values, lengths):
            totals[iid][r] += float(v) / n
        for iid, by in totals.items():
            ranked = sorted(by, key=lambda r: (-by[r], r))
            out[iid] = ranked[:2] if router == "top2" else ranked[:1]
    return out


def plain_null_scores(scorer: Any) -> np.ndarray:
    """Null-prompt scores of every property / entailment item **without** any context (the template with the null surface,
    e.g. "this is functionally related to" + candidate), in the item order `score_rows` uses; read with no term frames."""
    from ..evaluation import channel_probes as cp
    from . import e5_zeroshot as zs
    items = [i for c in scorer.concepts for i in scorer.items_by_concept[c] if i["test"] in {"property", "entailment"}]
    surfaces = {i["concept"]: i.get("null", zs.NULL_SURFACE) for i in items}
    prefixes, continuations, _ = zs._prompt_texts(items, surfaces)
    unique = sorted(set(zip(prefixes, continuations)))
    with scorer.condition({c: None for c in scorer.concepts}), e11.fast_continuations():
        values = dict(zip(unique, cp.continuation_logprob(scorer.adapter, [u[0] for u in unique], [u[1] for u in unique])))
    return np.asarray([values[(p, c)] for p, c in zip(prefixes, continuations)])


def score_rows(scorer: Any, row_frames: dict[str, list | None], item_contexts: dict[str, str] | None, *,
               null_scores: np.ndarray | None) -> dict[str, list[dict[str, Any]]]:
    """`ItemScorer.score` with per-item contexts and, if `null_scores` is given, that (context-free) null for PMI."""
    from . import e5_zeroshot as zs
    from .e9_dim3_baselines import with_context
    items = [i for c in scorer.concepts for i in scorer.items_by_concept[c]]
    if item_contexts:
        items = with_context(items, {i: t + "\n" for i, t in item_contexts.items()})
    prompt_items = [i for i in items if i["test"] in {"property", "entailment"}]
    statement_items = [i for i in items if i["test"] == "statement"]
    with scorer.condition(row_frames), e11._smaller(scorer.adapter, 2 if item_contexts else 1), e11.fast_continuations():
        prompts, _ = (zs.score_prompts(scorer.adapter, prompt_items, scorer.surface, null_scores=null_scores) if prompt_items
                      else ([], None))
        statements = e11.edit.score_statements(scorer.adapter, statement_items, scorer.surface) if statement_items else []
    kinds = {i["id"]: i.get("edge_kind") for i in items}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompts + statements:
        row = {k: v for k, v in row.items() if k != "pmi"}
        row["edge_kind"] = kinds.get(row["id"], row.get("edge_kind"))
        row.setdefault("consistent", None)
        out[row["concept"]].append(row)
    return {c: out.get(c, []) for c in scorer.concepts}


def frames_path(run_dir: Path, read_set: e11.ReadSet) -> Path:
    track = read_set.manifest["track"]
    return Path(run_dir) / f"e11-{track}-{read_set.kind}" / "frames.jsonl"


def summarize(item_rows: dict[str, dict[str, list[dict[str, Any]]]], kind: str, *, resamples: int, seed: int) -> dict[str, Any]:
    values = {cond: e11.test_values(rows, kind) for cond, rows in item_rows.items()}
    means = {cond: {t: (float(np.mean(list(v.values()))) if v else None, len(v)) for t, v in by_test.items()}
             for cond, by_test in values.items()}

    def block(pairs: Sequence[tuple[str, str]], holm: bool) -> list[dict[str, Any]]:
        rows = []
        for a, b in pairs:
            if a not in values or b not in values:
                continue
            for test in PRIMARY_TESTS:
                result = e11.compare(values[a].get(test, {}), values[b].get(test, {}), resamples=resamples, seed=seed)
                if result is not None:
                    rows.append({"a": a, "b": b, "test": test, **result})
        if holm and rows:
            for row, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
                row["p_holm"] = p
        return rows

    return {"means": means, "primary": block(PRIMARY, True), "secondary": block(SECONDARY, False)}


def fit_context(text: str, tokenizer: Any, budget: int | None) -> str:
    """`text` cut to whole lines within `budget` tokens (the first line, the recall call, is always kept)."""
    if budget is None or len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= budget:
        return text
    lines = text.split("\n")
    kept = lines[:1]
    for line in lines[1:]:
        if len(tokenizer("\n".join(kept + [line]), add_special_tokens=False)["input_ids"]) > budget:
            break
        kept.append(line)
    return "\n".join(kept)


def context_budget(run: Any, cap: int | None) -> int | None:
    """Tokens a prompt prefix may use: `cap`, and at most half the host's positions (the item prompt needs the rest)."""
    positions = e11.max_positions(run)
    limits = [x for x in (cap, positions // 2 if positions else None) if x]
    return min(limits) if limits else None


def evaluate_recall(run: Any, read_set: e11.ReadSet, frames: dict[str, dict[str, list | None]], *, style: str,
                    conditions: Sequence[str], store: sq.RecallStore | None, seed: int = 0, resamples: int = 2000,
                    max_context_tokens: int | None = 1024, null: str = "context", log=print) -> dict[str, Any]:
    """`null`: `context` (tier P: the null prompt is read after the same context as the item) or `plain` (RD-4b: the null
    prompt without any context, computed once). Per-item conditions (`explicit-rel*`) and `plain` use `score_rows`."""
    ctx = e11.run_context(run)
    budget = context_budget(run, max_context_tokens)
    scorer = e11.ItemScorer(run, read_set, ctx)
    concepts = scorer.concepts
    text = RecallText(run.ontology, ctx.lexicon, store)
    definitions = {d["concept"]: d["text"] for d in read_set.definitions if d["style"] == style}
    pools = filler_pools(run.ontology)
    item_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    info: dict[str, Any] = {}
    tokens: dict[str, float | None] = {}
    samples: dict[str, str] = {}
    timings: dict[str, float] = {}
    null_scores = plain_null_scores(scorer) if null == "plain" else None
    flags = gold_flags([i for c in concepts for i in scorer.items_by_concept[c]], frames, ctx)
    for name in conditions:
        t0 = time.monotonic()
        log(f"  {name}")
        if null == "plain" or name.startswith(("explicit-rel", "explicit-route")):
            row_frames, contexts, diag = item_condition_inputs(name, scorer, frames, text, definitions, pools, ctx, seed=seed)
            if contexts:
                contexts = {i: fit_context(t, run.tokenizer, budget) for i, t in contexts.items()}
            item_rows[name] = score_rows(scorer, row_frames, contexts, null_scores=null_scores)
        else:
            row_frames, contexts, diag = condition_inputs(name, concepts, frames, scorer.surface, text, definitions, pools, seed=seed)
            if contexts:
                contexts = {c: fit_context(t, run.tokenizer, budget) for c, t in contexts.items()}
            item_rows[name] = scorer.score(row_frames, contexts)
        for rows in item_rows[name].values():
            for row in rows:
                if row["id"] in flags:
                    row["gold_in"] = flags[row["id"]]
        info[name] = diag
        if contexts:
            sample = list(contexts.values())[:400]
            tokens[name] = float(np.mean([len(run.tokenizer(t, add_special_tokens=False)["input_ids"]) + 1 for t in sample]))
            samples[name] = sample[0]
        else:
            tokens[name] = 0.0
        timings[name] = round(time.monotonic() - t0, 2)
    gold_share = {r: float(np.mean([f[r] for f in flags.values()])) for r in frames} if flags else {}
    document = {"set": {"kind": read_set.kind, "track": read_set.manifest["track"], "concepts": len(concepts),
                        "items": len(read_set.items)}, "null": null, "gold_in_frame_share": gold_share,
                "style": style, "conditions": list(conditions), "compose": bool(run.channel is not None and run.channel.mode == "compose"),
                "vsa": store is not None, "context_budget": budget, "readers": {r: sum(f is not None for f in v.values()) for r, v in frames.items()},
                "diagnostics": info, "context_tokens": tokens, "context_samples": samples, "timings": timings,
                "items": summarize(item_rows, read_set.kind, resamples=resamples, seed=seed)}
    return {"document": document, "item_rows": item_rows}


def fmt_ci(row: dict[str, Any] | None) -> str:
    if not row:
        return "—"
    out = f"{row['mean']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
    return out + (f" (p {row['p_value']:.4f}" + (f", Holm {row['p_holm']:.4f})" if "p_holm" in row else ")"))


def render(document: dict[str, Any], *, smoke: bool) -> str:
    lines = (["**SMOKE TEST — not a result.**", ""] if smoke else []) + [
        f"# E14 RD-4 / RD-7 — implicit against explicit recall of read frames ({document['set']['track']} {document['set']['kind']})", "",
        f"Run: `{document.get('source', {}).get('run', '?')}`; style `{document['style']}`; {document['set']['concepts']} terms, "
        f"{document['set']['items']} items; compose channel: {document['compose']}; store decoding: {document['vsa']}.", "",
        "## Means", "", "| condition | property | paraphrase | entailment | statement | context tokens | note |", "|---|---:|---:|---:|---:|---:|---|"]
    means = document["items"]["means"]
    for cond in document["conditions"]:
        m = means.get(cond, {})
        cell = lambda t: f"{m[t][0]:.4f}" if t in m and m[t][0] is not None else "—"  # noqa: E731
        diag = document["diagnostics"].get(cond) or {}
        note = ", ".join(f"{k} {v:.3f}" for k, v in diag.items() if v is not None)
        tok = document["context_tokens"].get(cond)
        lines.append(f"| {cond} | {cell('property')} | {cell('paraphrase')} | {cell('entailment')} | {cell('statement_accuracy')} | "
                     f"{'' if tok is None else f'{tok:.1f}'} | {note} |")
    for title, key in (("Primary (Holm over contrasts × tests)", "primary"), ("Secondary (unadjusted)", "secondary")):
        lines += ["", f"## {title}", "", "| a − b | test | difference [95% CI] | n |", "|---|---|---|---:|"]
        for row in document["items"][key]:
            lines.append(f"| {row['a']} − {row['b']} | {row['test']} | {fmt_ci(row)} | {row['n']} |")
    if document.get("context_samples"):
        lines += ["", "## Prompt text, first term (examples)", ""]
        for cond, sample in list(document["context_samples"].items())[:6]:
            lines += [f"`{cond}`:", "", "```", sample[:800], "```", ""]
    return "\n".join(lines) + "\n"


def run_recall(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items).limit(args.limit)
    track = read_set.manifest["track"]
    style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    noise = tuple(float(x) for x in args.noise.split(",") if x.strip()) if args.noise else NOISE
    config = {"experiment": "e14-rd4-recall", "run": str(args.run), "items": str(args.items), "style": style, "limit": args.limit,
              "frames": str(args.frames) if args.frames else None, "conditions": args.conditions, "noise": list(noise),
              "seed": args.seed, "resamples": args.resamples, "max_context_tokens": args.max_context_tokens, "alias_table": str(args.alias_table) if args.alias_table else None,
              "smoke": bool(args.smoke), "null": args.null, "family": args.family,
              "teacher_frames": str(args.teacher_frames) if getattr(args, "teacher_frames", None) else None}
    output = Path(args.output)
    if args.overwrite:
        clear_output(output)
        (output / "predictions.jsonl").unlink(missing_ok=True)
    git_at_start = start_output(output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    ctx = e11.run_context(run)
    compose = run.channel is not None and run.channel.mode == "compose"
    store = None
    if compose and not args.no_vsa:
        from .e12_self_query import load_store
        store = load_store(Path(args.run), "own").store
    path = Path(args.frames) if args.frames else frames_path(Path(args.run), read_set)
    extra = [r.strip() for r in (getattr(args, "extra_readers", None) or "").split(",") if r.strip()]
    if args.family == "rd12c" and not extra:
        extra = list(RD12C_READERS)
    frames = load_reader_frames(path, style, (*READERS, *extra), ctx)
    missing = [r for r in READERS if r not in frames]
    if missing:
        raise SystemExit(f"{path}: no frames for readers {missing} (style {style})")
    if getattr(args, "teacher_frames", None):
        for row in e11._read_jsonl(Path(args.teacher_frames)):
            if row.get("style") == style and row.get("frame") is not None:
                frames["teacher"][row["concept"]] = ctx.ids(row["frame"])
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    elif args.family == "rd4b":
        conditions = list(RD4B_CONDITIONS) + (list(RD4B_COMPOSE) if compose else [])
    elif args.family == "rd4c":
        conditions = list(RD4C_CONDITIONS) + (list(RD4C_TEACHER) if args.teacher_frames else [])
    elif args.family == "rd12c":
        conditions = list(RD12C_CONDITIONS)
    else:
        conditions = default_conditions(compose, store is not None, noise)
    if not compose:
        bad = [c for c in conditions if c.split(":")[0] in {"implicit", "both", "noisy-implicit", "explicit-vsa"}]
        if bad:
            raise SystemExit(f"run has no compose channel: conditions {bad} need one")
    result = evaluate_recall(run, read_set, frames, style=style, conditions=conditions, store=store, seed=args.seed,
                             resamples=args.resamples, max_context_tokens=args.max_context_tokens, null=args.null)
    document = {"source": run.describe(), "frames_from": str(path), "smoke": bool(args.smoke), "family": args.family,
                **result["document"]}
    with (output / "predictions.jsonl").open("w") as handle:
        for cond, by_concept in result["item_rows"].items():
            for concept, rows in by_concept.items():
                for row in rows:
                    handle.write(json.dumps(json_ready({"condition": cond, **row})) + "\n")
    write_json(output / "summary.json", document)
    (output / "report.md").write_text(render(document, smoke=bool(args.smoke)))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


# -- pooled report ----------------------------------------------------------------------------------------------------

def _prediction_values(folder: Path, *, field: str = "correct", where=lambda row: True) -> dict[str, dict[str, dict[str, float]]]:
    """condition → test → item id → value, from a run folder's `predictions.jsonl` (paraphrase: the property items'
    `consistent`; with `field="raw_correct"` there is no paraphrase value)."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for line in (folder / "predictions.jsonl").open():
        row = json.loads(line)
        if not where(row):
            continue
        if row["test"] in {"property", "entailment"} and row.get(field) is not None:
            out[row["condition"]][row["test"]][row["id"]] = float(row[field])
        if row["test"] == "property" and field == "correct" and row.get("consistent") is not None:
            out[row["condition"]]["paraphrase"][row["id"]] = float(row["consistent"])
    return out


def pooled_contrast(folders: Sequence[Path], a: str, b: str, test: str, *, field: str = "correct", where=lambda row: True,
                    resamples: int, seed: int) -> dict[str, Any] | None:
    """a − b on one test, paired over items, the item differences of every seed pooled (as `e11.pooled_item_contrast`),
    restricted to the rows `where` keeps."""
    from .e5_zeroshot import paired_difference
    diffs = []
    for folder in folders:
        values = _prediction_values(folder, field=field, where=where)
        va, vb = values.get(a, {}).get(test, {}), values.get(b, {}).get(test, {})
        diffs += [va[i] - vb[i] for i in sorted(set(va) & set(vb))]
    if not diffs:
        return None
    return {**paired_difference(np.asarray(diffs), np.zeros(len(diffs)), resamples=resamples, seed=seed), "seeds": len(folders)}


def rd4b_group(folders: Sequence[Path], conds: set[str], *, resamples: int, seed: int, primary: Sequence = PRIMARY_B,
               secondary: Sequence = SECONDARY_B, stated_pairs: Sequence = STATED_B) -> dict[str, Any]:
    """RD-4b (pre-registration §13): primary (Holm over contrasts × tests), secondary, the stated-only subset, and the same
    contrasts on raw (uncalibrated) choices."""
    stated = lambda row: bool((row.get("gold_in") or {}).get("stated"))  # noqa: E731
    out: dict[str, Any] = {}
    blocks = (("primary", primary, "correct", lambda row: True), ("secondary", secondary, "correct", lambda row: True),
              ("stated_only", stated_pairs, "correct", stated), ("raw_primary", primary, "raw_correct", lambda row: True))
    for key, pairs, field, where in blocks:
        rows = []
        for a, b in pairs:
            if a not in conds or b not in conds:
                continue
            for test in PRIMARY_TESTS:
                r = pooled_contrast(folders, a, b, test, field=field, where=where, resamples=resamples, seed=seed)
                if r is not None:
                    rows.append({"a": a, "b": b, "test": test, **r})
        if key == "primary" and rows:
            for row, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
                row["p_holm"] = p
        out[key] = rows
    means: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in folders:
        for field, suffix in (("correct", ""), ("raw_correct", " raw")):
            for cond, by_test in _prediction_values(f, field=field).items():
                for test, values in by_test.items():
                    means[cond][test + suffix].append(float(np.mean(list(values.values()))))
            for cond, by_test in _prediction_values(f, field=field, where=stated).items():
                for test, values in by_test.items():
                    means[cond][test + suffix + " (stated)"].append(float(np.mean(list(values.values()))))
    out["means"] = {c: {t: float(np.mean(v)) for t, v in by.items()} for c, by in means.items()}
    out["stated_items"] = int(sum(1 for line in (folders[0] / "predictions.jsonl").open()
                                  if (r := json.loads(line))["condition"] == "none" and r["test"] == "property" and stated(r)))
    return out


def run_recall_report(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "family", "tier-p") in {"rd4b", "rd4c", "rd12c"}:
        return run_rd4b_report(args)
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for folder in present:
        s = json.loads((folder / "summary.json").read_text())
        model = Path(s["source"]["run"]).name.rsplit("-s", 1)[0]
        groups[(s["set"]["track"], s["set"]["kind"], model)].append(folder)
    out: dict[str, Any] = {"groups": [], "missing": missing}
    lines = ["# E14 RD-4 / RD-7 — pooled report (item differences of every seed pooled)", ""]
    if missing:
        lines += [f"Missing run folders: {', '.join(missing)}", ""]
    for (track, kind, model), folders in sorted(groups.items()):
        conds = set.intersection(*(set(json.loads((f / "summary.json").read_text())["conditions"]) for f in folders))
        primary, secondary = [], []
        for pairs, sink in ((PRIMARY, primary), (SECONDARY, secondary)):
            for a, b in pairs:
                if a not in conds or b not in conds:
                    continue
                for test in PRIMARY_TESTS:
                    r = e11.pooled_item_contrast(folders, a, b, test, resamples=args.resamples, seed=args.seed)
                    if r is not None:
                        sink.append({"a": a, "b": b, "test": test, **r})
        for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
            row["p_holm"] = p
        means: dict[str, dict[str, float]] = defaultdict(dict)
        for f in folders:
            for cond, by_test in json.loads((f / "summary.json").read_text())["items"]["means"].items():
                for test, (value, _) in by_test.items():
                    if value is not None:
                        means[cond].setdefault(test, []).append(value)  # type: ignore[arg-type]
        group = {"track": track, "kind": kind, "model": model, "seeds": len(folders), "primary": primary, "secondary": secondary,
                 "means": {c: {t: float(np.mean(v)) for t, v in by.items()} for c, by in means.items()}}
        out["groups"].append(group)
        lines += [f"## {track} {kind} — {model} ({len(folders)} seeds)", "", "| condition | property | paraphrase | entailment |",
                  "|---|---:|---:|---:|"]
        for cond in sorted(group["means"]):
            m = group["means"][cond]
            lines.append(f"| {cond} | " + " | ".join(f"{m[t]:.4f}" if t in m else "—" for t in PRIMARY_TESTS) + " |")
        for title, rows in (("Primary (Holm)", primary), ("Secondary", secondary)):
            lines += ["", f"### {title}", "", "| a − b | test | difference [95% CI] | n |", "|---|---|---|---:|"]
            lines += [f"| {r['a']} − {r['b']} | {r['test']} | {fmt_ci(r)} | {r['n']} |" for r in rows]
        lines.append("")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return out


def run_rd4b_report(args: argparse.Namespace) -> dict[str, Any]:
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for folder in present:
        s = json.loads((folder / "summary.json").read_text())
        groups[(s["set"]["track"], s["set"]["kind"], Path(s["source"]["run"]).name.rsplit("-s", 1)[0])].append(folder)
    family = getattr(args, "family", "rd4b")
    out: dict[str, Any] = {"family": family, "groups": [], "missing": missing}
    title = {"rd4b": "# E14 RD-4b — recall of the item's relation, context-free null (pre-registration §13)",
             "rd4c": "# E14 RD-4c — routing without the item's label, Claude's frames, larger hosts (pre-registration §16)",
             "rd12c": "# E14 RD-12c — the RD-12 cycle's read frames under selective recall (pre-registration §17)"}[family]
    lines = [title, "",
             "PMI choices use the null prompt **without** the context; `raw` columns are uncalibrated choices; `(stated)`: "
             "items whose gold answer is in the `stated` reader's frame.", ""]
    if missing:
        lines += [f"Missing run folders: {', '.join(missing)}", ""]
    for (track, kind, model), folders in sorted(groups.items()):
        conds = set.intersection(*(set(json.loads((f / "summary.json").read_text())["conditions"]) for f in folders))
        pairs = {}
        if family == "rd12c":
            pairs = {"primary": PRIMARY_D, "secondary": SECONDARY_D, "stated_pairs": ()}
        if family == "rd4c":
            host_group = model.startswith("Qwen") or "-C0p" in model
            pairs = {"primary": PRIMARY_HOST if host_group else PRIMARY_C, "secondary": SECONDARY_HOST if host_group else SECONDARY_C,
                     "stated_pairs": STATED_B}
        g = {"track": track, "kind": kind, "model": model, "seeds": len(folders),
             **rd4b_group(folders, conds, resamples=args.resamples, seed=args.seed, **pairs)}
        routes: dict[str, list[float]] = defaultdict(list)
        for f in folders:
            for cond, diag in json.loads((f / "summary.json").read_text())["diagnostics"].items():
                if (diag or {}).get("route_accuracy") is not None:
                    routes[cond].append(diag["route_accuracy"])
        g["route_accuracy"] = {c: float(np.mean(v)) for c, v in routes.items()}
        out["groups"].append(g)
        cols = [c for c in ("property", "paraphrase", "entailment", "property raw", "entailment raw", "property (stated)",
                            "property raw (stated)") if any(c in m for m in g["means"].values())]
        lines += [f"## {track} {kind} — {model} ({len(folders)} seeds; {g['stated_items']} stated property items per seed)", "",
                  "| condition | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for cond in sorted(g["means"]):
            m = g["means"][cond]
            lines.append(f"| {cond} | " + " | ".join(f"{m[c]:.4f}" if c in m else "—" for c in cols) + " |")
        if family == "rd12c":                                   # property accuracy by the item's relation
            by_rel: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for f in folders:
                for line in (f / "predictions.jsonl").open():
                    row = json.loads(line)
                    if row["test"] == "property":
                        by_rel[row["relation"]][row["condition"]].append(float(row["correct"]))
            g["property_by_relation"] = {r: {c: float(np.mean(v)) for c, v in by.items()} for r, by in by_rel.items()}
            conds_r = sorted({c for by in g["property_by_relation"].values() for c in by})
            lines += ["", "Property accuracy by the item's relation:", "", "| relation | " + " | ".join(conds_r) + " |",
                      "|---|" + "---:|" * len(conds_r)]
            for r, by in sorted(g["property_by_relation"].items()):
                lines.append(f"| {r} | " + " | ".join(f"{by[c]:.3f}" if c in by else "—" for c in conds_r) + " |")
        if g["route_accuracy"]:
            lines += ["", "Router accuracy (the asked relation chosen, among items whose frame holds it): "
                      + "; ".join(f"{c} {v:.3f}" for c, v in sorted(g["route_accuracy"].items()))]
        for title, key in (("Primary (Holm)", "primary"), ("Secondary", "secondary"), ("Stated-only items", "stated_only"),
                           ("Primary contrasts on raw choices", "raw_primary")):
            lines += ["", f"### {title}", "", "| a − b | test | difference [95% CI] | n |", "|---|---|---|---:|"]
            lines += [f"| {r['a']} − {r['b']} | {r['test']} | {fmt_ci(r)} | {r['n']} |" for r in g[key]]
        lines.append("")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return out


# -- RD-3: choosing the relation, and the ReAct-style reader --------------------------------------------------------------

def trigrams(text: str) -> set[str]:
    t = " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) > 2 else set()


class TrigramIndex:
    """Fuzzy candidates for a text span: atoms whose readable text shares character trigrams (Dice coefficient)."""

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts
        self.grams = {a: trigrams(t) for a, t in texts.items() if t}
        self.inverted: dict[str, set[int]] = defaultdict(set)
        for a, g in self.grams.items():
            for x in g:
                self.inverted[x].add(a)

    def top(self, span: str, *, k: int = 5, min_score: float = 0.6) -> list[tuple[int, float]]:
        g = trigrams(span)
        if not g:
            return []
        counts: Counter = Counter()
        for x in g:
            for a in self.inverted.get(x, ()):
                counts[a] += 1
        scored = [(a, 2 * c / (len(g) + len(self.grams[a]))) for a, c in counts.items()]
        scored = [(a, s) for a, s in scored if s >= min_score]
        return sorted(scored, key=lambda x: (-x[1], x[0]))[:k]


WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'()\[\],+-]*")


def word_spans(text: str, max_words: int = 4) -> list[tuple[int, int]]:
    words = [(m.start(), m.end()) for m in WORD.finditer(text)]
    return [(words[i][0], words[j][1]) for i in range(len(words)) for j in range(i, min(len(words), i + max_words))]


class RelationReaders:
    """The new RD-3 readers on one host: `mc-rel` (the finder's partners, the relation chosen by explicit multiple choice)
    and `react` (scored-choice ReAct: find → choose the concept → choose the relation → check)."""

    def __init__(self, host: Any, ctx: e11.TrackContext, *, k: int = 5, min_score: float = 0.6, max_fuzzy: int = 8,
                 alpha: float = 0.5) -> None:
        self.host, self.ctx = host, ctx
        self.alpha = float(alpha)
        self.last_lm_choice: int | None = None
        o = ctx.ontology
        self.writer = sq.RecallWriter(o["relation_names"], o["atomic_names"], ctx.lexicon)
        texts = {a: ctx.text(a) for a in range(len(o["atomic_names"]))}
        self.index = TrigramIndex({a: t for a, t in texts.items() if t})
        self.k, self.min_score, self.max_fuzzy = k, min_score, max_fuzzy
        self.choices = 0

    def log_prior(self, atom: int, candidates: Sequence[int]) -> np.ndarray:
        """log P(relation | partner): how often the atom fills each candidate relation in the ontology (add-alpha)."""
        counts = np.array([self.ctx.typing.pools[r][atom] for r in candidates], dtype=float) + self.alpha
        return np.log(counts / counts.sum())

    def relation(self, text: str, subject: str, atom: int) -> int | None:
        """The relation for a partner: argmax of the text evidence (PMI of the relation's statement with the definition:
        log p(statement | definition) − log p(statement | null prompt)) plus log P(relation | partner). The evidence-only
        choice is kept in `last_lm_choice` (the `mc-rel-lm` ablation)."""
        candidates = self.ctx.typing.candidates(atom)
        self.last_lm_choice = None
        if not candidates:
            return None
        if len(candidates) == 1:
            self.last_lm_choice = candidates[0]
            return candidates[0]
        options = [" " + self.writer.statement(subject, r, atom) for r in candidates]
        evidence = (self.host.score([f"{text}\n\nAccording to this definition,"] * len(options), options).logprob
                    - self.host.score(["According to this definition,"] * len(options), options).logprob)
        self.choices += 1
        self.last_lm_choice = candidates[int(np.argmax(evidence))]
        return candidates[int(np.argmax(evidence + self.log_prior(atom, candidates)))]

    def _headword_spans(self, text: str, headwords: Sequence[str]) -> list[tuple[int, int]]:
        spans = []
        for h in headwords:
            if h:
                spans += [(m.start(), m.end()) for m in re.finditer(re.escape(h), text, flags=re.IGNORECASE)]
        return spans

    def mc_rel(self, text: str, subject: str, headwords: Sequence[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """(`mc-rel` frame, `mc-rel-lm` frame): the finder's partners, each relation chosen with and without the prior."""
        exclude = self._headword_spans(text, headwords)
        frame, lm = [], []
        for m in self.ctx.fillers.find(text, exclude):
            atom = m.atoms[0]
            r = self.relation(text, subject, atom)
            if r is not None:
                frame.append((r, atom))
                lm.append((self.last_lm_choice, atom))
        return list(dict.fromkeys(frame)), list(dict.fromkeys(lm))

    def react(self, text: str, subject: str, headwords: Sequence[str]) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        exclude = self._headword_spans(text, headwords)
        mentions = self.ctx.fillers.find(text, exclude)
        partners: list[int] = []
        log: list[dict[str, Any]] = []
        for m in mentions:                                         # 1. exact mentions; ambiguous surfaces are chosen
            if len(m.atoms) == 1:
                partners.append(m.atoms[0])
                continue
            options = [" " + (self.ctx.text(a) or self.ctx.ontology["atomic_names"][a]) for a in m.atoms]
            best, _ = self.host.choose(f'{text}\n\nIn this definition, "{m.surface}" refers to', options)
            self.choices += 1
            partners.append(m.atoms[best])
            log.append({"span": m.surface, "kind": "ambiguous", "chosen": int(m.atoms[best])})
        taken = exclude + [(m.start, m.end) for m in mentions]     # 2. fuzzy spans the finder missed
        fuzzy = []
        for start, end in word_spans(text):
            if any(start < e and s < end for s, e in taken):
                continue
            cands = self.index.top(text[start:end], k=self.k, min_score=self.min_score)
            if cands:
                fuzzy.append((cands[0][1], start, end, cands))
        chosen_spans: list[tuple[int, int]] = []
        for score, start, end, cands in sorted(fuzzy, key=lambda x: -x[0]):
            if len(chosen_spans) >= self.max_fuzzy or any(start < e and s < end for s, e in chosen_spans):
                continue
            chosen_spans.append((start, end))
            options = [" " + self.index.texts[a] for a, _ in cands] + [" none of these"]
            best, _ = self.host.choose(f'{text}\n\nIn this definition, "{text[start:end]}" refers to', options)
            self.choices += 1
            pick = None if best == len(cands) else int(cands[best][0])
            log.append({"span": text[start:end], "kind": "fuzzy", "candidates": [int(a) for a, _ in cands], "chosen": pick})
            if pick is not None:
                partners.append(pick)
        frame: list[tuple[int, int]] = []                          # 3. relation; 4. check (typing, functional relations)
        seen_functional: set[int] = set()
        for atom in dict.fromkeys(partners):
            r = self.relation(text, subject, atom)
            if r is None or r not in self.ctx.typing.candidates(atom):
                continue
            if r in self.ctx.typing.functional:
                if r in seen_functional:
                    continue
                seen_functional.add(r)
            frame.append((r, atom))
        return list(dict.fromkeys(frame)), {"steps": log}


def frame_scores(pred: Sequence[tuple[int, int]] | None, gold: Sequence[tuple[int, int]], stated: Sequence[tuple[int, int]],
                 ambiguous_atoms: set[int]) -> dict[str, Any]:
    P, G, S = set(pred or []), set(gold), set(stated)
    precision = len(P & G) / len(P) if P else None
    stated_recall = len(P & S) / len(S) if S else None
    f1 = (2 * precision * stated_recall / (precision + stated_recall) if precision and stated_recall else 0.0) \
        if (P and S) else (0.0 if S else None)
    amb = [(r, a) for r, a in S if a in ambiguous_atoms]
    amb_hits = [(r, a) in P for r, a in amb if any(x == a for _, x in P)]
    return {"edges": len(P), "precision": precision, "recall": len(P & G) / len(G) if G else None, "stated_recall": stated_recall,
            "stated_f1": f1, "ambiguous": len(amb), "ambiguous_labelled": len(amb_hits), "ambiguous_correct": sum(amb_hits)}


def summarize_readers(per: dict[str, dict[str, dict[str, Any]]], *, resamples: int, seed: int) -> dict[str, Any]:
    from .e5_zeroshot import paired_difference
    out: dict[str, Any] = {"readers": {}, "contrasts": []}
    for reader, rows in per.items():
        vals = [r for r in rows.values()]
        mean = lambda k: float(np.mean([r[k] for r in vals if r[k] is not None])) if any(r[k] is not None for r in vals) else None  # noqa: E731
        amb_l = sum(r["ambiguous_labelled"] for r in vals)
        out["readers"][reader] = {"terms": len(vals), "precision": mean("precision"), "recall": mean("recall"),
                                  "stated_recall": mean("stated_recall"), "stated_f1": mean("stated_f1"), "edges": mean("edges"),
                                  "ambiguous_accuracy": sum(r["ambiguous_correct"] for r in vals) / amb_l if amb_l else None,
                                  "ambiguous_labelled": amb_l}
    for a, b in (("react", "linker"), ("react", "typeprior"), ("mc-rel", "linker"), ("mc-rel", "typeprior"), ("react", "mc-rel"),
                 ("mc-rel", "mc-rel-lm"), ("mc-rel-lm", "linker"),
                 ("react", "host"), ("teacher", "react")):
        if a not in per or b not in per:
            continue
        ids = sorted(c for c in set(per[a]) & set(per[b]) if per[a][c]["stated_f1"] is not None and per[b][c]["stated_f1"] is not None)
        if ids:
            out["contrasts"].append({"a": a, "b": b, "metric": "stated_f1", **paired_difference(
                np.array([per[a][c]["stated_f1"] for c in ids]), np.array([per[b][c]["stated_f1"] for c in ids]), resamples=resamples, seed=seed)})
    primary = [c for c in out["contrasts"] if c["a"] == "react" and c["b"] in {"linker", "typeprior"}]
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    return out


def run_relations(args: argparse.Namespace) -> dict[str, Any]:
    from ..logic.hosts import HostLM
    read_set = e11.load_read_set(args.items).limit(args.limit)
    track = read_set.manifest["track"]
    style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    ctx = e11.track_context(track)
    config = {"experiment": "e14-rd3-relations", "items": str(args.items), "style": style, "host": args.host, "frames": str(args.frames),
              "teacher_frames": str(args.teacher_frames) if args.teacher_frames else None, "limit": args.limit, "smoke": bool(args.smoke)}
    output = Path(args.output)
    if args.overwrite:
        clear_output(output)
        (output / "frames.jsonl").unlink(missing_ok=True)
    git_at_start = start_output(output, config)
    old = load_reader_frames(args.frames, style, ("oracle", "stated", "linker", "typeprior", "host"), ctx)
    if args.teacher_frames:
        old["teacher"] = {r["concept"]: ctx.ids(r["frame"]) for r in e11._read_jsonl(args.teacher_frames) if r.get("style") == style}
    host = HostLM(args.host, device=args.device, batch_size=args.batch_size)
    readers = RelationReaders(host, ctx, k=args.k, min_score=args.min_score, max_fuzzy=args.max_fuzzy)
    concepts = {c["concept"]: c for c in read_set.concepts}
    ambiguous = {a for a in range(len(ctx.ontology["atomic_names"])) if len(ctx.typing.candidates(a)) >= 2}
    per: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    frames_out = []
    t0 = time.monotonic()
    for d in read_set.definitions:
        if d["style"] != style or d["concept"] not in concepts:
            continue
        c = concepts[d["concept"]]
        cid, text = c["concept"], d["text"]
        subject = c.get("item_surface") or c["surface"]
        heads = [c["surface"], subject, d.get("headword") or ""]
        gold = old["oracle"].get(cid) or ctx.ids(c["frame"]) or []
        stated = old["stated"].get(cid) or []
        mc, mc_lm = readers.mc_rel(text, subject, heads)
        new = {"mc-rel": mc, "mc-rel-lm": mc_lm}
        new["react"], details = readers.react(text, subject, heads)
        for name, frame in {**{k: v.get(cid) for k, v in old.items() if k not in {"oracle", "stated"}}, **new}.items():
            per[name][cid] = frame_scores(frame, gold, stated, ambiguous)
        frames_out += [{"concept": cid, "style": style, "reader": name, "frame": ctx.names(frame)} for name, frame in new.items()]
        frames_out.append({"concept": cid, "style": style, "reader": "react-log", "details": details})
    document = {"source": host.describe(), "set": {"track": track, "kind": read_set.kind}, "style": style, "smoke": bool(args.smoke),
                "choices": readers.choices, "seconds": round(time.monotonic() - t0, 1),
                **summarize_readers(per, resamples=args.resamples, seed=args.seed)}
    e11._write_jsonl(output / "frames.jsonl", frames_out)
    write_json(output / "summary.json", document)
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [
        f"# E14 RD-3 — choosing the relation on stated edges ({track} {read_set.kind}, style `{style}`, host `{host.name}`)", "",
        "| reader | terms | edges | precision | recall (all gold) | stated recall | stated F1 | ambiguous accuracy (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    for name, r in document["readers"].items():
        lines.append(f"| {name} | {r['terms']} | {f(r['edges'])} | {f(r['precision'])} | {f(r['recall'])} | {f(r['stated_recall'])} | "
                     f"{f(r['stated_f1'])} | {f(r['ambiguous_accuracy'])} ({r['ambiguous_labelled']}) |")
    lines += ["", "| a − b | stated F1 difference [95% CI] | n |", "|---|---|---:|"]
    lines += [f"| {c['a']} − {c['b']} | {fmt_ci(c)} | {c['n']} |" for c in document["contrasts"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    finish_output(output, config, git_at_start=git_at_start, device=host.device, source=host.describe())
    return document


def run_relations_report(args: argparse.Namespace) -> dict[str, Any]:
    """RD-3's primary family per track: `react − linker` and `react − typeprior` (stated F1) over the hosts, Holm."""
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tables = []
    for folder in present:
        s = json.loads((folder / "summary.json").read_text())
        track = f"{s['set']['track']} {s['set']['kind']}"
        for c in s["contrasts"]:
            if c["a"] == "react" and c["b"] in {"linker", "typeprior"}:
                by_track[track].append({**c, "host": s["source"]["host"]})
        tables.append((track, s["source"]["host"], s["readers"]))
    for rows in by_track.values():
        for row, p in zip(rows, holm_adjust([r["p_value"] for r in rows])):
            row["p_holm_family"] = p
    lines = ["# E14 RD-3 — pooled report (Holm per track over hosts × {react − linker, react − typeprior})", ""]
    if missing:
        lines += [f"Missing: {', '.join(missing)}", ""]
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    for track, host, readers in sorted(tables):
        lines += [f"## {track} — {host}", "", "| reader | stated F1 | precision | stated recall | ambiguous accuracy |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {n} | {f(r['stated_f1'])} | {f(r['precision'])} | {f(r['stated_recall'])} | {f(r['ambiguous_accuracy'])} |"
                  for n, r in readers.items()]
        lines.append("")
    lines += ["## Primary", "", "| track | host | a − b | stated F1 difference [95% CI] | p (Holm, family) | n |", "|---|---|---|---|---:|---:|"]
    for track, rows in sorted(by_track.items()):
        lines += [f"| {track} | {r['host']} | {r['a']} − {r['b']} | {fmt_ci(r)} | {r['p_holm_family']:.4f} | {r['n']} |" for r in rows]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    doc = {"primary": by_track, "missing": missing}
    write_json(output / "report.json", doc)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("recall", help="RD-4 / RD-7: implicit against explicit recall of read frames")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--frames", type=Path, default=None, help="an E11 frames.jsonl (default: the run's own E11 output)")
    r.add_argument("--style", default=None)
    r.add_argument("--conditions", default=None, help="comma-separated (default: every condition the run supports)")
    r.add_argument("--noise", default=None, help="RD-7 noise levels (default 0.1,0.3,0.5)")
    r.add_argument("--no-vsa", action="store_true", help="skip the explicit-vsa conditions")
    r.add_argument("--alias-table", type=Path, default=None)
    r.add_argument("--checkpoint", default="final.pt")
    r.add_argument("--device", default=None)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-context-tokens", type=int, default=1024, help="cap on the prompt prefix (whole lines kept)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--resamples", type=int, default=4000)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--null", choices=("context", "plain"), default="context",
                   help="PMI null prompt: after the same context (tier P) or without it (RD-4b)")
    r.add_argument("--family", choices=("tier-p", "rd4b", "rd4c", "rd12c"), default="tier-p", help="default condition list")
    r.add_argument("--extra-readers", default=None, help="more reader names to load from --frames (rd12c: the RD-12 arms)")
    r.add_argument("--teacher-frames", type=Path, default=None,
                   help="RD-4c: Claude's frames (RD-3's teacher-frames.jsonl), read as the reader `teacher`")
    rr = sub.add_parser("recall-report", help="pool RD-4 run folders over seeds")
    rr.add_argument("--runs", nargs="+", required=True)
    rr.add_argument("--output", type=Path, required=True)
    rr.add_argument("--family", choices=("tier-p", "rd4b", "rd4c", "rd12c"), default="tier-p", help="which pre-registered contrasts")
    rr.add_argument("--resamples", type=int, default=4000)
    rr.add_argument("--seed", type=int, default=0)
    rl = sub.add_parser("relations", help="RD-3: relation choice on stated edges (mc-rel, react, and E11's readers)")
    rl.add_argument("--items", type=Path, required=True)
    rl.add_argument("--frames", type=Path, required=True, help="an E11 frames.jsonl with the stated / linker / typeprior / host frames")
    rl.add_argument("--teacher-frames", type=Path, default=None)
    rl.add_argument("--host", required=True, help="H-360, H-q17, a Hugging Face name or a local path")
    rl.add_argument("--output", type=Path, required=True)
    rl.add_argument("--style", default=None)
    rl.add_argument("--device", default=None)
    rl.add_argument("--batch-size", type=int, default=16)
    rl.add_argument("--k", type=int, default=5)
    rl.add_argument("--min-score", type=float, default=0.6)
    rl.add_argument("--max-fuzzy", type=int, default=8)
    rl.add_argument("--limit", type=int, default=None)
    rl.add_argument("--resamples", type=int, default=4000)
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument("--smoke", action="store_true")
    rl.add_argument("--overwrite", action="store_true")
    rr2 = sub.add_parser("relations-report", help="pool RD-3 run folders (Holm per track)")
    rr2.add_argument("--runs", nargs="+", required=True)
    rr2.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "relations":
        run_relations(args)
        return
    if args.command == "relations-report":
        run_relations_report(args)
        return
    if args.command == "recall":
        run_recall(args)
    elif args.command == "recall-report":
        run_recall_report(args)


if __name__ == "__main__":
    main()
