"""E11-M — many terms, each read once (author request 2026-10-07; preregistration §14).

A model learns N new terms (N ∈ {25, 50, 100, 200, 400}) from one reading of each definition, then answers questions
about them and reads text that uses several of them together. The routes:

- **frames** (`frame:<reader>`): every term's frame is written into the ontology at once (readers oracle, linker,
  linker-joint, typeprior; `none` = no frame). A use costs no context token. Composed rows are independent, so a term's
  row does not change with N; `interference` checks it.
- **in context** (`context:B<budget>`): the N definitions, in a seeded order, are put in the prompt up to a budget of B
  tokens (whole definitions; the rest is cut). Prompts are read after this prefix through a key/value cache.
- **retrieval** (`rag:k<k>`): BM25 over the N definitions with the prompt (or the passage / window) as the query; the
  top-k definitions go in the prompt.
- **gradient** (`gradient`): the host is updated on the N definitions one after another (compute-matched steps each,
  the dev learning rate), then tested; checkpoints after 25, 50, … terms on the same sequence; weights restored.

Tests: the E9 dimension-3 items of the first N terms (property, entailment, paraphrase, statement); the loss after each
new-term mention in passages that use several of them (T5: generated from the frames, 4 terms per passage; T4: natural
evaluation windows with ≥ 2 distinct read terms); interference (frames: rows vs N; gradient: accuracy of the first terms
as more are learned); context tokens and prefill FLOPs per use.

    python -m vsa_embed.experiments.e11_many build --read-set DIR --output DIR [--sizes 25 50 100 200 400] [--style prose]
    python -m vsa_embed.experiments.e11_many evaluate --run RUN --items DIR --output OUT [--routes frames,context,rag,gradient]
        [--readers oracle,linker,linker-joint,typeprior,none] [--budgets 2048 7168] [--rag-k 1 3] [--gradient-lr-from JSON]
    python -m vsa_embed.experiments.e11_many report --runs OUT ... --output DIR
    python -m vsa_embed.experiments.e11_many plan [--seeds 1 2 3]          (prints queue commands; nothing is submitted)
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .. import read_to_learn as rtl
from ..statistics import paired_ratio_bootstrap
from . import e5_zeroshot as zs
from . import e11_read_to_learn as e11
from .e5_common import E5Run, clear_output, finish_output, fmt, json_ready, open_run, start_output, write_json

SCHEMA = "e11-many/1"
SIZES = (25, 50, 100, 200, 400)
BUDGETS = (2048, 7168)                 # 2,048: a small model's working budget; 7,168: the 8,192 window minus 1,024 for the task
RAG_K = (1, 3)
ROUTES = ("frames", "context", "rag", "gradient")
READERS = ("oracle", "linker", "linker-joint", "typeprior", "none")
HEADER = "Glossary of new terms:\n"
PRIMARY = {"size": 200, "budget": 2048, "test": "property", "a": "frame:linker", "b": "context:B2048"}
ITEMS = e11.ITEMS
WORD = re.compile(r"\w+")


# -- the item set ------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ManySet:
    path: Path
    manifest: dict[str, Any]
    order: list[str]
    passages: list[dict[str, Any]]
    read_set: e11.ReadSet                      # the base read set restricted to `order`, in that order

    @property
    def sizes(self) -> list[int]:
        return list(self.manifest["sizes"])

    @property
    def style(self) -> str:
        return self.manifest["style"]

    @property
    def kind(self) -> str:
        return self.read_set.kind

    def first(self, n: int) -> list[str]:
        return self.order[:n]


def restrict(read_set: e11.ReadSet, order: Sequence[str], style: str) -> e11.ReadSet:
    keep = set(order)
    by_id = {c["concept"]: c for c in read_set.concepts}
    return dataclasses.replace(read_set, concepts=[by_id[c] for c in order],
                               definitions=[d for d in read_set.definitions if d["concept"] in keep and d["style"] == style],
                               items=[i for i in read_set.items if i["concept"] in keep])


def load_many(path: Path) -> ManySet:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not an {SCHEMA} item directory")
    order = json.loads((path / "order.json").read_text())
    passages = [json.loads(line) for line in (path / "passages.jsonl").read_text().splitlines() if line.strip()] \
        if (path / "passages.jsonl").exists() else []
    base = e11.load_read_set(Path(manifest["base"]))
    return ManySet(path, manifest, order, passages, restrict(base, order, manifest["style"]))


def t5_passage(terms: Sequence[dict[str, Any]], facts: dict[str, dict[str, list[str]]], rng: random.Random, *,
               facts_per_term: int = 2) -> tuple[str, list[list[Any]]]:
    """An internal-document passage using several new T5 terms: per term `facts_per_term` facts in the training fact
    templates (`benchmarks.glossary.FACT_TEMPLATES`), boilerplate between the term blocks; returns (text, mention spans
    [concept, start, end])."""
    from ..benchmarks.glossary import BOILERPLATE, FACT_TEMPLATES, PLURALS
    header = rng.choice(["Weekly sync notes.", "Status update from the platform review.", "Handover notes for the new quarter.",
                         "Notes from the planning meeting."])
    blocks = []
    for term in terms:
        surface, own = term["surface"], facts[term["concept"]]
        kind = (own.get("is_a") or ["term"])[0]
        pairs = [(r, o) for r, objs in own.items() for o in objs]
        sentences = []
        for relation, obj in rng.sample(pairs, min(facts_per_term, len(pairs))):
            template = rng.choice(FACT_TEMPLATES.get(relation, ["{s} " + relation.replace("_", " ") + " {o}."]))
            text = template.format(s=surface, o=obj, O=obj[:1].upper() + obj[1:], t=kind, a_o=rtl._a(obj),
                                   os=PLURALS.get(obj, obj + "s"))
            sentences.append(text[:1].upper() + text[1:])
        blocks.append(" ".join(sentences))
    parts = [header]
    for i, block in enumerate(blocks):
        parts.append(block)
        if i < len(blocks) - 1:
            parts.append(rng.choice(BOILERPLATE))
    text = " ".join(parts)
    mentions = []
    for term in terms:
        for m in re.finditer(rf"(?<![\w]){re.escape(term['surface'])}(?![\w])", text, re.I):
            mentions.append([term["concept"], m.start(), m.end()])
    return text, sorted(mentions, key=lambda x: x[1])


def build_many_set(base_dir: Path, out_dir: Path, *, sizes: Sequence[int] = SIZES, style: str | None = None, seed: int = 0,
                   passages_per_size: int = 50, terms_per_passage: int = 4, ctx: e11.TrackContext | None = None) -> dict[str, Any]:
    """A nested episode design over a read set: one seeded order of its terms (the first N terms are the episode of size
    N), and for `new` T5 sets `passages_per_size` passages per size, each using `terms_per_passage` of the first N terms."""
    base = e11.load_read_set(base_dir)
    style = style or e11.PRIMARY_STYLE.get(base.manifest["track"], base.styles[0])
    with_definition = {d["concept"] for d in base.definitions if d["style"] == style}
    pool = [c["concept"] for c in base.concepts if c["concept"] in with_definition]
    if base.kind == "heldout" and any("eval_occurrences" in c for c in base.concepts):
        # natural sets: only terms that occur in the evaluation text
        pool = [c["concept"] for c in base.concepts if c["concept"] in with_definition and c.get("eval_occurrences", 0) > 0]
    rng = random.Random(seed)
    rng.shuffle(pool)
    sizes = sorted({int(s) for s in sizes if int(s) <= len(pool)} | ({len(pool)} if max(sizes) > len(pool) else set()))
    order = pool[:max(sizes)]
    passages: list[dict[str, Any]] = []
    if base.kind == "new" and base.manifest["track"] == "t5":
        ctx = ctx or e11.track_context("t5")
        concepts = {c["concept"]: c for c in base.concepts}
        facts = {c: ctx.facts(ctx.ids(concepts[c]["frame"])) for c in order}
        for n in sizes:
            for p in range(passages_per_size):
                prng = random.Random(f"{seed}|passage|{n}|{p}")
                terms = [concepts[c] for c in prng.sample(order[:n], min(terms_per_passage, n))]
                text, mentions = t5_passage(terms, facts, prng)
                passages.append({"id": f"p{n}-{p:03d}", "size": n, "terms": [t["concept"] for t in terms], "text": text,
                                 "mentions": mentions})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "order.json").write_text(json.dumps(order, indent=0) + "\n")
    if passages:
        (out_dir / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    manifest = {"schema": SCHEMA, "base": str(base_dir), "kind": base.kind, "track": base.manifest["track"], "style": style,
                "sizes": sizes, "seed": seed, "pool": len(pool), "passages": len(passages), "passages_per_size": passages_per_size,
                "terms_per_passage": terms_per_passage, "created": time.strftime("%Y-%m-%d"),
                "design": "nested: the episode of size N is the first N terms of one seeded order; passages of size N use only "
                          "those terms; natural sets (heldout) use evaluation windows with >= 2 distinct read terms instead"}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


# -- contexts and retrieval --------------------------------------------------------------------------------------------

def definitions_context(definitions: Sequence[tuple[str, str]], tokenizer: Any, budget: int, *, seed: int = 0
                        ) -> dict[str, Any]:
    """The definitions (concept, text) in a seeded order under `HEADER`, whole definitions only, up to `budget` tokens."""
    order = list(definitions)
    random.Random(f"context|{seed}|{len(order)}").shuffle(order)
    used = len(tokenizer(HEADER, add_special_tokens=False)["input_ids"])
    lines, present, total = [], [], used
    lengths = [len(ids) for ids in tokenizer([text + "\n" for _, text in order], add_special_tokens=False)["input_ids"]] if order else []
    for (concept, text), n in zip(order, lengths):
        total += n
        if used + n <= budget:
            lines.append(text); present.append(concept); used += n
    return {"text": HEADER + "".join(line + "\n" for line in lines), "present": present, "tokens": used, "all_tokens": total}


class BM25:
    """Okapi BM25 over short documents (word tokens, lower case)."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(WORD.findall(d.lower())) for d in documents]
        self.lengths = np.asarray([sum(d.values()) for d in self.docs], dtype=float)
        self.average = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        words = WORD.findall(query.lower())
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.average, 1e-9))
            out[i] = sum(self.idf.get(w, 0.0) * d[w] * (self.k1 + 1) / (d[w] + norm) for w in words if w in d)
        return out

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        return [int(i) for i in np.argsort(-s, kind="stable")[:k]]


def retrieved_context(retriever: BM25, concepts: Sequence[str], texts: Sequence[str], query: str, k: int) -> tuple[str, list[str]]:
    hits = retriever.top(query, k)
    return HEADER + "".join(texts[i] + "\n" for i in hits), [concepts[i] for i in hits]


# -- passage and window losses ------------------------------------------------------------------------------------------

@torch.no_grad()
def passage_losses(run: E5Run, adapter: Any, passages: Sequence[dict[str, Any]], *, prefix: e11.PrefixCache | None = None,
                   batch: int = 8, window: int = e11.AFTER_WINDOW) -> list[list[tuple[float, int]]]:
    """Per passage, per mention: (summed loss, targets) over the `window` targets after the mention (the trainer's
    after-span window), with the passage read alone or after a cached `prefix`."""
    from ..integrations.transformers import chunked_causal_lm_loss
    tokenizer, model, device = run.tokenizer, run.model, run.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    out: list[list[tuple[float, int]]] = []
    for start in range(0, len(passages), batch):
        chunk = passages[start:start + batch]
        texts = [p["text"] for p in chunk]
        encoded = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False, padding=True, return_tensors="pt")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        offsets = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in zip(encoded["offset_mapping"], mask)]
        spans = adapter.spans_fn(texts, offsets) if getattr(adapter, "spans_fn", None) is not None and model.channel is not None else None
        if prefix is not None:
            hidden = prefix.hidden(ids, mask, spans)
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model.hidden_states(ids.to(device), mask.to(device),
                                             None if spans is None else {k: v.to(device) for k, v in spans.items()})
        head = model.model.get_output_embeddings()
        labels = ids.masked_fill(mask == 0, -100).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = chunked_causal_lm_loss(hidden, head.weight, labels, output_bias=getattr(head, "bias", None),
                                            reduction="none").float().cpu().numpy()
        for row, passage in enumerate(chunk):
            n = len(offsets[row])
            per = []
            for _, s, e in passage["mentions"]:
                last = max((t for t in range(n) if offsets[row][t][0] < e), default=None)
                if last is None:
                    per.append((0.0, 0)); continue
                lo, hi = last, min(n - 1, last + window)
                per.append((float(losses[row, lo:hi].sum()), max(0, hi - lo)))
            out.append(per)
    return out


@torch.no_grad()
def window_losses(run: E5Run, corpus_path: Path, wset: e11.WindowSet, *, prefix_of: Callable[[int], e11.PrefixCache | None] | None = None,
                  shared: e11.PrefixCache | None = None, select: Callable[[int, int], bool] | None = None, batch: int = 4
                  ) -> dict[str, np.ndarray]:
    """`e11.window_term_losses` with a context prefix: `shared` (one cache for every window) or `prefix_of(window)` (one
    cache per window, e.g. retrieved definitions; windows then run one at a time)."""
    from ..data.corpus import TokenCorpus, collate_windows
    from ..integrations.transformers import chunked_causal_lm_loss
    corpus = TokenCorpus.open(Path(corpus_path))
    min_subtokens = int(run.config["data"]["min_subtokens"])
    n_terms = len(wset.terms)
    sums = {k: np.zeros(n_terms) for k in ("other", "own")}
    counts = {k: np.zeros(n_terms) for k in ("other", "own")}
    order = [w for w in range(len(wset.starts)) if select is None or any(select(w, t) for _, _, t, _ in wset.spans[w])]
    step = 1 if prefix_of is not None else batch
    head = run.model.model.get_output_embeddings()
    device = run.device
    for b0 in range(0, len(order), step):
        chunk = order[b0:b0 + step]
        ids, spans = collate_windows([corpus.window(wset.starts[w], wset.length, min_subtokens=min_subtokens) for w in chunk])
        mask = torch.ones_like(ids)
        cache = prefix_of(chunk[0]) if prefix_of is not None else shared
        if cache is not None:
            hidden = cache.hidden(ids, mask, spans if run.model.channel else None)
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = run.model.hidden_states(ids.to(device), mask.to(device),
                                                 {k: v.to(device) for k, v in spans.items()} if run.model.channel else None)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = chunked_causal_lm_loss(hidden, head.weight, ids.to(device), output_bias=getattr(head, "bias", None),
                                            reduction="none").float().cpu().numpy()
        for row, w in enumerate(chunk):
            for end, _, t, own in wset.spans[w]:
                if select is not None and not select(w, t):
                    continue
                lo, hi = end, min(wset.length - 1, end + e11.AFTER_WINDOW)
                if lo < hi:
                    key = "own" if own else "other"
                    sums[key][t] += float(losses[row, lo:hi].sum()); counts[key][t] += hi - lo
    return {"sum_other": sums["other"], "count_other": counts["other"], "sum_own": sums["own"], "count_own": counts["own"]}


# -- evaluation ----------------------------------------------------------------------------------------------------------

def frames_of(results: Sequence[rtl.ReadResult], style: str) -> dict[str, dict[str, rtl.Frame | None]]:
    out: dict[str, dict[str, rtl.Frame | None]] = defaultdict(dict)
    for r in results:
        if r.style == style:
            out[r.reader][r.concept] = r.frame
    return out


def _values(rows_by_concept: dict[str, list[dict[str, Any]]], keep: set[str]) -> dict[str, dict[str, float]]:
    return e11.test_values({c: rows for c, rows in rows_by_concept.items() if c in keep}, "new")


def evaluate_many(run: E5Run, many: ManySet, *, routes: Sequence[str], readers: Sequence[str], budgets: Sequence[int],
                  rag_k: Sequence[int], gradient_lr: float | None, gradient_optimizer: str = "adam", gradient_backup: str = "gpu",
                  sizes: Sequence[int] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Every route on the episode sizes; returns item rows per (condition, size), passage / window losses, interference
    and cost records."""
    started = time.monotonic()
    ctx = e11.run_context(run)
    sizes = [n for n in (sizes or many.sizes) if n <= len(many.order)]
    compose = run.channel is not None and run.channel.mode == "compose"
    read_set = many.read_set
    style = many.style
    definitions = {d["concept"]: d["text"] for d in read_set.definitions}
    scorer = e11.ItemScorer(run, read_set, ctx) if read_set.items else None
    params = sum(p.numel() for p in run.model.parameters())
    items: dict[str, dict[str, list[dict[str, Any]]]] = {}           # condition → concept → rows (N-independent conditions)
    items_n: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    passages: dict[tuple[str, int], list[list[tuple[float, int]]]] = {}
    windows: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    costs: dict[str, dict[str, float]] = {}
    timings: dict[str, float] = {}
    none_frames = {c: None for c in many.order}
    passages_of = {n: [p for p in many.passages if p["size"] == n] for n in sizes}
    wsets: dict[int, e11.WindowSet] = {}
    entry_of = e11.link_entry_of(read_set, run)
    corpus_path = Path(run.config["data"]["eval"])
    if many.kind == "heldout":
        headwords = {entry_of[c["concept"]]: c["surface"] for c in read_set.concepts}
        full = e11.build_window_set(run, [entry_of[c] for c in many.order], headwords, corpus_path=corpus_path)
        for n in sizes:                     # windows with >= 2 distinct read terms among the first n
            keep_terms = set(range(n))
            selected = [w for w in range(len(full.starts)) if len({t for _, _, t, _ in full.spans[w] if t in keep_terms}) >= 2]
            wsets[n] = dataclasses.replace(full, starts=[full.starts[w] for w in selected],
                                           spans=[[s for s in full.spans[w] if s[2] in keep_terms] for w in selected])

    def record_text_routes(condition: str, frames: dict[str, rtl.Frame | None]) -> None:
        if many.passages and scorer is not None:
            with scorer.condition(frames):
                for n in sizes:
                    passages[(condition, n)] = passage_losses(run, scorer.adapter, passages_of[n])
        if many.kind == "heldout":
            for n in sizes:
                swapped = {entry_of[c]: (frames.get(c) if i < n else None) for i, c in enumerate(many.order)}
                with e11.swapped_frames(run.channel, swapped) if compose else contextlib.nullcontext():
                    windows[(condition, n)] = window_losses(run, corpus_path, wsets[n])

    # 1. frames (and no frame), written for every term at once
    t0 = time.monotonic()
    if "frames" in routes:
        wanted = [r for r in readers if r != "none"] if compose else []
        scorer_def = e11.DefinitionScorer(run, read_set) if compose and {"linker", "linker-joint"} & set(wanted) else None
        results, mentions = e11.run_readers(run, read_set, ctx, wanted, [style], scorer=scorer_def, log=log)
        frames = frames_of(results, style)
        frame_rows = e11.frame_rows(results, read_set, ctx, mentions)
        costs["write"] = {r: float(np.mean([x["cost"].get("forward_tokens", 0.0) for x in frame_rows if x["reader"] == r] or [0.0]))
                          for r in frames}
        for reader in [*frames, "none"]:
            log(f"  frames: {reader}")
            condition = "none" if reader == "none" else f"frame:{reader}"
            chosen = none_frames if reader == "none" else frames[reader]
            if scorer is not None:
                items[condition] = scorer.score(chosen)
            record_text_routes(condition, chosen)
        if compose and "oracle" in frames and scorer is not None:
            # interference: the first size's terms scored with only them written vs with every term written
            small = dataclasses.replace(read_set, concepts=read_set.concepts[:sizes[0]],
                                        definitions=[d for d in read_set.definitions if d["concept"] in set(many.order[:sizes[0]])],
                                        items=[i for i in read_set.items if i["concept"] in set(many.order[:sizes[0]])])
            alone = e11.ItemScorer(run, small, ctx).score({c: frames["oracle"].get(c) for c in many.order[:sizes[0]]})
            together = items["frame:oracle"]
            diffs = [abs(a["margin"] - b["margin"]) for c in alone for a, b in zip(alone[c], together[c]) if "margin" in a and "margin" in b]
            items_n[("interference:alone", sizes[0])] = alone
            costs["interference"] = {"items": len(diffs), "max_abs_margin_change": float(max(diffs)) if diffs else None}
        summary_frames = e11.summarize_frames(frame_rows) if frame_rows else {}
    else:
        summary_frames = {}
    timings["frames"] = time.monotonic() - t0

    # 2. in context: the first N definitions under a token budget
    t0 = time.monotonic()
    presence: dict[str, Any] = {}
    if "context" in routes:
        for budget in budgets:
            for n in sizes:
                block = definitions_context([(c, definitions[c]) for c in many.order[:n]], run.tokenizer, budget)
                condition = f"context:B{budget}"
                presence[f"{condition}|N{n}"] = {"present": len(block["present"]), "of": n, "tokens": block["tokens"],
                                                 "all_tokens": block["all_tokens"]}
                log(f"  context B={budget} N={n}: {len(block['present'])}/{n} definitions, {block['tokens']} tokens")
                if scorer is not None:
                    items_n[(condition, n)] = scorer.score(none_frames, concepts=many.order[:n], prefix=block["text"],
                                                           tag=f"{condition}|N{n}")
                    for c in many.order[:n]:
                        for row in items_n[(condition, n)][c]:
                            row["definition_in_context"] = c in set(block["present"])
                if many.passages and scorer is not None:
                    with scorer.condition(none_frames):
                        cache = e11.PrefixCache(run, scorer.adapter, block["text"])
                        passages[(condition, n)] = passage_losses(run, scorer.adapter, passages_of[n], prefix=cache)
                if many.kind == "heldout":
                    with e11.swapped_frames(run.channel, {entry_of[c]: None for c in many.order}) if compose else contextlib.nullcontext():
                        cache = e11.PrefixCache(run, run.adapter, block["text"])
                        windows[(condition, n)] = window_losses(run, corpus_path, wsets[n], shared=cache)
                costs[f"{condition}|N{n}"] = {"context_tokens_per_use": float(block["tokens"]),
                                              "prefill_flops_per_use": 2.0 * params * block["tokens"]}
    timings["context"] = time.monotonic() - t0

    # 3. retrieval: BM25 top-k definitions per prompt / passage / window
    t0 = time.monotonic()
    recall: dict[str, float] = {}
    if "rag" in routes:
        for k in rag_k:
            condition = f"rag:k{k}"
            for n in sizes:
                pool = many.order[:n]
                texts = [definitions[c] for c in pool]
                retriever = BM25(texts)
                tokens_used, hits = [], []
                if scorer is not None:
                    rows: dict[str, list[dict[str, Any]]] = {}
                    surface = {c["concept"]: c.get("item_surface") or c["surface"] for c in read_set.concepts}
                    for c in pool:
                        first = next((i for i in read_set.items if i["concept"] == c), None)
                        query = first["templates"][0].format(x=surface[c]) if first else surface[c]
                        context, got = retrieved_context(retriever, pool, texts, query, k)
                        hits.append(c in got)
                        tokens_used.append(len(run.tokenizer(context, add_special_tokens=False)["input_ids"]))
                        rows.update(scorer.score(none_frames, concepts=[c], prefix=context, tag=f"{condition}|{','.join(got)}"))
                    items_n[(condition, n)] = rows
                if many.passages and scorer is not None:
                    results_p = []
                    with scorer.condition(none_frames):
                        for p in passages_of[n]:
                            context, got = retrieved_context(retriever, pool, texts, p["text"], k * len(p["terms"]))
                            results_p += passage_losses(run, scorer.adapter, [p], prefix=e11.PrefixCache(run, scorer.adapter, context))
                    passages[(condition, n)] = results_p
                if many.kind == "heldout":
                    from ..data.corpus import TokenCorpus
                    corpus = TokenCorpus.open(corpus_path)

                    def prefix_of(w: int, retriever=retriever, pool=pool, texts=texts, n=n) -> e11.PrefixCache:
                        query = run.tokenizer.decode(np.asarray(corpus.tokens[wsets[n].starts[w]:wsets[n].starts[w] + wsets[n].length]).tolist())
                        context, _ = retrieved_context(retriever, pool, texts, query, k)
                        return e11.PrefixCache(run, run.adapter, context)
                    with e11.swapped_frames(run.channel, {entry_of[c]: None for c in many.order}) if compose else contextlib.nullcontext():
                        windows[(condition, n)] = window_losses(run, corpus_path, wsets[n], prefix_of=prefix_of)
                recall[f"{condition}|N{n}"] = float(np.mean(hits)) if hits else None
                costs[f"{condition}|N{n}"] = {"context_tokens_per_use": float(np.mean(tokens_used)) if tokens_used else None,
                                              "prefill_flops_per_use": 2.0 * params * float(np.mean(tokens_used)) if tokens_used else None}
    timings["rag"] = time.monotonic() - t0

    # 4. gradient: one pass over the definitions in order, checkpoints at every size
    t0 = time.monotonic()
    forgetting: dict[str, Any] = {}
    if "gradient" in routes:
        if gradient_lr is None:
            raise ValueError("the gradient route needs --gradient-lr or --gradient-lr-from")
        tasks = {t.concept: t for t in e11.read_tasks(read_set, ctx, [style])}
        steps_total = tokens_total = 0
        with e11.host_snapshot(run.model, backup=gradient_backup):
            done = 0
            for n in sizes:
                for c in many.order[done:n]:
                    task = tasks[c]
                    k = e11.matched_steps(len(rtl.linker_candidates(task, rtl.mentions_of(task, ctx.fillers), ctx.typing)))
                    e11.gradient_steps(run, scorer.adapter if scorer is not None else run.adapter, task.text, entry_of[c], steps=k,
                                       lr=gradient_lr, optimizer=gradient_optimizer)
                    steps_total += k
                    tokens_total += k * len(run.tokenizer(task.text, add_special_tokens=False)["input_ids"])
                done = n
                log(f"  gradient: {n} terms learned")
                if scorer is not None:
                    fresh = e11.ItemScorer(run, read_set, ctx)
                    items_n[("gradient", n)] = fresh.score(none_frames, concepts=many.order[:n])
                if many.passages and scorer is not None:
                    with scorer.condition(none_frames):
                        passages[("gradient", n)] = passage_losses(run, scorer.adapter, passages_of[n])
                if many.kind == "heldout":
                    with e11.swapped_frames(run.channel, {entry_of[c]: None for c in many.order}) if compose else contextlib.nullcontext():
                        windows[("gradient", n)] = window_losses(run, corpus_path, wsets[n])
        costs["gradient"] = {"steps": steps_total, "train_tokens": tokens_total, "train_flops": 6.0 * params * tokens_total,
                             "context_tokens_per_use": 0.0}
    timings["gradient"] = time.monotonic() - t0
    costs["frames_per_use"] = {"context_tokens_per_use": 0.0, "prefill_flops_per_use": 0.0}
    return {"items": items, "items_n": items_n, "passages": passages, "windows": windows, "costs": costs, "presence": presence,
            "recall": recall, "frames": summary_frames, "sizes": sizes, "timings": timings,
            "seconds": time.monotonic() - started, "compose": compose, "parameters": params}


# -- summaries -------------------------------------------------------------------------------------------------------------

def item_table(result: dict[str, Any], many: ManySet) -> dict[str, dict[int, dict[str, float]]]:
    """condition → size → test → item id → value (N-independent conditions repeated per size on the first N terms)."""
    table: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(dict)
    for n in result["sizes"]:
        keep = set(many.first(n))
        for condition, rows in result["items"].items():
            table[condition][n] = _values(rows, keep)
        for (condition, size), rows in result["items_n"].items():
            if size == n and not condition.startswith("interference"):
                table[condition][n] = _values(rows, keep)
    return table


def summarize(result: dict[str, Any], many: ManySet, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    table = item_table(result, many)
    tests = ("property", "entailment", "paraphrase", "statement_accuracy")
    means = {cond: {str(n): {t: (float(np.mean(list(v[t].values()))) if v.get(t) else None) for t in tests} for n, v in by_n.items()}
             for cond, by_n in table.items()}
    pairs = [("frame:linker", "context:B2048"), ("frame:oracle", "context:B2048"), ("frame:linker", "context:B7168"),
             ("frame:oracle", "context:B7168"), ("frame:linker", "none"), ("frame:oracle", "none"), ("frame:linker-joint", "none"),
             ("frame:typeprior", "none"), ("context:B2048", "none"), ("rag:k1", "frame:linker"), ("rag:k3", "frame:linker"),
             ("gradient", "frame:linker"), ("gradient", "none"), ("frame:linker", "frame:typeprior")]
    contrasts = []
    for a, b in pairs:
        for n in result["sizes"]:
            if n not in table.get(a, {}) or n not in table.get(b, {}):
                continue
            for test in tests:
                va, vb = table[a][n].get(test, {}), table[b][n].get(test, {})
                ids = sorted(set(va) & set(vb))
                if ids:
                    ci = zs.paired_difference(np.asarray([va[i] for i in ids]), np.asarray([vb[i] for i in ids]),
                                              resamples=resamples, seed=seed)
                    contrasts.append({"a": a, "b": b, "size": n, "test": test, **ci})
    # passages / windows: loss after the new-term mentions, per condition and size; contrasts vs `none`
    texts: dict[str, Any] = {"passages": {}, "windows": {}, "contrasts": []}
    for (condition, n), per in result["passages"].items():
        s = sum(x for p in per for x, _ in p); c = sum(y for p in per for _, y in p)
        texts["passages"].setdefault(condition, {})[str(n)] = {"loss": s / c if c else None, "targets": c}
    for (condition, n), data in result["windows"].items():
        c = data["count_other"].sum()
        texts["windows"].setdefault(condition, {})[str(n)] = {"loss": float(data["sum_other"].sum() / c) if c else None, "targets": int(c)}
    for (condition, n), per in result["passages"].items():
        ref = result["passages"].get(("none", n))
        if condition == "none" or ref is None:
            continue
        d = np.asarray([sum(x for x, _ in p) - sum(x for x, _ in r) for p, r in zip(per, ref)])
        cnt = np.asarray([sum(y for _, y in r) for r in ref], dtype=float)
        base = np.asarray([sum(x for x, _ in r) for r in ref])
        if cnt.sum():
            texts["contrasts"].append({"kind": "passages", "condition": condition, "size": n,
                                       **paired_ratio_bootstrap(d, cnt, base, resamples=resamples, seed=seed)})
    for (condition, n), data in result["windows"].items():
        ref = result["windows"].get(("none", n))
        if condition == "none" or ref is None or ref["count_other"].sum() <= 0:
            continue
        texts["contrasts"].append({"kind": "windows", "condition": condition, "size": n,
                                   **paired_ratio_bootstrap(data["sum_other"] - ref["sum_other"], ref["count_other"], ref["sum_other"],
                                                            resamples=resamples, seed=seed)})
    # gradient forgetting: the first size's terms after each later size
    forgetting = {}
    first = result["sizes"][0] if result["sizes"] else None
    if first is not None and "gradient" in table:
        keep = set(many.first(first))
        for n in result["sizes"]:
            rows = result["items_n"].get(("gradient", n))
            if rows:
                v = _values(rows, keep).get("property", {})
                forgetting[str(n)] = float(np.mean(list(v.values()))) if v else None
    primary = next((c for c in contrasts if c["a"] == PRIMARY["a"] and c["b"] == PRIMARY["b"] and c["size"] == PRIMARY["size"]
                    and c["test"] == PRIMARY["test"]), None)
    return {"means": means, "contrasts": contrasts, "texts": texts, "forgetting_first_terms_property": forgetting,
            "primary_within_run": primary, "presence": result["presence"], "recall": result["recall"], "costs": result["costs"],
            "frames": result["frames"]}


def render(header: dict[str, Any], summary: dict[str, Any], result: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E11-M many terms — {header['track']} {header['kind']} — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Sizes {result['sizes']}; routes {header['routes']}. Frames are written for every term at once; in-context prompts "
             "carry the first N definitions up to the budget; retrieval puts BM25's top-k definitions in the prompt; the gradient "
             "route learns the definitions one after another.", "", "## Property accuracy by number of terms learned", "",
             "| condition | " + " | ".join(f"N={n}" for n in result["sizes"]) + " |", "|---|" + "---:|" * len(result["sizes"])]
    for cond, by_n in sorted(summary["means"].items()):
        lines.append(f"| {cond} | " + " | ".join(fmt(by_n.get(str(n), {}).get("property"), 3) for n in result["sizes"]) + " |")
    if summary.get("primary_within_run"):
        p = summary["primary_within_run"]
        lines += ["", f"Within-run primary contrast (`{PRIMARY['a']} − {PRIMARY['b']}`, N={PRIMARY['size']}, property): "
                      f"{p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], n = {p['n']} (the pooled test is `report`)."]
    if summary["presence"]:
        lines += ["", "Definitions that fit the budget: " + ", ".join(f"{k}: {v['present']}/{v['of']} ({v['tokens']} tokens)"
                                                                       for k, v in summary["presence"].items())]
    if summary["recall"]:
        lines += ["", "Retrieval recall of the term's own definition: " + ", ".join(f"{k}: {v:.3f}" for k, v in summary["recall"].items() if v is not None)]
    for kind in ("passages", "windows"):
        block = summary["texts"][kind]
        if block:
            lines += ["", f"## Loss after new-term mentions ({kind})", "", "| condition | " + " | ".join(f"N={n}" for n in result["sizes"]) + " |",
                      "|---|" + "---:|" * len(result["sizes"])]
            for cond, by_n in sorted(block.items()):
                lines.append(f"| {cond} | " + " | ".join(fmt(by_n.get(str(n), {}).get("loss"), 4) for n in result["sizes"]) + " |")
    if summary["forgetting_first_terms_property"]:
        lines += ["", "Gradient route, property accuracy of the first terms as more are learned: " +
                  ", ".join(f"N={k}: {fmt(v, 3)}" for k, v in summary["forgetting_first_terms_property"].items())]
    if summary["costs"].get("interference"):
        lines += ["", f"Frame interference check: `{json.dumps(summary['costs']['interference'])}`."]
    lines += ["", f"Timings (s): `{json.dumps({k: round(v, 1) for k, v in result['timings'].items()})}`; total {result['seconds']:.0f}s."]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    many = load_many(args.items)
    routes = [r for r in args.routes.split(",") if r]
    readers = [r for r in args.readers.split(",") if r]
    lr = args.gradient_lr
    if args.gradient_lr_from:
        lr = float(json.loads(Path(args.gradient_lr_from).read_text())["lr"])
    sizes = args.sizes or many.sizes
    config = {"experiment": "e11-many", "run": str(args.run), "items": str(args.items), "routes": routes, "readers": readers,
              "sizes": sizes, "budgets": args.budgets, "rag_k": args.rag_k, "gradient_lr": lr, "gradient_optimizer": args.gradient_optimizer,
              "gradient_backup": args.gradient_backup, "alias_table": str(args.alias_table) if args.alias_table else None,
              "smoke": bool(args.smoke)}
    if args.overwrite:
        clear_output(args.output)
        (Path(args.output) / "texts.npz").unlink(missing_ok=True)
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    if "gradient" in routes and run.config["model"].get("host_mode") == "frozen":
        routes = [r for r in routes if r != "gradient"]          # P0: the gradient comparator is C0′
    result = evaluate_many(run, many, routes=routes, readers=readers, budgets=args.budgets, rag_k=args.rag_k, gradient_lr=lr,
                           gradient_optimizer=args.gradient_optimizer, gradient_backup=args.gradient_backup, sizes=sizes)
    summary = summarize(result, many, resamples=args.resamples)
    header = {"source": run.describe(), "track": many.manifest["track"], "kind": many.kind, "routes": routes, "smoke": bool(args.smoke)}
    with (Path(args.output) / "predictions.jsonl").open("w") as handle:
        for cond, by_n in item_table(result, many).items():
            for n, by_test in by_n.items():
                for test, values in by_test.items():
                    for item_id, value in values.items():
                        handle.write(json.dumps({"condition": cond, "size": n, "test": test, "id": item_id, "value": value}) + "\n")
    arrays = {}
    for (cond, n), per in result["passages"].items():
        arrays[f"passages::{cond}::{n}::sum"] = np.asarray([sum(x for x, _ in p) for p in per])
        arrays[f"passages::{cond}::{n}::count"] = np.asarray([sum(y for _, y in p) for p in per])
    for (cond, n), data in result["windows"].items():
        arrays[f"windows::{cond}::{n}::sum"] = data["sum_other"]
        arrays[f"windows::{cond}::{n}::count"] = data["count_other"]
    if arrays:
        np.savez_compressed(Path(args.output) / "texts.npz", **arrays)
    write_json(Path(args.output) / "summary.json", {**header, **summary, "sizes": result["sizes"], "timings": result["timings"],
                                                    "seconds": result["seconds"], "parameters": result["parameters"]})
    (Path(args.output) / "report.md").write_text(("**SMOKE TEST — not a result.**\n\n" if args.smoke else "") + render(header, summary, result))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return summary


# -- pooled report -------------------------------------------------------------------------------------------------------------

def pooled_contrast(folders: Sequence[Path], a: str, b: str, size: int, test: str, *, resamples: int = 2000) -> dict[str, Any] | None:
    diffs = []
    for folder in folders:
        values: dict[str, dict[str, float]] = defaultdict(dict)
        for line in (Path(folder) / "predictions.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["size"] == size and row["test"] == test and row["condition"] in {a, b}:
                values[row["condition"]][row["id"]] = row["value"]
        diffs += [values[a][i] - values[b][i] for i in sorted(set(values[a]) & set(values[b]))]
    if not diffs:
        return None
    return {**zs.paired_difference(np.asarray(diffs), np.zeros(len(diffs)), resamples=resamples), "seeds": len(folders)}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """M1 (preregistration §14): T5 many-terms, SmolLM2-360M C5, N = 200, property, frame:linker − context:B2048, item
    differences of seeds 1–3 pooled; secondaries alongside."""
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    sizes_of: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    missing = []
    for folder in args.runs:
        path = Path(folder) / "summary.json"
        if not path.exists():
            missing.append(str(folder)); continue
        summary = json.loads(path.read_text())
        key = (summary["track"], summary["source"]["size"], summary["source"]["condition"])
        groups[key].append(Path(folder))
        sizes_of[key] |= set(summary.get("sizes") or SIZES)
    rows = []
    for (track, size, condition), folders in sorted(groups.items()):
        for a, b in (("frame:linker", "context:B2048"), ("frame:oracle", "context:B2048"), ("frame:linker", "context:B7168"),
                     ("frame:linker", "none"), ("frame:oracle", "none"), ("rag:k1", "frame:linker"), ("gradient", "frame:linker"),
                     ("context:B2048", "none"), ("gradient", "none"), ("rag:k1", "none")):
            for n in sorted(sizes_of[(track, size, condition)]):
                result = pooled_contrast(folders, a, b, n, "property", resamples=args.resamples)
                if result:
                    primary = (track == "t5" and "360M" in size and condition.startswith("C5") and (a, b) == (PRIMARY["a"], PRIMARY["b"])
                               and n == PRIMARY["size"])
                    rows.append({"track": track, "host": size, "model": condition, "a": a, "b": b, "size": n, "primary": primary, **result})
    out = {"rows": rows, "missing": missing}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", out)
    lines = ["# E11-M many terms — pooled report", "", "| track | host | model | a − b | N | property difference [95% CI] | seeds | p |",
             "|---|---|---|---|---:|---|---:|---:|"]
    for r in sorted(rows, key=lambda r: (not r["primary"], r["track"], r["a"], r["b"], r["size"])):
        mark = " **(M1)**" if r["primary"] else ""
        lines.append(f"| {r['track']} | {r['host']} | {r['model']} | {r['a']} − {r['b']}{mark} | {r['size']} | {r['mean']:+.4f} "
                     f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['seeds']} | {fmt(r['p_value'], 4)} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return out


# -- plan ----------------------------------------------------------------------------------------------------------------------

MANY_PRIORITY = 53                         # after tier 1 (52) and before its report (54): see preregistration §14.8
# Idle-GPU seconds per job at SmolLM2-360M, extrapolated from the E11 smoke timings (§12): readers on 400 definitions,
# 5 frame conditions, 2 budgets × 5 sizes in context (cached prefix), 2 k × 5 sizes retrieval, the gradient pass with
# 5 checkpoints, passages; ×0.6 for C0′ (no readers, no frames).
MANY_SECONDS = {("t5", "C5"): 2700, ("t5", "C0p"): 1600, ("t4", "C5"): 2400, ("t4", "C0p"): 1500}


def plan_jobs(*, seeds: Sequence[int] = (1, 2, 3), python: str = "$PY", priority: int = MANY_PRIORITY,
              runs_root: Path = e11.E9_RUNS) -> list[dict[str, Any]]:
    jobs = []
    sets = {"t5": ITEMS / "t5-many-smollm2-v1", "t4": ITEMS / "t4-many-smollm2-v1"}
    for track, items in sets.items():
        for model in ("C5", "C0p"):
            for seed in seeds:
                run = runs_root / track / f"SmolLM2-360M-full-{model}-s{seed}"
                routes = "frames,context,rag,gradient"
                readers = ",".join(READERS) if model == "C5" else "none"
                command = [python, "-m", "vsa_embed.experiments.e11_many", "evaluate", "--run", str(run), "--items", str(items),
                           "--routes", routes, "--readers", readers, "--alias-table",
                           str(Path("~/data/vsa-llm/e9/alias-tables").expanduser() / f"{track}.json"),
                           "--gradient-lr-from", str(e11.ROOT / "dev" / "SmolLM2-360M" / "gradient_lr.json"),
                           "--output", str(run / f"e11-many-{track}")]
                jobs.append({"name": f"e11-many-{track}-SmolLM2-360M-{model}-s{seed}", "priority": priority, "min_free_gb": 10,
                             "hours": round(MANY_SECONDS[(track, model)] / 3600, 2), "after_training": track == "t4" and seed > 1,
                             "command": command})
    jobs.append({"name": "e11-many-report", "priority": priority + 1, "min_free_gb": 1, "hours": 0.0,
                 "command": [python, "-m", "vsa_embed.experiments.e11_many", "report", "--runs",
                             *[j["command"][-1] for j in jobs], "--output", str(e11.ROOT / "report-many")]})
    return jobs


def run_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = plan_jobs(seeds=args.seeds, priority=args.priority)
    now = later = 0.0
    for job in jobs:
        line = (f"PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name {job['name']} --priority {job['priority']} --min-free-gb "
                f"{job['min_free_gb']} --no-resume -- {' '.join(job['command'])}   # ≈ {job['hours']:.2f} GPU-h")
        if job.get("after_training"):
            later += job["hours"]
            print("# after T4 seeds 2–3 train: " + line)
        else:
            now += job["hours"]
            print(line)
    print(f"# now ≈ {now:.1f} GPU-h; after T4 seeds 2–3 ≈ {later:.1f} GPU-h (idle-GPU estimates)")
    return jobs


# -- CLI -----------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="nested many-terms episodes over a read set")
    build.add_argument("--read-set", type=Path, required=True); build.add_argument("--output", type=Path, required=True)
    build.add_argument("--sizes", type=int, nargs="*", default=list(SIZES)); build.add_argument("--style", default=None)
    build.add_argument("--seed", type=int, default=0); build.add_argument("--passages", type=int, default=50)
    build.add_argument("--terms-per-passage", type=int, default=4)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True); ev.add_argument("--checkpoint", default="final.pt")
    ev.add_argument("--alias-table", type=Path, default=None); ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--device", default=None); ev.add_argument("--routes", default=",".join(ROUTES))
    ev.add_argument("--readers", default=",".join(READERS)); ev.add_argument("--sizes", type=int, nargs="*", default=None)
    ev.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS)); ev.add_argument("--rag-k", type=int, nargs="*", default=list(RAG_K))
    ev.add_argument("--gradient-lr", type=float, default=None); ev.add_argument("--gradient-lr-from", type=Path, default=None)
    ev.add_argument("--gradient-optimizer", choices=["adam", "sgd"], default="adam")
    ev.add_argument("--gradient-backup", choices=["gpu", "cpu"], default="gpu")
    ev.add_argument("--resamples", type=int, default=2000); ev.add_argument("--overwrite", action="store_true")
    ev.add_argument("--smoke", action="store_true")
    report = sub.add_parser("report"); report.add_argument("--runs", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True); report.add_argument("--resamples", type=int, default=2000)
    plan = sub.add_parser("plan"); plan.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    plan.add_argument("--priority", type=int, default=MANY_PRIORITY)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build_many_set(args.read_set, args.output, sizes=args.sizes, style=args.style, seed=args.seed,
                                        passages_per_size=args.passages, terms_per_passage=args.terms_per_passage), indent=2))
    elif args.command == "evaluate":
        summary = run_evaluate(args)
        print(json.dumps({"primary_within_run": summary.get("primary_within_run")}, indent=2, default=str))
    elif args.command == "report":
        print(json.dumps(run_report(args)["rows"][:5], indent=2, default=str))
    else:
        run_plan(args)


if __name__ == "__main__":
    main()
