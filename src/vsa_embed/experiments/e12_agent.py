"""E12 — 3a: agentic self-query (author decision 62; pre-registration `experiments/e12-self-query/preregistration.md` §10.1).
The **feasibility pilot** harness: can a base LM, prompted with three worked ReAct-style demonstrations, decide to call the
recall tool, make a well-formed call and use the result?

**Tools** (over the run's own store, `vsa_embed.self_query`; observations are phase A's recall texts):
- `recall[<term>]` — the term's whole frame (its slots decoded);
- `recall[<term>, <relation words>]` — one relation (e.g. `recall[Brightwater Ledger, depends on]`);
- `lookup[<relation words>, <value>]` — reverse lookup: the five stores of the track scoring highest for the pair.
A term is resolved through the run's alias table (normalized) or the item set's new terms (twins, new words) by surface; a
relation by its words (`depends on`, `owned by`) or name; a value by its text or the relation's answer wording.

**Episode.** Prompt = an instruction line, three worked demonstrations on *training* terms (a role query, a two-hop chain of
two calls, a reverse lookup; their observations are the tool's real outputs), then `Question: <the item's prompt> ___?
Options: a | b | …` and `Thought:`. Greedy decoding through the run's model (channel on: the text is re-linked at every
step); when the generated text completes an `Action: …[…]` line the harness appends `Observation: <tool output>`; the episode
ends at `Answer:` (or after 3 calls / 40 tokens without one, when `Answer:` is appended). The answer is read as **forced
choice**: the item's candidates scored after the episode's text (as phase A scores a candidate after its prompt).

**Baselines on the same questions:** `no_tool` (demonstrations that answer directly; `Answer:` right after the question) and
`fixed` (phase A's recall of the item in context before the question; `Answer:`).

**Measures:** call-format success (the first action parses as a tool call and names a known term, relation or value), call
relevance (the first call's term is the item's term — two-hop: the anchor; reverse: a lookup of the item's relation and
value, or a recall of one of its two terms), share of episodes with ≥ 1 call, share answering, forced-choice accuracy per
condition (twins: the twin contrast too, when both twins of a pair are in the set).

    python -m vsa_embed.experiments.e12_agent pilot --run RUN --twins DIR --understanding DIR --output OUT
        [--questions 50] [--host-dtype bfloat16] [--batch 4] [--label PILOT]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .. import self_query as sq
from ..span_channel import normalize_alias
from . import e9_binding_items as role_items
from . import e9_understanding as und
from .e5_common import E5Run, finish_output, json_ready, open_run, start_output, write_json
from .e12_faithfulness import TextCache, head_argmax
from .e12_self_query import (Condition, ContextBuilder, ItemSet, LoadedStore, frame_ids, host_view, lexicon_for_track, load_item_set,
                             load_store)

HEAD_CHUNK = 16384                       # the float32 output head in vocabulary slices (Qwen3's 151k-row head, ≤ 4 GB)

SCHEMA = "e12-agent/1"
ACTION = re.compile(r"Action:\s*(recall|lookup)\[([^\]\n]*)\]")
INSTRUCTION = ("You can look up what is stored about a term. recall[term] gives everything stored about the term; recall[term, "
               "relation] gives one relation of it; lookup[relation, value] gives the terms that have that value. Think, act, "
               "read the observation, then answer with one of the options.\n")
MAX_ACTIONS = 3
MAX_SEGMENT = 40


# ---------------------------------------------------------------- questions


@dataclass
class Question:
    id: str
    kind: str                        # twins | two_hop | reverse
    concept: str                     # the item's term (two-hop / reverse: the anchor)
    stem: str                        # the item's first prompt, filled
    candidates: list[str]            # rendered continuations (with their leading space)
    gold: int
    item_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        options = " | ".join(c.strip() for c in self.candidates)
        return f"Question: {self.stem} ___? Options: {options}\n"


def questions(twins: ItemSet | None, understanding: ItemSet | None, *, twin_pairs: int, two_hop: int, reverse: int,
              seed: int = 0) -> list[Question]:
    """The pilot's questions: both twins of the first `twin_pairs` pairs on their first relation (`choice`, first template),
    and the first `two_hop` / `reverse` items of the understanding set (one per anchor, seeded order)."""
    out: list[Question] = []
    if twins is not None:
        for p in twins.prompts:
            meta = p.row["meta"]
            if p.row["kind"] != "choice" or meta["pair"] >= twin_pairs or meta["role"] != "r1":
                continue
            out.append(Question(f"{p.id}-q", "twins", p.concept, und.render(p.templates[0], p.fills), list(p.candidates), p.gold, p.id,
                                {"pair": meta["pair"], "twin": meta["twin"], "relation": p.row["relation"]}))
    if understanding is not None:
        rng = random.Random(seed)
        for family, count in (("two_hop", two_hop), ("reverse", reverse)):
            pool = [p for p in understanding.prompts if p.row["family"] == family]
            rng.shuffle(pool)
            seen: set[str] = set()
            for p in pool:
                if len([q for q in out if q.kind == family]) >= count:
                    break
                if p.concept in seen:
                    continue
                seen.add(p.concept)
                out.append(Question(f"{p.id}-q", family, p.concept, und.render(p.templates[0], p.fills), list(p.candidates), p.gold, p.id,
                                    dict(p.item.get("meta", {})) | {"relation": p.row["relation"], "slots": p.item["slots"]}))
    return out


# ---------------------------------------------------------------- tools


class Toolbox:
    """The three tools over one store (module docstring)."""

    def __init__(self, builder: ContextBuilder, store: LoadedStore, table: Any) -> None:
        self.builder, self.store, self.table = builder, store, table
        self.writer = builder.writer()
        self.relation_names = builder.relation_names
        self.surface_concept = {normalize_alias(c["surface"]): cid for cid, c in builder.concepts.items() if c.get("frame")}
        phrases = {}
        for r, name in enumerate(self.relation_names):
            for form in {name, name.replace("_", " "), sq.relation_phrase(name)}:
                phrases[normalize_alias(form)] = r
                phrases[normalize_alias(form).removeprefix("is ")] = r
        self.relation_of = phrases
        lexicon = builder.lexicon
        values: dict[str, int] = {}
        for a, atom in enumerate(builder.atomic_names):
            text = lexicon.text(atom)
            if text:
                values.setdefault(normalize_alias(text), a)
                values.setdefault(normalize_alias(text).removeprefix("the "), a)
        self.value_of = values

    def term(self, name: str) -> tuple[str, torch.Tensor, list[int]] | None:
        """(display name, store vector, slots) of a term named in a call; None when unknown."""
        key = normalize_alias(name.strip().strip("\"'"))
        if key in self.surface_concept:
            cid = self.surface_concept[key]
            frame = self.builder.frames[cid]
            return self.builder.concepts[cid]["surface"], self.builder.vector(self.store, cid), [r for r, _ in frame]
        entry = self.table.alias_to_entry.get(key, self.table.alias_to_entry.get(key.removeprefix("the ")))
        if entry is None or len(self.table.entry_concepts[entry]) != 1:
            return None
        vector = self.store.store.entry_vectors()[entry]
        slots = [r for r, _ in self.store.store.frame(entry)]
        return name.strip(), vector, slots

    def relation(self, words: str) -> int | None:
        key = normalize_alias(words.strip())
        return self.relation_of.get(key, self.relation_of.get(key.removeprefix("is ")))

    def value(self, words: str) -> int | None:
        key = normalize_alias(words.strip().strip("\"'"))
        return self.value_of.get(key, self.value_of.get(key.removeprefix("the ")))

    def call(self, kind: str, arguments: str) -> tuple[str, dict[str, Any]]:
        """(observation text, parse record) of one call. A call that fails but would parse under the other tool's name
        (`lookup[term, relation]` for `recall[term, relation]`) is marked `swapped` — recorded, never repaired."""
        text, record = self._call(kind, arguments)
        if not record.get("parsed"):
            other = "lookup" if kind == "recall" else "recall"
            record["swapped"] = bool(self._call(other, arguments)[1].get("parsed"))
        return text, record

    def _call(self, kind: str, arguments: str) -> tuple[str, dict[str, Any]]:
        parts = [a.strip() for a in arguments.split(",")]
        if kind == "recall":
            name, relation_words = parts[0], ", ".join(parts[1:]) if len(parts) > 1 else None
            found = self.term(name)
            if found is None:
                return f"(nothing is stored about {name})", {"parsed": False, "term": name}
            display, vector, slots = found
            if relation_words:
                relation = self.relation(relation_words)
                if relation is None:
                    return f"(unknown relation: {relation_words})", {"parsed": False, "term": display, "relation_words": relation_words}
                line = self.store.store.decode_role(vector, relation)
                call = f"recall({display}, {sq.relation_phrase(self.relation_names[relation])}):"
                return self.writer.render(display, [line], call=call), {"parsed": True, "term": display, "relation": self.relation_names[relation]}
            lines = self.store.store.decode_slots(vector, slots)
            return self.writer.render(display, lines), {"parsed": True, "term": display}
        if len(parts) < 2:
            return "(lookup needs a relation and a value)", {"parsed": False}
        relation, value = self.relation(parts[0]), self.value(", ".join(parts[1:]))
        if relation is None or value is None:
            return f"(unknown {'relation' if relation is None else 'value'})", {"parsed": False, "relation_words": parts[0]}
        pool, surfaces, _ = self.builder._pool(self.store)
        top = self.store.store.reverse(relation, value, pool, k=5)
        filler = self.writer.filler_text(relation, value)
        lines = [sq.RecallLine(relation, [sq.Filler(value, s)], subject=surfaces[i]) for i, s in top]
        call = f"lookup({sq.relation_phrase(self.relation_names[relation])}, {filler}):"
        return self.writer.render(filler, lines, call=call), {"parsed": True, "relation": self.relation_names[relation],
                                                              "value": self.builder.atomic_names[value]}


# ---------------------------------------------------------------- demonstrations


def relation_pool(ontology: dict[str, Any], relation: int) -> list[int]:
    """The atomics observed under a relation (sorted)."""
    relations, fillers = np.asarray(ontology["relations"]), np.asarray(ontology["fillers"])
    return sorted(set(fillers[relations == relation].tolist()))


def demonstrations(builder: ContextBuilder, toolbox: Toolbox, exclude: set[int], *, tool: bool = True) -> str:
    """Three worked episodes on training terms: a role query (depends on), a two-hop chain (owned by → reports to), a
    reverse lookup (owned by). Their observations are the tool's real outputs."""
    o = builder.ontology
    rid = builder.relation_id
    frequency = np.asarray(o.get("train_frequency") or np.zeros(int(o["entry_count"])))
    held = {int(e) for e in o.get("heldout_entries", ())}
    names = builder.names
    store = toolbox.store.store
    atom_entry = builder.atom_entry
    lex = builder.lexicon
    table = toolbox.table

    def usable(e: int) -> bool:
        return frequency[e] >= 10 and e not in held and e not in exclude and len(table.entry_concepts[e]) == 1 \
            and table.alias_to_entry.get(normalize_alias(names[e])) == e

    def text(a: int) -> str:
        return builder.writer().filler_text(None, a)

    def decoded(e: int, relation: int) -> int | None:
        line = store.decode_role(store.entry_vectors()[e], relation)
        return line.fillers[0].atom if line.fillers else None

    def frame_of(e: int, relation: int) -> list[int]:
        return [f for r, f in store.frame(e) if r == relation]

    # Every demonstration is verified against the tool's real output: its recalls decode the gold fillers and its lookup
    # lists the right term and not the other, so no demonstration shows an answer the observation does not support.
    episodes = []
    entries = [e for e in range(int(o["entry_count"])) if usable(e)]
    depends, owned, reports = rid.get("depends_on", -1), rid.get("owned_by", -1), rid.get("reports_to", -1)
    role = next((e for e in entries if len(frame_of(e, depends)) == 1 and decoded(e, depends) == frame_of(e, depends)[0]), None)
    chain = None
    for e in entries:
        owners = frame_of(e, owned)
        if len(owners) != 1 or atom_entry[owners[0]] < 0 or decoded(e, owned) != owners[0]:
            continue
        team = int(atom_entry[owners[0]])
        divisions = frame_of(team, reports)
        if len(divisions) == 1 and decoded(team, reports) == divisions[0]:
            chain = (e, owners[0], team)
            break
    if role is not None:
        gold = frame_of(role, depends)[0]
        options = [gold] + [f for f in relation_pool(o, depends) if f != gold][:1]
        random.Random(1).shuffle(options)
        question = f"Question: {names[role]} depends on ___? Options: {' | '.join(text(a) for a in options)}\n"
        if tool:
            obs, _ = toolbox.call("recall", f"{names[role]}, depends on")
            episodes.append(question + f"Thought: I need what {names[role]} depends on.\nAction: recall[{names[role]}, depends on]\n"
                            f"Observation: {obs}\nThought: The recall says {text(gold)}.\nAnswer: {text(gold)}\n")
        else:
            episodes.append(question + f"Answer: {text(gold)}\n")
    if chain is not None:
        e, owner, team = chain
        division = frame_of(team, reports)[0]
        options = [division] + [f for f in relation_pool(o, reports) if f != division][:2]
        random.Random(2).shuffle(options)
        question = f"Question: {names[e]} is owned by a team that reports to ___? Options: {' | '.join(text(a) for a in options)}\n"
        if tool:
            first, _ = toolbox.call("recall", f"{names[e]}, owned by")
            second, _ = toolbox.call("recall", f"{names[team]}, reports to")
            episodes.append(question + f"Thought: First I need the team that owns {names[e]}.\nAction: recall[{names[e]}, owned by]\n"
                            f"Observation: {first}\nThought: Now I need what {names[team]} reports to.\n"
                            f"Action: recall[{names[team]}, reports to]\nObservation: {second}\n"
                            f"Thought: The second recall says {text(division)}.\nAnswer: {text(division)}\n")
        else:
            episodes.append(question + f"Answer: {text(division)}\n")
    # reverse: a team whose lookup lists one of its terms (the answer) and not a term owned by another team
    for team_atom in relation_pool(o, owned):
        holders = [e for e in entries if team_atom in frame_of(e, owned)]
        if not holders or lex.text(builder.atomic_names[team_atom]) is None:
            continue
        obs, record = toolbox.call("lookup", f"owned by, {lex.text(builder.atomic_names[team_atom])}")
        listed = [e for e in holders if f"- {names[e]} is owned by" in obs]
        other = next((x for x in entries if team_atom not in frame_of(x, owned) and frame_of(x, owned) and f"- {names[x]} " not in obs), None)
        if not record.get("parsed") or not listed or other is None:
            continue
        answer = listed[0]
        pair = [names[answer], names[other]]
        random.Random(3).shuffle(pair)
        question = (f"Question: Of {pair[0]} and {pair[1]}, the one owned by {text(team_atom)} is ___? "
                    f"Options: {pair[0]} | {pair[1]}\n")
        if tool:
            episodes.append(question + f"Thought: I should look up what {text(team_atom)} owns.\n"
                            f"Action: lookup[owned by, {lex.text(builder.atomic_names[team_atom])}]\nObservation: {obs}\n"
                            f"Thought: {names[answer]} is in the lookup and {names[other]} is not.\nAnswer: {names[answer]}\n")
        else:
            episodes.append(question + f"Answer: {names[answer]}\n")
        break
    return "\n".join(episodes)


# ---------------------------------------------------------------- generation


@torch.no_grad()
def next_tokens(adapter: Any, texts: Sequence[str]) -> list[int]:
    """Greedy next token of each text (right-padded batch; the channel's spans re-linked on the whole text; the output head
    in float32 at each text's last token)."""
    tokenizer, model, device = adapter.tokenizer, adapter.model, adapter.device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = tokenizer(list(texts), return_offsets_mapping=True, add_special_tokens=False, padding=True, return_tensors="pt")
    ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    spans = None
    if adapter.spans_fn is not None:
        offsets = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in zip(encoded["offset_mapping"], encoded["attention_mask"])]
        spans = {k: v.to(device) for k, v in adapter.spans_fn(list(texts), offsets).items()}
    with adapter._autocast():
        hidden = model.base(inputs_embeds=model.embed(ids, spans), attention_mask=mask).last_hidden_state
    last = mask.sum(1) - 1
    head = model.model.get_output_embeddings()
    with torch.autocast(device.type, enabled=False):
        return head_argmax(head, hidden[torch.arange(ids.shape[0], device=device), last], chunk=HEAD_CHUNK).tolist()


@dataclass
class Episode:
    question: Question
    text: str
    generated: str = ""
    segment: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    answered: bool = False
    tokens: int = 0


def run_episodes(adapter: Any, toolbox: Toolbox, prompts: Sequence[tuple[Question, str]], *, batch: int = 4,
                 log: Callable[[str], None] = print) -> list[Episode]:
    """Greedy ReAct episodes (module docstring), `batch` at a time."""
    tokenizer = adapter.tokenizer
    episodes = [Episode(q, p) for q, p in prompts]
    for start in range(0, len(episodes), batch):
        group = episodes[start:start + batch]
        while not all(e.done for e in group):
            active = [e for e in group if not e.done]
            for e, token in zip(active, next_tokens(adapter, [e.text for e in active])):
                piece = tokenizer.decode([token])
                e.text += piece; e.generated += piece; e.segment += piece; e.tokens += 1
                if token == tokenizer.eos_token_id:
                    e.text += "\nAnswer:"; e.done = True
                    continue
                if "Answer:" in e.segment:
                    cut = e.text.rfind("Answer:") + len("Answer:")
                    e.text, e.done, e.answered = e.text[:cut], True, True
                    continue
                if "\nQuestion:" in e.segment:                 # the model began another question
                    e.text = e.text[:e.text.rfind("\nQuestion:")] + "\nAnswer:"; e.done = True
                    continue
                match = ACTION.search(e.segment)
                if match:
                    observation, record = toolbox.call(match.group(1), match.group(2))
                    e.actions.append({"kind": match.group(1), "arguments": match.group(2), **record})
                    e.text = e.text[:len(e.text) - len(e.segment)] + e.segment[:match.end()] + f"\nObservation: {observation}\nThought:"
                    e.segment = ""
                    if len(e.actions) >= MAX_ACTIONS:
                        e.text += " I have what I need.\nAnswer:"; e.done = True
                    continue
                if len(tokenizer.encode(e.segment, add_special_tokens=False)) >= MAX_SEGMENT:
                    e.text += "\nAnswer:"; e.done = True
        log(f"  agent: episodes {min(start + batch, len(episodes))}/{len(episodes)}")
    return episodes


def answer_scores(adapter: Any, prefixes: Sequence[str], questions_: Sequence[Question]) -> list[np.ndarray]:
    """Per question: Σ log p of each candidate after its prefix (the text ending at `Answer:`); the float32 head in
    vocabulary slices (`e12_faithfulness.TextCache`)."""
    flat_prefixes = [p for p, q in zip(prefixes, questions_) for _ in q.candidates]
    flat_candidates = [c for q in questions_ for c in q.candidates]
    sums, _ = TextCache(adapter, head_chunk=HEAD_CHUNK).scores(list(zip(flat_prefixes, flat_candidates)))
    out, cursor = [], 0
    for q in questions_:
        out.append(sums[cursor:cursor + len(q.candidates)]); cursor += len(q.candidates)
    return out


# ---------------------------------------------------------------- the pilot


def relevant(question: Question, action: dict[str, Any], surfaces: dict[str, str]) -> bool:
    """The first call is about the item: the item's term (two-hop: the anchor), or for a reverse item a lookup of its relation
    and value or a recall of one of its two terms."""
    term = normalize_alias(action.get("term") or "")
    if question.kind in {"twins", "two_hop"}:
        return action["kind"] == "recall" and term == normalize_alias(surfaces[question.concept])
    names = {normalize_alias(surfaces[c]) for c in question.meta["slots"].values()}
    if action["kind"] == "lookup":
        return action.get("relation") == question.meta["relation"] and action.get("value") == question.meta.get("filler")
    return term in names


@contextlib.contextmanager
def host_dtype(name: str | None) -> Iterator[None]:
    """Within the block runs load their host in `name` (bfloat16: the 1.7B host in ≤ 4 GB; the LoRA adapters stay float32)."""
    if not name:
        yield
        return
    from ..training import lm
    saved = lm.host_dtype
    lm.host_dtype = lambda config: lm.HOST_DTYPES[name]
    try:
        yield
    finally:
        lm.host_dtype = saved


def pilot(run: E5Run, store: LoadedStore, twins: ItemSet | None, understanding: ItemSet | None, *, twin_pairs: int, two_hop: int,
          reverse: int, batch: int, seed: int = 0, part: tuple[int, int] = (0, 1), log: Callable[[str], None] = print) -> dict[str, Any]:
    family = run.config.get("e9_family") or "smollm2"
    track = run.config.get("e9_track") or "t5"
    lexicon = lexicon_for_track(track, family, run.ontology)
    sets = [s for s in (twins, understanding) if s is not None]
    concepts = [c for s in sets for c in s.concepts]
    combined = ItemSet("agent", sets[0].path, {"track": track, "family": family}, list({c["concept"]: c for c in concepts}.values()),
                       [p for s in sets for p in s.prompts])
    builder = ContextBuilder(combined, run.ontology, lexicon, {"own": store}, seed=seed)
    toolbox = Toolbox(builder, store, run.table)
    qs = questions(twins, understanding, twin_pairs=twin_pairs, two_hop=two_hop, reverse=reverse, seed=seed)
    index, parts = part                                # a contiguous slice (twins stay with their partner when the size is even)
    size = -(-len(qs) // parts)
    qs = qs[index * size:(index + 1) * size]
    exclude = {int(c["entry"]) for c in concepts if c.get("entry") is not None}
    demos_tool = demonstrations(builder, toolbox, exclude, tool=True)
    demos_plain = demonstrations(builder, toolbox, exclude, tool=False)
    surfaces = {cid: c["surface"] for cid, c in builder.concepts.items()}
    by_item = {p.id: p for p in combined.prompts}
    started = time.monotonic()
    with host_view(run, combined) as (adapter, _):
        agent_prompts = [(q, INSTRUCTION + "\n" + demos_tool + "\n" + q.text() + "Thought:") for q in qs]
        episodes = run_episodes(adapter, toolbox, agent_prompts, batch=batch, log=log)
        agent_scores = answer_scores(adapter, [e.text for e in episodes], qs)
        plain_scores = answer_scores(adapter, [INSTRUCTION + "\n" + demos_plain + "\n" + q.text() + "Answer:" for q in qs], qs)
        fixed_contexts = {}
        for kind in ("twins", "understanding"):
            items = twins if kind == "twins" else understanding
            if items is None:
                continue
            ctx = ContextBuilder(items, run.ontology, lexicon, {"own": store}, seed=seed)
            texts, _ = ctx.build(Condition("recall", "own"))
            fixed_contexts.update(texts)
        fixed_scores = answer_scores(adapter, [INSTRUCTION + "\n" + demos_plain + "\n" + fixed_contexts.get(q.item_id, "") + "\n" + q.text()
                                               + "Answer:" for q in qs], qs)
    seconds = time.monotonic() - started
    rows = []
    for q, e, a, p, f in zip(qs, episodes, agent_scores, plain_scores, fixed_scores):
        first = e.actions[0] if e.actions else None
        rows.append({"id": q.id, "kind": q.kind, "gold": q.gold, "meta": json_ready(q.meta), "actions": e.actions, "answered": e.answered,
                     "tokens": e.tokens, "format_ok": bool(first and first.get("parsed")), "called": bool(e.actions),
                     "swapped": bool(first and not first.get("parsed") and first.get("swapped")),
                     "relevant": bool(first and first.get("parsed") and relevant(q, first, surfaces)),
                     "scores": {"agent": a.tolist(), "no_tool": p.tolist(), "fixed": f.tolist()},
                     "correct": {k: float(np.argmax(v) == q.gold) for k, v in (("agent", a), ("no_tool", p), ("fixed", f))},
                     "transcript": e.text[len(INSTRUCTION) + len(demos_tool) + 2:]})
    return {"rows": rows, "seconds": seconds, "demonstrations": demos_tool, "questions": len(qs), "item_rows": len(by_item)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("all", "twins", "two_hop", "reverse"):
        sub = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not sub:
            continue
        out[kind] = {"questions": len(sub), **{k: float(np.mean([r.get(k, False) for r in sub]))
                                               for k in ("called", "format_ok", "swapped", "relevant", "answered")},
                     "calls_per_episode": float(np.mean([len(r["actions"]) for r in sub])),
                     **{f"accuracy_{c}": float(np.mean([r["correct"][c] for r in sub])) for c in ("agent", "no_tool", "fixed")}}
    twins = defaultdict(dict)
    for r in rows:
        if r["kind"] == "twins":
            twins[r["meta"]["pair"]][r["meta"]["twin"]] = r
    contrast = defaultdict(list)
    for pair in twins.values():
        if set(pair) != {"A", "B"}:
            continue
        sign = 1.0 if pair["A"]["gold"] == 0 else -1.0
        for c in ("agent", "no_tool", "fixed"):
            a, b = pair["A"]["scores"][c], pair["B"]["scores"][c]
            z = sign * ((a[0] - a[1]) - (b[0] - b[1]))
            contrast[c].append(1.0 if z > 0 else 0.0 if z < 0 else 0.5)
    if contrast:
        out["twins_contrast"] = {c: float(np.mean(v)) for c, v in contrast.items()} | {"pairs": len(contrast["agent"])}
    return out


def render(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    lines = [f"# E12 3a — agentic self-query, feasibility pilot — {source['condition']} seed {source['seed']} ({source['size']})"
             + (f" — {header['label']}" if header.get("label") else ""), "",
             f"Host dtype: {header['host_dtype'] or 'as trained'}. Greedy decoding; at most {MAX_ACTIONS} calls, {MAX_SEGMENT} tokens per "
             "segment. Answers read as forced choice after `Answer:`.", "",
             "| set | questions | called | format ok | swapped tool | relevant | answered | calls / episode | agent | no tool | fixed pipeline |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, m in summary.items():
        if kind == "twins_contrast":
            continue
        lines.append(f"| {kind} | {m['questions']} | {m['called']:.2f} | {m['format_ok']:.2f} | {m.get('swapped', 0.0):.2f} | {m['relevant']:.2f} | "
                     f"{m['answered']:.2f} | {m['calls_per_episode']:.2f} | {m['accuracy_agent']:.2f} | {m['accuracy_no_tool']:.2f} | "
                     f"{m['accuracy_fixed']:.2f} |")
    lines += ["", "`format ok`: the first action parses as a call of its tool; `swapped tool`: it fails but would parse under the other "
              "tool's name (recorded, not repaired); `relevant`: the first call is about the item."]
    if "twins_contrast" in summary:
        t = summary["twins_contrast"]
        lines += ["", f"Twin contrast accuracy over {t['pairs']} pairs: agent {t['agent']:.2f}, no tool {t['no_tool']:.2f}, fixed {t['fixed']:.2f}."]
    return "\n".join(lines) + "\n"


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    part = tuple(int(x) for x in args.part.split("/")) if args.part else (0, 1)
    config = {"experiment": "e12-agent-pilot", "run": str(args.run), "store": str(args.store or args.run),
              "twins": str(args.twins) if args.twins else None, "understanding": str(args.understanding) if args.understanding else None,
              "twin_pairs": args.twin_pairs, "two_hop": args.two_hop, "reverse": args.reverse, "part": list(part), "batch": args.batch,
              "host_dtype": args.host_dtype, "seed": args.seed, "label": args.label}
    if args.overwrite:
        for name in ("summary.json", "episodes.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json"):
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    store = load_store(Path(args.store or args.run), "own")             # P0 hosts read a C5 run's store
    alias_table = args.alias_table
    if alias_table is None:
        import yaml

        from .e9_tracks import ensure_alias_table, track_spec
        run_config = yaml.safe_load((Path(args.run) / "resolved_config.yaml").read_text())
        alias_table = ensure_alias_table(track_spec(run_config.get("e9_track"), run_config.get("e9_family") or "smollm2"))
    with host_dtype(args.host_dtype):
        run = open_run(Path(args.run), device=args.device, batch_size=args.batch, max_length=4096, alias_table=alias_table)
    twins = load_item_set(args.twins, limit=args.twin_pairs) if args.twins else None
    understanding = load_item_set(args.understanding) if args.understanding else None
    result = pilot(run, store, twins, understanding, twin_pairs=args.twin_pairs, two_hop=args.two_hop, reverse=args.reverse,
                   batch=args.batch, seed=args.seed, part=part)
    summary = summarize(result["rows"])
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    header = {"source": run.describe(), "store": str(args.store or args.run), "host_dtype": args.host_dtype, "label": args.label,
              "part": list(part)}
    und.write_jsonl_gz(output / "episodes.jsonl.gz", result["rows"])
    write_json(output / "summary.json", {**header, "summary": summary, "seconds": result["seconds"], "peak_gb": peak,
                                         "demonstrations": result["demonstrations"]})
    (output / "report.md").write_text(render(summary, header) + "\n## Demonstrations (tool)\n\n```\n" + result["demonstrations"] + "```\n")
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {"output": str(output), "summary": summary, "seconds": result["seconds"], "peak_gb": peak}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("pilot", help="the 3a feasibility pilot on one run (GPU)")
    pi.add_argument("--run", type=Path, required=True); pi.add_argument("--output", type=Path, required=True)
    pi.add_argument("--twins", type=Path, default=None); pi.add_argument("--understanding", type=Path, default=None)
    pi.add_argument("--twin-pairs", type=int, default=10); pi.add_argument("--two-hop", type=int, default=15)
    pi.add_argument("--reverse", type=int, default=15); pi.add_argument("--batch", type=int, default=4)
    pi.add_argument("--host-dtype", default=None, choices=[None, "bfloat16", "float16"])
    pi.add_argument("--alias-table", type=Path, default=None); pi.add_argument("--device", default=None)
    pi.add_argument("--seed", type=int, default=0); pi.add_argument("--label", default=None)
    pi.add_argument("--store", type=Path, default=None, help="the run whose store the tools read (default: the run itself)")
    pi.add_argument("--part", default="", help="K/N: the K-th of N contiguous slices of the questions (jobs under 5 minutes)")
    pi.add_argument("--overwrite", action="store_true")
    me = sub.add_parser("merge", help="merge pilot parts into one summary and report")
    me.add_argument("--inputs", type=Path, nargs="+", required=True); me.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        merged = merge(args.inputs, args.output)
        print(json.dumps({"output": str(args.output), "summary": merged["summary"]}, indent=2))
        return
    result = run_pilot(args)
    print(json.dumps({"output": result["output"], "seconds": round(result["seconds"], 1), "peak_gb": result["peak_gb"],
                      "summary": result["summary"]}, indent=2))


def merge(inputs: Sequence[Path], output: Path) -> dict[str, Any]:
    """One summary and report from pilot parts (episodes concatenated; the header of the first part)."""
    rows, header, seconds, peaks = [], None, 0.0, []
    for folder in inputs:
        document = json.loads((Path(folder) / "summary.json").read_text())
        header = header or {k: document[k] for k in ("source", "store", "host_dtype", "label") if k in document}
        seconds += float(document.get("seconds", 0.0))
        peaks.append(document.get("peak_gb"))
        rows += und.read_jsonl(Path(folder) / "episodes.jsonl")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    und.write_jsonl_gz(output / "episodes.jsonl.gz", rows)
    write_json(output / "summary.json", {**header, "summary": summary, "seconds": seconds, "peak_gb": max([p for p in peaks if p] or [0.0]),
                                         "parts": [str(p) for p in inputs]})
    (output / "report.md").write_text(render(summary, header))
    return {"summary": summary}


if __name__ == "__main__":
    main()
