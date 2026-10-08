"""E12 — faithfulness of decoding (F1; author decision 62; pre-registration `experiments/e12-self-query/preregistration.md`
§6). Evaluation only, on finished E9 runs; the channel route only (no recall in context).

The recall tool says what a term's store holds. Is that what the model computes with? For every **decoded edge** of a term's
store — an edge whose filler the run's own recall (`vsa_embed.self_query`, static bundle, typed cleanup, the operator's primary
unbinding) puts in the edge's slot — the store is intervened on and the model's preference for the edge's filler is read on
the term's property items of the edge's relation (the E9 v2 new words: inserted frames, so the host knows the terms only
through the channel):

- **comprehensiveness:** remove the edge — decision 24's semantics: its weight is zeroed in the composer's weighted sum, the
  other weights kept (the bundle is normalized, so this renormalizes them; attention is not recomputed) — against the
  **matched random-edge control**: the mean over every other single-edge removal of the same term (the expectation of
  removing a uniformly drawn other edge). Per edge the gap `g = Δ_control − Δ_edge` in the margin `m(f) = mean over
  templates of [s(f) − mean_{c ≠ f} s(c)]` (summed log-probabilities); `g > 0`: removing the decoded edge lowers the
  preference for its filler more than removing another edge.
- **sufficiency:** keep only the edge (every other weight zeroed) against keeping only another edge: `m(f | keep e) − mean_j
  m(f | keep e_j)`.
- **specificity:** under the removal of e, the term's items of the *other* relations (their gold fillers): the edge is
  specific when its own item moves in the predicted direction by more than τ and by more than any other relation's item.
- **role specificity** (role-swap twins): remove (r1, X) from twin A; X's preference drops under r1 (its role) and, for a
  role-blind reader, equally under r2 (where X is the distractor). `RS = Δ_r2(X) − Δ_r1(X)` > 0 when the model reads the role;
  the swap of the twins' frames (step 1's `swap`) is recomputed with the same scorer.

**Thresholds** (pre-registered): an edge *moves* when its gap exceeds τ = 0.05 nats (sensitivity τ ∈ {0, 0.02, 0.1}); the
per-term outcome is the **net share** `mean_e [1(g > τ) − 1(g < −τ)]` over the term's decoded edges (0 when removing the decoded
edge is exchangeable with removing another edge), reported with the share `mean 1(g > τ)` itself. Undecoded edges get the
same measures (secondary).

Outputs (`RUN/self-query-faithfulness/`): `summary.json` (per term and per twin pair), `edges.jsonl.gz` (per edge: decode,
margins, gaps), `report.md`, `resolved_config.yaml`, `manifest.json`.

    python -m vsa_embed.experiments.e12_faithfulness evaluate --run RUN [--new-items DIR] [--twins DIR] [--limit N]
    python -m vsa_embed.experiments.e12_faithfulness queue --stage t5 --priority 50 [--models C5 C5ut] [--seeds …] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import yaml

from .. import self_query as sq
from . import e9_binding_items as role_items
from . import e9_ontology_edit as edit
from . import e9_understanding as und
from .e5_common import E5Run, edge_weight_hook, finish_output, json_ready, open_run, start_output, write_json
from .e12_self_query import RUN_NAME, frame_ids, load_store

SCHEMA = "e12-faithfulness/1"
ROOT = Path("experiments/e9-retrofit")
OUTPUT = "self-query-faithfulness"
TAU = 0.05
TAUS = (0.0, 0.02, 0.05, 0.1)
NEW_ITEMS = {("t5", "smollm2"): "new-words-t5-smollm2-v2", ("t5", "qwen3"): "new-words-t5-qwen3-v2", ("t4", "smollm2"): "new-words-t4-smollm2-v1"}
TWINS = {("t5", "smollm2"): "role-twins-t5-smollm2-v1", ("t5", "qwen3"): "role-twins-t5-qwen3-v1"}
RESULT_FILES = ("summary.json", "edges.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json")


# ---------------------------------------------------------------- scoring


def head_logprobs(head: Any, hidden: torch.Tensor, targets: torch.Tensor, *, chunk: int | None = None) -> torch.Tensor:
    """`log p(target)` of each row under the output head in float32; `chunk` computes the logits in vocabulary slices (the
    same values without a float32 copy of a large head: Qwen3's 151k × 2,048 is 1.2 GB)."""
    from torch.nn import functional as F
    hidden = hidden.float()
    bias = getattr(head, "bias", None)
    if not chunk:
        logits = F.linear(hidden, head.weight.float(), None if bias is None else bias.float())
        return torch.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1)
    vocab = head.weight.shape[0]
    total = torch.full((hidden.shape[0],), float("-inf"), device=hidden.device)
    picked = torch.zeros(hidden.shape[0], device=hidden.device)
    for start in range(0, vocab, chunk):
        part = F.linear(hidden, head.weight[start:start + chunk].float(), None if bias is None else bias[start:start + chunk].float())
        total = torch.logaddexp(total, torch.logsumexp(part, -1))
        inside = (targets >= start) & (targets < start + part.shape[1])
        if bool(inside.any()):
            picked[inside] = part[inside, targets[inside] - start]
    return picked - total


def head_argmax(head: Any, hidden: torch.Tensor, *, chunk: int | None = None) -> torch.Tensor:
    """The float32 output head's argmax token of each row (`chunk`: in vocabulary slices)."""
    from torch.nn import functional as F
    hidden = hidden.float()
    bias = getattr(head, "bias", None)
    if not chunk:
        return F.linear(hidden, head.weight.float(), None if bias is None else bias.float()).argmax(-1)
    best = torch.full((hidden.shape[0],), float("-inf"), device=hidden.device)
    index = torch.zeros(hidden.shape[0], dtype=torch.long, device=hidden.device)
    for start in range(0, head.weight.shape[0], chunk):
        part = F.linear(hidden, head.weight[start:start + chunk].float(), None if bias is None else bias[start:start + chunk].float())
        value, where = part.max(-1)
        better = value > best
        best = torch.where(better, value, best); index = torch.where(better, where + start, index)
    return index


class TextCache:
    """Texts tokenized and linked once, scored many times: an intervention changes the composer, never the texts. Scores
    are `e9_binding_items.continuation_scores`'s (Σ log p of the continuation tokens after the prefix; the same right-padded
    batches, spans, bf16 autocast and float32 output head; `last_hidden_state` is the final hidden state the adapter
    reads). `head_chunk` computes the float32 head in vocabulary slices (memory)."""

    def __init__(self, adapter: Any, *, head_chunk: int | None = None) -> None:
        self.adapter = adapter
        self.head_chunk = head_chunk
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    def _prepare(self, pairs: Sequence[tuple[str, str]]) -> None:
        missing = [pair for pair in dict.fromkeys(pairs) if pair not in self.rows]
        if not missing:
            return
        adapter = self.adapter
        texts = [p + c for p, c in missing]
        encoded = adapter.tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False, truncation=True,
                                    max_length=adapter.max_length)
        for (prefix, _), text, ids, offsets in zip(missing, texts, encoded["input_ids"], encoded["offset_mapping"]):
            offsets = [tuple(o) for o in offsets]
            spans = adapter.spans_fn([text], [offsets]) if adapter.spans_fn is not None else None
            targets = [(t - 1, int(ids[t])) for t in range(1, len(ids)) if offsets[t][1] > len(prefix)]
            self.rows[(prefix, text[len(prefix):])] = {"ids": list(ids), "spans": spans, "targets": targets}

    @torch.no_grad()
    def scores(self, pairs: Sequence[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """(Σ log p, token count) of each (prefix, continuation) pair."""
        from torch.nn import functional as F
        self._prepare(pairs)
        adapter = self.adapter
        model, device = adapter.model, adapter.device
        sums, counts = np.zeros(len(pairs)), np.zeros(len(pairs))
        order = sorted(range(len(pairs)), key=lambda i: len(self.rows[pairs[i]]["ids"]))
        pad = adapter.tokenizer.pad_token_id if adapter.tokenizer.pad_token_id is not None else adapter.tokenizer.eos_token_id
        head = model.model.get_output_embeddings()
        for start in range(0, len(order), adapter.batch_size):
            chunk = order[start:start + adapter.batch_size]
            rows = [self.rows[pairs[i]] for i in chunk]
            width = max(len(r["ids"]) for r in rows)
            ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
            mask = torch.zeros((len(rows), width), dtype=torch.long)
            for b, r in enumerate(rows):
                ids[b, :len(r["ids"])] = torch.tensor(r["ids"]); mask[b, :len(r["ids"])] = 1
            spans = None
            if adapter.spans_fn is not None:
                parts = [{**r["spans"], "batch": torch.full_like(r["spans"]["batch"], b)} for b, r in enumerate(rows)]
                spans = {k: torch.cat([p[k] for p in parts]).to(device) for k in parts[0]}
            with adapter._autocast():
                embeddings = model.embed(ids.to(device), spans) if hasattr(model, "embed") else model.get_input_embeddings()(ids.to(device))
                base = model.base if hasattr(model, "embed") else model.base_model
                hidden = base(inputs_embeds=embeddings, attention_mask=mask.to(device)).last_hidden_state
            select = [(b, col, token, i) for b, (r, i) in enumerate(zip(rows, chunk)) for col, token in r["targets"]]
            if not select:
                continue
            with torch.autocast(device.type, enabled=False):
                picked_hidden = hidden[torch.tensor([s[0] for s in select], device=device), torch.tensor([s[1] for s in select], device=device)].float()
                picked = head_logprobs(head, picked_hidden, torch.tensor([s[2] for s in select], device=device), chunk=self.head_chunk)
            owners = [s[3] for s in select]
            np.add.at(sums, owners, picked.cpu().double().numpy())
            np.add.at(counts, owners, 1.0)
        return sums, counts


def item_scores(adapter: Any, items: Sequence[dict[str, Any]], surfaces: dict[str, str], *, per_token: bool = False,
                cache: TextCache | None = None) -> dict[str, np.ndarray]:
    """Per item: the (templates × candidates) summed log-probability of each candidate after the prompt (cloze: per token);
    with a `TextCache`, its tokenized and linked texts are reused."""
    prefixes, continuations, shapes = [], [], []
    for item in items:
        for template in item["templates"]:
            prefix = und.render(template, {"x": surfaces[item["concept"]]})
            for candidate in item["candidates"]:
                prefixes.append(prefix); continuations.append(candidate)
        shapes.append((len(item["templates"]), len(item["candidates"])))
    if not prefixes:
        sums, counts = np.zeros(0), np.zeros(0)
    elif cache is not None:
        sums, counts = cache.scores(list(zip(prefixes, continuations)))
    else:
        sums, counts = role_items.continuation_scores(adapter, prefixes, continuations)
    values = sums / np.maximum(counts, 1) if per_token else sums
    out, cursor = {}, 0
    for item, (n_t, k) in zip(items, shapes):
        out[item["id"]] = values[cursor:cursor + n_t * k].reshape(n_t, k)
        cursor += n_t * k
    return out


def margin(s: np.ndarray, candidate: int) -> float:
    """`mean over templates of [s(f) − mean_{c ≠ f} s(c)]`."""
    others = np.delete(s, candidate, axis=1)
    return float(np.mean(s[:, candidate] - others.mean(1)))


@contextlib.contextmanager
def dropped_edges(composer: Any, drop: torch.Tensor) -> Iterator[None]:
    """Within the block the composer's weight of every schedule edge marked in `drop` (bool, one per schedule edge) is zero;
    the other weights are unchanged (decision 24: erased from the weighted sum, attention not recomputed)."""
    def transform(weights: torch.Tensor, edge_index: torch.Tensor, segments: torch.Tensor, concept_ids: torch.Tensor) -> torch.Tensor:
        return weights.masked_fill(drop.to(edge_index.device)[edge_index], 0.0)
    with edge_weight_hook(composer, transform):
        yield


def drop_mask(offsets: torch.Tensor, entries: Sequence[int], *, remove: int | None = None, keep: int | None = None) -> torch.Tensor:
    """Over the schedule's edges: for each entry with more than `remove` (`keep`) edges, its `remove`-th edge (every edge but
    its `keep`-th)."""
    mask = torch.zeros(int(offsets[-1]), dtype=torch.bool)
    for e in entries:
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if remove is not None and remove < hi - lo:
            mask[lo + remove] = True
        if keep is not None and keep < hi - lo:
            mask[lo:hi] = True
            mask[lo + keep] = False
    return mask


# ---------------------------------------------------------------- decoded edges


def decoded_edges(store: sq.RecallStore, frame: Sequence[tuple[int, int]], vector: torch.Tensor) -> list[dict[str, Any]]:
    """Per edge of `frame` (in frame order): whether the recall puts its filler in its slot (role-blind store: among the
    recalled fillers, without a role), with the cleanup score."""
    lines = store.decode_slots(vector, [r for r, _ in frame])
    recalled: dict[int | None, dict[int, float]] = defaultdict(dict)
    for line in lines:
        for f in line.fillers:
            recalled[line.relation][f.atom] = f.score
    out = []
    for r, f in frame:
        found = recalled.get(r, {}) if not store.role_blind else recalled.get(None, {})
        out.append({"relation": int(r), "filler": int(f), "decoded": f in found, "score": found.get(f)})
    return out


# ---------------------------------------------------------------- the evaluation


def _new_word_terms(items_dir: Path, ontology: dict[str, Any], *, limit: int | None, offset: int = 0) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    _, concepts, items = edit.load_item_dir(items_dir, edit.SCHEMA_NEW)
    concepts = concepts[offset:offset + limit] if limit else concepts[offset:]
    keep = {c["concept"] for c in concepts}
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["concept"] in keep and item["test"] == "property":
            by_concept[item["concept"]].append(item)
    return concepts, by_concept


def faithfulness_new_words(run: E5Run, store: sq.RecallStore, store_ontology: dict[str, Any], items_dir: Path, *,
                           limit: int | None = None, offset: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Comprehensiveness, sufficiency and specificity on the new words' property items (module docstring)."""
    ontology = run.ontology
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    concepts, items_by = _new_word_terms(items_dir, ontology, limit=limit, offset=offset)
    frames = [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in concepts]
    store_rel = {n: i for i, n in enumerate(store_ontology["relation_names"])}
    store_atom = {n: i for i, n in enumerate(store_ontology["atomic_names"])}
    decodes = [decoded_edges(store, f, store.frame_vector(frame_ids(c["frame"], store_rel, store_atom))) for c, f in zip(concepts, frames)]
    base = int(ontology["entry_count"])
    entry_of = {c["concept"]: base + i for i, c in enumerate(concepts)}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    adapter = edit.extended_adapter(run, {c["surface"]: entry_of[c["concept"]] for c in concepts})
    cache = TextCache(adapter)
    resolved = edit.link_check(adapter, concepts, [i for c in concepts for i in items_by[c["concept"]]], entry_of)
    linked = [c for c in concepts if resolved[c["concept"]]["status"] == "linked"]
    degrees = {c["concept"]: len(f) for c, f in zip(concepts, frames)}
    d_max = max(degrees.values(), default=0)
    channel = run.channel
    passes: dict[str, dict[str, np.ndarray]] = {}
    started = time.monotonic()
    with edit.inserted_entries(channel, len(concepts), frames):
        composer = channel.composer
        offsets = composer.schedule.offsets.cpu()

        def score(name: str, affected: Sequence[dict[str, Any]], drop: torch.Tensor | None) -> None:
            items = [i for c in affected for i in items_by[c["concept"]]]
            with dropped_edges(composer, drop) if drop is not None else contextlib.nullcontext():
                passes[name] = item_scores(adapter, items, surfaces, cache=cache)

        score("full", linked, None)
        for k in range(d_max):
            affected = [c for c in linked if degrees[c["concept"]] > k]
            ids = [entry_of[c["concept"]] for c in affected]
            score(f"remove{k}", affected, drop_mask(offsets, ids, remove=k))
            score(f"keep{k}", affected, drop_mask(offsets, ids, keep=k))
            log(f"  faithfulness: edge position {k + 1}/{d_max} ({len(affected)} terms, {time.monotonic() - started:.0f} s)")
    from .e12_self_query import lexicon_for_track
    lexicon = lexicon_for_track(run.config.get("e9_track"), run.config.get("e9_family") or "smollm2", ontology)
    edges = []
    linked_ids = {c["concept"] for c in linked}
    for c, frame, decode in zip(concepts, frames, decodes):
        if c["concept"] not in linked_ids:
            continue
        cid = c["concept"]
        items = items_by[cid]
        by_relation = {i["relation"]: i for i in items}
        full = passes["full"]
        d = len(frame)
        for k, ((r, f), dec) in enumerate(zip(frame, decode)):
            name = ontology["relation_names"][r]
            item = by_relation.get(name)
            text = lexicon.text(ontology["atomic_names"][f])
            # the relation's property item tests this edge when its gold candidate words this edge's filler (the builder's
            # gold is the relation's first filler; a second filler of the relation is no candidate)
            if item is None or not text or item["candidates"][int(item["gold"])] != lexicon.answer(name, text):
                continue
            cand = int(item["gold"])
            m_full = margin(full[item["id"]], cand)
            delta = margin(passes[f"remove{k}"][item["id"]], cand) - m_full
            others = [j for j in range(d) if j != k]
            control = float(np.mean([margin(passes[f"remove{j}"][item["id"]], cand) - m_full for j in others])) if others else 0.0
            keep_own = margin(passes[f"keep{k}"][item["id"]], cand)
            keep_other = float(np.mean([margin(passes[f"keep{j}"][item["id"]], cand) for j in others])) if others else keep_own
            off = []
            for other in items:
                if other["id"] == item["id"] or relation_id.get(other["relation"]) == r:
                    continue
                g = int(other["gold"])
                off.append(margin(passes[f"remove{k}"][other["id"]], g) - margin(full[other["id"]], g))
            edges.append({"concept": cid, "position": k, "relation": name, "filler": ontology["atomic_names"][f],
                          "decoded": bool(dec["decoded"]), "score": dec["score"], "degree": d, "item": item["id"],
                          "m_full": m_full, "delta": delta, "control": control, "gap": control - delta,
                          "sufficiency": keep_own - keep_other, "off_target": off, "off_max": float(max(np.abs(off))) if off else 0.0})
    return {"edges": edges, "resolved": resolved, "concepts": len(concepts), "linked": len(linked), "d_max": d_max,
            "decode": float(np.mean([e["decoded"] for dec in decodes for e in dec])) if decodes else None,
            "seconds": time.monotonic() - started}


def role_specificity(run: E5Run, store: sq.RecallStore, store_ontology: dict[str, Any], twins_dir: Path, *, limit: int | None = None,
                     log: Callable[[str], None] = print) -> dict[str, Any]:
    """Twins: remove (r1, X) or (r2, Y) from each twin's store and compare the removed filler's preference under its own role
    with its preference under the other role; plus the swap of the twins' frames (module docstring)."""
    manifest, concepts, items = role_items.load_items(twins_dir)
    if limit:
        concepts = [c for c in concepts if c["pair"] < limit]
    keep = {c["concept"] for c in concepts}
    items = [i for i in items if i["concept"] in keep and i["kind"] == "choice"]
    ontology = run.ontology
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    frames = [edit.resolve_frame(c["frame"], relation_id, atomic_id) for c in concepts]
    index = {c["concept"]: i for i, c in enumerate(concepts)}
    swapped = [frames[index[c["partner"]]] for c in concepts]
    base = int(ontology["entry_count"])
    ids = {c["concept"]: base + i for i, c in enumerate(concepts)}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    adapter = edit.extended_adapter(run, {c["surface"]: ids[c["concept"]] for c in concepts})
    cache = TextCache(adapter)
    resolved = edit.link_check(adapter, concepts, items, ids)
    store_rel = {n: i for i, n in enumerate(store_ontology["relation_names"])}
    store_atom = {n: i for i, n in enumerate(store_ontology["atomic_names"])}
    positions = {}
    for c, frame in zip(concepts, frames):
        r1, r2 = (relation_id[r] for r in c["relations"])
        positions[c["concept"]] = {name: next(k for k, (r, _) in enumerate(frame) if r == rel) for name, rel in (("r1", r1), ("r2", r2))}
    decode = {c["concept"]: decoded_edges(store, f, store.frame_vector(frame_ids(c["frame"], store_rel, store_atom)))
              for c, f in zip(concepts, frames)}
    passes: dict[str, dict[str, np.ndarray]] = {}
    started = time.monotonic()
    channel = run.channel
    with edit.inserted_entries(channel, len(concepts), frames):
        offsets = channel.composer.schedule.offsets.cpu()
        passes["full"] = item_scores(adapter, items, surfaces, cache=cache)
        for role in ("r1", "r2"):
            mask = torch.zeros(int(offsets[-1]), dtype=torch.bool)
            for c in concepts:
                mask[int(offsets[ids[c["concept"]]]) + positions[c["concept"]][role]] = True
            with dropped_edges(channel.composer, mask):
                passes[f"remove_{role}"] = item_scores(adapter, items, surfaces, cache=cache)
    with edit.inserted_entries(channel, len(concepts), swapped):
        passes["swap"] = item_scores(adapter, items, surfaces, cache=cache)
    by_key = {(i["concept"], i["relation"]): i for i in items}
    rows = []
    for c in concepts:
        if resolved[c["concept"]]["status"] != "linked":
            continue
        r1, r2 = c["relations"]
        mine = {"r1": r1, "r2": r2}
        for role, other in (("r1", "r2"), ("r2", "r1")):
            own_item, other_item = by_key[(c["concept"], mine[role])], by_key[(c["concept"], mine[other])]
            removed = own_item["gold"]                       # the removed edge's filler: the gold of its own role's item
            delta_own = margin(passes[f"remove_{role}"][own_item["id"]], removed) - margin(passes["full"][own_item["id"]], removed)
            delta_other = margin(passes[f"remove_{role}"][other_item["id"]], removed) - margin(passes["full"][other_item["id"]], removed)
            dec = decode[c["concept"]][positions[c["concept"]][role]]
            rows.append({"concept": c["concept"], "pair": c["pair"], "twin": c["twin"], "role": role, "relation": mine[role],
                         "decoded": bool(dec["decoded"]), "delta_own": delta_own, "delta_other": delta_other, "rs": delta_other - delta_own})
    twins = {c["concept"]: c for c in concepts}
    linked = {c for c, r in resolved.items() if r["status"] == "linked"}
    contrast_full = role_items.twin_scores(_rows_for(items, passes["full"]), "choice")
    contrast_swap = role_items.twin_scores(_rows_for(items, passes["swap"]), "choice")
    complete = {c["pair"] for c in concepts if c["concept"] in linked and c["partner"] in linked}
    swap = {str(p): {"own": contrast_full[p]["contrast"], "swap": contrast_swap[p]["contrast"]} for p in contrast_full if p in complete and p in contrast_swap}
    return {"rows": rows, "swap": swap, "resolved": resolved, "pairs": len(complete), "seconds": time.monotonic() - started,
            "twins": {k: {"pair": v["pair"]} for k, v in twins.items()}}


def _rows_for(items: Sequence[dict[str, Any]], scores: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """`e9_binding_items.score_items`-shaped rows (s only; correctness from the raw argmax) for `twin_scores`."""
    rows = []
    for item in items:
        s = scores[item["id"]]
        correct = [(1.0 if int(np.argmax(row)) == int(item["gold"]) else 0.0) if np.ptp(row) > 0 else 0.5 for row in s]
        rows.append({"id": item["id"], "kind": item["kind"], "concept": item["concept"], "relation": item["relation"],
                     "gold": int(item["gold"]), "s": s.tolist(), "correct": correct, "meta": item["meta"]})
    return rows


# ---------------------------------------------------------------- summaries


def net(values: Sequence[float], tau: float) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.mean((v > tau).astype(float) - (v < -tau).astype(float))) if v.size else float("nan")


def share(values: Sequence[float], tau: float) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.mean(v > tau)) if v.size else float("nan")


def term_units(edges: Sequence[dict[str, Any]], *, decoded: bool | None = True, tau: float = TAU) -> dict[str, dict[str, float]]:
    """Per term (over its edges with `decoded`; None: every edge): comprehensiveness net share and share, sufficiency net
    share, specificity share, mean gap (nats), edge count."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in edges:
        if decoded is None or e["decoded"] == decoded:
            by[e["concept"]].append(e)
    out = {}
    for concept, rows in by.items():
        gaps = [r["gap"] for r in rows]
        suff = [r["sufficiency"] for r in rows]
        specific = [float(r["delta"] < -tau and r["off_max"] < abs(r["delta"])) for r in rows]
        out[concept] = {"comprehensiveness": net(gaps, tau), "moved": share(gaps, tau), "reversed": share([-g for g in gaps], tau),
                        "sufficiency": net(suff, tau), "specificity": float(np.mean(specific)), "gap": float(np.mean(gaps)),
                        "delta": float(np.mean([r["delta"] for r in rows])), "edges": len(rows),
                        **{f"comprehensiveness_tau{t:g}": net(gaps, t) for t in TAUS}}
    return out


def pair_units(rows: Sequence[dict[str, Any]], *, tau: float = TAU) -> dict[str, dict[str, float]]:
    """Per twin pair: role-specificity net share over its four removals (and mean RS, nats)."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[str(r["pair"])].append(r["rs"])
    return {p: {"role_specificity": net(v, tau), "rs": float(np.mean(v)), "removals": len(v)} for p, v in by.items() if len(v) == 4}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "new_words" in result:
        edges = result["new_words"]["edges"]
        units = term_units(edges, decoded=True)
        out["new_words"] = {"terms": units, "undecoded": term_units(edges, decoded=False), "all_edges": term_units(edges, decoded=None),
                            "mean": {k: float(np.mean([u[k] for u in units.values()])) for k in next(iter(units.values()), {})},
                            "edges": len(edges), "decoded_edges": int(sum(e["decoded"] for e in edges)),
                            "decode": result["new_words"]["decode"], "linked": result["new_words"]["linked"]}
    if "twins" in result:
        units = pair_units(result["twins"]["rows"])
        swap = result["twins"]["swap"]
        out["twins"] = {"pairs": units, "mean": {"role_specificity": float(np.mean([u["role_specificity"] for u in units.values()])) if units else None,
                                                  "rs": float(np.mean([u["rs"] for u in units.values()])) if units else None,
                                                  "own": float(np.mean([v["own"] for v in swap.values()])) if swap else None,
                                                  "swap": float(np.mean([v["swap"] for v in swap.values()])) if swap else None},
                        "swap": swap}
    return out


def render(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    label = f" — {header['label']}" if header.get("label") else ""
    lines = [f"# E12 faithfulness of decoding (F1) — {source['condition']} seed {source['seed']} ({source['size']}){label}", "",
             f"Store: `{header['store']['run']}` ({header['store']['family']}, unbinding `{header['store']['method']}`). Channel route, no "
             f"recall in context. τ = {TAU} nats.", ""]
    nw = summary.get("new_words")
    if nw:
        m = nw["mean"]
        lines += ["## New words (property items)", "",
                  f"{nw['linked']} terms; {nw['decoded_edges']} of {nw['edges']} edges with an item are decoded (decode accuracy over every "
                  f"edge: {nw['decode']:.3f}).", "",
                  "| measure (decoded edges; mean over terms) | value |", "|---|---:|",
                  f"| comprehensiveness net share (τ = {TAU}) | {m.get('comprehensiveness', float('nan')):+.3f} |",
                  f"| moved beyond the matched control (share) | {m.get('moved', float('nan')):.3f} |",
                  f"| moved the other way (share) | {m.get('reversed', float('nan')):.3f} |",
                  f"| mean gap (nats) | {m.get('gap', float('nan')):+.4f} |",
                  f"| mean Δ removing the edge (nats) | {m.get('delta', float('nan')):+.4f} |",
                  f"| sufficiency net share | {m.get('sufficiency', float('nan')):+.3f} |",
                  f"| specificity (share of edges) | {m.get('specificity', float('nan')):.3f} |", ""]
        lines += ["Sensitivity (comprehensiveness net share by τ): " + ", ".join(f"τ = {t:g}: {m.get(f'comprehensiveness_tau{t:g}', float('nan')):+.3f}"
                                                                           for t in TAUS), ""]
    tw = summary.get("twins")
    if tw and tw["mean"]["rs"] is not None:
        m = tw["mean"]
        lines += ["## Twins (role specificity)", "",
                  f"{len(tw['pairs'])} pairs: role-specificity net share {m['role_specificity']:+.3f}, mean RS {m['rs']:+.4f} nats "
                  f"(> 0: the removed filler drops more under its own role); twin contrast own {m['own']:.3f}, swapped frames {m['swap']:.3f}.", ""]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    run_dir = Path(args.run)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    track, family = config.get("e9_track"), config.get("e9_family") or "smollm2"
    new_items = args.new_items or (ROOT / "items" / NEW_ITEMS[(track, family)] if (track, family) in NEW_ITEMS else None)
    twins = args.twins or (ROOT / "items" / TWINS[(track, family)] if (track, family) in TWINS else None)
    output = Path(args.output or run_dir / (OUTPUT + (f"-{args.tag}" if args.tag else "")))
    alias_table = args.alias_table or ensure_alias_table(track_spec(track, family))
    record = {"experiment": "e12-faithfulness", "run": str(run_dir), "new_items": str(new_items) if new_items else None,
              "twins": str(twins) if twins else None, "limit": args.limit, "offset": args.offset, "twin_limit": args.twin_limit, "tau": TAU,
              "batch_size": args.batch_size, "alias_table": str(alias_table) if alias_table else None, "label": args.label}
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, record)
    loaded = load_store(run_dir, "own")
    run = open_run(run_dir, device=args.device, batch_size=args.batch_size, alias_table=alias_table)
    if run.composer is None:
        raise ValueError(f"{run_dir}: F1 needs a composing channel")
    result: dict[str, Any] = {}
    if new_items is not None and not args.skip_new:
        result["new_words"] = faithfulness_new_words(run, loaded.store, loaded.ontology, Path(new_items), limit=args.limit or None,
                                                     offset=args.offset)
    if twins is not None and not args.skip_twins:
        result["twins"] = role_specificity(run, loaded.store, loaded.ontology, Path(twins), limit=args.twin_limit)
    summary = summarize(result)
    header = {"source": run.describe(), "store": loaded.describe(), "label": args.label}
    edges = result.get("new_words", {}).get("edges", [])
    und.write_jsonl_gz(output / "edges.jsonl.gz", [{"set": "new_words", **e} for e in edges] +
                       [{"set": "twins", **r} for r in result.get("twins", {}).get("rows", [])])
    seconds = {k: round(v.get("seconds", 0.0), 1) for k, v in result.items()}
    write_json(output / "summary.json", {**header, "summary": summary, "seconds": seconds})
    (output / "report.md").write_text(render(summary, header))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    finish_output(output, record, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {"output": str(output), "summary": summary, "seconds": seconds, "peak_gb": peak}


def evaluate_command(run_dir: Path, *, python: str = sys.executable, batch_size: int = 32) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e12_faithfulness", "evaluate", "--run", str(run_dir), "--overwrite", "--batch-size", str(batch_size)]


FAITH_BATCH = {"SmolLM2-135M": 64, "SmolLM2-360M": 32, "Qwen3-0.6B-Base": 16, "Qwen3-1.7B-Base": 8}


def queue_stage(stage: str, *, priority: int = 50, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                hosts: Sequence[str] | None = None, root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None,
                dry_run: bool = False) -> list[dict[str, Any]]:
    """One GPU-lane F1 job per composing run of `stage` (`<stage>-<stem>-self-query-faithfulness`, idempotent)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import _config_host, stage_python
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    host_names = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(host_names) if host_names else pinned_python())
    jobs = []
    for path in configs:
        match = RUN_NAME.match(path.stem)
        config = yaml.safe_load(path.read_text())
        if match is None or (config.get("channel") or {}).get("mode") != "compose":
            continue
        model, seed, host = match["model"], int(match["seed"]), match["host"]
        if (models and model not in models) or (seeds and seed not in seeds) or (hosts and host not in hosts):
            continue
        run_dir = Path(root) / "runs" / stage / path.stem
        jobs.append({"name": f"{stage}-{path.stem}-{OUTPUT}", "priority": int(priority), "model": model,
                     "command": evaluate_command(run_dir, python=python, batch_size=FAITH_BATCH.get(host, 16))})
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
    ev = sub.add_parser("evaluate", help="F1 on one composing run (GPU)")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--new-items", type=Path, default=None)
    ev.add_argument("--twins", type=Path, default=None); ev.add_argument("--output", type=Path, default=None)
    ev.add_argument("--tag", default=None); ev.add_argument("--alias-table", type=Path, default=None)
    ev.add_argument("--batch-size", type=int, default=32); ev.add_argument("--device", default=None)
    ev.add_argument("--limit", type=int, default=300, help="the first N new words (pre-registered: 300, the v1 base set; 0 = all)")
    ev.add_argument("--offset", type=int, default=0, help="pilots only: skip the first N new words (a term range split over jobs)")
    ev.add_argument("--twin-limit", type=int, default=None, help="pilots and smoke tests only: the first N twin pairs")
    ev.add_argument("--skip-new", action="store_true"); ev.add_argument("--skip-twins", action="store_true")
    ev.add_argument("--label", default=None); ev.add_argument("--overwrite", action="store_true")
    qu = sub.add_parser("queue", help="queue one F1 job per composing run of a stage")
    qu.add_argument("--stage", required=True); qu.add_argument("--priority", type=int, default=50)
    qu.add_argument("--models", nargs="*", default=None); qu.add_argument("--seeds", type=int, nargs="*", default=None)
    qu.add_argument("--hosts", nargs="*", default=None); qu.add_argument("--root", type=Path, default=ROOT)
    qu.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = run_evaluate(args)
        print(json.dumps({"output": result["output"], "seconds": result["seconds"], "peak_gb": result["peak_gb"]}, indent=2))
    else:
        jobs = queue_stage(args.stage, priority=args.priority, models=args.models, seeds=args.seeds, hosts=args.hosts, root=args.root,
                           dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
