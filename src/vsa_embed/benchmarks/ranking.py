"""Log-probability ranking benchmarks on a trained E9 run (decision 63, WP TK-B1): the shared harness of the toolkit's
*read*, *write* and *meta* sets (`manuscript/toolkit-methodology-2026-10.md` §§2–3).

An item (`README.md` in this package; schema `rank-items/1`) is a context, answer options and the terms the item is
about. Every option is scored by `log p(option | prefix)` summed over the option's tokens, and also length-normalized
(per UTF-8 byte, as `acc_norm` of lm-evaluation-harness) and per token; the prediction is the argmax and a tie that
includes the answer earns 1 / (number of tied options), so an uninformed condition sits exactly at chance.

**Conditions** (`--conditions`, `+`-joined parts; the prefix of an option is `[reading material] joiner context`):

| Condition | What the model gets |
|---|---|
| `none` | The run as trained; new terms get no row (their spans are dropped). Host only for runs without a channel |
| `channel-off` | No span is injected at all (known terms unlinked too) |
| `store:<reader>` | New terms' rows composed by the channel from the reader's frame (`SpanChannel.add_entries`, rows of every other entry unchanged); known terms linked by the run's linker. A reader that writes no frame leaves the term without a row |
| `frame-in-context:<reader>` | The reader's frame verbalized and prepended (IKE-style); rows as `none`; the token cost is recorded |
| `definition-in-context` | The terms' definitions prepended; rows as `none`; token cost recorded |
| `store:<r>+definition-in-context` … | Parts combine |

**Readers** (term → frame): `oracle` (the item's gold frame), `oracle:<name>` (an alternative gold frame of the term,
`alt_frames[name]`), `random` (equal-degree random frame: the same relations, fillers drawn frequency-weighted from
each relation's fillers, the E9 rule), `none`, and the E11 readers of the term's definition (`read_to_learn`):
`typeprior`, `pattern`, `stated`, `linker`, `linker-all`, `linker-joint` (the linker readers score candidate frames by
the definition's likelihood under the run itself).

**Linking.** New terms (`concept` null, or `insert: true`) enter the run's alias table under their surface (an
existing alias with the same surface is shadowed and counted), linked to a placeholder entry; per scored text the
placeholder spans are remapped to the entry that holds the condition's frame for that term (or dropped). Known terms are
linked by the run's own linker, as in training. Every scored text records whether each bound term actually linked.

**Efficiency and determinism.** Identical requests (text, scored span, bindings) are scored once, texts are sorted by
length and batched under a token budget, output logits are computed only at the scored positions, and long prefixes are
truncated on the left. The same inputs give the same numbers on the CPU; identical requests give bitwise-equal scores
on any device, so ties are exact.

    python -m vsa_embed.benchmarks.ranking evaluate --run RUN --items ITEMS.jsonl[.gz] --output OUT
        [--conditions none,store:oracle,...] [--limit N] [--device cpu|cuda] [--threads 4] [--batch-size 32]
        [--token-budget 16384] [--max-length 1024] [--smoke]
    python -m vsa_embed.benchmarks.ranking report --inputs OUT ... --output DIR [--contrasts SPEC.json]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import json
import random
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .. import read_to_learn as rtl
from ..span_channel import AliasTable, CausalLinker, normalize_alias
from ..statistics import holm_adjust

SCHEMA = "rank-items/1"
OUTPUT_SCHEMA = "rank-eval/1"
NORMS = ("sum", "per_byte", "per_token")             # summed log-prob; length-normalized (bytes); per-token mean
PRIMARY_NORM = "sum"
MODEL_READERS = frozenset({"linker", "linker-all", "linker-joint"})
DEFINITION_READERS = frozenset({"typeprior", "pattern", "stated"}) | MODEL_READERS
BASE_READERS = ("oracle", "random", "none", "typeprior", "pattern", "stated", "linker", "linker-all", "linker-joint")
DEFAULT_CONDITIONS = ("none", "channel-off", "store:oracle", "store:linker", "store:typeprior", "store:random",
                      "frame-in-context:oracle", "definition-in-context")
QUIET: Callable[[str], None] = lambda message: None   # noqa: E731
# WordNet-style relation phrases for verbalizing a frame when the track lexicon has no statement templates.
VERBAL = {"hypernym": "{x} is a kind of {y}.", "instance_hypernym": "{x} is an instance of {y}.",
          "part_meronym": "{x} has a part called {y}.", "member_meronym": "{x} has a member called {y}.",
          "substance_meronym": "{x} is made of {y}.", "part_holonym": "{x} is part of {y}.",
          "member_holonym": "{x} is a member of {y}.", "substance_holonym": "{x} is a substance in {y}.",
          "attribute": "{x} has the attribute {y}.", "similar_to": "{x} is similar to {y}.",
          "topic_domain": "{x} belongs to the field of {y}.", "entailment": "To {x} entails to {y}.",
          "cause": "To {x} causes to {y}.", "antonym": "{x} is the opposite of {y}.", "lexname": "{x} is a kind of {y}."}
SKIP_VERBAL = frozenset({"pos"})


# -- items ------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Term:
    surface: str
    concept: str | None = None
    frame: tuple[tuple[str, str], ...] | None = None        # (relation name, atomic name) edges
    definition: str | None = None
    insert: bool = True                                     # write a row for it (default: concept is null)
    alt_frames: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Term":
        def frame(value: Any) -> tuple[tuple[str, str], ...] | None:
            return None if value is None else tuple((str(r), str(a)) for r, a in value)
        concept = row.get("concept")
        insert = bool(row["insert"]) if row.get("insert") is not None else concept is None
        alts = tuple(sorted((str(k), frame(v) or ()) for k, v in (row.get("alt_frames") or {}).items()))
        if not str(row.get("surface") or "").strip():
            raise ValueError(f"a term needs a surface: {row}")
        return cls(str(row["surface"]), None if concept is None else str(concept), frame(row.get("frame")),
                   row.get("definition"), insert, alts)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"surface": self.surface, "concept": self.concept,
                               "frame": None if self.frame is None else [list(e) for e in self.frame],
                               "definition": self.definition}
        if self.insert != (self.concept is None):
            out["insert"] = self.insert
        if self.alt_frames:
            out["alt_frames"] = {k: [list(e) for e in v] for k, v in self.alt_frames}
        return out

    @property
    def key(self) -> str:
        """Identity of a term instance: the same surface with the same frame(s) and definition is one term."""
        return json.dumps([normalize_alias(self.surface), self.concept, self.frame, self.definition, self.alt_frames])

    def alt(self, name: str) -> tuple[tuple[str, str], ...] | None:
        return dict(self.alt_frames).get(name)


@dataclasses.dataclass
class Item:
    id: str
    set: str
    context: str
    options: list[str]
    answer: int
    terms: list[Term]
    option_terms: list[list[Term]] | None = None
    joiner: str = "\n"
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    kind: str = "rank"

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Item":
        kind = row.get("kind", "rank")
        if kind != "rank":
            raise ValueError(f"item {row.get('id')}: kind {kind!r} is not supported (only 'rank')")
        options = [str(o) for o in row["options"]]
        answer = int(row["answer"])
        if len(options) < 2 or not 0 <= answer < len(options):
            raise ValueError(f"item {row.get('id')}: needs ≥ 2 options and an answer index among them")
        option_terms = row.get("option_terms")
        if option_terms is not None and len(option_terms) != len(options):
            raise ValueError(f"item {row.get('id')}: option_terms must have one list per option")
        return cls(str(row["id"]), str(row.get("set", "")), str(row.get("context", "")), options, answer,
                   [Term.from_json(t) for t in row.get("terms") or []],
                   None if option_terms is None else [[Term.from_json(t) for t in ts] for ts in option_terms],
                   str(row.get("joiner", "\n")), dict(row.get("meta") or {}), kind)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "set": self.set, "kind": self.kind, "context": self.context,
                               "options": self.options, "answer": self.answer, "terms": [t.to_json() for t in self.terms]}
        if self.option_terms is not None:
            out["option_terms"] = [[t.to_json() for t in ts] for ts in self.option_terms]
        if self.joiner != "\n":
            out["joiner"] = self.joiner
        out["meta"] = self.meta
        return out

    def terms_of(self, option: int) -> list[Term]:
        """Terms that hold while option `option` is scored: the shared terms plus the option's own."""
        return list(self.terms) + (list(self.option_terms[option]) if self.option_terms is not None else [])

    def all_terms(self) -> list[Term]:
        return list(self.terms) + [t for ts in (self.option_terms or []) for t in ts]


def _open(path: Path, mode: str = "rt") -> Any:
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def load_items(path: Path, *, limit: int | None = None, spread: bool = False) -> list[Item]:
    """Items of a `rank-items/1` jsonl (optionally .gz); ids must be unique. `limit` keeps the first items, or with
    `spread` that many items evenly spaced over the file (smoke tests of sets ordered by kind)."""
    items, seen = [], set()
    with _open(Path(path)) as handle:
        for line in handle:
            if not line.strip():
                continue
            item = Item.from_json(json.loads(line))
            if item.id in seen:
                raise ValueError(f"{path}: duplicate item id {item.id}")
            seen.add(item.id)
            items.append(item)
            if limit and not spread and len(items) >= limit:
                break
    if limit and spread and len(items) > limit:
        items = [items[int(i)] for i in np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))]
    return items


def write_items(path: Path, items: Iterable[Item]) -> dict[str, Any]:
    """Write items (gzip if the name ends in .gz, with a fixed mtime so equal content gives equal bytes); returns counts
    and the sha256 of the uncompressed jsonl."""
    import hashlib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(i.to_json(), ensure_ascii=False, sort_keys=False) + "\n" for i in items]
    payload = "".join(lines).encode()
    if str(path).endswith(".gz"):
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle:
            handle.write(payload)
    else:
        path.write_bytes(payload)
    return {"items": len(lines), "jsonl_sha256": hashlib.sha256(payload).hexdigest(), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# -- conditions -------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Condition:
    name: str
    rows: str | None = None                 # reader whose frames compose new terms' rows (None: no row)
    context: tuple[str, ...] = ()           # "definition" and/or "frame:<reader>", in this order of mention
    channel: bool = True

    @property
    def readers(self) -> list[str]:
        out = [self.rows] if self.rows else []
        return out + [c.split(":", 1)[1] for c in self.context if c.startswith("frame:")]


def parse_condition(name: str) -> Condition:
    rows, context, channel = None, [], True
    for part in name.split("+"):
        part = part.strip()
        if part == "none":
            continue
        if part == "channel-off":
            channel = False
        elif part.startswith("store:"):
            rows = part.split(":", 1)[1]
        elif part == "definition-in-context":
            context.append("definition")
        elif part.startswith("frame-in-context:"):
            context.append("frame:" + part.split(":", 1)[1])
        else:
            raise ValueError(f"unknown condition part {part!r} in {name!r}")
    for reader in [rows] + [c.split(":", 1)[1] for c in context if c.startswith("frame:")]:
        if reader is not None and reader.split(":", 1)[0] not in BASE_READERS:
            raise ValueError(f"unknown reader {reader!r} in {name!r}")
    if rows is not None and not channel:
        raise ValueError(f"{name!r}: store rows need the channel")
    return Condition(name, None if rows in (None, "none") else rows, tuple(context), channel)


# -- frames: verbalization, random frames ----------------------------------------------------------------------------------

def verbalize(surface: str, frame: Sequence[tuple[int, int]], ontology: dict[str, Any], lexicon: Any) -> str:
    """A frame as sentences about `surface`: the track's statement templates (`self_query.RecallWriter`) when the lexicon
    has them, else WordNet-style phrases (`VERBAL`); edges whose filler has no readable text (part of speech) are
    skipped."""
    from ..self_query import RecallWriter
    relation_names, atomic_names = ontology["relation_names"], ontology["atomic_names"]
    templated = bool(getattr(lexicon, "templates", None))
    writer = RecallWriter(relation_names, atomic_names, lexicon, confidence=False) if templated else None
    sentences = []
    for r, a in frame:
        name = relation_names[r]
        if name in SKIP_VERBAL:
            continue
        text = lexicon.text(atomic_names[a]) if lexicon is not None else None
        if not text:
            continue
        if writer is not None:
            sentence = writer.statement(surface, r, a)
        else:
            sentence = VERBAL.get(name, "{x} " + name.replace("_", " ") + " {y}.").replace("{x}", surface).replace("{y}", text)
        sentences.append(sentence)
    return " ".join(dict.fromkeys(sentences))


def filler_pools(ontology: dict[str, Any]) -> dict[int, Counter]:
    pools: dict[int, Counter] = defaultdict(Counter)
    for r, f in zip(np.asarray(ontology["relations"]).tolist(), np.asarray(ontology["fillers"]).tolist()):
        pools[int(r)][int(f)] += 1
    return pools


def random_frame(frame: Sequence[tuple[int, int]], pools: dict[int, Counter], key: str, *, seed: int = 0) -> list[tuple[int, int]]:
    """Equal degree, the same relations, every filler drawn frequency-weighted from its relation's fillers (the gold filler
    excluded when the pool has another); seeded by the term instance, so it is the same in every run and condition."""
    rng = random.Random(zlib.crc32(f"{seed}|{key}".encode()))
    out = []
    for r, a in frame:
        pool = pools.get(int(r)) or Counter({int(a): 1})
        candidates = sorted((f, c) for f, c in pool.items() if f != a) or [(int(a), 1)]
        population, weights = zip(*candidates)
        out.append((int(r), int(rng.choices(population, weights)[0])))
    return out


# -- the scorer ---------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Request:
    text: str
    start: int                                   # score tokens whose character span ends after this offset
    bindings: tuple[tuple[int, int], ...] = ()   # (placeholder entry, variant entry or -1 = no row), sorted
    channel: bool = True


def extended_table(table: AliasTable, surfaces: Sequence[str], *, concept_base: int) -> tuple[AliasTable, dict[str, int], list[str]]:
    """The run's alias table plus one placeholder entry per new surface (ids after the existing entries). A surface that
    already is an alias is shadowed (it then links to the new term). Returns (table, surface → placeholder, shadowed)."""
    start = len(table.entry_concepts)
    keys = sorted({normalize_alias(s) for s in surfaces if normalize_alias(s)})
    placeholder = {k: start + i for i, k in enumerate(keys)}
    shadowed = [k for k in keys if k in table.alias_to_entry]
    entry_concepts = list(table.entry_concepts) + [(concept_base + i,) for i in range(len(keys))]
    return AliasTable({**table.alias_to_entry, **placeholder}, entry_concepts, table.holdout, table.normalization), placeholder, shadowed


class RankScorer:
    """Scores `Request`s on an opened run (`e5_common.open_run`): Σ log p of the scored tokens, their count and the byte
    length of the scored text; placeholder spans are remapped per request."""

    def __init__(self, run: Any, new_surfaces: Sequence[str], *, batch_size: int = 32, token_budget: int = 16384,
                 max_length: int = 1024) -> None:
        self.run, self.batch_size, self.token_budget, self.max_length = run, int(batch_size), int(token_budget), int(max_length)
        self.model, self.device, self.tokenizer = run.model, run.device, run.tokenizer
        adapter = run.adapter
        self.channel = getattr(self.model, "channel", None)
        linker = getattr(adapter, "linker", None)
        self.placeholder: dict[str, int] = {}
        self.shadowed: list[str] = []
        self.linker = None
        if self.channel is not None and linker is not None:
            concept_base = len((run.ontology or {}).get("concept_names") or ()) or (
                max((max(c) for c in linker.table.entry_concepts if c), default=-1) + 1)
            table, self.placeholder, self.shadowed = extended_table(linker.table, new_surfaces, concept_base=concept_base)
            self.linker = CausalLinker(table, boundary=linker.boundary, min_subtokens=linker.min_subtokens)
        self.entry_count = int(self.channel.entry_count) if self.channel is not None else 0
        self.placeholder_ids = frozenset(self.placeholder.values())
        self.stats = Counter()

    def placeholder_of(self, surface: str) -> int | None:
        return self.placeholder.get(normalize_alias(surface))

    @property
    def variant_base(self) -> int:
        """First entry id of inserted variants: after the trained entries and the placeholders (`inserted`)."""
        return self.entry_count + len(self.placeholder)

    @contextlib.contextmanager
    def inserted(self, frames: Sequence[Sequence[tuple[int, int]]]) -> Iterator[None]:
        """Within the block the channel holds the placeholders (never injected: they are always remapped) and then one
        entry per frame, at `variant_base + j` (`e9_ontology_edit.inserted_entries`; restored exactly on exit)."""
        from ..experiments.e9_ontology_edit import inserted_entries
        if self.channel is None or not frames:
            yield
            return
        dummy = [list(frames[0])] * len(self.placeholder)
        with inserted_entries(self.channel, len(dummy) + len(frames), dummy + [list(f) for f in frames]):
            yield

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _spans(self, texts: Sequence[str], offsets: Sequence[Sequence[tuple[int, int]]], requests: Sequence[Request]
               ) -> tuple[dict[str, torch.Tensor] | None, list[tuple[int, int]]]:
        """Linked spans with placeholders remapped per request; per request (bound terms, bound terms that linked)."""
        from ..span_channel import link_batch
        found = [(len(r.bindings), 0) for r in requests]
        if self.channel is None or self.linker is None:
            return None, found
        spans = link_batch(self.linker, texts, offsets)
        spans["confidence"] = spans["confidence"].half().float()     # as `ChannelModelAdapter.link_spans`
        if not spans["entry"].numel():
            return (spans if any(r.channel for r in requests) else None), found
        keep = torch.ones(spans["entry"].numel(), dtype=torch.bool)
        entry = spans["entry"].clone()
        linked: dict[int, set[int]] = defaultdict(set)
        for k, (b, e) in enumerate(zip(spans["batch"].tolist(), spans["entry"].tolist())):
            request = requests[b]
            if not request.channel:
                keep[k] = False
                continue
            if e in self.placeholder_ids:
                target = dict(request.bindings).get(e, -1)
                if e in dict(request.bindings):
                    linked[b].add(e)
                if target < 0:
                    keep[k] = False
                else:
                    entry[k] = int(target)
        found = [(len(r.bindings), len(linked.get(b, ()))) for b, r in enumerate(requests)]
        return {key: (entry if key == "entry" else value)[keep] for key, value in spans.items()}, found

    @torch.no_grad()
    def score(self, requests: Sequence[Request], *, log: Callable[[str], None] = QUIET) -> dict[str, np.ndarray]:
        """Per request: `sum` (Σ log p), `tokens` (scored tokens), `bytes` (UTF-8 bytes of the scored text), `linked`
        (share of bound terms that linked; 1 when none is bound), `truncated` (prefix cut on the left)."""
        unique: dict[Request, int] = {}
        for r in requests:
            unique.setdefault(r, len(unique))
        order_requests = list(unique)
        n = len(order_requests)
        sums, counts, linked, truncated = np.zeros(n), np.zeros(n), np.ones(n), np.zeros(n, dtype=bool)
        tok = self.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        started = time.monotonic()
        encoded = tok([r.text for r in order_requests], return_offsets_mapping=True, add_special_tokens=False) if n else {
            "input_ids": [], "offset_mapping": []}
        ids_all, offsets_all = [], []
        for i, (ids, offs) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
            if len(ids) > self.max_length:
                ids, offs = ids[-self.max_length:], offs[-self.max_length:]
                truncated[i] = True
            ids_all.append(list(ids)); offsets_all.append([tuple(o) for o in offs])
        order = sorted(range(n), key=lambda i: (len(ids_all[i]), i))
        head = self.model.model.get_output_embeddings() if hasattr(self.model, "embed") else self.model.get_output_embeddings()
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")
        cursor = 0
        self.stats["tokenize_seconds"] += time.monotonic() - started
        while cursor < n:
            chunk = [order[cursor]]
            while (cursor + len(chunk) < n and len(chunk) < self.batch_size
                   and (len(chunk) + 1) * len(ids_all[order[cursor + len(chunk)]]) <= self.token_budget):
                chunk.append(order[cursor + len(chunk)])
            cursor += len(chunk)
            width = max(1, max(len(ids_all[i]) for i in chunk))
            input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, i in enumerate(chunk):
                input_ids[row, :len(ids_all[i])] = torch.tensor(ids_all[i], dtype=torch.long)
                mask[row, :len(ids_all[i])] = 1
            batch_requests = [order_requests[i] for i in chunk]
            t0 = time.monotonic()
            spans, found = self._spans([r.text for r in batch_requests], [offsets_all[i] for i in chunk], batch_requests)
            for row, i in enumerate(chunk):
                bound, hit = found[row]
                linked[i] = hit / bound if bound else 1.0
            t1 = time.monotonic()
            with autocast:
                if hasattr(self.model, "embed"):
                    hidden = self.model.hidden_states(input_ids.to(self.device), mask.to(self.device),
                                                      None if spans is None else {k: v.to(self.device) for k, v in spans.items()})
                else:
                    hidden = self.model.base_model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
            rows, cols, targets, owners = [], [], [], []
            for row, i in enumerate(chunk):
                start = order_requests[i].start
                offs = offsets_all[i]
                for t in range(1, len(offs)):
                    if offs[t][1] > start:
                        rows.append(row); cols.append(t - 1); targets.append(ids_all[i][t]); owners.append(i)
            if rows:
                picked_hidden = hidden[torch.tensor(rows, device=hidden.device), torch.tensor(cols, device=hidden.device)].float()
                logits = F.linear(picked_hidden, head.weight.float(),
                                  None if getattr(head, "bias", None) is None else head.bias.float())
                picked = torch.log_softmax(logits, -1).gather(-1, torch.tensor(targets, device=logits.device)[:, None]).squeeze(-1)
                np.add.at(sums, owners, picked.cpu().double().numpy())
                np.add.at(counts, owners, 1.0)
            self.stats["link_seconds"] += t1 - t0
            self.stats["model_seconds"] += time.monotonic() - t1
            self.stats["batches"] += 1
            self.stats["forward_tokens"] += int(mask.sum())
            self.stats["padded_tokens"] += int(mask.numel())
        self.stats["requests"] += len(requests)
        self.stats["unique_requests"] += n
        self.stats["truncated"] += int(truncated.sum())
        self.stats["seconds"] += time.monotonic() - started
        position = [unique[r] for r in requests]
        nbytes = np.asarray([len(r.text[r.start:].encode()) for r in order_requests], dtype=np.float64)
        return {"sum": sums[position], "tokens": counts[position], "bytes": nbytes[position] if n else np.zeros(0),
                "linked": linked[position], "truncated": truncated[position]}

    def definition_scorer(self, surfaces: dict[str, str]) -> rtl.Scorer:
        """`read_to_learn.Scorer` for the linker readers: Σ log p of a definition's tokens after the headword, with the
        headword's placeholder bound to an entry holding the candidate frame (None: no row). `surfaces[task.concept]` is
        the term's surface."""
        def scorer(tasks: Sequence[rtl.ReadTask], variants: Sequence[Sequence[rtl.Frame | None]]) -> list[np.ndarray]:
            distinct: dict[tuple, int] = {}
            for vs in variants:
                for frame in vs:
                    if frame:
                        distinct.setdefault(tuple(sorted(map(tuple, frame))), len(distinct))
            frames = [list(k) for k in distinct]
            requests = []
            for task, vs in zip(tasks, variants):
                holder = self.placeholder_of(surfaces[task.concept])
                end = task.span[1] if task.span else 0
                for frame in vs:
                    target = self.variant_base + distinct[tuple(sorted(map(tuple, frame)))] if frame else -1
                    requests.append(Request(task.text, end, ((holder, target),) if holder is not None else ()))
            with self.inserted(frames):
                values = self.score(requests)["sum"]
            out, cursor = [], 0
            for vs in variants:
                out.append(np.asarray(values[cursor:cursor + len(vs)], dtype=float))
                cursor += len(vs)
            return out
        scorer.token_count = self.token_count            # type: ignore[attr-defined]
        return scorer


# -- the evaluation -----------------------------------------------------------------------------------------------------------

def scalar_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """The item's scalar meta fields (kept with every per-item result, so reports can filter on them)."""
    return {k: v for k, v in meta.items() if v is None or isinstance(v, (str, int, float, bool))}


def _frame_ids(frame: Sequence[tuple[str, str]] | None, relation_id: dict[str, int], atom_id: dict[str, int]
               ) -> list[tuple[int, int]] | None:
    """A named frame → ids of the run's ontology; None if absent or any name is unknown to the run."""
    if frame is None:
        return None
    try:
        return [(relation_id[r], atom_id[a]) for r, a in frame]
    except KeyError:
        return None


class Evaluation:
    """One run × one item set: frames per reader, then every condition's requests, scores and per-item results."""

    def __init__(self, run: Any, items: Sequence[Item], *, batch_size: int = 32, token_budget: int = 16384,
                 max_length: int = 1024, seed: int = 0, log: Callable[[str], None] = print) -> None:
        self.run, self.items, self.seed, self.log = run, list(items), seed, log
        new_surfaces = sorted({t.surface for i in self.items for t in i.all_terms() if t.insert})
        self.scorer = RankScorer(run, new_surfaces, batch_size=batch_size, token_budget=token_budget, max_length=max_length)
        ontology = run.ontology or {"relation_names": [], "atomic_names": [], "relations": [], "fillers": []}
        self.ontology = ontology
        self.relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
        self.atom_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
        self.terms: dict[str, Term] = {}
        for item in self.items:
            for t in item.all_terms():
                self.terms.setdefault(t.key, t)
        self._frames: dict[str, dict[str, list[tuple[int, int]] | None]] = {}
        self._reader_rows: dict[str, list[dict[str, Any]]] = {}
        self._ctx = None
        self._lexicon = None
        self._pools = None
        self.unresolved = sum(1 for t in self.terms.values() if t.frame is not None
                              and _frame_ids(t.frame, self.relation_id, self.atom_id) is None)

    # readers --------------------------------------------------------------------------------------------------
    @property
    def ctx(self) -> Any:
        if self._ctx is None:
            from ..experiments.e11_read_to_learn import run_context
            self._ctx = run_context(self.run)
        return self._ctx

    @property
    def lexicon(self) -> Any:
        if self._lexicon is None:
            from ..experiments.e9_dim3_baselines import lexicon_for_run
            self._lexicon, error = lexicon_for_run(self.run)
            if self._lexicon is None:
                raise ValueError(f"no lexicon for this run ({error}): frames cannot be verbalized")
        return self._lexicon

    def gold(self, term: Term, name: str = "oracle") -> list[tuple[int, int]] | None:
        frame = term.frame if name == "oracle" else term.alt(name.split(":", 1)[1])
        return _frame_ids(frame, self.relation_id, self.atom_id)

    def frames(self, reader: str) -> dict[str, list[tuple[int, int]] | None]:
        """Term key → frame (ids) written by `reader` (None: no frame)."""
        if reader in self._frames:
            return self._frames[reader]
        started = time.monotonic()
        out: dict[str, list[tuple[int, int]] | None] = {}
        rows: list[dict[str, Any]] = []
        if reader == "none":
            out = {k: None for k in self.terms}
        elif reader == "oracle" or reader.startswith("oracle:"):
            out = {k: self.gold(t, reader) for k, t in self.terms.items()}
        elif reader == "random":
            if self._pools is None:
                self._pools = filler_pools(self.ontology)
            for k, t in self.terms.items():
                gold = self.gold(t)
                out[k] = random_frame(gold, self._pools, k, seed=self.seed) if gold else None
        elif reader in DEFINITION_READERS:
            out = self._read_definitions(reader, rows)
        else:
            raise ValueError(f"unknown reader {reader!r}")
        self._frames[reader] = out
        if not rows:
            rows = [{"term": k, "reader": reader, "frame": self._names(f)} for k, f in out.items()]
        for row in rows:
            row.setdefault("seconds_total", time.monotonic() - started)
        self._reader_rows[reader] = rows
        return out

    def _names(self, frame: Sequence[tuple[int, int]] | None) -> list[list[str]] | None:
        if frame is None:
            return None
        return [[self.ontology["relation_names"][r], self.ontology["atomic_names"][a]] for r, a in frame]

    def _read_definitions(self, reader: str, rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]] | None]:
        keys = [k for k, t in self.terms.items() if t.definition and rtl.headword_span(t.definition, t.surface)]
        out: dict[str, list[tuple[int, int]] | None] = {k: None for k in self.terms}
        if not keys:
            return out
        ctx = self.ctx
        tasks = [rtl.ReadTask(k, "definition", self.terms[k].definition or "", self.terms[k].surface, self.gold(self.terms[k]) or [])
                 for k in keys]
        mentions = [rtl.mentions_of(t, ctx.fillers) for t in tasks]
        if reader == "typeprior":
            results = [rtl.read_typeprior(t, m, ctx.typing) for t, m in zip(tasks, mentions)]
        elif reader == "pattern":
            results = [rtl.read_pattern(t, m, ctx.typing, ctx.fillers) for t, m in zip(tasks, mentions)]
        elif reader == "stated":
            results = [rtl.read_stated(t, m) for t, m in zip(tasks, mentions)]
        else:
            if self.scorer.channel is None or self.scorer.channel.mode != "compose":
                self.log(f"  reader {reader}: the run does not compose rows; no frames")
                return out
            scorer = self.scorer.definition_scorer({k: self.terms[k].surface for k in keys})
            if reader == "linker-joint":
                results = rtl.read_linker_joint(tasks, mentions, ctx.typing, scorer, token_count=self.scorer.token_count)
            else:
                kept, every = rtl.read_linker(tasks, mentions, ctx.typing, scorer, token_count=self.scorer.token_count)
                results = kept if reader == "linker" else every
                self._frames.setdefault("linker-all" if reader == "linker" else "linker",
                                        {**{k: None for k in self.terms}, **{r.concept: r.frame for r in (every if reader == "linker" else kept)}})
        for task, found, result in zip(tasks, mentions, results):
            out[task.concept] = result.frame
            atoms = {a for m in found for a in m.atoms}
            rows.append({"term": task.concept, "reader": reader, "frame": self._names(result.frame),
                         "metrics": rtl.frame_metrics(result.frame, task.gold, found_atoms=atoms) if task.gold else None,
                         "cost": result.cost, "mentions": len(found)})
        return out

    # requests -------------------------------------------------------------------------------------------------
    def _prefix(self, item: Item, option: int, condition: Condition) -> tuple[str, str]:
        """(prefix, added reading material) of an option under a condition."""
        parts = []
        for part in condition.context:
            for term in item.terms_of(option):
                if part == "definition":
                    if term.definition:
                        parts.append(term.definition)
                else:
                    frame = self.frames(part.split(":", 1)[1]).get(term.key)
                    if frame:
                        text = verbalize(term.surface, frame, self.ontology, self.lexicon)
                        if text:
                            parts.append(text)
        added = item.joiner.join(dict.fromkeys(parts))
        return (added + item.joiner + item.context if added else item.context), added

    def evaluate(self, conditions: Sequence[Condition]) -> dict[str, Any]:
        per_item: dict[str, dict[str, Any]] = {i.id: {"id": i.id, "set": i.set, "answer": i.answer, "options": len(i.options),
                                                      "meta": scalar_meta(i.meta), "conditions": {}} for i in self.items}
        timing: dict[str, float] = {}
        for condition in conditions:
            started = time.monotonic()
            rows_frames = self.frames(condition.rows) if condition.rows else {}
            distinct: dict[tuple, int] = {}
            for frame in rows_frames.values():
                if frame:
                    distinct.setdefault(tuple(sorted(map(tuple, frame))), len(distinct))
            requests, owners, added_tokens, token_cache = [], [], {}, {}
            for item in self.items:
                for o, option in enumerate(item.options):
                    prefix, added = self._prefix(item, o, condition)
                    bindings = {}
                    for term in item.terms_of(o):
                        holder = self.scorer.placeholder_of(term.surface) if term.insert else None
                        if holder is None:
                            continue
                        frame = rows_frames.get(term.key) if condition.rows else None
                        bindings[holder] = self.scorer.variant_base + distinct[tuple(sorted(map(tuple, frame)))] if frame else -1
                    requests.append(Request(prefix + option, len(prefix), tuple(sorted(bindings.items())), condition.channel))
                    owners.append((item.id, o))
                    if added:
                        if added + item.joiner not in token_cache:
                            token_cache[added + item.joiner] = self.scorer.token_count(added + item.joiner)
                        added_tokens[(item.id, o)] = token_cache[added + item.joiner]
            with self.scorer.inserted([list(k) for k in distinct]) if distinct else contextlib.nullcontext():
                scores = self.scorer.score(requests)
            grouped: dict[str, list[int]] = defaultdict(list)
            for k, (item_id, _) in enumerate(owners):
                grouped[item_id].append(k)
            for item in self.items:
                ks = grouped[item.id]
                sums, tokens, nbytes = scores["sum"][ks], scores["tokens"][ks], scores["bytes"][ks]
                values = {"sum": sums, "per_byte": sums / np.maximum(nbytes, 1), "per_token": sums / np.maximum(tokens, 1)}
                record = {"sum": [round(float(s), 6) for s in sums], "tokens": [int(t) for t in tokens],
                          "bytes": [int(b) for b in nbytes], "correct": {}, "pred": {},
                          "linked": round(float(np.mean(scores["linked"][ks])), 4),
                          "context_tokens": float(np.mean([added_tokens.get((item.id, o), 0) for o in range(len(item.options))])),
                          "truncated": bool(np.any(scores["truncated"][ks]))}
                for norm, v in values.items():
                    best = np.flatnonzero(v == v.max())
                    record["correct"][norm] = (1.0 / len(best)) if item.answer in best else 0.0
                    record["pred"][norm] = int(best[0]) if len(best) == 1 else None
                per_item[item.id]["conditions"][condition.name] = record
            timing[condition.name] = time.monotonic() - started
            self.log(f"  {condition.name}: {len(requests)} options, {time.monotonic() - started:.1f}s, "
                     f"accuracy {np.mean([per_item[i.id]['conditions'][condition.name]['correct']['sum'] for i in self.items]):.4f}")
        return {"items": per_item, "timing": timing}

    def reader_summary(self) -> dict[str, Any]:
        """Per reader: terms read, frames written, micro precision / recall against the oracle frame, mean cost."""
        out = {}
        for reader, rows in self._reader_rows.items():
            metrics = [r["metrics"] for r in rows if r.get("metrics")]
            hits = sum(m["hits"] for m in metrics); pred = sum(m["edges"] for m in metrics); gold = sum(m["gold_edges"] for m in metrics)
            costs: dict[str, float] = defaultdict(float)
            for r in rows:
                for k, v in (r.get("cost") or {}).items():
                    costs[k] += float(v or 0)
            out[reader] = {"terms": len(rows), "frames": sum(1 for r in rows if r.get("frame")),
                           "precision": hits / pred if pred else None, "recall": hits / gold if gold else None,
                           "mean_edges": pred / max(1, len(metrics)) if metrics else None,
                           "cost_per_term": {k: v / max(1, len(rows)) for k, v in sorted(costs.items())}}
        return out


# -- statistics -----------------------------------------------------------------------------------------------------------------

def bootstrap_mean(values: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = values[np.random.default_rng(seed).integers(0, values.size, size=(resamples, values.size))].mean(1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "n": int(values.size)}


def paired(a: dict[str, float], b: dict[str, float], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """Mean of a − b over shared keys (items), percentile bootstrap CI and two-sided bootstrap p (`e5_zeroshot.paired_difference`)."""
    from ..experiments.e5_zeroshot import paired_difference
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    return paired_difference(np.asarray([a[k] for k in keys]), np.asarray([b[k] for k in keys]), resamples=resamples, seed=seed)


def correctness(per_item: dict[str, dict[str, Any]], condition: str, norm: str = PRIMARY_NORM,
                keep: Callable[[str], bool] | None = None) -> dict[str, float]:
    return {i: r["conditions"][condition]["correct"][norm] for i, r in per_item.items()
            if condition in r["conditions"] and (keep is None or keep(i))}


def summarize(per_item: dict[str, dict[str, Any]], conditions: Sequence[str], *, resamples: int = 2000, seed: int = 0,
              contrasts: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Accuracy per condition and normalization (bootstrap CI over items), per set; every condition − `none`, plus
    `contrasts`, paired over items."""
    sets = sorted({r["set"] for r in per_item.values()})
    out: dict[str, Any] = {"items": len(per_item), "sets": {}, "chance": float(np.mean([1.0 / r["options"] for r in per_item.values()]))
                           if per_item else None}
    for name in [None, *sets] if len(sets) > 1 else [None]:
        keep = (lambda i, s=name: per_item[i]["set"] == s) if name else None
        block: dict[str, Any] = {"accuracy": {}, "contrasts": [], "context_tokens": {}, "linked": {}}
        for cond in conditions:
            for norm in NORMS:
                values = np.asarray(list(correctness(per_item, cond, norm, keep).values()))
                block["accuracy"].setdefault(cond, {})[norm] = bootstrap_mean(values, resamples=resamples, seed=seed)
            rows = [r["conditions"][cond] for i, r in per_item.items() if cond in r["conditions"] and (keep is None or keep(i))]
            block["context_tokens"][cond] = float(np.mean([r["context_tokens"] for r in rows])) if rows else None
            block["linked"][cond] = float(np.mean([r["linked"] for r in rows])) if rows else None
        pairs = [(c, "none") for c in conditions if c != "none" and "none" in conditions] + [tuple(c) for c in contrasts]
        for a, b in dict.fromkeys(pairs):
            if a not in conditions or b not in conditions:
                continue
            result = paired(correctness(per_item, a, PRIMARY_NORM, keep), correctness(per_item, b, PRIMARY_NORM, keep),
                            resamples=resamples, seed=seed)
            if result:
                block["contrasts"].append({"a": a, "b": b, "norm": PRIMARY_NORM, **result})
        out["sets"][name or "all"] = block
    return out


# -- outputs --------------------------------------------------------------------------------------------------------------------

def render(header: dict[str, Any], summary: dict[str, Any]) -> str:
    from ..experiments.e5_common import fmt, fmt_ci
    lines = []
    if header.get("smoke"):
        lines += ["**SMOKE TEST — not a result.**", ""]
    source = header.get("source") or {}
    lines += [f"# Ranking benchmark: {header.get('items_name')} on {source.get('experiment') or source.get('run')}", "",
              f"- run: `{source.get('run')}` ({source.get('condition')}, seed {source.get('seed')}, channel {source.get('channel_mode')})",
              f"- items: `{header.get('items')}` ({summary['items']} items; chance {fmt(summary.get('chance'), 3)})",
              f"- conditions: {', '.join(header.get('conditions', []))}",
              f"- new surfaces: {header.get('new_surfaces')} (shadowed existing aliases: {header.get('shadowed')}); "
              f"frames unresolvable in this run: {header.get('unresolved_frames')}", ""]
    for name, block in summary["sets"].items():
        lines += [f"## {name}", "", "| Condition | Accuracy (sum) | per byte | per token | Context tokens | Linked |", "|---|---|---|---|---|---|"]
        for cond, by_norm in block["accuracy"].items():
            lines.append(f"| {cond} | {fmt_ci(by_norm['sum'], 4)} | {fmt(by_norm['per_byte']['mean'], 4)} | "
                         f"{fmt(by_norm['per_token']['mean'], 4)} | {fmt(block['context_tokens'].get(cond), 1)} | "
                         f"{fmt(block['linked'].get(cond), 3)} |")
        if block["contrasts"]:
            lines += ["", "| Contrast (accuracy, paired over items) | Δ [95% CI] | p | n |", "|---|---|---|---|"]
            for c in block["contrasts"]:
                lines.append(f"| {c['a']} − {c['b']} | {fmt_ci(c, 4)} | {fmt(c.get('p_value'), 4)} | {c['n']} |")
        lines.append("")
    readers = header.get("readers") or {}
    if readers:
        lines += ["## Readers (frames against the oracle frame)", "", "| Reader | Terms | Frames | Precision | Recall | Cost per term |",
                  "|---|---|---|---|---|---|"]
        for reader, r in readers.items():
            cost = ", ".join(f"{k} {v:.1f}" for k, v in r["cost_per_term"].items())
            lines.append(f"| {reader} | {r['terms']} | {r['frames']} | {fmt(r['precision'], 3)} | {fmt(r['recall'], 3)} | {cost or '—'} |")
        lines.append("")
    stats = header.get("scorer") or {}
    lines.append(f"Scorer: {stats.get('unique_requests', 0)} unique of {stats.get('requests', 0)} requests, "
                 f"{stats.get('forward_tokens', 0)} forward tokens, {stats.get('seconds', 0.0):.1f} s, truncated {stats.get('truncated', 0)}.")
    return "\n".join(lines) + "\n"


def available_conditions(run: Any, requested: Sequence[str]) -> tuple[list[Condition], list[str]]:
    """Conditions the run supports: without a channel, `store` is impossible and `channel-off` equals `none`."""
    channel = getattr(run.model, "channel", None)
    kept, skipped = [], []
    for name in requested:
        condition = parse_condition(name)
        if channel is None and (condition.rows or not condition.channel):
            skipped.append(name)
            continue
        if condition.rows in MODEL_READERS and channel is not None and channel.mode != "compose":
            skipped.append(name)
            continue
        kept.append(condition)
    return kept, skipped


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from ..experiments.e5_common import clear_output, finish_output, json_ready, open_run, start_output, write_json
    if args.threads:
        torch.set_num_threads(int(args.threads))
    items = load_items(args.items, limit=args.limit, spread=args.spread)
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.conditions else list(DEFAULT_CONDITIONS)
    config = {"experiment": "rank-benchmark", "run": str(args.run), "items": str(args.items), "limit": args.limit,
              "spread": bool(args.spread), "conditions": requested, "batch_size": args.batch_size, "token_budget": args.token_budget,
              "max_length": args.max_length, "seed": args.seed, "resamples": args.resamples, "smoke": bool(args.smoke),
              "device": args.device, "threads": args.threads}
    output = Path(args.output)
    if args.overwrite:
        clear_output(output)
        (output / "items.jsonl.gz").unlink(missing_ok=True)
        (output / "readers.jsonl.gz").unlink(missing_ok=True)
    git_at_start = start_output(output, config)
    run = open_run(args.run, device=args.device, batch_size=args.batch_size, max_length=args.max_length,
                   alias_table=args.alias_table)
    conditions, skipped = available_conditions(run, requested)
    started = time.monotonic()
    evaluation = Evaluation(run, items, batch_size=args.batch_size, token_budget=args.token_budget, max_length=args.max_length,
                            seed=args.seed)
    result = evaluation.evaluate(conditions)
    names = [c.name for c in conditions]
    contrasts = [tuple(c.split("|", 1)) for c in (args.contrasts or "").split(",") if "|" in c]
    summary = summarize(result["items"], names, resamples=args.resamples, seed=args.seed, contrasts=contrasts)
    header = {"schema": OUTPUT_SCHEMA, "source": run.describe(), "items": str(args.items), "items_name": Path(args.items).name,
              "conditions": names, "skipped_conditions": skipped, "smoke": bool(args.smoke),
              "new_surfaces": len(evaluation.scorer.placeholder), "shadowed": len(evaluation.scorer.shadowed),
              "shadowed_surfaces": evaluation.scorer.shadowed[:50], "unresolved_frames": evaluation.unresolved,
              "terms": len(evaluation.terms), "readers": evaluation.reader_summary(), "scorer": dict(evaluation.scorer.stats),
              "timing": result["timing"], "seconds": time.monotonic() - started}
    with gzip.open(output / "items.jsonl.gz", "wt") as handle:
        for row in result["items"].values():
            handle.write(json.dumps(json_ready(row)) + "\n")
    with gzip.open(output / "readers.jsonl.gz", "wt") as handle:
        for rows in evaluation._reader_rows.values():
            for row in rows:
                handle.write(json.dumps(json_ready(row)) + "\n")
    write_json(output / "summary.json", {**header, "summary": summary})
    (output / "report.md").write_text(render(header, summary))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary}


# -- pooled report over runs (seeds, models) -------------------------------------------------------------------------------------

def load_output(folder: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    with gzip.open(folder / "items.jsonl.gz", "rt") as handle:
        per_item = {row["id"]: row for row in map(json.loads, handle) if row}
    return summary, per_item


def model_label(source: dict[str, Any]) -> str:
    """`<host>-<model>` of an E9 run (e.g. `SmolLM2-360M-C5`), from its experiment name."""
    name = str(source.get("experiment") or Path(str(source.get("run"))).name)
    parts = name.split("-")
    if len(parts) >= 3 and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        host = next((p for p in parts if p.startswith(("SmolLM2", "Qwen"))), None)
        hostpart = "-".join(parts[parts.index(host):parts.index(host) + 2]) if host else "host"
        return f"{hostpart}-{parts[-2]}"
    return name


def pooled_contrast(tables: dict[tuple[str, str], dict[int, dict[str, float]]], a: tuple[str, str], b: tuple[str, str], *,
                    resamples: int = 2000, seed: int = 0) -> dict[str, Any] | None:
    """(model, condition) a − b: per item the mean over shared seeds of the difference; bootstrap over items."""
    ta, tb = tables.get(a), tables.get(b)
    if not ta or not tb:
        return None
    seeds = sorted(set(ta) & set(tb))
    if not seeds:
        seeds_a, seeds_b = sorted(ta), sorted(tb)
        if len(seeds_a) == 1 or len(seeds_b) == 1:      # a single-seed model (P0) against every seed of the other
            pairs = [(sa, sb) for sa in seeds_a for sb in seeds_b]
        else:
            return None
    else:
        pairs = [(s, s) for s in seeds]
    diffs: dict[str, list[float]] = defaultdict(list)
    for sa, sb in pairs:
        for item in set(ta[sa]) & set(tb[sb]):
            diffs[item].append(ta[sa][item] - tb[sb][item])
    if not diffs:
        return None
    keys = sorted(diffs)
    values = np.asarray([np.mean(diffs[k]) for k in keys])
    from ..experiments.e5_zeroshot import paired_difference
    result = paired_difference(values, np.zeros_like(values), resamples=resamples, seed=seed)
    return {**result, "seeds": [list(p) for p in pairs], "items": len(keys)}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from ..experiments.e5_common import fmt, fmt_ci, write_json
    tables: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    item_info: dict[str, dict[str, Any]] = {}
    sources = []
    for folder in args.inputs:
        folder = Path(folder)
        if not (folder / "summary.json").exists():
            sources.append({"folder": str(folder), "missing": True})
            continue
        summary, per_item = load_output(folder)
        label = model_label(summary["source"])
        seed = int(summary["source"].get("seed") or 0)
        sources.append({"folder": str(folder), "model": label, "seed": seed, "items": summary["items_name"],
                        "smoke": summary.get("smoke"), "conditions": summary["conditions"]})
        for item_id, row in per_item.items():
            item_info.setdefault(item_id, {"set": row["set"], **(row.get("meta") or {})})
        for cond in summary["conditions"]:
            tables[(label, cond)][seed] = correctness(per_item, cond, args.norm)
    spec = json.loads(Path(args.contrasts).read_text()) if args.contrasts else {"primary": [], "secondary": []}
    out: dict[str, Any] = {"sources": sources, "norm": args.norm, "accuracy": {}, "primary": [], "secondary": []}
    for (label, cond), by_seed in sorted(tables.items()):
        pooled: dict[str, list[float]] = defaultdict(list)
        for values in by_seed.values():
            for item, v in values.items():
                pooled[item].append(v)
        out["accuracy"][f"{label}|{cond}"] = {**bootstrap_mean(np.asarray([np.mean(v) for v in pooled.values()]),
                                                               resamples=args.resamples, seed=args.seed), "seeds": sorted(by_seed)}
    for family in ("primary", "secondary"):
        for c in spec.get(family, []):
            where = {**({"set": c["set"]} if c.get("set") else {}), **(c.get("where") or {})}
            keep = {i for i, info in item_info.items() if all(info.get(k) == v for k, v in where.items())}
            subset = {key: {seed: {i: v for i, v in values.items() if i in keep} for seed, values in by_seed.items()}
                      for key, by_seed in tables.items()} if where else tables
            result = pooled_contrast(subset, (c["a"]["model"], c["a"]["condition"]), (c["b"]["model"], c["b"]["condition"]),
                                     resamples=args.resamples, seed=args.seed)
            out[family].append({"name": c["name"], "a": c["a"], "b": c["b"], "where": where, "result": result})
    ps = [c["result"]["p_value"] for c in out["primary"] if c["result"]]
    adjusted = iter(holm_adjust(ps)) if ps else iter(())
    for c in out["primary"]:
        if c["result"]:
            c["holm_p"] = next(adjusted)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", out)
    lines = ["**SMOKE TEST — not a result.**", ""] if any(s.get("smoke") for s in sources) else []
    lines += [f"# Ranking benchmarks: pooled report ({args.norm})", "", f"Inputs: {len(sources)} run outputs.", "",
              "| Model | Condition | Accuracy [95% CI] | Seeds | Items |", "|---|---|---|---|---|"]
    for key, acc in out["accuracy"].items():
        label, cond = key.split("|", 1)
        lines.append(f"| {label} | {cond} | {fmt_ci(acc, 4)} | {acc['seeds']} | {acc['n']} |")
    for family in ("primary", "secondary"):
        if out[family]:
            lines += ["", f"## {family.capitalize()} contrasts" + (" (Holm over the primaries)" if family == "primary" else ""), "",
                      "| Name | a − b | Δ [95% CI] | p | Holm p | Items |", "|---|---|---|---|---|---|"]
            for c in out[family]:
                r = c["result"]
                desc = f"{c['a']['model']} {c['a']['condition']} − {c['b']['model']} {c['b']['condition']}" + (
                    f" ({', '.join(f'{k}={v}' for k, v in c['where'].items())})" if c.get("where") else "")
                lines.append(f"| {c['name']} | {desc} | {fmt_ci(r, 4) if r else 'not available'} | "
                             f"{fmt(r.get('p_value'), 4) if r else '—'} | {fmt(c.get('holm_p'), 4)} | {r['items'] if r else '—'} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="score an item set on one run under each condition")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--conditions", default="", help=f"comma-separated (default: {','.join(DEFAULT_CONDITIONS)})")
    ev.add_argument("--contrasts", default="", help="extra paired contrasts 'a|b,...' (every condition − none is always reported)")
    ev.add_argument("--limit", type=int, default=None); ev.add_argument("--device", default=None)
    ev.add_argument("--spread", action="store_true", help="with --limit: items evenly spaced over the file")
    ev.add_argument("--threads", type=int, default=None, help="torch CPU threads (CPU smoke: ≤ 4)")
    ev.add_argument("--batch-size", type=int, default=32); ev.add_argument("--token-budget", type=int, default=16384)
    ev.add_argument("--max-length", type=int, default=1024); ev.add_argument("--alias-table", type=Path, default=None)
    ev.add_argument("--seed", type=int, default=0); ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--smoke", action="store_true", help="label every output as a smoke test")
    ev.add_argument("--overwrite", action="store_true")
    rp = sub.add_parser("report", help="pool run outputs (seeds, models); pre-registered contrasts with Holm")
    rp.add_argument("--inputs", type=Path, nargs="+", required=True); rp.add_argument("--output", type=Path, required=True)
    rp.add_argument("--contrasts", type=Path, default=None, help="JSON {primary: [{name, a: {model, condition}, b: …}], secondary: […]}")
    rp.add_argument("--norm", choices=NORMS, default=PRIMARY_NORM)
    rp.add_argument("--resamples", type=int, default=2000); rp.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        document = run_evaluate(args)
        print(render(document, document["summary"]))
    else:
        run_report(args)
        print((Path(args.output) / "report.md").read_text())


if __name__ == "__main__":
    main()
