"""Span-level semantic channel (M4): alias linking, span tables, cardinality, injection.

Linking is deterministic and **prefix-causal**: whether a concept is injected at token `t`
depends only on the text up to the end of token `t`. Two rules follow, both found while
implementing formulation §4.2:

- `boundary="prefix"` (default): an alias "ends at token t" if the text up to the end of `t`
  ends with the alias and the alias starts at a word boundary. The right word boundary is *not*
  checked, because it depends on token `t+1` ("bank" + "ing"); requiring it would reveal the next
  token through the presence or absence of an injection. Among aliases ending at `t`, the longest
  one wins (looking back only), so "new york" is linked at "york" even if "city" follows, and
  "new york city" is linked again at "city".
- `boundary="next_token"`: the alias must end at a word boundary and the vector is injected at
  the token *after* the word (which already reveals that boundary), one step later.

Matching runs on character offsets, so one alias table serves every tokenizer; span lengths in
subtokens are computed per tokenizer from its offset mapping.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compose import FrameComposer, FrameSchedule

LINKER_VERSION = "1.1"   # 1.1: trie end marker can no longer collide with a "$" in text
_END = object()          # trie end marker; not a character, so it cannot collide with text
_WORD = re.compile(r"\w", re.UNICODE)
# Default fraction ρ of `SpanChannel.set_host_scale` (open decision 1): the injected row starts at ρ times the host's
# mean input-embedding row norm. 0.3 is SmolLM2-360M's own unscaled ratio for a composed (C3–C6) channel (a
# 256 → 960 projector gives rows of norm ≈ 1.12 against rows of 3.70; SmolLM2-135M: 0.87 / 3.18 ≈ 0.27), so scaling
# is a no-op-sized change there, while Qwen2.5-0.5B (rows of 0.46) starts ≈ 8× lower than unscaled.
HOST_SCALE_FRACTION = 0.3


def normalize_alias(text: str) -> str:
    """Lowercase, underscores to spaces, collapse whitespace."""
    return " ".join(text.replace("_", " ").lower().split())


def normalize_identifier_alias(text: str) -> str:
    """Identifier mode for code symbols: lowercase and collapse whitespace, but keep `_` (and `.`)
    verbatim, so `fetch_record_batch` and `lib.module.fetch_record_batch` match their occurrences in
    code. (`normalize_alias` turns `_` into a space, the WordNet lemma convention, so a snake_case
    alias could never match code.) The linker itself is unchanged: it matches the table's keys."""
    return " ".join(text.lower().split())


# Alias normalization modes of `AliasTable.from_pairs`; "default" is the one every recorded corpus used.
ALIAS_NORMALIZATIONS = {"default": normalize_alias, "identifier": normalize_identifier_alias}


@dataclass
class AliasTable:
    """Surface forms → link entries; an entry is a set of concepts sharing that surface form.

    A polysemous surface form links to one entry whose frame is the union of its concepts'
    frames (formulation §1.2); attentive composition then selects among them.
    """

    alias_to_entry: dict[str, int]
    entry_concepts: list[tuple[int, ...]]
    holdout: frozenset[int] = frozenset()
    normalization: str = "default"      # key of ALIAS_NORMALIZATIONS used to build the keys (provenance)

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, int]], *, holdout: Iterable[int] = (),
                   include_holdout: bool = False, normalization: str = "default") -> "AliasTable":
        """Build from (alias, concept) pairs. Unless `include_holdout`, held-out concepts and
        every alias pointing to them are removed (linker holdout, experiments §0.1).
        `normalization` selects the alias normalization (`ALIAS_NORMALIZATIONS`); "identifier"
        keeps underscores for code symbols (track T2)."""
        if normalization not in ALIAS_NORMALIZATIONS:
            raise ValueError(f"unknown alias normalization {normalization!r}; known: {sorted(ALIAS_NORMALIZATIONS)}")
        normalize = ALIAS_NORMALIZATIONS[normalization]
        held = frozenset(int(c) for c in holdout)
        by_alias: dict[str, set[int]] = {}
        for alias, concept in pairs:
            key = normalize(alias)
            if key and _WORD.search(key):
                by_alias.setdefault(key, set()).add(int(concept))
        if not include_holdout:
            # An alias that can mean a held-out concept is dropped entirely, not just narrowed:
            # otherwise its surviving senses would still expose the held-out string in training.
            by_alias = {a: cs for a, cs in by_alias.items() if not cs & held}
        entries: dict[tuple[int, ...], int] = {}
        alias_to_entry = {}
        for alias in sorted(by_alias):
            concepts = tuple(sorted(by_alias[alias]))
            alias_to_entry[alias] = entries.setdefault(concepts, len(entries))
        entry_concepts = [None] * len(entries)
        for concepts, index in entries.items():
            entry_concepts[index] = concepts
        return cls(alias_to_entry, entry_concepts, held, normalization)

    def without_holdout(self) -> "AliasTable":
        """Training view: drop every alias whose entry contains a held-out concept, keeping the
        entry numbering, so training and evaluation spans share entry ids."""
        kept = {alias: entry for alias, entry in self.alias_to_entry.items()
                if not set(self.entry_concepts[entry]) & self.holdout}
        return AliasTable(kept, self.entry_concepts, self.holdout, self.normalization)

    def heldout_entries(self) -> set[int]:
        return {i for i, concepts in enumerate(self.entry_concepts) if set(concepts) & self.holdout}

    def digest(self) -> str:
        record = {"aliases": sorted(self.alias_to_entry.items()), "entries": self.entry_concepts,
                  "holdout": sorted(self.holdout), "version": LINKER_VERSION}
        if getattr(self, "normalization", "default") != "default":     # default tables keep their recorded digest
            record["normalization"] = self.normalization
        payload = json.dumps(record).encode()
        return hashlib.sha256(payload).hexdigest()

    def entry_schedule(self, concept_frames: Sequence[Sequence[tuple[int, int]]]) -> FrameSchedule:
        """Frames of link entries: the union of member concepts' frames (duplicates removed)."""
        frames = []
        for concepts in self.entry_concepts:
            union: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for concept in concepts:
                for edge in concept_frames[concept]:
                    if tuple(edge) not in seen:
                        seen.add(tuple(edge)); union.append(tuple(edge))
            frames.append(union)
        return FrameSchedule.from_frames(frames)


@dataclass(frozen=True)
class Span:
    start_token: int
    end_token: int        # last subtoken of the alias
    inject_token: int     # position that receives the vector (== end_token for prefix mode)
    entry: int
    confidence: float
    length: int           # subtokens


class CausalLinker:
    """Longest backward alias match at every token end (see module docstring)."""

    def __init__(self, table: AliasTable, *, boundary: str = "prefix", min_subtokens: int = 2) -> None:
        if boundary not in {"prefix", "next_token"}:
            raise ValueError("boundary must be 'prefix' or 'next_token'")
        self.table, self.boundary, self.min_subtokens = table, boundary, min_subtokens
        # Reversed character trie: walk backwards from a token end.
        self.trie: dict[str, Any] = {}
        self.max_length = 0
        for alias, entry in table.alias_to_entry.items():
            node = self.trie
            for char in reversed(alias):
                node = node.setdefault(char, {})
            node[_END] = entry
            self.max_length = max(self.max_length, len(alias))

    def _match_ending_at(self, lowered: str, end: int) -> tuple[int, int] | None:
        """Longest alias equal to `lowered[s:end]` with a word boundary before `s`."""
        node, best, position = self.trie, None, end - 1
        spaced = False
        while position >= 0 and end - position <= self.max_length + 1:
            char = lowered[position]
            if char.isspace():
                if spaced:           # collapse runs of whitespace in the text
                    position -= 1; continue
                char, spaced = " ", True
            else:
                spaced = False
            node = node.get(char)
            if node is None:
                break
            start = position
            if _END in node and (start == 0 or not (lowered[start - 1].isalnum() or lowered[start - 1] == "_")):
                best = (start, node[_END])
            position -= 1
        return best

    def _matches_ending_at(self, lowered: str, end: int) -> list[tuple[int, int]]:
        """Every alias equal to `lowered[s:end]` with a word boundary before `s` (shortest first);
        `_match_ending_at` returns the last of these."""
        node, found, position = self.trie, [], end - 1
        spaced = False
        while position >= 0 and end - position <= self.max_length + 1:
            char = lowered[position]
            if char.isspace():
                if spaced:
                    position -= 1; continue
                char, spaced = " ", True
            else:
                spaced = False
            node = node.get(char)
            if node is None:
                break
            if _END in node and (position == 0 or not (lowered[position - 1].isalnum() or lowered[position - 1] == "_")):
                found.append((position, node[_END]))
            position -= 1
        return found

    def all_matches(self, text: str, offsets: Sequence[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
        """Every alias match before the longest-match choice of `link` (prefix boundary only):
        `(token, first_token, entry, characters)` for each alias ending at each token end and covering
        ≥ `min_subtokens` subtokens. `link` keeps, per token, the match with the most characters and drops
        it when it is shorter than `min_subtokens`; a shorter alias ending at the same token never covers
        more subtokens, so dropping short matches here first changes nothing (`data.match_corpus`)."""
        if self.boundary != "prefix":
            raise ValueError("all_matches supports the prefix boundary only")
        lowered = text.lower()
        starts = [start for start, _ in offsets]
        rows = []
        for token, (token_start, token_end) in enumerate(offsets):
            if token_end <= token_start:
                continue
            for char_start, entry in self._matches_ending_at(lowered, token_end):
                first = min(token, bisect_right(starts, char_start, 0, token + 1) - 1)
                if token - first + 1 >= self.min_subtokens:
                    rows.append((token, first, entry, token_end - char_start))
        return rows

    def link(self, text: str, offsets: Sequence[tuple[int, int]]) -> list[Span]:
        """Spans for one text given its tokenizer offset mapping (one (start, end) per token)."""
        lowered = text.lower()
        starts = [start for start, _ in offsets]
        spans: list[Span] = []
        for token, (token_start, token_end) in enumerate(offsets):
            if token_end <= token_start:
                continue
            match = self._match_ending_at(lowered, token_end)
            if match is None:
                continue
            char_start, entry = match
            if self.boundary == "next_token":
                # Inject at the next token, whose presence reveals whether the word ended.
                if token + 1 >= len(offsets):
                    continue
                if token_end < len(lowered) and (lowered[token_end].isalnum() or lowered[token_end] == "_"):
                    continue
                inject = token + 1
            else:
                inject = token
            # Token starts are non-decreasing: the last token starting at or before the alias start
            # covers it (binary search; a linear scan made long documents quadratic).
            first = min(token, bisect_right(starts, char_start, 0, token + 1) - 1)
            length = token - first + 1
            if length < self.min_subtokens:
                continue
            confidence = 1.0 / len(self.table.entry_concepts[entry])
            spans.append(Span(first, token, inject, entry, confidence, length))
        return spans


def link_batch(linker: CausalLinker, texts: Sequence[str], offsets: Sequence[Sequence[tuple[int, int]]]) -> dict[str, Tensor]:
    """Flatten spans of a batch into tensors for `SpanChannel`."""
    rows = [(b, s.start_token, s.end_token, s.inject_token, s.entry, s.confidence, s.length)
            for b, (text, offset) in enumerate(zip(texts, offsets)) for s in linker.link(text, offset)]
    if not rows:
        empty = torch.zeros(0, dtype=torch.long)
        return {"batch": empty, "start": empty, "end": empty, "inject": empty, "entry": empty,
                "confidence": torch.zeros(0), "length": empty}
    columns = list(zip(*rows))
    return {
        "batch": torch.tensor(columns[0]), "start": torch.tensor(columns[1]), "end": torch.tensor(columns[2]),
        "inject": torch.tensor(columns[3]), "entry": torch.tensor(columns[4]),
        "confidence": torch.tensor(columns[5], dtype=torch.float32), "length": torch.tensor(columns[6]),
    }


# -- cardinality ------------------------------------------------------------------------------

def alias_subtoken_lengths(table: AliasTable, tokenizer: Any) -> dict[str, int]:
    """Subtoken length of each alias as it appears mid-sentence (with a leading space)."""
    return {alias: len(tokenizer.encode(" " + alias, add_special_tokens=False)) for alias in table.alias_to_entry}


def cardinality_report(
    table: AliasTable, tokenizer: Any, texts: Iterable[str], *, thresholds: Sequence[int] = (1, 2, 3, 4),
    boundary: str = "prefix", window: int = 1024,
) -> list[dict[str, Any]]:
    """Feasibility table (formulation §4.1) for one tokenizer × alias table × corpus sample.

    For each `ℓ_min`: linkable entries (an alias of ≥ ℓ_min subtokens), entries linked in the
    corpus, span occurrences, fraction of tokens inside linked spans, occurrences-per-entry
    histogram, and mean distinct entries per `window`-token window (the channel's per-batch cost).
    """
    lengths = alias_subtoken_lengths(table, tokenizer)
    linker = CausalLinker(table, boundary=boundary, min_subtokens=1)
    all_spans: list[tuple[int, list[Span]]] = []
    total_tokens = 0
    for text in texts:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
        total_tokens += len(offsets)
        all_spans.append((len(offsets), linker.link(text, offsets)))
    rows = []
    for threshold in thresholds:
        linkable = {entry for alias, entry in table.alias_to_entry.items() if lengths[alias] >= threshold}
        counts: Counter[int] = Counter()
        covered = 0
        windows: list[int] = []
        for n_tokens, spans in all_spans:
            kept = [s for s in spans if s.length >= threshold]
            counts.update(s.entry for s in kept)
            covered_positions = {t for s in kept for t in range(s.start_token, s.end_token + 1)}
            covered += len(covered_positions)
            for start in range(0, max(n_tokens, 1), window):
                windows.append(len({s.entry for s in kept if start <= s.inject_token < start + window}))
        histogram = Counter(min(c, 1000) for c in counts.values())
        rows.append({
            "min_subtokens": threshold, "linkable_entries": len(linkable), "linked_entries": len(counts),
            "span_occurrences": sum(counts.values()), "covered_token_fraction": covered / max(1, total_tokens),
            "tokens": total_tokens,
            "entries_seen_once": histogram.get(1, 0), "entries_seen_2_to_9": sum(v for k, v in histogram.items() if 2 <= k < 10),
            "entries_seen_10_plus": sum(v for k, v in histogram.items() if k >= 10),
            "mean_distinct_entries_per_window": sum(windows) / max(1, len(windows)),
        })
    return rows


# -- the channel --------------------------------------------------------------------------------

class SpanChannel(nn.Module):
    """Add gated, projected concept vectors to input embeddings at linked positions.

    `h_t^{(0)} = E[x_t] + 1[t = inject] · g_t · P c_j(q_t)`, `g_t = σ(w_g·[E[x_t]; P c_j; conf] + b_g)`.
    `mode="free"` replaces composition by a free per-entry table (control C2); `mode="random"`
    uses fixed random per-entry vectors (control C1); `mode="hashed"` uses a hashed table keyed by
    the span's subtoken ids (control C1h). Otherwise rows come from the `FrameComposer`.

    Host scale (opt-in, `set_host_scale`; open decision 1): every row is multiplied by a fixed scalar
    `s = ρ · n̄_E / n̄_c`, with `n̄_E` the host's mean input-embedding row norm and `n̄_c` this channel's
    mean row norm when `set_host_scale` is called (at build time, i.e. at initialization), so the injected
    row starts at the same fraction ρ of an embedding row on every host and in every channel mode. `s` is
    the buffer `host_scale`, registered only then (channels without it keep their state-dict keys), so it is
    saved with every checkpoint and restored exactly by every rebuild that loads the state.
    """

    def __init__(self, composer: FrameComposer | None, model_dimension: int, *, entry_count: int,
                 mode: str = "compose", hashed_buckets: int = 0, gate_bias: float = -2.0,
                 semantic_dimension: int = 0, free_dimension: int = 0) -> None:
        super().__init__()
        if mode not in {"compose", "free", "random", "hashed"}:
            raise ValueError("mode must be compose, free, random or hashed")
        if mode == "compose" and composer is None:
            raise ValueError("compose mode needs a FrameComposer")
        self.mode, self.composer, self.entry_count = mode, composer, entry_count
        source_dimension = composer.atomics.shape[1] if mode == "compose" else model_dimension
        if mode == "compose":
            self.projector = nn.Linear(source_dimension, model_dimension, bias=False)
        elif mode == "free":
            # `free_dimension` > 0: a low-dimensional free table plus a projector, so the control's
            # parameter count can be matched to the composition channel.
            width = free_dimension or model_dimension
            self.table = nn.Embedding(entry_count, width)
            nn.init.normal_(self.table.weight, std=width**-0.5)
            self.free_projector = nn.Linear(width, model_dimension, bias=False) if free_dimension else None
            # Entries never trained (held-out concepts) fall back to the mean of trained rows (C2).
            self.register_buffer("unseen", torch.zeros(entry_count, dtype=torch.bool))
        elif mode == "random":
            self.register_buffer("table_fixed", torch.randn(entry_count, model_dimension) / model_dimension**0.5)
            self.scale = nn.Parameter(torch.ones(()))
        else:
            if hashed_buckets < 1:
                raise ValueError("hashed mode needs hashed_buckets")
            self.table = nn.Embedding(hashed_buckets, model_dimension)
            nn.init.normal_(self.table.weight, std=model_dimension**-0.5)
            self.hashed_buckets = hashed_buckets
        self.gate = nn.Linear(2 * model_dimension + 1, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)
        self.semantic_head = nn.Linear(model_dimension, model_dimension, bias=False) if semantic_dimension else None

    def rows(self, spans: dict[str, Tensor], input_ids: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        """The rows the channel injects (before the gate), host scale included."""
        rows = self._rows(spans, input_ids, context)
        scale = self._buffers.get("host_scale")
        return rows if scale is None else rows * scale.to(rows.dtype)

    @torch.no_grad()
    def mean_row_norm(self, sample: int = 4096) -> float:
        """Mean L2 norm of the unscaled rows of up to `sample` evenly spaced entries (context-free, as stored:
        a free table's own rows; a hashed memory's bucket rows)."""
        if self.mode == "hashed":
            weight = self.table.weight
            index = torch.linspace(0, weight.shape[0] - 1, min(sample, weight.shape[0])).round().long()
            return float(weight[index].float().norm(dim=-1).mean())
        device = next(self.parameters()).device
        entries = torch.linspace(0, self.entry_count - 1, min(sample, self.entry_count)).round().long().unique().to(device)
        return float(self._rows({"entry": entries}).float().norm(dim=-1).mean())

    def set_host_scale(self, host_row_norm: float, fraction: float = HOST_SCALE_FRACTION) -> dict[str, float]:
        """Fix `host_scale = fraction · host_row_norm / mean_row_norm()` (see the class docstring); returns
        the record {host_row_norm, channel_row_norm, fraction, scale}."""
        if not host_row_norm > 0 or not fraction > 0:
            raise ValueError("host_row_norm and fraction must be positive")
        reference = self.mean_row_norm()
        scale = float(fraction) * float(host_row_norm) / reference
        if "host_scale" in self._buffers:
            self.host_scale.fill_(scale)
        else:
            self.register_buffer("host_scale", torch.tensor(scale, dtype=torch.float32, device=next(self.parameters()).device))
        return {"host_row_norm": float(host_row_norm), "channel_row_norm": reference, "fraction": float(fraction), "scale": scale}

    def _rows(self, spans: dict[str, Tensor], input_ids: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        entries = spans["entry"]
        if self.mode == "compose":
            return self.projector(self.composer.compose(entries, context))
        if self.mode == "free":
            rows = self.table(entries)
            if bool(self.unseen.any()):
                fallback = self.table.weight[~self.unseen].mean(0)
                rows = torch.where(self.unseen[entries][:, None], fallback.expand_as(rows), rows)
            return self.free_projector(rows) if self.free_projector is not None else rows
        if self.mode == "random":
            return self.scale * self.table_fixed[entries]
        if input_ids is None:
            raise ValueError("hashed mode needs input_ids")
        keys = []
        for b, s, e in zip(spans["batch"].tolist(), spans["start"].tolist(), spans["end"].tolist()):
            keys.append(hash(tuple(input_ids[b, s:e + 1].tolist())) % self.hashed_buckets)
        return self.table(torch.tensor(keys, device=entries.device))

    def add_entries(self, count: int, frames: Sequence[Iterable[tuple[int, int]]] | None = None, *, seed: int = 0) -> Tensor:
        """Append `count` link entries at evaluation time (E9 zero-shot insertion); returns their ids.

        Composition (`compose`) appends the entries' `frames` to the composer (rows composed from the
        existing atomics and relations); the free table (C2) appends rows marked unseen, which fall
        back to the mean of the trained rows; the random control (C1) appends fixed random vectors
        drawn from `seed`; a hashed memory (C1h) is keyed by subtokens and needs nothing. Rows of
        existing entries are unchanged in every mode.
        """
        if count < 0 or (frames is not None and len(frames) != count):
            raise ValueError("frames must hold one frame per new entry")
        start = self.entry_count
        if self.mode == "compose":
            if frames is None:
                raise ValueError("compose mode needs the new entries' frames")
            ids = self.composer.add_concepts(frames)
            if int(ids[0] if count else start) != start:
                raise ValueError("the composer's concepts are not the channel's entries")
        elif self.mode == "free":
            weight = self.table.weight
            table = nn.Embedding(start + count, weight.shape[1], device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                table.weight.zero_()
                table.weight[:start] = weight.detach()
            table.weight.requires_grad_(weight.requires_grad)
            self.table = table
            self.unseen = torch.cat([self.unseen, torch.ones(count, dtype=torch.bool, device=self.unseen.device)])
        elif self.mode == "random":
            fixed = self.table_fixed
            extra = torch.randn(count, fixed.shape[1], generator=torch.Generator().manual_seed(seed)) / fixed.shape[1] ** 0.5
            self.table_fixed = torch.cat([fixed, extra.to(fixed)])
        self.entry_count = start + count
        return torch.arange(start, start + count)

    def set_unseen(self, entries: Tensor | Iterable[int]) -> None:
        """Mark entries that received no training signal (free-table control only)."""
        if self.mode == "free":
            self.unseen.zero_()
            index = torch.as_tensor(list(entries) if not isinstance(entries, Tensor) else entries, dtype=torch.long)
            if index.numel():
                self.unseen[index.to(self.unseen.device)] = True

    def forward(self, embeddings: Tensor, spans: dict[str, Tensor], *, input_ids: Tensor | None = None,
                context: Tensor | None = None) -> Tensor:
        """`embeddings`: (batch, time, d). `context`: (spans, context_dim) or None."""
        if spans["entry"].numel() == 0:
            return embeddings
        batch, position = spans["batch"].to(embeddings.device), spans["inject"].to(embeddings.device)
        rows = self.rows({k: v.to(embeddings.device) for k, v in spans.items()}, input_ids, context).to(embeddings.dtype)
        token_vectors = embeddings[batch, position]
        confidence = spans["confidence"].to(embeddings.device, embeddings.dtype)[:, None]
        gate = torch.sigmoid(self.gate(torch.cat([token_vectors, rows, confidence], -1)))
        addition = torch.zeros_like(embeddings)
        addition = addition.index_put((batch, position), gate * rows, accumulate=True)
        return embeddings + addition

    def semantic_loss(self, hidden: Tensor, spans: dict[str, Tensor], *, temperature: float = 0.1,
                      input_ids: Tensor | None = None) -> Tensor:
        """InfoNCE between `W_o h_{s−1}` and the span's concept row (formulation §4.3)."""
        if self.semantic_head is None:
            raise ValueError("channel built without semantic_dimension")
        keep = spans["start"] > 0
        if not bool(keep.any()):
            return hidden.new_zeros(())
        subset = {k: v[keep] for k, v in spans.items()}
        queries = self.semantic_head(hidden[subset["batch"], subset["start"] - 1])
        rows = self.rows(subset, input_ids)
        unique, inverse = torch.unique(subset["entry"], return_inverse=True)
        keys = torch.zeros(unique.numel(), rows.shape[1], device=rows.device, dtype=rows.dtype).index_copy(0, inverse, rows)
        logits = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T / temperature
        return F.cross_entropy(logits, inverse)
