"""Self-query (E12, author decision 62): decode a model's own concept store into text — the `recall` tool.

The E9 span channel composes a term's vector from its frame, `c = Σ_e T_{r_e}(a_e)`, and the model computes with that
vector. Because the store is a binding of roles and fillers, it can be *read back*: unbind a role with the operator's
own unbinding (`relations.RelationTransform.unbind`; the primary method of `relations.readout_method`), clean the result
up against the trained atomic dictionary (`cleanup`), and write the recovered edge as a sentence. This module is the
model-independent part of that read-back (the experiment harness is `vsa_embed.experiments.e12_self_query`):

- `RecallStore` — a trained `FrameComposer` read as a store. Stores are **static frame bundles** (every edge weight 1, as
  binding step 1's `static` condition and the readout arm's store); an entry's store or the store of any frame (a new
  word, a twin, an edited frame) composed from the trained atomics and relations.
  - `decode_slots(vector, slots)` — the frame read back: for each slot (relation, with its multiplicity m) the top-m
    fillers after unbinding and cleanup. `cleanup="typed"` (the default; the atomics observed under the relation in the
    ontology frames) or `"all"` (every atomic). Slotted operators (decision 61c) clean up within the relation's slot.
  - `decode_free(vector, thresholds)` — the same without the slot list: every relation is unbound and kept when its best
    typed cosine reaches the relation's presence threshold (`presence_thresholds` calibrates them on known frames).
  - `chain(vector, r1, r2, atom_entry)` — chained recall: unbind r1, clean up, follow the recovered filler to the entry it
    names (`atom_entry`), unbind r2 from *that* entry's store (two hops through local stores).
  - `reverse(relation, filler, pool)` — reverse lookup: scan a pool of stores for the pair, scoring each store by
    `cos(c, T_r(a_F))` (slotted: within the relation's slot).
  - A **role-blind store** (the untyped composer, method `bundle`, which has no unbinding) returns the bundle readout:
    one line "associated with" the top-K atomics of the vector itself, K = the number of slots (no role is stored, so
    none is claimed). A role query on it (`decode_role`) cleans the vector up within the relation's candidates — the
    type-restricted bundle readout. Translation (C5tr) unbinds by subtraction, as its probe does.
- `RecallLine` / `Recall` — what the tool returns: lines of (subject, relation or None, fillers with their cleanup
  cosines), plus the call that produced them.
- `RecallWriter` — writes a recall as text with confidences, in the track's own statement wording
  (`RelationTemplates.statement`, e.g. "X depends on Y."), one line per decoded edge, under the call that produced it:

      recall(dalkkloushfoltquark):
      - dalkkloushfoltquark is a system. (0.91)
      - dalkkloushfoltquark depends on Tindbreish Migration. (0.58)

  `style="fields"` writes "- depends on: Tindbreish Migration (0.58)"; `confidence=False` drops the numbers. A symbolic
  lookup (the gold frame) goes through the same writer with confidence 1.00, so recall and lookup differ only in content.

Everything here runs on the CPU in float32 and is deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .cleanup import relation_candidates
from .relations import readout_method

CLEANUPS = ("typed", "all")
STYLES = ("statements", "fields")
NEG = float("-inf")


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class Filler:
    atom: int
    score: float                         # cosine of the unbound vector (role-blind: of the store) to the atomic


@dataclass
class RecallLine:
    """One recalled line: `subject` (a surface; None = the call's subject), `relation` (None: a role-blind
    "associated with" line), the fillers (best first)."""

    relation: int | None
    fillers: list[Filler]
    subject: str | None = None


@dataclass
class Recall:
    """A tool result: the call (`kind` ∈ frame, role, chain, reverse, symbolic, bundle) and its lines."""

    kind: str
    lines: list[RecallLine]
    call: str = ""
    method: str = ""
    cleanup: str = "typed"
    meta: dict[str, Any] = field(default_factory=dict)

    def edges(self) -> list[tuple[int | None, int, float]]:
        """(relation, atom, score) of every line's fillers (role-blind lines: relation None)."""
        return [(line.relation, f.atom, f.score) for line in self.lines for f in line.fillers]


# ---------------------------------------------------------------- the store


class RecallStore:
    """A trained composer read as a store of static frame bundles (module docstring)."""

    def __init__(self, composer: Any, *, method: str | None = None, chunk: int = 2048) -> None:
        composer.eval()
        self.composer = composer
        schedule = composer.schedule
        self.method = method or readout_method(composer.transform)
        self.role_blind = self.method == "bundle"
        self.atomics = composer.atomic_vectors().detach().float().cpu()
        self.keys = F.normalize(self.atomics, dim=-1)
        self.relation_count = int(composer.relation_count)
        self.atomic_count = int(self.atomics.shape[0])
        relations, fillers = schedule.relations.cpu(), schedule.fillers.cpu()
        candidates = relation_candidates(relations, fillers, self.relation_count, self.atomic_count)
        candidates[~candidates.any(-1)] = True                  # a relation with no edge: every atomic is a candidate
        self.candidates = candidates
        counts = torch.zeros(self.relation_count, self.atomic_count, dtype=torch.long)
        if relations.numel():
            counts.index_put_((relations.long(), fillers.long()), torch.ones(relations.numel(), dtype=torch.long), accumulate=True)
        self.counts = counts
        # the relation each atomic is most often a filler of (role-blind lines word an atomic as that relation does)
        self.home = torch.where(counts.sum(0) > 0, counts.argmax(0), torch.full((self.atomic_count,), -1, dtype=torch.long))
        masks = composer.slot_masks() if hasattr(composer, "slot_masks") else None
        self.slot_masks = None if masks is None else masks.cpu()
        slot_of = composer.slot_of() if hasattr(composer, "slot_of") else None
        self.slot_of = None if slot_of is None else slot_of.cpu()
        self.chunk = int(chunk)
        self._entry_vectors: Tensor | None = None

    # -- stores --------------------------------------------------------------------------------------------------

    @property
    def offsets(self) -> np.ndarray:
        return self.composer.schedule.offsets.cpu().numpy()

    def frame(self, entry: int) -> list[tuple[int, int]]:
        schedule = self.composer.schedule
        lo, hi = int(schedule.offsets[entry]), int(schedule.offsets[entry + 1])
        return list(zip(schedule.relations[lo:hi].tolist(), schedule.fillers[lo:hi].tolist()))

    @torch.no_grad()
    def frame_vector(self, frame: Sequence[tuple[int, int]]) -> Tensor:
        """The static store `Σ_e T_{r_e}(a_e)` of a frame (zero for an empty frame)."""
        if not frame:
            return torch.zeros(self.atomics.shape[1])
        relations = torch.tensor([int(r) for r, _ in frame], dtype=torch.long)
        fillers = self.composer.atomic_vectors().detach()[torch.tensor([int(f) for _, f in frame], dtype=torch.long)]
        return self.composer.transform(relations.to(fillers.device), fillers).float().sum(0).cpu()

    @torch.no_grad()
    def frame_vectors(self, frames: Sequence[Sequence[tuple[int, int]]]) -> Tensor:
        return torch.stack([self.frame_vector(f) for f in frames]) if frames else torch.zeros(0, self.atomics.shape[1])

    @torch.no_grad()
    def entry_vectors(self) -> Tensor:
        """`(entries, d)` static stores of every entry of the composer's schedule (zero for an entry without edges)."""
        if self._entry_vectors is None:
            offsets = self.offsets
            out = torch.zeros(offsets.size - 1, self.atomics.shape[1])
            have = np.flatnonzero(np.diff(offsets) > 0)
            for start in range(0, have.size, 8192):
                ids = torch.as_tensor(have[start:start + 8192], dtype=torch.long)
                out[ids] = self.composer.raw_bundle(ids.to(self.composer.atomics.device), uniform=True)[0].float().cpu()
            self._entry_vectors = out
        return self._entry_vectors

    # -- unbinding and cleanup ------------------------------------------------------------------------------------

    @torch.no_grad()
    def unbound(self, relations: Tensor, vectors: Tensor) -> Tensor:
        """The filler estimates `T_r⁻¹ c` (role-blind store: the vector itself, whatever the role)."""
        if self.role_blind:
            return vectors.float()
        device = self.composer.atomics.device
        out = self.composer.unbind(relations.to(device), vectors.to(device, self.composer.atomics.dtype), method=self.method)
        return out.float().cpu()

    def _cosines(self, relations: Tensor, queries: Tensor) -> Tensor:
        """`(n, atomics)` cosine of each query to every atomic; slotted: within the relation's slot."""
        queries = F.normalize(queries.float(), dim=-1)
        if self.slot_masks is None:
            return queries @ self.keys.T
        scores = torch.empty(queries.shape[0], self.atomic_count)
        slots = self.slot_of[relations]
        for g in range(self.slot_masks.shape[0]):
            rows = (slots == g).nonzero(as_tuple=True)[0]
            if rows.numel():
                scores[rows] = queries[rows] @ F.normalize(self.atomics * self.slot_masks[g], dim=-1).T
        return scores

    @torch.no_grad()
    def filler_scores(self, relations: Tensor, vectors: Tensor, *, cleanup: str = "typed") -> Tensor:
        """`(n, atomics)` cleanup scores after unbinding `relations` from `vectors` (`typed`: −inf off the relation's
        candidates). A role-blind store scores the vector itself (`typed`: the type-restricted bundle readout)."""
        if cleanup not in CLEANUPS:
            raise ValueError(f"cleanup must be one of {CLEANUPS}")
        relations = torch.as_tensor(relations, dtype=torch.long)
        out = []
        for start in range(0, relations.numel(), self.chunk):
            part = slice(start, start + self.chunk)
            scores = self._cosines(relations[part], self.unbound(relations[part], vectors[part]))
            if cleanup == "typed":
                scores = scores.masked_fill(~self.candidates[relations[part]], NEG)
            out.append(scores)
        return torch.cat(out) if out else torch.zeros(0, self.atomic_count)

    @torch.no_grad()
    def bundle_scores(self, vectors: Tensor) -> Tensor:
        """`(n, atomics)` cosine of each store to every atomic: the role-blind bundle readout."""
        return F.normalize(vectors.float(), dim=-1) @ self.keys.T

    @staticmethod
    def _top(scores: Tensor, k: int) -> list[Filler]:
        finite = int(torch.isfinite(scores).sum())
        k = min(int(k), finite)
        if k <= 0:
            return []
        values, index = torch.topk(scores, k)
        return [Filler(int(a), float(s)) for a, s in zip(index.tolist(), values.tolist())]

    # -- the tool's operations ------------------------------------------------------------------------------------

    @torch.no_grad()
    def decode_slots(self, vector: Tensor, slots: Sequence[int], *, cleanup: str = "typed") -> list[RecallLine]:
        """The frame read back from `vector`: per distinct relation of `slots` (in first-occurrence order, multiplicity m)
        its top-m fillers. Role-blind store: one "associated with" line of the top-len(slots) atomics."""
        if not len(slots):
            return []
        if self.role_blind:
            return [RecallLine(None, self._top(self.bundle_scores(vector[None])[0], len(slots)))]
        multiplicity = Counter(int(r) for r in slots)
        order = list(dict.fromkeys(int(r) for r in slots))
        scores = self.filler_scores(torch.tensor(order), vector[None].expand(len(order), -1), cleanup=cleanup)
        return [RecallLine(r, self._top(scores[i], multiplicity[r])) for i, r in enumerate(order)]

    @torch.no_grad()
    def decode_role(self, vector: Tensor, relation: int, *, k: int = 1, cleanup: str = "typed") -> RecallLine:
        """`recall(term, role)`: the top-k fillers of one relation (role-blind store: the type-restricted bundle readout)."""
        scores = self.filler_scores(torch.tensor([int(relation)]), vector[None], cleanup=cleanup)[0]
        return RecallLine(int(relation), self._top(scores, k))

    @torch.no_grad()
    def best_scores(self, vectors: Tensor, *, cleanup: str = "typed") -> Tensor:
        """`(n, relations)`: per store and relation the best cleanup cosine after unbinding that relation."""
        n = vectors.shape[0]
        relations = torch.arange(self.relation_count).repeat(n)
        expanded = vectors.repeat_interleave(self.relation_count, 0)
        return self.filler_scores(relations, expanded, cleanup=cleanup).max(-1).values.view(n, self.relation_count)

    @torch.no_grad()
    def presence_thresholds(self, entries: Sequence[int], *, cleanup: str = "typed") -> Tensor:
        """Per relation the presence threshold that maximizes the F1 of "relation present in the frame" over the given
        entries' stores (best cleanup cosine ≥ threshold); NaN-free (a relation never present gets +inf)."""
        entries = [int(e) for e in entries]
        stores = self.entry_vectors()[torch.tensor(entries, dtype=torch.long)]
        best = self.best_scores(stores, cleanup=cleanup)
        present = torch.zeros(len(entries), self.relation_count, dtype=torch.bool)
        for i, e in enumerate(entries):
            for r, _ in self.frame(e):
                present[i, r] = True
        out = torch.full((self.relation_count,), math.inf)
        for r in range(self.relation_count):
            actual = present[:, r]
            if not bool(actual.any()):
                continue
            values, order = torch.sort(best[:, r], descending=True)
            hits = torch.cumsum(actual[order].double(), 0)
            # a threshold at the i-th largest value predicts every store scoring ≥ it: ties share one cut (the last of a run)
            last = torch.ones_like(values, dtype=torch.bool)
            last[:-1] = values[:-1] != values[1:]
            predicted = torch.arange(1, values.numel() + 1, dtype=torch.float64)
            f1 = torch.where(last, 2 * hits / (predicted + float(actual.sum())), torch.full_like(hits, -1.0))
            out[r] = float(values[int(torch.argmax(f1))])
        return out

    @torch.no_grad()
    def decode_free(self, vector: Tensor, thresholds: Tensor, *, cleanup: str = "typed") -> list[RecallLine]:
        """Slot-free read-back: every relation whose best cleanup cosine reaches its presence threshold, top-1 filler.
        Role-blind store: the bundle readout of the atomics whose cosine reaches the smallest finite threshold."""
        if self.role_blind:
            scores = self.bundle_scores(vector[None])[0]
            floor = float(thresholds[torch.isfinite(thresholds)].min()) if bool(torch.isfinite(thresholds).any()) else math.inf
            keep = [Filler(int(a), float(s)) for s, a in zip(*torch.sort(scores, descending=True)) if float(s) >= floor]
            return [RecallLine(None, keep)] if keep else []
        scores = self.filler_scores(torch.arange(self.relation_count), vector[None].expand(self.relation_count, -1), cleanup=cleanup)
        lines = []
        for r in range(self.relation_count):
            best = self._top(scores[r], 1)
            if best and best[0].score >= float(thresholds[r]):
                lines.append(RecallLine(r, best))
        return lines

    @torch.no_grad()
    def chain(self, vector: Tensor, first: int, second: int, atom_entry: np.ndarray, *, cleanup: str = "typed",
              bridge_vector: Callable[[int], Tensor] | None = None) -> tuple[list[RecallLine], dict[str, Any]]:
        """Chained recall along (first, second): hop 1 from `vector`, the recovered filler's entry (`atom_entry`; −1: the
        filler names no concept and the chain stops), hop 2 from that entry's store (`bridge_vector(entry)` overrides it).
        Returns the lines (hop 2's subject is left to the caller: meta `bridge`) and meta {bridge, hop1, hop2}."""
        hop1 = self.decode_role(vector, first, cleanup=cleanup)
        lines = [hop1]
        meta: dict[str, Any] = {"bridge": -1, "hop1": hop1.fillers[0].atom if hop1.fillers else -1, "hop2": -1}
        if not hop1.fillers:
            return lines, meta
        atom = hop1.fillers[0].atom
        bridge = int(atom_entry[atom]) if 0 <= atom < len(atom_entry) else -1
        meta["bridge"] = bridge
        if bridge < 0:
            return lines, meta
        store = bridge_vector(bridge) if bridge_vector is not None else self.entry_vectors()[bridge]
        hop2 = self.decode_role(store, second, cleanup=cleanup)
        lines.append(hop2)
        meta["hop2"] = hop2.fillers[0].atom if hop2.fillers else -1
        return lines, meta

    @torch.no_grad()
    def reverse_scores(self, relation: int, filler: int, pool: Tensor) -> Tensor:
        """`(pool,)` reverse-lookup scores `cos(c, T_r(a_F))` of each store in `pool` (role-blind: `cos(c, a_F)`;
        slotted: within the relation's slot)."""
        if self.role_blind:
            probe = self.keys[int(filler)]
            return F.normalize(pool.float(), dim=-1) @ probe
        device = self.composer.atomics.device
        relation_t = torch.tensor([int(relation)], device=device)
        bound = self.composer.transform(relation_t, self.composer.atomic_vectors().detach()[[int(filler)]]).float().cpu()[0]
        if self.slot_masks is None:
            return F.normalize(pool.float(), dim=-1) @ F.normalize(bound, dim=0)
        mask = self.slot_masks[self.slot_of[int(relation)]].float()
        return F.normalize(pool.float() * mask, dim=-1) @ F.normalize(bound * mask, dim=0)

    @torch.no_grad()
    def reverse(self, relation: int, filler: int, pool: Tensor, *, k: int = 5) -> list[tuple[int, float]]:
        """The top-k (pool index, score) of the reverse lookup."""
        scores = self.reverse_scores(relation, filler, pool)
        k = min(int(k), scores.numel())
        if k <= 0:
            return []
        values, index = torch.topk(scores, k)
        return list(zip(index.tolist(), values.tolist()))


def symbolic_lines(frame: Sequence[tuple[int, int]], *, role_blind: bool = False) -> list[RecallLine]:
    """The gold frame as recall lines (confidence 1.0), in the slot order `decode_slots` uses."""
    if role_blind:
        return [RecallLine(None, [Filler(int(f), 1.0) for _, f in frame])]
    order = list(dict.fromkeys(int(r) for r, _ in frame))
    return [RecallLine(r, [Filler(int(f), 1.0) for q, f in frame if int(q) == r]) for r in order]


def roleless(lines: Sequence[RecallLine]) -> list[RecallLine]:
    """The same recalled fillers with their roles removed: one "associated with" line, fillers in a role-independent
    order (by atomic id), so a twin and its partner read the same line (the role-stripped control)."""
    atoms = sorted({(f.atom, f.score) for line in lines for f in line.fillers}, key=lambda x: x[0])
    seen, fillers = set(), []
    for atom, score in atoms:
        if atom not in seen:
            seen.add(atom); fillers.append(Filler(atom, score))
    return [RecallLine(None, fillers)] if fillers else []


# ---------------------------------------------------------------- text


def relation_phrase(name: str) -> str:
    """`depends_on` → "depends on"; `is_a` → "is a"."""
    return name.replace("_", " ")


def statement_phrase(name: str) -> str:
    """The verb phrase of an untemplated relation in a statement: `contains_element` → "contains element", `replaces` →
    "replaces"; a one-word noun relation is a property the subject has (`charge` → "has charge")."""
    return relation_phrase(name) if "_" in name or name.endswith("s") else f"has {name}"


_SLOT = re.compile(r"\{(x|y)\}")


def _fill(template: str, x: str, y: str) -> str:
    """Fill `{x}` / `{y}` only (chemical names hold braces)."""
    return _SLOT.sub(lambda m: x if m.group(1) == "x" else y, template)


class RecallWriter:
    """Writes recalls as text (module docstring). `lexicon` is a track lexicon (`e9_tracks.TrackLexicon`: `templates`,
    `text`, `answer`); `atomic_names` / `relation_names` are the ontology's."""

    def __init__(self, relation_names: Sequence[str], atomic_names: Sequence[str], lexicon: Any, *, style: str = "statements",
                 confidence: bool = True, home: Sequence[int] | None = None, max_lines: int | None = 32) -> None:
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        self.relation_names, self.atomic_names, self.lexicon = list(relation_names), list(atomic_names), lexicon
        self.style, self.confidence, self.max_lines = style, bool(confidence), max_lines
        self.home = list(home) if home is not None else None
        self.templates = getattr(lexicon, "templates", {}) or {}

    def atom_text(self, atom: int) -> str:
        name = self.atomic_names[int(atom)]
        text = self.lexicon.text(name) if self.lexicon is not None else None
        return text or (name.split(":", 1)[1] if ":" in name else name)

    def filler_text(self, relation: int | None, atom: int) -> str:
        """The filler as the relation's answer words it (the item candidates' surface, without the leading space); a
        role-blind filler as its home relation words it."""
        if relation is None and self.home is not None and 0 <= int(atom) < len(self.home) and self.home[int(atom)] >= 0:
            relation = int(self.home[int(atom)])
        text = self.atom_text(atom)
        if relation is not None and self.relation_names[int(relation)] in self.templates:
            return self.lexicon.answer(self.relation_names[int(relation)], text).strip()
        return text

    def statement(self, subject: str, relation: int, atom: int) -> str:
        name = self.relation_names[int(relation)]
        text = self.atom_text(atom)
        spec = self.templates.get(name)
        if spec is not None and "{y}" in spec.statement:
            form = getattr(self.lexicon, "_filler", None)
            y = form(name, text) if callable(form) else text
            return _fill(spec.statement, subject, y)
        return f"{subject} {statement_phrase(name)} {text}."

    def _score(self, score: float) -> str:
        return f" ({score:.2f})" if self.confidence else ""

    def line(self, subject: str, line: RecallLine) -> list[str]:
        who = line.subject or subject
        if line.relation is None:
            if not line.fillers:
                return []
            parts = ", ".join(f"{self.filler_text(None, f.atom)}{self._score(f.score)}" for f in line.fillers)
            return [f"- {who} is associated with: {parts}"]
        out = []
        for f in line.fillers:
            if self.style == "fields":
                out.append(f"- {relation_phrase(self.relation_names[line.relation])}: {self.filler_text(line.relation, f.atom)}"
                           f"{self._score(f.score)}")
            else:
                out.append(f"- {self.statement(who, line.relation, f.atom)}{self._score(f.score)}")
        return out

    def render(self, subject: str, lines: Sequence[RecallLine], *, call: str | None = None) -> str:
        """`call` (default `recall(<subject>)`) followed by one text line per recalled filler."""
        body = [t for line in lines for t in self.line(subject, line)]
        if self.max_lines is not None:
            body = body[:self.max_lines]
        head = call if call is not None else f"recall({subject}):"
        return "\n".join([head, *body]) if body else f"{head}\n- (nothing recalled)"

    def render_blocks(self, blocks: Iterable[tuple[str, str, Sequence[RecallLine]]]) -> str:
        """Several calls in sequence (chained recall): (subject, call, lines) per block."""
        return "\n".join(self.render(subject, lines, call=call) for subject, call, lines in blocks)


# ---------------------------------------------------------------- accuracy of a read-back


def slot_accuracy(lines: Sequence[RecallLine], frame: Sequence[tuple[int, int]]) -> dict[int, float]:
    """Per relation of `frame`: the share of its gold fillers among the recalled fillers of that relation (role-blind
    lines: among every recalled filler, i.e. filler recall without the role)."""
    gold: dict[int, set[int]] = {}
    for r, f in frame:
        gold.setdefault(int(r), set()).add(int(f))
    recalled: dict[int | None, set[int]] = {}
    for line in lines:
        recalled.setdefault(line.relation, set()).update(f.atom for f in line.fillers)
    out = {}
    for r, atoms in gold.items():
        found = recalled.get(r, set()) | (recalled.get(None, set()))
        out[r] = len(atoms & found) / len(atoms)
    return out
