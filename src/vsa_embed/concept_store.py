"""One store, three verbs (methodology M2, decision 63): a thin facade over a trained span channel's concept store.

The toolkit's tools act on one store of composed concept vectors (`FrameComposer`: a term's row is the binding of its
frame's relation–filler edges). Before this module each tool lived in its own experiment; `ConceptStore` gives them
one interface over the same composer, ontology and lexicon, and exposes each verb as a tool call (`call`):

- `read(term, role=None, *, chain=None, reverse=None)` — the recall tool (`self_query.RecallStore`): the term's frame
  read back from its stored vector (unbind + clean-up), one role, a chained two-hop recall, or a reverse lookup
  (which stores hold this relation–filler pair); `recall_text` writes it with `self_query.RecallWriter`.
- `write(term, frame)` — insert a frame: an existing entry's frame is replaced (`authoring.replace_frames`), a new
  term is appended (`SpanChannel.add_entries` / `FrameComposer.add_concepts`); no gradient, no other row changes.
  `written(frames)` does it for a block only (evaluation conditions).
- `propose(evidence, proposer=...)` — the proposal step of *learn* (and of *write*): a `Proposer` turns evidence into
  `Proposal`s (entry, relation, atom). Registered: `rule_closure` (the default until the *learn* redesign lands: E10's
  rule closure — structural rules found in the store's own pairs, kept at an AMIE-style PCA confidence, whose
  predictions are new edges) and `reader` (`read_to_learn` readers on definitions). Any other proposer is a
  `"module:function"` path (TK-L's decompose-then-verify, `vsa_embed.learn`, plugs in that way).
- `accept(proposals, test)` — the acceptance step: an `AcceptanceTest` returns one `self_test.Decision` per proposal.
  `HeldOutUtilityTest` is the LM-level held-out utility (E7's verification, `authoring.window_losses`, with a one-sided
  t-test per proposal and Holm over the call's proposals, methodology M3); `composer_self_test` wraps
  `self_test.accept_edges` for E10's learnable composers. `null_proposals` builds M3's null world for a proposal set
  (each edge's filler swapped for another filler of the same relation and type) and `false_acceptance_rate` reads the
  test's false-acceptance rate off it. `commit` writes accepted edges into the frames.

Nothing here trains; frames are integer `(relation, atom)` lists as everywhere in the repository.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from .compose import FrameSchedule

Frame = list[tuple[int, int]]


@dataclass
class Proposal:
    """A proposed edge `(entry, relation, atom)`; `edge` is the index of an existing open edge (learnable composers)."""

    entry: int
    relation: int
    atom: int
    score: float = 0.0
    source: str = ""
    edge: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.entry), int(self.relation), int(self.atom))


@dataclass
class Evidence:
    """What a proposer may read: the entries it may propose for (None: every entry with a frame), texts per entry
    (definitions, contexts) and anything a proposer needs besides (`extra`: e.g. a scorer, hidden states)."""

    entries: list[int] | None = None
    texts: dict[int, list[str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


Proposer = Callable[["ConceptStore", Evidence], list[Proposal]]
AcceptanceTest = Callable[["ConceptStore", list[Proposal]], list[Any]]


def atom_entries(ontology: dict[str, Any]) -> np.ndarray:
    """Per atomic the single entry of the concept it names (`kind:value`, value a concept name), else −1."""
    names = {str(n): i for i, n in enumerate(ontology.get("concept_names") or [])}
    concepts = ontology.get("entry_concepts") or [(e,) for e in range(int(ontology["entry_count"]))]
    entries_of: dict[int, list[int]] = defaultdict(list)
    for e, members in enumerate(concepts):
        for c in members:
            entries_of[int(c)].append(e)
    out = np.full(len(ontology["atomic_names"]), -1, dtype=np.int64)
    for a, atom in enumerate(ontology["atomic_names"]):
        concept = names.get(atom.split(":", 1)[1] if ":" in atom else atom)
        found = entries_of.get(concept, []) if concept is not None else []
        if len(found) == 1:
            out[a] = found[0]
    return out


class ConceptStore:
    """The facade (module docstring). `composer` is the store; `channel` (optional) is needed to add new terms;
    `lexicon` (a track lexicon) words recalls; `table` (an `AliasTable`) resolves surfaces."""

    def __init__(self, composer: Any, ontology: dict[str, Any], *, channel: Any = None, lexicon: Any = None,
                 table: Any = None) -> None:
        self.composer, self.ontology, self.channel, self.lexicon, self.table = composer, ontology, channel, lexicon, table
        self.relation_names = list(ontology["relation_names"])
        self.atomic_names = list(ontology["atomic_names"])
        self.relation_id = {n: i for i, n in enumerate(self.relation_names)}
        self.atomic_id = {n: i for i, n in enumerate(self.atomic_names)}
        self._recall = None
        self._atom_entry: np.ndarray | None = None

    @classmethod
    def from_run(cls, run: Any, *, lexicon: Any = None) -> "ConceptStore":
        """The store of an opened run (`e5_common.E5Run`: its composer, channel, ontology and alias table)."""
        if run.composer is None:
            raise ValueError(f"{run.path} has no composing channel: nothing to read or write")
        if lexicon is None:
            from .experiments.e9_dim3_baselines import lexicon_for_run
            lexicon, _ = lexicon_for_run(run)
        return cls(run.composer, run.ontology, channel=run.channel, lexicon=lexicon, table=run.table)

    # -- frames and names ---------------------------------------------------------------------------------------------

    @property
    def schedule(self) -> FrameSchedule:
        return self.composer.schedule

    @property
    def entry_count(self) -> int:
        return int(self.schedule.concept_count)

    def frame(self, entry: int) -> Frame:
        s = self.schedule
        lo, hi = int(s.offsets[entry]), int(s.offsets[entry + 1])
        return list(zip(s.relations[lo:hi].tolist(), s.fillers[lo:hi].tolist()))

    def edges(self) -> set[tuple[int, int, int]]:
        """Every `(entry, relation, atom)` of the store."""
        s = self.schedule
        owners = np.repeat(np.arange(s.concept_count), s.degrees.cpu().numpy())
        return set(zip(owners.tolist(), s.relations.tolist(), s.fillers.tolist()))

    @property
    def atom_entry(self) -> np.ndarray:
        if self._atom_entry is None:
            self._atom_entry = atom_entries(self.ontology)
        return self._atom_entry

    def entry_of(self, term: int | str) -> int:
        """An entry id, a concept name or a surface the alias table links."""
        if isinstance(term, (int, np.integer)):
            return int(term)
        names = {str(n).removeprefix("synthetic:"): i for i, n in enumerate(self.ontology.get("concept_names") or [])}
        concepts = self.ontology.get("entry_concepts") or []
        if term in names:
            for e, members in enumerate(concepts):
                if names[term] in members:
                    return e
        if self.table is not None:
            from .span_channel import normalize_alias
            entry = self.table.alias_to_entry.get(normalize_alias(term))
            if entry is not None:
                return int(entry)
        raise KeyError(f"unknown term {term!r}")

    def name(self, entry: int) -> str:
        names = self.ontology.get("concept_names") or []
        concepts = self.ontology.get("entry_concepts") or []
        if entry < len(concepts) and concepts[entry]:
            return str(names[concepts[entry][0]]).removeprefix("synthetic:")
        return f"entry {entry}"

    def resolve(self, frame: Iterable[Sequence[Any]]) -> Frame:
        """`(relation, atom)` pairs given by id or by name."""
        out = []
        for relation, atom in frame:
            out.append((self.relation_id[relation] if isinstance(relation, str) else int(relation),
                        self.atomic_id[atom] if isinstance(atom, str) else int(atom)))
        return out

    # -- read -------------------------------------------------------------------------------------------------------

    @property
    def recall(self) -> Any:
        """The `self_query.RecallStore` of the current frames (rebuilt after a write)."""
        if self._recall is None:
            from .self_query import RecallStore
            self._recall = RecallStore(self.composer)
        return self._recall

    def read(self, term: int | str, role: int | str | None = None, *, chain: Sequence[int | str] | None = None,
             reverse: tuple[int | str, int | str] | None = None, k: int = 1, cleanup: str = "typed") -> Any:
        """`self_query.Recall` of one call: the term's whole frame (slot-aware: its own relations), one `role` (top-k),
        a `chain` (r1, r2) through the entry the first filler names, or a `reverse` (relation, atom) lookup over every
        stored entry (top-k holders; `term` is then ignored)."""
        from .self_query import Recall, RecallLine, Filler
        store = self.recall
        if reverse is not None:
            relation, atom = self.resolve([reverse])[0]
            hits = store.reverse(relation, atom, store.entry_vectors(), k=k)
            lines = [RecallLine(relation, [Filler(atom, score)], subject=self.name(e)) for e, score in hits]
            return Recall("reverse", lines, call=f"lookup({self.relation_names[relation]}, {self.atomic_names[atom]})",
                          method=store.method, cleanup=cleanup, meta={"holders": [e for e, _ in hits]})
        entry = self.entry_of(term)
        vector = store.entry_vectors()[entry]
        if chain is not None:
            first, second = (self.relation_id[r] if isinstance(r, str) else int(r) for r in chain)
            lines, meta = store.chain(vector, first, second, self.atom_entry, cleanup=cleanup)
            if len(lines) > 1 and meta["bridge"] >= 0:
                lines[1].subject = self.name(meta["bridge"])
            return Recall("chain", lines, call=f"recall({self.name(entry)}, {self.relation_names[first]} → "
                          f"{self.relation_names[second]})", method=store.method, cleanup=cleanup, meta=meta)
        if role is not None:
            relation = self.relation_id[role] if isinstance(role, str) else int(role)
            return Recall("role", [store.decode_role(vector, relation, k=k, cleanup=cleanup)],
                          call=f"recall({self.name(entry)}, {self.relation_names[relation]})", method=store.method, cleanup=cleanup)
        slots = [r for r, _ in self.frame(entry)]
        return Recall("frame", store.decode_slots(vector, slots, cleanup=cleanup), call=f"recall({self.name(entry)})",
                      method=store.method, cleanup=cleanup)

    def recall_text(self, recall: Any, subject: str, *, confidence: bool = True) -> str:
        from .self_query import RecallWriter
        writer = RecallWriter(self.relation_names, self.atomic_names, self.lexicon, confidence=confidence)
        return writer.render(subject, recall.lines, call=recall.call + ":")

    # -- write ------------------------------------------------------------------------------------------------------

    def write(self, term: int | str | None, frame: Iterable[Sequence[Any]]) -> int:
        """Write a frame: an existing entry's frame is replaced; `term=None` appends a new entry (needs the channel).
        Returns the entry id. No gradient; every other entry's frame is unchanged."""
        from .authoring import replace_frames
        frame = self.resolve(frame)
        self._recall = None
        if term is None:
            if self.channel is None:
                raise ValueError("a new term needs the channel (SpanChannel.add_entries)")
            return int(self.channel.add_entries(1, [frame])[0])
        entry = self.entry_of(term)
        self.composer.set_schedule(replace_frames(self.schedule, {entry: frame}))
        return entry

    @contextlib.contextmanager
    def written(self, frames: dict[int, Frame | None]) -> Iterator[None]:
        """Within the block the given entries read these frames (None or [] = no row: the channel's
        `skip_empty_frames`); restored exactly afterwards."""
        from .authoring import replace_frames
        original = self.schedule
        skip = getattr(self.channel, "skip_empty_frames", None)
        self.composer.set_schedule(replace_frames(original, {int(e): list(f or []) for e, f in frames.items()}))
        if self.channel is not None:
            self.channel.skip_empty_frames = True
        self._recall = None
        try:
            yield
        finally:
            self.composer.set_schedule(original)
            if self.channel is not None:
                self.channel.skip_empty_frames = skip
            self._recall = None

    def commit(self, proposals: Iterable[Proposal]) -> int:
        """Add proposals' edges to their entries' frames (duplicates skipped); returns the edges added."""
        from .authoring import replace_frames
        by_entry: dict[int, Frame] = defaultdict(list)
        for p in proposals:
            by_entry[int(p.entry)].append((int(p.relation), int(p.atom)))
        frames, added = {}, 0
        for entry, edges in by_entry.items():
            frame = self.frame(entry)
            new = [e for e in dict.fromkeys(edges) if e not in set(frame)]
            if new:
                frames[entry] = frame + new
                added += len(new)
        if frames:
            self.composer.set_schedule(replace_frames(self.schedule, frames))
            self._recall = None
        return added

    # -- propose and accept ---------------------------------------------------------------------------------------------

    def propose(self, evidence: Evidence | None = None, proposer: str | Proposer = "rule_closure", **options: Any) -> list[Proposal]:
        function = resolve_proposer(proposer)
        evidence = evidence or Evidence()
        return function(self, evidence, **options) if options else function(self, evidence)

    def accept(self, proposals: list[Proposal], test: AcceptanceTest) -> list[Any]:
        return test(self, proposals)

    # -- tool calls (meta layer) -----------------------------------------------------------------------------------

    TOOLS = {
        "read": "read(term, role=None, chain=None, reverse=None): what the store holds about a term, as statements",
        "write": "write(term, frame): store a frame [[relation, filler], ...] for a term (term=None: a new term)",
        "propose": "propose(entries=None, proposer='rule_closure'): candidate new edges",
        "accept": "accept(proposals, test): the edges that pass the acceptance test",
    }

    def call(self, name: str, **arguments: Any) -> str:
        """One verb as a tool call: text out (statements with confidences; proposals and decisions one per line)."""
        if name == "read":
            reverse = arguments.get("reverse")
            recall = self.read(arguments.get("term", 0) if reverse is None else 0, arguments.get("role"),
                               chain=arguments.get("chain"), reverse=reverse, k=int(arguments.get("k", 1 if reverse is None else 5)))
            subject = self.atomic_names[recall.lines[0].fillers[0].atom] if reverse is not None and recall.lines else \
                (self.name(self.entry_of(arguments["term"])) if reverse is None else "")
            return self.recall_text(recall, subject)
        if name == "write":
            entry = self.write(arguments.get("term"), arguments["frame"])
            return f"write({self.name(entry)}): {len(self.frame(entry))} edges"
        if name == "propose":
            proposals = self.propose(Evidence(entries=arguments.get("entries")), arguments.get("proposer", "rule_closure"))
            return "\n".join(self.describe(p) for p in proposals) or "(no proposal)"
        if name == "accept":
            decisions = self.accept(arguments["proposals"], arguments["test"])
            return "\n".join(f"{'accept' if d.accept else 'reject'} {self.describe(d.key)} (utility {d.mean:+.4f})"
                             for d in decisions) or "(nothing tested)"
        raise KeyError(f"unknown tool {name!r}; tools: {', '.join(self.TOOLS)}")

    def describe(self, p: Proposal) -> str:
        return f"{self.name(p.entry)} {self.relation_names[p.relation]} {self.atomic_names[p.atom]} ({p.score:.3f}, {p.source})"


# ---------------------------------------------------------------- proposers


def _filler_pools(store: ConceptStore) -> dict[int, Counter]:
    pools: dict[int, Counter] = defaultdict(Counter)
    s = store.schedule
    for r, a in zip(s.relations.tolist(), s.fillers.tolist()):
        pools[r][a] += 1
    return pools


def _kind(store: ConceptStore, atom: int) -> str:
    """An atom's type: the lexicon's (`atom_type`, e.g. `term:system`), else its name's prefix."""
    name = store.atomic_names[atom]
    typed = getattr(store.lexicon, "atom_type", None)
    return typed(name) if callable(typed) else name.partition(":")[0]


def functional_relations(store: ConceptStore, share: float = 0.95) -> set[int]:
    """Relations with one filler in ≥ `share` of the frames that use them (`read_to_learn.RelationTyping`'s rule)."""
    s = store.schedule
    owners = np.repeat(np.arange(s.concept_count), s.degrees.cpu().numpy())
    per_frame = Counter(zip(owners.tolist(), s.relations.tolist()))
    frames_with = Counter(r for (_, r) in per_frame)
    single = Counter(r for (_, r), n in per_frame.items() if n == 1)
    return {r for r in frames_with if single[r] / frames_with[r] >= share}


@dataclass
class RuleSettings:
    """`rule_closure` thresholds: AMIE-style support (rule instances present in the store) and PCA confidence (present
    among the instances whose head has some edge of the rule's head relation)."""

    min_support: int = 10
    min_confidence: float = 0.6
    kinds: tuple[str, ...] = ("inverse_of", "symmetric", "transitive", "composition_of")
    max_rules: int = 64
    composition_limit: int = 200_000


def mined_rules(store: ConceptStore, settings: RuleSettings | None = None) -> list[dict[str, Any]]:
    """Horn rules over the store's own pairs (nodes: entries, and atoms that name no entry): for each head relation r,
    `inverse_of s` (s(y, x) ⇒ r(x, y)), `symmetric` (r(y, x) ⇒ r(x, y)), `transitive` (r ∘ r ⇒ r) and `composition_of s, t`
    (s ∘ t ⇒ r), with their support, PCA confidence and the body pairs they imply."""
    from .ontology_hypotheses import compose_pairs, inverse
    settings = settings or RuleSettings()
    s = store.schedule
    owners = np.repeat(np.arange(s.concept_count), s.degrees.cpu().numpy()).tolist()
    atom_entry = store.atom_entry
    node = lambda a: ("e", int(atom_entry[a])) if atom_entry[a] >= 0 else ("a", int(a))      # noqa: E731
    pairs: dict[int, set] = defaultdict(set)
    for h, r, a in zip(owners, s.relations.tolist(), s.fillers.tolist()):
        pairs[r].add((("e", h), node(a)))
    heads_with = {r: {h for h, _ in p} for r, p in pairs.items()}
    rules = []
    for r, present in pairs.items():
        bodies: list[tuple[str, tuple[int, ...], set]] = []
        if "symmetric" in settings.kinds:
            bodies.append(("symmetric", (), inverse(present)))
        if "transitive" in settings.kinds:
            bodies.append(("transitive", (), compose_pairs(present, present, limit=settings.composition_limit)))
        for other, others in pairs.items():
            if other == r:
                continue
            if "inverse_of" in settings.kinds:
                bodies.append(("inverse_of", (other,), inverse(others)))
        if "composition_of" in settings.kinds:
            for first, a in pairs.items():
                for second, b in pairs.items():
                    if r not in (first, second):
                        composed = compose_pairs(a, b, limit=settings.composition_limit)
                        if composed:
                            bodies.append(("composition_of", (first, second), composed))
        for kind, args, body in bodies:
            body = {p for p in body if p[0][0] == "e"}
            support = len(body & present)
            pca = sum(1 for p in body if p[0] in heads_with[r])
            confidence = support / pca if pca else 0.0
            if support >= settings.min_support and confidence >= settings.min_confidence:
                rules.append({"head": r, "kind": kind, "args": args, "support": support, "confidence": confidence,
                              "predicted": body - present})
    rules.sort(key=lambda x: (-x["confidence"], -x["support"]))
    return rules[:settings.max_rules]


def rule_closure(store: ConceptStore, evidence: Evidence, settings: RuleSettings | None = None) -> list[Proposal]:
    """E10's rule closure as a proposer: every mined rule's predicted pairs that are new edges of an evidence entry, of
    a filler type the head relation already takes, and not a second filler of a functional relation the entry already
    fills; scored by the best predicting rule's confidence."""
    rules = mined_rules(store, settings)
    allowed = set(int(e) for e in evidence.entries) if evidence.entries is not None else None
    atom_of_entry: dict[int, int] = {}
    for a, e in enumerate(store.atom_entry.tolist()):
        if e >= 0:
            atom_of_entry.setdefault(int(e), a)
    pools = _filler_pools(store)
    kinds = {r: {_kind(store, a) for a in pool} for r, pool in pools.items()}
    functional = functional_relations(store)
    filled = {(e, r) for e, r, _ in store.edges()}
    best: dict[tuple[int, int, int], Proposal] = {}
    for rule in rules:
        r = rule["head"]
        for (_, h), (tag, t) in rule["predicted"]:
            if allowed is not None and h not in allowed:
                continue
            atom = t if tag == "a" else atom_of_entry.get(t)
            if atom is None or _kind(store, atom) not in kinds.get(r, set()):
                continue
            if r in functional and (h, r) in filled:
                continue
            key = (h, r, atom)
            if key not in best or rule["confidence"] > best[key].score:
                name = rule["kind"] + (":" + "∘".join(store.relation_names[x] for x in rule["args"]) if rule["args"] else "")
                best[key] = Proposal(h, r, atom, float(rule["confidence"]), "rule_closure", meta={"rule": name, "support": rule["support"]})
    return sorted(best.values(), key=lambda p: (-p.score, p.key))


def reader_proposer(store: ConceptStore, evidence: Evidence, *, reader: str = "typeprior") -> list[Proposal]:
    """*write*'s proposal step: a `read_to_learn` reader on each entry's texts (definitions). Static readers need the
    store's alias table and lexicon; the model reader `linker` needs `evidence.extra["scorer"]` (`read_to_learn.Scorer`)."""
    from . import read_to_learn as rtl
    fillers = rtl.build_filler_lexicon(store.ontology, store.table, store.lexicon.text)
    typing = rtl.RelationTyping(store.ontology, store.lexicon.atom_type if hasattr(store.lexicon, "atom_type") else
                                (lambda atom: atom.partition(":")[0]))
    tasks = [rtl.ReadTask(str(e), "", text, store.name(e), store.frame(e)) for e, texts in evidence.texts.items() for text in texts]
    mentions = [rtl.mentions_of(t, fillers) for t in tasks]
    if reader == "typeprior":
        results = [rtl.read_typeprior(t, m, typing) for t, m in zip(tasks, mentions)]
    elif reader == "pattern":
        results = [rtl.read_pattern(t, m, typing, fillers) for t, m in zip(tasks, mentions)]
    elif reader == "linker":
        results, _ = rtl.read_linker(tasks, mentions, typing, evidence.extra["scorer"])
    else:
        raise ValueError(f"reader must be typeprior, pattern or linker, not {reader!r}")
    return [Proposal(int(r.concept), rel, atom, 1.0, f"reader:{reader}") for r in results for rel, atom in (r.frame or [])]


PROPOSERS: dict[str, Proposer] = {"rule_closure": rule_closure, "reader": reader_proposer}


def register_proposer(name: str, proposer: Proposer) -> None:
    PROPOSERS[name] = proposer


def resolve_proposer(spec: str | Proposer) -> Proposer:
    """A registered name, a `"module:function"` path (an external proposer, e.g. `vsa_embed.learn:propose`) or a callable."""
    if callable(spec):
        return spec
    if spec in PROPOSERS:
        return PROPOSERS[spec]
    if ":" in spec:
        module, _, name = spec.partition(":")
        return getattr(importlib.import_module(module), name)
    raise KeyError(f"unknown proposer {spec!r}; registered: {', '.join(PROPOSERS)}")


# ---------------------------------------------------------------- acceptance tests


@contextlib.contextmanager
def variant_entries(channel: Any, frames: Sequence[Frame]) -> Iterator[torch.Tensor]:
    """Within the block the channel has one extra entry per frame (`SpanChannel.add_entries`); restored exactly after."""
    composer = channel.composer
    saved = (channel.entry_count, composer.frame_offsets, composer.frame_relations, composer.frame_fillers,
             getattr(composer, "delta", None))
    ids = channel.add_entries(len(frames), [list(f) for f in frames])
    try:
        yield ids
    finally:
        channel.entry_count = saved[0]
        composer.frame_offsets, composer.frame_relations, composer.frame_fillers = saved[1:4]
        if saved[4] is not None:
            composer.delta = saved[4]


def validation_windows(corpus: Any, entries: Iterable[int], *, per_entry: int = 8, length: int = 128, after: int = 8,
                       min_subtokens: int = 2, seed: int = 0) -> dict[int, list[Any]]:
    """Up to `per_entry` held-out windows per entry (`authoring.ValidationWindow`): `length` tokens of `corpus` ending
    `after` tokens past a seeded choice of the entry's linked occurrences (one per document region; spans fully inside)."""
    from .authoring import ValidationWindow
    spans = corpus.spans
    wanted = np.asarray(sorted(set(int(e) for e in entries)), dtype=np.int64)
    keep = np.flatnonzero((spans["length"] >= min_subtokens) & np.isin(spans["entry"], wanted))
    by_entry: dict[int, list[int]] = defaultdict(list)
    for i in keep.tolist():
        by_entry[int(spans["entry"][i])].append(i)
    rng = random.Random(seed)
    out: dict[int, list[Any]] = {}
    for entry in wanted.tolist():
        candidates = by_entry.get(entry, [])
        rng.shuffle(candidates)
        windows = []
        for i in candidates:
            end = int(spans["end"][i])
            start = end + after + 1 - length
            if start < 0 or end + after + 1 > len(corpus) or int(spans["start"][i]) < start:
                continue
            ids, local = corpus.window(start, length, min_subtokens=min_subtokens)
            match = np.flatnonzero((local["end"] == end - start) & (local["entry"] == entry))
            if not match.size:
                continue
            j = int(match[0])
            windows.append(ValidationWindow(ids, {k: np.asarray(local[k]) for k in ("start", "end", "inject", "entry", "length", "confidence")},
                                            entry, int(local["start"][j]), int(local["end"][j])))
            if len(windows) >= per_entry:
                break
        if windows:
            out[entry] = windows
    return out


@dataclass
class HeldOutUtilityTest:
    """LM-level held-out utility (methodology M3; E7's exact utility, `authoring.window_losses`): a proposal's utility on
    one validation window of its entry is the loss over the `after` tokens following the entry's span with its current
    frame minus the loss with the edge added (positive = the edge helps); the one-sided t-test p of a mean utility > 0
    over the windows is Holm-adjusted over every proposal of the call (null proposals included when they are tested in
    the same call), and a proposal is accepted iff its adjusted p < `alpha`. Entries with fewer than `min_windows`
    windows are not testable (rejected, reason recorded)."""

    lm: Any
    windows: dict[int, list[Any]]
    device: torch.device | str = "cpu"
    after: int = 8
    batch: int = 16
    alpha: float = 0.05
    min_windows: int = 3
    forwarded: int = 0

    def utilities(self, store: ConceptStore, proposals: list[Proposal]) -> list[np.ndarray | None]:
        from .authoring import window_losses
        device = torch.device(self.device)
        testable = [i for i, p in enumerate(proposals) if len(self.windows.get(int(p.entry), ())) >= self.min_windows]
        entries = sorted({int(proposals[i].entry) for i in testable})
        flat = [(e, w) for e in entries for w in self.windows[e]]
        base, tokens = window_losses(self.lm, [w for _, w in flat], [e for e, _ in flat], device, after=self.after, batch=self.batch)
        self.forwarded += tokens
        base_of: dict[int, np.ndarray] = defaultdict(list)
        for (e, _), value in zip(flat, base.tolist()):
            base_of[e].append(value)
        out: list[np.ndarray | None] = [None] * len(proposals)
        if not testable:
            return out
        frames = [store.frame(int(proposals[i].entry)) + [(int(proposals[i].relation), int(proposals[i].atom))] for i in testable]
        with variant_entries(self.lm.channel, frames) as ids:
            rows = [(i, int(v), w) for i, v in zip(testable, ids.tolist()) for w in self.windows[int(proposals[i].entry)]]
            losses, tokens = window_losses(self.lm, [w for _, _, w in rows], [v for _, v, _ in rows], device, after=self.after,
                                           batch=self.batch)
        self.forwarded += tokens
        variant: dict[int, list[float]] = defaultdict(list)
        for (i, _, _), value in zip(rows, losses.tolist()):
            variant[i].append(value)
        for i in testable:
            out[i] = np.asarray(base_of[int(proposals[i].entry)]) - np.asarray(variant[i])
        return out

    def __call__(self, store: ConceptStore, proposals: list[Proposal]) -> list[Any]:
        from scipy import stats

        from .self_test import Decision
        from .statistics import holm_adjust
        utilities = self.utilities(store, proposals)
        p_values = []
        for u in utilities:
            if u is None or u.size < 2:
                p_values.append(1.0)
                continue
            sd = float(u.std(ddof=1))
            if sd == 0:
                p_values.append(0.0 if float(u.mean()) > 0 else 1.0)
                continue
            p_values.append(float(stats.t.sf(float(u.mean()) / (sd / math.sqrt(u.size)), u.size - 1)))
        adjusted = holm_adjust(p_values) if p_values else []
        decisions = []
        for p, u, raw, adj in zip(proposals, utilities, p_values, adjusted):
            if u is None:
                decisions.append(Decision("edge", p, float("nan"), float("-inf"), 0, False, {"reason": "too few validation windows",
                                                                                              "p": 1.0, "p_holm": 1.0}))
                continue
            mean = float(u.mean())
            sd = float(u.std(ddof=1)) if u.size > 1 else float("inf")
            lower = mean - 1.645 * sd / math.sqrt(u.size) if u.size > 1 else float("-inf")
            decisions.append(Decision("edge", p, mean, lower, int(u.size), bool(adj < self.alpha and mean > 0),
                                      {"p": raw, "p_holm": adj}))
        return decisions


def composer_self_test(data: Any, *, resamples: int = 1000, seed: int = 0, alpha: float = 0.025) -> AcceptanceTest:
    """`self_test.accept_edges` as an acceptance test, for stores over a `LearnableOntologyComposer` (E10) whose proposals
    are existing open edges (`Proposal.edge`): accepted if removing the edge worsens held-out fit (lower bound > 0)."""
    from .self_test import accept_edges

    def test(store: ConceptStore, proposals: list[Proposal]) -> list[Any]:
        if any(p.edge is None for p in proposals):
            raise ValueError("composer_self_test tests existing edges: every proposal needs its edge index")
        decisions = accept_edges(store.composer, torch.tensor([int(p.edge) for p in proposals], dtype=torch.long), data,
                                 resamples=resamples, seed=seed, alpha=alpha)
        by_edge = {int(d.key): d for d in decisions}
        out = []
        for p in proposals:
            d = by_edge.get(int(p.edge))
            if d is None:
                from .self_test import Decision
                d = Decision("edge", int(p.edge), float("nan"), float("-inf"), 0, False, {"reason": "no held-out observations"})
            d.key = p
            out.append(d)
        return out
    return test


def null_proposals(store: ConceptStore, proposals: Sequence[Proposal], *, seed: int = 0,
                   exclude: set[tuple[int, int, int]] | None = None) -> list[Proposal]:
    """M3's null world for a proposal set: each proposal's filler swapped for another filler of the same relation and type
    (frequency-weighted from the relation's fillers in the store), never one of the entry's edges or of `exclude` (known
    true edges); every null proposal is false by construction."""
    rng = random.Random(seed)
    pools = _filler_pools(store)
    exclude = exclude or set()
    current = store.edges()
    out = []
    for p in proposals:
        kind = _kind(store, p.atom)
        options = [(a, n) for a, n in pools.get(int(p.relation), Counter()).items()
                   if a != p.atom and _kind(store, a) == kind and (p.entry, p.relation, a) not in current
                   and (p.entry, p.relation, a) not in exclude]
        if not options:
            continue
        total = sum(n for _, n in options)
        pick, acc = rng.random() * total, 0.0
        for a, n in options:
            acc += n
            if acc >= pick:
                break
        out.append(Proposal(p.entry, p.relation, a, p.score, "null", meta={"of": list(p.key)}))
    return out


def false_acceptance_rate(decisions: Sequence[Any]) -> dict[str, Any]:
    """Share of null proposals accepted, with its Wilson interval."""
    from .statistics import wilson_interval
    nulls = [d for d in decisions if getattr(d.key, "source", "") == "null"]
    accepted = sum(1 for d in nulls if d.accept)
    low, high = wilson_interval(accepted, len(nulls)) if nulls else (float("nan"), float("nan"))
    return {"null_proposals": len(nulls), "accepted": accepted, "rate": accepted / len(nulls) if nulls else float("nan"),
            "ci_low": low, "ci_high": high}
