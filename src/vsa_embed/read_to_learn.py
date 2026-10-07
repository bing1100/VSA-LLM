"""Read-to-learn (E11, author decision 59): learn a new term from one reading of its definition.

The pipeline: a definition (a dictionary entry, a glossary line, a ChEBI definition, a MeSH scope note) is turned into
a *frame* — relation–filler edges over the ontology's existing atomics and relations — the frame is written into the
ontology, the span channel composes the term's row from it, and the model uses the term with no gradient step and with
the definition no longer in context. This module holds the model-independent parts; readers that need the model get a
scorer or a generator injected (`vsa_embed.experiments.e11_read_to_learn` supplies them from a trained run).

- **Definitions.** `strip_markup` (ChEBI HTML), `headword_span`, and writers that verbalize a gold frame: T5 in three
  styles (`glossary` = the generator's own wording, in the training distribution; `dictionary` = labelled fields;
  `prose` = held-out wordings that never contain a property or statement template), T4 invented compounds in ChEBI
  style (`chebi`) and in prose (`prose`). `template_overlaps` lists the item templates a definition contains.
- **Concept finder** (`FillerLexicon`): every atomic filler's readable text plus the aliases of the entry it names, matched
  longest-first at word boundaries (the track linker's reversed trie over word-ish tokens), nested and headword matches
  dropped. This is the "frozen track linker" part of the linker reader: it finds the concepts a definition names and
  does no parsing.
- **Relation typing** (`RelationTyping`): relation r admits atom a iff r's fillers in the ontology include an atom of a's
  type (type level, so a "system" term can be a filler of depends_on, uses, replaces, …); the atom-level prior is the
  relation a was most often the filler of; functional relations (one filler in ≥ 95% of the frames that use them) keep
  one edge.
- **Readers** (each returns a `ReadResult`: frame, cost, details):
  `oracle` (gold frame), `stated` (gold edges whose filler the finder sees in the text: the readable ceiling),
  `typeprior` (finder + atom-level prior; no model), `pattern` (Hearst patterns of E7 + relation-name cue phrases),
  `linker` / `linker-all` (finder + the trained model chooses each filler's relation: the relation whose single bound
  edge makes the definition most likely under the channel; `linker` keeps an edge only if it beats no row at all),
  `host` (few-shot extraction by the host's own weights, the E7 template and parser), `teacher` (Claude through the
  B13 runner, E7's `TeacherAuthor`), `random` (equal-degree random frame, the E9 rule) and `none` (no frame).
- **Metrics** (`frame_metrics`): edge precision / recall / F1 against gold, filler recall of the finder, relation
  accuracy on found gold fillers, recall of the stated gold edges.
"""

from __future__ import annotations

import html
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .span_channel import AliasTable, CausalLinker, normalize_alias

Frame = list[tuple[int, int]]           # (relation id, atom id) edges
READERS = ("oracle", "stated", "typeprior", "pattern", "linker", "linker-all", "host", "teacher", "random", "none")
MODEL_READERS = frozenset({"linker", "linker-all", "host"})          # need the trained run
STATIC_READERS = ("oracle", "stated", "typeprior", "pattern", "random", "none")
WORDISH = re.compile(r"\w+|[^\w\s]")


# -- definitions ---------------------------------------------------------------------------------------------------

_TAGS = re.compile(r"<[^>]+>")


def strip_markup(text: str) -> str:
    """ChEBI definitions carry HTML (`<i>R</i>`, `C<small><sub>8</sub></small>`): tags removed, entities decoded,
    whitespace collapsed, a missing space after a sentence end restored (`salt)as` stays: only `.X` / `,X` are fixed)."""
    text = html.unescape(_TAGS.sub("", text or ""))
    text = re.sub(r"([.;,])(?=[A-Z][a-z])", r"\1 ", text)
    return " ".join(text.split())


def headword_span(text: str, headword: str) -> tuple[int, int] | None:
    """Character span of the first case-insensitive occurrence of `headword` at word boundaries."""
    match = re.search(rf"(?<![\w]){re.escape(headword)}(?![\w])", text, re.I)
    return (match.start(), match.end()) if match else None


def _a(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# T5: the relation order every writer follows (category facts first, then attributes, then term relations).
T5_ORDER = ("is_a", "area", "purpose", "status", "cadence", "tier", "owned_by", "subtype_of", "depends_on", "part_of",
            "measured_by", "governed_by", "replaces", "uses", "produces", "reports_to", "computed_from", "reported_in",
            "approved_by", "applies_to", "describes", "sponsored_by", "produced_by")
T5_STYLES = ("glossary", "dictionary", "prose")
T5_LABELS = {"status": "Status", "cadence": "Cadence", "tier": "Support tier", "owned_by": "Owned by", "subtype_of": "Subtype of",
             "depends_on": "Depends on", "part_of": "Part of", "measured_by": "Measured by", "governed_by": "Governed by",
             "replaces": "Replaces", "uses": "Uses", "produces": "Produces", "reports_to": "Reports to",
             "computed_from": "Computed from", "reported_in": "Reported in", "approved_by": "Approved by",
             "applies_to": "Applies to", "describes": "Describes", "sponsored_by": "Sponsored by", "produced_by": "Produced by"}
# Held-out wordings: none of them is a T5 property or statement template (`tracks.glossary.TEMPLATES`) or a training
# fact template (`benchmarks.glossary.FACT_TEMPLATES`); `template_overlaps` checks every written definition.
T5_PROSE = {
    "status": "For now, {s} has {o} status.", "cadence": "{s} takes place on a {o} schedule.", "tier": "{s} gets {o} support.",
    "owned_by": "Responsibility for {s} rests with {o}.", "subtype_of": "{s} is a particular version of {o}.",
    "depends_on": "For {s} to work, {o} has to be available.", "part_of": "{s} forms one component of {o}.",
    "measured_by": "Progress on {s} is followed through {o}.", "governed_by": "{s} has to comply with the rules in {o}.",
    "replaces": "{s} took over the role previously played by {o}.", "uses": "People working on {s} lean heavily on {o}.",
    "produces": "One result of {s} is {o}.", "reports_to": "{s} falls under {o} in the organisation.",
    "computed_from": "The figures behind {s} come from {o}.", "reported_in": "{s} appears regularly in {o}.",
    "approved_by": "{O} has to give its blessing to {s}.", "applies_to": "{O} falls within the scope of {s}.",
    "describes": "{s} documents how {o} works.", "sponsored_by": "{O} backs {s} as its sponsor.",
    "produced_by": "{s} is created by {o}.",
}


def t5_definition(name: str, facts: dict[str, list[str]], style: str, rng: random.Random) -> str:
    """A T5 definition of `name` stating every fact of `facts` (relation → readable fillers; term fillers with their
    article, as the track lexicon writes them). `glossary`: the generator's first line + one training fact template per
    remaining fact; `dictionary`: labelled fields; `prose`: held-out wordings (`T5_PROSE`), facts in random order."""
    from .benchmarks.glossary import FACT_TEMPLATES, PLURALS
    kind = (facts.get("is_a") or ["term"])[0]
    area = (facts.get("area") or [None])[0]
    purpose = (facts.get("purpose") or [None])[0]
    rest = [(r, o) for r in T5_ORDER if r not in {"is_a", "area", "purpose"} for o in facts.get(r, [])]
    rest += [(r, o) for r in facts if r not in T5_ORDER for o in facts[r]]
    if style == "glossary":
        owners = facts.get("owned_by") or []
        head = f"{name}: {_a(area) + ' ' + kind if area else _a(kind)}" + (f" that {purpose}" if purpose else "") + \
               (f", owned by {owners[0]}" if owners else "") + "."
        sentences = []
        rng.shuffle(rest)
        for relation, obj in rest:
            if relation == "owned_by" and owners and obj == owners[0]:
                continue
            template = rng.choice(FACT_TEMPLATES.get(relation, ["{s} " + relation.replace("_", " ") + " {o}."]))
            text = template.format(s=name, o=obj, O=_cap(obj), t=kind, a_o=_a(obj), os=PLURALS.get(obj, obj + "s"))
            sentences.append(_cap(text))
        return " ".join([head, *sentences])
    if style == "dictionary":
        parts = [f"{name} (n.), {_a(kind)}."]
        if area:
            parts.append(f"Area: {area}.")
        if purpose:
            parts.append(f"Purpose: {purpose}.")
        grouped: dict[str, list[str]] = defaultdict(list)
        for relation, obj in rest:
            grouped[relation].append(obj)
        parts += [f"{T5_LABELS.get(r, r.replace('_', ' ').capitalize())}: {'; '.join(objs)}." for r, objs in grouped.items()]
        return " ".join(parts)
    if style == "prose":
        first = f"In our organisation, {name} is {_a(kind)}" + (f" in the {area} domain" if area else "") + \
                (f" whose main job is that it {purpose}" if purpose else "") + "."
        rng.shuffle(rest)
        sentences = [_cap(T5_PROSE.get(r, "{s} is linked to {o}.").format(s=name, o=o, O=_cap(o))) for r, o in rest]
        return " ".join([first, *sentences])
    raise ValueError(f"unknown T5 style {style!r}; known: {T5_STYLES}")


T4_STYLES = ("chebi", "prose")
# `chebi`: the PubChem-style relation sentences of the T4 training entry texts (`tracks.chemistry.entry_text`);
# `prose`: other wordings of the same relations.
T4_PHRASES = {
    "has_parent_hydride": ("It derives from a hydride of {o}.", "Formally it can be traced back to the hydride {o}."),
    "is_conjugate_acid_of": ("It is a conjugate acid of {o}.", "Losing a proton turns it into {o}."),
    "is_conjugate_base_of": ("It is a conjugate base of {o}.", "Taking up a proton turns it into {o}."),
    "is_tautomer_of": ("It is a tautomer of {o}.", "It interconverts with {o} by moving a proton."),
    "is_enantiomer_of": ("It is an enantiomer of {o}.", "Its mirror-image form is {o}."),
    "has_part": ("It has {o} as a part.", "One of its components is {o}."),
    "is_substituent_group_from": ("It is a substituent group from {o}.", "As a group it is obtained from {o}."),
}


def t4_definition(name: str, facts: dict[str, list[str]], style: str, rng: random.Random) -> str:
    """An invented compound's definition from its frame: `chebi` (the ChEBI / PubChem phrasing of the training entry
    texts) or `prose` (other wordings). Element, charge and branch facts are never stated (ChEBI definitions rarely do)."""
    if style not in T4_STYLES:
        raise ValueError(f"unknown T4 style {style!r}; known: {T4_STYLES}")
    classes = facts.get("is_a") or []
    parents = facts.get("has_functional_parent") or []
    roles = facts.get("has_role") or []
    which = 0 if style == "chebi" else 1
    others = [T4_PHRASES[r][which].format(o=o) for r in T4_PHRASES for o in facts.get(r, [])]
    if style == "chebi":
        head = f"{name}: A member of the class of {classes[0]}" if classes else f"{name}: A chemical entity"
        head += f" that is functionally related to {_join(parents)}." if parents else "."
        parts = [head]
        if len(classes) > 1:
            parts.append(f"It is also a member of {_join(classes[1:])}.")
        if roles:
            parts.append(f"It has a role as {_join([_a(r) for r in roles])}.")
        return " ".join(parts + others)
    parts = [f"{name} is a compound" + (f" belonging to the {classes[0]}" if classes else "") +
             (f", derived from {_join(parents)}" if parents else "") + "."]
    if len(classes) > 1:
        parts.append(f"Chemists also count it among the {_join(classes[1:])}.")
    if roles:
        parts.append(f"In practice it acts as {_join([_a(r) for r in roles])}.")
    rng.shuffle(others)
    return " ".join(parts + others)


def template_overlaps(text: str, headword: str, templates: Iterable[str]) -> list[str]:
    """Item templates (`{x}` = the headword) whose instantiated text occurs in the definition (case-insensitive)."""
    lowered = " ".join(text.lower().split())
    return sorted({t for t in templates if " ".join(t.format(x=headword).lower().split()) in lowered})


@dataclass
class Definition:
    concept: str
    style: str
    text: str
    headword: str
    source: str
    licence: str
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.span is None:
            self.span = headword_span(self.text, self.headword)


# -- concept finder ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Mention:
    atoms: tuple[int, ...]              # atoms sharing the matched surface
    start: int
    end: int
    surface: str


class FillerLexicon:
    """Readable surfaces of atomic fillers → atom ids, matched in text longest-first at word boundaries."""

    def __init__(self, pairs: Iterable[tuple[str, int]], atom_names: Sequence[str], *, min_chars: int = 3) -> None:
        kept = [(s, int(a)) for s, a in pairs if s and len(normalize_alias(s)) >= min_chars and re.search(r"[A-Za-z]", s)]
        self.table = AliasTable.from_pairs(kept)
        self.linker = CausalLinker(self.table, min_subtokens=1)
        self.atom_names = list(atom_names)
        self.surfaces: dict[str, tuple[int, ...]] = {alias: self.table.entry_concepts[e] for alias, e in self.table.alias_to_entry.items()}

    def __len__(self) -> int:
        return len(self.surfaces)

    def find(self, text: str, exclude: Sequence[tuple[int, int]] = ()) -> list[Mention]:
        """Non-nested matches in text order; a match overlapping an `exclude` span (the headword) is dropped."""
        offsets = [(m.start(), m.end()) for m in WORDISH.finditer(text)]
        found = []
        for span in self.linker.link(text, offsets):
            start, end = offsets[span.start_token][0], offsets[span.end_token][1]
            found.append(Mention(tuple(self.table.entry_concepts[span.entry]), start, end, text[start:end]))
        found.sort(key=lambda m: (-(m.end - m.start), m.start))
        kept: list[Mention] = []
        for m in found:
            if any(m.start < e and s < m.end for s, e in exclude) or any(m.start < k.end and k.start < m.end for k in kept):
                continue
            kept.append(m)
        return sorted(kept, key=lambda m: m.start)

    def resolve(self, text: str) -> tuple[int, ...]:
        """Atoms a filler string names: an exact surface, else the longest surface found inside it."""
        key = normalize_alias(text.strip(" .,;:'\"`"))
        if key in self.surfaces:
            return self.surfaces[key]
        if key.startswith(("the ", "a ", "an ")) and key.split(" ", 1)[1] in self.surfaces:
            return self.surfaces[key.split(" ", 1)[1]]
        inside = self.find(text)
        return max(inside, key=lambda m: m.end - m.start).atoms if inside else ()


def build_filler_lexicon(ontology: dict[str, Any], table: AliasTable | None, text_of: Callable[[str], str | None], *,
                         exclude_kinds: Iterable[str] = (), min_chars: int = 3) -> FillerLexicon:
    """Surfaces of every atom: its readable text (`text_of`, the track lexicon; without a leading article too) and the
    aliases of the entry naming it (atom `kind:value` names concept `value`)."""
    excluded = set(exclude_kinds)
    atoms = list(ontology["atomic_names"])
    concept_names = ontology.get("concept_names") or []
    concept_index = {name: i for i, name in enumerate(concept_names)}
    aliases_of: dict[int, list[str]] = defaultdict(list)
    if table is not None:
        for alias, entry in table.alias_to_entry.items():
            for concept in table.entry_concepts[entry]:
                aliases_of[int(concept)].append(alias)
    pairs: list[tuple[str, int]] = []
    for i, name in enumerate(atoms):
        kind, _, value = name.partition(":")
        if kind in excluded:
            continue
        text = text_of(name)
        if text:
            pairs.append((text, i))
            stripped = re.sub(r"^(the|a|an) ", "", text, flags=re.I)
            if stripped != text:
                pairs.append((stripped, i))
        if value in concept_index:
            pairs += [(alias, i) for alias in aliases_of.get(concept_index[value], [])]
    return FillerLexicon(pairs, atoms, min_chars=min_chars)


# -- relation typing -------------------------------------------------------------------------------------------------

class RelationTyping:
    """Which relations admit which atoms (type level), the atom-level prior and the functional relations."""

    def __init__(self, ontology: dict[str, Any], atom_type: Callable[[str], str], *, functional_share: float = 0.95) -> None:
        self.relation_names = list(ontology["relation_names"])
        self.relation_id = {n: i for i, n in enumerate(self.relation_names)}
        atoms = list(ontology["atomic_names"])
        self.types = [atom_type(a) for a in atoms]
        offsets = np.asarray(ontology["offsets"], dtype=np.int64)
        relations = np.asarray(ontology["relations"], dtype=np.int64)
        fillers = np.asarray(ontology["fillers"], dtype=np.int64)
        self.pools: dict[int, Counter] = defaultdict(Counter)
        type_counts: dict[str, Counter] = defaultdict(Counter)
        for r, f in zip(relations.tolist(), fillers.tolist()):
            self.pools[r][f] += 1
            type_counts[self.types[f]][r] += 1
        self.type_relations = {t: [r for r, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))] for t, c in type_counts.items()}
        self.type_prior = {t: rs[0] for t, rs in self.type_relations.items() if rs}
        owner = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
        per_frame = Counter(zip(owner.tolist(), relations.tolist()))
        frames_with: Counter = Counter(r for (_, r) in per_frame)
        single: Counter = Counter(r for (_, r), n in per_frame.items() if n == 1)
        self.functional = {r for r in frames_with if single[r] / frames_with[r] >= functional_share}

    def candidates(self, atom: int) -> list[int]:
        """Relations whose fillers include an atom of this atom's type (most frequent first)."""
        return list(self.type_relations.get(self.types[atom], []))

    def prior(self, atom: int) -> int | None:
        """The relation this atom is most often the filler of; else the most frequent relation of its type."""
        counts = [(self.pools[r][atom], r) for r in self.candidates(atom) if self.pools[r][atom]]
        if counts:
            return max(counts, key=lambda c: (c[0], -c[1]))[1]
        return self.type_prior.get(self.types[atom])

    def frequency(self, atom: int) -> int:
        return sum(self.pools[r][atom] for r in self.candidates(atom))

    def admits(self, relation: int, atom: int) -> bool:
        return relation in self.candidates(atom)


def functional_dedupe(edges: Sequence[tuple[int, int, float]], functional: set[int]) -> Frame:
    """(relation, atom, score) → a frame: duplicates removed, one edge (the best score) per functional relation."""
    best: dict[Any, tuple[int, int, float]] = {}
    for r, a, s in edges:
        key = r if r in functional else (r, a)
        if key not in best or s > best[key][2]:
            best[key] = (r, a, s)
    ordered = sorted(best.values(), key=lambda e: (-e[2], e[0], e[1]))
    return [(r, a) for r, a, _ in ordered]


# -- readers ------------------------------------------------------------------------------------------------------------

@dataclass
class ReadTask:
    concept: str
    style: str
    text: str
    headword: str
    gold: Frame
    own_atom: int | None = None
    random_frame: Frame | None = None
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.span is None:
            self.span = headword_span(self.text, self.headword)

    def exclude(self) -> list[tuple[int, int]]:
        """Every occurrence of the headword (the term must not be read as its own filler)."""
        return [(m.start(), m.end()) for m in re.finditer(rf"(?<![\w]){re.escape(self.headword)}(?![\w])", self.text, re.I)]


@dataclass
class ReadResult:
    concept: str
    style: str
    reader: str
    frame: Frame | None                      # None: no frame (no row)
    cost: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


def mentions_of(task: ReadTask, lexicon: FillerLexicon) -> list[Mention]:
    mentions = lexicon.find(task.text, task.exclude())
    if task.own_atom is not None:
        mentions = [Mention(tuple(a for a in m.atoms if a != task.own_atom), m.start, m.end, m.surface) for m in mentions]
        mentions = [m for m in mentions if m.atoms]
    return mentions


def read_oracle(task: ReadTask) -> ReadResult:
    return ReadResult(task.concept, task.style, "oracle", list(task.gold) or None)


def read_stated(task: ReadTask, mentions: Sequence[Mention]) -> ReadResult:
    seen = {a for m in mentions for a in m.atoms}
    frame = [(r, a) for r, a in task.gold if a in seen]
    return ReadResult(task.concept, task.style, "stated", frame or None)


def read_none(task: ReadTask) -> ReadResult:
    return ReadResult(task.concept, task.style, "none", None)


def read_random(task: ReadTask) -> ReadResult:
    return ReadResult(task.concept, task.style, "random", list(task.random_frame or []) or None)


def _pick_atom(atoms: Sequence[int], typing: RelationTyping, relation: int | None = None) -> int | None:
    usable = [a for a in atoms if relation is None or typing.admits(relation, a)]
    return max(usable, key=lambda a: (typing.frequency(a), -a)) if usable else None


def read_typeprior(task: ReadTask, mentions: Sequence[Mention], typing: RelationTyping) -> ReadResult:
    """No model: each mention's atom with the relation it is most often the filler of."""
    edges = []
    for rank, m in enumerate(mentions):
        atom = _pick_atom(m.atoms, typing)
        relation = typing.prior(atom) if atom is not None else None
        if relation is not None:
            edges.append((relation, atom, -float(rank)))           # earlier mentions win a functional slot
    frame = functional_dedupe(edges, typing.functional)
    return ReadResult(task.concept, task.style, "typeprior", frame or None, details={"mentions": len(mentions)})


# Generic Hearst relations (`authoring_baselines.GENERIC_RELATIONS`) → the first track relation that exists.
GENERIC_TO_TRACK = {"is_a": ("is_a", "subtype_of", "parent", "hypernym", "instance_hypernym"),
                    "part_of": ("part_of", "part_holonym"), "has_part": ("has_part", "part_meronym"),
                    "member_of": ("member_of", "member_holonym"), "made_of": ("made_of", "substance_meronym")}


def cue_phrase(relation: str) -> str | None:
    """A relation's surface cue from its name alone: `owned_by` → "owned by", `has_functional_parent` → "functional
    parent", `is_conjugate_acid_of` → "conjugate acid of"; `is_a` has none (Hearst patterns cover it)."""
    words = relation.split("_")
    if words[:1] in (["is"], ["has"]):
        words = words[1:]
    phrase = " ".join(words).strip()
    return phrase if phrase and phrase not in {"a", "an"} else None


def read_pattern(task: ReadTask, mentions: Sequence[Mention], typing: RelationTyping, lexicon: FillerLexicon, *,
                 window: int = 60) -> ReadResult:
    """Hearst patterns (E7) for is-a / part-whole edges of the headword, then a relation-name cue phrase in the same
    sentence within `window` characters before a mention (nearest cue wins); mentions without a cue write nothing."""
    from .authoring_baselines import hearst_extract
    edges: list[tuple[int, int, float]] = []
    head = normalize_alias(task.headword)
    surfaces = lexicon.surfaces
    for _, generic, filler in hearst_extract(task.text, {head: True}, lambda key: surfaces.get(key)):
        names = [n for n in GENERIC_TO_TRACK.get(generic, ()) if n in typing.relation_id]
        atoms = [a for a in surfaces.get(filler, ()) if a != task.own_atom]
        for name in names:
            atom = _pick_atom(atoms, typing, typing.relation_id[name])
            if atom is not None:
                edges.append((typing.relation_id[name], atom, 1.0)); break
    claimed = {a for _, a, _ in edges}
    cues = [(r, re.compile(rf"(?<![\w]){re.escape(c)}(?![\w])", re.I)) for r, name in enumerate(typing.relation_names)
            if (c := cue_phrase(name))]
    for m in mentions:
        if claimed & set(m.atoms):
            continue
        sentence_start = max(task.text.rfind(". ", 0, m.start) + 1, m.start - window, 0)
        before = task.text[sentence_start:m.start]
        best = None
        for r, pattern in cues:
            for hit in pattern.finditer(before):
                atom = _pick_atom(m.atoms, typing, r)
                if atom is not None and (best is None or hit.end() > best[2]):
                    best = (r, atom, hit.end())
        if best is not None:
            edges.append((best[0], best[1], float(best[2]) / 1000.0))
    frame = functional_dedupe(edges, typing.functional)
    return ReadResult(task.concept, task.style, "pattern", frame or None)


# Scorer for the linker reader: per task, the summed log-probability of the definition's tokens after the headword,
# once per variant frame (None = no row); returns one array per task, aligned with the variants.
Scorer = Callable[[Sequence[ReadTask], Sequence[Sequence[Frame | None]]], list[np.ndarray]]


def linker_candidates(task: ReadTask, mentions: Sequence[Mention], typing: RelationTyping) -> list[tuple[int, int, int]]:
    """(mention index, relation, atom) candidates: every atom of a mention × every relation that admits its type."""
    return [(i, r, a) for i, m in enumerate(mentions) for a in m.atoms for r in typing.candidates(a)]


def read_linker(tasks: Sequence[ReadTask], mentions: Sequence[Sequence[Mention]], typing: RelationTyping, scorer: Scorer, *,
                margin: float = 0.0, token_count: Callable[[str], int] | None = None) -> tuple[list[ReadResult], list[ReadResult]]:
    """The self-reflective reader. Per mention, the model scores every (relation, atom) single-edge frame by the
    definition's log-likelihood after the headword, minus that with no row; the best relation per mention is kept
    (`linker`) iff its gain exceeds `margin` (the self-test), and kept regardless by `linker-all`. Functional relations
    keep the best-scoring edge. Returns (linker results, linker-all results)."""
    candidates = [linker_candidates(t, m, typing) for t, m in zip(tasks, mentions)]
    variants = [[None] + [[(r, a)] for _, r, a in c] for c in candidates]
    started = time.monotonic()
    scores = scorer(tasks, variants) if any(len(v) > 1 for v in variants) else [np.zeros(1) for _ in tasks]
    seconds = (time.monotonic() - started) / max(1, len(tasks))
    kept, every = [], []
    for task, cands, score in zip(tasks, candidates, scores):
        gains = np.asarray(score[1:], dtype=float) - float(score[0]) if len(score) > 1 else np.zeros(0)
        per_mention: dict[int, tuple[int, int, float]] = {}
        for (i, r, a), g in zip(cands, gains.tolist()):
            if i not in per_mention or g > per_mention[i][2]:
                per_mention[i] = (r, a, g)
        best = list(per_mention.values())
        tokens = token_count(task.text) if token_count else len(task.text.split())
        cost = {"forward_passes": float(len(cands) + 1), "forward_tokens": float((len(cands) + 1) * tokens), "seconds": seconds}
        details = {"candidates": len(cands), "gains": [[typing.relation_names[r], int(a), round(g, 4)] for r, a, g in best],
                   "baseline_logprob": float(score[0])}
        accepted = functional_dedupe([e for e in best if e[2] > margin], typing.functional)
        everything = functional_dedupe(best, typing.functional)
        kept.append(ReadResult(task.concept, task.style, "linker", accepted or None, dict(cost), details))
        every.append(ReadResult(task.concept, task.style, "linker-all", everything or None, dict(cost), details))
    return kept, every


def host_prompt(task: ReadTask, relations: Sequence[str], demonstrations: Sequence[dict[str, Any]]) -> str:
    """The E7 constrained few-shot template on one definition (`authoring.authoring_prompt_fewshot`): demonstrations
    carry `context` (a definition), `surface` and `edges` ((relation, readable filler) pairs)."""
    from .authoring import authoring_prompt_fewshot
    return authoring_prompt_fewshot(task.headword, task.text, relations, demonstrations)


def parse_edges(completion: str, relations: Sequence[str], lexicon: FillerLexicon, typing: RelationTyping, *,
                own_atom: int | None = None) -> tuple[Frame, dict[str, int]]:
    """`relation: filler` lines (the E7 parser) → a frame over known atoms; unresolved or ill-typed lines are counted."""
    from .authoring import cut_completion, parse_proposals
    edges: list[tuple[int, int, float]] = []
    counts = Counter()
    for rank, (relation, filler) in enumerate(parse_proposals(cut_completion(completion), relations)):
        counts["proposed"] += 1
        r = typing.relation_id[relation]
        atoms = [a for a in lexicon.resolve(filler) if a != own_atom]
        atom = _pick_atom(atoms, typing, r)
        if atom is None:
            counts["unresolved" if not atoms else "ill_typed"] += 1
            continue
        edges.append((r, atom, -float(rank)))
    return functional_dedupe(edges, typing.functional), dict(counts)


Generator = Callable[[Sequence[str]], tuple[list[str], int, int]]   # prompts → (completions, prompt tokens, generated tokens)


def read_host(tasks: Sequence[ReadTask], lexicon: FillerLexicon, typing: RelationTyping, generate: Generator,
              demonstrations: Callable[[ReadTask], Sequence[dict[str, Any]]], *, relations: Sequence[str] | None = None,
              name: str = "host") -> list[ReadResult]:
    """Prompted extraction by a language model (the run's own host weights): one greedy completion per definition."""
    relations = list(relations or typing.relation_names)
    prompts = [host_prompt(t, relations, demonstrations(t)) for t in tasks]
    started = time.monotonic()
    completions, prompt_tokens, generated = generate(prompts)
    seconds = (time.monotonic() - started) / max(1, len(tasks))
    out = []
    for task, prompt, completion in zip(tasks, prompts, completions):
        frame, counts = parse_edges(completion, relations, lexicon, typing, own_atom=task.own_atom)
        out.append(ReadResult(task.concept, task.style, name, frame or None,
                              {"prompt_tokens": prompt_tokens / max(1, len(tasks)), "generated_tokens": generated / max(1, len(tasks)),
                               "seconds": seconds}, {"completion": completion[:400], **counts}))
    return out


def read_teacher(tasks: Sequence[ReadTask], lexicon: FillerLexicon, typing: RelationTyping, teacher: Any, *,
                 batch: int = 8) -> list[ReadResult]:
    """Claude as the reader (`authoring_baselines.TeacherAuthor`: cached, model pinned, cost recorded): each definition is
    the concept's single context; fillers resolve through the concept finder."""
    items = [{"id": f"{t.concept}|{t.style}", "surface": t.headword, "contexts": [t.text]} for t in tasks]
    before = float(teacher.spent_usd)
    answers = teacher.author(items, batch=batch)
    spent = (float(teacher.spent_usd) - before) / max(1, len(tasks))
    out = []
    for task, item in zip(tasks, items):
        answer = answers.get(item["id"], {"edges": []})
        edges, counts = [], Counter()
        for rank, (relation, filler) in enumerate(answer.get("edges", [])):
            counts["proposed"] += 1
            if relation not in typing.relation_id:
                continue
            r = typing.relation_id[relation]
            atoms = [a for a in lexicon.resolve(filler) if a != task.own_atom]
            atom = _pick_atom(atoms, typing, r)
            if atom is None:
                counts["unresolved" if not atoms else "ill_typed"] += 1
                continue
            edges.append((r, atom, -float(rank)))
        frame = functional_dedupe(edges, typing.functional)
        out.append(ReadResult(task.concept, task.style, "teacher", frame or None, {"usd": spent},
                              {"error": answer.get("error"), **counts}))
    return out


# -- metrics ----------------------------------------------------------------------------------------------------------

def frame_metrics(predicted: Frame | None, gold: Frame, *, found_atoms: Iterable[int] = (),
                  ambiguous_atoms: Iterable[int] = ()) -> dict[str, float | int | None]:
    """Edge precision / recall / F1 against `gold`; `filler_recall` (gold fillers among `found_atoms`, the concept finder's
    reach); `relation_accuracy` (predicted edges whose atom is a gold filler that carry a gold relation for it), also on
    `ambiguous_atoms` only (fillers whose type admits several relations: where choosing the relation is a real choice);
    `stated_recall` (recall over the gold edges whose filler was found)."""
    pred = set(map(tuple, predicted or []))
    gold_set = set(map(tuple, gold))
    hits = len(pred & gold_set)
    precision = hits / len(pred) if pred else None
    recall = hits / len(gold_set) if gold_set else None
    f1 = (2 * precision * recall / (precision + recall) if precision and recall else 0.0) if pred and gold_set else (0.0 if gold_set else None)
    found = set(found_atoms)
    gold_atoms = {a for _, a in gold_set}
    on_gold_atoms = [(r, a) for r, a in pred if a in gold_atoms]
    ambiguous = set(ambiguous_atoms)
    on_ambiguous = [(r, a) for r, a in on_gold_atoms if a in ambiguous]
    stated = {(r, a) for r, a in gold_set if a in found}
    return {"edges": len(pred), "gold_edges": len(gold_set), "hits": hits, "precision": precision, "recall": recall, "f1": f1,
            "filler_recall": len(gold_atoms & found) / len(gold_atoms) if gold_atoms else None,
            "relation_accuracy": sum((r, a) in gold_set for r, a in on_gold_atoms) / len(on_gold_atoms) if on_gold_atoms else None,
            "ambiguous_edges": len(on_ambiguous),
            "ambiguous_relation_hits": sum((r, a) in gold_set for r, a in on_ambiguous),
            "stated_edges": len(stated), "stated_recall": len(pred & stated) / len(stated) if stated else None}
