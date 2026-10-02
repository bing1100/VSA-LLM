"""Enterprise-glossary track (T5 / task C7): a seeded private glossary with contamination-free names.

An invented organisation has systems, processes, teams, departments, committees, metrics, policies,
documents, projects, datasets and tools. Each term gets a pronounceable made-up name (stems checked
against a forbidden vocabulary — WordNet lemmas plus a general-text sample — and unique per term),
usually followed by a real head noun ("Brainsh Ledger", "Toulkzei Review Board"), sometimes an
acronym alias, and a frame of typed relations:

    is_a (term type) · subtype_of · area · purpose · status · cadence · tier · owned_by · depends_on
    part_of · measured_by · governed_by · replaces · uses · produces · reports_to · computed_from
    reported_in · approved_by · applies_to · describes · sponsored_by · produced_by

Documents in internal styles (glossary pages, memos, tickets, meeting notes, incident reports,
release notes, onboarding guides, chat threads, FAQs) state facts consistent with the frames. Term
mention rates follow a Zipf law (random rank per term), so the glossary has a frequent head and a
rare tail. Evaluation documents draw half of their focus terms uniformly, which over-samples the
tail so the rare and held-out strata of the E4 gate are powered (strata are reported separately).

Splits of terms:
- `train`: in training and evaluation documents;
- `heldout` (frozen, hashed; stratified by Zipf rank, never a team/department/committee): only in
  evaluation documents — every clause naming one is dropped from training documents, and
  `leakage_audit` checks that no held-out name, stem or acronym occurs in training text;
- `zeroshot` ("new terms defined only by frames"): in no document at all; nothing points to them.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from bisect import bisect_left, bisect_right
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Iterator

from .devtools import NameMaker

TYPES: dict[str, tuple[list[str], float]] = {
    # type: (head nouns, share of terms)
    "system": (["Platform", "Engine", "Gateway", "Hub", "Console", "Ledger", "Service", "Pipeline", "Vault", "Portal"], 0.20),
    "process": (["Review", "Intake", "Handoff", "Rollout", "Audit", "Sync", "Reconciliation", "Triage", "Sign-off"], 0.17),
    "metric": (["Index", "Score", "Rate", "Ratio", "Yield", "Coverage"], 0.09),
    "policy": (["Policy", "Standard", "Charter", "Protocol", "Guideline"], 0.07),
    "document": (["Playbook", "Runbook", "Brief", "Template", "Handbook"], 0.08),
    "project": (["Initiative", "Program", "Migration", "Project", "Rebuild"], 0.09),
    "dataset": (["Dataset", "Feed", "Registry", "Catalog", "Extract"], 0.10),
    "tool": (["Toolkit", "Dashboard", "Tracker", "Bot", "Plugin", "Scanner"], 0.10),
    "team": (["Team", "Squad", "Guild", "Desk", "Pod", "Crew"], 0.07),
    "department": (["Division", "Office", "Group"], 0.01),
    "committee": (["Board", "Council", "Forum", "Committee"], 0.02),
}
STRUCTURAL = ("team", "department", "committee")   # never held out: they are fillers of most frames
AREAS = {
    "finance": ["reconciles supplier invoices", "forecasts quarterly budgets", "approves capital expenses",
                "tracks cost centres", "closes the monthly books", "audits expense claims"],
    "logistics": ["routes inbound shipments", "schedules warehouse picks", "tracks pallet locations",
                  "books freight carriers", "balances regional stock", "plans delivery windows"],
    "security": ["rotates access credentials", "scans for exposed secrets", "reviews firewall changes",
                 "monitors login anomalies", "manages badge access", "classifies sensitive data"],
    "people operations": ["onboards new hires", "plans headcount", "runs performance reviews",
                          "tracks training completion", "manages leave requests", "handles relocation requests"],
    "sales": ["scores incoming leads", "prices enterprise deals", "tracks renewal risk",
              "assigns sales territories", "prepares customer quotes", "manages partner referrals"],
    "compliance": ["files regulatory reports", "tracks policy exceptions", "retains audit evidence",
                   "screens new vendors", "monitors sanctions lists", "certifies data retention"],
    "engineering": ["deploys service releases", "runs integration tests", "provisions build machines",
                    "tracks technical debt", "manages feature flags", "rolls back failed releases"],
    "customer support": ["routes support tickets", "drafts help-centre articles", "tracks response times",
                         "escalates priority cases", "collects customer feedback", "manages refund requests"],
    "procurement": ["negotiates supplier contracts", "issues purchase orders", "tracks contract renewals",
                    "evaluates supplier bids", "manages approved vendors", "consolidates spend data"],
    "data": ["deduplicates customer records", "builds nightly aggregates", "tracks data lineage",
             "publishes the metrics catalogue", "masks personal data", "validates schema changes"],
}
STATUSES = ["active", "pilot", "deprecated", "planned"]
CADENCES = ["daily", "weekly", "monthly", "quarterly"]
TIERS = ["tier 1", "tier 2", "tier 3"]
# type → [(relation, filler types, min, max)]
SCHEMA: dict[str, list[tuple[str, list[str], int, int]]] = {
    "system": [("owned_by", ["team"], 1, 1), ("depends_on", ["system", "dataset"], 0, 2), ("part_of", ["project"], 0, 1),
               ("measured_by", ["metric"], 0, 1), ("governed_by", ["policy"], 0, 1), ("replaces", ["system"], 0, 1)],
    "process": [("owned_by", ["team"], 1, 1), ("uses", ["system", "tool"], 1, 2), ("governed_by", ["policy"], 0, 1),
                ("measured_by", ["metric"], 0, 1), ("produces", ["document", "dataset"], 0, 1), ("part_of", ["project"], 0, 1)],
    "metric": [("owned_by", ["team"], 1, 1), ("computed_from", ["dataset"], 0, 1), ("reported_in", ["document"], 0, 1)],
    "policy": [("owned_by", ["team", "committee"], 1, 1), ("approved_by", ["committee"], 0, 1),
               ("applies_to", ["system", "process"], 1, 2)],
    "document": [("owned_by", ["team"], 1, 1), ("describes", ["system", "process"], 1, 1)],
    "project": [("owned_by", ["team"], 1, 1), ("sponsored_by", ["committee"], 0, 1), ("depends_on", ["project"], 0, 1)],
    "dataset": [("owned_by", ["team"], 1, 1), ("produced_by", ["system"], 0, 1), ("governed_by", ["policy"], 0, 1)],
    "tool": [("owned_by", ["team"], 1, 1), ("depends_on", ["system"], 0, 1)],
    "team": [("reports_to", ["department"], 1, 1)],
    "department": [],
    "committee": [("part_of", ["department"], 1, 1)],
}
ATTRIBUTES: dict[str, tuple[list[str], tuple[str, ...]]] = {
    "status": (STATUSES, ("system", "process", "project", "tool", "dataset", "policy")),
    "cadence": (CADENCES, ("process", "metric", "committee")),
    "tier": (TIERS, ("system", "dataset", "tool")),
}
TERM_RELATIONS = ["subtype_of", "owned_by", "depends_on", "part_of", "measured_by", "governed_by", "replaces", "uses",
                  "produces", "reports_to", "computed_from", "reported_in", "approved_by", "applies_to", "describes",
                  "sponsored_by", "produced_by"]
RELATIONS = ["is_a", "area", "purpose", "status", "cadence", "tier"] + TERM_RELATIONS
ARTICLE_TYPES = ("team", "department", "committee")


PLURALS = {"process": "processes", "policy": "policies", "committee": "committees"}


def _a(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def _display(term: dict[str, Any]) -> str:
    return f"the {term['name']}" if term["type"] in ARTICLE_TYPES else term["name"]


def _acronym(name: str) -> str:
    return "".join(word[0] for word in name.replace("-", " ").split()).upper()


def generate_glossary(*, seed: int, terms: int = 4000, zero_shot: int = 200, heldout_fraction: float = 0.10,
                      zipf: float = 1.1, acronym_fraction: float = 0.3, forbidden: set[str] | None = None) -> dict[str, Any]:
    """A deterministic private glossary (JSON-serialisable)."""
    rng = random.Random(seed)
    forbidden_lower = {w.lower() for w in (forbidden or set())}
    names = NameMaker(rng, forbidden_lower)

    def make_term(kind: str) -> dict[str, Any]:
        heads = TYPES[kind][0]
        stems = [names.stem().capitalize() for _ in range(rng.choice([1, 1, 2]))]
        bare = kind not in ARTICLE_TYPES and rng.random() < 0.15
        name = " ".join(stems) if bare and len(stems) == 2 else " ".join(stems + [rng.choice(heads)])
        aliases = [name]
        area = rng.choice(sorted(AREAS))
        return {"name": name, "type": kind, "stems": stems, "aliases": aliases, "area": area,
                "purpose": rng.choice(AREAS[area]), "attributes": {}, "relations": {}, "split": "train"}

    counts = {kind: max(2, round(terms * share)) for kind, (_, share) in TYPES.items()}
    glossary_terms = [make_term(kind) for kind, n in counts.items() for _ in range(n)]
    rng.shuffle(glossary_terms)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for term in glossary_terms:
        by_type.setdefault(term["type"], []).append(term)

    def assign(term: dict[str, Any], pool_terms: dict[str, list[dict[str, Any]]]) -> None:
        for attribute, (values, kinds) in ATTRIBUTES.items():
            if term["type"] in kinds:
                term["attributes"][attribute] = rng.choice(values)
        for relation, filler_types, low, high in SCHEMA[term["type"]]:
            pool = [t for kind in filler_types for t in pool_terms.get(kind, []) if t is not term]
            k = min(len(pool), rng.randint(low, high))
            if k:
                term["relations"][relation] = sorted(t["name"] for t in rng.sample(pool, k))
        same = [t for t in pool_terms.get(term["type"], []) if t is not term]
        if term["type"] not in ARTICLE_TYPES and same and rng.random() < 0.3:
            term["relations"]["subtype_of"] = [rng.choice(same)["name"]]

    for term in glossary_terms:
        assign(term, by_type)
    lookup = {t["name"]: t for t in glossary_terms}
    for term in glossary_terms:               # a replaced system is deprecated
        for old in term["relations"].get("replaces", []):
            lookup[old]["attributes"]["status"] = "deprecated"
    # Zipf mention weights over a random rank order.
    order = list(range(len(glossary_terms)))
    rng.shuffle(order)
    for rank, index in enumerate(order, start=1):
        glossary_terms[index]["rank"] = rank
        glossary_terms[index]["weight"] = rank ** -zipf
    # Held-out terms: stratified by rank decile, structural types excluded.
    eligible = sorted((t for t in glossary_terms if t["type"] not in STRUCTURAL), key=lambda t: t["rank"])
    bins = 10
    for b in range(bins):
        chunk = eligible[b * len(eligible) // bins:(b + 1) * len(eligible) // bins]
        for term in rng.sample(chunk, max(1, round(len(chunk) * heldout_fraction))):
            term["split"] = "heldout"
    # Zero-shot terms: frames over existing terms; nothing points to them.
    zero_terms = []
    for _ in range(zero_shot):
        kind = rng.choice([k for k in TYPES if k not in STRUCTURAL])
        term = make_term(kind)
        assign(term, by_type)
        term.update({"split": "zeroshot", "rank": None, "weight": 0.0})
        zero_terms.append(term)
    _assign_acronyms(glossary_terms + zero_terms, rng, forbidden_lower, acronym_fraction)
    return {"seed": seed, "zipf": zipf, "terms": glossary_terms + zero_terms}


def _assign_acronyms(terms: list[dict[str, Any]], rng: random.Random, forbidden: set[str], fraction: float) -> None:
    """Give a `fraction` of terms an acronym alias (≥ 3 letters). The prefix-causal linker matches an
    alias at a word start without checking the right boundary ("SKE" would link inside "skeleton"),
    so an acronym must not be a prefix of any forbidden word, any stem or another acronym."""
    words = sorted(forbidden | {stem.lower() for t in terms for stem in t["stems"]})

    def prefix_of_word(a: str) -> bool:
        i = bisect_left(words, a)
        return i < len(words) and words[i].startswith(a)

    chosen: list[str] = []
    for term in terms:
        acronym = _acronym(term["name"]).lower()
        if len(acronym) < 3 or rng.random() >= fraction or prefix_of_word(acronym):
            continue
        if any(c.startswith(acronym) or acronym.startswith(c) for c in chosen):
            continue
        chosen.append(acronym)
        term["aliases"].append(acronym.upper())


# -- frames ----------------------------------------------------------------------------------------

def term_frame(term: dict[str, Any]) -> list[tuple[str, str]]:
    """(relation, atomic name) edges of one term."""
    edges = [("is_a", f"type:{term['type']}"), ("area", f"area:{term['area']}"), ("purpose", f"purpose:{term['purpose']}")]
    edges += [(attribute, f"{attribute}:{value}") for attribute, value in sorted(term["attributes"].items())]
    edges += [(relation, f"term:{filler}") for relation in TERM_RELATIONS for filler in term["relations"].get(relation, [])]
    return edges


def term_facts(term: dict[str, Any]) -> dict[str, list[str]]:
    """Relation → readable filler names (team-like fillers with their article)."""
    facts: dict[str, list[str]] = {"is_a": [term["type"]], "area": [term["area"]], "purpose": [term["purpose"]]}
    for attribute, value in term["attributes"].items():
        facts[attribute] = [value]
    for relation, fillers in term["relations"].items():
        facts[relation] = list(fillers)
    return facts


# -- documents -------------------------------------------------------------------------------------

FACT_TEMPLATES: dict[str, list[str]] = {
    "owned_by": ["{s} is owned by {o}.", "{O} owns {s}.", "Questions about {s} go to {o}, which owns it.",
                 "{s} sits with {o}.", "Ownership of {s} lies with {o}."],
    "depends_on": ["{s} depends on {o}.", "{s} cannot run without {o}.", "{s} has a hard dependency on {o}.",
                   "If {o} is down, {s} stops working."],
    "part_of": ["{s} is part of {o}.", "{s} was delivered as part of {o}.", "{s} belongs to {o}."],
    "measured_by": ["{s} is measured by {o}.", "We track the health of {s} with {o}.", "{o} is the headline number for {s}."],
    "governed_by": ["{s} is governed by {o}.", "{o} applies to {s}.", "Changes to {s} must follow {o}."],
    "replaces": ["{s} replaces {o}.", "{s} is the successor of {o}.", "{o} is being retired in favour of {s}."],
    "uses": ["{s} uses {o}.", "During {s} the team works in {o}.", "{s} relies on {o} for most steps."],
    "produces": ["{s} produces {o}.", "The output of {s} is {o}.", "{o} is generated by {s}."],
    "reports_to": ["{s} reports to {o}.", "{s} sits within {o}.", "{O} is the parent organisation of {s}."],
    "computed_from": ["{s} is computed from {o}.", "{s} is derived from {o}.", "The input to {s} is {o}."],
    "reported_in": ["{s} is reported in {o}.", "You can find {s} in {o}.", "{o} lists {s} every cycle."],
    "approved_by": ["{s} is approved by {o}.", "{O} signs off on {s}.", "Any change to {s} needs approval from {o}."],
    "applies_to": ["{s} applies to {o}.", "{o} must comply with {s}.", "{s} covers {o}."],
    "describes": ["{s} describes {o}.", "{s} is the reference for {o}.", "For {o}, read {s}."],
    "sponsored_by": ["{s} is sponsored by {o}.", "{O} sponsors {s}.", "{s} reports progress to {o}, its sponsor."],
    "produced_by": ["{s} is produced by {o}.", "{o} writes {s}.", "{s} comes out of {o}."],
    "subtype_of": ["{s} is a kind of {o}.", "{s} is a specialised variant of {o}.", "{s} extends {o}."],
    "is_a": ["{s} is {a_o}.", "{s} is one of our {os}.", "Note that {s} is {a_o} in our catalogue."],
    "area": ["{s} belongs to the {o} area.", "{s} is {a_o} {t}.", "Within {o}, {s} is a key {t}."],
    "purpose": ["{s} {o}.", "The job of {s} is that it {o}.", "In short, {s} {o}."],
    "status": ["{s} is currently {o}.", "Status of {s}: {o}.", "{s} is {o} at the moment."],
    "cadence": ["{s} runs {o}.", "{s} happens on a {o} basis.", "We do {s} {o}."],
    "tier": ["{s} is {a_o} {t}.", "{s} is classified as {o}.", "Support level for {s}: {o}."],
}
BOILERPLATE = [
    "Please keep this page up to date.", "Reach out in the team channel if anything here is unclear.",
    "This page is reviewed every quarter.", "See the linked runbook for step-by-step instructions.",
    "Thanks to everyone who contributed.", "No customer impact is expected.", "The change has been rolled out to all regions.",
    "Next review is scheduled for the end of the month.", "Action items are tracked on the usual board.",
    "Feedback is welcome as always.", "We will revisit this after the next planning cycle.",
    "Nothing else changed this week.", "Remember to file access requests early.", "The previous notes are archived.",
    "A short recording of the walkthrough is available.", "Please do not share this page outside the company.",
    "The numbers below are preliminary and may change.", "Most of the work happened during the last sprint.",
    "If you are new here, start with the overview section.", "We are still collecting input from the regional leads.",
    "The holiday freeze applies from the middle of December.", "Budget questions should be raised during planning.",
    "Several people asked for a simpler summary, so here it is.", "The old process will stay available for a few weeks.",
    "Please add your name to the list if you want to help.", "There were no surprises in the latest review.",
    "Expect a short delay while the migration completes.", "This replaces the notes from the previous quarter.",
    "Comments are open until Friday.", "The template has been simplified based on feedback.",
]
_WORDS = re.compile(r"[a-z0-9]+")
PEOPLE = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy", "mallory", "oscar"]


class _Renderer:
    """Turns facts into sentences; hides every fact that names a hidden term."""

    def __init__(self, terms: list[dict[str, Any]], hidden: frozenset[str], rng: random.Random) -> None:
        self.lookup = {t["name"]: t for t in terms}
        self.hidden, self.rng = hidden, rng

    def name(self, term_name: str) -> str:
        return _display(self.lookup[term_name])

    def facts(self, term: dict[str, Any]) -> list[tuple[str, str]]:
        out = [("is_a", term["type"]), ("area", term["area"]), ("purpose", term["purpose"])]
        out += sorted(term["attributes"].items())
        out += [(relation, filler) for relation, fillers in sorted(term["relations"].items()) for filler in fillers
                if filler not in self.hidden]
        return out

    def sentence(self, term: dict[str, Any], relation: str, obj: str) -> str:
        template = self.rng.choice(FACT_TEMPLATES[relation])
        subject = _display(term)
        if relation in ("is_a", "area", "purpose", "status", "cadence", "tier"):
            o = obj
        else:
            o = self.name(obj)
        text = template.format(s=subject, o=o, O=o[:1].upper() + o[1:], t=term["type"], a_o=_a(o),
                               os=PLURALS.get(o, o + "s"))
        return text[:1].upper() + text[1:]

    def definition(self, term: dict[str, Any]) -> str:
        owner = [f for f in term["relations"].get("owned_by", []) if f not in self.hidden]
        tail = f", owned by {self.name(owner[0])}" if owner else ""
        return f"{term['name']}: {_a(term['area'])} {term['type']} that {term['purpose']}{tail}."

    def related(self, term: dict[str, Any]) -> list[str]:
        return [filler for _, fillers in sorted(term["relations"].items()) for filler in fillers if filler not in self.hidden]


def _glossary_page(term, r: _Renderer, rng) -> str:
    facts = r.facts(term)
    rng.shuffle(facts)
    lines = [f"# {term['name']}", "", r.definition(term), ""]
    lines += [r.sentence(term, rel, obj) for rel, obj in facts[:rng.randint(3, 7)]]
    if len(term["aliases"]) > 1:
        lines.append(f"Also known as {term['aliases'][1]}.")
    lines.append(rng.choice(BOILERPLATE))
    return "\n".join(lines)


def _memo(term, r: _Renderer, rng) -> str:
    owner = [f for f in term["relations"].get("owned_by", []) if f not in r.hidden]
    to = r.name(owner[0]) if owner else "all staff"
    facts = r.facts(term)
    rng.shuffle(facts)
    body = " ".join(r.sentence(term, rel, obj) for rel, obj in facts[:rng.randint(2, 4)])
    return (f"Memo\nTo: {to[:1].upper() + to[1:]}\nSubject: Update on {term['name']}\n\n"
            f"Quick update on {term['name']}. {body} {rng.choice(BOILERPLATE)}\n")


def _ticket(term, r: _Renderer, rng) -> str:
    deps = [f for f in term["relations"].get("depends_on", []) + term["relations"].get("uses", []) if f not in r.hidden]
    cause = f" The root cause looks like a problem in {r.name(rng.choice(deps))}." if deps else ""
    facts = r.facts(term)
    fact = r.sentence(term, *rng.choice(facts))
    number = rng.randint(1000, 99999)
    return (f"Ticket {number}: errors reported in {term['name']}\nComponent: {term['name']}\nPriority: "
            f"{rng.choice(['low', 'medium', 'high'])}\nDescription: Since this morning {term['name']} returns errors "
            f"for some requests.{cause} {fact}\nResolution: pending; {rng.choice(BOILERPLATE).lower()}\n")


def _meeting(term, r: _Renderer, rng, extra) -> str:
    lines = [f"Meeting notes: weekly sync on {term['name']}", "Attendees: " + ", ".join(rng.sample(PEOPLE, 3)), ""]
    for t in [term] + extra:
        facts = r.facts(t)
        lines.append(f"- {t['name']}: " + " ".join(r.sentence(t, *f) for f in rng.sample(facts, min(2, len(facts)))))
    lines.append(f"- Action items: {rng.choice(PEOPLE)} to follow up on {term['name']}.")
    return "\n".join(lines)


def _incident(term, r: _Renderer, rng) -> str:
    deps = [f for f in term["relations"].get("depends_on", []) if f not in r.hidden]
    cause = f"Root cause: an expired certificate in {r.name(deps[0])}, which {term['name']} depends on." if deps else \
        "Root cause: a configuration change that was rolled back."
    facts = r.facts(term)
    return (f"Incident report: {term['name']} degraded\nImpact: requests to {term['name']} were slow for "
            f"{rng.randint(5, 90)} minutes.\n{cause}\nContext: {r.sentence(term, *rng.choice(facts))}\n"
            f"Follow-up: {rng.choice(BOILERPLATE)}\n")


def _release(term, r: _Renderer, rng) -> str:
    facts = r.facts(term)
    items = "\n".join(f"- {r.sentence(term, *f)}" for f in rng.sample(facts, min(3, len(facts))))
    return f"Release notes: {term['name']} {rng.randint(1, 9)}.{rng.randint(0, 20)}\n{items}\n- {rng.choice(BOILERPLATE)}\n"


def _onboarding(term, r: _Renderer, rng, extra) -> str:
    names = ", ".join(t["name"] for t in [term] + extra)
    lines = [f"Onboarding guide", f"In your first weeks you will mostly work with {names}."]
    for t in [term] + extra:
        lines.append(r.definition(t))
    lines.append(rng.choice(BOILERPLATE))
    return "\n".join(lines)


def _chat(term, r: _Renderer, rng) -> str:
    a, b = rng.sample(PEOPLE, 2)
    facts = r.facts(term)
    rel, obj = rng.choice(facts)
    return (f"{a}: does anyone know about {term['name']}?\n{b}: sure. {r.sentence(term, rel, obj)}\n"
            f"{a}: thanks! and {r.sentence(term, *rng.choice(facts)).lower()[:-1]}?\n{b}: yes, exactly.\n")


def _faq(term, r: _Renderer, rng) -> str:
    facts = r.facts(term)
    qa = [f"Q: What is {term['name']}?\nA: {r.definition(term)}"]
    for rel, obj in rng.sample(facts, min(2, len(facts))):
        qa.append(f"Q: Tell me more about {term['name']}.\nA: {r.sentence(term, rel, obj)}")
    return "FAQ\n" + "\n".join(qa) + "\n"


def documents(glossary: dict[str, Any], *, split: str, seed: int, max_chars: int, glossary_pages: bool = False,
              uniform_focus: float = 0.0) -> Iterator[str]:
    """Training (`split="train"`) or evaluation documents until `max_chars` characters.

    Each document is about a focus term drawn by Zipf weight (with probability `uniform_focus`
    uniformly instead, which over-samples the tail; used for evaluation so the rare and held-out
    strata are powered) and mentions its frame neighbours. Training documents only use training
    terms, and facts naming held-out terms are dropped; evaluation documents use training and
    held-out terms with nothing hidden. Zero-shot terms never appear. `glossary_pages` first emits
    one glossary page per term (off by default: it would put every term above the rare stratum)."""
    rng = random.Random(seed * 1_000_003 + (1 if split == "train" else 2))
    terms = [t for t in glossary["terms"] if t["split"] != "zeroshot"]
    zero = frozenset(t["name"] for t in glossary["terms"] if t["split"] == "zeroshot")
    hidden = frozenset(t["name"] for t in terms if t["split"] == "heldout") | zero if split == "train" else zero
    r = _Renderer(terms, hidden, rng)
    pool = [t for t in terms if t["name"] not in hidden]
    cumulative = list(accumulate(t["weight"] for t in pool))
    lookup = {t["name"]: t for t in pool}

    def sample() -> dict[str, Any]:
        if uniform_focus and rng.random() < uniform_focus:
            return pool[rng.randrange(len(pool))]
        return pool[bisect_right(cumulative, rng.random() * cumulative[-1])]

    simple: list[Callable] = [_memo, _ticket, _incident, _release, _chat, _faq, _glossary_page]
    produced = 0
    if glossary_pages:
        for term in sorted(pool, key=lambda t: t["name"]):
            text = _glossary_page(term, r, rng)
            produced += len(text)
            yield text
            if produced >= max_chars:
                return
    while produced < max_chars:
        term = sample()
        roll = rng.random()
        if roll < 0.2:
            neighbours = [lookup[n] for n in r.related(term) if n in lookup][:2]
            extra = neighbours + [sample() for _ in range(rng.randint(0, 2))]
            extra = [t for t in extra if t is not term][:3]
            text = (_meeting if roll < 0.12 else _onboarding)(term, r, rng, extra)
        else:
            text = rng.choice(simple)(term, r, rng)
        text += "\n" + " ".join(rng.sample(BOILERPLATE, rng.randint(1, 3)))
        produced += len(text)
        yield text


def leakage_audit(glossary: dict[str, Any], train_texts: list[str]) -> dict[str, Any]:
    """No stem or acronym of a held-out or zero-shot term may occur as a whole word in training
    documents. Stems are unique per term and every name contains one, so this also excludes every
    name at a word boundary (the linker's matching rule); a stem inside a longer word
    ("flosh" in "Zurkflosh") is a different word and is not linked."""
    secret = [t for t in glossary["terms"] if t["split"] in ("heldout", "zeroshot")]
    words: set[str] = set()
    for text in train_texts:
        words.update(_WORDS.findall(text.lower()))
    leaked = []
    for term in secret:
        whole = {s.lower() for s in term["stems"]} | {a.lower() for a in term["aliases"][1:]}
        if whole & words:
            leaked.append(term["name"])
    return {"heldout_terms": sum(t["split"] == "heldout" for t in secret),
            "zeroshot_terms": sum(t["split"] == "zeroshot" for t in secret), "leaked_into_train": leaked}


def write_documents(glossary: dict[str, Any], out_dir: Path, *, seed: int, train_chars: int,
                    eval_chars: int, eval_uniform_focus: float = 0.5) -> dict[str, Any]:
    """Write `glossary.json`, `train.jsonl.gz` and `eval.jsonl.gz`; refuse to finish on leakage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "glossary.json").write_text(json.dumps(glossary, indent=1) + "\n")
    stats = {}
    texts_by_split: dict[str, list[str]] = {}
    for split, budget in (("train", train_chars), ("eval", eval_chars)):
        texts = list(documents(glossary, split=split, seed=seed, max_chars=budget,
                               uniform_focus=eval_uniform_focus if split == "eval" else 0.0))
        texts_by_split[split] = texts
        with gzip.open(out_dir / f"{split}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}) + "\n")
        stats[split] = {"documents": len(texts), "chars": sum(map(len, texts))}
    audit = leakage_audit(glossary, texts_by_split["train"])
    if audit["leaked_into_train"]:
        raise RuntimeError(f"held-out/zero-shot terms leaked into training docs: {audit['leaked_into_train'][:5]}")
    summary = {**stats, **audit, "leaked_into_train": 0}
    (out_dir / "documents_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def read_documents(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)["text"]
