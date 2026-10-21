"""E14 world generator (plan §2 X1): contamination-free logic worlds in invented vocabulary, with exact answers.

Every generator is seeded and returns plain JSON-able dicts. Worlds:

- `taxonomy` (LG-B1/B3/C3/C5, PrOntoQA-style): chains of classes `C0 ⊂ C1 ⊂ … ⊂ Cd` ("Every C0 is a C1"), negative rules
  ("No C2 is a D"), entities ("Brindle is a C0"), distractor chains, per-class property edges (store load); questions
  "Brindle is a X: true or false?" at proof depth 0–5, half true (a superclass) and half false (excluded by a negative rule
  on the chain); a `counterfactual` share uses real category words with rules that contradict general knowledge.
- `composition` (LG-B5, CLUTRR-style): six invented relations with a composition table `r_a ∘ r_b = r_c`, chains of persons
  `p0 r p1 r p2 …` of length k; question: "p0 is the ___ of p_k" over the relation names; held-out table cells.
- `propositional` (LG-B4, SimpleLogic-style): Horn rules over invented adjectives, a fact set, a query atom; samplers
  `rp` (rules first, labels by forward chaining) and `lp` (labels first; rule counts matched between labels).
- `fewshot` (LG-C4): a new relation pattern (inverse, symmetric, sub-relation) shown on k example pairs; a query pair of
  new entities.
- `trees` / `lists` (LG-D): decision trees over invented categorical and numeric attributes (tests "attr is v" and
  "attr is at most t"), decision lists with first-match and a default; entity records; edits (one node changed).
- `noisy_trees` (LG-D2, pre-registration §19): LG-D-style trees whose input records are corrupted (wrong or missing
  values) at noise levels 0.1 and 0.3; the answer is the outcome on the true record.
- `fol` (LG-P): first-order problems needing backward chaining, proof by cases (a disjunctive fact) or proof by
  contradiction (modus tollens), with few or many distractor clauses; half provable, half not.

`reference.py` gives the exact answers (`closure`, tree execution, the first-order reference prover).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gl", "kr", "pl", "st", "tr", "sk",
          "sn", "gr", "fl", "bl", "th", "sh", "ch", "w", "y", "j"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ou", "ee", "oo", "ei", "au"]
CODAS = ["", "", "", "n", "m", "s", "k", "p", "t", "rk", "nd", "sk", "lp", "mp", "x", "rt"]
REAL_CLASSES = ["animal", "plant", "metal", "vehicle", "fruit", "bird", "fish", "tool", "mineral", "insect", "reptile", "tree",
                "vegetable", "liquid", "gas", "planet", "instrument", "building", "fabric", "grain"]


class Namer:
    """Seeded invented words: 2–3 syllables, not a WordNet word, unique."""

    def __init__(self, seed: int, forbidden: set[str] | None = None) -> None:
        self.rng = random.Random(f"names:{seed}")
        if forbidden is None:
            from ..tracks.common import wordnet_forbidden
            forbidden = wordnet_forbidden()
        self.forbidden = set(forbidden)
        self.used: set[str] = set()

    def word(self, syllables: int | None = None) -> str:
        for _ in range(10_000):
            n = syllables or self.rng.choice([2, 2, 3])
            w = "".join(self.rng.choice(ONSETS) + self.rng.choice(NUCLEI) + self.rng.choice(CODAS) for _ in range(n))
            if 4 <= len(w) <= 12 and w not in self.used and w not in self.forbidden:
                self.used.add(w)
                return w
        raise RuntimeError("name space exhausted")

    def name(self) -> str:
        return self.word().capitalize()


def _plural(w: str) -> str:
    return w + ("es" if w.endswith(("s", "x", "sh", "ch")) else "s")


# -- taxonomy ------------------------------------------------------------------------------------------------------

def taxonomy(seed: int, *, questions_per_depth: int = 200, max_depth: int = 5, distractor_chains: int = 4,
             properties_per_class: int = 4, counterfactual_share: float = 0.25, namer: Namer | None = None) -> dict[str, Any]:
    """One theory per question.

    - True at depth d: the query is the class reached after d sub-steps (`chain[d]`); one distractor negative rule.
    - False at depth d ≥ 1: "No chain[d-1] is a N" and the query is N (d − 1 sub-steps, then the negative rule);
      false at depth 0: the entity fact "E is not a N".
    """
    rng = random.Random(f"taxonomy:{seed}")
    namer = namer or Namer(seed)
    items = []
    for depth in range(max_depth + 1):
        for q in range(questions_per_depth):
            counterfactual = rng.random() < counterfactual_share
            label = q % 2 == 0
            pool = rng.sample(REAL_CLASSES, len(REAL_CLASSES)) if counterfactual else None
            take = (lambda: pool.pop()) if counterfactual else (lambda: namer.word())
            chain = [take() for _ in range(max_depth + 1)]
            target = take()
            entity = namer.name()
            rules = [{"kind": "sub", "a": chain[i], "b": chain[i + 1]} for i in range(max_depth)]
            facts = [{"kind": "is", "entity": entity, "cls": chain[0]}]
            if label:
                query = chain[depth]
                rules.append({"kind": "neg", "a": chain[rng.randrange(max_depth + 1)], "b": namer.word()})
            else:
                query = target
                if depth == 0:
                    facts.append({"kind": "is_not", "entity": entity, "cls": target})
                else:
                    rules.append({"kind": "neg", "a": chain[depth - 1], "b": target})
            for _ in range(distractor_chains):
                d = [namer.word() for _ in range(3)]
                rules += [{"kind": "sub", "a": d[0], "b": d[1]}, {"kind": "sub", "a": d[1], "b": d[2]}]
            properties = {c: [namer.word() for _ in range(properties_per_class)] for c in chain}
            rng.shuffle(rules)
            items.append({"world": "taxonomy", "id": f"tax-{seed}-{depth}-{q}", "depth": depth, "label": label, "entity": entity,
                          "chain": chain, "rules": rules, "facts": facts, "properties": properties, "query": query,
                          "counterfactual": counterfactual})
    return {"world": "taxonomy", "seed": seed, "items": items}


def taxonomy_rule_text(rule: dict[str, Any]) -> str:
    if rule["kind"] == "sub":
        return f"Every {rule['a']} is a {rule['b']}."
    return f"No {rule['a']} is a {rule['b']}."


def taxonomy_fact_text(fact: dict[str, Any]) -> str:
    return f"{fact['entity']} is {'not ' if fact['kind'] == 'is_not' else ''}a {fact['cls']}."


# -- composition (CLUTRR-style) ----------------------------------------------------------------------------------------

def composition(seed: int, *, items_per_k: int = 100, ks: Sequence[int] = (2, 3, 4, 5, 6), relations: int = 6,
                heldout_cells: int = 6, namer: Namer | None = None) -> dict[str, Any]:
    rng = random.Random(f"composition:{seed}")
    namer = namer or Namer(seed + 1)
    names = [namer.word() for _ in range(relations)]
    table = {(a, b): rng.randrange(relations) for a in range(relations) for b in range(relations)}
    cells = sorted(table)
    heldout = set(rng.sample(cells, heldout_cells))
    items = []
    for k in ks:
        for q in range(items_per_k):
            people = [namer.name() for _ in range(k + 1)]
            rels = [rng.randrange(relations) for _ in range(k)]
            acc = rels[0]
            used = []
            for r in rels[1:]:
                used.append((acc, r))
                acc = table[(acc, r)]
            items.append({"world": "composition", "id": f"comp-{seed}-{k}-{q}", "depth": k, "people": people,
                          "relations": [names[r] for r in rels], "answer": names[acc],
                          "uses_heldout": any(c in heldout for c in used), "options": list(names)})
    return {"world": "composition", "seed": seed, "relation_names": names,
            "table": [[names[a], names[b], names[c]] for (a, b), c in sorted(table.items())],
            "heldout_cells": [[names[a], names[b]] for a, b in sorted(heldout)], "items": items}


# -- propositional (SimpleLogic-style) ---------------------------------------------------------------------------------

def _forward(facts: set[str], rules: Sequence[tuple[tuple[str, ...], str]]) -> dict[str, int]:
    depth = {f: 0 for f in facts}
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            if head not in depth and all(b in depth for b in body):
                depth[head] = 1 + max(depth[b] for b in body)
                changed = True
    return depth


def propositional(seed: int, *, sampler: str, items_per_depth: int = 100, max_depth: int = 5, atoms: int = 16,
                  rules: int = 18, namer: Namer | None = None) -> dict[str, Any]:
    """`rp`: a theory is drawn (rule count, rules, facts, query) and its label is whatever forward chaining gives; buckets of
    (depth, label) are filled to balance — so rule count stays correlated with the label (the SimpleLogic shortcut). `lp`:
    the rule count is drawn first, then the label, then theories with that rule count are drawn until the label matches —
    so the rule count is independent of the label by construction."""
    if sampler not in {"rp", "lp"}:
        raise ValueError("sampler is rp or lp")
    rng = random.Random(f"prop:{sampler}:{seed}")
    namer = namer or Namer(seed + 2)
    need = {(d, lab): items_per_depth // 2 for d in range(1, max_depth + 1) for lab in (True, False)}
    out: list[dict[str, Any]] = []

    def draw(n_rules: int) -> tuple[list, set, str, bool, int]:
        words = [namer.word() for _ in range(atoms)]
        rs = []
        for _ in range(n_rules):
            body = tuple(rng.sample(words, rng.randint(1, 3)))
            rs.append((body, rng.choice([w for w in words if w not in body])))
        facts = set(rng.sample(words, rng.randint(1, 4)))
        derived = _forward(facts, rs)
        query = rng.choice([w for w in words if w not in facts])
        label = query in derived
        depth = derived[query] if label else rng.randint(1, max_depth)
        return rs, facts, query, label, depth

    attempts = 0
    while any(need.values()) and attempts < 2_000_000:
        attempts += 1
        if sampler == "rp":
            rs, facts, query, label, depth = draw(rng.randint(rules - 6, rules + 6))
            if need.get((depth, label), 0) <= 0:
                continue
        else:
            n_rules = rng.randint(rules - 6, rules + 6)
            want_label = rng.random() < 0.5
            for _ in range(2000):
                rs, facts, query, label, depth = draw(n_rules)
                if label == want_label and need.get((depth, label), 0) > 0:
                    break
            else:
                continue
        need[(depth, label)] -= 1
        out.append({"world": "propositional", "id": f"prop-{sampler}-{seed}-{len(out)}", "sampler": sampler, "depth": depth,
                    "label": label, "facts": sorted(facts), "rules": [[list(b), h] for b, h in rs], "query": query,
                    "rule_count": len(rs)})
    return {"world": "propositional", "seed": seed, "sampler": sampler, "items": out}


def prop_rule_text(body: Sequence[str], head: str) -> str:
    return ("If something is " + " and ".join(body) + f", then it is {head}.") if body else f"Everything is {head}."


# -- few-shot rules (LG-C4) ---------------------------------------------------------------------------------------------

def fewshot(seed: int, *, ks: Sequence[int] = (1, 3, 5, 10), queries: int = 100, namer: Namer | None = None) -> dict[str, Any]:
    """A new rule shown on k examples, then a query on new entities. Kinds: `inverse` (x p y ⇒ y q x), `symmetric`
    (x p y ⇒ y p x), `subrelation` (x p y ⇒ x q y), `chain` (x p y ∧ y p z ⇒ x q z: a join, which one mapping vector cannot
    express). Each example is (premises, conclusion); the query gives premises and asks for the conclusion's missing entity."""
    rng = random.Random(f"fewshot:{seed}")
    namer = namer or Namer(seed + 3)
    items = []
    for kind in ("inverse", "symmetric", "subrelation", "chain"):
        for k in ks:
            for q in range(queries):
                p, r = namer.word(), namer.word()
                conc = p if kind == "symmetric" else r

                def make(x: str, y: str, z: str) -> tuple[list[list[str]], list[str]]:
                    if kind == "inverse":
                        return [[x, p, y]], [y, conc, x]
                    if kind == "symmetric":
                        return [[x, p, y]], [y, conc, x]
                    if kind == "subrelation":
                        return [[x, p, y]], [x, conc, y]
                    return [[x, p, y], [y, p, z]], [x, conc, z]
                examples = [list(make(namer.name(), namer.name(), namer.name())) for _ in range(k)]
                a, b, c = namer.name(), namer.name(), namer.name()
                premises, conclusion = make(a, b, c)
                answer = conclusion[2]
                options = [answer] + [o for o in (a, b, c, namer.name()) if o != answer][:3]
                rng.shuffle(options)
                items.append({"world": "fewshot", "id": f"few-{seed}-{kind}-{k}-{q}", "kind": kind, "k": k, "relation": p,
                              "conclusion_relation": conc, "examples": examples, "premises": premises,
                              "question": conclusion[:2], "answer": answer, "options": options})
    return {"world": "fewshot", "seed": seed, "items": items}


# -- decision trees and lists --------------------------------------------------------------------------------------------

@dataclass
class Attribute:
    name: str
    values: list[str] | None      # categorical; None = numeric 0–100


def _attributes(namer: Namer, n: int, numeric_share: float, rng: random.Random) -> list[Attribute]:
    out = []
    for i in range(n):
        if rng.random() < numeric_share:
            out.append(Attribute(namer.word() + " score", None))
        else:
            out.append(Attribute(namer.word(), [namer.word() for _ in range(rng.randint(3, 5))]))
    return out


def random_tree(depth: int, attrs: Sequence[Attribute], outcomes: Sequence[str], rng: random.Random) -> dict[str, Any]:
    if depth == 0:
        return {"leaf": rng.choice(list(outcomes))}
    a = rng.choice(list(attrs))
    test = {"attr": a.name, "op": "is", "value": rng.choice(a.values)} if a.values else \
        {"attr": a.name, "op": "le", "value": rng.randint(20, 80)}
    return {"test": test, "yes": random_tree(depth - 1, attrs, outcomes, rng), "no": random_tree(depth - 1, attrs, outcomes, rng)}


def trees(seed: int, *, depths: Sequence[int] = (2, 3, 4, 6), inputs_per_depth: int = 150, list_lengths: Sequence[int] = (3, 6, 12),
          inputs_per_list: int = 100, edits: int = 200, namer: Namer | None = None) -> dict[str, Any]:
    from .reference import run_list, run_tree
    rng = random.Random(f"trees:{seed}")
    namer = namer or Namer(seed + 4)
    items = []
    for depth in depths:
        for q in range(inputs_per_depth):
            numeric = q % 2 == 1
            attrs = _attributes(namer, 8, 0.5 if numeric else 0.0, rng)
            outcomes = [f"send it to the {namer.word()} desk" for _ in range(6)]
            tree = random_tree(depth, attrs, outcomes, rng)
            record = {a.name: (rng.choice(a.values) if a.values else rng.randint(0, 100)) for a in attrs}
            leaf, path = run_tree(tree, record)
            options = [leaf] + rng.sample([o for o in outcomes if o != leaf], 3)
            rng.shuffle(options)
            items.append({"world": "trees", "id": f"tree-{seed}-{depth}-{q}", "kind": "tree", "depth": depth, "numeric": numeric,
                          "entity": namer.name(), "attributes": [a.__dict__ for a in attrs], "tree": tree, "record": record,
                          "answer": leaf, "path": path, "options": options})
    for length in list_lengths:
        for q in range(inputs_per_list):
            attrs = _attributes(namer, 8, 0.3, rng)
            outcomes = [f"send it to the {namer.word()} desk" for _ in range(length + 1)]
            rules = []
            for i in range(length):
                a = rng.choice(attrs)
                cond = {"attr": a.name, "op": "is", "value": rng.choice(a.values)} if a.values else \
                    {"attr": a.name, "op": "le", "value": rng.randint(20, 80)}
                rules.append({"if": cond, "then": outcomes[i]})
            default = outcomes[-1]
            record = {a.name: (rng.choice(a.values) if a.values else rng.randint(0, 100)) for a in attrs}
            answer, fired = run_list(rules, default, record)
            options = [answer] + rng.sample([o for o in outcomes if o != answer], 3)
            rng.shuffle(options)
            items.append({"world": "trees", "id": f"list-{seed}-{length}-{q}", "kind": "list", "depth": length, "entity": namer.name(),
                          "attributes": [a.__dict__ for a in attrs], "rules": rules, "default": default, "record": record,
                          "answer": answer, "fired": fired, "options": options})
    tree_items = [i for i in items if i["kind"] == "tree"]
    for q in range(edits):
        base = tree_items[rng.randrange(len(tree_items))]
        tree = _copy(base["tree"])
        node = tree
        for step in base["path"][:-1]:                      # change the leaf the record reaches
            node = node["yes"] if step["result"] else node["no"]
        branch = "yes" if base["path"][-1]["result"] else "no"
        old = node[branch]["leaf"]
        new = rng.choice([o for o in base["options"] if o != old])
        node[branch] = {"leaf": new}
        leaf, path = run_tree(tree, base["record"])
        items.append({**base, "id": f"edit-{seed}-{q}", "kind": "edit", "tree": tree, "answer": leaf, "path": path,
                      "before": old, "base_tree": base["tree"], "base_id": base["id"]})
    return {"world": "trees", "seed": seed, "items": items}


def noisy_trees(seed: int, *, depths: Sequence[int] = (2, 3, 4, 6), inputs_per_depth: int = 150, noise: Sequence[float] = (0.1, 0.3),
                namer: Namer | None = None) -> dict[str, Any]:
    """LG-D2 (pre-registration §19): LG-D's trees (`random_tree`, 8 attributes, half the items numeric, 6 outcomes) under their own
    seed key and namer, each with its true record and, per noise level p, a noisy record. Each attribute is corrupted with
    probability p, half wrong (another value; numeric: a new draw from 0–100) and half missing (None). The noise is coupled
    across levels (one uniform draw, kind and wrong value per attribute), so the corruptions at 0.1 are a subset of those at 0.3.
    The answer is the tree's outcome on the true record. `trees` and the tier-P sets are untouched."""
    from .reference import run_tree
    rng = random.Random(f"trees-noisy:{seed}")
    namer = namer or Namer(seed + 40)
    items = []
    for depth in depths:
        for q in range(inputs_per_depth):
            numeric = q % 2 == 1
            attrs = _attributes(namer, 8, 0.5 if numeric else 0.0, rng)
            outcomes = [f"send it to the {namer.word()} desk" for _ in range(6)]
            tree = random_tree(depth, attrs, outcomes, rng)
            record = {a.name: (rng.choice(a.values) if a.values else rng.randint(0, 100)) for a in attrs}
            leaf, path = run_tree(tree, record)
            draws = {}
            for a in attrs:
                u, kind = rng.random(), ("wrong" if rng.random() < 0.5 else "missing")
                if a.values:
                    wrong: Any = rng.choice([v for v in a.values if v != record[a.name]])
                else:
                    wrong = rng.choice([v for v in range(101) if v != record[a.name]])
                draws[a.name] = (u, kind, wrong)
            noisy = {}
            for p in noise:
                corrupted = {name: kind for name, (u, kind, _) in draws.items() if u < p}
                rec = {name: (record[name] if name not in corrupted else (draws[name][2] if corrupted[name] == "wrong" else None))
                       for name in record}
                noisy[f"{p:g}"] = {"record": rec, "corrupted": corrupted}
            items.append({"world": "trees-noisy", "id": f"ntree-{seed}-{depth}-{q}", "kind": "tree", "depth": depth, "numeric": numeric,
                          "entity": namer.name(), "attributes": [a.__dict__ for a in attrs], "tree": tree, "record": record,
                          "answer": leaf, "path": path, "outcomes": outcomes, "noisy": noisy})
    return {"world": "trees-noisy", "seed": seed, "noise": list(noise), "items": items}


def _copy(tree: dict[str, Any]) -> dict[str, Any]:
    if "leaf" in tree:
        return dict(tree)
    return {"test": dict(tree["test"]), "yes": _copy(tree["yes"]), "no": _copy(tree["no"])}


def test_text(test: dict[str, Any]) -> str:
    return f"its {test['attr']} is {test['value']}" if test["op"] == "is" else f"its {test['attr']} is at most {test['value']}"


def tree_text(tree: dict[str, Any], indent: int = 0) -> list[str]:
    pad = "  " * indent
    if "leaf" in tree:
        return [f"{pad}then {tree['leaf']}."]
    return ([f"{pad}If {test_text(tree['test'])}:"] + tree_text(tree["yes"], indent + 1) + [f"{pad}Otherwise:"]
            + tree_text(tree["no"], indent + 1))


# -- first-order problems (LG-P) -----------------------------------------------------------------------------------------

def fol(seed: int, *, depths: Sequence[int] = (1, 2, 3, 4), per_depth: int = 100, distractors: Sequence[int] = (3, 30),
        decoy: float = 0.5, decoy_depth: int = 1, decoy_fanout: int = 1, namer: Namer | None = None) -> dict[str, Any]:
    """Problems about one entity. Clauses: {"kind": "rule", "body": A, "head": B, "neg": bool} (every A is [not] a B),
    {"kind": "fact", "cls": A, "neg": bool}, {"kind": "or", "cls": [A, B]}. Kinds: `horn` (backward chaining), `cases`
    (a disjunctive fact whose two branches both reach the goal), `contra` (modus tollens: A → … → G and not G; goal not A).
    Unprovable variants: `horn` and `contra` ask about an unrelated class; `cases` drops the last rule of one branch. Decoys:
    with probability `decoy`, each class that is a rule head (or the goal) also gets a rule from an unsupported class — a
    dead end that backward chaining may try (the choice points a search policy must handle). Harder problems (pre-registration
    §15): `decoy_fanout` such draws per head, each a chain of `decoy_depth` rules (unsupported → … → head), so a dead end is
    only seen as one after `decoy_depth` steps; the defaults (1, 1) give exactly the tier-P problems. Labels come from the
    exact reference (`reference.prove`, DPLL)."""
    from .reference import prove
    rng = random.Random(f"fol:{seed}")
    namer = namer or Namer(seed + 5)
    items = []
    kinds = ("horn", "cases", "contra")
    for nd in distractors:
        for depth in depths:
            for q in range(per_depth):
                kind = kinds[q % 3]
                provable = (q // 3) % 2 == 0
                entity = namer.name()
                clauses: list[dict[str, Any]] = []
                if kind == "horn":
                    chain = [namer.word() for _ in range(depth + 1)]
                    clauses.append({"kind": "fact", "cls": chain[0], "neg": False})
                    clauses += [{"kind": "rule", "body": chain[i], "head": chain[i + 1], "neg": False} for i in range(depth)]
                    goal = {"cls": chain[-1] if provable else namer.word(), "neg": False}
                elif kind == "cases":
                    g = namer.word()
                    ca = [namer.word() for _ in range(depth)] + [g]
                    cb = [namer.word() for _ in range(depth)] + [g]
                    clauses.append({"kind": "or", "cls": [ca[0], cb[0]]})
                    for c in (ca, cb):
                        clauses += [{"kind": "rule", "body": c[i], "head": c[i + 1], "neg": False} for i in range(len(c) - 1)]
                    if not provable:
                        clauses = [c for c in clauses if not (c["kind"] == "rule" and c["body"] == cb[-2] and c["head"] == g)]
                    goal = {"cls": g, "neg": False}
                else:
                    chain = [namer.word() for _ in range(depth + 1)]
                    clauses += [{"kind": "rule", "body": chain[i], "head": chain[i + 1], "neg": False} for i in range(depth)]
                    clauses.append({"kind": "fact", "cls": chain[-1], "neg": True})
                    goal = {"cls": chain[0] if provable else namer.word(), "neg": True}
                heads = sorted({c["head"] for c in clauses if c["kind"] == "rule"} | {goal["cls"]})
                for h in heads:                                              # decoys: dead-end rules into the proof's classes
                    for _ in range(decoy_fanout):
                        if rng.random() < decoy:
                            target = h
                            for _ in range(decoy_depth):                         # a dead-end chain into h
                                body = namer.word()
                                clauses.append({"kind": "rule", "body": body, "head": target, "neg": False, "decoy": True})
                                target = body
                for _ in range(nd):
                    clauses.append({"kind": "rule", "body": namer.word(), "head": namer.word(), "neg": rng.random() < 0.2})
                rng.shuffle(clauses)
                item = {"world": "fol", "id": f"fol-{seed}-{nd}-{depth}-{q}", "kind": kind, "depth": depth, "distractors": nd,
                        "entity": entity, "clauses": clauses, "goal": goal}
                item["label"] = prove(clauses, goal)["proved"]
                items.append(item)
    return {"world": "fol", "seed": seed, "items": items}


def clause_text(c: dict[str, Any], entity: str) -> str:
    if c["kind"] == "rule":
        return f"Every {c['body']} is {'not ' if c['neg'] else ''}a {c['head']}."
    if c["kind"] == "fact":
        return f"{entity} is {'not ' if c['neg'] else ''}a {c['cls']}."
    return f"{entity} is a {c['cls'][0]} or a {c['cls'][1]}."


def goal_text(goal: dict[str, Any], entity: str) -> str:
    return f"{entity} is {'not ' if goal['neg'] else ''}a {goal['cls']}"


def batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
