"""E14 exact reference (plan §2 X2): the answers every VSA condition is checked against.

- `taxonomy_closure` — the entity's derived classes (with the rule path that reaches each) and excluded classes.
- `run_tree` / `run_list` — exact execution of a decision tree (with the path) and a decision list (first match, default).
- `prove` — entailment for the one-entity first-order fragment of `world.fol` (unary predicates over one constant, so the
  problem is propositional): DPLL on the knowledge base plus the negated goal. Complete and sound.
- `datalog` — semi-naive forward chaining for binary-relation rules (LG-A3): rules of the shapes
  `p(x,y) → q(x,y)`, `p(x,y) → q(y,x)`, `p(x,y) ∧ r(y,z) → q(x,z)` and `p(x,y) ∧ r(x,z) → q(y,z)`, with an optional
  negated premise `¬s(x,y)` (stratified: negated relations are base relations).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def taxonomy_closure(item: dict[str, Any]) -> dict[str, Any]:
    sub = defaultdict(list)
    neg = defaultdict(list)
    for r in item["rules"]:
        (sub if r["kind"] == "sub" else neg)[r["a"]].append(r["b"])
    start = next(f["cls"] for f in item["facts"] if f["kind"] == "is")
    paths = {start: []}
    frontier = [start]
    while frontier:
        nxt = []
        for c in frontier:
            for b in sub[c]:
                if b not in paths:
                    paths[b] = paths[c] + [("sub", c, b)]
                    nxt.append(b)
        frontier = nxt
    excluded = {}
    for c, path in paths.items():
        for b in neg[c]:
            excluded.setdefault(b, path + [("neg", c, b)])
    for f in item["facts"]:
        if f["kind"] == "is_not":
            excluded.setdefault(f["cls"], [("fact_not", item["entity"], f["cls"])])
    return {"classes": paths, "excluded": excluded}


def taxonomy_answer(item: dict[str, Any]) -> bool | None:
    c = taxonomy_closure(item)
    if item["query"] in c["classes"]:
        return True
    if item["query"] in c["excluded"]:
        return False
    return None


def holds(test: dict[str, Any], record: dict[str, Any]) -> bool:
    value = record[test["attr"]]
    return value == test["value"] if test["op"] == "is" else float(value) <= float(test["value"])


def run_tree(tree: dict[str, Any], record: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    path = []
    node = tree
    while "leaf" not in node:
        result = holds(node["test"], record)
        path.append({"test": node["test"], "value": record[node["test"]["attr"]], "result": result})
        node = node["yes"] if result else node["no"]
    return node["leaf"], path


def run_list(rules: Sequence[dict[str, Any]], default: str, record: dict[str, Any]) -> tuple[str, int | None]:
    for i, rule in enumerate(rules):
        if holds(rule["if"], record):
            return rule["then"], i
    return default, None


# -- first-order (propositional over one entity) -------------------------------------------------------------------------

def cnf(clauses: Sequence[dict[str, Any]]) -> list[list[tuple[str, bool]]]:
    out = []
    for c in clauses:
        if c["kind"] == "rule":
            out.append([(c["body"], False), (c["head"], not c["neg"])])
        elif c["kind"] == "fact":
            out.append([(c["cls"], not c["neg"])])
        elif c["kind"] == "or":
            out.append([(x, True) for x in c["cls"]])
    return out


def satisfiable(clauses: list[list[tuple[str, bool]]], assignment: dict[str, bool] | None = None) -> bool:
    assignment = dict(assignment or {})
    clauses = [list(c) for c in clauses]
    while True:                                                  # unit propagation
        changed = False
        remaining = []
        for c in clauses:
            if any(assignment.get(v) == s for v, s in c):
                continue
            open_ = [(v, s) for v, s in c if v not in assignment]
            if not open_:
                return False
            if len(open_) == 1:
                v, s = open_[0]
                assignment[v] = s
                changed = True
            else:
                remaining.append(open_)
        clauses = remaining
        if not changed:
            break
    if not clauses:
        return True
    v, s = clauses[0][0]
    return satisfiable(clauses, {**assignment, v: s}) or satisfiable(clauses, {**assignment, v: not s})


def prove(clauses: Sequence[dict[str, Any]], goal: dict[str, Any]) -> dict[str, Any]:
    """Is `goal` (class, neg) entailed? KB ∧ ¬goal unsatisfiable; also whether the KB itself is consistent."""
    kb = cnf(clauses)
    consistent = satisfiable(kb)
    negated = [(goal["cls"], goal["neg"])]                       # ¬(goal): the opposite literal
    proved = consistent and not satisfiable(kb + [negated])
    return {"proved": bool(proved), "consistent": consistent}


# -- Datalog (LG-A3) -------------------------------------------------------------------------------------------------------

def datalog(facts: dict[str, set[tuple[Any, Any]]], rules: Sequence[dict[str, Any]], *, max_rounds: int = 64) -> dict[str, set]:
    """Rules: {"shape": "sub"|"inv"|"chain"|"star", "body": [p] or [p, r], "head": q, "not": s or None}. Semi-naive."""
    db = {k: set(v) for k, v in facts.items()}
    base = {k: set(v) for k, v in facts.items()}
    for r in rules:
        db.setdefault(r["head"], set())
        for b in r["body"]:
            db.setdefault(b, set())
    delta = {k: set(v) for k, v in db.items()}
    for _ in range(max_rounds):
        new: dict[str, set] = defaultdict(set)
        for r in rules:
            q, body, blocked = r["head"], r["body"], base.get(r.get("not") or "", set())
            if r["shape"] == "sub":
                cand = {(x, y) for x, y in delta[body[0]]}
            elif r["shape"] == "inv":
                cand = {(y, x) for x, y in delta[body[0]]}
            elif r["shape"] == "chain":
                p, s = body
                by_head = defaultdict(set)
                for y, z in db[s]:
                    by_head[y].add(z)
                by_tail = defaultdict(set)
                for x, y in db[p]:
                    by_tail[y].add(x)
                cand = {(x, z) for x, y in delta[p] for z in by_head[y]} | {(x, z) for y, z in delta[s] for x in by_tail[y]}
            elif r["shape"] == "star":
                p, s = body
                by_x = defaultdict(set)
                for x, z in db[s]:
                    by_x[x].add(z)
                by_xp = defaultdict(set)
                for x, y in db[p]:
                    by_xp[x].add(y)
                cand = {(y, z) for x, y in delta[p] for z in by_x[x]} | {(y, z) for x, z in delta[s] for y in by_xp[x]}
            else:
                raise ValueError(r["shape"])
            if r.get("not"):
                cand = {e for e in cand if e not in blocked}
            new[q] |= cand - db[q]
        if not any(new.values()):
            break
        for k, v in new.items():
            db[k] |= v
        delta = defaultdict(set, {k: set(v) for k, v in new.items()})
        for k in db:
            delta.setdefault(k, set())
    return db
