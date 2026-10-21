"""E14 VSA store and executors (plan §2 X3): the logic worlds held in role–filler vectors (the repo's `HRRAlgebra`) and
executed by unbinding and clean-up; every executor writes its conclusions as E12-style recall text.

- `Codebook` — seeded random vectors per symbol (fillers: Gaussian N(0, 1/d); roles: unitary) and clean-up over a set.
- Executors (each returns derived conclusions, a trace and the text the model reads):
  - `TaxonomyStore` — entity frame `IS⊛C0 [+ ISNOT⊛N]`, class frames `SUB⊛parent [+ NEG⊛excluded] + Σ PROP⊛p`; chain
    by unbinding SUB with clean-up per hop, collecting NEG; a presence threshold stops the chain.
  - `CompositionStore` — person frames `NEXT⊛p_{i+1} + LINK⊛relation`; follow NEXT, read LINK, compose names with the
    rule table (held by the executor, as rule records).
  - `PropositionStore` — a fact memory (bundle of true atoms; membership by cosine) and forward chaining over the rules.
  - `TreeStore` — node memory `TEST⊛attr + VAL⊛value | THR⊛B^t + YES⊛child + NO⊛child`, input records
    `Σ ATTR_a⊛value` (numbers by fractional power encoding `B^x`, decoded by a grid search).
  - `MappingRule` — one rule as one vector estimated from k example pairs (LG-C4; fact vectors `PRED⊛p + ARG1⊛a + ARG2⊛b`).
- Variants: `binding=False` (a plain bundle per frame: the no-binding store, which answers "associated with"),
  `shuffle` (relations permuted among their sources), `noise` (a share of edges given a random filler).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from ..algebra import HRRAlgebra, UnitaryHRRAlgebra
from . import reference as R

ALG = HRRAlgebra()


class Codebook:
    def __init__(self, width: int, seed: int) -> None:
        self.width, self.seed = int(width), int(seed)
        self._fill: dict[str, torch.Tensor] = {}
        self._role: dict[str, torch.Tensor] = {}

    def _gen(self, key: str) -> torch.Generator:
        return torch.Generator().manual_seed(abs(hash((self.seed, key))) % (2 ** 63))

    def filler(self, name: str) -> torch.Tensor:
        if name not in self._fill:
            self._fill[name] = torch.randn(self.width, generator=self._gen("f:" + name)) / math.sqrt(self.width)
        return self._fill[name]

    def role(self, name: str) -> torch.Tensor:
        if name not in self._role:
            self._role[name] = UnitaryHRRAlgebra.make_role(torch.randn(self.width, generator=self._gen("r:" + name)))
        return self._role[name]

    def bind(self, role: str, filler: str) -> torch.Tensor:
        return ALG.bind(self.role(role), self.filler(filler))

    def unbind(self, vector: torch.Tensor, role: str) -> torch.Tensor:
        return ALG.unbind(vector, self.role(role))

    def cleanup(self, query: torch.Tensor, names: Sequence[str]) -> tuple[str, float]:
        book = torch.stack([self.filler(n) for n in names])
        scores = F.normalize(book, dim=-1) @ F.normalize(query, dim=-1)
        i = int(scores.argmax())
        return names[i], float(scores[i])

    def ranked(self, query: torch.Tensor, names: Sequence[str], k: int) -> list[tuple[str, float]]:
        book = torch.stack([self.filler(n) for n in names])
        scores = F.normalize(book, dim=-1) @ F.normalize(query, dim=-1)
        top = torch.topk(scores, min(k, len(names)))
        return [(names[int(i)], float(s)) for s, i in zip(top.values, top.indices)]


def _hash_unit(text: str) -> float:
    return (abs(hash(text)) % 10_000) / 10_000


@dataclass
class Recall:
    lines: list[tuple[str, float | None]] = field(default_factory=list)   # (statement, confidence)
    conclusion: str | None = None
    conclusion_score: float | None = None
    answer: Any = None                                                     # the executor's answer (for scoring the store)

    def text(self, subject: str, *, confidence: bool = True, trace: bool = True) -> str:
        fmt = lambda s, c: f"- {s}" + (f" ({c:.2f})" if confidence and c is not None else "")  # noqa: E731
        body = [fmt(s, c) for s, c in self.lines] if trace else []
        if self.conclusion:
            body.append(f"Conclusion: {self.conclusion}" + (f" ({self.conclusion_score:.2f})" if confidence and self.conclusion_score is not None else ""))
        return "\n".join([f"recall({subject}):", *body]) if body else f"recall({subject}):\n- (nothing recalled)"


def _perturb(rules: list[dict[str, Any]], key_b: str, pool: Sequence[str], p: float, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for r in rules:
        r = dict(r)
        if rng.random() < p:
            r[key_b] = rng.choice([x for x in pool if x != r[key_b]])
        out.append(r)
    return out


# -- taxonomy -------------------------------------------------------------------------------------------------------

class TaxonomyStore:
    def __init__(self, item: dict[str, Any], *, width: int = 512, seed: int = 0, binding: bool = True, shuffle: bool = False,
                 noise: float = 0.0, threshold: float = 0.22, max_steps: int = 8, extra_load: int = 0) -> None:
        self.item, self.cb = item, Codebook(width, seed)
        self.binding, self.threshold, self.max_steps = binding, threshold, max_steps
        rng = random.Random(f"tax-store:{seed}:{item['id']}:{shuffle}:{noise}")
        rules = [dict(r) for r in item["rules"]]
        classes = sorted({r["a"] for r in rules} | {r["b"] for r in rules} | set(item["chain"]) | {item["query"]})
        if shuffle:                                                        # parents and excluded classes permuted
            subs = [r for r in rules if r["kind"] == "sub"]
            heads = [r["b"] for r in subs]
            rng.shuffle(heads)
            for r, h in zip(subs, heads):
                r["b"] = h
            negs = [r for r in rules if r["kind"] == "neg"]
            for r in negs:
                r["b"] = rng.choice(classes)
        if noise:
            rules = _perturb(rules, "b", classes, noise, rng)
        self.rules, self.classes = rules, classes
        self.frames: dict[str, torch.Tensor] = {}
        by: dict[str, list[tuple[str, str]]] = {}
        for r in rules:
            by.setdefault(r["a"], []).append(("SUB" if r["kind"] == "sub" else "NEG", r["b"]))
        for c, props in item["properties"].items():
            by.setdefault(c, []).extend(("PROP", p) for p in props)
        if extra_load:                                                     # stressed store: more edges per class frame
            for c in list(by):
                by[c].extend((f"X{j}", f"x{j}:{c}") for j in range(extra_load))
        ent = item["entity"]
        by[ent] = [("IS" if f["kind"] == "is" else "ISNOT", f["cls"]) for f in item["facts"]]
        self.names = sorted(set(classes) | {p for ps in item["properties"].values() for p in ps})
        for key, pairs in by.items():
            if binding:
                self.frames[key] = torch.stack([self.cb.bind(r, f) for r, f in pairs]).sum(0)
            else:
                self.frames[key] = torch.stack([self.cb.filler(f) for _, f in pairs]).sum(0)

    def symbolic_rules(self) -> dict[str, Any]:
        return {**self.item, "rules": self.rules}

    def _read(self, key: str, role: str) -> tuple[str, float] | None:
        if key not in self.frames:
            return None
        name, score = self.cb.cleanup(self.cb.unbind(self.frames[key], role), self.classes)
        return (name, score) if score >= self.threshold else None

    def execute(self) -> Recall:
        ent, query = self.item["entity"], self.item["query"]
        out = Recall()
        if not self.binding:                                               # role-blind: what the bundles are associated with
            seen, current = [], ent
            for _ in range(self.max_steps):
                if current not in self.frames:
                    break
                top = [(n, s) for n, s in self.cb.ranked(self.frames[current], self.names, 4) if n not in seen and s >= self.threshold]
                if not top:
                    break
                out.lines.append((f"{current if current != ent else ent} is associated with: " + ", ".join(n for n, _ in top),
                                  top[0][1]))
                seen += [n for n, _ in top]
                current = top[0][0]
            out.answer = None
            return out
        isnot = self._read(ent, "ISNOT")
        first = self._read(ent, "IS")
        positive: dict[str, float] = {}
        excluded: dict[str, float] = {}
        if isnot:
            excluded[isnot[0]] = isnot[1]
            out.lines.append((f"{ent} is not a {isnot[0]}.", isnot[1]))
        current = first
        if current:
            out.lines.append((f"{ent} is a {current[0]}.", current[1]))
        steps = 0
        while current and steps < self.max_steps:
            positive.setdefault(current[0], current[1])
            neg = self._read(current[0], "NEG")
            if neg:
                excluded.setdefault(neg[0], neg[1])
                out.lines.append((f"No {current[0]} is a {neg[0]}.", neg[1]))
            parent = self._read(current[0], "SUB")
            if not parent or parent[0] in positive:
                break
            out.lines.append((f"Every {current[0]} is a {parent[0]}.", parent[1]))
            current = parent
            steps += 1
        if query in positive:
            out.conclusion, out.conclusion_score, out.answer = f"{ent} is a {query}.", positive[query], True
            out.lines = _trim(out.lines, f" {query}.", keep_first=True)
        elif query in excluded:
            out.conclusion, out.conclusion_score, out.answer = f"{ent} is not a {query}.", excluded[query], False
            out.lines = _trim(out.lines, f" {query}.", keep_first=True)
        else:
            out.conclusion, out.answer = f"no conclusion about whether {ent} is a {query}.", None
        return out


def _trim(lines: list[tuple[str, float | None]], ending: str, *, keep_first: bool) -> list[tuple[str, float | None]]:
    """The trace up to the first line that ends with `ending` (the line that establishes the conclusion); negative lines
    about other classes before it are dropped, as the symbolic trace has only the path."""
    out = []
    for text, score in lines:
        if text.startswith("No ") and not text.endswith(ending):
            continue
        out.append((text, score))
        if text.endswith(ending):
            break
    return out


def taxonomy_symbolic(item: dict[str, Any]) -> Recall:
    """The same recall text from the exact closure (confidence 1.00): the trace to the query (or the whole chain)."""
    ent, query = item["entity"], item["query"]
    c = R.taxonomy_closure(item)
    out = Recall()
    path = c["classes"].get(query) or c["excluded"].get(query)
    start = next(f["cls"] for f in item["facts"] if f["kind"] == "is")
    if path is None:
        path = max(c["classes"].values(), key=len)
    out.lines.append((f"{ent} is a {start}.", 1.0))
    for step in path:
        if step[0] == "sub":
            out.lines.append((f"Every {step[1]} is a {step[2]}.", 1.0))
        elif step[0] == "neg":
            out.lines.append((f"No {step[1]} is a {step[2]}.", 1.0))
        else:
            out.lines.append((f"{ent} is not a {step[2]}.", 1.0))
    if query in c["classes"]:
        out.conclusion, out.conclusion_score, out.answer = f"{ent} is a {query}.", 1.0, True
    elif query in c["excluded"]:
        out.conclusion, out.conclusion_score, out.answer = f"{ent} is not a {query}.", 1.0, False
    else:
        out.conclusion, out.answer = f"no conclusion about whether {ent} is a {query}.", None
    return out


# -- composition ----------------------------------------------------------------------------------------------------

class CompositionStore:
    def __init__(self, world: dict[str, Any], item: dict[str, Any], *, width: int = 512, seed: int = 0, binding: bool = True,
                 noise: float = 0.0, threshold: float = 0.2) -> None:
        self.cb = Codebook(width, seed)
        self.item, self.binding, self.threshold = item, binding, threshold
        self.table = {(a, b): c for a, b, c in world["table"]}
        self.relations = list(world["relation_names"])
        rng = random.Random(f"comp-store:{seed}:{item['id']}:{noise}")
        rels = [r if rng.random() >= noise else rng.choice([x for x in self.relations if x != r]) for r in item["relations"]]
        self.people = list(item["people"])
        self.names = self.people + self.relations
        self.frames = {}
        for i, p in enumerate(self.people[:-1]):
            if binding:
                self.frames[p] = self.cb.bind("NEXT", self.people[i + 1]) + self.cb.bind("LINK", rels[i])
            else:
                self.frames[p] = self.cb.filler(self.people[i + 1]) + self.cb.filler(rels[i])

    def execute(self) -> Recall:
        out = Recall()
        p0 = self.people[0]
        current, acc = p0, None
        if not self.binding:
            for p in self.people[:-1]:
                top = self.cb.ranked(self.frames[p], self.names, 2)
                out.lines.append((f"{p} is associated with: " + ", ".join(n for n, _ in top), top[0][1]))
            return out
        for _ in range(len(self.people) - 1):
            if current not in self.frames:
                break
            nxt, s1 = self.cb.cleanup(self.cb.unbind(self.frames[current], "NEXT"), self.people)
            rel, s2 = self.cb.cleanup(self.cb.unbind(self.frames[current], "LINK"), self.relations)
            out.lines.append((f"{nxt} is the {rel} of {current}.", min(s1, s2)))
            if acc is None:
                acc = rel
            else:
                new = self.table[(acc, rel)]
                out.lines.append((f"rule: a {rel} of a {acc} is a {new}.", None))
                acc = new
            current = nxt
        if acc is not None:
            out.conclusion, out.answer = f"{current} is the {acc} of {p0}.", acc
        return out


def composition_symbolic(world: dict[str, Any], item: dict[str, Any]) -> Recall:
    table = {(a, b): c for a, b, c in world["table"]}
    out = Recall()
    acc = None
    for i, rel in enumerate(item["relations"]):
        out.lines.append((f"{item['people'][i + 1]} is the {rel} of {item['people'][i]}.", 1.0))
        if acc is None:
            acc = rel
        else:
            new = table[(acc, rel)]
            out.lines.append((f"rule: a {rel} of a {acc} is a {new}.", None))
            acc = new
    out.conclusion, out.conclusion_score, out.answer = f"{item['people'][-1]} is the {acc} of {item['people'][0]}.", 1.0, acc
    return out


# -- propositional --------------------------------------------------------------------------------------------------

class PropositionStore:
    def __init__(self, item: dict[str, Any], *, width: int = 512, seed: int = 0, shuffle: bool = False, threshold: float = 0.12,
                 max_rounds: int = 12) -> None:
        self.cb, self.item, self.threshold, self.max_rounds = Codebook(width, seed), item, threshold, max_rounds
        rules = [(tuple(b), h) for b, h in item["rules"]]
        if shuffle:
            heads = [h for _, h in rules]
            random.Random(f"prop-store:{seed}:{item['id']}").shuffle(heads)
            rules = [(b, h) for (b, _), h in zip(rules, heads)]
        self.rules = rules

    def execute(self) -> Recall:
        out = Recall()
        memory = torch.stack([self.cb.filler(f) for f in self.item["facts"]]).sum(0)
        known = {f: 1.0 for f in self.item["facts"]}
        for f in self.item["facts"]:
            out.lines.append((f"it is {f}.", None))
        for _ in range(self.max_rounds):
            added = False
            for body, head in self.rules:
                if head in known:
                    continue
                scores = [float(F.cosine_similarity(memory, self.cb.filler(b), dim=0)) for b in body]
                if min(scores) >= self.threshold:
                    known[head] = min(scores)
                    memory = memory + self.cb.filler(head)
                    out.lines.append((f"it is {' and '.join(body)}, so it is {head}.", min(scores)))
                    added = True
            if not added:
                break
        q = self.item["query"]
        out.answer = q in known
        out.conclusion = f"it is {q}." if out.answer else f"no rule shows that it is {q}."
        out.conclusion_score = known.get(q)
        return out


def proposition_symbolic(item: dict[str, Any]) -> Recall:
    from .world import _forward
    rules = [(tuple(b), h) for b, h in item["rules"]]
    depth = _forward(set(item["facts"]), rules)
    out = Recall()
    for f in item["facts"]:
        out.lines.append((f"it is {f}.", None))
    for body, head in rules:
        if head in depth and depth[head] > 0 and all(b in depth for b in body) and depth[head] == 1 + max(depth[b] for b in body):
            out.lines.append((f"it is {' and '.join(body)}, so it is {head}.", 1.0))
    q = item["query"]
    out.answer = q in depth
    out.conclusion = f"it is {q}." if out.answer else f"no rule shows that it is {q}."
    out.conclusion_score = 1.0 if out.answer else None
    return out


# -- decision trees and lists ---------------------------------------------------------------------------------------

class TreeStore:
    def __init__(self, item: dict[str, Any], *, width: int = 512, seed: int = 0, shuffle: bool = False, grid: int = 101) -> None:
        self.cb, self.item, self.grid = Codebook(width, seed), item, grid
        rng = torch.Generator().manual_seed(seed + 77)
        self.phases = torch.rand(width // 2 + 1, generator=rng) * 2 * math.pi       # B = exp(i·phases): unitary base for FPE
        self.phases[0] = 0.0
        if width % 2 == 0:
            self.phases[-1] = 0.0
        tree = item["tree"] if item["kind"] != "list" else None
        if shuffle and tree is not None:
            leaves = []
            def collect(n):
                if "leaf" in n:
                    leaves.append(n)
                else:
                    collect(n["yes"]); collect(n["no"])
            from .world import _copy
            tree = _copy(tree)
            collect(tree)
            values = [n["leaf"] for n in leaves]
            random.Random(f"tree-shuffle:{seed}:{item['id']}").shuffle(values)
            for n, v in zip(leaves, values):
                n["leaf"] = v
        self.tree = tree
        self.values = sorted({v for a in item["attributes"] if a["values"] for v in a["values"]})
        self.attrs = [a["name"] for a in item["attributes"]]
        self.record = self._encode_record(item["record"])

    def number(self, x: float) -> torch.Tensor:
        """B^x (fractional power encoding over 0–100, scaled to keep neighbouring integers distinguishable)."""
        spectrum = torch.polar(torch.ones_like(self.phases), self.phases * (x / 10.0))
        return torch.fft.irfft(spectrum, n=self.cb.width) / math.sqrt(self.cb.width) * math.sqrt(self.cb.width)

    def decode_number(self, v: torch.Tensor) -> float:
        if getattr(self, "_grid", None) is None:                          # the B^x codebook, built once per store
            self._grid = F.normalize(torch.stack([self.number(x) for x in range(self.grid)]), dim=-1)
        return float(int((self._grid @ F.normalize(v, dim=-1)).argmax()))

    def _encode_record(self, record: dict[str, Any]) -> torch.Tensor:
        parts = []
        for a in self.item["attributes"]:
            value = record[a["name"]]
            filler = self.cb.filler(value) if a["values"] else self.number(float(value))
            parts.append(ALG.bind(self.cb.role("ATTR:" + a["name"]), filler))
        return torch.stack(parts).sum(0)

    def read_value(self, attr: str, numeric: bool) -> tuple[Any, float]:
        cache = self.__dict__.setdefault("_reads", {})
        if attr in cache:
            return cache[attr]
        cache[attr] = self._read_value(attr, numeric)
        return cache[attr]

    def _read_value(self, attr: str, numeric: bool) -> tuple[Any, float]:
        v = ALG.unbind(self.record, self.cb.role("ATTR:" + attr))
        if numeric:
            return self.decode_number(v), float(F.cosine_similarity(v, self.number(self.decode_number(v)), dim=0))
        return self.cb.cleanup(v, self.values)

    def execute(self) -> Recall:
        out = Recall()
        numeric = {a["name"]: a["values"] is None for a in self.item["attributes"]}
        if self.item["kind"] == "list":
            for i, rule in enumerate(self.item["rules"]):
                test = rule["if"]
                value, score = self.read_value(test["attr"], numeric[test["attr"]])
                ok = value == test["value"] if test["op"] == "is" else float(value) <= float(test["value"])
                from .world import test_text
                out.lines.append((f"rule {i + 1}: {test_text(test)}? {'yes' if ok else 'no'} (read: {value})", score))
                if ok:
                    out.conclusion, out.conclusion_score, out.answer = rule["then"], score, rule["then"]
                    return out
            out.conclusion, out.answer = self.item["default"] + " (default)", self.item["default"]
            return out
        node = self.tree
        from .world import test_text
        while "leaf" not in node:
            test = node["test"]
            value, score = self.read_value(test["attr"], numeric[test["attr"]])
            ok = value == test["value"] if test["op"] == "is" else float(value) <= float(test["value"])
            out.lines.append((f"{test_text(test)}? {'yes' if ok else 'no'} (read: {value})", score))
            node = node["yes"] if ok else node["no"]
        out.conclusion, out.answer = node["leaf"], node["leaf"]
        return out


class StableCodebook(Codebook):
    """`Codebook` seeded by SHA-256 of (seed, key) instead of Python's per-process salted `hash`: the same vectors in every
    process (LG-D2's stores; the tier-P stores keep `Codebook`)."""

    def _gen(self, key: str) -> torch.Generator:
        import hashlib
        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2 ** 63))


class StableTreeStore(TreeStore):
    """`TreeStore` with a `StableCodebook` (LG-CF's states, pre-registration addendum 11): byte-identical in every process."""

    def _encode_record(self, record: dict[str, Any]) -> torch.Tensor:
        if not isinstance(self.cb, StableCodebook):
            self.cb = StableCodebook(self.cb.width, self.cb.seed)
        return super()._encode_record(record)


def soft_text(subject: str, lines: Sequence[tuple[str, float | None]], dist: dict[str, float], *, confidence: bool = True,
              floor: float = 0.05) -> tuple[str, str]:
    """A soft path as recall text: the trace, then (with two or more outcomes of weight ≥ `floor`) the possible outcomes, then
    the most probable outcome as the conclusion with its weight. Returns (text, answer)."""
    ranked = sorted(dist.items(), key=lambda kv: (-round(kv[1], 6), kv[0]))   # ties (to 1e-6) by name: BLAS rounding differs by process
    answer, top = ranked[0]
    fmt = lambda s, c: f"- {s}" + (f" ({c:.2f})" if confidence and c is not None else "")  # noqa: E731
    body = [fmt(s, c) for s, c in lines]
    shown = [(o, w) for o, w in ranked if w >= floor]
    if len(shown) > 1:
        body.append("- possible outcomes: " + "; ".join(f"{o} ({w:.2f})" for o, w in shown))
    body.append(f"Conclusion: {answer} ({top:.2f})")
    return "\n".join([f"recall({subject}):", *body]), answer


class NoisyTreeStore(TreeStore):
    """LG-D2 (pre-registration §19): `TreeStore` over a record with missing values (None: left out of the record vector), with a
    `StableCodebook`. A value is missing when its read confidence is below `presence`.
    - `execute`: the exact path; a test on a missing value counts as not met (the "otherwise" branch).
    - `execute_soft`: the soft path; at a missing value both branches are followed, weighted by the test's prior
      (`reference.missing_split`). The leaves reached are superposed into one bundle Σ w·leaf, decoded by least squares against
      the item's outcome vectors."""

    PRESENCE = 0.2

    def __init__(self, item: dict[str, Any], record: dict[str, Any], *, width: int = 512, seed: int = 0, presence: float | None = None) -> None:
        self.presence = self.PRESENCE if presence is None else float(presence)
        super().__init__({**item, "record": record}, width=width, seed=seed)

    def _encode_record(self, record: dict[str, Any]) -> torch.Tensor:
        if not isinstance(self.cb, StableCodebook):
            self.cb = StableCodebook(self.cb.width, self.cb.seed)
        parts = [torch.zeros(self.cb.width)]
        for a in self.item["attributes"]:
            value = record.get(a["name"])
            if value is None:
                continue
            filler = self.cb.filler(value) if a["values"] else self.number(float(value))
            parts.append(ALG.bind(self.cb.role("ATTR:" + a["name"]), filler))
        return torch.stack(parts).sum(0)

    def present(self, attr: str) -> tuple[bool, Any, float]:
        numeric = next(a["values"] is None for a in self.item["attributes"] if a["name"] == attr)
        value, score = self.read_value(attr, numeric)
        return score >= self.presence, value, score

    def _holds(self, test: dict[str, Any], value: Any) -> bool:
        return value == test["value"] if test["op"] == "is" else float(value) <= float(test["value"])

    def execute(self) -> Recall:
        from .world import test_text
        out = Recall()
        node = self.tree
        while "leaf" not in node:
            test = node["test"]
            ok, value, score = self.present(test["attr"])
            if not ok:
                out.lines.append((f"{test_text(test)}? unknown, so no (read: nothing)", score))
                node = node["no"]
                continue
            yes = self._holds(test, value)
            out.lines.append((f"{test_text(test)}? {'yes' if yes else 'no'} (read: {value})", score))
            node = node["yes"] if yes else node["no"]
        out.conclusion, out.answer = node["leaf"], node["leaf"]
        return out

    def execute_soft(self, subject: str, *, confidence: bool = True) -> tuple[str, str, dict[str, float]]:
        """(text, answer, decoded outcome weights). The trace follows the more probable branch at each missing value."""
        from .world import test_text
        reached: list[tuple[float, str]] = []
        lines: list[tuple[str, float | None]] = []

        def walk(node: dict[str, Any], w: float, known: dict[str, Any], main: bool) -> None:
            if w <= 0:
                return
            if "leaf" in node:
                reached.append((w, node["leaf"]))
                return
            test = node["test"]
            ok, value, score = self.present(test["attr"])
            if not ok:
                p, k_yes, k_no = R.missing_split(test, self.item["attributes"], known)
                if main:
                    lines.append((f"{test_text(test)}? unknown (read: nothing): yes with prior {p:.2f}, no with prior {1 - p:.2f}", score))
                walk(node["yes"], w * p, k_yes, main and p >= 0.5); walk(node["no"], w * (1 - p), k_no, main and p < 0.5)
                return
            yes = self._holds(test, value)
            if main:
                lines.append((f"{test_text(test)}? {'yes' if yes else 'no'} (read: {value})", score))
            walk(node["yes"] if yes else node["no"], w, known, main)
        walk(self.tree, 1.0, {}, True)
        outcomes = list(self.item["outcomes"])
        book = torch.stack([self.cb.filler(o) for o in outcomes])                      # (outcomes, d)
        bundle = sum(w * self.cb.filler(leaf) for w, leaf in reached)                    # the path bundle
        alpha = torch.linalg.lstsq(book.T, bundle[:, None]).solution[:, 0].clamp_min(0.0)
        alpha = alpha / alpha.sum() if float(alpha.sum()) > 0 else torch.full_like(alpha, 1.0 / len(outcomes))
        dist = {o: float(a) for o, a in zip(outcomes, alpha)}
        text, answer = soft_text(subject, lines, dist, confidence=confidence)
        return text, answer, dist


def noisy_tree_symbolic(item: dict[str, Any], record: dict[str, Any], *, soft: bool) -> tuple[str, str, dict[str, float]]:
    """LG-D2's symbolic twins, on the true missing set: `soft=False` the exact path with the "otherwise" convention, `soft=True`
    the exact outcome distribution (`reference.tree_leaf_distribution`) in `soft_text` form. Returns (text, answer, dist)."""
    from .world import test_text
    if not soft:
        leaf, path = R.run_tree_missing(item["tree"], record)
        out = Recall(lines=[(f"{test_text(s['test'])}? " + (f"{'yes' if s['result'] else 'no'} (read: {s['value']})" if s["value"] is not None
                              else "unknown, so no (read: nothing)"), 1.0) for s in path], conclusion=leaf, answer=leaf)
        return out.text(item["entity"]), leaf, {leaf: 1.0}
    dist = R.tree_leaf_distribution(item["tree"], record, item["attributes"])
    lines: list[tuple[str, float | None]] = []
    node, known = item["tree"], {}
    while "leaf" not in node:                                                           # the trace: the more probable branch
        test = node["test"]
        value = record.get(test["attr"])
        if value is None:
            p, k_yes, k_no = R.missing_split(test, item["attributes"], known)
            lines.append((f"{test_text(test)}? unknown (read: nothing): yes with prior {p:.2f}, no with prior {1 - p:.2f}", 1.0))
            node, known = (node["yes"], k_yes) if p >= 0.5 else (node["no"], k_no)
        else:
            yes = R.holds(test, record)
            lines.append((f"{test_text(test)}? {'yes' if yes else 'no'} (read: {value})", 1.0))
            node = node["yes"] if yes else node["no"]
    text, answer = soft_text(item["entity"], lines, dist)
    return text, answer, dist


def tree_symbolic(item: dict[str, Any]) -> Recall:
    from .world import test_text
    out = Recall()
    if item["kind"] == "list":
        answer, fired = R.run_list(item["rules"], item["default"], item["record"])
        for i, rule in enumerate(item["rules"][: (fired + 1) if fired is not None else None]):
            ok = R.holds(rule["if"], item["record"])
            out.lines.append((f"rule {i + 1}: {test_text(rule['if'])}? {'yes' if ok else 'no'} (read: {item['record'][rule['if']['attr']]})", 1.0))
        out.conclusion, out.answer = answer + ("" if fired is not None else " (default)"), answer
        return out
    leaf, path = R.run_tree(item["tree"], item["record"])
    for step in path:
        out.lines.append((f"{test_text(step['test'])}? {'yes' if step['result'] else 'no'} (read: {step['value']})", 1.0))
    out.conclusion, out.answer = leaf, leaf
    return out


# -- one rule as one vector (LG-C4) ---------------------------------------------------------------------------------

class MappingRule:
    """Fact vectors `PRED⊛p + ARG1⊛a + ARG2⊛b` (and, for two-premise examples, `+ ARG3⊛c` with a second PRED2 slot);
    the rule `M` is the per-frequency least-squares map from example premises to example conclusions; applied to the query's
    premises and decoded by unbinding ARG2 (the answer slot) against the candidate entities."""

    def __init__(self, item: dict[str, Any], *, width: int = 1024, seed: int = 0) -> None:
        self.cb, self.item = Codebook(width, seed), item

    def _premise(self, premises: Sequence[Sequence[str]]) -> torch.Tensor:
        v = torch.zeros(self.cb.width)
        for i, (x, p, y) in enumerate(premises):
            v = v + self.cb.bind(f"PRED{i}", p) + self.cb.bind(f"ARG{2 * i + 1}", x) + self.cb.bind(f"ARG{2 * i + 2}", y)
        return v

    def _conclusion(self, triple: Sequence[str]) -> torch.Tensor:
        x, p, y = triple
        return self.cb.bind("CPRED", p) + self.cb.bind("CARG1", x) + self.cb.bind("CARG2", y)

    def execute(self) -> Recall:
        xs = torch.stack([self._premise(prem) for prem, _ in self.item["examples"]])
        ys = torch.stack([self._conclusion(conc) for _, conc in self.item["examples"]])
        X, Y = torch.fft.rfft(xs), torch.fft.rfft(ys)
        Mw = (Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3)
        M = torch.fft.irfft(Mw, n=self.cb.width)
        out_v = ALG.bind(M, self._premise(self.item["premises"]))
        answer, score = self.cb.cleanup(self.cb.unbind(out_v, "CARG2"), self.item["options"])
        subject, rel = self.item["question"]
        out = Recall(lines=[(f"pattern of the {len(self.item['examples'])} examples applied to the new case", None)])
        out.conclusion, out.conclusion_score, out.answer = f"{subject} {rel} {answer}.", score, answer
        return out
