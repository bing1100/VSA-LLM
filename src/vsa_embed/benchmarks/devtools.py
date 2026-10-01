"""Developer-tools track (T2 / task C6): a synthetic private library with contamination-free names.

The generator invents modules, classes, functions and exceptions with pronounceable made-up names
(checked against WordNet lemmas so no real word is used), multi-token as snake_case/CamelCase
identifiers. Each symbol has a schema that becomes its frame:

    kind · belongs_to (module) · returns (type) · takes (parameter types) · raises · calls · inherits · category

A documentation corpus (API reference entries, tutorials, changelog notes, Q&A snippets) mentions
symbols consistently with their schemas. Held-out symbols never appear in training documents (the
linker holdout) and appear only in evaluation documents and probes, so zero-shot behaviour must
come from the composed schema. Probes are cloze prompts whose answer is a schema field.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ONSETS = ["b", "br", "d", "dr", "f", "fl", "g", "gr", "k", "kl", "m", "n", "p", "pl", "qu", "r", "s", "st", "t", "tr", "v", "z", "zh", "sk"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou", "ei"]
CODAS = ["", "n", "x", "rk", "lt", "m", "sh", "nd", "rb", "lk"]
BUILTIN_TYPES = ["int", "str", "float", "bool", "bytes", "list", "dict", "Path", "None", "Iterator"]
CATEGORIES = ["io", "parsing", "math", "network", "caching", "logging", "geometry", "scheduling", "crypto", "validation"]
KINDS = ["function", "class", "method", "exception", "constant"]
RELATIONS = ["kind", "belongs_to", "returns", "takes", "raises", "calls", "inherits", "category"]


@dataclass
class Symbol:
    name: str
    kind: str
    module: str
    category: str
    returns: str | None = None
    takes: list[str] = field(default_factory=list)
    raises: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    inherits: str | None = None
    owner: str | None = None          # class of a method
    heldout: bool = False


def _stem(rng: random.Random) -> str:
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(rng.choice([1, 2])))


class NameMaker:
    def __init__(self, rng: random.Random, forbidden: set[str]) -> None:
        self.rng, self.forbidden, self.used = rng, forbidden, set()

    def stem(self) -> str:
        while True:
            s = _stem(self.rng)
            if len(s) >= 4 and s not in self.forbidden and s not in self.used:
                self.used.add(s)
                return s

    def snake(self, parts: int = 2) -> str:
        return "_".join(self.stem() for _ in range(parts))

    def camel(self, parts: int = 2) -> str:
        return "".join(self.stem().capitalize() for _ in range(parts))


def generate_library(*, seed: int, modules: int = 12, classes_per_module: int = 6, functions_per_module: int = 14,
                     heldout_fraction: float = 0.15, forbidden: set[str] | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    names = NameMaker(rng, forbidden or set())
    module_names = [names.snake(1) for _ in range(modules)]
    symbols: list[Symbol] = []
    exceptions = [Symbol(names.camel(1) + "Error", "exception", rng.choice(module_names), rng.choice(CATEGORIES)) for _ in range(10)]
    symbols += exceptions
    classes: list[Symbol] = []
    for module in module_names:
        category = rng.choice(CATEGORIES)
        for _ in range(classes_per_module):
            cls = Symbol(names.camel(2), "class", module, category if rng.random() < 0.7 else rng.choice(CATEGORIES))
            classes.append(cls)
    for cls in classes:
        if rng.random() < 0.3:
            cls.inherits = rng.choice(classes).name
    symbols += classes
    types = BUILTIN_TYPES + [c.name for c in classes]
    functions: list[Symbol] = []
    for module in module_names:
        local_classes = [c for c in classes if c.module == module]
        for _ in range(functions_per_module):
            fn = Symbol(names.snake(2), "function", module, rng.choice([c.category for c in local_classes] + CATEGORIES[:2]))
            fn.returns = rng.choice(types)
            fn.takes = rng.sample(types, k=rng.randint(1, 3))
            fn.raises = [e.name for e in rng.sample(exceptions, k=rng.randint(0, 2))]
            functions.append(fn)
        for cls in local_classes:
            for _ in range(2):
                method = Symbol(names.snake(2), "method", module, cls.category, owner=cls.name)
                method.returns = rng.choice(types); method.takes = rng.sample(types, k=rng.randint(0, 2))
                functions.append(method)
    for fn in functions:
        fn.calls = [other.name for other in rng.sample(functions, k=rng.randint(0, 2)) if other is not fn]
    symbols += functions
    constants = [Symbol(names.snake(2).upper(), "constant", rng.choice(module_names), rng.choice(CATEGORIES)) for _ in range(20)]
    symbols += constants
    eligible = [s for s in symbols if s.kind in {"function", "class", "method"}]
    for symbol in rng.sample(eligible, k=int(len(eligible) * heldout_fraction)):
        symbol.heldout = True
    return {"seed": seed, "modules": module_names, "symbols": [asdict(s) for s in symbols]}


def frames(library: dict[str, Any]) -> dict[str, Any]:
    """Frame ontology: concepts = symbols; atoms = kinds, modules, types, categories, symbols as fillers."""
    symbols = library["symbols"]
    atoms: list[str] = [f"kind:{k}" for k in KINDS] + [f"module:{m}" for m in library["modules"]] + \
        [f"type:{t}" for t in BUILTIN_TYPES] + [f"category:{c}" for c in CATEGORIES] + [f"symbol:{s['name']}" for s in symbols]
    atom = {a: i for i, a in enumerate(atoms)}
    rel = {r: i for i, r in enumerate(RELATIONS)}
    def ref(name: str) -> int:
        return atom.get(f"type:{name}", atom.get(f"symbol:{name}"))
    concept_frames = []
    for s in symbols:
        frame = [(rel["kind"], atom[f"kind:{s['kind']}"]), (rel["belongs_to"], atom[f"module:{s['module']}"]),
                 (rel["category"], atom[f"category:{s['category']}"])]
        if s["returns"]:
            frame.append((rel["returns"], ref(s["returns"])))
        frame += [(rel["takes"], ref(t)) for t in s["takes"]]
        frame += [(rel["raises"], ref(e)) for e in s["raises"]]
        frame += [(rel["calls"], ref(c)) for c in s["calls"]]
        if s["inherits"]:
            frame.append((rel["inherits"], ref(s["inherits"])))
        if s["owner"]:
            frame.append((rel["belongs_to"], ref(s["owner"])))
        concept_frames.append(sorted(set(frame)))
    aliases = [(s["name"], i) for i, s in enumerate(symbols)]
    return {"atoms": atoms, "relations": RELATIONS, "frames": concept_frames, "aliases": aliases,
            "concepts": [s["name"] for s in symbols], "heldout": [i for i, s in enumerate(symbols) if s["heldout"]]}


def _describe(s: dict[str, Any], rng: random.Random, hide: frozenset[str] = frozenset()) -> str | None:
    """Description of a symbol; clauses naming a hidden (held-out) symbol are dropped, and None is
    returned when a hidden symbol is essential (its owner class or return type)."""
    kind = s["kind"]
    if kind == "function" or kind == "method":
        if s["owner"] in hide or s["returns"] in hide:
            return None
        where = f"method of `{s['owner']}`" if s["owner"] else f"function in the `{s['module']}` module"
        takes = ", ".join(t for t in s["takes"] if t not in hide) or "no arguments"
        text = f"`{s['name']}` is a {where} for {s['category']} work. It takes {takes} and returns {s['returns']}."
        raises = [e for e in s["raises"] if e not in hide]
        if raises:
            text += f" It raises {' or '.join(raises)} on invalid input."
        calls = [c for c in s["calls"] if c not in hide]
        if calls:
            text += f" Internally it calls {' and '.join(f'`{c}`' for c in calls)}."
        return text
    if kind == "class":
        text = f"`{s['name']}` is a class in the `{s['module']}` module used for {s['category']}."
        if s["inherits"] and s["inherits"] not in hide:
            text += f" It inherits from `{s['inherits']}`."
        return text
    if kind == "exception":
        return f"`{s['name']}` is an exception raised by the `{s['module']}` module."
    return f"`{s['name']}` is a constant defined in the `{s['module']}` module."


TEMPLATES = [
    "API reference. {desc}",
    "Tutorial: when you are doing {cat} tasks with the {mod} package, use {name}. {desc}",
    "Changelog: {name} was updated this release. {desc}",
    "Q: Which module provides {name}? A: It lives in {mod}. {desc}",
    "In our pipeline we rely on {name} heavily. {desc} Remember that {name} is part of {mod}.",
]


def documents(library: dict[str, Any], *, seed: int, per_symbol: int = 12) -> dict[str, list[str]]:
    """Training documents mention only non-held-out symbols; evaluation documents mention all."""
    rng = random.Random(seed + 1)
    hidden = frozenset(s["name"] for s in library["symbols"] if s["heldout"])
    train, evaluation = [], []
    for s in library["symbols"]:
        full = _describe(s, rng)
        redacted = _describe(s, rng, hidden)
        for k in range(per_symbol):
            to_eval = s["heldout"] or k % 6 == 0 or redacted is None
            desc = full if to_eval else redacted
            doc = rng.choice(TEMPLATES).format(desc=desc, cat=s["category"], mod=s["module"], name=s["name"])
            (evaluation if to_eval else train).append(doc)
    rng.shuffle(train); rng.shuffle(evaluation)
    return {"train": train, "eval": evaluation}


def probes(library: dict[str, Any]) -> list[dict[str, Any]]:
    """Cloze probes whose answer is a schema field (module, return type, kind)."""
    items = []
    for s in library["symbols"]:
        if s["kind"] in {"function", "method"}:
            items.append({"symbol": s["name"], "heldout": s["heldout"], "field": "returns",
                          "prompt": f"The function `{s['name']}` returns", "answer": f" {s['returns']}"})
        items.append({"symbol": s["name"], "heldout": s["heldout"], "field": "module",
                      "prompt": f"`{s['name']}` is defined in the module", "answer": f" `{s['module']}`"})
    return items


def leakage_audit(library: dict[str, Any], docs: dict[str, list[str]]) -> dict[str, Any]:
    held = [s["name"] for s in library["symbols"] if s["heldout"]]
    joined = "\n".join(docs["train"])
    leaked = [name for name in held if name in joined]
    return {"heldout_symbols": len(held), "leaked_into_train": leaked}


def build(out_dir: Path, *, seed: int = 0, forbidden: set[str] | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    library = generate_library(seed=seed, forbidden=forbidden)
    docs = documents(library, seed=seed)
    audit = leakage_audit(library, docs)
    if audit["leaked_into_train"]:
        raise RuntimeError(f"held-out symbols leaked into training docs: {audit['leaked_into_train'][:5]}")
    (out_dir / "library.json").write_text(json.dumps(library, indent=1) + "\n")
    (out_dir / "frames.json").write_text(json.dumps(frames(library)) + "\n")
    for split, texts in docs.items():
        (out_dir / f"{split}.jsonl").write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
    (out_dir / "probes.jsonl").write_text("\n".join(json.dumps(p) for p in probes(library)) + "\n")
    summary = {"symbols": len(library["symbols"]), "train_docs": len(docs["train"]), "eval_docs": len(docs["eval"]),
               **audit, "leaked_into_train": len(audit["leaked_into_train"])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
