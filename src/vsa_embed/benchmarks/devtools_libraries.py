"""Developer-tools track (T2) at scale: several seeded private libraries with contamination-free names.

C6 (`devtools.py`, benchmark v1) generates one small library and is kept unchanged (its v1 files
regenerate bit-identically). This module scales the idea to the E8 T2 track: `libraries` private
libraries, each in Python or TypeScript style, with packages, modules, classes, methods, functions,
exceptions (a hierarchy per library) and constants. Every symbol has a schema that becomes its frame:

    kind · library · language · member_of · category · purpose · returns · takes (parameter types)
    has_param (parameter names) · raises · inherits · overrides · deprecated_by · calls · since · status

**Identifiers** follow each language's conventions: packages and modules are one lower-case stem
("zelkora"), classes CamelCase ("BrainshToulk"), exceptions CamelCase + "Error"/"Warning"/"Timeout",
functions and methods snake_case in Python ("brainsh_toulk") and lowerCamelCase in TypeScript
("brainshToulk"), constants UPPER_SNAKE ("BRAINSH_TOULK"). They link through the linker's
`identifier` alias normalization (`span_channel.ALIAS_NORMALIZATIONS`), which keeps `_`; the default
mode turns `_` into a space, so snake_case names would never link in code. Stems are pronounceable
inventions, unique, never a forbidden word (WordNet lemmas plus the vocabulary of general and
real-library text). Because the linker is prefix-causal (the right word boundary is not checked),
identifiers are also **prefix-free**: no identifier is a prefix of another identifier or of a
forbidden word. The only shared identifiers are overriding methods (same name as the base method),
which the linker treats as one polysemous alias.

**Documents** (API reference entries, module source files with docstrings and comments, type stubs,
usage examples, tutorials, Q&A threads, issue reports with tracebacks, changelogs, migration notes,
code reviews, READMEs) state facts consistent with the schemas. Focus symbols are drawn by a Zipf law
over a random rank order, Zipf–Mandelbrot `(rank + zipf_offset) ** -zipf` (evaluation documents draw
`uniform_focus` of their focus symbols uniformly, which over-samples the tail).

Splits: `train` (in training and evaluation documents), `heldout` (frozen, stratified by Zipf rank;
functions, methods and leaf classes — a held-out class takes its methods with it; only in evaluation
documents: every clause naming one is dropped from training documents) and `zeroshot` (new API
symbols defined only by their schemas — E5.4 — in no document; nothing refers to them).
"""

from __future__ import annotations

import random
import re
from bisect import bisect_left, bisect_right, insort
from collections import Counter
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .devtools import NameMaker

RELATIONS = ["kind", "library", "language", "member_of", "category", "purpose", "returns", "takes", "has_param",
             "raises", "inherits", "overrides", "deprecated_by", "calls", "since", "status"]
KINDS = ["package", "module", "class", "exception", "function", "method", "constant"]
STATUSES = ["stable", "experimental", "deprecated"]
LANGUAGES = ["python", "typescript"]
BUILTIN_TYPES = {
    "python": ["int", "str", "float", "bool", "bytes", "list", "dict", "Path", "None", "Iterator", "datetime", "Callable"],
    "typescript": ["number", "string", "boolean", "Uint8Array", "Array", "Map", "Promise", "void", "Date", "Buffer",
                   "unknown", "RegExp"],
}
BASE_EXCEPTION = {"python": "Exception", "typescript": "Error"}
NONE_TYPE = {"python": "None", "typescript": "void"}
# category: (description, function/class purposes, exception conditions, parameter names)
CATEGORIES: dict[str, tuple[str, list[str], list[str], list[str]]] = {
    "io": ("file input and output",
           ["reads a file in chunks", "writes records to disk", "streams bytes from a socket", "flushes pending writes",
            "opens a file handle", "copies a directory tree", "watches a folder for changes", "tails a log file"],
           ["a file cannot be opened", "a write is interrupted"], ["path", "mode", "encoding", "chunk_size", "overwrite"]),
    "parsing": ("parsing structured text",
                ["parses a config file", "tokenizes an expression", "splits a record into fields", "parses a date string",
                 "reads a header block", "decodes an escape sequence", "builds a syntax tree", "extracts key-value pairs"],
                ["the input is malformed", "an unexpected token is found"], ["source", "strict", "delimiter", "schema", "line"]),
    "networking": ("network requests",
                   ["sends an HTTP request", "opens a websocket", "resolves a host name", "retries a failed request",
                    "downloads a resource", "pools open connections", "sets request headers", "follows redirects"],
                   ["a connection times out", "the server refuses the request"], ["url", "timeout", "headers", "retries", "proxy"]),
    "caching": ("caching",
                ["evicts stale cache entries", "warms the cache from disk", "computes a cache key", "invalidates a cached page",
                 "stores a value with a time to live", "reports cache hit rates", "shards the cache by key", "locks a cache entry"],
                ["a cache key is missing", "the cache is full"], ["key", "ttl", "capacity", "namespace", "default"]),
    "logging": ("logging and tracing",
                ["formats a log record", "rotates log files", "attaches a trace id", "filters noisy messages",
                 "ships logs to a collector", "sets the log level", "buffers log lines", "redacts secrets from logs"],
                ["a log handler is misconfigured", "the log sink is unavailable"], ["level", "message", "handler", "fmt", "extra"]),
    "geometry": ("computational geometry",
                 ["computes a convex hull", "intersects two polygons", "measures the distance between points",
                  "triangulates a mesh", "rotates a shape", "clips a line segment", "computes a bounding box", "simplifies a path"],
                 ["a polygon is degenerate", "coordinates are out of range"], ["points", "origin", "angle", "tolerance", "scale"]),
    "scheduling": ("job scheduling",
                   ["schedules a recurring job", "cancels a pending task", "computes the next run time", "throttles a worker",
                    "assigns jobs to a queue", "pauses the scheduler", "retries a failed job", "drains a task queue"],
                   ["a job deadline is missed", "the queue is closed"], ["job", "interval", "deadline", "priority", "queue"]),
    "crypto": ("cryptography",
               ["hashes a password", "signs a message", "verifies a signature", "derives a key from a passphrase",
                "encrypts a payload", "rotates an encryption key", "generates a random nonce", "checks a certificate chain"],
               ["a signature does not verify", "a key has the wrong length"], ["key", "salt", "digest", "payload", "rounds"]),
    "validation": ("input validation",
                   ["validates an email address", "checks a value against a schema", "normalizes a phone number",
                    "rejects empty fields", "validates a date range", "coerces a value to a number", "collects validation errors",
                    "checks a password policy"],
                   ["a value fails validation", "a required field is missing"], ["value", "rules", "required", "field", "allow_empty"]),
    "serialization": ("serialization",
                      ["serializes an object to JSON", "decodes a binary frame", "encodes a message", "writes a CSV row",
                       "converts a record to a dictionary", "packs integers into bytes", "loads a snapshot", "dumps a schema"],
                      ["an object cannot be serialized", "the payload is truncated"], ["obj", "indent", "compact", "fields", "version"]),
    "concurrency": ("concurrency",
                    ["spawns a worker pool", "waits for all futures", "acquires a lock", "runs a task in the background",
                     "limits concurrent calls", "joins worker threads", "cancels running tasks", "shares state between workers"],
                    ["a lock cannot be acquired", "a worker crashes"], ["workers", "lock", "callback", "max_pending", "daemon"]),
    "storage": ("data storage",
                ["opens a database session", "runs a migration", "inserts a batch of rows", "commits a transaction",
                 "compacts a table", "backs up a bucket", "lists stored objects", "deletes expired records"],
                ["a transaction conflicts", "a bucket does not exist"], ["table", "rows", "bucket", "session", "batch_size"]),
    "testing": ("testing",
                ["creates a test fixture", "mocks a network call", "compares two snapshots", "seeds the random generator",
                 "collects test results", "runs a test suite", "measures code coverage", "generates fake records"],
                ["a fixture cannot be created", "a snapshot does not match"], ["fixture", "seed", "expected", "actual", "verbose"]),
    "configuration": ("configuration",
                      ["loads settings from the environment", "merges two config layers", "watches a config file",
                       "resolves a setting by name", "validates a config section", "exports the active settings",
                       "applies command-line overrides", "reloads the configuration"],
                      ["a setting is unknown", "a config layer is invalid"], ["name", "env", "defaults", "section", "override"]),
    "metrics": ("metrics and monitoring",
                ["records a counter value", "computes a histogram", "exports metrics to a dashboard", "samples request latency",
                 "aggregates a time series", "tags a metric", "resets a gauge", "alerts on a threshold"],
                ["a metric name is invalid", "the exporter is unreachable"], ["metric", "labels", "window", "threshold", "unit"]),
    "auth": ("authentication",
             ["issues an access token", "refreshes a session", "checks user permissions", "revokes an api key",
              "logs a user out", "validates a login form", "hashes a session id", "maps roles to scopes"],
             ["a token has expired", "a user lacks a permission"], ["user", "token", "scopes", "expires_in", "audience"]),
}
COMMON_PARAMS = ["options", "callback", "verbose", "name", "limit", "context"]
_VERSION_MINOR = 9
_WORDS = re.compile(r"[a-z0-9]+")


# -- names ------------------------------------------------------------------------------------------

class IdentifierMaker:
    """Unique, prefix-free identifiers built from invented stems (see the module docstring)."""

    def __init__(self, rng: random.Random, forbidden: set[str]) -> None:
        self.rng = rng
        self.stems = NameMaker(rng, forbidden)
        self.words = sorted(forbidden)
        self.taken: set[str] = set()
        self.taken_sorted: list[str] = []

    def _free(self, identifier: str) -> bool:
        low = identifier.lower()
        if low in self.taken or any(low[:k] in self.taken for k in range(1, len(low))):
            return False
        for pool in (self.words, self.taken_sorted):
            i = bisect_left(pool, low)
            if i < len(pool) and pool[i].startswith(low):
                return False
        return True

    def make(self, build: Callable[[list[str]], str], parts: int) -> tuple[str, list[str]]:
        while True:
            stems = [self.stems.stem() for _ in range(parts)]
            identifier = build(stems)
            if self._free(identifier):
                low = identifier.lower()
                self.taken.add(low); insort(self.taken_sorted, low)
                return identifier, stems

    def lower(self) -> tuple[str, list[str]]:
        return self.make(lambda s: s[0], 1)

    def camel(self, parts: int = 2, suffix: str = "") -> tuple[str, list[str]]:
        return self.make(lambda s: "".join(p.capitalize() for p in s) + suffix, parts)

    def lower_camel(self) -> tuple[str, list[str]]:
        return self.make(lambda s: s[0] + "".join(p.capitalize() for p in s[1:]), 2)

    def snake(self) -> tuple[str, list[str]]:
        return self.make(lambda s: "_".join(s), 2)

    def callable_name(self, language: str) -> tuple[str, list[str]]:
        """A function or method name: snake_case (Python) or lowerCamelCase (TypeScript)."""
        return self.snake() if language == "python" else self.lower_camel()

    def upper(self) -> tuple[str, list[str]]:
        return self.make(lambda s: "_".join(s).upper(), 2)


def _camel_param(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.capitalize() for p in rest)


def param_display(name: str, language: str) -> str:
    return _camel_param(name) if language == "typescript" else name


# -- generation -------------------------------------------------------------------------------------

def qualname(symbol: dict[str, Any]) -> str:
    """Dotted path of a private symbol: library[.module][.Owner].name."""
    if symbol["kind"] == "package":
        return symbol["library"]
    if symbol["kind"] == "module":
        return f"{symbol['library']}.{symbol['name']}"
    owner = f".{symbol['owner']}" if symbol["owner"] else ""
    return f"{symbol['library']}.{symbol['module']}{owner}.{symbol['name']}"


def _symbol(name: str, stems: list[str], kind: str, library: dict[str, Any], **fields: Any) -> dict[str, Any]:
    base = {"name": name, "stems": stems, "kind": kind, "library": library["name"], "language": library["language"],
            "module": None, "owner": None, "category": None, "purpose": None, "params": [], "returns": None, "raises": [],
            "calls": [], "inherits": None, "overrides": None, "deprecated_by": None, "since": library["versions"][0],
            "status": "stable", "split": "train", "rank": None, "weight": 0.0}
    base.update(fields)
    return base


def _versions(rng: random.Random) -> list[str]:
    major = rng.randint(0, 2)
    minors = sorted(rng.sample(range(_VERSION_MINOR + 1), rng.randint(5, 8)))
    out = [f"{major}.{m}" for m in minors]
    out += [f"{major + 1}.{m}" for m in sorted(rng.sample(range(5), rng.randint(1, 3)))]
    return out


def _pick_type(rng: random.Random, types: tuple[list[str], list[str]], class_share: float = 0.4) -> str:
    """A builtin type, or (with probability `class_share`) a class of the library."""
    builtin, classes = types
    return rng.choice(classes) if classes and rng.random() < class_share else rng.choice(builtin)


def _params(rng: random.Random, category: str, types: tuple[list[str], list[str]], count: int) -> list[dict[str, Any]]:
    names = rng.sample(CATEGORIES[category][3] + COMMON_PARAMS, count)
    params = []
    for i, name in enumerate(names):
        optional = i >= max(1, count - rng.randint(0, 2))
        params.append({"name": name, "type": _pick_type(rng, types), "optional": optional})
    return params


def generate_private_libraries(*, seed: int, libraries: int = 16, typescript_fraction: float = 0.3,
                               modules: tuple[int, int] = (10, 16), classes: tuple[int, int] = (5, 9),
                               functions: tuple[int, int] = (10, 18), methods: tuple[int, int] = (2, 6),
                               exceptions: tuple[int, int] = (10, 16), constants: tuple[int, int] = (0, 2),
                               heldout_fraction: float = 0.10, zero_shot: int = 600, zipf: float = 1.2,
                               zipf_offset: float = 0.0,
                               deprecated_fraction: float = 0.04, experimental_fraction: float = 0.05,
                               forbidden: set[str] | None = None) -> dict[str, Any]:
    """A deterministic set of private libraries (JSON-serialisable; see the module docstring)."""
    rng = random.Random(seed)
    # every word of this module's own source is forbidden too, so no template word can equal a stem
    own_words = set(_WORDS.findall(Path(__file__).read_text().lower()))
    names = IdentifierMaker(rng, {w.lower() for w in (forbidden or set())} | own_words)
    n_ts = round(libraries * typescript_fraction)
    languages = ["typescript"] * n_ts + ["python"] * (libraries - n_ts)
    rng.shuffle(languages)
    libs: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    for language in languages:
        name, stems = names.lower()
        lib = {"name": name, "stems": stems, "language": language, "versions": _versions(rng),
               "categories": sorted(rng.sample(sorted(CATEGORIES), rng.randint(3, 5))), "modules": []}
        libs.append(lib)
        builtin = BUILTIN_TYPES[language]
        symbols.append(_symbol(name, stems, "package", lib, category=lib["categories"][0]))
        mods = []
        for _ in range(rng.randint(*modules)):
            mod_name, mod_stems = names.lower()
            mods.append(_symbol(mod_name, mod_stems, "module", lib, category=rng.choice(lib["categories"])))
        lib["modules"] = [m["name"] for m in mods]
        symbols += mods
        # exceptions: one base error per library, the others inherit from it or from each other
        base_name, base_stems = names.camel(1, "Error")
        excs = [_symbol(base_name, base_stems, "exception", lib, module=mods[0]["name"], category=mods[0]["category"],
                        inherits=BASE_EXCEPTION[language], purpose="any error raised by the library")]
        for _ in range(rng.randint(*exceptions) - 1):
            mod = rng.choice(mods)
            exc_name, exc_stems = names.camel(rng.choice([1, 2]), rng.choice(["Error"] * 5 + ["Warning", "Timeout"]))
            parent = rng.choice(excs) if rng.random() < 0.3 else excs[0]
            excs.append(_symbol(exc_name, exc_stems, "exception", lib, module=mod["name"], category=mod["category"],
                                inherits=parent["name"], purpose=rng.choice(CATEGORIES[mod["category"]][2])))
        symbols += excs
        lib_classes: list[dict[str, Any]] = []
        lib_functions: list[dict[str, Any]] = []
        for mod in mods:
            category = mod["category"]
            local = []
            for _ in range(rng.randint(*classes)):
                cat = category if rng.random() < 0.8 else rng.choice(lib["categories"])
                cls_name, cls_stems = names.camel(2)
                cls = _symbol(cls_name, cls_stems, "class", lib, module=mod["name"], category=cat,
                              purpose=rng.choice(CATEGORIES[cat][1]), since=rng.choice(lib["versions"]))
                if lib_classes and rng.random() < 0.25:
                    cls["inherits"] = rng.choice(lib_classes)["name"]
                local.append(cls)
                lib_classes.append(cls)
            symbols += local
            types = ([t for t in builtin if t != NONE_TYPE[language]], [c["name"] for c in lib_classes])
            for _ in range(rng.randint(*functions)):
                cat = category if rng.random() < 0.8 else rng.choice(lib["categories"])
                fn_name, fn_stems = names.callable_name(language)
                fn = _symbol(fn_name, fn_stems, "function", lib, module=mod["name"], category=cat,
                             purpose=rng.choice(CATEGORIES[cat][1]), since=rng.choice(lib["versions"]),
                             params=_params(rng, cat, types, rng.randint(0, 4)),
                             returns=_pick_type(rng, types) if rng.random() < 0.85 else NONE_TYPE[language],
                             raises=sorted(e["name"] for e in rng.sample(excs, rng.choice([0, 1, 1, 2]))))
                lib_functions.append(fn)
                symbols.append(fn)
            for cls in local:
                for _ in range(rng.randint(*methods)):
                    m_name, m_stems = names.callable_name(language)
                    method = _symbol(m_name, m_stems, "method", lib, module=mod["name"], owner=cls["name"],
                                     category=cls["category"], purpose=rng.choice(CATEGORIES[cls["category"]][1]),
                                     since=rng.choice(lib["versions"]), params=_params(rng, cls["category"], types, rng.randint(0, 3)),
                                     returns=_pick_type(rng, types) if rng.random() < 0.8 else NONE_TYPE[language],
                                     raises=sorted(e["name"] for e in rng.sample(excs, rng.choice([0, 0, 1, 2]))))
                    lib_functions.append(method)
                    symbols.append(method)
            for _ in range(rng.randint(*constants)):
                c_name, c_stems = names.upper()
                symbols.append(_symbol(c_name, c_stems, "constant", lib, module=mod["name"], category=category,
                                       returns=rng.choice(builtin[:3]), since=rng.choice(lib["versions"])))
        # overriding methods: a subclass re-implements a method of its base (same name, a polysemous alias)
        by_class = {}
        for s in lib_functions:
            if s["kind"] == "method":
                by_class.setdefault(s["owner"], []).append(s)
        for cls in lib_classes:
            base_methods = by_class.get(cls["inherits"] or "", [])
            if base_methods and rng.random() < 0.5:
                base = rng.choice(base_methods)
                if base["overrides"] or any(m["name"] == base["name"] for m in by_class.get(cls["name"], [])):
                    continue
                method = _symbol(base["name"], base["stems"], "method", lib, module=cls["module"], owner=cls["name"],
                                 category=cls["category"], purpose=base["purpose"], since=rng.choice(lib["versions"]),
                                 params=[dict(p) for p in base["params"]], returns=base["returns"], raises=list(base["raises"]),
                                 overrides=f"{base['owner']}.{base['name']}")
                by_class.setdefault(cls["name"], []).append(method)
                lib_functions.append(method)
                symbols.append(method)
        callable_names = [s["name"] for s in lib_functions if not s["overrides"]]
        for fn in lib_functions:
            fn["calls"] = sorted({c for c in rng.sample(callable_names, rng.choice([0, 1, 1, 2])) if c != fn["name"]})
        # status: deprecated symbols point to a newer function of the same module
        for fn in lib_functions:
            if fn["overrides"] or fn["kind"] != "function":
                continue
            roll = rng.random()
            if roll < deprecated_fraction:
                newer = [g for g in lib_functions if g["kind"] == "function" and g["module"] == fn["module"] and g is not fn
                         and not g["deprecated_by"] and g["status"] != "deprecated"]
                if newer:
                    fn["status"], fn["deprecated_by"] = "deprecated", rng.choice(newer)["name"]
            elif roll < deprecated_fraction + experimental_fraction:
                fn["status"] = "experimental"
    _assign_weights(symbols, rng, zipf, zipf_offset)
    _choose_heldout(symbols, rng, heldout_fraction)
    symbols += _zero_shot(libs, symbols, names, rng, zero_shot)
    return {"seed": seed, "zipf": zipf, "zipf_offset": zipf_offset, "libraries": libs, "symbols": symbols}


def _assign_weights(symbols: list[dict[str, Any]], rng: random.Random, zipf: float, offset: float = 0.0) -> None:
    """Zipf–Mandelbrot focus weights `(rank + offset) ** -zipf` over a random rank order (the offset
    flattens the head so no single symbol dominates the documents)."""
    order = list(range(len(symbols)))
    rng.shuffle(order)
    for rank, index in enumerate(order, start=1):
        symbols[index]["rank"] = rank
        symbols[index]["weight"] = (rank + offset) ** -zipf


def referenced_names(symbols: Iterable[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    """(library, name) → relations through which other symbols refer to it."""
    refs: dict[tuple[str, str], set[str]] = {}
    for s in symbols:
        lib = s["library"]
        for relation, names in (("returns", [s["returns"]]), ("takes", [p["type"] for p in s["params"]]),
                                ("inherits", [s["inherits"]]), ("raises", s["raises"]), ("member_of", [s["owner"]]),
                                ("calls", s["calls"]), ("deprecated_by", [s["deprecated_by"]]),
                                ("overrides", [s["overrides"].split(".")[1] if s["overrides"] else None])):
            for name in names:
                if name:
                    refs.setdefault((lib, name), set()).add(relation)
    return refs


def _choose_heldout(symbols: list[dict[str, Any]], rng: random.Random, fraction: float) -> None:
    """Held-out symbols, stratified by Zipf rank decile: functions, methods (not overriding or
    overridden) and leaf classes (never a type, base or owner of another symbol's reference); a
    held-out class takes its methods with it."""
    refs = referenced_names(symbols)
    overridden = {(s["library"], s["overrides"]) for s in symbols if s["overrides"]}
    overriding_owners = {(s["library"], s["owner"]) for s in symbols if s["overrides"]}
    def eligible(s: dict[str, Any]) -> bool:
        if s["kind"] in ("function", "method"):
            return not s["overrides"] and (s["library"], f"{s['owner']}.{s['name']}") not in overridden
        if s["kind"] == "class":        # a leaf: no other symbol uses it as a type or base, no overriding methods
            return not refs.get((s["library"], s["name"]), set()) & {"returns", "takes", "inherits", "raises"} and \
                (s["library"], s["name"]) not in overriding_owners
        return False
    pool = sorted((s for s in symbols if eligible(s)), key=lambda s: s["rank"])
    bins = 10
    chosen: list[dict[str, Any]] = []
    for b in range(bins):
        chunk = pool[b * len(pool) // bins:(b + 1) * len(pool) // bins]
        chosen += rng.sample(chunk, max(1, round(len(chunk) * fraction))) if chunk else []
    held_classes = {(s["library"], s["name"]) for s in chosen if s["kind"] == "class"}
    chosen_ids = {id(s) for s in chosen}
    for s in symbols:
        if id(s) in chosen_ids or (s["kind"] == "method" and (s["library"], s["owner"]) in held_classes):
            s["split"] = "heldout"


def _zero_shot(libs: list[dict[str, Any]], symbols: list[dict[str, Any]], names: IdentifierMaker,
               rng: random.Random, count: int) -> list[dict[str, Any]]:
    """New API symbols defined only by schemas over existing symbols (E5.4); nothing refers to them."""
    out: list[dict[str, Any]] = []
    by_lib: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for s in symbols:
        if s["split"] == "train":
            by_lib.setdefault(s["library"], {}).setdefault(s["kind"], []).append(s)
    lib_index = {lib["name"]: lib for lib in libs}
    while len(out) < count:
        lib = lib_index[rng.choice(sorted(by_lib))]
        pool = by_lib[lib["name"]]
        mods = pool["module"]
        types = ([t for t in BUILTIN_TYPES[lib["language"]] if t != NONE_TYPE[lib["language"]]],
                 [c["name"] for c in pool.get("class", [])])
        excs = pool["exception"]
        callables = [s["name"] for s in pool.get("function", []) + pool.get("method", []) if not s["overrides"]]
        roll = rng.random()
        mod = rng.choice(mods)
        cat = mod["category"] if rng.random() < 0.8 else rng.choice(lib["categories"])
        def callable_symbol(kind: str, owner: str | None, module: str, category: str) -> dict[str, Any]:
            fn_name, fn_stems = names.callable_name(lib["language"])
            return _symbol(fn_name, fn_stems, kind, lib, module=module, owner=owner, category=category,
                           purpose=rng.choice(CATEGORIES[category][1]), since=lib["versions"][-1],
                           params=_params(rng, category, types, rng.randint(1, 4)), returns=_pick_type(rng, types),
                           raises=sorted(e["name"] for e in rng.sample(excs, rng.choice([0, 1, 1, 2]))),
                           calls=sorted(rng.sample(callables, rng.choice([0, 1, 2]))), split="zeroshot", status="experimental")
        if roll < 0.5:
            out.append(callable_symbol("function", None, mod["name"], cat))
        elif roll < 0.7:
            cls_name, cls_stems = names.camel(2)
            classes = pool.get("class", [])
            cls = _symbol(cls_name, cls_stems, "class", lib, module=mod["name"], category=cat,
                          purpose=rng.choice(CATEGORIES[cat][1]), since=lib["versions"][-1], split="zeroshot",
                          status="experimental", inherits=rng.choice(classes)["name"] if classes and rng.random() < 0.4 else None)
            out.append(cls)
            for _ in range(rng.randint(1, 3)):
                if len(out) < count:
                    out.append(callable_symbol("method", cls["name"], mod["name"], cat))
        else:
            owner = rng.choice(pool.get("class", []))
            out.append(callable_symbol("method", owner["name"], owner["module"], owner["category"]))
    return out[:count]


# -- facts ------------------------------------------------------------------------------------------

def symbol_facts(symbol: dict[str, Any]) -> dict[str, list[str]]:
    """Relation → readable filler names (identifiers as written in code)."""
    s = symbol
    facts: dict[str, list[str]] = {"kind": [s["kind"]], "library": [s["library"]], "language": [s["language"]],
                                   "since": [s["since"]], "status": [s["status"]]}
    if s["kind"] not in ("package", "module"):
        facts["member_of"] = [s["owner"] or s["module"]]
    elif s["kind"] == "module":
        facts["member_of"] = [s["library"]]
    if s["category"]:
        facts["category"] = [CATEGORIES[s["category"]][0]]
    if s["purpose"]:
        facts["purpose"] = [s["purpose"]]
    if s["returns"] and s["kind"] in ("function", "method"):
        facts["returns"] = [s["returns"]]
    if s["params"]:
        facts["takes"] = sorted({p["type"] for p in s["params"]})
        facts["has_param"] = [param_display(p["name"], s["language"]) for p in s["params"]]
    for relation in ("raises", "calls"):
        if s[relation]:
            facts[relation] = list(s[relation])
    for relation in ("inherits", "overrides", "deprecated_by"):
        if s[relation]:
            facts[relation] = [s[relation]]
    return facts


# -- documents --------------------------------------------------------------------------------------

FACT_TEMPLATES: dict[str, list[str]] = {
    "purpose": ["{x} {y}.", "In short, {x} {y}.", "Use {x} when your code {y}.", "{x} is the helper that {y}."],
    "returns": ["{x} returns {y}.", "The return value of {x} is {y_a}.", "Calling {x} gives you {y_a}.", "{x} hands back {y_a}."],
    "takes": ["{x} takes {y_a} argument.", "One argument of {x} is {y_a}.", "{x} expects {y_a} as input."],
    "has_param": ["{x} accepts a parameter named {y}.", "The {y} parameter of {x} is documented below.",
                  "Pass {y} to {x} to change its behaviour."],
    "raises": ["{x} raises {y} when {cond}.", "{x} may raise {y}.", "Expect {y} from {x} if {cond}.",
               "Wrap calls to {x} in a handler for {y}."],
    "calls": ["Internally, {x} calls {y}.", "{x} delegates part of the work to {y}.", "{x} is implemented on top of {y}."],
    "inherits": ["{x} inherits from {y}.", "{x} is a subclass of {y}.", "{x} extends {y}."],
    "overrides": ["{x} overrides {y}.", "{x} replaces the base implementation {y}.", "This version of {x} overrides {y}."],
    "deprecated_by": ["{x} is deprecated; use {y} instead.", "{x} is deprecated in favour of {y}.",
                      "Prefer {y}: {x} will be removed in a future release."],
    "member_of": ["{x} is defined in {y}.", "{x} lives in {y}.", "You can find {x} in {y}."],
    "since": ["{x} was added in version {y}.", "{x} is available since {lib} {y}.", "New in version {y}: {x}."],
    "status": ["{x} is {y}.", "The API status of {x} is {y}."],
    "category": ["{x} is part of the {y} tools of {lib}.", "{x} is used for {y}."],
    "kind": ["{x} is a {y}.", "In the API reference {x} is listed as a {y}."],
}
BOILERPLATE = [
    "See the changelog for details.", "Contributions are welcome.", "This page is generated from the docstrings.",
    "Feedback is welcome on the issue tracker.", "The examples assume the default configuration.",
    "All examples were tested with the latest release.", "Thread safety is not guaranteed unless stated otherwise.",
    "Pin the version in production environments.", "Breaking changes are listed in the migration guide.",
    "Questions can be asked in the discussions forum.", "Run the test suite before sending a pull request.",
    "The public API follows semantic versioning.", "Type hints are included in the package.",
    "Logging is disabled by default.", "Benchmarks are in the repository.", "This section was reviewed for the latest release.",
]
USERS = ["alex", "sam", "jordan", "kim", "lee", "morgan", "riley", "taylor", "casey", "devon", "jamie", "robin"]


def _a(phrase: str) -> str:
    return ("an " if phrase.lstrip("`")[:1].lower() in "aeiou" else "a ") + phrase


class _Writer:
    """Renders documents about private symbols; every fact naming a hidden symbol is dropped."""

    def __init__(self, data: dict[str, Any], hidden: frozenset[tuple[str, str]], rng: random.Random) -> None:
        self.rng, self.hidden = rng, hidden
        self.libs = {lib["name"]: lib for lib in data["libraries"]}
        self.symbols = [s for s in data["symbols"] if s["split"] != "zeroshot"]
        self.by_name: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.members: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.by_version: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.raisers: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.callers: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.of_kind: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for s in self.symbols:
            self.of_kind.setdefault((s["library"], s["kind"]), []).append(s)
            lib = s["library"]
            self.by_name.setdefault((lib, s["name"]), []).append(s)
            if s["owner"]:
                self.members.setdefault((lib, "class:" + s["owner"]), []).append(s)
            elif s["module"]:
                self.members.setdefault((lib, "module:" + s["module"]), []).append(s)
            self.by_version.setdefault((lib, s["since"]), []).append(s)
            for e in s["raises"]:
                self.raisers.setdefault((lib, e), []).append(s)
            for c in s["calls"]:
                self.callers.setdefault((lib, c), []).append(s)

    # helpers --------------------------------------------------------------------------------------
    def visible(self, lib: str, name: str | None) -> bool:
        return bool(name) and (lib, name) not in self.hidden

    def shown(self, s: dict[str, Any], names: Iterable[str]) -> list[str]:
        return [n for n in names if self.visible(s["library"], n)]

    def lookup(self, lib: str, name: str) -> dict[str, Any] | None:
        found = self.by_name.get((lib, name))
        return found[0] if found else None

    def code(self, name: str) -> str:
        return f"`{name}`"

    def comment(self, s: dict[str, Any], text: str) -> str:
        return ("# " if s["language"] == "python" else "// ") + text

    def type_text(self, s: dict[str, Any], type_name: str) -> str:
        return type_name

    def params_text(self, s: dict[str, Any], *, with_types: bool = True) -> str:
        parts = []
        for p in s["params"]:
            name = param_display(p["name"], s["language"])
            if s["language"] == "python":
                text = f"{name}: {p['type']}" if with_types else name
                parts.append(text + (f" = {self.default(s, p['type'])}" if p["optional"] else ""))
            else:
                parts.append(f"{name}{'?' if p['optional'] else ''}: {p['type']}" if with_types else name)
        if s["kind"] == "method" and s["language"] == "python":
            parts.insert(0, "self")
        return ", ".join(parts)

    def default(self, s: dict[str, Any], type_name: str) -> str:
        py = s["language"] == "python"
        table = {"int": "0", "float": "1.0", "bool": "False", "str": '""', "None": "None", "number": "0",
                 "boolean": "false", "string": '""'}
        return table.get(type_name, "None" if py else "undefined")

    def value(self, s: dict[str, Any], type_name: str) -> str:
        py = s["language"] == "python"
        table = {"int": "3", "float": "0.5", "bool": "True", "str": '"input.txt"', "bytes": 'b"payload"', "list": "[1, 2, 3]",
                 "dict": '{"mode": "fast"}', "Path": 'Path("data")', "None": "None", "Iterator": "iter(items)",
                 "datetime": "datetime.now()", "Callable": "print", "number": "3", "string": '"input.txt"',
                 "boolean": "true", "Uint8Array": "new Uint8Array(16)", "Array": "[1, 2, 3]", "Map": "new Map()",
                 "Promise": "Promise.resolve(1)", "void": "undefined", "Date": "new Date()", "Buffer": 'Buffer.from("x")',
                 "unknown": "{}", "RegExp": "/x+/"}
        if type_name in table:
            return table[type_name]
        if py:
            return f"{type_name}()"
        return f"new {type_name}()"

    def signature(self, s: dict[str, Any]) -> str:
        py = s["language"] == "python"
        kind = s["kind"]
        if kind in ("function", "method"):
            if py:
                return f"def {s['name']}({self.params_text(s)}) -> {s['returns']}:"
            prefix = "export function " if kind == "function" else ""
            return f"{prefix}{s['name']}({self.params_text(s)}): {s['returns']}"
        if kind in ("class", "exception"):
            base = s["inherits"] if self.visible(s["library"], s["inherits"]) or s["inherits"] in BASE_EXCEPTION.values() else None
            if py:
                return f"class {s['name']}({base}):" if base else f"class {s['name']}:"
            return f"export class {s['name']}" + (f" extends {base}" if base else "")
        if kind == "constant":
            value = self.value(s, s["returns"])
            return f"{s['name']}: {s['returns']} = {value}" if py else f"export const {s['name']}: {s['returns']} = {value};"
        return s["name"]

    def import_line(self, s: dict[str, Any], names: list[str]) -> str:
        lib = s["library"]
        if s["language"] == "python":
            return f"from {lib}.{s['module']} import {', '.join(names)}"
        return f'import {{ {", ".join(names)} }} from "{lib}/{s["module"]}";'

    def where(self, s: dict[str, Any]) -> str:
        if s["kind"] == "method":
            return f"a method of {self.code(s['owner'])}"
        if s["kind"] == "module":
            return f"a module of the {self.code(s['library'])} package"
        if s["kind"] == "package":
            return "a package"
        return f"{_a(s['kind'])} in the {self.code(s['module'])} module"

    def fact_sentences(self, s: dict[str, Any], *, limit: int | None = None) -> list[str]:
        facts = symbol_facts(s)
        lib = s["library"]
        out = []
        relations = [r for r in ("purpose", "returns", "takes", "has_param", "raises", "calls", "inherits", "overrides",
                                 "deprecated_by", "member_of", "since", "status", "category") if r in facts]
        self.rng.shuffle(relations)
        for relation in relations[:limit]:
            fillers = facts[relation]
            if relation in ("raises", "calls", "inherits", "deprecated_by", "returns", "takes"):
                fillers = [f for f in fillers if self.visible(lib, f) or f in BUILTIN_TYPES[s["language"]]
                           or f in BASE_EXCEPTION.values()]
            if relation == "overrides":
                fillers = [f for f in fillers if self.visible(lib, f.split(".")[1]) and self.visible(lib, f.split(".")[0])]
            if relation == "status" and fillers == ["stable"] and self.rng.random() < 0.7:
                continue
            if relation == "returns" and fillers[0] in NONE_TYPE.values():
                fillers = []
            if not fillers:
                continue
            y = self.rng.choice(fillers)
            template = self.rng.choice(FACT_TEMPLATES[relation])
            code_like = relation in ("returns", "takes", "raises", "calls", "inherits", "overrides", "deprecated_by",
                                     "member_of", "has_param")
            shown = self.code(y) if code_like else y
            cond = self.rng.choice(CATEGORIES[s["category"]][2]) if s["category"] else "the input is invalid"
            if relation == "raises":
                exc = self.lookup(lib, y)
                if exc and exc["purpose"] and exc["purpose"] != "any error raised by the library":
                    cond = exc["purpose"]
            text = template.format(x=self.code(s["name"]), y=shown, y_a=_a(shown) if relation in ("returns", "takes") else shown,
                                   cond=cond, lib=lib)
            out.append(text[:1].upper() + text[1:])
        return out

    def docblock(self, s: dict[str, Any], indent: str) -> list[str]:
        """Docstring (Python, Google style) or JSDoc (TypeScript) lines."""
        lib = s["library"]
        summary = (s["purpose"] or f"{s['kind']} of {lib}").capitalize() + "."
        py = s["language"] == "python"
        lines: list[str] = []
        if py:
            lines.append(f'{indent}"""{summary}')
            if s["params"]:
                lines += ["", f"{indent}Args:"]
                lines += [f"{indent}    {param_display(p['name'], 'python')} ({p['type']}): the {p['name'].replace('_', ' ')}"
                          + (" (optional)." if p["optional"] else ".") for p in s["params"]]
            if s["kind"] in ("function", "method") and s["returns"] not in ("None", "void"):
                lines += ["", f"{indent}Returns:", f"{indent}    {s['returns']}: the result."]
            raises = self.shown(s, s["raises"])
            if raises:
                lines += ["", f"{indent}Raises:"] + [f"{indent}    {e}: if {self.rng.choice(CATEGORIES[s['category']][2])}."
                                                     for e in raises]
            if s["deprecated_by"] and self.visible(lib, s["deprecated_by"]):
                lines += ["", f"{indent}.. deprecated:: use {s['deprecated_by']} instead."]
            lines.append(f'{indent}"""')
            return lines
        lines.append(f"{indent}/**")
        lines.append(f"{indent} * {summary}")
        for p in s["params"]:
            lines.append(f"{indent} * @param {param_display(p['name'], 'typescript')} - the {p['name'].replace('_', ' ')}.")
        if s["kind"] in ("function", "method") and s["returns"] not in ("None", "void"):
            lines.append(f"{indent} * @returns {_a(s['returns'])}.")
        for e in self.shown(s, s["raises"]):
            lines.append(f"{indent} * @throws {{{e}}} if {self.rng.choice(CATEGORIES[s['category']][2])}.")
        lines.append(f"{indent} * @since {s['since']}")
        if s["deprecated_by"] and self.visible(lib, s["deprecated_by"]):
            lines.append(f"{indent} * @deprecated Use {s['deprecated_by']} instead.")
        lines.append(f"{indent} */")
        return lines

    def body(self, s: dict[str, Any], indent: str) -> list[str]:
        py = s["language"] == "python"
        lines = []
        raises = self.shown(s, s["raises"])
        calls = self.shown(s, s["calls"])
        if raises:
            first = s["params"][0]["name"] if s["params"] else "state"
            first = param_display(first, s["language"])
            lines.append(indent + self.comment(s, self.rng.choice(["guard against bad input", "fail early",
                                                                  "validate before doing any work"])))
            if py:
                lines += [f"{indent}if not {first}:", f'{indent}    raise {raises[0]}("{first} is required")']
            else:
                lines += [f"{indent}if (!{first}) {{", f'{indent}  throw new {raises[0]}("{first} is required");', f"{indent}}}"]
        for c in calls:
            target = self.lookup(s["library"], c)
            if target is None:
                continue
            receiver = "self." if py and target["kind"] == "method" and target["owner"] == s["owner"] else \
                ("this." if not py and target["kind"] == "method" and target["owner"] == s["owner"] else "")
            lines.append(indent + self.comment(s, f"{target['purpose'] or 'helper'}"))
            lines.append(f"{indent}{'result = ' if py else 'const result = '}{receiver}{c}({', '.join(param_display(p['name'], s['language']) for p in s['params'][:1])}){'' if py else ';'}")
        if s["returns"] in ("None", "void"):
            lines.append(f"{indent}{'return' if py else 'return;'}")
        else:
            value = "result" if calls else self.value(s, s["returns"])
            lines.append(f"{indent}return {value}{'' if py else ';'}")
        return lines

    def definition(self, s: dict[str, Any], indent: str = "") -> list[str]:
        py = s["language"] == "python"
        inner = indent + ("    " if py else "  ")
        if s["kind"] in ("function", "method"):
            if py:
                return [indent + self.signature(s)] + self.docblock(s, inner) + self.body(s, inner)
            return self.docblock(s, indent) + [indent + self.signature(s) + " {"] + self.body(s, inner) + [indent + "}"]
        if s["kind"] in ("class", "exception"):
            members = [m for m in self.members.get((s["library"], "class:" + s["name"]), []) if self.visible(s["library"], m["name"])]
            if py:
                lines = [indent + self.signature(s)] + self.docblock(s, inner)
                for m in members[:3]:
                    lines += [""] + self.definition(m, inner)
                if not members:
                    lines.append(inner + "pass" if s["kind"] == "exception" else inner + "...")
                return lines
            lines = self.docblock(s, indent) + [indent + self.signature(s) + " {"]
            for m in members[:3]:
                lines += self.definition(m, inner)
            return lines + [indent + "}"]
        return [indent + self.signature(s)]

    # renderers ------------------------------------------------------------------------------------
    def api_entry(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        title = qualname(s)
        lines = [f"{title}", "=" * min(60, len(title)), "", f"{self.code(s['name'])} is {self.where(s)}."]
        if s["kind"] in ("function", "method", "class", "exception", "constant"):
            lines += ["", "```", self.signature(s), "```", ""]
        lines += self.fact_sentences(s)
        if s["params"]:
            lines += ["", "Parameters:"] + [f"- {param_display(p['name'], s['language'])} ({p['type']})"
                                             + (", optional" if p["optional"] else "") for p in s["params"]]
        if s["kind"] in ("class", "exception"):
            members = [m["name"] for m in self.members.get((lib, "class:" + s["name"]), []) if self.visible(lib, m["name"])]
            if members:
                lines += ["", "Methods: " + ", ".join(self.code(m) for m in members) + "."]
        if s["kind"] == "exception":
            raisers = [r["name"] for r in self.raisers.get((lib, s["name"]), []) if self.visible(lib, r["name"])]
            if raisers:
                picked = self.rng.sample(raisers, min(4, len(raisers)))
                lines += ["", f"Raised by: {', '.join(self.code(r) for r in picked)}."]
        if s["kind"] == "module":
            members = [m for m in self.members.get((lib, "module:" + s["name"]), []) if self.visible(lib, m["name"])]
            self.rng.shuffle(members)
            lines += ["", "Members:"] + [f"- {self.code(m['name'])}: {m['kind']}" + (f" that {m['purpose']}" if m["purpose"] and
                                         m["kind"] != "exception" else "") for m in members[:10]]
        lines += ["", self.rng.choice(BOILERPLATE)]
        return "\n".join(lines)

    def source_file(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        anchor = self.lookup(lib, s["owner"]) if s["kind"] == "method" else s
        module = anchor["module"] or s["name"]
        siblings = [m for m in self.members.get((lib, "module:" + module), []) if self.visible(lib, m["name"]) and m is not anchor]
        picked = [anchor] + self.rng.sample(siblings, min(len(siblings), self.rng.randint(1, 3)))
        py = s["language"] == "python"
        path = f"{lib}/{module}.{'py' if py else 'ts'}"
        lines = [f"{'#' if py else '//'} {path}", self.comment(s, f"{CATEGORIES[anchor['category']][0]} helpers for {lib}"), ""]
        by_module: dict[str, list[str]] = {}
        for e in sorted({e for p in picked for e in self.shown(p, p["raises"])}):
            exc = self.lookup(lib, e)
            if exc and exc["module"] != module:
                by_module.setdefault(exc["module"], []).append(e)
        for mod, names in sorted(by_module.items()):
            lines.append(self.import_line({"library": lib, "module": mod, "language": s["language"]}, names))
        lines.append("")
        for p in picked:
            lines += self.definition(p) + [""]
        return "\n".join(lines)

    def stub(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        module = s["name"] if s["kind"] == "module" else s["module"]
        members = [m for m in self.members.get((lib, "module:" + module), []) if self.visible(lib, m["name"])]
        py = s["language"] == "python"
        lines = [f"{'#' if py else '//'} {lib}/{module}.{'pyi' if py else 'd.ts'} (type declarations)"]
        for m in members[:14]:
            if m["kind"] == "function":
                lines.append((self.signature(m) + (" ..." if py else ";")).replace("export function", "export declare function"))
            elif m["kind"] == "constant":
                lines.append(self.signature(m))
            elif m["kind"] in ("class", "exception"):
                lines.append(self.signature(m) + (" ..." if py else " {}"))
                for method in [x for x in self.members.get((lib, "class:" + m["name"]), []) if self.visible(lib, x["name"])][:3]:
                    lines.append(("    " if py else "  ") + self.signature(method) + (" ..." if py else ";"))
        return "\n".join(lines)

    def usage(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        py = s["language"] == "python"
        target = s
        lines = [self.rng.choice(["Example", "Usage", "Quick example", "Recipe"]) + f": {s['purpose'] or 'basic usage'}", ""]
        code: list[str] = []
        if s["kind"] in ("function", "class", "constant", "exception"):
            code.append(self.import_line(s, [s["name"]]))
        if s["kind"] == "method":
            owner = self.lookup(lib, s["owner"])
            if owner:
                code.append(self.import_line(owner, [owner["name"]]))
                code.append(f"{'client = ' if py else 'const client = new '}{owner['name']}(){'' if py else ';'}")
        args = ", ".join(self.value(s, p["type"]) for p in s["params"] if not p["optional"])
        call = f"{'client.' if s['kind'] == 'method' else ''}{s['name']}({args})"
        if s["kind"] == "class":
            call = f"{s['name']}()" if py else f"new {s['name']}()"
        if s["kind"] == "constant":
            call = s["name"]
        raises = self.shown(s, s["raises"])
        assign = "result = " if py else "const result = "
        if raises and py:
            code += ["try:", f"    {assign}{call}", f"except {raises[0]} as err:", f"    {self.comment(s, 'handle the failure')}",
                     "    print(err)"]
        elif raises:
            code += ["try {", f"  {assign}{call};", "} catch (err) {", f"  if (err instanceof {raises[0]}) {{",
                     f"    {self.comment(s, 'handle the failure')}", "  }", "}"]
        else:
            code.append(f"{assign}{call}{'' if py else ';'}")
            if s["returns"] and s["returns"] not in ("None", "void"):
                code.append(self.comment(s, f"result is {_a(s['returns'])}"))
        lines += ["```" + ("python" if py else "ts")] + code + ["```", ""]
        lines += self.fact_sentences(s, limit=2)
        return "\n".join(lines)

    def tutorial(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        related = [self.lookup(lib, n) for n in self.shown(s, s["calls"] + ([s["returns"]] if s["returns"] else []))]
        related = [r for r in related if r is not None and r is not s]
        peers = [m for m in self.members.get((lib, "module:" + (s["module"] or "")), [])
                 if self.visible(lib, m["name"]) and m["category"] == s["category"] and m is not s]
        related += self.rng.sample(peers, min(len(peers), 2))
        topic = CATEGORIES[s["category"]][0] if s["category"] else "getting started"
        lines = [f"Tutorial: {topic} with {lib}", "", f"This guide shows how {self.code(s['name'])} fits into a typical "
                 f"{topic} workflow."]
        for step, r in enumerate([s] + related[:3], start=1):
            sentence = self.rng.choice(self.fact_sentences(r, limit=2) or [f"{self.code(r['name'])} is {self.where(r)}."])
            lines += ["", f"Step {step}. {sentence}"]
            if r["kind"] in ("function", "method", "class"):
                lines += ["```", self.signature(r), "```"]
        lines += ["", self.rng.choice(BOILERPLATE)]
        return "\n".join(lines)

    def qa(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        asker, answerer = self.rng.sample(USERS, 2)
        raises = self.shown(s, s["raises"])
        if raises and self.rng.random() < 0.5:
            question = f"Why does {self.code(s['name'])} raise {self.code(raises[0])}?"
        elif s["purpose"]:
            question = f"Which {lib} API {s['purpose']}?"
        else:
            question = f"What does {self.code(s['name'])} do?"
        answer = " ".join(self.fact_sentences(s, limit=3)) or f"{self.code(s['name'])} is {self.where(s)}."
        lines = [f"Question ({asker}): {question}", "", f"Answer ({answerer}): You want {self.code(s['name'])}, which is "
                 f"{self.where(s)}. {answer}"]
        if s["kind"] in ("function", "method"):
            lines += ["", "```", self.signature(s), "```"]
        lines += ["", f"{asker}: thanks, that fixed it."]
        return "\n".join(lines)

    def issue(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        py = s["language"] == "python"
        raises = self.shown(s, s["raises"]) or [e["name"] for e in self.of_kind.get((lib, "exception"), [])[:1]]
        callers = [c for c in self.callers.get((lib, s["name"]), []) if self.visible(lib, c["name"])]
        caller = self.rng.choice(callers) if callers else None
        number = self.rng.randint(100, 9999)
        lines = [f"Issue #{number}: {self.code(s['name'])} fails" + (f" with {raises[0]}" if raises else ""),
                 "", f"Version: {lib} {s['since']}", "", "Steps to reproduce: call it from our service.", ""]
        error = raises[0] if raises else ("ValueError" if py else "TypeError")
        if py:
            lines.append("Traceback (most recent call last):")
            if caller:
                lines += [f'  File "{lib}/{caller["module"]}.py", line {self.rng.randint(10, 400)}, in {caller["name"]}',
                          f"    {s['name']}(...)"]
            lines += [f'  File "{lib}/{s["module"] or s["name"]}.py", line {self.rng.randint(10, 400)}, in {s["name"]}',
                      f"    raise {error}(...)", f"{error}: {self.rng.choice(CATEGORIES[s['category']][2])}"]
        else:
            lines.append(f"{error}: {self.rng.choice(CATEGORIES[s['category']][2])}")
            lines.append(f"    at {s['name']} ({lib}/{s['module'] or s['name']}.ts:{self.rng.randint(10, 400)}:{self.rng.randint(2, 40)})")
            if caller:
                lines.append(f"    at {caller['name']} ({lib}/{caller['module']}.ts:{self.rng.randint(10, 400)}:{self.rng.randint(2, 40)})")
        lines += ["", "Maintainer reply: " + (" ".join(self.fact_sentences(s, limit=2)) or "Thanks for the report."),
                  self.rng.choice(BOILERPLATE)]
        return "\n".join(lines)

    def changelog(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        peers = [p for p in self.by_version.get((lib, s["since"]), []) if self.visible(lib, p["name"]) and p is not s]
        picked = [s] + self.rng.sample(peers, min(len(peers), self.rng.randint(2, 5)))
        lines = [f"{lib} {s['since']} release notes", ""]
        for p in picked:
            roll = self.rng.random()
            if p["deprecated_by"] and self.visible(lib, p["deprecated_by"]):
                lines.append(f"- Deprecated {self.code(p['name'])}; use {self.code(p['deprecated_by'])} instead.")
            elif roll < 0.5:
                where = f" to {self.code(p['owner'] or p['module'])}" if p["kind"] not in ("package", "module") else ""
                lines.append(f"- Added {self.code(p['name'])}{where}.")
            elif roll < 0.75 and self.shown(p, p["raises"]):
                lines.append(f"- {self.code(p['name'])} now raises {self.code(self.shown(p, p['raises'])[0])} instead of failing silently.")
            else:
                sentence = self.fact_sentences(p, limit=1)
                lines.append(f"- Fixed a bug in {self.code(p['name'])}." + (f" {sentence[0]}" if sentence else ""))
        return "\n".join(lines)

    def migration(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        new = self.lookup(lib, s["deprecated_by"]) if s["deprecated_by"] and self.visible(lib, s["deprecated_by"]) else None
        if new is None:
            return self.api_entry(s)
        py = s["language"] == "python"
        old_call = f"{s['name']}({', '.join(self.value(s, p['type']) for p in s['params'] if not p['optional'])})"
        new_call = f"{new['name']}({', '.join(self.value(new, p['type']) for p in new['params'] if not p['optional'])})"
        return "\n".join([f"Migration guide: {self.code(s['name'])} → {self.code(new['name'])}", "",
                          f"{self.code(s['name'])} is deprecated since {lib} {self.rng.choice(self.libs[lib]['versions'])}. "
                          f"Replace it with {self.code(new['name'])}, which {new['purpose'] or 'does the same job'}.", "",
                          "Before:", "```", ("" if py else "const r = ") + old_call, "```", "After:", "```",
                          ("" if py else "const r = ") + new_call, "```", "", self.rng.choice(BOILERPLATE)])

    def review(self, s: dict[str, Any]) -> str:
        lib = s["library"]
        reviewer = self.rng.choice(USERS)
        sentences = self.fact_sentences(s, limit=2)
        hint = sentences[0] if sentences else f"{self.code(s['name'])} is {self.where(s)}."
        kind = self.rng.choice(["Code review", "Commit message", "Pull request comment"])
        line = self.rng.randint(5, 300)
        return (f"{kind} ({reviewer}) on {lib}/{s['module'] or s['name']}, line {line}:\n"
                f"Please use {self.code(s['name'])} here. {hint} " + self.rng.choice(BOILERPLATE))

    def readme(self, s: dict[str, Any]) -> str:
        lib = self.libs[s["library"]]
        py = lib["language"] == "python"
        mods = self.of_kind.get((lib["name"], "module"), [])
        lines = [f"# {lib['name']}", "", f"{lib['name']} is a {'Python' if py else 'TypeScript'} library for "
                 + ", ".join(CATEGORIES[c][0] for c in lib["categories"]) + ".", "", "## Installation", "", "```",
                 f"{'pip install' if py else 'npm install'} {lib['name']}", "```", "", "## Modules", ""]
        lines += [f"- {self.code(m['name'])}: {CATEGORIES[m['category']][0]}" for m in mods]
        heads = sorted((x for x in self.of_kind.get((lib["name"], "function"), []) if self.visible(lib["name"], x["name"])),
                       key=lambda x: x["rank"])[:4]
        if heads:
            lines += ["", "## Quickstart", ""]
            for h in heads:
                lines += [f"{self.code(h['name'])} {h['purpose']}.", "```", self.import_line(h, [h["name"]]), "```"]
        lines += ["", f"Current version: {lib['versions'][-1]}.", self.rng.choice(BOILERPLATE)]
        return "\n".join(lines)

    def render(self, s: dict[str, Any]) -> str:
        kind = s["kind"]
        if kind == "package":
            choices = [(self.readme, 0.6), (self.changelog, 0.4)]
        elif kind == "module":
            choices = [(self.api_entry, 0.45), (self.stub, 0.35), (self.tutorial, 0.2)]
        elif kind == "exception":
            choices = [(self.api_entry, 0.35), (self.qa, 0.25), (self.issue, 0.25), (self.changelog, 0.15)]
        elif kind == "constant":
            choices = [(self.api_entry, 0.5), (self.usage, 0.3), (self.changelog, 0.2)]
        elif kind == "class":
            choices = [(self.api_entry, 0.25), (self.source_file, 0.2), (self.tutorial, 0.15), (self.usage, 0.15),
                       (self.qa, 0.1), (self.stub, 0.05), (self.changelog, 0.1)]
        else:
            choices = [(self.api_entry, 0.2), (self.source_file, 0.15), (self.usage, 0.15), (self.tutorial, 0.1),
                       (self.qa, 0.1), (self.issue, 0.08), (self.changelog, 0.07), (self.stub, 0.05), (self.review, 0.05)]
            if s["deprecated_by"]:
                choices.append((self.migration, 0.1))
        roll = self.rng.random() * sum(w for _, w in choices)
        for renderer, weight in choices:
            roll -= weight
            if roll <= 0:
                return renderer(s)
        return choices[-1][0](s)


def hidden_names(data: dict[str, Any], split: str) -> frozenset[tuple[str, str]]:
    """(library, name) pairs a `split` document may not mention: zero-shot symbols always, held-out
    symbols in training documents."""
    hide = {"zeroshot"} | ({"heldout"} if split == "train" else set())
    names = {(s["library"], s["name"]) for s in data["symbols"] if s["split"] in hide}
    # an overriding method shares its name with its base: hide the name only if every bearer is hidden
    bearers: dict[tuple[str, str], set[str]] = {}
    for s in data["symbols"]:
        bearers.setdefault((s["library"], s["name"]), set()).add(s["split"])
    return frozenset(n for n in names if bearers[n] <= hide)


def library_documents(data: dict[str, Any], *, split: str, seed: int, max_chars: int,
                      uniform_focus: float = 0.0) -> Iterator[str]:
    """Training (`split="train"`) or evaluation documents until `max_chars` characters (see the
    module docstring). Training documents never mention held-out or zero-shot symbols; evaluation
    documents never mention zero-shot symbols."""
    rng = random.Random(seed * 1_000_003 + (1 if split == "train" else 2))
    hidden = hidden_names(data, split)
    writer = _Writer(data, hidden, rng)
    pool = [s for s in writer.symbols if (s["library"], s["name"]) not in hidden]
    cumulative = list(accumulate(s["weight"] for s in pool))
    produced = 0
    while produced < max_chars:
        if uniform_focus and rng.random() < uniform_focus:
            focus = pool[rng.randrange(len(pool))]
        else:
            focus = pool[min(len(pool) - 1, bisect_right(cumulative, rng.random() * cumulative[-1]))]
        text = writer.render(focus)
        produced += len(text)
        yield text


# -- alias matching and the leakage audit -----------------------------------------------------------

_RUN = re.compile(r"[a-z0-9_.$]+")


class AliasMatcher:
    """Occurrences of identifier-mode aliases (lower-case letters, digits, `_`, `.` and `$`; no spaces)
    under the linker's matching rule: an alias starts at a word boundary (the previous character is not
    a letter, digit or underscore) and the right boundary is not checked (prefix-causal linking). Counts
    are a superset of linked spans (a span also needs a token boundary where the alias ends)."""

    def __init__(self, aliases: Iterable[str]) -> None:
        self.aliases = {a.lower() for a in aliases}
        bad = [a for a in self.aliases if not re.fullmatch(r"[a-z0-9_.$]+", a)]
        if bad:
            raise ValueError(f"aliases must be identifier paths (letters, digits, _ . $): {bad[:5]}")
        self.prefixes = {a[:k] for a in self.aliases for k in range(1, len(a) + 1)}
        self._cache: dict[str, tuple[str, ...]] = {}

    def in_run(self, run: str) -> tuple[str, ...]:
        found = self._cache.get(run)
        if found is not None:
            return found
        hits = []
        starts = [0] + [i + 1 for i, c in enumerate(run) if c in ".$"]
        for start in starts:
            prefix = ""
            for char in run[start:]:
                prefix += char
                if prefix not in self.prefixes:
                    break
                if prefix in self.aliases:
                    hits.append(prefix)
        found = tuple(hits)
        if len(self._cache) < 4_000_000:
            self._cache[run] = found
        return found

    def count(self, text: str, counts: Counter | None = None) -> Counter:
        counts = Counter() if counts is None else counts
        for run, n in Counter(_RUN.findall(text.lower())).items():
            for alias in self.in_run(run):
                counts[alias] += n
        return counts

    def lines_without(self, text: str) -> str:
        """`text` without the lines that contain an alias occurrence."""
        return "\n".join(line for line in text.split("\n") if not any(self.in_run(r) for r in _RUN.findall(line.lower())))


def private_aliases(symbol: dict[str, Any]) -> list[str]:
    return [symbol["name"]]


def leakage_audit(data: dict[str, Any], train_texts: Iterable[str]) -> dict[str, Any]:
    """No alias of a held-out or zero-shot symbol may occur in training text under the linker's rule,
    and no stem of one may occur as a whole word."""
    secret = [s for s in data["symbols"] if s["split"] in ("heldout", "zeroshot")]
    hidden = hidden_names(data, "train")
    secret = [s for s in secret if (s["library"], s["name"]) in hidden]
    matcher = AliasMatcher(a.lower() for s in secret for a in private_aliases(s))
    stems = {stem.lower(): s["name"] for s in secret for stem in s["stems"]}
    counts: Counter = Counter()
    words: set[str] = set()
    for text in train_texts:
        matcher.count(text, counts)
        words.update(_WORDS.findall(text.lower()))
    leaked = sorted({s["name"] for s in secret if counts[s["name"].lower()]} | {stems[w] for w in words & set(stems)})
    return {"heldout_symbols": sum(s["split"] == "heldout" for s in data["symbols"]),
            "zeroshot_symbols": sum(s["split"] == "zeroshot" for s in data["symbols"]), "leaked_into_train": leaked}
