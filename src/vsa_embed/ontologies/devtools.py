"""Developer tools (track T2) as a frame ontology: private libraries plus real API symbols.

Concepts are API symbols: the private libraries' symbols (`benchmarks/devtools_libraries.py`; training
and held-out symbols — zero-shot symbols are added by the track builder as synthetic concepts) and the
real symbols of `data/api_docs.py` (installed Python packages, the standard library, Node.js). Both
share one relation set and one atomic dictionary, so a parameter name ("timeout"), a builtin type
("int") or a kind ("method") is the same atom in both:

    kind · library · language · member_of · category · purpose · returns · takes · has_param · raises
    inherits · overrides · deprecated_by · calls · since · status

Atoms: fixed (kinds, libraries, languages, categories, purposes, statuses, `type:other`) plus the
fillers most used by any concept (zero-shot symbols included, so their frames resolve) — symbols
(`symbol:<path>`), modules (`module:<path>`), types (`type:<name>`), parameters (`param:<name>`) and
versions (`version:<library> <version>`) — up to `max_atomics`. A filler outside the dictionary falls
back to a coarser atom (a symbol to its kind, a module to its library, a type to `type:other`;
parameters and versions are dropped), as WordNet/MeSH fall back to an ancestor. Frames keep at most
`max_degree` edges in a fixed relation priority. Aliases: the identifier of a private symbol; the
aliases of a real symbol (`data/api_docs.py`).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from ..benchmarks.devtools_libraries import (BUILTIN_TYPES, CATEGORIES, KINDS, LANGUAGES, RELATIONS, STATUSES,
                                             qualname)
from .wordnet import FrameOntology

PRIORITY = ["kind", "library", "member_of", "category", "purpose", "returns", "raises", "inherits", "overrides",
            "deprecated_by", "takes", "has_param", "status", "since", "calls", "language"]
REAL_KINDS = ["property"]
_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def param_atom(name: str) -> str:
    """`param:<snake_case name>` (camelCase and `*args` normalised, so both languages share atoms)."""
    return "param:" + _SNAKE.sub("_", name.lstrip("*")).lower()


def real_canonical(record: dict[str, Any]) -> str:
    """Display name of a real symbol: its shortest dotted alias under the library's own name, else the
    shortest dotted alias, else the shortest alias."""
    aliases = sorted(record["aliases"], key=lambda a: (len(a), a))
    top = record["path"].split(":")[-1].split(".")[0]
    own = [a for a in aliases if "." in a and a.split(".")[0] == top]
    dotted = [a for a in aliases if "." in a]
    return (own or dotted or aliases)[0]


def private_concept(symbol: dict[str, Any]) -> str:
    return f"private:{qualname(symbol)}"


def real_concept(record: dict[str, Any]) -> str:
    return f"real:{record['path']}"


class _PrivateIndex:
    def __init__(self, symbols: list[dict[str, Any]]) -> None:
        self.by_name: dict[tuple[str, str], dict[str, Any]] = {}
        for s in symbols:
            self.by_name.setdefault((s["library"], s["name"]), s)
        self.methods = {(s["library"], s["owner"], s["name"]): s for s in symbols if s["kind"] == "method"}

    def symbol_atom(self, library: str, name: str) -> str | None:
        s = self.by_name.get((library, name))
        return f"symbol:{qualname(s)}" if s else None


def private_frame(symbol: dict[str, Any], index: _PrivateIndex) -> list[tuple[str, str]]:
    """(relation, atom name) edges of a private symbol, in `PRIORITY` order."""
    s, lib = symbol, symbol["library"]
    builtin = set(BUILTIN_TYPES[s["language"]]) | {"Exception", "Error"}
    def type_atom(name: str | None) -> str | None:
        if not name:
            return None
        return f"type:{name}" if name in builtin else index.symbol_atom(lib, name)
    edges: dict[str, list[str | None]] = {r: [] for r in PRIORITY}
    edges["kind"] = [f"kind:{s['kind']}"]
    edges["library"] = [f"library:{lib}"]
    edges["language"] = [f"language:{s['language']}"]
    if s["kind"] == "module":
        edges["member_of"] = [f"library:{lib}"]
    elif s["owner"]:
        owner = index.by_name.get((lib, s["owner"]))
        edges["member_of"] = [f"symbol:{qualname(owner)}" if owner else f"symbol:{lib}.{s['module']}.{s['owner']}"]
    elif s["module"]:
        edges["member_of"] = [f"module:{lib}.{s['module']}"]
    if s["category"]:
        edges["category"] = [f"category:{s['category']}"]
    if s["purpose"] and s["kind"] != "exception":
        edges["purpose"] = [f"purpose:{s['purpose']}"]
    if s["kind"] in ("function", "method", "constant"):
        edges["returns"] = [type_atom(s["returns"])]
    edges["takes"] = [type_atom(p["type"]) for p in s["params"]]
    edges["has_param"] = [param_atom(p["name"]) for p in s["params"]]
    edges["raises"] = [index.symbol_atom(lib, e) for e in s["raises"]]
    edges["inherits"] = [type_atom(s["inherits"])]
    if s["overrides"]:
        owner, name = s["overrides"].split(".")
        base = index.methods.get((lib, owner, name))
        edges["overrides"] = [f"symbol:{qualname(base)}" if base else None]
    edges["deprecated_by"] = [index.symbol_atom(lib, s["deprecated_by"])] if s["deprecated_by"] else []
    edges["calls"] = [index.symbol_atom(lib, c) for c in s["calls"]]
    edges["since"] = [f"version:{lib} {s['since']}"]
    edges["status"] = [f"status:{s['status']}"]
    return [(r, a) for r in PRIORITY for a in dict.fromkeys(edges[r]) if a]


class _RealIndex:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.by_path = {r["path"]: r for r in records}
        self.by_alias: dict[str, dict[str, Any]] = {}
        self.by_tail: dict[str, list[dict[str, Any]]] = {}
        for r in records:
            for a in r["aliases"]:
                self.by_alias.setdefault(a, r)
        self.canonical = {r["path"]: real_canonical(r) for r in records}

    def find(self, name: str | None) -> dict[str, Any] | None:
        if not name:
            return None
        return self.by_path.get(name) or self.by_alias.get(name) or self.by_path.get(f"node:{name}")

    def symbol_atom(self, record: dict[str, Any]) -> str:
        return f"symbol:{self.canonical[record['path']]}"

    def type_atom(self, name: str | None) -> str | None:
        """A known symbol, else a bare builtin-like type name; unions and generics keep their head."""
        if not name:
            return None
        name = name.strip().strip("'\"")
        head = re.split(r"[\[|, ]", name)[0].strip()
        if not head or head in ("None", "object", "Any", "Optional", "Union", "undefined", "null"):
            return "type:None" if head == "None" else None
        found = self.find(head)
        if found is not None and found["kind"] in ("class", "exception"):
            return self.symbol_atom(found)
        tail = head.split(".")[-1]
        return f"type:{tail}" if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tail) else None


def real_module(record: dict[str, Any], canonical: str) -> str | None:
    head = canonical.rpartition(".")[0]
    return head or record.get("module")


def real_frame(record: dict[str, Any], index: _RealIndex) -> list[tuple[str, str]]:
    r = record
    canonical = index.canonical[r["path"]]
    edges: dict[str, list[str | None]] = {k: [] for k in PRIORITY}
    edges["kind"] = [f"kind:{r['kind']}"]
    edges["library"] = [f"library:{r['library']}"]
    edges["language"] = [f"language:{r['language']}"]
    owner = index.find(r.get("owner"))
    if owner is not None:
        edges["member_of"] = [index.symbol_atom(owner)]
    else:
        module = real_module(r, canonical)
        edges["member_of"] = [f"module:{module}" if module else None]
    if r["kind"] in ("function", "method", "property"):
        edges["returns"] = [index.type_atom(r.get("returns"))]
    edges["takes"] = [index.type_atom(p.get("annotation")) for p in r.get("params", [])]
    edges["has_param"] = [param_atom(p["name"]) for p in r.get("params", []) if p["name"].lstrip("*")]
    edges["raises"] = [index.type_atom(e) for e in r.get("raises", [])]
    edges["inherits"] = [index.type_atom(b) for b in r.get("bases", []) if b not in ("object", "Generic", "Protocol")]
    overridden = index.find(r.get("overrides"))
    edges["overrides"] = [index.symbol_atom(overridden)] if overridden else []
    replacement = index.find(r.get("deprecated_by"))
    edges["deprecated_by"] = [index.symbol_atom(replacement)] if replacement else []
    if r.get("since"):
        edges["since"] = [f"version:{r['library']} {r['since']}"]
    edges["status"] = [f"status:{r['status']}"]
    return [(k, a) for k in PRIORITY for a in dict.fromkeys(edges[k]) if a]


def fixed_atoms(libraries: Iterable[str]) -> list[str]:
    purposes = [p for _, verbs, _, _ in CATEGORIES.values() for p in verbs]
    return ([f"kind:{k}" for k in KINDS + REAL_KINDS] + [f"library:{lib}" for lib in libraries]
            + [f"language:{lang}" for lang in LANGUAGES + ["javascript"]] + [f"category:{c}" for c in CATEGORIES]
            + [f"purpose:{p}" for p in dict.fromkeys(purposes)] + [f"status:{s}" for s in STATUSES] + ["type:other"])


def build_devtools_ontology(private: dict[str, Any], real: list[dict[str, Any]], *, real_heldout: Iterable[str] = (),
                            max_atomics: int = 8192, max_degree: int = 16) -> FrameOntology:
    """Concepts: private training + held-out symbols, then real symbols. `metadata` keeps per-concept
    `splits`, `sources` and `surfaces`, and `zeroshot_frames` (qualified name → resolved edges) for the
    private zero-shot symbols, which the track builder adds as synthetic concepts."""
    held_real = set(real_heldout)
    p_index = _PrivateIndex([s for s in private["symbols"]])
    r_index = _RealIndex(real)
    concepts = [s for s in private["symbols"] if s["split"] != "zeroshot"]
    zero = [s for s in private["symbols"] if s["split"] == "zeroshot"]
    frames_by_name = [private_frame(s, p_index) for s in concepts] + [real_frame(r, r_index) for r in real]
    zero_frames = {qualname(s): private_frame(s, p_index) for s in zero}
    kind_of: dict[str, str] = {}
    library_of: dict[str, str] = {}
    for s in private["symbols"]:
        kind_of[f"symbol:{qualname(s)}"] = s["kind"]
        if s["kind"] == "module":
            library_of[f"module:{qualname(s)}"] = s["library"]
    for r in real:
        kind_of[r_index.symbol_atom(r)] = r["kind"]
        module = real_module(r, r_index.canonical[r["path"]])
        if module:
            library_of[f"module:{module}"] = r["library"]
    libraries = sorted({lib["name"] for lib in private["libraries"]} | {r["library"] for r in real})
    fixed = fixed_atoms(libraries)
    fixed_set = set(fixed)
    use: Counter[str] = Counter(a for frame in list(frames_by_name) + list(zero_frames.values()) for _, a in frame
                                if a not in fixed_set)
    budget = max(0, max_atomics - len(fixed))
    chosen = [a for a, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = fixed + chosen
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_index = {name: i for i, name in enumerate(RELATIONS)}
    replaced = Counter()

    def fit(frame: list[tuple[str, str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for relation, atom in frame:
            if atom not in atomic_index:
                prefix = atom.split(":", 1)[0]
                fallback = {"symbol": f"kind:{kind_of.get(atom, 'class')}", "module": f"library:{library_of.get(atom, '')}",
                            "type": "type:other"}.get(prefix)
                replaced[prefix] += 1
                if fallback not in atomic_index:
                    continue
                atom = fallback
            if (relation, atom) not in out and len(out) < max_degree:
                out.append((relation, atom))
        return out

    named = [fit(f) for f in frames_by_name]
    frames = [[(relation_index[r], atomic_index[a]) for r, a in f] for f in named]
    names = [private_concept(s) for s in concepts] + [real_concept(r) for r in real]
    aliases = [(s["name"], i) for i, s in enumerate(concepts)]
    aliases += [(a, len(concepts) + j) for j, r in enumerate(real) for a in r["aliases"]]
    ontology = FrameOntology("devtools", names, list(RELATIONS), atomic_names, frames, aliases,
                             {"seed": private["seed"], "private_symbols": len(concepts), "real_symbols": len(real),
                              "max_atomics": max_atomics, "max_degree": max_degree,
                              "fillers_replaced": dict(sorted(replaced.items()))})
    ontology.metadata["splits"] = [s["split"] for s in concepts] + ["heldout" if r["path"] in held_real else "train" for r in real]
    ontology.metadata["sources"] = ["private"] * len(concepts) + ["real"] * len(real)
    ontology.metadata["surfaces"] = [s["name"] for s in concepts] + [r_index.canonical[r["path"]] for r in real]
    ontology.metadata["zeroshot_frames"] = {name: fit(frame) for name, frame in zero_frames.items()}
    return ontology
