"""Real API documentation for the developer-tools track (T2): symbols with schemas, and documentation units.

Three open sources, read without importing or executing anything:

- **Installed Python packages** (permissive licences, recorded per package from the distribution
  metadata): every module file is parsed with `ast`. A symbol is a public class, exception, function,
  method or property with a docstring; its schema is read from the code — parameters and their
  annotations, return annotation (or the numpydoc/Google "Returns" type), raised exceptions (`raise`
  statements and "Raises" sections), base classes (resolved through the module's imports), overridden
  base methods, `.. deprecated::` / "use X instead", `.. versionadded::`. Its documentation unit is the
  qualified signature line, the docstring and the comment lines of its body ("code comments").
- **The CPython 3.12 documentation** (text build, PSF licence): pages split into sections; the
  Sphinx inventory (`objects.inv`) lists the documented standard-library objects, which define the
  standard-library symbols (implemented in C or in Python; AST details are attached when found).
- **The Node.js v20 API** (`all.json`, MIT): modules, classes and methods with typed parameters,
  return types, stability and the version that added them.

**Aliases** are public dotted paths, linked with the `identifier` alias normalization
(`span_channel.ALIAS_NORMALIZATIONS`, which keeps `_`): the definition path when every component is
public ("numpy.linalg.solve"), re-exports ("torch.nn.Module" for a class defined in a private module),
the conventional import names ("np.", "pd.", "nx.", "plt.", "nn."), plus a bare CamelCase class name
("OrderedDict") when it is not an English word and names one symbol only. Undotted names ("any",
Node's "require") are never aliases; a component starting with `_` is private. An alias naming several
symbols is dropped, and a symbol without an alias is dropped.
"""

from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import html
import io
import json
import re
import sys
import tarfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

PY_CONVENTIONS = {"numpy": "np", "pandas": "pd", "networkx": "nx", "matplotlib.pyplot": "plt", "torch.nn": "nn"}
STDLIB_SKIP = {"test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", "venv", "pydoc_data", "site-packages",
               "lib-dynload", "config-3.12-x86_64-linux-gnu", "__pycache__", "__phello__", "antigravity", "this"}
SKIP_DIRS = {"tests", "test", "testing", "_testing", "benchmarks", "__pycache__", "conftest"}
DOC_SECTIONS = ("library", "tutorial", "howto", "reference", "faq", "whatsnew", "deprecations", "using", "installing",
                "distributing", "glossary.txt")
INVENTORY_KINDS = {"py:function": "function", "py:class": "class", "py:exception": "exception", "py:method": "method",
                   "py:classmethod": "method", "py:staticmethod": "method", "py:module": "module"}
EXCEPTION_SUFFIXES = ("Error", "Exception", "Warning", "Exit", "Interrupt")
_CLEAN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*")
_CAMEL = re.compile(r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*")
_HEADING = re.compile(r"^(=+|-+|\*+|~+|\^+|\"+|#+)$")


def clean_path(path: str) -> bool:
    """A public identifier path: every dotted component starts with a letter (`_x` is private)."""
    return bool(_CLEAN.fullmatch(path))


# -- Python sources ---------------------------------------------------------------------------------

def package_files(root: Path, package: str) -> Iterator[tuple[str, Path]]:
    """(dotted module name, file) of a package directory, tests skipped, sorted."""
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent)
        parts = list(rel.with_suffix("").parts)
        if any(p in SKIP_DIRS or p.startswith("test_") for p in parts[:-1]) or parts[-1].startswith("test_"):
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if all(p.isidentifier() for p in parts):
            yield ".".join(parts), path


def stdlib_files(lib_dir: Path) -> Iterator[tuple[str, Path]]:
    """Standard-library module files (test and GUI packages skipped)."""
    for path in sorted(lib_dir.glob("*.py")):
        if path.stem not in STDLIB_SKIP and path.stem.isidentifier():
            yield path.stem, path
    for directory in sorted(p for p in lib_dir.iterdir() if p.is_dir() and p.name not in STDLIB_SKIP and p.name.isidentifier()):
        if (directory / "__init__.py").exists():
            yield from package_files(directory, directory.name)


def _resolve_relative(module: str, is_package: bool, target: str | None, level: int) -> str:
    if level == 0:
        return target or ""
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    base = base[:len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([target] if target else []))


def _module_statements(tree: ast.Module) -> Iterator[ast.stmt]:
    """Module-level statements, including those inside `if`/`try` blocks (e.g. TYPE_CHECKING imports)."""
    stack = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.Try)):
            children = list(node.body) + list(node.orelse)
            if isinstance(node, ast.Try):
                children += [s for h in node.handlers for s in h.body] + list(node.finalbody)
            stack.extend(reversed(children))


def _docstring_section(doc: str, names: tuple[str, ...]) -> list[str]:
    """Lines of a numpydoc ("Raises\\n------") or Google ("Raises:") section."""
    lines = doc.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        numpy_style = stripped in names and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}
        google_style = stripped.rstrip(":") in names and stripped.endswith(":")
        if not (numpy_style or google_style):
            continue
        start = i + 2 if numpy_style else i + 1
        for follow in lines[start:]:
            if not follow.strip():
                if out:
                    break
                continue
            if follow.strip() and set(follow.strip()) == {"-"}:
                out = out[:-1]
                break
            out.append(follow)
        break
    return out


def docstring_raises(doc: str) -> list[str]:
    names = []
    for line in _docstring_section(doc, ("Raises",)):
        match = re.match(r"\s*:?(?:exc:)?`*~?([A-Za-z_][\w.]*)`*\s*(?::|$|\s)", line)
        if match and match.group(1).split(".")[-1].endswith(EXCEPTION_SUFFIXES):
            names.append(match.group(1))
    return names


def docstring_returns(doc: str) -> str | None:
    lines = _docstring_section(doc, ("Returns",))
    if not lines:
        return None
    first = lines[0].strip()
    match = re.match(r"(?:[A-Za-z_]\w*\s*:\s*)?([A-Za-z_][\w.]*)", first)
    return match.group(1) if match else None


def docstring_deprecation(doc: str) -> tuple[bool, str | None]:
    lower = doc.lower()
    position = lower.find("deprecat")
    if position < 0:
        return False, None
    head = doc[max(0, position - 40):position + 300]
    match = re.search(r"(?:[Uu]se|in favou?r of|replaced by|[Ss]ee)\s+[`'\":]*(?:func:|meth:|class:|obj:)?[`'\"]*~?"
                      r"([A-Za-z_][\w.]*[A-Za-z0-9])", head)
    deprecated = ".. deprecated::" in doc or "deprecated" in lower[:200]
    return deprecated, (match.group(1) if match and deprecated else None)


def docstring_since(doc: str) -> str | None:
    match = re.search(r"\.\. versionadded::\s*v?(\d+(?:\.\d+)?)", doc)
    return match.group(1) if match else None


def _raises_in_body(node: ast.AST) -> list[str]:
    found: list[str] = []
    stack = list(getattr(node, "body", []))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.Raise) and child.exc is not None:
            target = child.exc.func if isinstance(child.exc, ast.Call) else child.exc
            if isinstance(target, (ast.Name, ast.Attribute)):
                name = ast.unparse(target)
                if name.split(".")[-1][:1].isupper():
                    found.append(name)
        stack.extend(ast.iter_child_nodes(child))
    return sorted(set(found))


def _comments(lines: list[str], start: int, end: int, skip: list[tuple[int, int]]) -> list[str]:
    out = []
    for number in range(start, end + 1):
        if any(a <= number <= b for a, b in skip):
            continue
        text = lines[number - 1].strip() if number - 1 < len(lines) else ""
        if not text.startswith("#"):
            continue
        body = text.lstrip("#").strip()
        if len(body.split()) < 3 or body.lower().startswith(("noqa", "type:", "pragma", "pylint", "fmt:", "isort", "mypy")):
            continue
        out.append("# " + body)
    return out[:20]


def _params(args: ast.arguments) -> list[dict[str, Any]]:
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params = []
    for arg, default in zip(positional, defaults):
        params.append({"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
                       "optional": default is not None})
    if args.vararg:
        params.append({"name": "*" + args.vararg.arg, "annotation": None, "optional": True})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append({"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None,
                       "optional": default is not None})
    if args.kwarg:
        params.append({"name": "**" + args.kwarg.arg, "annotation": None, "optional": True})
    return [p for p in params if p["name"] not in ("self", "cls")]


def _signature(args: ast.arguments) -> str:
    args = copy.deepcopy(args)
    if args.posonlyargs and args.posonlyargs[0].arg in ("self", "cls"):
        args.posonlyargs = args.posonlyargs[1:]
    elif args.args and args.args[0].arg in ("self", "cls"):
        args.args = args.args[1:]
    return "(" + ast.unparse(args) + ")"


def parse_module(module: str, path: Path, *, is_package: bool) -> dict[str, Any] | None:
    """Imports, re-exports and public definitions of one module file."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    lines = source.splitlines()
    imports: dict[str, str] = {}
    stars: list[str] = []
    exported: list[str] | None = None
    for node in _module_statements(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    imports[alias.asname] = alias.name
                else:
                    imports[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(module, is_package, node.module, node.level)
            for alias in node.names:
                if alias.name == "*":
                    stars.append(base)
                else:
                    imports[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                exported = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    definitions = []
    local_names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    def resolve(name: str | None) -> str | None:
        if not name:
            return None
        head, _, rest = name.partition(".")
        if head in imports:
            return imports[head] + ("." + rest if rest else "")
        if head in local_names:
            return f"{module}.{name}"
        return name

    def doc_record(node: ast.AST) -> tuple[str | None, list[tuple[int, int]]]:
        doc = ast.get_docstring(node, clean=True)
        body = getattr(node, "body", [])
        span = [(body[0].lineno, body[0].end_lineno)] if doc and body else []
        return doc, span

    def function_record(node: ast.FunctionDef | ast.AsyncFunctionDef, path_: str, owner: str | None) -> dict[str, Any]:
        doc, span = doc_record(node)
        decorators = [ast.unparse(d) for d in node.decorator_list]
        deprecated, replacement = docstring_deprecation(doc or "")
        returns = ast.unparse(node.returns) if node.returns else docstring_returns(doc or "")
        return {"path": path_, "kind": "property" if "property" in decorators else ("method" if owner else "function"),
                "module": module, "owner": owner, "name": node.name, "params": _params(node.args),
                "signature": _signature(node.args), "returns": resolve(returns) if returns else None,
                "raises": sorted({resolve(r) for r in _raises_in_body(node) + docstring_raises(doc or "")}),
                "deprecated": deprecated or any("deprecat" in d for d in decorators), "deprecated_by": replacement,
                "since": docstring_since(doc or ""), "doc": doc,
                "comments": _comments(lines, node.lineno, node.end_lineno or node.lineno, span)}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            definitions.append(function_record(node, f"{module}.{node.name}", None))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            doc, span = doc_record(node)
            class_path = f"{module}.{node.name}"
            bases = [resolve(ast.unparse(b)) for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))]
            methods, method_spans = [], []
            init_params, init_signature = [], "()"
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_spans.append((child.lineno, child.end_lineno or child.lineno))
                    if child.name == "__init__":
                        init_params, init_signature = _params(child.args), _signature(child.args)
                    elif not child.name.startswith("_"):
                        methods.append(function_record(child, f"{class_path}.{child.name}", class_path))
            deprecated, replacement = docstring_deprecation(doc or "")
            is_exception = node.name.endswith(EXCEPTION_SUFFIXES) or any(
                (b or "").split(".")[-1].endswith(EXCEPTION_SUFFIXES) or (b or "") in ("Exception", "BaseException") for b in bases)
            definitions.append({"path": class_path, "kind": "exception" if is_exception else "class", "module": module,
                                "owner": None, "name": node.name, "bases": bases, "params": init_params,
                                "signature": init_signature, "returns": None, "raises": [], "deprecated": deprecated,
                                "deprecated_by": replacement, "since": docstring_since(doc or ""), "doc": doc,
                                "comments": _comments(lines, node.lineno, node.end_lineno or node.lineno, span + method_spans)})
            definitions += methods
    return {"module": module, "doc": ast.get_docstring(tree, clean=True), "imports": imports, "stars": stars,
            "exported": exported, "definitions": definitions}


class PythonIndex:
    """Definitions and public paths across parsed modules (re-exports resolved)."""

    def __init__(self, modules: Iterable[dict[str, Any]]) -> None:
        self.modules = {m["module"]: m for m in modules}
        self.defs = {d["path"]: d for m in self.modules.values() for d in m["definitions"]}
        self._memo: dict[str, str | None] = {}

    def public_names(self, module: str, depth: int = 0) -> list[str]:
        m = self.modules.get(module)
        if m is None or depth > 4:
            return []
        if m["exported"] is not None:
            return [n for n in m["exported"] if not n.startswith("_")]
        names = [d["name"] for d in m["definitions"] if d["owner"] is None]
        names += [n for n in m["imports"] if not n.startswith("_")]
        for star in m["stars"]:
            names += self.public_names(star, depth + 1)
        return sorted(set(names))

    def resolve(self, path: str, depth: int = 0) -> str | None:
        """Definition path (or module name) that a dotted public path refers to."""
        if path in self._memo:
            return self._memo[path]
        if depth > 8:
            return None
        result = None
        if path in self.defs or path in self.modules:
            result = path
        else:
            head, _, name = path.rpartition(".")
            if head:
                module = self.resolve(head, depth + 1)
                if module in self.modules:
                    m = self.modules[module]
                    candidate = f"{module}.{name}"
                    if candidate in self.defs or candidate in self.modules:
                        result = candidate
                    elif name in m["imports"]:
                        result = self.resolve(m["imports"][name], depth + 1)
                    else:
                        for star in m["stars"]:
                            found = self.resolve(f"{star}.{name}", depth + 1)
                            if found:
                                result = found
                                break
                elif module in self.defs:                       # a method path through a class alias
                    candidate = f"{module}.{name}"
                    result = candidate if candidate in self.defs else None
        self._memo[path] = result
        return result

    def public_paths(self, top: str) -> dict[str, set[str]]:
        """Definition path → clean public paths reachable from module paths under `top`."""
        paths: dict[str, set[str]] = {}
        for module in sorted(self.modules):
            if module != top and not module.startswith(top + "."):
                continue
            if not clean_path(module):
                continue
            for name in self.public_names(module):
                public = f"{module}.{name}"
                target = self.resolve(public)
                if target in self.defs and clean_path(public):
                    paths.setdefault(target, set()).add(public)
        for path, d in self.defs.items():
            if clean_path(path) and (path.split(".")[0] == top):
                paths.setdefault(path, set()).add(path)
        for path, d in self.defs.items():                     # members through every alias of their class
            if d["owner"] and d["owner"] in paths:
                paths.setdefault(path, set()).update(f"{a}.{d['name']}" for a in paths[d["owner"]])
        return paths


def convention_aliases(path: str) -> list[str]:
    out = []
    for prefix, short in PY_CONVENTIONS.items():
        if path.startswith(prefix + "."):
            out.append(short + path[len(prefix):])
    return out


def parse_inventory(path: Path) -> list[tuple[str, str]]:
    """(name, domain:role) of a Sphinx `objects.inv` (version 2)."""
    raw = path.read_bytes()
    header_end = 0
    for _ in range(4):
        header_end = raw.index(b"\n", header_end) + 1
    body = zlib.decompress(raw[header_end:]).decode("utf-8")
    out = []
    for line in body.splitlines():
        match = re.match(r"(.+?)\s+(\S+:\S+)\s+(-?\d+)\s+(\S*)\s+(.*)", line)
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def distribution_licence(import_name: str) -> dict[str, str | None]:
    """Distribution name, version and licence of an installed top-level package."""
    from importlib import metadata
    dists = metadata.packages_distributions().get(import_name) or []
    if not dists:
        return {"distribution": None, "version": None, "licence": None}
    name = sorted(dists)[0]
    meta = metadata.metadata(name)
    licence = meta.get("License-Expression")
    text = (meta.get("License") or "").strip()
    if not licence and text and "\n" not in text and len(text) <= 40:
        licence = text
    if not licence:
        classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
        licence = "; ".join(classifiers) or (text.splitlines()[0][:80] if text else None)
    return {"distribution": name, "version": metadata.version(name), "licence": licence}


# -- CPython documentation -------------------------------------------------------------------------

def cpython_doc_units(tar_path: Path, *, max_chars: int = 6000) -> Iterator[dict[str, Any]]:
    """Sections of the CPython text documentation (pages split at heading underlines, merged up to
    `max_chars`)."""
    with tarfile.open(tar_path, "r:bz2") as archive:
        members = sorted((m for m in archive.getmembers() if m.isfile() and m.name.endswith(".txt")), key=lambda m: m.name)
        for member in members:
            rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not rel.startswith(DOC_SECTIONS):
                continue
            text = archive.extractfile(member).read().decode("utf-8")
            lines = text.split("\n")
            sections, current = [], []
            for i, line in enumerate(lines):
                if i + 1 < len(lines) and _HEADING.match(lines[i + 1].strip()) and line.strip() and \
                        len(lines[i + 1].strip()) >= min(len(line.strip()), 3) and current:
                    sections.append("\n".join(current).strip()); current = []
                current.append(line)
            sections.append("\n".join(current).strip())
            merged: list[str] = []
            for section in sections:
                if merged and len(merged[-1]) + len(section) < max_chars and (len(section) < 400 or len(merged[-1]) < 400):
                    merged[-1] += "\n\n" + section
                elif section:
                    merged.append(section)
            for index, section in enumerate(merged):
                for part, chunk in enumerate(_split(section, max_chars * 2)):
                    yield {"unit": f"cpython:{rel}#{index}.{part}", "page": f"cpython:{rel}", "library": "stdlib",
                           "symbol": None, "text": chunk}


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > limit:
            out.append(current); current = ""
        current = (current + "\n\n" + paragraph) if current else paragraph
    if current:
        out.append(current)
    return out


# -- Node.js API ------------------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")


def html_text(fragment: str) -> str:
    text = re.sub(r"<li>", "- ", fragment)
    text = re.sub(r"</(p|li|pre|h\d|blockquote|ul|ol|table|tr)>", "\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _code_path(text_raw: str) -> str | None:
    match = re.search(r"`([A-Za-z_$][\w$.]*)(?:\(|`)", text_raw or "")
    return match.group(1) if match else None


def _node_type(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\{([^}|]+)", text)
    if match:
        return match.group(1).strip().replace("<", "").replace(">", "")
    return None


def node_symbols_and_units(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Methods and classes of the Node.js API (`all.json`) and one documentation unit each."""
    data = json.loads(path.read_text())
    symbols: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []

    def stability(node: dict[str, Any]) -> str:
        level = node.get("stability")
        return {0: "deprecated", 1: "experimental"}.get(level, "stable")

    def since(node: dict[str, Any]) -> str | None:
        added = (node.get("meta") or {}).get("added") or []
        match = re.match(r"v(\d+)", added[0]) if added else None
        return f"v{match.group(1)}" if match else None

    def walk(node: dict[str, Any], module: str | None, owner: str | None) -> None:
        kind = node.get("type")
        name = node.get("name")
        if kind == "module" and module is None:
            module = name
        if kind == "class":
            path_ = _code_path(node.get("textRaw", "")) or name
            sym = {"path": f"node:{path_}", "kind": "exception" if path_.endswith("Error") else "class",
                   "library": "node", "module": module, "owner": None, "aliases": [path_], "params": [],
                   "returns": None, "raises": [], "bases": [], "status": stability(node), "since": since(node),
                   "deprecated_by": None}
            extends = re.search(r"Extends:\s*\{?<a[^>]*>?([\w.]+)", node.get("desc", "") or "")
            if extends:
                sym["bases"] = [extends.group(1)]
            symbols.append(sym)
            units.append({"unit": f"node:{path_}", "page": f"node:{module}", "library": "node", "symbol": sym["path"],
                          "text": f"Class: {path_}\n\n" + html_text(node.get("desc", "") or "")})
            owner = sym["path"]
        if kind in ("method", "classMethod", "ctor"):
            path_ = _code_path(node.get("textRaw", ""))
            if path_:
                signature = (node.get("signatures") or [{}])[0]
                params = [{"name": p.get("name", ""), "annotation": p.get("type"), "optional": bool(p.get("optional"))}
                          for p in signature.get("params", []) if p.get("name")]
                returns = (signature.get("return") or {}).get("type")
                sym = {"path": f"node:{path_}", "kind": "method" if owner else "function", "library": "node",
                       "module": module, "owner": owner, "aliases": [path_], "params": params,
                       "returns": _node_type("{" + returns + "}") if returns else None, "raises": [], "bases": [],
                       "status": stability(node), "since": since(node), "deprecated_by": None}
                symbols.append(sym)
                lines = [node.get("textRaw", "").replace("`", "")]
                for p in signature.get("params", []):
                    if p.get("textRaw"):
                        lines.append("* " + p["textRaw"].replace("`", ""))
                if signature.get("return", {}).get("textRaw"):
                    lines.append("* " + signature["return"]["textRaw"].replace("`", ""))
                units.append({"unit": f"node:{path_}#{len(units)}", "page": f"node:{module}", "library": "node",
                              "symbol": sym["path"], "text": "\n".join(lines) + "\n\n" + html_text(node.get("desc", "") or "")})
        if kind == "module" and node.get("desc"):
            units.append({"unit": f"node:module:{name}#{len(units)}", "page": f"node:{module}", "library": "node",
                          "symbol": None, "text": f"{node.get('textRaw', name)}\n\n" + html_text(node["desc"])})
        for key in ("modules", "classes", "methods", "classMethods", "ctors", "miscs", "globals"):
            for child in node.get(key, []) or []:
                walk(child, module, owner if key in ("methods", "classMethods", "ctors") else
                     (owner if kind == "class" else None))

    for top in data.get("modules", []) + data.get("globals", []) + data.get("classes", []) + data.get("methods", []):
        walk(top, None, None)
    # one record per path (the JSON repeats a few entries); aliases: underscore-free paths only
    seen: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        if sym["path"] not in seen and clean_path(sym["aliases"][0]):
            seen[sym["path"]] = sym
    return list(seen.values()), units


# -- snapshot ---------------------------------------------------------------------------------------

def _python_record(d: dict[str, Any], aliases: set[str], library: str, source: str) -> dict[str, Any]:
    return {"path": d["path"], "kind": d["kind"], "library": library, "language": "python", "module": d["module"],
            "owner": d["owner"], "aliases": sorted(aliases), "params": d["params"], "returns": d.get("returns"),
            "raises": d.get("raises", []), "bases": d.get("bases", []), "status": "deprecated" if d.get("deprecated") else "stable",
            "since": d.get("since"), "deprecated_by": d.get("deprecated_by"), "source": source}


def _python_unit(d: dict[str, Any], canonical: str, library: str) -> dict[str, Any]:
    prefix = "class " if d["kind"] in ("class", "exception") else ""
    lines = [f"{prefix}{canonical}{d.get('signature') or '()'}" + (f" -> {d['returns']}" if d.get("returns") else ""), ""]
    lines += ["   " + line if line else "" for line in (d.get("doc") or "").splitlines()]
    if d.get("comments"):
        lines += ["", "   Implementation notes (source comments):"] + ["   " + c for c in d["comments"]]
    return {"unit": f"py:{d['path']}", "page": f"py:{d['module']}", "library": library, "symbol": d["path"],
            "text": "\n".join(lines).strip()}


def build_snapshot(out_dir: Path, *, packages: list[str], site_packages: Path, stdlib_dir: Path, cpython_docs: Path,
                   inventory: Path, node_api: Path, english_words: set[str]) -> dict[str, Any]:
    """Extract every source into `symbols.jsonl.gz` and `units.jsonl.gz` plus `manifest.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    licences: dict[str, Any] = {}
    file_digest = hashlib.sha256()
    # third-party packages
    for package in packages:
        root = site_packages / package
        parsed = []
        for module, path in package_files(root, package):
            file_digest.update(path.read_bytes())
            result = parse_module(module, path, is_package=path.name == "__init__.py")
            if result:
                parsed.append(result)
        index = PythonIndex(parsed)
        paths = index.public_paths(package)
        licences[package] = distribution_licence(package)
        seen_docs: set[str] = set()
        for path_, d in sorted(index.defs.items()):
            aliases = {a for a in paths.get(path_, set()) if clean_path(a)}
            if not aliases or not d.get("doc"):
                continue
            digest = hashlib.sha256(d["doc"].encode()).hexdigest()
            canonical = min(aliases, key=lambda a: (len(a), a))
            records.append(_python_record(d, aliases | {c for a in aliases for c in convention_aliases(a)}, package, "ast"))
            if digest in seen_docs:          # identical docstrings (generated model docs) are kept once
                units.append({"unit": f"py:{path_}", "page": f"py:{d['module']}", "library": package, "symbol": path_,
                              "text": f"{canonical}{d.get('signature') or '()'}\n\n   See the documentation of the parent class."})
                continue
            seen_docs.add(digest)
            units.append(_python_unit(d, canonical, package))
    # standard library: the documented objects of the inventory
    inventory_objects = [(n, INVENTORY_KINDS[r]) for n, r in parse_inventory(inventory) if r in INVENTORY_KINDS]
    parsed = []
    for module, path in stdlib_files(stdlib_dir):
        file_digest.update(path.read_bytes())
        result = parse_module(module, path, is_package=path.name == "__init__.py")
        if result:
            parsed.append(result)
    index = PythonIndex(parsed)
    licences["stdlib"] = {"distribution": "CPython", "version": sys.version.split()[0], "licence": "PSF-2.0"}
    kinds = {n: k for n, k in inventory_objects}
    for name, kind in sorted(kinds.items()):
        if kind == "module" or not clean_path(name):
            continue
        target = index.resolve(name)
        d = index.defs.get(target) if target else None
        parent = name.rpartition(".")[0]
        owner = parent if kinds.get(parent) in ("class", "exception") else None
        module = parent if owner is None else owner.rpartition(".")[0]
        if d is None:
            d = {"path": name, "kind": kind, "module": module, "owner": owner, "params": [], "returns": None, "raises": [],
                 "bases": [], "deprecated": False, "deprecated_by": None, "since": None, "doc": None, "comments": []}
        record = _python_record({**d, "path": name, "module": module, "owner": owner,
                                 "kind": "exception" if kind == "exception" else (d["kind"] if d["kind"] != "function" else kind)},
                                {name}, "stdlib", "inventory+ast" if target else "inventory")
        records.append(record)
        if d.get("doc"):
            units.append(_python_unit({**d, "path": name, "module": module}, name, "stdlib"))
    file_digest.update(cpython_docs.read_bytes()); file_digest.update(inventory.read_bytes())
    units += list(cpython_doc_units(cpython_docs))
    # Node.js
    file_digest.update(node_api.read_bytes())
    node_records, node_units = node_symbols_and_units(node_api)
    for r in node_records:
        r.update(language="javascript", source="node-api")
    records += node_records
    units += node_units
    licences["node"] = {"distribution": "Node.js", "version": "20.20.2", "licence": "MIT"}
    # every alias is dotted ("any", "require" would link everywhere) except bare CamelCase class names
    # that are not English words and name one symbol only
    for r in records:
        r["aliases"] = [a for a in r["aliases"] if "." in a]
    bare = Counter(r["path"].split(".")[-1] for r in records if r["kind"] in ("class", "exception"))
    for r in records:
        name = r["path"].split(".")[-1].split(":")[-1]
        if r["kind"] in ("class", "exception") and _CAMEL.fullmatch(name) and len(name) >= 6 and bare[name] == 1 \
                and name.lower() not in english_words:
            r["aliases"] = sorted(set(r["aliases"]) | {name})
    # an alias naming several symbols is dropped (no polysemy among real symbols), and so is a symbol left without one
    owners: dict[str, set[str]] = {}
    for r in records:
        for a in r["aliases"]:
            owners.setdefault(a.lower(), set()).add(r["path"])
    for r in records:
        r["aliases"] = [a for a in r["aliases"] if len(owners[a.lower()]) == 1]
    records = [r for r in records if r["aliases"]]
    # overridden base methods (Python): a method of a class whose resolved base defines the same name
    by_path = {r["path"]: r for r in records}
    for r in records:
        if r["kind"] in ("method", "property") and r["owner"] in by_path:
            for base in by_path[r["owner"]].get("bases", []):
                candidate = f"{base}.{r['path'].rsplit('.', 1)[1]}"
                if candidate in by_path and candidate != r["path"]:
                    r["overrides"] = candidate
                    break
    kept = {r["path"] for r in records}
    units = [u for u in units if u["symbol"] is None or u["symbol"] in kept]
    stats = {"symbols": len(records), "units": len(units), "unit_chars": sum(len(u["text"]) for u in units),
             "by_library": dict(sorted(Counter(r["library"] for r in records).items())),
             "by_kind": dict(sorted(Counter(r["kind"] for r in records).items())),
             "aliases": sum(len(r["aliases"]) for r in records), "source_digest": file_digest.hexdigest(),
             "licences": licences}
    hashes = {}
    for name, rows in (("symbols.jsonl.gz", records), ("units.jsonl.gz", units)):
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
            for row in rows:
                handle.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
        (out_dir / name).write_bytes(buffer.getvalue())
        hashes[name] = hashlib.sha256(buffer.getvalue()).hexdigest()
    stats["files"] = hashes
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def snapshot_sha256(snapshot_dir: Path) -> str:
    """One hash over the snapshot's two data files."""
    digest = hashlib.sha256()
    for name in ("symbols.jsonl.gz", "units.jsonl.gz"):
        digest.update(hashlib.sha256((snapshot_dir / name).read_bytes()).digest())
    return digest.hexdigest()
