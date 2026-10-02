"""Google Product Taxonomy as a frame ontology (track T3, product catalogues).

Concept = taxonomy category (`gpt:<id>`). Frame edges, in priority order (at most `max_degree`):

- `parent`: the immediate parent category; `top_category` and `second_category`: the level-1 and
  level-2 ancestors; `depth`: the level;
- `name_token`: content words of the category's own name, `path_token`: content words of its
  ancestors' names (lower-cased, naively singularised) — the path tokens are the taxonomy's atomics;
- `has_attribute`: product attributes of the Shopify Standard Product Taxonomy categories that
  Shopify maps onto this Google category (optional; Shopify's taxonomy is MIT-licensed and versioned
  against Google's 2021-09-21 release).

Fillers: category atoms for every category used as a filler, plus token, depth and attribute atoms,
bounded by `max_atomics` (least-used attributes and tokens are dropped first).

Aliases: the category name; below level 2, its parts split on "&" / "," / " and " ("Skillets & Frying
Pans" → "Skillets", "Frying Pans"; a one-word modifier part takes the head noun: "Liquid & Frozen Eggs" →
"Liquid Eggs"); the multi-word names of the finer Shopify categories mapped onto it ("Bird Cage Water
Dishes" → "Bird Cage Food & Water Dishes"); and singular forms of multi-word aliases ("Frying Pan").
Single-word singulars ("Oranges" → "orange") and single-word Shopify names ("Cotton") are not added —
they are mostly colours, materials or other words with a general meaning. Single generic words
("accessories", "supplies", "parts", …) and aliases shared by more than `max_alias_senses` categories
are dropped: they carry no category information.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .wordnet import FrameOntology

RELATION_NAMES = ["parent", "top_category", "second_category", "depth", "name_token", "path_token", "has_attribute"]
STOP = {"&", "and", "or", "for", "of", "the", "with", "in", "on", "to", "a", "by", "other", "non"}
GENERIC = {"accessories", "accessory", "supplies", "supply", "parts", "part", "other", "kits", "kit", "sets", "set",
           "products", "product", "equipment", "items", "item", "tools", "tool", "goods", "materials", "material",
           "general", "systems", "system", "components", "component", "care", "cases", "case", "covers", "cover"}
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def singular(word: str) -> str:
    """Naive English singular of one word (good enough for taxonomy nouns)."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def singular_phrase(phrase: str) -> str:
    words = phrase.split()
    return " ".join(words[:-1] + [singular(words[-1])]) if words else phrase


def name_parts(name: str) -> list[str]:
    """"Skillets & Frying Pans" → ["Skillets", "Frying Pans"]."""
    parts = re.split(r"\s*(?:&|,|\band\b)\s*", name)
    return [p.strip() for p in parts if p.strip()]


def alias_forms(name: str, depth: int) -> set[str]:
    """The name and, below level 2, its "&"/"," parts. A one-word part that is not a plural noun
    modifies the last part's head ("Liquid & Frozen Eggs" → "Liquid Eggs", "Frozen Eggs"); a plural
    one stands alone ("Skillets & Frying Pans" → "Skillets", "Frying Pans")."""
    forms = {name}
    parts = name_parts(name)
    if depth <= 2 or len(parts) < 2:
        return forms
    head = parts[-1].split()[-1]
    for part in parts:
        words = part.split()
        if len(words) == 1 and part is not parts[-1] and not part.lower().endswith("s") and len(parts[-1].split()) > 1:
            forms.add(f"{part} {head}")
        else:
            forms.add(part)
    return forms


def tokens(name: str) -> list[str]:
    return [singular(w) for w in _WORD.findall(name.lower()) if w not in STOP]


def parse_taxonomy(path: Path) -> list[tuple[str, list[str]]]:
    """(id, path of names) per line of `taxonomy-with-ids.<locale>.txt`."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ident, _, full = line.partition(" - ")
        rows.append((ident.strip(), [p.strip() for p in full.split(" > ")]))
    return rows


def shopify_attributes(categories_json: Path, mapping_txt: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Google full path → Shopify attribute names, and Google full path → names of the Shopify
    categories mapped onto it."""
    data = json.loads(Path(categories_json).read_text())
    by_full = {c["full_name"]: c for vertical in data["verticals"] for c in vertical["categories"]}
    attributes: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    source = None
    for line in Path(mapping_txt).read_text(encoding="utf-8").splitlines():
        if line.startswith("→ "):
            source = line[2:].strip()
        elif line.startswith("⇒ ") and source:
            target = line[2:].strip()
            category = by_full.get(source)
            if category:
                attributes.setdefault(target, set()).update(a["name"] for a in category.get("attributes", []))
                names.setdefault(target, set()).add(category["name"])
            source = None
    return attributes, names


def build_google_product_ontology(path: Path, *, shopify_categories: Path | None = None, shopify_mapping: Path | None = None,
                                  max_atomics: int = 8192, max_degree: int = 16, max_alias_senses: int = 8) -> FrameOntology:
    rows = parse_taxonomy(path)
    by_path = {" > ".join(p): ident for ident, p in rows}
    attributes, shopify_names = (shopify_attributes(shopify_categories, shopify_mapping)
                                 if shopify_categories and shopify_mapping else ({}, {}))
    parent_of = {ident: by_path.get(" > ".join(p[:-1])) for ident, p in rows}
    used_categories = {x for x in parent_of.values() if x} | {by_path[p[0]] for _, p in rows} | \
        {by_path[" > ".join(p[:2])] for _, p in rows if len(p) >= 2}
    token_use: Counter[str] = Counter(t for _, p in rows for name in p for t in tokens(name))
    attribute_use: Counter[str] = Counter(a for _, p in rows for a in attributes.get(" > ".join(p), ()))
    depths = sorted({len(p) for _, p in rows})
    fixed = [f"category:{c}" for c in sorted(used_categories, key=int)] + [f"depth:{d}" for d in depths]
    budget = max(0, max_atomics - len(fixed))
    ranked = sorted([(n, f"token:{t}") for t, n in token_use.items()] + [(n, f"attribute:{a}") for a, n in attribute_use.items()],
                    key=lambda kv: (-kv[0], kv[1]))
    atomic_names = fixed + [name for _, name in ranked[:budget]]
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_index = {name: i for i, name in enumerate(RELATION_NAMES)}
    frames, names, alias_sets = [], [], []
    for ident, p in rows:
        frame: list[tuple[int, int]] = []
        def push(relation: str, atom: str) -> None:
            index = atomic_index.get(atom)
            if index is not None and (relation_index[relation], index) not in frame and len(frame) < max_degree:
                frame.append((relation_index[relation], index))
        if parent_of[ident]:
            push("parent", f"category:{parent_of[ident]}")
        push("top_category", f"category:{by_path[p[0]]}")
        if len(p) >= 2:
            push("second_category", f"category:{by_path[' > '.join(p[:2])]}")
        push("depth", f"depth:{len(p)}")
        for t in tokens(p[-1])[:4]:
            push("name_token", f"token:{t}")
        for a in sorted(attributes.get(" > ".join(p), ()), key=lambda a: (-attribute_use[a], a))[:6]:
            push("has_attribute", f"attribute:{a}")
        for ancestor in reversed(p[:-1]):
            for t in tokens(ancestor):
                push("path_token", f"token:{t}")
        frames.append(frame); names.append(f"gpt:{ident}")
        surfaces = alias_forms(p[-1], len(p)) | {n for n in shopify_names.get(" > ".join(p), set()) if " " in n}
        surfaces |= {singular_phrase(s) for s in list(surfaces) if " " in s}
        alias_sets.append({s for s in surfaces if not (s.lower() in GENERIC or s.lower() in STOP or len(s) < 3)})
    senses: Counter[str] = Counter(a.lower() for aliases in alias_sets for a in aliases)
    aliases = [(a, i) for i, s in enumerate(alias_sets) for a in sorted(s) if senses[a.lower()] <= max_alias_senses]
    ontology = FrameOntology("google_product", names, list(RELATION_NAMES), atomic_names, frames, aliases,
                             {"source": str(path), "categories": len(rows), "max_atomics": max_atomics, "max_degree": max_degree,
                              "max_alias_senses": max_alias_senses, "shopify": bool(attributes),
                              "dropped_generic_aliases": sum(1 for a, n in senses.items() if n > max_alias_senses)})
    ontology.metadata["paths"] = {f"gpt:{ident}": p for ident, p in rows}
    ontology.metadata["attributes"] = {f"gpt:{ident}": sorted(attributes.get(" > ".join(p), ())) for ident, p in rows}
    return ontology
