"""ChEBI as a frame ontology (track T4, chemistry; ChEBI OBO release, CC BY 4.0).

Concept = ChEBI entity with at least `min_star` stars (3 = manually annotated by the ChEBI team),
not obsolete. Frame edges, in priority order (at most `max_degree`; the first `role_budget` roles come
before the element/charge/branch atoms, further roles fill any remaining degree):

- `is_a`: superclasses (chemical classes, or role classes for roles);
- the ChEBI relationships `has_role`, `has_functional_parent`, `has_parent_hydride`, `has_part`,
  `is_conjugate_acid_of`, `is_conjugate_base_of`, `is_tautomer_of`, `is_enantiomer_of`,
  `is_substituent_group_from`;
- `contains_element`: elements of the generalized empirical formula except hydrogen (functional
  groups and elements are the natural atomics of chemistry);
- `charge`: negative / neutral / positive (from the formula's charge);
- `branch`: the top-level ChEBI branch (chemical entity, role, subatomic particle).

Fillers come from a bounded atomic dictionary: element, charge and branch atoms plus the entities
most used as fillers; a filler outside the dictionary is replaced by its nearest in-dictionary
`is_a` ancestor (as MeSH and WordNet do), or dropped if none exists.

Aliases are the name and the EXACT/RELATED synonyms (IUPAC names, INNs, brand names, source names).
Dropped: strings under 3 characters, formula-like strings ("C9H8O4", "Na+"), English stopwords,
synonyms (not names) that are a single word of at most 4 letters (mostly abbreviations such as "AND"
that collide with English once lower-cased), and aliases longer than `max_alias_chars` (they never
occur in running text and only inflate the linker trie).
"""

from __future__ import annotations

import gzip
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterator

from .wordnet import FrameOntology

RELATIONSHIPS = {
    "RO:0000087": "has_role", "RO:0018038": "has_functional_parent", "RO:0018040": "has_parent_hydride",
    "BFO:0000051": "has_part", "RO:0018034": "is_conjugate_acid_of", "RO:0018033": "is_conjugate_base_of",
    "RO:0018036": "is_tautomer_of", "RO:0018039": "is_enantiomer_of", "RO:0018037": "is_substituent_group_from",
}
RELATION_NAMES = ["is_a", *RELATIONSHIPS.values(), "contains_element", "charge", "branch"]
BRANCHES = {"CHEBI:24431": "chemical entity", "CHEBI:50906": "role", "CHEBI:36342": "subatomic particle"}
_FORMULA = re.compile(r"^(\(?[A-Z][a-z]?\d*\)?n?)+([+-]\d*|\d*[+-])?(\.(\d*)?(\(?[A-Z][a-z]?\d*\)?)+)*$")
_ELEMENT = re.compile(r"([A-Z][a-z]?)")
_SYNONYM = re.compile(r'^"((?:[^"\\]|\\.)*)"\s+(EXACT|RELATED|BROAD|NARROW)\s*([^\[]*)\[')
_DEF = re.compile(r'^"((?:[^"\\]|\\.)*)"')
STOPWORDS = {"the", "and", "for", "are", "was", "with", "this", "that", "from", "has", "have", "not", "but", "all",
             "can", "its", "may", "use", "one", "two", "who", "how", "out", "any", "new", "per", "via"}


def parse_obo(path: Path) -> Iterator[dict[str, Any]]:
    """`[Term]` stanzas of a (gzipped) ChEBI OBO file as dicts."""
    opener = gzip.open if str(path).endswith(".gz") else open
    record: dict[str, Any] | None = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("["):
                if record and record.get("id"):
                    yield record
                record = {"synonyms": [], "is_a": [], "relations": [], "star": 0} if line == "[Term]" else None
                continue
            if record is None or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key == "id":
                record["id"] = value
            elif key == "name":
                record["name"] = value
            elif key == "def":
                match = _DEF.match(value)
                record["definition"] = match.group(1).replace('\\"', '"') if match else ""
            elif key == "subset" and value.endswith(":STAR"):
                record["star"] = int(value.split(":")[0])
            elif key == "synonym":
                match = _SYNONYM.match(value)
                if match:
                    record["synonyms"].append((match.group(1).replace('\\"', '"'), match.group(2), match.group(3).strip()))
            elif key == "is_a":
                record["is_a"].append(value.split(" ! ")[0].split()[0])
            elif key == "relationship":
                parts = value.split(" ! ")[0].split()
                if len(parts) >= 2 and parts[0] in RELATIONSHIPS:
                    record["relations"].append((RELATIONSHIPS[parts[0]], parts[1]))
            elif key == "property_value":
                parts = value.split(" ", 2)
                if parts[0].endswith("generalized_empirical_formula"):
                    record["formula"] = parts[1].strip('"')
                elif parts[0].endswith(":charge"):
                    try:
                        record["charge"] = int(parts[1].strip('"'))
                    except ValueError:
                        pass
            elif key == "is_obsolete" and value.strip() == "true":
                record["obsolete"] = True
        if record and record.get("id"):
            yield record


def keep_alias(text: str, *, is_name: bool, max_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < 3 or len(stripped) > max_chars or stripped.lower() in STOPWORDS:
        return False
    if _FORMULA.match(stripped.replace(" ", "")):
        return False
    if not is_name and " " not in stripped and len(stripped) <= 4 and stripped.isalpha():
        return False
    return any(ch.isalpha() for ch in stripped)


def formula_elements(formula: str | None) -> list[str]:
    if not formula:
        return []
    return sorted(set(_ELEMENT.findall(formula)) - {"H", "R"})


def build_chebi_ontology(path: Path, *, min_star: int = 3, max_atomics: int = 8192, max_degree: int = 24,
                         role_budget: int = 8, max_alias_chars: int = 120) -> FrameOntology:
    records = {r["id"]: r for r in parse_obo(path) if not r.get("obsolete") and r.get("name")}
    concepts = sorted((r for r in records.values() if r["star"] >= min_star), key=lambda r: int(r["id"].split(":")[1]))
    parents = {cid: r["is_a"] for cid, r in records.items()}

    branch_cache: dict[str, str | None] = {}
    def branch(cid: str) -> str | None:
        if cid in branch_cache:
            return branch_cache[cid]
        seen, queue, found = {cid}, deque([cid]), None
        while queue and found is None:
            node = queue.popleft()
            if node in BRANCHES:
                found = BRANCHES[node]; break
            for parent in parents.get(node, []):
                if parent not in seen:
                    seen.add(parent); queue.append(parent)
        branch_cache[cid] = found
        return found

    use: Counter[str] = Counter()
    for r in concepts:
        use.update(r["is_a"]); use.update(target for _, target in r["relations"])
    elements = sorted({e for r in concepts for e in formula_elements(r.get("formula"))})
    fixed = ([f"element:{e}" for e in elements] + ["charge:negative", "charge:neutral", "charge:positive"]
             + [f"branch:{b}" for b in BRANCHES.values()])
    budget = max(0, max_atomics - len(fixed))
    filler_ids = [cid for cid, _ in sorted(use.items(), key=lambda kv: (-kv[1], int(kv[0].split(":")[1]) if kv[0].split(":")[1].isdigit() else 0))
                  if cid in records][:budget]
    atomic_names = [f"chebi:{cid}" for cid in filler_ids] + fixed
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_index = {name: i for i, name in enumerate(RELATION_NAMES)}
    cache: dict[str, int | None] = {}

    def in_dictionary(cid: str) -> int | None:
        """Nearest in-dictionary ancestor (breadth-first over is_a), including itself."""
        if cid in cache:
            return cache[cid]
        seen, queue, found = {cid}, deque([cid]), None
        while queue:
            node = queue.popleft()
            if f"chebi:{node}" in atomic_index:
                found = atomic_index[f"chebi:{node}"]; break
            for parent in parents.get(node, []):
                if parent not in seen:
                    seen.add(parent); queue.append(parent)
        cache[cid] = found
        return found

    frames, names, aliases = [], [], []
    replaced = dropped = 0
    for i, r in enumerate(concepts):
        frame: list[tuple[int, int]] = []
        def push(relation: str, atom: int | None) -> None:
            if atom is not None and (relation_index[relation], atom) not in frame and len(frame) < max_degree:
                frame.append((relation_index[relation], atom))
        structural = [("is_a", p) for p in r["is_a"]] + [(rel, t) for rel, t in r["relations"] if rel != "has_role"]
        roles = [(rel, t) for rel, t in r["relations"] if rel == "has_role"]
        edges = structural + roles[:role_budget]
        for relation, target in edges:
            atom = in_dictionary(target)
            if atom is None:
                dropped += 1; continue
            if atomic_names[atom] != f"chebi:{target}":
                replaced += 1
            push(relation, atom)
        for element in formula_elements(r.get("formula")):
            push("contains_element", atomic_index[f"element:{element}"])
        if "charge" in r:
            push("charge", atomic_index["charge:" + ("negative" if r["charge"] < 0 else "positive" if r["charge"] > 0 else "neutral")])
        top = branch(r["id"])
        if top:
            push("branch", atomic_index[f"branch:{top}"])
        for relation, target in roles[role_budget:]:     # remaining roles fill any spare degree
            atom = in_dictionary(target)
            if atom is not None:
                push(relation, atom)
        frames.append(frame); names.append(r["id"])
        surface = {r["name"]: True}
        for text, _, _ in r["synonyms"]:
            surface.setdefault(text, False)
        aliases += [(text, i) for text, is_name in surface.items() if keep_alias(text, is_name=is_name, max_chars=max_alias_chars)]
    ontology = FrameOntology("chebi", names, list(RELATION_NAMES), atomic_names, frames, aliases,
                             {"source": str(path), "entities": len(records), "min_star": min_star, "max_atomics": max_atomics,
                              "max_degree": max_degree, "role_budget": role_budget, "max_alias_chars": max_alias_chars,
                              "fillers_replaced_by_ancestor": replaced, "fillers_dropped": dropped})
    ontology.metadata["labels"] = [r["name"] for r in concepts]
    ontology.metadata["definitions"] = [r.get("definition", "") for r in concepts]
    ontology.metadata["records"] = {r["id"]: r for r in concepts}
    ontology.metadata["all_names"] = {cid: rec["name"] for cid, rec in records.items()}
    ontology.metadata["all_surface_forms"] = {text.lower() for rec in records.values()
                                              for text in [rec["name"], *(syn for syn, _, _ in rec["synonyms"])]}
    return ontology
