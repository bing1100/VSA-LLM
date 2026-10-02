"""EuroVoc as a frame ontology (track T6, legal/regulatory; Publications Office SKOS core export).

Concept = EuroVoc descriptor (a `skos:Concept` in the EuroVoc scheme with a label in `language`).
Frame edges, in priority order (at most `max_degree`):

- `broader` (BT) and `top_term` (the root of the descriptor's BT hierarchy);
- `microthesaurus`: each microthesaurus (MT) scheme the descriptor is in, and `domain`: the MT's
  domain (the first two digits of the MT notation, e.g. MT 6011 → domain 60 agri-foodstuffs);
- `related` (RT).

Fillers: descriptor atoms for every descriptor used as a broader, related or top-term filler, plus
the MT and domain atoms; out-of-dictionary descriptors fall back to their nearest in-dictionary
broader term. Aliases: the preferred label and every non-preferred term (`altLabel`, "UF").
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .wordnet import FrameOntology

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SKOS = "{http://www.w3.org/2004/02/skos/core#}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
EUROVOC = "http://eurovoc.europa.eu/"
THESAURUS = EUROVOC + "100141"
RELATION_NAMES = ["broader", "top_term", "microthesaurus", "domain", "related"]


def _open(path: Path):
    if str(path).endswith(".zip"):
        archive = zipfile.ZipFile(path)
        member = next(n for n in archive.namelist() if n.endswith(".rdf"))
        return io.BufferedReader(archive.open(member))
    return open(path, "rb")


def parse_eurovoc(path: Path, *, language: str = "en") -> dict[str, Any]:
    """Descriptors, microthesaurus schemes and domains from the SKOS core RDF/XML (zip or rdf)."""
    concepts: dict[str, dict[str, Any]] = {}
    schemes: dict[str, dict[str, Any]] = {}
    with _open(path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != RDF + "Description":
                continue
            uri = element.get(RDF + "about", "")
            types = {t.get(RDF + "resource") for t in element.findall(RDF + "type")}
            def labels(tag: str) -> list[str]:
                return [" ".join((e.text or "").split()) for e in element.findall(SKOS + tag) if e.get(XML_LANG) == language and e.text]
            def refs(tag: str) -> list[str]:
                return [e.get(RDF + "resource") for e in element.findall(SKOS + tag) if e.get(RDF + "resource")]
            pref = labels("prefLabel")
            notation = element.findtext(SKOS + "notation")
            if SKOS.strip("{}") + "ConceptScheme" in types:
                schemes[uri] = {"label": pref[0] if pref else "", "notation": notation}
            elif SKOS.strip("{}") + "Concept" in types and pref:
                concepts[uri] = {"uri": uri, "label": pref[0], "alt": labels("altLabel"), "broader": refs("broader"),
                                 "related": refs("related"), "schemes": refs("inScheme"), "notation": notation,
                                 "definition": (labels("definition") or labels("scopeNote") or [""])[0]}
            element.clear()
    domains = {c["notation"]: uri for uri, c in concepts.items() if EUROVOC + "domains" in c["schemes"] and c["notation"]}
    descriptors = {uri: c for uri, c in concepts.items() if THESAURUS in c["schemes"]}
    microthesauri = {uri: s for uri, s in schemes.items() if s.get("notation") and uri != THESAURUS}
    return {"descriptors": descriptors, "microthesauri": microthesauri, "domains": {n: concepts[u] for n, u in domains.items()}}


def mt_name(label: str) -> str:
    """"6011 animal product" → "animal product"."""
    head, _, rest = label.partition(" ")
    return rest if head.isdigit() else label


def domain_name(label: str) -> str:
    """"28 SOCIAL QUESTIONS" → "social questions"."""
    head, _, rest = label.partition(" ")
    return (rest if head.isdigit() else label).lower()


def build_eurovoc_ontology(path: Path, *, language: str = "en", max_atomics: int = 8192, max_degree: int = 16,
                           max_related: int = 6) -> FrameOntology:
    parsed = parse_eurovoc(path, language=language)
    descriptors, microthesauri, domains = parsed["descriptors"], parsed["microthesauri"], parsed["domains"]
    uris = sorted(descriptors, key=lambda u: (len(u), u))
    broader = {u: [b for b in descriptors[u]["broader"] if b in descriptors] for u in uris}

    def top_term(uri: str) -> str:
        seen, node = {uri}, uri
        while broader.get(node):
            nxt = broader[node][0]
            if nxt in seen:
                break
            seen.add(nxt); node = nxt
        return node

    tops = {u: top_term(u) for u in uris}
    use: Counter[str] = Counter()
    for u in uris:
        use.update(broader[u]); use.update(r for r in descriptors[u]["related"] if r in descriptors)
        if tops[u] != u:
            use[tops[u]] += 1
    mt_atoms = [f"mt:{microthesauri[s]['notation']}" for s in sorted(microthesauri, key=lambda s: microthesauri[s]["notation"])]
    domain_atoms = [f"domain:{n}" for n in sorted(domains)]
    fixed = mt_atoms + domain_atoms
    budget = max(0, max_atomics - len(fixed))
    filler_uris = [u for u, _ in sorted(use.items(), key=lambda kv: (-kv[1], kv[0]))[:budget]]
    atomic_names = [f"eurovoc:{u.rsplit('/', 1)[-1]}" for u in filler_uris] + fixed
    atomic_index = {name: i for i, name in enumerate(atomic_names)}
    relation_index = {name: i for i, name in enumerate(RELATION_NAMES)}
    cache: dict[str, int | None] = {}

    def in_dictionary(uri: str) -> int | None:
        if uri in cache:
            return cache[uri]
        seen, queue, found = {uri}, deque([uri]), None
        while queue:
            node = queue.popleft()
            key = f"eurovoc:{node.rsplit('/', 1)[-1]}"
            if key in atomic_index:
                found = atomic_index[key]; break
            for parent in broader.get(node, []):
                if parent not in seen:
                    seen.add(parent); queue.append(parent)
        cache[uri] = found
        return found

    frames, names, aliases = [], [], []
    for i, u in enumerate(uris):
        c = descriptors[u]
        frame: list[tuple[int, int]] = []
        def push(relation: str, atom: int | None) -> None:
            if atom is not None and (relation_index[relation], atom) not in frame and len(frame) < max_degree:
                frame.append((relation_index[relation], atom))
        for b in broader[u]:
            push("broader", in_dictionary(b))
        if tops[u] != u:
            push("top_term", in_dictionary(tops[u]))
        for scheme in c["schemes"]:
            if scheme in microthesauri:
                notation = microthesauri[scheme]["notation"]
                push("microthesaurus", atomic_index.get(f"mt:{notation}"))
                push("domain", atomic_index.get(f"domain:{notation[:2]}"))
        for r in [r for r in c["related"] if r in descriptors][:max_related]:
            push("related", in_dictionary(r))
        frames.append(frame); names.append(f"eurovoc:{u.rsplit('/', 1)[-1]}")
        aliases += [(label, i) for label in {c["label"], *c["alt"]}]
    ontology = FrameOntology("eurovoc", names, list(RELATION_NAMES), atomic_names, frames, aliases,
                             {"source": str(path), "language": language, "descriptors": len(uris),
                              "microthesauri": len(microthesauri), "domains": len(domains),
                              "max_atomics": max_atomics, "max_degree": max_degree})
    ontology.metadata["records"] = {f"eurovoc:{u.rsplit('/', 1)[-1]}": {
        "label": descriptors[u]["label"], "alt": descriptors[u]["alt"], "definition": descriptors[u]["definition"],
        "broader": [descriptors[b]["label"] for b in broader[u]],
        "top_term": descriptors[tops[u]]["label"] if tops[u] != u else None,
        "microthesauri": [mt_name(microthesauri[s]["label"]) for s in descriptors[u]["schemes"] if s in microthesauri],
        "domains": sorted({domain_name(domains[microthesauri[s]["notation"][:2]]["label"])
                           for s in descriptors[u]["schemes"] if s in microthesauri and microthesauri[s]["notation"][:2] in domains}),
        "related": [descriptors[r]["label"] for r in descriptors[u]["related"] if r in descriptors],
        "broader_uris": broader[u], "schemes": [s for s in descriptors[u]["schemes"] if s in microthesauri],
    } for u in uris}
    ontology.metadata["mt_labels"] = {f"{EUROVOC}{s.rsplit('/', 1)[-1]}": mt_name(m["label"]) for s, m in microthesauri.items()}
    ontology.metadata["mt_notation"] = {s: m["notation"] for s, m in microthesauri.items()}
    ontology.metadata["domain_labels"] = {n: domain_name(d["label"]) for n, d in domains.items()}
    ontology.metadata["domain_uris"] = {d["uri"]: domain_name(d["label"]) for d in domains.values()}
    return ontology
