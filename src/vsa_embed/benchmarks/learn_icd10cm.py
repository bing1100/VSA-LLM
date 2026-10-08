"""Holdout H3: the ICD-10-CM FY2027 new codes (effective 1 October 2026), placed into the FY2026 classification.

**Sources** (`learn_data.SOURCES`, CMS, public domain): the FY2027 and FY2026 code-description files (order files:
every code and header with its billable flag and titles) and tabular XML (chapters, blocks, nested `diag` elements with
inclusion terms, includes / excludes notes and 7th-character definitions). The April 1, 2026 update changed no codes
(its addenda list 0 additions), so FY2026 = one code set; its tabular XML is the April 2026 one.

**Counts** (verified 2026-10-08): 190 new billable codes and 48 new header (non-billable) codes = 238 new code strings
in the FY2027 order file — the "190–238 new codes" of the plan are these two counts. 30 FY2026 billable codes stop being
billable (15 become headers, 15 are deleted; 21 code strings are deleted in all, headers included).

**Hierarchy.** Node ids are dotted codes (`I42.01`), blocks `block:<first>-<last>` and chapters `chapter:<n>`. A code's
parent is its longest proper prefix in the same release's code set (headers included; the placeholder `X` and 7th
characters fall out naturally), a 3-character category's parent is its block, a block's its chapter.

**Placement items** (`icd10cm-fy2027`, one per new code): the gold parent is the *nearest ancestor that exists in
FY2026* (walking up the FY2027 chain through new headers); the gold relations add the FY2026 block and chapter. The
definition is the title plus the tabular inclusion terms and includes notes; aliases are the inclusion terms.
`candidates` is null (= every FY2026 node). Dev/test by `learn_data.split_of` (20/80).

**Choice items** (`icd10cm-fy2027-mc`, MedConceptsQA format, 4 options): `code2title` ("What is the description of
the medical code C78.31 in ICD10CM?" → titles) and `title2code` (→ codes). Distractors: up to 2 siblings (other children
of the code's FY2027 parent), the rest from the same chapter (never an ancestor or descendant), drawn with a per-item
seeded generator.

**UMLS map** (`map_umls`, licensed UMLS 2022AB read in place from the repository's git-ignored `data/`): the gold
parents and their ancestors → CUIs (SAB ICD10CM) → MeSH (SAB MSH) and SNOMED CT (SAB SNOMEDCT_US) codes, plus exact
English string matches of the new codes' titles and inclusion terms. Written only under `~/data/vsa-llm/toolkit-learn/`
(mode 700); the repository gets aggregate counts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import learn_data as ld

SET_PLACEMENT = "icd10cm-fy2027"
SET_CHOICE = "icd10cm-fy2027-mc"
VERSION = "v1"
RAW = ld.DATA_ROOT / "icd10cm" / "raw"
ORDER_FILES = {   # release → (zip, member)
    "fy2027": ("2027-code-descriptions-tabular-order.zip", "Code Descriptions/icd10cm_order_2027.txt"),
    "fy2026": ("april-1-2026-code-descriptions-tabular-order.zip", "Code Descriptions/icd10cm_order_2026.txt"),
}
TABULAR_FILES = {
    "fy2027": ("2027-code-tables-tabular-index.zip", "Table and Index/icd10cm_tabular_2027.xml"),
    "fy2026": ("april-1-2026-code-tables-tabular-index.zip", "Table and Index/icd10cm_tabular_2026.xml"),
}
NOTE_TAGS = ("inclusionTerm", "includes", "excludes1", "excludes2", "codeFirst", "useAdditionalCode", "codeAlso", "notes")
UMLS_META = Path("/home/bhux/workplace/VSA-LLM/data/umls-2022AB-full/2022AB-full/2022ab-1-meta/2022AB/META")


def dotted(code: str) -> str:
    """`I4201` → `I42.01` (categories stay `I42`)."""
    code = code.strip().replace(".", "")
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def undotted(code: str) -> str:
    return code.replace(".", "")


# -- parsing -------------------------------------------------------------------------------------------------------------

def parse_order(lines: Iterable[str]) -> dict[str, dict[str, Any]]:
    """CMS order file (fixed width: order 1–5, code 7–13, header flag 15, short title 17–76, long title 78–) →
    {dotted code: {code, billable, short, long, order}}."""
    codes: dict[str, dict[str, Any]] = {}
    for line in lines:
        line = line.rstrip("\r\n")
        if len(line) < 16 or not line[:5].strip().isdigit():
            continue
        code, flag = line[6:13].strip(), line[14]
        if flag not in "01":
            raise ValueError(f"order file: bad header flag in {line[:20]!r}")
        codes[dotted(code)] = {"code": code, "billable": flag == "1", "short": line[16:76].strip(),
                               "long": line[77:].strip(), "order": int(line[:5])}
    return codes


@dataclass
class Tabular:
    """The tabular list: chapters, blocks (sections) and the notes of every `diag` element."""

    chapters: dict[str, str] = field(default_factory=dict)                 # "chapter:9" → title
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)        # "block:I30-I5A" → {title, chapter, first, last}
    diags: dict[str, dict[str, Any]] = field(default_factory=dict)         # dotted code → {desc, notes, block, chapter, seven}
    category_block: dict[str, str] = field(default_factory=dict)           # 3-character category → block id


def _notes(element: ET.Element) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for tag in NOTE_TAGS:
        texts = [" ".join((n.text or "").split()) for child in element.findall(tag) for n in child.findall("note")]
        if texts:
            out[tag] = [t for t in texts if t]
    return out


def parse_tabular(source: bytes | str | Path) -> Tabular:
    """Parse `icd10cm_tabular_<year>.xml` (bytes, text or path)."""
    if isinstance(source, Path):
        source = source.read_bytes()
    root = ET.fromstring(source.encode() if isinstance(source, str) else source)
    tab = Tabular()
    for chapter in root.findall("chapter"):
        chapter_id = f"chapter:{chapter.findtext('name').strip()}"
        tab.chapters[chapter_id] = " ".join((chapter.findtext("desc") or "").split())
        for section in chapter.findall("section"):
            raw_id = section.get("id", "").strip()
            first, _, last = raw_id.partition("-")
            block_id = f"block:{raw_id}"
            tab.blocks[block_id] = {"title": " ".join((section.findtext("desc") or "").split()), "chapter": chapter_id,
                                    "first": first, "last": last or first}

            def walk(element: ET.Element, inherited_seven: dict[str, str]) -> None:
                for diag in element.findall("diag"):
                    name = (diag.findtext("name") or "").strip()
                    seven = {e.get("char"): " ".join((e.text or "").split())
                             for d in diag.findall("sevenChrDef") for e in d.findall("extension")}
                    own_seven = seven or inherited_seven
                    tab.diags[name] = {"desc": " ".join((diag.findtext("desc") or "").split()), "notes": _notes(diag),
                                       "block": block_id, "chapter": chapter_id, "seven": own_seven}
                    if len(name) == 3:
                        tab.category_block[name] = block_id
                    walk(diag, own_seven)

            walk(section, {})
    return tab


def read_zip_member(zip_path: Path, member: str) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member)


@dataclass
class Release:
    name: str
    codes: dict[str, dict[str, Any]]
    tabular: Tabular
    _index: dict[str, Any] = field(default_factory=dict, repr=False)

    def node_ids(self) -> set[str]:
        return set(self.codes) | set(self.tabular.blocks) | set(self.tabular.chapters)

    def index(self) -> dict[str, Any]:
        """Cached: parent of every code, children of every node, chapter of every code, codes per chapter."""
        if not self._index:
            parent = {code: code_parent(code, self) for code in self.codes}
            children: dict[str, list[str]] = defaultdict(list)
            for code, up in parent.items():
                children[up].append(code)
            chapter = {code: self.tabular.blocks.get(self.tabular.category_block.get(undotted(code)[:3], ""), {}).get("chapter")
                       for code in self.codes}
            by_chapter: dict[str | None, list[str]] = defaultdict(list)
            for code in sorted(self.codes):
                by_chapter[chapter[code]].append(code)
            self._index.update(parent=parent, children={k: sorted(v) for k, v in children.items()}, chapter=chapter,
                               by_chapter=dict(by_chapter))
        return self._index


def load_release(name: str, raw: Path = RAW) -> Release:
    order_zip, order_member = ORDER_FILES[name]
    tab_zip, tab_member = TABULAR_FILES[name]
    lines = read_zip_member(raw / order_zip, order_member).decode("latin-1").splitlines()
    return Release(name, parse_order(lines), parse_tabular(read_zip_member(raw / tab_zip, tab_member)))


# -- hierarchy -----------------------------------------------------------------------------------------------------------

def code_parent(code: str, release: Release) -> str | None:
    """Parent node id of a code within its release: the longest proper prefix in the code set; a category's block."""
    plain = undotted(code)
    for k in range(len(plain) - 1, 2, -1):          # down to the 3-character category
        candidate = dotted(plain[:k])
        if candidate in release.codes:
            return candidate
    return release.tabular.category_block.get(plain[:3])


def node_parent(node: str, release: Release) -> str | None:
    if node.startswith("chapter:"):
        return None
    if node.startswith("block:"):
        return release.tabular.blocks[node]["chapter"]
    return code_parent(node, release)


def chain(node: str, release: Release) -> list[str]:
    """Ancestors of a node, nearest first, up to its chapter."""
    out, current = [], node_parent(node, release)
    while current is not None and len(out) < 16:
        out.append(current)
        current = node_parent(current, release)
    return out


def chapter_of(code: str, release: Release) -> str | None:
    up = chain(code, release)
    return up[-1] if up and up[-1].startswith("chapter:") else None


def block_map(after: Release, before: Release) -> dict[str, str | None]:
    """FY2027 block id → FY2026 block id: the same id, else the same first category, else the same title."""
    by_first = {b["first"]: bid for bid, b in before.tabular.blocks.items()}
    by_title = {b["title"].rsplit("(", 1)[0].strip().lower(): bid for bid, b in before.tabular.blocks.items()}
    out: dict[str, str | None] = {}
    for bid, block in after.tabular.blocks.items():
        if bid in before.tabular.blocks:
            out[bid] = bid
        elif block["first"] in by_first:
            out[bid] = by_first[block["first"]]
        else:
            out[bid] = by_title.get(block["title"].rsplit("(", 1)[0].strip().lower())
    return out


def definition_of(code: str, release: Release) -> tuple[str, list[str], dict[str, list[str]]]:
    """(definition, inclusion terms, all tabular notes) of a code: title + inclusion terms + includes notes. A code
    without its own `diag` (7th-character extensions) uses its title only, plus the base diag's notes listed in meta."""
    title = release.codes[code]["long"]
    diag = release.tabular.diags.get(code)
    notes = dict(diag["notes"]) if diag else {}
    inclusion = notes.get("inclusionTerm", [])
    includes = notes.get("includes", [])
    parts = [title.rstrip(".") + "."]
    if inclusion:
        parts.append("Inclusion terms: " + "; ".join(inclusion) + ".")
    if includes:
        parts.append("Includes: " + "; ".join(includes) + ".")
    return " ".join(parts), inclusion, notes


# -- items ---------------------------------------------------------------------------------------------------------------

def new_codes(after: Release, before: Release) -> list[str]:
    return sorted(set(after.codes) - set(before.codes), key=lambda c: after.codes[c]["order"])


def placement_items(after: Release, before: Release, *, dev_fraction: float = 0.2) -> list[dict[str, Any]]:
    blocks = block_map(after, before)
    before_nodes = before.node_ids()
    items = []
    for code in new_codes(after, before):
        info = after.codes[code]
        up = chain(code, after)
        mapped = [blocks.get(n, n) if n.startswith("block:") else n for n in up]
        nearest = next((n for n in mapped if n and n in before_nodes), None)
        if nearest is None:
            raise ValueError(f"{code}: no FY2026 ancestor")
        gap = mapped.index(nearest)
        block_after = next((n for n in up if n.startswith("block:")), None)
        block_before = blocks.get(block_after) if block_after else None
        chapter = up[-1]
        definition, inclusion, notes = definition_of(code, after)
        relations = [["parent", nearest]]
        if block_before:
            relations.append(["block", block_before])
        relations.append(["chapter", chapter])
        parent_before = before.codes.get(nearest, {})
        meta = {
            "code": info["code"], "billable": info["billable"], "short_title": info["short"],
            "immediate_parent_fy2027": up[0], "immediate_parent_new": up[0] not in before_nodes, "new_levels_above": gap,
            "parent_billable_in_fy2026": bool(parent_before.get("billable")), "block_fy2027": block_after,
            "block_fy2026": block_before, "chapter": chapter, "chapter_title": after.tabular.chapters.get(chapter),
            "depth": len(up), "split": ld.split_of(code, dev_fraction=dev_fraction),
            "tabular_notes": {k: v for k, v in notes.items() if k not in ("inclusionTerm", "includes")},
            "has_own_diag": code in after.tabular.diags,
        }
        items.append(ld.placement_item(set_name=SET_PLACEMENT, record=code, name=info["long"], aliases=inclusion,
                                       definition=definition, gold_parents=[nearest], gold_relations=relations,
                                       candidates=None, evidence={}, meta=meta))
    return items


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()[:16], 16))


def distractor_codes(code: str, after: Release, *, seed: int, k: int = 3, siblings: int = 2) -> list[str]:
    """Up to `siblings` other children of the code's parent, the rest from the same chapter (never an ancestor or a
    descendant of `code`); titles distinct from the answer's and from each other (case-insensitive)."""
    index = after.index()
    parent, chapter, plain = index["parent"][code], index["chapter"][code], undotted(code)
    titles = {after.codes[code]["long"].casefold()}

    def eligible(candidate: str) -> bool:
        other = undotted(candidate)
        related = plain.startswith(other) or other.startswith(plain)
        return not related and after.codes[candidate]["long"].casefold() not in titles

    rng = _rng(seed, code, "distractors")
    chosen: list[str] = []
    sibling_pool = [c for c in index["children"].get(parent, []) if c != code]
    for candidate in rng.sample(sibling_pool, len(sibling_pool)):
        if len(chosen) >= siblings:
            break
        if eligible(candidate):
            chosen.append(candidate); titles.add(after.codes[candidate]["long"].casefold())
    chapter_pool = [c for c in index["by_chapter"].get(chapter, []) if c != code and c not in chosen]
    for candidate in rng.sample(chapter_pool, len(chapter_pool)):
        if len(chosen) >= k:
            break
        if eligible(candidate):
            chosen.append(candidate); titles.add(after.codes[candidate]["long"].casefold())
    if len(chosen) < k:
        raise ValueError(f"{code}: only {len(chosen)} distractors")
    return chosen


def choice_items(after: Release, codes: Sequence[str], *, seed: int, dev_fraction: float = 0.2) -> list[dict[str, Any]]:
    """`code2title` and `title2code` items (4 options, MedConceptsQA wording for code → title)."""
    items = []
    letters = "ABCD"
    for code in codes:
        distractors = distractor_codes(code, after, seed=seed)
        rng = _rng(seed, code, "order")
        order = [code, *distractors]
        rng.shuffle(order)
        answer = order.index(code)
        title = after.codes[code]["long"]
        parent_of = after.index()["parent"]
        sources = {c: ("sibling" if parent_of[c] == parent_of[code] else "chapter") for c in distractors}
        base = {"code": code, "billable": after.codes[code]["billable"], "option_codes": order,
                "distractor_source": sources, "split": ld.split_of(code, dev_fraction=dev_fraction),
                "vocab": "ICD10CM", "release": "FY2027"}
        question = f"What is the description of the medical code {code} in ICD10CM?"
        titles = [after.codes[c]["long"] for c in order]
        prompt = question + "\n" + "\n".join(f"{letters[i]}. {t}" for i, t in enumerate(titles))
        items.append(ld.rank_item(set_name=SET_CHOICE, item_id=f"{SET_CHOICE}:code2title:{code}", context=question,
                                  options=titles, answer=answer, terms=[{"surface": code, "concept": code}],
                                  meta={**base, "direction": "code2title", "mcq_prompt": prompt, "answer_letter": letters[answer]}))
        question = f"Which ICD10CM code has the description: {title}?"
        prompt = question + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(order))
        items.append(ld.rank_item(set_name=SET_CHOICE, item_id=f"{SET_CHOICE}:title2code:{code}", context=question,
                                  options=order, answer=answer, terms=[{"surface": title, "concept": code}],
                                  meta={**base, "direction": "title2code", "mcq_prompt": prompt, "answer_letter": letters[answer]}))
    return items


# -- snapshot --------------------------------------------------------------------------------------------------------------

def snapshot_rows(release: Release) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    nodes, edges = [], []
    for chapter_id, title in release.tabular.chapters.items():
        nodes.append(ld.snapshot_node(chapter_id, title, kind="chapter"))
    for block_id, block in release.tabular.blocks.items():
        nodes.append(ld.snapshot_node(block_id, block["title"], kind="block", meta={"chapter": block["chapter"]}))
        edges.append((block_id, "parent", block["chapter"]))
    for code, info in release.codes.items():
        definition, inclusion, notes = definition_of(code, release)
        kind = "category" if len(undotted(code)) == 3 else ("code" if info["billable"] else "subcategory")
        nodes.append(ld.snapshot_node(code, info["long"], aliases=inclusion, definition=definition, kind=kind,
                                      meta={"code": info["code"], "billable": info["billable"], "short_title": info["short"],
                                            "tabular_notes": {k: v for k, v in notes.items()
                                                              if k not in ("inclusionTerm", "includes")}}))
        parent = code_parent(code, release)
        if parent is None:
            raise ValueError(f"{release.name}: {code} has no parent")
        edges.append((code, "parent", parent))
    return nodes, edges


# -- UMLS map (licensed; local only) ----------------------------------------------------------------------------------------

def mrconso_lines(meta: Path = UMLS_META) -> Iterator[str]:
    """MRCONSO.RRF from its gzip parts (`MRCONSO.RRF.aa.gz`, `.ab.gz`, …), joined as one byte stream (a part boundary
    may fall inside a line)."""
    parts = sorted(meta.glob("MRCONSO.RRF.*.gz"))
    if not parts:
        raise FileNotFoundError(f"no MRCONSO parts under {meta}")
    tail = b""
    for part in parts:
        with gzip.open(part, "rb") as handle:
            while True:
                block = handle.read(1 << 24)
                if not block:
                    break
                block = tail + block
                lines = block.split(b"\n")
                tail = lines.pop()
                for line in lines:
                    yield line.decode("utf-8")
    if tail:
        yield tail.decode("utf-8")


def normalize_string(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def map_umls(items: Sequence[dict[str, Any]], after: Release, before: Release, lines: Iterable[str] | None = None,
             *, lines_again: Iterable[str] | None = None) -> dict[str, Any]:
    """Parents (and their code ancestors) → CUIs → MeSH / SNOMED CT; new codes' title strings → CUIs → MeSH / SNOMED.
    Two passes over MRCONSO (`lines`, `lines_again`; default: the licensed release in place)."""
    codes = set()
    for item in items:
        for parent in item["gold_parents"]:
            if parent in before.codes:
                codes.add(parent)
                codes.update(n for n in chain(parent, before) if n in before.codes)
    strings: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for text in [item["name"], *item["aliases"]]:
            key = normalize_string(text.replace(", NOS", ""))
            if key:
                strings[key].add(item["record"])
    code_cuis: dict[str, set[str]] = defaultdict(set)
    string_cuis: dict[str, set[str]] = defaultdict(set)
    sab_versions: Counter[str] = Counter()
    for line in (lines if lines is not None else mrconso_lines()):
        f = line.split("|")
        if len(f) < 17:
            continue
        cui, lat, sab, code, text, suppress = f[0], f[1], f[11], f[13], f[14], f[16]
        if sab == "ICD10CM":
            sab_versions["ICD10CM"] += 1
            if code in codes:
                code_cuis[code].add(cui)
        if lat == "ENG" and suppress != "O":
            key = normalize_string(text)
            if key in strings:
                for record in strings[key]:
                    string_cuis[record].add(cui)
    wanted = set().union(*code_cuis.values(), *string_cuis.values()) if (code_cuis or string_cuis) else set()
    targets: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: {"MSH": {}, "SNOMEDCT_US": {}})
    for line in (lines_again if lines_again is not None else mrconso_lines()):
        f = line.split("|")
        if len(f) < 17 or f[0] not in wanted:
            continue
        cui, sab, tty, code, text, suppress = f[0], f[11], f[12], f[13], f[14], f[16]
        if sab not in ("MSH", "SNOMEDCT_US") or suppress == "O":
            continue
        if sab == "MSH" and not code[:1] in ("D", "C"):
            continue
        names = targets[cui][sab]
        preferred = tty in ("MH", "NM", "PT", "FN")
        if code not in names or preferred:
            names[code] = text

    def expand(cuis: set[str]) -> dict[str, Any]:
        mesh = {c: n for cui in sorted(cuis) for c, n in targets[cui]["MSH"].items()}
        snomed = {c: n for cui in sorted(cuis) for c, n in targets[cui]["SNOMEDCT_US"].items()}
        return {"cuis": sorted(cuis), "mesh": dict(sorted(mesh.items())), "snomed": dict(sorted(snomed.items()))}

    parents = [{"code": code, **expand(code_cuis.get(code, set()))} for code in sorted(codes)]
    matches = [{"code": item["record"], "name": item["name"], **expand(string_cuis.get(item["record"], set()))}
               for item in items]
    gold = {p for item in items for p in item["gold_parents"] if p in before.codes}
    summary = {
        "parent_codes": len(gold), "parent_codes_with_cui": sum(1 for p in parents if p["code"] in gold and p["cuis"]),
        "parent_codes_with_mesh": sum(1 for p in parents if p["code"] in gold and p["mesh"]),
        "parent_codes_with_snomed": sum(1 for p in parents if p["code"] in gold and p["snomed"]),
        "codes_with_ancestors": len(parents), "codes_with_ancestors_with_cui": sum(1 for p in parents if p["cuis"]),
        "new_codes": len(items), "new_codes_with_string_cui": sum(1 for m in matches if m["cuis"]),
        "new_codes_with_string_mesh": sum(1 for m in matches if m["mesh"]),
        "new_codes_with_string_snomed": sum(1 for m in matches if m["snomed"]),
        "icd10cm_rows": sab_versions["ICD10CM"],
    }
    return {"parents": parents, "string_matches": matches, "summary": summary}


# -- build -----------------------------------------------------------------------------------------------------------------

def build(out_dir: Path, local_dir: Path, *, raw: Path = RAW, seed: int = 20261008) -> dict[str, Any]:
    after, before = load_release("fy2027", raw), load_release("fy2026", raw)
    items = placement_items(after, before)
    codes = [i["record"] for i in items]
    choice = choice_items(after, codes, seed=seed)
    out_dir = Path(out_dir)
    files = {}
    for split in ("dev", "test"):
        files[f"placement-{split}"] = ld.write_jsonl(out_dir / f"placement-{split}.jsonl",
                                                     [i for i in items if i["meta"]["split"] == split])
        files[f"choice-{split}"] = ld.write_jsonl(out_dir / f"choice-{split}.jsonl",
                                                  [i for i in choice if i["meta"]["split"] == split])
    nodes, edges = snapshot_rows(before)
    snapshot = ld.write_snapshot(Path(local_dir) / "snapshot-fy2026", nodes, edges, description={
        "ontology": "ICD-10-CM", "release": "FY2026 (October 1, 2025; unchanged on April 1, 2026)",
        "source": f"CMS {ORDER_FILES['fy2026'][0]}, {TABULAR_FILES['fy2026'][0]}", "licence": ld.CMS_PD})
    added_billable = sum(1 for c in codes if after.codes[c]["billable"])
    lost_billable = sorted(c for c, v in before.codes.items() if v["billable"] and not after.codes.get(c, {}).get("billable"))
    stats = {
        "fy2026_codes": len(before.codes), "fy2026_billable": sum(v["billable"] for v in before.codes.values()),
        "fy2027_codes": len(after.codes), "fy2027_billable": sum(v["billable"] for v in after.codes.values()),
        "new_codes": len(codes), "new_billable": added_billable, "new_headers": len(codes) - added_billable,
        "fy2026_billable_not_billable_in_fy2027": len(lost_billable),
        "of_which_became_headers": sum(1 for c in lost_billable if c in after.codes),
        "deleted_codes": len(set(before.codes) - set(after.codes)),
        "new_by_chapter": dict(sorted(Counter(i["meta"]["chapter"] for i in items).items(), key=lambda kv: int(kv[0].split(":")[1]))),
        "immediate_parent_new": sum(i["meta"]["immediate_parent_new"] for i in items),
        "gold_parent_kind": dict(Counter("block" if i["gold_parents"][0].startswith("block:") else
                                         ("chapter" if i["gold_parents"][0].startswith("chapter:") else "code")
                                         for i in items)),
        "gold_parent_was_billable_in_fy2026": sum(i["meta"]["parent_billable_in_fy2026"] for i in items),
        "split": dict(Counter(i["meta"]["split"] for i in items)),
        "with_inclusion_terms": sum(1 for i in items if i["aliases"]),
    }
    manifest = {
        "set": SET_PLACEMENT, "version": VERSION, "holdout": "H3 (decision 63; manuscript/toolkit-methodology-2026-10.md §2 M1)",
        "before": "ICD-10-CM FY2026", "after": "ICD-10-CM FY2027 (effective 2026-10-01)",
        "licence": ld.CMS_PD, "sources": [s.__dict__ for s in ld.SOURCES if s.name == "icd10cm"],
        "files": files, "stats": stats, "seed": seed, "split_rule": f"learn_data.split_of(code), salt {ld.SPLIT_SALT!r}, dev 20%",
        "snapshot": {"path": str(Path(local_dir) / "snapshot-fy2026"), "nodes": snapshot["nodes"], "edges": snapshot["edges"],
                     "node_kinds": snapshot["node_kinds"]},
        "choice": {"directions": ["code2title", "title2code"], "options": 4,
                   "distractors": "≤ 2 siblings (same FY2027 parent) + same-chapter codes; never an ancestor or descendant; "
                                  "per-item seeded (sha256 of seed:code)",
                   "context": "MedConceptsQA wording; meta.mcq_prompt holds the lettered prompt"},
        "built": time.strftime("%Y-%m-%d"),
        "command": "PYTHONPATH=src python -m vsa_embed.benchmarks.learn_data icd10cm",
    }
    previous = out_dir / "manifest.json"
    if previous.exists():                      # keep the counts of an earlier `icd10cm-umls` run
        umls = json.loads(previous.read_text()).get("umls_map")
        if umls:
            manifest["umls_map"] = umls
    ld.write_manifest(out_dir, manifest)
    return manifest


def build_umls(out_dir: Path, local_dir: Path, *, raw: Path = RAW, meta: Path = UMLS_META) -> dict[str, Any]:
    after, before = load_release("fy2027", raw), load_release("fy2026", raw)
    items = placement_items(after, before)
    result = map_umls(items, after, before, mrconso_lines(meta), lines_again=mrconso_lines(meta))
    target = ld.private_dir(Path(local_dir) / "umls-2022ab")
    files = {"parents": ld.write_jsonl(target / "parents.jsonl", result["parents"]),
             "string_matches": ld.write_jsonl(target / "new_code_string_matches.jsonl", result["string_matches"])}
    sab = {}
    mrsab = meta / "MRSAB.RRF.gz"
    if mrsab.exists():
        with gzip.open(mrsab, "rt", encoding="utf-8") as handle:
            for line in handle:
                f = line.split("|")
                if len(f) > 21 and f[3] in ("ICD10CM", "MSH", "SNOMEDCT_US") and f[21] == "Y":   # RSAB, CURVER
                    sab[f[3]] = f[2]                                                             # VSAB
    info = {"umls": "2022AB (licensed; read in place)", "source_versions": sab, "summary": result["summary"],
            "files": files, "path": str(target), "built": time.strftime("%Y-%m-%d"),
            "note": "licence-derived: never committed; the repository manifest keeps the summary counts only"}
    (target / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    manifest_path = Path(out_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["umls_map"] = {"summary": result["summary"], "source_versions": sab, "local_path": str(target),
                                "licence": "UMLS 2022AB (licensed): table kept local, counts only here"}
        ld.write_manifest(Path(out_dir), manifest)
    return info


def add_cli(sub: Any) -> None:
    parser = sub.add_parser("icd10cm", help="ICD-10-CM FY2027 placement + choice items (holdout H3)")
    parser.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"{SET_PLACEMENT}-{VERSION}")
    parser.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"{SET_PLACEMENT}-{VERSION}")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.set_defaults(run=lambda a: build(a.out, a.local, raw=a.raw, seed=a.seed))
    umls = sub.add_parser("icd10cm-umls", help="map the H3 parents to UMLS CUIs → MeSH / SNOMED (licensed; local only)")
    umls.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"{SET_PLACEMENT}-{VERSION}")
    umls.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"{SET_PLACEMENT}-{VERSION}")
    umls.add_argument("--raw", type=Path, default=RAW)
    umls.add_argument("--meta", type=Path, default=UMLS_META)
    umls.set_defaults(run=lambda a: build_umls(a.out, a.local, raw=a.raw, meta=a.meta))
