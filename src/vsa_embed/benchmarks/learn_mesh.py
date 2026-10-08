"""The MeSH 2025 → 2026 time split: place the records NLM added in 2026 into MeSH 2025 (a new benchmark; decision 63).

**Releases.** Before: `desc2025.gz` / `supp2025.gz` (NLM's 2025 archive, final 2025 state). After: `desc2026.gz` /
`supp2026.gz` (the files T7 uses, Last-Modified 2026-08-12). New records are UIs present after and absent before.

**Items** (`mesh-2025-2026`, common placement format, one per new record):
- *descriptor*: gold parents = the descriptors owning the immediate parent tree numbers in 2026; a parent that is
  itself new is replaced by its nearest ancestors that exist in 2025 (`meta.parents_new`). Gold relations add the
  pharmacological actions and see-related descriptors that exist in 2025. Definition = the scope note.
- *SCR* (supplementary concept record): gold parents = the HeadingMappedTo descriptors (a heading new in 2026 → its
  nearest 2025 ancestors, `meta.mapped_new`); gold relations = `mapped_to` + `pharmacological_action`. Definition =
  the record note without its bibliographic parts (`e11_read_to_learn.scr_definition`).
- `candidates` is null = every descriptor of the 2025 snapshot.
- **Evidence:** abstracts of T7's local PubMed 2025–26 text (the 20 newest 2026 baseline files + the 80 newest 2026
  update files, PMID ≥ 41,025,505, each PMID once) that mention the record by any of its names — whole tokens, T7's
  matcher (`mesh_novel.mention_key`) and alias policy (≥ 4 characters, no bare abbreviations or function words); a
  name shared with any other 2026 record is not counted. Descriptors also get the number of those abstracts NLM
  indexed with them. `pubmed_2025_26_eval_docs` counts abstracts on T7's evaluation side (PMID bucket < 2,000).
- **Primary set:** ≥ 5 mentioning abstracts. Dev 20% / test 80% by `learn_data.split_of(UI)`; the rest is in
  `placement-low-evidence.jsonl` with its split assigned the same way.
- `meta.t7_selected` / `meta.t7_heldout`: overlap with T7's frozen selection and holdout.

**New edges for known records** (`new-edges.jsonl`): edges added in 2026 between records that both exist in 2025
(descriptor parent / pharmacological action / see-related; SCR mapped heading / pharmacological action) — the
"update the mapping" gold of the *learn* tool.

**Before snapshot:** `snapshot-2025/` (descriptors: nodes + parent / pharmacological_action / see_also edges, and the
repository's ontology tensor format `ontology.pt` from `mesh.build_mesh_track_ontology`, zero training frequencies)
and `snapshot-2025-scr/` (2025 SCRs with mapped_to / pharmacological_action edges into the descriptors).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ontologies import mesh as mesh_mod
from ..ontologies import mesh_novel
from . import learn_data as ld

SET_NAME = "mesh-2025-2026"
VERSION = "v1"
MESH_DIR = ld.DATA_ROOT / "mesh"
BEFORE = {"desc": MESH_DIR / "2025" / "desc2025.gz", "supp": MESH_DIR / "2025" / "supp2025.gz"}
AFTER = {"desc": MESH_DIR / "desc2026.gz", "supp": MESH_DIR / "supp-2026" / "supp2026.gz"}
PUBMED_DIRS = (ld.DATA_ROOT / "pubmed" / "text-updates-2026", ld.DATA_ROOT / "pubmed" / "text-2026")
MIN_PMID = 41_025_505            # T7's `pubmed.min_pmid`
EVAL_BUCKETS = 2_000             # T7's `pubmed.eval_buckets`
MIN_DOCS = 5
T7_SELECTION = Path("experiments/t7-new-vocabulary/screen/selection.tsv")
T7_HOLDOUT = Path("experiments/t7-new-vocabulary/runs/v1/holdout_concepts.txt")
PMID_SAMPLE = 20
ALL_SCR_CLASSES = tuple(mesh_novel.SCR_CLASSES)


# -- the diff ------------------------------------------------------------------------------------------------------------

def tree_owner(descriptors: Iterable[dict[str, Any]]) -> dict[str, str]:
    return {tree: d["ui"] for d in descriptors for tree in d["trees"]}


def nearest_existing(uis: Iterable[str], parents: dict[str, list[str]], existing: set[str], *, depth: int = 32) -> list[str]:
    """Each UI if it exists before, else its nearest ancestors (breadth-first over `parents`) that exist before."""
    out: list[str] = []
    for ui in uis:
        frontier, seen = [ui], set()
        for _ in range(depth):
            found = [u for u in frontier if u in existing]
            if found:
                out.extend(found); break
            seen.update(frontier)
            frontier = [p for u in frontier for p in parents.get(u, []) if p not in seen]
            if not frontier:
                break
    return list(dict.fromkeys(out))


def record_terms(record: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(" ".join(t["string"].split()) for t in record.get("term_info", []) if t.get("string")))


def candidate_keys(records: Sequence[dict[str, Any]], *, descriptors: bool) -> dict[str, set[str]]:
    """UI → mention keys of its names under T7's alias policy (all SCR classes)."""
    policy = mesh_novel.NovelVocabularyPolicy(scr_classes=ALL_SCR_CLASSES, include_descriptors=True,
                                              exclude_descriptor_classes=())
    pairs, _ = (mesh_novel.candidate_aliases(records, [], policy) if descriptors
                else mesh_novel.candidate_aliases([], records, policy))
    keys: dict[str, set[str]] = defaultdict(set)
    for alias, ui in pairs:
        keys[ui].add(mesh_novel.mention_key(alias))
    return keys


def all_keys(descriptors: Sequence[dict[str, Any]], scrs: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """Mention key → owning UIs over every name of a release (to drop names shared between records)."""
    owners: dict[str, set[str]] = defaultdict(set)
    for record in [*descriptors, *scrs]:
        for term in record_terms(record):
            owners[mesh_novel.mention_key(term)].add(record["ui"])
    return owners


def diff(before_desc: Sequence[dict[str, Any]], before_supp: Sequence[dict[str, Any]],
         after_desc: Sequence[dict[str, Any]], after_supp: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """New and deleted records, gold edges of the new ones, and edges added between records existing in both."""
    desc_before = {d["ui"]: d for d in before_desc}
    desc_after = {d["ui"]: d for d in after_desc}
    supp_before = {s["ui"]: s for s in before_supp}
    supp_after = {s["ui"]: s for s in after_supp}
    parents_after = mesh_mod._parents(list(after_desc))
    parents_before = mesh_mod._parents(list(before_desc))
    owner_after = tree_owner(after_desc)
    existing = set(desc_before)
    new_desc = sorted(set(desc_after) - existing)
    new_supp = sorted(set(supp_after) - set(supp_before))
    scr_by_name: dict[str, str] = {}
    for ui, record in supp_before.items():
        for term in record_terms(record):
            scr_by_name.setdefault(term.casefold(), ui)

    descriptors = []
    for ui in new_desc:
        record = desc_after[ui]
        immediate = []
        for tree in record["trees"]:
            if "." in tree:
                owner = owner_after.get(tree.rsplit(".", 1)[0])
                if owner and owner not in immediate:
                    immediate.append(owner)
        gold = nearest_existing(immediate, parents_after, existing)
        promoted = next((scr_by_name[t.casefold()] for t in [record["name"], *record_terms(record)]
                         if t.casefold() in scr_by_name), None)
        descriptors.append({
            "ui": ui, "kind": "descriptor", "name": record["name"], "aliases": record_terms(record),
            "definition": record["note"] or None, "gold_parents": gold,
            "relations": [("parent", p) for p in gold]
                         + [("pharmacological_action", a) for a in record["actions"] if a in existing]
                         + [("see_also", r) for r in record["related"] if r in existing],
            "meta": {"trees_2026": record["trees"], "tree_parents_2026": immediate,
                     "parents_new": [p for p in immediate if p not in existing],
                     "top_branches": sorted({t[0] for t in record["trees"]}), "descriptor_class": record["descriptor_class"],
                     "introduced": record["introduced"], "promoted_from_scr_2025": promoted,
                     "promoted_scr_still_in_2026": bool(promoted and promoted in supp_after),
                     "actions_new": [a for a in record["actions"] if a not in existing],
                     "top_level": not record["trees"] or all("." not in t for t in record["trees"])}})

    scrs = []
    for ui in new_supp:
        record = supp_after[ui]
        mapped = [m for m in record["mapped"] if m in desc_after]
        gold = nearest_existing(mapped, parents_after, existing)
        scrs.append({
            "ui": ui, "kind": "scr", "name": record["name"], "aliases": record_terms(record),
            "note": record["note"], "gold_parents": gold,
            "relations": [("mapped_to", g) for g in gold]
                         + [("pharmacological_action", a) for a in nearest_existing(
                             [a for a in record["actions"] if a in desc_after], parents_after, existing)],
            "meta": {"record_class": mesh_novel.SCR_CLASSES.get(record["scr_class"], record["scr_class"]),
                     "mapped_2026": record["mapped"], "mapped_major": record["mapped_major"],
                     "mapped_new": [m for m in mapped if m not in existing], "actions_2026": record["actions"],
                     "introduced": record["introduced"]}})

    new_edges = []
    for ui in sorted(set(desc_before) & set(desc_after)):
        old, new = desc_before[ui], desc_after[ui]
        for relation, before_targets, after_targets in (
                ("parent", parents_before.get(ui, []), parents_after.get(ui, [])),
                ("pharmacological_action", old["actions"], new["actions"]),
                ("see_also", old["related"], new["related"])):
            for target in after_targets:
                if target not in before_targets and target in existing:
                    new_edges.append({"source": ui, "kind": "descriptor", "relation": relation, "target": target,
                                      "removed_in_2026": sorted(set(before_targets) - set(after_targets))})
    for ui in sorted(set(supp_before) & set(supp_after)):
        old, new = supp_before[ui], supp_after[ui]
        for relation, before_targets, after_targets in (("mapped_to", old["mapped"], new["mapped"]),
                                                        ("pharmacological_action", old["actions"], new["actions"])):
            for target in after_targets:
                if target not in before_targets and target in existing:
                    new_edges.append({"source": ui, "kind": "scr", "relation": relation, "target": target,
                                      "removed_in_2026": sorted(set(before_targets) - set(after_targets))})
    stats = {"descriptors_2025": len(desc_before), "descriptors_2026": len(desc_after),
             "scrs_2025": len(supp_before), "scrs_2026": len(supp_after),
             "new_descriptors": len(new_desc), "new_scrs": len(new_supp),
             "deleted_descriptors": len(set(desc_before) - set(desc_after)),
             "deleted_scrs": len(set(supp_before) - set(supp_after)),
             "new_edges_known_records": dict(Counter(f"{e['kind']}:{e['relation']}" for e in new_edges))}
    return {"descriptors": descriptors, "scrs": scrs, "new_edges": new_edges, "stats": stats}


# -- evidence ------------------------------------------------------------------------------------------------------------

class DocumentMatcher(mesh_novel.MentionCounter):
    """`MentionCounter` returning the set of keys a document holds."""

    def keys_in(self, text: str) -> set[str]:
        found: set[str] = set()
        tokens = mesh_novel.TOKEN.findall(text.lower())
        for i, token in enumerate(tokens):
            longest = self.longest.get(token)
            if not longest:
                continue
            for n in range(1, min(longest, len(tokens) - i) + 1):
                key = " ".join(tokens[i:i + n])
                if key in self.keys:
                    found.add(key)
        return found


_MATCHER: DocumentMatcher | None = None
_INDEXED: frozenset[str] = frozenset()


def _init(keys: list[str], indexed: list[str]) -> None:
    global _MATCHER, _INDEXED
    _MATCHER, _INDEXED = DocumentMatcher(keys), frozenset(indexed)


def _scan_file(job: tuple[str, int]) -> tuple[dict[str, list[int]], dict[str, list[int]], int]:
    """(key → PMIDs mentioning it, descriptor UI → PMIDs indexed with it, documents read) of one parquet file."""
    import pyarrow.parquet as pq
    path, min_pmid = job
    mentions: dict[str, list[int]] = defaultdict(list)
    indexed: dict[str, list[int]] = defaultdict(list)
    documents = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["pmid", "text", "mesh"], batch_size=2048):
        data = batch.to_pydict()
        for pmid, text, headings in zip(data["pmid"], data["text"], data["mesh"]):
            pmid = int(pmid)
            if pmid < min_pmid:
                continue
            documents += 1
            for key in _MATCHER.keys_in(text):
                mentions[key].append(pmid)
            for ui in headings or ():
                if ui in _INDEXED:
                    indexed[ui].append(pmid)
    return dict(mentions), dict(indexed), documents


def pubmed_paths(directories: Sequence[Path] = PUBMED_DIRS) -> list[Path]:
    return [p for d in directories for p in sorted(Path(d).glob("*.parquet"))]


def scan_pubmed(paths: Sequence[Path], keys: Iterable[str], indexed: Iterable[str], *, min_pmid: int = MIN_PMID,
                workers: int = 6) -> tuple[dict[str, set[int]], dict[str, set[int]], int]:
    """PMIDs per key and per indexed descriptor over the files (each PMID once per key)."""
    keys, indexed = sorted(set(keys)), sorted(set(indexed))
    jobs = [(str(p), int(min_pmid)) for p in paths]
    mentions: dict[str, set[int]] = defaultdict(set)
    indexing: dict[str, set[int]] = defaultdict(set)
    pmids: set[int] = set()
    documents = 0

    def merge(result: tuple[dict[str, list[int]], dict[str, list[int]], int]) -> None:
        nonlocal documents
        found, tagged, n = result
        for key, values in found.items():
            mentions[key].update(values)
        for ui, values in tagged.items():
            indexing[ui].update(values)
        documents += n

    if workers <= 1:
        _init(keys, indexed)
        for job in jobs:
            merge(_scan_file(job))
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init,
                                 initargs=(keys, indexed)) as pool:
            for result in pool.map(_scan_file, jobs):
                merge(result)
    return dict(mentions), dict(indexing), documents


def is_eval_pmid(pmid: int) -> bool:
    from ..data.pubmed import pmid_bucket
    return pmid_bucket(pmid) < EVAL_BUCKETS


def evidence_for(keys: set[str], mentions: dict[str, set[int]], indexed: set[int] | None) -> dict[str, Any]:
    pmids = sorted(set().union(*(mentions.get(k, set()) for k in keys))) if keys else []
    evidence = {"pubmed_2025_26_docs": len(pmids), "pubmed_2025_26_eval_docs": sum(1 for p in pmids if is_eval_pmid(p)),
                "pubmed_2025_26_pmids": pmids[:PMID_SAMPLE], "counted_names": len(keys)}
    if indexed is not None:
        evidence["pubmed_indexed_docs"] = len(indexed)
    return evidence


# -- build -----------------------------------------------------------------------------------------------------------------

def read_t7(repo: Path = Path(".")) -> tuple[set[str], set[str]]:
    selection = repo / T7_SELECTION
    holdout = repo / T7_HOLDOUT
    selected = {line.split("\t", 1)[0] for line in selection.read_text().splitlines()[1:] if line} if selection.exists() else set()
    held = set(holdout.read_text().split()) if holdout.exists() else set()
    return selected, held


def make_items(changes: dict[str, Any], *, evidence: dict[str, dict[str, Any]], t7_selected: set[str], t7_heldout: set[str],
               definitions: dict[str, str | None], min_docs: int = MIN_DOCS) -> list[dict[str, Any]]:
    items = []
    for record in [*changes["descriptors"], *changes["scrs"]]:
        ui = record["ui"]
        found = evidence.get(ui, {"pubmed_2025_26_docs": 0})
        meta = {"kind": record["kind"], **record["meta"], "split": ld.split_of(ui),
                "primary": found["pubmed_2025_26_docs"] >= min_docs and bool(record["gold_parents"]),
                "t7_selected": ui in t7_selected, "t7_heldout": ui in t7_heldout}
        if not record["gold_parents"]:
            meta["excluded"] = "no gold parent in 2025 (top-level descriptor or SCR without a descriptor heading)"
        definition = record.get("definition") if record["kind"] == "descriptor" else definitions.get(ui)
        items.append(ld.placement_item(set_name=SET_NAME, record=ui, name=record["name"], aliases=record["aliases"],
                                       definition=definition, gold_parents=record["gold_parents"],
                                       gold_relations=[list(r) for r in record["relations"]], candidates=None,
                                       evidence=found, meta=meta))
    return items


def snapshot_rows(descriptors: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    parents = mesh_mod._parents(list(descriptors))
    known = {d["ui"] for d in descriptors}
    nodes, edges = [], []
    for d in descriptors:
        nodes.append(ld.snapshot_node(d["ui"], d["name"], aliases=record_terms(d), definition=d["note"] or None,
                                      kind="descriptor", meta={"trees": d["trees"], "descriptor_class": d["descriptor_class"],
                                                               "introduced": d["introduced"]}))
        edges += [(d["ui"], "parent", p) for p in parents[d["ui"]]]
        edges += [(d["ui"], "pharmacological_action", a) for a in d["actions"] if a in known]
        edges += [(d["ui"], "see_also", r) for r in d["related"] if r in known]
    return nodes, edges


def scr_snapshot_rows(scrs: Sequence[dict[str, Any]], known: set[str]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """SCR nodes plus descriptor stubs (kind `descriptor_ref`) so the edges close over the node set."""
    nodes, edges = [], []
    for s in scrs:
        nodes.append(ld.snapshot_node(s["ui"], s["name"], aliases=record_terms(s), definition=s["note"] or None, kind="scr",
                                      meta={"record_class": mesh_novel.SCR_CLASSES.get(s["scr_class"], s["scr_class"]),
                                            "introduced": s["introduced"]}))
        edges += [(s["ui"], "mapped_to", m) for m in s["mapped"] if m in known]
        edges += [(s["ui"], "pharmacological_action", a) for a in s["actions"] if a in known]
    targets = sorted({t for _, _, t in edges})
    nodes += [ld.snapshot_node(t, t, kind="descriptor_ref") for t in targets]
    return nodes, edges


def write_tensor_ontology(path: Path, desc_path: Path) -> dict[str, Any]:
    """The 2025 descriptors in the trainer's channel-ontology format (`t1_open_corpus.channel_ontology`; T1 alias policy,
    no holdout, zero training frequencies)."""
    import numpy as np
    import torch
    from ..experiments.t1_open_corpus import channel_ontology
    from ..span_channel import AliasTable
    ontology = mesh_mod.build_mesh_track_ontology(Path(desc_path))
    table = AliasTable.from_pairs(ontology.alias_pairs)
    record = channel_ontology(table, ontology, np.zeros(len(table.entry_concepts), dtype=np.int64), "none (before snapshot)")
    record["metadata"] = {"source": str(desc_path), "release": "MeSH 2025", "train_frequency": "zeros (no corpus)"}
    torch.save(record, path)
    return {"file": path.name, "sha256": ld.file_sha256(path), "entries": record["entry_count"],
            "atomics": record["atomic_count"], "relations": record["relation_names"], "alias_table_sha256": record["alias_table_sha256"]}


def build(out_dir: Path, local_dir: Path, *, workers: int = 6, min_docs: int = MIN_DOCS, pubmed: Sequence[Path] | None = None,
          before: dict[str, Path] = BEFORE, after: dict[str, Path] = AFTER, tensor: bool = True) -> dict[str, Any]:
    started = time.time()
    desc_before = sorted(mesh_mod._parse(before["desc"]), key=lambda r: r["ui"])
    desc_after = sorted(mesh_mod._parse(after["desc"]), key=lambda r: r["ui"])
    supp_after = sorted(mesh_novel.parse_supplementary(after["supp"]), key=lambda r: r["ui"])
    supp_before = sorted(mesh_novel.parse_supplementary(before["supp"]), key=lambda r: r["ui"])
    changes = diff(desc_before, supp_before, desc_after, supp_after)
    new_desc = {r["ui"] for r in changes["descriptors"]}
    new_records = [desc for desc in desc_after if desc["ui"] in new_desc]
    new_supp = {r["ui"] for r in changes["scrs"]}
    new_scr_records = [s for s in supp_after if s["ui"] in new_supp]
    keys = {**candidate_keys(new_records, descriptors=True), **candidate_keys(new_scr_records, descriptors=False)}
    owners = all_keys(desc_after, supp_after)
    shared = Counter()
    for ui in list(keys):
        own = {k for k in keys[ui] if owners.get(k, {ui}) <= {ui}}
        shared[ui] = len(keys[ui]) - len(own)
        keys[ui] = own
    paths = list(pubmed) if pubmed is not None else pubmed_paths()
    mentions, indexing, documents = scan_pubmed(paths, set().union(*keys.values()) if keys else set(), new_desc,
                                                workers=workers)
    evidence = {ui: evidence_for(keys.get(ui, set()), mentions, indexing.get(ui, set()) if ui in new_desc else None)
                for ui in [*new_desc, *new_supp]}
    for ui, count in shared.items():
        evidence[ui]["names_shared_with_other_records"] = count
    from ..experiments.e11_read_to_learn import scr_definition
    definitions = {r["ui"]: scr_definition(r["note"]) for r in changes["scrs"]}
    t7_selected, t7_heldout = read_t7()
    items = make_items(changes, evidence=evidence, t7_selected=t7_selected, t7_heldout=t7_heldout, definitions=definitions,
                       min_docs=min_docs)
    out_dir, local_dir = Path(out_dir), Path(local_dir)
    primary = [i for i in items if i["meta"]["primary"]]
    files = {f"placement-{split}": ld.write_jsonl(out_dir / f"placement-{split}.jsonl",
                                                  [i for i in primary if i["meta"]["split"] == split])
             for split in ("dev", "test")}
    files["placement-low-evidence"] = ld.write_jsonl(out_dir / "placement-low-evidence.jsonl",
                                                     [i for i in items if not i["meta"]["primary"]])
    edges = [{"id": f"{SET_NAME}:edge:{e['source']}:{e['relation']}:{e['target']}", "set": SET_NAME, **e,
              "meta": {"split": ld.split_of(e["source"])}} for e in changes["new_edges"]]
    files["new-edges"] = ld.write_jsonl(out_dir / "new-edges.jsonl", edges)
    nodes, snapshot_edges = snapshot_rows(desc_before)
    snapshot = ld.write_snapshot(local_dir / "snapshot-2025", nodes, snapshot_edges, description={
        "ontology": "MeSH descriptors", "release": "MeSH 2025 (NLM 2025 archive)", "source": str(before["desc"]),
        "licence": ld.NLM_TERMS})
    if tensor:
        snapshot["ontology_pt"] = write_tensor_ontology(local_dir / "snapshot-2025" / "ontology.pt", before["desc"])
    scr_nodes, scr_edges = scr_snapshot_rows(supp_before, {d["ui"] for d in desc_before})
    scr_snapshot = ld.write_snapshot(local_dir / "snapshot-2025-scr", scr_nodes, scr_edges, description={
        "ontology": "MeSH supplementary concept records", "release": "MeSH 2025", "source": str(before["supp"]),
        "licence": ld.NLM_TERMS, "note": "descriptor_ref nodes stand for descriptors of snapshot-2025"})

    def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {"items": len(rows), "by_kind": dict(Counter(i["meta"]["kind"] for i in rows)),
                "by_split": dict(Counter(i["meta"]["split"] for i in rows)),
                "scr_classes": dict(Counter(i["meta"].get("record_class") for i in rows if i["meta"]["kind"] == "scr")),
                "t7_selected": sum(i["meta"]["t7_selected"] for i in rows), "t7_heldout": sum(i["meta"]["t7_heldout"] for i in rows),
                "with_definition": sum(1 for i in rows if i["definition"]),
                "with_pharmacological_action": sum(1 for i in rows if any(r[0] == "pharmacological_action" for r in i["gold_relations"])),
                "parents_or_mapped_new": sum(1 for i in rows if i["meta"].get("parents_new") or i["meta"].get("mapped_new"))}

    rest = [i for i in items if not i["meta"]["primary"]]
    t7_new = {ui for ui in t7_selected if ui in new_supp or ui in new_desc}
    manifest = {
        "set": SET_NAME, "version": VERSION, "built": time.strftime("%Y-%m-%d"),
        "before": {k: str(v) for k, v in before.items()}, "after": {k: str(v) for k, v in after.items()},
        "licence": ld.NLM_TERMS + "; items carry MeSH UIs, names, scope notes and SCR notes (modified: bibliographic parts removed)",
        "acknowledgement": "MeSH and PubMed courtesy of the U.S. National Library of Medicine",
        "stats": changes["stats"], "files": files,
        "primary": {**summary(primary), "rule": f"≥ {min_docs} mentioning abstracts and ≥ 1 gold parent in 2025"},
        "low_evidence": {**summary(rest), "zero_docs": sum(1 for i in rest if i["evidence"]["pubmed_2025_26_docs"] == 0),
                         "docs_1_to_4": sum(1 for i in rest if 0 < i["evidence"]["pubmed_2025_26_docs"] < min_docs),
                         "no_gold_parent": sum(1 for i in items if not i["gold_parents"])},
        "evidence": {"pubmed_files": [p.name for p in paths], "documents_scanned": documents, "min_pmid": MIN_PMID,
                     "matcher": "mesh_novel.mention_key whole tokens; T7 alias policy (all SCR classes); names shared with "
                                "another 2026 record dropped", "eval_side": f"pmid_bucket < {EVAL_BUCKETS}",
                     "pmid_sample": f"first {PMID_SAMPLE} PMIDs (ascending) per record"},
        "t7_overlap": {"t7_selected_records": len(t7_selected), "t7_selected_that_are_new_in_2026": len(t7_new),
                       "t7_heldout_that_are_new_in_2026": len(t7_heldout & (new_supp | new_desc)),
                       "primary_items_t7_selected": sum(i["meta"]["t7_selected"] for i in primary),
                       "primary_items_t7_heldout": sum(i["meta"]["t7_heldout"] for i in primary)},
        "split_rule": f"learn_data.split_of(UI), salt {ld.SPLIT_SALT!r}, dev 20%",
        "snapshot": {"path": str(local_dir / "snapshot-2025"), "nodes": snapshot["nodes"], "edges": snapshot["edges"],
                     "relations": snapshot["relations"], "ontology_pt": snapshot.get("ontology_pt"),
                     "scr_path": str(local_dir / "snapshot-2025-scr"), "scr_nodes": scr_snapshot["nodes"],
                     "scr_edges": scr_snapshot["edges"]},
        "seconds": round(time.time() - started, 1),
        "command": "PYTHONPATH=src python -m vsa_embed.benchmarks.learn_data mesh",
    }
    ld.write_manifest(out_dir, manifest)
    return manifest


def add_cli(sub: Any) -> None:
    parser = sub.add_parser("mesh", help="MeSH 2025 → 2026 placement items (new descriptors and SCRs)")
    parser.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"{SET_NAME}-{VERSION}")
    parser.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"{SET_NAME}-{VERSION}")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-docs", type=int, default=MIN_DOCS)
    parser.add_argument("--no-tensor", action="store_true", help="skip ontology.pt")
    parser.set_defaults(run=lambda a: build(a.out, a.local, workers=a.workers, min_docs=a.min_docs, tensor=not a.no_tensor))
