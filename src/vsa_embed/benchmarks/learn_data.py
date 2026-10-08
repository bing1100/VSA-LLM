"""Evaluation data for the *learn* tool and holdout H3 (decision 63, work package TK-H3L).

Five item sets, each placing records that are new in a later release of an ontology into the earlier ("before")
release, plus two ranking sets in the MedConceptsQA format:

| Set | Builder | Before → after | Licence of the items |
|---|---|---|---|
| `mesh-2025-2026` | `learn_mesh` | MeSH 2025 → 2026 (new descriptors, new SCRs; evidence from PubMed 2025–26) | public domain (NLM) |
| `mesh-2024-2025` (secondary) | `learn_mesh --split 2024-2025` | MeSH 2024 → 2025 (same code) | public domain (NLM) |
| `icd10cm-fy2027` | `learn_icd10cm` | ICD-10-CM FY2026 → FY2027 (holdout H3; placement + code↔title choice) | public domain (CMS/NCHS) |
| `medconceptsqa-icd10cm` | `learn_public` | MedConceptsQA (Shoham & Rappoport 2024), ICD-10-CM subsets | Apache-2.0 |
| `oet-snomed-{disease,cpp}` | `learn_public` | SNOMED CT US 2014-09-01 → 2017-03-01 (Dong et al., CIKM 2023) | SNOMED/UMLS-derived: local only |
| `taxoexpan-semeval-{noun,verb}`, `tmn-wordnet-{noun,verb}` | `learn_public` | WordNet 3.0 + SemEval-2016 Task 14 (TaxoExpan; TMN protocol) | Apache-2.0 code; WordNet / Wiktionary-derived data: local only |

**Placement item** (one per new record; `validate_placement`):

    {"id", "set", "record", "name", "aliases": [...], "definition": str | null,
     "gold_parents": [ids], "gold_relations": [[relation, filler id], ...],
     "candidates": [ids] | null (= every node of the before snapshot), "evidence": {source: count | ids}, "meta": {...}}

**Ranking item** (`validate_rank`): {"id", "set", "kind": "rank", "context", "options": [str], "answer": int,
"terms": [{"surface", "concept"}], "meta"} — options are scored by log-likelihood after `context`.

**Before snapshot** (`write_snapshot`): `nodes.jsonl` ({"id", "name", "aliases", "definition", "kind", "meta"}) and
`edges.jsonl` ({"source", "relation", "target"}: `source --relation--> target`, e.g. a child's `parent`), plus
`snapshot.json` (counts and sha256). Where the repository has an ontology tensor format for the ontology (MeSH:
`t1_open_corpus.channel_ontology`), `ontology.pt` is written as well.

Splits are frozen by sha256: `split_of(record)` hashes `"{salt}:{record}"`; the split files' digests are pinned in
each set's `manifest.json`. Every download is listed in `SOURCES` with its sha256 (`fetch`), and recorded in
`~/data/vsa-llm/DATA_SOURCES.md` / `SHA256SUMS`.

    python -m vsa_embed.benchmarks.learn_data fetch [--source NAME ...]
    python -m vsa_embed.benchmarks.learn_data mesh | icd10cm | icd10cm-umls | medconceptsqa | oet | taxo [options]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DATA_ROOT = Path("~/data/vsa-llm").expanduser()
LOCAL_ROOT = DATA_ROOT / "toolkit-learn"                   # licence-restricted or large outputs (never committed)
REPO_ITEMS = Path("experiments/toolkit-learn/items")       # committed item sets
SPLIT_SALT = "toolkit-learn-v1"
PLACEMENT_KEYS = ("id", "set", "record", "name", "aliases", "definition", "gold_parents", "gold_relations",
                  "candidates", "evidence", "meta")
RANK_KEYS = ("id", "set", "kind", "context", "options", "answer", "terms", "meta")


# -- downloads -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """One downloaded file: where it comes from, where it lives under `DATA_ROOT`, and its pinned sha256."""

    name: str
    url: str
    path: str
    sha256: str
    licence: str


GDRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
CMS = "https://www.cms.gov/files/zip/"
MESH_2025 = "https://nlmpubs.nlm.nih.gov/projects/mesh/2025/xmlmesh/"
MESH_2024 = "https://nlmpubs.nlm.nih.gov/projects/mesh/2024/xmlmesh/"
MCQA_REVISION = "98c30d83762e51a397c9a7b0eeee6722e751da17"
MCQA = f"https://huggingface.co/datasets/ofir408/MedConceptsQA/resolve/{MCQA_REVISION}/"
NLM_TERMS = "NLM MeSH terms and conditions (free of charge; acknowledge NLM; mark modified records)"
CMS_PD = "public domain (CMS / NCHS ICD-10-CM)"

SOURCES: tuple[Source, ...] = (
    Source("mesh2024", MESH_2024 + "desc2024.gz", "mesh/2024/desc2024.gz",
           "f141dff3ac09ca325b4215fa9a4380cd582078ef23f4f0efb320b994bf5ed80e", NLM_TERMS),
    Source("mesh2024", MESH_2024 + "supp2024.gz", "mesh/2024/supp2024.gz",
           "0c56efb58e70ba97ebe53fbf0a4b6f5fa2e26faaf89db7a05c856f00f7e16bec", NLM_TERMS),
    Source("mesh2024", MESH_2024 + "desc2024.zip", "mesh/2024/desc2024.zip",
           "c001a3473ccca774e65901b5eb25d81d5f3d5074ae497a6e8ceba825928522da", NLM_TERMS),
    Source("mesh2024", MESH_2024 + "supp2024.zip", "mesh/2024/supp2024.zip",
           "688dbd053b5be4e9a8b41d94c02b424edd0238eb76bc4441b28cf174a8a420d6", NLM_TERMS),
    Source("mesh2025", MESH_2025 + "desc2025.gz", "mesh/2025/desc2025.gz",
           "99285a75544779de23747f062830b8e36746bdfc5fc44a25ece5855172439c8f", NLM_TERMS),
    Source("mesh2025", MESH_2025 + "supp2025.gz", "mesh/2025/supp2025.gz",
           "211de9e7f4b4bf8b927ae3b68aa580ae8439afd40ed26a4346c9c322651662fe", NLM_TERMS),
    Source("mesh2025", MESH_2025 + "desc2025.zip", "mesh/2025/desc2025.zip",
           "8a7c9696bcbfdcafe9358635115246c1dba8b20af9d8319043d6a99f2aa19b56", NLM_TERMS),
    Source("mesh2025", MESH_2025 + "supp2025.zip", "mesh/2025/supp2025.zip",
           "c65857e60355b31c7e25205a019c67d17e08d245f9dc79fba0be4074ff4b9765", NLM_TERMS),
    Source("icd10cm", CMS + "2027-code-descriptions-tabular-order.zip", "icd10cm/raw/2027-code-descriptions-tabular-order.zip",
           "91c6c9d1117764ce72375a0f3a5493b1725dafbc1ab283b55799076c9e194965", CMS_PD),
    Source("icd10cm", CMS + "2027-code-tables-tabular-index.zip", "icd10cm/raw/2027-code-tables-tabular-index.zip",
           "37baa476323714be16f95c9b2c96812bf6f3623d9f15188866e584dc529b0298", CMS_PD),
    Source("icd10cm", CMS + "2027-icd-10-addendum.zip", "icd10cm/raw/2027-icd-10-addendum.zip",
           "16986a34d1458e549217229686f1a487cf6419ee6acb1043fbc770f8fabdd026", CMS_PD),
    Source("icd10cm", CMS + "2027-conversion-table.zip", "icd10cm/raw/2027-conversion-table.zip",
           "9478b8f95e137177e18be3df906c35566674b16313b12740d9e8e3778cb0e67a", CMS_PD),
    Source("icd10cm", CMS + "2027-version-update-summary.zip", "icd10cm/raw/2027-version-update-summary.zip",
           "c26b19c9246f24ecaccf056eb21640bf1cdded07527910e24bb1d006494f2860", CMS_PD),
    Source("icd10cm", CMS + "2026-code-descriptions-tabular-order.zip", "icd10cm/raw/2026-code-descriptions-tabular-order.zip",
           "55a9124a27ca78a4a5c41e1101eda064ec2e08d105ff9e78d1fbd1a83298ef11", CMS_PD),
    Source("icd10cm", CMS + "april-1-2026-code-descriptions-tabular-order.zip",
           "icd10cm/raw/april-1-2026-code-descriptions-tabular-order.zip",
           "4fd9d8b37f02ab42827c7e7be30595c005b0cc3a6bae7a515e3f4c86b6918688", CMS_PD),
    Source("icd10cm", CMS + "april-1-2026-code-tables-tabular-index.zip", "icd10cm/raw/april-1-2026-code-tables-tabular-index.zip",
           "c623e47143c156d1427fe7cd2bee7ac1ae6994bdd2196b9ec27d4a3219bcfd48", CMS_PD),
    Source("medconceptsqa", MCQA + "README.md", "toolkit-learn/raw/medconceptsqa/README.md",
           "839c51282b44479ca46efd96dea5e3a835f3475fb0681c258df6a9ebe2c26b19", "Apache-2.0"),
    *(Source("medconceptsqa", MCQA + f"icd10cm_{level}/{split}-00000-of-00001.parquet",
             f"toolkit-learn/raw/medconceptsqa/icd10cm_{level}/{split}-00000-of-00001.parquet", digest, "Apache-2.0")
      for level, split, digest in (
          ("easy", "dev", "9be7be36508147f6ab51d4f4349e5d2e8239846b28ef192d8fabe830fdff595a"),
          ("easy", "test", "1da61c45ba7ec859e5f59a7be6ef7abac2b49be644f1a99ff4e05bd065dba9e4"),
          ("medium", "dev", "158028474d476fd2ced8f847a20eadc183f5835d526da890b06b70fee2c569aa"),
          ("medium", "test", "4220d5bb4b74649343f7772fb635aeb96255533c167c553132b76e0c6e4fa410"),
          ("hard", "dev", "f284946ff946dea0eab813450ecc92e258180d9c48621cf48cdca564528e9869"),
          ("hard", "test", "68e99a24c54c33852d2c4d6eba8bf1ccb32a3052f6a52c6bfeae150a546566d5"))),
    Source("oet", "https://zenodo.org/api/records/10432003/files/OET-data-ver4.zip/content",
           "toolkit-learn/raw/oet/OET-data-ver4.zip",
           "a8bd20cabf34d1b46ea3b6d65c165f1e60ad83b69441489e3ff19cfe409708a1",
           "CC-BY-4.0 (Zenodo 10.5281/zenodo.10432003); contains SNOMED CT-derived content (licence held)"),
    Source("taxoexpan", GDRIVE + "15N9C_RMZTLDV8QDjzNJnvrr3RjCQjofF", "toolkit-learn/raw/taxoexpan/README.txt",
           "55eb2b0e85c93ba5ab2aa8c69a754b8a7ab159f52ca68d0f76f79c7a3955db75", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "152XBONrZikBXDQ5Yxq_xYe9LqPcOJxZH", "toolkit-learn/raw/taxoexpan/semeval-noun/wordnet_noun.terms",
           "0f598d076eaf27d5031403dc9efc1693c5c3270ffab320779c2fd36f254234a8", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "150wgawZjXJXDFAH1CWhM5PWjkNsE9ns7", "toolkit-learn/raw/taxoexpan/semeval-noun/wordnet_noun.taxo",
           "50e97c3aaad010a486f9f98498f9568adc17c5001dd30f9789c85079e6609640", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "14rvP3_lvCqZYoGvkTbeLXQt_UcFJ7BhG",
           "toolkit-learn/raw/taxoexpan/semeval-noun/wordnet_noun.fasttext_mode4.pickle.bin",
           "c8488ebe7ab09813728010e44d216ed81d674f316f9b96c5a2e515af77219950", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "14pVBj9zrs5JFY0mGyg0SkDHjQtn9gbf5", "toolkit-learn/raw/taxoexpan/semeval-verb/wordnet_verb.terms",
           "2d39d05e1ad6efac09f1873e8822897f285428657c2337752bd2760da1794e00", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "14nID4CtxfRwsPgS3vDiTxV21j0khIH3O", "toolkit-learn/raw/taxoexpan/semeval-verb/wordnet_verb.taxo",
           "ebb822f30cc0e646979dc9ad5a026094495accbbcf6334ebef902086fdf063c7", "TaxoExpan data (repo: Apache-2.0)"),
    Source("taxoexpan", GDRIVE + "14YhlfErX7dtixsYl2dxzBhFRVoQoeLAj",
           "toolkit-learn/raw/taxoexpan/semeval-verb/wordnet_verb.fasttext_mode4.pickle.bin",
           "4076ef79d4ce6325829c511384abe14ec81fcb6fe3c9f47562c9ba856ad98e5a", "TaxoExpan data (repo: Apache-2.0)"),
)

#: Sources that could not be retrieved (2026-10-08) and why.
UNAVAILABLE = {
    "tmn-wordnet-noun": "https://drive.google.com/file/d/1S6ijwV7phg6ZlJbUgSZjPuJTcN98bWwe (TMN README) → HTTP 404",
    "tmn-wordnet-verb": "https://drive.google.com/file/d/13LqeaaPq6vS8ah-dgkJO2eWGp107eSfT (TMN README) → HTTP 404",
}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_source(source: Source, root: Path = DATA_ROOT, *, timeout: int = 1800) -> dict[str, Any]:
    """Download `source` unless the file is present with the pinned sha256; the file is hashed twice (two reads) and
    must match the pin and the server's Content-Length. Returns the record kept in `fetch.json`."""
    import requests
    path = root / source.path
    if path.exists() and file_sha256(path) == source.sha256:
        return {"path": source.path, "status": "present", "sha256": source.sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with requests.get(source.url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        with open(partial, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
        headers = {k: response.headers.get(k) for k in ("Content-Length", "Last-Modified", "ETag", "Content-Type")}
    size = partial.stat().st_size
    if length is not None and int(length) != size:
        raise IOError(f"{source.url}: {size} bytes, Content-Length {length}")
    first, second = file_sha256(partial), file_sha256(partial)
    if first != second or first != source.sha256:
        raise IOError(f"{source.url}: sha256 {first} / {second}, pinned {source.sha256}")
    os.replace(partial, path)
    return {"path": source.path, "url": source.url, "bytes": size, "sha256": first, "headers": headers,
            "retrieved": time.strftime("%Y-%m-%d"), "status": "downloaded"}


# -- items -------------------------------------------------------------------------------------------------------------

def split_of(record: str, *, dev_fraction: float = 0.2, salt: str = SPLIT_SALT) -> str:
    """`dev` or `test`, from the sha256 of `"{salt}:{record}"` (independent of every other record and of order)."""
    value = int(hashlib.sha256(f"{salt}:{record}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "dev" if value < dev_fraction else "test"


def placement_item(*, set_name: str, record: str, name: str, aliases: Iterable[str] = (), definition: str | None = None,
                   gold_parents: Iterable[str] = (), gold_relations: Iterable[Sequence[str]] = (),
                   candidates: Sequence[str] | None = None, evidence: dict[str, Any] | None = None,
                   meta: dict[str, Any] | None = None, item_id: str | None = None) -> dict[str, Any]:
    """A placement item in the common format (aliases deduplicated in order, the name excluded)."""
    seen, kept = {name.casefold()}, []
    for alias in aliases:
        if alias and alias.casefold() not in seen:
            seen.add(alias.casefold()); kept.append(alias)
    parents = list(dict.fromkeys(gold_parents))
    relations = [list(r) for r in dict.fromkeys(tuple(r) for r in gold_relations)]
    item = {"id": item_id or f"{set_name}:{record}", "set": set_name, "record": record, "name": name, "aliases": kept,
            "definition": definition or None, "gold_parents": parents, "gold_relations": relations,
            "candidates": list(candidates) if candidates is not None else None, "evidence": dict(evidence or {}),
            "meta": dict(meta or {})}
    validate_placement(item)
    return item


def validate_placement(item: dict[str, Any]) -> None:
    missing = [k for k in PLACEMENT_KEYS if k not in item]
    if missing:
        raise ValueError(f"placement item {item.get('id')}: missing {missing}")
    if not isinstance(item["record"], str) or not item["record"] or not isinstance(item["name"], str):
        raise ValueError(f"placement item {item['id']}: record and name must be strings")
    if not isinstance(item["aliases"], list) or not isinstance(item["gold_parents"], list):
        raise ValueError(f"placement item {item['id']}: aliases and gold_parents must be lists")
    for relation in item["gold_relations"]:
        if not (isinstance(relation, list) and len(relation) == 2 and all(isinstance(x, str) for x in relation)):
            raise ValueError(f"placement item {item['id']}: bad relation {relation!r}")
    if item["candidates"] is not None and not isinstance(item["candidates"], list):
        raise ValueError(f"placement item {item['id']}: candidates must be a list or null")
    if not isinstance(item["evidence"], dict) or not isinstance(item["meta"], dict):
        raise ValueError(f"placement item {item['id']}: evidence and meta must be objects")


def rank_item(*, set_name: str, item_id: str, context: str, options: Sequence[str], answer: int,
              terms: Iterable[dict[str, Any]] = (), meta: dict[str, Any] | None = None) -> dict[str, Any]:
    item = {"id": item_id, "set": set_name, "kind": "rank", "context": context, "options": list(options),
            "answer": int(answer), "terms": [dict(t) for t in terms], "meta": dict(meta or {})}
    validate_rank(item)
    return item


def validate_rank(item: dict[str, Any]) -> None:
    missing = [k for k in RANK_KEYS if k not in item]
    if missing:
        raise ValueError(f"rank item {item.get('id')}: missing {missing}")
    options = item["options"]
    if item["kind"] != "rank" or len(options) < 2 or not all(isinstance(o, str) and o for o in options):
        raise ValueError(f"rank item {item['id']}: needs kind 'rank' and ≥ 2 non-empty options")
    if len({o.casefold() for o in options}) != len(options):
        raise ValueError(f"rank item {item['id']}: duplicate options")
    if not 0 <= item["answer"] < len(options):
        raise ValueError(f"rank item {item['id']}: answer {item['answer']} out of range")
    for term in item["terms"]:
        if set(term) != {"surface", "concept"} or not isinstance(term["surface"], str):
            raise ValueError(f"rank item {item['id']}: bad term {term!r}")


def dumps(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write items (one sorted-key JSON object per line; `.gz` → gzip with mtime 0, so the bytes are reproducible).
    Returns {"file", "items", "sha256"} where sha256 is that of the uncompressed payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    if path.suffix == ".gz":
        with open(path, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            for item in items:
                line = (dumps(item) + "\n").encode()
                handle.write(line); digest.update(line); count += 1
    else:
        with open(path, "wb") as handle:
            for item in items:
                line = (dumps(item) + "\n").encode()
                handle.write(line); digest.update(line); count += 1
    return {"file": path.name, "items": count, "sha256": digest.hexdigest()}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    return path


def private_dir(path: Path) -> Path:
    """Create a directory readable by the owner only (licence-derived outputs, as `t1c/`)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


# -- before snapshots ----------------------------------------------------------------------------------------------------

def snapshot_node(node_id: str, name: str, *, aliases: Iterable[str] = (), definition: str | None = None, kind: str = "",
                  meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": node_id, "name": name, "aliases": [a for a in dict.fromkeys(aliases) if a and a != name],
            "definition": definition or None, "kind": kind, "meta": dict(meta or {})}


def write_snapshot(out_dir: Path, nodes: Iterable[dict[str, Any]], edges: Iterable[tuple[str, str, str]], *,
                   description: dict[str, Any], gzip_files: bool = True) -> dict[str, Any]:
    """Write the before release as `nodes.jsonl[.gz]` and `edges.jsonl[.gz]` (edges sorted, deduplicated) plus
    `snapshot.json`. Every edge endpoint must be a node."""
    out_dir = Path(out_dir)
    nodes = sorted(nodes, key=lambda n: n["id"])
    ids = {n["id"] for n in nodes}
    if len(ids) != len(nodes):
        raise ValueError("duplicate node ids in snapshot")
    edge_rows = sorted(set((str(s), str(r), str(t)) for s, r, t in edges))
    dangling = [(s, r, t) for s, r, t in edge_rows if s not in ids or t not in ids]
    if dangling:
        raise ValueError(f"{len(dangling)} snapshot edges with an endpoint outside the nodes, e.g. {dangling[:3]}")
    suffix = ".jsonl.gz" if gzip_files else ".jsonl"
    node_info = write_jsonl(out_dir / f"nodes{suffix}", nodes)
    edge_info = write_jsonl(out_dir / f"edges{suffix}", ({"source": s, "relation": r, "target": t} for s, r, t in edge_rows))
    relations: dict[str, int] = {}
    for _, relation, _ in edge_rows:
        relations[relation] = relations.get(relation, 0) + 1
    kinds: dict[str, int] = {}
    for node in nodes:
        kinds[node["kind"]] = kinds.get(node["kind"], 0) + 1
    info = {**description, "nodes": node_info, "edges": edge_info, "node_kinds": dict(sorted(kinds.items())),
            "relations": dict(sorted(relations.items())),
            "format": "nodes: {id, name, aliases, definition, kind, meta}; edges: {source, relation, target} "
                      "(source --relation--> target; for `parent`, target is the parent)"}
    (out_dir / "snapshot.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return info


def load_snapshot(directory: Path) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, str]]]:
    """(nodes by id, edges as (source, relation, target)) of a snapshot written by `write_snapshot`."""
    directory = Path(directory)
    suffix = ".jsonl.gz" if (directory / "nodes.jsonl.gz").exists() else ".jsonl"
    nodes = {n["id"]: n for n in read_jsonl(directory / f"nodes{suffix}")}
    edges = [(e["source"], e["relation"], e["target"]) for e in read_jsonl(directory / f"edges{suffix}")]
    return nodes, edges


def ancestors(node: str, parents: dict[str, Sequence[str]], *, limit: int = 64) -> set[str]:
    """Every ancestor of `node` over a parent map (cycle-safe)."""
    seen: set[str] = set()
    frontier = list(parents.get(node, ()))
    while frontier and len(seen) < 10 ** 7:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(parents.get(current, ()))
    return seen


# -- CLI -----------------------------------------------------------------------------------------------------------------

def run_fetch(args: argparse.Namespace) -> dict[str, Any]:
    chosen = [s for s in SOURCES if not args.source or s.name in args.source]
    records = [fetch_source(s, args.root) for s in chosen]
    return {"fetched": records, "unavailable": UNAVAILABLE}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download (or verify) the pinned sources")
    fetch.add_argument("--source", nargs="*", default=None, help="source names (default: all)")
    fetch.add_argument("--root", type=Path, default=DATA_ROOT)
    fetch.set_defaults(run=run_fetch)
    from . import learn_icd10cm, learn_mesh, learn_public
    learn_mesh.add_cli(sub)
    learn_icd10cm.add_cli(sub)
    learn_public.add_cli(sub)
    args = parser.parse_args(argv)
    result = args.run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
