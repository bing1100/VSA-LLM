"""Adapters for the public *learn* benchmarks into the common formats of `learn_data`.

**MedConceptsQA** (Shoham & Rappoport 2024; HF `ofir408/MedConceptsQA` at a pinned revision; Apache-2.0): every
question becomes a ranking item — context = the question line ("What is the description of the medical code X in
ICD10CM?"), the four options, answer index, the code as the linked term. `meta.in_fy2026` / `in_fy2027` mark codes
still valid in the ICD-10-CM releases of `learn_icd10cm`. The full test split stays local; the dev split and a frozen
sample (500 per level, chosen by the sha256 of the question id) are committed with the Apache-2.0 notice.

**OET** (Dong, Chen, He, Horrocks, CIKM 2023; Zenodo 10.5281/zenodo.10432003, ver4, CC-BY-4.0; SNOMED CT-derived): the
concept-placement data of MM-S14-Disease and MM-S14-CPP. One placement item per out-of-KB (new in SNOMED CT US
2017-03-01) concept of the official `valid-NIL` (→ dev) and `test-NIL` files, gold parents = the parents in the
2014-09-01 release (atomic SCTIDs or complex `[EX.]` expressions), gold relations add the children (positions
`<parent, child>` in `meta.positions`), the MedMentions contexts in `meta.contexts`. OET splits mentions, not concepts:
the **primary test** (`placement-test.jsonl`) keeps the test-NIL concepts absent from valid-NIL (concept-disjoint); the
mention split (`placement-test-mention-split.jsonl`, every test-NIL concept) is secondary. The before snapshot is the
2014 entity and edge catalogue. Everything SNOMED-derived stays under `~/data/vsa-llm/toolkit-learn/` (mode 700).

**TaxoExpan / TMN** (Shen et al. WWW 2020; Zhang et al. AAAI 2021): the WordNet 3.0 noun and verb taxonomies with the
SemEval-2016 Task 14 lemmas, as distributed by TaxoExpan (`wordnet_{noun,verb}.terms/.taxo`).
- `taxoexpan-semeval-{noun,verb}` (**primary**): TaxoExpan's official split, read from its dataset pickle (`train/
  validation/test_node_ids`) with a stub unpickler that executes nothing; the before taxonomy is the training-node
  subgraph.
- `tmn-wordnet-{noun,verb}` (**secondary; not comparable to published TMN numbers**): TMN's datasets are these same files
  (identical node and edge counts) but TMN's split files are gone (Google Drive 404, `learn_data.UNAVAILABLE`), so the
  TMN protocol is re-drawn: 1,000 validation and 1,000
  test nodes sampled from all non-root nodes (seeded), the before taxonomy keeps the other nodes and bridges each
  removed node's parents to its children; gold = nearest kept ancestors (parents) and descendants (children).
- **Leakage check** (`leakage_report`): test nodes with siblings in the training taxonomy (allowed by the protocol,
  reported), test names equal to a training node's name, gold parents that had to be bridged, and test nodes that are
  WordNet synsets (present in the repository's WordNet track).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import learn_data as ld

RAW = ld.LOCAL_ROOT / "raw"
MCQA_SET = "medconceptsqa-icd10cm"
MCQA_LEVELS = ("easy", "medium", "hard")
OET_PARTS = {"disease": "MM-S14-Disease", "cpp": "MM-S14-CPP"}
TAXO_SETS = {"noun": "semeval-noun/wordnet_noun", "verb": "semeval-verb/wordnet_verb"}
TAXO_ROLES = {   # author, 2026-10-08
    "taxoexpan": {"role": "primary", "label": "TaxoExpan official split"},
    "tmn": {"role": "secondary", "label": "TMN protocol, re-drawn split: not comparable to published TMN numbers"},
}
VERSION = "v1"


# -- MedConceptsQA ---------------------------------------------------------------------------------------------------------

QUESTION = re.compile(r"medical code (?P<code>\S+) in (?P<vocab>\w+)\?")


def medconceptsqa_item(row: dict[str, Any], *, level: str, split: str, fy2026: set[str] | None = None,
                       fy2027: set[str] | None = None) -> dict[str, Any]:
    """One MedConceptsQA row → a ranking item (raises ValueError on an inconsistent row)."""
    context = row["question"].split("\n", 1)[0].strip()
    options = [row[f"option{i}"] for i in range(1, 5)]
    answer = "ABCD".index(row["answer_id"])
    if options[answer] != row["answer"]:
        raise ValueError(f"question {row['question_id']}: answer text differs from option {row['answer_id']}")
    match = QUESTION.search(context)
    code = match.group("code") if match else None
    meta = {"question_id": int(row["question_id"]), "level": level, "split": split, "vocab": row["vocab"],
            "answer_letter": row["answer_id"], "mcq_prompt": row["question"]}
    if code and fy2026 is not None:
        meta["in_fy2026"] = code in fy2026
    if code and fy2027 is not None:
        meta["in_fy2027"] = code in fy2027
    return ld.rank_item(set_name=MCQA_SET, item_id=f"{MCQA_SET}:{level}:{split}:{row['question_id']}", context=context,
                        options=options, answer=answer, terms=[{"surface": code, "concept": code}] if code else [], meta=meta)


def in_sample(question_id: int, per_level: int, total: int, salt: str = ld.SPLIT_SALT) -> bool:
    value = int(hashlib.sha256(f"{salt}:mcqa:{question_id}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return value < per_level / max(total, 1)


def build_medconceptsqa(out_dir: Path, local_dir: Path, *, raw: Path = RAW / "medconceptsqa", sample: int = 500) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from . import learn_icd10cm as icd
    fy2026 = set(icd.load_release("fy2026").codes)
    fy2027 = set(icd.load_release("fy2027").codes)
    out_dir, local_dir = Path(out_dir), Path(local_dir)
    files, stats, committed_dev, committed_sample = {}, {}, [], []
    for level in MCQA_LEVELS:
        for split in ("dev", "test"):
            rows = pq.read_table(raw / f"icd10cm_{level}" / f"{split}-00000-of-00001.parquet").to_pylist()
            items, skipped = [], Counter()
            for row in rows:
                try:
                    items.append(medconceptsqa_item(row, level=level, split=split, fy2026=fy2026, fy2027=fy2027))
                except ValueError as error:
                    skipped[str(error).split(": ", 1)[-1].split(" (")[0][:60]] += 1
            files[f"{level}-{split}"] = ld.write_jsonl(local_dir / f"{level}-{split}.jsonl.gz", items)
            stats[f"{level}-{split}"] = {"rows": len(rows), "items": len(items), "skipped": dict(skipped),
                                         "codes_not_in_fy2026": sum(1 for i in items if not i["meta"].get("in_fy2026")),
                                         "codes_not_in_fy2027": sum(1 for i in items if not i["meta"].get("in_fy2027"))}
            if split == "dev":
                committed_dev += items
            else:
                committed_sample += [i for i in items if in_sample(i["meta"]["question_id"], sample, len(items))]
    repo_files = {"dev": ld.write_jsonl(out_dir / "dev.jsonl", committed_dev),
                  "test-sample": ld.write_jsonl(out_dir / "test-sample.jsonl", committed_sample)}
    manifest = {
        "set": MCQA_SET, "version": VERSION, "built": time.strftime("%Y-%m-%d"),
        "source": f"huggingface.co/datasets/ofir408/MedConceptsQA@{ld.MCQA_REVISION} (icd10cm_easy/medium/hard)",
        "licence": "Apache-2.0 (MedConceptsQA, Shoham & Rappoport 2024); modified: converted to the ranking format",
        "committed": repo_files, "local": {"path": str(local_dir), "files": files}, "stats": stats,
        "sample_rule": f"sha256('{ld.SPLIT_SALT}:mcqa:<question_id>') < {sample}/n per level (≈ {sample} per level)",
        "format": "ranking item: context = question line; options = option1..4; answer = index of answer_id; "
                  "meta.mcq_prompt = the original lettered question",
        "fy2027_overlap": "MedConceptsQA predates FY2027: none of the H3 new codes can occur (see codes_not_in_fy2027 for "
                          "codes deleted since)",
        "command": "PYTHONPATH=src python -m vsa_embed.benchmarks.learn_data medconceptsqa",
    }
    ld.write_manifest(out_dir, manifest)
    return manifest


# -- OET (SNOMED CT time split) ------------------------------------------------------------------------------------------------

def _zip_jsonl(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    with archive.open(name) as handle:
        return [json.loads(line) for line in handle if line.strip()]


OWL_LABEL = re.compile(r'AnnotationAssertion\(rdfs:label (?:<http://snomed\.info/id/|:)(\d+)>? "((?:[^"\\]|\\.)*)"')


def owl_labels(archive: zipfile.ZipFile, name: str, wanted: set[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    with archive.open(name) as handle:
        for raw in handle:
            if b"rdfs:label" not in raw:
                continue
            match = OWL_LABEL.search(raw.decode("utf-8", "replace"))
            if match and match.group(1) in wanted:
                labels[match.group(1)] = match.group(2).replace('\\"', '"')
    return labels


def split_ids(value: str) -> list[str]:
    return [v for v in (value or "").split("|") if v and v != "SCTID_NULL"]


def oet_items(rows: Sequence[dict[str, Any]], *, set_name: str, split: str, labels: dict[str, str]) -> list[dict[str, Any]]:
    """Group mention rows by their new concept (`label_concept_ori`)."""
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_concept[row["label_concept_ori"]].append(row)
    items = []
    for concept, group in sorted(by_concept.items()):
        parents = list(dict.fromkeys(p for r in group for p in split_ids(r["parents_concept"])))
        children = list(dict.fromkeys(c for r in group for c in split_ids(r["children_concept"])))
        positions = []
        for r in group:
            for pair in (r.get("parents-children_concept") or "").split("|"):
                if not pair:
                    continue
                parent, _, child = pair.rpartition("-")
                positions.append((parent, None if child == "SCTID_NULL" else child))
        synonyms = list(dict.fromkeys(s for r in group for s in (r.get("synonyms") or "").split("|") if s))
        mentions = Counter(r["mention"] for r in group)
        name = labels.get(concept) or (synonyms[0] if synonyms else mentions.most_common(1)[0][0])
        description = next((r["label"] for r in group if r.get("label")), None)
        contexts = [{"left": r["context_left"], "mention": r["mention"], "right": r["context_right"]} for r in group]
        meta = {"split": split, "oet_split": f"{split}-NIL" if split == "test" else "valid-NIL", "umls_cui": group[0]["label_concept_UMLS"],
                "positions": [list(p) for p in dict.fromkeys(positions)], "complex_parents": [p for p in parents if not p.isdigit()],
                "parent_titles": list(dict.fromkeys(t for r in group for t in (r.get("parents") or "").split("|") if t)),
                "mention_forms": dict(mentions), "contexts": contexts, "name_source": "2017 OWL rdfs:label" if concept in labels
                else ("synonym" if synonyms else "most frequent mention")}
        items.append(ld.placement_item(set_name=set_name, record=concept, name=name, aliases=[*synonyms, *mentions],
                                       definition=description, gold_parents=parents,
                                       gold_relations=[*(["parent", p] for p in parents), *(["child", c] for c in children)],
                                       candidates=None, evidence={"medmentions_mentions": len(group)}, meta=meta))
    return items


def oet_snapshot(entities: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]], list[dict[str, Any]]]:
    nodes = {e["idx"]: ld.snapshot_node(e["idx"], e["title"], aliases=(e.get("synonyms") or "").split("|"),
                                        definition=e.get("text") or None, kind="concept") for e in entities}
    triples, candidate_edges = [], []
    for edge in edges:
        parent, child = edge["parent_idx"], edge["child_idx"]
        if parent not in nodes:
            nodes[parent] = ld.snapshot_node(parent, edge.get("parent") or parent, kind="complex")
        candidate_edges.append({"parent": parent, "child": None if child == "SCTID_NULL" else child, "degree": edge.get("degree")})
        if child != "SCTID_NULL":
            if child not in nodes:
                nodes[child] = ld.snapshot_node(child, edge.get("child") or child, kind="complex")
            triples.append((child, "parent", parent))
    return list(nodes.values()), triples, candidate_edges


def build_oet(out_dir: Path, local_dir: Path, *, zip_path: Path = RAW / "oet" / "OET-data-ver4.zip") -> dict[str, Any]:
    out_dir = Path(out_dir)
    local_dir = ld.private_dir(Path(local_dir))
    archive = zipfile.ZipFile(zip_path)
    summary = {}
    for short, part in OET_PARTS.items():
        kind = part.split("-")[-1]
        set_name = f"oet-snomed-{short}"
        rows = {split: _zip_jsonl(archive, f"{part}/mention-level-(concept-placement)/{name}.jsonl")
                for split, name in (("dev", "valid-NIL"), ("test", "test-NIL"))}
        wanted = {r["label_concept_ori"] for split_rows in rows.values() for r in split_rows}
        labels = owl_labels(archive, f"{part}/ontology/SNOMEDCT-US-20170301-{kind}-final.owl", wanted)
        target = ld.private_dir(local_dir / short)
        entities = _zip_jsonl(archive, f"{part}/ontology/SNOMEDCT-US-20140901-{kind}_syn_attr_hyp-all.jsonl")
        edges = _zip_jsonl(archive, f"{part}/ontology/SNOMEDCT-US-20140901-{kind}-edges-all.jsonl")
        nodes, triples, candidate_edges = oet_snapshot(entities, edges)
        snapshot_ids = {n["id"] for n in nodes}
        items_by_split = {split: oet_items(split_rows, set_name=set_name, split=split, labels=labels)
                          for split, split_rows in rows.items()}
        dev_concepts = {i["record"] for i in items_by_split["dev"]}
        for split, items in items_by_split.items():
            for item in items:
                item["meta"]["gold_parents_not_in_snapshot"] = [p for p in item["gold_parents"] if p not in snapshot_ids]
                if split == "test":
                    item["meta"]["also_in_dev"] = item["record"] in dev_concepts
        disjoint = [i for i in items_by_split["test"] if not i["meta"]["also_in_dev"]]
        files = {"placement-dev": ld.write_jsonl(target / "placement-dev.jsonl", items_by_split["dev"]),
                 "placement-test": ld.write_jsonl(target / "placement-test.jsonl", disjoint),
                 "placement-test-mention-split": ld.write_jsonl(target / "placement-test-mention-split.jsonl",
                                                                items_by_split["test"])}
        snapshot = ld.write_snapshot(target / "snapshot-2014", nodes, triples, description={
            "ontology": f"SNOMED CT US {part} subset", "release": "20140901 (OET entity + edge catalogue, atomic + complex)",
            "source": f"{zip_path.name}:{part}/ontology", "licence": "SNOMED CT (licensed) / OET CC-BY-4.0: local only"})
        candidates = ld.write_jsonl(target / "snapshot-2014" / "candidate_edges.jsonl.gz", candidate_edges)
        overlap = sum(1 for i in items_by_split["test"] if i["meta"]["also_in_dev"])
        summary[set_name] = {
            "files": files, "local": str(target),
            "concepts": {"dev": len(items_by_split["dev"]), "test (primary, concept-disjoint)": len(disjoint),
                         "test-mention-split (secondary)": len(items_by_split["test"])},
            "mentions": {s: len(v) for s, v in rows.items()},
            "concepts_in_both_dev_and_test": overlap,
            "note_split": "OET splits mentions, not concepts. Primary test = placement-test.jsonl, the test-NIL concepts that "
                          "do not occur in valid-NIL (author, 2026-10-08); secondary = placement-test-mention-split.jsonl, "
                          "every test-NIL concept (meta.also_in_dev marks the overlap)",
            "items_with_complex_parent": {s: sum(1 for i in v if i["meta"]["complex_parents"]) for s, v in items_by_split.items()},
            "gold_parents_missing_from_snapshot": sum(1 for v in items_by_split.values() for i in v
                                                      for p in i["gold_parents"] if p not in snapshot_ids),
            "named_from_owl": sum(1 for v in items_by_split.values() for i in v if i["meta"]["name_source"].startswith("2017")),
            "snapshot": {"nodes": snapshot["nodes"], "edges": snapshot["edges"], "node_kinds": snapshot["node_kinds"],
                         "candidate_edges": candidates},
        }
    manifest = {
        "set": "oet-snomed", "version": VERSION, "built": time.strftime("%Y-%m-%d"),
        "source": "Zenodo 10.5281/zenodo.10432003 (OET ver4), CC-BY-4.0; Dong, Chen, He, Horrocks, CIKM 2023",
        "licence": "SNOMED CT-derived (licence held; decision 58): every item and snapshot stays under ~/data/vsa-llm "
                   "(mode 700); this manifest carries counts and digests only",
        "releases": {"before": "SNOMED CT US Edition 20140901", "after": "SNOMED CT US Edition 20170301",
                     "shipped_in_zip": "both as OWL (`*-final.owl`) plus the 2014 entity / edge catalogues",
                     "held_locally": "SNOMED CT International 2022-05-31 (RF2 Full + Snapshot) — not needed by the adapter",
                     "missing_for_rebuilding_from_scratch": ["SNOMED CT US Edition 20140901 RF2", "SNOMED CT US Edition 20170301 RF2",
                                                            "UMLS 2017AA (MedMentions' UMLS)", "MedMentions st21pv"]},
        "protocol": "valid-NIL → dev; primary test = test-NIL concepts absent from valid-NIL (concept-disjoint); secondary "
                    "test = OET's mention split (all test-NIL concepts). ver4 has out-of-KB mentions only in evaluation. OET "
                    "ranks edges <parent, child> from candidate_edges (child null = leaf position); gold edges in meta.positions",
        "sets": summary,
        "command": "PYTHONPATH=src python -m vsa_embed.benchmarks.learn_data oet",
    }
    ld.write_manifest(out_dir, manifest)
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# -- TaxoExpan / TMN -------------------------------------------------------------------------------------------------------------

class _Stub:
    """Stand-in for every class a pickle names: records arguments, executes nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __call__(self, *args: Any, **kwargs: Any) -> "_Stub":
        return _Stub(*args)

    def __setstate__(self, state: Any) -> None:
        self.state = state


class StubUnpickler(pickle.Unpickler):
    """Unpickler whose globals are inert stubs (only `set` and `frozenset` are real), so a third-party pickle can be
    read without its libraries (DGL) and without running any of its code."""

    SAFE = {("builtins", "set"): set, ("builtins", "frozenset"): frozenset}

    def find_class(self, module: str, name: str) -> Any:
        return self.SAFE.get((module, name)) or type(f"{module}.{name}", (_Stub,), {})


def taxoexpan_split(pickle_path: Path) -> dict[str, list[str]]:
    """TaxoExpan's official node split (term ids) from its dataset pickle."""
    with open(pickle_path, "rb") as handle:
        data = StubUnpickler(handle).load()
    vocab = [entry.rsplit("@@@", 1)[0].split("||", 1)[1] for entry in data["vocab"]]
    return {name: [vocab[i] for i in data[f"{name}_node_ids"]] for name in ("train", "validation", "test")}


def read_taxonomy(stem: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """(term id → surface name, [(parent, child)]) of a TaxoExpan-format taxonomy."""
    terms = {}
    for line in Path(f"{stem}.terms").read_text(encoding="utf-8").splitlines():
        if line.strip():
            term_id, surface = line.split("\t", 1)
            terms[term_id] = surface.split("||", 1)[0]
    edges = [tuple(line.split("\t")[:2]) for line in Path(f"{stem}.taxo").read_text(encoding="utf-8").splitlines() if line.strip()]
    return terms, edges


def tmn_split(terms: dict[str, str], edges: Sequence[tuple[str, str]], *, seed: int, size: int = 1000) -> dict[str, list[str]]:
    """TMN's protocol (re-drawn): `size` validation and `size` test nodes sampled from the non-root nodes."""
    has_parent = {c for _, c in edges}
    pool = sorted(t for t in terms if t in has_parent)
    rng = random.Random(seed)
    chosen = rng.sample(pool, 2 * size)
    validation, test = sorted(chosen[:size]), sorted(chosen[size:])
    removed = set(chosen)
    return {"train": sorted(t for t in terms if t not in removed), "validation": validation, "test": test}


def kept_relatives(node: str, step: dict[str, list[str]], kept: set[str]) -> list[str]:
    """Nearest nodes in `kept` along `step` (parents or children), walking through removed nodes."""
    out, frontier, seen = [], list(step.get(node, [])), set()
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if current in kept:
            out.append(current)
        else:
            frontier.extend(step.get(current, []))
    return list(dict.fromkeys(out))


def before_taxonomy(edges: Sequence[tuple[str, str]], kept: set[str]) -> list[tuple[str, str]]:
    """Edges among kept nodes plus bridges over removed nodes (each removed node's kept ancestors → kept descendants)."""
    parents, children = defaultdict(list), defaultdict(list)
    for p, c in edges:
        parents[c].append(p); children[p].append(c)
    out = {(p, c) for p, c in edges if p in kept and c in kept}
    for node in {n for e in edges for n in e} - kept:
        for p in kept_relatives(node, parents, kept):
            for c in kept_relatives(node, children, kept):
                out.add((p, c))
    return sorted(out)


def wordnet_definitions(ids: Iterable[str]) -> dict[str, str]:
    """Synset glosses from the installed NLTK WordNet (empty if unavailable)."""
    try:
        from nltk.corpus import wordnet
        out = {}
        for term in ids:
            if re.fullmatch(r".+\.[nvasr]\.\d+", term):
                try:
                    out[term] = wordnet.synset(term).definition()
                except Exception:   # noqa: BLE001 - ids that are not synsets of the installed WordNet
                    pass
        return out
    except LookupError:
        return {}


def taxo_items(set_name: str, terms: dict[str, str], edges: Sequence[tuple[str, str]], split: dict[str, list[str]], *,
               completion: bool, definitions: dict[str, str]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    kept = set(split["train"])
    parents, children = defaultdict(list), defaultdict(list)
    for p, c in edges:
        parents[c].append(p); children[p].append(c)
    items = []
    for name, out_split in (("validation", "dev"), ("test", "test")):
        for node in split[name]:
            gold = kept_relatives(node, parents, kept)
            kids = kept_relatives(node, children, kept) if completion else []
            positions = [[p, c] for p in gold for c in (kids or [None])]
            direct = parents.get(node, [])
            items.append(ld.placement_item(
                set_name=set_name, record=node, name=terms[node].replace("_", " "), aliases=[], definition=definitions.get(node),
                gold_parents=gold, gold_relations=[*(["parent", p] for p in gold), *(["child", c] for c in kids)],
                candidates=None, evidence={},
                meta={"split": out_split, "official_split": name, "surface": terms[node], "positions": positions,
                      "direct_parents": direct, "parents_bridged": [p for p in direct if p not in kept],
                      "is_wordnet_synset": bool(re.fullmatch(r".+\.[nv]\.\d+", node)), "leaf": not children.get(node)}))
    return items, before_taxonomy(edges, kept)


def leakage_report(items: Sequence[dict[str, Any]], terms: dict[str, str], edges: Sequence[tuple[str, str]], kept: set[str]) -> dict[str, Any]:
    children = defaultdict(set)
    for p, c in edges:
        children[p].add(c)
    kept_names = Counter(terms[t].replace("_", " ").casefold() for t in kept)
    out = {}
    for split in ("dev", "test"):
        rows = [i for i in items if i["meta"]["split"] == split]
        siblings = [len({s for p in i["meta"]["direct_parents"] for s in children[p]} - {i["record"]} & kept) for i in rows]
        out[split] = {
            "items": len(rows),
            "with_training_sibling": sum(1 for s in siblings if s > 0),
            "share_with_training_sibling": round(sum(1 for s in siblings if s > 0) / max(len(rows), 1), 4),
            "mean_training_siblings": round(sum(siblings) / max(len(rows), 1), 2),
            "name_equals_training_node": sum(1 for i in rows if kept_names[i["name"].casefold()] > 0),
            "gold_parent_bridged": sum(1 for i in rows if i["meta"]["parents_bridged"]),
            "no_gold_parent": sum(1 for i in rows if not i["gold_parents"]),
            "wordnet_synsets": sum(1 for i in rows if i["meta"]["is_wordnet_synset"]),
            "leaves": sum(1 for i in rows if i["meta"]["leaf"]),
        }
    return out


def build_taxo(out_dir: Path, local_dir: Path, *, raw: Path = RAW / "taxoexpan", seed: int = 20210202,
               tmn_size: int = 1000) -> dict[str, Any]:
    out_dir, local_dir = Path(out_dir), Path(local_dir)
    sets = {}
    for pos, stem in TAXO_SETS.items():
        terms, edges = read_taxonomy(raw / stem)
        definitions = wordnet_definitions(terms)
        for protocol in ("taxoexpan", "tmn"):
            if protocol == "taxoexpan":
                set_name = f"taxoexpan-semeval-{pos}"
                split = taxoexpan_split(Path(f"{raw / stem}.fasttext_mode4.pickle.bin"))
                completion = False
            else:
                set_name = f"tmn-wordnet-{pos}"
                split = tmn_split(terms, edges, seed=seed, size=tmn_size)
                completion = True
            items, before = taxo_items(set_name, terms, edges, split, completion=completion, definitions=definitions)
            role = TAXO_ROLES[protocol]
            for item in items:
                item["meta"].update(role)
            kept = set(split["train"])
            target = local_dir / set_name
            files = {f"placement-{s}": ld.write_jsonl(target / f"placement-{s}.jsonl", [i for i in items if i["meta"]["split"] == s])
                     for s in ("dev", "test")}
            nodes = [ld.snapshot_node(t, terms[t].replace("_", " "), definition=definitions.get(t), kind="synset"
                                      if re.fullmatch(r".+\.[nv]\.\d+", t) else "lemma", meta={"surface": terms[t]})
                     for t in sorted(kept)]
            snapshot = ld.write_snapshot(target / "snapshot-train", nodes, [(c, "parent", p) for p, c in before], description={
                "ontology": f"WordNet 3.0 {pos} taxonomy + SemEval-2016 Task 14 lemmas (TaxoExpan files)",
                "release": f"training taxonomy of the {protocol} split", "source": str(raw / stem)})
            (target / "split.json").write_text(json.dumps(split) + "\n")
            sets[set_name] = {
                **role, "protocol": protocol, "official_split": protocol == "taxoexpan",
                "split_source": "TaxoExpan dataset pickle (train/validation/test_node_ids)" if protocol == "taxoexpan"
                else f"re-drawn with seed {seed} (TMN's own split files unavailable: Google Drive 404)",
                "nodes": len(terms), "edges": len(edges), "split_sizes": {k: len(v) for k, v in split.items()},
                "split_sha256": hashlib.sha256((target / "split.json").read_bytes()).hexdigest(),
                "files": files, "local": str(target),
                "snapshot": {"nodes": snapshot["nodes"], "edges": snapshot["edges"]},
                "leakage": leakage_report(items, terms, edges, kept),
                "with_definition": sum(1 for i in items if i["definition"]),
            }
    manifest = {
        "set": "taxonomy-expansion", "version": VERSION, "built": time.strftime("%Y-%m-%d"),
        "sources": "TaxoExpan Google Drive folder (README.txt, wordnet_{noun,verb}.{terms,taxo}, dataset pickles); "
                   "TMN uses the same files (identical node/edge counts) — its split files are unavailable",
        "licence": "TaxoExpan code Apache-2.0; data: WordNet 3.0 (Princeton WordNet licence) + SemEval-2016 Task 14 lemmas "
                   "(Wiktionary-derived, CC BY-SA) — no explicit data licence, so items stay local; counts and digests here",
        "unavailable": {k: v for k, v in ld.UNAVAILABLE.items() if k.startswith("tmn")},
        "skipped": {"MAG-CS / MAG-Full / MAG-Psy": "Microsoft Academic Graph retired (decision 63 instruction: skip)"},
        "leakage_note": "siblings of test nodes in the training taxonomy are allowed by both protocols and reported; every "
                        "WordNet synset of the TMN-protocol sets is also a concept of the repository's WordNet track "
                        "(models trained there have seen its edges)",
        "sets": sets,
        "command": "PYTHONPATH=src python -m vsa_embed.benchmarks.learn_data taxo",
    }
    ld.write_manifest(out_dir, manifest)
    return manifest


def add_cli(sub: Any) -> None:
    parser = sub.add_parser("medconceptsqa", help="MedConceptsQA ICD-10-CM → ranking items")
    parser.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"{MCQA_SET}-{VERSION}")
    parser.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"{MCQA_SET}-{VERSION}")
    parser.add_argument("--sample", type=int, default=500)
    parser.set_defaults(run=lambda a: build_medconceptsqa(a.out, a.local, sample=a.sample))
    oet = sub.add_parser("oet", help="SNOMED CT time-split enrichment (OET) → placement items (local only)")
    oet.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"oet-snomed-{VERSION}")
    oet.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"oet-snomed-{VERSION}")
    oet.set_defaults(run=lambda a: build_oet(a.out, a.local))
    taxo = sub.add_parser("taxo", help="TaxoExpan / TMN WordNet sets → placement items (local only) + leakage report")
    taxo.add_argument("--out", type=Path, default=ld.REPO_ITEMS / f"taxonomy-expansion-{VERSION}")
    taxo.add_argument("--local", type=Path, default=ld.LOCAL_ROOT / f"taxonomy-expansion-{VERSION}")
    taxo.add_argument("--seed", type=int, default=20210202)
    taxo.set_defaults(run=lambda a: build_taxo(a.out, a.local, seed=a.seed))
