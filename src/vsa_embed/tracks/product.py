"""T3 product catalogues: Google Product Taxonomy (+ Shopify attributes); corpus = Amazon ESCI products.

Domain documents (`prepare`): every US-locale product of the ESCI product catalogue (Apache-2.0) as
"title / brand / colour / bullet points / description" (HTML tags stripped), in a seeded order
(hash of the product id); 2% of products (by hash) go to evaluation. Training documents stop at
`max_train_chars` (enough for the 100M-token budget), so no general text is mixed in.

Synthetic concepts (E5.4 "new SKUs and product types"): invented names placed under an existing
internal category, in three kinds — `type_invented` (two invented stems, no surface cue),
`type_headed` (invented stem + a head noun of the parent's name: a surface cue) and `sku` (invented
brand + model code + head noun). Frames: parent, top and second-level category, depth, name tokens,
the parent's path tokens and attributes. Names are checked against WordNet, the text samples and
every alias.

Items:
- `category_probe` (product → taxonomy category, linear probe): titles of evaluation products whose
  title links exactly one category through a multi-character alias; labels are that category's
  level-1 and level-2 ancestors (distant supervision through the linker, so the label is the link's
  category, not a curated product label — ESCI has none);
- `relevance_probe` (query–product relevance, linear probe): ESCI US "small version" judgements
  (E/S/C/I), stratified samples of ESCI's own train and test splits, flagged by whether the product
  title links a held-out category;
- `zeroshot_property`, `zeroshot_entailment`: frame facts of synthetic and held-out categories.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..benchmarks.devtools import NameMaker
from ..ontologies.google_product import build_google_product_ontology, singular, tokens
from ..ontologies.wordnet import FrameOntology
from ..span_channel import CausalLinker
from . import Track
from .common import RelationTemplates, SyntheticConcept, choice_items, entailment_items, mention_item

TEMPLATES = {
    "parent": RelationTemplates(["{x} is a kind of", "In the product catalogue, {x} is listed under",
                                 "Shoppers find {x} in the category"], "{x} is a kind of {y}."),
    "top_category": RelationTemplates(["{x} belongs to the department", "{x} is sold in the department",
                                       "The top-level category of {x} is"], "{x} belongs to the department {y}."),
    "second_category": RelationTemplates(["{x} is part of the product group", "Within its department, {x} falls under",
                                          "The product group of {x} is"], "{x} is part of the product group {y}."),
    "has_attribute": RelationTemplates(["When listing {x}, sellers must specify its", "An important attribute of {x} is its",
                                        "Customers filter {x} by"], "Listings of {x} specify its {y}."),
}
_TAGS = re.compile(r"<[^>]+>")
_TOKENS = re.compile(r"\w+|[^\w\s]")


def _bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(html.unescape(_TAGS.sub(" ", text)).split())


def product_text(row: dict[str, Any]) -> str:
    lines = [_clean(row["product_title"])]
    if row.get("product_brand"):
        lines.append(f"Brand: {_clean(row['product_brand'])}")
    if row.get("product_color"):
        lines.append(f"Color: {_clean(row['product_color'])}")
    bullets = [_clean(b) for b in (row.get("product_bullet_point") or "").split("\n")]
    lines += [f"- {b}" for b in bullets if b]
    description = _clean(row.get("product_description"))
    if description:
        lines.append(description)
    return "\n".join(line for line in lines if line)


class ProductTrack(Track):
    name = "t3"

    def _raw(self, key: str) -> Path:
        return Path(self.config["product"][key]).expanduser()

    def ontology(self) -> FrameOntology:
        if not hasattr(self, "_ontology_cache"):
            self._ontology_cache = build_google_product_ontology(
                self._raw("taxonomy"), shopify_categories=self._raw("shopify_categories"),
                shopify_mapping=self._raw("shopify_mapping"), max_atomics=int(self.config["ontology"]["max_atomics"]),
                max_degree=int(self.config["ontology"]["max_degree"]),
                max_alias_senses=int(self.config["product"]["max_alias_senses"]))
        c = self._ontology_cache
        return FrameOntology(c.name, list(c.concept_names), list(c.relation_names), list(c.atomic_names),
                             [list(f) for f in c.frames], list(c.alias_pairs), dict(c.metadata))

    def _us_products(self) -> pa.Table:
        table = pq.read_table(self._raw("products"), filters=[("product_locale", "=", "us")])
        keys = np.array([int(hashlib.sha256(p.encode()).hexdigest()[:12], 16) for p in table.column("product_id").to_pylist()])
        return table.take(pa.array(np.argsort(keys, kind="stable")))

    def prepare(self) -> dict[str, Any]:
        summary_path = self.docs_dir / "documents_summary.json"
        if summary_path.exists() and (self.docs_dir / "train.jsonl.gz").exists():
            return json.loads(summary_path.read_text())
        import gzip
        table = self._us_products()
        eval_percent, budget = int(self.config["product"]["eval_percent"]), int(self.config["product"]["max_train_chars"])
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        stats = {"eval": {"documents": 0, "chars": 0}, "train": {"documents": 0, "chars": 0}}
        with gzip.open(self.docs_dir / "eval.jsonl.gz.part", "wt", encoding="utf-8") as eval_out, \
                gzip.open(self.docs_dir / "train.jsonl.gz.part", "wt", encoding="utf-8") as train_out:
            for batch in table.to_batches(max_chunksize=4096):
                for row in batch.to_pylist():
                    text = product_text(row)
                    if not text:
                        continue
                    split = "eval" if _bucket(row["product_id"]) < eval_percent else "train"
                    if split == "train" and stats["train"]["chars"] >= budget:
                        continue
                    (eval_out if split == "eval" else train_out).write(
                        json.dumps({"text": text, "product_id": row["product_id"]}, ensure_ascii=False) + "\n")
                    stats[split]["documents"] += 1; stats[split]["chars"] += len(text)
        for split in ("eval", "train"):
            Path(self.docs_dir / f"{split}.jsonl.gz.part").rename(self.docs_dir / f"{split}.jsonl.gz")
        summary = {**stats, "us_products": table.num_rows, "eval_percent": eval_percent, "max_train_chars": budget}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    # -- synthetic product types and SKUs --------------------------------------------------------
    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        settings = self.config["synthetic"]
        rng = random.Random(int(self.config["seed"]) + 11)
        names = NameMaker(rng, {w.lower() for w in forbidden})
        paths = ontology.metadata["paths"]
        attributes = ontology.metadata["attributes"]
        atoms = set(ontology.atomic_names)
        by_path = {" > ".join(p): cid for cid, p in paths.items()}
        internal = sorted({by_path[" > ".join(p[:-1])] for p in paths.values() if len(p) >= 3})
        children: dict[str, list[str]] = {}
        for cid, p in paths.items():
            if len(p) >= 2:
                children.setdefault(by_path[" > ".join(p[:-1])], []).append(cid)
        existing = {a.lower() for a, _ in ontology.alias_pairs}
        out: list[SyntheticConcept] = []
        kinds = ["type_invented", "type_headed", "sku"]
        for k in range(int(settings["count"])):
            kind = kinds[k % 3]
            parent = rng.choice(internal)
            p = paths[parent]
            sibling = paths[rng.choice(children[parent])][-1]
            head = singular(sibling.split()[-1]).capitalize() if sibling.split() else "Item"
            if kind == "type_invented":
                name = f"{names.stem().capitalize()} {names.stem().capitalize()}"
            elif kind == "type_headed":
                name = f"{names.stem().capitalize()} {head}"
            else:
                code = f"{rng.choice('ABCDEFGHJKLMNPRSTVXZ')}{rng.choice('ABCDEFGHJKLMNPRSTVXZ')}-{rng.randint(100, 990)}"
                name = f"{names.stem().capitalize()} {code} {head}"
            if name.lower() in existing:
                continue
            frame = [("parent", f"category:{parent.split(':')[1]}"), ("top_category", f"category:{by_path[p[0]].split(':')[1]}")]
            facts = {"parent": [p[-1]], "top_category": [p[0]]}
            if len(p) >= 2:
                frame.append(("second_category", f"category:{by_path[' > '.join(p[:2])].split(':')[1]}"))
                facts["second_category"] = [p[1]]
            frame.append(("depth", f"depth:{min(len(p) + 1, max(len(q) for q in paths.values()))}"))
            if kind != "type_invented":
                frame += [("name_token", f"token:{t}") for t in tokens(head)]
            attrs = attributes.get(parent, [])[:6]
            frame += [("has_attribute", f"attribute:{a}") for a in attrs]
            if attrs:
                facts["has_attribute"] = [a.lower() for a in attrs]
            for ancestor in reversed(p):
                frame += [("path_token", f"token:{t}") for t in tokens(ancestor)]
            frame = [edge for i, edge in enumerate(frame) if edge[1] in atoms and edge not in frame[:i]]
            out.append(SyntheticConcept(name, [name], frame[:int(self.config["ontology"]["max_degree"])], facts,
                                        {"kind": kind, "parent": parent, "surface_cue": kind != "type_invented"}))
        return out

    # -- items -------------------------------------------------------------------------------------
    def _facts(self, ontology: FrameOntology, cid: str) -> dict[str, list[str]]:
        p = ontology.metadata["paths"][cid]
        facts = {"top_category": [p[0]]}
        if len(p) >= 2:
            facts["parent"] = [p[-2]]
        if len(p) >= 3:
            facts["second_category"] = [p[1]]
        attrs = ontology.metadata["attributes"].get(cid, [])
        if attrs:
            facts["has_attribute"] = [a.lower() for a in attrs[:6]]
        return facts

    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        ontology: FrameOntology = context["ontology"]
        table = context["table"]
        seed = int(self.config["seed"])
        rng = random.Random(seed + 5)
        settings = self.config["items"]
        held = {ontology.concept_names[c] for c in context["holdout_concepts"]}
        paths = ontology.metadata["paths"]
        linker = CausalLinker(table, min_subtokens=1)

        def linked(text: str) -> list[tuple[str, int, int]]:
            offsets = [(m.start(), m.end()) for m in _TOKENS.finditer(text)]
            out = []
            for span in linker.link(text, offsets):
                start, end = offsets[span.start_token][0], offsets[span.end_token][1]
                alias = text[start:end]
                if len(alias) >= 6 or " " in alias:
                    for concept in table.entry_concepts[span.entry]:
                        name = ontology.concept_names[concept]
                        if name.startswith("gpt:"):
                            out.append((name, start, end))
            return out

        # 1. category probe on evaluation product titles.
        by_split: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
        for doc in self.documents("eval"):
            title = doc.split("\n", 1)[0][:300]
            links = linked(title)
            if len({c for c, _, _ in links}) != 1:
                continue
            concept, start, end = links[-1]
            p = paths[concept]
            split = "heldout" if concept in held else "train"
            by_split[split].append({"track": "t3", "task": "category_probe", "split": split, "concept": concept,
                                    "text": title, "span": [start, end], "surface": title[start:end],
                                    "labels": {"top_category": p[0], "second_category": p[1] if len(p) > 1 else p[0]}})
        probe = []
        for split, cap in (("train", int(settings["category_probe_train"])), ("heldout", int(settings["category_probe_heldout"]))):
            rows = by_split[split]
            probe += rows if len(rows) <= cap else rng.sample(rows, cap)
        for s in context["synthetic"]:
            text = f"{s.name}, brand new, ships in two days"
            item = mention_item(text, s.name, track="t3", task="category_probe", split="synthetic",
                                concept=f"synthetic:{s.name}",
                                labels={"top_category": s.facts["top_category"][0],
                                        "second_category": s.facts.get("second_category", s.facts["top_category"])[0]},
                                kind=s.meta["kind"])
            if item:
                probe.append(item)
        for i, item in enumerate(probe):
            item["id"] = f"t3-category_probe-{i:06d}"

        # 2. query–product relevance (ESCI small version, US).
        examples = pq.read_table(self._raw("examples"), filters=[("product_locale", "=", "us"), ("small_version", "=", 1)]).to_pandas()
        picked = []
        for split, cap in (("train", int(settings["relevance_train"])), ("test", int(settings["relevance_test"]))):
            part = examples[examples["split"] == split]
            picked.append(part.sample(n=min(cap, len(part)), random_state=seed))
        import pandas as pd
        chosen = pd.concat(picked)
        products = pq.read_table(self._raw("products"), columns=["product_id", "product_title"],
                                 filters=[("product_locale", "=", "us")])
        mask = pc.is_in(products.column("product_id"), value_set=pa.array(chosen["product_id"].unique()))
        titles = dict(zip(*[products.filter(mask).column(c).to_pylist() for c in ("product_id", "product_title")]))
        relevance = []
        for row in chosen.itertuples():
            title = _clean(titles.get(row.product_id))[:300]
            if not title:
                continue
            concepts = {c for c, _, _ in linked(title)}
            split = "heldout" if concepts & held else "train" if concepts else "unlinked"
            relevance.append({"id": f"t3-relevance_probe-{len(relevance):06d}", "track": "t3", "task": "relevance_probe",
                              "split": split, "esci_split": row.split, "query_id": int(row.query_id),
                              "query": row.query.strip(), "product_id": row.product_id, "product_title": title,
                              "text": f"Query: {row.query.strip()}\nProduct: {title}", "label": row.esci_label,
                              "linked_concepts": sorted(concepts)})

        # 3. zero-shot frame facts (synthetic and held-out categories).
        records = [{"concept": f"synthetic:{s.name}", "surface": s.name, "split": "synthetic", "facts": s.facts}
                   for s in context["synthetic"]]
        held_records = []
        for cid in sorted(held):
            p = paths[cid]
            held_records.append({"concept": cid, "surface": p[-1], "split": "heldout", "facts": self._facts(ontology, cid)})
        pools: dict[str, set[str]] = {}
        for cid in paths:
            for rel, fillers in self._facts(ontology, cid).items():
                pools.setdefault(rel, set()).update(fillers)
        pools_sorted = {k: sorted(v) for k, v in pools.items()}
        k = int(settings["relations_per_concept"])
        zero_property = choice_items(records + held_records, TEMPLATES, pools_sorted, track="t3", task="zeroshot_property",
                                     seed=seed + 1, max_relations=k)
        zero_entail = entailment_items(records + held_records, TEMPLATES, pools_sorted, track="t3",
                                       task="zeroshot_entailment", seed=seed + 2, max_relations=k)
        return {"category_probe": probe, "relevance_probe": relevance, "zeroshot_property": zero_property,
                "zeroshot_entailment": zero_entail}
