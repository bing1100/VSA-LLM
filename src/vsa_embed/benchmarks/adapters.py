"""Benchmark adapters: raw files (`benchmarks.sources`) → ranking items (`rank-items/1`, `benchmarks/README.md`).

*Write* sets with WordNet frames (the E9 WordNet track's synset atomics; `--ontology` defaults to its SmolLM2 build):

- `comps-wugs` — COMPS-WUGS (Misra, Rayz & Ettinger, EACL 2023): "A wug is a dog. Therefore, a wug can bark." as a
  minimal pair over what the nonce word is (`option_terms`: the acceptable vs the unacceptable parent), the same
  property continuation on both sides. Frame of the nonce = `hypernym` → the parent's WordNet synset (COMPS's own sense
  keys, `concept_senses.csv`) if it is an atomic of the track, else its nearest is-a ancestor that is (breadth-first over
  hypernyms and instance hypernyms; depth recorded), plus the parent's `lexname` and `pos` edges (every WordNet frame has
  them). `alt_frames.parent` = the frame of the entry the parent's name links to (the parent's own trained frame).
- `alcuna` — ALCUNA (Yin, Huang & Wan, EMNLP 2023): artificial taxa; the ranking forms only (multiple choice:
  the four options; boolean: Yes / No; "I don't know" items and fill-in-the-blank are left out). Frame of the artificial
  entity = `hypernym` → the WordNet taxon of its parent entity (the full name, else the genus of a binomial, else a
  one-word higher taxon), mapped to the nearest atomic as above, plus `lexname` / `pos`. Entities whose parent does not
  map are left out (coverage in the manifest). The definition is the artificial entity's own properties, verbalized.
- `entity-inferences` — Entity Inferences (Onoe et al., ACL 2023): the definition (its masked span filled) is the
  reading material, the probe sentence ranks its labels. `frame` is null (Wikidata frames come from the T8 track,
  TK-B2); the entity type's QID is kept in `meta`.

*Read* / *meta* sets without frames (TK-B2 adds Wikidata frames): `reversal` (fictitious name–description pairs,
forward and reverse, both directions), `lre` (47 relations, the first prompt template), `bear` (BEAR, every template),
`popqa` (PopQA as 5-way ranking: the gold object against 4 objects of the same property). Their subjects are real
entities the host knows (`insert: false`, linked by the run's own linker), except the fictitious reversal names (new).

Distractors and answer positions are drawn from `random.Random("<seed>|<item id>")`, so a build is reproducible.

    python -m vsa_embed.benchmarks.adapters build --set comps-wugs|alcuna|entity-inferences|reversal|lre|bear|popqa
        [--output DIR] [--ontology ONT] [--seed 0] [--commit-dir experiments/toolkit-bench/items]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from . import sources as src
from .ranking import SCHEMA, Item, Term, write_items

WORDNET_ONTOLOGY = Path("~/data/vsa-llm/c3/wordnet-smollm2-v1/ontology.pt").expanduser()
TAXON_LEXNAMES = ("noun.animal", "noun.plant")
VERSION = "v1"


# -- WordNet frames ---------------------------------------------------------------------------------------------------------

class WordNetFrames:
    """Maps WordNet synsets onto the track's atomics and writes is-a frames for new terms."""

    def __init__(self, ontology: dict[str, Any], wordnet: Any = None, *, table: Any = None) -> None:
        import numpy as np
        if wordnet is None:
            from nltk.corpus import wordnet
        self.wn, self.ontology, self.table = wordnet, ontology, table
        self.atoms = set(ontology["atomic_names"])
        fillers = np.asarray(ontology["fillers"]).tolist()
        names = ontology["atomic_names"]
        self.frequency = Counter(names[f] for f in fillers)

    def nearest_atomic(self, synset: Any) -> tuple[Any | None, int | None]:
        """The synset itself if it is an atomic, else the closest is-a ancestor that is (ties: most frequent filler, then
        name)."""
        if f"synset:{synset.name()}" in self.atoms:
            return synset, 0
        frontier, seen, depth = [synset], {synset}, 0
        while frontier:
            depth += 1
            hits, nxt = [], []
            for s in frontier:
                for h in s.hypernyms() + s.instance_hypernyms():
                    if h in seen:
                        continue
                    seen.add(h); nxt.append(h)
                    if f"synset:{h.name()}" in self.atoms:
                        hits.append(h)
            if hits:
                return sorted(hits, key=lambda h: (-self.frequency[f"synset:{h.name()}"], h.name()))[0], depth
            frontier = nxt
        return None, None

    def isa_frame(self, synset: Any) -> tuple[list[list[str]] | None, dict[str, Any]]:
        """`hypernym` → nearest atomic, plus the synset's `lexname` and `pos` edges when the track has those atomics."""
        atomic, depth = self.nearest_atomic(synset)
        info = {"synset": synset.name(), "atomic": None if atomic is None else f"synset:{atomic.name()}", "depth": depth}
        if atomic is None:
            return None, info
        frame = [["hypernym", f"synset:{atomic.name()}"]]
        for name in (f"lexname:{synset.lexname()}", f"pos:{synset.pos()}"):
            if name in self.atoms:
                frame.append(["lexname" if name.startswith("lexname") else "pos", name])
        return frame, info

    def entry_frame(self, surface: str) -> list[list[str]] | None:
        """The trained frame of the entry `surface` links to (exact alias), or None."""
        if self.table is None:
            return None
        from ..span_channel import normalize_alias
        entry = self.table.alias_to_entry.get(normalize_alias(surface))
        if entry is None:
            return None
        o = self.ontology
        lo, hi = int(o["offsets"][entry]), int(o["offsets"][entry + 1])
        return [[o["relation_names"][int(r)], o["atomic_names"][int(a)]] for r, a in zip(o["relations"][lo:hi].tolist(), o["fillers"][lo:hi].tolist())]


def load_wordnet_frames(ontology_path: Path = WORDNET_ONTOLOGY, *, with_table: bool = True) -> WordNetFrames:
    import torch
    ontology = torch.load(ontology_path, weights_only=False)
    table = None
    if with_table:
        from ..evaluation.channel_probes import resolve_alias_table
        table, _ = resolve_alias_table(ontology, Path(ontology_path))
    return WordNetFrames(ontology, table=table)


def _rng(seed: int, key: str) -> random.Random:
    return random.Random(f"{seed}|{key}")


def _shuffle_options(gold: str, distractors: Sequence[str], rng: random.Random) -> tuple[list[str], int]:
    options = [gold, *distractors]
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(0)


def _a(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


# -- write sets -------------------------------------------------------------------------------------------------------------

def build_comps_wugs(raw: Path, frames: WordNetFrames, *, seed: int = 0, limit: int | None = None) -> tuple[list[Item], dict[str, Any]]:
    rows = [json.loads(line) for line in open(raw / "data/comps/comps_wugs.jsonl") if line.strip()]
    senses = {r["concept"]: r for r in csv.DictReader(open(raw / "data/concept_senses.csv"))}
    mapped: dict[str, tuple[list[list[str]] | None, dict[str, Any]]] = {}
    for concept in sorted({r["acceptable_concept"] for r in rows} | {r["unacceptable_concept"] for r in rows}):
        sense = senses.get(concept)
        if sense is None:
            mapped[concept] = (None, {"synset": None, "atomic": None, "depth": None})
            continue
        synset = frames.wn.lemma_from_key(sense["sensekey"]).synset()
        mapped[concept] = frames.isa_frame(synset)
    items, counts = [], Counter()
    for r in rows[:limit] if limit else rows:
        nonce = r["prefix_acceptable"].split()[1]
        definitions, contexts = [], []
        for prefix in (r["prefix_acceptable"], r["prefix_unacceptable"]):
            head, _, tail = prefix.partition(". ")
            definitions.append(head + ".")
            contexts.append(tail)
        if contexts[0] != contexts[1] or nonce not in contexts[0]:
            counts["skipped_context_mismatch"] += 1
            continue
        option_terms = []
        for concept, definition in zip((r["acceptable_concept"], r["unacceptable_concept"]), definitions):
            frame, _ = mapped[concept]
            alt = frames.entry_frame(concept.replace("_", " "))
            term = {"surface": nonce, "concept": None, "frame": frame, "definition": definition}
            if alt:
                term["alt_frames"] = {"parent": alt}
            option_terms.append([Term.from_json(term)])
        info = {c: mapped[c][1] for c in (r["acceptable_concept"], r["unacceptable_concept"])}
        same = mapped[r["acceptable_concept"]][0] == mapped[r["unacceptable_concept"]][0]
        counts["frames_equal"] += int(same)
        counts["items"] += 1
        items.append(Item(f"comps-wugs-{r['id']}", "comps-wugs", contexts[0], [" " + r["property_phrase"]] * 2, 0, [],
                          option_terms, " ", {"comps_id": r["id"], "base_id": r["base_id"], "property": r["property"],
                                              "acceptable": r["acceptable_concept"], "unacceptable": r["unacceptable_concept"],
                                              "negative_sample_type": r["negative_sample_type"], "similarity": r["similarity"],
                                              "nonce": nonce, "wordnet": info, "frames_equal": same,
                                              "exact_both": all(v["depth"] == 0 for v in info.values())}))
    depths = Counter(str(v[1]["depth"]) for v in mapped.values())
    coverage = {"concepts": len(mapped), "mapped": sum(1 for v in mapped.values() if v[0]),
                "exact_atomic": sum(1 for v in mapped.values() if v[1]["depth"] == 0), "ancestor_depths": dict(sorted(depths.items())),
                "parent_entry_frames": sum(1 for c in mapped if frames.entry_frame(c.replace("_", " "))),
                "pairs": counts["items"], "pairs_with_equal_frames": counts["frames_equal"],
                "pairs_exact_both": sum(1 for i in items if i.meta["exact_both"]),
                "skipped": counts["skipped_context_mismatch"]}
    return items, coverage


_TAXON_PREFIXES = ("", "genus_", "family_", "order_", "class_", "suborder_", "superfamily_", "subfamily_", "phylum_")


def taxon_synset(name: str, wordnet: Any) -> tuple[str | None, Any]:
    """The WordNet taxon for an EOL name: the full name; else the genus of a binomial (`X`, `genus_X`); else a one-word
    higher taxon under its usual WordNet prefixes. Only animal and plant noun senses count (first sense wins)."""
    words = name.replace("-", " ").split()
    if not words:
        return None, None
    tries = [("full", "_".join(words))]
    if len(words) >= 2:
        tries += [("genus", words[0]), ("genus", "genus_" + words[0])]
    else:
        tries += [("group", p + words[0]) for p in _TAXON_PREFIXES[1:]]
    for kind, lemma in tries:
        found = [s for s in wordnet.synsets(lemma.lower(), pos="n") if s.lexname() in TAXON_LEXNAMES]
        if found:
            return kind, found[0]
    return None, None


def alcuna_definition(entity: dict[str, Any], parent: dict[str, Any], *, max_values: int = 8) -> str:
    """The artificial entity's own properties as text (what ALCUNA gives the model as new knowledge)."""
    name = entity["name"]
    rank = entity.get("rank") or "taxon"
    parts = [f"{name} is {_a(rank)} closely related to {parent['name']}."]
    for prop in entity.get("properties") or []:
        values = [str(v) for v in prop.get("values") or []][:max_values]
        if not values:
            continue
        if prop.get("type") == "relation":
            parts.append(f"{name} {prop['name']} {', '.join(values)}.")
        else:
            parts.append(f"The {prop['name']} of {name}: {', '.join(values)}.")
    return " ".join(parts)


_MC = re.compile(r"\n\s*(\d+)\.\s*")


def alcuna_choice(question: str) -> tuple[str, list[str]] | None:
    """Question stem and its options from ALCUNA's multiple-choice text ("…?\\n0. a\\n\\n1. b …")."""
    pieces = _MC.split("\n" + question.strip())
    stem, numbered = pieces[0].strip(), pieces[1:]
    if not stem or len(numbered) < 4 or len(numbered) % 2:
        return None
    options = [numbered[i + 1].strip() for i in range(0, len(numbered), 2)]
    if [int(numbered[i]) for i in range(0, len(numbered), 2)] != list(range(len(options))) or not all(options):
        return None
    return stem, options


def build_alcuna(raw: Path, frames: WordNetFrames, *, seed: int = 0, per_type: int = 1000,
                 forms: Sequence[str] = ("multi-choice", "boolean")) -> tuple[list[Item], dict[str, Any]]:
    meta_rows = [json.loads(line) for line in open(raw / "dataset/meta_data.jsonl") if line.strip()]
    questions = json.load(open(raw / "dataset/id2question.json"))
    entities: dict[str, dict[str, Any]] = {}
    kinds, depths = Counter(), Counter()
    for m in meta_rows:
        entity, parent = m["artificial_entity"], m["parent_entity"]
        kind, synset = taxon_synset(parent["name"], frames.wn)
        kinds[kind or "unmapped"] += 1
        frame, info = frames.isa_frame(synset) if synset is not None else (None, {"synset": None, "atomic": None, "depth": None})
        depths[str(info["depth"])] += 1
        entities[str(entity["id"])] = {"entity": entity, "parent": parent, "frame": frame, "mapping": kind, "wordnet": info,
                                       "definition": alcuna_definition(entity, parent)}
    pools: dict[tuple[str, str], list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    skipped = Counter()
    for eid, qs in questions.items():
        e = entities.get(str(eid))
        if e is None:
            skipped["no_metadata"] += len(qs); continue
        for k, q in enumerate(qs):
            if q["form"] not in forms:
                continue
            if e["frame"] is None:
                skipped["parent_unmapped"] += 1; continue
            if e["entity"]["name"].lower() not in q["question"].lower():
                skipped["name_not_in_question"] += 1; continue
            if q["form"] == "boolean" and q["answers"] not in (["Yes"], ["No"]):
                skipped["boolean_not_yes_no"] += 1; continue
            pools[(q["form"], q["type"])].append((str(eid), k, q))
    items = []
    for (form, qtype), pool in sorted(pools.items()):
        chosen = sorted(pool, key=lambda x: (x[0], x[1]))
        _rng(seed, f"alcuna|{form}|{qtype}").shuffle(chosen)
        for eid, k, q in sorted(chosen[:per_type], key=lambda x: (int(x[0]), x[1])):
            e = entities[eid]
            if form == "multi-choice":
                parsed = alcuna_choice(q["question"])
                if parsed is None:
                    skipped["unparsed_choice"] += 1; continue
                stem, options = parsed
                answer = int(q["answers"][0])
                set_name = "alcuna-mc"
            else:
                stem, options, answer, set_name = q["question"].strip(), ["Yes", "No"], 0 if q["answers"] == ["Yes"] else 1, "alcuna-bool"
            term = Term.from_json({"surface": e["entity"]["name"], "concept": None, "frame": e["frame"], "definition": e["definition"]})
            items.append(Item(f"alcuna-{eid}-{k}", set_name, f"Question: {stem}\nAnswer:", [" " + o for o in options], answer,
                              [term], None, "\n", {"entity_id": int(eid), "entity": e["entity"]["name"], "rank": e["entity"].get("rank"),
                                                   "parent": e["parent"]["name"], "parent_rank": e["parent"].get("rank"),
                                                   "type": qtype, "form": form, "difference": (q.get("meta_data") or {}).get("difference"),
                                                   "mapping": e["mapping"], "wordnet": e["wordnet"]}))
    coverage = {"entities": len(entities), "mapping": dict(kinds), "mapped": sum(1 for e in entities.values() if e["frame"]),
                "ancestor_depths": dict(sorted(depths.items())),
                "questions_by_form_type": {f"{f}|{t}": len(p) for (f, t), p in sorted(pools.items())},
                "skipped": dict(skipped), "per_type_cap": per_type, "items": len(items),
                "items_by_set": dict(Counter(i.set for i in items))}
    return items, coverage


_EXTRA = re.compile(r"<extra_id_\d+>")


def _fill(label: str) -> str:
    return " ".join(_EXTRA.sub(" ", label).split())


def build_entity_inferences(raw: Path, *, seed: int = 0) -> tuple[list[Item], dict[str, Any]]:
    items, counts = [], Counter()
    for path in sorted((raw / "data/entity_inferences").glob("*.json")):
        for row in (json.loads(line) for line in open(path) if line.strip()):
            target = _fill(row["def_target"])
            definition = " ".join(row["definition"].replace("<extra_id_0>", target).split())
            qid = re.search(r"\((Q\w+)\)", row.get("qid") or "")
            for t_name, probe in sorted(row["probe_sentences"].items()):
                sentence = probe["probe_sentence"]
                if "<extra_id_0>" not in sentence:
                    counts["no_mask"] += 1; continue
                before, after = sentence.split("<extra_id_0>", 1)
                labels = [_fill(x) for x in probe["labels"]]
                gold = _fill(row["label"]) if isinstance(row.get("label"), str) else None
                if gold not in labels:
                    counts["gold_not_in_labels"] += 1; continue
                context = before.rstrip()
                lead = " " if before.endswith(" ") or not before else ""
                options = [lead + label + after.rstrip() if after.strip() else lead + label for label in labels]
                if not context:
                    options = [o.lstrip() for o in options]
                items.append(Item(f"ei-{path.stem}-{row['ex_id']}-{t_name}", "entity-inferences", context, options, labels.index(gold),
                                  [Term.from_json({"surface": row["ent_str"], "concept": None, "frame": None, "definition": definition})],
                                  None, "\n", {"subset": path.stem, "ex_id": row["ex_id"], "category": row.get("category"),
                                               "entity": row["ent_str"], "type": (row.get("qid") or "").split(" (")[0],
                                               "type_qid": qid.group(1) if qid and qid.group(1) != "QXXXXXXX" else None,
                                               "attribute": row.get("attribute"), "wikidata_qid": None}))
                counts["items"] += 1
    return items, {"items": len(items), "skipped": {k: v for k, v in counts.items() if k != "items"},
                   "by_subset": dict(Counter(i.meta["subset"] for i in items))}


# -- read / meta sets (frames from TK-B2) ------------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path) if line.strip()]


def build_reversal(raw: Path, *, seed: int = 0, distractors: int = 9) -> tuple[list[Item], dict[str, Any]]:
    """Fictitious name–description pairs: forward and reverse tests of both training directions (p2d: trained
    "name → description"; d2p: trained "description → name"). Options are the gold completion and `distractors` others
    of the same test file; the person's training statement is the term's definition."""
    base = raw / "name_description_dataset"
    p2d_train, d2p_train = _jsonl(base / "p2d_prompts_train.jsonl"), _jsonl(base / "d2p_prompts_train.jsonl")
    statements: dict[str, str] = {}
    for row in d2p_train:                                   # description prompt → name completion
        statements.setdefault(row["completion"].strip(), row["prompt"] + row["completion"])
    p2d_names = sorted({row["completion"].strip() for row in _jsonl(base / "p2d_reverse_prompts_test.jsonl")})
    for row in p2d_train:                                   # name prompt → description completion
        for name in p2d_names:
            if name in row["prompt"] and name not in statements:
                statements[name] = row["prompt"] + row["completion"]
    items, counts = [], Counter()
    specs = [("p2d_reverse_prompts_test", "reversal-p2d-reverse", "name"), ("d2p_prompts_test", "reversal-d2p-forward", "name"),
             ("p2d_prompts_test", "reversal-p2d-forward", "description"), ("d2p_reverse_prompts_test", "reversal-d2p-reverse", "description")]
    names_all = sorted(statements, key=len, reverse=True)
    for file, set_name, completes in specs:
        rows = _jsonl(base / f"{file}.jsonl")
        pool = sorted({r["completion"] for r in rows})
        for k, row in enumerate(rows):
            rng = _rng(seed, f"{set_name}|{k}")
            others = [c for c in pool if c.strip() != row["completion"].strip()]
            options, answer = _shuffle_options(row["completion"], rng.sample(others, min(distractors, len(others))), rng)
            if completes == "name":
                option_terms = [[Term.from_json({"surface": o.strip(), "concept": None, "frame": None,
                                                 "definition": statements.get(o.strip())})] for o in options]
                terms, person = [], row["completion"].strip()
            else:
                person = next((n for n in names_all if n in row["prompt"]), None)
                if person is None:
                    counts["no_person"] += 1; continue
                option_terms, terms = None, [Term.from_json({"surface": person, "concept": None, "frame": None,
                                                             "definition": statements.get(person)})]
            items.append(Item(f"{set_name}-{k}", set_name, row["prompt"], options, answer, terms, option_terms, "\n",
                              {"person": person, "file": file, "direction": set_name.split("-", 1)[1], "wikidata_qid": None}))
    return items, {"items": len(items), "persons_with_statement": len(statements), "skipped": dict(counts),
                   "by_set": dict(Counter(i.set for i in items))}


def build_lre(raw: Path, *, seed: int = 0, distractors: int = 9) -> tuple[list[Item], dict[str, Any]]:
    items, by_relation = [], {}
    for kind, names in src.LRE_RELATIONS.items():
        for name in names:
            data = json.load(open(raw / f"data/{kind}/{name}.json"))
            template = data["prompt_templates"][0]
            objects_of: dict[str, set[str]] = defaultdict(set)
            for s in data["samples"]:
                objects_of[s["subject"]].add(s["object"])
            pool = sorted({s["object"] for s in data["samples"]})
            n = 0
            for k, s in enumerate(data["samples"]):
                others = [o for o in pool if o not in objects_of[s["subject"]]]
                if not others:
                    continue
                rng = _rng(seed, f"lre|{name}|{k}")
                options, answer = _shuffle_options(" " + s["object"], [" " + o for o in rng.sample(others, min(distractors, len(others)))], rng)
                items.append(Item(f"lre-{name}-{k}", "lre", template.replace("{}", s["subject"]), options, answer,
                                  [Term.from_json({"surface": s["subject"], "concept": None, "frame": None, "definition": None,
                                                   "insert": False})], None, "\n",
                                  {"relation": name, "relation_type": kind, "subject": s["subject"], "object": s["object"],
                                   "domain": data["properties"].get("domain_name"), "range": data["properties"].get("range_name"),
                                   "template": template, "wikidata_qid": None}))
                n += 1
            by_relation[name] = n
    return items, {"items": len(items), "relations": len(by_relation), "by_relation": by_relation}


def build_bear(raw: Path, *, seed: int = 0) -> tuple[list[Item], dict[str, Any]]:
    import pandas as pd
    frame = pd.read_parquet(raw / "BEAR/test-00000-of-00001.parquet")
    items, skipped = [], Counter()
    for row in frame.itertuples(index=False):
        template, subject = row.template, row.subject
        if template.count("[Y]") != 1:
            skipped["template"] += 1; continue
        before, after = template.split("[Y]")
        before, after = before.replace("[X]", subject), after.replace("[X]", subject)
        answers = [str(a) for a in row.answer_options]
        if before.endswith(" "):
            context, options = before[:-1], [" " + a + after for a in answers]
        else:
            context, options = before, [a + after for a in answers]
        items.append(Item(f"bear-{row.composite_id}", "bear", context, options, int(row.correct),
                          [Term.from_json({"surface": subject, "concept": None, "frame": None, "definition": None, "insert": False})],
                          None, "\n", {"relation": row.relation, "item": int(row.item), "template_index": int(row.template_index),
                                       "subject": subject, "object": answers[int(row.correct)], "wikidata_pid": row.relation,
                                       "wikidata_qid": None}))
    return items, {"items": len(items), "relations": len({i.meta["relation"] for i in items}), "skipped": dict(skipped),
                   "mean_options": sum(len(i.options) for i in items) / max(1, len(items))}


def build_popqa(raw: Path, *, seed: int = 0, distractors: int = 4) -> tuple[list[Item], dict[str, Any]]:
    import pandas as pd
    frame = pd.read_csv(raw / "test.tsv", sep="\t")
    pools: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        pools[row.prop].add(str(row.obj))
    items, skipped = [], Counter()
    for row in frame.itertuples(index=False):
        accepted = {a.lower() for a in json.loads(row.possible_answers)}
        others = sorted(o for o in pools[row.prop] if o.lower() not in accepted)
        if len(others) < 1:
            skipped["no_distractor"] += 1; continue
        rng = _rng(seed, f"popqa|{row.id}")
        options, answer = _shuffle_options(" " + str(row.obj), [" " + o for o in rng.sample(others, min(distractors, len(others)))], rng)
        qid = str(row.s_uri).rsplit("/", 1)[-1]
        items.append(Item(f"popqa-{row.id}", "popqa", f"Q: {row.question} A:", options, answer,
                          [Term.from_json({"surface": str(row.subj), "concept": qid, "frame": None, "definition": None, "insert": False})],
                          None, "\n", {"subject": row.subj, "object": row.obj, "prop": row.prop, "prop_id": int(row.prop_id),
                                       "wikidata_qid": qid, "object_qid": str(row.o_uri).rsplit("/", 1)[-1],
                                       "s_pop": int(row.s_pop), "o_pop": int(row.o_pop),
                                       "possible_answers": json.loads(row.possible_answers)}))
    return items, {"items": len(items), "props": len(pools), "skipped": dict(skipped)}


# -- builds ------------------------------------------------------------------------------------------------------------------

SETS: dict[str, tuple[str, Callable[..., tuple[list[Item], dict[str, Any]]], bool]] = {
    # set → (source, builder, needs WordNet frames)
    "comps-wugs": ("comps", build_comps_wugs, True),
    "alcuna": ("alcuna", build_alcuna, True),
    "entity-inferences": ("entity-inferences", build_entity_inferences, False),
    "reversal": ("reversal", build_reversal, False),
    "lre": ("lre", build_lre, False),
    "bear": ("bear", build_bear, False),
    "popqa": ("popqa", build_popqa, False),
}
DESCRIPTIONS = {
    "comps-wugs": "COMPS-WUGS minimal pairs over the nonce word's parent; WordNet is-a frames (E9 WordNet track atomics)",
    "alcuna": "ALCUNA ranking forms (multiple choice, Yes/No) on artificial taxa whose parent maps to WordNet; is-a frames",
    "entity-inferences": "Entity Inferences: definition as reading material, probe labels ranked; frames from TK-B2",
    "reversal": "Reversal curse, fictitious name-description pairs: forward and reverse, both directions; frames from TK-B2",
    "lre": "LRE relations: subject prompt (first template), gold object vs 9 objects of the relation; frames from TK-B2",
    "bear": "BEAR: every template, the item's answer options; frames from TK-B2",
    "popqa": "PopQA as 5-way ranking (gold object vs 4 objects of the same property); frames from TK-B2",
}


def build(name: str, output: Path | None = None, *, ontology: Path = WORDNET_ONTOLOGY, seed: int = 0, root: Path = src.ROOT,
          commit_dir: Path | None = None, frames: WordNetFrames | None = None, **options: Any) -> dict[str, Any]:
    source_name, builder, needs_frames = SETS[name]
    raw = src.raw_dir(source_name, root)
    source = src.SOURCES[source_name]
    if needs_frames:
        frames = frames or load_wordnet_frames(ontology)
        items, coverage = builder(raw, frames, seed=seed, **options)
    else:
        items, coverage = builder(raw, seed=seed, **options)
    suffix = f"-wordnet-{VERSION}" if needs_frames else f"-{VERSION}"
    output = Path(output) if output else Path(root).expanduser() / source_name / "items" / f"{name}{suffix}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    info = write_items(output / "items.jsonl.gz", items)
    source_record = json.loads((Path(root).expanduser() / source_name / "source.json").read_text())
    manifest = {"schema": SCHEMA, "set": name + suffix, "description": DESCRIPTIONS[name], "created": time.strftime("%Y-%m-%d"),
                "builder": f"vsa_embed.benchmarks.adapters build --set {name}", "seed": seed, "options": options,
                "source": {"name": source_name, "title": source.title, "citation": source.citation, "licence": source.licence,
                           "licence_evidence": source.licence_evidence, "pinned": source.commit,
                           "files": [{"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"]} for f in source_record["files"]]},
                "frames": ({"ontology": str(ontology), "ontology_sha256": src.sha256_file(Path(ontology)),
                            "rule": "hypernym → nearest WordNet is-a atomic of the track (depth recorded) + lexname + pos"}
                           if needs_frames else "none (Wikidata frames from the T8 track, TK-B2)"),
                "counts": {"items": len(items), "by_set": dict(Counter(i.set for i in items))}, "coverage": coverage,
                "items_file": {"path": "items.jsonl.gz", **info}, "committed_items": bool(source.commit_items)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if commit_dir is not None:
        target = Path(commit_dir) / (name + suffix)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "manifest.json", target / "manifest.json")
        if source.commit_items:
            shutil.copy2(output / "items.jsonl.gz", target / "items.jsonl.gz")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", dest="name", choices=list(SETS), nargs="+", required=True)
    b.add_argument("--output", type=Path, default=None, help="only with a single --set")
    b.add_argument("--ontology", type=Path, default=WORDNET_ONTOLOGY); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--root", type=Path, default=src.ROOT); b.add_argument("--commit-dir", type=Path, default=None)
    b.add_argument("--per-type", type=int, default=1000, help="ALCUNA: items per (form, question type)")
    args = parser.parse_args(argv)
    frames = load_wordnet_frames(args.ontology) if any(SETS[n][2] for n in args.name) else None
    for name in args.name:
        options = {"per_type": args.per_type} if name == "alcuna" else {}
        manifest = build(name, args.output if len(args.name) == 1 else None, ontology=args.ontology, seed=args.seed, root=args.root,
                         commit_dir=args.commit_dir, frames=frames, **options)
        print(json.dumps({"set": manifest["set"], "counts": manifest["counts"], "coverage": manifest["coverage"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()
