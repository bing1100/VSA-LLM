"""E5.4 zero-shot insertion through the span channel (D5.4; experiments.md E5.4, proposal H-E).

Dossier experiment 02's strict protocol with the span channel instead of tokenizer surgery: reserved
concepts never received a gradient (held-out entries, linked only at evaluation), definitions never
appear in prompts, and contamination is controlled with invented names. A *source* decides which row
the channel injects for the reserved concepts (everything else in the model is unchanged):

- structure-only: `own` (the run's own row: the composed row for composition runs — the candidate;
  the mean-vector fallback for C2; nothing for C0), `none` (no injection), `random` (matched-norm
  random vector), `mean_row` (mean of trained rows: the C2 fallback inside this model),
  `surface_mean` (mean input embedding of the alias's subtokens), `graph_projection` (spectral
  embedding of the typed frame graph, folded in for unseen rows, ridge-mapped to rows fitted on seen
  concepts);
- text evidence (reported separately, never part of the structure-only claim): `definition_mean`
  (mean input embedding of the definition), `definition_encoder` (the frozen model's mean hidden
  state over the definition, ridge-mapped), `alacarte` (Khodak et al. 2018: linear map from the
  average context embedding of a few example contexts), `college` (CoLLEGe-style: a small MLP from
  the model's pooled hidden states over a few example contexts, trained on seen concepts).

Maps are fitted on seen concepts (training frequency ≥ 10) against the run's own rows; baseline
rows are rescaled to the mean norm of seen rows (except `none` and `mean_row`). Example contexts come
from the run's evaluation corpus, odd-numbered documents only; corpus-based tests use even-numbered
documents only.

Tests (per item; paired bootstrap of `own` − baseline over items, Holm over baselines per test):
- `property` (which relation filler fits): PMI of each candidate continuation, `log p(y | T(x)) −
  log p(y | T(null))`, argmax = gold; every item has ≥ 2 template paraphrases;
- `entailment` (is-a): an ancestor two or three levels up vs a non-ancestor;
- `paraphrase`: the property argmax agrees across the paraphrases;
- `semantic_rank` (runs trained with `semantic_weight > 0`): rank of the reserved concept's row among
  all entries' rows for the semantic head's query `W_o h_{s−1}` at its eval-corpus occurrences;
- `after_loss`: loss on the 8 tokens after each eval-corpus occurrence (even documents).

Scenarios (item builders): `c3_heldout` (C3 held-out WordNet concepts, real aliases; contaminated on
pretrained hosts, so from-scratch models only), `c3_synthetic` (the same concepts and frames under
invented names — contamination-free on every host), `c6_devtools` (held-out symbols of the C6
synthetic private library). Other tracks (T1-open MeSH, T3–T6) plug in by writing an item directory
in the same format (or registering a builder in `SCENARIOS`).

Item directory (`e5-zeroshot-items/1`): `manifest.json` (scenario, contamination_free,
`definitions` source, `corpus_tests`, tokenizer, ℓ_min, seed, counts, checks), `concepts.jsonl`
(`concept`, `surface`, `entry` or null, `synthetic`, `source_concept`, `pos`, `definition` — text
evidence, never in prompts — and `gold`), `items.jsonl` (`id`, `concept`, `test`, `relation`,
`templates` with `{x}`, `null`, `candidates` (continuations with a leading space), `gold` index).
An optional `definitions.jsonl` (`entry`, `definition`) gives definitions of seen entries for the
text-evidence maps of external scenarios.

    python -m vsa_embed.experiments.e5_zeroshot items --scenario c3_heldout --ontology ONT --output DIR
    python -m vsa_embed.experiments.e5_zeroshot items --scenario c3_synthetic --ontology ONT --output DIR
    python -m vsa_embed.experiments.e5_zeroshot items --scenario c6_devtools --benchmark BENCH --output DIR
    python -m vsa_embed.experiments.e5_zeroshot evaluate --run RUN --items DIR --output OUT [--sources …]
        [--quantize int8|int4 [--quantize-channel]]

`--quantize` repeats the evaluation on the run after the post-training weight-only quantization of
`e4_quant` (output head FP; `--quantize-channel` = its variant B with the channel quantized too); the
baseline rows are then fitted against the quantized model's own rows (E9 dimensions 1 and 3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data.corpus import TokenCorpus, collate_windows, eval_windows
from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable, CausalLinker, normalize_alias
from ..statistics import holm_adjust
from ..training.lm import AFTER_WINDOW
from .e5_common import (E5Run, canonical_surfaces, clear_output, entry_rows, finish_output, fmt, fmt_ci, forward_tokens,
                        json_ready, open_run, override_rows, start_output, status_of, write_json)

SCHEMA = "e5-zeroshot-items/1"
STRUCTURE_SOURCES = ("own", "none", "random", "mean_row", "surface_mean", "graph_projection")
TEXT_SOURCES = ("definition_mean", "definition_encoder", "alacarte", "college")
ALL_SOURCES = STRUCTURE_SOURCES + TEXT_SOURCES
UNSCALED = {"own", "none", "mean_row"}
NULL_SURFACE = "thing"

# Template paraphrases per (part of speech, relation); `{x}` is the concept's surface form.
TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("n", "hypernym"): ["The {x} is a kind of", "Every {x} is a type of"],
    ("n", "instance_hypernym"): ["{x} is an instance of", "{x} is one example of"],
    ("n", "part_holonym"): ["The {x} is part of the", "The {x} is a component of the"],
    ("n", "member_holonym"): ["The {x} is a member of the", "The {x} belongs to the group"],
    ("n", "substance_holonym"): ["The {x} is a substance found in", "The {x} is an ingredient of"],
    ("n", "part_meronym"): ["The {x} has a part called the", "One part of the {x} is the"],
    ("n", "member_meronym"): ["The {x} has a member called the", "One member of the {x} is the"],
    ("n", "substance_meronym"): ["The {x} is made of", "The {x} contains"],
    ("n", "topic_domain"): ["The {x} belongs to the field of", "The {x} is a term used in"],
    ("n", "lexname"): ["The word {x} refers to a kind of", "In general, the {x} is a kind of"],
    ("v", "hypernym"): ["To {x} is a way to", "To {x} means to"],
    ("v", "entailment"): ["To {x} you must", "If you {x}, you also"],
    ("v", "cause"): ["To {x} something causes it to", "When you {x} something, it will"],
    ("v", "lexname"): ["To {x} is an act of", "The verb {x} is about"],
}
ENTAILMENT_TEMPLATES = {"n": ["The {x} is a kind of", "Every {x} is a type of"], "v": ["To {x} is a way to", "To {x} means to"]}
LEXNAME_WORDS = {
    "noun.Tops": "entity", "noun.act": "act", "noun.animal": "animal", "noun.artifact": "artifact",
    "noun.attribute": "attribute", "noun.body": "body part", "noun.cognition": "idea", "noun.communication": "communication",
    "noun.event": "event", "noun.feeling": "feeling", "noun.food": "food", "noun.group": "group", "noun.location": "location",
    "noun.motive": "motive", "noun.object": "natural object", "noun.person": "person", "noun.phenomenon": "phenomenon",
    "noun.plant": "plant", "noun.possession": "possession", "noun.process": "process", "noun.quantity": "quantity",
    "noun.relation": "relation", "noun.shape": "shape", "noun.state": "state", "noun.substance": "substance",
    "noun.time": "time", "verb.body": "body", "verb.change": "change", "verb.cognition": "thinking",
    "verb.communication": "communication", "verb.competition": "competition", "verb.consumption": "consumption",
    "verb.contact": "contact", "verb.creation": "creation", "verb.emotion": "emotion", "verb.motion": "motion",
    "verb.perception": "perception", "verb.possession": "possession", "verb.social": "social life",
    "verb.stative": "being", "verb.weather": "weather",
}
DEVTOOLS_TEMPLATES = {
    "returns": ["The function `{x}` returns", "Calling `{x}` gives a value of type"],
    "module": ["`{x}` is defined in the module", "You can import `{x}` from the module"],
    "kind": ["`{x}` is a", "In the library, `{x}` is a"],
    "category": ["`{x}` is used for", "People use `{x}` for"],
}


# -- item files -----------------------------------------------------------------------------------

def write_items(out_dir: Path, manifest: dict[str, Any], concepts: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "concepts.jsonl").write_text("".join(json.dumps(json_ready(c)) + "\n" for c in concepts))
    (out_dir / "items.jsonl").write_text("".join(json.dumps(json_ready(i)) + "\n" for i in items))
    tests = defaultdict(int)
    for item in items:
        tests[item["test"]] += 1
    manifest = {"schema": SCHEMA, **manifest, "counts": {"concepts": len(concepts), "items": len(items), **tests}}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_items(items_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    items_dir = Path(items_dir)
    manifest = json.loads((items_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{items_dir} is not an {SCHEMA} item directory")
    read = lambda name: [json.loads(line) for line in (items_dir / name).read_text().splitlines() if line.strip()]
    concepts, items = read("concepts.jsonl"), read("items.jsonl")
    known = {c["concept"] for c in concepts}
    for item in items:
        if item["concept"] not in known:
            raise ValueError(f"item {item['id']} refers to unknown concept {item['concept']}")
        if not 0 <= int(item["gold"]) < len(item["candidates"]) or len(item["templates"]) < 1:
            raise ValueError(f"item {item['id']} is malformed")
    definitions = {c["concept"]: c.get("definition") for c in concepts}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    for item in items:
        definition = definitions[item["concept"]]
        prompts = [t.format(x=surfaces[item["concept"]]) + c for t in item["templates"] for c in item["candidates"]]
        if definition and len(definition) > 12 and any(definition in p for p in prompts):
            raise ValueError(f"the definition of {item['concept']} appears in a prompt of {item['id']}")
    return manifest, concepts, items


# -- scenario builders -------------------------------------------------------------------------------

def _first_lemma(synset_name: str) -> str:
    return synset_name.rsplit(".", 2)[0].replace("_", " ")


def _filler_text(atomic: str) -> str | None:
    kind, _, value = atomic.partition(":")
    if kind == "synset":
        return _first_lemma(value)
    if kind == "lexname":
        return LEXNAME_WORDS.get(value)
    return None


def _ancestors(synset: Any) -> dict[str, int]:
    """Ancestor synset name → shortest hypernym distance."""
    out: dict[str, int] = {}
    frontier, depth = [synset], 0
    while frontier and depth < 20:
        depth += 1
        nxt = []
        for s in frontier:
            for parent in s.hypernyms() + s.instance_hypernyms():
                if parent.name() not in out:
                    out[parent.name()] = depth; nxt.append(parent)
        frontier = nxt
    return out


def reserved_wordnet_entries(ontology: dict[str, Any], table: AliasTable, tokenizer: Any, *, min_subtokens: int,
                             max_concepts: int, seed: int, wordnet: Any) -> list[tuple[int, dict[str, Any]]]:
    """Held-out single-concept noun/verb entries whose canonical alias links (≥ ℓ_min subtokens),
    a seeded sample of `max_concepts` (the same sample for `c3_heldout` and `c3_synthetic`)."""
    names = ontology["concept_names"]
    held = sorted(int(e) for e in ontology["heldout_entries"])
    single = [e for e in held if len(table.entry_concepts[e]) == 1
              and names[table.entry_concepts[e][0]].rsplit(".", 2)[1] in {"n", "v"}]
    surfaces = canonical_surfaces(table, tokenizer, min_subtokens, single)
    eligible = [e for e in single if e in surfaces and surfaces[e]["linkable"]]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible, min(max_concepts, len(eligible))))
    return [(e, surfaces[e]) for e in chosen]


def build_c3_items(ontology_path: Path, out_dir: Path, *, synthetic: bool, tokenizer_name: str = "gpt2",
                   min_subtokens: int = 2, max_concepts: int = 400, distractors: int = 4, seed: int = 0,
                   wordnet: Any = None, alias_table: Path | None = None, contamination_texts: Iterable[str] = (),
                   name_seed: int = 7) -> dict[str, Any]:
    """`c3_heldout` (real aliases) or `c3_synthetic` (invented names on the same concepts and frames)."""
    from transformers import AutoTokenizer
    if wordnet is None:
        from nltk.corpus import wordnet
    ontology = torch.load(ontology_path, weights_only=False)
    table, table_info = cp.resolve_alias_table(ontology, Path(ontology_path), alias_table=alias_table)
    if table_info.get("holdout_names") and Path(table_info["holdout_names"]).is_relative_to(cp.REPO_ROOT):
        table_info["holdout_names"] = str(Path(table_info["holdout_names"]).relative_to(cp.REPO_ROOT))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    reserved = reserved_wordnet_entries(ontology, table, tokenizer, min_subtokens=min_subtokens, max_concepts=max_concepts,
                                        seed=seed, wordnet=wordnet)
    names, atoms, relations = ontology["concept_names"], ontology["atomic_names"], ontology["relation_names"]
    offsets = torch.as_tensor(ontology["offsets"]); rel = torch.as_tensor(ontology["relations"]); fil = torch.as_tensor(ontology["fillers"])
    # Filler pools per (POS, relation) over every entry's frame (readable strings).
    pools: dict[tuple[str, str], set[str]] = defaultdict(set)
    pos_of_entry = [names[c[0]].rsplit(".", 2)[1] if c else "?" for c in table.entry_concepts]
    for e in range(len(table.entry_concepts)):
        pos = pos_of_entry[e]
        for r, f in zip(rel[offsets[e]:offsets[e + 1]].tolist(), fil[offsets[e]:offsets[e + 1]].tolist()):
            text = _filler_text(atoms[f])
            if text and (pos, relations[r]) in TEMPLATES:
                pools[(pos, relations[r])].add(text)
    noun_pool = sorted(pools[("n", "hypernym")])
    verb_pool = sorted(pools[("v", "hypernym")])
    invented: dict[int, str] = {}
    checks: dict[str, Any] = {}
    if synthetic:
        invented, checks = invent_names([e for e, _ in reserved], tokenizer, table, wordnet, min_subtokens=min_subtokens,
                                        seed=name_seed, contamination_texts=contamination_texts)
    rng = random.Random(seed + 1)
    concepts, items = [], []
    prefix = "c3s" if synthetic else "c3h"
    for entry, surface_info in reserved:
        concept_index = table.entry_concepts[entry][0]
        synset = wordnet.synset(names[concept_index])
        pos = synset.pos()
        cid = f"{prefix}-{entry}"
        surface = invented[entry] if synthetic else surface_info["surface"]
        frame = [(relations[r], atoms[f]) for r, f in zip(rel[offsets[entry]:offsets[entry + 1]].tolist(),
                                                         fil[offsets[entry]:offsets[entry + 1]].tolist())]
        ancestors = _ancestors(synset)
        ancestor_texts = {_first_lemma(a) for a in ancestors}
        gold: dict[str, list[str]] = defaultdict(list)
        for relation, atom in frame:
            text = _filler_text(atom)
            if text and (pos, relation) in TEMPLATES and text != surface_info["surface"]:
                gold[relation].append(text)
        lexname = dict(frame).get("lexname")
        concepts.append({"concept": cid, "surface": surface, "entry": entry, "synthetic": synthetic,
                         "source_concept": names[concept_index], "pos": pos, "definition": synset.definition(),
                         "gold": dict(gold), "lexname": lexname})
        for relation in sorted(gold):
            right = gold[relation][0]
            exclude = set(gold[relation]) | {surface_info["surface"]} | (ancestor_texts if relation in {"hypernym", "lexname"} else set())
            pool = [t for t in sorted(pools[(pos, relation)]) if t not in exclude]
            if len(pool) < distractors:
                continue
            wrong = rng.sample(pool, distractors)
            candidates = [right] + wrong
            rng.shuffle(candidates)
            items.append({"id": f"{cid}-property-{relation}", "concept": cid, "test": "property", "relation": relation,
                          "templates": TEMPLATES[(pos, relation)], "null": NULL_SURFACE,
                          "candidates": [" " + c for c in candidates], "gold": candidates.index(right)})
        pool = noun_pool if pos == "n" else verb_pool
        deep = sorted(a for a, d in ancestors.items() if d in (2, 3))
        for depth_rank, ancestor in enumerate(rng.sample(deep, min(2, len(deep)))):
            right = _first_lemma(ancestor)
            negatives = [t for t in pool if t not in ancestor_texts and t != surface_info["surface"] and t != right]
            if not negatives or right == surface_info["surface"]:
                continue
            wrong = rng.choice(negatives)
            candidates = [right, wrong]
            rng.shuffle(candidates)
            items.append({"id": f"{cid}-entailment-{depth_rank}", "concept": cid, "test": "entailment",
                          "relation": f"ancestor_depth_{ancestors[ancestor]}", "templates": ENTAILMENT_TEMPLATES[pos],
                          "null": NULL_SURFACE, "candidates": [" " + c for c in candidates], "gold": candidates.index(right)})
    manifest = {"scenario": "c3_synthetic" if synthetic else "c3_heldout",
                "contamination_free": synthetic,
                "description": ("C3 held-out WordNet concepts under invented names (frames of held-out entries)" if synthetic
                                else "C3 held-out WordNet concepts with their real aliases (from-scratch models only)"),
                "ontology": str(ontology_path), "ontology_alias_sha256": table.digest(), "alias_table": table_info,
                "tokenizer": tokenizer_name, "min_subtokens": min_subtokens, "seed": seed, "max_concepts": max_concepts,
                "distractors": distractors, "definitions": "wordnet", "corpus_tests": not synthetic,
                "synthetic_aliases": synthetic, "null_surface": NULL_SURFACE, "checks": checks,
                "selection": "held-out single-concept noun/verb entries whose shortest alias has ≥ ℓ_min subtokens; "
                             "seeded uniform sample (identical for c3_heldout and c3_synthetic at equal seed and size)"}
    return write_items(out_dir, manifest, concepts, items)


def invent_names(entries: Sequence[int], tokenizer: Any, table: AliasTable, wordnet: Any, *, min_subtokens: int,
                 seed: int, contamination_texts: Iterable[str] = ()) -> tuple[dict[int, str], dict[str, Any]]:
    """Pronounceable invented names (C6 generator), not a WordNet lemma, not an alias, not one token
    of the tokenizer, ≥ `min_subtokens` subtokens mid-sentence and absent from `contamination_texts`."""
    from ..benchmarks.devtools import NameMaker
    lemmas = {l.lower() for s in wordnet.all_synsets() for l in s.lemma_names()}
    forbidden = lemmas | {a for alias in table.alias_to_entry for a in alias.split()}
    haystack = "\n".join(t.lower() for t in contamination_texts)
    maker = NameMaker(random.Random(seed), forbidden)
    vocabulary = {t.lstrip("Ġ▁ ").lower() for t in tokenizer.get_vocab()}
    names: dict[int, str] = {}
    rejected = 0
    for entry in entries:
        while True:
            name = maker.stem() + maker.stem()
            if (name in forbidden or name in vocabulary or len(tokenizer.encode(" " + name, add_special_tokens=False)) < min_subtokens
                    or (haystack and name in haystack)):
                rejected += 1
                continue
            names[entry] = name
            break
    checks = {"names": len(names), "rejected_candidates": rejected, "checked_against": ["wordnet lemmas", "alias table words",
              "tokenizer vocabulary"] + (["contamination texts"] if haystack else []), "contamination_chars": len(haystack)}
    return names, checks


def build_c6_items(benchmark_dir: Path, out_dir: Path, *, tokenizer_name: str = "gpt2", min_subtokens: int = 2,
                   distractors: int = 4, seed: int = 0) -> dict[str, Any]:
    """Held-out symbols of the C6 synthetic private library (contamination-free by construction)."""
    from ..benchmarks.devtools import BUILTIN_TYPES, CATEGORIES
    library = json.loads((Path(benchmark_dir) / "library.json").read_text())
    symbols = library["symbols"]
    rng = random.Random(seed)
    classes = [s["name"] for s in symbols if s["kind"] == "class"]
    types = sorted(set(BUILTIN_TYPES) | set(classes))
    concepts, items = [], []
    from ..benchmarks.devtools import _describe
    for s in symbols:
        if not s["heldout"] or s["kind"] not in {"function", "method", "class"}:
            continue
        cid = f"c6-{s['name']}"
        gold = {"module": [s["module"]], "kind": [s["kind"]], "category": [s["category"]]}
        if s["returns"]:
            gold["returns"] = [s["returns"]]
        concepts.append({"concept": cid, "surface": s["name"], "entry": None, "synthetic": False, "source_concept": s["name"],
                         "pos": s["kind"], "definition": _describe(s, random.Random(0)), "gold": gold})
        pools = {"module": library["modules"], "kind": ["function", "class", "method"], "category": CATEGORIES, "returns": types}
        for relation, right_list in sorted(gold.items()):
            right = right_list[0]
            pool = [p for p in pools[relation] if p != right and p != s["name"]]
            wrong = rng.sample(pool, min(distractors, len(pool)))
            candidates = [right] + wrong
            rng.shuffle(candidates)
            fmt_candidate = (lambda c: f" `{c}`") if relation == "module" else (lambda c: f" {c}")
            items.append({"id": f"{cid}-property-{relation}", "concept": cid, "test": "property", "relation": relation,
                          "templates": DEVTOOLS_TEMPLATES[relation], "null": NULL_SURFACE,
                          "candidates": [fmt_candidate(c) for c in candidates], "gold": candidates.index(right)})
    manifest = {"scenario": "c6_devtools", "contamination_free": True,
                "description": "held-out symbols of the C6 synthetic private library (T2)", "benchmark": str(benchmark_dir),
                "tokenizer": tokenizer_name, "min_subtokens": min_subtokens, "seed": seed, "distractors": distractors,
                "definitions": "devtools", "corpus_tests": True, "synthetic_aliases": False, "null_surface": NULL_SURFACE,
                "checks": {}, "selection": "every held-out function, method and class symbol"}
    return write_items(out_dir, manifest, concepts, items)


SCENARIOS: dict[str, Callable[..., dict[str, Any]]] = {
    "c3_heldout": lambda **kw: build_c3_items(synthetic=False, **kw),
    "c3_synthetic": lambda **kw: build_c3_items(synthetic=True, **kw),
    "c6_devtools": build_c6_items,
}


def register_scenario(name: str, builder: Callable[..., dict[str, Any]]) -> None:
    """Plug-in point for other tracks (T1-open MeSH, T3–T6): the builder writes an item directory."""
    if name in SCENARIOS:
        raise ValueError(f"scenario {name} already registered")
    SCENARIOS[name] = builder


# -- evaluation: linking and entries -------------------------------------------------------------------------

def scenario_adapter(run: E5Run, concepts: Sequence[dict[str, Any]]) -> cp.ChannelModelAdapter:
    """The run's adapter; with synthetic concepts, its linker also maps each invented name to the entry."""
    synthetic = {normalize_alias(c["surface"]): int(c["entry"]) for c in concepts if c.get("synthetic")}
    if not synthetic or run.adapter.linker is None:
        return run.adapter
    table = run.adapter.linker.table
    clash = sorted(a for a in synthetic if a in table.alias_to_entry)
    if clash:
        raise ValueError(f"invented names already are aliases: {clash[:5]}")
    extended = AliasTable({**table.alias_to_entry, **synthetic}, table.entry_concepts, table.holdout)
    linker = CausalLinker(extended, boundary=run.adapter.linker.boundary, min_subtokens=run.adapter.linker.min_subtokens)
    return dataclasses.replace(run.adapter, linker=linker, spans_fn=None)


def resolve_entries(adapter: cp.ChannelModelAdapter, concepts: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]],
                    heldout: set[int], frequency: np.ndarray | None) -> dict[str, dict[str, Any]]:
    """Per concept: the entry the linker injects at the surface in its first prompt, and its status."""
    first = {}
    for item in items:
        first.setdefault(item["concept"], item["templates"][0])
    texts, spans, ids = [], [], []
    for c in concepts:
        template = first.get(c["concept"], "{x}")
        text = template.format(x=c["surface"])
        start = template.index("{x}")
        texts.append(text); spans.append((start, start + len(c["surface"]))); ids.append(c["concept"])
    linked = adapter.link_targets(texts, spans) if adapter.linker is not None else [[] for _ in texts]
    out = {}
    for c, found in zip(concepts, linked):
        expected = c.get("entry")
        entry = found[0] if len(found) == 1 else None
        if entry is None:
            status = "unlinked"
        elif expected is not None and int(expected) != entry:
            status = "mislinked"
        else:
            status = status_of(entry, heldout, frequency)
        out[c["concept"]] = {"entry": entry if entry is not None else expected, "linked": entry is not None, "status": status}
    return out


# -- evaluation: prompt scoring ----------------------------------------------------------------------------

def _prompt_texts(items: Sequence[dict[str, Any]], surfaces: dict[str, str]) -> tuple[list[str], list[str], list[tuple[int, int, int]]]:
    prefixes, continuations, index = [], [], []
    for i, item in enumerate(items):
        for t, template in enumerate(item["templates"]):
            prefix = template.format(x=surfaces[item["concept"]])
            for c, candidate in enumerate(item["candidates"]):
                prefixes.append(prefix); continuations.append(candidate); index.append((i, t, c))
    return prefixes, continuations, index


def score_prompts(adapter: cp.ChannelModelAdapter, items: Sequence[dict[str, Any]], surfaces: dict[str, str],
                  null_scores: np.ndarray | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Per item: PMI scores per template × candidate, correctness and paraphrase agreement."""
    prefixes, continuations, index = _prompt_texts(items, surfaces)
    raw = cp.continuation_logprob(adapter, prefixes, continuations) if prefixes else np.zeros(0)
    if null_scores is None:
        null_surfaces = {item["concept"]: item.get("null", NULL_SURFACE) for item in items}
        null_prefixes, null_continuations, _ = _prompt_texts(items, null_surfaces)
        unique = sorted(set(zip(null_prefixes, null_continuations)))
        values = dict(zip(unique, cp.continuation_logprob(adapter, [u[0] for u in unique], [u[1] for u in unique]))) if unique else {}
        null_scores = np.asarray([values[(p, c)] for p, c in zip(null_prefixes, null_continuations)])
    pmi = raw - null_scores
    results = []
    cursor = 0
    for item in items:
        k, n_t = len(item["candidates"]), len(item["templates"])
        scores = pmi[cursor:cursor + k * n_t].reshape(n_t, k)
        raws = raw[cursor:cursor + k * n_t].reshape(n_t, k)
        cursor += k * n_t
        argmax = scores.argmax(1)
        correct = (argmax == int(item["gold"])).astype(float)
        results.append({"id": item["id"], "concept": item["concept"], "test": item["test"], "relation": item["relation"],
                        "correct": float(correct.mean()), "all_templates_correct": int(correct.all()),
                        "raw_correct": float((raws.argmax(1) == int(item["gold"])).mean()),
                        "consistent": int(len(set(argmax.tolist())) == 1) if n_t > 1 else None,
                        "margin": float(np.mean([s[int(item["gold"])] - np.max(np.delete(s, int(item["gold"]))) for s in scores])),
                        "pmi": scores.round(4).tolist()})
    return results, null_scores


# -- evaluation: baseline rows ---------------------------------------------------------------------------------

def ridge_map(x: np.ndarray, y: np.ndarray, alphas: Sequence[float] = tuple(10.0 ** p for p in range(-2, 5))):
    """Multi-output ridge on standardized inputs, strength chosen by generalized cross-validation."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mean, std = x.mean(0), x.std(0) + 1e-8
    y_mean = y.mean(0)
    u, s, vt = np.linalg.svd((x - mean) / std, full_matrices=False)
    uy = u.T @ (y - y_mean)
    best = None
    for alpha in alphas:
        shrink = s ** 2 / (s ** 2 + alpha)
        residual = (y - y_mean) - u @ (shrink[:, None] * uy)
        score = float((residual ** 2).mean() / max(1e-12, 1 - shrink.sum() / len(y)) ** 2)
        if best is None or score < best[0]:
            best = (score, alpha)
    w = vt.T @ ((s / (s ** 2 + best[1]))[:, None] * uy)
    return (lambda z: ((np.asarray(z, dtype=np.float64) - mean) / std) @ w + y_mean), best[1]


def graph_features(ontology: dict[str, Any], fit_entries: Sequence[int], *, rank: int = 64) -> Callable[[Sequence[int]], np.ndarray]:
    """Spectral embedding of the typed frame graph (entries × (relation, filler) incidence, tf-idf,
    row-normalized), fitted by truncated SVD on `fit_entries` only; any entry is folded in from its frame."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    offsets = np.asarray(ontology["offsets"]); relations = np.asarray(ontology["relations"])
    fillers = np.asarray(ontology["fillers"]); atomic_count = int(ontology["atomic_count"])
    entry_count = len(offsets) - 1
    columns = relations * atomic_count + fillers
    rows = np.repeat(np.arange(entry_count), np.diff(offsets))
    width = int(columns.max()) + 1 if columns.size else 1
    incidence = csr_matrix((np.ones(len(columns)), (rows, columns)), shape=(entry_count, width))
    incidence.data[:] = 1.0
    fit = np.asarray(sorted(set(int(e) for e in fit_entries)))
    df = np.asarray((incidence[fit] > 0).sum(0)).ravel()
    idf = np.log((1 + len(fit)) / (1 + df)) + 1.0

    def weighted(entries: np.ndarray):
        m = incidence[entries].multiply(idf[None, :]).tocsr()
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        return csr_matrix(m.multiply(1.0 / np.maximum(norms, 1e-12)[:, None]))

    k = max(1, min(rank, min(len(fit), width) - 1))
    _, s, vt = svds(weighted(fit), k=k, random_state=0)
    projection = vt.T / np.maximum(s, 1e-12)[None, :]
    return lambda entries: np.asarray(weighted(np.asarray(list(entries), dtype=np.int64)) @ projection)


def _scaled(vector: np.ndarray, norm: float) -> torch.Tensor:
    vector = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(vector)
    return torch.tensor(vector * (norm / length) if length > 1e-12 else vector, dtype=torch.float32)


def context_snippets(run: E5Run, entries: Iterable[int], *, k: int, window: int = 24, parity: int = 1,
                     replace: dict[int, str] | None = None) -> dict[int, list[tuple[str, int, int]]]:
    """Up to `k` example contexts per entry from the run's evaluation corpus (documents of the given
    parity): `(text, char_start, char_end)` of the mention; `replace` swaps a mention's text for an
    invented name (synthetic concepts)."""
    from .e5_common import document_index
    corpus = TokenCorpus.open(Path(run.config["data"]["eval"]))
    wanted = {int(e) for e in entries}
    spans = corpus.spans
    keep = (spans["length"] >= run.min_subtokens) & np.isin(spans["entry"], np.asarray(sorted(wanted), dtype=np.int64))
    starts, ends, found = spans["start"][keep], spans["end"][keep], spans["entry"][keep]
    docs = document_index(corpus, starts)
    out: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    tokenizer = run.tokenizer
    for s, e, entry, doc in zip(starts.tolist(), ends.tolist(), found.tolist(), docs.tolist()):
        if doc % 2 != parity or len(out[entry]) >= k:
            continue
        lo, hi = max(0, s - window), min(len(corpus), e + 1 + window)
        ids = np.asarray(corpus.tokens[lo:hi], dtype=np.int64)
        eos = int(corpus.manifest.get("eos_id", -1))
        before, mention, after = ids[:s - lo], ids[s - lo:e + 1 - lo], ids[e + 1 - lo:]
        before = before[np.flatnonzero(before == eos)[-1] + 1:] if (before == eos).any() else before
        after = after[:np.flatnonzero(after == eos)[0]] if (after == eos).any() else after
        left = tokenizer.decode(before.tolist()); text = tokenizer.decode(mention.tolist()); right = tokenizer.decode(after.tolist())
        if replace and entry in replace:
            text = (" " if text.startswith(" ") else "") + replace[entry]
        offset = len(left) + (len(text) - len(text.lstrip()))
        out[entry].append((left + text + right, offset, len(left) + len(text)))
    return dict(out)


@torch.no_grad()
def pooled_states(adapter: cp.ChannelModelAdapter, channel: Any, texts: Sequence[str], exclude: Sequence[tuple[int, int] | None],
                  drop_entries: Iterable[int]) -> np.ndarray:
    """Mean final hidden state per text over tokens outside `exclude` (character span), with the
    channel live except for `drop_entries` (zero rows, i.e. not injected)."""
    drop = {int(e): torch.zeros(channel.gate.in_features // 2) for e in drop_entries} if channel is not None else {}
    out = []
    with override_rows(channel, drop) if channel is not None else _null():
        states, offsets, _, _ = adapter._forward(list(texts))
    for state, offs, span in zip(states, offsets, exclude):
        mask = torch.ones(len(offs), dtype=torch.bool)
        if span is not None:
            for i, (a, b) in enumerate(offs):
                if a < span[1] and b > span[0]:
                    mask[i] = False
        out.append((state[mask] if mask.any() else state).mean(0).numpy())
    return np.stack(out) if out else np.zeros((0, 0))


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def mean_embedding(tokenizer: Any, wte: np.ndarray, texts: Sequence[str], exclude: Sequence[tuple[int, int] | None]) -> np.ndarray:
    encoded = tokenizer(list(texts), return_offsets_mapping=True, add_special_tokens=False)
    out = []
    for ids, offs, span in zip(encoded["input_ids"], encoded["offset_mapping"], exclude):
        keep = [i for i, (a, b) in zip(ids, offs) if span is None or not (a < span[1] and b > span[0])]
        out.append(wte[keep or ids].mean(0))
    return np.stack(out)


class ContextGenerator(nn.Module):
    """CoLLEGe-style generator: pooled context features of a few examples → a row."""

    def __init__(self, dimension: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_generator(features: dict[int, np.ndarray], targets: dict[int, np.ndarray], *, steps: int, seed: int,
                    batch: int = 64) -> ContextGenerator:
    """Episodes: a random subset of a seen entry's example contexts is averaged and mapped to its row
    (loss 1 − cos); features are standardized inside the module's first layer by a fixed affine."""
    entries = sorted(set(features) & set(targets))
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dimension = next(iter(features.values())).shape[1]
    all_features = np.concatenate([features[e] for e in entries])
    mean, std = all_features.mean(0), all_features.std(0) + 1e-6
    model = ContextGenerator(dimension, hidden=min(512, 2 * dimension))
    model.register_buffer("mean", torch.tensor(mean, dtype=torch.float32)); model.register_buffer("std", torch.tensor(std, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stacked = {e: torch.tensor(features[e], dtype=torch.float32) for e in entries}
    y = {e: torch.tensor(targets[e], dtype=torch.float32) for e in entries}
    for _ in range(steps):
        pick = torch.randint(len(entries), (batch,), generator=generator).tolist()
        xs, ys = [], []
        for i in pick:
            f = stacked[entries[i]]
            size = int(torch.randint(1, f.shape[0] + 1, (1,), generator=generator))
            rows = torch.randperm(f.shape[0], generator=generator)[:size]
            xs.append(f[rows].mean(0)); ys.append(y[entries[i]])
        x, target = (torch.stack(xs) - model.mean) / model.std, torch.stack(ys)
        loss = (1 - F.cosine_similarity(model(x), target)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model.eval()


def definition_lookup(manifest: dict[str, Any], items_dir: Path, run: E5Run, wordnet: Any = None) -> Callable[[int], str | None] | None:
    """Definitions of seen entries for the text-evidence maps (never placed in prompts)."""
    sidecar = Path(items_dir) / "definitions.jsonl"
    if sidecar.exists():
        table = {int(r["entry"]): r["definition"] for r in map(json.loads, sidecar.read_text().splitlines()) if r}
        return table.get
    source = manifest.get("definitions")
    if source == "wordnet" and run.ontology is not None and run.ontology.get("concept_names") is not None:
        if wordnet is None:
            from nltk.corpus import wordnet
        names = run.ontology["concept_names"]

        def gloss(entry: int) -> str | None:
            texts = []
            for c in run.table.entry_concepts[entry]:
                try:
                    texts.append(wordnet.synset(names[c]).definition())
                except Exception:
                    pass
            return "; ".join(texts) or None
        return gloss
    if source == "devtools" and manifest.get("benchmark"):
        from ..benchmarks.devtools import _describe
        library = json.loads((Path(manifest["benchmark"]) / "library.json").read_text())
        by_alias = {normalize_alias(s["name"]): s for s in library["symbols"]}
        entry_symbol = {e: by_alias[a] for a, e in run.table.alias_to_entry.items() if a in by_alias}
        return lambda entry: _describe(entry_symbol[entry], random.Random(0)) if entry in entry_symbol else None
    return None


def baseline_rows(run: E5Run, adapter: cp.ChannelModelAdapter, concepts: Sequence[dict[str, Any]], entries: dict[str, int],
                  sources: Sequence[str], *, manifest: dict[str, Any], items_dir: Path, fit_entries: int, contexts: int,
                  seed: int, generator_steps: int = 1500, wordnet: Any = None, log: Callable[[str], None] = print
                  ) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    """Rows of the reserved entries under each baseline source (`own` is not overridden)."""
    channel = run.channel
    info: dict[str, Any] = {}
    out: dict[str, dict[int, torch.Tensor]] = {}
    reserved = sorted(set(entries.values()))
    surface_of = {entries[c["concept"]]: c["surface"] for c in concepts if c["concept"] in entries}
    width = channel.gate.in_features // 2
    if "none" in sources:
        out["none"] = {e: torch.zeros(width) for e in reserved}
    wanted = [s for s in sources if s not in {"own", "none"}]
    if not wanted:
        return out, info
    frequency = np.asarray(run.ontology["train_frequency"]) if run.ontology is not None else None
    heldout = set(run.adapter.heldout_entries) | set(reserved)
    rng = np.random.default_rng(seed)
    seen = np.asarray([e for e in np.flatnonzero(frequency >= 10) if e not in heldout], dtype=np.int64) if frequency is not None else np.zeros(0, int)
    fit = np.sort(rng.choice(seen, size=min(fit_entries, len(seen)), replace=False)) if len(seen) else seen
    fit_rows = entry_rows(channel, torch.as_tensor(fit)).numpy() if len(fit) else np.zeros((0, width))
    norm = float(np.linalg.norm(fit_rows, axis=1).mean()) if len(fit) else 1.0
    info.update(fit_entries=int(len(fit)), mean_row_norm=norm)
    wte = run.model.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    targets = dict(zip(fit.tolist(), fit_rows))
    if "random" in sources:
        out["random"] = {e: _scaled(np.random.default_rng([seed, e]).standard_normal(width), norm) for e in reserved}
    if "mean_row" in sources and len(fit):
        mean = torch.tensor(fit_rows.mean(0), dtype=torch.float32)
        out["mean_row"] = {e: mean for e in reserved}
    if "surface_mean" in sources:
        vectors = mean_embedding(run.tokenizer, wte, [" " + surface_of[e] for e in reserved], [None] * len(reserved))
        out["surface_mean"] = {e: _scaled(v, norm) for e, v in zip(reserved, vectors)}
    if "graph_projection" in sources and run.ontology is not None and len(fit) >= 3:
        embed = graph_features(run.ontology, fit)
        predict, alpha = ridge_map(embed(fit), fit_rows)
        out["graph_projection"] = {e: _scaled(v, norm) for e, v in zip(reserved, predict(embed(reserved)))}
        info["graph_projection_alpha"] = alpha
    definitions = definition_lookup(manifest, items_dir, run, wordnet)
    reserved_definitions = {entries[c["concept"]]: c.get("definition") for c in concepts if c["concept"] in entries}
    if definitions is not None and {"definition_mean", "definition_encoder"} & set(sources) and len(fit):
        fit_defs = [(e, definitions(int(e))) for e in fit]
        fit_defs = [(e, d) for e, d in fit_defs if d]
        have = [e for e in reserved if reserved_definitions.get(e)]
        info["definition_fit_entries"] = len(fit_defs); info["definition_reserved"] = len(have)
        if "definition_mean" in sources and have:
            vectors = mean_embedding(run.tokenizer, wte, [reserved_definitions[e] for e in have], [None] * len(have))
            out["definition_mean"] = {e: _scaled(v, norm) for e, v in zip(have, vectors)}
        if "definition_encoder" in sources and have and len(fit_defs) >= 3:
            x_fit = pooled_states(adapter, channel, [d for _, d in fit_defs], [None] * len(fit_defs), heldout)
            predict, alpha = ridge_map(x_fit, np.stack([targets[int(e)] for e, _ in fit_defs]))
            x_new = pooled_states(adapter, channel, [reserved_definitions[e] for e in have], [None] * len(have), heldout)
            out["definition_encoder"] = {e: _scaled(v, norm) for e, v in zip(have, predict(x_new))}
            info["definition_encoder_alpha"] = alpha
    elif {"definition_mean", "definition_encoder"} & set(sources):
        info["definitions"] = "not available"
    if {"alacarte", "college"} & set(sources) and len(fit) and run.config["data"].get("eval"):
        replace = {entries[c["concept"]]: c["surface"] for c in concepts if c.get("synthetic") and c["concept"] in entries}
        log(f"  example contexts for {len(fit)} fit + {len(reserved)} reserved entries")
        fit_snippets = context_snippets(run, fit.tolist(), k=contexts, parity=1)
        new_snippets = context_snippets(run, reserved, k=contexts, parity=1, replace=replace)
        info["context_fit_entries"] = len(fit_snippets); info["context_reserved"] = len(new_snippets)
        flat = lambda snippets: [(e, t, (a, b)) for e, rows in sorted(snippets.items()) for t, a, b in rows]
        fit_flat, new_flat = flat(fit_snippets), flat(new_snippets)
        if "alacarte" in sources and fit_flat and new_flat:
            def averaged(rows):
                vectors = mean_embedding(run.tokenizer, wte, [t for _, t, _ in rows], [s for _, _, s in rows])
                grouped = defaultdict(list)
                for (e, _, _), v in zip(rows, vectors):
                    grouped[e].append(v)
                return {e: np.mean(v, 0) for e, v in grouped.items()}
            fit_avg, new_avg = averaged(fit_flat), averaged(new_flat)
            keys = sorted(fit_avg)
            predict, alpha = ridge_map(np.stack([fit_avg[e] for e in keys]), np.stack([targets[e] for e in keys]))
            out["alacarte"] = {e: _scaled(predict(v[None])[0], norm) for e, v in new_avg.items()}
            info["alacarte_alpha"] = alpha
        if "college" in sources and fit_flat and new_flat:
            def features(rows, drop):
                pooled = pooled_states(adapter, channel, [t for _, t, _ in rows], [s for _, _, s in rows], drop)
                grouped = defaultdict(list)
                for (e, _, _), v in zip(rows, pooled):
                    grouped[e].append(v)
                return {e: np.stack(v) for e, v in grouped.items()}
            fit_features = features(fit_flat, set(fit.tolist()) | heldout)
            new_features = features(new_flat, heldout)
            model = train_generator(fit_features, targets, steps=generator_steps, seed=seed)
            with torch.no_grad():
                out["college"] = {e: _scaled(model((torch.tensor(f.mean(0), dtype=torch.float32) - model.mean) / model.std).numpy(), norm)
                                  for e, f in new_features.items()}
    # A baseline with no evidence for a reserved entry (no definition, no example context) falls back
    # to the mean trained row (the C2 rule), never to the candidate's own row; coverage is recorded.
    if len(fit):
        mean = torch.tensor(fit_rows.mean(0), dtype=torch.float32)
        for source, rows in out.items():
            missing = [e for e in reserved if e not in rows]
            info[f"{source}_coverage"] = 1 - len(missing) / max(1, len(reserved))
            for e in missing:
                rows[e] = mean
    return out, info


# -- evaluation: corpus tests ----------------------------------------------------------------------------------------

@torch.no_grad()
def corpus_occurrences(run: E5Run, reserved: set[int], *, windows: int | None) -> list[tuple[int, dict[str, torch.Tensor], torch.Tensor, list[int]]]:
    """Evaluation windows and, per window, the spans of reserved entries in even documents. `windows`
    = None tiles the whole evaluation corpus with non-overlapping windows (the default); a number
    takes the trainer's evenly spread evaluation windows."""
    from .e5_common import document_index, eos_positions
    corpus = TokenCorpus.open(Path(run.config["data"]["eval"]))
    seq_len = int(run.config["model"]["seq_len"])
    starts = eval_windows(corpus, count=windows or max(1, len(corpus) // seq_len), length=seq_len)
    boundaries = eos_positions(corpus)
    out = []
    for start in starts:
        ids, spans = corpus.window(start, seq_len, min_subtokens=run.min_subtokens)
        hits = [i for i, entry in enumerate(spans["entry"].tolist()) if entry in reserved]
        if not hits:
            continue
        docs = document_index(corpus, start + spans["start"][hits], boundaries)
        hits = [h for h, d in zip(hits, docs.tolist()) if d % 2 == 0]
        if hits:
            batch_ids, batch_spans = collate_windows([(ids, spans)])
            out.append((start, batch_spans, batch_ids, hits))
    return out


@torch.no_grad()
def corpus_tests(run: E5Run, occurrences, rows: dict[int, torch.Tensor] | None, *, keys: torch.Tensor | None) -> list[dict[str, Any]]:
    """After-loss and (with a semantic head and `keys` = all entries' rows) semantic rank per occurrence."""
    channel = run.channel
    results = []
    head = getattr(channel, "semantic_head", None) if channel is not None else None
    for start, spans, ids, hits in occurrences:
        with override_rows(channel, rows) if channel is not None else _null():
            losses, hidden = forward_tokens(run, ids, spans)
        for h in hits:
            entry, s, e = int(spans["entry"][h]), int(spans["start"][h]), int(spans["end"][h])
            lo, hi = e, min(ids.shape[1] - 1, e + AFTER_WINDOW)
            record = {"id": f"{start}:{e}", "entry": entry, "start": s,
                      "after_loss": float(losses[0, lo:hi].mean()) if hi > lo else None}
            if head is not None and keys is not None and s > 0:
                query = F.normalize(head(hidden[0, s - 1].float().to(next(head.parameters()).device)).float().cpu(), dim=-1)
                scores = keys @ query
                gold = scores[entry]
                others = torch.cat([scores[:entry], scores[entry + 1:]])
                record["rank"] = float(1 + (others > gold).sum() + 0.5 * (others == gold).sum())
            results.append(record)
    return results


# -- evaluate ---------------------------------------------------------------------------------------------------------

def available_sources(run: E5Run, requested: Sequence[str] | None) -> list[str]:
    mode = run.mode
    if mode in {"none", "hashed", "random"}:
        allowed = ["own"] + (["none"] if mode == "random" else [])
    else:
        allowed = list(ALL_SOURCES)
    return [s for s in (requested or allowed) if s in allowed]


def evaluate(run: E5Run, items_dir: Path, *, sources: Sequence[str] | None = None, fit_entries: int = 4000, contexts: int = 4,
             windows: int | None = None, seed: int = 0, generator_steps: int = 1500, wordnet: Any = None,
             log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest, concepts, items = load_items(items_dir)
    adapter = scenario_adapter(run, concepts)
    heldout = set(run.adapter.heldout_entries)
    frequency = run.adapter.train_frequency
    resolved = resolve_entries(adapter, concepts, items, heldout, frequency)
    entries = {cid: info["entry"] for cid, info in resolved.items() if info["linked"] and info["status"] != "mislinked"}
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    sources = available_sources(run, sources)
    started = time.monotonic()
    rows_by_source: dict[str, dict[int, torch.Tensor]] = {}
    info: dict[str, Any] = {}
    if run.channel is not None and entries:
        rows_by_source, info = baseline_rows(run, adapter, concepts, entries, sources, manifest=manifest, items_dir=items_dir,
                                             fit_entries=fit_entries, contexts=contexts, seed=seed,
                                             generator_steps=generator_steps, wordnet=wordnet, log=log)
    sources = [s for s in sources if s == "own" or s in rows_by_source]
    per_source: dict[str, dict[str, Any]] = {}
    null_scores = None
    reserved = set(entries.values())
    occurrences = []
    keys_own = None
    if manifest.get("corpus_tests") and reserved and run.config["data"].get("eval"):
        occurrences = corpus_occurrences(run, reserved, windows=windows)
        if run.channel is not None and getattr(run.channel, "semantic_head", None) is not None and run.mode != "hashed":
            entry_count = int(run.ontology["entry_count"])
            keys_own = entry_rows(run.channel, torch.arange(entry_count))
    for source in sources:
        log(f"  source {source}")
        rows = rows_by_source.get(source)
        with override_rows(run.channel, rows) if run.channel is not None else _null():
            prompt_results, null_scores = score_prompts(adapter, items, surfaces, null_scores)
        keys = None
        if keys_own is not None:
            keys = keys_own.clone()
            for entry, vector in (rows or {}).items():
                keys[entry] = vector
            keys = F.normalize(keys, dim=-1)
        corpus_results = corpus_tests(run, occurrences, rows, keys=keys) if occurrences else []
        per_source[source] = {"prompts": prompt_results, "corpus": corpus_results}
    return {"manifest": manifest, "resolved": resolved, "sources": sources, "results": per_source, "info": info,
            "seconds": time.monotonic() - started}


def paired_difference(a: np.ndarray, b: np.ndarray, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Mean of `a − b` with a percentile bootstrap interval over items and a two-sided bootstrap p."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    draws = diffs[np.random.default_rng(seed).integers(0, len(diffs), size=(resamples, len(diffs)))].mean(1)
    p = float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (resamples + 1)))
    if np.all(diffs == 0):
        p = 1.0
    return {"mean": float(diffs.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "n": int(len(diffs)), "p_value": p}


def _subset(results: list[dict[str, Any]], test: str, keep: set[str]) -> list[dict[str, Any]]:
    return [r for r in results if r["test"] == test and r["concept"] in keep]


def summarize(evaluation: dict[str, Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    resolved = evaluation["resolved"]
    linked = {c for c, info in resolved.items() if info["linked"] and info["status"] != "mislinked"}
    everything = set(resolved)
    first = evaluation["results"][evaluation["sources"][0]]["prompts"] if evaluation["sources"] else []
    chance = {test: float(np.mean([1.0 / len(r["pmi"][0]) for r in first if r["test"] == test and r["concept"] in linked]))
              for test in ("property", "entailment") if any(r["test"] == test and r["concept"] in linked for r in first)}
    statuses = Counter(info["status"] for info in resolved.values())
    out: dict[str, Any] = {"concepts": len(resolved), "linked_concepts": len(linked), "status_counts": dict(sorted(statuses.items())),
                           "chance": chance, "sources": {}, "comparisons": []}

    def metric_vectors(source: str, subset: set[str]) -> dict[str, tuple[list[str], np.ndarray]]:
        result = evaluation["results"][source]
        vectors = {}
        for test in ("property", "entailment"):
            rows = _subset(result["prompts"], test, subset)
            vectors[test] = ([r["id"] for r in rows], np.asarray([r["correct"] for r in rows], dtype=float))
        rows = [r for r in _subset(result["prompts"], "property", subset) if r["consistent"] is not None]
        vectors["paraphrase"] = ([r["id"] for r in rows], np.asarray([r["consistent"] for r in rows], dtype=float))
        corpus = result["corpus"]
        losses = [r for r in corpus if r["after_loss"] is not None]
        vectors["after_loss"] = ([r["id"] for r in losses], np.asarray([r["after_loss"] for r in losses], dtype=float))
        ranks = [r for r in corpus if "rank" in r]
        vectors["semantic_mrr"] = ([r["id"] for r in ranks], np.asarray([1.0 / r["rank"] for r in ranks], dtype=float))
        vectors["semantic_r10"] = ([r["id"] for r in ranks], np.asarray([float(r["rank"] <= 10) for r in ranks], dtype=float))
        return vectors

    for subset_name, subset in (("linked", linked), ("all", everything)):
        for source in evaluation["sources"]:
            vectors = metric_vectors(source, subset)
            out["sources"].setdefault(source, {})[subset_name] = {
                test: {"mean": float(v.mean()) if v.size else None, "n": int(v.size)} for test, (_, v) in vectors.items()}
    reference = "own"
    if reference in evaluation["sources"]:
        ref = metric_vectors(reference, linked)
        for test in ref:
            block = []
            for source in evaluation["sources"]:
                if source == reference:
                    continue
                ids_b, vb = metric_vectors(source, linked)[test]
                ids_a, va = ref[test]
                if not va.size or ids_a != ids_b:
                    continue
                ci = paired_difference(va, vb, resamples=resamples, seed=seed)
                lower_is_better = test == "after_loss"
                block.append({"test": test, "baseline": source, "family": "text_evidence" if source in TEXT_SOURCES else "structure",
                              "difference": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": ci["n"],
                              "p_value": ci["p_value"], "better": (ci["mean"] < 0) if lower_is_better else (ci["mean"] > 0)})
            for family in ("structure", "text_evidence"):
                rows = [b for b in block if b["family"] == family]
                for row, adjusted in zip(rows, holm_adjust([r["p_value"] for r in rows]) if rows else []):
                    row["p_holm"] = adjusted
                    row["significant"] = adjusted < 0.05
            out["comparisons"] += block
    return out


def render(summary: dict[str, Any], header: dict[str, Any]) -> str:
    source = header["source"]
    manifest = header["manifest"]
    quantized = f", {source['quantization']['variant']}" if source.get("quantization") else ""
    lines = [f"# E5.4 zero-shot insertion — {manifest['scenario']} — {source['condition']} seed {source['seed']} ({source['size']}{quantized})", "",
             f"Run `{source['run']}` (channel `{source['channel_mode']}`); items `{header['items']}` "
             f"({manifest['counts']['concepts']} concepts, {manifest['counts']['items']} prompt items; contamination-free on "
             f"pretrained hosts: **{manifest['contamination_free']}**). Linked concepts: {summary['linked_concepts']} of "
             f"{summary['concepts']} ({summary['status_counts']}).", "",
             "`own` is the run's own row (composition: the structure-only candidate; C2: the mean-vector fallback; C0: no channel).", ""]
    tests = ("property", "entailment", "paraphrase", "semantic_mrr", "semantic_r10", "after_loss")
    for family, names in (("Structure-only sources", STRUCTURE_SOURCES), ("Text-evidence sources (reported separately)", TEXT_SOURCES)):
        present = [s for s in names if s in summary["sources"]]
        if not present:
            continue
        lines += [f"## {family}", "", "| source | " + " | ".join(tests) + " |", "|---|" + "---:|" * len(tests)]
        for s in present:
            linked = summary["sources"][s]["linked"]
            lines.append(f"| {s} | " + " | ".join(fmt(linked[t]["mean"], 4) for t in tests) + " |")
        lines.append("")
    if summary["comparisons"]:
        lines += ["## `own` − baseline (linked concepts; paired bootstrap; Holm within family and test)", "",
                  "| test | baseline | family | difference [95% CI] | n | p (Holm) | own better |", "|---|---|---|---|---:|---:|---|"]
        for c in summary["comparisons"]:
            lines.append(f"| {c['test']} | {c['baseline']} | {c['family']} | {fmt_ci({'mean': c['difference'], 'ci_low': c['ci_low'], 'ci_high': c['ci_high']})} | "
                         f"{c['n']} | {fmt(c.get('p_holm'), 4)} | {'yes' if c.get('significant') and c['better'] else 'no'} |")
        lines.append("")
    lines += [f"Chance: {', '.join(f'{t} {fmt(v, 3)}' for t, v in summary.get('chance', {}).items())}.", "",
              "Property/entailment: accuracy averaged over template paraphrases (PMI against the null surface); paraphrase: argmax "
              "agreement across paraphrases; semantic rank among all entries' rows; after_loss in nats (lower is better).", ""]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    requested = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    config = {"experiment": "e5.4-zero-shot", "run": str(args.run), "checkpoint": args.checkpoint, "items": str(args.items),
              "sources": requested, "fit_entries": args.fit_entries, "contexts": args.contexts, "windows": args.windows,
              "seed": args.seed, "resamples": args.resamples, "generator_steps": args.generator_steps}
    quantize = getattr(args, "quantize", None)
    if quantize:                               # recorded only when used (earlier configs resolve as before)
        config.update(quantize=quantize, quantize_channel=bool(args.quantize_channel), group_size=args.group_size)
    if getattr(args, "overwrite", False):
        clear_output(args.output)
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, quantize=quantize,
                   quantize_channel=bool(getattr(args, "quantize_channel", False)), group_size=getattr(args, "group_size", "auto"))
    evaluation = evaluate(run, args.items, sources=requested, fit_entries=args.fit_entries, contexts=args.contexts,
                          windows=args.windows, seed=args.seed, generator_steps=args.generator_steps)
    summary = summarize(evaluation, resamples=args.resamples, seed=args.seed)
    header = {"source": run.describe(), "items": str(args.items), "manifest": evaluation["manifest"]}
    with (args.output / "predictions.jsonl").open("w") as handle:
        for source, result in evaluation["results"].items():
            for row in result["prompts"]:
                handle.write(json.dumps({"source": source, **{k: v for k, v in row.items() if k != "pmi"}}) + "\n")
            for row in result["corpus"]:
                handle.write(json.dumps({"source": source, "test": "corpus", **row}) + "\n")
    write_json(args.output / "summary.json", {**header, "resolved": evaluation["resolved"], "baselines": evaluation["info"],
                                              "summary": summary, "seconds": evaluation["seconds"]})
    (args.output / "report.md").write_text(render(summary, header))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return {**header, "summary": summary}


def run_items(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")
    if args.scenario in {"c3_heldout", "c3_synthetic"}:
        texts: list[str] = []
        if args.scenario == "c3_synthetic" and args.contamination_corpus:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            for path in args.contamination_corpus:
                corpus = TokenCorpus.open(path)
                limit = min(len(corpus), args.contamination_tokens)
                for start in range(0, limit, 1_000_000):
                    texts.append(tokenizer.decode(np.asarray(corpus.tokens[start:min(limit, start + 1_000_000)]).tolist()))
        return SCENARIOS[args.scenario](ontology_path=args.ontology, out_dir=args.output, tokenizer_name=args.tokenizer,
                                        min_subtokens=args.min_subtokens, max_concepts=args.max_concepts, seed=args.seed,
                                        contamination_texts=texts)
    if args.scenario == "c6_devtools":
        return build_c6_items(args.benchmark, args.output, tokenizer_name=args.tokenizer, min_subtokens=args.min_subtokens,
                              seed=args.seed)
    return SCENARIOS[args.scenario](out_dir=args.output, seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("items", help="build an item directory for a scenario")
    build.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ontology", type=Path, default=Path("~/data/vsa-llm/c3/wordnet-gpt2-v1/ontology.pt").expanduser())
    build.add_argument("--benchmark", type=Path, default=Path("experiments/c6-devtools-benchmark/v1"))
    build.add_argument("--tokenizer", default="gpt2"); build.add_argument("--min-subtokens", type=int, default=2)
    build.add_argument("--max-concepts", type=int, default=400); build.add_argument("--seed", type=int, default=0)
    build.add_argument("--contamination-corpus", type=Path, nargs="*", default=[],
                       help="token corpora whose first --contamination-tokens tokens must not contain an invented name")
    build.add_argument("--contamination-tokens", type=int, default=20_000_000)
    ev = sub.add_parser("evaluate", help="evaluate a trained run on an item directory")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--items", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True); ev.add_argument("--checkpoint", default="final.pt")
    ev.add_argument("--sources", default="", help="comma-separated subset of " + ",".join(ALL_SOURCES))
    ev.add_argument("--fit-entries", type=int, default=4000); ev.add_argument("--contexts", type=int, default=4)
    ev.add_argument("--windows", type=int, default=None,
                    help="evaluation windows for corpus tests (default: tile the whole evaluation corpus)")
    ev.add_argument("--generator-steps", type=int, default=1500)
    ev.add_argument("--seed", type=int, default=0); ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--batch-size", type=int, default=64); ev.add_argument("--device", default=None)
    ev.add_argument("--quantize", choices=["int8", "int4"], default=None,
                    help="evaluate after post-training weight-only quantization as in e4_quant (int4 needs CUDA)")
    ev.add_argument("--quantize-channel", action="store_true", help="with --quantize: variant B (channel quantized too)")
    ev.add_argument("--group-size", default="auto", help="INT4 group size, or auto (as e4_quant)")
    ev.add_argument("--overwrite", action="store_true", help="replace the result files of a previous evaluation in --output")
    args = parser.parse_args(argv)
    if args.command == "evaluate" and args.quantize_channel and not args.quantize:
        parser.error("--quantize-channel needs --quantize")
    if args.command == "items":
        print(json.dumps(run_items(args), indent=2, default=str))
    else:
        result = run_evaluate(args)
        print(json.dumps({s: v["linked"] for s, v in result["summary"]["sources"].items()}, indent=2))


if __name__ == "__main__":
    main()
