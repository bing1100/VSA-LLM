"""E9 dimension 3: zero-shot learning by editing the ontology — after training, with no weight update.

A trained model reads concept rows that the span channel composes from its dictionary (atomics,
relations, the composition and projection it learned). Editing the ontology therefore changes
what the model is told about a word without touching a weight. Two tests (execution.md, E9):

**(a) New words.** `count` (default 300) new concepts with invented names (pronounceable,
≥ ℓ_min subtokens under the host tokenizer, not a WordNet lemma, not a word of any alias, not a
vocabulary token, absent from a sample of the training corpus and from the E5.4 synthetic names)
get frames that are *new combinations* of existing atomics and relations: the frame template of a
real concept (its donor) keeps the category edges (`hypernym` / `instance_hypernym`, `lexname`,
`pos`) and every other edge's filler is resampled from the fillers that relation takes for that part
of speech (frequency-weighted), so that no existing entry carries the category edge together with
any resampled edge (no concept of the ontology states the combination). Donors are drawn uniformly
from the eligible entries (single-concept nouns/verbs with ≥ 1 non-category edge that has a
template), so the degree distribution is that of the ontology's content-bearing frames; category-only
frames (lexname, pos, hypernym: 45% of WordNet entries) cannot form a new combination and are not
donors. At evaluation the new names enter the alias table and the new frames the composer's schedule
(`SpanChannel.add_entries`), and the channel composes their rows from the trained atomic and relation
vectors; no other entry's row changes. Tests (the E5.4 item machinery and templates):
`property` (PMI argmax among the gold filler and 4 distractors; `property_new` = the resampled
edges, i.e. the new combinations), `entailment` (an ancestor 2–3 levels up vs a non-ancestor),
`paraphrase` (argmax agreement across template paraphrases) and a loss test, `statement`: sentences
that state each frame fact in held-out wordings (not the property templates), scoring the
log-probability of the correct filler against the distractor fillers after the word (per-token
log-probability; accuracy, margin and the gold filler's loss in nats/token).
Sources (rows the channel injects for the new entries): `own` (the frame composed — the candidate;
for C2 the free table's fallback row; for C0′/P0 nothing), `none` (name only, the channel sees
nothing: what C0′/P0 can do), `mean_row` (mean trained row, the C2 rule inside this model) and
`random_frame` (a random frame of equal degree: the same relations, every filler — category edges
included — drawn frequency-weighted from that relation's fillers for the part of speech, i.e. a
plausible but wrong frame, composed like the candidate).

**(b) Edited words.** For `count` (default 200) existing concepts (half held out, half seen in
training) one relation–filler edge is changed: the `hypernym` (or `instance_hypernym`) filler is
replaced by a plausible alternative of the same type — a filler of the same relation with the same
WordNet lexicographer file, neither an ancestor of the concept nor a descendant of the old filler.
The metrics follow knowledge-editing evaluations (ROME, Meng et al. 2022, "Locating and Editing
Factual Associations in GPT"; MEMIT, Meng et al. 2023, "Mass-Editing Memory in a Transformer"),
with `d = log p(new | prompt) − log p(old | prompt)` per template:

- efficacy: `ES` = share of edit prompts (the property templates) with `d > 0` after the edit, and
  the magnitude `EM` = mean change of `d` from before to after;
- generalization: `PS` / `PM`, the same on paraphrases (the held-out statement wordings);
- specificity: `NS` = share of neighbourhood prompts (unedited concepts that share the old filler
  where possible, else other linked concepts, against the edited concept's new filler) that still
  prefer their true filler after the edit, and the neighbourhood change of `d` (0 when rows are
  independent, as composed rows are: a concept's row depends on its own frame only — exactly 0 on the
  CPU; on CUDA, where the composer's `index_add_` sums by atomics in no fixed order and bf16 rounds
  the result, repeated evaluations differ by up to ≈ 0.01 nats, which is noise, not leakage);
- `score` = harmonic mean of ES, PS and NS (as in ROME/MEMIT);
- a control edit to a different random filler of the same pool: `EM − EM_control` isolates the move
  toward the *specified* filler from the effect of removing the old one.

All edits are applied at once (rows are independent, so this equals one edit at a time). Models
without composition (C2's free table, C0′, P0) cannot be edited: their `after` equals `before`, which
is reported as such (the contrast that matters for a frozen-weight edit).

Every evaluation can run on the quantized model (`--quantize int8|int4`, `--quantize-channel` =
variant B; as `e4_quant`, output head FP).

    python -m vsa_embed.experiments.e9_ontology_edit items --kind new --ontology ONT --tokenizer TOK --output DIR
        [--count 300] [--contamination-corpus TRAIN] [--reserved-names E5_ITEMS_DIR ...]
    python -m vsa_embed.experiments.e9_ontology_edit items --kind edits --ontology ONT --tokenizer TOK --output DIR [--count 200]
    python -m vsa_embed.experiments.e9_ontology_edit evaluate --run RUN --new-items DIR --edit-items DIR --output OUT
        [--quantize int4 [--quantize-channel]] [--sources own,none,mean_row,random_frame]

Item directories (committed under `experiments/e9-retrofit/items/`): `manifest.json`, `concepts.jsonl`
and `items.jsonl` in the E5.4 item format (`id`, `concept`, `test`, `relation`, `templates` with
`{x}`, `null`, `candidates` with a leading space, `gold`), frames stored by atomic and relation *name*
so that they resolve against any run of the same ontology.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from ..compose import FrameSchedule
from ..evaluation import channel_probes as cp
from ..span_channel import AliasTable, CausalLinker, SpanChannel, normalize_alias
from ..statistics import holm_adjust
from . import e5_zeroshot as zs
from .e5_common import (E5Run, canonical_surfaces, clear_output, finish_output, fmt, fmt_ci, json_ready, open_run,
                        override_rows, start_output, write_json)

SCHEMA_NEW = "e9-new-words/1"
SCHEMA_EDITS = "e9-edits/1"
CATEGORY_RELATIONS = ("hypernym", "instance_hypernym")
KEPT_RELATIONS = frozenset({*CATEGORY_RELATIONS, "lexname", "pos"})
NEW_SOURCES = ("own", "none", "mean_row", "random_frame")
EDIT_CONDITIONS = ("before", "after", "control")
ROOT = Path("experiments/e9-retrofit")

# Held-out wordings for the loss test (new words) and the edit paraphrases: one pair per property
# template of E5.4 (`zs.TEMPLATES`), never the property templates themselves.
STATEMENT_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("n", "hypernym"): ["Our guide explained that the {x} is a sort of", "Looked up in a dictionary, the {x} is classed as a kind of"],
    ("n", "instance_hypernym"): ["In the encyclopedia, {x} is listed under the heading", "Our guide described {x} as a particular"],
    ("n", "part_holonym"): ["You will find the {x} as one piece of the", "Our guide explained that the {x} belongs to the"],
    ("n", "member_holonym"): ["Our guide explained that every {x} is counted in the", "The {x} is listed among the members of the"],
    ("n", "substance_holonym"): ["Our guide explained that the {x} is found inside", "Chemists extract the {x} from"],
    ("n", "part_meronym"): ["Our guide explained that every {x} comes with a", "If you take apart a {x}, you will find a"],
    ("n", "member_meronym"): ["Our guide explained that the {x} includes the", "Among the members of the {x} is the"],
    ("n", "substance_meronym"): ["Our guide explained that the {x} consists of", "A chemist would say the {x} contains"],
    ("n", "topic_domain"): ["Our guide explained that the {x} is discussed in", "You would study the {x} as part of"],
    ("n", "lexname"): ["Our guide explained that the {x} is a matter of", "Dictionaries classify the {x} under"],
    ("v", "hypernym"): ["Our guide explained that to {x} is to", "When people {x}, what they really do is"],
    ("v", "entailment"): ["Our guide explained that whoever tries to {x} must", "Anyone who wants to {x} also needs to"],
    ("v", "cause"): ["Our guide explained that to {x} something makes it", "If you {x} something, it will"],
    ("v", "lexname"): ["Our guide explained that to {x} is a matter of", "Dictionaries classify the verb {x} under"],
}


# -- shared ontology views ---------------------------------------------------------------------------

@dataclasses.dataclass
class OntologyView:
    """Readable access to an ontology's entry frames (entries = link entries of the alias table)."""

    ontology: dict[str, Any]
    table: AliasTable

    def __post_init__(self) -> None:
        o = self.ontology
        self.offsets = np.asarray(o["offsets"], dtype=np.int64)
        self.relations = np.asarray(o["relations"], dtype=np.int64)
        self.fillers = np.asarray(o["fillers"], dtype=np.int64)
        self.relation_names: list[str] = list(o["relation_names"])
        self.atomic_names: list[str] = list(o["atomic_names"])
        self.relation_id = {name: i for i, name in enumerate(self.relation_names)}
        self.atomic_id = {name: i for i, name in enumerate(self.atomic_names)}
        names = o.get("concept_names")
        self.pos = [names[c[0]].rsplit(".", 2)[1] if names is not None and c else "?" for c in self.table.entry_concepts]
        self.heldout = {int(e) for e in o.get("heldout_entries", ())}
        self.frequency = np.asarray(o["train_frequency"]) if o.get("train_frequency") is not None else None

    @property
    def entry_count(self) -> int:
        return len(self.offsets) - 1

    def frame(self, entry: int) -> list[tuple[int, int]]:
        lo, hi = self.offsets[entry], self.offsets[entry + 1]
        return list(zip(self.relations[lo:hi].tolist(), self.fillers[lo:hi].tolist()))

    def readable(self, frame: Iterable[tuple[int, int]]) -> list[list[str]]:
        return [[self.relation_names[r], self.atomic_names[f]] for r, f in frame]

    def text(self, filler: int) -> str | None:
        return zs._filler_text(self.atomic_names[filler])

    def edge_index(self) -> dict[tuple[int, int], set[int]]:
        """(relation, filler) → entries whose frame holds that edge."""
        index: dict[tuple[int, int], set[int]] = defaultdict(set)
        rows = np.repeat(np.arange(self.entry_count), np.diff(self.offsets))
        for e, r, f in zip(rows.tolist(), self.relations.tolist(), self.fillers.tolist()):
            index[(r, f)].add(e)
        return index

    def filler_pools(self, *, by_pos: bool = True) -> dict[tuple[str, int], Counter]:
        """(part of speech or "*", relation) → filler usage counts over every entry's frame."""
        pools: dict[tuple[str, int], Counter] = defaultdict(Counter)
        rows = np.repeat(np.arange(self.entry_count), np.diff(self.offsets))
        for e, r, f in zip(rows.tolist(), self.relations.tolist(), self.fillers.tolist()):
            pools[(self.pos[e] if by_pos else "*", r)][f] += 1
        return pools


def resolve_frame(view_names: Sequence[Sequence[str]], relation_id: dict[str, int], atomic_id: dict[str, int]) -> list[tuple[int, int]]:
    """A frame stored by names → ids of a run's ontology (raises on an unknown name)."""
    try:
        return [(relation_id[r], atomic_id[a]) for r, a in view_names]
    except KeyError as missing:
        raise ValueError(f"the run's ontology has no {missing} (items were built for another ontology)") from None


def _weighted_choice(rng: random.Random, pool: Counter, exclude: set[int], accept: Callable[[int], bool] | None = None,
                     tries: int = 50) -> int | None:
    items = [(f, c) for f, c in pool.items() if f not in exclude]
    if not items:
        return None
    population, weights = zip(*sorted(items))
    for _ in range(tries):
        choice = rng.choices(population, weights)[0]
        if accept is None or accept(choice):
            return choice
    candidates = [f for f in population if accept is None or accept(f)]
    return rng.choice(candidates) if candidates else None


def _write_items(out_dir: Path, schema: str, manifest: dict[str, Any], concepts: list[dict[str, Any]],
                 items: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "concepts.jsonl").write_text("".join(json.dumps(json_ready(c)) + "\n" for c in concepts))
    (out_dir / "items.jsonl").write_text("".join(json.dumps(json_ready(i)) + "\n" for i in items))
    counts = Counter(item["test"] for item in items)
    manifest = {"schema": schema, **manifest, "counts": {"concepts": len(concepts), "items": len(items), **dict(sorted(counts.items()))}}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def load_item_dir(items_dir: Path, schema: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    items_dir = Path(items_dir)
    manifest = json.loads((items_dir / "manifest.json").read_text())
    if manifest.get("schema") != schema:
        raise ValueError(f"{items_dir} is not an {schema} item directory")
    read = lambda name: [json.loads(line) for line in (items_dir / name).read_text().splitlines() if line.strip()]
    concepts, items = read("concepts.jsonl"), read("items.jsonl")
    known = {c["concept"] for c in concepts}
    for item in items:
        if item["concept"] not in known:
            raise ValueError(f"item {item['id']} refers to unknown concept {item['concept']}")
        if not 0 <= int(item["gold"]) < len(item["candidates"]) or not item["templates"]:
            raise ValueError(f"item {item['id']} is malformed")
    return manifest, concepts, items


def _contamination_texts(paths: Sequence[Path], tokenizer_name: str, tokens: int) -> list[str]:
    """The first `tokens` tokens of each token corpus, decoded (the E5.4 contamination check)."""
    from transformers import AutoTokenizer
    from ..data.corpus import TokenCorpus
    if not paths:
        return []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    texts = []
    for path in paths:
        corpus = TokenCorpus.open(Path(path))
        limit = min(len(corpus), tokens)
        for start in range(0, limit, 1_000_000):
            texts.append(tokenizer.decode(np.asarray(corpus.tokens[start:min(limit, start + 1_000_000)]).tolist()))
    return texts


def _reserved_names(directories: Sequence[Path]) -> set[str]:
    names: set[str] = set()
    for directory in directories:
        path = Path(directory) / "concepts.jsonl"
        if path.exists():
            names |= {json.loads(line)["surface"].lower() for line in path.read_text().splitlines() if line.strip()}
    return names


# -- (a) new words: items ------------------------------------------------------------------------------

def _distractors(rng: random.Random, pool: Sequence[str], exclude: set[str], k: int) -> list[str] | None:
    options = [t for t in pool if t not in exclude]
    return rng.sample(options, k) if len(options) >= k else None


def build_new_word_items(ontology_path: Path, out_dir: Path, *, tokenizer_name: str, count: int = 300, distractors: int = 4,
                         seed: int = 0, min_subtokens: int = 2, wordnet: Any = None, alias_table: Path | None = None,
                         contamination_texts: Iterable[str] = (), reserved_names: Iterable[str] = (),
                         name_seed: int = 11) -> dict[str, Any]:
    """Item directory for (a): new concepts with invented names and new-combination frames."""
    from transformers import AutoTokenizer
    if wordnet is None:
        from nltk.corpus import wordnet
    ontology = torch.load(ontology_path, weights_only=False)
    table, table_info = cp.resolve_alias_table(ontology, Path(ontology_path), alias_table=alias_table)
    view = OntologyView(ontology, table)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    rng = random.Random(seed)
    index = view.edge_index()
    pools = view.filler_pools()
    category_ids = {view.relation_id[r] for r in CATEGORY_RELATIONS if r in view.relation_id}
    kept_ids = {view.relation_id[r] for r in KEPT_RELATIONS if r in view.relation_id}
    # Readable filler texts per (POS, relation name) for distractors (as E5.4).
    texts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (pos, r), counter in pools.items():
        name = view.relation_names[r]
        if (pos, name) in zs.TEMPLATES:
            texts[(pos, name)] |= {t for t in (view.text(f) for f in counter) if t}
    text_pools = {key: sorted(values) for key, values in texts.items()}

    def templated(pos: str, r: int, f: int) -> bool:
        return (pos, view.relation_names[r]) in zs.TEMPLATES and view.text(f) is not None

    eligible = []
    for e in range(view.entry_count):
        if len(table.entry_concepts[e]) != 1 or e in view.heldout or view.pos[e] not in {"n", "v"}:
            continue
        frame = view.frame(e)
        categories = [(r, f) for r, f in frame if r in category_ids and templated(view.pos[e], r, f)]
        others = [(r, f) for r, f in frame if r not in kept_ids]
        if categories and any((view.pos[e], view.relation_names[r]) in zs.TEMPLATES for r, _ in others):
            eligible.append(e)
    order = eligible[:]
    rng.shuffle(order)
    frames_seen = {frozenset(view.frame(e)) for e in range(view.entry_count)}
    built: list[dict[str, Any]] = []
    rejected = Counter()
    for donor in order:
        if len(built) >= count:
            break
        pos, frame = view.pos[donor], view.frame(donor)
        category = next((r, f) for r, f in frame if r in category_ids and templated(pos, r, f))
        with_category = index[category]
        new_frame: list[tuple[int, int]] = []
        resampled: list[tuple[int, int]] = []
        ok = True
        for r, f in frame:
            if r in kept_ids:
                new_frame.append((r, f)); continue
            current = {x for _, x in frame} | {x for _, x in new_frame}
            category_text = view.text(category[1])
            needs_text = (pos, view.relation_names[r]) in zs.TEMPLATES
            choice = _weighted_choice(
                rng, pools[(pos, r)], current | {f},
                lambda x, r=r, needs_text=needs_text: not (with_category & index.get((r, x), set()))
                and (not needs_text or (view.text(x) is not None and view.text(x) != category_text)))
            if choice is None:
                ok = False; break
            new_frame.append((r, choice)); resampled.append((r, choice))
        if not ok:
            rejected["no filler"] += 1; continue
        key = frozenset(new_frame)
        if key in frames_seen or not any(templated(pos, r, f) for r, f in resampled):
            rejected["combination exists"] += 1; continue
        frames_seen.add(key)
        random_frame = [(r, _weighted_choice(rng, pools[(pos, r)], set())) for r, _ in new_frame]
        built.append({"donor": donor, "pos": pos, "frame": new_frame, "category": category, "resampled": resampled,
                      "random_frame": random_frame})
    if len(built) < count:
        raise ValueError(f"only {len(built)} new-combination frames could be built (asked for {count})")
    reserved = {n.lower() for n in reserved_names}
    names, checks = zs.invent_names(list(range(count + 64)), tokenizer, table, wordnet, min_subtokens=min_subtokens,
                                    seed=name_seed, contamination_texts=contamination_texts)
    surfaces = [n for _, n in sorted(names.items()) if n.lower() not in reserved][:count]
    if len(surfaces) < count:
        raise ValueError("not enough invented names after removing reserved ones")
    checks.update(reserved_names=len(reserved), reserved_clashes=sum(n.lower() in reserved for n in names.values()))
    concept_names = ontology.get("concept_names")
    concepts, items = [], []
    noun_pool, verb_pool = text_pools.get(("n", "hypernym"), []), text_pools.get(("v", "hypernym"), [])
    for i, (spec, surface) in enumerate(zip(built, surfaces)):
        cid = f"e9n-{i:04d}"
        pos, category = spec["pos"], spec["category"]
        kind, _, category_synset = view.atomic_names[category[1]].partition(":")
        ancestors = zs._ancestors(wordnet.synset(category_synset)) if kind == "synset" else {}
        ancestor_texts = {zs._first_lemma(a) for a in ancestors} | {view.text(category[1])}
        gold: dict[str, list[str]] = defaultdict(list)
        kinds: dict[str, str] = {}
        for r, f in spec["frame"]:
            name, text = view.relation_names[r], view.text(f)
            if text and (pos, name) in zs.TEMPLATES:
                gold[name].append(text)
                kinds.setdefault(name, "resampled" if (r, f) in spec["resampled"] else "category")
        concepts.append({"concept": cid, "surface": surface, "entry": None, "synthetic": True, "pos": pos,
                         "donor_entry": spec["donor"],
                         "donor_concept": concept_names[table.entry_concepts[spec["donor"]][0]] if concept_names else None,
                         "category": view.atomic_names[category[1]], "frame": view.readable(spec["frame"]),
                         "resampled": view.readable(spec["resampled"]), "random_frame": view.readable(spec["random_frame"]),
                         "degree": len(spec["frame"]), "gold": dict(gold), "definition": None})
        for relation in sorted(gold):
            right = gold[relation][0]
            exclude = set(gold[relation]) | {surface} | (ancestor_texts if relation in {"hypernym", "instance_hypernym", "lexname"} else set())
            for test, templates in (("property", zs.TEMPLATES[(pos, relation)]), ("statement", STATEMENT_TEMPLATES[(pos, relation)])):
                wrong = _distractors(rng, text_pools.get((pos, relation), []), exclude, distractors)
                if wrong is None:
                    continue
                candidates = [right] + wrong
                rng.shuffle(candidates)
                items.append({"id": f"{cid}-{test}-{relation}", "concept": cid, "test": test, "relation": relation,
                              "edge_kind": kinds[relation], "templates": templates, "null": zs.NULL_SURFACE,
                              "candidates": [" " + c for c in candidates], "gold": candidates.index(right)})
        deep = sorted(a for a, d in ancestors.items() if d in (1, 2))      # 2–3 levels above the new concept
        for rank, ancestor in enumerate(rng.sample(deep, min(2, len(deep)))):
            right = zs._first_lemma(ancestor)
            negatives = [t for t in (noun_pool if pos == "n" else verb_pool) if t not in ancestor_texts and t != right]
            if not negatives:
                continue
            candidates = [right, rng.choice(negatives)]
            rng.shuffle(candidates)
            items.append({"id": f"{cid}-entailment-{rank}", "concept": cid, "test": "entailment",
                          "relation": f"ancestor_depth_{ancestors[ancestor] + 1}", "edge_kind": "category",
                          "templates": zs.ENTAILMENT_TEMPLATES[pos], "null": zs.NULL_SURFACE,
                          "candidates": [" " + c for c in candidates], "gold": candidates.index(right)})
    degrees = Counter(len(s["frame"]) for s in built)
    donor_degrees = Counter(len(view.frame(e)) for e in eligible)
    manifest = {"scenario": "e9_new_words", "contamination_free": True,
                "description": "new concepts with invented names and frames that are new combinations of existing atomics "
                               "and relations (zero-shot insertion by ontology editing)",
                "ontology": str(ontology_path), "ontology_alias_sha256": table.digest(), "alias_table": table_info,
                "tokenizer": tokenizer_name, "min_subtokens": min_subtokens, "seed": seed, "name_seed": name_seed,
                "count": count, "distractors": distractors, "null_surface": zs.NULL_SURFACE, "checks": checks,
                "construction": {"donors_eligible": len(eligible), "donors_tried": sum(rejected.values()) + len(built),
                                 "rejected": dict(rejected), "kept_relations": sorted(KEPT_RELATIONS),
                                 "rule": "category edges of a real donor kept; every other filler resampled (frequency-weighted, "
                                         "same relation and part of speech) so that no existing entry has the category edge "
                                         "together with any resampled edge; donors are not held out",
                                 "degree_histogram": dict(sorted(degrees.items())),
                                 "eligible_donor_degree_histogram": dict(sorted(donor_degrees.items()))},
                "statement_templates": "STATEMENT_TEMPLATES (held-out wordings, never the property templates)"}
    return _write_items(out_dir, SCHEMA_NEW, manifest, concepts, items)


# -- (b) edited words: items ----------------------------------------------------------------------------

def build_edit_items(ontology_path: Path, out_dir: Path, *, tokenizer_name: str, count: int = 200, heldout_fraction: float = 0.5,
                     neighbors: int = 2, seed: int = 0, min_subtokens: int = 2, wordnet: Any = None,
                     alias_table: Path | None = None) -> dict[str, Any]:
    """Item directory for (b): one hypernym edge of existing concepts changed to a same-type filler."""
    from transformers import AutoTokenizer
    if wordnet is None:
        from nltk.corpus import wordnet
    ontology = torch.load(ontology_path, weights_only=False)
    table, table_info = cp.resolve_alias_table(ontology, Path(ontology_path), alias_table=alias_table)
    view = OntologyView(ontology, table)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    rng = random.Random(seed)
    names = ontology["concept_names"]
    category_ids = {view.relation_id[r] for r in CATEGORY_RELATIONS if r in view.relation_id}
    pools = view.filler_pools()
    lexnames: dict[int, str | None] = {}

    def lexname(filler: int) -> str | None:
        if filler not in lexnames:
            kind, _, value = view.atomic_names[filler].partition(":")
            try:
                lexnames[filler] = wordnet.synset(value).lexname() if kind == "synset" else None
            except Exception:
                lexnames[filler] = None
        return lexnames[filler]

    def edge(e: int) -> tuple[int, int] | None:
        """The entry's single templated category edge (entries with several are skipped)."""
        found = [(r, f) for r, f in view.frame(e) if r in category_ids]
        if len(found) != 1:
            return None
        r, f = found[0]
        return (r, f) if (view.pos[e], view.relation_names[r]) in zs.TEMPLATES and view.text(f) and lexname(f) else None

    single = [e for e in range(view.entry_count) if len(table.entry_concepts[e]) == 1 and view.pos[e] in {"n", "v"}
              and edge(e) is not None]
    surfaces = canonical_surfaces(table, tokenizer, min_subtokens, single)
    linkable = [e for e in single if e in surfaces and surfaces[e]["linkable"]]
    frequency = view.frequency if view.frequency is not None else np.zeros(view.entry_count)
    held = [e for e in linkable if e in view.heldout]
    seen = [e for e in linkable if e not in view.heldout and frequency[e] >= 1]
    n_held = min(len(held), round(count * heldout_fraction))
    chosen = rng.sample(held, n_held) + rng.sample(seen, min(len(seen), count - n_held))
    edits: list[dict[str, Any]] = []
    for e in chosen:
        r, old = edge(e)
        synset = wordnet.synset(names[table.entry_concepts[e][0]])
        concept_ancestors = set(zs._ancestors(synset))
        old_synset = view.atomic_names[old].partition(":")[2]
        old_text, surface = view.text(old), surfaces[e]["surface"]
        frame_fillers = {f for _, f in view.frame(e)}

        def plausible(x: int) -> bool:
            value = view.atomic_names[x].partition(":")[2]
            if lexname(x) != lexname(old) or value in concept_ancestors or not view.text(x):
                return False
            if view.text(x) in {old_text, surface}:
                return False
            try:
                return old_synset not in zs._ancestors(wordnet.synset(value))
            except Exception:
                return False

        new = _weighted_choice(rng, pools[(view.pos[e], r)], frame_fillers | {old}, plausible)
        control = None if new is None else _weighted_choice(
            rng, pools[(view.pos[e], r)], frame_fillers | {old, new}, lambda x: plausible(x) and view.text(x) != view.text(new))
        if new is None or control is None:
            continue
        edits.append({"entry": e, "relation": r, "old": old, "new": new, "control": control, "surface": surface})
    edited = {d["entry"] for d in edits}
    index = view.edge_index()
    linkable_set = set(linkable)
    concepts, items = [], []
    neighbour_concepts: dict[int, dict[str, Any]] = {}
    for d in edits:
        e, r, pos = d["entry"], d["relation"], view.pos[d["entry"]]
        relation = view.relation_names[r]
        cid = f"e9e-{e}"
        status = cp.entry_status([e], view.heldout, view.frequency)
        old_text, new_text, control_text = view.text(d["old"]), view.text(d["new"]), view.text(d["control"])
        siblings = sorted(x for x in index[(r, d["old"])] if x in linkable_set and x not in edited and view.pos[x] == pos)
        picked = rng.sample(siblings, min(neighbors, len(siblings)))
        for _ in range(1000):                     # fill up with other linked concepts of the same part of speech
            if len(picked) >= neighbors:
                break
            x = rng.choice(linkable)
            if x not in edited and x not in picked and view.pos[x] == pos and view.text(edge(x)[1]) != new_text:
                picked.append(x)
        neighbour_ids = []
        for x in picked:
            xr, xf = edge(x)
            nid = f"e9e-nb-{x}"
            neighbour_concepts.setdefault(x, {"concept": nid, "role": "neighbour", "entry": x, "surface": surfaces[x]["surface"],
                                              "pos": view.pos[x], "source_concept": names[table.entry_concepts[x][0]],
                                              "relation": view.relation_names[xr], "true": view.atomic_names[xf],
                                              "true_text": view.text(xf), "status": cp.entry_status([x], view.heldout, view.frequency)})
            neighbour_ids.append(nid)
            true_text = view.text(xf)
            if true_text == new_text:
                continue
            items.append({"id": f"{cid}-neighbourhood-{x}", "concept": nid, "edit": cid, "test": "neighbourhood",
                          "relation": view.relation_names[xr], "templates": zs.TEMPLATES[(pos, view.relation_names[xr])],
                          "null": zs.NULL_SURFACE, "candidates": [" " + true_text, " " + new_text], "gold": 0})
        concepts.append({"concept": cid, "role": "edited", "entry": e, "surface": d["surface"], "pos": pos,
                         "source_concept": names[table.entry_concepts[e][0]], "status": status, "relation": relation,
                         "old": view.atomic_names[d["old"]], "new": view.atomic_names[d["new"]],
                         "control": view.atomic_names[d["control"]], "old_text": old_text, "new_text": new_text,
                         "control_text": control_text, "neighbours": neighbour_ids})
        for test, templates in (("efficacy", zs.TEMPLATES[(pos, relation)]), ("paraphrase", STATEMENT_TEMPLATES[(pos, relation)])):
            items.append({"id": f"{cid}-{test}", "concept": cid, "edit": cid, "test": test, "relation": relation,
                          "templates": templates, "null": zs.NULL_SURFACE, "candidates": [" " + old_text, " " + new_text],
                          "gold": 0})
    concepts += [neighbour_concepts[x] for x in sorted(neighbour_concepts)]
    statuses = Counter(c["status"] for c in concepts if c["role"] == "edited")
    manifest = {"scenario": "e9_edits", "description": "one hypernym edge of existing concepts replaced by a same-type filler "
                "(knowledge editing through the ontology)", "ontology": str(ontology_path),
                "ontology_alias_sha256": table.digest(), "alias_table": table_info, "tokenizer": tokenizer_name,
                "min_subtokens": min_subtokens, "seed": seed, "count": count, "heldout_fraction": heldout_fraction,
                "neighbours": neighbors, "edited_status": dict(sorted(statuses.items())),
                "selection": "single-concept noun/verb entries with exactly one templated hypernym/instance-hypernym edge whose "
                             "canonical alias has ≥ ℓ_min subtokens; seeded sample of held-out and seen (training frequency ≥ 1) "
                             "entries",
                "rule": "new filler: same relation, same part of speech, same WordNet lexicographer file as the old filler, "
                        "not an ancestor of the concept, not a descendant of the old filler, not already in the frame "
                        "(frequency-weighted); control: another filler drawn by the same rule",
                "candidates": "efficacy/paraphrase [old, new]; neighbourhood [true, edited concept's new]; gold = index 0"}
    return _write_items(out_dir, SCHEMA_EDITS, manifest, concepts, items)


# -- evaluation: channel surgery -------------------------------------------------------------------------

@contextlib.contextmanager
def inserted_entries(channel: SpanChannel | None, count: int, frames: Sequence[Sequence[tuple[int, int]]] | None = None,
                     *, seed: int = 0) -> Iterator[torch.Tensor | None]:
    """Within the block the channel has `count` extra entries (`SpanChannel.add_entries`); on exit
    it is restored exactly (the same tensors and parameters as before)."""
    if channel is None:
        yield None
        return
    composer = channel.composer if channel.mode == "compose" else None
    saved: dict[str, Any] = {"entry_count": channel.entry_count}
    if composer is not None:
        saved.update(offsets=composer.frame_offsets, relations=composer.frame_relations, fillers=composer.frame_fillers,
                     delta=getattr(composer, "delta", None))
    if channel.mode == "free":
        saved.update(table=channel.table, unseen=channel.unseen)
    if channel.mode == "random":
        saved["table_fixed"] = channel.table_fixed
    ids = channel.add_entries(count, frames if composer is not None else None, seed=seed)
    try:
        yield ids
    finally:
        channel.entry_count = saved["entry_count"]
        if composer is not None:
            composer.frame_offsets, composer.frame_relations, composer.frame_fillers = saved["offsets"], saved["relations"], saved["fillers"]
            if saved["delta"] is not None:
                composer.delta = saved["delta"]
        if channel.mode == "free":
            channel.table, channel.unseen = saved["table"], saved["unseen"]
        if channel.mode == "random":
            channel.table_fixed = saved["table_fixed"]


def edited_schedule(schedule: FrameSchedule, edits: Sequence[tuple[int, int, int, int]]) -> FrameSchedule:
    """`edits`: (entry, relation, old filler, new filler); every matching edge of the entry is changed."""
    fillers = schedule.fillers.clone()
    for entry, relation, old, new in edits:
        lo, hi = int(schedule.offsets[entry]), int(schedule.offsets[entry + 1])
        hit = (schedule.relations[lo:hi] == relation) & (schedule.fillers[lo:hi] == old)
        if not bool(hit.any()):
            raise ValueError(f"entry {entry} has no edge ({relation}, {old})")
        fillers[lo:hi][hit] = new
    return FrameSchedule(schedule.offsets, schedule.relations, fillers)


@contextlib.contextmanager
def applied_edits(channel: SpanChannel | None, edits: Sequence[tuple[int, int, int, int]]) -> Iterator[bool]:
    """Within the block the composer reads the edited schedule; yields False when nothing can be edited."""
    if channel is None or channel.mode != "compose" or not edits:
        yield False
        return
    composer = channel.composer
    original = composer.schedule
    composer.set_schedule(edited_schedule(original, edits))
    try:
        yield True
    finally:
        composer.set_schedule(original)


def extended_adapter(run: E5Run, surfaces: dict[str, int]) -> cp.ChannelModelAdapter:
    """The run's adapter whose linker also maps each invented surface to its new entry id."""
    base = run.adapter
    if base.linker is None:
        return base
    table = base.linker.table
    added = {normalize_alias(s): int(e) for s, e in surfaces.items()}
    clash = sorted(a for a in added if a in table.alias_to_entry)
    if clash:
        raise ValueError(f"invented names already are aliases: {clash[:5]}")
    concept_base = len(run.ontology.get("concept_names") or ()) or (max((max(c) for c in table.entry_concepts if c), default=-1) + 1)
    extra = max(added.values(), default=len(table.entry_concepts) - 1) + 1 - len(table.entry_concepts)
    entry_concepts = list(table.entry_concepts) + [(concept_base + i,) for i in range(max(0, extra))]
    extended = AliasTable({**table.alias_to_entry, **added}, entry_concepts, table.holdout, table.normalization)
    linker = CausalLinker(extended, boundary=base.linker.boundary, min_subtokens=base.linker.min_subtokens)
    frequency = base.train_frequency
    if frequency is not None and extra > 0:
        frequency = np.concatenate([frequency, np.zeros(extra, dtype=frequency.dtype)])
    return dataclasses.replace(base, linker=linker, spans_fn=None, train_frequency=frequency)


def link_check(adapter: cp.ChannelModelAdapter, concepts: Sequence[dict[str, Any]], items: Sequence[dict[str, Any]],
               expected: dict[str, int | None]) -> dict[str, dict[str, Any]]:
    """Per concept: whether its surface injects the expected entry at the position its prompts read,
    in its first prompt, and how many other links fall inside the surface (spurious sub-word links)."""
    first: dict[str, str] = {}
    for item in items:
        first.setdefault(item["concept"], item["templates"][0])
    texts, spans = [], []
    for c in concepts:
        template = first.get(c["concept"], "{x}")
        start = template.index("{x}")
        texts.append(template.format(x=c["surface"])); spans.append((start, start + len(c["surface"])))
    found = adapter.link_targets(texts, spans) if adapter.linker is not None else [[] for _ in texts]
    inner = [0] * len(texts)
    if adapter.linker is not None:
        encoded = adapter.tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        for i, (text, offsets, (s, e)) in enumerate(zip(texts, encoded["offset_mapping"], spans)):
            offsets = [tuple(o) for o in offsets]
            read = cp.read_index(offsets, s, e)
            inner[i] = sum(1 for span in adapter.linker.link(text, offsets)
                           if span.inject_token != read and offsets[span.inject_token][0] >= s and offsets[span.inject_token][1] <= e)
    out = {}
    for c, entries, n in zip(concepts, found, inner):
        want = expected.get(c["concept"])
        status = ("linked" if entries == [want] else "unlinked" if not entries else "mislinked") if want is not None else "no entry"
        out[c["concept"]] = {"entry": want, "found": entries, "status": status, "inner_links": n}
    return out


# -- evaluation: scoring ------------------------------------------------------------------------------------

def continuation_stats(adapter: cp.ModelAdapter, prefixes: Sequence[str], continuations: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Summed `log p(continuation | prefix)` and the number of continuation tokens (as
    `channel_probes.continuation_logprob`, which returns the sum only)."""
    texts = [p + c for p, c in zip(prefixes, continuations)]
    scores, offsets = adapter.token_logprobs(texts) if texts else ([], [])
    sums, counts = np.zeros(len(texts)), np.zeros(len(texts))
    for i, (score, offs, prefix) in enumerate(zip(scores, offsets, prefixes)):
        tokens = [t for t in range(1, len(offs)) if offs[t][1] > len(prefix)]
        sums[i] = sum(float(score[t - 1]) for t in tokens)
        counts[i] = len(tokens)
    return sums, counts


def score_statements(adapter: cp.ModelAdapter, items: Sequence[dict[str, Any]], surfaces: dict[str, str]) -> list[dict[str, Any]]:
    """Loss test: per-token log-probability of each candidate filler after the statement prefix;
    correct = gold has the highest, margin = gold − best distractor, gold_loss in nats/token."""
    prefixes, continuations, _ = zs._prompt_texts(items, surfaces)
    sums, counts = continuation_stats(adapter, prefixes, continuations)
    per_token = sums / np.maximum(counts, 1)
    out, cursor = [], 0
    for item in items:
        k, n_t, gold = len(item["candidates"]), len(item["templates"]), int(item["gold"])
        values = per_token[cursor:cursor + k * n_t].reshape(n_t, k)
        cursor += k * n_t
        others = np.delete(values, gold, axis=1)
        out.append({"id": item["id"], "concept": item["concept"], "test": item["test"], "relation": item["relation"],
                    "edge_kind": item.get("edge_kind"), "correct": float((values.argmax(1) == gold).mean()),
                    "gold_loss": float(-values[:, gold].mean()), "margin": float((values[:, gold] - others.max(1)).mean())})
    return out


def score_pairs(adapter: cp.ModelAdapter, items: Sequence[dict[str, Any]], surfaces: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Edit prompts: per item, `d = log p(candidates[1]) − log p(candidates[0])` per template (summed log-probabilities)."""
    prefixes, continuations, _ = zs._prompt_texts(items, surfaces)
    sums, _ = continuation_stats(adapter, prefixes, continuations)
    out, cursor = {}, 0
    for item in items:
        n_t = len(item["templates"])
        values = sums[cursor:cursor + 2 * n_t].reshape(n_t, 2)
        cursor += 2 * n_t
        d = values[:, 1] - values[:, 0]
        out[item["id"]] = {"d": d.tolist(), "d_mean": float(d.mean()), "new_preferred": float((d > 0).mean())}
    return out


# -- evaluation: (a) new words -----------------------------------------------------------------------------

def available_new_sources(run: E5Run, requested: Sequence[str] | None) -> list[str]:
    mode = run.mode
    allowed = (["own"] if mode in {"none", "hashed"} else ["own", "none"] if mode == "random"
               else ["own", "none", "mean_row"] + (["random_frame"] if mode == "compose" else []))
    return [s for s in (requested or NEW_SOURCES) if s in allowed]


def evaluate_new_words(run: E5Run, items_dir: Path, *, sources: Sequence[str] | None = None, fit_entries: int = 4000,
                       seed: int = 0, log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest, concepts, items = load_item_dir(items_dir, SCHEMA_NEW)
    ontology = run.ontology
    if ontology is None:
        raise ValueError("the run has no ontology")
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    frames = [resolve_frame(c["frame"], relation_id, atomic_id) for c in concepts]
    random_frames = [resolve_frame(c["random_frame"], relation_id, atomic_id) for c in concepts]
    base_count = int(ontology["entry_count"])
    entry_of = {c["concept"]: base_count + i for i, c in enumerate(concepts)}
    for c in concepts:
        c["entry"] = entry_of[c["concept"]]
    adapter = extended_adapter(run, {c["surface"]: entry_of[c["concept"]] for c in concepts})
    resolved = link_check(adapter, concepts, items, entry_of)
    surfaces = {c["concept"]: c["surface"] for c in concepts}
    sources = available_new_sources(run, sources)
    started = time.monotonic()
    rows: dict[str, dict[int, torch.Tensor]] = {}
    info: dict[str, Any] = {"new_entries": [base_count, base_count + len(concepts)]}
    channel = run.channel
    if channel is not None and {"none", "mean_row"} & set(sources):
        wanted = [s for s in ("none", "mean_row") if s in sources]
        linked = {c["concept"]: entry_of[c["concept"]] for c in concepts}
        with inserted_entries(channel, len(concepts), frames, seed=seed):
            rows, fit_info = zs.baseline_rows(run, adapter, concepts, linked, wanted, manifest=manifest, items_dir=Path(items_dir),
                                              fit_entries=fit_entries, contexts=0, seed=seed, log=log)
        info.update(fit_info)
    # A baseline whose row could not be built (no seen entries to average) is dropped, never scored as `own`.
    sources = [s for s in sources if s not in {"none", "mean_row"} or s in rows]
    prompt_items = [i for i in items if i["test"] in {"property", "entailment"}]
    statement_items = [i for i in items if i["test"] == "statement"]
    null_scores = None
    per_source: dict[str, Any] = {}
    for source in sources:
        log(f"  new words: source {source}")
        frames_now = random_frames if source == "random_frame" else frames
        with inserted_entries(channel, len(concepts), frames_now, seed=seed), \
                (override_rows(channel, rows.get(source)) if channel is not None else contextlib.nullcontext()):
            prompts, null_scores = zs.score_prompts(adapter, prompt_items, surfaces, null_scores)
            statements = score_statements(adapter, statement_items, surfaces)
        kinds = {i["id"]: i.get("edge_kind") for i in items}
        for row in prompts:
            row["edge_kind"] = kinds.get(row["id"])
        per_source[source] = {"prompts": prompts, "statements": statements}
    return {"manifest": manifest, "resolved": resolved, "sources": sources, "results": per_source, "info": info,
            "seconds": time.monotonic() - started}


NEW_TESTS = ("property", "property_new", "property_category", "entailment", "paraphrase", "statement_accuracy",
             "statement_margin", "statement_loss")
LOWER_IS_BETTER = {"statement_loss"}


def new_word_vectors(result: dict[str, Any], keep: set[str]) -> dict[str, tuple[list[str], np.ndarray]]:
    """Per test: (item ids, per-item values) over the concepts in `keep`."""
    prompts = [r for r in result["prompts"] if r["concept"] in keep]
    statements = [r for r in result["statements"] if r["concept"] in keep]
    pick = lambda rows, key: ([r["id"] for r in rows], np.asarray([r[key] for r in rows], dtype=float))
    prop = [r for r in prompts if r["test"] == "property"]
    return {
        "property": pick(prop, "correct"),
        "property_new": pick([r for r in prop if r.get("edge_kind") == "resampled"], "correct"),
        "property_category": pick([r for r in prop if r.get("edge_kind") == "category"], "correct"),
        "entailment": pick([r for r in prompts if r["test"] == "entailment"], "correct"),
        "paraphrase": pick([r for r in prop if r["consistent"] is not None], "consistent"),
        "statement_accuracy": pick(statements, "correct"),
        "statement_margin": pick(statements, "margin"),
        "statement_loss": pick(statements, "gold_loss"),
    }


def summarize_new_words(evaluation: dict[str, Any], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    resolved = evaluation["resolved"]
    linked = {c for c, r in resolved.items() if r["status"] == "linked"}
    out: dict[str, Any] = {"concepts": len(resolved), "linked_concepts": len(linked),
                           "status_counts": dict(sorted(Counter(r["status"] for r in resolved.values()).items())),
                           "inner_links": int(sum(r["inner_links"] for r in resolved.values())), "sources": {}, "comparisons": []}
    first = next(iter(evaluation["results"].values()), None)
    if first is not None:
        out["chance"] = {t: float(np.mean([1.0 / len(r["pmi"][0]) for r in first["prompts"] if r["test"] == t])) if any(
            r["test"] == t for r in first["prompts"]) else None for t in ("property", "entailment")}
    vectors = {s: new_word_vectors(r, linked) for s, r in evaluation["results"].items()}
    for source, by_test in vectors.items():
        out["sources"][source] = {t: {"mean": float(v.mean()) if v.size else None, "n": int(v.size)} for t, (_, v) in by_test.items()}
    if "own" in vectors:
        for test in NEW_TESTS:
            block = []
            ids_a, a = vectors["own"][test]
            for source in evaluation["sources"]:
                if source == "own" or not a.size:
                    continue
                ids_b, b = vectors[source][test]
                if ids_a != ids_b:
                    continue
                ci = zs.paired_difference(a, b, resamples=resamples, seed=seed)
                better = ci["mean"] < 0 if test in LOWER_IS_BETTER else ci["mean"] > 0
                block.append({"test": test, "baseline": source, "difference": ci["mean"], "ci_low": ci["ci_low"],
                              "ci_high": ci["ci_high"], "n": ci["n"], "p_value": ci["p_value"], "better": bool(better)})
            for row, adjusted in zip(block, holm_adjust([r["p_value"] for r in block]) if block else []):
                row.update(p_holm=adjusted, significant=adjusted < 0.05)
            out["comparisons"] += block
    return out


# -- evaluation: (b) edited words ------------------------------------------------------------------------------

def evaluate_edits(run: E5Run, items_dir: Path, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    manifest, concepts, items = load_item_dir(items_dir, SCHEMA_EDITS)
    ontology = run.ontology
    if ontology is None:
        raise ValueError("the run has no ontology")
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atomic_id = {n: i for i, n in enumerate(ontology["atomic_names"])}
    edited = [c for c in concepts if c.get("role") == "edited"]
    adapter = run.adapter
    resolved = link_check(adapter, concepts, items, {c["concept"]: int(c["entry"]) for c in concepts})
    statuses = {c["concept"]: cp.entry_status([int(c["entry"])], adapter.heldout_entries, adapter.train_frequency) for c in concepts}
    for c in concepts:
        resolved[c["concept"]]["entry_status"] = statuses[c["concept"]]
    surfaces = {c["concept"]: c["surface"] for c in concepts}

    def edits(target: str) -> list[tuple[int, int, int, int]]:
        return [(int(c["entry"]), relation_id[c["relation"]], atomic_id[c["old"]], atomic_id[c[target]]) for c in edited]

    started = time.monotonic()
    results: dict[str, dict[str, dict[str, Any]]] = {}
    applicable = run.channel is not None and run.channel.mode == "compose"
    for condition in EDIT_CONDITIONS:
        if condition != "before" and not applicable:
            results[condition] = results["before"]          # nothing to edit: the model is unchanged
            continue
        log(f"  edits: {condition}")
        with applied_edits(run.channel, [] if condition == "before" else edits("new" if condition == "after" else "control")):
            results[condition] = score_pairs(adapter, items, surfaces)
    return {"manifest": manifest, "resolved": resolved, "results": results, "edit_applicable": applicable,
            "seconds": time.monotonic() - started}


def summarize_edits(evaluation: dict[str, Any], items: Sequence[dict[str, Any]], concepts: Sequence[dict[str, Any]], *,
                    resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    resolved, results = evaluation["resolved"], evaluation["results"]
    linked = {c for c, r in resolved.items() if r["status"] == "linked"}
    edit_status = {c["concept"]: ("heldout" if resolved[c["concept"]]["entry_status"] == "heldout" else "seen")
                   for c in concepts if c.get("role") == "edited"}
    out: dict[str, Any] = {"edit_applicable": evaluation["edit_applicable"], "edited": len(edit_status),
                           "edited_linked": sum(c in linked for c in edit_status), "subsets": {}}
    for subset in ("all", "seen", "heldout"):
        keep = {c for c, s in edit_status.items() if c in linked and (subset == "all" or s == subset)}
        entry: dict[str, Any] = {"edits": len(keep)}
        for test in ("efficacy", "paraphrase", "neighbourhood"):
            rows = [i for i in items if i["test"] == test and i["edit"] in keep
                    and (test != "neighbourhood" or resolved.get(i["concept"], {}).get("status") == "linked")]
            ids = [i["id"] for i in rows]
            if not ids:
                entry[test] = {"n": 0}
                continue
            d = {c: np.asarray([results[c][i]["d_mean"] for i in ids]) for c in EDIT_CONDITIONS}
            share = {c: np.asarray([results[c][i]["new_preferred"] for i in ids]) for c in EDIT_CONDITIONS}
            success = (lambda c: 1 - share[c]) if test == "neighbourhood" else (lambda c: share[c])
            change = zs.paired_difference(d["after"], d["before"], resamples=resamples, seed=seed)
            control = zs.paired_difference(d["control"], d["before"], resamples=resamples, seed=seed)
            target = zs.paired_difference(d["after"], d["control"], resamples=resamples, seed=seed)
            entry[test] = {"n": len(ids), **{f"success_{c}": float(success(c).mean()) for c in EDIT_CONDITIONS},
                           "magnitude": change, "control_magnitude": control, "target_minus_control": target,
                           "max_abs_change": float(np.abs(d["after"] - d["before"]).max()),
                           "mean_abs_change": float(np.abs(d["after"] - d["before"]).mean())}
        values = [entry.get(t, {}).get("success_after") for t in ("efficacy", "paraphrase", "neighbourhood")]
        entry["score"] = (3 / sum(1 / v for v in values) if all(v for v in values) else
                          0.0 if all(v is not None for v in values) else None)
        out["subsets"][subset] = entry
    out["statuses"] = dict(sorted(Counter(edit_status.values()).items()))
    out["neighbours"] = sum(1 for c in concepts if c.get("role") == "neighbour")
    return out


# -- report ---------------------------------------------------------------------------------------------------------

def render(header: dict[str, Any], new_summary: dict[str, Any] | None, edit_summary: dict[str, Any] | None) -> str:
    source = header["source"]
    quantized = source.get("quantization", {}).get("variant") if source.get("quantization") else "bf16 (as trained)"
    lines = [f"# E9 dimension 3 — zero-shot by ontology editing — {source['condition']} seed {source['seed']} ({source['size']})", "",
             f"Run `{source['run']}`, channel `{source['channel_mode']}`, weights: **{quantized}**. No weight is updated: "
             "new words and edits change only the alias table and the composer's frame schedule.", ""]
    if new_summary is not None:
        lines += ["## (a) New words", "",
                  f"{new_summary['linked_concepts']} of {new_summary['concepts']} invented names link to their new entry "
                  f"({new_summary['status_counts']}; spurious sub-word links inside the names: {new_summary['inner_links']}). "
                  f"Chance: property {fmt((new_summary.get('chance') or {}).get('property'), 3)}, entailment "
                  f"{fmt((new_summary.get('chance') or {}).get('entailment'), 3)}.", "",
                  "| source | " + " | ".join(NEW_TESTS) + " |", "|---|" + "---:|" * len(NEW_TESTS)]
        for name, by_test in new_summary["sources"].items():
            lines.append(f"| {name} | " + " | ".join(fmt(by_test[t]["mean"], 4) for t in NEW_TESTS) + " |")
        if new_summary["comparisons"]:
            lines += ["", "`own` − baseline (paired bootstrap over items; Holm over baselines within each test):", "",
                      "| test | baseline | difference [95% CI] | n | p (Holm) | own better |", "|---|---|---|---:|---:|---|"]
            for c in new_summary["comparisons"]:
                lines.append(f"| {c['test']} | {c['baseline']} | {fmt_ci({'mean': c['difference'], 'ci_low': c['ci_low'], 'ci_high': c['ci_high']})} "
                             f"| {c['n']} | {fmt(c.get('p_holm'), 4)} | {'yes' if c.get('significant') and c['better'] else 'no'} |")
        lines += ["", "`property_new`: the resampled edges (the new combinations); `statement_*`: the loss test in held-out "
                  "wordings (per-token log-probability; `statement_loss` = gold filler nats/token, lower is better).", ""]
    if edit_summary is not None:
        lines += ["## (b) Edited words", "",
                  f"{edit_summary['edited_linked']} of {edit_summary['edited']} edited concepts link ({edit_summary['statuses']}); "
                  f"{edit_summary['neighbours']} neighbour concepts." + ("" if edit_summary["edit_applicable"] else
                  " **This model has no composed rows: the edit cannot change it (after = before).**"), "",
                  "| subset | edits | ES before → after (control) | EM [95% CI] | EM − EM_control [95% CI] | PS after | PM [95% CI] | "
                  "NS after | neighbourhood abs Δd mean / max | score |", "|---|---:|---|---|---|---:|---|---:|---:|---:|"]
        for subset, e in edit_summary["subsets"].items():
            eff, par, nb = e.get("efficacy", {}), e.get("paraphrase", {}), e.get("neighbourhood", {})
            if not eff.get("n"):
                lines.append(f"| {subset} | {e['edits']} | — | — | — | — | — | — | — | — |")
                continue
            lines.append(f"| {subset} | {e['edits']} | {fmt(eff['success_before'], 3)} → {fmt(eff['success_after'], 3)} "
                         f"({fmt(eff['success_control'], 3)}) | {fmt_ci(eff['magnitude'])} | {fmt_ci(eff['target_minus_control'])} | "
                         f"{fmt(par.get('success_after'), 3)} | {fmt_ci(par.get('magnitude'))} | {fmt(nb.get('success_after'), 3)} | "
                         f"{fmt(nb.get('mean_abs_change'), 4)} / {fmt(nb.get('max_abs_change'), 4)} | {fmt(e.get('score'), 3)} |")
        lines += ["", "ES/PS: share of edit / paraphrase prompts preferring the new filler over the old (ROME/MEMIT efficacy and "
                  "generalization); EM/PM: mean change of `log p(new) − log p(old)` from before to after the edit (nats); "
                  "NS: share of neighbourhood prompts still preferring their true filler over the edited concept's new one "
                  "(specificity); control: the same edit to a different same-type filler; score: harmonic mean of ES, PS, NS.", ""]
    return "\n".join(lines)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    requested = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    config = {"experiment": "e9-ontology-edit", "run": str(args.run), "checkpoint": args.checkpoint,
              "new_items": str(args.new_items) if args.new_items else None,
              "edit_items": str(args.edit_items) if args.edit_items else None, "sources": requested,
              "fit_entries": args.fit_entries, "seed": args.seed, "resamples": args.resamples, "quantize": args.quantize,
              "quantize_channel": bool(args.quantize_channel), "group_size": args.group_size}
    if not args.new_items and not args.edit_items:
        raise ValueError("pass --new-items and/or --edit-items")
    if args.overwrite:
        clear_output(args.output)
    git_at_start = start_output(args.output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, quantize=args.quantize,
                   quantize_channel=args.quantize_channel, group_size=args.group_size)
    header = {"source": run.describe(), "new_items": config["new_items"], "edit_items": config["edit_items"]}
    new_summary = edit_summary = None
    document: dict[str, Any] = {**header}
    with (args.output / "predictions.jsonl").open("w") as handle:
        if args.new_items:
            evaluation = evaluate_new_words(run, args.new_items, sources=requested, fit_entries=args.fit_entries, seed=args.seed)
            new_summary = summarize_new_words(evaluation, resamples=args.resamples, seed=args.seed)
            for source, result in evaluation["results"].items():
                for row in result["prompts"] + result["statements"]:
                    handle.write(json.dumps(json_ready({"part": "new", "source": source,
                                                        **{k: v for k, v in row.items() if k != "pmi"}})) + "\n")
            document["new_words"] = {"manifest": evaluation["manifest"], "resolved": evaluation["resolved"],
                                     "info": evaluation["info"], "summary": new_summary, "seconds": evaluation["seconds"]}
        if args.edit_items:
            evaluation = evaluate_edits(run, args.edit_items)
            _, concepts, items = load_item_dir(args.edit_items, SCHEMA_EDITS)
            edit_summary = summarize_edits(evaluation, items, concepts, resamples=args.resamples, seed=args.seed)
            for condition, by_item in evaluation["results"].items():
                for item_id, row in by_item.items():
                    handle.write(json.dumps(json_ready({"part": "edit", "condition": condition, "id": item_id, **row})) + "\n")
            document["edits"] = {"manifest": evaluation["manifest"], "resolved": evaluation["resolved"],
                                 "summary": edit_summary, "seconds": evaluation["seconds"]}
    write_json(args.output / "summary.json", document)
    (args.output / "report.md").write_text(render(header, new_summary, edit_summary))
    finish_output(args.output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


def run_items(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"{args.output} is not empty")
    if args.kind == "new":
        texts = _contamination_texts(args.contamination_corpus, args.tokenizer, args.contamination_tokens)
        return build_new_word_items(args.ontology, args.output, tokenizer_name=args.tokenizer, count=args.count or 300,
                                    seed=args.seed, min_subtokens=args.min_subtokens, contamination_texts=texts,
                                    reserved_names=_reserved_names(args.reserved_names), name_seed=args.name_seed)
    return build_edit_items(args.ontology, args.output, tokenizer_name=args.tokenizer, count=args.count or 200, seed=args.seed,
                            min_subtokens=args.min_subtokens)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("items", help="build an item directory (new words or edits)")
    build.add_argument("--kind", required=True, choices=["new", "edits"])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--ontology", type=Path, default=Path("~/data/vsa-llm/c3/wordnet-smollm2-v1/ontology.pt").expanduser())
    build.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    build.add_argument("--min-subtokens", type=int, default=2); build.add_argument("--count", type=int, default=None)
    build.add_argument("--seed", type=int, default=0); build.add_argument("--name-seed", type=int, default=11)
    build.add_argument("--contamination-corpus", type=Path, nargs="*", default=[],
                       help="token corpora whose first --contamination-tokens tokens must not contain an invented name")
    build.add_argument("--contamination-tokens", type=int, default=20_000_000)
    build.add_argument("--reserved-names", type=Path, nargs="*", default=[],
                       help="item directories whose surfaces the invented names must avoid (e.g. the E5.4 synthetic items)")
    ev = sub.add_parser("evaluate", help="evaluate a trained run on the new-word and/or edit items")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--new-items", type=Path, default=None); ev.add_argument("--edit-items", type=Path, default=None)
    ev.add_argument("--checkpoint", default="final.pt")
    ev.add_argument("--sources", default="", help="comma-separated subset of " + ",".join(NEW_SOURCES))
    ev.add_argument("--fit-entries", type=int, default=4000); ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--resamples", type=int, default=2000)
    ev.add_argument("--batch-size", type=int, default=64); ev.add_argument("--device", default=None)
    ev.add_argument("--quantize", choices=["int8", "int4"], default=None,
                    help="evaluate after post-training weight-only quantization as in e4_quant (int4 needs CUDA)")
    ev.add_argument("--quantize-channel", action="store_true", help="with --quantize: variant B (channel quantized too)")
    ev.add_argument("--group-size", default="auto", help="INT4 group size, or auto (as e4_quant)")
    ev.add_argument("--overwrite", action="store_true", help="replace the result files of a previous evaluation in --output")
    args = parser.parse_args(argv)
    if args.command == "items":
        print(json.dumps(run_items(args), indent=2, default=str))
        return
    if args.quantize_channel and not args.quantize:
        parser.error("--quantize-channel needs --quantize")
    document = run_evaluate(args)
    print(json.dumps({"new_words": (document.get("new_words") or {}).get("summary", {}).get("sources"),
                      "edits": {k: v.get("score") for k, v in ((document.get("edits") or {}).get("summary", {}).get("subsets") or {}).items()}},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
