"""E12 — 3b: learning from tool-using traces (author decision 62; pre-registration
`experiments/e12-self-query/preregistration.md` §12, amendment 16.3). Training (LoRA on a finished E9 run) and evaluation.

**Question.** If the model practises answering role questions with its own recalled frame in context, does it learn to read
roles from the channel — telling role-swap twins apart **without the tool**? And does a model trained on tool-using traces
learn to call the tool well?

**Training questions** (`items`; built once on the CPU, committed: `experiments/e12-self-query/items/traces-t5-smollm2-v1`):
- *training twins* (`twins/`, the E9 role-item format): 2,000 new pairs built as the test twins (`e9_binding_items.build_twins`,
  the test set's two relation pairs), item seed 1, name seed 29; names disjoint from every E9 item set, frames disjoint from
  every test frame (twins and new words), no {X, Y} filler pair of a test twin;
- *seen terms*: every seen entry (training frequency ≥ 10, not held out, not synthetic, one concept, a linkable surface) that
  no E9/E12 item set names (anchors, bridges, partners, edited entries), one question per templated relation of its frame;
  gold = the relation's first filler; the other option = the term's own filler of another relation of the same answer type
  (`TrackLexicon.atom_type`) when it has one, else a frequency-weighted filler of the relation's pool;
- one question per (term, relation), worded with one of the property templates' first two paraphrases (a seeded draw; the
  twins' `choice` wordings); the held-out wording (`cloze`) is never trained.

**Arms** (the same questions; the loss on the model's own turns only — never on questions or observations):
- `T` — 3a's ReAct format: `Question: … ___? Options: a | b` / `Thought: I should recall <term>, <relation words>.` /
  `Action: recall[<term>, <relation words>]` / `Observation: <the run's own recall, exactly as 3a's tool prints it>` /
  `Thought: The recall says <decoded option>.` / `Answer: <gold>`;
- `I` — `Question: …` / `Answer: <gold>` (no tool); on a C5ut run it is the I-ut control;
- `S` — stepwise internalization (Deng et al. 2024): epoch 1 T's traces, epoch 2 without the observation, epoch 3 I's format;
- `L` — the control: the same number of sequences and tokens per epoch as T, windows of the T5 training corpus (LM loss);
- `base` — no training: the untrained run under the same tests (3a's few-shot reference at this host size).

**Decision 64 (amendment 16.5): the hypothesis is "LoRA learns to use the decoded store"**, not "LoRA learns to unbind a
learned-HRR role" (phase A: the roles live in the store, not the host — a C0′ host reading C5's store ties C5 — and a fixed
random binding decodes as well as a learned one). Two hosts test it, the same training and tests (`D64_ARMS`):
- *C0′ host* (`--store`: the tool reads C5's store of the same seed; the host has no channel): arms `T` and `base` —
  `T@C0p` ≈ `T` with the decoded store (the fixed pipeline, the agentic twins), and ≈ 0.5 without the tool (no channel);
- *C5rf* (its own fixed-random store): arms `T`, `I`, `L` — B1's contrast I − L and arm T's decoded-store tests ≈ C5's.
Read as equivalence contrasts (`decision64`; ±0.05 with the decoded store in context, ±0.075 for B1's I − L), reported as
secondaries; B1 and B2 are unchanged.

**Training** (§12): rank-16 LoRA (`integrations.transformers.add_lora`, its default targets and α = 32) on the host; the
host's weights and the whole channel (composer, projector, gate, P1 context) frozen; 3 epochs, AdamW lr 2·10⁻⁴, 32
sequences per optimizer step, 3% warmup, then the trainer's cosine decay to 10% (`training.lm._lr`); weight decay 0.1, betas
(0.9, 0.95), gradient clip 1.0 (the E9 trainer's values); one sequence per question (no packing); bf16 autocast on the GPU.
Training twins are inserted with their gold frames and linked (`e12_self_query.host_view`), as at evaluation.

**Tests** (every arm): the test twins without the tool (`none`: the channel only; `choice` and `cloze`) and with the fixed
pipeline (`recall:own`), phase A's scorer; the first 300 v2 new words (`none`); WP-UB two-hop (≤ 150 anchors per subset;
`none` and chained `recall:own`); the twins in the trained question format without the tool (3a's `no_tool` prompt, both
twins of every pair, first relation); arms T, I and base: 3a's agentic episodes on the first 100 pairs (both twins, first
relation; `e12_agent.pilot`).

Outputs (`RUN/self-query-traces-<arm>/`): `summary.json` (per test the phase-A summaries — per-pair contrasts — and the agent
summary), `training.jsonl` (per step loss, tokens, seconds), `episodes.jsonl.gz` (agentic arms), `report.md`,
`adapters.pt` (the LoRA weights; git-ignored under `runs/`), `resolved_config.yaml`, `manifest.json`.

    python -m vsa_embed.experiments.e12_traces items --output experiments/e12-self-query/items/traces-t5-smollm2-v1     (CPU)
    python -m vsa_embed.experiments.e12_traces run --run RUN --arm T [--store RUN] [--train-items DIR] [--overwrite]     (GPU)
    python -m vsa_embed.experiments.e12_traces report --runs experiments/e9-retrofit/runs/t5 --output DIR                 (CPU)
    python -m vsa_embed.experiments.e12_traces commands --stage t5 --priority 54.4497 [--set decision64]                 (print jobs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .. import self_query as sq
from . import e9_binding_items as role_items
from . import e9_ontology_edit as edit
from . import e9_understanding as und
from . import e12_agent as agent
from . import e12_self_query as sqx
from .e5_common import E5Run, finish_output, json_ready, open_run, start_output, write_json

SCHEMA = "e12-traces/1"
ROOT = Path("experiments/e9-retrofit")
E12_ROOT = Path("experiments/e12-self-query")
ITEMS = E12_ROOT / "items" / "traces-t5-smollm2-v1"
OUTPUT_PREFIX = "self-query-traces-"
ARMS = ("T", "I", "S", "L", "base")
TRAINED = ("T", "I", "S", "L")
AGENTIC = ("T", "I", "base")
RESULT_FILES = ("summary.json", "training.jsonl", "episodes.jsonl.gz", "report.md", "adapters.pt", "resolved_config.yaml",
                "manifest.json")
# Pre-registered (§12).
RANK = 16
EPOCHS = 3
LR = 2e-4
SEQUENCES_PER_STEP = 32
WARMUP = 0.03
TRAIN_PAIRS = 2000
ITEM_SEED = 1
NAME_SEED = 29
AGENT_PAIRS = 100
# Values §12 leaves open, fixed by rule before any run (amendment 16.3; nothing is tuned).
ALPHA = 32.0                      # add_lora's default (scale α / r = 2)
WEIGHT_DECAY = 0.1                # the E9 trainer's AdamW settings
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
MIN_LR_RATIO = 0.1                # cosine decay to 10% after warmup (training.lm._lr)
QUESTION_SEED = 0                 # template draw and option order of the training questions
NEW_WORDS = 300                   # phase A's pre-registered subsets (§3)
UNDERSTANDING_ANCHORS = 150
# Decision 64 (amendment 16.5): host model → arms; the tool of a host without a store reads C5's store of its seed.
D64_ARMS = {"C0p": ("T", "base"), "C5rf": ("T", "I", "L")}
D64_STORE = {"C0p": "C5"}
TOOL_MARGIN = 0.05                # equivalence margin: contrasts with the decoded store in context (fixed pipeline, agentic)
B1_MARGIN = 0.075                 # equivalence margin: the C5rf − C5 difference of B1's contrast I − L (half B1's 0.15 band)


# ---------------------------------------------------------------- training questions (CPU, once)


def item_references(items_roots: Sequence[Path], track: str) -> dict[str, Any]:
    """What a training set must avoid: the surfaces (lower-case) of every concept of every item directory under
    `items_roots`; for the track's item sets (`-<track>-` in the directory name) the entries they name, the frames of their
    new terms (twins, new words) by name, and the twins' {X, Y} filler pairs by name."""
    surfaces: set[str] = set()
    entries: set[int] = set()
    frames: list[list[list[str]]] = []
    fillers: set[tuple[str, str]] = set()
    sources = []
    for root in items_roots:
        for path in sorted(Path(root).glob("*/concepts.jsonl")):
            rows = und.read_jsonl(path)
            surfaces |= {str(r["surface"]).lower() for r in rows if r.get("surface")}
            if f"-{track}-" not in path.parent.name and not path.parent.name.endswith(f"-{track}"):
                continue
            sources.append(path.parent.name)
            for r in rows:
                if r.get("entry") is not None:
                    entries.add(int(r["entry"]))
                if r.get("entry") is None and r.get("frame"):
                    frames.append([list(e) for e in r["frame"]])
                if r.get("twin") == "A" and r.get("fillers"):
                    fillers.add(tuple(sorted((r["fillers"]["X"], r["fillers"]["Y"]))))
    return {"surfaces": surfaces, "entries": entries, "frames": frames, "fillers": fillers, "track_sets": sources}


def seen_questions(ctx: und.BuildContext, excluded: set[int], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One question per templated relation of every eligible seen entry (module docstring)."""
    view, lexicon = ctx.view, ctx.lexicon
    templated = set(role_items.templated_relations(ctx))
    frequency = view.frequency if view.frequency is not None else np.zeros(view.entry_count)
    questions, skipped = [], Counter()
    for e in range(view.entry_count):
        if frequency[e] < 10 or e in view.heldout or e in ctx.synthetic or e in excluded or len(ctx.table.entry_concepts[e]) != 1:
            continue
        info = ctx.surfaces.get(e)
        if not info or not info.get("linkable"):
            skipped["unlinkable"] += 1
            continue
        surface = ctx.shown(e)
        frame = view.frame(e)
        done: set[int] = set()
        for r, gold in frame:
            name = view.relation_names[r]
            if r in done or name not in templated:
                continue
            done.add(r)
            text = ctx.text(gold)
            if not text:
                skipped["no text"] += 1
                continue
            mine = {f for q, f in frame if q == r}
            texts_mine = {ctx.text(f) for f in mine}
            kind = lexicon.atom_type(view.atomic_names[gold])
            own = [f for q, f in frame if q != r and f not in mine and ctx.text(f) and ctx.text(f) not in texts_mine
                   and lexicon.atom_type(view.atomic_names[f]) == kind]
            if own:
                other, source = rng.choice(sorted(set(own))), "own"
            else:
                other = edit._weighted_choice(rng, ctx.pools[name], mine, lambda x: bool(ctx.text(x)) and ctx.text(x) not in texts_mine)
                source = "pool"
            if other is None:
                skipped["no distractor"] += 1
                continue
            options = [gold, other]
            rng.shuffle(options)
            templates = lexicon.prompts("*", name)
            questions.append({"id": f"seen-{e}-{name}", "source": "seen", "concept": f"seen-{e}", "entry": int(e), "surface": surface,
                              "relation": name, "template": templates[rng.randrange(len(templates))],
                              "candidates": [lexicon.answer(name, ctx.text(a)) for a in options], "gold": options.index(gold),
                              "options": [view.atomic_names[a] for a in options], "distractor": source})
    return questions, dict(skipped)


def build_training_items(ctx: und.BuildContext, output: Path, *, test_twins: Path, items_roots: Sequence[Path],
                         pairs: int = TRAIN_PAIRS, seed: int = ITEM_SEED, name_seed: int = NAME_SEED, question_seed: int = QUESTION_SEED,
                         contamination_texts: Sequence[str] = (), wordnet: Any = None, min_shared: int = 20) -> dict[str, Any]:
    """The 3b training questions (module docstring): `twins/` (E9 role items) and `questions.jsonl.gz` (one per (term,
    relation)), with the disjointness checks in `manifest.json`."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    view = ctx.view
    atomic_id = {n: i for i, n in enumerate(view.atomic_names)}
    refs = item_references(items_roots, ctx.track)
    test_manifest, test_concepts, _ = role_items.load_items(test_twins)
    relation_pairs = [(p["r1"], p["r2"]) for p in test_manifest["relation_pairs"]]
    known = [f for f in refs["frames"] if all(r in view.relation_id and a in atomic_id for r, a in f)]
    exclude_frames = [edit.resolve_frame(f, view.relation_id, atomic_id) for f in known]
    exclude_fillers = [(atomic_id[x], atomic_id[y]) for x, y in sorted(refs["fillers"]) if x in atomic_id and y in atomic_id]
    twins_manifest = role_items.build_twins(ctx, output / "twins", count=pairs, seed=seed, name_seed=name_seed, min_shared=min_shared,
                                            contamination_texts=contamination_texts, reserved_names=refs["surfaces"], wordnet=wordnet,
                                            pairs=relation_pairs, exclude_frames=exclude_frames, exclude_fillers=exclude_fillers)
    _, concepts, items = role_items.load_items(output / "twins")
    rng = random.Random(question_seed)
    questions = []
    surface_of = {c["concept"]: c["surface"] for c in concepts}
    for item in items:
        if item["kind"] != "choice":
            continue
        k = rng.randrange(len(item["templates"]))
        questions.append({"id": item["id"], "source": "twins", "concept": item["concept"], "entry": None, "surface": surface_of[item["concept"]],
                          "relation": item["relation"], "template": item["templates"][k], "candidates": list(item["candidates"]),
                          "gold": int(item["gold"]), "pair": item["meta"]["pair"], "twin": item["meta"]["twin"]})
    seen, skipped = seen_questions(ctx, refs["entries"], rng)
    questions += seen
    und.write_jsonl_gz(output / "questions.jsonl.gz", questions)
    # disjointness checks (each must be 0)
    test_frames = {frozenset(map(tuple, f)) for f in exclude_frames}
    test_pairs = {frozenset(p) for p in exclude_fillers}
    built_frames = {frozenset((view.relation_id[r], atomic_id[a]) for r, a in c["frame"]) for c in concepts}
    built_pairs = {frozenset((atomic_id[c["fillers"]["X"]], atomic_id[c["fillers"]["Y"]])) for c in concepts if c["twin"] == "A"}
    checks = {"twin names in reserved item-set names": sum(c["surface"].lower() in refs["surfaces"] for c in concepts),
              "twin frames equal to a test frame": len(built_frames & test_frames),
              "twin filler pairs of a test twin": len(built_pairs & test_pairs),
              "seen entries named by an item set": sum(q["entry"] in refs["entries"] for q in seen),
              "seen entries held out": sum(q["entry"] in view.heldout for q in seen)}
    if any(checks.values()):
        raise AssertionError(f"training items overlap the test sets: {checks}")
    manifest = {"schema": SCHEMA, "kind": "traces", "track": ctx.track, "family": ctx.family, "tokenizer": ctx.tokenizer_name,
                "twins": {"pairs": twins_manifest["pairs"], "seed": seed, "name_seed": name_seed, "relation_pairs": relation_pairs,
                          "sha256": twins_manifest["sha256"]},
                "question_seed": question_seed, "test_twins": str(test_twins), "test_twins_sha256": test_manifest.get("sha256"),
                "reference_sets": refs["track_sets"], "reserved_names": len(refs["surfaces"]), "excluded_entries": len(refs["entries"]),
                "excluded_frames": len(test_frames), "excluded_filler_pairs": len(test_pairs), "checks": checks,
                "counts": {"questions": len(questions), **dict(Counter(q["source"] for q in questions)),
                           "seen_entries": len({q["entry"] for q in seen}), "seen_distractor": dict(Counter(q["distractor"] for q in seen)),
                           "seen_skipped": skipped, "relations": dict(sorted(Counter(q["relation"] for q in questions).items()))},
                "rules": {"twins": "e9_binding_items.build_twins with the test set's relation pairs, item seed 1, name seed 29; names "
                                   "disjoint from every item set's surfaces; frames disjoint from every test frame; no test {X, Y} pair",
                          "seen": "training frequency >= 10, not held out, not synthetic, one concept, linkable surface, named by no item "
                                  "set of the track; one question per templated relation; gold = the relation's first filler; other "
                                  "option = the term's own filler of another relation of the same answer type (atom_type), else a "
                                  "frequency-weighted filler of the relation's pool",
                          "wording": "one of the property templates' first two paraphrases (seeded); options in a seeded order (twins: "
                                     "[X, Y] as the test twins)"}}
    manifest["sha256"] = {"questions.jsonl.gz": hashlib.sha256((output / "questions.jsonl.gz").read_bytes()).hexdigest()}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_questions(items: Path, *, limit: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], sqx.ItemSet]:
    """(manifest, questions, the training twins as an item set). `limit` (smoke tests): the first N questions of each source."""
    items = Path(items)
    manifest = json.loads((items / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{items} is not an {SCHEMA} directory")
    questions = und.read_jsonl(items / "questions.jsonl")
    if limit:
        kept: dict[str, int] = Counter()
        out = []
        for q in questions:
            if kept[q["source"]] < limit:
                out.append(q); kept[q["source"]] += 1
        questions = out
    twins = sqx.load_item_set(items / "twins")
    used = {q["concept"] for q in questions if q["source"] == "twins"}
    twins.concepts = [c for c in twins.concepts if c["concept"] in used]
    twins.prompts = [p for p in twins.prompts if p.concept in used]
    return manifest, questions, twins


# ---------------------------------------------------------------- the arms' texts


@dataclass
class Sequence_:
    """One training sequence: text and the character ranges of the model's own turns (the loss)."""

    text: str
    spans: list[tuple[int, int]]
    question: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def question_text(q: dict[str, Any]) -> str:
    """3a's question line (`e12_agent.Question.text`)."""
    stem = und.render(q["template"], {"x": q["surface"]})
    return agent.Question(q["id"], "train", q["concept"], stem, list(q["candidates"]), int(q["gold"]), q["id"]).text()


def _assemble(parts: Sequence[tuple[str, bool]]) -> tuple[str, list[tuple[int, int]]]:
    text, spans = "", []
    for piece, loss in parts:
        if loss and piece:
            spans.append((len(text), len(text) + len(piece)))
        text += piece
    return text, spans


def render(q: dict[str, Any], form: str, *, observation: str | None = None, said: str | None = None) -> Sequence_:
    """One question in an arm's format: `I` (question, answer), `T` (the full trace), `T-obs` (the trace without its
    observation: S's second epoch). The loss covers the model's turns: what 3a's harness lets the model generate."""
    head = question_text(q)
    gold = " " + q["candidates"][int(q["gold"])].strip()
    if form == "I":
        text, spans = _assemble([(head + "Answer:", False), (gold, True), ("\n", False)])
        return Sequence_(text, spans, q["id"], {"form": form})
    if form not in {"T", "T-obs"}:
        raise ValueError(f"unknown form {form!r}")
    words = sq.relation_phrase(q["relation"])
    act = f" I should recall {q['surface']}, {words}.\nAction: recall[{q['surface']}, {words}]"
    thought = f" The recall says {said}." if said else " The recall names neither option."
    parts = [(head + "Thought:", False), (act, True)]
    if form == "T":
        if observation is None:
            raise ValueError("the T form needs the observation")
        parts.append((f"\nObservation: {observation}\nThought:", False))
    else:
        parts.append(("\nThought:", False))
    parts += [(thought + "\nAnswer:" + gold, True), ("\n", False)]
    text, spans = _assemble(parts)
    return Sequence_(text, spans, q["id"], {"form": form})


class Recaller:
    """3a's `recall[<term>, <relation>]` over the run's own store for the training terms (training twins by surface, seen
    terms through the alias table): the observation text exactly as the tool prints it, and the decoded option."""

    def __init__(self, run: E5Run, store: sqx.LoadedStore, twins: sqx.ItemSet, lexicon: Any) -> None:
        self.builder = sqx.ContextBuilder(twins, run.ontology, lexicon, {"own": store})
        self.toolbox = agent.Toolbox(self.builder, store, run.table)
        self.store = store
        self.lexicon = lexicon

    def __call__(self, q: dict[str, Any]) -> tuple[str, str | None, bool | None]:
        words = sq.relation_phrase(q["relation"])
        observation, record = self.toolbox.call("recall", f"{q['surface']}, {words}")
        if not record.get("parsed"):
            raise ValueError(f"the tool cannot read the training term {q['surface']!r}: {record}")
        _, vector, _ = self.toolbox.term(q["surface"])
        line = self.store.store.decode_role(vector, self.builder.relation_id[q["relation"]])
        if not line.fillers:
            return observation, None, None
        text = self.toolbox.writer.atom_text(line.fillers[0].atom)
        worded = self.lexicon.answer(q["relation"], text)
        options = [c.strip() for c in q["candidates"]]
        said = worded.strip() if worded.strip() in options else None
        return observation, said, (None if said is None else options.index(said) == int(q["gold"]))


def arm_epochs(arm: str, questions: Sequence[dict[str, Any]], recaller: Recaller | None = None, *, epochs: int = EPOCHS
               ) -> tuple[list[list[Sequence_]], dict[str, Any]]:
    """The arm's sequences per epoch (L: built by `corpus_epochs`); the trace statistics."""
    info: dict[str, Any] = {}
    if arm == "I":
        data = [render(q, "I") for q in questions]
        return [data] * epochs, info
    if arm not in {"T", "S"}:
        raise ValueError(f"arm {arm!r} has no question sequences")
    if recaller is None:
        raise ValueError(f"arm {arm} needs the run's store")
    full, without = [], []
    agree = Counter()
    for q in questions:
        observation, said, right = recaller(q)
        agree["gold" if right else "other option" if right is False else "neither option"] += 1
        full.append(render(q, "T", observation=observation, said=said))
        without.append(render(q, "T-obs", said=said))
    info["recall_says"] = dict(agree)
    if arm == "T":
        return [full] * epochs, info
    stages = [full, without, [render(q, "I") for q in questions]]               # S: T, T without observation, I
    return [stages[min(k, len(stages) - 1)] for k in range(epochs)], info


# ---------------------------------------------------------------- encoding and training


@dataclass
class Encoded:
    ids: list[int]
    labels: list[int]
    spans: dict[str, torch.Tensor] | None


def encode(adapter: Any, sequence: Sequence_) -> Encoded:
    """Token ids, labels (−100 off the model's turns) and channel spans (the adapter's linker) of one sequence."""
    tokenizer = adapter.tokenizer
    encoded = tokenizer(sequence.text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = list(encoded["input_ids"]), [tuple(o) for o in encoded["offset_mapping"]]
    if len(ids) > adapter.max_length:
        raise ValueError(f"a training sequence has {len(ids)} tokens, more than max_length {adapter.max_length}")
    labels = [i if any(lo <= o[0] < hi for lo, hi in sequence.spans) and o[1] > o[0] else -100 for i, o in zip(ids, offsets)]
    spans = adapter.spans_fn([sequence.text], [offsets]) if adapter.spans_fn is not None else None
    return Encoded(ids, labels, spans)


def corpus_epochs(run: E5Run, *, sequences: int, length: int, epochs: int, seed: int) -> list[list[Encoded]]:
    """L's data: `sequences` windows of `length` tokens of the run's training corpus per epoch (fresh windows each epoch;
    held-out entries never linked, as in training; LM loss on every token)."""
    from ..data.corpus import TokenCorpus, sample_batch
    corpus = TokenCorpus.open(Path(run.config["data"]["train"]))
    mask = None
    if run.ontology is not None:
        mask = np.ones(int(run.ontology["entry_count"]), dtype=bool)
        mask[list(run.ontology.get("heldout_entries", ()))] = False
    out = []
    for epoch in range(epochs):
        data = []
        for k in range(sequences):
            ids, spans = sample_batch(corpus, seed=seed, step=epoch, micro_step=k, batch=1, length=length,
                                      min_subtokens=run.min_subtokens, entry_mask=mask)
            row = ids[0].tolist()
            data.append(Encoded(row, list(row), spans if run.channel is not None else None))
        out.append(data)
    return out


def _collate(rows: Sequence[Encoded], pad: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
    width = max(len(r.ids) for r in rows)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, :len(r.ids)] = torch.tensor(r.ids); labels[b, :len(r.labels)] = torch.tensor(r.labels); mask[b, :len(r.ids)] = 1
    spans = None
    if rows and rows[0].spans is not None:
        parts = [{**r.spans, "batch": torch.full_like(r.spans["batch"], b)} for b, r in enumerate(rows)]
        spans = {k: torch.cat([p[k] for p in parts]).to(device) for k in parts[0]}
    return ids.to(device), labels.to(device), mask.to(device), spans


@torch.no_grad()
def merge_lora(module: torch.nn.Module) -> int:
    """Fold every existing `LoRALinear` of `module` into its base projection (`W += (α / r) · B A`, the same function) and
    unwrap it; returns how many were merged. A host trained with LoRA (the E9 Qwen3 runs) thus becomes a plain host whose
    weights — its run's adapters included — stay frozen under 3b's new adapters. The SmolLM2 runs have none."""
    from ..integrations.transformers import LoRALinear
    merged = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                delta = child.scale * (child.lora_b.float() @ child.lora_a.float())
                weight = base.weight
                weight.copy_((weight.float() + (delta.T if child.transposed else delta)).to(weight.dtype))
                setattr(parent, name, base)
                merged += 1
    return merged


def attach_lora(model: Any, *, rank: int = RANK, alpha: float = ALPHA) -> list[torch.nn.Parameter]:
    """Freeze the whole `ChannelLM` (host and channel) and add rank-`rank` LoRA adapters (float32) on the host's default
    targets (`add_lora`); a host that already carries adapters has them merged first (`merge_lora`). Returns the new
    adapters' parameters."""
    from ..integrations.transformers import add_lora
    model.merged_lora = merge_lora(model.model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = add_lora(model.model, rank=rank, alpha=alpha, dtype=torch.float32)
    if not params:
        raise ValueError("no LoRA target projection in the host")
    return params


def lora_state(model: Any) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if ".lora_a" in n or ".lora_b" in n}


def train(adapter: Any, epochs: Sequence[Sequence[Encoded]], *, seed: int, lr: float = LR, per_step: int = SEQUENCES_PER_STEP,
          micro: int = 16, max_steps: int | None = None, log: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """LoRA training (module docstring): `per_step` sequences per optimizer step, the loss the mean over the step's loss
    tokens; the data of epoch k in a (seed, k)-seeded order. `max_steps` (smoke tests only) stops early."""
    from ..training.lm import _lr
    model, device = adapter.model, adapter.device
    params = attach_lora(model)
    optimizer = torch.optim.AdamW([{"params": params, "weight_decay": WEIGHT_DECAY}], lr=lr, betas=BETAS, fused=device.type == "cuda")
    steps = [math.ceil(len(data) / per_step) for data in epochs]
    total = sum(steps)
    warmup = max(1, round(WARMUP * total))
    pad = adapter.tokenizer.pad_token_id if adapter.tokenizer.pad_token_id is not None else adapter.tokenizer.eos_token_id
    model.eval()                                       # no dropout; the adapters still receive gradients
    step, started, tokens_seen, loss_tokens_seen = 0, time.monotonic(), 0, 0
    history = []
    for epoch, data in enumerate(epochs):
        order = np.random.default_rng([seed, epoch]).permutation(len(data))
        for k in range(steps[epoch]):
            if max_steps is not None and step >= max_steps:
                break
            t0 = time.monotonic()
            batch = [data[i] for i in order[k * per_step:(k + 1) * per_step]]
            count = sum(sum(1 for x in r.labels[1:] if x != -100) for r in batch)
            rate = _lr(step, total, warmup, lr, MIN_LR_RATIO)
            for group in optimizer.param_groups:
                group["lr"] = rate
            total_loss = 0.0
            for start in range(0, len(batch), micro):
                ids, labels, mask, spans = _collate(batch[start:start + micro], pad, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(ids, attention_mask=mask, spans=spans, labels=labels, reduction="none")
                loss = out["loss"].float().sum() / max(count, 1)
                loss.backward()
                total_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += sum(len(r.ids) for r in batch); loss_tokens_seen += count
            row = {"step": step, "epoch": epoch + 1, "loss": total_loss, "lr": rate, "sequences": len(batch), "tokens": tokens_seen,
                   "loss_tokens": loss_tokens_seen, "seconds": round(time.monotonic() - t0, 3)}
            history.append(row)
            if log is not None:
                log(row)
    seconds = time.monotonic() - started
    return {"steps": step, "planned_steps": total, "warmup_steps": warmup, "tokens": tokens_seen, "loss_tokens": loss_tokens_seen,
            "seconds": seconds, "tokens_per_s": tokens_seen / max(seconds, 1e-9), "first_loss": history[0]["loss"] if history else None,
            "last_loss": history[-1]["loss"] if history else None, "history": history, "parameters": int(sum(p.numel() for p in params))}


# ---------------------------------------------------------------- tests of an arm


def pair_contrast(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    """Per twin pair: the twin contrast of 3a's question rows (`e12_agent.summarize`): 1 when the twins' preferences differ in
    the direction of their frames, 0 the other way, 0.5 on a tie."""
    twins: dict[Any, dict[str, Any]] = defaultdict(dict)
    for r in rows:
        twins[r["meta"]["pair"]][r["meta"]["twin"]] = r
    out = {}
    for pair, both in twins.items():
        if set(both) != {"A", "B"}:
            continue
        sign = 1.0 if both["A"]["gold"] == 0 else -1.0
        a, b = both["A"]["scores"][key], both["B"]["scores"][key]
        z = sign * ((a[0] - a[1]) - (b[0] - b[1]))
        out[str(pair)] = 1.0 if z > 0 else 0.0 if z < 0 else 0.5
    return out


def question_format(run: E5Run, store: sqx.LoadedStore, twins: sqx.ItemSet, *, seed: int = 0) -> dict[str, Any]:
    """The twins in the trained question format without the tool: 3a's `no_tool` prompt (instruction, three direct-answer
    demonstrations, `Question: … Answer:`), both twins of every pair on their first relation, forced choice."""
    lexicon = sqx.lexicon_for_track(twins.track, twins.manifest.get("family") or "smollm2", run.ontology)
    builder = sqx.ContextBuilder(twins, run.ontology, lexicon, {"own": store}, seed=seed)
    toolbox = agent.Toolbox(builder, store, run.table)
    exclude = {int(c["entry"]) for c in twins.concepts if c.get("entry") is not None}
    demos = agent.demonstrations(builder, toolbox, exclude, tool=False)
    pairs = len({c["pair"] for c in twins.concepts})
    qs = agent.questions(twins, None, twin_pairs=pairs, two_hop=0, reverse=0, seed=seed)
    started = time.monotonic()
    with sqx.host_view(run, twins) as (adapter, _):
        scores = agent.answer_scores(adapter, [agent.INSTRUCTION + "\n" + demos + "\n" + q.text() + "Answer:" for q in qs], qs)
    rows = [{"id": q.id, "gold": q.gold, "meta": q.meta, "scores": {"no_tool": s.tolist()}, "correct": float(np.argmax(s) == q.gold)}
            for q, s in zip(qs, scores)]
    units = pair_contrast(rows, "no_tool")
    return {"units": {k: {"contrast": v} for k, v in units.items()}, "mean": {"contrast": float(np.mean(list(units.values()))) if units else None,
                                                                          "accuracy": float(np.mean([r["correct"] for r in rows])) if rows else None,
                                                                          "n": len(units)},
            "seconds": time.monotonic() - started}


def evaluate_arm(run: E5Run, store: sqx.LoadedStore, *, twins: Path, new_words: Path | None, understanding: Path | None, agentic: bool,
                 twin_limit: int | None = None, new_limit: int = NEW_WORDS, understanding_limit: int = UNDERSTANDING_ANCHORS,
                 agent_pairs: int = AGENT_PAIRS, agent_batch: int = 4, max_lines: int | None = 32,
                 log: Callable[[str], None] = print) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The arm's tests (module docstring): per set the phase-A summary (per-unit scores) and timings; the agentic rows."""
    stores = {"own": store}
    out: dict[str, Any] = {}
    item_set = sqx.load_item_set(twins, limit=twin_limit)
    evaluation = sqx.evaluate(run, item_set, sqx.parse_conditions("none,recall:own"), stores, max_lines=max_lines, log=log)
    out["twins"] = {"summary": sqx.summarize(item_set, evaluation["results"], evaluation["records"], evaluation["resolved"]),
                    "timings": evaluation["timings"], "seconds": evaluation["seconds"]}
    if new_words is not None:
        new = sqx.load_item_set(new_words, limit=new_limit)
        evaluation = sqx.evaluate(run, new, sqx.parse_conditions("none"), stores, max_lines=max_lines, log=log)
        out["new_words"] = {"summary": sqx.summarize(new, evaluation["results"], evaluation["records"], evaluation["resolved"]),
                            "timings": evaluation["timings"], "seconds": evaluation["seconds"]}
    if understanding is not None:
        two_hop = sqx.load_item_set(understanding, limit=understanding_limit, families=("two_hop",))
        evaluation = sqx.evaluate(run, two_hop, sqx.parse_conditions("none,recall:own"), stores, max_lines=max_lines, log=log)
        out["two_hop"] = {"summary": sqx.summarize(two_hop, evaluation["results"], evaluation["records"], evaluation["resolved"]),
                          "timings": evaluation["timings"], "seconds": evaluation["seconds"]}
    out["question_format"] = question_format(run, store, item_set)
    rows: list[dict[str, Any]] = []
    if agentic and agent_pairs > 0:
        subset = sqx.load_item_set(twins, limit=agent_pairs)
        result = agent.pilot(run, store, subset, None, twin_pairs=agent_pairs, two_hop=0, reverse=0, batch=agent_batch, log=log)
        rows = result["rows"]
        summary = agent.summarize(rows)
        out["agent"] = {"summary": summary, "units": {k: {"contrast": v} for k, v in pair_contrast(rows, "agent").items()},
                        "units_no_tool": {k: {"contrast": v} for k, v in pair_contrast(rows, "no_tool").items()},
                        "seconds": result["seconds"], "demonstrations": result["demonstrations"]}
    return out, rows


# ---------------------------------------------------------------- one job: train an arm, then test it


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run)
    match = sqx.RUN_NAME.match(run_dir.name)
    model_name = match["model"] if match else run_dir.name
    arm = args.arm
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    output = Path(args.output or run_dir / f"{OUTPUT_PREFIX}{arm}")
    store_dir = Path(args.store) if args.store else run_dir
    config = {"experiment": "e12-traces", "arm": arm, "run": str(run_dir), "model": model_name, "train_items": str(args.train_items),
              "store": str(store_dir), "twins": str(args.twins), "new_words": str(args.new_words) if args.new_words else None,
              "understanding": str(args.understanding) if args.understanding else None, "seed": args.seed,
              "hyperparameters": {"rank": RANK, "alpha": ALPHA, "epochs": EPOCHS, "lr": LR, "sequences_per_step": SEQUENCES_PER_STEP,
                                  "warmup": WARMUP, "min_lr_ratio": MIN_LR_RATIO, "weight_decay": WEIGHT_DECAY, "betas": list(BETAS),
                                  "grad_clip": GRAD_CLIP, "micro_batch": args.micro_batch},
              "smoke": {"question_limit": args.question_limit, "max_steps": args.max_steps, "twin_limit": args.twin_limit,
                        "new_limit": args.new_limit, "understanding_limit": args.understanding_limit, "agent_pairs": args.agent_pairs},
              "batch_size": args.batch_size, "max_length": args.max_length, "max_lines": args.max_lines, "label": args.label}
    import yaml
    run_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    if args.store is None and (run_config.get("channel") or {}).get("mode", "none") == "none":
        raise ValueError(f"{run_dir} has no store (channel mode none): pass --store, the run whose store the tool reads "
                         "(decision 64: a C0′ host reads C5's store of its seed)")
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    from .e9_tracks import ensure_alias_table, track_spec
    track, family = run_config.get("e9_track") or "t5", run_config.get("e9_family") or "smollm2"
    alias_table = args.alias_table or ensure_alias_table(track_spec(track, family))
    seed = int(args.seed if args.seed is not None else (int(match["seed"]) if match else 0))
    store = sqx.load_store(store_dir, "own")              # "own": the store the tool reads (`--store`; default the run's own)
    run = open_run(run_dir, device=args.device, batch_size=args.batch_size, max_length=args.max_length, alias_table=alias_table)
    started = time.monotonic()
    training: dict[str, Any] = {"arm": arm, "trained": arm in TRAINED}
    log_path = output / "training.jsonl"
    log_path.write_text("")

    def log_step(row: dict[str, Any]) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if row["step"] % 10 == 0 or row["step"] == 1:
            print(json.dumps({k: row[k] for k in ("step", "epoch", "loss", "lr", "tokens")}), flush=True)

    if arm in TRAINED:
        manifest, questions, train_twins = load_questions(Path(args.train_items), limit=args.question_limit)
        lexicon = sqx.lexicon_for_track(track, family, run.ontology)
        with sqx.host_view(run, train_twins) as (adapter, _):
            recaller = Recaller(run, store, train_twins, lexicon) if arm in {"T", "S", "L"} else None
            t_epochs, info = arm_epochs("T", questions, recaller) if arm in {"T", "L"} else (None, {})
            if arm == "L":
                t_tokens = [len(adapter.tokenizer(s.text, add_special_tokens=False)["input_ids"]) for s in t_epochs[0]]
                length = min(adapter.max_length, max(8, round(float(np.mean(t_tokens)))))
                info.update(matched_sequences=len(t_tokens), matched_tokens_per_epoch=int(sum(t_tokens)), window_length=length)
                data = corpus_epochs(run, sequences=len(t_tokens), length=length, epochs=EPOCHS, seed=seed * 1000 + 7)
            else:
                texts, info = (t_epochs, info) if arm == "T" else arm_epochs(arm, questions, recaller)
                cache: dict[int, Encoded] = {}
                data = []
                for epoch_texts in texts:
                    rows = []
                    for s in epoch_texts:
                        key = id(s)
                        if key not in cache:
                            cache[key] = encode(adapter, s)
                        rows.append(cache[key])
                    data.append(rows)
                info["examples"] = [s.text for s in texts[0][:2]] + ([texts[-1][0].text] if len(texts) > 1 else [])
            info["sequences_per_epoch"] = [len(d) for d in data]
            info["tokens_per_epoch"] = [int(sum(len(r.ids) for r in d)) for d in data]
            info["loss_tokens_per_epoch"] = [int(sum(sum(1 for x in r.labels[1:] if x != -100) for r in d)) for d in data]
            result = train(adapter, data, seed=seed, micro=args.micro_batch, max_steps=args.max_steps, log=log_step)
        history = result.pop("history")
        training.update(result, **info, questions=len(questions), items_manifest_sha256=manifest.get("sha256"),
                        merged_run_lora=int(getattr(run.model, "merged_lora", 0)))
        torch.save({"state": lora_state(run.model), "rank": RANK, "alpha": ALPHA, "arm": arm, "run": str(run_dir), "seed": seed,
                    "steps": result["steps"]}, output / "adapters.pt")
        training["final_losses"] = [h["loss"] for h in history[-5:]]
    tests, rows = evaluate_arm(run, store, twins=Path(args.twins), new_words=Path(args.new_words) if args.new_words else None,
                               understanding=Path(args.understanding) if args.understanding else None, agentic=arm in AGENTIC,
                               twin_limit=args.twin_limit, new_limit=args.new_limit, understanding_limit=args.understanding_limit,
                               agent_pairs=args.agent_pairs, agent_batch=args.agent_batch, max_lines=args.max_lines)
    seconds = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    header = {"source": run.describe(), "arm": arm, "model": model_name, "label": args.label, "store": store.describe()}
    if rows:
        und.write_jsonl_gz(output / "episodes.jsonl.gz", rows)
    write_json(output / "summary.json", {**header, "training": training, "tests": tests, "seconds": seconds, "peak_gb": peak})
    (output / "report.md").write_text(render_arm_report(header, training, tests))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {"output": str(output), "seconds": seconds, "peak_gb": peak, "training": {k: training.get(k) for k in
                                                                                     ("steps", "tokens", "tokens_per_s", "first_loss", "last_loss")},
            "tests": _headline(tests)}


def _headline(tests: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for condition, block in tests["twins"]["summary"]["conditions"].items():
        for kind in role_items.ITEM_KINDS:
            mean = block.get(kind, {}).get("mean", {})
            if mean.get("n"):
                out[f"twins {condition} {kind}"] = mean.get("contrast")
    if "new_words" in tests:
        out["new words none"] = tests["new_words"]["summary"]["conditions"]["none"]["property"]["mean"].get("accuracy")
    if "two_hop" in tests:
        for condition, block in tests["two_hop"]["summary"]["conditions"].items():
            out[f"two-hop {condition}"] = float(np.mean([u["accuracy"] for u in block["items"]["units"].values()])) if block["items"]["units"] else None
    out["twins question format (no tool)"] = tests["question_format"]["mean"]["contrast"]
    if "agent" in tests:
        s = tests["agent"]["summary"]
        out["agent: format ok"] = s.get("all", {}).get("format_ok")
        out["agent: twin contrast"] = s.get("twins_contrast", {}).get("agent")
    return out


def render_arm_report(header: dict[str, Any], training: dict[str, Any], tests: dict[str, Any]) -> str:
    source = header["source"]
    label = f" — {header['label']}" if header.get("label") else ""
    lines = [f"# E12 3b — arm {header['arm']} on {source['condition']} seed {source['seed']} ({source['size']}){label}", ""]
    if training.get("trained"):
        lines += [f"Training: {training.get('steps')} steps ({training.get('planned_steps')} planned), {training.get('tokens')} tokens "
                  f"({training.get('loss_tokens')} in the loss), {training.get('tokens_per_s', 0):.0f} tokens/s; loss {training.get('first_loss')} → "
                  f"{training.get('last_loss')}. Sequences per epoch {training.get('sequences_per_epoch')}.", ""]
        if training.get("recall_says"):
            lines += [f"Recall in the T traces: {training['recall_says']}.", ""]
    else:
        lines += ["No training (the untrained run).", ""]
    lines += ["| test | value |", "|---|---:|"] + [f"| {k} | {'—' if v is None else f'{v:.3f}'} |" for k, v in _headline(tests).items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the stage report (B1, B2, secondaries)


def discover(runs_root: Path, *, hosts: Sequence[str] | None = None) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """host → arm label (C5: T, I, S, L, base; C5ut: I-ut; C5rf: T-rf, I-rf, L-rf; another host model: `<arm>@<model>`, e.g.
    T@C0p) → seed → summary document."""
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(Path(runs_root).iterdir()):
        match = sqx.RUN_NAME.match(run.name)
        if match is None or (hosts and match["host"] not in hosts):
            continue
        for folder in sorted(run.glob(f"{OUTPUT_PREFIX}*")):
            if not (folder / "summary.json").exists():
                continue
            arm = folder.name[len(OUTPUT_PREFIX):]
            label = arm if match["model"] == "C5" else f"{arm}-{match['model'].removeprefix('C5')}" if match["model"].startswith("C5") \
                else f"{arm}@{match['model']}"
            out[match["host"]][label][int(match["seed"])] = json.loads((folder / "summary.json").read_text())
    return {h: dict(a) for h, a in out.items()}


def units(arms: dict[str, dict[int, dict[str, Any]]], arm: str, test: str = "twins", condition: str = "none", kind: str = "choice",
          metric: str = "contrast") -> dict[int, dict[str, float]]:
    """seed → unit → value of one arm and test (`twins` / `new_words` / `two_hop`: phase-A conditions; `question_format`,
    `agent`, `agent_no_tool`: pair contrasts)."""
    out = {}
    for seed, doc in arms.get(arm, {}).items():
        tests = doc.get("tests", {})
        if test in {"question_format"}:
            block = tests.get(test, {}).get("units")
        elif test in {"agent", "agent_no_tool"}:
            block = tests.get("agent", {}).get("units" if test == "agent" else "units_no_tool")
        else:
            conditions = tests.get(test, {}).get("summary", {}).get("conditions", {})
            found = conditions.get(condition, {})
            block = (found.get(kind) or found.get("items") or found.get("property") or {}).get("units")
        if block:
            out[seed] = {k: float(v[metric]) for k, v in block.items() if isinstance(v.get(metric), (int, float))}
    return out


def analyse(runs_root: Path, *, hosts: Sequence[str] | None = None, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    from .e12_report import _holm, contrast
    found = discover(runs_root, hosts=hosts)
    analysis: dict[str, Any] = {"runs_root": str(runs_root), "hosts": {}}
    for host, arms in sorted(found.items()):
        block: dict[str, Any] = {"arms": {a: sorted(s) for a, s in arms.items()}}
        primary: dict[str, Any] = {}
        for arm in ("I", "S"):
            a, l = units(arms, arm), units(arms, "L")
            if a and l:
                primary[f"B1: {arm} − L"] = contrast(a, l, resamples=resamples, seed=seed)
        _holm(primary)
        block["b1"] = primary
        block["b1_reading"] = b1_reading(primary)
        sec: dict[str, Any] = {}

        def add(name: str, a: dict, b: dict | None) -> None:
            if a and (b is None or b):
                sec[name] = contrast(a, b, resamples=resamples, seed=seed)

        add("T − L (twins, no tool)", units(arms, "T"), units(arms, "L"))
        add("T − I (twins, no tool): do the traces add to the answers?", units(arms, "T"), units(arms, "I"))
        add("S − I (twins, no tool)", units(arms, "S"), units(arms, "I"))
        iut = units(arms, "I-ut")
        if iut:
            add("I-ut − 0.5 (twins, no tool; > 0 = a leak)", {s: {k: v - 0.5 for k, v in u.items()} for s, u in iut.items()}, None)
        for arm in ("I", "S", "T"):
            add(f"{arm} − L (twins cloze, never-trained wording)", units(arms, arm, kind="cloze"), units(arms, "L", kind="cloze"))
            add(f"{arm} − L (twins, trained question format, no tool)", units(arms, arm, "question_format"), units(arms, "L", "question_format"))
            add(f"{arm} − L (twins, recall:own)", units(arms, arm, condition="recall:own"), units(arms, "L", condition="recall:own"))
            add(f"{arm} − L (new words, no tool; accuracy)", units(arms, arm, "new_words", metric="accuracy"),
                units(arms, "L", "new_words", metric="accuracy"))
            add(f"{arm} − L (two-hop, no tool; accuracy)", units(arms, arm, "two_hop", metric="accuracy"),
                units(arms, "L", "two_hop", metric="accuracy"))
            add(f"{arm} − L (two-hop, chained recall; accuracy)", units(arms, arm, "two_hop", condition="recall:own", metric="accuracy"),
                units(arms, "L", "two_hop", condition="recall:own", metric="accuracy"))
        add("L − base (twins, no tool): the size-matched update itself", units(arms, "L"), units(arms, "base"))
        b2: dict[str, Any] = {"test: T − I (agentic twin contrast)": contrast(units(arms, "T", "agent"), units(arms, "I", "agent"),
                                                                             resamples=resamples, seed=seed)
                              if units(arms, "T", "agent") and units(arms, "I", "agent") else {"available": False}}
        b2["descriptive"] = {arm: _agent_means(arms, arm) for arm in (*AGENTIC, "T@C0p", "base@C0p", "T-rf") if arm in arms}
        block["b2"] = b2
        block["secondaries"] = sec
        block["decision64"] = decision64(arms, resamples=resamples, seed=seed)
        block["means"] = {arm: means(arms, arm) for arm in sorted(arms)}
        block["predictions"] = predictions(block)
        analysis["hosts"][host] = block
    return analysis


def _difference(a: dict[int, dict[str, float]], b: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """seed → unit → a − b over the seeds and units both have (an arm contrast as unit values: B1's I − L)."""
    out = {}
    for s in sorted(set(a) & set(b)):
        common = set(a[s]) & set(b[s])
        if common:
            out[s] = {k: a[s][k] - b[s][k] for k in common}
    return out


def _shift(values: dict[int, dict[str, float]], by: float) -> dict[int, dict[str, float]]:
    return {s: {k: v - by for k, v in u.items()} for s, u in values.items()}


def decision64(arms: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Amendment 16.5's secondaries (decision 64): does what LoRA learns depend on the host's channel training (C0′, reading
    C5's store) or on a learned binding (C5rf)? `contrasts` — the predicted ≈ 0 contrasts, each with its equivalence test
    (±`TOOL_MARGIN` with the decoded store in context, ±`B1_MARGIN` for B1's I − L; two one-sided t tests at α = 0.05, no
    multiplicity adjustment: secondaries, never promoted); `controls` — the checks they are read with."""
    from .e12_report import contrast, margin_reading, margin_test
    rows: dict[str, Any] = {}
    controls: dict[str, Any] = {}

    def add(target: dict[str, Any], name: str, a: dict, b: dict | None, margin: float | None = None) -> None:
        if not a or (b is not None and not b):
            return
        result = contrast(a, b, resamples=resamples, seed=seed)
        if margin is not None:
            margin_test(result, margin, kind="equivalence")
        result["reading"] = margin_reading(result)
        target[name] = result

    tool = {"twins recall:own": dict(condition="recall:own"), "agentic twins": dict(test="agent")}
    for label, keys in tool.items():
        add(rows, f"C0′ − C5 host, arm T, {label} (the tool reads C5's store)", units(arms, "T@C0p", **keys), units(arms, "T", **keys), TOOL_MARGIN)
    add(rows, "C5rf − C5, B1's contrast I − L (twins, no tool)", _difference(units(arms, "I-rf"), units(arms, "L-rf")),
        _difference(units(arms, "I"), units(arms, "L")), B1_MARGIN)
    for label, keys in tool.items():
        add(rows, f"C5rf − C5, arm T, {label} (each reads its own store)", units(arms, "T-rf", **keys), units(arms, "T", **keys), TOOL_MARGIN)
    for arm in ("T@C0p", "base@C0p"):
        add(controls, f"{arm} − 0.5 (twins, no tool; no channel: ≈ 0)", _shift(units(arms, arm), 0.5), None)
    add(controls, "T@C0p − base@C0p (agentic twin contrast: the traces teach the C0′ host the protocol)", units(arms, "T@C0p", "agent"),
        units(arms, "base@C0p", "agent"))
    add(controls, "(T − base) on C0′ − (T − base) on C5 (agentic twin contrast)",
        _difference(units(arms, "T@C0p", "agent"), units(arms, "base@C0p", "agent")), _difference(units(arms, "T", "agent"), units(arms, "base", "agent")))
    add(controls, "B1 on C5rf: I-rf − L-rf (twins, no tool)", units(arms, "I-rf"), units(arms, "L-rf"))
    add(controls, "T-rf − L-rf (twins, no tool)", units(arms, "T-rf"), units(arms, "L-rf"))
    return {"contrasts": rows, "controls": controls, "margins": {"tool": TOOL_MARGIN, "b1": B1_MARGIN}}


def _agent_means(arms: dict[str, dict[int, dict[str, Any]]], arm: str) -> dict[str, Any]:
    out: dict[str, list[float]] = defaultdict(list)
    for doc in arms.get(arm, {}).values():
        s = doc.get("tests", {}).get("agent", {}).get("summary", {})
        for key in ("format_ok", "relevant", "answered", "called", "swapped"):
            if key in s.get("all", {}):
                out[key].append(s["all"][key])
        for key in ("agent", "no_tool", "fixed"):
            if key in s.get("twins_contrast", {}):
                out[f"twin contrast {key}"].append(s["twins_contrast"][key])
    return {k: float(np.mean(v)) for k, v in out.items()} | {"seeds": len(arms.get(arm, {}))}


def means(arms: dict[str, dict[int, dict[str, Any]]], arm: str) -> dict[str, float | None]:
    def avg(values: dict[int, dict[str, float]]) -> float | None:
        per = [float(np.mean(list(v.values()))) for v in values.values() if v]
        return float(np.mean(per)) if per else None
    return {"twins none choice": avg(units(arms, arm)), "twins none cloze": avg(units(arms, arm, kind="cloze")),
            "twins recall:own choice": avg(units(arms, arm, condition="recall:own")),
            "twins question format": avg(units(arms, arm, "question_format")),
            "new words none": avg(units(arms, arm, "new_words", metric="accuracy")),
            "two-hop none": avg(units(arms, arm, "two_hop", metric="accuracy")),
            "two-hop recall:own": avg(units(arms, arm, "two_hop", condition="recall:own", metric="accuracy")),
            "agent twin contrast": avg(units(arms, arm, "agent"))}


def b1_reading(rows: dict[str, Any]) -> str:
    significant = [n for n, r in rows.items() if r.get("available") and r.get("p_holm", 1.0) < 0.05 and r["model"]["mean"] > 0]
    negative = [n for n, r in rows.items() if r.get("available") and r.get("p_holm", 1.0) < 0.05 and r["model"]["mean"] < 0]
    if not any(r.get("available") for r in rows.values()):
        return "not available"
    if significant:
        return "internalization shown by " + " and ".join(s.split(": ")[1] for s in significant) + " (Holm p < 0.05)"
    if negative:
        return "training lowered the no-tool twin contrast below the control: " + ", ".join(negative)
    return "not shown: neither I − L nor S − L is significant"


def predictions(block: dict[str, Any]) -> dict[str, Any]:
    """The §12 predictions, checked: B1 > 0 but ≤ 0.65; S ≥ I; I-ut ≈ 0.5; T's well-formed calls ≥ 0.9."""
    m = block["means"]
    out = {}
    for arm in ("I", "S"):
        if m.get(arm, {}).get("twins none choice") is not None:
            out[f"{arm} twins none ≤ 0.65"] = m[arm]["twins none choice"] <= 0.65
    if m.get("S", {}).get("twins none choice") is not None and m.get("I", {}).get("twins none choice") is not None:
        out["S ≥ I"] = m["S"]["twins none choice"] >= m["I"]["twins none choice"]
    leak = block["secondaries"].get("I-ut − 0.5 (twins, no tool; > 0 = a leak)")
    if leak and leak.get("available"):
        out["I-ut ≈ 0.5 (CI includes 0.5)"] = leak["model"]["ci_low"] <= 0 <= leak["model"]["ci_high"]
    t = block["b2"]["descriptive"].get("T", {})
    if "format_ok" in t:
        out["T well-formed calls ≥ 0.9"] = t["format_ok"] >= 0.9
    d64 = block.get("decision64") or {}
    for prefix, label in (("C0′ − C5 host", "16.5: C0′ host ≈ C5 host with the decoded store (arm T, ±0.05)"),
                          ("C5rf − C5", "16.5: C5rf ≈ C5 (B1's I − L ±0.075; arm T ±0.05)")):
        tests = [r for n, r in d64.get("contrasts", {}).items() if n.startswith(prefix) and r.get("margin")]
        if tests:
            out[label] = all(r["margin"]["shown"] for r in tests)
    for arm in ("T@C0p", "base@C0p"):
        r = d64.get("controls", {}).get(f"{arm} − 0.5 (twins, no tool; no channel: ≈ 0)")
        if r and r.get("available"):
            out[f"16.5: {arm} without the tool ≈ 0.5 (CI includes 0.5)"] = r["model"]["ci_low"] <= 0 <= r["model"]["ci_high"]
    return out


def render_report(analysis: dict[str, Any], *, title: str, label: str | None = None) -> str:
    from .e12_report import _cell
    lines = [f"# {title}" + (f" — {label}" if label else ""), "",
             "Pre-registration: `experiments/e12-self-query/preregistration.md` §12, amendments 16.3 and 16.5. Contrasts: twin pairs (or items) × "
             "seeds crossed model (Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm over B1's two contrasts.", ""]
    if label:
        lines += [f"**{label}: these numbers do not count toward the pre-registered endpoints.**", ""]
    for host, block in analysis["hosts"].items():
        lines += [f"## {host}", "", f"Arms × seeds: {block['arms']}", "", "### B1 — internalization (twin contrast without the tool, choice)", "",
                  "| contrast | estimate |", "|---|---|"] + [f"| {n} | {_cell(r)} |" for n, r in block["b1"].items()]
        lines += ["", f"Reading: {block['b1_reading']}", "", "### B2 — tool use learned (agentic twins, first 100 pairs)", "",
                  f"Test T − I: {_cell(block['b2']['test: T − I (agentic twin contrast)'])}", "",
                  "| arm | " + " | ".join(("format_ok", "relevant", "answered", "twin contrast agent", "twin contrast no_tool",
                                            "twin contrast fixed")) + " |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, d in block["b2"]["descriptive"].items():
            lines.append(f"| {arm} | " + " | ".join(_fmt(d.get(k)) for k in ("format_ok", "relevant", "answered", "twin contrast agent",
                                                                             "twin contrast no_tool", "twin contrast fixed")) + " |")
        lines += ["", "### Means by arm (seed-averaged)", "", "| arm | " + " | ".join(next(iter(block["means"].values())).keys()) + " |",
                  "|---|" + "---:|" * len(next(iter(block["means"].values())))]
        lines += [f"| {arm} | " + " | ".join(_fmt(v) for v in m.values()) + " |" for arm, m in block["means"].items()]
        lines += ["", "### Secondaries (family SB; reported, never promoted)", "", "| contrast | estimate |", "|---|---|"]
        lines += [f"| {n} | {_cell(r)} |" for n, r in block["secondaries"].items()]
        d64 = block.get("decision64") or {}
        if d64.get("contrasts") or d64.get("controls"):
            lines += ["", "### Decision 64 (amendment 16.5): does LoRA learn to use the decoded store, whatever the host or binding?", "",
                      "Equivalence: two one-sided t tests at α = 0.05 (90% CI inside ±margin; ±0.05 with the decoded store in context, ±0.075 "
                      "for B1's I − L); secondaries, never promoted.", "", "| contrast | estimate | 90% CI | reading |", "|---|---|---|---|"]
            for n, r in d64.get("contrasts", {}).items():
                ci = f"[{r['margin']['ci90_low']:+.4f}, {r['margin']['ci90_high']:+.4f}]" if r.get("margin") else "—"
                lines.append(f"| {n} | {_cell(r)} | {ci} | {r['reading']} |")
            lines += ["", "| control | estimate | reading |", "|---|---|---|"]
            lines += [f"| {n} | {_cell(r)} | {r['reading']} |" for n, r in d64.get("controls", {}).items()]
        lines += ["", "Predictions (§12, 16.5): " + ", ".join(f"{k}: {'yes' if v else 'no'}" for k, v in block["predictions"].items()), ""]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    config = {"experiment": "e12-traces-report", "runs": str(args.runs), "hosts": args.hosts, "label": args.label}
    if args.overwrite:
        for name in ("analysis.json", "report.md", "resolved_config.yaml", "manifest.json"):
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = start_output(args.output, config)
    started = time.monotonic()
    analysis = analyse(args.runs, hosts=args.hosts, resamples=args.resamples)
    write_json(args.output / "analysis.json", json_ready(analysis))
    (args.output / "report.md").write_text(render_report(analysis, title=args.title, label=args.label))
    finish_output(args.output, config, git_at_start=git_at_start, device="cpu", seconds=round(time.monotonic() - started, 1))
    return {"output": str(args.output), "hosts": list(analysis["hosts"])}


# ---------------------------------------------------------------- queue commands (printed; never queued here)


def job_commands(stage: str = "t5", *, host: str = "SmolLM2-360M", seeds: Sequence[int] = (1, 2, 3), priority: float = 54.4497,
                 python: str = "$PY", root: Path = ROOT) -> list[dict[str, Any]]:
    """One GPU job per (run, arm): C5 × {T, I, S, L, base}, C5ut × {I} (I-ut, no agentic episodes: its tool reads a role-blind
    store), seeds 1–3; names `<stage>-<run>-<folder>`."""
    items = root / "items"
    jobs = []
    for model, arms in (("C5", ("T", "I", "S", "L", "base")), ("C5ut", ("I",))):
        for s in seeds:
            run = root / "runs" / stage / f"{host}-full-{model}-s{s}"
            for arm in arms:
                command = [python, "-m", "vsa_embed.experiments.e12_traces", "run", "--run", str(run), "--arm", arm,
                           "--train-items", str(ITEMS), "--twins", str(items / "role-twins-t5-smollm2-v1"),
                           "--new-words", str(items / "new-words-t5-smollm2-v2"), "--understanding", str(items / "understanding-t5-smollm2-v1"),
                           "--batch-size", "24", "--max-length", "1024", *(["--agent-pairs", "0"] if model != "C5" else []), "--overwrite"]
                jobs.append({"name": f"{stage}-{run.name}-{OUTPUT_PREFIX}{arm}", "priority": priority, "lane": "gpu", "command": command})
    return jobs


# GPU-h per job (upper bounds; amendment 16.3's measured rates: training 0.031 GPU-h per M tokens with padding, the tests
# ≈ 7 min, the agentic episodes ≈ 10 min per arm and seed).
D64_GPU_H = {"T": 0.45, "base": 0.28, "I": 0.17, "L": 0.28}


def decision64_jobs(stage: str = "t5", *, host: str = "SmolLM2-360M", seeds: Sequence[int] = (1, 2, 3), priority: float = 54.44971,
                    python: str = "$PY", root: Path = ROOT) -> list[dict[str, Any]]:
    """Amendment 16.5's arms (`D64_ARMS`), seeds 1–3: the C0′ host (T, base; the tool reads C5's store of the seed) at
    `priority`, C5rf (T, I, L; I without agentic episodes — B2 is read on C5) at `priority` + 0.00001."""
    items = root / "items"
    jobs = []
    for k, (model, arms) in enumerate(D64_ARMS.items()):
        for s in seeds:
            run = root / "runs" / stage / f"{host}-full-{model}-s{s}"
            store = [["--store", str(root / "runs" / stage / f"{host}-full-{D64_STORE[model]}-s{s}")]] if model in D64_STORE else []
            for arm in arms:
                command = [python, "-m", "vsa_embed.experiments.e12_traces", "run", "--run", str(run), "--arm", arm,
                           *(x for pair in store for x in pair), "--train-items", str(ITEMS), "--twins", str(items / "role-twins-t5-smollm2-v1"),
                           "--new-words", str(items / "new-words-t5-smollm2-v2"), "--understanding", str(items / "understanding-t5-smollm2-v1"),
                           "--batch-size", "24", "--max-length", "1024", *(["--agent-pairs", "0"] if arm == "I" else []), "--overwrite"]
                jobs.append({"name": f"{stage}-{run.name}-{OUTPUT_PREFIX}{arm}", "priority": round(priority + 0.00001 * k, 5), "lane": "gpu",
                             "command": command, "gpu_h": D64_GPU_H[arm]})
    return jobs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    it = sub.add_parser("items", help="build the training questions (CPU, once)")
    it.add_argument("--output", type=Path, default=ITEMS); it.add_argument("--track", default="t5"); it.add_argument("--family", default="smollm2")
    it.add_argument("--test-twins", type=Path, default=ROOT / "items" / "role-twins-t5-smollm2-v1")
    it.add_argument("--pairs", type=int, default=TRAIN_PAIRS); it.add_argument("--seed", type=int, default=ITEM_SEED)
    it.add_argument("--name-seed", type=int, default=NAME_SEED); it.add_argument("--contamination-tokens", type=int, default=20_000_000)
    ru = sub.add_parser("run", help="train one arm on one run, then test it (GPU)")
    ru.add_argument("--run", type=Path, required=True); ru.add_argument("--arm", required=True, choices=ARMS)
    ru.add_argument("--train-items", type=Path, default=ITEMS); ru.add_argument("--twins", type=Path, default=ROOT / "items" / "role-twins-t5-smollm2-v1")
    ru.add_argument("--new-words", type=Path, default=None); ru.add_argument("--understanding", type=Path, default=None)
    ru.add_argument("--output", type=Path, default=None); ru.add_argument("--alias-table", type=Path, default=None)
    ru.add_argument("--store", type=Path, default=None,
                    help="the run whose store the tool reads (default: the run's own; decision 64: a C0′ host reads C5's store)")
    ru.add_argument("--device", default=None); ru.add_argument("--seed", type=int, default=None, help="default: the run's seed")
    ru.add_argument("--batch-size", type=int, default=24); ru.add_argument("--max-length", type=int, default=1024)
    ru.add_argument("--micro-batch", type=int, default=16, help="sequences per forward (memory only; 32 per optimizer step)")
    ru.add_argument("--agent-batch", type=int, default=4)
    ru.add_argument("--question-limit", type=int, default=None, help="smoke tests only: the first N questions per source")
    ru.add_argument("--max-steps", type=int, default=None, help="smoke tests only: stop after N optimizer steps")
    ru.add_argument("--twin-limit", type=int, default=None, help="smoke tests only: the first N test pairs")
    ru.add_argument("--new-limit", type=int, default=NEW_WORDS); ru.add_argument("--understanding-limit", type=int, default=UNDERSTANDING_ANCHORS)
    ru.add_argument("--agent-pairs", type=int, default=AGENT_PAIRS, help="agentic arms: the first N pairs (0: no episodes)")
    ru.add_argument("--max-lines", type=int, default=32, help="recalled lines per call in the fixed pipeline (phase A: 32)")
    ru.add_argument("--label", default=None)
    ru.add_argument("--overwrite", action="store_true")
    re_ = sub.add_parser("report", help="B1, B2 and the secondaries across a stage's runs (CPU)")
    re_.add_argument("--runs", type=Path, required=True); re_.add_argument("--output", type=Path, required=True)
    re_.add_argument("--hosts", nargs="*", default=None); re_.add_argument("--resamples", type=int, default=2000)
    re_.add_argument("--title", default="E12 3b — learning from tool-using traces"); re_.add_argument("--label", default=None)
    re_.add_argument("--overwrite", action="store_true")
    co = sub.add_parser("commands", help="print the GPU jobs (never queues)")
    co.add_argument("--stage", default="t5"); co.add_argument("--priority", type=float, default=None)
    co.add_argument("--set", default="3b", choices=("3b", "decision64"), help="3b: §12's jobs; decision64: amendment 16.5's arms")
    args = parser.parse_args(argv)
    if args.command == "items":
        from .e9_tracks import track_spec
        ctx = und.BuildContext.for_track(args.track, args.family)
        spec = track_spec(args.track, args.family)
        texts = edit._contamination_texts([spec.data_root / "train"], ctx.tokenizer_name, args.contamination_tokens)
        manifest = build_training_items(ctx, args.output, test_twins=args.test_twins, items_roots=[ROOT / "items"], pairs=args.pairs,
                                        seed=args.seed, name_seed=args.name_seed, contamination_texts=texts)
        print(json.dumps({"output": str(args.output), "counts": manifest["counts"], "checks": manifest["checks"]}, indent=2))
    elif args.command == "run":
        print(json.dumps(run_arm(args), indent=2, default=str))
    elif args.command == "report":
        print(json.dumps(run_report(args)))
    else:
        jobs = (decision64_jobs(args.stage, priority=args.priority or 54.44971) if args.set == "decision64"
                else job_commands(args.stage, priority=args.priority or 54.4497))
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "gpu_h": j.get("gpu_h"), "command": " ".join(j["command"])}
                          for j in jobs], indent=2))


if __name__ == "__main__":
    main()
