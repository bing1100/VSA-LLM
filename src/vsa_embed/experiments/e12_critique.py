"""E12 — 3c: calibrated self-critique with a null-world control (author decision 62; pre-registration
`experiments/e12-self-query/preregistration.md` §13, amendment 16.3). Evaluation only, on finished E9 runs.

**Beliefs.** A belief is a decoded edge of the recall tool (`vsa_embed.self_query`, the run's own static store, typed
cleanup, the operator's primary unbinding): for an item of relation r the top-m fillers of slot r (m = r's multiplicity in
the term's frame; a role-blind store: the type-restricted bundle readout), the item's belief being the best of them that
is one of the item's options. Its **confidence p** is calibrated per relation: P(decode correct | cleanup cosine, margin to
the next candidate, log frame size), a logistic score of the three features followed by isotonic regression of correctness
on that score (`Calibrator`), fitted on the store's seen entries (≤ 3,000, `e12_self_query.seen_entries`; decode
correctness is known there) and applied to test terms. Calibration is reported on held-out terms and new words: ECE (15
equal-width bins), Brier score, AUROC of correct against wrong decodes — against the store and against the world.

**Behaviour.** The model's answer without the tool (phase A's `none`: the channel only), its confidence the softmax of the
candidates' mean PMI over the item's templates. **Evidence** (T5 held-out terms only: never linked in training, present in
the evaluation corpus): one evaluation-corpus sentence naming the term and exactly one of the item's options
(`evidence` builds them once: `experiments/e12-self-query/items/critique-evidence-t5-v1`), put in context; new words have
none.

**Null world.** The tool reads a corrupted store: each test term's frame replaced by a same-type false frame — E9's
`random_frame` (fillers of each relation redrawn frequency-weighted; new words: the frame stored with the item set, held-out
entries: `e9_tracks.random_frames`) whose tested edge (the item's gold) is redrawn among the item's distractors
(frequency-weighted), so the false belief is an option a method can adopt; twins: the roles swapped within the pair (the
partner's frame). Composed and decoded exactly as the real store; the channel's injection and the behaviour stay the real
world's. Null frames are seeded by term, the same for every run.

**Items.** New words: the first 300 v2 words' property items, split by word into a seeded dev half and a test half; held-out
terms: the T5 WP-C7 zero-shot property items of the held-out split (`zeroshot_property.jsonl`, three paraphrases, four
options); twins (secondary): the test twins' `choice` items.

**This job (GPU)** scores, per item set, `none`, `recall:own` (the true recall in context), `null` (the false recall in
context) and, on held-out terms, `evidence`; it records every item's beliefs in both worlds and the calibration data.
**The rule loop, K1 and K2 are computed by `report` (CPU)**: θ chosen on the dev half, the loop answering with the belief
when p ≥ θ and else with the behaviour, flagging the items whose available signals disagree (flagged items are the first
abstained), each method abstaining on its own lowest-confidence 20%.

**Text competitors (decision 64, amendment 16.5).** On the new words and the held-out terms (K1's pool) the job also scores
phase A's `symbolic` (the gold relations as text) and `definition` (the E11 prose definition) contexts — on a run whose store
carries roles (`--competitors auto`; the C5ut job, the null world's role-blind control, skips them) — and records the prompt
tokens every context adds to each item (`tokens`). `report` runs **the same rule loop with each text in the prompt instead
of the store decode** (`text_loop`: the belief is the host's answer with the text in context, its confidence the softmax
calibrated on the dev half, θ chosen on the dev half by the same rule; `loop_recall_text` = the store's decode as text, read
by the host) and tests **K1b = loop(store) − loop(definition)** for non-inferiority at −0.05 (accuracy at 80% coverage),
read with the prompt tokens each loop adds (the store's rule loop: none). K2 is unchanged.

**Model loop** (`loop`, secondary; Qwen3-1.7B-Base C5, seeds 1–2): a few-shot critique prompt (the question, the model's own
answer, the recall with p, the evidence sentence or "none") scored by forced choice over the decisions (Keep, Revise,
Unsure) and, after `Revise to`, over the other options.

Outputs: `RUN/self-query-critique/` and `RUN/self-query-critique-loop/` (`summary.json`, `items.jsonl.gz`, `report.md`,
`resolved_config.yaml`, `manifest.json`); the report `experiments/e12-self-query/report/<stage>-3c/`.

    python -m vsa_embed.experiments.e12_critique evidence [--output DIR]                                         (CPU, once)
    python -m vsa_embed.experiments.e12_critique evaluate --run RUN [--sets twins,new,heldout]                    (GPU)
    python -m vsa_embed.experiments.e12_critique loop --run RUN [--store RUN] [--items 300]                      (GPU)
    python -m vsa_embed.experiments.e12_critique report --runs experiments/e9-retrofit/runs/t5 --output DIR        (CPU)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .. import self_query as sq
from ..span_channel import normalize_alias
from . import e9_ontology_edit as edit
from . import e9_understanding as und
from . import e12_self_query as sqx
from .e5_common import E5Run, finish_output, json_ready, open_run, start_output, write_json

SCHEMA = "e12-critique/1"
ROOT = Path("experiments/e9-retrofit")
E12_ROOT = Path("experiments/e12-self-query")
OUTPUT = "self-query-critique"
LOOP_OUTPUT = "self-query-critique-loop"
WPC7 = Path("experiments/t5-enterprise-glossary/items/zeroshot_property.jsonl")
EVIDENCE = E12_ROOT / "items" / "critique-evidence-t5-v1"
EVAL_DOCS = Path("/home/bhux/data/vsa-llm/tracks/t5-glossary/v1/docs/eval.jsonl.gz")
RESULT_FILES = ("summary.json", "items.jsonl.gz", "report.md", "resolved_config.yaml", "manifest.json")
SETS = ("twins", "new", "heldout")
# Pre-registered (§13).
ECE_BINS = 15
COVERAGE = 0.8
NEW_WORDS = 300
SEEN_CAP = 3000
# Fixed before any run (amendment 16.3).
THETAS = tuple(round(0.05 * k, 2) for k in range(21))
SPLIT_SEED = 0
NULL_SEED = 0
MIN_EDGES = 50                 # a relation with fewer decoded seen edges (or < 5 of a class) uses the pooled calibrator
L2 = 1.0
# Decision 64 (amendment 16.5).
COMPETITORS = ("symbolic", "definition")       # phase A's text conditions, on K1's pool (new words, held-out terms)
COMPETITOR_SETS = ("new", "heldout")
TEXT_LOOPS = {"loop_recall_text": "recall:own", "loop_symbolic": "symbolic", "loop_definition": "definition"}
K1B_MARGIN = 0.05              # K1b's non-inferiority margin (accuracy at 80% coverage)


# ---------------------------------------------------------------- calibration


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def logistic_fit(x: np.ndarray, y: np.ndarray, *, l2: float = L2, iterations: int = 50) -> dict[str, Any]:
    """Ridge logistic regression (Newton) on standardized features; the bias is not penalized."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = np.hstack([np.ones((len(x), 1)), (x - mu) / sd])
    w = np.zeros(z.shape[1])
    penalty = np.full(z.shape[1], l2); penalty[0] = 0.0
    for _ in range(iterations):
        p = _sigmoid(z @ w)
        gradient = z.T @ (p - y) + penalty * w
        hessian = (z * (p * (1 - p))[:, None]).T @ z + np.diag(penalty) + 1e-9 * np.eye(z.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def logistic_score(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return z @ np.asarray(model["w"][1:]) + model["w"][0]


def isotonic_fit(score: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    """Isotonic (increasing) regression of `y` on `score`; one value per distinct score."""
    from scipy.optimize import isotonic_regression
    order = np.argsort(score, kind="stable")
    fitted = isotonic_regression(y[order].astype(float), increasing=True).x
    xs, inverse = np.unique(score[order], return_inverse=True)
    values = np.zeros(xs.size)
    np.add.at(values, inverse, fitted)
    values /= np.bincount(inverse)
    return {"x": xs.tolist(), "y": np.maximum.accumulate(values).tolist()}


def isotonic_predict(model: dict[str, list[float]], score: np.ndarray) -> np.ndarray:
    return np.interp(score, np.asarray(model["x"]), np.asarray(model["y"]))


class Calibrator:
    """P(decode correct | cosine, margin, log frame size) per relation (module docstring); the pooled model for relations
    with too few decoded seen edges."""

    def __init__(self) -> None:
        self.models: dict[int, dict[str, Any]] = {}
        self.pooled: dict[str, Any] | None = None
        self.info: dict[str, Any] = {}

    @staticmethod
    def _one(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if y.min() == y.max():
            return {"constant": float(y.mean())}
        logistic = logistic_fit(x, y)
        return {"logistic": logistic, "isotonic": isotonic_fit(logistic_score(logistic, x), y)}

    @staticmethod
    def _apply(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
        if "constant" in model:
            return np.full(len(x), model["constant"])
        return isotonic_predict(model["isotonic"], logistic_score(model["logistic"], x))

    def fit(self, x: np.ndarray, y: np.ndarray, relations: np.ndarray) -> "Calibrator":
        x, y, relations = np.asarray(x, float), np.asarray(y, float), np.asarray(relations)
        self.pooled = self._one(x, y)
        for r in sorted(set(relations.tolist())):
            keep = relations == r
            positives = int(y[keep].sum())
            if keep.sum() >= MIN_EDGES and min(positives, int(keep.sum()) - positives) >= 5:
                self.models[int(r)] = self._one(x[keep], y[keep])
        self.info = {"edges": int(len(y)), "correct": float(y.mean()) if len(y) else None, "own_models": sorted(self.models),
                     "pooled_relations": sorted(int(r) for r in set(relations.tolist()) if int(r) not in self.models)}
        return self

    def predict(self, x: np.ndarray, relations: np.ndarray) -> np.ndarray:
        x, relations = np.asarray(x, float).reshape(-1, 3), np.asarray(relations)
        out = np.zeros(len(x))
        for r in set(relations.tolist()):
            keep = relations == r
            out[keep] = self._apply(self.models.get(int(r), self.pooled), x[keep])
        return np.clip(out, 0.0, 1.0)

    def describe(self) -> dict[str, Any]:
        return {"info": self.info, "models": {str(k): v for k, v in self.models.items()}, "pooled": self.pooled}


def decoded(store: sq.RecallStore, vectors: torch.Tensor, relations: Sequence[int], ms: Sequence[int]) -> list[list[tuple[int, float, float]]]:
    """Per query (store vector, relation, multiplicity m): the top-m decoded fillers as (atom, cosine, margin to the
    (m+1)-th candidate) — the slot's read-back (`decode_slots` / `decode_role`)."""
    if not len(relations):
        return []
    scores = store.filler_scores(torch.as_tensor(list(relations), dtype=torch.long), vectors, cleanup="typed")
    out = []
    for row, m in zip(scores, ms):
        finite = int(torch.isfinite(row).sum())
        k = min(int(m) + 1, finite)
        if k <= 0:
            out.append([]); continue
        values, index = torch.topk(row, k)
        values, index = values.tolist(), index.tolist()
        nxt = values[int(m)] if len(values) > int(m) else None
        out.append([(a, s, s - nxt if nxt is not None else s) for a, s in list(zip(index, values))[:int(m)]])
    return out


def frame_slots(frame: Sequence[tuple[int, int]]) -> list[tuple[int, int, list[int]]]:
    """(relation, multiplicity, gold fillers) per distinct relation of a frame, in first-occurrence order."""
    gold: dict[int, list[int]] = {}
    for r, f in frame:
        gold.setdefault(int(r), []).append(int(f))
    return [(r, len(v), v) for r, v in gold.items()]


def edge_records(store: sq.RecallStore, vectors: torch.Tensor, frames: Sequence[Sequence[tuple[int, int]]],
                 worlds: Sequence[Sequence[tuple[int, int]]] | None = None) -> dict[str, np.ndarray]:
    """Every decoded edge of each store vector's frame: features (cosine, margin, log degree), relation, correct against
    the store's frame, and (with `worlds`) against the world's frame."""
    queries, owners = [], []
    for i, frame in enumerate(frames):
        for r, m, _ in frame_slots(frame):
            queries.append((i, r, m))
    if not queries:
        return {"x": np.zeros((0, 3)), "relation": np.zeros(0, int), "store": np.zeros(0), "world": np.zeros(0)}
    found = decoded(store, vectors[[q[0] for q in queries]], [q[1] for q in queries], [q[2] for q in queries])
    x, rel, right, world = [], [], [], []
    for (i, r, m), fillers in zip(queries, found):
        gold = {int(f) for q, f in frames[i] if int(q) == r}
        true = {int(f) for q, f in worlds[i] if int(q) == r} if worlds is not None else gold
        for atom, cos, margin in fillers:
            x.append((cos, margin, math.log(len(frames[i])))); rel.append(r)
            right.append(float(atom in gold)); world.append(float(atom in true))
    return {"x": np.asarray(x, float), "relation": np.asarray(rel, int), "store": np.asarray(right), "world": np.asarray(world)}


def fit_calibrator(store: sqx.LoadedStore, *, cap: int = SEEN_CAP, seed: int = 0) -> tuple[Calibrator, dict[str, np.ndarray]]:
    """The calibrator of a store, fitted on its seen entries' decoded edges."""
    entries = sqx.seen_entries(store.ontology, cap=cap, seed=seed)
    vectors = store.store.entry_vectors()[torch.tensor(entries, dtype=torch.long)] if entries else torch.zeros(0, store.store.atomics.shape[1])
    frames = [store.store.frame(e) for e in entries]
    data = edge_records(store.store, vectors, frames)
    return Calibrator().fit(data["x"], data["store"], data["relation"]), data | {"entries": np.asarray(entries)}


def ece(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float | None:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return None
    index = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        keep = index == b
        if keep.any():
            total += keep.mean() * abs(y[keep].mean() - p[keep].mean())
    return float(total)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> list[dict[str, float]]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    index = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": b, "low": b / bins, "high": (b + 1) / bins, "n": int((index == b).sum()), "confidence": float(p[index == b].mean()),
             "accuracy": float(y[index == b].mean())} for b in range(bins) if (index == b).any()]


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    p, y = np.asarray(p, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(p)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {"n": int(len(p)), "accuracy": float(y.mean()) if len(y) else None, "mean_p": float(p.mean()) if len(p) else None,
            "ece": ece(p, y), "brier": float(np.mean((p - y) ** 2)) if len(p) else None, "auroc": auroc(p, y),
            "reliability": reliability(p, y) if len(p) else []}


# ---------------------------------------------------------------- items, evidence and null frames


def wpc7_heldout(path: Path, ontology: dict[str, Any], table: Any, *, split: str = "heldout") -> sqx.ItemSet:
    """The WP-C7 zero-shot property items of the held-out terms as a phase-A item set (one item per group: its three
    paraphrases as templates, four options; PMI against "this"); each term resolved to its entry through the alias table."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["split"] == split:
            groups.setdefault(r["group"], []).append(r)
    held = {int(e) for e in ontology.get("heldout_entries", ())}
    concepts: dict[str, dict[str, Any]] = {}
    prompts = []
    for group, members in groups.items():
        members = sorted(members, key=lambda r: r["paraphrase"])
        first = members[0]
        entry = table.alias_to_entry.get(normalize_alias(first["surface"]))
        if entry is None or (split == "heldout" and int(entry) not in held):
            continue
        cid = f"ho-{entry}"
        concepts.setdefault(cid, {"concept": cid, "surface": first["surface"], "entry": int(entry), "subset": split})
        templates = [r["prompt"].replace(first["surface"], "{x}", 1) for r in members]
        item = {"id": group, "concept": cid, "test": "property", "relation": first["relation"], "templates": templates,
                "candidates": list(first["choices"]), "gold": int(first["label"]), "null": "this"}
        prompts.append(sqx.Prompt(group, cid, templates, {"x": first["surface"]}, {"x": "this"}, list(first["choices"]), int(first["label"]),
                                  False, {"concept": cid, "test": "property", "relation": first["relation"], "edge_kind": "heldout"}, item))
    return sqx.ItemSet("heldout", Path(path), {"track": "t5", "family": "smollm2", "source": str(path)}, list(concepts.values()), prompts)


SENTENCE = re.compile(r"(?<=[.!?])\s+")


def evidence_sentences(item_set: sqx.ItemSet, docs: Path) -> list[dict[str, Any]]:
    """Per item: the first evaluation-corpus sentence (document order; lines split at sentence ends) that names the term and
    exactly one of the item's options (a whole-word match); none when no sentence does."""
    sentences = []
    with gzip.open(docs, "rt") as handle:
        for d, line in enumerate(handle):
            for part in json.loads(line)["text"].split("\n"):
                for s in SENTENCE.split(part):
                    if s.strip():
                        sentences.append((d, s.strip()))
    surfaces = {c["concept"]: c["surface"] for c in item_set.concepts}
    by_term: dict[str, list[int]] = defaultdict(list)
    wanted = set(surfaces.values())
    for i, (_, s) in enumerate(sentences):
        for term in wanted:
            if term in s:
                by_term[term].append(i)
    out = []
    for p in item_set.prompts:
        options = [c.strip() for c in p.candidates]
        found = None
        for i in by_term.get(surfaces[p.concept], []):
            text = sentences[i][1]
            hits = [k for k, o in enumerate(options) if re.search(r"(?<![\w-])" + re.escape(o) + r"(?![\w-])", text)]
            if len(hits) == 1:
                found = {"sentence": text, "document": sentences[i][0], "option": hits[0]}
                break
        out.append({"item": p.id, "concept": p.concept, "relation": p.row["relation"], "gold": p.gold, **(found or {"sentence": None})})
    return out


def build_evidence(output: Path, *, items: Path = WPC7, docs: Path = EVAL_DOCS) -> dict[str, Any]:
    """The evidence sentences of the T5 held-out WP-C7 items (CPU, once; the corpus is synthetic)."""
    from .e9_tracks import ensure_alias_table, track_spec
    spec = track_spec("t5", "smollm2")
    import vsa_embed.evaluation.channel_probes as cp
    table = cp.load_alias_table(ensure_alias_table(spec))
    ontology = torch.load(spec.ontology, weights_only=False)
    item_set = wpc7_heldout(items, ontology, table)
    rows = evidence_sentences(item_set, docs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    und.write_jsonl_gz(output / "evidence.jsonl.gz", rows)
    present = [r for r in rows if r.get("sentence")]
    manifest = {"schema": SCHEMA, "kind": "evidence", "items": str(items), "items_sha256": hashlib.sha256(Path(items).read_bytes()).hexdigest(),
                "docs": str(docs), "docs_sha256": hashlib.sha256(Path(docs).read_bytes()).hexdigest(), "terms": len(item_set.concepts),
                "counts": {"items": len(rows), "with_evidence": len(present), "evidence_states_gold": sum(r["option"] == r["gold"] for r in present),
                           "by_relation": dict(sorted(Counter(r["relation"] for r in present).items()))},
                "rule": "the first evaluation-corpus sentence (document order; each line split at sentence ends) that contains the term's "
                        "surface and exactly one of the item's options as a whole word sequence",
                "sha256": {"evidence.jsonl.gz": hashlib.sha256((output / "evidence.jsonl.gz").read_bytes()).hexdigest()}}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_evidence(path: Path = EVIDENCE) -> dict[str, dict[str, Any]]:
    return {r["item"]: r for r in und.read_jsonl(Path(path) / "evidence.jsonl")}


class Options:
    """Option text ↔ atomic of a relation (the options are the relation's answer wording of its fillers)."""

    def __init__(self, ontology: dict[str, Any], lexicon: Any) -> None:
        names, relation_names = ontology["atomic_names"], ontology["relation_names"]
        self.by_relation: dict[str, dict[str, int]] = defaultdict(dict)
        self.counts: dict[int, Counter] = defaultdict(Counter)
        for r, f in zip(np.asarray(ontology["relations"]).tolist(), np.asarray(ontology["fillers"]).tolist()):
            self.counts[r][f] += 1
        templates = getattr(lexicon, "templates", {}) or {}
        for r, pool in self.counts.items():
            name = relation_names[r]
            for f in pool:
                text = lexicon.text(names[f])
                if not text:
                    continue
                keys = {self.key(text)}
                if name in templates:
                    keys.add(self.key(lexicon.answer(name, text)))
                for k in keys:
                    self.by_relation[name].setdefault(k, f)

    @staticmethod
    def key(text: str) -> str:
        return normalize_alias(text.strip())

    def atoms(self, relation: str, candidates: Sequence[str]) -> list[int | None]:
        table = self.by_relation.get(relation, {})
        return [table.get(self.key(c)) for c in candidates]


def null_frame(frame: Sequence[tuple[int, int]], base: Sequence[tuple[int, int]], tested: dict[int, tuple[int, list[int]]],
               counts: dict[int, Counter], rng: random.Random) -> list[tuple[int, int]]:
    """The false frame: `base` (E9's random frame: the same relations, fillers redrawn) whose tested edges are redrawn among
    their item's distractors — `tested`: relation → (gold, distractor atoms); every edge of a tested relation avoids the gold."""
    out, done = [], set()
    for (r, f), (rb, fb) in zip(frame, base):
        if int(rb) != int(r):
            raise ValueError("a random frame must keep the frame's relations")
        r = int(r)
        if r in tested:
            gold, distractors = tested[r]
            if r not in done and distractors:
                pool = Counter({a: counts[r].get(a, 0) + 1 for a in distractors})
                out.append((r, edit._weighted_choice(rng, pool, set()))); done.add(r)
                continue
            if int(fb) == gold:
                fb = edit._weighted_choice(rng, counts[r], {gold}) or fb
        out.append((r, int(fb)))
    return out


# ---------------------------------------------------------------- the evaluation (GPU)


def _mean_pmi(row: dict[str, Any]) -> list[float]:
    return np.asarray(row["pmi"], float).mean(0).round(5).tolist()


class Critique:
    """Per item set: frames (real and null), beliefs with calibrated p, contexts of every condition."""

    def __init__(self, run: E5Run, store: sqx.LoadedStore, calibrator: Calibrator, lexicon: Any, *, seed: int = NULL_SEED,
                 max_lines: int | None = 32) -> None:
        self.run, self.store, self.calibrator, self.lexicon, self.seed = run, store, calibrator, lexicon, seed
        self.max_lines = max_lines
        self.ontology = run.ontology
        self.relation_id = {n: i for i, n in enumerate(self.ontology["relation_names"])}
        self.atomic_id = {n: i for i, n in enumerate(self.ontology["atomic_names"])}
        self.options = Options(self.ontology, lexicon)
        self.writer = sq.RecallWriter(self.ontology["relation_names"], self.ontology["atomic_names"], lexicon,
                                      home=sqx.home_relations(self.ontology), max_lines=max_lines)

    def frames(self, item_set: sqx.ItemSet) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
        """(real, null) frame per concept (module docstring)."""
        real, null = {}, {}
        offsets = np.asarray(self.ontology["offsets"])
        relations, fillers = np.asarray(self.ontology["relations"]), np.asarray(self.ontology["fillers"])
        by_concept = item_set.by_concept
        for cid, c in by_concept.items():
            if c.get("frame"):
                real[cid] = sqx.frame_ids(c["frame"], self.relation_id, self.atomic_id)
            elif c.get("entry") is not None:
                lo, hi = int(offsets[int(c["entry"])]), int(offsets[int(c["entry"]) + 1])
                real[cid] = list(zip(relations[lo:hi].tolist(), fillers[lo:hi].tolist()))
        if item_set.kind == "twins":
            return real, {cid: real[c["partner"]] for cid, c in by_concept.items() if c.get("partner") in real}
        tested: dict[str, dict[int, tuple[int, list[int]]]] = defaultdict(dict)
        for p in item_set.prompts:
            r = self.relation_id[p.row["relation"]]
            atoms = self.options.atoms(p.row["relation"], p.candidates)
            gold = atoms[p.gold]
            if gold is None:
                continue
            tested[p.concept][r] = (gold, [a for k, a in enumerate(atoms) if k != p.gold and a is not None and a != gold])
        from .e9_tracks import random_frames
        entries = {cid: int(c["entry"]) for cid, c in by_concept.items() if c.get("entry") is not None and not c.get("random_frame")}
        drawn = random_frames(self.ontology, entries.values(), seed=self.seed) if entries else {}
        for cid, frame in real.items():
            c = by_concept[cid]
            base = sqx.frame_ids(c["random_frame"], self.relation_id, self.atomic_id) if c.get("random_frame") else drawn[entries[cid]]
            rng = random.Random(f"{self.seed}|{cid}|null")
            null[cid] = null_frame(frame, base, tested.get(cid, {}), self.options.counts, rng)
        return real, null

    def vectors(self, frames: dict[str, list[tuple[int, int]]], item_set: sqx.ItemSet, *, real: bool) -> dict[str, torch.Tensor]:
        """Store vectors: the entry's bundle (real world, existing entries) or the frame composed (new terms; the null world)."""
        out = {}
        for cid, frame in frames.items():
            c = item_set.by_concept[cid]
            if real and c.get("entry") is not None:
                out[cid] = self.store.store.entry_vectors()[int(c["entry"])]
            else:
                out[cid] = self.store.store.frame_vector(frame)
        return out

    def beliefs(self, item_set: sqx.ItemSet, frames: dict[str, list[tuple[int, int]]], vectors: dict[str, torch.Tensor],
                worlds: dict[str, list[tuple[int, int]]]) -> dict[str, dict[str, Any]]:
        """Per item: the belief (option index or None, atom, p, features) from `vectors`, checked against `frames` (the store)
        and `worlds` (the true frames)."""
        prompts = [p for p in item_set.prompts if p.concept in vectors]
        queries = []
        for p in prompts:
            r = self.relation_id[p.row["relation"]]
            m = max(1, sum(1 for q, _ in frames[p.concept] if q == r))
            queries.append((p, r, m))
        found = decoded(self.store.store, torch.stack([vectors[p.concept] for p, _, _ in queries]) if queries else torch.zeros(0),
                        [r for _, r, _ in queries], [m for _, _, m in queries]) if queries else []
        out = {}
        for (p, r, m), fillers in zip(queries, found):
            atoms = self.options.atoms(p.row["relation"], p.candidates)
            degree = len(frames[p.concept])
            choice = next(((k, a, cos, margin) for a, cos, margin in fillers for k, x in enumerate(atoms) if x is not None and x == a), None)
            store_gold = {f for q, f in frames[p.concept] if q == r}
            world_gold = {f for q, f in worlds[p.concept] if q == r}
            record: dict[str, Any] = {"m": m, "decoded": [self.ontology["atomic_names"][a] for a, _, _ in fillers]}
            if choice is None:
                record.update(answer=None, p=None)
            else:
                k, a, cos, margin = choice
                p_value = float(self.calibrator.predict(np.asarray([[cos, margin, math.log(degree)]]), np.asarray([r]))[0])
                record.update(answer=k, atom=self.ontology["atomic_names"][a], p=p_value, cos=cos, margin=margin,
                              correct_store=bool(a in store_gold), correct_world=bool(a in world_gold))
            out[p.id] = record
        return out

    def null_contexts(self, item_set: sqx.ItemSet, null: dict[str, list[tuple[int, int]]], vectors: dict[str, torch.Tensor]) -> dict[str, str]:
        """The false store's whole-frame recall (slot-aware, the null frame's slots), written as phase A's recall."""
        texts = {}
        for cid, frame in null.items():
            c = item_set.by_concept[cid]
            lines = self.store.store.decode_slots(vectors[cid], [r for r, _ in frame])
            texts[cid] = self.writer.render(c["surface"], lines)
        return {p.id: texts[p.concept] for p in item_set.prompts if p.concept in texts}

    def edges(self, item_set: sqx.ItemSet, frames: dict[str, list[tuple[int, int]]], vectors: dict[str, torch.Tensor],
              worlds: dict[str, list[tuple[int, int]]]) -> dict[str, list[float]]:
        """Every decoded edge of the set's terms: calibrated p, correct against the store and against the world."""
        cids = [c for c in frames if c in vectors]
        if not cids:
            return {"p": [], "store": [], "world": []}
        data = edge_records(self.store.store, torch.stack([vectors[c] for c in cids]), [frames[c] for c in cids], [worlds[c] for c in cids])
        p = self.calibrator.predict(data["x"], data["relation"]) if len(data["x"]) else np.zeros(0)
        return {"p": p.round(5).tolist(), "store": data["store"].tolist(), "world": data["world"].tolist()}


def split_of(item_set: sqx.ItemSet, *, seed: int = SPLIT_SEED) -> dict[str, str]:
    """New words: a seeded half of the words is the dev half (θ and the behaviour's calibration), the rest the test half;
    every other set is test."""
    if item_set.kind != "new":
        return {c["concept"]: "test" for c in item_set.concepts}
    words = sorted(c["concept"] for c in item_set.concepts)
    random.Random(seed).shuffle(words)
    dev = set(words[:len(words) // 2])
    return {w: "dev" if w in dev else "test" for w in words}


def evaluate_set(critique: Critique, item_set: sqx.ItemSet, *, evidence: dict[str, dict[str, Any]] | None,
                 competitors: Sequence[str] = (), log: Callable[[str], None] = print) -> dict[str, Any]:
    """Score one item set under `none`, `recall:own`, `null` (and `evidence`, and the text `competitors`: phase A's
    `symbolic` / `definition`); beliefs in both worlds; calibration edges; per item the prompt tokens each context adds."""
    run = critique.run
    real, null = critique.frames(item_set)
    real_vectors = critique.vectors(real, item_set, real=True)
    null_vectors = critique.vectors(null, item_set, real=False)
    beliefs = {"real": critique.beliefs(item_set, real, real_vectors, real),
               "null": critique.beliefs(item_set, null, null_vectors, real)}
    builder = sqx.ContextBuilder(item_set, run.ontology, critique.lexicon, {"own": critique.store}, max_lines=critique.max_lines)
    contexts = {"none": {}, "recall:own": builder.build(sqx.Condition("recall", "own"))[0],
                "null": critique.null_contexts(item_set, null, null_vectors)}
    if evidence is not None:
        contexts["evidence"] = {p.id: evidence[p.id]["sentence"] for p in item_set.prompts if (evidence.get(p.id) or {}).get("sentence")}
    used, skipped = [], []
    for kind in competitors:                           # decision 64: the text competitors (phase A's conditions, reused)
        try:
            contexts[kind] = builder.build(sqx.Condition(kind))[0]
        except ValueError as error:                    # a track without a definition writer (T5 and T4 have one): skipped
            if kind != "definition" or "no definition writer" not in str(error):
                raise
            skipped.append(kind)
            log(f"  critique ({item_set.kind}): {error}: the definition competitor is skipped")
            continue
        used.append(kind)
    scores: dict[str, dict[str, list[float]]] = {}
    tokens: dict[str, dict[str, int]] = {}
    timings, longest = {}, {}
    started = time.monotonic()
    with sqx.host_view(run, item_set) as (adapter, ids):
        resolved = sqx.link_status(adapter, item_set, ids)
        cache = None
        for name, context in contexts.items():
            t0 = time.monotonic()
            prompts = [p for p in item_set.prompts if name == "none" or p.id in context]
            log(f"  critique ({item_set.kind}): {name} — {len(prompts)} items")
            with sqx.smaller_batches(adapter, 2 if context else 1):
                rows, cache, longest[name] = sqx.score_prompts(adapter, prompts, context, cache)
            scores[name] = {r["id"]: _mean_pmi(r) for r in rows}
            tokens[name] = sqx.context_tokens(adapter.tokenizer, prompts, context) if context else {}
            timings[name] = round(time.monotonic() - t0, 1)
    split = split_of(item_set)
    linked = {c for c, r in resolved.items() if r.get("status") in ("linked", "no entry")}
    items = []
    for p in item_set.prompts:
        if p.concept not in linked or p.id not in scores["none"]:
            continue
        ev = (evidence or {}).get(p.id) or {}
        items.append({"id": p.id, "set": item_set.kind, "split": split.get(p.concept, "test"), "concept": p.concept, "relation": p.row["relation"],
                      "gold": int(p.gold), "options": len(p.candidates), "scores": {k: v[p.id] for k, v in scores.items() if p.id in v},
                      "tokens": {k: tokens[k].get(p.id, 0) for k, v in scores.items() if p.id in v},
                      "belief": {"real": beliefs["real"].get(p.id), "null": beliefs["null"].get(p.id)},
                      "evidence": {"present": bool(ev.get("sentence")), "option": ev.get("option")} if evidence is not None else None,
                      "null_option": _null_option(critique, p, null)})
    calibration = {"real": critique.edges(item_set, real, real_vectors, real), "null": critique.edges(item_set, null, null_vectors, real)}
    added = {k: float(np.mean(list(v.values()))) for k, v in tokens.items() if v}
    return {"items": items, "calibration_edges": calibration, "timings": timings, "longest_tokens": longest, "tokens_added": added,
            "competitors": used, "competitors_skipped": skipped, "seconds": time.monotonic() - started, "linked": len(linked),
            "concepts": len(item_set.concepts),
            "null_frames": {cid: [[critique.ontology["relation_names"][r], critique.ontology["atomic_names"][f]] for r, f in frame]
                            for cid, frame in null.items()}}


def _null_option(critique: Critique, prompt: sqx.Prompt, null: dict[str, list[tuple[int, int]]]) -> int | None:
    """The option that the null frame's tested edge words (the false belief a method can adopt)."""
    if prompt.concept not in null:
        return None
    r = critique.relation_id[prompt.row["relation"]]
    atoms = critique.options.atoms(prompt.row["relation"], prompt.candidates)
    first = next((f for q, f in null[prompt.concept] if q == r), None)
    return next((k for k, a in enumerate(atoms) if a is not None and a == first), None)


def competitors_enabled(store: sqx.LoadedStore, setting: str = "auto") -> bool:
    """Decision 64: score the text competitors on K1's pool — `on`, `off`, or `auto`: when the store carries roles (the K1
    host, C5); the C5ut job (the null world's role-blind control) skips them, the texts not depending on the store."""
    if setting not in {"auto", "on", "off"}:
        raise ValueError(f"--competitors must be auto, on or off, not {setting!r}")
    return setting == "on" or (setting == "auto" and not store.store.role_blind)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    import yaml
    run_dir = Path(args.run)
    run_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    track, family = run_config.get("e9_track") or "t5", run_config.get("e9_family") or "smollm2"
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    output = Path(args.output or run_dir / (OUTPUT + (f"-{args.tag}" if args.tag else "")))
    alias_table = args.alias_table or ensure_alias_table(track_spec(track, family))
    items_root = ROOT / "items"
    paths = {"twins": args.twins or items_root / f"role-twins-{track}-{family}-v1", "new": args.new_words or items_root / f"new-words-{track}-{family}-v2",
             "heldout": args.heldout or WPC7}
    config = {"experiment": "e12-critique", "run": str(run_dir), "sets": sets, "paths": {k: str(v) for k, v in paths.items()},
              "evidence": str(args.evidence), "limits": {"twins": args.twin_limit, "new": args.new_limit, "heldout": args.heldout_limit},
              "seen_cap": args.seen_cap, "null_seed": NULL_SEED, "split_seed": SPLIT_SEED, "batch_size": args.batch_size,
              "max_length": args.max_length, "competitors": args.competitors, "label": args.label}
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    store = sqx.load_store(Path(args.store or run_dir), "own")
    use_competitors = competitors_enabled(store, args.competitors)
    calibrator, seen = fit_calibrator(store, cap=args.seen_cap)
    run = open_run(run_dir, device=args.device, batch_size=args.batch_size, max_length=args.max_length, alias_table=alias_table)
    lexicon = sqx.lexicon_for_track(track, family, run.ontology)
    critique = Critique(run, store, calibrator, lexicon, max_lines=args.max_lines)
    results = {}
    started = time.monotonic()
    for name in sets:
        if name == "twins":
            item_set = sqx.load_item_set(paths["twins"], limit=args.twin_limit)
            item_set.prompts = [p for p in item_set.prompts if p.row.get("kind") == "choice"]
            evidence = None
        elif name == "new":
            item_set = sqx.load_item_set(paths["new"], limit=args.new_limit)
            evidence = None
        else:
            item_set = wpc7_heldout(paths["heldout"], run.ontology, run.table)
            if args.heldout_limit:
                keep = {c["concept"] for c in item_set.concepts[:args.heldout_limit]}
                item_set.concepts = [c for c in item_set.concepts if c["concept"] in keep]
                item_set.prompts = [p for p in item_set.prompts if p.concept in keep]
            evidence = load_evidence(args.evidence)
        results[name] = evaluate_set(critique, item_set, evidence=evidence,
                                     competitors=COMPETITORS if use_competitors and name in COMPETITOR_SETS else ())
    seconds = time.monotonic() - started
    header = {"source": run.describe(), "store": store.describe(), "label": args.label}
    items = [i for r in results.values() for i in r["items"]]
    und.write_jsonl_gz(output / "items.jsonl.gz", items)
    seen_p = calibrator.predict(seen["x"], seen["relation"]) if len(seen["x"]) else np.zeros(0)
    summary = {**header, "calibrator": calibrator.describe(), "seen": calibration_metrics(seen_p, seen["store"]),
               "sets": {k: {kk: vv for kk, vv in v.items() if kk not in {"items"}} | {"items": len(v["items"])} for k, v in results.items()},
               "seconds": seconds}
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    summary["peak_gb"] = peak
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(render_run(summary, items))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {"output": str(output), "seconds": seconds, "peak_gb": peak, "timings": {k: v["timings"] for k, v in results.items()}}


def render_run(summary: dict[str, Any], items: Sequence[dict[str, Any]]) -> str:
    source = summary["source"]
    label = f" — {summary['label']}" if summary.get("label") else ""
    lines = [f"# E12 3c — critique inputs, {source['condition']} seed {source['seed']} ({source['size']}){label}", "",
             f"Calibrator: {summary['calibrator']['info']}. Seen-entry calibration: ECE {_f(summary['seen']['ece'])}, Brier "
             f"{_f(summary['seen']['brier'])}, AUROC {_f(summary['seen']['auroc'])}.", "",
             "| set | items | none | recall:own | null | evidence | symbolic | definition | belief (real) correct | belief (null) = null option |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    contexts = ("recall:own", "null", "evidence", *COMPETITORS)
    usage = []
    for name in SETS:
        sub = [i for i in items if i["set"] == ("new" if name == "new" else name)]
        if not sub:
            continue
        acc = lambda key: _f(np.mean([np.argmax(i["scores"][key]) == i["gold"] for i in sub if key in i["scores"]])) \
            if any(key in i["scores"] for i in sub) else "—"
        real = [i["belief"]["real"] for i in sub if i["belief"]["real"] and i["belief"]["real"]["answer"] is not None]
        false = [i for i in sub if i["belief"]["null"] and i["belief"]["null"]["answer"] is not None and i["null_option"] is not None]
        lines.append(f"| {name} | {len(sub)} | {acc('none')} | {acc('recall:own')} | {acc('null')} | {acc('evidence')} | {acc('symbolic')} | "
                     f"{acc('definition')} | {_f(np.mean([b['correct_world'] for b in real])) if real else '—'} | "
                     f"{_f(np.mean([i['belief']['null']['answer'] == i['null_option'] for i in false])) if false else '—'} |")
        spent = {k: [i["tokens"][k] for i in sub if k in (i.get("tokens") or {})] for k in contexts}
        usage.append(f"| {name} | " + " | ".join(f"{np.mean(v):.1f}" if v else "—" for v in spent.values()) + " |")
    lines += ["", "Prompt tokens each context adds per item (mean; added once to each of the item's prompts; the store's rule loop "
              "reads the decode without the host: none):", "", "| set | " + " | ".join(contexts) + " |", "|---|" + "---:|" * len(contexts)]
    return "\n".join(lines + usage) + "\n"


def _f(value: Any) -> str:
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.3f}"


# ---------------------------------------------------------------- the rule loop, K1 and K2 (CPU)


def softmax(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, float)
    e = np.exp(v - v.max())
    return e / e.sum()


def selective(answers: Sequence[int | None], confidence: Sequence[float], *, coverage: float = COVERAGE,
              flagged: Sequence[bool] | None = None) -> np.ndarray:
    """Which items a method answers at `coverage`: it abstains on its own lowest-ranked share (rank: unflagged before
    flagged, then by confidence; ties by item order); an item without an answer is ranked lowest."""
    n = len(answers)
    keep = int(round(coverage * n))
    flagged = flagged if flagged is not None else [False] * n
    key = [(0 if a is None else 1, 0 if f else 1, float(c), -i) for i, (a, c, f) in enumerate(zip(answers, confidence, flagged))]
    order = sorted(range(n), key=lambda i: key[i], reverse=True)
    answered = np.zeros(n, bool)
    answered[order[:keep]] = True
    answered &= np.asarray([a is not None for a in answers], bool)
    return answered


def isotonic_from(conf: np.ndarray, correct: np.ndarray) -> dict[str, list[float]] | None:
    if len(conf) < 2 or correct.min() == correct.max():
        return None
    return isotonic_fit(conf, correct)


def methods(items: Sequence[dict[str, Any]], *, world: str, theta: float, behaviour_cal: dict[str, list[float]] | None) -> dict[str, dict[str, Any]]:
    """Answers, confidences and flags of every method on `items` in `world` (real / null): `no_tool` (the behaviour, its
    softmax), `naive` (always the belief; else the behaviour), `loop` (the belief when p ≥ θ, else the behaviour; flagged
    when the available signals disagree), `loop_revise` (secondary: an evidence answer that disagrees replaces the loop's),
    `recall_context` (secondary: the host's answer with the recall in context)."""
    out: dict[str, dict[str, list]] = {m: {"answer": [], "confidence": [], "flagged": []} for m in
                                      ("no_tool", "naive", "loop", "loop_revise", "recall_context")}
    context_key = "recall:own" if world == "real" else "null"
    for item in items:
        none = softmax(item["scores"]["none"])
        h, q = int(np.argmax(none)), float(none.max())
        ch = float(isotonic_predict(behaviour_cal, np.asarray([q]))[0]) if behaviour_cal else q
        belief = item["belief"][world] or {}
        b, p = belief.get("answer"), belief.get("p")
        v = None
        if item.get("evidence") and item["evidence"].get("present") and "evidence" in item["scores"]:
            v = int(np.argmax(item["scores"]["evidence"]))
        signals = [x for x in (b, h, v) if x is not None]
        disagree = len(set(signals)) > 1
        out["no_tool"]["answer"].append(h); out["no_tool"]["confidence"].append(q); out["no_tool"]["flagged"].append(False)
        out["naive"]["answer"].append(b if b is not None else h); out["naive"]["confidence"].append(p if b is not None else ch)
        out["naive"]["flagged"].append(False)
        use_belief = b is not None and p is not None and p >= theta
        a, c = (b, p) if use_belief else (h, ch)
        out["loop"]["answer"].append(a); out["loop"]["confidence"].append(c); out["loop"]["flagged"].append(disagree)
        ra = v if v is not None and v != a else a
        out["loop_revise"]["answer"].append(ra); out["loop_revise"]["confidence"].append(c); out["loop_revise"]["flagged"].append(disagree)
        ctx = item["scores"].get(context_key)
        if ctx is not None:
            s = softmax(ctx)
            out["recall_context"]["answer"].append(int(np.argmax(s))); out["recall_context"]["confidence"].append(float(s.max()))
        else:
            out["recall_context"]["answer"].append(None); out["recall_context"]["confidence"].append(0.0)
        out["recall_context"]["flagged"].append(False)
    return out


def text_loop(items: Sequence[dict[str, Any]], key: str, *, theta: float, behaviour_cal: dict[str, list[float]] | None,
              context_cal: dict[str, list[float]] | None) -> dict[str, list]:
    """The rule loop with a text in the prompt instead of the store decode (decision 64, amendment 16.5): the belief is the
    host's answer with the `key` context (`symbolic`: the gold relations as text; `definition`: the prose definition;
    `recall:own`: the store's decode as text), its confidence that answer's softmax calibrated on the dev half
    (`context_cal`); the loop answers with it when that confidence is ≥ θ, else with the behaviour (calibrated as the store
    loop's), and flags the item when the available signals (this belief, the behaviour, the evidence answer) disagree.
    `tokens`: the prompt tokens the text adds to the item (0 without the context)."""
    out: dict[str, list] = {"answer": [], "confidence": [], "flagged": [], "tokens": []}
    for item in items:
        none = softmax(item["scores"]["none"])
        h, q = int(np.argmax(none)), float(none.max())
        ch = float(isotonic_predict(behaviour_cal, np.asarray([q]))[0]) if behaviour_cal else q
        v = None
        if item.get("evidence") and item["evidence"].get("present") and "evidence" in item["scores"]:
            v = int(np.argmax(item["scores"]["evidence"]))
        context = item["scores"].get(key)
        b = p = None
        if context is not None:
            s = softmax(context)
            b = int(np.argmax(s))
            p = float(isotonic_predict(context_cal, np.asarray([s.max()]))[0]) if context_cal else float(s.max())
        disagree = len({x for x in (b, h, v) if x is not None}) > 1
        a, c = (b, p) if b is not None and p >= theta else (h, ch)
        out["answer"].append(a); out["confidence"].append(c); out["flagged"].append(disagree)
        out["tokens"].append(int((item.get("tokens") or {}).get(key, 0)) if context is not None else 0)
    return out


def context_calibration(dev: Sequence[dict[str, Any]], key: str) -> dict[str, list[float]] | None:
    """Isotonic calibration, on the dev half, of the softmax confidence of the host's answer with the `key` context."""
    rows = [i for i in dev if key in i["scores"]]
    conf = np.asarray([softmax(i["scores"][key]).max() for i in rows])
    correct = np.asarray([float(np.argmax(i["scores"][key]) == i["gold"]) for i in rows])
    return isotonic_from(conf, correct)


def scored(items: Sequence[dict[str, Any]], method: dict[str, list], *, coverage: float = COVERAGE, world: str = "real") -> dict[str, Any]:
    """Per item at `coverage`: answered, correct, adopted (the null world's false option), and the K units
    `1(answered ∧ correct) / coverage` (their mean is the accuracy at that coverage) and `1(answered ∧ adopted) / coverage`."""
    answered = selective(method["answer"], method["confidence"], coverage=coverage, flagged=method["flagged"])
    correct = np.asarray([a is not None and a == i["gold"] for a, i in zip(method["answer"], items)], bool)
    adopted = np.asarray([a is not None and i.get("null_option") is not None and a == i["null_option"] for a, i in zip(method["answer"], items)], bool)
    share = max(answered.mean(), 1e-9)
    return {"answered": answered, "correct": correct, "adopted": adopted,
            "k1": (answered & correct) / share, "k2": (answered & adopted) / share,
            "accuracy_at": float((answered & correct).sum() / max(answered.sum(), 1)),
            "adoption_at": float((answered & adopted).sum() / max(answered.sum(), 1)),
            "accuracy_full": float(correct.mean()) if len(correct) else None, "adoption_full": float(adopted.mean()) if len(adopted) else None,
            "flag_rate": float(np.mean(method["flagged"])) if method["flagged"] else None}


def coverage_curve(items: Sequence[dict[str, Any]], method: dict[str, list], *, key: str = "correct") -> list[dict[str, float]]:
    out = []
    for coverage in [round(0.05 * k, 2) for k in range(4, 21)]:
        s = scored(items, method, coverage=coverage)
        out.append({"coverage": coverage, "accuracy": float((s["answered"] & s["correct"]).sum() / max(s["answered"].sum(), 1)),
                    "adoption": float((s["answered"] & s["adopted"]).sum() / max(s["answered"].sum(), 1))})
    return out


def choose_theta(dev: Sequence[dict[str, Any]], behaviour_cal: dict[str, list[float]] | None,
                 build: Callable[[float], dict[str, list]] | None = None) -> tuple[float, list[dict[str, float]]]:
    """θ maximizing the rule loop's accuracy at 80% coverage on the dev half (real world); ties: the smallest θ. `build`
    (θ → the method; default the store's rule loop) chooses a text loop's θ by the same rule (amendment 16.5)."""
    if build is None:
        build = lambda theta: methods(dev, world="real", theta=theta, behaviour_cal=behaviour_cal)["loop"]
    table = []
    for theta in THETAS:
        s = scored(dev, build(theta))
        table.append({"theta": theta, "accuracy": s["accuracy_at"]})
    best = max(table, key=lambda r: (round(r["accuracy"], 12), -r["theta"]))
    return float(best["theta"]), table


def behaviour_calibration(dev: Sequence[dict[str, Any]]) -> dict[str, list[float]] | None:
    """Isotonic calibration of the behaviour's softmax confidence on the dev half (the loop's confidence when it answers
    with the behaviour)."""
    conf = np.asarray([softmax(i["scores"]["none"]).max() for i in dev])
    correct = np.asarray([float(np.argmax(i["scores"]["none"]) == i["gold"]) for i in dev])
    return isotonic_from(conf, correct)


def analyse_run(items: Sequence[dict[str, Any]], *, test_sets: Sequence[str] = ("new", "heldout")) -> dict[str, Any]:
    """One run: θ and the behaviour calibration on the dev half; every method in both worlds on the test items (pooled
    over `test_sets` and per set); per-item K units; curves."""
    dev = [i for i in items if i["set"] == "new" and i["split"] == "dev"]
    cal = behaviour_calibration(dev) if dev else None
    theta, table = choose_theta(dev, cal) if dev else (0.5, [])
    out: dict[str, Any] = {"theta": theta, "theta_table": table, "dev_items": len(dev), "behaviour_calibration": cal, "pools": {},
                           "text_loops": {}}
    # decision 64: the same rule loop with a text in the prompt instead of the store decode; θ and the text answer's
    # calibration on the dev half, by the store loop's rule
    for name, key in TEXT_LOOPS.items():
        if not any(key in i["scores"] for i in items):
            continue
        context_cal = context_calibration(dev, key) if dev else None
        build = lambda t, key=key, context_cal=context_cal: text_loop(dev, key, theta=t, behaviour_cal=cal, context_cal=context_cal)
        loop_theta, loop_table = choose_theta(dev, cal, build=build) if any(key in i["scores"] for i in dev) else (0.5, [])
        out["text_loops"][name] = {"context": key, "theta": loop_theta, "theta_table": loop_table, "context_calibration": context_cal}
    pools = {"pooled": [i for i in items if i["set"] in test_sets and i["split"] == "test"]}
    for name in ("new", "heldout", "twins"):
        pools[name] = [i for i in items if i["set"] == name and i["split"] == "test"]
    for pool, sub in pools.items():
        if not sub:
            continue
        block: dict[str, Any] = {"items": len(sub), "ids": [i["id"] for i in sub],
                                 "tokens_evidence": float(np.mean([(i.get("tokens") or {}).get("evidence", 0) for i in sub]))}
        for world in ("real", "null"):
            ms = methods(sub, world=world, theta=theta, behaviour_cal=cal)
            context_key = "recall:own" if world == "real" else "null"
            spent = {m: 0.0 for m in ms} | {"recall_context": float(np.mean([(i.get("tokens") or {}).get(context_key, 0) for i in sub]))}
            if world == "real":                        # the text sources are uncorrupted: no null-world counterpart (K2 unchanged)
                for name, spec in out["text_loops"].items():
                    if any(spec["context"] in i["scores"] for i in sub):
                        ms[name] = text_loop(sub, spec["context"], theta=spec["theta"], behaviour_cal=cal, context_cal=spec["context_calibration"])
                        spent[name] = float(np.mean(ms[name]["tokens"]))
            block[world] = {}
            for name, method in ms.items():
                s = scored(sub, method, world=world)
                block[world][name] = {"k1": s["k1"].tolist(), "k2": s["k2"].tolist(), "accuracy_at": s["accuracy_at"],
                                      "adoption_at": s["adoption_at"], "accuracy_full": s["accuracy_full"],
                                      "adoption_full": s["adoption_full"], "flag_rate": s["flag_rate"], "tokens": spent[name],
                                      "curve": coverage_curve(sub, method)}
        out["pools"][pool] = block
    return out


def discover(runs_root: Path, *, hosts: Sequence[str] | None = None, folder: str = OUTPUT) -> dict[str, dict[str, dict[int, Path]]]:
    out: dict[str, dict[str, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(Path(runs_root).iterdir()):
        match = sqx.RUN_NAME.match(run.name)
        if match is None or (hosts and match["host"] not in hosts) or not (run / folder / "summary.json").exists():
            continue
        out[match["host"]][match["model"]][int(match["seed"])] = run / folder
    return {h: dict(m) for h, m in out.items()}


def _units(per_seed: dict[int, dict[str, Any]], pool: str, world: str, method: str, key: str) -> dict[int, dict[str, float]]:
    out = {}
    for seed, analysis in per_seed.items():
        block = analysis["pools"].get(pool)
        if block and method in block.get(world, {}):
            out[seed] = dict(zip(block["ids"], block[world][method][key]))
    return out


def analyse(runs_root: Path, *, hosts: Sequence[str] | None = None, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    from .e12_report import _holm, contrast
    found = discover(runs_root, hosts=hosts)
    analysis: dict[str, Any] = {"runs_root": str(runs_root), "hosts": {}}
    for host, models in sorted(found.items()):
        block: dict[str, Any] = {"runs": {m: sorted(s) for m, s in models.items()}, "models": {}}
        for model, seeds in sorted(models.items()):
            per_seed, calibration = {}, {}
            for s, folder in sorted(seeds.items()):
                items = und.read_jsonl(folder / "items.jsonl")
                summary = json.loads((folder / "summary.json").read_text())
                per_seed[s] = analyse_run(items)
                calibration[s] = {"seen entries (the fit set, in-sample)": {k: v for k, v in summary["seen"].items() if k != "reliability"}}
                for name, set_block in summary["sets"].items():
                    for world in ("real", "null"):
                        edges = set_block["calibration_edges"][world]
                        for target in ("store", "world"):
                            calibration[s][f"{name} {world} vs {target}"] = calibration_metrics(np.asarray(edges["p"]), np.asarray(edges[target]))
            m_block: dict[str, Any] = {"theta": {s: a["theta"] for s, a in per_seed.items()}, "calibration": calibration}
            primary: dict[str, Any] = {}
            loop1, base1 = _units(per_seed, "pooled", "real", "loop", "k1"), _units(per_seed, "pooled", "real", "no_tool", "k1")
            if loop1 and base1:
                primary["K1: rule loop − no tool (accuracy at 80% coverage)"] = contrast(loop1, base1, resamples=resamples, seed=seed)
            loop2, naive2 = _units(per_seed, "pooled", "null", "loop", "k2"), _units(per_seed, "pooled", "null", "naive", "k2")
            if loop2 and naive2:
                primary["K2: rule loop − naive recall (false-belief adoption at 80% coverage, null world)"] = contrast(loop2, naive2, resamples=resamples, seed=seed)
            _holm(primary)
            m_block["primary"] = primary
            m_block["k1b"] = k1b(per_seed, resamples=resamples, seed=seed)
            sec: dict[str, Any] = {}
            for pool in ("new", "heldout", "twins"):
                for name, (world, a, b, key) in {"K1": ("real", "loop", "no_tool", "k1"), "K2": ("null", "loop", "naive", "k2"),
                                                 "K1 naive − no tool": ("real", "naive", "no_tool", "k1"),
                                                 "K1 loop+evidence revision − no tool": ("real", "loop_revise", "no_tool", "k1"),
                                                 "K2 loop+evidence revision − naive": ("null", "loop_revise", "naive", "k2"),
                                                 "K2 recall in context (host reads the false recall) − naive": ("null", "recall_context", "naive", "k2")}.items():
                    ua, ub = _units(per_seed, pool, world, a, key), _units(per_seed, pool, world, b, key)
                    if ua and ub:
                        sec[f"{name} ({pool})"] = contrast(ua, ub, resamples=resamples, seed=seed)
            m_block["secondaries"] = sec
            m_block["means"] = _method_means(per_seed)
            m_block["curves"] = {s: {pool: {w: {m: v["curve"] for m, v in b[w].items()} for w in ("real", "null")}
                                     for pool, b in a["pools"].items()} for s, a in per_seed.items()}
            m_block["reading"] = reading(primary, sec)
            block["models"][model] = m_block
        analysis["hosts"][host] = block
    return analysis


def k1b(per_seed: dict[int, dict[str, Any]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Decision 64 (amendment 16.5): **K1b** = rule loop (store) − rule loop (the prose definition in the prompt instead of
    the store decode), accuracy at 80% coverage on K1's pool (items × seeds crossed model), non-inferiority at −`K1B_MARGIN`
    (one-sided α = 0.025: the 95% CI's lower bound above the margin); its own test, outside K1 / K2's Holm family. Read with
    the prompt tokens each loop adds per item (`tokens`). Secondaries (never promoted): K1b per set, the loop against the
    gold relations as text, the store decode as text against the definition (both read by the host), the competitors' own
    K1."""
    from .e12_report import contrast, margin_reading, margin_test
    out: dict[str, Any] = {"margin": K1B_MARGIN, "secondaries": {}, "tokens": {}}

    def test(a: str, b: str, pool: str, margin: float | None) -> dict[str, Any] | None:
        ua, ub = _units(per_seed, pool, "real", a, "k1"), _units(per_seed, pool, "real", b, "k1")
        if not ua or not ub:
            return None
        result = contrast(ua, ub, resamples=resamples, seed=seed)
        if margin is not None:
            margin_test(result, margin, kind="noninferiority")
        result["reading"] = margin_reading(result)
        return result

    primary = test("loop", "loop_definition", "pooled", K1B_MARGIN)
    out["K1b"] = primary or {"available": False}
    rows = {f"K1b ({pool})": ("loop", "loop_definition", pool, K1B_MARGIN) for pool in ("new", "heldout")}
    for pool in ("pooled", "new", "heldout"):
        rows[f"loop (store) − loop (gold relations as text) ({pool})"] = ("loop", "loop_symbolic", pool, K1B_MARGIN)
    rows["loop (store decode as text, host reads) − loop (definition) (pooled)"] = ("loop_recall_text", "loop_definition", "pooled", K1B_MARGIN)
    for name in ("loop_definition", "loop_symbolic", "loop_recall_text"):
        rows[f"K1 of {name}: {name} − no tool (pooled)"] = (name, "no_tool", "pooled", None)
    for label, args in rows.items():
        result = test(*args)
        if result is not None:
            out["secondaries"][label] = result
    for method in ("no_tool", "loop", "recall_context", *TEXT_LOOPS):
        values = [a["pools"]["pooled"]["real"][method]["tokens"] for a in per_seed.values()
                  if "pooled" in a["pools"] and method in a["pools"]["pooled"]["real"] and "tokens" in a["pools"]["pooled"]["real"][method]]
        accuracy = [a["pools"]["pooled"]["real"][method]["accuracy_at"] for a in per_seed.values()
                    if "pooled" in a["pools"] and method in a["pools"]["pooled"]["real"]]
        if values:
            out["tokens"][method] = {"tokens_per_item": float(np.mean(values)), "accuracy_at": float(np.mean(accuracy))}
    evidence = [a["pools"]["pooled"].get("tokens_evidence") for a in per_seed.values() if "pooled" in a["pools"]]
    out["tokens_evidence"] = float(np.mean([e for e in evidence if e is not None])) if any(e is not None for e in evidence) else None
    out["reading"] = k1b_reading(out)
    return out


def k1b_reading(block: dict[str, Any]) -> str:
    result = block.get("K1b") or {}
    if not result.get("available"):
        return "not available (no definition competitor scored)"
    tokens = block.get("tokens", {})
    saved = (tokens.get("loop_definition", {}).get("tokens_per_item", 0.0) - tokens.get("loop", {}).get("tokens_per_item", 0.0)) \
        if "loop_definition" in tokens else None
    text = result["reading"]
    return text + ("" if saved is None else f"; the store's rule loop adds {saved:.0f} fewer prompt tokens per item than the definition loop")


def _method_means(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for analysis in per_seed.values():
        for pool, block in analysis["pools"].items():
            for world in ("real", "null"):
                for method, v in block[world].items():
                    for key in ("accuracy_at", "adoption_at", "accuracy_full", "adoption_full", "flag_rate", "tokens"):
                        if v.get(key) is not None:
                            out[f"{pool} {world} {method}"][key].append(v[key])
    return {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in out.items()}


def reading(primary: dict[str, Any], secondaries: dict[str, Any]) -> dict[str, str]:
    out = {}
    for name, r in primary.items():
        if not r.get("available"):
            out[name.split(":")[0]] = "not available"
            continue
        sig = r.get("p_holm", 1.0) < 0.05
        mean = r["model"]["mean"]
        if name.startswith("K1"):
            out["K1"] = ("the rule loop beats the no-tool answers at 80% coverage" if sig and mean > 0 else
                         "the rule loop is worse than the no-tool answers" if sig else "no difference shown")
        else:
            out["K2"] = ("the loop adopts the false belief less than naive recall" if sig and mean < 0 else
                         "the loop adopts the false belief more than naive recall" if sig else
                         "no difference shown: the loop accepts false structure as naive recall does")
    ho = secondaries.get("K2 (heldout)")
    if ho and ho.get("available") and ho["model"]["ci_low"] <= 0 <= ho["model"]["ci_high"]:
        out["refutation"] = "K2 ≈ 0 on held-out terms with evidence in context: the loop does not use evidence against its own store"
    return out


def render_report(analysis: dict[str, Any], *, title: str, label: str | None = None) -> str:
    from .e12_report import _cell
    lines = [f"# {title}" + (f" — {label}" if label else ""), "",
             "Pre-registration: `experiments/e12-self-query/preregistration.md` §13, amendments 16.3 and 16.5. K1 and K2: items × seeds crossed "
             "model of the per-item units `1(answered ∧ correct) / 0.8` and `1(answered ∧ adopted) / 0.8` (their means are the accuracy "
             "and the false-belief adoption at 80% coverage), Holm over the two.", ""]
    if label:
        lines += [f"**{label}: these numbers do not count toward the pre-registered endpoints.**", ""]
    for host, block in analysis["hosts"].items():
        lines += [f"## {host}", "", f"Runs: {block['runs']}", ""]
        for model, m in block["models"].items():
            lines += [f"### {model}", "", f"θ per seed (dev half of the new words): {m['theta']}", "", "| endpoint | estimate |", "|---|---|"]
            lines += [f"| {n} | {_cell(r)} |" for n, r in m["primary"].items()]
            lines += ["", "Reading: " + "; ".join(f"{k}: {v}" for k, v in m["reading"].items()), ""]
            kb = m.get("k1b") or {}
            if kb:
                lines += [f"**K1b (amendment 16.5)** — rule loop (store) − rule loop (the prose definition in the prompt), accuracy at 80% "
                          f"coverage, non-inferiority at −{kb['margin']:g} (one-sided α = 0.025; not in K1 / K2's Holm family): "
                          f"{_cell(kb.get('K1b'))}. Reading: {kb['reading']}.", "",
                          "| loop (real world, pooled test items; seed means) | accuracy at 80% | prompt tokens added per item |", "|---|---:|---:|"]
                lines += [f"| {name} | {_f(v['accuracy_at'])} | {v['tokens_per_item']:.1f} |" for name, v in kb.get("tokens", {}).items()]
                if kb.get("tokens_evidence") is not None:
                    lines.append(f"| (every loop's flag: the evidence sentence, held-out items) | — | {kb['tokens_evidence']:.1f} |")
                lines += ["", "| K1b secondary | estimate | reading |", "|---|---|---|"]
                lines += [f"| {n} | {_cell(r)} | {r.get('reading', '')} |" for n, r in kb.get("secondaries", {}).items()] + [""]
            lines += [
                      "| pool / world / method | accuracy at 80% | adoption at 80% | accuracy (all) | adoption (all) | flag rate | prompt tokens added |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for name, v in m["means"].items():
                lines.append(f"| {name} | {_f(v.get('accuracy_at'))} | {_f(v.get('adoption_at'))} | {_f(v.get('accuracy_full'))} | "
                             f"{_f(v.get('adoption_full'))} | {_f(v.get('flag_rate'))} | "
                             f"{'—' if v.get('tokens') is None else f'{v['tokens']:.1f}'} |")
            lines += ["", "| calibration of p (seed-averaged) | n | accuracy | mean p | ECE | Brier | AUROC |", "|---|---:|---:|---:|---:|---:|---:|"]
            keys = sorted({k for c in m["calibration"].values() for k in c})
            for key in keys:
                vals = [c[key] for c in m["calibration"].values() if key in c]
                avg = lambda f: (float(np.mean([v[f] for v in vals if v.get(f) is not None])) if any(v.get(f) is not None for v in vals) else None)
                lines.append(f"| {key} | {'—' if avg('n') is None else f"{avg('n'):.0f}"} | {_f(avg('accuracy'))} | {_f(avg('mean_p'))} | {_f(avg('ece'))} | {_f(avg('brier'))} | {_f(avg('auroc'))} |")
            lines += ["", "| secondary | estimate |", "|---|---|"] + [f"| {n} | {_cell(r)} |" for n, r in m["secondaries"].items()] + [""]
            first = next(iter(m["curves"].values()), {}).get("pooled")
            if first:
                lines += ["Coverage–accuracy (pooled test items, first seed; real world): coverage → no tool / naive / loop", "",
                          "| coverage | no tool | naive | loop | loop (null world): adoption |", "|---:|---:|---:|---:|---:|"]
                for k, point in enumerate(first["real"]["loop"]):
                    lines.append(f"| {point['coverage']:.2f} | {first['real']['no_tool'][k]['accuracy']:.3f} | {first['real']['naive'][k]['accuracy']:.3f} | "
                                 f"{point['accuracy']:.3f} | {first['null']['loop'][k]['adoption']:.3f} |")
                lines.append("")
    loops = analysis.get("model_loop")
    if loops:
        lines += ["## Model loop (secondary)", ""] + [f"- {k}: {_cell(v) if isinstance(v, dict) and 'model' in v else v}" for k, v in loops.items()] + [""]
    return "\n".join(lines) + "\n"


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    config = {"experiment": "e12-critique-report", "runs": str(args.runs), "hosts": args.hosts, "loop_runs": str(args.loop_runs) if args.loop_runs else None,
              "label": args.label}
    if args.overwrite:
        for name in ("analysis.json", "report.md", "resolved_config.yaml", "manifest.json"):
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = start_output(args.output, config)
    started = time.monotonic()
    analysis = analyse(args.runs, hosts=args.hosts, resamples=args.resamples)
    if args.loop_runs:
        analysis["model_loop"] = analyse_loops(args.loop_runs, resamples=args.resamples)
    write_json(args.output / "analysis.json", json_ready(analysis))
    (args.output / "report.md").write_text(render_report(analysis, title=args.title, label=args.label))
    finish_output(args.output, config, git_at_start=git_at_start, device="cpu", seconds=round(time.monotonic() - started, 1))
    return {"output": str(args.output), "hosts": list(analysis["hosts"])}


# ---------------------------------------------------------------- the model loop (GPU, secondary)


LOOP_INSTRUCTION = ("Check an answer against what is stored about the term and, when there is one, the evidence. Reply Keep, "
                    "Revise to <option>, or Unsure.\n")
DECISIONS = (" Keep", " Revise", " Unsure")


def critique_case(question: str, answer: str, recall: str, evidence: str | None) -> str:
    return f"{question}Answer given: {answer}\nRecall: {recall}\nEvidence: {evidence or 'none'}\nDecision:"


def recall_statement(writer: sq.RecallWriter, surface: str, relation: int, atom: int, p: float) -> str:
    return f"{writer.statement(surface, relation, atom)} (p = {p:.2f})"


def loop_demonstrations(critique: Critique, seen: dict[str, np.ndarray], *, exclude: set[int]) -> str:
    """Four worked cases on seen training entries, their recall lines real decodes with their calibrated p: Keep (the answer
    agrees with a confident recall), Revise (a wrong answer, a confident correct recall), Keep against the recall (a real
    decode error, the evidence — the relation's training statement — agreeing with the answer), Unsure (a low-p recall
    that disagrees, no evidence)."""
    store, o = critique.store.store, critique.ontology
    lex, writer = critique.lexicon, critique.writer
    templates = getattr(lex, "templates", {}) or {}
    names = sqx.concept_names(o)
    entries = seen["entries"]
    frames = [store.frame(int(e)) for e in entries]
    edges = []                                   # (entry, relation, decoded atom, p, correct)
    vectors = store.entry_vectors()[torch.tensor(entries.tolist(), dtype=torch.long)] if len(entries) else None
    for i, e in enumerate(entries.tolist()):
        if e in exclude:
            continue
        for r, m, gold in frame_slots(frames[i]):
            if m != 1 or o["relation_names"][r] not in templates or not lex.text(o["atomic_names"][gold[0]]):
                continue
            edges.append((e, r, gold[0], i))
        if len(edges) > 4000:
            break
    if not edges:
        return ""
    found = decoded(store, vectors[[x[3] for x in edges]], [x[1] for x in edges], [1] * len(edges))
    rows = []
    for (e, r, gold, i), fillers in zip(edges, found):
        if not fillers:
            continue
        a, cos, margin = fillers[0]
        p = float(critique.calibrator.predict(np.asarray([[cos, margin, math.log(len(frames[i]))]]), np.asarray([r]))[0])
        rows.append({"entry": e, "relation": r, "gold": gold, "atom": a, "p": p, "correct": a == gold})
    pools = critique.options.counts

    def case(row: dict[str, Any], answer_atom: int, evidence: str | None, decision: str, options: list[int]) -> str:
        name, relation = names[row["entry"]], o["relation_names"][row["relation"]]
        rng = random.Random(f"demo|{row['entry']}|{relation}")
        rng.shuffle(options)
        question = agent_question(lex.prompts("*", relation)[0], name, [lex.answer(relation, lex.text(o["atomic_names"][x])) for x in options])
        answer = lex.answer(relation, lex.text(o["atomic_names"][answer_atom])).strip()
        recall = recall_statement(writer, name, row["relation"], row["atom"], row["p"])
        return critique_case(question, answer, recall, evidence) + f" {decision}\n"

    def distractors(row: dict[str, Any], k: int, avoid: set[int]) -> list[int]:
        pool = [a for a, _ in pools[row["relation"]].most_common() if a not in avoid and lex.text(o["atomic_names"][a])]
        return pool[:k]

    out = []
    confident = sorted([r for r in rows if r["correct"]], key=lambda r: -r["p"])
    wrong = sorted([r for r in rows if not r["correct"]], key=lambda r: -r["p"])
    low = sorted(rows, key=lambda r: r["p"])
    if confident:
        r = confident[0]
        opts = [r["gold"]] + distractors(r, 3, {r["gold"]})
        out.append(case(r, r["gold"], None, "Keep", opts))
    if len(confident) > 1:
        r = confident[1]
        other = distractors(r, 3, {r["gold"]})
        out.append(case(r, other[0], None, "Revise to " + lex.answer(o["relation_names"][r["relation"]], lex.text(o["atomic_names"][r["gold"]])).strip(),
                        [r["gold"]] + other))
    if wrong:
        r = wrong[0]
        relation = o["relation_names"][r["relation"]]
        statement = _statement(templates[relation].statement, names[r["entry"]], lex.text(o["atomic_names"][r["gold"]]))
        opts = [r["gold"], r["atom"]] + distractors(r, 2, {r["gold"], r["atom"]})
        out.append(case(r, r["gold"], statement, "Keep", opts))
    if low:
        unsure = [x for x in low if x["p"] < 0.6]                 # a genuinely low-p decode: the median of those, else the lowest
        r = unsure[len(unsure) // 2] if unsure else low[0]
        other = distractors(r, 3, {r["gold"], r["atom"]})
        out.append(case(r, other[0], None, "Unsure", list(dict.fromkeys([r["atom"], r["gold"]] + other))[:4]))
    return "\n".join(out)


def _statement(template: str, x: str, y: str) -> str:
    return template.replace("{x}", x).replace("{y}", y)


def agent_question(template: str, surface: str, candidates: Sequence[str]) -> str:
    stem = und.render(template, {"x": surface})
    return f"Question: {stem} ___? Options: {' | '.join(c.strip() for c in candidates)}\n"


def loop_items(item_sets: dict[str, sqx.ItemSet], *, count: int, seed: int = 0) -> dict[str, list[str]]:
    """A seeded sample of `count` items per set (new words: test half only)."""
    out = {}
    for name, item_set in item_sets.items():
        split = split_of(item_set)
        ids = sorted(p.id for p in item_set.prompts if split.get(p.concept, "test") == "test")
        random.Random(f"{seed}|{name}|loop").shuffle(ids)
        out[name] = sorted(ids[:count])
    return out


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, track_spec
    from .e12_agent import HEAD_CHUNK, host_dtype
    from .e12_faithfulness import TextCache
    import yaml
    run_dir = Path(args.run)
    run_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    track, family = run_config.get("e9_track") or "t5", run_config.get("e9_family") or "smollm2"
    output = Path(args.output or run_dir / LOOP_OUTPUT)
    config = {"experiment": "e12-critique-loop", "run": str(run_dir), "store": str(args.store or run_dir), "items": args.items,
              "host_dtype": args.host_dtype, "label": args.label}
    if args.overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config)
    store = sqx.load_store(Path(args.store or run_dir), "own")
    calibrator, seen = fit_calibrator(store, cap=args.seen_cap)
    alias_table = args.alias_table or ensure_alias_table(track_spec(track, family))
    with host_dtype(args.host_dtype):
        run = open_run(run_dir, device=args.device, batch_size=args.batch_size, max_length=args.max_length, alias_table=alias_table)
    lexicon = sqx.lexicon_for_track(track, family, run.ontology)
    critique = Critique(run, store, calibrator, lexicon, max_lines=args.max_lines)
    new = sqx.load_item_set(args.new_words or ROOT / "items" / f"new-words-{track}-{family}-v2", limit=NEW_WORDS)
    held = wpc7_heldout(args.heldout or WPC7, run.ontology, run.table)
    chosen = loop_items({"new": new, "heldout": held}, count=args.items)
    evidence = load_evidence(args.evidence)
    exclude = {int(c["entry"]) for c in held.concepts}
    demos = loop_demonstrations(critique, seen, exclude=exclude)
    rows, started = [], time.monotonic()
    for name, item_set in (("new", new), ("heldout", held)):
        keep = set(chosen[name])
        item_set.prompts = [p for p in item_set.prompts if p.id in keep]
        item_set.concepts = [c for c in item_set.concepts if c["concept"] in {p.concept for p in item_set.prompts}]
        real, null = critique.frames(item_set)
        real_vectors, null_vectors = critique.vectors(real, item_set, real=True), critique.vectors(null, item_set, real=False)
        beliefs = {"real": critique.beliefs(item_set, real, real_vectors, real), "null": critique.beliefs(item_set, null, null_vectors, real)}
        with sqx.host_view(run, item_set) as (adapter, _):
            behaviour, _, _ = sqx.score_prompts(adapter, item_set.prompts, {})
            none = {r["id"]: _mean_pmi(r) for r in behaviour}
            cache = TextCache(adapter, head_chunk=HEAD_CHUNK)
            for p in item_set.prompts:
                if p.id not in none:
                    continue
                own = int(np.argmax(none[p.id]))
                question = agent_question(p.templates[0], p.fills["x"], p.candidates)
                ev = evidence.get(p.id, {}).get("sentence") if name == "heldout" else None
                record = {"id": p.id, "set": name, "gold": p.gold, "own": own, "none": none[p.id], "null_option": _null_option(critique, p, null),
                          "evidence": bool(ev), "worlds": {}}
                for world in ("real", "null"):
                    belief = beliefs[world].get(p.id) or {}
                    if belief.get("answer") is None:
                        recall = "(nothing recalled)"
                    else:
                        atom = critique.atomic_id[belief["atom"]]
                        recall = recall_statement(critique.writer, p.fills["x"], critique.relation_id[p.row["relation"]], atom, belief["p"])
                    prefix = LOOP_INSTRUCTION + "\n" + demos + "\n" + critique_case(question, p.candidates[own].strip(), recall, ev)
                    decision, _ = cache.scores([(prefix, d) for d in DECISIONS])
                    others = [k for k in range(len(p.candidates)) if k != own]
                    revise, _ = cache.scores([(prefix + " Revise to", p.candidates[k]) for k in others])
                    pd = softmax(decision)
                    pr = softmax(revise) if len(others) else np.zeros(0)
                    choice = int(np.argmax(pd))
                    if choice == 0:
                        answer, confidence = own, float(pd[0])
                    elif choice == 1 and len(others):
                        answer, confidence = others[int(np.argmax(pr))], float(pd[1] * pr.max())
                    else:
                        answer, confidence = None, 0.0
                    record["worlds"][world] = {"belief": belief, "decision": DECISIONS[choice].strip(), "answer": answer, "confidence": confidence,
                                               "decision_scores": decision.round(4).tolist(), "revise_scores": revise.round(4).tolist()}
                rows.append(record)
    seconds = time.monotonic() - started
    und.write_jsonl_gz(output / "items.jsonl.gz", rows)
    summary = {"source": run.describe(), "store": store.describe(), "label": args.label, "items": len(rows), "seconds": seconds,
               "demonstrations": demos, "loop": summarize_loop(rows)}
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() and run.device.type == "cuda" else None
    summary["peak_gb"] = peak
    write_json(output / "summary.json", summary)
    (output / "report.md").write_text(f"# E12 3c — model loop, {run.describe()['condition']} seed {run.describe()['seed']}"
                                      + (f" — {args.label}" if args.label else "") + "\n\n```\n" + json.dumps(summary["loop"], indent=2) + "\n```\n")
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe(), peak_gb=peak)
    return {"output": str(output), "seconds": seconds, "items": len(rows), "loop": summary["loop"]}


def loop_units(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per item: K1 units of the model loop and of the no-tool answer (real world), K2 units of the model loop and of naive
    recall (null world), each method abstaining on its own lowest-confidence 20% (Unsure: no answer)."""
    out: dict[str, dict[str, float]] = {}
    if not rows:
        return out
    real = {"answer": [r["worlds"]["real"]["answer"] for r in rows], "confidence": [r["worlds"]["real"]["confidence"] for r in rows],
            "flagged": [False] * len(rows)}
    null = {"answer": [r["worlds"]["null"]["answer"] for r in rows], "confidence": [r["worlds"]["null"]["confidence"] for r in rows],
            "flagged": [False] * len(rows)}
    base = {"answer": [r["own"] for r in rows], "confidence": [float(softmax(r["none"]).max()) for r in rows], "flagged": [False] * len(rows)}
    naive_answers = [r["worlds"]["null"]["belief"].get("answer") if r["worlds"]["null"]["belief"] else None for r in rows]
    naive = {"answer": [a if a is not None else r["own"] for a, r in zip(naive_answers, rows)],
             "confidence": [(r["worlds"]["null"]["belief"] or {}).get("p") or 0.0 for r in rows], "flagged": [False] * len(rows)}
    items = [{"gold": r["gold"], "null_option": r["null_option"]} for r in rows]
    s = {k: scored(items, m) for k, m in (("loop_real", real), ("loop_null", null), ("no_tool", base), ("naive", naive))}
    for i, r in enumerate(rows):
        out[r["id"]] = {"k1_loop": float(s["loop_real"]["k1"][i]), "k1_no_tool": float(s["no_tool"]["k1"][i]),
                        "k2_loop": float(s["loop_null"]["k2"][i]), "k2_naive": float(s["naive"]["k2"][i])}
    return out


def summarize_loop(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    units = loop_units(rows)
    if not units:
        return {}
    mean = lambda key: float(np.mean([u[key] for u in units.values()]))
    decisions = {w: dict(Counter(r["worlds"][w]["decision"] for r in rows)) for w in ("real", "null")}
    return {"items": len(rows), "k1_loop": mean("k1_loop"), "k1_no_tool": mean("k1_no_tool"), "k2_loop": mean("k2_loop"),
            "k2_naive": mean("k2_naive"), "decisions": decisions}


def analyse_loops(runs_root: Path, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    from .e12_report import _holm, contrast
    found = discover(runs_root, folder=LOOP_OUTPUT)
    out: dict[str, Any] = {}
    for host, models in found.items():
        for model, seeds in models.items():
            per = {s: loop_units(und.read_jsonl(folder / "items.jsonl")) for s, folder in seeds.items()}
            pick = lambda key: {s: {k: v[key] for k, v in u.items()} for s, u in per.items()}
            block = {"K1 (model loop − no tool)": contrast(pick("k1_loop"), pick("k1_no_tool"), resamples=resamples, seed=seed),
                     "K2 (model loop − naive recall)": contrast(pick("k2_loop"), pick("k2_naive"), resamples=resamples, seed=seed)}
            _holm(block)
            for k, v in block.items():
                out[f"{host} {model}: {k}"] = v
    return out


# ---------------------------------------------------------------- queue commands (printed; never queued here)


def job_commands(*, priority: float = 54.4498, python: str = "$PY", root: Path = ROOT) -> list[dict[str, Any]]:
    """3c's GPU jobs: T5 SmolLM2-360M C5 (every set) and C5ut (twins and new words: the null world's role-blind control) ×
    seeds 1–3; the model loop on Qwen3-1.7B-Base C5 × seeds 1–2."""
    jobs = []
    for model, sets in (("C5", "twins,new,heldout"), ("C5ut", "twins,new")):
        for s in (1, 2, 3):
            run = root / "runs" / "t5" / f"SmolLM2-360M-full-{model}-s{s}"
            jobs.append({"name": f"t5-{run.name}-{OUTPUT}", "priority": priority, "lane": "gpu",
                         "command": [python, "-m", "vsa_embed.experiments.e12_critique", "evaluate", "--run", str(run), "--sets", sets,
                                     "--batch-size", "24", "--max-length", "1024", "--overwrite"]})
    for s in (1, 2):
        run = root / "runs" / "t5-qwen3" / f"Qwen3-1.7B-Base-lora-C5-s{s}"
        jobs.append({"name": f"t5-qwen3-{run.name}-{LOOP_OUTPUT}", "priority": priority, "lane": "gpu",
                     "command": [python, "-m", "vsa_embed.experiments.e12_critique", "loop", "--run", str(run), "--items", "300",
                                 "--host-dtype", "bfloat16", "--batch-size", "4", "--max-length", "4096", "--overwrite"]})
    return jobs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evidence", help="build the held-out terms' evidence sentences (CPU, once)")
    ev.add_argument("--output", type=Path, default=EVIDENCE); ev.add_argument("--items", type=Path, default=WPC7)
    ev.add_argument("--docs", type=Path, default=EVAL_DOCS)
    for name in ("evaluate", "loop"):
        p = sub.add_parser(name, help="score the critique inputs of one run (GPU)" if name == "evaluate" else "the model loop on one run (GPU)")
        p.add_argument("--run", type=Path, required=True); p.add_argument("--store", type=Path, default=None)
        p.add_argument("--output", type=Path, default=None); p.add_argument("--alias-table", type=Path, default=None)
        p.add_argument("--twins", type=Path, default=None); p.add_argument("--new-words", type=Path, default=None)
        p.add_argument("--heldout", type=Path, default=None); p.add_argument("--evidence", type=Path, default=EVIDENCE)
        p.add_argument("--seen-cap", type=int, default=SEEN_CAP); p.add_argument("--device", default=None)
        p.add_argument("--batch-size", type=int, default=24); p.add_argument("--max-length", type=int, default=1024)
        p.add_argument("--max-lines", type=int, default=32, help="recalled lines per call (phase A: 32)")
        p.add_argument("--label", default=None); p.add_argument("--overwrite", action="store_true")
        if name == "evaluate":
            p.add_argument("--sets", default="twins,new,heldout"); p.add_argument("--tag", default=None)
            p.add_argument("--twin-limit", type=int, default=None, help="smoke tests only: the first N pairs")
            p.add_argument("--new-limit", type=int, default=NEW_WORDS, help="the first N new words (pre-registered: 300)")
            p.add_argument("--heldout-limit", type=int, default=None, help="smoke tests only: the first N held-out terms")
            p.add_argument("--competitors", default="auto", choices=("auto", "on", "off"),
                           help="decision 64: score the symbolic / definition competitors on the new words and held-out terms "
                                "(auto: when the store carries roles)")
        else:
            p.add_argument("--items", type=int, default=300, help="items per set (new-word test half; held-out terms)")
            p.add_argument("--host-dtype", default=None, choices=[None, "bfloat16", "float16"])
    re_ = sub.add_parser("report", help="the rule loop, K1, K2 and calibration across a stage's runs (CPU)")
    re_.add_argument("--runs", type=Path, required=True); re_.add_argument("--output", type=Path, required=True)
    re_.add_argument("--hosts", nargs="*", default=None); re_.add_argument("--loop-runs", type=Path, default=None)
    re_.add_argument("--resamples", type=int, default=2000); re_.add_argument("--label", default=None)
    re_.add_argument("--title", default="E12 3c — calibrated self-critique"); re_.add_argument("--overwrite", action="store_true")
    sub.add_parser("commands", help="print the GPU jobs (never queues)")
    args = parser.parse_args(argv)
    if args.command == "evidence":
        print(json.dumps(build_evidence(args.output, items=args.items, docs=args.docs)["counts"], indent=2))
    elif args.command == "evaluate":
        print(json.dumps(run_evaluate(args), indent=2, default=str))
    elif args.command == "loop":
        print(json.dumps(run_loop(args), indent=2, default=str))
    elif args.command == "report":
        print(json.dumps(run_report(args)))
    else:
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in job_commands()], indent=2))


if __name__ == "__main__":
    main()
