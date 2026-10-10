"""C3 teacher ontology (decision 65, phase 1b; arm C5teach): WordNet frames written by Claude from each sense's gloss.

The arm differs from C5 only in which edges each entry has: the same 101,500 entries (C3 `ontology.pt`), the same 16
relation types, fillers drawn from the same 8,192 atomics, the same held-out entries and training frequencies. Only the
`offsets / relations / fillers` arrays are replaced.

**Teacher.** Claude (`claude -p`, model pinned, `--effort` recorded) through E7's `authoring_baselines.TeacherAuthor`
(cache by prompt, model ids and USD recorded per call), as E11's `read_teacher` uses it; `CappedTeacher` adds the C3
prompt, a short fixed system prompt, a per-call `--max-budget-usd`, counting of failed calls' spend, parallel workers and a
HARD cumulative cap (`--max-usd`): before each call it reserves a bound on the call's cost and launches only while
`spent + in-flight reservations + the next reservation ≤ cap`; the ledger (`ledger.jsonl`, every attempt with its USD,
tokens and seconds) is the store's spend record, so the cap holds across resumed invocations.

**Unit.** One item per WordNet synset (a sense: its first lemma, its part of speech and its gloss = definition and
examples; never the synset id or the synonym list), batched `--batch` per call with opaque ids (`s1`, …). An entry's teacher
frame is the union of its synsets' teacher frames, the rule that built C3's WordNet entry frames
(`AliasTable.entry_schedule`), so synsets shared by several entries are read once (117,659 synsets for 101,500 entries).

**Parser** (`FillerMapper`): fillers are asked for as WordNet synset names (`lemma.pos.NN`). A filler that is an atomic maps
exactly (`exact`); an existing synset outside the 8,192-atomic dictionary maps to its nearest in-dictionary hypernym, the
rule the WordNet builder applied to WordNet's own pointers (`ancestor`, `ontologies.wordnet.build_wordnet_ontology`); a
non-existent sense number or a bare word falls back to the word's senses of the part of speech the relation expects, first
atomic sense or its in-dictionary ancestor (`lemma`, `lemma_ancestor`); `lexname` fillers must be one of the 45
lexicographer files and `pos` fillers one of n / v / a / s / r. Everything else is recorded as unmapped with a reason
(`unknown_lemma`, `no_ancestor`, `bad_lexname`, `bad_pos`, `ill_typed`, `self`). A synset frame keeps one lexname and one
pos, lists edges in the builder's order (lexname, pos, pointers in `POINTERS` order) and is capped at 16 edges (the C3
`max_degree`). A synset whose answer maps to no edge keeps a pos-only frame (the part of speech shown in the prompt),
counted as `pos_fallback`.

**Commands** (the full run is one command, resumable and cached; never queued by the agent that wrote it):

    python -m vsa_embed.experiments.c3_teacher sample   --output experiments/e4-small-lm/scale-v1-toolkit/teacher-pilot
    python -m vsa_embed.experiments.c3_teacher teach    --scope pilot --sample …/teacher-pilot/sample.json --max-usd 5
    python -m vsa_embed.experiments.c3_teacher evaluate --sample …/teacher-pilot/sample.json --output …/teacher-pilot
    python -m vsa_embed.experiments.c3_teacher teach    --scope full --max-usd <approved>     # or --scope hybrid
    python -m vsa_embed.experiments.c3_teacher ontology --scope full                          # → $STORE/ontology-full.pt
    python -m vsa_embed.experiments.c3_teacher project  --pilot …/teacher-pilot

`teach` skips every synset with a valid answer in `$STORE/answers.jsonl` (append-only; the latest line per synset wins), so
an interrupted or capped run resumes where it stopped. `hybrid` = the 5,900 held-out entries plus 5,900 trained entries
matched on evaluation-corpus frequency band × sense-count band (`hybrid_entries`); its ontology keeps C3's WordNet frames
elsewhere, and `reference` writes the evaluation strata of the two teacher entry sets (`eval.reference_strata`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..authoring_baselines import TeacherAuthor, teacher_schema
from ..ontologies.wordnet import POINTERS

C3_ROOT = Path("~/data/vsa-llm/c3/wordnet-gpt2-v1").expanduser()
STORE = Path("~/data/vsa-llm/c3-teacher/wordnet-gpt2-v1").expanduser()
PILOT_DIR = Path("experiments/e4-small-lm/scale-v1-toolkit/teacher-pilot")
PROMPT_VERSION = "c3-teacher-v1"
MODEL = "claude-opus-5-5"
MAX_DEGREE = 16                       # C3 `ontology.max_degree` (per synset frame)
SCOPES = ("pilot", "full", "hybrid")
# Pilot strata (decision 65): held-out entries by evaluation-corpus span count (they have no training frequency), trained
# entries by training frequency (the trainer's after_unseen / after_rare_seen / after_mid / after_frequent split, ≥ 100 cut).
HELDOUT_BANDS = ((0, 1), (1, 3), (3, 10), (10, 30), (30, None))
TRAINED_BANDS = ((0, 1), (1, 10), (10, 100), (100, 1000), (1000, None))
SENSE_BANDS = ((1, 2), (2, 4), (4, None))
POS_LABEL = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective (satellite)", "r": "adverb"}
POS_WORDS = {"n": "n", "noun": "n", "v": "v", "verb": "v", "a": "a", "adj": "a", "adjective": "a", "s": "s",
             "satellite": "s", "adjective satellite": "s", "satellite adjective": "s", "adjective (satellite)": "s",
             "r": "r", "adv": "r", "adverb": "r"}
# Part of speech a relation's filler has (the bare-word fallback); "same" = the sense's own, "adj" = a / s.
EXPECTED_POS = {"hypernym": "same", "instance_hypernym": "n", "part_meronym": "n", "member_meronym": "n",
                "substance_meronym": "n", "part_holonym": "n", "member_holonym": "n", "substance_holonym": "n",
                "attribute": "flip", "similar_to": "adj", "topic_domain": "n", "entailment": "v", "cause": "v",
                "antonym": "same"}
RELATION_GUIDE = {
    "hypernym": "a more general concept this sense is a kind of (for a verb: the more general action); one or two",
    "instance_hypernym": "for a named individual (a person, place, organization, event or work): the class it is an instance of; "
                         "use it instead of hypernym",
    "part_meronym": "a part of it",
    "member_meronym": "a member of it (for a group, taxon or collection)",
    "substance_meronym": "a substance it is made of",
    "part_holonym": "a whole it is a part of",
    "member_holonym": "a group, taxon or collection it is a member of",
    "substance_holonym": "a thing it is a substance or ingredient of",
    "attribute": "for an adjective: the noun attribute it is a value of (heavy -> weight); for such a noun: its adjective values",
    "similar_to": "for an adjective: a closely similar adjective",
    "topic_domain": "the field or domain the sense belongs to (e.g. medicine, law, music, sport)",
    "entailment": "for a verb: an action it entails (snore -> sleep)",
    "cause": "for a verb: the action or state it causes (kill -> die)",
    "antonym": "a direct opposite",
    "lexname": "the WordNet lexicographer file of the sense, exactly one of: {lexnames}",
    "pos": "the part of speech: n (noun), v (verb), a (adjective), s (satellite adjective) or r (adverb)",
}
SYSTEM_PROMPT = ("You are an expert lexicographer who knows WordNet 3.0. You write WordNet-style frames for word senses from "
                 "their dictionary glosses. Reply only through the JSON schema; no other text.")


class BudgetReached(RuntimeError):
    """The cumulative cap would be exceeded by the next call."""


# -- WordNet senses ---------------------------------------------------------------------------------------------------

def wordnet():
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()              # the lazy loader is not thread-safe: load before any worker thread touches it
    return wn


def gloss(synset: Any) -> str:
    """WordNet's gloss: the definition and the example sentences (in quotes)."""
    examples = "; ".join(f'"{e}"' for e in synset.examples())
    return synset.definition() + (f"; {examples}" if examples else "")


def sense_item(synset: Any, item_id: str) -> dict[str, Any]:
    """One prompt item: the sense's first lemma, part of speech and gloss (no synset id, no synonym list)."""
    return {"id": item_id, "surface": synset.lemma_names()[0].replace("_", " "), "pos": POS_LABEL[synset.pos()],
            "contexts": [gloss(synset)]}


def c3_prompt(items: Sequence[dict[str, Any]], relations: Sequence[str], lexnames: Sequence[str], *, max_edges: int = 10) -> str:
    guide = "\n".join(f"- {r}: {RELATION_GUIDE[r].format(lexnames=', '.join(lexnames))}" for r in relations)
    lines = ["For each word sense below (a word, its part of speech and its gloss: definition; examples in quotes), list the "
             "facts a WordNet lexicographer would record about that sense, as (relation, filler) pairs, using ONLY these "
             "relations:", guide, "",
             "Write every filler except those of lexname and pos as a WordNet synset name lemma.pos.NN (e.g. canine.n.02, "
             "travel.v.01, large.a.01), with the sense number you believe is right. Always give lexname and pos; give the "
             f"other relations only where they hold and are salient (typically 1 to 4 facts besides lexname and pos, never more "
             f"than {max_edges} in all). Answer with JSON matching the schema, one entry per sense id.", "", "Senses:"]
    for item in items:
        lines.append(f"{item['id']} | {item['surface']} | {item['pos']} | {' '.join(item['contexts'][0].split())}")
    return "\n".join(lines) + "\n"


# -- the teacher (E7's TeacherAuthor with the C3 prompt, a cap and workers) ---------------------------------------------

def teacher_cli_runner(prompt: str, schema: dict[str, Any], model: str, *, system_prompt: str | None = None,
                       effort: str | None = None, max_budget_usd: float | None = None, timeout: int = 900) -> dict[str, Any]:
    """`judging.claude_cli_runner` plus a system prompt, an effort level and a per-call budget. A non-zero exit that still
    printed the JSON envelope (e.g. the per-call budget reached) returns the envelope, so its cost is counted."""
    command = ["claude", "-p", prompt, "--output-format", "json", "--model", model, "--tools", "", "--setting-sources", "",
               "--no-session-persistence", "--json-schema", json.dumps(schema)]
    if system_prompt:
        command += ["--system-prompt", system_prompt]
    if effort:
        command += ["--effort", effort]
    if max_budget_usd:
        command += ["--max-budget-usd", f"{max_budget_usd:.4f}"]
    with tempfile.TemporaryDirectory() as empty:
        completed = subprocess.run(command, cwd=empty, capture_output=True, text=True, timeout=timeout)
    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude exited {completed.returncode} without an envelope: {completed.stderr[:300]}") from None
    if completed.returncode != 0:
        envelope.setdefault("is_error", True)
    return envelope


@dataclass
class CappedTeacher(TeacherAuthor):
    """`TeacherAuthor` with the C3 prompt, a cumulative USD cap over a ledger, and parallel batches."""

    lexnames: Sequence[str] = ()
    system_prompt: str = SYSTEM_PROMPT
    effort: str | None = "medium"
    prompt_version: str = PROMPT_VERSION
    max_usd: float | None = None
    reserve_usd: float = 0.30            # the first calls' cost bound; then 2 × the largest call seen
    ledger: Path | None = None
    in_flight: float = field(default=0.0, init=False)
    lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ledger is not None and Path(self.ledger).exists():
            rows = [json.loads(line) for line in Path(self.ledger).read_text().splitlines() if line.strip()]
            self.spent_usd = sum(float(r.get("usd") or 0) for r in rows)
            costs = [float(r.get("usd") or 0) for r in rows]
            if costs:
                self.reserve_usd = max(self.reserve_usd, 2 * max(costs))

    def _key(self, prompt: str, schema: dict[str, Any]) -> str:
        payload = {"model": self.model, "prompt": prompt, "schema": schema, "system": self.system_prompt,
                   "effort": self.effort, "version": self.prompt_version}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _log(self, row: dict[str, Any]) -> None:
        if self.ledger is not None:
            Path(self.ledger).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.ledger).open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    def _reserve(self) -> float:
        with self.lock:
            reserve = self.reserve_usd
            if self.max_usd is not None and self.spent_usd + self.in_flight + reserve > self.max_usd:
                raise BudgetReached(f"spent {self.spent_usd:.4f} + in flight {self.in_flight:.4f} + next {reserve:.4f} "
                                    f"> cap {self.max_usd:.2f}")
            self.in_flight += reserve
            return reserve

    def request(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prompt = c3_prompt(items, self.relations, self.lexnames)
        schema = teacher_schema(self.relations)
        key = self._key(prompt, schema)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with self.lock:
                self.cached += 1
            return json.loads(path.read_text())
        runner = self.runner or teacher_cli_runner
        record: dict[str, Any] = {"model": self.model, "effort": self.effort, "prompt_version": self.prompt_version,
                                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "items": [i["id"] for i in items]}
        for attempt in range(self.retries + 1):
            reserve = self._reserve()           # BudgetReached propagates: nothing was spent on this attempt
            started, cost, envelope, error = time.monotonic(), 0.0, {}, None
            try:
                envelope = runner(prompt, schema, self.model, system_prompt=self.system_prompt, effort=self.effort,
                                  max_budget_usd=reserve)
                cost = float(envelope.get("total_cost_usd") or 0)
                verdict = envelope.get("structured_output")
                if not isinstance(verdict, dict) or envelope.get("is_error") or not isinstance(verdict.get("concepts"), list):
                    raise ValueError(f"no structured output ({envelope.get('subtype')})")
                record.update(verdict=verdict, cost_usd=cost, models_used=sorted((envelope.get("modelUsage") or {}).keys()),
                              attempts=attempt + 1, usage=envelope.get("usage"))
            except (ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as caught:
                error = str(caught)[:300]
                if not envelope:            # no envelope: the cost is unknown, so the whole reservation is charged
                    cost = reserve
                record.update(error=error, attempts=attempt + 1)
            seconds = time.monotonic() - started
            with self.lock:
                self.in_flight -= reserve
                self.spent_usd += cost
                self.calls += 1
                self.reserve_usd = max(self.reserve_usd, 2 * cost)
                self._log({"key": key, "items": len(items), "attempt": attempt + 1, "usd": cost, "seconds": round(seconds, 2),
                           "usage": envelope.get("usage"), "models": sorted((envelope.get("modelUsage") or {}).keys()),
                           "error": error, "time": time.time()})
            if "verdict" in record:
                break
        if "verdict" in record:
            record.pop("error", None)
            path.write_text(json.dumps(record, indent=2) + "\n")
        return record


def answers_of(record: dict[str, Any], items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Item id → {"edges": [(relation, raw filler)], "error"} (relations outside the vocabulary are dropped by the schema)."""
    if "verdict" not in record:
        return {i["id"]: {"edges": [], "error": record.get("error") or "no verdict"} for i in items}
    answered = {c.get("id"): c.get("edges", []) for c in record["verdict"]["concepts"] if isinstance(c, dict)}
    out = {}
    for item in items:
        if item["id"] not in answered:
            out[item["id"]] = {"edges": [], "error": "missing from the answer"}
            continue
        edges = [(str(e.get("relation")), str(e.get("filler", "")).strip()) for e in answered[item["id"]]
                 if isinstance(e, dict) and e.get("relation") and str(e.get("filler", "")).strip()]
        out[item["id"]] = {"edges": list(dict.fromkeys(edges)), "error": None}
    return out


# -- answers store ------------------------------------------------------------------------------------------------------

def load_answers(store: Path) -> dict[str, dict[str, Any]]:
    """Synset name → its latest answer line (`answers.jsonl`; an errored line is superseded by a later valid one)."""
    path = Path(store) / "answers.jsonl"
    out: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("error") is None or row["synset"] not in out or out[row["synset"]].get("error") is not None:
                    out[row["synset"]] = row
    return out


def teach(synsets: Sequence[str], *, store: Path = STORE, max_usd: float, batch: int = 25, workers: int = 4,
          effort: str = "medium", model: str = MODEL, runner: Callable | None = None, relations: Sequence[str] | None = None,
          lexnames: Sequence[str] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Teacher answers for every synset without a valid one in the store, `batch` per call, `workers` calls at a time,
    until done or the cumulative cap is reached. Returns counts and the spend."""
    wn = wordnet()
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    onto_relations, onto_lexnames = _c3_vocabulary()
    teacher = CappedTeacher(store / "calls", list(relations or onto_relations), {}, model=model, runner=runner, retries=1,
                            lexnames=list(lexnames or onto_lexnames), effort=effort, max_usd=max_usd, ledger=store / "ledger.jsonl")
    done = {s for s, row in load_answers(store).items() if row.get("error") is None}
    todo = [s for s in sorted(set(synsets)) if s not in done]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    log(json.dumps({"synsets": len(set(synsets)), "answered": len(set(synsets)) - len(todo), "calls_needed": len(batches),
                    "spent_before": round(teacher.spent_usd, 4), "cap": max_usd}))
    lock, stats, started = threading.Lock(), Counter(), time.monotonic()
    answers_path = store / "answers.jsonl"

    poses = {n: wn.synset(n).pos() for n in todo}
    items_of = {tuple(names): [sense_item(wn.synset(n), f"s{k + 1}") for k, n in enumerate(names)] for names in batches}

    def run(names: list[str]) -> None:
        items = items_of[tuple(names)]
        record = teacher.request(items)
        answered = answers_of(record, items)
        rows = [{"synset": n, "word": i["surface"], "pos": poses[n], "edges": answered[i["id"]]["edges"],
                 "error": answered[i["id"]]["error"], "prompt_version": PROMPT_VERSION, "model": model, "effort": effort}
                for n, i in zip(names, items)]
        with lock:
            with answers_path.open("a") as handle:
                handle.write("".join(json.dumps(r) + "\n" for r in rows))
            stats["answered"] += sum(r["error"] is None for r in rows)
            stats["errors"] += sum(r["error"] is not None for r in rows)
            stats["batches"] += 1

    stopped = None
    pending, running = iter(batches), set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:

        def launch() -> None:
            names = next(pending, None)
            if names is not None:
                running.add(pool.submit(run, names))

        for _ in range(max(1, workers)):            # keep `workers` calls in flight; stop launching at the cap
            launch()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                running.discard(future)
                try:
                    future.result()
                except BudgetReached as caught:
                    stopped = str(caught)
                if stopped is None:
                    launch()
                if stats["batches"] and stats["batches"] % 10 == 0:
                    log(json.dumps({"batches": stats["batches"], "spent": round(teacher.spent_usd, 4)}))
    result = {"synsets": len(set(synsets)), "todo": len(todo), "answered": stats["answered"], "errors": stats["errors"],
              "batches": stats["batches"], "spent_usd_total": round(teacher.spent_usd, 4), "calls": teacher.calls,
              "cached": teacher.cached, "seconds": round(time.monotonic() - started, 1), "stopped": stopped,
              "model": model, "effort": effort, "batch": batch, "workers": workers, "time": time.time()}
    with (store / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(result) + "\n")
    return result


# -- parser: fillers → atomics --------------------------------------------------------------------------------------

SYNSET_NAME = re.compile(r"^(?P<lemma>.+)\.(?P<pos>[nvasr])\.(?P<num>\d{1,3})$")


class FillerMapper:
    """(relation, filler text) of a sense → an atomic id with how it was mapped, or None with the reason."""

    def __init__(self, atomic_names: Sequence[str], relation_names: Sequence[str], wn: Any | None = None) -> None:
        self.wn = wn or wordnet()
        self.relation_id = {r: i for i, r in enumerate(relation_names)}
        self.atomic_index = {name: i for i, name in enumerate(atomic_names)}
        self.lexname = {n.split(":", 1)[1].lower(): i for n, i in self.atomic_index.items() if n.startswith("lexname:")}
        self.pos = {n.split(":", 1)[1]: i for n, i in self.atomic_index.items() if n.startswith("pos:")}
        self._ancestor: dict[str, int | None] = {}

    def in_dictionary(self, synset: Any, depth: int = 0) -> int | None:
        """The builder's rule: the synset's atomic, else its nearest in-dictionary hypernym (depth < 12)."""
        name = synset.name()
        if f"synset:{name}" in self.atomic_index:
            return self.atomic_index[f"synset:{name}"]
        if name in self._ancestor:
            return self._ancestor[name]
        result = None
        if depth < 12:
            for parent in sorted(synset.hypernyms() + synset.instance_hypernyms(), key=lambda s: s.name()):
                result = self.in_dictionary(parent, depth + 1)
                if result is not None:
                    break
        self._ancestor[name] = result
        return result

    def _synset(self, name: str) -> Any | None:
        try:
            return self.wn.synset(name)
        except Exception:          # nltk raises WordNetError / ValueError for unknown names
            return None

    def _expected_pos(self, relation: str, own_pos: str) -> tuple[str, ...]:
        rule = EXPECTED_POS.get(relation, "same")
        own = "a" if own_pos == "s" else own_pos
        if rule == "same":
            return (own,)
        if rule == "flip":
            return ("n",) if own == "a" else ("a",)
        if rule == "adj":
            return ("a",)
        return (rule,)

    def _by_lemma(self, lemma: str, poses: Iterable[str]) -> tuple[int | None, str]:
        candidates = [s for p in poses for s in self.wn.synsets(lemma.replace(" ", "_"), pos=p)]
        if not candidates:
            return None, "unknown_lemma"
        for synset in candidates:
            if f"synset:{synset.name()}" in self.atomic_index:
                return self.atomic_index[f"synset:{synset.name()}"], "lemma"
        atom = self.in_dictionary(candidates[0])
        return (atom, "lemma_ancestor") if atom is not None else (None, "no_ancestor")

    def map(self, relation: str, filler: str, *, own: str) -> tuple[int | None, str]:
        """Atomic id and how (`exact` / `ancestor` / `lemma` / `lemma_ancestor` / `lexname` / `pos`), or (None, reason)."""
        text = " ".join(str(filler).strip().strip("'\"`").split())
        key = text.lower()
        own_pos = own.split(".")[-2] if own.count(".") >= 2 else "n"
        if relation == "lexname":
            key = key.removeprefix("lexname:")
            return (self.lexname[key], "lexname") if key in self.lexname else (None, "bad_lexname")
        if relation == "pos":
            key = POS_WORDS.get(key.removeprefix("pos:"))
            return (self.pos[key], "pos") if key in self.pos else (None, "bad_pos")
        if relation not in self.relation_id:
            return None, "unknown_relation"
        if key.removeprefix("lexname:") in self.lexname or key.startswith(("lexname:", "pos:")):
            return None, "ill_typed"
        name = key.removeprefix("synset:").replace(" ", "_")
        match = SYNSET_NAME.match(name)
        if match:
            synset = self._synset(f"{match['lemma']}.{match['pos']}.{int(match['num']):02d}")
            if synset is not None:
                atom = self.in_dictionary(synset)
                how = "exact" if atom is not None and self.atomic_index.get(f"synset:{synset.name()}") == atom else "ancestor"
                result = (atom, how) if atom is not None else (None, "no_ancestor")
            else:
                result = self._by_lemma(match["lemma"], ("a",) if match["pos"] in "as" else (match["pos"],))
        else:
            result = self._by_lemma(name, self._expected_pos(relation, own_pos))
        if result[0] is not None and self.atomic_index.get(f"synset:{own}") == result[0]:
            return None, "self"
        return result

    def frame(self, synset: str, edges: Sequence[tuple[str, str]], *, max_degree: int = MAX_DEGREE
              ) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
        """A synset's teacher frame (builder order, one lexname / pos, ≤ `max_degree` edges) and one record per edge."""
        records, by_relation = [], defaultdict(list)
        for relation, filler in edges:
            atom, how = self.map(relation, filler, own=synset)
            records.append({"relation": relation, "filler": filler, "atom": atom, "how": how})
            if atom is not None:
                by_relation[relation].append(atom)
        order = ["lexname", "pos", *POINTERS]
        frame: list[tuple[int, int]] = []
        for relation in order:
            atoms = by_relation.get(relation, [])
            for atom in (atoms[:1] if relation in ("lexname", "pos") else atoms):
                edge = (self.relation_id[relation], atom)
                if edge not in frame and len(frame) < max_degree:
                    frame.append(edge)
        return frame, records


# -- C3 ontology, entries, frames ---------------------------------------------------------------------------------------

def load_c3(root: Path = C3_ROOT) -> dict[str, Any]:
    import torch
    return torch.load(Path(root) / "ontology.pt", weights_only=False)


def _c3_vocabulary(root: Path = C3_ROOT) -> tuple[list[str], list[str]]:
    onto = load_c3(root)
    lexnames = [n.split(":", 1)[1] for n in onto["atomic_names"] if n.startswith("lexname:")]
    return list(onto["relation_names"]), lexnames


def entry_frame(ontology: dict[str, Any], entry: int) -> list[tuple[int, int]]:
    offsets = ontology["offsets"]
    lo, hi = int(offsets[entry]), int(offsets[entry + 1])
    return list(zip(ontology["relations"][lo:hi].tolist(), ontology["fillers"][lo:hi].tolist()))


def entry_synsets(ontology: dict[str, Any], entries: Iterable[int]) -> list[str]:
    names = ontology["concept_names"]
    return sorted({names[c] for e in entries for c in ontology["entry_concepts"][int(e)]})


def entry_aliases(ontology: dict[str, Any]) -> dict[int, list[str]]:
    """Entry → its aliases, rebuilt from WordNet's lemmas (same entry numbering as C3; checked)."""
    from ..ontologies.wordnet import build_wordnet_ontology
    from ..span_channel import AliasTable
    built = build_wordnet_ontology(wordnet(), max_atomics=int(ontology["atomic_count"]), max_degree=MAX_DEGREE)
    table = AliasTable.from_pairs(built.alias_pairs, include_holdout=True)
    if [tuple(c) for c in table.entry_concepts] != [tuple(c) for c in ontology["entry_concepts"]]:
        raise ValueError("rebuilt alias table does not match the C3 entry numbering")
    aliases: dict[int, list[str]] = defaultdict(list)
    for alias, entry in sorted(table.alias_to_entry.items()):
        aliases[entry].append(alias)
    return aliases


def synset_frames(answers: dict[str, dict[str, Any]], mapper: FillerMapper, synsets: Iterable[str]
                  ) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    """Teacher frame per synset (pos-only fallback for an answer that maps to nothing) and the parse statistics."""
    frames, how, reasons, stats = {}, Counter(), Counter(), Counter()
    for name in synsets:
        row = answers.get(name)
        if row is None or row.get("error") is not None:
            stats["unanswered"] += 1
            continue
        frame, records = mapper.frame(name, [tuple(e) for e in row["edges"]])
        stats["answered"] += 1
        stats["proposed"] += len(records)
        for r in records:
            (how if r["atom"] is not None else reasons)[r["how"]] += 1
        if not frame:
            stats["pos_fallback"] += 1
            frame = [(mapper.relation_id["pos"], mapper.pos[row["pos"]])]
        frames[name] = frame
    mapped = sum(how.values())
    return frames, {**stats, "mapped": mapped, "parse_rate": mapped / max(1, stats["proposed"]),
                    "mapped_by": dict(how), "unmapped_by": dict(reasons)}


def union_frame(ontology: dict[str, Any], entry: int, frames: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
    """The entry's teacher frame: the union of its synsets' frames (None if a synset has no frame)."""
    union, seen = [], set()
    for concept in ontology["entry_concepts"][entry]:
        frame = frames.get(ontology["concept_names"][concept])
        if frame is None:
            return None
        for edge in frame:
            if edge not in seen:
                seen.add(edge); union.append(edge)
    return union


def teacher_ontology(ontology: dict[str, Any], frames: dict[str, list[tuple[int, int]]], *, entries: Iterable[int] | None,
                     provenance: dict[str, Any]) -> dict[str, Any]:
    """C3's ontology with teacher entry frames for `entries` (all entries if None) and WordNet frames elsewhere."""
    import torch

    from ..compose import FrameSchedule
    chosen = set(range(int(ontology["entry_count"]))) if entries is None else {int(e) for e in entries}
    rows, missing = [], []
    for entry in range(int(ontology["entry_count"])):
        frame = union_frame(ontology, entry, frames) if entry in chosen else entry_frame(ontology, entry)
        if frame is None:
            missing.append(entry)
            frame = []
        rows.append(frame)
    if missing:
        raise ValueError(f"{len(missing)} teacher entries have unanswered synsets (first: {missing[:5]}); run `teach` first")
    schedule = FrameSchedule.from_frames(rows)
    out = dict(ontology)
    out.update(offsets=schedule.offsets, relations=schedule.relations, fillers=schedule.fillers,
               teacher={**provenance, "teacher_entries": len(chosen), "edges": int(schedule.relations.numel()),
                        "base_edges": int(torch.as_tensor(ontology["relations"]).numel())})
    return out


# -- entry sets ---------------------------------------------------------------------------------------------------------

def eval_counts(root: Path = C3_ROOT, *, min_subtokens: int = 2) -> np.ndarray:
    """Spans per entry in the C3 evaluation corpus (ℓ ≥ `min_subtokens`, the trainer's evaluation spans)."""
    from ..data.corpus import TokenCorpus
    corpus = TokenCorpus.open(Path(root) / "eval")
    keep = corpus.spans["length"] >= min_subtokens
    return np.bincount(corpus.spans["entry"][keep], minlength=int(load_c3(root)["entry_count"]))


def band_of(value: int, bands: Sequence[tuple[int, int | None]]) -> int:
    for k, (lo, hi) in enumerate(bands):
        if value >= lo and (hi is None or value < hi):
            return k
    raise ValueError(value)


def band_name(bands: Sequence[tuple[int, int | None]], k: int) -> str:
    lo, hi = bands[k]
    return f"{lo}+" if hi is None else (f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}")


def pilot_sample(ontology: dict[str, Any], counts: np.ndarray, *, per_group: int = 200, seed: int = 20261010
                 ) -> list[dict[str, Any]]:
    """`per_group` held-out entries stratified by evaluation-corpus count band and `per_group` trained entries by training
    frequency band, equal allocation (`per_group / 5` per band; a short band's remainder goes to the next band)."""
    rng = np.random.default_rng(seed)
    held = sorted(int(e) for e in ontology["heldout_entries"])
    held_set = set(held)
    frequency = np.asarray(ontology["train_frequency"])
    trained = [e for e in range(int(ontology["entry_count"])) if e not in held_set]
    out = []
    for group, pool, value, bands in (("heldout", held, counts, HELDOUT_BANDS), ("trained", trained, frequency, TRAINED_BANDS)):
        by_band: dict[int, list[int]] = defaultdict(list)
        for e in pool:
            by_band[band_of(int(value[e]), bands)].append(e)
        carry = 0
        for k in range(len(bands)):
            want = per_group // len(bands) + carry
            members = by_band.get(k, [])
            take = sorted(rng.choice(members, size=min(want, len(members)), replace=False).tolist()) if members else []
            carry = want - len(take)
            out += [{"entry": int(e), "group": group, "band": band_name(bands, k), "value": int(value[e]),
                     "senses": len(ontology["entry_concepts"][e])} for e in take]
    return out


def hybrid_entries(ontology: dict[str, Any], counts: np.ndarray, *, seed: int = 20261011) -> dict[str, list[int]]:
    """The fair cheaper variant: every held-out entry and as many trained entries (seen in training) matched on
    evaluation-count band × sense-count band; a cell short of trained entries is filled from the same count band."""
    rng = np.random.default_rng(seed)
    held = sorted(int(e) for e in ontology["heldout_entries"])
    held_set = set(held)
    frequency = np.asarray(ontology["train_frequency"])
    senses = [len(c) for c in ontology["entry_concepts"]]
    cell = lambda e: (band_of(int(counts[e]), HELDOUT_BANDS), band_of(senses[e], SENSE_BANDS))
    need = Counter(cell(e) for e in held)
    pool: dict[tuple[int, int], list[int]] = defaultdict(list)
    for e in range(int(ontology["entry_count"])):
        if e not in held_set and frequency[e] > 0:
            pool[cell(e)].append(e)
    chosen: list[int] = []
    short: Counter = Counter()
    for key in sorted(need):
        members = pool[key]
        take = rng.choice(members, size=min(need[key], len(members)), replace=False).tolist() if members else []
        chosen += take
        short[key[0]] += need[key] - len(take)
    taken = set(chosen)
    for band, missing in sorted(short.items()):
        rest = [e for key, members in pool.items() if key[0] == band for e in members if e not in taken]
        extra = rng.choice(rest, size=min(missing, len(rest)), replace=False).tolist() if rest and missing else []
        chosen += extra; taken |= set(extra)
    return {"heldout": held, "trained": sorted(int(e) for e in chosen)}


# -- pilot evaluation -----------------------------------------------------------------------------------------------------

def agreement(pairs: Sequence[tuple[list[tuple[int, int]], list[tuple[int, int]]]], *, skip: set[int] = frozenset()
              ) -> dict[str, float | None]:
    """Edge precision / recall of teacher frames against WordNet frames (micro over all edges and macro over entries),
    relations in `skip` left out; also the filler (atom) overlap whatever the relation."""
    hits = predicted = gold = 0
    precisions, recalls, atom_hits, atom_pred, atom_gold = [], [], 0, 0, 0
    for teacher, wordnet_frame in pairs:
        t = {e for e in teacher if e[0] not in skip}
        w = {e for e in wordnet_frame if e[0] not in skip}
        h = len(t & w)
        hits += h; predicted += len(t); gold += len(w)
        if t:
            precisions.append(h / len(t))
        if w:
            recalls.append(h / len(w))
        ta, wa = {a for _, a in t}, {a for _, a in w}
        atom_hits += len(ta & wa); atom_pred += len(ta); atom_gold += len(wa)
    return {"precision": hits / predicted if predicted else None, "recall": hits / gold if gold else None,
            "macro_precision": float(np.mean(precisions)) if precisions else None,
            "macro_recall": float(np.mean(recalls)) if recalls else None,
            "teacher_edges_per_entry": predicted / max(1, len(pairs)), "wordnet_edges_per_entry": gold / max(1, len(pairs)),
            "filler_precision": atom_hits / atom_pred if atom_pred else None,
            "filler_recall": atom_hits / atom_gold if atom_gold else None, "entries": len(pairs)}


def per_relation(pairs: Sequence[tuple[list[tuple[int, int]], list[tuple[int, int]]]], relation_names: Sequence[str]
                 ) -> dict[str, dict[str, Any]]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for teacher, wordnet_frame in pairs:
        t, w = set(teacher), set(wordnet_frame)
        for r, _ in t:
            counts[r]["teacher"] += 1
        for r, _ in w:
            counts[r]["wordnet"] += 1
        for r, _ in t & w:
            counts[r]["hits"] += 1
    return {relation_names[r]: {"teacher": c["teacher"], "wordnet": c["wordnet"], "hits": c["hits"],
                                "precision": c["hits"] / c["teacher"] if c["teacher"] else None,
                                "recall": c["hits"] / c["wordnet"] if c["wordnet"] else None}
            for r, c in sorted(counts.items())}


def ledger_stats(store: Path) -> dict[str, Any]:
    path = Path(store) / "ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    ok = [r for r in rows if not r.get("error")]
    usage = Counter()
    for r in rows:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] += v
    runs_path = Path(store) / "runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()] if runs_path.exists() else []
    return {"attempts": len(rows), "failed_attempts": len(rows) - len(ok), "usd": sum(float(r.get("usd") or 0) for r in rows),
            "items": sum(int(r.get("items") or 0) for r in ok), "call_seconds": [float(r["seconds"]) for r in rows],
            "usd_per_call": [float(r.get("usd") or 0) for r in ok], "usage": dict(usage),
            "wall_seconds": sum(float(r["seconds"]) for r in runs) if runs else None,
            "workers": sorted({int(r["workers"]) for r in runs}), "batch": sorted({int(r["batch"]) for r in runs}),
            "effort": sorted({str(r["effort"]) for r in runs}),
            "models": sorted({m for r in rows for m in (r.get("models") or [])})}


def readable(frame: Sequence[tuple[int, int]], ontology: dict[str, Any], marks: set[tuple[int, int]] | None = None) -> list[str]:
    names, relations = ontology["atomic_names"], ontology["relation_names"]
    return [f"{relations[r]}: {names[a].split(':', 1)[1]}" + (" ✓" if marks is not None and (r, a) in marks else "")
            for r, a in frame]


def evaluate_pilot(sample: Sequence[dict[str, Any]], *, store: Path = STORE, output: Path = PILOT_DIR, examples: int = 10,
                   seed: int = 7, root: Path = C3_ROOT) -> dict[str, Any]:
    """Pilot metrics (cost, time, parse rate, edges per entry, agreement with WordNet) and qualitative examples."""
    ontology = load_c3(root)
    mapper = FillerMapper(ontology["atomic_names"], ontology["relation_names"])
    answers = load_answers(store)
    synsets = entry_synsets(ontology, [s["entry"] for s in sample])
    frames, parse = synset_frames(answers, mapper, synsets)
    aliases = entry_aliases(ontology)
    skip = {mapper.relation_id["lexname"], mapper.relation_id["pos"]}
    rows, pairs = [], defaultdict(list)
    for s in sample:
        teacher = union_frame(ontology, s["entry"], frames)
        if teacher is None:
            continue
        wn_frame = entry_frame(ontology, s["entry"])
        rows.append({**s, "name": aliases[s["entry"]][0], "teacher": teacher, "wordnet": wn_frame})
        for key in ("all", s["group"], f"{s['group']}:{s['band']}"):
            pairs[key].append((teacher, wn_frame))
    ledger = ledger_stats(store)
    entries_done = len(rows)
    summary = {
        "smoke": False, "pilot": True, "prompt_version": PROMPT_VERSION, "sample": len(sample), "entries_scored": entries_done,
        "synsets": len(synsets), "parse": parse,
        "cost": {"usd_total": ledger["usd"], "usd_per_entry": ledger["usd"] / max(1, entries_done),
                 "usd_per_synset": ledger["usd"] / max(1, parse.get("answered", 0)), "attempts": ledger["attempts"],
                 "failed_attempts": ledger["failed_attempts"], "usage": ledger["usage"], "models": ledger["models"],
                 "effort": ledger["effort"], "batch": ledger["batch"],
                 "median_usd_per_call": float(np.median(ledger["usd_per_call"])) if ledger["usd_per_call"] else None},
        "time": {"wall_seconds": ledger["wall_seconds"], "workers": ledger["workers"],
                 "wall_seconds_per_entry": (ledger["wall_seconds"] or 0) / max(1, entries_done),
                 "median_call_seconds": float(np.median(ledger["call_seconds"])) if ledger["call_seconds"] else None,
                 "call_seconds_total": float(sum(ledger["call_seconds"])),
                 "call_seconds_per_synset": float(sum(ledger["call_seconds"])) / max(1, parse.get("answered", 0))},
        "agreement": {key: {"all_relations": agreement(p), "without_lexname_pos": agreement(p, skip=skip)}
                      for key, p in sorted(pairs.items())},
        "per_relation": per_relation(pairs["all"], ontology["relation_names"]),
    }
    rng = np.random.default_rng(seed)
    chosen = []
    for group in ("heldout", "trained"):
        members = [r for r in rows if r["group"] == group and len(r["wordnet"]) >= 3]
        chosen += [members[i] for i in sorted(rng.choice(len(members), size=min(examples // 2, len(members)), replace=False))]
    summary["examples"] = [{"entry": r["entry"], "name": r["name"], "group": r["group"], "band": r["band"],
                            "senses": r["senses"],
                            "wordnet": readable(r["wordnet"], ontology, set(r["teacher"])),
                            "teacher": readable(r["teacher"], ontology, set(r["wordnet"]))} for r in chosen]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (output / "frames.jsonl").write_text("".join(json.dumps({"entry": r["entry"], "name": r["name"], "group": r["group"],
                                                             "band": r["band"], "teacher": readable(r["teacher"], ontology),
                                                             "wordnet": readable(r["wordnet"], ontology)}) + "\n" for r in rows))
    pilot_answers = [answers[s] for s in synsets if s in answers]
    (output / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in pilot_answers))
    (output / "report.md").write_text(pilot_report(summary))
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def pilot_report(summary: dict[str, Any]) -> str:
    c, t, p = summary["cost"], summary["time"], summary["parse"]
    lines = ["# C3 teacher ontology — pilot (decision 65, phase 1b)", "",
             f"{summary['entries_scored']} entries ({summary['synsets']} synsets) read by the teacher "
             f"(`{', '.join(c['models']) or 'n/a'}`, prompt `{summary['prompt_version']}`).", "",
             f"- spend: **${c['usd_total']:.2f}** = ${c['usd_per_entry']:.4f} per entry, ${c['usd_per_synset']:.4f} per synset "
             f"({c['attempts']} calls, {c['failed_attempts']} failed; median ${_fmt(c['median_usd_per_call'])} per call)",
             f"- time: wall {_fmt(t['wall_seconds'], 0)} s = {_fmt(t['wall_seconds_per_entry'], 2)} s per entry (parallel calls); "
             f"median call {_fmt(t['median_call_seconds'], 1)} s; {_fmt(t['call_seconds_per_synset'], 2)} call-seconds per synset",
             f"- parse: {p.get('answered', 0)} synsets answered, {p.get('unanswered', 0)} not; {p.get('proposed', 0)} proposed "
             f"edges, **{p['parse_rate']:.1%} mapped** to atomics ({', '.join(f'{k} {v}' for k, v in sorted(p['mapped_by'].items()))}); "
             f"unmapped: {', '.join(f'{k} {v}' for k, v in sorted(p['unmapped_by'].items())) or 'none'}; "
             f"pos-only fallback frames: {p.get('pos_fallback', 0)}", "",
             "## Agreement with the WordNet entry frames", "",
             "| entries | n | teacher edges / entry | WordNet edges / entry | precision | recall | macro P | macro R | "
             "w/o lexname+pos: edges T / W | P | R | filler P | filler R |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for key, value in summary["agreement"].items():
        a, b = value["all_relations"], value["without_lexname_pos"]
        lines.append(f"| {key} | {a['entries']} | {a['teacher_edges_per_entry']:.2f} | {a['wordnet_edges_per_entry']:.2f} | "
                     f"{_fmt(a['precision'])} | {_fmt(a['recall'])} | {_fmt(a['macro_precision'])} | {_fmt(a['macro_recall'])} | "
                     f"{b['teacher_edges_per_entry']:.2f} / {b['wordnet_edges_per_entry']:.2f} | {_fmt(b['precision'])} | "
                     f"{_fmt(b['recall'])} | {_fmt(b['filler_precision'])} | {_fmt(b['filler_recall'])} |")
    lines += ["", "## Per relation (all sampled entries)", "", "| relation | teacher | WordNet | hits | precision | recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for relation, r in summary["per_relation"].items():
        lines.append(f"| {relation} | {r['teacher']} | {r['wordnet']} | {r['hits']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} |")
    lines += ["", "## Examples (✓ = edge in both frames)", ""]
    for ex in summary["examples"]:
        lines += [f"**{ex['name']}** ({ex['group']}, band {ex['band']}, {ex['senses']} senses)", "",
                  f"- WordNet: {'; '.join(ex['wordnet'])}", f"- teacher: {'; '.join(ex['teacher'])}", ""]
    return "\n".join(lines) + "\n"


# -- projection ---------------------------------------------------------------------------------------------------------

def projection(summary: dict[str, Any], ontology: dict[str, Any], counts: np.ndarray, *, workers: int = 4) -> dict[str, Any]:
    """USD and hours for the full ontology and the hybrid variant, from the pilot's per-synset spend and call-seconds."""
    per_synset_usd = summary["cost"]["usd_per_synset"]
    per_synset_seconds = summary["time"]["call_seconds_per_synset"]
    hybrid = hybrid_entries(ontology, counts)
    scopes = {"full": entry_synsets(ontology, range(int(ontology["entry_count"]))),
              "hybrid": entry_synsets(ontology, hybrid["heldout"] + hybrid["trained"]),
              "hybrid_heldout_only": entry_synsets(ontology, hybrid["heldout"])}
    out = {"per_synset_usd": per_synset_usd, "per_synset_call_seconds": per_synset_seconds, "workers": workers,
           "hybrid_entries": {k: len(v) for k, v in hybrid.items()}}
    for name, synsets in scopes.items():
        out[name] = {"synsets": len(synsets), "usd": len(synsets) * per_synset_usd,
                     "hours": len(synsets) * per_synset_seconds / workers / 3600}
    return out


# -- reference strata (hybrid) ----------------------------------------------------------------------------------------

def write_reference(path: Path, groups: dict[str, Iterable[int]], *, root: Path = C3_ROOT, windows: int = 1024,
                    length: int = 1024, min_subtokens: int = 2) -> Path:
    """`eval.reference_strata` masks (the trainer's after-span rule) for each teacher entry set on the C3 windows."""
    from ..data.corpus import TokenCorpus, eval_windows
    from ..training.lm import save_reference_strata
    from .e13_cycle import reference_masks
    corpus = TokenCorpus.open(Path(root) / "eval")
    starts = eval_windows(corpus, count=windows, length=length)
    masks = reference_masks(corpus, starts, length, min_subtokens, {f"after_teacher_{k}": set(map(int, v)) for k, v in groups.items()})
    save_reference_strata(Path(path), starts, masks, length)
    return Path(path)


# -- CLI ------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="write the pilot sample (200 held-out + 200 trained entries)")
    sample.add_argument("--output", type=Path, default=PILOT_DIR); sample.add_argument("--per-group", type=int, default=200)
    run = sub.add_parser("teach", help="teacher answers for a scope's synsets (resumable, cached, capped)")
    run.add_argument("--scope", choices=SCOPES, required=True); run.add_argument("--sample", type=Path, default=PILOT_DIR / "sample.json")
    run.add_argument("--store", type=Path, default=STORE); run.add_argument("--max-usd", type=float, required=True)
    run.add_argument("--batch", type=int, default=25); run.add_argument("--workers", type=int, default=4)
    run.add_argument("--effort", default="medium"); run.add_argument("--model", default=MODEL)
    run.add_argument("--limit", type=int, default=None, help="only the first N synsets of the scope (calibration)")
    ev = sub.add_parser("evaluate", help="pilot metrics and examples")
    ev.add_argument("--sample", type=Path, default=PILOT_DIR / "sample.json"); ev.add_argument("--store", type=Path, default=STORE)
    ev.add_argument("--output", type=Path, default=PILOT_DIR)
    onto = sub.add_parser("ontology", help="write the teacher ontology (.pt) for C5teach")
    onto.add_argument("--scope", choices=("full", "hybrid"), required=True); onto.add_argument("--store", type=Path, default=STORE)
    onto.add_argument("--output", type=Path, default=None)
    ref = sub.add_parser("reference", help="reference strata of the hybrid teacher entry sets")
    ref.add_argument("--store", type=Path, default=STORE); ref.add_argument("--output", type=Path, default=None)
    proj = sub.add_parser("project", help="USD / hours for the full and hybrid runs from the pilot")
    proj.add_argument("--pilot", type=Path, default=PILOT_DIR); proj.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "sample":
        rows = pilot_sample(load_c3(), eval_counts(), per_group=args.per_group)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "sample.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(json.dumps(Counter(f"{r['group']}:{r['band']}" for r in rows)))
    elif args.command == "teach":
        ontology = load_c3()
        if args.scope == "pilot":
            entries = [r["entry"] for r in json.loads(args.sample.read_text())]
        elif args.scope == "hybrid":
            groups = hybrid_entries(ontology, eval_counts())
            entries = groups["heldout"] + groups["trained"]
        else:
            entries = range(int(ontology["entry_count"]))
        synsets = entry_synsets(ontology, entries)[:args.limit]
        print(json.dumps(teach(synsets, store=args.store, max_usd=args.max_usd, batch=args.batch, workers=args.workers,
                               effort=args.effort, model=args.model), indent=1))
    elif args.command == "evaluate":
        summary = evaluate_pilot(json.loads(args.sample.read_text()), store=args.store, output=args.output)
        print(json.dumps({k: summary[k] for k in ("cost", "time", "parse")}, indent=1, default=float))
    elif args.command == "ontology":
        import torch
        ontology = load_c3()
        mapper = FillerMapper(ontology["atomic_names"], ontology["relation_names"])
        if args.scope == "hybrid":
            groups = hybrid_entries(ontology, eval_counts())
            entries = groups["heldout"] + groups["trained"]
        else:
            entries = None
        synsets = entry_synsets(ontology, range(int(ontology["entry_count"])) if entries is None else entries)
        frames, parse = synset_frames(load_answers(args.store), mapper, synsets)
        provenance = {"scope": args.scope, "prompt_version": PROMPT_VERSION, "parse": parse, "store": str(args.store),
                      "answers_sha256": hashlib.sha256((args.store / "answers.jsonl").read_bytes()).hexdigest()}
        out = teacher_ontology(ontology, frames, entries=entries, provenance=provenance)
        path = args.output or args.store / f"ontology-{args.scope}.pt"
        torch.save(out, path)
        print(json.dumps({"path": str(path), **out["teacher"]}, indent=1, default=str))
    elif args.command == "reference":
        groups = hybrid_entries(load_c3(), eval_counts())
        print(write_reference(args.output or args.store / "reference-hybrid-1024x1024.npz", groups))
    else:
        summary = json.loads((args.pilot / "summary.json").read_text())
        result = projection(summary, load_c3(), eval_counts(), workers=args.workers)
        (args.pilot / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
