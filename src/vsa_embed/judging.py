"""LLM-judge harness (B13): Claude Code called headless as the grader.

Each item is graded in `calls` independent calls, cycling through prompt paraphrases. Verdicts are
cached on disk keyed by (model, prompt, schema, call index), so re-running a study replays them
without new API calls and the later clinician study can rate the identical items. Blinding and
option order are seeded. Agreement across calls is Fleiss' κ; agreement with an author-labelled
calibration set is reported separately.

The CLI is run lean — no tools, no settings, no session persistence — from an empty directory:

    claude -p <prompt> --output-format json --model <model> --tools "" --setting-sources ""
           --no-session-persistence --json-schema <schema>

Alternatively (`exchange_dir`), verdicts come from an interactive Claude Code session through files:
each ungraded (item, call) is written to `requests/<key>.json` (prompt, schema, response path — no
item id, gold or system label), the session writes the verdict object to `responses/<key>.json`,
and re-running the study validates and caches it. Answered requests are removed, so `requests/`
always lists the outstanding work.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

Runner = Callable[[str, dict[str, Any], str], dict[str, Any]]


def claude_cli_runner(prompt: str, schema: dict[str, Any], model: str, *, timeout: int = 300) -> dict[str, Any]:
    """Call Claude Code headless; returns the parsed JSON envelope."""
    with tempfile.TemporaryDirectory() as empty:
        completed = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--model", model, "--tools", "",
             "--setting-sources", "", "--no-session-persistence", "--json-schema", json.dumps(schema)],
            cwd=empty, capture_output=True, text=True, timeout=timeout,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"claude exited {completed.returncode}: {completed.stderr[:500]}")
    return json.loads(completed.stdout)


def schema_problem(value: Any, schema: dict[str, Any]) -> str | None:
    """Why `value` violates the (flat) JSON schemas used here, or None if it conforms."""
    kind = schema.get("type")
    checks = {"object": lambda v: isinstance(v, dict), "string": lambda v: isinstance(v, str),
              "boolean": lambda v: isinstance(v, bool),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)}
    if kind in checks and not checks[kind](value):
        return f"expected {kind}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{value!r} not in {schema['enum']}"
    if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
        return f"{value!r} outside [{schema.get('minimum')}, {schema.get('maximum')}]"
    if kind == "object":
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"missing {missing}"
        for name, sub in schema.get("properties", {}).items():
            problem = schema_problem(value[name], sub) if name in value else None
            if problem:
                return f"{name}: {problem}"
    return None


@dataclass
class JudgeClient:
    cache_dir: Path
    model: str = "claude-opus-5-5"
    calls: int = 3
    retries: int = 2
    runner: Runner | None = None
    exchange_dir: Path | None = None
    spent_usd: float = field(default=0.0, init=False)

    def _key(self, prompt: str, schema: dict[str, Any], call: int) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "schema": schema, "call": call}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def grade(self, item_id: str, prompts: Sequence[str], schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Grade one item `calls` times; returns one verdict record per call."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        runner = self.runner or (lambda p, s, m: claude_cli_runner(p, s, m))
        records = []
        for call in range(self.calls):
            prompt = prompts[call % len(prompts)]
            key = self._key(prompt, schema, call)
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                records.append(json.loads(path.read_text())); continue
            record: dict[str, Any] = {"item": item_id, "call": call, "model": self.model,
                                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            for attempt in range(0 if self.exchange_dir is not None else self.retries + 1):
                try:
                    envelope = runner(prompt, schema, self.model)
                    verdict = envelope.get("structured_output")
                    if verdict is None or envelope.get("is_error"):
                        raise ValueError("no structured output")
                    record.update(verdict=verdict, cost_usd=envelope.get("total_cost_usd"),
                                  models_used=sorted((envelope.get("modelUsage") or {}).keys()), attempts=attempt + 1)
                    self.spent_usd += float(envelope.get("total_cost_usd") or 0)
                    break
                except (ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
                    record.update(error=str(error)[:300], attempts=attempt + 1)
            if self.exchange_dir is not None:
                record.update(self._exchange(key, prompt, schema, call))
            if "verdict" in record:
                record.pop("error", None)
                path.write_text(json.dumps(record, indent=2) + "\n")   # only successful verdicts are cached
            records.append(record)
        return records

    def _exchange(self, key: str, prompt: str, schema: dict[str, Any], call: int) -> dict[str, Any]:
        """Read the session's verdict for `key`, or post the request and report it pending."""
        request, response = self.exchange_dir / "requests" / f"{key}.json", self.exchange_dir / "responses" / f"{key}.json"
        if response.exists():
            try:
                verdict = json.loads(response.read_text())
                problem = schema_problem(verdict, schema)
            except json.JSONDecodeError as error:
                problem = f"not JSON: {error}"
            if problem is None:
                request.unlink(missing_ok=True)
                return {"verdict": verdict, "judge": "claude-code-session", "cost_usd": None}
            error = f"invalid response: {problem}"
        else:
            error = "pending"
        request.parent.mkdir(parents=True, exist_ok=True); response.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps({"key": key, "model": self.model, "call": call, "prompt": prompt, "schema": schema,
                                       "response_path": str(response.resolve()), "last_error": error}, indent=2) + "\n")
        return {"error": error}


def blind_options(options: dict[str, Any], seed: int) -> tuple[list[Any], dict[str, str]]:
    """Shuffle labelled options (e.g. system → neighbour list) and hide the labels.

    Returns the shuffled option contents and a map from blinded letters (A, B, …) to labels."""
    labels = sorted(options)
    random.Random(seed).shuffle(labels)
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    return [options[label] for label in labels], dict(zip(letters, labels))


def fleiss_kappa(ratings: Sequence[Sequence[Any]]) -> float:
    """Fleiss' κ for items each rated by the same number of raters (here: repeated calls)."""
    ratings = [list(r) for r in ratings if len(r) > 1]
    if not ratings:
        return float("nan")
    n = len(ratings[0])
    if any(len(r) != n for r in ratings):
        raise ValueError("every item needs the same number of ratings")
    categories = sorted({c for r in ratings for c in r}, key=str)
    counts = [[r.count(c) for c in categories] for r in ratings]
    p_items = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    p_bar = sum(p_items) / len(p_items)
    totals = [sum(row[j] for row in counts) for j in range(len(categories))]
    p_cat = [t / (len(ratings) * n) for t in totals]
    p_e = sum(p * p for p in p_cat)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def majority(values: Sequence[Any]) -> Any:
    """Most common value; ties broken by first occurrence."""
    counts: dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def calibration_agreement(judged: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    """Accuracy of majority judge verdicts against author labels on the calibration items."""
    shared = [k for k in gold if k in judged]
    if not shared:
        return {"items": 0, "accuracy": float("nan")}
    return {"items": len(shared), "accuracy": sum(judged[k] == gold[k] for k in shared) / len(shared)}
