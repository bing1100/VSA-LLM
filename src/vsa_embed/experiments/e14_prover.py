"""E14 LG-P — the language model as the prover's search policy (proposal §8; plan §7; pre-registration §6).

**Sound tools.** A one-entity first-order problem (`logic.world.fol`) is proved by depth-first search over three sound
steps; the policy only orders the candidate steps at each choice point:
- `back[rule]` — to show L, show the rule's premise (the rule's conclusion is L);
- `cases[A or B]` — show the goal twice, once assuming A and once assuming B (both must succeed);
- `contra` — to show "not C", assume C and derive a contradiction: show some H for which "not H" is known.
A goal closes automatically when it is a known fact or an assumption. Every step tried counts against the budget; a dead
end backtracks to the next candidate. Because every step is sound, a "proved" answer is entailed whatever the policy.

**Policies** (`--policies`):
- `score` — the host ranks the candidate steps (length-normalised log-probability of each step's description);
- `random`, `bfs` (iterative deepening: every step order, shortest proofs first), `sos` (set of support: steps whose
  premise is a known fact first, then steps whose premise is the head of another clause, dead ends last), `ceiling`
  (`sos` with a budget of 1,000);
- generative (Qwen hosts): `react` (the model writes the number of the next step; an unparseable answer wastes the step),
  `unsound` (as `react`, plus an option "accept it as true without proof", which closes the goal), `cot` (the model writes
  a proof in prose and an answer; no tools).

**Premise selection (LG-P3, CPU):** the rule memory as VSA vectors `HEAD⊛head + BODY⊛body`; `rules_for` by unbinding HEAD
and ranking against the goal's class (top-k), against exact symbol match; recall@k at 50 and 500 clauses.

    python -m vsa_embed.experiments.e14_prover run --host H-q17 --output OUT [--policies score,random,bfs,sos,ceiling,react,unsound,cot]
        [--budgets 10,30] [--limit N]
    python -m vsa_embed.experiments.e14_prover retrieval --output OUT
    python -m vsa_embed.experiments.e14_prover report --runs OUT ... --output DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..logic import world as W
from ..statistics import holm_adjust
from .e5_common import finish_output, start_output, write_json
from .e5_zeroshot import paired_difference

Literal = tuple[str, bool]          # (class, negated)
BOTTOM: Literal = ("⊥", False)


@dataclass
class Step:
    kind: str                       # back | cases | contra | target | accept
    clause: dict[str, Any] | None
    subgoals: list[tuple[Literal, frozenset]]
    text: str


@dataclass
class Search:
    budget: int
    steps: int = 0
    calls: int = 0
    accepted: bool = False          # an unsound "accept" closed some goal
    trace: list[str] = field(default_factory=list)


def lit_text(lit: Literal, entity: str) -> str:
    return "a contradiction" if lit == BOTTOM else f"{entity} is {'not ' if lit[1] else ''}a {lit[0]}"


class Problem:
    def __init__(self, item: dict[str, Any]) -> None:
        self.item = item
        self.entity = item["entity"]
        self.clauses = item["clauses"]
        self.facts = {(c["cls"], c["neg"]) for c in self.clauses if c["kind"] == "fact"}
        self.rules = [c for c in self.clauses if c["kind"] == "rule"]
        self.ors = [c for c in self.clauses if c["kind"] == "or"]
        self.heads = {c["head"] for c in self.rules}
        self.goal: Literal = (item["goal"]["cls"], item["goal"]["neg"])

    def text(self) -> str:
        return "\n".join(f"{i + 1}. {W.clause_text(c, self.entity)}" for i, c in enumerate(self.clauses))

    def known(self, lit: Literal, assumptions: frozenset) -> bool:
        return lit in self.facts or lit in assumptions

    def candidates(self, goal: Literal, assumptions: frozenset, *, allow_accept: bool = False) -> list[Step]:
        e = self.entity
        out: list[Step] = []
        if goal == BOTTOM:
            for h, neg in sorted(self.facts | set(assumptions)):
                if neg:
                    out.append(Step("target", None, [((h, False), assumptions)], f"derive that {e} is a {h}, which contradicts "
                                    f"'{e} is not a {h}'"))
        else:
            for c in self.rules:
                if c["head"] == goal[0] and c["neg"] == goal[1]:
                    out.append(Step("back", c, [((c["body"], False), assumptions)],
                                    f"use '{W.clause_text(c, e)}': show that {e} is a {c['body']}"))
            for c in self.ors:
                a, b = c["cls"]
                if (a, False) not in assumptions and (b, False) not in assumptions:
                    out.append(Step("cases", c, [(goal, assumptions | {(a, False)}), (goal, assumptions | {(b, False)})],
                                    f"split on '{W.clause_text(c, e)}': show it once assuming {e} is a {a}, once assuming {e} is a {b}"))
            if goal[1] and (goal[0], False) not in assumptions:
                out.append(Step("contra", None, [(BOTTOM, assumptions | {(goal[0], False)})],
                                f"assume {e} is a {goal[0]} and derive a contradiction"))
        if allow_accept and goal != BOTTOM:
            out.append(Step("accept", None, [], f"accept that {lit_text(goal, e)} without proof"))
        return out


Policy = Callable[[Problem, Literal, frozenset, list[Step], Search], list[int]]


def dfs(problem: Problem, goal: Literal, assumptions: frozenset, policy: Policy, search: Search, depth: int, seen: frozenset,
        *, allow_accept: bool = False, max_depth: int = 14) -> bool:
    if goal != BOTTOM and problem.known(goal, assumptions):
        return True
    if depth >= max_depth or (goal, assumptions) in seen:
        return False
    steps = problem.candidates(goal, assumptions, allow_accept=allow_accept)
    if not steps:
        return False
    order = policy(problem, goal, assumptions, steps, search)
    for i in order:
        if search.steps >= search.budget:
            return False
        search.steps += 1
        step = steps[i]
        search.trace.append(f"{lit_text(goal, problem.entity)}: {step.text}")
        if step.kind == "accept":
            search.accepted = True
            return True
        if all(dfs(problem, g, a, policy, search, depth + 1, seen | {(goal, assumptions)}, allow_accept=allow_accept,
                   max_depth=max_depth) for g, a in step.subgoals):
            return True
    return False


# -- policies ---------------------------------------------------------------------------------------------------------

def random_policy(seed: int) -> Policy:
    def f(problem, goal, assumptions, steps, search):
        order = list(range(len(steps)))
        random.Random(f"{seed}:{problem.item['id']}:{search.steps}").shuffle(order)
        return order
    return f


def sos_policy(problem, goal, assumptions, steps, search):
    def rank(step: Step) -> int:
        if step.kind == "accept":
            return 9
        premises = [g for g, _ in step.subgoals]
        if any(g == BOTTOM for g in premises):
            return 3
        if all(problem.known(g, a) for g, a in step.subgoals):
            return 0
        if step.kind == "cases":
            return 1
        return 2 if all(g[0] in problem.heads or problem.known(g, a) for g, a in step.subgoals) else 5
    return sorted(range(len(steps)), key=lambda i: (rank(steps[i]), i))


def sos_rank(problem: Problem, step: Step) -> int:
    """`sos_policy`'s tier of a step (0 best)."""
    if step.kind == "accept":
        return 9
    premises = [g for g, _ in step.subgoals]
    if any(g == BOTTOM for g in premises):
        return 3
    if all(problem.known(g, a) for g, a in step.subgoals):
        return 0
    if step.kind == "cases":
        return 1
    return 2 if all(g[0] in problem.heads or problem.known(g, a) for g, a in step.subgoals) else 5


def sos_score_policy(host: Any) -> Policy:
    """Pre-registration §16: set of support's tiers, the host's scores breaking ties within a tier (the host is asked only
    when a tier holds more than one step)."""
    scorer = score_policy(host)

    def f(problem, goal, assumptions, steps, search):
        ranks = [sos_rank(problem, s) for s in steps]
        if len(set(ranks)) == len(ranks):
            return sorted(range(len(steps)), key=lambda i: ranks[i])
        model_order = scorer(problem, goal, assumptions, steps, search)
        position = {i: k for k, i in enumerate(model_order)}
        return sorted(range(len(steps)), key=lambda i: (ranks[i], position[i]))
    return f


def score_policy(host: Any) -> Policy:
    def f(problem, goal, assumptions, steps, search):
        if len(steps) == 1:
            return [0]
        assume = "" if not assumptions else " (assuming " + "; ".join(lit_text(a, problem.entity) for a in sorted(assumptions)) + ")"
        prompt = (f"Facts and rules:\n{problem.text()}\n\nWe want to show that {lit_text(problem.goal, problem.entity)}.\n"
                  f"Now we need to show that {lit_text(goal, problem.entity)}{assume}.\nThe best next step is to")
        _, values = host.choose(prompt, [" " + s.text for s in steps])
        search.calls += 1
        return list(np.argsort(-values, kind="stable"))
    return f


def generative_policy(host: Any) -> Policy:
    def f(problem, goal, assumptions, steps, search):
        assume = "" if not assumptions else " (assuming " + "; ".join(lit_text(a, problem.entity) for a in sorted(assumptions)) + ")"
        options = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(steps))
        prompt = (f"Facts and rules:\n{problem.text()}\n\nWe want to show that {lit_text(problem.goal, problem.entity)}.\n"
                  f"Now we need to show that {lit_text(goal, problem.entity)}{assume}.\nPossible next steps:\n{options}\n"
                  f"Reply with the number of the best next step.\nStep number:")
        text = host.generate([prompt], max_new_tokens=4, stop=("\n",))[0]
        search.calls += 1
        m = re.search(r"\d+", text)
        first = int(m.group()) - 1 if m else -1
        if not 0 <= first < len(steps):
            search.steps += 1                                   # an unparseable or out-of-range answer wastes a step
            search.trace.append(f"invalid step answer {text!r}")
            return []
        return [first] + [i for i in range(len(steps)) if i != first]
    return f


def bfs_prove(problem: Problem, budget: int) -> Search:
    """Iterative deepening: every step order up to depth d, d = 1, 2, …; expansions counted."""
    for limit in range(1, 15):
        search = Search(budget=budget)
        if dfs(problem, problem.goal, frozenset(), lambda p, g, a, s, se: list(range(len(s))), search, 0, frozenset(),
               max_depth=limit):
            return search
        if search.steps >= budget:
            return search
    return search


def cot_answer(host: Any, problem: Problem) -> tuple[bool | None, str]:
    prompt = (f"Facts and rules:\n{problem.text()}\n\nQuestion: Is it true that {lit_text(problem.goal, problem.entity)}? "
              f"Reason step by step, then end with 'Answer: True' or 'Answer: False'.\nReasoning:")
    text = host.generate([prompt], max_new_tokens=160, stop=("\n\n\n",))[0]
    m = re.findall(r"Answer:\s*(True|False)", text)
    return (m[-1] == "True") if m else None, text


def run_policy(problem: Problem, name: str, budget: int, *, host: Any = None, seed: int = 0) -> dict[str, Any]:
    t0 = time.monotonic()
    if name == "cot":
        answer, text = cot_answer(host, problem)
        return {"proved": answer is True, "answered": answer is not None, "steps": 0, "calls": 1, "accepted": False,
                "seconds": time.monotonic() - t0, "trace": [text[:400]]}
    if name == "bfs":
        search = bfs_prove(problem, budget)
        proved = bool(search.trace) and _closed(problem, search)
        return {"proved": proved, "steps": search.steps, "calls": 0, "accepted": False, "seconds": time.monotonic() - t0,
                "trace": search.trace[-12:]}
    policy = {"random": random_policy(seed), "sos": sos_policy, "ceiling": sos_policy}.get(name)
    if name == "score":
        policy = score_policy(host)
    elif name == "sos-score":
        policy = sos_score_policy(host)
    elif name in {"react", "unsound"}:
        policy = generative_policy(host)
    search = Search(budget=1000 if name == "ceiling" else budget)
    proved = dfs(problem, problem.goal, frozenset(), policy, search, 0, frozenset(), allow_accept=name == "unsound")
    return {"proved": bool(proved), "steps": search.steps, "calls": search.calls, "accepted": search.accepted,
            "seconds": time.monotonic() - t0, "trace": search.trace[-12:]}


def _closed(problem: Problem, search: Search) -> bool:
    """bfs: re-run the last (successful) depth limit to report whether the proof closed within budget."""
    for limit in range(1, 15):
        s = Search(budget=search.budget)
        if dfs(problem, problem.goal, frozenset(), lambda p, g, a, st, se: list(range(len(st))), s, 0, frozenset(), max_depth=limit):
            return s.steps <= search.budget
        if s.steps >= search.budget:
            return False
    return False


# -- premise selection (LG-P3) ------------------------------------------------------------------------------------------

def retrieval_study(seed: int, *, sizes: Sequence[int] = (50, 500), width: int = 1024, k: int = 5, queries: int = 300) -> dict[str, Any]:
    from ..logic.store import ALG, Codebook
    from torch.nn import functional as F
    import torch
    out = {}
    namer = W.Namer(seed + 99)
    for size in sizes:
        cb = Codebook(width, seed)
        classes = [namer.word() for _ in range(max(40, size // 2))]
        rng = random.Random(f"ret:{seed}:{size}")
        rules = [(rng.choice(classes), rng.choice(classes)) for _ in range(size)]
        vecs = torch.stack([cb.bind("HEAD", h) + cb.bind("BODY", b) for b, h in rules])
        hits = []
        for _ in range(queries):
            goal = rng.choice([h for _, h in rules])
            truth = {i for i, (_, h) in enumerate(rules) if h == goal}
            probe = cb.filler(goal)
            scores = F.normalize(ALG.unbind(vecs, cb.role("HEAD").expand_as(vecs)), dim=-1) @ F.normalize(probe, dim=0)
            top = set(torch.topk(scores, min(k + len(truth) - 1, len(rules))).indices.tolist())
            hits.append(len(top & truth) / len(truth))
        out[size] = {"recall_at_k": float(np.mean(hits)), "k": k, "width": width, "classes": len(classes)}
    return out


# -- run, report --------------------------------------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    policies = [p.strip() for p in args.policies.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    config = {"experiment": "e14-lg-p", "host": args.host, "policies": policies, "budgets": budgets, "seed": args.seed,
              "limit": args.limit, "smoke": bool(args.smoke)}
    output = Path(args.output)
    git_at_start = start_output(output, config)
    host = None
    if any(p in {"score", "react", "unsound", "cot", "sos-score"} for p in policies):
        from ..logic.hosts import HostLM
        host = HostLM(args.host, device=args.device, batch_size=args.batch_size)
    hard = {"depths": tuple(int(x) for x in args.depths.split(",")), "decoy": args.decoy, "decoy_depth": args.decoy_depth,
            "decoy_fanout": args.decoy_fanout} if args.depths else {}
    config["problems"] = hard or "tier-p"
    items = W.fol(args.seed, per_depth=args.per_depth, namer=W.Namer(args.seed), **hard)["items"]
    if args.provable_only:
        items = [i for i in items if i["label"]]
    if args.limit:
        items = items[: args.limit]
    rows = []
    t0 = time.monotonic()
    derive = {"score", "random", "sos", "react", "sos-score"} if args.derive_budgets else set()
    for item in items:
        problem = Problem(item)
        for name in policies:
            base = {"id": item["id"], "kind": item["kind"], "depth": item["depth"], "distractors": item["distractors"],
                    "label": item["label"], "policy": name}
            if name in derive:
                # Depth-first search with a deterministic policy tries the same steps in the same order whatever the
                # budget, so one run at the largest budget gives every smaller one: found within b ⇔ found in ≤ b steps.
                r = run_policy(problem, name, max(budgets), host=host, seed=args.seed)
                for budget in budgets:
                    hit = bool(r["proved"]) and r["steps"] <= budget
                    rows.append({**base, "budget": budget, "correct": int(hit == item["label"]), "proved": hit,
                                 "steps": r["steps"] if hit else budget, "calls": r["calls"] if budget == max(budgets) else None,
                                 "accepted": r["accepted"], "seconds": r["seconds"] if budget == max(budgets) else None,
                                 "derived": budget != max(budgets), "trace": r["trace"] if budget == max(budgets) else []})
                continue
            for budget in (budgets if name not in {"cot", "ceiling"} else budgets[-1:]):
                r = run_policy(problem, name, budget, host=host, seed=args.seed)
                rows.append({**base, "budget": budget, "correct": int(r["proved"] == item["label"]),
                             **{k: v for k, v in r.items() if k != "trace"}, "trace": r["trace"]})
    summary: dict[str, Any] = {"source": host.describe() if host else {"host": "none"}, "items": len(items), "policies": policies,
                               "problems": config["problems"], "provable_only": bool(args.provable_only),
                               "budgets": budgets, "smoke": bool(args.smoke), "seconds": round(time.monotonic() - t0, 1), "cells": {}}
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(r["policy"], r["budget"])].append(r)
    for (p, b), rs in sorted(by.items()):
        provable = [r for r in rs if r["label"]]
        proved = [r for r in rs if r["proved"]]
        summary["cells"][f"{p}|{b}"] = {
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "found_within_budget": float(np.mean([r["proved"] for r in provable])) if provable else None,
            "soundness": float(np.mean([r["label"] for r in proved])) if proved else None,
            "mean_steps_found": float(np.mean([r["steps"] for r in provable if r["proved"]])) if any(r["proved"] for r in provable) else None,
            "calls": float(np.mean([r["calls"] for r in rs if r["calls"] is not None])) if any(r["calls"] is not None for r in rs) else None,
            "n": len(rs)}
    with (output / "rows.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    write_json(output / "summary.json", summary)
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [
        f"# E14 LG-P — search policies on `{summary['source'].get('host')}` ({len(items)} problems)", "",
        "| policy | budget | accuracy | proofs found (provable) | soundness | steps (found) | model calls |", "|---|---:|---:|---:|---:|---:|---:|"]
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    for key, c in summary["cells"].items():
        p, b = key.split("|")
        lines.append(f"| {p} | {b} | {f(c['accuracy'])} | {f(c['found_within_budget'])} | {f(c['soundness'])} | {f(c['mean_steps_found'])} | {'—' if c['calls'] is None else f"{c['calls']:.1f}"} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    finish_output(output, config, git_at_start=git_at_start, device=getattr(host, "device", "cpu"), source=summary["source"])
    return summary


def run_retrieval(args: argparse.Namespace) -> dict[str, Any]:
    doc = {"seed": args.seed, "results": retrieval_study(args.seed)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "retrieval.json", doc)
    (out / "retrieval.md").write_text("# E14 LG-P3 — premise selection by VSA retrieval (recall@k of the matching rules)\n\n"
                                      + "\n".join(f"- {n} clauses: recall@{v['k']} {v['recall_at_k']:.3f} (width {v['width']})"
                                                  for n, v in doc["results"].items()) + "\n")
    return doc


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    present = [Path(p) for p in args.runs if (Path(p) / "rows.jsonl").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "rows.jsonl").exists()]
    doc: dict[str, Any] = {"hosts": {}, "missing": missing}
    lines = ["# E14 LG-P — pooled report", ""] + ([f"Missing: {', '.join(missing)}", ""] if missing else [])
    for run_dir in present:
        host = json.loads((run_dir / "summary.json").read_text())["source"].get("host")
        rows = [json.loads(l) for l in (run_dir / "rows.jsonl").read_text().splitlines()]
        prim = []
        for b in sorted({r["budget"] for r in rows if r["policy"] == "score"}):
            a = {r["id"]: r["proved"] for r in rows if r["policy"] == "score" and r["budget"] == b and r["label"]}
            c = {r["id"]: r["proved"] for r in rows if r["policy"] == "random" and r["budget"] == b and r["label"]}
            ids = sorted(set(a) & set(c))
            if ids:
                prim.append({"budget": b, **paired_difference(np.array([a[i] for i in ids], float), np.array([c[i] for i in ids], float),
                                                               resamples=args.resamples, seed=args.seed)})
        for row, p in zip(prim, holm_adjust([r["p_value"] for r in prim]) if prim else []):
            row["p_holm"] = p
        secondary = []
        for x, y in (("score", "sos"), ("react", "random"), ("react", "score"), ("sos-score", "sos"), ("sos-score", "score")):
            for b in sorted({r["budget"] for r in rows if r["policy"] == x}):
                a = {r["id"]: r["proved"] for r in rows if r["policy"] == x and r["budget"] == b and r["label"]}
                c = {r["id"]: r["proved"] for r in rows if r["policy"] == y and r["budget"] == b and r["label"]}
                ids = sorted(set(a) & set(c))
                if ids:
                    secondary.append({"a": x, "b": y, "budget": b, **paired_difference(
                        np.array([a[i] for i in ids], float), np.array([c[i] for i in ids], float), resamples=args.resamples, seed=args.seed)})
        summary = json.loads((run_dir / "summary.json").read_text())
        doc["hosts"][host] = {"primary": prim, "secondary": secondary, "cells": summary["cells"],
                              "problems": summary.get("problems", "tier-p")}
        lines += [f"## {host}", "", "Primary — proofs found on provable problems, score − random (Holm over budgets):", ""]
        lines += [f"- budget {r['budget']}: {r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] (Holm p {r['p_holm']:.4f}, n {r['n']})" for r in prim]
        if secondary:
            lines += ["", "Secondary (found on provable problems):", ""]
            lines += [f"- {r['a']} − {r['b']}, budget {r['budget']}: {r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                      f"(p {r['p_value']:.4f}, n {r['n']})" for r in secondary]
        lines += ["", "| policy · budget | accuracy | found | soundness |", "|---|---:|---:|---:|"]
        f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        lines += [f"| {k} | {f(c['accuracy'])} | {f(c['found_within_budget'])} | {f(c['soundness'])} |" for k, c in summary["cells"].items()]
        lines.append("")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", default="none"); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--policies", default="score,random,bfs,sos,ceiling"); r.add_argument("--budgets", default="10,30")
    r.add_argument("--seed", type=int, default=1); r.add_argument("--per-depth", type=int, default=100); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--device", default=None); r.add_argument("--batch-size", type=int, default=16); r.add_argument("--smoke", action="store_true")
    r.add_argument("--depths", default=None, help="harder problems (pre-registration §15): proof depths, e.g. 2,3,4")
    r.add_argument("--decoy", type=float, default=0.5); r.add_argument("--decoy-depth", type=int, default=1)
    r.add_argument("--decoy-fanout", type=int, default=1)
    r.add_argument("--provable-only", action="store_true", help="only provable problems (the primary's population)")
    r.add_argument("--derive-budgets", action="store_true",
                   help="deterministic policies run once at the largest budget; smaller budgets are read off its step count")
    q = sub.add_parser("retrieval"); q.add_argument("--output", type=Path, required=True); q.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("report"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"run": run, "retrieval": run_retrieval, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
