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

**Premise selection at scale (LG-P3b, CPU; pre-registration §19):** the same memory at up to 10⁵ clauses, with correlated
class names (`independent`, `shared` sub-words, `near-dup` clusters) and codes (`random`, character-trigram `subword`,
`mixed`), on exact and one-edit typo queries, against an exact symbol index and the index with one-edit expansion.

    python -m vsa_embed.experiments.e14_prover run --host H-q17 --output OUT [--policies score,random,bfs,sos,ceiling,react,unsound,cot]
        [--budgets 10,30] [--limit N]
    python -m vsa_embed.experiments.e14_prover retrieval --output OUT
    python -m vsa_embed.experiments.e14_prover retrieval-scale --output OUT [--seeds 1,2,3] [--sizes ...] [--widths ...]
    python -m vsa_embed.experiments.e14_prover report --runs OUT ... --output DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


# -- premise selection at scale with correlated class names (LG-P3b; pre-registration §19) ------------------------------

P3B_SIZES = (50, 500, 10_000, 100_000)
P3B_WIDTHS = (1024, 4096, 8192)
P3B_NAMES = ("independent", "shared", "near-dup")
P3B_CODES = ("random", "subword", "mixed")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def stable_generator(*key: Any):
    """A torch generator seeded from SHA-256 of `key` (reproducible in every process, unlike Python's salted `hash`)."""
    import hashlib
    import torch
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little") % (2 ** 63))


def edits1(name: str) -> set[str]:
    """Every string at Levenshtein distance 1 from `name` over a–z (deletion, substitution, insertion)."""
    out = {name[:i] + name[i + 1:] for i in range(len(name))}
    out |= {name[:i] + ch + name[i + 1:] for i in range(len(name)) for ch in LETTERS if ch != name[i]}
    out |= {name[:i] + ch + name[i:] for i in range(len(name) + 1) for ch in LETTERS}
    out.discard(name)
    return out


def one_edit(name: str, rng: random.Random) -> str:
    op, i = rng.choice(("sub", "ins", "del")), rng.randrange(len(name))
    if op == "del" and len(name) > 4:
        return name[:i] + name[i + 1:]
    if op == "ins":
        return name[:i] + rng.choice(LETTERS) + name[i:]
    return name[:i] + rng.choice([c for c in LETTERS if c != name[i]]) + name[i + 1:]


def p3b_class_names(seed: int, n_classes: int, regime: str) -> list[str]:
    """`independent`: invented words (LG-P3's namer); `shared`: two-stem compounds from about √(2C) stems; `near-dup`:
    `shared` compounds in clusters of 4 (a base and 3 one-edit variants)."""
    namer = W.Namer(seed + 99 + {"independent": 0, "shared": 1000, "near-dup": 2000}[regime])
    if regime == "independent":
        return [namer.word() for _ in range(n_classes)]
    rng = random.Random(f"p3b-names:{seed}:{regime}:{n_classes}")
    bases = n_classes if regime == "shared" else -(-n_classes // 4)
    stems = [namer.word(rng.choice((1, 2))) for _ in range(int(math.ceil(math.sqrt(2 * bases))) + 2)]
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < bases:
        a, b = rng.sample(stems, 2)
        if a + b not in seen:
            seen.add(a + b); names.append(a + b)
    if regime == "shared":
        return names
    out: list[str] = []
    for base in names:
        cluster = [base]
        while len(cluster) < 4:
            v = one_edit(base, rng)
            if v not in seen and v.isalpha():
                seen.add(v); cluster.append(v)
        out += cluster
    return out[:n_classes]


def trigrams(name: str) -> list[str]:
    s = f"#{name}#"
    return [s[i:i + 3] for i in range(len(s) - 2)]


def p3b_codes(names: Sequence[str], code: str, width: int, seed: int, *, chunk: int = 4096):
    """Unit class vectors (rows of a C × D tensor): `random` (independent per name), `subword` (the normalised bundle of the
    name's character trigrams), `mixed` (half the energy each)."""
    import torch
    from torch.nn import functional as F
    out = torch.empty(len(names), width)
    if code in ("subword", "mixed"):
        vocab = sorted({t for n in names for t in trigrams(n)})
        tid = {t: i for i, t in enumerate(vocab)}
        tri = torch.stack([torch.randn(width, generator=stable_generator("p3b-trigram", seed, width, t)) for t in vocab])
        for lo in range(0, len(names), chunk):
            rows, cols, vals = [], [], []
            for r, n in enumerate(names[lo:lo + chunk]):
                counts: dict[int, int] = defaultdict(int)
                for t in trigrams(n):
                    counts[tid[t]] += 1
                for c, v in counts.items():
                    rows.append(r); cols.append(c); vals.append(float(v))
            S = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals), (min(chunk, len(names) - lo), len(vocab)))
            out[lo:lo + chunk] = F.normalize(torch.sparse.mm(S, tri), dim=-1)
        del tri
    if code in ("random", "mixed"):
        for lo in range(0, len(names), chunk):
            r = F.normalize(torch.randn(min(chunk, len(names) - lo), width, generator=stable_generator("p3b-random", seed, width, lo)), dim=-1)
            out[lo:lo + chunk] = r if code == "random" else F.normalize(math.sqrt(0.5) * out[lo:lo + chunk] + math.sqrt(0.5) * r, dim=-1)
    return out


def p3b_query_codes(names: Sequence[str], code: str, width: int, seed: int):
    """Codes of query strings that are not class names (typos): the same code functions, the random part per string."""
    import torch
    from torch.nn import functional as F
    rand = torch.stack([F.normalize(torch.randn(width, generator=stable_generator("p3b-query", seed, width, n)), dim=0) for n in names])
    if code == "random":
        return rand
    sub = []
    for n in names:
        sub.append(F.normalize(sum(torch.randn(width, generator=stable_generator("p3b-trigram", seed, width, t)) for t in trigrams(n)), dim=0))
    sub = torch.stack(sub)
    return sub if code == "subword" else F.normalize(math.sqrt(0.5) * sub + math.sqrt(0.5) * rand, dim=-1)


def _ranked_recall(order: Sequence[int], truth: set[int]) -> dict[str, float]:
    t = len(truth)
    hit = lambda m: len(set(order[:m]) & truth) / t  # noqa: E731
    return {"recall_at_5": hit(5 + t - 1), "recall_at_50": hit(50 + t - 1), "r_precision": hit(t)}


def retrieval_scale_study(seed: int, *, sizes: Sequence[int] = P3B_SIZES, widths: Sequence[int] = P3B_WIDTHS,
                          regimes: Sequence[str] = P3B_NAMES, codes: Sequence[str] = P3B_CODES, queries: int = 300,
                          chunk: int = 4096, log=print) -> list[dict[str, Any]]:
    """LG-P3b (pre-registration §19): LG-P3's clause memory (one vector HEAD⊛head + BODY⊛body per clause; unbind HEAD, rank by
    cosine with the goal's class vector) at up to 10⁵ clauses, with correlated class names and codes, against an exact
    symbol index (and the index with one-edit expansion); exact-name and one-edit typo queries."""
    import torch
    from torch.nn import functional as F
    from ..algebra import UnitaryHRRAlgebra
    rows: list[dict[str, Any]] = []
    for size in sizes:
        n_classes = max(40, size // 2)
        for regime in regimes:
            t0 = time.monotonic()
            names = p3b_class_names(seed, n_classes, regime)
            name_set = set(names)
            by_name = {n: i for i, n in enumerate(names)}
            rng = random.Random(f"p3b:{seed}:{size}:{regime}")
            rules = [(rng.randrange(n_classes), rng.randrange(n_classes)) for _ in range(size)]           # (body, head)
            heads = torch.tensor([h for _, h in rules]); bodies = torch.tensor([b for b, _ in rules])
            clauses_of: dict[int, list[int]] = defaultdict(list)
            for i, (_, h) in enumerate(rules):
                clauses_of[h].append(i)
            goals = [rng.choice(rules)[1] for _ in range(queries)]
            truths = [set(clauses_of[g]) for g in goals]
            typos = []
            for g in goals:
                for _ in range(1000):
                    v = one_edit(names[g], rng)
                    if v not in name_set:
                        break
                typos.append(v)
            base = {"exp": "lgp3b", "seed": seed, "clauses": size, "classes": n_classes, "names": regime}
            # exact symbol index (head symbol → clauses) and the index looked up with one-edit expansion
            for query, strings in (("exact", [names[g] for g in goals]), ("typo", typos)):
                for method in ("exact", "exact+edit1"):
                    vals = []
                    for s, truth in zip(strings, truths):
                        order = list(clauses_of.get(by_name[s], [])) if s in by_name else []
                        if method == "exact+edit1":
                            for m in sorted(edits1(s) & name_set):
                                order += clauses_of.get(by_name[m], [])
                        vals.append(_ranked_recall(order, truth))
                    rows.append({**base, "codes": "-", "width": 0, "query": query, "method": method, "index_entries": size,
                                 **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}})
            for code in codes:
                for width in widths:
                    t1 = time.monotonic()
                    Fc = p3b_codes(names, code, width, seed)
                    HEAD, BODY = (UnitaryHRRAlgebra.make_role(torch.randn(width, generator=stable_generator("p3b-role", seed, width, r)))
                                  for r in ("HEAD", "BODY"))
                    G = {"exact": Fc[torch.tensor(goals)], "typo": p3b_query_codes(typos, code, width, seed)}
                    Q = torch.cat([F.normalize(G["exact"], dim=-1), F.normalize(G["typo"], dim=-1)])          # (2q, D)
                    scores = torch.empty(size, Q.shape[0])
                    Hf, Bf = torch.fft.rfft(HEAD), torch.fft.rfft(BODY)
                    for lo in range(0, size, chunk):                  # clause vectors built and unbound chunk by chunk:
                        h, b = heads[lo:lo + chunk], bodies[lo:lo + chunk]   # HRRAlgebra's bind and unbind, in the spectrum
                        Vf = torch.fft.rfft(Fc[h]) * Hf + torch.fft.rfft(Fc[b]) * Bf
                        U = F.normalize(torch.fft.irfft(Vf * Hf.conj(), n=width), dim=-1)
                        scores[lo:lo + chunk] = U @ Q.T
                        del Vf, U
                    near = (Fc @ F.normalize(G["exact"], dim=-1).T)                                     # (C, q)
                    near[torch.tensor(goals), torch.arange(queries)] = -2.0
                    nearest = float(near.max(0).values.mean())
                    del near
                    for qi, query in enumerate(("exact", "typo")):
                        vals = []
                        for j, truth in enumerate(truths):
                            col = scores[:, qi * queries + j]
                            order = torch.topk(col, min(50 + len(truth) - 1, size)).indices.tolist()
                            vals.append(_ranked_recall(order, truth))
                        rows.append({**base, "codes": code, "width": width, "query": query, "method": "vsa", "vsa_floats": size * width,
                                     "nearest_class_cos": nearest, "seconds": round(time.monotonic() - t1, 1),
                                     **{k: float(np.mean([v[k] for v in vals])) for k in vals[0]}})
                    del Fc, scores
                    log(f"  {size} clauses, {regime}, {code}, {width}: vsa exact recall@5 {rows[-2]['recall_at_5']:.3f}, "
                        f"typo {rows[-1]['recall_at_5']:.3f}, nearest-class cos {nearest:.2f} ({rows[-1]['seconds']} s)")
            log(f"{size} clauses, {regime}: {time.monotonic() - t0:.0f} s")
    return rows


def p3b_report(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Means over seeds: recall@5 of the VSA retrieval by clauses × names × codes × width, exact queries then typo queries; the
    exact index and its one-edit expansion; R-precision and recall@50 at 10⁵ clauses."""
    f = lambda x: f"{x:.3f}"  # noqa: E731
    mean = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(int(float(r["clauses"])), r["names"], r["codes"], int(float(r["width"])), r["query"], r["method"])].append(r)
    sizes = sorted({k[0] for k in by})
    widths = sorted({k[3] for k in by if k[5] == "vsa"})
    lines = ["# E14 LG-P3b — premise retrieval at scale with correlated class names (pre-registration §19)", "",
             f"Seeds {sorted({int(float(r['seed'])) for r in rows})}; means over seeds; 300 queries per cell. recall@5 = share of the "
             "truth (every clause whose head is the goal) in the top 5 + |truth| − 1.", ""]
    for query in ("exact", "typo"):
        lines += [f"## {query} queries — recall@5", "",
                  "| clauses | names | " + " | ".join(f"vsa {c} {w}" for c in P3B_CODES for w in widths) + " | exact | exact+edit1 |",
                  "|---:|---|" + "---:|" * (len(P3B_CODES) * len(widths) + 2)]
        for n in sizes:
            for reg in P3B_NAMES:
                cells = [by.get((n, reg, c, w, query, "vsa")) for c in P3B_CODES for w in widths]
                ex = [by.get((n, reg, "-", 0, query, m)) for m in ("exact", "exact+edit1")]
                if not any(cells):
                    continue
                lines.append(f"| {n:,} | {reg} | " + " | ".join(f(mean(c, 'recall_at_5')) if c else "—" for c in cells + ex) + " |")
        lines.append("")
    big = max(sizes)
    lines += [f"## {big:,} clauses, exact queries — R-precision / recall@50 / nearest-class cosine (vsa)", "",
              "| names | codes | " + " | ".join(str(w) for w in widths) + " |", "|---|---|" + "---:|" * len(widths)]
    for reg in P3B_NAMES:
        for c in P3B_CODES:
            cells = [by.get((big, reg, c, w, "exact", "vsa")) for w in widths]
            if any(cells):
                lines.append(f"| {reg} | {c} | " + " | ".join(f"{mean(x, 'r_precision'):.3f} / {mean(x, 'recall_at_50'):.3f} / "
                                                             f"{mean(x, 'nearest_class_cos'):.2f}" if x else "—" for x in cells) + " |")
    lines += ["", f"Memory at {big:,} clauses: the exact index holds {big:,} entries; the VSA memory {big:,} × D floats.", ""]
    return lines


def run_retrieval_scale(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = tuple(int(x) for x in args.sizes.split(","))
    widths = tuple(int(x) for x in args.widths.split(","))
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for seed in seeds:
        rows += retrieval_scale_study(seed, sizes=sizes, widths=widths, queries=args.queries)
        from .e14_algebra import write_csv
        write_csv(rows, out / "retrieval-scale.csv")
    doc = {"experiment": "e14-lgp3b", "seeds": seeds, "sizes": sizes, "widths": widths, "queries": args.queries,
           "smoke": bool(args.smoke), "rows": len(rows), "seconds": round(time.monotonic() - t0, 1)}
    write_json(out / "summary.json", doc)
    (out / "report.md").write_text("\n".join((["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + p3b_report(rows)) + "\n")
    return doc


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
    q3 = sub.add_parser("retrieval-scale", help="LG-P3b (pre-registration §19): retrieval at 10^4-10^5 clauses, correlated names")
    q3.add_argument("--output", type=Path, required=True); q3.add_argument("--seeds", default="1,2,3")
    q3.add_argument("--sizes", default=",".join(map(str, P3B_SIZES))); q3.add_argument("--widths", default=",".join(map(str, P3B_WIDTHS)))
    q3.add_argument("--queries", type=int, default=300); q3.add_argument("--threads", type=int, default=4)
    q3.add_argument("--smoke", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"run": run, "retrieval": run_retrieval, "retrieval-scale": run_retrieval_scale, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
