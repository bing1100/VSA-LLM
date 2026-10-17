"""E14 LG-A — the algebra on the CPU (plan §3; pre-registration §3), and the CPU parts of LG-C (plan §5).

Experiments (`run --experiments`, default all; each writes `<name>.csv` and a section of `report.md`):
- `a1` mapping vectors: rule kind × estimator × k examples × width × fillers (random, correlated 0.3, trained T5 C5 rows) ×
  applied to one fact or an 8-fact frame.
- `a2` chains: load × width × degree profile × clean-up (all / typed) × storage (per entity / one global bundle) × algebra
  (unitary HRR, MAP, FHRR); accuracy up to each hop 1–8.
- `a3` joins (the Datalog fragment): rule shape × number of rules × width; edges decoded from the store by a calibrated
  presence threshold, rules applied to the decoded edges, closure precision / recall / F1 against exact Datalog.
- `a4` nesting and order: nested role–filler terms with and without clean-up per level, convolution chains against
  permutation chains (order), argument swaps in n-ary facts.
- `a5` connectives: k-of-n soft matching, negation as failure by a calibrated threshold (ECE, selective accuracy), ∃ queries.
- `a6` numbers: fractional power encoding; nearness by decoding; `x ≤ t` by decode-and-compare and by a range bundle.
- `a7` trees and lists: node memory, nested vector, path bundle, decision list; depth; numeric tests; input noise.
- `a8` resonator networks: factoring products of 2–3 codebook vectors.
- `a11` the trained stores' own rules: T4 C5 (s1) inverse, symmetric and 2-hop `is_a`, seen against held-out entries.
- `a12` baselines: RotatE / ComplEx (transductive) and AMIE-style mining on the a3 world.
- `c1` similarity completion, `c2` alias resolution by character-trigram hypervectors, `c3` noisy facts with abstention,
  `c4` rules from k examples: mapping vector against AMIE-style mining and E10's structural hypotheses.

    python -m vsa_embed.experiments.e14_algebra run --output experiments/e14-logic/runs/lga [--experiments a1,a2] [--seeds 1,2,3]
        [--quick] [--threads 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..algebra import HRRAlgebra, UnitaryHRRAlgebra
from ..logic import reference as R

HRR = HRRAlgebra()


# -- algebras ------------------------------------------------------------------------------------------------------------

class Alg:
    """`hrr` (unitary roles, Gaussian fillers), `map` (bipolar, elementwise product), `fhrr` (unit phasors, D/2 complex)."""

    def __init__(self, kind: str, width: int, seed: int) -> None:
        self.kind, self.width = kind, width
        self.g = torch.Generator().manual_seed(seed)

    def fillers(self, n: int) -> torch.Tensor:
        d = self.width
        if self.kind == "hrr":
            return torch.randn(n, d, generator=self.g) / math.sqrt(d)
        if self.kind == "map":
            return torch.randint(0, 2, (n, d), generator=self.g).float() * 2 - 1
        return torch.polar(torch.ones(n, d // 2), torch.rand(n, d // 2, generator=self.g) * 2 * math.pi)

    def roles(self, n: int) -> torch.Tensor:
        if self.kind == "hrr":
            return UnitaryHRRAlgebra.make_role(torch.randn(n, self.width, generator=self.g))
        return self.fillers(n)

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return HRR.bind(a, b) if self.kind == "hrr" else a * b

    def unbind(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.kind == "hrr":
            return HRR.unbind(v, r)
        return v * r if self.kind == "map" else v * r.conj()

    def sim(self, q: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """(n, m) cosine similarities."""
        if self.kind == "fhrr":
            qn = q / q.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
            bn = book / book.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
            return (qn @ bn.conj().T).real
        return F.normalize(q, dim=-1) @ F.normalize(book, dim=-1).T


def inverse_vec(v: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft(1.0 / torch.fft.rfft(v), n=v.shape[-1])


def correlated(n: int, d: int, rho: float, g: torch.Generator) -> torch.Tensor:
    shared = torch.randn(d, generator=g)
    z = torch.randn(n, d, generator=g)
    return (math.sqrt(rho) * shared + math.sqrt(1 - rho) * z) / math.sqrt(d)


_TRAINED: dict[str, torch.Tensor] = {}


def trained_rows(run: str = "experiments/e9-retrofit/runs/t5/SmolLM2-360M-full-C5-s1") -> torch.Tensor:
    """The composed store vectors (static frame bundles) of a trained run's entries: correlated, learned fillers."""
    if run not in _TRAINED:
        from .e12_self_query import load_store
        store = load_store(Path(run), "own").store
        rows = store.entry_vectors().detach().float()
        _TRAINED[run] = rows[rows.norm(dim=-1) > 0]
    return _TRAINED[run]


# -- A1: mapping vectors --------------------------------------------------------------------------------------------------

def a1(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    ks = (1, 3, 10, 30, 100) if not quick else (3, 10)
    widths = (256, 1024, 4096) if not quick else (256, 1024)
    for filler_kind in ("random", "correlated", "trained"):
        for width in (widths if filler_kind != "trained" else (int(trained_rows().shape[1]),)):
            g = torch.Generator().manual_seed(seed * 1000 + width)
            n_ent = 2000
            if filler_kind == "random":
                ent = torch.randn(n_ent, width, generator=g) / math.sqrt(width)
            elif filler_kind == "correlated":
                ent = correlated(n_ent, width, 0.3, g)
            else:
                pool = trained_rows()
                idx = torch.randperm(pool.shape[0], generator=g)[:n_ent]
                ent = pool[idx] / pool[idx].norm(dim=-1, keepdim=True) * 1.0
            role = lambda n: UnitaryHRRAlgebra.make_role(torch.randn(n, width, generator=g))  # noqa: E731
            preds = role(20)
            PRED, A1_, A2_, A3_ = role(4)
            train_ids, test_ids = torch.arange(0, n_ent // 2), torch.arange(n_ent // 2, n_ent)
            for kind in ("inverse", "symmetric", "subrelation", "rotate3"):
                p, q = preds[0], preds[1] if kind != "symmetric" else preds[0]

                def fact(pr, a, b, c=None):
                    v = HRR.bind(PRED, pr) + HRR.bind(A1_, a) + HRR.bind(A2_, b)
                    return v + HRR.bind(A3_, c) if c is not None else v

                def conclusion(a, b, c=None):
                    if kind in ("inverse", "symmetric"):
                        return fact(q, b, a)
                    if kind == "subrelation":
                        return fact(q, a, b)
                    return fact(q, c, a, b)

                def draw(ids, n):
                    perm = ids[torch.randperm(len(ids), generator=g)]
                    return perm[:n], perm[n:2 * n], perm[2 * n:3 * n] if 3 * n <= len(perm) else perm[torch.randint(len(perm), (n,), generator=g)]

                ta, tb, tc = draw(test_ids, 300)
                three = kind == "rotate3"
                test_in = fact(p, ent[ta], ent[tb], ent[tc] if three else None)
                others = torch.stack([fact(preds[2 + j % 18], ent[torch.randint(n_ent, (300,), generator=g)],
                                           ent[torch.randint(n_ent, (300,), generator=g)]) for j in range(7)]).sum(0)
                gold_slots = {"inverse": (tb, ta), "symmetric": (tb, ta), "subrelation": (ta, tb), "rotate3": (tc, ta)}[kind]

                def score(M, applied_to: str) -> float:
                    x = test_in + (others if applied_to == "frame8" else 0)
                    out = HRR.bind(M.expand_as(x), x)
                    ok_p = (F.normalize(HRR.unbind(out, PRED.expand_as(out)), dim=-1) @ F.normalize(preds, dim=-1).T).argmax(-1) == (1 if kind != "symmetric" else 0)
                    ok_1 = (F.normalize(HRR.unbind(out, A1_.expand_as(out)), dim=-1) @ F.normalize(ent, dim=-1).T).argmax(-1) == gold_slots[0]
                    ok_2 = (F.normalize(HRR.unbind(out, A2_.expand_as(out)), dim=-1) @ F.normalize(ent, dim=-1).T).argmax(-1) == gold_slots[1]
                    return float((ok_p & ok_1 & ok_2).float().mean())

                if kind in ("inverse", "symmetric"):
                    hand = HRR.bind(A2_, inverse_vec(A1_)) + HRR.bind(A1_, inverse_vec(A2_)) + HRR.bind(q, inverse_vec(p))
                elif kind == "subrelation":
                    hand = HRR.bind(q, inverse_vec(p)) + HRR.bind(A1_, inverse_vec(A1_)) + HRR.bind(A2_, inverse_vec(A2_))
                else:
                    hand = HRR.bind(q, inverse_vec(p)) + HRR.bind(A1_, inverse_vec(A3_)) + HRR.bind(A2_, inverse_vec(A1_)) + HRR.bind(A3_, inverse_vec(A2_))
                for applied in ("fact", "frame8"):
                    rows.append({"exp": "a1", "seed": seed, "fillers": filler_kind, "width": width, "kind": kind, "estimator": "hand",
                                 "k": 0, "applied_to": applied, "accuracy": score(hand, applied)})
                for k in ks:
                    a, b, c = draw(train_ids, k)
                    x = fact(p, ent[a], ent[b], ent[c] if three else None)
                    y = conclusion(ent[a], ent[b], ent[c] if three else None)
                    mean_m = HRR.unbind(y, x).mean(0)
                    X, Y = torch.fft.rfft(x), torch.fft.rfft(y)
                    lsq = torch.fft.irfft((Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3), n=width)
                    for name, M in (("mean", mean_m), ("lsq", lsq)):
                        for applied in ("fact", "frame8"):
                            rows.append({"exp": "a1", "seed": seed, "fillers": filler_kind, "width": width, "kind": kind, "estimator": name,
                                         "k": k, "applied_to": applied, "accuracy": score(M, applied)})
    return rows


# -- A2: chains -----------------------------------------------------------------------------------------------------------

def a2(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    depth = 8
    n_ent, n_rel, n_chains = (5000, 24, 200) if not quick else (1500, 24, 60)
    for alg_kind in ("hrr", "map", "fhrr"):
        for width in ((256, 512, 1024, 2048) if not quick else (256, 1024)):
            for load in ((4, 8, 16, 32) if not quick else (8, 16)):
                for profile in ("fixed", "heavy"):
                    A = Alg(alg_kind, width, seed * 7919 + width + load)
                    ent = A.fillers(n_ent)
                    rel = A.roles(n_rel)
                    rng = random.Random(f"a2:{seed}:{alg_kind}:{width}:{load}:{profile}")
                    perm = list(range(n_ent))
                    rng.shuffle(perm)
                    chains = [perm[i * (depth + 1):(i + 1) * (depth + 1)] for i in range(n_chains)]
                    nxt = {}
                    for c in chains:
                        for i in range(depth):
                            nxt[c[i]] = c[i + 1]
                    frames = torch.zeros_like(A.bind(rel[0].expand(n_ent, -1), ent))
                    for e in range(n_ent):
                        L = load if profile == "fixed" else max(1, min(4 * load, int(round(load * 0.5 * (1 + rng.paretovariate(1.5) - 1)))))
                        k_other = L - (1 if e in nxt else 0)
                        rs = [rng.randrange(1, n_rel) for _ in range(max(0, k_other))]
                        ps = [rng.randrange(n_ent) for _ in rs]
                        v = sum((A.bind(rel[r], ent[p]) for r, p in zip(rs, ps)), torch.zeros_like(frames[0]))
                        if e in nxt:
                            v = v + A.bind(rel[0], ent[nxt[e]])
                        frames[e] = v
                    targets = sorted({c[i] for c in chains for i in range(1, depth + 1)})
                    for cleanup in ("all", "typed"):
                        book_ids = torch.arange(n_ent) if cleanup == "all" else torch.tensor(targets)
                        book = ent[book_ids]
                        cur = torch.tensor([c[0] for c in chains])
                        alive = torch.ones(len(chains), dtype=torch.bool)
                        acc = {}
                        for h in range(1, depth + 1):
                            q = A.unbind(frames[cur], rel[0].expand(len(cur), -1))
                            cur = book_ids[A.sim(q, book).argmax(-1)]
                            alive &= cur == torch.tensor([c[h] for c in chains])
                            acc[h] = float(alive.float().mean())
                        rows.append({"exp": "a2", "seed": seed, "algebra": alg_kind, "width": width, "load": load, "profile": profile,
                                     "cleanup": cleanup, "storage": "per-entity", **{f"hop{h}": v for h, v in acc.items()}})
                    if load <= 8 and profile == "fixed":           # one global bundle of every fact: the documented failure
                        G = torch.zeros_like(frames[0])
                        for e in range(n_ent):
                            G = G + A.bind(ent[e], frames[e])
                        cur = torch.tensor([c[0] for c in chains])
                        alive = torch.ones(len(chains), dtype=torch.bool)
                        acc = {}
                        for h in range(1, depth + 1):
                            q = A.unbind(A.unbind(G.expand(len(cur), -1), ent[cur]), rel[0].expand(len(cur), -1))
                            cur = A.sim(q, ent).argmax(-1)
                            alive &= cur == torch.tensor([c[h] for c in chains])
                            acc[h] = float(alive.float().mean())
                        rows.append({"exp": "a2", "seed": seed, "algebra": alg_kind, "width": width, "load": load, "profile": profile,
                                     "cleanup": "all", "storage": "global", **{f"hop{h}": v for h, v in acc.items()}})
    return rows


# -- A3: joins (Datalog fragment) ----------------------------------------------------------------------------------------

SHAPES = ("chain", "star", "inherit", "conj", "naf", "recursion")


def a3_world(seed: int, *, n_ent: int = 1000, n_base: int = 8, load: int = 8) -> dict[str, set]:
    rng = random.Random(f"a3w:{seed}")
    facts: dict[str, set] = defaultdict(set)
    for e in range(n_ent):
        for _ in range(load):
            facts[f"b{rng.randrange(n_base)}"].add((e, rng.randrange(n_ent)))
    return facts


def a3_rules(shape: str, n_rules: int, n_base: int, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for i in range(n_rules):
        b1, b2 = f"b{rng.randrange(n_base)}", f"b{rng.randrange(n_base)}"
        h = f"h{i}"
        if shape == "chain":
            out.append({"shape": "chain", "body": [b1, b2], "head": h})
        elif shape == "star":
            out.append({"shape": "star", "body": [b1, b2], "head": h})
        elif shape == "inherit":                                    # is_a(x,c) ∧ p(c,v) → p'(x,v): a chain with is_a first
            out.append({"shape": "chain", "body": ["b0", b2], "head": h})
        elif shape == "conj":                                      # p(x,y) ∧ q(x,y) → h: as a chain through the identity? use sub ∧ filter
            out.append({"shape": "sub", "body": [b1], "head": h, "not": None, "and": b2})
        elif shape == "naf":
            out.append({"shape": "sub", "body": [b1], "head": h, "not": b2})
        else:
            out += [{"shape": "sub", "body": [b1], "head": f"anc{i}"}, {"shape": "chain", "body": [f"anc{i}", b1], "head": f"anc{i}"}]
    return out


def apply_rules(facts: dict[str, set], rules: list[dict[str, Any]]) -> dict[str, set]:
    plain = [r for r in rules if "and" not in r or r.get("and") is None]
    db = R.datalog(facts, [{k: v for k, v in r.items() if k != "and"} for r in plain])
    for r in rules:                                                 # 2-premise conjunction on the same pair
        if r.get("and"):
            db[r["head"]] = {e for e in facts.get(r["body"][0], set()) if e in facts.get(r["and"], set())}
    return db


def decode_world(facts: dict[str, set], *, width: int, seed: int, threshold: float, n_ent: int = 1000, n_base: int = 8) -> dict[str, set]:
    """Per-entity frames Σ R_r ⊛ e_o; every (entity, relation) unbound; fillers above `threshold` kept."""
    A = Alg("hrr", width, seed)
    ent, rel = A.fillers(n_ent), A.roles(n_base)
    frames = torch.zeros(n_ent, width)
    for r, pairs in facts.items():
        ri = int(r[1:])
        for s, o in pairs:
            frames[s] += A.bind(rel[ri], ent[o])
    out: dict[str, set] = defaultdict(set)
    keys = F.normalize(ent, dim=-1)
    for ri in range(n_base):
        q = A.unbind(frames, rel[ri].expand(n_ent, -1))
        scores = F.normalize(q, dim=-1) @ keys.T
        s_idx, o_idx = torch.nonzero(scores >= threshold, as_tuple=True)
        out[f"b{ri}"] = set(zip(s_idx.tolist(), o_idx.tolist()))
    return out


def calibrate_threshold(width: int, seed: int) -> float:
    facts = a3_world(seed + 10_000)
    best, best_f1 = 0.2, -1.0
    for t in np.arange(0.08, 0.6, 0.02):
        dec = decode_world(facts, width=width, seed=seed + 10_000, threshold=float(t))
        tp = sum(len(dec[r] & facts[r]) for r in facts)
        p = tp / max(1, sum(len(dec[r]) for r in dec)); rc = tp / max(1, sum(len(v) for v in facts.values()))
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def a3(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    facts = a3_world(seed)
    for width in ((256, 1024) if not quick else (1024,)):
        t = calibrate_threshold(width, seed)
        dec = decode_world(facts, width=width, seed=seed, threshold=t)
        bp, br, bf = prf({(r, e) for r in dec for e in dec[r]}, {(r, e) for r in facts for e in facts[r]})
        for shape in SHAPES:
            for n_rules in ((1, 5, 20) if not quick else (5,)):
                rules = a3_rules(shape, n_rules, 8, random.Random(f"a3r:{seed}:{shape}:{n_rules}"))
                gold = apply_rules(facts, rules)
                got = apply_rules(dec, rules)
                heads = sorted({r["head"] for r in rules})
                G = {(h, e) for h in heads for e in gold.get(h, set())}
                P = {(h, e) for h in heads for e in got.get(h, set())}
                p, r, f1 = prf(P, G)
                rows.append({"exp": "a3", "seed": seed, "width": width, "threshold": t, "shape": shape, "rules": n_rules,
                             "base_precision": bp, "base_recall": br, "base_f1": bf, "precision": p, "recall": r, "f1": f1,
                             "derived_gold": len(G), "derived_pred": len(P), "false_per_true": (len(P - G) / max(1, len(P & G)))})
    return rows


# -- A4: nesting and order -------------------------------------------------------------------------------------------------

_PERM: dict[int, torch.Tensor] = {}


def perm(v: torch.Tensor) -> torch.Tensor:
    """A fixed random permutation of the coordinates (a cyclic shift would commute with circular convolution)."""
    d = v.shape[-1]
    if d not in _PERM:
        _PERM[d] = torch.randperm(d, generator=torch.Generator().manual_seed(4242))
    return v[..., _PERM[d]]


def a4(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    for width in (256, 1024):
        g = torch.Generator().manual_seed(seed * 31 + width)
        funcs = torch.randn(30, width, generator=g) / math.sqrt(width)
        atoms = torch.randn(200, width, generator=g) / math.sqrt(width)
        FUNC, ARG = (UnitaryHRRAlgebra.make_role(torch.randn(2, width, generator=g)))
        for depth in (1, 2, 3, 4):
            trials = 200
            rng = random.Random(f"a4:{seed}:{width}:{depth}")
            ok_nc = ok_cl = 0
            for _ in range(trials):
                fs = [rng.randrange(30) for _ in range(depth)]
                a = rng.randrange(200)
                terms = [atoms[a]]
                for f in reversed(fs):                              # term = FUNC⊛f + ARG⊛inner (normalised per level)
                    t = HRR.bind(FUNC, funcs[f]) + HRR.bind(ARG, terms[-1])
                    terms.append(t / t.norm())
                memory = torch.stack(terms[:-1])                    # stored intermediate terms (for clean-up per level)
                for cleanup in (False, True):
                    v, ok = terms[-1], True
                    for level, f in enumerate(fs):
                        got_f = int((F.normalize(HRR.unbind(v, FUNC), dim=0) @ F.normalize(funcs, dim=-1).T).argmax())
                        ok &= got_f == f
                        v = HRR.unbind(v, ARG)
                        if cleanup:
                            v = memory[int((F.normalize(v, dim=0) @ F.normalize(memory, dim=-1).T).argmax())]
                    ok &= int((F.normalize(v, dim=0) @ F.normalize(atoms, dim=-1).T).argmax()) == a
                    if cleanup:
                        ok_cl += ok
                    else:
                        ok_nc += ok
            rows.append({"exp": "a4", "seed": seed, "width": width, "test": "nested-decode", "depth": depth,
                         "no_cleanup": ok_nc / trials, "cleanup_per_level": ok_cl / trials})
        # order: f(g(a)) against g(f(a)); plain convolution chain and permutation chain
        for enc in ("conv", "perm"):
            trials, ok = 300, 0
            rng = random.Random(f"a4o:{seed}:{width}:{enc}")
            for _ in range(trials):
                f, h = rng.sample(range(30), 2)
                a = atoms[rng.randrange(200)]
                if enc == "conv":
                    t1 = HRR.bind(funcs[f], HRR.bind(funcs[h], a)); t2 = HRR.bind(funcs[h], HRR.bind(funcs[f], a))
                else:
                    t1 = HRR.bind(funcs[f], perm(HRR.bind(funcs[h], perm(a)))); t2 = HRR.bind(funcs[h], perm(HRR.bind(funcs[f], perm(a))))
                noisy = t1 + torch.randn(width, generator=g) * 0.3 * t1.norm() / math.sqrt(width)
                s1, s2 = float(F.cosine_similarity(noisy, t1, dim=0)), float(F.cosine_similarity(noisy, t2, dim=0))
                ok += 1.0 if s1 > s2 + 1e-9 else (0.5 if abs(s1 - s2) <= 1e-9 else 0.0)
            rows.append({"exp": "a4", "seed": seed, "width": width, "test": "order", "encoding": enc, "accuracy": ok / trials})
        for arity in (2, 3, 4, 5):                                  # argument swap in n-ary facts
            roles = UnitaryHRRAlgebra.make_role(torch.randn(arity, width, generator=g))
            trials, ok = 200, 0
            rng = random.Random(f"a4a:{seed}:{width}:{arity}")
            for _ in range(trials):
                args = rng.sample(range(200), arity)
                v = sum(HRR.bind(roles[i], atoms[a]) for i, a in enumerate(args))
                got = [int((F.normalize(HRR.unbind(v, roles[i]), dim=0) @ F.normalize(atoms, dim=-1).T).argmax()) for i in range(arity)]
                ok += got == args
            rows.append({"exp": "a4", "seed": seed, "width": width, "test": "arity", "arity": arity, "accuracy": ok / trials})
    return rows


# -- A5: connectives and quantifiers -------------------------------------------------------------------------------------

def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def a5(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    for width in (256, 1024):
        for load in (8, 16):
            g = torch.Generator().manual_seed(seed * 17 + width + load)
            atoms = torch.randn(500, width, generator=g) / math.sqrt(width)
            rng = random.Random(f"a5:{seed}:{width}:{load}")

            def bundle(ids):
                return atoms[list(ids)].sum(0)
            pos, neg = [], []                                       # membership (negation as failure): present vs absent
            for _ in range(400):
                ids = rng.sample(range(500), load)
                b = bundle(ids)
                pos.append(float(F.cosine_similarity(b, atoms[ids[0]], dim=0)))
                neg.append(float(F.cosine_similarity(b, atoms[rng.choice([i for i in range(500) if i not in ids])], dim=0)))
            half = len(pos) // 2
            thr = (np.mean(pos[:half]) + np.max(neg[:half])) / 2           # calibrated on the first half
            test_s = np.array(pos[half:] + neg[half:]); test_y = np.array([1] * (len(pos) - half) + [0] * (len(neg) - half))
            pred = (test_s >= thr).astype(int)
            conf = 1 / (1 + np.exp(-(test_s - thr) * 20))
            correct = (pred == test_y).astype(float)
            margin = np.abs(test_s - thr)
            keep = margin >= np.quantile(margin, 0.2)
            rows.append({"exp": "a5", "seed": seed, "width": width, "load": load, "test": "naf", "auc": auc(pos, neg),
                         "accuracy": float(correct.mean()), "ece": ece(np.where(pred == 1, conf, 1 - conf), correct),
                         "selective_accuracy_80": float(correct[keep].mean())})
            for n in (2, 3, 4):                                     # k-of-n soft matching: all n premises against n − 1
                full, part = [], []
                for _ in range(300):
                    prem = rng.sample(range(500), n)
                    other = rng.sample([i for i in range(500) if i not in prem], load - n)
                    record_full = bundle(prem + other)
                    record_part = bundle(prem[:-1] + other + [rng.choice([i for i in range(500) if i not in prem + other])])
                    score = lambda rec: sum(float(F.cosine_similarity(rec, atoms[p], dim=0)) for p in prem)  # noqa: E731
                    full.append(score(record_full)); part.append(score(record_part))
                rows.append({"exp": "a5", "seed": seed, "width": width, "load": load, "test": f"and-{n}", "auc": auc(full, part)})
            ex_pos, ex_neg = [], []                                 # ∃: is there any fact with predicate p (PRED⊛p + …)
            P = torch.randn(20, width, generator=g) / math.sqrt(width)
            ROLE = UnitaryHRRAlgebra.make_role(torch.randn(width, generator=g))
            for _ in range(300):
                ps = rng.sample(range(20), min(load, 19))
                mem = sum(HRR.bind(ROLE, P[p]) + atoms[rng.randrange(500)] for p in ps)
                q = HRR.unbind(mem, ROLE)
                ex_pos.append(float(F.cosine_similarity(q, P[ps[0]], dim=0)))
                ex_neg.append(float(F.cosine_similarity(q, P[[i for i in range(20) if i not in ps][0]], dim=0)) if len(ps) < 20 else 0.0)
            rows.append({"exp": "a5", "seed": seed, "width": width, "load": load, "test": "exists", "auc": auc(ex_pos, ex_neg)})
    return rows


# -- A6: numbers --------------------------------------------------------------------------------------------------------------

def a6(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    for width in (256, 1024):
        for levels in (10, 100):
            g = torch.Generator().manual_seed(seed * 13 + width + levels)
            phases = torch.rand(width // 2 + 1, generator=g) * 2 * math.pi
            phases[0] = 0
            phases[-1] = 0
            scale = 10.0 / (100 / levels)
            num = lambda x: torch.fft.irfft(torch.polar(torch.ones_like(phases), phases * (x / scale)), n=width)  # noqa: E731
            grid = torch.stack([num(float(x)) for x in range(levels + 1)])
            roles = UnitaryHRRAlgebra.make_role(torch.randn(8, width, generator=g))
            rng = random.Random(f"a6:{seed}:{width}:{levels}")
            exact = le_dec = le_bun = 0
            trials = 300
            for _ in range(trials):
                values = [rng.randint(0, levels) for _ in range(8)]
                record = sum(HRR.bind(roles[i], num(float(v))) for i, v in enumerate(values))
                v = HRR.unbind(record, roles[0])
                dec = int((F.normalize(grid, dim=-1) @ F.normalize(v, dim=0)).argmax())
                exact += dec == values[0]
                t = rng.randint(1, levels - 1)
                le_dec += (dec <= t) == (values[0] <= t)
                rbun = grid[: t + 1].sum(0)                              # range bundle: Σ_{x ≤ t} B^x
                inside = float(F.cosine_similarity(v, rbun, dim=0))
                outside = float(F.cosine_similarity(v, grid[t + 1:].sum(0), dim=0)) if t < levels else -1
                le_bun += (inside > outside) == (values[0] <= t)
            rows.append({"exp": "a6", "seed": seed, "width": width, "levels": levels, "decode_exact": exact / trials,
                         "le_decode_compare": le_dec / trials, "le_range_bundle": le_bun / trials})
    return rows


# -- A7: trees and lists --------------------------------------------------------------------------------------------------

def a7(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    from ..logic import store as S
    from ..logic import world as W
    rows = []
    for width in (256, 1024):
        for numeric in (False, True):
            for noise in (0.0, 0.1):
                namer = W.Namer(seed * 100 + width, forbidden=set())
                world = W.trees(seed * 10 + width, depths=(1, 2, 3, 4, 6, 8) if not quick else (2, 4), inputs_per_depth=40 if not quick else 10,
                                list_lengths=(3, 6, 12), inputs_per_list=30 if not quick else 8, edits=0, namer=namer)["items"]
                rng = random.Random(f"a7:{seed}:{width}:{numeric}:{noise}")
                by: dict[tuple, list] = defaultdict(list)
                for item in world:
                    if item["kind"] == "tree" and item["numeric"] != numeric:
                        continue
                    record = dict(item["record"])
                    if noise:
                        for a in item["attributes"]:
                            if rng.random() < noise:
                                record[a["name"]] = rng.choice(a["values"]) if a["values"] else rng.randint(0, 100)
                    noisy = {**item, "record": record}
                    gold = item["answer"]
                    node = S.TreeStore(noisy, width=width, seed=seed).execute().answer
                    by[(item["kind"], item["depth"], "node-memory")].append(node == gold)
                    by[(item["kind"], item["depth"], "exact-on-noisy")].append(
                        (R.run_tree(item["tree"], record)[0] if item["kind"] == "tree" else R.run_list(item["rules"], item["default"], record)[0]) == gold)
                    if item["kind"] == "tree":
                        by[(item["kind"], item["depth"], "path-bundle")].append(_path_bundle(noisy, width, seed) == gold)
                        by[(item["kind"], item["depth"], "nested-vector")].append(_nested_tree(noisy, width, seed) == gold)
                for (kind, depth, enc), v in sorted(by.items()):
                    rows.append({"exp": "a7", "seed": seed, "width": width, "numeric": numeric, "noise": noise, "kind": kind,
                                 "depth": depth, "encoding": enc, "accuracy": float(np.mean(v)), "n": len(v)})
    return rows


def _record_vec(cb, item, store) -> torch.Tensor:
    return store.record


def _path_bundle(item: dict[str, Any], width: int, seed: int) -> str:
    """Flat bundle of root-to-leaf paths: each path's conditions bundled; the input's record matched by the share of the
    path's conditions it satisfies (read by unbinding and clean-up), the best path's leaf."""
    from ..logic import store as S
    from ..logic.world import test_text  # noqa: F401
    st = S.TreeStore(item, width=width, seed=seed)
    numeric = {a["name"]: a["values"] is None for a in item["attributes"]}
    paths: list[tuple[list[tuple[dict, bool]], str]] = []

    def walk(node, conds):
        if "leaf" in node:
            paths.append((conds, node["leaf"]))
            return
        walk(node["yes"], conds + [(node["test"], True)])
        walk(node["no"], conds + [(node["test"], False)])
    walk(item["tree"], [])
    best, best_score = None, -1.0
    for conds, leaf in paths:
        sat = 0
        for test, want in conds:
            value, _ = st.read_value(test["attr"], numeric[test["attr"]])
            ok = value == test["value"] if test["op"] == "is" else float(value) <= float(test["value"])
            sat += ok == want
        score = sat / max(1, len(conds))
        if score > best_score:
            best, best_score = leaf, score
    return best


def _nested_tree(item: dict[str, Any], width: int, seed: int) -> str:
    """The whole tree in one vector (children bound inside parents, normalised per level), read without clean-up."""
    from ..logic import store as S
    st = S.TreeStore(item, width=width, seed=seed)
    cb = st.cb
    numeric = {a["name"]: a["values"] is None for a in item["attributes"]}
    leaves = sorted({o for o in item["options"]})

    def enc(node):
        if "leaf" in node:
            return F.normalize(cb.filler(node["leaf"]), dim=0)
        t = node["test"]
        v = cb.bind("TEST", "attr:" + t["attr"]) + cb.bind("VAL", str(t["value"])) + HRR.bind(cb.role("YES"), enc(node["yes"])) \
            + HRR.bind(cb.role("NO"), enc(node["no"]))
        return F.normalize(v, dim=0)
    v = enc(item["tree"])
    attrs = [a["name"] for a in item["attributes"]]
    node = item["tree"]
    for _ in range(12):
        if "leaf" in node:
            break
        name, _ = cb.cleanup(HRR.unbind(v, cb.role("TEST")), ["attr:" + a for a in attrs])
        attr = name.split(":", 1)[1]
        # the test value is cleaned up among values of the decoded attribute's kind (numbers or that attribute's categories)
        spec = next(a for a in item["attributes"] if a["name"] == attr)
        test_vals = [str(x) for x in range(101)] if spec["values"] is None else [str(x) for x in spec["values"]]
        val, _ = cb.cleanup(HRR.unbind(v, cb.role("VAL")), test_vals)
        value, _ = st.read_value(attr, numeric[attr])
        ok = (str(value) == val) if not numeric[attr] else float(value) <= float(val)
        v = HRR.unbind(v, cb.role("YES" if ok else "NO"))
        node = node["yes"] if ok else node["no"]                    # only to stop at the leaf depth
    return cb.cleanup(v, leaves)[0]


# -- A8: resonator networks ---------------------------------------------------------------------------------------------

def a8(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = []
    width = 1024
    for factors in (2, 3):
        for size in ((50, 200, 1000) if not quick else (50, 200)):
            g = torch.Generator().manual_seed(seed * 101 + factors * 7 + size)
            books = [torch.randint(0, 2, (size, width), generator=g).float() * 2 - 1 for _ in range(factors)]   # bipolar (MAP)
            trials, ok, iters_used = 50 if not quick else 15, 0, []
            for t in range(trials):
                idx = [int(torch.randint(size, (1,), generator=g)) for _ in range(factors)]
                s = torch.ones(width)
                for b, i in zip(books, idx):
                    s = s * b[i]
                est = [b.sum(0) for b in books]                       # start: the superposition of each codebook
                for it in range(100):
                    changed = False
                    for f in range(factors):                       # sequential (one factor at a time) updates
                        others = torch.ones(width)
                        for j in range(factors):
                            if j != f:
                                others = others * est[j]
                        new = torch.sign(books[f].T @ (books[f] @ (s * others))) + 1e-9
                        changed |= not torch.equal(torch.sign(new), torch.sign(est[f]))
                        est[f] = torch.sign(new)
                    if not changed:
                        break
                got = [int((books[f] @ est[f]).argmax()) for f in range(factors)]
                ok += got == idx
                iters_used.append(it + 1)
            rows.append({"exp": "a8", "seed": seed, "width": width, "factors": factors, "codebook": size, "accuracy": ok / trials,
                         "mean_iterations": float(np.mean(iters_used))})
    return rows


# -- A11: the trained stores' own rules -------------------------------------------------------------------------------------

def a11(seed: int, *, quick: bool = False, run: str = "experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1") -> list[dict[str, Any]]:
    from .e12_self_query import load_store
    from .e9_binding_chain import atom_entries
    loaded = load_store(Path(run), "own")
    store, o = loaded.store, loaded.ontology
    rel = {n: i for i, n in enumerate(o["relation_names"])}
    offsets, relations, fillers = np.asarray(o["offsets"]), np.asarray(o["relations"]), np.asarray(o["fillers"])
    atom_entry = atom_entries(o)
    frame = lambda e: list(zip(relations[offsets[e]:offsets[e + 1]].tolist(), fillers[offsets[e]:offsets[e + 1]].tolist()))  # noqa: E731
    held = {int(e) for e in o.get("heldout_entries", ())}
    freq = np.asarray(o.get("train_frequency") or np.zeros(len(offsets) - 1))
    vectors = store.entry_vectors()
    rng = random.Random(f"a11:{seed}")
    rows = []

    def entry_of(atom: int) -> int | None:
        e = int(atom_entry[atom]) if atom < len(atom_entry) else -1
        return e if e >= 0 else None

    def decode(e: int, r: int, m: int) -> list[int]:
        lines = store.decode_slots(vectors[e], [r] * m)
        return [f.atom for line in lines for f in line.fillers]

    pools = {"held-out": sorted(held), "seen": [e for e in np.flatnonzero(freq >= 10).tolist() if e not in held]}
    rules = [("inverse", "is_conjugate_acid_of", "is_conjugate_base_of"), ("inverse", "is_conjugate_base_of", "is_conjugate_acid_of"),
             ("symmetric", "is_enantiomer_of", "is_enantiomer_of"), ("symmetric", "is_tautomer_of", "is_tautomer_of")]
    for subset, pool in pools.items():
        for kind, r1, r2 in rules:
            if r1 not in rel:
                continue
            cases = [(e, a) for e in pool for r, a in frame(e) if r == rel[r1]]
            rng.shuffle(cases)
            cases = cases[:400 if not quick else 80]
            decoded_ok = derived_ok = consistent = 0
            for e, a in cases:
                got = decode(e, rel[r1], 1)[0]
                decoded_ok += got == a
                partner = entry_of(a)
                own_atom = next((i for i, x in enumerate(atom_entry) if x == e), None) if False else None
                if partner is not None:
                    # rule consistency in the gold ontology: does the partner's frame hold the converse edge back to e?
                    back = {x for r, x in frame(partner) if r == rel[r2]}
                    consistent += any(entry_of(x) == e for x in back)
                    derived_ok += (got == a) and any(entry_of(x) == e for x in back)
            n = max(1, len(cases))
            rows.append({"exp": "a11", "seed": seed, "subset": subset, "rule": f"{kind}:{r1}->{r2}", "cases": len(cases),
                         "decode_accuracy": decoded_ok / n, "converse_in_ontology": consistent / n, "derived_and_storable": derived_ok / n})
        if "is_a" in rel:                                            # 2-hop is_a through the parents' own stores
            cases = [e for e in pool if any(r == rel["is_a"] for r, _ in frame(e))]
            rng.shuffle(cases)
            cases = cases[:300 if not quick else 60]
            prec, rec = [], []
            for e in cases:
                parents = [a for r, a in frame(e) if r == rel["is_a"]]
                gold2 = {a2 for a in parents if (pe := entry_of(a)) is not None for r, a2 in frame(pe) if r == rel["is_a"]}
                if not gold2:
                    continue
                got_parents = decode(e, rel["is_a"], len(parents))
                got2 = set()
                for a in got_parents:
                    pe = entry_of(a)
                    if pe is not None:
                        m = sum(1 for r, _ in frame(pe) if r == rel["is_a"]) or 1
                        got2 |= set(decode(pe, rel["is_a"], m))
                if got2:
                    prec.append(len(got2 & gold2) / len(got2))
                rec.append(len(got2 & gold2) / len(gold2))
            rows.append({"exp": "a11", "seed": seed, "subset": subset, "rule": "transitive:is_a 2 hops", "cases": len(rec),
                         "precision": float(np.mean(prec)) if prec else None, "recall": float(np.mean(rec)) if rec else None})
    return rows


# -- A12: baselines ----------------------------------------------------------------------------------------------------------

def a12(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    from .. import kg_baselines as kg
    rows = []
    facts = a3_world(seed, n_ent=400)
    rules = a3_rules("chain", 3, 8, random.Random(f"a12:{seed}")) + a3_rules("star", 2, 8, random.Random(f"a12s:{seed}"))
    for i, r in enumerate(rules):
        r["head"] = f"h{i}"
    gold = apply_rules(facts, rules)
    heads = [r["head"] for r in rules]
    # AMIE-style mining on base + derived facts: are the planted rules found?
    relations = {k: set(v) for k, v in gold.items() if v}
    mined = kg.mine_rules(relations, heads=heads, min_pca=0.5)
    planted = {(r["head"], tuple(r["body"])) for r in rules if r["shape"] == "chain"}
    found = {(m.rule.head, tuple(b for b, _ in m.rule.body)) for m in mined if m.rule.kind == "chain"}
    rows.append({"exp": "a12", "seed": seed, "baseline": "amie-mining", "planted_chain_rules": len(planted),
                 "recovered": len(planted & found), "mined": len(mined)})
    # RotatE / ComplEx: train on base facts + 80% of derived facts; predict the held-out 20% of derived facts (seen entities)
    rel_names = sorted(gold)
    rid = {r: i for i, r in enumerate(rel_names)}
    triples, test = [], []
    rng = random.Random(f"a12t:{seed}")
    for r in rel_names:
        for s, o in sorted(gold[r]):
            (test if r in heads and rng.random() < 0.2 else triples).append((s, rid[r], o))
    T = torch.tensor(triples, dtype=torch.long)
    for kind in ("rotate", "complex"):
        model = kg.train_kge(T, 400, len(rel_names), kind=kind, dimension=64, epochs=400 if not quick else 40, seed=seed)
        hits, hits10 = [], []
        with torch.no_grad():
            for s_, r, o in test[:400]:
                scores = model.score(torch.full((400,), s_), torch.full((400,), r), torch.arange(400))
                hits.append(int(scores.argmax()) == o)
                hits10.append(o in torch.topk(scores, 10).indices.tolist())
        rows.append({"exp": "a12", "seed": seed, "baseline": kind, "derived_hits_at_1_seen": float(np.mean(hits)) if hits else None,
                     "derived_hits_at_10_seen": float(np.mean(hits10)) if hits10 else None,
                     "test_facts": len(test), "unseen_entities": "no embedding (transductive)"})
    # the VSA executor on the same queries: decode base edges, apply the rules
    t = calibrate_threshold(1024, seed)
    dec = decode_world(facts, width=1024, seed=seed, threshold=t, n_ent=400)
    got = apply_rules(dec, rules)
    hits = [(o in {y for x, y in got.get(rel_names[r], set()) if x == s}) for s, r, o in test[:400]]
    rows.append({"exp": "a12", "seed": seed, "baseline": "vsa-executor (width 1024)", "derived_hits_at_1_seen": float(np.mean(hits)) if hits else None,
                 "test_facts": len(test), "unseen_entities": "works (rules apply to any composed entity)"})
    return rows


# -- LG-C CPU parts --------------------------------------------------------------------------------------------------------

def c1(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    """Type-level regularities that hold only mostly: each entity's hidden property is its type's property with probability
    0.8, else another; a share `drop` of entities lack the edge. Completion of the missing edges: the VSA store by the
    majority among the 5 most similar frames of the same type; symbolic rule induction (the majority property of the type,
    AMIE-style); exact Datalog over the stated facts (no rule: nothing)."""
    rows = []
    for width in (256, 1024):
        for drop in (0.25, 0.5):
            g = torch.Generator().manual_seed(seed * 3 + width)
            n_types, per = 20, 30
            TYPE, PROP, OTHER = UnitaryHRRAlgebra.make_role(torch.randn(3, width, generator=g))
            types = torch.randn(n_types, width, generator=g) / math.sqrt(width)
            props = torch.randn(n_types, width, generator=g) / math.sqrt(width)
            noise_atoms = torch.randn(500, width, generator=g) / math.sqrt(width)
            rng = random.Random(f"c1:{seed}:{width}:{drop}")
            frames, ty_of, hidden, missing = [], [], [], []
            for t in range(n_types):
                for _ in range(per):
                    prop = t if rng.random() < 0.8 else rng.choice([x for x in range(n_types) if x != t])
                    has = rng.random() >= drop
                    v = HRR.bind(TYPE, types[t]) + sum(HRR.bind(OTHER, noise_atoms[rng.randrange(500)]) for _ in range(4))
                    if has:
                        v = v + HRR.bind(PROP, props[prop])
                    frames.append(v); ty_of.append(t); hidden.append(prop); missing.append(not has)
            Fm = torch.stack(frames)
            read_type = (F.normalize(HRR.unbind(Fm, TYPE.expand_as(Fm)), dim=-1) @ F.normalize(types, dim=-1).T).argmax(-1).tolist()
            read_prop = (F.normalize(HRR.unbind(Fm, PROP.expand_as(Fm)), dim=-1) @ F.normalize(props, dim=-1).T).argmax(-1).tolist()
            vsa, sym = [], []
            for i in [i for i, m in enumerate(missing) if m]:
                same = [j for j in range(len(frames)) if j != i and not missing[j] and read_type[j] == read_type[i]]
                if not same:
                    vsa.append(False); sym.append(False); continue
                near = sorted(same, key=lambda j: -float(F.cosine_similarity(Fm[i], Fm[j], dim=0)))[:5]
                vsa.append(Counter(read_prop[j] for j in near).most_common(1)[0][0] == hidden[i])
                sym.append(Counter(hidden[j] for j in same).most_common(1)[0][0] == hidden[i])
            rows.append({"exp": "c1", "seed": seed, "width": width, "drop": drop, "vsa_knn_completion": float(np.mean(vsa)),
                         "symbolic_rule_induction": float(np.mean(sym)), "datalog_no_rule": 0.0, "ceiling_type_rule": 0.8,
                         "missing": int(sum(missing))})
    return rows


def c2(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    """Alias resolution: names as bundles of position-permuted character-trigram hypervectors; mentions with 1–2 edits or
    abbreviations resolved by nearest name vector, against exact match and Levenshtein distance."""
    from ..logic import world as W
    rows = []
    namer = W.Namer(seed + 500, forbidden=set())
    names = [namer.word(3) for _ in range(500)]
    rng = random.Random(f"c2:{seed}")

    def lev(a, b):
        d = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, d[0] = d[0], i
            for j, cb in enumerate(b, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
        return d[-1]

    def mutate(n, kind):
        if kind == "edit1":
            i = rng.randrange(len(n)); return n[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + n[i + 1:]
        if kind == "edit2":
            return mutate(mutate(n, "edit1"), "edit1")
        return n[: max(3, len(n) // 2)]                                # abbreviation: the first half
    for width in (1024, 4096):
        g = torch.Generator().manual_seed(seed + width)
        chars = {c: torch.randn(width, generator=g) / math.sqrt(width) for c in "abcdefghijklmnopqrstuvwxyz "}

        def enc(s):
            s = f" {s} "
            grams = [HRR.bind(HRR.bind(torch.roll(chars[s[i]], 2), torch.roll(chars[s[i + 1]], 1)), chars[s[i + 2]]) for i in range(len(s) - 2)]
            return torch.stack(grams).sum(0)
        book = torch.stack([enc(n) for n in names])
        for kind in ("edit1", "edit2", "abbrev"):
            vsa = exact = levn = pref = 0
            trials = 300
            for _ in range(trials):
                i = rng.randrange(len(names))
                m = mutate(names[i], kind)
                vsa += int((F.normalize(book, dim=-1) @ F.normalize(enc(m), dim=0)).argmax()) == i
                exact += m == names[i]
                best = min(range(len(names)), key=lambda j: lev(m, names[j]))
                levn += best == i
                starts = [j for j, n in enumerate(names) if n.startswith(m)]
                pref += (starts[0] if len(starts) == 1 else best) == i       # a fair fuzzy front end: unique prefix, else edit distance
            rows.append({"exp": "c2", "seed": seed, "width": width, "noise": kind, "vsa_trigram": vsa / trials, "exact_match": exact / trials,
                         "levenshtein": levn / trials, "prefix_or_levenshtein": pref / trials})
    return rows


def c3(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    """Wrong edges injected into the a3 world; closure precision / recall from the noisy facts: exact Datalog (all
    facts trusted) against the store with a stricter abstention threshold."""
    rows = []
    clean = a3_world(seed)
    rules = a3_rules("chain", 5, 8, random.Random(f"c3:{seed}"))
    gold = apply_rules(clean, rules)
    G = {(r["head"], e) for r in rules for e in gold[r["head"]]}
    for p in (0.0, 0.1, 0.3):
        rng = random.Random(f"c3n:{seed}:{p}")
        noisy = {r: {(s, rng.randrange(1000)) if rng.random() < p else (s, o) for s, o in pairs} for r, pairs in clean.items()}
        sym = apply_rules(noisy, rules)
        P = {(r["head"], e) for r in rules for e in sym[r["head"]]}
        sp, sr, sf = prf(P, G)
        t = calibrate_threshold(1024, seed)
        out = {"exp": "c3", "seed": seed, "noise": p, "datalog_precision": sp, "datalog_recall": sr, "datalog_f1": sf}
        for name, thr in (("vsa", t), ("vsa-abstain", t + 0.06)):
            dec = decode_world(noisy, width=1024, seed=seed, threshold=thr)
            got = apply_rules(dec, rules)
            Q = {(r["head"], e) for r in rules for e in got[r["head"]]}
            vp, vr, vf = prf(Q, G)
            out.update({f"{name}_precision": vp, f"{name}_recall": vr, f"{name}_f1": vf})
        rows.append(out)
    return rows


def c4(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    """Rules from k examples: the mapping vector (store.MappingRule) against AMIE-style mining from the same k example facts
    and E10's structural hypotheses; accuracy of the conclusion for new entities."""
    from .. import kg_baselines as kg
    from ..logic import store as S
    from ..logic import world as W
    rows = []
    world = W.fewshot(seed, queries=40 if not quick else 10, namer=W.Namer(seed + 700, forbidden=set()))["items"]
    by: dict[tuple, list] = defaultdict(list)
    for item in world:
        by[(item["kind"], item["k"], "mapping-vector")].append(S.MappingRule(item, width=1024, seed=seed).execute().answer == item["answer"])
        rels: dict[str, set] = defaultdict(set)
        for prem, conc in item["examples"]:
            for x, p, y in prem:
                rels[p].add((x, y))
            rels[conc[1]].add((conc[0], conc[2]))
        for x, p, y in item["premises"]:
            rels[p].add((x, y))
        mined = kg.mine_rules(dict(rels), heads=[item["conclusion_relation"]], min_support=1, min_pca=0.5)
        subject, rel = item["question"]
        answer = None
        for m in mined:
            preds = kg.rule_predictions(m.rule, dict(rels))
            cands = [o for s, o in preds if s == subject]
            if cands:
                answer = cands[0]
                break
        by[(item["kind"], item["k"], "amie-mining")].append(answer == item["answer"])
    for (kind, k, method), v in sorted(by.items()):
        rows.append({"exp": "c4", "seed": seed, "kind": kind, "k": k, "method": method, "accuracy": float(np.mean(v)), "n": len(v)})
    return rows


EXPERIMENTS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5, "a6": a6, "a7": a7, "a8": a8, "a11": a11, "a12": a12,
    "c1": c1, "c2": c2, "c3": c3, "c4": c4}
SINGLE_SEED = {"a11"}


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(name: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    """Mean over seeds of every numeric metric, grouped by the non-metric columns."""
    metrics = {k for r in rows for k, v in r.items() if isinstance(v, float) and k not in {"threshold"}}
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = tuple((k, r[k]) for k in sorted(r) if k not in metrics and k not in {"seed", "exp"})
        groups[key].append(r)
    cols = sorted(metrics)
    lines = [f"## {name}", "", "| " + " | ".join([k for k, _ in next(iter(groups))] + cols + ["seeds"]) + " |",
             "|" + "---|" * (len(next(iter(groups))) + len(cols) + 1)]
    for key, rs in groups.items():
        vals = []
        for c in cols:
            xs = [r[c] for r in rs if isinstance(r.get(c), (int, float)) and r.get(c) is not None]
            vals.append(f"{np.mean(xs):.3f}" if xs else "—")
        lines.append("| " + " | ".join([str(v) for _, v in key] + vals + [str(len(rs))]) + " |")
    return lines + [""]


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    names = [n.strip() for n in args.experiments.split(",")] if args.experiments else list(EXPERIMENTS)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"experiments": {}, "seeds": seeds, "quick": bool(args.quick)}
    report = ["# E14 LG-A (algebra, CPU) and LG-C CPU parts" + (" — QUICK SMOKE, not a result" if args.quick else ""), "",
              f"Seeds {seeds}; means over seeds. Pre-registration `experiments/e14-logic/preregistration.md` §3; plan §3, §5.", ""]
    for name in names:
        t0 = time.monotonic()
        rows = []
        for seed in (seeds[:1] if name in SINGLE_SEED else seeds):
            rows += EXPERIMENTS[name](seed, quick=args.quick)
        write_csv(rows, out / f"{name}.csv")
        doc["experiments"][name] = {"rows": len(rows), "seconds": round(time.monotonic() - t0, 1)}
        report += summarize(name, rows)
        (out / "report.md").write_text("\n".join(report) + "\n")
        (out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{name}: {len(rows)} rows, {doc['experiments'][name]['seconds']} s", flush=True)
    return doc


def rebuild_report(args: argparse.Namespace) -> None:
    out = Path(args.output)
    report = ["# E14 LG-A (algebra, CPU) and LG-C CPU parts", "",
              "Means over seeds. Pre-registration `experiments/e14-logic/preregistration.md` §3; plan §3, §5.", ""]
    for name in EXPERIMENTS:
        path = out / f"{name}.csv"
        if not path.exists():
            continue
        rows = []
        for r in csv.DictReader(path.open()):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v) if v not in ("", None) and k not in {"seed"} and not v.lower() in {"true", "false"} else v
                except ValueError:
                    row[k] = v
            rows.append(row)
        report += summarize(name, rows)
    (out / "report.md").write_text("\n".join(report) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("report"); rp.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True); r.add_argument("--experiments", default=None)
    r.add_argument("--seeds", default="1,2,3"); r.add_argument("--quick", action="store_true"); r.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "report":
        rebuild_report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
