"""E14 addendum 1 (pre-registration §10): concatenate instead of superpose (the HRRCat principle of the author's
HRRBERT paper), and higher widths — the LG-A tests rerun on the CPU.

Layouts, compared with one wide HRR of the same total width D:
- `wide` — one HRR; every binding superposed.
- `cat-redundant` — k = 4 independent HRR codes of width D/4 with the same content; cosine scores averaged over blocks.
- `cat-role` — a fact as [predicate | arg1 | arg2 | arg3] raw fillers, one block per role; a rule is a block-circulant map
  estimated by per-frequency least squares over the blocks (it can permute blocks and transform one block alone).
- `cat-fact` — a frame as the concatenation of its 8 facts (one block each; inside a block `wide`, or `cat-role` for
  `cat-fact-role`); a rule applies blockwise.
- `cat-slot` — one block per relation (decision 61c's slotted layout); `cat-slot8` — 8 load-balanced relation groups.
- `ensemble-3` — the three trained T4 C5 stores (s1–s3), cosine scores averaged.

Experiments (`run --experiments`): `x1` rules, `x2` chains, `x3` joins, `x4` nested decode and trees at high width,
`x5` negation and numbers at high width, `x6` resonator at high width, `x7` trained-store ensemble.

    python -m vsa_embed.experiments.e14_concat run --output experiments/e14-logic/runs/lga-concat [--experiments x1,x2] [--seeds 1,2,3]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..algebra import HRRAlgebra, UnitaryHRRAlgebra
from ..relations import balanced_slots
from . import e14_algebra as A

HRR = HRRAlgebra()
WIDTHS = (1024, 4096, 8192, 16384)


def unitary(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    return UnitaryHRRAlgebra.make_role(torch.randn(n, d, generator=g))


def gauss(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    return torch.randn(n, d, generator=g) / math.sqrt(d)


def cos_argmax(q: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return (F.normalize(q, dim=-1) @ F.normalize(book, dim=-1).T).argmax(-1)


# -- X1: rules under each layout ------------------------------------------------------------------------------------------

def block_lsq(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Per-frequency least squares for a block-circulant map: X, Y are (k, blocks, width) examples; returns M of shape
    (freq, blocks_out, blocks_in) with Y_j(ω) ≈ Σ_i M_ji(ω) X_i(ω)."""
    Xf, Yf = torch.fft.rfft(X), torch.fft.rfft(Y)                    # (k, b, f)
    Xf, Yf = Xf.permute(2, 0, 1), Yf.permute(2, 0, 1)                # (f, k, b)
    XtX = Xf.conj().transpose(1, 2) @ Xf + ridge * torch.eye(Xf.shape[-1], dtype=Xf.dtype)
    XtY = Xf.conj().transpose(1, 2) @ Yf
    return torch.linalg.solve(XtX, XtY).transpose(1, 2)              # (f, b_out, b_in)


def block_apply(M: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """X (n, blocks, width) → (n, blocks, width) through the block-circulant map M (freq, b_out, b_in)."""
    width = X.shape[-1]
    Xf = torch.fft.rfft(X).permute(2, 0, 1)                          # (f, n, b)
    Yf = Xf @ M.transpose(1, 2)                                      # (f, n, b_out)
    return torch.fft.irfft(Yf.permute(1, 2, 0), n=width)


KINDS = ("inverse", "symmetric", "subrelation", "rotate3")


def conclusion_roles(kind: str) -> tuple[int, tuple[int, int, int]]:
    """(predicate index of the conclusion: 1 = q, 0 = p) and the argument order of the conclusion over (a, b, c)."""
    return {"inverse": (1, (1, 0, 2)), "symmetric": (0, (1, 0, 2)), "subrelation": (1, (0, 1, 2)), "rotate3": (1, (2, 0, 1))}[kind]


def x1(seed: int, *, quick: bool = False, widths: Sequence[int] = WIDTHS, ks: Sequence[int] = (3, 10, 30)) -> list[dict[str, Any]]:
    rows = []
    n_ent, n_test = 2000, 300
    for D in (widths if not quick else widths[:2]):
        for kind in KINDS:
            g = torch.Generator().manual_seed(seed * 9973 + D + KINDS.index(kind))
            q_idx, order = conclusion_roles(kind)
            three = kind == "rotate3"
            rng = torch.Generator().manual_seed(seed * 31 + D)
            perm = torch.randperm(n_ent, generator=rng)
            train, test = perm[: n_ent // 2], perm[n_ent // 2:]

            def triples(ids, n):
                pick = ids[torch.randint(len(ids), (n, 3), generator=rng)]
                return pick[:, 0], pick[:, 1], pick[:, 2]
            ta, tb, tc = triples(test, n_test)
            gold_args = [x for x in (ta, tb, tc)]
            gold = [gold_args[order[0]], gold_args[order[1]]]       # the conclusion's ARG1 and ARG2
            for layout in ("wide", "cat-redundant", "cat-role"):
                blocks = {"wide": 1, "cat-redundant": 4, "cat-role": 4}[layout]
                bw = D // blocks
                if layout == "cat-role":
                    ent = gauss(n_ent, bw, g); preds = gauss(20, bw, g)
                    def fact(pr, a, b, c):
                        return torch.stack([preds[pr].expand(len(a), -1), ent[a], ent[b], ent[c] if three else torch.zeros(len(a), bw)], 1)
                    def concl(a, b, c):
                        args = (a, b, c)
                        return fact(q_idx, args[order[0]], args[order[1]], args[order[2]] if three else c)
                    def decode(out):
                        p_ok = cos_argmax(out[:, 0], preds) == q_idx
                        return p_ok & (cos_argmax(out[:, 1], ent) == gold[0]) & (cos_argmax(out[:, 2], ent) == gold[1])
                    # hand-written block map: predicate block p→q (q ⊛ p⁻¹), argument blocks permuted
                    hand = torch.zeros(bw // 2 + 1, 4, 4, dtype=torch.complex64)
                    pf, qf = torch.fft.rfft(preds[0]), torch.fft.rfft(preds[q_idx])
                    hand[:, 0, 0] = qf / pf
                    src = {0: 1, 1: 2, 2: 3}
                    for out_slot, arg in enumerate(order[: 3 if three else 2]):
                        hand[:, 1 + out_slot, src[arg]] = 1.0
                    estimators = {"hand": hand}
                    for k in ks:
                        a, b, c = triples(train, k)
                        estimators[f"lsq-k{k}"] = block_lsq(fact(0, a, b, c), concl(a, b, c))
                    test_in = fact(0, ta, tb, tc)
                    for name, M in estimators.items():
                        rows.append({"exp": "x1", "seed": seed, "width": D, "layout": layout, "kind": kind, "estimator": name,
                                     "applied_to": "fact", "accuracy": float(decode(block_apply(M, test_in)).float().mean())})
                    continue
                # wide and cat-redundant: per block an independent HRR fact code PRED⊛p + ARG1⊛a + ARG2⊛b (+ ARG3⊛c)
                codes = []
                for _ in range(blocks):
                    codes.append({"ent": gauss(n_ent, bw, g), "preds": unitary(20, bw, g), "roles": unitary(4, bw, g)})
                def hfact(cd, pr, a, b, c):
                    P, R1, R2, R3 = cd["roles"]
                    v = HRR.bind(P, cd["preds"][pr]) + HRR.bind(R1, cd["ent"][a]) + HRR.bind(R2, cd["ent"][b])
                    return v + HRR.bind(R3, cd["ent"][c]) if three else v
                def hconcl(cd, a, b, c):
                    args = (a, b, c)
                    return hfact(cd, q_idx, args[order[0]], args[order[1]], args[order[2]] if three else c)
                for k in ks:
                    a, b, c = triples(train, k)
                    scores = {"p": 0, "a1": 0, "a2": 0}
                    for cd in codes:
                        x, y = hfact(cd, 0, a, b, c), hconcl(cd, a, b, c)
                        X, Y = torch.fft.rfft(x), torch.fft.rfft(y)
                        M = torch.fft.irfft((Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3), n=bw)
                        out = HRR.bind(M.expand(n_test, -1), hfact(cd, 0, ta, tb, tc))
                        P, R1, R2, _ = cd["roles"]
                        cos = lambda v, book: F.normalize(v, dim=-1) @ F.normalize(book, dim=-1).T  # noqa: E731
                        scores["p"] = scores["p"] + cos(HRR.unbind(out, P.expand_as(out)), cd["preds"])
                        scores["a1"] = scores["a1"] + cos(HRR.unbind(out, R1.expand_as(out)), cd["ent"])
                        scores["a2"] = scores["a2"] + cos(HRR.unbind(out, R2.expand_as(out)), cd["ent"])
                    ok = (scores["p"].argmax(-1) == q_idx) & (scores["a1"].argmax(-1) == gold[0]) & (scores["a2"].argmax(-1) == gold[1])
                    rows.append({"exp": "x1", "seed": seed, "width": D, "layout": layout, "kind": kind, "estimator": f"lsq-k{k}",
                                 "applied_to": "fact", "accuracy": float(ok.float().mean())})
            # frames of 8 facts: wide bundle against one block per fact (inside: wide HRR, or cat-role)
            for layout in ("wide", "cat-fact", "cat-fact-role"):
                k = 10
                others_p = torch.randint(2, 20, (n_test, 7), generator=rng)
                oa, ob, oc = [ids[torch.randint(n_ent, (n_test, 7), generator=rng)] for ids in (torch.arange(n_ent),) * 3]
                if layout == "wide":
                    cd = {"ent": gauss(n_ent, D, g), "preds": unitary(20, D, g), "roles": unitary(4, D, g)}
                    P, R1, R2, R3 = cd["roles"]
                    def hf(pr, a, b, c):
                        v = HRR.bind(P, cd["preds"][pr]) + HRR.bind(R1, cd["ent"][a]) + HRR.bind(R2, cd["ent"][b])
                        return v + HRR.bind(R3, cd["ent"][c]) if three else v
                    a, b, c = triples(train, k)
                    args = (a, b, c)
                    x, y = hf(torch.zeros(k, dtype=torch.long), a, b, c), hf(torch.full((k,), q_idx), args[order[0]], args[order[1]], args[order[2]] if three else c)
                    X, Y = torch.fft.rfft(x), torch.fft.rfft(y)
                    M = torch.fft.irfft((Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3), n=D)
                    frame = hf(torch.zeros(n_test, dtype=torch.long), ta, tb, tc) + sum(hf(others_p[:, j], oa[:, j], ob[:, j], oc[:, j]) for j in range(7))
                    out = HRR.bind(M.expand(n_test, -1), frame)
                    ok = (cos_argmax(HRR.unbind(out, P.expand_as(out)), cd["preds"]) == q_idx) & \
                         (cos_argmax(HRR.unbind(out, R1.expand_as(out)), cd["ent"]) == gold[0]) & \
                         (cos_argmax(HRR.unbind(out, R2.expand_as(out)), cd["ent"]) == gold[1])
                else:
                    bw = D // 8                                      # the target fact sits in block 0; the rule applies blockwise
                    if layout == "cat-fact":
                        cd = {"ent": gauss(n_ent, bw, g), "preds": unitary(20, bw, g), "roles": unitary(4, bw, g)}
                        P, R1, R2, R3 = cd["roles"]
                        def hf(pr, a, b, c):
                            v = HRR.bind(P, cd["preds"][pr]) + HRR.bind(R1, cd["ent"][a]) + HRR.bind(R2, cd["ent"][b])
                            return v + HRR.bind(R3, cd["ent"][c]) if three else v
                        a, b, c = triples(train, k)
                        args = (a, b, c)
                        x, y = hf(torch.zeros(k, dtype=torch.long), a, b, c), hf(torch.full((k,), q_idx), args[order[0]], args[order[1]], args[order[2]] if three else c)
                        X, Y = torch.fft.rfft(x), torch.fft.rfft(y)
                        M = torch.fft.irfft((Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3), n=bw)
                        out = HRR.bind(M.expand(n_test, -1), hf(torch.zeros(n_test, dtype=torch.long), ta, tb, tc))
                        ok = (cos_argmax(HRR.unbind(out, P.expand_as(out)), cd["preds"]) == q_idx) & \
                             (cos_argmax(HRR.unbind(out, R1.expand_as(out)), cd["ent"]) == gold[0]) & \
                             (cos_argmax(HRR.unbind(out, R2.expand_as(out)), cd["ent"]) == gold[1])
                    else:
                        rb = bw // 4
                        ent, preds = gauss(n_ent, rb, g), gauss(20, rb, g)
                        def rf(pr, a, b, c):
                            pv = preds[pr] if pr.dim() else preds[pr].expand(len(a), -1)
                            return torch.stack([pv, ent[a], ent[b], ent[c] if three else torch.zeros(len(a), rb)], 1)
                        a, b, c = triples(train, k)
                        args = (a, b, c)
                        M = block_lsq(rf(torch.zeros(k, dtype=torch.long), a, b, c),
                                      rf(torch.full((k,), q_idx), args[order[0]], args[order[1]], args[order[2]] if three else c))
                        out = block_apply(M, rf(torch.zeros(n_test, dtype=torch.long), ta, tb, tc))
                        ok = (cos_argmax(out[:, 0], preds) == q_idx) & (cos_argmax(out[:, 1], ent) == gold[0]) & (cos_argmax(out[:, 2], ent) == gold[1])
                rows.append({"exp": "x1", "seed": seed, "width": D, "layout": layout, "kind": kind, "estimator": f"lsq-k{k}",
                             "applied_to": "frame8", "accuracy": float(ok.float().mean())})
    # trained fillers: cat-role blocks at the trained rows' native width (256 each) against wide 256 (A1)
    pool = A.trained_rows()
    tw = int(pool.shape[1])
    g = torch.Generator().manual_seed(seed * 7 + 1)
    idx = torch.randperm(pool.shape[0], generator=g)[:n_ent]
    ent = pool[idx] / pool[idx].norm(dim=-1, keepdim=True)
    preds = gauss(20, tw, g)
    rng = torch.Generator().manual_seed(seed * 11)
    perm = torch.randperm(n_ent, generator=rng)
    train, test = perm[: n_ent // 2], perm[n_ent // 2:]
    for kind in KINDS:
        q_idx, order = conclusion_roles(kind)
        three = kind == "rotate3"
        def rf(pr, a, b, c):
            return torch.stack([preds[pr], ent[a], ent[b], ent[c] if three else torch.zeros(len(a), tw)], 1)
        pick = lambda ids, n: [ids[torch.randint(len(ids), (n,), generator=rng)] for _ in range(3)]  # noqa: E731
        ta, tb, tc = pick(test, n_test)
        gold_args = (ta, tb, tc)
        for k in ks:
            a, b, c = pick(train, k)
            args = (a, b, c)
            M = block_lsq(rf(torch.zeros(k, dtype=torch.long), a, b, c),
                          rf(torch.full((k,), q_idx), args[order[0]], args[order[1]], args[order[2]] if three else c))
            out = block_apply(M, rf(torch.zeros(n_test, dtype=torch.long), ta, tb, tc))
            ok = (cos_argmax(out[:, 0], preds) == q_idx) & (cos_argmax(out[:, 1], ent) == gold_args[order[0]]) & \
                 (cos_argmax(out[:, 2], ent) == gold_args[order[1]])
            rows.append({"exp": "x1", "seed": seed, "width": 4 * tw, "layout": "cat-role-trained", "kind": kind, "estimator": f"lsq-k{k}",
                         "applied_to": "fact", "accuracy": float(ok.float().mean())})
    return rows


# -- X2: chains under each layout (vectorised in the frequency domain) -------------------------------------------------------

def x2(seed: int, *, quick: bool = False, widths: Sequence[int] = WIDTHS, loads: Sequence[int] = (16, 32, 64)) -> list[dict[str, Any]]:
    rows = []
    n_ent, n_rel, n_chains, depth = (4000, 24, 200, 8) if not quick else (1500, 24, 60, 6)
    for load in (loads if not quick else loads[:1]):
        for profile in ("fixed", "heavy"):
            rng = random.Random(f"x2:{seed}:{load}:{profile}")
            perm = list(range(n_ent))
            rng.shuffle(perm)
            chains = [perm[i * (depth + 1):(i + 1) * (depth + 1)] for i in range(n_chains)]
            nxt = {c[i]: c[i + 1] for c in chains for i in range(depth)}
            heavy_m = max(1, load // 4)
            edges = []                                              # (entity, relation, filler)
            heavy_truth = {}
            for e in range(n_ent):
                L = load if profile == "fixed" else max(2, min(4 * load, int(round(load * 0.5 * rng.paretovariate(1.5)))))
                m = max(1, L // 4)
                hv = [rng.randrange(n_ent) for _ in range(m)]
                heavy_truth[e] = set(hv)
                edges += [(e, 1, o) for o in hv]
                if e in nxt:
                    edges.append((e, 0, nxt[e]))
                rest = L - m - (1 if e in nxt else 0)
                edges += [(e, rng.randrange(2, n_rel), rng.randrange(n_ent)) for _ in range(max(0, rest))]
            E = torch.tensor(edges, dtype=torch.long)
            probe = rng.sample(range(n_ent), 300)
            starts = torch.tensor([c[0] for c in chains])
            targets = [torch.tensor([c[h] for c in chains]) for h in range(1, depth + 1)]
            counts = [0] * n_rel
            for _, r, _ in edges:
                counts[r] += 1
            slot8 = balanced_slots(counts, 8)
            for D in (widths if not quick else widths[:2]):
                for layout in ("wide", "cat-redundant", "cat-slot", "cat-slot8"):
                    g = torch.Generator().manual_seed(seed * 101 + D + load)
                    if layout == "wide":
                        blocks = [(D, list(range(n_rel)))]
                    elif layout == "cat-redundant":
                        blocks = [(D // 4, list(range(n_rel)))] * 4
                    elif layout == "cat-slot":
                        blocks = [(D // n_rel, [r]) for r in range(n_rel)]
                    else:
                        blocks = [(D // 8, [r for r in range(n_rel) if slot8[r] == s]) for s in range(8)]
                    hop_scores: list[list[torch.Tensor]] = []
                    chain_cur = starts.clone()
                    alive = torch.ones(n_chains, dtype=torch.bool)
                    acc = {}
                    # build every block's frames once
                    built = []
                    for bw, rels in blocks:
                        ent = gauss(n_ent, bw, g)
                        roles = unitary(n_rel, bw, g)
                        mask = torch.isin(E[:, 1], torch.tensor(rels))
                        sub = E[mask]
                        ef, rf = torch.fft.rfft(ent), torch.fft.rfft(roles)
                        frames_f = torch.zeros(n_ent, bw // 2 + 1, dtype=torch.complex64)
                        for start in range(0, sub.shape[0], 8192):
                            part = sub[start:start + 8192]
                            frames_f.index_add_(0, part[:, 0], rf[part[:, 1]] * ef[part[:, 2]])
                        built.append((bw, set(rels), ent, roles, frames_f))

                    def scores_for(entities: torch.Tensor, relation: int) -> torch.Tensor:
                        total, used = 0, 0
                        for bw, rels, ent, roles, frames_f in built:
                            if relation not in rels:
                                continue
                            q = torch.fft.irfft(frames_f[entities] * torch.fft.rfft(roles[relation]).conj(), n=bw)
                            total = total + F.normalize(q, dim=-1) @ F.normalize(ent, dim=-1).T
                            used += 1
                        return total / max(1, used)
                    for h in range(depth):
                        chain_cur = scores_for(chain_cur, 0).argmax(-1)
                        alive &= chain_cur == targets[h]
                        acc[f"hop{h + 1}"] = float(alive.float().mean())
                    hs = scores_for(torch.tensor(probe), 1)
                    rec = []
                    for row, e in enumerate(probe):
                        m = len(heavy_truth[e])
                        top = set(torch.topk(hs[row], m).indices.tolist())
                        rec.append(len(top & heavy_truth[e]) / m)
                    rows.append({"exp": "x2", "seed": seed, "width": D, "load": load, "profile": profile, "layout": layout, **acc,
                                 "heavy_relation_recall": float(np.mean(rec))})
    return rows


# -- X3: joins, wide against one block per relation ----------------------------------------------------------------------------

def world_scores(facts: dict[str, set], *, width: int, seed: int, layout: str, n_ent: int = 1000, n_base: int = 8) -> dict[str, torch.Tensor]:
    """Per base relation the (entities × entities) clean-up cosines after unbinding: `wide` (one HRR frame per entity) or
    `cat-slot` (relation r's bindings in block r only, width // n_base)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    if layout == "wide":
        ent, rel = gauss(n_ent, width, g), unitary(n_base, width, g)
        ef, rf = torch.fft.rfft(ent), torch.fft.rfft(rel)
        frames_f = torch.zeros(n_ent, width // 2 + 1, dtype=torch.complex64)
        for r, pairs in facts.items():
            if pairs:
                P = torch.tensor(sorted(pairs))
                frames_f.index_add_(0, P[:, 0], rf[int(r[1:])] * ef[P[:, 1]])
        keys = F.normalize(ent, dim=-1)
        for ri in range(n_base):
            q = torch.fft.irfft(frames_f * rf[ri].conj(), n=width)
            out[f"b{ri}"] = F.normalize(q, dim=-1) @ keys.T
        return out
    bw = width // n_base
    for ri in range(n_base):
        ent, role = gauss(n_ent, bw, g), unitary(1, bw, g)[0]
        ef, rf = torch.fft.rfft(ent), torch.fft.rfft(role)
        frames_f = torch.zeros(n_ent, bw // 2 + 1, dtype=torch.complex64)
        pairs = facts.get(f"b{ri}", set())
        if pairs:
            P = torch.tensor(sorted(pairs))
            frames_f.index_add_(0, P[:, 0], rf * ef[P[:, 1]])
        q = torch.fft.irfft(frames_f * rf.conj(), n=bw)
        out[f"b{ri}"] = F.normalize(q, dim=-1) @ F.normalize(ent, dim=-1).T
    return out


def threshold_facts(scores: dict[str, torch.Tensor], t: float) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for r, m in scores.items():
        s_idx, o_idx = torch.nonzero(m >= t, as_tuple=True)
        out[r] = set(zip(s_idx.tolist(), o_idx.tolist()))
    return out


def x3(seed: int, *, quick: bool = False, widths: Sequence[int] = (256, 1024, 4096), loads: Sequence[int] = (8, 16)) -> list[dict[str, Any]]:
    rows = []
    for load in loads:
        facts = A.a3_world(seed, load=load)
        calib = A.a3_world(seed + 10_000, load=load)
        masks = {}
        for r in (f"b{i}" for i in range(8)):                         # the calibration world's true edges as masks
            m = torch.zeros(1000, 1000, dtype=torch.bool)
            if calib.get(r):
                P = torch.tensor(sorted(calib[r]))
                m[P[:, 0], P[:, 1]] = True
            masks[r] = m
        n_true = sum(int(m.sum()) for m in masks.values())
        for width in (widths if not quick else widths[:1]):
            for layout in ("wide", "cat-slot"):
                cs = world_scores(calib, width=width, seed=seed + 10_000, layout=layout)
                best, best_f1 = 0.2, -1.0                            # threshold calibrated on a separate world, per layout
                for t in np.arange(0.06, 0.9, 0.02):
                    tp = sum(int(((cs[r] >= t) & masks[r]).sum()) for r in masks)
                    pred = sum(int((cs[r] >= t).sum()) for r in masks)
                    p_, r_ = tp / max(1, pred), tp / max(1, n_true)
                    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
                    if f1 > best_f1:
                        best, best_f1 = float(t), f1
                dec = threshold_facts(world_scores(facts, width=width, seed=seed, layout=layout), best)
                _, _, bf = A.prf({(r, e) for r in dec for e in dec[r]}, {(r, e) for r in facts for e in facts[r]})
                for shape in A.SHAPES:
                    rules = A.a3_rules(shape, 20, 8, random.Random(f"a3r:{seed}:{shape}:20"))
                    gold, got = A.apply_rules(facts, rules), A.apply_rules(dec, rules)
                    heads = sorted({r["head"] for r in rules})
                    p, r, f1 = A.prf({(h, e) for h in heads for e in got.get(h, set())}, {(h, e) for h in heads for e in gold.get(h, set())})
                    rows.append({"exp": "x3", "seed": seed, "width": width, "load": load, "layout": layout, "shape": shape,
                                 "threshold": best, "base_f1": bf, "precision": p, "recall": r, "f1": f1})
    return rows


# -- X4–X6: the earlier tests at higher widths ---------------------------------------------------------------------------------

def x4(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = [{**r, "exp": "x4-a4"} for r in A.a4(seed, quick=quick, widths=(256, 4096) if quick else (4096, 8192), depths=(2, 4, 6, 8))
            if r["test"] == "nested-decode"]
    rows += [{**r, "exp": "x4-a7"} for r in A.a7(seed, quick=quick, widths=(4096,) if quick else (4096, 8192, 16384), noises=(0.0,),
                                                  numerics=(False,)) if r["kind"] == "tree"]
    return rows


def x5(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    rows = [{**r, "exp": "x5-a5"} for r in A.a5(seed, quick=quick, widths=(4096, 8192), loads=(32, 64))]
    rows += [{**r, "exp": "x5-a6"} for r in A.a6(seed, quick=quick, widths=(4096, 8192), level_set=(100, 1000))]
    return rows


def x6(seed: int, *, quick: bool = False) -> list[dict[str, Any]]:
    return [{**r, "exp": "x6-a8"} for r in A.a8(seed, quick=quick, widths=(2048, 4096, 8192) if not quick else (2048,), sizes=(200, 1000))]


# -- X7: the three trained T4 stores combined --------------------------------------------------------------------------------

def x7(seed: int, *, quick: bool = False, track: str = "t4") -> list[dict[str, Any]]:
    from .e12_self_query import load_store
    runs = [Path(f"experiments/e9-retrofit/runs/{track}/SmolLM2-360M-full-C5-s{s}") for s in (1, 2, 3)]
    stores = [load_store(r, f"s{i + 1}") for i, r in enumerate(runs)]
    o = stores[0].ontology
    offsets, relations, fillers = np.asarray(o["offsets"]), np.asarray(o["relations"]), np.asarray(o["fillers"])
    held = {int(e) for e in o.get("heldout_entries", ())}
    freq = np.asarray(o.get("train_frequency") or np.zeros(len(offsets) - 1))
    rng = random.Random(f"x7:{seed}")
    pools = {"held-out": sorted(held), "seen": [e for e in np.flatnonzero(freq >= 10).tolist() if e not in held]}
    vectors = [s.store.entry_vectors() for s in stores]
    rows = []
    for subset, pool in pools.items():
        sample = rng.sample(pool, min(len(pool), 600 if not quick else 80))
        q_rel, q_ent, gold = [], [], []
        for e in sample:
            lo, hi = int(offsets[e]), int(offsets[e + 1])
            by = defaultdict(set)
            for r, a in zip(relations[lo:hi].tolist(), fillers[lo:hi].tolist()):
                by[r].add(a)
            for r, atoms in by.items():
                q_rel.append(r); q_ent.append(e); gold.append(atoms)
        rel_t = torch.tensor(q_rel)
        per_store = [s.store.filler_scores(rel_t, vectors[i][torch.tensor(q_ent)]) for i, s in enumerate(stores)]
        names = o["relation_names"]

        def accuracy(scores: torch.Tensor) -> tuple[float, dict[str, float]]:
            hits, by_rel = [], defaultdict(list)
            for row, atoms in enumerate(gold):
                top = set(torch.topk(scores[row], len(atoms)).indices.tolist())
                v = len(top & atoms) / len(atoms)
                hits.append(v); by_rel[names[q_rel[row]]].append(v)
            return float(np.mean(hits)), {k: float(np.mean(v)) for k, v in by_rel.items()}
        for i, sc in enumerate(per_store):
            a, by_rel = accuracy(sc)
            rows.append({"exp": "x7", "seed": seed, "subset": subset, "store": f"s{i + 1}", "slot_accuracy": a, "slots": len(gold),
                         **{f"rel:{k}": v for k, v in by_rel.items() if k in {"is_a", "has_role", "is_conjugate_acid_of", "is_enantiomer_of"}}})
        finite = lambda t: torch.where(torch.isfinite(t), t, torch.full_like(t, -1.0))  # noqa: E731
        mean = sum(finite(sc) for sc in per_store) / 3
        mean = torch.where(torch.isfinite(per_store[0]), mean, per_store[0])
        a, by_rel = accuracy(mean)
        rows.append({"exp": "x7", "seed": seed, "subset": subset, "store": "ensemble-3", "slot_accuracy": a, "slots": len(gold),
                     **{f"rel:{k}": v for k, v in by_rel.items() if k in {"is_a", "has_role", "is_conjugate_acid_of", "is_enantiomer_of"}}})
    return rows


# -- X8: tree layouts chosen by the tree's structure -------------------------------------------------------------------------

def x8(seed: int, *, quick: bool = False, depths: Sequence[int] = (2, 4, 6, 8, 10, 12)) -> list[dict[str, Any]]:
    """Complete binary trees stored (a) in one nested vector, (b) in a node memory (associative addresses, clean-up per
    step), (c) heap-concatenated (node i in block i, children 2i+1 / 2i+2: positional), (d) level-concatenated (one block
    per depth, that level's nodes superposed under position keys). Accuracy of the leaf reached, and total width."""
    rows = []
    n_feat, n_val, n_cls = 20, 8, 10
    for depth in (depths if not quick else depths[:3]):
        rng = random.Random(f"x8:{seed}:{depth}")
        trees = []
        for _ in range(5 if not quick else 2):
            n_int = 2 ** depth - 1
            f = [rng.randrange(n_feat) for _ in range(n_int)]
            v = [rng.randrange(n_val) for _ in range(n_int)]
            leaf = [rng.randrange(n_cls) for _ in range(2 ** depth)]
            inputs = [[rng.randrange(n_val) for _ in range(n_feat)] for _ in range(100 if not quick else 30)]
            truth = []
            for x in inputs:
                i = 0
                while i < n_int:
                    i = 2 * i + 1 if x[f[i]] == v[i] else 2 * i + 2
                truth.append(leaf[i - n_int])
            trees.append((f, v, leaf, inputs, truth))
        n_nodes = 2 ** (depth + 1) - 1

        def codes(width: int, g: torch.Generator) -> dict[str, torch.Tensor]:
            TEST, VAL, YES, NO = unitary(4, width, g)
            return {"TEST": TEST, "VAL": VAL, "YES": YES, "NO": NO, "feat": gauss(n_feat, width, g), "frole": unitary(n_feat, width, g),
                    "val": gauss(n_val, width, g), "cls": gauss(n_cls, width, g)}

        rc = codes(4096, torch.Generator().manual_seed(seed * 7 + depth))   # the input record: same encoding for every layout

        def record(c, x):
            return HRR.bind(rc["frole"], rc["val"][torch.tensor(x)]).sum(0)

        def test_step(c, node_vec, rec):
            fi = int(cos_argmax(HRR.unbind(node_vec, c["TEST"])[None], c["feat"])[0])
            want = int(cos_argmax(HRR.unbind(node_vec, c["VAL"])[None], c["val"])[0])
            have = int(cos_argmax(HRR.unbind(rec, rc["frole"][fi])[None], rc["val"])[0])
            return have == want

        for layout, widths in (("nested", (4096, 16384)), ("node-memory", (64, 256)), ("heap-concat", (64, 256)), ("level-concat", (1024, 4096))):
            if layout == "nested" and depth > 10:
                continue                                              # 2^depth recursive bindings at width D: impractical and at chance
            for width in widths:
                g = torch.Generator().manual_seed(seed * 1009 + depth * 31 + width)
                c = codes(width, g)
                correct = total = 0
                for f, v, leaf, inputs, truth in trees:
                    n_int = len(f)
                    node_test = lambda i: HRR.bind(c["TEST"], c["feat"][f[i]]) + HRR.bind(c["VAL"], c["val"][v[i]])  # noqa: E731
                    if layout == "heap-concat":
                        blocks = torch.stack([node_test(i) if i < n_int else c["cls"][leaf[i - n_int]] for i in range(n_nodes)])
                    elif layout == "node-memory":
                        addr = gauss(n_nodes, width, g)
                        child = lambda j: addr[j] if j < n_int else c["cls"][leaf[j - n_int]]  # noqa: E731
                        mem = torch.stack([node_test(i) + HRR.bind(c["YES"], child(2 * i + 1)) + HRR.bind(c["NO"], child(2 * i + 2))
                                           for i in range(n_int)])
                        book = torch.cat([addr[:n_int], c["cls"]])
                    elif layout == "level-concat":
                        pos = {lv: unitary(2 ** lv, width, g) for lv in range(depth + 1)}
                        levels = []
                        for lv in range(depth + 1):
                            first = 2 ** lv - 1
                            items = [node_test(first + j) if lv < depth else c["cls"][leaf[j]] for j in range(2 ** lv)]
                            levels.append(HRR.bind(pos[lv], torch.stack(items)).sum(0))
                    else:
                        def enc(i):
                            if i >= n_int:
                                return F.normalize(c["cls"][leaf[i - n_int]], dim=0)
                            t = node_test(i) + HRR.bind(c["YES"], enc(2 * i + 1)) + HRR.bind(c["NO"], enc(2 * i + 2))
                            return F.normalize(t, dim=0)
                        tree_vec = enc(0)
                    for x, want_leaf in zip(inputs, truth):
                        rec = record(c, x)
                        if layout == "heap-concat":
                            i = 0
                            while i < n_int:
                                i = 2 * i + 1 if test_step(c, blocks[i], rec) else 2 * i + 2
                            got = int(cos_argmax(blocks[i][None], c["cls"])[0])
                        elif layout == "node-memory":
                            vec, got = mem[0], None
                            for _ in range(depth):
                                nxt = HRR.unbind(vec, c["YES"] if test_step(c, vec, rec) else c["NO"])
                                j = int(cos_argmax(nxt[None], book)[0])
                                if j >= n_int:
                                    got = j - n_int
                                    break
                                vec = mem[j]
                            got = -1 if got is None else got
                        elif layout == "level-concat":
                            j = 0
                            for lv in range(depth):
                                node = HRR.unbind(levels[lv], pos[lv][j])
                                j = 2 * j + (0 if test_step(c, node, rec) else 1)
                            got = int(cos_argmax(HRR.unbind(levels[depth], pos[depth][j])[None], c["cls"])[0])
                        else:
                            vec = tree_vec
                            for _ in range(depth):
                                vec = HRR.unbind(vec, c["YES"] if test_step(c, vec, rec) else c["NO"])
                            got = int(cos_argmax(vec[None], c["cls"])[0])
                        correct += got == want_leaf
                        total += 1
                total_width = {"nested": width, "node-memory": width * (2 ** depth - 1), "heap-concat": width * n_nodes,
                               "level-concat": width * (depth + 1)}[layout]
                rows.append({"exp": "x8", "seed": seed, "depth": depth, "layout": layout, "block_width": width,
                             "total_width": total_width, "accuracy": correct / total})
    return rows


EXPERIMENTS: dict[str, Callable[..., list[dict[str, Any]]]] = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6, "x7": x7, "x8": x8}
SINGLE_SEED = {"x7"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    names = [n.strip() for n in args.experiments.split(",")] if args.experiments else list(EXPERIMENTS)
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"experiments": {}, "seeds": seeds, "quick": bool(args.quick)}
    for name in names:
        t0 = time.monotonic()
        rows = []
        for seed in (seeds[:1] if name in SINGLE_SEED else seeds):
            rows += EXPERIMENTS[name](seed, quick=args.quick)
        A.write_csv(rows, out / f"{name}.csv")
        doc["experiments"][name] = {"rows": len(rows), "seconds": round(time.monotonic() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{name}: {len(rows)} rows, {doc['experiments'][name]['seconds']} s", flush=True)
    report = ["# E14 addendum 1 — concatenation layouts and higher widths (CPU)" + (" — QUICK SMOKE" if args.quick else ""), "",
              "Means over seeds. Pre-registration `experiments/e14-logic/preregistration.md` §10.", ""]
    for name in EXPERIMENTS:
        path = out / f"{name}.csv"
        if path.exists():
            import csv
            rows = []
            for r in csv.DictReader(path.open()):
                row = {}
                for k, v in r.items():
                    try:
                        row[k] = float(v) if v not in ("", None) and k != "seed" and v.lower() not in {"true", "false"} else v
                    except ValueError:
                        row[k] = v
                rows.append(row)
            report += A.summarize(name, rows)
    (out / "report.md").write_text("\n".join(report) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True); r.add_argument("--experiments", default=None)
    r.add_argument("--seeds", default="1,2,3"); r.add_argument("--quick", action="store_true"); r.add_argument("--threads", type=int, default=4)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
