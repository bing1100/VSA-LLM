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

Addendum 8 (pre-registration §19):
- `x8b` (`run --experiments x8b`; not part of the default list): irregular and sparse trees (`gw`, `deep-sparse`, `spine`,
  depths 8–32) under X8's layouts plus `preorder-concat` (pre-order blocks, a side array of subtree sizes), `preorder-vsa`
  (kind and skip offset in the blocks) and `hybrid` (heap for depths 0–4, a node memory per subtree below).
- `x7b` (its own command): X7's decode for one trained store as wide as the three-store ensemble (C5w768), against the
  ensemble and the single 256-wide stores, with the pre-registered contrasts (bootstrap over entries).

    python -m vsa_embed.experiments.e14_concat run --output experiments/e14-logic/runs/lga-concat [--experiments x1,x2] [--seeds 1,2,3]
    python -m vsa_embed.experiments.e14_concat run --experiments x8b --output experiments/e14-logic/runs/lga-concat-x8b
    python -m vsa_embed.experiments.e14_concat x7b --output experiments/e14-logic/runs/lga-concat-x7b [--wide RUN]
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


# -- X7b: one trained store as wide as the three-store ensemble (pre-registration §19, addendum 8) ---------------------------

X7_RELATIONS = ("is_a", "has_role", "is_conjugate_acid_of", "is_enantiomer_of")
X7B_WIDE = Path("experiments/e14-logic/runs/x7b/SmolLM2-360M-full-C5w768-s1")
X7B_EXTRA = {f"C5w512-s{s}": Path(f"experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5w512-s{s}") for s in (1, 2, 3)}


def x7_slots(ontology: dict[str, Any], seed: int, *, quick: bool = False) -> dict[str, tuple[list[int], list[int], list[set]]]:
    """X7's slots, drawn exactly as `x7` draws them: per subset (held-out, seen) the (relation, entry, gold fillers) of the
    sampled entries."""
    offsets, relations, fillers = (np.asarray(ontology[k]) for k in ("offsets", "relations", "fillers"))
    held = {int(e) for e in ontology.get("heldout_entries", ())}
    freq = np.asarray(ontology.get("train_frequency") or np.zeros(len(offsets) - 1))
    rng = random.Random(f"x7:{seed}")
    pools = {"held-out": sorted(held), "seen": [e for e in np.flatnonzero(freq >= 10).tolist() if e not in held]}
    out = {}
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
        out[subset] = (q_rel, q_ent, gold)
    return out


def slot_hits(scores: torch.Tensor, gold: Sequence[set]) -> np.ndarray:
    """Per slot: the share of its gold fillers among the top-m scores (m = the number of gold fillers), as X7."""
    return np.array([len(set(torch.topk(scores[row], len(atoms)).indices.tolist()) & atoms) / len(atoms) for row, atoms in enumerate(gold)])


def cluster_bootstrap(diffs: np.ndarray, clusters: Sequence[int], *, resamples: int = 4000, seed: int = 0) -> dict[str, Any]:
    """Mean slot difference with a bootstrap over clusters (entries): resampled entries, their slots' mean; two-sided p as
    `e5_zeroshot.paired_difference`."""
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    sums, counts = np.zeros(len(ids)), np.zeros(len(ids))
    for d, c in zip(diffs, clusters):
        sums[pos[c]] += d; counts[pos[c]] += 1
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(resamples, len(ids)))
    draws = sums[pick].sum(1) / counts[pick].sum(1)
    p = 1.0 if np.all(diffs == 0) else float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (resamples + 1)))
    return {"mean": float(np.mean(diffs)), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "n_slots": int(len(diffs)), "n_entries": len(ids), "p_value": p}


def final_eval(run: Path, strata: Sequence[str] = ("all", "after", "after_heldout")) -> dict[str, float]:
    """The last evaluation's losses by stratum (`metrics.jsonl`)."""
    path = Path(run) / "metrics.jsonl"
    if not path.exists():
        return {}
    evals = [json.loads(l) for l in path.read_text().splitlines() if '"eval"' in l]
    if not evals:
        return {}
    last = max(r["step"] for r in evals)
    return {r["stratum"]: float(r["loss"]) for r in evals if r["step"] == last and r["stratum"] in strata}


def init_store(reference_run: Path, dimension: int, seed: int, label: str):
    """An untrained composer of `reference_run`'s channel (its resolved config and ontology) at width `dimension`, initialised
    under `torch.manual_seed(seed)`: the decode of the store before any training (X7b's control)."""
    import yaml
    from .. import self_query as sq
    from ..compose import FrameComposer, FrameSchedule
    from ..training import lm
    from .e12_self_query import LoadedStore
    config = lm.resolve_config(yaml.safe_load((Path(reference_run) / "resolved_config.yaml").read_text()))
    settings = config["channel"]
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    relations = int(ontology["relation_count"])
    schedule = lm.frame_variant(FrameSchedule(ontology["offsets"], ontology["relations"], ontology["fillers"]), settings["frames"],
                                relations, seed=int(config["seed"]))
    window = int(settings["context_window"])
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        composer = FrameComposer(schedule, int(ontology["atomic_count"]), relations, int(dimension), operator=settings["operator"],
                                 mode=settings["composition"], concept_factor=settings["concept_factor"],
                                 key_dimension=int(settings["key_dimension"]),
                                 context_dimension=int(settings["key_dimension"]) if window else 0)
    return LoadedStore(label, Path(reference_run), sq.RecallStore(composer.eval()), ontology, composer.operator,
                       {n: i for i, n in enumerate(ontology["relation_names"])}, {n: i for i, n in enumerate(ontology["atomic_names"])})


def x7b(*, wide: Path = X7B_WIDE, wide_label: str = "C5w768", track: str = "t4", seed: int = 1, quick: bool = False,
        extra: dict[str, Path] | None = None, resamples: int = 4000, reference: Path | None = Path("experiments/e14-logic/runs/lga-concat/x7.csv"),
        base_runs: Sequence[Path] | None = None, init_widths: Sequence[int] = (256, 768)) -> dict[str, Any]:
    """X7's decode for the three 256-wide C5 stores, their ensemble, the 768-wide store (`wide`), any `extra` stores whose run
    exists, and untrained composers at `init_widths` (`init-<width>`, seed 1, the C5 s1 config); the pre-registered contrasts
    (§19)."""
    from ..statistics import holm_adjust
    from .e12_self_query import load_store
    runs = list(base_runs) if base_runs else [Path(f"experiments/e9-retrofit/runs/{track}/SmolLM2-360M-full-C5-s{s}") for s in (1, 2, 3)]
    labels = [f"s{i + 1}" for i in range(len(runs))]
    present_extra = {k: Path(v) for k, v in (extra if extra is not None else X7B_EXTRA).items() if (Path(v) / "final.pt").exists()}
    missing_extra = sorted(set(extra if extra is not None else X7B_EXTRA) - set(present_extra))
    stores = {lab: load_store(r, lab) for lab, r in zip(labels, runs)}
    stores[wide_label] = load_store(Path(wide), wide_label)
    for k, v in present_extra.items():
        stores[k] = load_store(v, k)
    for w in init_widths:
        stores[f"init-{w}"] = init_store(runs[0], w, 1, f"init-{w}")
    o = stores["s1"].ontology
    for lab, st in stores.items():                                      # every store must read the same ontology
        for key in ("offsets", "relations", "fillers"):
            if not np.array_equal(np.asarray(st.ontology[key]), np.asarray(o[key])):
                raise ValueError(f"{lab}: ontology {key} differs from s1's")
    names = o["relation_names"]
    widths = {lab: int(st.store.entry_vectors().shape[1]) for lab, st in stores.items()}
    slots = x7_slots(o, seed, quick=quick)
    rows, hits_by, contrasts = [], {}, {"primary": [], "secondary": []}
    for subset, (q_rel, q_ent, gold) in slots.items():
        rel_t = torch.tensor(q_rel)
        scores = {lab: st.store.filler_scores(rel_t, st.store.entry_vectors()[torch.tensor(q_ent)]) for lab, st in stores.items()}
        finite = lambda t: torch.where(torch.isfinite(t), t, torch.full_like(t, -1.0))  # noqa: E731
        mean = sum(finite(scores[lab]) for lab in labels) / len(labels)
        scores["ensemble-3"] = torch.where(torch.isfinite(scores["s1"]), mean, scores["s1"])
        hits = {lab: slot_hits(sc, gold) for lab, sc in scores.items()}
        hits["mean-single"] = np.mean([hits[lab] for lab in labels], axis=0)
        hits_by[subset] = hits
        for lab, h in hits.items():
            by_rel = defaultdict(list)
            for r, v in zip(q_rel, h):
                by_rel[names[r]].append(v)
            rows.append({"exp": "x7b", "seed": seed, "subset": subset, "store": lab, "width": widths.get(lab, 3 * 256 if lab == "ensemble-3" else ""),
                         "slot_accuracy": float(np.mean(h)), "slots": len(gold),
                         **{f"rel:{k}": float(np.mean(v)) for k, v in by_rel.items() if k in X7_RELATIONS}})
        kw = {"resamples": resamples, "seed": 0}
        contrasts["primary"].append({"subset": subset, "a": wide_label, "b": "ensemble-3",
                                     **cluster_bootstrap(hits[wide_label] - hits["ensemble-3"], q_ent, **kw)})
        for b in ("s1", "mean-single", *present_extra, *(f"init-{w}" for w in init_widths)):
            contrasts["secondary"].append({"subset": subset, "a": wide_label, "b": b, **cluster_bootstrap(hits[wide_label] - hits[b], q_ent, **kw)})
        for w in init_widths:                                           # what training does to decoding at each width
            trained = "mean-single" if w == 256 else (wide_label if w == widths.get(wide_label) else None)
            if trained and trained != wide_label:
                contrasts["secondary"].append({"subset": subset, "a": trained, "b": f"init-{w}",
                                               **cluster_bootstrap(hits[trained] - hits[f"init-{w}"], q_ent, **kw)})
        for rel in X7_RELATIONS:
            keep = [i for i, r in enumerate(q_rel) if names[r] == rel]
            if keep:
                for b in ("ensemble-3", "s1"):
                    contrasts["secondary"].append({"subset": subset, "relation": rel, "a": wide_label, "b": b,
                                                   **cluster_bootstrap((hits[wide_label] - hits[b])[keep], [q_ent[i] for i in keep], **kw)})
    for c, p in zip(contrasts["primary"], holm_adjust([c["p_value"] for c in contrasts["primary"]])):
        c["p_holm"] = p
    check = None
    if reference is not None and Path(reference).exists() and not quick and not base_runs:
        import csv
        ref = {(r["subset"], r["store"]): float(r["slot_accuracy"]) for r in csv.DictReader(Path(reference).open())}
        mine = {(r["subset"], r["store"]): r["slot_accuracy"] for r in rows}
        check = {f"{s}|{st}": {"x7": v, "x7b": mine.get((s, st)), "same": abs(v - mine.get((s, st), float("nan"))) < 1e-9}
                 for (s, st), v in ref.items()}
    lm = {lab: final_eval(r) for lab, r in [("s1", runs[0]), (wide_label, Path(wide)), *present_extra.items()]}
    return {"rows": rows, "contrasts": contrasts, "x7_reproduced": check, "widths": widths, "language_model_loss": lm,
            "extra_present": sorted(present_extra), "extra_missing": missing_extra, "wide": str(wide), "seed": seed, "quick": quick}


def run_x7b(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    extra = None if args.extra is None else {Path(p).name: Path(p) for p in args.extra}
    doc = x7b(wide=args.wide, wide_label=args.wide_label, seed=args.seed, quick=args.quick, extra=extra, resamples=args.resamples,
              base_runs=args.base_runs, init_widths=tuple(int(w) for w in args.init_widths.split(",") if w))
    A.write_csv(doc["rows"], out / "x7b.csv")
    (out / "x7b.json").write_text(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1) + "\n")
    f = lambda c: f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] (slots {c['n_slots']}, entries {c['n_entries']}" + \
        (f", Holm p {c['p_holm']:.4f})" if "p_holm" in c else f", p {c['p_value']:.4f})")  # noqa: E731
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [
        "# E14 X7b — one trained store as wide as the three-store ensemble (pre-registration §19)", "",
        f"Wide store: `{doc['wide']}` (width {doc['widths'].get(args.wide_label)}). Slots as X7 (seed {doc['seed']}); per-slot decode "
        "accuracy, top-m against the gold fillers; bootstrap over entries.", "",
        "| subset | store | width | slot accuracy | " + " | ".join(X7_RELATIONS) + " |", "|---|---|---:|---:|" + "---:|" * len(X7_RELATIONS)]
    for r in doc["rows"]:
        lines.append(f"| {r['subset']} | {r['store']} | {r['width']} | {r['slot_accuracy']:.3f} | "
                     + " | ".join(f"{r[f'rel:{k}']:.3f}" if f"rel:{k}" in r else "—" for k in X7_RELATIONS) + " |")
    lines += ["", f"**Primary: {args.wide_label} − ensemble-3 (Holm over the 2 subsets)**", ""]
    lines += [f"- {c['subset']}: {f(c)}" for c in doc["contrasts"]["primary"]]
    lines += ["", "Secondary:", ""]
    lines += [f"- {c['subset']}{' · ' + c['relation'] if 'relation' in c else ''}: {c['a']} − {c['b']} {f(c)}" for c in doc["contrasts"]["secondary"]]
    if doc["x7_reproduced"] is not None:
        same = all(v["same"] for v in doc["x7_reproduced"].values())
        lines += ["", f"X7's rows reproduced exactly: {'yes' if same else 'NO'} ({len(doc['x7_reproduced'])} rows)."]
    lines += ["", "Language-model loss at the last evaluation (descriptive): "
              + "; ".join(f"{lab}: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(d.items())) for lab, d in doc["language_model_loss"].items() if d)]
    if doc["extra_missing"]:
        lines += ["", f"Extra stores not present (skipped): {', '.join(doc['extra_missing'])}."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return doc


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


# -- X8b: irregular and sparse trees (pre-registration §19, addendum 8) -------------------------------------------------------

X8B_FAMILIES = ("gw", "deep-sparse", "spine")
X8B_DEPTHS = (8, 12, 16, 24, 32)
X8B_TOP = 4                                                       # deep-sparse: complete to depth 4; hybrid: heap for depths 0–4
X8B_MAX_INTERNAL = 3000                                           # trees above this many internal nodes are redrawn


def x8b_tree(rng: random.Random, family: str, depth: int, *, n_feat: int = 20, n_val: int = 8, n_cls: int = 10) -> dict[str, Any]:
    """One binary decision tree of maximum depth exactly `depth`; nodes {"f", "v", "yes", "no"} or {"leaf"}."""
    def leaf() -> dict[str, Any]:
        return {"leaf": rng.randrange(n_cls)}

    def node(yes: dict[str, Any], no: dict[str, Any]) -> dict[str, Any]:
        return {"f": rng.randrange(n_feat), "v": rng.randrange(n_val), "yes": yes, "no": no}

    if family == "spine":
        def spine(level: int) -> dict[str, Any]:
            if level == depth - 1:
                return node(leaf(), leaf())
            if rng.random() < 0.5:
                return node(spine(level + 1), leaf())
            return node(leaf(), spine(level + 1))
        return spine(0)
    q = 0.55 if family == "gw" else 0.5
    top = 1 if family == "gw" else X8B_TOP
    for _ in range(100_000):
        count = [0]

        def grow(level: int) -> dict[str, Any]:
            if level < depth and (level < top or rng.random() < q) and count[0] < X8B_MAX_INTERNAL:
                count[0] += 1
                f, v = rng.randrange(n_feat), rng.randrange(n_val)
                return {"f": f, "v": v, "yes": grow(level + 1), "no": grow(level + 1)}
            return leaf()
        tree = grow(0)
        if count[0] < X8B_MAX_INTERNAL and x8b_shape(tree)["depth"] == depth:
            return tree
    raise RuntimeError(f"no {family} tree of depth {depth}")


def x8b_shape(tree: dict[str, Any]) -> dict[str, int]:
    out = {"depth": 0, "internal": 0, "leaves": 0}

    def walk(n: dict[str, Any], level: int) -> None:
        out["depth"] = max(out["depth"], level)
        if "leaf" in n:
            out["leaves"] += 1
            return
        out["internal"] += 1
        walk(n["yes"], level + 1); walk(n["no"], level + 1)
    walk(tree, 0)
    return out


def x8b_inputs(rng: random.Random, tree: dict[str, Any], n: int, *, n_feat: int = 20, n_val: int = 8
               ) -> tuple[list[list[int]], list[int], list[int]]:
    """`n` records, each built to follow the path to a uniformly drawn reachable leaf (constraints "f is v" on yes-steps,
    "f is not v" on no-steps; the other features at random); returns records, true classes and path lengths."""
    leaves: list[tuple[dict[int, int], dict[int, set], int, int]] = []

    def walk(nd: dict[str, Any], eq: dict[int, int], neq: dict[int, set], length: int) -> None:
        if "leaf" in nd:
            leaves.append((eq, neq, nd["leaf"], length))
            return
        f, v = nd["f"], nd["v"]
        if eq.get(f, v) == v and v not in neq.get(f, set()):                          # the yes-branch is consistent
            walk(nd["yes"], {**eq, f: v}, neq, length + 1)
        if eq.get(f) != v and len(neq.get(f, set()) | {v}) < n_val:                   # the no-branch is consistent
            walk(nd["no"], eq, {**neq, f: neq.get(f, set()) | {v}}, length + 1)
    walk(tree, {}, {}, 0)
    records, classes, lengths = [], [], []
    for _ in range(n):
        eq, neq, cls, length = leaves[rng.randrange(len(leaves))]
        x = [eq[f] if f in eq else rng.choice([v for v in range(n_val) if v not in neq.get(f, set())]) for f in range(n_feat)]
        nd = tree
        while "leaf" not in nd:
            nd = nd["yes"] if x[nd["f"]] == nd["v"] else nd["no"]
        assert nd["leaf"] == cls
        records.append(x); classes.append(cls); lengths.append(length)
    return records, classes, lengths


def _preorder(tree: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Pre-order node list, subtree sizes, and each node's depth."""
    nodes, sizes, levels = [], [], []

    def walk(n: dict[str, Any], level: int) -> int:
        i = len(nodes)
        nodes.append(n); sizes.append(1); levels.append(level)
        if "leaf" not in n:
            sizes[i] = 1 + walk(n["yes"], level + 1) + walk(n["no"], level + 1)
        return sizes[i]
    walk(tree, 0)
    return nodes, sizes, levels


def x8b(seed: int, *, quick: bool = False, depths: Sequence[int] = X8B_DEPTHS, families: Sequence[str] = X8B_FAMILIES,
        trees_per_cell: int = 5, inputs_per_tree: int = 100) -> list[dict[str, Any]]:
    """Irregular trees (`x8b_tree`) under X8's layouts and the pre-order layouts; accuracy of the leaf reached (batched over a
    tree's inputs). Pre-registration §19 (X8b)."""
    rows = []
    n_feat, n_val, n_cls = 20, 8, 10
    if quick:
        depths, trees_per_cell, inputs_per_tree = depths[:2], 2, 30

    def codes(width: int, g: torch.Generator) -> dict[str, torch.Tensor]:
        TEST, VAL, YES, NO, KIND, SKIP, CLS = unitary(7, width, g)
        INTERNAL, LEAF = gauss(2, width, g)
        return {"TEST": TEST, "VAL": VAL, "YES": YES, "NO": NO, "KIND": KIND, "SKIP": SKIP, "CLS": CLS,
                "kinds": torch.stack([INTERNAL, LEAF]), "feat": gauss(n_feat, width, g), "val": gauss(n_val, width, g),
                "cls": gauss(n_cls, width, g)}

    def cleanup(q: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        return cos_argmax(q, book)

    for fi_, family in enumerate(families):
        for depth in depths:
            rng = random.Random(f"x8b:{seed}:{family}:{depth}")
            trees = [x8b_tree(rng, family, depth) for _ in range(trees_per_cell)]
            data = [x8b_inputs(rng, t, inputs_per_tree) for t in trees]
            g_rec = torch.Generator().manual_seed(seed * 7 + depth + 50 * fi_ + 1000)
            rc = {"frole": unitary(n_feat, 4096, g_rec), "val": gauss(n_val, 4096, g_rec)}   # X8's input record, every layout

            def records(xs: list[list[int]]) -> torch.Tensor:
                X = torch.tensor(xs)
                return HRR.bind(rc["frole"].expand(len(xs), -1, -1), rc["val"][X]).sum(1)

            recs = [records(xs) for xs, _, _ in data]

            def tests(c: dict[str, torch.Tensor], blocks: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
                """X8's `test_step`, batched: the node's feature and value by clean-up, the record's value of that feature."""
                f = cleanup(HRR.unbind(blocks, c["TEST"].expand_as(blocks)), c["feat"])
                want = cleanup(HRR.unbind(blocks, c["VAL"].expand_as(blocks)), c["val"])
                have = cleanup(HRR.unbind(rec, rc["frole"][f]), rc["val"])
                return have == want

            layouts = [("nested", w) for w in (4096, 16384)] + [(lay, w) for lay in
                       ("node-memory", "heap-concat", "preorder-concat", "preorder-vsa", "hybrid") for w in (64, 256)]
            for layout, width in layouts:
                g = torch.Generator().manual_seed(seed * 1009 + depth * 31 + width + 7919 * (fi_ + 1) + 104_729)
                c = codes(width, g)
                hit = defaultdict(list)
                widths_, books = [], []
                for tree, (xs, truth, lengths), rec in zip(trees, data, recs):
                    nodes, sizes, levels = _preorder(tree)
                    B = len(xs)
                    test_of = lambda n: HRR.bind(c["TEST"], c["feat"][n["f"]]) + HRR.bind(c["VAL"], c["val"][n["v"]])  # noqa: E731
                    got = torch.full((B,), -1, dtype=torch.long)
                    if layout in ("heap-concat", "preorder-concat"):
                        if layout == "heap-concat":                  # node at heap index i; children 2i+1 / 2i+2
                            index, heap = {}, {}

                            def place(n: dict[str, Any], i: int) -> None:
                                heap[i] = n
                                if "leaf" not in n:
                                    place(n["yes"], 2 * i + 1); place(n["no"], 2 * i + 2)
                            place(tree, 0)
                            keys = sorted(heap)
                            index = {k: j for j, k in enumerate(keys)}   # storage row of each occupied heap slot
                            blocks = torch.stack([c["cls"][heap[k]["leaf"]] if "leaf" in heap[k] else test_of(heap[k]) for k in keys])
                            internal = lambda i: (2 * i + 1) in index  # noqa: E731
                            child = lambda i, yes: 2 * i + 1 if yes else 2 * i + 2  # noqa: E731
                            widths_.append((2 ** (depth + 1) - 1) * width); books.append(0)
                        else:                                        # pre-order; the sizes array holds the structure
                            index = {i: i for i in range(len(nodes))}
                            blocks = torch.stack([c["cls"][n["leaf"]] if "leaf" in n else test_of(n) for n in nodes])
                            internal = lambda i: sizes[i] > 1  # noqa: E731
                            child = lambda i, yes: i + 1 if yes else i + 1 + sizes[i + 1]  # noqa: E731
                            widths_.append(len(nodes) * width); books.append(0)
                        pos = [0] * B
                        active = list(range(B))
                        for _ in range(depth + 1):
                            inner = [k for k in active if internal(pos[k])]
                            for k in active:
                                if not internal(pos[k]):
                                    got[k] = int(cleanup(blocks[index[pos[k]]][None], c["cls"])[0])
                            if not inner:
                                break
                            ok = tests(c, blocks[[index[pos[k]] for k in inner]], rec[inner])
                            for k, y in zip(inner, ok.tolist()):
                                pos[k] = child(pos[k], y)
                            active = inner
                    elif layout == "preorder-vsa":                   # structure in the blocks: KIND and SKIP⊛P[size of yes]
                        n_nodes = len(nodes)
                        P = gauss(n_nodes, width, g)
                        blocks = torch.stack([HRR.bind(c["KIND"], c["kinds"][1]) + HRR.bind(c["CLS"], c["cls"][n["leaf"]]) if "leaf" in n else
                                              test_of(n) + HRR.bind(c["KIND"], c["kinds"][0]) + HRR.bind(c["SKIP"], P[sizes[i + 1] - 1])
                                              for i, n in enumerate(nodes)])
                        widths_.append(n_nodes * width); books.append(n_nodes * width)
                        pos = torch.zeros(B, dtype=torch.long)
                        active = torch.arange(B)
                        for _ in range(depth + 1):
                            if not len(active):
                                break
                            bl = blocks[pos[active]]
                            kind = cleanup(HRR.unbind(bl, c["KIND"].expand_as(bl)), c["kinds"])
                            leaf_k, inner = active[kind == 1], active[kind == 0]
                            if len(leaf_k):
                                lb = blocks[pos[leaf_k]]
                                got[leaf_k] = cleanup(HRR.unbind(lb, c["CLS"].expand_as(lb)), c["cls"])
                            if not len(inner):
                                break
                            ib = blocks[pos[inner]]
                            yes = tests(c, ib, rec[inner])
                            skip = cleanup(HRR.unbind(ib, c["SKIP"].expand_as(ib)), P) + 1
                            nxt = torch.where(yes, pos[inner] + 1, pos[inner] + 1 + skip)
                            bad = nxt >= n_nodes
                            got[inner[bad]] = -1
                            pos[inner[~bad]] = nxt[~bad]
                            active = inner[~bad]
                    elif layout in ("node-memory", "hybrid"):
                        # node memory: each internal node a block TEST + YES⊛child + NO⊛child, a child by its address (internal)
                        # or its class vector (leaf); clean-up over the memory's addresses and the classes.
                        def memory(root: dict[str, Any], skip_root: bool) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], torch.Tensor | None]:
                            inner_nodes = []

                            def collect(n: dict[str, Any]) -> None:
                                if "leaf" not in n:
                                    inner_nodes.append(n); collect(n["yes"]); collect(n["no"])
                            collect(root)
                            addr = gauss(len(inner_nodes), width, g)
                            idx = {id(n): j for j, n in enumerate(inner_nodes)}
                            ch = lambda n: addr[idx[id(n)]] if "leaf" not in n else c["cls"][n["leaf"]]  # noqa: E731
                            mem = torch.stack([test_of(n) + HRR.bind(c["YES"], ch(n["yes"])) + HRR.bind(c["NO"], ch(n["no"]))
                                               for n in inner_nodes])
                            return mem, torch.cat([addr, c["cls"]]), inner_nodes, addr

                        def walk_memory(mem: torch.Tensor, book: torch.Tensor, n_int: int, start: torch.Tensor, members: torch.Tensor,
                                        steps: int) -> None:
                            vec, act = start, members
                            for _ in range(steps):
                                if not len(act):
                                    return
                                yes = tests(c, vec, rec[act])
                                nxt = HRR.unbind(vec, torch.where(yes[:, None], c["YES"], c["NO"]))
                                j = cleanup(nxt, book)
                                is_leaf = j >= n_int
                                got[act[is_leaf]] = j[is_leaf] - n_int
                                act, vec = act[~is_leaf], mem[j[~is_leaf]]
                        if layout == "node-memory":
                            mem, book, inner_nodes, _ = memory(tree, False)
                            widths_.append(len(inner_nodes) * width); books.append(len(inner_nodes) * width)
                            walk_memory(mem, book, len(inner_nodes), mem[0].expand(B, -1), torch.arange(B), depth)
                        else:                                       # heap for depths 0..TOP, a node memory per subtree below
                            heap = {}

                            def place(n: dict[str, Any], i: int, level: int) -> None:
                                heap[i] = (n, level)
                                if "leaf" not in n and level < X8B_TOP:
                                    place(n["yes"], 2 * i + 1, level + 1); place(n["no"], 2 * i + 2, level + 1)
                            place(tree, 0, 0)
                            subs, total, book_total = {}, (2 ** (X8B_TOP + 1) - 1) * width, 0
                            for i, (n, level) in heap.items():
                                if level == X8B_TOP and "leaf" not in n:
                                    mem, book, inner_nodes, _ = memory(n, True)
                                    subs[i] = (mem, book, len(inner_nodes))
                                    total += (len(inner_nodes) - 1) * width     # the subtree root's block is the heap slot
                                    book_total += len(inner_nodes) * width
                            widths_.append(total); books.append(book_total)
                            pos = [0] * B
                            active = list(range(B))
                            for _ in range(X8B_TOP + 1):
                                inner = [k for k in active if "leaf" not in heap[pos[k]][0] and heap[pos[k]][1] < X8B_TOP]
                                for k in active:
                                    if "leaf" in heap[pos[k]][0]:
                                        got[k] = int(cleanup(c["cls"][heap[pos[k]][0]["leaf"]][None], c["cls"])[0])
                                if not inner:
                                    break
                                ok = tests(c, torch.stack([test_of(heap[pos[k]][0]) for k in inner]), rec[inner])
                                for k, y in zip(inner, ok.tolist()):
                                    pos[k] = 2 * pos[k] + (1 if y else 2)
                                active = inner
                            for i, (mem, book, n_int) in subs.items():
                                members = torch.tensor([k for k in range(B) if pos[k] == i and int(got[k]) == -1], dtype=torch.long)
                                if len(members):
                                    walk_memory(mem, book, n_int, mem[0].expand(len(members), -1), members, depth - X8B_TOP)
                    else:                                            # nested: one vector; a KIND marker tells internal from leaf
                        def enc(n: dict[str, Any]) -> torch.Tensor:
                            if "leaf" in n:
                                return F.normalize(HRR.bind(c["KIND"], c["kinds"][1]) + HRR.bind(c["CLS"], c["cls"][n["leaf"]]), dim=0)
                            t = test_of(n) + HRR.bind(c["KIND"], c["kinds"][0]) + HRR.bind(c["YES"], enc(n["yes"])) + HRR.bind(c["NO"], enc(n["no"]))
                            return F.normalize(t, dim=0)
                        tree_vec = enc(tree)
                        widths_.append(width); books.append(0)
                        vec, act = tree_vec.expand(B, -1), torch.arange(B)
                        for _ in range(depth + 1):
                            if not len(act):
                                break
                            kind = cleanup(HRR.unbind(vec, c["KIND"].expand_as(vec)), c["kinds"])
                            lf = kind == 1
                            if lf.any():
                                lv = vec[lf]
                                got[act[lf]] = cleanup(HRR.unbind(lv, c["CLS"].expand_as(lv)), c["cls"])
                            act, vec = act[~lf], vec[~lf]
                            if not len(act):
                                break
                            yes = tests(c, vec, rec[act])
                            vec = HRR.unbind(vec, torch.where(yes[:, None], c["YES"], c["NO"]))
                    correct = (got == torch.tensor(truth)).tolist()
                    for ok, length in zip(correct, lengths):
                        hit["all"].append(ok)
                        hit["path 1-4" if length <= 4 else "path 5-8" if length <= 8 else "path 9-16" if length <= 16 else "path 17+"].append(ok)
                shapes = [x8b_shape(t) for t in trees]
                rows.append({"exp": "x8b", "seed": seed, "family": family, "depth": depth, "layout": layout, "block_width": width,
                             "internal_nodes": float(np.mean([s["internal"] for s in shapes])),
                             "nodes": float(np.mean([s["internal"] + s["leaves"] for s in shapes])),
                             "total_width": float(np.mean(widths_)), "codebook_width": float(np.mean(books)),
                             "accuracy": float(np.mean(hit["all"])),
                             **{f"acc_{k.replace(' ', '_')}": float(np.mean(v)) for k, v in hit.items() if k != "all"},
                             **{f"n_{k.replace(' ', '_')}": len(v) for k, v in hit.items() if k != "all"}})
    return rows


def x8b_report(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Means over seeds: accuracy by family × depth × layout (block width), and total widths."""
    by: dict[tuple, list] = defaultdict(list)
    for r in rows:
        by[(r["family"], int(float(r["depth"])), r["layout"], int(float(r["block_width"])))].append(r)
    fams = sorted({k[0] for k in by}, key=lambda f: X8B_FAMILIES.index(f) if f in X8B_FAMILIES else 9)
    cols = [("nested", 4096), ("nested", 16384)] + [(lay, w) for lay in ("node-memory", "hybrid", "preorder-vsa", "preorder-concat",
                                                                          "heap-concat") for w in (64, 256)]
    mean = lambda rs, k: float(np.mean([float(r[k]) for r in rs if r.get(k) not in (None, "")])) if any(r.get(k) not in (None, "") for r in rs) else float("nan")  # noqa: E731
    lines = ["## x8b — irregular trees (accuracy of the leaf reached; means over seeds)", ""]
    for fam in fams:
        depths = sorted({k[1] for k in by if k[0] == fam})
        lines += [f"### {fam}", "", "| depth | internal nodes | " + " | ".join(f"{lay} {w}" for lay, w in cols) + " |",
                  "|---:|---:|" + "---:|" * len(cols)]
        for d in depths:
            cells = [by.get((fam, d, lay, w), []) for lay, w in cols]
            n_int = mean(next(c for c in cells if c), "internal_nodes")
            lines.append(f"| {d} | {n_int:.0f} | " + " | ".join(f"{mean(c, 'accuracy'):.3f}" if c else "—" for c in cells) + " |")
        lines += ["", "Total width (block layouts at b = 64; × 4 at b = 256):", "",
                  "| depth | node-memory | hybrid | preorder | heap (nominal) |", "|---:|---:|---:|---:|---:|"]
        for d in depths:
            w = {lay: mean(by.get((fam, d, lay, 64), []), "total_width") for lay in ("node-memory", "hybrid", "preorder-concat", "heap-concat")}
            lines.append(f"| {d} | {w['node-memory']:,.0f} | {w['hybrid']:,.0f} | {w['preorder-concat']:,.0f} | {w['heap-concat']:.3g} |")
        lines.append("")
    return lines


EXPERIMENTS: dict[str, Callable[..., list[dict[str, Any]]]] = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6, "x7": x7, "x8": x8}
# Addendum 8 (pre-registration §19): run only when named (`--experiments x8b`), so the default `run` is unchanged.
EXTRA_EXPERIMENTS: dict[str, Callable[..., list[dict[str, Any]]]] = {"x8b": x8b}
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
            rows += {**EXPERIMENTS, **EXTRA_EXPERIMENTS}[name](seed, quick=args.quick)
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
    if (out / "x8b.csv").exists():                                # addendum 8 (pre-registration §19)
        import csv
        report += ["X8b: pre-registration §19 (addendum 8).", ""] + x8b_report(list(csv.DictReader((out / "x8b.csv").open())))
    (out / "report.md").write_text("\n".join(report) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True); r.add_argument("--experiments", default=None)
    r.add_argument("--seeds", default="1,2,3"); r.add_argument("--quick", action="store_true"); r.add_argument("--threads", type=int, default=4)
    b = sub.add_parser("x7b", help="addendum 8 (pre-registration §19): X7's decode for a store as wide as the ensemble")
    b.add_argument("--output", type=Path, required=True); b.add_argument("--wide", type=Path, default=X7B_WIDE)
    b.add_argument("--wide-label", default="C5w768"); b.add_argument("--seed", type=int, default=1)
    b.add_argument("--extra", nargs="*", default=None, help="other run folders to decode if present (default: E9's C5w512 s1-s3)")
    b.add_argument("--base-runs", nargs="*", type=Path, default=None, help="the 256-wide stores (default: T4 C5 s1-s3)")
    b.add_argument("--resamples", type=int, default=4000); b.add_argument("--quick", action="store_true")
    b.add_argument("--init-widths", default="256,768", help="untrained composers decoded as controls ('' for none)")
    b.add_argument("--smoke", action="store_true"); b.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    (run_x7b if args.command == "x7b" else run)(args)


if __name__ == "__main__":
    main()
