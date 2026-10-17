"""Illustration for the VSA predicate-logic proposal (2026-10-17): can the repo's HRR algebra execute rules?

Not a pre-registered experiment and not a result for the paper: a CPU sanity check of the algebra that the
proposal (`2026-10-17-vsa-predicate-logic.md`) builds on, with random vectors only (no language model, no data).

- A. A rule as one mapping vector, applied to entities never seen when it was formed: the inverse rule
     parent(a, b) => child(b, a) on facts PRED*p + ARG1*a + ARG2*b. The mapping is written by hand (role swap) or
     estimated from k example pairs (mean correlation, Plate/Kanerva; or per-frequency least squares, Neumann).
- B. A transitive rule executed by chaining: located_in is unbound from an entity's frame (L edges bundled),
     snapped to the nearest entity (clean-up), that entity's frame is fetched, and so on for d hops.
- C. A decision tree executed by unbinding: each node is TEST*feature + VAL*value + YES*child + NO*child.
     (i) node memory: children are separate stored vectors, cleaned up after every step;
     (ii) one nested vector: the whole tree in a single vector, no clean-up until the leaf.

Run from VSA-LLM/:  PYTHONPATH=src python manuscript/proposals/vsa_logic_demo.py --out manuscript/proposals/vsa_logic_demo.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.nn import functional as F

from vsa_embed.algebra import HRRAlgebra, UnitaryHRRAlgebra

ALG = HRRAlgebra()


def fillers(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    """Plate's HRR fillers: i.i.d. N(0, 1/d) elements."""
    return torch.randn(n, d, generator=g) / math.sqrt(d)


def roles(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    """Unitary roles (unit-magnitude spectrum), so unbinding by correlation is the exact inverse."""
    return UnitaryHRRAlgebra.make_role(torch.randn(n, d, generator=g)) * 1.0


def inverse(v: torch.Tensor) -> torch.Tensor:
    """Exact inverse in the Fourier domain (ridge-free; fine for unitary vectors)."""
    d = v.shape[-1]
    return torch.fft.irfft(1.0 / torch.fft.rfft(v), n=d)


def cleanup(q: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return (F.normalize(q, dim=-1) @ F.normalize(book, dim=-1).T).argmax(-1)


# -- A. inverse rule as a mapping vector ------------------------------------------------------------------------

def demo_rule_mapping(d: int, seed: int, ks=(1, 3, 10, 30, 100, 300), n_entities=2000, n_preds=20, n_test=1000):
    g = torch.Generator().manual_seed(seed)
    ent = fillers(n_entities, d, g)
    preds = roles(n_preds, d, g)                      # predicate symbols (unitary, so child*parent^-1 is exact)
    PRED, ARG1, ARG2 = roles(3, d, g)
    parent, child = preds[0], preds[1]
    train_ids, test_ids = torch.arange(0, n_entities // 2), torch.arange(n_entities // 2, n_entities)

    def fact(p, a, b):
        return ALG.bind(PRED, p) + ALG.bind(ARG1, a) + ALG.bind(ARG2, b)

    def pairs(ids, n):
        perm = ids[torch.randperm(len(ids), generator=g)]
        a, b = perm[:n], perm[n:2 * n] if 2 * n <= len(perm) else perm[torch.randint(len(perm), (n,), generator=g)]
        return a, b

    ta, tb = pairs(test_ids, n_test // 2 if n_test > len(test_ids) // 2 else n_test)
    test_in = fact(parent, ent[ta], ent[tb])

    def score(M):
        out = ALG.bind(M.expand_as(test_in), test_in)
        ok_p = cleanup(ALG.unbind(out, PRED.expand_as(out)), preds) == 1
        ok_1 = cleanup(ALG.unbind(out, ARG1.expand_as(out)), ent) == tb
        ok_2 = cleanup(ALG.unbind(out, ARG2.expand_as(out)), ent) == ta
        return {"all": (ok_p & ok_1 & ok_2).float().mean().item(), "pred": ok_p.float().mean().item(),
                "arg1": ok_1.float().mean().item(), "arg2": ok_2.float().mean().item()}

    analytic = (ALG.bind(ARG2, inverse(ARG1)) + ALG.bind(ARG1, inverse(ARG2)) + ALG.bind(child, inverse(parent)))
    out = {"analytic": score(analytic), "learned_mean": {}, "learned_lsq": {}}
    for k in ks:
        a, b = pairs(train_ids, k)
        x, y = fact(parent, ent[a], ent[b]), fact(child, ent[b], ent[a])
        # Plate / Kanerva: mean of example correlations (approximate inverse = involution).
        out["learned_mean"][k] = score(ALG.unbind(y, x).mean(0))
        # Neumann-style: per-frequency least squares, M(w) = sum Y conj(X) / (sum |X|^2 + ridge).
        X, Y = torch.fft.rfft(x), torch.fft.rfft(y)
        Mw = (Y * X.conj()).sum(0) / ((X.abs() ** 2).sum(0) + 1e-3)
        out["learned_lsq"][k] = score(torch.fft.irfft(Mw, n=d))
    return out


# -- B. transitive rule by chained unbinding with clean-up ------------------------------------------------------

def demo_chain(d: int, seed: int, load: int, depth=6, n_entities=5000, n_relations=24, n_chains=300):
    g = torch.Generator().manual_seed(seed)
    ent = fillers(n_entities, d, g)
    rel = roles(n_relations, d, g)
    LOC = 0
    # chains e_0 -> e_1 -> ... -> e_depth under located_in; every entity also has load-1 other random edges
    nxt = torch.full((n_entities,), -1, dtype=torch.long)
    chains = torch.randperm(n_entities, generator=g)[: n_chains * (depth + 1)].view(n_chains, depth + 1)
    for c in chains:
        for i in range(depth):
            nxt[c[i]] = c[i + 1]
    frames = torch.zeros(n_entities, d)
    for e in range(n_entities):
        others = torch.randint(1, n_relations, (load - (1 if nxt[e] >= 0 else 0),), generator=g)
        partners = torch.randint(n_entities, (len(others),), generator=g)
        frames[e] = ALG.bind(rel[others], ent[partners]).sum(0)
        if nxt[e] >= 0:
            frames[e] += ALG.bind(rel[LOC], ent[nxt[e]])
    cur = chains[:, 0].clone()
    acc = {}
    alive = torch.ones(n_chains, dtype=torch.bool)
    for h in range(1, depth + 1):
        q = ALG.unbind(frames[cur], rel[LOC].expand(n_chains, d))
        cur = cleanup(q, ent)
        alive &= cur == chains[:, h]
        acc[h] = alive.float().mean().item()
    return acc


# -- C. decision tree execution -------------------------------------------------------------------------------

def demo_tree(d: int, seed: int, depth: int, n_features=20, n_values=8, n_classes=10, n_trees=10, n_inputs=200):
    g = torch.Generator().manual_seed(seed)
    TEST, VAL, YES, NO = roles(4, d, g)
    feat_id = fillers(n_features, d, g)               # the feature's name, stored at TEST
    feat_role = roles(n_features, d, g)               # the feature's role in an input record
    vals = fillers(n_values, d, g)
    classes = fillers(n_classes, d, g)
    res = {"node_memory": [], "nested_vector": []}
    for _ in range(n_trees):
        n_internal = 2 ** depth - 1
        f = torch.randint(n_features, (n_internal,), generator=g)
        v = torch.randint(n_values, (n_internal,), generator=g)
        leaf = torch.randint(n_classes, (2 ** depth,), generator=g)
        # (i) node memory: internal node i has children 2i+1 (yes), 2i+2 (no); leaves are class vectors
        node_key = fillers(n_internal, d, g)          # each node's address in the memory
        book = torch.cat([node_key, classes])         # clean-up over node addresses and class labels

        def child_vec(j):
            return node_key[j] if j < n_internal else classes[leaf[j - n_internal]]
        mem = torch.stack([ALG.bind(TEST, feat_id[f[i]]) + ALG.bind(VAL, vals[v[i]])
                           + ALG.bind(YES, child_vec(2 * i + 1)) + ALG.bind(NO, child_vec(2 * i + 2))
                           for i in range(n_internal)])

        # (ii) one nested vector, built bottom-up (normalized per level, as a bundle would be)
        def nested(i):
            if i >= n_internal:
                return F.normalize(classes[leaf[i - n_internal]], dim=-1)
            s = (ALG.bind(TEST, feat_id[f[i]]) + ALG.bind(VAL, vals[v[i]])
                 + ALG.bind(YES, nested(2 * i + 1)) + ALG.bind(NO, nested(2 * i + 2)))
            return F.normalize(s, dim=-1)
        tree_vec = nested(0)

        xs = torch.randint(n_values, (n_inputs, n_features), generator=g)
        records = ALG.bind(feat_role.expand(n_inputs, -1, -1), vals[xs]).sum(1)   # x = sum_f FEAT_f * value
        truth = []
        for x in xs:
            i = 0
            while i < n_internal:
                i = 2 * i + 1 if x[f[i]] == v[i] else 2 * i + 2
            truth.append(leaf[i - n_internal].item())
        truth = torch.tensor(truth)

        for kind in ("node_memory", "nested_vector"):
            correct = 0
            for row in range(n_inputs):
                vec = mem[0] if kind == "node_memory" else tree_vec
                for _ in range(depth):
                    fi = cleanup(ALG.unbind(vec, TEST), feat_id)
                    want = cleanup(ALG.unbind(vec, VAL), vals)
                    have = cleanup(ALG.unbind(records[row], feat_role[fi]), vals)
                    branch = YES if have == want else NO
                    nxt = ALG.unbind(vec, branch)
                    if kind == "node_memory":
                        j = cleanup(nxt, book).item()
                        vec = mem[j] if j < n_internal else classes[j - n_internal]
                    else:
                        vec = nxt                     # no clean-up: the next level is read from the noisy vector
                correct += int(cleanup(vec, classes).item() == truth[row].item())
            res[kind].append(correct / n_inputs)
    return {k: sum(x) / len(x) for k, x in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    seeds = list(range(1, args.seeds + 1))

    def mean(dicts):
        if isinstance(dicts[0], dict):
            return {k: mean([x[k] for x in dicts]) for k in dicts[0]}
        return sum(dicts) / len(dicts)

    out = {"A_rule_mapping": {}, "B_chain": {}, "C_tree": {}, "seeds": seeds}
    for d in (256, 1024, 4096):
        out["A_rule_mapping"][d] = mean([demo_rule_mapping(d, s) for s in seeds])
    for d in (256, 1024):
        for load in (4, 8, 16, 32):
            out["B_chain"][f"d{d}-load{load}"] = mean([demo_chain(d, s, load) for s in seeds])
    for d in (256, 1024):
        for depth in (1, 2, 3, 4, 6, 8):
            out["C_tree"][f"d{d}-depth{depth}"] = mean([demo_tree(d, s, depth, n_trees=4 if depth >= 6 else 10)
                                                        for s in seeds])
    out["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
