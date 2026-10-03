"""Summaries and the markdown report of an E10 run (reads metric rows only)."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import numpy as np

from vsa_embed.statistics import mean_confidence_interval, wilson_interval

COMPONENTS = ("atomics", "relations", "mapping", "frames")


def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _ci(values: list[float]) -> dict[str, Any]:
    values = [v for v in (_num(x) for x in values) if v == v]
    if not values:
        return {"mean": float("nan"), "ci_low": None, "ci_high": None, "n": 0}
    return mean_confidence_interval(values)


def _fmt(ci: dict[str, Any] | float | None, digits: int = 3) -> str:
    if ci is None:
        return "n/a"
    if isinstance(ci, (int, float)):
        return "n/a" if ci != ci else f"{ci:.{digits}f}"
    mean = ci.get("mean")
    if mean is None or mean != mean:
        return "n/a"
    if ci.get("ci_low") is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{ci['ci_low']:.{digits}f}, {ci['ci_high']:.{digits}f}]"


def _sig(ci: dict[str, Any]) -> bool:
    return ci.get("ci_low") is not None and ci["ci_low"] > 0


# -- part a -----------------------------------------------------------------------------------------

def summarize_a(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "a"]
    if not rows:
        return {}
    metrics = ("val_fit", "test_fit", "f1", "auc", "atomic_cosine", "relation_cosine", "spurious_removed",
               "spurious_mass_mean", "true_asserted_mass_mean")
    key = lambda r: tuple(r[c] for c in COMPONENTS)  # noqa: E731
    grid = defaultdict(lambda: defaultdict(list))
    by_seed = defaultdict(dict)
    for r in rows:
        for m in metrics:
            grid[key(r)][m].append(r.get(m))
        by_seed[key(r)][r["seed"]] = r
    table = {"|".join(k): {m: _ci(v) for m, v in d.items()} for k, d in grid.items()}

    def paired(a: tuple, b: tuple, metric: str) -> dict[str, Any]:
        seeds = sorted(set(by_seed[a]) & set(by_seed[b]))
        return _ci([_num(by_seed[a][s].get(metric)) - _num(by_seed[b][s].get(metric)) for s in seeds])

    base = ("l2", "l2", "l2", "l2")
    contrasts = {}
    for i, comp in enumerate(COMPONENTS):
        for setting in ("fixed", "free"):
            alt = tuple(setting if j == i else "l2" for j in range(4))
            contrasts[f"{comp}={setting} − all l2"] = {m: paired(alt, base, m) for m in ("val_fit", "test_fit", "f1", "auc")}
        only = tuple("l2" if j == i else "fixed" for j in range(4))
        contrasts[f"only {comp} l2 − all fixed"] = {m: paired(only, ("fixed",) * 4, m) for m in ("val_fit", "test_fit", "f1", "auc")}
    contrasts["all free − all l2"] = {m: paired(("free",) * 4, base, m) for m in ("val_fit", "test_fit", "f1", "auc")}
    contrasts["all fixed − all l2"] = {m: paired(("fixed",) * 4, base, m) for m in ("val_fit", "test_fit", "f1", "auc")}
    main = {}
    for i, comp in enumerate(COMPONENTS):
        for setting in ("fixed", "l2", "free"):
            sel = [r for r in rows if r[comp] == setting]
            main[f"{comp}={setting}"] = {m: float(np.nanmean([_num(r.get(m)) for r in sel])) for m in ("val_fit", "test_fit", "f1")}
    best = max(table, key=lambda k: table[k]["test_fit"]["mean"])
    return {"table": table, "contrasts": contrasts, "main_effects": main, "best_test_fit": best}


# -- part b -----------------------------------------------------------------------------------------

def summarize_b(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "b"]
    out = {}
    for rate in sorted({r["rate"] for r in rows}):
        sel = [r for r in rows if r["rate"] == rate]
        out[str(rate)] = {m: _ci([r.get(m) for r in sel]) for m in
                          ("precision", "recall", "f1", "auc", "r_precision", "prevalence", "frequency_auc",
                           "frequency_r_precision", "val_fit", "test_fit")}
        out[str(rate)]["auc_minus_random"] = _ci([_num(r.get("auc")) - 0.5 for r in sel])
        out[str(rate)]["precision_minus_prevalence"] = _ci([_num(r.get("precision")) - _num(r.get("prevalence")) for r in sel])
        out[str(rate)]["r_precision_minus_frequency"] = _ci([_num(r.get("r_precision")) - _num(r.get("frequency_r_precision")) for r in sel])
    return out


# -- part c (+ riddle, trajectories) -------------------------------------------------------------------

def summarize_c(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "c"]
    out: dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        methods = sorted({r["method"] for r in sel}, key=lambda m: ["oracle", "additive", "additive_norule", "all_at_once",
                                                                     "m3", "stem_cell"].index(m) if m in
                         ["oracle", "additive", "additive_norule", "all_at_once", "m3", "stem_cell"] else 99)
        table = {}
        by = {m: {r["seed"]: r for r in sel if r["method"] == m} for m in methods}
        for m in methods:
            rs = list(by[m].values())
            entry = {k: _ci([r.get(k) for r in rs]) for k in ("ari_gold_edges", "ari_all", "permuted_ari_mean",
                                                                "mean_best_jaccard", "detection_precision",
                                                                "detection_recall", "val_fit", "test_fit", "clusters",
                                                                "accepted_slots", "proposed_slots")}
            entry["ari_minus_permuted"] = _ci([_num(r.get("ari_gold_edges")) - _num(r.get("permuted_ari_mean")) for r in rs])
            entry["permuted_p_max"] = max((_num(r.get("permuted_ari_p")) for r in rs), default=float("nan"))
            from vsa_embed.statistics import holm_adjust
            pvals = [_num(r.get("permuted_ari_p")) for r in rs if r.get("permuted_ari_p") is not None]
            entry["permutation_holm_max_p"] = max(holm_adjust(pvals)) if pvals else float("nan")
            rels = defaultdict(list)
            for r in rs:
                for name, v in (r.get("per_relation") or {}).items():
                    rels[name].append(v["best_jaccard"])
            entry["per_relation_best_jaccard"] = {k: _ci(v) for k, v in rels.items()}
            rec = defaultdict(list)
            for r in rs:
                for name, v in (r.get("relation_recovery") or {}).items():
                    rec[name].append(v["jaccard"])
            entry["relation_recovery_jaccard"] = {k: _ci(v) for k, v in rec.items()}
            table[m] = entry

        def paired(a: str, b: str, metric: str) -> dict[str, Any]:
            seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
            return _ci([_num(by[a][s].get(metric)) - _num(by[b][s].get(metric)) for s in seeds])

        contrasts = {}
        for other in ("all_at_once", "m3", "stem_cell", "additive_norule"):
            if other in by and "additive" in by:
                contrasts[f"additive − {other}"] = {k: paired("additive", other, k)
                                                    for k in ("ari_gold_edges", "mean_best_jaccard", "test_fit")}
        if "additive" in by and "additive_norule" in by:
            def rec_mean(r):
                values = [v["jaccard"] for v in (r.get("relation_recovery") or {}).values()]
                return float(np.mean(values)) if values else float("nan")
            seeds = sorted(set(by["additive"]) & set(by["additive_norule"]))
            contrasts["rule enforcement: relation recovery (additive − additive_norule)"] = _ci(
                [rec_mean(by["additive"][s]) - rec_mean(by["additive_norule"][s]) for s in seeds])
        out[mode] = {"methods": table, "contrasts": contrasts, "riddle": summarize_riddle(sel),
                     "trajectories": summarize_trajectories(sel), "slot_decisions": summarize_slot_decisions(sel)}
    return out


def summarize_riddle(rows: list[dict], draws: int = 400, seed: int = 0) -> dict[str, Any]:
    entries = [e for r in rows if r["method"] in {"additive", "all_at_once"} for e in (r.get("riddle") or [])]
    accepted = [e for e in entries if e["accept"] and e.get("best_relation")]
    adopted = [e for e in accepted if e.get("adopted")]
    out = {"accepted_slots": len(accepted), "with_adopted_rule": len(adopted),
           "adopted_true": sum(bool(e.get("adopted_true")) for e in adopted),
           "adopted_designed": sum(bool(e.get("adopted_designed")) for e in adopted)}
    by_rel = defaultdict(lambda: defaultdict(int))
    for e in accepted:
        rel = e["best_relation"]
        by_rel[rel]["slots"] += 1
        by_rel[rel]["adopted"] += bool(e.get("adopted"))
        by_rel[rel]["true"] += bool(e.get("adopted_true"))
        by_rel[rel]["designed"] += bool(e.get("adopted_designed"))
        if e.get("adopted"):
            by_rel[rel][f"rule:{e.get('adopted_mapped', e['adopted'])}"] += 1
    out["by_relation"] = {k: dict(v) for k, v in by_rel.items()}
    precisions = [e["rule_prediction_precision"] for e in adopted if e.get("rule_prediction_precision") is not None]
    out["rule_prediction_precision"] = _ci(precisions)
    # accuracy vs number of candidate hypotheses: one true hypothesis + (m − 1) false ones, drawn from the
    # full scored hypothesis space of each slot; the adopted one is the passing hypothesis with the
    # largest lower bound (none adopted counts as a miss).
    rng = random.Random(seed)
    curve = {}
    pools = []
    for e in accepted:
        hyps = [h for h in e.get("hypotheses", []) if h["name"] != "unstructured" and h.get("lower") is not None]
        true = [h for h in hyps if h.get("designed")]          # the relation's designed (informative) property
        false = [h for h in hyps if not h.get("true")]
        if true and false:
            pools.append((true, false))
    for m in (1, 2, 4, 8, 16, 32):
        hits, n = 0, 0
        for true, false in pools:
            if m - 1 > len(false):
                continue
            for _ in range(max(1, draws // max(1, len(pools)))):
                subset = [rng.choice(true)] + rng.sample(false, m - 1)
                passing = [h for h in subset if h.get("passed")]
                best = max(passing, key=lambda h: (_num(h["lower"]), _num(h["mean"]))) if passing else None
                hits += bool(best and best.get("true")); n += 1
        curve[str(m)] = {"accuracy": hits / n if n else float("nan"), "draws": n,
                         "slots": sum(1 for _, f in pools if m - 1 <= len(f))}
    out["accuracy_vs_hypotheses"] = curve
    full = [len(e.get("hypotheses", [])) for e in accepted]
    out["hypotheses_per_slot"] = float(np.mean(full)) if full else float("nan")
    return out


def summarize_trajectories(rows: list[dict]) -> dict[str, Any]:
    """Overextension → refinement: early vs final purity, size and number of relations per slot, plus
    the soft-mass composition of each blank slot over its first steps."""
    soft_curve = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] != "additive":
            continue
        starts = {}
        for t in sorted(r.get("trajectory") or [], key=lambda t: t["step"]):
            starts.setdefault(t["slot"], t["step"])
            rel = t["step"] - starts[t["slot"]]
            if t.get("target") and t.get("soft_target_share") is not None and t["soft_target_share"] == t["soft_target_share"]:
                for key in ("soft_target_share", "soft_other_relation_share", "soft_distractor_share", "soft_effective_size"):
                    soft_curve[rel][key].append(t[key])
    soft = {str(step): {k: float(np.mean(v)) for k, v in d.items()} for step, d in sorted(soft_curve.items())
            if step in (0, 25, 50, 100, 200, 400, 700)}
    early_purity, final_purity, early_rel, final_rel, early_size, final_size, peak_size = [], [], [], [], [], [], []
    for r in rows:
        if r["method"] != "additive":
            continue
        by_slot = defaultdict(list)
        for t in r.get("trajectory") or []:
            by_slot[t["slot"]].append(t)
        for slot, traj in by_slot.items():
            traj = [t for t in sorted(traj, key=lambda t: t["step"]) if t["members"] > 0 and t["purity"] is not None]
            if len(traj) < 2 or not traj[-1]["target"]:
                continue
            early_purity.append(traj[0]["purity"]); final_purity.append(traj[-1]["purity"])
            early_rel.append(traj[0]["relations_present"]); final_rel.append(traj[-1]["relations_present"])
            early_size.append(traj[0]["members"]); final_size.append(traj[-1]["members"])
            peak_size.append(max(t["members"] for t in traj))
    if not early_purity:
        return {"soft_by_step": soft} if soft else {}
    return {"soft_by_step": soft, "slots": len(early_purity), "early_purity": _ci(early_purity), "final_purity": _ci(final_purity),
            "purity_gain": _ci([f - e for e, f in zip(early_purity, final_purity)]),
            "early_relations_present": _ci(early_rel), "final_relations_present": _ci(final_rel),
            "early_members": _ci(early_size), "peak_members": _ci(peak_size), "final_members": _ci(final_size),
            "overextended_then_refined": sum(1 for e, f, es, fs in zip(early_purity, final_purity, early_rel, final_rel)
                                             if f > e + 0.02 or fs < es)}


def summarize_slot_decisions(rows: list[dict]) -> dict[str, Any]:
    decisions = [d for r in rows if r["method"] in {"additive", "additive_norule", "all_at_once"}
                 for d in (r.get("decisions") or [])]
    return _decision_summary([d["accept"] for d in decisions], [d["gold_good"] for d in decisions])


def _decision_summary(accept: list[bool], gold: list[bool], insample: list[bool] | None = None,
                      seed: int = 0, draws: int = 2000) -> dict[str, Any]:
    n = len(accept)
    if not n:
        return {"n": 0}
    acc = sum(a == g for a, g in zip(accept, gold)) / n
    rate = sum(accept) / n
    prevalence = sum(gold) / n
    expected_random = rate * prevalence + (1 - rate) * (1 - prevalence)
    rng = np.random.default_rng(seed)
    gold_arr = np.array(gold, dtype=bool)
    sims = []
    for _ in range(draws):
        rand = rng.random(n) < rate
        sims.append(float((rand == gold_arr).mean()))
    p_value = (1 + sum(s >= acc for s in sims)) / (1 + draws)
    lo, hi = wilson_interval(sum(a == g for a, g in zip(accept, gold)), n)
    accepted = [g for a, g in zip(accept, gold) if a]
    out = {"n": n, "accuracy": acc, "accuracy_ci": [lo, hi], "acceptance_rate": rate, "prevalence": prevalence,
           "random_matched_accuracy": expected_random, "random_matched_p": p_value,
           "accept_all_accuracy": prevalence, "precision_of_accepted": sum(accepted) / len(accepted) if accepted else float("nan"),
           "wrong_acceptance_rate": (1 - sum(accepted) / len(accepted)) if accepted else float("nan")}
    if insample is not None:
        out["insample_accuracy"] = sum(a == g for a, g in zip(insample, gold)) / n
        out["insample_acceptance_rate"] = sum(insample) / n
    return out


# -- part d, e, seed, dream, variety, continual -------------------------------------------------------

def summarize_d(rows: list[dict]) -> dict[str, Any]:
    edges = [r for r in rows if r["part"] == "d_edges"]
    out = {}
    for tag in sorted({r["tag"] for r in edges}):
        sel = [r for r in edges if r["tag"] == tag]
        s = _decision_summary([r["accept"] for r in sel], [r["gold"] for r in sel], [r["insample_accept"] for r in sel])
        acc_audit = [_num(r["audit_mean"]) for r in sel if r["accept"]]
        rej_audit = [_num(r["audit_mean"]) for r in sel if not r["accept"]]
        s["audit_utility_accepted"] = float(np.nanmean(acc_audit)) if acc_audit else float("nan")
        s["audit_utility_rejected"] = float(np.nanmean(rej_audit)) if rej_audit else float("nan")
        per_seed = []
        for seed in sorted({r["seed"] for r in sel}):
            ss = [r for r in sel if r["seed"] == seed]
            d = _decision_summary([r["accept"] for r in ss], [r["gold"] for r in ss])
            per_seed.append(d["accuracy"] - d["random_matched_accuracy"])
        s["accuracy_minus_random_by_seed"] = _ci(per_seed)
        out[tag] = s
    return out


def summarize_e(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "e"]
    out: dict[str, Any] = {}
    for k in sorted({r["k"] for r in rows}):
        sel = [r for r in rows if r["k"] == k]
        methods = sorted({r["method"] for r in sel})
        entry = {m: {key: _ci([r[key] for r in sel if r["method"] == m]) for key in ("precision", "recall", "f1", "fit")}
                 for m in methods}
        by = {m: {r["seed"]: r for r in sel if r["method"] == m} for m in methods}
        def paired(a, b, key):
            seeds = sorted(set(by.get(a, {})) & set(by.get(b, {})))
            return _ci([_num(by[a][s][key]) - _num(by[b][s][key]) for s in seeds])
        entry["contrasts"] = {"discovered − nearest_neighbour (F1)": paired("discovered", "nearest_neighbour", "f1"),
                              "discovered − known_only (F1)": paired("discovered", "known_only", "f1"),
                              "discovered − random (F1)": paired("discovered", "random", "f1"),
                              "discovered − nearest_neighbour (fit)": paired("discovered", "nearest_neighbour", "fit"),
                              "discovered − context_mean (fit)": paired("discovered", "context_mean", "fit")}
        levels = defaultdict(list)
        for r in sel:
            if r["method"] == "discovered":
                for lv, v in (r.get("f1_by_level") or {}).items():
                    levels[lv].append(v)
        entry["discovered_f1_by_level"] = {lv: _ci(v) for lv, v in sorted(levels.items())}
        out[str(k)] = entry
    return out


def summarize_seed(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "seed"]
    out = {}
    order = ["empty", "core", "noisy", "curated30", "full"]
    for cond in sorted({r["condition"] for r in rows}, key=lambda c: order.index(c) if c in order else 99):
        sel = [r for r in rows if r["condition"] == cond]
        entry = {"edges_f1_all": _ci([r["edges_all"]["f1"] for r in sel]),
                 "edges_precision_all": _ci([r["edges_all"]["precision"] for r in sel]),
                 "edges_recall_all": _ci([r["edges_all"]["recall"] for r in sel]),
                 "edges_f1_train": _ci([r["edges_train"]["f1"] for r in sel]),
                 "candidate_auc": _ci([r.get("candidate_auc") for r in sel]),
                 "offered_relation_jaccard_mean": _ci([r.get("offered_relation_jaccard_mean") for r in sel]),
                 "relations_recovered": _ci([r["relations_recovered"] for r in sel]),
                 "accepted_slots": _ci([r["accepted_slots"] for r in sel]),
                 "val_fit": _ci([r["val_fit"] for r in sel]), "test_fit": _ci([r["test_fit"] for r in sel])}
        for probe in ("isa_multihop_accuracy", "isa_hop1_accuracy", "located_in_transitivity", "part_inverse_consistency",
                      "similar_symmetry"):
            entry[probe] = _ci([r.get(probe) for r in sel])
            entry[f"gold_{probe}"] = _ci([r["gold_probes"].get(probe) for r in sel])
        rel = defaultdict(list)
        for r in sel:
            for name, v in r["relation_jaccard"].items():
                rel[name].append(v)
        entry["relation_jaccard"] = {k: _ci(v) for k, v in rel.items()}
        out[cond] = entry
    return out


def summarize_dream(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "dream"]
    out = {}
    keys = ("wrong_edges_repaired", "correct_edges_damaged", "merged_split_ari", "wrong_slot_repaired",
            "wrong_slot_reopened_or_removed", "correct_slot_disturbed", "val_fit", "audit_fit_train", "test_fit")
    for every in sorted({r["every"] for r in rows}):
        sel = [r for r in rows if r["every"] == every]
        entry = {k: _ci([float(r["after"][k]) if r["after"].get(k) is not None else None for r in sel]) for k in keys}
        entry["passes"] = sel[0]["passes"]
        accepted = defaultdict(int)
        for r in sel:
            for p in r.get("log", []):
                for rev in p["revisions"]:
                    if rev.get("proposal") == "applied":
                        accepted[rev["kind"]] += 1
                    elif rev.get("proposal") == "revisit":
                        accepted["reopened"] += int(bool(rev.get("reopened")))
                    elif rev.get("proposal") in {"remove_edges", "relabel_edges"}:
                        accepted[rev["proposal"]] += rev.get("accepted", 0)
        entry["applied_revisions"] = dict(accepted)
        out[str(every)] = entry
    if rows:
        out["before"] = {k: _ci([float(r["before"][k]) if r["before"].get(k) is not None else None for r in rows]) for k in keys}
    return out


def summarize_variety(rows: list[dict]) -> dict[str, Any]:
    edges = [r for r in rows if r["part"] == "variety"]
    out = {}
    for views in sorted({r["views"] for r in edges}):
        sel = [r for r in edges if r["views"] == views]
        s = _decision_summary([r["accept"] for r in sel], [r["gold"] for r in sel], [r["insample_accept"] for r in sel])
        art = [r for r in sel if r["artifact"]]
        s["artifact_proposals"] = len(art)
        s["artifact_acceptance_rate"] = sum(r["accept"] for r in art) / len(art) if art else float("nan")
        s["artifact_audit_utility_if_accepted"] = float(np.nanmean([_num(r["audit_mean"]) for r in art if r["accept"]])) \
            if any(r["accept"] for r in art) else float("nan")
        per_seed = []
        for seed in sorted({r["seed"] for r in sel}):
            acc = [r for r in sel if r["seed"] == seed and r["accept"]]
            per_seed.append(sum(not r["gold"] for r in acc) / len(acc) if acc else float("nan"))
        s["wrong_acceptance_by_seed"] = _ci(per_seed)
        out[str(views)] = s
    return out


def summarize_continual(rows: list[dict]) -> dict[str, Any]:
    rows = [r for r in rows if r["part"] == "continual"]
    out = {}
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond]
        stages = max(len(r["curve"]) for r in sel)
        curve = [_ci([r["curve"][i]["matched"] for r in sel if len(r["curve"]) > i]) for i in range(stages)]
        final = _ci([r["curve"][-1]["matched"] for r in sel])
        retention = defaultdict(lambda: {"at_arrival": [], "final": []})
        framed = defaultdict(list)
        for r in sel:
            for name, v in r["retention"].items():
                retention[name]["at_arrival"].append(v["at_arrival"]); retention[name]["final"].append(v["final"])
            for name, v in r["curve"][-1]["per_relation"].items():
                framed[name].append(v.get("jaccard_framed"))
        out[cond] = {"matched_by_stage": curve, "final_matched": final,
                     "final_jaccard_framed": {k: _ci(v) for k, v in framed.items()},
                     "final_val_fit": _ci([r["curve"][-1]["val_fit"] for r in sel]),
                     "retention": {k: {"at_arrival": _ci(v["at_arrival"]), "final": _ci(v["final"]),
                                       "drop": _ci([a - f for a, f in zip(v["at_arrival"], v["final"])])}
                                   for k, v in retention.items()}}
    return out


def verdicts(summary: dict[str, Any]) -> dict[str, Any]:
    """H-H refutation clauses (execution.md §E10), evaluated where the parts ran."""
    out = {}
    b = summary.get("b", {})
    if b:
        out["(i) erased edges recovered above matched random candidates"] = {
            rate: bool(_sig(v["auc_minus_random"]) and _sig(v["precision_minus_prevalence"])) for rate, v in b.items()}
    c = summary.get("c", {})
    if c:
        out["(ii) blank slots align with hidden relations above chance (ARI − permuted, additive)"] = {
            mode: bool(_sig(v["methods"]["additive"]["ari_minus_permuted"])) for mode, v in c.items() if "additive" in v["methods"]}
        out["(ii′) same, per-seed permutation test (Holm over seeds; added after the first run, reported alongside)"] = {
            mode: bool(v["methods"]["additive"].get("permutation_holm_max_p", 1.0) < 0.05) for mode, v in c.items()
            if "additive" in v["methods"]}
    d = summary.get("d", {})
    if d:
        out["(iii) self-acceptance beats random acceptance at the matched rate (edges)"] = {
            tag: bool(_sig(v["accuracy_minus_random_by_seed"])) for tag, v in d.items()}
        slots = c.get("absent", {}).get("slot_decisions", {})
        if slots.get("n"):
            out["(iii) self-acceptance beats random acceptance at the matched rate (slots, pooled)"] = bool(
                slots["accuracy"] > slots["random_matched_accuracy"] and slots["random_matched_p"] < 0.05)
    e = summary.get("e", {})
    if e:
        out["(iv) new-word frame inference beats nearest-neighbour frames (F1)"] = {
            k: bool(_sig(v["contrasts"]["discovered − nearest_neighbour (F1)"])) for k, v in e.items()}
    return out


def summarize(rows: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    summary = {"a": summarize_a(rows), "b": summarize_b(rows), "c": summarize_c(rows), "d": summarize_d(rows),
               "e": summarize_e(rows), "seed_ontology": summarize_seed(rows), "dream": summarize_dream(rows),
               "variety": summarize_variety(rows), "continual": summarize_continual(rows)}
    summary["verdicts"] = verdicts(summary)
    return summary


# -- rendering ------------------------------------------------------------------------------------------

def render_report(summary: dict[str, Any], config: dict[str, Any]) -> str:
    L: list[str] = [f"# {config.get('stage', 'E10')} — self-learned semantics (H-H)", "",
                    f"Seeds {config['seeds']}; CPU single-threaded jobs; world `{config['world'].get('kind', 'synthetic')}`. "
                    "Means over seeds with 95% t-intervals in brackets. Gold labels and audit observations are used only "
                    "by evaluation; every learning-side decision uses held-out self-tests.", ""]
    v = summary.get("verdicts", {})
    if v:
        L += ["## H-H refutation clauses", "", "| Clause | Result |", "|---|---|"]
        for k, val in v.items():
            L.append(f"| {k} | {val} |")
        L.append("")
    a = summary.get("a")
    if a:
        L += ["## (a) Learnability ablation (30% erased, 5% spurious prior edges, noisy priors)", "",
              "Contrasts against everything learnable with L2-to-prior (paired over seeds):", "",
              "| Contrast | Δ held-out-obs fit (train concepts) | Δ test-concept fit (zero-shot) | Δ recovery F1 | Δ recovery AUC |",
              "|---|---|---|---|---|"]
        for name, c in a["contrasts"].items():
            L.append(f"| {name} | {_fmt(c['val_fit'], 4)} | {_fmt(c['test_fit'], 4)} | {_fmt(c['f1'])} | {_fmt(c['auc'])} |")
        L += ["", "Selected grid cells:", "", "| atomics / relations / mapping / frames | val fit | test fit | recovery F1 | AUC | atomic cos | spurious removed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        shown = ["l2|l2|l2|l2", "free|free|free|free", "fixed|fixed|fixed|fixed", "fixed|fixed|fixed|l2", "l2|l2|l2|fixed",
                 "fixed|l2|l2|l2", "l2|fixed|l2|l2", "l2|l2|fixed|l2", "l2|l2|l2|free", a["best_test_fit"]]
        for key in dict.fromkeys(shown):
            if key in a["table"]:
                t = a["table"][key]
                L.append(f"| {key.replace('|', ' / ')} | {_fmt(t['val_fit'], 4)} | {_fmt(t['test_fit'], 4)} | {_fmt(t['f1'])} | "
                         f"{_fmt(t['auc'])} | {_fmt(t['atomic_cosine'])} | {_fmt(t['spurious_removed'])} |")
        L += ["", f"Best test-concept fit in the full 3⁴ grid: `{a['best_test_fit']}`.", ""]
    b = summary.get("b")
    if b:
        L += ["## (b) Erasure & recovery", "",
              "| Erased (principal fixed) | Precision | Recall | F1 | AUC | R-precision | Random (prevalence / AUC 0.5) | Filler-frequency AUC / R-prec | Test fit |",
              "|---|---|---|---|---|---|---|---|---|"]
        for rate, m in b.items():
            L.append(f"| {float(rate):.0%} ({1 - float(rate):.0%}) | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
                     f"{_fmt(m['auc'])} | {_fmt(m['r_precision'])} | {_fmt(m['prevalence'])} / 0.5 | "
                     f"{_fmt(m['frequency_auc'])} / {_fmt(m['frequency_r_precision'])} | {_fmt(m['test_fit'], 4)} |")
        L.append("")
    c = summary.get("c")
    if c:
        for mode, cm in c.items():
            L += [f"## (c) Blank-relation discovery — hidden relations {mode}", "",
                  "| Method | ARI (gold edges) | ARI − permuted | permutation p (Holm, max over seeds) | mean best Jaccard | detection P | detection R | accepted / proposed slots | test fit |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for m, e in cm["methods"].items():
                L.append(f"| {m} | {_fmt(e['ari_gold_edges'])} | {_fmt(e['ari_minus_permuted'])} | {_fmt(e.get('permutation_holm_max_p'))} | {_fmt(e['mean_best_jaccard'])} | "
                         f"{_fmt(e['detection_precision'])} | {_fmt(e['detection_recall'])} | "
                         f"{_fmt(e['accepted_slots'], 1)} / {_fmt(e['proposed_slots'], 1)} | {_fmt(e['test_fit'], 4)} |")
            L.append("")
            L += ["Per hidden relation, best Jaccard of a discovered cluster (candidate level):", "",
                  "| Method | " + " | ".join(next(iter(cm["methods"].values()))["per_relation_best_jaccard"]) + " |",
                  "|---|" + "---|" * len(next(iter(cm["methods"].values()))["per_relation_best_jaccard"])]
            for m, e in cm["methods"].items():
                L.append(f"| {m} | " + " | ".join(_fmt(x) for x in e["per_relation_best_jaccard"].values()) + " |")
            L.append("")
            for m in ("additive", "additive_norule", "all_at_once"):
                rr = cm["methods"].get(m, {}).get("relation_recovery_jaccard")
                if rr:
                    L.append(f"Relation recovery over all framed heads (accepted slots incl. rule-implied edges), {m}: "
                             + "; ".join(f"{k} {_fmt(x)}" for k, x in rr.items()) + ".")
            L.append("")
            for name, con in cm["contrasts"].items():
                if "rule" in name:
                    L.append(f"- {name}: {_fmt(con)}")
                else:
                    L.append(f"- {name}: ARI {_fmt(con['ari_gold_edges'])}, mean best Jaccard {_fmt(con['mean_best_jaccard'])}, "
                             f"test fit {_fmt(con['test_fit'], 4)}")
            r = cm["riddle"]
            L += ["", f"**Riddle-style interpretation (E10.6, {mode}).** {r['accepted_slots']} accepted slots, "
                  f"{r['with_adopted_rule']} adopted a structural rule; {r['adopted_true']} of those rules are true of the "
                  f"matched hidden relation, {r['adopted_designed']} are its designed property. Rule predictions' gold "
                  f"precision {_fmt(r['rule_prediction_precision'])}. Hypotheses scored per slot (full space): "
                  f"{_fmt(r['hypotheses_per_slot'], 1)}.", ""]
            if r["by_relation"]:
                L += ["| Matched hidden relation | slots | adopted | true | designed | rules adopted |", "|---|---:|---:|---:|---:|---|"]
                for rel, d in r["by_relation"].items():
                    rules = ", ".join(f"{k[5:]}×{v}" for k, v in d.items() if k.startswith("rule:"))
                    L.append(f"| {rel} | {d.get('slots', 0)} | {d.get('adopted', 0)} | {d.get('true', 0)} | {d.get('designed', 0)} | {rules} |")
                L.append("")
            L += ["Accuracy of the adopted hypothesis vs number of candidate hypotheses (one true + m−1 false):", "",
                  "| m | " + " | ".join(r["accuracy_vs_hypotheses"]) + " |", "|---|" + "---:|" * len(r["accuracy_vs_hypotheses"]),
                  "| accuracy | " + " | ".join(_fmt(x["accuracy"], 2) for x in r["accuracy_vs_hypotheses"].values()) + " |",
                  "| slots | " + " | ".join(str(x["slots"]) for x in r["accuracy_vs_hypotheses"].values()) + " |", ""]
            t = cm["trajectories"]
            if t:
                L += [f"**Human-likeness (additive, {mode}).** Over {t['slots']} slots: purity of a slot's members w.r.t. its "
                      f"final hidden relation rises from {_fmt(t['early_purity'])} (first committed members) to "
                      f"{_fmt(t['final_purity'])}; hidden relations present {_fmt(t['early_relations_present'], 2)} → "
                      f"{_fmt(t['final_relations_present'], 2)}; members {_fmt(t['early_members'], 1)} (peak "
                      f"{_fmt(t['peak_members'], 1)}) → {_fmt(t['final_members'], 1)}. Slots that overextended then refined: "
                      f"{t['overextended_then_refined']}.", ""]
            if t and t.get("soft_by_step"):
                L += ["Soft hypothesis of a newly opened slot (mass-weighted shares of the pairs it holds; steps since opening):", "",
                      "| step | target relation | other hidden relations | distractors | effective size |", "|---:|---:|---:|---:|---:|"]
                for step, m in t["soft_by_step"].items():
                    L.append(f"| {step} | {m['soft_target_share']:.2f} | {m['soft_other_relation_share']:.2f} | "
                             f"{m['soft_distractor_share']:.2f} | {m['soft_effective_size']:.0f} |")
                L.append("")
            sd = cm["slot_decisions"]
            if sd.get("n"):
                L += [f"Slot self-acceptance ({mode}; additive, no-rule and all-at-once runs pooled): accuracy {sd['accuracy']:.3f} "
                      f"(Wilson {sd['accuracy_ci'][0]:.2f}–{sd['accuracy_ci'][1]:.2f}, n = {sd['n']}) vs random at the matched "
                      f"rate {sd['random_matched_accuracy']:.3f} (p = {sd['random_matched_p']:.3f}); accept-all "
                      f"{sd['accept_all_accuracy']:.3f}; wrong-acceptance rate {_fmt(sd['wrong_acceptance_rate'])}.", ""]
    d = summary.get("d")
    if d:
        L += ["## (d) Self-tested acceptance of edge hypotheses (held-out observations; audit never read)", "",
              "| Source | n | accept rate | accuracy (Wilson) | random @ matched rate | accept-all | in-sample (confirmation-bias control) | acc − random by seed | audit Δ accepted / rejected |",
              "|---|---:|---:|---|---:|---:|---|---|---|"]
        for tag, s in d.items():
            L.append(f"| {tag} | {s['n']} | {s['acceptance_rate']:.2f} | {s['accuracy']:.3f} ({s['accuracy_ci'][0]:.2f}–{s['accuracy_ci'][1]:.2f}) | "
                     f"{s['random_matched_accuracy']:.3f} | {s['accept_all_accuracy']:.3f} | {s['insample_accuracy']:.3f} "
                     f"(accepts {s['insample_acceptance_rate']:.2f}) | {_fmt(s['accuracy_minus_random_by_seed'])} | "
                     f"{s['audit_utility_accepted']:+.4f} / {s['audit_utility_rejected']:+.4f} |")
        L.append("")
    e = summary.get("e")
    if e:
        L += ["## (e) New-word frame inference (fast mapping)", "",
              "| k | discovered F1 | known-only F1 | oracle-dictionary F1 | nearest-neighbour F1 | random F1 | discovered fit | NN fit | context-mean fit | gold-frame fit |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, entry in e.items():
            g = lambda m, key: _fmt(entry.get(m, {}).get(key), 3 if key == "f1" else 4)  # noqa: E731
            L.append(f"| {k} | {g('discovered', 'f1')} | {g('known_only', 'f1')} | {g('oracle', 'f1')} | {g('nearest_neighbour', 'f1')} | "
                     f"{g('random', 'f1')} | {g('discovered', 'fit')} | {g('nearest_neighbour', 'fit')} | {g('context_mean', 'fit')} | "
                     f"{g('gold_frame', 'fit')} |")
        L.append("")
        for k, entry in e.items():
            L.append(f"- k = {k}: " + "; ".join(f"{name} {_fmt(ci)}" for name, ci in entry["contrasts"].items()))
        last = list(e.values())[-1]
        if last.get("discovered_f1_by_level"):
            L += ["", "Discovered-relation F1 by hierarchy depth at the largest k (exploratory 'basic-level' readout): "
                  + "; ".join(f"depth {lv}: {_fmt(ci)}" for lv, ci in last["discovered_f1_by_level"].items()) + "."]
        L.append("")
    s = summary.get("seed_ontology")
    if s:
        L += ["## E10.4 Seed ontology → recovered logical ontology", "",
              "| Start | edge F1 (all framed) | P / R | candidate AUC | candidate-level relation Jaccard (mean) | relations recovered (J ≥ 0.5, of 7) | accepted slots | multi-hop is-a acc (gold) | transitivity located_in (gold) | inverse part (gold) | symmetry similar (gold) | test fit |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cond, m in s.items():
            L.append(f"| {cond} | {_fmt(m['edges_f1_all'])} | {_fmt(m['edges_precision_all'], 2)} / {_fmt(m['edges_recall_all'], 2)} | "
                     f"{_fmt(m['candidate_auc'])} | {_fmt(m.get('offered_relation_jaccard_mean'))} | {_fmt(m['relations_recovered'], 1)} | {_fmt(m['accepted_slots'], 1)} | "
                     f"{_fmt(m['isa_multihop_accuracy'], 2)} ({_fmt(m['gold_isa_multihop_accuracy'], 2)}) | "
                     f"{_fmt(m['located_in_transitivity'], 2)} ({_fmt(m['gold_located_in_transitivity'], 2)}) | "
                     f"{_fmt(m['part_inverse_consistency'], 2)} ({_fmt(m['gold_part_inverse_consistency'], 2)}) | "
                     f"{_fmt(m['similar_symmetry'], 2)} ({_fmt(m['gold_similar_symmetry'], 2)}) | {_fmt(m['test_fit'], 4)} |")
        L.append("")
    dr = summary.get("dream")
    if dr:
        L += ["## E10.5 Dreaming (offline self-revision) after injected corruptions", "",
              "| Dream every (steps) | passes | wrong edges repaired | correct edges damaged | merged relations split (ARI) | wrong slot repaired | wrong slot reopened/removed | correct slots disturbed | audit fit (train concepts) | applied revisions |",
              "|---|---:|---|---|---|---|---|---|---|---|"]
        for every, m in dr.items():
            if every == "before":
                continue
            label = "off" if every == "0" else (f"revisit only, every {every[1:]}" if every.startswith("-") else every)
            L.append(f"| {label} | {m['passes']} | {_fmt(m['wrong_edges_repaired'], 2)} | {_fmt(m['correct_edges_damaged'], 3)} | "
                     f"{_fmt(m['merged_split_ari'], 2)} | {_fmt(m['wrong_slot_repaired'], 2)} | {_fmt(m['wrong_slot_reopened_or_removed'], 2)} | "
                     f"{_fmt(m['correct_slot_disturbed'], 2)} | {_fmt(m['audit_fit_train'], 4)} | {m['applied_revisions']} |")
        b0 = dr.get("before")
        if b0:
            L.append(f"\nBefore any revision (right after injection): audit fit {_fmt(b0['audit_fit_train'], 4)}.")
        L.append("")
    va = summary.get("variety")
    if va:
        L += ["## E10.7 Data variety vs self-confirmation bias", "",
              "| Sources per concept (8 non-audit observations) | proposals | wrong self-acceptance rate | by seed | artifact proposals | artifact acceptance | audit Δ of accepted artifacts | in-sample acceptance |",
              "|---|---:|---|---|---:|---:|---|---|"]
        for views, m in va.items():
            L.append(f"| {views} | {m['n']} | {_fmt(m['wrong_acceptance_rate'])} | {_fmt(m['wrong_acceptance_by_seed'])} | "
                     f"{m['artifact_proposals']} | {_fmt(m['artifact_acceptance_rate'])} | {_fmt(m['artifact_audit_utility_if_accepted'], 4)} | "
                     f"{_fmt(m.get('insample_acceptance_rate'))} |")
        L.append("")
    co = summary.get("continual")
    if co:
        L += ["## E10.8 Continual additive learning (hidden relations arrive one per stage)", "",
              "| Condition | relations matched (Jaccard ≥ 0.5 on offered candidate pairs) by stage | final | final Jaccard on offered pairs | final Jaccard over all framed heads (incl. rule closure) | retention drop (arrival − final) | final val fit |",
              "|---|---|---|---|---|---|---|"]
        for cond, m in co.items():
            stages = " → ".join(_fmt(x, 2) for x in m["matched_by_stage"])
            drops = "; ".join(f"{k} {_fmt(v['drop'], 2)}" for k, v in m["retention"].items())
            finals = "; ".join(f"{k} {_fmt(v['final'], 2)}" for k, v in m["retention"].items())
            framed = "; ".join(f"{k} {_fmt(v, 2)}" for k, v in m.get("final_jaccard_framed", {}).items())
            L.append(f"| {cond} | {stages} | {_fmt(m['final_matched'], 2)} | {finals} | {framed} | {drops} | {_fmt(m['final_val_fit'], 4)} |")
        L.append("")
    timing = summary.get("job_cpu_seconds_by_part")
    if timing:
        L += ["CPU seconds by part (sum over jobs): " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()) + ".", ""]
    return "\n".join(L)


# =====================================================================================================
# E10 baselines (WP-PQ2; `e10_baselines`): summaries and report section. Opt-in: nothing above uses it.
# =====================================================================================================

BASELINE_RECOVERY_METHODS = ("prior_corr", "amie", "transe", "rotate", "complex", "itere")
BASELINE_EDGE_VARIANTS = ("reused", "fresh", "ttest", "holm")
BASELINE_EXPLAINERS = ("riddle", "amie", "rotate", "hrr")


def _reference_b_rows(config: dict[str, Any]) -> dict[tuple[int, float], dict]:
    """Part-b rows of the learnable ontology's E10.0 run (`recovery.reference`), by (seed, rate)."""
    import json
    from pathlib import Path
    path = (config.get("recovery") or {}).get("reference")
    if not path or not Path(path).exists():
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("part") == "b":
            out[(int(row["seed"]), float(row["rate"]))] = row
    return out


def _axiom_reading(row: dict, scores: list, threshold: float, *, passes: list | None = None) -> dict[str, Any]:
    """Precision / recall / F1 / AUC of one axiom reading (scores ≥ threshold, or `passes`) against the gold axioms."""
    from vsa_embed.kg_baselines import auc
    gold = [bool(g) for g in (row["axiom_gold"] if "axiom_gold" in row else row["gold"])]
    kinds = row.get("kinds") or row.get("axiom_kinds")
    values = [(-2.0 if s is None else float(s)) for s in scores]
    chosen = [bool(p) for p in passes] if passes is not None else [v >= threshold for v in values]
    tp = sum(c and g for c, g in zip(chosen, gold))
    predicted, positives = sum(chosen), sum(gold)
    precision = tp / predicted if predicted else float("nan")
    recall = tp / positives if positives else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if predicted and positives and precision + recall > 0 else 0.0
    by_kind = {}
    for kind in ("symmetric", "inverse", "transitive", "chain"):
        idx = [i for i, k in enumerate(kinds) if k == kind and gold[i]]
        if idx:
            by_kind[kind] = sum(chosen[i] for i in idx) / len(idx)
    return {"precision": precision, "recall": recall, "f1": f1, "predicted": predicted, "gold": positives,
            "auc": auc(values, gold), "recall_by_kind": by_kind}


def summarize_baselines(rows: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    """Summary of an `e10_baselines` run: main seeds only (dev-seed rows set the thresholds)."""
    thresholds = config.get("thresholds") or {}
    main = [r for r in rows if not r.get("dev")]
    out: dict[str, Any] = {"seeds": sorted({int(r["seed"]) for r in main}), "thresholds": thresholds}
    # ---- recovery (D-B2)
    reference = _reference_b_rows(config)
    rec = [r for r in main if r["part"] == "recovery"]
    recovery: dict[str, Any] = {}
    for rate in sorted({float(r["rate"]) for r in rec}):
        block: dict[str, Any] = {"methods": {}, "pool_matches_reference": True}
        sel = [r for r in rec if float(r["rate"]) == rate]
        ref = {s: reference.get((s, rate)) for s in sorted({int(r["seed"]) for r in sel})}
        for s, row in ref.items():
            any_row = next(r for r in sel if int(r["seed"]) == s)
            if row is None or int(row.get("candidates", -1)) != int(any_row["candidates"]):
                block["pool_matches_reference"] = False
        if ref and all(v is not None for v in ref.values()):
            block["methods"]["learnable ontology"] = {m: _ci([ref[s][m] for s in ref]) for m in ("auc", "r_precision", "f1")}
            block["frequency_baseline"] = {"auc": _ci([ref[s].get("frequency_auc") for s in ref]),
                                           "r_precision": _ci([ref[s].get("frequency_r_precision") for s in ref])}
        for method in BASELINE_RECOVERY_METHODS:
            ms = {int(r["seed"]): r for r in sel if r["method"] == method}
            if not ms:
                continue
            entry = {m: _ci([ms[s][m] for s in ms]) for m in ("auc", "r_precision", "f1", "precision", "recall", "validation_auc")}
            if all(ref.get(s) for s in ms):
                entry["auc_minus_learnable"] = _ci([_num(ms[s]["auc"]) - _num(ref[s]["auc"]) for s in ms])
                entry["f1_minus_learnable"] = _ci([_num(ms[s]["f1"]) - _num(ref[s]["f1"]) for s in ms])
            block["methods"][method] = entry
        recovery[str(rate)] = block
    out["recovery"] = recovery
    # ---- axioms (D-B1)
    ax = [r for r in main if r["part"] == "axioms"]
    edges03 = {int(r["seed"]): r for r in main if r["part"] == "edges" and r.get("hrr") is not None}
    readings: dict[str, list[dict]] = defaultdict(list)
    for r in ax:
        readings["AMIE (defaults: PCA ≥ 0.1, HC ≥ 0.01)"].append(_axiom_reading(r, r["amie_pca"], 0, passes=r["amie_passes"]))
        tuned = thresholds.get("amie", float("inf"))
        readings[f"AMIE (default filters and PCA ≥ {tuned:.3g}, dev-chosen)"].append(
            _axiom_reading(r, r["amie_pca"], tuned, passes=[bool(p) and (s or 0) >= tuned for p, s in zip(r["amie_passes"], r["amie_pca"])]))
        readings[f"IterE-style RotatE phases (≥ {thresholds.get('rotate', float('nan')):.3g}, dev-chosen)"].append(
            _axiom_reading(r, r["rotate"], thresholds.get("rotate", float("inf"))))
        e = edges03.get(int(r["seed"]))
        if e is not None:
            readings[f"learnable ontology's HRR roles (≥ {thresholds.get('hrr', float('nan')):.3g}, dev-chosen)"].append(
                _axiom_reading(e, e["hrr"], thresholds.get("hrr", float("inf"))))
    axioms = {}
    for name, items in readings.items():
        kinds = sorted({k for it in items for k in it["recall_by_kind"]})
        axioms[name] = {"precision": _ci([it["precision"] for it in items]), "recall": _ci([it["recall"] for it in items]),
                        "f1": _ci([it["f1"] for it in items]), "auc": _ci([it["auc"] for it in items]),
                        "predicted": _ci([it["predicted"] for it in items]), "gold": _ci([it["gold"] for it in items]),
                        "recall_by_kind": {k: _ci([it["recall_by_kind"].get(k) for it in items]) for k in kinds}}
    out["axioms"] = axioms
    out["gold_axioms"] = sorted({a for r in ax for a in r.get("gold_axioms", [])})
    closures: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in ax:
        for key, v in (r.get("closures") or {}).items():
            for m in ("axioms", "closure", "erased_recall", "precision_vs_gold"):
                closures[key][m].append(v.get(m))
    out["axiom_closures"] = {k: {m: _ci(v) for m, v in d.items()} for k, d in closures.items()}
    # ---- edges (D-B5)
    ed = [r for r in main if r["part"] == "edges"]
    edges: dict[str, Any] = {}
    for scenario in sorted({r["scenario"] for r in ed}):
        sel = [r for r in ed if r["scenario"] == scenario and r.get("gold") is not None]
        block = {"proposals": sum(int(r["proposals"]) for r in sel)}
        for variant in BASELINE_EDGE_VARIANTS:
            if not any(variant in r for r in sel):          # e.g. no fresh split on WordNet (no clean targets)
                continue
            per_seed: dict[str, list] = defaultdict(list)
            pooled = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for r in sel:
                g = [bool(x) for x in r["gold"]]; a = [bool(x) for x in r[variant]]
                tp = sum(x and y for x, y in zip(a, g)); fp = sum(x and not y for x, y in zip(a, g))
                fn = sum((not x) and y for x, y in zip(a, g)); tn = sum((not x) and (not y) for x, y in zip(a, g))
                for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
                    pooled[k] += v
                n = len(g)
                per_seed["acceptance"].append(sum(a) / n if n else float("nan"))
                per_seed["accuracy"].append((tp + tn) / n if n else float("nan"))
                per_seed["wrong_among_accepted"].append(fp / (tp + fp) if tp + fp else float("nan"))
                per_seed["false_acceptance"].append(fp / (fp + tn) if fp + tn else float("nan"))
                per_seed["recall"].append(tp / (tp + fn) if tp + fn else float("nan"))
            block[variant] = {**{k: _ci(v) for k, v in per_seed.items()}, "pooled": pooled}
        block["prevalence"] = _ci([sum(map(bool, r["gold"])) / len(r["gold"]) for r in sel if r["gold"]])
        edges[scenario] = block
    out["edges"] = edges
    # ---- discovery (D-B1 identification, D-B4 null worlds, D-B5 fresh holdout)
    dis = [r for r in main if r["part"] == "discovery"]
    discovery: dict[str, Any] = {}
    for world in ("absent", "collapsed", "null_distractors", "null_permuted"):
        for holdout in ("reused", "fresh"):
            sel = [r for r in dis if r["world"] == world and r["holdout"] == holdout]
            if not sel:
                continue
            null = world.startswith("null")
            accepted = [[s for s in r["slots"] if s["accept"]] for r in sel]
            real = [[] if null else [s for s in acc if (s.get("gold_precision") or 0) >= 0.5] for acc in accepted]
            false_counts = [len(a) - len(rr) for a, rr in zip(accepted, real)]
            entry: dict[str, Any] = {
                "runs": len(sel), "accepted_slots": _ci([len(a) for a in accepted]),
                "proposed_slots": _ci([len(r["slots"]) for r in sel]), "false_accepted_slots": _ci(false_counts),
                "runs_with_false_acceptance": sum(1 for c in false_counts if c > 0),
                "offered_rules": _ci([len(r.get("offered_rules") or []) for r in sel]),
                "accepted_sizes": [s["size"] for a in accepted for s in a]}
            flat = [s for a in accepted for s in a]
            for explainer in BASELINE_EXPLAINERS:
                adopted = [s for s in flat if s[explainer]["adopted"]]
                entry[explainer] = {"accepted": len(flat), "adopted": len(adopted),
                                    "true": sum(bool(s[explainer]["true"]) for s in adopted),
                                    "designed": sum(bool(s[explainer]["designed"]) for s in adopted)}
            if not null:
                rec_r: dict[str, list] = defaultdict(list); rec_a: dict[str, list] = defaultdict(list)
                for r in sel:
                    best: dict[str, float] = defaultdict(float)
                    for s in r["slots"]:
                        if not (s["accept"] and s.get("best_relation")):
                            continue
                        value = s.get("amie_closure_jaccard")
                        value = s.get("slot_jaccard_framed") if value is None else value
                        best[s["best_relation"]] = max(best[s["best_relation"]], float(value or 0.0))
                    for name, v in (r.get("relation_recovery") or {}).items():
                        rec_r[name].append(v["jaccard"]); rec_a[name].append(best.get(name, 0.0))
                entry["closure_jaccard_riddle"] = {k: _ci(v) for k, v in rec_r.items()}
                entry["closure_jaccard_amie"] = {k: _ci(v) for k, v in rec_a.items()}
                entry["rule_precision"] = {
                    "riddle": _ci([s["riddle_rule_precision"] for s in flat if s.get("riddle_rule_precision") is not None]),
                    "amie": _ci([s["amie_rule_precision"] for s in flat if s.get("amie_rule_precision") is not None])}
            discovery[f"{world}/{holdout}"] = entry
    out["discovery"] = discovery
    return out


def render_baselines(summary: dict[str, Any], config: dict[str, Any]) -> str:
    """Markdown report of an `e10_baselines` run (claim D baselines; novelty check §5.5)."""
    th = summary.get("thresholds") or {}
    L = [f"# {config.get('stage', 'E10 baselines')} — baselines for claim D (WP-PQ2)", "",
         f"Seeds {summary.get('seeds')}; dev seeds {config.get('dev_seeds')} set the operator-axiom thresholds only "
         f"(RotatE {th.get('rotate', float('nan')):.3f}, HRR roles {th.get('hrr', float('nan')):.3f}, tuned AMIE "
         f"{th.get('amie', float('nan')):.3f}). World and scenarios: `{config.get('base_config')}`. Means over seeds with "
         "95% t-intervals in brackets. Baselines see only asserted edges (and, for the learnable ontology, training and "
         "validation observations); gold is read by the evaluation only.", ""]
    rec = summary.get("recovery") or {}
    if rec:
        matches = all(b.get("pool_matches_reference") for b in rec.values())
        base = config.get("base_stage") or "E10"
        reference = (config.get("recovery") or {}).get("reference")
        L += [f"## D-B2 Erased-edge recovery on the {base} (b) candidate pools", "",
              "Methods are fit on 90% of the asserted edges; F1 uses a threshold chosen on the other 10% plus distractors. "
              f"The learnable ontology's row (and the filler-frequency row) is its committed {base} run `{reference}` (same "
              "pools; " + ("pool sizes match" if matches else "**pool sizes differ**") + "), whose F1 uses its fixed mass ≥ 0.5 "
              "rule. With one erased edge per two distractors, accepting every candidate gives F1 = 0.5, so AUC and "
              "R-precision are the primary comparison.", "",
              "| Erased | Method | AUC | R-precision | F1 | ΔAUC vs learnable | ΔF1 vs learnable |", "|---|---|---|---|---|---|---|"]
        for rate, block in rec.items():
            for method, m in block["methods"].items():
                L.append(f"| {float(rate):.0%} | {method} | {_fmt(m['auc'])} | {_fmt(m['r_precision'])} | {_fmt(m['f1'])} | "
                         f"{_fmt(m.get('auc_minus_learnable'))} | {_fmt(m.get('f1_minus_learnable'))} |")
            fb = block.get("frequency_baseline")
            if fb:
                L.append(f"| {float(rate):.0%} | filler-frequency ranking | {_fmt(fb['auc'])} | {_fmt(fb['r_precision'])} | — | | |")
        L.append("")
    ax = summary.get("axioms") or {}
    if ax:
        L += ["## D-B1 Horn axioms on the 30%-erasure graph (vs axioms with confidence ≥ 0.9 on the complete gold graph)", "",
              f"Gold axioms (union over seeds): {', '.join(summary.get('gold_axioms', []))}.", "",
              "| Reading | precision | recall | F1 | AUC of the score | axioms accepted | recall: symmetric / inverse / transitive / chain |",
              "|---|---|---|---|---|---|---|"]
        for name, m in ax.items():
            kinds = m["recall_by_kind"]
            L.append(f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['auc'])} | "
                     f"{_fmt(m['predicted'], 1)} | " + " / ".join(_fmt(kinds.get(k), 2) for k in ("symmetric", "inverse", "transitive", "chain")) + " |")
        cl = summary.get("axiom_closures") or {}
        if cl:
            L += ["", "Closure of the accepted axioms over the observed graph (one application; `amie_dev` = support ≥ 2 and "
                  "PCA ≥ the dev threshold; `rotate` = supported axioms above the dev threshold, at most 50, as injected by "
                  "the IterE loop):", "",
                  "| Axioms from | axioms | closure triples | recall of erased edges | precision vs gold |", "|---|---|---|---|---|"]
            for key, m in cl.items():
                L.append(f"| {key} | {_fmt(m['axioms'], 1)} | {_fmt(m['closure'], 1)} | {_fmt(m['erased_recall'])} | {_fmt(m['precision_vs_gold'])} |")
        L.append("")
    ed = summary.get("edges") or {}
    if ed:
        L += ["## D-B5 Edge self-test: re-used validation split vs fresh split vs multiplicity correction", "",
              "`reused` = E10.0 (lower bound > 0 on the one validation split); `fresh` = a fresh simulated draw of the same "
              "size per proposal; `ttest` = one-sided t-test per proposal (α 0.025) on the re-used split; `holm` = Holm over "
              "all proposals of a run. `null` = nothing erased: every proposal is a distractor.", "",
              "| Scenario | proposals | variant | acceptance | accuracy | wrong among accepted | false acceptance (of non-gold) | recall |",
              "|---|---:|---|---|---|---|---|---|"]
        for scenario, block in ed.items():
            label = "null (no erasure)" if scenario == "null" else f"erasure {float(scenario):.0%}"
            for variant in BASELINE_EDGE_VARIANTS:
                m = block.get(variant)
                if m:
                    L.append(f"| {label} | {block['proposals']} | {variant} | {_fmt(m['acceptance'])} | {_fmt(m['accuracy'])} | "
                             f"{_fmt(m['wrong_among_accepted'])} | {_fmt(m['false_acceptance'])} | {_fmt(m['recall'])} |")
        L.append("")
    dis = summary.get("discovery") or {}
    if dis:
        L += ["## D-B4 / D-B5 Blank-slot discovery: real, null and fresh-holdout worlds", "",
              "Null worlds have no hidden relation (`null_distractors`: every relation asserted, only distractors offered; "
              "`null_permuted`: the absent scenario with the offered pairs' tails permuted), so every accepted slot is a false "
              "discovery. In the real worlds a slot counts as false if fewer than half of its pairs are offered gold pairs of a "
              "hidden relation.", "",
              "| World / holdout | runs | proposed slots | accepted slots | false accepted slots | runs with ≥ 1 false acceptance | AMIE rules on the offered pairs |",
              "|---|---:|---|---|---|---:|---|"]
        for key, m in dis.items():
            L.append(f"| {key} | {m['runs']} | {_fmt(m['proposed_slots'], 2)} | {_fmt(m['accepted_slots'], 2)} | "
                     f"{_fmt(m['false_accepted_slots'], 2)} | {m['runs_with_false_acceptance']} | {_fmt(m['offered_rules'], 1)} |")
        L += ["", "Explaining the accepted slots (pooled over seeds): adopted / true of the slot's relation / its designed property. "
              "`riddle` = E10.6 held-out hypothesis tests; `amie` = best AMIE rule on the captured pairs; `rotate` = IterE-style "
              "axiom from a RotatE fit; `hrr` = axiom read off the slot's own learned operator.", "",
              "| World / holdout | accepted slots | riddle | AMIE | RotatE | HRR roles |", "|---|---:|---|---|---|---|"]
        for key, m in dis.items():
            cells = [f"{m[e]['adopted']} / {m[e]['true']} / {m[e]['designed']}" for e in BASELINE_EXPLAINERS]
            L.append(f"| {key} | {m['riddle']['accepted']} | " + " | ".join(cells) + " |")
        rows = [(k, m) for k, m in dis.items() if m.get("closure_jaccard_riddle")]
        if rows:
            L += ["", "Relation completion with the adopted rule (Jaccard over all framed heads, best accepted slot per hidden "
                  "relation; riddle = E10.0 crystallization with rule enforcement, AMIE = captured pairs ∪ the AMIE rule's predictions):", "",
                  "| World / holdout | relation | riddle | AMIE |", "|---|---|---|---|"]
            for key, m in rows:
                for name, v in m["closure_jaccard_riddle"].items():
                    L.append(f"| {key} | {name} | {_fmt(v)} | {_fmt(m['closure_jaccard_amie'].get(name))} |")
            L += ["", "Gold precision of the adopted rule's predictions beyond the captured pairs (mean over accepted slots with "
                  "an adopted rule and a matched relation): " + "; ".join(
                      f"{key}: riddle {_fmt(m['rule_precision']['riddle'])}, AMIE {_fmt(m['rule_precision']['amie'])}"
                      for key, m in rows) + "."]
        L.append("")
    timing = summary.get("job_cpu_seconds_by_part")
    if timing:
        L += ["CPU seconds by part (sum over jobs): " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items())
              + (f"; wall {summary['wall_seconds']:.0f} s." if summary.get("wall_seconds") else "."), ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    """Re-render a report from a run folder's rows: `--baselines RUN_DIR` (an `e10_baselines` run)."""
    import argparse
    import json
    from pathlib import Path

    import yaml
    parser = argparse.ArgumentParser(description="E10 report from a run folder's metrics.jsonl")
    parser.add_argument("--baselines", type=Path, required=True, help="an e10_baselines run folder")
    parser.add_argument("--output", type=Path, default=None, help="markdown file (default: print)")
    args = parser.parse_args(argv)
    config = yaml.safe_load((args.baselines / "resolved_config.yaml").read_text())
    rows = [json.loads(line) for line in (args.baselines / "metrics.jsonl").read_text().splitlines() if line.strip()]
    summary = summarize_baselines(rows, config)
    stored = args.baselines / "summary.json"
    if stored.exists():
        old = json.loads(stored.read_text())
        for key in ("job_cpu_seconds_by_part", "wall_seconds"):
            if key in old:
                summary[key] = old[key]
    text = render_baselines(summary, config)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
