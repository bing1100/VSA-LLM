"""E12 — the stage report: Q1 (recall tool), F1 (faithfulness of decoding) and the secondaries across a stage's runs
(author decision 62; pre-registration `experiments/e12-self-query/preregistration.md` §§4, 6, 8).

Reads `RUN/self-query-<items>/summary.json` (`e12_self_query evaluate`) and `RUN/self-query-faithfulness/summary.json`
(`e12_faithfulness evaluate`) of every run of a stage folder and computes, per host size:

- **Q1** — role-swap twin contrast accuracy (`choice`), on the C5 host: `recall:own − none` and `recall:own − recall:C5ut`,
  twin pairs × seeds crossed random-effects model (`statistics.crossed_components`, Satterthwaite t, two-sided; one seed: a
  one-sample t), Holm over the two; the two-way cluster bootstrap as the check.
- **F1** — comprehensiveness net share of the decoded edges (τ = 0.05), per new word: **F1a** C5 against 0 and **F1b**
  C5 − C5ut, terms × seeds, Holm over the two.
- secondaries: every host × condition (mean contrast / accuracy with the decode accuracy of what the condition put in
  context), `symbolic − recall`, `definition − symbolic`, `wrong` (the flip), the hosts without a store (C0′, C2, P0 + the C5
  store), recall on fully decoded pairs only, cloze items; natural T4, WP-UB two-hop / reverse and new-word items by
  condition; F1's sufficiency, specificity, role specificity (twins), τ sensitivity and undecoded edges.

Batch reports write their own folder (`experiments/e12-self-query/report/<stage>-<tag>`), never an E9 `report/<stage>`.

    python -m vsa_embed.experiments.e12_report --runs experiments/e9-retrofit/runs/t5 --output experiments/e12-self-query/report/t5-phase-a \\
        [--twins role-twins-t5-smollm2-v1] [--natural …] [--understanding …] [--new-words …] [--hosts SmolLM2-360M] [--label PILOT]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import holm_adjust
from .e5_common import finish_output, json_ready, start_output, write_json
from .e9_power import analyse_table, paired_table
from .e12_faithfulness import OUTPUT as FAITH_OUTPUT
from .e12_self_query import OUTPUT_PREFIX, RUN_NAME

PRIMARY_HOST_MODEL = "C5"
Q1_CANDIDATE = "recall:own"
Q1_REFERENCES = ("none", "recall:C5ut")
F1_REFERENCE = "C5ut"
TAU_KEY = "comprehensiveness"


def _parts(run: Path, base: str) -> list[dict[str, Any]]:
    """The summary documents of `run/<base>` and of its tagged parts `run/<base>-<tag>` (a job split by conditions or by term
    range), in name order."""
    folders = [run / base] + sorted(p for p in run.glob(f"{base}-*") if p.is_dir())
    return [json.loads((f / "summary.json").read_text()) for f in folders if (f / "summary.json").exists()]


def merge_phase_a(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One phase-A document from parts that scored different conditions of the same item set (later parts win)."""
    merged = copy.deepcopy(documents[0])
    for doc in documents[1:]:
        merged["summary"]["conditions"].update(copy.deepcopy(doc["summary"]["conditions"]))
    return merged


def merge_faithfulness(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One F1 document from parts that scored different term ranges (and, once, the twins)."""
    merged = copy.deepcopy(documents[0])
    summary = merged["summary"]
    for doc in documents[1:]:
        part = doc["summary"]
        if "new_words" in part:
            if "new_words" not in summary:
                summary["new_words"] = copy.deepcopy(part["new_words"])
            else:
                for block in ("terms", "undecoded", "all_edges"):
                    summary["new_words"][block].update(part["new_words"][block])
                for key in ("edges", "decoded_edges", "linked"):
                    summary["new_words"][key] = summary["new_words"].get(key, 0) + part["new_words"].get(key, 0)
        if "twins" in part:
            if "twins" not in summary:
                summary["twins"] = copy.deepcopy(part["twins"])
            else:
                summary["twins"]["pairs"].update(part["twins"]["pairs"]); summary["twins"]["swap"].update(part["twins"]["swap"])
    if "new_words" in summary:
        terms = summary["new_words"]["terms"]
        keys = next(iter(terms.values()), {}).keys()
        summary["new_words"]["mean"] = {k: float(np.mean([t[k] for t in terms.values()])) for k in keys}
    return merged


def discover(runs_root: Path, *, hosts: Sequence[str] | None = None, sets: Sequence[str] = ()) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """host → model → seed → {set name: phase-A summary document, "faithfulness": F1 summary document} (split jobs merged)."""
    out: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(Path(runs_root).iterdir()):
        match = RUN_NAME.match(run.name)
        if match is None or (hosts and match["host"] not in hosts):
            continue
        record: dict[str, Any] = {}
        for name in sets:
            parts = _parts(run, f"{OUTPUT_PREFIX}{name}")
            if parts:
                record[name] = merge_phase_a(parts)
        parts = _parts(run, FAITH_OUTPUT)
        if parts:
            record["faithfulness"] = merge_faithfulness(parts)
        if record:
            out[match["host"]][match["model"]][int(match["seed"])] = record
    return {h: dict(m) for h, m in out.items()}


def _units(document: dict[str, Any] | None, condition: str, block: str, metric: str) -> dict[str, float] | None:
    if document is None:
        return None
    found = document["summary"]["conditions"].get(condition, {}).get(block)
    if not found:
        return None
    return {k: float(v[metric]) for k, v in found["units"].items() if isinstance(v.get(metric), (int, float))}


def seeds_of(models: dict[str, dict[int, dict[str, Any]]], model: str, set_name: str, condition: str, block: str = "choice",
             metric: str = "contrast") -> dict[int, dict[str, float]]:
    """seed → unit → value of one host model, item set, condition and metric. P0 (one run) reads `recall:C5@<seed>` as the
    seed-`seed` value of `recall:C5` and repeats its store-free conditions for every seed."""
    out: dict[int, dict[str, float]] = {}
    for seed, record in models.get(model, {}).items():
        units = _units(record.get(set_name), condition, block, metric)
        if units:
            out[seed] = units
    if model == "P0" and models.get("P0"):
        record = next(iter(models["P0"].values())).get(set_name)
        if record is not None:
            stores = [c for c in record["summary"]["conditions"] if re.fullmatch(r"recall:C5@\d+", c)]
            if condition == "recall:C5" and stores:
                out = {int(c.split("@")[1]): u for c in stores if (u := _units(record, c, block, metric))}
    return out


def contrast(candidate: dict[int, dict[str, float]], reference: dict[int, dict[str, float]] | None, *, resamples: int = 2000,
             seed: int = 0) -> dict[str, Any]:
    """Candidate − reference over units × seeds (crossed model, two-way bootstrap); a single-run reference stands for
    every seed of the candidate."""
    if reference is not None and len(reference) == 1 and len(candidate) > 1:
        only = next(iter(reference.values()))
        reference = {s: only for s in candidate}
    if candidate and reference is not None and len(candidate) == 1 and len(reference) > 1:
        only = next(iter(candidate.values()))
        candidate = {s: only for s in reference}
    units, seeds, table = paired_table(candidate, reference)
    if len(units) < 2:
        return {"available": False, "units": len(units), "seeds": seeds}
    return {"available": True, "units": len(units), "seeds": seeds, **analyse_table(table, resamples=resamples, seed=seed)}


def _holm(results: dict[str, dict[str, Any]]) -> None:
    names = [n for n, r in results.items() if r.get("available")]
    for name, adjusted in zip(names, holm_adjust([results[n]["model"]["p_value"] for n in names]) if names else []):
        results[name]["p_holm"] = adjusted


def margin_test(result: dict[str, Any], margin: float, *, kind: str) -> dict[str, Any]:
    """A margin reading of one crossed-model contrast (amendment 16.5), from its mean, SE and Satterthwaite df:
    `equivalence` — two one-sided t tests at α = 0.05 (the 90% CI inside ±margin); `noninferiority` — the one-sided t test
    of mean > −margin at α = 0.025 (the 95% CI's lower bound above −margin). Adds `margin` = {kind, margin, p, shown,
    ci90_low, ci90_high} to `result` (in place) and returns it; an unavailable contrast is left unchanged."""
    from scipy import stats
    if not result.get("available"):
        return result
    m = result["model"]
    mean, se, df = float(m["mean"]), float(m.get("se") or 0.0), float(m.get("df") or float("inf"))
    t = (lambda q: float(stats.t.ppf(q, df))) if math.isfinite(df) else (lambda q: float(stats.norm.ppf(q)))
    sf = (lambda x: float(stats.t.sf(x, df))) if math.isfinite(df) else (lambda x: float(stats.norm.sf(x)))
    if kind == "equivalence":
        p = max(sf((mean + margin) / se), sf((margin - mean) / se)) if se > 0 else float(abs(mean) >= margin)
        shown, alpha = p < 0.05, 0.05
    elif kind == "noninferiority":
        p = sf((mean + margin) / se) if se > 0 else float(mean <= -margin)
        shown, alpha = p < 0.025, 0.025
    else:
        raise ValueError(f"unknown margin test {kind!r}")
    half = t(0.95) * se
    result["margin"] = {"kind": kind, "margin": margin, "alpha": alpha, "p": p, "shown": bool(shown), "ci90_low": mean - half,
                        "ci90_high": mean + half}
    return result


def margin_reading(result: dict[str, Any] | None) -> str:
    """`≈` (equivalent within the margin), `non-inferior`, `differs` (95% CI excludes 0) or `inconclusive`."""
    if not result or not result.get("available"):
        return "not available"
    m, test = result["model"], result.get("margin") or {}
    excludes = m["ci_low"] > 0 or m["ci_high"] < 0
    if test.get("kind") == "equivalence":
        if test["shown"]:
            return f"≈ (90% CI within ±{test['margin']:g})"
        return "differs (95% CI excludes 0)" if excludes else f"inconclusive (neither within ±{test['margin']:g} nor different from 0)"
    if test.get("kind") == "noninferiority":
        if test["shown"]:
            return f"non-inferior at −{test['margin']:g}" + (" and superior (95% CI above 0)" if m["ci_low"] > 0 else "")
        if m["ci_high"] < -test["margin"]:
            return f"inferior by more than the margin (95% CI below −{test['margin']:g})"
        return f"non-inferiority not shown (95% CI lower bound ≤ −{test['margin']:g})"
    return "differs (95% CI excludes 0)" if excludes else "no difference shown"


def _mean_by_seed(values: dict[int, dict[str, float]]) -> float | None:
    means = [float(np.mean(list(v.values()))) for v in values.values() if v]
    return float(np.mean(means)) if means else None


# ---------------------------------------------------------------- Q1


def q1(models: dict[str, dict[int, dict[str, Any]]], twins: str, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    candidate = seeds_of(models, PRIMARY_HOST_MODEL, twins, Q1_CANDIDATE)
    out: dict[str, Any] = {"candidate": f"{PRIMARY_HOST_MODEL} {Q1_CANDIDATE}", "contrasts": {}}
    if not candidate:
        out["available"] = False
        return out
    for reference in Q1_REFERENCES:
        ref = seeds_of(models, PRIMARY_HOST_MODEL, twins, reference)
        if ref:
            out["contrasts"][f"{Q1_CANDIDATE} − {reference}"] = contrast(candidate, ref, resamples=resamples, seed=seed)
    _holm(out["contrasts"])
    out["available"] = any(r.get("available") for r in out["contrasts"].values())
    symbolic = seeds_of(models, PRIMARY_HOST_MODEL, twins, "symbolic")
    out["means"] = {c: _mean_by_seed(seeds_of(models, PRIMARY_HOST_MODEL, twins, c)) for c in (Q1_CANDIDATE, *Q1_REFERENCES, "symbolic")}
    out["decode"] = _mean_by_seed(seeds_of(models, PRIMARY_HOST_MODEL, twins, Q1_CANDIDATE, metric="decode_all"))
    out["reading"] = q1_reading(out, symbolic_mean=_mean_by_seed(symbolic))
    return out


def q1_reading(result: dict[str, Any], *, symbolic_mean: float | None) -> str:
    rows = result["contrasts"]
    significant = {name: r.get("available") and r.get("p_holm", 1.0) < 0.05 and r["model"]["mean"] > 0 for name, r in rows.items()}
    negative = any(r.get("available") and r.get("p_holm", 1.0) < 0.05 and r["model"]["mean"] < 0 for r in rows.values())
    if symbolic_mean is not None and symbolic_mean < 0.6 and not any(significant.values()):
        return "(e) uninformative: the host cannot use even a gold frame in context (symbolic ≤ 0.6)"
    if negative:
        return "(d) recall hurts: a significant negative contrast"
    if len(significant) == len(Q1_REFERENCES) and all(significant.values()):
        return "(a) the self-query reads the stored roles: recall from the bound store beats no recall and the role-blind store"
    if any(significant.values()):
        return "(b) partial: one of the two contrasts is significant and positive"
    return "(c) no gain from recall"


# ---------------------------------------------------------------- F1


def f1_units(models: dict[str, dict[int, dict[str, Any]]], model: str, key: str = TAU_KEY, block: str = "terms") -> dict[int, dict[str, float]]:
    out = {}
    for seed, record in models.get(model, {}).items():
        doc = record.get("faithfulness")
        if doc is None or "new_words" not in doc["summary"]:
            continue
        out[seed] = {k: float(v[key]) for k, v in doc["summary"]["new_words"][block].items() if v.get(key) is not None}
    return out


def twin_units(models: dict[str, dict[int, dict[str, Any]]], model: str, key: str = "role_specificity") -> dict[int, dict[str, float]]:
    out = {}
    for seed, record in models.get(model, {}).items():
        doc = record.get("faithfulness")
        if doc is None or "twins" not in doc["summary"]:
            continue
        out[seed] = {k: float(v[key]) for k, v in doc["summary"]["twins"]["pairs"].items()}
    return out


def f1(models: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    c5 = f1_units(models, PRIMARY_HOST_MODEL)
    out: dict[str, Any] = {"contrasts": {}}
    if not c5:
        out["available"] = False
        return out
    out["contrasts"]["F1a: C5 − 0"] = contrast(c5, None, resamples=resamples, seed=seed)
    reference = f1_units(models, F1_REFERENCE)
    if reference:
        out["contrasts"][f"F1b: C5 − {F1_REFERENCE}"] = contrast(c5, reference, resamples=resamples, seed=seed)
    _holm(out["contrasts"])
    out["available"] = True
    out["secondaries"] = {}
    for key in ("moved", "sufficiency", "specificity", "gap", "comprehensiveness_tau0", "comprehensiveness_tau0.02", "comprehensiveness_tau0.1"):
        for model in (PRIMARY_HOST_MODEL, F1_REFERENCE):
            values = f1_units(models, model, key)
            if values:
                out["secondaries"][f"{model} {key}"] = contrast(values, None, resamples=resamples, seed=seed)
    undecoded = f1_units(models, PRIMARY_HOST_MODEL, block="undecoded")
    if undecoded:
        out["secondaries"]["C5 comprehensiveness, undecoded edges"] = contrast(undecoded, None, resamples=resamples, seed=seed)
    for model in (PRIMARY_HOST_MODEL, F1_REFERENCE):
        role = twin_units(models, model)
        if role:
            out["secondaries"][f"{model} role specificity (twins)"] = contrast(role, None, resamples=resamples, seed=seed)
    c5_role, ref_role = twin_units(models, PRIMARY_HOST_MODEL), twin_units(models, F1_REFERENCE)
    if c5_role and ref_role:
        out["secondaries"][f"role specificity C5 − {F1_REFERENCE}"] = contrast(c5_role, ref_role, resamples=resamples, seed=seed)
    out["reading"] = f1_reading(out)
    return out


def f1_table(models: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per composing model: seed-averaged term means of the F1 measures and the twins' role specificity."""
    rows = []
    for model in sorted(models):
        row: dict[str, Any] = {"model": model}
        for key in ("comprehensiveness", "moved", "sufficiency", "specificity", "gap"):
            row[key] = _mean_by_seed(f1_units(models, model, key))
        row["decoded_edges"] = _mean_by_seed({s: {"n": float(r["faithfulness"]["summary"]["new_words"]["decoded_edges"])}
                                              for s, r in models[model].items() if "faithfulness" in r and "new_words" in r["faithfulness"]["summary"]})
        row["role_specificity"] = _mean_by_seed(twin_units(models, model))
        row["rs"] = _mean_by_seed(twin_units(models, model, "rs"))
        row["seeds"] = sorted(s for s, r in models[model].items() if "faithfulness" in r)
        if row["seeds"]:
            rows.append(row)
    return rows


def f1_reading(result: dict[str, Any]) -> dict[str, str]:
    rows = result["contrasts"]
    a = rows.get("F1a: C5 − 0", {})
    b = next((r for n, r in rows.items() if n.startswith("F1b")), {})
    out = {}
    if a.get("available"):
        if a.get("p_holm", 1.0) < 0.05 and a["model"]["mean"] > 0:
            out["F1a"] = "(a) decoded edges are causally used: removing one moves its filler more than removing another edge"
        elif a.get("p_holm", 1.0) < 0.05:
            out["F1a"] = "(c) anti-faithful: removing a decoded edge moves its filler less than removing another edge"
        else:
            out["F1a"] = "(b) not shown: decoded edges are not distinguishable from other edges by their effect"
    if b.get("available"):
        if b.get("p_holm", 1.0) < 0.05:
            out["F1b"] = f"binding changes edge-level faithfulness ({'C5 higher' if b['model']['mean'] > 0 else 'C5 lower'})"
        else:
            out["F1b"] = "no difference between the bound and the untyped store at the filler level (the channel route reads fillers)"
    return out


# ---------------------------------------------------------------- secondaries


def table(models: dict[str, dict[int, dict[str, Any]]], set_name: str, block: str, metrics: Sequence[str]) -> list[dict[str, Any]]:
    """Per host model × condition: the seed-averaged means of `metrics` over units, and the seeds."""
    rows = []
    for model in sorted(models):
        conditions: dict[str, None] = {}
        for record in models[model].values():
            doc = record.get(set_name)
            if doc:
                conditions.update(dict.fromkeys(doc["summary"]["conditions"]))
        for condition in conditions:
            row = {"model": model, "condition": condition}
            for metric in metrics:
                row[metric] = _mean_by_seed(seeds_of(models, model, set_name, condition, block, metric))
            row["seeds"] = sorted(seeds_of(models, model, set_name, condition, block, metrics[0]))
            if any(row.get(m) is not None for m in metrics):
                rows.append(row)
    return rows


def twin_secondaries(models: dict[str, dict[int, dict[str, Any]]], twins: str, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pairs = [("C5", "symbolic", "recall:own"), ("C5", "definition", "symbolic"), ("C5", "recall:own", "roleless:own"),
             ("C5", "recall:own", "wrong:own"), ("C5", "recall:own", "recall:C5tr"), ("C5ut", "recall:own", "none"),
             ("C5ut", "recall:C5", "recall:own"), ("C0p", "recall:C5", "none"), ("C2", "recall:C5", "none"), ("P0", "recall:C5", "none"),
             ("C0p", "recall:C5", "roleless:C5"), ("C5rf", "recall:own", "none"), ("C5tr", "recall:own", "none")]
    for model, a, b in pairs:
        ca, cb = seeds_of(models, model, twins, a), seeds_of(models, model, twins, b)
        if ca and cb:
            out[f"{model}: {a} − {b}"] = contrast(ca, cb, resamples=resamples, seed=seed)
    for model, a in (("C5", "recall:own"), ("C0p", "recall:C5"), ("P0", "recall:C5")):
        values = seeds_of(models, model, twins, a)
        if values:
            out[f"{model}: {a} − 0.5"] = contrast({s: {k: v - 0.5 for k, v in u.items()} for s, u in values.items()}, None,
                                                  resamples=resamples, seed=seed)
    cross = []
    for host in ("C5", "C0p", "P0"):
        a = seeds_of(models, PRIMARY_HOST_MODEL, twins, "recall:own")
        b = seeds_of(models, host, twins, "recall:C5")
        if host != "C5" and a and b:
            out[f"C5 recall:own − {host} recall:C5 (does the channel add to the tool?)"] = contrast(a, b, resamples=resamples, seed=seed)
            cross.append(host)
    # recall on fully decoded pairs only (decode_all = 1) and on the rest
    for subset in (1.0, 0.0):
        values: dict[int, dict[str, float]] = {}
        for seed_, record in models.get(PRIMARY_HOST_MODEL, {}).items():
            doc = record.get(twins)
            block = (doc or {}).get("summary", {}).get("conditions", {}).get("recall:own", {}).get("choice")
            if block:
                values[seed_] = {k: float(v["contrast"]) for k, v in block["units"].items() if float(v.get("decode_all", -1)) == subset}
        if any(values.values()):
            out[f"C5 recall:own on pairs with decode_all = {subset:g}"] = {"mean": _mean_by_seed(values),
                                                                           "pairs": int(np.mean([len(v) for v in values.values()]))}
    return out


# ---------------------------------------------------------------- the report


def analyse(runs_root: Path, *, hosts: Sequence[str] | None = None, twins: str | None = None, natural: str | None = None,
            understanding: str | None = None, new_words: str | None = None, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    sets = [s for s in (twins, natural, understanding, new_words) if s]
    found = discover(runs_root, hosts=hosts, sets=sets)
    analysis: dict[str, Any] = {"runs_root": str(runs_root), "sets": {"twins": twins, "natural": natural, "understanding": understanding,
                                                                      "new_words": new_words}, "hosts": {}}
    for host, models in sorted(found.items()):
        block: dict[str, Any] = {"runs": {m: sorted(s) for m, s in models.items()}}
        if twins:
            block["q1"] = q1(models, twins, resamples=resamples, seed=seed)
            block["twins_table"] = {kind: table(models, twins, kind, ("contrast", "item", "decode", "decode_all")) for kind in ("choice", "cloze")}
            block["twins_secondaries"] = twin_secondaries(models, twins, resamples=resamples, seed=seed)
        block["f1"] = f1(models, resamples=resamples, seed=seed)
        block["f1_table"] = f1_table(models)
        if natural:
            block["natural_table"] = {kind: table(models, natural, kind, ("contrast", "item", "decode")) for kind in ("choice", "cloze")}
        if understanding:
            block["understanding"] = understanding_table(models, understanding)
        if new_words:
            block["new_words_table"] = table(models, new_words, "property", ("accuracy", "decode"))
        analysis["hosts"][host] = block
    return analysis


def understanding_table(models: dict[str, dict[int, dict[str, Any]]], set_name: str) -> list[dict[str, Any]]:
    rows = []
    for model in sorted(models):
        per: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in models[model].values():
            doc = record.get(set_name)
            if not doc:
                continue
            for condition, block in doc["summary"]["conditions"].items():
                for group, m in block["items"]["by"].items():
                    per[(condition, group)].append(m)
        for (condition, group), ms in sorted(per.items()):
            row = {"model": model, "condition": condition, "group": group, "seeds": len(ms)}
            for key in ("accuracy", "score", "hop1_correct", "bridge_correct", "hop2_correct", "pair_correct", "anchor_in_top", "slot_correct", "n"):
                vals = [m[key] for m in ms if key in m]
                if vals:
                    row[key] = float(np.mean(vals))
            rows.append(row)
    return rows


def _cell(result: dict[str, Any] | None) -> str:
    if not result or not result.get("available", "mean" in (result or {})):
        return "—"
    if "model" not in result:
        return f"{result.get('mean', float('nan')):.3f}" + (f" ({result['pairs']} pairs)" if "pairs" in result else "")
    m = result["model"]
    holm = f", Holm {result['p_holm']:.3g}" if "p_holm" in result else ""
    return f"{m['mean']:+.4f} [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] (p {m['p_value']:.3g}{holm}; {result['units']} units × {len(result['seeds'])} seeds)"


def render(analysis: dict[str, Any], *, title: str, label: str | None = None) -> str:
    lines = [f"# {title}" + (f" — {label}" if label else ""), "",
             "Pre-registration: `experiments/e12-self-query/preregistration.md`. Contrasts: units × seeds crossed model "
             "(Satterthwaite t, 95% CI, two-sided p; one seed: one-sample t), Holm within Q1 and within F1.", ""]
    if label:
        lines += [f"**{label}: these numbers do not count toward the pre-registered endpoints.**", ""]
    for host, block in analysis["hosts"].items():
        lines += [f"## {host}", "", f"Runs: {block['runs']}", ""]
        q = block.get("q1")
        if q and q.get("available"):
            lines += ["### Q1 — recall tool on role-swap twins (C5 host, contrast accuracy, choice)", "", "| contrast | estimate |", "|---|---|"]
            lines += [f"| {name} | {_cell(r)} |" for name, r in q["contrasts"].items()]
            means = ", ".join(f"{c} {v:.3f}" for c, v in q["means"].items() if v is not None)
            lines += ["", f"Means: {means}; pairs fully decoded by the C5 recall: {q['decode']:.3f}." if q.get("decode") is not None
                      else f"Means: {means}.", "", f"Reading: {q['reading']}", ""]
        for kind, rows in (block.get("twins_table") or {}).items():
            if rows:
                lines += [f"### Twins by host × condition ({kind})", "", "| host model | condition | contrast | item | decode (filler) | all four slots | seeds |",
                          "|---|---|---:|---:|---:|---:|---|"]
                lines += [f"| {r['model']} | {r['condition']} | {_f(r.get('contrast'))} | {_f(r.get('item'))} | {_f(r.get('decode'))} | "
                          f"{_f(r.get('decode_all'))} | {r['seeds']} |" for r in rows]
                lines.append("")
        sec = block.get("twins_secondaries")
        if sec:
            lines += ["### Twin secondaries", "", "| contrast | estimate |", "|---|---|"] + [f"| {n} | {_cell(r)} |" for n, r in sec.items()] + [""]
        f = block.get("f1")
        if f and f.get("available"):
            lines += ["### F1 — faithfulness of decoding (channel route; new words; comprehensiveness net share, τ = 0.05)", "",
                      "| contrast | estimate |", "|---|---|"] + [f"| {n} | {_cell(r)} |" for n, r in f["contrasts"].items()]
            lines += ["", "| secondary | estimate |", "|---|---|"] + [f"| {n} | {_cell(r)} |" for n, r in f["secondaries"].items()]
            lines += ["", f"Reading: {f['reading']}", ""]
        if block.get("f1_table"):
            lines += ["### F1 by model (seed-averaged term means; twins: role specificity per pair)", "",
                      "| model | comprehensiveness | moved | sufficiency | specificity | gap (nats) | decoded edges | role specificity | RS (nats) | seeds |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            lines += [f"| {r['model']} | {_f(r['comprehensiveness'], signed=True)} | {_f(r['moved'])} | {_f(r['sufficiency'], signed=True)} | "
                      f"{_f(r['specificity'])} | {_f(r['gap'], signed=True)} | {_f(r['decoded_edges'])} | {_f(r['role_specificity'], signed=True)} | "
                      f"{_f(r['rs'], signed=True)} | {r['seeds']} |" for r in block["f1_table"]]
            lines.append("")
        for name in ("natural_table",):
            for kind, rows in (block.get(name) or {}).items():
                if rows:
                    lines += [f"### Natural role items ({kind})", "", "| host model | condition | contrast | item | decode | seeds |", "|---|---|---:|---:|---:|---|"]
                    lines += [f"| {r['model']} | {r['condition']} | {_f(r.get('contrast'))} | {_f(r.get('item'))} | {_f(r.get('decode'))} | {r['seeds']} |" for r in rows]
                    lines.append("")
        if block.get("understanding"):
            families = sorted({r["group"].split("/")[0] for r in block["understanding"]})
            heading = ("WP-UB two-hop and reverse items" if set(families) <= {"two_hop", "reverse"}       # opt-in relation families
                       else "WP-UB items (" + ", ".join(families) + ")")
            lines += [f"### {heading}", "", "| host model | condition | family / subset | accuracy | − chance | hop 1 | bridge | hop 2 | pair | seeds |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| {r['model']} | {r['condition']} | {r['group']} | {_f(r.get('accuracy'))} | {_f(r.get('score'), signed=True)} | "
                      f"{_f(r.get('hop1_correct'))} | {_f(r.get('bridge_correct'))} | {_f(r.get('hop2_correct'))} | {_f(r.get('pair_correct'))} | {r['seeds']} |"
                      for r in block["understanding"]]
            lines.append("")
        if block.get("new_words_table"):
            lines += ["### New words (property items)", "", "| host model | condition | accuracy | decode | seeds |", "|---|---|---:|---:|---|"]
            lines += [f"| {r['model']} | {r['condition']} | {_f(r.get('accuracy'))} | {_f(r.get('decode'))} | {r['seeds']} |" for r in block["new_words_table"]]
            lines.append("")
    return "\n".join(lines) + "\n"


def _f(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--twins", default=None); parser.add_argument("--natural", default=None)
    parser.add_argument("--understanding", default=None); parser.add_argument("--new-words", default=None)
    parser.add_argument("--hosts", nargs="*", default=None); parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--title", default="E12 — self-query")
    parser.add_argument("--label", default=None, help="e.g. PILOT: printed on the report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.name == Path(args.runs).name:
        raise ValueError(f"{args.output}: a batch report writes its own folder (`<stage>-<tag>`), never the stage's base report "
                         "folder (amendment 12.4)")
    config = {"experiment": "e12-report", "runs": str(args.runs), "twins": args.twins, "natural": args.natural,
              "understanding": args.understanding, "new_words": args.new_words, "hosts": args.hosts, "label": args.label}
    if args.overwrite:
        for name in ("analysis.json", "report.md", "resolved_config.yaml", "manifest.json"):
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = start_output(args.output, config)
    started = time.monotonic()
    analysis = analyse(args.runs, hosts=args.hosts, twins=args.twins, natural=args.natural, understanding=args.understanding,
                       new_words=args.new_words, resamples=args.resamples, seed=args.seed)
    write_json(args.output / "analysis.json", json_ready(analysis))
    (args.output / "report.md").write_text(render(analysis, title=args.title, label=args.label))
    finish_output(args.output, config, git_at_start=git_at_start, device="cpu", seconds=round(time.monotonic() - started, 1))
    print(json.dumps({"output": str(args.output), "hosts": list(analysis["hosts"])}))


if __name__ == "__main__":
    main()
