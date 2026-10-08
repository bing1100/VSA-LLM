"""E9 binding and unbinding program: analysis across runs and the stage report (decision 60; pre-registration
`experiments/e9-retrofit/preregistration-binding.md` §6–7).

Reads every run folder of a stage (`RUN/binding-probe` from `e9_binding_probe`, `RUN/role-*` from `e9_binding_items`)
and computes:

- **P1** — role-swap twin contrast accuracy (`choice` items, source `own`), C5 − C5ut and C5 − C5tr: pairs × seeds
  crossed random-effects model (`statistics.crossed_components`, Satterthwaite t, two-sided), Holm over the two; the
  two-way cluster bootstrap as the check;
- **P2** — algebraic filler recovery on held-out concepts (probe A, `static` bundle, `typed` cleanup, the operator's
  primary unbinding, MRR per concept): C5 − the frequency baseline, and C5 − C5rf; concepts × seeds crossed model;
- secondaries: every arm against C5 on the twins, the natural role items, probe A by condition, frame size (the
  capacity curve) and role ambiguity, role recovery, probe B (LRE) against probe A, the hidden-state probes.

    python -m vsa_embed.experiments.e9_binding_report --runs experiments/e9-retrofit/runs/t5 \\
        --output experiments/e9-retrofit/report/t5-binding [--twins role-twins-t5-smollm2-v1] [--natural role-natural-t5-smollm2-v1]
        [--hosts SmolLM2-360M] [--licensed] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import holm_adjust
from . import e9_binding_items as items_mod
from . import e9_binding_probe as probe_mod
from .e5_common import clear_output, finish_output, json_ready, start_output, write_json
from .e9_power import analyse_table, paired_table

CANDIDATE = "C5"
P1_REFERENCES = ("C5ut", "C5tr")
P2_REFERENCE = "C5rf"
ARMS_ORDER = ("C5", "C5rf", "C5ut", "C5tr", "C5sh", "C6m", "C6d", "C6g", "C2", "C0p", "P0")
NAME = re.compile(r"^(?P<host>.+)-(?P<mode>full|lora|frozen)-(?P<model>[^-]+)-s(?P<seed>\d+)$")


def discover(runs_root: Path, *, hosts: Sequence[str] | None = None, twins: str | None = None, natural: str | None = None
             ) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    """host → model → seed → {"path", "probe": load_probe(...), "readout": readout evaluation, "twins": evaluation,
    "natural": evaluation}."""
    from .e9_binding_readout import OUTPUT as READOUT_OUTPUT, load_evaluation as load_readout
    found: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        match = NAME.match(run.name)
        if not match or (hosts and match["host"] not in hosts):
            continue
        entry = {"path": run, "probe": probe_mod.load_probe(run / probe_mod.OUTPUT), "readout": load_readout(run / READOUT_OUTPUT)}
        if twins:
            entry["twins"] = items_mod.load_evaluation(items_mod.output_folder(run, twins))
        if natural:
            entry["natural"] = items_mod.load_evaluation(items_mod.output_folder(run, natural))
        found[match["host"]][match["model"]][int(match["seed"])] = entry
    return found


def _by_seed(models: dict[str, dict[int, dict[str, Any]]], model: str, field: str) -> dict[int, Any]:
    return {s: v[field] for s, v in models.get(model, {}).items() if v.get(field) is not None}


# ---------------------------------------------------------------- P1 and the item secondaries


def item_units(evaluations: dict[int, dict[str, Any]], *, source: str = "own", kind: str = "choice", metric: str = "contrast"
               ) -> dict[int, dict[str, float]]:
    """seed → unit → value (unit keys as strings)."""
    out = {}
    for seed, evaluation in evaluations.items():
        units = items_mod.unit_scores(evaluation, source, kind)
        if units:
            out[seed] = {str(k): float(v[metric]) for k, v in units.items()}
    return out


def _single(seeds_a: dict[int, Any], seeds_b: dict[int, Any]) -> dict[int, Any]:
    """P0 has one run (no training randomness): it stands for every seed of the other model."""
    if len(seeds_b) == 1 and len(seeds_a) > 1:
        only = next(iter(seeds_b.values()))
        return {s: only for s in seeds_a}
    return seeds_b


def contrast(candidate: dict[int, dict[str, float]], reference: dict[int, dict[str, float]] | None, *, resamples: int, seed: int
             ) -> dict[str, Any]:
    """Candidate − reference (or candidate − 0) over units × seeds: crossed model and two-way bootstrap."""
    if reference is not None:
        reference = _single(candidate, reference)
    units, seeds, table = paired_table(candidate, reference)
    if len(units) < 2:
        return {"available": False, "units": len(units), "seeds": seeds}
    result = analyse_table(table, resamples=resamples, seed=seed)
    return {"available": True, "units": len(units), "seeds": seeds, **result}


def p1(models: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P1: twin contrast accuracy, C5 − C5ut and C5 − C5tr (Holm over the two)."""
    twins = {m: item_units(_by_seed(models, m, "twins")) for m in models}
    out: dict[str, Any] = {"means": {m: _seed_mean(v) for m, v in twins.items() if v}, "contrasts": {}}
    if not twins.get(CANDIDATE):
        out["available"] = False
        return out
    rows = []
    for reference in P1_REFERENCES:
        if twins.get(reference):
            result = contrast(twins[CANDIDATE], twins[reference], resamples=resamples, seed=seed)
            out["contrasts"][f"{CANDIDATE} − {reference}"] = result
            if result.get("available"):
                rows.append((f"{CANDIDATE} − {reference}", result["model"]["p_value"]))
    for (name, _), adjusted in zip(rows, holm_adjust([p for _, p in rows]) if rows else []):
        out["contrasts"][name]["p_holm"] = adjusted
    out["available"] = bool(rows)
    out["reading"] = p1_reading(out)
    return out


def p1_reading(result: dict[str, Any]) -> str:
    rows = [v for v in result["contrasts"].values() if v.get("available")]
    if len(rows) < len(P1_REFERENCES):
        return "incomplete"
    significant = [v["model"]["mean"] > 0 and v.get("p_holm", 1.0) < 0.05 for v in rows]
    if all(significant):
        return "(a) binding matters where a role must be read out: C5 beats both role-blind operators"
    if any(significant):
        return "(b) partial: C5 beats one role-blind operator only"
    if any(v["model"]["mean"] < 0 and v.get("p_holm", 1.0) < 0.05 for v in rows):
        return "(d) a role-blind operator beats learned binding on the twins"
    return "(c) no evidence that C5's binding is read out behaviourally"


def _seed_mean(values: dict[int, dict[str, float]]) -> dict[str, Any]:
    per_seed = {s: float(np.mean(list(v.values()))) for s, v in values.items() if v}
    return {"mean": float(np.mean(list(per_seed.values()))) if per_seed else None, "per_seed": per_seed}


def item_secondaries(models: dict[str, dict[int, dict[str, Any]]], field: str, *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per model, item kind and source the seed-averaged contrast accuracy; every model against C5 and against chance
    (Holm over the models within a kind)."""
    out: dict[str, Any] = {"means": {}, "vs_candidate": {}, "vs_chance": {}}
    for kind in items_mod.ITEM_KINDS:
        units = {m: item_units(_by_seed(models, m, field), kind=kind) for m in models}
        out["means"][kind] = {m: _seed_mean(v) for m, v in units.items() if v}
        for source in ("none", "swap"):
            own = {m: item_units(_by_seed(models, m, field), kind=kind, source=source) for m in models}
            for m, v in own.items():
                if v:
                    out["means"][kind].setdefault(f"{m} [{source}]", _seed_mean(v))
        block = {}
        for model, values in units.items():
            if model == CANDIDATE or not values or not units.get(CANDIDATE):
                continue
            block[f"{CANDIDATE} − {model}"] = contrast(units[CANDIDATE], values, resamples=resamples, seed=seed)
        _holm(block)
        out["vs_candidate"][kind] = block
        chance = {}
        for model, values in units.items():
            if values:
                centred = {s: {k: v - 0.5 for k, v in d.items()} for s, d in values.items()}
                chance[f"{model} − 0.5"] = contrast(centred, None, resamples=resamples, seed=seed)
        _holm(chance)
        out["vs_chance"][kind] = chance
    return out


def _holm(block: dict[str, Any]) -> None:
    names = [k for k, v in block.items() if v.get("available")]
    for name, adjusted in zip(names, holm_adjust([block[k]["model"]["p_value"] for k in names]) if names else []):
        block[name]["p_holm"] = adjusted


# ---------------------------------------------------------------- P2 and the probe secondaries


def probe_entries(edges: dict[str, np.ndarray], key: str, *, subset: str = "heldout", prefix: str = "ae", mask: str | None = None
                  ) -> dict[str, float]:
    """entry → mean reciprocal rank of `key` over its edges of `subset` (optionally only the edges with flag `mask`)."""
    if edges is None or f"{key}.rr" not in edges:
        return {}
    owners = edges[f"{prefix}_entry"]
    position = {int(e): i for i, e in enumerate(edges["entries"].tolist())}
    subsets = edges["entry_subset"]
    edge_subset = np.asarray([subsets[position[int(e)]] for e in owners.tolist()])
    keep = np.isfinite(edges[f"{key}.rr"].astype(np.float64))
    if subset != "all":
        keep &= edge_subset == subset
    if mask:
        keep &= edges[f"{prefix}_{mask}"] if mask in ("ambiguous", "reuse", "multi") else ~edges[f"{prefix}_ambiguous"]
    values = probe_mod.entry_means(edges[f"{key}.rr"].astype(np.float64), owners, keep)
    return {str(e): v for e, v in values.items()}


def primary_probe_key(summary: dict[str, Any]) -> str | None:
    methods = (summary.get("algebraic") or {}).get("methods")
    return f"{probe_mod.primary_key(methods)}.typed" if methods else None


def probe_units(models: dict[str, dict[int, dict[str, Any]]], model: str, *, key: str | None = None, subset: str = "heldout",
                prefix: str = "ae", mask: str | None = None) -> dict[int, dict[str, float]]:
    out = {}
    for seed, found in _by_seed(models, model, "probe").items():
        edges = found["edges"]
        name = key or primary_probe_key(found["summary"])
        if edges is None or name is None:
            continue
        values = probe_entries(edges, name, subset=subset, prefix=prefix, mask=mask)
        if values:
            out[seed] = values
    return out


def p2(models: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """P2: held-out algebraic filler recovery (static, typed, primary unbinding; MRR per concept): C5 − frequency baseline
    and C5 − C5rf."""
    candidate = probe_units(models, CANDIDATE)
    out: dict[str, Any] = {"available": bool(candidate), "contrasts": {}}
    if not candidate:
        return out
    baseline = probe_units(models, CANDIDATE, key="b.frequency.filler.typed")
    chance = probe_units(models, CANDIDATE, key="b.chance.filler.typed")
    out["means"] = {"C5": _seed_mean(candidate), "frequency": _seed_mean(baseline), "chance": _seed_mean(chance)}
    out["contrasts"]["C5 − frequency"] = contrast(candidate, baseline, resamples=resamples, seed=seed)
    out["contrasts"]["C5 − chance"] = contrast(candidate, chance, resamples=resamples, seed=seed)
    reference = probe_units(models, P2_REFERENCE)
    if reference:
        out["means"][P2_REFERENCE] = _seed_mean(reference)
        out["contrasts"][f"C5 − {P2_REFERENCE}"] = contrast(candidate, reference, resamples=resamples, seed=seed)
    out["reading"] = p2_reading(out)
    return out


def p2_reading(result: dict[str, Any]) -> dict[str, str]:
    readings = {}
    freq = result["contrasts"].get("C5 − frequency", {})
    if freq.get("available"):
        m = freq["model"]
        readings["readable"] = ("(a) the learned-HRR store is algebraically readable beyond the frequency prior"
                                if m["mean"] > 0 and m["p_value"] < 0.05 else
                                "(b) not readable beyond the frequency prior" if m["p_value"] >= 0.05 else
                                "(c) worse than the frequency prior: training made the store unreadable")
    fixed = result["contrasts"].get(f"C5 − {P2_REFERENCE}", {})
    if fixed.get("available"):
        m = fixed["model"]
        readings["learned_vs_fixed"] = ("learned binding is more decodable than fixed random unitary binding"
                                        if m["mean"] > 0 and m["p_value"] < 0.05 else
                                        "learned binding is less decodable than fixed random unitary binding"
                                        if m["mean"] < 0 and m["p_value"] < 0.05 else "no difference between learned and fixed binding")
    return readings


def probe_secondaries(models: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Descriptive and paired secondaries from the per-run probe summaries and edges."""
    out: dict[str, Any] = {"algebraic": {}, "learned": {}, "ambiguous": {}, "lre_vs_algebraic": {}}
    for model in models:
        found = _by_seed(models, model, "probe")
        if not found:
            continue
        summaries = {s: f["summary"]["summary"] for s, f in found.items()}
        algebraic = {s: v.get("algebraic") for s, v in summaries.items() if v.get("algebraic")}
        if algebraic:
            out["algebraic"][model] = _average_blocks(algebraic)
        learned = {s: v.get("learned") for s, v in summaries.items() if v.get("learned")}
        if learned:
            out["learned"][model] = _average_blocks(learned)
    composing = [m for m in models if any((f["summary"].get("algebraic") or {}) for f in _by_seed(models, m, "probe").values())]
    candidate_amb = probe_units(models, CANDIDATE, mask="ambiguous", subset="all")
    for model in composing:
        if model == CANDIDATE:
            continue
        other = probe_units(models, model, mask="ambiguous", subset="all")
        if candidate_amb and other:
            out["ambiguous"][f"{CANDIDATE} − {model} (ambiguous fillers, all entries)"] = contrast(candidate_amb, other,
                                                                                                  resamples=resamples, seed=seed)
    for model in composing:
        algebraic = probe_units(models, model, subset="heldout")
        learned = probe_units(models, model, key="l.composed_static", subset="heldout", prefix="le")
        if algebraic and learned:
            out["lre_vs_algebraic"][f"{model}: LRE − algebraic (held-out)"] = contrast(learned, algebraic, resamples=resamples, seed=seed)
    return out


def _average_blocks(blocks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Seed-average of the per-run aggregates: key → subset → {mrr, top1}."""
    keys = sorted(set.intersection(*(set(b) for b in blocks.values())))
    out = {}
    for key in keys:
        out[key] = {}
        for subset in (*probe_mod.SUBSETS, "all"):
            values = [b[key]["subsets"].get(subset) for b in blocks.values()]
            values = [v for v in values if v]
            if values:
                out[key][subset] = {"mrr": float(np.mean([v["mrr"] for v in values])),
                                    "top1": float(np.mean([v["top1"] for v in values])) if all("top1" in v for v in values) else None,
                                    "seeds": len(values)}
        degree = defaultdict(list)
        for b in blocks.values():
            for name, v in (b[key].get("degree", {}).get("all") or {}).items():
                degree[name].append(v["mrr"])
        out[key]["degree_all"] = {k: float(np.mean(v)) for k, v in sorted(degree.items())}
        flags = defaultdict(list)
        for b in blocks.values():
            for name, v in (b[key].get("flags", {}).get("all") or {}).items():
                flags[name].append(v["mrr"])
        out[key]["flags_all"] = {k: float(np.mean(v)) for k, v in sorted(flags.items())}
    return out


# ---------------------------------------------------------------- step 2: the readout arms (pre-registration §13)

READOUT_CANDIDATE = "U5"
R1_REFERENCES = ("U5ut", "U5tr")
READOUT_FAMILIES = ("U5u", "U5sb", "U5bu", "U5sl")
R2_STRATUM = "after_heldout"


def window_losses(run: Path, *, readout: dict[str, Any] | None = None, variant: str | None = None) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """(strata, sums, counts) per evaluation window at the run's final evaluation (`eval_windows.npz`), or of a readout
    evaluation's `on` / `off` variant (`RUN/readout/windows.npz`)."""
    from ..training.lm import load_window_losses
    if variant is not None:
        arrays = (readout or {}).get("windows")
        if not arrays or f"sum_{variant}" not in arrays:
            return None
        return list(arrays["strata"].tolist()), arrays[f"sum_{variant}"], arrays[f"count_{variant}"]
    path = Path(run) / "eval_windows.npz"
    if not path.exists():
        return None
    saved = load_window_losses(path)
    if not saved["evals"]:
        return None
    sums, counts = saved["evals"][max(saved["evals"])]
    return saved["strata"], sums, counts


def loss_contrast(a: dict[int, tuple[list[str], np.ndarray, np.ndarray]], b: dict[int, tuple[list[str], np.ndarray, np.ndarray]],
                  stratum: str, *, resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Relative loss difference a − b on `stratum`: per evaluation window the paired difference summed over the common
    seeds, a window bootstrap (`statistics.paired_ratio_bootstrap`; the relative difference from the same resamples)."""
    from ..statistics import paired_ratio_bootstrap
    seeds = sorted(set(a) & set(b))
    if len(b) == 1 and len(a) > 1:                                  # a single reference run stands for every seed
        only = next(iter(b.values()))
        b = {s: only for s in a}
        seeds = sorted(a)
    if not seeds:
        return {"available": False}
    d = n = base = None
    for s in seeds:
        (strata_a, sums_a, counts_a), (strata_b, sums_b, counts_b) = a[s], b[s]
        if stratum not in strata_a or stratum not in strata_b:
            return {"available": False}
        ia, ib = strata_a.index(stratum), strata_b.index(stratum)
        if sums_a.shape[1] != sums_b.shape[1] or not np.array_equal(counts_a[ia], counts_b[ib]):
            return {"available": False, "detail": "different evaluation windows or masks"}
        d = (sums_a[ia] - sums_b[ib]) if d is None else d + (sums_a[ia] - sums_b[ib])
        n = counts_a[ia].astype(np.float64) if n is None else n + counts_a[ia]
        base = sums_b[ib].astype(np.float64) if base is None else base + sums_b[ib]
    result = paired_ratio_bootstrap(d, n, base, resamples=resamples, seed=seed)
    return {"available": True, "seeds": seeds, "stratum": stratum, **result}


def _losses(models: dict[str, dict[int, dict[str, Any]]], model: str, variant: str | None = None) -> dict[int, Any]:
    out = {}
    for seed, entry in models.get(model, {}).items():
        found = window_losses(entry["path"], readout=entry.get("readout"), variant=variant)
        if found is not None:
            out[seed] = found
    return out


def step2(models: dict[str, dict[int, dict[str, Any]]], *, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """R1 (twins: U5 − U5ut, U5 − U5tr; Holm), R2 (U5 − C5 relative loss on `after_heldout`) and the step-2 secondaries."""
    out: dict[str, Any] = {"available": READOUT_CANDIDATE in models}
    if not out["available"]:
        return out
    twins = {m: item_units(_by_seed(models, m, "twins")) for m in models}
    rows, contrasts = [], {}
    for reference in R1_REFERENCES:
        if twins.get(READOUT_CANDIDATE) and twins.get(reference):
            result = contrast(twins[READOUT_CANDIDATE], twins[reference], resamples=resamples, seed=seed)
            contrasts[f"{READOUT_CANDIDATE} − {reference}"] = result
            if result.get("available"):
                rows.append((f"{READOUT_CANDIDATE} − {reference}", result["model"]["p_value"]))
    for (name, _), adjusted in zip(rows, holm_adjust([p for _, p in rows]) if rows else []):
        contrasts[name]["p_holm"] = adjusted
    out["R1"] = {"contrasts": contrasts, "means": {m: _seed_mean(v) for m, v in twins.items() if v}}
    out["R2"] = loss_contrast(_losses(models, READOUT_CANDIDATE), _losses(models, CANDIDATE), R2_STRATUM, seed=seed)
    families = {}
    for model in (*READOUT_FAMILIES, CANDIDATE):
        reference = "U5ut" if model != CANDIDATE else None
        if model == CANDIDATE and twins.get(READOUT_CANDIDATE) and twins.get(CANDIDATE):
            families[f"{READOUT_CANDIDATE} − {CANDIDATE}"] = contrast(twins[READOUT_CANDIDATE], twins[CANDIDATE], resamples=resamples, seed=seed)
        elif reference and twins.get(model) and twins.get(reference):
            families[f"{model} − {reference}"] = contrast(twins[model], twins[reference], resamples=resamples, seed=seed)
    _holm(families)
    out["twins_secondary"] = families
    on_off, against_c5 = {}, {}
    for model in models:
        on, off = _losses(models, model, "on"), _losses(models, model, "off")
        if on and off:
            on_off[model] = {s: loss_contrast(off, on, s, seed=seed) for s in ("all", "after", R2_STRATUM)}
            against_c5[model] = {"on": loss_contrast(on, _losses(models, CANDIDATE), R2_STRATUM, seed=seed),
                                 "off": loss_contrast(off, _losses(models, CANDIDATE), R2_STRATUM, seed=seed)}
    out["off_minus_on"] = on_off
    out["against_C5"] = against_c5
    diagnostics = {}
    for model in models:
        blocks = [e["readout"]["summary"].get("diagnostics", {}).get("subsets", {}) for e in models[model].values()
                  if e.get("readout") and e["readout"].get("summary")]
        blocks = [b for b in blocks if b]
        if blocks:
            diagnostics[model] = {subset: {key: float(np.mean([b[subset][key] for b in blocks if subset in b]))
                                           for key in ("role_accuracy", "role_mass", "none_mass", "read_mrr", "oracle_mrr", "oracle_top1")}
                                  for subset in ("all", "heldout", "seen") if any(subset in b for b in blocks)}
    out["diagnostics"] = diagnostics
    out["reading"] = r_reading(out)
    return out


def r_reading(result: dict[str, Any]) -> dict[str, str]:
    readings = {}
    rows = [v for v in result["R1"]["contrasts"].values() if v.get("available")]
    if len(rows) == len(R1_REFERENCES):
        significant = [v["model"]["mean"] > 0 and v.get("p_holm", 1.0) < 0.05 for v in rows]
        negative = [v["model"]["mean"] < 0 and v.get("p_holm", 1.0) < 0.05 for v in rows]
        readings["R1"] = ("(a) binding matters once a role is read out" if all(significant) else "(b) partial" if any(significant)
                          else "(d) a role-blind readout beats binding" if any(negative)
                          else "(c) even an explicit readout trained by the LM loss does not use binding")
    else:
        readings["R1"] = "incomplete"
    r2 = result.get("R2", {})
    if r2.get("available"):
        lo, hi = r2.get("relative_ci_low"), r2.get("relative_ci_high")
        readings["R2"] = ("(a) the readout lowers held-out loss" if hi is not None and hi < 0 else
                          "(c) the readout hurts held-out loss" if lo is not None and lo > 0 else "(b) no evidence either way")
    return readings


# ---------------------------------------------------------------- report


def analyse(runs_root: Path, *, hosts: Sequence[str] | None = None, twins: str | None = None, natural: str | None = None,
            resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    found = discover(runs_root, hosts=hosts, twins=twins, natural=natural)
    out: dict[str, Any] = {"runs": str(runs_root), "hosts": {}}
    for host, models in sorted(found.items()):
        block = {"runs": {m: sorted(v) for m, v in models.items()},
                 "P1": p1(models, resamples=resamples, seed=seed) if twins else {"available": False},
                 "P2": p2(models, resamples=resamples, seed=seed), "probes": probe_secondaries(models, resamples=resamples, seed=seed)}
        if twins:
            block["twins"] = item_secondaries(models, "twins", resamples=resamples, seed=seed)
        if natural:
            block["natural"] = item_secondaries(models, "natural", resamples=resamples, seed=seed)
        block["step2"] = step2(models, resamples=resamples, seed=seed)
        out["hosts"][host] = block
    return out


def _cell(result: dict[str, Any] | None) -> str:
    if not result or not result.get("available"):
        return "n/a"
    m = result["model"]
    holm = result.get("p_holm")
    return (f"{m['mean']:+.4f} [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] (p {m['p_value']:.3g}"
            + (f", Holm {holm:.3g}" if holm is not None else "") + f"; {result['units']} units × {len(result['seeds'])} seeds)")


def render(analysis: dict[str, Any], *, title: str) -> str:
    lines = [f"# {title}", "", "Pre-registration: `experiments/e9-retrofit/preregistration-binding.md`. Contrasts: units × seeds "
             "crossed model (Satterthwaite t; 95% CI; two-sided p).", ""]
    for host, block in analysis["hosts"].items():
        lines += [f"## {host}", "", f"Runs: {block['runs']}", ""]
        p1_block = block.get("P1", {})
        if p1_block.get("available") or p1_block.get("contrasts"):
            lines += ["### P1 — role-swap twins (contrast accuracy, choice items)", "",
                      "| contrast | estimate |", "|---|---|"]
            lines += [f"| {k} | {_cell(v)} |" for k, v in p1_block.get("contrasts", {}).items()]
            lines += ["", f"Reading: {p1_block.get('reading', 'n/a')}", ""]
            means = p1_block.get("means", {})
            lines += ["| model | mean contrast accuracy |", "|---|---:|"]
            lines += [f"| {m} | {v['mean']:.3f} |" for m, v in sorted(means.items()) if v.get("mean") is not None]
            lines.append("")
        p2_block = block.get("P2", {})
        if p2_block.get("available"):
            lines += ["### P2 — algebraic filler recovery, held-out (static bundle, typed cleanup; MRR per concept)", "",
                      "| contrast | estimate |", "|---|---|"]
            lines += [f"| {k} | {_cell(v)} |" for k, v in p2_block["contrasts"].items()]
            lines += ["", f"Reading: {p2_block.get('reading')}", ""]
        algebraic = block["probes"].get("algebraic", {})
        if algebraic:
            lines += ["### Probe A by arm (seed-averaged MRR; held-out / all)", "", "| arm | probe | held-out | all |", "|---|---|---:|---:|"]
            for model in sorted(algebraic, key=lambda m: ARMS_ORDER.index(m) if m in ARMS_ORDER else 99):
                for key, value in algebraic[model].items():
                    if key.startswith("a.static") or key.startswith("b.frequency"):
                        held, every = value.get("heldout", {}).get("mrr"), value.get("all", {}).get("mrr")
                        lines.append(f"| {model} | `{key}` | {'–' if held is None else f'{held:.3f}'} | {'–' if every is None else f'{every:.3f}'} |")
            lines.append("")
        learned = block["probes"].get("learned", {})
        if learned:
            lines += ["### Probe B (LRE-style) by model (seed-averaged MRR, held-out)", "", "| model | representation | held-out | seen |",
                      "|---|---|---:|---:|"]
            for model in sorted(learned, key=lambda m: ARMS_ORDER.index(m) if m in ARMS_ORDER else 99):
                for key, value in learned[model].items():
                    held, seen = value.get("heldout", {}).get("mrr"), value.get("seen", {}).get("mrr")
                    lines.append(f"| {model} | `{key}` | {'–' if held is None else f'{held:.3f}'} | {'–' if seen is None else f'{seen:.3f}'} |")
            lines.append("")
        for name in ("ambiguous", "lre_vs_algebraic"):
            rows = block["probes"].get(name, {})
            if rows:
                lines += [f"### {name.replace('_', ' ')}", "", "| contrast | estimate |", "|---|---|"]
                lines += [f"| {k} | {_cell(v)} |" for k, v in rows.items()]
                lines.append("")
        step = block.get("step2") or {}
        if step.get("available"):
            lines += ["### Step 2 — readout arms (pre-registration §13)", "", "| endpoint | contrast | estimate |", "|---|---|---|"]
            lines += [f"| R1 | {k} | {_cell(v)} |" for k, v in step["R1"]["contrasts"].items()]
            r2 = step.get("R2", {})
            if r2.get("available"):
                lines.append(f"| R2 | U5 − C5, {r2['stratum']} (relative) | {r2['relative']:+.4f} [{r2['relative_ci_low']:+.4f}, "
                             f"{r2['relative_ci_high']:+.4f}] (p {r2['p_value']:.3g}; seeds {r2['seeds']}) |")
            lines += [f"| S2.1 | {k} | {_cell(v)} |" for k, v in step.get("twins_secondary", {}).items()]
            lines += ["", f"Reading: {step.get('reading')}", ""]
            if step.get("off_minus_on"):
                lines += ["| arm | gate off − on, after_heldout (relative) | on − C5 | off − C5 |", "|---|---:|---:|---:|"]
                for model, v in sorted(step["off_minus_on"].items()):
                    cell = lambda r: "n/a" if not r.get("available") else f"{r['relative']:+.4f} [{r['relative_ci_low']:+.4f}, {r['relative_ci_high']:+.4f}]"
                    vs = step["against_C5"].get(model, {})
                    lines.append(f"| {model} | {cell(v.get(R2_STRATUM, {}))} | {cell(vs.get('on', {}))} | {cell(vs.get('off', {}))} |")
                lines.append("")
            if step.get("diagnostics"):
                lines += ["| arm | role accuracy | role mass | no-query mass | read MRR | oracle MRR (all / held-out) |", "|---|---:|---:|---:|---:|---:|"]
                for model, v in sorted(step["diagnostics"].items()):
                    a, h = v.get("all", {}), v.get("heldout", {})
                    lines.append(f"| {model} | {a.get('role_accuracy', float('nan')):.3f} | {a.get('role_mass', float('nan')):.3f} | "
                                 f"{a.get('none_mass', float('nan')):.3f} | {a.get('read_mrr', float('nan')):.3f} | "
                                 f"{a.get('oracle_mrr', float('nan')):.3f} / {h.get('oracle_mrr', float('nan')):.3f} |")
                lines.append("")
        for field in ("twins", "natural"):
            section = block.get(field)
            if not section:
                continue
            lines += [f"### {field} items: contrast accuracy by model", "", "| kind | model | mean |", "|---|---|---:|"]
            for kind, means in section["means"].items():
                lines += [f"| {kind} | {m} | {v['mean']:.3f} |" for m, v in sorted(means.items()) if v.get("mean") is not None]
            lines += ["", "| kind | contrast | estimate |", "|---|---|---|"]
            for kind, rows in section["vs_candidate"].items():
                lines += [f"| {kind} | {k} | {_cell(v)} |" for k, v in rows.items()]
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--twins", default=None, help="role-twins item directory name (its RUN/<name> folders)")
    parser.add_argument("--natural", default=None, help="role-natural item directory name")
    parser.add_argument("--hosts", nargs="*", default=None); parser.add_argument("--licensed", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = {"experiment": "e9-binding-report", **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if args.overwrite:
        clear_output(args.output)
        for name in ("analysis.json",):
            if (args.output / name).is_file():
                (args.output / name).unlink()
    git_at_start = start_output(args.output, config)
    started = time.monotonic()
    twins = Path(args.twins).name if args.twins else None
    natural = Path(args.natural).name if args.natural else None
    analysis = analyse(args.runs, hosts=args.hosts, twins=twins, natural=natural, resamples=args.resamples, seed=args.seed)
    analysis["seconds"] = round(time.monotonic() - started, 1)
    analysis["licensed"] = bool(args.licensed)
    write_json(args.output / "analysis.json", json_ready(analysis))
    (args.output / "report.md").write_text(render(analysis, title=args.title or f"E9 binding program — {Path(args.runs).name}"))
    finish_output(args.output, config, git_at_start=git_at_start, device="cpu")
    print(json.dumps({"output": str(args.output), "hosts": list(analysis["hosts"])}))


if __name__ == "__main__":
    main()
