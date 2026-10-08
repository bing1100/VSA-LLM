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
    """host → model → seed → {"probe": load_probe(...), "twins": evaluation, "natural": evaluation}."""
    found: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in sorted(Path(runs_root).iterdir()) if Path(runs_root).exists() else []:
        match = NAME.match(run.name)
        if not match or (hosts and match["host"] not in hosts):
            continue
        entry = {"probe": probe_mod.load_probe(run / probe_mod.OUTPUT)}
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
