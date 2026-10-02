"""R6 self-authoring report (WP-E7): authoring quality, round-1 utility, loss gaps, compute, cross-authoring.

    python -m vsa_embed.experiments.e7_report --run experiments/e7-self-authoring/runs/general-v1 [--output <dir>]

Reads the E7 run folder (track, discovery, proposals, quality, judge results, round-1 verification
and cards, training runs under `train/` and `cross/`) and writes `report.md` + `summary.json`
(default `<run>/report-r6/`). Loss gaps reuse `e4_report`: the final evaluation of each run, paired
by seed and by evaluation window (cluster bootstrap over windows pooling seeds, 10,000 resamples),
Holm across the comparisons of each stratum. Strata are the fixed reference strata of each seed
(`ref_*`, identical target tokens in every condition): `ref_authored_new_self` (concepts that entered
only through self-authoring) and `ref_masked` (masked gold concepts) are the primary endpoint;
`ref_unlinked` is locality (relative to the compute-matched control, one-sided 0.5% margin).

G5 items (experiments.md E7 gate): self → self beats the compute-matched control and random frames on
the primary endpoint with CIs excluding zero (3 seeds); locality within 0.5%; verification improves
authored-edge precision over the no-verification condition (precision vs the hidden gold on masked
concepts, bootstrap over concepts pooled across seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from vsa_embed.experiments.e7_authoring import _default, _devtools_gold_edges, _json, gold, gold_edge_sets, load_track, visible

COMPARISONS = [("self", "cm"), ("self", "random"), ("self", "entigraph"), ("self", "selfnv"), ("teacher", "cm"),
               ("gold", "cm"), ("entigraph", "cm"), ("selfnv", "cm"), ("random", "cm"), ("teacher", "self")]
STRATA = ["ref_authored_new_self", "ref_masked", "ref_authored_new_teacher", "ref_authored_masked_self", "ref_new_all",
          "ref_base_after", "ref_unlinked", "all"]
PRIMARY = ("ref_authored_new_self", "ref_masked")
LOCALITY_MARGIN = 0.005
CROSS_COMPARISONS = [("authored", "none"), ("curated", "none"), ("authored", "curated")]
CROSS_STRATA = ["ref_authored_new", "ref_masked", "ref_unlinked", "all", "after", "after_rare"]


def _load_runs(folder: Path) -> dict[str, dict[int, Any]]:
    from vsa_embed.experiments.e4_report import load_run
    grid: dict[str, dict[int, Any]] = defaultdict(dict)
    if not folder.exists():
        return grid
    for metrics in sorted(folder.glob("*/metrics.jsonl")):
        run = load_run(metrics.parent)
        if run.complete:
            grid[run.condition][run.seed] = run
    return grid


def loss_gaps(grid: dict[str, dict[int, Any]], comparisons: Sequence[tuple[str, str]], strata: Sequence[str], *,
              resamples: int = 10_000) -> list[dict[str, Any]]:
    from vsa_embed.experiments.e4_report import paired_difference
    from vsa_embed.statistics import holm_adjust
    rows = []
    for stratum in strata:
        block = []
        for treatment, control in comparisons:
            if treatment in grid and control in grid:
                result = paired_difference(grid[treatment], grid[control], stratum, resamples=resamples, seed=0)
                if result is not None:
                    block.append({"treatment": treatment, "control": control, "stratum": stratum, **result})
        ps = [r.get("p_value") for r in block]
        valid = [i for i, p in enumerate(ps) if p is not None and math.isfinite(p)]
        adjusted = holm_adjust([ps[i] for i in valid]) if valid else []
        for i, p in zip(valid, adjusted):
            block[i]["p_holm"] = p
        rows += block
    return rows


def edge_precision(run: Path, track: dict[str, Any]) -> dict[str, Any]:
    """Precision of accepted (verified) vs proposed (unverified) edges against the hidden gold, per author and
    pooled over seeds, with the verified − unverified difference bootstrapped over masked concepts."""
    gold_data, view = gold(track), visible(track)
    gold_edges = gold_edge_sets(track, gold_data, view) if track["kind"] == "general" else _devtools_gold_edges(track, gold_data, view)
    masked = {s: int(e) for s, e in gold_data["masked_strings"].items()}
    aset = json.loads((run / "authoring_set.json").read_text())
    surface = {str(c["index"]): c["surface"] for c in aset["candidates"]}
    relation_label = {relation: label for label, relation, _ in track["relations"]}
    lemmas_of: dict[int, set[str]] = defaultdict(set)          # dictionary atom → the filler strings naming it
    for key, atoms in view["lexicon"].items():
        for atom in atoms:
            lemmas_of[atom].add(key)
    out: dict[str, Any] = {}
    for path in sorted((run / "round1").glob("s*/verify-*.json")):
        verified = json.loads(path.read_text())
        base_atoms = int(verified["atomic_count"])
        author = verified["author"]
        entry = out.setdefault(author, {"per_concept": {"accepted": [], "noverify": []}})
        for key in ("accepted", "noverify"):
            for k, frame in verified[key].items():
                if surface[k] not in masked or not frame:
                    continue
                hits = 0
                for relation, atom in frame:
                    label = relation_label.get(view["relation_names"][relation])
                    names = {verified["new_atoms"][atom - base_atoms]} if atom >= base_atoms else lemmas_of.get(atom, set())
                    hits += any(r == label and names & lemmas for r, lemmas in gold_edges.get(masked[surface[k]], []))
                entry["per_concept"][key].append((hits, len(frame), f"{path.parent.name}:{k}"))
    for author, entry in out.items():
        for key in ("accepted", "noverify"):
            rows = entry["per_concept"][key]
            hits, total = sum(h for h, _, _ in rows), sum(n for _, n, _ in rows)
            entry[key] = {"precision": hits / total if total else None, "edges": total, "concepts": len(rows)}
        a = {c: (h, n) for h, n, c in entry["per_concept"]["accepted"]}
        b = {c: (h, n) for h, n, c in entry["per_concept"]["noverify"]}
        shared = sorted(set(a) | set(b))
        if shared:
            rng = np.random.default_rng(0)
            diffs = []
            for _ in range(2000):
                pick = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
                ha, na = sum(a.get(c, (0, 0))[0] for c in pick), sum(a.get(c, (0, 0))[1] for c in pick)
                hb, nb = sum(b.get(c, (0, 0))[0] for c in pick), sum(b.get(c, (0, 0))[1] for c in pick)
                if na and nb:
                    diffs.append(ha / na - hb / nb)
            if diffs:
                entry["verified_minus_unverified"] = {"mean": float(np.mean(diffs)), "ci_low": float(np.quantile(diffs, 0.025)),
                                                      "ci_high": float(np.quantile(diffs, 0.975)), "concepts": len(shared)}
        del entry["per_concept"]
    return out


def compute_table(run: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(list((run / "discovery").glob("*.json")) + list((run / "proposals").glob("*.json")) +
                       list((run / "round1").glob("s*/verify-*.json")) + list((run / "round1").glob("entigraph.json"))):
        data = json.loads(path.read_text())
        for entry in data.get("ledger", []):
            rows.append({"source": str(path.relative_to(run)), **entry})
    return rows


def gate(gaps: list[dict[str, Any]], precision: dict[str, Any], consumer: str) -> dict[str, Any]:
    def find(t: str, c: str, s: str) -> dict[str, Any] | None:
        return next((g for g in gaps if g["treatment"] == t and g["control"] == c and g["stratum"] == s), None)

    def better(row: dict[str, Any] | None) -> bool | None:
        if row is None or row.get("ci_high") is None:
            return None
        return bool(row["ci_high"] < 0 and (row.get("p_holm") is None or row["p_holm"] < 0.05))

    items = {}
    for control in ("cm", "random"):
        items[f"self_beats_{control}"] = {s: better(find("self", control, s)) for s in PRIMARY}
    locality = find("self", "cm", "ref_unlinked")
    items["locality"] = None if locality is None or locality.get("relative_ci_high") is None else bool(
        locality["relative_ci_high"] <= LOCALITY_MARGIN)
    difference = (precision.get(consumer) or {}).get("verified_minus_unverified")
    items["verification_improves_precision"] = None if difference is None else bool(difference["ci_low"] > 0)
    items["seeds"] = sorted({s for g in gaps if g["treatment"] == "self" for s in g.get("seeds", [])})
    return items


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _rate(r: dict[str, Any] | None) -> str:
    if not r or r.get("value") is None:
        return "—"
    return f"{r['value']:.3f} [{r['low']:.3f}, {r['high']:.3f}] (n={r['n']})"


def render(summary: dict[str, Any]) -> str:
    track = summary["track"]
    lines = [f"# R6 self-authoring ({track['kind']} track)", "",
             "Status labels: every number below comes from the run folders listed in `summary.json`; "
             "LLM-judged plausibility is an estimate (experiments §0.12).", "", "## D7.0 setup", ""]
    prep = summary.get("prepare") or {}
    if prep:
        lines.append(f"Masked concepts: {prep['masked']['concepts']:,} (sha256 `{prep['masked']['sha256'][:16]}…`); "
                     f"visible aliases {prep['base']['aliases']:,}.")
    lines += ["", "## D7.1 authoring quality (masked gold concepts in the authoring set)", "",
              "| author | kind | concepts | edges | precision | recall | F1 | relation accuracy |", "|---|---|---:|---:|---|---|---:|---|"]
    for name, q in (summary.get("quality") or {}).get("authors", {}).items():
        lines.append(f"| {name} | {q.get('kind')} | {q['masked_concepts_authored']} | {q['edges']} | {_rate(q['precision'])} | "
                     f"{_rate(q['recall'])} | {_fmt(q['f1'], 3)} | {_rate(q['relation_accuracy'])} |")
    discovery = (summary.get("quality") or {}).get("discovery", {})
    if discovery:
        cutoffs = [k for k in next(iter(discovery.values())) if k.startswith("@")]
        lines += ["", f"Discovery recall of the {summary['quality'].get('discoverable_masked_entries')} discoverable masked concepts:", "",
                  "| host | " + " | ".join(cutoffs) + " |", "|---|" + "---|" * len(cutoffs)]
        for host, d in discovery.items():
            lines.append(f"| {host} | " + " | ".join(_rate(d[c]) for c in cutoffs) + " |")
    judged = summary.get("judge")
    if judged:
        lines += ["", f"LLM-judged plausibility of edges without gold ({judged['model']}, {judged['calls']} calls/item, "
                      f"Fleiss κ {_fmt(judged['fleiss_kappa'], 2)}, calibration accuracy {_rate(judged['calibration_accuracy'])}):", "",
                  "| author | true | true or partly |", "|---|---|---|"]
        for name, p in judged["plausibility"].items():
            lines.append(f"| {name} | {_rate(p['true'])} | {_rate(p['true_or_partly'])} |")
    lines += ["", "## D7.2 round 0 → 1", "", "| seed | author | proposed edges | accepted | concepts accepted | median U |",
              "|---|---|---:|---:|---:|---:|"]
    for row in summary.get("verification", []):
        s = row["summary"]
        lines.append(f"| {row['seed']} | {row['author']} | {s['proposed_edges']} | {s['accepted_edges']} | "
                     f"{s['concepts_with_accepted_edges']} | {_fmt(s['median_accepted_utility'])} |")
    precision = summary.get("edge_precision") or {}
    if precision:
        lines += ["", "Authored-edge precision vs the hidden gold (masked concepts, pooled over seeds):", "",
                  "| author | verified | unverified | verified − unverified [95% CI] |", "|---|---|---|---|"]
        for author, p in precision.items():
            d = p.get("verified_minus_unverified")
            lines.append(f"| {author} | {_fmt(p['accepted']['precision'], 3)} ({p['accepted']['edges']}) | "
                         f"{_fmt(p['noverify']['precision'], 3)} ({p['noverify']['edges']}) | "
                         + (f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "—") + " |")
    gaps = summary.get("loss_gaps") or []
    if gaps:
        lines += ["", "Loss gaps at the final evaluation (treatment − control, nats/token; negative = treatment better):", "",
                  "| stratum | comparison | Δ [95% CI] | relative | p (Holm) | seeds | method |", "|---|---|---|---|---|---|---|"]
        for g in gaps:
            ci = f"{g['delta']:+.4f} [{_fmt(g.get('ci_low'))}, {_fmt(g.get('ci_high'))}]"
            rel = "—" if g.get("relative") is None else f"{100 * g['relative']:+.2f}%"
            lines.append(f"| {g['stratum']} | {g['treatment']} − {g['control']} | {ci} | {rel} | {_fmt(g.get('p_holm'), 3)} | "
                         f"{len(g['seeds'])} | {g.get('method')} |")
    verdict = summary.get("gate")
    if verdict:
        lines += ["", "### G5 items", "", "```", json.dumps(verdict, indent=1), "```"]
    compute = summary.get("compute") or []
    if compute:
        lines += ["", "## Compute accounting (outside training)", "", "| source | stage | parameters | forward | prompt | generated | PFLOP | s |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for e in compute:
            lines.append(f"| {e['source']} | {e['stage']} | {e['parameters']:.3g} | {e['forward_tokens']:,} | {e['prompt_tokens']:,} | "
                         f"{e['generated_tokens']:,} | {e['flops'] / 1e15:.3f} | {_fmt(e.get('seconds'), 0)} |")
    for row in summary.get("compute_matched", []):
        lines.append(f"\nCompute-matched control, seed {row['seed']}: {row['round_tokens']:,} + {row['extra_tokens']:,} = "
                     f"{row['total_tokens']:,} training tokens (stages {', '.join(row['stages'] or [])}).")
    cards = summary.get("cards") or []
    if cards:
        lines += ["", "## Authoring cards (accepted, highest utility)", "", "| seed | author | concept | edge | U [lower] | n |",
                  "|---|---|---|---|---|---:|"]
        for c in cards:
            lines.append(f"| {c['seed']} | {c['author']} | {c['surface']} | {c['relation']} → {c['filler']} | "
                         f"{c['utility_mean']:+.4f} [{c['utility_low']:+.4f}] | {c['n']} |")
    cross = summary.get("cross_gaps") or []
    if cross:
        lines += ["", "## D7.3 cross-authoring (50M from scratch)", "", "| stratum | comparison | Δ [95% CI] | relative | p (Holm) |",
                  "|---|---|---|---|---|"]
        for g in cross:
            rel = "—" if g.get("relative") is None else f"{100 * g['relative']:+.2f}%"
            lines.append(f"| {g['stratum']} | {g['treatment']} − {g['control']} | {g['delta']:+.4f} [{_fmt(g.get('ci_low'))}, "
                         f"{_fmt(g.get('ci_high'))}] | {rel} | {_fmt(g.get('p_holm'), 3)} |")
    return "\n".join(lines) + "\n"


def write_report(run: Path, output: Path | None = None, *, resamples: int = 10_000, top_cards: int = 15) -> dict[str, Any]:
    run = Path(run)
    output = output or run / "report-r6"
    track = load_track(run)
    summary: dict[str, Any] = {"track": {"kind": track["kind"], "consumer": track["consumer"], "data_root": track["data_root"]}}
    for name, path in (("prepare", run / "summary.json"), ("quality", run / "quality" / "quality.json")):
        if path.exists():
            summary[name] = json.loads(path.read_text())
    if (run / "judge" / "results.json").exists():
        summary["judge"] = json.loads((run / "judge" / "results.json").read_text())["summary"]
    summary["verification"] = [{"seed": int(p.parent.name[1:]), "author": json.loads(p.read_text())["author"],
                                "summary": json.loads(p.read_text())["summary"]} for p in sorted((run / "round1").glob("s*/verify-*.json"))]
    summary["edge_precision"] = edge_precision(run, track) if summary["verification"] else {}
    cards = []
    for path in sorted((run / "round1").glob("s*/cards-*.jsonl")):
        for line in path.read_text().splitlines():
            card = json.loads(line)
            if card["accepted"]:
                cards.append({"seed": int(path.parent.name[1:]), "author": path.stem[len("cards-"):], **card})
    summary["cards"] = sorted(cards, key=lambda c: -c["utility_mean"])[:top_cards]
    summary["compute"] = compute_table(run)
    data = Path(track["data_root"]).expanduser() / "round1"
    summary["compute_matched"] = []
    for path in sorted(data.glob("s*/cm/materialized.json")):
        cm = json.loads(path.read_text())["compute_matched"]
        summary["compute_matched"].append({"seed": int(path.parent.parent.name[1:]), **{k: cm[k] for k in
                                           ("round_tokens", "extra_tokens", "total_tokens", "stages")}})
    grid = _load_runs(run / "train")
    summary["loss_gaps"] = loss_gaps(grid, COMPARISONS, STRATA, resamples=resamples)
    summary["gate"] = gate(summary["loss_gaps"], summary["edge_precision"], track["consumer"]) if summary["loss_gaps"] else None
    summary["cross_gaps"] = loss_gaps(_load_runs(run / "cross"), CROSS_COMPARISONS, CROSS_STRATA, resamples=resamples)
    summary["runs"] = {c: sorted(str(r.path) for r in seeds.values()) for c, seeds in grid.items()}
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "summary.json", summary)
    (output / "report.md").write_text(render(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    summary = write_report(args.run, args.output, resamples=args.resamples)
    print(json.dumps(summary.get("gate"), indent=1, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
