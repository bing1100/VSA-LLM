"""R5 report (explainability and zero-shot, gate G4) from E5 output folders.

    python -m vsa_embed.experiments.e5_report --inputs experiments/e5-explainability/runs --output reports/R5-explainability

Every folder under `--inputs` with `summary.json` and a `resolved_config.yaml` whose `experiment`
starts with `e5.` is read: E5.1 faithfulness, E5.2 sense alignment, E5.4 zero-shot, E5.5 frequency
(one folder per trained run) and E5.3 rating studies (one folder per graded item set). Runs are
grouped by (model size, condition) — parsed from the source run's experiment name as in E4 — and
seeds are aggregated with a seed-level Student-t interval. Zero-shot `own` rows of different
conditions (composition vs the C2 fallback vs C0) are also compared item by item, pooling seeds
(per-item mean over shared seeds, bootstrap over items). Writes `report.md`, `summary.json` and
`figures/*.png`. Rating results are labelled "LLM-graded estimate"; text-evidence zero-shot sources
are tabulated apart from the structure-only claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ..statistics import mean_confidence_interval
from .e4_report import _GRID, _MUTED, _PALETTE, condition_order
from .e5_common import fmt, fmt_ci, write_json
from .e5_zeroshot import STRUCTURE_SOURCES, TEXT_SOURCES, paired_difference

KINDS = {"e5.1-faithfulness": "faithfulness", "e5.2-sense-alignment": "senses", "e5.4-zero-shot": "zeroshot",
         "e5.5-frequency": "frequency", "e5.3-rating": "rating"}
_INK, _INK2 = "#0b0b0b", "#52514e"


def discover(roots: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        for summary_path in sorted(Path(root).rglob("summary.json")):
            folder = summary_path.parent
            config_path = folder / "resolved_config.yaml"
            if not config_path.exists():
                continue
            config = yaml.safe_load(config_path.read_text()) or {}
            kind = KINDS.get(str(config.get("experiment")))
            if kind is None:
                continue
            summary = json.loads(summary_path.read_text())
            source = summary.get("source") or {}
            found[kind].append({"path": folder, "config": config, "summary": summary,
                                "size": source.get("size", "?"), "condition": source.get("condition", "?"),
                                "seed": source.get("seed", 0)})
    return found


def _ci(values: Sequence[float | None]) -> dict[str, Any] | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return mean_confidence_interval(values) if values else None


def _groups(entries: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        groups[(e["size"], e["condition"])].append(e)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0][0], condition_order(kv[0][1]))))


def aggregate_faithfulness(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (size, condition), runs in _groups(entries).items():
        ks = sorted({int(s[1:]) for r in runs for s in r["summary"]["summary"]["strata"] if s[1:].isdigit()})
        for k in ks:
            stratum = f"k{k}"
            blocks = [r["summary"]["summary"]["strata"].get(stratum, {}) for r in runs]
            blocks = [b for b in blocks if b.get("loss")]
            if not blocks:
                continue
            get = lambda key: [b["loss"][key]["mean"] for b in blocks]
            significant = sum(any(c["significant"] and c["stratum"] == stratum and c["metric"] == "loss" and c["measure"] == "comprehensiveness"
                                  and c["contrast"] == "top-random" for c in r["summary"]["summary"]["contrasts"]) for r in runs)
            rows.append({"size": size, "condition": condition, "k": k, "seeds": len(blocks),
                         "uniform_weights": float(np.mean([r["summary"]["summary"]["checks"].get("uniform_weight_fraction") or 0 for r in runs])),
                         "channel_effect": _ci([b["channel_effect"]["mean"] for b in blocks]),
                         **{f"comp_{o}": _ci(get(f"comprehensiveness_{o}")) for o in ("top", "random", "bottom")},
                         "comp_top_minus_random": _ci(get("comprehensiveness_top_minus_random")),
                         "suff_top_minus_random": _ci(get("sufficiency_top_minus_random")),
                         "seeds_significant_top_gt_random": significant})
    return rows


def aggregate_senses(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (size, condition), runs in _groups(entries).items():
        for subset in ("all", "heldout", "seen"):
            blocks = [r["summary"]["summary"].get(subset, {}) for r in runs]
            blocks = [b for b in blocks if b.get("n")]
            if not blocks:
                continue
            rows.append({"size": size, "condition": condition, "subset": subset, "seeds": len(blocks), "n": int(np.mean([b["n"] for b in blocks])),
                         "sense_accuracy": _ci([b["sense_accuracy"] for b in blocks]), "wn1_accuracy": _ci([b["wn1_accuracy"] for b in blocks]),
                         "semcor_mfs_accuracy": _ci([b["semcor_mfs_accuracy"] for b in blocks]),
                         "sense_minus_wn1": _ci([b["sense_minus_wn1"]["mean"] for b in blocks]),
                         "sense_minus_semcor_mfs": _ci([b["sense_minus_semcor_mfs"]["mean"] for b in blocks]),
                         "gold_share_lift": _ci([(b.get("gold_share_lift") or {}).get("mean") for b in blocks]),
                         "tie_rate": float(np.mean([b["tie_rate"] for b in blocks]))})
    return rows


def aggregate_frequency(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (size, condition), runs in _groups(entries).items():
        names = sorted({n for r in runs for n in r["summary"]["summary"]})
        for name in names:
            values = [r["summary"]["summary"][name] for r in runs if name in r["summary"]["summary"]]
            rows.append({"size": size, "condition": condition, "representation": name, "seeds": len(values), "n": values[0]["n"],
                         "r2": _ci([v["r2"] for v in values]), "spearman": _ci([v["spearman"] for v in values]),
                         "quartile_auc": _ci([v["quartile_auc"] for v in values])})
    return rows


def _zeroshot_predictions(entry: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    """(source, test) → item id → value, from a zero-shot run's predictions.jsonl."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    path = entry["path"] / "predictions.jsonl"
    if not path.exists():
        return out
    linked = {c for c, info in entry["summary"].get("resolved", {}).items() if info.get("linked") and info.get("status") != "mislinked"}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("test") in ("property", "entailment") and row.get("concept") in linked:
            out[(row["source"], row["test"])][row["id"]] = float(row["correct"])
    return out


def aggregate_zeroshot(entries: Sequence[dict[str, Any]], *, resamples: int = 2000) -> dict[str, Any]:
    scenarios: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        scenarios[e["summary"]["manifest"]["scenario"]].append(e)
    out = {}
    for scenario, runs in sorted(scenarios.items()):
        rows, comparisons, cross = [], [], []
        for (size, condition), group in _groups(runs).items():
            sources = sorted({s for r in group for s in r["summary"]["summary"]["sources"]},
                             key=lambda s: (s not in STRUCTURE_SOURCES, (STRUCTURE_SOURCES + TEXT_SOURCES).index(s)
                                            if s in STRUCTURE_SOURCES + TEXT_SOURCES else 99))
            for source in sources:
                metrics = [r["summary"]["summary"]["sources"][source]["linked"] for r in group if source in r["summary"]["summary"]["sources"]]
                rows.append({"size": size, "condition": condition, "source": source, "seeds": len(metrics),
                             "family": "text_evidence" if source in TEXT_SOURCES else "structure",
                             **{t: _ci([m.get(t, {}).get("mean") for m in metrics]) for t in
                                ("property", "entailment", "paraphrase", "semantic_mrr", "after_loss")}})
            for c in {(c["test"], c["baseline"], c["family"]) for r in group for c in r["summary"]["summary"]["comparisons"]}:
                values = [x["difference"] for r in group for x in r["summary"]["summary"]["comparisons"]
                          if (x["test"], x["baseline"], x["family"]) == c]
                significant = sum(1 for r in group for x in r["summary"]["summary"]["comparisons"]
                                  if (x["test"], x["baseline"], x["family"]) == c and x.get("significant") and x["better"])
                comparisons.append({"size": size, "condition": condition, "test": c[0], "baseline": c[1], "family": c[2],
                                    "difference": _ci(values), "seeds": len(values), "seeds_significant": significant})
        # Cross-condition comparison of the `own` rows, item by item (seeds pooled per item).
        by_condition: dict[tuple[str, str], dict[int, Any]] = defaultdict(dict)
        for r in runs:
            by_condition[(r["size"], r["condition"])][int(r["seed"])] = _zeroshot_predictions(r)
        keys = sorted(by_condition, key=lambda k: (k[0], condition_order(k[1])))
        for a in keys:
            for b in keys:
                if a >= b or a[0] != b[0]:
                    continue
                for test in ("property", "entailment"):
                    seeds = sorted(set(by_condition[a]) & set(by_condition[b]))
                    per_item: dict[str, list[float]] = defaultdict(list)
                    for seed in seeds:
                        va, vb = by_condition[a][seed].get(("own", test), {}), by_condition[b][seed].get(("own", test), {})
                        for item in set(va) & set(vb):
                            per_item[item].append(va[item] - vb[item])
                    if len(per_item) < 2:
                        continue
                    diffs = np.asarray([np.mean(v) for _, v in sorted(per_item.items())])
                    result = paired_difference(diffs, np.zeros_like(diffs), resamples=resamples)
                    cross.append({"size": a[0], "a": a[1], "b": b[1], "test": test, "seeds": len(seeds), **result})
        chances = [r["summary"]["summary"].get("chance", {}).get("property") for r in runs]
        chances = [c for c in chances if c is not None]
        out[scenario] = {"chance": float(np.mean(chances)) if chances else None,
                         "rows": rows, "comparisons": sorted(comparisons, key=lambda c: (c["size"], condition_order(c["condition"]),
                                                                                       c["test"], c["family"], c["baseline"])),
                         "cross_condition": cross,
                         "contamination_free": runs[0]["summary"]["manifest"].get("contamination_free")}
    return out


def aggregate_rating(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for e in entries:
        summary = e["summary"]
        manifest = summary.get("manifest", {})
        for study, entry in summary["analysis"]["studies"].items():
            rows.append({"path": str(e["path"]), "item_set": manifest.get("study"), "study": study, "judge": summary.get("judge"),
                         "concept_set": manifest.get("concept_set"), "space": manifest.get("space"), **entry})
    return rows


# -- rendering ----------------------------------------------------------------------------------------------------------

def render(summary: dict[str, Any], figures: dict[str, str]) -> str:
    lines = ["# R5 — explainability and zero-shot (E5, gate G4)", "",
             "Generated by `vsa_embed.experiments.e5_report` from the E5 output folders listed in `summary.json`. Seed-level "
             "means with Student-t 95% intervals; per-run bootstrap intervals are in each run's `report.md`.", ""]
    faith = summary.get("faithfulness") or []
    if faith:
        lines += ["## E5.1 Faithfulness (H-F)", "",
                  "Δloss on the 8 tokens after spans with more than k edges (nats/token). Comprehensiveness = removing the k edges; "
                  "top − random > 0 means the attention ranks edges faithfully. A static (C3) composer has uniform weights: "
                  "its top-k is random by construction.", "",
                  "| size | condition | k | seeds | uniform weights | channel effect | comp. top | comp. random | comp. bottom | top − random | suff. top − random | seeds sig. |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in faith:
            lines.append(f"| {r['size']} | {r['condition']} | {r['k']} | {r['seeds']} | {fmt(r['uniform_weights'], 2)} | {fmt_ci(r['channel_effect'])} | "
                         f"{fmt_ci(r['comp_top'])} | {fmt_ci(r['comp_random'])} | {fmt_ci(r['comp_bottom'])} | {fmt_ci(r['comp_top_minus_random'])} | "
                         f"{fmt_ci(r['suff_top_minus_random'])} | {r['seeds_significant_top_gt_random']}/{r['seeds']} |")
        lines.append("")
    senses = summary.get("senses") or []
    if senses:
        lines += ["## E5.2 Sense alignment (H-C)", "", "| size | condition | subset | seeds | items | attention | WN first sense | SemCor MFS | attention − WN1 | attention − MFS | gold-mass lift | ties |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in senses:
            lines.append(f"| {r['size']} | {r['condition']} | {r['subset']} | {r['seeds']} | {r['n']} | {fmt_ci(r['sense_accuracy'])} | "
                         f"{fmt_ci(r['wn1_accuracy'])} | {fmt_ci(r['semcor_mfs_accuracy'])} | {fmt_ci(r['sense_minus_wn1'])} | "
                         f"{fmt_ci(r['sense_minus_semcor_mfs'])} | {fmt_ci(r['gold_share_lift'])} | {fmt(r['tie_rate'], 2)} |")
        lines.append("")
    rating = summary.get("rating") or []
    if rating:
        lines += ["## E5.3 Rating study (LLM-graded estimate)", ""]
        for r in rating:
            judge = r.get("judge") or {}
            counts = "; ".join(f"{s} {c['success']}/{c['total']}" for s, c in (r.get("counts") or {}).items())
            fisher = "; ".join(f"{f['reference']} > {o}: p = {f['p_one_tailed']:.3g}" for o, f in (r.get("fisher") or {}).items())
            prefs = "; ".join(f"vs {o}: {p.get('reference', 0)}–{p.get('other', 0)} (tie {p.get('tie', 0)}), p = {fmt(p['p_sign_one_tailed'], 4)}"
                              for o, p in (r.get("preference") or {}).items())
            lines.append(f"- **{r['item_set']} / {r['study']}** ({r.get('concept_set') or ''} {r.get('space') or ''}; judge {judge.get('model')} via "
                         f"{judge.get('harness')}, {judge.get('calls')} calls/item): {counts or prefs or '—'}"
                         + (f"; Fisher one-tailed {fisher}" if fisher else "")
                         + f"; κ {fmt(r.get('fleiss_kappa'), 3)}; calibration {fmt((r.get('calibration') or {}).get('accuracy'), 3)} "
                         f"({(r.get('calibration') or {}).get('items', 0)} items).")
        lines.append("")
    zeroshot = summary.get("zeroshot") or {}
    if zeroshot:
        lines += ["## E5.4 Zero-shot insertion (H-E)", "",
                  "Accuracy on linked reserved concepts (property, entailment), paraphrase agreement, semantic-head MRR and after-span loss. "
                  "`own` = the run's own row. Text-evidence sources are reported separately from the structure-only claim.", ""]
        for scenario, block in zeroshot.items():
            lines += [f"### {scenario} (contamination-free on pretrained hosts: {block['contamination_free']})", ""]
            for family, title in (("structure", "Structure-only"), ("text_evidence", "Text evidence")):
                rows = [r for r in block["rows"] if r["family"] == family]
                if not rows:
                    continue
                lines += [f"{title}:", "", "| size | condition | source | seeds | property | entailment | paraphrase | semantic MRR | after loss |",
                          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
                for r in rows:
                    lines.append(f"| {r['size']} | {r['condition']} | {r['source']} | {r['seeds']} | {fmt_ci(r['property'])} | {fmt_ci(r['entailment'])} | "
                                 f"{fmt_ci(r['paraphrase'])} | {fmt_ci(r['semantic_mrr'])} | {fmt_ci(r['after_loss'])} |")
                lines.append("")
            if block["comparisons"]:
                lines += ["`own` − baseline within each run (seed-level mean; seeds with a Holm-significant gain):", "",
                          "| size | condition | test | family | baseline | difference | seeds sig. |", "|---|---|---|---|---|---:|---:|"]
                for c in block["comparisons"]:
                    lines.append(f"| {c['size']} | {c['condition']} | {c['test']} | {c['family']} | {c['baseline']} | {fmt_ci(c['difference'])} | "
                                 f"{c['seeds_significant']}/{c['seeds']} |")
                lines.append("")
            if block["cross_condition"]:
                lines += ["Across conditions (`own` rows, item-level, seeds pooled):", "", "| size | A | B | test | A − B [95% CI] | items | p |",
                          "|---|---|---|---|---|---:|---:|"]
                for c in block["cross_condition"]:
                    lines.append(f"| {c['size']} | {c['a']} | {c['b']} | {c['test']} | {fmt_ci(c)} | {c['n']} | {fmt(c['p_value'], 4)} |")
                lines.append("")
    frequency = summary.get("frequency") or []
    if frequency:
        lines += ["## E5.5 Frequency disentanglement", "", "Ridge-probe R² of log training frequency (lower = less frequency information).", "",
                  "| size | condition | representation | seeds | items | R² | Spearman ρ | quartile AUC |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in frequency:
            lines.append(f"| {r['size']} | {r['condition']} | {r['representation']} | {r['seeds']} | {r['n']:,} | {fmt_ci(r['r2'])} | "
                         f"{fmt_ci(r['spearman'])} | {fmt_ci(r['quartile_auc'])} |")
        lines.append("")
    if figures:
        lines += ["## Figures", ""] + [f"![{title}]({path})" for title, path in figures.items()] + [""]
    if not any(summary.get(k) for k in ("faithfulness", "senses", "rating", "zeroshot", "frequency")):
        lines += ["No E5 outputs were found.", ""]
    return "\n".join(lines)


def _style(ax) -> None:
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK2, labelsize=8)


def plot(summary: dict[str, Any], out: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figures = {}
    faith = summary.get("faithfulness") or []
    if faith:
        panels = sorted({(r["size"], r["condition"]) for r in faith}, key=lambda p: (p[0], condition_order(p[1])))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.0), squeeze=False, sharey=True)
        for ax, (size, condition) in zip(axes.flat, panels):
            rows = sorted((r for r in faith if (r["size"], r["condition"]) == (size, condition)), key=lambda r: r["k"])
            ks = [r["k"] for r in rows]
            for slot, order in enumerate(("top", "random", "bottom")):
                means = [(r[f"comp_{order}"] or {}).get("mean") for r in rows]
                ax.plot(ks, means, marker="o", markersize=4, linewidth=2, color=_PALETTE[slot], label=f"remove {order}-k")
            ax.axhline(0, color=_MUTED, linewidth=0.8)
            ax.set_title(f"{condition} ({size})", fontsize=10, color=_INK, loc="left"); ax.set_xticks(ks)
            ax.set_xlabel("k edges removed", fontsize=8, color=_INK2)
            _style(ax)
        axes.flat[0].set_ylabel("Δ after-span loss (nats/token)", fontsize=8, color=_INK2)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("E5.1 comprehensiveness (mean over seeds)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        fig.savefig(out / "figures" / "faithfulness.png", dpi=150); plt.close(fig)
        figures["E5.1 comprehensiveness"] = "figures/faithfulness.png"
    frequency = summary.get("frequency") or []
    if frequency:
        names = [n for n in ("concept_surface", "concept_rows", "concept_rows_init", "concept_rows_unit", "token_rows")
                 if any(r["representation"] == n for r in frequency)]
        groups = sorted({(r["size"], r["condition"]) for r in frequency}, key=lambda p: (p[0], condition_order(p[1])))
        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(groups) * max(1, len(names)) / 2), 3.2))
        width = 0.8 / max(1, len(names))
        for slot, name in enumerate(names):
            xs, ys = [], []
            for g, key in enumerate(groups):
                row = next((r for r in frequency if (r["size"], r["condition"]) == key and r["representation"] == name), None)
                if row and row["r2"]:
                    xs.append(g + slot * width); ys.append(row["r2"]["mean"])
            ax.bar(xs, ys, width=width * 0.92, color=_PALETTE[slot], label=name)
        ax.set_xticks([g + width * (len(names) - 1) / 2 for g in range(len(groups))])
        ax.set_xticklabels([f"{c}\n{s}" for s, c in groups], fontsize=8)
        ax.set_ylabel("R² of log frequency", fontsize=8, color=_INK2); _style(ax)
        ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper left")
        ax.set_title("E5.5 frequency predictability (lower = more disentangled)", fontsize=10, color=_INK, loc="left")
        fig.tight_layout(); fig.savefig(out / "figures" / "frequency.png", dpi=150); plt.close(fig)
        figures["E5.5 frequency predictability"] = "figures/frequency.png"
    zeroshot = summary.get("zeroshot") or {}
    for scenario, block in zeroshot.items():
        rows = [r for r in block["rows"] if r["property"]]
        if not rows:
            continue
        groups = sorted({(r["size"], r["condition"]) for r in rows}, key=lambda p: (p[0], condition_order(p[1])))
        counts = [len([r for r in rows if (r["size"], r["condition"]) == g]) for g in groups]
        fig, axes = plt.subplots(1, len(groups), figsize=(1.2 + 0.42 * sum(counts) + 0.6 * len(groups), 3.4), squeeze=False,
                                 sharey=True, gridspec_kw={"width_ratios": [max(2, c) for c in counts]})
        chance = block.get("chance")
        for ax, key in zip(axes.flat, groups):
            subset = [r for r in rows if (r["size"], r["condition"]) == key]
            colours = [_PALETTE[0] if r["family"] == "structure" else _PALETTE[1] for r in subset]
            ax.bar(range(len(subset)), [r["property"]["mean"] for r in subset], color=colours, width=0.7)
            ax.set_xlim(-0.6, max(2, len(subset)) - 0.4)
            if chance:
                ax.axhline(chance, color=_MUTED, linewidth=0.9, linestyle="--")
            ax.set_xticks(range(len(subset))); ax.set_xticklabels([r["source"] for r in subset], rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{key[1]} ({key[0]})", fontsize=10, color=_INK, loc="left"); _style(ax)
        axes.flat[0].set_ylabel("property accuracy", fontsize=8, color=_INK2)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [Patch(color=_PALETTE[0]), Patch(color=_PALETTE[1])] + ([Line2D([], [], color=_MUTED, linestyle="--")] if chance else [])
        fig.legend(handles, ["structure-only", "text evidence"] + (["chance"] if chance else []), loc="upper right",
                   frameon=False, fontsize=8, ncol=3)
        fig.suptitle(f"E5.4 zero-shot property selection — {scenario}", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        name = f"figures/zeroshot-{scenario}.png"
        fig.savefig(out / name, dpi=150); plt.close(fig)
        figures[f"E5.4 zero-shot — {scenario}"] = name
    return figures


def write_report(inputs: Sequence[Path], output: Path, *, resamples: int = 2000) -> dict[str, Any]:
    found = discover(inputs)
    summary = {"inputs": [str(p) for p in inputs],
               "folders": {kind: [str(e["path"]) for e in entries] for kind, entries in found.items()},
               "faithfulness": aggregate_faithfulness(found.get("faithfulness", [])),
               "senses": aggregate_senses(found.get("senses", [])),
               "frequency": aggregate_frequency(found.get("frequency", [])),
               "zeroshot": aggregate_zeroshot(found.get("zeroshot", []), resamples=resamples),
               "rating": aggregate_rating(found.get("rating", []))}
    output.mkdir(parents=True, exist_ok=True)
    figures = plot(summary, output)
    (output / "report.md").write_text(render(summary, figures))
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    summary = write_report(args.inputs, args.output, resamples=args.resamples)
    print(json.dumps({k: len(v) for k, v in summary["folders"].items()}, indent=2))


if __name__ == "__main__":
    main()
