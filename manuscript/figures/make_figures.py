"""Regenerate every manuscript figure and table from committed run folders.

    ~/anaconda3/envs/vsa-repro/bin/python manuscript/figures/make_figures.py [--only e0 e2 cardinality generation e4]

Reads only committed artifacts under `experiments/` (run folders: `summary.json`, `metrics.*`,
`manifest.json`, `cardinality*.json`, `feasibility.json`; the B10 `generation.json`) and writes:

- `manuscript/figures/<name>.png` and `.pdf`
- `manuscript/tables/<name>.md`
- `manuscript/figures/sources.json`: for every output, the input files and their sha256 and the
  commit recorded in the run's manifest, so a figure can be traced to the run that produced it.

Outputs are deterministic (no timestamps in PNG/PDF metadata), so re-running on unchanged inputs gives
no diff. E4 figures/tables are stubs: they are produced only when
`experiments/e4-small-lm/analysis/*/summary.json` (written by `vsa_embed.experiments.e4_report`) exist;
otherwise the E4 table states that the analysis is pending. Nothing here trains, evaluates or touches
`.jobs/`; CPU only.

Statistics: seed-level 95% Student-t intervals (3 seeds, t = 4.303), the same rule as
`vsa_embed.statistics.mean_confidence_interval`, which the committed reports use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIG = ROOT / "manuscript" / "figures"
TAB = ROOT / "manuscript" / "tables"

# ---------------------------------------------------------------- style
# Colour-blind-safe categorical order (validated: adjacent CVD ΔE ≥ 9.1, normal-vision ΔE ≥ 19.6 on a light
# surface; the first three slots also pass all-pairs). Same order as `e4_report._PALETTE`, so E4 figures match.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b")
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "svg.hashsalt": "cg-vsa",
})

SOURCES: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------- helpers

def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_commit(run: Path) -> str | None:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text())
    sha = m.get("git_sha")
    return None if not sha else sha[:7] + ("-dirty" if m.get("git_dirty") else "")


def record(output: str, inputs: Iterable[Path], runs: Iterable[Path] = ()) -> None:
    SOURCES[output] = {
        "inputs": {rel(p): sha256(p) for p in sorted(set(inputs))},
        "run_commits": {rel(r): manifest_commit(r) for r in sorted(set(runs))},
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def t_ci(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean and 95% Student-t interval over seeds (as vsa_embed.statistics.mean_confidence_interval)."""
    data = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(data))
    if n == 1:
        return mean, float("nan"), float("nan")
    half = T975.get(n - 1, 1.96) * float(np.std(data, ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def save(fig: plt.Figure, name: str) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", metadata={"Software": None})
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)
    return [f"figures/{name}.png", f"figures/{name}.pdf"]


def write_table(name: str, text: str) -> str:
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.md").write_text(text.rstrip() + "\n")
    return f"tables/{name}.md"


def fmt_ci(mean: float, low: float, high: float, digits: int = 4, signed: bool = False) -> str:
    sign = "+" if signed else ""
    if not math.isfinite(mean):
        return "n/a"
    if not math.isfinite(low):
        return f"{mean:{sign}.{digits}f}"
    return f"{mean:{sign}.{digits}f} [{low:{sign}.{digits}f}, {high:{sign}.{digits}f}]"


def fmt_rate(values: Iterable[float], digits: int = 3) -> str:
    """Bounded rates over seeds: mean (min–max). `n/a` if undefined in any seed, as in the committed reports."""
    data = [float(v) for v in values]
    if not data or any(not math.isfinite(v) for v in data):
        return "n/a"
    lo, hi = min(data), max(data)
    mean = float(np.mean(data))
    return f"{mean:.{digits}f}" if lo == hi else f"{mean:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]], align: Sequence[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK, ha="right", va="bottom")


def generated_note(sources: Iterable[Path]) -> str:
    return ("_Generated by `manuscript/figures/make_figures.py` from " +
            ", ".join(f"`{rel(s)}`" for s in sources) + ". Do not edit by hand._")


# ---------------------------------------------------------------- E0

E0_DEV = EXP / "e0-synthetic-identifiability" / "runs" / "e0-development"
E0_SYNC = EXP / "e0-synthetic-identifiability" / "runs" / "d02-parent-sync"
D01_LEARNERS = [("m0", "M0 static bundle"), ("salience", "01b feature salience"), ("m1_p0", "M1, no query (P0)"),
                ("m1_q", "M1 with context query"), ("free_query_q", "free per-context weights")]
D02_POLICIES = [("m3", "M3 (null)"), ("m3_anderson", "M3 Anderson–Darling"), ("coherence_only", "coherence only"),
                ("random", "random splits")]


def e0() -> list[str]:
    dev = pd.read_csv(E0_DEV / "metrics.csv")
    sync = pd.read_csv(E0_SYNC / "metrics.csv")
    summary = load_json(E0_DEV / "summary.json")
    outputs: list[str] = []

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.3], "wspace": 0.5})
    legend_kw = dict(loc="upper left", bbox_to_anchor=(-0.02, -0.3), ncol=1, handlelength=2.0, fontsize=7)

    # (a) D0.1: held-out concept × held-out context cosine vs teacher context strength. Intervals are drawn for
    # the two learners the gate compares (M1 with query, M0); the table has every interval.
    ax = axes[0]
    d01 = dev[(dev.experiment == "D0.1") & (dev.split == "test")]
    for i, (learner, label) in enumerate(D01_LEARNERS):
        xs, ms, lo, hi = [], [], [], []
        for strength in sorted(d01.strength.unique()):
            m, l, h = t_ci(d01[(d01.learner == learner) & (d01.strength == strength)].cosine)
            xs.append(strength); ms.append(m); lo.append(l); hi.append(h)
        ax.plot(xs, ms, color=SERIES[i], marker=MARKERS[i], label=label, zorder=3 + (learner == "m1_q"))
        if learner in ("m0", "m1_q"):
            ax.fill_between(xs, lo, hi, color=SERIES[i], alpha=0.14, linewidth=0, zorder=2)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xticks([0, 0.5, 1, 2, 4]); ax.set_xticklabels(["0\n(static)", "0.5", "1", "2", "4"])
    ax.set_xlabel("teacher context strength"); ax.set_ylabel("held-out cosine")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("D0.1 context query", loc="left")
    ax.legend(**legend_kw)
    panel_label(ax, "a")

    # (b) D0.3: held-out cosine vs rank for the concept-factor options.
    ax = axes[1]
    d03 = dev[(dev.experiment == "D0.3") & (dev.split == "test")]
    for i, (factor, label) in enumerate([("free", "free factor"), ("induced", "induced factor"), ("hybrid", "hybrid factor (+δ)")]):
        rows = d03[(d03.factor == factor) & (d03.learner != "m0")]
        ranks = sorted(rows["rank"].unique())
        stats = [t_ci(rows[rows["rank"] == r].cosine) for r in ranks]
        ax.errorbar(ranks, [s[0] for s in stats], yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                    color=SERIES[i], marker=MARKERS[i], capsize=2, elinewidth=0.8, label=label,
                    linestyle="--" if factor == "hybrid" else "-", zorder=3)
    m0 = t_ci(d03[d03.learner == "m0"].cosine)[0]
    ax.axhline(m0, color=INK2, linewidth=0.9, zorder=2, label="M0 static bundle")
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlabel("rank k (teacher k* = 4)"); ax.set_ylabel("held-out cosine (composition-disjoint)")
    ax.set_ylim(0.5, 1.03)
    ax.set_title("D0.3 factored mapping", loc="left")
    ax.legend(**legend_kw)
    panel_label(ax, "b")

    # (c) D0.2 with the M3 default (parent fallback + synced parents): bars = mean over seeds, dots = seeds.
    ax = axes[2]
    d02 = sync[sync.experiment == "D0.2"]
    metrics = [("precision", "precision"), ("recall", "recall"), ("false_split_rate", "false-split rate")]
    groups = [(t, p) for t in ("atomics", "relations") for p, _ in D02_POLICIES]
    width = 0.8 / len(metrics)
    for j, (metric, mlabel) in enumerate(metrics):
        x = np.arange(len(groups)) + (j - 1) * width
        for xi, (target, policy) in zip(x, groups):
            values = d02[(d02.target == target) & (d02.policy == policy)][metric].dropna().to_numpy()
            if len(values):
                ax.bar(xi, values.mean(), width=width - 0.04, color=SERIES[j], zorder=3, label=mlabel if xi == x[0] else None)
                ax.plot(np.full(len(values), xi), values, "o", markersize=2.2, color=INK, alpha=0.75, zorder=4,
                        markeredgewidth=0)
    for level, text in ((0.9, "0.9"), (0.05, "0.05")):
        ax.axhline(level, color=INK2, linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=2)
        ax.text(len(groups) - 0.3, level, f"gate\n{text}", fontsize=6.0, color=INK2, ha="left", va="center", clip_on=False)
    short = {"m3": "M3", "m3_anderson": "AD", "coherence_only": "coh.", "random": "rand."}
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels([short[p] for _, p in groups], rotation=45, ha="right", fontsize=6.8)
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    for k, target in enumerate(("atomics", "relations")):
        ax.text(k * len(D02_POLICIES) + (len(D02_POLICIES) - 1) / 2, 1.02, target, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axvline(len(D02_POLICIES) - 0.5, color=AXIS, linewidth=0.6)
    ax.set_ylabel("rate at convergence")
    ax.set_title("D0.2 split detection", loc="left", pad=12)
    ax.legend(**{**legend_kw, "bbox_to_anchor": (-0.02, -0.36)})
    panel_label(ax, "c")
    outputs += save(fig, "e0_identifiability")
    record("figures/e0_identifiability", [E0_DEV / "metrics.csv", E0_SYNC / "metrics.csv"], [E0_DEV, E0_SYNC])

    # ---- table
    lines = ["# E0 — synthetic identifiability (CPU, 3 seeds, single-threaded)", "",
             f"Runs: `{rel(E0_DEV)}` (commit `{manifest_commit(E0_DEV)}`) for D0.1, D0.3 and the original D0.2; "
             f"`{rel(E0_SYNC)}` (commit `{manifest_commit(E0_SYNC)}`) for D0.2 with the M3 default "
             "(`route_unobserved: parent`, `sync_parent: true`). Intervals: 95% Student-t over 3 seeds.", "",
             "## D0.1 contextual composition (held-out concepts × held-out contexts)", ""]
    rows = []
    for teacher, strength in [("static", 0.0), ("contextual", 0.5), ("contextual", 1.0), ("contextual", 2.0), ("contextual", 4.0)]:
        key = f"{teacher}_{strength}"
        diff = summary["d01"][key]["m1_q_minus_m0_test_cosine"]
        cells = []
        for learner, _ in D01_LEARNERS:
            cells.append(fmt_ci(*t_ci(d01[(d01.learner == learner) & (d01.strength == strength) & (d01.teacher == teacher)].cosine), digits=3))
        rows.append([f"{teacher} {strength:g}", *cells,
                     fmt_ci(diff["mean"], diff["ci_low"], diff["ci_high"], digits=4, signed=True)])
    lines.append(md_table(["Teacher (strength)", *[l for _, l in D01_LEARNERS], "M1(q) − M0 (paired)"], rows,
                          ["---"] + ["---:"] * (len(D01_LEARNERS) + 1)))
    lines += ["", f"D0.1 gate: **{'PASS' if summary['d01']['gate_passed'] else 'FAIL'}**.", "",
              "## D0.3 factored mapping (held-out, composition-disjoint concepts; cosine)", ""]
    rows = []
    for factor in ("free", "induced", "hybrid"):
        cells = [fmt_ci(*t_ci(d03[(d03.factor == factor) & (d03["rank"] == r) & (d03.learner != "m0")].cosine), digits=3)
                 for r in (4, 8, 16, 32)]
        rows.append([factor, *cells])
    rows.append(["M0 static bundle", fmt_ci(*t_ci(d03[d03.learner == "m0"].cosine), digits=3), "", "", ""])
    lines.append(md_table(["Concept factor", "k = 4", "k = 8", "k = 16", "k = 32"], rows, ["---", "---:", "---:", "---:", "---:"]))
    lines += ["", f"D0.3 gate: **{'PASS' if summary['d03']['gate_passed'] else 'FAIL'}**.", ""]
    for title, frame, run in [("D0.2 split detection — M3 default (parent fallback, synced parents)", sync, E0_SYNC),
                              ("D0.2 split detection — original development run", dev, E0_DEV)]:
        lines += [f"## {title}", "", f"Run `{rel(run)}`. Precision, recall and false-split rate at convergence (after "
                  "consolidation); ARI on training usages; held-out cosine of composed concepts.", ""]
        rows = []
        d = frame[frame.experiment == "D0.2"]
        for target in ("atomics", "relations"):
            for policy, label in [("none", "no growth"), *D02_POLICIES, ("oracle", "oracle")]:
                g = d[(d.target == target) & (d.policy == policy)]
                rows.append([target, label, f"{g.splits.mean():.1f}", fmt_rate(g.precision), fmt_rate(g.recall),
                             fmt_rate(g.ari), fmt_rate(g.false_split_rate), fmt_rate(g.raw_false_split_rate),
                             fmt_ci(*t_ci(g.test_cosine), digits=4)])
        lines.append(md_table(["Target", "Policy", "Splits", "Precision", "Recall", "ARI (train usages)",
                               "False-split rate", "Raw false-split rate", "Held-out cosine"], rows,
                              ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        paired = []
        for target in ("atomics", "relations"):
            base = d[(d.target == target) & (d.policy == "none")].set_index("seed").test_cosine
            for policy, label in D02_POLICIES:
                other = d[(d.target == target) & (d.policy == policy)].set_index("seed").test_cosine
                diffs = (other - base).dropna()
                paired.append([target, label, fmt_ci(*t_ci(diffs.tolist()), digits=4, signed=True), len(diffs)])
        lines += ["", "Held-out cosine change from growth, paired by seed (policy − no growth; 95% t interval):", "",
                  md_table(["Target", "Policy", "Δ held-out cosine", "Seeds"], paired, ["---", "---", "---:", "---:"]), ""]
    lines += ["Gate (experiments.md E0.2): precision ≥ 0.9, recall ≥ 0.9, ARI ≥ 0.8, false-split rate ≤ 5%. Rates: mean over "
              "3 seeds (min–max); `n/a` = undefined in at least one seed (no splits, or nothing to split). Held-out cosine: "
              "mean [95% t interval].", "", generated_note([E0_DEV / "metrics.csv", E0_DEV / "summary.json", E0_SYNC / "metrics.csv"])]
    outputs.append(write_table("e0", "\n".join(lines)))
    record("tables/e0", [E0_DEV / "metrics.csv", E0_DEV / "summary.json", E0_SYNC / "metrics.csv"], [E0_DEV, E0_SYNC])
    return outputs


# ---------------------------------------------------------------- E2

E2_RUN = EXP / "e2-mapping-operator-frontier" / "runs" / "v2"
E2_MAPPINGS = ["binary", "salience", "induced_k4", "induced_k8", "induced_k16", "induced_k32", "hybrid_k8"]
E2_MAPPING_LABELS = {"binary": "binary (M0)", "salience": "salience (01b)", "induced_k4": "induced k=4",
                     "induced_k8": "induced k=8", "induced_k16": "induced k=16", "induced_k32": "induced k=32",
                     "hybrid_k8": "hybrid k=8 + δ"}
E2_OPERATORS = ["hrr", "hrr_identity", "diagonal", "low_rank_tied", "orthogonal", "random_fixed:hrr", "untyped"]
E2_OPERATOR_LABELS = {"hrr": "hrr", "hrr_identity": "hrr (id. init)", "diagonal": "diagonal",
                      "low_rank_tied": "low-rank (tied)", "orthogonal": "orthogonal", "random_fixed:hrr": "random fixed",
                      "untyped": "untyped"}
FAMILY_LABEL = {"mesh": "MeSH 2026", "wordnet": "WordNet 3.0"}


def e2_cells() -> pd.DataFrame:
    rows = [json.loads(line) for line in (E2_RUN / "metrics.jsonl").read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def e2() -> list[str]:
    df = e2_cells()
    summary = load_json(E2_RUN / "summary.json")
    full = df[(df.budget == 1.0) & (~df.shuffled)]
    outputs: list[str] = []
    families = ["wordnet", "mesh"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), gridspec_kw={"wspace": 0.55})
    for ax, family in zip(axes, families):
        f = full[full.family == family]
        grid = np.full((len(E2_MAPPINGS), len(E2_OPERATORS)), np.nan)
        half = np.full_like(grid, np.nan)
        for i, mapping in enumerate(E2_MAPPINGS):
            for j, operator in enumerate(E2_OPERATORS):
                m, lo, hi = t_ci(f[(f.mapping == mapping) & (f.operator == operator)].test_mrr)
                grid[i, j], half[i, j] = m, (hi - m)
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        im = ax.imshow(grid, cmap=SEQUENTIAL, vmin=vmin - 0.15 * (vmax - vmin), vmax=vmax, aspect="auto")
        sel = summary[family]["selected_by_validation"]
        si, sj = E2_MAPPINGS.index(sel["mapping"]), E2_OPERATORS.index(sel["operator"])
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                dark = (value - vmin) / (vmax - vmin + 1e-12) > 0.55
                colour = "#ffffff" if dark else INK
                ax.text(j, i - 0.12, f"{100 * value:.2f}", ha="center", va="center", fontsize=6.4, color=colour)
                ax.text(j, i + 0.26, f"±{100 * half[i, j]:.2f}", ha="center", va="center", fontsize=5.0, color=colour)
        ax.add_patch(plt.Rectangle((sj - 0.5, si - 0.5), 1, 1, fill=False, edgecolor=SERIES[1], linewidth=1.8))
        ax.set_xticks(range(len(E2_OPERATORS))); ax.set_xticklabels([E2_OPERATOR_LABELS[o] for o in E2_OPERATORS], rotation=40, ha="right")
        ax.set_yticks(range(len(E2_MAPPINGS))); ax.set_yticklabels([E2_MAPPING_LABELS[m] for m in E2_MAPPINGS])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{FAMILY_LABEL[family]}: test MRR × 100 (node-disjoint)", loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cbar.outline.set_visible(False); cbar.ax.tick_params(labelsize=6.5, length=2)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.1f}"))
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.text(0.5, -0.13, "Cell: mean over 3 seeds; ± = half-width of the 95% t interval (most cell differences are inside it). "
             "Outlined: the cell selected on validation MRR. Colour scale is per family.", ha="center", fontsize=7, color=INK2)
    outputs += save(fig, "e2_mapping_operator_heatmap")
    record("figures/e2_mapping_operator_heatmap", [E2_RUN / "metrics.jsonl", E2_RUN / "summary.json"], [E2_RUN])

    # ---- paired contrasts (forest plot)
    contrasts = [("induced_vs_salience", "induced − salience"), ("induced_vs_binary", "induced − binary"),
                 ("hybrid_vs_induced_k8", "hybrid k8 − induced k8"), ("selected_vs_random_fixed", "selected − random fixed"),
                 ("selected_vs_untyped", "selected − untyped")]
    shuffled = [(op, f"true − shuffled labels: {E2_OPERATOR_LABELS[op]}") for op in E2_OPERATORS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, sharex=True, gridspec_kw={"wspace": 0.08})
    labels = [l for _, l in contrasts] + [l for _, l in shuffled]
    for ax, family in zip(axes, families):
        s = summary[family]
        values = [s[k] for k, _ in contrasts] + [s["true_minus_shuffled_labels"][op] for op, _ in shuffled]
        y = np.arange(len(values))[::-1]
        for yi, v, label in zip(y, values, labels):
            colour = SERIES[0] if label.startswith("true") else SERIES[1]
            excludes = v["ci_low"] > 0 or v["ci_high"] < 0
            ax.plot([100 * v["ci_low"], 100 * v["ci_high"]], [yi, yi], color=colour, linewidth=1.6, solid_capstyle="round")
            ax.plot(100 * v["mean"], yi, marker="o" if excludes else "o", markersize=4.8,
                    markerfacecolor=colour if excludes else "#ffffff", markeredgecolor=colour, markeredgewidth=1.3)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.axhline(len(shuffled) - 0.5, color=AXIS, linewidth=0.6)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_title(f"{FAMILY_LABEL[family]}", loc="left")
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Δ test MRR × 100 (paired over 3 seeds, 95% t CI)", fontsize=8, color=INK2, y=-0.03)
    fig.text(0.55, -0.1, "Filled marker: CI excludes 0. Orange: contrasts on the validation-selected operator (MeSH: diagonal; "
             "WordNet: low-rank tied).\nBlue: true minus derangement-shuffled relation labels, primary mapping (induced k = 8).",
             ha="center", va="top", fontsize=6.8, color=INK2)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    outputs += save(fig, "e2_contrasts")
    record("figures/e2_contrasts", [E2_RUN / "summary.json"], [E2_RUN])

    # ---- table
    lines = ["# E2 — mapping × operator frontier on frozen GPT-2 concept anchors", "",
             f"Run `{rel(E2_RUN)}` (commit `{manifest_commit(E2_RUN)}`, RTX 3090). 6,000 concepts per family, node-disjoint "
             "split (test 20%, validation 10%), 3 seeds (11, 22, 33), composer width 256, 600 steps. "
             "Test MRR among the evaluated concepts (shared candidate set, mid-rank ties). Mean and 95% Student-t "
             "interval over seeds. Selection is on validation MRR only.", ""]
    for family in families:
        f = full[full.family == family]
        s = summary[family]
        lines += [f"## {FAMILY_LABEL[family]}", "", "Test MRR × 100 (mapping × operator):", ""]
        rows = []
        for mapping in E2_MAPPINGS:
            row = [E2_MAPPING_LABELS[mapping]]
            for operator in E2_OPERATORS:
                m, lo, hi = t_ci(f[(f.mapping == mapping) & (f.operator == operator)].test_mrr)
                mark = "**" if (mapping, operator) == (s["selected_by_validation"]["mapping"], s["selected_by_validation"]["operator"]) else ""
                row.append(f"{mark}{100 * m:.2f} ± {100 * (hi - m):.2f}{mark}")
            rows.append(row)
        lines.append(md_table(["Mapping", *[E2_OPERATOR_LABELS[o] for o in E2_OPERATORS]], rows,
                              ["---"] + ["---:"] * len(E2_OPERATORS)))
        lines += ["", f"Bold: validation-selected cell ({s['selected_by_validation']['mapping']} / "
                      f"{s['selected_by_validation']['operator']}).", "", "Paired contrasts (test MRR, 95% t CI over seeds):", ""]
        rows = [[label, fmt_ci(s[k]["mean"], s[k]["ci_low"], s[k]["ci_high"], signed=True)] for k, label in contrasts]
        rows += [[label, fmt_ci(s["true_minus_shuffled_labels"][op]["mean"], s["true_minus_shuffled_labels"][op]["ci_low"],
                                s["true_minus_shuffled_labels"][op]["ci_high"], signed=True)] for op, label in shuffled]
        lines.append(md_table(["Contrast", "Δ test MRR"], rows, ["---", "---:"]))
        lines += ["", f"Gate: induced ≥ salience **{s['gate_induced_ge_salience']}**; induced > binary (CI > 0) "
                      f"**{s['gate_induced_beats_binary']}**.", ""]
        fam = f[(f.mapping == s["selected_by_validation"]["mapping"]) & (f.operator == s["selected_by_validation"]["operator"])]
        lines += [f"Selected cell: test cosine {fmt_ci(*t_ci(fam.test_cosine), digits=3)}, test variance explained "
                  f"{fmt_ci(*t_ci(fam.test_variance_explained), digits=2)}, parameters {int(fam.parameters.mean()):,} "
                  f"(atomic {int(fam.params_atomic.mean()):,}, relation {int(fam.params_relation.mean()):,}, global "
                  f"{int(fam.params_global.mean()):,}, concept-local {int(fam.params_concept_local.mean()):,}).", ""]
    budget_rows = []
    for family in families:
        for key, curve in sorted(summary[family]["budget_curves"].items()):
            if key.split("|")[1] not in ("hrr", "random_fixed:hrr", "untyped"):
                continue
            budget_rows.append([FAMILY_LABEL[family], key.replace("|", " / "),
                                *[f"{100 * curve[b]:.2f}" for b in ("0.25", "0.5", "1.0")]])
    lines += ["## Training-budget curves (test MRR × 100, mean over seeds)", "",
              md_table(["Family", "Mapping / operator", "25% of train", "50%", "100%"], budget_rows,
                       ["---", "---", "---:", "---:", "---:"]), "", generated_note([E2_RUN / "metrics.jsonl", E2_RUN / "summary.json"])]
    outputs.append(write_table("e2", "\n".join(lines)))
    record("tables/e2", [E2_RUN / "metrics.jsonl", E2_RUN / "summary.json"], [E2_RUN])
    return outputs


# ---------------------------------------------------------------- corpora, holdouts and cardinality

C3_RUN = EXP / "c3-general-corpus" / "runs" / "v2"
HOST_RUNS = {"SmolLM2": EXP / "c3-general-corpus" / "runs" / "host-smollm2-v1",
             "Qwen2.5": EXP / "c3-general-corpus" / "runs" / "host-qwen2.5-v1"}
T1_RUN = EXP / "t1-open-clinical" / "runs" / "v1"
TRACK_RUNS = {"T3 product": EXP / "t3-product-catalogues" / "runs" / "v1",
              "T4 chemistry": EXP / "t4-chemistry" / "runs" / "v1",
              "T5 glossary": EXP / "t5-enterprise-glossary" / "runs" / "v1",
              "T6 legal": EXP / "t6-legal-regulatory" / "runs" / "v1"}
C6_SUMMARY = EXP / "c6-devtools-benchmark" / "v1" / "summary.json"
TOKENIZERS = [("gpt2", "GPT-2 BPE"), ("HuggingFaceTB/SmolLM2-135M", "SmolLM2"), ("Qwen/Qwen2.5-0.5B", "Qwen2.5")]


def cardinality_sources() -> list[tuple[str, str, dict[str, list[dict[str, Any]]], Path]]:
    """(panel title, corpus description, {tokenizer: rows}, source file)."""
    out = [("C3 general", "FineWeb-Edu × WordNet 3.0", load_json(C3_RUN / "cardinality.json"), C3_RUN / "cardinality.json")]
    by_source = load_json(T1_RUN / "cardinality_by_source.json")
    out.append(("T1-open clinical", "PubMed 2026 × MeSH 2026", by_source["pubmed"], T1_RUN / "cardinality_by_source.json"))
    descriptions = {"T3 product": "ESCI products × Google/Shopify taxonomy", "T4 chemistry": "ChEBI text × ChEBI 255",
                    "T5 glossary": "synthetic enterprise documents × glossary", "T6 legal": "EUR-Lex × EuroVoc 4.24"}
    for name, run in TRACK_RUNS.items():
        out.append((name, descriptions[name], load_json(run / "cardinality.json"), run / "cardinality.json"))
    return out


def cardinality() -> list[str]:
    sources = cardinality_sources()
    outputs: list[str] = []
    metrics = [("covered_token_fraction", "covered-token fraction"),
               ("mean_distinct_entries_per_window", "distinct linked entries\nper 1,024 tokens")]
    fig, axes = plt.subplots(len(metrics), len(sources), figsize=(7.4, 3.5), sharex=True, sharey="row",
                             gridspec_kw={"hspace": 0.18, "wspace": 0.12})
    for col, (title, _, table, _) in enumerate(sources):
        for row, (metric, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for k, (tok, tlabel) in enumerate(TOKENIZERS):
                rows = sorted(table.get(tok, []), key=lambda r: r["min_subtokens"])
                if not rows:
                    continue
                ax.plot([r["min_subtokens"] for r in rows], [max(r[metric], 1e-4) for r in rows], color=SERIES[k],
                        marker=MARKERS[k], label=tlabel, linestyle=("-", "--", ":")[k])
            ax.set_yscale("log")
            ax.set_xticks([1, 2, 3, 4]); ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=8)
            if col == 0:
                ax.set_ylabel(mlabel)
            ax.axvline(2, color=AXIS, linewidth=0.6, zorder=1)
    fig.supxlabel("ℓ_min (minimum span length in subtokens)", fontsize=8, color=INK2, y=0.0)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(3.4, -1.62), ncol=3, handlelength=2.2)
    fig.text(0.5, -0.15, "Evaluation samples of each corpus (C3: first 2,000 documents; T1: PubMed evaluation sample; T3–T6: "
             "domain evaluation sample). Vertical rule: the default ℓ_min = 2.", ha="center", fontsize=7, color=INK2)
    outputs += save(fig, "cardinality_by_lmin")
    record("figures/cardinality_by_lmin", [s[3] for s in sources], [C3_RUN, T1_RUN, *TRACK_RUNS.values()])

    # ---- cardinality table
    lines = ["# Span cardinality by ℓ_min, tokenizer and corpus", "",
             "Linked-span statistics on each corpus's evaluation sample (formulation §4.1). `linked` = distinct ontology "
             "entries linked; `covered` = fraction of tokens inside linked spans; `/1k` = mean distinct linked entries per "
             "1,024-token window; `≥10` = entries seen at least 10 times in the sample.", ""]
    for title, description, table, source in sources:
        lines += [f"## {title} — {description}", "", f"Source `{rel(source)}`.", ""]
        rows = []
        for tok, tlabel in TOKENIZERS:
            for r in sorted(table.get(tok, []), key=lambda r: r["min_subtokens"]):
                rows.append([tlabel, r["min_subtokens"], f"{r['linkable_entries']:,}", f"{r['linked_entries']:,}",
                             f"{r['span_occurrences']:,}", f"{r['covered_token_fraction']:.3f}",
                             f"{r['mean_distinct_entries_per_window']:.1f}", f"{r['entries_seen_10_plus']:,}",
                             f"{r['tokens']:,}"])
        lines.append(md_table(["Tokenizer", "ℓ_min", "Linkable", "Linked", "Spans", "Covered", "/1k", "≥10", "Sample tokens"],
                              rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
        lines.append("")
    lines.append(generated_note([s[3] for s in sources]))
    outputs.append(write_table("cardinality", "\n".join(lines)))
    record("tables/cardinality", [s[3] for s in sources], [C3_RUN, T1_RUN, *TRACK_RUNS.values()])

    # ---- corpora and holdouts table
    outputs.append(corpora_table())
    return outputs


def _short(sha: str | None, n: int = 16) -> str:
    return f"`{sha[:n]}…`" if sha else "—"


def corpora_table() -> str:
    rows, inputs, runs = [], [], []
    c3 = load_json(C3_RUN / "summary.json"); inputs.append(C3_RUN / "summary.json"); runs.append(C3_RUN)
    rows.append(["C3 general", "FineWeb-Edu sample/10BT (2 shards) × WordNet 3.0", "GPT-2 BPE",
                 f"{c3['train_corpus']['tokens']:,}", f"{c3['eval_corpus']['tokens']:,}",
                 f"{c3['ontology']['concepts']:,} / {c3['ontology']['entries']:,}",
                 f"{c3['ontology']['atomics']:,} / {c3['ontology']['relations']}",
                 f"{c3['holdout']['heldout_entries']:,} ({c3['holdout']['concepts']:,} concepts)",
                 _short(c3["holdout"]["sha256"]), "general (E4 core)", f"`{rel(C3_RUN)}` @ `{manifest_commit(C3_RUN)}`"])
    for host, run in HOST_RUNS.items():
        s = load_json(run / "summary.json"); inputs.append(run / "summary.json"); runs.append(run)
        rows.append([f"C3 host corpus ({host})", "same documents, re-tokenized; same linker and holdout", s["tokenizer"],
                     f"{s['train_corpus']['tokens']:,}", f"{s['eval_corpus']['tokens']:,}", "as C3", "as C3",
                     f"{s['holdout']['heldout_entries']:,} (0 spans in train)", _short(s["holdout"]["sha256"]),
                     "continued pretraining", f"`{rel(run)}` @ `{manifest_commit(run)}`"])
    t1 = load_json(T1_RUN / "summary.json"); inputs.append(T1_RUN / "summary.json"); runs.append(T1_RUN)
    feas = load_json(T1_RUN / "feasibility.json"); inputs.append(T1_RUN / "feasibility.json")
    for tok, info in t1["corpora"].items():
        t1_verdict = next((r["verdict"] for r in feas["by_tokenizer"].get(tok, []) if r["min_subtokens"] == 2), "—")
        c = info["corpora"]
        rows.append(["T1-open clinical" + (" (CPT)" if tok != "gpt2" else ""), "PubMed 2026 baseline (20 newest files) 50/50 "
                     "with C3 FineWeb-Edu × MeSH 2026", tok, f"{c['train']['tokens']:,}",
                     f"{c['eval-pubmed']['tokens']:,} (PubMed) / {c['eval-general']['tokens']:,} (general)",
                     f"{t1['ontology']['concepts']:,} / {t1['ontology']['entries']:,}",
                     f"{t1['ontology']['atomics']:,} / {t1['ontology']['relations']}",
                     f"{t1['holdout']['heldout_entries']:,}", _short(t1["holdout"]["sha256"]),
                     f"ℓ_min 2: {t1_verdict}", f"`{rel(T1_RUN)}` @ `{manifest_commit(T1_RUN)}`"])
    for name, run in TRACK_RUNS.items():
        s = load_json(run / "summary.json"); inputs.append(run / "summary.json"); runs.append(run)
        verdict = s["feasibility"]["verdict"] + (f" at ℓ_min {s['feasibility']['min_subtokens']}" if s["feasibility"].get("min_subtokens") else
                                                 " (seen-rare stratum; held-out + locality only)")
        mix = s["mix"]
        rows.append([name, f"{Path(s['ontology']['source']).name} (domain share {mix['domain_fraction']:.2f})" if "source" in s["ontology"]
                     else f"synthetic glossary (domain share {mix['domain_fraction']:.2f})", s["train_corpus"]["tokenizer"],
                     f"{s['train_corpus']['tokens']:,}", f"{s['eval_corpus']['tokens']:,} (domain) / {s['eval_general_corpus']['tokens']:,} (general)",
                     f"{s['ontology']['concepts']:,} / {s['ontology']['entries']:,}",
                     f"{s['ontology']['atomics']:,} / {s['ontology']['relations']}",
                     f"{s['holdout']['heldout_entries']:,} (+{s['synthetic']['kept']} synthetic)", _short(s["holdout"]["sha256"]),
                     verdict, f"`{rel(run)}` @ `{manifest_commit(run)}`"])
    c6 = load_json(C6_SUMMARY); inputs.append(C6_SUMMARY)
    rows.append(["T2 developer tools (C6 benchmark v1)", "synthetic private library: docs and usage", "—",
                 f"{c6['train_docs']:,} docs", f"{c6['eval_docs']:,} docs", f"{c6['symbols']} symbols", "—",
                 f"{c6['heldout_symbols']} symbols (leaked into train: {c6['leaked_into_train']})", "—",
                 "track corpus not yet built (execution.md decision 18)", f"`{rel(C6_SUMMARY.parent)}`"])
    text = "\n".join([
        "# Corpora, ontologies and frozen holdouts", "",
        "Every held-out list is node- and alias-disjoint and was hashed before the first run that could see it "
        "(tasks.md rule 2). Token counts are exact counts of the built token streams. `Concepts / entries`: ontology "
        "concepts and linker entries (surface-form groups); `atomics / relations`: dictionary sizes after the "
        "max-atomics cap. Corpus text is never committed; run folders hold counts, hashes and manifests.", "",
        md_table(["Corpus", "Text × ontology", "Tokenizer", "Train tokens", "Eval tokens", "Concepts / entries",
                  "Atomics / relations", "Held-out entries", "Holdout sha256", "Use / feasibility", "Run @ commit"], rows,
                 ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---", "---", "---"]),
        "", generated_note(inputs)])
    record("tables/corpora", inputs, runs)
    return write_table("corpora", text)


# ---------------------------------------------------------------- B10 sparse vs full generation

B10 = EXP / "b10-benchmarks" / "generation.json"


def generation() -> list[str]:
    data = load_json(B10)
    rows = _generation_rows(data)
    outputs: list[str] = []
    cpu = [r for r in rows if r["device"] == "cpu" and r["batch_spans"] == 1024 and r["mode"] in ("bundle", "attentive")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"wspace": 0.32})
    for ax, (key, ylabel) in zip(axes, [("ms", "latency per train step (ms, median)"), ("mib", "extra peak memory per train step (MiB)")]):
        for c, path in enumerate(("sparse", "full")):
            for mode, style in (("bundle", "-"), ("attentive", "--")):
                pts = sorted((r["entries"], r[f"train_{path}_{key}"]) for r in cpu if r["mode"] == mode and r.get(f"train_{path}_{key}") is not None)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[c], linestyle=style, marker=MARKERS[c],
                        label=f"{path}, {'M0 bundle' if mode == 'bundle' else 'M1 attentive'}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1e4, 5e4, 1e5]); ax.set_xticklabels(["10k", "50k", "100k"])
        ax.minorticks_off()
        ax.set_xlabel("concept entries N in the table"); ax.set_ylabel(ylabel)
    axes[0].set_title("CPU, 1,024 linked spans per batch", loc="left")
    axes[1].set_title("CPU, same workload", loc="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.12, -0.25), ncol=4, handlelength=2.2, columnspacing=1.0)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    outputs += save(fig, "generation_sparse_vs_full")
    record("figures/generation_sparse_vs_full", [B10])

    table_rows = []
    for r in sorted(cpu, key=lambda r: (r["mode"], r["entries"])):
        table_rows.append([r["mode"], f"{r['entries']:,}", f"{r['unique_entries']:,}",
                           *[_fmt_num(r.get(k)) for k in ("forward_sparse_ms", "forward_full_ms", "train_sparse_ms", "train_full_ms",
                                                         "train_sparse_mib", "train_full_mib")],
                           _ratio(r.get("train_full_ms"), r.get("train_sparse_ms")), _ratio(r.get("train_full_mib"), r.get("train_sparse_mib"))])
    meta = data.get("machine", {})
    text = "\n".join([
        "# Sparse vs full channel-row generation (B10, CPU)", "",
        "Composing only the concepts linked in a batch (sparse) vs materializing the whole concept table every step "
        "(full). Composer width 256, 16 relations, 8,192 atomics, mean frame degree 6, 1,024 linked spans per batch "
        "drawn Zipf(1.1), 4 CPU threads; latency = median per step, memory = extra peak resident memory per step. "
        "GPU numbers in the source were measured on a shared GPU and are qualitative only (see the B10 report).", "",
        md_table(["Mode", "Entries N", "Unique in batch", "Fwd sparse (ms)", "Fwd full (ms)", "Train sparse (ms)",
                  "Train full (ms)", "Train sparse (MiB)", "Train full (MiB)", "Latency full/sparse", "Memory full/sparse"],
                 table_rows, ["---"] + ["---:"] * 10),
        "", (f"Commit `{meta.get('git_sha', '')[:7]}`. " if meta.get("git_sha") else "") + generated_note([B10])])
    outputs.append(write_table("generation", text))
    record("tables/generation", [B10])
    return outputs


def _fmt_num(value: Any) -> str:
    return "—" if value is None else (f"{value:,.1f}" if value >= 10 else f"{value:.2f}")


def _ratio(a: Any, b: Any) -> str:
    return "—" if a is None or b in (None, 0) else f"{a / b:.0f}×"


def _generation_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pivot `generation.json` cells (one per device × mode × N × batch × path × phase) into one row per
    (device, mode, N, batch) with `<phase>_<path>_ms` / `<phase>_<path>_mib` columns (None if OOM / n/a)."""
    merged: dict[tuple, dict[str, Any]] = {}
    for cell in data["cells"]:
        key = (cell["device"], cell["mode"], cell["entries"], cell["batch_spans"])
        row = merged.setdefault(key, {"device": cell["device"], "mode": cell["mode"], "entries": cell["entries"],
                                      "batch_spans": cell["batch_spans"], "unique_entries": cell.get("unique_entries")})
        ok = cell.get("status") == "ok"
        prefix = f"{cell['phase']}_{cell['path']}"
        row[f"{prefix}_ms"] = cell.get("latency_ms") if ok else None
        row[f"{prefix}_mib"] = cell.get("peak_mib") if ok else None
    return list(merged.values())


# ---------------------------------------------------------------- E4 (stub until analyses exist)

E4_ANALYSIS = EXP / "e4-small-lm" / "analysis"
E4_STRATA = [("after_heldout", "held-out"), ("after_rare", "seen rare"), ("after", "after span"), ("inside", "inside span"),
             ("unlinked", "unlinked (locality)"), ("all", "all tokens")]


def e4() -> list[str]:
    summaries = sorted(E4_ANALYSIS.glob("*/summary.json")) if E4_ANALYSIS.exists() else []
    if not summaries:
        text = "\n".join(["# E4 — small-LM joint training", "",
                          f"**Pending.** No `{rel(E4_ANALYSIS)}/*/summary.json` exists yet (written by "
                          "`python -m vsa_embed.experiments.e4_report --runs … --output experiments/e4-small-lm/analysis/<name>`). "
                          "Re-run this script once an analysis is committed; the S0 sanity pilot (not a gate) and D4.x cohorts "
                          "will then appear here as paired-difference tables and forest plots."])
        return [write_table("e4", text)]
    outputs, lines, inputs = [], ["# E4 — small-LM joint training (from `e4_report` analyses)", ""], []
    for path in summaries:
        name = path.parent.name
        summary = load_json(path)
        inputs.append(path)
        baseline = summary.get("baseline", "C0")
        for label, cohort in summary.get("cohorts", {}).items():
            base = cohort.get("baseline", baseline)
            paired = cohort.get("paired", {}).get(base, {})
            flags = (" — **exploratory**: " + "; ".join(cohort["exploratory"])) if cohort.get("exploratory") else ""
            lines += [f"## {name}: {label}{flags}", "",
                      f"Seeds per condition: {cohort.get('seeds')}. Paired difference condition − {base} at the final evaluation "
                      "(negative = lower loss), relative to the baseline, with the analysis's CI and Holm flag.", ""]
            strata = [(s, l) for s, l in E4_STRATA if s in cohort.get("strata", [])]
            rows = []
            for condition, by_stratum in paired.items():
                cells = []
                for s, _ in strata:
                    c = by_stratum.get(s)
                    if not c or c.get("relative") is None:
                        cells.append("—"); continue
                    lo, hi = c.get("relative_ci_low"), c.get("relative_ci_high")
                    star = "*" if c.get("significant") else ""
                    cells.append(f"{100 * c['relative']:+.2f}%" + (f" [{100 * lo:+.2f}, {100 * hi:+.2f}]" if lo is not None else "") + star)
                rows.append([condition, *cells])
            lines += [md_table(["Condition", *[l for _, l in strata]], rows, ["---"] + ["---:"] * len(strata)), ""]
            gate = cohort.get("gate", {})
            if gate:
                lines += ["Gate items: " + "; ".join(f"{c}: {g.get('overall')}" for c, g in gate.items()), ""]
            if paired:
                outputs += _e4_forest(f"e4_{name}_{_slug(label)}", label, paired, strata, base)
                record(f"figures/e4_{name}_{_slug(label)}", [path])
    lines.append(generated_note(inputs))
    outputs.append(write_table("e4", "\n".join(lines)))
    record("tables/e4", inputs)
    return outputs


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-").lower()[:40]


def _e4_forest(name: str, label: str, paired: dict[str, Any], strata: list[tuple[str, str]], base: str) -> list[str]:
    conditions = list(paired)[:8]          # categorical cap: fold the rest rather than generate hues
    fig, axes = plt.subplots(1, len(strata), figsize=(min(7.4, 1.3 * len(strata) + 1.2), 0.32 * len(conditions) + 1.2),
                             sharey=True, gridspec_kw={"wspace": 0.1})
    axes = np.atleast_1d(axes)
    y = np.arange(len(conditions))[::-1]
    for ax, (s, slabel) in zip(axes, strata):
        for k, (yi, condition) in enumerate(zip(y, conditions)):
            c = paired[condition].get(s)
            if not c or c.get("relative") is None:
                continue
            colour = SERIES[k % len(SERIES)]
            lo, hi = c.get("relative_ci_low"), c.get("relative_ci_high")
            if lo is not None:
                ax.plot([100 * lo, 100 * hi], [yi, yi], color=colour, linewidth=1.6)
            filled = bool(c.get("significant"))
            ax.plot(100 * c["relative"], yi, "o", markerfacecolor=colour if filled else "#ffffff", markeredgecolor=colour)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(slabel, fontsize=7.5, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(conditions)
    fig.supxlabel(f"relative loss change vs {base} (%), filled = Holm-significant", fontsize=7.5, color=INK2)
    fig.suptitle(label, fontsize=8.5, x=0.02, ha="left")
    return save(fig, name)


# ---------------------------------------------------------------- main

PARTS = {"e0": e0, "e2": e2, "cardinality": cardinality, "generation": generation, "e4": e4}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(PARTS), help="parts to regenerate (default: all)")
    args = parser.parse_args(argv)
    sources_path = FIG / "sources.json"
    if sources_path.exists():
        SOURCES.update(load_json(sources_path))
    written: list[str] = []
    for part in args.only or list(PARTS):
        out = PARTS[part]()
        written += out
        print(f"{part}: " + ", ".join(out))
    FIG.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps(dict(sorted(SOURCES.items())), indent=2) + "\n")


if __name__ == "__main__":
    main()
