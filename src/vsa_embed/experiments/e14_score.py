"""E14 scoring of the model-reading item sets (plan §§4–6; pre-registration §§4–5) on a frozen host, and the pooled report.

`score` — every item of every set under every condition: the condition's text, a blank line, the question; the option
with the highest length-normalised log-probability is the answer. Per-item rows go to `predictions.jsonl.gz` in the
`--data` folder (large); the summary (accuracy by set × condition × depth and subset; prompt tokens) to `--output`.

`report` — the pre-registered contrasts, paired over items:
- LG-B1 primary (§4): `exec-vsa − text` on taxonomy true/false items by depth 1–5, Holm over depths, per host;
  `exec-vsa − exec-sym`; gate G1 (both SmolLM2 hosts: positive at every depth ≥ 3 after Holm; |exec-vsa − exec-sym| ≤ 0.03).
- LG-B4: `(exec-vsa − text)` under LP minus under RP; B5: `exec-vsa − text` by k; B6: by rule-base size; C3: noise;
  C4: `icl` against `exec-vsa` by kind and k; C5: `exec-vsa − exec-stale` on propagation and locality.
- LG-D primary (§5): `path-vsa − tree-text` on trees by depth, Holm over depths; lists; edits (`path-vsa` vs `path-stale`).

`noisy-report` — LG-D2 (§19) on the `trees-noisy` set: `path-vsa@p − tree-text@p` and `path-vsa-soft@p − path-vsa@p`
(p = 0.1, 0.3; Holm over 4), by subset and depth, and how often the host follows its executor.

    python -m vsa_embed.experiments.e14_score score --host H-360 --items DIR --output OUT --data DATA [--sets ...] [--limit N]
    python -m vsa_embed.experiments.e14_score report --runs OUT ... --output DIR
    python -m vsa_embed.experiments.e14_score noisy-report --runs OUT ... --output DIR     # LG-D2 (pre-registration §19)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import holm_adjust
from .e5_common import finish_output, start_output, write_json
from .e5_zeroshot import paired_difference
from .e14_items import read_set

SETS = ("taxonomy", "rulebase", "edits", "prop-rp", "prop-lp", "composition", "fewshot", "trees")


def prompt_of(context: str, question: str) -> str:
    return f"{context}\n\n{question}" if context else question


def score_set(host: Any, rows: Sequence[dict[str, Any]], *, log=print) -> list[dict[str, Any]]:
    prompts, options, keys = [], [], []
    for r in rows:
        for cond, ctx in r["contexts"].items():
            p = prompt_of(ctx, r["question"])
            for j, o in enumerate(r["options"]):
                prompts.append(p); options.append(o); keys.append((r["id"], cond, j))
    log(f"  {len(rows)} items, {len(prompts)} sequences")
    s = host.score(prompts, options)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (rid, cond, j), v in zip(keys, s.mean):
        values[(rid, cond)].append(float(v))
    out = []
    by_id = {r["id"]: r for r in rows}
    for (rid, cond), vs in values.items():
        r = by_id[rid]
        choice = int(np.argmax(vs))
        out.append({"id": rid, "set": r["set"], "condition": cond, "choice": choice, "gold": r["gold"], "correct": int(choice == r["gold"]),
                    "depth": r.get("depth"), "format": r.get("format"), "scores": [round(v, 4) for v in vs],
                    **{k: r[k] for k in ("label", "counterfactual", "kind", "k", "sampler", "rule_count", "rules", "uses_heldout", "numeric",
                                         "noise") if k in r}})
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = f"{r['set']}|{r.get('format')}|{r['condition']}"
        acc[key]["all"].append(r["correct"])
        acc[key][f"depth={r.get('depth')}"].append(r["correct"])
        for k in ("kind", "sampler", "counterfactual", "rules", "uses_heldout", "numeric"):
            if k in r and r[k] is not None:
                acc[key][f"{k}={r[k]}"].append(r["correct"])
    return {key: {sub: {"acc": float(np.mean(v)), "n": len(v)} for sub, v in subs.items()} for key, subs in acc.items()}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    from ..logic.hosts import HostLM
    sets = [s.strip() for s in args.sets.split(",")] if args.sets else list(SETS)
    config = {"experiment": "e14-lg-score", "host": args.host, "items": str(args.items), "sets": sets, "limit": args.limit,
              "seed": args.seed, "smoke": bool(args.smoke)}
    output, data = Path(args.output), Path(args.data)
    git_at_start = start_output(output, config)
    data.mkdir(parents=True, exist_ok=True)
    host = HostLM(args.host, device=args.device, batch_size=args.batch_size)
    rows_all: list[dict[str, Any]] = []
    tokens: dict[str, dict[str, float]] = {}
    timings = {}
    for name in sets:
        path = Path(args.items) / f"{name}-s{args.seed}.jsonl.gz"
        rows = read_set(path)
        if args.limit:
            rows = rows[: args.limit]
        t0 = time.monotonic()
        print(f"{name}: {path}")
        scored = score_set(host, rows)
        rows_all += scored
        timings[name] = round(time.monotonic() - t0, 1)
        sample = rows[: min(40, len(rows))]
        tokens[name] = {c: float(np.mean([host.count(r["contexts"][c]) for r in sample if c in r["contexts"]]))
                        for c in sorted({c for r in sample for c in r["contexts"]})}
    with gzip.open(data / "predictions.jsonl.gz", "wt") as handle:
        for r in rows_all:
            handle.write(json.dumps(r) + "\n")
    summary = {"source": host.describe(), "sets": sets, "smoke": bool(args.smoke), "timings": timings, "context_tokens": tokens,
               "predictions": str(data / "predictions.jsonl.gz"), "accuracy": summarize(rows_all)}
    write_json(output / "summary.json", summary)
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [f"# E14 LG — scores on `{host.name}`", "",
             "| set · format · condition | all | by depth |", "|---|---:|---|"]
    for key in sorted(summary["accuracy"]):
        a = summary["accuracy"][key]
        depths = ", ".join(f"{k.split('=')[1]}: {v['acc']:.2f}" for k, v in sorted(a.items()) if k.startswith("depth="))
        lines.append(f"| {key} | {a['all']['acc']:.3f} ({a['all']['n']}) | {depths} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    finish_output(output, config, git_at_start=git_at_start, device=host.device, source=host.describe())
    return summary


# -- report ---------------------------------------------------------------------------------------------------------------

def load_predictions(run: Path) -> list[dict[str, Any]]:
    summary = json.loads((run / "summary.json").read_text())
    with gzip.open(summary["predictions"], "rt") as handle:
        return [json.loads(line) for line in handle]


def contrast(rows: Sequence[dict[str, Any]], a: str, b: str, *, where=lambda r: True, resamples: int, seed: int) -> dict[str, Any] | None:
    va = {r["id"]: r["correct"] for r in rows if r["condition"] == a and where(r)}
    vb = {r["id"]: r["correct"] for r in rows if r["condition"] == b and where(r)}
    ids = sorted(set(va) & set(vb))
    if not ids:
        return None
    return {"a": a, "b": b, **paired_difference(np.array([va[i] for i in ids]), np.array([vb[i] for i in ids]), resamples=resamples, seed=seed)}


def host_report(rows: Sequence[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tf = lambda s: (lambda r: r["set"] == s and r.get("format") in ("tf", "mc"))  # noqa: E731
    primary = []
    for d in range(1, 6):
        c = contrast(rows, "exec-vsa", "text", where=lambda r, d=d: tf("taxonomy")(r) and r["depth"] == d, resamples=resamples, seed=seed)
        if c:
            primary.append({**c, "depth": d})
    for row, p in zip(primary, holm_adjust([r["p_value"] for r in primary]) if primary else []):
        row["p_holm"] = p
    out["B1_primary"] = primary
    out["B1_vsa_vs_sym"] = contrast(rows, "exec-vsa", "exec-sym", where=tf("taxonomy"), resamples=resamples, seed=seed)
    sec = []
    for a, b in (("exec-vsa", "none"), ("text", "none"), ("text+demo", "text"), ("exec-vsa-conclusion", "exec-vsa"),
                 ("exec-vsa-noconf", "exec-vsa"), ("exec-shuffled", "exec-vsa"), ("exec-nobind", "exec-vsa"),
                 ("exec-vsa-stressed", "exec-vsa"), ("text-retrieved", "text"), ("exec-vsa@0.1", "exec-sym@0.1"),
                 ("exec-vsa@0.3", "exec-sym@0.3"), ("exec-vsa@0.3", "exec-vsa")):
        c = contrast(rows, a, b, where=tf("taxonomy"), resamples=resamples, seed=seed)
        if c:
            sec.append(c)
    cf = [contrast(rows, "exec-vsa", "text", where=lambda r, v=v: tf("taxonomy")(r) and r.get("counterfactual") == v,
                   resamples=resamples, seed=seed) for v in (True, False)]
    out["B1_secondary"] = sec
    out["B3_counterfactual"] = {"counterfactual": cf[0], "neutral": cf[1]}
    out["B7_cloze"] = contrast(rows, "exec-vsa", "text", where=lambda r: r["set"] == "taxonomy" and r.get("format") == "cloze",
                               resamples=resamples, seed=seed)
    b4 = {}
    for smp in ("rp", "lp"):
        b4[smp] = contrast(rows, "exec-vsa", "text", where=tf(f"prop-{smp}"), resamples=resamples, seed=seed)
    if b4.get("rp") and b4.get("lp"):
        b4["lp_minus_rp"] = b4["lp"]["mean"] - b4["rp"]["mean"]
    out["B4"] = b4
    out["B5_by_k"] = {k: contrast(rows, "exec-vsa", "text", where=lambda r, k=k: tf("composition")(r) and r["depth"] == k,
                                  resamples=resamples, seed=seed) for k in (2, 3, 4, 5, 6)}
    out["B6_by_rules"] = {n: {c: contrast(rows, c, "text", where=lambda r, n=n: tf("rulebase")(r) and r.get("rules") == n,
                                          resamples=resamples, seed=seed) for c in ("exec-vsa", "text-retrieved")} for n in (10, 50)}
    out["C4"] = {f"{kind}|k={k}": contrast(rows, "exec-vsa", "icl", where=lambda r, kind=kind, k=k: r["set"] == "fewshot"
                                           and r.get("kind") == kind and r.get("k") == k, resamples=resamples, seed=seed)
                 for kind in ("inverse", "symmetric", "subrelation", "chain") for k in (1, 3, 5, 10)}
    out["C5"] = {kind: contrast(rows, "exec-vsa", "exec-stale", where=lambda r, kind=kind: r["set"] == "edits" and r.get("kind") == kind,
                                resamples=resamples, seed=seed) for kind in ("propagation", "locality")}
    out["C5_text"] = contrast(rows, "exec-vsa", "text-edited", where=lambda r: r["set"] == "edits", resamples=resamples, seed=seed)
    d_primary = []
    for d in (2, 3, 4, 6):
        c = contrast(rows, "path-vsa", "tree-text", where=lambda r, d=d: r["set"] == "trees" and r.get("kind") == "tree" and r["depth"] == d,
                     resamples=resamples, seed=seed)
        if c:
            d_primary.append({**c, "depth": d})
    for row, p in zip(d_primary, holm_adjust([r["p_value"] for r in d_primary]) if d_primary else []):
        row["p_holm"] = p
    out["D_primary"] = d_primary
    out["D_lists"] = {n: contrast(rows, "path-vsa", "tree-text", where=lambda r, n=n: r["set"] == "trees" and r.get("kind") == "list"
                                  and r["depth"] == n, resamples=resamples, seed=seed) for n in (3, 6, 12)}
    out["D_edits"] = {"vsa_vs_stale": contrast(rows, "path-vsa", "path-stale", where=lambda r: r["set"] == "trees" and r.get("kind") == "edit",
                                               resamples=resamples, seed=seed),
                      "vsa_vs_text": contrast(rows, "path-vsa", "tree-text", where=lambda r: r["set"] == "trees" and r.get("kind") == "edit",
                                              resamples=resamples, seed=seed)}
    out["D_secondary"] = [c for c in (contrast(rows, a, b, where=lambda r: r["set"] == "trees" and r.get("kind") == "tree", resamples=resamples, seed=seed)
                                      for a, b in (("path-vsa", "path-sym"), ("leaf-vsa", "path-vsa"), ("tree-bullets", "tree-text"),
                                                   ("path-shuffled", "path-vsa"), ("tree-text", "none"))) if c]
    return out


def fmt(c: dict[str, Any] | None) -> str:
    if not c:
        return "—"
    s = f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] (n {c['n']}"
    return s + (f", Holm p {c['p_holm']:.4f})" if "p_holm" in c else f", p {c['p_value']:.4f})")


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    doc: dict[str, Any] = {"hosts": {}, "missing": missing}
    lines = ["# E14 LG-B / LG-C / LG-D — pooled report", ""] + ([f"Missing: {', '.join(missing)}", ""] if missing else [])
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in present:                                     # a host's job folders (e.g. LG-B/C and LG-D) are pooled
        by_host[json.loads((run / "summary.json").read_text())["source"]["host"]] += load_predictions(run)
    for host, rows in by_host.items():
        rep = host_report(rows, resamples=args.resamples, seed=args.seed)
        doc["hosts"][host] = rep
        lines += [f"## {host}", "", "**LG-B1 primary: exec-vsa − text (taxonomy, by depth; Holm)**", ""]
        lines += [f"- depth {c['depth']}: {fmt(c)}" for c in rep["B1_primary"]]
        lines += [f"- exec-vsa − exec-sym: {fmt(rep['B1_vsa_vs_sym'])}", "", "Secondary (taxonomy):", ""]
        lines += [f"- {c['a']} − {c['b']}: {fmt(c)}" for c in rep["B1_secondary"]]
        lines += [f"- B3 exec-vsa − text, counterfactual: {fmt(rep['B3_counterfactual']['counterfactual'])}; neutral: {fmt(rep['B3_counterfactual']['neutral'])}",
                  f"- B7 cloze exec-vsa − text: {fmt(rep['B7_cloze'])}",
                  f"- B4 exec-vsa − text, RP: {fmt(rep['B4'].get('rp'))}; LP: {fmt(rep['B4'].get('lp'))}; LP − RP: "
                  f"{rep['B4'].get('lp_minus_rp', float('nan')):+.3f}",
                  "- B5 exec-vsa − text by k: " + "; ".join(f"{k}: {fmt(v)}" for k, v in rep["B5_by_k"].items()),
                  "- B6 by rule-base size: " + "; ".join(f"{n} rules: exec-vsa − text {fmt(v['exec-vsa'])}, retrieved − text {fmt(v['text-retrieved'])}"
                                                        for n, v in rep["B6_by_rules"].items()),
                  "- C4 exec-vsa − icl: " + "; ".join(f"{k}: {fmt(v)}" for k, v in rep["C4"].items() if v),
                  f"- C5 exec-vsa − exec-stale: propagation {fmt(rep['C5']['propagation'])}; locality {fmt(rep['C5']['locality'])}; "
                  f"exec-vsa − text-edited {fmt(rep['C5_text'])}",
                  "", "**LG-D primary: path-vsa − tree-text (trees, by depth; Holm)**", ""]
        lines += [f"- depth {c['depth']}: {fmt(c)}" for c in rep["D_primary"]]
        lines += ["- lists: " + "; ".join(f"length {n}: {fmt(v)}" for n, v in rep["D_lists"].items()),
                  f"- edits: path-vsa − path-stale {fmt(rep['D_edits']['vsa_vs_stale'])}; path-vsa − tree-text {fmt(rep['D_edits']['vsa_vs_text'])}"]
        lines += [f"- {c['a']} − {c['b']}: {fmt(c)}" for c in rep["D_secondary"]]
        lines.append("")
    small = [h for h in doc["hosts"] if "SmolLM2" in h]
    if small:
        def passes(h: str) -> bool:
            rep = doc["hosts"][h]
            deep = [c for c in rep["B1_primary"] if c["depth"] >= 3]
            return bool(deep) and all(c["mean"] > 0 and c.get("p_holm", 1) < 0.05 for c in deep) and \
                rep["B1_vsa_vs_sym"] is not None and abs(rep["B1_vsa_vs_sym"]["mean"]) <= 0.03
        doc["gate_G1"] = {h: passes(h) for h in small}
        lines += ["## Gate G1 (pre-registration §4)", "", *[f"- {h}: {'pass' if v else 'fail'}" for h, v in doc["gate_G1"].items()]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def run_verdict_report(args: argparse.Namespace) -> dict[str, Any]:
    """LG-B verdict format (pre-registration §14): per host, on `taxonomy-verdict` true/false items, `exec-vsa-verdict − text`
    by depth 1–5 (Holm over depths; the G1 re-test), `exec-vsa-verdict − exec-sym-verdict`, and the format effect
    `<source>-verdict − <source>`; accuracy by gold label; the same on `rulebase-verdict`. Gate G1′: both SmolLM2 hosts
    positive (Holm) at every depth ≥ 3 and |exec-vsa-verdict − exec-sym-verdict| ≤ 0.03."""
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    doc: dict[str, Any] = {"hosts": {}, "missing": missing}
    lines = ["# E14 LG-B verdict format — pooled report (pre-registration §14)", ""] + ([f"Missing: {', '.join(missing)}", ""] if missing else [])
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in present:
        by_host[json.loads((run / "summary.json").read_text())["source"]["host"]] += load_predictions(run)
    kw = {"resamples": args.resamples, "seed": args.seed}
    for host, rows in by_host.items():
        tax = lambda r: r["set"] == "taxonomy-verdict"  # noqa: E731
        primary = []
        for d in range(1, 6):
            c = contrast(rows, "exec-vsa-verdict", "text", where=lambda r, d=d: tax(r) and r.get("depth") == d, **kw)
            if c:
                primary.append({**c, "depth": d})
        for c, p in zip(primary, holm_adjust([c["p_value"] for c in primary]) if primary else []):
            c["p_holm"] = p
        rep: dict[str, Any] = {"primary": primary,
                               "vsa_vs_sym": contrast(rows, "exec-vsa-verdict", "exec-sym-verdict", where=tax, **kw),
                               "format": {f"{s}-verdict − {s}": contrast(rows, f"{s}-verdict", s, where=tax, **kw)
                                          for s in ("exec-vsa", "exec-sym", "exec-vsa-conclusion")},
                               "format_false": contrast(rows, "exec-vsa-verdict", "exec-vsa", where=lambda r: tax(r) and not r["label"], **kw),
                               "rulebase": {"exec-vsa-verdict − text": contrast(rows, "exec-vsa-verdict", "text",
                                                                                where=lambda r: r["set"] == "rulebase-verdict", **kw),
                                            "exec-vsa-verdict − exec-vsa": contrast(rows, "exec-vsa-verdict", "exec-vsa",
                                                                                    where=lambda r: r["set"] == "rulebase-verdict", **kw)}}
        acc: dict[tuple, list[int]] = defaultdict(list)
        for r in rows:
            if r["set"] in ("taxonomy-verdict", "rulebase-verdict"):
                acc[(r["set"], r["condition"], bool(r["label"]))].append(r["correct"])
        rep["by_label"] = {f"{s}|{c}|{'true' if lab else 'false'}": {"acc": float(np.mean(v)), "n": len(v)} for (s, c, lab), v in acc.items()}
        doc["hosts"][host] = rep
        lines += [f"## {host}", "", "**Primary: exec-vsa-verdict − text (taxonomy, by depth; Holm)**", ""]
        lines += [f"- depth {c['depth']}: {fmt(c)}" for c in primary]
        lines += [f"- exec-vsa-verdict − exec-sym-verdict: {fmt(rep['vsa_vs_sym'])}", "", "Format effect (taxonomy):", ""]
        lines += [f"- {k}: {fmt(v)}" for k, v in rep["format"].items()]
        lines += [f"- exec-vsa-verdict − exec-vsa, false items only: {fmt(rep['format_false'])}", "", "Rulebase:", ""]
        lines += [f"- {k}: {fmt(v)}" for k, v in rep["rulebase"].items()]
        lines += ["", "| set · condition | true items | false items |", "|---|---:|---:|"]
        for s in ("taxonomy-verdict", "rulebase-verdict"):
            for c in sorted({k.split("|")[1] for k in rep["by_label"] if k.startswith(s + "|")}):
                t, f = rep["by_label"].get(f"{s}|{c}|true"), rep["by_label"].get(f"{s}|{c}|false")
                lines.append(f"| {s} · {c} | {t['acc']:.3f} ({t['n']}) | {f['acc']:.3f} ({f['n']}) |" if t and f else f"| {s} · {c} | — | — |")
        lines.append("")
    small = [h for h in doc["hosts"] if "SmolLM2" in h]
    if small:
        def passes(h: str) -> bool:
            rep = doc["hosts"][h]
            deep = [c for c in rep["primary"] if c["depth"] >= 3]
            return bool(deep) and all(c["mean"] > 0 and c.get("p_holm", 1) < 0.05 for c in deep) and \
                rep["vsa_vs_sym"] is not None and abs(rep["vsa_vs_sym"]["mean"]) <= 0.03
        doc["gate_G1_verdict"] = {h: passes(h) for h in small}
        lines += ["## Gate G1′ (pre-registration §14)", "", *[f"- {h}: {'pass' if v else 'fail'}" for h, v in doc["gate_G1_verdict"].items()]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


NOISE_LEVELS = ("0.1", "0.3")


def noisy_host_report(rows: Sequence[dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    """LG-D2 (pre-registration §19) for one host's `trees-noisy` predictions."""
    kw = {"resamples": resamples, "seed": seed}
    rows = [r for r in rows if r["set"] == "trees-noisy"]
    every = lambda r: True  # noqa: E731
    primary = []
    for fam, (a, b) in (("a", ("path-vsa@{p}", "tree-text@{p}")), ("b", ("path-vsa-soft@{p}", "path-vsa@{p}"))):
        for p in NOISE_LEVELS:
            c = contrast(rows, a.format(p=p), b.format(p=p), where=every, **kw)
            if c:
                primary.append({**c, "family": fam, "noise": p})
    for c, ph in zip(primary, holm_adjust([c["p_value"] for c in primary]) if primary else []):
        c["p_holm"] = ph
    secondary = []
    for p in NOISE_LEVELS:
        subsets = {"all": every, "affected": lambda r, p=p: r["noise"][p]["affected"],
                   "missing on path": lambda r, p=p: r["noise"][p]["missing_on_path"],
                   "wrong on path": lambda r, p=p: r["noise"][p]["wrong_on_path"],
                   "unaffected": lambda r, p=p: not r["noise"][p]["affected"]}
        for a, b in ((f"path-vsa@{p}", "path-vsa"), (f"tree-text@{p}", "tree-text"), (f"path-vsa-soft@{p}", f"path-sym-soft@{p}"),
                     (f"path-vsa@{p}", f"tree-text@{p}"), (f"path-vsa-soft@{p}", f"path-vsa@{p}"), (f"tree-text@{p}", f"none@{p}")):
            for name, where in subsets.items():
                c = contrast(rows, a, b, where=where, **kw)
                if c:
                    secondary.append({**c, "noise": p, "subset": name})
            for d in sorted({r["depth"] for r in rows}):
                c = contrast(rows, a, b, where=lambda r, d=d: r["depth"] == d, **kw)
                if c:
                    secondary.append({**c, "noise": p, "subset": f"depth {d}"})
    acc: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["condition"]]["all"].append(r["correct"])
        acc[r["condition"]][f"depth {r['depth']}"].append(r["correct"])
    accuracy = {c: {k: float(np.mean(v)) for k, v in subs.items()} for c, subs in acc.items()}
    follow, ceilings = {}, {}
    for p in NOISE_LEVELS:
        for cond in ("path-vsa", "path-vsa-soft", "path-sym-soft"):
            rs = [r for r in rows if r["condition"] == f"{cond}@{p}"]
            if not rs:
                continue
            ex = [r["noise"][p]["exec"][cond] for r in rs]
            wrong = [r for r, e in zip(rs, ex) if e != r["gold"]]
            follow[f"{cond}@{p}"] = {"follow": float(np.mean([r["choice"] == e for r, e in zip(rs, ex)])),
                                     "follow_when_executor_wrong": float(np.mean([r["choice"] == r["noise"][p]["exec"][cond] for r in wrong])) if wrong else None,
                                     "n_executor_wrong": len(wrong)}
            ceilings[f"{cond}@{p}"] = float(np.mean([e == r["gold"] for r, e in zip(rs, ex)]))
    return {"primary": primary, "secondary": secondary, "accuracy": accuracy, "follow": follow, "executor_ceilings": ceilings,
            "items": len({r["id"] for r in rows})}


def run_noisy_report(args: argparse.Namespace) -> dict[str, Any]:
    """LG-D2 (pre-registration §19): per host, primary (Holm over 4) (a) `path-vsa@p − tree-text@p` and (b)
    `path-vsa-soft@p − path-vsa@p` for p = 0.1, 0.3; secondary contrasts by subset and depth; accuracy by condition; how often
    the host follows its executor (also when the executor is wrong); the executors' own ceilings."""
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    doc: dict[str, Any] = {"hosts": {}, "missing": missing}
    lines = ["# E14 LG-D2 — decision trees on noisy or partial records (pre-registration §19)", ""] + \
        ([f"Missing: {', '.join(missing)}", ""] if missing else [])
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in present:
        by_host[json.loads((run / "summary.json").read_text())["source"]["host"]] += load_predictions(run)
    smoke = any(json.loads((run / "summary.json").read_text()).get("smoke") for run in present)
    if smoke:
        lines[:0] = ["**SMOKE TEST — not a result.**", ""]
    for host, rows in by_host.items():
        rep = noisy_host_report(rows, resamples=args.resamples, seed=args.seed)
        doc["hosts"][host] = rep
        lines += [f"## {host} ({rep['items']} items)", "", "**Primary (Holm over 4)**", ""]
        lines += [f"- ({c['family']}) {c['a']} − {c['b']}: {fmt(c)}" for c in rep["primary"]]
        conds = ["tree-text", "path-vsa"] + [f"{c}@{p}" for p in NOISE_LEVELS for c in ("none", "tree-text", "path-vsa", "path-vsa-soft", "path-sym-soft")]
        depths = sorted({k for a in rep["accuracy"].values() for k in a if k.startswith("depth")}, key=lambda k: int(k.split()[1]))
        lines += ["", "| condition | all | " + " | ".join(depths) + " | executor ceiling | follows executor | … when it is wrong |",
                  "|---|---:|" + "---:|" * (len(depths) + 3)]
        for c in conds:
            if c not in rep["accuracy"]:
                continue
            a, fo = rep["accuracy"][c], rep["follow"].get(c)
            ce = rep["executor_ceilings"].get(c)
            lines.append(f"| {c} | {a['all']:.3f} | " + " | ".join(f"{a.get(d, float('nan')):.3f}" for d in depths)
                         + f" | {'—' if ce is None else f'{ce:.3f}'} | {'—' if not fo else f"{fo['follow']:.3f}"} | "
                         + f"{'—' if not fo or fo['follow_when_executor_wrong'] is None else f"{fo['follow_when_executor_wrong']:.3f} (n {fo['n_executor_wrong']})"} |")
        lines += ["", "Secondary:", ""]
        lines += [f"- p {c['noise']}, {c['subset']}: {c['a']} − {c['b']} {fmt(c)}" for c in rep["secondary"] if not c["subset"].startswith("depth")]
        lines += ["", "By depth:", ""]
        lines += [f"- p {c['noise']}, {c['subset']}: {c['a']} − {c['b']} {fmt(c)}" for c in rep["secondary"] if c["subset"].startswith("depth")]
        lines.append("")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--host", required=True); s.add_argument("--items", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True); s.add_argument("--data", type=Path, required=True)
    s.add_argument("--sets", default=None); s.add_argument("--seed", type=int, default=1); s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default=None); s.add_argument("--batch-size", type=int, default=16); s.add_argument("--smoke", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--runs", nargs="+", required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--resamples", type=int, default=4000); r.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verdict-report", help="LG-B verdict format (pre-registration §14)")
    v.add_argument("--runs", nargs="+", required=True); v.add_argument("--output", type=Path, required=True)
    v.add_argument("--resamples", type=int, default=4000); v.add_argument("--seed", type=int, default=0)
    nr = sub.add_parser("noisy-report", help="LG-D2 trees on noisy or partial records (pre-registration §19)")
    nr.add_argument("--runs", nargs="+", required=True); nr.add_argument("--output", type=Path, required=True)
    nr.add_argument("--resamples", type=int, default=4000); nr.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"score": run_score, "report": run_report, "verdict-report": run_verdict_report, "noisy-report": run_noisy_report}[args.command](args)


if __name__ == "__main__":
    main()
