"""Exploratory diagnostics of E14 tier P (2026-10-20; not pre-registered, read after the primaries).

(a) LG-B true/false items: accuracy by gold label (are the SmolLM2 hosts' near-chance scores a negation failure?).
(b) RD-4 T4/T5 property items: PMI-calibrated against raw accuracy by relation and condition (does the null prompt,
    which also reads the recall context, cancel the recall's evidence?), and whether the oracle frame states the gold.

Run from the repo root: python experiments/e14-logic/report/diagnostics-2026-10-20.py
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / "data/vsa-llm/e14"
RUNS = Path("experiments/e14-logic/runs")
OUT = Path("experiments/e14-logic/report/diagnostics-2026-10-20.md")
LG_HOSTS = ["SmolLM2-135M", "SmolLM2-360M", "Qwen3-0.6B", "Qwen3-1.7B"]
LG_CONDS = ["text", "exec-sym", "exec-vsa", "exec-vsa-conclusion"]
RD4_CONDS = ["none", "implicit:oracle", "explicit-sym:oracle", "explicit-sym:typeprior", "explicit-sym:linker",
             "noisy-explicit:0.3", "definition"]


def lg_by_label() -> list[str]:
    lines = ["## (a) LG-B true/false accuracy by gold label (exec conditions state the conclusion)", "",
             "| host | set | condition | label True | label False |", "|---|---|---|---:|---:|"]
    for host in LG_HOSTS:
        acc = defaultdict(lambda: [0, 0])
        with gzip.open(DATA / f"runs/lg/{host}-bc/predictions.jsonl.gz", "rt") as handle:
            for line in handle:
                r = json.loads(line)
                if r["format"] == "tf" and r["set"] in ("taxonomy", "prop-rp") and r["condition"] in LG_CONDS:
                    k = (r["set"], r["condition"], bool(r["label"]))
                    acc[k][0] += r["correct"]; acc[k][1] += 1
        for s in ("taxonomy", "prop-rp"):
            for c in LG_CONDS:
                t, f = acc.get((s, c, True)), acc.get((s, c, False))
                if t and f:
                    lines.append(f"| {host} | {s} | {c} | {t[0] / t[1]:.3f} ({t[1]}) | {f[0] / f[1]:.3f} ({f[1]}) |")
    return lines


def rd4_pmi_raw() -> list[str]:
    lines = ["", "## (b) RD-4 property items, C5 s1–3 pooled: PMI-correct / raw-correct", ""]
    for track in ("t4-heldout", "t5-new"):
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for s in (1, 2, 3):
            for line in open(RUNS / f"rd4/{track}/SmolLM2-360M-C5-s{s}/predictions.jsonl"):
                r = json.loads(line)
                if r["test"] != "property":
                    continue
                for k in ((r["condition"], r["relation"] if track == "t4-heldout" else "all"), (r["condition"], "ALL")):
                    acc[k][0] += r["correct"]; acc[k][1] += r.get("raw_correct") or 0; acc[k][2] += 1
        rels = sorted({k[1] for k in acc})
        lines += [f"**{track}**", "", "| condition | " + " | ".join(rels) + " |", "|---|" + "---:|" * len(rels)]
        for c in RD4_CONDS:
            cells = [f"{acc[(c, r)][0] / acc[(c, r)][2]:.3f} / {acc[(c, r)][1] / acc[(c, r)][2]:.3f}" if acc[(c, r)][2] else "—"
                     for r in rels]
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def rd4_gold_in_frame() -> list[str]:
    recs = json.load(gzip.open(DATA / "chebi-255-records.json.gz", "rt"))
    name = lambda cid: ((recs.get(cid.replace("chebi:", "")) or {}).get("name") or "").strip().lower()
    frames = {}
    for line in open("experiments/e9-retrofit/runs/t4/SmolLM2-360M-full-C5-s1/e11-t4-heldout/frames.jsonl"):
        r = json.loads(line)
        if r["reader"] == "oracle":
            frames[r["concept"]] = r["frame"]
    strip = lambda s: s.strip().lower().removeprefix("an ").removeprefix("a ")
    counts = defaultdict(lambda: [0, 0, 0])
    for line in open("experiments/e11-read-to-learn/items/t4-heldout-smollm2-v1/items.jsonl"):
        it = json.loads(line)
        if it["test"] != "property" or it["concept"] not in frames:
            continue
        partners = {name(p) for _, p in frames[it["concept"]] if p.startswith("chebi:")}
        cands = [strip(c) for c in it["candidates"]]
        c = counts[it["relation"]]
        c[0] += 1; c[1] += cands[it["gold"]] in partners
        c[2] += any(x in partners for i, x in enumerate(cands) if i != it["gold"])
    lines = ["## (c) T4 property items: does the oracle recall state the answer?", "",
             "| relation | items | gold among the frame's partners | a distractor among them |", "|---|---:|---:|---:|"]
    lines += [f"| {r} | {n} | {g} | {d} |" for r, (n, g, d) in sorted(counts.items())]
    return lines


if __name__ == "__main__":
    text = ["# E14 tier P — exploratory diagnostics (2026-10-20)", "", __doc__.strip().split("\n\n")[0], ""]
    text += lg_by_label() + rd4_pmi_raw() + rd4_gold_in_frame()
    OUT.write_text("\n".join(text) + "\n")
    print(OUT.read_text())
