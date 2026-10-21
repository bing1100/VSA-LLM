"""E14 RD-14 — completion under selective recall: do inferred facts help or hurt when recalled? (pre-registration addendum 11,
`experiments/e14-logic/preregistration-addendum-11-cheap.md`).

The frames of RD-4b's T4 held-out items are recalled selectively (`explicit-rel`, the item's own relation: the oracle router,
so routing is held fixed) in four forms: (i) gold; (ii) gold without the asked relation, which under selective recall is an
empty recall; (iii) the asked relation's facts **inferred** from the frame's other edges by a link predictor (`infer_frames`:
a neighbour vote over the ontology's known entries, idf-weighted; held-out entries excluded), written unmarked (as gold lines
read) or marked "(inferred)"; (iv) none. Secondary: the known edges themselves shown (`explicit-known:oracle`), with the
inferred lines added (`explicit-known-infer[-marked]`), and the definition. Scoring, items, hosts and the pooled report
are RD-4b's (`e14_reading`: context-free null, `rd4b_group`); this module only builds the readers and the new recall kinds.

    python -m vsa_embed.experiments.e14_inferred recall --run RUN --items DIR --output OUT [--frames F.jsonl] [--limit N] [--smoke]
    python -m vsa_embed.experiments.e14_inferred report --runs OUT ... --output DIR
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..statistics import holm_adjust
from . import e11_read_to_learn as e11
from . import e14_reading as rd
from .e5_common import clear_output, finish_output, json_ready, open_run, start_output, write_json

CONDITIONS = ("none", "explicit-rel:oracle", "explicit-rel:minus", "explicit-rel:infer-nn", "explicit-rel-marked:infer-nn",
              "explicit-known:oracle", "explicit-known-infer:infer-nn", "explicit-known-infer-marked:infer-nn", "definition")
KINDS = ("explicit-rel-marked", "explicit-known", "explicit-known-infer", "explicit-known-infer-marked")
PRIMARY = (("explicit-rel:infer-nn", "explicit-rel:minus"), ("explicit-rel-marked:infer-nn", "explicit-rel:infer-nn"),
           ("explicit-rel:infer-nn", "explicit-rel:oracle"))
SECONDARY = (("explicit-rel:minus", "none"), ("explicit-rel:oracle", "none"), ("explicit-rel:infer-nn", "none"),
             ("explicit-rel-marked:infer-nn", "explicit-rel:minus"), ("explicit-known:oracle", "none"),
             ("explicit-known:oracle", "explicit-rel:minus"), ("explicit-known-infer:infer-nn", "explicit-known:oracle"),
             ("explicit-known-infer-marked:infer-nn", "explicit-known-infer:infer-nn"), ("definition", "explicit-rel:infer-nn"))
SPLITS = {"inferred right": lambda row: bool((row.get("gold_in") or {}).get("infer-nn")),
          "inferred wrong": lambda row: "infer-nn" in (row.get("gold_in") or {}) and not row["gold_in"]["infer-nn"]}
MARK = "(inferred)"


# -- the link predictor ------------------------------------------------------------------------------------------------

def infer_frames(gold: dict[str, list | None], ontology: dict[str, Any], *, neighbours: int = 20, max_lines: int = 3
                 ) -> tuple[dict[str, list | None], dict[str, Any]]:
    """Link prediction from the known edges. For each relation r of a concept's gold frame, r's fillers are predicted from
    the frame's other edges: the ontology entries (held-out and synthetic entries excluded) sharing those edges are scored by
    the summed idf of the shared edges, the `neighbours` best vote for their own fillers of r, weighted by their score;
    without any neighbour, r's most frequent fillers. m_r lines are kept, m_r = the mean number of r-edges per frame holding
    r (rounded, 1 … `max_lines`): a per-relation constant, so the gold count is not leaked. Returns the inferred frames
    (concept → (r, a) over every relation of its gold frame) and per-relation precision and recall against the gold."""
    o = ontology
    offsets = np.asarray(o["offsets"], dtype=np.int64)
    relations, fillers = np.asarray(o["relations"], dtype=np.int64), np.asarray(o["fillers"], dtype=np.int64)
    entries, n_atoms = int(o["entry_count"]), len(o["atomic_names"])
    owner = np.repeat(np.arange(entries), np.diff(offsets))
    excluded = np.asarray(sorted({int(e) for e in o.get("heldout_entries", ())} | {int(e) for e in o.get("synthetic_entries", ())}),
                          dtype=np.int64)
    keep = ~np.isin(owner, excluded)
    key = relations * n_atoms + fillers
    order = np.argsort(key[keep], kind="stable")
    keys_sorted, owners_sorted = key[keep][order], owner[keep][order]
    uniq, starts, counts = np.unique(keys_sorted, return_index=True, return_counts=True)
    posting = {int(k): (int(s), int(n)) for k, s, n in zip(uniq, starts, counts)}
    n_frames = max(1, len(np.unique(owner[keep])))
    per_relation: dict[int, list[int]] = defaultdict(list)
    for (_, r), n in Counter(zip(owner[keep].tolist(), relations[keep].tolist())).items():
        per_relation[r].append(n)
    m_of = {r: int(min(max_lines, max(1, round(float(np.mean(ns)))))) for r, ns in per_relation.items()}
    marginal = {r: [a for a, _ in Counter(fillers[keep][relations[keep] == r].tolist()).most_common(max_lines)] for r in per_relation}
    out: dict[str, list | None] = {}
    hits: dict[int, list[float]] = defaultdict(list)
    recall: dict[int, list[float]] = defaultdict(list)
    for concept, frame in gold.items():
        frame = [(int(r), int(a)) for r, a in (frame or [])]
        if not frame:
            out[concept] = None
            continue
        inferred: list[tuple[int, int]] = []
        for r in dict.fromkeys(r for r, _ in frame):
            score = np.zeros(entries)
            for q, a in frame:
                hit = posting.get(q * n_atoms + a) if q != r else None
                if hit:
                    s, n = hit
                    score[owners_sorted[s:s + n]] += math.log(n_frames / n)
            votes: Counter = Counter()
            for e in np.argsort(-score, kind="stable")[:neighbours]:
                if score[e] <= 0:
                    break
                lo, hi = int(offsets[e]), int(offsets[e + 1])
                for a in fillers[lo:hi][relations[lo:hi] == r].tolist():
                    votes[int(a)] += float(score[e])
            ranked = [a for a, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))] or list(marginal.get(r, []))
            chosen = ranked[:m_of.get(r, 1)]
            truth = {a for q, a in frame if q == r}
            hits[r] += [float(a in truth) for a in chosen]
            recall[r].append(float(len(truth & set(chosen)) / len(truth)))
            inferred += [(r, a) for a in chosen]
        out[concept] = inferred
    names = list(o["relation_names"])
    diag = {names[r]: {"precision": float(np.mean(hits[r])) if hits[r] else None, "recall": float(np.mean(recall[r])),
                       "lines": m_of.get(r), "n_edges": len(hits[r])} for r in sorted(recall)}
    diag["all"] = {"precision": float(np.mean([h for hs in hits.values() for h in hs])) if hits else None,
                   "recall": float(np.mean([v for vs in recall.values() for v in vs])) if recall else None}
    return out, diag


# -- recall text of the new kinds ---------------------------------------------------------------------------------------

def mark_lines(text: str, marker: str = MARK) -> str:
    """Recall text with each line's confidence tag replaced by `marker`."""
    return re.sub(r" \(\d\.\d\d\)$", f" {marker}", text, flags=re.M)


def mixed_recall(text: rd.RecallText, subject: str, known: Sequence[tuple[int, int]], asked: Sequence[tuple[int, int]], *,
                 marked: bool) -> str:
    """One recall: the `known` edges' lines, then the `asked` edges' lines (marked "(inferred)" if `marked`)."""
    head = f"recall({subject}):"
    known_lines = text.symbolic(subject, list(known)).split("\n")[1:] if known else []
    asked_text = text.symbolic(subject, list(asked)) if asked else ""
    asked_lines = (mark_lines(asked_text) if marked else asked_text).split("\n")[1:] if asked else []
    body = known_lines + asked_lines
    return "\n".join([head, *body]) if body else f"{head}\n- (nothing recalled)"


def rd14_contexts(name: str, scorer: Any, frames: dict[str, dict[str, list | None]], text: rd.RecallText, ctx: e11.TrackContext
                  ) -> tuple[dict[str, str], dict[str, Any]]:
    """Per-item contexts of a `KINDS` condition `<kind>:<reader>`: `explicit-rel-marked` (the reader's lines of the asked
    relation, marked); `explicit-known` (the gold edges of the other relations); `explicit-known-infer[-marked]` (those,
    then the reader's lines of the asked relation)."""
    kind, _, arg = name.partition(":")
    if kind not in KINDS:
        raise ValueError(f"condition {name!r}: kind must be one of {KINDS}")
    reader = frames.get(arg)
    if reader is None:
        raise KeyError(f"condition {name!r}: no {arg!r} frames")
    rid, gold = ctx.relation_id, frames["oracle"]
    out, lines = {}, []
    for c in scorer.concepts:
        frame, own = list(gold.get(c) or []), list(reader.get(c) or [])
        for item in scorer.items_by_concept[c]:
            rel = rid.get(item.get("relation"))
            known = [e for e in frame if int(e[0]) != rel] if kind.startswith("explicit-known") else []
            asked = [e for e in own if rel is None or int(e[0]) == rel] if kind != "explicit-known" else []
            lines.append(len(known) + len(asked))
            out[item["id"]] = mixed_recall(text, scorer.surface[c], known, asked, marked=kind.endswith("-marked"))
    return out, {"lines_per_item": float(np.mean(lines)) if lines else 0.0, "items_without_lines": int(sum(n == 0 for n in lines))}


# -- the run ------------------------------------------------------------------------------------------------------------

def evaluate(run: Any, read_set: e11.ReadSet, frames: dict[str, dict[str, list | None]], *, style: str, conditions: Sequence[str],
             seed: int = 0, resamples: int = 2000, max_context_tokens: int | None = 1024, log=print) -> dict[str, Any]:
    """`e14_reading.evaluate_recall` with the context-free null (RD-4b) for every condition, the `KINDS` conditions added."""
    ctx = e11.run_context(run)
    budget = rd.context_budget(run, max_context_tokens)
    scorer = e11.ItemScorer(run, read_set, ctx)
    text = rd.RecallText(run.ontology, ctx.lexicon, None)
    definitions = {d["concept"]: d["text"] for d in read_set.definitions if d["style"] == style}
    pools = rd.filler_pools(run.ontology)
    null_scores = rd.plain_null_scores(scorer)
    flags = rd.gold_flags([i for c in scorer.concepts for i in scorer.items_by_concept[c]], frames, ctx)
    item_rows, info, tokens, samples = {}, {}, {}, {}
    for name in conditions:
        log(f"  {name}")
        if name.partition(":")[0] in KINDS:
            contexts, diag = rd14_contexts(name, scorer, frames, text, ctx)
            row_frames: dict[str, list | None] = {c: None for c in scorer.concepts}
        else:
            row_frames, contexts, diag = rd.item_condition_inputs(name, scorer, frames, text, definitions, pools, ctx, seed=seed)
        if contexts:
            contexts = {i: rd.fit_context(t, run.tokenizer, budget) for i, t in contexts.items()}
        item_rows[name] = rd.score_rows(scorer, row_frames, contexts, null_scores=null_scores)
        for rows in item_rows[name].values():
            for row in rows:
                if row["id"] in flags:
                    row["gold_in"] = flags[row["id"]]
        info[name] = diag
        sample = list(contexts.values())[:400] if contexts else []
        tokens[name] = float(np.mean([len(run.tokenizer(t, add_special_tokens=False)["input_ids"]) + 1 for t in sample])) if sample else 0.0
        if sample:
            samples[name] = sample[0]
    summary = rd.summarize(item_rows, read_set.kind, resamples=resamples, seed=seed)
    values = {cond: e11.test_values(rows, read_set.kind) for cond, rows in item_rows.items()}

    def block(pairs: Sequence[tuple[str, str]], holm: bool) -> list[dict[str, Any]]:    # RD-14's pairs (`rd.summarize` has RD-4's)
        rows = [{"a": a, "b": b, "test": t, **r} for a, b in pairs if a in values and b in values for t in rd.PRIMARY_TESTS
                if (r := e11.compare(values[a].get(t, {}), values[b].get(t, {}), resamples=resamples, seed=seed)) is not None]
        for row, p in zip(rows, holm_adjust([r["p_value"] for r in rows]) if holm and rows else []):
            row["p_holm"] = p
        return rows
    summary["primary"], summary["secondary"] = block(PRIMARY, True), block(SECONDARY, False)
    document = {"set": {"kind": read_set.kind, "track": read_set.manifest["track"], "concepts": len(scorer.concepts), "items": len(read_set.items)},
                "null": "plain", "gold_in_frame_share": {r: float(np.mean([f[r] for f in flags.values()])) for r in frames} if flags else {},
                "style": style, "conditions": list(conditions), "compose": False, "vsa": False, "context_budget": budget,
                "readers": {r: sum(f is not None for f in v.values()) for r, v in frames.items()}, "diagnostics": info,
                "context_tokens": tokens, "context_samples": samples, "items": summary}
    return {"document": document, "item_rows": item_rows}


def render(document: dict[str, Any], *, smoke: bool) -> str:
    lines = rd.render(document, smoke=smoke).split("\n")
    lines[2 if smoke else 0] = f"# E14 RD-14 — inferred facts under selective recall ({document['set']['track']} {document['set']['kind']})"
    block = ["", "## The link predictor against the gold frames (no model)", "", "| relation | precision | recall | lines | n |",
             "|---|---:|---:|---:|---:|"]
    cell = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for r, d in document["inference"].items():
        block.append(f"| {r} | {cell(d.get('precision'))} | {cell(d.get('recall'))} | {d.get('lines', '—')} | {d.get('n_edges', '—')} |")
    return "\n".join(lines + block) + "\n"


def run_recall(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items).limit(args.limit)
    track = read_set.manifest["track"]
    style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    config = {"experiment": "e14-rd14-inferred", "run": str(args.run), "items": str(args.items), "style": style, "limit": args.limit,
              "frames": str(args.frames) if args.frames else None, "conditions": args.conditions, "seed": args.seed,
              "resamples": args.resamples, "max_context_tokens": args.max_context_tokens, "neighbours": args.neighbours,
              "alias_table": str(args.alias_table) if args.alias_table else None, "smoke": bool(args.smoke), "null": "plain", "family": "rd14"}
    output = Path(args.output)
    if args.overwrite:
        clear_output(output)
        (output / "predictions.jsonl").unlink(missing_ok=True)
    git_at_start = start_output(output, config)
    run = open_run(args.run, checkpoint=args.checkpoint, device=args.device, batch_size=args.batch_size, alias_table=args.alias_table)
    ctx = e11.run_context(run)
    path = Path(args.frames) if args.frames else rd.frames_path(Path(args.run), read_set)
    frames = rd.load_reader_frames(path, style, ("oracle",), ctx)
    if "oracle" not in frames:
        raise SystemExit(f"{path}: no oracle frames (style {style})")
    frames["infer-nn"], inference = infer_frames(frames["oracle"], run.ontology, neighbours=args.neighbours)
    frames["minus"] = {c: [] for c in frames["oracle"]}
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()] if args.conditions else list(CONDITIONS)
    result = evaluate(run, read_set, frames, style=style, conditions=conditions, seed=args.seed, resamples=args.resamples,
                      max_context_tokens=args.max_context_tokens)
    document = {"source": run.describe(), "frames_from": str(path), "smoke": bool(args.smoke), "family": "rd14", "inference": inference,
                **result["document"]}
    with (output / "predictions.jsonl").open("w") as handle:
        for cond, by_concept in result["item_rows"].items():
            for concept, rows in by_concept.items():
                for row in rows:
                    handle.write(json.dumps(json_ready({"condition": cond, **row})) + "\n")
    write_json(output / "summary.json", document)
    (output / "report.md").write_text(render(document, smoke=bool(args.smoke)))
    finish_output(output, config, git_at_start=git_at_start, device=run.device, source=run.describe())
    return document


# -- the pooled report --------------------------------------------------------------------------------------------------

def run_report(args: argparse.Namespace) -> dict[str, Any]:
    """RD-4b's pooled report (`rd4b_group`) with RD-14's primary (Holm over 3 contrasts × 3 tests) and secondary families, the
    primary contrasts on the items whose inferred line names the gold answer and on the others, and the predictor's scores."""
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for folder in present:
        s = json.loads((folder / "summary.json").read_text())
        groups[(s["set"]["track"], s["set"]["kind"], Path(s["source"]["run"]).name.rsplit("-s", 1)[0])].append(folder)
    out: dict[str, Any] = {"family": "rd14", "groups": [], "missing": missing}
    lines = ["# E14 RD-14 — inferred facts under selective recall with the oracle router (pre-registration addendum 11)", "",
             "PMI choices use the null prompt without the context; `raw` columns are uncalibrated choices.", ""]
    if missing:
        lines += [f"Missing run folders: {', '.join(missing)}", ""]
    for (track, kind, model), folders in sorted(groups.items()):
        conds = set.intersection(*(set(json.loads((f / "summary.json").read_text())["conditions"]) for f in folders))
        g = {"track": track, "kind": kind, "model": model, "seeds": len(folders),
             **rd.rd4b_group(folders, conds, resamples=args.resamples, seed=args.seed, primary=PRIMARY, secondary=SECONDARY, stated_pairs=())}
        for name, where in SPLITS.items():
            g[f"split: {name}"] = [{"a": a, "b": b, "test": t, **r} for a, b in PRIMARY if a in conds and b in conds for t in rd.PRIMARY_TESTS
                                   if (r := rd.pooled_contrast(folders, a, b, t, where=where, resamples=args.resamples, seed=args.seed))]
        g["inference"] = json.loads((folders[0] / "summary.json").read_text()).get("inference", {})
        out["groups"].append(g)
        cols = [c for c in ("property", "paraphrase", "entailment", "property raw", "entailment raw") if any(c in m for m in g["means"].values())]
        lines += [f"## {track} {kind} — {model} ({len(folders)} seeds)", "", "| condition | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for cond in sorted(g["means"]):
            m = g["means"][cond]
            lines.append(f"| {cond} | " + " | ".join(f"{m[c]:.4f}" if c in m else "—" for c in cols) + " |")
        lines += ["", "Link predictor (precision / recall against the gold frames; lines per relation): " + "; ".join(
            f"{r} {d['precision']:.3f} / {d['recall']:.3f}" + (f" ({d['lines']})" if d.get("lines") else "")
            for r, d in g["inference"].items() if d.get("precision") is not None)]
        for title, key in (("Primary (Holm)", "primary"), ("Secondary", "secondary"), ("Primary contrasts on raw choices", "raw_primary"),
                           *((f"Primary contrasts, {n}", f"split: {n}") for n in SPLITS)):
            lines += ["", f"### {title}", "", "| a − b | test | difference [95% CI] | n |", "|---|---|---|---:|"]
            lines += [f"| {r['a']} − {r['b']} | {r['test']} | {rd.fmt_ci(r)} | {r['n']} |" for r in g[key]]
        lines.append("")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", out)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("recall", help="RD-14: inferred facts under selective recall with the oracle router")
    r.add_argument("--run", type=Path, required=True); r.add_argument("--items", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--frames", type=Path, default=None, help="an E11 frames.jsonl (default: the run's own E11 output)")
    r.add_argument("--style", default=None); r.add_argument("--conditions", default=None)
    r.add_argument("--neighbours", type=int, default=20, help="neighbours voting in the link predictor")
    r.add_argument("--alias-table", type=Path, default=None); r.add_argument("--checkpoint", default="final.pt")
    r.add_argument("--device", default=None); r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--limit", type=int, default=None); r.add_argument("--max-context-tokens", type=int, default=1024)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--resamples", type=int, default=4000)
    r.add_argument("--smoke", action="store_true"); r.add_argument("--overwrite", action="store_true")
    rr = sub.add_parser("report", help="pool RD-14 run folders over seeds")
    rr.add_argument("--runs", nargs="+", required=True); rr.add_argument("--output", type=Path, required=True)
    rr.add_argument("--resamples", type=int, default=4000); rr.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"recall": run_recall, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
