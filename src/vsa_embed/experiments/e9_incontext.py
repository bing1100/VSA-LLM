"""E9 in-context frame baseline for the loss (2026-10-11 addendum, `experiments/e9-retrofit/preregistration-addendum-
2026-10-11.md` §3; novelty check §7 P0 "in-context verbalized frame / definition", claims A-B3). Evaluation only, on
finished runs.

    python -m vsa_embed.experiments.e9_incontext score --run RUN [--output DIR] [--variants none frame other]
        [--max-prefix 1024] [--seed 0] [--limit-windows N] [--eval-batch N] [--resume | --overwrite]
    python -m vsa_embed.experiments.e9_incontext report --inputs DIR [DIR ...] --output DIR [--resamples 10000] [--overwrite]

**What it measures.** Each evaluation window of a run (the trainer's fixed windows, `eval_windows`) is scored with a text
prefix in front of it, and only the window's own targets j ∈ [1, L) count, stratified exactly as the trainer does
(`training.lm.stratum_masks` on the window alone), so every number is paired with the run's final evaluation. Variants:

- `none` — the window alone (replays the run's final evaluation; `own_check` against `eval_windows.npz`);
- `frame` — a glossary of the verbalized frames of the entries linked in the window, one line `<name>: <frame text>` per
  entry in order of first occurrence (the verbalization table of `e9_rowsource`: the same frame text the definition
  encoder C6d reads), under a `Glossary:` header and a blank line;
- `other` — the same number of glossary lines, each the name and frame of an entry linked somewhere in the evaluation
  windows but not in this one (a seeded draw per window): the prefix-length and domain-priming control.

A glossary is capped at `--max-prefix` tokens; lines that do not fit are left out, and the coverage is recorded (entries
and held-out entries whose line is included). Channel models (C2, C5, arms) inject at the window's spans, shifted by the
prefix length, and never in the prefix; the P1 context pooling of a span within its first 8 tokens may then read prefix
tokens (recorded, not corrected). The licensed clinical track (T1c) is refused: its frames are SNOMED CT-derived.

**Outputs** (`experiments/e9-retrofit/incontext/<stage>/<stem>/`): `windows.npz` (`strata`, `starts`, `count`, and per
variant `sum_<v>` strata × windows, `prefix_<v>` prefix tokens, `entries_<v>` / `included_<v>` / `heldout_<v>` /
`heldout_included_<v>` per window), `summary.json`, `resolved_config.yaml`, `manifest.json`.

**Report.** Per track, relative loss differences pooled over common seeds per window with 95% cluster bootstraps over
windows (`statistics.paired_ratio_bootstrap`; Holm over the comparisons within a stratum): in-context benefit for the
continued-trained host (C0′ frame − C0′) and the original host (P0 frame − P0), the channel (C5 − C0′), the channel
given the glossary (C5 frame − C0′ frame), the glossary given the channel (C5 frame − C5), both against neither
(C5 frame − C0′), and the glossary's term specificity (frame − other, C0′ and C5); with the glossary's token cost.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..data.corpus import SPAN_FIELDS, TokenCorpus, collate_windows, eval_windows
from ..provenance import git_state, prepare_output_dir, write_run_metadata
from ..statistics import holm_adjust, paired_ratio_bootstrap
from ..training.lm import load_final, load_window_losses, resolve_config, stratum_masks
from .e9_rescore import _sink_arrays, _write_record

ROOT = Path("experiments/e9-retrofit")
VARIANTS = ("none", "frame", "other")
POSITION_FIELDS = ("start", "end", "inject")
HEADER, LINE_END, TAIL = "Glossary:\n", "\n", "\n"
REPORT_STRATA = ("after_heldout", "after_unseen", "after_rare_seen", "after_len3plus", "inside", "unlinked", "all")
# (label, (model, variant), (model, variant)): candidate − reference
COMPARISONS = (
    ("in-context, C0′ (C0′ frame − C0′)", ("C0'", "frame"), ("C0'", "none")),
    ("in-context, P0 (P0 frame − P0)", ("P0", "frame"), ("P0", "none")),
    ("channel (C5 − C0′)", ("C5", "none"), ("C0'", "none")),
    ("channel given the glossary (C5 frame − C0′ frame)", ("C5", "frame"), ("C0'", "frame")),
    ("glossary given the channel (C5 frame − C5)", ("C5", "frame"), ("C5", "none")),
    ("both vs neither (C5 frame − C0′)", ("C5", "frame"), ("C0'", "none")),
    ("glossary specificity, C0′ (frame − other)", ("C0'", "frame"), ("C0'", "other")),
    ("glossary specificity, C5 (frame − other)", ("C5", "frame"), ("C5", "other")),
)
LICENSED = ("t1c",)
ALPHA = 0.05
STEM = re.compile(r"^(?P<host>.+)-(?P<mode>[^-]+)-(?P<model>[^-]+)-s(?P<seed>\d+)$")
KIND, REPORT_KIND = "e9-incontext", "e9-incontext-report"


# ---------------------------------------------------------------- glossary


def load_glossary(track: str, family: str = "smollm2", path: Path | None = None) -> dict[int, tuple[str, str]]:
    """Entry → (display name, verbalized frame) from `e9_rowsource`'s verbalization table (written if missing)."""
    from . import e9_rowsource
    path = Path(path) if path else e9_rowsource.ROOT / f"{track}-{family}" / "verbalized.jsonl.gz"
    if not path.exists():
        e9_rowsource.write_track_texts(track, family, path)
    out: dict[int, tuple[str, str]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            out[int(row["entry"])] = (str(row.get("name") or ""), str(row["text"]))
    return out


def glossary_line(entry: int, glossary: dict[int, tuple[str, str]]) -> str:
    name, text = glossary.get(int(entry), ("", ""))
    return f"{name}: {text}" if name else text


def build_prefix(lines: Sequence[str], tokenizer: Any, max_tokens: int) -> tuple[list[int], int]:
    """Token ids of `HEADER + lines… + TAIL`, adding whole lines in order while the total stays ≤ `max_tokens`; returns
    (ids, lines included). No line fits → an empty prefix (the window is scored alone)."""
    if not lines or max_tokens <= 0:
        return [], 0
    head = tokenizer.encode(HEADER, add_special_tokens=False)
    tail = tokenizer.encode(TAIL, add_special_tokens=False)
    ids, used = list(head), 0
    for line in lines:
        piece = tokenizer.encode(line + LINE_END, add_special_tokens=False)
        if len(ids) + len(piece) + len(tail) > max_tokens:
            break
        ids.extend(piece)
        used += 1
    return (ids + tail, used) if used else ([], 0)


def ordered_entries(spans: dict[str, np.ndarray]) -> list[int]:
    """The window's linked entries in order of first occurrence."""
    order = np.argsort(np.asarray(spans["start"]), kind="stable")
    return list(dict.fromkeys(int(e) for e in np.asarray(spans["entry"])[order]))


def plan_prefixes(window_spans: Sequence[dict[str, np.ndarray]], glossary: dict[int, tuple[str, str]], tokenizer: Any, variant: str,
                  *, max_tokens: int, heldout: set[int], seed: int) -> tuple[list[list[int]], dict[str, np.ndarray]]:
    """Per window the prefix ids of `variant` and its coverage record (entries, included, held-out, held-out included)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    entries = [ordered_entries(s) for s in window_spans]
    pool = np.asarray(sorted({e for es in entries for e in es if e in glossary}), dtype=np.int64)
    prefixes: list[list[int]] = []
    record = {k: np.zeros(len(entries), dtype=np.int32) for k in ("prefix", "entries", "included", "heldout", "heldout_included")}
    for w, own in enumerate(entries):
        record["entries"][w] = len(own)
        record["heldout"][w] = sum(e in heldout for e in own)
        if variant == "none" or not own:
            prefixes.append([])
            continue
        if variant == "frame":
            chosen = own
        else:                                            # other: the same number of lines from entries not in this window
            candidates = pool[~np.isin(pool, own)]
            rng = np.random.default_rng([seed, w])
            chosen = rng.choice(candidates, size=min(len(own), len(candidates)), replace=False).tolist() if len(candidates) else []
        ids, used = build_prefix([glossary_line(e, glossary) for e in chosen], tokenizer, max_tokens)
        prefixes.append(ids)
        record["prefix"][w] = len(ids)
        record["included"][w] = used
        if variant == "frame":
            record["heldout_included"][w] = sum(e in heldout for e in own[:used])
    return prefixes, record


# ---------------------------------------------------------------- scoring


def shift_spans(spans: dict[str, torch.Tensor], offsets: torch.Tensor) -> dict[str, torch.Tensor]:
    """The span table with every position field moved right by its row's prefix length."""
    shift = offsets[spans["batch"]] if spans["batch"].numel() else spans["batch"]
    return {k: (v + shift if k in POSITION_FIELDS else v) for k, v in spans.items()}


def score_windows(model: Any, corpus: TokenCorpus, starts: Sequence[int], prefixes: Sequence[Sequence[int]], config: dict[str, Any],
                  frequency: np.ndarray | None, heldout: set[int], device: torch.device, *, batch: int,
                  pad_id: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(strata, sums strata × windows, counts) of the windows' own targets with each window's prefix in front."""
    model.eval()
    length, min_subtokens = int(config["model"]["seq_len"]), int(config["data"]["min_subtokens"])
    sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            windows = [corpus.window(int(s), length, min_subtokens=min_subtokens) for s in starts[i:i + batch]]
            ids, spans = collate_windows(windows)
            pre = [list(p) for p in prefixes[i:i + batch]]
            offsets = torch.as_tensor([len(p) for p in pre], dtype=torch.long)
            total = int(offsets.max()) + length
            inputs = torch.full((len(windows), total), int(pad_id), dtype=torch.long)
            for b, p in enumerate(pre):
                if p:
                    inputs[b, :len(p)] = torch.as_tensor(p, dtype=torch.long)
                inputs[b, len(p):len(p) + length] = ids[b]
            inputs_d = inputs.to(device)
            shifted = shift_spans(spans, offsets) if model.channel else None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                per_token = model(inputs_d, spans={k: v.to(device) for k, v in shifted.items()} if shifted else None,
                                  labels=inputs_d, reduction="none")["loss"].float().cpu()
            # window target j ∈ [1, L) sits at input position P + j, i.e. per-token index P + j − 1
            index = offsets[:, None] + torch.arange(length - 1)[None, :]
            window_loss = per_token.gather(1, index)
            for name, mask in stratum_masks(ids, spans, frequency, heldout).items():
                sums, counts = sink.setdefault(name, ([], []))
                sums.append(window_loss.double().masked_fill(~mask, 0.0).sum(1).numpy())
                counts.append(mask.sum(1).numpy().astype(np.int32))
    return _sink_arrays(sink)


def own_check(run_dir: Path, strata: list[str], starts: Sequence[int], sums: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """`none` against the run's final evaluation (`eval_windows.npz`), paired by window start."""
    path = Path(run_dir) / "eval_windows.npz"
    if not path.exists():
        return {"available": False}
    saved = load_window_losses(path)
    if not saved["evals"]:
        return {"available": False}
    column = {int(s): i for i, s in enumerate(saved["starts"].tolist())}
    if any(int(s) not in column for s in starts):
        return {"available": True, "paired": False}
    run_sums, run_counts = saved["evals"][max(saved["evals"])]
    common = [s for s in strata if s in saved["strata"]]
    rows, mine, cols = [saved["strata"].index(s) for s in common], [strata.index(s) for s in common], [column[int(s)] for s in starts]
    return {"available": True, "paired": True, "windows": len(cols),
            "counts_equal": bool(np.array_equal(run_counts[rows][:, cols], counts[mine])),
            "max_abs_window_sum_diff": float(np.abs(run_sums[rows][:, cols] - sums[mine]).max()) if common else None}


def load_incontext(folder: Path) -> dict[str, Any] | None:
    folder = Path(folder)
    if not (folder / "windows.npz").exists() or not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("kind") != KIND:
        return None
    with np.load(folder / "windows.npz") as data:
        return {"summary": summary, "folder": str(folder), "strata": data["strata"].tolist(), "starts": data["starts"],
                "count": data["count"], "sums": {k[4:]: data[k] for k in data.files if k.startswith("sum_")},
                "records": {k: data[k] for k in data.files if not k.startswith(("sum_", "strata", "starts", "count"))}}


def _write(path: Path, strata: list[str], starts: Sequence[int], counts: np.ndarray, sums: dict[str, np.ndarray],
           records: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, strata=np.asarray(strata), starts=np.asarray(starts, dtype=np.int64), count=counts,
                            **{f"sum_{k}": v for k, v in sums.items()}, **records)
    temporary.replace(path)


def default_output(run_dir: Path, root: Path = ROOT) -> Path:
    return Path(root) / "incontext" / Path(run_dir).parent.name / Path(run_dir).name


def score_run(run_dir: Path, output: Path | None = None, *, variants: Sequence[str] = VARIANTS, max_prefix: int = 1024, seed: int = 0,
              limit_windows: int | None = None, device: str = "cuda", eval_batch: int | None = None, glossary: Path | None = None,
              tokenizer: Any = None, resume: bool = False, overwrite: bool = False, root: Path = ROOT, log: Any = print) -> dict[str, Any]:
    run_dir = Path(run_dir)
    variants = ["none", *[v for v in dict.fromkeys(variants) if v != "none"]]
    if set(variants) - set(VARIANTS):
        raise ValueError(f"variants must be among {VARIANTS}")
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu", mmap=True)
    config = resolve_config(final["config"])
    del final
    track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
    if track in LICENSED:
        raise ValueError(f"{track} is a licensed track: no in-context glossary of its frames")
    output = Path(output) if output else default_output(run_dir, root)
    if overwrite and output.exists():
        for name in ("windows.npz", "summary.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    if not resume and (output / "windows.npz").exists():
        raise FileExistsError(f"{output} holds an in-context scoring; pass --resume or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    git_at_start = git_state()
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    heldout = set(int(e) for e in ontology["heldout_entries"])
    frequency = np.asarray(ontology["train_frequency"]) if ontology.get("train_frequency") is not None else None
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    length, min_subtokens = int(config["model"]["seq_len"]), int(config["data"]["min_subtokens"])
    starts = eval_windows(corpus, count=config["eval"]["windows"], length=length)
    if limit_windows:
        starts = starts[:int(limit_windows)]
    existing = load_incontext(output) if resume else None
    if existing is not None and (not np.array_equal(existing["starts"], np.asarray(starts))
                                 or existing["summary"]["seed"] != seed or existing["summary"]["max_prefix"] != max_prefix):
        raise ValueError(f"{output} was scored with other windows, seed or prefix cap; use --overwrite")
    if tokenizer is None:
        from transformers import AutoTokenizer
        from .e9_tracks import FAMILY_TOKENIZERS
        tokenizer = AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True)
    entries_text = load_glossary(track, family, glossary)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_final(run_dir / "final.pt", target)
    host_config = getattr(getattr(model, "model", None), "config", None)
    positions = getattr(host_config, "max_position_embeddings", None) or getattr(host_config, "n_positions", None)
    if positions is not None and max_prefix + length > int(positions):
        raise ValueError(f"window {length} + prefix cap {max_prefix} exceeds the host's {positions} positions")
    batch = int(eval_batch) if eval_batch else max(1, int(config["eval"]["batch"]) // (1 + (max_prefix + length - 1) // length))
    stage, stem = run_dir.parent.name, run_dir.name
    match = STEM.match(stem)
    summary: dict[str, Any] = {
        "kind": KIND, "smoke": bool(limit_windows), "run": str(run_dir), "id": f"{stage}__{stem}", "stage": stage, "stem": stem,
        "host": match["host"] if match else config["model"].get("pretrained"), "host_mode": match["mode"] if match else None,
        "model": ("C0'" if match and match["model"] == "C0p" else match["model"]) if match else None, "run_seed": int(config["seed"]),
        "track": track, "family": family, "channel": config["channel"]["mode"], "variants": variants, "seed": int(seed),
        "max_prefix": int(max_prefix), "windows": len(starts), "batch": batch, "device": str(target),
        "git_sha": git_at_start.get("git_sha"), "git_dirty": git_at_start.get("git_dirty"),
        "seconds": dict(existing["summary"].get("seconds", {})) if existing else {}, "prefix": {}}
    sums: dict[str, np.ndarray] = dict(existing["sums"]) if existing else {}
    records: dict[str, np.ndarray] = dict(existing["records"]) if existing else {}
    strata, counts = (existing["strata"], existing["count"]) if existing else (None, None)
    window_spans = [corpus.window(int(s), length, min_subtokens=min_subtokens)[1] for s in starts]
    pad = int(getattr(tokenizer, "eos_token_id", None) or 0)
    for variant in variants:
        if variant in sums:
            continue
        prefixes, record = plan_prefixes(window_spans, entries_text, tokenizer, variant, max_tokens=max_prefix, heldout=heldout, seed=seed)
        began = time.monotonic()
        names, values, these_counts = score_windows(model, corpus, starts, prefixes, config, frequency, heldout, target, batch=batch,
                                                    pad_id=pad)
        if strata is None:
            strata, counts = names, these_counts
        elif names != strata or not np.array_equal(these_counts, counts):
            raise RuntimeError(f"{run_dir}: variant {variant} produced other strata or target counts")
        sums[variant] = values
        records.update({f"{k}_{variant}": v for k, v in record.items()})
        summary["seconds"][variant] = round(time.monotonic() - began, 2)
        _write(output / "windows.npz", strata, starts, counts, sums, records)
        _write_record(output / "summary.json", summary)
        log(json.dumps({"run": summary["id"], "variant": variant, "seconds": summary["seconds"][variant],
                        "prefix_tokens_mean": float(record["prefix"].mean())}))
    del model
    if target.type == "cuda":
        torch.cuda.empty_cache()
    for variant in variants:
        entries, included = records[f"entries_{variant}"], records[f"included_{variant}"]
        held, held_in = records[f"heldout_{variant}"], records[f"heldout_included_{variant}"]
        summary["prefix"][variant] = {"tokens_mean": float(records[f"prefix_{variant}"].mean()),
                                      "tokens_max": int(records[f"prefix_{variant}"].max()),
                                      "entry_coverage": float(included.sum() / max(1, entries.sum())) if variant != "none" else None,
                                      "heldout_coverage": (float(held_in.sum() / max(1, held.sum())) if variant == "frame" else None)}
    summary["losses"] = {v: {s: (float(sums[v][i].sum() / counts[i].sum()) if counts[i].sum() else None)
                             for i, s in enumerate(strata) if s in REPORT_STRATA} for v in variants}
    summary["own_check"] = own_check(run_dir, strata, starts, sums["none"], counts)
    _write_record(output / "summary.json", summary)
    settings = {"run": str(run_dir), "variants": variants, "max_prefix": int(max_prefix), "seed": int(seed), "limit_windows": limit_windows,
                "eval_batch": eval_batch, "device": device, "glossary": str(glossary) if glossary else "e9_rowsource verbalization table"}
    write_run_metadata(output, settings, git_at_start=git_at_start, device=target, kind=KIND, keys=sorted(sums))
    return summary


# ---------------------------------------------------------------- report


def discover(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for root in inputs:
        for path in sorted(Path(root).rglob("summary.json")):
            out = load_incontext(path.parent)
            if out is not None:
                found[out["summary"]["id"]] = out
    return list(found.values())


def compare(outputs: Sequence[dict[str, Any]], candidate: tuple[str, str], reference: tuple[str, str], stratum: str, *,
            resamples: int, seed: int) -> dict[str, Any] | None:
    """`candidate − reference` on `stratum`, pooled over the common seeds per window (a P0 run pairs with every seed of the
    other model when one side is P0); 95% cluster bootstrap over windows."""
    by = {}
    for o in outputs:
        by.setdefault(o["summary"]["model"], {})[int(o["summary"]["run_seed"])] = o
    a, b = by.get(candidate[0], {}), by.get(reference[0], {})
    if not a or not b:
        return None
    pairs = [(a[s], b[s]) for s in sorted(set(a) & set(b))]
    if not pairs and "P0" in (candidate[0], reference[0]):          # P0 is seed-independent (seed 1): pair it with every seed
        pairs = [(a[s], b[min(b)]) for s in sorted(a)] if reference[0] == "P0" else [(a[min(a)], b[s]) for s in sorted(b)]
    pairs = [(x, y) for x, y in pairs if candidate[1] in x["sums"] and reference[1] in y["sums"] and stratum in x["strata"]
             and np.array_equal(x["starts"], y["starts"])]
    if not pairs:
        return None
    d = n = base = 0
    for x, y in pairs:
        i, j = x["strata"].index(stratum), y["strata"].index(stratum)
        d = d + (x["sums"][candidate[1]][i] - y["sums"][reference[1]][j])
        n = n + x["count"][i].astype(np.float64)
        base = base + y["sums"][reference[1]][j]
    if float(np.sum(n)) <= 0:
        return None
    boot = paired_ratio_bootstrap(d, n, base, resamples=resamples, seed=seed)
    return {"seeds": [int(x["summary"]["run_seed"]) for x, _ in pairs], "delta": boot["mean"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "p_value": boot["p_value"], "relative": boot["relative"],
            "relative_ci_low": boot["relative_ci_low"], "relative_ci_high": boot["relative_ci_high"], "tokens": int(np.sum(n)),
            "smoke": any(x["summary"].get("smoke") or y["summary"].get("smoke") for x, y in pairs)}


def analyze(outputs: Sequence[dict[str, Any]], *, resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"groups": {}}
    groups: dict[str, list[dict[str, Any]]] = {}
    for o in outputs:
        s = o["summary"]
        groups.setdefault(f"{s['stage']} · {s['host']}", []).append(o)
    for label, members in sorted(groups.items()):
        block: dict[str, Any] = {"models": sorted({f"{o['summary']['model']} s{o['summary']['run_seed']}" for o in members}),
                                 "strata": {}, "prefix": {}, "own_checks": {o["summary"]["id"]: o["summary"].get("own_check") for o in members}}
        for stratum in REPORT_STRATA:
            rows = {name: c for name, cand, ref in COMPARISONS
                    if (c := compare(members, cand, ref, stratum, resamples=resamples, seed=seed)) is not None}
            tested = [c for c in rows.values() if c.get("p_value") is not None]
            for comparison, adjusted in zip(tested, holm_adjust([c["p_value"] for c in tested])):
                comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
            if rows:
                block["strata"][stratum] = rows
        for o in members:
            block["prefix"].setdefault(o["summary"]["model"], o["summary"].get("prefix"))
        result["groups"][label] = block
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:+.2f}%"


def render(summary: dict[str, Any], *, title: str) -> str:
    lines = [f"# {title}", "",
             "In-context frame baseline (`e9_incontext`; addendum 2026-10-11 §3): each evaluation window scored with a glossary of "
             "the verbalized frames of its linked entries in front (`frame`), the same number of other entries' lines (`other`) or "
             "nothing (`none`); only the window's own targets count, with the trainer's strata. Rows are `candidate − reference`, "
             "relative to the reference loss, pooled over common seeds per window, 95% cluster bootstrap over windows, Holm over "
             "the comparisons within a stratum (* = Holm p < 0.05). Negative = the candidate predicts better.", ""]
    for label, group in summary["groups"].items():
        lines += [f"## {label}", "", "Models: " + ", ".join(group["models"]), ""]
        prefix = group["prefix"].get("C0'") or next(iter(group["prefix"].values()), None) or {}
        if prefix.get("frame"):
            f, o = prefix["frame"], prefix.get("other") or {}
            lines += [f"Glossary cost: `frame` {f['tokens_mean']:.0f} tokens per window on average (max {f['tokens_max']}), entry "
                      f"coverage {100 * (f['entry_coverage'] or 0):.1f}%, held-out entry coverage {100 * (f['heldout_coverage'] or 0):.1f}%; "
                      f"`other` {o.get('tokens_mean', float('nan')):.0f} tokens.", ""]
        for stratum, rows in group["strata"].items():
            lines += [f"### `{stratum}`", "", "| comparison | relative [95% CI] | Δ nats/token | Holm p | tokens |", "|---|---|---|---|---:|"]
            for name, c in rows.items():
                star = "*" if c.get("significant") else ""
                lines.append(f"| {name} | {_pct(c['relative'])} [{_pct(c['relative_ci_low'])}, {_pct(c['relative_ci_high'])}]{star} | "
                             f"{c['delta']:+.4f} | {c.get('holm_p', float('nan')):.3g} | {c['tokens']:,} |")
            lines.append("")
        checks = [c for c in group["own_checks"].values() if c and c.get("paired")]
        if checks:
            lines += ["`none` replays the final evaluations: max |window sum − eval_windows.npz| "
                      f"{max(c['max_abs_window_sum_diff'] or 0 for c in checks):.3g}; counts equal in "
                      f"{sum(bool(c['counts_equal']) for c in checks)} of {len(checks)} runs.", ""]
    return "\n".join(lines)


def write_report(inputs: Sequence[Path], output: Path, *, resamples: int = 10_000, seed: int = 0, overwrite: bool = False,
                 title: str = "E9 in-context frame baseline") -> dict[str, Any]:
    output = Path(output)
    if overwrite and output.exists():
        for name in ("report.md", "summary.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    outputs = discover(inputs)
    if not outputs:
        raise FileNotFoundError(f"no in-context scorings under {', '.join(map(str, inputs))}")
    git_at_start = prepare_output_dir(output)
    summary = analyze(outputs, resamples=resamples, seed=seed)
    summary.update(kind=REPORT_KIND, inputs=[str(p) for p in inputs], runs=sorted(o["summary"]["id"] for o in outputs))
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (Path(output) / "report.md").write_text(render(summary, title=title))
    write_run_metadata(Path(output), {"inputs": [str(p) for p in inputs], "resamples": resamples, "seed": seed},
                       git_at_start=git_at_start, kind=REPORT_KIND, runs=len(outputs))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--run", type=Path, required=True); score.add_argument("--output", type=Path, default=None)
    score.add_argument("--variants", nargs="+", default=list(VARIANTS)); score.add_argument("--max-prefix", type=int, default=1024)
    score.add_argument("--seed", type=int, default=0); score.add_argument("--limit-windows", type=int, default=None)
    score.add_argument("--eval-batch", type=int, default=None); score.add_argument("--device", default="cuda")
    score.add_argument("--glossary", type=Path, default=None)
    score.add_argument("--resume", action="store_true"); score.add_argument("--overwrite", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--inputs", type=Path, nargs="+", required=True); report.add_argument("--output", type=Path, required=True)
    report.add_argument("--resamples", type=int, default=10_000); report.add_argument("--seed", type=int, default=0)
    report.add_argument("--overwrite", action="store_true"); report.add_argument("--title", default="E9 in-context frame baseline")
    args = parser.parse_args(argv)
    if args.command == "score":
        result = score_run(args.run, args.output, variants=args.variants, max_prefix=args.max_prefix, seed=args.seed,
                           limit_windows=args.limit_windows, device=args.device, eval_batch=args.eval_batch, glossary=args.glossary,
                           resume=args.resume, overwrite=args.overwrite)
        print(json.dumps({k: result[k] for k in ("id", "prefix", "own_check", "seconds")}, indent=2, default=str))
    else:
        write_report(args.inputs, args.output, resamples=args.resamples, seed=args.seed, overwrite=args.overwrite, title=args.title)


if __name__ == "__main__":
    sys.exit(main())
