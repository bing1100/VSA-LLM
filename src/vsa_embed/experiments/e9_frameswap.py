"""E9 evaluation-time frame swap (decision 64, author 2026-10-09; pre-registration
`experiments/e9-retrofit/preregistration-frameswap.md`): within one trained model, does a term's OWN frame matter?

T4's arm battery left the question open: C5 beats C0′ after held-out terms, but the shuffled-frames arm C5sh (every entry,
held-out included, reads another entry's frame) keeps that gain while losing it on seen, rare and unseen terms. Retraining
answers "does a model trained on wrong frames do as well"; this tool answers "does this model use the right frame", on
any finished run, without training: the run is re-scored on its own evaluation windows with the frames of a target entry
set swapped, emptied or averaged, and every other entry left as trained.

**Variants** (applied to the target entries only; every other span reads its own row):

- `own` — the run as trained (the reference; bf16 autocast, `training.lm.evaluate`; `own_check` replays the run's final
  evaluation in `eval_windows.npz`, window by window);
- `other` — each target entry reads another target entry's frame: one seeded cycle through the target set (no fixed
  point; the C5sh rule for distinct labels, `real_relations.derange_labels`), so the frame read has the same training
  status — the cleanest specificity test;
- `other-any` — each target entry reads the frame of a random non-target entry (drawn without replacement while the pool
  lasts): what C5sh gave held-out entries;
- `empty` — no injection for target entries (zero row, `e5_common.override_rows`; the channel stays on elsewhere);
- `mean` — the mean context-free row over the non-target entries (`e5_common.entry_rows`' rule), for every target entry.

The derangement and the draws are seeded by `(--seed, the run's seed, the target set)`, so the seeds of a model read
different derangements and the pooled estimate averages over them.

**Mechanism.** `other` / `other-any` remap the entry id of the target spans at the model's input (`ChannelLM.embed`): the
channel composes the other entry's frame with the span's own local context (P1) and an unbinding readout (U5*) reads the
other entry's frame store, while the strata come from the original spans. With an `induced` concept factor (every E9
compose arm) a composed row is a function of the frame and the context only, so the remap is exactly a frame swap (a
`hybrid` / `free` factor would move with the frame). The same remap gives a row-source arm (C6m / C6d / C6g) the other
entry's frozen source vector through its trained projector. `empty` and `mean` override rows (`override_rows`, which also
blocks the readout for those entries). A separate module rather than an `e9_rescore` variant: rescore variants are weight
variants of one model grouped by quantizer; these are entry-set variants with target sets, window subsetting and a
within-model report of their own (they reuse its filler tables and helpers).

**Target sets** (`--entries`; several in one job share `own`): `heldout` (the ontology's held-out entries, never linked in
training), `unseen` (training frequency 0, not held out), `rare_seen` (1–9). A compose channel's target set and pools drop
entries with an empty frame (no row in any variant). Each set's matched stratum is `after_heldout` / `after_unseen` /
`after_rare_seen`.

**Channels.** Frame channels (C5 and its arms C5rf / C5ut / C5tr / C5sh; readout arms U5*) and row-source arms (C6*).
C5sh's own frame is the shuffled frame it was trained with. C2 (free table) is scored with a warning: its held-out rows are
one fallback row (`other` = `own` exactly) and its unseen rows were never trained. Runs without per-entry rows (C0′, P0:
no channel; hashed memories) are refused.

**Windows.** The run's evaluation windows and strata, with the filler strata when the track's filler table exists
(`--fillers auto`; `build` builds it; a licensed track uses none unless given a path). A variant only changes windows
that hold a target span; the others keep `own`'s sums (`--all-windows` evaluates them too). `--limit-windows N` keeps the
first N windows (CPU smokes; the summary is flagged `smoke`).

**Outputs** (default `experiments/e9-retrofit/frameswap/<stage>/<run>/`, outside the run folder, so a licensed stage's
outputs stay under `frameswap/t1c*`, whose `.npz` files git ignores): `windows.npz` (`strata`, `starts`, `count`,
`sum_own`, `sum_<variant>@<entries>`: strata × windows, paired with each other and with `eval_windows.npz`; `touched_<entries>`:
windows holding a target span), `summary.json` (manifest: git sha, run, variants, seeds, target counts; per variant the
stratified losses and the point difference to `own`; `own_check`), `resolved_config.yaml`, `manifest.json`. No output holds
an entry id or name: target sets are reported as counts (the mapping is a function of the ontology and the seeds).
`--resume` keeps the variants already in `windows.npz`.

**Report** (`report`): per model group (host × training mode × model) of the given output folders, target set, stratum
and variant, the paired difference `variant − own` (absolute, and relative to `own`'s loss; token-weighted, pooled over the
seeds per window as `e9_report`'s dimension 1; 95% cluster bootstrap over evaluation windows,
`statistics.paired_ratio_bootstrap`; Holm over the variants within a target set × stratum), per-seed rows and `report.md`.
Primary: `other − own` on `after_heldout` (positive = the own frame helps: the held-out gain is ontology-specific).

    python -m vsa_embed.experiments.e9_frameswap score --run RUN [--entries heldout unseen rare_seen]
        [--variants own other other-any empty mean] [--seed 0] [--limit-windows N] [--fillers auto|build|none|PATH] [--resume]
    python -m vsa_embed.experiments.e9_frameswap report --inputs experiments/e9-retrofit/frameswap/t4 [--output DIR]
    python -m vsa_embed.experiments.e9_frameswap commands [--root PATH] > experiments/e9-retrofit/frameswap/queue-commands.sh
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from ..data.corpus import TokenCorpus, eval_windows
from ..provenance import git_state, prepare_output_dir, write_run_metadata
from ..row_sources import load_filler_index
from ..statistics import holm_adjust, paired_ratio_bootstrap
from ..training.lm import evaluate, load_final, load_window_losses, resolve_config
from .e5_common import override_rows
from .e9_rescore import _sink_arrays, _write_record, ensure_filler_table, filler_table_path

ROOT = Path("experiments/e9-retrofit")
VARIANTS = ("own", "other", "other-any", "empty", "mean")
SWAPS = VARIANTS[1:]
ENTRY_SETS = {"heldout": "after_heldout", "unseen": "after_unseen", "rare_seen": "after_rare_seen"}
SET_INDEX = {name: i for i, name in enumerate(ENTRY_SETS)}
REPORT_STRATA = ("{}", "{}_filler", "{}_nonfiller", "after", "unlinked", "all")
ALPHA = 0.05
STEM = re.compile(r"^(?P<host>.+)-(?P<mode>[^-]+)-(?P<model>[^-]+)-s(?P<seed>\d+)$")
KIND, REPORT_KIND = "e9-frameswap", "e9-frameswap-report"


def variant_key(variant: str, entries: str) -> str:
    return "own" if variant == "own" else f"{variant}@{entries}"


# ---------------------------------------------------------------- target sets and swap plans


def target_entries(ontology: dict[str, Any], kind: str) -> np.ndarray:
    """Sorted entry ids of a target set (`heldout`, `unseen`, `rare_seen`; see the module docstring)."""
    count = int(ontology["entry_count"])
    heldout = np.zeros(count, dtype=bool)
    heldout[np.asarray(list(ontology["heldout_entries"]), dtype=np.int64)] = True
    if kind == "heldout":
        return np.flatnonzero(heldout)
    if kind not in ENTRY_SETS:
        raise ValueError(f"--entries must be among {sorted(ENTRY_SETS)}")
    if ontology.get("train_frequency") is None:
        raise ValueError(f"the {kind} set needs the ontology's train_frequency")
    frequency = np.asarray(ontology["train_frequency"])[:count]
    return np.flatnonzero(~heldout & ((frequency == 0) if kind == "unseen" else (frequency >= 1) & (frequency <= 9)))


def derangement(entries: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    """Each entry → another entry of the same set: one random cycle through the set (no fixed point); {} below 2."""
    order = rng.permutation(np.unique(np.asarray(entries, dtype=np.int64)))
    return dict(zip(order.tolist(), np.roll(order, -1).tolist())) if order.size >= 2 else {}


def draw_others(entries: np.ndarray, pool: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    """Each entry → a random entry of `pool` outside the set (without replacement while the pool lasts)."""
    entries = np.unique(np.asarray(entries, dtype=np.int64))
    pool = np.setdiff1d(np.asarray(pool, dtype=np.int64), entries)
    if not entries.size or not pool.size:
        return {}
    picks = rng.permutation(pool)[:entries.size] if pool.size >= entries.size else rng.choice(pool, entries.size)
    return dict(zip(entries.tolist(), picks.tolist()))


@torch.no_grad()
def mean_row(channel: Any, entries: np.ndarray, *, chunk: int = 8192) -> torch.Tensor:
    """Mean context-free row (model width, after the host scale; float32, CPU) over `entries`."""
    device = next(channel.parameters()).device
    total = torch.zeros(channel.gate.in_features // 2, dtype=torch.float64)
    for part in torch.as_tensor(np.asarray(entries, dtype=np.int64)).split(chunk):
        total += channel.rows({"entry": part.to(device)}).double().sum(0).cpu()
    return (total / max(1, len(entries))).float()


@contextlib.contextmanager
def remapped(model: Any, mapping: dict[int, int]) -> Iterator[None]:
    """Within the block every span of an entry in `mapping` reaches the model (channel rows with the span's own context,
    readout store) as the mapped entry; every other span, and the caller's strata, are unchanged."""
    if not mapping:
        yield
        return
    size = max(int(model.channel.entry_count), max(mapping) + 1, max(mapping.values()) + 1)
    lookup = torch.arange(size)
    lookup[torch.tensor(list(mapping), dtype=torch.long)] = torch.tensor(list(mapping.values()), dtype=torch.long)
    original = model.embed

    def embed(input_ids: torch.Tensor, spans: dict[str, torch.Tensor] | None) -> torch.Tensor:
        if spans is not None and spans["entry"].numel():
            spans = {**spans, "entry": lookup.to(spans["entry"].device)[spans["entry"]]}
        return original(input_ids, spans)

    model.embed = embed
    try:
        yield
    finally:
        del model.embed


def variant_context(model: Any, variant: str, plan: dict[str, Any]) -> contextlib.AbstractContextManager:
    """The context in which `model` evaluates `variant` for one target set's `plan` (`swap_plan`)."""
    if variant == "own":
        return contextlib.nullcontext()
    if variant in ("other", "other-any"):
        return remapped(model, plan[variant])
    if variant in ("empty", "mean"):                  # only the target entries the windows link need an overriding row
        row = torch.zeros(model.channel.gate.in_features // 2) if variant == "empty" else plan["mean"]
        return override_rows(model.channel, {int(e): row for e in plan.get("present", plan["targets"])})
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def usable_entries(channel: Any) -> np.ndarray:
    """Entries with a row of their own: a compose channel's entries with a non-empty frame; every entry otherwise."""
    if channel.mode == "compose":
        return channel.composer.schedule.degrees.cpu().numpy() > 0
    return np.ones(int(channel.entry_count), dtype=bool)


def swap_plan(channel: Any, ontology: dict[str, Any], kind: str, variants: Sequence[str], *, seed: int, run_seed: int,
              linked: np.ndarray | None = None) -> dict[str, Any]:
    """Target entries and each variant's mapping / row for one target set (entry ids stay in memory, never in outputs);
    `linked`: the entries the evaluation windows link (`present` = the targets among them)."""
    usable = usable_entries(channel)
    found = target_entries(ontology, kind)
    found = found[found < usable.size]
    targets = found[usable[found]]
    inside = np.zeros(usable.size, dtype=bool)
    inside[targets] = True
    pool = np.flatnonzero(usable & ~inside)
    rng = np.random.default_rng([int(seed), int(run_seed), SET_INDEX[kind]])
    plan: dict[str, Any] = {"targets": targets, "record": {"entries": int(targets.size), "dropped_empty_frames": int(found.size - targets.size),
                                                           "pool": int(pool.size), "stratum": ENTRY_SETS[kind]}}
    if linked is not None:
        plan["present"] = np.intersect1d(targets, linked)
        plan["record"]["entries_in_windows"] = int(plan["present"].size)
    if "other" in variants:
        plan["other"] = derangement(targets, rng)
    if "other-any" in variants:
        plan["other-any"] = draw_others(targets, pool, rng)
        plan["record"]["other_any_with_replacement"] = bool(pool.size < targets.size)
    if "mean" in variants and pool.size:
        plan["mean"] = mean_row(channel, pool)
    return plan


# ---------------------------------------------------------------- scoring a run


def _resolve_fillers(spec: str | Path | None, track: str | None, family: str, licensed: bool) -> Path | None:
    if spec is None or str(spec) == "none":
        return None
    if str(spec) == "auto":
        if licensed or not track:
            return None
        path = filler_table_path(track, family)
        return path if path.exists() else None
    if str(spec) == "build":
        if licensed or not track:
            raise ValueError("--fillers build needs an unlicensed track recorded in the run (e9_track); pass a path instead")
        return ensure_filler_table(track, family)
    return Path(spec)


def _licensed(track: str | None) -> bool:
    from .e9_tracks import TRACKS
    return bool(track and track in TRACKS and TRACKS[track].licensed) or bool(track and track.startswith("t1c"))


def own_check(run_dir: Path, strata: list[str], starts: Sequence[int], sums: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    """`own` against the run's final evaluation in `eval_windows.npz`, paired by window start (a window subset works)."""
    path = Path(run_dir) / "eval_windows.npz"
    if not path.exists():
        return {"available": False}
    saved = load_window_losses(path)
    if not saved["evals"]:
        return {"available": False}
    column = {int(s): i for i, s in enumerate(saved["starts"].tolist())}
    if any(int(s) not in column for s in starts):
        return {"available": True, "paired": False, "detail": "other evaluation windows"}
    tokens = max(saved["evals"])
    run_sums, run_counts = saved["evals"][tokens]
    common = [s for s in strata if s in saved["strata"]]
    rows, mine, cols = [saved["strata"].index(s) for s in common], [strata.index(s) for s in common], [column[int(s)] for s in starts]
    theirs, ours = run_sums[rows][:, cols], sums[mine]
    record = {"available": True, "paired": True, "final_tokens": int(tokens), "windows": len(cols), "strata": len(common),
              "counts_equal": bool(np.array_equal(run_counts[rows][:, cols], counts[mine])),
              "max_abs_window_sum_diff": float(np.abs(theirs - ours).max()) if common else None}
    if "all" in common:                               # the loss difference per target token on `all`
        i = common.index("all")
        record["all_loss_diff"] = float((ours[i].sum() - theirs[i].sum()) / max(1, int(counts[strata.index("all")].sum())))
    return record


def _stratified(sums: np.ndarray, counts: np.ndarray, strata: list[str]) -> dict[str, dict[str, float]]:
    total, n = sums.sum(1), counts.sum(1)
    return {s: {"loss": float(total[i] / n[i]) if n[i] else float("nan"), "tokens": int(n[i])} for i, s in enumerate(strata)}


def _differences(sums: dict[str, np.ndarray], counts: np.ndarray, strata: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Point differences `variant − own` (absolute per target token, relative to own's loss) per stratum."""
    own, n = sums["own"].sum(1), counts.sum(1)
    out = {}
    for key, values in sums.items():
        if key != "own":
            d = values.sum(1) - own
            out[key] = {s: {"delta": float(d[i] / n[i]), "relative": float(d[i] / own[i]) if own[i] else float("nan")}
                        for i, s in enumerate(strata) if n[i]}
    return out


def load_frameswap(folder: Path) -> dict[str, Any] | None:
    """`{"summary", "strata", "starts", "count", "sums": {key: strata × windows}, "touched": {set: bool[windows]}}` or None."""
    folder = Path(folder)
    if not (folder / "windows.npz").exists() or not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("kind") != KIND:
        return None
    with np.load(folder / "windows.npz") as data:
        return {"summary": summary, "folder": str(folder), "strata": data["strata"].tolist(), "starts": data["starts"],
                "count": data["count"], "sums": {k[4:]: data[k] for k in data.files if k.startswith("sum_")},
                "touched": {k[8:]: data[k] for k in data.files if k.startswith("touched_")}}


def _write_windows(path: Path, strata: list[str], starts: Sequence[int], counts: np.ndarray, sums: dict[str, np.ndarray],
                   touched: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, strata=np.asarray(strata), starts=np.asarray(starts, dtype=np.int64), count=counts,
                            **{f"sum_{k}": v for k, v in sums.items()}, **{f"touched_{k}": v for k, v in touched.items()})
    temporary.replace(path)


def default_output(run_dir: Path, root: Path = ROOT) -> Path:
    return Path(root) / "frameswap" / Path(run_dir).parent.name / Path(run_dir).name


def score_run(run_dir: Path, output: Path | None = None, *, entries: Sequence[str] = ("heldout",), variants: Sequence[str] = VARIANTS,
              seed: int = 0, limit_windows: int | None = None, fillers: str | Path | None = "auto", device: str = "cuda",
              eval_batch: int | None = None, all_windows: bool = False, resume: bool = False, overwrite: bool = False,
              root: Path = ROOT, log: Any = print) -> dict[str, Any]:
    """Evaluate `run_dir` under each variant × target set on its evaluation windows (see the module docstring)."""
    run_dir = Path(run_dir)
    variants = ["own", *[v for v in dict.fromkeys(variants) if v != "own"]]
    unknown = sorted(set(variants) - set(VARIANTS)) + sorted(set(entries) - set(ENTRY_SETS))
    if unknown:
        raise ValueError(f"unknown variants or target sets {unknown}; variants {VARIANTS}, sets {tuple(ENTRY_SETS)}")
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu", mmap=True)
    config = resolve_config(final["config"])
    del final
    mode = config["channel"]["mode"]
    if mode in ("none", "hashed"):
        raise ValueError(f"{run_dir.name}: a frame swap needs per-entry rows; channel mode {mode!r} has none (C0′, P0, C1h)")
    warnings = (["C2 (free table): held-out rows are one fallback row (other = own exactly) and unseen rows were never trained"]
                if mode == "free" else [])
    if eval_batch:
        config["eval"]["batch"] = int(eval_batch)
    track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
    licensed = _licensed(track)
    output = Path(output) if output else default_output(run_dir, root)
    if licensed and not output.resolve().is_relative_to((Path(root) / "frameswap").resolve()):
        raise ValueError(f"a licensed track ({track}) writes frame-swap outputs only under {Path(root) / 'frameswap'}")
    fillers = _resolve_fillers(fillers, track, family, licensed)
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    heldout = set(int(e) for e in ontology["heldout_entries"])
    frequency = np.asarray(ontology["train_frequency"]) if ontology.get("train_frequency") is not None else None
    index = load_filler_index(fillers, ontology) if fillers else None
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    length, min_subtokens = int(config["model"]["seq_len"]), int(config["data"]["min_subtokens"])
    starts = eval_windows(corpus, count=config["eval"]["windows"], length=length)
    if limit_windows:
        starts = starts[:int(limit_windows)]
    if overwrite and output.exists():
        for name in ("windows.npz", "summary.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    if not resume and (output / "windows.npz").exists():
        raise FileExistsError(f"{output} holds a frame swap; pass --resume or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    git_at_start = git_state()
    existing = load_frameswap(output) if resume else None
    if existing is not None and (not np.array_equal(existing["starts"], np.asarray(starts)) or existing["summary"]["seed"] != seed
                                 or existing["summary"].get("fillers") != (str(fillers) if fillers else None)):
        raise ValueError(f"{output} was scored with other windows, seed or filler table; use --overwrite")
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_final(run_dir / "final.pt", target)
    run_seed = int(config["seed"])
    stage, stem = run_dir.parent.name, run_dir.name
    match = STEM.match(stem)
    channel = model.channel
    summary: dict[str, Any] = {
        "kind": KIND, "smoke": bool(limit_windows), "run": str(run_dir), "id": f"{stage}__{stem}", "stage": stage, "stem": stem,
        "host": match["host"] if match else config["model"].get("pretrained"), "host_mode": match["mode"] if match else None,
        "model": match["model"] if match else None, "run_seed": run_seed, "track": track, "licensed": licensed,
        "channel": {"mode": mode, "frames": config["channel"].get("frames"), "operator": config["channel"].get("operator"),
                    "concept_factor": config["channel"].get("concept_factor"), "readout": bool(getattr(channel, "readout", None))},
        "variants": variants, "entries": list(entries), "seed": int(seed), "windows": len(starts), "limit_windows": limit_windows,
        "all_windows": bool(all_windows), "fillers": str(fillers) if fillers else None, "device": str(target),
        "git_sha": git_at_start.get("git_sha"), "git_dirty": git_at_start.get("git_dirty"), "warnings": warnings,
        "targets": {}, "seconds": dict(existing["summary"].get("seconds", {})) if existing else {}}
    for w in warnings:
        log(json.dumps({"run": summary["id"], "warning": w}))
    sums: dict[str, np.ndarray] = dict(existing["sums"]) if existing else {}
    touched: dict[str, np.ndarray] = dict(existing["touched"]) if existing else {}
    strata, counts = (existing["strata"], existing["count"]) if existing else (None, None)

    def run(subset: list[int]) -> tuple[list[str], np.ndarray, np.ndarray]:
        sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        evaluate(model, corpus, subset, config, frequency, heldout, target, window_sink=sink, fillers=index)
        return _sink_arrays(sink)

    def save() -> None:
        _write_windows(output / "windows.npz", strata, starts, counts, sums, touched)
        _write_record(output / "summary.json", summary)

    if "own" not in sums:
        began = time.monotonic()
        strata, own, counts = run(list(starts))
        sums["own"] = own
        summary["seconds"]["own"] = round(time.monotonic() - began, 2)
        save()
        log(json.dumps({"run": summary["id"], "variant": "own", "seconds": summary["seconds"]["own"]}))
    window_entries = [corpus.window(int(s), length, min_subtokens=min_subtokens)[1]["entry"] for s in starts]
    linked = np.unique(np.concatenate([np.zeros(0, dtype=np.int64), *window_entries]).astype(np.int64))
    for kind in entries:
        plan = swap_plan(channel, ontology, kind, variants, seed=seed, run_seed=run_seed, linked=linked)
        hit = np.asarray([bool(np.isin(e, plan["targets"]).any()) for e in window_entries], dtype=bool)
        touched[kind] = hit
        record = {**plan["record"], "windows_touched": int(hit.sum()),
                  "stratum_tokens": int(counts[strata.index(ENTRY_SETS[kind])].sum()) if ENTRY_SETS[kind] in strata else 0,
                  "skipped": {}}
        summary["targets"][kind] = record
        subset = list(range(len(starts))) if all_windows else np.flatnonzero(hit).tolist()
        for variant in variants[1:]:
            key = variant_key(variant, kind)
            if key in sums:
                continue
            if (variant in ("other", "other-any") and not plan.get(variant)) or (variant == "mean" and "mean" not in plan):
                record["skipped"][variant] = "fewer than 2 target entries" if variant == "other" else "empty target set or pool"
                continue
            began = time.monotonic()
            values = sums["own"].copy()
            if subset:
                with variant_context(model, variant, plan):
                    names, these, these_counts = run([starts[i] for i in subset])
                if names != strata or not np.array_equal(these_counts, counts[:, subset]):
                    raise RuntimeError(f"{run_dir}: {key} produced other strata or target counts")
                values[:, subset] = these
            sums[key] = values
            summary["seconds"][key] = round(time.monotonic() - began, 2)
            save()
            log(json.dumps({"run": summary["id"], "variant": key, "windows": len(subset), "seconds": summary["seconds"][key]}))
    del model
    if target.type == "cuda":
        torch.cuda.empty_cache()
    summary["losses"] = {key: _stratified(values, counts, strata) for key, values in sums.items()}
    summary["differences"] = _differences(sums, counts, strata)
    summary["own_check"] = own_check(run_dir, strata, starts, sums["own"], counts)
    save()
    settings = {"run": str(run_dir), "entries": list(entries), "variants": variants, "seed": int(seed), "limit_windows": limit_windows,
                "fillers": str(fillers) if fillers else None, "eval_batch": eval_batch, "all_windows": bool(all_windows), "device": device}
    write_run_metadata(output, settings, git_at_start=git_at_start, device=target, kind=KIND, keys=sorted(sums))
    return summary


# ---------------------------------------------------------------- report


def discover(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Frame-swap outputs in the given folders (a folder itself, or its sub-folders)."""
    found: dict[str, dict[str, Any]] = {}
    for folder in map(Path, inputs):
        for candidate in [folder, *sorted(p for p in folder.iterdir() if p.is_dir())] if folder.is_dir() else []:
            data = load_frameswap(candidate)
            if data is not None:
                found.setdefault(str(candidate.resolve()), data)
    return list(found.values())


def pooled_difference(outputs: Sequence[dict[str, Any]], key: str, stratum: str, *, resamples: int, seed: int) -> dict[str, Any] | None:
    """`key − own` on `stratum`, token-weighted, pooled over the outputs (seeds) per window — or with windows as separate
    clusters when the seeds' windows differ; 95% cluster bootstrap over windows."""
    usable = [o for o in outputs if key in o["sums"] and stratum in o["strata"]]
    if not usable:
        return None
    parts = [(o["sums"][key][o["strata"].index(stratum)] - o["sums"]["own"][o["strata"].index(stratum)],
              o["count"][o["strata"].index(stratum)].astype(np.float64), o["sums"]["own"][o["strata"].index(stratum)]) for o in usable]
    same = all(np.array_equal(o["starts"], usable[0]["starts"]) for o in usable)
    d, n, b = ((sum(p[i] for p in parts) for i in range(3)) if same else (np.concatenate([p[i] for p in parts]) for i in range(3)))
    if n.sum() <= 0:
        return None
    boot = paired_ratio_bootstrap(d, n, b, resamples=resamples, seed=seed)
    return {"seeds": sorted(int(o["summary"]["run_seed"]) for o in usable), "pooled_by_window": same, "delta": boot["mean"],
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "p_value": boot["p_value"], "relative": boot["relative"],
            "relative_ci_low": boot["relative_ci_low"], "relative_ci_high": boot["relative_ci_high"],
            "windows": boot["nonempty_clusters"], "tokens": int(n.sum()), "own_loss": float(b.sum() / n.sum()),
            "smoke": any(o["summary"].get("smoke") for o in usable)}


def group_outputs(outputs: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for o in sorted(outputs, key=lambda o: (o["summary"]["stage"], o["summary"]["stem"])):
        s = o["summary"]
        groups.setdefault(f"{s['stage']} · {s['host']} · {s['host_mode']} · {s['model']}", []).append(o)
    return groups


def analyze(outputs: Sequence[dict[str, Any]], *, resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Per group, target set, stratum and variant: the pooled and per-seed `variant − own` (Holm over variants)."""
    result: dict[str, Any] = {"groups": {}}
    for label, members in group_outputs(outputs).items():
        sets = [k for k in ENTRY_SETS if any(k in o["summary"]["entries"] for o in members)]
        group: dict[str, Any] = {"seeds": sorted(int(o["summary"]["run_seed"]) for o in members), "stage": members[0]["summary"]["stage"],
                                 "smoke": any(o["summary"].get("smoke") for o in members), "sets": {},
                                 "warnings": sorted({w for o in members for w in o["summary"].get("warnings", [])}),
                                 "own_checks": {int(o["summary"]["run_seed"]): o["summary"].get("own_check") for o in members},
                                 "targets": {int(o["summary"]["run_seed"]): o["summary"].get("targets") for o in members}}
        for kind in sets:
            matched = ENTRY_SETS[kind]
            strata = [s.format(matched) for s in REPORT_STRATA if any(s.format(matched) in o["strata"] for o in members)]
            block: dict[str, Any] = {"stratum": matched, "strata": {}, "per_seed": {}}
            for stratum in strata:
                rows = {v: c for v in SWAPS if (c := pooled_difference(members, variant_key(v, kind), stratum, resamples=resamples,
                                                                       seed=seed)) is not None}
                tested = [c for c in rows.values() if c.get("p_value") is not None]
                for comparison, adjusted in zip(tested, holm_adjust([c["p_value"] for c in tested])):
                    comparison.update(holm_p=adjusted, significant=adjusted < ALPHA)
                block["strata"][stratum] = rows
            for o in members:
                per = {v: c for v in SWAPS if (c := pooled_difference([o], variant_key(v, kind), matched, resamples=resamples,
                                                                      seed=seed)) is not None}
                block["per_seed"][int(o["summary"]["run_seed"])] = per
            group["sets"][kind] = block
        primary = group["sets"].get("heldout", {}).get("strata", {}).get("after_heldout", {}).get("other")
        group["primary"] = None if primary is None else {
            **{k: primary.get(k) for k in ("delta", "ci_low", "ci_high", "relative", "relative_ci_low", "relative_ci_high", "holm_p")},
            "ontology_specific": bool(primary["delta"] > 0 and primary.get("holm_p", 1.0) < ALPHA)}
        result["groups"][label] = group
    return result


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:+.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:+.{digits}f}"


def render(summary: dict[str, Any], *, title: str) -> str:
    smoke = any(g["smoke"] for g in summary["groups"].values())
    lines = [f"# {'SMOKE — ' if smoke else ''}{title}", "",
             "Evaluation-time frame swap (`e9_frameswap`; pre-registration `experiments/e9-retrofit/preregistration-frameswap.md`). "
             "Each row is `variant − own` within the same trained model on its evaluation windows (positive = the variant is worse, "
             "i.e. the run's own rows help): Δ in nats per target token and relative to `own`'s loss, token-weighted and pooled "
             "over seeds per window, with 95% cluster-bootstrap intervals over windows; Holm over the four variants within a "
             "target set × stratum (* = Holm p < 0.05). Variants: `other` = another target entry's frame (derangement within the "
             "set), `other-any` = a random non-target entry's frame, `empty` = no injection, `mean` = the mean non-target row.", "",
             "**Reading rule:** a held-out gain counts as ontology-specific only if `other − own` > 0 on `after_heldout` with "
             "Holm p < 0.05.", ""]
    if smoke:
        lines += ["**SMOKE:** at least one input was scored on a window subset (`--limit-windows`); the numbers check the "
                  "pipeline only.", ""]
    for label, group in summary["groups"].items():
        lines += [f"## {label}", "", f"Seeds: {', '.join(map(str, group['seeds']))}" + (" (single seed: intervals cover windows only)"
                                                                                    if len(group["seeds"]) == 1 else "") + ".", ""]
        primary = group.get("primary")
        if primary:
            lines += [f"**Primary** (`other − own`, `after_heldout`): Δ {_num(primary['delta'])} "
                      f"[{_num(primary['ci_low'])}, {_num(primary['ci_high'])}], {_pct(primary['relative'])} "
                      f"[{_pct(primary['relative_ci_low'])}, {_pct(primary['relative_ci_high'])}], Holm p "
                      f"{primary['holm_p']:.3g} → {'ontology-specific' if primary['ontology_specific'] else 'not shown ontology-specific'}.", ""]
        for w in group["warnings"]:
            lines += [f"> Warning: {w}", ""]
        for kind, block in group["sets"].items():
            counts = {s: (t or {}).get(kind, {}) for s, t in group["targets"].items()}
            first = next(iter(counts.values()), {})
            lines += [f"### Target set `{kind}` (matched stratum `{block['stratum']}`)", "",
                      f"{first.get('entries', '—')} target entries ({first.get('dropped_empty_frames', 0)} dropped: empty frame); "
                      f"windows holding a target span per seed: {', '.join(str(c.get('windows_touched', '—')) for c in counts.values())}.", "",
                      "| stratum | variant | Δ nats/token [95% CI] | relative [95% CI] | Holm p | tokens |", "|---|---|---|---|---|---:|"]
            for stratum, rows in block["strata"].items():
                for variant, c in rows.items():
                    lines.append(f"| `{stratum}` | {variant} | {_num(c['delta'])} [{_num(c['ci_low'])}, {_num(c['ci_high'])}] | "
                                 f"{_pct(c['relative'])} [{_pct(c['relative_ci_low'])}, {_pct(c['relative_ci_high'])}] | "
                                 f"{c.get('holm_p', float('nan')):.3g}{'*' if c.get('significant') else ''} | {c['tokens']:,} |")
            lines += ["", f"Per seed (`{block['stratum']}`):", "", "| seed | own loss | " + " | ".join(SWAPS) + " |",
                      "|---|---|" + "---|" * len(SWAPS)]
            for s, per in block["per_seed"].items():
                own = next((c["own_loss"] for c in per.values()), None)
                lines.append(f"| {s} | {'—' if own is None else f'{own:.4f}'} | " + " | ".join(
                    f"{_pct(per[v]['relative'])} [{_pct(per[v]['relative_ci_low'])}, {_pct(per[v]['relative_ci_high'])}]" if v in per else "—"
                    for v in SWAPS) + " |")
            lines.append("")
        checks = {s: c for s, c in group["own_checks"].items() if c and c.get("paired")}
        if checks:
            lines += ["`own` replays the runs' final evaluation: max |window sum − eval_windows.npz| per seed " + ", ".join(
                f"s{s} {c['max_abs_window_sum_diff']:.3g} (counts {'equal' if c['counts_equal'] else 'DIFFER'})" for s, c in checks.items())
                + ".", ""]
    return "\n".join(lines) + "\n"


def write_report(inputs: Sequence[Path], output: Path, *, resamples: int = 10_000, seed: int = 0, overwrite: bool = False,
                 title: str = "E9 frame swap — is the own frame used?") -> dict[str, Any]:
    output = Path(output)
    if overwrite and output.exists():
        for name in ("report.md", "summary.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    outputs = discover(inputs)
    if not outputs:
        raise FileNotFoundError(f"no frame-swap outputs (summary.json + windows.npz) under {', '.join(map(str, inputs))}")
    git_at_start = prepare_output_dir(output)
    summary = {"kind": REPORT_KIND, "inputs": [o["summary"]["id"] for o in outputs], "resamples": resamples, "seed": seed,
               **analyze(outputs, resamples=resamples, seed=seed)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (output / "report.md").write_text(render(summary, title=title))
    write_run_metadata(output, {"inputs": [str(p) for p in inputs], "resamples": resamples, "seed": seed, "title": title},
                       git_at_start=git_at_start, kind=REPORT_KIND, outputs=len(outputs))
    return summary


# ---------------------------------------------------------------- queue commands (printed, never queued here)

# Blocks of decision 64 (author 2026-10-09): finished runs first, then each pending track at its evaluation level.
FINISHED = (("t4", "SmolLM2-360M", ("C5", "C5sh", "C6d"), (1, 2, 3)), ("t5", "SmolLM2-360M", ("C5", "C5sh", "C6d"), (1, 2, 3)),
            ("t5", "SmolLM2-135M", ("C5",), (1, 2, 3)), ("t1", "SmolLM2-135M", ("C5",), (1,)), ("t1", "SmolLM2-360M", ("C5",), (1,)),
            ("wordnet", "SmolLM2-135M", ("C5",), (1,)), ("wordnet", "SmolLM2-360M", ("C5",), (1,)), ("t4", "SmolLM2-135M", ("C5",), (1,)))
FINISHED_PRIORITY, FINISHED_REPORT_PRIORITY = 51.5, 51.6
PENDING = {"t7": (52, 53.9), "t7rood": (54.4971, 54.4972), "t8": (54.4991, 54.4992), "t1c-rood": (54.49982, 54.49983)}
PENDING_MODELS, PENDING_SEEDS = ("C5", "C5sh", "C6d"), (1, 2, 3)
# GPU seconds per evaluation window (1,024 tokens, bf16, batch 16) from `e9_rescore`'s `ref` variant (T5: 27 s and 14 s for
# 1,024 windows), plus a model load; a run's passes = 1 (`own`) + 4 variants × Σ_sets (share of windows holding a target).
SECONDS_PER_WINDOW = {"SmolLM2-360M": 0.027, "SmolLM2-135M": 0.014}
LOAD_SECONDS = 60.0


def touched_shares(run_dir: Path) -> dict[str, float] | None:
    """Share of evaluation windows with targets in each set's matched stratum (from any run's `eval_windows.npz`)."""
    path = Path(run_dir) / "eval_windows.npz"
    if not path.exists():
        return None
    saved = load_window_losses(path)
    counts = saved["evals"][max(saved["evals"])][1]
    return {k: float((counts[saved["strata"].index(s)] > 0).mean()) for k, s in ENTRY_SETS.items() if s in saved["strata"]}


def estimate_hours(host: str, windows: int, shares: dict[str, float] | None) -> float:
    shares = shares or {k: 1.0 for k in ENTRY_SETS}
    passes = 1 + len(SWAPS) * sum(shares.get(k, 1.0) for k in ENTRY_SETS)
    return (LOAD_SECONDS + passes * windows * SECONDS_PER_WINDOW.get(host, 0.027)) / 3600


def queue_lines(root: Path = ROOT, *, check_root: Path | None = None) -> list[str]:
    """The `jobqueue add` lines of decision 64's blocks (paths relative to the repository root); `check_root` is where run
    folders and configs are looked up (e.g. the main checkout's `experiments/e9-retrofit`)."""
    from .e9_tracks import TRACKS
    check = Path(check_root or root)
    py = "$PY"
    lines: list[str] = []

    def job(name: str, priority: float, command: str, hours: float, free: int = 5) -> str:
        return (f"PYTHONPATH=src {py} -m vsa_embed.jobqueue add --name {name} --priority {priority} --min-free-gb {free} --no-resume "
                f"-- {py} -m vsa_embed.experiments.e9_frameswap {command}   # ≈ {hours:.2f} GPU-h")

    def score(stage: str, stem: str, priority: float, hours: float) -> str:
        run, out = root / "runs" / stage / stem, root / "frameswap" / stage / stem
        fillers = " --fillers none" if TRACKS.get(stage) is not None and TRACKS[stage].licensed else ""
        return job(f"{stage}-{stem}-frameswap", priority, f"score --run {run} --output {out} --entries heldout unseen rare_seen "
                   f"--variants own other other-any empty mean{fillers} --resume", hours)

    def report(stage: str, priority: float) -> str:
        folder = root / "frameswap" / stage
        return job(f"{stage}-frameswap-report", priority, f"report --inputs {folder} --output {folder / 'report'} --overwrite", 0.0, free=1)

    blocks: list[tuple[str, list[str], float]] = []
    body, total, stages = [], 0.0, []
    for stage, host, models, seeds in FINISHED:
        spec = TRACKS[stage]
        for model in models:
            for s in seeds:
                stem = f"{host}-full-{model}-s{s}"
                if not (check / "runs" / stage / stem / "final.pt").exists():
                    body.append(f"# skipped {stage}/{stem}: no final.pt")
                    continue
                hours = estimate_hours(host, spec.windows, touched_shares(check / "runs" / stage / stem))
                body.append(score(stage, stem, FINISHED_PRIORITY, hours)); total += hours
        stages += [stage] if stage not in stages else []
    body += [report(stage, FINISHED_REPORT_PRIORITY) for stage in stages]
    blocks.append((f"finished runs at {FINISHED_PRIORITY} (reports at {FINISHED_REPORT_PRIORITY})", body, total))
    for stage, (priority, report_priority) in PENDING.items():
        spec, body, total = TRACKS[stage], [], 0.0
        reference = next((r for r in sorted((check / "runs" / stage).glob("*")) if (r / "eval_windows.npz").exists()), None) \
            if (check / "runs" / stage).is_dir() and not spec.licensed else None      # a licensed track's files are never read here
        shares = touched_shares(reference) if reference is not None else None
        for model in PENDING_MODELS:
            for s in PENDING_SEEDS:
                stem = f"SmolLM2-360M-full-{model}-s{s}"
                if not (check / "configs" / stage / f"{stem}.yaml").exists():
                    body.append(f"# skipped {stage}/{stem}: no config")
                    continue
                hours = estimate_hours("SmolLM2-360M", spec.windows, shares)
                body.append(score(stage, stem, priority, hours)); total += hours
        body.append(report(stage, report_priority))
        basis = "measured target-window shares" if shares else "upper bound: every window holds a target of every set"
        blocks.append((f"{stage} at {priority} (report at {report_priority}; {spec.windows:,} windows; {basis})", body, total))
    for title, body, total in blocks:
        jobs = sum(line.startswith("PYTHONPATH") for line in body)
        lines += ["", f"# --- {title}: {jobs} job(s), ≈ {total:.1f} GPU-h (idle-GPU estimate)", *body]
    return lines


HEADER = "\n".join([
    "#!/usr/bin/env bash",
    "# E9 frame swap (decision 64, author 2026-10-09; experiments/e9-retrofit/preregistration-frameswap.md): one scoring job per",
    "# run (target sets heldout, unseen, rare_seen; variants own, other, other-any, empty, mean) and one CPU-lane report per",
    "# stage. Printed by `python -m vsa_embed.experiments.e9_frameswap commands`; NOT EXECUTED by the agent that wrote them.",
    "# Run from the repository root of the main checkout after merging (a CPU-lane `-report` job starts only once every job",
    "# ahead of it is done; jobs of a pending track wait for its training at a lower priority number):",
    "#   cd /home/bhux/workplace/VSA-LLM && PY=/home/bhux/anaconda3/envs/vsa-repro/bin/python",
    "# T1c-ROOD lines write only under experiments/e9-retrofit/frameswap/t1c-rood/ (aggregates; its .npz files are git-ignored).",
    "set -euo pipefail",
    ': "${PY:?set PY to the pinned interpreter}"'])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score one finished run under the frame-swap variants")
    score.add_argument("--run", type=Path, required=True); score.add_argument("--output", type=Path, default=None)
    score.add_argument("--entries", nargs="+", default=["heldout"], choices=list(ENTRY_SETS))
    score.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    score.add_argument("--seed", type=int, default=0); score.add_argument("--limit-windows", type=int, default=None)
    score.add_argument("--fillers", default="auto", help="auto | build | none | a filler table path")
    score.add_argument("--device", default="cuda"); score.add_argument("--eval-batch", type=int, default=None)
    score.add_argument("--all-windows", action="store_true", help="evaluate every window under every variant")
    score.add_argument("--resume", action="store_true"); score.add_argument("--overwrite", action="store_true")
    report = sub.add_parser("report", help="paired variant − own per model group (all seeds)")
    report.add_argument("--inputs", type=Path, nargs="+", required=True); report.add_argument("--output", type=Path, default=None)
    report.add_argument("--resamples", type=int, default=10_000); report.add_argument("--seed", type=int, default=0)
    report.add_argument("--title", default="E9 frame swap — is the own frame used?"); report.add_argument("--overwrite", action="store_true")
    commands = sub.add_parser("commands", help="print the queue-commands script (never queues)")
    commands.add_argument("--root", type=Path, default=None, help="where run folders and configs are checked (default: ./experiments/e9-retrofit)")
    args = parser.parse_args(argv)
    if args.command == "score":
        summary = score_run(args.run, args.output, entries=args.entries, variants=args.variants, seed=args.seed,
                            limit_windows=args.limit_windows, fillers=args.fillers, device=args.device, eval_batch=args.eval_batch,
                            all_windows=args.all_windows, resume=args.resume, overwrite=args.overwrite)
        print(json.dumps({"run": summary["id"], "targets": summary["targets"], "own_check": summary["own_check"]}, default=str))
    elif args.command == "report":
        output = args.output or Path(args.inputs[0]) / "report"
        summary = write_report(args.inputs, output, resamples=args.resamples, seed=args.seed, overwrite=args.overwrite, title=args.title)
        print(json.dumps({label: g["primary"] for label, g in summary["groups"].items()}, default=float))
    else:
        print("\n".join([HEADER, *queue_lines(check_root=args.root)]))


if __name__ == "__main__":
    main()
