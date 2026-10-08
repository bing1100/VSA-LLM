"""E9 binding and unbinding program, step 2: evaluation of the unbinding readout arms (decisions 60–61; pre-registration
`experiments/e9-retrofit/preregistration-binding.md` §13). Evaluation only, on finished runs with `channel.readout`.

- **Readout on / off.** The trainer's stratified losses on the run's own evaluation windows (`training.lm.evaluate`, bf16
  autocast as trained) with the readout on (a replay of the final evaluation, checked with `e9_freqbias`'s replay rule)
  and with its gate switched off (`readout.readout_disabled`): the operator's loss without the readout (decision 61). Per
  window sums and counts for both (`windows.npz`), so the report pairs them by window.
- **Role prediction.** At every position t where the readout is active and the next tokens (from t + 1) verbalize a
  filler f of the readout's concept (the filler table of `e9_rescore`; the longest match), the relations that hold f in
  the concept's frame are the correct roles. Recorded: whether the predicted relation (argmax over relations, "no query"
  excluded) is correct (chance 1/R), the probability mass on correct relations, and on "no query".
- **Filler recovery.** At the same positions: the rank of f among every atomic by cosine with the readout's mixed vector
  (before the projector; `read`), and with the store read at the correct relation (`oracle`: the readout's own unbinding
  and cleanup given the role, which isolates the store from the query).

Outputs (`RUN/readout/`): `summary.json` (losses on / off per stratum, the replay check, diagnostics by subset of the
concept, by frame degree and by relation), `windows.npz` (strata × windows sums / counts, on and off), `positions.npz`
(per diagnosed position: concept, filler, correct, mass, none, ranks), `report.md`, `manifest.json`.

    python -m vsa_embed.experiments.e9_binding_readout evaluate --run RUN [--windows N] [--device cuda]
    python -m vsa_embed.experiments.e9_binding_readout queue --stage t5 --models U5 U5ut … --priority 55 [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from ..data.corpus import TokenCorpus, collate_windows, eval_windows
from ..readout import readout_disabled
from ..training import lm
from .e5_common import finish_output, json_ready, start_output, write_json

OUTPUT = "readout"
RESULT_FILES = ("summary.json", "windows.npz", "positions.npz", "report.md", "resolved_config.yaml", "manifest.json")
ROOT = Path("experiments/e9-retrofit")
SUBSETS = ("seen", "rare", "unseen", "heldout")


def subset_of(entry: int, frequency: np.ndarray | None, heldout: set[int]) -> str:
    if entry in heldout:
        return "heldout"
    count = int(frequency[entry]) if frequency is not None else 0
    return "seen" if count >= 10 else "rare" if count >= 1 else "unseen"


def filler_at(row: Sequence[int], position: int, atoms: frozenset[int], index: Any) -> int | None:
    """The atomic of `atoms` whose token sequence starts at `position` in `row` (the longest match), else None."""
    if position >= len(row):
        return None
    best = None
    for atomic, sequence in index.by_first.get(row[position], ()):
        if atomic in atoms and tuple(row[position:position + len(sequence)]) == sequence:
            if best is None or len(sequence) > best[1]:
                best = (atomic, len(sequence))
    return None if best is None else best[0]


@torch.no_grad()
def diagnose(model: Any, corpus: TokenCorpus, starts: Sequence[int], config: dict[str, Any], ontology: dict[str, Any],
             fillers: Any, device: torch.device, *, batch: int) -> dict[str, np.ndarray]:
    """Per position where the readout is active and the next tokens verbalize a filler of its concept (module
    docstring): concept, filler, degree, prediction correct, mass on the correct roles, mass on "no query", and the
    filler's rank by the readout's mixed vector (`read`) and by the store read at the correct role (`oracle`)."""
    readout = model.channel.readout
    composer = model.channel.composer
    offsets = np.asarray(ontology["offsets"], dtype=np.int64)
    relations = np.asarray(ontology["relations"], dtype=np.int64)
    filler_ids = np.asarray(ontology["fillers"], dtype=np.int64)
    count = composer.relation_count
    atomics = F.normalize(composer.atomic_vectors().float(), dim=-1)
    length, minimum = int(config["model"]["seq_len"]), int(config["data"]["min_subtokens"])
    rows_out: dict[str, list[Any]] = defaultdict(list)
    model.eval()
    for start in range(0, len(starts), batch):
        windows = [corpus.window(s, length, min_subtokens=minimum) for s in starts[start:start + batch]]
        ids, spans = collate_windows(windows)
        ids_d = ids.to(device)
        readout.capture = []
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(ids_d, spans={k: v.to(device) for k, v in spans.items()})
            captured = readout.capture
        finally:
            readout.capture = None
        tokens = ids.tolist()
        for record in captured:
            rows, cols = record["rows"].cpu().numpy(), record["cols"].cpu().numpy()
            concepts = record["concepts"].cpu().numpy()
            keep, found = [], []
            for i, (b, t, e) in enumerate(zip(rows.tolist(), cols.tolist(), concepts.tolist())):
                lo, hi = offsets[e], offsets[e + 1]
                atom = filler_at(tokens[b], t + 1, frozenset(filler_ids[lo:hi].tolist()), fillers)
                if atom is not None:
                    keep.append(i); found.append((e, atom, set(relations[lo:hi][filler_ids[lo:hi] == atom].tolist()), hi - lo))
            if not keep:
                continue
            index = torch.tensor(keep, device=record["probabilities"].device)
            probabilities = record["probabilities"][index].float()
            predicted = probabilities[:, :count].argmax(-1).cpu().numpy()
            mixed = F.normalize(record["mixed"][index].float(), dim=-1)
            read_scores = mixed @ atomics.T
            entries = torch.tensor([e for e, *_ in found], device=device)
            unique, inverse = torch.unique(entries, return_inverse=True)
            table = readout.filler_table(unique)                         # (unique, relations, dimension)
            for j, (e, atom, roles, degree) in enumerate(found):
                correct_mass = float(probabilities[j, sorted(roles)].sum())
                role = sorted(roles)[0]
                oracle = F.normalize(table[inverse[j], role].float(), dim=-1) @ atomics.T
                rank_read = 1.0 + float((read_scores[j] > read_scores[j, atom]).sum())
                rank_oracle = 1.0 + float((oracle > oracle[atom]).sum())
                rows_out["concept"].append(e); rows_out["filler"].append(atom); rows_out["degree"].append(degree)
                rows_out["correct"].append(float(int(predicted[j]) in roles)); rows_out["mass"].append(correct_mass)
                rows_out["none"].append(float(probabilities[j, count])); rows_out["rank_read"].append(rank_read)
                rows_out["rank_oracle"].append(rank_oracle); rows_out["role"].append(role)
                rows_out["candidate_roles"].append(len(roles))
    return {k: np.asarray(v) for k, v in rows_out.items()}


def _summary_block(positions: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any] | None:
    if not mask.any():
        return None
    return {"positions": int(mask.sum()), "concepts": int(np.unique(positions["concept"][mask]).size),
            "role_accuracy": float(positions["correct"][mask].mean()), "role_mass": float(positions["mass"][mask].mean()),
            "none_mass": float(positions["none"][mask].mean()),
            "read_mrr": float((1.0 / positions["rank_read"][mask]).mean()), "read_top1": float((positions["rank_read"][mask] == 1).mean()),
            "oracle_mrr": float((1.0 / positions["rank_oracle"][mask]).mean()),
            "oracle_top1": float((positions["rank_oracle"][mask] == 1).mean())}


def evaluate_run(run_dir: Path, output: Path | None = None, *, device: str | None = None, windows: int | None = None,
                 fillers: Path | str | None = "auto", overwrite: bool = False) -> dict[str, Any]:
    from ..row_sources import load_filler_index
    from .e9_freqbias import prefix_reference_check
    from .e9_rescore import ensure_filler_table
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / OUTPUT
    config_record = {"experiment": "e9-binding-readout", "run": str(run_dir), "windows": windows, "fillers": str(fillers)}
    if overwrite:
        for name in RESULT_FILES:
            if (output / name).is_file():
                (output / name).unlink()
    git_at_start = start_output(output, config_record)
    started = time.monotonic()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")
    config = lm.resolve_config(final["config"])
    del final
    if not (config.get("channel") or {}).get("readout"):
        raise ValueError(f"{run_dir} has no unbinding readout (channel.readout)")
    model = lm.load_final(run_dir / "final.pt", device_t)
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    frequency = np.asarray(ontology["train_frequency"]) if ontology.get("train_frequency") is not None else None
    heldout = {int(e) for e in ontology.get("heldout_entries", ())}
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    starts = eval_windows(corpus, count=int(config["eval"]["windows"]), length=int(config["model"]["seq_len"]))
    if windows:
        starts = starts[:windows]
    record: dict[str, Any] = {"run": str(run_dir), "stem": run_dir.name, "windows": len(starts), "variants": {}}
    arrays: dict[str, np.ndarray] = {"starts": np.asarray(starts, dtype=np.int64)}
    for variant, context in (("on", contextlib.nullcontext()), ("off", readout_disabled(model))):
        sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        with context:
            results = lm.evaluate(model, corpus, starts, config, frequency, heldout, device_t, window_sink=sink)
        strata = list(sink)
        sums = np.stack([np.concatenate(sink[s][0]) for s in strata])
        counts = np.stack([np.concatenate(sink[s][1]) for s in strata])
        arrays["strata"] = np.asarray(strata)
        arrays[f"sum_{variant}"], arrays[f"count_{variant}"] = sums, counts
        record["variants"][variant] = {s: {"loss": v["loss"], "targets": v["tokens"]} for s, v in results.items()}
        if variant == "on":
            record["ref_check"] = prefix_reference_check(run_dir, strata, starts, sums, counts, composing=True)
    track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
    table_path = ensure_filler_table(track, family) if fillers == "auto" and track else (Path(fillers) if fillers not in (None, "auto") else None)
    if table_path is not None:
        index = load_filler_index(table_path, ontology)
        positions = diagnose(model, corpus, starts, config, ontology, index, device_t, batch=int(config["eval"]["batch"]))
        record["diagnostics"] = {"relations": model.channel.composer.relation_count, "chance": 1.0 / model.channel.composer.relation_count,
                                 "subsets": {}, "degree": {}, "relations_by_role": {}}
        if positions:
            subsets = np.asarray([subset_of(int(e), frequency, heldout) for e in positions["concept"]])
            for subset in (*SUBSETS, "all"):
                block = _summary_block(positions, np.ones_like(subsets, dtype=bool) if subset == "all" else subsets == subset)
                if block:
                    record["diagnostics"]["subsets"][subset] = block
            for degree in sorted(set(positions["degree"].tolist())):
                block = _summary_block(positions, positions["degree"] == degree)
                if block:
                    record["diagnostics"]["degree"][str(int(degree))] = block
            names = list(ontology.get("relation_names") or [])
            for role in sorted(set(positions["role"].tolist())):
                block = _summary_block(positions, positions["role"] == role)
                if block:
                    record["diagnostics"]["relations_by_role"][names[role] if names else str(role)] = block
            np.savez_compressed(output / "positions.npz", **{k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in positions.items()})
    record["seconds"] = round(time.monotonic() - started, 1)
    np.savez_compressed(output / "windows.npz", **arrays)
    write_json(output / "summary.json", json_ready(record))
    (output / "report.md").write_text(render(record))
    finish_output(output, config_record, git_at_start=git_at_start, device=device_t)
    return record


def render(record: dict[str, Any]) -> str:
    lines = [f"# Readout evaluation — {record['stem']}", "", f"{record['windows']} evaluation windows; replay "
             f"{'ok' if (record.get('ref_check') or {}).get('replay_ok') else 'not checked / failed'}.", "",
             "| stratum | loss (readout on) | loss (readout off) | off − on |", "|---|---:|---:|---:|"]
    on, off = record["variants"].get("on", {}), record["variants"].get("off", {})
    for stratum in ("all", "after", "after_heldout", "after_rare", "after_frequent", "inside", "unlinked"):
        if stratum in on and stratum in off:
            lines.append(f"| {stratum} | {on[stratum]['loss']:.4f} | {off[stratum]['loss']:.4f} | {off[stratum]['loss'] - on[stratum]['loss']:+.4f} |")
    diagnostics = record.get("diagnostics")
    if diagnostics and diagnostics.get("subsets"):
        lines += ["", f"Role prediction (chance {diagnostics['chance']:.3f}) and filler recovery at positions before a filler:", "",
                  "| subset | positions | role accuracy | role mass | no-query mass | read MRR | oracle MRR |", "|---|---:|---:|---:|---:|---:|---:|"]
        for subset, v in diagnostics["subsets"].items():
            lines.append(f"| {subset} | {v['positions']} | {v['role_accuracy']:.3f} | {v['role_mass']:.3f} | {v['none_mass']:.3f} | "
                         f"{v['read_mrr']:.3f} | {v['oracle_mrr']:.3f} |")
    return "\n".join(lines) + "\n"


def load_evaluation(folder: Path) -> dict[str, Any] | None:
    folder = Path(folder)
    if not (folder / "summary.json").exists():
        return None
    out: dict[str, Any] = {"summary": json.loads((folder / "summary.json").read_text())}
    if (folder / "windows.npz").exists():
        with np.load(folder / "windows.npz") as data:
            out["windows"] = {k: data[k] for k in data.files}
    if (folder / "positions.npz").exists():
        with np.load(folder / "positions.npz") as data:
            out["positions"] = {k: data[k] for k in data.files}
    return out


def evaluate_command(run_dir: Path, *, python: str = sys.executable) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_binding_readout", "evaluate", "--run", str(run_dir), "--overwrite"]


def queue_stage(stage: str, *, priority: int = 55, models: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
                root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None, dry_run: bool = False) -> list[dict[str, Any]]:
    """One GPU-lane evaluation per config of `stage` whose channel has a readout (named `<stage>-<stem>-readout`)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import _config_host, stage_python, stem_model
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        config = yaml.safe_load(path.read_text())
        model = stem_model(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if not (config.get("channel") or {}).get("readout"):
            continue
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        jobs.append({"name": f"{stage}-{path.stem}-{OUTPUT}", "priority": int(priority), "model": model,
                     "command": evaluate_command(Path(root) / "runs" / stage / path.stem, python=python)})
    if dry_run:
        return jobs
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"], min_free_gb=5,
                env={"PYTHONPATH": "src"}, resume_args=[])
            queued.append(job)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="readout on / off losses and readout diagnostics of one run")
    ev.add_argument("--run", type=Path, required=True); ev.add_argument("--output", type=Path, default=None)
    ev.add_argument("--device", default=None); ev.add_argument("--windows", type=int, default=None, help="smoke tests only")
    ev.add_argument("--fillers", default="auto"); ev.add_argument("--overwrite", action="store_true")
    queue = sub.add_parser("queue", help="queue one evaluation per readout config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--priority", type=int, default=55)
    queue.add_argument("--models", nargs="*", default=None); queue.add_argument("--seeds", type=int, nargs="*", default=None)
    queue.add_argument("--root", type=Path, default=ROOT); queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        record = evaluate_run(args.run, args.output, device=args.device, windows=args.windows,
                              fillers=None if args.fillers == "none" else args.fillers, overwrite=args.overwrite)
        print(json.dumps({"run": record["stem"], "seconds": record["seconds"], "replay": (record.get("ref_check") or {}).get("replay_ok")}))
    else:
        jobs = queue_stage(args.stage, priority=args.priority, models=args.models, seeds=args.seeds, root=args.root, dry_run=args.dry_run)
        print(json.dumps([{"name": j["name"], "priority": j["priority"], "command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
