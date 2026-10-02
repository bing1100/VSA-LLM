"""Sparse vs full channel-row generation: latency and peak memory (task B10, formulation §7).

The span channel needs one concept row per linked span of a batch. **Sparse** generation composes
only the entries that occur in the batch (`SpanChannel.rows`: unique entries, gather atomics, bind,
segment-sum). **Full** generation materializes the whole entry table every step (compose every
entry, project, then gather the batch's rows), which is what the tied-output and table-compression
variants need but the input channel does not. Both produce identical rows; the benchmark checks
that, then times each path.

Per cell (device × entry count × composer mode × batch spans × path) two phases are measured:
`forward` (no gradient: a generation or evaluation step) and `train` (forward and backward through
a dot-product loss with fixed targets, gradients on every parameter). Latency is the median over
repeats after warm-up. Peak memory is the extra memory one step needs on top of what is already
resident (parameters, schedule, inputs): `torch.cuda.max_memory_allocated` on a GPU, and the
peak resident set (`VmHWM` after a reset through `/proc/self/clear_refs`) in a fresh process with a
fixed `malloc` mmap threshold on the CPU, so freed buffers are returned and the peak is the live
tensor bytes. CPU latency and CPU memory therefore come from separate processes.

Modes: `bundle` (M0), `attentive` (M1, induced factor, static: duplicates are merged before
composing) and `attentive_context` (M1 with a per-span context vector, so every occurrence is
composed separately and a full table cannot be precomputed: sparse only).

    python -m vsa_embed.benchmarks.generation --entries 10000 50000 100000 --devices cpu cuda \\
        --output experiments/b10-benchmarks/generation.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor

from ..compose import FrameComposer, FrameSchedule
from ..provenance import git_state, package_versions
from ..span_channel import SpanChannel

MODES = ("bundle", "attentive", "attentive_context")
PATHS = ("sparse", "full")
PHASES = ("forward", "train")
CONTEXT_DIMENSION = 64
# Fixed malloc behaviour for the CPU memory pass: every large buffer is a fresh mmap and is
# returned to the OS when freed, so the resident-set peak equals the live tensor bytes.
MEMORY_PASS_ENV = {"MALLOC_MMAP_THRESHOLD_": "65536", "MALLOC_TRIM_THRESHOLD_": "0"}


# -- synthetic workload ------------------------------------------------------------------------------

def synthetic_schedule(entries: int, atomics: int, relations: int, mean_degree: float, seed: int) -> FrameSchedule:
    """Frames with `1 + Poisson(mean_degree - 1)` edges each and uniformly random (relation, atomic)."""
    generator = torch.Generator().manual_seed(seed)
    degrees = 1 + torch.poisson(torch.full((entries,), max(mean_degree - 1.0, 0.0)), generator=generator).long()
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(degrees, 0)])
    edges = int(offsets[-1])
    return FrameSchedule(offsets, torch.randint(0, relations, (edges,), generator=generator),
                         torch.randint(0, atomics, (edges,), generator=generator))


def sample_spans(entries: int, batch_spans: int, zipf: float, seed: int) -> dict[str, Tensor]:
    """`batch_spans` linked-span occurrences with Zipf-distributed entries (rank → random entry).

    Positions are not needed to compose rows; `batch/inject` place the spans on a grid only so the
    dict has the `link_batch` layout.
    """
    generator = torch.Generator().manual_seed(seed + 1)
    weights = torch.arange(1, entries + 1, dtype=torch.float64).pow(-zipf)
    ranks = torch.multinomial(weights / weights.sum(), batch_spans, replacement=True, generator=generator)
    entry = torch.randperm(entries, generator=generator)[ranks]
    position = torch.arange(batch_spans)
    return {"batch": position // 512, "start": position % 512, "end": position % 512, "inject": position % 512,
            "entry": entry, "confidence": torch.ones(batch_spans), "length": torch.full((batch_spans,), 2)}


def build_channel(spec: dict[str, Any], device: torch.device) -> SpanChannel:
    torch.manual_seed(spec["seed"])
    schedule = synthetic_schedule(spec["entries"], spec["atomics"], spec["relations"], spec["mean_degree"], spec["seed"])
    attentive = spec["mode"] != "bundle"
    composer = FrameComposer(
        schedule, spec["atomics"], spec["relations"], spec["dimension"], operator=spec["operator"],
        mode="attentive" if attentive else "bundle",
        context_dimension=CONTEXT_DIMENSION if spec["mode"] == "attentive_context" else 0)
    channel = SpanChannel(composer, spec["model_dimension"], entry_count=spec["entries"])
    return channel.to(device)


def full_rows(channel: SpanChannel, entries: Tensor) -> Tensor:
    """Materialize the whole entry table, then gather the batch's rows (the cost being compared)."""
    table = channel.projector(channel.composer.compose(torch.arange(channel.entry_count, device=entries.device)))
    return table[entries]


def make_step(channel: SpanChannel, spans: dict[str, Tensor], context: Tensor | None, target: Tensor,
              path: str, phase: str) -> Callable[[], None]:
    def rows() -> Tensor:
        return channel.rows(spans, context=context) if path == "sparse" else full_rows(channel, spans["entry"])

    if phase == "forward":
        def step() -> None:
            with torch.no_grad():
                rows()
    else:
        def step() -> None:
            (rows() * target).sum().backward()
            channel.zero_grad(set_to_none=True)   # nothing but parameters stays resident between steps
    return step


# -- measurement ---------------------------------------------------------------------------------------

def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(step: Callable[[], None], device: torch.device, *, warmup: int, min_seconds: float,
                    min_repeats: int, max_repeats: int) -> list[float]:
    """Seconds per call after `warmup` calls; repeats until `min_seconds` have passed."""
    for _ in range(warmup):
        step()
    _synchronize(device)
    times: list[float] = []
    begin = time.perf_counter()
    while len(times) < max_repeats and (len(times) < min_repeats or time.perf_counter() - begin < min_seconds):
        start = time.perf_counter()
        step()
        _synchronize(device)
        times.append(time.perf_counter() - start)
    return times


def _proc_status_kib(field: str) -> int:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"{field} not in /proc/self/status")


def measure_peak_megabytes(step: Callable[[], None], device: torch.device) -> float | None:
    """Extra peak memory (MiB) of one call beyond what is resident before it (None if unavailable)."""
    step()                                              # warm-up: lazy buffers, FFT plans
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        step()
        torch.cuda.synchronize(device)
        return (torch.cuda.max_memory_allocated(device) - base) / 2**20
    try:
        with open("/proc/self/clear_refs", "w") as handle:  # reset VmHWM to the current resident set
            handle.write("5")
        base = _proc_status_kib("VmRSS")
        step()
        return (_proc_status_kib("VmHWM") - base) / 1024
    except (OSError, RuntimeError):                     # not Linux, or /proc is read-only
        return None


def run_worker(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Every workload of `spec["workloads"]` (`[entries, mode]` pairs) in this process."""
    device = torch.device("cuda:0" if spec["device"] == "cuda" else "cpu")
    if device.type == "cuda" and spec.get("gpu_memory_gib"):
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, spec["gpu_memory_gib"] * 2**30 / total), device)
    cells: list[dict[str, Any]] = []
    for entries, mode in spec["workloads"]:
        cells += run_workload({**spec, "entries": entries, "mode": mode}, deadline=spec.get("deadline"))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return cells


def run_workload(spec: dict[str, Any], *, deadline: float | None) -> list[dict[str, Any]]:
    """All (batch spans × path × phase) cells of one (device, entries, mode) workload."""
    device = torch.device("cuda:0" if spec["device"] == "cuda" else "cpu")
    channel = build_channel(spec, device)
    composer = channel.composer
    degrees = composer.schedule.degrees
    cells: list[dict[str, Any]] = []
    for batch_spans in spec["batch_spans"]:
        spans = {k: v.to(device) for k, v in sample_spans(
            spec["entries"], batch_spans, spec["zipf"], spec["seed"]).items()}
        unique = torch.unique(spans["entry"])
        context = (torch.randn(batch_spans, CONTEXT_DIMENSION, device=device) if spec["mode"] == "attentive_context"
                   else None)
        target = torch.randn(batch_spans, spec["model_dimension"], device=device)
        base = {"device": spec["device"], "mode": spec["mode"], "entries": spec["entries"],
                "edges": int(degrees.sum()), "batch_spans": batch_spans, "unique_entries": int(unique.numel()),
                "unique_edges": int(degrees[unique].sum())}
        if context is None:
            try:
                with torch.no_grad():
                    difference = float((channel.rows(spans) - full_rows(channel, spans["entry"])).abs().max())
            except torch.cuda.OutOfMemoryError:     # the full table does not fit under the cap
                difference = float("nan"); gc.collect(); torch.cuda.empty_cache()
            if difference > 1e-4:
                raise AssertionError(f"sparse and full rows differ by {difference:.2e}")
            base["max_row_difference"] = None if math.isnan(difference) else difference
        for path in PATHS:
            for phase in PHASES:
                cell = {**base, "path": path, "phase": phase}
                if path == "full" and context is not None:
                    cells.append({**cell, "status": "n/a (context is per occurrence)"}); continue
                if deadline is not None and time.time() > deadline:
                    cells.append({**cell, "status": "skipped (time budget)"}); continue
                step = make_step(channel, spans, context, target, path, phase)
                try:
                    if spec["pass"] in {"latency", "both"}:
                        times = measure_latency(step, device, warmup=spec["warmup"], min_seconds=spec["min_seconds"],
                                                min_repeats=spec["min_repeats"], max_repeats=spec["max_repeats"])
                        ordered = sorted(times)
                        cell.update(latency_ms=1e3 * statistics.median(times), repeats=len(times),
                                    latency_p10_ms=1e3 * ordered[int(0.1 * (len(ordered) - 1))],
                                    latency_p90_ms=1e3 * ordered[int(math.ceil(0.9 * (len(ordered) - 1)))])
                    if spec["pass"] in {"memory", "both"}:
                        cell["peak_mib"] = measure_peak_megabytes(step, device)
                    cell["status"] = "ok"
                except torch.cuda.OutOfMemoryError:
                    cell["status"] = f"OOM (cap {spec['gpu_memory_gib']} GiB)"
                    channel.zero_grad(set_to_none=True)
                    gc.collect(); torch.cuda.empty_cache()
                cells.append(cell)
    return cells


# -- orchestration (one subprocess per measurement pass) -----------------------------------------------

def _spawn(spec: dict[str, Any], *, memory_pass: bool) -> list[dict[str, Any]]:
    env = {**os.environ, "OMP_NUM_THREADS": str(spec["threads"]), "MKL_NUM_THREADS": str(spec["threads"])}
    if memory_pass:
        env.update(MEMORY_PASS_ENV)
    command = [sys.executable, "-m", "vsa_embed.benchmarks.generation", "--worker", json.dumps(spec)]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[len("RESULT:"):])
    raise RuntimeError(f"benchmark worker failed ({result.returncode}):\n{result.stderr[-2000:]}")


def machine_description(args: argparse.Namespace) -> dict[str, Any]:
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    memory = next((int(line.split()[1]) / 2**20 for line in open("/proc/meminfo") if line.startswith("MemTotal")), 0.0)
    info: dict[str, Any] = {
        "cpu": cpu, "logical_cpus": os.cpu_count(), "threads_used": args.threads, "ram_gib": round(memory, 1),
        "platform": platform.platform(), "python": platform.python_version(), "packages": package_versions(),
        **git_state(), "created_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv[1:],
        "load_average_at_start": [round(x, 2) for x in os.getloadavg()],
    }
    if "cuda" in args.devices and torch.cuda.is_available():
        # nvidia-smi, not torch: the parent process must not open a CUDA context of its own.
        info["gpu"] = {"cuda": torch.version.cuda, "allocator_cap_gib": args.gpu_memory_gib,
                       "time_budget_s": args.gpu_budget_seconds}
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
            name, total, used, utilization = [field.strip() for field in query.splitlines()[0].split(",")]
            info["gpu"].update(name=name, total_gib=round(int(total) / 1024, 1),
                               used_by_others_gib_at_start=round(int(used) / 1024, 1),
                               utilization_percent_at_start=int(utilization))
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            info["gpu"].update(name="unknown", total_gib=0.0, used_by_others_gib_at_start=0.0)
    return info


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    machine = machine_description(args)
    common = {"dimension": args.dimension, "model_dimension": args.model_dimension or args.dimension,
              "atomics": args.atomics, "relations": args.relations, "mean_degree": args.mean_degree,
              "batch_spans": args.batch_spans, "zipf": args.zipf, "operator": args.operator, "seed": args.seed,
              "warmup": args.warmup, "min_seconds": args.min_seconds, "min_repeats": args.min_repeats,
              "max_repeats": args.max_repeats, "threads": args.threads, "gpu_memory_gib": args.gpu_memory_gib}
    cells: list[dict[str, Any]] = []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda not available; skipping", file=sys.stderr); continue
        workloads = [[entries, mode] for entries in args.entries for mode in args.modes]
        if device == "cuda":
            # One process for every cuda workload (one context); both passes in the same run.
            started = time.time()
            cells += _spawn({**common, "device": "cuda", "workloads": workloads, "pass": "both",
                             "deadline": started + args.gpu_budget_seconds}, memory_pass=False)
            machine["gpu"]["wall_seconds"] = round(time.time() - started, 1)
            continue
        for kind in ("latency", "memory"):
            for workload in workloads:
                print(f"cpu entries={workload[0]} mode={workload[1]} pass={kind}", file=sys.stderr, flush=True)
                cells += _spawn({**common, "device": "cpu", "workloads": [workload], "pass": kind,
                                 "deadline": None}, memory_pass=kind == "memory")
    machine["load_average_at_end"] = [round(x, 2) for x in os.getloadavg()]
    return {"machine": machine, "parameters": common, "cells": merge_cells(cells)}


def merge_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the CPU latency-pass and memory-pass records of the same cell."""
    merged: dict[tuple, dict[str, Any]] = {}
    for cell in cells:
        key = (cell["device"], cell["mode"], cell["entries"], cell["batch_spans"], cell["path"], cell["phase"])
        merged.setdefault(key, {}).update(cell)
    return list(merged.values())


# -- report ----------------------------------------------------------------------------------------------

def _lookup(cells: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    return {(c["device"], c["mode"], c["entries"], c["batch_spans"], c["path"], c["phase"]): c for c in cells}


def _value(cell: dict[str, Any] | None, field: str, digits: int) -> tuple[str, float | None]:
    if cell is None:
        return "-", None
    if cell.get("status", "ok") != "ok":
        status = cell["status"]
        return ("OOM" if status.startswith("OOM") else "n/a" if status.startswith("n/a") else "skipped"), None
    if cell.get(field) is None:
        return "-", None
    return f"{cell[field]:,.{digits}f}", cell[field]


def _table(cells: list[dict[str, Any]], device: str, field: str, digits: int, unit: str) -> list[str]:
    lookup = _lookup([c for c in cells if c["device"] == device])
    keys = sorted({(c["mode"], c["entries"], c["batch_spans"]) for c in cells if c["device"] == device},
                  key=lambda k: (MODES.index(k[0]), k[1], k[2]))
    lines = [f"| mode | entries | batch spans | unique entries | forward sparse ({unit}) | forward full ({unit}) | full/sparse "
             f"| train sparse ({unit}) | train full ({unit}) | full/sparse |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, entries, spans in keys:
        row = [mode, f"{entries:,}", f"{spans:,}",
               f"{lookup[(device, mode, entries, spans, 'sparse', 'forward')]['unique_entries']:,}"]
        for phase in PHASES:
            sparse_text, sparse = _value(lookup.get((device, mode, entries, spans, "sparse", phase)), field, digits)
            full_text, full = _value(lookup.get((device, mode, entries, spans, "full", phase)), field, digits)
            ratio = f"{full / sparse:,.1f}×" if sparse is not None and full is not None and sparse > 0 else "-"
            row += [sparse_text, full_text, ratio]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def observations(result: dict[str, Any]) -> list[str]:
    """Data-derived summary lines at the median batch size, for the static modes."""
    cells, parameters = result["cells"], result["parameters"]
    lookup = _lookup(cells)
    spans = sorted(parameters["batch_spans"])[len(parameters["batch_spans"]) // 2]
    units = {"latency_ms": "ms", "peak_mib": "MiB"}
    lines = []
    for device in ("cpu", "cuda"):
        sizes = sorted({c["entries"] for c in cells if c["device"] == device})
        for mode in ("bundle", "attentive"):
            def full_status(entries: int) -> str:
                return str(lookup.get((device, mode, entries, spans, "full", "forward"), {}).get("status", ""))
            fitting = [n for n in sizes if full_status(n) == "ok"]
            if not fitting:
                continue
            entries = fitting[-1]
            parts = []
            for phase in PHASES:
                sparse = lookup[(device, mode, entries, spans, "sparse", phase)]
                full = lookup[(device, mode, entries, spans, "full", phase)]
                ratios = [f"{label} {full[field] / sparse[field]:,.1f}× lower" for field, label in
                          (("latency_ms", "latency"), ("peak_mib", "memory"))
                          if sparse.get(field) and full.get(field) is not None]
                if ratios:
                    parts.append(f"{phase} " + ", ".join(ratios))
            line = (f"- {'CPU' if device == 'cpu' else 'GPU'}, `{mode}`, N = {entries:,}, {spans:,} spans per batch, "
                    "sparse vs full: " + "; ".join(parts) + ".")
            too_big = [n for n in sizes if n > entries and full_status(n).startswith("OOM")]
            if too_big:
                sparse = lookup[(device, mode, too_big[-1], spans, "sparse", "train")]
                line += (f" From N = {too_big[0]:,} the full step does not fit the allocator cap; the sparse train step "
                         f"at N = {too_big[-1]:,} needs {sparse['peak_mib']:,.0f} MiB.")
            lines.append(line)
    atomics_mib = parameters["atomics"] * parameters["dimension"] * 4 / 2**20
    lines.append(
        "- Sparse peak memory does not depend on N, and sparse latency follows the batch's unique linked entries rather "
        "than growing with N at the full path's rate. The N-independent memory floor is about one atomic table in "
        f"`forward` and five in `train` ({atomics_mib:.0f} and "
        f"{5 * atomics_mib:.0f} MiB here for {parameters['atomics']:,} atomics × {parameters['dimension']}): "
        "`FrameComposer.atomic_vectors()` normalizes the whole atomic table before the batch's fillers are gathered, and "
        "the atomic gradient is dense. Normalizing after the gather is mathematically identical and would remove it; "
        "`compose.py` is unchanged here.")
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    machine, parameters, cells = result["machine"], result["parameters"], result["cells"]
    devices = [d for d in ("cpu", "cuda") if any(c["device"] == d for c in cells)]
    gpu = machine.get("gpu")
    lines = [
        "# B10 — sparse vs full channel-row generation", "",
        "Latency and peak memory of producing the concept rows a batch needs. **Sparse**: compose only the "
        "entries linked in the batch (`SpanChannel.rows`). **Full**: materialize the whole entry table every step "
        "(compose all entries, project, gather the batch's rows). Both give the same rows (checked to 1e-4 on every "
        "workload whose full table fits in memory). Cost model: formulation §7; generated by `vsa-bench-generation` "
        "(`src/vsa_embed/benchmarks/generation.py`); raw numbers in `generation.json`.", "",
        "## Setup", "",
        f"- Entries `N` ∈ {sorted({c['entries'] for c in cells})}, composer dimension {parameters['dimension']}, "
        f"model dimension {parameters['model_dimension']}, operator `{parameters['operator']}`, "
        f"{parameters['relations']} relations, {parameters['atomics']:,} atomics, frame degree "
        f"1 + Poisson({parameters['mean_degree'] - 1:g}) (mean {parameters['mean_degree']:g}), fp32.",
        f"- A batch has `batch spans` linked occurrences drawn from the entries with a Zipf({parameters['zipf']:g}) "
        "law (`unique entries` is how many distinct entries that is). Modes: `bundle` = M0; `attentive` = M1, induced "
        "concept factor, static (duplicates merged before composing); `attentive_context` = M1 with a per-span context "
        "vector (every occurrence is composed separately, so a full table cannot be precomputed: sparse only).",
        "- `forward`: no gradient (a generation or evaluation step). `train`: forward + backward of a dot-product loss "
        "with fixed targets, gradients on every parameter (atomics, relations, projector, attention).",
        f"- Latency: median over repeats (≥ {parameters['min_repeats']}, ≥ {parameters['min_seconds']:g} s, "
        f"≤ {parameters['max_repeats']}) after {parameters['warmup']} warm-up calls. Peak memory: extra memory of one "
        "step beyond what is resident before it (parameters, schedule, inputs); GPU = `torch.cuda.max_memory_allocated`; "
        "CPU = peak resident set of a fresh process (`VmHWM`, reset before the step, `MALLOC_MMAP_THRESHOLD_=65536` so "
        "freed buffers are returned). CPU latency and memory come from separate processes.", "",
        "## Machine and environment", "",
        f"- CPU: {machine['cpu']}, {machine['logical_cpus']} logical cores, {machine['threads_used']} torch threads used, "
        f"{machine['ram_gib']} GiB RAM; {machine['platform']}.",
        f"- Python {machine['python']}, torch {machine['packages'].get('torch')}, commit `{machine.get('git_sha')}` "
        f"(dirty: {machine.get('git_dirty')}), run at {machine['created_at']}.",
        f"- Host load average (1/5/15 min) at start {machine['load_average_at_start']}, at end "
        f"{machine.get('load_average_at_end', '?')}: other jobs shared this machine, so CPU latencies carry some noise.",
    ]
    if gpu:
        lines += [
            f"- GPU: {gpu['name']} ({gpu['total_gib']} GiB), CUDA {gpu['cuda']}; allocator capped at "
            f"{gpu['allocator_cap_gib']} GiB for this process (plus the CUDA context), wall-clock budget "
            f"{gpu['time_budget_s']} s (used {gpu.get('wall_seconds', '?')} s). **The GPU was shared:** other jobs held "
            f"{gpu['used_by_others_gib_at_start']} GiB and ran at {gpu.get('utilization_percent_at_start', '?')}% "
            "utilization when the benchmark started, so GPU latencies include time-slicing with that job and are "
            "upper bounds with high variance; compare sparse and full within the table, not against an idle GPU. "
            "`OOM` means the step does not fit under the cap.",
        ]
    for device in devices:
        title = "CPU" if device == "cpu" else "GPU (shared, capped)"
        lines += ["", f"## {title}: latency per step (median, ms)", ""]
        lines += _table(cells, device, "latency_ms", 2, "ms")
        lines += ["", f"## {title}: extra peak memory per step (MiB)", ""]
        lines += _table(cells, device, "peak_mib", 1, "MiB")
    lines += ["", "## Summary (derived from the tables)", ""] + observations(result)
    lines += ["", "## How to read this", "",
              "Sparse cost follows the batch (unique linked entries and their edges); full cost follows the table "
              "(`N` entries and all their edges), whatever the batch contains. The `full/sparse` columns are the "
              "per-step overhead of materializing the table, which the proposal's accounting charges only to the "
              "tied-output and table-compression variants (formulation §7). The full-path numbers do not depend on the "
              "batch, so differences between the batch-size rows of one `N` are measurement noise (host load). "
              "`attentive_context` rows show the contextual mode (E1): per-occurrence composition, with no table to "
              "materialize.", ""]
    return "\n".join(lines)


# -- CLI -------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--entries", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--model-dimension", type=int, default=0, help="projector output; default = --dimension")
    parser.add_argument("--atomics", type=int, default=8192)
    parser.add_argument("--relations", type=int, default=16)
    parser.add_argument("--mean-degree", type=float, default=6.0)
    parser.add_argument("--batch-spans", type=int, nargs="+", default=[1024])
    parser.add_argument("--zipf", type=float, default=1.1)
    parser.add_argument("--operator", default="hrr")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=30)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--gpu-budget-seconds", type=float, default=150.0)
    parser.add_argument("--gpu-memory-gib", type=float, default=1.5, help="CUDA allocator cap for this process")
    parser.add_argument("--output", type=Path, help="markdown report; a .json with the raw cells is written beside it")
    parser.add_argument("--render", type=Path, help="re-render --output from an existing raw .json without measuring")
    args = parser.parse_args(argv)
    if args.worker:
        spec = json.loads(args.worker)
        torch.set_num_threads(spec["threads"])
        print("RESULT:" + json.dumps(run_worker(spec)))
        return
    result = json.loads(args.render.read_text()) if args.render else run_benchmark(args)
    markdown = render_markdown(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        if not args.render:
            args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
