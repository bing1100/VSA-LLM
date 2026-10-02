import json
from pathlib import Path

import pytest
import torch

from vsa_embed.benchmarks import generation as bench


def tiny_spec(**overrides) -> dict:
    spec = {"device": "cpu", "entries": 300, "mode": "bundle", "dimension": 16, "model_dimension": 16, "atomics": 64,
            "relations": 4, "mean_degree": 4.0, "batch_spans": [40], "zipf": 1.1, "operator": "hrr", "seed": 0,
            "warmup": 1, "min_seconds": 0.0, "min_repeats": 2, "max_repeats": 3, "threads": 1, "pass": "both",
            "gpu_memory_gib": 0}
    return {**spec, **overrides}


def test_synthetic_schedule_is_deterministic_and_valid() -> None:
    a = bench.synthetic_schedule(500, 64, 4, 6.0, seed=3)
    b = bench.synthetic_schedule(500, 64, 4, 6.0, seed=3)
    assert torch.equal(a.offsets, b.offsets) and torch.equal(a.fillers, b.fillers)
    assert a.concept_count == 500 and int(a.degrees.min()) >= 1
    assert abs(float(a.degrees.float().mean()) - 6.0) < 0.5
    assert int(a.fillers.max()) < 64 and int(a.relations.max()) < 4


def test_sampled_spans_are_zipfian_over_valid_entries() -> None:
    spans = bench.sample_spans(1000, 4000, 1.1, seed=1)
    entries = spans["entry"]
    assert entries.shape == (4000,) and int(entries.max()) < 1000
    counts = torch.bincount(entries, minlength=1000).sort(descending=True).values
    assert int(counts[0]) > 10 * int(counts[500:].float().mean().clamp_min(1))   # heavy head, long tail
    assert torch.unique(entries).numel() < 4000                                 # repeats exist


@pytest.mark.parametrize("mode", bench.MODES)
def test_sparse_and_full_paths_produce_the_same_rows(mode: str) -> None:
    channel = bench.build_channel(tiny_spec(mode=mode, entries=120), torch.device("cpu"))
    spans = bench.sample_spans(120, 50, 1.1, seed=0)
    with torch.no_grad():
        sparse = channel.rows(spans)
        full = bench.full_rows(channel, spans["entry"])
    if mode == "attentive_context":      # needs a context per occurrence, which the full table cannot have
        assert sparse.shape == (50, 16)
        return
    torch.testing.assert_close(sparse, full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", bench.MODES)
def test_worker_reports_every_cell(mode: str) -> None:
    cells = bench.run_worker({**tiny_spec(), "workloads": [[300, mode]]})
    assert {(c["path"], c["phase"]) for c in cells} == {(p, f) for p in bench.PATHS for f in bench.PHASES}
    for cell in cells:
        if cell["path"] == "full" and mode == "attentive_context":
            assert cell["status"].startswith("n/a")
            continue
        assert cell["status"] == "ok" and cell["latency_ms"] > 0 and cell["repeats"] >= 2
        assert cell["unique_entries"] <= 40 and cell["unique_edges"] <= cell["edges"]
        assert cell["peak_mib"] is None or cell["peak_mib"] >= 0
    sparse = next(c for c in cells if (c["path"], c["phase"]) == ("sparse", "forward"))
    assert sparse.get("max_row_difference", 0.0) <= 1e-4


def test_expired_deadline_skips_remaining_cells() -> None:
    cells = bench.run_worker({**tiny_spec(), "workloads": [[300, "bundle"]], "deadline": 0.0})
    assert all(c["status"].startswith("skipped") for c in cells)


def test_markdown_report_has_tables_environment_and_special_cells() -> None:
    def cell(entries: int, path: str, phase: str, latency: float | None, peak: float | None,
             status: str = "ok") -> dict:
        return {"device": "cpu", "mode": "bundle", "entries": entries, "edges": 6 * entries, "batch_spans": 64,
                "unique_entries": 30, "unique_edges": 180, "path": path, "phase": phase, "status": status,
                "latency_ms": latency, "peak_mib": peak}
    cells = [cell(1000, "sparse", "forward", 2.0, 5.0), cell(1000, "full", "forward", 20.0, 50.0),
             cell(1000, "sparse", "train", 4.0, 10.0), cell(1000, "full", "train", 40.0, 100.0)]
    for phase in bench.PHASES:     # a larger table whose full step does not fit under the cap
        cells += [cell(5000, "sparse", phase, 2.0, 5.0), cell(5000, "full", phase, None, None, "OOM (cap 1.5 GiB)")]
    result = {"machine": {"cpu": "test cpu", "logical_cpus": 4, "threads_used": 2, "ram_gib": 8.0, "platform": "linux",
                          "python": "3.12", "packages": {"torch": "2.x"}, "git_sha": "abc", "git_dirty": False,
                          "created_at": "now", "load_average_at_start": [0.1, 0.1, 0.1]},
              "parameters": {"dimension": 8, "model_dimension": 8, "operator": "hrr", "relations": 2, "atomics": 16,
                             "mean_degree": 6.0, "zipf": 1.1, "batch_spans": [64], "min_repeats": 2, "min_seconds": 0.1,
                             "max_repeats": 5, "warmup": 1},
              "cells": cells}
    markdown = bench.render_markdown(result)
    assert "## Machine and environment" in markdown and "test cpu" in markdown
    assert "10.0×" in markdown          # forward latency ratio
    assert "OOM" in markdown and "does not fit the allocator cap" in markdown


@pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc")
def test_cli_writes_markdown_and_raw_json(tmp_path: Path) -> None:
    output = tmp_path / "generation.md"
    bench.main(["--entries", "200", "--dimension", "16", "--atomics", "64", "--relations", "4", "--modes", "bundle",
                "--batch-spans", "32", "--min-seconds", "0", "--min-repeats", "2", "--max-repeats", "3",
                "--threads", "1", "--output", str(output)])
    assert "## CPU: latency per step" in output.read_text()
    raw = json.loads(output.with_suffix(".json").read_text())
    assert {c["path"] for c in raw["cells"]} == {"sparse", "full"}
    rerendered = tmp_path / "again.md"
    bench.main(["--render", str(output.with_suffix(".json")), "--output", str(rerendered)])
    assert rerendered.read_text() == output.read_text()
