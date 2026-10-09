import json
import sys
from pathlib import Path

from vsa_embed.jobqueue import add, jobs, lane_of, recover, run_next


def test_priority_order_logging_and_outcomes(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    marker = tmp_path / "order.txt"
    for name, priority in (("late", 20), ("early", 10)):
        add(queue, [sys.executable, "-c", f"open({str(marker)!r}, 'a').write('{name}\\n')"], name=name,
            priority=priority, min_free_gb=0)
    add(queue, [sys.executable, "-c", "raise SystemExit(3)"], name="bad", priority=30, min_free_gb=0)
    for _ in range(3):
        run_next(queue)
    assert marker.read_text().split() == ["early", "late"]
    status = {j["name"]: j for j in jobs(queue)}
    assert status["early"]["status"] == "done" and status["bad"]["status"] == "failed"
    assert status["bad"]["returncode"] == 3 and Path(status["early"]["log"]).is_file()


def test_dead_running_job_is_requeued_with_resume_args(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    out = tmp_path / "args.txt"
    path = add(queue, [sys.executable, "-c", f"import sys; open({str(out)!r}, 'w').write(' '.join(sys.argv[1:]))"],
               name="train", min_free_gb=0)
    job = json.loads(path.read_text()); job.update(status="running", pid=999999999, attempts=1)
    path.write_text(json.dumps(job))
    assert recover(queue) == ["train"]
    run_next(queue)
    assert out.read_text() == "--resume"


def test_disk_space_blocks_a_job(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    add(queue, [sys.executable, "-c", "pass"], name="big", min_free_gb=10**9)
    assert run_next(queue)["status"] == "blocked"


def test_lanes_keep_cpu_jobs_off_the_gpu_runner_and_wait_for_inputs(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    marker = tmp_path / "order.txt"
    def job(name: str, priority: int, **kwargs) -> None:
        add(queue, [sys.executable, "-c", f"open({str(marker)!r}, 'a').write('{name}\\n')"], name=name,
            priority=priority, min_free_gb=0, **kwargs)
    job("train", 10); job("t5-report", 20); job("train2", 30); job("flagged", 40, lane="cpu")
    assert lane_of(jobs(queue)[1], "-report") == "cpu" and lane_of(jobs(queue)[1]) == "gpu"
    assert run_next(queue, lane="cpu", cpu_pattern="-report") is None  # its input "train" is not done
    assert run_next(queue, lane="gpu", cpu_pattern="-report")["name"] == "train"
    assert run_next(queue, lane="gpu", cpu_pattern="-report")["name"] == "train2"  # report skipped
    assert run_next(queue, lane="gpu", cpu_pattern="-report") is None
    assert run_next(queue, lane="cpu", cpu_pattern="-report")["name"] == "t5-report"
    assert run_next(queue, lane="cpu", cpu_pattern="-report")["name"] == "flagged"
    assert marker.read_text().split() == ["train", "train2", "t5-report", "flagged"]


def test_cpu_lane_waits_for_a_lower_priority_gpu_job(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    add(queue, [sys.executable, "-c", "pass"], name="train", priority=10, min_free_gb=0)
    add(queue, [sys.executable, "-c", "pass"], name="report", priority=20, min_free_gb=0, lane="cpu")
    assert run_next(queue, lane="cpu") is None
    assert run_next(queue)["name"] == "train"  # default lane "all" = the original runner
    assert run_next(queue, lane="cpu")["name"] == "report"


def test_a_blocked_job_is_retried_first_and_later_jobs_do_not_jump_it(tmp_path: Path) -> None:
    queue = tmp_path / "q"
    add(queue, [sys.executable, "-c", "pass"], name="train", priority=10, min_free_gb=10**9)
    add(queue, [sys.executable, "-c", "pass"], name="eval", priority=20, min_free_gb=0)
    assert run_next(queue, lane="gpu")["status"] == "blocked"
    assert run_next(queue, lane="gpu")["name"] == "train"            # retried, not skipped
    path = queue / "train.json"
    job = json.loads(path.read_text()); job["min_free_gb"] = 0; path.write_text(json.dumps(job))
    assert run_next(queue, lane="gpu")["name"] == "train"            # disk freed: it runs first
    assert run_next(queue, lane="gpu")["name"] == "eval"


def test_a_second_runner_waits_for_the_live_job_of_its_lane(tmp_path: Path) -> None:
    import os
    queue = tmp_path / "q"
    path = add(queue, [sys.executable, "-c", "pass"], name="train", priority=10, min_free_gb=0)
    add(queue, [sys.executable, "-c", "pass"], name="next", priority=20, min_free_gb=0)
    add(queue, [sys.executable, "-c", "pass"], name="t5-report", priority=30, min_free_gb=0, lane="cpu")
    job = json.loads(path.read_text()); job.update(status="running", pid=os.getpid()); path.write_text(json.dumps(job))
    assert run_next(queue, lane="gpu") is None                       # live job in the gpu lane
    assert run_next(queue) is None                                   # the original runner waits too
    job.update(pid=999999999); path.write_text(json.dumps(job))      # its process is gone: recovered and resumed
    assert run_next(queue, lane="gpu")["name"] == "train"


def test_a_job_whose_runner_is_alive_is_not_recovered_after_its_process_exits(tmp_path: Path) -> None:
    import os
    queue = tmp_path / "q"
    path = add(queue, [sys.executable, "-c", "pass"], name="train", priority=10, min_free_gb=0)
    job = json.loads(path.read_text()); job.update(status="running", pid=999999999, runner_pid=os.getpid())
    path.write_text(json.dumps(job))
    assert recover(queue) == [] and run_next(queue) is None          # the runner is about to record the outcome
    job.update(runner_pid=999999998); path.write_text(json.dumps(job))
    assert recover(queue) == ["train"]


def test_the_cli_takes_fractional_priorities_and_keeps_integral_ones_integers() -> None:
    from vsa_embed.jobqueue import _priority
    assert _priority("54.498") == 54.498 and isinstance(_priority("54"), int) and isinstance(_priority("54.0"), int)


def test_cancel_withdraws_an_unrun_job_and_unblocks_the_cpu_lane(tmp_path: Path) -> None:
    from vsa_embed.jobqueue import cancel
    import pytest
    queue = tmp_path / "q"
    add(queue, [sys.executable, "-c", "pass"], name="dropped", priority=10, min_free_gb=0)
    add(queue, [sys.executable, "-c", "pass"], name="ran", priority=5, min_free_gb=0)
    add(queue, [sys.executable, "-c", "pass"], name="x-report", priority=20, min_free_gb=0)
    run_next(queue, lane="gpu", cpu_pattern="-report")                 # "ran" (priority 5)
    with pytest.raises(ValueError):                                    # a finished job stays
        cancel(queue, "ran", reason="test")
    target = cancel(queue, "dropped", reason="not needed")
    assert target == queue / "cancelled" / "dropped.json" and not (queue / "dropped.json").exists()
    assert json.loads(target.read_text())["cancelled"]["reason"] == "not needed"
    assert [j["name"] for j in jobs(queue)] == ["ran", "x-report"]
    assert run_next(queue, lane="cpu", cpu_pattern="-report")["name"] == "x-report"   # no longer held back
