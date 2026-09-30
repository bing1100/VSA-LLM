import json
import sys
from pathlib import Path

from vsa_embed.jobqueue import add, jobs, recover, run_next


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
