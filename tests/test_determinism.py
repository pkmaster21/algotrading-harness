"""Acceptance: two identical runs produce identical outputs."""

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tests.determinism_run import run

REPO = Path(__file__).resolve().parent.parent


def test_two_runs_in_one_process_are_identical() -> None:
    first, second = run(), run()
    assert len(first.fills) > 100
    forced = Counter(f.fill_date for f in first.fills if f.kind == "forced_close")
    assert max(forced.values()) >= 3  # several symbols force-closed on the same session
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_runs_are_identical_across_hash_seeds() -> None:
    outputs = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-m", "tests.determinism_run"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        outputs.add(proc.stdout)
    assert len(outputs) == 1
    assert run().model_dump_json() + "\n" in outputs
