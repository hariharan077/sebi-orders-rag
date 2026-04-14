from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sebi_eval_release_gate.py"


class ReleaseGateTests(unittest.TestCase):
    def test_release_gate_passes_when_all_thresholds_are_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_release_gate_fixture(run_dir, true_bug_count=0)

            completed = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(run_dir)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((run_dir / "release_gate.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])

    def test_release_gate_fails_when_true_bug_threshold_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_release_gate_fixture(run_dir, true_bug_count=2)

            completed = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(run_dir)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1, completed.stdout)
            payload = json.loads((run_dir / "release_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            failed_check = next(check for check in payload["checks"] if check["metric"] == "true_bug_count")
            self.assertFalse(failed_check["passed"])


def _write_release_gate_fixture(run_dir: Path, *, true_bug_count: int) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "contamination_count": 0,
                "wrong_answer_regression_pass_count": 1,
                "wrong_answer_regression_total": 1,
                "structured_current_info_accuracy": 1.0,
                "numeric_fact_accuracy": 1.0,
                "equivalent_route_accuracy": 1.0,
                "candidate_list_correctness": 1.0,
                "true_bug_count": true_bug_count,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "per_case_results.jsonl").write_text("", encoding="utf-8")
    (run_dir / "true_bug_queue.json").write_text(
        json.dumps(
            {
                "summary": {
                    "true_bug_count": true_bug_count,
                }
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
