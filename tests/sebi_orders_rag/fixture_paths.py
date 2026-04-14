from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
CONTROL_PACK_FIXTURE_ROOT = FIXTURES_ROOT / "control_pack"
EVAL_FAILURE_DUMP_FIXTURE_ROOT = FIXTURES_ROOT / "eval_failure_dump"
