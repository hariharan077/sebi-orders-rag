from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import confusion_penalty_map, load_control_pack, resolve_strict_matter_lock
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class ExactEntityLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_locks_vishvaraj_query_to_one_record(self) -> None:
        lock = resolve_strict_matter_lock(
            query="Tell me more about the IPO of Vishvaraj Environment Limited",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(
            lock.locked_record_keys,
            ("derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",),
        )
        self.assertIn("single_matter_lock", lock.reason_codes)

    def test_explicit_comparison_disables_single_matter_lock(self) -> None:
        lock = resolve_strict_matter_lock(
            query=(
                "Compare the IPO of Vishvaraj Environment Limited and "
                "Varyaa Creations Limited"
            ),
            control_pack=self.pack,
        )

        self.assertTrue(lock.comparison_intent)
        self.assertFalse(lock.strict_scope_required)
        self.assertFalse(lock.strict_single_matter)

    def test_ambiguous_generic_title_does_not_force_single_matter_lock(self) -> None:
        lock = resolve_strict_matter_lock(
            query="Tell me more about Certain Investment Advisers",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertFalse(lock.strict_single_matter)
        self.assertTrue(lock.ambiguous)

    def test_confusion_penalty_marks_known_bad_pair(self) -> None:
        lock = resolve_strict_matter_lock(
            query="Tell me more about the IPO of Vishvaraj Environment Limited",
            control_pack=self.pack,
        )

        penalties = confusion_penalty_map(control_pack=self.pack, strict_lock=lock)

        self.assertIn("external:98714", penalties)
        self.assertLess(penalties["external:98714"], 0.1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
