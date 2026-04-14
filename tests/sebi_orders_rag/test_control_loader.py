from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import load_control_pack
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class ControlLoaderTests(unittest.TestCase):
    def test_loads_real_control_pack_with_expected_counts(self) -> None:
        pack = load_control_pack(REAL_CONTROL_PACK)

        self.assertIsNotNone(pack)
        assert pack is not None
        self.assertEqual(len(pack.document_index), 235)
        self.assertEqual(len(pack.confusion_pairs), 25)
        self.assertEqual(len(pack.eval_queries), 71)
        self.assertEqual(len(pack.wrong_answer_examples), 11)
        self.assertEqual(len(pack.entity_aliases), 297)
        self.assertTrue(pack.strict_answer_rule.strict_single_matter_required)
        self.assertIn(
            "answer from that matter only",
            pack.strict_answer_rule.text.lower(),
        )

    def test_indexes_record_keys_aliases_and_confusion_pairs(self) -> None:
        pack = load_control_pack(REAL_CONTROL_PACK)
        assert pack is not None

        self.assertIn(
            "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1",
            pack.documents_by_record_key,
        )
        self.assertIn("adani green energy", pack.alias_variants)
        self.assertIn(
            "external:98714",
            {
                pair.record_key_b
                for pair in pack.confusion_map[
                    "derived:58a47f2f3ab8169bffd6d73800717916db1fee7c0b8ee6259945c332c2e066a1"
                ]
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
