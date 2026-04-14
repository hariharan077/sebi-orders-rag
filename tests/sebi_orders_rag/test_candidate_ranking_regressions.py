from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import load_control_pack, resolve_strict_matter_lock
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class CandidateRankingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_dump_ranked_queries_now_lock_to_expected_single_record(self) -> None:
        cases = (
            (
                "Tell me more about Hardcastle and Waud Manufacturing Ltd.",
                "derived:3960cfc96e2c1274ff5cbefde13b3cdc6481c71c1456bf71ba8e8e56fa52411e",
            ),
            (
                "Tell me more about Prime Broking Company India Limited",
                "external:30189",
            ),
            (
                "Tell me more about Adani Green Energy Limited by Pranav Adani",
                "derived:551259f200f62065e076213d712072bed57ea9c610044f61b542791220f62c09",
            ),
            (
                "Tell me more about Kisley Plantation Limited",
                "external:88411",
            ),
            (
                "Tell me more about Neelgiri Forest Ltd",
                "external:87947",
            ),
            (
                "What IPO proceeds were raised in the Pacheli Industrial Finance Limited matter?",
                "derived:69093d449801258e05a31962b30bba4735b0524616cc6679f7eba25e3480fd46",
            ),
            (
                "What exemption did SEBI grant in the JP Morgan Chase Bank N.A. matter?",
                "external:100486",
            ),
        )

        for query, expected_record_key in cases:
            with self.subTest(query=query):
                lock = resolve_strict_matter_lock(
                    query=query,
                    control_pack=self.pack,
                )

                self.assertTrue(lock.strict_scope_required)
                self.assertTrue(lock.strict_single_matter)
                self.assertFalse(lock.ambiguous)
                self.assertEqual(lock.locked_record_keys, (expected_record_key,))
                self.assertIn("single_matter_lock", lock.reason_codes)

    def test_generic_cochin_query_does_not_lock_to_demutualisation_scheme_aliases(self) -> None:
        lock = resolve_strict_matter_lock(
            query="Tell me more about Cochin Stock Exchange Limited.",
            control_pack=self.pack,
        )

        self.assertFalse(lock.strict_scope_required)
        self.assertFalse(lock.ambiguous)
        self.assertEqual(lock.locked_record_keys, ())
        self.assertEqual(lock.matched_aliases, ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
