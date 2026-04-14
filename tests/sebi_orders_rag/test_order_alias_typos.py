from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import load_control_pack, resolve_strict_matter_lock
from app.sebi_orders_rag.normalization.aliases import generate_order_alias_variants
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class OrderAliasTyposTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_paresh_nathanlal_typo_resolves_to_single_matter(self) -> None:
        lock = resolve_strict_matter_lock(
            query="What was the Paresh Nathanlal case",
            control_pack=self.pack,
            matter_reference_signals=("case",),
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(lock.locked_record_keys, ("external:98776",))
        self.assertIn("alias_match", lock.reason_codes)

    def test_wealthmax_short_form_still_resolves_safely(self) -> None:
        lock = resolve_strict_matter_lock(
            query="wealthmax solution investment advisor",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(lock.locked_record_keys, ("external:100851",))

    def test_versus_style_alias_generation_does_not_crash(self) -> None:
        variants = generate_order_alias_variants("tushar oil food ltd vs sebi")

        self.assertIn("tushar oil food ltd vs sebi", variants)
        self.assertIn("tushar oil food ltd", variants)
        self.assertIn("sebi vs tushar oil food ltd", variants)

    def test_tushar_vs_sebi_locks_to_single_sat_matter(self) -> None:
        lock = resolve_strict_matter_lock(
            query="summarise tushar oil food ltd vs sebi",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(lock.locked_record_keys, ("external:29938",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
