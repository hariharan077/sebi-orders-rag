from __future__ import annotations

import unittest

from app.sebi_orders_rag.control import load_control_pack, resolve_strict_matter_lock
from .fixture_paths import CONTROL_PACK_FIXTURE_ROOT

REAL_CONTROL_PACK = CONTROL_PACK_FIXTURE_ROOT


class EntityAliasResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = load_control_pack(REAL_CONTROL_PACK)
        assert cls.pack is not None

    def test_resolves_wealthmax_with_shortened_entity_variant(self) -> None:
        lock = resolve_strict_matter_lock(
            query="wealthmax solution investment advisor",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(lock.locked_record_keys, ("external:100851",))

    def test_resolves_person_name_with_trailing_initial_variant(self) -> None:
        lock = resolve_strict_matter_lock(
            query="Paresh Nathalal Chauhan A",
            control_pack=self.pack,
        )

        self.assertTrue(lock.strict_scope_required)
        self.assertTrue(lock.strict_single_matter)
        self.assertEqual(lock.locked_record_keys, ("external:98776",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
