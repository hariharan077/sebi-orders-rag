from __future__ import annotations

import unittest

from app.sebi_orders_rag.retrieval.filters import (
    build_shared_filter_clauses,
    normalize_metadata_filters,
)
from app.sebi_orders_rag.schemas import MetadataFilterInput


class RetrievalFilterTests(unittest.TestCase):
    def test_normalize_metadata_filters_deduplicates_and_trims(self) -> None:
        filters = normalize_metadata_filters(
            MetadataFilterInput(
                record_key="  external:100725  ",
                bucket_name=" orders-of-aa-under-rti-act ",
                document_version_ids=(5, 5, 2),
                section_keys=(" sec-2 ", "sec-2", ""),
                section_types=(" operative_order ", "operative_order"),
            )
        )

        self.assertEqual(filters.record_key, "external:100725")
        self.assertEqual(filters.bucket_name, "orders-of-aa-under-rti-act")
        self.assertEqual(filters.document_version_ids, (2, 5))
        self.assertEqual(filters.section_keys, ("sec-2",))
        self.assertEqual(filters.section_types, ("operative_order",))

    def test_build_shared_filter_clauses_uses_expected_aliases(self) -> None:
        clauses, params = build_shared_filter_clauses(
            MetadataFilterInput(
                record_key="external:100725",
                bucket_name="orders-of-aa-under-rti-act",
                document_version_ids=(7,),
                section_keys=("section-0002-operative_order-order",),
                section_types=("operative_order",),
            ),
            source_alias="src",
            version_alias="ver",
            chunk_alias="chk",
        )

        self.assertEqual(
            clauses,
            [
                "src.record_key = %s",
                "src.bucket_name = %s",
                "ver.document_version_id = ANY(%s)",
                "chk.section_key = ANY(%s)",
                "chk.section_type = ANY(%s)",
            ],
        )
        self.assertEqual(
            params,
            [
                "external:100725",
                "orders-of-aa-under-rti-act",
                [7],
                ["section-0002-operative_order-order"],
                ["operative_order"],
            ],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
