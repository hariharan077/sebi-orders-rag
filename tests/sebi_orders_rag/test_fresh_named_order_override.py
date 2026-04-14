from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from uuid import uuid4

from app.sebi_orders_rag.control.models import ControlPack, DocumentIndexRow, EntityAliasRow, StrictAnswerRule
from app.sebi_orders_rag.router.decision import AdaptiveQueryRouter
from app.sebi_orders_rag.schemas import ChatSessionStateRecord


class FreshNamedOrderOverrideTests(unittest.TestCase):
    def test_new_named_matter_overrides_stale_active_context(self) -> None:
        router = AdaptiveQueryRouter(control_pack=_build_control_pack())
        state = ChatSessionStateRecord(
            session_id=uuid4(),
            active_record_keys=("external:ganga-prasad-rti",),
            active_primary_title="Appeal No. 22 of 2023 filed by Ganga Prasad",
            active_primary_entity="Ganga Prasad",
        )

        decision = router.decide(
            query="what did sebi find in hemant ghai order?",
            session_state=state,
        )

        self.assertEqual(decision.route_mode, "exact_lookup")
        self.assertTrue(decision.analysis.fresh_query_override)
        self.assertFalse(decision.analysis.active_order_override)
        self.assertEqual(
            decision.analysis.strict_lock_record_keys,
            ("external:hemant-ghai-settlement",),
        )


def _build_control_pack() -> ControlPack:
    hemant_doc = DocumentIndexRow(
        record_key="external:hemant-ghai-settlement",
        exact_title="Settlement Order in the matter of Hemant Ghai",
        bucket_category="settlement-orders",
        order_date=date(2024, 1, 10),
        main_entities=("Hemant Ghai",),
        short_summary="Settlement order relating to Hemant Ghai.",
        summary_source="fixture",
        procedural_type="settlement",
        manifest_status="ingested",
        manifest_error=None,
        ingested=True,
        document_version_id=301,
        detail_url="https://example.com/hemant-ghai",
        pdf_url="https://example.com/hemant-ghai.pdf",
        local_filename="hemant_ghai.pdf",
    )
    ganga_doc = DocumentIndexRow(
        record_key="external:ganga-prasad-rti",
        exact_title="Appeal No. 22 of 2023 filed by Ganga Prasad",
        bucket_category="orders-of-aa-under-rti-act",
        order_date=date(2023, 9, 10),
        main_entities=("Ganga Prasad",),
        short_summary="RTI appellate order involving Ganga Prasad.",
        summary_source="fixture",
        procedural_type="rti_appeal",
        manifest_status="ingested",
        manifest_error=None,
        ingested=True,
        document_version_id=302,
        detail_url="https://example.com/ganga-prasad",
        pdf_url="https://example.com/ganga-prasad.pdf",
        local_filename="ganga_prasad.pdf",
    )
    alias_rows = (
        EntityAliasRow(
            canonical_name="Hemant Ghai",
            short_name=None,
            abbreviations=(),
            old_name=None,
            new_name=None,
            related_record_keys=(hemant_doc.record_key,),
            related_titles=(hemant_doc.exact_title,),
        ),
        EntityAliasRow(
            canonical_name="Ganga Prasad",
            short_name=None,
            abbreviations=(),
            old_name=None,
            new_name=None,
            related_record_keys=(ganga_doc.record_key,),
            related_titles=(ganga_doc.exact_title,),
        ),
    )
    return ControlPack(
        root=Path(".").resolve(),
        document_index=(hemant_doc, ganga_doc),
        confusion_pairs=(),
        eval_queries=(),
        wrong_answer_examples=(),
        entity_aliases=alias_rows,
        strict_answer_rule=StrictAnswerRule(
            text="Stay inside one matter.",
            strict_single_matter_required=True,
        ),
        documents_by_record_key={
            hemant_doc.record_key: hemant_doc,
            ganga_doc.record_key: ganga_doc,
        },
        aliases_by_record_key={
            hemant_doc.record_key: (alias_rows[0],),
            ganga_doc.record_key: (alias_rows[1],),
        },
        alias_variants={
            "hemant ghai": (alias_rows[0],),
            "ganga prasad": (alias_rows[1],),
        },
        confusion_map={},
    )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
