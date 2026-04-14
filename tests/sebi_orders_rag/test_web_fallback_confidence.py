from __future__ import annotations

import unittest

from app.sebi_orders_rag.answering.confidence import assess_web_fallback_confidence
from app.sebi_orders_rag.web_fallback.models import WebSearchSource


class WebFallbackConfidenceTests(unittest.TestCase):
    def test_official_web_result_can_be_high_confidence(self) -> None:
        assessment = assess_web_fallback_confidence(
            answer_status="answered",
            sources=(
                WebSearchSource(
                    source_title="SEBI Official Circular",
                    source_url="https://www.sebi.gov.in/legal/circulars/apr-2026/test.html",
                    domain="sebi.gov.in",
                    source_type="official_web",
                    record_key="official_web:sebi.gov.in",
                ),
                WebSearchSource(
                    source_title="Department of Economic Affairs",
                    source_url="https://dea.gov.in/test",
                    domain="dea.gov.in",
                    source_type="official_web",
                    record_key="official_web:dea.gov.in",
                ),
            ),
            preferred_source_type="official_web",
            preferred_domains=("sebi.gov.in", "gov.in", "nic.in"),
        )

        self.assertGreaterEqual(assessment.confidence, 0.8)
        self.assertFalse(assessment.should_abstain)

    def test_weak_general_web_result_stays_low_confidence(self) -> None:
        assessment = assess_web_fallback_confidence(
            answer_status="answered",
            sources=(
                WebSearchSource(
                    source_title="Blog post",
                    source_url="https://example.com/post",
                    domain="example.com",
                    source_type="general_web",
                    record_key="general_web:example.com",
                ),
            ),
            preferred_source_type="general_web",
        )

        self.assertLess(assessment.confidence, 0.6)
        self.assertTrue(assessment.should_abstain)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
