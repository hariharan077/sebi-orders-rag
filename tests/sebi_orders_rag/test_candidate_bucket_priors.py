from __future__ import annotations

import unittest
from datetime import date

from app.sebi_orders_rag.control.candidate_selection import (
    select_exact_lookup_resolution,
    sort_exact_lookup_candidates,
)
from app.sebi_orders_rag.schemas import ExactLookupCandidate


class CandidateBucketPriorTests(unittest.TestCase):
    def test_sat_court_candidates_are_ranked_ahead_of_other_buckets(self) -> None:
        candidates = (
            ExactLookupCandidate(
                document_version_id=301,
                document_id=401,
                record_key="external:prime-broking-other",
                bucket_name="orders-under-regulation-30a",
                external_record_id="prime-broking-other",
                order_date=date(2026, 3, 12),
                title="Prime Broking Company (India) Limited",
                match_score=0.77,
            ),
            ExactLookupCandidate(
                document_version_id=302,
                document_id=402,
                record_key="external:prime-broking-sat",
                bucket_name="orders-of-sat",
                external_record_id="prime-broking-sat",
                order_date=date(2017, 7, 10),
                title="Prime Broking Company (India) Limited vs NSE",
                match_score=0.73,
            ),
        )

        ordered = sort_exact_lookup_candidates(candidates, sat_court_query=True)

        self.assertEqual(ordered[0].record_key, "external:prime-broking-sat")
        self.assertEqual(ordered[1].record_key, "external:prime-broking-other")

    def test_strong_single_sat_candidate_resolves_directly(self) -> None:
        resolution = select_exact_lookup_resolution(
            (
                ExactLookupCandidate(
                    document_version_id=501,
                    document_id=601,
                    record_key="external:tushar-sat",
                    bucket_name="orders-of-sat",
                    external_record_id="tushar-sat",
                    order_date=date(2015, 6, 30),
                    title="Tushar Oil Food Ltd vs SEBI",
                    match_score=0.79,
                ),
                ExactLookupCandidate(
                    document_version_id=502,
                    document_id=602,
                    record_key="external:tushar-other",
                    bucket_name="orders-under-regulation-30a",
                    external_record_id="tushar-other",
                    order_date=date(2026, 1, 5),
                    title="Tushar Oil Food Ltd",
                    match_score=0.71,
                ),
            ),
            sat_court_query=True,
        )

        self.assertEqual(resolution.selected_document_id, 501)
        self.assertFalse(resolution.should_clarify)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
