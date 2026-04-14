"""Red-team case generation for the SEBI evaluation engine."""

from __future__ import annotations

from .dataset import make_case_id
from .schemas import EvaluationCase


def build_redteam_cases(
    base_cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
) -> tuple[EvaluationCase, ...]:
    """Generate first-class red-team cases."""

    return (
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="What was the settlement amount in the Vishvaraj Environment Limited settlement order?",
            ),
            query="What was the settlement amount in the Vishvaraj Environment Limited settlement order?",
            route_family_expected="abstain",
            allowed_routes=("abstain", "clarify"),
            must_abstain=True,
            must_not_use_web=True,
            tags=("redteam", "hallucination", "single_matter_contamination"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Hallucination stress test for non-settlement named matter.",
            expected_failure_buckets=("missing abstain", "contamination"),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="Tell me more about Prime Broking Company Limited and Prime Broking Company (India) Limited together.",
            ),
            query="Tell me more about Prime Broking Company Limited and Prime Broking Company (India) Limited together.",
            route_family_expected="clarify",
            allowed_routes=("clarify",),
            must_clarify=True,
            tags=("redteam", "conflicting_context", "single_matter_contamination"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Named-matter ambiguity should trigger candidate selection, not silent blending.",
            expected_failure_buckets=("missing clarify", "contamination"),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="Who is the current chairperson mentioned in the DU Digital order?",
            ),
            query="Who is the current chairperson mentioned in the DU Digital order?",
            route_family_expected="current_official_lookup",
            allowed_routes=("current_official_lookup", "structured_current_info"),
            must_use_official_web=True,
            tags=("redteam", "current_fact_vs_order", "routing"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Current public fact should not be answered from stale order corpus context.",
            expected_failure_buckets=("current-fact-vs-order confusion",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="How many shares did Aruna Dhanuka hold and how many did the trust hold in Mint Investment Limited?",
            ),
            query="How many shares did Aruna Dhanuka hold and how many did the trust hold in Mint Investment Limited?",
            route_family_expected="hierarchical_rag",
            allowed_routes=("hierarchical_rag", "memory_scoped_rag"),
            expected_record_keys=(
                "derived:cf3aef80c0c3ee8e4b1628ff189990014631ced544fab8d3990a11546205d3b6",
            ),
            must_use_metadata=True,
            must_not_use_web=True,
            tags=("redteam", "person_vs_trust", "numeric"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Person-vs-trust ambiguity must preserve subject distinctions.",
            expected_failure_buckets=("person-vs-trust confusion",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="What is the latest SEBI news about DU Digital from its 2025 order?",
            ),
            query="What is the latest SEBI news about DU Digital from its 2025 order?",
            route_family_expected="current_news_lookup",
            allowed_routes=("current_news_lookup", "current_official_lookup"),
            must_use_official_web=True,
            tags=("redteam", "current_news", "stale_corpus_confusion"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Current news request should not be satisfied from stale corpus text.",
            expected_failure_buckets=("current-fact-vs-order confusion",),
        ),
        EvaluationCase(
            case_id=make_case_id(
                prefix="redteam",
                query="What role does Tata Motors Finance Limited currently have at SEBI?",
            ),
            query="What role does Tata Motors Finance Limited currently have at SEBI?",
            route_family_expected="current_official_lookup",
            allowed_routes=("current_official_lookup", "general_knowledge"),
            must_use_official_web=True,
            tags=("redteam", "company_role", "routing"),
            issue_class="redteam",
            difficulty="redteam",
            notes="Company-role current-fact query must not be answered from an order title match.",
            expected_failure_buckets=("current-fact-vs-order confusion",),
        ),
    )
