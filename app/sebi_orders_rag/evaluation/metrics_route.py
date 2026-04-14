"""Route and planner evaluation metrics."""

from __future__ import annotations

from .schemas import EvaluationCase, RouteMetrics

_NAMED_SINGLE_MATTER_ROUTES = frozenset(
    {"exact_lookup", "hierarchical_rag", "memory_scoped_rag"}
)
_GENERAL_EXPLANATORY_ROUTES = frozenset({"direct_llm", "general_knowledge", "smalltalk"})
_CURRENT_INFO_ROUTES = frozenset(
    {"structured_current_info", "current_official_lookup", "historical_official_lookup"}
)
_LATEST_NEWS_ROUTES = frozenset({"current_news_lookup", "current_official_lookup"})


def evaluate_route_metrics(
    *,
    case: EvaluationCase,
    route_mode: str,
    answer_status: str,
    debug: dict[str, object],
    actual_record_keys: tuple[str, ...],
) -> RouteMetrics:
    """Return route/planner correctness for one case."""

    expected_route = case.route_family_expected
    strict_route_match = bool(expected_route is None or route_mode == expected_route)
    equivalent_route_match, equivalent_route_reason = equivalent_route_match_reason(
        case=case,
        route_mode=route_mode,
        answer_status=answer_status,
        debug=debug,
        actual_record_keys=actual_record_keys,
    )
    planner_debug = dict(debug.get("planner_debug", {}) or {})
    route_debug = dict(debug.get("route_debug", {}) or {})
    web_debug = dict(debug.get("web_fallback_debug", {}) or {})

    planner_choice_correct = None
    if planner_debug.get("used"):
        planned = str(planner_debug.get("execution_route_mode") or "").strip()
        planner_choice_correct = (planned == route_mode) if planned else None

    internal_first_policy_correct = None
    if case.must_not_use_web:
        internal_first_policy_correct = not (
            bool(web_debug.get("official_web_attempted"))
            or bool(web_debug.get("general_web_attempted"))
        )
    elif case.must_use_official_web:
        internal_first_policy_correct = bool(web_debug.get("official_web_attempted")) or (
            route_mode in _CURRENT_INFO_ROUTES
        )

    web_fallback_correct = None
    if case.must_use_official_web:
        web_fallback_correct = bool(web_debug.get("official_web_attempted")) or (
            route_mode in _CURRENT_INFO_ROUTES
        )
    elif case.must_not_use_web:
        web_fallback_correct = not (
            bool(web_debug.get("official_web_attempted"))
            or bool(web_debug.get("general_web_attempted"))
        )

    active_matter_follow_up_correct = None
    if case.must_use_active_matter:
        active_matter_follow_up_correct = bool(route_debug.get("active_order_override")) and (
            route_mode == "memory_scoped_rag"
        )

    company_role_routing_correct = None
    if "company_role" in case.tags:
        company_role_routing_correct = route_mode in {
            "current_official_lookup",
            "general_knowledge",
        }

    return RouteMetrics(
        strict_route_match=strict_route_match,
        equivalent_route_match=equivalent_route_match,
        equivalent_route_reason=equivalent_route_reason,
        planner_choice_correct=planner_choice_correct,
        internal_first_policy_correct=internal_first_policy_correct,
        web_fallback_correct=web_fallback_correct,
        active_matter_follow_up_correct=active_matter_follow_up_correct,
        company_role_routing_correct=company_role_routing_correct,
    )


def equivalent_route_match_reason(
    *,
    case: EvaluationCase,
    route_mode: str,
    answer_status: str,
    debug: dict[str, object],
    actual_record_keys: tuple[str, ...],
) -> tuple[bool, str | None]:
    """Return explicit equivalent-route reasoning."""

    expected_route = case.route_family_expected
    if expected_route is None or route_mode == expected_route:
        return True, "strict_route_match" if expected_route else None

    route_debug = dict(debug.get("route_debug", {}) or {})
    candidate_list_debug = dict(debug.get("candidate_list_debug", {}) or {})

    if (
        expected_route in _GENERAL_EXPLANATORY_ROUTES
        and route_mode in _GENERAL_EXPLANATORY_ROUTES
        and route_debug.get("appears_general_explanatory")
        and not route_debug.get("appears_matter_specific")
    ):
        return True, "general_explanatory_equivalent"

    if (
        expected_route in _NAMED_SINGLE_MATTER_ROUTES
        and route_mode in _NAMED_SINGLE_MATTER_ROUTES
        and case.expected_record_keys
        and set(actual_record_keys) == set(case.expected_record_keys)
        and route_debug.get("strict_scope_required")
    ):
        return True, "named_single_matter_equivalent"

    if (
        answer_status == "clarify"
        and bool(candidate_list_debug.get("used"))
        and route_debug.get("strict_scope_required")
        and not route_debug.get("strict_single_matter")
        and case.expected_record_keys
        and set(case.expected_record_keys).issubset(
            set(candidate_list_debug.get("record_keys", []) or ())
        )
    ):
        return True, "ambiguous_named_matter_clarify"

    if (
        expected_route in _CURRENT_INFO_ROUTES
        and route_mode in _CURRENT_INFO_ROUTES
        and (
            case.must_use_structured_current_info
            or case.must_use_official_web
            or route_debug.get("appears_structured_current_info")
            or route_debug.get("appears_current_official_lookup")
            or route_debug.get("appears_historical_official_lookup")
        )
    ):
        return True, "current_info_equivalent"

    if (
        expected_route in _LATEST_NEWS_ROUTES
        and route_mode in _LATEST_NEWS_ROUTES
        and route_debug.get("appears_current_news_lookup")
    ):
        return True, "latest_news_equivalent"

    allowed = set(case.allowed_routes)
    if allowed and route_mode in allowed:
        return True, "allowed_route_override"

    return False, None
