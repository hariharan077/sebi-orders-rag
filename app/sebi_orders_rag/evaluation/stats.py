"""Pandas/numpy aggregation helpers for evaluation runs."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .schemas import CaseEvaluationResult


def results_to_frame(results: tuple[CaseEvaluationResult, ...] | list[CaseEvaluationResult]) -> pd.DataFrame:
    """Flatten case results into a dataframe."""

    rows = [item.to_flat_dict() for item in results]
    if not rows:
        return pd.DataFrame(
            columns=[
                "case_id",
                "query",
                "issue_class",
                "difficulty",
                "route_family_expected",
                "route_mode",
                "answer_status",
                "confidence",
                "passed",
            ]
        )
    return pd.DataFrame(rows)


def summarize_case_results(
    results: tuple[CaseEvaluationResult, ...] | list[CaseEvaluationResult],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the high-level acceptance summary for one run."""

    frame = results_to_frame(results)
    total_cases = int(len(frame))
    if total_cases == 0:
        return {
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "strict_route_accuracy": 0.0,
            "equivalent_route_accuracy": 0.0,
            "faithfulness_average": 0.0,
            "hallucination_rate": 0.0,
            "contamination_count": 0,
            "candidate_list_correctness": 0.0,
            "numeric_fact_accuracy": 0.0,
            "structured_current_info_accuracy": 0.0,
            "current_fact_routing_accuracy": 0.0,
            "wrong_answer_regression_pass_count": 0,
            "wrong_answer_regression_total": 0,
            "stale_expectation_count": 0,
            "true_bug_count": 0,
            "confidence_ece": 0.0,
            "confidence_brier": 0.0,
            "answer_status_counts": {},
            "failure_bucket_counts": {},
            "metadata": dict(metadata or {}),
        }

    faithfulness_average = _series_mean(frame["faithfulness"])
    hallucination_rate = _series_mean(frame["hallucination_rate"])
    candidate_list_correctness = _series_mean(
        frame.loc[
            frame["candidate_list_correctness"].notna()
            & (
                frame["primary_failure_bucket"].eq("wrong candidate ranking")
                | frame["query"].str.contains("clarif", case=False, na=False)
                | frame["answer_status"].eq("clarify")
            ),
            "candidate_list_correctness",
        ]
    )
    numeric_fact_accuracy = _series_mean(
        frame.loc[frame["expected_fact_count"].fillna(0) > 0, "numeric_accuracy"]
    )
    structured_current_info_accuracy = _series_mean(
        frame.loc[
            frame["query"].str.contains("designation|chairman|income|charge", case=False, na=False)
            | frame["route_family_expected"].isin(
                ["structured_current_info", "current_official_lookup", "historical_official_lookup"]
            ),
            "equivalent_route_match",
        ]
    )
    current_fact_routing_accuracy = _series_mean(
        frame.loc[
            frame["route_family_expected"].isin(
                ["structured_current_info", "current_official_lookup", "historical_official_lookup", "current_news_lookup"]
            ),
            "equivalent_route_match",
        ]
    )
    regression_mask = frame["issue_class"].eq("regression") | frame["tags"].apply(
        lambda values: "wrong_answer_regression" in set(values or [])
    )
    wrong_answer_regression_total = int(regression_mask.sum())
    wrong_answer_regression_pass_count = int(
        frame.loc[regression_mask, "passed"].fillna(False).sum()
    )
    confidence_stats = confidence_bin_frame(frame)
    summary = {
        "total_cases": total_cases,
        "passed_cases": int(frame["passed"].fillna(False).sum()),
        "failed_cases": int((~frame["passed"].fillna(False)).sum()),
        "strict_route_accuracy": round(_series_mean(frame["strict_route_match"]), 4),
        "equivalent_route_accuracy": round(_series_mean(frame["equivalent_route_match"]), 4),
        "faithfulness_average": round(faithfulness_average, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "contamination_count": int(
            (
                frame["primary_failure_bucket"].eq("contamination")
                | frame["failure_buckets"].apply(lambda values: "contamination" in set(values or []))
            ).sum()
        ),
        "candidate_list_correctness": round(candidate_list_correctness, 4),
        "numeric_fact_accuracy": round(numeric_fact_accuracy, 4),
        "structured_current_info_accuracy": round(structured_current_info_accuracy, 4),
        "current_fact_routing_accuracy": round(current_fact_routing_accuracy, 4),
        "wrong_answer_regression_pass_count": wrong_answer_regression_pass_count,
        "wrong_answer_regression_total": wrong_answer_regression_total,
        "stale_expectation_count": int(frame["stale_expectation"].fillna(False).sum()),
        "true_bug_count": int(frame["true_bug"].fillna(False).sum()),
        "answer_status_counts": (
            frame["answer_status"].value_counts(dropna=False).sort_index().to_dict()
        ),
        "failure_bucket_counts": (
            frame["primary_failure_bucket"].value_counts(dropna=False).sort_index().to_dict()
        ),
        "route_confusion": confusion_matrix_frame(frame).to_dict(orient="index"),
        "confidence_distribution": confidence_stats.to_dict(orient="records"),
        "confidence_ece": round(expected_calibration_error(confidence_stats), 4),
        "confidence_brier": round(brier_score(frame), 4),
        "metadata": dict(metadata or {}),
    }
    return json.loads(json.dumps(summary, default=_json_default))


def metrics_by_tag_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by tag."""

    if frame.empty:
        return pd.DataFrame()
    exploded = frame.explode("tags")
    exploded = exploded[exploded["tags"].notna() & exploded["tags"].astype(str).ne("")]
    if exploded.empty:
        return pd.DataFrame(columns=["tag", "cases", "pass_rate"])
    grouped = exploded.groupby("tags", dropna=False)
    result = grouped.agg(
        cases=("case_id", "count"),
        pass_rate=("passed", "mean"),
        strict_route_accuracy=("strict_route_match", "mean"),
        equivalent_route_accuracy=("equivalent_route_match", "mean"),
        faithfulness_average=("faithfulness", "mean"),
        numeric_fact_accuracy=("numeric_accuracy", "mean"),
    ).reset_index()
    return result.rename(columns={"tags": "tag"}).sort_values("cases", ascending=False)


def metrics_by_route_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by expected route."""

    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("route_family_expected", dropna=False)
    result = grouped.agg(
        cases=("case_id", "count"),
        pass_rate=("passed", "mean"),
        strict_route_accuracy=("strict_route_match", "mean"),
        equivalent_route_accuracy=("equivalent_route_match", "mean"),
        hallucination_rate=("hallucination_rate", "mean"),
    ).reset_index()
    return result.sort_values(["cases", "route_family_expected"], ascending=[False, True])


def metrics_by_prompt_family_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by prompt family."""

    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby("prompt_family", dropna=False)
    result = grouped.agg(
        cases=("case_id", "count"),
        pass_rate=("passed", "mean"),
        faithfulness_average=("faithfulness", "mean"),
        answer_correctness=("answer_correctness", "mean"),
        reasoning_quality=("reasoning_quality", "mean"),
    ).reset_index()
    return result.sort_values(["cases", "prompt_family"], ascending=[False, True])


def confidence_bin_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build calibration bins."""

    if frame.empty:
        return pd.DataFrame(columns=["confidence_bin", "cases", "mean_confidence", "precision"])
    grouped = frame.groupby("confidence_bin", dropna=False)
    result = grouped.agg(
        cases=("case_id", "count"),
        mean_confidence=("confidence", "mean"),
        precision=("passed", "mean"),
        abstain_rate=("answer_status", lambda values: np.mean([value in {"abstained", "clarify"} for value in values])),
    ).reset_index()
    return result.sort_values("confidence_bin")


def confusion_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return route confusion counts."""

    if frame.empty:
        return pd.DataFrame()
    confusion = pd.crosstab(
        frame["route_family_expected"],
        frame["route_mode"],
        rownames=["expected_route"],
        colnames=["actual_route"],
        dropna=False,
    )
    return confusion


def compare_result_frames(base: pd.DataFrame, head: pd.DataFrame) -> pd.DataFrame:
    """Compare two result frames by route family."""

    base_metrics = metrics_by_route_frame(base).rename(
        columns={
            "cases": "base_cases",
            "pass_rate": "base_pass_rate",
            "strict_route_accuracy": "base_strict_route_accuracy",
            "equivalent_route_accuracy": "base_equivalent_route_accuracy",
            "hallucination_rate": "base_hallucination_rate",
        }
    )
    head_metrics = metrics_by_route_frame(head).rename(
        columns={
            "cases": "head_cases",
            "pass_rate": "head_pass_rate",
            "strict_route_accuracy": "head_strict_route_accuracy",
            "equivalent_route_accuracy": "head_equivalent_route_accuracy",
            "hallucination_rate": "head_hallucination_rate",
        }
    )
    comparison = base_metrics.merge(
        head_metrics,
        on="route_family_expected",
        how="outer",
    )
    metric_columns = [column for column in comparison.columns if column != "route_family_expected"]
    comparison[metric_columns] = comparison[metric_columns].fillna(0)
    comparison["pass_rate_delta"] = comparison["head_pass_rate"] - comparison["base_pass_rate"]
    comparison["equivalent_route_accuracy_delta"] = (
        comparison["head_equivalent_route_accuracy"]
        - comparison["base_equivalent_route_accuracy"]
    )
    return comparison.sort_values(
        "route_family_expected",
        key=lambda values: values.fillna("").astype(str),
    )


def expected_calibration_error(confidence_bins: pd.DataFrame) -> float:
    """Compute a simple expected calibration error."""

    if confidence_bins.empty:
        return 0.0
    total = float(confidence_bins["cases"].sum())
    if total <= 0:
        return 0.0
    error = 0.0
    for _, row in confidence_bins.iterrows():
        error += abs(float(row["precision"]) - float(row["mean_confidence"])) * (
            float(row["cases"]) / total
        )
    return error


def brier_score(frame: pd.DataFrame) -> float:
    """Compute a binary Brier score over pass/fail correctness."""

    if frame.empty:
        return 0.0
    confidence = frame["confidence"].astype(float).to_numpy()
    correct = frame["passed"].astype(float).to_numpy()
    return float(np.mean(np.square(confidence - correct)))


def _series_mean(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _json_default(value: object) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
