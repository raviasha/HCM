"""
Aggregation service — prepares anonymized/aggregated payloads for the provider.

This is the critical security boundary: all functions here take raw local data
and produce ONLY sanitized, aggregated outputs suitable for network transmission.
"""

from __future__ import annotations

import json
from typing import Any

from client.services.pii_classifier import (
    ColumnClassification,
    apply_pii_policy,
    scrub_text_pii,
)


def build_schema_metadata(
    backend,
    company_name: str,
    pii_classifications: list[ColumnClassification],
) -> dict:
    """
    Build sanitized schema metadata for the provider.
    PII columns get their sample values redacted.
    """
    meta = backend.get_schema_metadata(company_name)
    sample_rows = meta.get("sample_rows", [])
    safe_samples = apply_pii_policy(sample_rows, pii_classifications)

    # Filter unique_counts: exclude PII columns
    excluded = {c.column_name for c in pii_classifications if c.handling == "exclude"}
    unique_counts = {
        k: v for k, v in meta.get("unique_counts", {}).items()
        if k not in excluded
    }

    return {
        "dtypes": meta.get("dtypes", {}),
        "shape": meta.get("shape", []),
        "sample_rows": safe_samples,
        "unique_counts": unique_counts,
        "null_counts": meta.get("null_counts", {}),
    }


def build_insight_payload(
    company_name: str,
    analysis_mode: str,
    target_info: dict,
    column_mapping: dict,
    structured_results: dict,
    department_stats: list[dict],
    group_by_dept: list[dict],
    risk_factors: list[dict],
    feedback_by_dept: dict[str, list[str]],
    ml_results: dict | None = None,
    ml_skip_reason: str | None = None,
    pii_handling_summary: list[dict] | None = None,
) -> dict:
    """
    Build the full InsightRequest payload.
    All feedback texts are already PII-scrubbed at this point.
    """
    total_entries = sum(len(v) for v in feedback_by_dept.values())

    return {
        "company_name": company_name,
        "analysis_mode": analysis_mode,
        "target_info": target_info,
        "column_mapping": column_mapping,
        "structured_results": structured_results,
        "department_stats": department_stats,
        "group_by_dept": group_by_dept,
        "risk_factors": risk_factors,
        "feedback": {
            "feedback_by_dept": feedback_by_dept,
            "total_entries": total_entries,
        },
        "ml_results": ml_results,
        "ml_skip_reason": ml_skip_reason,
        "pii_handling_summary": pii_handling_summary,
    }


def scrub_feedback_for_provider(
    feedback_by_dept: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Double-scrub feedback text before sending to provider.
    Even though ChromaDB stores scrubbed text, we re-scrub for safety.
    """
    scrubbed: dict[str, list[str]] = {}
    for dept, texts in feedback_by_dept.items():
        scrubbed[dept] = [scrub_text_pii(t)[0] for t in texts]
    return scrubbed


def build_insights_context_for_ask(insight_response: dict) -> str:
    """
    Build a cached context string from the insight response for Ask AI.
    This is a text summary the provider uses as context for Q&A.
    """
    parts = []

    if insight_response.get("executive_summary"):
        parts.append(f"## Executive Summary\n{insight_response['executive_summary']}")

    if insight_response.get("domain_context"):
        ctx = insight_response["domain_context"]
        if ctx.get("target_info"):
            parts.append(f"## Target Variable\n{json.dumps(ctx['target_info'], indent=2)}")
        if ctx.get("themes"):
            parts.append(f"## Key Themes\n{json.dumps(ctx['themes'], indent=2)}")
        if ctx.get("correlations"):
            parts.append(f"## Correlations\n{json.dumps(ctx['correlations'], indent=2)}")

    if insight_response.get("kpis"):
        parts.append(f"## KPIs\n{json.dumps(insight_response['kpis'], indent=2)}")

    if insight_response.get("recommendations"):
        parts.append(
            f"## Recommendations\n{json.dumps(insight_response['recommendations'], indent=2)}"
        )

    ml = insight_response.get("ml_results")
    if ml:
        parts.append(f"## ML Results\n{json.dumps(ml, indent=2, default=str)}")

    return "\n\n".join(parts)
