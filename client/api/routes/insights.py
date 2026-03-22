"""
Client-side insight generation route.

Orchestrates the full analysis pipeline:
  - Steps 1-5 run LOCALLY (schema analysis, plan execution, ML, feedback)
  - Aggregated/anonymized data is sent to the provider for GPT-4o insights
  - Provider returns InsightResponse; client caches it

Raw data NEVER leaves this container.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from shared.contracts import (
    InsightResponse,
    AnalysisSection,
    MLResults,
    AskRequest,
    AskResponse,
    ChartSpec,
)
from client.services.data_service import get_backend
from client.services import chroma_service
from client.services import provider_client
from client.services.aggregation_service import (
    build_schema_metadata,
    build_insight_payload,
    scrub_feedback_for_provider,
    build_insights_context_for_ask,
)
from client.services.pii_classifier import scrub_text_pii, apply_pii_policy
from client.services.audit_service import log_event

router = APIRouter(prefix="/api/insights", tags=["AI Insights"])

# In-memory cache
_insights_cache: dict[str, InsightResponse] = {}
_ask_context_cache: dict[str, str] = {}

logger = logging.getLogger(__name__)


class _GenerateInsightsRequest:
    """Simple request body for the generate endpoint."""
    def __init__(self, company_name: str, analysis_mode: str = "quick"):
        self.company_name = company_name
        self.analysis_mode = analysis_mode


@router.post("/generate", response_model=InsightResponse)
async def generate_insights(request: dict):
    """
    Generate insights: local data processing + remote GPT-4o orchestration.
    """
    company = request.get("company_name", "")
    analysis_mode = request.get("analysis_mode", "quick")
    backend = get_backend()

    if not backend.is_loaded(company):
        raise HTTPException(status_code=400, detail=f"No data for '{company}'")

    if chroma_service.get_feedback_count(company) == 0:
        raise HTTPException(status_code=400, detail=f"No feedback for '{company}'")

    if not backend.has_pii_classification(company):
        raise HTTPException(
            status_code=400,
            detail="Schema must be reviewed and approved first.",
        )

    try:
        return await _run_pipeline(company, backend, analysis_mode)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline failed for %s", company)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(e).__name__}: {str(e)[:500]}",
        )


async def _run_pipeline(
    company: str,
    backend,
    analysis_mode: str = "quick",
) -> InsightResponse:
    """Run the full analysis pipeline with local + remote stages."""

    pii_classifications = backend.get_pii_classification(company)

    # ── LOCAL Step 1: Schema analysis via provider ──

    if not backend.has_mapping(company):
        meta = build_schema_metadata(backend, company, pii_classifications)
        column_mapping = provider_client.analyze_schema(
            company_name=company,
            dtypes=meta["dtypes"],
            shape=meta["shape"],
            sample_rows=meta["sample_rows"],
            unique_counts=meta["unique_counts"],
            null_counts=meta["null_counts"],
            pii_columns=[c.to_dict() for c in pii_classifications],
        )
        backend.set_schema_mapping(company, column_mapping)
    else:
        meta = build_schema_metadata(backend, company, pii_classifications)

    target_info = backend.get_target_info(company)

    # ── LOCAL Step 2: Analysis plan via provider ──

    mapping = backend._mapping(company)
    safe_samples = meta["sample_rows"]

    analysis_plan = provider_client.generate_plan(
        company_name=company,
        column_mapping=mapping,
        dtypes=meta["dtypes"],
        shape=meta["shape"],
        sample_rows=safe_samples,
        unique_counts=meta["unique_counts"],
    )

    # ── LOCAL Step 3: Execute analysis plan on local pandas ──

    plan_results = backend.execute_analysis_plan(company, analysis_plan)

    all_structured_results = {}
    for step in analysis_plan:
        all_structured_results[step["id"]] = {
            "type": step["type"],
            "description": step["description"],
            "params": step.get("params", {}),
            "result": plan_results.get(step["id"]),
        }

    dept_col = mapping.get("department_col", "Department")
    group_by_dept = []
    risk_factors = []
    for step in analysis_plan:
        r = plan_results.get(step["id"])
        if step["type"] in ("group_by_target", "group_attrition"):
            group_by = step.get("params", {}).get("group_by", "")
            if group_by == dept_col and isinstance(r, list):
                group_by_dept = r
        elif step["type"] == "correlations" and isinstance(r, list):
            risk_factors.extend(r)

    risk_factors.sort(key=lambda x: abs(x.get("correlation", 0)), reverse=True)
    risk_factors = risk_factors[:12]

    department_stats = backend.get_department_stats(company)

    # ── LOCAL Step 3.5: ML Pipeline (deep mode only) ──

    ml_results_raw = None
    ml_skip_reason = None
    if analysis_mode == "deep":
        try:
            from client.services import ml_service
            ml_df = backend._df(company)
            ml_mapping = backend._mapping(company)
            ml_results_raw = ml_service.run_ml_pipeline(
                ml_df, ml_mapping,
                pii_classifications=pii_classifications,
            )
            if ml_results_raw.get("_skip_reason"):
                ml_skip_reason = ml_results_raw["_skip_reason"]
                ml_results_raw = None
        except Exception as e:
            logger.exception("ML pipeline failed for %s", company)
            ml_skip_reason = f"ML pipeline error: {e}"

    # ── LOCAL Step 4-5: Retrieve and scrub feedback ──

    # Get all feedback grouped by department from local ChromaDB
    feedback_by_dept_raw = chroma_service.get_feedback_by_department(company)

    # Scrub all PII from feedback text before sending to provider
    feedback_by_dept = scrub_feedback_for_provider(feedback_by_dept_raw)

    # ── Build PII handling summary ──

    pii_handling_summary = None
    if pii_classifications:
        pii_handling_summary = [
            {
                "column": c.column_name,
                "category": c.pii_category,
                "handling": c.handling,
            }
            for c in pii_classifications
            if c.handling != "pass_through"
        ]

    # ── REMOTE: Send aggregated data to provider for GPT-4o insights ──

    payload = build_insight_payload(
        company_name=company,
        analysis_mode=analysis_mode,
        target_info=target_info,
        column_mapping=mapping,
        structured_results=all_structured_results,
        department_stats=department_stats,
        group_by_dept=group_by_dept,
        risk_factors=risk_factors,
        feedback_by_dept=feedback_by_dept,
        ml_results=ml_results_raw,
        ml_skip_reason=ml_skip_reason,
        pii_handling_summary=pii_handling_summary,
    )

    response_data = provider_client.generate_insights(payload)

    # Parse response into InsightResponse
    sections = []
    for s in response_data.get("sections", []):
        chart_list = []
        for c in s.get("charts", []):
            chart_list.append(ChartSpec(
                chart_type=c.get("chart_type", "bar"),
                title=c.get("title", ""),
                data=c.get("data", []),
                x=c.get("x", ""),
                y=c.get("y", ""),
                color=c.get("color"),
            ))
        sections.append(AnalysisSection(
            id=s.get("id", ""),
            title=s.get("title", ""),
            category=s.get("category", "structured"),
            narrative=s.get("narrative", ""),
            charts=chart_list,
            metrics=s.get("metrics", []),
        ))

    ml_results_model = None
    ml_data = response_data.get("ml_results")
    if ml_data and isinstance(ml_data, dict) and ml_data.get("feature_importance"):
        ml_results_model = MLResults(**ml_data)

    # Build domain_context for Ask AI
    domain_context = {
        "target_info": target_info,
        "structured_results": all_structured_results,
        "themes": response_data.get("sections", []),
        "correlations": [],
        "recommendations": response_data.get("recommendations", []),
    }

    response = InsightResponse(
        company_name=company,
        executive_summary=response_data.get("executive_summary", ""),
        target_description=target_info.get("target_description", ""),
        domain_context=domain_context,
        analysis_mode=analysis_mode,
        kpis=response_data.get("kpis", []),
        sections=sections,
        recommendations=response_data.get("recommendations", []),
        ml_results=ml_results_model,
        ml_skip_reason=response_data.get("ml_skip_reason", ml_skip_reason),
        pii_handling_summary=response_data.get("pii_handling_summary"),
    )

    _insights_cache[company] = response

    # Cache context for Ask AI
    _ask_context_cache[company] = build_insights_context_for_ask(response_data)

    log_event("insight_generation", company, {"analysis_mode": analysis_mode})
    return response


# ── Ask AI (forwards to provider with cached context) ────────────────

@router.post("/ask", response_model=AskResponse)
async def ask_ai(request: AskRequest):
    """Forward question + cached aggregated context to provider."""
    context = _ask_context_cache.get(request.company_name, "")
    if not context and request.insights_context:
        context = request.insights_context

    try:
        result = provider_client.ask_question(
            company_name=request.company_name,
            question=request.question,
            insights_context=context,
            conversation_history=request.conversation_history,
        )
        return AskResponse(
            answer=result.get("answer", ""),
            chart=result.get("chart_spec") or result.get("chart"),
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cached Insights ─────────────────────────────────────────────────

@router.get("/cached/{company_name}", response_model=InsightResponse)
async def get_cached_insights(company_name: str):
    cached = _insights_cache.get(company_name)
    if not cached:
        raise HTTPException(status_code=404, detail="No cached insights")
    return cached
