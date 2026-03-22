"""
Server-side insight generation routes.

Accept ONLY anonymized/aggregated data from the client container.
All GPT-4o calls happen here; no raw employee data crosses the wire.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from shared.contracts import (
    SchemaAnalysisRequest,
    AnalysisPlanRequest,
    InsightRequest,
    InsightResponse,
    AnalysisSection,
    MLResults,
    AskRequest,
    AskResponse,
    PiiValidationRequest,
    PiiValidationResponse,
)
from server.api.services import openai_service

router = APIRouter()


# ── Schema Analysis ──────────────────────────────────────────────────

@router.post("/api/analyze-schema")
async def analyze_schema(req: SchemaAnalysisRequest) -> dict:
    """Analyze column metadata and return column mapping."""
    try:
        result = openai_service.analyze_csv_schema(
            dtypes=req.dtypes,
            shape=req.shape,
            sample_rows=req.sample_rows,
            unique_counts=req.unique_counts,
            null_counts=req.null_counts,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analysis Plan ────────────────────────────────────────────────────

@router.post("/api/generate-plan")
async def generate_plan(req: AnalysisPlanRequest) -> dict:
    """Generate dynamic analysis steps from column mapping + metadata."""
    try:
        analyses = openai_service.generate_analysis_plan(
            column_mapping=req.column_mapping,
            dtypes=req.dtypes,
            shape=req.shape,
            sample_rows=req.sample_rows,
            unique_counts=req.unique_counts,
        )
        return {"analyses": analyses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Full Insight Generation Pipeline ────────────────────────────────

@router.post("/api/generate-insights", response_model=InsightResponse)
async def generate_insights(req: InsightRequest) -> InsightResponse:
    """
    Run the full GPT-4o insight pipeline on pre-aggregated data.
    Client has already run ML, structured analyses, and PII scrubbing.
    """
    try:
        sections: list[AnalysisSection] = []

        # 1. Sentiment analysis on PII-scrubbed feedback
        sentiment_by_dept = openai_service.analyze_sentiment_batch(
            feedback_by_department=req.feedback.feedback_by_dept,
            company_name=req.company_name,
        )

        # 2. Theme extraction
        all_texts = []
        for dept_texts in req.feedback.feedback_by_dept.values():
            all_texts.extend(dept_texts)
        themes = openai_service.extract_themes(
            all_feedback_texts=all_texts,
            company_name=req.company_name,
        )

        # 3. Feedback queries (for context)
        openai_service.formulate_feedback_queries(
            company_name=req.company_name,
            group_by_dept=req.group_by_dept,
            risk_factors=req.risk_factors,
            department_stats=req.department_stats,
        )

        # 4. ML narrative (deep mode only)
        ml_results_model = None
        if req.ml_results and req.ml_results.get("feature_importance"):
            ml_narrative = openai_service.generate_ml_narrative(
                company_name=req.company_name,
                ml_results=req.ml_results,
                target_info=req.target_info,
            )
            ml_data = {**req.ml_results, "ml_narrative": ml_narrative}
            ml_results_model = MLResults(**ml_data)

        # 5. Recommendations
        recommendations = openai_service.generate_recommendations(
            company_name=req.company_name,
            department_stats=req.department_stats,
            sentiment_by_dept=sentiment_by_dept,
            themes=themes,
            target_info=req.target_info,
            ml_results=req.ml_results,
        )

        # 6. Correlations
        correlations = openai_service.analyze_correlations(
            company_name=req.company_name,
            group_by_dept=req.group_by_dept,
            sentiment_by_dept=sentiment_by_dept,
            target_info=req.target_info,
        )

        # 7. Dashboard spec
        dashboard = openai_service.generate_dashboard_spec(
            company_name=req.company_name,
            structured_results=req.structured_results,
            sentiment_by_dept=sentiment_by_dept,
            themes=themes,
            correlations=correlations,
            target_info=req.target_info,
            ml_results=req.ml_results,
        )

        # 8. Executive summary
        executive_summary = openai_service.generate_executive_summary(
            company_name=req.company_name,
            structured_results=req.structured_results,
            sentiment_by_dept=sentiment_by_dept,
            themes=themes,
            target_info=req.target_info,
            ml_results=req.ml_results,
        )

        # Build sections from dashboard spec
        for s in dashboard.get("sections", []):
            charts = s.get("charts", [])
            sections.append(AnalysisSection(
                id=s.get("id", ""),
                title=s.get("title", ""),
                category=s.get("category", "structured"),
                narrative=s.get("narrative", ""),
                charts=charts,
                metrics=s.get("metrics", []),
            ))

        return InsightResponse(
            company_name=req.company_name,
            executive_summary=executive_summary,
            target_description=req.target_info.get("target_description", ""),
            domain_context=req.target_info,
            analysis_mode=req.analysis_mode,
            kpis=dashboard.get("kpis", []),
            sections=sections,
            recommendations=recommendations,
            ml_results=ml_results_model,
            ml_skip_reason=req.ml_skip_reason,
            pii_handling_summary=(
                [e.dict() for e in req.pii_handling_summary]
                if req.pii_handling_summary
                else None
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Ask AI (context-only Q&A) ───────────────────────────────────────

@router.post("/api/ask", response_model=AskResponse)
async def ask_ai(req: AskRequest) -> AskResponse:
    """Answer user questions using cached aggregated context only."""
    try:
        result = openai_service.ask_question(
            company_name=req.company_name,
            question=req.question,
            insights_context=req.insights_context,
            conversation_history=req.conversation_history,
        )
        return AskResponse(
            answer=result.get("answer", ""),
            chart=result.get("chart_spec"),
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── PII Validation ──────────────────────────────────────────────────

@router.post("/api/validate-pii", response_model=PiiValidationResponse)
async def validate_pii(req: PiiValidationRequest) -> PiiValidationResponse:
    """AI review of PII classifications — can only tighten (upgrade)."""
    try:
        merged = openai_service.validate_pii_classification(req.classifications)
        return PiiValidationResponse(columns=merged)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
