"""
AI Insights generation routes.
Orchestrates the full analysis pipeline: structured data analysis,
sentiment analysis, theme extraction, correlation analysis,
executive summary, and retention recommendations.
"""

from fastapi import APIRouter, HTTPException

from api.models.schemas import (
    GenerateInsightsRequest,
    InsightResponse,
    SummaryKPIs,
    DepartmentAttrition,
    OvertimeAnalysis,
    RiskFactor,
    DepartmentStats,
    SentimentResult,
    ThemeResult,
    CorrelationInsight,
    RetentionRecommendation,
)
from api.services.data_service import get_backend
from api.services import chroma_service
from api.services import openai_service

router = APIRouter(prefix="/api/insights", tags=["AI Insights"])

# In-memory cache for generated insights (per company)
_insights_cache: dict[str, InsightResponse] = {}


@router.post("/generate", response_model=InsightResponse)
async def generate_insights(request: GenerateInsightsRequest):
    """
    Generate comprehensive AI insights by running the full analysis pipeline.

    Steps:
    1. Fetch structured attrition stats from data service
    2. Fetch all VoE feedback from ChromaDB
    3. Run GPT-4o sentiment analysis on feedback
    4. Run GPT-4o theme extraction on feedback
    5. Compute sentiment-attrition correlations via GPT-4o
    6. Generate executive summary via GPT-4o
    7. Generate per-department retention recommendations via GPT-4o

    Returns a complete InsightResponse payload.
    """
    company = request.company_name
    backend = get_backend()

    # Validate data is loaded
    if not backend.is_loaded(company):
        raise HTTPException(
            status_code=400,
            detail=f"No attrition data loaded for '{company}'. Upload CSV first.",
        )

    feedback_count = chroma_service.get_feedback_count(company)
    if feedback_count == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No feedback data for '{company}'. Upload JSON first.",
        )

    # ── Step 1: Structured data analysis ──

    summary_kpis = backend.get_summary_kpis(company)
    attrition_by_dept = backend.get_attrition_by_department(company)
    risk_factors = backend.get_risk_factors(company)
    department_stats = backend.get_department_stats(company)
    overtime_analysis = backend.get_overtime_analysis(company)

    # ── Step 2: Fetch all feedback from ChromaDB ──

    feedback_by_dept = chroma_service.get_feedback_by_department(company)
    all_feedback_texts = []
    for texts in feedback_by_dept.values():
        all_feedback_texts.extend(texts)

    # ── Step 3: Sentiment analysis via GPT-4o ──

    sentiment_by_dept = openai_service.analyze_sentiment_batch(
        feedback_by_dept, company
    )

    # ── Step 4: Theme extraction via GPT-4o ──

    themes = openai_service.extract_themes(all_feedback_texts, company)

    # ── Step 5: Correlation analysis via GPT-4o ──

    correlations = openai_service.analyze_correlations(
        company, attrition_by_dept, sentiment_by_dept
    )

    # ── Step 6: Executive summary via GPT-4o ──

    executive_summary = openai_service.generate_executive_summary(
        company_name=company,
        summary_kpis=summary_kpis,
        attrition_by_dept=attrition_by_dept,
        risk_factors=risk_factors,
        sentiment_by_dept=sentiment_by_dept,
        themes=themes,
    )

    # ── Step 7: Retention recommendations via GPT-4o ──

    retention_recs = openai_service.generate_retention_recommendations(
        company_name=company,
        department_stats=department_stats,
        sentiment_by_dept=sentiment_by_dept,
        themes=themes,
    )

    # ── Assemble response ──

    response = InsightResponse(
        company_name=company,
        executive_summary=executive_summary,
        summary_kpis=SummaryKPIs(**summary_kpis),
        department_attrition=[
            DepartmentAttrition(**d) for d in attrition_by_dept
        ],
        department_stats=[DepartmentStats(**d) for d in department_stats],
        risk_factors=[RiskFactor(**r) for r in risk_factors],
        overtime_analysis=OvertimeAnalysis(**overtime_analysis),
        sentiment_by_department=[
            SentimentResult(**s) for s in sentiment_by_dept
        ],
        themes=[ThemeResult(**t) for t in themes],
        correlations=[CorrelationInsight(**c) for c in correlations],
        retention_recommendations=[
            RetentionRecommendation(**r) for r in retention_recs
        ],
    )

    # Cache for subsequent reads
    _insights_cache[company] = response
    return response


@router.get("/status")
async def get_insights_status(company_name: str):
    """Check if insights have been generated for a company."""
    return {
        "company_name": company_name,
        "has_insights": company_name in _insights_cache,
        "has_attrition_data": get_backend().is_loaded(company_name),
        "feedback_count": chroma_service.get_feedback_count(company_name),
    }


@router.get("/cached", response_model=InsightResponse)
async def get_cached_insights(company_name: str):
    """Retrieve previously generated insights from cache."""
    if company_name not in _insights_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No insights generated yet for '{company_name}'",
        )
    return _insights_cache[company_name]
