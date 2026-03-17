"""
AI Insights generation routes.
Orchestrates the full analysis pipeline: dynamic schema analysis,
analysis plan execution, sentiment analysis, theme extraction,
correlation analysis, executive summary, and recommendations.

All logic is domain-generic — the target variable (attrition, engagement,
performance, etc.) is detected dynamically from the CSV schema.
"""

from fastapi import APIRouter, HTTPException

from api.models.schemas import (
    GenerateInsightsRequest,
    InsightResponse,
    AnalysisSection,
    AskRequest,
    AskResponse,
    ChartSpec,
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
    Generate comprehensive AI insights via a dynamic analysis pipeline.

    Steps:
    1. GPT-4o analyzes CSV schema to identify columns + target variable
    2. GPT-4o generates a dynamic analysis plan
    3. Execute analysis plan against pandas
    4. GPT-4o formulates targeted ChromaDB queries per department
    5. Retrieve semantically relevant feedback from ChromaDB
    6. Run GPT-4o sentiment analysis on targeted feedback
    7. Run GPT-4o theme extraction on targeted feedback
    8. Compute sentiment-target metric correlations via GPT-4o
    9. Generate executive summary via GPT-4o
    10. Generate recommendations via GPT-4o
    11. GPT-4o generates dynamic dashboard specification

    Returns a complete InsightResponse payload.
    """
    company = request.company_name
    backend = get_backend()

    # Validate data is loaded
    if not backend.is_loaded(company):
        raise HTTPException(
            status_code=400,
            detail=f"No structured data loaded for '{company}'. Upload CSV first.",
        )

    feedback_count = chroma_service.get_feedback_count(company)
    if feedback_count == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No feedback data for '{company}'. Upload JSON first.",
        )

    # ── Step 1: Dynamic schema analysis via GPT-4o ──

    if not backend.has_mapping(company):
        schema_metadata = backend.get_schema_metadata(company)
        column_mapping = openai_service.analyze_csv_schema(schema_metadata)
        backend.set_schema_mapping(company, column_mapping)
    else:
        schema_metadata = backend.get_schema_metadata(company)

    target_info = backend.get_target_info(company)

    # ── Step 2: GPT-4o generates dynamic analysis plan ──

    analysis_plan = openai_service.generate_analysis_plan(
        metadata=schema_metadata,
        column_mapping=backend._mapping(company),
    )

    # ── Step 3: Execute analysis plan against pandas ──

    plan_results = backend.execute_analysis_plan(company, analysis_plan)

    # Collect ALL plan results for downstream AI calls
    all_structured_results = {}
    for step in analysis_plan:
        all_structured_results[step["id"]] = {
            "type": step["type"],
            "description": step["description"],
            "params": step.get("params", {}),
            "result": plan_results.get(step["id"]),
        }

    # Extract group-by-department results and risk factors for feedback queries
    m = backend._mapping(company)
    dept_col = m.get("department_col", "Department")

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

    # ── Step 4: GPT-4o formulates targeted queries per department ──

    dept_queries = openai_service.formulate_feedback_queries(
        company_name=company,
        group_by_dept=group_by_dept,
        risk_factors=risk_factors,
        department_stats=department_stats,
    )

    # ── Step 5: Retrieve targeted feedback from ChromaDB per department ──

    targeted_feedback_by_dept: dict[str, list[str]] = {}
    for dept, query_text in dept_queries.items():
        results = chroma_service.query_feedback(
            company_name=company,
            query_text=query_text,
            department=dept,
            n_results=30,
        )
        targeted_feedback_by_dept[dept] = [r["document"] for r in results]

    all_feedback_texts = []
    for texts in targeted_feedback_by_dept.values():
        all_feedback_texts.extend(texts)

    # ── Step 6: Sentiment analysis via GPT-4o ──

    sentiment_by_dept = openai_service.analyze_sentiment_batch(
        targeted_feedback_by_dept, company
    )

    # ── Step 7: Theme extraction via GPT-4o ──

    themes = openai_service.extract_themes(all_feedback_texts, company)

    # ── Step 8: Correlation analysis via GPT-4o ──

    correlations = openai_service.analyze_correlations(
        company, group_by_dept, sentiment_by_dept, target_info
    )

    # ── Step 9: Executive summary via GPT-4o ──

    executive_summary = openai_service.generate_executive_summary(
        company_name=company,
        structured_results=all_structured_results,
        sentiment_by_dept=sentiment_by_dept,
        themes=themes,
        target_info=target_info,
    )

    # ── Step 10: Recommendations via GPT-4o ──

    recommendations_raw = openai_service.generate_recommendations(
        company_name=company,
        department_stats=department_stats,
        sentiment_by_dept=sentiment_by_dept,
        themes=themes,
        target_info=target_info,
    )

    # ── Step 11: GPT-4o generates dynamic dashboard specification ──

    dashboard_spec = openai_service.generate_dashboard_spec(
        company_name=company,
        structured_results=all_structured_results,
        sentiment_by_dept=sentiment_by_dept,
        themes=themes,
        correlations=correlations,
        target_info=target_info,
    )

    # Build AnalysisSection objects from dashboard spec
    sections = []
    for s in dashboard_spec.get("sections", []):
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

    # ── Assemble response ──

    # Build domain context dict for Ask AI
    domain_context = {
        "target_info": target_info,
        "structured_results": all_structured_results,
        "sentiment_by_dept": sentiment_by_dept,
        "themes": [{"theme": t.get("theme", ""), "count": t.get("count", 0)} for t in themes],
        "correlations": correlations,
        "recommendations": recommendations_raw,
    }

    response = InsightResponse(
        company_name=company,
        executive_summary=executive_summary,
        target_description=target_info.get("target_description", ""),
        domain_context=domain_context,

        # Dynamic dashboard
        kpis=dashboard_spec.get("kpis", []),
        sections=sections,
        recommendations=dashboard_spec.get("recommendations", []),
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
        "has_data": get_backend().is_loaded(company_name),
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


@router.post("/ask", response_model=AskResponse)
async def ask_question_endpoint(request: AskRequest):
    """
    Interactive Q&A over company data. Uses cached insights as context,
    with tool calls to pandas and ChromaDB for fresh queries.
    """
    company = request.company_name

    if company not in _insights_cache:
        raise HTTPException(
            status_code=400,
            detail=f"No insights generated yet for '{company}'. Run /generate first.",
        )

    cached = _insights_cache[company]
    # Build context from the domain_context dict + executive summary
    ctx = cached.domain_context or {}
    insights_context = (
        f"Executive Summary:\n{cached.executive_summary}\n\n"
        f"Target Variable: {cached.target_description}\n\n"
        f"Structured Results: {ctx.get('structured_results', {})}\n\n"
        f"Sentiment by Department: {ctx.get('sentiment_by_dept', [])}\n\n"
        f"Themes: {ctx.get('themes', [])}\n\n"
        f"Correlations: {ctx.get('correlations', [])}\n\n"
        f"Recommendations: {ctx.get('recommendations', [])}"
    )

    backend = get_backend()

    result = openai_service.ask_question(
        company_name=company,
        question=request.question,
        insights_context=insights_context,
        conversation_history=request.conversation_history,
        data_backend=backend,
    )

    chart = None
    cs = result.get("chart_spec")
    if cs and isinstance(cs, dict):
        chart = ChartSpec(**cs)
    elif cs and isinstance(cs, list) and len(cs) > 0 and isinstance(cs[0], dict):
        chart = ChartSpec(**cs[0])

    return AskResponse(
        answer=result["answer"],
        chart=chart,
        sources=result.get("sources", []),
    )
