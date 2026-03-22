"""
OpenAI service for the provider backend.

All functions receive ONLY anonymized/aggregated data from the client.
No raw DataFrames, no raw employee IDs, no un-scrubbed feedback text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import OpenAI

from server.config.settings import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL

# ── Prompt loader ────────────────────────────────────────────────────

_PROMPTS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"

_client: Optional[OpenAI] = None


def _prompt(key: str) -> str:
    with open(_PROMPTS_FILE, "r") as f:
        prompts = yaml.safe_load(f)
    text = prompts.get(key)
    if text is None:
        raise KeyError(f"Prompt key '{key}' not found in {_PROMPTS_FILE}")
    return text.strip()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _chat_json(system: str, user: str, temperature: float = 0.2) -> Any:
    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content or "{}"
    return json.loads(text)


# ── Schema Analysis (column names + dtypes + safe samples only) ──────

def analyze_csv_schema(
    dtypes: dict,
    shape: list,
    sample_rows: list[dict],
    unique_counts: dict,
    null_counts: dict,
) -> dict:
    """Analyze column metadata and return structured column mapping."""
    system = _prompt("schema_analysis")
    user = (
        "Analyze this HR dataset and identify the role of each column.\n\n"
        f"## Columns & Types\n{json.dumps(dtypes, indent=2)}\n\n"
        f"## Shape\n{json.dumps(shape)}\n\n"
        f"## Sample Rows (first 5, PII anonymized)\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"## Unique Value Counts\n{json.dumps(unique_counts, indent=2)}\n\n"
        f"## Null Counts\n{json.dumps(null_counts, indent=2)}\n"
    )
    return _chat_json(system, user)


def generate_analysis_plan(
    column_mapping: dict,
    dtypes: dict,
    shape: list,
    sample_rows: list[dict],
    unique_counts: dict,
) -> list[dict]:
    """Generate dynamic analysis plan from schema metadata."""
    system = _prompt("analysis_plan")
    user = (
        "Plan the analyses for this HR dataset.\n\n"
        f"## Column Mapping\n{json.dumps(column_mapping, indent=2)}\n\n"
        f"## Columns & Types\n{json.dumps(dtypes, indent=2)}\n\n"
        f"## Shape\n{json.dumps(shape)}\n\n"
        f"## Sample Rows (first 5, PII anonymized)\n{json.dumps(sample_rows, indent=2, default=str)}\n\n"
        f"## Unique Value Counts\n{json.dumps(unique_counts, indent=2)}\n"
    )
    result = _chat_json(system, user)
    return result.get("analyses", [])


# ── PII Validation (AI additive review) ──────────────────────────────

_STRICTNESS_ORDER = {"safe": 0, "quasi_identifier": 1, "direct_pii": 2, "identifier": 3}
_HANDLING_FOR_CATEGORY = {
    "safe": "pass_through",
    "quasi_identifier": "aggregate_only",
    "direct_pii": "exclude",
    "identifier": "exclude",
}


def validate_pii_classification(classifications: list[dict]) -> list[dict]:
    """AI review of PII classifications — can only upgrade (tighten)."""
    system = _prompt("pii_validation")
    review_input = [
        {
            "column_name": c["column_name"],
            "dtype": c["dtype"],
            "pii_category": c["pii_category"],
            "handling": c["handling"],
        }
        for c in classifications
    ]
    user = (
        "Review these column PII classifications for an HR dataset.\n\n"
        f"{json.dumps(review_input, indent=2)}\n"
    )
    result = _chat_json(system, user)
    ai_columns = result.get("columns", [])
    ai_map = {c["column_name"]: c for c in ai_columns}

    merged = []
    for orig in classifications:
        col_name = orig["column_name"]
        ai_col = ai_map.get(col_name)
        if ai_col:
            orig_level = _STRICTNESS_ORDER.get(orig["pii_category"], 0)
            ai_level = _STRICTNESS_ORDER.get(ai_col.get("pii_category", "safe"), 0)
            if ai_level > orig_level:
                new_cat = ai_col["pii_category"]
                merged.append({
                    **orig,
                    "pii_category": new_cat,
                    "handling": _HANDLING_FOR_CATEGORY.get(new_cat, orig["handling"]),
                    "confidence": ai_col.get("confidence", orig["confidence"]),
                    "reason": f"AI upgrade: {ai_col.get('reason', orig['reason'])}",
                })
                continue
        merged.append(orig)
    return merged


# ── Sentiment Analysis (receives PII-scrubbed text) ─────────────────

def analyze_sentiment_batch(
    feedback_by_department: dict[str, list[str]],
    company_name: str,
) -> list[dict]:
    system = _prompt("sentiment_analysis")
    dept_summaries = []
    for dept, texts in feedback_by_department.items():
        sampled = texts[:40]
        dept_summaries.append(
            f"## {dept} ({len(texts)} total entries, showing {len(sampled)})\n"
            + "\n".join(f"- {t}" for t in sampled)
        )
    user = (
        f"Company: {company_name}\n\n"
        "Analyze the sentiment of employee feedback for each department below.\n\n"
        + "\n\n".join(dept_summaries)
    )
    result = _chat_json(system, user)
    return result.get("departments", [])


# ── Theme Extraction (receives PII-scrubbed text) ───────────────────

def extract_themes(
    all_feedback_texts: list[str],
    company_name: str,
) -> list[dict]:
    system = _prompt("theme_extraction")
    sampled = all_feedback_texts[:150]
    user = (
        f"Company: {company_name}\n"
        f"Total feedback entries: {len(all_feedback_texts)}\n"
        f"Showing {len(sampled)} entries:\n\n"
        + "\n".join(f"- {t}" for t in sampled)
    )
    result = _chat_json(system, user)
    return result.get("themes", [])


# ── Executive Summary ────────────────────────────────────────────────

def _sanitize_ml_results_for_llm(ml_results: dict | None) -> dict | None:
    if not ml_results:
        return ml_results
    sanitized = {**ml_results}
    risk = sanitized.get("risk_scores", {})
    if risk and "top_risk_employees" in risk:
        sanitized["risk_scores"] = {**risk}
        sanitized["risk_scores"]["top_risk_employees"] = [
            {
                "risk_score": emp.get("risk_score"),
                "risk_level": emp.get("risk_level"),
                "department": emp.get("department"),
            }
            for emp in risk["top_risk_employees"]
        ]
    return sanitized


def generate_executive_summary(
    company_name: str,
    structured_results: dict,
    sentiment_by_dept: list[dict],
    themes: list[dict],
    target_info: dict,
    ml_results: dict | None = None,
) -> str:
    system = _prompt("executive_summary")
    user = (
        f"# {company_name} — Workforce Analytics Data\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Structured Analysis Results\n{json.dumps(structured_results, indent=2, default=str)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Themes from Employee Feedback\n{json.dumps(themes, indent=2)}\n"
    )
    if ml_results and ml_results.get("feature_importance"):
        safe_ml = _sanitize_ml_results_for_llm(ml_results)
        user += (
            f"\n\n## ML Predictive Analytics\n"
            f"### Top Predictors\n{json.dumps(safe_ml.get('feature_importance', [])[:5], indent=2)}\n"
            f"### Risk Distribution\n{json.dumps(safe_ml.get('risk_scores', {}).get('distribution', {}), indent=2)}\n"
            f"### What-If Scenarios\n{json.dumps(safe_ml.get('what_if_scenarios', []), indent=2)}\n"
        )
    return _chat(system, user, temperature=0.4)


# ── ML Narrative ─────────────────────────────────────────────────────

def generate_ml_narrative(
    company_name: str,
    ml_results: dict,
    target_info: dict,
) -> str:
    system = _prompt("ml_narrative")
    safe_ml = _sanitize_ml_results_for_llm(ml_results) or ml_results
    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Feature Importance (Top Predictors)\n"
        f"{json.dumps(safe_ml.get('feature_importance', []), indent=2)}\n\n"
        f"## Risk Score Distribution\n"
        f"{json.dumps(safe_ml.get('risk_scores', {}).get('distribution', {}), indent=2)}\n\n"
        f"## High-Risk Departments\n"
        f"{json.dumps(safe_ml.get('risk_scores', {}).get('high_risk_departments', []), indent=2)}\n\n"
        f"## Survival Analysis\n"
        f"{json.dumps(safe_ml.get('survival_analysis'), indent=2, default=str)}\n\n"
        f"## Employee Segments (Clustering)\n"
        f"{json.dumps(safe_ml.get('clustering', {}).get('profiles', []), indent=2)}\n\n"
        f"## What-If Scenarios\n"
        f"{json.dumps(safe_ml.get('what_if_scenarios', []), indent=2)}\n"
    )
    return _chat(system, user, temperature=0.4)


# ── Recommendations ──────────────────────────────────────────────────

def generate_recommendations(
    company_name: str,
    department_stats: list[dict],
    sentiment_by_dept: list[dict],
    themes: list[dict],
    target_info: dict,
    ml_results: dict | None = None,
) -> list[dict]:
    system = _prompt("recommendations")
    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Department Statistics\n{json.dumps(department_stats, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Feedback Themes\n{json.dumps(themes, indent=2)}\n"
    )
    if ml_results and ml_results.get("feature_importance"):
        safe_ml = _sanitize_ml_results_for_llm(ml_results)
        user += (
            f"\n\n## ML Predictive Analytics\n"
            f"### Top Risk Predictors\n{json.dumps(safe_ml.get('feature_importance', [])[:5], indent=2)}\n"
            f"### High-Risk Departments\n{json.dumps(safe_ml.get('risk_scores', {}).get('high_risk_departments', []), indent=2)}\n"
            f"### What-If Scenarios\n{json.dumps(safe_ml.get('what_if_scenarios', []), indent=2)}\n"
            f"\nUse these ML predictions to make recommendations more specific and data-driven.\n"
        )
    result = _chat_json(system, user)
    return result.get("departments", [])


# ── Correlation Insights ─────────────────────────────────────────────

def analyze_correlations(
    company_name: str,
    group_by_dept: list[dict],
    sentiment_by_dept: list[dict],
    target_info: dict,
) -> list[dict]:
    system = _prompt("correlations")
    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Target Metric by Department\n{json.dumps(group_by_dept, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n"
    )
    result = _chat_json(system, user)
    return result.get("correlations", [])


# ── Dashboard Specification ──────────────────────────────────────────

def generate_dashboard_spec(
    company_name: str,
    structured_results: dict,
    sentiment_by_dept: list[dict],
    themes: list[dict],
    correlations: list[dict],
    target_info: dict,
    ml_results: dict | None = None,
) -> dict:
    system = _prompt("generate_dashboard")
    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Structured Analysis Results\n{json.dumps(structured_results, indent=2, default=str)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2, default=str)}\n\n"
        f"## Feedback Themes\n{json.dumps(themes, indent=2, default=str)}\n\n"
        f"## Sentiment–Target Metric Correlations\n{json.dumps(correlations, indent=2, default=str)}\n"
    )
    if ml_results and ml_results.get("feature_importance"):
        safe_ml = _sanitize_ml_results_for_llm(ml_results)
        user += (
            f"\n\n## ML Predictive Analytics Results\n"
            f"### Feature Importance\n{json.dumps(safe_ml.get('feature_importance', []), indent=2)}\n"
            f"### Risk Score Distribution\n{json.dumps(safe_ml.get('risk_scores', {}).get('distribution', {}), indent=2)}\n"
            f"### What-If Scenarios\n{json.dumps(safe_ml.get('what_if_scenarios', []), indent=2)}\n"
            f"\nInclude the ML findings in your executive summary KPIs and recommendations.\n"
        )
    result = _chat_json(system, user)
    return {
        "kpis": result.get("kpis", []),
        "sections": result.get("sections", []),
        "recommendations": result.get("recommendations", []),
    }


# ── Feedback Query Formulation ───────────────────────────────────────

def formulate_feedback_queries(
    company_name: str,
    group_by_dept: list[dict],
    risk_factors: list[dict],
    department_stats: list[dict],
) -> dict[str, str]:
    system = _prompt("query_formulation")
    user = (
        f"Company: {company_name}\n\n"
        f"## Target Metric by Department\n{json.dumps(group_by_dept, indent=2)}\n\n"
        f"## Top Risk/Influence Factors\n{json.dumps(risk_factors[:8], indent=2)}\n\n"
        f"## Department Stats\n{json.dumps(department_stats, indent=2)}\n"
    )
    result = _chat_json(system, user)
    return {
        item["department"]: item["query"]
        for item in result.get("queries", [])
    }


# ── Ask AI (context-only, no live data queries) ─────────────────────

def ask_question(
    company_name: str,
    question: str,
    insights_context: str,
    conversation_history: list[dict],
) -> dict:
    """
    Interactive Q&A using cached aggregated context only.
    No live pandas/ChromaDB queries — all context pre-aggregated by client.
    """
    system = (
        _prompt("ask_question")
        + f"\n\n## Cached Insights Context\n{insights_context}"
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    client = _get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=messages,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"answer": text, "chart_spec": None, "sources": []}

    return {
        "answer": parsed.get("answer", text),
        "chart_spec": parsed.get("chart_spec"),
        "sources": ["cached_insights"],
    }


# ── Embedding Proxy ──────────────────────────────────────────────────

def generate_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Generate embeddings for PII-scrubbed texts using OpenAI API."""
    client = _get_client()
    response = client.embeddings.create(
        input=texts,
        model=model or EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]
