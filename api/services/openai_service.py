"""
OpenAI service for AI-powered insight generation.

Handles all GPT-4o interactions:
  - Sentiment analysis of employee feedback
  - Theme/topic extraction from VoE data
  - Executive summary generation
  - Retention recommendations per department
  - Sentiment-attrition correlation narrative

All prompts run server-side (no user prompt needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import OpenAI

from config.settings import OPENAI_API_KEY, CHAT_MODEL

# ── Prompt loader ────────────────────────────────────────────────────

_PROMPTS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "prompts.yaml"


def _prompt(key: str) -> str:
    """Load a system prompt by key from config/prompts.yaml (re-read each call)."""
    with open(_PROMPTS_FILE, "r") as f:
        prompts = yaml.safe_load(f)
    text = prompts.get(key)
    if text is None:
        raise KeyError(f"Prompt key '{key}' not found in {_PROMPTS_FILE}")
    return text.strip()

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    """Send a chat completion request and return the assistant message."""
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
    """Send a chat completion expecting JSON output."""
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


# ── Sentiment Analysis ───────────────────────────────────────────────

def analyze_sentiment_batch(
    feedback_by_department: dict[str, list[str]],
    company_name: str,
) -> list[dict]:
    """
    Analyze sentiment of feedback grouped by department.

    Returns list of dicts:
        [{"department": "Engineering", "positive": 20, "neutral": 15,
          "negative": 35, "avg_score": -0.25}, ...]
    """
    system = _prompt("sentiment_analysis")

    # Prepare a summary for each department (truncate if too many)
    dept_summaries = []
    for dept, texts in feedback_by_department.items():
        sampled = texts[:40]  # Limit to avoid token overflow
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


# ── Theme Extraction ─────────────────────────────────────────────────

def extract_themes(
    all_feedback_texts: list[str],
    company_name: str,
) -> list[dict]:
    """
    Identify top themes/topics from all feedback.

    Returns list of dicts:
        [{"theme": "Work-Life Balance", "count": 45,
          "sample_quotes": ["quote1", "quote2", "quote3"]}, ...]
    """
    system = _prompt("theme_extraction")

    # Sample feedback to stay within token limits
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

def generate_executive_summary(
    company_name: str,
    summary_kpis: dict,
    attrition_by_dept: list[dict],
    risk_factors: list[dict],
    sentiment_by_dept: list[dict],
    themes: list[dict],
) -> str:
    """
    Generate a narrative executive summary from all analytical outputs.
    Returns a markdown-formatted summary string.
    """
    system = _prompt("executive_summary")

    user = (
        f"# {company_name} — Workforce Analytics Data\n\n"
        f"## Summary KPIs\n{json.dumps(summary_kpis, indent=2)}\n\n"
        f"## Attrition by Department\n{json.dumps(attrition_by_dept, indent=2)}\n\n"
        f"## Top Risk Factors\n{json.dumps(risk_factors[:8], indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Themes from Employee Feedback\n{json.dumps(themes, indent=2)}\n"
    )

    return _chat(system, user, temperature=0.4)


# ── Retention Recommendations ────────────────────────────────────────

def generate_retention_recommendations(
    company_name: str,
    department_stats: list[dict],
    sentiment_by_dept: list[dict],
    themes: list[dict],
) -> list[dict]:
    """
    Generate per-department retention recommendations.

    Returns list of dicts:
        [{"department": "Engineering", "risk_level": "High",
          "key_issues": ["Burnout", "Career stagnation"],
          "recommendations": ["Implement ...", "Create ..."]}, ...]
    """
    system = _prompt("retention_recommendations")

    user = (
        f"Company: {company_name}\n\n"
        f"## Department Statistics\n{json.dumps(department_stats, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Feedback Themes\n{json.dumps(themes, indent=2)}\n"
    )

    result = _chat_json(system, user)
    return result.get("departments", [])


# ── Correlation Insights ─────────────────────────────────────────────

def analyze_correlations(
    company_name: str,
    attrition_by_dept: list[dict],
    sentiment_by_dept: list[dict],
) -> list[dict]:
    """
    Analyze the correlation between sentiment scores and attrition rates
    across departments. Provide a narrative for each department.

    Returns list of dicts:
        [{"department": "Engineering", "attrition_rate": 0.45,
          "avg_sentiment_score": -0.25, "narrative": "..."}, ...]
    """
    system = _prompt("correlations")

    user = (
        f"Company: {company_name}\n\n"
        f"## Attrition by Department\n{json.dumps(attrition_by_dept, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n"
    )

    result = _chat_json(system, user)
    return result.get("correlations", [])
