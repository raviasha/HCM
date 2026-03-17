"""
OpenAI service for AI-powered insight generation.

Handles all GPT-4o interactions:
  - Sentiment analysis of employee feedback
  - Theme/topic extraction
  - Executive summary generation
  - HCM recommendations per department
  - Sentiment-target metric correlation narrative

All prompts are domain-generic — the AI adapts to the dataset's
target variable (attrition, engagement, performance, etc.)
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
    structured_results: dict,
    sentiment_by_dept: list[dict],
    themes: list[dict],
    target_info: dict,
) -> str:
    """
    Generate a narrative executive summary from all analytical outputs.
    Returns a markdown-formatted summary string.
    """
    system = _prompt("executive_summary")

    user = (
        f"# {company_name} — Workforce Analytics Data\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Structured Analysis Results\n{json.dumps(structured_results, indent=2, default=str)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Themes from Employee Feedback\n{json.dumps(themes, indent=2)}\n"
    )

    return _chat(system, user, temperature=0.4)


# ── Retention Recommendations ────────────────────────────────────────

def generate_recommendations(
    company_name: str,
    department_stats: list[dict],
    sentiment_by_dept: list[dict],
    themes: list[dict],
    target_info: dict,
) -> list[dict]:
    """
    Generate per-department HCM recommendations.

    Returns list of dicts:
        [{"department": "Engineering", "risk_level": "High",
          "key_issues": ["Burnout", "Career stagnation"],
          "recommendations": ["Implement ...", "Create ..."]}, ...]
    """
    system = _prompt("recommendations")

    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Department Statistics\n{json.dumps(department_stats, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n\n"
        f"## Top Feedback Themes\n{json.dumps(themes, indent=2)}\n"
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
    """
    Analyze the correlation between sentiment scores and the target metric
    across departments. Provide a narrative for each department.

    Returns list of dicts:
        [{"department": "Engineering", "target_rate": 0.45,
          "avg_sentiment_score": -0.25, "narrative": "..."}, ...]
    """
    system = _prompt("correlations")

    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n{json.dumps(target_info, indent=2)}\n\n"
        f"## Target Metric by Department\n{json.dumps(group_by_dept, indent=2)}\n\n"
        f"## Sentiment by Department\n{json.dumps(sentiment_by_dept, indent=2)}\n"
    )

    result = _chat_json(system, user)
    return result.get("correlations", [])


# ── Dynamic Schema Analysis ──────────────────────────────────────────

def analyze_csv_schema(metadata: dict) -> dict:
    """
    Ask GPT-4o to analyze CSV column metadata and return a structured
    column mapping (attrition col, department col, numeric cols, etc.).
    """
    system = _prompt("schema_analysis")

    user = (
        "Analyze this HR dataset and identify the role of each column.\n\n"
        f"## Columns & Types\n{json.dumps(metadata['dtypes'], indent=2)}\n\n"
        f"## Shape\n{json.dumps(metadata['shape'])}\n\n"
        f"## Sample Rows (first 5)\n{json.dumps(metadata['sample_rows'], indent=2, default=str)}\n\n"
        f"## Unique Value Counts\n{json.dumps(metadata['unique_counts'], indent=2)}\n\n"
        f"## Null Counts\n{json.dumps(metadata['null_counts'], indent=2)}\n"
    )

    return _chat_json(system, user)


def generate_analysis_plan(
    metadata: dict,
    column_mapping: dict,
) -> list[dict]:
    """
    Ask GPT-4o to generate a dynamic analysis plan based on the
    schema metadata and column mapping.

    Returns list of analysis step dicts.
    """
    system = _prompt("analysis_plan")

    user = (
        "Plan the analyses for this HR dataset.\n\n"
        f"## Column Mapping\n{json.dumps(column_mapping, indent=2)}\n\n"
        f"## Columns & Types\n{json.dumps(metadata['dtypes'], indent=2)}\n\n"
        f"## Shape\n{json.dumps(metadata['shape'])}\n\n"
        f"## Sample Rows (first 5)\n{json.dumps(metadata['sample_rows'], indent=2, default=str)}\n\n"
        f"## Unique Value Counts\n{json.dumps(metadata['unique_counts'], indent=2)}\n"
    )

    result = _chat_json(system, user)
    return result.get("analyses", [])


# ── Dynamic Dashboard Specification ──────────────────────────────────

def generate_dashboard_spec(
    company_name: str,
    structured_results: dict,
    sentiment_by_dept: list[dict],
    themes: list[dict],
    correlations: list[dict],
    target_info: dict,
) -> dict:
    """
    Ask GPT-4o to generate a complete dashboard specification
    (KPIs, chart sections, recommendations) from all analysis results.

    Returns: {"kpis": [...], "sections": [...], "recommendations": [...]}
    """
    system = _prompt("generate_dashboard")

    user = (
        f"Company: {company_name}\n\n"
        f"## Target Variable\n"
        f"{json.dumps(target_info, indent=2)}\n\n"
        f"## Structured Analysis Results\n"
        f"{json.dumps(structured_results, indent=2, default=str)}\n\n"
        f"## Sentiment by Department\n"
        f"{json.dumps(sentiment_by_dept, indent=2, default=str)}\n\n"
        f"## Feedback Themes\n"
        f"{json.dumps(themes, indent=2, default=str)}\n\n"
        f"## Sentiment–Target Metric Correlations\n"
        f"{json.dumps(correlations, indent=2, default=str)}\n"
    )

    result = _chat_json(system, user)
    return {
        "kpis": result.get("kpis", []),
        "sections": result.get("sections", []),
        "recommendations": result.get("recommendations", []),
    }


# ── Targeted Feedback Query Formulation ──────────────────────────────

def formulate_feedback_queries(
    company_name: str,
    group_by_dept: list[dict],
    risk_factors: list[dict],
    department_stats: list[dict],
) -> dict[str, str]:
    """
    Ask GPT-4o to formulate a semantic search query per department
    based on structured data findings.

    Returns: {"Engineering": "career growth promotion frustration ...", ...}
    """
    system = _prompt("query_formulation")

    user = (
        f"Company: {company_name}\n\n"
        f"## Target Metric by Department\n{json.dumps(group_by_dept, indent=2)}\n\n"
        f"## Top Risk/Influence Factors\n"
        f"{json.dumps(risk_factors[:8], indent=2)}\n\n"
        f"## Department Stats\n{json.dumps(department_stats, indent=2)}\n"
    )

    result = _chat_json(system, user)
    return {
        item["department"]: item["query"]
        for item in result.get("queries", [])
    }


# ── Interactive Q&A with Tool Calling ────────────────────────────────

_ASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_structured_data",
            "description": (
                "Query the employee CSV data for structured statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": [
                            "kpis", "group_by_department", "risk_factors",
                            "department_stats", "group_by_factor",
                        ],
                        "description": "Type of structured query to run.",
                    },
                    "factor": {
                        "type": "string",
                        "description": (
                            "Column name for group_by_factor "
                            "(e.g. 'Age', 'MonthlyIncome', 'YearsAtCompany')."
                        ),
                    },
                },
                "required": ["query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_feedback",
            "description": (
                "Semantic search over employee feedback "
                "stored in a vector database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "department": {
                        "type": "string",
                        "description": "Optional department filter.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default 15).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def ask_question(
    company_name: str,
    question: str,
    insights_context: str,
    conversation_history: list[dict],
    data_backend,
) -> dict:
    """
    Interactive Q&A with tool calling. GPT-4o can call back into
    pandas (structured data) and ChromaDB (feedback search) as needed.

    Returns: {"answer": str, "chart_spec": dict|None, "sources": list[str]}
    """
    from api.services import chroma_service

    system = (
        _prompt("ask_question")
        + f"\n\n## Cached Insights Context\n{insights_context}"
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    client = _get_client()
    sources: set[str] = set()
    choice = None

    # Agentic loop — allow up to 5 tool-call rounds
    for _ in range(5):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=messages,
            tools=_ASK_TOOLS,
            response_format={"type": "json_object"},
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            messages.append(choice.message)
            break

        # Process tool calls
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn_name == "query_structured_data":
                sources.add("structured_data")
                result = data_backend.query(
                    company_name,
                    args["query_type"],
                    args.get("factor", ""),
                )
            elif fn_name == "search_feedback":
                sources.add("feedback")
                raw = chroma_service.query_feedback(
                    company_name=company_name,
                    query_text=args["query"],
                    department=args.get("department"),
                    n_results=args.get("n_results", 15),
                )
                result = [
                    {"text": r["document"], "department": r["metadata"]["department"]}
                    for r in raw
                ]
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    # Parse final response
    text = (choice.message.content or "{}") if choice else "{}"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"answer": text, "chart_spec": None, "sources": []}

    if not sources:
        sources.add("cached_insights")

    return {
        "answer": parsed.get("answer", text),
        "chart_spec": parsed.get("chart_spec"),
        "sources": list(sources),
    }
