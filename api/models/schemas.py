"""
Pydantic models for API request/response contracts.
These models define the data shapes exchanged between
FastAPI backend and Streamlit frontend.

All models are domain-generic — the AI dynamically determines
what analyses to run based on the uploaded dataset.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


# ── Chart / Section building blocks ──────────────────────────────────

class ChartSpec(BaseModel):
    """Lightweight chart specification the frontend renders with Plotly."""
    chart_type: str  # "bar", "pie", "scatter", "line", "table"
    title: str
    data: list[dict]
    x: str = ""
    y: str = ""
    color: Optional[str] = None


class AnalysisSection(BaseModel):
    """A dynamically generated dashboard section with charts and narrative."""
    id: str
    title: str
    category: str  # "structured", "voe", "correlation"
    narrative: str = ""
    charts: list[ChartSpec] = []
    metrics: list[dict] = []  # [{"label": str, "value": str, "description": str}]


# ── Insight Pipeline Response ────────────────────────────────────────

class InsightResponse(BaseModel):
    """Full response from the insight generation pipeline."""
    company_name: str
    executive_summary: str

    # Domain context (set dynamically from schema analysis)
    target_description: str = ""  # e.g. "Employee Attrition", "Engagement Score"
    domain_context: dict = {}  # raw analysis context for Ask AI

    # Dynamic AI-generated dashboard
    kpis: list[dict] = []
    sections: list[AnalysisSection] = []
    recommendations: list[dict] = []


# ── API Requests ─────────────────────────────────────────────────────

class GenerateInsightsRequest(BaseModel):
    company_name: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    row_count: Optional[int] = None


# ── Interactive Q&A ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    company_name: str
    question: str
    conversation_history: list[dict] = []


class AskResponse(BaseModel):
    answer: str
    chart: Optional[ChartSpec] = None
    sources: list[str] = []
