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
    category: str  # "structured", "voe", "correlation", "ml"
    narrative: str = ""
    charts: list[ChartSpec] = []
    metrics: list[dict] = []  # [{"label": str, "value": str, "description": str}]


# ── ML Results ───────────────────────────────────────────────────────

class MLResults(BaseModel):
    """Results from the ML predictive analytics pipeline."""
    feature_importance: list[dict] = []
    risk_scores: dict = {}
    survival_analysis: Optional[dict] = None
    clustering: dict = {}
    what_if_scenarios: list[dict] = []
    model_metrics: dict = {}
    data_quality: dict = {}
    ml_narrative: str = ""


# ── Insight Pipeline Response ────────────────────────────────────────

class InsightResponse(BaseModel):
    """Full response from the insight generation pipeline."""
    company_name: str
    executive_summary: str

    # Domain context (set dynamically from schema analysis)
    target_description: str = ""  # e.g. "Employee Attrition", "Engagement Score"
    domain_context: dict = {}  # raw analysis context for Ask AI

    # Analysis mode
    analysis_mode: str = "quick"  # "quick" or "deep"

    # Dynamic AI-generated dashboard
    kpis: list[dict] = []
    sections: list[AnalysisSection] = []
    recommendations: list[dict] = []

    # ML results (only populated in "deep" mode)
    ml_results: Optional[MLResults] = None
    ml_skip_reason: Optional[str] = None

    # PII handling transparency — tells the dashboard what was excluded/hashed
    pii_handling_summary: Optional[list[dict]] = None


# ── API Requests ─────────────────────────────────────────────────────

class GenerateInsightsRequest(BaseModel):
    company_name: str
    analysis_mode: str = "quick"  # "quick" or "deep"


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


# ── Schema Review & PII Consent ──────────────────────────────────────

class ColumnClassificationOut(BaseModel):
    """PII classification for a single column."""
    column_name: str
    dtype: str
    pii_category: str   # identifier | direct_pii | quasi_identifier | safe
    handling: str        # exclude | hash | aggregate_only | pass_through
    confidence: str      # high | medium | low
    reason: str


class PiiDetectionOut(BaseModel):
    """A single PII match detected in free text."""
    pii_type: str
    original: str
    start: int
    end: int


class FeedbackSampleOut(BaseModel):
    """A feedback entry with original and PII-scrubbed text."""
    original_text: str
    scrubbed_text: str
    pii_detections: list[PiiDetectionOut] = []
    metadata: dict = {}


class SchemaReviewResponse(BaseModel):
    """Returned by /api/data/analyze-schema after PII classification."""
    company_name: str
    row_count: int
    column_count: int
    columns: list[ColumnClassificationOut]
    pii_summary: dict  # {"identifier": 1, "direct_pii": 0, ...}
    sample_rows: list[dict]  # PII columns already redacted
    feedback_columns: list[ColumnClassificationOut] = []
    feedback_samples: list[FeedbackSampleOut] = []
    feedback_entry_count: int = 0
    feedback_text_pii_count: int = 0


class ApproveSchemaRequest(BaseModel):
    """User-approved column classifications (can only tighten, not loosen)."""
    company_name: str
    columns: list[ColumnClassificationOut]
    feedback_columns: list[ColumnClassificationOut] = []
