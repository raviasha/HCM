"""
Shared API contracts between the client container and provider backend.

These Pydantic models define the ONLY data shapes that cross the network
boundary. The client sends anonymized/aggregated data; the provider
returns AI-generated insights. Raw data never leaves the client.

Both `client/` and `server/` import from this module.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


# ═════════════════════════════════════════════════════════════════════
#  Building blocks (shared by both sides)
# ═════════════════════════════════════════════════════════════════════

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
    metrics: list[dict] = []


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


class ColumnClassificationOut(BaseModel):
    """PII classification for a single column."""
    column_name: str
    dtype: str
    pii_category: str   # identifier | direct_pii | quasi_identifier | safe
    handling: str        # exclude | hash | aggregate_only | pass_through
    confidence: str      # high | medium | low
    reason: str


class PiiHandlingEntry(BaseModel):
    """One entry in the PII handling summary."""
    column: str
    category: str
    handling: str


# ═════════════════════════════════════════════════════════════════════
#  Client → Server requests (anonymized payloads only)
# ═════════════════════════════════════════════════════════════════════

class SchemaAnalysisRequest(BaseModel):
    """
    Client sends sanitized schema metadata for GPT-4o column mapping.
    Contains column names, dtypes, and PII-redacted sample rows.
    NO raw cell values for PII columns.
    """
    company_name: str
    dtypes: dict[str, str]                # column_name → dtype string
    shape: list[int]                       # [rows, cols]
    sample_rows: list[dict]                # PII-redacted via apply_pii_policy
    unique_counts: dict[str, int]          # column_name → n_unique
    null_counts: dict[str, int]            # column_name → n_nulls
    pii_columns: list[ColumnClassificationOut] = []  # for reference


class AnalysisPlanRequest(BaseModel):
    """Client sends column mapping + metadata for GPT-4o to generate analysis plan."""
    company_name: str
    column_mapping: dict                   # GPT-4o-generated mapping
    dtypes: dict[str, str]                 # filtered (PII excluded from analysis plan)
    shape: list[int]
    sample_rows: list[dict]                # PII-redacted
    unique_counts: dict[str, int]          # filtered


class FeedbackPayload(BaseModel):
    """PII-scrubbed feedback text organized by department."""
    feedback_by_dept: dict[str, list[str]]  # dept → list of scrubbed texts
    total_entries: int = 0


class InsightRequest(BaseModel):
    """
    The main payload from client → provider for insight generation.
    Contains ONLY aggregated/anonymized data — no raw rows.
    """
    company_name: str
    analysis_mode: str = "quick"           # "quick" or "deep"

    # Schema metadata (column names + types, no raw values)
    target_info: dict                      # target_col, target_type, target_description
    column_mapping: dict                   # GPT-4o-generated column roles

    # Pre-computed structured analysis results (aggregated)
    structured_results: dict               # step_id → {type, description, params, result}
    department_stats: list[dict]           # [{department, headcount, target_rate, ...}]
    group_by_dept: list[dict]              # department-level target metric breakdowns
    risk_factors: list[dict]               # top correlations [{feature, correlation}]

    # ML results (deep mode; already aggregated, hashed IDs)
    ml_results: Optional[dict] = None      # feature_importance, risk_scores, clustering, ...
    ml_skip_reason: Optional[str] = None

    # PII-scrubbed feedback text
    feedback: FeedbackPayload

    # PII transparency
    pii_handling_summary: Optional[list[PiiHandlingEntry]] = None


class AskRequest(BaseModel):
    """Interactive Q&A request — only cached aggregated context sent."""
    company_name: str
    question: str
    conversation_history: list[dict] = []
    insights_context: str = ""             # Cached executive summary + aggregated context


class EmbedRequest(BaseModel):
    """Client sends PII-scrubbed text for embedding via provider's OpenAI key."""
    texts: list[str]
    model: str = "text-embedding-3-small"


class PiiValidationRequest(BaseModel):
    """Client sends column classifications for AI review (names + dtypes only)."""
    classifications: list[dict]


# ═════════════════════════════════════════════════════════════════════
#  Server → Client responses
# ═════════════════════════════════════════════════════════════════════

class InsightResponse(BaseModel):
    """Full response from the insight generation pipeline."""
    company_name: str
    executive_summary: str

    target_description: str = ""
    domain_context: dict = {}

    analysis_mode: str = "quick"

    kpis: list[dict] = []
    sections: list[AnalysisSection] = []
    recommendations: list[dict] = []

    ml_results: Optional[MLResults] = None
    ml_skip_reason: Optional[str] = None

    pii_handling_summary: Optional[list[dict]] = None


class AskResponse(BaseModel):
    answer: str
    chart: Optional[ChartSpec] = None
    sources: list[str] = []


class EmbedResponse(BaseModel):
    """Embeddings returned from the provider's embed proxy."""
    embeddings: list[list[float]]


class PiiValidationResponse(BaseModel):
    """AI-validated PII classifications (can only tighten)."""
    columns: list[dict]


# ═════════════════════════════════════════════════════════════════════
#  Client-only models (not sent to server — local UI/data handling)
# ═════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    success: bool
    message: str
    row_count: Optional[int] = None


class PiiDetectionOut(BaseModel):
    pii_type: str
    original: str
    start: int
    end: int


class FeedbackSampleOut(BaseModel):
    original_text: str
    scrubbed_text: str
    pii_detections: list[PiiDetectionOut] = []
    metadata: dict = {}


class SchemaReviewResponse(BaseModel):
    """Returned by the client's local schema review endpoint."""
    company_name: str
    row_count: int
    column_count: int
    columns: list[ColumnClassificationOut]
    pii_summary: dict
    sample_rows: list[dict]
    feedback_columns: list[ColumnClassificationOut] = []
    feedback_samples: list[FeedbackSampleOut] = []
    feedback_entry_count: int = 0
    feedback_text_pii_count: int = 0


class ApproveSchemaRequest(BaseModel):
    company_name: str
    columns: list[ColumnClassificationOut]
    feedback_columns: list[ColumnClassificationOut] = []
