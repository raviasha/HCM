"""
Pydantic models for API request/response contracts.
These models define the data shapes exchanged between
FastAPI backend and Streamlit frontend.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import List, Optional


# ── Attrition / Structured Data ──────────────────────────────────────

class DepartmentAttrition(BaseModel):
    department: str
    headcount: int
    attrition_count: int
    attrition_rate: float


class RiskFactor(BaseModel):
    factor: str
    correlation: float


class DepartmentStats(BaseModel):
    department: str
    headcount: int
    avg_tenure: float
    avg_satisfaction: float
    avg_engagement: float
    attrition_rate: float
    overtime_pct: float


class SummaryKPIs(BaseModel):
    total_headcount: int
    overall_attrition_rate: float
    avg_engagement_score: float
    avg_tenure: float


class OvertimeAnalysis(BaseModel):
    overtime_attrition_rate: float
    no_overtime_attrition_rate: float
    overtime_headcount: int
    no_overtime_headcount: int


# ── Voice of Employee / Feedback ─────────────────────────────────────

class FeedbackEntry(BaseModel):
    feedback_id: int
    employee_id: int
    feedback_type: str
    date: str
    question_prompt: str
    response_text: str
    department: str


class SentimentResult(BaseModel):
    department: str
    positive: int
    neutral: int
    negative: int
    avg_score: float


class ThemeResult(BaseModel):
    theme: str
    count: int
    sample_quotes: list[str]


# ── AI Insights ──────────────────────────────────────────────────────

class RetentionRecommendation(BaseModel):
    department: str
    risk_level: str  # "High", "Medium", "Low"
    recommendations: list[str]
    key_issues: list[str]


class CorrelationInsight(BaseModel):
    department: str
    attrition_rate: float
    avg_sentiment_score: float
    narrative: str


class InsightResponse(BaseModel):
    """Full response from the insight generation pipeline."""
    company_name: str
    executive_summary: str
    summary_kpis: SummaryKPIs
    department_attrition: list[DepartmentAttrition]
    department_stats: list[DepartmentStats]
    risk_factors: list[RiskFactor]
    overtime_analysis: OvertimeAnalysis
    sentiment_by_department: list[SentimentResult]
    themes: list[ThemeResult]
    correlations: list[CorrelationInsight]
    retention_recommendations: list[RetentionRecommendation]


# ── API Requests ─────────────────────────────────────────────────────

class GenerateInsightsRequest(BaseModel):
    company_name: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    row_count: Optional[int] = None
