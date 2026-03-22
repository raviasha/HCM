"""
HTTP client for the provider backend.

All calls send ONLY anonymized/aggregated data.
Raw employee data never leaves the client container.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from client.config.settings import PROVIDER_URL

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def _url(path: str) -> str:
    return f"{PROVIDER_URL}{path}"


# ── Schema Analysis ──────────────────────────────────────────────────

def analyze_schema(
    company_name: str,
    dtypes: dict,
    shape: list,
    sample_rows: list[dict],
    unique_counts: dict,
    null_counts: dict,
    pii_columns: list[dict] | None = None,
) -> dict:
    """Send sanitized schema metadata to provider for GPT-4o column mapping."""
    payload = {
        "company_name": company_name,
        "dtypes": dtypes,
        "shape": shape,
        "sample_rows": sample_rows,
        "unique_counts": unique_counts,
        "null_counts": null_counts,
        "pii_columns": pii_columns or [],
    }
    resp = httpx.post(_url("/api/analyze-schema"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Analysis Plan ────────────────────────────────────────────────────

def generate_plan(
    company_name: str,
    column_mapping: dict,
    dtypes: dict,
    shape: list,
    sample_rows: list[dict],
    unique_counts: dict,
) -> list[dict]:
    """Send metadata to provider for GPT-4o analysis plan generation."""
    payload = {
        "company_name": company_name,
        "column_mapping": column_mapping,
        "dtypes": dtypes,
        "shape": shape,
        "sample_rows": sample_rows,
        "unique_counts": unique_counts,
    }
    resp = httpx.post(_url("/api/generate-plan"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("analyses", [])


# ── Full Insight Generation ──────────────────────────────────────────

def generate_insights(payload: dict) -> dict:
    """
    Send aggregated/anonymized data to provider for full insight generation.
    payload matches InsightRequest schema.
    """
    resp = httpx.post(_url("/api/generate-insights"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── Ask AI ───────────────────────────────────────────────────────────

def ask_question(
    company_name: str,
    question: str,
    insights_context: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Send question + cached context to provider."""
    payload = {
        "company_name": company_name,
        "question": question,
        "conversation_history": conversation_history or [],
        "insights_context": insights_context,
    }
    resp = httpx.post(_url("/api/ask"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── PII Validation ──────────────────────────────────────────────────

def validate_pii(classifications: list[dict]) -> list[dict]:
    """Send column classifications to provider for AI review."""
    payload = {"classifications": classifications}
    resp = httpx.post(_url("/api/validate-pii"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("columns", classifications)


# ── Embed Proxy ──────────────────────────────────────────────────────

def embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Get embeddings from provider's embed proxy."""
    payload = {"texts": texts, "model": model}
    resp = httpx.post(_url("/api/embed"), json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("embeddings", [])
