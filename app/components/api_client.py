"""
API client module — all HTTP calls from Streamlit to FastAPI.
Centralizes request logic and error handling.
"""

import httpx
import streamlit as st
from typing import Optional

TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # 5 min for AI generation


def _base_url() -> str:
    """Resolve API base URL: st.secrets → env var → localhost fallback."""
    try:
        return st.secrets["API_BASE_URL"].rstrip("/")
    except (KeyError, AttributeError):
        pass
    from config.settings import API_BASE_URL
    return API_BASE_URL.rstrip("/")


def _url(path: str) -> str:
    return f"{_base_url()}{path}"


def check_api_health() -> tuple[bool, str]:
    """Return (is_healthy, message). Used to surface config errors early."""
    url = _base_url()
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.get(f"{url}/health")
            resp.raise_for_status()
            return True, url
    except httpx.ConnectError:
        return False, (
            f"Cannot reach the API at **{url}**.\n\n"
            "If you are on Streamlit Cloud, add `API_BASE_URL` to your app "
            "secrets pointing to your Render backend URL "
            "(e.g. `https://hcm-api.onrender.com`)."
        )
    except Exception as e:
        return False, f"API health check failed: {e}"


# ── Upload endpoints ─────────────────────────────────────────────────

def upload_csv(file_bytes: bytes, filename: str, company_name: str) -> dict:
    """Upload structured CSV data to FastAPI."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/data/upload-csv"),
            files={"file": (filename, file_bytes, "text/csv")},
            data={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


def upload_feedback_json(file_bytes: bytes, filename: str, company_name: str) -> dict:
    """Upload qualitative feedback JSON to FastAPI."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/data/upload-json"),
            files={"file": (filename, file_bytes, "application/json")},
            data={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


# ── Insights endpoint ────────────────────────────────────────────────

def generate_insights(company_name: str, analysis_mode: str = "quick") -> dict:
    """Trigger the full AI insight generation pipeline."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/insights/generate"),
            json={"company_name": company_name, "analysis_mode": analysis_mode},
        )
        resp.raise_for_status()
        return resp.json()


def get_insights_status(company_name: str) -> dict:
    """Check if insights are available."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            _url("/api/insights/status"),
            params={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


def get_cached_insights(company_name: str) -> Optional[dict]:
    """Fetch cached insights if available."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                _url("/api/insights/cached"),
                params={"company_name": company_name},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError:
        return None


def ask_question(company_name: str, question: str, conversation_history: list[dict] = None) -> dict:
    """Send a follow-up question to the AI Q&A endpoint."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/insights/ask"),
            json={
                "company_name": company_name,
                "question": question,
                "conversation_history": conversation_history or [],
            },
        )
        resp.raise_for_status()
        return resp.json()


def delete_company_data(company_name: str) -> dict:
    """Delete all data for a company (GDPR right to erasure)."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.delete(
            _url(f"/api/data/{company_name}"),
        )
        resp.raise_for_status()
        return resp.json()
