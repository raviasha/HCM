"""
API client module — all HTTP calls from Streamlit to FastAPI.
Centralizes request logic and error handling.
"""

import httpx
import streamlit as st
from typing import Any, Optional

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

def upload_attrition_csv(file_bytes: bytes, filename: str, company_name: str) -> dict:
    """Upload attrition CSV to FastAPI."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/attrition/upload"),
            files={"file": (filename, file_bytes, "text/csv")},
            data={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


def upload_feedback_json(file_bytes: bytes, filename: str, company_name: str) -> dict:
    """Upload VoE feedback JSON to FastAPI."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/feedback/upload"),
            files={"file": (filename, file_bytes, "application/json")},
            data={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


# ── Insights endpoint ────────────────────────────────────────────────

def generate_insights(company_name: str) -> dict:
    """Trigger the full AI insight generation pipeline."""
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            _url("/api/insights/generate"),
            json={"company_name": company_name},
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


# ── Individual data endpoints ────────────────────────────────────────

def get_attrition_by_department(company_name: str) -> list[dict]:
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            _url("/api/attrition/by-department"),
            params={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()


def get_kpis(company_name: str) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(
            _url("/api/attrition/kpis"),
            params={"company_name": company_name},
        )
        resp.raise_for_status()
        return resp.json()
