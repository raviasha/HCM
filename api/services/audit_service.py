"""
Audit logging service for GDPR Article 30 compliance.

Logs data processing activities without storing any PII.
Records: action type, company name, timestamp, metadata (row counts, model used, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("audit")

# In-memory audit log (replace with persistent store in production)
_audit_log: list[dict] = []


def log_event(
    action: str,
    company_name: str,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Record a data processing event.

    Args:
        action: Type of action (e.g., "data_upload", "insight_generation", "data_deletion")
        company_name: Company identifier (not PII)
        details: Additional context — must NOT contain PII
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "company_name": company_name,
        "details": details or {},
    }
    _audit_log.append(entry)
    logger.info("AUDIT | %s | %s | %s", action, company_name, details or "")


def get_log(company_name: str | None = None, limit: int = 100) -> list[dict]:
    """Retrieve audit log entries, optionally filtered by company."""
    if company_name:
        entries = [e for e in _audit_log if e["company_name"] == company_name]
    else:
        entries = list(_audit_log)
    return entries[-limit:]
