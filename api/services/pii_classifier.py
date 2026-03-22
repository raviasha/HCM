"""
Deterministic PII classifier for HR/HCM datasets.

Classifies each column as one of:
  - identifier:       Unique row key (employee ID, etc.) → excluded from AI
  - direct_pii:       Names, emails, phones, SSN, etc. → excluded from AI
  - quasi_identifier:  DOB, zip code, exact salary, etc. → aggregated only
  - safe:             Department, job role, scores, etc. → passed through

Classification uses ONLY deterministic Python heuristics (regex on column
names + value-pattern sampling + uniqueness checks).  No AI is involved
in the detection step itself, so there is zero chance of non-deterministic
field omission.

GDPR data-minimisation: Python extracts the schema and classifies every
column *before* any data is sent to an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

# ── Classification types ─────────────────────────────────────────────

PiiCategory = Literal["identifier", "direct_pii", "quasi_identifier", "safe"]
Handling = Literal["exclude", "hash", "aggregate_only", "pass_through"]


@dataclass
class ColumnClassification:
    column_name: str
    dtype: str
    pii_category: PiiCategory
    handling: Handling
    confidence: Literal["high", "medium", "low"]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Column-name patterns (case-insensitive) ──────────────────────────
# Each tuple: (compiled regex, pii_category, handling, reason)

_NAME_RULES: list[tuple[re.Pattern, PiiCategory, Handling, str]] = [
    # ── Identifiers ──────────────────────────────────────────────────
    (re.compile(r"^(employee|emp|staff|worker|person|member)[\s_]?id$", re.I),
     "identifier", "exclude", "Employee identifier column"),
    (re.compile(r"^id$", re.I),
     "identifier", "exclude", "Generic row identifier"),
    (re.compile(r"(national|passport|govt|government|driver.?s?)[\s_]?(id|number|no|num)", re.I),
     "direct_pii", "exclude", "Government-issued identifier"),
    (re.compile(r"\b(ssn|social[\s_]?security|sin|tax[\s_]?id|tin)\b", re.I),
     "direct_pii", "exclude", "Tax / social security number"),

    # ── Direct PII: names ────────────────────────────────────────────
    (re.compile(r"^(full[\s_]?name|employee[\s_]?name|staff[\s_]?name|worker[\s_]?name|person[\s_]?name|name)$", re.I),
     "direct_pii", "exclude", "Person name"),
    (re.compile(r"^(first|last|middle|maiden|family|given|sur)[\s_]?name$", re.I),
     "direct_pii", "exclude", "Person name component"),

    # ── Direct PII: contact ──────────────────────────────────────────
    (re.compile(r"(e[\s_\-]?mail|email[\s_]?addr)", re.I),
     "direct_pii", "exclude", "Email address"),
    (re.compile(r"(phone|mobile|cell|fax|tel)[\s_]?(number|num|no)?", re.I),
     "direct_pii", "exclude", "Phone number"),
    (re.compile(r"(home|mailing|street|postal|residential)[\s_]?addr", re.I),
     "direct_pii", "exclude", "Physical address"),
    (re.compile(r"^address$", re.I),
     "direct_pii", "exclude", "Physical address"),

    # ── Direct PII: financial ────────────────────────────────────────
    (re.compile(r"(bank|account|routing|iban|swift)[\s_]?(number|num|no|acct)?", re.I),
     "direct_pii", "exclude", "Financial account information"),
    (re.compile(r"(credit|debit)[\s_]?card", re.I),
     "direct_pii", "exclude", "Payment card information"),

    # ── Quasi-identifiers ────────────────────────────────────────────
    (re.compile(r"(date[\s_]?of[\s_]?birth|dob|birth[\s_]?date|birthday)", re.I),
     "quasi_identifier", "aggregate_only", "Date of birth"),
    (re.compile(r"(zip[\s_]?code|postal[\s_]?code|postcode)", re.I),
     "quasi_identifier", "aggregate_only", "Postal / zip code"),
    (re.compile(r"^(city|town|state|province|county|country)$", re.I),
     "quasi_identifier", "aggregate_only", "Geographic location"),
    (re.compile(r"(hire|join|start|termination|exit|end)[\s_]?date", re.I),
     "quasi_identifier", "aggregate_only", "Employment date (re-identification risk)"),
    (re.compile(r"^(salary|annual[\s_]?salary|base[\s_]?pay|compensation|total[\s_]?comp)$", re.I),
     "quasi_identifier", "aggregate_only", "Exact compensation (re-identification risk)"),
]

# ── Value-pattern detectors ──────────────────────────────────────────

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s().]{7,}\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _sample_values(series: pd.Series, n: int = 20) -> list[str]:
    """Return up to n non-null string representations of the series."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    sample = non_null.sample(min(n, len(non_null)), random_state=42)
    return [str(v) for v in sample]


def _detect_value_pattern(values: list[str]) -> tuple[PiiCategory, Handling, str] | None:
    """Check sampled values for PII patterns. Returns classification or None."""
    if not values:
        return None

    email_hits = sum(1 for v in values if _EMAIL_RE.search(v))
    if email_hits / len(values) >= 0.5:
        return "direct_pii", "exclude", "Values match email pattern"

    phone_hits = sum(1 for v in values if _PHONE_RE.fullmatch(v.strip()))
    if phone_hits / len(values) >= 0.5:
        return "direct_pii", "exclude", "Values match phone number pattern"

    ssn_hits = sum(1 for v in values if _SSN_RE.search(v))
    if ssn_hits / len(values) >= 0.3:
        return "direct_pii", "exclude", "Values match SSN/tax-ID pattern"

    uuid_hits = sum(1 for v in values if _UUID_RE.fullmatch(v.strip()))
    if uuid_hits / len(values) >= 0.5:
        return "identifier", "exclude", "Values match UUID pattern"

    return None


# ── Main classifier ──────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> list[ColumnClassification]:
    """
    Classify every column in a DataFrame as identifier / direct_pii /
    quasi_identifier / safe using deterministic heuristics only.

    Returns one ColumnClassification per column, in column order.
    """
    n_rows = len(df)
    results: list[ColumnClassification] = []

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        classified = False

        # ── 1. Column-name pattern matching ──────────────────────────
        for pattern, category, handling, reason in _NAME_RULES:
            if pattern.search(col):
                results.append(ColumnClassification(
                    column_name=col, dtype=dtype_str,
                    pii_category=category, handling=handling,
                    confidence="high", reason=reason,
                ))
                classified = True
                break

        if classified:
            continue

        # ── 2. Value-pattern detection (string columns) ──────────────
        if df[col].dtype == "object":
            values = _sample_values(df[col])
            vp = _detect_value_pattern(values)
            if vp:
                cat, hand, reason = vp
                results.append(ColumnClassification(
                    column_name=col, dtype=dtype_str,
                    pii_category=cat, handling=hand,
                    confidence="medium",
                    reason=f"{reason} (detected in sampled values)",
                ))
                continue

        # ── 3. Uniqueness heuristic (likely identifier) ──────────────
        if n_rows >= 50:
            n_unique = df[col].nunique()
            uniqueness_ratio = n_unique / n_rows if n_rows > 0 else 0
            if uniqueness_ratio > 0.95 and df[col].dtype in ("object", "int64", "int32"):
                results.append(ColumnClassification(
                    column_name=col, dtype=dtype_str,
                    pii_category="identifier", handling="exclude",
                    confidence="medium",
                    reason=f"High uniqueness ({uniqueness_ratio:.0%}) suggests row-level identifier",
                ))
                continue

        # ── 4. Default: safe ─────────────────────────────────────────
        results.append(ColumnClassification(
            column_name=col, dtype=dtype_str,
            pii_category="safe", handling="pass_through",
            confidence="high", reason="No PII indicators detected",
        ))

    return results


def classify_feedback_keys(sample_entry: dict) -> list[ColumnClassification]:
    """
    Classify the keys of a single feedback JSON entry using the same
    name-pattern rules. Returns one ColumnClassification per key.
    """
    results: list[ColumnClassification] = []
    for key in sample_entry:
        dtype_str = type(sample_entry[key]).__name__
        classified = False
        for pattern, category, handling, reason in _NAME_RULES:
            if pattern.search(key):
                results.append(ColumnClassification(
                    column_name=key, dtype=dtype_str,
                    pii_category=category, handling=handling,
                    confidence="high", reason=reason,
                ))
                classified = True
                break
        if not classified:
            results.append(ColumnClassification(
                column_name=key, dtype=dtype_str,
                pii_category="safe", handling="pass_through",
                confidence="high", reason="No PII indicators detected",
            ))
    return results


def apply_pii_policy(
    rows: list[dict],
    classifications: list[ColumnClassification],
) -> list[dict]:
    """
    Apply PII handling policy to a list of row dicts based on stored
    classifications. This is the single deterministic sanitization point
    used by ALL LLM-calling functions.

    - exclude:        column dropped entirely
    - hash:           value replaced with first 8 chars of SHA-256
    - aggregate_only: value replaced with placeholder
    - pass_through:   value kept as-is
    """
    import hashlib

    if not rows or not classifications:
        return rows

    policy = {c.column_name: c.handling for c in classifications}

    sanitized = []
    for idx, row in enumerate(rows):
        clean: dict = {}
        for key, val in row.items():
            handling = policy.get(key, "pass_through")
            if handling == "exclude":
                continue
            elif handling == "hash":
                raw = str(val) if val is not None else ""
                clean[key] = hashlib.sha256(raw.encode()).hexdigest()[:8]
            elif handling == "aggregate_only":
                clean[key] = "[aggregated]"
            else:
                clean[key] = val
        sanitized.append(clean)
    return sanitized


# ── Free-text PII scrubber ───────────────────────────────────────────

@dataclass
class PiiDetection:
    """A single PII match found inside free text."""
    pii_type: str          # e.g. "email", "phone", "ssn", "name"
    original: str          # the matched substring
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


# Patterns for PII inside free text (order matters — longer/more specific first)
_TEXT_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_TEXT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_TEXT_PHONE_RE = re.compile(
    r"(?<!\d)"                           # not preceded by a digit
    r"(?:\+?1[\s.-]?)?"                  # optional US country code
    r"(?:\(?\d{3}\)?[\s.\-]?)"           # area code
    r"\d{3}[\s.\-]?\d{4}"               # subscriber number
    r"(?!\d)",                            # not followed by a digit
)
_TEXT_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
# Title-case name sequences (2-3 words, each starting uppercase, not common words)
_COMMON_WORDS = frozenset({
    "The", "This", "That", "What", "When", "Where", "Which", "There", "Their",
    "They", "These", "Those", "There", "Have", "Would", "Could", "Should",
    "About", "After", "Before", "Between", "During", "Through", "Each",
    "From", "Into", "Over", "Under", "With", "Without", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January",
    "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Human", "Resources",
    "Work", "Life", "Balance", "Career", "Growth", "Team", "Senior",
    "Junior", "Manager", "Director", "Vice", "President", "Contact",
    "Please", "Thank", "Hello", "Dear", "Best", "Regards", "New", "York",
    "San", "Los", "Las", "Bay", "Area", "Company", "Department",
})
_TEXT_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})(?:\s+([A-Z][a-z]{1,15}))?\b"
)


def scrub_text_pii(text: str) -> tuple[str, list[PiiDetection]]:
    """
    Detect and redact PII in free-form text using regex heuristics.

    Returns:
        (scrubbed_text, list_of_detections)
    """
    if not text:
        return text, []

    detections: list[PiiDetection] = []

    # Collect all matches with their spans (process in reverse order later)
    matches: list[tuple[int, int, str, str]] = []  # (start, end, pii_type, original)

    for m in _TEXT_SSN_RE.finditer(text):
        matches.append((m.start(), m.end(), "ssn", m.group()))

    for m in _TEXT_EMAIL_RE.finditer(text):
        matches.append((m.start(), m.end(), "email", m.group()))

    for m in _TEXT_PHONE_RE.finditer(text):
        # Skip if it overlaps with an SSN match
        if any(s <= m.start() < e or s < m.end() <= e for s, e, t, _ in matches if t == "ssn"):
            continue
        matches.append((m.start(), m.end(), "phone", m.group()))

    for m in _TEXT_CC_RE.finditer(text):
        digits_only = re.sub(r"[\s-]", "", m.group())
        if len(digits_only) >= 13:
            # Skip if it overlaps with SSN or phone
            if any(s <= m.start() < e or s < m.end() <= e for s, e, _, _ in matches):
                continue
            matches.append((m.start(), m.end(), "credit_card", m.group()))

    for m in _TEXT_NAME_RE.finditer(text):
        words = [w for w in [m.group(1), m.group(2), m.group(3)] if w]
        # Skip if ALL words are common English words
        if all(w in _COMMON_WORDS for w in words):
            continue
        # Skip if it overlaps with an already-found match
        if any(s <= m.start() < e or s < m.end() <= e for s, e, _, _ in matches):
            continue
        matches.append((m.start(), m.end(), "name", m.group()))

    if not matches:
        return text, []

    # Sort by start position descending so we can replace right-to-left
    matches.sort(key=lambda x: x[0], reverse=True)

    _REPLACEMENT = {
        "ssn": "[SSN]",
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "credit_card": "[CREDIT_CARD]",
        "name": "[NAME]",
    }

    scrubbed = text
    for start, end, pii_type, original in matches:
        detections.append(PiiDetection(
            pii_type=pii_type, original=original, start=start, end=end,
        ))
        scrubbed = scrubbed[:start] + _REPLACEMENT.get(pii_type, "[REDACTED]") + scrubbed[end:]

    # Reverse detections so they're in forward order
    detections.reverse()
    return scrubbed, detections
