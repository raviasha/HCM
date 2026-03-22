"""
Data service for structured HCM data.

Abstracts all data access behind plain dict/list return types so the
API layer has zero dependency on pandas. This enables a clean swap to
DuckDB or any other backend for production scale (100K+ rows).

The target variable (attrition flag, engagement score, performance rating,
etc.) is detected dynamically from the schema mapping — nothing is
hardcoded to a single domain.

Current implementation: PandasBackend (in-memory DataFrame).
"""

from __future__ import annotations

import io
import time
from typing import Protocol, Optional, Any

import numpy as np
import pandas as pd

# Maximum time-to-live for uploaded datasets (seconds) — 24 hours
_TTL_SECONDS = 24 * 60 * 60


# ── Abstract Backend Protocol ────────────────────────────────────────

class DataBackend(Protocol):
    """
    Protocol defining the data access interface.
    All methods return plain dicts/lists — no pandas types leak out.
    """

    def load(self, file_content: bytes, company_name: str) -> int: ...
    def is_loaded(self, company_name: str) -> bool: ...
    def has_mapping(self, company_name: str) -> bool: ...
    def get_schema_metadata(self, company_name: str) -> dict: ...
    def set_schema_mapping(self, company_name: str, mapping: dict) -> None: ...
    def get_target_info(self, company_name: str) -> dict: ...
    def execute_analysis_plan(self, company_name: str, plan: list[dict]) -> dict[str, Any]: ...
    def query(self, company_name: str, query_type: str, factor: str = "") -> list[dict] | dict: ...


# ── Pandas Backend ───────────────────────────────────────────────────

class PandasBackend:
    """In-memory pandas backend. Suitable for datasets up to ~100K rows."""

    def __init__(self):
        self._datasets: dict[str, pd.DataFrame] = {}
        self._mappings: dict[str, dict] = {}
        self._load_times: dict[str, float] = {}
        self._pii_classifications: dict[str, list] = {}  # company → list[ColumnClassification]
        self._feedback_pii_classifications: dict[str, list] = {}  # company → list[ColumnClassification]

    def _expire_stale(self) -> None:
        """Remove datasets older than _TTL_SECONDS."""
        now = time.time()
        stale = [k for k, t in self._load_times.items() if now - t > _TTL_SECONDS]
        for key in stale:
            self._datasets.pop(key, None)
            self._mappings.pop(key, None)
            self._load_times.pop(key, None)
            self._pii_classifications.pop(key, None)
            self._feedback_pii_classifications.pop(key, None)

    def delete(self, company_name: str) -> bool:
        """Delete all in-memory data for a company. Returns True if data existed."""
        existed = company_name in self._datasets
        self._datasets.pop(company_name, None)
        self._mappings.pop(company_name, None)
        self._load_times.pop(company_name, None)
        self._pii_classifications.pop(company_name, None)
        self._feedback_pii_classifications.pop(company_name, None)
        return existed

    def load(self, file_content: bytes, company_name: str) -> int:
        self._expire_stale()
        df = pd.read_csv(io.BytesIO(file_content))
        self._datasets[company_name] = df
        self._mappings.pop(company_name, None)
        self._pii_classifications.pop(company_name, None)
        self._feedback_pii_classifications.pop(company_name, None)
        self._load_times[company_name] = time.time()
        return len(df)

    def is_loaded(self, company_name: str) -> bool:
        self._expire_stale()
        return company_name in self._datasets

    def has_mapping(self, company_name: str) -> bool:
        return company_name in self._mappings

    # ── PII classification storage ───────────────────────────────────

    def set_pii_classification(self, company_name: str, classifications: list) -> None:
        """Store approved PII classifications for a company."""
        self._pii_classifications[company_name] = classifications

    def get_pii_classification(self, company_name: str) -> list | None:
        """Retrieve stored PII classifications for a company."""
        return self._pii_classifications.get(company_name)

    def has_pii_classification(self, company_name: str) -> bool:
        return company_name in self._pii_classifications

    # ── Feedback PII classification storage ──────────────────────────

    def set_feedback_pii_classification(self, company_name: str, classifications: list) -> None:
        """Store approved feedback key PII classifications."""
        self._feedback_pii_classifications[company_name] = classifications

    def get_feedback_pii_classification(self, company_name: str) -> list | None:
        """Retrieve stored feedback key PII classifications."""
        return self._feedback_pii_classifications.get(company_name)

    def _df(self, company_name: str) -> pd.DataFrame:
        if company_name not in self._datasets:
            raise ValueError(f"No data loaded for company: {company_name}")
        return self._datasets[company_name]

    def _mapping(self, company_name: str) -> dict:
        if company_name not in self._mappings:
            raise ValueError(
                f"No schema mapping for company: {company_name}. "
                "Run schema analysis first."
            )
        return self._mappings[company_name]

    # ── Schema discovery ─────────────────────────────────────────────

    def get_schema_metadata(self, company_name: str) -> dict:
        """Extract column metadata for GPT-4o schema analysis."""
        df = self._df(company_name)
        sample = df.head(5).fillna("").to_dict(orient="records")
        desc = {}
        for col, stats in df.describe(include="all").to_dict().items():
            desc[col] = {k: (v if v is not None and v == v else None)
                         for k, v in stats.items()}
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "sample_rows": sample,
            "numeric_summary": desc,
            "null_counts": {col: int(v) for col, v in df.isnull().sum().items()},
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
        }

    def set_schema_mapping(self, company_name: str, mapping: dict) -> None:
        """Apply GPT-4o's column mapping and derive computed columns."""
        self._mappings[company_name] = mapping
        df = self._df(company_name)

        # Create binary target flag (for binary targets like attrition)
        target_col = mapping.get("target_col")
        target_type = mapping.get("target_type")
        if target_col and target_col in df.columns and target_type == "binary":
            yes_vals = [
                v.lower()
                for v in mapping.get("target_positive_values", ["yes", "1", "true"])
            ]
            df["_target_flag"] = (
                df[target_col].astype(str).str.strip().str.lower()
                .isin(yes_vals).astype(int)
            )
        elif target_col and target_col in df.columns and target_type == "numeric":
            # For numeric targets, copy the column as-is
            df["_target_flag"] = pd.to_numeric(df[target_col], errors="coerce")

        # Create flags for identified binary columns
        for col, info in mapping.get("binary_cols", {}).items():
            if col in df.columns:
                yes_vals = [
                    v.lower()
                    for v in info.get("yes_values", ["yes", "1", "true"])
                ]
                df[f"_{col}_flag"] = (
                    df[col].astype(str).str.strip().str.lower()
                    .isin(yes_vals).astype(int)
                )

        self._datasets[company_name] = df

    def get_target_info(self, company_name: str) -> dict:
        """Return metadata about the detected target variable."""
        m = self._mapping(company_name)
        return {
            "target_col": m.get("target_col"),
            "target_type": m.get("target_type"),
            "target_description": m.get("target_description", ""),
            "has_target": m.get("target_col") is not None,
        }

    # ── Generic query methods (used by Ask AI tool calling) ──────────

    def get_summary_kpis(self, company_name: str) -> dict:
        df = self._df(company_name)
        m = self._mapping(company_name)
        target_type = m.get("target_type")

        kpis: dict = {"total_headcount": int(len(df))}

        if "_target_flag" in df.columns:
            if target_type == "binary":
                kpis["target_rate"] = round(float(df["_target_flag"].mean()), 4)
            else:
                kpis["target_mean"] = round(float(df["_target_flag"].mean()), 4)

        eng_col = m.get("engagement_col")
        if eng_col and eng_col in df.columns:
            kpis["avg_engagement"] = round(float(df[eng_col].mean()), 2)

        tenure_col = m.get("tenure_col")
        if tenure_col and tenure_col in df.columns:
            kpis["avg_tenure"] = round(float(df[tenure_col].mean()), 1)

        return kpis

    def get_group_by_department(self, company_name: str) -> list[dict]:
        """Target metric grouped by department."""
        m = self._mapping(company_name)
        dept_col = m.get("department_col")
        if not dept_col:
            return []
        return self._group_by_target(company_name, dept_col)

    def get_risk_factors(self, company_name: str) -> list[dict]:
        """Correlations of numeric columns with the target variable."""
        df = self._df(company_name)
        m = self._mapping(company_name)

        if "_target_flag" not in df.columns:
            return []

        numeric_cols = list(m.get("numeric_cols", []))
        for col in m.get("binary_cols", {}):
            flag_col = f"_{col}_flag"
            if flag_col in df.columns:
                numeric_cols.append(flag_col)

        correlations = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            try:
                corr = float(
                    df[col].astype(float).corr(df["_target_flag"].astype(float))
                )
                if not np.isnan(corr):
                    label = col.replace("_flag", "").lstrip("_")
                    correlations.append({"factor": label, "correlation": round(corr, 4)})
            except (ValueError, TypeError):
                continue

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations[:12]

    def get_department_stats(self, company_name: str) -> list[dict]:
        df = self._df(company_name)
        m = self._mapping(company_name)
        dept_col = m.get("department_col", "Department")
        id_col = m.get("employee_id_col", df.columns[0])
        target_type = m.get("target_type")

        if dept_col not in df.columns:
            return []

        count_col = id_col if id_col in df.columns else df.columns[0]
        agg_dict: dict = {"headcount": (count_col, "count")}

        if "_target_flag" in df.columns:
            if target_type == "binary":
                agg_dict["target_count"] = ("_target_flag", "sum")
            else:
                agg_dict["target_mean"] = ("_target_flag", "mean")

        tenure_col = m.get("tenure_col")
        if tenure_col and tenure_col in df.columns:
            agg_dict["avg_tenure"] = (tenure_col, "mean")

        satisfaction_cols = m.get("satisfaction_cols", [])
        first_sat = next((c for c in satisfaction_cols if c in df.columns), None)
        if first_sat:
            agg_dict["avg_satisfaction"] = (first_sat, "mean")

        eng_col = m.get("engagement_col")
        if eng_col and eng_col in df.columns:
            agg_dict["avg_engagement"] = (eng_col, "mean")

        grouped = df.groupby(dept_col).agg(**agg_dict).reset_index()

        if "target_count" in grouped.columns:
            grouped["target_rate"] = round(
                grouped["target_count"] / grouped["headcount"], 4
            )

        grouped.rename(columns={dept_col: "department"}, inplace=True)
        grouped = grouped.round(4)
        return grouped.to_dict(orient="records")

    def get_group_by_factor(self, company_name: str, factor: str) -> list[dict]:
        """Target metric bucketed by a specific factor column."""
        return self._group_by_target(company_name, factor)

    # ── Internal helpers ─────────────────────────────────────────────

    def _group_by_target(self, company_name: str, group_col: str) -> list[dict]:
        """Group target metric by a categorical or bucketed column."""
        df = self._df(company_name)
        m = self._mapping(company_name)
        target_type = m.get("target_type")
        id_col = m.get("employee_id_col", df.columns[0])
        count_col = id_col if id_col in df.columns else df.columns[0]

        if group_col not in df.columns:
            return []

        col = df[group_col]
        use_buckets = col.dtype not in ["object", "bool"] and col.nunique() > 20

        if use_buckets:
            df_temp = df.copy()
            df_temp["_bucket"] = pd.qcut(col, q=4, duplicates="drop").astype(str)
            work_col = "_bucket"
            work_df = df_temp
        else:
            work_col = group_col
            work_df = df

        agg: dict = {"headcount": (count_col, "count")}
        if "_target_flag" in work_df.columns:
            if target_type == "binary":
                agg["target_count"] = ("_target_flag", "sum")
            else:
                agg["target_mean"] = ("_target_flag", "mean")

        grouped = work_df.groupby(work_col).agg(**agg).reset_index()

        if "target_count" in grouped.columns:
            grouped["target_rate"] = round(
                grouped["target_count"] / grouped["headcount"], 4
            )

        grouped.rename(columns={work_col: "group"}, inplace=True)
        return grouped.to_dict(orient="records")

    # ── Generic analysis plan executor ───────────────────────────────

    def execute_analysis_plan(
        self, company_name: str, plan: list[dict]
    ) -> dict[str, Any]:
        """
        Execute a GPT-4o-generated analysis plan.
        Each step: id, type, description, params.
        Returns dict mapping step id → result.
        """
        results: dict[str, Any] = {}
        for step in plan:
            step_id = step["id"]
            step_type = step["type"]
            params = step.get("params", {})
            try:
                if step_type == "summary_stats":
                    results[step_id] = self._exec_summary_stats(company_name, params)
                elif step_type in ("group_by_target", "group_attrition"):
                    results[step_id] = self._exec_group_by_target(company_name, params)
                elif step_type == "correlations":
                    results[step_id] = self._exec_correlations(company_name, params)
                elif step_type == "binary_split":
                    results[step_id] = self._exec_binary_split(company_name, params)
                elif step_type == "distribution":
                    results[step_id] = self._exec_distribution(company_name, params)
                else:
                    results[step_id] = {"error": f"Unknown type: {step_type}"}
            except Exception as e:
                results[step_id] = {"error": str(e)}
        return results

    def _exec_summary_stats(self, company_name: str, params: dict) -> dict:
        df = self._df(company_name)
        m = self._mapping(company_name)
        target_type = m.get("target_type")

        result: dict = {"total_headcount": int(len(df))}

        if "_target_flag" in df.columns:
            if target_type == "binary":
                result["target_rate"] = round(float(df["_target_flag"].mean()), 4)
            else:
                result["target_mean"] = round(float(df["_target_flag"].mean()), 4)

        for col in params.get("avg_cols", []):
            if col in df.columns:
                result[f"avg_{col}"] = round(float(df[col].mean()), 2)

        return result

    def _exec_group_by_target(self, company_name: str, params: dict) -> list[dict]:
        return self._group_by_target(company_name, params["group_by"])

    def _exec_correlations(self, company_name: str, params: dict) -> list[dict]:
        df = self._df(company_name)

        if "_target_flag" not in df.columns:
            return []

        correlations = []
        for col in params.get("columns", []):
            if col not in df.columns:
                continue
            try:
                corr = float(
                    df[col].astype(float).corr(df["_target_flag"].astype(float))
                )
                if not np.isnan(corr):
                    correlations.append({"factor": col, "correlation": round(corr, 4)})
            except (ValueError, TypeError):
                continue

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations

    def _exec_binary_split(self, company_name: str, params: dict) -> dict:
        df = self._df(company_name)
        m = self._mapping(company_name)
        target_type = m.get("target_type")
        split_col = params["split_by"]
        yes_vals = [v.lower() for v in params.get("yes_values", ["yes", "1", "true"])]

        if split_col not in df.columns or "_target_flag" not in df.columns:
            return {"split_by": split_col, "yes_target": 0.0, "no_target": 0.0,
                    "yes_headcount": 0, "no_headcount": 0}

        flag = df[split_col].astype(str).str.strip().str.lower().isin(yes_vals)
        yes_df = df[flag]
        no_df = df[~flag]

        if target_type == "binary":
            yes_val = round(float(yes_df["_target_flag"].mean()) if len(yes_df) > 0 else 0.0, 4)
            no_val = round(float(no_df["_target_flag"].mean()) if len(no_df) > 0 else 0.0, 4)
        else:
            yes_val = round(float(yes_df["_target_flag"].mean()) if len(yes_df) > 0 else 0.0, 4)
            no_val = round(float(no_df["_target_flag"].mean()) if len(no_df) > 0 else 0.0, 4)

        return {
            "split_by": split_col,
            "yes_target": yes_val,
            "no_target": no_val,
            "yes_headcount": int(len(yes_df)),
            "no_headcount": int(len(no_df)),
        }

    def _exec_distribution(self, company_name: str, params: dict) -> list[dict]:
        df = self._df(company_name)
        col_name = params["column"]
        bins = params.get("bins", 4)

        if col_name not in df.columns:
            return []

        m = self._mapping(company_name)
        target_type = m.get("target_type")
        id_col = m.get("employee_id_col", df.columns[0])
        count_col = id_col if id_col in df.columns else df.columns[0]

        df_temp = df.copy()
        df_temp["_bucket"] = pd.qcut(
            df_temp[col_name], q=bins, duplicates="drop"
        ).astype(str)

        agg: dict = {"headcount": (count_col, "count")}
        if "_target_flag" in df_temp.columns:
            if target_type == "binary":
                agg["target_count"] = ("_target_flag", "sum")
            else:
                agg["target_mean"] = ("_target_flag", "mean")

        grouped = df_temp.groupby("_bucket").agg(**agg).reset_index()
        grouped.rename(columns={"_bucket": "bucket"}, inplace=True)

        if "target_count" in grouped.columns:
            grouped["target_rate"] = round(
                grouped["target_count"] / grouped["headcount"], 4
            )

        return grouped.to_dict(orient="records")

    # ── Query dispatch (used by Ask AI tool calling) ─────────────────

    def query(self, company_name: str, query_type: str, factor: str = "") -> list[dict] | dict:
        """Dispatch a structured data query by type."""
        dispatch = {
            "kpis": lambda: self.get_summary_kpis(company_name),
            "group_by_department": lambda: self.get_group_by_department(company_name),
            "risk_factors": lambda: self.get_risk_factors(company_name),
            "department_stats": lambda: self.get_department_stats(company_name),
            "group_by_factor": lambda: self.get_group_by_factor(company_name, factor),
        }
        handler = dispatch.get(query_type)
        if handler is None:
            return {"error": f"Unknown query_type: {query_type}"}
        return handler()


# ── Module-level singleton ───────────────────────────────────────────

_backend: Optional[PandasBackend] = None


def get_backend() -> PandasBackend:
    global _backend
    if _backend is None:
        _backend = PandasBackend()
    return _backend
