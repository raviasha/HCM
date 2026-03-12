"""
Data service for structured employee/attrition data.

Abstracts all data access behind plain dict/list return types so the
API layer has zero dependency on pandas. This enables a clean swap to
DuckDB or any other backend for production scale (100K+ rows).

Current implementation: PandasBackend (in-memory DataFrame).
Production upgrade: Swap PandasBackend for DuckDBBackend — same interface,
backed by DuckDB for 100K+ rows. Change DATA_BACKEND in config/settings.py.
"""

from __future__ import annotations

import io
from typing import Protocol, Optional

import numpy as np
import pandas as pd


# ── Abstract Backend Protocol ────────────────────────────────────────

class DataBackend(Protocol):
    """
    Protocol defining the data access interface.
    All methods return plain dicts/lists — no pandas types leak out.
    """

    def load(self, file_content: bytes, company_name: str) -> int:
        """Load CSV data. Returns row count."""
        ...

    def is_loaded(self, company_name: str) -> bool:
        ...

    def get_summary_kpis(self, company_name: str) -> dict:
        ...

    def get_attrition_by_department(self, company_name: str) -> list[dict]:
        ...

    def get_risk_factors(self, company_name: str) -> list[dict]:
        ...

    def get_department_stats(self, company_name: str) -> list[dict]:
        ...

    def get_overtime_analysis(self, company_name: str) -> dict:
        ...

    def get_attrition_by_factor(self, company_name: str, factor: str) -> list[dict]:
        ...


# ── Pandas Backend ───────────────────────────────────────────────────

class PandasBackend:
    """In-memory pandas backend. Suitable for datasets up to ~100K rows."""

    def __init__(self):
        self._datasets: dict[str, pd.DataFrame] = {}

    def load(self, file_content: bytes, company_name: str) -> int:
        df = pd.read_csv(io.BytesIO(file_content))
        # Normalize attrition column to binary int
        if "Attrition" in df.columns:
            df["_attrition_flag"] = (
                df["Attrition"].astype(str).str.strip().str.lower().map(
                    {"yes": 1, "1": 1, "true": 1}
                ).fillna(0).astype(int)
            )
        self._datasets[company_name] = df
        return len(df)

    def is_loaded(self, company_name: str) -> bool:
        return company_name in self._datasets

    def _df(self, company_name: str) -> pd.DataFrame:
        if company_name not in self._datasets:
            raise ValueError(f"No data loaded for company: {company_name}")
        return self._datasets[company_name]

    def get_summary_kpis(self, company_name: str) -> dict:
        df = self._df(company_name)
        return {
            "total_headcount": int(len(df)),
            "overall_attrition_rate": round(
                float(df["_attrition_flag"].mean()), 4
            ),
            "avg_engagement_score": round(
                float(df["EngagementScore"].mean()), 2
            ) if "EngagementScore" in df.columns else 0.0,
            "avg_tenure": round(
                float(df["YearsAtCompany"].mean()), 1
            ) if "YearsAtCompany" in df.columns else 0.0,
        }

    def get_attrition_by_department(self, company_name: str) -> list[dict]:
        df = self._df(company_name)
        grouped = df.groupby("Department").agg(
            headcount=("EmployeeID", "count"),
            attrition_count=("_attrition_flag", "sum"),
        ).reset_index()
        grouped["attrition_rate"] = round(
            grouped["attrition_count"] / grouped["headcount"], 4
        )
        grouped.rename(columns={"Department": "department"}, inplace=True)
        return grouped.to_dict(orient="records")

    def get_risk_factors(self, company_name: str) -> list[dict]:
        df = self._df(company_name)
        # Compute point-biserial / Pearson correlation with attrition
        numeric_cols = [
            "Age", "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "TotalWorkingYears", "DistanceFromHome",
            "PerformanceRating", "JobSatisfaction", "EnvironmentSatisfaction",
            "WorkLifeBalance", "RelationshipSatisfaction", "TrainingTimesLastYear",
            "StockOptionLevel", "PercentSalaryHike", "EngagementScore",
        ]
        # Add OverTime as numeric
        df_temp = df.copy()
        if "OverTime" in df_temp.columns:
            df_temp["OverTime_flag"] = (
                df_temp["OverTime"].astype(str).str.lower().map(
                    {"yes": 1, "1": 1, "true": 1}
                ).fillna(0).astype(int)
            )
            numeric_cols.append("OverTime_flag")

        available = [c for c in numeric_cols if c in df_temp.columns]
        correlations = []
        for col in available:
            try:
                corr = float(df_temp[col].astype(float).corr(
                    df_temp["_attrition_flag"].astype(float)
                ))
                if not np.isnan(corr):
                    label = col.replace("_flag", "")
                    correlations.append({
                        "factor": label,
                        "correlation": round(corr, 4),
                    })
            except (ValueError, TypeError):
                continue

        # Sort by absolute correlation, descending
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return correlations[:12]

    def get_department_stats(self, company_name: str) -> list[dict]:
        df = self._df(company_name)
        agg_dict = {
            "headcount": ("EmployeeID", "count"),
            "attrition_count": ("_attrition_flag", "sum"),
        }

        optional_cols = {
            "avg_tenure": ("YearsAtCompany", "mean"),
            "avg_satisfaction": ("JobSatisfaction", "mean"),
            "avg_engagement": ("EngagementScore", "mean"),
        }
        for key, (col, func) in optional_cols.items():
            if col in df.columns:
                agg_dict[key] = (col, func)

        grouped = df.groupby("Department").agg(**agg_dict).reset_index()
        grouped["attrition_rate"] = round(
            grouped["attrition_count"] / grouped["headcount"], 4
        )

        # Overtime percentage per department
        if "OverTime" in df.columns:
            ot = df.copy()
            ot["ot_flag"] = (
                ot["OverTime"].astype(str).str.lower().map(
                    {"yes": 1, "1": 1, "true": 1}
                ).fillna(0).astype(int)
            )
            ot_pct = ot.groupby("Department")["ot_flag"].mean().reset_index()
            ot_pct.columns = ["Department", "overtime_pct"]
            grouped = grouped.merge(ot_pct, on="Department", how="left")
        else:
            grouped["overtime_pct"] = 0.0

        grouped.rename(columns={"Department": "department"}, inplace=True)

        # Fill missing optional columns
        for col in ["avg_tenure", "avg_satisfaction", "avg_engagement"]:
            if col not in grouped.columns:
                grouped[col] = 0.0

        grouped = grouped.round(4)
        return grouped.to_dict(orient="records")

    def get_overtime_analysis(self, company_name: str) -> dict:
        df = self._df(company_name)
        if "OverTime" not in df.columns:
            return {
                "overtime_attrition_rate": 0.0,
                "no_overtime_attrition_rate": 0.0,
                "overtime_headcount": 0,
                "no_overtime_headcount": 0,
            }

        ot_yes = df[df["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"])]
        ot_no = df[~df["OverTime"].astype(str).str.lower().isin(["yes", "1", "true"])]

        return {
            "overtime_attrition_rate": round(
                float(ot_yes["_attrition_flag"].mean()) if len(ot_yes) > 0 else 0.0, 4
            ),
            "no_overtime_attrition_rate": round(
                float(ot_no["_attrition_flag"].mean()) if len(ot_no) > 0 else 0.0, 4
            ),
            "overtime_headcount": int(len(ot_yes)),
            "no_overtime_headcount": int(len(ot_no)),
        }

    def get_attrition_by_factor(
        self, company_name: str, factor: str
    ) -> list[dict]:
        """Get attrition rate bucketed by a specific factor."""
        df = self._df(company_name)
        if factor not in df.columns:
            return []

        col = df[factor]
        if col.dtype in ["object", "bool"]:
            grouped = df.groupby(factor).agg(
                headcount=("EmployeeID", "count"),
                attrition_count=("_attrition_flag", "sum"),
            ).reset_index()
        else:
            # Bucket numeric into quartiles
            df_temp = df.copy()
            df_temp["_bucket"] = pd.qcut(
                col, q=4, duplicates="drop"
            ).astype(str)
            grouped = df_temp.groupby("_bucket").agg(
                headcount=("EmployeeID", "count"),
                attrition_count=("_attrition_flag", "sum"),
            ).reset_index()
            grouped.rename(columns={"_bucket": factor}, inplace=True)

        grouped["attrition_rate"] = round(
            grouped["attrition_count"] / grouped["headcount"], 4
        )
        return grouped.to_dict(orient="records")


# ── Module-level singleton ───────────────────────────────────────────

_backend: Optional[PandasBackend] = None


def get_backend() -> PandasBackend:
    global _backend
    if _backend is None:
        _backend = PandasBackend()
    return _backend
