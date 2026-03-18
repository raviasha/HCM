"""
ML Predictive Analytics Service.

Provides 5 ML capabilities on top of descriptive analytics:
  1. Feature Importance (with model comparison & cross-validation)
  2. Risk Scoring (predict_proba / predict)
  3. Survival Analysis (Kaplan-Meier via lifelines)
  4. Clustering (KMeans with silhouette-based k selection)
  5. What-If Scenarios (simulate shifting top features)

Data-quality steps: duplicate removal, IQR outlier capping, imputation,
scaling. Model selection: RandomForest vs LogisticRegression/Ridge with
5-fold cross-validation. All performance metrics are surfaced in results.

All models are lightweight (sklearn, lifelines) and fit within
Render free-tier constraints (~512MB RAM).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ── Main entry point ─────────────────────────────────────────────────


def run_ml_pipeline(df: pd.DataFrame, mapping: dict) -> dict:
    """
    Run the full ML pipeline and return results for all 5 capabilities.

    Pipeline steps:
      0. Duplicate removal
      1. Feature preparation (outlier capping, imputation, scaling, encoding)
      2. Train/test split (80/20)
      3. Model comparison (RF vs LR/Ridge) with 5-fold CV
      4. Feature importance from best model
      5. Risk scoring, survival analysis, clustering, what-if scenarios

    Returns dict with keys: feature_importance, risk_scores,
    survival_analysis, clustering, what_if_scenarios, model_metrics,
    data_quality.
    """
    target_col = mapping.get("target_col")
    target_type = mapping.get("target_type")

    if not target_col or target_col not in df.columns:
        logger.warning("No target column found — skipping ML pipeline")
        return _empty_results("No target variable detected in the dataset.")

    if len(df) < 50:
        logger.warning("Dataset too small (%d rows) for ML", len(df))
        return _empty_results(
            f"Dataset has only {len(df)} rows — need at least 50 for ML analysis."
        )

    # ── Step 0: Data quality ─────────────────────────────────────────
    original_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = original_rows - len(df)
    logger.info("Removed %d duplicate rows", duplicates_removed)

    data_quality = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "rows_after_dedup": len(df),
    }

    # ── Step 1: Feature preparation ──────────────────────────────────
    try:
        X, y, feature_names, cat_cols, num_cols, outliers_capped = (
            _prepare_features(df, mapping)
        )
    except Exception as e:
        logger.exception("Feature preparation failed")
        return _empty_results(f"Could not prepare features: {e}")

    data_quality["outliers_capped"] = outliers_capped
    data_quality["features_used"] = len(feature_names)
    data_quality["rows_after_cleaning"] = len(y)

    if X.shape[1] < 2:
        return _empty_results("Fewer than 2 usable features — insufficient for ML.")

    # ── Step 2: Train/test split ─────────────────────────────────────
    stratify = y if target_type == "binary" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify,
        )
    except ValueError:
        # Fallback if stratification fails (e.g. too few samples in a class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

    # ── Step 3: Model comparison with cross-validation ───────────────
    try:
        best_model, model_metrics = _compare_models(
            X_train, y_train, X_test, y_test, target_type
        )
    except Exception as e:
        logger.exception("Model comparison failed, falling back to RandomForest")
        best_model, model_metrics = _fallback_model(
            X_train, y_train, X_test, y_test, target_type
        )

    results: dict[str, Any] = {
        "model_metrics": model_metrics,
        "data_quality": data_quality,
    }

    # 4. Feature Importance (from best model, trained on full data for final predictions)
    final_model = None
    try:
        final_model, importances = _feature_importance(
            X, y, feature_names, target_type, best_model
        )
        results["feature_importance"] = importances
    except Exception as e:
        logger.exception("Feature importance failed")
        results["feature_importance"] = []

    # 5. Risk Scoring
    try:
        if final_model is not None:
            results["risk_scores"] = _risk_scoring(
                final_model, X, df, mapping, target_type, feature_names
            )
        else:
            results["risk_scores"] = {}
    except Exception as e:
        logger.exception("Risk scoring failed")
        results["risk_scores"] = {}

    # 6. Survival Analysis (binary targets with tenure only)
    try:
        results["survival_analysis"] = _survival_analysis(df, mapping)
    except Exception as e:
        logger.exception("Survival analysis failed")
        results["survival_analysis"] = None

    # 7. Clustering
    try:
        results["clustering"] = _clustering(X, feature_names, df, mapping)
    except Exception as e:
        logger.exception("Clustering failed")
        results["clustering"] = {}

    # 8. What-If Scenarios
    try:
        if final_model is not None and results.get("feature_importance"):
            results["what_if_scenarios"] = _what_if_scenarios(
                final_model, X, feature_names,
                results["feature_importance"],
                target_type,
            )
        else:
            results["what_if_scenarios"] = []
    except Exception as e:
        logger.exception("What-if scenarios failed")
        results["what_if_scenarios"] = []

    return results


def _empty_results(reason: str) -> dict:
    return {
        "feature_importance": [],
        "risk_scores": {},
        "survival_analysis": None,
        "clustering": {},
        "what_if_scenarios": [],
        "model_metrics": {},
        "data_quality": {},
        "_skip_reason": reason,
    }


# ── Feature Preparation (with outlier capping) ──────────────────────


def _prepare_features(
    df: pd.DataFrame, mapping: dict
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], int]:
    """
    Prepare feature matrix X and target vector y.
    Includes IQR-based outlier capping for numeric features.
    Returns (X, y, feature_names, cat_col_names, num_col_names, outliers_capped).
    """
    target_col = mapping["target_col"]
    target_type = mapping["target_type"]
    id_col = mapping.get("employee_id_col")

    # Exclude target, id, and internal columns from features
    exclude = {target_col, "_target_flag"}
    if id_col:
        exclude.add(id_col)
    exclude.update(c for c in df.columns if c.startswith("_"))

    feature_cols = [c for c in df.columns if c not in exclude]

    # Separate numeric and categorical
    num_cols = [
        c for c in feature_cols
        if df[c].dtype in ("int64", "float64", "int32", "float32")
    ]
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" and df[c].nunique() <= 30
    ]

    # Build y
    if target_type == "binary":
        if "_target_flag" in df.columns:
            y = df["_target_flag"].values.astype(float)
        else:
            pos_vals = [
                v.lower()
                for v in mapping.get("target_positive_values", ["yes", "1", "true"])
            ]
            y = (
                df[target_col]
                .astype(str).str.strip().str.lower()
                .isin(pos_vals).astype(float).values
            )
    else:
        y = pd.to_numeric(df[target_col], errors="coerce").values

    # IQR-based outlier capping for numeric features
    outliers_capped = 0
    num_df = df[num_cols].copy() if num_cols else pd.DataFrame()
    for col in num_cols:
        vals = pd.to_numeric(num_df[col], errors="coerce")
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = vals.copy()
        vals = vals.clip(lower, upper)
        outliers_capped += int((before != vals).sum())
        num_df[col] = vals

    # Impute numeric
    num_data = (
        num_df.values.astype(float) if num_cols
        else np.empty((len(df), 0))
    )
    if num_cols:
        imp = SimpleImputer(strategy="median")
        num_data = imp.fit_transform(num_data)
        scaler = StandardScaler()
        num_data = scaler.fit_transform(num_data)

    # Encode categorical
    cat_data = np.empty((len(df), 0))
    encoded_cat_names: list[str] = []
    if cat_cols:
        cat_raw = df[cat_cols].fillna("_missing_").astype(str).values
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        cat_data = enc.fit_transform(cat_raw)
        encoded_cat_names = list(cat_cols)

    # Combine
    X = np.hstack([num_data, cat_data]) if cat_cols else num_data
    feature_names = num_cols + encoded_cat_names

    # Remove rows with NaN in y
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]

    return X, y, feature_names, cat_cols, num_cols, outliers_capped


# ── Model Comparison & Cross-Validation ──────────────────────────────


def _compare_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_type: str,
) -> tuple[str, dict]:
    """
    Compare RandomForest vs LogisticRegression/Ridge using 5-fold CV.
    Returns (best_model_name, metrics_dict).
    """
    if target_type == "binary":
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=500, random_state=42, solver="lbfgs",
            ),
        }
        scoring = "roc_auc"
    else:
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            ),
            "Ridge": Ridge(alpha=1.0, random_state=42),
        }
        scoring = "r2"

    comparison: dict[str, dict] = {}
    best_name = "RandomForest"
    best_cv_mean = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cross-validation on training set
        n_splits = min(5, max(2, int(len(y_train) / 10)))
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=n_splits, scoring=scoring,
        )

        entry: dict[str, Any] = {
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
        }

        if target_type == "binary":
            entry["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            entry["f1"] = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            entry["precision"] = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
            entry["recall"] = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                try:
                    entry["auc_roc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
                except ValueError:
                    entry["auc_roc"] = None
            else:
                entry["auc_roc"] = None
        else:
            entry["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
            entry["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
            entry["r2"] = round(float(r2_score(y_test, y_pred)), 4)

        comparison[name] = entry

        if float(cv_scores.mean()) > best_cv_mean:
            best_cv_mean = float(cv_scores.mean())
            best_name = name

    return best_name, {
        "best_model": best_name,
        "comparison": comparison,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "target_type": target_type,
    }


def _fallback_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_type: str,
) -> tuple[str, dict]:
    """Fallback: just train a RandomForest without comparison."""
    if target_type == "binary":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    entry: dict[str, Any] = {}
    if target_type == "binary":
        entry["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        entry["f1"] = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    else:
        entry["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        entry["r2"] = round(float(r2_score(y_test, y_pred)), 4)

    return "RandomForest", {
        "best_model": "RandomForest",
        "comparison": {"RandomForest": entry},
        "test_size": len(y_test),
        "train_size": len(y_train),
        "target_type": target_type,
    }


# ── 1. Feature Importance ────────────────────────────────────────────


def _feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_type: str,
    best_model_name: str,
) -> tuple[Any, list[dict]]:
    """Train the best model on full data and extract feature importances."""
    if best_model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=500, random_state=42, solver="lbfgs",
        )
    elif best_model_name == "Ridge":
        model = Ridge(alpha=1.0, random_state=42)
    elif target_type == "binary":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        )

    model.fit(X, y)

    # Extract importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For linear models, use absolute coefficient values
        coefs = model.coef_.ravel()
        importances = np.abs(coefs)
        # Normalize to sum to 1
        total = importances.sum()
        if total > 0:
            importances = importances / total
    else:
        importances = np.zeros(X.shape[1])

    # Determine direction via correlation with target
    correlations = []
    for i in range(X.shape[1]):
        valid = ~(np.isnan(X[:, i]) | np.isnan(y))
        if valid.sum() > 10:
            corr = np.corrcoef(X[valid, i], y[valid])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)

    results = []
    for i, name in enumerate(feature_names):
        results.append({
            "feature": name,
            "importance": round(float(importances[i]), 4),
            "direction": "positive" if correlations[i] > 0 else "negative",
            "correlation": round(float(correlations[i]), 4),
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return model, results[:15]


# ── 2. Risk Scoring ──────────────────────────────────────────────────


def _risk_scoring(
    model: Any,
    X: np.ndarray,
    df: pd.DataFrame,
    mapping: dict,
    target_type: str,
    feature_names: list[str],
) -> dict:
    """Score each employee for risk and bucket into High/Medium/Low."""
    id_col = mapping.get("employee_id_col", df.columns[0])
    dept_col = mapping.get("department_col", "Department")

    if target_type == "binary" and hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)

    # Only score rows that were in the valid set (non-NaN target)
    target_col = mapping["target_col"]
    if "_target_flag" in df.columns:
        valid_mask = df["_target_flag"].notna().values
    else:
        valid_mask = pd.to_numeric(
            df[target_col], errors="coerce"
        ).notna().values

    valid_df = df[valid_mask].copy()
    valid_df["_risk_score"] = scores

    # Bucket into High/Medium/Low using quantiles
    q67 = float(np.percentile(scores, 67))
    q33 = float(np.percentile(scores, 33))

    valid_df["_risk_level"] = pd.cut(
        valid_df["_risk_score"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low", "Medium", "High"],
    )

    # Distribution
    dist = valid_df["_risk_level"].value_counts().to_dict()
    distribution = {str(k): int(v) for k, v in dist.items()}

    # Top risk employees (top 20)
    if target_type == "binary":
        top_risk = valid_df.nlargest(20, "_risk_score")
    else:
        top_risk = valid_df.nsmallest(20, "_risk_score")

    top_risk_list = []
    for _, row in top_risk.iterrows():
        emp = {
            "employee_id": str(row.get(id_col, "")),
            "risk_score": round(float(row["_risk_score"]), 4),
            "risk_level": str(row.get("_risk_level", "Unknown")),
        }
        if dept_col in row.index:
            emp["department"] = str(row[dept_col])
        top_risk_list.append(emp)

    # Department-level risk
    high_risk_depts = []
    if dept_col in valid_df.columns:
        dept_risk = valid_df.groupby(dept_col).agg(
            avg_risk=("_risk_score", "mean"),
            high_count=("_risk_level", lambda x: (x == "High").sum()),
            total=("_risk_score", "count"),
        ).reset_index()
        dept_risk["high_risk_pct"] = round(
            dept_risk["high_count"] / dept_risk["total"], 4
        )
        dept_risk = dept_risk.sort_values(
            "avg_risk", ascending=(target_type != "binary")
        )

        for _, row in dept_risk.iterrows():
            high_risk_depts.append({
                "department": str(row[dept_col]),
                "avg_risk_score": round(float(row["avg_risk"]), 4),
                "high_risk_count": int(row["high_count"]),
                "total_employees": int(row["total"]),
                "high_risk_pct": round(float(row["high_risk_pct"]), 4),
            })

    return {
        "distribution": distribution,
        "top_risk_employees": top_risk_list,
        "high_risk_departments": high_risk_depts,
    }


# ── 3. Survival Analysis ────────────────────────────────────────────


def _survival_analysis(df: pd.DataFrame, mapping: dict) -> Optional[dict]:
    """
    Kaplan-Meier survival analysis per department.
    Only applicable for binary targets with a tenure-like column.
    """
    target_type = mapping.get("target_type")
    tenure_col = mapping.get("tenure_col")
    dept_col = mapping.get("department_col")

    if target_type != "binary":
        return None
    if not tenure_col or tenure_col not in df.columns:
        return None

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        logger.warning("lifelines not installed — skipping survival analysis")
        return None

    if "_target_flag" not in df.columns:
        return None

    work_df = df[[tenure_col, "_target_flag"]].copy()
    if dept_col and dept_col in df.columns:
        work_df["_dept"] = df[dept_col]
    else:
        work_df["_dept"] = "All"

    work_df = work_df.dropna()

    T = work_df[tenure_col].values.astype(float)
    E = work_df["_target_flag"].values.astype(int)

    kmf = KaplanMeierFitter()
    curves: dict[str, list[dict]] = {}
    medians: dict[str, Optional[float]] = {}

    # Limit to top 8 departments by headcount
    dept_counts = work_df["_dept"].value_counts()
    departments = dept_counts.head(8).index.tolist()

    for dept in departments:
        mask = work_df["_dept"] == dept
        if mask.sum() < 10:
            continue
        kmf.fit(T[mask], E[mask], label=str(dept))
        surv = kmf.survival_function_
        curve_data = []
        for time_val, row in surv.iterrows():
            curve_data.append({
                "time": round(float(time_val), 2),
                "survival_prob": round(float(row.iloc[0]), 4),
            })
        curves[str(dept)] = curve_data
        med = kmf.median_survival_time_
        medians[str(dept)] = (
            round(float(med), 2) if not np.isinf(med) else None
        )

    if not curves:
        return None

    return {
        "curves": curves,
        "median_survival": medians,
        "tenure_column": tenure_col,
    }


# ── 4. Clustering ────────────────────────────────────────────────────


def _clustering(
    X: np.ndarray,
    feature_names: list[str],
    df: pd.DataFrame,
    mapping: dict,
) -> dict:
    """KMeans clustering with silhouette-based k selection."""
    if X.shape[0] < 20 or X.shape[1] < 2:
        return {}

    # Try k=2 to 5
    best_k = 2
    best_score = -1.0
    max_k = min(5, X.shape[0] // 10)
    if max_k < 2:
        max_k = 2

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, sample_size=min(1000, len(X)))
        if score > best_score:
            best_score = score
            best_k = k

    # Fit final model
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(X)

    target_col = mapping.get("target_col")
    target_type = mapping.get("target_type")
    dept_col = mapping.get("department_col")

    # Build valid-row dataframe
    if "_target_flag" in df.columns:
        valid_mask = df["_target_flag"].notna().values
    else:
        valid_mask = pd.to_numeric(
            df[target_col], errors="coerce"
        ).notna().values
    valid_df = df[valid_mask].copy()
    valid_df["_cluster"] = labels

    # Cluster profiles
    profiles = []
    for c in range(best_k):
        cluster_mask = valid_df["_cluster"] == c
        cluster_df = valid_df[cluster_mask]
        profile: dict[str, Any] = {
            "cluster_id": c,
            "headcount": int(len(cluster_df)),
            "pct_of_total": round(len(cluster_df) / len(valid_df), 4),
        }

        # Target metric per cluster
        if "_target_flag" in cluster_df.columns:
            if target_type == "binary":
                profile["target_rate"] = round(
                    float(cluster_df["_target_flag"].mean()), 4
                )
            else:
                profile["target_mean"] = round(
                    float(cluster_df["_target_flag"].mean()), 4
                )

        # Top department in cluster
        if dept_col and dept_col in cluster_df.columns:
            top_dept = cluster_df[dept_col].value_counts().head(3)
            profile["top_departments"] = {
                str(k): int(v) for k, v in top_dept.items()
            }

        # Centroid feature means (top 5 distinguishing)
        centroid = km.cluster_centers_[c]
        global_mean = X.mean(axis=0)
        diffs = centroid - global_mean
        top_idx = np.argsort(np.abs(diffs))[::-1][:5]
        profile["distinguishing_features"] = [
            {
                "feature": feature_names[i],
                "cluster_mean": round(float(centroid[i]), 3),
                "global_mean": round(float(global_mean[i]), 3),
                "difference": round(float(diffs[i]), 3),
            }
            for i in top_idx
            if i < len(feature_names)
        ]

        profiles.append(profile)

    # PCA 2D projection for visualization
    pca_2d = []
    pca_variance = None
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        pca_variance = round(float(pca.explained_variance_ratio_.sum()), 4)
        # Sample up to 500 points for chart data
        sample_idx = np.random.RandomState(42).choice(
            len(coords), size=min(500, len(coords)), replace=False
        )
        for i in sample_idx:
            pca_2d.append({
                "x": round(float(coords[i, 0]), 3),
                "y": round(float(coords[i, 1]), 3),
                "cluster": f"Segment {int(labels[i])}",
            })

    return {
        "n_clusters": best_k,
        "silhouette_score": round(float(best_score), 4),
        "pca_variance_explained": pca_variance,
        "profiles": profiles,
        "pca_2d": pca_2d,
    }


# ── 5. What-If Scenarios ─────────────────────────────────────────────


def _what_if_scenarios(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    importances: list[dict],
    target_type: str,
) -> list[dict]:
    """Simulate shifting top features by ±1 std and predict impact."""
    top_features = importances[:3]

    scenarios = []
    for feat_info in top_features:
        fname = feat_info["feature"]
        if fname not in feature_names:
            continue
        feat_idx = feature_names.index(fname)

        current_std = float(X[:, feat_idx].std())
        if current_std < 1e-6:
            continue

        # Current prediction
        if target_type == "binary" and hasattr(model, "predict_proba"):
            current_rate = float(model.predict_proba(X)[:, 1].mean())
        else:
            current_rate = float(model.predict(X).mean())

        # Scenario: improve by 1 std (direction depends on correlation)
        direction = feat_info.get("direction", "positive")
        X_new = X.copy()
        if direction == "positive":
            X_new[:, feat_idx] -= current_std
            scenario_desc = f"Reduce {fname} by 1 std"
        else:
            X_new[:, feat_idx] += current_std
            scenario_desc = f"Increase {fname} by 1 std"

        if target_type == "binary" and hasattr(model, "predict_proba"):
            new_rate = float(model.predict_proba(X_new)[:, 1].mean())
        else:
            new_rate = float(model.predict(X_new).mean())

        impact = current_rate - new_rate
        pct_change = (
            (impact / current_rate * 100)
            if abs(current_rate) > 1e-6
            else 0.0
        )

        scenarios.append({
            "feature": fname,
            "scenario": scenario_desc,
            "current_rate": round(current_rate, 4),
            "predicted_rate": round(new_rate, 4),
            "absolute_impact": round(impact, 4),
            "pct_change": round(pct_change, 2),
            "importance": feat_info["importance"],
        })

    return scenarios
