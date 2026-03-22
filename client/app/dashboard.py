"""
HCM AI Insights Dashboard — Streamlit entry point.

This is the main UI for the prototype. It provides:
  - Sidebar: Company name input + file uploaders + Analyze button
  - 3-4 tabs: AI Insights | Descriptive Analytics | Predictive Analytics (deep) | User Guide & Privacy

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from client.app.components import api_client
import pandas as pd
import plotly.express as px


def _render_dynamic_chart(chart_spec: dict, key: str = ""):
    """Render a chart from a GPT-4o-generated ChartSpec."""
    chart_type = chart_spec.get("chart_type", "bar")
    title = chart_spec.get("title", "")
    data = chart_spec.get("data", [])
    x = chart_spec.get("x", "")
    y = chart_spec.get("y", "")
    color = chart_spec.get("color")

    if not data:
        return

    df = pd.DataFrame(data)

    # Distinguish color-as-column-name vs color-as-literal-value (e.g. "green")
    color_col = None
    color_literal = None
    if color and color in df.columns:
        color_col = color
    elif color:
        color_literal = color

    fig = None
    if chart_type == "bar" and x and y:
        fig = px.bar(df, x=x, y=y, color=color_col, title=title)
    elif chart_type == "pie" and x and y:
        fig = px.pie(df, names=x, values=y, title=title)
    elif chart_type == "scatter" and x and y:
        fig = px.scatter(df, x=x, y=y, color=color_col, title=title)
    elif chart_type == "line" and x and y:
        fig = px.line(df, x=x, y=y, color=color_col, title=title)
    elif chart_type == "table":
        st.dataframe(df, use_container_width=True, key=f"df_{key}" if key else None)
        return

    if fig:
        if color_literal:
            fig.update_traces(marker_color=color_literal)
        st.plotly_chart(fig, use_container_width=True, key=key if key else None)

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="HCM AI Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric { background-color: #f8fafc; padding: 12px; border-radius: 8px; }
    h1 { color: #1e40af; }
    .insight-card {
        background-color: #f0f9ff;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .risk-high { border-left-color: #ef4444 !important; background-color: #fef2f2 !important; }
    .risk-medium { border-left-color: #f59e0b !important; background-color: #fffbeb !important; }
    .risk-low { border-left-color: #10b981 !important; background-color: #f0fdf4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialization ─────────────────────────────────────

if "insights" not in st.session_state:
    st.session_state.insights = None
if "company_name" not in st.session_state:
    st.session_state.company_name = ""
if "upload_status" not in st.session_state:
    st.session_state.upload_status = {"csv": False, "json": False}
if "upload_filenames" not in st.session_state:
    st.session_state.upload_filenames = {"csv": "Structured Data (CSV)", "json": "Qualitative Feedback (JSON)"}
if "schema_review" not in st.session_state:
    st.session_state.schema_review = None
if "schema_approved" not in st.session_state:
    st.session_state.schema_approved = False
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "quick"
# Per-tab Ask AI state
for _prefix in ("desc", "pred"):
    if f"{_prefix}_chat_history" not in st.session_state:
        st.session_state[f"{_prefix}_chat_history"] = []
    if f"{_prefix}_api_history" not in st.session_state:
        st.session_state[f"{_prefix}_api_history"] = []
    if f"{_prefix}_explore_history" not in st.session_state:
        st.session_state[f"{_prefix}_explore_history"] = []
    if f"{_prefix}_explore_api_history" not in st.session_state:
        st.session_state[f"{_prefix}_explore_api_history"] = []
    if f"{_prefix}_explore_charts" not in st.session_state:
        st.session_state[f"{_prefix}_explore_charts"] = {}

# ── API connectivity check ────────────────────────────────────────────
if st.session_state.api_ok is None:
    healthy, msg = api_client.check_api_health()
    st.session_state.api_ok = healthy
    if not healthy:
        st.session_state.api_error = msg

if not st.session_state.api_ok:
    st.error(f"⚠️ **API Unreachable**\n\n{st.session_state.get('api_error', '')}")
    st.info("Once the API URL is configured, reload this page.")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/analytics.png", width=64)
    st.title("HCM AI Insights")
    st.markdown("---")

    st.subheader("📁 Data Upload")

    # GDPR data processing notice
    st.caption(
        "⚖️ **Data Processing Notice:** Uploaded data is analyzed for PII before "
        "any information is sent to OpenAI GPT-4o. You will review the full schema "
        "and PII classification, and must approve handling before analysis begins. "
        "Data is held in memory only and auto-deleted after 24 hours."
    )

    # Apply pending sample company name to the widget BEFORE it renders
    if "_pending_company" in st.session_state:
        st.session_state["_company_input"] = st.session_state.pop("_pending_company")

    company_name = st.text_input(
        "Company Name",
        key="_company_input",
        placeholder="e.g., NovaTech Solutions",
    )

    # Show persistent data-loaded indicator
    if st.session_state.upload_status.get("csv") and st.session_state.upload_status.get("json") and st.session_state.company_name:
        st.success(f"✅ {st.session_state.company_name} data loaded")
        if st.button("🗑️ Delete Data", help="Remove all uploaded data (GDPR right to erasure)", key="_delete_data"):
            try:
                api_client.delete_company_data(st.session_state.company_name)
                st.session_state.upload_status = {"csv": False, "json": False}
                st.session_state.insights = None
                st.session_state.schema_review = None
                st.session_state.schema_approved = False
                st.session_state.company_name = ""
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    csv_file = st.file_uploader(
        "Structured Data (CSV)",
        type=["csv"],
        help="Upload the employee data CSV file",
    )

    json_file = st.file_uploader(
        "Qualitative Feedback (JSON)",
        type=["json"],
        help="Upload the employee feedback JSON file",
    )

    st.markdown("---")

    # Use sample data buttons
    st.subheader("🗂️ Sample Data")
    st.caption("Pre-generated datasets for demo:")

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_company = None
    with sample_col1:
        if st.button("NovaTech", use_container_width=True):
            sample_company = ("NovaTech Solutions", "novatech")
    with sample_col2:
        if st.button("Meridian", use_container_width=True):
            sample_company = ("Meridian Retail Group", "meridian")
    with sample_col3:
        if st.button("Pinnacle", use_container_width=True):
            sample_company = ("Pinnacle Healthcare", "pinnacle")

    st.markdown("---")

    # Analysis mode toggle
    st.subheader("🧪 Analysis Mode")
    analysis_mode = st.radio(
        "Choose analysis depth",
        options=["quick", "deep"],
        index=0 if st.session_state.analysis_mode == "quick" else 1,
        format_func=lambda x: "⚡ Quick Insights (Descriptive)" if x == "quick" else "🔬 Deep Analysis (Descriptive & Predictive)",
        help="Quick = descriptive analytics only (~30-60s). Deep = descriptive + ML predictive analytics (~60-120s).",
    )
    st.session_state.analysis_mode = analysis_mode

    st.markdown("---")

    _data_ready = st.session_state.upload_status["csv"] and st.session_state.upload_status["json"]
    review_button = st.button(
        "📋 Review Schema" if _data_ready else "📋 Upload & Review Schema",
        use_container_width=True,
        type="primary",
    )


# ── Handle sample data selection ─────────────────────────────────────

def load_sample_data(name: str, slug: str):
    """Load pre-generated sample data files for a company."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic" / slug

    csv_path = data_dir / "employees.csv"
    json_path = data_dir / "feedback.json"

    if not csv_path.exists() or not json_path.exists():
        st.error(f"Sample data not found at {data_dir}. Run: python scripts/generate_data.py")
        return False

    with st.spinner(f"Loading {name} data..."):
        # Upload CSV
        csv_bytes = csv_path.read_bytes()
        result = api_client.upload_csv(csv_bytes, "employees.csv", name)
        if not result.get("success"):
            st.error(f"CSV upload failed: {result.get('message')}")
            return False

        # Upload JSON
        json_bytes = json_path.read_bytes()
        result = api_client.upload_feedback_json(json_bytes, "feedback.json", name)
        if not result.get("success"):
            st.error(f"JSON upload failed: {result.get('message')}")
            return False

    st.session_state.company_name = name
    st.session_state.upload_status = {"csv": True, "json": True}
    st.session_state.upload_filenames = {"csv": "employees.csv", "json": "feedback.json"}
    return True


if sample_company:
    name, slug = sample_company
    try:
        if load_sample_data(name, slug):
            st.session_state["_pending_company"] = name
            # Auto-trigger schema review for PII classification
            with st.spinner("Analyzing schema for PII..."):
                schema = api_client.analyze_schema(name)
                st.session_state.schema_review = schema
                st.session_state.schema_approved = False
                st.session_state.insights = None
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to load sample data: {e}")


# ── Handle manual upload + schema review ─────────────────────────────

if review_button:
    if not company_name:
        st.sidebar.error("Please enter a company name")
    elif not csv_file and not st.session_state.upload_status["csv"]:
        st.sidebar.error("Please upload a CSV file")
    elif not json_file and not st.session_state.upload_status["json"]:
        st.sidebar.error("Please upload a feedback JSON file")
    else:
        try:
            # Upload files if provided
            if csv_file:
                with st.spinner("Uploading structured data..."):
                    result = api_client.upload_csv(
                        csv_file.getvalue(), csv_file.name, company_name
                    )
                    if result.get("success"):
                        st.session_state.upload_status["csv"] = True
                        st.session_state.upload_filenames["csv"] = csv_file.name
                    else:
                        st.sidebar.error(f"CSV upload failed: {result.get('message')}")

            if json_file:
                with st.spinner("Uploading feedback data..."):
                    result = api_client.upload_feedback_json(
                        json_file.getvalue(), json_file.name, company_name
                    )
                    if result.get("success"):
                        st.session_state.upload_status["json"] = True
                        st.session_state.upload_filenames["json"] = json_file.name
                    else:
                        st.sidebar.error(f"JSON upload failed: {result.get('message')}")

            # Analyze schema for PII classification
            if st.session_state.upload_status["csv"] and st.session_state.upload_status["json"]:
                with st.spinner("Analyzing schema for PII classification..."):
                    schema = api_client.analyze_schema(company_name)
                    st.session_state.schema_review = schema
                    st.session_state.schema_approved = False
                    st.session_state.insights = None
                    st.session_state.company_name = company_name
                    st.rerun()
            else:
                st.sidebar.error("Both files must be uploaded before schema review")

        except Exception as e:
            st.error(f"Error during schema analysis: {str(e)}")


# ── Schema Review Panel ──────────────────────────────────────────────

_CATEGORY_ICONS = {
    "identifier": "🔴",
    "direct_pii": "🔴",
    "quasi_identifier": "🟡",
    "safe": "🟢",
}
_HANDLING_LABELS = {
    "exclude": "🚫 Excluded — never sent to AI",
    "hash": "🔒 Hashed — one-way, irreversible",
    "aggregate_only": "📊 Aggregated — no individual values",
    "pass_through": "✅ Passed through for analysis",
}
_STRICTNESS_ORDER = {"safe": 0, "quasi_identifier": 1, "direct_pii": 2, "identifier": 3}
_HANDLING_FOR_CATEGORY = {
    "identifier": "exclude",
    "direct_pii": "hash",
    "quasi_identifier": "aggregate_only",
    "safe": "pass_through",
}


def _render_schema_review():
    """Render the PII schema review panel for user approval."""
    review = st.session_state.schema_review
    if not review:
        return

    st.title("📋 Schema Review & PII Classification")
    st.markdown(
        "Before any data is sent to OpenAI for analysis, please review how each "
        "column in your dataset has been classified. **You can tighten protections** "
        "(move a column to a stricter category) but cannot loosen them."
    )

    # Summary metrics
    summary = review.get("pii_summary", {})
    m_cols = st.columns(4)
    m_cols[0].metric("Total Columns", review.get("column_count", 0))
    m_cols[1].metric("Rows", review.get("row_count", 0))
    pii_count = summary.get("identifier", 0) + summary.get("direct_pii", 0)
    m_cols[2].metric("🔴 PII Columns", pii_count)
    m_cols[3].metric("🟡 Quasi-identifiers", summary.get("quasi_identifier", 0))

    st.markdown("---")

    columns = review.get("columns", [])

    # Per-column tighten controls
    csv_label = st.session_state.upload_filenames.get("csv", "Structured Data (CSV)")
    st.subheader(f"📄 {csv_label} — Column Classifications")
    st.caption("You may only tighten a classification (move right → stricter). Cannot loosen.")

    if "schema_adjustments" not in st.session_state:
        st.session_state.schema_adjustments = {}

    category_options = ["safe", "quasi_identifier", "direct_pii", "identifier"]

    for i, col in enumerate(columns):
        col_name = col["column_name"]
        orig_category = col["pii_category"]
        orig_strictness = _STRICTNESS_ORDER.get(orig_category, 0)
        current_cat = st.session_state.schema_adjustments.get(col_name, orig_category)
        icon = _CATEGORY_ICONS.get(current_cat, "⚪")

        with st.expander(
            f"{icon} **{col_name}** — {current_cat.replace('_', ' ').title()}",
            expanded=(current_cat != "safe"),
        ):
            e1, e2 = st.columns([2, 1])
            with e1:
                st.caption(f"**Type:** {col['dtype']}  ·  **Confidence:** {col['confidence'].title()}")
                st.caption(f"**Reason:** {col['reason']}")
                handling_key = _HANDLING_FOR_CATEGORY.get(current_cat, col["handling"])
                st.caption(f"**Handling:** {_HANDLING_LABELS.get(handling_key, handling_key)}")
            with e2:
                # Only allow tightening — options at same or stricter level
                allowed = [c for c in category_options if _STRICTNESS_ORDER[c] >= orig_strictness]
                current_idx = allowed.index(current_cat) if current_cat in allowed else 0
                new_cat = st.selectbox(
                    "Override category",
                    options=allowed,
                    index=current_idx,
                    key=f"pii_override_{i}",
                    format_func=lambda x: f"{_CATEGORY_ICONS.get(x, '')} {x.replace('_', ' ').title()}",
                )
                if new_cat != current_cat:
                    st.session_state.schema_adjustments[col_name] = new_cat

    # Sample data preview (PII-redacted)
    sample_rows = review.get("sample_rows", [])
    if sample_rows:
        st.markdown("---")
        st.subheader("📊 Sample Data Preview (PII-redacted)")
        st.caption("This is how your data will look when sent to the AI — PII columns are excluded or hashed.")
        st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)

    # Handling explanation
    st.markdown("---")
    with st.expander("ℹ️ What do these protection levels mean?"):
        st.markdown(
            "| Level | Icon | What happens |\n"
            "|-------|------|-------------|\n"
            "| **Identifier** | 🔴 | Column is **completely excluded** — never sent to OpenAI |\n"
            "| **Direct PII** | 🔴 | Values are **one-way hashed** — AI sees only anonymized tokens |\n"
            "| **Quasi-identifier** | 🟡 | Only **aggregated statistics** are sent (counts, averages) — no individual values |\n"
            "| **Safe** | 🟢 | Data is **passed through** for full analysis |"
        )

    # ── Section 2: Unstructured Data (JSON) PII Review ───────────────

    fb_columns = review.get("feedback_columns", [])
    fb_samples = review.get("feedback_samples", [])
    fb_entry_count = review.get("feedback_entry_count", 0)
    fb_text_pii_count = review.get("feedback_text_pii_count", 0)

    if fb_columns or fb_samples:
        st.markdown("---")
        st.header("📝 Section 2: Unstructured Data (JSON) PII Review")
        st.markdown(
            "Your qualitative feedback JSON has been analyzed for PII — both in the "
            "**metadata keys** (like employee IDs) and **inside the free text** itself "
            "(emails, phone numbers, names). Text PII is automatically scrubbed before "
            "any feedback is sent to OpenAI."
        )

        # Feedback summary metrics
        fb_m = st.columns(4)
        fb_m[0].metric("Total Entries", fb_entry_count)
        fb_m[1].metric("Text PII Found", f"{fb_text_pii_count} in samples")
        fb_pii_keys = sum(1 for c in fb_columns if c["pii_category"] in ("identifier", "direct_pii"))
        fb_m[2].metric("🔴 PII Keys", fb_pii_keys)
        fb_m[3].metric("Metadata Keys", len(fb_columns))

        # Feedback key classifications
        if fb_columns:
            json_label = st.session_state.upload_filenames.get("json", "Qualitative Feedback (JSON)")
            st.subheader(f"📄 {json_label} — Metadata Key Classifications")
            st.caption("You may only tighten a classification (move right → stricter). Cannot loosen.")
            if "fb_schema_adjustments" not in st.session_state:
                st.session_state.fb_schema_adjustments = {}

            for j, fc in enumerate(fb_columns):
                key_name = fc["column_name"]
                orig_cat = fc["pii_category"]
                orig_strict = _STRICTNESS_ORDER.get(orig_cat, 0)
                current_cat = st.session_state.fb_schema_adjustments.get(key_name, orig_cat)
                icon = _CATEGORY_ICONS.get(current_cat, "⚪")

                with st.expander(f"{icon} **{key_name}** — {current_cat.replace('_', ' ').title()}"):
                    fb_e1, fb_e2 = st.columns([2, 1])
                    with fb_e1:
                        st.caption(f"**Type:** {fc['dtype']}")
                        st.caption(f"**Reason:** {fc['reason']}")
                        handling_key = _HANDLING_FOR_CATEGORY.get(current_cat, fc["handling"])
                        st.caption(f"**Handling:** {_HANDLING_LABELS.get(handling_key, handling_key)}")
                    with fb_e2:
                        allowed = [c for c in category_options if _STRICTNESS_ORDER[c] >= orig_strict]
                        cur_idx = allowed.index(current_cat) if current_cat in allowed else 0
                        new_fb_cat = st.selectbox(
                            "Override category",
                            options=allowed,
                            index=cur_idx,
                            key=f"fb_pii_override_{j}",
                            format_func=lambda x: f"{_CATEGORY_ICONS.get(x, '')} {x.replace('_', ' ').title()}",
                        )
                        if new_fb_cat != current_cat:
                            st.session_state.fb_schema_adjustments[key_name] = new_fb_cat

        # Sample text preview (original vs scrubbed)
        if fb_samples:
            st.markdown("---")
            st.subheader("📊 Sample Feedback Preview (5 of {})".format(fb_entry_count))
            st.caption(
                "Left: original text. Right: scrubbed version sent to AI. "
                "PII patterns (emails, phones, names) are automatically replaced with placeholders."
            )

            for idx, sample in enumerate(fb_samples):
                detections = sample.get("pii_detections", [])
                det_label = f" — **{len(detections)} PII found**" if detections else ""
                meta = sample.get("metadata", {})
                dept = meta.get("department", "")
                dept_label = f" · {dept}" if dept else ""

                with st.expander(
                    f"Entry {idx + 1}{dept_label}{det_label}",
                    expanded=(len(detections) > 0),
                ):
                    orig_col, scrub_col = st.columns(2)
                    with orig_col:
                        st.markdown("**Original Text:**")
                        # Highlight PII in the original text
                        display_text = sample.get("original_text", "")
                        if detections:
                            # Build highlighted version (replace PII spans with red markup)
                            parts = []
                            last_end = 0
                            for d in sorted(detections, key=lambda x: x["start"]):
                                parts.append(display_text[last_end:d["start"]])
                                parts.append(
                                    f'<span style="background:#fecaca;padding:1px 4px;'
                                    f'border-radius:3px;font-weight:600;">'
                                    f'{d["original"]}</span>'
                                )
                                last_end = d["end"]
                            parts.append(display_text[last_end:])
                            st.markdown("".join(parts), unsafe_allow_html=True)
                        else:
                            st.text(display_text)
                    with scrub_col:
                        st.markdown("**Scrubbed (sent to AI):**")
                        st.text(sample.get("scrubbed_text", ""))

                    if detections:
                        det_summary = ", ".join(
                            f"`{d['pii_type']}`: {d['original']}" for d in detections
                        )
                        st.caption(f"Detected: {det_summary}")

    # Approve & Run button
    st.markdown("")
    a1, a2 = st.columns([3, 1])
    with a1:
        st.info(
            "By clicking **Approve & Run Analysis**, you confirm that the PII handling "
            "shown above is acceptable and authorize sending the processed data to OpenAI GPT-4o."
        )
    with a2:
        if st.button("✅ Approve & Run Analysis", type="primary", use_container_width=True):
            # Build final columns with any user adjustments
            final_columns = []
            for col in columns:
                col_copy = dict(col)
                adjusted_cat = st.session_state.schema_adjustments.get(col["column_name"])
                if adjusted_cat:
                    col_copy["pii_category"] = adjusted_cat
                    col_copy["handling"] = _HANDLING_FOR_CATEGORY.get(adjusted_cat, col["handling"])
                final_columns.append(col_copy)

            # Build final feedback columns with any user adjustments
            final_fb_columns = []
            for fc in fb_columns:
                fc_copy = dict(fc)
                adj_fb_cat = st.session_state.get("fb_schema_adjustments", {}).get(fc["column_name"])
                if adj_fb_cat:
                    fc_copy["pii_category"] = adj_fb_cat
                    fc_copy["handling"] = _HANDLING_FOR_CATEGORY.get(adj_fb_cat, fc["handling"])
                final_fb_columns.append(fc_copy)

            try:
                with st.spinner("Approving schema..."):
                    api_client.approve_schema(
                        st.session_state.company_name,
                        final_columns,
                        feedback_columns=final_fb_columns,
                    )
                    st.session_state.schema_approved = True

                mode_label = (
                    "🔬 Deep Analysis (Descriptive & Predictive)"
                    if st.session_state.analysis_mode == "deep"
                    else "⚡ Quick Insights (Descriptive)"
                )
                with st.spinner(f"{mode_label} — analyzing your data... This may take 30-120 seconds."):
                    new_insights = api_client.generate_insights(
                        st.session_state.company_name,
                        analysis_mode=st.session_state.analysis_mode,
                    )
                    st.session_state.insights = new_insights
                    st.session_state.schema_adjustments = {}
                    st.session_state.fb_schema_adjustments = {}
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Main content ─────────────────────────────────────────────────────

# Schema review panel — shown when schema is analyzed but not yet approved
if st.session_state.schema_review and not st.session_state.schema_approved:
    _render_schema_review()
    st.stop()

# Edge case: schema approved but insights generation failed
if st.session_state.schema_approved and st.session_state.insights is None and st.session_state.company_name:
    st.title("📊 HCM AI Insights Dashboard")
    st.success("✅ Schema approved — PII handling confirmed.")
    if st.button("🚀 Run Analysis", type="primary"):
        mode_label = (
            "🔬 Deep Analysis (Descriptive & Predictive)"
            if st.session_state.analysis_mode == "deep"
            else "⚡ Quick Insights (Descriptive)"
        )
        with st.spinner(f"{mode_label} — analyzing your data... This may take 30-120 seconds."):
            new_insights = api_client.generate_insights(
                st.session_state.company_name,
                analysis_mode=st.session_state.analysis_mode,
            )
            st.session_state.insights = new_insights
            st.rerun()
    st.stop()

insights = st.session_state.insights

# ── ML Predictive Analytics Tab ──────────────────────────────────────────


def _trust_badge(level: str, explanation: str):
    """Render a coloured trust-level indicator."""
    cfg = {
        "high": ("🟢", "#10b981", "#f0fdf4", "High Confidence"),
        "medium": ("🟡", "#f59e0b", "#fffbeb", "Medium Confidence"),
        "low": ("🔴", "#ef4444", "#fef2f2", "Low Confidence"),
    }
    icon, border, bg, label = cfg.get(level, cfg["medium"])
    st.markdown(
        f'<div style="background:{bg}; border-left:4px solid {border}; '
        f'padding:10px 14px; border-radius:6px; margin:8px 0;">'
        f'<strong>{icon} {label}</strong> &mdash; {explanation}</div>',
        unsafe_allow_html=True,
    )


def _takeaway_box(text: str):
    """Render an actionable key-takeaway box."""
    st.markdown(
        f'<div style="background:#f0f9ff; border-left:4px solid #3b82f6; '
        f'padding:10px 14px; border-radius:6px; margin:4px 0 12px 0;">'
        f'<strong>💡 Key Takeaway:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


def _model_trust_level(mm: dict) -> tuple[str, str]:
    """Derive an overall trust level from model metrics."""
    if not mm:
        return "low", "No model metrics available — treat results as directional only."
    best = mm.get("best_model", "")
    comp = mm.get("comparison", {}).get(best, {})
    tt = mm.get("target_type", "")
    if tt == "binary":
        auc = comp.get("auc_roc")
        acc = comp.get("accuracy")
        if auc is not None:
            if auc >= 0.80:
                return "high", f"AUC-ROC {auc:.2f} — model discriminates well between outcomes."
            if auc >= 0.65:
                return "medium", f"AUC-ROC {auc:.2f} — moderate discrimination; directionally useful."
            return "low", f"AUC-ROC {auc:.2f} — weak model; use results as rough guidance only."
        if acc is not None:
            if acc >= 0.80:
                return "high", f"Accuracy {acc:.0%} — predictions are reliable."
            if acc >= 0.65:
                return "medium", f"Accuracy {acc:.0%} — moderate reliability."
            return "low", f"Accuracy {acc:.0%} — use with caution."
    else:
        r2 = comp.get("r2")
        if r2 is not None:
            if r2 >= 0.50:
                return "high", f"R² {r2:.2f} — model explains most variance."
            if r2 >= 0.25:
                return "medium", f"R² {r2:.2f} — moderate explanatory power; directionally useful."
            return "low", f"R² {r2:.2f} — limited predictive power; treat as exploratory."
    return "medium", "Metrics incomplete — interpret with moderate confidence."


def _render_ml_tab(insights: dict):
    """Render the Predictive Analytics tab with all ML results."""
    ml = insights.get("ml_results")
    if not ml:
        skip_reason = insights.get("ml_skip_reason")
        if skip_reason:
            st.warning(f"⚠️ Predictive analytics could not run: {skip_reason}")
        else:
            st.info("No ML results available. Re-run analysis in Deep mode.")
        return

    # ── PII Transparency Banner ──────────────────────────────────────
    pii_summary = insights.get("pii_handling_summary")
    if pii_summary:
        excluded = [p["column"] for p in pii_summary if p["handling"] == "exclude"]
        hashed = [p["column"] for p in pii_summary if p["handling"] == "hash"]
        agg_only = [p["column"] for p in pii_summary if p["handling"] == "aggregate_only"]

        lines = ["**🔒 Privacy-Aware Analytics** — The following columns were modified or excluded to protect personal data:"]
        if excluded:
            lines.append(f"- **Excluded** from ML features: `{'`, `'.join(excluded)}`")
        if hashed:
            lines.append(f"- **Hashed** (anonymised): `{'`, `'.join(hashed)}`")
        if agg_only:
            lines.append(f"- **Aggregate only** (not used as individual features): `{'`, `'.join(agg_only)}`")
        lines.append("")
        lines.append("Employee IDs in risk tables are shown as **privacy-safe hashes**. "
                      "Feature importance and risk scores reflect only non-PII predictors.")
        st.info("\n".join(lines))

    # ── Data Quality & Model Metrics Banner ──────────────────────────
    dq = ml.get("data_quality", {})
    mm = ml.get("model_metrics", {})

    if dq or mm:
        st.subheader("📊 Model Performance & Data Quality")

        # Data quality metrics
        if dq:
            dq_cols = st.columns(4)
            with dq_cols[0]:
                st.metric("Original Rows", dq.get("original_rows", "—"))
            with dq_cols[1]:
                st.metric("Duplicates Removed", dq.get("duplicates_removed", 0))
            with dq_cols[2]:
                st.metric("Outliers Capped", dq.get("outliers_capped", 0))
            with dq_cols[3]:
                st.metric("Features Used", dq.get("features_used", "—"))

        # Model comparison metrics
        if mm:
            best = mm.get("best_model", "")
            comparison = mm.get("comparison", {})
            target_type = mm.get("target_type", "")
            st.markdown(f"**Best Model:** `{best}` &nbsp;|&nbsp; "
                        f"**Train:** {mm.get('train_size', '?')} rows &nbsp;|&nbsp; "
                        f"**Test:** {mm.get('test_size', '?')} rows (80/20 split)")

            if comparison:
                # Build comparison table
                comp_rows = []
                for name, m in comparison.items():
                    row = {"Model": name}
                    row["CV Mean"] = m.get("cv_mean", "—")
                    row["CV Std"] = m.get("cv_std", "—")
                    if target_type == "binary":
                        row["Accuracy"] = m.get("accuracy", "—")
                        row["AUC-ROC"] = m.get("auc_roc", "—")
                        row["F1"] = m.get("f1", "—")
                        row["Precision"] = m.get("precision", "—")
                        row["Recall"] = m.get("recall", "—")
                    else:
                        row["R²"] = m.get("r2", "—")
                        row["RMSE"] = m.get("rmse", "—")
                        row["MAE"] = m.get("mae", "—")
                    if name == best:
                        row["Model"] = f"✅ {name}"
                    comp_rows.append(row)
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True, key="ml_model_comp")

                # Show cross-validation fold scores for best model
                best_m = comparison.get(best, {})
                cv_scores = best_m.get("cv_scores", [])
                if cv_scores:
                    with st.expander(f"📈 {best} — {len(cv_scores)}-Fold Cross-Validation Scores"):
                        cv_df = pd.DataFrame([{"Fold": i + 1, "Score": s} for i, s in enumerate(cv_scores)])
                        fig = px.bar(cv_df, x="Fold", y="Score",
                                     title=f"Cross-Validation Scores (mean={best_m.get('cv_mean', 0):.4f} ± {best_m.get('cv_std', 0):.4f})")
                        fig.update_layout(yaxis_title="Score", xaxis_title="Fold")
                        st.plotly_chart(fig, use_container_width=True, key="ml_cv_chart")

        # Overall trust badge for the model
        trust_level, trust_text = _model_trust_level(mm)
        _trust_badge(trust_level, trust_text)
        st.markdown("---")

    # ML Narrative (AI-generated interpretation)
    narrative = ml.get("ml_narrative", "")
    if narrative:
        st.markdown(narrative)
        st.markdown("---")

    # ── 1. Feature Importance ────────────────────────────────────────
    fi = ml.get("feature_importance", [])
    if fi:
        st.subheader("🎯 Feature Importance — Top Predictors")

        # PII exclusion note
        if insights.get("pii_handling_summary"):
            _excl = [p["column"] for p in insights["pii_handling_summary"]
                     if p["handling"] in ("exclude", "hash", "aggregate_only")]
            if _excl:
                st.caption(
                    f"🔒 **Privacy Note:** {len(_excl)} column(s) excluded from predictive "
                    f"modelling due to PII classification. Only safe features are shown below."
                )

        # Show model info badge
        if mm:
            best = mm.get("best_model", "")
            best_m = mm.get("comparison", {}).get(best, {})
            target_type = mm.get("target_type", "")
            if target_type == "binary":
                metric_label = f"AUC-ROC: {best_m.get('auc_roc', '—')} | Accuracy: {best_m.get('accuracy', '—')}"
            else:
                metric_label = f"R²: {best_m.get('r2', '—')} | RMSE: {best_m.get('rmse', '—')}"
            st.caption(f"Model: **{best}** | {metric_label}")

        fi_df = pd.DataFrame(fi)
        if "importance" in fi_df.columns and "feature" in fi_df.columns:
            fi_df = fi_df.sort_values("importance", ascending=True)
            color_map = {"positive": "#ef4444", "negative": "#10b981"}
            fig = px.bar(
                fi_df, x="importance", y="feature", orientation="h",
                title="Top Predictors of Target Variable",
                color="direction" if "direction" in fi_df.columns else None,
                color_discrete_map=color_map,
            )
            fig.update_layout(yaxis_title="", xaxis_title="Importance Score", height=max(400, len(fi_df) * 28))
            st.plotly_chart(fig, use_container_width=True, key="ml_fi_chart")

            with st.expander("📊 Raw Feature Importance Data"):
                display_cols = [c for c in ["feature", "importance", "direction", "correlation"] if c in fi_df.columns]
                st.dataframe(fi_df[display_cols].sort_values("importance", ascending=False), use_container_width=True)

        # Trust + Takeaway for Feature Importance
        fi_trust, fi_trust_text = _model_trust_level(mm)
        _trust_badge(fi_trust, fi_trust_text)
        top_feat = fi[0] if fi else {}
        if top_feat:
            direction_word = "increases" if top_feat.get("direction") == "positive" else "decreases"
            _takeaway_box(
                f"<strong>{top_feat['feature']}</strong> is the strongest predictor "
                f"(importance {top_feat.get('importance', 0):.3f}) and {direction_word} the target. "
                f"Prioritise interventions around this factor first."
            )
        st.markdown("---")

    # ── 2. Risk Scores ───────────────────────────────────────────────
    risk = ml.get("risk_scores", {})
    if risk:
        st.subheader("⚠️ Risk Score Distribution")

        dist = risk.get("distribution", {})
        if dist:
            total_employees = sum(dist.values())
            r_cols = st.columns(3)
            for i, (level, count) in enumerate(dist.items()):
                with r_cols[i % 3]:
                    icon = "🔴" if "high" in level.lower() else "🟡" if "medium" in level.lower() else "🟢"
                    pct = round(count / total_employees * 100, 1) if total_employees > 0 else 0
                    st.metric(f"{icon} {level}", f"{count} ({pct}%)")

            # Pie chart of distribution
            dist_df = pd.DataFrame([{"Level": k, "Count": v} for k, v in dist.items()])
            fig = px.pie(dist_df, names="Level", values="Count", title="Risk Distribution",
                         color="Level", color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"})
            st.plotly_chart(fig, use_container_width=True, key="ml_risk_pie")

        # Department risk
        hi_depts = risk.get("high_risk_departments", [])
        if hi_depts:
            dept_df = pd.DataFrame(hi_depts)
            if "department" in dept_df.columns and "high_risk_pct" in dept_df.columns:
                fig = px.bar(dept_df, x="department", y="high_risk_pct", title="High-Risk % by Department",
                             color="high_risk_pct", color_continuous_scale="OrRd")
                fig.update_layout(xaxis_title="", yaxis_title="High-Risk %")
                st.plotly_chart(fig, use_container_width=True, key="ml_risk_dept")

        # Top risk employees table
        top_risk = risk.get("top_risk_employees", [])
        if top_risk:
            with st.expander(f"👤 Top {len(top_risk)} Highest-Risk Employees"):
                st.caption(
                    "⚠️ **AI Disclaimer:** Risk scores are statistical estimates generated by "
                    "machine learning models. They should not be used as the sole basis for "
                    "employment decisions. Always combine with human judgment and contextual knowledge."
                )
                # PII notice for hashed IDs
                if insights.get("pii_handling_summary"):
                    st.caption(
                        "🔒 **Privacy Note:** Employee IDs are shown as SHA-256 hashes "
                        "to protect personal data. Each hash uniquely represents an employee "
                        "without revealing their actual identifier."
                    )
                st.dataframe(pd.DataFrame(top_risk), use_container_width=True)

        # Trust + Takeaway for Risk Scores
        risk_trust, risk_trust_text = _model_trust_level(mm)
        _trust_badge(risk_trust, risk_trust_text)
        # Build takeaway from distribution + departments
        if dist:
            total_emp = sum(dist.values())
            high_count = dist.get("High", 0)
            high_pct = round(high_count / total_emp * 100, 1) if total_emp else 0
            takeaway_parts = [f"{high_pct}% of employees ({high_count}) are flagged <strong>High Risk</strong>."]
            if hi_depts:
                worst = hi_depts[0]
                takeaway_parts.append(
                    f"<strong>{worst.get('department', '?')}</strong> has the highest concentration "
                    f"at {worst.get('high_risk_pct', 0):.0%} high-risk."
                )
            takeaway_parts.append("Target retention programmes at these groups immediately.")
            _takeaway_box(" ".join(takeaway_parts))
        st.markdown("---")

    # ── 3. Survival Analysis ─────────────────────────────────────────
    surv = ml.get("survival_analysis")
    if surv and surv.get("curves"):
        st.subheader("📉 Survival Analysis — Retention Curves")
        st.caption(f"Tenure column: **{surv.get('tenure_column', '—')}**")

        curves = surv["curves"]
        # Build combined DF for line chart
        all_rows = []
        for dept, points in curves.items():
            for pt in points:
                all_rows.append({
                    "Department": dept,
                    "Tenure": pt.get("time", pt.get("timeline", 0)),
                    "Survival Probability": pt.get("survival_prob", 0),
                })
        if all_rows:
            surv_df = pd.DataFrame(all_rows)
            fig = px.line(surv_df, x="Tenure", y="Survival Probability", color="Department",
                          title="Kaplan-Meier Retention Curves by Department")
            fig.update_layout(yaxis_range=[0, 1.05], xaxis_title="Tenure (years)", yaxis_title="Probability of Staying")
            st.plotly_chart(fig, use_container_width=True, key="ml_surv_chart")

        # Median survival times
        medians = surv.get("median_survival", {})
        if medians:
            med_df = pd.DataFrame([{"Department": k, "Median Tenure (years)": v} for k, v in medians.items()])
            st.dataframe(med_df, use_container_width=True, key="ml_surv_median")

        # Trust + Takeaway for Survival Analysis
        n_curves = len(curves)
        if n_curves >= 4:
            _trust_badge("high", f"Kaplan-Meier curves for {n_curves} departments — robust sample coverage.")
        elif n_curves >= 2:
            _trust_badge("medium", f"Curves for {n_curves} departments — reasonable but limited coverage.")
        else:
            _trust_badge("low", f"Only {n_curves} department(s) had enough data — interpret cautiously.")
        if medians:
            valid_medians = {k: v for k, v in medians.items() if v is not None}
            if valid_medians:
                shortest_dept = min(valid_medians, key=valid_medians.get)
                longest_dept = max(valid_medians, key=valid_medians.get)
                _takeaway_box(
                    f"<strong>{shortest_dept}</strong> has the shortest median retention "
                    f"({valid_medians[shortest_dept]:.1f} yrs) vs <strong>{longest_dept}</strong> "
                    f"({valid_medians[longest_dept]:.1f} yrs). "
                    f"Investigate what {shortest_dept} can learn from {longest_dept}'s practices."
                )
        st.markdown("---")

    # ── 4. Employee Segments (Clustering) ────────────────────────────
    clust = ml.get("clustering", {})
    profiles = clust.get("profiles", [])
    if profiles:
        st.subheader("👥 Employee Segments — Cluster Analysis")

        # Clustering quality metrics
        sil = clust.get("silhouette_score")
        n_clust = clust.get("n_clusters")
        pca_var = clust.get("pca_variance_explained")
        metric_parts = []
        if n_clust:
            metric_parts.append(f"**{n_clust} segments**")
        if sil is not None:
            metric_parts.append(f"Silhouette Score: **{sil:.4f}**")
        if pca_var is not None:
            metric_parts.append(f"PCA Variance Explained: **{pca_var:.1%}**")
        if metric_parts:
            st.caption(" | ".join(metric_parts))

        for ci, profile in enumerate(profiles):
            label = f"Segment {profile.get('cluster_id', ci)}"
            headcount = profile.get("headcount", 0)
            pct = profile.get("pct_of_total", 0)
            target_rate = profile.get("target_rate", profile.get("target_mean", "—"))
            target_str = f"{target_rate:.1%}" if isinstance(target_rate, float) and target_rate <= 1.0 else str(target_rate)

            with st.expander(f"**{label}** — {headcount} employees ({pct:.1%}) | Target: {target_str}"):
                # Top departments
                top_depts = profile.get("top_departments", {})
                if top_depts:
                    st.markdown("**Top Departments:** " + ", ".join(f"{k} ({v})" for k, v in top_depts.items()))

                # Distinguishing features
                dist_feats = profile.get("distinguishing_features", [])
                if dist_feats:
                    if isinstance(dist_feats, list):
                        feat_df = pd.DataFrame(dist_feats)
                    else:
                        feat_df = pd.DataFrame([{"Feature": k, "Value": v} for k, v in dist_feats.items()])
                    st.dataframe(feat_df, use_container_width=True, key=f"ml_clust_feat_{ci}")

        # PCA scatter plot
        pca_points = clust.get("pca_2d", [])
        if pca_points:
            pca_df = pd.DataFrame(pca_points)
            x_col = "x" if "x" in pca_df.columns else "pc1" if "pc1" in pca_df.columns else None
            y_col = "y" if "y" in pca_df.columns else "pc2" if "pc2" in pca_df.columns else None
            if x_col and y_col:
                fig = px.scatter(pca_df, x=x_col, y=y_col,
                                 color="cluster" if "cluster" in pca_df.columns else None,
                                 title="Employee Segments — PCA Projection",
                                 labels={x_col: "Principal Component 1", y_col: "Principal Component 2"})
                st.plotly_chart(fig, use_container_width=True, key="ml_pca_scatter")

        # Trust + Takeaway for Clustering
        sil_val = clust.get("silhouette_score", 0)
        if sil_val >= 0.50:
            _trust_badge("high", f"Silhouette score {sil_val:.2f} — well-separated, meaningful segments.")
        elif sil_val >= 0.25:
            _trust_badge("medium", f"Silhouette score {sil_val:.2f} — segments overlap somewhat; use as directional groupings.")
        else:
            _trust_badge("low", f"Silhouette score {sil_val:.2f} — weak separation; treat clusters as rough groupings.")
        # Takeaway: highlight the riskiest segment
        if profiles:
            riskiest = max(
                profiles,
                key=lambda p: p.get("target_rate", p.get("target_mean", 0)) or 0,
            )
            seg_id = riskiest.get("cluster_id", "?")
            seg_hc = riskiest.get("headcount", 0)
            seg_rate = riskiest.get("target_rate", riskiest.get("target_mean", 0))
            rate_str = f"{seg_rate:.1%}" if isinstance(seg_rate, float) and seg_rate <= 1 else str(round(seg_rate, 2))
            top_d = riskiest.get("top_departments", {})
            dept_hint = f" (mostly in {list(top_d.keys())[0]})" if top_d else ""
            _takeaway_box(
                f"<strong>Segment {seg_id}</strong>{dept_hint} has the highest target rate "
                f"at {rate_str} with {seg_hc} employees. "
                f"Deep-dive into this group's distinguishing features to design targeted action plans."
            )
        st.markdown("---")

    # ── 5. What-If Scenarios ─────────────────────────────────────────
    wif = ml.get("what_if_scenarios", [])
    if wif:
        st.subheader("🔮 What-If Scenario Analysis")
        for si, scenario in enumerate(wif):
            feat = scenario.get("feature", f"Feature {si + 1}")
            desc = scenario.get("scenario", "change")
            baseline = scenario.get("current_rate", 0)
            projected = scenario.get("predicted_rate", 0)
            change_pct = scenario.get("pct_change", 0)
            importance = scenario.get("importance", 0)

            st.markdown(f"**{feat}** — {desc} &nbsp; *(importance: {importance:.4f})*")
            sc_cols = st.columns(3)
            with sc_cols[0]:
                st.metric("Baseline Rate", f"{baseline:.4f}")
            with sc_cols[1]:
                st.metric("Projected Rate", f"{projected:.4f}")
            with sc_cols[2]:
                st.metric("Change", f"{abs(change_pct):.1f}%", delta=f"{change_pct:.1f}%")

            # Mini bar chart comparing baseline vs projected
            comp_df = pd.DataFrame([
                {"Scenario": "Baseline", "Rate": baseline},
                {"Scenario": "Projected", "Rate": projected},
            ])
            fig = px.bar(comp_df, x="Scenario", y="Rate", title=f"Impact: {desc}",
                         color="Scenario", color_discrete_map={"Baseline": "#94a3b8", "Projected": "#3b82f6"})
            st.plotly_chart(fig, use_container_width=True, key=f"ml_wif_{si}")

            if si < len(wif) - 1:
                st.markdown("---")

        # Trust + Takeaway for What-If (after all scenarios)
        wif_trust, wif_trust_text = _model_trust_level(mm)
        _trust_badge(wif_trust, wif_trust_text)
        # Find the scenario with the largest absolute impact
        biggest = max(wif, key=lambda s: abs(s.get("pct_change", 0)))
        direction_verb = "reduce" if biggest.get("pct_change", 0) > 0 else "increase"
        _takeaway_box(
            f"The highest-impact lever is <strong>{biggest.get('feature', '?')}</strong>: "
            f"{biggest.get('scenario', '?')} could {direction_verb} the target metric by "
            f"~{abs(biggest.get('pct_change', 0)):.1f}%. "
            f"This is the single most actionable change to prioritise."
        )


# ── User Guide content (reusable) ────────────────────────────────────────

def _render_user_guide():
    """Render the full User Guide content."""
    st.subheader("📖 User Guide")
    st.markdown("""
---

### What Is This Tool?

HCM AI Insights is an **AI-powered workforce analytics dashboard** that turns raw employee data into
actionable intelligence — automatically. Upload any HR dataset (structured CSV + qualitative feedback JSON),
and GPT-4o will detect the key metrics, run a full analysis pipeline, and generate an interactive dashboard
tailored to *your* data.

No templates. No manual configuration. The AI figures out what matters in your data and builds the
analysis around it.

---

### Getting Started

| Step | Action |
|------|--------|
| **1** | Enter a **Company Name** in the sidebar |
| **2** | Upload a **Structured Data CSV** (employee records with any columns) |
| **3** | Upload a **Qualitative Feedback JSON** (survey responses, exit interviews, etc.) |
| **4** | Choose **Analysis Mode**: ⚡ Quick Insights (Descriptive, ~30-60s) or 🔬 Deep Analysis (Descriptive & Predictive, ~60-120s) |
| **5** | Click **🚀 Analyze** |
| **6** | Explore the generated tabs! |

> **Quick demo:** Click **NovaTech**, **Meridian**, or **Pinnacle** in the sidebar to load
> pre-generated sample data instantly, then hit Analyze.

---

### Dashboard Tabs

#### 🧠 AI Insights
The executive summary and strategic recommendations layer — your starting point:
- **Executive Summary** — a C-suite-ready narrative synthesizing all findings
- **Key Charts at a Glance** — one representative chart from every analysis section in a compact grid
- **Recommendations** — prioritized by risk level (🔴 High / 🟡 Medium / 🟢 Low) with
  specific key issues and actionable steps

#### 📊 Descriptive Analytics
All descriptive analysis in one place — the primary metric analysis, employee feedback, and correlations:
- **Structured Analysis** *(dynamic name)* — Department-level breakdowns, risk factor identification,
  statistical distributions, interactive Plotly charts. Tab name adapts to your data
  (e.g. "Attrition Analysis", "Engagement Score Analysis", "Burnout Index Analysis").
- **Employee Feedback** — AI-classified sentiment analysis and theme extraction from qualitative feedback,
  sentiment by department, top themes with representative quotes.
- **Correlation Insights** — Cross-analysis linking quantitative metrics with qualitative sentiment.
- **💡 Ask AI** *(embedded)* — Ask follow-up questions about the descriptive results in natural language.
  Get text answers and auto-generated charts.
- **✏️ Explore & Customize** *(embedded)* — Modify any chart — change colors, chart types, filters,
  or create entirely new visualizations.

#### 🔮 Predictive Analytics *(Deep mode only)*
Machine learning predictions and advanced analytics — only appears when you run in **Deep** mode:
- **Feature Importance** — Which factors most strongly predict your target variable, with model comparison
  (RandomForest vs LogisticRegression/Ridge) and cross-validation metrics
- **Risk Scoring** — High / Medium / Low risk distribution with department-level breakdown
- **Survival Analysis** — Kaplan-Meier retention curves showing expected tenure by department *(binary targets only)*
- **Employee Segments** — K-Means clustering to identify distinct employee profiles with PCA visualization
- **What-If Scenarios** — Simulated impact of changing key factors on the target metric
- **Trust Indicators** — Green/yellow/red confidence badges after every analysis section
- **💡 Ask AI** *(embedded)* — Ask follow-up questions about the predictive results.
- **✏️ Explore & Customize** *(embedded)* — Modify charts or create new visualizations.

#### 📖 User Guide & Privacy
This documentation page — including the full **Data Privacy & Security** section with architecture
diagrams, GDPR compliance mapping, and FAQ below.

---

### Special Features

| Feature | Description |
|---------|-------------|
| **Zero-Config Schema Detection** | Upload *any* HR CSV — the AI auto-detects columns, data types, and the primary target variable (binary or numeric) |
| **Dynamic Analysis Plans** | GPT-4o generates a custom analysis plan for each dataset — no hardcoded templates |
| **Adaptive Feedback Ingestion** | JSON feedback files with *any* key names are auto-mapped (the system detects text, ID, department, date fields automatically) |
| **Semantic Search (ChromaDB)** | Qualitative feedback is embedded and stored in a vector database for targeted, department-level semantic retrieval |
| **11-Step AI Pipeline** | Schema analysis → plan generation → execution → targeted feedback retrieval → sentiment → themes → correlation → summary → recommendations → dashboard spec |
| **ML Predictive Analytics** | Feature importance, risk scoring, survival analysis, clustering, what-if scenarios — activated in Deep mode |
| **Analysis Mode Toggle** | Choose ⚡ Quick Insights (Descriptive, ~30-60s) or 🔬 Deep Analysis (Descriptive & Predictive, ~60-120s) |
| **Interactive Charts** | All visualizations are Plotly-based — hover, zoom, pan, and download as PNG |
| **Conversational AI** | Multi-turn conversations in Ask AI and Explore tabs with full context memory |
| **Dynamic Tab Names** | Tab labels adapt to reflect what was actually analyzed in your data |
| **Works Across Industries** | Tech, retail, healthcare, finance — any HCM dataset works out of the box |

---

### Data Format Guide

**Structured CSV** — Any employee-level CSV with columns like:
- Employee ID, department, age, tenure, salary, role, etc.
- A target/outcome column (e.g., Attrition, EngagementScore, BurnoutIndex)
- The AI will figure out which column is the target and how to analyze it

**Qualitative Feedback JSON** — An array of objects, e.g.:
```json
[
  {
    "feedback_id": "F001",
    "department": "Engineering",
    "response_text": "I feel overworked and undervalued...",
    "feedback_type": "exit_interview"
  }
]
```
Field names are flexible — the system auto-detects text content, department, ID, and date fields
regardless of what you name them.

---

### Sample Datasets

Three pre-loaded demo companies showcase the tool's versatility:

| Company | Industry | Target Variable | Focus Areas |
|---------|----------|----------------|-------------|
| **NovaTech Solutions** | Technology | Attrition (Yes/No) | Engineering burnout, career stagnation, compensation gaps |
| **Meridian Retail Group** | Retail | Engagement Score (1–10) | Hourly wage impact, shift patterns, store-region disparities |
| **Pinnacle Healthcare** | Healthcare | Burnout Index (1–10) | Patient load, shift patterns, unit-level stress |

---

### Tips & Best Practices

1. **Larger datasets = richer insights** — 200+ employee records give the AI more patterns to find
2. **Include diverse feedback** — mix of surveys, exit interviews, and open comments yields the best theme extraction
3. **Use Ask AI for drill-downs** — after the initial analysis, use the embedded Ask AI in each tab to go deeper
4. **Customize charts in-place** — use the embedded Explore & Customize section to modify any chart without leaving your current tab
5. **Re-analyze anytime** — upload updated data and hit Analyze again to refresh everything

---

## 🔒 Data Privacy & Security

HCM AI Insights is built with a **Privacy-First Architecture** that ensures your employee data
never leaves your environment. This section explains exactly how it works.

---

### Architecture Overview: Two-Tier Privacy Design

The system is split into two isolated components:

```
┌─────────────────────────────────────────────────┐
│           YOUR ENVIRONMENT (Client)              │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Streamlit │  │ Local    │  │  ChromaDB    │  │
│  │ Dashboard │  │ FastAPI  │  │  (Vectors)   │  │
│  │ (UI)      │  │ (Port    │  │              │  │
│  │           │  │  8001)   │  │              │  │
│  └─────┬─────┘  └────┬─────┘  └──────────────┘  │
│        │             │                           │
│        └──────┬──────┘                           │
│               │                                  │
│   ┌───────────▼──────────────┐                   │
│   │   Raw Data Store         │                   │
│   │   • CSV employee records │                   │
│   │   • JSON feedback        │                   │
│   │   • ML models (sklearn)  │                   │
│   │   • PII classifier       │                   │
│   └──────────────────────────┘                   │
│               │                                  │
│   ╔═══════════▼══════════════╗                   │
│   ║  AGGREGATION & SCRUBBING ║ ◄── Security      │
│   ║  • Remove PII columns    ║     Boundary      │
│   ║  • Scrub text for PII    ║                   │
│   ║  • Aggregate statistics  ║                   │
│   ║  • Hash employee IDs     ║                   │
│   ╚═══════════╤══════════════╝                   │
│               │                                  │
│        Only anonymized                           │
│        aggregated data ▼                         │
└───────────────┼─────────────────────────────────┘
                │ HTTPS (encrypted)
┌───────────────▼─────────────────────────────────┐
│        PROVIDER BACKEND (Cloud)                  │
│   ┌──────────────────────────────┐               │
│   │  FastAPI (Port 8000)         │               │
│   │  • GPT-4o API orchestration  │               │
│   │  • Embedding proxy           │               │
│   │  • NO data storage           │               │
│   │  • NO raw data access        │               │
│   └──────────────────────────────┘               │
│         │                                        │
│         ▼                                        │
│   ┌──────────────┐                               │
│   │  OpenAI API  │                               │
│   │  (GPT-4o)    │                               │
│   └──────────────┘                               │
└──────────────────────────────────────────────────┘
```

| Component | Location | Has Raw Data? | Purpose |
|-----------|----------|:---:|--------|
| **Streamlit Dashboard** | Your network | ✅ | User interface |
| **Client FastAPI** | Your network | ✅ | Data processing, ML, PII classification |
| **ChromaDB** | Your network | ✅ | Feedback vector storage |
| **Provider Backend** | Cloud | ❌ | GPT-4o orchestration only |
| **OpenAI API** | Cloud | ❌ | Receives only anonymized text |

> **Key Guarantee:** Raw employee data — names, emails, IDs, salaries, phone numbers,
> and all other PII — **never crosses the network boundary**. The provider backend
> receives only aggregated statistics and PII-scrubbed text.

---

### Complete User Journey & Data Flow

This diagram traces every step from the moment you interact with the dashboard
to the final insights — showing exactly **where** each operation happens and
**what data** moves between components.

```
 YOU (Browser)
  │
  │ ① Open dashboard (https://localhost:8501)
  │
  ▼
┌══════════════════════════════════════════════════════════════════════════════┐
║                    YOUR ENVIRONMENT (Client Container)                     ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  STREAMLIT DASHBOARD (Port 8501)                                   │   ║
║  │  • You see the upload page                                         │   ║
║  │  • You upload CSV + JSON files                                     │   ║
║  └──────────────────────────┬──────────────────────────────────────────┘   ║
║                             │                                             ║
║            ② Files sent to local API (never leaves your machine)          ║
║                             │                                             ║
║  ┌──────────────────────────▼──────────────────────────────────────────┐   ║
║  │  LOCAL FASTAPI (Port 8001)                                         │   ║
║  │                                                                    │   ║
║  │  ③ LOAD DATA INTO MEMORY                                          │   ║
║  │     • CSV → pandas DataFrame (in-memory, not saved to disk)        │   ║
║  │     • JSON feedback → held for ChromaDB ingestion                  │   ║
║  │     • 24-hour auto-expiry timer starts                             │   ║
║  │                                                                    │   ║
║  │  ④ PII CLASSIFICATION (100% local Python — no AI involved)         │   ║
║  │     • Column names checked against regex patterns                  │   ║
║  │     • Sample values scanned for email/phone/SSN patterns           │   ║
║  │     • Uniqueness ratio computed to detect identifiers              │   ║
║  │     • Each column classified: safe / quasi_identifier / direct_pii │   ║
║  │                                                                    │   ║
║  │  ⑤ SCHEMA REVIEW (you see this in the dashboard)                   │   ║
║  │     ┌──────────────────────────────────────────────────────────┐    │   ║
║  │     │  Column         │ PII Category     │ Handling           │    │   ║
║  │     │  EmployeeName   │ direct_pii       │ ❌ EXCLUDED        │    │   ║
║  │     │  Email          │ direct_pii       │ ❌ EXCLUDED        │    │   ║
║  │     │  Salary         │ quasi_identifier │ 📊 AGGREGATE ONLY  │    │   ║
║  │     │  Department     │ safe             │ ✅ PASS THROUGH    │    │   ║
║  │     │  Age            │ safe             │ ✅ PASS THROUGH    │    │   ║
║  │     └──────────────────────────────────────────────────────────┘    │   ║
║  │     You review, optionally tighten, and click APPROVE              │   ║
║  │                                                                    │   ║
║  │  ⑥ FEEDBACK INGESTION (local ChromaDB)                             │   ║
║  │     • Free-text feedback scrubbed for PII:                         │   ║
║  │       "John Smith said..." → "[NAME] said..."                      │   ║
║  │     • Employee IDs hashed (SHA-256, first 16 chars)                │   ║
║  │     • Scrubbed text embedded via provider (see step ⑦a)            │   ║
║  │     • Vectors stored locally in ChromaDB                           │   ║
║  └──────────────────────────┬──────────────────────────────────────────┘   ║
║                             │                                             ║
║  ╔══════════════════════════▼══════════════════════════════════════════╗   ║
║  ║  🔒 AGGREGATION & SCRUBBING LAYER (Security Boundary)              ║   ║
║  ║                                                                    ║   ║
║  ║  Everything below this line is ANONYMIZED before leaving:          ║   ║
║  ║  • PII columns → completely removed                               ║   ║
║  ║  • Quasi-identifiers → replaced with "[aggregated]"               ║   ║
║  ║  • Employee IDs → hashed                                          ║   ║
║  ║  • Feedback text → PII-scrubbed (names/emails/phones replaced)    ║   ║
║  ║  • Numbers → only department-level aggregates (means, counts)     ║   ║
║  ╚══════════════════════════╤══════════════════════════════════════════╝   ║
║                             │                                             ║
║  ⑦ THREE REQUESTS CROSS THE NETWORK (aggregated data only):              ║
║                             │                                             ║
║     ┌───────────────────────┼───────────────────────────────────┐         ║
║     │                       │                                   │         ║
║   ⑦a EMBED              ⑦b SCHEMA + PLAN                 ⑦c INSIGHTS    ║
║   Scrubbed text →        Column names +                   Dept stats +   ║
║   embedding vectors      dtypes + safe                    correlations + ║
║   (for ChromaDB)         sample rows →                    scrubbed       ║
║                          column mapping +                 feedback →     ║
║                          analysis plan                    executive      ║
║                                                           summary +     ║
║                                                           charts +      ║
║                                                           recommendations║
║     │                       │                                   │         ║
╚═════╪═══════════════════════╪═══════════════════════════════════╪═════════╝
      │  HTTPS (encrypted)    │                                   │
      ▼                       ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PROVIDER BACKEND (Cloud / Render)                     │
│                                                                        │
│   ⑧ GPT-4o ORCHESTRATION (stateless — nothing stored)                  │
│      • Receives aggregated stats → generates narrative insights        │
│      • Receives column names → decides analysis plan                   │
│      • Receives scrubbed text → writes sentiment analysis              │
│      • Calls OpenAI API with your data + system prompts                │
│      • Returns JSON responses                                          │
│      • Immediately discards all inputs after response                  │
│                                                                        │
│   ┌────────────────┐                                                   │
│   │   OpenAI API   │  Sees only: aggregated stats, scrubbed text,      │
│   │   (GPT-4o)     │  column names. Never sees raw employee data.      │
│   └────────┬───────┘                                                   │
│            │                                                           │
│   ⑨ Returns: executive summary, chart specs, recommendations (JSON)   │
│                                                                        │
└────────────┼───────────────────────────────────────────────────────────┘
             │
             ▼
┌══════════════════════════════════════════════════════════════════════════┐
║                    YOUR ENVIRONMENT (back to client)                    ║
║                                                                        ║
║  ⑩ LOCAL ML PIPELINE (Deep Mode — runs entirely on your machine)       ║
║     • scikit-learn: RandomForest vs Logistic Regression (5-fold CV)    ║
║     • Feature importance, risk scoring, what-if scenarios              ║
║     • lifelines: Kaplan-Meier survival analysis                        ║
║     • KMeans clustering with PCA visualization                         ║
║     • PII columns excluded from ML features automatically             ║
║     • Employee IDs hashed in risk score output                         ║
║     • Trained models NEVER leave your machine                          ║
║                                                                        ║
║  ⑪ RESULTS RENDERED IN STREAMLIT                                       ║
║     • KPI cards, interactive Plotly charts, department breakdowns      ║
║     • AI narrative, recommendations, ML insights                       ║
║     • Ask AI: your question + cached context → provider → answer       ║
║     • All displayed data stays in browser memory                       ║
║                                                                        ║
║  ⑫ DATA EXPIRY                                                         ║
║     • Auto-expires after 24 hours (in-memory TTL)                      ║
║     • Or: click "Delete Data" for immediate erasure                    ║
║     • ChromaDB collection deleted, DataFrame purged, cache cleared     ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**What never leaves your environment:**

| Data Type | Stays Local? | Proof |
|-----------|:---:|--------|
| Raw CSV rows | ✅ | Loaded into pandas in-memory, never serialized to network |
| Employee names, emails, SSNs, phones | ✅ | Dropped by PII classifier before any network call |
| Individual salary/compensation values | ✅ | Replaced with `[aggregated]` — only dept averages sent |
| Trained ML models (sklearn objects) | ✅ | Fit and predict run locally, model never serialized |
| ChromaDB vector database | ✅ | Stored on local disk, embeddings computed via proxy |
| Original feedback text (pre-scrubbing) | ✅ | Scrubbed copy sent; original stays in ChromaDB locally |
| Risk scores per individual employee | ✅ | Computed locally, employee IDs hashed in output |

**What crosses the network (anonymized only):**

| Data Sent | Example | Why |
|-----------|---------|-----|
| Column names + data types | `{"Department": "object", "Age": "int64"}` | GPT-4o needs schema to plan analyses |
| Safe sample values | `{"Department": "Engineering", "Age": 34}` | Context for column mapping |
| Dept-level aggregates | `{"Sales": {headcount: 120, attrition_rate: 0.23}}` | GPT-4o writes narrative from stats |
| Correlation numbers | `[{factor: "overtime", correlation: 0.34}]` | For recommendations |
| PII-scrubbed feedback | `"[NAME] said work-life balance is poor"` | Sentiment and theme analysis |
| Embedding text (scrubbed) | Same scrubbed feedback | ChromaDB needs vectors |

---

### The 3-Step PII Review Process

Before any analysis begins, every column in your data goes through a rigorous
PII classification and approval workflow:

```
  Step 1: Upload           Step 2: PII Review         Step 3: Approve
 ┌──────────────┐       ┌────────────────────┐       ┌──────────────────┐
 │  Upload CSV  │       │  Every column is   │       │  You review and  │
 │  + JSON to   │──────►│  classified by     │──────►│  approve (or     │
 │  local       │       │  Python + AI       │       │  tighten) each   │
 │  container   │       │  into 4 PII tiers  │       │  classification  │
 └──────────────┘       └────────────────────┘       └──────────────────┘
                                                              │
                             Analysis proceeds only ◄─────────┘
                             AFTER explicit user approval
```

**Step 1 — Upload:** Your CSV and JSON files are uploaded to the **local** client
container. Data stays in-memory (with 24-hour auto-expiry). Nothing is sent anywhere yet.

**Step 2 — PII Classification:** A deterministic Python classifier scans every column
using regex patterns, uniqueness checks, and value sampling. Then AI validates the
classifications (it can only *tighten* — never loosen — a classification). Each column
is assigned one of four tiers:

| PII Tier | Examples | Handling | Sent to AI? |
|----------|----------|----------|:-----------:|
| 🔴 **Identifier** | Employee ID, SSN | **Excluded** completely | ❌ Never |
| 🟠 **Direct PII** | Name, email, phone, address | **Excluded** completely | ❌ Never |
| 🟡 **Quasi-Identifier** | Date of birth, exact salary, zip code | **Aggregated only** (averages, ranges) | ⚠️ Aggregated |
| 🟢 **Safe** | Department, job role, tenure bands | **Passed through** | ✅ Yes |

**Step 3 — User Approval:** You see a full table of every column, its detected PII
category, and proposed handling. You can **upgrade** any column to a stricter tier
(e.g., mark "salary" as Direct PII instead of Quasi-Identifier), but you **cannot
downgrade** — this is a one-way ratchet for safety. Feedback text samples are also
shown with detected PII highlighted and scrubbed versions displayed side-by-side.

Analysis **will not proceed** until you explicitly click "Approve & Run Analysis."

---

### What Exactly Is Sent to the Cloud?

After PII review and approval, the client container prepares an anonymized payload.
Here is a precise breakdown:

**✅ SENT to Provider (anonymized/aggregated):**
- Column names and data types (e.g., "Department: string")
- Aggregated statistics (department averages, distribution counts)
- PII-scrubbed feedback text (names/emails/IDs replaced with `[PERSON]`, `[EMAIL]`, etc.)
- ML results (feature importance scores, risk distributions — no individual records)
- Department-level breakdowns (headcount, average scores)

**❌ NEVER SENT to Provider:**
- Raw CSV rows or individual employee records
- Employee names, emails, phone numbers, or addresses
- Employee IDs (even hashed versions stay local)
- Exact salary figures (only aggregated ranges)
- Social Security Numbers, bank details, or government IDs
- The uploaded CSV or JSON files themselves

---

### Feedback Text PII Scrubbing

Employee feedback text gets special treatment because free-text can contain
any kind of personal information:

```
  BEFORE SCRUBBING (stays local only):
  "John Smith in Building 7 told me at john@company.com that
   his manager Sarah gives unfair reviews. Call me at 555-0147."

  AFTER SCRUBBING (this is what the AI sees):
  "[PERSON] in Building 7 told me at [EMAIL] that
   his manager [PERSON] gives unfair reviews. Call me at [PHONE]."
```

The scrubber detects and redacts: **names, emails, phone numbers, SSNs,
street addresses, dates of birth, and employee IDs** — using both regex
patterns and context-aware heuristics.

---

### Embedding Security

When feedback text needs to be converted to vector embeddings (for semantic
search), the text is:
1. **PII-scrubbed first** (locally)
2. Sent to the provider's `/api/embed` proxy endpoint
3. The provider calls OpenAI's embedding API and returns vectors
4. Vectors are stored **locally** in ChromaDB

The provider serves as a **key proxy only** — it holds the OpenAI API key
so your client container doesn't need it. The provider never stores the
text or vectors.

---

### ML & Analytics: All Local

Predictive analytics (Deep mode) run **entirely within your container**:

| ML Capability | Runs Where | Libraries |
|---------------|:----------:|:---------:|
| Feature Importance | 🏠 Local | scikit-learn |
| Risk Scoring | 🏠 Local | scikit-learn |
| Survival Analysis | 🏠 Local | lifelines |
| Employee Clustering | 🏠 Local | scikit-learn |
| What-If Scenarios | 🏠 Local | scikit-learn |

Only the **aggregated results** (e.g., "top 5 risk predictors", "3 employee
segments with average profiles") are sent to the provider for narrative
generation by GPT-4o.

---

### GDPR Compliance Features

| GDPR Article | Feature | Implementation |
|:---:|---------|----------------|
| **Art. 5** | Data Minimisation | Only aggregated/anonymized data sent to AI |
| **Art. 13** | Transparency | PII handling summary displayed on every dashboard |
| **Art. 17** | Right to Erasure | One-click "Delete All Data" removes everything |
| **Art. 25** | Privacy by Design | Two-tier architecture, PII classifier, one-way ratchet |
| **Art. 30** | Record of Processing | Audit log tracks all data processing events |
| **Art. 32** | Security of Processing | Encrypted transport, no persistent storage on provider |
| **Art. 35** | Impact Assessment | PII review screen serves as built-in DPIA checkpoint |

---

### Data Lifecycle & Auto-Expiry

```
  Upload ──► PII Review ──► Analysis ──► Results Displayed ──► Auto-Expire
    │                                                              │
    │            24 hours max in-memory                             │
    │◄─────────────────────────────────────────────────────────────►│
    │                                                              │
    └─── OR: User clicks "Delete Data" for immediate erasure ──────┘
```

- **In-memory only:** Data is held in Python process memory, not written to disk databases
- **Auto-expiry:** Datasets are automatically purged after 24 hours
- **Manual deletion:** The sidebar "Delete Data" button immediately removes all data,
  embeddings, cached insights, and audit entries for a company
- **No provider persistence:** The provider backend is stateless — it processes
  requests and discards all context after responding

---

### Privacy Indicators in the Dashboard

Throughout the dashboard, you'll see privacy indicators:

- **🔒 Privacy-Aware Analytics** banner — lists which columns were excluded or modified
- **SHA-256 hash badges** — employee IDs in risk tables are cryptographic hashes, not real IDs
- **PII handling summary** — shows exactly how each sensitive column was treated
- **"Privacy Note"** callouts — appear next to any chart that references PII-adjacent data

---

### FAQ: Security Questions Clients Ask

| Question | Answer |
|----------|--------|
| *Can the cloud provider see my employee names?* | **No.** All PII is stripped before any data leaves your container. |
| *Is data stored on the cloud server?* | **No.** The provider is stateless — it processes and forgets. |
| *What if the AI misses PII in feedback text?* | The scrubber uses multiple detection layers (regex + context). You also review samples before approval. |
| *Can I make a column more private than the system suggests?* | **Yes.** You can upgrade any column to a stricter tier. You just can't downgrade. |
| *Where are my files stored?* | In your local container's memory only. Never on the provider's server. |
| *Do you use my data to train AI models?* | **No.** OpenAI API calls use the data processing agreement — your data is not used for training. |
| *What happens if I close the browser?* | Data persists in the container for up to 24 hours, then auto-expires. |
| *Can I run everything on-premises?* | **Yes.** Deploy both containers in your own infrastructure with a self-hosted LLM. |

---

*Powered by GPT-4o · FastAPI · Streamlit · ChromaDB · Plotly — Privacy by Design*
    """)


if insights is None:
    # Landing state
    st.title("📊 HCM AI Insights Dashboard")

    landing_tab1, landing_tab2 = st.tabs(["🏠 Welcome", "📖 User Guide & Privacy"])

    with landing_tab1:
        st.markdown("""
        ### Welcome to the AI-Powered Workforce Analytics Prototype

        This tool analyzes employee data and qualitative feedback
        to automatically generate actionable insights using GPT-4o.

        **How to use:**
        1. Enter a company name in the sidebar
        2. Upload a structured CSV file and a qualitative feedback JSON file
        3. Choose an **Analysis Mode** — ⚡ Quick Insights (Descriptive) or 🔬 Deep Analysis (Descriptive & Predictive)
        4. Click **Upload & Review Schema** — the system classifies every column for PII
        5. Review the PII classifications, tighten protections if needed, then click **Approve & Run Analysis**
        6. Explore the generated tabs!

        **Or try a sample dataset** using the buttons in the sidebar.

        ---

        #### What you'll get:
        - 🧠 **AI Insights** — Executive summary, KPI snapshot, and prioritized recommendations
        - 📊 **Descriptive Analytics** — Structured analysis, employee feedback, correlations, with built-in Ask AI and chart customization
        - 🔮 **Predictive Analytics** *(Deep mode)* — Feature importance, risk scoring, survival analysis, clustering, what-if scenarios

        ---

        #### Special Features
        - **Zero-Config Detection** — AI auto-detects your CSV schema, target variable, and JSON feedback keys
        - **Works Across Industries** — Tech, retail, healthcare, finance — any HCM dataset works out of the box
        - **Conversational AI** — Ask follow-up questions and customize any chart via natural language
        - **ML Predictive Analytics** — Feature importance, risk scoring, survival analysis, clustering, what-if scenarios *(Deep mode)*
        - **11-Step AI Pipeline** — Schema → plan → execution → retrieval → sentiment → themes → correlation → summary → recommendations → dashboard

        *Check the* 📖 **User Guide & Privacy** *tab above for full documentation and data security details.*
        """)

    with landing_tab2:
        _render_user_guide()

else:
    # ── Dashboard with insights ──────────────────────────────────────

    st.title(f"📊 {insights['company_name']} — AI Insights Dashboard")

    # Show analysis mode badge
    mode = insights.get("analysis_mode", "quick")
    if mode == "deep":
        st.caption("🔬 **Deep Analysis** — Descriptive & Predictive analytics")
    else:
        st.caption("⚡ **Quick Insights** — Descriptive analytics")

    # ── Helper: render a dynamic section ─────────────────────────────

    def _render_section(section: dict, key_prefix: str):
        """Render an AnalysisSection: metrics, narrative, charts."""
        metrics = section.get("metrics", [])
        if metrics:
            cols = st.columns(min(len(metrics), 4))
            for i, m in enumerate(metrics):
                with cols[i % len(cols)]:
                    st.metric(m.get("label", ""), m.get("value", ""))
        narrative = section.get("narrative", "")
        if narrative:
            st.markdown(narrative)
        for j, chart in enumerate(section.get("charts", [])):
            chart_dict = chart if isinstance(chart, dict) else chart.dict() if hasattr(chart, "dict") else {}
            _render_dynamic_chart(chart_dict, key=f"{key_prefix}_chart_{j}")

    # ── Dynamic KPI Cards ────────────────────────────────────────────

    dynamic_kpis = insights.get("kpis", [])
    if dynamic_kpis:
        kpi_cols = st.columns(min(len(dynamic_kpis), 6))
        for i, kpi in enumerate(dynamic_kpis):
            with kpi_cols[i % len(kpi_cols)]:
                st.metric(
                    label=kpi.get("label", ""),
                    value=kpi.get("value", ""),
                    help=kpi.get("description", ""),
                )

    st.markdown("---")

    # ── Separate sections by category ────────────────────────────────

    all_sections = insights.get("sections", [])
    structured_sections = [s for s in all_sections if s.get("category") == "structured"]
    voe_sections = [s for s in all_sections if s.get("category") == "voe"]
    correlation_sections = [s for s in all_sections if s.get("category") == "correlation"]

    # ── Embedded Ask AI helper ───────────────────────────────────────

    def _render_ask_ai(context_key: str):
        """Render an embedded Ask AI section using text_input + button."""
        import json as _json

        chat_key = f"{context_key}_chat_history"
        api_key = f"{context_key}_api_history"

        st.markdown("### 💡 Ask AI")
        st.caption(
            "Ask follow-up questions in natural language. The AI has full context "
            "of the insights already generated, plus access to the raw data."
        )

        # Render previous messages
        for mi, msg in enumerate(st.session_state[chat_key]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    _render_dynamic_chart(msg["chart"], key=f"{context_key}_ask_hist_{mi}")

        # Text input + send button
        ask_col1, ask_col2 = st.columns([5, 1])
        with ask_col1:
            user_q = st.text_input(
                "Your question",
                placeholder="e.g., Show me attrition by age group in Engineering",
                key=f"{context_key}_ask_input",
                label_visibility="collapsed",
            )
        with ask_col2:
            send = st.button("Send", key=f"{context_key}_ask_send", use_container_width=True, type="primary")

        if send and user_q:
            st.session_state[chat_key].append({"role": "user", "content": user_q})
            st.session_state[api_key].append({"role": "user", "content": user_q})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = api_client.ask_question(
                            company_name=insights["company_name"],
                            question=user_q,
                            conversation_history=st.session_state[api_key],
                        )
                        answer = result.get("answer", "Sorry, I couldn't generate a response.")
                        chart_spec = result.get("chart")

                        st.markdown(answer)
                        if chart_spec:
                            _render_dynamic_chart(chart_spec, key=f"{context_key}_ask_new_{len(st.session_state[chat_key])}")

                        st.session_state[chat_key].append({
                            "role": "assistant", "content": answer, "chart": chart_spec,
                        })
                        st.session_state[api_key].append({
                            "role": "assistant", "content": answer,
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state[chat_key].append({
                            "role": "assistant", "content": error_msg, "chart": None,
                        })
            st.rerun()

    # ── Embedded Explore & Customize helper ──────────────────────────

    def _render_explore(context_key: str, sections_to_show: list):
        """Render an embedded Explore & Customize section."""
        import json as _json

        hist_key = f"{context_key}_explore_history"
        api_key = f"{context_key}_explore_api_history"
        charts_key = f"{context_key}_explore_charts"

        st.markdown("### ✏️ Explore & Customize")
        st.caption(
            "Modify any chart above — change colors, chart types, filters, or "
            "create entirely new visualizations."
        )

        # AI-customized charts
        if st.session_state[charts_key]:
            st.markdown("#### 🤖 Custom Charts")
            for ei, (title, spec) in enumerate(st.session_state[charts_key].items()):
                st.markdown(f"**{title}**")
                _render_dynamic_chart(spec, key=f"{context_key}_exp_custom_{ei}")

        # Chat history
        for mi, msg in enumerate(st.session_state[hist_key]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    _render_dynamic_chart(msg["chart"], key=f"{context_key}_exp_hist_{mi}")

        # Text input + send button
        exp_col1, exp_col2 = st.columns([5, 1])
        with exp_col1:
            explore_prompt = st.text_input(
                "Customize",
                placeholder="e.g., Color all bars green, Show attrition as a pie chart...",
                key=f"{context_key}_exp_input",
                label_visibility="collapsed",
            )
        with exp_col2:
            exp_send = st.button("Send", key=f"{context_key}_exp_send", use_container_width=True, type="primary")

        if exp_send and explore_prompt:
            # Build context from provided sections
            all_chart_context = []
            for sec in sections_to_show:
                sec_dict = sec if isinstance(sec, dict) else sec.dict() if hasattr(sec, "dict") else {}
                all_chart_context.append({
                    "title": sec_dict.get("title", ""),
                    "category": sec_dict.get("category", ""),
                    "narrative": sec_dict.get("narrative", ""),
                    "charts": [
                        (c if isinstance(c, dict) else c.dict() if hasattr(c, "dict") else {})
                        for c in sec_dict.get("charts", [])
                    ],
                })
            full_q = (
                f"{explore_prompt}\n\n"
                f"The user is viewing these dashboard charts:\n"
                f"{_json.dumps(all_chart_context, default=str)}\n"
                f"Modify an existing chart or create a new visualization based on the request. "
                f"Always include a chart_spec in your response."
            )

            st.session_state[hist_key].append({"role": "user", "content": explore_prompt})
            st.session_state[api_key].append({"role": "user", "content": full_q})

            with st.chat_message("assistant"):
                with st.spinner("Customizing..."):
                    try:
                        result = api_client.ask_question(
                            company_name=insights["company_name"],
                            question=full_q,
                            conversation_history=st.session_state[api_key],
                        )
                        answer = result.get("answer", "")
                        chart_spec = result.get("chart")

                        st.markdown(answer)
                        if chart_spec:
                            _render_dynamic_chart(chart_spec, key=f"{context_key}_exp_new_{len(st.session_state[hist_key])}")
                            ct = chart_spec.get("title", explore_prompt[:40])
                            st.session_state[charts_key][ct] = chart_spec

                        st.session_state[hist_key].append({
                            "role": "assistant", "content": answer, "chart": chart_spec,
                        })
                        st.session_state[api_key].append({
                            "role": "assistant", "content": answer,
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state[hist_key].append({
                            "role": "assistant", "content": error_msg, "chart": None,
                        })
            st.rerun()

        # Reset
        if st.session_state[hist_key]:
            if st.button("🔄 Reset Customizations", key=f"{context_key}_exp_reset"):
                st.session_state[hist_key] = []
                st.session_state[api_key] = []
                st.session_state[charts_key] = {}
                st.rerun()

    # ── Build dynamic tab list ────────────────────────────────────────
    tab_labels = ["🧠 AI Insights", "📊 Descriptive Analytics"]
    has_ml_tab = mode == "deep"
    if has_ml_tab:
        tab_labels.append("🔮 Predictive Analytics")
    tab_labels.append("📖 User Guide & Privacy")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── Tab 1: AI Insights ───────────────────────────────────────────

    with tabs[tab_idx]:
        # Executive Summary
        st.subheader("📋 Executive Summary")
        st.markdown(
            f'<div class="insight-card">{insights["executive_summary"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Recommendations
        st.subheader("🎯 Recommendations")
        dynamic_recs = insights.get("recommendations", [])
        if dynamic_recs:
            for rec in dynamic_recs:
                risk = rec.get("risk_level", "Medium")
                title = rec.get("title", "General")
                icon = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                with st.expander(f"{icon} {title} — {risk} Risk"):
                    if rec.get("key_issues"):
                        st.markdown("**Key Issues:**")
                        for issue in rec["key_issues"]:
                            st.markdown(f"- {issue}")
                    if rec.get("recommendations"):
                        st.markdown("**Recommendations:**")
                        for i, r in enumerate(rec["recommendations"], 1):
                            st.markdown(f"{i}. {r}")

    # ── Tab 2: Descriptive Analytics ─────────────────────────────────
    tab_idx += 1

    with tabs[tab_idx]:
        # Structured analysis sections
        if structured_sections:
            target_desc = insights.get("target_description", "").strip()
            if target_desc:
                short = target_desc.replace("Employee ", "")
                st.subheader(f"📉 {short} Analysis")
            else:
                st.subheader("📉 Structured Analysis")

            for idx, section in enumerate(structured_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.markdown(f"#### {sec_dict.get('title', f'Analysis {idx + 1}')}")
                _render_section(sec_dict, f"struct_{idx}")
                if idx < len(structured_sections) - 1:
                    st.markdown("---")

            st.markdown("---")

        # Voice of Employee sections
        if voe_sections:
            st.subheader("💬 Employee Feedback")

            for idx, section in enumerate(voe_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.markdown(f"#### {sec_dict.get('title', f'Feedback Analysis {idx + 1}')}")
                _render_section(sec_dict, f"voe_{idx}")
                if idx < len(voe_sections) - 1:
                    st.markdown("---")

            st.markdown("---")

        # Correlation sections
        if correlation_sections:
            st.subheader("🔗 Correlation Insights")

            for idx, section in enumerate(correlation_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.markdown(f"#### {sec_dict.get('title', 'Correlation Analysis')}")
                _render_section(sec_dict, f"corr_{idx}")
                if idx < len(correlation_sections) - 1:
                    st.markdown("---")

            st.markdown("---")

        if not structured_sections and not voe_sections and not correlation_sections:
            st.info("No analysis sections were generated. Run Analyze first.")

        # Embedded Ask AI + Explore
        st.markdown("---")
        _render_ask_ai("desc")
        st.markdown("---")
        _render_explore("desc", all_sections)

    # ── Tab 3: Predictive Analytics (deep mode only) ─────────────────
    if has_ml_tab:
        tab_idx += 1
        with tabs[tab_idx]:
            _render_ml_tab(insights)

            # Embedded Ask AI + Explore
            st.markdown("---")
            _render_ask_ai("pred")
            st.markdown("---")
            _render_explore("pred", all_sections)

    # ── Tab 4: User Guide & Privacy ────────────────────────────────────
    tab_idx += 1

    with tabs[tab_idx]:
        _render_user_guide()
