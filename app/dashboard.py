"""
HCM AI Insights Dashboard — Streamlit entry point.

This is the main UI for the prototype. It provides:
  - Sidebar: Company name input + file uploaders + Analyze button
  - 4 tabs: Overview | Attrition Analysis | Voice of Employee | AI Insights

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.components import api_client
from app.components.kpi_cards import render_kpi_cards
from app.components import charts

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
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None

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
    company_name = st.text_input(
        "Company Name",
        value=st.session_state.company_name,
        placeholder="e.g., NovaTech Solutions",
    )

    csv_file = st.file_uploader(
        "Attrition Data (CSV)",
        type=["csv"],
        help="Upload the employee attrition CSV file",
    )

    json_file = st.file_uploader(
        "Voice of Employee (JSON)",
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

    analyze_button = st.button(
        "🚀 Analyze",
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
        result = api_client.upload_attrition_csv(csv_bytes, "employees.csv", name)
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
    return True


if sample_company:
    name, slug = sample_company
    try:
        if load_sample_data(name, slug):
            st.sidebar.success(f"✅ {name} data loaded!")
            company_name = name
    except Exception as e:
        st.sidebar.error(f"Failed to load sample data: {e}")


# ── Handle manual upload + analyze ───────────────────────────────────

if analyze_button:
    if not company_name:
        st.sidebar.error("Please enter a company name")
    elif not csv_file and not st.session_state.upload_status["csv"]:
        st.sidebar.error("Please upload an attrition CSV file")
    elif not json_file and not st.session_state.upload_status["json"]:
        st.sidebar.error("Please upload a VoE JSON file")
    else:
        try:
            # Upload files if provided
            if csv_file:
                with st.spinner("Uploading attrition data..."):
                    result = api_client.upload_attrition_csv(
                        csv_file.getvalue(), csv_file.name, company_name
                    )
                    if result.get("success"):
                        st.session_state.upload_status["csv"] = True
                    else:
                        st.sidebar.error(f"CSV upload failed: {result.get('message')}")

            if json_file:
                with st.spinner("Uploading feedback data..."):
                    result = api_client.upload_feedback_json(
                        json_file.getvalue(), json_file.name, company_name
                    )
                    if result.get("success"):
                        st.session_state.upload_status["json"] = True
                    else:
                        st.sidebar.error(f"JSON upload failed: {result.get('message')}")

            # Generate insights
            if st.session_state.upload_status["csv"] and st.session_state.upload_status["json"]:
                with st.spinner("🤖 AI is analyzing your data... This may take 30-60 seconds."):
                    insights = api_client.generate_insights(company_name)
                    st.session_state.insights = insights
                    st.session_state.company_name = company_name
            else:
                st.sidebar.error("Both files must be uploaded before analysis")

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")


# ── Main content ─────────────────────────────────────────────────────

insights = st.session_state.insights

if insights is None:
    # Landing state
    st.title("📊 HCM AI Insights Dashboard")
    st.markdown("""
    ### Welcome to the AI-Powered Workforce Analytics Prototype

    This tool analyzes employee attrition data and Voice of Employee feedback
    to automatically generate actionable insights using GPT-4o.

    **How to use:**
    1. Enter a company name in the sidebar
    2. Upload an attrition CSV file and a Voice of Employee JSON file
    3. Click **Analyze** — the AI will mine patterns and generate insights

    **Or try a sample dataset** using the buttons in the sidebar.

    ---

    #### What you'll get:
    - 📈 **Attrition Analysis** — Risk factors, department-level breakdowns
    - 💬 **Sentiment Analysis** — AI-classified employee sentiment by department
    - 🔍 **Theme Extraction** — Top topics surfaced from employee feedback
    - 🧠 **AI Executive Summary** — GPT-4o generated narrative with recommendations
    - 🔗 **Correlation Insights** — Sentiment vs attrition cross-analysis
    """)

else:
    # ── Dashboard with insights ──────────────────────────────────────

    st.title(f"📊 {insights['company_name']} — AI Insights Dashboard")

    # KPI Cards
    render_kpi_cards(insights["summary_kpis"])

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Overview",
        "📉 Attrition Analysis",
        "💬 Voice of Employee",
        "🧠 AI Insights",
    ])

    # ── Tab 1: Overview ──────────────────────────────────────────────

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig = charts.attrition_by_department_bar(insights["department_attrition"])
            st.plotly_chart(fig, use_container_width=True, key="overview_attrition_bar")

        with col2:
            fig = charts.overtime_comparison_bar(insights["overtime_analysis"])
            st.plotly_chart(fig, use_container_width=True, key="overview_overtime_bar")

        # Department comparison table
        fig = charts.department_stats_table(insights["department_stats"])
        st.plotly_chart(fig, use_container_width=True, key="overview_dept_table")

    # ── Tab 2: Attrition Analysis ────────────────────────────────────

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            fig = charts.risk_factors_bar(insights["risk_factors"])
            st.plotly_chart(fig, use_container_width=True, key="attrition_risk_bar")

        with col2:
            fig = charts.attrition_by_department_bar(insights["department_attrition"])
            st.plotly_chart(fig, use_container_width=True, key="attrition_dept_bar")

        # Overtime deep-dive
        st.subheader("Overtime Impact")
        ot = insights["overtime_analysis"]
        ot_col1, ot_col2, ot_col3, ot_col4 = st.columns(4)
        with ot_col1:
            st.metric("Overtime Workers", f"{ot['overtime_headcount']:,}")
        with ot_col2:
            st.metric("OT Attrition Rate", f"{ot['overtime_attrition_rate']*100:.1f}%")
        with ot_col3:
            st.metric("Non-OT Workers", f"{ot['no_overtime_headcount']:,}")
        with ot_col4:
            st.metric("Non-OT Attrition", f"{ot['no_overtime_attrition_rate']*100:.1f}%")

    # ── Tab 3: Voice of Employee ─────────────────────────────────────

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            fig = charts.sentiment_distribution_pie(insights["sentiment_by_department"])
            st.plotly_chart(fig, use_container_width=True, key="voe_sentiment_pie")

        with col2:
            fig = charts.sentiment_by_department_bar(insights["sentiment_by_department"])
            st.plotly_chart(fig, use_container_width=True, key="voe_sentiment_bar")

        # Themes
        fig = charts.theme_frequency_bar(insights["themes"])
        st.plotly_chart(fig, use_container_width=True, key="voe_themes_bar")

        # Sample quotes per theme
        st.subheader("📝 Sample Feedback by Theme")
        for theme in insights["themes"]:
            with st.expander(f"**{theme['theme']}** ({theme.get('count', '?')} mentions)"):
                for quote in theme.get("sample_quotes", []):
                    st.markdown(f"> _{quote}_")

    # ── Tab 4: AI Insights ───────────────────────────────────────────

    with tab4:
        # Executive Summary
        st.subheader("📋 Executive Summary")
        st.markdown(
            f'<div class="insight-card">{insights["executive_summary"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Sentiment vs Attrition correlation
        st.subheader("🔗 Sentiment–Attrition Correlation")
        if insights.get("correlations"):
            fig = charts.sentiment_vs_attrition_scatter(insights["correlations"])
            st.plotly_chart(fig, use_container_width=True, key="ai_correlation_scatter")

            # Narrative per department
            for corr in insights["correlations"]:
                dept = corr["department"]
                narrative = corr.get("narrative", "")
                if narrative:
                    st.markdown(f"**{dept}:** {narrative}")

        st.markdown("---")

        # Retention Recommendations
        st.subheader("🎯 Retention Recommendations by Department")
        for rec in insights.get("retention_recommendations", []):
            risk = rec.get("risk_level", "Medium")
            css_class = f"risk-{risk.lower()}"

            with st.expander(
                f"{'🔴' if risk == 'High' else '🟡' if risk == 'Medium' else '🟢'} "
                f"{rec['department']} — {risk} Risk"
            ):
                st.markdown("**Key Issues:**")
                for issue in rec.get("key_issues", []):
                    st.markdown(f"- {issue}")

                st.markdown("**Recommendations:**")
                for i, r in enumerate(rec.get("recommendations", []), 1):
                    st.markdown(f"{i}. {r}")
