"""
HCM AI Insights Dashboard — Streamlit entry point.

This is the main UI for the prototype. It provides:
  - Sidebar: Company name input + file uploaders + Analyze button
  - 6 tabs: Overview | Structured Analysis | Voice of Employee | AI Insights | Ask AI | Explore & Customize

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.components import api_client
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
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_api_history" not in st.session_state:
    st.session_state.conversation_api_history = []
if "explore_history" not in st.session_state:
    st.session_state.explore_history = []
if "explore_api_history" not in st.session_state:
    st.session_state.explore_api_history = []
if "explore_charts" not in st.session_state:
    st.session_state.explore_charts = {}

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

    This tool analyzes employee data and qualitative feedback
    to automatically generate actionable insights using GPT-4o.

    **How to use:**
    1. Enter a company name in the sidebar
    2. Upload a structured CSV file and a qualitative feedback JSON file
    3. Click **Analyze** — the AI will mine patterns and generate insights

    **Or try a sample dataset** using the buttons in the sidebar.

    ---

    #### What you'll get:
    - 📈 **Structured Analysis** — Risk factors, department-level breakdowns, key metrics
    - 💬 **Sentiment Analysis** — AI-classified employee sentiment by department
    - 🔍 **Theme Extraction** — Top topics surfaced from employee feedback
    - 🧠 **AI Executive Summary** — GPT-4o generated narrative with recommendations
    - 🔗 **Correlation Insights** — Qualitative vs quantitative cross-analysis
    """)

else:
    # ── Dashboard with insights ──────────────────────────────────────

    st.title(f"📊 {insights['company_name']} — AI Insights Dashboard")

    # ── Helper: render a dynamic section ─────────────────────────────

    def _render_section(section: dict, key_prefix: str):
        """Render an AnalysisSection: metrics, narrative, charts."""
        # Metrics row
        metrics = section.get("metrics", [])
        if metrics:
            cols = st.columns(min(len(metrics), 4))
            for i, m in enumerate(metrics):
                with cols[i % len(cols)]:
                    st.metric(m.get("label", ""), m.get("value", ""))
        # Narrative
        narrative = section.get("narrative", "")
        if narrative:
            st.markdown(narrative)
        # Charts
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

    # ── Dynamic tab names based on what was analyzed ─────────────────
    target_desc = insights.get("target_description", "").strip()
    if target_desc:
        # e.g. "Employee Attrition" → "Attrition Analysis", "Engagement Score" → "Engagement Score Analysis"
        short = target_desc.replace("Employee ", "")
        structured_tab_label = f"📉 {short} Analysis"
    else:
        structured_tab_label = "📉 Structured Analysis"

    # Derive VoE tab name from section titles if available
    if voe_sections:
        first_voe = voe_sections[0] if isinstance(voe_sections[0], dict) else {}
        voe_hint = first_voe.get("title", "")
        if "sentiment" in voe_hint.lower() or "feedback" in voe_hint.lower():
            voe_tab_label = "💬 Employee Feedback"
        else:
            voe_tab_label = f"💬 {voe_hint}" if voe_hint else "💬 Qualitative Insights"
    else:
        voe_tab_label = "💬 Qualitative Insights"

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview",
        structured_tab_label,
        voe_tab_label,
        "🧠 AI Insights",
        "💡 Ask AI",
        "✏️ Explore & Customize",
    ])

    # ── Tab 1: Overview ──────────────────────────────────────────────

    with tab1:
        if all_sections:
            # Show first chart from each section as an overview
            overview_charts = []
            for sec in all_sections:
                sec_charts = sec.get("charts", [])
                if sec_charts:
                    first = sec_charts[0]
                    overview_charts.append((sec.get("title", ""), first))

            for idx in range(0, len(overview_charts), 2):
                cols = st.columns(2)
                for ci, col in enumerate(cols):
                    if idx + ci < len(overview_charts):
                        title, chart_data = overview_charts[idx + ci]
                        with col:
                            chart_dict = chart_data if isinstance(chart_data, dict) else chart_data.dict() if hasattr(chart_data, "dict") else {}
                            _render_dynamic_chart(chart_dict, key=f"overview_{idx}_{ci}")
        else:
            st.info("No analysis sections available. Run Analyze first.")

    # ── Tab 2: Structured Analysis (dynamic) ─────────────────────────

    with tab2:
        if structured_sections:
            for idx, section in enumerate(structured_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.subheader(sec_dict.get("title", f"Analysis {idx + 1}"))
                _render_section(sec_dict, f"struct_{idx}")
                if idx < len(structured_sections) - 1:
                    st.markdown("---")
        else:
            st.info("No structured analysis sections were generated. Run Analyze first.")

    # ── Tab 3: Voice of Employee (dynamic) ───────────────────────────

    with tab3:
        if voe_sections:
            for idx, section in enumerate(voe_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.subheader(sec_dict.get("title", f"Feedback Analysis {idx + 1}"))
                _render_section(sec_dict, f"voe_{idx}")
                if idx < len(voe_sections) - 1:
                    st.markdown("---")
        else:
            st.info("No qualitative analysis sections were generated. Run Analyze first.")

    # ── Tab 4: AI Insights ───────────────────────────────────────────

    with tab4:
        # Executive Summary
        st.subheader("📋 Executive Summary")
        st.markdown(
            f'<div class="insight-card">{insights["executive_summary"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Correlation sections (dynamic)
        if correlation_sections:
            for idx, section in enumerate(correlation_sections):
                sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
                st.subheader(sec_dict.get("title", "Correlation Analysis"))
                _render_section(sec_dict, f"corr_{idx}")
                if idx < len(correlation_sections) - 1:
                    st.markdown("---")

        st.markdown("---")

        # Retention Recommendations (dynamic first, legacy fallback)
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


    # ── Tab 5: Ask AI ────────────────────────────────────────────────

    @st.fragment
    def _ask_ai_fragment():
        st.subheader("💡 Ask a Question About Your Data")
        st.caption(
            "Ask follow-up questions in natural language. The AI has full context "
            "of the insights already generated, plus access to the raw CSV and feedback data."
        )

        # Render previous messages
        for mi, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    _render_dynamic_chart(msg["chart"], key=f"ask_hist_{mi}")

        # Chat input
        if prompt := st.chat_input("e.g., Show me attrition by age group in Engineering"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.conversation_api_history.append({"role": "user", "content": prompt})

            # Call API
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = api_client.ask_question(
                            company_name=insights["company_name"],
                            question=prompt,
                            conversation_history=st.session_state.conversation_api_history,
                        )
                        answer = result.get("answer", "Sorry, I couldn't generate a response.")
                        chart_spec = result.get("chart")

                        st.markdown(answer)
                        if chart_spec:
                            _render_dynamic_chart(chart_spec, key=f"ask_new_{len(st.session_state.chat_history)}")

                        st.session_state.chat_history.append({
                            "role": "assistant", "content": answer, "chart": chart_spec
                        })
                        st.session_state.conversation_api_history.append({
                            "role": "assistant", "content": answer
                        })

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": error_msg, "chart": None
                        })

    with tab5:
        _ask_ai_fragment()

    # ── Tab 6: Explore & Customize ───────────────────────────────────

    @st.fragment
    def _explore_fragment():
        st.subheader("✏️ Explore & Customize Analyses")
        st.caption(
            "All AI-generated charts are shown below. Use the chat to modify any "
            "chart — change colors, chart types, filter data, add comparisons, or "
            "create entirely new visualizations. Your original tabs stay untouched."
        )

        import json as _json

        # ── Show ALL sections (structured + VoE + correlation) ───────
        for idx, section in enumerate(all_sections):
            sec_dict = section if isinstance(section, dict) else section.dict() if hasattr(section, "dict") else {}
            cat = sec_dict.get("category", "")
            icon = "📉" if cat == "structured" else "💬" if cat == "voe" else "🔗"
            st.markdown(f"#### {icon} {sec_dict.get('title', f'Section {idx + 1}')}")
            _render_section(sec_dict, f"explore_{idx}")
            if idx < len(all_sections) - 1:
                st.markdown("---")

        if not all_sections:
            st.info("No analysis sections available. Run Analyze first.")

        # ── AI-customized charts ─────────────────────────────────────
        if st.session_state.explore_charts:
            st.markdown("---")
            st.markdown("### 🤖 Custom Charts")
            for ei, (title, spec) in enumerate(st.session_state.explore_charts.items()):
                st.markdown(f"**{title}**")
                _render_dynamic_chart(spec, key=f"exp_custom_{ei}")

        # ── Chat history ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Customize Any Chart")

        for mi, msg in enumerate(st.session_state.explore_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("chart"):
                    _render_dynamic_chart(msg["chart"], key=f"exp_hist_{mi}")

        # ── Single unified chat input ────────────────────────────────
        if explore_prompt := st.chat_input(
            "e.g., Color all bars green, Show attrition as a pie chart, Add a trend line...",
            key="explore_chat",
        ):
            # Build context from ALL sections
            all_chart_context = []
            for sec in all_sections:
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
                f"The user is viewing ALL dashboard charts:\n"
                f"{_json.dumps(all_chart_context, default=str)}\n"
                f"Modify an existing chart or create a new visualization based on the request. "
                f"Always include a chart_spec in your response."
            )

            with st.chat_message("user"):
                st.markdown(explore_prompt)
            st.session_state.explore_history.append({
                "role": "user", "content": explore_prompt,
            })
            st.session_state.explore_api_history.append({"role": "user", "content": full_q})

            with st.chat_message("assistant"):
                with st.spinner("Customizing..."):
                    try:
                        result = api_client.ask_question(
                            company_name=insights["company_name"],
                            question=full_q,
                            conversation_history=st.session_state.explore_api_history,
                        )
                        answer = result.get("answer", "")
                        chart_spec = result.get("chart")

                        st.markdown(answer)
                        if chart_spec:
                            _render_dynamic_chart(chart_spec, key=f"exp_new_{len(st.session_state.explore_history)}")
                            ct = chart_spec.get("title", explore_prompt[:40])
                            st.session_state.explore_charts[ct] = chart_spec

                        st.session_state.explore_history.append({
                            "role": "assistant", "content": answer, "chart": chart_spec,
                        })
                        st.session_state.explore_api_history.append({
                            "role": "assistant", "content": answer,
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.explore_history.append({
                            "role": "assistant", "content": error_msg, "chart": None,
                        })

        # ── Reset ────────────────────────────────────────────────────
        st.markdown("---")
        if st.session_state.explore_history:
            if st.button("🔄 Reset All Customizations", key="explore_reset"):
                st.session_state.explore_history = []
                st.session_state.explore_api_history = []
                st.session_state.explore_charts = {}
                st.rerun()

    with tab6:
        _explore_fragment()
