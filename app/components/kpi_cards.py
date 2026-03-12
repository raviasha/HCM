"""
KPI card rendering helpers for Streamlit dashboard.
"""

import streamlit as st


def render_kpi_cards(kpis: dict):
    """Render a row of 4 KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Headcount",
            value=f"{kpis.get('total_headcount', 0):,}",
        )

    with col2:
        rate = kpis.get("overall_attrition_rate", 0)
        st.metric(
            label="Attrition Rate",
            value=f"{rate * 100:.1f}%",
            delta=f"{'High' if rate > 0.20 else 'Moderate' if rate > 0.12 else 'Low'}",
            delta_color="inverse",  # Red for "High", green for "Low"
        )

    with col3:
        st.metric(
            label="Avg Engagement",
            value=f"{kpis.get('avg_engagement_score', 0):.1f}/10",
        )

    with col4:
        st.metric(
            label="Avg Tenure",
            value=f"{kpis.get('avg_tenure', 0):.1f} yrs",
        )
