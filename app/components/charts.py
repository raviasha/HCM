"""
Plotly chart builders for the HCM AI Insights dashboard.
All functions return plotly Figure objects ready for st.plotly_chart().
"""

import plotly.express as px
import plotly.graph_objects as go


# ── Color palette ────────────────────────────────────────────────────

RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
SENTIMENT_COLORS = {"positive": "#10B981", "neutral": "#6B7280", "negative": "#EF4444"}
BRAND_BLUE = "#3B82F6"
BRAND_PURPLE = "#8B5CF6"


def _risk_level(rate: float) -> str:
    if rate >= 0.30:
        return "High"
    elif rate >= 0.18:
        return "Medium"
    return "Low"


# ── Attrition Charts ────────────────────────────────────────────────

def attrition_by_department_bar(attrition_data: list[dict]) -> go.Figure:
    """Horizontal bar chart of attrition % per department, color-coded by risk."""
    sorted_data = sorted(attrition_data, key=lambda x: x["attrition_rate"], reverse=True)

    departments = [d["department"] for d in sorted_data]
    rates = [d["attrition_rate"] * 100 for d in sorted_data]
    colors = [RISK_COLORS[_risk_level(d["attrition_rate"])] for d in sorted_data]
    headcounts = [d["headcount"] for d in sorted_data]

    fig = go.Figure(go.Bar(
        x=rates,
        y=departments,
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1f}% ({h})" for r, h in zip(rates, headcounts)],
        textposition="auto",
    ))
    fig.update_layout(
        title="Attrition Rate by Department",
        xaxis_title="Attrition Rate (%)",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def risk_factors_bar(risk_data: list[dict]) -> go.Figure:
    """Horizontal bar chart of top risk factors by correlation with attrition."""
    top = risk_data[:10]
    factors = [d["factor"] for d in reversed(top)]
    corrs = [d["correlation"] for d in reversed(top)]
    colors = [RISK_COLORS["High"] if c > 0 else BRAND_BLUE for c in corrs]

    fig = go.Figure(go.Bar(
        x=corrs,
        y=factors,
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.3f}" for c in corrs],
        textposition="auto",
    ))
    fig.update_layout(
        title="Top Attrition Risk Factors (Correlation)",
        xaxis_title="Correlation with Attrition",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def overtime_comparison_bar(overtime_data: dict) -> go.Figure:
    """Bar chart comparing attrition rate for overtime vs no-overtime."""
    categories = ["Overtime", "No Overtime"]
    rates = [
        overtime_data["overtime_attrition_rate"] * 100,
        overtime_data["no_overtime_attrition_rate"] * 100,
    ]
    counts = [
        overtime_data["overtime_headcount"],
        overtime_data["no_overtime_headcount"],
    ]

    fig = go.Figure(go.Bar(
        x=categories,
        y=rates,
        marker_color=[RISK_COLORS["High"], BRAND_BLUE],
        text=[f"{r:.1f}% (n={c})" for r, c in zip(rates, counts)],
        textposition="auto",
    ))
    fig.update_layout(
        title="Attrition: Overtime vs No Overtime",
        yaxis_title="Attrition Rate (%)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Sentiment Charts ─────────────────────────────────────────────────

def sentiment_distribution_pie(sentiment_data: list[dict]) -> go.Figure:
    """Pie chart of overall positive/neutral/negative feedback distribution."""
    totals = {"positive": 0, "neutral": 0, "negative": 0}
    for dept in sentiment_data:
        totals["positive"] += dept.get("positive", 0)
        totals["neutral"] += dept.get("neutral", 0)
        totals["negative"] += dept.get("negative", 0)

    labels = list(totals.keys())
    values = list(totals.values())
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Overall Sentiment Distribution",
        height=350,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def sentiment_by_department_bar(sentiment_data: list[dict]) -> go.Figure:
    """Grouped bar chart of sentiment breakdown per department."""
    departments = [d["department"] for d in sentiment_data]

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        fig.add_trace(go.Bar(
            name=sentiment.capitalize(),
            x=departments,
            y=[d.get(sentiment, 0) for d in sentiment_data],
            marker_color=color,
        ))

    fig.update_layout(
        title="Sentiment Breakdown by Department",
        barmode="group",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def theme_frequency_bar(themes: list[dict]) -> go.Figure:
    """Bar chart of top themes from employee feedback."""
    sorted_themes = sorted(themes, key=lambda x: x.get("count", 0), reverse=True)
    names = [t["theme"] for t in sorted_themes]
    counts = [t["count"] for t in sorted_themes]

    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color=BRAND_PURPLE,
        text=counts,
        textposition="auto",
    ))
    fig.update_layout(
        title="Top Feedback Themes",
        xaxis_title="Frequency",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Correlation Charts ───────────────────────────────────────────────

def sentiment_vs_attrition_scatter(correlations: list[dict]) -> go.Figure:
    """
    Scatter plot: X = avg sentiment score, Y = attrition rate.
    Each point is a department, bubble size = relative.
    """
    departments = [c["department"] for c in correlations]
    sentiment_scores = [c.get("avg_sentiment_score", 0) for c in correlations]
    attrition_rates = [c.get("attrition_rate", 0) * 100 for c in correlations]

    fig = go.Figure(go.Scatter(
        x=sentiment_scores,
        y=attrition_rates,
        mode="markers+text",
        text=departments,
        textposition="top center",
        marker=dict(
            size=20,
            color=attrition_rates,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Attrition %"),
        ),
    ))
    fig.update_layout(
        title="Sentiment vs Attrition by Department",
        xaxis_title="Avg Sentiment Score (-1 to +1)",
        yaxis_title="Attrition Rate (%)",
        height=450,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ── Department Comparison Table ──────────────────────────────────────

def department_stats_table(dept_stats: list[dict]) -> go.Figure:
    """Interactive table of department-level statistics."""
    sorted_stats = sorted(dept_stats, key=lambda x: x.get("attrition_rate", 0), reverse=True)

    header_values = [
        "Department", "Headcount", "Attrition Rate",
        "Avg Tenure", "Avg Satisfaction", "Avg Engagement", "Overtime %"
    ]
    cell_values = [
        [d["department"] for d in sorted_stats],
        [d["headcount"] for d in sorted_stats],
        [f"{d['attrition_rate'] * 100:.1f}%" for d in sorted_stats],
        [f"{d.get('avg_tenure', 0):.1f}" for d in sorted_stats],
        [f"{d.get('avg_satisfaction', 0):.1f}" for d in sorted_stats],
        [f"{d.get('avg_engagement', 0):.1f}" for d in sorted_stats],
        [f"{d.get('overtime_pct', 0) * 100:.1f}%" for d in sorted_stats],
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=header_values,
            fill_color=BRAND_BLUE,
            font_color="white",
            align="left",
        ),
        cells=dict(
            values=cell_values,
            align="left",
        ),
    ))
    fig.update_layout(
        title="Department Comparison",
        height=350,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig
