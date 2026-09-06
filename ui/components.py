"""ui / components for ManGo or Stay."""

from config import PRIMARY_LIGHT
from config import RIPENESS_BADGE_CLASS
from config import RIPENESS_COLORS
import plotly.graph_objects as go
import streamlit as st

def badge_html(text, css_class):
    return f"<span class='badge {css_class}'>{text}</span>"

def grade_badge(grade):
    css_class = {
        "Premium": "badge-premium",
        "Grade 1": "badge-grade-1",
        "Grade 2": "badge-grade-2",
        "Reject": "badge-reject",
        "Unavailable": "badge-unavailable",
        "A": "badge-A",
        "B": "badge-B",
        "C": "badge-C",
        "D": "badge-D",
    }.get(grade, "badge-unavailable")
    return badge_html(grade or "Unavailable", css_class)

def ripeness_badge(level):
    return badge_html(level or "Unavailable", RIPENESS_BADGE_CLASS.get(level, "badge-unavailable"))

def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def page_header(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="page-kicker">{kicker}</div>
        <h1 class="page-title">{title}</h1>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )

def step_indicator(current_step: int):
    steps = ["Add photo", "Analyse", "View result"]
    html = "<div class='step-indicator'>"
    for i, label in enumerate(steps, start=1):
        if i < current_step:
            css_class = "step-done"
        elif i == current_step:
            css_class = "step-active"
        else:
            css_class = "step-todo"
        html += (
            f"<div class='step {css_class}'>"
            f"<span class='step-num'>{i}</span><span>{label}</span></div>"
        )
        if i < len(steps):
            html += "<div class='step-connector'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def confidence_gauge(confidence: float, ripeness_label: str):
    colour = RIPENESS_COLORS.get(ripeness_label, PRIMARY_LIGHT)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 10}},
                "bar": {"color": colour, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FBE9E7"},
                    {"range": [50, 80], "color": "#FFF8E1"},
                    {"range": [80, 100], "color": "#E8F5E9"},
                ],
            },
        )
    )
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=15, b=10))
    return fig

def render_confidence_gauge(confidence: float, ripeness_label: str):
    """Render a fluid gauge whose geometry and label scale together."""
    value = max(0.0, min(100.0, float(confidence)))
    colour = RIPENESS_COLORS.get(ripeness_label, PRIMARY_LIGHT)
    st.markdown(
        f"""<div class="confidence-gauge">
        <svg viewBox="0 0 400 225" role="img"
             aria-label="Prediction confidence: {value:.1f}%"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M 35 195 A 165 165 0 0 1 365 195" fill="none"
                  stroke="#E8F5E9" stroke-width="30" />
            <path d="M 35 195 A 165 165 0 0 1 365 195" fill="none"
                  stroke="#FFF8E1" stroke-width="30" pathLength="100"
                  stroke-dasharray="80 100" />
            <path d="M 35 195 A 165 165 0 0 1 365 195" fill="none"
                  stroke="#FBE9E7" stroke-width="30" pathLength="100"
                  stroke-dasharray="50 100" />
            <path d="M 35 195 A 165 165 0 0 1 365 195" fill="none"
                  stroke="{colour}" stroke-width="13" pathLength="100"
                  stroke-dasharray="{value} 100" />
            <g fill="#737B87" font-family="Arial, sans-serif" text-anchor="middle">
                <text x="200" y="183" font-size="30">{value:.1f}%</text>
                <text x="16" y="213" font-size="11">0</text>
                <text x="200" y="12" font-size="11">50</text>
                <text x="384" y="213" font-size="11">100</text>
            </g>
        </svg></div>""",
        unsafe_allow_html=True,
    )


def empty_state(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-mark"></div>
            <div style="font-weight:680; color:#37413C; margin-bottom:5px;">{title}</div>
            <div style="font-size:12px; line-height:1.55;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

