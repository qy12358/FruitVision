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

