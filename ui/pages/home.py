"""ui / pages / home for ManGo or Stay."""

from config import RIPENESS_COLORS
from ui.components import empty_state
from ui.components import metric_card
from ui.components import page_header
from ui.navigation import go_to
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def render(history_df):
    page_header(
        "Overview",
        "Harumanis mango assessment at a glance",
        "The figures on this page are calculated only from assessments that have actually been saved by the system.",
    )

    total_assessments = len(history_df)
    average_confidence = history_df["Confidence"].mean() if total_assessments else np.nan
    average_defect = history_df["Defect %"].mean() if total_assessments else np.nan
    high_severity_count = int((history_df["Severity"] == "High").sum()) if total_assessments else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Saved assessments", f"{total_assessments:,}")
    with c2:
        metric_card("Average confidence", f"{average_confidence:.1f}%" if pd.notna(average_confidence) else "No data")
    with c3:
        metric_card("Average defect coverage", f"{average_defect:.2f}%" if pd.notna(average_defect) else "No data")
    with c4:
        metric_card("High-severity assessments", f"{high_severity_count:,}")

    st.write("")
    if history_df.empty:
        empty_state(
            "No saved assessments yet",
            "Complete an assessment to get started. Successful mango assessments are saved automatically and will appear here.",
        )
    else:
        st.markdown("<div class='section-title'>Ripeness distribution</div>", unsafe_allow_html=True)
        distribution = history_df["Ripeness"].fillna("Unavailable").value_counts().reset_index()
        distribution.columns = ["Ripeness", "Count"]
        distribution_fig = px.pie(
            distribution,
            names="Ripeness",
            values="Count",
            hole=0.55,
            color="Ripeness",
            color_discrete_map=RIPENESS_COLORS,
        )
        distribution_fig.update_traces(textposition="inside", textinfo="percent+label")
        distribution_fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(distribution_fig, use_container_width=True, config={"displayModeBar": False})

        left, right = st.columns([6, 4])
        with left:
            st.markdown("<div class='section-title'>Recent assessments</div>", unsafe_allow_html=True)
            recent = history_df.head(8).copy()
            recent["Confidence"] = recent["Confidence"].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "Unavailable")
            recent["Defect %"] = recent["Defect %"].map(lambda value: f"{value:.2f}%" if pd.notna(value) else "Unavailable")
            recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                recent[["ID", "Batch ID", "Ripeness", "Confidence", "Grade", "Defect %", "Date"]],
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.markdown("<div class='section-title'>Current records</div>", unsafe_allow_html=True)
            latest = history_df.iloc[0]
            st.markdown(f"**Latest assessment**  \\n{latest['ID']}")
            st.markdown(f"**Latest ripeness**  \\n{latest['Ripeness'] or 'Unavailable'}")
            st.markdown(f"**Latest quality grade**  \\n{latest['Grade'] or 'Unavailable'}")
            if st.button("Open latest assessment", use_container_width=True):
                st.session_state.selected_assessment_id = latest["ID"]
                go_to("Assessment Details")

    st.write("")
    st.markdown("<div class='section-title'>Before you take a photo</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="guide-card">
            <div class="guide-number">01</div>
            <div class="guide-title">Keep the whole mango visible</div>
            <div class="guide-copy">Place one Harumanis mango fully inside the frame and keep the image in focus.</div>
        </div>
        <div class="guide-card">
            <div class="guide-number">02</div>
            <div class="guide-title">Use even lighting</div>
            <div class="guide-copy">Avoid strong shadows and glare that can hide colour or surface detail.</div>
        </div>
        <div class="guide-card">
            <div class="guide-number">03</div>
            <div class="guide-title">Choose a simple background</div>
            <div class="guide-copy">A clear background helps the system separate the mango from its surroundings.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start an assessment", type="primary", use_container_width=True, key="home_start_assessment"):
        go_to("Assess Mango")

