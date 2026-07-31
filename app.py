"""
FruitVision AI — Fruit Ripeness & Surface Quality Assessment System
Run with:  streamlit run app.py
Requires:  pip install streamlit pandas numpy plotly pillow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# THEME / CSS
# ------------------------------------------------------------------
PRIMARY = "#2E7D32"
PRIMARY_LIGHT = "#4CAF50"
SECONDARY = "#FF9800"
BG_LIGHT = "#F5F5F5"
RED = "#F44336"
YELLOW = "#FFC107"
GREEN = "#4CAF50"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}

    .main {{
        background-color: {BG_LIGHT};
    }}

    .metric-card {{
        background: white;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eee;
    }}

    .metric-title {{
        font-size: 13px;
        color: #757575;
        font-weight: 500;
        margin-bottom: 4px;
    }}

    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: #1A1A2E;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: white;
    }}
    .badge-ripe {{ background-color: {GREEN}; }}
    .badge-semi {{ background-color: {YELLOW}; color:#1A1A2E; }}
    .badge-unripe {{ background-color: {SECONDARY}; }}
    .badge-overripe {{ background-color: {RED}; }}
    .badge-A {{ background-color: {GREEN}; }}
    .badge-B {{ background-color: #8BC34A; }}
    .badge-C {{ background-color: {YELLOW}; color:#1A1A2E; }}
    .badge-D {{ background-color: {RED}; }}

    .app-header {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 6px 0 18px 0;
    }}

    .logo-text {{
        font-size: 22px;
        font-weight: 700;
        color: {PRIMARY};
    }}

    .section-title {{
        font-size: 20px;
        font-weight: 700;
        color: #1A1A2E;
        margin: 10px 0 14px 0;
    }}

    div.stButton > button {{
        border-radius: 10px;
        font-weight: 600;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# MOCK DATA
# ------------------------------------------------------------------
FRUITS = ["Apple", "Banana", "Mango", "Orange", "Tomato"]
RIPENESS_LEVELS = ["Unripe", "Semi-Ripe", "Ripe", "Overripe"]
GRADES = ["A", "B", "C", "D"]
DEFECT_TYPES = ["Bruise", "Scratch", "Rot", "Pest Damage"]
SEVERITY = ["Low", "Medium", "High"]

RIPENESS_BADGE_CLASS = {
    "Unripe": "badge-unripe",
    "Semi-Ripe": "badge-semi",
    "Ripe": "badge-ripe",
    "Overripe": "badge-overripe",
}


@st.cache_data
def generate_history(n=150):
    rows = []
    start = datetime.now() - timedelta(days=60)
    for i in range(n):
        dt = start + timedelta(hours=random.randint(0, 60 * 24))
        rows.append(
            {
                "ID": f"FR-2026-{dt.strftime('%m%d')}-{i:03d}",
                "Fruit Type": random.choice(FRUITS),
                "Ripeness": random.choice(RIPENESS_LEVELS),
                "Grade": random.choices(GRADES, weights=[40, 30, 20, 10])[0],
                "Defect %": round(random.uniform(0, 15), 1),
                "Date": dt,
            }
        )
    df = pd.DataFrame(rows).sort_values("Date", ascending=False).reset_index(drop=True)
    return df


history_df = generate_history()

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div class='logo-text'>🍎 FruitVision AI</div>",
        unsafe_allow_html=True,
    )
    st.caption("AI-Powered Fruit Quality Inspection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Dashboard", "New Assessment", "Assessment Details", "History / Reports", "Settings"],
        label_visibility="collapsed",
    )

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def badge_html(text, css_class):
    return f"<span class='badge {css_class}'>{text}</span>"


def grade_badge(grade):
    return badge_html(grade, f"badge-{grade}")


def ripeness_badge(level):
    return badge_html(level, RIPENESS_BADGE_CLASS.get(level, "badge-ripe"))


def metric_card(title, value, delta=None, delta_color="normal"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------
if page == "Dashboard":
    st.markdown("<div class='section-title'>Dashboard Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Fruits Analyzed", f"{len(history_df):,}")
        st.caption("⬆ 8.2% vs last week")
    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Ripeness Distribution</div>", unsafe_allow_html=True)
        pie_df = history_df["Ripeness"].value_counts().reset_index()
        pie_df.columns = ["Ripeness", "Count"]
        fig = px.pie(pie_df, names="Ripeness", values="Count", hole=0.55,
                     color_discrete_sequence=[SECONDARY, YELLOW, GREEN, RED])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        grade_a_pct = round((history_df["Grade"] == "A").mean() * 100, 1)
        metric_card("Quality Grade A %", f"{grade_a_pct}%")
        st.progress(grade_a_pct / 100)
    with c4:
        defect_rate = round((history_df["Defect %"] > 5).mean() * 100, 1)
        metric_card("Defect Detection Rate", f"{defect_rate}%")
        st.caption("⬇ 1.4% vs last week")

    st.write("")
    left, right = st.columns([6, 4])

    with left:
        st.markdown("<div class='section-title'>Recent Assessments</div>", unsafe_allow_html=True)
        recent = history_df.head(8).copy()
        recent["Ripeness"] = recent["Ripeness"].apply(ripeness_badge)
        recent["Grade"] = recent["Grade"].apply(grade_badge)
        recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d %H:%M")
        st.write(
            recent[["ID", "Fruit Type", "Ripeness", "Grade", "Defect %", "Date"]].to_html(
                escape=False, index=False
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div class='section-title'>Quick Stats & Alerts</div>", unsafe_allow_html=True)
        st.info("📷 Live camera feed preview (placeholder)")
        st.image(
            "https://placehold.co/400x220/2E7D32/FFFFFF?text=Live+Camera+Feed",
            use_container_width=True,
        )
        st.success(f"Today's summary: {random.randint(40,90)} fruits analyzed")
        st.warning("⚠ 3 fruits flagged with high defect severity")
        if st.button("➕ Start New Assessment", type="primary", use_container_width=True):
            st.toast("Navigate to 'New Assessment' from the sidebar", icon="✅")

# ------------------------------------------------------------------
# PAGE 2: NEW ASSESSMENT
# ------------------------------------------------------------------
elif page == "New Assessment":
    col_back, col_title = st.columns([1, 9])
    with col_back:
        st.button("← Back")
    with col_title:
        st.markdown("<div class='section-title'>New Fruit Assessment</div>", unsafe_allow_html=True)

    top1, top2 = st.columns(2)
    with top1:
        fruit_type = st.selectbox("Select Fruit Type", FRUITS)
    with top2:
        batch_id = st.text_input("Batch ID", value=f"BATCH-{datetime.now().strftime('%Y%m%d')}-01")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.subheader("📷 Image Upload / Scan")
        uploaded = st.file_uploader("Drag & drop or browse (JPG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"])
        use_camera = st.camera_input("Or use your webcam")
        image_ready = uploaded is not None or use_camera is not None
        if uploaded:
            st.image(uploaded, caption="Preview", use_container_width=True)
        elif use_camera:
            st.image(use_camera, caption="Captured", use_container_width=True)

        analyze_clicked = st.button("🔍 Analyze Fruit", type="primary", disabled=not image_ready,
                                     use_container_width=True)

    with right:
        st.subheader("Real-Time Results")
        if not image_ready:
            st.markdown("_Upload or capture an image to begin analysis._")
        elif analyze_clicked:
            progress = st.progress(0, text="Analyzing...")
            for pct in range(0, 101, 20):
                time.sleep(0.15)
                progress.progress(pct, text="Analyzing..." if pct < 100 else "Done")

            confidence = round(random.uniform(85, 99), 1)
            ripeness = random.choice(RIPENESS_LEVELS)
            grade = random.choices(GRADES, weights=[40, 30, 20, 10])[0]
            defect_pct = round(random.uniform(0, 12), 1)
            severity = "Low" if defect_pct < 4 else ("Medium" if defect_pct < 9 else "High")

            with st.expander("Module 1: Image Preprocessing", expanded=True):
                st.checkbox("Show preprocessed image", value=True, key="pp_toggle")
                st.caption("Resized ✓ · Background removed ✓ · Contrast enhanced ✓")

            with st.expander("Module 2: Ripeness Classification", expanded=True):
                st.markdown(ripeness_badge(ripeness), unsafe_allow_html=True)
                st.metric("Confidence Score", f"{confidence}%")
                st.progress(confidence / 100)
                mc1, mc2 = st.columns(2)
                mc1.metric("Color Value", round(random.uniform(0.4, 0.95), 2))
                mc2.metric("Texture Score", round(random.uniform(0.5, 0.98), 2))

            with st.expander("Module 3: Surface Quality Grading", expanded=True):
                st.markdown(grade_badge(grade), unsafe_allow_html=True)
                stars = {"A": "★★★★★", "B": "★★★★☆", "C": "★★★☆☆", "D": "★★☆☆☆"}[grade]
                st.markdown(f"**{stars}**")
                mc1, mc2 = st.columns(2)
                mc1.metric("Surface Smoothness", round(random.uniform(0.5, 0.99), 2))
                mc2.metric("Shape Score", round(random.uniform(0.5, 0.99), 2))

            with st.expander("Module 4: Blemish & Damage Quantification", expanded=True):
                st.metric("Defect Percentage", f"{defect_pct}%")
                st.write("Defect types identified: " + ", ".join(random.sample(DEFECT_TYPES, k=2)))
                sev_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[severity]
                st.write(f"Severity: {sev_color} **{severity}**")

            st.write("")
            b1, b2, b3 = st.columns(3)
            b1.button("📄 Generate Report", use_container_width=True)
            b2.button("💾 Save to History", use_container_width=True)
            b3.selectbox("Export as", ["PDF", "CSV", "JSON"], label_visibility="collapsed")
        else:
            st.markdown("_Click **Analyze Fruit** to run the assessment._")

# ------------------------------------------------------------------
# PAGE 3: ASSESSMENT DETAILS
# ------------------------------------------------------------------
elif page == "Assessment Details":
    selected_id = st.selectbox("Select Assessment", history_df["ID"].head(30))
    row = history_df[history_df["ID"] == selected_id].iloc[0]

    st.markdown("<div class='section-title'>Assessment Details</div>", unsafe_allow_html=True)
    top1, top2, top3 = st.columns([2, 2, 2])
    top1.markdown(f"**Assessment ID:** {row['ID']}")
    top2.markdown(f"**Date:** {row['Date'].strftime('%Y-%m-%d %H:%M')}")
    with top3:
        st.button("⬇ Export PDF")

    left, right = st.columns([6, 4])

    with left:
        tabs = st.tabs(["Original", "Processed", "Defect Overlay"])
        with tabs[0]:
            st.image("https://placehold.co/500x400/4CAF50/FFFFFF?text=Original+Image", use_container_width=True)
        with tabs[1]:
            st.image("https://placehold.co/500x400/2E7D32/FFFFFF?text=Processed+Image", use_container_width=True)
        with tabs[2]:
            st.image("https://placehold.co/500x400/F44336/FFFFFF?text=Defect+Heatmap", use_container_width=True)

    with right:
        st.markdown("#### Summary")
        st.write(f"**Fruit Type:** {row['Fruit Type']}")
        st.markdown(f"**Ripeness Level:** {ripeness_badge(row['Ripeness'])}", unsafe_allow_html=True)
        st.markdown(f"**Quality Grade:** {grade_badge(row['Grade'])}", unsafe_allow_html=True)
        overall_score = round(100 - row["Defect %"] * 2 + random.uniform(-3, 3), 1)
        st.metric("Overall Score", f"{overall_score}/100")
        recommendation = "✅ Ready for Packaging" if row["Defect %"] < 6 else "⚠ Needs Sorting"
        st.write(f"**Recommendation:** {recommendation}")

        st.markdown("#### Detailed Metrics")
        with st.expander("Module 1: Preprocessing"):
            st.write("Resolution: 1024x1024 · Background removed ✓ · Enhanced ✓")
        with st.expander("Module 2: Ripeness"):
            st.write(f"Color value: {round(random.uniform(0.4,0.95),2)} · Texture: {round(random.uniform(0.5,0.98),2)}")
        with st.expander("Module 3: Quality"):
            st.write(f"Shape score: {round(random.uniform(0.5,0.99),2)} · Surface defects: {row['Defect %']}%")
        with st.expander("Module 4: Blemish"):
            st.write(f"Area affected: {row['Defect %']}% · Types: {', '.join(random.sample(DEFECT_TYPES, k=2))}")

# ------------------------------------------------------------------
# PAGE 4: HISTORY / REPORTS
# ------------------------------------------------------------------
elif page == "History / Reports":
    st.markdown("<div class='section-title'>Assessment History</div>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    search = s1.text_input("🔍 Search by ID / Type")
    fruit_filter = s2.multiselect("Filter Fruit", FRUITS, default=FRUITS)
    date_range = s3.date_input("Date Range", [])
    s4.button("⬇ Export All", use_container_width=True)

    filtered = history_df[history_df["Fruit Type"].isin(fruit_filter)]
    if search:
        filtered = filtered[
            filtered["ID"].str.contains(search, case=False)
            | filtered["Fruit Type"].str.contains(search, case=False)
        ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Assessments", len(filtered))
    m2.metric("Avg Quality Score", f"{round(100 - filtered['Defect %'].mean()*2,1) if len(filtered) else 0}%")
    m3.metric("Defect Rate", f"{round((filtered['Defect %']>5).mean()*100,1) if len(filtered) else 0}%")
    m4.metric("Most Common Defect", "Bruising")

    st.write("")
    disp = filtered.copy()
    disp["Ripeness"] = disp["Ripeness"].apply(ripeness_badge)
    disp["Grade"] = disp["Grade"].apply(grade_badge)
    disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d %H:%M")
    st.write(disp.head(20).to_html(escape=False, index=False), unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='section-title'>Analytics</div>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    with g1:
        trend = history_df.copy()
        trend["Day"] = trend["Date"].dt.date
        trend_agg = trend.groupby("Day")["Defect %"].mean().reset_index()
        fig = px.line(trend_agg, x="Day", y="Defect %", title="Quality Trend (Avg Defect %) Over Time")
        fig.update_traces(line_color=PRIMARY_LIGHT)
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        bar_data = history_df.groupby(["Fruit Type", "Ripeness"]).size().reset_index(name="Count")
        fig2 = px.bar(bar_data, x="Fruit Type", y="Count", color="Ripeness", title="Ripeness Distribution by Fruit Type",
                      color_discrete_sequence=[SECONDARY, YELLOW, GREEN, RED])
        st.plotly_chart(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        defect_dist = pd.Series(random.choices(DEFECT_TYPES, k=200)).value_counts().reset_index()
        defect_dist.columns = ["Defect Type", "Count"]
        fig3 = px.pie(defect_dist, names="Defect Type", values="Count", title="Defect Type Distribution")
        st.plotly_chart(fig3, use_container_width=True)

    with g4:
        heat = history_df.copy()
        heat["Month"] = heat["Date"].dt.strftime("%b")
        heat_agg = heat.groupby(["Month", "Fruit Type"])["Defect %"].mean().reset_index()
        heat_pivot = heat_agg.pivot(index="Fruit Type", columns="Month", values="Defect %")
        fig4 = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r", title="Seasonal Quality Variation")
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------------
# PAGE 5: SETTINGS
# ------------------------------------------------------------------
elif page == "Settings":
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)

    tabs = st.tabs(["General", "Camera", "AI Model", "Quality Thresholds"])

    with tabs[0]:
        st.text_input("System Name", value="FruitVision AI")
        st.selectbox("Default Fruit Type", FRUITS)
        st.selectbox("Image Resolution", ["720p", "1080p", "4K"])
        st.selectbox("Language", ["English", "Malay", "Chinese"])
        st.toggle("Dark Mode")

    with tabs[1]:
        st.selectbox("Camera Selection", ["Webcam 0", "Industrial Cam A", "Industrial Cam B"])
        st.selectbox("Resolution", ["720p", "1080p", "4K"])
        st.toggle("Auto-Capture Mode")
        st.slider("Exposure", 0, 100, 50)
        st.slider("Contrast", 0, 100, 50)

    with tabs[2]:
        st.selectbox("Model Selection", ["MobileNetV2", "ResNet50"])
        st.slider("Confidence Threshold", 0, 100, 80)
        st.number_input("Batch Processing Size", min_value=1, max_value=256, value=32)
        st.radio("Processing Unit", ["GPU", "CPU"], horizontal=True)

    with tabs[3]:
        st.write("**Ripeness Thresholds**")
        st.slider("Unripe → Semi-Ripe", 0, 100, 25)
        st.slider("Semi-Ripe → Ripe", 0, 100, 55)
        st.slider("Ripe → Overripe", 0, 100, 85)
        st.write("**Quality Grading Thresholds**")
        st.slider("Grade A minimum score", 0, 100, 90)
        st.slider("Grade B minimum score", 0, 100, 75)
        st.slider("Grade C minimum score", 0, 100, 60)
        st.write("**Defect Thresholds**")
        st.slider("Defect Warning Threshold (%)", 0, 100, 8)
        st.slider("Auto-Reject Threshold (%)", 0, 100, 20)

    st.write("")
    st.button("💾 Save Settings", type="primary")