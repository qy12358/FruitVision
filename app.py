"""
FruitVision AI — Fruit Ripeness & Surface Quality Assessment System
Run with:  streamlit run app.py
Requires:  pip install streamlit pandas numpy plotly pillow opencv-python tensorflow scikit-learn seaborn matplotlib
"""

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import time
from pathlib import Path

from modules.preprocessing import ImagePreprocessor
from modules.mango_identifier import MangoIdentifier
from modules.ripeness_classifier import HybridRipenessClassifier
from modules.blemish_detector import BlemishDetector

preprocessor = ImagePreprocessor()


def model_signature(path: str) -> int:
    """Invalidate Streamlit's model cache after a retraining run."""
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_resource
def load_mango_identifier(signature: int = 0) -> MangoIdentifier:
    """Load the optional binary mango gate once per Streamlit process."""
    return MangoIdentifier(
        model_path="models/mango_identifier.keras",
    )


blemish_detector = BlemishDetector(
    min_blemish_area=8,
    boundary_erosion=15
)

@st.cache_resource
def load_ripeness_classifier(signature: int = 0) -> HybridRipenessClassifier:
    """
    Load the hybrid (EfficientNetB0 + 63-feature colour/statistical
    branch) ripeness classifier once and cache it across Streamlit
    reruns/user sessions, so the model isn't reloaded from disk on every
    button click.
    """
    return HybridRipenessClassifier(
        model_path="models/mango_ripeness.keras",
        class_indices_path="models/class_indices.json",
    )


def format_class_label(raw_name: str) -> str:
    """
    Cosmetic only: turns a raw dataset folder / model class name like
    'semi_ripe' into a display-friendly 'Semi-Ripe'. Never used for any
    classification logic — purely for badges/labels in the UI.
    """
    return raw_name.replace("_", "-").replace(" ", "-").title()


def split_histogram_dict(colour_histogram: dict):
    """
    modules.ripeness_classifier returns the 48-bin colour histogram as a
    flat dict {"hist_h_0": ..., "hist_h_1": ..., ..., "hist_v_15": ...}.
    Split it back into three 16-value lists (one per HSV channel) for
    charting.
    """
    h_vals = [colour_histogram[f"hist_h_{i}"] for i in range(16)]
    s_vals = [colour_histogram[f"hist_s_{i}"] for i in range(16)]
    v_vals = [colour_histogram[f"hist_v_{i}"] for i in range(16)]
    return h_vals, s_vals, v_vals


# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🥭",  # Changed to Mango icon
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
    .badge-rotten {{ background-color: {RED}; }}
    .badge-A {{ background-color: {GREEN}; }}
    .badge-B {{ background-color: #8BC34A; }}
    .badge-C {{ background-color: {YELLOW}; color:#1A1A2E; }}
    .badge-D {{ background-color: {RED}; }}

    .chip {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        background-color: #FFF3E0;
        color: #E65100;
        margin-left: 8px;
        vertical-align: middle;
    }}

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

    /* ---- Step indicator ---- */
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 4px 0 22px 0;
        flex-wrap: wrap;
    }}
    .step {{
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 6px 14px 6px 8px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }}
    .step-num {{
        width: 20px; height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
    }}
    .step-todo {{ background:#EDEDED; color:#9E9E9E; }}
    .step-todo .step-num {{ background:#DADADA; color:#9E9E9E; }}
    .step-active {{ background:#E8F5E9; color:{PRIMARY}; }}
    .step-active .step-num {{ background:{PRIMARY}; color:white; }}
    .step-done {{ background:#E8F5E9; color:{PRIMARY}; }}
    .step-done .step-num {{ background:{PRIMARY_LIGHT}; color:white; }}
    .step-connector {{ width: 20px; height: 2px; background:#DADADA; }}

    /* ---- Result hero card ---- */
    .hero-card {{
        background: white;
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        border-left: 6px solid var(--hero-color, {PRIMARY});
        margin-bottom: 18px;
    }}
    .hero-title {{
        font-size: 13px;
        color: #757575;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }}
    .hero-advice {{
        font-size: 15px;
        color: #444;
        margin-top: 10px;
    }}

    /* ---- Empty state ---- */
    .empty-state {{
        text-align: center;
        padding: 46px 20px;
        color: #9E9E9E;
    }}
    .empty-state-icon {{ font-size: 40px; margin-bottom: 10px; }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# NAVIGATION / SESSION STATE
# ------------------------------------------------------------------
PAGES = ["Dashboard", "New Assessment", "Assessment Details", "History / Reports", "Settings"]
PAGE_ICONS = {
    "Dashboard": "📊",
    "New Assessment": "➕",
    "Assessment Details": "🔍",
    "History / Reports": "🕒",
    "Settings": "⚙️",
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "saved_assessments" not in st.session_state:
    st.session_state.saved_assessments = []

# Apply any pending navigation request BEFORE the sidebar widget is
# created below — Streamlit forbids setting a widget-bound session_state
# key after that widget has already been instantiated in the same run.
if "_pending_nav" in st.session_state:
    st.session_state.page = st.session_state.pop("_pending_nav")


def go_to(page_name: str):
    st.session_state["_pending_nav"] = page_name
    st.rerun()


RIPENESS_ADVICE = {
    "Ripe": ("✅", "Ready to eat or sell now."),
    "Semi-Ripe": ("🕓", "A few more days needed before it's fully ripe."),
    "Unripe": ("🌱", "Needs more time to ripen — not ready yet."),
    "Rotten": ("⚠️", "Not suitable for sale or consumption."),
}

# ------------------------------------------------------------------
# MOCK DATA
# ------------------------------------------------------------------
FRUITS = ["Mango"]

RIPENESS_LEVELS = ["Ripe", "Rotten", "Semi-Ripe", "Unripe"]
GRADES = ["A", "B", "C", "D"]
DEFECT_TYPES = ["Bruise", "Scratch", "Rot", "Pest Damage"]
SEVERITY = ["Low", "Medium", "High"]

RIPENESS_BADGE_CLASS = {
    "Unripe": "badge-unripe",
    "Semi-Ripe": "badge-semi",
    "Ripe": "badge-ripe",
    "Rotten": "badge-rotten",
}

RIPENESS_COLORS = {
    "Unripe": SECONDARY,
    "Semi-Ripe": YELLOW,
    "Ripe": GREEN,
    "Rotten": RED,
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
        "<div class='logo-text'>🥭 FruitVision AI</div>",  # Changed to Mango icon
        unsafe_allow_html=True,
    )
    st.caption("Snap a photo of a mango — get instant ripeness, quality and defect results.")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        PAGES,
        format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
        label_visibility="collapsed",
        key="page",
    )
    st.markdown("---")
    if page != "New Assessment":
        if st.button("➕ New Assessment", use_container_width=True, type="primary"):
            go_to("New Assessment")
    st.caption(f"📌 {len(st.session_state.saved_assessments)} assessment(s) saved this session")

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


def step_indicator(current_step: int):
    """1 = choosing a photo, 2 = ready to analyze, 3 = showing results."""
    steps = ["Upload photo", "Analyze", "See results"]
    html = "<div class='step-indicator'>"
    for i, label in enumerate(steps, start=1):
        if i < current_step:
            cls = "step-done"
        elif i == current_step:
            cls = "step-active"
        else:
            cls = "step-todo"
        html += (
            f"<div class='step {cls}'>"
            f"<span class='step-num'>{i}</span><span>{label}</span></div>"
        )
        if i < len(steps):
            html += "<div class='step-connector'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def confidence_gauge(confidence: float, ripeness_label: str):
    color = RIPENESS_COLORS.get(ripeness_label, PRIMARY_LIGHT)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 10}},
                "bar": {"color": color, "thickness": 0.28},
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


def empty_state(icon: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <div style="font-weight:600; color:#616161; margin-bottom:4px;">{title}</div>
            <div style="font-size:13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def combine_object_blemish_results(image, objects, analyses):
    """Place per-mango damage results back into the native scene image."""
    height, width = image.shape[:2]
    damage_mask = np.zeros((height, width), dtype=np.uint8)
    safe_mask = np.zeros((height, width), dtype=np.uint8)
    blackhat = np.zeros((height, width), dtype=np.uint8)
    candidate_mask = np.zeros((height, width), dtype=np.uint8)
    component_masks = {
        key: np.zeros((height, width), dtype=np.uint8)
        for key in (
            "dark_spots", "brown_lesions", "severe_dark", "wet_damage",
            "scratch_mask", "white_surface",
        )
    }
    defect_types = []
    component_count = 0

    for mango_object, analysis in zip(objects, analyses):
        x, y, object_width, object_height = mango_object["bbox"]
        y1 = min(height, y + object_height)
        x1 = min(width, x + object_width)
        crop_height = max(0, y1 - y)
        crop_width = max(0, x1 - x)
        if crop_height == 0 or crop_width == 0:
            continue
        crop_slice = np.s_[y:y1, x:x1]
        damage_mask[crop_slice] = np.maximum(
            damage_mask[crop_slice], analysis["damage_mask"][:crop_height, :crop_width]
        )
        safe_mask[crop_slice] = np.maximum(
            safe_mask[crop_slice], analysis["safe_mango_mask"][:crop_height, :crop_width]
        )
        blackhat[crop_slice] = np.maximum(
            blackhat[crop_slice], analysis["blackhat"][:crop_height, :crop_width]
        )
        candidate_mask[crop_slice] = np.maximum(
            candidate_mask[crop_slice], analysis["candidate_mask"][:crop_height, :crop_width]
        )
        for key, canvas in component_masks.items():
            canvas[crop_slice] = np.maximum(
                canvas[crop_slice], analysis[key][:crop_height, :crop_width]
            )
        component_count += analysis["component_count"]
        defect_types.extend(
            item for item in analysis["defect_types"] if item != "None"
        )

    mango_area = int(cv2.countNonZero(safe_mask))
    damage_area = int(cv2.countNonZero(damage_mask))
    defect_percentage = damage_area / mango_area * 100.0 if mango_area else 0.0
    if defect_percentage < 1.5:
        severity, grade = "Low", "A"
    elif defect_percentage < 3.0:
        severity, grade = "Medium", "B"
    elif defect_percentage < 5.0:
        severity, grade = "Medium", "C"
    else:
        severity, grade = "High", "D"

    overlay = image.copy()
    overlay[damage_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
    contours, _ = cv2.findContours(
        damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return {
        "defect_percentage": round(float(defect_percentage), 2),
        "damage_percentage": round(float(defect_percentage), 2),
        "blemish_pixel_area": damage_area,
        "mango_pixel_area": mango_area,
        "severity": severity,
        "grade": grade,
        "defect_types": list(dict.fromkeys(defect_types)) or ["None"],
        "component_count": component_count,
        "components": [component for analysis in analyses for component in analysis["components"]],
        "damage_mask": damage_mask,
        "overlay": overlay,
        "safe_mango_mask": safe_mask,
        "blackhat": blackhat,
        "candidate_mask": candidate_mask,
        **component_masks,
    }


# Generate Report Helper Function
def generate_report_string(fruit_type, batch_id, ripeness, confidence, grade, defect_pct, severity, defect_types):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# FruitVision AI - Inspection Report

**Assessment ID:** FR-{datetime.now().strftime('%Y%m%d%H%M%S')}
**Date & Time:** {timestamp}
**Batch ID:** {batch_id}
**Fruit Type:** {fruit_type}

## Analysis Results
* **Predicted Ripeness:** {ripeness if ripeness else "N/A"}
* **Confidence Level:** {confidence:.1f}%
* **Surface Quality Grade:** {grade}
* **Defect Percentage:** {defect_pct:.2f}%
* **Severity:** {severity}
* **Defect Types Detected:** {', '.join(defect_types) if defect_types else 'None'}

---
*This report was generated automatically by FruitVision AI.*
"""
    return report


# ------------------------------------------------------------------
# PAGE 1: DASHBOARD
# ------------------------------------------------------------------
if page == "Dashboard":
    st.markdown("<div class='section-title'>Dashboard Overview</div>", unsafe_allow_html=True)
    st.caption("A quick look at everything analyzed so far.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Fruits Analyzed", f"{len(history_df):,}")
        st.caption("⬆ 8.2% vs last week")
    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Ripeness Distribution</div>", unsafe_allow_html=True)
        # Calculate and display exact percentages inside the card
        dist = history_df["Ripeness"].value_counts()
        total = len(history_df)
        pct_str = "  |  ".join([f"{k}: {v/total*100:.1f}%" for k, v in dist.items()])
        st.caption(pct_str)
        pie_df = dist.reset_index()
        pie_df.columns = ["Ripeness", "Count"]
        fig = px.pie(
            pie_df, names="Ripeness", values="Count", hole=0.55,
            color="Ripeness", color_discrete_map=RIPENESS_COLORS,
        )
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
            go_to("New Assessment")

# ------------------------------------------------------------------
# PAGE 2: NEW ASSESSMENT
# ------------------------------------------------------------------
elif page == "New Assessment":
    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("← Back"):
            go_to("Dashboard")
    with col_title:
        st.markdown("<div class='section-title'>New Fruit Assessment</div>", unsafe_allow_html=True)

    top1, top2 = st.columns(2)
    with top1:
        fruit_type = st.selectbox("Select Fruit Type", FRUITS)
    with top2:
        batch_id = st.text_input(
            "Batch ID",
            value=f"BATCH-{datetime.now().strftime('%Y%m%d')}-01",
            help="Used to group fruits scanned together, e.g. from the same crate.",
        )

    st.divider()

    # === STEP 1: PHOTO UPLOAD ===
    st.subheader("📷 Step 1: Upload Photo")
    upload_tab, camera_tab = st.tabs(["📁 Upload Photo", "📷 Use Camera"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Drag & drop or browse (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
    with camera_tab:
        use_camera = st.camera_input("Take a photo")

    if uploaded and use_camera:
        st.caption("You've provided both a file and a photo — using the uploaded file.")

    image_ready = uploaded is not None or use_camera is not None
    image = None
    analyze_clicked = False

    if uploaded:
        file_bytes = np.asarray(
            bytearray(uploaded.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif use_camera:
        file_bytes = np.asarray(
            bytearray(use_camera.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.write("")
    analyze_clicked = st.button(
        "🔍 Analyze Fruit",
        type="primary",
        disabled=not image_ready,
        use_container_width=True,
    )
    if not image_ready:
        st.caption("Upload a photo or use your camera above to enable analysis.")

    st.divider()

    # === STEP 2: RESULTS ===
    st.subheader("📊 Step 2: Analysis Results")

    current_step = 3 if analyze_clicked else (2 if image_ready else 1)
    step_indicator(current_step)

    if not image_ready:
        empty_state(
            "📸",
            "No photo yet",
            "Upload a photo or use your camera above to get started.",
        )
    elif not analyze_clicked:
        empty_state(
            "🔍",
            "Ready when you are",
            "Click **Analyze Fruit** above to run the AI assessment.",
        )
    elif analyze_clicked:
        start = time.time()
        # The detector must see every foreground component.  Selecting only
        # the largest component here would lose separated mangoes in a tray
        # or multi-fruit photograph.
        result = preprocessor.preprocess(
            image,
            keep_all_components=True,
        )
        preprocessing_time = time.time() - start

        # ------------------------------------------------------------------
        # MODULE 0: MANGO IDENTITY GATE
        # ------------------------------------------------------------------
        mango_identifier = load_mango_identifier(
            model_signature("models/mango_identifier.keras")
        )
        detected_objects = mango_identifier.identify_objects(
            image,
            preprocessed=result,
        )
        accepted_objects = [
            item for item in detected_objects if item["is_mango"]
        ]
        rejected_objects = [
            item for item in detected_objects if not item["is_mango"]
        ]
        best_gate_score = max(
            (item["score"] for item in detected_objects),
            default=0.0,
        )
        gate_method = (
            detected_objects[0]["method"]
            if detected_objects else "HSV + contour fallback"
        )
        if not accepted_objects:
            st.error("Rejected: this image does not contain a clear mango.")
            st.caption(
                f"Mango gate score: {best_gate_score * 100:.1f}% "
                f"({gate_method}). Ripeness analysis was not run."
            )
            gate_col1, gate_col2 = st.columns(2)
            with gate_col1:
                st.image(image, channels="BGR", caption="Rejected input")
            with gate_col2:
                st.image(
                    result["mask"],
                    clamp=True,
                    caption="Foreground candidate used by mango gate",
                )
            st.stop()

        st.info(
            f"Detected {len(accepted_objects)} mango object(s) in the image."
        )
        if rejected_objects:
            st.warning(
                f"Detected {len(accepted_objects)} mango(s). "
                f"Ignored {len(rejected_objects)} non-mango object(s)."
            )

        with st.spinner("Detecting blemishes and surface damage..."):
            object_blemish_results = [
                blemish_detector.analyze(
                    image=mango_object["crop"],
                    mango_mask=mango_object["processed"]["mask"],
                )
                for mango_object in accepted_objects
            ]
            blemish_result = combine_object_blemish_results(
                image=result["original"],
                objects=accepted_objects,
                analyses=object_blemish_results,
            )

        defect_pct = blemish_result["defect_percentage"]
        severity = blemish_result["severity"]
        grade = blemish_result["grade"]
        defect_types = blemish_result["defect_types"]

        mango_pixel_count = int(cv2.countNonZero(result["mask"]))
        total_pixel_count = result["mask"].shape[0] * result["mask"].shape[1]

        # ==========================================================
        # === MODULE 2: HYBRID RIPENESS CLASSIFICATION ===
        # ==========================================================
        ripeness = None
        confidence = 0.0
        raw_ripeness = None
        classification_error = None
        probabilities = {}
        hsv_features = {}
        statistical_analysis = {}
        colour_histogram = {}
        feature_count = 0
        inference_time_ms = 0.0
        object_predictions = []

        try:
            classifier = load_ripeness_classifier(
                model_signature("models/mango_ripeness.keras")
            )
        except FileNotFoundError as e:
            classification_error = str(e)
        else:
            with st.spinner("Analyzing your photo..."):
                for mango_object in accepted_objects:
                    object_result = classifier.predict(
                        mango_object["processed"]["segmented"],
                        mask=mango_object["processed"]["mask"],
                    )
                    object_predictions.append((mango_object, object_result))

            # Keep the existing single-object UI variables for the report and
            # gauge.  The individual results are rendered below as well.
            ripeness_result = object_predictions[0][1]

            raw_prediction = ripeness_result["prediction"]
            prediction = format_class_label(raw_prediction)
            confidence = ripeness_result["confidence"]
            probabilities = ripeness_result["probabilities"]
            hsv_features = ripeness_result["hsv_features"]
            statistical_analysis = ripeness_result["statistical_analysis"]
            colour_histogram = ripeness_result["colour_histogram"]
            feature_count = ripeness_result["feature_count"]
            inference_time_ms = ripeness_result["inference_time_ms"]

            ripeness = prediction
            raw_ripeness = raw_prediction

        if classification_error:
            st.error(classification_error)
            st.caption(
                "Run `python training/train_model.py` to generate "
                "`models/mango_ripeness.keras` and "
                "`models/class_indices.json`, then rerun the app."
            )
        else:
            # ---- HERO RESULT CARD --------------------------------
            icon, advice = RIPENESS_ADVICE.get(prediction, ("ℹ️", ""))
            hero_color = RIPENESS_COLORS.get(prediction, PRIMARY)
            st.markdown(
                f"""
                <div class="hero-card" style="--hero-color:{hero_color};">
                    <div class="hero-title">Predicted ripeness</div>
                    {ripeness_badge(prediction)}
                    <div class="hero-advice">{icon} {advice}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence < 70.0:
                st.warning(
                    f"Low model confidence ({confidence:.1f}%). "
                    "Please retake the photo under even lighting or review "
                    "the mango manually."
                )

            if len(object_predictions) > 1:
                st.markdown("##### Individual mango results")
                object_columns = st.columns(
                    min(3, len(object_predictions))
                )
                for object_position, (mango_object, object_result) in enumerate(
                    object_predictions
                ):
                    object_label = format_class_label(
                        object_result["prediction"]
                    )
                    with object_columns[object_position % len(object_columns)]:
                        st.image(
                            mango_object["processed"]["segmented_rgb"],
                            caption=f"Mango {object_position + 1}",
                        )
                        st.markdown(
                            f"**{object_label}** "
                            f"({object_result['confidence']:.1f}% confidence)"
                        )
                        if object_result["confidence"] < 70.0:
                            st.caption("Low confidence — manual review recommended.")

            gc1, gc2 = st.columns([1, 1])
            with gc1:
                st.plotly_chart(
                    confidence_gauge(confidence, prediction),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.caption("Model confidence")
            with gc2:
                st.image(result["segmented_rgb"], caption="What the AI focused on")

        # ==========================================================
        # === ACTIONS (SAVE / REPORT) ===
        # ==========================================================
        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            report_text = generate_report_string(
                fruit_type, batch_id, ripeness, confidence, grade,
                defect_pct, severity, defect_types
            )
            st.download_button(
                label="📄 Generate Report",
                data=report_text,
                file_name=f"FR-{datetime.now().strftime('%Y%m%d%H%M%S')}_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with b2:
            if st.button("💾 Save to History", use_container_width=True):
                new_id = f"FR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                st.session_state.saved_assessments.append(
                    {
                        "ID": new_id,
                        "Fruit Type": fruit_type,
                        "Ripeness": ripeness,
                        "Grade": grade,
                        "Defect %": defect_pct,
                        "Date": datetime.now(),
                    }
                )
                st.success(f"Saved as {new_id} — view it in History / Reports.")
        with b3:
            st.selectbox("Export as", ["PDF", "CSV", "JSON"], label_visibility="collapsed")

        # ---- TECHNICAL DETAILS (collapsed by default) --------
        st.write("")
        st.markdown("##### Want to see how we got this result?")

        with st.expander("🖼️ Image preprocessing"):
            st.caption(
                f"Aspect-preserving letterbox → BGR→HSV → Gaussian filter → "
                f"{result['contrast_method']} contrast enhancement → "
                f"HSV segmentation → morphological cleanup → connected "
                f"components. Preprocessing took "
                f"{preprocessing_time:.3f}s."
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, channels="BGR", caption="Original")
                st.image(result["resized"], channels="BGR", caption="Model input (224×224, aspect-preserved)")
            with col2:
                st.image(
                    result["mask"], clamp=True,
                    caption="Segmentation mask (white = mango)",
                )
                st.image(
                    result["segmented_rgb"],
                    caption="Background removed",
                )
            st.metric(
                "Mango Pixels Detected",
                f"{mango_pixel_count:,} / {total_pixel_count:,} "
                f"({mango_pixel_count / total_pixel_count * 100:.1f}%)"
            )
            if mango_pixel_count < 0.02 * total_pixel_count:
                st.warning(
                    "⚠️ Very few mango pixels detected — segmentation may have "
                    "failed for this image. The ripeness result above may be "
                    "unreliable."
                )

        with st.expander("🧪 Ripeness features (HSV, colour, statistics)"):
            st.caption(
                "These numbers are fed into the trained AI model as learned "
                "inputs — they are not manually mapped to ripeness classes "
                "using fixed rules."
            )
            st.markdown("**Probability by class**")
            sorted_probs = sorted(
                probabilities.items(), key=lambda kv: kv[1], reverse=True
            )
            display_labels = [format_class_label(c) for c, _ in sorted_probs]
            prob_fig = go.Figure(
                data=[
                    go.Bar(
                        x=display_labels,
                        y=[v for _, v in sorted_probs],
                        marker_color=[
                            RIPENESS_COLORS.get(label, PRIMARY_LIGHT)
                            for label in display_labels
                        ],
                        text=[f"{v:.1f}%" for _, v in sorted_probs],
                        textposition="outside",
                    )
                ]
            )
            prob_fig.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=280,
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(prob_fig, use_container_width=True)

            st.markdown("**HSV statistics (mean / std / median)**")
            hv1, hv2, hv3 = st.columns(3)
            hv1.metric("Hue", f"{hsv_features['mean_h']:.1f}",
                        help=f"std {hsv_features['std_h']:.1f}, "
                             f"median {statistical_analysis['median_h']:.1f}")
            hv2.metric("Saturation", f"{hsv_features['mean_s']:.1f}",
                        help=f"std {hsv_features['std_s']:.1f}, "
                             f"median {statistical_analysis['median_s']:.1f}")
            hv3.metric("Value", f"{hsv_features['mean_v']:.1f}",
                        help=f"std {hsv_features['std_v']:.1f}, "
                             f"median {statistical_analysis['median_v']:.1f}")

            st.markdown("**Colour histogram (48 features)**")
            h_vals, s_vals, v_vals = split_histogram_dict(colour_histogram)
            hist_fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Hue", "Saturation", "Value"),
            )
            hist_fig.add_trace(
                go.Bar(x=list(range(16)), y=h_vals, marker_color=PRIMARY), row=1, col=1
            )
            hist_fig.add_trace(
                go.Bar(x=list(range(16)), y=s_vals, marker_color=SECONDARY), row=1, col=2
            )
            hist_fig.add_trace(
                go.Bar(x=list(range(16)), y=v_vals, marker_color=PRIMARY_LIGHT), row=1, col=3
            )
            hist_fig.update_layout(height=260, showlegend=False, margin=dict(t=40, b=20))
            hist_fig.update_xaxes(title_text="Bin")
            hist_fig.update_yaxes(title_text="Proportion", row=1, col=1)
            st.plotly_chart(hist_fig, use_container_width=True)

            st.markdown("**Model information**")
            mi1, mi2, mi3, mi4 = st.columns(4)
            mi1.metric("Architecture", "EfficientNetB0 + Fusion")
            mi2.metric("Inputs", f"224×224 + {feature_count} feats")
            mi3.metric("Classes", str(len(probabilities)))
            mi4.metric("Inference", f"{inference_time_ms:.1f} ms")

        with st.expander(f"🎨 Surface quality grading {'':s}"):
            st.caption("🚧 Preview data — this module is not implemented yet.")
            st.markdown(grade_badge(grade), unsafe_allow_html=True)
            stars = {"A": "★★★★★", "B": "★★★★☆", "C": "★★★☆☆", "D": "★★☆☆☆"}[grade]
            st.markdown(f"**{stars}**")
            mc1, mc2 = st.columns(2)
            mc1.metric("Surface Smoothness", round(random.uniform(0.5, 0.99), 2))
            mc2.metric("Shape Score", round(random.uniform(0.5, 0.99), 2))

        # ==========================================================
        # === CLEAN BLEMISH & DAMAGE DETECTION UI (DROPDOWN STYLE) ===
        # ==========================================================
        # Changed to an expander to match the "dropdown list" style of other UI sections.
        # Set to expanded=False to be closed by default as requested.
        with st.expander("🔬 Surface Defect & Blemish Analysis", expanded=False):
            # --- Side-by-Side Image Comparison ---
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="📷 Original Photo",
                    use_container_width=True,
                )
            with col_img2:
                st.image(
                    cv2.cvtColor(blemish_result["overlay"], cv2.COLOR_BGR2RGB),
                    caption="🔴 Defect Overlay (Red = Blemish/Damage)",
                    use_container_width=True,
                )

            # --- Blemish Metrics (Using CSS 'metric-card') ---
            st.markdown("##### Blemish Metrics")
            bm1, bm2, bm3, bm4 = st.columns(4)
            with bm1:
                metric_card("Defect Percentage", f"{defect_pct:.2f}%")
            with bm2:
                metric_card("Severity", severity)
            with bm3:
                metric_card("Quality Grade", grade)
            with bm4:
                metric_card("Detected Regions", blemish_result["component_count"])

            st.write(f"**Detected Defect Types:** `{', '.join(defect_types) if defect_types else 'None'}`")

            # --- Hidden Technical Debug Data ---
            with st.expander("🧪 View Technical Debug Data (Binary Damage Mask & Blackhat)"):
                cdb1, cdb2 = st.columns(2)
                with cdb1:
                    st.image(blemish_result["damage_mask"], clamp=True, caption="Binary Damage Mask (White = Defect)")
                with cdb2:
                    st.image(blemish_result["blackhat"], clamp=True, caption="Blackhat Transform (Highlights Dark Spots)")

# ------------------------------------------------------------------
# PAGE 3: ASSESSMENT DETAILS
# ------------------------------------------------------------------
elif page == "Assessment Details":
    st.markdown("<div class='section-title'>Assessment Details</div>", unsafe_allow_html=True)
    selected_id = st.selectbox("Select Assessment", history_df["ID"].head(30))
    row = history_df[history_df["ID"] == selected_id].iloc[0]

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
        with st.expander("Preprocessing"):
            st.write("Resolution: 224x224 · CLAHE enhancement ✓ · Mango segmented ✓")
        with st.expander("Ripeness"):
            st.write(
                f"Predicted class: {row['Ripeness']} · "
                f"Model: EfficientNetB0 + 63-Feature Fusion (Hybrid) · "
                f"Confidence: {round(random.uniform(70,99),1)}%"
            )
        with st.expander("Quality (preview)"):
            st.write(f"Shape score: {round(random.uniform(0.5,0.99),2)} · Surface defects: {row['Defect %']}%")
        with st.expander("Blemish (preview)"):
            st.write(f"Area affected: {row['Defect %']}% · Types: {', '.join(random.sample(DEFECT_TYPES, k=2))}")

# ------------------------------------------------------------------
# PAGE 4: HISTORY / REPORTS
# ------------------------------------------------------------------
elif page == "History / Reports":
    st.markdown("<div class='section-title'>Assessment History</div>", unsafe_allow_html=True)

    if st.session_state.saved_assessments:
        st.markdown("##### This session")
        session_df = pd.DataFrame(st.session_state.saved_assessments)
        session_disp = session_df.copy()
        session_disp["Ripeness"] = session_disp["Ripeness"].apply(ripeness_badge)
        session_disp["Grade"] = session_disp["Grade"].apply(grade_badge)
        session_disp["Date"] = session_disp["Date"].dt.strftime("%Y-%m-%d %H:%M")
        st.write(
            session_disp[["ID", "Fruit Type", "Ripeness", "Grade", "Defect %", "Date"]].to_html(
                escape=False, index=False
            ),
            unsafe_allow_html=True,
        )
        st.write("")

    st.markdown("##### All assessments")
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

    if filtered.empty:
        empty_state("🗂️", "No matching assessments", "Try a different search term or fruit filter.")
    else:
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
        fig2 = px.bar(
            bar_data, x="Fruit Type", y="Count", color="Ripeness",
            title="Ripeness Distribution by Fruit Type",
            color_discrete_map=RIPENESS_COLORS,
        )
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
        st.selectbox("Model Selection", ["EfficientNetB0 + 63-Feature Fusion (Hybrid)"], disabled=True)
        st.caption(
            "Fuses EfficientNetB0 deep features with a 63-value colour/statistical "
            "feature vector (6 HSV stats + 48-bin colour histogram + 9 statistical measures) "
            "in a single trained classifier — see training/train_model.py."
        )
        st.slider("Confidence Threshold", 0, 100, 80,
                   help="Predictions below this confidence are flagged for manual review.")
        st.number_input("Batch Processing Size", min_value=1, max_value=256, value=32)
        st.radio("Processing Unit", ["GPU", "CPU"], horizontal=True)

    with tabs[3]:
        st.write("**Ripeness Thresholds**")
        st.caption(
            "Ripeness is predicted by the trained hybrid classifier. HSV, colour-histogram, "
            "and statistical features are learned numerical inputs to that model, not "
            "manual thresholds."
        )
        st.write("**Quality Grading Thresholds**")
        st.slider("Grade A minimum score", 0, 100, 90)
        st.slider("Grade B minimum score", 0, 100, 75)
        st.slider("Grade C minimum score", 0, 100, 60)
        st.write("**Defect Thresholds**")
        st.slider("Defect Warning Threshold (%)", 0, 100, 8)
        st.slider("Auto-Reject Threshold (%)", 0, 100, 20)

    st.write("")
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved.")
