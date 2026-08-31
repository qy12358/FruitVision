"""ManGo or Stay — mango ripeness and surface quality assessment."""
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.assessment_service import create_batch_id, decode_image, process_assessment
from modules.blemish_detector import BlemishDetector
from modules.calibration import pixels_per_mm
from modules.history_storage import load_history, save_assessment, save_assessments
from modules.pdf_report import generate_assessment_pdf, generate_batch_pdf
from modules.preprocessing import ImagePreprocessor
from modules.ripeness_classifier import HybridRipenessClassifier

SYSTEM_NAME = "ManGo or Stay"
CLASS_ORDER = ["ripe", "rotten", "semi_ripe", "unripe"]
PRIMARY, PRIMARY_LIGHT, SECONDARY = "#2E7D32", "#4CAF50", "#FF9800"
BG_LIGHT, RED, YELLOW, GREEN = "#F5F5F5", "#F44336", "#FFC107", "#4CAF50"
RIPENESS_COLORS = {"Unripe": SECONDARY, "Semi-Ripe": YELLOW, "Ripe": GREEN, "Rotten": RED}
RIPENESS_BADGE_CLASS = {"Unripe": "badge-unripe", "Semi-Ripe": "badge-semi",
                         "Ripe": "badge-ripe", "Rotten": "badge-rotten"}
RIPENESS_ADVICE = {
    "Ripe": ("✅", "Ready to eat or sell now."),
    "Semi-Ripe": ("🕓", "A few more days are needed before it is fully ripe."),
    "Unripe": ("🌱", "Needs more time to ripen."),
    "Rotten": ("⚠️", "Not suitable for sale or consumption."),
}

preprocessor = ImagePreprocessor(resize=(224, 224))
blemish_detector = BlemishDetector(min_blemish_area=8, boundary_erosion=15)
st.set_page_config(page_title=SYSTEM_NAME, page_icon="🥭", layout="wide",
                   initial_sidebar_state="expanded")

# Restored teammate theme and reusable visual components.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
.main {{ background-color:{BG_LIGHT}; }}
.block-container {{ padding-top:1.7rem; padding-bottom:2rem; }}
.metric-card,.soft-card {{ background:white;border-radius:16px;padding:18px 20px;box-shadow:0 2px 10px rgba(0,0,0,.06);border:1px solid #eee; }}
.metric-card {{ min-height:94px; }}
.metric-title {{ font-size:13px;color:#757575;font-weight:500;margin-bottom:4px; }}
.metric-value {{ font-size:28px;font-weight:700;color:#1A1A2E; }}
.metric-subtitle {{ font-size:12px;color:#8a8a8a;margin-top:4px; }}
.badge {{ display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;color:white; }}
.badge-ripe,.badge-A {{ background-color:{GREEN}; }}
.badge-semi,.badge-C {{ background-color:{YELLOW};color:#1A1A2E; }}
.badge-unripe {{ background-color:{SECONDARY}; }} .badge-rotten,.badge-D {{ background-color:{RED}; }}
.badge-B {{ background-color:#8BC34A; }}
.chip {{ display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:#FFF3E0;color:#E65100;margin-left:8px;vertical-align:middle; }}
.logo-text {{ font-size:22px;font-weight:700;color:{PRIMARY}; }}
.section-title {{ font-size:20px;font-weight:700;color:#1A1A2E;margin:10px 0 14px; }}
div.stButton>button,div.stDownloadButton>button {{ border-radius:10px;font-weight:600; }}
.step-indicator {{ display:flex;align-items:center;gap:6px;margin:4px 0 22px;flex-wrap:wrap; }}
.step {{ display:flex;align-items:center;gap:7px;padding:6px 14px 6px 8px;border-radius:20px;font-size:13px;font-weight:600; }}
.step-num {{ width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px; }}
.step-todo {{ background:#EDEDED;color:#9E9E9E; }} .step-todo .step-num {{ background:#DADADA;color:#9E9E9E; }}
.step-active,.step-done {{ background:#E8F5E9;color:{PRIMARY}; }}
.step-active .step-num {{ background:{PRIMARY};color:white; }} .step-done .step-num {{ background:{PRIMARY_LIGHT};color:white; }}
.step-connector {{ width:20px;height:2px;background:#DADADA; }}
.hero-card {{ background:white;border-radius:18px;padding:24px 26px;box-shadow:0 2px 14px rgba(0,0,0,.07);border-left:6px solid var(--hero-color,{PRIMARY});margin-bottom:18px; }}
.hero-title {{ font-size:13px;color:#757575;font-weight:600;text-transform:uppercase;letter-spacing:.03em;margin-bottom:6px; }}
.hero-advice {{ font-size:15px;color:#444;margin-top:10px; }}
.empty-state {{ text-align:center;padding:46px 20px;color:#9E9E9E; }} .empty-state-icon {{ font-size:40px;margin-bottom:10px; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    classifier = HybridRipenessClassifier("models/efficientnet_fruit.keras", "models/class_indices.json")
    if classifier.class_names != CLASS_ORDER:
        raise RuntimeError(f"Class order must remain {CLASS_ORDER}; found {classifier.class_names}.")
    return classifier


def badge_html(text, css_class): return f"<span class='badge {css_class}'>{text}</span>"
def grade_badge(grade): return badge_html(grade, f"badge-{grade}")
def ripeness_badge(level): return badge_html(level, RIPENESS_BADGE_CLASS.get(level, "badge-ripe"))


def metric_card(title, value, subtitle=None):
    sub = f"<div class='metric-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"<div class='metric-card'><div class='metric-title'>{title}</div>"
                f"<div class='metric-value'>{value}</div>{sub}</div>", unsafe_allow_html=True)


def section_title(text):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def empty_state(icon, title, subtitle):
    st.markdown(f"<div class='empty-state'><div class='empty-state-icon'>{icon}</div>"
                f"<div style='font-weight:600;color:#616161'>{title}</div>"
                f"<div style='font-size:13px'>{subtitle}</div></div>", unsafe_allow_html=True)


def step_indicator(current):
    html = "<div class='step-indicator'>"
    for index, label in enumerate(["Upload photo", "Analyze", "See results"], 1):
        state = "step-done" if index < current else "step-active" if index == current else "step-todo"
        html += f"<div class='step {state}'><span class='step-num'>{index}</span>{label}</div>"
        if index < 3: html += "<div class='step-connector'></div>"
    st.markdown(html + "</div>", unsafe_allow_html=True)


def confidence_gauge(confidence, ripeness):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence,
        number={"suffix": "%", "font": {"size": 30}}, gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": RIPENESS_COLORS.get(ripeness, PRIMARY_LIGHT), "thickness": .28},
            "bgcolor": "white", "borderwidth": 0, "steps": [
                {"range": [0, 50], "color": "#FBE9E7"}, {"range": [50, 80], "color": "#FFF8E1"},
                {"range": [80, 100], "color": "#E8F5E9"}]}))
    fig.update_layout(height=190, margin=dict(l=20, r=20, t=15, b=10))
    return fig


def history_frame(): return pd.DataFrame(load_history())


def record_from_result(result):
    return {"ID": result["assessment_id"], "Batch ID": result.get("batch_id", "Single"),
            "Date": result["assessed_at"], "Mango": result["filename"],
            "Ripeness": result["predicted_ripeness"], "Confidence": result["confidence"],
            "Grade": result["grade"], "Defect %": result["defect_percentage"],
            "Severity": result["severity"], "Defect Types": result["defect_types"],
            "Processing Info": result["processing_info"], "Original Image": result.get("original_image"),
            "Processed Image": result["preprocessing"]["segmented_rgb"],
            "Defect Overlay": result.get("blemish", {}).get("overlay")}


def historical_record(assessment_id):
    record = dict(next(item for item in load_history() if item["ID"] == assessment_id))
    live = next((item for item in st.session_state.saved_assessments if item["ID"] == assessment_id), None)
    if live:
        for key in ("Original Image", "Processed Image", "Defect Overlay"):
            if live.get(key) is not None: record[key] = live[key]
    return record


def render_result(result):
    prediction = result["predicted_ripeness"]
    icon, advice = RIPENESS_ADVICE.get(prediction, ("ℹ️", ""))
    st.markdown(f"<div class='hero-card' style='--hero-color:{RIPENESS_COLORS.get(prediction, PRIMARY)}'>"
                f"<div class='hero-title'>Predicted ripeness</div>{ripeness_badge(prediction)}"
                f"<div class='hero-advice'>{icon} {advice}</div></div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    left.plotly_chart(confidence_gauge(result["confidence"], prediction), use_container_width=True,
                      config={"displayModeBar": False}); left.caption("Model confidence")
    right.image(result["preprocessing"]["segmented_rgb"], caption="What the AI focused on")
    section_title("Want to see how we got this result?")
    prep, blemish = result["preprocessing"], result["blemish"]
    with st.expander("🖼️ Image preprocessing"):
        st.caption(result["processing_info"])
        a, b = st.columns(2)
        a.image(result["original_image"], channels="BGR", caption="Original mango")
        a.image(prep["resized"], channels="BGR", caption="Letterboxed 224×224")
        b.image(prep["mask"], clamp=True, caption="Segmentation mask")
        b.image(prep["segmented_rgb"], caption="Background removed")
    with st.expander("🧪 Ripeness model details"):
        probabilities = result["ripeness_result"]["probabilities"]
        labels = [name.replace("_", "-").title() for name in probabilities]
        fig = go.Figure(go.Bar(x=labels, y=list(probabilities.values()),
            marker_color=[RIPENESS_COLORS.get(label, PRIMARY_LIGHT) for label in labels],
            text=[f"{value:.1f}%" for value in probabilities.values()], textposition="outside"))
        fig.update_layout(height=280, yaxis_title="Probability (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("🔬 Surface Defect & Blemish Analysis"):
        a, b = st.columns(2)
        a.image(result["original_image"], channels="BGR", caption="Original photo")
        b.image(blemish["overlay"], channels="BGR", caption="Defect overlay")
        cols = st.columns(4)
        values = [("Defect Percentage", f"{result['defect_percentage']:.2f}%"), ("Severity", result["severity"]),
                  ("Quality Grade", result["grade"]), ("Detected Regions", blemish["component_count"])]
        for col, value in zip(cols, values):
            with col: metric_card(*value)
        st.write(f"**Detected Defect Types:** `{', '.join(result['defect_types']) or 'None'}`")


PAGES = [
    "Dashboard",
    "New Assessment",
    "Batch Assessment",
    "History / Reports",
    "Settings"
]

ICONS = {
    "Dashboard": "📊",
    "New Assessment": "➕",
    "Batch Assessment": "📦",
    "History / Reports": "🕒",
    "Settings": "⚙️"
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "saved_assessments" not in st.session_state:
    st.session_state.saved_assessments = load_history()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# Apply navigation request BEFORE the radio widget is created
if "_pending_nav" in st.session_state:
    st.session_state.page = st.session_state.pop("_pending_nav")


def go_to(page_name):
    st.session_state["_pending_nav"] = page_name
    st.rerun()


with st.sidebar:
    st.markdown(
        "<div class='logo-text'>🥭 ManGo or Stay</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Snap a photo of a mango — get instant ripeness, quality and defect results."
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        PAGES,
        format_func=lambda name: f"{ICONS[name]}  {name}",
        label_visibility="collapsed",
        key="page"
    )

    st.markdown("---")

    if page != "New Assessment":
        if st.button(
            "➕ New Assessment",
            use_container_width=True,
            type="primary"
        ):
            go_to("New Assessment")

    st.caption(
        f"📌 {len(load_history())} assessment(s) saved persistently"
    )


if page == "Dashboard":
    section_title("Dashboard Overview"); st.caption("A quick look at every real mango assessment saved so far.")
    history = history_frame()
    if history.empty:
        empty_state("🥭", "No saved mango assessments yet", "Complete and save an assessment to populate this dashboard. No results are fabricated.")
    else:
        cols = st.columns(4)
        values = [("Total Mangoes Assessed", f"{len(history):,}"), ("Most Common Ripeness", history["Ripeness"].mode().iat[0]),
                  ("Grade A Mangoes", int((history["Grade"] == "A").sum())), ("Average Defect", f"{history['Defect %'].mean():.2f}%")]
        for col, value in zip(cols, values):
            with col: metric_card(*value)
        st.write(""); left, right = st.columns([6, 4])
        with left:
            section_title("Recent Assessments")
            recent = history.sort_values("Date", ascending=False).head(8).copy()
            recent["Ripeness"] = recent["Ripeness"].apply(ripeness_badge); recent["Grade"] = recent["Grade"].apply(grade_badge)
            recent["Date"] = recent["Date"].apply(lambda value: value.strftime("%Y-%m-%d %H:%M") if value else "")
            st.write(recent[["ID", "Batch ID", "Mango", "Ripeness", "Grade", "Defect %", "Date"]].to_html(escape=False, index=False), unsafe_allow_html=True)
        with right:
            section_title("Ripeness Distribution")
            dist = history["Ripeness"].value_counts().rename_axis("Ripeness").reset_index(name="Count")
            pie = px.pie(dist, names="Ripeness", values="Count", color="Ripeness", color_discrete_map=RIPENESS_COLORS, hole=.55)
            pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=260)
            st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
        a, b = st.columns(2)
        with a:
            grades = history["Grade"].value_counts().rename_axis("Grade").reset_index(name="Count")
            st.plotly_chart(px.bar(grades, x="Grade", y="Count", title="Quality Grade Distribution", color="Grade",
                color_discrete_map={"A": GREEN, "B": "#8BC34A", "C": YELLOW, "D": RED}), use_container_width=True)
        with b:
            trend = history.dropna(subset=["Date"]).copy(); trend["Day"] = trend["Date"].dt.date
            trend = trend.groupby("Day")["Defect %"].mean().reset_index()
            chart = px.line(trend, x="Day", y="Defect %", title="Average Defect Percentage"); chart.update_traces(line_color=PRIMARY_LIGHT)
            st.plotly_chart(chart, use_container_width=True)

elif page == "New Assessment":
    section_title("New Mango Assessment"); st.caption("Upload a Harumanis mango photo or capture one with your camera.")
    upload_tab, camera_tab = st.tabs(["📁 Upload Photo", "📷 Use Camera"])
    with upload_tab: uploaded = st.file_uploader("Drag & drop or browse (JPG, PNG, WebP)", type=["jpg", "jpeg", "png", "webp"], key="single_upload")
    with camera_tab: camera = st.camera_input("Take a mango photo")
    source = uploaded or camera; step_indicator(3 if st.session_state.latest_result else 2 if source else 1)
    if source and st.button("🔍 Analyze Mango", type="primary", use_container_width=True):
        try:
            image = decode_image(source.getvalue())
            with st.spinner("Analyzing ripeness and surface quality..."):
                result = process_assessment(image, preprocessor, load_classifier(), blemish_detector,
                                            getattr(source, "name", "camera.jpg"), batch_id="Single")
            result["original_image"] = image; st.session_state.latest_result = result
        except Exception as exc: st.error(f"Assessment failed: {exc}")
    result = st.session_state.latest_result
    if result:
        render_result(result); record = record_from_result(result); save_col, pdf_col = st.columns(2)
        if save_col.button("💾 Save to History", use_container_width=True):
            if save_assessment(record): st.session_state.saved_assessments.append(record); st.success(f"Saved as {record['ID']}.")
            else: st.info(f"Assessment {record['ID']} is already in history.")
        pdf_col.download_button("📄 Download PDF Report", generate_assessment_pdf(record), f"{record['ID']}_report.pdf",
                                "application/pdf", use_container_width=True)
    elif not source: empty_state("📸", "No photo yet", "Upload a photo or use your camera above to get started.")

elif page == "Batch Assessment":
    section_title("Batch Assessment"); st.caption("Assess multiple mangoes with the same real analysis pipeline.")
    files = st.file_uploader("📦 Upload mango images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="batch_upload")
    if files and st.button("🔍 Analyze Batch", type="primary", use_container_width=True):
        batch_id, results, errors = create_batch_id(), [], []; classifier = load_classifier(); progress = st.progress(0)
        for index, file in enumerate(files):
            try:
                image = decode_image(file.getvalue()); result = process_assessment(image, preprocessor, classifier, blemish_detector, file.name, batch_id=batch_id)
                result["original_image"] = image; results.append(result)
            except Exception as exc: errors.append(f"{file.name}: {exc}")
            progress.progress((index + 1) / len(files))
        st.session_state.batch_results, st.session_state.batch_id = results, batch_id
        for error in errors: st.warning(error)
    results = st.session_state.get("batch_results", [])
    if results:
        batch_id = st.session_state.get("batch_id", results[0]["batch_id"])
        st.markdown(f"<div class='soft-card'><b>Batch ID</b><span class='chip'>{batch_id}</span><br>"
                    f"<span style='color:#757575;font-size:13px'>{len(results)} mangoes processed</span></div>", unsafe_allow_html=True)
        records = [record_from_result(result) for result in results]
        summary = pd.DataFrame([{"Assessment ID": r["ID"], "Batch ID": r["Batch ID"], "Filename": r["Mango"],
            "Ripeness": r["Ripeness"], "Confidence": round(r["Confidence"], 2), "Grade": r["Grade"],
            "Defect %": round(r["Defect %"], 2), "Severity": r["Severity"]} for r in records])
        styled = summary.copy(); styled["Ripeness"] = styled["Ripeness"].apply(ripeness_badge); styled["Grade"] = styled["Grade"].apply(grade_badge)
        st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
        save_col, csv_col, pdf_col = st.columns(3)
        if save_col.button("💾 Save Batch to History", use_container_width=True):
            added = save_assessments(records); st.session_state.saved_assessments.extend(added)
            if added: st.success(f"{len(added)} assessments saved under Batch ID {batch_id}.")
            else: st.info("All batch assessments are already in history.")
        csv_col.download_button("⬇ Export Batch CSV", summary.to_csv(index=False).encode("utf-8"), f"{batch_id}_results.csv", "text/csv", use_container_width=True)
        pdf_col.download_button("📄 Download Batch PDF", generate_batch_pdf(batch_id, records), f"{batch_id}_report.pdf", "application/pdf", use_container_width=True)
    elif not files: empty_state("📦", "No batch selected", "Upload multiple mango images to begin a batch assessment.")

elif page == "History / Reports":
    section_title("Assessment History"); st.caption("Persistent, real assessment records. No mock data is displayed.")
    history = history_frame()
    if history.empty: empty_state("🗂️", "No saved assessments", "Save an assessment to make it available here after restarts.")
    else:
        cols = st.columns(4); values = [("Total Assessments", len(history)), ("Average Confidence", f"{history['Confidence'].mean():.1f}%"),
            ("Average Defect", f"{history['Defect %'].mean():.2f}%"), ("Batch Count", history.loc[history["Batch ID"] != "Single", "Batch ID"].nunique())]
        for col, value in zip(cols, values):
            with col: metric_card(*value)
        search = st.text_input("🔍 Search by assessment ID, batch ID or filename"); filtered = history.copy()
        if search:
            filtered = filtered[filtered["ID"].str.contains(search, case=False, na=False) | filtered["Batch ID"].str.contains(search, case=False, na=False) | filtered["Mango"].str.contains(search, case=False, na=False)]
        if filtered.empty: empty_state("🗂️", "No matching assessments", "Try a different search term.")
        else:
            display = filtered.copy(); display["Ripeness"] = display["Ripeness"].apply(ripeness_badge); display["Grade"] = display["Grade"].apply(grade_badge)
            display["Date"] = display["Date"].apply(lambda value: value.strftime("%Y-%m-%d %H:%M") if value else "")
            st.write(display[["ID", "Batch ID", "Date", "Mango", "Ripeness", "Confidence", "Grade", "Defect %", "Severity", "Defect Types"]].to_html(escape=False, index=False), unsafe_allow_html=True)
        section_title("Individual PDF Report")
        options = filtered["ID"].tolist() if not filtered.empty else history["ID"].tolist(); selected_id = st.selectbox("Select Assessment ID", options)
        record = historical_record(selected_id); detail = st.columns(4)
        for col, value in zip(detail, [("Mango", record["Mango"]), ("Ripeness", record["Ripeness"]), ("Quality Grade", record["Grade"]), ("Defect", f"{record['Defect %']:.2f}%")]):
            with col: metric_card(*value)
        st.download_button("📄 Download Selected PDF", generate_assessment_pdf(record), f"{selected_id}_report.pdf", "application/pdf", use_container_width=True)

elif page == "Settings":
    section_title("Settings"); general, model, calibration = st.tabs(["General", "AI Model", "Optional Calibration"])
    with general:
        st.text_input("System Name", SYSTEM_NAME, disabled=True); st.text_input("Mango Type", "Harumanis mango", disabled=True)
        st.selectbox("Language", ["English", "Malay", "Chinese"])
    with model:
        st.selectbox("Model Selection", ["EfficientNetB0 + 63-Feature Fusion (Hybrid)"], disabled=True)
        st.caption("The trained hybrid model and fixed class order remain unchanged."); st.write(f"**Class order:** {', '.join(CLASS_ORDER)}")
    with calibration:
        st.warning("Physical measurements are valid only when a known-size reference in the same image plane is supplied.")
        reference_pixels = st.number_input("Reference length in pixels", min_value=0.0, value=0.0)
        reference_mm = st.number_input("Known reference length in mm", min_value=0.0, value=0.0)
        if reference_pixels > 0 and reference_mm > 0: st.success(f"Calibration factor: {pixels_per_mm(reference_pixels, reference_mm):.4f} pixels/mm")
        else: st.caption("Calibration inactive. No pixel-to-mm conversion is performed.")
