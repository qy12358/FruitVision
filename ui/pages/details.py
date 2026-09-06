"""ui / pages / details for ManGo or Stay."""

from config import PRIMARY
from config import PRIMARY_LIGHT
from config import RIPENESS_COLORS
from config import SECONDARY
from plotly.subplots import make_subplots
from services.histograms import format_class_label
from services.histograms import split_histogram_dict
from services.reports import report_data_from_record
from services.storage import decode_png
from services.storage import fetch_assessment
from ui.components import empty_state
from ui.components import grade_badge
from ui.components import metric_card
from ui.components import page_header
from ui.education import render_hsv_histogram_tips
from ui.navigation import go_to
import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render(history_df):
    page_header(
        "Review",
        "Assessment details",
        "Review the actual saved result, images, measurements and reports for a previous assessment.",
    )

    if history_df.empty:
        empty_state(
            "No saved assessments are available",
            "Complete a mango assessment first. Successful assessments are saved automatically and can then be reviewed here.",
        )
        if st.button("Assess a mango", type="primary"):
            go_to("Assess Mango")
    else:
        ids = history_df["ID"].tolist()
        selected_session_id = st.session_state.get("selected_assessment_id")
        default_index = ids.index(selected_session_id) if selected_session_id in ids else 0
        selected_id = st.selectbox("Select assessment", ids, index=default_index)
        st.session_state.selected_assessment_id = selected_id
        record = fetch_assessment(selected_id)

        if record is None:
            st.error("This assessment could not be found in the saved records.")
        else:
            created = record.get("created_at_dt")
            created_text = created.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(created) else "Unavailable"
            h1, h2, h3 = st.columns(3)
            h1.markdown(f"**Assessment ID**  \n{record['assessment_id']}")
            h2.markdown(f"**Batch ID**  \n{record.get('batch_id') or 'Not specified'}")
            h3.markdown(f"**Date**  \n{created_text}")

            _, pdf_report, report_error = report_data_from_record(record)
            with st.container():
                st.download_button(
                    "Export PDF",
                    data=pdf_report or b"",
                    file_name=f"{record['assessment_id']}_report.pdf",
                    mime="application/pdf",
                    disabled=pdf_report is None,
                    width="stretch",
                )
                if report_error:
                    st.caption(report_error)

            original = decode_png(record.get("original_image"), cv2.IMREAD_COLOR)
            processed = decode_png(record.get("processed_image"), cv2.IMREAD_COLOR)
            overlay = decode_png(record.get("overlay_image"), cv2.IMREAD_COLOR)
            damage_mask = decode_png(record.get("damage_mask"), cv2.IMREAD_GRAYSCALE)
            blackhat = decode_png(record.get("blackhat_image"), cv2.IMREAD_GRAYSCALE)
            fruit_mask = decode_png(record.get("fruit_mask"), cv2.IMREAD_GRAYSCALE)

            image_tabs = st.tabs(["Original", "Processed mango", "Defect overlay", "Analysis masks"])
            with image_tabs[0]:
                if original is not None:
                    st.image(original, channels="BGR", caption="Original saved photo", width="stretch")
                else:
                    st.info("The original image is not available for this record.")
            with image_tabs[1]:
                if processed is not None:
                    st.image(processed, channels="BGR", caption="Mango area used for ripeness classification", width="stretch")
                else:
                    st.info("The processed image is not available for this record.")
            with image_tabs[2]:
                if overlay is not None:
                    st.image(overlay, channels="BGR", caption="Saved defect overlay", width="stretch")
                else:
                    st.info("The defect overlay is not available for this record.")
            with image_tabs[3]:
                mask_cols = st.columns(3)
                with mask_cols[0]:
                    if fruit_mask is not None:
                        st.image(fruit_mask, clamp=True, caption="Mango mask", width="stretch")
                with mask_cols[1]:
                    if damage_mask is not None:
                        st.image(damage_mask, clamp=True, caption="Damage mask", width="stretch")
                with mask_cols[2]:
                    if blackhat is not None:
                        st.image(blackhat, clamp=True, caption="Black-hat response", width="stretch")

            st.markdown("<div class='section-title'>Saved result</div>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                metric_card("Ripeness", record.get("ripeness") or "Unavailable")
            with r2:
                metric_card("Confidence", f"{float(record.get('confidence') or 0.0):.1f}%")
            with r3:
                metric_card("Quality grade", record.get("grade") or "Unavailable")
            with r4:
                metric_card("Defect coverage", f"{float(record.get('defect_percentage') or 0.0):.2f}%")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Blemish coverage", f"{float(record.get('blemish_percentage') or 0.0):.2f}%")
            s2.metric("Damage coverage", f"{float(record.get('damage_percentage') or 0.0):.2f}%")
            s3.metric("Severity", record.get("severity") or "Unavailable")
            s4.metric("Inference time", f"{float(record.get('inference_time_ms') or 0.0):.1f} ms")
            st.write(f"**Detected defect types:** {', '.join(record.get('defect_types', [])) or 'None'}")

            with st.expander("Preprocessing details"):
                detected_pixels = int(record.get("mango_pixel_count") or 0)
                total_pixels = int(record.get("total_pixel_count") or 0)
                percentage = detected_pixels / total_pixels * 100.0 if total_pixels else 0.0
                st.write(f"Contrast method: {record.get('contrast_method') or 'Unavailable'}")
                st.write(f"Preprocessing time: {float(record.get('preprocessing_time') or 0.0):.3f} seconds")
                st.write(f"Detected mango area: {detected_pixels:,} / {total_pixels:,} pixels ({percentage:.1f}%)")
                st.write(f"Detected mango objects: {int(record.get('accepted_count') or 0)}")
                st.write(f"Ignored non-mango objects: {int(record.get('rejected_count') or 0)}")

            with st.expander("Ripeness evidence"):
                probabilities = record.get("probabilities", {})
                if probabilities:
                    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
                    labels = [format_class_label(label) for label, _ in sorted_probs]
                    probability_fig = go.Figure(
                        go.Bar(
                            x=labels,
                            y=[float(value) for _, value in sorted_probs],
                            marker_color=[RIPENESS_COLORS.get(label, PRIMARY_LIGHT) for label in labels],
                            text=[f"{float(value):.1f}%" for _, value in sorted_probs],
                            textposition="outside",
                        )
                    )
                    probability_fig.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 100], height=280)
                    st.plotly_chart(probability_fig, width="stretch")
                else:
                    st.info("Class probabilities were not stored for this assessment.")

                histogram = record.get("colour_histogram", {})
                if histogram:
                    h_vals, s_vals, v_vals = split_histogram_dict(histogram)
                    hist_fig = make_subplots(
                        rows=1,
                        cols=3,
                        subplot_titles=("Hue: colour", "Saturation: vividness", "Value: brightness"),
                    )
                    hist_fig.add_trace(go.Bar(x=list(range(16)), y=h_vals, marker_color=PRIMARY), row=1, col=1)
                    hist_fig.add_trace(go.Bar(x=list(range(16)), y=s_vals, marker_color=SECONDARY), row=1, col=2)
                    hist_fig.add_trace(go.Bar(x=list(range(16)), y=v_vals, marker_color=PRIMARY_LIGHT), row=1, col=3)
                    hist_fig.update_layout(height=300, showlegend=False, margin=dict(t=50, b=20))
                    st.plotly_chart(hist_fig, width="stretch")
                    render_hsv_histogram_tips(record.get("ripeness") or "Unavailable", h_vals, s_vals, v_vals)

            with st.expander("Quality grading details"):
                quality = record.get("quality_result", {})
                st.markdown(grade_badge(record.get("grade")), unsafe_allow_html=True)
                st.write(quality.get("reason", "No grading explanation is available."))

