"""ui / assessment_results for ManGo or Stay."""

from config import FRUIT_TYPE
from config import PRIMARY
from config import PRIMARY_LIGHT
from config import RIPENESS_ADVICE
from config import RIPENESS_COLORS
from config import SECONDARY
from datetime import datetime
from modules.report_generator import (
    format_coverage,
    generate_pdf_report,
    generate_report_string,
)
from plotly.subplots import make_subplots
from services.histograms import format_class_label
from services.histograms import split_histogram_dict
from ui.components import confidence_gauge
from ui.components import grade_badge
from ui.components import metric_card
from ui.components import ripeness_badge
from ui.education import render_hsv_histogram_tips
from ui.education import render_model_selection_tip
import plotly.graph_objects as go
import streamlit as st

def render_rejected_analysis(analysis: dict):
    st.error("This photo could not be assessed because a clear Harumanis mango was not detected.")
    st.caption(
        f"Mango detection score: {analysis['best_gate_score'] * 100:.1f}% "
        f"({analysis['gate_method']}). Ripeness analysis was not run."
    )
    with st.expander("Why this photo could not be assessed"):
        detected_objects = analysis.get("detected_objects", [])
        if detected_objects:
            for item in detected_objects:
                probability = item.get("model_probability")
                if probability is None:
                    st.write(f"Object {item['object_index']}: no CNN mango probability was available")
                else:
                    st.write(f"Object {item['object_index']}: mango probability {probability * 100:.1f}%")
                st.caption(item.get("reason", "No additional detection details are available."))
        else:
            st.caption("No foreground candidate was found.")

    c1, c2 = st.columns(2)
    with c1:
        st.image(analysis["image"], channels="BGR", caption="Submitted photo", use_container_width=True)
    with c2:
        st.image(
            analysis["result"]["fruit_mask"],
            clamp=True,
            caption="Detected foreground area",
            use_container_width=True,
        )

def render_live_assessment(analysis: dict, batch_id: str):
    result = analysis["result"]
    blemish_result = analysis["blemish_result"]
    ripeness = analysis.get("ripeness")
    confidence = float(analysis.get("confidence", 0.0))
    classification_error = analysis.get("classification_error")
    probabilities = analysis.get("probabilities", {})
    hsv_features = analysis.get("hsv_features", {})
    statistical_analysis = analysis.get("statistical_analysis", {})
    colour_histogram = analysis.get("colour_histogram", {})
    quality_result = analysis.get("quality_result", {})
    defect_types = analysis.get("defect_types", [])
    grade = analysis.get("grade")

    if analysis.get("accepted_count", 0) > 1:
        st.info(
            f"{analysis['accepted_count']} mangoes were detected. Individual ripeness predictions are shown below; "
            "the surface defect measurements refer to the combined detected mango area."
        )
    elif analysis.get("accepted_count", 0) == 1:
        st.caption("One mango was detected and assessed.")
    if analysis.get("rejected_count", 0):
        st.caption(f"{analysis['rejected_count']} non-mango object(s) were ignored.")

    if classification_error:
        st.error(classification_error)
        st.caption(
            "The image and surface analysis completed, but the ripeness model could not produce a result. "
            "Check that the trained model files are available in the models folder."
        )
    else:
        advice = RIPENESS_ADVICE.get(ripeness, "")
        hero_colour = RIPENESS_COLORS.get(ripeness, PRIMARY)
        st.markdown(
            f"""
            <div class="hero-card" style="--hero-color:{hero_colour};">
                <div class="hero-title">Predicted ripeness</div>
                {ripeness_badge(ripeness)}
                <div class="hero-advice">{advice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if confidence < 70.0:
            st.warning(
                f"Model confidence is {confidence:.1f}%. Consider taking another photo under even lighting "
                "and checking the mango manually."
            )

        object_predictions = analysis.get("object_predictions", [])
        if len(object_predictions) > 1:
            st.markdown("##### Individual mango results")
            columns = st.columns(min(3, len(object_predictions)))
            for position, (mango_object, object_result) in enumerate(object_predictions):
                label = format_class_label(object_result["prediction"])
                with columns[position % len(columns)]:
                    st.image(
                        mango_object["processed"]["segmented_rgb"],
                        caption=f"Mango {position + 1}",
                        use_container_width=True,
                    )
                    st.markdown(f"**{label}** ({float(object_result['confidence']):.1f}% confidence)")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                confidence_gauge(confidence, ripeness),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption("Prediction confidence")
        with c2:
            st.image(result["segmented_rgb"], caption="Mango area analysed by the AI", use_container_width=True)

    st.write("")
    b1, b2 = st.columns(2)
    report_text = generate_report_string(
        fruit_type=FRUIT_TYPE,
        batch_id=batch_id,
        ripeness=ripeness,
        confidence=confidence,
        quality_result=quality_result,
        severity=analysis.get("severity"),
        defect_types=defect_types,
    )
    with b1:
        st.download_button(
            label="Download report summary",
            data=report_text,
            file_name=f"mango-assessment-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    try:
        pdf_report = generate_pdf_report(
            fruit_type=FRUIT_TYPE,
            batch_id=batch_id,
            ripeness=ripeness,
            confidence=confidence,
            quality_result=quality_result,
            severity=analysis.get("severity"),
            defect_types=defect_types,
            original_image=analysis["image"],
            overlay_image=blemish_result["overlay"],
        )
        pdf_error = None
    except (RuntimeError, ValueError) as exc:
        pdf_report = None
        pdf_error = str(exc)

    with b2:
        st.download_button(
            label="Export PDF",
            data=pdf_report or b"",
            file_name=f"mango-assessment-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf",
            mime="application/pdf",
            disabled=pdf_report is None,
            use_container_width=True,
        )
        if pdf_error:
            st.caption(pdf_error)

    saved_id = st.session_state.get("current_saved_id")
    if saved_id:
        st.caption(
            f"This assessment was saved automatically to History & Reports. Assessment ID: {saved_id}"
        )

    st.write("")
    st.markdown("<div class='section-title'>Understand this result</div>", unsafe_allow_html=True)

    with st.expander("Fruit identification", expanded=True):
        accepted_count = int(analysis.get("accepted_count", 0))
        rejected_count = int(analysis.get("rejected_count", 0))
        detection_score = float(analysis.get("best_gate_score", 0.0)) * 100.0
        detection_method = analysis.get("gate_method", "Unavailable")

        st.caption(
            "The system first checks whether the submitted photo contains a mango. "
            "Ripeness and surface analysis continue only after a mango has been identified."
        )
        i1, i2, i3 = st.columns(3)
        i1.metric("Identification result", "Mango detected")
        i2.metric("Mango detection score", f"{detection_score:.1f}%")
        i3.metric("Mangoes detected", str(accepted_count))
        st.caption(f"Identification method: {detection_method}.")
        if rejected_count:
            st.caption(
                f"{rejected_count} additional object(s) were identified as non-mango objects and excluded from the assessment."
            )

    with st.expander("How your photo was prepared"):
        st.caption(
            f"Aspect-preserving letterbox, BGR to HSV conversion, Gaussian filtering, "
            f"{result['contrast_method']} contrast enhancement, HSV segmentation, morphological cleanup "
            f"and connected-component processing were applied. Preprocessing took "
            f"{analysis['preprocessing_time']:.3f} seconds."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.image(analysis["image"], channels="BGR", caption="Original photo", use_container_width=True)
            st.image(result["resized"], channels="BGR", caption="Model input", use_container_width=True)
        with col2:
            st.image(result["candidate_mask"], clamp=True, caption="Foreground candidate", use_container_width=True)
            st.image(result["leaf_mask"], clamp=True, caption="Removed leaf layer", use_container_width=True)
            st.image(result["fruit_mask"], clamp=True, caption="Detected mango area", use_container_width=True)
            st.image(result["segmented_rgb"], caption="Background removed", use_container_width=True)
        detected = analysis.get("mango_pixel_count", 0)
        total = analysis.get("total_pixel_count", 0)
        proportion = (detected / total * 100.0) if total else 0.0
        st.metric("Mango area detected", f"{detected:,} / {total:,} ({proportion:.1f}%)")

    with st.expander("How ripeness was assessed"):
        if classification_error:
            st.warning("Ripeness feature details are unavailable because the ripeness classifier did not complete.")
        else:
            st.caption(
                "The AI uses the segmented mango image together with colour and statistical measurements. "
                "The HSV values are supporting features rather than fixed ripeness rules."
            )
            sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            display_labels = [format_class_label(label) for label, _ in sorted_probs]
            probability_fig = go.Figure(
                data=[
                    go.Bar(
                        x=display_labels,
                        y=[float(value) for _, value in sorted_probs],
                        marker_color=[RIPENESS_COLORS.get(label, PRIMARY_LIGHT) for label in display_labels],
                        text=[f"{float(value):.1f}%" for _, value in sorted_probs],
                        textposition="outside",
                    )
                ]
            )
            probability_fig.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=280,
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(probability_fig, use_container_width=True)

            if hsv_features and statistical_analysis:
                st.markdown("**HSV colour summary**")
                h1, h2, h3 = st.columns(3)
                h1.metric(
                    "Hue",
                    f"{float(hsv_features.get('mean_h', 0)):.1f}",
                    help=(
                        f"Standard deviation {float(hsv_features.get('std_h', 0)):.1f}; "
                        f"median {float(statistical_analysis.get('median_h', 0)):.1f}."
                    ),
                )
                h2.metric(
                    "Saturation",
                    f"{float(hsv_features.get('mean_s', 0)):.1f}",
                    help=(
                        f"Standard deviation {float(hsv_features.get('std_s', 0)):.1f}; "
                        f"median {float(statistical_analysis.get('median_s', 0)):.1f}."
                    ),
                )
                h3.metric(
                    "Value",
                    f"{float(hsv_features.get('mean_v', 0)):.1f}",
                    help=(
                        f"Standard deviation {float(hsv_features.get('std_v', 0)):.1f}; "
                        f"median {float(statistical_analysis.get('median_v', 0)):.1f}."
                    ),
                )

            if colour_histogram:
                st.markdown("**HSV colour distribution**")
                h_vals, s_vals, v_vals = split_histogram_dict(colour_histogram)
                histogram_fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=("Hue: colour", "Saturation: vividness", "Value: brightness"),
                )
                histogram_fig.add_trace(go.Bar(x=list(range(16)), y=h_vals, marker_color=PRIMARY), row=1, col=1)
                histogram_fig.add_trace(go.Bar(x=list(range(16)), y=s_vals, marker_color=SECONDARY), row=1, col=2)
                histogram_fig.add_trace(go.Bar(x=list(range(16)), y=v_vals, marker_color=PRIMARY_LIGHT), row=1, col=3)
                histogram_fig.update_layout(height=300, showlegend=False, margin=dict(t=50, b=20))
                histogram_fig.update_xaxes(title_text="Bin (low to high)")
                histogram_fig.update_yaxes(title_text="Proportion", row=1, col=1)
                st.plotly_chart(histogram_fig, use_container_width=True)
                with st.expander("Understand the HSV colour chart", expanded=True):
                    render_hsv_histogram_tips(ripeness, h_vals, s_vals, v_vals)

            st.markdown("**Model details**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Architecture", "EfficientNetB0 + Fusion")
            m2.metric("Inputs", f"224 x 224 + {analysis.get('feature_count', 0)} features")
            m3.metric("Classes", str(len(probabilities)))
            m4.metric("Inference", f"{analysis.get('inference_time_ms', 0.0):.1f} ms")
            with st.expander("Why this AI model is used"):
                render_model_selection_tip()

    with st.expander("Surface defect and blemish analysis"):
        st.caption(
            "The system measures visible blemishes and surface damage first. These measurements are then used by the quality grading stage below."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.image(analysis["image"], channels="BGR", caption="Original photo", use_container_width=True)
        with c2:
            st.image(blemish_result["overlay"], channels="BGR", caption="Defect overlay", use_container_width=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            metric_card("Total defect", f"{analysis['defect_pct']:.2f}%")
        with b2:
            metric_card("Blemish coverage", f"{analysis['blemish_pct']:.2f}%")
        with b3:
            metric_card("Damage coverage", f"{analysis['damage_pct']:.2f}%")
        st.write(f"**Detected defect types:** {', '.join(defect_types) if defect_types else 'None'}")
        with st.expander("Advanced image analysis"):
            d1, d2 = st.columns(2)
            with d1:
                st.image(blemish_result["damage_mask"], clamp=True, caption="Binary damage mask", use_container_width=True)
            with d2:
                st.image(blemish_result["blackhat"], clamp=True, caption="Black-hat transform", use_container_width=True)

    with st.expander("How the surface quality grade was produced"):
        st.caption(
            "After surface defects have been measured, the quality grading function combines blemish coverage, damage coverage and the ripeness result to produce the final surface quality grade."
        )
        st.markdown(grade_badge(grade), unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Blemish coverage", format_coverage(quality_result.get("blemish_coverage")))
        s2.metric("Damage coverage", format_coverage(quality_result.get("damage_coverage")))
        s3.metric("Severity", analysis.get("severity", "Unavailable"))
        st.caption(quality_result.get("reason", "No grading explanation is available."))

