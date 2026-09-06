"""ui / assessment_results for ManGo or Stay."""

from config import FRUIT_TYPE
from config import PRIMARY
from config import PRIMARY_LIGHT
from config import RIPENESS_ADVICE
from config import RIPENESS_COLORS
from config import SECONDARY
from datetime import datetime

from modules.report_generator import (
    build_report_details,
    format_coverage,
    generate_pdf_report,
)
from modules.quality_grader import QualityGrader, SURFACE_GRADING_RULES, SURFACE_GRADING_HEADERS

from plotly.subplots import make_subplots
from services.histograms import format_class_label
from services.histograms import split_histogram_dict
from ui.components import confidence_gauge
from ui.components import grade_badge
from ui.components import metric_card
from ui.components import ripeness_badge
from ui.education import render_hsv_histogram_tips

import plotly.graph_objects as go
import streamlit as st


def render_rejected_analysis(analysis: dict):
    st.error(
        "This photo could not be assessed because a clear Harumanis mango was not detected."
    )

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
                    st.write(
                        f"Object {item['object_index']}: "
                        "no CNN mango probability was available"
                    )
                else:
                    st.write(
                        f"Object {item['object_index']}: "
                        f"mango probability {probability * 100:.1f}%"
                    )

                st.caption(
                    item.get(
                        "reason",
                        "No additional detection details are available.",
                    )
                )

        else:
            st.caption("No foreground candidate was found.")

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            analysis["image"],
            channels="BGR",
            caption="Submitted photo",
            width="stretch",
        )

    with c2:
        st.image(
            analysis["result"]["fruit_mask"],
            clamp=True,
            caption="Detected foreground area",
            width="stretch",
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

    defect_coverage = blemish_result.get("defect_percentage", analysis.get("defect_pct"))
    quality_result = QualityGrader().grade_defects(defect_coverage, ripeness)
    defect_types = analysis.get("defect_types", [])
    grade = quality_result["grade"]

    # ============================================================
    # MANGO COUNT
    # ============================================================

    if analysis.get("accepted_count", 0) > 1:
        st.info(
            f"{analysis['accepted_count']} mangoes were detected. "
            "Individual ripeness predictions are shown below; "
            "the surface defect measurements refer to the combined detected mango area."
        )

    elif analysis.get("accepted_count", 0) == 1:
        st.caption(
            "One mango was detected and assessed."
        )

    if analysis.get("rejected_count", 0):
        st.caption(
            f"{analysis['rejected_count']} non-mango object(s) were ignored."
        )

    # ============================================================
    # OVERALL RESULT
    # ============================================================

    result_columns = st.columns([1, 1.35], gap="large")

    with result_columns[0]:
        st.image(
            analysis["image"],
            channels="BGR",
            caption="Uploaded mango photo",
            width="stretch",
        )

    with result_columns[1]:
        if classification_error:
            st.error(classification_error)
            st.caption(
                "The image and surface analysis completed, but the ripeness model "
                "could not produce a result. Check that the trained model files "
                "are available in the models folder."
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

            st.plotly_chart(
                confidence_gauge(confidence, ripeness),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption("Prediction confidence")

            if confidence < 70.0:
                st.warning(
                    f"Model confidence is {confidence:.1f}%. "
                    "Consider taking another photo under even lighting "
                    "and checking the mango manually."
                )

        summary_columns = st.columns(3)
        with summary_columns[0]:
            metric_card("Confidence", f"{confidence:.1f}%")
        with summary_columns[1]:
            metric_card("Quality grade", grade or "Unavailable")
        with summary_columns[2]:
            metric_card(
                "Defect coverage",
                format_coverage(defect_coverage),
            )

    # ============================================================
    # MULTIPLE MANGO RESULTS
    # ============================================================

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
                    width="stretch",
                )
                st.markdown(
                    f"**{label}** "
                    f"({float(object_result['confidence']):.1f}% confidence)"
                )

    # ============================================================
    # REPORT DOWNLOAD
    # ============================================================

    st.write("")

    # ============================================================
    # PDF REPORT
    # ============================================================

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
            report_details=build_report_details(analysis),
        )

        pdf_error = None

    except (
        RuntimeError,
        ValueError,
    ) as exc:
        pdf_report = None
        pdf_error = str(exc)

    st.download_button(
        label="Export PDF",
        data=pdf_report or b"",
        file_name=(
            f"mango-assessment-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        ),
        mime="application/pdf",
        disabled=pdf_report is None,
        width="stretch",
    )

    if pdf_error:
        st.caption(pdf_error)

    saved_id = st.session_state.get(
        "current_saved_id"
    )

    if saved_id:
        st.caption(
            "This assessment was saved automatically to History & Reports. "
            f"Assessment ID: {saved_id}"
        )

    st.write("")

    st.markdown(
        "<div class='section-title'>Understand this result</div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # FRUIT IDENTIFICATION
    # ============================================================

    with st.expander(
        "Fruit identification",
        expanded=True,
    ):
        accepted_count = int(
            analysis.get(
                "accepted_count",
                0,
            )
        )

        rejected_count = int(
            analysis.get(
                "rejected_count",
                0,
            )
        )

        detection_score = (
            float(
                analysis.get(
                    "best_gate_score",
                    0.0,
                )
            )
            * 100.0
        )

        detection_method = analysis.get(
            "gate_method",
            "Unavailable",
        )

        st.caption(
            "The system first checks whether the submitted photo contains a mango. "
            "Ripeness and surface analysis continue only after a mango has been identified."
        )

        i1, i2, i3 = st.columns(3)

        i1.metric(
            "Identification result",
            "Mango detected",
        )

        i2.metric(
            "Mango detection score",
            f"{detection_score:.1f}%",
        )

        i3.metric(
            "Mangoes detected",
            str(accepted_count),
        )

        st.caption(
            f"Identification method: {detection_method}."
        )

        if rejected_count:
            st.caption(
                f"{rejected_count} additional object(s) were identified as "
                "non-mango objects and excluded from the assessment."
            )

    # ============================================================
    # IMAGE PREPARATION
    # ============================================================

    with st.expander(
        "How your photo was prepared"
    ):
        st.caption(
            f"Aspect-preserving letterbox, BGR to HSV conversion, Gaussian filtering, "
            f"{result['contrast_method']} contrast enhancement, HSV segmentation, "
            f"morphological cleanup and connected-component processing were applied. "
            f"Preprocessing took {analysis['preprocessing_time']:.3f} seconds."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                analysis["image"],
                channels="BGR",
                caption="Original photo",
                width="stretch",
            )

            st.image(
                result["resized"],
                channels="BGR",
                caption="Model input",
                width="stretch",
            )

        with col2:
            st.image(
                result["candidate_mask"],
                clamp=True,
                caption="Foreground candidate",
                width="stretch",
            )

            st.image(
                result["leaf_mask"],
                clamp=True,
                caption="Removed leaf layer",
                width="stretch",
            )

            st.image(
                result["fruit_mask"],
                clamp=True,
                caption="Detected mango area",
                width="stretch",
            )

            st.image(
                result["segmented_rgb"],
                caption="Background removed",
                width="stretch",
            )

        detected = analysis.get(
            "mango_pixel_count",
            0,
        )

        total = analysis.get(
            "total_pixel_count",
            0,
        )

        proportion = (
            detected / total * 100.0
            if total
            else 0.0
        )

        st.metric(
            "Mango area detected",
            f"{detected:,} / {total:,} ({proportion:.1f}%)",
        )

    # ============================================================
    # RIPENESS DETAILS
    # ============================================================

    with st.expander(
        "How ripeness was assessed"
    ):
        if classification_error:
            st.warning(
                "Ripeness feature details are unavailable because "
                "the ripeness classifier did not complete."
            )

        else:
            st.caption(
                "The AI uses the segmented mango image together with colour "
                "and statistical measurements. The HSV values are supporting "
                "features rather than fixed ripeness rules."
            )

            sorted_probs = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            display_labels = [
                format_class_label(label)
                for label, _
                in sorted_probs
            ]

            probability_fig = go.Figure(
                data=[
                    go.Bar(
                        x=display_labels,
                        y=[
                            float(value)
                            for _, value
                            in sorted_probs
                        ],
                        marker_color=[
                            RIPENESS_COLORS.get(
                                label,
                                PRIMARY_LIGHT,
                            )
                            for label
                            in display_labels
                        ],
                        text=[
                            f"{float(value):.1f}%"
                            for _, value
                            in sorted_probs
                        ],
                        textposition="outside",
                    )
                ]
            )

            probability_fig.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[
                    0,
                    100,
                ],
                height=280,
                margin=dict(
                    t=10,
                    b=20,
                ),
            )

            st.plotly_chart(
                probability_fig,
                width="stretch",
            )

            # ====================================================
            # HSV COLOUR SUMMARY
            # ====================================================

            if (
                hsv_features
                and statistical_analysis
            ):
                st.markdown(
                    "**HSV colour summary**"
                )

                h1, h2, h3 = st.columns(3)

                h1.metric(
                    "Hue",
                    f"{float(hsv_features.get('mean_h', 0)):.1f}",
                    help=(
                        f"Standard deviation "
                        f"{float(hsv_features.get('std_h', 0)):.1f}; "
                        f"median "
                        f"{float(statistical_analysis.get('median_h', 0)):.1f}."
                    ),
                )

                h2.metric(
                    "Saturation",
                    f"{float(hsv_features.get('mean_s', 0)):.1f}",
                    help=(
                        f"Standard deviation "
                        f"{float(hsv_features.get('std_s', 0)):.1f}; "
                        f"median "
                        f"{float(statistical_analysis.get('median_s', 0)):.1f}."
                    ),
                )

                h3.metric(
                    "Value",
                    f"{float(hsv_features.get('mean_v', 0)):.1f}",
                    help=(
                        f"Standard deviation "
                        f"{float(hsv_features.get('std_v', 0)):.1f}; "
                        f"median "
                        f"{float(statistical_analysis.get('median_v', 0)):.1f}."
                    ),
                )

            # ====================================================
            # HSV DISTRIBUTION
            # ====================================================

            if colour_histogram:
                st.markdown(
                    "**HSV colour distribution**"
                )

                (
                    h_vals,
                    s_vals,
                    v_vals,
                ) = split_histogram_dict(
                    colour_histogram
                )

                histogram_fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=(
                        "Hue: colour",
                        "Saturation: vividness",
                        "Value: brightness",
                    ),
                )

                histogram_fig.add_trace(
                    go.Bar(
                        x=list(range(16)),
                        y=h_vals,
                        marker_color=PRIMARY,
                    ),
                    row=1,
                    col=1,
                )

                histogram_fig.add_trace(
                    go.Bar(
                        x=list(range(16)),
                        y=s_vals,
                        marker_color=SECONDARY,
                    ),
                    row=1,
                    col=2,
                )

                histogram_fig.add_trace(
                    go.Bar(
                        x=list(range(16)),
                        y=v_vals,
                        marker_color=PRIMARY_LIGHT,
                    ),
                    row=1,
                    col=3,
                )

                histogram_fig.update_layout(
                    height=300,
                    showlegend=False,
                    margin=dict(
                        t=50,
                        b=20,
                    ),
                )

                histogram_fig.update_xaxes(
                    title_text="Bin (low to high)"
                )

                histogram_fig.update_yaxes(
                    title_text="Proportion",
                    row=1,
                    col=1,
                )

                st.plotly_chart(
                    histogram_fig,
                    width="stretch",
                )

                with st.expander(
                    "Understand the HSV colour chart",
                    expanded=True,
                ):
                    render_hsv_histogram_tips(
                        ripeness,
                        h_vals,
                        s_vals,
                        v_vals,
                    )

    # ============================================================
    # SURFACE DEFECT ANALYSIS
    # ONLY THIS SECTION HAS BEEN CHANGED
    # ============================================================

    with st.expander(
        "Surface defect analysis"
    ):

        # --------------------------------------------------------
        # GET ACTUAL CURRENT VALUES
        # --------------------------------------------------------

        severity = analysis.get(
            "severity",
            "Unavailable",
        )

        defect_regions = int(
            blemish_result.get(
                "component_count",
                0,
            )
        )

        defect_pixels = int(
            blemish_result.get(
                "defect_pixel_area",
                0,
            )
        )

        mango_pixels = int(
            blemish_result.get(
                "mango_pixel_area",
                0,
            )
        )

        # --------------------------------------------------------
        # DESCRIPTION + INFO BUTTON
        # --------------------------------------------------------

        info_text_col, info_icon_col = st.columns(
            [
                12,
                1,
            ]
        )

        with info_text_col:
            st.caption(
                "The system detects visible defective regions on the mango "
                "surface and measures how much of the visible mango area is affected."
            )

        with info_icon_col:
            with st.popover(
                "ⓘ",
                help="How the surface defect result is determined",
            ):

                st.markdown(
                    "### How this result was determined"
                )

                # ==================================================
                # DEFECT COVERAGE INFORMATION
                # ==================================================

                st.markdown(
                    f"""
### Defect coverage: **{defect_coverage:.2f}%**

The trained YOLO segmentation model identifies pixels that belong to visible surface defects.

The calculation used is:

`Defect coverage (%) = Detected defect pixels ÷ Mango surface pixels × 100`

For this assessment:

- Detected defect pixels: **{defect_pixels:,}**
- Analysed mango surface pixels: **{mango_pixels:,}**
"""
                )

                if mango_pixels > 0:
                    st.markdown(
                        f"""
Calculation:

`{defect_pixels:,} ÷ {mango_pixels:,} × 100 = {defect_coverage:.2f}%`

Therefore, **{defect_coverage:.2f}%** of the analysed visible mango surface was detected as defective.
"""
                    )

                else:
                    st.warning(
                        "The mango surface pixel count is unavailable, "
                        "so the percentage calculation cannot be displayed."
                    )

                st.divider()

                # ==================================================
                # SEVERITY INFORMATION
                # ==================================================

                st.markdown(
                    f"""
### Severity: **{severity}**

Severity is determined from the defect coverage.

| Defect coverage | Severity |
|---|---|
| Less than 1.50% | Low |
| 1.50% to less than 5.00% | Medium |
| 5.00% or more | High |
"""
                )

                if defect_coverage < 1.5:
                    st.info(
                        f"The defect coverage is **{defect_coverage:.2f}%**. "
                        "This is below **1.50%**, therefore the severity is **Low**."
                    )

                elif defect_coverage < 5.0:
                    st.info(
                        f"The defect coverage is **{defect_coverage:.2f}%**. "
                        "This is between **1.50%** and less than **5.00%**, "
                        "therefore the severity is **Medium**."
                    )

                else:
                    st.info(
                        f"The defect coverage is **{defect_coverage:.2f}%**. "
                        "This is **5.00% or higher**, therefore the severity is **High**."
                    )

                st.divider()

                # ==================================================
                # DEFECT REGION INFORMATION
                # ==================================================

                st.markdown(
                    f"""
### Defect regions: **{defect_regions}**

A defect region represents one individual defective surface area detected by the trained segmentation model.

The model can detect several separate defective areas on the same mango.

For this assessment, the model detected:

**{defect_regions} separate defect region(s).**
"""
                )

                st.caption(
                    "The number of defect regions tells how many separate "
                    "surface-defect areas were found. It is different from "
                    "defect coverage, which measures the total affected surface area."
                )

        # --------------------------------------------------------
        # ORIGINAL MANGO + DEFECT OVERLAY
        # --------------------------------------------------------

        c1, c2 = st.columns(2)

        with c1:
            st.image(
                analysis["image"],
                channels="BGR",
                caption="Original mango",
                width="stretch",
            )

        with c2:
            st.image(
                blemish_result["overlay"],
                channels="BGR",
                caption="Detected defect",
                width="stretch",
            )

        # --------------------------------------------------------
        # FARMER-FACING METRICS
        # --------------------------------------------------------

        b1, b2, b3 = st.columns(3)

        with b1:
            metric_card(
                "Defect coverage",
                f"{defect_coverage:.2f}%",
            )

        with b2:
            metric_card(
                "Severity",
                severity,
            )

        with b3:
            metric_card(
                "Defect regions",
                str(defect_regions),
            )

            st.markdown(
            "<div style='height: 18px;'></div>",
            unsafe_allow_html=True,
        )
    # ============================================================
    # QUALITY GRADE
    # ============================================================

    with st.expander(
        "How the quality grade was produced",
        expanded=True,
    ):
        st.markdown(
            '<div class="grading-result">'
            '<div class="grading-result-label">Final quality grade</div>'
            f'{grade_badge(grade)}'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Criteria used for this grade**")
        s1, s2 = st.columns(2)

        s1.metric(
            "Ripeness stage",
            ripeness or "Unavailable",
        )

        s2.metric(
            "Total defect coverage",
            format_coverage(
                quality_result.get(
                    "defect_coverage"
                )
            ),
        )

        st.info(
            quality_result.get(
                "reason",
                "No grading explanation is available.",
            )
        )
        st.caption(
            "The system combines ripeness stage and total defect coverage to assign the grade. "
            "Rotten mangoes are always rejected. Severity and defect-region count are "
            "supporting measurements shown in Surface defect analysis."
        )
        st.markdown("**Grading rule reference**")
        st.caption(
            "The system uses the rules in this table to assign the quality grade "
            "based on the mango's ripeness stage and total defect coverage."
        )
        st.table([dict(zip(SURFACE_GRADING_HEADERS, row))
                  for row in SURFACE_GRADING_RULES])
