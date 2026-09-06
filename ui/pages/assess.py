"""ui / pages / assess for ManGo or Stay."""

from services.assessment import perform_assessment
from services.storage import save_assessment

from ui.assessment_results import (
    render_live_assessment,
    render_rejected_analysis,
)

from ui.components import (
    empty_state,
    page_header,
    step_indicator,
)

from ui.live_yolo_camera import (
    render_live_yolo_camera,
)

import cv2
import hashlib
import numpy as np
import sqlite3
import streamlit as st


def render(history_df):

    # ============================================================
    # PAGE HEADER
    # ============================================================

    page_header(
        "New assessment",
        "Assess a Harumanis mango",
        (
            "Add one clear mango photo to estimate ripeness, "
            "inspect visible surface defects and produce "
            "a quality grade."
        ),
    )


    # ============================================================
    # SESSION STATE FOR YOLO
    # ============================================================

    if "yolo_selected_frame" not in st.session_state:
        st.session_state.yolo_selected_frame = None

    if "yolo_detection_confidence" not in st.session_state:
        st.session_state.yolo_detection_confidence = None

    # ============================================================
    # BATCH ID
    # ============================================================

    batch_id = st.session_state.batch_id_input

    st.markdown(
        (
            '<div class="fixed-produce">'
            '<div class="fixed-produce-label">Batch ID</div>'
            f'<div class="fixed-produce-value">{batch_id}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        "The batch ID is generated automatically "
        "for this assessment session."
    )

    st.divider()

    # ============================================================
    # STEP 1 — ADD IMAGE
    # ============================================================

    st.markdown(
        "<div class='section-title'>"
        "1. Add a mango photo"
        "</div>",
        unsafe_allow_html=True,
    )

    (
        upload_tab,
        camera_tab,
        live_tab,
    ) = st.tabs(
        [
            "Upload a photo",
            "Take a photo",
            "Live mango detection",
        ]
    )

    # ============================================================
    # UPLOAD PHOTO
    # ============================================================

    with upload_tab:

        uploaded = st.file_uploader(
            "Choose a clear photo (JPG, PNG or WebP)",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="assessment_upload",
        )

    # ============================================================
    # TAKE PHOTO
    # ============================================================

    with camera_tab:

        use_camera = st.camera_input(
            "Take a clear photo of the mango",
            key="assessment_camera",
        )

    # ============================================================
    # LIVE YOLO DETECTION
    # ============================================================

    with live_tab:

        live_detection = render_live_yolo_camera()

    # ============================================================
    # IMAGE VARIABLES
    # ============================================================

    image = None
    image_ready = False

    source_type = None
    source_fingerprint = None

    # Live YOLO should analyse immediately
    # after clicking Analyse detected mango.
    run_analysis_now = False

    # ============================================================
    # PRIORITY 1 — LIVE YOLO
    # ============================================================

    if live_detection is not None:

        image = live_detection["image"].copy()

        image_ready = True
        source_type = "live_yolo"
        run_analysis_now = True

        # Store frame so it survives Streamlit reruns.
        st.session_state.yolo_selected_frame = (
            image.copy()
        )

        st.session_state.yolo_detection_confidence = (
            live_detection["confidence"]
        )

        success, encoded = cv2.imencode(
            ".jpg",
            image,
        )

        if success:

            source_fingerprint = hashlib.sha256(
                encoded.tobytes()
            ).hexdigest()

    # ============================================================
    # PRIORITY 2 — UPLOAD PHOTO
    # ============================================================

    elif uploaded is not None:

        source_bytes = uploaded.getvalue()

        file_array = np.frombuffer(
            source_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            file_array,
            cv2.IMREAD_COLOR,
        )

        source_type = "upload"

        if image is None:

            st.error(
                "The uploaded image could not be opened. "
                "Please choose another JPG, PNG or WebP image."
            )

        else:

            image_ready = True

            source_fingerprint = hashlib.sha256(
                source_bytes
            ).hexdigest()

    # ============================================================
    # PRIORITY 3 — TAKE PHOTO
    # ============================================================

    elif use_camera is not None:

        source_bytes = use_camera.getvalue()

        file_array = np.frombuffer(
            source_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            file_array,
            cv2.IMREAD_COLOR,
        )

        source_type = "camera"

        if image is None:

            st.error(
                "The camera photo could not be opened. "
                "Please take another photo."
            )

        else:

            image_ready = True

            source_fingerprint = hashlib.sha256(
                source_bytes
            ).hexdigest()

    # ============================================================
    # PRIORITY 4 — PREVIOUS YOLO FRAME
    #
    # Keeps the selected YOLO frame available after reruns.
    # ============================================================

    elif (
        st.session_state.yolo_selected_frame
        is not None
    ):

        image = (
            st.session_state
            .yolo_selected_frame
            .copy()
        )

        image_ready = True
        source_type = "live_yolo_saved"

        success, encoded = cv2.imencode(
            ".jpg",
            image,
        )

        if success:

            source_fingerprint = hashlib.sha256(
                encoded.tobytes()
            ).hexdigest()

    # ============================================================
    # RESET OLD RESULT WHEN IMAGE CHANGES
    # ============================================================

    if (
        image_ready
        and source_fingerprint is not None
        and source_fingerprint
        != st.session_state.current_image_hash
    ):

        st.session_state.current_image_hash = (
            source_fingerprint
        )

        st.session_state.current_analysis = None
        st.session_state.current_saved_id = None

    # ============================================================
    # IMAGE PREVIEW
    # ============================================================

    if image_ready:

        st.markdown(
            "<div class='section-title'>"
            "Selected image"
            "</div>",
            unsafe_allow_html=True,
        )

        preview_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------
        # LIVE YOLO CAPTION
        # --------------------------------------------------------

        if source_type in (
            "live_yolo",
            "live_yolo_saved",
        ):

            confidence = (
                st.session_state
                .yolo_detection_confidence
            )

            if confidence is not None:

                caption = (
                    "Live YOLO mango capture — "
                    f"detection confidence "
                    f"{confidence * 100:.1f}%"
                )

            else:

                caption = (
                    "Live YOLO mango capture"
                )

        # --------------------------------------------------------
        # CAMERA CAPTION
        # --------------------------------------------------------

        elif source_type == "camera":

            caption = "Camera photo"

        # --------------------------------------------------------
        # UPLOAD CAPTION
        # --------------------------------------------------------

        else:

            caption = "Uploaded photo"

        st.image(
            preview_rgb,
            caption=caption,
            width=520,
        )

    # ============================================================
    # NORMAL ANALYSE BUTTON
    #
    # Only upload and take-photo use this button.
    #
    # Live YOLO already has:
    #
    # Analyse detected mango
    #
    # inside render_live_yolo_camera().
    # ============================================================

    normal_analyse_clicked = False

    if source_type in (
        "upload",
        "camera",
    ):

        normal_analyse_clicked = st.button(
            "Analyse mango",
            type="primary",
            disabled=not image_ready,
            width="stretch",
            key="analyse_normal_mango",
        )

    # ============================================================
    # SHOULD FULL ANALYSIS RUN?
    # ============================================================

    should_run_assessment = (
        run_analysis_now
        or normal_analyse_clicked
    )

    # ============================================================
    # FULL ASSESSMENT
    #
    # SAME perform_assessment() FOR:
    #
    # Upload
    # Take photo
    # Live YOLO
    # ============================================================

    if (
        should_run_assessment
        and image is not None
    ):

        with st.spinner(
            "Assessing mango ripeness, "
            "surface blemishes, damage "
            "and quality..."
        ):

            try:

                completed_analysis = (
                    perform_assessment(
                        image
                    )
                )

                st.session_state.current_analysis = (
                    completed_analysis
                )

            except Exception as exc:

                st.session_state.current_analysis = None
                st.session_state.current_saved_id = None

                st.error(
                    "The assessment could not be completed: "
                    f"{exc}"
                )

            else:

                st.session_state.current_saved_id = None

                # =================================================
                # AUTO-SAVE SUCCESSFUL RESULT
                # =================================================

                if (
                    completed_analysis.get("status")
                    == "complete"
                ):

                    try:

                        saved_id = save_assessment(
                            completed_analysis,
                            batch_id,
                        )

                    except (
                        sqlite3.Error,
                        ValueError,
                    ) as exc:

                        st.error(
                            "The assessment completed, "
                            "but it could not be saved "
                            "automatically: "
                            f"{exc}"
                        )

                    else:

                        st.session_state.current_saved_id = (
                            saved_id
                        )

                        st.session_state.selected_assessment_id = (
                            saved_id
                        )

                        # IMPORTANT:
                        #
                        # NO st.rerun() HERE.
                        #
                        # The result should display immediately.

    # ============================================================
    # STEP 2 — ASSESSMENT RESULT
    # ============================================================

    st.divider()

    st.markdown(
        "<div class='section-title'>"
        "2. Assessment result"
        "</div>",
        unsafe_allow_html=True,
    )

    current_analysis = (
        st.session_state.current_analysis
    )

    # ============================================================
    # STEP INDICATOR
    # ============================================================

    if current_analysis is not None:

        current_step = 3

    elif image_ready:

        current_step = 2

    else:

        current_step = 1

    step_indicator(
        current_step
    )

    # ============================================================
    # NOTHING SELECTED
    # ============================================================

    if (
        current_analysis is None
        and not image_ready
    ):

        empty_state(
            "Add a mango photo to begin",
            (
                "Upload a photo, take a photo, "
                "or use live mango detection."
            ),
        )

    # ============================================================
    # IMAGE READY BUT NOT ANALYSED
    # ============================================================

    elif current_analysis is None:

        if source_type in (
            "live_yolo",
            "live_yolo_saved",
        ):

            empty_state(
                "Mango detected",
                (
                    "The mango was captured from "
                    "the live YOLO detector. "
                    "The assessment result will "
                    "appear here."
                ),
            )

        else:

            empty_state(
                "Photo ready",
                "Select Analyse mango to run the assessment.",
            )

    # ============================================================
    # REJECTED IMAGE
    # ============================================================

    elif (
        current_analysis.get("status")
        == "rejected"
    ):

        render_rejected_analysis(
            current_analysis
        )

    # ============================================================
    # COMPLETE RESULT
    # ============================================================

    else:

        render_live_assessment(
            current_analysis,
            batch_id,
        )