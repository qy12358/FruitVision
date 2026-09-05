"""ui / pages / assess for ManGo or Stay."""

from config import FRUIT_TYPE
from services.assessment import perform_assessment
from services.storage import save_assessment
from ui.assessment_results import render_live_assessment
from ui.assessment_results import render_rejected_analysis
from ui.components import empty_state
from ui.components import page_header
from ui.components import step_indicator
import cv2
import hashlib
import numpy as np
import sqlite3
import streamlit as st

def render(history_df):
    page_header(
        "New assessment",
        "Assess a Harumanis mango",
        "Add one clear mango photo to estimate ripeness, inspect visible surface defects and produce a quality grade.",
    )

    st.markdown(
        """
        <div class="notice-box">
        <strong>For a reliable assessment</strong><br>
        Keep the whole mango visible, use even lighting, avoid strong glare or deep shadows,
        and place the fruit against a simple background where possible.
        </div>
        """,
        unsafe_allow_html=True,
    )

    batch_id = st.session_state.batch_id_input
    top1, top2 = st.columns([1, 2])
    with top1:
        st.markdown(
            f"""
            <div class="fixed-produce">
                <div class="fixed-produce-label">Produce</div>
                <div class="fixed-produce-value">{FRUIT_TYPE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown(
            f"""
            <div class="fixed-produce">
                <div class="fixed-produce-label">Batch ID</div>
                <div class="fixed-produce-value">{batch_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The batch ID is generated automatically for this assessment session.")

    st.divider()
    st.markdown("<div class='section-title'>1. Add a mango photo</div>", unsafe_allow_html=True)
    upload_tab, camera_tab = st.tabs(["Upload a photo", "Take a photo"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Choose a clear photo (JPG, PNG or WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="assessment_upload",
        )
    with camera_tab:
        use_camera = st.camera_input("Take a clear photo of the mango", key="assessment_camera")

    source_bytes = None
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        if use_camera is not None:
            st.caption("Both sources are present; the uploaded photo will be used.")
    elif use_camera is not None:
        source_bytes = use_camera.getvalue()

    image = None
    image_ready = bool(source_bytes)
    if image_ready:
        file_array = np.frombuffer(source_bytes, dtype=np.uint8)
        image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if image is None:
            image_ready = False
            st.error("The selected image could not be opened. Please choose another JPG, PNG or WebP file.")
        else:
            fingerprint = hashlib.sha256(source_bytes).hexdigest()
            if fingerprint != st.session_state.current_image_hash:
                st.session_state.current_image_hash = fingerprint
                st.session_state.current_analysis = None
                st.session_state.current_saved_id = None

    analyse_clicked = st.button(
        "Analyse mango",
        type="primary",
        disabled=not image_ready,
        use_container_width=True,
    )

    if analyse_clicked and image is not None:
        with st.spinner("Assessing the mango image..."):
            try:
                completed_analysis = perform_assessment(image)
                st.session_state.current_analysis = completed_analysis
            except Exception as exc:
                st.session_state.current_analysis = None
                st.session_state.current_saved_id = None
                st.error(f"The assessment could not be completed: {exc}")
            else:
                st.session_state.current_saved_id = None

                # A completed mango assessment is saved automatically. Rejected
                # non-mango images are not added to assessment history.
                if completed_analysis.get("status") == "complete":
                    try:
                        saved_id = save_assessment(completed_analysis, batch_id)
                    except (sqlite3.Error, ValueError) as exc:
                        st.error(
                            "The assessment completed, but it could not be saved automatically: "
                            f"{exc}"
                        )
                    else:
                        st.session_state.current_saved_id = saved_id
                        st.session_state.selected_assessment_id = saved_id
                        st.rerun()

    st.divider()
    st.markdown("<div class='section-title'>2. Assessment result</div>", unsafe_allow_html=True)

    current_analysis = st.session_state.current_analysis
    if current_analysis is not None:
        current_step = 3
    elif image_ready:
        current_step = 2
    else:
        current_step = 1
    step_indicator(current_step)

    if current_analysis is None and not image_ready:
        empty_state(
            "Add a mango photo to begin",
            "Upload a photo or take one with your camera. Completed mango assessments are saved automatically and can also be exported.",
        )
    elif current_analysis is None:
        empty_state("Photo ready", "Select Analyse mango to run the assessment.")
    elif current_analysis.get("status") == "rejected":
        render_rejected_analysis(current_analysis)
    else:
        render_live_assessment(current_analysis, batch_id)

