"""Live YOLO mango detection."""

import threading

import av
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from services.models import (
    load_yolo_mango_detector,
    model_signature,
)


class MangoVideoProcessor(
    VideoProcessorBase
):
    """
    Process browser camera frames
    continuously using YOLO.
    """

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_crop = None
        self.latest_detection = None

        self.detector = (
            load_yolo_mango_detector(
                model_signature(
                    "models/mango_yolo.pt"
                )
            )
        )

    def recv(
        self,
        frame: av.VideoFrame,
    ) -> av.VideoFrame:

        # Browser frame → OpenCV BGR.
        image = frame.to_ndarray(
            format="bgr24"
        )

        # Run YOLO.
        annotated, detections = (
            self.detector.annotate(
                image
            )
        )

        latest_crop = None
        latest_detection = None

        # If multiple mangoes exist,
        # use highest-confidence detection.
        if detections:

            best_detection = max(
                detections,
                key=lambda item:
                item["confidence"],
            )

            latest_detection = (
                best_detection.copy()
            )

            crop = best_detection.get(
                "crop"
            )

            if crop is not None:
                latest_crop = (
                    crop.copy()
                )

        # Store latest raw camera frame.
        with self.lock:

            self.latest_frame = (
                image.copy()
            )

            self.latest_crop = (
                latest_crop
            )

            self.latest_detection = (
                latest_detection
            )

        # Return image with bounding boxes.
        return (
            av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24",
            )
        )

    def get_latest_detection(
        self,
    ):
        """
        Safely return the latest detected
        mango and corresponding camera frame.
        """

        with self.lock:

            frame = (
                self.latest_frame.copy()
                if self.latest_frame
                is not None
                else None
            )

            crop = (
                self.latest_crop.copy()
                if self.latest_crop
                is not None
                else None
            )

            detection = (
                self.latest_detection.copy()
                if self.latest_detection
                is not None
                else None
            )

        return (
            frame,
            crop,
            detection,
        )


def render_live_yolo_camera():
    """
    Display live YOLO camera.

    Returns a dictionary when the user clicks
    Analyse detected mango:

    {
        "image": full_camera_frame,
        "crop": mango_crop,
        "confidence": confidence
    }

    Otherwise returns None.
    """

    st.markdown(
        "### Live mango detection"
    )

    st.caption(
        "Point the camera at a mango. "
        "YOLO will detect and track the "
        "mango in real time."
    )

    # ========================================================
    # Start browser camera
    # ========================================================

    try:

        ctx = webrtc_streamer(
            key=(
                "mango-yolo-live-camera"
            ),

            mode=(
                WebRtcMode.SENDRECV
            ),

            video_processor_factory=(
                MangoVideoProcessor
            ),

            media_stream_constraints={
                "video": True,
                "audio": False,
            },

            async_processing=True,
        )

    except Exception as exc:

        st.error(
            "Unable to start the "
            f"camera: {exc}"
        )

        return None

    # ========================================================
    # Camera not started
    # ========================================================

    if not ctx.state.playing:

        st.info(
            "Start the camera to begin "
            "live mango detection."
        )

        return None

    st.caption(
        "Keep the mango clearly inside "
        "the detection box."
    )

    # ========================================================
    # One-button live analysis
    # ========================================================

    analyse_live_clicked = (
        st.button(
            "🥭 Analyse detected mango",
            type="primary",
            width="stretch",
            key=(
                "analyse_detected_mango"
            ),
        )
    )

    if not analyse_live_clicked:

        return None

    # ========================================================
    # Get video processor
    # ========================================================

    processor = (
        ctx.video_processor
    )

    if processor is None:

        st.warning(
            "The camera is still starting. "
            "Please wait a moment and "
            "try again."
        )

        return None

    # ========================================================
    # Capture latest detected frame
    # ========================================================

    (
        frame,
        crop,
        detection,
    ) = (
        processor
        .get_latest_detection()
    )

    if frame is None:

        st.warning(
            "No camera frame is "
            "available yet."
        )

        return None

    # ========================================================
    # Mango must be detected
    # ========================================================

    if detection is None:

        st.warning(
            "No mango is currently detected. "
            "Place a mango clearly inside "
            "the camera view and try again."
        )

        return None

    confidence = float(
        detection.get(
            "confidence",
            0.0,
        )
    )

    st.success(
        "Mango detected with "
        f"{confidence * 100:.1f}% "
        "confidence. Starting "
        "full assessment..."
    )

    # IMPORTANT:
    #
    # We return the FULL frame because
    # perform_assessment() already contains
    # the existing mango-object identification
    # and processing pipeline.
    #
    # The crop is also returned for future use.
    return {
        "image": frame.copy(),

        "crop": (
            crop.copy()
            if crop is not None
            else None
        ),

        "confidence": confidence,

        "detection": detection,
    }