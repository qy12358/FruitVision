"""services / models for ManGo or Stay."""

from pathlib import Path

import streamlit as st

from modules.blemish_detector import BlemishDetector
from modules.mango_identifier import MangoIdentifier
from modules.preprocessing import ImagePreprocessor
from modules.quality_grader import QualityGrader
from modules.ripeness_classifier import HybridRipenessClassifier
from modules.yolo_mango_detector import YoloMangoDetector


# ================================================================
# SHARED COMPONENTS
# ================================================================

preprocessor = ImagePreprocessor()

quality_grader = QualityGrader()


# ================================================================
# MODEL SIGNATURE
# ================================================================

def model_signature(
    path: str,
) -> int:
    """
    Return the model modification timestamp.

    Streamlit uses this value to invalidate the cached model
    automatically whenever a model file is replaced.
    """

    try:

        return (
            Path(path)
            .stat()
            .st_mtime_ns
        )

    except OSError:

        return 0


# ================================================================
# MANGO IDENTIFIER
# ================================================================

@st.cache_resource
def load_mango_identifier(
    signature: int = 0,
) -> MangoIdentifier:
    """
    Load the mango identification model once.
    """

    return MangoIdentifier(
        model_path=(
            "models/"
            "mango_identifier.keras"
        ),
    )


# ================================================================
# RIPENESS CLASSIFIER
# ================================================================

@st.cache_resource
def load_ripeness_classifier(
    signature: int = 0,
) -> HybridRipenessClassifier:
    """
    Load the trained mango ripeness classifier once.
    """

    return HybridRipenessClassifier(
        model_path=(
            "models/"
            "mango_ripeness.keras"
        ),

        class_indices_path=(
            "models/"
            "class_indices.json"
        ),
    )


# ================================================================
# LIVE YOLO MANGO DETECTOR
# ================================================================

@st.cache_resource
def load_yolo_mango_detector(
    signature: int = 0,
) -> YoloMangoDetector:
    """
    Load the trained YOLO mango object detector.

    Used by:
        ui/live_yolo_camera.py
    """

    return YoloMangoDetector(
        model_path=(
            "models/"
            "mango_yolo.pt"
        ),

        confidence_threshold=0.50,

        image_size=640,
    )


# ================================================================
# TRAINED YOLO DEFECT SEGMENTATION MODEL
# ================================================================

@st.cache_resource
def load_blemish_detector(
    signature: int = 0,
) -> BlemishDetector:
    """
    Load the trained YOLO mango surface-defect
    segmentation model.

    Current model:
        class 0 = defect
    """

    return BlemishDetector(
        model_path=(
            "models/"
            "mango_defect_seg.pt"
        ),

        confidence_threshold=0.25,

        iou_threshold=0.50,

        image_size=640,

        mask_threshold=0.50,
    )