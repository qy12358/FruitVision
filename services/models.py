"""services / models for ManGo or Stay."""

from pathlib import Path

import streamlit as st

from modules.blemish_detector import (
    BlemishDetector,
)
from modules.mango_identifier import (
    MangoIdentifier,
)
from modules.preprocessing import (
    ImagePreprocessor,
)
from modules.quality_grader import (
    QualityGrader,
)
from modules.ripeness_classifier import (
    HybridRipenessClassifier,
)
from modules.yolo_mango_detector import (
    YoloMangoDetector,
)


# ============================================================
# Shared processing objects
# ============================================================

preprocessor = ImagePreprocessor()

quality_grader = QualityGrader()

blemish_detector = BlemishDetector(
    min_blemish_area=8,
    boundary_erosion=15,
)


# ============================================================
# Model signature
# ============================================================

def model_signature(
    path: str,
) -> int:
    """
    Return model modification time.

    Streamlit uses this to invalidate the cached
    model automatically after retraining.
    """

    try:

        return Path(
            path
        ).stat().st_mtime_ns

    except OSError:

        return 0


# ============================================================
# Mango identifier
# ============================================================

@st.cache_resource
def load_mango_identifier(
    signature: int = 0,
) -> MangoIdentifier:
    """Load existing mango/non-mango classifier."""

    return MangoIdentifier(
        model_path=(
            "models/"
            "mango_identifier.keras"
        ),
    )


# ============================================================
# Ripeness classifier
# ============================================================

@st.cache_resource
def load_ripeness_classifier(
    signature: int = 0,
) -> HybridRipenessClassifier:
    """Load trained mango ripeness model."""

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


# ============================================================
# YOLO mango detector
# ============================================================

@st.cache_resource
def load_yolo_mango_detector(
    signature: int = 0,
) -> YoloMangoDetector:
    """
    Load trained YOLO mango detector.

    The model is cached so it is NOT loaded
    again for every video frame.
    """

    return YoloMangoDetector(
        model_path=(
            "models/"
            "mango_yolo.pt"
        ),
        confidence_threshold=0.50,
        image_size=640,
    )