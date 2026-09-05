"""services / models for ManGo or Stay."""

from modules.blemish_detector import BlemishDetector
from modules.mango_identifier import MangoIdentifier
from modules.preprocessing import ImagePreprocessor
from modules.quality_grader import QualityGrader
from modules.ripeness_classifier import HybridRipenessClassifier
from pathlib import Path
import streamlit as st

preprocessor = ImagePreprocessor()

quality_grader = QualityGrader()

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

