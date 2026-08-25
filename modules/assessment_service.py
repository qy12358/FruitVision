"""Shared single/batch mango assessment pipeline."""
from __future__ import annotations

from datetime import datetime
import time
import uuid
import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    return image


def display_label(raw_name: str) -> str:
    return raw_name.replace("_", "-").replace(" ", "-").title()


def create_assessment_id() -> str:
    """Create a unique, sortable ID for one mango assessment."""
    return f"MS-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:6]}"


def create_batch_id() -> str:
    """Create one shared ID for a newly analysed batch."""
    return f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def process_assessment(
    image,
    preprocessor,
    classifier,
    blemish_detector,
    filename="image",
    batch_id="Single",
    assessment_id=None,
):
    """Run the exact same pipeline for single and batch assessments."""
    started = time.perf_counter()
    preprocessing = preprocessor.preprocess(image)
    preprocessing_seconds = time.perf_counter() - started
    blemish = blemish_detector.analyze(preprocessing["resized"], preprocessing["mask"])
    ripeness = classifier.predict(preprocessing["segmented"], mask=preprocessing["mask"])
    return {
        "assessment_id": assessment_id or create_assessment_id(),
        "batch_id": batch_id,
        "filename": filename,
        "predicted_ripeness": display_label(ripeness["prediction"]),
        "raw_ripeness": ripeness["prediction"],
        "confidence": float(ripeness["confidence"]),
        "grade": blemish["grade"],
        "defect_percentage": float(blemish["defect_percentage"]),
        "severity": blemish["severity"],
        "defect_types": blemish["defect_types"],
        "preprocessing": preprocessing,
        "blemish": blemish,
        "ripeness_result": ripeness,
        "processing_info": (
            f"Letterbox 224x224; BGR to HSV; Gaussian filtering; "
            f"{preprocessing['contrast_method']}; mango segmentation; "
            f"EfficientNetB0 + 63-feature fusion; preprocessing "
            f"{preprocessing_seconds:.3f}s; inference "
            f"{ripeness['inference_time_ms']:.1f}ms"
        ),
        "assessed_at": datetime.now(),
    }
