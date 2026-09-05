"""services / assessment for ManGo or Stay."""

from services.histograms import format_class_label
from services.models import blemish_detector
from services.models import load_mango_identifier
from services.models import load_ripeness_classifier
from services.models import model_signature
from services.models import preprocessor
from services.models import quality_grader
import cv2
import numpy as np
import time

def combine_object_blemish_results(image, objects, analyses):
    """Place per-mango damage results back into the native scene image."""
    height, width = image.shape[:2]
    damage_mask = np.zeros((height, width), dtype=np.uint8)
    safe_mask = np.zeros((height, width), dtype=np.uint8)
    blackhat = np.zeros((height, width), dtype=np.uint8)
    candidate_mask = np.zeros((height, width), dtype=np.uint8)
    component_masks = {
        key: np.zeros((height, width), dtype=np.uint8)
        for key in (
            "dark_spots", "brown_lesions", "severe_dark", "wet_damage",
            "scratch_mask", "white_surface",
        )
    }
    defect_types = []
    component_count = 0

    for mango_object, analysis in zip(objects, analyses):
        x, y, object_width, object_height = mango_object["bbox"]
        y1 = min(height, y + object_height)
        x1 = min(width, x + object_width)
        crop_height = max(0, y1 - y)
        crop_width = max(0, x1 - x)
        if crop_height == 0 or crop_width == 0:
            continue
        crop_slice = np.s_[y:y1, x:x1]
        damage_mask[crop_slice] = np.maximum(
            damage_mask[crop_slice], analysis["damage_mask"][:crop_height, :crop_width]
        )
        safe_mask[crop_slice] = np.maximum(
            safe_mask[crop_slice], analysis["safe_mango_mask"][:crop_height, :crop_width]
        )
        blackhat[crop_slice] = np.maximum(
            blackhat[crop_slice], analysis["blackhat"][:crop_height, :crop_width]
        )
        candidate_mask[crop_slice] = np.maximum(
            candidate_mask[crop_slice], analysis["candidate_mask"][:crop_height, :crop_width]
        )
        for key, canvas in component_masks.items():
            canvas[crop_slice] = np.maximum(
                canvas[crop_slice], analysis[key][:crop_height, :crop_width]
            )
        component_count += analysis["component_count"]
        defect_types.extend(item for item in analysis["defect_types"] if item != "None")

    mango_area = int(cv2.countNonZero(safe_mask))
    blemish_mask = np.zeros((height, width), dtype=np.uint8)
    for key in ("dark_spots", "brown_lesions", "white_surface"):
        blemish_mask = np.maximum(blemish_mask, component_masks[key])
    damage_only_mask = np.zeros((height, width), dtype=np.uint8)
    for key in ("severe_dark", "wet_damage", "scratch_mask"):
        damage_only_mask = np.maximum(damage_only_mask, component_masks[key])

    blemish_area = int(cv2.countNonZero(blemish_mask))
    damage_area = int(cv2.countNonZero(damage_only_mask))
    defect_area = int(cv2.countNonZero(damage_mask))
    defect_percentage = defect_area / mango_area * 100.0 if mango_area else 0.0
    blemish_percentage = blemish_area / mango_area * 100.0 if mango_area else 0.0
    damage_percentage = damage_area / mango_area * 100.0 if mango_area else 0.0
    if defect_percentage < 1.5:
        severity = "Low"
    elif defect_percentage < 5.0:
        severity = "Medium"
    else:
        severity = "High"

    overlay = image.copy()
    overlay[damage_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return {
        "defect_percentage": round(float(defect_percentage), 2),
        "damage_percentage": round(float(damage_percentage), 2),
        "blemish_percentage": round(float(blemish_percentage), 2),
        "blemish_pixel_area": blemish_area,
        "damage_pixel_area": damage_area,
        "defect_pixel_area": defect_area,
        "mango_pixel_area": mango_area,
        "severity": severity,
        "grade": "Unavailable",
        "defect_types": list(dict.fromkeys(defect_types)) or ["None"],
        "component_count": component_count,
        "components": [component for analysis in analyses for component in analysis["components"]],
        "damage_mask": damage_mask,
        "defect_mask": damage_mask,
        "blemish_mask": blemish_mask,
        "damage_only_mask": damage_only_mask,
        "overlay": overlay,
        "safe_mango_mask": safe_mask,
        "blackhat": blackhat,
        "candidate_mask": candidate_mask,
        **component_masks,
    }

def perform_assessment(image: np.ndarray) -> dict:
    """Run the existing preprocessing, identification, blemish and ripeness functions."""
    start = time.time()
    result = preprocessor.preprocess(image, keep_all_components=True)
    preprocessing_time = time.time() - start

    mango_identifier = load_mango_identifier(model_signature("models/mango_identifier.keras"))
    detected_objects = mango_identifier.identify_objects(image, preprocessed=result)
    accepted_objects = [item for item in detected_objects if item["is_mango"]]
    rejected_objects = [item for item in detected_objects if not item["is_mango"]]
    best_gate_score = max((item["score"] for item in detected_objects), default=0.0)
    gate_method = detected_objects[0]["method"] if detected_objects else "HSV + contour fallback"

    if not accepted_objects:
        return {
            "status": "rejected",
            "image": image,
            "result": result,
            "detected_objects": detected_objects,
            "best_gate_score": best_gate_score,
            "gate_method": gate_method,
            "preprocessing_time": preprocessing_time,
        }

    object_blemish_results = [
        blemish_detector.analyze(
            image=mango_object["crop"],
            mango_mask=mango_object["processed"]["mask"],
        )
        for mango_object in accepted_objects
    ]
    blemish_result = combine_object_blemish_results(
        image=result["original"],
        objects=accepted_objects,
        analyses=object_blemish_results,
    )

    defect_pct = blemish_result["defect_percentage"]
    blemish_pct = blemish_result["blemish_percentage"]
    damage_pct = blemish_result["damage_percentage"]
    severity = blemish_result["severity"]
    defect_types = blemish_result["defect_types"]

    mango_pixel_count = int(cv2.countNonZero(result["mask"]))
    total_pixel_count = int(result["mask"].shape[0] * result["mask"].shape[1])

    ripeness = None
    confidence = 0.0
    raw_ripeness = None
    classification_error = None
    probabilities = {}
    hsv_features = {}
    statistical_analysis = {}
    colour_histogram = {}
    feature_count = 0
    inference_time_ms = 0.0
    object_predictions = []

    try:
        classifier = load_ripeness_classifier(model_signature("models/mango_ripeness.keras"))
    except FileNotFoundError as exc:
        classification_error = str(exc)
    else:
        for mango_object in accepted_objects:
            object_result = classifier.predict(
                mango_object["processed"]["segmented"],
                mask=mango_object["processed"]["mask"],
            )
            object_predictions.append((mango_object, object_result))

        if object_predictions:
            ripeness_result = object_predictions[0][1]
            raw_ripeness = ripeness_result["prediction"]
            ripeness = format_class_label(raw_ripeness)
            confidence = float(ripeness_result["confidence"])
            probabilities = ripeness_result["probabilities"]
            hsv_features = ripeness_result["hsv_features"]
            statistical_analysis = ripeness_result["statistical_analysis"]
            colour_histogram = ripeness_result["colour_histogram"]
            feature_count = int(ripeness_result["feature_count"])
            inference_time_ms = float(ripeness_result["inference_time_ms"])
        else:
            classification_error = "The ripeness model did not return a prediction for the detected mango."

    quality_result = quality_grader.grade(
        blemish_coverage=blemish_pct,
        damage_coverage=damage_pct,
        ripeness=ripeness,
    )
    grade = quality_result["grade"]

    return {
        "status": "complete",
        "image": image,
        "result": result,
        "accepted_objects": accepted_objects,
        "rejected_objects": rejected_objects,
        "accepted_count": len(accepted_objects),
        "rejected_count": len(rejected_objects),
        "best_gate_score": best_gate_score,
        "gate_method": gate_method,
        "blemish_result": blemish_result,
        "defect_pct": defect_pct,
        "blemish_pct": blemish_pct,
        "damage_pct": damage_pct,
        "severity": severity,
        "quality_result": quality_result,
        "grade": grade,
        "defect_types": defect_types,
        "mango_pixel_count": mango_pixel_count,
        "total_pixel_count": total_pixel_count,
        "ripeness": ripeness,
        "confidence": confidence,
        "raw_ripeness": raw_ripeness,
        "classification_error": classification_error,
        "probabilities": probabilities,
        "hsv_features": hsv_features,
        "statistical_analysis": statistical_analysis,
        "colour_histogram": colour_histogram,
        "feature_count": feature_count,
        "inference_time_ms": inference_time_ms,
        "object_predictions": object_predictions,
        "preprocessing_time": preprocessing_time,
    }

