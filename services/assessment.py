"""services / assessment for ManGo or Stay."""

import time

import cv2
import numpy as np

from services.histograms import (
    format_class_label,
)

from services.models import (
    load_blemish_detector,
    load_mango_identifier,
    load_ripeness_classifier,
    model_signature,
    preprocessor,
    quality_grader,
)


# ================================================================
# COMBINE PER-MANGO DEFECT RESULTS
# ================================================================

def combine_object_blemish_results(
    image,
    objects,
    analyses,
):
    """
    Place YOLO defect-segmentation masks from individual
    mango crops back onto the complete scene image.
    """

    height, width = (
        image.shape[:2]
    )

    defect_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    mango_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    component_count = 0

    components = []

    defect_types = []

    # ============================================================
    # EACH ACCEPTED MANGO
    # ============================================================

    for mango_object, analysis in zip(
        objects,
        analyses,
    ):

        x, y, object_width, object_height = (
            mango_object["bbox"]
        )

        x2 = min(
            width,
            x + object_width,
        )

        y2 = min(
            height,
            y + object_height,
        )

        crop_width = max(
            0,
            x2 - x,
        )

        crop_height = max(
            0,
            y2 - y,
        )

        if (
            crop_width <= 0
            or crop_height <= 0
        ):
            continue

        # ========================================================
        # DEFECT MASK
        # ========================================================

        crop_defect_mask = (
            analysis[
                "defect_mask"
            ]
        )

        if crop_defect_mask.shape[:2] != (
            crop_height,
            crop_width,
        ):

            crop_defect_mask = cv2.resize(
                crop_defect_mask,
                (
                    crop_width,
                    crop_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        scene_slice = np.s_[
            y:y2,
            x:x2,
        ]

        defect_mask[
            scene_slice
        ] = np.maximum(
            defect_mask[
                scene_slice
            ],
            crop_defect_mask[
                :crop_height,
                :crop_width,
            ],
        )

        # ========================================================
        # MANGO MASK
        # ========================================================

        crop_mango_mask = (
            analysis[
                "safe_mango_mask"
            ]
        )

        if crop_mango_mask.shape[:2] != (
            crop_height,
            crop_width,
        ):

            crop_mango_mask = cv2.resize(
                crop_mango_mask,
                (
                    crop_width,
                    crop_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        mango_mask[
            scene_slice
        ] = np.maximum(
            mango_mask[
                scene_slice
            ],
            crop_mango_mask[
                :crop_height,
                :crop_width,
            ],
        )

        # ========================================================
        # COMPONENT DETAILS
        # ========================================================

        component_count += int(
            analysis.get(
                "component_count",
                0,
            )
        )

        for component in analysis.get(
            "components",
            [],
        ):

            copied_component = (
                component.copy()
            )

            # Move crop bounding box coordinates
            # back to scene coordinates.

            bbox = copied_component.get(
                "bbox"
            )

            if bbox is not None:

                bx1, by1, bx2, by2 = (
                    bbox
                )

                copied_component[
                    "bbox"
                ] = (
                    bx1 + x,
                    by1 + y,
                    bx2 + x,
                    by2 + y,
                )

            components.append(
                copied_component
            )

        for defect_type in analysis.get(
            "defect_types",
            [],
        ):

            if defect_type != "None":

                defect_types.append(
                    defect_type
                )

    # ============================================================
    # FINAL PIXEL AREA
    # ============================================================

    mango_pixel_area = int(
        cv2.countNonZero(
            mango_mask
        )
    )

    defect_pixel_area = int(
        cv2.countNonZero(
            defect_mask
        )
    )

    # ============================================================
    # DEFECT COVERAGE
    # ============================================================

    if mango_pixel_area > 0:

        defect_percentage = (
            defect_pixel_area
            / mango_pixel_area
            * 100.0
        )

    else:

        defect_percentage = 0.0

    defect_percentage = max(
        0.0,
        min(
            100.0,
            defect_percentage,
        ),
    )

    # ============================================================
    # CURRENT MODEL HAS ONE CLASS ONLY
    #
    # Therefore:
    #
    # blemish = all detected surface defects
    # damage  = unavailable / 0
    # ============================================================

    blemish_percentage = (
        defect_percentage
    )

    damage_percentage = 0.0

    blemish_pixel_area = (
        defect_pixel_area
    )

    damage_pixel_area = 0

    # ============================================================
    # SEVERITY
    # ============================================================

    if defect_percentage < 1.5:

        severity = "Low"

    elif defect_percentage < 5.0:

        severity = "Medium"

    else:

        severity = "High"

    # ============================================================
    # OVERLAY
    # ============================================================

    red_layer = image.copy()

    red_layer[
        defect_mask > 0
    ] = (
        0,
        0,
        255,
    )

    overlay = cv2.addWeighted(
        image,
        0.72,
        red_layer,
        0.28,
        0,
    )

    contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (
            0,
            0,
            255,
        ),
        1,
    )

    # ============================================================
    # COMPATIBILITY MASKS FOR CURRENT UI
    # ============================================================

    empty_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    return {
        "defect_percentage": round(
            float(
                defect_percentage
            ),
            2,
        ),

        "blemish_percentage": round(
            float(
                blemish_percentage
            ),
            2,
        ),

        "damage_percentage": round(
            float(
                damage_percentage
            ),
            2,
        ),

        "defect_pixel_area": (
            defect_pixel_area
        ),

        "blemish_pixel_area": (
            blemish_pixel_area
        ),

        "damage_pixel_area": (
            damage_pixel_area
        ),

        "mango_pixel_area": (
            mango_pixel_area
        ),

        "severity": severity,

        "grade": "Unavailable",

        "defect_types": (
            list(
                dict.fromkeys(
                    defect_types
                )
            )
            or ["None"]
        ),

        "component_count": (
            component_count
        ),

        "components": components,

        "damage_mask": (
            defect_mask
        ),

        "defect_mask": (
            defect_mask
        ),

        "blemish_mask": (
            defect_mask
        ),

        "damage_only_mask": (
            empty_mask.copy()
        ),

        "safe_mango_mask": (
            mango_mask
        ),

        "candidate_mask": (
            defect_mask.copy()
        ),

        # Old field retained so current UI does not crash.
        "blackhat": (
            empty_mask.copy()
        ),

        "dark_spots": (
            defect_mask.copy()
        ),

        "brown_lesions": (
            empty_mask.copy()
        ),

        "severe_dark": (
            empty_mask.copy()
        ),

        "wet_damage": (
            empty_mask.copy()
        ),

        "scratch_mask": (
            empty_mask.copy()
        ),

        "white_surface": (
            empty_mask.copy()
        ),

        "overlay": overlay,
    }


# ================================================================
# MAIN ASSESSMENT PIPELINE
# ================================================================

def perform_assessment(
    image: np.ndarray,
) -> dict:

    """
    Complete Mango assessment:

    1. Preprocess image
    2. Identify mango objects
    3. YOLO defect segmentation
    4. Ripeness classification
    5. Surface-quality grading
    """

    # ============================================================
    # PREPROCESSING
    # ============================================================

    start = time.time()

    result = preprocessor.preprocess(
        image,
        keep_all_components=True,
    )

    preprocessing_time = (
        time.time()
        - start
    )

    # ============================================================
    # MANGO IDENTIFICATION
    # ============================================================

    mango_identifier = (
        load_mango_identifier(
            model_signature(
                "models/"
                "mango_identifier.keras"
            )
        )
    )

    detected_objects = (
        mango_identifier
        .identify_objects(
            image,
            preprocessed=result,
        )
    )

    accepted_objects = [
        item
        for item in detected_objects
        if item["is_mango"]
    ]

    rejected_objects = [
        item
        for item in detected_objects
        if not item["is_mango"]
    ]

    best_gate_score = max(
        (
            item["score"]
            for item
            in detected_objects
        ),
        default=0.0,
    )

    gate_method = (
        detected_objects[0]["method"]
        if detected_objects
        else "HSV + contour fallback"
    )

    # ============================================================
    # REJECT IF NO MANGO
    # ============================================================

    if not accepted_objects:

        return {
            "status": "rejected",

            "image": image,

            "result": result,

            "detected_objects": (
                detected_objects
            ),

            "best_gate_score": (
                best_gate_score
            ),

            "gate_method": (
                gate_method
            ),

            "preprocessing_time": (
                preprocessing_time
            ),
        }

    # ============================================================
    # LOAD TRAINED YOLO DEFECT MODEL
    # ============================================================

    blemish_detector = (
        load_blemish_detector(
            model_signature(
                "models/"
                "mango_defect_seg.pt"
            )
        )
    )

    # ============================================================
    # YOLO DEFECT SEGMENTATION PER MANGO
    # ============================================================

    object_blemish_results = []

    for mango_object in accepted_objects:

        analysis = (
            blemish_detector.analyze(
                image=(
                    mango_object[
                        "crop"
                    ]
                ),

                mango_mask=(
                    mango_object[
                        "processed"
                    ][
                        "mask"
                    ]
                ),
            )
        )

        object_blemish_results.append(
            analysis
        )

    # ============================================================
    # COMBINE MULTIPLE MANGO RESULTS
    # ============================================================

    blemish_result = (
        combine_object_blemish_results(
            image=result[
                "original"
            ],

            objects=accepted_objects,

            analyses=(
                object_blemish_results
            ),
        )
    )

    defect_pct = float(
        blemish_result[
            "defect_percentage"
        ]
    )

    blemish_pct = float(
        blemish_result[
            "blemish_percentage"
        ]
    )

    damage_pct = float(
        blemish_result[
            "damage_percentage"
        ]
    )

    severity = (
        blemish_result[
            "severity"
        ]
    )

    defect_types = (
        blemish_result[
            "defect_types"
        ]
    )

    # ============================================================
    # MANGO AREA
    # ============================================================

    mango_pixel_count = int(
        cv2.countNonZero(
            result["mask"]
        )
    )

    total_pixel_count = int(
        result["mask"].shape[0]
        * result["mask"].shape[1]
    )

    # ============================================================
    # RIPENESS DEFAULT VALUES
    # ============================================================

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

    # ============================================================
    # LOAD RIPENESS MODEL
    # ============================================================

    try:

        classifier = (
            load_ripeness_classifier(
                model_signature(
                    "models/"
                    "mango_ripeness.keras"
                )
            )
        )

    except FileNotFoundError as exc:

        classification_error = str(
            exc
        )

    else:

        # ========================================================
        # RIPENESS PER MANGO
        # ========================================================

        for mango_object in accepted_objects:

            object_result = (
                classifier.predict(
                    mango_object[
                        "processed"
                    ][
                        "segmented"
                    ],

                    mask=(
                        mango_object[
                            "processed"
                        ][
                            "mask"
                        ]
                    ),
                )
            )

            object_predictions.append(
                (
                    mango_object,
                    object_result,
                )
            )

        # ========================================================
        # PRIMARY RIPENESS RESULT
        # ========================================================

        if object_predictions:

            ripeness_result = (
                object_predictions[0][1]
            )

            raw_ripeness = (
                ripeness_result[
                    "prediction"
                ]
            )

            ripeness = (
                format_class_label(
                    raw_ripeness
                )
            )

            confidence = float(
                ripeness_result[
                    "confidence"
                ]
            )

            probabilities = (
                ripeness_result[
                    "probabilities"
                ]
            )

            hsv_features = (
                ripeness_result[
                    "hsv_features"
                ]
            )

            statistical_analysis = (
                ripeness_result[
                    "statistical_analysis"
                ]
            )

            colour_histogram = (
                ripeness_result[
                    "colour_histogram"
                ]
            )

            feature_count = int(
                ripeness_result[
                    "feature_count"
                ]
            )

            inference_time_ms = float(
                ripeness_result[
                    "inference_time_ms"
                ]
            )

        else:

            classification_error = (
                "The ripeness model did not "
                "return a prediction for "
                "the detected mango."
            )

    # ============================================================
    # QUALITY GRADING
    #
    # Current defect model:
    #
    # blemish coverage = total defect coverage
    # damage coverage = 0
    # ============================================================

    quality_result = (
        quality_grader.grade(
            blemish_coverage=(
                blemish_pct
            ),

            damage_coverage=(
                damage_pct
            ),

            ripeness=ripeness,
        )
    )

    grade = (
        quality_result[
            "grade"
        ]
    )

    # ============================================================
    # FINAL RESULT
    # ============================================================

    return {
        "status": "complete",

        "image": image,

        "result": result,

        "accepted_objects": (
            accepted_objects
        ),

        "rejected_objects": (
            rejected_objects
        ),

        "accepted_count": len(
            accepted_objects
        ),

        "rejected_count": len(
            rejected_objects
        ),

        "best_gate_score": (
            best_gate_score
        ),

        "gate_method": (
            gate_method
        ),

        "blemish_result": (
            blemish_result
        ),

        "defect_pct": (
            defect_pct
        ),

        "blemish_pct": (
            blemish_pct
        ),

        "damage_pct": (
            damage_pct
        ),

        "severity": (
            severity
        ),

        "quality_result": (
            quality_result
        ),

        "grade": (
            grade
        ),

        "defect_types": (
            defect_types
        ),

        "mango_pixel_count": (
            mango_pixel_count
        ),

        "total_pixel_count": (
            total_pixel_count
        ),

        "ripeness": (
            ripeness
        ),

        "confidence": (
            confidence
        ),

        "raw_ripeness": (
            raw_ripeness
        ),

        "classification_error": (
            classification_error
        ),

        "probabilities": (
            probabilities
        ),

        "hsv_features": (
            hsv_features
        ),

        "statistical_analysis": (
            statistical_analysis
        ),

        "colour_histogram": (
            colour_histogram
        ),

        "feature_count": (
            feature_count
        ),

        "inference_time_ms": (
            inference_time_ms
        ),

        "object_predictions": (
            object_predictions
        ),

        "object_blemish_results": (
            object_blemish_results
        ),

        "preprocessing_time": (
            preprocessing_time
        ),
    }