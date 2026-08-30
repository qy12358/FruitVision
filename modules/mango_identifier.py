"""Mango identity gate used before ripeness classification.

Ripeness is a four-class problem, so a ripeness model must not be asked to
decide whether an arbitrary input is a mango.  This module provides that
separate gate.  If ``models/mango_identifier.keras`` exists, its probability
is combined with image-processing evidence.  Before the binary model is
trained, the conservative HSV/contour fallback still rejects most obvious
non-mango inputs and prevents a ripeness label from being shown for them.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.preprocessing import ImagePreprocessor


class MangoIdentifier:
    """Detect whether an image contains one mango suitable for analysis."""

    def __init__(
        self,
        model_path: str = "models/mango_identifier.keras",
        input_size: tuple = (224, 224),
        acceptance_threshold: float = 0.55,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.acceptance_threshold = acceptance_threshold
        self.preprocessor = ImagePreprocessor(resize=input_size)
        self.model = None

        # TensorFlow is intentionally optional here.  The classical gate is
        # useful during development and keeps the application usable until
        # the binary identity model has been trained.
        if self.model_path.exists():
            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False,
                )
            except Exception:
                # A broken optional model must not make the app crash.  The
                # fallback below remains conservative and explainable.
                self.model = None

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _classical_evidence(self, image: np.ndarray, processed: dict) -> dict:
        mask = processed["mask"]
        hsv = processed["hsv"]
        image_area = float(mask.shape[0] * mask.shape[1])
        object_area = float(cv2.countNonZero(mask))
        area_ratio = object_area / image_area if image_area else 0.0

        contour = self._largest_contour(mask)
        if contour is None or object_area <= 0:
            return {
                "area_ratio": area_ratio,
                "colour_ratio": 0.0,
                "solidity": 0.0,
                "extent": 0.0,
                "circularity": 0.0,
                "aspect_ratio": 0.0,
                "orange_ratio": 0.0,
                "orange_core_circularity": 0.0,
                "orange_core_aspect_ratio": 0.0,
                "round_orange": False,
                "warm_ratio": 0.0,
                "warm_circularity": 0.0,
                "warm_aspect_ratio": 0.0,
                "round_warm_fruit": False,
                "warm_hue_median": 0.0,
                "red_ratio": 0.0,
                "round_red_fruit": False,
                "yellow_ratio": 0.0,
                "round_yellow_fruit": False,
                "green_ratio": 0.0,
                "green_leaf_like": False,
                "classical_score": 0.0,
            }

        inside = mask > 0
        h, s, v = cv2.split(hsv)
        # Harumanis can be green, yellow-green, yellow, or brown when rotten.
        # Red hue wraps around OpenCV's 0/179 boundary, so include both ends.
        mango_colour = (
            (((h >= 3) & (h <= 105)) | (h >= 165))
            & (s >= 28)
            & (v >= 30)
        )
        colour_ratio = float(np.mean(mango_colour[inside])) if object_area else 0.0
        orange_colour = (
            (h >= 5)
            & (h <= 28)
            & (s >= 80)
            & (v >= 60)
        )
        orange_ratio = float(np.mean(orange_colour[inside])) if object_area else 0.0

        # Use the orange pixels themselves as a second shape cue.  A citrus
        # fruit may have a leaf or stem attached, making the outer contour
        # look less circular.  Its dominant orange core is still usually
        # round, so this catches the orange shown in the UI more reliably.
        orange_mask = np.where(orange_colour & inside, 255, 0).astype(np.uint8)
        orange_mask = cv2.morphologyEx(
            orange_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        orange_core_contour = self._largest_contour(orange_mask)
        orange_core_circularity = 0.0
        orange_core_aspect_ratio = 0.0
        if orange_core_contour is not None:
            core_area = max(float(cv2.contourArea(orange_core_contour)), 1.0)
            core_perimeter = max(float(cv2.arcLength(orange_core_contour, True)), 1.0)
            core_x, core_y, core_width, core_height = cv2.boundingRect(
                orange_core_contour
            )
            orange_core_circularity = float(
                4.0 * np.pi * core_area / (core_perimeter * core_perimeter)
            )
            orange_core_aspect_ratio = max(core_width, core_height) / max(
                min(core_width, core_height), 1
            )

        warm_colour = (
            (((h <= 35) | (h >= 165))
             & (s >= 65)
             & (v >= 45))
        )
        warm_ratio = float(np.mean(warm_colour[inside])) if object_area else 0.0
        warm_hues = h[inside & warm_colour].astype(np.float32)
        warm_hue_median = float(np.median(warm_hues)) if warm_hues.size else 0.0
        red_colour = (((h <= 8) | (h >= 170)) & (s >= 70) & (v >= 45))
        red_ratio = float(np.mean(red_colour[inside])) if object_area else 0.0
        yellow_colour = ((h >= 18) & (h <= 45) & (s >= 65) & (v >= 45))
        yellow_ratio = float(np.mean(yellow_colour[inside])) if object_area else 0.0
        green_colour = ((h >= 35) & (h <= 95) & (s >= 70) & (v >= 35))
        green_ratio = float(np.mean(green_colour[inside])) if object_area else 0.0
        warm_mask = np.where(warm_colour & inside, 255, 0).astype(np.uint8)
        warm_mask = cv2.morphologyEx(
            warm_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        warm_contour = self._largest_contour(warm_mask)
        warm_circularity = 0.0
        warm_aspect_ratio = 0.0
        if warm_contour is not None:
            warm_area = max(float(cv2.contourArea(warm_contour)), 1.0)
            warm_perimeter = max(float(cv2.arcLength(warm_contour, True)), 1.0)
            warm_x, warm_y, warm_width, warm_height = cv2.boundingRect(warm_contour)
            warm_circularity = float(
                4.0 * np.pi * warm_area / (warm_perimeter * warm_perimeter)
            )
            warm_aspect_ratio = max(warm_width, warm_height) / max(
                min(warm_width, warm_height), 1
            )

        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        x, y, width, height = cv2.boundingRect(contour)
        bbox_area = max(float(width * height), 1.0)
        solidity = contour_area / hull_area
        extent = contour_area / bbox_area
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        circularity = float(4.0 * np.pi * contour_area / (perimeter * perimeter))
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        # A citrus-like round orange is a common hard negative.  Harumanis
        # mangoes are generally asymmetric/oblong; do not let colour alone
        # turn a round orange into a mango.  This remains a hard rejection
        # even if the optional CNN is overconfident.
        round_orange = (
            orange_ratio >= 0.75
            and (
                (
                    circularity >= 0.60
                    and aspect_ratio <= 1.35
                )
                or (
                    orange_core_circularity >= 0.68
                    and orange_core_aspect_ratio <= 1.35
                )
            )
            and warm_hue_median <= 18.5
        )
        # Apples and citrus can have a leaf/stem attached, so use the
        # dominant warm-colour core as well as the outer contour.  Harumanis
        # mangoes in this project are asymmetric enough to avoid this rule;
        # the rule is deliberately limited to very round warm objects.
        round_warm_fruit = (
            warm_ratio >= 0.50
            and warm_hue_median <= 18.5
            and (
                (
                    warm_circularity >= 0.80
                    and warm_aspect_ratio <= 1.50
                )
                or (
                    orange_core_circularity >= 0.80
                    and orange_core_aspect_ratio <= 1.50
                )
            )
        )
        round_red_fruit = (
            red_ratio >= 0.12
            and warm_ratio >= 0.35
            and circularity >= 0.45
            and aspect_ratio <= 1.60
        )
        round_yellow_fruit = (
            yellow_ratio >= 0.80
            and warm_hue_median >= 25.0
            and circularity >= 0.50
            and extent >= 0.55
            and aspect_ratio <= 2.05
        )
        green_leaf_like = (
            green_ratio >= 0.55
            and extent < 0.55
            and circularity < 0.65
        )

        # A mango is normally a single, compact, coloured foreground object.
        # These are evidence scores, not ripeness rules.
        area_score = float(np.clip(area_ratio / 0.08, 0.0, 1.0))
        shape_score = float(np.clip((solidity - 0.55) / 0.40, 0.0, 1.0))
        extent_score = float(np.clip((extent - 0.30) / 0.45, 0.0, 1.0))
        classical_score = (
            0.35 * colour_ratio
            + 0.25 * area_score
            + 0.25 * shape_score
            + 0.15 * extent_score
        )
        if round_orange:
            classical_score *= 0.35

        return {
            "area_ratio": area_ratio,
            "colour_ratio": colour_ratio,
            "solidity": solidity,
            "extent": extent,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "orange_ratio": orange_ratio,
            "orange_core_circularity": orange_core_circularity,
            "orange_core_aspect_ratio": orange_core_aspect_ratio,
            "round_orange": round_orange,
            "warm_ratio": warm_ratio,
            "warm_circularity": warm_circularity,
            "warm_aspect_ratio": warm_aspect_ratio,
            "round_warm_fruit": round_warm_fruit,
            "warm_hue_median": warm_hue_median,
            "red_ratio": red_ratio,
            "round_red_fruit": round_red_fruit,
            "yellow_ratio": yellow_ratio,
            "round_yellow_fruit": round_yellow_fruit,
            "green_ratio": green_ratio,
            "green_leaf_like": green_leaf_like,
            "classical_score": float(classical_score),
        }

    @staticmethod
    def _split_object_masks(mask: np.ndarray) -> list[np.ndarray]:
        """Split separate or touching foreground fruits into object masks."""
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if cv2.countNonZero(binary) == 0:
            return []

        # Start with ordinary connected components for clearly separated
        # fruits.  Watershed is used only when a component contains multiple
        # distance-transform peaks, which is the usual touching-fruit case.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        components = []
        for label in range(1, num_labels):
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            if cv2.countNonZero(component) > 0:
                components.append(component)

        if len(components) > 1:
            minimum_area = max(100, int(cv2.countNonZero(binary) * 0.0025))
            filtered = [
                component for component in components
                if cv2.countNonZero(component) >= minimum_area
            ]
            return filtered or [max(components, key=cv2.countNonZero)]

        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        maximum = float(distance.max())
        if maximum <= 0:
            return components
        peaks = np.where(distance >= 0.42 * maximum, 255, 0).astype(np.uint8)
        peak_count, peak_labels, _, _ = cv2.connectedComponentsWithStats(
            peaks,
            connectivity=8,
        )
        if peak_count <= 2:
            return components

        # Keep only the distance-transform peaks as foreground seeds.  All
        # other foreground pixels must remain 0 (unknown) so watershed can
        # assign them to the nearest seed; marking them as 1 would turn them
        # into background and stop the split from working.
        markers = np.zeros_like(peak_labels, dtype=np.int32)
        markers[peak_labels > 0] = peak_labels[peak_labels > 0] + 1
        markers[binary == 0] = 1
        watershed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_image, markers)

        split = []
        minimum_area = max(100, int(cv2.countNonZero(binary) * 0.04))
        for label in range(2, int(markers.max()) + 1):
            object_mask = np.where(markers == label, 255, 0).astype(np.uint8)
            if cv2.countNonZero(object_mask) >= minimum_area:
                split.append(object_mask)
        return split or components

    def _model_probability(self, processed: dict) -> float | None:
        if self.model is None:
            return None

        rgb = processed["segmented_rgb"].astype(np.float32)
        rgb = processed.get("model_segmented_rgb", rgb)
        prediction = np.asarray(
            self.model.predict(np.expand_dims(rgb, axis=0), verbose=0)
        ).reshape(-1)
        if prediction.size == 0:
            return None
        # Binary models may expose either sigmoid(1) or softmax(2).
        if prediction.size == 1:
            return float(prediction[0])
        return float(prediction[-1])

    def identify_objects(
        self,
        image: np.ndarray,
        preprocessed: dict | None = None,
    ) -> list[dict]:
        """Detect and score each foreground fruit independently."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR image with 3 channels.")

        # Keep every connected foreground component for this stage.  The
        # ripeness pipeline may select its largest component, but doing that
        # before identity detection would silently discard a second mango.
        full = preprocessed or self.preprocessor.preprocess(
            image,
            keep_all_components=True,
        )
        analysis_image = full["original"]
        object_masks = self._split_object_masks(full["mask"])
        objects = []

        for object_index, object_mask in enumerate(object_masks, start=1):
            x, y, width, height = cv2.boundingRect(object_mask)
            padding = max(4, int(round(0.08 * max(width, height))))
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(analysis_image.shape[1], x + width + padding)
            y1 = min(analysis_image.shape[0], y + height + padding)
            crop = analysis_image[y0:y1, x0:x1]
            object_crop_mask = object_mask[y0:y1, x0:x1]
            object_processed = self.preprocessor.preprocess(
                crop,
                mask_override=object_crop_mask,
            )
            evidence = self._classical_evidence(crop, object_processed)
            model_probability = self._model_probability(object_processed)

            if model_probability is None:
                score = evidence["classical_score"]
                method = "HSV + contour fallback"
                accepted = (
                    not evidence["round_orange"]
                    and not evidence["round_warm_fruit"]
                    and not evidence["round_red_fruit"]
                    and not evidence["round_yellow_fruit"]
                    and not evidence["green_leaf_like"]
                    and evidence["area_ratio"] >= 0.025
                    and evidence["colour_ratio"] >= 0.35
                    and score >= self.acceptance_threshold
                )
            else:
                score = 0.75 * model_probability + 0.25 * evidence["classical_score"]
                method = "binary CNN + HSV/contour evidence"
                accepted = (
                    evidence["area_ratio"] >= 0.015
                    and model_probability >= 0.50
                    and score >= self.acceptance_threshold
                    # Hard negative: a round, strongly orange object is not
                    # accepted as Harumanis even when the CNN is overconfident.
                    and not evidence["round_orange"]
                    and not evidence["round_warm_fruit"]
                    and not evidence["round_red_fruit"]
                    and not evidence["round_yellow_fruit"]
                    and not evidence["green_leaf_like"]
                )

            objects.append({
                "object_index": object_index,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "crop": crop,
                "processed": object_processed,
                "mask": object_processed["mask"],
                "is_mango": bool(accepted),
                "score": float(score),
                "method": method,
                "model_probability": model_probability,
                "reason": (
                    "Mango detected."
                    if accepted
                    else "Rejected as non-mango."
                ),
                **evidence,
            })
        return objects

    def identify(self, image: np.ndarray, preprocessed: dict | None = None) -> dict:
        """Return an auditable mango/non-mango decision."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR image with 3 channels.")

        objects = self.identify_objects(image, preprocessed=preprocessed)
        accepted_objects = [item for item in objects if item["is_mango"]]
        best = max(objects, key=lambda item: item["score"], default=None)
        if best is None:
            return {
                "is_mango": False,
                "score": 0.0,
                "method": "HSV + contour fallback",
                "model_probability": None,
                "reason": "No foreground object was found.",
                "objects": objects,
            }

        return {
            "is_mango": bool(accepted_objects),
            "score": float(best["score"]),
            "method": best["method"],
            "model_probability": best["model_probability"],
            "reason": (
                "Mango detected."
                if accepted_objects
                else "Input was rejected as non-mango or no clear mango was found."
            ),
            "objects": objects,
            **{
                key: best[key]
                for key in (
                    "area_ratio", "colour_ratio", "solidity", "extent",
                    "circularity", "aspect_ratio", "orange_ratio", "round_orange",
                    "orange_core_circularity", "orange_core_aspect_ratio",
                    "classical_score", "warm_ratio", "warm_circularity",
                    "warm_aspect_ratio", "round_warm_fruit",
                    "warm_hue_median", "red_ratio", "round_red_fruit",
                    "yellow_ratio", "round_yellow_fruit", "green_ratio",
                    "green_leaf_like",
                )
            },
        }
