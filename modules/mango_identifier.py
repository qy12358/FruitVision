"""CNN mango/non-mango identity gate used before ripeness classification.

Ripeness is a four-class problem, so a ripeness model must not be asked to
decide whether an arbitrary input is a mango.  This module provides that
separate binary gate.  The trained MobileNetV2 CNN is the required decision
maker; the classical image-processing values are retained as audit evidence.
If the trained gate is unavailable, the module fails closed instead of
guessing from colour or shape.
"""

from pathlib import Path

import cv2
import numpy as np

from modules.preprocessing import ImagePreprocessor


def fast_identity_processed(image: np.ndarray, input_size=(128, 128)) -> dict:
    """Make the lightweight, shared training/inference identity tensor."""
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    background = np.all(rgb >= 245, axis=2) | np.all(rgb <= 10, axis=2)
    mask = (~background).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        padding = max(2, int(round(0.08 * max(width, height))))
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(rgb.shape[1], x + width + padding), min(rgb.shape[0], y + height + padding)
        crop, crop_mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        scale = min(input_size[0] / crop.shape[1], input_size[1] / crop.shape[0])
        new_size = (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale))))
        crop = cv2.resize(crop, new_size, interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask, new_size, interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((*input_size[::-1], 3), dtype=np.uint8)
        canvas_mask = np.zeros(input_size[::-1], dtype=np.uint8)
        ox, oy = (input_size[0] - new_size[0]) // 2, (input_size[1] - new_size[1]) // 2
        canvas[oy:oy + new_size[1], ox:ox + new_size[0]] = crop
        canvas_mask[oy:oy + new_size[1], ox:ox + new_size[0]] = crop_mask
        rgb, mask = canvas, canvas_mask
    segmented_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
    return {"model_segmented_rgb": segmented_rgb, "model_segmented_hsv": cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2HSV)}


def extract_identity_features(processed: dict) -> np.ndarray:
    """Build a compact, model-independent feature vector for identity."""
    rgb = processed["model_segmented_rgb"].astype(np.uint8)
    hsv = processed["model_segmented_hsv"]
    mask = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 0
    if not np.any(mask):
        mask = np.ones(rgb.shape[:2], dtype=bool)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # A small grayscale/edge thumbnail is much faster than HOG while still
    # capturing the silhouette and surface texture needed by the SVM.
    texture = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_AREA).ravel()
    edges = cv2.Canny((gray * 255).astype(np.uint8), 40, 120)
    edge_thumbnail = (
        cv2.resize(edges, (12, 12), interpolation=cv2.INTER_AREA).ravel() / 255.0
    )
    colour = []
    for channel, bins, value_range in (
        (hsv[..., 0], 18, (0, 180)),
        (hsv[..., 1], 16, (0, 256)),
        (hsv[..., 2], 16, (0, 256)),
    ):
        histogram, _ = np.histogram(
            channel[mask], bins=bins, range=value_range
        )
        histogram = histogram.astype(np.float32)
        colour.extend(histogram / max(float(histogram.sum()), 1.0))

    contour = MangoIdentifier._largest_contour(
        (mask.astype(np.uint8) * 255)
    )
    shape = [float(mask.mean())]
    if contour is not None:
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
        x, y, width, height = cv2.boundingRect(contour)
        shape.extend([
            contour_area / hull_area,
            contour_area / max(float(width * height), 1.0),
            4.0 * np.pi * contour_area / (perimeter * perimeter),
            max(width, height) / max(min(width, height), 1),
        ])
    else:
        shape.extend([0.0, 0.0, 0.0, 0.0])
    return np.concatenate([
        texture.astype(np.float32),
        edge_thumbnail.astype(np.float32),
        np.asarray(colour + shape, dtype=np.float32),
    ])


class MangoIdentifier:
    """Detect whether an image contains one mango suitable for analysis."""

    def __init__(
        self,
        model_path: str = "models/mango_identifier.keras",
        input_size: tuple = (128, 128),
        acceptance_threshold: float = 0.75,
        cnn_threshold: float = 0.85,
        allow_legacy_sklearn: bool = False,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.acceptance_threshold = acceptance_threshold
        # The gate is trained from the Harumanis mango dataset and hard
        # non-mango classes. A green/oval object alone must never be enough
        # to call something a mango.
        self.cnn_threshold = cnn_threshold
        self.allow_legacy_sklearn = allow_legacy_sklearn
        self.preprocessor = ImagePreprocessor(resize=input_size)
        self.cnn_preprocessor = ImagePreprocessor(resize=(224, 224))
        self.model = None
        self.model_kind = None

        # Prefer the CNN explicitly. A stale joblib file must never silently
        # override a newly trained CNN just because the old default path is
        # still present.
        keras_candidates = [self.model_path]
        if self.model_path.suffix.lower() != ".keras":
            keras_candidates.insert(0, self.model_path.with_suffix(".keras"))
        for keras_path in keras_candidates:
            if not keras_path.exists():
                continue
            try:
                import tensorflow as tf

                self.model = tf.keras.models.load_model(keras_path, compile=False)
                self.model_kind = "keras"
                self.model_path = keras_path
                break
            except Exception:
                self.model = None

        # A legacy classical classifier is opt-in only. Its colour/shape
        # features are useful audit evidence, but cannot safely identify an
        # arbitrary user-uploaded fruit such as papaya by themselves.
        if self.model is None and self.allow_legacy_sklearn:
            joblib_candidates = [self.model_path]
            if self.model_path.suffix.lower() != ".joblib":
                joblib_candidates.append(self.model_path.with_suffix(".joblib"))
            for joblib_path in joblib_candidates:
                if not joblib_path.exists():
                    continue
                try:
                    import joblib

                    loaded_model = joblib.load(joblib_path)
                    self.model = (
                        loaded_model["classifier"]
                        if isinstance(loaded_model, dict) and "classifier" in loaded_model
                        else loaded_model
                    )
                    self.model_kind = "sklearn"
                    self.model_path = joblib_path
                    break
                except Exception:
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
                "papaya_like": False,
                "apple_like": False,
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
        papaya_like = (
            yellow_ratio >= 0.85
            and warm_hue_median >= 25.0
            and aspect_ratio >= 2.0
            and extent >= 0.60
            and circularity >= 0.55
        )
        apple_like = (
            warm_ratio >= 0.80
            and aspect_ratio <= 2.0
            and extent < 0.55
            and circularity < 0.45
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
            "papaya_like": papaya_like,
            "apple_like": apple_like,
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
            # Tiny detached stems/highlights are not separate fruit objects.
            # Keep substantial components so a real second mango survives,
            # while a small dark/bright fragment in one rotten mango does not
            # become a second mango prediction.
            minimum_area = max(100, int(cv2.countNonZero(binary) * 0.01))
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

        sklearn_probability = None
        if self.model_kind == "sklearn":
            fast_processed = fast_identity_processed(processed["original"], self.input_size)
            features = extract_identity_features(fast_processed).reshape(1, -1)
            prediction = self.model.predict_proba(features)
            sklearn_probability = float(prediction[0, -1])

        if sklearn_probability is not None:
            return sklearn_probability

        if self.model_kind == "keras":
            cnn_processed = self.cnn_preprocessor.preprocess(
                processed["original"], mask_override=processed["mask"]
            )
            cnn_rgb = cnn_processed["model_segmented_rgb"].astype(np.float32)
            segmented_prediction = np.asarray(
                self.model.predict(np.expand_dims(cnn_rgb, axis=0), verbose=0)
            ).reshape(-1)
            raw_bgr = self.cnn_preprocessor.resize_image(processed["original"])
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            raw_prediction = np.asarray(
                self.model.predict(np.expand_dims(raw_rgb, axis=0), verbose=0)
            ).reshape(-1)
            if segmented_prediction.size and raw_prediction.size:
                segmented_probability = float(
                    segmented_prediction[0]
                    if segmented_prediction.size == 1
                    else segmented_prediction[-1]
                )
                raw_probability = float(
                    raw_prediction[0]
                    if raw_prediction.size == 1
                    else raw_prediction[-1]
                )
                return 0.70 * segmented_probability + 0.30 * raw_probability
        return None

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
        scene_object_count = len(object_masks)
        object_areas = [cv2.countNonZero(item) for item in object_masks]
        total_object_area = max(1, sum(object_areas))
        dominant_object_share = max(object_areas) / total_object_area if object_areas else 0.0
        candidate_binary = np.where(
            full["candidate_mask"] > 0,
            255,
            0,
        ).astype(np.uint8)
        _, _, candidate_stats, _ = cv2.connectedComponentsWithStats(
            candidate_binary,
            connectivity=8,
        )
        candidate_foreground_area = max(
            1,
            int(cv2.countNonZero(candidate_binary)),
        )
        candidate_component_count = sum(
            int(area) >= 0.01 * candidate_foreground_area
            for area in candidate_stats[1:, cv2.CC_STAT_AREA]
        )
        candidate_areas = sorted(
            (int(area) for area in candidate_stats[1:, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        substantial_candidate_count = sum(
            area >= 0.15 * candidate_areas[0]
            for area in candidate_areas
        ) if candidate_areas else 0
        # A natural photograph whose foreground has fragmented into many
        # components is usually foliage/background, not a clean mango scene.
        # Two or three components are still allowed for multiple mangoes.
        scene_clutter = (
            (scene_object_count > 3 and dominant_object_share < 0.65)
            or (
                candidate_component_count > 2
                and substantial_candidate_count < 2
                and dominant_object_share < 0.65
            )
        )
        objects = []

        for object_index, object_mask in enumerate(object_masks, start=1):
            object_area = int(cv2.countNonZero(object_mask))
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
                method = "identity CNN unavailable - fail closed"
                # Classical evidence is returned for diagnostics only. It is
                # never allowed to identify an arbitrary uploaded image.
                accepted = False
            else:
                # The CNN is the species decision. Classical evidence adds
                # conservative rejection checks for obvious look-alikes and
                # fragmented scenes; it never rescues a weak CNN prediction.
                score = 0.90 * model_probability + 0.10 * evidence["classical_score"]
                method = "binary dataset gate + HSV/contour evidence"
                accepted = (
                    model_probability >= self.cnn_threshold
                    and score >= self.acceptance_threshold
                    and not evidence["round_orange"]
                    and not evidence["round_red_fruit"]
                    and not evidence["papaya_like"]
                    and not evidence["apple_like"]
                    and not scene_clutter
                )

            rejection_reasons = []
            if model_probability is None:
                rejection_reasons.append(
                    "trained mango CNN is unavailable; species identification "
                    "is disabled for safety"
                )
            if evidence["area_ratio"] < (0.025 if model_probability is None else 0.003):
                rejection_reasons.append("foreground area is too small")
            if model_probability is not None and model_probability < self.cnn_threshold:
                rejection_reasons.append(
                    f"mango model probability is below {self.cnn_threshold:.0%}"
                )
            if score < self.acceptance_threshold:
                rejection_reasons.append(
                    f"combined gate score is below {self.acceptance_threshold:.2f}"
                )
            for flag, label in (
                ("round_orange", "round orange/citrus-like object"),
                ("round_warm_fruit", "round warm-colour object"),
                ("round_red_fruit", "round red-fruit-like object"),
                ("green_leaf_like", "leaf-like foreground shape"),
                ("papaya_like", "papaya-like silhouette"),
                ("apple_like", "apple-like silhouette"),
            ):
                if evidence[flag]:
                    rejection_reasons.append(label)
            if scene_clutter:
                rejection_reasons.append(
                    f"foreground fragmented into {scene_object_count} objects"
                )

            objects.append({
                "object_index": object_index,
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                # Absolute mask area lets the UI choose the most prominent
                # mango when several mango candidates are visible.
                "foreground_area": object_area,
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
                    else "Rejected: " + "; ".join(rejection_reasons)
                ),
                "rejection_reasons": rejection_reasons,
                "scene_object_count": scene_object_count,
                "candidate_component_count": candidate_component_count,
                "substantial_candidate_count": substantial_candidate_count,
                "scene_clutter": scene_clutter,
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
                    "green_leaf_like", "papaya_like", "apple_like",
                )
            },
        }
