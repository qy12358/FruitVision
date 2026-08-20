"""
FruitVision AI
Module 3 — Adaptive Mango Blemish & Damage Detection

Detects:
    - Small dark spots
    - Large dark spots
    - Brown lesions
    - Bruises
    - Rot-like regions
    - Scratches
    - Surface damage
    - Large damaged / rotten areas

Designed for:
    - Random mango images
    - Different mango sizes
    - Different lighting conditions
    - Green mangoes
    - Yellow mangoes
    - Ripe mangoes
    - Pale mangoes
    - Small blemishes
    - Large blemishes
    - Patchy damage
    - Multiple damaged regions

Attempts to ignore:
    - Background
    - Mango boundary
    - Strong shadows
    - Specular highlights
    - Normal pale lenticels
    - Tiny image noise
    - Natural colour variation

Input:
    image      : OpenCV BGR image
    mango_mask : Binary mango segmentation mask

Output:
    Dictionary compatible with app.py
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class BlemishDetector:

    def __init__(
        self,
        min_blemish_area=8,
        boundary_erosion=10,
    ):
        """
        Parameters
        ----------
        min_blemish_area : int
            Minimum defect area in pixels.

        boundary_erosion : int
            Distance from mango boundary to ignore.
        """

        self.min_blemish_area = max(2, int(min_blemish_area))
        self.boundary_erosion = max(1, int(boundary_erosion))

    # ==============================================================
    # MAIN ANALYSIS
    # ==============================================================

    def analyze(
        self,
        image: np.ndarray,
        mango_mask: np.ndarray
    ) -> Dict:

        if image is None:
            raise ValueError("Input image is None.")

        if mango_mask is None:
            raise ValueError("Mango mask is None.")

        if image.size == 0:
            raise ValueError("Input image is empty.")

        if mango_mask.size == 0:
            raise ValueError("Mango mask is empty.")

        # ----------------------------------------------------------
        # 1. Resize image to segmentation mask
        # ----------------------------------------------------------

        mask_h, mask_w = mango_mask.shape[:2]

        resized = cv2.resize(
            image,
            (mask_w, mask_h),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------------
        # 2. Make mask binary
        # ----------------------------------------------------------

        mask = np.where(
            mango_mask > 0,
            255,
            0
        ).astype(np.uint8)

        # Remove tiny segmentation noise
        mask = self._clean_mango_mask(mask)

        # ----------------------------------------------------------
        # 3. Validate mango mask
        # ----------------------------------------------------------

        original_mango_area = cv2.countNonZero(mask)

        if original_mango_area == 0:
            return self._empty_result(mask)

        # ----------------------------------------------------------
        # 4. Remove mango boundary
        # ----------------------------------------------------------

        safe_mask = self._create_safe_mask(mask)

        safe_area = cv2.countNonZero(safe_mask)

        if safe_area == 0:
            # If erosion removed everything, fall back to original mask
            safe_mask = mask.copy()
            safe_area = original_mango_area

        # ----------------------------------------------------------
        # 5. Colour spaces
        # ----------------------------------------------------------

        hsv = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2HSV
        )

        lab = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2LAB
        )

        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        H, S, V = cv2.split(hsv)
        L, A, B = cv2.split(lab)

        # ----------------------------------------------------------
        # 6. Illumination normalization
        #
        # Important:
        # A shadow should not automatically become a defect.
        #
        # We calculate local brightness differences instead of
        # relying only on absolute brightness.
        # ----------------------------------------------------------

        gray_float = gray.astype(np.float32)

        local_mean_small = cv2.GaussianBlur(
            gray_float,
            (0, 0),
            sigmaX=5
        )

        local_mean_large = cv2.GaussianBlur(
            gray_float,
            (0, 0),
            sigmaX=15
        )

        darkness_small = (
            local_mean_small - gray_float
        )

        darkness_large = (
            local_mean_large - gray_float
        )

        # ----------------------------------------------------------
        # 7. Multi-scale blackhat
        #
        # Small kernel -> small blemishes
        # Large kernel -> large blemishes
        # ----------------------------------------------------------

        blackhat = self._multi_scale_blackhat(gray)

        # ----------------------------------------------------------
        # 8. Adaptive statistics from the mango itself
        # ----------------------------------------------------------

        fruit_pixels = safe_mask > 0

        fruit_v = V[fruit_pixels]
        fruit_s = S[fruit_pixels]
        fruit_l = L[fruit_pixels]
        fruit_blackhat = blackhat[fruit_pixels]

        if len(fruit_v) == 0:
            return self._empty_result(mask)

        # Robust statistics
        v_median = float(np.median(fruit_v))
        v_p10 = float(np.percentile(fruit_v, 10))
        v_p20 = float(np.percentile(fruit_v, 20))
        v_p80 = float(np.percentile(fruit_v, 80))

        s_median = float(np.median(fruit_s))

        bh_p85 = float(np.percentile(fruit_blackhat, 85))
        bh_p90 = float(np.percentile(fruit_blackhat, 90))
        bh_p95 = float(np.percentile(fruit_blackhat, 95))

        # Adaptive thresholds
        blackhat_threshold_small = max(
            8.0,
            bh_p90
        )

        blackhat_threshold_large = max(
            12.0,
            bh_p85
        )

        # ----------------------------------------------------------
        # 9. DETECTOR A
        #
        # Very dark / black spots
        # ----------------------------------------------------------

        dark_threshold = min(
            110,
            max(
                45,
                v_p20 - 12
            )
        )

        dark_spots = (
            (V < dark_threshold)
            &
            (
                blackhat >
                blackhat_threshold_small * 0.65
            )
            &
            (
                darkness_small > 2
            )
        )

        # ----------------------------------------------------------
        # 10. DETECTOR B
        #
        # Adaptive dark regions
        #
        # Helps when the mango is generally bright but a region
        # becomes significantly darker than its surroundings.
        # ----------------------------------------------------------

        adaptive_dark = (
            (darkness_large > 10)
            &
            (blackhat > blackhat_threshold_large * 0.50)
            &
            (V < max(175, v_median))
        )

        # ----------------------------------------------------------
        # 11. DETECTOR C
        #
        # Brown / orange / bruised regions
        #
        # Uses:
        # BGR relationships
        # HSV saturation
        # LAB colour difference
        # ----------------------------------------------------------

        blue = resized[:, :, 0].astype(np.int16)
        green = resized[:, :, 1].astype(np.int16)
        red = resized[:, :, 2].astype(np.int16)

        brown_red_advantage = red - green
        brown_green_advantage = green - blue

        brown_condition_1 = (
            brown_red_advantage > 2
        )

        brown_condition_2 = (
            brown_green_advantage > -5
        )

        brown_condition_3 = (
            A.astype(np.int16) > 128
        )

        brown_condition_4 = (
            B.astype(np.int16) > 120
        )

        brown_condition_5 = (
            S > 35
        )

        brown_condition_6 = (
            V < 220
        )

        brown_condition_7 = (
            darkness_small > 1
        )

        brown_lesions = (
            brown_condition_1
            &
            brown_condition_2
            &
            brown_condition_3
            &
            brown_condition_4
            &
            brown_condition_5
            &
            brown_condition_6
            &
            brown_condition_7
        )

        # ----------------------------------------------------------
        # 12. DETECTOR D
        #
        # Strong brown / rotten patches
        #
        # More aggressive than the normal brown detector.
        # ----------------------------------------------------------

        strong_brown = (
            (A.astype(np.int16) > 132)
            &
            (B.astype(np.int16) > 125)
            &
            (S > 45)
            &
            (V < 205)
            &
            (
                blackhat >
                blackhat_threshold_large * 0.45
            )
        )

        # ----------------------------------------------------------
        # 13. DETECTOR E
        #
        # Very dark severe damage / rot
        # ----------------------------------------------------------

        severe_dark = (
            (V < 85)
            &
            (S > 15)
            &
            (
                darkness_large > 7
            )
        )

        # ----------------------------------------------------------
        # 14. DETECTOR F
        #
        # Wet / decaying regions
        #
        # Rot does not always appear black.
        # Some rotten areas are grey/brown and relatively bright.
        # ----------------------------------------------------------

        wet_rot = (
            (V > 55)
            &
            (V < 190)
            &
            (S < 125)
            &
            (darkness_large > 12)
            &
            (
                blackhat >
                blackhat_threshold_large * 0.70
            )
        )

        # ----------------------------------------------------------
        # 15. DETECTOR G
        #
        # Local contrast anomaly
        #
        # Detects regions whose brightness differs strongly from
        # their surrounding mango area.
        # ----------------------------------------------------------

        local_contrast = np.abs(
            gray_float - local_mean_large
        )

        contrast_threshold = max(
            10,
            float(np.percentile(
                local_contrast[fruit_pixels],
                88
            ))
        )

        local_anomaly = (
            (local_contrast > contrast_threshold)
            &
            (
                blackhat >
                blackhat_threshold_small * 0.40
            )
        )

        # ----------------------------------------------------------
        # 16. COMBINE DETECTORS
        # ----------------------------------------------------------

        candidate = (
            dark_spots
            |
            adaptive_dark
            |
            brown_lesions
            |
            strong_brown
            |
            severe_dark
            |
            wet_rot
            |
            local_anomaly
        )

        candidate &= fruit_pixels

        # ----------------------------------------------------------
        # 17. REMOVE SPECULAR HIGHLIGHTS
        #
        # Very bright + low saturation regions are normally glare.
        # ----------------------------------------------------------

        specular_highlight = (
            (V > 225)
            &
            (S < 75)
        )

        # Only remove if there is no strong evidence of darkness
        safe_highlight_removal = (
            specular_highlight
            &
            (darkness_large < 12)
        )

        candidate &= ~safe_highlight_removal

        # ----------------------------------------------------------
        # 18. REMOVE NORMAL PALE LENTICELS
        #
        # Lenticels are generally:
        # bright
        # small
        # low saturation
        # weak blackhat response
        # ----------------------------------------------------------

        normal_lenticels = (
            (V > max(160, v_p80))
            &
            (S < 95)
            &
            (blackhat < blackhat_threshold_small * 0.65)
            &
            (darkness_small < 5)
        )

        candidate &= ~normal_lenticels

        # ----------------------------------------------------------
        # 19. Remove weak global shadows
        #
        # A large shadow should not automatically be classified as
        # damage.
        #
        # We retain regions having strong local anomaly evidence.
        # ----------------------------------------------------------

        shadow_like = (
            (V < 80)
            &
            (S < 70)
            &
            (blackhat < blackhat_threshold_small * 0.55)
            &
            (darkness_large < 8)
        )

        candidate &= ~shadow_like

        # ----------------------------------------------------------
        # 20. Convert to binary mask
        # ----------------------------------------------------------

        damage_mask = (
            candidate.astype(np.uint8) * 255
        )

        # ----------------------------------------------------------
        # 21. Morphological cleanup
        #
        # IMPORTANT:
        # We use a very small opening so small blemishes are NOT
        # destroyed.
        # ----------------------------------------------------------

        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2)
        )

        damage_mask = cv2.morphologyEx(
            damage_mask,
            cv2.MORPH_OPEN,
            small_kernel,
            iterations=1
        )

        # Fill small gaps
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        damage_mask = cv2.morphologyEx(
            damage_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        # ----------------------------------------------------------
        # 22. Connected component analysis
        # ----------------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                damage_mask,
                connectivity=8
            )
        )

        cleaned_mask = np.zeros_like(damage_mask)

        components = []

        # ----------------------------------------------------------
        # 23. Boundary safety guard
        # ----------------------------------------------------------

        guard_kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        safety_guard = cv2.erode(
            safe_mask,
            guard_kernel,
            iterations=1
        )

        # ----------------------------------------------------------
        # 24. Process every detected region
        # ----------------------------------------------------------

        for i in range(1, num_labels):

            area = int(
                stats[i, cv2.CC_STAT_AREA]
            )

            x = int(
                stats[i, cv2.CC_STAT_LEFT]
            )

            y = int(
                stats[i, cv2.CC_STAT_TOP]
            )

            w = int(
                stats[i, cv2.CC_STAT_WIDTH]
            )

            h = int(
                stats[i, cv2.CC_STAT_HEIGHT]
            )

            # Ignore very tiny noise
            if area < self.min_blemish_area:
                continue

            component_mask = (
                labels == i
            )

            # ------------------------------------------------------
            # Check distance from mango boundary
            # ------------------------------------------------------

            touching_boundary = np.any(
                component_mask &
                (safety_guard == 0)
            )

            if touching_boundary:
                continue

            # ------------------------------------------------------
            # Calculate component properties
            # ------------------------------------------------------

            perimeter = cv2.arcLength(
                cv2.findContours(
                    component_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )[0][0],
                True
            ) if area > 0 else 0

            circularity = (
                (4 * np.pi * area) /
                (perimeter * perimeter)
                if perimeter > 0
                else 0
            )

            aspect_ratio = (
                max(w, h) /
                max(1, min(w, h))
            )

            fill_ratio = (
                area /
                max(1, w * h)
            )

            # ------------------------------------------------------
            # Component pixel statistics
            # ------------------------------------------------------

            pixels = component_mask

            mean_v = float(
                np.mean(V[pixels])
            )

            mean_s = float(
                np.mean(S[pixels])
            )

            mean_h = float(
                np.mean(H[pixels])
            )

            mean_blackhat = float(
                np.mean(blackhat[pixels])
            )

            mean_darkness = float(
                np.mean(darkness_large[pixels])
            )

            mean_a = float(
                np.mean(A[pixels])
            )

            mean_b = float(
                np.mean(B[pixels])
            )

            # ------------------------------------------------------
            # Add component
            # ------------------------------------------------------

            cleaned_mask[component_mask] = 255

            components.append(
                {
                    "area": area,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,

                    "aspect_ratio": round(
                        float(aspect_ratio),
                        2
                    ),

                    "circularity": round(
                        float(circularity),
                        3
                    ),

                    "fill_ratio": round(
                        float(fill_ratio),
                        3
                    ),

                    "mean_h": round(
                        mean_h,
                        2
                    ),

                    "mean_s": round(
                        mean_s,
                        2
                    ),

                    "mean_v": round(
                        mean_v,
                        2
                    ),

                    "mean_blackhat": round(
                        mean_blackhat,
                        2
                    ),

                    "mean_darkness": round(
                        mean_darkness,
                        2
                    ),

                    "mean_lab_a": round(
                        mean_a,
                        2
                    ),

                    "mean_lab_b": round(
                        mean_b,
                        2
                    ),
                }
            )

        # ----------------------------------------------------------
        # 25. Recalculate damage area
        # ----------------------------------------------------------

        mango_area = cv2.countNonZero(
            safe_mask
        )

        damage_area = cv2.countNonZero(
            cleaned_mask
        )

        if mango_area > 0:
            defect_percentage = (
                damage_area /
                mango_area
            ) * 100.0
        else:
            defect_percentage = 0.0

        defect_percentage = min(
            max(defect_percentage, 0.0),
            100.0
        )

        # ----------------------------------------------------------
        # 26. Classify defect types
        # ----------------------------------------------------------

        defect_types = self.classify_defects(
            resized,
            cleaned_mask,
            components
        )

        # ----------------------------------------------------------
        # 27. Severity and quality grade
        #
        # The grading is based on actual damaged surface area.
        # ----------------------------------------------------------

        severity, grade = self._calculate_grade(
            defect_percentage
        )

        # ----------------------------------------------------------
        # 28. Create overlay
        # ----------------------------------------------------------

        overlay = self._create_overlay(
            resized,
            cleaned_mask,
            components
        )

        # ----------------------------------------------------------
        # 29. Return result
        # ----------------------------------------------------------

        return {
            "defect_percentage": round(
                float(defect_percentage),
                2
            ),

            "damage_percentage": round(
                float(defect_percentage),
                2
            ),

            "blemish_pixel_area": int(
                damage_area
            ),

            "mango_pixel_area": int(
                mango_area
            ),

            "severity": severity,

            "grade": grade,

            "defect_types": defect_types,

            "component_count": len(
                components
            ),

            "components": components,

            "damage_mask": cleaned_mask,

            "overlay": overlay,

            "safe_mango_mask": safe_mask,

            "blackhat": blackhat,
        }

    # ==============================================================
    # MANGO MASK CLEANING
    # ==============================================================

    def _clean_mango_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        # Keep largest connected object
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                cleaned,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return cleaned

        largest_label = 1
        largest_area = stats[
            1,
            cv2.CC_STAT_AREA
        ]

        for i in range(2, num_labels):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area > largest_area:
                largest_area = area
                largest_label = i

        result = np.zeros_like(cleaned)

        result[
            labels == largest_label
        ] = 255

        return result

    # ==============================================================
    # SAFE MANGO MASK
    # ==============================================================

    def _create_safe_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:

        h, w = mask.shape

        # Adaptive boundary erosion
        #
        # Do not use a huge kernel on a small mango.
        mango_area = cv2.countNonZero(mask)

        mango_radius = max(
            1,
            int(
                np.sqrt(mango_area / np.pi)
            )
        )

        erosion_pixels = min(
            self.boundary_erosion,
            max(
                1,
                int(mango_radius * 0.06)
            )
        )

        kernel_size = (
            erosion_pixels * 2 + 1
        )

        kernel_size = max(
            3,
            min(kernel_size, 21)
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size
            )
        )

        safe_mask = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        return safe_mask

    # ==============================================================
    # MULTI-SCALE BLACKHAT
    # ==============================================================

    def _multi_scale_blackhat(
        self,
        gray: np.ndarray
    ) -> np.ndarray:

        blackhat = np.zeros_like(
            gray
        )

        # Small -> medium -> large blemishes
        kernel_sizes = [
            5,
            7,
            11,
            15,
            21,
            31
        ]

        for size in kernel_sizes:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (size, size)
            )

            bh = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            blackhat = np.maximum(
                blackhat,
                bh
            )

        return blackhat

    # ==============================================================
    # DEFECT CLASSIFICATION
    # ==============================================================

    def classify_defects(
        self,
        image: np.ndarray,
        damage_mask: np.ndarray,
        components: List[Dict]
    ) -> List[str]:

        if not components:
            return ["None"]

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        H, S, V = cv2.split(hsv)
        L, A, B = cv2.split(lab)

        scratch_found = False
        rot_found = False
        bruise_found = False
        dark_spot_found = False
        surface_found = False

        for comp in components:

            x = comp["x"]
            y = comp["y"]
            w = comp["width"]
            h = comp["height"]
            area = comp["area"]

            aspect_ratio = comp[
                "aspect_ratio"
            ]

            circularity = comp[
                "circularity"
            ]

            mean_v = comp[
                "mean_v"
            ]

            mean_s = comp[
                "mean_s"
            ]

            mean_a = comp[
                "mean_lab_a"
            ]

            mean_b = comp[
                "mean_lab_b"
            ]

            mean_darkness = comp[
                "mean_darkness"
            ]

            mean_blackhat = comp[
                "mean_blackhat"
            ]

            # ------------------------------------------------------
            # Scratch detection
            # ------------------------------------------------------

            if (
                aspect_ratio >= 4.0
                and area >= 10
            ):
                scratch_found = True

            # Also detect thinner scratches
            elif (
                aspect_ratio >= 3.0
                and area >= 15
                and circularity < 0.45
            ):
                scratch_found = True

            # ------------------------------------------------------
            # Very dark region
            # ------------------------------------------------------

            if (
                mean_v < 75
                and mean_darkness > 5
            ):
                dark_spot_found = True

            # ------------------------------------------------------
            # Rot detection
            # ------------------------------------------------------

            if (
                mean_v < 95
                and mean_darkness > 8
                and mean_blackhat > 8
            ):
                rot_found = True

            # Large rotten regions
            if (
                area > 300
                and mean_darkness > 10
                and mean_v < 150
            ):
                rot_found = True

            # ------------------------------------------------------
            # Bruise / brown lesion
            # ------------------------------------------------------

            brown_like = (
                mean_a > 128
                and mean_b > 120
                and mean_s > 30
            )

            if brown_like:
                bruise_found = True

            # ------------------------------------------------------
            # Generic surface defect
            # ------------------------------------------------------

            if (
                area >= self.min_blemish_area
            ):
                surface_found = True

        defect_types = []

        if scratch_found:
            defect_types.append(
                "Scratch"
            )

        if rot_found:
            defect_types.append(
                "Rot / Dark Lesion"
            )

        if bruise_found:
            defect_types.append(
                "Bruise / Brown Spot"
            )

        if dark_spot_found and not rot_found:
            defect_types.append(
                "Dark Spot"
            )

        if (
            not defect_types
            and surface_found
        ):
            defect_types.append(
                "Surface Blemish"
            )

        if not defect_types:
            defect_types.append(
                "None"
            )

        return list(
            dict.fromkeys(
                defect_types
            )
        )

    # ==============================================================
    # SEVERITY / GRADE
    # ==============================================================

    def _calculate_grade(
        self,
        defect_percentage: float
    ) -> Tuple[str, str]:

        """
        Surface quality grading:

        < 2%      -> A
        2 - <5%   -> B
        5 - <10%  -> C
        >=10%     -> D
        """

        if defect_percentage < 2:
            return "Low", "A"

        elif defect_percentage < 5:
            return "Medium", "B"

        elif defect_percentage < 10:
            return "Medium", "C"

        else:
            return "High", "D"

    # ==============================================================
    # OVERLAY
    # ==============================================================

    def _create_overlay(
        self,
        image: np.ndarray,
        damage_mask: np.ndarray,
        components: List[Dict]
    ) -> np.ndarray:

        overlay = image.copy()

        # ----------------------------------------------------------
        # Red transparent defect area
        # ----------------------------------------------------------

        red_layer = np.zeros_like(
            image
        )

        red_layer[:, :, 2] = 255

        damage_pixels = (
            damage_mask > 0
        )

        overlay[damage_pixels] = (
            (
                overlay[damage_pixels]
                .astype(np.float32)
                * 0.45
            )
            +
            (
                red_layer[damage_pixels]
                .astype(np.float32)
                * 0.55
            )
        ).astype(np.uint8)

        # ----------------------------------------------------------
        # Draw contours
        # ----------------------------------------------------------

        contours, _ = cv2.findContours(
            damage_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            (0, 0, 255),
            2
        )

        # ----------------------------------------------------------
        # Draw bounding boxes
        # ----------------------------------------------------------

        for index, comp in enumerate(
            components,
            start=1
        ):

            x = comp["x"]
            y = comp["y"]
            w = comp["width"]
            h = comp["height"]

            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                1
            )

            # Component number
            cv2.putText(
                overlay,
                str(index),
                (x, max(12, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        return overlay

    # ==============================================================
    # EMPTY RESULT
    # ==============================================================

    def _empty_result(
        self,
        mask: np.ndarray
    ) -> Dict:

        return {
            "defect_percentage": 0.0,

            "damage_percentage": 0.0,

            "blemish_pixel_area": 0,

            "mango_pixel_area": 0,

            "severity": "Low",

            "grade": "A",

            "defect_types": ["None"],

            "component_count": 0,

            "components": [],

            "damage_mask": np.zeros_like(
                mask
            ),

            "overlay": None,

            "safe_mango_mask": mask,

            "blackhat": np.zeros_like(
                mask
            ),
        }
