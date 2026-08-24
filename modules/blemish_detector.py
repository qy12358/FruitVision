import cv2
import numpy as np
from typing import Dict, List, Tuple


class BlemishDetector:
    """
    FruitVision AI
    Module 3 — Small Mango Blemish & Damage Detection

    FOCUS:
        - Small dark / black spots
        - Small brown blemishes
        - Small surface damage
        - Small scratches

    INTENTIONALLY IGNORES:
        - Large blemishes
        - Large rot regions
        - Large brown lesions
        - Large decay regions
        - Large connected damage
        - Large edge damage

    Overlay:
        - Red detection only
        - No putText()
        - No labels
        - No confidence text
    """

    def __init__(
        self,

        # ------------------------------------------------------------
        # SMALL DEFECT LIMITS
        # ------------------------------------------------------------

        min_blemish_area: int = 3,

        # IMPORTANT:
        # Lower this if you want smaller detections.
        # Increase slightly if too much noise is detected.
        max_blemish_area: int = 700,

        # Maximum bounding-box dimensions for a SMALL defect.
        max_blemish_width: int = 60,
        max_blemish_height: int = 60,

        # A component cannot occupy more than this fraction
        # of the mango safe area.
        max_blemish_fraction: float = 0.025,

        # Distance from mango boundary.
        # Small defects near the boundary are ignored.
        boundary_erosion: int = 7,

        # ------------------------------------------------------------
        # DETECTION SENSITIVITY
        # ------------------------------------------------------------

        dark_threshold_percentile: float = 12.0,
        blackhat_percentile: float = 90.0,

        # Small-scale blackhat only.
        blackhat_sizes: Tuple[int, ...] = (
            3,
            5,
            7,
            9,
            11,
        ),

        # Minimum confidence for a component.
        confidence_threshold: float = 0.22,
    ):

        self.min_blemish_area = max(
            1,
            int(min_blemish_area)
        )

        self.max_blemish_area = max(
            self.min_blemish_area + 1,
            int(max_blemish_area)
        )

        self.max_blemish_width = max(
            5,
            int(max_blemish_width)
        )

        self.max_blemish_height = max(
            5,
            int(max_blemish_height)
        )

        self.max_blemish_fraction = float(
            max_blemish_fraction
        )

        self.boundary_erosion = max(
            1,
            int(boundary_erosion)
        )

        self.dark_threshold_percentile = float(
            dark_threshold_percentile
        )

        self.blackhat_percentile = float(
            blackhat_percentile
        )

        self.blackhat_sizes = tuple(
            int(x)
            for x in blackhat_sizes
            if int(x) >= 3
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

    # ================================================================
    # MAIN ANALYSIS
    # ================================================================

    def analyze(
        self,
        image: np.ndarray,
        mango_mask: np.ndarray
    ) -> Dict:

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if mango_mask is None:
            raise ValueError(
                "Mango mask is None."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must be a BGR image "
                "with shape (H, W, 3)."
            )

        # ------------------------------------------------------------
        # Resize image to mask
        # ------------------------------------------------------------

        mask_h, mask_w = mango_mask.shape[:2]

        if image.shape[:2] != (
            mask_h,
            mask_w
        ):
            image = cv2.resize(
                image,
                (mask_w, mask_h),
                interpolation=cv2.INTER_AREA
            )

        image = image.copy()

        # ------------------------------------------------------------
        # Binary mango mask
        # ------------------------------------------------------------

        mask = np.where(
            mango_mask > 0,
            255,
            0
        ).astype(np.uint8)

        mask = self._clean_mango_mask(mask)

        # ------------------------------------------------------------
        # Safe mango area
        # ------------------------------------------------------------

        safe_mask = self._make_safe_mask(
            mask
        )

        if cv2.countNonZero(safe_mask) == 0:
            return self._empty_result(mask)

        # ------------------------------------------------------------
        # Image preprocessing
        # ------------------------------------------------------------

        # Very light smoothing.
        # Do NOT use aggressive blur because tiny blemishes
        # can disappear.
        denoised = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

        # ------------------------------------------------------------
        # Color spaces
        # ------------------------------------------------------------

        hsv = cv2.cvtColor(
            denoised,
            cv2.COLOR_BGR2HSV
        )

        lab = cv2.cvtColor(
            denoised,
            cv2.COLOR_BGR2LAB
        )

        gray = cv2.cvtColor(
            denoised,
            cv2.COLOR_BGR2GRAY
        )

        H, S, V = cv2.split(hsv)
        L, A, B = cv2.split(lab)

        # ------------------------------------------------------------
        # Local brightness
        # ------------------------------------------------------------

        local_mean = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=5,
            sigmaY=5
        )

        local_mean = np.maximum(
            local_mean.astype(np.float32),
            1.0
        )

        gray_float = gray.astype(
            np.float32
        )

        local_darkness = (
            local_mean -
            gray_float
        )

        normalized_darkness = (
            local_darkness /
            local_mean
        )

        # ------------------------------------------------------------
        # SMALL-SCALE BLACKHAT
        # ------------------------------------------------------------

        blackhat = self._small_scale_blackhat(
            gray
        )

        # ------------------------------------------------------------
        # Mango statistics
        # ------------------------------------------------------------

        fruit_pixels = (
            safe_mask > 0
        )

        fruit_v = V[
            fruit_pixels
        ].astype(np.float32)

        fruit_s = S[
            fruit_pixels
        ].astype(np.float32)

        fruit_a = A[
            fruit_pixels
        ].astype(np.float32)

        fruit_b = B[
            fruit_pixels
        ].astype(np.float32)

        fruit_blackhat = blackhat[
            fruit_pixels
        ].astype(np.float32)

        if fruit_v.size == 0:
            return self._empty_result(mask)

        # ------------------------------------------------------------
        # Adaptive thresholds
        # ------------------------------------------------------------

        v10 = np.percentile(
            fruit_v,
            10
        )

        v20 = np.percentile(
            fruit_v,
            20
        )

        v50 = np.percentile(
            fruit_v,
            50
        )

        s40 = np.percentile(
            fruit_s,
            40
        )

        a50 = np.percentile(
            fruit_a,
            50
        )

        b50 = np.percentile(
            fruit_b,
            50
        )

        bh85 = np.percentile(
            fruit_blackhat,
            85
        )

        bh90 = np.percentile(
            fruit_blackhat,
            self.blackhat_percentile
        )

        # ------------------------------------------------------------
        # 1. SMALL DARK SPOTS
        # ------------------------------------------------------------

        dark_spots = (
            (V <= v10)
            &
            (
                blackhat >= max(
                    10,
                    bh90
                )
            )
            &
            (
                normalized_darkness >= 0.035
            )
        )

        # ------------------------------------------------------------
        # 2. SMALL BROWN BLEMISHES
        # ------------------------------------------------------------

        blue = denoised[:, :, 0].astype(
            np.int16
        )

        green = denoised[:, :, 1].astype(
            np.int16
        )

        red = denoised[:, :, 2].astype(
            np.int16
        )

        red_green = red - green
        green_blue = green - blue

        brown_color = (
            (red_green > 2)
            &
            (green_blue > 0)
        )

        brown_lab = (
            (A.astype(np.int16) >= a50)
            &
            (B.astype(np.int16) >= b50)
        )

        brown_lesions = (
            brown_color
            &
            brown_lab
            &
            (
                S >
                max(
                    28,
                    s40
                )
            )
            &
            (
                (
                    blackhat >
                    max(
                        8,
                        bh85 * 0.65
                    )
                )
                |
                (
                    local_darkness > 4
                )
            )
        )

        # ------------------------------------------------------------
        # 3. SMALL VERY DARK DAMAGE
        # ------------------------------------------------------------

        severe_dark = (
            (V < max(
                55,
                v20 * 0.60
            ))
            &
            (S > 18)
            &
            (normalized_darkness > 0.065)
            &
            (
                blackhat >
                max(
                    10,
                    bh90 * 0.70
                )
            )
        )

        # ------------------------------------------------------------
        # 4. SMALL WET / DARK SURFACE DAMAGE
        # ------------------------------------------------------------

        wet_damage = (
            (V > 50)
            &
            (V < min(
                185,
                v50 + 20
            ))
            &
            (S < 130)
            &
            (normalized_darkness > 0.075)
            &
            (
                blackhat >
                max(
                    10,
                    bh90 * 0.80
                )
            )
        )

        # ------------------------------------------------------------
        # 5. SMALL SCRATCHES
        # ------------------------------------------------------------

        scratch_mask = self._detect_small_scratches(
            gray,
            safe_mask
        )

        # ------------------------------------------------------------
        # 6. VERY SMALL WHITE/FUZZY SURFACE DAMAGE
        # ------------------------------------------------------------

        # Keep this conservative.
        # Large white regions will be removed later by the
        # component-size filter.
        white_surface = (
            (V > 175)
            &
            (S < 65)
            &
            (blackhat > 12)
            &
            (normalized_darkness > 0.025)
        )

        # ------------------------------------------------------------
        # COMBINE
        # ------------------------------------------------------------

        candidate = (
            dark_spots
            |
            brown_lesions
            |
            severe_dark
            |
            wet_damage
            |
            scratch_mask
            |
            white_surface
        )

        candidate &= (
            safe_mask > 0
        )

        # ------------------------------------------------------------
        # Remove obvious highlights
        # ------------------------------------------------------------

        highlights = (
            (V > 225)
            &
            (S < 85)
            &
            (normalized_darkness < 0.05)
        )

        candidate &= ~highlights

        # ------------------------------------------------------------
        # Remove normal pale lenticels
        # ------------------------------------------------------------

        normal_lenticels = (
            (V > max(
                150,
                v50
            ))
            &
            (S < 105)
            &
            (normalized_darkness < 0.05)
            &
            (
                blackhat <
                max(
                    9,
                    bh85 * 0.65
                )
            )
        )

        candidate &= ~normal_lenticels

        # ------------------------------------------------------------
        # Morphological cleaning
        # ------------------------------------------------------------

        damage_mask = (
            candidate.astype(
                np.uint8
            ) * 255
        )

        # Very small opening.
        # Larger opening would destroy tiny blemishes.
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        damage_mask = cv2.morphologyEx(
            damage_mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        # Small closing to connect fragmented pixels.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        damage_mask = cv2.morphologyEx(
            damage_mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        damage_mask[
            safe_mask == 0
        ] = 0

        # ------------------------------------------------------------
        # SMALL COMPONENT FILTER
        # ------------------------------------------------------------

        cleaned_mask, components = (
            self._filter_small_components(
                damage_mask=damage_mask,
                safe_mask=safe_mask,
                hsv=hsv,
                blackhat=blackhat,
                normalized_darkness=normalized_darkness
            )
        )

        # ------------------------------------------------------------
        # Area
        # ------------------------------------------------------------

        mango_area = cv2.countNonZero(
            safe_mask
        )

        damage_area = cv2.countNonZero(
            cleaned_mask
        )

        defect_percentage = (
            damage_area /
            mango_area *
            100.0
            if mango_area > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Classification
        # ------------------------------------------------------------

        defect_types = self.classify_defects(
            image=denoised,
            damage_mask=cleaned_mask,
            components=components
        )

        # ------------------------------------------------------------
        # Grade
        # ------------------------------------------------------------

        severity, grade = (
            self._calculate_grade(
                defect_percentage
            )
        )

        # ------------------------------------------------------------
        # Overlay
        # ------------------------------------------------------------

        overlay = self._create_overlay(
            denoised,
            cleaned_mask
        )

        # ------------------------------------------------------------
        # Return
        # ------------------------------------------------------------

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

            # Debug masks
            "candidate_mask": damage_mask,

            "dark_spots": (
                dark_spots.astype(
                    np.uint8
                ) * 255
            ),

            "brown_lesions": (
                brown_lesions.astype(
                    np.uint8
                ) * 255
            ),

            "severe_dark": (
                severe_dark.astype(
                    np.uint8
                ) * 255
            ),

            "wet_damage": (
                wet_damage.astype(
                    np.uint8
                ) * 255
            ),

            "scratch_mask": (
                scratch_mask.astype(
                    np.uint8
                ) * 255
            ),

            "white_surface": (
                white_surface.astype(
                    np.uint8
                ) * 255
            ),
        }

    # ================================================================
    # CLEAN MANGO MASK
    # ================================================================

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
            kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                cleaned,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return cleaned

        largest = (
            1 +
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )

        result = np.zeros_like(
            mask
        )

        result[
            labels == largest
        ] = 255

        return result

    # ================================================================
    # SAFE MANGO MASK
    # ================================================================

    def _make_safe_mask(
        self,
        mask: np.ndarray
    ) -> np.ndarray:

        erosion = max(
            3,
            int(self.boundary_erosion)
        )

        if erosion % 2 == 0:
            erosion += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion, erosion)
        )

        safe = cv2.erode(
            mask,
            kernel,
            iterations=1
        )

        return safe

    # ================================================================
    # SMALL-SCALE BLACKHAT
    # ================================================================

    def _small_scale_blackhat(
        self,
        gray: np.ndarray
    ) -> np.ndarray:

        blackhat = np.zeros_like(
            gray
        )

        # IMPORTANT:
        # No 15x15 or 21x21.
        #
        # This prevents large blemishes from becoming
        # strong candidates.
        for size in self.blackhat_sizes:

            if size % 2 == 0:
                size += 1

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

    # ================================================================
    # SMALL SCRATCH DETECTION
    # ================================================================

    def _detect_small_scratches(
        self,
        gray: np.ndarray,
        safe_mask: np.ndarray
    ) -> np.ndarray:

        # ------------------------------------------------------------
        # Small horizontal scratch
        # ------------------------------------------------------------

        horizontal_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (9, 3)
            )
        )

        horizontal = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            horizontal_kernel
        )

        # ------------------------------------------------------------
        # Small vertical scratch
        # ------------------------------------------------------------

        vertical_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 9)
            )
        )

        vertical = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            vertical_kernel
        )

        response = np.maximum(
            horizontal,
            vertical
        )

        fruit_response = response[
            safe_mask > 0
        ]

        if fruit_response.size == 0:
            return np.zeros_like(
                gray,
                dtype=bool
            )

        threshold = max(
            20,
            np.percentile(
                fruit_response,
                97
            )
        )

        scratch = (
            (response >= threshold)
            &
            (safe_mask > 0)
        )

        scratch_u8 = (
            scratch.astype(
                np.uint8
            ) * 255
        )

        # Very light cleaning
        scratch_u8 = cv2.morphologyEx(
            scratch_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            )
        )

        return scratch_u8 > 0

    # ================================================================
    # SMALL COMPONENT FILTER
    # ================================================================

    def _filter_small_components(
        self,
        damage_mask: np.ndarray,
        safe_mask: np.ndarray,
        hsv: np.ndarray,
        blackhat: np.ndarray,
        normalized_darkness: np.ndarray
    ) -> Tuple[
        np.ndarray,
        List[Dict]
    ]:

        H, S, V = cv2.split(
            hsv
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                damage_mask,
                connectivity=8
            )
        )

        cleaned = np.zeros_like(
            damage_mask
        )

        components = []

        mango_area = cv2.countNonZero(
            safe_mask
        )

        # ------------------------------------------------------------
        # IMPORTANT:
        # Everything above this area is rejected.
        #
        # There is NO exception for large rot.
        # ------------------------------------------------------------

        for i in range(
            1,
            num_labels
        ):

            area = int(
                stats[
                    i,
                    cv2.CC_STAT_AREA
                ]
            )

            # --------------------------------------------------------
            # Too small
            # --------------------------------------------------------

            if area < self.min_blemish_area:
                continue

            # --------------------------------------------------------
            # TOO LARGE -> ALWAYS REJECT
            # --------------------------------------------------------

            if area > self.max_blemish_area:
                continue

            x = int(
                stats[
                    i,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    i,
                    cv2.CC_STAT_TOP
                ]
            )

            w = int(
                stats[
                    i,
                    cv2.CC_STAT_WIDTH
                ]
            )

            h = int(
                stats[
                    i,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            # --------------------------------------------------------
            # Bounding box size
            # --------------------------------------------------------

            if w > self.max_blemish_width:
                continue

            if h > self.max_blemish_height:
                continue

            # --------------------------------------------------------
            # Mango-relative size
            # --------------------------------------------------------

            if mango_area > 0:

                fraction = (
                    area /
                    mango_area
                )

                if fraction > self.max_blemish_fraction:
                    continue

            # --------------------------------------------------------
            # Component pixels
            # --------------------------------------------------------

            component_pixels = (
                labels == i
            )

            # --------------------------------------------------------
            # Small defects must be away from the mango edge.
            #
            # This prevents background/shadow/edge regions from
            # being mistaken as small damage.
            # --------------------------------------------------------

            if np.any(
                component_pixels &
                (safe_mask == 0)
            ):
                continue

            # --------------------------------------------------------
            # Shape
            # --------------------------------------------------------

            aspect_ratio = (
                max(w, h) /
                max(
                    1,
                    min(w, h)
                )
            )

            bbox_area = max(
                1,
                w * h
            )

            fill_ratio = (
                area /
                bbox_area
            )

            contour = (
                self._component_contour(
                    component_pixels
                )
            )

            perimeter = cv2.arcLength(
                contour,
                True
            )

            circularity = (
                4.0 *
                np.pi *
                area /
                (perimeter ** 2)
                if perimeter > 0
                else 0.0
            )

            # --------------------------------------------------------
            # Pixel statistics
            # --------------------------------------------------------

            mean_v = float(
                np.mean(
                    V[
                        component_pixels
                    ]
                )
            )

            mean_s = float(
                np.mean(
                    S[
                        component_pixels
                    ]
                )
            )

            mean_blackhat = float(
                np.mean(
                    blackhat[
                        component_pixels
                    ]
                )
            )

            mean_darkness = float(
                np.mean(
                    normalized_darkness[
                        component_pixels
                    ]
                )
            )

            # --------------------------------------------------------
            # Confidence
            # --------------------------------------------------------

            confidence = (
                self._component_confidence(
                    area=area,
                    aspect_ratio=aspect_ratio,
                    fill_ratio=fill_ratio,
                    circularity=circularity,
                    mean_v=mean_v,
                    mean_s=mean_s,
                    mean_blackhat=mean_blackhat,
                    mean_darkness=mean_darkness
                )
            )

            if confidence < self.confidence_threshold:
                continue

            # --------------------------------------------------------
            # Keep
            # --------------------------------------------------------

            cleaned[
                component_pixels
            ] = 255

            components.append({

                "area": area,

                "x": x,
                "y": y,
                "width": w,
                "height": h,

                "aspect_ratio": round(
                    float(aspect_ratio),
                    3
                ),

                "fill_ratio": round(
                    float(fill_ratio),
                    3
                ),

                "circularity": round(
                    float(circularity),
                    3
                ),

                "mean_v": round(
                    mean_v,
                    2
                ),

                "mean_s": round(
                    mean_s,
                    2
                ),

                "mean_blackhat": round(
                    mean_blackhat,
                    2
                ),

                "mean_darkness": round(
                    mean_darkness,
                    4
                ),

                "confidence": round(
                    float(confidence),
                    3
                )
            })

        return (
            cleaned,
            components
        )

    # ================================================================
    # COMPONENT CONTOUR
    # ================================================================

    def _component_contour(
        self,
        component_mask: np.ndarray
    ):

        temp = (
            component_mask.astype(
                np.uint8
            ) * 255
        )

        contours, _ = cv2.findContours(
            temp,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.array(
                [[[0, 0]]],
                dtype=np.int32
            )

        return max(
            contours,
            key=cv2.contourArea
        )

    # ================================================================
    # COMPONENT CONFIDENCE
    # ================================================================

    def _component_confidence(
        self,
        area: int,
        aspect_ratio: float,
        fill_ratio: float,
        circularity: float,
        mean_v: float,
        mean_s: float,
        mean_blackhat: float,
        mean_darkness: float
    ) -> float:

        score = 0.0

        # ------------------------------------------------------------
        # Small defects get enough score
        # ------------------------------------------------------------

        if area >= 30:
            score += 0.15

        elif area >= 15:
            score += 0.10

        else:
            score += 0.08

        # ------------------------------------------------------------
        # Darkness
        # ------------------------------------------------------------

        score += min(
            0.28,
            mean_darkness * 2.0
        )

        # ------------------------------------------------------------
        # Blackhat
        # ------------------------------------------------------------

        score += min(
            0.28,
            mean_blackhat / 100.0
        )

        # ------------------------------------------------------------
        # Saturation
        # ------------------------------------------------------------

        if mean_s > 55:
            score += 0.12

        elif mean_s > 35:
            score += 0.07

        # ------------------------------------------------------------
        # Scratch shape
        # ------------------------------------------------------------

        if aspect_ratio > 4:
            score += 0.12

        elif aspect_ratio > 2:
            score += 0.06

        # ------------------------------------------------------------
        # Compact spot
        # ------------------------------------------------------------

        if circularity > 0.20:
            score += 0.05

        if fill_ratio > 0.25:
            score += 0.04

        # ------------------------------------------------------------
        # Bright weak region suppression
        # ------------------------------------------------------------

        if (
            mean_v > 220
            and
            mean_blackhat < 20
        ):
            score *= 0.30

        return float(
            np.clip(
                score,
                0.0,
                1.0
            )
        )

    # ================================================================
    # CLASSIFICATION
    # ================================================================

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

        H, S, V = cv2.split(
            hsv
        )

        scratch_found = False
        dark_found = False
        brown_found = False
        surface_found = False

        for comp in components:

            x = comp["x"]
            y = comp["y"]
            w = comp["width"]
            h = comp["height"]
            area = comp["area"]

            aspect = comp[
                "aspect_ratio"
            ]

            # --------------------------------------------------------
            # Scratch
            # --------------------------------------------------------

            if (
                (
                    aspect >= 4.0
                    or
                    aspect <= 0.25
                )
                and
                area >= 10
            ):
                scratch_found = True
                continue

            # --------------------------------------------------------
            # Component region
            # --------------------------------------------------------

            region = np.zeros_like(
                damage_mask
            )

            region[
                y:y + h,
                x:x + w
            ] = damage_mask[
                y:y + h,
                x:x + w
            ]

            pixels = (
                region > 0
            )

            if not np.any(pixels):
                continue

            actual_v = float(
                np.mean(
                    V[pixels]
                )
            )

            actual_s = float(
                np.mean(
                    S[pixels]
                )
            )

            actual_darkness = (
                comp[
                    "mean_darkness"
                ]
            )

            # --------------------------------------------------------
            # Dark small spot
            # --------------------------------------------------------

            if (
                actual_v < 90
                or
                actual_darkness > 0.13
            ):
                dark_found = True

            # --------------------------------------------------------
            # Brown blemish
            # --------------------------------------------------------

            elif actual_s > 45:
                brown_found = True

            # --------------------------------------------------------
            # Other small surface damage
            # --------------------------------------------------------

            else:
                surface_found = True

        defect_types = []

        if dark_found:
            defect_types.append(
                "Small Dark Spot"
            )

        if brown_found:
            defect_types.append(
                "Small Brown Blemish"
            )

        if scratch_found:
            defect_types.append(
                "Small Scratch"
            )

        if surface_found:
            defect_types.append(
                "Small Surface Damage"
            )

        if not defect_types:
            defect_types.append(
                "Small Surface Blemish"
            )

        return list(
            dict.fromkeys(
                defect_types
            )
        )

    # ================================================================
    # GRADE
    # ================================================================

    def _calculate_grade(
        self,
        defect_percentage: float
    ) -> Tuple[str, str]:

        if defect_percentage < 0.5:

            return "Low", "A"

        elif defect_percentage < 1.5:

            return "Low", "A"

        elif defect_percentage < 3.0:

            return "Medium", "B"

        elif defect_percentage < 5.0:

            return "Medium", "C"

        else:

            return "High", "D"

    # ================================================================
    # OVERLAY
    # ================================================================

    def _create_overlay(
        self,
        image: np.ndarray,
        damage_mask: np.ndarray
    ) -> np.ndarray:

        overlay = image.copy()

        # Red detection.
        overlay[
            damage_mask > 0
        ] = (
            0,
            0,
            255
        )

        result = cv2.addWeighted(
            image,
            0.72,
            overlay,
            0.28,
            0
        )

        # ------------------------------------------------------------
        # Only contours.
        #
        # NO putText()
        # NO labels
        # NO confidence
        # ------------------------------------------------------------

        contours, _ = cv2.findContours(
            damage_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            result,
            contours,
            -1,
            (0, 0, 255),
            1
        )

        return result

    # ================================================================
    # EMPTY RESULT
    # ================================================================

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

            "defect_types": [
                "None"
            ],

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

            "candidate_mask": np.zeros_like(
                mask
            ),

            "dark_spots": np.zeros_like(
                mask
            ),

            "brown_lesions": np.zeros_like(
                mask
            ),

            "severe_dark": np.zeros_like(
                mask
            ),

            "wet_damage": np.zeros_like(
                mask
            ),

            "scratch_mask": np.zeros_like(
                mask
            ),

            "white_surface": np.zeros_like(
                mask
            ),
        }
