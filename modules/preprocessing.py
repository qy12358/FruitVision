import cv2
import numpy as np


class ImagePreprocessor:
    """
    Module 1 — Traditional Image Preprocessing

    Processing pipeline:

        Original Image
             ↓
        Resize
             ↓
        BGR → HSV
             ↓
        Gaussian Filtering
             ↓
        V-channel Contrast Enhancement
             ↓
        Mango Segmentation
             ↓
        Morphological Cleaning
             ↓
        Largest Component Selection
             ↓
        Segmented Mango Image
             ↓
        Module 2

    Main techniques:

        1. Resize
        2. BGR → HSV conversion
        3. Gaussian filtering
        4. CLAHE contrast enhancement
        5. HSV-based segmentation
        6. Morphological opening
        7. Morphological closing
        8. Connected-component analysis

    Important:

        The segmentation thresholds are NOT ripeness thresholds.

        They are only used to identify likely foreground/mango
        pixels and remove background.

        Ripeness classification is performed by Module 2.
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(
        self,
        resize=(224, 224),
        use_clahe=True,
    ):
        """
        Initialize the preprocessing module.

        Args:
            resize:
                Target image size.

            use_clahe:
                Whether to use CLAHE for V-channel
                contrast enhancement.
        """

        self.resize = resize

        self.use_clahe = use_clahe

        # --------------------------------------------------------------------
        # CLAHE configuration
        # --------------------------------------------------------------------
        #
        # clipLimit controls contrast enhancement strength.
        #
        # tileGridSize divides the image into local regions.
        #
        # These are image-processing parameters, NOT ripeness thresholds.
        #
        # --------------------------------------------------------------------

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

    # ========================================================================
    # RESIZE
    # ========================================================================

    def resize_image(
        self,
        image,
    ):
        """
        Resize image to the target size without changing its aspect ratio.

        Images are letterboxed with black pixels.  Directly resizing a
        portrait mango to 224x224 makes it wider or narrower than the real
        fruit, which corrupts contour/shape information and changes the
        visual proportions seen by the classifier.
        """

        target_w, target_h = self.resize
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Input image has an invalid size.")

        scale = min(target_w / width, target_h / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )

        canvas = np.zeros(
            (target_h, target_w, image.shape[2]),
            dtype=image.dtype,
        )
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2
        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
        ] = resized
        return canvas

    # ========================================================================
    # BGR → HSV
    # ========================================================================

    def bgr_to_hsv(
        self,
        image,
    ):
        """
        Convert an OpenCV BGR image to HSV.

        OpenCV HSV ranges:

            H = 0–179
            S = 0–255
            V = 0–255
        """

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

    # ========================================================================
    # GAUSSIAN FILTER
    # ========================================================================

    def gaussian_filter(
        self,
        image,
    ):
        """
        Apply Gaussian blur.

        Purpose:

            - reduce small image noise
            - smooth small intensity variations
            - make segmentation more stable
        """

        return cv2.GaussianBlur(
            image,
            (5, 5),
            0,
        )

    # ========================================================================
    # HISTOGRAM EQUALIZATION
    # ========================================================================

    def histogram_equalization(
        self,
        hsv_image,
    ):
        """
        Apply standard histogram equalization to the V channel.

        H and S are preserved.

        This method is retained as an alternative enhancement
        technique for experiments and comparison.

        It is NOT used together with CLAHE by default.
        """

        h, s, v = cv2.split(
            hsv_image
        )

        v_equalized = cv2.equalizeHist(
            v
        )

        equalized = cv2.merge(
            (
                h,
                s,
                v_equalized,
            )
        )

        return equalized

    # ========================================================================
    # CLAHE CONTRAST ENHANCEMENT
    # ========================================================================

    def clahe_enhancement(
        self,
        hsv_image,
    ):
        """
        Apply CLAHE to the V channel.

        CLAHE:

            Contrast Limited Adaptive Histogram Equalization

        Purpose:

            - improve local contrast
            - handle uneven illumination
            - preserve more local image detail
            - improve robustness under different lighting conditions

        Only the V channel is modified.

        H and S remain unchanged.
        """

        h, s, v = cv2.split(
            hsv_image
        )

        v_enhanced = self.clahe.apply(
            v
        )

        enhanced = cv2.merge(
            (
                h,
                s,
                v_enhanced,
            )
        )

        return enhanced

    # ========================================================================
    # CONTRAST ENHANCEMENT SELECTOR
    # ========================================================================

    def enhance_contrast(
        self,
        hsv_image,
    ):
        """
        Select the V-channel contrast enhancement method.

        Default:

            CLAHE

        Alternative:

            Standard histogram equalization
        """

        if self.use_clahe:

            return self.clahe_enhancement(
                hsv_image
            )

        return self.histogram_equalization(
            hsv_image
        )

    # ========================================================================
    # MANGO SEGMENTATION
    # ========================================================================

    def segment_mango(
        self,
        hsv_image,
        keep_all_components=False,
        source_bgr=None,
    ):
        """
        Segment the likely mango foreground.

        The segmentation uses HSV information.

        The purpose is ONLY to separate the mango from
        the background.

        It does NOT determine:

            ripe
            semi_ripe
            unripe
            rotten

        Those decisions belong to Module 2.

        Returns:

            segmented_hsv:
                HSV image with background removed.

            mask:
                Binary mango mask.
        """

        # --------------------------------------------------------------------
        # Extract HSV channels
        # --------------------------------------------------------------------

        h, s, v = cv2.split(
            hsv_image
        )

        # --------------------------------------------------------------------
        # Saturation mask
        # --------------------------------------------------------------------
        #
        # Very low saturation areas are commonly:
        #
        #   - white background
        #   - gray background
        #   - neutral surfaces
        #
        # This is NOT a ripeness threshold.
        #
        # It is only a foreground candidate threshold.
        #
        # --------------------------------------------------------------------

        saturation_mask = cv2.inRange(
            s,
            35,
            255,
        )

        # --------------------------------------------------------------------
        # Value mask
        # --------------------------------------------------------------------
        #
        # Remove extremely dark pixels.
        #
        # This helps reduce:
        #
        #   - black background
        #   - very dark noise
        #
        # --------------------------------------------------------------------

        value_mask = cv2.inRange(
            v,
            35,
            255,
        )

        # --------------------------------------------------------------------
        # Combine masks
        # --------------------------------------------------------------------

        candidate = cv2.bitwise_and(
            saturation_mask,
            value_mask,
        )

        # Saturation/value alone marks colourful foliage and outdoor
        # backgrounds as foreground.  Use GrabCut at the native resolution
        # when the source image is available, while retaining the HSV
        # candidate as a hard boundary.
        mask = candidate
        # GrabCut is intentionally bounded to moderate images.  The mask is
        # still returned at native resolution, but running five graph-cut
        # iterations on a multi-megapixel upload is unnecessarily slow.
        border_pixels = np.concatenate(
            (
                candidate[0, :], candidate[-1, :],
                candidate[:, 0], candidate[:, -1],
            )
        )
        border_candidate_ratio = float(np.mean(border_pixels > 0))
        if (
            source_bgr is not None
            and source_bgr.shape[:2] == hsv_image.shape[:2]
            and max(source_bgr.shape[:2]) <= 640
            and border_candidate_ratio >= 0.15
        ):
            height, width = candidate.shape[:2]
            grabcut_mask = np.full(
                (height, width), cv2.GC_PR_BGD, dtype=np.uint8
            )
            border = max(2, int(round(min(height, width) * 0.025)))
            grabcut_mask[:border, :] = cv2.GC_BGD
            grabcut_mask[-border:, :] = cv2.GC_BGD
            grabcut_mask[:, :border] = cv2.GC_BGD
            grabcut_mask[:, -border:] = cv2.GC_BGD
            seed_size = max(3, int(round(min(height, width) * 0.025)))
            if seed_size % 2 == 0:
                seed_size += 1
            # Warm hues are useful sure-foreground seeds for yellow/orange
            # mangoes, apples and citrus.  Saturated green seeds support
            # green mangoes but are deliberately not taken from the whole
            # candidate mask, otherwise an outdoor leafy background becomes
            # a single giant foreground seed.
            warm_seed = (
                (((h <= 35) | (h >= 165)) & (s >= 70) & (v >= 45))
                & (candidate > 0)
            )
            green_seed = (
                ((h >= 35) & (h <= 95) & (s >= 75) & (v <= 220))
                & (candidate > 0)
            )
            # Prefer warm fruit pixels because green foliage is a common
            # false foreground.  Fall back to green only when the image has
            # no meaningful warm fruit evidence (e.g. an unripe green mango
            # on a plain background).
            fruit_seed = warm_seed
            if cv2.countNonZero(fruit_seed.astype(np.uint8)) < 0.01 * height * width:
                fruit_seed = green_seed
            fruit_seed = fruit_seed.astype(np.uint8) * 255
            fruit_seed = cv2.morphologyEx(
                fruit_seed,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            fruit_seed = cv2.morphologyEx(
                fruit_seed,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )
            grabcut_mask[fruit_seed == 0] = cv2.GC_PR_BGD
            grabcut_mask[candidate == 0] = cv2.GC_BGD
            grabcut_mask[fruit_seed > 0] = cv2.GC_PR_FGD

            seed_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (seed_size, seed_size)
            )
            sure_foreground = cv2.erode(fruit_seed, seed_kernel)
            if cv2.countNonZero(sure_foreground) == 0:
                sure_foreground = cv2.erode(
                    fruit_seed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                )
            grabcut_mask[sure_foreground > 0] = cv2.GC_FGD

            if np.any(grabcut_mask == cv2.GC_FGD):
                background_model = np.zeros((1, 65), np.float64)
                foreground_model = np.zeros((1, 65), np.float64)
                try:
                    cv2.grabCut(
                        source_bgr,
                        grabcut_mask,
                        None,
                        background_model,
                        foreground_model,
                        5,
                        cv2.GC_INIT_WITH_MASK,
                    )
                    grabcut_foreground = np.where(
                        (grabcut_mask == cv2.GC_FGD)
                        | (grabcut_mask == cv2.GC_PR_FGD),
                        255,
                        0,
                    ).astype(np.uint8)
                    mask = cv2.bitwise_and(grabcut_foreground, candidate)
                    # A graph cut that retains almost the whole frame has
                    # failed to separate a natural background.  In that
                    # case, use the conservative fruit-colour seed instead
                    # of returning a scene-sized foreground mask.
                    if (
                        cv2.countNonZero(mask)
                        > 0.72 * mask.shape[0] * mask.shape[1]
                        and cv2.countNonZero(fruit_seed) > 0
                    ):
                        mask = fruit_seed
                except cv2.error:
                    # Keep the deterministic candidate for degenerate or
                    # very small inputs where GrabCut cannot initialize.
                    mask = candidate

        # ====================================================================
        # MORPHOLOGICAL PROCESSING
        # ====================================================================

        # --------------------------------------------------------------------
        # Create elliptical kernel
        # --------------------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        # --------------------------------------------------------------------
        # Opening
        # --------------------------------------------------------------------
        #
        # Removes small isolated foreground noise.
        #
        # Erosion → removes small objects.
        # Dilation → restores the main object.
        #
        # --------------------------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        # --------------------------------------------------------------------
        # Closing
        # --------------------------------------------------------------------
        #
        # Fills small holes and connects nearby regions.
        #
        # Dilation → closes small gaps.
        # Erosion → restores object boundary.
        #
        # --------------------------------------------------------------------

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        # ====================================================================
        # CONNECTED COMPONENT ANALYSIS
        # ====================================================================

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        if num_labels > 1 and not keep_all_components:

            # --------------------------------------------------------------
            # Ignore label 0 because it represents the background.
            # --------------------------------------------------------------

            largest_label = 1 + np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )

            mask = np.where(
                labels == largest_label,
                255,
                0,
            ).astype(
                np.uint8
            )

        # ====================================================================
        # APPLY MASK
        # ====================================================================

        segmented_hsv = cv2.bitwise_and(
            hsv_image,
            hsv_image,
            mask=mask,
        )

        return (
            segmented_hsv,
            mask,
        )

    # ========================================================================
    # APPLY MASK TO RGB IMAGE
    # ========================================================================

    def apply_mask_to_rgb(
        self,
        rgb_image,
        mask,
    ):
        """
        Apply the segmentation mask to an RGB image.

        Background pixels become black.

        This is useful for visualization and for the
        EfficientNet image branch.
        """

        return cv2.bitwise_and(
            rgb_image,
            rgb_image,
            mask=mask,
        )

    # ========================================================================
    # GET MANGO PIXELS
    # ========================================================================

    def get_mango_pixels(
        self,
        hsv_image,
        mask,
    ):
        """
        Extract HSV pixels belonging only to the segmented mango.

        This is useful for Module 2 HSV statistical features.

        Instead of calculating:

            mean HSV of entire image

        Module 2 can calculate:

            mean HSV of mango pixels only.

        This prevents the background from affecting
        the HSV statistics.
        """

        mango_pixels = hsv_image[
            mask > 0
        ]

        return mango_pixels

    # ========================================================================
    # PREPROCESS
    # ========================================================================

    def preprocess(
        self,
        image,
        keep_all_components=False,
        mask_override=None,
    ):
        """
        Execute the complete Module 1 preprocessing pipeline.

        Pipeline:

            Original Image
                 ↓
            Resize
                 ↓
            BGR → HSV
                 ↓
            Gaussian Filtering
                 ↓
            CLAHE on V channel
                 ↓
            HSV Segmentation
                 ↓
            Morphological Opening
                 ↓
            Morphological Closing
                 ↓
            Largest Connected Component
                 ↓
            Segmented Mango
                 ↓
            Module 2
        """

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR image with 3 channels.")

        # Segment at native resolution.  The model-sized representation is
        # produced after masking, so the contour and the visible result do
        # not depend on a 224x224 thumbnail.
        original = np.ascontiguousarray(image.copy())

        # ====================================================================
        # STEP 1 — MODEL RESIZE (segmentation stays native)
        # ====================================================================

        resized = self.resize_image(
            original
        )

        # ====================================================================
        # STEP 2 — BGR → HSV
        # ====================================================================

        hsv = self.bgr_to_hsv(
            original
        )

        # ====================================================================
        # STEP 3 — ORIGINAL HSV CHANNELS
        # ====================================================================

        h, s, v = cv2.split(
            hsv
        )

        # ====================================================================
        # STEP 4 — GAUSSIAN FILTER
        # ====================================================================

        # Smooth the BGR image before converting to HSV.  Blurring the H
        # channel directly is not ideal because hue is circular (0 and 179
        # are neighbours), while BGR smoothing is well-defined.
        gaussian_bgr = self.gaussian_filter(
            original
        )
        gaussian = self.bgr_to_hsv(
            gaussian_bgr
        )

        # ====================================================================
        # STEP 5 — CONTRAST ENHANCEMENT
        # ====================================================================

        enhanced = self.enhance_contrast(
            gaussian
        )

        # ====================================================================
        # STEP 6 — MANGO SEGMENTATION
        # ====================================================================

        if mask_override is None:
            segmented_for_mask, mask = self.segment_mango(
                enhanced,
                keep_all_components=keep_all_components,
                source_bgr=original,
            )
        else:
            if mask_override.shape[:2] != original.shape[:2]:
                raise ValueError("mask_override must match the input image size.")
            mask = np.where(mask_override > 0, 255, 0).astype(np.uint8)

        # Keep the original colour values for the classifier.  The enhanced
        # HSV image is only used to make the foreground mask.  Applying CLAHE
        # to the ripeness input would change the colour evidence the model is
        # supposed to learn from.
        segmented_hsv = cv2.bitwise_and(
            hsv,
            hsv,
            mask=mask,
        )
        segmented_bgr = cv2.bitwise_and(
            original,
            original,
            mask=mask,
        )
        segmented_rgb = cv2.cvtColor(
            segmented_bgr,
            cv2.COLOR_BGR2RGB,
        )

        model_segmented_bgr = self.resize_image(segmented_bgr)
        model_segmented_hsv = cv2.cvtColor(
            model_segmented_bgr,
            cv2.COLOR_BGR2HSV,
        )
        model_segmented_rgb = cv2.cvtColor(
            model_segmented_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ====================================================================
        # STEP 8 — MANGO PIXELS
        # ====================================================================

        mango_pixels = (
            self.get_mango_pixels(
                hsv,
                mask,
            )
        )

        # ====================================================================
        # RETURN ALL RESULTS
        # ====================================================================

        return {

            # =================================================================
            # Basic preprocessing
            # =================================================================

            "original": original,

            # Neural-network-sized raw image.  The native-resolution fields
            # below are the authoritative segmentation outputs.
            "resized": resized,

            "hsv": hsv,

            "hue": h,

            "saturation": s,

            "value": v,

            "gaussian": gaussian_bgr,
            "gaussian_hsv": gaussian,

            # =================================================================
            # Contrast enhancement
            # =================================================================

            "enhanced": enhanced,

            # Keep this alias so existing Module 2 code can
            # continue using "equalized".
            #
            # This now represents the final enhanced HSV image.
            "equalized": enhanced,

            # =================================================================
            # Segmentation
            # =================================================================

            "segmented": segmented_hsv,

            "segmented_bgr": segmented_bgr,

            "segmented_rgb": segmented_rgb,

            "model_segmented_hsv": model_segmented_hsv,

            "model_segmented_bgr": model_segmented_bgr,

            "model_segmented_rgb": model_segmented_rgb,

            "mask": mask,

            "mango_pixels": mango_pixels,

            # =================================================================
            # Metadata
            # =================================================================

            "original_size": original.shape,

            "resized_size": resized.shape,

            "kernel": (5, 5),

            "segmentation_kernel": (5, 5),

            "contrast_method": (
                "CLAHE"
                if self.use_clahe
                else "Histogram Equalization"
            ),
        }
