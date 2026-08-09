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
        Resize image to the target size.

        INTER_AREA is appropriate for image downsampling.
        """

        return cv2.resize(
            image,
            self.resize,
            interpolation=cv2.INTER_AREA,
        )

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
            30,
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
            30,
            255,
        )

        # --------------------------------------------------------------------
        # Combine masks
        # --------------------------------------------------------------------

        mask = cv2.bitwise_and(
            saturation_mask,
            value_mask,
        )

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
            kernel,
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

        if num_labels > 1:

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

        # ====================================================================
        # STEP 1 — RESIZE
        # ====================================================================

        resized = self.resize_image(
            image
        )

        # ====================================================================
        # STEP 2 — BGR → HSV
        # ====================================================================

        hsv = self.bgr_to_hsv(
            resized
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

        gaussian = self.gaussian_filter(
            hsv
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

        segmented, mask = (
            self.segment_mango(
                enhanced
            )
        )

        # ====================================================================
        # STEP 7 — CONVERT SEGMENTED IMAGE
        # ====================================================================

        segmented_rgb = cv2.cvtColor(
            segmented,
            cv2.COLOR_HSV2RGB,
        )

        # ====================================================================
        # STEP 8 — MANGO PIXELS
        # ====================================================================

        mango_pixels = (
            self.get_mango_pixels(
                enhanced,
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

            "resized": resized,

            "hsv": hsv,

            "hue": h,

            "saturation": s,

            "value": v,

            "gaussian": gaussian,

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

            "segmented": segmented,

            "segmented_rgb": segmented_rgb,

            "mask": mask,

            "mango_pixels": mango_pixels,

            # =================================================================
            # Metadata
            # =================================================================

            "original_size": image.shape,

            "resized_size": resized.shape,

            "kernel": (5, 5),

            "segmentation_kernel": (5, 5),

            "contrast_method": (
                "CLAHE"
                if self.use_clahe
                else "Histogram Equalization"
            ),
        }