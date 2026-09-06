import cv2
import numpy as np


class ImagePreprocessor:
    """Prepare native-resolution fruit masks and colour-preserving model inputs.

    Background colours are estimated in Lab space. GrabCut refines compact
    foreground bodies, and thickness-based morphology removes thin stems
    and leaves. HSV is used for seed hints and downstream colour features,
    never as a hard boundary that discards pale or rotten skin.
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
        self, hsv_image, keep_all_components=False, source_bgr=None,
        return_layers=False,
    ):
        """Separate compact fruit bodies using background colour and GrabCut.

        HSV colour is a seed hint only. Pale highlights and dark lesions are
        allowed inside the body. Thin attachments are removed geometrically,
        since green skin and green leaves cannot be separated by hue alone.
        All returned masks match the native input resolution.
        """
        native_h, native_w = hsv_image.shape[:2]
        if native_h == 0 or native_w == 0:
            raise ValueError("Cannot segment an empty image.")
        if source_bgr is None:
            source_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        if source_bgr.shape[:2] != (native_h, native_w):
            raise ValueError("Segmentation source must match the HSV image size.")
        scale = min(1.0, 640.0 / max(native_h, native_w))
        width = max(1, round(native_w * scale))
        height = max(1, round(native_h * scale))
        source = cv2.resize(source_bgr, (width, height), interpolation=cv2.INTER_AREA)
        empty = np.zeros((height, width), np.uint8)
        candidate = empty.copy()
        mask = empty.copy()
        removed = empty.copy()

        if min(height, width) >= 12:
            lab = cv2.cvtColor(
                cv2.GaussianBlur(source, (5, 5), 0), cv2.COLOR_BGR2LAB,
            ).astype(np.float32)
            border_size = max(1, round(min(height, width) * 0.02))
            border = np.zeros((height, width), bool)
            border[:border_size] = True
            border[-border_size:] = True
            border[:, :border_size] = True
            border[:, -border_size:] = True
            samples = np.ascontiguousarray(lab[border][::3])
            cv2.setRNGSeed(0)
            _, _, centres = cv2.kmeans(
                samples, min(5, len(samples)), None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
                3, cv2.KMEANS_PP_CENTERS,
            )
            distance = np.min(np.linalg.norm(
                lab[:, :, None, :] - centres[None, None, :, :], axis=3,
            ), axis=2)
            candidate = np.where(distance > 18, 255, 0).astype(np.uint8)
            candidate[border] = 0
            candidate = cv2.morphologyEx(
                candidate, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(candidate, contours, -1, 255, cv2.FILLED)

            thickness = cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
            radius = max(3, round(float(thickness.max()) * 0.28))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1),
            )
            bodies = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(bodies)
            if count > 1:
                largest = int(stats[1:, cv2.CC_STAT_AREA].max())
                for label in range(1, count):
                    if stats[label, cv2.CC_STAT_AREA] >= max(bodies.size * 0.015, largest * 0.12):
                        mask[labels == label] = 255

            # Probable foreground may include highlights/lesions. Certain
            # foreground uses colour evidence to avoid forcing neutral
            # shadows or background patches into the final fruit mask.
            graph_mask = np.full(mask.shape, cv2.GC_PR_BGD, np.uint8)
            graph_mask[mask > 0] = cv2.GC_PR_FGD
            sure = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            hue, saturation, _ = cv2.split(cv2.cvtColor(source, cv2.COLOR_BGR2HSV))
            blue, green, red = cv2.split(source.astype(np.int16))
            colour = (
                ((saturation >= 35) & (green - blue >= 15) & (hue >= 20) & (hue <= 100))
                | ((saturation >= 55) & ((hue <= 25) | (hue >= 165)))
            )
            colour_seed = np.where(colour, sure, 0).astype(np.uint8)
            if np.any(colour_seed):
                sure = colour_seed
            graph_mask[sure > 0] = cv2.GC_FGD
            graph_mask[border] = cv2.GC_BGD
            if np.any(graph_mask == cv2.GC_FGD):
                try:
                    cv2.grabCut(
                        source, graph_mask, None, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK,
                    )
                    mask = np.where(
                        (graph_mask == cv2.GC_FGD) | (graph_mask == cv2.GC_PR_FGD), 255, 0,
                    ).astype(np.uint8)
                except cv2.error:
                    # Preserve the compact body candidate if graph fitting
                    # cannot initialize on a uniform or degenerate image.
                    pass
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            removed = cv2.subtract(mask, opened)
            # Seal narrow cuts left by specular highlights or lesions after
            # removing thin attachments. This does not reconnect detached
            # leaves because the opening has already removed them.
            # Closing dilates before eroding. Give the dilation room outside
            # the image; otherwise a constant-zero border cuts a radius-wide
            # strip off foreground near the frame (flat tops/ends on fruit).
            padded = cv2.copyMakeBorder(
                opened, radius, radius, radius, radius,
                cv2.BORDER_CONSTANT, value=0,
            )
            mask = cv2.morphologyEx(
                padded, cv2.MORPH_CLOSE, kernel,
                borderType=cv2.BORDER_CONSTANT, borderValue=0,
            )[radius:-radius, radius:-radius]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            clean = np.zeros_like(mask)
            if count > 1:
                largest = int(stats[1:, cv2.CC_STAT_AREA].max())
                for label in range(1, count):
                    x, y, body_w, body_h, area = stats[label]
                    opposite_edges = (
                        (x <= border_size and x + body_w >= width - border_size)
                        or (y <= border_size and y + body_h >= height - border_size)
                    )
                    if (
                        area >= max(mask.size * 0.015, largest * 0.12)
                        and max(body_w, body_h) / max(1, min(body_w, body_h)) < 3.2
                        and not opposite_edges
                    ):
                        clean[labels == label] = 255
            mask = clean
            if not keep_all_components and np.any(mask):
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        mask = cv2.resize(mask, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        candidate = cv2.resize(candidate, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        removed = cv2.resize(removed, (native_w, native_h), interpolation=cv2.INTER_NEAREST)
        removed = cv2.bitwise_and(removed, cv2.bitwise_not(mask))
        segmented = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        if return_layers:
            return segmented, mask, candidate, removed
        return segmented, mask

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
        """Segment fruit bodies and return native masks plus letterboxed inputs.

        Gaussian/contrast images remain available for diagnostics. The
        segmentation uses the source BGR image, and both classification and
        defect analysis retain the original fruit colour values.
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
            segmented_for_mask, mask, candidate_mask, leaf_mask = self.segment_mango(
                enhanced,
                keep_all_components=keep_all_components,
                source_bgr=original,
                return_layers=True,
            )
        else:
            if mask_override.shape[:2] != original.shape[:2]:
                raise ValueError("mask_override must match the input image size.")
            mask = np.where(mask_override > 0, 255, 0).astype(np.uint8)
            candidate_mask = mask.copy()
            leaf_mask = np.zeros_like(mask)

        # Keep the original colour values for the classifier.  The enhanced
        # HSV image is retained for diagnostics only. Applying CLAHE
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

            # Layered segmentation outputs.  All masks remain at the native
            # input resolution; only the model input is letterboxed to 224px.
            "candidate_mask": candidate_mask,
            "leaf_mask": leaf_mask,
            "fruit_mask": mask,

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
