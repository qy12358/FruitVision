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
        return_layers=False,
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
        candidate_mask = candidate.copy()
        mask = candidate.copy()
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
        # Run the graph cut on a bounded proxy image, then restore the result
        # to native resolution.  The old implementation skipped GrabCut for
        # most uploaded photographs and therefore kept attached leaves in the
        # fruit mask.
        if (
            source_bgr is not None
            and source_bgr.shape[:2] == hsv_image.shape[:2]
            and cv2.countNonZero(candidate) >= 0.01 * candidate.size
        ):
            native_height, native_width = candidate.shape[:2]
            proxy_scale = min(
                1.0,
                640.0 / max(native_height, native_width),
            )
            if proxy_scale < 1.0:
                proxy_width = max(1, int(round(native_width * proxy_scale)))
                proxy_height = max(1, int(round(native_height * proxy_scale)))
                grab_source = cv2.resize(
                    source_bgr,
                    (proxy_width, proxy_height),
                    interpolation=cv2.INTER_AREA,
                )
                grab_hsv = cv2.resize(
                    hsv_image,
                    (proxy_width, proxy_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                grab_candidate = cv2.resize(
                    candidate,
                    (proxy_width, proxy_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                grab_source = source_bgr
                grab_hsv = hsv_image
                grab_candidate = candidate

            grab_h, grab_s, grab_v = cv2.split(grab_hsv)
            height, width = grab_candidate.shape[:2]
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
                (((grab_h <= 35) | (grab_h >= 165))
                 & (grab_s >= 70) & (grab_v >= 45))
                & (grab_candidate > 0)
            )
            green_seed = (
                ((grab_h >= 35) & (grab_h <= 95)
                 & (grab_s >= 75) & (grab_v <= 220))
                & (grab_candidate > 0)
            )
            # Pale Harumanis skin can be bright and low-saturation, so it
            # falls outside the strict HSV green seed above. In darker
            # brown/gray scenes, use channel dominance as an additional
            # fruit seed: mango skin is usually green-dominant while the
            # background is not. This prevents the background from becoming
            # one connected foreground region and avoids clipping the fruit.
            source_blue, source_green, source_red = cv2.split(grab_source)
            green_dominant = (
                (source_green.astype(np.float32) >= 1.03 * source_red)
                & (source_green.astype(np.float32) >= 1.03 * source_blue)
            )
            pale_green_seed = (
                (grab_h >= 25) & (grab_h <= 105)
                & (grab_s >= 20) & (grab_v >= 45)
                & green_dominant
                & (grab_candidate > 0)
            )
            green_seed = green_seed | pale_green_seed
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
            # Bright green mango skin can have saturation below the green
            # seed threshold. In that case GrabCut was seeded only by the
            # darker stem, producing the exact failure where the stem remains
            # white and the mango body disappears. Use the largest candidate
            # component as a conservative fruit seed when colour seeds cover
            # too little of the candidate foreground.
            candidate_area = max(1, cv2.countNonZero(grab_candidate))
            seed_area = cv2.countNonZero(fruit_seed)
            if seed_area < 0.20 * candidate_area:
                component_count, component_labels, component_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        grab_candidate,
                        connectivity=8,
                    )
                )
                component_labels_to_try = sorted(
                    range(1, component_count),
                    key=lambda label: component_stats[label, cv2.CC_STAT_AREA],
                    reverse=True,
                )
                fallback_label = None
                for label in component_labels_to_try:
                    component_area = component_stats[label, cv2.CC_STAT_AREA]
                    if component_area < 0.03 * grab_candidate.size:
                        break
                    if fallback_label is None:
                        fallback_label = label
                    component_x = component_stats[label, cv2.CC_STAT_LEFT]
                    component_y = component_stats[label, cv2.CC_STAT_TOP]
                    component_w = component_stats[label, cv2.CC_STAT_WIDTH]
                    component_h = component_stats[label, cv2.CC_STAT_HEIGHT]
                    touches_frame = (
                        component_x == 0
                        or component_y == 0
                        or component_x + component_w >= width
                        or component_y + component_h >= height
                    )
                    if not touches_frame:
                        fallback_label = label
                        break
                if fallback_label is not None:
                    fruit_seed = np.where(
                        component_labels == fallback_label,
                        255,
                        0,
                    ).astype(np.uint8)
            grabcut_mask[fruit_seed == 0] = cv2.GC_PR_BGD
            grabcut_mask[grab_candidate == 0] = cv2.GC_BGD
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
                        grab_source,
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
                    if proxy_scale < 1.0:
                        grabcut_foreground = cv2.resize(
                            grabcut_foreground,
                            (native_width, native_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    native_fruit_seed = (
                        cv2.resize(
                            fruit_seed,
                            (native_width, native_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if proxy_scale < 1.0
                        else fruit_seed
                    )
                    mask = cv2.bitwise_and(
                        grabcut_foreground,
                        candidate,
                    )
                    # If the HSV candidate touches the frame, it may contain
                    # a dark/brown background connected to the mango. When
                    # GrabCut returns that background-connected region, use
                    # the largest clean pale-green seed component instead.
                    # This is especially important for green Harumanis fruit
                    # photographed against brown or gray backgrounds.
                    seed_count, seed_labels, seed_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            native_fruit_seed,
                            connectivity=8,
                        )
                    )
                    if seed_count > 1:
                        largest_seed_label = 1 + int(
                            np.argmax(seed_stats[1:, cv2.CC_STAT_AREA])
                        )
                        largest_seed_area = int(
                            seed_stats[largest_seed_label, cv2.CC_STAT_AREA]
                        )
                        largest_seed = np.where(
                            seed_labels == largest_seed_label,
                            255,
                            0,
                        ).astype(np.uint8)
                        seed_is_substantial = (
                            largest_seed_area >= 0.05 * candidate.size
                        )
                        candidate_touches_frame = border_candidate_ratio > 0.01
                        graphcut_area = cv2.countNonZero(mask)
                        if (
                            seed_is_substantial
                            and candidate_touches_frame
                            and graphcut_area > 1.20 * largest_seed_area
                        ):
                            mask = cv2.bitwise_and(
                                largest_seed,
                                candidate,
                            )
                    # A graph cut that retains almost the whole frame has
                    # failed to separate a natural background.  In that
                    # case, use the conservative fruit-colour seed instead
                    # of returning a scene-sized foreground mask.
                    if (
                        cv2.countNonZero(mask)
                        > 0.72 * mask.shape[0] * mask.shape[1]
                        and cv2.countNonZero(fruit_seed) > 0
                    ):
                        if proxy_scale < 1.0:
                            fruit_seed = cv2.resize(
                                fruit_seed,
                                (native_width, native_height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        mask = cv2.bitwise_and(fruit_seed, candidate)

                    # GrabCut can keep only a high-contrast stem or branch
                    # when a pale green mango has nearly the same colour as
                    # its background.  Recover the dominant candidate
                    # component when the graph-cut result has little overlap
                    # with it.  The identity gate still decides whether this
                    # recovered foreground is actually a mango.
                    candidate_count, candidate_labels, candidate_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            candidate,
                            connectivity=8,
                        )
                    )
                    if candidate_count > 1:
                        candidate_labels_to_try = sorted(
                            range(1, candidate_count),
                            key=lambda label: candidate_stats[
                                label,
                                cv2.CC_STAT_AREA,
                            ],
                            reverse=True,
                        )
                        largest_candidate_label = candidate_labels_to_try[0]
                        for label in candidate_labels_to_try:
                            component_area = candidate_stats[
                                label,
                                cv2.CC_STAT_AREA,
                            ]
                            if component_area < 0.03 * candidate.size:
                                break
                            component_x = candidate_stats[
                                label,
                                cv2.CC_STAT_LEFT,
                            ]
                            component_y = candidate_stats[
                                label,
                                cv2.CC_STAT_TOP,
                            ]
                            component_w = candidate_stats[
                                label,
                                cv2.CC_STAT_WIDTH,
                            ]
                            component_h = candidate_stats[
                                label,
                                cv2.CC_STAT_HEIGHT,
                            ]
                            touches_frame = (
                                component_x == 0
                                or component_y == 0
                                or component_x + component_w >= candidate.shape[1]
                                or component_y + component_h >= candidate.shape[0]
                            )
                            if not touches_frame:
                                largest_candidate_label = label
                                break
                        largest_candidate_area = candidate_stats[
                            largest_candidate_label,
                            cv2.CC_STAT_AREA,
                        ]
                        largest_candidate = np.where(
                            candidate_labels == largest_candidate_label,
                            255,
                            0,
                        ).astype(np.uint8)
                        largest_overlap = cv2.countNonZero(
                            cv2.bitwise_and(mask, largest_candidate)
                        )
                        graphcut_area = cv2.countNonZero(mask)
                        # A pale/dark green mango can have strong overlap
                        # with the candidate while GrabCut still contracts
                        # one side of the silhouette.  Recover the dominant
                        # isolated candidate in that case; otherwise the
                        # downstream classifier receives a visibly clipped
                        # mango.  The existing low-overlap branch remains for
                        # cases where GrabCut selects the wrong object.
                        if (
                            largest_candidate_area >= 0.03 * candidate.size
                            and (
                                (
                                    largest_overlap < 0.35 * largest_candidate_area
                                    and graphcut_area < 0.60 * largest_candidate_area
                                )
                                or (
                                    largest_overlap >= 0.65 * largest_candidate_area
                                    and graphcut_area < 0.90 * largest_candidate_area
                                )
                            )
                        ):
                            mask = largest_candidate
                except cv2.error:
                    # GrabCut can fail to initialize on high-resolution,
                    # low-contrast scenes. Prefer the validated green fruit
                    # seed in that case; reverting to the raw candidate can
                    # reintroduce a background-connected mask.
                    native_fruit_seed = (
                        cv2.resize(
                            fruit_seed,
                            (native_width, native_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if proxy_scale < 1.0
                        else fruit_seed
                    )
                    seed_count, seed_labels, seed_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            native_fruit_seed,
                            connectivity=8,
                        )
                    )
                    if seed_count > 1:
                        largest_seed_label = 1 + int(
                            np.argmax(seed_stats[1:, cv2.CC_STAT_AREA])
                        )
                        largest_seed_area = int(
                            seed_stats[largest_seed_label, cv2.CC_STAT_AREA]
                        )
                        if largest_seed_area >= 0.05 * candidate.size:
                            mask = np.where(
                                seed_labels == largest_seed_label,
                                255,
                                0,
                            ).astype(np.uint8)
                        else:
                            mask = candidate.copy()
                    else:
                        mask = candidate.copy()

        # ================================================================
        # BACKGROUND-CONNECTED CANDIDATE RECOVERY
        # ================================================================
        #
        # The saturation/value candidate is intentionally permissive, but
        # that means a brown or gray wall can become one large connected
        # component with a pale green unripe mango.  GrabCut can then return
        # the wall as foreground and the later component selection keeps a
        # clipped or scene-sized object.  Before the leaf and morphology
        # stages, recover the largest connected pale-green component from
        # the original BGR image.  This is a fallback only when the HSV
        # candidate touches the frame and the colour component is substantial
        # and overlaps the current foreground.
        if (
            source_bgr is not None
            and source_bgr.shape[:2] == hsv_image.shape[:2]
            and border_candidate_ratio > 0.01
        ):
            source_blue, source_green, source_red = cv2.split(source_bgr)
            pale_green_recovery = (
                (h >= 25) & (h <= 105)
                & (s >= 18) & (v >= 45)
                & (source_green.astype(np.float32) >= 1.03 * source_red)
                & (source_green.astype(np.float32) >= 1.03 * source_blue)
            ).astype(np.uint8) * 255
            pale_green_recovery = cv2.bitwise_and(
                pale_green_recovery,
                candidate,
            )
            pale_green_recovery = cv2.morphologyEx(
                pale_green_recovery,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            pale_green_recovery = cv2.morphologyEx(
                pale_green_recovery,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            )
            recovery_count, recovery_labels, recovery_stats, _ = (
                cv2.connectedComponentsWithStats(
                    pale_green_recovery,
                    connectivity=8,
                )
            )
            if recovery_count > 1:
                recovery_label = 1 + int(
                    np.argmax(recovery_stats[1:, cv2.CC_STAT_AREA])
                )
                recovery_area = int(
                    recovery_stats[recovery_label, cv2.CC_STAT_AREA]
                )
                current_area = cv2.countNonZero(mask)
                recovery_component = np.where(
                    recovery_labels == recovery_label,
                    255,
                    0,
                ).astype(np.uint8)
                recovery_overlap = cv2.countNonZero(
                    cv2.bitwise_and(mask, recovery_component)
                )
                if (
                    recovery_area >= 0.05 * candidate.size
                    and recovery_overlap >= 0.15 * recovery_area
                    and (
                        current_area > 1.15 * recovery_area
                        or cv2.countNonZero(candidate) > 1.50 * recovery_area
                    )
                ):
                    # The stem of an unripe Harumanis mango can be yellow
                    # rather than green, so it may be a separate colour
                    # component just above the body. Include only small
                    # nearby yellow/green components; a large component is
                    # usually the brown/gray background connected to the
                    # permissive HSV candidate.
                    attached_colour = (
                        (h >= 20) & (h <= 105)
                        & (s >= 20) & (v >= 45)
                        & (source_green.astype(np.float32) >= 0.75 * source_red)
                        & (source_green.astype(np.float32) >= 1.15 * source_blue)
                    ).astype(np.uint8) * 255
                    attached_colour = cv2.bitwise_and(
                        attached_colour,
                        candidate,
                    )
                    attached_colour = cv2.morphologyEx(
                        attached_colour,
                        cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    )
                    attached_count, attached_labels, attached_stats, _ = (
                        cv2.connectedComponentsWithStats(
                            attached_colour,
                            connectivity=8,
                        )
                    )
                    near_body = cv2.dilate(
                        recovery_component,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
                    )
                    recovered_mask = recovery_component.copy()
                    for attached_label in range(1, attached_count):
                        attached_area = int(
                            attached_stats[attached_label, cv2.CC_STAT_AREA]
                        )
                        if not (
                            0.001 * candidate.size <= attached_area
                            <= 0.10 * candidate.size
                        ):
                            continue
                        attached_component = np.where(
                            attached_labels == attached_label,
                            255,
                            0,
                        ).astype(np.uint8)
                        if cv2.countNonZero(
                            cv2.bitwise_and(attached_component, near_body)
                        ) > 0:
                            recovered_mask = cv2.bitwise_or(
                                recovered_mask,
                                attached_component,
                            )
                    # Close the small gap between a detached stem component
                    # and the fruit body so the final largest-component
                    # filter retains the complete mango object.
                    mask = cv2.morphologyEx(
                        recovered_mask,
                        cv2.MORPH_CLOSE,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
                    )

        # ================================================================
        # LEAF LAYER
        # ================================================================
        # A green leaf and an unripe green mango are visually similar.  Only
        # remove green regions when the same object also contains a meaningful
        # warm fruit core (yellow/orange/brown).  This protects all-green
        # unripe mangoes while removing the obvious green leaves attached to
        # yellow mangoes in product photographs.
        leaf_mask = np.zeros_like(mask)
        if cv2.countNonZero(mask) > 0:
            warm_pixels = (
                (((h <= 38) | (h >= 165)) & (s >= 55) & (v >= 35))
                & (mask > 0)
            )
            # HSV hue is not sufficient here: the yellow-green leaf in the
            # supplied mango2 photograph has almost the same hue as the
            # mango skin.  In BGR, foliage is usually green-dominant whereas
            # yellow mango skin is red-dominant.  Use both signals.
            green_pixels = (
                ((h >= 25) & (h <= 100) & (s >= 45) & (v >= 30))
                & (mask > 0)
            )
            if source_bgr is not None and source_bgr.shape[:2] == mask.shape:
                blue_channel, green_channel, red_channel = cv2.split(source_bgr)
                green_dominant = (
                    (green_channel.astype(np.float32)
                     > 1.05 * red_channel.astype(np.float32))
                    & (green_channel.astype(np.float32)
                       > 1.05 * blue_channel.astype(np.float32))
                )
                green_pixels &= green_dominant
            fruit_area = float(cv2.countNonZero(mask))
            warm_fraction = float(np.count_nonzero(warm_pixels)) / fruit_area
            if warm_fraction >= 0.04:
                green_layer = (green_pixels.astype(np.uint8) * 255)
                leaf_kernel_size = max(
                    5,
                    int(round(min(mask.shape[:2]) * 0.012)) | 1,
                )
                green_layer = cv2.morphologyEx(
                    green_layer,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (leaf_kernel_size, leaf_kernel_size),
                    ),
                )
                num_green, green_labels, green_stats, _ = (
                    cv2.connectedComponentsWithStats(
                        green_layer,
                        connectivity=8,
                    )
                )
                for label in range(1, num_green):
                    component_area = green_stats[label, cv2.CC_STAT_AREA]
                    component_width = green_stats[label, cv2.CC_STAT_WIDTH]
                    component_height = green_stats[label, cv2.CC_STAT_HEIGHT]
                    short_side = max(1, min(component_width, component_height))
                    long_side = max(component_width, component_height)
                    component_ratio = long_side / short_side
                    if (
                        component_area >= max(25, 0.002 * fruit_area)
                        and component_area <= 0.25 * fruit_area
                        and component_ratio >= 2.2
                    ):
                        leaf_mask[green_labels == label] = 255

                # The color layer can be broken into small pieces by veins or
                # highlights.  An elongated green contour is still a leaf;
                # remove it from the final fruit mask but retain it for the
                # technical visualization.
                mask = cv2.bitwise_and(
                    mask,
                    cv2.bitwise_not(leaf_mask),
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
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        # Dark rotten areas and deep shadows can fall below the HSV value
        # candidate threshold.  They are holes inside the fruit, not
        # background.  Fill only external fruit contours so the native
        # silhouette is preserved and the dark pixels remain available in
        # the original colour image for damage/ripeness analysis.
        external_contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if external_contours:
            filled_mask = np.zeros_like(mask)
            cv2.drawContours(
                filled_mask,
                external_contours,
                -1,
                255,
                thickness=cv2.FILLED,
            )
            if cv2.countNonZero(filled_mask) >= cv2.countNonZero(mask):
                mask = filled_mask

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

        if return_layers:
            return (
                segmented_hsv,
                mask,
                candidate_mask,
                leaf_mask,
            )
        return segmented_hsv, mask

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
