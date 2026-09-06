"""
modules/ripeness_classifier.py

Module 2 — Hybrid Ripeness Classification

Classifies a Module-1-preprocessed mango image into one of the
trained model's ripeness classes.

Hybrid architecture:

    Module 1 segmented HSV image
             |
       +-----+-------------------------+
       |                               |
       v                               v
    HSV -> RGB                  Feature Extraction
       |                       /       |        \\
       v                      /        |         \\
 EfficientNetB0        HSV Statistics  Colour     Statistical
       |                              Histogram   Analysis
       |                                  \\        /
       |                                   \\      /
       |                                    v    v
       |                              63 Features
       |                                    |
       |                              Dense Branch
       |                                    |
       +---------------+--------------------+
                       |
                 Feature Fusion
                       |
                  Dense Layers
                       |
                    Softmax
                       |
                Ripeness Class


IMPORTANT:

HSV values and colour statistics are NOT interpreted using
hard-coded ripeness thresholds.

They are numerical features supplied to the trained neural
network. The relationship between the extracted features
and ripeness is learned during model training.

Module 1 remains responsible for:

    - Resize
    - BGR → HSV
    - Gaussian filtering
    - CLAHE
    - Mango segmentation
    - Morphological processing
    - Largest component selection

Module 2 is responsible for:

    - Colour histogram analysis
    - Statistical analysis
    - HSV feature extraction
    - EfficientNetB0 feature extraction
    - Feature fusion
    - Ripeness classification
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# HSV statistical features
# ---------------------------------------------------------------------------
#
# 6 features:
#
#   mean H
#   mean S
#   mean V
#   std H
#   std S
#   std V
#
# ---------------------------------------------------------------------------

HSV_STAT_FEATURE_NAMES = [
    "mean_h",
    "mean_s",
    "mean_v",
    "std_h",
    "std_s",
    "std_v",
]


# ---------------------------------------------------------------------------
# Additional statistical features
# ---------------------------------------------------------------------------
#
# 9 features:
#
#   mean H
#   std H
#   median H
#
#   mean S
#   std S
#   median S
#
#   mean V
#   std V
#   median V
#
# These complement the six basic HSV statistical features.
#
# ---------------------------------------------------------------------------

STATISTICAL_FEATURE_NAMES = [
    "mean_h",
    "std_h",
    "median_h",
    "mean_s",
    "std_s",
    "median_s",
    "mean_v",
    "std_v",
    "median_v",
]


# ---------------------------------------------------------------------------
# Colour histogram configuration
# ---------------------------------------------------------------------------
#
# 16 bins for each HSV channel:
#
#   H = 16 bins
#   S = 16 bins
#   V = 16 bins
#
# Total:
#
#   16 × 3 = 48 histogram features
#
# ---------------------------------------------------------------------------

HISTOGRAM_BINS = 16

HISTOGRAM_FEATURE_NAMES = []

for channel_name in ["h", "s", "v"]:

    for bin_index in range(
        HISTOGRAM_BINS
    ):

        HISTOGRAM_FEATURE_NAMES.append(
            f"hist_{channel_name}_{bin_index}"
        )


# ============================================================================
# TOTAL FEATURE CONFIGURATION
# ============================================================================

NUM_HSV_STAT_FEATURES = (
    len(HSV_STAT_FEATURE_NAMES)
)

NUM_HISTOGRAM_FEATURES = (
    len(HISTOGRAM_FEATURE_NAMES)
)

NUM_STATISTICAL_FEATURES = (
    len(STATISTICAL_FEATURE_NAMES)
)

TOTAL_FEATURES = (
    NUM_HSV_STAT_FEATURES
    + NUM_HISTOGRAM_FEATURES
    + NUM_STATISTICAL_FEATURES
)


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

# OpenCV 8-bit HSV ranges:
#
# H = 0 ... 179
# S = 0 ... 255
# V = 0 ... 255
#
# Used for the six basic HSV statistical features.
#
HSV_SCALE = np.array(
    [
        179.0,
        255.0,
        255.0,
        179.0,
        255.0,
        255.0,
    ],
    dtype=np.float32,
)


# Used for the nine statistical features:
#
# mean H
# std H
# median H
# mean S
# std S
# median S
# mean V
# std V
# median V
#
STATISTICAL_SCALE = np.array(
    [
        179.0,
        179.0,
        179.0,
        255.0,
        255.0,
        255.0,
        255.0,
        255.0,
        255.0,
    ],
    dtype=np.float32,
)


# ============================================================================
# BASIC HSV STATISTICAL ANALYSIS
# ============================================================================

def extract_hsv_features(
    hsv_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract six basic HSV statistical features.

    Features:

        mean H
        mean S
        mean V
        standard deviation H
        standard deviation S
        standard deviation V

    The mask is used so that only segmented mango pixels
    contribute to the statistics.

    Args:
        hsv_image:
            8-bit HSV image.

        mask:
            Binary mango segmentation mask.

    Returns:
        Six-element float32 feature vector.
    """

    if mask is not None:

        valid_pixels = (
            mask > 0
        )

        h = (
            hsv_image[..., 0][valid_pixels]
            .astype(np.float32)
        )

        s = (
            hsv_image[..., 1][valid_pixels]
            .astype(np.float32)
        )

        v = (
            hsv_image[..., 2][valid_pixels]
            .astype(np.float32)
        )

    else:

        h = (
            hsv_image[..., 0]
            .astype(np.float32)
            .reshape(-1)
        )

        s = (
            hsv_image[..., 1]
            .astype(np.float32)
            .reshape(-1)
        )

        v = (
            hsv_image[..., 2]
            .astype(np.float32)
            .reshape(-1)
        )

    # ------------------------------------------------------------------------
    # Safety handling
    # ------------------------------------------------------------------------

    if len(h) == 0:

        return np.zeros(
            6,
            dtype=np.float32,
        )

    return np.array(
        [
            np.mean(h),
            np.mean(s),
            np.mean(v),
            np.std(h),
            np.std(s),
            np.std(v),
        ],
        dtype=np.float32,
    )


# ============================================================================
# ADDITIONAL STATISTICAL ANALYSIS
# ============================================================================

def extract_statistical_features(
    hsv_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract nine statistical features.

    Features:

        H:
            mean
            standard deviation
            median

        S:
            mean
            standard deviation
            median

        V:
            mean
            standard deviation
            median

    Only segmented mango pixels are used when a mask is provided.

    Returns:

        Nine-element float32 feature vector.
    """

    if mask is not None:

        valid_pixels = (
            mask > 0
        )

        h = (
            hsv_image[..., 0][valid_pixels]
            .astype(np.float32)
        )

        s = (
            hsv_image[..., 1][valid_pixels]
            .astype(np.float32)
        )

        v = (
            hsv_image[..., 2][valid_pixels]
            .astype(np.float32)
        )

    else:

        h = (
            hsv_image[..., 0]
            .astype(np.float32)
            .reshape(-1)
        )

        s = (
            hsv_image[..., 1]
            .astype(np.float32)
            .reshape(-1)
        )

        v = (
            hsv_image[..., 2]
            .astype(np.float32)
            .reshape(-1)
        )

    # ------------------------------------------------------------------------
    # Safety handling
    # ------------------------------------------------------------------------

    if len(h) == 0:

        return np.zeros(
            9,
            dtype=np.float32,
        )

    return np.array(
        [
            np.mean(h),
            np.std(h),
            np.median(h),

            np.mean(s),
            np.std(s),
            np.median(s),

            np.mean(v),
            np.std(v),
            np.median(v),
        ],
        dtype=np.float32,
    )


# ============================================================================
# COLOUR HISTOGRAM ANALYSIS
# ============================================================================

def extract_colour_histogram(
    hsv_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract normalized colour histograms from HSV channels.

    Histogram configuration:

        H → 16 bins
        S → 16 bins
        V → 16 bins

    Total:

        48 histogram features.

    The histogram is calculated only from segmented mango
    pixels when a mask is provided.

    Histograms are normalized so that each channel histogram
    sums approximately to 1.

    Returns:

        48-element float32 feature vector.
    """

    if mask is not None:

        valid_pixels = (
            mask > 0
        )

        h = (
            hsv_image[..., 0][valid_pixels]
            .astype(np.float32)
        )

        s = (
            hsv_image[..., 1][valid_pixels]
            .astype(np.float32)
        )

        v = (
            hsv_image[..., 2][valid_pixels]
            .astype(np.float32)
        )

    else:

        h = (
            hsv_image[..., 0]
            .astype(np.float32)
            .reshape(-1)
        )

        s = (
            hsv_image[..., 1]
            .astype(np.float32)
            .reshape(-1)
        )

        v = (
            hsv_image[..., 2]
            .astype(np.float32)
            .reshape(-1)
        )

    # ------------------------------------------------------------------------
    # Safety handling
    # ------------------------------------------------------------------------

    if len(h) == 0:

        return np.zeros(
            48,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------------
    # Histogram ranges
    # ------------------------------------------------------------------------
    #
    # H = 0 ... 180
    # S = 0 ... 256
    # V = 0 ... 256
    #
    # ------------------------------------------------------------------------

    h_hist, _ = np.histogram(
        h,
        bins=HISTOGRAM_BINS,
        range=(0, 180),
    )

    s_hist, _ = np.histogram(
        s,
        bins=HISTOGRAM_BINS,
        range=(0, 256),
    )

    v_hist, _ = np.histogram(
        v,
        bins=HISTOGRAM_BINS,
        range=(0, 256),
    )

    # ------------------------------------------------------------------------
    # Convert to float
    # ------------------------------------------------------------------------

    h_hist = h_hist.astype(
        np.float32
    )

    s_hist = s_hist.astype(
        np.float32
    )

    v_hist = v_hist.astype(
        np.float32
    )

    # ------------------------------------------------------------------------
    # Normalize each histogram independently
    # ------------------------------------------------------------------------

    if h_hist.sum() > 0:

        h_hist /= h_hist.sum()

    if s_hist.sum() > 0:

        s_hist /= s_hist.sum()

    if v_hist.sum() > 0:

        v_hist /= v_hist.sum()

    return np.concatenate(
        [
            h_hist,
            s_hist,
            v_hist,
        ]
    ).astype(
        np.float32
    )


# ============================================================================
# NORMALIZE FEATURES
# ============================================================================

def normalize_hsv_features(
    raw_features: np.ndarray,
) -> np.ndarray:
    """
    Normalize the six basic HSV statistical features.
    """

    return (
        raw_features / HSV_SCALE
    ).astype(
        np.float32
    )


def normalize_statistical_features(
    raw_features: np.ndarray,
) -> np.ndarray:
    """
    Normalize the nine additional statistical features.
    """

    return (
        raw_features
        / STATISTICAL_SCALE
    ).astype(
        np.float32
    )


# ============================================================================
# EXTRACT COMPLETE 63-FEATURE VECTOR
# ============================================================================

def extract_all_features(
    hsv_image: np.ndarray,
    mask: np.ndarray | None = None,
):
    """
    Extract the complete Module 2 feature vector.

    Feature composition:

        1. Basic HSV statistics
           6 features

        2. Colour histograms
           48 features

        3. Additional statistical analysis
           9 features

        --------------------------------
        Total = 63 features

    Returns:

        normalized_features:
            63-element normalized feature vector.

        raw_features:
            Dictionary containing all raw feature groups.
    """

    # ------------------------------------------------------------------------
    # 1. Basic HSV statistics
    # ------------------------------------------------------------------------

    raw_hsv = extract_hsv_features(
        hsv_image,
        mask,
    )

    normalized_hsv = (
        normalize_hsv_features(
            raw_hsv
        )
    )

    # ------------------------------------------------------------------------
    # 2. Colour histogram
    # ------------------------------------------------------------------------

    histogram = (
        extract_colour_histogram(
            hsv_image,
            mask,
        )
    )

    # Histogram is already normalized to [0, 1].
    normalized_histogram = (
        histogram.astype(
            np.float32
        )
    )

    # ------------------------------------------------------------------------
    # 3. Additional statistical analysis
    # ------------------------------------------------------------------------

    raw_statistics = (
        extract_statistical_features(
            hsv_image,
            mask,
        )
    )

    normalized_statistics = (
        normalize_statistical_features(
            raw_statistics
        )
    )

    # ------------------------------------------------------------------------
    # Combine all features
    # ------------------------------------------------------------------------

    combined_features = np.concatenate(
        [
            normalized_hsv,
            normalized_histogram,
            normalized_statistics,
        ]
    ).astype(
        np.float32
    )

    if len(combined_features) != TOTAL_FEATURES:

        raise RuntimeError(
            "Unexpected feature count. "
            f"Expected {TOTAL_FEATURES}, "
            f"got {len(combined_features)}."
        )

    raw_features = {

        "hsv_statistics": {
            name: float(value)
            for name, value in zip(
                HSV_STAT_FEATURE_NAMES,
                raw_hsv,
            )
        },

        "colour_histogram": {
            name: float(value)
            for name, value in zip(
                HISTOGRAM_FEATURE_NAMES,
                histogram,
            )
        },

        "statistical_analysis": {
            name: float(value)
            for name, value in zip(
                STATISTICAL_FEATURE_NAMES,
                raw_statistics,
            )
        },
    }

    return (
        combined_features,
        raw_features,
    )


# ============================================================================
# HYBRID CLASSIFIER
# ============================================================================

class HybridRipenessClassifier:
    """
    Wrapper around the trained hybrid:

        EfficientNetB0
                +
        63 colour/statistical features

    model.
    """

    def __init__(
        self,
        model_path: str = "models/mango_ripeness.keras",
        class_indices_path: str = "models/class_indices.json",
        input_size: tuple = (224, 224),
    ):

        self.model_path = model_path
        self.class_indices_path = (
            class_indices_path
        )
        self.input_size = input_size

        # --------------------------------------------------------------------
        # Validate model
        # --------------------------------------------------------------------

        if not Path(
            self.model_path
        ).exists():

            raise FileNotFoundError(
                f"Ripeness model not found at "
                f"'{self.model_path}'. "
                f"Run training/train_model.py first."
            )

        # --------------------------------------------------------------------
        # Validate class mapping
        # --------------------------------------------------------------------

        if not Path(
            self.class_indices_path
        ).exists():

            raise FileNotFoundError(
                f"Class index mapping not found at "
                f"'{self.class_indices_path}'. "
                f"Run training/train_model.py first."
            )

        # --------------------------------------------------------------------
        # Load class mapping
        # --------------------------------------------------------------------

        with open(
            self.class_indices_path,
            "r",
            encoding="utf-8",
        ) as f:

            raw_mapping = json.load(
                f
            )

        # Sort according to integer class index.
        self.class_names = [
            raw_mapping[str(i)]
            for i in range(
                len(raw_mapping)
            )
        ]

        # --------------------------------------------------------------------
        # Load trained model
        # --------------------------------------------------------------------

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
        )
        self._infer = tf.function(
            lambda inputs: self.model(inputs, training=False),
            reduce_retracing=True,
        )

    # =========================================================================
    # IMAGE BRANCH
    # =========================================================================

    def prepare_image_branch(
        self,
        segmented_hsv: np.ndarray,
        model_segmented_hsv: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Prepare Module 1's segmented HSV image for EfficientNetB0.

        Pipeline:

            Module 1 segmented HSV
                    ↓
                HSV → RGB
                    ↓
                 Resize
                    ↓
               float32 [0,255]
                    ↓
              Batch dimension

        IMPORTANT:

        Do NOT divide by 255 here.

        Keras EfficientNetB0 performs its own input rescaling.
        """

        if model_segmented_hsv is not None:
            expected = (self.input_size[1], self.input_size[0], 3)
            if model_segmented_hsv.shape != expected:
                raise ValueError(f"Model HSV image must have shape {expected}.")
            rgb = cv2.cvtColor(model_segmented_hsv, cv2.COLOR_HSV2RGB)
            return np.expand_dims(rgb.astype(np.float32), axis=0)

        # Legacy callers only supply native HSV. Resize in BGR, as Module 1
        # does during training; interpolating hue directly changes colours.
        segmented_bgr = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2BGR)
        height, width = segmented_bgr.shape[:2]
        target_width, target_height = self.input_size
        scale = min(target_width / width, target_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            segmented_bgr,
            (new_width, new_height),
            interpolation=interpolation,
        )
        canvas = np.zeros(
            (target_height, target_width, 3),
            dtype=resized.dtype,
        )
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
        ] = resized
        # This is the representation used by the existing trained model:
        # HSV channels converted to an RGB-shaped tensor.  Keep it stable so
        # a source-level preprocessing refactor does not silently invalidate
        # the saved ripeness weights.
        model_hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(model_hsv, cv2.COLOR_HSV2RGB)

        normalized = rgb.astype(
            np.float32
        )

        return np.expand_dims(
            normalized,
            axis=0,
        )

    # =========================================================================
    # FEATURE BRANCH
    # =========================================================================

    def prepare_feature_branch(
        self,
        segmented_hsv: np.ndarray,
        mask: np.ndarray | None = None,
    ):
        """
        Extract the complete 63-feature vector from the
        Module 1 segmented mango.

        Returns:

            normalized_batch:
                Shape (1, 63), used by the neural network.

            raw_features:
                Dictionary containing the raw feature groups.
        """

        normalized_features, raw_features = (
            extract_all_features(
                segmented_hsv,
                mask,
            )
        )

        normalized_batch = (
            np.expand_dims(
                normalized_features,
                axis=0,
            )
        )

        return (
            normalized_batch,
            raw_features,
        )

    # =========================================================================
    # BACKWARD-COMPATIBLE METHOD
    # =========================================================================

    def prepare_hsv_branch(
        self,
        segmented_hsv: np.ndarray,
        mask: np.ndarray | None = None,
    ):
        """
        Backward-compatible wrapper.

        The old implementation returned only six HSV features.

        The new implementation returns the complete 63-feature
        Module 2 feature vector.
        """

        return self.prepare_feature_branch(
            segmented_hsv,
            mask,
        )

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(
        self,
        segmented_hsv: np.ndarray,
        mask: np.ndarray | None = None,
        model_segmented_hsv: np.ndarray | None = None,
    ) -> dict:
        """
        Run hybrid ripeness prediction.

        Args:
            segmented_hsv:
                The "segmented" output from Module 1.

            mask:
                The mango segmentation mask from Module 1.

            model_segmented_hsv:
                Module 1's model-sized HSV output, matching the training input.

        Returns:
            Dictionary containing:

                prediction
                confidence
                probabilities
                hsv_features
                colour_histogram
                statistical_analysis
                feature_count
                inference_time_ms
        """

        # --------------------------------------------------------------------
        # Prepare EfficientNet branch
        # --------------------------------------------------------------------

        image_input = (
            self.prepare_image_branch(
                segmented_hsv,
                model_segmented_hsv=model_segmented_hsv,
            )
        )

        # --------------------------------------------------------------------
        # Prepare 63-feature branch
        # --------------------------------------------------------------------

        feature_input, raw_features = (
            self.prepare_feature_branch(
                segmented_hsv,
                mask,
            )
        )

        # --------------------------------------------------------------------
        # Model inference
        # --------------------------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        raw_probs = np.asarray(self._infer(
            {
                "image_input": image_input,
                "feature_input": feature_input,
            },
        ))[0]

        inference_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        # --------------------------------------------------------------------
        # Determine predicted class
        # --------------------------------------------------------------------

        predicted_index = int(
            np.argmax(
                raw_probs
            )
        )

        predicted_label = (
            self.class_names[
                predicted_index
            ]
        )

        confidence = (
            float(
                raw_probs[
                    predicted_index
                ]
            )
            * 100
        )

        # --------------------------------------------------------------------
        # Probability distribution
        # --------------------------------------------------------------------

        probabilities = {
            class_name: float(prob) * 100
            for class_name, prob in zip(
                self.class_names,
                raw_probs,
            )
        }

        # --------------------------------------------------------------------
        # Return result
        # --------------------------------------------------------------------

        return {

            "prediction": (
                predicted_label
            ),

            "confidence": (
                confidence
            ),

            "probabilities": (
                probabilities
            ),

            "hsv_features": (
                raw_features[
                    "hsv_statistics"
                ]
            ),

            "colour_histogram": (
                raw_features[
                    "colour_histogram"
                ]
            ),

            "statistical_analysis": (
                raw_features[
                    "statistical_analysis"
                ]
            ),

            "feature_count": (
                TOTAL_FEATURES
            ),

            "inference_time_ms": (
                inference_time_ms
            ),
        }


# ============================================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================================

EfficientNetClassifier = (
    HybridRipenessClassifier
)
