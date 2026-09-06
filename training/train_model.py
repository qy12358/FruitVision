"""
training/train_model.py

Module 2 — Hybrid Mango Ripeness Classification

Architecture:

    Original Mango Image
             |
             v
    Module 1 Preprocessing
             |
             v
      Segmented HSV Image
             |
       +-----+-----------------------------+
       |                                   |
       v                                   v
    HSV -> RGB                       Feature Extraction
       |                           /        |          \\
       v                          /         |           \\
 EfficientNetB0          HSV Statistics  Colour      Statistical
       |                              Histogram       Analysis
       |                                  \\             /
       |                                   \\           /
       |                                    v         v
       |                              63 Features
       |                                   |
       |                              Dense Branch
       |                                   |
       +------------------+----------------+
                          |
                    Feature Fusion
                          |
                     Dense Layers
                          |
                       Softmax
                          |
                    Ripeness Class


Dataset layout:

dataset/mango_harumanis/harumanis_phases_V2/images/
        ripe/
        rotten/
        semi_ripe/
        unripe/

The original images are split deterministically into train/validation/test.
The separate augmented folder is not used for evaluation because it contains
near-duplicates of the originals and would leak information across splits.


Important:

- Class names are read automatically from training folders.
- class_indices.json is generated automatically.
- No HSV ripeness threshold is hard-coded.
- Colour histograms are not used as hard-coded classification rules.
- Statistical features are not used as hard-coded classification rules.
- All extracted features are learned by the neural network.
- Class weights are automatically calculated.
- Module 1 preprocessing is used during training.
- The exact same Module 1 preprocessing output is used during inference.
- EfficientNet receives pixel values in [0,255].
- Keras EfficientNetB0 performs its own input rescaling.
- The feature branch contains 63 normalized features.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json
import hashlib
import random
import re
import sys
from pathlib import Path


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
)

from tensorflow.keras.applications import (
    EfficientNetB0,
)

from tensorflow.keras.preprocessing.image import ( 
    ImageDataGenerator,
)

from tensorflow.keras.utils import (
    Sequence,
)

from modules.preprocessing import (
    ImagePreprocessor,
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42

random.seed(
    SEED
)

np.random.seed(
    SEED
)

tf.random.set_seed(
    SEED
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mango_harumanis"
    / "harumanis_phases_V2"
    / "images"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_OUTPUT_PATH = (
    MODEL_DIR
    / "mango_ripeness.keras"
)

CLASS_INDICES_PATH = (
    MODEL_DIR
    / "class_indices.json"
)

HISTORY_PLOT_PATH = (
    MODEL_DIR
    / "training_history.png"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

IMG_SIZE = (
    224,
    224,
)

BATCH_SIZE = 32

NUM_CLASSES = 4


# Phase 1:
# Train classification head while EfficientNet is frozen.
PHASE1_EPOCHS = 15


# Phase 2:
# Fine-tune upper EfficientNet layers.
PHASE2_EPOCHS = 15


# Number of EfficientNet layers to unfreeze.  The clean, non-conflicting
# dataset is small, so updating a large part of the ImageNet backbone makes
# the model memorize camera/background cues.  A short final fine-tune is
# enough to adapt fruit texture while keeping generic visual features.
FINE_TUNE_LAYERS = 10


PHASE1_LR = 1e-3

PHASE2_LR = 1e-5


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# Basic HSV statistics
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
# Additional statistical analysis
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
# Colour histogram
# ---------------------------------------------------------------------------

HISTOGRAM_BINS = 16


HISTOGRAM_FEATURE_NAMES = []

for channel_name in [
    "h",
    "s",
    "v",
]:

    for bin_index in range(
        HISTOGRAM_BINS
    ):

        HISTOGRAM_FEATURE_NAMES.append(
            f"hist_{channel_name}_{bin_index}"
        )


# ---------------------------------------------------------------------------
# Feature counts
# ---------------------------------------------------------------------------

NUM_HSV_STAT_FEATURES = (
    len(
        HSV_STAT_FEATURE_NAMES
    )
)

NUM_HISTOGRAM_FEATURES = (
    len(
        HISTOGRAM_FEATURE_NAMES
    )
)

NUM_STATISTICAL_FEATURES = (
    len(
        STATISTICAL_FEATURE_NAMES
    )
)

TOTAL_FEATURES = (
    NUM_HSV_STAT_FEATURES
    + NUM_HISTOGRAM_FEATURES
    + NUM_STATISTICAL_FEATURES
)


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================

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
# IMAGE EXTENSIONS
# ============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================================
# HSV STATISTICAL ANALYSIS
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
        std H
        std S
        std V

    Only segmented mango pixels are used when mask is provided.
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
            std
            median

        S:
            mean
            std
            median

        V:
            mean
            std
            median
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
    Extract normalized 16-bin histograms for H, S and V.

    Total:

        16 × 3 = 48 features.
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

    if len(h) == 0:

        return np.zeros(
            48,
            dtype=np.float32,
        )

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

    h_hist = h_hist.astype(
        np.float32
    )

    s_hist = s_hist.astype(
        np.float32
    )

    v_hist = v_hist.astype(
        np.float32
    )

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
# NORMALIZATION
# ============================================================================

def normalize_hsv_features(
    features: np.ndarray,
) -> np.ndarray:
    """
    Normalize the six HSV statistical features.
    """

    return (
        features / HSV_SCALE
    ).astype(
        np.float32
    )


def normalize_statistical_features(
    features: np.ndarray,
) -> np.ndarray:
    """
    Normalize the nine additional statistical features.
    """

    return (
        features / STATISTICAL_SCALE
    ).astype(
        np.float32
    )


# ============================================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================================

def extract_all_features(
    hsv_image: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract the complete 63-feature vector.

    Composition:

        6  HSV statistics
        48 colour histogram features
        9  statistical features

        Total = 63
    """

    # ------------------------------------------------------------------------
    # HSV statistics
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
    # Colour histogram
    # ------------------------------------------------------------------------

    histogram = (
        extract_colour_histogram(
            hsv_image,
            mask,
        )
    )

    # ------------------------------------------------------------------------
    # Additional statistical analysis
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
    # Combine
    # ------------------------------------------------------------------------

    combined_features = np.concatenate(
        [
            normalized_hsv,
            histogram,
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

    return combined_features


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_classes(
    dataset_dir: Path,
) -> list:
    """
    Automatically discover class folders.

    Folder names define the class names.

    Classes are sorted alphabetically so the same deterministic
    mapping is used throughout the pipeline.
    """

    if not dataset_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{dataset_dir}"
        )

    class_names = sorted(
        [
            folder.name
            for folder in dataset_dir.iterdir()
            if folder.is_dir()
        ]
    )

    if not class_names:

        raise RuntimeError(
            f"No class folders were found in:\n"
            f"{dataset_dir}"
        )

    return class_names


# ============================================================================
# BUILD FILE / LABEL LIST
# ============================================================================

def build_file_label_list(
    dataset_dir: str | Path,
    class_names: list,
):
    """
    Build image file paths and corresponding integer labels.
    """

    dataset_dir = Path(
        dataset_dir
    )

    class_to_index = {
        name: index
        for index, name in enumerate(
            class_names
        )
    }

    files = []

    labels = []

    for class_name in class_names:

        class_dir = (
            dataset_dir
            / class_name
        )

        if not class_dir.exists():

            print(
                "WARNING: Class folder "
                f"does not exist: {class_dir}"
            )

            continue

        for file_path in sorted(
            class_dir.rglob("*")
        ):

            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                files.append(
                    str(file_path)
                )

                labels.append(
                    class_to_index[
                        class_name
                    ]
                )

    return (
        files,
        labels,
    )


def build_stratified_splits(
    dataset_dir: str | Path,
    class_names: list,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    include_augmented_train: bool = False,
):
    """Create deterministic per-class train/validation/test splits.

    Splitting within each class keeps every maturity class represented while
    ensuring the held-out sets contain original images only.  The same helper
    is imported by ``evaluate_model.py`` so evaluation uses exactly the split
    used during training.
    """
    dataset_dir = Path(dataset_dir)
    rng = random.Random(SEED)
    split_files = {"train": [], "validation": [], "test": []}
    split_labels = {"train": [], "validation": [], "test": []}

    # Deduplicate before splitting.  The prepared dataset contains exact
    # copies under different maturity labels; retaining those would make the
    # model receive contradictory targets and can leak a duplicate into the
    # held-out set.  Files with conflicting labels are excluded rather than
    # guessing which label is correct.
    by_hash = {}
    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                by_hash.setdefault(digest, []).append((str(path), class_index))

    conflict_count = 0
    class_files_by_index = {index: [] for index in range(len(class_names))}
    for items in by_hash.values():
        labels = {label for _, label in items}
        if len(labels) != 1:
            conflict_count += 1
            continue
        class_index = next(iter(labels))
        class_files_by_index[class_index].append(items[0][0])

    if conflict_count:
        print(
            "WARNING: excluded "
            f"{conflict_count} exact-duplicate hash groups with conflicting labels."
        )

    for class_index, class_name in enumerate(class_names):
        class_files = class_files_by_index[class_index]
        rng.shuffle(class_files)
        total = len(class_files)
        test_count = max(1, int(round(total * test_fraction)))
        validation_count = max(1, int(round(total * validation_fraction)))
        if test_count + validation_count >= total:
            validation_count = 1
            test_count = 1

        test_files = class_files[:test_count]
        validation_files = class_files[test_count:test_count + validation_count]
        train_files = class_files[test_count + validation_count:]

        for split_name, files in (
            ("train", train_files),
            ("validation", validation_files),
            ("test", test_files),
        ):
            split_files[split_name].extend(files)
            split_labels[split_name].extend([class_index] * len(files))
    if include_augmented_train:
        augmented_dir = (
            dataset_dir.parent.parent
            / "harumanis_phases_V2 augmented"
        )
        class_to_index = {
            name.lower(): index
            for index, name in enumerate(class_names)
        }
        augmented_added = 0
        if augmented_dir.exists():
            for path in sorted(augmented_dir.rglob("*")):
                if not (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    continue
                label_index = class_to_index.get(path.parent.name.lower())
                # Augmented images are training-only data. Use every valid
                # maturity-labelled augmentation so the underrepresented
                # rotten class is not reduced to the subset whose original
                # happened to land in the training split.
                if label_index is not None:
                    split_files["train"].append(str(path))
                    split_labels["train"].append(label_index)
                    augmented_added += 1
            print(
                f"Added {augmented_added} augmented images to training only."
            )

    return (
        split_files["train"], split_labels["train"],
        split_files["validation"], split_labels["validation"],
        split_files["test"], split_labels["test"],
    )


def validate_image_files(file_paths):
    """Fail before training if any selected dataset image cannot be decoded."""

    unreadable = [
        path for path in file_paths
        if cv2.imread(path, cv2.IMREAD_COLOR) is None
    ]
    if unreadable:
        preview = "\n".join(f"- {path}" for path in unreadable[:20])
        more = "" if len(unreadable) <= 20 else f"\n- ... and {len(unreadable) - 20} more"
        raise RuntimeError(
            "Unreadable ripeness training images detected before training:\n"
            f"{preview}{more}\n"
            "Replace or remove these files, then run training again."
        )


# ============================================================================
# HYBRID DATA SEQUENCE
# ============================================================================

class HybridSequence(Sequence):
    """
    Keras Sequence for the hybrid model.

    Each sample produces two inputs:

        image_input:
            Module 1 segmented HSV converted to RGB.

        feature_input:
            63 normalized features extracted from the SAME
            Module 1 segmented HSV image and segmentation mask.

    This guarantees training and inference use the same
    representation.
    """

    def __init__(
        self,
        file_paths,
        labels,
        num_classes,
        batch_size=32,
        img_size=(224, 224),
        augment=False,
        shuffle=True,
    ):

        self.file_paths = list(
            file_paths
        )

        self.labels = list(
            labels
        )

        self.num_classes = (
            num_classes
        )

        self.batch_size = (
            batch_size
        )

        self.img_size = img_size

        self.augment = augment

        self.shuffle = shuffle

        # --------------------------------------------------------------------
        # Module 1 preprocessor
        # --------------------------------------------------------------------

        self.preprocessor = (
            ImagePreprocessor(
                resize=img_size
            )
        )

        # --------------------------------------------------------------------
        # Image augmentation
        # --------------------------------------------------------------------

        self.augmenter = (
            ImageDataGenerator(
                rotation_range=25,
                width_shift_range=0.15,
                height_shift_range=0.15,
                shear_range=0.10,
                zoom_range=0.20,
                horizontal_flip=True,
                brightness_range=[
                    0.8,
                    1.2,
                ],
                fill_mode="nearest",
            )
        )

        self.indices = np.arange(
            len(
                self.file_paths
            )
        )

        self.on_epoch_end()

    def __len__(
        self,
    ):
        """
        Number of batches per epoch.
        """

        return int(
            np.ceil(
                len(
                    self.file_paths
                )
                / self.batch_size
            )
        )

    def on_epoch_end(
        self,
    ):
        """
        Shuffle training data after each epoch.
        """

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )

    def __getitem__(
        self,
        batch_index,
    ):
        """
        Generate one batch.
        """

        start = (
            batch_index
            * self.batch_size
        )

        end = min(
            start
            + self.batch_size,
            len(
                self.file_paths
            ),
        )

        batch_indices = (
            self.indices[
                start:end
            ]
        )

        batch_size_actual = (
            len(
                batch_indices
            )
        )

        # --------------------------------------------------------------------
        # Allocate arrays
        # --------------------------------------------------------------------

        image_batch = np.zeros(
            (
                batch_size_actual,
                self.img_size[0],
                self.img_size[1],
                3,
            ),
            dtype=np.float32,
        )

        feature_batch = np.zeros(
            (
                batch_size_actual,
                TOTAL_FEATURES,
            ),
            dtype=np.float32,
        )

        label_batch = np.zeros(
            (
                batch_size_actual,
            ),
            dtype=np.int32,
        )

        # --------------------------------------------------------------------
        # Process every image
        # --------------------------------------------------------------------

        for (
            batch_position,
            dataset_index,
        ) in enumerate(
            batch_indices
        ):

            image_path = (
                self.file_paths[
                    dataset_index
                ]
            )

            label = (
                self.labels[
                    dataset_index
                ]
            )

            # ----------------------------------------------------------------
            # Load original image
            # ----------------------------------------------------------------

            image_bgr = cv2.imread(
                image_path
            )

            if image_bgr is None:

                raise ValueError(
                    "Unable to read image:\n"
                    f"{image_path}"
                )

            # ----------------------------------------------------------------
            # Data augmentation
            # ----------------------------------------------------------------

            if self.augment:

                image_rgb_for_aug = (
                    cv2.cvtColor(
                        image_bgr,
                        cv2.COLOR_BGR2RGB,
                    )
                )

                image_rgb_for_aug = (
                    self.augmenter.random_transform(
                        image_rgb_for_aug
                    )
                )

                image_rgb_for_aug = (
                    np.clip(
                        image_rgb_for_aug,
                        0,
                        255,
                    ).astype(
                        np.uint8
                    )
                )

                image_bgr = (
                    cv2.cvtColor(
                        image_rgb_for_aug,
                        cv2.COLOR_RGB2BGR,
                    )
                )

            # ----------------------------------------------------------------
            # Safety check
            # ----------------------------------------------------------------

            image_bgr = (
                np.clip(
                    image_bgr,
                    0,
                    255,
                ).astype(
                    np.uint8
                )
            )

            # ----------------------------------------------------------------
            # MODULE 1 PREPROCESSING
            # ----------------------------------------------------------------

            preprocessed = (
                self.preprocessor.preprocess(
                    image_bgr
                )
            )

            segmented_hsv = preprocessed["segmented"]
            mask = preprocessed["mask"]

            # ----------------------------------------------------------------
            # EfficientNet image branch
            # ----------------------------------------------------------------

            # The preprocessor segments at native resolution.  Use its
            # aspect-preserving model-sized HSV representation for
            # EfficientNet while retaining the native mask for the feature
            # branch.  This preserves the representation used by the saved
            # ripeness model.
            image_rgb = cv2.cvtColor(
                preprocessed["model_segmented_hsv"],
                cv2.COLOR_HSV2RGB,
            )

            image_batch[
                batch_position
            ] = image_rgb.astype(
                np.float32
            )

            # ----------------------------------------------------------------
            # 63-feature branch
            # ----------------------------------------------------------------

            features = (
                extract_all_features(
                    segmented_hsv,
                    mask,
                )
            )

            feature_batch[
                batch_position
            ] = features

            # ----------------------------------------------------------------
            # Label
            # ----------------------------------------------------------------

            label_batch[
                batch_position
            ] = label

        # --------------------------------------------------------------------
        # One-hot encode labels
        # --------------------------------------------------------------------

        y = (
            tf.keras.utils.to_categorical(
                label_batch,
                num_classes=self.num_classes,
            )
        )

        return (
            {
                "image_input": image_batch,

                "feature_input": feature_batch,
            },
            y,
        )


# ============================================================================
# BUILD HYBRID MODEL
# ============================================================================

def build_model(
    num_classes: int,
):
    """
    Build the hybrid EfficientNetB0 + 63-feature model.
    """

    # ========================================================================
    # EfficientNet branch
    # ========================================================================

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3,
        ),
        pooling="avg",
    )

    # Phase 1:
    # Freeze EfficientNet.
    base_model.trainable = False

    image_input = layers.Input(
        shape=(
            IMG_SIZE[0],
            IMG_SIZE[1],
            3,
        ),
        name="image_input",
    )

    image_features = base_model(
        image_input,
        training=False,
    )

    image_features = layers.Dropout(
        0.30
    )(
        image_features
    )

    # ========================================================================
    # 63-FEATURE BRANCH
    # ========================================================================

    feature_input = layers.Input(
        shape=(
            TOTAL_FEATURES,
        ),
        name="feature_input",
    )

    # ------------------------------------------------------------------------
    # Dense layer
    # ------------------------------------------------------------------------

    feature_features = layers.Dense(
        32,
        activation="relu",
        name="feature_dense_1",
    )(
        feature_input
    )

    feature_features = (
        layers.BatchNormalization(
            name="feature_batch_norm"
        )(
            feature_features
        )
    )

    feature_features = layers.Dropout(
        0.20
    )(
        feature_features
    )

    # ------------------------------------------------------------------------
    # Second dense layer
    # ------------------------------------------------------------------------

    feature_features = layers.Dense(
        16,
        activation="relu",
        name="feature_dense_2",
    )(
        feature_features
    )

    # ========================================================================
    # FEATURE FUSION
    # ========================================================================

    fused_features = (
        layers.Concatenate(
            name="feature_fusion"
        )(
            [
                image_features,
                feature_features,
            ]
        )
    )

    # ========================================================================
    # CLASSIFICATION HEAD
    # ========================================================================

    x = layers.Dense(
        64,
        activation="relu",
        name="fusion_dense_1",
    )(
        fused_features
    )

    x = layers.Dropout(
        0.40
    )(
        x
    )

    x = layers.Dense(
        32,
        activation="relu",
        name="fusion_dense_2",
    )(
        x
    )

    x = layers.Dropout(
        0.30
    )(
        x
    )

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="ripeness_output",
    )(
        x
    )

    # ========================================================================
    # MODEL
    # ========================================================================

    model = models.Model(
        inputs={
            "image_input": image_input,
            "feature_input": feature_input,
        },
        outputs=outputs,
        name=(
            "Hybrid_EfficientNetB0_"
            "ColourHistogram_"
            "Statistical_Ripeness"
        ),
    )

    # ========================================================================
    # PHASE 1 COMPILATION
    # ========================================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=PHASE1_LR
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return (
        model,
        base_model,
    )


# ============================================================================
# CALCULATE CLASS WEIGHTS
# ============================================================================

def calculate_class_weights(
    labels,
    num_classes,
):
    """
    Calculate balanced class weights.

    Formula:

        total_samples
        -------------------------
        num_classes * class_count
    """

    labels = np.asarray(
        labels
    )

    class_counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    total_samples = len(
        labels
    )

    class_weights = {}

    print(
        "\n===== CLASS DISTRIBUTION ====="
    )

    for class_index in range(
        num_classes
    ):

        count = int(
            class_counts[
                class_index
            ]
        )

        if count == 0:

            raise RuntimeError(
                f"Class index {class_index} "
                f"has ZERO training images."
            )

        weight = (
            total_samples
            / (
                num_classes
                * count
            )
        )

        class_weights[
            class_index
        ] = float(
            weight
        )

    return (
        class_counts,
        class_weights,
    )


# ============================================================================
# CHECK FIRST BATCH
# ============================================================================

def inspect_first_batch(
    train_sequence,
    class_names,
):
    """
    Inspect one training batch before model training.
    """

    print(
        "\n===== FIRST BATCH SANITY CHECK ====="
    )

    x_batch, y_batch = (
        train_sequence[0]
    )

    print(
        "Image input shape:",
        x_batch[
            "image_input"
        ].shape,
    )

    print(
        "Feature input shape:",
        x_batch[
            "feature_input"
        ].shape,
    )

    print(
        "Expected feature count:",
        TOTAL_FEATURES,
    )

    print(
        "Label shape:",
        y_batch.shape,
    )

    # ------------------------------------------------------------------------
    # First five feature vectors
    # ------------------------------------------------------------------------

    print(
        "\nFirst 5 feature vectors:"
    )

    for i in range(
        min(
            5,
            len(
                x_batch[
                    "feature_input"
                ]
            ),
        )
    ):

        print(
            x_batch[
                "feature_input"
            ][i]
        )

    # ------------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------------

    print(
        "\nFirst 10 labels:"
    )

    for i in range(
        min(
            10,
            len(y_batch),
        )
    ):

        label_index = int(
            np.argmax(
                y_batch[i]
            )
        )

        print(
            f"{i}: "
            f"{class_names[label_index]} "
            f"{y_batch[i]}"
        )

    # ------------------------------------------------------------------------
    # Input ranges
    # ------------------------------------------------------------------------

    image_min = np.min(
        x_batch[
            "image_input"
        ]
    )

    image_max = np.max(
        x_batch[
            "image_input"
        ]
    )

    feature_min = np.min(
        x_batch[
            "feature_input"
        ]
    )

    feature_max = np.max(
        x_batch[
            "feature_input"
        ]
    )

    print(
        "\nInput ranges:"
    )

    print(
        f"Image:   "
        f"{image_min:.2f} "
        f"-> "
        f"{image_max:.2f}"
    )

    print(
        f"Feature: "
        f"{feature_min:.4f} "
        f"-> "
        f"{feature_max:.4f}"
    )

    # ------------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------------

    if image_max <= 1.0:

        print(
            "\nWARNING:"
            "\nEfficientNet image input appears "
            "to be normalized to [0,1]."
            "\nThis version expects [0,255]."
        )

    if feature_max > 1.5:

        print(
            "\nWARNING:"
            "\nSome feature values appear "
            "not to be normalized."
        )

    print(
        "\n===== FIRST BATCH CHECK COMPLETE ====="
    )


# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(
    history_phase1,
    history_phase2,
):
    """
    Combine Phase 1 and Phase 2 histories.
    """

    acc = (
        history_phase1.history[
            "accuracy"
        ]
        +
        history_phase2.history[
            "accuracy"
        ]
    )

    val_acc = (
        history_phase1.history[
            "val_accuracy"
        ]
        +
        history_phase2.history[
            "val_accuracy"
        ]
    )

    loss = (
        history_phase1.history[
            "loss"
        ]
        +
        history_phase2.history[
            "loss"
        ]
    )

    val_loss = (
        history_phase1.history[
            "val_loss"
        ]
        +
        history_phase2.history[
            "val_loss"
        ]
    )

    fine_tune_start = len(
        history_phase1.history[
            "accuracy"
        ]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # ------------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------------

    axes[0].plot(
        acc,
        label="Training Accuracy",
    )

    axes[0].plot(
        val_acc,
        label="Validation Accuracy",
    )

    axes[0].axvline(
        fine_tune_start,
        color="gray",
        linestyle="--",
        label="Fine-tuning Start",
    )

    axes[0].set_title(
        "Hybrid Model Training vs Validation Accuracy"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Accuracy"
    )

    axes[0].legend()

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    axes[1].plot(
        loss,
        label="Training Loss",
    )

    axes[1].plot(
        val_loss,
        label="Validation Loss",
    )

    axes[1].axvline(
        fine_tune_start,
        color="gray",
        linestyle="--",
        label="Fine-tuning Start",
    )

    axes[1].set_title(
        "Hybrid Model Training vs Validation Loss"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Loss"
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        HISTORY_PLOT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nTraining curves saved to:\n"
        f"{HISTORY_PLOT_PATH}"
    )


# ============================================================================
# TRAINING
# ============================================================================

def train():

    # ------------------------------------------------------------------------
    # Create model directory
    # ------------------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "HYBRID MANGO RIPENESS CLASSIFIER"
    )

    print(
        "EfficientNetB0 + Colour Histogram + Statistical Features"
    )

    print(
        "Module 1 Preprocessing Integrated"
    )

    print(
        f"Total Feature Count: {TOTAL_FEATURES}"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------------
    # Dataset paths
    # ------------------------------------------------------------------------

    print(
        "\nDataset paths:"
    )

    print(f"Original labelled images: {DATASET_DIR}")

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nMango image directory not found:\n"
            f"{DATASET_DIR}"
        )

    # ------------------------------------------------------------------------
    # Discover classes
    # ------------------------------------------------------------------------

    class_names = discover_classes(DATASET_DIR)

    print(
        "\n===== CLASS ORDER ====="
    )

    for index, name in enumerate(
        class_names
    ):

        print(
            f"{index} -> {name}"
        )

    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            f"\nExpected {NUM_CLASSES} classes, "
            f"but found {len(class_names)}:\n"
            f"{class_names}"
        )

    # ------------------------------------------------------------------------
    # Save class mapping
    # ------------------------------------------------------------------------

    class_indices = {
        str(index): name
        for index, name in enumerate(
            class_names
        )
    }

    with open(
        CLASS_INDICES_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            class_indices,
            f,
            indent=4,
        )

    print(
        "\nClass mapping saved to:\n"
        f"{CLASS_INDICES_PATH}"
    )

    # ------------------------------------------------------------------------
    # Build leakage-safe file lists
    # ------------------------------------------------------------------------

    (
        train_files, train_labels,
        val_files, val_labels,
        test_files, test_labels,
    ) = build_stratified_splits(
        DATASET_DIR,
        class_names,
        include_augmented_train=True,
    )
    validate_image_files(train_files + val_files + test_files)

    if not train_files:

        raise RuntimeError(
            "No training images were found."
        )

    if not val_files:

        raise RuntimeError(
            "No validation images were found."
        )

    if not test_files:
        raise RuntimeError("No test images were found.")

    # ------------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------------

    (
        train_counts,
        class_weights,
    ) = calculate_class_weights(
        train_labels,
        len(class_names),
    )

    print(
        "\nTraining images:"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name}: "
            f"{train_counts[index]}"
        )

    print(
        "\n===== CLASS WEIGHTS ====="
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name}: "
            f"{class_weights[index]:.4f}"
        )

    # ------------------------------------------------------------------------
    # Validation distribution
    # ------------------------------------------------------------------------

    val_counts = np.bincount(
        np.asarray(
            val_labels
        ),
        minlength=len(
            class_names
        ),
    )

    print(
        "\n===== VALIDATION DISTRIBUTION ====="
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name}: "
            f"{val_counts[index]}"
        )

    test_counts = np.bincount(
        np.asarray(test_labels),
        minlength=len(class_names),
    )
    print("\n===== TEST DISTRIBUTION (HELD OUT) =====")
    for index, class_name in enumerate(class_names):
        print(f"{class_name}: {test_counts[index]}")

    # ------------------------------------------------------------------------
    # Build training sequence
    # ------------------------------------------------------------------------

    train_sequence = (
        HybridSequence(
            train_files,
            train_labels,
            num_classes=len(
                class_names
            ),
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            augment=True,
            shuffle=True,
        )
    )

    # ------------------------------------------------------------------------
    # Build validation sequence
    # ------------------------------------------------------------------------

    val_sequence = (
        HybridSequence(
            val_files,
            val_labels,
            num_classes=len(
                class_names
            ),
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            augment=False,
            shuffle=False,
        )
    )

    # ------------------------------------------------------------------------
    # Inspect first batch
    # ------------------------------------------------------------------------

    inspect_first_batch(
        train_sequence,
        class_names,
    )

    # ------------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------------

    print(
        "\n===== BUILDING HYBRID MODEL ====="
    )

    model, base_model = build_model(
        num_classes=len(
            class_names
        )
    )

    model.summary()

    # ------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                MODEL_OUTPUT_PATH
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # =========================================================================
    # PHASE 1
    # =========================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 1 — TRAIN CLASSIFICATION HEAD"
    )

    print(
        "EfficientNetB0 base: FROZEN"
    )

    print(
        "=" * 70
    )

    history_phase1 = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================================================================
    # PHASE 2
    # =========================================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 2 — FINE-TUNE EFFICIENTNETB0"
    )

    print(
        "=" * 70
    )

    base_model.trainable = True

    total_layers = len(
        base_model.layers
    )

    freeze_until = max(
        0,
        total_layers
        - FINE_TUNE_LAYERS,
    )

    # Freeze all layers first.
    for layer in base_model.layers:

        layer.trainable = False

    # Unfreeze only the final layers.
    for layer in base_model.layers[
        freeze_until:
    ]:

        layer.trainable = True

    print(
        f"Total EfficientNet layers: "
        f"{total_layers}"
    )

    print(
        f"Fine-tuning last "
        f"{FINE_TUNE_LAYERS} layers."
    )

    # ------------------------------------------------------------------------
    # Recompile
    # ------------------------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=PHASE2_LR
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy"
        ],
    )

    history_phase2 = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================

    model.save(
        MODEL_OUTPUT_PATH
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nModel saved to:\n"
        f"{MODEL_OUTPUT_PATH}"
    )

    print(
        "\nClass mapping:\n"
        f"{CLASS_INDICES_PATH}"
    )

    # =========================================================================
    # TRAINING CURVES
    # =========================================================================

    plot_training_history(
        history_phase1,
        history_phase2,
    )

    # =========================================================================
    # FINAL TRAINING SUMMARY
    # =========================================================================

    best_val_acc_phase1 = max(
        history_phase1.history[
            "val_accuracy"
        ]
    )

    best_val_acc_phase2 = max(
        history_phase2.history[
            "val_accuracy"
        ]
    )

    print(
        "\n===== TRAINING SUMMARY ====="
    )

    print(
        "Best Phase 1 validation accuracy: "
        f"{best_val_acc_phase1 * 100:.2f}%"
    )

    print(
        "Best Phase 2 validation accuracy: "
        f"{best_val_acc_phase2 * 100:.2f}%"
    )

    print(
        "\nFeature configuration:"
    )

    print(
        f"HSV statistics: "
        f"{NUM_HSV_STAT_FEATURES}"
    )

    print(
        f"Colour histogram: "
        f"{NUM_HISTOGRAM_FEATURES}"
    )

    print(
        f"Statistical analysis: "
        f"{NUM_STATISTICAL_FEATURES}"
    )

    print(
        f"Total features: "
        f"{TOTAL_FEATURES}"
    )

    print(
        "\nNext step:"
    )

    print(
        "python training/evaluate_model.py"
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    train()
