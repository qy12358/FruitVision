"""
training/evaluate_model.py

Evaluates the trained HYBRID:

    EfficientNetB0 + HSV statistical features

ripeness classifier on the held-out Test set.

Reports:

- Overall Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Full sklearn classification report
- Confusion Matrix
- Ground-truth distribution
- Prediction distribution
- Automatic warning for strong single-class bias

The Test set uses the same HybridSequence as training,
which means it uses the same Module 1 preprocessing pipeline.

Run:

    python training/evaluate_model.py
"""

import json
import sys
from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.train_model import (
    HybridSequence,
    build_file_label_list,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "Test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_fruit.keras"
)

CLASS_INDICES_PATH = (
    PROJECT_ROOT
    / "models"
    / "class_indices.json"
)

CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "models"
    / "confusion_matrix.png"
)

# If more than 85% of all predictions are assigned to
# one class, report a possible bias warning.

BIAS_WARNING_THRESHOLD = 0.85


# ============================================================================
# LOAD CLASS NAMES
# ============================================================================

def load_class_names() -> list:
    """
    Load class names from class_indices.json.

    No class names are hard-coded here.
    """

    if not CLASS_INDICES_PATH.exists():

        raise FileNotFoundError(
            f"'{CLASS_INDICES_PATH}' not found.\n"
            f"Run training/train_model.py first."
        )

    with open(
        CLASS_INDICES_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        raw_mapping = json.load(f)

    class_names = [
        raw_mapping[str(i)]
        for i in range(
            len(raw_mapping)
        )
    ]

    return class_names


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate():

    print(
        "=" * 70
    )

    print(
        "HYBRID MANGO RIPENESS MODEL EVALUATION"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------------------
    # Validate model
    # ------------------------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}\n\n"
            f"Run training/train_model.py first."
        )

    # ------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------

    print(
        f"\nLoading model from:\n"
        f"{MODEL_PATH}"
    )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    # ------------------------------------------------------------------------
    # Load class mapping
    # ------------------------------------------------------------------------

    class_names = (
        load_class_names()
    )

    print(
        "\nClass order "
        f"(from {CLASS_INDICES_PATH}):"
    )

    for index, name in enumerate(
        class_names
    ):

        print(
            f"  {index} -> {name}"
        )

    # ------------------------------------------------------------------------
    # Validate Test directory
    # ------------------------------------------------------------------------

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{TEST_DIR}"
        )

    # ------------------------------------------------------------------------
    # Build Test file list
    # ------------------------------------------------------------------------

    test_files, test_labels = (
        build_file_label_list(
            TEST_DIR,
            class_names,
        )
    )

    if not test_files:

        raise RuntimeError(
            f"No test images found under:\n"
            f"{TEST_DIR}"
        )

    # ------------------------------------------------------------------------
    # Ground-truth distribution
    # ------------------------------------------------------------------------

    print(
        "\n===== GROUND-TRUTH DISTRIBUTION ====="
    )

    gt_counts = {
        name: test_labels.count(index)
        for index, name in enumerate(
            class_names
        )
    }

    for name, count in (
        gt_counts.items()
    ):

        print(
            f"  {name}: {count}"
        )

    # ------------------------------------------------------------------------
    # Create test sequence
    #
    # HybridSequence now performs the same Module 1 preprocessing
    # used during training and inference.
    # ------------------------------------------------------------------------

    test_seq = HybridSequence(
        test_files,
        test_labels,
        num_classes=len(
            class_names
        ),
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        augment=False,
        shuffle=False,
    )

    # ------------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------------

    print(
        "\nRunning inference on Test set..."
    )

    probs = model.predict(
        test_seq,
        verbose=1,
    )

    # Safety trim.
    probs = probs[
        :len(test_labels)
    ]

    y_pred = np.argmax(
        probs,
        axis=1,
    )

    # shuffle=False guarantees that this
    # remains aligned with predictions.
    y_true = np.asarray(
        test_labels
    )

    # ------------------------------------------------------------------------
    # Prediction distribution
    # ------------------------------------------------------------------------

    print(
        "\n===== PREDICTION DISTRIBUTION ====="
    )

    pred_counts = {}

    for index, name in enumerate(
        class_names
    ):

        count = int(
            (
                y_pred == index
            ).sum()
        )

        pred_counts[name] = count

        print(
            f"  {name}: {count}"
        )

    # ------------------------------------------------------------------------
    # Bias warning
    # ------------------------------------------------------------------------

    if len(y_pred) > 0:

        dominant_class = max(
            pred_counts,
            key=pred_counts.get,
        )

        dominant_share = (
            pred_counts[
                dominant_class
            ]
            / len(y_pred)
        )

        if (
            dominant_share
            > BIAS_WARNING_THRESHOLD
        ):

            print(
                "\nWARNING: Model is strongly "
                "biased toward one class."
            )

            print(
                f"  '{dominant_class}' accounts "
                f"for {dominant_share * 100:.1f}% "
                f"of all predictions."
            )

            print(
                "\nCheck:"
            )

            print(
                "  1. Training class distribution"
            )

            print(
                "  2. Class weights"
            )

            print(
                "  3. Training/inference preprocessing"
            )

            print(
                "  4. Test dataset quality"
            )

    # =========================================================================
    # CORE METRICS
    # =========================================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        "\n===== TEST SET PERFORMANCE ====="
    )

    print(
        f"Accuracy:  {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall:    {recall * 100:.2f}%"
    )

    print(
        f"F1-score:  {f1 * 100:.2f}%"
    )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(
        "\n===== CLASSIFICATION REPORT ====="
    )

    print(
        report
    )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(class_names))
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    print(
        "\n===== CONFUSION MATRIX ====="
    )

    print(
        cm_df
    )

    # ------------------------------------------------------------------------
    # Save heatmap
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=True,
    )

    plt.title(
        "Confusion Matrix — "
        "Hybrid Ripeness Classification"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nConfusion matrix heatmap saved to:\n"
        f"{CONFUSION_MATRIX_PATH}"
    )

    # =========================================================================
    # RETURN RESULTS
    # =========================================================================

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm_df,
        "classification_report": report,
        "ground_truth_distribution": gt_counts,
        "prediction_distribution": pred_counts,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    evaluate()