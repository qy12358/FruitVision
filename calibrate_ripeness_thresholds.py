"""
Calibrates ripeness thresholds from a fruit image dataset (e.g. Fruit-360:
https://github.com/fruits-360/) WITHOUT relying on any ripeness label.

WHY THIS VERSION IS DIFFERENT
------------------------------
Fruit-360's class folders are organised mostly by *variety*
("Apple Braeburn", "Banana", "Tomato 4", ...), not by ripeness stage.
Only a handful of folders (e.g. "Tomato not Ripened") happen to encode a
ripeness word, so calibrating only on those folders would (a) barely use
any of the dataset and (b) make the whole ripeness system secretly
dependent on the presence of that rare naming convention - which defeats
the point of a general, dataset-agnostic classifier.

Instead, this script treats ripeness calibration as an UNSUPERVISED
problem, the same way Otsu's method (already used elsewhere in this
project for defect segmentation) finds a threshold by maximising
between-class separation instead of being told which pixels are which:

    1. Read every image in the dataset, grouped only by FRUIT TYPE
       (inferred from the folder name - never by ripeness word).
    2. Run the SAME Module 1 preprocessing pipeline used at inference time
       (ImagePreprocessor: resize -> BGR2HSV -> Gaussian filter), then the
       Module 2 feature pipeline (segment_foreground + extract_hsv_features)
       on the Gaussian-filtered HSV output - exactly what app.py hands to
       RipenessClassifier.analyze() at inference time. Calibrating on raw,
       un-preprocessed HSV would fit thresholds to a different feature
       distribution than the one the classifier actually sees in
       production, so this step matters. No label is attached to the result.
    3. For each fruit type, run 1-D k-means (k=2) separately on:
         - dark_pixel_ratio values   -> overripe_dark_ratio threshold
         - brown_pixel_ratio values  -> overripe_brown_ratio threshold
       The higher-valued cluster in each becomes the "Overripe-looking"
       group; the threshold is the midpoint between the two cluster
       centres. This is a direct extension of Otsu-style variance
       maximisation to 1-D colour statistics.
    4. Remove the samples flagged as Overripe-looking, then run 1-D
       k-means (k=2) on the mean_hue of the remaining samples. The
       higher-hue cluster is the "Unripe-colour" group (still green),
       the lower-hue cluster is the "Ripe-colour" group (yellow/
       orange/red). unripe_hue_min is the midpoint between the two
       cluster centres; ripe_hue_min is the 5th percentile of the
       Ripe-colour cluster (a lower guard-band).

The output ripeness_thresholds.json has exactly the same shape as
before, so RipenessClassifier and app.py need no changes.

Usage:
    python calibrate_ripeness_thresholds.py \
        --data_dir "path/to/fruits-360/Training" \
        --out ripeness_thresholds.json
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from modules.preprocessing import ImagePreprocessor
from modules.ripeness import RipenessClassifier

FRUIT_KEYWORDS = ["apple", "banana", "mango", "tomato", "orange"]

# Minimum number of usable images before we trust a data-driven threshold
# for a fruit type. Below this we fall back to DEFAULT_THRESHOLDS rather
# than calibrate on too few samples.
MIN_SAMPLES = 40


def infer_fruit_type(folder_name):
    """Fruit type only - never ripeness. Used purely to group images so a
    banana's colour statistics aren't mixed in with an apple's."""
    name = folder_name.lower()
    for fruit in FRUIT_KEYWORDS:
        if fruit in name:
            return fruit
    return "default"


def kmeans_1d(values, k=2, iters=200, seed=42):
    """Minimal 1-D k-means (no sklearn dependency).

    Used instead of manual/guessed thresholds: cluster centres emerge
    purely from the shape of the data's own distribution, mirroring how
    Otsu's method picks a threshold by maximising between-class variance.

    Returns (sorted_centers_ascending, labels) where labels index into
    sorted_centers_ascending.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return None, None
    if n < k:
        # not enough points to cluster meaningfully
        centers = np.sort(values)
        return centers, np.arange(n)

    rng = np.random.default_rng(seed)
    # init centers at evenly spaced percentiles for stability/reproducibility
    centers = np.percentile(values, np.linspace(10, 90, k))

    for _ in range(iters):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            pts = values[labels == i]
            if len(pts) > 0:
                new_centers[i] = pts.mean()
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    order = np.argsort(centers)
    sorted_centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([remap[l] for l in labels])
    return sorted_centers, sorted_labels


def calibrate_overripe_threshold(ratio_values, default_value):
    """Splits a dark/brown ratio distribution into 'normal' vs
    'overripe-looking' via 1-D k-means, and returns the midpoint between
    the two cluster centres as the decision threshold."""
    if len(ratio_values) < MIN_SAMPLES:
        return default_value, np.zeros(len(ratio_values), dtype=bool)

    centers, labels = kmeans_1d(ratio_values, k=2)
    if centers is None or len(centers) < 2:
        return default_value, np.zeros(len(ratio_values), dtype=bool)

    threshold = float((centers[0] + centers[1]) / 2)
    is_overripe = labels == 1  # cluster 1 = higher-valued (sorted ascending)
    return threshold, is_overripe


def calibrate_hue_thresholds(hue_values, default_unripe, default_ripe):
    """Splits remaining (non-overripe) hue values into Unripe-colour vs
    Ripe-colour clusters via 1-D k-means."""
    if len(hue_values) < MIN_SAMPLES:
        return default_unripe, default_ripe

    centers, labels = kmeans_1d(hue_values, k=2)
    if centers is None or len(centers) < 2:
        return default_unripe, default_ripe

    ripe_cluster_vals = hue_values[labels == 0]   # lower-hue cluster
    unripe_hue_min = float((centers[0] + centers[1]) / 2)
    ripe_hue_min = float(np.percentile(ripe_cluster_vals, 5)) if len(ripe_cluster_vals) else default_ripe
    return round(unripe_hue_min, 1), round(max(ripe_hue_min, 0), 1)


def main(data_dir, out_path, max_images_per_class):
    preprocessor = ImagePreprocessor()   # Module 1 - same pipeline as app.py
    classifier = RipenessClassifier()    # Module 2

    # fruit_type -> lists of colour features gathered from every image,
    # with NO ripeness label attached at any point.
    stats = {}

    class_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(class_dirs)} class folders in {data_dir}")

    for class_dir in class_dirs:
        folder_name = os.path.basename(class_dir)
        fruit_type = infer_fruit_type(folder_name)

        image_paths = (glob.glob(os.path.join(class_dir, "*.jpg")) +
                       glob.glob(os.path.join(class_dir, "*.png")))[:max_images_per_class]

        bucket = stats.setdefault(fruit_type, {"mean_hue": [], "dark_ratio": [], "brown_ratio": []})

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # --- Module 1: identical preprocessing to app.py -------------
            # (resize -> BGR2HSV -> Gaussian filter). We deliberately use
            # "gaussian", not "equalized": app.py comments explain that
            # ripeness features are extracted on the denoised-but-not-yet-
            # equalised HSV image, since equalisation would distort the
            # V-channel statistics the ripeness rules rely on.
            pre_result = preprocessor.preprocess(img)
            hsv_for_ripeness = pre_result["gaussian"]

            # --- Module 2: same feature extraction as inference time -----
            mask = classifier.segment_foreground(hsv_for_ripeness)
            feats = classifier.extract_hsv_features(hsv_for_ripeness, mask)

            bucket["mean_hue"].append(feats["mean_hue"])
            bucket["dark_ratio"].append(feats["dark_pixel_ratio"])
            bucket["brown_ratio"].append(feats["brown_pixel_ratio"])

        print(f"  {folder_name} -> fruit={fruit_type}, n_images={len(image_paths)}")

    thresholds = {}
    for fruit_type, feats in stats.items():
        mean_hue = np.array(feats["mean_hue"])
        dark_ratio = np.array(feats["dark_ratio"])
        brown_ratio = np.array(feats["brown_ratio"])

        if len(mean_hue) < MIN_SAMPLES:
            print(f"  [skip] {fruit_type}: only {len(mean_hue)} samples (< {MIN_SAMPLES}); "
                  f"keeping DEFAULT_THRESHOLDS for this fruit type.")
            continue

        # Step A: unsupervised overripe split (dark ratio, brown ratio independently)
        overripe_dark_th, overripe_by_dark = calibrate_overripe_threshold(dark_ratio, 0.12)
        overripe_brown_th, overripe_by_brown = calibrate_overripe_threshold(brown_ratio, 0.18)
        is_overripe = overripe_by_dark | overripe_by_brown

        # Step B: unsupervised unripe/ripe split on the remaining hue values
        remaining_hue = mean_hue[~is_overripe]
        unripe_hue_min, ripe_hue_min = calibrate_hue_thresholds(remaining_hue, 40.0, 10.0)

        thresholds[fruit_type] = {
            "unripe_hue_min": unripe_hue_min,
            "ripe_hue_min": ripe_hue_min,
            "overripe_dark_ratio": round(overripe_dark_th, 3),
            "overripe_brown_ratio": round(overripe_brown_th, 3),
        }

    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nCalibrated thresholds written to {out_path}")
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to Fruit-360 Training folder")
    parser.add_argument("--out", default="ripeness_thresholds.json")
    parser.add_argument("--max_images_per_class", type=int, default=200)
    args = parser.parse_args()
    main(args.data_dir, args.out, args.max_images_per_class)