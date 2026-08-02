"""
Module 2: Ripeness Classification
----------------------------------
Analyses the colour characteristics of the segmented fruit using HSV
features and classifies it as Unripe / Ripe / Overripe using rule-based
thresholds on those colour features.

IMPORTANT (design note):
This classifier NEVER looks at dataset labels. It only ever sees a raw
HSV image. The three-way decision (Unripe/Ripe/Overripe) is made purely
from measured colour statistics (mean Hue, dark-pixel ratio, brown-pixel
ratio), the same way a human inspector reads colour. This is what lets
the system run on Fruit-360 (or any other unlabeled fruit image set)
without needing per-image ripeness ground truth.

The *thresholds* used below (unripe_hue_min, ripe_hue_min,
overripe_dark_ratio, overripe_brown_ratio) are not hand-guessed: they are
produced by calibrate_ripeness_thresholds.py, which derives them from the
natural clustering of colour statistics across the dataset (see that
file's docstring). If no calibration file is present, DEFAULT_THRESHOLDS
below is used as a literature-informed fallback.

Pipeline:
    1. Foreground segmentation  - removes the plain background typical of
       Fruit-360 images so only fruit pixels feed the statistics.
    2. HSV colour feature extraction - mean/std of H, S, V.
    3. Colour histogram analysis - Hue histogram -> dominant hue bin,
       proportion of green / yellow-orange / red / dark-brown pixels.
    4. Statistical feature analysis - mean, std, skewness of Hue.
    5. Rule-based classification - thresholds loaded from a JSON file
       produced by calibrate_ripeness_thresholds.py (falls back to
       reasonable defaults if no calibration file is supplied).
"""

import json
import os
import numpy as np
import cv2

# Fallback thresholds (used until calibrate_ripeness_thresholds.py has been
# run on a dataset subset and produced ripeness_thresholds.json). These are
# literature-informed starting points, not fitted to any labelled data.
#
# IMPORTANT - "same formula, different numbers":
# Every fruit type below is judged by the exact same rule (see classify()):
#   1. mean_hue >= unripe_hue_min                              -> Unripe
#   2. ripe_hue_min <= mean_hue < unripe_hue_min                -> Ripe
#   3. dark_ratio >= overripe_dark_ratio OR
#      brown_ratio >= overripe_brown_ratio (checked first)      -> Overripe
# Only the four numbers change per fruit, not the decision logic. This is
# because every fruit ripens through the same underlying HSV mechanism
# (chlorophyll breakdown shifts Hue from green towards yellow/orange/red;
# over-maturation increases dark/brown surface area from bruising, mould,
# or skin browning) - but WHERE that shift happens on the Hue scale, and
# HOW MUCH browning counts as "overripe", is fruit-specific. Below is the
# physical reasoning behind each fruit's numbers:
#
#   default (fallback for any fruit type not explicitly listed):
#     Generic green->red/yellow progression; conservative middle-of-the-
#     road values used when no fruit-specific evidence is available.
#
#   banana:
#     Unripe = green skin, Hue stays high (>=45). Ripe = skin turns
#     yellow, Hue drops into the 20-45 band. Overripe = banana skins
#     develop dark brown/black spots very early and very visibly compared
#     to other fruits, so its dark/brown ratio thresholds (0.10 / 0.15)
#     are set LOWER than apple/mango/orange - a banana needs less dark
#     surface area than an apple before it's called Overripe.
#
#   mango:
#     Unripe = green skin, Hue >=42. Ripe = skin turns yellow/orange,
#     Hue in 10-42. Overripe = mango bruises and internal breakdown show
#     as dark patches and a notably higher brown ratio, so
#     overripe_brown_ratio (0.20) is set HIGHER than the other fruits -
#     mango skin naturally keeps some darker blemish-like patches even
#     when only Ripe, so the bar for "Overripe" brown coverage is raised
#     to avoid false positives.
#
#   tomato:
#     Unripe = green skin, Hue >=35. Ripe = skin turns deep red, which
#     sits at the LOW end of the Hue wheel (near 0/180 wraparound), so
#     ripe_hue_min is set to 0 - a fully red tomato's mean Hue can be
#     right at the bottom of the scale and should still count as Ripe,
#     not fall through to the Overripe default case. Overripe = mould/
#     rot spots, standard dark/brown thresholds (0.10 / 0.18).
#
#   apple:
#     Unripe = green-skinned cultivars, Hue >=38. Ripe = red/yellow
#     cultivars, Hue in 0-38 (apples span a wide ripe-colour range across
#     cultivars, hence the wide band down to 0). Overripe = bruising and
#     skin browning, standard thresholds (0.12 / 0.18).
#
#   orange:
#     Unripe = green rind, Hue >=40. Ripe = orange rind, Hue in 8-40.
#     Oranges change hue LESS dramatically than banana/mango/tomato once
#     ripe (rind colour is already close to "orange" for much of
#     maturation), so more weight falls on the dark/brown ratio checks
#     (mould spots, rind staining) to catch Overripe cases that a pure
#     Hue-based rule would miss; thresholds kept at the standard
#     0.12 / 0.18 since orange rind browning behaves similarly to apple.
DEFAULT_THRESHOLDS = {
    "default": {"unripe_hue_min": 40, "ripe_hue_min": 15,
                "overripe_dark_ratio": 0.12, "overripe_brown_ratio": 0.18},
    "banana":  {"unripe_hue_min": 45, "ripe_hue_min": 20,
                "overripe_dark_ratio": 0.10, "overripe_brown_ratio": 0.15},
    "mango":   {"unripe_hue_min": 42, "ripe_hue_min": 10,
                "overripe_dark_ratio": 0.12, "overripe_brown_ratio": 0.20},
    "tomato":  {"unripe_hue_min": 35, "ripe_hue_min": 0,
                "overripe_dark_ratio": 0.10, "overripe_brown_ratio": 0.18},
    "apple":   {"unripe_hue_min": 38, "ripe_hue_min": 0,
                "overripe_dark_ratio": 0.12, "overripe_brown_ratio": 0.18},
    "orange":  {"unripe_hue_min": 40, "ripe_hue_min": 8,
                "overripe_dark_ratio": 0.12, "overripe_brown_ratio": 0.18},
}


class RipenessClassifier:
    def __init__(self, thresholds_path=None, bins=30):
        self.bins = bins
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds_path and os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                loaded = json.load(f)
            self.thresholds.update(loaded)  # calibrated values override defaults

    # ---------------- 1. Foreground segmentation ----------------
    def segment_foreground(self, hsv_image, s_thresh=25, v_thresh=200):
        """Removes the plain white/light background typical of Fruit-360 images."""
        h, s, v = cv2.split(hsv_image)
        background = (s < s_thresh) & (v > v_thresh)
        mask = (~background).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ---------------- 2-4. Feature extraction + histogram + stats ----------------
    def extract_hsv_features(self, hsv_image, mask=None):
        h, s, v = cv2.split(hsv_image)

        if mask is not None:
            fg = mask > 0
            if fg.sum() == 0:          # segmentation wiped everything -> fallback
                fg = np.ones_like(mask, dtype=bool)
        else:
            fg = np.ones(h.shape, dtype=bool)

        h_vals = h[fg].astype(np.float32)
        s_vals = s[fg].astype(np.float32)
        v_vals = v[fg].astype(np.float32)

        # Mean / Std HSV (statistical feature analysis)
        mean_h, std_h = float(np.mean(h_vals)), float(np.std(h_vals))
        mean_s, std_s = float(np.mean(s_vals)), float(np.std(s_vals))
        mean_v, std_v = float(np.mean(v_vals)), float(np.std(v_vals))
        skew_h = self._skewness(h_vals)

        # Colour histogram analysis
        hist_h, _ = np.histogram(h_vals, bins=self.bins, range=(0, 180))
        hist_h_norm = hist_h / max(hist_h.sum(), 1)
        dominant_hue = int(np.argmax(hist_h_norm)) * (180 / self.bins)

        total = max(len(h_vals), 1)
        green_ratio = float(np.sum((h_vals >= 35) & (h_vals <= 85)) / total)
        yellow_orange_ratio = float(np.sum((h_vals >= 10) & (h_vals < 35)) / total)
        red_ratio = float(np.sum((h_vals < 10) | (h_vals > 160)) / total)
        dark_ratio = float(np.sum(v_vals < 60) / total)
        brown_ratio = float(np.sum((s_vals < 100) & (v_vals < 120)) / total)

        return {
            "mean_hue": mean_h, "std_hue": std_h,
            "mean_saturation": mean_s, "std_saturation": std_s,
            "mean_value": mean_v, "std_value": std_v,
            "skew_hue": skew_h,
            "hist_hue": hist_h_norm.tolist(),
            "dominant_hue": dominant_hue,
            "green_ratio": green_ratio,
            "yellow_orange_ratio": yellow_orange_ratio,
            "red_ratio": red_ratio,
            "dark_pixel_ratio": dark_ratio,
            "brown_pixel_ratio": brown_ratio,
        }

    @staticmethod
    def _skewness(values):
        if len(values) < 2:
            return 0.0
        mean, std = np.mean(values), np.std(values)
        if std == 0:
            return 0.0
        return float(np.mean(((values - mean) / std) ** 3))

    # ---------------- 5. Rule-based classification ----------------
    def classify(self, features, fruit_type="default"):
        """
        Fruit-specific, dataset-calibrated rule-based classification.

        SAME FORMULA FOR EVERY FRUIT TYPE - only the threshold VALUES in
        `th` differ (looked up per fruit_type; see the comments above
        DEFAULT_THRESHOLDS for the physical reasoning behind each fruit's
        numbers, and calibrate_ripeness_thresholds.py for how `th` is
        statistically derived from the dataset when calibration data is
        available).

        Decision order (checked top to bottom, first match wins):

          1. OVERRIPE check (checked FIRST, before Hue):
             dark_ratio >= overripe_dark_ratio  OR
             brown_ratio >= overripe_brown_ratio
             -> physically: enough of the fruit's surface is dark/brown
                (bruising, mould, skin browning) that it is called
                Overripe regardless of what the Hue says - a fruit that
                LOOKS ripe in colour but is heavily spotted/bruised
                should still be flagged Overripe, so this check overrides
                the Hue-based checks below.

          2. UNRIPE check:
             mean_hue >= unripe_hue_min
             -> physically: the fruit's average colour is still closer
                to green than to its ripe colour (chlorophyll hasn't
                broken down enough yet) -> Unripe.

          3. RIPE check:
             ripe_hue_min <= mean_hue < unripe_hue_min
             -> physically: the fruit's average colour has moved into
                its expected ripe colour band (yellow/orange/red,
                depending on fruit) without yet showing enough dark/
                brown surface area to be Overripe -> Ripe.

          4. Fallback:
             mean_hue < ripe_hue_min and none of the above matched
             -> treated as Overripe (very low Hue with no clear ripe-band
                match usually indicates an extreme colour reading, e.g.
                very dark/discoloured surface, so the conservative
                Overripe call is used).

        Confidence score: NOT a fixed number - it scales with how far the
        measured feature sits past the relevant threshold (the "margin"),
        clipped to [60, 99]. A fruit just barely over a threshold gets a
        lower confidence than one deep inside a class's typical range.
        """
        th = self.thresholds.get(fruit_type.lower(), self.thresholds["default"])

        mean_h = features["mean_hue"]
        dark_ratio = features["dark_pixel_ratio"]
        brown_ratio = features["brown_pixel_ratio"]

        if dark_ratio >= th["overripe_dark_ratio"] or brown_ratio >= th["overripe_brown_ratio"]:
            label = "Overripe"
            margin = max(dark_ratio - th["overripe_dark_ratio"],
                         brown_ratio - th["overripe_brown_ratio"])
            confidence = float(np.clip(0.70 + margin * 2, 0.60, 0.99))

        elif mean_h >= th["unripe_hue_min"]:
            label = "Unripe"
            margin = (mean_h - th["unripe_hue_min"]) / 30
            confidence = float(np.clip(0.65 + margin, 0.60, 0.99))

        elif th["ripe_hue_min"] <= mean_h < th["unripe_hue_min"]:
            label = "Ripe"
            band = max(th["unripe_hue_min"] - th["ripe_hue_min"], 1)
            center = (th["unripe_hue_min"] + th["ripe_hue_min"]) / 2
            margin = 1 - abs(mean_h - center) / (band / 2)
            confidence = float(np.clip(0.65 + margin * 0.3, 0.60, 0.99))

        else:
            label = "Overripe"
            confidence = 0.65

        return label, round(confidence * 100, 1)

    # ---------------- Convenience: run the whole module in one call ----------------
    def analyze(self, hsv_image, fruit_type="default", use_segmentation=True):
        mask = self.segment_foreground(hsv_image) if use_segmentation else None
        features = self.extract_hsv_features(hsv_image, mask)
        label, confidence = self.classify(features, fruit_type)
        return {"ripeness": label, "confidence": confidence, "features": features, "mask": mask}