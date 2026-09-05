"""services / histograms for ManGo or Stay."""

import numpy as np

def format_class_label(raw_name: str) -> str:
    """
    Cosmetic only: turns a raw dataset folder / model class name like
    'semi_ripe' into a display-friendly 'Semi-Ripe'. Never used for any
    classification logic — purely for badges/labels in the UI.
    """
    return raw_name.replace("_", "-").replace(" ", "-").title()

def split_histogram_dict(colour_histogram: dict):
    """
    modules.ripeness_classifier returns the 48-bin colour histogram as a
    flat dict {"hist_h_0": ..., "hist_h_1": ..., ..., "hist_v_15": ...}.
    Split it back into three 16-value lists (one per HSV channel) for
    charting.
    """
    h_vals = [colour_histogram[f"hist_h_{i}"] for i in range(16)]
    s_vals = [colour_histogram[f"hist_s_{i}"] for i in range(16)]
    v_vals = [colour_histogram[f"hist_v_{i}"] for i in range(16)]
    return h_vals, s_vals, v_vals

def histogram_profile(values):
    """Return dominant-bin and weighted-centre information for a histogram."""
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if arr.size == 0 or total <= 0:
        return {
            "dominant_bin": 0,
            "dominant_pct": 0.0,
            "weighted_bin": 0.0,
        }

    proportions = arr / total
    dominant_bin = int(np.argmax(proportions))
    return {
        "dominant_bin": dominant_bin,
        "dominant_pct": float(proportions[dominant_bin] * 100.0),
        "weighted_bin": float(np.dot(np.arange(arr.size), proportions)),
    }

def hue_bin_description(bin_index: int, bins: int = 16) -> str:
    """Describe one OpenCV H histogram bin in human-readable hue degrees."""
    low_cv = 180.0 * bin_index / bins
    high_cv = 180.0 * (bin_index + 1) / bins
    low_deg = low_cv * 2.0
    high_deg = high_cv * 2.0
    centre = (low_deg + high_deg) / 2.0

    if centre < 15 or centre >= 345:
        family = "red"
    elif centre < 45:
        family = "orange"
    elif centre < 75:
        family = "yellow"
    elif centre < 105:
        family = "yellow-green"
    elif centre < 165:
        family = "green"
    elif centre < 195:
        family = "cyan"
    elif centre < 255:
        family = "blue"
    elif centre < 285:
        family = "violet"
    elif centre < 345:
        family = "magenta"
    else:
        family = "red"

    return f"bin {bin_index} (about {low_deg:.0f}°–{high_deg:.0f}°, {family})"

def sv_bin_description(bin_index: int, channel_name: str, bins: int = 16) -> str:
    """Describe an S/V histogram bin as an approximate percentage range."""
    low_pct = 100.0 * bin_index / bins
    high_pct = 100.0 * (bin_index + 1) / bins
    return f"bin {bin_index} (about {low_pct:.0f}%–{high_pct:.0f}% {channel_name.lower()})"

def weighted_level(weighted_bin: float, kind: str) -> str:
    """Convert an HSV histogram centre into a simple user-friendly description."""
    ratio = weighted_bin / 15.0
    if kind == "saturation":
        if ratio < 0.33:
            return "mostly low / muted saturation"
        if ratio < 0.66:
            return "mostly medium saturation"
        return "mostly high / vivid saturation"

    if ratio < 0.33:
        return "mostly dark / low brightness"
    if ratio < 0.66:
        return "mostly medium brightness"
    return "mostly bright / high brightness"

def ripeness_histogram_guidance(prediction: str) -> str:
    """Explain the current HSV pattern without treating it as a hard ripeness rule."""
    guidance = {
        "Unripe": (
            "Unripe mangoes may contain stronger green-region hue information because "
            "chlorophyll is still prominent. Harumanis skin can remain green even when "
            "ripe, so the system does not use green colour alone to decide that a mango "
            "is unripe."
        ),
        "Semi-Ripe": (
            "A semi-ripe mango is in transition, so its colour distribution can contain "
            "a mixture of hue regions together with intermediate saturation and brightness. "
            "The exact pattern can vary between individual mangoes and lighting conditions."
        ),
        "Ripe": (
            "A ripe Harumanis mango can show changes in hue, colour intensity and brightness, "
            "but its skin may still remain green. For this reason, the system considers the "
            "complete HSV distribution together with the image features instead of using a "
            "simple rule such as 'yellow = ripe'."
        ),
        "Rotten": (
            "Deteriorated or damaged areas can introduce darker, brownish or uneven regions. "
            "These areas may reduce Value (brightness) or spread the colour distribution over "
            "more bins. Visible defects can therefore affect the histogram as well."
        ),
    }
    return guidance.get(
        prediction,
        "The HSV chart is supporting information. The final ripeness result is produced by "
        "the trained model using the complete image and colour-feature pattern.",
    )

