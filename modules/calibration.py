"""Optional spatial calibration helpers.

Physical measurements are valid only when a known reference dimension in the
same image plane is supplied. This module never invents a pixels-to-mm factor.
"""
from __future__ import annotations


def pixels_per_mm(reference_pixels: float, reference_mm: float) -> float:
    if reference_pixels <= 0 or reference_mm <= 0:
        raise ValueError("Reference pixel length and physical length must both be positive.")
    return reference_pixels / reference_mm


def pixels_to_mm(pixel_length: float, calibration_factor: float | None) -> float | None:
    if calibration_factor is None:
        return None
    if calibration_factor <= 0:
        raise ValueError("Calibration factor must be positive.")
    return pixel_length / calibration_factor