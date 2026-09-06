"""Decode uploaded photos without exposing colours hidden by transparency."""

import cv2
import numpy as np


def decode_uploaded_image(data: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(data, dtype=np.uint8)
    if not encoded.size:
        return None
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3 and raw.shape[2] == 4:
        # Composite on white before segmentation. RGB values beneath alpha=0
        # are arbitrary and must never become visible classifier evidence.
        maximum = np.iinfo(raw.dtype).max
        alpha = raw[:, :, 3:4].astype(np.float32) / maximum
        colour = raw[:, :, :3].astype(np.float32) * (255.0 / maximum)
        return np.rint(colour * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    # Retain OpenCV's existing orientation and 8-bit conversion for JPEGs
    # and other opaque uploads.
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
