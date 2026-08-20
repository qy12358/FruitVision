"""'

FruitVision AI
Module 3 — Mango Blemish & Damage Detection

Detects:
- Dark/black spots
- Brown lesions
- Rot-like regions
- Scratches
- Surface damage

Attempts to ignore:
- Mango background
- Shadows
- Specular highlights
- Normal pale lenticels
- Tiny image noise
- Mango boundary
"""

import cv2
import numpy as np
from typing import Dict

class BlemishDetector:
    def __init__(
        self,
        min_blemish_area=8,
        boundary_erosion=15
    ):
        self.min_blemish_area = min_blemish_area
        self.boundary_erosion = boundary_erosion

    def analyze(self, image: np.ndarray, mango_mask: np.ndarray) -> Dict:
        if image is None:
            raise ValueError("Input image is None.")
        if mango_mask is None:
            raise ValueError("Mango mask is None.")

        # Resize image to mask dimensions
        mask_h, mask_w = mango_mask.shape[:2]
        resized = cv2.resize(image, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
        
        # Ensure mask is binary
        mask = np.where(mango_mask > 0, 255, 0).astype(np.uint8)

        # --- Remove fruit boundary ---
        erosion_size = max(3, self.boundary_erosion)
        if erosion_size % 2 == 0:
            erosion_size += 1
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        safe_mask = cv2.erode(mask, erosion_kernel, iterations=1)

        # --- Convert color spaces ---
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        H, S, V = cv2.split(hsv)

        # --- Local brightness (Darkness) ---
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=7)
        darkness = local_mean.astype(np.float32) - gray.astype(np.float32)

        # --- Black-Hat Detection (Highlights dark objects) ---
        blackhat = np.zeros_like(gray)
        for kernel_size in [7, 11, 17, 25]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            blackhat = np.maximum(blackhat, bh)

        # --- Statistics from mango ---
        fruit_pixels = safe_mask > 0
        if np.count_nonzero(fruit_pixels) == 0:
            return self._empty_result(mask)

        fruit_v = V[fruit_pixels]
        fruit_blackhat = blackhat[fruit_pixels]

        v_dark_threshold = np.percentile(fruit_v, 12)
        bh_threshold = max(np.percentile(fruit_blackhat, 94), 20) # Floor to prevent tiny noise on pale mangoes

        # --- 1. DARK / BLACK SPOTS ---
        dark_spots = (V <= v_dark_threshold) & (blackhat >= bh_threshold) & (darkness >= 4)

        # --- 2. BROWN / BLACK LESIONS (Optimized for green and pale mangoes) ---
        blue, green, red = cv2.split(resized)
        red_i, green_i, blue_i = red.astype(np.int16), green.astype(np.int16), blue.astype(np.int16)
        
        # Enhanced brown detection: Lower thresholds for subtle light brown bruises
        brown_difference = (red_i - green_i > 2) & (green_i - blue_i > 0)
        brown_lesions = brown_difference & (S > 45) & (blackhat > bh_threshold * 0.65) & (darkness > 3)

        # --- 3. VERY DARK DAMAGE / WET ROT ---
        severe_dark = (V < 70) & (S > 25) & (darkness > 8)

        # --- 4. WET ROT WITH GLARE (To handle shiny, shriveled Image 8) ---
        # Even with glare (high V), wet rot usually has a very high 'darkness' score
        wet_rot = (V > 70) & (V < 200) & (S < 100) & (darkness > 18) & (blackhat > bh_threshold * 1.2)

        # --- COMBINE DETECTIONS ---
        candidate = dark_spots | brown_lesions | severe_dark | wet_rot
        candidate &= (safe_mask > 0)

        # --- REMOVE SPECULAR HIGHLIGHTS (But keep wet rot if it's truly dark underneath) ---
        # Only mask highlights if the area is bright and NOT dark
        false_highlights = (V > 215) & (S < 90) & (darkness < 15)
        candidate &= ~false_highlights

        # --- REMOVE PALE NORMAL LENTICELS ---
        # Made stricter (V > 160) so brown gum/spots on pale mangoes won't be filtered out
        normal_lenticels = (V > 160) & (S < 100) & (darkness < 8) & (blackhat < bh_threshold * 0.7)
        candidate &= ~normal_lenticels

        # --- MORPHOLOGICAL CLEANING ---
        damage_mask = candidate.astype(np.uint8) * 255
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, open_kernel)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, close_kernel)

        # --- REMOVE TINY COMPONENTS & BOUNDARY TOUCHING ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damage_mask, connectivity=8)
        cleaned_mask = np.zeros_like(damage_mask)
        components = []

        # Create a safety guard to prevent detecting shadows that touch the inner edge
        safety_guard = cv2.erode(safe_mask, np.ones((5,5), np.uint8), iterations=1)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            comp_bbox = np.zeros_like(damage_mask)
            comp_bbox[labels == i] = 255
            
            # Reject if touching the boundary guard
            if np.any(comp_bbox & ~safety_guard):
                continue

            if area >= self.min_blemish_area:
                cleaned_mask[labels == i] = 255
                components.append({
                    "area": int(area), "x": int(x), "y": int(y),
                    "width": int(w), "height": int(h)
                })

        # --- AREA CALCULATION ---
        mango_area = cv2.countNonZero(safe_mask)
        damage_area = cv2.countNonZero(cleaned_mask)
        defect_percentage = min((damage_area / mango_area) * 100, 100.0) if mango_area > 0 else 0.0

        # --- DEFECT CLASSIFICATION ---
        defect_types = self.classify_defects(resized, cleaned_mask, components)

        # --- SEVERITY & GRADE ---
        if defect_percentage < 2: severity, grade = "Low", "A"
        elif defect_percentage < 5: severity, grade = "Medium", "B"
        elif defect_percentage < 10: severity, grade = "Medium", "C"
        else: severity, grade = "High", "D"

        # --- OVERLAY ---
        overlay = resized.copy()
        overlay[cleaned_mask > 0] = (0, 0, 255)
        debug_image = cv2.addWeighted(resized, 0.72, overlay, 0.28, 0)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 1)

        return {
            "defect_percentage": round(float(defect_percentage), 2),
            "damage_percentage": round(float(defect_percentage), 2),
            "blemish_pixel_area": int(damage_area),
            "mango_pixel_area": int(mango_area),
            "severity": severity, "grade": grade,
            "defect_types": defect_types, "component_count": len(components),
            "components": components, "damage_mask": cleaned_mask,
            "overlay": debug_image, "safe_mango_mask": safe_mask, "blackhat": blackhat
        }

    def classify_defects(self, image, damage_mask, components):
        if len(components) == 0: return ["None"]
        defect_types = []
        H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        scratch_found, rot_found, bruise_found = False, False, False

        for comp in components:
            x, y, w, h, area = comp["x"], comp["y"], comp["width"], comp["height"], comp["area"]
            if h == 0: continue

            aspect_ratio = w / h
            if (aspect_ratio > 4 or aspect_ratio < 0.25) and area >= 15:
                scratch_found = True

            region_mask = np.zeros(damage_mask.shape, dtype=np.uint8)
            cv2.rectangle(region_mask, (x, y), (x + w, y + h), 255, -1)
            region_mask = cv2.bitwise_and(region_mask, damage_mask)
            pixels = region_mask > 0

            if np.count_nonzero(pixels) == 0: continue
            mean_v, mean_s = np.mean(V[pixels]), np.mean(S[pixels])

            if mean_v < 80: rot_found = True
            if mean_s > 50: bruise_found = True

        if scratch_found: defect_types.append("Scratch")
        if rot_found: defect_types.append("Rot / Dark Lesion")
        if bruise_found: defect_types.append("Bruise / Brown Spot")
        if not defect_types: defect_types.append("Surface Blemish")
        
        return list(dict.fromkeys(defect_types))

    def _empty_result(self, mask):
        return { "defect_percentage": 0.0, "damage_percentage": 0.0, "blemish_pixel_area": 0, "mango_pixel_area": 0, "severity": "Low", "grade": "A", "defect_types": ["None"], "component_count": 0, "components": [], "damage_mask": np.zeros_like(mask), "overlay": None, "safe_mango_mask": mask, "blackhat": np.zeros_like(mask) }
