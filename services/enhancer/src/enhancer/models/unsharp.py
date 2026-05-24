from __future__ import annotations

import cv2
import numpy as np

from enhancer.models.base import StageParams


class UnsharpEnhancer:
    name = "unsharp"
    version = "cv-1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        amount = float(params.get("sharp_amount", 1.0))
        radius = float(params.get("sharp_radius", 1.5))
        blurred = cv2.GaussianBlur(image_bgr, ksize=(0, 0), sigmaX=radius)
        sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
