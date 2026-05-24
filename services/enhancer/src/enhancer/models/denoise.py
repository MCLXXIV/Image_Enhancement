from __future__ import annotations

import cv2
import numpy as np

from enhancer.models.base import StageParams


class NlmDenoiseEnhancer:
    name = "denoise"
    version = "cv-nlm-1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        strength = float(params.get("denoise_strength", 0.4))
        h_luma = max(3.0, strength * 15.0)
        h_color = max(3.0, strength * 15.0)
        return cv2.fastNlMeansDenoisingColored(
            image_bgr,
            None,
            h=h_luma,
            hColor=h_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )
