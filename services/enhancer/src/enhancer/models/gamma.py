from __future__ import annotations

import numpy as np

from enhancer.models.base import StageParams


class GammaEnhancer:
    name = "gamma"
    version = "cv-1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        gamma = float(params.get("gamma", 0.6))
        gamma = max(0.1, gamma)
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        result: np.ndarray = lut[image_bgr]
        return result
