from __future__ import annotations

import cv2
import numpy as np

from enhancer.models.base import StageParams


class ClaheEnhancer:
    name = "clahe"
    version = "cv-1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        clip = float(params.get("clahe_clip", 2.0))
        tile = int(params.get("clahe_tile", 8))
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        luma, chroma_a, chroma_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        luma = clahe.apply(luma)
        return cv2.cvtColor(cv2.merge((luma, chroma_a, chroma_b)), cv2.COLOR_LAB2BGR)
