from __future__ import annotations

import numpy as np

from enhancer.models.base import StageParams


class GrayWorldEnhancer:
    """Баланс белого по gray-world: подгоняет средние BGR-каналов к общему среднему."""

    name = "white_balance"
    version = "cv-1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        del params
        img = image_bgr.astype(np.float32)
        means = img.reshape(-1, 3).mean(axis=0)
        gray = means.mean()
        if (means <= 1e-6).any():
            return image_bgr
        scale = gray / means
        balanced = img * scale
        result: np.ndarray = np.clip(balanced, 0, 255).astype(np.uint8)
        return result
