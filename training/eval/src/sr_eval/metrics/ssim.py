from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from sr_eval.io_utils import rgb_to_y, shave


class SSIMMetric:
    """SSIM, по умолчанию считается на Y-канале BT.601 как в SR-литературе."""

    name = "ssim"

    def __init__(self, on_y_channel: bool = True, crop_border: int = 0) -> None:
        self._on_y = on_y_channel
        self._border = crop_border

    def compute(self, sr: np.ndarray, gt: np.ndarray) -> float:
        sr_c = shave(sr, self._border)
        gt_c = shave(gt, self._border)

        if self._on_y:
            sr_arr = rgb_to_y(sr_c)
            gt_arr = rgb_to_y(gt_c)
            value = structural_similarity(gt_arr, sr_arr, data_range=255.0)
        else:
            value = structural_similarity(gt_c, sr_c, data_range=255, channel_axis=-1)
        return float(value)
