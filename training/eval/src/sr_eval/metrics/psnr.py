from __future__ import annotations

import math

import numpy as np

from sr_eval.io_utils import rgb_to_y, shave


class PSNRMetric:
    """PSNR в дБ, по умолчанию считается на Y-канале BT.601 как в SR-литературе."""

    name = "psnr"

    def __init__(self, on_y_channel: bool = True, crop_border: int = 0) -> None:
        self._on_y = on_y_channel
        self._border = crop_border

    def compute(self, sr: np.ndarray, gt: np.ndarray) -> float:
        sr_c = shave(sr, self._border)
        gt_c = shave(gt, self._border)

        if self._on_y:
            sr_arr = rgb_to_y(sr_c)
            gt_arr = rgb_to_y(gt_c)
            data_range = 255.0
        else:
            sr_arr = sr_c.astype(np.float32)
            gt_arr = gt_c.astype(np.float32)
            data_range = 255.0

        mse = float(np.mean((sr_arr - gt_arr) ** 2))
        if mse == 0.0:
            return float("inf")
        return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
