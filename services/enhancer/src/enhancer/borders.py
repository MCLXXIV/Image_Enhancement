"""Срез чёрных полос (леттербокс) по краям кадра до оценки качества."""

from __future__ import annotations

import numpy as np

BLACK_LEVEL = 16
BLACK_LINE_FRAC = 0.97
MAX_CROP_FRAC = 0.4


def _leading_black(mask: np.ndarray) -> int:
    """Сколько подряд чёрных линий от начала массива."""
    if mask.all():
        return len(mask)
    return int(np.argmax(~mask))


def crop_black_bars(image_bgr: np.ndarray) -> np.ndarray:
    """Срезает сплошные почти-чёрные полосы у краёв; яркость берём по максимуму каналов."""
    lum = image_bgr.max(axis=2)
    h, w = lum.shape
    black = lum < BLACK_LEVEL
    black_row = black.mean(axis=1) >= BLACK_LINE_FRAC
    black_col = black.mean(axis=0) >= BLACK_LINE_FRAC

    cap_v = int(h * MAX_CROP_FRAC)
    cap_h = int(w * MAX_CROP_FRAC)
    top = min(_leading_black(black_row), cap_v)
    bottom = min(_leading_black(black_row[::-1]), cap_v)
    left = min(_leading_black(black_col), cap_h)
    right = min(_leading_black(black_col[::-1]), cap_h)

    if not (top or bottom or left or right):
        return image_bgr
    y0, y1, x0, x1 = top, h - bottom, left, w - right
    if y1 - y0 < 1 or x1 - x0 < 1:
        return image_bgr
    return image_bgr[y0:y1, x0:x1]
