"""Срез чёрных полос (леттербокс) по краям кадра до оценки качества."""

from __future__ import annotations

import numpy as np

BLACK_MEDIAN = 4
MIN_KEEP_FRAC = 0.2


def _content_bounds(is_bar: np.ndarray) -> tuple[int, int]:
    """Границы [start, end) самого длинного непрерывного блока контента (не-полос).

    Если такой блок короче MIN_KEEP_FRAC стороны, по этой оси полос нет, возвращаем всю сторону
    (иначе единичная не-чёрная строка/столбец схлопнул бы кадр).
    """
    n = len(is_bar)
    best_len, best = 0, (0, n)
    i = 0
    while i < n:
        if is_bar[i]:
            i += 1
            continue
        j = i
        while j < n and not is_bar[j]:
            j += 1
        if j - i > best_len:
            best_len, best = j - i, (i, j)
        i = j
    return best if best_len >= n * MIN_KEEP_FRAC else (0, n)


def crop_black_bars(image_bgr: np.ndarray) -> np.ndarray:
    """Срезает чёрный леттербокс вокруг фото, оставляя самый крупный блок контента.

    Полоса определяется по медиане яркости строки/столбца (UI-текст на чёрном разрежён и
    медиану не поднимает), поэтому статус-бары и кнопки навигации скриншота уходят вместе с полосой.
    """
    lum = image_bgr.max(axis=2)
    h, w = lum.shape
    y0, y1 = _content_bounds(np.median(lum, axis=1) <= BLACK_MEDIAN)
    x0, x1 = _content_bounds(np.median(lum, axis=0) <= BLACK_MEDIAN)
    if (y0, y1) == (0, h) and (x0, x1) == (0, w):
        return image_bgr
    return image_bgr[y0:y1, x0:x1]
