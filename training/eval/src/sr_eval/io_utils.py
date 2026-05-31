from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    """Читает изображение и возвращает HWC uint8 RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _center_crop(img: np.ndarray, h: int, w: int) -> np.ndarray:
    big_h, big_w = img.shape[:2]
    top = (big_h - h) // 2
    left = (big_w - w) // 2
    return img[top : top + h, left : left + w]


def match_size(sr: np.ndarray, gt: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """Центр-кроп обоих до общего минимума, при расхождении больше tol px бросает ошибку."""
    gh, gw = gt.shape[:2]
    sh, sw = sr.shape[:2]
    if abs(gh - sh) > tol or abs(gw - sw) > tol:
        raise ValueError(
            f"Размеры расходятся слишком сильно: GT={gh}x{gw}, SR={sh}x{sw}, tol={tol}. "
            "Кроп тут не спасёт, это разный масштаб, чини пайплайн."
        )
    h, w = min(gh, sh), min(gw, sw)
    return _center_crop(sr, h, w), _center_crop(gt, h, w)


def shave(img: np.ndarray, border: int) -> np.ndarray:
    """Обрезает border пикселей с каждой стороны для сопоставимости с SR-бенчмарками."""
    if border <= 0:
        return img
    return img[border:-border, border:-border]


def rgb_to_y(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """Переводит RGB uint8 в Y-канал BT.601 float32 в диапазоне [0, 255]."""
    img = img_rgb_uint8.astype(np.float32)
    y = 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0
    return y
