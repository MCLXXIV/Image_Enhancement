"""Метрики качества изображения, на вход BGR uint8."""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

UNDEREXPOSED_THRESHOLD = 0.10
OVEREXPOSED_THRESHOLD = 0.95
EXPOSURE_TARGET = 0.6


class QualityMetrics(BaseModel):
    brightness_mean: float
    brightness_median: float
    contrast_std: float
    entropy: float
    sharpness_laplacian_var: float
    saturation_mean: float
    colorfulness: float
    underexposed_ratio: float
    overexposed_ratio: float
    exposure_abs_error_to_target: float
    channel_imbalance: float


def _to_gray_norm(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _entropy(gray_uint8: np.ndarray) -> float:
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _colorfulness(image_bgr: np.ndarray) -> float:
    # Формула из Hasler & Süsstrunk (2003).
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_metrics(image_bgr: np.ndarray) -> QualityMetrics:
    gray_norm = _to_gray_norm(image_bgr)
    gray_uint8 = (gray_norm * 255).astype(np.uint8)

    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    brightness_mean = float(gray_norm.mean())

    channel_means = image_bgr.reshape(-1, 3).mean(axis=0)
    channel_imbalance = float(channel_means.max() - channel_means.min()) / 255.0

    return QualityMetrics(
        brightness_mean=brightness_mean,
        brightness_median=float(np.median(gray_norm)),
        contrast_std=float(gray_norm.std()),
        entropy=_entropy(gray_uint8),
        sharpness_laplacian_var=float(laplacian.var()),
        saturation_mean=float(hsv[..., 1].mean()) / 255.0,
        colorfulness=_colorfulness(image_bgr),
        underexposed_ratio=float((gray_norm < UNDEREXPOSED_THRESHOLD).mean()),
        overexposed_ratio=float((gray_norm > OVEREXPOSED_THRESHOLD).mean()),
        exposure_abs_error_to_target=float(abs(brightness_mean - EXPOSURE_TARGET)),
        channel_imbalance=channel_imbalance,
    )
