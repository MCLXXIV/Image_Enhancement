"""Real-world деградация HR в LR: blur, noise, downscale, JPEG."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DegradationConfig:
    scale: int = 4
    blur_sigma_range: tuple[float, float] = (0.2, 2.0)
    noise_sigma_range: tuple[float, float] = (1.0, 12.0)
    jpeg_quality_range: tuple[int, int] = (60, 95)


def degrade_hr_to_lr(
    hr_bgr: np.ndarray, cfg: DegradationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Производит LR из HR одной случайной комбинацией blur+noise+downscale+JPEG."""
    img = hr_bgr.astype(np.float32)

    sigma = float(rng.uniform(*cfg.blur_sigma_range))
    if sigma > 0:
        img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)

    h, w = img.shape[:2]
    lr_h, lr_w = h // cfg.scale, w // cfg.scale
    img = cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    noise_sigma = float(rng.uniform(*cfg.noise_sigma_range))
    if noise_sigma > 0:
        img = img + rng.normal(0, noise_sigma, img.shape).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)

    quality = int(rng.integers(cfg.jpeg_quality_range[0], cfg.jpeg_quality_range[1] + 1))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img
