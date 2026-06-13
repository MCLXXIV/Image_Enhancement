"""Декодирование/кодирование JPEG, сборка X-Enhance-* заголовков и публикация метрик качества."""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

from enhancer.observability import (
    enhance_iqa_after,
    enhance_iqa_before,
    enhance_psnr_vs_input,
    enhance_quality_after,
    enhance_quality_before,
    enhance_scale_factor,
)
from enhancer.pipeline import EnhanceResult

_QUALITY_METRICS_FOR_HISTOGRAM: tuple[str, ...] = (
    "brightness_mean",
    "contrast_std",
    "saturation_mean",
)

JPEG_QUALITY = 92


def decode_image(raw: bytes) -> np.ndarray:
    """Декод в BGR uint8 с учётом EXIF-ориентации (cv2.imdecode её игнорирует)."""
    try:
        with Image.open(io.BytesIO(raw)) as pil:
            rgb = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="cannot decode image") from None
        return image


def encode_jpeg(image_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failure")
    return buf.tobytes()


def build_headers(result: EnhanceResult) -> dict[str, str]:
    return {
        "X-Enhance-Applied": ",".join(s.value for s in result.applied) or "none",
        "X-Enhance-Skipped": "true" if result.skipped else "false",
        "X-Enhance-Fallback": "true" if result.fallback else "false",
        "X-Enhance-Cropped": "true" if result.cropped else "false",
        "X-Enhance-Photo-Type": result.photo_type.value,
        "X-Enhance-Latency-Ms": f"{result.total_latency_ms:.1f}",
        "X-Enhance-Psnr-Vs-Input": f"{result.psnr_vs_input:.2f}",
        "X-Enhance-Scale-Factor": f"{result.scale_factor:.2f}",
        "X-Enhance-Quality-Before": result.metrics_before.model_dump_json(),
        "X-Enhance-Quality-After": result.metrics_after.model_dump_json(),
        "X-Enhance-Iqa-Before": json.dumps(result.iqa_before),
        "X-Enhance-Iqa-After": json.dumps(result.iqa_after),
        "X-Enhance-Model-Versions": json.dumps(result.model_versions),
    }


def observe_quality(result: EnhanceResult) -> None:
    before = result.metrics_before.model_dump()
    after = result.metrics_after.model_dump()
    for metric_name in _QUALITY_METRICS_FOR_HISTOGRAM:
        enhance_quality_before.labels(metric=metric_name).observe(before[metric_name])
        enhance_quality_after.labels(metric=metric_name).observe(after[metric_name])
    if not result.skipped:
        enhance_psnr_vs_input.observe(result.psnr_vs_input)
        enhance_scale_factor.observe(result.scale_factor)
        for metric_name, value in result.iqa_before.items():
            enhance_iqa_before.labels(metric=metric_name).observe(value)
        for metric_name, value in result.iqa_after.items():
            enhance_iqa_after.labels(metric=metric_name).observe(value)
