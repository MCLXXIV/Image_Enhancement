"""Декодирование/кодирование JPEG, сборка X-Enhance-* заголовков и публикация метрик качества."""

from __future__ import annotations

import json

import cv2
import numpy as np
from fastapi import HTTPException

from enhancer.observability import enhance_quality_after, enhance_quality_before
from enhancer.pipeline import EnhanceResult

_QUALITY_METRICS_FOR_HISTOGRAM: tuple[str, ...] = (
    "brightness_mean",
    "contrast_std",
    "saturation_mean",
)

JPEG_QUALITY = 92


def decode_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
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
        "X-Enhance-Latency-Ms": f"{result.total_latency_ms:.1f}",
        "X-Enhance-Quality-Before": result.metrics_before.model_dump_json(),
        "X-Enhance-Quality-After": result.metrics_after.model_dump_json(),
        "X-Enhance-Model-Versions": json.dumps(result.model_versions),
    }


def observe_quality(result: EnhanceResult) -> None:
    before = result.metrics_before.model_dump()
    after = result.metrics_after.model_dump()
    for metric_name in _QUALITY_METRICS_FOR_HISTOGRAM:
        enhance_quality_before.labels(metric=metric_name).observe(before[metric_name])
        enhance_quality_after.labels(metric=metric_name).observe(after[metric_name])
