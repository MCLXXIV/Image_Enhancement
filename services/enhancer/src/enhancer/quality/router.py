"""По метрикам качества решает, какую ML-модель применить (или пропустить фото)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from enhancer.quality.metrics import QualityMetrics
from enhancer.settings import settings


class Tag(StrEnum):
    LOW_LIGHT = "low_light"
    OVEREXPOSED = "overexposed"
    LOW_CONTRAST = "low_contrast"
    COLOR_CAST = "color_cast"
    BLURRY = "blurry"
    LOW_RES = "low_res"


class Stage(StrEnum):
    LOW_LIGHT = "low_light"  # Retinexformer: экспозиция/тон
    RESTORE = "restore"  # SCUNet: шум/JPEG/блюр без апскейла
    SAFMN = "safmn"  # Real-SAFMN++: SR + restoration на мелких фото


class RouteDecision(BaseModel):
    tags: list[Tag]
    stages: list[Stage]
    skip: bool


BRIGHTNESS_LOW = 0.35
BRIGHTNESS_HIGH = 0.75
CONTRAST_LOW = 0.12
SHARPNESS_LOW = 60.0
UNDEREXPOSED_LOW = 0.20
OVEREXPOSED_HIGH = 0.10
CHANNEL_IMBALANCE_HIGH = 0.12


def route(
    m: QualityMetrics,
    width: int | None = None,
    height: int | None = None,
    available_stages: set[Stage] | None = None,
) -> RouteDecision:
    """Выбирает модели по метрикам. Порядок стадий: сначала тон, потом детали и апскейл."""
    available = available_stages or set()
    tags: list[Tag] = []
    stages: list[Stage] = []

    is_dark = m.brightness_mean < BRIGHTNESS_LOW or m.underexposed_ratio > UNDEREXPOSED_LOW
    is_low_contrast = m.contrast_std < CONTRAST_LOW
    is_overexposed = m.brightness_mean > BRIGHTNESS_HIGH or m.overexposed_ratio > OVEREXPOSED_HIGH

    if is_dark:
        tags.append(Tag.LOW_LIGHT)
    if is_low_contrast:
        tags.append(Tag.LOW_CONTRAST)
    if is_overexposed:
        tags.append(Tag.OVEREXPOSED)
    if m.channel_imbalance > CHANNEL_IMBALANCE_HIGH:
        tags.append(Tag.COLOR_CAST)

    needs_tone = (is_dark or is_low_contrast) and not is_overexposed
    if needs_tone and Stage.LOW_LIGHT in available:
        stages.append(Stage.LOW_LIGHT)

    is_blurry = m.sharpness_laplacian_var < SHARPNESS_LOW
    max_side = max(width or 0, height or 0)
    is_low_res = 0 < max_side < settings.low_res_max_side

    if is_blurry:
        tags.append(Tag.BLURRY)
    if is_low_res:
        tags.append(Tag.LOW_RES)

    if is_low_res and Stage.SAFMN in available:
        stages.append(Stage.SAFMN)
    elif is_blurry and Stage.RESTORE in available:
        stages.append(Stage.RESTORE)
    elif is_blurry and Stage.SAFMN in available and not is_low_res:
        stages.append(Stage.SAFMN)

    return RouteDecision(tags=tags, stages=stages, skip=len(stages) == 0)
