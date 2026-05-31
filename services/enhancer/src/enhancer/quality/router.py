"""По метрикам качества решает, какие стадии включить."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from enhancer.quality.metrics import QualityMetrics


class Tag(StrEnum):
    LOW_LIGHT = "low_light"
    OVEREXPOSED = "overexposed"
    LOW_CONTRAST = "low_contrast"
    NOISY = "noisy"
    BLURRY = "blurry"
    COLOR_CAST = "color_cast"
    LOW_RES = "low_res"


class Stage(StrEnum):
    GAMMA = "gamma"
    CLAHE = "clahe"
    WHITE_BALANCE = "white_balance"
    UNSHARP = "unsharp"
    DENOISE = "denoise"
    SAFMN = "safmn"


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
LOW_RES_MAX_SIDE = 1280


def route(
    m: QualityMetrics,
    width: int | None = None,
    height: int | None = None,
    available_stages: set[Stage] | None = None,
) -> RouteDecision:
    """Маршрутизатор стадий; SAFMN включается при BLURRY/LOW_RES если зарегистрирован, иначе UNSHARP."""
    tags: list[Tag] = []
    stages: list[Stage] = []
    safmn_available = available_stages is not None and Stage.SAFMN in available_stages

    if m.brightness_mean < BRIGHTNESS_LOW or m.underexposed_ratio > UNDEREXPOSED_LOW:
        tags.append(Tag.LOW_LIGHT)
        stages.append(Stage.GAMMA)

    if m.brightness_mean > BRIGHTNESS_HIGH or m.overexposed_ratio > OVEREXPOSED_HIGH:
        tags.append(Tag.OVEREXPOSED)
        if Stage.GAMMA in stages:
            stages.remove(Stage.GAMMA)
        stages.append(Stage.CLAHE)

    if m.contrast_std < CONTRAST_LOW and Stage.CLAHE not in stages:
        tags.append(Tag.LOW_CONTRAST)
        stages.append(Stage.CLAHE)

    if m.channel_imbalance > CHANNEL_IMBALANCE_HIGH:
        tags.append(Tag.COLOR_CAST)
        stages.append(Stage.WHITE_BALANCE)

    is_blurry = m.sharpness_laplacian_var < SHARPNESS_LOW
    is_low_res = (
        safmn_available
        and width is not None
        and height is not None
        and max(width, height) < LOW_RES_MAX_SIDE
    )
    if is_blurry:
        tags.append(Tag.BLURRY)
        stages.append(Stage.SAFMN if safmn_available else Stage.UNSHARP)
    if is_low_res:
        tags.append(Tag.LOW_RES)
        if Stage.SAFMN not in stages:
            stages.append(Stage.SAFMN)

    skip = len(stages) == 0
    return RouteDecision(tags=tags, stages=stages, skip=skip)
