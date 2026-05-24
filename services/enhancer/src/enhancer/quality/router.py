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


class Stage(StrEnum):
    GAMMA = "gamma"
    CLAHE = "clahe"
    WHITE_BALANCE = "white_balance"
    UNSHARP = "unsharp"
    DENOISE = "denoise"


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


def route(m: QualityMetrics) -> RouteDecision:
    tags: list[Tag] = []
    stages: list[Stage] = []

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

    if m.sharpness_laplacian_var < SHARPNESS_LOW:
        tags.append(Tag.BLURRY)
        stages.append(Stage.UNSHARP)

    skip = len(stages) == 0
    return RouteDecision(tags=tags, stages=stages, skip=skip)
