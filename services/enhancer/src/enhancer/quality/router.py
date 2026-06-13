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
    FLOOR_PLAN = "floor_plan"


class PhotoType(StrEnum):
    """Тип фото от классификатора, задаёт ветку роутинга."""

    REAL_ESTATE = "real_estate"
    FLOOR_PLAN = "floor_plan"
    SCREENSHOT = "screenshot"


class Stage(StrEnum):
    LOW_LIGHT = "low_light"
    EXPOSURE = "exposure"
    RESTORE = "restore"
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


def route(
    m: QualityMetrics,
    width: int | None = None,
    height: int | None = None,
    available_stages: set[Stage] | None = None,
    photo_type: PhotoType = PhotoType.REAL_ESTATE,
) -> RouteDecision:
    """Выбирает модели по метрикам и типу фото. Порядок стадий: сначала тон, потом детали и апскейл.

    План (FLOOR_PLAN) получает только апскейл/денойз, тон ему не правим (CoTF/Retinexformer
    испортят чертёж). SCREENSHOT сюда не доходит: pipeline режет полосы и переклассифицирует кадр.
    """
    available = available_stages or set()
    tags: list[Tag] = []
    stages: list[Stage] = []
    skip_tone = photo_type == PhotoType.FLOOR_PLAN

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
    if skip_tone:
        tags.append(Tag.FLOOR_PLAN)

    is_washed_out = m.brightness_mean > BRIGHTNESS_HIGH
    needs_tone = (is_dark or is_low_contrast) and not is_washed_out
    if skip_tone:
        pass  # плану тон не правим, апскейл/денойз ниже остаются доступны
    elif is_washed_out and Stage.EXPOSURE in available:
        stages.append(Stage.EXPOSURE)
    elif needs_tone and Stage.LOW_LIGHT in available:
        stages.append(Stage.LOW_LIGHT)

    is_blurry = m.sharpness_laplacian_var < SHARPNESS_LOW
    max_side = max(width or 0, height or 0)
    is_low_res = 0 < max_side < settings.low_res_max_side

    if is_blurry:
        tags.append(Tag.BLURRY)
    if is_low_res:
        tags.append(Tag.LOW_RES)

    # Тёмный кадр после осветления (low_light) всегда поднимает шум из теней, его чистит SCUNet.
    # SAFMN на мелких кадрах денойзит сам, поэтому restore нужен только крупным тёмным/размытым.
    needs_restore = is_blurry or (is_dark and not skip_tone)
    if is_low_res and Stage.SAFMN in available:
        stages.append(Stage.SAFMN)
    elif needs_restore and Stage.RESTORE in available:
        stages.append(Stage.RESTORE)

    return RouteDecision(tags=tags, stages=stages, skip=len(stages) == 0)
