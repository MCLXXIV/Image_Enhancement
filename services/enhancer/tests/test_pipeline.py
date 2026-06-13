import cv2
import numpy as np

from enhancer.models.base import StageParams
from enhancer.pipeline import Pipeline
from enhancer.quality.router import PhotoType, Stage
from enhancer.schemas import EnhanceParams


class _ClassifierStub:
    """Заглушка классификатора: отдаёт типы по очереди (последний повторяется)."""

    name = "photo_type"
    version = "stub@1"

    def __init__(self, *types: PhotoType) -> None:
        self._types = list(types)

    def predict(self, image_bgr: np.ndarray) -> PhotoType:
        return self._types.pop(0) if len(self._types) > 1 else self._types[0]


class _UpscaleStub:
    """Заглушка SR-стадии: апскейлит x2, без torch."""

    name = "safmn"
    version = "stub@1"

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        return cv2.resize(image_bgr, (w * 2, h * 2))


def test_pipeline_skips_without_stages(neutral_image: np.ndarray) -> None:
    result = Pipeline(stages={}, iqa=None).run(neutral_image)
    assert result.skipped
    assert result.applied == []
    np.testing.assert_array_equal(result.image, neutral_image)


def test_force_without_available_stages_is_skipped(neutral_image: np.ndarray) -> None:
    # force=True не должен выдавать оригинал за enhanced, когда применять нечего.
    result = Pipeline(stages={}, iqa=None).run(neutral_image, EnhanceParams(force=True))
    assert result.skipped
    assert result.applied == []
    assert not result.fallback
    np.testing.assert_array_equal(result.image, neutral_image)


def test_pipeline_crops_black_bars(
    letterboxed_image: np.ndarray, neutral_image: np.ndarray
) -> None:
    result = Pipeline(stages={}, iqa=None).run(letterboxed_image)
    assert result.cropped
    assert result.image.shape == neutral_image.shape


def test_screenshot_is_cropped_then_reclassified(
    letterboxed_image: np.ndarray, neutral_image: np.ndarray
) -> None:
    # Классификатор: сначала screenshot, после среза полос кадр становится обычным фото.
    clf = _ClassifierStub(PhotoType.SCREENSHOT, PhotoType.REAL_ESTATE)
    result = Pipeline(stages={}, iqa=None, classifier=clf).run(letterboxed_image)
    assert result.cropped
    assert result.photo_type == PhotoType.REAL_ESTATE
    assert result.image.shape == neutral_image.shape


def test_non_screenshot_is_not_cropped(letterboxed_image: np.ndarray) -> None:
    # Классификатор сказал real_estate: полосы НЕ режем (даже если они есть).
    clf = _ClassifierStub(PhotoType.REAL_ESTATE)
    result = Pipeline(stages={}, iqa=None, classifier=clf).run(letterboxed_image)
    assert not result.cropped
    assert result.image.shape == letterboxed_image.shape


def test_floor_plan_skips_tone_keeps_upscale(neutral_image: np.ndarray) -> None:
    clf = _ClassifierStub(PhotoType.FLOOR_PLAN)
    dark_small = neutral_image // 4  # тёмный мелкий кадр: real_estate ушёл бы в low_light
    stages = {Stage.SAFMN: _UpscaleStub(), Stage.LOW_LIGHT: _Identity("low_light")}
    result = Pipeline(stages=stages, iqa=None, classifier=clf).run(dark_small)
    assert result.photo_type == PhotoType.FLOOR_PLAN
    assert Stage.SAFMN in result.applied
    assert Stage.LOW_LIGHT not in result.applied  # плану тон не правим


def test_pipeline_applies_injected_stage(neutral_image: np.ndarray) -> None:
    result = Pipeline(stages={Stage.SAFMN: _UpscaleStub()}, iqa=None).run(neutral_image)
    assert Stage.SAFMN in result.applied
    assert result.scale_factor == 2.0


def test_force_safmn_overrides_router(neutral_image: np.ndarray) -> None:
    big = cv2.resize(neutral_image, (1400, 1400), interpolation=cv2.INTER_NEAREST)
    result = Pipeline(stages={Stage.SAFMN: _UpscaleStub()}, iqa=None).run(
        big, EnhanceParams(force_safmn=True)
    )
    assert Stage.SAFMN in result.applied


class _Identity:
    """Заглушка-стадия без изменения изображения."""

    name = "id"
    version = "stub@1"

    def __init__(self, name: str) -> None:
        self.name = name

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        return image_bgr


def _large_dark(neutral_image: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Крупный (>1280) тёмный, но резкий кадр: dark + не blurry, чтобы restore шёл по ветке шума."""
    tiled = np.tile(neutral_image, (6, 6, 1))[:1400, :1400]
    return (tiled.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def _noisy(image_bgr: np.ndarray, sigma: float = 30.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=image_bgr.shape)
    return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def test_only_runs_just_forced_stage(neutral_image: np.ndarray) -> None:
    # Крупный тёмный шумный кадр: роутер сам позвал бы low_light + restore.
    big_dark = _noisy(_large_dark(neutral_image))
    stages = {Stage.LOW_LIGHT: _Identity("low_light"), Stage.RESTORE: _Identity("restore")}
    auto = Pipeline(stages=stages, iqa=None).run(big_dark)
    assert Stage.RESTORE in auto.applied  # без only роутер тянет и restore (шум поднят)

    only = Pipeline(stages=stages, iqa=None).run(
        big_dark, EnhanceParams(force=True, only=True, force_lowlight=True)
    )
    assert only.applied == [Stage.LOW_LIGHT]  # only оставляет ровно форснутую стадию


def test_conditional_denoise_skips_restore_when_low_noise(neutral_image: np.ndarray) -> None:
    # Крупный тёмный, но чистый кадр: low_light (заглушка) не поднял шум, restore не нужен.
    big_dark = _large_dark(neutral_image)
    stages = {Stage.LOW_LIGHT: _Identity("low_light"), Stage.RESTORE: _Identity("restore")}
    result = Pipeline(stages=stages, iqa=None).run(big_dark)
    assert Stage.LOW_LIGHT in result.applied
    assert Stage.RESTORE not in result.applied


def test_conditional_denoise_runs_restore_when_noisy(neutral_image: np.ndarray) -> None:
    big_dark = _noisy(_large_dark(neutral_image))
    stages = {Stage.LOW_LIGHT: _Identity("low_light"), Stage.RESTORE: _Identity("restore")}
    result = Pipeline(stages=stages, iqa=None).run(big_dark)
    assert Stage.RESTORE in result.applied


def test_iqa_gate_triggers_fallback(neutral_image: np.ndarray) -> None:
    class _WorseIqa:
        available = True

        def score(self, image_bgr: np.ndarray) -> dict[str, float]:
            return {"brisque": 50.0}

        def improved(self, before: dict[str, float], after: dict[str, float]) -> bool:
            return False

    result = Pipeline(stages={Stage.SAFMN: _UpscaleStub()}, iqa=_WorseIqa()).run(neutral_image)
    assert result.fallback
    np.testing.assert_array_equal(result.image, neutral_image)
