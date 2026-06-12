import cv2
import numpy as np

from enhancer.quality.metrics import compute_metrics
from enhancer.quality.router import Stage, Tag, route

ALL_STAGES = {Stage.LOW_LIGHT, Stage.EXPOSURE, Stage.RESTORE, Stage.SAFMN}


def test_no_stages_available_skips(neutral_image: np.ndarray) -> None:
    decision = route(compute_metrics(neutral_image), 256, 256, available_stages=set())
    assert decision.skip
    assert decision.stages == []


def test_small_image_routes_to_safmn(neutral_image: np.ndarray) -> None:
    decision = route(compute_metrics(neutral_image), 256, 256, ALL_STAGES)
    assert Tag.LOW_RES in decision.tags
    assert Stage.SAFMN in decision.stages


def test_dark_image_routes_to_lowlight_first(dark_image: np.ndarray) -> None:
    decision = route(compute_metrics(dark_image), 256, 256, ALL_STAGES)
    assert Tag.LOW_LIGHT in decision.tags
    assert decision.stages[0] == Stage.LOW_LIGHT  # тон применяется до апскейла


def test_overexposed_routes_to_exposure(overexposed_image: np.ndarray) -> None:
    decision = route(compute_metrics(overexposed_image), 1400, 1400, ALL_STAGES)
    assert Tag.OVEREXPOSED in decision.tags
    assert Stage.EXPOSURE in decision.stages
    assert Stage.LOW_LIGHT not in decision.stages


def test_dark_with_highlights_runs_exposure_before_lowlight(dark_image: np.ndarray) -> None:
    img = dark_image.copy()
    img[:48, :48] = 255  # яркий засвет на тёмном кадре
    decision = route(compute_metrics(img), 256, 256, ALL_STAGES)
    assert Stage.EXPOSURE in decision.stages
    assert Stage.LOW_LIGHT in decision.stages
    assert decision.stages.index(Stage.EXPOSURE) < decision.stages.index(Stage.LOW_LIGHT)


def test_large_blurry_routes_to_restore_not_safmn(blurry_image: np.ndarray) -> None:
    big = cv2.resize(blurry_image, (1400, 1400))
    decision = route(compute_metrics(big), 1400, 1400, ALL_STAGES)
    assert Tag.BLURRY in decision.tags
    assert Stage.RESTORE in decision.stages
    assert Stage.SAFMN not in decision.stages  # большое фото не апскейлим


def test_large_blurry_without_restore_does_not_upscale(blurry_image: np.ndarray) -> None:
    big = cv2.resize(blurry_image, (1400, 1400))
    decision = route(compute_metrics(big), 1400, 1400, {Stage.SAFMN})
    assert Stage.SAFMN not in decision.stages


def test_large_neutral_image_skipped(neutral_image: np.ndarray) -> None:
    big = cv2.resize(neutral_image, (1400, 1400), interpolation=cv2.INTER_NEAREST)
    decision = route(compute_metrics(big), 1400, 1400, ALL_STAGES)
    assert decision.skip
