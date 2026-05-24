import numpy as np

from enhancer.quality.metrics import compute_metrics
from enhancer.quality.router import Stage, Tag, route


def test_neutral_image_is_skipped(neutral_image: np.ndarray) -> None:
    decision = route(compute_metrics(neutral_image))
    assert decision.skip
    assert decision.stages == []


def test_dark_image_routes_to_gamma(dark_image: np.ndarray) -> None:
    decision = route(compute_metrics(dark_image))
    assert Tag.LOW_LIGHT in decision.tags
    assert Stage.GAMMA in decision.stages
    assert not decision.skip


def test_blurry_image_routes_to_unsharp(blurry_image: np.ndarray) -> None:
    decision = route(compute_metrics(blurry_image))
    assert Tag.BLURRY in decision.tags
    assert Stage.UNSHARP in decision.stages


def test_color_cast_routes_to_white_balance(color_cast_image: np.ndarray) -> None:
    decision = route(compute_metrics(color_cast_image))
    assert Tag.COLOR_CAST in decision.tags
    assert Stage.WHITE_BALANCE in decision.stages
