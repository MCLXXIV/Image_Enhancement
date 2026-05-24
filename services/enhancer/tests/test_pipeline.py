import numpy as np

from enhancer.pipeline import Pipeline
from enhancer.quality.router import Stage
from enhancer.schemas import EnhanceParams


def test_pipeline_skips_neutral_image(neutral_image: np.ndarray) -> None:
    result = Pipeline().run(neutral_image)
    assert result.skipped
    assert result.applied == []
    np.testing.assert_array_equal(result.image, neutral_image)


def test_pipeline_brightens_dark_image(dark_image: np.ndarray) -> None:
    result = Pipeline().run(dark_image)
    assert not result.skipped
    assert Stage.GAMMA in result.applied
    assert result.metrics_after.brightness_mean > result.metrics_before.brightness_mean


def test_pipeline_white_box_forces_denoise(neutral_image: np.ndarray) -> None:
    params = EnhanceParams(force=True, denoise_strength=0.2)
    result = Pipeline().run(neutral_image, params)
    assert Stage.DENOISE in result.applied


def test_pipeline_force_flag_disables_skip(neutral_image: np.ndarray) -> None:
    params = EnhanceParams(force=True, clahe_clip=2.5)
    result = Pipeline().run(neutral_image, params)
    assert Stage.CLAHE in result.applied
