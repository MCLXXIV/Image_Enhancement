import cv2
import numpy as np

from enhancer.models.base import StageParams
from enhancer.pipeline import Pipeline
from enhancer.quality.router import Stage
from enhancer.schemas import EnhanceParams


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
