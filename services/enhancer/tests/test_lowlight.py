import numpy as np

from enhancer.models.lowlight import protect_highlights


def test_bright_areas_keep_original() -> None:
    original = np.full((8, 8, 3), 255, dtype=np.uint8)
    enhanced = np.zeros((8, 8, 3), dtype=np.uint8)
    enhanced[:, :, 2] = 255  # ядерно-красный результат
    out = protect_highlights(original, enhanced, 0.7, 0.95)
    np.testing.assert_array_equal(out, original)


def test_dark_areas_take_enhanced() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    enhanced = np.full((8, 8, 3), 120, dtype=np.uint8)
    out = protect_highlights(original, enhanced, 0.7, 0.95)
    np.testing.assert_array_equal(out, enhanced)
