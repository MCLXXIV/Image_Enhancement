import cv2
import numpy as np

from enhancer.models.lowlight import blend_strength, protect_highlights


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


def test_strength_blends_toward_original() -> None:
    original = np.full((8, 8, 3), 40, dtype=np.uint8)
    enhanced = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = blend_strength(original, enhanced, 0.75)
    np.testing.assert_array_equal(out, np.full((8, 8, 3), 160, dtype=np.uint8))


def test_strength_full_returns_enhanced_unchanged() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    enhanced = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = blend_strength(original, enhanced, 1.0)
    np.testing.assert_array_equal(out, enhanced)


def test_orange_halo_is_desaturated() -> None:
    original = np.full((8, 8, 3), (40, 50, 70), dtype=np.uint8)  # тёмный тёплый
    enhanced = np.full((8, 8, 3), (20, 110, 230), dtype=np.uint8)  # насыщенный оранжевый (OOD)
    out = protect_highlights(original, enhanced, 0.5, 0.9)
    sat_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    sat_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    assert sat_out < sat_enhanced * 0.6
