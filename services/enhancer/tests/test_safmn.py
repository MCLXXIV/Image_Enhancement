import numpy as np

from enhancer.models.safmn import blend_bicubic, choose_scale


def test_single_scale_always_picked() -> None:
    assert choose_scale(640, [4], 1920) == 4


def test_blend_full_strength_returns_sr_unchanged() -> None:
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    sr = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = blend_bicubic(original, sr, 1.0)
    np.testing.assert_array_equal(out, sr)


def test_blend_mixes_bicubic_into_sr() -> None:
    original = np.full((4, 4, 3), 40, dtype=np.uint8)
    sr = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = blend_bicubic(original, sr, 0.75)
    np.testing.assert_array_equal(out, np.full((8, 8, 3), 160, dtype=np.uint8))


def test_picks_scale_closest_to_target() -> None:
    assert choose_scale(960, [2, 4], 1920) == 2
    assert choose_scale(400, [2, 4], 1920) == 4
    assert choose_scale(1200, [2, 4], 1920) == 2


def test_tie_prefers_smaller_scale() -> None:
    assert choose_scale(640, [2, 4], 1920) == 2
