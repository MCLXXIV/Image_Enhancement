import cv2
import numpy as np

from enhancer.models.lowlight import (
    _is_hdr_scene,
    adaptive_strength,
    blend_strength,
    fuse_exposures,
    protect_highlights,
    well_exposedness,
)


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


def test_adaptive_strength_low_noise_keeps_base() -> None:
    assert adaptive_strength(0.8, noise_sigma=1.0, lo=2.0, hi=8.0, smin=0.5) == 0.8


def test_adaptive_strength_high_noise_drops_to_min() -> None:
    assert adaptive_strength(0.8, noise_sigma=20.0, lo=2.0, hi=8.0, smin=0.5) == 0.5


def test_adaptive_strength_interpolates_in_band() -> None:
    out = adaptive_strength(0.8, noise_sigma=5.0, lo=2.0, hi=8.0, smin=0.5)
    assert 0.5 < out < 0.8


def test_hdr_scene_detected_on_dark_with_bright_source() -> None:
    img = np.full((64, 64, 3), 20, dtype=np.uint8)  # тёмный фон
    img[:6, :6] = 255  # яркая лампа
    assert _is_hdr_scene(img)


def test_uniform_bright_frame_is_not_hdr() -> None:
    assert not _is_hdr_scene(np.full((64, 64, 3), 230, dtype=np.uint8))


def test_well_exposedness_peaks_at_midtones() -> None:
    mid = np.full((4, 4, 3), 0.5, dtype=np.float32)
    bright = np.full((4, 4, 3), 0.97, dtype=np.float32)
    dark = np.full((4, 4, 3), 0.03, dtype=np.float32)
    assert well_exposedness(mid).mean() > well_exposedness(bright).mean()
    assert well_exposedness(mid).mean() > well_exposedness(dark).mean()


def test_fusion_suppresses_bloom_keeps_shadows() -> None:
    # Левая зона в оригинале яркая (источник света), правая, тень.
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    original[:, :32] = 200  # источник: в оригинале экспонирован нормально
    original[:, 32:] = 20  # тень
    # Retinexformer: источник раздул в bloom (245), тень поднял в средние тона (120).
    enhanced = np.zeros((64, 64, 3), dtype=np.uint8)
    enhanced[:, :32] = 245
    enhanced[:, 32:] = 120
    fused = fuse_exposures(original, enhanced)
    left = fused[:, 4:20].mean()  # bloom-зона
    right = fused[:, 44:60].mean()  # тень
    assert left < enhanced[:, :32].mean()  # bloom подтянут к оригиналу
    assert right > original[:, 32:].mean() + 40  # тень реально осветлена


def test_rolloff_keeps_peak_source_compresses_halo() -> None:
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    original[:, :22] = 255  # пиковый источник (выгорел в оригинале)
    original[:, 22:44] = 160  # ореол вокруг
    original[:, 44:] = 20  # тень
    enhanced = np.zeros((64, 64, 3), dtype=np.uint8)
    enhanced[:, :22] = 255
    enhanced[:, 22:44] = 235  # осветление раздуло ореол
    enhanced[:, 44:] = 120
    fused = fuse_exposures(original, enhanced)
    assert fused[:, 4:18].mean() > 230  # пиковый источник сохранён, не посерел
    assert fused[:, 26:40].mean() < enhanced[:, 22:44].mean()  # ореол поджат


def test_orange_halo_is_desaturated() -> None:
    original = np.full((8, 8, 3), (40, 50, 70), dtype=np.uint8)  # тёмный тёплый
    enhanced = np.full((8, 8, 3), (20, 110, 230), dtype=np.uint8)  # насыщенный оранжевый (OOD)
    out = protect_highlights(original, enhanced, 0.5, 0.9)
    sat_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    sat_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    assert sat_out < sat_enhanced * 0.6
