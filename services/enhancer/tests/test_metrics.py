import numpy as np

from enhancer.quality.metrics import compute_metrics


def test_dark_image_has_low_brightness(dark_image: np.ndarray) -> None:
    metrics = compute_metrics(dark_image)
    assert metrics.brightness_mean < 0.25
    assert metrics.underexposed_ratio > 0.2


def test_neutral_image_within_normal_range(neutral_image: np.ndarray) -> None:
    metrics = compute_metrics(neutral_image)
    assert 0.35 < metrics.brightness_mean < 0.75
    assert metrics.contrast_std > 0.12
    assert metrics.sharpness_laplacian_var > 100


def test_blurry_image_has_low_sharpness(
    neutral_image: np.ndarray, blurry_image: np.ndarray
) -> None:
    neutral = compute_metrics(neutral_image)
    blurry = compute_metrics(blurry_image)
    assert blurry.sharpness_laplacian_var < neutral.sharpness_laplacian_var / 2
