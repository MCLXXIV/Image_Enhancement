import cv2
import numpy as np

from enhancer.quality.metrics import compute_metrics, estimate_noise_sigma


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


def test_noise_sigma_rises_with_added_noise() -> None:
    rng = np.random.default_rng(seed=1)
    flat = np.full((128, 128), 128, dtype=np.uint8)
    noisy = np.clip(flat + rng.normal(0, 15, flat.shape), 0, 255).astype(np.uint8)
    clean_sigma = estimate_noise_sigma(flat)
    noisy_sigma = estimate_noise_sigma(noisy)
    assert clean_sigma < 1.0
    assert noisy_sigma > clean_sigma + 5.0


def test_blur_reduces_noise_sigma() -> None:
    rng = np.random.default_rng(seed=2)
    noisy = np.clip(rng.normal(128, 20, (128, 128)), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(noisy, (0, 0), sigmaX=2.0)
    assert estimate_noise_sigma(blurred) < estimate_noise_sigma(noisy)
