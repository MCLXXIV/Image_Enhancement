"""Общие фикстуры с детерминированными синтетическими изображениями."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

_COLOR_PALETTE = (
    (50, 50, 200),
    (50, 200, 50),
    (200, 50, 50),
    (50, 200, 200),
    (200, 200, 50),
    (200, 50, 200),
    (200, 200, 200),
    (50, 50, 50),
)


def _color_grid(h: int = 256, w: int = 256, square: int = 32) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            idx = (x // square + (y // square) * 3) % len(_COLOR_PALETTE)
            img[y : y + square, x : x + square] = _COLOR_PALETTE[idx]
    return img


@pytest.fixture
def neutral_image() -> np.ndarray:
    """Хорошо экспонированное контрастное фото, роутер должен его пропустить."""
    rng = np.random.default_rng(seed=42)
    base = _color_grid()
    noise = rng.integers(-5, 5, size=base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def dark_image(neutral_image: np.ndarray) -> np.ndarray:
    """Тёмная версия neutral_image, стадия low_light должна её осветлить."""
    return (neutral_image.astype(np.float32) * 0.25).clip(0, 255).astype(np.uint8)


@pytest.fixture
def overexposed_image(neutral_image: np.ndarray) -> np.ndarray:
    return (neutral_image.astype(np.float32) * 1.6 + 120).clip(0, 255).astype(np.uint8)


@pytest.fixture
def blurry_image(neutral_image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(neutral_image, ksize=(0, 0), sigmaX=4.0)


@pytest.fixture
def color_cast_image(neutral_image: np.ndarray) -> np.ndarray:
    """Сдвинутый баланс белого с ослабленным синим каналом."""
    img = neutral_image.astype(np.float32)
    img[..., 0] *= 0.4
    return img.clip(0, 255).astype(np.uint8)


@pytest.fixture
def document_image() -> np.ndarray:
    """Чертёж: белый фон с тонкими тёмными линиями, как план квартиры."""
    img = np.full((256, 256, 3), 250, dtype=np.uint8)
    img[40:200, 40] = 30
    img[40:200, 200] = 30
    img[40, 40:201] = 30
    img[200, 40:201] = 30
    return img


@pytest.fixture
def letterboxed_image(neutral_image: np.ndarray) -> np.ndarray:
    """neutral_image с чёрными полосами сверху и снизу."""
    h, w = neutral_image.shape[:2]
    framed = np.zeros((h + 60, w, 3), dtype=np.uint8)
    framed[30 : 30 + h] = neutral_image
    return framed


@pytest.fixture
def jpeg_bytes(neutral_image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", neutral_image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def dark_jpeg_bytes(dark_image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", dark_image)
    assert ok
    return buf.tobytes()
