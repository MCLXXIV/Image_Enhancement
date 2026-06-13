import io

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from enhancer.image_io import decode_image


def _jpeg_with_orientation(arr_rgb: np.ndarray, orientation: int) -> bytes:
    img = Image.fromarray(arr_rgb, mode="RGB")
    exif = img.getexif()
    exif[0x0112] = orientation  # тег Orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_decode_applies_exif_orientation() -> None:
    # Сохранённый landscape 80x40 с orientation=6 при декоде разворачивается в portrait 40x80.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(40, 80, 3), dtype=np.uint8)
    out = decode_image(_jpeg_with_orientation(arr, 6))
    assert out.shape[0] == 80
    assert out.shape[1] == 40


def test_decode_without_exif_keeps_shape() -> None:
    arr = np.full((40, 80, 3), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    out = decode_image(buf.getvalue())
    assert out.shape[:2] == (40, 80)


def test_decode_garbage_raises() -> None:
    with pytest.raises(HTTPException):
        decode_image(b"not-an-image")
