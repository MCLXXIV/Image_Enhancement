import numpy as np

from enhancer.borders import crop_black_bars


def test_crop_removes_letterbox(letterboxed_image: np.ndarray, neutral_image: np.ndarray) -> None:
    cropped = crop_black_bars(letterboxed_image)
    assert cropped.shape == neutral_image.shape


def test_no_crop_on_clean_image(neutral_image: np.ndarray) -> None:
    out = crop_black_bars(neutral_image)
    assert out.shape == neutral_image.shape


def test_all_black_capped_at_max_frac() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = crop_black_bars(img)
    assert out.shape[0] >= 1 and out.shape[1] >= 1
