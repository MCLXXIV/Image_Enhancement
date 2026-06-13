import numpy as np

from enhancer.borders import crop_black_bars


def test_crop_removes_letterbox(letterboxed_image: np.ndarray, neutral_image: np.ndarray) -> None:
    cropped = crop_black_bars(letterboxed_image)
    assert cropped.shape == neutral_image.shape


def test_ui_text_in_bar_still_cropped(
    letterboxed_image: np.ndarray, neutral_image: np.ndarray
) -> None:
    # Разреженный яркий UI (статус-бар) в чёрной полосе не должен мешать срезу: медиана строки ~0.
    framed = letterboxed_image.copy()
    framed[10, 20:40] = 255
    cropped = crop_black_bars(framed)
    assert cropped.shape == neutral_image.shape


def test_no_crop_on_clean_image(neutral_image: np.ndarray) -> None:
    assert crop_black_bars(neutral_image).shape == neutral_image.shape


def test_dark_content_not_cropped() -> None:
    # Равномерно тёмный кадр (ночное фото) ярче чистого чёрного, полос нет, не режем.
    dim = np.full((200, 200, 3), 30, dtype=np.uint8)
    assert crop_black_bars(dim).shape == dim.shape


def test_all_black_returns_original() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_black_bars(img).shape == img.shape
