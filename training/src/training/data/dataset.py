"""HR/LR пары на лету; HR кропается из реальных фото, LR строится через degradation pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.degradation import DegradationConfig, degrade_hr_to_lr

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _bgr_to_chw_float(img_bgr: np.ndarray) -> torch.Tensor:
    img = img_bgr.astype(np.float32) / 255.0
    img = img[:, :, ::-1].copy()
    return torch.from_numpy(img).permute(2, 0, 1)


class RealEstateSRDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Случайный HR-кроп из фото + соответствующий LR; HR/LR в формате CHW float [0, 1]."""

    def __init__(
        self,
        hr_dir: Path,
        hr_crop_size: int = 256,
        degradation: DegradationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self._files = sorted(
            p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self._files:
            raise ValueError(f"Нет изображений в {hr_dir}")
        self._crop = hr_crop_size
        self._deg = degradation or DegradationConfig()
        self._seed = seed

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed=self._seed + idx)
        img = cv2.imread(str(self._files[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Не удалось прочитать {self._files[idx]}")

        h, w = img.shape[:2]
        crop = self._crop
        if h < crop or w < crop:
            img = cv2.resize(img, (max(w, crop), max(h, crop)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        y0 = int(rng.integers(0, h - crop + 1))
        x0 = int(rng.integers(0, w - crop + 1))
        hr = img[y0 : y0 + crop, x0 : x0 + crop]

        lr = degrade_hr_to_lr(hr, self._deg, rng)
        return _bgr_to_chw_float(lr), _bgr_to_chw_float(hr)
