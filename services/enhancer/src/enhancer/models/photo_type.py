"""Классификатор типа фото (недвига / план / скриншот), заменяет эвристики детекта."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch

from enhancer.quality.router import PhotoType

_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhotoTypeClassifier:
    """MobileNetV3-small поверх псевдо-лейблов SigLIP2: предсказывает PhotoType по кадру BGR."""

    name = "photo_type"

    def __init__(self, weights_path: Path, device: str | None = None) -> None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        arch = ckpt["arch"]
        self.img_size = int(ckpt["img_size"])
        self.classes = [PhotoType(c) for c in ckpt["classes"]]
        norm = ckpt.get("normalize", {"mean": _DEFAULT_MEAN, "std": _DEFAULT_STD})
        self.device = _resolve_device(device)
        self.version = f"{arch}@1"

        model = timm.create_model(arch, pretrained=False, num_classes=len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        self.model = model.eval().to(self.device)

        self._mean = torch.tensor(norm["mean"], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(norm["std"], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> PhotoType:
        """Тип кадра. Resize без кропа: чёрные полосы скриншота это признак, их нельзя срезать."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).to(self.device).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
        t = (t - self._mean) / self._std
        logits = self.model(t)
        return self.classes[int(logits.argmax(dim=1).item())]
