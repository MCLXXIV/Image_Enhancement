"""Exposure-стадия: IAT правит пере- и недосвет. Размер и разрешение не меняет."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from enhancer.models._iat_arch import IAT
from enhancer.models.base import StageParams


class ExposureEnhancer:
    """IAT (exposure) под Enhancer Protocol. Лёгкая модель под засветы/пересвет."""

    name = "exposure"

    def __init__(self, weights_path: Path, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model = IAT(type="exp")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.version = "iat@exposure"

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        img = image_bgr.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        out = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
        out = np.transpose(out, (1, 2, 0))[:, :, ::-1]
        return (out * 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        tensor = self._to_tensor(image_bgr)
        _, _, enhanced = self._model(tensor)
        return self._from_tensor(enhanced)
