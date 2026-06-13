"""Exposure-стадия: CoTF (CoNet, 3D-LUT) правит дневной пере-/недосвет, размер не меняет."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from enhancer.models._cotf_arch import CoNet
from enhancer.models.base import StageParams

N_VERTICES_3D = 17
INPUT_RESOLUTION = 256


class ExposureEnhancer:
    """CoTF (exposure) под Enhancer Protocol. 3D-LUT тоновая коррекция, домен MSEC."""

    name = "exposure"

    def __init__(self, weights_path: Path, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model = CoNet(n_vertices_3d=N_VERTICES_3D, input_resolution=INPUT_RESOLUTION)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.version = "cotf@msec"

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
        enhanced = self._model(self._to_tensor(image_bgr))
        return self._from_tensor(enhanced)
