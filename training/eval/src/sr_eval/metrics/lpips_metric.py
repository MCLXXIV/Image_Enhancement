from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class LPIPSMetric:
    """LPIPS на полном RGB в диапазоне [-1, 1] с ленивым импортом torch и lpips."""

    name = "lpips"

    def __init__(self, net: str = "alex", device: str | None = None) -> None:
        import lpips
        import torch

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._model = lpips.LPIPS(net=net).to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def _to_tensor(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        x = img_rgb_uint8.astype(np.float32) / 127.5 - 1.0
        tensor = self._torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    def compute(self, sr: np.ndarray, gt: np.ndarray) -> float:
        with self._torch.no_grad():
            value = self._model(self._to_tensor(sr), self._to_tensor(gt))
        return float(value.item())
