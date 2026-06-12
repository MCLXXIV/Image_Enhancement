"""Restoration стадия: SCUNet чистит шум/JPEG/лёгкий блюр без апскейла (scale=1)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from enhancer.models._scunet_arch import SCUNet
from enhancer.models.base import StageParams


class RestorationEnhancer:
    """SCUNet под Enhancer Protocol. Сохраняет разрешение и яркость, убирает шум/артефакты."""

    name = "restore"

    def __init__(
        self,
        weights_path: Path,
        device: str | None = None,
        dim: int = 64,
        config: Sequence[int] = (2, 2, 2, 2, 2, 2, 2),
        tile: int = 256,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tile = tile
        self._model = SCUNet(in_nc=3, config=tuple(config), dim=dim)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.version = "scunet-real@1"

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        img = image_bgr.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()  # BGR -> RGB
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        out = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
        out = np.transpose(out, (1, 2, 0))[:, :, ::-1]  # RGB -> BGR
        return (out * 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def _infer_tiled(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, h, w = tensor.shape
        if self._tile <= 0 or (h <= self._tile and w <= self._tile):
            return self._model(tensor)
        pad = 16
        out = tensor.new_zeros((1, 3, h, w))
        for y in range(0, h, self._tile):
            for x in range(0, w, self._tile):
                y1, x1 = y, x
                y2, x2 = min(y + self._tile, h), min(x + self._tile, w)
                py1, px1 = max(y1 - pad, 0), max(x1 - pad, 0)
                py2, px2 = min(y2 + pad, h), min(x2 + pad, w)
                patch = tensor[:, :, py1:py2, px1:px2]
                sr_patch = self._model(patch)
                ct, cl = y1 - py1, x1 - px1
                ch, cw = y2 - y1, x2 - x1
                out[:, :, y1:y2, x1:x2] = sr_patch[:, :, ct : ct + ch, cl : cl + cw]
        return out

    @torch.inference_mode()
    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        tensor = self._to_tensor(image_bgr)
        restored = self._infer_tiled(tensor)
        return self._from_tensor(restored)
