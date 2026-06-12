"""Low-light стадия: Retinexformer осветляет, чистит шум и правит цвет тёмных фото."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from enhancer.models._retinexformer_arch import RetinexFormer
from enhancer.models.base import StageParams


class LowLightEnhancer:
    """Retinexformer под Enhancer Protocol. Меняет экспозицию/тон/цвет, не апскейлит."""

    name = "low_light"

    def __init__(
        self,
        weights_path: Path,
        device: str | None = None,
        n_feat: int = 40,
        stage: int = 1,
        num_blocks: Sequence[int] = (1, 2, 2),
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._level = 2
        self._model = RetinexFormer(
            in_channels=3, out_channels=3, n_feat=n_feat, stage=stage, num_blocks=list(num_blocks)
        )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.version = "retinexformer@lol-v2-real"

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        img = image_bgr.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()  # BGR -> RGB
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        out = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
        out = np.transpose(out, (1, 2, 0))[:, :, ::-1]  # RGB -> BGR
        return (out * 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        tensor = self._to_tensor(image_bgr)
        _, _, h, w = tensor.shape
        factor = 2**self._level
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        out = self._model(tensor)
        out = out[:, :, :h, :w]
        return self._from_tensor(out)
