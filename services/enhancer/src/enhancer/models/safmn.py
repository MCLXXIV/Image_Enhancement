"""SAFMN-стадия pipeline: апскейл изображения с тайлингом."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from enhancer.models._safmn_arch import SAFMN
from enhancer.models.base import StageParams


def choose_scale(long_side: int, scales: list[int], target_long_side: int) -> int:
    """Масштаб, чей выход по длинной стороне ближе всего к target. Тай-брейк в пользу меньшего."""
    return min(scales, key=lambda s: abs(long_side * s - target_long_side))


class SAFMNEnhancer:
    """Real_SAFMN++ под Enhancer Protocol. Держит модели под разные масштабы (x2/x4)."""

    name = "safmn"

    def __init__(
        self,
        weights_by_scale: dict[int, Path],
        device: str | None = None,
        tile: int = 256,
        tile_pad: int = 16,
        use_fp16: bool = False,
        dim: int = 128,
        n_blocks: int = 16,
        ffn_scale: float = 2.0,
        target_long_side: int = 1920,
    ) -> None:
        if not weights_by_scale:
            raise ValueError("SAFMNEnhancer requires at least one scale checkpoint")
        self._tile = tile
        self._tile_pad = tile_pad
        self._target_long_side = target_long_side
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._use_fp16 = use_fp16 and self._device.type == "cuda"
        self._models: dict[int, SAFMN] = {}
        for scale, path in weights_by_scale.items():
            model = SAFMN(dim=dim, n_blocks=n_blocks, ffn_scale=ffn_scale, upscaling_factor=scale)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = (
                checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self._models[scale] = model
        self._scales = sorted(self._models)
        self.version = "real-safmn-pp@x" + ",".join(str(s) for s in self._scales)

    def _pick_scale(self, h: int, w: int) -> int:
        return choose_scale(max(h, w), self._scales, self._target_long_side)

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        img = image_bgr.astype(np.float32) / 255.0
        img = img[:, :, ::-1].copy()
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        out = tensor.squeeze(0).clamp(0, 1).cpu().numpy()
        out = np.transpose(out, (1, 2, 0))[:, :, ::-1]
        return (out * 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def _infer_tiled(self, tensor: torch.Tensor, model: SAFMN, scale: int) -> torch.Tensor:
        _, _, h, w = tensor.shape
        out = tensor.new_zeros((1, 3, h * scale, w * scale))
        for y in range(0, h, self._tile):
            for x in range(0, w, self._tile):
                y1, x1 = y, x
                y2, x2 = min(y + self._tile, h), min(x + self._tile, w)
                py1, px1 = max(y1 - self._tile_pad, 0), max(x1 - self._tile_pad, 0)
                py2, px2 = min(y2 + self._tile_pad, h), min(x2 + self._tile_pad, w)
                patch = tensor[:, :, py1:py2, px1:px2]
                sr_patch = model(patch)
                ct, cl = (y1 - py1) * scale, (x1 - px1) * scale
                ch, cw = (y2 - y1) * scale, (x2 - x1) * scale
                out[:, :, y1 * scale : y2 * scale, x1 * scale : x2 * scale] = sr_patch[
                    :, :, ct : ct + ch, cl : cl + cw
                ]
        return out

    @torch.inference_mode()
    def _infer_full(self, tensor: torch.Tensor, model: SAFMN) -> torch.Tensor:
        return model(tensor)

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        scale = self._pick_scale(h, w)
        model = self._models[scale]
        tensor = self._to_tensor(image_bgr)
        if self._use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sr = (
                    self._infer_tiled(tensor, model, scale)
                    if self._tile > 0
                    else self._infer_full(tensor, model)
                )
            sr = sr.float()
        else:
            sr = (
                self._infer_tiled(tensor, model, scale)
                if self._tile > 0
                else self._infer_full(tensor, model)
            )
        return self._from_tensor(sr)
