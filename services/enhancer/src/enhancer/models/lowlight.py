"""Low-light стадия: Retinexformer осветляет, чистит шум и правит цвет тёмных фото."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from enhancer.models._retinexformer_arch import RetinexFormer
from enhancer.models.base import StageParams

HIGHLIGHT_LO = 0.5
HIGHLIGHT_HI = 0.9
CHROMA_HEADROOM = 12.0


def protect_highlights(
    original_bgr: np.ndarray, enhanced_bgr: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Яркость светлых зон и насыщенность держим от оригинала, гасит OOD оранжевые ореолы и края."""
    orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    enh = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    lum = orig[:, :, 0] / 255.0
    weight = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)
    sigma = max(original_bgr.shape[:2]) * 0.008
    if sigma > 0:
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma)

    y_out = orig[:, :, 0] * weight + enh[:, :, 0] * (1.0 - weight)

    cr_o, cb_o = orig[:, :, 1] - 128.0, orig[:, :, 2] - 128.0
    cr_e, cb_e = enh[:, :, 1] - 128.0, enh[:, :, 2] - 128.0
    chroma_o = np.sqrt(cr_o * cr_o + cb_o * cb_o)
    chroma_e = np.sqrt(cr_e * cr_e + cb_e * cb_e)
    scale = np.minimum(1.0, (chroma_o + CHROMA_HEADROOM) / (chroma_e + 1e-6))
    cr_out = cr_o * weight + cr_e * scale * (1.0 - weight) + 128.0
    cb_out = cb_o * weight + cb_e * scale * (1.0 - weight) + 128.0

    out = np.stack([y_out, cr_out, cb_out], axis=2).round().clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def blend_strength(
    original_bgr: np.ndarray, enhanced_bgr: np.ndarray, strength: float
) -> np.ndarray:
    """Подмешивает оригинал к выходу модели, <1.0 возвращает родные тени и текстуру."""
    if strength >= 1.0:
        return enhanced_bgr
    blended = enhanced_bgr.astype(np.float32) * strength + original_bgr.astype(np.float32) * (
        1.0 - strength
    )
    return blended.round().clip(0, 255).astype(np.uint8)


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
        strength: float = 1.0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._strength = float(np.clip(strength, 0.0, 1.0))
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
        enhanced = self._from_tensor(out)
        protected = protect_highlights(image_bgr, enhanced, HIGHLIGHT_LO, HIGHLIGHT_HI)
        return blend_strength(image_bgr, protected, self._strength)
