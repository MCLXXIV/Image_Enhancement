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
from enhancer.quality.metrics import estimate_noise_sigma

HIGHLIGHT_LO = 0.5
HIGHLIGHT_HI = 0.9
CHROMA_HEADROOM = 12.0

# HDR-сцена: яркие источники (лампы, вывески) на тёмном фоне. На ней защитную маску
# расширяем вниз (HIGHLIGHT_LO_HDR), иначе Retinexformer раздувает ореол вокруг источника.
HIGHLIGHT_LO_HDR = 0.35
HDR_BRIGHT_RATIO = 0.002
HDR_MEAN_MAX = 0.4


def _is_hdr_scene(image_bgr: np.ndarray) -> bool:
    """Тёмный кадр с точечными засветами: средняя яркость низкая, но есть выгоревшие пиксели."""
    lum = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bright_ratio = float((lum > HIGHLIGHT_HI).mean())
    return bright_ratio > HDR_BRIGHT_RATIO and float(lum.mean()) < HDR_MEAN_MAX


WELL_EXP_SIGMA = 0.12
FUSION_LEVELS = 6
# Третья (затемнённая) экспозиция в fusion: поджимает раздутые ореолы вокруг источников.
# Степенная кривая сохраняет пик (1.0 -> 1.0), сжимает средне-яркие тона (гало bloom).
HIGHLIGHT_ROLLOFF_GAMMA = 1.8


def well_exposedness(image_f: np.ndarray, sigma: float = WELL_EXP_SIGMA) -> np.ndarray:
    """Вес пикселя по близости к середине тонов: пере-/недосвеченные зоны получают низкий вес."""
    e = np.exp(-((image_f - 0.5) ** 2) / (2.0 * sigma * sigma))
    return e.prod(axis=2).astype(np.float32)


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    cur = img
    for _ in range(1, levels):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gp = _gaussian_pyramid(img, levels)
    lp: list[np.ndarray] = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[-1])
    return lp


def exposure_fusion(images: list[np.ndarray], weights: list[np.ndarray], levels: int) -> np.ndarray:
    """Сливает N экспозиций (BGR float [0,1]) по весам через лапласовы пирамиды (без гало)."""
    wsum = np.zeros_like(weights[0])
    for w in weights:
        wsum = wsum + w
    wsum = wsum + 1e-12

    blended: list[np.ndarray] | None = None
    for img, w in zip(images, weights, strict=True):
        gw = _gaussian_pyramid((w / wsum).astype(np.float32), levels)
        lp = _laplacian_pyramid(img, levels)
        contrib = [lp[level] * gw[level][:, :, None] for level in range(levels)]
        blended = (
            contrib
            if blended is None
            else [blended[level] + contrib[level] for level in range(levels)]
        )

    assert blended is not None
    out = blended[-1]
    for level in range(levels - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[level].shape[1], blended[level].shape[0]))
        out = out + blended[level]
    return out


def fuse_exposures(original_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> np.ndarray:
    """Тон-компрессия HDR-сцены: вернуть оригинальные света, сохранив вытянутые осветлением тени.

    Третья экспозиция (затемнённый выход) поджимает раздутые ореолы вокруг источников света.
    """
    orig = original_bgr.astype(np.float32) / 255.0
    enh = enhanced_bgr.astype(np.float32) / 255.0
    rolled = np.power(enh, HIGHLIGHT_ROLLOFF_GAMMA).astype(np.float32)
    h, w = orig.shape[:2]
    levels = max(1, min(FUSION_LEVELS, int(np.log2(max(2, min(h, w))))))
    images = [orig, enh, rolled]
    weights = [well_exposedness(orig), well_exposedness(enh), well_exposedness(rolled)]
    fused = exposure_fusion(images, weights, levels)
    return (fused * 255.0).clip(0, 255).round().astype(np.uint8)


def adaptive_strength(base: float, noise_sigma: float, lo: float, hi: float, smin: float) -> float:
    """Чем шумнее вход, тем ближе сила к smin: меньше осветления, меньше вытянутого шума."""
    if hi <= lo:
        return base
    t = float(np.clip((noise_sigma - lo) / (hi - lo), 0.0, 1.0))
    return base + (smin - base) * t


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
        noise_lo: float = 2.0,
        noise_hi: float = 8.0,
        strength_min: float = 0.5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._noise_lo = float(noise_lo)
        self._noise_hi = float(noise_hi)
        self._strength_min = float(np.clip(strength_min, 0.0, 1.0))
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

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        is_hdr = _is_hdr_scene(image_bgr)
        lo = HIGHLIGHT_LO_HDR if is_hdr else HIGHLIGHT_LO
        protected = protect_highlights(image_bgr, enhanced, lo, HIGHLIGHT_HI)
        if is_hdr:
            protected = fuse_exposures(image_bgr, protected)

        noise = estimate_noise_sigma(gray)
        strength = adaptive_strength(
            self._strength, noise, self._noise_lo, self._noise_hi, self._strength_min
        )
        return blend_strength(image_bgr, protected, strength)
