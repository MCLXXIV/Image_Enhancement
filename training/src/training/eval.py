"""Валидация SAFMN на held-out HR-папке: средние PSNR/SSIM/LPIPS."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import nn

from training.data.degradation import DegradationConfig, degrade_hr_to_lr

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _bgr_to_chw_float(img_bgr: np.ndarray) -> torch.Tensor:
    img = img_bgr.astype(np.float32) / 255.0
    img = img[:, :, ::-1].copy()
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def _chw_to_bgr_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).clamp_(0, 1).cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))[:, :, ::-1]
    return (arr * 255.0).round().astype(np.uint8)


def try_load_lpips(device: torch.device) -> nn.Module | None:
    """Возвращает lpips-метрику если пакет установлен, иначе None."""
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    val_dir: Path,
    scale: int,
    device: torch.device,
    seed: int = 0,
    max_samples: int = 50,
    lpips_fn: nn.Module | None = None,
) -> dict[str, float]:
    """Прогон SAFMN на val HR, средние метрики; LR строится через ту же degradation, что в трейне."""
    files = sorted(
        p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )[:max_samples]
    if not files:
        raise ValueError(f"Нет изображений в {val_dir}")

    deg = DegradationConfig(scale=scale)
    model.eval()
    psnrs: list[float] = []
    ssims: list[float] = []
    lpipses: list[float] = []

    for idx, path in enumerate(files):
        rng = np.random.default_rng(seed + idx)
        hr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if hr is None:
            continue

        h, w = hr.shape[:2]
        h2, w2 = (h // scale) * scale, (w // scale) * scale
        hr = hr[:h2, :w2]
        lr = degrade_hr_to_lr(hr, deg, rng)
        lr_t = _bgr_to_chw_float(lr).to(device)
        sr_t = model(lr_t).float()
        sr = _chw_to_bgr_uint8(sr_t)
        if sr.shape[:2] != hr.shape[:2]:
            sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_AREA)

        psnrs.append(float(cv2.PSNR(hr, sr)))
        ssims.append(float(structural_similarity(hr, sr, channel_axis=-1, data_range=255)))
        if lpips_fn is not None:
            sr_n = _bgr_to_chw_float(sr).to(device) * 2 - 1
            hr_n = _bgr_to_chw_float(hr).to(device) * 2 - 1
            lpipses.append(float(lpips_fn(sr_n, hr_n).item()))

    metrics = {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims))}
    if lpipses:
        metrics["lpips"] = float(np.mean(lpipses))
    return metrics
