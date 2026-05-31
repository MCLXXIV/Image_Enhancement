"""Загрузка весов SAFMN и инференс с опциональным тайлингом для больших изображений."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from safmn_infer.arch import SAFMN


def load_model(weights_path: Path, scale: int, device: torch.device) -> SAFMN:
    """Конфигурация совпадает с upstream-инференсом Real_SAFMN++ (dim=128, n_blocks=16)."""
    model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=scale)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _img_to_tensor(img_bgr_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    img = img_bgr_uint8.astype(np.float32) / 255.0
    img = img[:, :, ::-1].copy()
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _tensor_to_img(tensor: torch.Tensor) -> np.ndarray:
    out = tensor.squeeze(0).clamp_(0, 1).cpu().numpy()
    out = np.transpose(out, (1, 2, 0))[:, :, ::-1]
    return (out * 255.0).round().astype(np.uint8)


@torch.inference_mode()
def _infer_full(model: SAFMN, tensor: torch.Tensor) -> torch.Tensor:
    return model(tensor)


@torch.inference_mode()
def _infer_tiled(
    model: SAFMN, tensor: torch.Tensor, tile: int, tile_pad: int, scale: int
) -> torch.Tensor:
    """Тайлинг с overlap, каждый тайл инференсится с паддингом и обрезается до своих границ."""
    _, _, h, w = tensor.shape
    out = tensor.new_zeros((1, 3, h * scale, w * scale))

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = y, x
            y2, x2 = min(y + tile, h), min(x + tile, w)
            py1, px1 = max(y1 - tile_pad, 0), max(x1 - tile_pad, 0)
            py2, px2 = min(y2 + tile_pad, h), min(x2 + tile_pad, w)

            patch = tensor[:, :, py1:py2, px1:px2]
            sr_patch = model(patch)

            crop_top = (y1 - py1) * scale
            crop_left = (x1 - px1) * scale
            crop_h = (y2 - y1) * scale
            crop_w = (x2 - x1) * scale
            sr_crop = sr_patch[:, :, crop_top : crop_top + crop_h, crop_left : crop_left + crop_w]

            out[:, :, y1 * scale : y2 * scale, x1 * scale : x2 * scale] = sr_crop
    return out


def upscale_image(
    model: SAFMN,
    img_bgr: np.ndarray,
    device: torch.device,
    tile: int = 0,
    tile_pad: int = 16,
    use_fp16: bool = False,
) -> np.ndarray:
    """Принимает BGR uint8 как cv2.imread и возвращает BGR uint8 увеличенный в scale раз."""
    tensor = _img_to_tensor(img_bgr, device)
    autocast_dtype = torch.float16 if (use_fp16 and device.type == "cuda") else None
    ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with ctx:
        sr = (
            _infer_tiled(model, tensor, tile, tile_pad, model.upscaling_factor)
            if tile > 0
            else _infer_full(model, tensor)
        )
    return _tensor_to_img(sr.float())
