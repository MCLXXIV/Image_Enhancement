"""L1 + опциональный VGG-perceptual loss для дообучения SAFMN."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    """L1 между VGG19 feature maps; первые conv-блоки до relu5_1."""

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for p in vgg.parameters():
            p.requires_grad_(False)
        self._feat = nn.Sequential(*list(vgg.children())[:36]).eval()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.l1_loss(self._feat(self._norm(sr)), self._feat(self._norm(hr)))


class SRLoss(nn.Module):
    """L1 + weight * perceptual; perceptual_weight=0 даёт чистый L1."""

    def __init__(self, perceptual_weight: float = 0.1) -> None:
        super().__init__()
        self._l1 = nn.L1Loss()
        self._perceptual_weight = perceptual_weight
        self._perceptual = VGGPerceptualLoss() if perceptual_weight > 0 else None

    def forward(
        self, sr: torch.Tensor, hr: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l1 = self._l1(sr, hr)
        total = l1
        components = {"l1": float(l1.item())}
        if self._perceptual is not None:
            perc = self._perceptual(sr, hr)
            total = l1 + self._perceptual_weight * perc
            components["perceptual"] = float(perc.item())
        components["total"] = float(total.item())
        return total, components
