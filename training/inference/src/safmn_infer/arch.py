"""SAFMN architecture, vendored из sunny2109/SAFMN без зависимости от basicsr."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, shrinkage_rate: float = 0.25) -> None:
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CCM(nn.Module):
    def __init__(self, dim: int, growth_rate: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ccm(x)


class SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation с multi-scale ветвями через adaptive max pool."""

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        return self.act(self.aggr(torch.cat(out, dim=1))) * x


class AttBlock(nn.Module):
    def __init__(self, dim: int, ffn_scale: float = 2.0) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.safm = SAFM(dim)
        self.ccm = CCM(dim, ffn_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.safm(self.norm1(x)) + x
        x = self.ccm(self.norm2(x)) + x
        return x


class SAFMN(nn.Module):
    """SAFMN backbone, веса Real_SAFMN++ грузятся как обычный state_dict."""

    def __init__(
        self, dim: int = 128, n_blocks: int = 16, ffn_scale: float = 2.0, upscaling_factor: int = 4
    ) -> None:
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x)
