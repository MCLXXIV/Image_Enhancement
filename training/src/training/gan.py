"""PatchGAN discriminator (Pix2Pix-style) и adversarial loss для дообучения SAFMN."""

from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN: классифицирует каждый патч HR/SR как real/fake."""

    def __init__(self, channels: int = 3, base: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_c = base
        for i in range(1, n_layers):
            out_c = min(base * 2**i, 512)
            layers += [
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_c = out_c
        out_c = min(base * 2**n_layers, 512)
        layers += [
            nn.Conv2d(in_c, out_c, 4, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GANLoss(nn.Module):
    """BCEWithLogits на real (target=1) и fake (target=0)."""

    def __init__(self) -> None:
        super().__init__()
        self._bce = nn.BCEWithLogitsLoss()

    def discriminator_loss(
        self, d_real: torch.Tensor, d_fake: torch.Tensor
    ) -> torch.Tensor:
        real_loss = self._bce(d_real, torch.ones_like(d_real))
        fake_loss = self._bce(d_fake, torch.zeros_like(d_fake))
        return (real_loss + fake_loss) * 0.5

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        return self._bce(d_fake, torch.ones_like(d_fake))
