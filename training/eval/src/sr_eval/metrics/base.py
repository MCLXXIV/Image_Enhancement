from __future__ import annotations

from typing import Protocol

import numpy as np


class Metric(Protocol):
    """Контракт метрики, принимает sr и gt в формате HWC uint8 RGB одного размера."""

    name: str

    def compute(self, sr: np.ndarray, gt: np.ndarray) -> float: ...
