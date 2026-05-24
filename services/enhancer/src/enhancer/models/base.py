from __future__ import annotations

from typing import Protocol

import numpy as np

StageParams = dict[str, float]


class Enhancer(Protocol):
    name: str
    version: str

    def apply(self, image_bgr: np.ndarray, params: StageParams) -> np.ndarray: ...
