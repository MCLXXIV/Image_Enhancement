"""No-reference IQA (BRISQUE/NIQE, lower=better) для verify-гейта улучшения."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from enhancer.observability import log

# Метрики, где меньше = лучше. Используются для прод-гейта.
LOWER_IS_BETTER = ("brisque", "niqe")
# Допуск: считаем «не хуже», если деградация по метрике в пределах этого относительного порога.
IMPROVE_TOLERANCE = 0.02


class IqaScorer:
    """Ленивая обёртка над pyiqa. Модели создаются один раз (в lifespan)."""

    def __init__(
        self, device: str | None = None, metrics: tuple[str, ...] = LOWER_IS_BETTER
    ) -> None:
        self._metrics: dict[str, Any] = {}
        self._device: Any = None
        try:
            import pyiqa
            import torch

            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._device = dev
            for name in metrics:
                self._metrics[name] = pyiqa.create_metric(name, device=dev)
            log.info("iqa.loaded", metrics=list(self._metrics))
        except Exception as exc:
            log.warning("iqa.unavailable", error=str(exc))

    @property
    def available(self) -> bool:
        return bool(self._metrics)

    def score(self, image_bgr: np.ndarray) -> dict[str, float]:
        if not self._metrics:
            return {}
        import torch

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        out: dict[str, float] = {}
        for name, metric in self._metrics.items():
            try:
                out[name] = float(metric(tensor.to(self._device) if self._device else tensor))
            except Exception as exc:
                log.warning("iqa.score_failed", metric=name, error=str(exc))
        return out

    def improved(self, before: dict[str, float], after: dict[str, float]) -> bool:
        """True, если after не хуже before по совокупности lower-is-better метрик."""
        common = [m for m in before if m in after]
        if not common:
            return True  # не смогли измерить, не блокируем
        worse = 0
        for m in common:
            b, a = before[m], after[m]
            if b <= 0:
                continue
            if a > b * (1 + IMPROVE_TOLERANCE):
                worse += 1
        return worse < len(common)  # сломал, только если стало хуже по всем метрикам
