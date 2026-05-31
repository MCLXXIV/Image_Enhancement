"""Фабрика стадий: возвращает словарь со всеми загруженными Enhancer'ами."""

from __future__ import annotations

from pathlib import Path

from enhancer.models.base import Enhancer
from enhancer.models.clahe import ClaheEnhancer
from enhancer.models.denoise import NlmDenoiseEnhancer
from enhancer.models.gamma import GammaEnhancer
from enhancer.models.unsharp import UnsharpEnhancer
from enhancer.models.whitebalance import GrayWorldEnhancer
from enhancer.observability import log
from enhancer.quality.router import Stage
from enhancer.settings import settings


def _try_build_safmn() -> Enhancer | None:
    """Возвращает SAFMN-стадию если веса доступны, иначе None и warning в логи."""
    if not settings.safmn_weights_path:
        log.info("safmn.not_configured", hint="set SAFMN_WEIGHTS_PATH to enable SR stage")
        return None
    weights = Path(settings.safmn_weights_path)
    if not weights.is_file():
        log.warning("safmn.weights_missing", path=str(weights))
        return None
    try:
        from enhancer.models.safmn import SAFMNEnhancer

        stage = SAFMNEnhancer(
            weights_path=weights,
            scale=settings.safmn_scale,
            device=settings.safmn_device,
            tile=settings.safmn_tile,
            use_fp16=settings.safmn_fp16,
        )
        log.info(
            "safmn.loaded",
            weights=str(weights),
            scale=settings.safmn_scale,
            device=settings.safmn_device or "auto",
            tile=settings.safmn_tile,
        )
        return stage
    except Exception as exc:
        log.exception("safmn.load_failed", error=str(exc))
        return None


def build_default_stages() -> dict[Stage, Enhancer]:
    stages: dict[Stage, Enhancer] = {
        Stage.GAMMA: GammaEnhancer(),
        Stage.CLAHE: ClaheEnhancer(),
        Stage.WHITE_BALANCE: GrayWorldEnhancer(),
        Stage.UNSHARP: UnsharpEnhancer(),
        Stage.DENOISE: NlmDenoiseEnhancer(),
    }
    safmn = _try_build_safmn()
    if safmn is not None:
        stages[Stage.SAFMN] = safmn
    return stages
