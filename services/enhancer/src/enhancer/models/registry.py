"""Фабрика стадий: грузит ML-модели (SAFMN / Zero-DCE++ / SCUNet) если веса доступны."""

from __future__ import annotations

from pathlib import Path

from enhancer.models.base import Enhancer
from enhancer.observability import log
from enhancer.quality.router import Stage
from enhancer.settings import settings


def _try_build_safmn() -> Enhancer | None:
    """Real-SAFMN++ (SR + restoration). None если веса не заданы/отсутствуют."""
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
            dim=settings.safmn_dim,
            n_blocks=settings.safmn_n_blocks,
            ffn_scale=settings.safmn_ffn_scale,
        )
        log.info("safmn.loaded", weights=str(weights), scale=settings.safmn_scale)
        return stage
    except Exception as exc:
        log.exception("safmn.load_failed", error=str(exc))
        return None


def _try_build_lowlight() -> Enhancer | None:
    """Retinexformer (low-light / экспозиция / цвет). None если веса не заданы/отсутствуют."""
    if not settings.lowlight_weights_path:
        log.info("lowlight.not_configured", hint="set LOWLIGHT_WEIGHTS_PATH to enable low-light")
        return None
    weights = Path(settings.lowlight_weights_path)
    if not weights.is_file():
        log.warning("lowlight.weights_missing", path=str(weights))
        return None
    try:
        from enhancer.models.lowlight import LowLightEnhancer

        num_blocks = [int(x) for x in settings.lowlight_num_blocks.split(",") if x.strip()]
        stage = LowLightEnhancer(
            weights_path=weights,
            device=settings.lowlight_device,
            n_feat=settings.lowlight_n_feat,
            stage=settings.lowlight_stage,
            num_blocks=num_blocks,
        )
        log.info("lowlight.loaded", weights=str(weights))
        return stage
    except Exception as exc:
        log.exception("lowlight.load_failed", error=str(exc))
        return None


def _try_build_restore() -> Enhancer | None:
    """SCUNet (restoration scale=1). None если веса не заданы/отсутствуют."""
    if not settings.restore_weights_path:
        log.info("restore.not_configured", hint="set RESTORE_WEIGHTS_PATH to enable restoration")
        return None
    weights = Path(settings.restore_weights_path)
    if not weights.is_file():
        log.warning("restore.weights_missing", path=str(weights))
        return None
    try:
        from enhancer.models.restore import RestorationEnhancer

        config = [int(x) for x in settings.restore_config.split(",") if x.strip()]
        stage = RestorationEnhancer(
            weights_path=weights,
            device=settings.restore_device,
            dim=settings.restore_dim,
            config=config,
            tile=settings.restore_tile,
        )
        log.info("restore.loaded", weights=str(weights))
        return stage
    except Exception as exc:
        log.exception("restore.load_failed", error=str(exc))
        return None


def build_default_stages() -> dict[Stage, Enhancer]:
    """Собирает доступные ML-стадии. Формульных CV-стадий больше нет."""
    stages: dict[Stage, Enhancer] = {}
    builders: list[tuple[Stage, Enhancer | None]] = [
        (Stage.LOW_LIGHT, _try_build_lowlight()),
        (Stage.RESTORE, _try_build_restore()),
        (Stage.SAFMN, _try_build_safmn()),
    ]
    for stage, enhancer in builders:
        if enhancer is not None:
            stages[stage] = enhancer
    return stages
