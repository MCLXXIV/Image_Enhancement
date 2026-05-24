"""Главный pipeline: считает метрики, выбирает стадии, применяет и проверяет, что не стало хуже."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from enhancer.models.base import Enhancer, StageParams
from enhancer.models.registry import build_default_stages
from enhancer.observability import enhance_request_duration_seconds, enhance_requests_total, log
from enhancer.quality.metrics import QualityMetrics, compute_metrics
from enhancer.quality.router import RouteDecision, Stage, route
from enhancer.schemas import EnhanceParams


@dataclass
class EnhanceResult:
    image: np.ndarray
    applied: list[Stage]
    skipped: bool
    fallback: bool
    metrics_before: QualityMetrics
    metrics_after: QualityMetrics
    model_versions: dict[str, str]
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0


def _stage_params(p: EnhanceParams) -> StageParams:
    out: StageParams = {}
    if p.gamma is not None:
        out["gamma"] = p.gamma
    if p.clahe_clip is not None:
        out["clahe_clip"] = p.clahe_clip
    if p.sharp_amount is not None:
        out["sharp_amount"] = p.sharp_amount
    if p.denoise_strength is not None:
        out["denoise_strength"] = p.denoise_strength
    return out


def _is_regression(before: QualityMetrics, after: QualityMetrics) -> bool:
    """Стало хуже, если резкость упала больше чем на 25% или контраст ниже 70% от исходного."""
    if before.sharpness_laplacian_var > 0 and (
        after.sharpness_laplacian_var / before.sharpness_laplacian_var < 0.75
    ):
        return True
    return bool(before.contrast_std > 0 and after.contrast_std / before.contrast_std < 0.70)


def _override_stages(decision: RouteDecision, params: EnhanceParams) -> list[Stage]:
    """Если пользователь руками задал параметр стадии в params, принудительно её включает."""
    stages = list(decision.stages)
    if params.gamma is not None and Stage.GAMMA not in stages:
        stages.append(Stage.GAMMA)
    if params.clahe_clip is not None and Stage.CLAHE not in stages:
        stages.append(Stage.CLAHE)
    if params.sharp_amount is not None and Stage.UNSHARP not in stages:
        stages.append(Stage.UNSHARP)
    if params.denoise_strength is not None and Stage.DENOISE not in stages:
        stages.append(Stage.DENOISE)
    return stages


class Pipeline:
    def __init__(self, stages: dict[Stage, Enhancer] | None = None) -> None:
        self.stages = stages if stages is not None else build_default_stages()

    def run(self, image_bgr: np.ndarray, params: EnhanceParams | None = None) -> EnhanceResult:
        params = params or EnhanceParams()
        total_t0 = time.perf_counter()

        with enhance_request_duration_seconds.labels(stage="assess").time():
            metrics_before = compute_metrics(image_bgr)

        with enhance_request_duration_seconds.labels(stage="route").time():
            decision = route(metrics_before)

        force = bool(params.force)
        stages_to_apply = _override_stages(decision, params)
        if decision.skip and not force and not stages_to_apply:
            metrics_after = metrics_before
            total_ms = (time.perf_counter() - total_t0) * 1000
            enhance_request_duration_seconds.labels(stage="total").observe(total_ms / 1000)
            enhance_requests_total.labels(outcome="skipped").inc()
            return EnhanceResult(
                image=image_bgr,
                applied=[],
                skipped=True,
                fallback=False,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                model_versions=self._versions([]),
                total_latency_ms=total_ms,
            )

        stage_latency: dict[str, float] = {}
        current = image_bgr
        stage_params = _stage_params(params)
        for stage in stages_to_apply:
            enhancer = self.stages[stage]
            t0 = time.perf_counter()
            with enhance_request_duration_seconds.labels(stage=stage.value).time():
                current = enhancer.apply(current, stage_params)
            stage_latency[stage.value] = (time.perf_counter() - t0) * 1000

        with enhance_request_duration_seconds.labels(stage="verify").time():
            metrics_after = compute_metrics(current)

        fallback = False
        if not force and _is_regression(metrics_before, metrics_after):
            log.warning(
                "pipeline.fallback",
                reason="regression",
                sharpness_before=metrics_before.sharpness_laplacian_var,
                sharpness_after=metrics_after.sharpness_laplacian_var,
                contrast_before=metrics_before.contrast_std,
                contrast_after=metrics_after.contrast_std,
            )
            current = image_bgr
            metrics_after = metrics_before
            fallback = True

        total_ms = (time.perf_counter() - total_t0) * 1000
        enhance_request_duration_seconds.labels(stage="total").observe(total_ms / 1000)
        outcome = "fallback" if fallback else "enhanced"
        enhance_requests_total.labels(outcome=outcome).inc()

        return EnhanceResult(
            image=current,
            applied=stages_to_apply,
            skipped=False,
            fallback=fallback,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            model_versions=self._versions(stages_to_apply),
            stage_latency_ms=stage_latency,
            total_latency_ms=total_ms,
        )

    def _versions(self, applied: list[Stage]) -> dict[str, str]:
        return {stage.value: self.stages[stage].version for stage in applied}
