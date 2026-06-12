"""Главный pipeline: считает метрики, выбирает модель, применяет и проверяет, что стало лучше."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from enhancer.models.base import Enhancer, StageParams
from enhancer.models.registry import build_default_stages
from enhancer.observability import enhance_request_duration_seconds, enhance_requests_total, log
from enhancer.quality.iqa import IqaScorer
from enhancer.quality.metrics import QualityMetrics, compute_metrics
from enhancer.quality.router import RouteDecision, Stage, route
from enhancer.schemas import EnhanceParams

# Порядок применения стадий: сначала тон, потом детали/апскейл.
_STAGE_ORDER: dict[Stage, int] = {
    Stage.EXPOSURE: 0,
    Stage.LOW_LIGHT: 1,
    Stage.RESTORE: 2,
    Stage.SAFMN: 2,
}


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
    psnr_vs_input: float = 0.0
    scale_factor: float = 1.0
    iqa_before: dict[str, float] = field(default_factory=dict)
    iqa_after: dict[str, float] = field(default_factory=dict)


def _override_stages(
    decision: RouteDecision, params: EnhanceParams, available: set[Stage]
) -> list[Stage]:
    """Авто-стадии из роутера + принудительные force_* оверрайды. Дедуп + порядок."""
    stages = list(decision.stages)
    forced = {
        Stage.LOW_LIGHT: params.force_lowlight,
        Stage.EXPOSURE: params.force_exposure,
        Stage.RESTORE: params.force_restore,
        Stage.SAFMN: params.force_safmn,
    }
    for stage, on in forced.items():
        if on and stage not in stages:
            stages.append(stage)
    deduped = [s for s in dict.fromkeys(stages) if s in available]
    return sorted(deduped, key=lambda s: _STAGE_ORDER.get(s, 99))


def _psnr_vs_input(original: np.ndarray, current: np.ndarray) -> float:
    """PSNR между оригиналом и результатом; SR-выход даунскейлится до размера оригинала."""
    if current.shape[:2] != original.shape[:2]:
        h, w = original.shape[:2]
        current = cv2.resize(current, (w, h), interpolation=cv2.INTER_AREA)
    psnr: float = cv2.PSNR(original, current)
    if psnr == float("inf") or psnr != psnr:  # noqa: PLR0124
        return 100.0
    return psnr


class Pipeline:
    def __init__(
        self,
        stages: dict[Stage, Enhancer] | None = None,
        iqa: IqaScorer | None = None,
    ) -> None:
        self.stages = stages if stages is not None else build_default_stages()
        self.iqa = iqa

    def run(self, image_bgr: np.ndarray, params: EnhanceParams | None = None) -> EnhanceResult:
        params = params or EnhanceParams()
        total_t0 = time.perf_counter()
        available = set(self.stages.keys())
        h, w = image_bgr.shape[:2]

        with enhance_request_duration_seconds.labels(stage="assess").time():
            metrics_before = compute_metrics(image_bgr)

        with enhance_request_duration_seconds.labels(stage="route").time():
            decision = route(metrics_before, width=w, height=h, available_stages=available)

        force = bool(params.force)
        stages_to_apply = _override_stages(decision, params, available)
        if not stages_to_apply:
            return self._skipped_result(image_bgr, metrics_before, total_t0)

        stage_latency: dict[str, float] = {}
        current = image_bgr
        empty: StageParams = {}
        for stage in stages_to_apply:
            enhancer = self.stages[stage]
            t0 = time.perf_counter()
            with enhance_request_duration_seconds.labels(stage=stage.value).time():
                current = enhancer.apply(current, empty)
            stage_latency[stage.value] = (time.perf_counter() - t0) * 1000

        with enhance_request_duration_seconds.labels(stage="verify").time():
            metrics_after = compute_metrics(current)
            psnr_input = _psnr_vs_input(image_bgr, current)
            iqa_before, iqa_after = self._score_iqa(image_bgr, current)

        fallback = False
        if not force and self.iqa is not None and not self.iqa.improved(iqa_before, iqa_after):
            log.warning(
                "pipeline.fallback", reason="iqa_regression", before=iqa_before, after=iqa_after
            )
            current, metrics_after, psnr_input = image_bgr, metrics_before, 100.0
            iqa_after = iqa_before
            fallback = True

        total_ms = (time.perf_counter() - total_t0) * 1000
        enhance_request_duration_seconds.labels(stage="total").observe(total_ms / 1000)
        enhance_requests_total.labels(outcome="fallback" if fallback else "enhanced").inc()

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
            psnr_vs_input=psnr_input,
            scale_factor=current.shape[1] / image_bgr.shape[1],
            iqa_before=iqa_before,
            iqa_after=iqa_after,
        )

    def _score_iqa(
        self, original: np.ndarray, current: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        if self.iqa is None or not self.iqa.available:
            return {}, {}
        return self.iqa.score(original), self.iqa.score(current)

    def _skipped_result(
        self, image_bgr: np.ndarray, metrics: QualityMetrics, total_t0: float
    ) -> EnhanceResult:
        total_ms = (time.perf_counter() - total_t0) * 1000
        enhance_request_duration_seconds.labels(stage="total").observe(total_ms / 1000)
        enhance_requests_total.labels(outcome="skipped").inc()
        return EnhanceResult(
            image=image_bgr,
            applied=[],
            skipped=True,
            fallback=False,
            metrics_before=metrics,
            metrics_after=metrics,
            model_versions=self._versions([]),
            total_latency_ms=total_ms,
            psnr_vs_input=100.0,
            scale_factor=1.0,
        )

    def _versions(self, applied: list[Stage]) -> dict[str, str]:
        return {stage.value: self.stages[stage].version for stage in applied}
