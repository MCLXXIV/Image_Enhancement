from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from enhancer import __version__
from enhancer.image_io import build_headers, decode_image, encode_jpeg, observe_quality
from enhancer.observability import (
    configure_logging,
    enhance_model_info,
    enhance_requests_total,
    log,
)
from enhancer.pipeline import Pipeline
from enhancer.quality.iqa import IqaScorer
from enhancer.schemas import EnhanceParams, HealthResponse
from enhancer.settings import settings


def _warmup(pipeline: Pipeline) -> None:
    """Прогон каждой модели на старте, чтобы первый запрос не ловил cold start CUDA-ядер."""
    dummy = np.full((320, 320, 3), 64, dtype=np.uint8)
    for stage, enhancer in pipeline.stages.items():
        try:
            enhancer.apply(dummy, {})
        except Exception as exc:  # noqa: BLE001
            log.warning("warmup.failed", stage=stage.value, error=str(exc))
    if pipeline.iqa is not None and pipeline.iqa.available:
        pipeline.iqa.score(dummy)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    iqa = IqaScorer(device=settings.iqa_device) if settings.iqa_gate_enabled else None
    pipeline = Pipeline(iqa=iqa)
    app.state.pipeline = pipeline
    for stage, enhancer in pipeline.stages.items():
        enhance_model_info.labels(stage=stage.value, version=enhancer.version).set(1)
    _warmup(pipeline)
    log.info("enhancer.startup", version=__version__, stages=[s.value for s in pipeline.stages])
    yield
    log.info("enhancer.shutdown")


app = FastAPI(title="Image Enhancer", version=__version__, lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/enhance")
async def enhance(
    image: Annotated[UploadFile, File()],
    params: Annotated[str | None, Form()] = None,
) -> Response:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="unsupported media type")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        parsed_params = EnhanceParams.model_validate_json(params) if params else EnhanceParams()
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    pipeline: Pipeline = app.state.pipeline
    img = decode_image(raw)

    try:
        result = pipeline.run(img, parsed_params)
    except Exception:
        enhance_requests_total.labels(outcome="error").inc()
        log.exception("pipeline.error")
        raise HTTPException(status_code=500, detail="pipeline failure") from None

    observe_quality(result)
    log.info(
        "enhance.done",
        applied=[s.value for s in result.applied],
        skipped=result.skipped,
        fallback=result.fallback,
        latency_ms=round(result.total_latency_ms, 1),
    )

    return Response(
        content=encode_jpeg(result.image),
        media_type="image/jpeg",
        headers=build_headers(result),
    )
