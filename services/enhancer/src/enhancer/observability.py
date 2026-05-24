import logging
import sys

import structlog
from prometheus_client import Counter, Gauge, Histogram

from enhancer.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("enhancer")

enhance_requests_total = Counter(
    "enhance_requests_total",
    "Total enhance requests by outcome",
    labelnames=("outcome",),
)

enhance_request_duration_seconds = Histogram(
    "enhance_request_duration_seconds",
    "Enhance request duration by pipeline stage",
    labelnames=("stage",),
    buckets=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0),
)

enhance_quality_before = Histogram(
    "enhance_quality_before",
    "Image quality metric value before enhancement",
    labelnames=("metric",),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

enhance_quality_after = Histogram(
    "enhance_quality_after",
    "Image quality metric value after enhancement",
    labelnames=("metric",),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

enhance_model_info = Gauge(
    "enhance_model_info",
    "Currently loaded stage model versions (constant=1, version in label)",
    labelnames=("stage", "version"),
)
