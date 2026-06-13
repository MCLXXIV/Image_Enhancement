from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog

from sr_eval.aggregate import write_results
from sr_eval.evaluator import evaluate
from sr_eval.metrics.base import Metric
from sr_eval.metrics.psnr import PSNRMetric
from sr_eval.metrics.ssim import SSIMMetric
from sr_eval.pairing import build_pairs


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(message)s", stream=sys.stderr)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
    )


def _build_metrics(args: argparse.Namespace) -> list[Metric]:
    metrics: list[Metric] = [
        PSNRMetric(on_y_channel=args.y_channel, crop_border=args.crop_border),
        SSIMMetric(on_y_channel=args.y_channel, crop_border=args.crop_border),
    ]
    if not args.no_lpips:
        try:
            from sr_eval.metrics.lpips_metric import LPIPSMetric

            metrics.append(LPIPSMetric(net=args.lpips_net, device=args.device))
        except ImportError as exc:
            structlog.get_logger("sr_eval").warning(
                "main.lpips_disabled", reason=str(exc), hint="pip install '.[lpips]'"
            )
    return metrics


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sr-eval",
        description="Полный референс-эвал (PSNR/SSIM/LPIPS) для апскейл/энхансер-выходов",
    )
    parser.add_argument("--gt", type=Path, required=True, help="Папка с GT-изображениями")
    parser.add_argument("--sr", type=Path, required=True, help="Папка с upscale/enhanced")
    parser.add_argument("--out", type=Path, default=Path("results"), help="Куда писать CSV/JSON")
    parser.add_argument(
        "--tol",
        type=int,
        default=8,
        help="Допустимое расхождение размеров (px) для центр-кропа; больше дает ошибку",
    )
    parser.add_argument(
        "--y-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Считать PSNR/SSIM на Y-канале BT.601 (как в SR-литературе)",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Срезать N px бордюра перед PSNR/SSIM (обычно = scale, для сопоставимости с бенчмарками)",
    )
    parser.add_argument("--no-lpips", action="store_true", help="Отключить LPIPS (без torch)")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", default=None, help="cpu/cuda (auto если не указано)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    log = structlog.get_logger("sr_eval")

    if not args.gt.is_dir() or not args.sr.is_dir():
        log.error("main.bad_input", gt=str(args.gt), sr=str(args.sr))
        return 2

    pairs = build_pairs(args.gt, args.sr)
    metrics = _build_metrics(args)
    log.info("main.metrics", metrics=[m.name for m in metrics])

    df = evaluate(pairs, metrics, tol=args.tol)
    summary = write_results(df, args.out)

    for metric_name, stats in summary.items():
        log.info("main.summary", metric=metric_name, **{k: round(v, 4) for k, v in stats.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
