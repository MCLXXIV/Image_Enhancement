"""CLI инференса SAFMN по папке изображений, выход совместим с sr-eval по именам и формату PNG."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import structlog
import torch
from tqdm import tqdm

from safmn_infer.inference import load_model, upscale_image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


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


def _list_inputs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="safmn-infer",
        description="Real_SAFMN++ инференс: папка входов → папка апскейлов",
    )
    parser.add_argument("--input", type=Path, required=True, help="Папка с входными изображениями")
    parser.add_argument("--output", type=Path, required=True, help="Куда писать апскейлы (PNG)")
    parser.add_argument("--weights", type=Path, required=True, help="Путь к .pth с весами SAFMN")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--device", default=None, help="cpu/cuda (auto если не указано)")
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Размер тайла в пикселях LR-входа. 0 = без тайлинга. На 1080p +scale=4 без тайла нужен >16ГБ VRAM",
    )
    parser.add_argument("--tile-pad", type=int, default=16, help="Overlap-паддинг между тайлами")
    parser.add_argument("--fp16", action="store_true", help="autocast fp16 на CUDA (быстрее, меньше VRAM)")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписывать существующие выходы")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    log = structlog.get_logger("safmn_infer")

    if not args.input.is_dir():
        log.error("main.bad_input", input=str(args.input))
        return 2
    if not args.weights.is_file():
        log.error("main.weights_missing", weights=str(args.weights))
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info("main.device", device=device_str, scale=args.scale, tile=args.tile, fp16=args.fp16)

    model = load_model(args.weights, scale=args.scale, device=device)
    log.info(
        "main.model_loaded",
        params=sum(p.numel() for p in model.parameters()),
        weights=str(args.weights),
    )

    inputs = _list_inputs(args.input)
    log.info("main.inputs", count=len(inputs))

    skipped = failed = 0
    for src in tqdm(inputs, desc="safmn"):
        dst = args.output / f"{src.stem}.png"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("main.read_failed", path=str(src))
            failed += 1
            continue

        try:
            sr = upscale_image(
                model, img, device, tile=args.tile, tile_pad=args.tile_pad, use_fp16=args.fp16
            )
        except torch.cuda.OutOfMemoryError:
            log.error("main.oom", path=str(src), hint="попробуй --tile 256 или меньше")
            failed += 1
            continue

        if not cv2.imwrite(str(dst), sr):
            log.warning("main.write_failed", path=str(dst))
            failed += 1

    log.info("main.done", processed=len(inputs) - skipped - failed, skipped=skipped, failed=failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
