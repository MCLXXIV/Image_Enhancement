"""Скачать веса трёх ML-моделей пайплайна в data/weights/.

Модели:
  - Real-SAFMN++ (SR + restoration): SAFMN_L_Real_LSDIR_x4.pth (HuggingFace, dim=128)
  - Retinexformer (low-light): LOL_v2_real.pth (Google Drive, gdown)
  - SCUNet (restoration scale=1): scunet_color_real_psnr.pth (config=4x7, dim=64)

После скачивания печатает готовые значения env-переменных для сервиса.
Для Retinexformer нужен пакет gdown (есть в зависимостях dev).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# kind: "url" для прямого скачивания, "gdrive" для gdown по file id.
WEIGHTS: dict[str, dict[str, str]] = {
    "safmn": {
        "kind": "url",
        "filename": "SAFMN_L_Real_LSDIR_x4.pth",
        "src": "https://huggingface.co/Meloo/SAFMN/resolve/main/SAFMN_L_Real_LSDIR_x4.pth",
        "env": "SAFMN_WEIGHTS_PATH",
    },
    "lowlight": {
        "kind": "gdrive",
        "filename": "LOL_v2_real.pth",
        "src": "1tChRwTfqhs-A67QzG8a9Lrx7qKB3m89K",
        "env": "LOWLIGHT_WEIGHTS_PATH",
    },
    "restore": {
        "kind": "url",
        "filename": "scunet_color_real_psnr.pth",
        "src": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "env": "RESTORE_WEIGHTS_PATH",
    },
}


def _download(spec: dict[str, str], dest: Path, force: bool) -> None:
    if dest.is_file() and not force:
        print(f"skip (exists): {dest}")
        return
    if spec["kind"] == "gdrive":
        import gdown

        gdown.download(id=spec["src"], output=str(dest), quiet=False)
    else:
        print(f"downloading {spec['src']} -> {dest}")
        req = urllib.request.Request(spec["src"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310
            fh.write(resp.read())
    print(f"done: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/weights"))
    parser.add_argument("--only", choices=sorted(WEIGHTS), help="скачать только одну модель")
    parser.add_argument("--force", action="store_true", help="перекачать даже если файл есть")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = {args.only: WEIGHTS[args.only]} if args.only else WEIGHTS

    print("=== env для сервиса (абсолютные пути) ===")
    for key, spec in targets.items():
        dest = args.output_dir / spec["filename"]
        try:
            _download(spec, dest, args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {key}: {exc}")
            continue
        print(f"{spec['env']}={dest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
