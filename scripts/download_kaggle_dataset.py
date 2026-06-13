"""Скачивает Kaggle-датасет фото и раскладывает плоской папкой в data/photos/."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import kagglehub

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _unique_link_name(path: Path, used_names: set[str]) -> str:
    name = path.name
    if name not in used_names:
        used_names.add(name)
        return name

    prefix = path.parent.name or "image"
    candidate = f"{prefix}_{name}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    stem = path.stem
    suffix = path.suffix
    idx = 1
    while True:
        candidate = f"{prefix}_{stem}_{idx}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def _refresh_flat_links(images: list[Path], photos_dir: Path) -> None:
    photos_dir.mkdir(parents=True, exist_ok=True)

    for old_link in photos_dir.iterdir():
        if old_link.is_symlink():
            old_link.unlink()

    used_names: set[str] = set()
    for image in images:
        name = _unique_link_name(image, used_names)
        (photos_dir / name).symlink_to(image.resolve())


def _print_summary(download_path: Path, images: list[Path], photos_dir: Path | None) -> None:
    suffix_counts = Counter(path.suffix.lower() for path in images)
    print(f"download_path={download_path}")
    print(f"images={len(images)}")
    if suffix_counts:
        print(
            "suffixes="
            + ",".join(f"{suffix}:{count}" for suffix, count in sorted(suffix_counts.items()))
        )
    if photos_dir is not None:
        print(f"photos_dir={photos_dir}")
    print("sample=" + ",".join(str(path) for path in images[:5]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dimabimov/aaa-projec")
    parser.add_argument("--output-dir", type=Path, default=Path("data/kaggle/aaa-projec"))
    parser.add_argument("--photos-dir", type=Path, default=Path("data/photos"))
    parser.add_argument(
        "--flat-links",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a flat symlink directory for training --hr-dir.",
    )
    parser.add_argument("--force", action="store_true", help="Force KaggleHub re-download.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    download_path = Path(
        kagglehub.dataset_download(
            args.dataset,
            output_dir=str(args.output_dir),
            force_download=args.force,
        )
    )

    images = _iter_images(download_path)
    photos_dir = args.photos_dir if args.flat_links else None
    if photos_dir is not None:
        _refresh_flat_links(images, photos_dir)

    _print_summary(download_path, images, photos_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
