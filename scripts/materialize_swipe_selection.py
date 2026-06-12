"""Materialize images selected in quality_swipe_selected.json.

Use this after exporting JSON from data/review/quality_swipe.html. The default
mode creates symlinks, which is fast and does not duplicate image data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _safe_name(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(char if char in allowed else "_" for char in text)


def _selected_items(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict) and isinstance(payload.get("selected"), list):
        return [item for item in payload["selected"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise SystemExit("Selection JSON must contain a top-level selected list.")


def _source_path(item: dict[str, object], base_dir: Path | None) -> Path:
    raw_path = item.get("absPath") or item.get("path") or item.get("src")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("selection item has no absPath/path/src")
    path = Path(raw_path)
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"relative path needs --base-dir: {raw_path}")
        path = base_dir / path
    return path.resolve()


def _link_or_copy(source: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "symlink":
        dest.symlink_to(source)
    elif mode == "hardlink":
        dest.hardlink_to(source)
    elif mode == "copy":
        shutil.copy2(source, dest)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/selected"))
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--base-dir", type=Path, help="Resolve relative paths from this directory.")
    parser.add_argument("--group-by-bucket", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = json.loads(args.selection_json.read_text(encoding="utf-8"))
    items = _selected_items(payload)
    base_dir = args.base_dir.resolve() if args.base_dir else None
    output_dir = args.output_dir.resolve()

    written = 0
    missing = 0
    for index, item in enumerate(items, start=1):
        try:
            source = _source_path(item, base_dir)
        except ValueError as exc:
            print(f"skip index={index} reason={exc}")
            missing += 1
            continue

        if not source.exists():
            print(f"missing source={source}")
            missing += 1
            continue

        bucket = str(item.get("bucket") or "selected")
        prefix = _safe_name(bucket) if args.group_by_bucket else ""
        name = f"{index:05d}_{_safe_name(source.name)}"
        dest = output_dir / prefix / name if args.group_by_bucket else output_dir / name
        _link_or_copy(source, dest, args.mode)
        written += 1

    print(f"written={written}")
    print(f"missing={missing}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
