from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import structlog

log = structlog.get_logger("sr_eval")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class Pair(NamedTuple):
    name: str
    gt: Path
    sr: Path


def _index_dir(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if path.stem in index:
            raise ValueError(
                f"Дубликат stem '{path.stem}' в {directory}: {index[path.stem]} и {path}"
            )
        index[path.stem] = path
    return index


def build_pairs(gt_dir: Path, sr_dir: Path) -> list[Pair]:
    """Сопоставление файлов по имени без расширения, форматы могут различаться."""
    gt_index = _index_dir(gt_dir)
    sr_index = _index_dir(sr_dir)

    common = sorted(gt_index.keys() & sr_index.keys())
    only_gt = gt_index.keys() - sr_index.keys()
    only_sr = sr_index.keys() - gt_index.keys()

    if only_gt:
        log.warning("pairing.unmatched_gt", count=len(only_gt), sample=sorted(only_gt)[:5])
    if only_sr:
        log.warning("pairing.unmatched_sr", count=len(only_sr), sample=sorted(only_sr)[:5])

    if not common:
        raise ValueError(f"Нет ни одной пары: GT={gt_dir}, SR={sr_dir}")

    log.info("pairing.built", pairs=len(common), gt_total=len(gt_index), sr_total=len(sr_index))
    return [Pair(name=k, gt=gt_index[k], sr=sr_index[k]) for k in common]
