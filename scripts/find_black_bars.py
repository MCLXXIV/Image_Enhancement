"""Эвристикой (как в enhancer.borders) находит фото с чёрными полосами: CSV кандидатов + HTML для отсева.

HTML открываешь локально, снимаешь галки с ложных, жмёшь «Скачать CSV» -> разметка screenshot (filename) для Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CSV_FIELDS = [
    "filename",
    "rel_path",
    "width",
    "height",
    "top_frac",
    "bottom_frac",
    "left_frac",
    "right_frac",
    "bar_frac",
    "error",
]


@dataclass(frozen=True)
class DetectJob:
    image_path: str
    source_root: str
    thumb_max_side: int
    black_median: int
    min_keep_frac: float


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _content_bounds(is_bar: np.ndarray, min_keep_frac: float) -> tuple[int, int]:
    """Границы самого длинного блока не-полос; короче min_keep_frac стороны - полос по оси нет."""
    n = len(is_bar)
    best_len, best = 0, (0, n)
    i = 0
    while i < n:
        if is_bar[i]:
            i += 1
            continue
        j = i
        while j < n and not is_bar[j]:
            j += 1
        if j - i > best_len:
            best_len, best = j - i, (i, j)
        i = j
    return best if best_len >= n * min_keep_frac else (0, n)


def _detect_one(job: DetectJob) -> dict[str, object]:
    path = Path(job.image_path)
    source_root = Path(job.source_root)
    try:
        rel_path = path.relative_to(source_root).as_posix()
    except ValueError:
        rel_path = path.name
    base: dict[str, object] = {
        "filename": path.name,
        "rel_path": rel_path,
        "width": 0,
        "height": 0,
        "top_frac": 0.0,
        "bottom_frac": 0.0,
        "left_frac": 0.0,
        "right_frac": 0.0,
        "bar_frac": 0.0,
        "error": "",
    }
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            full_w, full_h = image.size
            image.thumbnail((job.thumb_max_side, job.thumb_max_side), Image.Resampling.BILINEAR)
            lum = np.asarray(image, dtype=np.uint8).max(axis=2)
        h, w = lum.shape
        y0, y1 = _content_bounds(np.median(lum, axis=1) <= job.black_median, job.min_keep_frac)
        x0, x1 = _content_bounds(np.median(lum, axis=0) <= job.black_median, job.min_keep_frac)
        kept = (y1 - y0) * (x1 - x0) / (h * w)
        base.update(
            width=full_w,
            height=full_h,
            top_frac=round(y0 / h, 4),
            bottom_frac=round((h - y1) / h, 4),
            left_frac=round(x0 / w, 4),
            right_frac=round((w - x1) / w, 4),
            bar_frac=round(1.0 - kept, 4),
        )
    except Exception as exc:  # noqa: BLE001 - битый файл не должен ронять скан
        base["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return base


def _detect_all(args: argparse.Namespace, paths: list[Path]) -> list[dict[str, object]]:
    jobs = [
        DetectJob(
            image_path=str(p),
            source_root=str(args.images_dir.resolve()),
            thumb_max_side=args.thumb_max_side,
            black_median=args.black_median,
            min_keep_frac=args.min_keep_frac,
        )
        for p in paths
    ]
    rows: list[dict[str, object]] = []
    start = time.monotonic()

    def progress(idx: int) -> None:
        if idx == 1 or idx % args.progress_every == 0 or idx == len(jobs):
            rate = idx / max(0.001, time.monotonic() - start)
            print(f"scanned={idx}/{len(jobs)} rate={rate:.1f}/s")

    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            rows.append(_detect_one(job))
            progress(idx)
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for idx, row in enumerate(pool.map(_detect_one, jobs, chunksize=16), start=1):
                    rows.append(row)
                    progress(idx)
        except (OSError, PermissionError) as exc:
            print(f"multiprocessing_unavailable={type(exc).__name__}; fallback=sequential")
            rows = [_detect_one(job) for job in jobs]
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _html_src(image_rel: str, images_dir: Path, html_path: Path) -> str:
    abs_path = (images_dir / image_rel).resolve()
    rel = os.path.relpath(abs_path, html_path.parent)
    return quote(Path(rel).as_posix(), safe="/")


def _build_html(args: argparse.Namespace, candidates: list[dict[str, object]]) -> None:
    html_path = args.html.resolve()
    html_path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "filename": c["filename"],
            "src": _html_src(str(c["rel_path"]), args.images_dir, html_path),
            "bar": float(c["bar_frac"]),
            "top": float(c["top_frac"]),
            "bottom": float(c["bottom_frac"]),
        }
        for c in candidates
    ]
    data = json.dumps(items, ensure_ascii=False)
    page = _HTML_TEMPLATE.replace("__DATA__", data).replace("__COUNT__", str(len(items)))
    html_path.write_text(page, encoding="utf-8")
    print(f"html={html_path} candidates={len(items)}")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка скриншотов с чёрными полосами</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0f1411; color:#eef2ec;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .bar { position:sticky; top:0; z-index:5; display:flex; gap:12px; align-items:center;
    padding:12px 16px; background:#181d1a; border-bottom:1px solid #333; flex-wrap:wrap; }
  button { background:#202721; color:#eef2ec; border:1px solid #3a443b; border-radius:8px;
    padding:8px 14px; font:inherit; cursor:pointer; }
  button:hover { border-color:#58a6ff; }
  .primary { background:#1f6feb; border-color:#1f6feb; }
  .muted { color:#9aa39a; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px,1fr)); gap:10px; padding:14px; }
  .card { border:1px solid #333; border-radius:8px; overflow:hidden; background:#11160f; cursor:pointer; }
  .card.off { opacity:0.32; }
  .card img { width:100%; height:200px; object-fit:contain; background:
    repeating-conic-gradient(#1a1f1a 0% 25%, #141914 0% 50%) 50% / 18px 18px; display:block; }
  .cap { padding:6px 8px; font-size:12px; display:flex; justify-content:space-between; gap:6px; }
  .cap .fn { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .tick { color:#29b36b; font-weight:700; }
  .card.off .tick { color:#555; }
</style>
</head>
<body>
<div class="bar">
  <strong>Скриншоты с чёрными полосами</strong>
  <span class="muted">всего кандидатов: __COUNT__</span>
  <span class="spacer" style="flex:1"></span>
  <span class="muted">отмечено: <b id="cnt">0</b></span>
  <button id="all">Выбрать все</button>
  <button id="none">Снять все</button>
  <button id="dl" class="primary">Скачать CSV</button>
</div>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const grid = document.getElementById('grid');
const cnt = document.getElementById('cnt');
const state = DATA.map(() => true);

function refresh() { cnt.textContent = state.filter(Boolean).length; }

DATA.forEach((it, i) => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML =
    '<img loading="lazy" src="' + it.src + '">' +
    '<div class="cap"><span class="fn" title="' + it.filename + '">' + it.filename + '</span>' +
    '<span><span class="tick">&#10003;</span> ' + (it.bar*100).toFixed(0) + '%</span></div>';
  card.onclick = () => { state[i] = !state[i]; card.classList.toggle('off', !state[i]); refresh(); };
  grid.appendChild(card);
});
refresh();

document.getElementById('all').onclick = () => {
  state.fill(true); [...grid.children].forEach(c => c.classList.remove('off')); refresh();
};
document.getElementById('none').onclick = () => {
  state.fill(false); [...grid.children].forEach(c => c.classList.add('off')); refresh();
};
document.getElementById('dl').onclick = () => {
  const rows = ['filename,label'];
  DATA.forEach((it, i) => { if (state[i]) rows.push(it.filename + ',screenshot'); });
  const blob = new Blob([rows.join('\n') + '\n'], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'black_bar_screenshots.csv';
  a.click();
  URL.revokeObjectURL(a.href);
};
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, default=Path("data/photos"))
    p.add_argument("--out-csv", type=Path, default=Path("data/labels/black_bar_candidates.csv"))
    p.add_argument("--html", type=Path, default=Path("data/labels/black_bar_review.html"))
    p.add_argument(
        "--min-frac",
        type=float,
        default=0.03,
        help="минимальная доля срезаемых полос, чтобы фото попало в кандидаты",
    )
    p.add_argument("--black-median", type=int, default=4, help="порог медианы яркости полосы (как в borders.py)")
    p.add_argument("--min-keep-frac", type=float, default=0.2)
    p.add_argument("--thumb-max-side", type=int, default=512, help="до какого размера ужимать перед детектом")
    p.add_argument("--limit", type=int, default=0, help="ограничить число сканируемых фото (0 = все)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--progress-every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images_dir.is_dir():
        raise SystemExit(f"images dir not found: {args.images_dir} (сначала `make data`)")

    paths = _iter_images(args.images_dir)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.images_dir}")
    print(f"images_found={len(paths)} workers={args.workers}")

    rows = _detect_all(args, paths)
    _write_csv(args.out_csv, rows)

    errors = [r for r in rows if r["error"]]
    candidates = sorted(
        (r for r in rows if not r["error"] and float(r["bar_frac"]) >= args.min_frac),
        key=lambda r: float(r["bar_frac"]),
        reverse=True,
    )
    _build_html(args, candidates)
    print(
        f"csv={args.out_csv} scanned={len(rows)} candidates={len(candidates)} errors={len(errors)}\n"
        f"открой {args.html}, сними галки с ложных, жми «Скачать CSV» -> black_bar_screenshots.csv"
    )


if __name__ == "__main__":
    main()
