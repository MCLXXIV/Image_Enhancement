"""Build a static swipe-review HTML for selecting good and bad images.

The script scans a local image directory, computes lightweight no-reference
quality metrics, ranks images, and writes a self-contained HTML reviewer. The
HTML references image files by relative path, so it stays small even for large
datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CSV_FIELDS = [
    "path",
    "rel_path",
    "width",
    "height",
    "bytes",
    "brightness_mean",
    "contrast_std",
    "entropy",
    "sharpness_laplacian_var",
    "saturation_mean",
    "colorfulness",
    "underexposed_ratio",
    "overexposed_ratio",
    "channel_imbalance",
    "quality_score",
    "error",
]


@dataclass(frozen=True)
class ScoreJob:
    image_path: str
    source_root: str
    thumb_max_side: int


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _entropy(gray_uint8: np.ndarray) -> float:
    hist = np.bincount(gray_uint8.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _laplacian_variance(gray_norm: np.ndarray) -> float:
    if gray_norm.shape[0] < 3 or gray_norm.shape[1] < 3:
        return 0.0
    center = gray_norm[1:-1, 1:-1]
    laplacian = (
        -4.0 * center
        + gray_norm[:-2, 1:-1]
        + gray_norm[2:, 1:-1]
        + gray_norm[1:-1, :-2]
        + gray_norm[1:-1, 2:]
    )
    return float(laplacian.var())


def _colorfulness(rgb: np.ndarray) -> float:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    mean_root = math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return float((std_root + 0.3 * mean_root) / 255.0)


def _quality_score(
    *,
    brightness_mean: float,
    contrast_std: float,
    entropy: float,
    sharpness_laplacian_var: float,
    saturation_mean: float,
    colorfulness: float,
    underexposed_ratio: float,
    overexposed_ratio: float,
    channel_imbalance: float,
    width: int,
    height: int,
) -> float:
    exposure_score = 1.0 - _clamp01(abs(brightness_mean - 0.55) / 0.42)
    contrast_score = _clamp01((contrast_std - 0.08) / 0.20)
    entropy_score = _clamp01((entropy - 4.2) / 3.0)
    sharpness_score = _clamp01(
        math.log1p(sharpness_laplacian_var * 8000.0) / math.log1p(120.0)
    )
    saturation_score = _clamp01((saturation_mean - 0.04) / 0.42)
    color_score = _clamp01(colorfulness / 0.55)
    size_score = _clamp01(min(width, height) / 720.0)
    clipping_penalty = _clamp01((underexposed_ratio + overexposed_ratio) / 0.45)
    imbalance_penalty = _clamp01(channel_imbalance / 0.28)

    raw = (
        0.24 * exposure_score
        + 0.17 * contrast_score
        + 0.22 * sharpness_score
        + 0.17 * entropy_score
        + 0.07 * saturation_score
        + 0.05 * color_score
        + 0.08 * size_score
        - 0.22 * clipping_penalty
        - 0.08 * imbalance_penalty
    )
    return round(100.0 * _clamp01(raw), 4)


def _empty_error_row(path: Path, source_root: Path, error: str) -> dict[str, str | float | int]:
    try:
        rel_path = path.relative_to(source_root).as_posix()
    except ValueError:
        rel_path = path.name
    return {
        "path": str(path.resolve()),
        "rel_path": rel_path,
        "width": 0,
        "height": 0,
        "bytes": path.stat().st_size if path.exists() else 0,
        "brightness_mean": 0.0,
        "contrast_std": 0.0,
        "entropy": 0.0,
        "sharpness_laplacian_var": 0.0,
        "saturation_mean": 0.0,
        "colorfulness": 0.0,
        "underexposed_ratio": 0.0,
        "overexposed_ratio": 0.0,
        "channel_imbalance": 0.0,
        "quality_score": -1.0,
        "error": error[:300],
    }


def _score_one(job: ScoreJob) -> dict[str, str | float | int]:
    path = Path(job.image_path)
    source_root = Path(job.source_root)
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            image = image.convert("RGB")
            image.thumbnail((job.thumb_max_side, job.thumb_max_side), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.float32)

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

        cmax = rgb.max(axis=2)
        cmin = rgb.min(axis=2)
        saturation = np.divide(cmax - cmin, cmax, out=np.zeros_like(cmax), where=cmax > 0.0)

        channel_means = rgb.reshape(-1, 3).mean(axis=0)
        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        entropy = _entropy(gray_uint8)
        sharpness = _laplacian_variance(gray)
        saturation_mean = float(saturation.mean())
        colorfulness = _colorfulness(rgb)
        underexposed_ratio = float((gray < 0.10).mean())
        overexposed_ratio = float((gray > 0.95).mean())
        channel_imbalance = float(channel_means.max() - channel_means.min()) / 255.0
        quality_score = _quality_score(
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
            entropy=entropy,
            sharpness_laplacian_var=sharpness,
            saturation_mean=saturation_mean,
            colorfulness=colorfulness,
            underexposed_ratio=underexposed_ratio,
            overexposed_ratio=overexposed_ratio,
            channel_imbalance=channel_imbalance,
            width=width,
            height=height,
        )

        return {
            "path": str(path.resolve()),
            "rel_path": path.relative_to(source_root).as_posix(),
            "width": width,
            "height": height,
            "bytes": path.stat().st_size,
            "brightness_mean": round(brightness_mean, 6),
            "contrast_std": round(contrast_std, 6),
            "entropy": round(entropy, 6),
            "sharpness_laplacian_var": round(sharpness, 8),
            "saturation_mean": round(saturation_mean, 6),
            "colorfulness": round(colorfulness, 6),
            "underexposed_ratio": round(underexposed_ratio, 6),
            "overexposed_ratio": round(overexposed_ratio, 6),
            "channel_imbalance": round(channel_imbalance, 6),
            "quality_score": quality_score,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - keep scanning after corrupt files.
        return _empty_error_row(path, source_root, f"{type(exc).__name__}: {exc}")


def _write_scores(path: Path, rows: Iterable[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _score_images(args: argparse.Namespace, scores_csv: Path) -> list[dict[str, str]]:
    if scores_csv.exists() and not args.rescore:
        print(f"reuse_scores={scores_csv}")
        return _read_scores(scores_csv)

    image_paths = _iter_images(args.images_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found under {args.images_dir}")

    print(f"images_found={len(image_paths)}")
    print(f"scores_csv={scores_csv}")
    start = time.monotonic()
    rows: list[dict[str, str | float | int]] = []
    jobs = [
        ScoreJob(
            image_path=str(path),
            source_root=str(args.images_dir.resolve()),
            thumb_max_side=args.thumb_max_side,
        )
        for path in image_paths
    ]

    def append_with_progress(idx: int, row: dict[str, str | float | int]) -> None:
        rows.append(row)
        if idx == 1 or idx % args.progress_every == 0 or idx == len(jobs):
            elapsed = max(0.001, time.monotonic() - start)
            rate = idx / elapsed
            print(f"scored={idx}/{len(jobs)} rate={rate:.1f}/s")

    def score_sequentially() -> None:
        for idx, job in enumerate(jobs, start=1):
            append_with_progress(idx, _score_one(job))

    if args.workers <= 1:
        score_sequentially()
    else:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for idx, row in enumerate(
                    pool.map(_score_one, jobs, chunksize=args.chunksize), start=1
                ):
                    append_with_progress(idx, row)
        except (OSError, PermissionError) as exc:
            print(f"multiprocessing_unavailable={type(exc).__name__}; fallback=sequential")
            rows.clear()
            start = time.monotonic()
            score_sequentially()

    _write_scores(scores_csv, rows)
    return _read_scores(scores_csv)


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return 0


def _html_src(image_path: Path, html_path: Path) -> str:
    rel = os.path.relpath(image_path, html_path.parent)
    return quote(Path(rel).as_posix(), safe="/")


def _make_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _select_candidates(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    valid_rows = [row for row in rows if not row.get("error") and _float(row, "quality_score") >= 0.0]
    valid_rows.sort(key=lambda row: _float(row, "quality_score"), reverse=True)

    best_rows = valid_rows[: args.best_count]
    worst_rows = list(reversed(valid_rows[-args.worst_count :])) if args.worst_count else []

    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    html_path = args.output.resolve()

    def add(row: dict[str, str], bucket: str) -> None:
        image_path = Path(row["path"]).resolve()
        item_id = _make_id(str(image_path))
        if item_id in seen:
            return
        seen.add(item_id)
        candidates.append(
            {
                "id": item_id,
                "bucket": bucket,
                "src": _html_src(image_path, html_path),
                "path": row.get("rel_path") or image_path.name,
                "absPath": str(image_path),
                "width": _int(row, "width"),
                "height": _int(row, "height"),
                "bytes": _int(row, "bytes"),
                "qualityScore": _float(row, "quality_score"),
                "brightness": _float(row, "brightness_mean"),
                "contrast": _float(row, "contrast_std"),
                "entropy": _float(row, "entropy"),
                "sharpness": _float(row, "sharpness_laplacian_var"),
                "underexposed": _float(row, "underexposed_ratio"),
                "overexposed": _float(row, "overexposed_ratio"),
            }
        )

    for row in best_rows:
        add(row, "best")
    for row in worst_rows:
        add(row, "worst")

    if args.shuffle:
        random.Random(args.seed).shuffle(candidates)

    return candidates


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image quality swipe review</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101412;
      --panel: #181d1a;
      --panel-2: #202721;
      --text: #f2f5ef;
      --muted: #aab3aa;
      --border: #384139;
      --keep: #29b36b;
      --reject: #e05b4f;
      --accent: #58a6ff;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
      overflow: hidden;
    }
    button, select {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      height: 38px;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    select { padding: 0 8px; }
    .app {
      min-height: 100%;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .toolbar, .footer {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-color: var(--border);
      background: rgba(24, 29, 26, 0.96);
      min-width: 0;
    }
    .toolbar {
      border-bottom: 1px solid var(--border);
      flex-wrap: wrap;
    }
    .footer {
      border-top: 1px solid var(--border);
      justify-content: center;
      flex-wrap: wrap;
    }
    .stats {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
      min-width: 0;
    }
    .stats strong { color: var(--text); font-weight: 650; }
    .spacer { flex: 1 1 auto; }
    .stage {
      position: relative;
      overflow: hidden;
      display: grid;
      place-items: center;
      padding: 16px;
      touch-action: none;
      min-height: 0;
    }
    .card {
      width: min(980px, calc(100vw - 32px));
      height: min(72vh, calc(100vh - 170px));
      min-height: 360px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #050705;
      display: grid;
      grid-template-rows: 1fr auto;
      overflow: hidden;
      box-shadow: 0 16px 56px rgba(0, 0, 0, 0.45);
      transform-origin: center bottom;
      will-change: transform;
      user-select: none;
    }
    .imageWrap {
      min-height: 0;
      display: grid;
      place-items: center;
      background:
        linear-gradient(45deg, rgba(255,255,255,0.05) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.05) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.05) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.05) 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
    }
    img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }
    .meta {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      background: var(--panel);
      border-top: 1px solid var(--border);
      min-width: 0;
    }
    .path {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
      font-size: 14px;
    }
    .metrics {
      display: flex;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--panel-2);
    }
    .decision {
      position: absolute;
      top: 28px;
      z-index: 2;
      font-size: 40px;
      font-weight: 800;
      letter-spacing: 0;
      opacity: 0;
      pointer-events: none;
      text-transform: uppercase;
    }
    .decision.keep {
      right: 36px;
      color: var(--keep);
    }
    .decision.reject {
      left: 36px;
      color: var(--reject);
    }
    .empty {
      color: var(--muted);
      text-align: center;
      line-height: 1.5;
      padding: 24px;
    }
    .primaryKeep {
      border-color: rgba(41, 179, 107, 0.7);
      background: rgba(41, 179, 107, 0.14);
    }
    .primaryReject {
      border-color: rgba(224, 91, 79, 0.7);
      background: rgba(224, 91, 79, 0.14);
    }
    .targetReached {
      color: var(--keep);
    }
    @media (max-width: 720px) {
      .toolbar, .footer {
        padding: 8px;
        gap: 8px;
      }
      .stage { padding: 8px; }
      .card {
        width: calc(100vw - 16px);
        height: calc(100vh - 176px);
        min-height: 300px;
      }
      .meta {
        grid-template-columns: 1fr;
      }
      .metrics {
        overflow-x: auto;
        padding-bottom: 2px;
      }
      button { padding: 0 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="toolbar">
      <div class="stats">
        <span>Seen <strong id="seenCount">0</strong>/<strong id="totalCount">0</strong></span>
        <span>Kept <strong id="keptCount">0</strong>/<strong id="targetCount">0</strong></span>
        <span>Rejected <strong id="rejectedCount">0</strong></span>
      </div>
      <div class="spacer"></div>
      <select id="bucketFilter" title="Queue filter">
        <option value="all">All</option>
        <option value="best">Best</option>
        <option value="worst">Worst</option>
      </select>
      <button id="undoBtn" type="button" title="Undo last decision">Undo</button>
      <button id="resetBtn" type="button" title="Clear decisions">Reset</button>
      <button id="jsonBtn" type="button" title="Export selected JSON">JSON</button>
      <button id="csvBtn" type="button" title="Export selected CSV">CSV</button>
      <button id="htmlBtn" type="button" title="Export selected HTML">HTML</button>
    </header>

    <main class="stage" id="stage">
      <div class="decision reject" id="rejectStamp">SKIP</div>
      <div class="decision keep" id="keepStamp">KEEP</div>
      <section class="card" id="card">
        <div class="imageWrap"><img id="image" alt=""></div>
        <div class="meta">
          <div class="path" id="path"></div>
          <div class="metrics" id="metrics"></div>
        </div>
      </section>
      <div class="empty" id="empty" hidden>No more images in this queue.</div>
    </main>

    <footer class="footer">
      <button class="primaryReject" id="rejectBtn" type="button" title="Reject current image">&lt; Skip</button>
      <button class="primaryKeep" id="keepBtn" type="button" title="Keep current image">Keep &gt;</button>
    </footer>
  </div>

  <script id="candidate-data" type="application/json">__ITEMS_JSON__</script>
  <script>
    const ITEMS = JSON.parse(document.getElementById("candidate-data").textContent);
    const TARGET = __TARGET__;
    const STORAGE_KEY = "__STORAGE_KEY__";
    const GENERATED_AT = "__GENERATED_AT__";
    const state = loadState();
    let currentId = null;
    let dragStart = null;

    const els = {
      stage: document.getElementById("stage"),
      card: document.getElementById("card"),
      image: document.getElementById("image"),
      path: document.getElementById("path"),
      metrics: document.getElementById("metrics"),
      empty: document.getElementById("empty"),
      seenCount: document.getElementById("seenCount"),
      totalCount: document.getElementById("totalCount"),
      keptCount: document.getElementById("keptCount"),
      rejectedCount: document.getElementById("rejectedCount"),
      targetCount: document.getElementById("targetCount"),
      bucketFilter: document.getElementById("bucketFilter"),
      undoBtn: document.getElementById("undoBtn"),
      resetBtn: document.getElementById("resetBtn"),
      jsonBtn: document.getElementById("jsonBtn"),
      csvBtn: document.getElementById("csvBtn"),
      htmlBtn: document.getElementById("htmlBtn"),
      keepBtn: document.getElementById("keepBtn"),
      rejectBtn: document.getElementById("rejectBtn"),
      keepStamp: document.getElementById("keepStamp"),
      rejectStamp: document.getElementById("rejectStamp"),
    };

    function loadState() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (saved && typeof saved === "object") {
          return {
            decisions: saved.decisions || {},
            history: saved.history || [],
            filter: saved.filter || "all",
          };
        }
      } catch (_err) {}
      return { decisions: {}, history: [], filter: "all" };
    }

    function saveState() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }

    function queueItems() {
      const filter = state.filter;
      return filter === "all" ? ITEMS : ITEMS.filter((item) => item.bucket === filter);
    }

    function currentItem() {
      return ITEMS.find((item) => item.id === currentId) || null;
    }

    function nextUndecided() {
      return queueItems().find((item) => state.decisions[item.id] === undefined) || null;
    }

    function formatNumber(value, digits) {
      if (!Number.isFinite(value)) return "0";
      return value.toFixed(digits);
    }

    function render() {
      els.bucketFilter.value = state.filter;
      const total = queueItems().length;
      const seen = queueItems().filter((item) => state.decisions[item.id] !== undefined).length;
      const kept = ITEMS.filter((item) => state.decisions[item.id] === true).length;
      const rejected = ITEMS.filter((item) => state.decisions[item.id] === false).length;
      els.totalCount.textContent = total;
      els.seenCount.textContent = seen;
      els.keptCount.textContent = kept;
      els.rejectedCount.textContent = rejected;
      els.targetCount.textContent = TARGET;
      els.keptCount.classList.toggle("targetReached", kept >= TARGET);

      const item = currentItem() || nextUndecided();
      currentId = item ? item.id : null;
      const disabled = !item;
      els.card.hidden = disabled;
      els.empty.hidden = !disabled;
      els.keepBtn.disabled = disabled;
      els.rejectBtn.disabled = disabled;
      els.undoBtn.disabled = state.history.length === 0;

      if (!item) return;
      els.image.src = item.src;
      els.image.alt = item.path;
      els.path.textContent = item.path;
      els.metrics.innerHTML = "";
      [
        item.bucket,
        "q " + formatNumber(item.qualityScore, 1),
        item.width + "x" + item.height,
        "b " + formatNumber(item.brightness, 2),
        "c " + formatNumber(item.contrast, 2),
        "s " + formatNumber(item.sharpness, 4),
      ].forEach((text) => {
        const pill = document.createElement("span");
        pill.className = "pill";
        pill.textContent = text;
        els.metrics.appendChild(pill);
      });
      resetCardTransform();
      preloadNext();
    }

    function preloadNext() {
      const next = queueItems().filter((item) => state.decisions[item.id] === undefined)[1];
      if (!next) return;
      const image = new Image();
      image.src = next.src;
    }

    function decide(keep) {
      const item = currentItem();
      if (!item) return;
      state.decisions[item.id] = keep;
      state.history.push({ id: item.id, keep });
      currentId = null;
      saveState();
      render();
    }

    function undo() {
      const last = state.history.pop();
      if (!last) return;
      delete state.decisions[last.id];
      currentId = last.id;
      saveState();
      render();
    }

    function resetAll() {
      if (!confirm("Clear all swipe decisions for this page?")) return;
      state.decisions = {};
      state.history = [];
      currentId = null;
      saveState();
      render();
    }

    function selectedItems() {
      return ITEMS.filter((item) => state.decisions[item.id] === true);
    }

    function downloadBlob(filename, type, text) {
      const blob = new Blob([text], { type });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 500);
    }

    function exportJson() {
      const payload = {
        generatedAt: new Date().toISOString(),
        sourceGeneratedAt: GENERATED_AT,
        targetKept: TARGET,
        selectedCount: selectedItems().length,
        selected: selectedItems(),
      };
      downloadBlob("quality_swipe_selected.json", "application/json", JSON.stringify(payload, null, 2));
    }

    function csvEscape(value) {
      const text = String(value ?? "");
      return '"' + text.replaceAll('"', '""') + '"';
    }

    function exportCsv() {
      const columns = ["id", "bucket", "path", "absPath", "qualityScore", "width", "height"];
      const lines = [columns.join(",")];
      selectedItems().forEach((item) => {
        lines.push(columns.map((column) => csvEscape(item[column])).join(","));
      });
      downloadBlob("quality_swipe_selected.csv", "text/csv", lines.join("\n") + "\n");
    }

    function exportHtml() {
      const selected = selectedItems();
      const cards = selected.map((item) => `
        <figure>
          <img src="${escapeHtml(item.src)}" alt="">
          <figcaption>${escapeHtml(item.path)}<br>bucket=${escapeHtml(item.bucket)} score=${formatNumber(item.qualityScore, 1)}</figcaption>
        </figure>
      `).join("");
      const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Selected images</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
    header { position: sticky; top: 0; background: #181818; padding: 12px 16px; border-bottom: 1px solid #333; }
    main { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; padding: 10px; }
    figure { margin: 0; background: #191919; border: 1px solid #333; border-radius: 8px; overflow: hidden; }
    img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #050505; display: block; }
    figcaption { padding: 8px; color: #bbb; font-size: 12px; overflow-wrap: anywhere; }
  </style>
</head>
<body>
  <header>Selected ${selected.length} / target ${TARGET}</header>
  <main>${cards}</main>
</body>
</html>`;
      downloadBlob("quality_swipe_selected.html", "text/html", html);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function resetCardTransform() {
      els.card.style.transition = "";
      els.card.style.transform = "";
      els.keepStamp.style.opacity = "0";
      els.rejectStamp.style.opacity = "0";
    }

    function setDragTransform(dx) {
      const rotate = dx / 28;
      els.card.style.transform = `translateX(${dx}px) rotate(${rotate}deg)`;
      els.keepStamp.style.opacity = String(Math.max(0, Math.min(1, dx / 140)));
      els.rejectStamp.style.opacity = String(Math.max(0, Math.min(1, -dx / 140)));
    }

    function finishDrag(dx) {
      const threshold = Math.min(160, Math.max(90, window.innerWidth * 0.16));
      if (Math.abs(dx) >= threshold) {
        const keep = dx > 0;
        els.card.style.transition = "transform 160ms ease";
        els.card.style.transform = `translateX(${keep ? window.innerWidth : -window.innerWidth}px) rotate(${keep ? 18 : -18}deg)`;
        window.setTimeout(() => decide(keep), 140);
        return;
      }
      els.card.style.transition = "transform 140ms ease";
      resetCardTransform();
    }

    els.stage.addEventListener("pointerdown", (event) => {
      if (!currentItem()) return;
      dragStart = { x: event.clientX, y: event.clientY };
      els.stage.setPointerCapture(event.pointerId);
      els.card.style.transition = "";
    });

    els.stage.addEventListener("pointermove", (event) => {
      if (!dragStart) return;
      const dx = event.clientX - dragStart.x;
      const dy = event.clientY - dragStart.y;
      if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return;
      setDragTransform(dx);
    });

    function onPointerEnd(event) {
      if (!dragStart) return;
      const dx = event.clientX - dragStart.x;
      dragStart = null;
      finishDrag(dx);
    }

    els.stage.addEventListener("pointerup", onPointerEnd);
    els.stage.addEventListener("pointercancel", onPointerEnd);
    els.keepBtn.addEventListener("click", () => decide(true));
    els.rejectBtn.addEventListener("click", () => decide(false));
    els.undoBtn.addEventListener("click", undo);
    els.resetBtn.addEventListener("click", resetAll);
    els.jsonBtn.addEventListener("click", exportJson);
    els.csvBtn.addEventListener("click", exportCsv);
    els.htmlBtn.addEventListener("click", exportHtml);
    els.bucketFilter.addEventListener("change", () => {
      state.filter = els.bucketFilter.value;
      currentId = null;
      saveState();
      render();
    });

    window.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      if (event.key === "ArrowRight" || event.key.toLowerCase() === "d") decide(true);
      if (event.key === "ArrowLeft" || event.key.toLowerCase() === "a") decide(false);
      if (event.key.toLowerCase() === "u") undo();
    });

    render();
  </script>
</body>
</html>
"""


def _write_html(args: argparse.Namespace, candidates: list[dict[str, object]]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    candidate_hash = hashlib.sha1(
        ",".join(str(item["id"]) for item in candidates).encode("utf-8")
    ).hexdigest()[:16]
    storage_basis = f"{args.images_dir.resolve()}|{candidate_hash}|{args.target_kept}"
    storage_key = "quality-swipe:" + hashlib.sha1(storage_basis.encode("utf-8")).hexdigest()[:16]
    items_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (
        HTML_TEMPLATE.replace("__ITEMS_JSON__", items_json)
        .replace("__TARGET__", str(args.target_kept))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__GENERATED_AT__", generated_at)
    )
    args.output.write_text(html, encoding="utf-8")


def _write_manifest(args: argparse.Namespace, candidates: list[dict[str, object]]) -> None:
    if not args.manifest:
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "imagesDir": str(args.images_dir.resolve()),
        "targetKept": args.target_kept,
        "candidateCount": len(candidates),
        "candidates": candidates,
    }
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("data/photos"))
    parser.add_argument("--output", type=Path, default=Path("data/review/quality_swipe.html"))
    parser.add_argument("--scores-csv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--best-count", type=int, default=5000)
    parser.add_argument("--worst-count", type=int, default=1000)
    parser.add_argument("--target-kept", type=int, default=4000)
    parser.add_argument("--thumb-max-side", type=int, default=384)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Only scan the first N images; useful for smoke tests.")
    parser.add_argument("--rescore", action="store_true", help="Ignore an existing scores CSV.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidates after ranking.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.images_dir = args.images_dir.resolve()
    args.output = args.output.resolve()
    if args.scores_csv is None:
        args.scores_csv = args.output.with_name("quality_scores.csv")
    else:
        args.scores_csv = args.scores_csv.resolve()
    if args.manifest is None:
        args.manifest = args.output.with_name("quality_swipe_manifest.json")
    else:
        args.manifest = args.manifest.resolve()

    if not args.images_dir.exists():
        raise SystemExit(f"images dir does not exist: {args.images_dir}")

    rows = _score_images(args, args.scores_csv)
    candidates = _select_candidates(args, rows)
    if not candidates:
        raise SystemExit("No valid images were scored.")
    _write_html(args, candidates)
    _write_manifest(args, candidates)
    print(f"review_html={args.output}")
    print(f"manifest={args.manifest}")
    print(f"candidates={len(candidates)} best={args.best_count} worst={args.worst_count}")
    print(f"target_kept={args.target_kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
