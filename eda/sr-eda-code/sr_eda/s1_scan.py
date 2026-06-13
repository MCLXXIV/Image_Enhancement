"""Стадия 1. Технический скан + оценка деградаций. Один декод на файл.

Считает на каждое фото: размеры/формат/EXIF/альфа, оценку JPEG-качества из
таблиц квантования, bytes-per-pixel, резкость (на стандартизированном размере!),
шум (на нативном центр-кропе!), blockiness (на нативном разрешении по сетке 8x8),
яркость, perceptual hash, md5. Битые файлы не валят процесс — пишутся с error
(это сами по себе корнер-кейсы).

Запуск:  python s1_scan.py        Резюмируется автоматически.
"""
import hashlib
import io
from multiprocessing import Pool
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from skimage.restoration import estimate_sigma
from tqdm import tqdm

import config as C
from utils import ChunkWriter, list_images, rel_id

Image.MAX_IMAGE_PIXELS = None

# Стандартная таблица квантования яркости (JPEG Annex K). Сравниваем СУММЫ
# таблиц, а не поэлементно: сумма инвариантна к порядку (zigzag vs natural).
_STD_LUM = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)
_STD_LUM_SUM = float(_STD_LUM.sum())


def est_jpeg_quality(img) -> float | None:
    """Приблизительная оценка качества JPEG (1..100) из таблицы квантования.
    Внимание: если фото пересохраняли несколько раз, видна только ПОСЛЕДНЯЯ
    компрессия. Точность ~±3, на q>96 занижает из-за клиппинга таблиц."""
    qt = getattr(img, "quantization", None)
    if not qt:
        return None
    table = qt.get(0) if 0 in qt else next(iter(qt.values()))
    t = np.asarray(table, dtype=np.float64)
    if t.size != 64 or t.sum() <= 0:
        return None
    scale = 100.0 * t.sum() / _STD_LUM_SUM
    q = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(np.clip(q, 1.0, 100.0))


def blockiness_score(gray_native: np.ndarray) -> float | None:
    """Отношение перепадов на границах блоков 8x8 к перепадам внутри блоков.
    ~1.0 — блочности нет; >~1.1 — заметная. Осмысленно ТОЛЬКО на нативном
    разрешении и только если фото не ресайзили после сжатия."""
    h, w = gray_native.shape
    if min(h, w) < 64:
        return None
    ch, cw = min(h, C.BLOCKINESS_CROP), min(w, C.BLOCKINESS_CROP)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    y0 -= y0 % 8  # выравнивание кропа по сетке JPEG (она привязана к (0,0))
    x0 -= x0 % 8
    g = gray_native[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    boundary = dv[:, 7::8].mean() + dh[7::8, :].mean()
    interior = dv[:, 3::8].mean() + dh[3::8, :].mean()
    return float(boundary / (interior + 1e-6))


def _exif_str(ex, tag):
    v = ex.get(tag)
    if v is None:
        return None
    try:
        return str(v).strip()[:64] or None
    except Exception:
        return None


def analyze_one(path_str: str) -> dict:
    path = Path(path_str)
    row = {"id": rel_id(path), "error": None}
    try:
        data = path.read_bytes()
        row["file_bytes"] = len(data)
        row["md5"] = hashlib.md5(data).hexdigest()

        img = Image.open(io.BytesIO(data))
        fmt = img.format
        row["format"] = fmt
        row["mode"] = img.mode
        row["has_alpha"] = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        row["jpeg_q_est"] = est_jpeg_quality(img) if fmt == "JPEG" else None

        ex = img.getexif()
        row["exif_make"] = _exif_str(ex, 271)
        row["exif_model"] = _exif_str(ex, 272)
        row["exif_software"] = _exif_str(ex, 305)  # Photoshop и т.п. — след редактуры

        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size  # "визуальные" размеры, с учётом EXIF-ориентации
        row.update(width=w, height=h, mp=w * h / 1e6,
                   aspect=w / h, long_side=max(w, h))
        row["bpp"] = len(data) / (w * h)

        huge = (w * h / 1e6) > C.HUGE_MP  # панорамы: нативные метрики пропускаем
        if not huge:
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            row["blockiness"] = blockiness_score(gray) if fmt == "JPEG" else None
            ch, cw = min(C.NOISE_CROP, h), min(C.NOISE_CROP, w)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            row["noise_sigma"] = float(estimate_sigma(
                gray[y0:y0 + ch, x0:x0 + cw], channel_axis=None))
        else:
            row["blockiness"] = None
            row["noise_sigma"] = None

        small = img.copy()
        small.thumbnail((C.ANALYSIS_MAX_SIDE,) * 2, Image.BILINEAR)
        gs = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
        row["lap_var"] = float(cv2.Laplacian(gs, cv2.CV_64F).var())
        row["brightness"] = float(gs.mean())
        row["dark_frac"] = float((gs < 30).mean())
        row["phash"] = str(imagehash.phash(small))
    except Exception as e:  # noqa: BLE001 — битый файл это данные, а не падение
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def _init_worker():
    cv2.setNumThreads(0)  # иначе N_WORKERS * потоки cv2 душат друг друга


def main():
    files = list_images()
    if not files:
        raise SystemExit(f"В {C.DATA_DIR} не найдено изображений. Проверьте EDA_DATA_DIR.")
    writer = ChunkWriter(C.WORK_DIR / "stage1")
    done = writer.done_ids()
    todo = [str(p) for p in files if rel_id(p) not in done]
    print(f"Всего файлов: {len(files)}; уже посчитано: {len(done)}; осталось: {len(todo)}")
    if not todo:
        return
    with Pool(C.N_WORKERS, initializer=_init_worker) as pool:
        for row in tqdm(pool.imap_unordered(analyze_one, todo, chunksize=16),
                        total=len(todo), desc="stage1"):
            writer.add(row)
            writer.maybe_flush()
    writer.flush()
    print("Готово:", C.WORK_DIR / "stage1")


if __name__ == "__main__":
    main()
