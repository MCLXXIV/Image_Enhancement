"""Общие утилиты пайплайна."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import config as C

Image.MAX_IMAGE_PIXELS = None  # в датасете могут быть панорамы — не считаем их бомбой


# ---------- файлы ----------

def list_images(root: Path = None):
    root = Path(root or C.DATA_DIR)
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in C.IMG_EXTS]


def rel_id(p) -> str:
    return Path(p).relative_to(C.DATA_DIR).as_posix()


def abs_path(rid: str) -> Path:
    return C.DATA_DIR / rid


# ---------- резюмируемая запись чанков ----------

class ChunkWriter:
    """Копит строки и сбрасывает их parquet-чанками. После рестарта
    done_ids() говорит, что уже посчитано, — стадию можно перезапускать."""

    def __init__(self, out_dir: Path, key: str = "id"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.buf = []
        self.n = len(list(self.dir.glob("chunk_*.parquet")))

    def done_ids(self):
        ids = set()
        for f in self.dir.glob("chunk_*.parquet"):
            ids.update(pd.read_parquet(f, columns=[self.key])[self.key].tolist())
        return ids

    def add(self, row: dict):
        self.buf.append(row)

    def maybe_flush(self):
        if len(self.buf) >= C.FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        pd.DataFrame(self.buf).to_parquet(self.dir / f"chunk_{self.n:05d}.parquet")
        self.n += 1
        self.buf = []


def load_chunks(out_dir) -> pd.DataFrame:
    files = sorted(Path(out_dir).glob("chunk_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет чанков в {out_dir} — стадия ещё не запускалась?")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------- картинки ----------

def load_pil(path, max_side=None) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def load_embeddings():
    """Эмбеддинги stage2: (список id, матрица NxD float16, L2-нормированная)."""
    emb_dir = C.WORK_DIR / "emb"
    ids = pd.read_parquet(emb_dir / "ids.parquet")["id"].tolist()
    parts = sorted(emb_dir.glob("part_*.npy"))
    if not parts:
        raise FileNotFoundError("Эмбеддинги не найдены — запустите s2_embed.py")
    E = np.concatenate([np.load(p) for p in parts], axis=0)
    assert len(E) == len(ids), f"эмбеддингов {len(E)}, а id {len(ids)} — стадия 2 не дошла до конца"
    return ids, E


# ---------- контактные листы ----------

def contact_sheet(ids, out_png, cols=6, thumb=224, labels=None, max_items=None):
    ids = list(ids)[: (max_items or C.SHEET_ITEMS)]
    if not ids:
        return
    rows = math.ceil(len(ids) / cols)
    canvas = Image.new("RGB", (cols * thumb, rows * thumb), (245, 245, 245))
    drw = ImageDraw.Draw(canvas)
    for k, rid in enumerate(ids):
        try:
            im = ImageOps.fit(load_pil(abs_path(rid), max_side=thumb * 2), (thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (200, 60, 60))
        x, y = (k % cols) * thumb, (k // cols) * thumb
        canvas.paste(im, (x, y))
        if labels is not None:
            drw.rectangle([x, y, x + thumb, y + 14], fill=(0, 0, 0))
            drw.text((x + 3, y + 2), str(labels[k])[:36], fill=(255, 255, 255))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
