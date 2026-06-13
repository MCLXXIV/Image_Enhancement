"""Общий конфиг EDA-пайплайна.

На Kaggle: если в /kaggle/input подключён ровно один датасет, путь
подхватится автоматически. Иначе задайте через переменную окружения:
    import os; os.environ["EDA_DATA_DIR"] = "/kaggle/input/<slug-датасета>"
"""
from pathlib import Path
import os


def _autodetect_kaggle_input():
    base = Path("/kaggle/input")
    if base.exists():
        subs = [d for d in base.iterdir() if d.is_dir()]
        if len(subs) == 1:
            return subs[0]
    return None


DATA_DIR = Path(os.environ.get("EDA_DATA_DIR") or _autodetect_kaggle_input() or "./data")
WORK_DIR = Path(
    os.environ.get("EDA_WORK_DIR")
    or ("/kaggle/working/eda" if Path("/kaggle/working").exists() else "./eda_out")
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ---------- stage 1: технический скан + деградации ----------
ANALYSIS_MAX_SIDE = 1024   # резкость/яркость считаем на стандартизированном размере
NOISE_CROP = 512           # шум — на НАТИВНОМ центр-кропе (даунскейл убивает шум)
BLOCKINESS_CROP = 1024     # blockiness — на нативном разрешении, центр-кроп по сетке 8x8
HUGE_MP = 36.0             # >36 МП (панорамы): не декодим в полный numpy, нативные метрики пропускаем
N_WORKERS = os.cpu_count() or 4
FLUSH_EVERY = 2000         # чекпоинт каждые N изображений

# ---------- stage 2: CLIP-эмбеддинги ----------
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMB_BATCH = 256
EMB_PART = 8192            # размер части (резюмирование по частям)

# ---------- stage 3: no-reference IQA ----------
IQA_METRICS = ["musiq", "niqe"]   # musiq: выше=лучше; niqe: ниже=лучше
IQA_MAX_SIDE = 768
IQA_SAMPLE = 25000         # None = весь датасет (в ~3.5 раза дольше)
IQA_SEED = 42

# ---------- stage 4: семантика ----------
UMAP_NEIGHBORS = 30
HDBSCAN_MIN_CLUSTER = 150
KMEANS_FALLBACK_K = 40
SHEET_ITEMS = 24           # картинок на контактный лист

# ---------- stage 5: дубликаты ----------
PHASH_HAMMING_T = 6        # порог Хэмминга для phash (64 бита)
PHASH_BUCKET_CAP = 300     # защита от вырожденных хэшей (однотонные стены)
EMB_DUP_COS = 0.965        # порог косинусной близости эмбеддингов
EMB_DUP_TOPK = 6
PHASH_CONFIRM_COS = 0.90   # phash-пара подтверждается эмбеддингами, если они есть

# ---------- stage 6: сегментация по нужности SR и eval-сеты ----------
SEG_LARGE_SIDE = 1600
SEG_SMALL_SIDE = 1000
SEG_TINY_SIDE = 500
SEG_JPEG_Q_GOOD = 88
SEG_JPEG_Q_BAD = 60
SEG_BLOCKY = 1.12
EVAL_REAL_N = 150          # реальный eval-сет (без эталона)
EVAL_HR_N = 200            # доноры эталонов для синтетических пар
CORNER_TOP_K = 40          # кандидатов в корнер-кейсы на каждую причину
