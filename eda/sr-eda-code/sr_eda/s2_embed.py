"""Стадия 2. CLIP-эмбеддинги всего датасета (GPU).

Эмбеддинги — главный "мультитул" EDA: на них строится кластеризация,
zero-shot разметка сцен и флагов, и поиск почти-дубликатов. Считаем один
раз, переиспользуем трижды.

Пишет: WORK_DIR/emb/ids.parquet + part_XXXX.npy (float16, L2-нормированные).
Резюмируется по частям: готовые part-файлы пропускаются.

Зависимости:  pip install open_clip_torch
"""
import math

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config as C
from utils import abs_path, load_chunks

Image.MAX_IMAGE_PIXELS = None


class ImgDataset(Dataset):
    def __init__(self, ids, preprocess):
        self.ids = ids
        self.pre = preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        try:
            img = Image.open(abs_path(self.ids[i]))
            img = ImageOps.exif_transpose(img).convert("RGB")
            return self.pre(img)
        except Exception:
            # stage1 уже отсеял битые; редкие новые сбои -> нулевой вектор,
            # такие строки потом видны по нулевой норме эмбеддинга
            return torch.zeros(3, 224, 224)


def main():
    import open_clip

    df = load_chunks(C.WORK_DIR / "stage1")
    ids = sorted(df.loc[df["error"].isna(), "id"].tolist())

    out = C.WORK_DIR / "emb"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_parquet(out / "ids.parquet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        C.CLIP_MODEL, pretrained=C.CLIP_PRETRAINED, device=device)
    model.eval()

    n_parts = math.ceil(len(ids) / C.EMB_PART)
    for part in range(n_parts):
        part_file = out / f"part_{part:04d}.npy"
        if part_file.exists():
            continue
        chunk = ids[part * C.EMB_PART:(part + 1) * C.EMB_PART]
        dl = DataLoader(ImgDataset(chunk, preprocess), batch_size=C.EMB_BATCH,
                        num_workers=2, pin_memory=(device == "cuda"))
        embs = []
        with torch.no_grad():
            for x in tqdm(dl, desc=f"emb part {part + 1}/{n_parts}"):
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=(device == "cuda")):
                    e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                embs.append(e.float().cpu().numpy().astype(np.float16))
        np.save(part_file, np.concatenate(embs, axis=0))
    print("Готово:", out)


if __name__ == "__main__":
    main()
