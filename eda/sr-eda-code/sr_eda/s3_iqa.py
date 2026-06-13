"""Стадия 3. No-reference оценка качества (MUSIQ, NIQE) через pyiqa.

По умолчанию — детерминированный сэмпл IQA_SAMPLE фото (MUSIQ на 90к — часы;
для распределений и порогов сегментации сэмпла в 25к достаточно). Поставьте
IQA_SAMPLE = None в config.py, чтобы прогнать всё.

musiq: выше = лучше.  niqe: ниже = лучше (и ненадёжен на гладких стенах —
малотекстурные интерьеры он считает "плохими"; интерпретировать с MUSIQ в паре).

Зависимости:  pip install pyiqa
"""
import numpy as np
import torch
from tqdm import tqdm

import config as C
from utils import ChunkWriter, abs_path, load_chunks, load_pil


def main():
    import pyiqa

    df = load_chunks(C.WORK_DIR / "stage1")
    ids = sorted(df.loc[df["error"].isna(), "id"].tolist())
    if C.IQA_SAMPLE and len(ids) > C.IQA_SAMPLE:
        rng = np.random.default_rng(C.IQA_SEED)
        ids = sorted(rng.choice(np.array(ids, dtype=object),
                                size=C.IQA_SAMPLE, replace=False).tolist())

    writer = ChunkWriter(C.WORK_DIR / "stage3")
    done = writer.done_ids()
    todo = [i for i in ids if i not in done]
    print(f"IQA: всего {len(ids)}, осталось {len(todo)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {name: pyiqa.create_metric(name, device=device)
               for name in C.IQA_METRICS}

    for rid in tqdm(todo, desc="stage3"):
        row = {"id": rid}
        try:
            img = load_pil(abs_path(rid), max_side=C.IQA_MAX_SIDE)
            x = (torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
                 .float().div(255.0).unsqueeze(0).to(device))
            for name, metric in metrics.items():
                if name == "niqe" and min(img.size) < 100:
                    row[name] = None  # NIQE падает/врёт на крошечных картинках
                    continue
                with torch.no_grad():
                    row[name] = float(metric(x).item())
        except Exception as e:  # noqa: BLE001
            row["iqa_error"] = f"{type(e).__name__}: {e}"
        writer.add(row)
        writer.maybe_flush()
    writer.flush()
    print("Готово:", C.WORK_DIR / "stage3")


if __name__ == "__main__":
    main()
