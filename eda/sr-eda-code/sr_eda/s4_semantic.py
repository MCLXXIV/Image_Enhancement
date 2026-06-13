"""Стадия 4. Семантика поверх готовых эмбеддингов (stage2):

1) zero-shot классификация сцены (кухня/санузел/фасад/аэро/планировка/...) —
   это и есть основа стратификации eval-сета; заодно ловит хлам, прошедший
   мимо Qwen (планировки, документы, скриншоты);
2) zero-shot флаги проблемного контента (текст, вотермарки, люди, ночь,
   зеркала, "фото экрана") — кандидаты в корнер-кейсы;
3) кластеризация UMAP(10) -> HDBSCAN (fallback: MiniBatchKMeans) — ловит
   группы, для которых мы не придумали промпт;
4) контактные листы по каждой сцене/флагу/кластеру — для проверки глазами.

Важно: zero-shot CLIP на МЕЛКИЙ текст (номер дома вдали) слаб — флаг text
ловит крупные вывески/баннеры. Если текст окажется важным корнер-кейсом,
добавить отдельный детектор текста (например, DBNet) следующим шагом.
То же с лицами: при желании заменить флаг people детектором (facenet-pytorch).

Зависимости:  pip install open_clip_torch umap-learn
"""
import numpy as np
import pandas as pd
import torch

import config as C
from utils import contact_sheet, load_embeddings

SCENE_PROMPTS = {
    "kitchen":     "a photo of a kitchen interior",
    "bathroom":    "a photo of a bathroom or toilet interior",
    "bedroom":     "a photo of a bedroom interior",
    "living_room": "a photo of a living room interior with a sofa",
    "hallway":     "a photo of a hallway or corridor inside an apartment",
    "balcony":     "a photo of a balcony or loggia",
    "empty_room":  "a photo of an empty room under renovation with bare walls",
    "facade":      "a photo of the exterior facade of a residential building",
    "courtyard":   "a photo of a courtyard, street or residential area outdoors",
    "aerial":      "an aerial drone photo of buildings and land",
    "entrance":    "a photo of a building entrance, staircase or lobby",
    "parking":     "a photo of a garage or parking lot",
    "house_plan":  "a floor plan drawing of an apartment",
    "document":    "a scan or screenshot of a document with text",
}

FLAG_PROMPTS = {
    "text":      ("a photo with visible signs, banners or large text",
                  "a photo without any text or signs"),
    "watermark": ("a photo with a watermark or logo overlay on top",
                  "a clean photograph without watermark"),
    "people":    ("a photo with people or a person visible",
                  "a photo of an empty place without people"),
    "night":     ("a photo taken at night or in a very dark room",
                  "a photo taken in bright daylight"),
    "mirror":    ("a photo with a large mirror and reflections",
                  "a photo without mirrors"),
    "screen":    ("a photo of a computer screen, or a photo of another photo",
                  "an original photograph of a real place"),
}


_TEXT_ENCODER = {}


def encode_prompts(prompts, device):
    import open_clip
    if "model" not in _TEXT_ENCODER:
        model, _, _ = open_clip.create_model_and_transforms(
            C.CLIP_MODEL, pretrained=C.CLIP_PRETRAINED, device=device)
        model.eval()
        _TEXT_ENCODER["model"] = model
        _TEXT_ENCODER["tok"] = open_clip.get_tokenizer(C.CLIP_MODEL)
    model, tok = _TEXT_ENCODER["model"], _TEXT_ENCODER["tok"]
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()


def cluster(E32):
    """UMAP(10, cosine) -> HDBSCAN; при сбое — MiniBatchKMeans прямо по эмбеддингам."""
    try:
        import umap
        x10 = umap.UMAP(n_components=10, n_neighbors=C.UMAP_NEIGHBORS,
                        metric="cosine", random_state=42).fit_transform(E32)
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=C.HDBSCAN_MIN_CLUSTER).fit_predict(x10)
        method = "umap10+hdbscan"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {type(e).__name__}: {e} -> fallback MiniBatchKMeans")
        from sklearn.cluster import MiniBatchKMeans
        labels = MiniBatchKMeans(n_clusters=C.KMEANS_FALLBACK_K, random_state=42,
                                 n_init=3).fit_predict(E32)
        method = "kmeans"
    return labels, method


def umap2d(E32):
    try:
        import umap
        return umap.UMAP(n_components=2, n_neighbors=C.UMAP_NEIGHBORS,
                         metric="cosine", random_state=42).fit_transform(E32)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(E32)


def main():
    ids, E = load_embeddings()
    E32 = E.astype(np.float32)
    bad = np.linalg.norm(E32, axis=1) < 0.5  # нулевые векторы от сбоев загрузки
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- zero-shot сцены ----
    scene_names = list(SCENE_PROMPTS)
    T = encode_prompts([SCENE_PROMPTS[k] for k in scene_names], device)
    logits = 100.0 * (E32 @ T.T)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    zs_idx = probs.argmax(axis=1)
    out = pd.DataFrame({
        "id": ids,
        "zs_label": [scene_names[i] for i in zs_idx],
        "zs_conf": probs.max(axis=1),
    })

    # ---- zero-shot флаги (контраст пары промптов) ----
    flag_names = list(FLAG_PROMPTS)
    flat = [p for pair in FLAG_PROMPTS.values() for p in pair]
    F = encode_prompts(flat, device)
    for k, name in enumerate(flag_names):
        sp = E32 @ F[2 * k]
        sn = E32 @ F[2 * k + 1]
        out[f"p_{name}"] = 1.0 / (1.0 + np.exp(-100.0 * (sp - sn)))

    # ---- кластеризация и 2D-карта ----
    labels, method = cluster(E32)
    out["cluster"] = labels
    xy = umap2d(E32)
    out["umap_x"], out["umap_y"] = xy[:, 0], xy[:, 1]
    out.loc[bad, ["zs_label"]] = "load_failed"

    out.to_parquet(C.WORK_DIR / "semantic.parquet")
    print(f"Кластеризация: {method}; кластеров: {len(set(labels)) - (1 if -1 in labels else 0)}; "
          f"шум: {(labels == -1).mean():.1%}")

    # ---- карта датасета ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = C.WORK_DIR / "report"
    rep.mkdir(parents=True, exist_ok=True)
    sub = out.sample(min(40000, len(out)), random_state=0)
    plt.figure(figsize=(10, 8))
    for lbl, grp in sub.groupby("zs_label"):
        plt.scatter(grp.umap_x, grp.umap_y, s=2, alpha=0.4, label=lbl)
    plt.legend(markerscale=6, fontsize=8, ncol=2)
    plt.title("UMAP датасета, цвет = zero-shot сцена")
    plt.savefig(rep / "umap_scenes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- контактные листы для проверки глазами ----
    sheets = rep / "sheets"
    for lbl, grp in out.groupby("zs_label"):
        top = grp.sort_values("zs_conf", ascending=False).head(C.SHEET_ITEMS)
        contact_sheet(top["id"], sheets / f"scene_{lbl}.png",
                      labels=[f"{c:.2f}" for c in top["zs_conf"]])
    for name in flag_names:
        top = out.sort_values(f"p_{name}", ascending=False).head(C.SHEET_ITEMS)
        contact_sheet(top["id"], sheets / f"flag_{name}.png",
                      labels=[f"{v:.2f}" for v in top[f"p_{name}"]])
    big = out[out.cluster >= 0].cluster.value_counts().head(12).index
    for cl in big:
        grp = out[out.cluster == cl].sample(min(C.SHEET_ITEMS, (out.cluster == cl).sum()),
                                            random_state=0)
        contact_sheet(grp["id"], sheets / f"cluster_{cl:03d}.png")
    print("Готово:", C.WORK_DIR / "semantic.parquet", "и", sheets)


if __name__ == "__main__":
    main()
