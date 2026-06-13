"""Стадия 5. Дубликаты и почти-дубликаты. Три сигнала:

1) md5 файла           — байт-в-байт копии;
2) perceptual hash     — те же кадры после ресайза/пересжатия (Хэмминг <= T,
                         кандидаты ищутся бандингом 4x16 бит, не попарно);
3) CLIP-эмбеддинги     — кропы/правки/чуть другой кадр (cos >= порога).

Критическая деталь: phash склонен к ложным срабатываниям на "плоских" фото
(однотонная стена в двух разных квартирах). Поэтому phash-пары дополнительно
подтверждаются эмбеддингами (cos >= PHASH_CONFIRM_COS), если stage2 уже
посчитан. Вырожденные бакеты (> PHASH_BUCKET_CAP кандидатов) пропускаются.

Канонический представитель группы — максимальное разрешение (при равенстве —
больший файл). Всё остальное в группе при сборке eval/train надо исключать,
иначе eval "протечёт" в train.

Выход: WORK_DIR/duplicates.parquet (id, dup_group, dup_group_size, is_canonical)
       + контактные листы крупнейших групп для проверки глазами.
"""
import numpy as np
import pandas as pd

import config as C
from utils import contact_sheet, load_chunks, load_embeddings


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def phash_candidate_pairs(h_int):
    """Кандидаты по бандингу: 64-битный хэш режем на 4 банда по 16 бит;
    пары с Хэммингом <= 6 обязаны совпасть хотя бы в одном банде
    (6 ошибок не могут задеть все 4 банда)."""
    buckets = {}
    for i, h in enumerate(h_int):
        if h is None:
            continue
        for b in range(4):
            buckets.setdefault((b, (h >> (16 * b)) & 0xFFFF), []).append(i)
    pairs = set()
    skipped = 0
    for idxs in buckets.values():
        if len(idxs) > C.PHASH_BUCKET_CAP:
            skipped += 1
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if bin(h_int[i] ^ h_int[j]).count("1") <= C.PHASH_HAMMING_T:
                    pairs.add((min(i, j), max(i, j)))
    if skipped:
        print(f"[warn] пропущено {skipped} вырожденных phash-бакетов "
              f"(однотонные фото); их дубликаты ловятся эмбеддингами")
    return pairs


def embedding_pairs(E32):
    try:
        import faiss
        index = faiss.IndexFlatIP(E32.shape[1])
        index.add(E32)
        D, I = index.search(E32, C.EMB_DUP_TOPK)
    except Exception:
        print("[warn] faiss не найден -> sklearn NearestNeighbors (медленнее, минуты-десятки минут)")
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=C.EMB_DUP_TOPK, metric="cosine").fit(E32)
        dist, I = nn.kneighbors(E32)
        D = 1.0 - dist
    pairs = set()
    for i in range(len(E32)):
        for k in range(I.shape[1]):
            j = int(I[i, k])
            if j != i and float(D[i, k]) >= C.EMB_DUP_COS:
                pairs.add((min(i, j), max(i, j)))
    return pairs


def main():
    df = load_chunks(C.WORK_DIR / "stage1").reset_index(drop=True)
    n = len(df)
    uf = UnionFind(n)

    # --- эмбеддинги (если посчитаны) ---
    # ЯВНЫЕ маппинги, чтобы не путать пространства индексов:
    #   s1_to_emb[stage1_idx] -> emb_idx ;  emb_to_s1[emb_idx] -> stage1_idx
    E32, s1_to_emb, emb_to_s1 = None, {}, {}
    try:
        emb_ids, E = load_embeddings()
        E32 = E.astype(np.float32)
        s1_idx_by_id = {rid: i for i, rid in enumerate(df["id"])}
        for emb_idx, rid in enumerate(emb_ids):
            s1_idx = s1_idx_by_id.get(rid)
            if s1_idx is not None:
                s1_to_emb[s1_idx] = emb_idx
                emb_to_s1[emb_idx] = s1_idx
    except FileNotFoundError:
        print("[warn] эмбеддингов нет — поиск только по md5 и phash")

    # --- 1) точные копии по md5 ---
    n_exact = 0
    for _, grp in df.dropna(subset=["md5"]).groupby("md5"):
        idx = grp.index.tolist()
        for j in idx[1:]:
            uf.union(idx[0], j)
            n_exact += 1

    # --- 2) phash с подтверждением эмбеддингами ---
    h_int = [int(h, 16) if isinstance(h, str) else None for h in df.get("phash", pd.Series([None] * n))]
    n_ph, n_rej = 0, 0
    for i, j in phash_candidate_pairs(h_int):
        if E32 is not None and i in s1_to_emb and j in s1_to_emb:
            cos = float(E32[s1_to_emb[i]] @ E32[s1_to_emb[j]])
            if cos < C.PHASH_CONFIRM_COS:
                n_rej += 1
                continue
        uf.union(i, j)
        n_ph += 1

    # --- 3) почти-дубликаты по эмбеддингам ---
    n_emb = 0
    if E32 is not None:
        for a, b in embedding_pairs(E32):
            uf.union(emb_to_s1[a], emb_to_s1[b])
            n_emb += 1

    # --- сборка групп ---
    roots = np.array([uf.find(i) for i in range(n)])
    df["dup_group"] = roots
    sizes = df.groupby("dup_group")["id"].transform("size")
    df["dup_group_size"] = sizes
    # каноник группы: максимальное разрешение, при равенстве — больший файл
    score = df["mp"].fillna(0) * 1e9 + df["file_bytes"].fillna(0)
    df["is_canonical"] = False
    canon_idx = score.groupby(df["dup_group"]).idxmax()
    df.loc[canon_idx, "is_canonical"] = True

    out = df[["id", "dup_group", "dup_group_size", "is_canonical"]]
    out.to_parquet(C.WORK_DIR / "duplicates.parquet")

    in_groups = int((df["dup_group_size"] > 1).sum())
    n_groups = int((df.groupby("dup_group").size() > 1).sum())
    print(f"Рёбра: md5={n_exact}, phash={n_ph} (отклонено эмбеддингами: {n_rej}), emb={n_emb}")
    print(f"Групп дубликатов: {n_groups}; фото в группах: {in_groups} "
          f"({in_groups / max(n, 1):.1%}); лишних (не каноники): {in_groups - n_groups}")

    sheets = C.WORK_DIR / "report" / "sheets"
    top = (df[df.dup_group_size > 1].groupby("dup_group").size()
           .sort_values(ascending=False).head(8).index)
    for k, g in enumerate(top):
        grp = df[df.dup_group == g]
        contact_sheet(grp["id"], sheets / f"dup_group_{k:02d}.png",
                      labels=["CANON" if c else "" for c in grp["is_canonical"]])


if __name__ == "__main__":
    main()
