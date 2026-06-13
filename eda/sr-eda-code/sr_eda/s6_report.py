"""Стадия 6. Итог: всё сливается в одну таблицу, дальше:

1) графики всех ключевых распределений (report/*.png);
2) сегментация "нужен ли SR": skip_good / sr_target / sr_risky / junk_leftover / broken;
3) предложение eval-сетов: eval_real.csv (реальные плохие, стратифицированно)
   и eval_hr_source.csv (доноры эталонов для синтетических пар);
4) corner_candidates.csv — кандидаты в корнер-кейсы по каждой причине;
5) summary.md с ключевыми числами.

Стадия устойчива к отсутствию stage3/4/5: считает на том, что есть
(но качество стратификации без stage4 хуже — там сцены).

ВАЖНО: пороги сегментации в config.py — стартовые. После первого прогона
откройте report/sheets/segment_*.png и подгоните глазами.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from utils import contact_sheet, load_chunks

REP = C.WORK_DIR / "report"
SHEETS = REP / "sheets"


def _try(loader, what):
    try:
        return loader()
    except FileNotFoundError:
        print(f"[warn] {what} не найден — пропускаю")
        return None


def load_all() -> pd.DataFrame:
    df = load_chunks(C.WORK_DIR / "stage1")
    iqa = _try(lambda: load_chunks(C.WORK_DIR / "stage3"), "stage3 (IQA)")
    sem = _try(lambda: pd.read_parquet(C.WORK_DIR / "semantic.parquet"), "stage4 (семантика)")
    dup = _try(lambda: pd.read_parquet(C.WORK_DIR / "duplicates.parquet"), "stage5 (дубликаты)")
    for extra in (iqa, sem, dup):
        if extra is not None:
            df = df.merge(extra, on="id", how="left")
    for col in ["jpeg_q_est", "lap_var", "noise_sigma", "blockiness", "bpp",
                "brightness", "dark_frac", "musiq", "niqe", "zs_conf"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------- графики ----------------

def hist(df, col, title, log=False, clip_q=0.995, bins=60):
    if col not in df:
        return
    x = df[col].dropna()
    if x.empty:
        return
    x = x.clip(upper=x.quantile(clip_q))
    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=bins, log=log)
    plt.title(title)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(REP / f"hist_{col}.png", dpi=130)
    plt.close()


def bar(series, name, title, top=20):
    vc = series.value_counts().head(top)
    if vc.empty:
        return
    plt.figure(figsize=(8, 4))
    vc.plot.bar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(REP / f"bar_{name}.png", dpi=130)
    plt.close()


def make_plots(df):
    hist(df, "long_side", "Длинная сторона, px")
    hist(df, "mp", "Мегапиксели")
    hist(df, "aspect", "Аспект-рейшио (w/h)", clip_q=0.99)
    hist(df, "file_bytes", "Размер файла, байт", log=True)
    hist(df[df["format"] == "JPEG"], "bpp", "Bytes per pixel (только JPEG)")
    hist(df, "jpeg_q_est", "Оценка качества JPEG из таблиц квантования")
    hist(df, "lap_var", "Резкость: variance of Laplacian (станд. размер)", log=True)
    hist(df, "noise_sigma", "Оценка шума (нативный центр-кроп)")
    hist(df, "blockiness", "Блочность JPEG (1.0 = нет)")
    hist(df, "brightness", "Средняя яркость")
    hist(df, "dark_frac", "Доля тёмных пикселей (<30)")
    hist(df, "musiq", "MUSIQ (выше = лучше)")
    hist(df, "niqe", "NIQE (ниже = лучше)")
    bar(df["format"], "format", "Форматы файлов")
    if "zs_label" in df:
        bar(df["zs_label"], "scene", "Zero-shot сцены")
    if "exif_model" in df:
        bar(df["exif_model"].dropna(), "exif_model", "EXIF модели камер (где есть)")


# ---------------- сегментация по нужности SR ----------------

def segment(df) -> pd.Series:
    seg = pd.Series("sr_target", index=df.index, dtype=object)

    seg[df["error"].notna()] = "broken"

    if "zs_label" in df:
        junk = df["zs_label"].isin(["house_plan", "document"]) & (df["zs_conf"].fillna(0) > 0.45)
        seg[junk & (seg != "broken")] = "junk_leftover"

    # уже хорошие: большие + резкие (порог = квантиль резкости среди больших) + не пережатые
    big = df["long_side"].fillna(0) >= C.SEG_LARGE_SIDE
    sharp_thr = df.loc[big, "lap_var"].quantile(0.40) if big.any() else np.inf
    good = big & (df["lap_var"] >= sharp_thr) & (df["jpeg_q_est"].fillna(95) >= C.SEG_JPEG_Q_GOOD)
    seg[good & (seg == "sr_target")] = "skip_good"

    # рискованные: SR скорее добьёт, чем спасёт
    risky = (df["long_side"].fillna(0) < C.SEG_TINY_SIDE)
    risky |= (df["blockiness"].fillna(1.0) >= C.SEG_BLOCKY) & \
             (df["jpeg_q_est"].fillna(100) <= C.SEG_JPEG_Q_BAD)
    if "musiq" in df and df["musiq"].notna().any():
        risky |= df["musiq"] <= df["musiq"].quantile(0.05)
    seg[risky & (seg == "sr_target")] = "sr_risky"
    return seg


# ---------------- eval-сеты ----------------

def stratified_sample(df, strata_cols, n, seed=0):
    strata_cols = [c for c in strata_cols if c in df]
    if not strata_cols or df.empty:
        return df.sample(min(n, len(df)), random_state=seed)
    g = df.groupby(strata_cols, observed=True)
    total = len(df)
    parts = []
    for _, grp in g:  # итерация по группам надёжнее get_group на категориях
        k = max(1, int(round(len(grp) / total * n)))
        parts.append(grp.sample(min(k, len(grp)), random_state=seed))
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    return out


def build_eval_sets(df):
    canon = df["is_canonical"].fillna(True) if "is_canonical" in df else pd.Series(True, index=df.index)
    clean_flags = pd.Series(True, index=df.index)
    for f in ["p_screen", "p_watermark"]:
        if f in df:
            clean_flags &= df[f].fillna(0) < 0.6

    # реальный eval: целевые для SR, стратифицированно по сцене и размеру
    pool = df[(df["segment"] == "sr_target") & canon & clean_flags].copy()
    pool["size_bucket"] = pd.cut(pool["long_side"],
                                 [0, C.SEG_TINY_SIDE, C.SEG_SMALL_SIDE, C.SEG_LARGE_SIDE, np.inf],
                                 labels=["tiny", "small", "mid", "large"])
    eval_real = stratified_sample(pool, ["zs_label", "size_bucket"], C.EVAL_REAL_N)
    cols = [c for c in ["id", "zs_label", "size_bucket", "long_side", "jpeg_q_est",
                        "lap_var", "musiq"] if c in eval_real]
    eval_real[cols].to_csv(REP / "eval_real.csv", index=False)
    contact_sheet(eval_real["id"].head(C.SHEET_ITEMS), SHEETS / "eval_real.png")

    # доноры эталонов: только skip_good, без текста/людей/вотермарок,
    # самые качественные по musiq (или резкости, если IQA не считался)
    hr = df[(df["segment"] == "skip_good") & canon & clean_flags].copy()
    for f in ["p_text", "p_people"]:
        if f in hr:
            hr = hr[hr[f].fillna(0) < 0.5]
    sort_col = "musiq" if ("musiq" in hr and hr["musiq"].notna().sum() > 50) else "lap_var"
    hr = hr.sort_values(sort_col, ascending=False).head(C.EVAL_HR_N * 4)
    eval_hr = stratified_sample(hr, ["zs_label"], C.EVAL_HR_N)
    eval_hr[[c for c in cols if c in eval_hr]].to_csv(REP / "eval_hr_source.csv", index=False)
    contact_sheet(eval_hr["id"].head(C.SHEET_ITEMS), SHEETS / "eval_hr_source.png")
    return eval_real, eval_hr


def build_corner_candidates(df):
    rows = []

    def top(mask_or_series, reason, ascending=False):
        if isinstance(mask_or_series, pd.Series) and mask_or_series.dtype != bool:
            s = mask_or_series.dropna().sort_values(ascending=ascending).head(C.CORNER_TOP_K)
            for rid, v in zip(df.loc[s.index, "id"], s):
                rows.append({"reason": reason, "id": rid, "score": float(v)})
        else:
            for rid in df.loc[mask_or_series, "id"].head(C.CORNER_TOP_K):
                rows.append({"reason": reason, "id": rid, "score": None})

    for f in ["text", "watermark", "people", "night", "mirror", "screen"]:
        if f"p_{f}" in df:
            top(df[f"p_{f}"], f"flag_{f}")
    if "musiq" in df:
        top(df["musiq"], "worst_musiq", ascending=True)
    if "blockiness" in df:
        top(df["blockiness"], "most_blocky")
    top(df["long_side"].where(df["long_side"] > 0), "tiny_images", ascending=True)
    top(df["aspect"].where(df["aspect"].notna()), "extreme_aspect")
    top(df["dark_frac"], "darkest")
    top(df["error"].notna(), "broken_file")

    out = pd.DataFrame(rows)
    out.to_csv(REP / "corner_candidates.csv", index=False)
    for reason, grp in out.groupby("reason"):
        contact_sheet(grp["id"], SHEETS / f"corner_{reason}.png",
                      labels=[f"{s:.2f}" if pd.notna(s) else "" for s in grp["score"]])
    return out


def write_summary(df, eval_real, eval_hr, corners):
    q = lambda col, p: (df[col].quantile(p) if col in df and df[col].notna().any() else float("nan"))
    seg_counts = df["segment"].value_counts()
    lines = [
        "# EDA summary",
        f"- Всего файлов: **{len(df)}**, битых: {int(df['error'].notna().sum())}",
        f"- Форматы: {df['format'].value_counts().to_dict()}",
        f"- Длинная сторона: p10={q('long_side', .1):.0f}, медиана={q('long_side', .5):.0f}, p90={q('long_side', .9):.0f}",
        f"- JPEG quality (оценка): p10={q('jpeg_q_est', .1):.0f}, медиана={q('jpeg_q_est', .5):.0f}, p90={q('jpeg_q_est', .9):.0f}",
        f"- EXIF присутствует у {df['exif_model'].notna().mean():.1%} фото (площадки обычно его режут)",
    ]
    if "dup_group_size" in df:
        extra = int((~df["is_canonical"].fillna(True)).sum())
        lines.append(f"- Дубликаты: лишних копий {extra} ({extra / len(df):.1%}) — исключены из eval-пулов")
    if "zs_label" in df:
        lines.append(f"- Топ сцен: {df['zs_label'].value_counts().head(8).to_dict()}")
    lines += [
        f"- Сегменты SR: {seg_counts.to_dict()}",
        f"- eval_real: {len(eval_real)} фото; eval_hr_source: {len(eval_hr)}; "
        f"corner-кандидатов: {len(corners)}",
        "",
        "Дальше: открыть report/sheets/*.png, проверить сегменты и сцены глазами, "
        "подогнать пороги в config.py и перезапустить только s6.",
    ]
    (REP / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    REP.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    df = load_all()
    make_plots(df)
    df["segment"] = segment(df)
    bar(df["segment"], "segment", "Сегментация: нужен ли SR")
    for s, grp in df.groupby("segment"):
        contact_sheet(grp["id"].sample(min(C.SHEET_ITEMS, len(grp)), random_state=0),
                      SHEETS / f"segment_{s}.png")
    df.to_parquet(C.WORK_DIR / "eda_full.parquet")
    eval_real, eval_hr = build_eval_sets(df)
    corners = build_corner_candidates(df)
    write_summary(df, eval_real, eval_hr, corners)


if __name__ == "__main__":
    main()
