from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger("sr_eval")

_NON_METRIC_COLS = {"name", "h", "w"}


def write_results(df: pd.DataFrame, out_dir: Path) -> dict[str, dict[str, float]]:
    """Сохранить per-image CSV и summary JSON. Вернуть агрегаты для логирования."""
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_path = out_dir / "per_image.csv"
    df.to_csv(per_image_path, index=False)

    metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]
    summary_df = df[metric_cols].agg(["mean", "std", "median", "min", "max"])
    summary: dict[str, dict[str, float]] = {
        col: {stat: float(summary_df.loc[stat, col]) for stat in summary_df.index}
        for col in metric_cols
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    log.info("aggregate.written", per_image=str(per_image_path), summary=str(summary_path))
    return summary
