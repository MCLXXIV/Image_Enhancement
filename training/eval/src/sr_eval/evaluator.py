from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import structlog
from tqdm import tqdm

from sr_eval.io_utils import load_rgb, match_size
from sr_eval.metrics.base import Metric
from sr_eval.pairing import Pair

log = structlog.get_logger("sr_eval")


def evaluate(pairs: Sequence[Pair], metrics: Sequence[Metric], tol: int) -> pd.DataFrame:
    """Один проход по парам со всеми метриками, невалидные пары пропускаются с warning."""
    rows: list[dict[str, object]] = []
    skipped = 0
    for pair in tqdm(pairs, desc="eval"):
        try:
            sr_raw = load_rgb(pair.sr)
            gt_raw = load_rgb(pair.gt)
            sr, gt = match_size(sr_raw, gt_raw, tol=tol)
        except (OSError, ValueError) as exc:
            log.warning("evaluator.pair_skipped", name=pair.name, reason=str(exc))
            skipped += 1
            continue

        row: dict[str, object] = {"name": pair.name, "h": sr.shape[0], "w": sr.shape[1]}
        for metric in metrics:
            row[metric.name] = metric.compute(sr, gt)
        rows.append(row)

    if not rows:
        raise RuntimeError("Ни одна пара не прошла оценку, нечего агрегировать.")

    log.info("evaluator.done", evaluated=len(rows), skipped=skipped)
    return pd.DataFrame(rows)