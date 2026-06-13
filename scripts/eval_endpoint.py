from __future__ import annotations

import argparse
import html
import json
import mimetypes
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

# Режим оценки на группу: restore = controlled degradation, nr = no-reference.
DEFAULT_GROUP_MODES = {
    "hr_donor": "restore",
    "lr": "nr",
    "corner_case": "nr",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# Кроп HR до кратного 12 перед downscale: делится на любой scale из {2,3,4}.
CROP_MULTIPLE = 12


@dataclass
class RowResult:
    group: str
    mode: str
    name: str
    status: int
    scenario: str = ""  # для restore-режима: downscale/darken/overexpose/noise
    degrade_amount: str = ""  # фактическая сила деградации (напр. x3, k0.31, s24)
    latency_ms: float = 0.0
    applied: str = ""
    skipped: str = ""
    fallback: str = ""
    scale_factor: float = 0.0
    # после восстановления / улучшения (vs оригинал)
    psnr: float = float("nan")
    ssim: float = float("nan")
    lpips: float = float("nan")
    # до восстановления, испорченное (vs оригинал); для restore-режима
    psnr_before: float = float("nan")
    ssim_before: float = float("nan")
    lpips_before: float = float("nan")
    # no-reference IQA из заголовков ручки; для nr-режима
    brisque_before: float = float("nan")
    brisque_after: float = float("nan")
    niqe_before: float = float("nan")
    niqe_after: float = float("nan")
    error: str = ""
    # пути превью для HTML (role -> относительный путь), в CSV не пишется
    assets: dict[str, str] = field(default_factory=dict)


def build_multipart(image_bytes: bytes, filename: str, params: dict | None) -> tuple[bytes, str]:
    """Сборка multipart/form-data вручную, без requests."""
    boundary = f"----eval-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
    lines: list[bytes] = []
    if params:
        lines += [
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="params"',
            b"Content-Type: application/json",
            b"",
            json.dumps(params).encode(),
        ]
    lines += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode(),
        f"Content-Type: {mime}".encode(),
        b"",
        image_bytes,
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(lines), boundary


def post_enhance(
    url: str, image_bytes: bytes, filename: str, params: dict | None, timeout: float
) -> tuple[int, bytes, dict[str, str], float]:
    """Возвращает (status, body, headers, latency_ms)."""
    body, boundary = build_multipart(image_bytes, filename, params)
    req = urlrequest.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    t0 = time.perf_counter()
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
            # Starlette отдаёт кастомные заголовки в нижнем регистре, нормализуем ключи.
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, data, headers, (time.perf_counter() - t0) * 1000
    except HTTPError as e:
        return e.code, b"", {}, (time.perf_counter() - t0) * 1000
    except (URLError, TimeoutError) as e:
        return 0, b"", {"_err": type(e).__name__}, (time.perf_counter() - t0) * 1000


def _to_gray_f32(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR между двумя uint8 BGR одинакового размера (по всем каналам)."""
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-9:
        return 99.0
    return float(10.0 * np.log10((255.0**2) / mse))


def ssim(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """SSIM по яркости, гауссово окно 11x11 sigma=1.5 (как в оригинальной статье)."""
    a = _to_gray_f32(a_bgr)
    b = _to_gray_f32(b_bgr)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    win = (11, 11)
    mu_a = cv2.GaussianBlur(a, win, 1.5)
    mu_b = cv2.GaussianBlur(b, win, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, win, 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, win, 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, win, 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(ssim_map.mean())


class LpipsScorer:
    """Опциональный LPIPS через pyiqa. Если pyiqa нет, возвращает NaN."""

    def __init__(self, enabled: bool = True) -> None:
        self._metric = None
        if not enabled:
            return
        try:
            import pyiqa  # type: ignore
            import torch

            self._torch = torch
            self._metric = pyiqa.create_metric("lpips", device="cpu")
        except Exception:  # pyiqa/torch/веса недоступны
            self._metric = None

    @property
    def available(self) -> bool:
        return self._metric is not None

    def score(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
        if self._metric is None:
            return float("nan")
        t = self._torch

        def prep(img: np.ndarray):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return t.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

        with t.no_grad():
            return float(self._metric(prep(a_bgr), prep(b_bgr)).item())


def _crop_to_multiple(img: np.ndarray, k: int) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % k, : w - w % k]


# Каждая деградация берёт случайную силу из диапазона (args.rng), чтобы кейсы были
# разными, и возвращает (испорченное изображение, метка силы для отчёта).

def degrade_downscale(hr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    scale = int(args.rng.choice(args.downscale_scales))
    h, w = hr.shape[:2]
    out = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return out, f"x{scale}"


def degrade_darken(hr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    factor = float(args.rng.uniform(*args.darken_range))
    out = np.clip(hr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, f"k{factor:.2f}"


def degrade_overexpose(hr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    factor = float(args.rng.uniform(*args.brighten_range))
    out = np.clip(hr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out, f"k{factor:.2f}"


def degrade_noise(hr: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    sigma = float(args.rng.uniform(*args.noise_range))
    noise = args.rng.normal(0.0, sigma, hr.shape).astype(np.float32)
    out = np.clip(hr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, f"s{sigma:.0f}"


# Сценарий: имя -> (force-флаг модели, функция деградации).
SCENARIOS = {
    "downscale": ("force_safmn", degrade_downscale),
    "darken": ("force_lowlight", degrade_darken),
    "overexpose": ("force_exposure", degrade_overexpose),
    "noise": ("force_restore", degrade_noise),
}


def eval_restore(
    hr_bgr: np.ndarray,
    scenario: str,
    url: str,
    filename: str,
    args: argparse.Namespace,
    lpips: LpipsScorer,
    sink: ReportSink | None,
    row: RowResult,
) -> None:
    """Портим HR по сценарию, восстанавливаем одной моделью (only), считаем FR-метрики."""
    flag, degrade = SCENARIOS[scenario]
    hr = _crop_to_multiple(hr_bgr, CROP_MULTIPLE)
    h, w = hr.shape[:2]
    degraded, row.degrade_amount = degrade(hr, args)

    ok, dg_jpeg = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
    if not ok:
        row.error = "degraded-encode-failed"
        return

    params = {"force": True, "only": True, flag: True}
    status, body, headers, latency = post_enhance(url, dg_jpeg.tobytes(), filename, params, timeout=args.timeout)
    row.status = status
    row.latency_ms = latency
    if status != 200 or not body:
        row.error = headers.get("_err", f"http-{status}")
        return

    row.applied = headers.get("x-enhance-applied", "")
    row.skipped = headers.get("x-enhance-skipped", "")
    row.fallback = headers.get("x-enhance-fallback", "")
    row.scale_factor = _safe_float(headers.get("x-enhance-scale-factor"))

    out = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if out is None:
        row.error = "out-decode-failed"
        return
    restored = out if out.shape[:2] == (h, w) else cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    degraded_hr = (
        degraded
        if degraded.shape[:2] == (h, w)
        else cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)
    )

    row.psnr = psnr(restored, hr)
    row.ssim = ssim(restored, hr)
    row.psnr_before = psnr(degraded_hr, hr)
    row.ssim_before = ssim(degraded_hr, hr)
    if lpips.available:
        row.lpips = lpips.score(restored, hr)
        row.lpips_before = lpips.score(degraded_hr, hr)

    if sink is not None:
        tag = f"{scenario}"
        row.assets["hr"] = sink.save(row.group, filename, f"{tag}_hr", hr)
        row.assets["degraded"] = sink.save(row.group, filename, f"{tag}_degraded", degraded_hr)
        row.assets["restored"] = sink.save(row.group, filename, f"{tag}_restored", restored)


def eval_nr(
    img_bytes: bytes,
    url: str,
    filename: str,
    timeout: float,
    params: dict | None,
    sink: ReportSink | None,
    row: RowResult,
) -> None:
    """Прогон оригинала через полный авто-пайплайн, no-ref IQA из заголовков."""
    status, body, headers, latency = post_enhance(url, img_bytes, filename, params, timeout)
    row.status = status
    row.latency_ms = latency
    if status != 200:
        row.error = headers.get("_err", f"http-{status}")
        return

    row.applied = headers.get("x-enhance-applied", "")
    row.skipped = headers.get("x-enhance-skipped", "")
    row.fallback = headers.get("x-enhance-fallback", "")
    row.scale_factor = _safe_float(headers.get("x-enhance-scale-factor"))

    iqa_before = _safe_json(headers.get("x-enhance-iqa-before"))
    iqa_after = _safe_json(headers.get("x-enhance-iqa-after"))
    row.brisque_before = _safe_float(iqa_before.get("brisque"))
    row.brisque_after = _safe_float(iqa_after.get("brisque"))
    row.niqe_before = _safe_float(iqa_before.get("niqe"))
    row.niqe_after = _safe_float(iqa_after.get("niqe"))

    if sink is not None:
        before = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        after = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if before is not None:
            row.assets["before"] = sink.save(row.group, filename, "before", before)
        if after is not None:
            row.assets["after"] = sink.save(row.group, filename, "after", after)


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _safe_json(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class ReportSink:
    """Сохраняет уменьшенные превью для HTML-отчёта в <out_dir>/assets."""

    def __init__(self, out_dir: Path, preview_max: int = 720, jpeg_q: int = 85) -> None:
        self.out_dir = out_dir
        self.preview_max = preview_max
        self.jpeg_q = jpeg_q

    def save(self, group: str, name: str, role: str, img_bgr: np.ndarray) -> str:
        h, w = img_bgr.shape[:2]
        long = max(h, w)
        if long > self.preview_max:
            k = self.preview_max / long
            img_bgr = cv2.resize(
                img_bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA
            )
        rel = Path("assets") / group / f"{Path(name).stem}__{role}.jpg"
        dst = self.out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        return str(rel)


def _mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]  # отбрасываем NaN
    return statistics.mean(clean) if clean else float("nan")


def _percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return float("nan")
    k = max(0, min(len(clean) - 1, int(round((q / 100) * (len(clean) - 1)))))
    return clean[k]


def _latency_line(ok: list[RowResult]) -> None:
    lat = [r.latency_ms for r in ok]
    if not lat:
        return
    print(
        f"latency ms: p50={statistics.median(lat):.0f} p95={_percentile(lat, 95):.0f} "
        f"p99={_percentile(lat, 99):.0f} max={max(lat):.0f}"
    )
    slo = sum(1 for x in lat if x > 1000)
    print(
        f"SLO p99<=1000ms: {'PASS' if _percentile(lat, 99) <= 1000 else 'FAIL'} "
        f"(нарушений {slo}/{len(lat)})"
    )


def print_summary(rows: list[RowResult]) -> None:
    restore_rows = [r for r in rows if r.mode == "restore"]
    nr_rows = [r for r in rows if r.mode == "nr"]

    if restore_rows:
        print("\n##### Восстановление (синтетическая деградация хороших фото) #####")
        by_sc: dict[str, list[RowResult]] = defaultdict(list)
        for r in restore_rows:
            by_sc[r.scenario].append(r)
        for sc, rs in by_sc.items():
            ok = [r for r in rs if r.status == 200 and not r.error]
            flag = SCENARIOS.get(sc, ("?", None))[0]
            print(f"\n=== {sc} -> {flag} ({len(ok)}/{len(rs)} ок) ===")
            _latency_line(ok)
            print(f"applied: {dict(Counter(r.applied or '-' for r in ok))}")
            if ok:
                print(
                    f"PSNR испорчено={_mean([r.psnr_before for r in ok]):.2f} "
                    f"восстановлено={_mean([r.psnr for r in ok]):.2f} "
                    f"(Δ {_mean([r.psnr - r.psnr_before for r in ok]):+.2f})"
                )
                print(
                    f"SSIM испорчено={_mean([r.ssim_before for r in ok]):.4f} "
                    f"восстановлено={_mean([r.ssim for r in ok]):.4f} "
                    f"(Δ {_mean([r.ssim - r.ssim_before for r in ok]):+.4f})"
                )
                lp = [r.lpips for r in ok if r.lpips == r.lpips]
                if lp:
                    print(
                        f"LPIPS испорчено={_mean([r.lpips_before for r in ok]):.4f} "
                        f"восстановлено={_mean([r.lpips for r in ok]):.4f} "
                        f"(Δ {_mean([r.lpips - r.lpips_before for r in ok]):+.4f}, lower=better)"
                    )
                gain = sum(1 for r in ok if r.psnr > r.psnr_before)
                print(f"модель улучшила PSNR vs испорченного: {gain}/{len(ok)}")

    if nr_rows:
        print("\n##### Реальные плохие фото (no-reference) #####")
        by_group: dict[str, list[RowResult]] = defaultdict(list)
        for r in nr_rows:
            by_group[r.group].append(r)
        for group, rs in by_group.items():
            ok = [r for r in rs if r.status == 200 and not r.error]
            print(f"\n=== {group} ({len(ok)}/{len(rs)} ок) ===")
            _latency_line(ok)
            print(f"applied (авто-роутер): {dict(Counter(r.applied or '-' for r in ok))}")
            print(
                f"skipped: {sum(r.skipped == 'true' for r in ok)} | "
                f"fallback: {sum(r.fallback == 'true' for r in ok)}"
            )
            enhanced = [r for r in ok if r.skipped != "true"]
            if enhanced:
                print(
                    f"BRISQUE before={_mean([r.brisque_before for r in enhanced]):.2f} "
                    f"after={_mean([r.brisque_after for r in enhanced]):.2f} "
                    f"(Δ {_mean([r.brisque_after - r.brisque_before for r in enhanced]):+.2f}, "
                    f"lower=better)"
                )
                print(
                    f"NIQE before={_mean([r.niqe_before for r in enhanced]):.2f} "
                    f"after={_mean([r.niqe_after for r in enhanced]):.2f} "
                    f"(Δ {_mean([r.niqe_after - r.niqe_before for r in enhanced]):+.2f}, "
                    f"lower=better)"
                )


CSV_FIELDS = [
    "group", "mode", "scenario", "degrade_amount", "name", "status", "latency_ms", "applied",
    "skipped", "fallback",
    "scale_factor", "psnr", "ssim", "lpips", "psnr_before", "ssim_before", "lpips_before",
    "brisque_before", "brisque_after", "niqe_before", "niqe_after", "error",
]


def write_csv(rows: list[RowResult], path: Path) -> None:
    import csv

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            d = r.__dict__
            w.writerow(
                {
                    k: ("" if (isinstance(d[k], float) and d[k] != d[k]) else d[k])
                    for k in CSV_FIELDS
                }
            )


def _fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def _delta_span(delta: float, lower_better: bool, digits: int = 2) -> str:
    if delta != delta:
        return '<span class="d">n/a</span>'
    good = (delta < 0) if lower_better else (delta > 0)
    cls = "good" if good else ("bad" if delta != 0 else "d")
    return f'<span class="{cls}">{delta:+.{digits}f}</span>'


def _img_tag(rel: str, caption: str) -> str:
    if not rel:
        return (
            f'<figure class="ph"><div class="miss">нет</div>'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    return (
        f'<figure class="ph"><a href="{rel}" target="_blank">'
        f'<img loading="lazy" src="{rel}"/></a><figcaption>{caption}</figcaption></figure>'
    )


def _badges(row: RowResult) -> str:
    out = []
    if row.skipped == "true":
        out.append('<span class="bdg skip">skipped</span>')
    if row.fallback == "true":
        out.append('<span class="bdg fb">fallback</span>')
    if row.error:
        out.append(f'<span class="bdg err">{html.escape(row.error)}</span>')
    out.append(f'<span class="bdg ap">{html.escape(row.applied or "none")}</span>')
    return " ".join(out)


def _restore_card(row: RowResult) -> str:
    name = html.escape(row.name)
    imgs = (
        _img_tag(row.assets.get("hr", ""), "оригинал")
        + _img_tag(row.assets.get("degraded", ""), "испорчено")
        + _img_tag(row.assets.get("restored", ""), "восстановлено")
    )
    metrics = (
        f'<div class="m">PSNR испорчено {_fmt(row.psnr_before)}, восстановлено {_fmt(row.psnr)} '
        f"({_delta_span(row.psnr - row.psnr_before, lower_better=False)})</div>"
        f'<div class="m">SSIM испорчено {_fmt(row.ssim_before, 4)}, восстановлено {_fmt(row.ssim, 4)} '
        f"({_delta_span(row.ssim - row.ssim_before, lower_better=False, digits=4)})</div>"
        f'<div class="m">LPIPS испорчено {_fmt(row.lpips_before, 4)}, '
        f"восстановлено {_fmt(row.lpips, 4)} "
        f"({_delta_span(row.lpips - row.lpips_before, lower_better=True, digits=4)})</div>"
    )
    sort_key = row.psnr - row.psnr_before
    sk = f"{sort_key:.6f}" if sort_key == sort_key else "0"
    return (
        f'<div class="card" data-scn="{row.scenario}" data-name="{name.lower()}" data-sort="{sk}">'
        f'<div class="hd"><b>{name}</b> '
        f'<span class="bdg scn">{row.scenario} {html.escape(row.degrade_amount)}</span> '
        f'<span class="lat">{row.latency_ms:.0f} мс</span> {_badges(row)}</div>'
        f'<div class="imgs">{imgs}</div>'
        f'<div class="metrics">{metrics}</div></div>'
    )


def _nr_card(row: RowResult) -> str:
    name = html.escape(row.name)
    imgs = _img_tag(row.assets.get("before", ""), "до") + _img_tag(
        row.assets.get("after", ""), "после"
    )
    metrics = (
        f'<div class="m">BRISQUE до {_fmt(row.brisque_before)}, после {_fmt(row.brisque_after)} '
        f"({_delta_span(row.brisque_after - row.brisque_before, lower_better=True)})</div>"
        f'<div class="m">NIQE до {_fmt(row.niqe_before)}, после {_fmt(row.niqe_after)} '
        f"({_delta_span(row.niqe_after - row.niqe_before, lower_better=True)})</div>"
    )
    d = row.brisque_after - row.brisque_before
    sk = f"{d:.6f}" if d == d else "0"
    state = "skipped" if row.skipped == "true" else "fallback" if row.fallback == "true" else "ok"
    return (
        f'<div class="card" data-group="{row.group}" data-state="{state}" '
        f'data-name="{name.lower()}" data-sort="{sk}">'
        f'<div class="hd"><b>{name}</b> <span class="lat">{row.latency_ms:.0f} мс</span> '
        f"{_badges(row)}</div>"
        f'<div class="imgs">{imgs}</div>'
        f'<div class="metrics">{metrics}</div></div>'
    )


def write_html_report(rows: list[RowResult], out_dir: Path) -> None:
    restore_rows = [r for r in rows if r.mode == "restore"]
    nr_rows = [r for r in rows if r.mode == "nr"]

    scns = [s for s in SCENARIOS if any(r.scenario == s for r in restore_rows)]
    scn_btns = "".join(f'<button class="rflt" data-v="{s}">{s}</button>' for s in scns)
    restore_cards = "\n".join(_restore_card(r) for r in restore_rows)

    nr_groups = sorted({r.group for r in nr_rows})
    nr_btns = "".join(f'<button class="nflt" data-v="{g}">{g}</button>' for g in nr_groups)
    nr_cards = "\n".join(_nr_card(r) for r in nr_rows)

    doc = _HTML_TEMPLATE.format(
        n_restore=len(restore_rows),
        n_nr=len(nr_rows),
        scn_btns=scn_btns,
        nr_btns=nr_btns,
        restore_cards=restore_cards,
        nr_cards=nr_cards,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


_HTML_TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Оценка автоулучшения eval_set</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#15171c;color:#e6e8ec;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}}
header{{position:sticky;top:0;background:#1b1e25;padding:12px 16px;border-bottom:1px solid #2c3038;z-index:5}}
h1{{font-size:18px;margin:0 0 10px}}
.tabs{{display:flex;gap:8px;margin-bottom:10px}}
.tab{{background:#2a2f3a;color:#e6e8ec;border:1px solid #3a414e;border-radius:6px;padding:6px 14px;cursor:pointer;font-weight:600}}
.tab.on{{background:#3d6df0;border-color:#3d6df0}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
button.rflt,button.nflt,button.act{{background:#2a2f3a;color:#e6e8ec;border:1px solid #3a414e;border-radius:6px;padding:5px 10px;cursor:pointer}}
button.on{{background:#3d6df0;border-color:#3d6df0}}
input.q{{background:#2a2f3a;color:#e6e8ec;border:1px solid #3a414e;border-radius:6px;padding:5px 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:14px;padding:16px}}
.card{{background:#1b1e25;border:1px solid #2c3038;border-radius:10px;padding:10px}}
.hd{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}}
.lat{{color:#8b93a1;font-size:12px}}
.imgs{{display:flex;gap:6px}}
.ph{{flex:1;margin:0;min-width:0}}
.ph img{{width:100%;border-radius:6px;display:block;background:#000;cursor:zoom-in}}
.ph figcaption{{font-size:11px;color:#8b93a1;text-align:center;margin-top:3px}}
.miss{{aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;background:#23272f;color:#5a6373;border-radius:6px}}
.metrics{{margin-top:8px;font-size:12px;color:#c2c7d0}}
.m{{margin:2px 0}}
.good{{color:#3ec27a}}.bad{{color:#e8625a}}.d{{color:#8b93a1}}
.bdg{{font-size:11px;padding:1px 7px;border-radius:10px;background:#2a2f3a;color:#c2c7d0}}
.bdg.ap{{background:#26344f;color:#8fb4ff}}
.bdg.scn{{background:#2c3f2c;color:#9bd49b}}
.bdg.skip{{background:#3a3f49;color:#aeb6c2}}
.bdg.fb{{background:#4a3a1f;color:#e8b35a}}
.bdg.err{{background:#4a2222;color:#e8625a}}
.pane{{display:none}}
.pane.on{{display:block}}
.hidden{{display:none}}
</style></head>
<body>
<header>
<h1>Оценка автоулучшения, eval_set</h1>
<div class="tabs">
<button class="tab on" data-pane="restore">Восстановление ({n_restore})</button>
<button class="tab" data-pane="nr">Реальные плохие ({n_nr})</button>
</div>
<div class="controls" id="ctl-restore">
<button class="rflt on" data-v="">все сценарии</button>
{scn_btns}
<button class="act" data-pane="restore" data-act="sort">слабее восстановило сверху</button>
<input class="q" data-pane="restore" type="search" placeholder="поиск по имени">
</div>
<div class="controls hidden" id="ctl-nr">
<button class="nflt on" data-v="">все группы</button>
{nr_btns}
<button class="act" data-pane="nr" data-act="enh">только enhanced</button>
<button class="act" data-pane="nr" data-act="all">сбросить</button>
<button class="act" data-pane="nr" data-act="sort">слабее улучшило сверху</button>
<input class="q" data-pane="nr" type="search" placeholder="поиск по имени">
</div>
</header>
<div class="grid pane on" id="restore">
{restore_cards}
</div>
<div class="grid pane" id="nr">
{nr_cards}
</div>
<script>
const panes={{restore:document.getElementById('restore'),nr:document.getElementById('nr')}};
let scn='', nrGroup='', nrState='', qr='', qn='';
function applyRestore(){{
  for(const c of panes.restore.children){{
    const okS=!scn||c.dataset.scn===scn;
    const okQ=!qr||c.dataset.name.includes(qr);
    c.classList.toggle('hidden',!(okS&&okQ));
  }}
}}
function applyNr(){{
  for(const c of panes.nr.children){{
    const okG=!nrGroup||c.dataset.group===nrGroup;
    const okS=!nrState||c.dataset.state==='ok';
    const okQ=!qn||c.dataset.name.includes(qn);
    c.classList.toggle('hidden',!(okG&&okS&&okQ));
  }}
}}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));t.classList.add('on');
  const p=t.dataset.pane;
  panes.restore.classList.toggle('on',p==='restore');
  panes.nr.classList.toggle('on',p==='nr');
  document.getElementById('ctl-restore').classList.toggle('hidden',p!=='restore');
  document.getElementById('ctl-nr').classList.toggle('hidden',p!=='nr');
}});
document.querySelectorAll('.rflt').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.rflt').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');scn=b.dataset.v;applyRestore();
}});
document.querySelectorAll('.nflt').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.nflt').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');nrGroup=b.dataset.v;applyNr();
}});
document.querySelectorAll('.act').forEach(b=>b.onclick=()=>{{
  const pane=panes[b.dataset.pane], a=b.dataset.act;
  if(a==='sort'){{
    const arr=[...pane.children];
    arr.sort((x,y)=>parseFloat(x.dataset.sort)-parseFloat(y.dataset.sort));
    arr.forEach(c=>pane.appendChild(c));
  }} else if(a==='enh'){{nrState='enh';applyNr();}}
  else if(a==='all'){{nrState='';nrGroup='';qn='';applyNr();}}
}});
document.querySelectorAll('.q').forEach(inp=>inp.oninput=e=>{{
  if(e.target.dataset.pane==='restore'){{qr=e.target.value.toLowerCase();applyRestore();}}
  else{{qn=e.target.value.toLowerCase();applyNr();}}
}});
</script>
</body></html>
"""


def _iter_group_images(group_dir: Path) -> list[Path]:
    return sorted(p for p in group_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description="Оценка POST /enhance на data/eval_set")
    ap.add_argument("--url", default="http://localhost:8000/enhance")
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval_set"))
    ap.add_argument(
        "--groups",
        default="hr_donor,lr,corner_case",
        help="через запятую; режим берётся из DEFAULT_GROUP_MODES",
    )
    ap.add_argument(
        "--scenarios",
        default="downscale,darken,overexpose,noise",
        help="сценарии деградации для restore-режима (hr_donor)",
    )
    ap.add_argument("--downscale-scales", default="2,3,4", help="из этих факторов downscale берёт случайный")
    ap.add_argument("--darken-range", default="0.25,0.5", help="диапазон множителя яркости для darken")
    ap.add_argument("--brighten-range", default="1.4,2.0", help="диапазон множителя для overexpose")
    ap.add_argument("--noise-range", default="10,30", help="диапазон sigma шума (0-255) для noise")
    ap.add_argument("--seed", type=int, default=42, help="seed рандома деградаций (воспроизводимость)")
    ap.add_argument("--jpeg-q", type=int, default=95, help="качество JPEG для испорченного входа")
    ap.add_argument("--limit", type=int, default=0, help="макс. фото на группу (0 = все)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=Path("data/eval_set/eval_results.csv"))
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="каталог для HTML-отчёта с превью (если не задан, отчёт не строится)",
    )
    ap.add_argument("--preview-max", type=int, default=720, help="длинная сторона превью в HTML")
    ap.add_argument("--no-lpips", action="store_true", help="не считать LPIPS даже если есть pyiqa")
    args = ap.parse_args()

    if not args.eval_set.exists():
        print(f"нет каталога {args.eval_set}", file=sys.stderr)
        return 2

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    bad = [s for s in scenarios if s not in SCENARIOS]
    if bad:
        print(f"неизвестные сценарии: {bad}. Доступно: {list(SCENARIOS)}", file=sys.stderr)
        return 2

    # параметры деградаций: диапазоны + общий детерминированный rng
    args.downscale_scales = [int(x) for x in args.downscale_scales.split(",")]
    args.darken_range = tuple(float(x) for x in args.darken_range.split(","))
    args.brighten_range = tuple(float(x) for x in args.brighten_range.split(","))
    args.noise_range = tuple(float(x) for x in args.noise_range.split(","))
    args.rng = np.random.default_rng(args.seed)

    # быстрая проверка, что ручка жива
    health = args.url.rsplit("/", 1)[0] + "/healthz"
    try:
        with urlrequest.urlopen(health, timeout=5) as r:  # noqa: S310
            r.read()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ручка недоступна ({health}): {type(exc).__name__}. Подними сервис: make up",
            file=sys.stderr,
        )
        return 2

    lpips = LpipsScorer(enabled=not args.no_lpips)
    print(f"LPIPS: {'включён' if lpips.available else 'выключен (pyiqa нет или --no-lpips)'}")
    sink = ReportSink(args.html, preview_max=args.preview_max) if args.html else None

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    all_rows: list[RowResult] = []

    for group in groups:
        group_dir = args.eval_set / group
        if not group_dir.is_dir():
            print(f"пропуск {group}: нет каталога {group_dir}", file=sys.stderr)
            continue
        mode = DEFAULT_GROUP_MODES.get(group, "nr")
        images = _iter_group_images(group_dir)
        if args.limit:
            images = images[: args.limit]
        unit = f"{len(images)} фото x {len(scenarios)} сценариев" if mode == "restore" else f"{len(images)} фото"
        print(f"\n>>> {group} ({mode}): {unit}")

        for i, path in enumerate(images, start=1):
            if mode == "restore":
                hr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                for sc in scenarios:
                    row = RowResult(group=group, mode=mode, name=path.name, status=0, scenario=sc)
                    if hr is None:
                        row.error = "read-failed"
                    else:
                        try:
                            eval_restore(hr, sc, args.url, path.name, args, lpips, sink, row)
                        except Exception as exc:  # noqa: BLE001
                            row.error = f"{type(exc).__name__}: {exc}"
                    all_rows.append(row)
            else:
                row = RowResult(group=group, mode=mode, name=path.name, status=0)
                try:
                    eval_nr(path.read_bytes(), args.url, path.name, args.timeout, None, sink, row)
                except Exception as exc:  # noqa: BLE001
                    row.error = f"{type(exc).__name__}: {exc}"
                all_rows.append(row)
            if i % max(1, len(images) // 10) == 0 or i == len(images):
                print(f"  {i}/{len(images)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, args.out)
    print(f"\nдетальный CSV: {args.out}")
    if sink is not None:
        write_html_report(all_rows, args.html)
        print(f"HTML-отчёт: {args.html / 'index.html'}")
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
