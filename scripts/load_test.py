"""Нагрузочный тест POST /enhance: гонит N запросов с заданным concurrency и печатает статистику."""

from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def build_multipart(image_path: Path, params: dict | None) -> tuple[bytes, str]:
    """Сборка multipart/form-data вручную без requests."""
    boundary = f"----loadtest-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_bytes = image_path.read_bytes()

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
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"'.encode(),
        f"Content-Type: {mime}".encode(),
        b"",
        image_bytes,
        f"--{boundary}--".encode(),
        b"",
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def do_request(url: str, body: bytes, boundary: str, timeout: float) -> tuple[int, float, str]:
    """Возвращает (status, latency_ms, applied)."""
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
            applied = resp.headers.get("X-Enhance-Applied", "?")
            resp.read()
            return resp.status, (time.perf_counter() - t0) * 1000, applied
    except HTTPError as e:
        return e.code, (time.perf_counter() - t0) * 1000, "http-error"
    except (URLError, TimeoutError) as e:
        return 0, (time.perf_counter() - t0) * 1000, f"err:{type(e).__name__}"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((q / 100) * (len(s) - 1)))))
    return s[k]


def main() -> int:
    ap = argparse.ArgumentParser(description="POST /enhance load test")
    ap.add_argument("--url", default="http://localhost:8000/enhance")
    ap.add_argument("--image", default="test.jpeg", type=Path)
    ap.add_argument("--total", type=int, default=50, help="всего запросов")
    ap.add_argument("--concurrency", type=int, default=4, help="параллельных потоков")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--params",
        default=None,
        help='JSON с EnhanceParams, например \'{"safmn_only": true}\'',
    )
    args = ap.parse_args()

    if not args.image.exists():
        print(f"картинка не найдена: {args.image}", file=sys.stderr)
        return 2

    params = json.loads(args.params) if args.params else None
    body, boundary = build_multipart(args.image, params)

    print(
        f"target: {args.url} | image: {args.image} ({len(body) / 1024:.1f} KB body)\n"
        f"total: {args.total} | concurrency: {args.concurrency} | "
        f"params: {params or '{}'}"
    )

    latencies: list[float] = []
    statuses: Counter[int] = Counter()
    applied_counter: Counter[str] = Counter()

    wall_t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(do_request, args.url, body, boundary, args.timeout)
            for _ in range(args.total)
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            status, ms, applied = fut.result()
            statuses[status] += 1
            applied_counter[applied] += 1
            if status == 200:
                latencies.append(ms)
            if i % max(1, args.total // 10) == 0:
                print(f"  {i}/{args.total} done")
    wall_s = time.perf_counter() - wall_t0

    ok = statuses[200]
    rps = args.total / wall_s if wall_s > 0 else 0.0
    print("\n=== результат ===")
    print(f"wall:       {wall_s:.2f}s")
    print(f"throughput: {rps:.2f} req/s")
    print(f"success:    {ok}/{args.total} ({ok / args.total * 100:.1f}%)")
    print(f"statuses:   {dict(statuses)}")
    print(f"applied:    {dict(applied_counter)}")

    if latencies:
        print("\n=== latency (ms, только 200) ===")
        print(f"min:    {min(latencies):.0f}")
        print(f"p50:    {statistics.median(latencies):.0f}")
        print(f"p95:    {percentile(latencies, 95):.0f}")
        print(f"p99:    {percentile(latencies, 99):.0f}")
        print(f"max:    {max(latencies):.0f}")
        print(f"mean:   {statistics.mean(latencies):.0f}")
        slo_violations = sum(1 for x in latencies if x > 1000)
        print(
            f"SLO p99 ≤ 1000ms: "
            f"{'PASS' if percentile(latencies, 99) <= 1000 else 'FAIL'} "
            f"(нарушений: {slo_violations}/{len(latencies)})"
        )

    return 0 if ok == args.total else 1


if __name__ == "__main__":
    sys.exit(main())
