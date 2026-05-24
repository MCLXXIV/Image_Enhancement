import json

from fastapi.testclient import TestClient

from enhancer.api import app


def _client() -> TestClient:
    return TestClient(app)


def test_healthz_returns_ok() -> None:
    with _client() as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_metrics_endpoint_serves_prometheus_format() -> None:
    with _client() as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_enhance_neutral_image_skipped(jpeg_bytes: bytes) -> None:
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("neutral.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Enhance-Skipped"] == "true"
    assert response.headers["X-Enhance-Applied"] == "none"
    before = json.loads(response.headers["X-Enhance-Quality-Before"])
    after = json.loads(response.headers["X-Enhance-Quality-After"])
    assert before == after


def test_enhance_dark_image_applies_gamma(dark_jpeg_bytes: bytes) -> None:
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("dark.jpg", dark_jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.headers["X-Enhance-Skipped"] == "false"
    assert "gamma" in response.headers["X-Enhance-Applied"]
    before = json.loads(response.headers["X-Enhance-Quality-Before"])
    after = json.loads(response.headers["X-Enhance-Quality-After"])
    assert after["brightness_mean"] > before["brightness_mean"]


def test_enhance_rejects_non_image() -> None:
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("note.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 415


def test_enhance_rejects_undecodable_bytes() -> None:
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("garbage.jpg", b"not-a-jpeg", "image/jpeg")},
        )
    assert response.status_code == 400


def test_enhance_white_box_params(jpeg_bytes: bytes) -> None:
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("x.jpg", jpeg_bytes, "image/jpeg")},
            data={"params": json.dumps({"clahe_clip": 3.0, "force": True})},
        )
    assert response.status_code == 200
    assert "clahe" in response.headers["X-Enhance-Applied"]
