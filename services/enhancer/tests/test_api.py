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


def test_enhance_returns_jpeg_with_headers(jpeg_bytes: bytes) -> None:
    # Без весов модели не зарегистрированы, фото пропускается, но контракт ответа сохраняется.
    with _client() as client:
        response = client.post(
            "/enhance",
            files={"image": ("neutral.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Enhance-Skipped"] == "true"
    assert response.headers["X-Enhance-Applied"] == "none"
    assert "X-Enhance-Iqa-Before" in response.headers
    assert "X-Enhance-Iqa-After" in response.headers
    before = json.loads(response.headers["X-Enhance-Quality-Before"])
    after = json.loads(response.headers["X-Enhance-Quality-After"])
    assert before == after


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
