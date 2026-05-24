import io
import json
import os

import requests
import streamlit as st
from PIL import Image

ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8000")

st.set_page_config(page_title="Image Enhancer Demo", layout="wide")
st.title("Автоулучшение фото - demo")

with st.sidebar:
    st.header("White-box параметры")
    use_white_box = st.checkbox("Передавать параметры вручную", value=False)
    force = st.checkbox("force (применить даже если 'хорошее')", value=False)
    gamma = st.slider("gamma (<1 светлее, >1 темнее)", 0.5, 2.0, 1.0, 0.05)
    clahe_clip = st.slider("CLAHE clip", 1.0, 6.0, 2.0, 0.5)
    sharp_amount = st.slider("Unsharp amount", 0.0, 2.0, 0.0, 0.1)
    denoise_strength = st.slider("Denoise strength", 0.0, 1.0, 0.0, 0.05)

uploaded = st.file_uploader("Загрузите фото (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Выберите файл, чтобы увидеть before/after.")
    st.stop()

raw = uploaded.read()

params: dict[str, float | bool] = {}
if use_white_box:
    if force:
        params["force"] = True
    if gamma != 1.0:
        params["gamma"] = gamma
    if sharp_amount > 0:
        params["sharp_amount"] = sharp_amount
    if denoise_strength > 0:
        params["denoise_strength"] = denoise_strength
    params["clahe_clip"] = clahe_clip

data = {"params": json.dumps(params)} if params else None

with st.spinner(f"POST {ENHANCER_URL}/enhance..."):
    response = requests.post(
        f"{ENHANCER_URL}/enhance",
        files={"image": (uploaded.name, raw, uploaded.type or "image/jpeg")},
        data=data,
        timeout=30,
    )

if response.status_code != 200:
    st.error(f"Сервис вернул {response.status_code}: {response.text}")
    st.stop()

col_before, col_after = st.columns(2)
with col_before:
    st.subheader("Before")
    st.image(Image.open(uploaded), use_container_width=True)
with col_after:
    st.subheader("After")
    st.image(Image.open(io.BytesIO(response.content)), use_container_width=True)

st.subheader("Pipeline")
st.write(
    {
        "applied": response.headers.get("X-Enhance-Applied"),
        "skipped": response.headers.get("X-Enhance-Skipped"),
        "fallback": response.headers.get("X-Enhance-Fallback"),
        "latency_ms": response.headers.get("X-Enhance-Latency-Ms"),
        "model_versions": json.loads(response.headers.get("X-Enhance-Model-Versions", "{}")),
    }
)

col_mb, col_ma = st.columns(2)
with col_mb:
    st.subheader("Quality before")
    st.json(json.loads(response.headers.get("X-Enhance-Quality-Before", "{}")))
with col_ma:
    st.subheader("Quality after")
    st.json(json.loads(response.headers.get("X-Enhance-Quality-After", "{}")))
