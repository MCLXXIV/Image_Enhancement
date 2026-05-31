import io
import json
import os

import requests
import streamlit as st
from PIL import Image

ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8000")

st.set_page_config(page_title="Image Enhancer Demo", layout="wide")
st.title("Автоулучшение фото - demo")
st.caption(
    "Эвристика смотрит метрики, если фото плохое - SAFMN апскейлит, иначе пропускается. "
    "Форсировать можно через белый ящик слева."
)

with st.sidebar:
    st.header("Режим")
    safmn_only = st.checkbox(
        "Только SAFMN (raw модель, без эвристик и fallback)", value=False
    )
    st.divider()
    st.header("White-box параметры (CV-стадии)")
    use_white_box = st.checkbox("Передавать параметры вручную", value=False, disabled=safmn_only)
    force = st.checkbox(
        "force (применить даже если 'хорошее')", value=False, disabled=safmn_only
    )
    use_safmn = st.checkbox(
        "Форсировать SAFMN", value=False, disabled=safmn_only
    )
    gamma = st.slider("gamma (<1 светлее, >1 темнее)", 0.5, 2.0, 1.0, 0.05, disabled=safmn_only)
    clahe_clip = st.slider("CLAHE clip", 1.0, 6.0, 2.0, 0.5, disabled=safmn_only)
    sharp_amount = st.slider("Unsharp amount", 0.0, 2.0, 0.0, 0.1, disabled=safmn_only)
    denoise_strength = st.slider("Denoise strength", 0.0, 1.0, 0.0, 0.05, disabled=safmn_only)

uploaded = st.file_uploader("Загрузите фото (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Выберите файл, чтобы увидеть before/after.")
    st.stop()

raw = uploaded.read()

params: dict[str, float | bool] = {}
if safmn_only:
    params["safmn_only"] = True
elif use_white_box:
    if force:
        params["force"] = True
    if use_safmn:
        params["use_safmn"] = True
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
        timeout=120,
    )

if response.status_code != 200:
    st.error(f"Сервис вернул {response.status_code}: {response.text}")
    st.stop()

before_img = Image.open(uploaded)
after_img = Image.open(io.BytesIO(response.content))

col_before, col_after = st.columns(2)
with col_before:
    st.subheader(f"Before {before_img.size[0]}x{before_img.size[1]}")
    st.image(before_img, use_container_width=True)
with col_after:
    st.subheader(f"After {after_img.size[0]}x{after_img.size[1]}")
    st.image(after_img, use_container_width=True)

applied = response.headers.get("X-Enhance-Applied", "none")
scale_factor = float(response.headers.get("X-Enhance-Scale-Factor", "1.0"))
psnr_vs_input = float(response.headers.get("X-Enhance-Psnr-Vs-Input", "100"))
latency = float(response.headers.get("X-Enhance-Latency-Ms", "0"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applied", applied)
c2.metric("Scale factor", f"×{scale_factor:.2f}")
c3.metric("PSNR vs input", f"{psnr_vs_input:.1f} дБ")
c4.metric("Latency", f"{latency:.0f} мс")

st.subheader("Pipeline")
st.write(
    {
        "applied": applied,
        "skipped": response.headers.get("X-Enhance-Skipped"),
        "fallback": response.headers.get("X-Enhance-Fallback"),
        "latency_ms": latency,
        "scale_factor": scale_factor,
        "psnr_vs_input": psnr_vs_input,
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

st.download_button(
    "Скачать улучшенный JPEG",
    data=response.content,
    file_name=f"enhanced_{uploaded.name}",
    mime="image/jpeg",
)
