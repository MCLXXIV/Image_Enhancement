import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8000")
DISPLAY_MAX = 720


def clickable_image(label: str, data: bytes, mime: str, display_width: int) -> None:
    uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    st.caption(label)
    st.markdown(
        f'<a href="{uri}" target="_blank">'
        f'<img src="{uri}" width="{display_width}" style="cursor: zoom-in"/></a>',
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Автоулучшение фото", layout="centered")
st.title("Автоулучшение фото")
st.caption("Загрузите фото, система сама определит, что с ним сделать, и вернёт улучшенную версию.")

uploaded = st.file_uploader("Фото объявления (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Перетащите файл, чтобы увидеть сравнение до/после.")
    st.stop()

raw = uploaded.read()

with st.spinner("Обрабатываем…"):
    response = requests.post(
        f"{ENHANCER_URL}/enhance",
        files={"image": (uploaded.name, raw, uploaded.type or "image/jpeg")},
        timeout=120,
    )

if response.status_code != 200:
    st.error(f"Сервис вернул {response.status_code}: {response.text}")
    st.stop()

before_img = Image.open(io.BytesIO(raw)).convert("RGB")
after_img = Image.open(io.BytesIO(response.content)).convert("RGB")

applied = response.headers.get("X-Enhance-Applied", "none")
skipped = response.headers.get("X-Enhance-Skipped") == "true"
fallback = response.headers.get("X-Enhance-Fallback") == "true"
scale = float(response.headers.get("X-Enhance-Scale-Factor", "1.0"))
latency = float(response.headers.get("X-Enhance-Latency-Ms", "0"))

if skipped:
    st.success("Фото уже хорошего качества, улучшение не требуется.")
    st.image(before_img, use_container_width=True)
elif fallback:
    st.warning("Улучшение не дало выигрыша по качеству, оставили оригинал.")
    st.image(before_img, use_container_width=True)
else:
    disp_scale = min(1.0, DISPLAY_MAX / after_img.width)
    before_mime = uploaded.type or "image/jpeg"
    before_w = max(1, round(before_img.width * disp_scale))
    after_w = max(1, round(after_img.width * disp_scale))
    clickable_image("До", raw, before_mime, before_w)
    clickable_image("После", response.content, "image/jpeg", after_w)
    st.caption(f"Применено: {applied} · {latency:.0f} мс. Клик по фото открывает его в реальном размере.")
    st.download_button(
        "Скачать улучшенное фото",
        data=response.content,
        file_name=f"enhanced_{uploaded.name}",
        mime="image/jpeg",
    )

with st.expander("Детали (для отладки)"):
    st.json(
        {
            "applied": applied,
            "skipped": skipped,
            "fallback": fallback,
            "scale_factor": scale,
            "latency_ms": latency,
            "iqa_before": json.loads(response.headers.get("X-Enhance-Iqa-Before", "{}")),
            "iqa_after": json.loads(response.headers.get("X-Enhance-Iqa-After", "{}")),
            "model_versions": json.loads(response.headers.get("X-Enhance-Model-Versions", "{}")),
        }
    )
