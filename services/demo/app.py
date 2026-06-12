import io
import json
import os

import requests
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8000")

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
elif fallback:
    st.warning("Улучшение не дало выигрыша по качеству, оставили оригинал.")
else:
    # Для честного слайдера приводим «до» к размеру «после».
    before_for_slider = before_img.resize(after_img.size)
    image_comparison(img1=before_for_slider, img2=after_img, label1="До", label2="После")
    st.caption(f"Применено: {applied} · масштаб ×{scale:.1f} · {latency:.0f} мс")
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
