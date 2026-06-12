import base64
import io
import json
import os

import requests
import streamlit as st
from PIL import Image

ENHANCER_URL = os.environ.get("ENHANCER_URL", "http://enhancer:8000")
DISPLAY_MAX = 720

# Модели пайплайна: ключ force-флага, человекочитаемое имя, домен дефекта.
SANDBOX_MODELS = [
    ("force_exposure", "exposure (IAT)", "пересвет, выгоревший дневной кадр"),
    ("force_lowlight", "low_light (Retinexformer)", "тёмное, низкоконтрастное, тон и цвет"),
    ("force_restore", "restore (SCUNet)", "шум, JPEG, лёгкий блюр"),
    ("force_safmn", "safmn (Real-SAFMN++)", "апскейл и restoration мелких фото"),
]


def clickable_image(label: str, data: bytes, mime: str, display_width: int) -> None:
    uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    st.caption(label)
    st.markdown(
        f'<a href="{uri}" target="_blank">'
        f'<img src="{uri}" width="{display_width}" style="cursor: zoom-in"/></a>',
        unsafe_allow_html=True,
    )


def call_enhance(name: str, raw: bytes, mime: str, params: dict | None) -> requests.Response:
    data = {"params": json.dumps(params)} if params else None
    return requests.post(
        f"{ENHANCER_URL}/enhance",
        files={"image": (name, raw, mime)},
        data=data,
        timeout=120,
    )


def render_auto() -> None:
    st.caption("Система сама определит, что с фото сделать, и вернёт улучшенную версию.")
    uploaded = st.file_uploader(
        "Фото объявления (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="auto_up"
    )
    if uploaded is None:
        st.info("Перетащите файл, чтобы увидеть сравнение до/после.")
        return

    raw = uploaded.read()
    mime = uploaded.type or "image/jpeg"
    with st.spinner("Обрабатываем"):
        response = call_enhance(uploaded.name, raw, mime, None)
    if response.status_code != 200:
        st.error(f"Сервис вернул {response.status_code}: {response.text}")
        return

    before_img = Image.open(io.BytesIO(raw)).convert("RGB")
    after_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    applied = response.headers.get("X-Enhance-Applied", "none")
    skipped = response.headers.get("X-Enhance-Skipped") == "true"
    fallback = response.headers.get("X-Enhance-Fallback") == "true"
    latency = float(response.headers.get("X-Enhance-Latency-Ms", "0"))

    if skipped:
        st.success("Фото уже хорошего качества, улучшение не требуется.")
        st.image(before_img, use_container_width=True)
    elif fallback:
        st.warning("Улучшение не дало выигрыша по качеству, оставили оригинал.")
        forced = call_enhance(uploaded.name, raw, mime, {"force": True})
        disp_scale = min(1.0, DISPLAY_MAX / before_img.width)
        before_w = max(1, round(before_img.width * disp_scale))
        clickable_image("Оригинал (его и вернули)", raw, mime, before_w)
        if forced.status_code == 200:
            forced_after = Image.open(io.BytesIO(forced.content)).convert("RGB")
            after_w = max(1, round(forced_after.width * disp_scale))
            clickable_image(
                "Результат улучшения (не прошёл гейт качества)", forced.content, "image/jpeg",
                after_w,
            )
    else:
        disp_scale = min(1.0, DISPLAY_MAX / after_img.width)
        before_w = max(1, round(before_img.width * disp_scale))
        after_w = max(1, round(after_img.width * disp_scale))
        clickable_image("До", raw, mime, before_w)
        clickable_image("После", response.content, "image/jpeg", after_w)
        st.caption(
            f"Применено: {applied} · {latency:.0f} мс. Клик по фото открывает реальный размер."
        )
        st.download_button(
            "Скачать улучшенное фото",
            data=response.content,
            file_name=f"enhanced_{uploaded.name}",
            mime="image/jpeg",
        )

    _debug_expander(response)


def render_sandbox() -> None:
    st.caption(
        "Ручной выбор моделей в обход роутера и гейта качества. Запускаются строго "
        "отмеченные модели (роутер игнорируется) в порядке: exposure, low_light, restore/safmn."
    )
    uploaded = st.file_uploader(
        "Фото (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="sandbox_up"
    )

    chosen: list[str] = []
    cols = st.columns(2)
    for i, (flag, title, domain) in enumerate(SANDBOX_MODELS):
        with cols[i % 2]:
            if st.checkbox(title, key=f"cb_{flag}", help=domain):
                chosen.append(flag)

    if uploaded is None:
        st.info("Загрузите фото и отметьте модели, чтобы посмотреть их работу.")
        return
    if not chosen:
        st.warning("Отметьте хотя бы одну модель.")
        return

    raw = uploaded.read()
    mime = uploaded.type or "image/jpeg"
    # only=true игнорирует роутер (строго выбранные стадии), force=true снимает skip и IQA-гейт.
    params: dict = {"force": True, "only": True}
    for flag in chosen:
        params[flag] = True

    with st.spinner("Прогоняем выбранные модели"):
        response = call_enhance(uploaded.name, raw, mime, params)
    if response.status_code != 200:
        st.error(f"Сервис вернул {response.status_code}: {response.text}")
        return

    before_img = Image.open(io.BytesIO(raw)).convert("RGB")
    after_img = Image.open(io.BytesIO(response.content)).convert("RGB")
    applied = response.headers.get("X-Enhance-Applied", "none")
    latency = float(response.headers.get("X-Enhance-Latency-Ms", "0"))

    disp_scale = min(1.0, DISPLAY_MAX / after_img.width)
    before_w = max(1, round(before_img.width * disp_scale))
    after_w = max(1, round(after_img.width * disp_scale))
    clickable_image("До", raw, mime, before_w)
    clickable_image("После", response.content, "image/jpeg", after_w)
    st.caption(f"Применено: {applied} · {latency:.0f} мс.")
    st.download_button(
        "Скачать результат",
        data=response.content,
        file_name=f"sandbox_{uploaded.name}",
        mime="image/jpeg",
    )
    _debug_expander(response)


def _debug_expander(response: requests.Response) -> None:
    with st.expander("Детали (для отладки)"):
        st.json(
            {
                "applied": response.headers.get("X-Enhance-Applied", "none"),
                "skipped": response.headers.get("X-Enhance-Skipped") == "true",
                "fallback": response.headers.get("X-Enhance-Fallback") == "true",
                "scale_factor": float(response.headers.get("X-Enhance-Scale-Factor", "1.0")),
                "latency_ms": float(response.headers.get("X-Enhance-Latency-Ms", "0")),
                "iqa_before": json.loads(response.headers.get("X-Enhance-Iqa-Before", "{}")),
                "iqa_after": json.loads(response.headers.get("X-Enhance-Iqa-After", "{}")),
                "model_versions": json.loads(
                    response.headers.get("X-Enhance-Model-Versions", "{}")
                ),
            }
        )


st.set_page_config(page_title="Автоулучшение фото", layout="centered")
st.title("Автоулучшение фото")

auto_tab, sandbox_tab = st.tabs(["Авто", "Песочница моделей"])
with auto_tab:
    render_auto()
with sandbox_tab:
    render_sandbox()
