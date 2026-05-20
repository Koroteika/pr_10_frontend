import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# ── Настройки ─────────────────────────────────────────────────────
API_URL     = "https://pr-10-backend.onrender.com/predict"
CLASS_NAMES = ["Asian Green Bee-Eater", "Brown-Headed Barbet", "Common Kingfisher"]
IMG_SIZE    = 128

# ── Интерфейс ─────────────────────────────────────────────────────
st.set_page_config(page_title="Bird Classifier", page_icon="🐦", layout="centered")

st.title("🐦 Классификация птиц")
st.markdown(
    "Загрузите фотографию птицы или нарисуйте её на холсте — "
    "модель определит вид из трёх классов: "
    "**Asian Green Bee-Eater**, **Brown-Headed Barbet**, **Common Kingfisher**."
)
st.divider()

# ── Выбор режима ввода ────────────────────────────────────────────
mode = st.radio(
    "Выберите способ ввода изображения:",
    ("📁 Загрузить изображение", "✏️ Нарисовать на холсте"),
    horizontal=True
)

image_bytes = None   # итоговые байты для отправки на API


def preprocess_and_encode(pil_image: Image.Image) -> bytes:
    """Изменяет размер до 128×128, конвертирует в RGB и возвращает байты JPEG."""
    pil_image = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


# ── Режим 1: Загрузка файла ───────────────────────────────────────
if mode == "📁 Загрузить изображение":
    uploaded = st.file_uploader(
        "Выберите изображение (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Загруженное изображение", use_container_width=True)
        image_bytes = preprocess_and_encode(pil_img)


# ── Режим 2: Рисование на холсте ─────────────────────────────────
else:
    st.markdown("Нарисуйте птицу на холсте ниже:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=st.slider("Толщина кисти", 1, 30, 8),
        stroke_color=st.color_picker("Цвет кисти", "#000000"),
        background_color="#FFFFFF",
        height=350,
        width=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        # image_data — numpy array (H, W, 4) RGBA
        canvas_array = canvas_result.image_data.astype(np.uint8)
        pil_img = Image.fromarray(canvas_array, mode="RGBA").convert("RGB")
        image_bytes = preprocess_and_encode(pil_img)


# ── Кнопка классификации ──────────────────────────────────────────
st.divider()
classify_btn = st.button("🔍 Классифицировать", type="primary", use_container_width=True)

if classify_btn:
    if image_bytes is None:
        st.warning("⚠️ Сначала загрузите изображение или нарисуйте птицу на холсте.")
    else:
        with st.spinner("Отправляем изображение на сервер..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                    timeout=60   # бесплатный Render может «засыпать» — ждём дольше
                )
                response.raise_for_status()
                data = response.json()

                predicted_class = data["predicted_class"]
                confidence      = data["confidence"]
                probabilities   = data["probabilities"]

                # ── Результат ─────────────────────────────────────
                st.success(f"✅ Результат классификации")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Предсказанный класс", predicted_class)
                with col2:
                    st.metric("Уверенность модели", f"{confidence * 100:.2f}%")

                st.divider()

                # ── Визуализация вероятностей ─────────────────────
                st.subheader("📊 Распределение вероятностей по классам")

                classes = list(probabilities.keys())
                probs   = list(probabilities.values())
                colors  = [
                    "#2ecc71" if c == predicted_class else "#3498db"
                    for c in classes
                ]

                fig, ax = plt.subplots(figsize=(8, 3.5))
                bars = ax.barh(classes, probs, color=colors, edgecolor="black",
                               linewidth=0.6)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Вероятность")
                ax.set_title("Вероятности классификации")
                ax.grid(axis="x", alpha=0.3)

                for bar, prob in zip(bars, probs):
                    ax.text(
                        bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{prob * 100:.2f}%",
                        va="center", fontsize=11, fontweight="bold"
                    )

                plt.tight_layout()
                st.pyplot(fig)

                # ── Детальная таблица ─────────────────────────────
                st.subheader("📋 Подробные результаты")
                for cls, prob in sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True
                ):
                    icon = "🥇" if cls == predicted_class else "  "
                    st.write(f"{icon} **{cls}**: {prob * 100:.4f}%")

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Сервер не ответил вовремя. "
                    "Бесплатный Render засыпает при простое — подождите 30 секунд и попробуйте снова."
                )
            except requests.exceptions.ConnectionError:
                st.error("🔌 Не удалось подключиться к серверу. Проверьте, что API запущен.")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# ── Подвал ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Модель: CNN + BatchNorm + Dropout | "
    "Датасет: Indian Birds (3 класса) | "
    f"[API]({API_URL.replace('/predict', '/docs')})"
)