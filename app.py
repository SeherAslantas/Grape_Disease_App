
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Sayfa ayarları
st.set_page_config(
    page_title="Grape Disease Detection AI",
    page_icon="🍇",
    layout="wide"
)

# Başlık
st.title("🍇 Grape Disease Detection AI")
st.write("Yapay zekâ ile üzüm yaprağı hastalıklarını tespit edin.")

# Sınıflar
labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

# Modelleri yükle
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("Grape_modell.keras")
    vgg = tf.keras.models.load_model("Grape_ModelTL.keras")
    return cnn, vgg

try:
    cnn_model, vgg_model = load_models()
except Exception as e:
    st.error(f"Modeller yüklenirken hata oluştu: {e}")
    st.stop()

# Model seçimi
model_choice = st.selectbox(
    "Tahmin için model seçin:",
    ["CNN", "VGG16 (Transfer Learning)"]
)

# Görsel yükleme
uploaded_file = st.file_uploader(
    "Üzüm yaprağı fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Yüklenen yaprak görüntüsü",
                 use_container_width=True)

    # Görüntü ön işleme
    img = image.resize((128, 128))
    img_array = np.array(img).astype(np.float32) / 255.0
    if model_choice == "CNN":
    img_array = img_array[..., ::-1]
    img_array = np.expand_dims(img_array, axis=0)

    # Tahmin
    if st.button("Hastalığı Tespit Et"):

        selected_model = (
            cnn_model if model_choice == "CNN"
            else vgg_model
        )

        prediction = selected_model.predict(img_array, verbose=0)[0]

        predicted_index = np.argmax(prediction)
        predicted_class = labels[predicted_index]
        confidence = prediction[predicted_index] * 100

        with col2:
            st.subheader("Tahmin Sonucu")
            st.success(f"Tespit edilen sınıf: {predicted_class}")
            st.metric("Güven oranı", f"%{confidence:.2f}")

            st.subheader("Sınıf Olasılıkları")

            for label, probability in zip(labels, prediction):
                st.write(f"{label}: %{probability * 100:.2f}")
                st.progress(float(probability))

st.divider()
st.caption("Grape Disease Detection AI | CNN & Transfer Learning")
