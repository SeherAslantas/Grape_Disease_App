
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Grape Disease Detection - CNN",
    page_icon="🍇",
    layout="wide"
)

st.title("🍇 Grape Disease Detection - CNN")
st.write("CNN modeli ile üzüm yaprağı hastalıklarını tespit edin.")

labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("Grape_modell.keras")

try:
    model = load_cnn()
    st.success("CNN modeli başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Üzüm yaprağı fotoğrafı yükleyin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Yüklenen yaprak görüntüsü", use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array[..., ::-1]
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Hastalığı Tespit Et"):
        prediction = model.predict(img_array, verbose=0)[0]

        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index] * 100

        with col2:
            st.subheader("Tahmin Sonucu")
            st.success(f"Tespit edilen sınıf: {labels[predicted_index]}")
            st.metric("Güven oranı", f"%{confidence:.2f}")

            st.subheader("Sınıf Olasılıkları")

            for label, probability in zip(labels, prediction):
                st.write(f"{label}: %{probability * 100:.2f}")
                st.progress(float(probability))
