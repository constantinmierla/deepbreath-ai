import streamlit as st
import numpy as np
import cv2
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Clasificare Cancer Pulmonar", layout="wide")

IMG_SIZE = (224, 224)
CLASS_NAMES = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']
MODEL_PATH = 'improved_chest_cancer_model.keras'
CSV_PATH = 'predictions.csv'

@st.cache_resource
def load_chest_cancer_model():
    return load_model(MODEL_PATH)

model = load_chest_cancer_model()

st.title("🔬 Clasificare Cancer Pulmonar")
st.markdown("Încarcă o imagine pentru a detecta automat tipul de cancer pulmonar.")

uploaded_file = st.file_uploader("📤 Încarcă o imagine (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        img = cv2.resize(image, IMG_SIZE)
        img = img / 255.0
        input_array = np.expand_dims(img, axis=0)

        predictions = model.predict(input_array)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="📷 Imagine încărcată", use_container_width=True)

        with col2:
            st.markdown(f"### ✅ Clasă prezisă: `{CLASS_NAMES[class_idx]}`")
            st.markdown(f"### 🔒 Încredere: `{confidence * 100:.2f}%`")
            st.markdown("### 📊 Distribuția probabilităților:")
            for name, prob in zip(CLASS_NAMES, predictions[0]):
                st.progress(float(prob), text=f"{name}: {prob:.2%}")

            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, predictions[0], color='teal')
            ax.set_ylabel("Probabilitate")
            ax.set_ylim([0, 1])
            ax.set_title("Distribuția probabilităților")

            ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')

            plt.tight_layout()
            st.pyplot(fig)

        prob_dict = {name: f"{prob:.4f}" for name, prob in zip(CLASS_NAMES, predictions[0])}
        save_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': uploaded_file.name,
            'predicted_class': CLASS_NAMES[class_idx],
            'confidence': f"{confidence * 100:.1f}%",
            **prob_dict
        }

        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=save_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(save_data)

        st.success("✅ Predicție salvată în `predictions.csv`")

# === Istoric predicții ===
if os.path.exists(CSV_PATH):
    st.markdown("### 📁 Istoric ultimele 10 predicții")
    df = pd.read_csv(CSV_PATH)
    st.dataframe(df.tail(10), use_container_width=True)

    with open(CSV_PATH, "rb") as f:
        st.download_button("⬇️ Descarcă toate predicțiile (CSV)", f, file_name="predictions.csv")