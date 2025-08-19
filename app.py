import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model.h5")

def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("🌿 Plant Disease Detector")
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]
    
    result = "Diseased 🌱" if pred > 0.5 else "Healthy 🌿"
    st.success(f"Prediction: {result}")
