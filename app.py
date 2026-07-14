import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load SavedModel
model = tf.saved_model.load("saved_model")
infer = model.signatures["serving_default"]

# Nama kelas CIFAR-10
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

st.title("CNN CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader(
    "Upload sebuah gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    image = image.resize((32, 32))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    input_tensor = tf.convert_to_tensor(image)

    output = infer(input_tensor)
    prediction = list(output.values())[0].numpy()

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    st.success(f"Prediksi: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")