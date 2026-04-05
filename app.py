
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    return model

model = load_model()
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# UI
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload a brain MRI scan and the AI will detect the tumor type")
st.write("**Model accuracy: 97.68% | Built with EfficientNetV2**")

st.divider()

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan", width=300)

    # Preprocess
    img_array = np.array(image.resize((224, 224))).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing MRI scan..."):
        preds = model.predict(img_array, verbose=0)[0]
        pred_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.divider()

    # Result
    if pred_class == "No Tumor":
        st.success(f"✅ Result: {pred_class} ({confidence:.1f}% confidence)")
    else:
        st.error(f"⚠️ Tumor Detected: {pred_class} ({confidence:.1f}% confidence)")

    # Confidence bars
    st.subheader("Confidence for each class:")
    for cls, prob in zip(CLASSES, preds):
        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    st.divider()
    st.warning("⚕️ This is an AI tool for educational purposes only. Always consult a qualified radiologist for medical diagnosis.")

st.divider()
st.markdown("**Built by Noor Khan | KLE Technological University**")
st.markdown("GitHub: github.com/eadashah474-spec/brain-tumor-detection")
