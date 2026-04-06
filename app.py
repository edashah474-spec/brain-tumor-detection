
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Loading AI model... please wait 30 seconds"):
            file_id = "1_xnYZcf3qNoLUG_AUve0SaqzexdZxNwZ"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload a brain MRI scan — AI will detect the tumor type instantly")
st.write("**Accuracy: 97.68% | EfficientNetV2 | Built by Noor Khan**")
st.divider()

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", width=300)

    img_array = np.array(image.resize((224, 224))).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing MRI scan..."):
        preds = model.predict(img_array, verbose=0)[0]
        pred_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.divider()

    if pred_class == "No Tumor":
        st.success(f"✅ Result: {pred_class} ({confidence:.1f}% confidence)")
    else:
        st.error(f"⚠️ Tumor Detected: {pred_class} ({confidence:.1f}% confidence)")

    st.subheader("Confidence breakdown:")
    for cls, prob in zip(CLASSES, preds):
        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    st.divider()
    st.warning("⚕️ For educational purposes only. Always consult a qualified radiologist for medical diagnosis.")

st.divider()
st.markdown("**Noor Khan | CS AI-ML | KLE Technological University, Belgaum**")
st.markdown("**GitHub:** github.com/eadashah474-spec/brain-tumor-detection")
