import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import sys

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection from MRI")
st.write("**Accuracy: 97.68% | EfficientNetV2 | Built by Noor Khan**")
st.divider()

@st.cache_resource
def load_model():
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "tensorflow-cpu==2.15.0", "-q"
    ])
    import tensorflow as tf
    model_path = "best_model.keras"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            file_id = "1_xnYZcf3qNoLUG_AUve0SaqzexdZxNwZ"
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                model_path, quiet=False
            )
    return tf.keras.models.load_model(model_path)

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", width=300)

    with st.spinner("Loading AI model... first load takes 2-3 minutes"):
        model = load_model()

    img_array = np.array(
        image.resize((224, 224))
    ).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing MRI scan..."):
        preds = model.predict(img_array, verbose=0)[0]
        pred_class = CLASSES[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.divider()
    if pred_class == "No Tumor":
        st.success(
            f"✅ Result: {pred_class} ({confidence:.1f}% confidence)"
        )
    else:
        st.error(
            f"⚠️ Tumor Detected: {pred_class} ({confidence:.1f}% confidence)"
        )

    st.subheader("Confidence breakdown:")
    for cls, prob in zip(CLASSES, preds):
        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    st.divider()
    st.warning(
        "⚕️ For educational purposes only. "
        "Always consult a radiologist."
    )

st.divider()
st.markdown(
    "**Safa Sayyed| CS AI-ML | KLE Technological University**"
)
st.markdown(
    "**GitHub:** github.com/eadashah474-spec/brain-tumor-detection"
)
