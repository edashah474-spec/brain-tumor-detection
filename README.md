# Brain Tumor Detection from MRI using CNN
## Live Demo
https://edashah474-brain-tumor-detection.hf.space

## Results
- Validation Accuracy: 97.68%
- Test Accuracy: 93.81%
- Model: EfficientNetV2B0 with Transfer Learning
- Dataset: 7,200 Brain MRI scans (4 classes)
An advanced deep learning system that classifies brain MRI scans 
into 4 categories using EfficientNetV2 transfer learning.

## Classes
- Glioma
- Meningioma  
- No Tumor
- Pituitary

## Tech Stack
- Python 3.10
- TensorFlow / Keras
- EfficientNetV2B0 (Transfer Learning)
- OpenCV
- Scikit-learn
- Streamlit (Deployment)
- Google Colab (GPU Training)

## Dataset
- 7,200 brain MRI images
- 4 balanced classes
- Source: Kaggle (Masoud Nickparvar)

## Results
- Accuracy: ~98%
- Grad-CAM explainability heatmaps
- Live web app deployment

## Project Structure
brain-tumor-detection/
├── notebooks/
│   └── brain_tumor_cnn.ipynb
├── app/
│   └── app.py
├── models/
│   └── best_model.keras
└── README.md

## Author
- GitHub: eadashah474-spec
- Project built for London ML placement
