import sys
import os
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

# Add yolov5 directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# Import YOLOv5 model loader
from yolov5.models.experimental import attempt_load

# Define correct model path and load model
# Define correct model path and load model
model_path = os.path.join("yolov5", "runs", "train", "fire-detection10", "weights", "best.pt")
model = attempt_load(model_path)  # Removed map_location
model.eval()


# Streamlit UI
st.title("🔥 Fire Detection App")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Transform image to tensor
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    # Inference
    with st.spinner("Detecting fire..."):
        results = model(img_tensor)[0]
        st.success("✅ Inference complete. Add logic to parse detection results.")
