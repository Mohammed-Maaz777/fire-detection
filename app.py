import os
import urllib.request
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

# Define model URL and local path
model_url = "https://drive.google.com/uc?export=download&id=14xOO-qgHb7k-ugZIK0mXaDgO_N21W8Td"
model_path = "yolov5/weights/best.pt"

# Create weights directory if it doesn't exist
os.makedirs("yolov5/weights", exist_ok=True)

# Download the model if not already present
if not os.path.exists(model_path):
    with st.spinner("🔄 Downloading model..."):
        urllib.request.urlretrieve(model_url, model_path)
        st.success("✅ Model downloaded!")

# Add yolov5 to system path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# Import YOLOv5 model loader
from yolov5.models.experimental import attempt_load

# Load model
model = attempt_load(model_path)
model.eval()

# Streamlit UI
st.title("🔥 Fire Detection App")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to tensor
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    # Inference
    with st.spinner("🚒 Detecting fire..."):
        results = model(img_tensor)[0]
        st.success("✅ Inference complete. Add logic to parse detection results.")
