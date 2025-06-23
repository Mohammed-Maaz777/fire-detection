import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

from yolov5.models.experimental import attempt_load

# Load model using absolute path
model_path = os.path.join(os.path.dirname(__file__), 'yolov5', 'weights', 'best.pt')
model = attempt_load(model_path, map_location='cpu')
model.eval()

# Streamlit UI
st.title("🔥 Fire Detection App")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    with st.spinner("Detecting fire..."):
        results = model(img_tensor)[0]
        
        # Dummy logic – replace with custom thresholding or class detection
        st.warning("⚠️ Fire detection results generated. (Add logic to check if fire is actually detected)")
