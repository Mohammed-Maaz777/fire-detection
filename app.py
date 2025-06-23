import streamlit as st
import torch
from PIL import Image
import os

model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/fire-detection10/weights/best.pt', source='local')

st.title("🔥 Fire Detection App")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting fire..."):
        results = model(img, size=640)
        results.save()
        st.image("runs/detect/exp/image0.jpg", caption="Detection Result")

        if "fire" in results.names.values():
            st.error("⚠️ Fire detected!")
        else:
            st.success("✅ No fire detected.")
