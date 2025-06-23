import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import os
import subprocess

st.title("🔥 Fire Detection App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(tmp_path)

    if st.button("Detect Fire"):
        with st.spinner("Running fire detection..."):
            result_path = "output.mp4"
            os.system(f"python detect_fire.py --source {tmp_path} --output {result_path}")
            st.success("Detection complete!")
            st.video(result_path)
