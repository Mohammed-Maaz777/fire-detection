import streamlit as st
import os
import tempfile

st.title("🔥 Fire Detection App")
st.markdown("Upload a video and detect fire using a YOLOv5 model.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    st.video(input_path)

    if st.button("Detect Fire"):
        with st.spinner("Running fire detection..."):
            output_path = os.path.join(tempfile.gettempdir(), "output_fire.mp4")
            os.system(f"python detect_fire.py --source \"{input_path}\" --output \"{output_path}\"")

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                st.success("Fire detection complete!")
                st.video(output_path)
            else:
                st.error("Detection failed or no fire detected.")
