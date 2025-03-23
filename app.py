import os
import sys
import subprocess
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# ================================
# ✅ Ensure YOLOv5 is available
# ================================
YOLO_DIR = "yolov5"
if not os.path.exists(YOLO_DIR):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", f"{YOLO_DIR}/requirements.txt"])

# Add yolov5 to path
sys.path.append(YOLO_DIR)

from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device

# ================================
# ✅ Download best.pt from Google Drive if not found
# ================================
def download_model():
    model_path = "yolov5/runs/train/exp3/weights/best.pt"
    if not os.path.exists(model_path):
        st.write("⬇️ Downloading model from Google Drive...")
        try:
            import gdown
        except ImportError:
            subprocess.run(["pip", "install", "gdown"])
            import gdown
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        gdown.download(id="1xHSX7_0Y3a0LBHD56c_96W1dSHVxIm_e", output=model_path, quiet=False)

# ================================
# ✅ Load model
# ================================
@st.cache_resource
def load_model():
    download_model()
    model_path = "yolov5/runs/train/exp3/weights/best.pt"
    device = select_device('')
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ================================
# ✅ Streamlit UI
# ================================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv5.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Inference
    results = model(img_bgr, size=640)
    results.render()  # Updates results.imgs with boxes and labels

    # Display
    st.image(results.ims[0], caption="Detected Image", use_column_width=True)
