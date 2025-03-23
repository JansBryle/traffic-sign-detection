import os
import sys
import subprocess

# =========================
# 📌 Install libGL for OpenCV
# =========================
if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
    print("⚠️ libGL.so.1 not found. Installing...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1"], check=True)
    print("✅ libGL installed successfully!")

# ================
# 📌 Import packages
# ================
import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

# ========================
# 📌 YOLOv5 Setup
# ========================
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("yolov5")

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ============================
# 📌 Load model from Google Drive (mounted beforehand)
# ============================
@st.cache_resource
def load_model():
    model_path = "yolov5/runs/train/exp3/weights/best.pt"  # or update to your actual path
    device = select_device("")
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ====================
# 📌 Streamlit UI
# ====================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Preprocess
    img = torch.from_numpy(img_bgr).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)[0]

    # Draw boxes
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_np.shape).round()

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img_np, caption="Detected Signs", use_column_width=True)
