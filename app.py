import os
import sys
import subprocess
import gdown

# =========================
# 📌 Install libGL for OpenCV
# =========================
if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])

# ======================
# 📌 Download YOLO weights from Google Drive (best.pt)
# ======================
weights_dir = "yolov5/runs/train/exp3/weights"
os.makedirs(weights_dir, exist_ok=True)
weights_path = os.path.join(weights_dir, "best.pt")

if not os.path.exists(weights_path):
    # Replace this with your correct Google Drive file ID
    file_id = "1PKFLbKkFXt6zXk2nbS7JX2e95QgAH_9F"  # Example ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", weights_path, quiet=False)

# ======================
# 📌 Import libraries
# ======================
import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

# Add YOLOv5 folder to path
sys.path.append("yolov5")

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ======================
# 📌 Load model
# ======================
@st.cache_resource
def load_model():
    device = select_device("")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ======================
# 📌 Streamlit UI
# ======================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Preprocess image
    img = torch.from_numpy(img_bgr).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)[0]

    # Draw results
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_np.shape).round()

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(img_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img_np, caption="Detected Traffic Signs", use_column_width=True)
