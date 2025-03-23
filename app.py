import os
import sys
import subprocess

# 🛠 Fix for OpenCV libGL.so.1 error on Streamlit Cloud
if not os.path.exists("/usr/lib/libGL.so.1"):
    print("⚠️ libGL.so.1 not found. Installing...")
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])
    print("✅ libGL installed successfully!")

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

# Clone YOLOv5 if not present
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

# Add yolov5 to system path
sys.path.append("yolov5")

# Import YOLOv5 modules
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load model
@st.cache_resource
def load_model():
    weights_path = "yolov5/runs/train/exp3/weights/best.pt"  # Change if needed
    device = select_device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# Streamlit UI
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Prepare image for inference
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        pred = model(img)[0]
    detections = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw detections
    if detections is not None and len(detections):
        detections = detections.cpu()
        for *xyxy, conf, cls in detections:
            label = f'{int(cls.item())} {conf:.2f}'
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(image_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(image_np, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
