import os
import subprocess
import sys
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Fix OpenCV libGL issue
if not os.path.exists("/usr/lib/libGL.so.1"):
    st.warning("⚠️ libGL.so.1 not found. Installing...")
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])
    st.success("✅ libGL installed successfully!")

# Clone YOLOv5 if not already
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("yolov5")

# Import YOLOv5 components
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Load the model
@st.cache_resource
def load_model():
    device = select_device("")
    model = attempt_load("yolov5/runs/train/exp3/weights/best.pt", map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# UI
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs using YOLOv5.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        detections = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw results
    if detections is not None and len(detections):
        detections = detections.cpu().numpy().astype(int)
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"ID {int(cls)} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
