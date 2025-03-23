import os
import sys
import subprocess

# ============================
# 📦 Fix libGL.so.1 issue
# ============================
if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
    print("⚠️ libGL.so.1 missing. Installing libgl1...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1"], check=True)
    print("✅ libGL.so.1 installed!")

# ============================
# 🧠 Imports
# ============================
import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Add YOLOv5 to path
sys.path.append(os.path.join(os.getcwd(), "yolov5"))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ============================
# 🚀 Load model
# ============================
@st.cache_resource
def load_model():
    model_path = "yolov5/runs/train/exp3/weights/best.pt"  # Update path if needed
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ============================
# 🌐 Streamlit UI
# ============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Prepare image for model
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run YOLO inference
    with torch.no_grad():
        pred = model(img)[0]
    detections = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw bounding boxes
    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], image_np.shape).round()
        for *xyxy, conf, cls in detections:
            label = f'{int(cls)} {conf:.2f}'
            cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(image_np, caption='Detected Traffic Signs', use_column_width=True)
