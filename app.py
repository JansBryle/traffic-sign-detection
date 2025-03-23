import os
import sys
import subprocess
import streamlit as st
import torch
import numpy as np
from PIL import Image

# ==============================
# 📌 Ensure Dependencies are Installed
# ==============================
try:
    import cv2
except ModuleNotFoundError:
    st.warning("⚠️ OpenCV not found. Installing...")
    subprocess.run(["pip", "install", "opencv-python-headless==4.11.0.86"])
    st.success("✅ OpenCV installed! Restarting app...")
    st.experimental_rerun()

# ==============================
# 📌 Ensure YOLOv5 is Installed
# ==============================
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("./yolov5")
from ultralytics import YOLO  # Import YOLO after installation

# ==============================
# 📌 Load the Model
# ==============================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# ==============================
# 📌 Streamlit UI for Image Upload
# ==============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Draw bounding boxes
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, result.tolist())
        label = f"Class {cls}: {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
