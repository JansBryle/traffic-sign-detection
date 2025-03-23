import os
import sys
import subprocess
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ================================
# 📌 Ensure YOLOv5 is Installed
# ================================
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

# Add YOLOv5 directory to Python path
sys.path.append("./yolov5")  # Ensures Python can find YOLOv5

# Import YOLOv5 modules
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ================================
# 📌 Convert Model to be Compatible with OpenCV 4
# ================================
MODEL_PATH = "best.pt"
CONVERTED_MODEL_PATH = "best_converted.pt"

if not os.path.exists(CONVERTED_MODEL_PATH):
    st.write("🔄 Converting YOLO model for OpenCV 4 compatibility...")
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    torch.save(model, CONVERTED_MODEL_PATH)
    st.write("✅ Model converted successfully!")

# ================================
# 📌 Load the Model
# ================================
@st.cache_resource
def load_model():
    device = select_device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(CONVERTED_MODEL_PATH, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ================================
# 🎨 Streamlit UI
# ================================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert image for OpenCV
    image = Image.open(uploaded_file)
    img0 = np.array(image)

    # Convert image for YOLOv5
    img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))  # Resize to YOLO input size
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 640, 640]

    # Run inference
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)

    # Draw detections
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Class {int(cls)}: {conf:.2f}"
                cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img0, caption="Detected Traffic Signs", use_column_width=True)
