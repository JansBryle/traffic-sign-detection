import os
import sys
import subprocess

# ========================
# 🔧 Fix for OpenCV issue
# ========================
try:
    import cv2
except ImportError:
    print("⚠️ OpenCV or system dependency not found. Installing libGL and OpenCV...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1"], check=True)
    subprocess.run(["pip", "install", "opencv-python-headless==4.11.0.86"], check=True)
    print("✅ Dependencies installed! Please restart the app manually.")
    st.stop()

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import gdown

# Add yolov5 to path
YOLO_DIR = os.path.join(os.getcwd(), "yolov5")
sys.path.append(YOLO_DIR)

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ===============================
# 📥 Download model from Google Drive
# ===============================
MODEL_PATH = "best.pt"
GDRIVE_FILE_ID = "1XKWD_WG4h0MOY0kA0zoKzFWTJJ1Hvukv"  # replace with your real file ID

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# ===============================
# 📦 Load YOLOv5 model
# ===============================
@st.cache_resource
def load_model():
    device = select_device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ===============================
# 🖼 Streamlit UI
# ===============================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv5.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
    detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Draw boxes
    for *xyxy, conf, cls in detections:
        label = f"{int(cls)}: {conf:.2f}"
        xyxy = [int(x.item()) for x in xyxy]
        cv2.rectangle(image_np, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(image_np, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
