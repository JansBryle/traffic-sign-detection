import os
import sys
import subprocess
import torch
import streamlit as st
import numpy as np
from PIL import Image

# Install libGL if needed (for OpenCV compatibility)
if not os.path.exists("/usr/lib/libGL.so.1"):
    st.write("⚠️ libGL.so.1 not found. Installing required libraries...")
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])
    st.write("✅ libGL installed successfully!")

# Install OpenCV if not found
try:
    import cv2
except ImportError:
    st.write("⚠️ OpenCV not found. Installing OpenCV...")
    subprocess.run(["pip", "install", "opencv-python-headless==4.11.0.86"])
    st.write("✅ OpenCV installed! Please reboot the app.")
    st.stop()

# Clone YOLOv5 if not already present
if not os.path.exists("yolov5"):
    st.write("📥 Cloning YOLOv5 repo...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

# Add YOLOv5 to path
sys.path.append("yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load the model from Google Drive
@st.cache_resource
def load_model():
    import gdown
    model_url = "https://drive.google.com/uc?id=1P8Lvz_fMZYX2UDEI5KkSwsL26iZ-lGyv"  # Direct ID to best.pt
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.write("📥 Downloading model weights...")
        gdown.download(model_url, model_path, quiet=False)
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# Streamlit UI
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv5!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Preprocess
    img_resized = cv2.resize(img, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)[0]
    detections = non_max_suppression(pred, 0.4, 0.5)[0]

    # Draw boxes
    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], img.shape).round()
        for *xyxy, conf, cls in detections:
            label = f"{int(cls)} {conf:.2f}"
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(img, caption="Detections", use_column_width=True)
