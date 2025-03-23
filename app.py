import os
import sys
import subprocess
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==============================
# 📌 Ensure Dependencies Are Installed
# ==============================

try:
    import cv2
except ImportError:
    st.warning("⚠️ OpenCV not found. Installing OpenCV...")
    subprocess.run(["pip", "install", "opencv-python-headless"])
    st.success("✅ OpenCV installed! Please restart the app manually.")
    sys.exit()

# ==============================
# 📌 Ensure YOLOv5 is Installed
# ==============================
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("./yolov5")  # Ensure Python finds YOLOv5

# Import YOLOv5 modules
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ==============================
# 📌 Load YOLOv5 Model
# ==============================
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Ensure this file is in your repo
    device = select_device("cpu")  # Change to 'cuda' if GPU available
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ==============================
# 📌 Streamlit UI
# ==============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert image for OpenCV
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Preprocess image for YOLO
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Run YOLOv5 inference
    with torch.no_grad():
        pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw detections
    for det in pred:
        x1, y1, x2, y2, conf, cls = map(int, det.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {int(cls)}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image, caption="Detected Traffic Signs", use_column_width=True)
