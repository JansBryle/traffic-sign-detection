import os
import sys
import subprocess
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# 📌 Fix OpenCV shared lib issue
if not os.path.exists("/usr/lib/libGL.so.1"):
    st.write("⚠️ libGL.so.1 not found. Installing...")
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])
    st.write("✅ libGL installed.")

# 📦 Add yolov5 to path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(YOLOV5_PATH)

# 🔧 YOLOv5 imports (from your local yolov5 folder)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# 📁 Load model weights from Google Drive (you must upload manually beforehand)
MODEL_PATH = 'yolov5/runs/train/exp3/weights/best.pt'  # update this if needed

@st.cache_resource
def load_model():
    device = select_device('')
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# 🎨 Streamlit UI
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Preprocessing
    img0 = img.copy()
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{int(cls)} {conf:.2f}'
            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    st.image(img0, caption="Detected Traffic Signs", use_column_width=True)
