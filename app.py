import os
import subprocess

# ✅ Auto-install libGL if missing (for OpenCV)
if not os.path.exists("/usr/lib/x86_64-linux-gnu/libGL.so.1"):
    print("⚠️ libGL.so.1 not found. Installing...")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx"], check=True)
    print("✅ libGL.so.1 installed!")

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# ================================
# ✅ Ensure YOLOv5 is available
# ================================
import sys
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])
sys.path.append("yolov5")

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ================================
# 🔄 Load YOLOv5 model
# ================================
@st.cache_resource
def load_model():
    model_path = "best.pt"
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ================================
# 🌐 Streamlit UI
# ================================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs using YOLOv5!")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    for *box, conf, cls in pred:
        x1, y1, x2, y2 = map(int, box)
        label = f"{int(cls)} {conf:.2f}"
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(img_np, caption="🔍 Detection Results", use_column_width=True)
