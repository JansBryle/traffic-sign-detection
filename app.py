import os
import sys
import subprocess
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# ===============================
# 📌 Clone YOLOv5 if not present
# ===============================
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.plots import output_to_target
from yolov5.utils.torch_utils import select_device

# =================================
# 📌 Load Model from local path
# =================================
MODEL_PATH = "yolov5/runs/train/exp3/weights/best.pt"  # <- Replace with the correct path

@st.cache_resource
def load_model():
    device = select_device("cpu")
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ==================================
# 📌 Streamlit User Interface
# ==================================
st.title("🚦 Traffic Sign Detection")
st.write("Upload an image to detect traffic signs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    img_tensor = torch.from_numpy(img).to(device).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], image_np.shape).round()
        for *xyxy, conf, cls in pred:
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(image_np, caption="Detected Image", use_column_width=True)
