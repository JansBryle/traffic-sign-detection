import os
import sys
import subprocess
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Add yolov5 to path
if not os.path.exists("yolov5"):
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])
sys.path.append("yolov5")

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Download weights if not found
if not os.path.exists("best.pt"):
    import gdown
    gdown.download("https://drive.google.com/uc?id=1xHSX7_0Y3a0LBHD56c_96W1dSHVxIm_e", "best.pt", quiet=False)

@st.cache_resource
def load_model():
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = attempt_load("best.pt", map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# UI
st.title("🚦 Traffic Sign Detection App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Preprocess
    im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im = torch.from_numpy(im).to(device).float() / 255.0
    im = im.permute(2, 0, 1).unsqueeze(0)

    # Inference
    pred = model(im)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], img.shape).round()
        for *xyxy, conf, cls in pred:
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img, caption="Detected Traffic Signs", use_column_width=True)
