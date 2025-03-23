import os
import sys
import subprocess
import torch
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

# =====================================
# 🧠 Install and Import YOLOv5 if Needed
# =====================================
YOLO_DIR = "yolov5"
if not os.path.exists(YOLO_DIR):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", f"{YOLO_DIR}/requirements.txt"])

sys.path.append(YOLO_DIR)

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# =====================================
# ⚙️ Check for OpenCV libGL
# =====================================
if not os.path.exists("/usr/lib/libGL.so.1"):
    st.write("⚠️ OpenCV dependency not found. Installing libGL...")
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "libgl1"])
    st.write("✅ libGL installed! Please reboot the app.")
    st.stop()

# =====================================
# 📦 Load YOLOv5 Model from Google Drive
# =====================================
@st.cache_resource
def load_model():
    model_path = "best.pt"

    # If not downloaded yet, download from Google Drive
    if not os.path.exists(model_path):
        st.write("📥 Downloading model from Google Drive...")
        gdown_cmd = f"gdown --id 1j8x3EymuT_3V3aapnYF--dMaKs5WBjWR -O {model_path}"
        subprocess.run(gdown_cmd, shell=True)
        st.write("✅ Model downloaded!")

    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# =====================================
# 🎯 Streamlit Interface
# =====================================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_image = np.array(image)

    # Preprocess
    img = torch.from_numpy(orig_image).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw detections
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig_image.shape).round()
        for *xyxy, conf, cls in pred:
            label = f'{int(cls)} {conf:.2f}'
            x1, y1, x2, y2 = map(int, xyxy)
            orig_image = cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            orig_image = cv2.putText(orig_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.5, (0, 255, 0), 2)

    st.image(orig_image, caption="🧠 Detected Traffic Signs", use_column_width=True)
