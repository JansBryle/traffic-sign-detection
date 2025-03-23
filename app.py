import os
import sys
import subprocess
import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

# ===============================
# 📌 Ensure YOLOv5 is installed
# ===============================
YOLO_DIR = "yolov5"
if not os.path.exists(YOLO_DIR):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", f"{YOLO_DIR}/requirements.txt"])

sys.path.append(YOLO_DIR)

# ===============================
# 📌 Install best.pt if not present
# ===============================
MODEL_PATH = f"{YOLO_DIR}/best.pt"
if not os.path.exists(MODEL_PATH):
    st.write("⬇️ Downloading best.pt from Google Drive...")
    subprocess.run([
        "pip", "install", "gdown"
    ])
    subprocess.run([
        "gdown", "--id", "1OrI8pUInm5Xm18EHzbbneSKfsfvRW7Ti", "--output", MODEL_PATH
    ])

# ===============================
# 📌 Import after ensuring YOLOv5 is setup
# ===============================
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# ===============================
# 📌 Load Model
# ===============================
@st.cache_resource
def load_model():
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ===============================
# 🚀 Streamlit App
# ===============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs using YOLOv5!")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_tensor = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_tensor = torch.from_numpy(img_tensor).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)
        preds = non_max_suppression(preds, conf_thres=0.4, iou_thres=0.5)[0]

    for det in preds:
        x1, y1, x2, y2, conf, cls = map(int, det.tolist())
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {cls}: {conf:.2f}"
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img_np, caption="🔍 Detection Results", use_column_width=True)
