import os
import sys
import subprocess
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ============================
# 🔧 Install OpenCV if Missing
# ============================
try:
    _ = cv2.__version__
except Exception:
    st.warning("⚠️ OpenCV not found. Installing...")
    subprocess.run(["pip", "install", "opencv-python-headless==4.11.0.86"])
    st.success("✅ OpenCV installed! Please restart the app.")
    st.stop()

# ============================
# 🔽 Download best.pt from Google Drive if not present
# ============================
model_url = "https://drive.google.com/uc?id=1fSOVVSpN1fLu_Q_2LU_OOekUrfbFHz6R"
model_path = "best.pt"

if not os.path.exists(model_path):
    st.write("⬇️ Downloading trained model from Google Drive...")
    subprocess.run(["gdown", model_url, "-O", model_path], check=True)
    st.success("✅ Model downloaded!")

# ============================
# 🔁 Add yolov5 path
# ============================
if not os.path.exists("yolov5"):
    st.write("🔄 Cloning YOLOv5 repository...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])

sys.path.append("yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ============================
# 🧠 Load Model
# ============================
@st.cache_resource
def load_model():
    device = select_device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ============================
# 🖼️ Streamlit UI
# ============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    orig = np.array(image)

    # Preprocess image
    img = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw boxes
    if pred is not None:
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig.shape).round()
        for *xyxy, conf, cls in pred:
            label = f"{int(cls)} {conf:.2f}"
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(orig, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(orig, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(orig, caption="📸 Detected Traffic Signs", use_column_width=True)
