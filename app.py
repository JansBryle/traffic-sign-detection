import os
import sys
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import subprocess

# ✅ Clone YOLOv5 if not present
if not os.path.exists("yolov5"):
    st.info("Cloning YOLOv5 repo...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"])

# ✅ Add yolov5 to Python path
sys.path.append("yolov5")

# ✅ YOLOv5 Imports
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# ✅ Load YOLOv5 Model
@st.cache_resource
def load_model():
    device = select_device("")
    model = attempt_load("yolov5/runs/train/exp/weights/best.pt", device=device)  # <-- update path if needed
    model.eval()
    return model, device

model, device = load_model()

# ✅ Streamlit UI
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image to detect traffic signs using YOLOv5!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    original_image = image_np.copy()

    # ✅ Preprocess
    img_resized = cv2.resize(image_np, (640, 640))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # ✅ Inference
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
        detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # ✅ Draw Detections (scale boxes to original size)
    if detections is not None and len(detections):
        detections = detections.cpu()
        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], original_image.shape).round()
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(original_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(original_image, caption="📸 Detected Traffic Signs", use_container_width=True)
