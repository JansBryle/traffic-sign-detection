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
from yolov5.utils.augmentations import letterbox

# ✅ Load YOLOv5 Model
@st.cache_resource
def load_model():
    device = select_device("")
    model = attempt_load("yolov5/runs/train/exp2/weights/best.pt", device=device)
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

    # ✅ Preprocess using letterbox
    img_resized = letterbox(image_np, new_shape=640)[0]
    img = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension

    # ✅ Inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # ✅ Draw Detections
    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], image_np.shape).round()
        detections = detections.cpu().numpy()
        class_names = ["prohibitory", "danger", "mandatory", "other"]

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    st.image(image_np, caption="📸 Detected Traffic Signs", use_container_width=True)
