import os
import sys
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# ==============================
# 📌 Ensure YOLOv5 Modules are Available
# ==============================
try:
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.general import non_max_suppression
    from yolov5.utils.torch_utils import select_device
except ModuleNotFoundError:
    st.error("❌ YOLOv5 module not found! Ensure requirements are properly installed.")
    sys.exit()

# ==============================
# 📌 Load the YOLOv5 Model
# ==============================
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Path to trained weights
    device = select_device("cpu")  # Force CPU usage
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# ==============================
# 📌 Streamlit UI for Image Upload
# ==============================
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert image for processing
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convert color space
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Convert to tensor
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        pred = model(img)
    pred = non_max_suppression(pred[0], conf_thres=0.4, iou_thres=0.5)

    # Draw detections
    if pred[0] is not None:
        for det in pred[0]:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4]
            cls = det[5]
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)