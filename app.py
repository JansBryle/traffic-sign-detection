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
    from ultralytics import YOLO  # Using the official Ultralytics YOLO package
except ModuleNotFoundError:
    st.error("❌ YOLOv5 module not found! Ensure requirements are properly installed.")
    sys.exit()

# ==============================
# 📌 Load the YOLOv5 Model
# ==============================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Load the trained YOLO model
    return model

model = load_model()

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

    # Run inference
    results = model(image_np)  # Pass image directly into YOLO model

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index

            # Draw rectangle and label
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
