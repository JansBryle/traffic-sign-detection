import os
import sys
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Append YOLOv5 path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join('yolov5', 'runs', 'train', 'exp3', 'weights', 'best.pt')
    device = select_device('cpu')  # or '0' if using GPU on local
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# Streamlit UI
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image and detect traffic signs!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    # Draw
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image_np.shape).round()
        for *xyxy, conf, cls in pred:
            label = f"{int(cls)} {conf:.2f}"
            cv2.rectangle(image_np, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(image_np, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(image_np, caption="Detected Traffic Signs", use_column_width=True)
