import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolo_inference import detect_traffic_sign

# Streamlit App Title
st.title("🚦 Traffic Sign Detection App")
st.write("Upload an image, and the model will classify the traffic sign.")

# Upload Image
uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform YOLO detection
    st.write("Processing image... 🔍")
    detected_image, detected_label = detect_traffic_sign(image)

    # Display the result
    st.image(detected_image, caption=f"Detected: {detected_label}", use_column_width=True)
    st.success(f"Traffic Sign Identified: {detected_label}")
