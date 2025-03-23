import cv2
import numpy as np
import torch
from PIL import Image

# Load the YOLO model (Ensure you have the correct path to your trained weights)
MODEL_PATH = "best.pt"  # Change this to your actual model's path
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)

def detect_traffic_sign(image):
    """
    Performs YOLO-based traffic sign detection on the input image.

    Args:
        image (numpy.ndarray): The image to process.

    Returns:
        detected_image (numpy.ndarray): Image with detected bounding boxes.
        detected_label (str): Label of the detected traffic sign.
    """
    # Convert image to RGB (YOLOv5 expects RGB format)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Convert results to pandas DataFrame
    detections = results.pandas().xyxy[0]  # Extract bounding boxes, labels, and scores

    # If no detections, return the original image
    if detections.empty:
        return image, "No traffic sign detected"

    # Draw bounding boxes on the image
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        label = row["name"]
        confidence = row["confidence"]

        # Draw bounding box and label
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(image, f"{label} ({confidence:.2f})", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Return the modified image and label of the first detected object
    return image, label
