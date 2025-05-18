import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
st.set_page_config(page_title="📷 Meal Scan", page_icon="📷")

st.title("📷 Meal Detection")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Replace with your actual path
    return model

model = load_model()

# Upload or take a photo
img_file = st.camera_input("Take a photo") or st.file_uploader("Or upload a meal image", type=["jpg", "jpeg", "png"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    # Run YOLO prediction
    results = model.predict(image)

    # Show results
    boxes = results[0].boxes
    img_with_boxes = results[0].plot()  # NumPy array

    st.image(img_with_boxes, caption="Detection Result", use_column_width=True)

    # Optionally list detected classes
    classes = results[0].names
    detected = [classes[int(cls)] for cls in boxes.cls]
    st.write("Detected objects:", detected)