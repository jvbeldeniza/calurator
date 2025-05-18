import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
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

    # Run YOLO prediction
    results = model.predict(image, conf=0.25)

    # Show results
    boxes = results[0].boxes
    img_with_boxes = results[0].plot()  # NumPy array

    st.image(img_with_boxes, caption="Detection Result", use_column_width=True)

    # Detected classes
    classes = results[0].names
    detected = [classes[int(cls)] for cls in boxes.cls]

    st.write("### Detected Objects:")
    for i, label in enumerate(detected, start=1):
        st.write(f"{i}. {label}")

    # Nutrition info
    nutrition_data = {
        "Food": ["Chicken", "Rice", "Broccoli", "Egg", "Apple", "Milk"],
        "Calories (kcal)": [165, 130, 55, 155, 52, 42],
        "Protein (g)": [31, 2.7, 3.7, 13, 0.3, 3.4],
        "Fat (g)": [3.6, 0.3, 0.6, 11, 0.2, 1.0],
        "Carbs (g)": [0, 28, 11.2, 1.1, 14, 5]
    }

    df_nutrition = pd.DataFrame(nutrition_data)

    st.write("### 🍽️ Nutrition Info (per 100g)")

    for food in detected:
        row = df_nutrition[df_nutrition["Food"] == food]
        if not row.empty:
            st.write(f"**{food}**")
            st.write(f"- Calories: {int(row['Calories (kcal)'].values[0])} kcal")
            st.write(f"- Protein: {row['Protein (g)'].values[0]} g")
            st.write(f"- Fat: {row['Fat (g)'].values[0]} g")
            st.write(f"- Carbs: {row['Carbs (g)'].values[0]} g")
            st.markdown("---")
        else:
            st.warning(f"Nutritional data for **{food}** not found.")