import streamlit as st
from PIL import ImageDraw, ImageFont
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

    image_drawn = image.copy()
    draw = ImageDraw.Draw(image_drawn)

    # Load bounding box info
    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy()
    classes = results[0].names
    detected = [classes[int(cls)] for cls in boxes.cls]

    # Draw bounding boxes and labels
    for (x1, y1, x2, y2), cls_id in zip(coords, boxes.cls):
        label = classes[int(cls_id)]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")

    # Show image with bounding boxes overlaid
    st.image(image_drawn, use_container_width=True)

    # Detected classes
    classes = results[0].names
    detected = [classes[int(cls)] for cls in boxes.cls]

    # Nutrition info
    nutrition_data = {
        "Food": ["Chicken", "JFC - Burger Steak", "Broccoli", "Egg", "Apple", "Milk"],
        "Calories (kcal)": [165, 130, 55, 155, 52, 42],
        "Protein (g)": [31, 2.7, 3.7, 13, 0.3, 3.4],
        "Fat (g)": [3.6, 0.3, 0.6, 11, 0.2, 1.0],
        "Carbs (g)": [0, 28, 11.2, 1.1, 14, 5]
    }

    df_nutrition = pd.DataFrame(nutrition_data)

    st.write("### 🍽️ Nutrition Info (per 100g)")

    # for food in detected:
    #     row = df_nutrition[df_nutrition["Food"] == food]
    #     if not row.empty:
    #         st.write(f"**{food}**")
    #         st.write(f"- Calories: {int(row['Calories (kcal)'].values[0])} kcal")
    #         st.write(f"- Protein: {row['Protein (g)'].values[0]} g")
    #         st.write(f"- Fat: {row['Fat (g)'].values[0]} g")
    #         st.write(f"- Carbs: {row['Carbs (g)'].values[0]} g")
    #         st.markdown("---")
    #     else:
    #         st.warning(f"Nutritional data for **{food}** not found.")
    filtered_df = df_nutrition[df_nutrition["Food"].isin(detected)]

    # Optional: Reorder rows to match detection order
    filtered_df["Detection Order"] = filtered_df["Food"].apply(lambda x: detected.index(x))
    filtered_df = filtered_df.sort_values("Detection Order").drop(columns="Detection Order")

    if 'food_log' not in st.session_state:
        st.session_state.food_log = []

    # Editable table
    edited_df = st.data_editor(
        filtered_df.set_index("Food"),
        num_rows="dynamic",
        use_container_width=True
    )

    # Update session_state food_log after edits
    st.session_state.food_log = []
    for _, row in edited_df.reset_index().iterrows():
        entry = {"Food": row["Food"], "Calories": int(row["Calories (kcal)"])}
        st.session_state.food_log.append(entry)

    