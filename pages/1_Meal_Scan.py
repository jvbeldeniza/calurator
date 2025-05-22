import streamlit as st
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import csv
import faiss
import numpy as np
from transformers import AutoTokenizer, pipeline
import re

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


    
    # -------------------------------
    # Step 4: Retrieval + Generation
    # -------------------------------

    # Retrieval function
    def get_top_k(query, k=5):
        query_vec = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_vec, k)
        return [qa_pairs[i] for i in I[0]]

    # Text generation pipeline
    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def generate_answer(context_list, query):
        context = "\n".join([entry["text"] for entry in context_list])
        prompt = f"Context: {context}\nQuestion: What are the nutritional facts of {query}?"

        # Truncate to fit within model limit
        tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        truncated_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        output = pipe(truncated_prompt, max_new_tokens=100)
        return output[0].get("generated_text", output[0].get("text", "")).strip()

    query = classes[0]
    print(query)
    
    top_docs = get_top_k(query)
    answer = generate_answer(top_docs, detected)
    print("Generated Answer:")
    st.success(answer)  

    # -----------------------------------------------

# Extract nutrient, value, and unit
    match_first_word = re.match(r'^([A-Za-z]+)Calories', classes[0]+answer)
    first_word = match_first_word.group(1) if match_first_word else None

    # Step 2: Extract nutrition info (name, value, unit)
    pattern = r'([A-Za-z ]+?)\s+([\d.]+)\s*(mg|g|kcal)?'
    nutrients = re.findall(pattern, answer)

    # Build dict with units in column names
    data = {}
    for name, value, unit in nutrients:
        col_name = f"{name.strip()} ({unit})" if unit else name.strip()
        data[col_name] = float(value) if '.' in value else int(value)

    # Add first word as column
    data = {"Food": first_word, **data}

    # Convert to DataFrame
    df_nutrition = pd.DataFrame([data])

    st.dataframe(df_nutrition)


    