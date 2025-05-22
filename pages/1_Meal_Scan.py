import streamlit as st
from PIL import ImageDraw, ImageFont, Image
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
import faiss
import csv
import re

st.set_page_config(page_title="📷 Meal Scan", page_icon="📷")
st.title("📷 Meal Detection")

# -------------------------------
# Caching heavy resources
# -------------------------------

@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_generation_pipeline():
    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return pipe, tokenizer

# Load models
y_model = load_yolo_model()
embedding_model = load_embedding_model()
pipe, tokenizer = load_generation_pipeline()

# Upload or take a photo
img_file = st.camera_input("Take a photo") or st.file_uploader("Or upload a meal image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Load and format QA pairs
# -------------------------------
qa_pairs = []
fields = [
    ("Calories (kcal)", "Calories"),
    ("Total Fat (g)", "Total Fat"),
    ("Saturated Fat (g)", "Saturated Fat"),
    ("Trans Fat (g)", "Trans Fat"),
    ("Cholesterol (mg)", "Cholesterol"),
    ("Sodium (mg)", "Sodium"),
    ("Total Carbohydrates (g)", "Total Carbohydrates"),
    ("Dietary Fiber (g)", "Dietary Fiber"),
    ("Sugars (g)", "Sugars"),
    ("Protein (g)", "Protein"),
]

with open("nutrition_table.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        item = row['Menu Item']
        question = f"What are the nutritional facts of {item}?"
        answer_parts = []
        for key, label in fields:
            value = row.get(key, "").strip()
            if value:
                suffix = "g" if any(x in label for x in ["Fat", "Carbohydrates", "Protein", "Sugars", "Fiber"]) else "mg" if label in ["Sodium", "Cholesterol"] else ""
                answer_parts.append(f"{label} {value} {suffix}")
            else:
                answer_parts.append(f"{label} missing")
        answer = ", ".join(answer_parts) + "."
        qa_pairs.append({
            "question": question,
            "answer": answer,
            "text": f"Question: {question}\nAnswer: {answer}"
        })

# -------------------------------
# Embedding and FAISS index
# -------------------------------
embeddings = embedding_model.encode([entry["text"] for entry in qa_pairs], convert_to_numpy=True)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

def get_top_k(query, k=5):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    return [qa_pairs[i] for i in I[0]]

def generate_answer(context_list, query):
    context = "\n".join([entry["text"] for entry in context_list])
    prompt = f"Context: {context}\nQuestion: What are the nutritional facts of {query}?"

    # Truncate prompt to fit within model token limit
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    truncated_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    output = pipe(truncated_prompt, max_new_tokens=100)
    return output[0].get("generated_text", output[0].get("text", "")).strip()

# -------------------------------
# Image analysis
# -------------------------------
if img_file:
    image = Image.open(img_file).convert("RGB")

    # Run YOLO prediction
    results = y_model.predict(image, conf=0.25)
    image_drawn = image.copy()
    draw = ImageDraw.Draw(image_drawn)

    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy()
    classes = results[0].names
    detected = [classes[int(cls)] for cls in boxes.cls]

    for (x1, y1, x2, y2), cls_id in zip(coords, boxes.cls):
        label = classes[int(cls_id)]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red")

    st.image(image_drawn, use_container_width=True)

    # Most confident detection
    confidences = results[0].boxes.conf.cpu().numpy()
    max_conf_idx = confidences.argmax()
    class_id = int(results[0].boxes.cls[max_conf_idx])
    class_name = results[0].names[class_id]

    # Get answer from top docs
    top_docs = get_top_k(class_name)
    answer = generate_answer(top_docs, class_name)

    st.write(class_name)
    st.success(answer)

    # -------------------------------
    # Parse nutrition data
    # -------------------------------
    pattern = r'([A-Za-z ]+?)\s+([\d.]+)\s*(mg|g|kcal)?'
    nutrients = re.findall(pattern, answer)

    data = {}
    for name, value, unit in nutrients:
        col_name = f"{name.strip()} ({unit})" if unit else name.strip()
        data[col_name] = float(value) if '.' in value else int(value)

    data = {"Food": class_name, **data}
    df_nutrition = pd.DataFrame([data])

    if "nutrition_data" not in st.session_state:
        st.session_state.nutrition_data = []

    st.dataframe(df_nutrition,hide_index=True)
    st.session_state.nutrition_data.append(df_nutrition.iloc[0])
