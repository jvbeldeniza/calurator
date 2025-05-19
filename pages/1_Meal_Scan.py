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

    st.write("### 🍽️ Nutrition Info")

    # RAG--------------------------------------------

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sample nutrition facts
    texts = [
        "Avocado is rich in healthy fats and contains nearly 20 vitamins and minerals.",
        "Chicken breast is a great source of lean protein.",
        "Quinoa is a high-protein grain suitable for gluten-free diets.",
        "Carrots are high in beta-carotene, fiber, and antioxidants.",
        "Chicken Joy has 620 calories."
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(texts)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Store docs for retrieval
    def get_top_k(query, k=2):
        query_vec = embedding_model.encode([query])
        D, I = index.search(query_vec, k)
        return [texts[i] for i in I[0]]
    
    qa_pipeline = pipeline("text-generation", model="sshleifer/tiny-gpt2", max_new_tokens=50)

    def generate_answer(context, query):
        prompt = f"Context: {context}\nQuestion: What is the calories of {query}?\nAnswer:"
        output = qa_pipeline(prompt, do_sample=False)[0]["generated_text"]
        return output.split("Answer:")[-1].strip()
    
    query = "Chicken Joy"

    if query:
        docs = get_top_k(query)
        context = " ".join(docs)
        st.markdown("**Retrieved Context:**")
        st.info(context)

        answer = generate_answer(context, query)
        st.markdown("**Generated Answer:**")
        st.success(answer)  

    # -----------------------------------------------

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
    
    for _, row in edited_df.reset_index().iterrows():
        entry = {"Food": row["Food"], "Calories": int(row["Calories (kcal)"])}
        st.session_state.food_log.append(entry)

    