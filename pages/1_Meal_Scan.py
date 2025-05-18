import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.title("📷 Meal Scan")

picture = st.camera_input("Take a picture of your meal")

if picture:
    img = Image.open(picture)
    st.image(img, caption="Captured Meal", use_column_width=True)