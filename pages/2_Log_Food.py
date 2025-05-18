import streamlit as st
import pandas as pd

st.title("📝 Log Food")

if 'food_log' not in st.session_state:
    st.session_state.food_log = []

food = st.text_input("Food Item")
calories = st.number_input("Calories", min_value=0)

if st.button("Add Entry"):
    if food and calories:
        st.session_state.food_log.append({"Food": food, "Calories": calories})
        st.success(f"Added {food} ({calories} cal)")

df = pd.DataFrame(st.session_state.food_log)
st.dataframe(df)