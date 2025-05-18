import streamlit as st
import pandas as pd

st.title("📊 Dashboard")

DAILY_CALORIE_GOAL = 2000

df = pd.DataFrame(st.session_state.get("food_log", []))
total_calories = df["Calories"].sum() if not df.empty else 0
remaining = DAILY_CALORIE_GOAL - total_calories

st.metric("Remaining Calories", f"{remaining} cal")
st.progress((DAILY_CALORIE_GOAL - remaining) / DAILY_CALORIE_GOAL)