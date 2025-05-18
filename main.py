import streamlit as st
import pandas as pd


st.set_page_config(page_title="Food Tracker", page_icon="🍽️")

st.title("🍽️ Welcome to the Food Tracker App")
st.markdown("Use the sidebar to navigate between Meal Scan & Log Food")


st.title("📊 Dashboard")

DAILY_CALORIE_GOAL = 2000

df = pd.DataFrame(st.session_state.get("food_log", []))
total_calories = df["Calories"].sum() if not df.empty else 0
remaining = max(0, DAILY_CALORIE_GOAL - total_calories)

# Show metric and linear progress bar
st.metric("Remaining Calories", f"{remaining} cal")
st.progress(min(total_calories / DAILY_CALORIE_GOAL, 1.0))

# Add donut (PI progress) chart
fig = go.Figure(data=[
    go.Pie(
        values=[total_calories, remaining],
        labels=["Consumed", "Remaining"],
        hole=0.7,
        marker_colors=["#FF6961", "#90EE90"],
        textinfo="none"
    )
])

fig.update_layout(
    showlegend=True,
    height=300,
    margin=dict(t=10, b=10, l=10, r=10),
)

st.plotly_chart(fig, use_container_width=True)