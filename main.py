import streamlit as st
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(page_title="Food Tracker", page_icon="🍽️")

st.title("🍽️ Welcome to the Food Tracker App")
st.markdown("Use the sidebar to navigate between Meal Scan & Log Food")


st.title("📊 Dashboard")

DAILY_CALORIE_GOAL = 2000

# Load logged data
df = pd.DataFrame(st.session_state.get("food_log", []))
total_calories = df["Calories"].sum() if not df.empty else 0
remaining = max(0, DAILY_CALORIE_GOAL - total_calories)

# Calculate progress percentage
progress_pct = min(total_calories / DAILY_CALORIE_GOAL, 1.0) * 100

# Donut chart with center annotation
fig = go.Figure(data=[
    go.Pie(
        values=[total_calories, remaining],
        labels=["Consumed", "Remaining"],
        hole=0.7,
        marker_colors=["#FF6961", "#90EE90"],
        textinfo="none",
        sort=False,
        direction='clockwise'
    )
])

# Add text in the center of the donut
fig.update_layout(
    annotations=[
        dict(
            text=f"<b>{remaining} / {DAILY_CALORIE_GOAL}</b>",
            font_size=20,
            showarrow=False
        )
    ],
    showlegend=True,
    height=350,
    margin=dict(t=30, b=10, l=10, r=10),
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("Base Goal: {DAILY_CALORIE_GOAL}")
st.markdown("Consumed: {remaining}")