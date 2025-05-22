import streamlit as st
import pandas as pd
import plotly.graph_objects as go



st.set_page_config(page_title="Food Tracker", page_icon="🍽️")


st.title("🍽️ Welcome to the Food Tracker App")
st.markdown("Use the sidebar to navigate between Meal Scan & Log Food")


st.title("📊 Dashboard")

DAILY_CALORIE_GOAL = 2000

# Load logged data
df = pd.DataFrame(st.session_state.get("nutrition_data", []))
total_calories = df["Calories (kcal)"].sum() if not df.empty else 0
remaining = max(0, DAILY_CALORIE_GOAL - total_calories)

# Calculate progress percentage
progress_pct = min(total_calories / DAILY_CALORIE_GOAL, 1.0) * 100
st.markdown("**Calories**")
st.markdown(f"Remaining: {remaining}")
# Donut chart with center annotation
fig = go.Figure(data=[
    go.Pie(
        values=[total_calories, remaining],
        labels=["Consumed", "Remaining"],
        hole=0.7,
        marker_colors=["#181818", "#FFFFFF"],
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
st.markdown(f"Base Goal: {DAILY_CALORIE_GOAL}")
st.markdown(f"Consumed: {total_calories}")
st.write("DataFrame columns:", df.columns.tolist())
if df["Sodium (mg)"].sum()>2000:
    st.warning("Reduce sodium intake, recommended intake has be exceeded. ")



# st.markdown("History")

# st.dataframe(df, hide_index=True)

if "nutrition_data" not in st.session_state:
    st.session_state.nutrition_data = []

if st.session_state.nutrition_data:
    st.subheader("History")
    df_all = pd.DataFrame(st.session_state.nutrition_data)
    st.dataframe(df_all,hide_index=True)

