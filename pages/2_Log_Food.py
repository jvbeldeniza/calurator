# import streamlit as st
# import pandas as pd

# st.title("📝 Log Food")

# # Initialize the session state log
# if "nutrition_data" not in st.session_state:
#     st.session_state.nutrition_data = []

# # Input form
# food = st.text_input("Food")
# calories = st.number_input("Calories", min_value=0)

# if st.button("Add Entry"):
#     if food and calories:
#         st.session_state.nutrition_data.append({"Food": food, "Calories": calories})
#         st.success(f"Added {food} ({calories} cal)")

# # Create a DataFrame from the session log
# df = pd.DataFrame(st.session_state.nutrition_data)

# # Editable table
# edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# # Button to update session state
# if st.button("💾 Save Changes"):
#     st.session_state.nutrition_data = edited_df.to_dict(orient="records")
#     st.success("Food log updated!")

import streamlit as st
import pandas as pd

st.title("📝 Log Food")

# Define all column headers
nutrition_columns = [
    "Food", "Calories", "Total Fat (g)", "Saturated Fat (g)", "Trans Fat (g)",
    "Cholesterol (mg)", "Sodium (mg)", "Total Carbohydrates (g)",
    "Dietary Fiber (g)", "Sugars (g)", "Protein (g)"
]

# Initialize the session state log
if "nutrition_data" not in st.session_state:
    st.session_state.nutrition_data = []

# Input form
food = st.text_input("Food")
calories = st.number_input("Calories", min_value=0)

if st.button("Add Entry"):
    if food and calories:
        # Create entry with only Food and Calories, others as None
        entry = {col: None for col in nutrition_columns}
        entry["Food"] = food
        entry["Calories"] = calories

        st.session_state.nutrition_data.append(entry)
        st.success(f"Added {food} ({calories} cal)")

# Create a DataFrame with all entries
df = pd.DataFrame(st.session_state.nutrition_data)

# Ensure all expected columns exist
for col in nutrition_columns:
    if col not in df.columns:
        df[col] = None

# Editable table
edited_df = st.data_editor(df[nutrition_columns], num_rows="dynamic", use_container_width=True)

# Save changes
if st.button("💾 Save Changes"):
    st.session_state.nutrition_data = edited_df.to_dict(orient="records")
    st.success("Food log updated!")

