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

# Define all the nutrition table columns
nutrition_columns = [
    "Food", "Calories", "Total Fat (g)", "Saturated Fat (g)", "Trans Fat (g)",
    "Cholesterol (mg)", "Sodium (mg)", "Total Carbohydrates (g)",
    "Dietary Fiber (g)", "Sugars (g)", "Protein (g)"
]

# Initialize session state
if "nutrition_data" not in st.session_state:
    st.session_state.nutrition_data = []

# Normalize and validate session data before creating DataFrame
raw_data = st.session_state.get("nutrition_data", [])

# Food input form
food = st.text_input("Food")
calories = st.number_input("Calories", min_value=0, step=1)

if st.button("➕ Add Entry"):
    if food and calories:
        # Create a new row with only food and calories, others blank
        new_entry = {col: None for col in nutrition_columns}
        new_entry["Food"] = food
        new_entry["Calories"] = calories
        st.session_state.nutrition_data.append(new_entry)
        st.success(f"Added {food} ({calories} cal)")



# # Ensure the data is a list of dicts
# if isinstance(raw_data, dict):
#     raw_data = [raw_data]

# valid_data = []
# for entry in raw_data:
#     if isinstance(entry, dict):
#         normalized_entry = {col: entry.get(col, None) for col in nutrition_columns}
#         valid_data.append(normalized_entry)

# # Build the editable DataFrame
# df = pd.DataFrame(valid_data)

# # Show editable table
# edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# # Save updates
# if st.button("💾 Save Changes"):
#     # Ensure all rows maintain correct format
#     cleaned_data = []
#     for row in edited_df.to_dict(orient="records"):
#         fixed_row = {col: row.get(col, None) for col in nutrition_columns}
#         cleaned_data.append(fixed_row)
#     st.session_state.nutrition_data = cleaned_data
#     st.success("✅ Food log updated!")


if "nutrition_data" not in st.session_state:
    st.session_state.nutrition_data = []

if st.session_state.nutrition_data:
    st.subheader("Food Log")
    df_all = pd.DataFrame(st.session_state.nutrition_data)
    edited_df = st.data_editor(df_all, num_rows="dynamic", use_container_width=True)
if st.button("💾 Save Changes"):
    st.session_state.nutrition_data = edited_df.to_dict(orient="records")