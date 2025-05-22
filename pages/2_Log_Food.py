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

# Initialize session state if not already
if "nutrition_data" not in st.session_state:
    st.session_state.nutrition_data = []

# Input form
food = st.text_input("Food")
calories = st.number_input("Calories", min_value=0, step=1)

if st.button("Add Entry"):
    if food.strip() and calories is not None:
        st.session_state.nutrition_data.append({
            "Food": food.strip(),
            "Calories": int(calories)
        })
        st.success(f"Added {food} ({calories} cal)")
    else:
        st.warning("Please enter both Food and Calories.")

# Filter out malformed entries before displaying
valid_entries = [
    entry for entry in st.session_state.nutrition_data
    if isinstance(entry, dict)
    and "Food" in entry and entry["Food"]
    and "Calories" in entry and isinstance(entry["Calories"], (int, float))
]

# Display editable table
if valid_entries:
    df = pd.DataFrame(valid_entries)
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # Save edited table back to session state
    if st.button("💾 Save Changes"):
        st.session_state.nutrition_data = edited_df.to_dict(orient="records")
        st.success("Food log updated!")
else:
    st.info("No valid entries yet. Add some food items!")
