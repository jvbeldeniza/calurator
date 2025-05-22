# import streamlit as st
# import pandas as pd

# st.title("📝 Log Food")

# if 'food_log' not in st.session_state:
#     st.session_state.food_log = []

# food = st.text_input("Food Item")
# calories = st.number_input("Calories", min_value=0)

# if st.button("Add Entry"):
#     if food and calories:
#         st.session_state.food_log.append({"Food": food, "Calories": calories})
#         st.success(f"Added {food} ({calories} cal)")

# df = pd.DataFrame(st.session_state.food_log)
# st.dataframe(df)

import streamlit as st
import pandas as pd

st.title("📝 Log Food")

# Initialize the session state log
if 'food_log' not in st.session_state:
    st.session_state.food_log = []

# Input form
food = st.text_input("Food Item")
calories = st.number_input("Calories", min_value=0)

if st.button("Add Entry"):
    if food and calories:
        st.session_state.food_log.append({"Food": food, "Calories": calories})
        st.success(f"Added {food} ({calories} cal)")

# Create a DataFrame from the session log
df = pd.DataFrame(st.session_state.food_log)

# Editable table
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# Button to update session state
if st.button("💾 Save Changes"):
    st.session_state.food_log = edited_df.to_dict(orient="records")
    st.success("Food log updated!")