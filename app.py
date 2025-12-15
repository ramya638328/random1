import streamlit as st
import pickle
import numpy as np

# --------------------------------
# Load trained Random Forest model
# --------------------------------
@st.cache_resource
def load_model():
    with open("random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    return model

rf_model = load_model()

# --------------------------------
# Streamlit UI
# --------------------------------
st.title("🎓 Student Grade Prediction App")
st.write("Predict student grade using Random Forest model")

# --------------------------------
# User Inputs
# --------------------------------
st.header("Enter Student Details")

# NOTE:
# These values MUST match LabelEncoder logic used in training

Std_Branch = st.selectbox(
    "Student Branch",
    options=[0, 1, 2, 3],
    help="Encoded values used during training"
)

Std_Course = st.selectbox(
    "Student Course",
    options=[0, 1, 2, 3],
    help="Encoded values used during training"
)

Std_Marks = st.number_input(
    "Student Marks",
    min_value=0,
    max_value=100,
    step=1
)

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict Grade"):
    input_data = np.array([[Std_Branch, Std_Course, Std_Marks]])
    prediction = rf_model.predict(input_data)[0]

    # Decode prediction
    grade_map = {0: "A", 1: "B", 2: "C"}
    predicted_grade = grade_map.get(prediction, "Unknown")

    st.success(f"🎯 Predicted Grade: **{predicted_grade}**")
