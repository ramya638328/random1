import streamlit as st
import pickle
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    with open("random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    return model

rf_model = load_model()

# Streamlit UI
st.title("🎓 Student Grade Prediction")

Std_Branch = st.number_input("Student Branch (encoded)", min_value=0, max_value=10)
Std_Course = st.number_input("Student Course (encoded)", min_value=0, max_value=10)
Std_Marks = st.number_input("Student Marks", min_value=0, max_value=100)

if st.button("Predict Grade"):
    input_data = np.array([[Std_Branch, Std_Course, Std_Marks]])
    prediction = rf_model.predict(input_data)[0]

    grade_map = {0: "A", 1: "B", 2: "C"}
    predicted_grade = grade_map.get(prediction, "Unknown")

    st.success(f"Predicted Grade: {predicted_grade}")



