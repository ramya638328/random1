import streamlit as st
import pandas as pd
import pickle

st.title("Random Forest Prediction App")

# ===== Upload trained model =====
st.subheader("Step 1: Upload your trained model (.pkl)")
model_file = st.file_uploader("Choose a .pkl file", type=["pkl"])
rf_model = None
if model_file is not None:
    try:
        rf_model = pickle.load(model_file)
        if not hasattr(rf_model, "predict"):
            st.error("Uploaded file is not a trained model. Please upload a valid .pkl model.")
            rf_model = None
        else:
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# ===== Optional dataset upload =====
st.subheader("Step 2: Upload dataset (.csv) - optional")
data_file = st.file_uploader("Choose a CSV file", type=["csv"])
df = None
if data_file is not None:
    try:
        df = pd.read_csv(data_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ===== Input features and prediction =====
if rf_model is not None:
    st.subheader("Step 3: Enter Input Features")

    # Replace these with your actual feature names and types
    feature1 = st.number_input("Distance (km)", min_value=0.0)
    feature2 = st.number_input("Preparation Time (min)", min_value=0.0)
    feature3 = st.selectbox("Traffic Level", options=[0, 1, 2])

    input_data = pd.DataFrame([[feature1, feature2, feature3]],
                              columns=['Distance_km', 'Preparation_Time_min', 'Traffic_Level'])

    if st.button("Predict"):
        try:
            prediction = rf_model.predict(input_data)[0]
            st.success(f"Predicted Output: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===== Display charts if dataset is uploaded =====
if df is not None:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Data Chart")
        st.line_chart(df[numeric_cols])
