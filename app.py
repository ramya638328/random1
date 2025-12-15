import streamlit as st
import pandas as pd
import pickle

st.title("Random Forest Prediction App")

# Upload trained model
model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
rf_model = None
if model_file is not None:
    try:
        rf_model = pickle.load(model_file)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Upload dataset (optional, if needed)
data_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
df = None
if data_file is not None:
    try:
        df = pd.read_csv(data_file)
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Only show inputs if model is loaded
if rf_model is not None:
    st.subheader("Enter Input Features")

    # Example: Adjust these inputs according to your dataset/features
    feature1 = st.number_input("Feature 1 (e.g., Distance_km)", min_value=0.0)
    feature2 = st.number_input("Feature 2 (e.g., Preparation_Time_min)", min_value=0.0)
    feature3 = st.selectbox("Feature 3 (e.g., Traffic_Level)", options=[0,1,2])  # adjust categories

    input_data = pd.DataFrame([[feature1, feature2, feature3]],
                              columns=['Distance_km','Preparation_Time_min','Traffic_Level'])

    if st.button("Predict"):
        try:
            prediction = rf_model.predict(input_data)[0]
            st.success(f"Predicted Output: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Optional: display uploaded data charts
if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head())
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.line_chart(df[numeric_cols])
