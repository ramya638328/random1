import streamlit as st
import pickle
import pandas as pd

st.title("ML Prediction App")

# Step 1: Upload model
model_file = st.file_uploader(
    "Step 1: Upload your trained model (.pkl)",
    type=["pkl"]
)

# Step 2: Upload dataset (optional)
data_file = st.file_uploader(
    "Step 2: Upload dataset (.csv) - optional",
    type=["csv"]
)

# Load model
if model_file is not None:
    model = pickle.load(model_file)
    st.success("Model loaded successfully")

# Load dataset
if data_file is not None:
    df = pd.read_csv(data_file)
    st.success("Dataset loaded successfully")
    st.dataframe(df.head())
