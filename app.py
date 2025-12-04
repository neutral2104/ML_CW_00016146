import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

@st.cache_data
def load_artifacts():
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        with open("feature_cols.json", "r") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except:
        return None, None, None

model, scaler, feature_cols = load_artifacts()

st.set_page_config(page_title="Happiness Rank Predictor", layout="centered")
st.title("World Happiness — Rank Predictor (2023)")

st.markdown("""
This app predicts a country's **happiness rank** using the Gradient Boosting model
trained in the coursework. Enter feature values and press **Predict**.
""")

if model is None:
    st.error("Model, scaler or feature columns file missing. Please rerun notebook.")
    st.stop()

st.subheader("Enter feature values:")
inputs = {}
cols = st.columns(2)

for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        inputs[col] = st.number_input(col, value=0.0, format="%.3f")

input_df = pd.DataFrame([inputs], columns=feature_cols)
scaled_input = scaler.transform(input_df.values)

if st.button("Predict Rank"):
    pred = model.predict(scaled_input)[0]
    st.success(f"Predicted Happiness Rank: **{pred:.2f}**")
