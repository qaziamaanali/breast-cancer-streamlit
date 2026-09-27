import streamlit as st
import pickle
import pandas as pd


# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Page settings
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️"
)


# Title
st.title("🎗️ Breast Cancer Prediction")

st.write(
    "This application uses a Logistic Regression model "
    "to predict whether a tumor is benign or malignant "
    "using the radius_mean feature."
)


# Input
radius_mean = st.number_input(
    "Enter Radius Mean",
    min_value=0.0,
    max_value=50.0,
    value=14.0,
    step=0.1
)


# Prediction
if st.button("Predict"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[radius_mean]],
        columns=["radius_mean"]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get malignant probability
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Malignant")
    else:
        st.success("✅ Prediction: Benign")

    st.write(
        f"Malignant Probability: {probability * 100:.2f}%"
    )

    st.progress(float(probability))


st.divider()

st.caption(
    "Educational machine-learning demonstration only. "
    "This is not a medical diagnostic tool."
)