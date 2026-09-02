import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("decision_tree_model.pkl", "rb") as file:
    model = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# Title
st.title("🤖 Machine Learning Prediction App")
st.write("Enter the required values below to get a prediction.")

# Input fields
feature_1 = st.number_input("Load_Current", value=0.0)
feature_2 = st.number_input("Power_Factor", value=0.0)
feature_3 = st.number_input("Pf_error", value=0.0)

# Prediction button
if st.button("Predict"):
    
    # Arrange inputs in the same order used during model training
    input_data = np.array([
        feature_1,
        feature_2,
        feature_3
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Value: {prediction[0]:.2f}")