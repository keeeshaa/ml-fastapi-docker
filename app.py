import os

import requests
import streamlit as st

# Address of the FastAPI service
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Iris Flower Classifier")
st.write("Enter flower measurements (in cm) and get a prediction from the ML API.")

sepal_length = st.number_input("Sepal length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        result = response.json()
        st.success(f"Predicted species: {result['prediction']}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure FastAPI is running.")
