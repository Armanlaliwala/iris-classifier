import streamlit as st
import pickle
import numpy as np

# Load saved models
models = {
    "Logistic Regression": "lr_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "KNN": "knn_model.pkl",
    "SVM": "svm_model.pkl",
    "XGBoost": "xgb_model.pkl"
}

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.subheader("Created by Arman Laliwala 🚀")

# Dropdown to select model
selected_model = st.selectbox("Select a Model", list(models.keys()))

# Load the chosen model
with open(models[selected_model], "rb") as file:
    model = pickle.load(file)

st.sidebar.header("Enter Flower Measurements")

# User input fields
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.2)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.5)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3)

# Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    
    # Mapping prediction to class names
    class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    st.success(f"🌿 The predicted flower species is **{class_names[prediction]}** using **{selected_model}** model!")
