# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import time
from PIL import Image

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train a model with hyperparameter tuning
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, 'iris_model_svm.pkl')
    return accuracy

# Check if the model is already trained
try:
    model = joblib.load('iris_model_svm.pkl')
except:
    accuracy = train_model()
    model = joblib.load('iris_model_svm.pkl')
    st.sidebar.success(f'Model trained with accuracy: {accuracy:.2f}')

# Title of the app
st.title("Iris Flower Species Classification")

# Sidebar for user input
st.sidebar.header("User Input Features")

# User input for feature selection
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Media upload section
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload an image of an Iris flower (optional)", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.sidebar.image(img, caption='Uploaded Image', use_column_width=True)

# Prepare input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Display input data for debugging
st.write("Input Data for Prediction:")
st.write(pd.DataFrame(input_data, columns=feature_names))

# Progress indicator
st.sidebar.text("Making predictions...")
progress_bar = st.sidebar.progress(0)

for percent_complete in range(100):
    time.sleep(0.02)  # Simulate a processing delay
    progress_bar.progress(percent_complete + 1)

# Classify the input data
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.subheader("Prediction:")
st.write(f"The predicted species is: **{target_names[prediction[0]]}**")

# Display prediction probabilities
st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(proba_df)

# Display the dataset
st.subheader("Iris Dataset")
st.dataframe(pd.DataFrame(data=iris.data, columns=iris.feature_names))

# Footer message
st.sidebar.text("Built with Streamlit")