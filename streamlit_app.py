import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Sidebar for user input
st.sidebar.title("Iris Flower Classification")
st.sidebar.subheader("User Inputs")

# Slider for sepal length
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
# Slider for sepal width
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
# Slider for petal length
petal_length = st.sidebar.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
# Slider for petal width
petal_width = st.sidebar.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Create a DataFrame from user inputs
user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")

# Make predictions
if st.sidebar.button("Predict"):
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)
    
    # Display prediction results
    st.subheader("Prediction")
    st.write(f"The predicted class is: {target_names[prediction][0]}")
    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame(prediction_proba, columns=target_names))

    # Display confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    st.subheader("Confusion Matrix")
    st.write(cm)

    # Display classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, model.predict(X_test), target_names=target_names))

# Display media upload option
st.sidebar.subheader("Upload Media")
uploaded_file = st.sidebar.file_uploader("Upload an image, video, or audio file", type=["jpg", "png", "mp4", "wav"])
if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
    elif uploaded_file.type.startswith("audio"):
        st.audio(uploaded_file)

# Display graphs
st.subheader("Data Visualization")
st.write("Iris Dataset Scatter Plot")
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Sepal Length vs Sepal Width")
st.pyplot(plt)

# Progress and Status Updates
st.sidebar.subheader("Application Status")
st.sidebar.info("Model has been trained." if 'model' in locals() else "Model is not yet trained.")

# GitHub link
st.sidebar.subheader("GitHub Repository")
st.sidebar.markdown("[View on GitHub](https://github.com/desilvadsppl/Programming-Assignment-01-IntelligentSystems.git)")

# Instructions for deployment
st.sidebar.subheader("Deployment")
st.sidebar.info("Deploy your Streamlit app on Streamlit Cloud.")