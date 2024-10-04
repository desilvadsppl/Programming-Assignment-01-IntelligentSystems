import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import time

# Page Title
st.title("Streamlit ML Application - Iris Flower Classification")

# 1. Sidebar for User Inputs
# ---------------------------
st.sidebar.header("User Input Parameters")


def user_input_features():
        # Adding sliders in the sidebar for input parameters
        sepal_length = st.sidebar.slider('Sepal length (cm)', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)

        # Collecting the user input into a dataframe
        data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features


# Getting user input
df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(df)

# 2. File Upload Feature (Optional)
# ---------------------------------
st.subheader("Upload and Display Image")
uploaded_file = st.file_uploader("Choose an image...",
                                 type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file,
                 caption='Uploaded Image.',
                 use_column_width=True)
        st.write("Image Uploaded Successfully!")

# 3. Display Progress Bar and Status Update
# -----------------------------------------
# Simulate a long computation
progress_bar = st.progress(0)
status_text = st.empty()  # Placeholder for status message

# Simulate a process with status updates
for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Processing... {i + 1}% complete")
        time.sleep(0.02)  # Simulate computation time

st.success("Process Completed!")

# 4. Load Iris Dataset and Train Model
# ------------------------------------
iris = load_iris()
X = iris.data
Y = iris.target

# Train a RandomForest Classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Make predictions based on user input
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# 5. Display Model Predictions
# ----------------------------
# Display the prediction and prediction probability
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(f"The predicted class is: **{iris.target_names[prediction][0]}**")

st.subheader('Prediction Probability')
st.write(prediction_proba)

# 6. Graphical Visualization of the Dataset
# -----------------------------------------
st.subheader('Visualization of the Iris Dataset')

# Scatter plot of the dataset
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_title("Sepal Length vs Sepal Width")
plt.colorbar(scatter, ax=ax, label='Classes')

# Display the scatter plot in Streamlit
st.pyplot(fig)

# Display a histogram of the class distribution
st.subheader('Histogram of the Iris Dataset')
fig, ax = plt.subplots()
ax.hist(Y, bins=3, edgecolor='k')
ax.set_xlabel("Classes")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of Class Distribution in Iris Dataset")

# Display the histogram in Streamlit
st.pyplot(fig)

# 7. Button Interaction Example
# -----------------------------
# Add a button for additional interactivity
if st.button("Show Dataset Description"):
        st.subheader("Iris Dataset Description")
        st.write(iris.DESCR)

# 8. Organizing Layout with Containers
# ------------------------------------
# Create a container for displaying information about the model
with st.container():
        st.write("### Random Forest Classifier Information")
        st.write(
            "The Random Forest classifier was trained on the Iris dataset to classify different types of flowers based on their features."
        )