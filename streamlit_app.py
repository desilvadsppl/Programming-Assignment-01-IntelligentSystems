import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import requests
import json

# Title and description
st.title("Image Classification using ResNet")
st.write("Upload an image and the app will classify it using a pre-trained ResNet model.")

# Sidebar - App Information
st.sidebar.header("About the App")
st.sidebar.write("This app uses a pre-trained ResNet model to classify uploaded images. "
                 "You can upload an image, and the model will predict the class of the image.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

if uploaded_file is not None:
    # Show progress
    status_text.text("Loading image...")
    progress_bar.progress(10)

    # Load the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Define the image transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Show progress
    status_text.text("Preparing image for classification...")
    progress_bar.progress(30)

    # Preprocess the image and convert it to a tensor
    img_tensor = preprocess(img).unsqueeze(0)

    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Show progress
    status_text.text("Classifying image...")
    progress_bar.progress(60)

    # Perform image classification
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get the predicted class
    _, predicted = outputs.max(1)
    class_index = predicted.item()

    # Load the labels for ImageNet classes
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(LABELS_URL)
    labels = response.json()

    # Function to map class index to label
    def get_class_label(index):
        return labels[str(index)][1]  # The second value is the human-readable label

    # Get the predicted label
    predicted_label = get_class_label(class_index)

    # Show progress
    status_text.text("Finalizing results...")
    progress_bar.progress(90)

    # Display the prediction
    st.write(f"Prediction: {predicted_label}")

    # Show progress completion
    progress_bar.progress(100)
    status_text.text("Classification complete!")

    # Visualize results (Top 5 Predictions)
    st.write("### Visualization of Model Confidence")
    topk = torch.topk(outputs, 5)
    topk_indices = topk.indices[0].tolist()
    topk_scores = topk.values[0].tolist()

    fig, ax = plt.subplots()
    top_labels = [get_class_label(idx) for idx in topk_indices]
    ax.barh(top_labels, topk_scores)
    ax.invert_yaxis()
    st.pyplot(fig)

    # Sidebar Info: Top 5 Predictions
    st.sidebar.subheader("Top Predicted Classes")
    for i in range(5):
        st.sidebar.write(f"{i + 1}: {top_labels[i]} ({topk_scores[i]:.4f})")

else:
    st.write("Please upload an image to proceed.")

# Footer
st.sidebar.markdown("Created by: **D.S.P. Pubuditha Lakshan De Silva**")