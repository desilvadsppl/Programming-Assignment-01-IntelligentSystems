import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests
import json
import matplotlib.pyplot as plt

# Title and description
st.title("Image Classification using ResNet")
st.write("Upload an image and classify it using a pre-trained ResNet model.")

# Sidebar - App Information and Settings
with st.sidebar:
    st.header("About the App")
    st.write("This app uses a pre-trained ResNet model to classify uploaded images into one of the 1,000 ImageNet categories.")
    st.write("Upload an image, then press the 'Classify Image' button to get the predicted class.")
    st.write("You will also see the top 5 predicted classes and a confidence bar chart.")
    st.write("Developed using PyTorch and Streamlit.")
    st.write('''
Common Image Categories to Test:

**Animals**:
- Dogs (e.g., golden retriever, border collie)
- Cats (e.g., Siamese cat, tabby)
- Birds (e.g., flamingo, parrot, hawk)
- Fish (e.g., goldfish, shark)

**Objects**:
- Everyday items like chairs, laptops, coffee mugs, and bottles.
- Vehicles such as cars, bicycles, buses, or airplanes.
- Sports equipment like basketballs, tennis rackets, or skis.

**Scenes**:
- Natural landscapes like mountains, beaches, or forests.
- Urban settings with buildings, bridges, or streets.

**Foods**:
- Fruits (e.g., apple, banana, orange)
- Dishes (e.g., pizza, hamburger, sushi)

**People**:
- Images of human activities (e.g., person riding a horse, person playing soccer).

**Furniture and Household Items**:
- Sofas, tables, lamps, and other common home items.

**General Guidelines**:
- High-Quality, Clear Images: Make sure the images are clear and recognizable by the model.
- Single Object Focus: It's ideal if the image focuses on a single object for better classification accuracy.
- Real-World Images: You can use photos taken with your camera or images from the internet that match these categories.
''')
    confidence_threshold = st.slider("Confidence Threshold (for top classes)", 0.0, 1.0, 0.5, 0.1)  # Optional feature for future use

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Button to trigger image classification
classify_button = st.button("Classify Image")

# Progress bar and status update
progress_bar = st.progress(0)
status_text = st.empty()

if uploaded_file is not None and classify_button:
    # Containers for better layout
    with st.container():
        # Show progress
        status_text.text("Loading image...")
        progress_bar.progress(10)

        # Load and display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Define image transformations for ResNet
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

        # Load the labels for ImageNet classes from a public URL
        LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        response = requests.get(LABELS_URL)
        labels = response.json()

        # Function to get the class label from the predicted index
        def get_class_label(index):
            return labels[str(index)][1]  # The second value is the human-readable label

        # Get the human-readable class label for the prediction
        class_label = get_class_label(class_index)

        # Show progress
        status_text.text("Finalizing results...")
        progress_bar.progress(90)

        # Display the prediction
        st.write(f"Prediction: **{class_label}**")

        # Show progress completion
        progress_bar.progress(100)
        status_text.text("Classification complete!")

        # Visualization of the top 5 predictions
        st.write("### Visualization of Model Confidence")
        fig, ax = plt.subplots()

        # Top 5 predictions
        top_k = torch.topk(outputs, 5).indices.squeeze(0).tolist()
        top_labels = [get_class_label(i) for i in top_k]
        top_values = [outputs[0, i].item() for i in top_k]

        # Bar chart for top 5 predicted classes
        ax.barh(top_labels, top_values)
        ax.invert_yaxis()  # Highest probability at the top
        st.pyplot(fig)

        # Sidebar Info - Top Predicted Classes
        st.sidebar.subheader("Top Predicted Classes")
        for i in range(5):
            st.sidebar.write(f"{i + 1}: {get_class_label(top_k[i])} with confidence {top_values[i]:.4f}")

else:
    st.write("Please upload an image and press the 'Classify Image' button to proceed.")

# Footer in sidebar
st.sidebar.markdown("Created by: **D.S.P.Pubuditha Lakshan De Silva (ITBIN-2110-0020)**")