# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:04:51 2024

@author: jacob
"""

    
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import cv2
import tempfile

# Load the trained SVM model
best_svm_model = joblib.load("test_model3.pkl")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")  

# Function to preprocess an image before prediction
def preprocess_uploaded_image(file_path, target_size=(28, 28), lower_pixel=110, upper_pixel=235):
    # Read image
    test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise FileNotFoundError("Image not found or unable to read")

    # Resize image
    img_resized = cv2.resize(test_image, target_size, interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)

    # Thresholding
    img_processed = np.where((img_resized <= lower_pixel), 0, img_resized)
    img_processed = np.where((img_processed > upper_pixel), 255, img_processed)

    return img_processed

def main():
    st.title("Handwritten Digit Recognition")

    st.write("Upload an image of a handwritten digit (black on white background) for prediction.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Display the uploaded image
        uploaded_image = Image.open(tmp_file_path)
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_uploaded_image(tmp_file_path)

        # Display the processed image
        st.image(processed_image, caption='Processed Image', use_column_width=True)

        # Reshape image for prediction
        processed_image_flat = processed_image.reshape(-1, 784)

        # Make prediction
        prediction = best_svm_model.predict(processed_image_flat)
        st.write(f"Predicted digit: {prediction[0]}")

if __name__ == "__main__":
    main()