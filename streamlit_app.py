pip install -r requirements.txt
import streamlit as st
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Define path to the Keras model file
MODEL_FILE = "keras_model.h5"

# Load the Keras model
model = load_model(MODEL_FILE)

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to RGB if it has four channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    # Resize image to match model input shape
    image = image.resize((224, 224))
    # Convert image to numpy array and normalize pixel values
    image_array = np.array(image) / 255.0
    # Expand dimensions to create a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to predict eye disease
def predict_eye_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Perform inference
    predictions = model.predict(processed_image)
    return predictions

# Streamlit app
def main():
    st.title("Eye Disease Classification")
    st.write("Upload an image of an eye to get a prediction of the eye condition.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict eye disease
        st.write("Classifying...")
        predictions = predict_eye_disease(image)

        # Display raw predictions
        st.write("Raw Predictions:", predictions)

        # Convert predictions to class labels
        class_labels = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal Eye"]
        predicted_class = class_labels[np.argmax(predictions)]

        # Display predicted class
        st.write("Predicted Eye Disease:", predicted_class)

if __name__ == "__main__":
    main()
