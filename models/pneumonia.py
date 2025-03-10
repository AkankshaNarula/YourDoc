import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import tempfile
import gdown

# Define custom loss functions used in model training
def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1.0 - (intersection + 1e-7) / (union + 1e-7)

def loss_fn(y_true, y_pred):
    return iou_loss(y_true, y_pred)

custom_objects = {"iou_loss": iou_loss, "loss_fn": loss_fn}

# Function to normalize age (MinMax Scaling as in training)
def normalize_age(age, min_age=0, max_age=100):
    return (age - min_age) / (max_age - min_age)

# Function to preprocess input image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def detect_pneumonia(image_path, age, sex, position):
    """
    Detect pneumonia in a chest X-ray image.
    
    Args:
        image_path: Path to the chest X-ray image.
        age: Patient's age.
        sex: Patient's sex (0 for Male, 1 for Female).
        position: Image position (0 for AP, 1 for PA).
    
    Returns:
        Dict containing segmentation mask path, probability, and original image path.
    """
    
    # Google Drive URL for model download
    MODEL_URL = "https://drive.google.com/uc?id=1LW5vrzUQpCTZL7oSFS1xQ8sIw4aYGO9o"
    model_path = "model.keras"

    # Download model if not present
    if not os.path.exists(model_path):
        st.info("Downloading pneumonia model... Please wait.")
        gdown.download(MODEL_URL, model_path, quiet=False)
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Pneumonia model not found at {model_path}. Please ensure the model file exists.")
        return None
    
    # Load pneumonia model
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load and preprocess the chest X-ray image
    xray_image = Image.open(image_path).convert("L")  # Convert to grayscale
    xray_image_np = np.array(xray_image)
    xray_image_np = cv2.cvtColor(xray_image_np, cv2.COLOR_GRAY2RGB)  # Ensure 3 channels
    
    # Preprocess image
    preprocessed_image = preprocess_image(xray_image_np)
    normalized_age = normalize_age(age)
    
    metadata = np.array([[normalized_age, sex, position]], dtype=np.float32)
    
    # Run prediction
    prediction = model.predict([preprocessed_image, metadata])
    
    mask = prediction[0][0, :, :, 0]  # Extracting mask
    pneumonia_prob = prediction[1][0, 0]  # Extracting probability
    
    # Save the segmentation mask as a PNG file
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    
    # Create a temporary file for the mask
    temp_mask_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    mask_filename = temp_mask_file.name
    mask_image.save(mask_filename)
    
    return {
        "mask_path": mask_filename,
        "probability": float(pneumonia_prob),
        "original_image": image_path
    }