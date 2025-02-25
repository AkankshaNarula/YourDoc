import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained UNet hybrid model
MODEL_PATH = "/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/saved_models/model_unet.keras"

# Define custom loss functions used in training
def iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return 1.0 - (intersection + 1e-7) / (union + 1e-7)

def loss_fn(y_true, y_pred):
    return iou_loss(y_true, y_pred)

custom_objects = {"iou_loss": iou_loss, "loss_fn": loss_fn}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

# Function to normalize age (MinMax Scaling as in training)
def normalize_age(age, min_age=0, max_age=100):
    return (age - min_age) / (max_age - min_age)

# Function to preprocess input image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict mask and pneumonia classification
def predict_pneumonia(image, age, sex, position):
    preprocessed_image = preprocess_image(image)
    normalized_age = normalize_age(age)
    
    metadata = np.array([[normalized_age, sex, position]], dtype=np.float32)
    
    prediction = model.predict([preprocessed_image, metadata])

    mask = prediction[0][0, :, :, 0]  # Extracting mask
    pneumonia_prob = prediction[1][0, 0]  # Extracting probability
    return mask, pneumonia_prob

# Streamlit UI
st.title("🫁 Pneumonia Detection System")
st.markdown("""
This tool detects pneumonia from chest X-ray images using a hybrid UNet model.
Upload an X-ray image and enter patient details to get a segmentation mask and classification result.
""")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Metadata Input
age = st.slider("Patient Age", min_value=0, max_value=100, value=50)
sex = st.radio("Sex", ["Male", "Female"])
position = st.radio("Position", ["AP (Anterior-Posterior)", "PA (Posterior-Anterior)"])

# Mapping for model input
sex_mapping = {"Male": 0, "Female": 1}
position_mapping = {"AP (Anterior-Posterior)": 0, "PA (Posterior-Anterior)": 1}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Convert image to NumPy array
    image_np = np.array(image.convert("L"))  # Convert to grayscale
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)  # Ensure 3 channels

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Processing image..."):
            mask, pneumonia_prob = predict_pneumonia(image_np, age, sex_mapping[sex], position_mapping[position])

        # Display results
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_np, cmap="gray")
        ax[0].set_title("Original X-ray")
        ax[0].axis("off")

        ax[1].imshow(image_np, cmap="gray")
        ax[1].imshow(mask, cmap="jet", alpha=0.5)  # Overlay mask
        ax[1].set_title("Predicted Mask")
        ax[1].axis("off")

        st.pyplot(fig)

        # Diagnosis Result
        st.subheader("Diagnosis Result")
        probability_percentage = pneumonia_prob * 100  # Convert to percentage
        if pneumonia_prob > 0.5:
            st.error(f"⚠️ **Pneumonia Detected!** (Probability: {probability_percentage:.2f}%)")
        else:
            st.success(f"✅ **No Pneumonia Detected.** (Probability: {probability_percentage:.2f}%)")
