import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the updated trained model
MODEL_PATH = "saved_models/updated_model_unet.keras"

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

# Custom CSS styling (remains the same)
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 250px;
        max-width: 250px;
        position: fixed;
    }
    section[data-testid="stSidebarContent"] {
        background-color: #f0f2f6;
        padding-top: 2rem;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1565C0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for patient information
st.sidebar.markdown("<div class='sub-header'>Patient Information</div>", unsafe_allow_html=True)

patient_name = st.sidebar.text_input("Patient Name", "John Doe")
age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=45)

sex_options = {"Male": 0, "Female": 1}
sex_choice = st.sidebar.radio("Sex", list(sex_options.keys()))
sex = sex_options[sex_choice]

position_options = {"Anterior-Posterior (AP)": 0, "Posterior-Anterior (PA)": 1}
position_choice = st.sidebar.radio("X-ray Position", list(position_options.keys()))
position = position_options[position_choice]

# Symptoms input
symptoms = st.sidebar.text_area("Symptoms", 
                               placeholder="Enter patient's symptoms...",
                               value="Cough, fever, shortness of breath")

# Model information
st.sidebar.markdown("<div class='sub-header'>Model Information</div>", unsafe_allow_html=True)
st.sidebar.info("""
This application uses:
- Updated UNet model for pneumonia detection
- DeepLabV3 model for tuberculosis detection
- Gemini LLM for analysis and scheduling

Make sure the model files are in the same directory:
- updated_model_unet.keras
- _best_model.pt
""")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Convert image to NumPy array
    image_np = np.array(image.convert("L"))  # Convert to grayscale
    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)  # Ensure 3 channels

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Processing image..."):
            mask, pneumonia_prob = predict_pneumonia(image_np, age, sex, position)

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