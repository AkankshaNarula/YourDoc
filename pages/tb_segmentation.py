import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import base64
from io import BytesIO
import os
import tempfile
from models.tuberculosis import detect_tuberculosis

# Custom CSS to make the sidebar fixed and always expanded
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
    
    /* Custom styles as provided */
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
    .result-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .priority-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .priority-medium {
        color: #FB8C00;
        font-weight: bold;
    }
    .priority-low {
        color: #388E3C;
        font-weight: bold;
    }
    .calendar-event {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #1E88E5;
    }
    .feature-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .feature-title {
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 5px;
    }
    .feature-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .feature-description {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫁 Tuberculosis Detection System")
st.markdown("""
This tool detects tuberculosis from chest X-ray images using a DeepLabV3 segmentation model.
Upload an X-ray image and enter patient details to get a segmentation mask and lung analysis.
""")

# Sidebar for patient information
st.sidebar.markdown("<div class='sub-header'>Patient Information</div>", unsafe_allow_html=True)

patient_name = st.sidebar.text_input("Patient Name", "John Doe", key="tb_patient_name")
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
- DeepLabV3 model for tuberculosis detection
    
Make sure the model files are in the same directory:
- saved_models/_best_model.pt
""")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def display_feature(title, value, description, unit="", formatter=lambda x: f"{x:.2f}"):
    """Helper function to display a feature card"""
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-title">{title}</div>
        <div class="feature-value">{formatter(value)}{unit}</div>
        <div class="feature-description">{description}</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    temp_image_path = "temp_xray.png"
    image.save(temp_image_path)
    
    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Processing image..."):
            # Call the enhanced detection function from models.tuberculosis
            result = detect_tuberculosis(temp_image_path, age, sex, position)
            
            if not result:
                st.error("Error processing the image. Please check if the model exists.")
            else:
                mask_path = result["mask_path"]
                features = result["features"]
                
                # Display images: original, mask, and overlay
                mask_image = Image.open(mask_path)
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["Original vs Mask", "Lung Analysis", "Features for AI"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original X-ray", use_column_width=True)
                    with col2:
                        # Create overlay
                        image_np = np.array(image.convert("L"))
                        mask_np = np.array(mask_image)
                        
                        # Resize mask if needed
                        if mask_np.shape != image_np.shape:
                            mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(image_np, cmap="gray")
                        ax.imshow(mask_np, cmap="jet", alpha=0.5)
                        ax.set_title("Lung Segmentation Overlay")
                        ax.axis("off")
                        st.pyplot(fig)
                
                with tab2:
                    st.markdown("<div class='sub-header'>Lung Analysis</div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_feature(
                            "Lung Area Ratio", 
                            features["lung_area_ratio"] * 100, 
                            "Percentage of image area occupied by lungs",
                            unit="%",
                            formatter=lambda x: f"{x:.1f}"
                        )
                        
                        display_feature(
                            "Mean Intensity", 
                            features["mean_intensity"], 
                            "Average brightness within lung regions (0-255)"
                        )
                        
                        display_feature(
                            "Standard Deviation", 
                            features["std_intensity"], 
                            "Variation in lung intensity (texture heterogeneity)"
                        )
                        
                        display_feature(
                            "Opacity Score", 
                            features["opacity_score"], 
                            "Overall opacity level (higher values suggest more consolidation)",
                            formatter=lambda x: f"{x:.3f}"
                        )
                    
                    with col2:
                        display_feature(
                            "Left-to-Right Ratio", 
                            features["left_to_right_ratio"], 
                            "Ratio of left lung area to right lung area",
                            formatter=lambda x: f"{x:.3f}"
                        )
                        
                        display_feature(
                            "Min Intensity", 
                            features["min_intensity"], 
                            "Darkest pixel in lung regions (0-255)",
                            formatter=lambda x: f"{int(x)}"
                        )
                        
                        display_feature(
                            "Max Intensity", 
                            features["max_intensity"], 
                            "Brightest pixel in lung regions (0-255)",
                            formatter=lambda x: f"{int(x)}"
                        )
                        
                        display_feature(
                            "Lung Area", 
                            features["lung_area_pixels"], 
                            "Total lung area in pixels",
                            formatter=lambda x: f"{int(x):,}"
                        )
                
                with tab3:
                    st.markdown("<div class='sub-header'>AI-Ready Data</div>", unsafe_allow_html=True)
                    st.info("This data is formatted for sending to an AI model for further analysis")
                    
                    # Display the patient info and features
                    ai_data = {
                        "patient": {
                            "name": patient_name,
                            "age": age,
                            "sex": sex_choice,
                            "position": position_choice,
                            "symptoms": symptoms
                        },
                        "features": features
                    }
                    
                    # Show the data structure without the base64 images (to avoid cluttering the UI)
                    st.json(ai_data)
                    
                    st.markdown("**Note:** The full data sent to AI also includes base64-encoded images of the original X-ray, segmentation mask, and overlay visualization.")
                
                st.success("✅ Analysis completed!")
                
                # Clean up temporary files
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
