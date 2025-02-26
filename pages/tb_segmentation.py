import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import gdown
import os

def detect_tuberculosis(image_path, age, sex, position, device=None):
 
    # Google Drive URL for model download
    MODEL_URL = "https://drive.google.com/file/d/1AkpP6LV7WPl4Es9Axuc2b-CZzC5LSYlv/view?usp=sharing"
    model_path = "_best_model.pt"
    
    # Download model if not present
    if not os.path.exists(model_path):
        st.info("Downloading model... Please wait.")
        gdown.download(MODEL_URL, model_path, quiet=False)

    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            output = model(image)
        except TypeError:
            metadata = torch.tensor([[age / 100.0, sex, position]], dtype=torch.float32).to(device)
            output = model(image, metadata)
        
        mask = output.squeeze().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask * 255).astype(np.uint8)
    
    mask_image = Image.fromarray(mask)
    mask_filename = "tb_mask.png"
    mask_image.save(mask_filename)
    
    return {
        "mask_path": mask_filename,
        "original_image": image_path
    }

# Streamlit UI
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
</style>
""", unsafe_allow_html=True)
st.title("🫁 Tuberculosis Detection System")
st.markdown("""
This tool detects tuberculosis from chest X-ray images using a DeepLabV3 segmentation model.
Upload an X-ray image and enter patient details to get a segmentation mask.
""")


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
- DeepLabV3 model for tuberculosis detection
    
Make sure the model files are in the same directory:
- _best_model.pt
""")

# Image Upload
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    temp_image_path = "temp_xray.png"
    image.save(temp_image_path)
    
    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Processing image..."):
            result = detect_tuberculosis(temp_image_path, age, sex, position)
            mask_path = result["mask_path"]
        
        mask_image = Image.open(mask_path)
        image_np = np.array(image.convert("L"))
        mask_np = np.array(mask_image)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_np, cmap="gray")
        ax[0].set_title("Original X-ray")
        ax[0].axis("off")
        
        ax[1].imshow(image_np, cmap="gray")
        ax[1].imshow(mask_np, cmap="jet", alpha=0.5)
        ax[1].set_title("Predicted Mask")
        ax[1].axis("off")
        
        st.pyplot(fig)
        st.success("✔️ Segmentation completed.")
