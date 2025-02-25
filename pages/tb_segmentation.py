import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def detect_tuberculosis(image_path, age, sex, position, device=None):
    model_path = '/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/saved_models/_best_model.pt'
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

st.title("🫁 Tuberculosis Detection System")
st.markdown("""
This tool detects tuberculosis from chest X-ray images using a DeepLabV3 segmentation model.
Upload an X-ray image and enter patient details to get a segmentation mask.
""")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

age = st.slider("Patient Age", min_value=0, max_value=100, value=50)
sex = st.radio("Sex", ["Male", "Female"])
position = st.radio("Position", ["AP (Anterior-Posterior)", "PA (Posterior-Anterior)"])

sex_mapping = {"Male": 0, "Female": 1}
position_mapping = {"AP (Anterior-Posterior)": 0, "PA (Posterior-Anterior)": 1}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    temp_image_path = "temp_xray.png"
    image.save(temp_image_path)
    
    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Processing image..."):
            result = detect_tuberculosis(temp_image_path, age, sex_mapping[sex], position_mapping[position])
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