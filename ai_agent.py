import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import cv2
import os
import datetime
from io import BytesIO
import tempfile
import io
import sys

from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.file import FileTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage

# Set page config
st.set_page_config(
    page_title="Lung Disease Analysis System",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-header'>Lung Disease Analysis System</div>", unsafe_allow_html=True)
st.write("Upload a chest X-ray image and get AI-powered analysis for pneumonia and tuberculosis")

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
    model_path = '/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/saved_models/model_unet.keras'
    
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

def detect_tuberculosis(image_path, age, sex, position, device=None):
    """
    Detect tuberculosis in a chest X-ray image.
    
    Args:
        image_path: Path to the chest X-ray image.
        age: Patient's age.
        sex: Patient's sex (0 for Male, 1 for Female).
        position: Image position (0 for AP, 1 for PA).
        device: Device to run the model on (cpu or cuda).
    
    Returns:
        Dict containing segmentation mask path, probability, and original image path.
    """
    model_path = '/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/saved_models/_best_model.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"TB model not found at {model_path}. Please ensure the model file exists.")
        return None
    
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Define preprocessing transformations
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
        transforms.Resize((256, 256)),  # Resize to model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    
    # Preprocess image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Try running prediction
    with torch.no_grad():
        try:
            output = model(image)  # Try passing only the image
        except TypeError:
            metadata = torch.tensor([[age / 100.0, sex, position]], dtype=torch.float32).to(device)
            output = model(image, metadata)  # If error, try with metadata
        
        # Ensure mask has the correct shape (remove extra dimensions)
        mask = output.squeeze().cpu().numpy()
        
        # Ensure it's a 2D array for PIL
        if mask.ndim == 3:
            mask = mask[0]  # Take first channel if needed
        
        # Normalize mask and convert to uint8 for saving
        mask = (mask * 255).astype(np.uint8)
    
    # Save the segmentation mask as a PNG file
    mask_image = Image.fromarray(mask)
    
    # Create a temporary file for the mask
    temp_mask_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    mask_filename = temp_mask_file.name
    mask_image.save(mask_filename)
    
    return {
        "mask_path": mask_filename,
        "original_image": image_path
    }

# Capture output from print_response
class CaptureOutput:
    def __init__(self):
        self.value = ""
        self._redirect_stdout()

    def _redirect_stdout(self):
        self.old_stdout = sys.stdout
        sys.stdout = self

    def write(self, string):
        self.value += string

    def flush(self):
        pass

    def reset(self):
        self.value = ""

    def restore_stdout(self):
        sys.stdout = self.old_stdout

def create_agents():
    # Initialize the Gemini model
    gemini_model = Gemini(
        id="gemini-1.5-flash",  # Specify the desired Gemini model version
        api_key=os.environ.get('GEMINI_API_KEY', 'AIzaSyArW-r0ojXCNFfOMk-lau2mg2lUTUfJvH4')  # Use environment variable if available
    )
    
    # Create specialized agents for different tasks
    # 1. Disease Detection Agent for Pneumonia
    detection_agent_1 = Agent(
        name="Lung Disease Detection Agent - Pneumonia",
        model=gemini_model,
        tools=[FileTools()],
        instructions=[
            "You are specialized in detecting lung diseases from X-ray images",
            "Provide detailed results.",
            '''Always give a response. Analyze the provided lung images and its corresponding segmentation mask. Describe any observed abnormalities, including their size, location, and characteristics. Assess the likelihood of pneumonia or other pulmonary conditions based on the visual evidence. Provide a detailed report that includes:
            
            Findings:
            
            - Lung Fields: Describe areas of increased opacity or consolidation, specifying size and location.
            - Air Bronchograms: Note the presence of air-filled bronchi outlined by surrounding consolidation.
            - Pleural Space: Assess for pleural effusion or pneumothorax.
            - Cardiomediastinal Silhouette: Evaluate heart size and mediastinal contours.
            - Bones and Soft Tissues: Inspect for abnormalities in ribs, spine, or soft tissues.
            
            Impression:
            
            - Summarize findings suggestive of pneumonia or other conditions.
            - Classify the severity of any detected infection (e.g., mild, moderate, severe).
            - Recommend further evaluation or follow-up if necessary.'''
        ],
        markdown=True,
    )
    
    # 2. Disease Detection Agent for Tuberculosis
    detection_agent_2 = Agent(
        name="Lung Disease Detection Agent - Tuberculosis",
        model=gemini_model,
        tools=[FileTools()],
        instructions=[
            "You are specialized in detecting lung diseases from X-ray images",
            "Provide detailed results.",
            '''Always give a response. Analyze the provided lung images and its corresponding segmentation mask. Describe any observed abnormalities, including their size, location, and characteristics. Assess the likelihood of tuberculosis or other pulmonary conditions based on the visual evidence. Provide a detailed report that includes:
            
            Findings:
            
            - Lung Fields: Describe areas of increased opacity or consolidation, specifying size and location.
            - Air Bronchograms: Note the presence of air-filled bronchi outlined by surrounding consolidation.
            - Pleural Space: Assess for pleural effusion or pneumothorax.
            - Cardiomediastinal Silhouette: Evaluate heart size and mediastinal contours.
            - Bones and Soft Tissues: Inspect for abnormalities in ribs, spine, or soft tissues.
            
            Impression:
            
            - Summarize findings suggestive of tuberculosis or other conditions.
            - Classify the severity of any detected infection (e.g., mild, moderate, severe).
            - Recommend further evaluation or follow-up if necessary.'''
        ],
        show_tool_calls=True,
    )
    
    # 3. Doctor Verification Agent
    verification_agent = Agent(
        name="Medical Verification Agent",
        model=gemini_model,
        tools=[DuckDuckGo()],
        instructions=[
            "Give highest weightage to doctor's review and now using that generate a final report",
        ],
        storage=SqlAgentStorage(table_name="verification_agent", db_file="lung_agents.db"),
        markdown=True,
    )
    
    return detection_agent_1, detection_agent_2, verification_agent

# Main Streamlit app
def main():
    import sys
    
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
    
    # Model information
    st.sidebar.markdown("<div class='sub-header'>Model Information</div>", unsafe_allow_html=True)
    st.sidebar.info("""
    This application uses:
    - UNet model for pneumonia detection
    - PyTorch model for tuberculosis detection
    - Gemini LLM for analysis
    
    Make sure the model files are in the same directory:
    - model_unet.keras
    - _best_model.pt
    """)
    
    # API key input
    gemini_api_key = st.sidebar.text_input("Gemini API Key (optional)", 
                                          value=os.environ.get('GEMINI_API_KEY', 'AIzaSyArW-r0ojXCNFfOMk-lau2mg2lUTUfJvH4'),
                                          type="password")
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='sub-header'>Upload Chest X-ray Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
            
            # Save the uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file_path = temp_file.name
            image.save(temp_file_path)
            
            # Process button
            process_button = st.button("Process X-ray")
            
            if process_button:
                with st.spinner("Processing X-ray image..."):
                    # Run analysis
                    try:
                        # Create agents
                        detection_agent_1, detection_agent_2, verification_agent = create_agents()
                        
                        # Create temporary files for analysis results
                        pneumonia_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                        pneumonia_analysis_path = pneumonia_file.name
                        pneumonia_file.close()
                        
                        tb_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                        tuberculosis_analysis_path = tb_file.name
                        tb_file.close()
                        
                        final_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                        final_result_path = final_file.name
                        final_file.close()
                        
                        # Run pneumonia detection
                        st.info("Running pneumonia detection...")
                        pneumonia_result = detect_pneumonia(temp_file_path, age, sex, position)
                        
                        if pneumonia_result is None:
                            st.error("Error in pneumonia detection. Please ensure the model is available.")
                        else:
                            # Extract file paths
                            original_path = pneumonia_result["original_image"]
                            pneumonia_mask_path = pneumonia_result["mask_path"]
                            pneumonia_prob = pneumonia_result["probability"]
                            
                            # Generate prompt for pneumonia analysis
                            pneumonia_prompt = f"""
                            Process the following patient information and chest X-ray:
                            
                            Patient Name: '{patient_name}'
                            Age: {age}
                            Sex: {"Male" if sex == 0 else "Female"}
                            Position: {"AP" if position == 0 else "PA"}
                            Pneumonia Probability: {pneumonia_prob * 100:.2f}%
                            
                            I am passing one image segmentation mask for pneumonia. Return detailed analysis with the location of infection, 
                            whether the patient suffers from pneumonia or is normal. If the segmentation mask is all black it means no 
                            abnormality is present. Save the detailed analysis to a file with name {pneumonia_analysis_path}
                            """
                            
                            # Redirect stdout to capture response
                            capture = CaptureOutput()
                            
                            # Get pneumonia analysis using print_response
                            detection_agent_1.print_response(
                                pneumonia_prompt, 
                                images=[pneumonia_mask_path, original_path],
                                markdown=True
                            )
                            
                            # Restore stdout
                            capture.restore_stdout()
                            
                            # Run tuberculosis detection
                            st.info("Running tuberculosis detection...")
                            tb_result = detect_tuberculosis(temp_file_path, age, sex, position)
                            
                            if tb_result is None:
                                st.error("Error in tuberculosis detection. Please ensure the model is available.")
                            else:
                                # Extract file paths
                                tb_mask_path = tb_result["mask_path"]
                                
                                # Generate prompt for tuberculosis analysis
                                tb_prompt = f"""
                                Process the following patient information and chest X-ray:
                                
                                Patient Name: '{patient_name}'
                                Age: {age}
                                Sex: {"Male" if sex == 0 else "Female"}
                                Position: {"AP" if position == 0 else "PA"}
                                
                                I am passing one image segmentation mask for tuberculosis. Return detailed analysis and 
                                location of infection whether the patient suffers from tuberculosis or is normal. 
                                Save the answer to a file named {tuberculosis_analysis_path}
                                """
                                
                                # Redirect stdout to capture response
                                capture = CaptureOutput()
                                
                                # Get tuberculosis analysis using print_response
                                detection_agent_2.print_response(
                                    tb_prompt, 
                                    images=[tb_mask_path],
                                    markdown=True
                                )
                                
                                # Restore stdout
                                capture.restore_stdout()
                                
                                # Save session states for later use
                                st.session_state.analysis_complete = True
                                st.session_state.pneumonia_analysis_path = pneumonia_analysis_path
                                st.session_state.tuberculosis_analysis_path = tuberculosis_analysis_path
                                st.session_state.final_result_path = final_result_path
                                st.session_state.pneumonia_mask_path = pneumonia_mask_path
                                st.session_state.tb_mask_path = tb_mask_path
                                st.session_state.original_path = original_path
                                st.session_state.verification_agent = verification_agent
                                
                                # Rerun to show doctor's review section
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    with col2:
        if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
            st.markdown("<div class='sub-header'>Analysis Results</div>", unsafe_allow_html=True)
            
            # Display pneumonia mask
            if os.path.exists(st.session_state.pneumonia_mask_path):
                pneumonia_mask = Image.open(st.session_state.pneumonia_mask_path)
                st.image(pneumonia_mask, caption="Pneumonia Segmentation Mask", use_column_width=True)
            
            # Display TB mask
            if os.path.exists(st.session_state.tb_mask_path):
                tb_mask = Image.open(st.session_state.tb_mask_path)
                st.image(tb_mask, caption="Tuberculosis Segmentation Mask", use_column_width=True)
            
            # Display analysis results in tabs
            tab1, tab2, tab3 = st.tabs(["Pneumonia Analysis", "Tuberculosis Analysis", "Final Report"])
            
            with tab1:
                if os.path.exists(st.session_state.pneumonia_analysis_path):
                    with open(st.session_state.pneumonia_analysis_path, "r") as f:
                        pneumonia_content = f.read()
                    st.markdown(pneumonia_content)
            
            with tab2:
                if os.path.exists(st.session_state.tuberculosis_analysis_path):
                    with open(st.session_state.tuberculosis_analysis_path, "r") as f:
                        tuberculosis_content = f.read()
                    st.markdown(tuberculosis_content)
            
            with tab3:
                if 'final_report_generated' in st.session_state and st.session_state.final_report_generated:
                    if os.path.exists(st.session_state.final_result_path):
                        with open(st.session_state.final_result_path, "r") as f:
                            final_content = f.read()
                        st.markdown(final_content)
                else:
                    st.info("Final report will be generated after doctor's review")
            
            # Doctor's review
            st.markdown("<div class='sub-header'>Doctor's Review</div>", unsafe_allow_html=True)
            doctor_review = st.text_area("Enter the doctor's review:", height=150, 
                                        placeholder="Enter your professional assessment of the patient's condition...")
            
            if st.button("Generate Final Report"):
                if doctor_review:
                    with st.spinner("Generating final report..."):
                        # Create next prompt for verification
                        with open(st.session_state.pneumonia_analysis_path, "r") as pneumo_file:
                            pneumonia_content = pneumo_file.read().strip()
                        
                        with open(st.session_state.tuberculosis_analysis_path, "r") as tb_file:
                            tuberculosis_content = tb_file.read().strip()
                        
                        final_prompt = f"""
                        Pneumonia Analysis: {pneumonia_content}
                        
                        Tuberculosis Analysis: {tuberculosis_content}
                        
                        Doctor's Review: {doctor_review}
                        
                        Giving highest weightage to doctor's review and now using that generate a final report. 
                        Save the results in {st.session_state.final_result_path}
                        """
                        
                        # Redirect stdout to capture response
                        capture = CaptureOutput()
                        
                        # Get final verification using print_response
                        st.session_state.verification_agent.print_response(
                            final_prompt, 
                            markdown=True
                        )
                        
                        # Restore stdout
                        capture.restore_stdout()
                        
                        st.session_state.final_report_generated = True
                        st.rerun()
                else:
                    st.warning("Please enter the doctor's review before generating the final report")

# Run the app
if __name__ == "__main__":
    main()
