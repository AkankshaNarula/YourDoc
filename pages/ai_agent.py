import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
import cv2
import os
import datetime
import json
from io import BytesIO
import tempfile
import io
import sys
import gdown
import os
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.file import FileTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlecalendar import GoogleCalendarTools
from phi.storage.agent.sqlite import SqlAgentStorage
from tzlocal import get_localzone_name

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
        st.info("Downloading model... Please wait.")
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
    model_path = 'saved_models/_best_model.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"TB model not found at {model_path}. Please ensure the model file exists.")
        return None
    
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
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
        api_key='AIzaSyArW-r0ojXCNFfOMk-lau2mg2lUTUfJvH4' # Use environment variable if available
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
    
    # 4. Patient Prioritization Agent
    prioritization_agent = Agent(
        name="Patient Prioritization Agent",
        model=gemini_model,
        tools=[FileTools()],
        instructions=[
            "You are a medical prioritization assistant specialized in analyzing lung disease reports.",
            "Your task is to analyze the current patient's report along with records of other patients waiting for appointments.",
            "Determine the priority order of patients based on the severity of their conditions.",
            "Assign priority levels (High, Medium, Low) to each patient and suggest appointment scheduling order.",
            "For high priority cases, suggest same-day or next-day appointments.",
            "For medium priority, suggest appointments within 2-5 days.",
            "For low priority, suggest appointments within 1-2 weeks.",
            "Always prioritize elderly patients (65+) and those with severe symptoms."
        ],
        storage=SqlAgentStorage(table_name="prioritization_agent", db_file="lung_agents.db"),
        markdown=True,
    )
    
    # 5. Calendar Scheduling Agent
    calendar_agent = Agent(
        name="Medical Scheduling Assistant",
        model=gemini_model,
        tools=[GoogleCalendarTools(credentials_path="/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/client_secret_104996183240-63co19c13llg7rhf8jc3o6hgptspmsto.apps.googleusercontent.com.json")],
        instructions=[
            f"You are scheduling assistant. Today is {datetime.datetime.now()} and the users timezone is {get_localzone_name()}.",
            "You schedule medical appointments in the doctor's calendar based on priority assignments.",
            "Make sure to check availability before scheduling.",
            "Schedule high priority cases first, followed by medium and then low priority cases.",
            "Appointments should be 30 minutes for regular check-ups and 45 minutes for severe cases.",
            "For emergency cases marked as 'urgent', find the earliest available slot on the same day.",
            "Avoid scheduling appointments back-to-back; allow 15-minute breaks between patients.",
            "When scheduling, include the patient's name, age, and condition in the appointment title.",
        ],
        show_tool_calls=True,
    )
    
    return detection_agent_1, detection_agent_2, verification_agent, prioritization_agent, calendar_agent

# Previous patient records (dummy data)
def load_patient_records():
    # Check if we already have patient records saved
    if os.path.exists("patient_records.json"):
        with open("patient_records.json", "r") as f:
            return json.load(f)
    
    # Create dummy patient records
    dummy_records = [
        {
            "patient_id": "P001",
            "name": "Alice Johnson",
            "age": 67,
            "sex": "Female",
            "condition": "Suspected pneumonia",
            "last_visit": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
            "symptoms": "Persistent cough, chest pain, fever (38.2°C)",
            "severity": "Medium",
            "report_summary": "Bilateral basilar opacities with possible consolidation in right lower lobe. Patient has history of COPD. Requires follow-up within 3 days."
        },
        {
            "patient_id": "P002",
            "name": "Robert Chen",
            "age": 42,
            "sex": "Male",
            "condition": "Tuberculosis treatment follow-up",
            "last_visit": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
            "symptoms": "Mild cough, weight stable, no night sweats",
            "severity": "Low",
            "report_summary": "Follow-up chest X-ray showing improvement in tuberculosis lesions. Patient completing month 2 of TB treatment with good medication adherence."
        },
        {
            "patient_id": "P003",
            "name": "Emma Garcia",
            "age": 78,
            "sex": "Female",
            "condition": "Severe pneumonia",
            "last_visit": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "symptoms": "High fever (39.5°C), severe shortness of breath, chest pain",
            "severity": "High",
            "report_summary": "Extensive bilateral consolidation with possible pleural effusion. Patient has history of heart failure. Urgent follow-up required."
        },
        {
            "patient_id": "P004",
            "name": "James Wilson",
            "age": 35,
            "sex": "Male",
            "condition": "Routine TB screening (healthcare worker)",
            "last_visit": (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
            "symptoms": "No symptoms, routine screening",
            "severity": "Low",
            "report_summary": "Clear lung fields. Routine screening with no abnormalities detected."
        },
        {
            "patient_id": "P005",
            "name": "Sarah Kim",
            "age": 56,
            "sex": "Female",
            "condition": "Suspected tuberculosis",
            "last_visit": (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
            "symptoms": "Chronic cough (3+ months), weight loss, night sweats",
            "severity": "Medium",
            "report_summary": "Right upper lobe infiltrate with small cavity formation. High suspicion for active tuberculosis infection. Requires prompt diagnosis confirmation."
        }
    ]
    
    # Save the dummy records to a file
    with open("patient_records.json", "w") as f:
        json.dump(dummy_records, f, indent=4)
    
    return dummy_records

# Save current patient to records
def save_current_patient(patient_data):
    # Load existing records
    records = load_patient_records()
    
    # Generate a new patient ID
    patient_id = f"P{len(records) + 1:03d}"
    
    # Create new patient record
    new_patient = {
        "patient_id": patient_id,
        "name": patient_data["name"],
        "age": patient_data["age"],
        "sex": patient_data["sex"],
        "condition": patient_data["condition"],
        "last_visit": datetime.datetime.now().strftime("%Y-%m-%d"),
        "symptoms": patient_data["symptoms"],
        "severity": "Unknown",  # Will be updated by prioritization agent
        "report_summary": patient_data["report_summary"]
    }
    
    # Add to records
    records.append(new_patient)
    
    # Save updated records
    with open("patient_records.json", "w") as f:
        json.dump(records, f, indent=4)
    
    return patient_id

# Main Streamlit app
def main():
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
    - UNet model for pneumonia detection
    - DeepLabV3 model for tuberculosis detection
    - Gemini LLM for analysis and scheduling
    
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
    
    # Calendar credentials
    calendar_credentials_path = st.sidebar.text_input("Path to Google Calendar Credentials", 
                                                    value="/Users/akankshanarula/Desktop/Google Girl Hackathon/YourDoc/client_secret_104996183240-63co19c13llg7rhf8jc3o6hgptspmsto.apps.googleusercontent.com.json",
                                                    help="Path to the credentials.json file for Google Calendar API")
    
    # Main content area
    st.markdown("<div class='main-header'>Lung Disease Analysis System</div>", unsafe_allow_html=True)
    st.write("Upload a chest X-ray image and get AI-powered analysis for pneumonia and tuberculosis")
    
    # Create tabs for different sections
    tabs = st.tabs(["Image Analysis", "Patient Prioritization", "Appointment Scheduling"])
    
    # Tab 1: Image Analysis
    with tabs[0]:
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
                            detection_agent_1, detection_agent_2, verification_agent, prioritization_agent, calendar_agent = create_agents()
                            
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
                                    st.session_state.prioritization_agent = prioritization_agent
                                    st.session_state.calendar_agent = calendar_agent
                                    st.session_state.patient_info = {
                                        "name": patient_name,
                                        "age": age,
                                        "sex": "Male" if sex == 0 else "Female",
                                        "symptoms": symptoms,
                                        "condition": "Suspected lung condition",
                                        "report_summary": "Pending doctor's review"
                                    }
                                    
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
                            
                            verification_prompt = f"""
                            I need you to generate a final medical report based on the following information:
                            
                            PNEUMONIA ANALYSIS:
                            {pneumonia_content}
                            
                            TUBERCULOSIS ANALYSIS:
                            {tuberculosis_content}
                            
                            DOCTOR'S REVIEW:
                            {doctor_review}
                            
                            Using these inputs, generate a comprehensive final report that:
                            1. Summarizes the key findings from both analyses
                            2. Incorporates the doctor's professional opinion
                            3. Provides a clear diagnosis (if possible)
                            4. Recommends appropriate next steps for treatment
                            5. Suggests a follow-up timeline
                            
                            Save this report to the file {st.session_state.final_result_path}
                            """
                            
                            # Get final verification
                            capture = CaptureOutput()
                            
                            st.session_state.verification_agent.print_response(
                                verification_prompt,
                                markdown=True
                            )
                            
                            capture.restore_stdout()
                            
                            # Update patient info with report summary
                            with open(st.session_state.final_result_path, "r") as f:
                                final_report = f.read()
                                # Extract a summary from the report (first 200 chars or so)
                                report_summary = final_report[:200] + "..." if len(final_report) > 200 else final_report
                            
                            st.session_state.patient_info["report_summary"] = report_summary
                            
                            # Set flag to show final report
                            st.session_state.final_report_generated = True
                            
                            # Save patient to records
                            patient_id = save_current_patient(st.session_state.patient_info)
                            st.session_state.current_patient_id = patient_id
                            
                            # Rerun to show final report
                            st.rerun()
    
    # Tab 2: Patient Prioritization
    with tabs[1]:
        st.markdown("<div class='sub-header'>Patient Prioritization</div>", unsafe_allow_html=True)
        st.write("View and prioritize patients awaiting appointments")
        
        # Load patient records
        patient_records = load_patient_records()
        
        # Display patient records
        if patient_records:
            st.write(f"Total patients waiting: {len(patient_records)}")
            
            # Check if we have a final report to prioritize
            if 'final_report_generated' in st.session_state and st.session_state.final_report_generated:
                if st.button("Prioritize Current Patient"):
                    with st.spinner("Prioritizing patients..."):
                        # Get the final report
                        with open(st.session_state.final_result_path, "r") as f:
                            final_report = f.read()
                        
                        # Create JSON string of patient records
                        patient_records_json = json.dumps(patient_records, indent=2)
                        
                        # Create prioritization prompt
                        prioritization_prompt = f"""
                        I need you to prioritize patients for medical appointments.
                        
                        CURRENT PATIENT REPORT:
                        {final_report}
                        
                        CURRENT PATIENT INFO:
                        Name: {st.session_state.patient_info["name"]}
                        Age: {st.session_state.patient_info["age"]}
                        Sex: {st.session_state.patient_info["sex"]}
                        Symptoms: {st.session_state.patient_info["symptoms"]}
                        
                        OTHER PATIENTS WAITING:
                        {patient_records_json}
                        
                        Based on the severity and urgency of each case, assign a priority level (High, Medium, or Low) to each patient including the current patient.
                        Return a prioritized list of patients with recommended appointment timeframes. Explain your reasoning.
                        """
                        
                        # Get prioritization
                        capture = CaptureOutput()
                        
                        # Run prioritization
                        st.session_state.prioritization_agent.print_response(
                            prioritization_prompt,
                            markdown=True
                        )
                        
                        capture.restore_stdout()
                        
                        # Set flag for prioritization complete
                        st.session_state.prioritization_complete = True
                        st.rerun()
            
            # Display patient table
            for patient in patient_records:
                # Determine style class based on severity
                severity_class = ""
                if "severity" in patient:
                    if patient["severity"].lower() == "high":
                        severity_class = "priority-high"
                    elif patient["severity"].lower() == "medium":
                        severity_class = "priority-medium"
                    elif patient["severity"].lower() == "low":
                        severity_class = "priority-low"
                
                # Create patient card
                st.markdown(f"""
                <div class="result-container">
                    <h3>{patient["name"]} <span class="{severity_class}">({patient["severity"]})</span></h3>
                    <p><strong>Patient ID:</strong> {patient["patient_id"]} | <strong>Age:</strong> {patient["age"]} | <strong>Sex:</strong> {patient["sex"]}</p>
                    <p><strong>Condition:</strong> {patient["condition"]}</p>
                    <p><strong>Symptoms:</strong> {patient["symptoms"]}</p>
                    <p><strong>Last Visit:</strong> {patient["last_visit"]}</p>
                    <p><strong>Report Summary:</strong> {patient["report_summary"]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # If prioritization was just done, show the results
            if 'prioritization_complete' in st.session_state and st.session_state.prioritization_complete:
                st.markdown("<div class='sub-header'>Prioritization Results</div>", unsafe_allow_html=True)
                
                # Display the prioritization results (this would come from the prioritization agent)
                st.info("Patients have been prioritized based on severity and urgency. High priority patients should be scheduled first.")
                
                # Clear the flag to avoid showing this section again on rerun
                st.session_state.prioritization_complete = False
        else:
            st.warning("No patient records found.")
    
    # Tab 3: Appointment Scheduling
    with tabs[2]:
        st.markdown("<div class='sub-header'>Appointment Scheduling</div>", unsafe_allow_html=True)
        st.write("Schedule appointments for prioritized patients")
        
        # Check if we have Google Calendar credentials
        calendar_credentials_valid = False
        if calendar_credentials_path and os.path.exists(calendar_credentials_path):
            calendar_credentials_valid = True
        
        if not calendar_credentials_valid:
            st.warning("Google Calendar credentials are required for scheduling. Please enter the path to your credentials.json file in the sidebar.")
        
        # Load patient records
        patient_records = load_patient_records()
        
        # Display scheduling options
        if patient_records:
            # Select patient to schedule
            patient_options = [f"{p['name']} ({p['patient_id']})" for p in patient_records]
            selected_patient = st.selectbox("Select patient to schedule", patient_options)
            
            # Extract patient ID from selection
            selected_patient_id = selected_patient.split("(")[1].split(")")[0]
            
            # Find the selected patient in records
            selected_patient_data = next((p for p in patient_records if p["patient_id"] == selected_patient_id), None)
            
            if selected_patient_data:
                # Display patient details
                st.markdown(f"""
                <div class="result-container">
                    <h3>{selected_patient_data["name"]}</h3>
                    <p><strong>Patient ID:</strong> {selected_patient_data["patient_id"]} | <strong>Age:</strong> {selected_patient_data["age"]} | <strong>Sex:</strong> {selected_patient_data["sex"]}</p>
                    <p><strong>Condition:</strong> {selected_patient_data["condition"]}</p>
                    <p><strong>Priority:</strong> <span class="priority-{selected_patient_data["severity"].lower()}">{selected_patient_data["severity"]}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Choose date range for scheduling
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", datetime.datetime.now().date())
                with col2:
                    end_date = st.date_input("End Date", datetime.datetime.now().date() + datetime.timedelta(days=7))
                
                # Appointment duration
                duration = st.radio("Appointment Duration", ["30 minutes", "45 minutes", "60 minutes"])
                
                # Schedule button
                if st.button("Find Available Slots") and calendar_credentials_valid:
                    with st.spinner("Checking calendar for available slots..."):
                        # Format the query for the calendar agent
                        duration_minutes = int(duration.split()[0])
                        
                        # Generate scheduling prompt
                        scheduling_prompt = f"""
                        I need to schedule an appointment for a patient with the following details:
                        
                        Name: {selected_patient_data["name"]}
                        Age: {selected_patient_data["age"]}
                        Sex: {selected_patient_data["sex"]}
                        Condition: {selected_patient_data["condition"]}
                        Priority: {selected_patient_data["severity"]}
                        
                        Please find available slots between {start_date} and {end_date} for a {duration} appointment.
                        Based on the patient's priority level ({selected_patient_data["severity"]}), recommend the best time for scheduling.
                        For high priority, prefer same-day or next-day appointments.
                        For medium priority, prefer appointments within 2-5 days.
                        For low priority, scheduling within 1-2 weeks is acceptable.
                        """
                        
                        # Get scheduling recommendation
                        capture = CaptureOutput()
                        
                        # Call calendar agent
                        st.session_state.calendar_agent.print_response(
                            scheduling_prompt,
                            markdown=True
                        )
                        
                        capture.restore_stdout()
                        
                        # Display the available slots (would be returned by the calendar agent)
                        st.markdown("<div class='sub-header'>Available Slots</div>", unsafe_allow_html=True)
                        
                        # Example slots (in a real app, these would come from the calendar API)
                        example_slots = [
                            {"date": start_date + datetime.timedelta(days=1), "time": "09:00 AM"},
                            {"date": start_date + datetime.timedelta(days=1), "time": "02:30 PM"},
                            {"date": start_date + datetime.timedelta(days=2), "time": "11:15 AM"},
                            {"date": start_date + datetime.timedelta(days=3), "time": "10:00 AM"},
                        ]
                        
                        for slot in example_slots:
                            slot_str = f"{slot['date'].strftime('%A, %B %d, %Y')} at {slot['time']}"
                            if st.button(f"Schedule for {slot_str}"):
                                # This would actually create the calendar event
                                st.success(f"Appointment scheduled for {selected_patient_data['name']} on {slot_str}")
                                
                                # Create sample calendar event
                                st.markdown(f"""
                                <div class="calendar-event">
                                    <h4>Appointment: {selected_patient_data['name']}</h4>
                                    <p><strong>Date & Time:</strong> {slot_str}</p>
                                    <p><strong>Duration:</strong> {duration}</p>
                                    <p><strong>Patient ID:</strong> {selected_patient_data['patient_id']}</p>
                                    <p><strong>Condition:</strong> {selected_patient_data['condition']}</p>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.warning("No patient records found.")

if __name__ == "__main__":
    main()
