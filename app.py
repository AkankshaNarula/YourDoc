import streamlit as st
from PIL import Image
import os
from pages import ai_agent, pneumonia_detection, tb_segmentation

# Set page configuration
st.set_page_config(
    page_title="YourDoc AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"  # Keep sidebar expanded by default
)

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

# Function to display the home page
def home():
    # Use custom styling for title
    st.markdown('<div class="main-header">Medical Imaging AI Platform</div>', unsafe_allow_html=True)
    
    # Add "Proposed Solution" title with custom styling
    st.markdown('<div class="sub-header" style="text-align: center;">Proposed Solution</div>', unsafe_allow_html=True)
    
    # Introduction section
    st.markdown('<div class="sub-header">Welcome to our Medical Imaging AI Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    This platform provides AI-powered tools for medical imaging analysis, 
    focusing on respiratory conditions like pneumonia and tuberculosis.
    
    ### Key Features:
    - AI-assisted diagnosis of pneumonia from chest X-rays
    - Tuberculosis segmentation and detection in chest CT scans
    - Interactive AI agent for medical imaging consultation
    """)
    
    # Demo section with columns
    st.markdown('<div class="sub-header">Demo</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header" style="font-size: 1.2rem;">Pneumonia Detection</div>', unsafe_allow_html=True)
        # Placeholder for demo image - replace with your actual image path
        demo_img_path = os.path.join("assets", "pneumonia_demo.png")
        
        # Use a placeholder image if the file doesn't exist
        try:
            img = Image.open(demo_img_path)
            st.image(img, caption="Pneumonia Detection Demo", use_column_width=True)
        except:
            st.image("/api/placeholder/500/300", caption="Pneumonia Detection Demo")
        
        st.markdown("""
        Our pneumonia detection system uses deep learning to analyze chest X-rays
        and identify signs of pneumonia with high accuracy.
        """)
    
    with col2:
        st.markdown('<div class="sub-header" style="font-size: 1.2rem;">Tuberculosis Segmentation</div>', unsafe_allow_html=True)
        # Placeholder for demo image - replace with your actual image path
        demo_img_path = os.path.join("assets", "tb_demo.png")
        
        # Use a placeholder image if the file doesn't exist
        try:
            img = Image.open(demo_img_path)
            st.image(img, caption="TB Segmentation Demo", use_column_width=True)
        except:
            st.image("/api/placeholder/500/300", caption="TB Segmentation Demo")
        
        st.markdown("""
        The tuberculosis segmentation tool can identify and highlight TB-affected 
        areas in CT scans, assisting radiologists in diagnosis and treatment planning.
        """)
    
    # Technical overview
    st.markdown('<div class="sub-header">Technical Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    Our system employs state-of-the-art deep learning models:
    
    - **Pneumonia Detection**: Convolutional Neural Network trained on over 5,000 chest X-rays
    - **TB Segmentation**: U-Net architecture optimized for lung CT segmentation
    - **AI Agent**: Natural language processing for interactive medical imaging consultation
    
    All models have been validated with healthcare professionals and achieve high accuracy rates 
    in clinical testing environments.
    """)
    
    # Call to action
    st.markdown('<div class="sub-header">Get Started</div>', unsafe_allow_html=True)
    st.markdown("""
    Explore the features of the platform directly:
    
    1. **AI Agent**: Consult with our AI on medical imaging questions
    2. **Pneumonia Detection**: Upload and analyze chest X-rays
    3. **TB Segmentation**: Process CT scans for tuberculosis analysis
    """)

# Directly display the home page content
home()

# Add footer
st.markdown("---")
st.markdown("© 2025 YourDoc AI Platform | Developed for healthcare professionals")
