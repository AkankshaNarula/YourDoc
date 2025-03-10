import streamlit as st

def render_header():
    """
    Renders the application header.
    """
    st.markdown("<div class='main-header'>Lung Disease Analysis System</div>", unsafe_allow_html=True)
    st.markdown("""
    A comprehensive platform for chest X-ray analysis, patient prioritization, and appointment scheduling.
    This system combines advanced image processing models with AI agents to assist in lung disease diagnosis.
    """)
    st.markdown("---")

def display_patient_card(patient):
    """
    Display a patient's information in a card format.
    
    Args:
        patient (dict): Dictionary containing patient information
    """
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
        <h3>{patient["name"]} {f'<span class="{severity_class}">({patient["severity"]})</span>' if "severity" in patient else ""}</h3>
        <p><strong>Patient ID:</strong> {patient.get("patient_id", "N/A")} | <strong>Age:</strong> {patient.get("age", "N/A")} | <strong>Sex:</strong> {patient.get("sex", "N/A")}</p>
        <p><strong>Condition:</strong> {patient.get("condition", "Unknown")}</p>
        <p><strong>Symptoms:</strong> {patient.get("symptoms", "None reported")}</p>
        <p><strong>Last Visit:</strong> {patient.get("last_visit", "No previous visits")}</p>
        <p><strong>Report Summary:</strong> {patient.get("report_summary", "No report available")}</p>
    </div>
    """, unsafe_allow_html=True)

def render_result_container(patient_data):
    """
    Renders a detailed result container for a patient.
    
    Args:
        patient_data (dict): Dictionary containing patient information
    """
    st.markdown("<div class='sub-header'>Patient Details</div>", unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Name:** {patient_data.get('name', 'N/A')}")
        st.markdown(f"**Age:** {patient_data.get('age', 'N/A')}")
        st.markdown(f"**Sex:** {patient_data.get('sex', 'N/A')}")
        st.markdown(f"**Patient ID:** {patient_data.get('patient_id', 'N/A')}")
    
    with col2:
        st.markdown(f"**Condition:** {patient_data.get('condition', 'Unknown')}")
        st.markdown(f"**Symptoms:** {patient_data.get('symptoms', 'None reported')}")
        
        # Render priority with appropriate color if available
        if "severity" in patient_data:
            severity = patient_data["severity"]
            severity_color = ""
            
            if severity.lower() == "high":
                severity_color = "#D32F2F"
            elif severity.lower() == "medium":
                severity_color = "#FB8C00"
            elif severity.lower() == "low":
                severity_color = "#388E3C"
                
            st.markdown(f"**Priority:** <span style='color:{severity_color};font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
        
        st.markdown(f"**Last Visit:** {patient_data.get('last_visit', 'No previous visits')}")
    
    # Show report summary if available
    if "report_summary" in patient_data and patient_data["report_summary"]:
        st.markdown("### Medical Report Summary")
        st.info(patient_data["report_summary"])

def display_calendar_event(patient_data, slot_str, duration):
    """
    Display a calendar event for a patient.
    
    Args:
        patient_data (dict): Dictionary containing patient information
        slot_str (str): String representation of the appointment slot
        duration (str): Duration of the appointment
    """
    st.markdown("<div class='sub-header'>Scheduled Appointment</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="calendar-event">
        <h4>Appointment for: {patient_data.get('name', 'N/A')}</h4>
        <p><strong>Date & Time:</strong> {slot_str}</p>
        <p><strong>Duration:</strong> {duration}</p>
        <p><strong>Patient ID:</strong> {patient_data.get('patient_id', 'N/A')}</p>
        <p><strong>Condition:</strong> {patient_data.get('condition', 'Unknown')}</p>
        <p><strong>Priority:</strong> {patient_data.get('severity', 'Not specified')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a confirmation message
    st.success(f"Appointment has been scheduled successfully for {patient_data.get('name', 'the patient')}.")
    
    # Add option to send notification
    if st.button("Send appointment notification to patient"):
        st.info(f"Notification sent to {patient_data.get('name', 'the patient')}.")

def load_css():
    """
    Sets custom CSS for the Streamlit application.
    """
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
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #1E88E5;
        }
        .stButton>button {
            background-color: #1976D2;
            color: white;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
    </style>
    """, unsafe_allow_html=True)