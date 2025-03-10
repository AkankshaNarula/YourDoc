import os
import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import datetime
import json
from models.pneumonia import detect_pneumonia
from models.tuberculosis import detect_tuberculosis
from agents.agent_factory import create_agents
from utils.output_capture import CaptureOutput
from utils.patient_records import load_patient_records, save_current_patient
from ui.components import render_header, display_patient_card, render_result_container, display_calendar_event, load_css
from prompts.prompts import get_pneumonia_prompt, get_tuberculosis_prompt, get_verification_prompt, get_prioritization_prompt, get_scheduling_prompt





def main():

    load_css()
    
    
    st.sidebar.markdown("<div class='sub-header'>Patient Information</div>", unsafe_allow_html=True)
    
    patient_name = st.sidebar.text_input("Patient Name", "Akanksha")
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
    
   
    st.sidebar.markdown("<div class='sub-header'>Model Information</div>", unsafe_allow_html=True)
    st.sidebar.info("""
    This application uses:
    - UNet model for pneumonia detection
    - DeepLabV3 model for tuberculosis detection
    - Gemini LLM for analysis and scheduling
    
    """)
    
    
    gemini_api_key = st.sidebar.text_input("Gemini API Key (optional)", 
                                          value=os.environ.get('GEMINI_API_KEY', 'AIzaSyArW-r0ojXCNFfOMk-lau2mg2lUTUfJvH4'),
                                          type="password")
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key
    
    
    calendar_credentials_path = st.sidebar.text_input("Path to Google Calendar Credentials", 
                                                    value="/path/to/client_secret.json",
                                                    help="Path to the credentials.json file for Google Calendar API")
    
    
    render_header()
    
    
    tabs = st.tabs(["Image Analysis", "Patient Prioritization", "Appointment Scheduling"])
    
   
    with tabs[0]:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<div class='sub-header'>Upload Chest X-ray Image</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
               
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
                
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file_path = temp_file.name
                image.save(temp_file_path)
                
                
                process_button = st.button("Process X-ray")
                
                if process_button:
                    with st.spinner("Processing X-ray image..."):
                       
                        try:
                            
                            detection_agent_1, detection_agent_2, verification_agent, prioritization_agent, calendar_agent = create_agents()
                            
                            
                            pneumonia_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                            pneumonia_analysis_path = pneumonia_file.name
                            pneumonia_file.close()
                            
                            tb_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                            tuberculosis_analysis_path = tb_file.name
                            tb_file.close()
                            
                            final_file = tempfile.NamedTemporaryFile(delete=False, mode='w+t', suffix='.txt')
                            final_result_path = final_file.name
                            final_file.close()
                            
                            
                            st.info("Running pneumonia detection...")
                            pneumonia_result = detect_pneumonia(temp_file_path, age, sex, position)
                            
                            if pneumonia_result is None:
                                st.error("Error in pneumonia detection. Please ensure the model is available.")
                            else:
                                # Extract pneumonia results
                                original_path = pneumonia_result["original_image"]
                                pneumonia_mask_path = pneumonia_result["mask_path"]
                                pneumonia_prob = pneumonia_result.get("probability", 0.0)
                                pneumonia_original_img_b64 = pneumonia_result.get("original_image_b64", "")
                                pneumonia_mask_b64 = pneumonia_result.get("mask_b64", "")
                                pneumonia_masked_img_b64 = pneumonia_result.get("masked_image_b64", "")
                                pneumonia_features = pneumonia_result.get("features", {})
                                
                                # Get the patient sex and position as strings
                                patient_sex_str = "Male" if sex == 0 else "Female"
                                position_str = "AP" if position == 0 else "PA"
                                
                                # Get pneumonia prompt with additional data
                                pneumonia_prompt = get_pneumonia_prompt(
                                    patient_name=patient_name,
                                    age=age,
                                    sex=patient_sex_str,
                                    position=position_str,
                                    probability=pneumonia_prob * 100,
                                    output_path=pneumonia_analysis_path
                                )
                                
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
                                    # Extract tuberculosis results
                                    tb_mask_path = tb_result["mask_path"]
                                    tb_original_img_b64 = tb_result.get("original_image_b64", "")
                                    tb_mask_b64 = tb_result.get("mask_b64", "")
                                    tb_masked_img_b64 = tb_result.get("masked_image_b64", "")
                                    tb_features = tb_result.get("features", {})
                                    
                                    # Get tuberculosis prompt with the additional data
                                    tb_prompt = get_tuberculosis_prompt(
                                        patient_name=patient_name,
                                        age=age,
                                        sex=patient_sex_str,
                                        position=position_str,
                                        features=tb_features,
                                        mask_b64=tb_mask_b64,
                                        original_img_b64=tb_original_img_b64,
                                        masked_img_b64=tb_masked_img_b64,
                                        output_path=tuberculosis_analysis_path
                                    )
                                    
                                    # Redirect stdout to capture response
                                    capture = CaptureOutput()
                                    
                                    # Get tuberculosis analysis using print_response
                                    detection_agent_2.print_response(
                                        tb_prompt, 
                                        images=[tb_mask_path, original_path],  # Sending both mask and original image
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
                                    
                                    # Store feature data in session state for potential use later
                                    st.session_state.pneumonia_features = pneumonia_features
                                    st.session_state.tb_features = tb_features
                                    
                                    st.session_state.patient_info = {
                                        "name": patient_name,
                                        "age": age,
                                        "sex": "Male" if sex == 0 else "Female",
                                        "symptoms": symptoms,
                                        "condition": "Suspected lung condition",
                                        "report_summary": "Pending doctor's review",
                                        "radiographic_features": {
                                            "pneumonia": pneumonia_features,
                                            "tuberculosis": tb_features
                                        }
                                    }
                                    
                                    # Rerun to show doctor's review section
                                    st.rerun()
                        
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
        
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
                
                # Display radiographic features if available
                if 'pneumonia_features' in st.session_state or 'tb_features' in st.session_state:
                    with st.expander("View Radiographic Features"):
                        if 'pneumonia_features' in st.session_state:
                            st.subheader("Pneumonia Analysis Features")
                            for key, value in st.session_state.pneumonia_features.items():
                                st.text(f"{key}: {value}")
                        
                        if 'tb_features' in st.session_state:
                            st.subheader("Tuberculosis Analysis Features")
                            for key, value in st.session_state.tb_features.items():
                                st.text(f"{key}: {value}")
                
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
                            # Read analysis content
                            with open(st.session_state.pneumonia_analysis_path, "r") as pneumo_file:
                                pneumonia_content = pneumo_file.read().strip()
                            
                            with open(st.session_state.tuberculosis_analysis_path, "r") as tb_file:
                                tuberculosis_content = tb_file.read().strip()
                            
                            # Get verification prompt from prompt file
                            verification_prompt = get_verification_prompt(
                                pneumonia_content=pneumonia_content,
                                tuberculosis_content=tuberculosis_content,
                                doctor_review=doctor_review,
                                output_path=st.session_state.final_result_path
                            )
                            
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
                        
                        # Get prioritization prompt from prompt file
                        prioritization_prompt = get_prioritization_prompt(
                            final_report=final_report,
                            patient_info=st.session_state.patient_info,
                            patient_records_json=patient_records_json
                        )
                        
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
                display_patient_card(patient)
            
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
                render_result_container(selected_patient_data)
                
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
                        
                        # Get scheduling prompt from prompt file
                        scheduling_prompt = get_scheduling_prompt(
                            patient_data=selected_patient_data,
                            start_date=start_date,
                            end_date=end_date,
                            duration=duration
                        )
                        
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
                                display_calendar_event(selected_patient_data, slot_str, duration)
        else:
            st.warning("No patient records found.")

if __name__ == "__main__":
    main()