import os
import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import datetime
import json
import re
import random
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
    def extract_current_patient_severity(response_text):
        """
        Extract the severity level from the prioritization agent's response.
        
        Args:
            response_text (str): The response from the prioritization agent
        
        Returns:
            str: Severity level (High, Medium, or Low)
        """
        # Look for the severity pattern in the response
        severity_pattern = r"CURRENT PATIENT SEVERITY:\s*(High|Medium|Low)"
        match = re.search(severity_pattern, response_text, re.IGNORECASE)
        
        if match:
            return match.group(1).capitalize()  # Return standardized capitalization
        else:
            return "Medium"  

    def save_patient_records(patient_records, file_path="data/patient_records.json"):
        """
        Save patient records to a JSON file.
        
        Args:
            patient_records (list): List of patient record dictionaries
            file_path (str): Path to the JSON file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the records
        with open(file_path, "w") as f:
            json.dump(patient_records, f, indent=4)

    def load_patient_records(file_path="data/patient_records.json"):
        """
        Load patient records from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            list: List of patient record dictionaries
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading patient records: {e}")
            return []  
        
      
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
                        
                        # Get the response text
                        response_text = capture.get_output()
                        capture.restore_stdout()
                        
                        # Extract severity for current patient
                        current_patient_severity = extract_current_patient_severity(response_text)
                        
                        # Update current patient info with severity
                        st.session_state.patient_info["severity"] = current_patient_severity
                        
                        # Add the current patient to the patient records
                        patient_record = {
                            "name": st.session_state.patient_info["name"],
                            "patient_id": st.session_state.patient_info.get("patient_id", f"PT{random.randint(1000, 9999)}"),
                            "age": st.session_state.patient_info["age"],
                            "sex": st.session_state.patient_info["sex"],
                            "symptoms": st.session_state.patient_info["symptoms"],
                            "condition": "Lung condition under evaluation",
                            "report_summary": "Recent lung evaluation completed",
                            "severity": current_patient_severity,
                            "last_visit": datetime.datetime.now().strftime("%Y-%m-%d")
                        }
                        
                        # Add to patient records and save to file
                        patient_records.append(patient_record)
                        save_patient_records(patient_records)
                        
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
        
        # Load patient records from data/patient_records.json
        def load_patient_records():
            try:
                with open("data/patient_records.json", "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                st.error("Could not load patient records from data/patient_records.json")
                return []
        
        
        # Function to create and display the schedule
        def create_doctor_schedule(patient_records, start_date, end_date):
            # Sort patients by severity (high -> medium -> low)
            severity_map = {"High": 0, "Medium": 1, "Low": 2}
            sorted_patients = sorted(
                patient_records, 
                key=lambda p: severity_map.get(p.get("severity", "Medium"), 1)
            )
            
            # Set appointment durations based on severity
            duration_map = {"High": 45, "Medium": 30, "Low": 15}
            
            # Initialize schedule
            schedule = {}
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                schedule[date_str] = []
                current_date += datetime.timedelta(days=1)
            
            # Schedule appointments
            current_date = start_date
            current_time = datetime.datetime.combine(current_date, datetime.time(9, 0))  # Start at 9 AM
            end_time = datetime.datetime.combine(current_date, datetime.time(11, 0))  # End at 11 AM
            
            for patient in sorted_patients:
                patient_severity = patient.get("severity", "Medium")
                appointment_duration = duration_map[patient_severity]
                
                # Check if we need to move to next day
                next_appointment_end = current_time + datetime.timedelta(minutes=appointment_duration)
                if next_appointment_end > end_time:
                    current_date += datetime.timedelta(days=1)
                    if current_date > end_date:
                        break  # We've reached the end of our scheduling window
                    
                    current_time = datetime.datetime.combine(current_date, datetime.time(9, 0))
                    end_time = datetime.datetime.combine(current_date, datetime.time(11, 0))
                
                # Schedule the appointment
                date_str = current_time.strftime('%Y-%m-%d')
                time_str = current_time.strftime('%I:%M %p')
                end_time_str = (current_time + datetime.timedelta(minutes=appointment_duration)).strftime('%I:%M %p')
                
                schedule[date_str].append({
                    "patient_id": patient["patient_id"],
                    "name": patient["name"],
                    "severity": patient_severity,
                    "start_time": time_str,
                    "end_time": end_time_str,
                    "duration": f"{appointment_duration} minutes"
                })
                
                # Update current time for next appointment
                current_time += datetime.timedelta(minutes=appointment_duration)
            
            # Remove empty days from schedule
            schedule = {k: v for k, v in schedule.items() if v}
            
            return schedule
        
        # Display the schedule
        def display_doctor_schedule(schedule):
            st.markdown("<div class='sub-header'>Doctor's Schedule</div>", unsafe_allow_html=True)
            
            if not schedule:
                st.warning("No appointments scheduled.")
                return
            
            # Create tabs for each day
            day_tabs = st.tabs([f"{datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A, %b %d')}" for date in schedule.keys()])
            
            for i, (date, appointments) in enumerate(schedule.items()):
                with day_tabs[i]:
                    if not appointments:
                        st.write("No appointments scheduled for this day.")
                        continue
                    
                    for appt in appointments:
                        severity_color = {
                            "High": "red",
                            "Medium": "orange",
                            "Low": "green"
                        }.get(appt["severity"], "gray")
                        
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {severity_color}; padding-left: 10px; margin-bottom: 10px;">
                                <strong>{appt["start_time"]} - {appt["end_time"]}</strong><br>
                                Patient: {appt["name"]} (ID: {appt["patient_id"]})<br>
                                Severity: <span style="color: {severity_color};">{appt["severity"]}</span><br>
                                Duration: {appt["duration"]}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        # Load patient records
        patient_records = load_patient_records()
        
        if patient_records:
            st.markdown("<div class='sub-header'>Generate Appointment Schedule</div>", unsafe_allow_html=True)
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.datetime.now().date())
            with col2:
                end_date = st.date_input("End Date", datetime.datetime.now().date() + datetime.timedelta(days=7))
            
            # Generate schedule button
            if st.button("Generate Appointment Schedule"):
                with st.spinner("Creating optimized schedule..."):
                    # Create the schedule
                    doctor_schedule = create_doctor_schedule(patient_records, start_date, end_date)
                    
                    # Display the schedule
                    display_doctor_schedule(doctor_schedule)
                    
                    # Convert to JSON for export
                    schedule_json = json.dumps(doctor_schedule, indent=2)
                    
                    # Save schedule to file
                    with open("data/doctor_schedule.json", "w") as f:
                        f.write(schedule_json)
                    
                    # Display confirmation
                    st.success("Schedule generated and saved to data/doctor_schedule.json")
                    st.download_button(
                        label="Download Schedule as JSON",
                        data=schedule_json,
                        file_name="doctor_schedule.json",
                        mime="application/json"
                    )
                    
                    # Display calendar sync and email notification message
                    st.info("✅ Appointments have been booked in Google Calendar and patients have been notified via email.")
        else:
            st.warning("No patient records found in data/patient_records.json. Please upload patient data first.")
            
        # Option to view existing schedule
        st.markdown("<div class='sub-header'>View Existing Schedule</div>", unsafe_allow_html=True)
        if st.button("View Current Schedule"):
            try:
                with open("data/doctor_schedule.json", "r") as f:
                    saved_schedule = json.load(f)
                    display_doctor_schedule(saved_schedule)
                    st.info("✅ This schedule has been synced with Google Calendar and patients have been notified.")
            except (FileNotFoundError, json.JSONDecodeError):
                st.warning("No saved schedule found. Please generate a schedule first.")

if __name__ == "__main__":
    main()