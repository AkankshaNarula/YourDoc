"""
This module contains all the prompts used by the agents in the application.
Centralizing prompts makes them easier to maintain and modify.
"""

def get_pneumonia_prompt(patient_name, age, sex, position, probability, output_path):
    """
    Generate prompt for pneumonia analysis.
    
    Args:
        patient_name (str): Name of the patient
        age (int): Age of the patient
        sex (str): Sex of the patient (Male/Female)
        position (str): X-ray position (AP/PA)
        probability (float): Pneumonia probability (0-100)
        output_path (str): Path to save the analysis output
        
    Returns:
        str: Formatted prompt for the pneumonia analysis agent
    """
    return f"""
    Process the following patient information and chest X-ray:
    
    Patient Name: '{patient_name}'
    Age: {age}
    Sex: {sex}
    Position: {position}
    Pneumonia Probability: {probability:.2f}%
    
    I am passing one image segmentation mask for pneumonia. Return detailed analysis with the location of infection, 
    whether the patient suffers from pneumonia or is normal. If the segmentation mask is all black it means no 
    abnormality is present. Save the detailed analysis to a file with name {output_path}
    """

def get_tuberculosis_prompt(patient_name, age, sex, position, features, mask_b64, original_img_b64, masked_img_b64, output_path):
    """
    Generate prompt for tuberculosis analysis.
    
    Args:
        patient_name (str): Name of the patient
        age (int): Age of the patient
        sex (str): Sex of the patient (Male/Female)
        position (str): X-ray position (AP/PA)
        features (dict): Dictionary containing extracted radiographic features
        mask_b64 (str): Base64 encoded segmentation mask
        original_img_b64 (str): Base64 encoded original image
        masked_img_b64 (str): Base64 encoded masked image
        output_path (str): Path to save the analysis output
        
    Returns:
        str: Formatted prompt for the tuberculosis analysis agent
    """
    features_str = "\n".join([f"  - {key}: {value}" for key, value in features.items()])
    
    return f"""
    Process the following patient information, chest X-ray images, and quantitative features:
    
    Patient Name: '{patient_name}'
    Age: {age}
    Sex: {sex}
    Position: {position}
    
    IMPORTANT CONTEXT: I am providing three base64-encoded images:
    1. The original chest X-ray
    2. The tuberculosis segmentation mask 
    3. The masked image highlighting potential areas of concern
    
    I'm also providing the following quantitative radiographic features extracted from the images:
    {features_str}
    
    Based on these features and images, please provide a comprehensive tuberculosis assessment including:
    
    1. TB Classification: Determine if tuberculosis is present or if the scan appears normal
    2. Probability of TB: Estimate the likelihood of TB infection as a percentage
    3. Severity Assessment: If TB is present, classify as minimal, moderate, or advanced
    4. Location Analysis: Specify affected lung regions (upper/middle/lower lobes, left/right lung)
    5. Pattern Description: Note any cavitation, consolidation, nodular patterns, or fibrotic changes
    6. Clinical Correlation: How findings relate to patient demographics and potential risk factors
    7. Recommendations: Suggested follow-up tests or clinical actions
    
    IMPORTANT: TB typically presents as areas of increased opacity (brightness in X-ray) in the upper lobes, 
    potentially with cavitation. Asymmetry between lungs can be significant. The lung_area_ratio, 
    opacity_score, and left_to_right_ratio are particularly relevant for TB assessment.
    
    Save the detailed analysis to a file named {output_path}
    """

def get_verification_prompt(pneumonia_content, tuberculosis_content, doctor_review, output_path):
    """
    Generate prompt for verification and final report generation.
    
    Args:
        pneumonia_content (str): Content of the pneumonia analysis
        tuberculosis_content (str): Content of the tuberculosis analysis
        doctor_review (str): Doctor's review text
        output_path (str): Path to save the final report
        
    Returns:
        str: Formatted prompt for the verification agent
    """
    return f"""
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
    
    Save this report to the file {output_path}
    """

def get_prioritization_prompt(final_report, patient_info, patient_records_json):
    """
    Generate prompt for patient prioritization.
    
    Args:
        final_report (str): The final medical report
        patient_info (dict): Information about the current patient
        patient_records_json (str): JSON string of all patient records
        
    Returns:
        str: Formatted prompt for the prioritization agent
    """
    return f"""
    I need you to prioritize patients for medical appointments.
    
    CURRENT PATIENT REPORT:
    {final_report}
    
    CURRENT PATIENT INFO:
    Name: {patient_info["name"]}
    Age: {patient_info["age"]}
    Sex: {patient_info["sex"]}
    Symptoms: {patient_info["symptoms"]}
    
    OTHER PATIENTS WAITING:
    {patient_records_json}
    
    Based on the severity and urgency of each case, assign a priority level (High, Medium, or Low) to each patient including the current patient.
    Return a prioritized list of patients with recommended appointment timeframes. Explain your reasoning.
    """

def get_scheduling_prompt(patient_data, start_date, end_date, duration):
    """
    Generate prompt for appointment scheduling.
    
    Args:
        patient_data (dict): Patient data dictionary
        start_date (datetime.date): Start date for appointment search
        end_date (datetime.date): End date for appointment search
        duration (str): Duration of the appointment
        
    Returns:
        str: Formatted prompt for the calendar agent
    """
    return f"""
    I need to schedule an appointment for a patient with the following details:
    
    Name: {patient_data["name"]}
    Age: {patient_data["age"]}
    Sex: {patient_data["sex"]}
    Condition: {patient_data["condition"]}
    Priority: {patient_data["severity"] if "severity" in patient_data else "Medium"}
    
    Please find available slots between {start_date} and {end_date} for a {duration} appointment.
    Based on the patient's priority level ({patient_data["severity"] if "severity" in patient_data else "Medium"}), recommend the best time for scheduling.
    For high priority, prefer same-day or next-day appointments.
    For medium priority, prefer appointments within 2-5 days.
    For low priority, scheduling within 1-2 weeks is acceptable.
    """