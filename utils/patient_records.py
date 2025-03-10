import os
import json
import datetime

def load_patient_records():
    """
    Load patient records from JSON file or create dummy data if none exists
    
    Returns:
        list: List of patient record dictionaries
    """
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

def save_current_patient(patient_data):
    """
    Save current patient to records
    
    Args:
        patient_data: Dictionary containing patient information
        
    Returns:
        str: Generated patient ID
    """
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