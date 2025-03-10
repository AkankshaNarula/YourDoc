from phi.model.google import Gemini
from .pneumonia_agent import create_pneumonia_agent
from .tuberculosis_agent import create_tuberculosis_agent
from .verification_agent import create_verification_agent
from .prioritization_agent import create_prioritization_agent
from .calendar_agent import create_calendar_agent

def create_agents(api_key=None):
    """
    Create and initialize all agents needed for the application
    
    Args:
        api_key: Optional API key for Gemini model
        
    Returns:
        tuple: Tuple containing all configured agents
    """
    # Initialize the Gemini model
    gemini_model = Gemini(
        id="gemini-1.5-flash",  # Specify the desired Gemini model version
        api_key=api_key or 'AIzaSyArW-r0ojXCNFfOMk-lau2mg2lUTUfJvH4'  # Use provided key or default
    )
    
    # Create specialized agents
    detection_agent_1 = create_pneumonia_agent(gemini_model)
    detection_agent_2 = create_tuberculosis_agent(gemini_model)
    verification_agent = create_verification_agent(gemini_model)
    prioritization_agent = create_prioritization_agent(gemini_model)
    calendar_agent = create_calendar_agent(gemini_model)
    
    return detection_agent_1, detection_agent_2, verification_agent, prioritization_agent, calendar_agent