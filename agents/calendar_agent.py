from phi.agent import Agent
from phi.model.google import Gemini
import datetime
from tzlocal import get_localzone_name

def create_calendar_agent(model):
    """
    Create an agent for scheduling medical appointments
    
    Args:
        model: The LLM model to use
        
    Returns:
        Agent: Configured calendar agent
    """
    return Agent(
        name="Medical Scheduling Assistant",
        model=model,
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