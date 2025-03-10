from phi.agent import Agent
from phi.model.google import Gemini
import datetime
from tzlocal import get_localzone_name

def create_notification_agent(model):
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
            f"You are notification assistant. Today is {datetime.datetime.now()} and the users timezone is {get_localzone_name()}.",
            "Mail the patients with their reports and scheduled time",
    
        ],
        show_tool_calls=True,
    )