from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.file import FileTools
from phi.storage.agent.sqlite import SqlAgentStorage

def create_prioritization_agent(model):
    """
    Create an agent for prioritizing patients based on condition severity
    
    Args:
        model: The LLM model to use
        
    Returns:
        Agent: Configured prioritization agent
    """
    return Agent(
        name="Patient Prioritization Agent",
        model=model,
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