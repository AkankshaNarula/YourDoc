from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage

def create_verification_agent(model):
    """
    Create an agent for verification of medical reports
    
    Args:
        model: The LLM model to use
        
    Returns:
        Agent: Configured verification agent
    """
    return Agent(
        name="Medical Verification Agent",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            "Give highest weightage to doctor's review and now using that generate a final report",
        ],
        storage=SqlAgentStorage(table_name="verification_agent", db_file="lung_agents.db"),
        markdown=True,
    )