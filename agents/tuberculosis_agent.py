from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.file import FileTools

def create_tuberculosis_agent(model):
    """
    Create an agent specialized in detecting tuberculosis from X-ray images
    
    Args:
        model: The LLM model to use
        
    Returns:
        Agent: Configured tuberculosis detection agent
    """
    return Agent(
        name="Lung Disease Detection Agent - Tuberculosis",
        model=model,
        tools=[FileTools()],
        instructions=[
            "You are specialized in detecting lung diseases from X-ray images",
            "Provide detailed results.",
            '''Always give a response. Analyze the provided lung images and its corresponding segmentation mask. Describe any observed abnormalities, including their size, location, and characteristics. Assess the likelihood of tuberculosis or other pulmonary conditions based on the visual evidence. Provide a detailed report that includes:
            
            Findings:
            
            - Lung Fields: Describe areas of increased opacity or consolidation, specifying size and location.
            - Air Bronchograms: Note the presence of air-filled bronchi outlined by surrounding consolidation.
            - Pleural Space: Assess for pleural effusion or pneumothorax.
            - Cardiomediastinal Silhouette: Evaluate heart size and mediastinal contours.
            - Bones and Soft Tissues: Inspect for abnormalities in ribs, spine, or soft tissues.
            
            Impression:
            
            - Summarize findings suggestive of tuberculosis or other conditions.
            - Classify the severity of any detected infection (e.g., mild, moderate, severe).
            - Recommend further evaluation or follow-up if necessary.'''
        ],
        show_tool_calls=True,
    )