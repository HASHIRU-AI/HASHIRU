from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import ollama
from googlesearch import search

# Enum for Model Types
class ModelType(Enum):
    LM_3B = "LM-3B"
    LM_5B = "LM-5B"
    LM_7B = "LM-7B"
    LLM = "LLM"

# Enum for AI Companies
class AICompany(Enum):
    OPENAI = "OpenAI"
    GOOGLE = "Google"
    META = "Meta"
    CLAUDE = "Claude"
    MISTRAL = "Mistral"

# Enum for Agent Specializations
class Specialization(Enum):
    NLP = "Natural Language Processing"
    CV = "Computer Vision"
    RL = "Reinforcement Learning"
    ML = "Machine Learning"
    DATA_SCIENCE = "Data Science"

# Enum for Model Parameters (Temperature, num_ctx, etc.)
class ModelParameters(Enum):
    NUM_CTX = 4096
    TEMPERATURE = 0.7  # A typical temperature value for model responses
    TOP_K = 50         # Number of top tokens to consider during generation

class Subtask(BaseModel):
    subtask_id: str = Field(..., description="Unique identifier for the subtask")
    description: str = Field(..., description="Description of the subtask")
    assigned_to: str = Field(..., description="ID of the agent or API handling the subtask")

class Agent(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the hired agent")
    model_type: ModelType = Field(..., description="Parameters of model used: 3 billion, 5 billion, 7 billion, LLM")
    company: AICompany = Field(..., description="Company name of the agent: OpenAI, Google, Meta, Claude, Mistral")
    specialization: Specialization = Field(..., description="Task specialization of the agent")
    cost: float = Field(..., description="Cost of hiring the agent")

class APIUtilization(BaseModel):
    api_name: str = Field(..., description="Name of the external API used")
    endpoint: str = Field(..., description="API endpoint URL")
    parameters: Dict[str, str] = Field(..., description="Input parameters and their types")
    reasoning: str = Field(..., description="Explanation for using this API")

class AgentManagement(BaseModel):
    hired: List[Agent] = Field(default=[], description="List of hired agents")

class CEOResponse(BaseModel):
    decision: str = Field(..., description="Decision made by the CEO: Hire or Assign_API")
    task_breakdown: List[Subtask] = Field(..., description="List of decomposed subtasks")
    agent_management: AgentManagement = Field(..., description="Details of agent hiring")
    api_utilization: Optional[List[APIUtilization]] = Field(default=None, description="List of utilized APIs, if any")

class OllamaModelManager:
    def __init__(self, model_name="HASHIRU-CEO", system_prompt_file="system.prompt", tools=[]):
        self.model_name = model_name
        # Get the directory of the current script and construct the path to system.prompt
        script_dir = Path(__file__).parent
        self.system_prompt_file = script_dir / system_prompt_file
        self.tools = tools

    def is_model_loaded(self, model):
        loaded_models = [m.model for m in ollama.list().models]
        return model in loaded_models or f'{model}:latest' in loaded_models

    def create_model(self, base_model):
        with open(self.system_prompt_file, 'r', encoding="utf8") as f:
            system = f.read()

        if not self.is_model_loaded(self.model_name):
            print(f"Creating model {self.model_name}")
            ollama.create(
                model=self.model_name,
                from_=base_model,
                system=system,
                parameters={"num_ctx": ModelParameters.NUM_CTX.value, "temperature": ModelParameters.TEMPERATURE.value}
            )

    def request(self, prompt):
        response = ollama.chat(
            model=self.model_name, 
            messages=[{"role": "user", "content": prompt}],
            format=CEOResponse.model_json_schema(),
            tools=self.tools
        )
        response = CEOResponse.model_validate_json(response['message']['content'])
        return response

# Define the web search tool function.
def web_search(website: str, query: str) -> List[str]:
    """
    Searches the specified website for the given query.
    The search query is formed by combining the website domain and the query string.
    """
    search_query = f"site:{website} {query}"
    results = []
    for result in search(search_query, num_results=10):
        # Filter out irrelevant search pages
        if "/search?num=" not in result:
            results.append(result)
    return results

if __name__ == "__main__":
    # Define the tool metadata for orchestration.
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'web_search',
                'description': 'Search for results on a specified website using a query string. '
                               'The CEO model should define which website to search from and the query to use.',
                'parameters': {
                    'type': 'object',
                    'required': ['website', 'query'],
                    'properties': {
                        'website': {'type': 'string', 'description': 'The website domain to search from (e.g., huggingface.co)'},
                        'query': {'type': 'string', 'description': 'The search query to use on the specified website'},
                    },
                },
            },
        }
    ]
    
    # Create the Ollama model manager and ensure the model is set up.
    model_manager = OllamaModelManager()
    model_manager.create_model("mistral")
    
    # Example prompt instructing the CEO model to create a strategy for Ashton Hall.
    # The prompt explicitly mentions that it can use the web_search tool if needed,
    # and that it is allowed to choose the website for the search.
    task_prompt = (
        "Your task is to create a marketing strategy for Ashton Hall, a morning routine creator with 10M followers. "
    )
    
    # Request a CEO response with the prompt.
    response = model_manager.request(task_prompt)
    print("\nCEO Response:")
    print(response)
