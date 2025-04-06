from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import ollama

from CEO.ask_user import AskUser
from CEO.tool_loader import ToolLoader

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
    def __init__(self, toolsLoader: ToolLoader, model_name="HASHIRU-CEO", system_prompt_file="./models/system.prompt"):
        self.model_name = model_name
        # Get the directory of the current script and construct the path to system.prompt
        script_dir = Path(__file__).parent
        self.system_prompt_file = system_prompt_file
        self.toolsLoader = toolsLoader
        self.toolsLoader.load_tools()
        self.create_model(model_name)

    def is_model_loaded(self, model):
        loaded_models = [m.model for m in ollama.list().models]
        return model in loaded_models or f'{model}:latest' in loaded_models

    def create_model(self, base_model='llama3.2'):
        with open(self.system_prompt_file, 'r', encoding="utf8") as f:
            system = f.read()

        if not self.is_model_loaded(self.model_name):
            print(f"Creating model {self.model_name}")
            ollama.create(
                model=self.model_name,
                from_='mistral',
                system=system,
                parameters={"num_ctx": ModelParameters.NUM_CTX.value, "temperature": ModelParameters.TEMPERATURE.value}
            )

    def request(self, messages):
        print(f"messages: {messages}")
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            # format=CEOResponse.model_json_schema(),
            tools=self.toolsLoader.getTools(),
        )
        # response = CEOResponse.model_validate_json(response['message']['content'])
        if "EOF" in response.message.content:
            return messages
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                print(f"Tool Name: {tool_call.function.name}, Arguments: {tool_call.function.arguments}")
                toolResponse = self.toolsLoader.runTool(tool_call.function.name, tool_call.function.arguments)
                print(f"Tool Response: {toolResponse}")
                role = "tool"
                if "role" in toolResponse:
                    role = toolResponse["role"]
                messages.append({"role": role, "content": str(toolResponse)})
            self.request(messages)
        else:
            print("No tool calls found in the response.")
            messages.append({"role": "assistant", "content": response.message.content})
            print(f"Messages: {messages}")
            ask_user_tool = AskUser()
            ask_user_response = ask_user_tool.run(prompt=response.message.content)
            messages.append({"role": "user", "content": ask_user_response})
            self.request(messages)
            # return messages