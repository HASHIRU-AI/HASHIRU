from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional, Tuple
import os
import json
import ollama
from src.manager.utils.singleton import singleton
from src.manager.utils.streamlit_interface import output_assistant_response
from google import genai
from google.genai import types
from google.genai.types import *
from groq import Groq
import os
from dotenv import load_dotenv
from src.manager.budget_manager import BudgetManager

MODEL_PATH = "./src/models/"
MODEL_FILE_PATH = "./src/models/models.json"


class Agent(ABC):

    def __init__(self, agent_name: str,
                 base_model: str,
                 system_prompt: str,
                 create_resource_cost: int,
                 invoke_resource_cost: int,
                 create_expense_cost: int = 0,
                 invoke_expense_cost: int = 0,
                 output_expense_cost: int = 0):
        self.agent_name = agent_name
        self.base_model = base_model
        self.system_prompt = system_prompt
        self.create_resource_cost = create_resource_cost
        self.invoke_resource_cost = invoke_resource_cost
        self.create_expense_cost = create_expense_cost
        self.invoke_expense_cost = invoke_expense_cost
        self.output_expense_cost = output_expense_cost
        self.create_model()

    @abstractmethod
    def create_model(self) -> None:
        """Create and Initialize agent"""
        pass

    @abstractmethod
    def ask_agent(self, prompt: str) -> str:
        """ask agent a question"""
        pass

    @abstractmethod
    def delete_agent(self) -> None:
        """delete agent"""
        pass

    @abstractmethod
    def get_type(self) -> None:
        """get agent type"""
        pass

    def get_costs(self):
        return {
            "create_resource_cost": self.create_resource_cost,
            "invoke_resource_cost": self.invoke_resource_cost,
            "create_expense_cost": self.create_expense_cost,
            "invoke_expense_cost": self.invoke_expense_cost,
            "output_expense_cost": self.output_expense_cost,
        }


class OllamaAgent(Agent):
    type = "local"

    def create_model(self):
        ollama_response = ollama.create(
            model=self.agent_name,
            from_=self.base_model,
            system=self.system_prompt,
            stream=False
        )

    def ask_agent(self, prompt):
        output_assistant_response(f"Asked Agent {self.agent_name} a question")
        agent_response = ollama.chat(
            model=self.agent_name,
            messages=[{"role": "user", "content": prompt}],
        )
        output_assistant_response(
            f"Agent {self.agent_name} answered with {agent_response.message.content}")
        return agent_response.message.content

    def delete_agent(self):
        ollama.delete(self.agent_name)

    def get_type(self):
        return self.type


class GeminiAgent(Agent):
    type = "cloud"

    def __init__(self,
                 agent_name: str,
                 base_model: str,
                 system_prompt: str,
                 create_resource_cost: int,
                 invoke_resource_cost: int,
                 create_expense_cost: int = 0,
                 invoke_expense_cost: int = 0,
                 output_expense_cost: int = 0):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required for Gemini models. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        # Initialize the Gemini API
        self.client = genai.Client(api_key=self.api_key)
        self.chat = self.client.chats.create(model=base_model)

        # Call parent constructor after API setup
        super().__init__(agent_name,
                         base_model,
                         system_prompt,
                         create_resource_cost,
                         invoke_resource_cost,
                         create_expense_cost,
                         invoke_expense_cost,
                         output_expense_cost)

    def create_model(self):
        self.messages = []

    def ask_agent(self, prompt):
        response = self.chat.send_message(
            message=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
            )
        )
        return response.text

    def delete_agent(self):
        self.messages = []

    def get_type(self):
        return self.type


class GroqAgent(Agent):
    type = "cloud"

    def __init__(
        self,
        agent_name: str,
        base_model: str,
        system_prompt: str,
        create_resource_cost: int,
        invoke_resource_cost: int,
        create_expense_cost: int = 0,
        invoke_expense_cost: int = 0,
        output_expense_cost: int = 0
    ):
        # Call the parent class constructor first
        super().__init__(agent_name, base_model, system_prompt,
                         create_resource_cost, invoke_resource_cost,
                         create_expense_cost, invoke_expense_cost,
                         output_expense_cost)

        # Groq-specific API client setup
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your .env file or environment.")
        self.client = Groq(api_key=api_key)

        if self.base_model and "groq-" in self.base_model:
            self.groq_api_model_name = self.base_model.split("groq-", 1)[1]
        else:
            # Fallback or error if the naming convention isn't followed.
            # This ensures that if a non-prefixed model name is somehow passed,
            # it might still work, or you can raise an error.
            self.groq_api_model_name = self.base_model
            print(f"Warning: GroqAgent base_model '{self.base_model}' does not follow 'groq-' prefix convention.")

    def create_model(self) -> None:
        """
        Create and Initialize agent.
        For Groq, models are pre-existing on their cloud.
        This method is called by Agent's __init__.
        """
        pass

    def ask_agent(self, prompt: str) -> str:
        """Ask agent a question"""
        if not self.client:
            raise ConnectionError("Groq client not initialized. Check API key and constructor.")
        if not self.groq_api_model_name:
            raise ValueError("Groq API model name not set. Check base_model configuration.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.groq_api_model_name, # Use the derived model name for Groq API
            )
            result = response.choices[0].message.content
            return result
        except Exception as e:
            # Handle API errors or other exceptions during the call
            print(f"Error calling Groq API: {e}")
            raise  # Re-raise the exception or handle it as appropriate

    def delete_agent(self) -> None:
        """Delete agent"""
        pass

    def get_type(self) -> str: # Ensure return type hint matches Agent ABC
        """Get agent type"""
        return self.type

@singleton
class AgentManager():
    budget_manager: BudgetManager = BudgetManager()
    is_creation_enabled: bool = True
    is_cloud_invocation_enabled: bool = True
    is_local_invocation_enabled: bool = True

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._agent_types = {
            "ollama": OllamaAgent,
            "gemini": GeminiAgent,
            "groq": GroqAgent,
        }

        self._load_agents()

    def set_creation_mode(self, status: bool):
        self.is_creation_enabled = status
        if status:
            output_assistant_response("Agent creation mode is enabled.")
        else:
            output_assistant_response("Agent creation mode is disabled.")

    def set_cloud_invocation_mode(self, status: bool):
        self.is_cloud_invocation_enabled = status
        if status:
            output_assistant_response("Cloud invocation mode is enabled.")
        else:
            output_assistant_response("Cloud invocation mode is disabled.")

    def set_local_invocation_mode(self, status: bool):
        self.is_local_invocation_enabled = status
        if status:
            output_assistant_response("Local invocation mode is enabled.")
        else:
            output_assistant_response("Local invocation mode is disabled.")

    def create_agent(self, agent_name: str,
                     base_model: str, system_prompt: str,
                     description: str = "", create_resource_cost: float = 0,
                     invoke_resource_cost: float = 0,
                     create_expense_cost: float = 0,
                     invoke_expense_cost: float = 0,
                     output_expense_cost: float = 0,
                     **additional_params) -> Tuple[Agent, int]:
        if not self.is_creation_enabled:
            raise ValueError("Agent creation mode is disabled.")

        if agent_name in self._agents:
            raise ValueError(f"Agent {agent_name} already exists")

        self._agents[agent_name] = self.create_agent_class(
            agent_name,
            base_model,
            system_prompt,
            description=description,
            create_resource_cost=create_resource_cost,
            invoke_resource_cost=invoke_resource_cost,
            create_expense_cost=create_expense_cost,
            invoke_expense_cost=invoke_expense_cost,
            output_expense_cost=output_expense_cost,
            **additional_params  # For any future parameters we might want to add
        )

        # save agent to file
        self._save_agent(
            agent_name,
            base_model,
            system_prompt,
            description=description,
            create_resource_cost=create_resource_cost,
            invoke_resource_cost=invoke_resource_cost,
            create_expense_cost=create_expense_cost,
            invoke_expense_cost=invoke_expense_cost,
            output_expense_cost=output_expense_cost,
            **additional_params  # For any future parameters we might want to add
        )
        return (self._agents[agent_name],
                self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    def validate_budget(self,
                        resource_cost: float = 0,
                        expense_cost: float = 0) -> None:
        if not self.budget_manager.can_spend_resource(resource_cost):
            raise ValueError(f"Do not have enough resource budget to create/use the agent. "
                             + f"Creating/Using the agent costs {resource_cost} but only {self.budget_manager.get_current_remaining_resource_budget()} is remaining")
        if not self.budget_manager.can_spend_expense(expense_cost):
            raise ValueError(f"Do not have enough expense budget to create/use the agent. "
                             + f"Creating/Using the agent costs {expense_cost} but only {self.budget_manager.get_current_remaining_expense_budget()} is remaining")

    def create_agent_class(self,
                           agent_name: str,
                           base_model: str,
                           system_prompt: str,
                           description: str = "",
                           create_resource_cost: float = 0,
                           invoke_resource_cost: float = 0,
                           create_expense_cost: float = 0,
                           invoke_expense_cost: float = 0,
                           output_expense_cost: float = 0,
                           **additional_params) -> Agent:
        agent_type = self._get_agent_type(base_model)
        agent_class = self._agent_types.get(agent_type)

        if not agent_class:
            raise ValueError(f"Unsupported base model {base_model}")

        created_agent = agent_class(agent_name,
                                    base_model,
                                    system_prompt,
                                    create_resource_cost,
                                    invoke_resource_cost,
                                    create_expense_cost,
                                    invoke_expense_cost,
                                    output_expense_cost,)

        self.validate_budget(create_resource_cost,
                             create_expense_cost)

        self.budget_manager.add_to_resource_budget(create_resource_cost)
        self.budget_manager.add_to_expense_budget(create_expense_cost)
        # create agent
        return created_agent

    def get_agent(self, agent_name: str) -> Agent:
        """Get existing agent by name"""
        if agent_name not in self._agents:
            raise ValueError(f"Agent {agent_name} does not exists")
        return self._agents[agent_name]

    def list_agents(self) -> dict:
        """Return agent information (name, description, costs)"""
        try:
            if os.path.exists(MODEL_FILE_PATH):
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    full_models = json.loads(f.read())

                # Create a simplified version with only the description and costs
                simplified_agents = {}
                for name, data in full_models.items():
                    simplified_agents[name] = {
                        "description": data.get("description", ""),
                        "create_resource_cost": data.get("create_resource_cost", 0),
                        "invoke_resource_cost": data.get("invoke_resource_cost", 0),
                        "create_expense_cost": data.get("create_expense_cost", 0),
                        "invoke_expense_cost": data.get("invoke_expense_cost", 0),
                        "base_model": data.get("base_model", ""),
                    }
                return simplified_agents
            else:
                return {}
        except Exception as e:
            output_assistant_response(f"Error listing agents: {e}")
            return {}

    def delete_agent(self, agent_name: str) -> int:
        agent: Agent = self.get_agent(agent_name)

        self.budget_manager.remove_from_resource_expense(
            agent.create_resource_cost)
        agent.delete_agent()

        del self._agents[agent_name]
        try:
            if os.path.exists(MODEL_FILE_PATH):
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    models = json.loads(f.read())

                del models[agent_name]
                with open(MODEL_FILE_PATH, "w", encoding="utf8") as f:
                    f.write(json.dumps(models, indent=4))
        except Exception as e:
            output_assistant_response(f"Error deleting agent: {e}")
        return (self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    def ask_agent(self, agent_name: str, prompt: str) -> Tuple[str, int]:
        agent: Agent = self.get_agent(agent_name)
        print(agent.get_type())
        print(agent_name)
        print(self.is_local_invocation_enabled,
              self.is_cloud_invocation_enabled)
        if not self.is_local_invocation_enabled and agent.get_type() == "local":
            raise ValueError("Local invocation mode is disabled.")

        if not self.is_cloud_invocation_enabled and agent.get_type() == "cloud":
            raise ValueError("Cloud invocation mode is disabled.")

        n_tokens = len(prompt.split())/1000000

        self.validate_budget(agent.invoke_resource_cost,
                             agent.invoke_expense_cost*n_tokens)

        self.budget_manager.add_to_expense_budget(
            agent.invoke_expense_cost*n_tokens)

        response = agent.ask_agent(prompt)
        n_tokens = len(response.split())/1000000
        self.budget_manager.add_to_expense_budget(
            agent.output_expense_cost*n_tokens)
        return (response,
                self.budget_manager.get_current_remaining_resource_budget(),
                self.budget_manager.get_current_remaining_expense_budget())

    def _save_agent(self,
                    agent_name: str,
                    base_model: str,
                    system_prompt: str,
                    description: str = "",
                    create_resource_cost: float = 0,
                    invoke_resource_cost: float = 0,
                    create_expense_cost: float = 0,
                    invoke_expense_cost: float = 0,
                    output_expense_cost: float = 0,
                    **additional_params) -> None:
        """Save a single agent to the models.json file"""
        try:
            # Ensure the directory exists
            os.makedirs(MODEL_PATH, exist_ok=True)

            # Read existing models file or create empty dict if it doesn't exist
            try:
                with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                    models = json.loads(f.read())
            except (FileNotFoundError, json.JSONDecodeError):
                models = {}

            # Update the models dict with the new agent
            models[agent_name] = {
                "base_model": base_model,
                "description": description,
                "system_prompt": system_prompt,
                "create_resource_cost": create_resource_cost,
                "invoke_resource_cost": invoke_resource_cost,
                "create_expense_cost": create_expense_cost,
                "invoke_expense_cost": invoke_expense_cost,
                "output_expense_cost": output_expense_cost,
            }

            # Add any additional parameters that were passed
            for key, value in additional_params.items():
                models[agent_name][key] = value

            # Write the updated models back to the file
            with open(MODEL_FILE_PATH, "w", encoding="utf8") as f:
                f.write(json.dumps(models, indent=4))

        except Exception as e:
            output_assistant_response(f"Error saving agent {agent_name}: {e}")

    def _get_agent_type(self, base_model) -> str:

        if base_model == "llama3.2":
            return "ollama"
        elif base_model == "mistral":
            return "ollama"
        elif base_model == "deepseek-r1":
            return "ollama"
        elif "gemini" in base_model:
            return "gemini"
        elif "groq" in base_model:
            return "groq"
        else:
            return "unknown"

    def _load_agents(self) -> None:
        """Load agent configurations from disk"""
        try:
            if not os.path.exists(MODEL_FILE_PATH):
                return

            with open(MODEL_FILE_PATH, "r", encoding="utf8") as f:
                models = json.loads(f.read())

            for name, data in models.items():
                if name in self._agents:
                    continue
                base_model = data["base_model"]
                system_prompt = data["system_prompt"]
                create_resource_cost = data.get("create_resource_cost", 0)
                invoke_resource_cost = data.get("invoke_resource_cost", 0)
                create_expense_cost = data.get("create_expense_cost", 0)
                invoke_expense_cost = data.get("invoke_expense_cost", 0)
                output_expense_cost = data.get("output_expense_cost", 0)
                model_type = self._get_agent_type(base_model)
                manager_class = self._agent_types.get(model_type)

                if manager_class:
                    # Create the agent with the appropriate manager class
                    self._agents[name] = self.create_agent_class(
                        name,
                        base_model,
                        system_prompt,
                        description=data.get("description", ""),
                        create_resource_cost=create_resource_cost,
                        invoke_resource_cost=invoke_resource_cost,
                        create_expense_cost=create_expense_cost,
                        invoke_expense_cost=invoke_expense_cost,
                        output_expense_cost=output_expense_cost,
                        **data.get("additional_params", {})
                    )
                    self._agents[name] = manager_class(
                        name,
                        base_model,
                        system_prompt,
                        create_resource_cost,
                        invoke_resource_cost,
                        create_expense_cost,
                        invoke_expense_cost,
                        output_expense_cost
                    )
        except Exception as e:
            output_assistant_response(f"Error loading agents: {e}")
