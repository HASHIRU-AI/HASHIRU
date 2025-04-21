from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import os
import json
import ollama
from src.singleton import singleton

class Agent(ABC):
    
    def __init__(self, agent_name: str, base_model: str, system_prompt: str):
        self.agent_name = agent_name
        self.base_model = base_model
        self.system_prompt = system_prompt
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
    def delete_agent(self) ->None:
        """delete agent"""
        pass
    
class OllamaAgent(Agent):
    
    def create_model(self):
        ollama_response = ollama.create(
            model = self.agent_name,
            from_ = self.base_model,
            system = self.system_prompt,
            stream = False
        )
    
    def ask_agent(self, prompt):
        print(f"Asked Agent {self.agent_name} a question")
        agent_response = ollama.chat(
            model=self.agent_name,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"Agent {self.agent_name} answered with {agent_response.message.content}")
        return agent_response.message.content
    
    def delete_agent(self):
        ollama.delete(self.agent_name)
    
@singleton
class AgentManager():
    
    def __init__(self):
        self._agents = {}
        self._agent_types ={
            ollama: OllamaAgent
        }
        
        self._load_agents()
    
    def create_agent(self, agent_name: str, base_model: str, system_prompt: str, description: str = "", create_cost: float = 0, invoke_cost: float = 0, 
                 **additional_params) -> Agent:
        
        if agent_name in self._agents:
            raise ValueError(f"Agent {agent_name} already exists")
        
        agent_type = self._get_agent_type(base_model)
        agent_class = self._agent_types.get(agent_type)
        
        if not agent_class:
            raise ValueError(f"Unsupported base model {base_model}")
        
        # create agent
        self._agents[agent_name] = agent_class(agent_name, base_model, system_prompt)
        
        #save agent to file
        self._save_agent(
            agent_name, 
            base_model, 
            system_prompt, 
            description=description,
            create_cost=create_cost,
            invoke_cost=invoke_cost,
            **additional_params  # For any future parameters we might want to add
        )
        
        return self._agents[agent_name]
    
    def get_agent(self, agent_name: str) -> Agent:
        """Get existing agent by name"""
        if agent_name in self._agents:
            raise ValueError(f"Agent {agent_name} already exists")
        return self._agents[agent_name]
        
    def list_agents(self) -> dict:
        """Return agent information (name, description, costs)"""
        try:
            if os.path.exists("./models/models.json"):
                with open("./models/models.json", "r", encoding="utf8") as f:
                    full_models = json.loads(f.read())
                    
                # Create a simplified version with only the description and costs
                simplified_agents = {}
                for name, data in full_models.items():
                    simplified_agents[name] = {
                        "description": data.get("description", ""),
                        "create_cost": data.get("create_cost", 0),
                        "invoke_cost": data.get("invoke_cost", 0)
                    }
                return simplified_agents
            else:
                return {}
        except Exception as e:
            print(f"Error listing agents: {e}")
            return {}
    
    def delete_agent(self, agent_name: str) -> None:
        if agent_name not in self._agents:
            raise ValueError(f"Agent {agent_name} does not exist")
        self._agents[agent_name].delete_agent()
        
        del self._agents[agent_name]
        try:
            if os.path.exists("./models/models.json"):
                with open("./models/models.json", "r", encoding="utf8") as f:
                    models = json.loads(f.read())
                    
                del models[agent_name]
                with open("./models/models.json", "w", encoding="utf8") as f:
                    f.write(json.dumps(models, indent=4))
        except Exception as e:
            print(f"Error deleting agent: {e}")
    
    
    def _save_agent(self, agent_name: str, base_model: str, system_prompt: str, 
                    description: str = "", create_cost: float = 0, invoke_cost: float = 0, 
                    **additional_params) -> None:
        """Save a single agent to the models.json file"""
        try:
            # Ensure the directory exists
            os.makedirs("./models", exist_ok=True)
            
            # Read existing models file or create empty dict if it doesn't exist
            try:
                with open("./models/models.json", "r", encoding="utf8") as f:
                    models = json.loads(f.read())
            except (FileNotFoundError, json.JSONDecodeError):
                models = {}
            
            # Update the models dict with the new agent
            models[agent_name] = {
                "base_model": base_model,
                "description": description,
                "system_prompt": system_prompt,
                "create_cost": create_cost,
                "invoke_cost": invoke_cost,
            }
            
            # Add any additional parameters that were passed
            for key, value in additional_params.items():
                models[agent_name][key] = value
            
            # Write the updated models back to the file
            with open("./models/models.json", "w", encoding="utf8") as f:
                f.write(json.dumps(models, indent=4))
                
        except Exception as e:
            print(f"Error saving agent {agent_name}: {e}")

    def _get_agent_type(self, base_model)->str:
        if base_model.startswith("ollama"):
            return "ollama"
        else:
            return "unknown"
    
    def _load_agents(self) -> None:
        """Load agent configurations from disk"""
        try:
            if not os.path.exists("./models/models.json"):
                return
                
            with open("./models/models.json", "r", encoding="utf8") as f:
                models = json.loads(f.read())
            
            for name, data in models.items():
                base_model = data["base_model"]
                system_prompt = data["system_prompt"]
                
                model_type = self._get_model_type(base_model)
                manager_class = self._model_types.get(model_type)
                
                if manager_class:
                    # Create the agent with the appropriate manager class
                    self._agents[name] = manager_class(
                        name, 
                        base_model,
                        system_prompt
                    )
        except Exception as e:
            print(f"Error loading agents: {e}")