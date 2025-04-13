import importlib
from CEO.budget_manager import BudgetManager  
__all__ = ['AgentCreator']

class AgentCreator():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "AgentCreator",
        "description": "Creates an AI agent for you. Please make sure to invoke the created agent using the AskAgent tool.",
        "parameters": {
            "type": "object",
            "properties":{
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be created. This name cannot have spaces or special characters. It should be a single word.",
                },
                "base_model": {
                    "type": "string",
                    "description": "A base model from which the new agent mode is to be created. Available models are: llama3.2"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "This is the system prompt that will be used to create the agent. It should be a string that describes the role of the agent and its capabilities."
                },
                "description": {
                    "type": "string",
                    "description": "Description of the agent. This is a string that describes the agent and its capabilities. It should be a single line description.",
                },
            },
            "required": ["agent_name", "base_model", "system_prompt", "description"],
        }
    }
    
    def does_agent_exist(self, agent_name):
        ollama = importlib.import_module("ollama")
        all_agents = [a.model for a in ollama.list().models]
        if agent_name in all_agents or f'{agent_name}:latest' in all_agents:
            return True 
        
        return False

    def run(self, **kwargs):
        print("Running Agent Creator")
        agent_name = kwargs.get("agent_name")
        base_model = kwargs.get("base_model")
        system_prompt = kwargs.get("system_prompt")
        ollama = importlib.import_module("ollama")
        json = importlib.import_module("json")

        agent_creation_cost = 60
        budget_manager = BudgetManager()
        print("budget_manager_instance:", budget_manager)
        if not budget_manager.can_spend(agent_creation_cost):
            return {
                "status": "error",
                "message": f"Could not create {agent_name}. Creating the agent costs {agent_creation_cost} but only {budget_manager.get_current_remaining_budget()} is remaining",
                "output": None
            }
        if self.does_agent_exist(agent_name):
            return {
                "status": "error",
                "message": "Agent already exists",
                "output": None
            }
        
        budget_manager.add_to_expense(agent_creation_cost)
        ollama_response = ollama.create(
            model = agent_name,
            from_ = base_model,
            system = system_prompt,
            stream = False
        )
        
        with open("./models/models.json", "r", encoding="utf8") as f:
            models = f.read()
        models = json.loads(models)
        models[agent_name] = {
            "base_model": base_model,
            "description": kwargs.get("description"),
            "creation_cost": agent_creation_cost
        }
        with open("./models/models.json", "w", encoding="utf8") as f:
            f.write(json.dumps(models, indent=4))

        if "success" in ollama_response["status"]:
            return {
                "status": "success",
                "message": "Agent successfully created",
                "current_expense": budget_manager.get_current_expense()
            }
        else:
            return {
                "status": "error",
                "message": "Agent creation failed",
            }