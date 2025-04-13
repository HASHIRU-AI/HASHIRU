import importlib
from src.budget_manager import BudgetManager  

__all__ = ['FireAgent']

class FireAgent():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "FireAgent",
        "description": "Fires an AI agent for you.",
        "parameters": {
            "type": "object",
            "properties":{
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be fired. This name cannot have spaces or special characters. It should be a single word.",
                },
            },
            "required": ["agent_name"],
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
        agent_name= kwargs.get("agent_name")
        ollama = importlib.import_module("ollama")
        json = importlib.import_module("json")

        if not self.does_agent_exist(agent_name):
            return {
                "status": "error",
                "message": "Agent does not exists",
                "output": None
            }
        ollama_response = ollama.delete(agent_name)
        budget_manager = BudgetManager()
        
        with open("./models/models.json", "r", encoding="utf8") as f:
            models = f.read()
        models = json.loads(models)
        budget_manager.add_to_expense(-1* int(models[agent_name]["create_cost"]))
        del models[agent_name]
        with open("./models/models.json", "w", encoding="utf8") as f:
            f.write(json.dumps(models, indent=4))

        if "success" in ollama_response["status"]:
            return {
                "status": "success",
                "message": "Agent successfully fired.",
                "current_expense": budget_manager.get_current_expense()
            }
        else:
            return {
                "status": "error",
                "message": "Agent firing failed",
            }