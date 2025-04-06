import importlib

__all__ = ['AgentCreator']

class AgentCreator():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "AgentCreator",
        "description": "Creates an AI agent for an given AI model with a given system prompt",
        "parameters": {
            "type": "object",
            "properties":{
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be created"
                },
                "base_model": {
                    "type": "string",
                    "description": "A base model from which the new agent mode is to be created"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "A string containing the system prompt for the AI agent"
                }
            }
        }
    }

    def __init__(self):
        pass
    
    def does_agent_exist(self, agent_name):
        ollama = importlib.import_module("ollama")
        all_agents = [a.model for a in ollama.list().models]
        if agent_name in all_agents or f'{agent_name}:latest' in all_agents:
            return True 
        
        return False

    def run(self, **kwargs):
        print("Running Agent Creator")
        agent_name= kwargs.get("agent_name")
        base_model = kwargs.get("base_model")
        system_prompt = kwargs.get("system_prompt")
        ollama = importlib.import_module("ollama")

        if self.does_agent_exist(agent_name):
            return {
                "status": "error",
                "message": "Agent already exists",
                "output": None
            }
        ollama_response = ollama.create(
            model = agent_name,
            from_ = base_model,
            system = system_prompt,
            stream = False
        )

        if "success" in ollama_response["status"]:
            return {
                "status": "success",
                "message": "Agent successfully created",
                "output": None
            }
        else:
            return {
                "status": "error",
                "message": "Agent creation failed",
                "output": None
            }