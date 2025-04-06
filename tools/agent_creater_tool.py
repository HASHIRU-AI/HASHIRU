import importlib

__all__ = ['AgentCreator']

class AgentCreator():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "AgentCreator",
        "description": "Creates an AI agent using the ollama library. Before creating an Agent, please get the list of available models using the GetAgents tool. Once the model is created, you can use the AskAgent tool to ask the agent a question.",
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
        json = importlib.import_module("json")

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

        with open("./models/models.json", "r", encoding="utf8") as f:
            models = f.read()
        models = json.loads(models)
        models[agent_name] = {
            "base_model": base_model,
            "description": kwargs.get("description")
        }
        with open("./models/models.json", "w", encoding="utf8") as f:
            f.write(json.dumps(models, indent=4))

        if "success" in ollama_response["status"]:
            return {
                "status": "success",
                "message": "Agent successfully created",
                "output": models
            }
        else:
            return {
                "status": "error",
                "message": "Agent creation failed",
                "output": models
            }