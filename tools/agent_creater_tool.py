import importlib

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
                    "description": "A base model from which the new agent mode is to be created. Available models are: llama3.2, mistral"
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
        },
        "creates": {
            "selector": "base_model",
            "types": {
                "llama3.2":{
                    "description": "3 Billion parameter model",
                    "create_cost": 10,
                    "invoke_cost": 20,
                },
                "mistral":{
                    "description": "7 Billion parameter model",
                    "create_cost": 20,
                    "invoke_cost": 50,
                }
            }
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
            try:
                with open("./models/models.json", "r", encoding="utf8") as f:
                    models = f.read()
                models = json.loads(models)
                models[agent_name] = {
                    "base_model": base_model,
                    "description": kwargs.get("description"),
                    "create_cost": self.inputSchema["creates"]["types"][base_model]["create_cost"],
                    "invoke_cost": self.inputSchema["creates"]["types"][base_model]["invoke_cost"],
                }
                with open("./models/models.json", "w", encoding="utf8") as f:
                    f.write(json.dumps(models, indent=4))

                return {
                    "status": "success",
                    "message": "Agent successfully created",
                    "cost": self.inputSchema["creates"]["types"][base_model]["create_cost"],
                }
            except Exception as e:
                print("Error while writing to models.json", e)
                return {
                    "status": "error",
                    "message": f"Agent creation failed: {e}",
                }
        else:
            return {
                "status": "error",
                "message": "Agent creation failed",
            }