from src.agent_manager import AgentManager
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
                    "description": "A base model from which the new agent mode is to be created. Available models are: llama3.2, mistral, gemini-2.5-flash-preview-04-17, gemini-2.5-pro-preview-03-25, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro, and gemini-2.0-flash-live-001"
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
                },
                "gemini-2.5-flash-preview-04-17": {
                    "description": "Adaptive thinking, cost efficiency",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-2.5-pro-preview-03-25": {
                    "description": "Enhanced thinking and reasoning, multimodal understanding, advanced coding, and more",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-2.0-flash": {
                    "description": "Next generation features, speed, thinking, realtime streaming, and multimodal generation",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-2.0-flash-lite": {
                    "description": "Cost efficiency and low latency",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-1.5-flash": {
                    "description": "Fast and versatile performance across a diverse variety of tasks",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-1.5-flash-8b": {
                    "description": "High volume and lower intelligence tasks",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                "gemini-1.5-pro": {
                    "description": "Complex reasoning tasks requiring more intelligence",
                    "create_cost": 20,
                    "invoke_cost": 50
                },
                # "gemini-embedding-exp": {
                #     "description": "Measuring the relatedness of text strings",
                #     "create_cost": 20,
                #     "invoke_cost": 50
                # },
                # "imagen-3.0-generate-002": {
                #     "description": "Our most advanced image generation model",
                #     "create_cost": 20,
                #     "invoke_cost": 50
                # },
                # "veo-2.0-generate-001": {
                #     "description": "High quality video generation",
                #     "create_cost": 20,
                #     "invoke_cost": 50
                # },
                "gemini-2.0-flash-live-001": {
                    "description": "Low-latency bidirectional voice and video interactions",
                    "create_cost": 20,
                    "invoke_cost": 50
                }
            }
        }
    }


    def run(self, **kwargs):
        print("Running Agent Creator")
        agent_name = kwargs.get("agent_name")
        base_model = kwargs.get("base_model")
        system_prompt = kwargs.get("system_prompt")
        description = kwargs.get("description")
        create_cost = self.inputSchema["creates"]["types"][base_model]["create_cost"]
        invoke_cost = self.inputSchema["creates"]["types"][base_model]["invoke_cost"]

        agent_manager = AgentManager()
        try:
            _, remaining_budget = agent_manager.create_agent(
                agent_name=agent_name,
                base_model=base_model,
                system_prompt=system_prompt,
                description=description,
                create_cost=create_cost,
                invoke_cost=invoke_cost
            )
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }
        
        return {
            "status": "success",
            "message": "Agent successfully created",
            "remaining_budget": remaining_budget,
        }