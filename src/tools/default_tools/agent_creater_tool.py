from src.manager.agent_manager import AgentManager
from src.tools.default_tools.agent_cost_manager import AgentCostManager
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
                    "description": "A base model from which the new agent mode is to be created. Check the available models using the AgentCostManager tool.",
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


    def run(self, **kwargs):
        print("Running Agent Creator")
        agent_name = kwargs.get("agent_name")
        base_model = kwargs.get("base_model")

        print(f"[DEBUG] Selected Model: {base_model}")

        system_prompt = kwargs.get("system_prompt")
        description = kwargs.get("description")
        model_costs = AgentCostManager().get_costs()
        create_cost = model_costs[base_model]["create_cost"]
        if base_model not in model_costs:
            print(f"[WARN] Auto-selected model '{base_model}' not in schema. Falling back to gemini-2.0-flash")
            base_model = "gemini-2.0-flash"
        invoke_cost = model_costs[base_model]["invoke_cost"]

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