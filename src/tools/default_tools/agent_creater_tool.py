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
        if base_model not in model_costs:
            return {
                "status": "error",
                "message": f"Model {base_model} not found in the cost manager.",
                "output": None
            }
        create_resource_cost = model_costs[base_model].get("create_resource_cost", 0)
        invoke_resource_cost = model_costs[base_model].get("invoke_resource_cost", 0)
        create_expense_cost = model_costs[base_model].get("create_expense_cost", 0)
        invoke_expense_cost = model_costs[base_model].get("invoke_expense_cost", 0)

        agent_manager = AgentManager()
        try:
            _, remaining_resource_budget, remaining_expense_budget = agent_manager.create_agent(
                agent_name=agent_name,
                base_model=base_model,
                system_prompt=system_prompt,
                description=description,
                create_resource_cost=create_resource_cost,
                invoke_resource_cost=invoke_resource_cost,
                create_expense_cost=create_expense_cost,
                invoke_expense_cost=invoke_expense_cost
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
            "remaining_resource_budget": remaining_resource_budget,
            "remaining_expense_budget": remaining_expense_budget
        }