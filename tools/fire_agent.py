from src.budget_manager import BudgetManager  
from src.agent_manager import AgentManager

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

    def run(self, **kwargs):
        print("Running Fire Agent")
        agent_name= kwargs.get("agent_name")

        agent_manager = AgentManager()
        
        agent = agent_manager.get_agent(agent_name=agent_name)
        
        agent_costs = agent.get_costs()
        
        try:
            agent_manager.delete_agent(agent_name=agent_name)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }
            
        budget_manager = BudgetManager()
        
        budget_manager.add_to_expense(-1* int(agent_costs["create_cost"]))

        return {
            "status": "success",
            "message": "Agent successfully fired.",
            "current_expense": budget_manager.get_current_expense()
        }