from src.budget_manager import BudgetManager
from src.agent_manager import AgentManager

__all__ = ['AskAgent']


class AskAgent():
    dependencies = ["ollama==0.4.7",
                    "pydantic==2.11.1",
                    "pydantic_core==2.33.0"]

    inputSchema = {
        "name": "AskAgent",
        "description": "Asks an AI agent a question and gets a response. The agent must be created using the AgentCreator tool before using this tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the AI agent that is to be asked a question. This name cannot have spaces or special characters. It should be a single word.",
                },
                "prompt": {
                    "type": "string",
                    "description": "This is the prompt that will be used to ask the agent a question. It should be a string that describes the question to be asked.",
                }
            },
            "required": ["agent_name", "prompt"],
        }
    }

    def run(self, **kwargs):
        print("Asking agent a question")

        agent_name = kwargs.get("agent_name")
        prompt = kwargs.get("prompt")
        agent_manger = AgentManager()
        budget_manager = BudgetManager()

        try:
            agent = agent_manger.get_agent(agent_name=agent_name)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }

        agent_costs = agent.get_costs()
        agent_question_cost = agent_costs["invoke_cost"]
        print("Agent question cost", agent_question_cost)
        
        if not budget_manager.can_spend(agent_question_cost):
            return {
                "status": "error",
                "message": f"Do not have enough budget to ask the agent a question. Asking the agent costs {agent_question_cost} but only {budget_manager.get_current_remaining_budget()} is remaining",
                "output": None
            }

        agent_response = agent.ask_agent(prompt=prompt)
        print("Agent response", agent_response)
        return {
            "status": "success",
            "message": "Agent has replied to the given prompt",
            "output": agent_response,
        }
