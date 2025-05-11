from src.manager.budget_manager import BudgetManager
from src.manager.agent_manager import AgentManager

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

        try:
            agent_response, remaining_budget = agent_manger.ask_agent(agent_name=agent_name, prompt=prompt)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Error occurred: {str(e)}",
                "output": None
            }

        print("Agent response", agent_response)
        return {
            "status": "success",
            "message": "Agent has replied to the given prompt",
            "output": agent_response,
            "remaining_budget": remaining_budget
        }
