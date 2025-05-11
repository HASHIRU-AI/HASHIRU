
__all__ = ['UpdateMemory']

from src.manager.budget_manager import BudgetManager


class UpdateMemory():
    dependencies = []

    inputSchema = {
        "name": "UpdateMemory",
        "description": "Updates the memory of the AI agent. This tool is used to update the memory of the AI agent with new information.",
        "parameters": {
            "type": "object",
            "properties":{
                "memory": {
                    "type": "string",
                    "description": "The new memory to be added to the AI agent's memory.",
                }
            },
            "required": ["memory"],
        },
    }


    def run(self, **kwargs):
        return {
            "status": "fail",
            "message": "This tool is not implemented yet",
            "output": None
        }
        