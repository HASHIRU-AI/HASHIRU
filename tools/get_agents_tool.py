import importlib
import json

__all__ = ['GetAgents']

class GetAgents():
    dependencies = []

    inputSchema = {
        "name": "GetAgents",
        "description": "Retrieves a list of available AI agents. This tool is used to get the list of available models that can be invoked using the AskAgent tool.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        with open("./models/models.json", "r", encoding="utf8") as f:
            models = f.read()
        models = json.loads(models)
        return {
            "status": "success",
            "message": "Agents list retrieved successfully",
            "agents": models,
        }
