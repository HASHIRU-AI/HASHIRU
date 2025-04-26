from src.agent_manager import AgentManager

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

    def run(self, **kwargs):
        
        agent_manger = AgentManager()
        agents = agent_manger.list_agents()

        return {
            "status": "success",
            "message": "Agents list retrieved successfully",
            "agents": agents,
        }
