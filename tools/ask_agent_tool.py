import importlib

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

    def does_agent_exist(self, ollama, agent_name):
        all_agents = [a.model for a in ollama.list().models]
        if agent_name in all_agents or f'{agent_name}:latest' in all_agents:
            return True

        return False

    def run(self, **kwargs):
        print("Asking agent a question")

        agent_name = kwargs.get("agent_name")
        prompt = kwargs.get("prompt")

        ollama = importlib.import_module("ollama")
        if not self.does_agent_exist(ollama, agent_name):
            print("Agent does not exist")
            return {
                "status": "error",
                "message": "Agent does not exists",
                "output": None
            }

        agent_response = ollama.chat(
            model=agent_name,
            messages=[{"role": "user", "content": prompt}],
        )
        print("Agent response", agent_response.message.content)
        return {
            "status": "success",
            "message": "Agent has replied to the given prompt",
            "output": agent_response.message.content,
        }
