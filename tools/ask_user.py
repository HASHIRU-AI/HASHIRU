import importlib

__all__ = ['AskUser']


class AskUser():
    dependencies = []

    inputSchema = {
        "name": "AskUser",
        "description": "Asks a question to the user and gets a response. Only use this when you need more clarification from the user on the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
            },
            "required": ["question"],
        }
    }

    def run(self, **kwargs):
        print("Running Ask User tool")
        question = kwargs.get("question")
        print(f"Question: {question}")
        output = input(question)
        return {
            "status": "success",
            "message": "Ask User tool executed successfully",
            "output": output,
            "role": "user",
        }
