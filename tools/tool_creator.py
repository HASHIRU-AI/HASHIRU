import importlib

__all__ = ['ToolCreator']


class ToolCreator():
    dependencies = ["os"]

    inputSchema = {
        "name": "ToolCreator",
        "description": "Creates a tool for the given function",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the tool to create",
                },
                "tool_code": {
                    "type": "string",
                    "description": "The code of the tool to create",
                },
            },
            "required": ["name", "content"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        print("Running Tool Creator")
        name = kwargs.get("name")
        content = kwargs.get("tool_code")
        print(f"Tool Name: {name}")
        print(f"Tool Content: {content}")
        # Create the tool file
        tool_file_path = f"tools/{name}.py"
        with open(tool_file_path, "w") as tool_file:
            tool_file.write(content)
        print(f"Tool file created at {tool_file_path}")
        return {
            "status": "success",
            "message": "Tool created successfully",
            "output": {
                "tool_file_path": tool_file_path,
                "tool_name": name,
            }
        }