import importlib

__all__ = ['ToolDeletor']


class ToolDeletor():
    dependencies = ["os"]

    inputSchema = {
        "name": "ToolDeletor",
        "description": "Deletes a tool for the given function",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the tool to create",
                },
                "file_path": {
                    "type": "string",
                    "description": "The path of the tool to create",
                },
            },
            "required": ["name", "file_path"],
        }
    }

    def run(self, **kwargs):
        print("Running Tool Deletor")
        name = kwargs.get("name")
        file_path = kwargs.get("file_path")
        os = importlib.import_module("os")
        os.remove(file_path)
        return {
            "status": "success",
            "message": "Tool deleted successfully",
            "output": {
                "tool_file_path": file_path,
                "tool_name": name,
            }
        }