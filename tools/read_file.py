import importlib

__all__ = ['ReadFile']


class ReadFile():
    dependencies = []

    inputSchema = {
        "name": "ReadFile",
        "description": "Reads a file and returns its content",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
            },
            "required": ["file_path"],
        }
    }

    def __init__(self):
        pass

    def run(self, **kwargs):
        print("Running Read File tool")
        file_path = kwargs.get("file_path")
        print(f"File Path: {file_path}")
        try:
            with open(file_path, "r", encoding="utf8") as f:
                content = f.read()
            return {
                "status": "success",
                "message": "File read successfully",
                "error": None,
                "output": content
            }
        except Exception as e:
            return {
                "status": "error",
                "message": "Failed to read file",
                "error": str(e),
                "output": None
            }
